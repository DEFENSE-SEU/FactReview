"""Run declared citation probes with explicit live or immutable-cache boundaries.

This is a selected Literature-branch check, not a whole-paper accuracy evaluation.
Synthetic citing documents stay separate from original screened claims.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import sys
import uuid
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assessment import assess_claim
from common import run_stats
from common.env import load_env_file
from llm.client import llm_json
from llm.diagnostics import redact_provider_details
from schemas.claim import Claim, EvidenceNeed
from schemas.materials import SharedMaterials
from util.cutoff_date import parse_submission_deadline
from verification.dispatch import _validate_result
from verification.literature import (
    _claim_source_excerpts,
    _default_adapter,
    _same_identity,
    literature_queries,
    verify_literature,
)
from verification.theory import _fully_supported


def _save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _path(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _implementation():
    paths = [*sorted((ROOT / "src").rglob("*.py")), Path(__file__)]
    return {str(path.relative_to(ROOT)): _hash(path) for path in paths}


def _safe(value, cfg=None):
    secrets = [v for k, v in os.environ.items() if v and re.search(r"API_KEY|TOKEN|SECRET|PASSWORD", k)]
    cleaned = redact_provider_details(value, cfg, secrets=secrets)
    for key, base in os.environ.items():
        if key.endswith("BASE_URL") and base.startswith(("https://", "http://")):
            cleaned = redact_provider_details(cleaned, base_url=base)
    return cleaned


def _expectation(case, claim):
    expected = case["expectation"]
    if set(expected) - {
        "supported_conditions",
        "unsupported_conditions",
        "status_in",
        "citations",
        "unresolved",
    }:
        raise ValueError("Unknown expectation field")
    ids = {c.id for c in claim.conditions}
    positive, negative = expected["supported_conditions"], expected["unsupported_conditions"]
    if (
        not isinstance(positive, list)
        or not isinstance(negative, list)
        or any(not isinstance(c, str) or c not in ids for c in positive + negative)
        or len(set(positive + negative)) != len(positive + negative)
        or set(positive + negative) != ids
    ):
        raise ValueError("Expectations must partition every original condition exactly once")
    if not expected.get("status_in") or not set(expected["status_in"]) <= {
        "supported",
        "questioned",
        "unverified",
        "flawed",
    }:
        raise ValueError("Expected assessment statuses are required")
    citations, unresolved = expected.get("citations", []), expected.get("unresolved", [])
    if bool(citations) == bool(unresolved):
        raise ValueError("Declare either verifiable citations or exact unresolved bibliography entries")
    for row in citations:
        if (
            set(row) != {"id", "covered"}
            or not row["id"]
            or not row["covered"]
            or not set(row["covered"]) <= ids
        ):
            raise ValueError("A required citation needs an identifier and original condition coverage")
    if any(not isinstance(text, str) or not text.strip() for text in unresolved):
        raise ValueError("Unresolved oracles require exact bibliography entries")
    if unresolved and (case["modes"] != ["original-cache"] or positive):
        raise ValueError("Stable unresolved oracles require original-cache mode and no full support")
    return expected


def _prepare(plan_path, names, mode, output_root):
    plan = _load(plan_path)
    if plan.get("version") != 1 or not isinstance(plan.get("cases"), list):
        raise ValueError("Expected a version 1 citation probe plan")
    cases = {c["name"]: c for c in plan["cases"]}
    if (
        len(cases) != len(plan["cases"])
        or not names
        or len(set(names)) != len(names)
        or not set(names) <= cases.keys()
    ):
        raise ValueError("Select distinct declared cases")
    protected = {_path(p): digest for p, digest in plan["protected_files"].items()}
    protected[plan_path] = _hash(plan_path)
    if any(not p.is_file() or _hash(p) != digest for p, digest in protected.items()):
        raise ValueError("A protected plan input changed or is missing")
    selected = []
    for name in names:
        case = deepcopy(cases[name])
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or mode not in case["modes"]:
            raise ValueError("Invalid case name or unauthorized boundary mode")
        parse_submission_deadline(case["submission_deadline"])
        if case["kind"] == "original":
            source = _path(case["source_run"])
            paths = [source / "materials/materials.json", source / "screening/screening.json"]
            if any(p not in protected for p in paths) or output_root.is_relative_to(source):
                raise ValueError("Original inputs need hashes; outputs must stay outside the original run")
            raw = _load(paths[1])["claims"]
            matches = [c for c in raw if c["id"] == case["raw_claim"]["id"]]
            if len(matches) != 1 or matches[0] != case["raw_claim"]:
                raise ValueError("Original claim differs from the declared immutable record")
            material = SharedMaterials.model_validate(_load(paths[0]))
            claim = Claim.model_validate(matches[0])
            _claim_source_excerpts(claim, material)
        elif case["kind"] == "synthetic":
            if case.get("synthetic") is not True:
                raise ValueError("Synthetic inputs require an explicit label")
            claim, material = Claim.model_validate(case["claim"]), None
        else:
            raise ValueError("Unknown input kind")
        if EvidenceNeed.LITERATURE not in claim.needs:
            raise ValueError("Selected claims must request Literature")
        _expectation(case, claim)
        cache = None
        if mode == "original-cache":
            cache_path = _path(case["cache_path"])
            if case["kind"] != "original" or cache_path not in protected:
                raise ValueError("Cache replay needs an original protected audit")
            cache = _load(cache_path)
            if cache["submission_deadline"] != case["submission_deadline"]:
                raise ValueError("Cached cutoff differs from declared cutoff")
            if cache["claim_source_excerpt"] != case["raw_claim"]["source_quote"]:
                raise ValueError("Cached source differs from original claim")
        selected.append((case, claim, material, cache))
    return plan, selected, protected


class Cache:
    """Exact saved requests only. A miss raises and never invokes a provider."""

    def __init__(self, audit):
        self.audit = audit

    def _get(self, field, key, value):
        rows = [r for r in self.audit[field] if r[key] == value]
        if len(rows) != 1:
            raise ValueError(f"Cache miss or ambiguous request: {field}/{value}")
        return deepcopy(rows[0]["response"])

    async def search(self, *, query, cutoff_date):
        if cutoff_date.to_string() != self.audit["submission_deadline"]:
            raise ValueError("Cache cutoff mismatch")
        return self._get("queries", "query", query)

    async def lookup_metadata(self, *, identifier):
        return self._get("metadata_lookups", "id", identifier)

    async def read_papers(self, *, items):
        if (
            len(items) != 1
            or items[0].get("question") != "Describe the mechanism, target setting, and evaluation protocol."
        ):
            raise ValueError("Cached reader request differs")
        return self._get("reads", "id", items[0]["id"])

    def compare(self, **kwargs):
        # Old audits retain comparisons, not the original transport envelope.
        # The reconstructed envelope is labelled in every cached model record.
        comparisons = self.audit["comparisons"]
        if not isinstance(comparisons, list):
            raise ValueError("Missing cached comparison list")
        return {"status": "ok", "comparisons": deepcopy(comparisons)}


def _response_ok(kind, value):
    if not isinstance(value, dict) or value.get("error"):
        return False
    if kind == "model":
        return (
            value.get("status") == "ok"
            and isinstance(value.get("comparisons"), list)
            and all(isinstance(c, dict) for c in value["comparisons"])
        )
    if value.get("success") is not True:
        return False
    if kind == "search":
        return (
            bool(value.get("provider"))
            and isinstance(value.get("papers"), list)
            and all(isinstance(p, dict) for p in value["papers"])
            and isinstance(value.get("question_results", []), list)
            and all(
                isinstance(q, dict) and q.get("success") is True and not q.get("error")
                for q in value.get("question_results", [])
            )
        )
    if kind == "metadata":
        return isinstance(value.get("paper"), dict)
    return (
        isinstance(value.get("items"), list)
        and bool(value["items"])
        and all(
            isinstance(i, dict) and i.get("success") is True and not i.get("error") for i in value["items"]
        )
    )


class Recorded:
    def __init__(self, adapter, call, directory, boundary):
        self.adapter, self.call, self.directory, self.boundary = adapter, call, directory, boundary
        self.records = []

    def clean(self, value, cfg=None):
        value = _safe(value, cfg)
        for name in ("search_cfg", "read_cfg"):
            config = getattr(self.adapter, name, None)
            if not is_dataclass(config):
                continue
            fields = asdict(config)
            secrets = [v for k, v in fields.items() if k.endswith("api_key") and v]
            value = redact_provider_details(value, secrets=secrets)
            for key, base in fields.items():
                if key.endswith("base_url") and base:
                    value = redact_provider_details(value, base_url=base, secrets=secrets)
        return value

    async def invoke(self, kind, function, request):
        cfg = request.get("cfg")
        record = {
            "kind": kind,
            "boundary": self.boundary[kind],
            "request": {k: v for k, v in request.items() if k != "cfg"},
            "status": "started",
        }
        if kind == "search":
            record["request"]["cutoff_date"] = request["cutoff_date"].to_string()
        if kind == "model":
            record.update(
                provider=getattr(cfg, "provider", None)
                if self.boundary[kind] != "cached"
                else "original-cache",
                model=getattr(cfg, "model", None)
                if self.boundary[kind] != "cached"
                else "unrecorded-original-model",
            )
            record["reconstructed_envelope"] = self.boundary[kind] == "cached"
        path = self.directory / f"request-{len(self.records) + 1:03d}.json"
        self.records.append(record)
        _save(path, self.clean(record, cfg))
        try:
            response = function(**request)
            if inspect.isawaitable(response):
                response = await response
            record["response_sha256"] = _object_hash(response)
            response = self.clean(response, cfg)
            record["response"] = response
            record["status"] = "returned" if _response_ok(kind, response) else "unsuccessful_response"
            return response
        except Exception as exc:
            record.update(status="failed", error=self.clean(f"{type(exc).__name__}: {exc}", cfg))
            raise RuntimeError(record["error"]) from None
        finally:
            clean = self.clean(record, cfg)
            record.clear()
            record.update(clean)
            _save(path, record)

    async def search(self, **kwargs):
        return await self.invoke("search", self.adapter.search, kwargs)

    async def lookup_metadata(self, **kwargs):
        return await self.invoke("metadata", self.adapter.lookup_metadata, kwargs)

    async def read_papers(self, **kwargs):
        return await self.invoke("read", self.adapter.read_papers, kwargs)

    async def compare(self, **kwargs):
        return await self.invoke("model", self.call, kwargs)


def _comparison_valid(comparison, wanted, condition_ids, passages):
    covered = comparison.get("covered")
    if not isinstance(covered, list) or any(not isinstance(c, str) for c in covered):
        return False
    try:
        _fully_supported(covered, comparison.get("fully_supported_conditions", []))
    except ValueError:
        return False
    return (
        comparison.get("relation") in {"supports", "partial", "unclear", "contradicts"}
        and wanted <= set(covered) <= condition_ids
        and len(set(covered)) == len(covered)
        and all(
            isinstance(comparison.get(k), str) and comparison[k].strip()
            for k in ("mechanism", "setting", "protocol")
        )
        and isinstance(comparison.get("quote"), str)
        and bool(comparison["quote"].strip())
        and any(comparison["quote"] in e["text"] for e in passages)
    )


def _evaluate(case, claim, materials, audit, result, assessed, recorder):
    expected = case["expectation"]
    supported = sorted(
        {
            c
            for e in result.evidence
            if e.sufficient and e.direction == "support" and e.affects_claim
            for c in e.covered
        }
    )
    checks = {
        "condition_coverage": set(expected["supported_conditions"]) <= set(supported)
        and not set(expected["unsupported_conditions"]) & set(supported),
        "assessment": assessed.status.value in expected["status_in"],
        "query_requests": [r["request"]["query"] for r in recorder.records if r["kind"] == "search"]
        == literature_queries(claim, materials)
        and len(audit["queries"]) == 3,
    }
    failures = [r for r in recorder.records if r["status"] != "returned"]
    # An immutable unresolved replay can contain a documented, unrelated old
    # reader failure. Its oracle requires the exact local identity exclusion;
    # it does not make a claim about that old reader's health.
    checks["boundary_health"] = not any(
        not (
            expected.get("unresolved")
            and r["boundary"] == "cached"
            and r["kind"] == "read"
            and r["status"] == "unsuccessful_response"
        )
        for r in failures
    )
    for text in expected.get("unresolved", []):
        key = "unresolved:" + _object_hash(text)[:12]
        checks[key] = any(
            r.get("reason") == "unresolved_identity_metadata"
            and r.get("paper", {}).get("bibliography_text") == text
            and not r["paper"].get("id")
            for r in audit["excluded"]
        ) and any(text in issue for issue in result.issues)
        checks["no_empty_id_reads"] = all(r["id"] for r in audit["reads"])
    model_records = [r for r in recorder.records if r["kind"] == "model" and r["status"] == "returned"]
    payload = {}
    if expected.get("citations"):
        if len(model_records) == 1:
            payload = json.loads(model_records[0]["request"]["prompt"].split("\nDATA_JSON:\n", 1)[1])
        checks["model_source_input"] = payload.get("claim") == claim.model_dump(mode="json") and payload.get(
            "source_excerpts"
        ) == _claim_source_excerpts(claim, materials)
    for required in expected.get("citations", []):
        identifier, wanted = required["id"], set(required["covered"])
        binding = [
            r
            for r in audit["citation_bindings"]
            if _same_identity(r["paper_id"], identifier)
            and r["cited"]
            and wanted <= set(r["citation_condition_ids"])
        ]
        reads = [
            r
            for r in audit["reads"]
            if _same_identity(r["id"], identifier)
            and r.get("identity_verified") is True
            and not r.get("identity_rejected")
        ]
        # The reader may return multiple same-ID items. Production chooses one;
        # only the exact passages it actually sent to the model may satisfy the
        # probe oracle. Never union unused passages from the raw response.
        sources = [s for s in payload.get("sources", []) if _same_identity(s["paper_id"], identifier)]
        selected = sources[0] if len(sources) == 1 else {}
        source_verified = (
            bool(reads)
            and selected.get("reader_identity_verified") is True
            and selected.get("full_text") is True
            and selected.get("cited") is True
            and any(
                selected.get("citation_sources") == b["citation_sources"]
                and selected.get("citation_condition_ids") == b["citation_condition_ids"]
                for b in binding
            )
        )
        passages = [
            e
            for e in selected.get("passages", [])
            if source_verified
            and e.get("source") == "full_text"
            and isinstance(e.get("text"), str)
            and type(e.get("page")) is int
            and e["page"] > 0
        ]
        # A partial comparison may deliberately produce no Evidence. Require a
        # healthy, grounded comparison in the recorded model response instead.
        comparisons = [
            c
            for c in audit["comparisons"]
            if c.get("purpose") == "citation_support"
            and _same_identity(str(c.get("paper_id", "")), identifier)
        ]
        valid = [
            c for c in comparisons if _comparison_valid(c, wanted, {x.id for x in claim.conditions}, passages)
        ]
        checks["citation:" + identifier] = bool(
            binding and passages and valid and len(valid) == len(comparisons) and len(model_records) == 1
        )
    return {
        "checks": checks,
        "expectations_passed": all(checks.values()),
        "supported_conditions": supported,
        "assessed_status": assessed.status.value,
        "failed_calls": len(failures),
        "failed_live_calls": sum(r["boundary"] == "live" for r in failures),
    }


async def run_probe(plan_path, names, output_root, *, mode="live", adapter=None, call=None):
    plan_path, output_root = _path(plan_path), _path(output_root)
    if mode not in {"live", "original-cache"} or (
        mode == "original-cache" and (adapter is not None or call is not None)
    ):
        raise ValueError("Invalid mode or injected cache bypass")
    plan, selected, protected = _prepare(plan_path, names, mode, output_root)
    directory = output_root / ("literature-probe-" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    implementation = _implementation()
    _save(directory / "implementation.json", implementation)
    _save(directory / "plan.json", plan)
    summary = {
        "status": "running",
        "mode": mode,
        "cases": [],
        "training_runs": 0,
        "docker_calls": 0,
        "source_hashes": {str(p): h for p, h in protected.items()},
        "boundary": "Selected Literature only; fixed materials and claims; synthetic inputs explicitly separate. PDF bytes/hash unavailable through current reader interface; no extra download performed.",
    }
    _save(directory / "summary.json", summary)
    for index, (case, claim, materials, cache) in enumerate(selected, 1):
        out = directory / f"case-{index:03d}"
        out.mkdir()
        _save(out / "declared-input.json", case)
        if materials is None:
            design = deepcopy(case["materials"])
            markdown = out / "synthetic-paper.md"
            markdown.write_text(design["markdown"], encoding="utf-8")
            materials = SharedMaterials.model_validate(
                {
                    **design,
                    "paper_key": case["name"],
                    "source_pdf": "synthetic:no-pdf",
                    "markdown_path": str(markdown),
                    "content_list_path": "synthetic:no-parser-output",
                    "provider": "synthetic citing-paper fixture",
                }
            )
            _claim_source_excerpts(claim, materials)
        _save(out / "claim.json", claim.model_dump(mode="json"))
        _save(out / "materials.json", materials.model_dump(mode="json"))
        row = {
            "name": case["name"],
            "claim": claim.id,
            "synthetic": case["kind"] == "synthetic",
            "directory": str(out),
            "status": "running",
        }
        summary["cases"].append(row)
        _save(directory / "summary.json", summary)
        target = Cache(cache) if cache is not None else adapter if adapter is not None else _default_adapter()
        boundaries = {
            k: "cached" if cache is not None else "injected" if adapter is not None else "live"
            for k in ("search", "metadata", "read")
        }
        boundaries["model"] = "cached" if cache is not None else "injected" if call is not None else "live"
        selected_call = target.compare if cache is not None else call if call is not None else llm_json
        recorder = Recorded(target, selected_call, out, boundaries)
        try:
            with run_stats.run_scope(out / "run_stats.json"):
                result = await verify_literature(
                    claim,
                    materials,
                    submission_deadline=case["submission_deadline"],
                    searcher=recorder,
                    reader=recorder,
                    call=recorder.compare,
                    output_dir=out / "audit",
                )
                _validate_result(claim, EvidenceNeed.LITERATURE, result)
            _save(out / "result.json", _safe(result.model_dump(mode="json")))
            assessed = assess_claim(
                claim.model_copy(
                    update={
                        "evidence": result.evidence,
                        "questions": result.questions,
                        "notes": result.issues,
                    },
                    deep=True,
                )
            )
            _save(out / "assessed.json", _safe(assessed.model_dump(mode="json")))
            audits = list((out / "audit").glob("*-search-audit.json"))
            if len(audits) != 1:
                raise ValueError("Expected exactly one saved Literature audit")
            audit = _load(audits[0])
            row.update(
                _evaluate(case, claim, materials, audit, result, assessed, recorder), status="completed"
            )
        except Exception as exc:
            row.update(
                status="failed", error=_safe(f"{type(exc).__name__}: {exc}"), expectations_passed=False
            )
        row["boundaries"] = boundaries
        row["failed_calls"] = sum(r["status"] != "returned" for r in recorder.records)
        row["failed_live_calls"] = sum(
            r["status"] != "returned" and r["boundary"] == "live" for r in recorder.records
        )
        row["cached_failure_note"] = (
            "Historical unrelated reader failures remain visible; the unresolved oracle requires its own exact bibliography exclusion and explanatory issue."
            if case["expectation"].get("unresolved")
            else None
        )
        row["calls"] = [
            {
                k: r[k]
                for k in ("kind", "boundary", "status", "response_sha256", "provider", "model", "error")
                if k in r
            }
            for r in recorder.records
        ]
        row["usage"] = run_stats.with_totals(run_stats.read(out / "run_stats.json"))["total"]
        row["usage_note"] = (
            "Production run_stats as recorded; cached/injected calls do not imply live token usage. Zero recorded tokens do not establish provider usage availability."
        )
        _save(directory / "summary.json", _safe(summary))
        print(
            json.dumps(
                {
                    "case": row["name"],
                    "status": row["status"],
                    "expectations_passed": row.get("expectations_passed", False),
                }
            ),
            flush=True,
        )
    summary["source_unchanged"] = all(p.is_file() and _hash(p) == h for p, h in protected.items())
    summary["implementation_unchanged"] = implementation == _implementation()
    summary["boundary_call_counts"] = {
        boundary: sum(call["boundary"] == boundary for row in summary["cases"] for call in row["calls"])
        for boundary in ("live", "cached", "injected")
    }
    summary["call_count_note"] = (
        "Adapter/model boundary invocations; internal HTTP attempts are not exposed by this interface. LLM usage/retries remain in run_stats."
    )
    summary["status"] = "completed" if all(c["status"] == "completed" for c in summary["cases"]) else "failed"
    summary["ok"] = (
        summary["source_unchanged"]
        and summary["implementation_unchanged"]
        and all(c.get("expectations_passed") is True for c in summary["cases"])
    )
    _save(directory / "summary.json", _safe(summary))
    return directory, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--mode", choices=("live", "original-cache"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/v2_literature"))
    args = parser.parse_args(argv)
    if args.mode == "live":
        load_env_file(ROOT / ".env")
    directory, summary = asyncio.run(run_probe(args.plan, args.case, args.output_root, mode=args.mode))
    print(str(directory), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
