"""Run real experimental verification on selected claims from a saved v2 run.

Parser/claim extraction artifacts are reused explicitly. This makes no retrieval,
Docker or training calls and cannot measure whole-paper review accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assessment import assess_claim
from common import run_stats
from common.env import load_env_file
from llm.client import llm_json
from llm.diagnostics import redact_provider_details
from schemas.claim import Claim, ClaimStatus, EvidenceNeed
from schemas.materials import SharedMaterials
from verification.contracts import RejectedPlan
from verification.dispatch import _validate_result
from verification.experiments import verify_experiments


def _save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_probe(source, claim_ids, output_root, *, expectations=None, call=None):
    """Save actual observations; optional expectations are external test oracles."""
    source, output_root = Path(source).resolve(), Path(output_root).resolve()
    paths = {
        "materials": source / "materials/materials.json",
        "screening": source / "screening/screening.json",
    }
    hashes = {key: _hash(path) for key, path in paths.items()}
    materials = SharedMaterials.model_validate_json(paths["materials"].read_text(encoding="utf-8"))
    saved = json.loads(paths["screening"].read_text(encoding="utf-8"))["claims"]
    claims = [Claim.model_validate(row) for row in saved]
    indexed = {claim.id: claim for claim in claims}
    if len(indexed) != len(claims) or len(set(claim_ids)) != len(claim_ids) or not claim_ids:
        raise ValueError("Select distinct claim IDs from an unambiguous saved claim list")
    if not set(claim_ids).issubset(indexed):
        raise ValueError("A selected claim ID is absent from the saved run")
    expectations = expectations or {}
    if not set(expectations).issubset(claim_ids):
        raise ValueError("Expectations must refer only to selected claims")
    for claim_id, expected in expectations.items():
        if set(expected) - {"supported_conditions", "unsupported_conditions"}:
            raise ValueError("Unknown expected result field")
        positive = expected.get("supported_conditions", [])
        negative = expected.get("unsupported_conditions", [])
        valid = {c.id for c in indexed[claim_id].conditions}
        if (
            not isinstance(positive, list)
            or not isinstance(negative, list)
            or any(not isinstance(value, str) for value in positive + negative)
            or not set(positive + negative).issubset(valid)
            or set(positive) & set(negative)
            or len(set(positive)) != len(positive)
            or len(set(negative)) != len(negative)
        ):
            raise ValueError("Expected coverage must name distinct valid, non-conflicting conditions")
    directory = output_root / ("experiment-probe-" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    implementation = {
        str(path.relative_to(ROOT)): _hash(path) for path in sorted((ROOT / "src").rglob("*.py"))
    }
    implementation["scripts/check_v2_experiments.py"] = _hash(Path(__file__))
    _save(directory / "implementation.json", implementation)
    _save(directory / "expectations.json", expectations)
    summary = {
        "boundary": "Selected saved claims and original parsed materials; live experimental model calls only. "
        "No fresh extraction, retrieval, Docker, training or whole-paper reassessment.",
        "source": str(source),
        "source_hashes": hashes,
        "cases": [],
        "status": "running",
        "expectations_supplied": bool(expectations),
        "model_boundary": "injected" if call is not None else "live",
    }
    _save(directory / "summary.json", summary)
    for position, claim_id in enumerate(claim_ids):
        out = directory / f"case-{position + 1:03d}"
        out.mkdir()
        claim = indexed[claim_id].model_copy(deep=True)
        claim.evidence, claim.notes, claim.questions, claim.status = [], [], [], ClaimStatus.UNVERIFIED
        _save(out / "claim.json", claim.model_dump(mode="json"))
        model_calls = []

        def recorded_call(**kwargs):
            cfg = kwargs["cfg"]
            record = {key: value for key, value in kwargs.items() if key != "cfg"}
            record.update(provider=cfg.provider, model=cfg.model, status="started")
            record_path = out / f"model-{uuid.uuid4().hex}.json"
            _save(record_path, redact_provider_details(record, cfg))
            try:
                response = (call or llm_json)(**kwargs)
                record["response"] = response
                if (
                    not isinstance(response, dict)
                    or response.get("status", "ok") not in {"ok", "success"}
                    or response.get("error")
                ):
                    raise RuntimeError(f"Model returned an unsuccessful response: {response}")
                record["status"] = "returned"
                return response
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(redact_provider_details(record["error"], cfg)) from None
            finally:
                record = redact_provider_details(record, cfg)
                model_calls.append(
                    {key: record[key] for key in ("module", "provider", "model", "error") if key in record}
                )
                _save(record_path, record)

        row = {"claim": claim_id, "directory": str(out), "status": "running"}
        summary["cases"].append(row)
        _save(directory / "summary.json", summary)
        try:
            with run_stats.run_scope(out / "run_stats.json"):
                try:
                    result = verify_experiments(claim, materials, call=recorded_call)
                except RejectedPlan as exc:
                    result = exc.observations
                    if result.plans:
                        raise ValueError("Rejected-plan recovery cannot contain execution plans") from exc
                    result.issues.append(f"Execution plan rejected: {exc}")
                _validate_result(claim, EvidenceNeed.EXPERIMENTS, result)
            _save(out / "result.json", result.model_dump(mode="json"))
            claim.evidence = result.evidence
            claim.questions = result.questions
            claim.notes = result.issues
            assessed = assess_claim(claim)
            _save(out / "assessed.json", assessed.model_dump(mode="json"))
            covered = sorted(
                {
                    c
                    for e in result.evidence
                    if e.sufficient
                    and e.direction == "support"
                    and e.affects_claim
                    and (e.source != "execution" or e.aligned is True)
                    for c in e.covered
                }
            )
            row.update(
                status="completed",
                assessed_status=assessed.status.value,
                supported_conditions=covered,
                evidence=len(result.evidence),
                issues=result.issues,
            )
            if claim_id in expectations:
                expected = expectations[claim_id]
                row["expectations_passed"] = set(expected.get("supported_conditions", [])).issubset(
                    covered
                ) and not set(expected.get("unsupported_conditions", [])) & set(covered)
        except Exception as exc:
            # Configured credentials are already removed by the production LLM boundary.
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["model_calls"] = model_calls
        row["failed_model_calls"] = sum("error" in record for record in model_calls)
        _save(directory / "summary.json", summary)
        print(
            json.dumps({key: value for key, value in row.items() if key not in {"issues", "error"}}),
            flush=True,
        )
    summary["source_unchanged"] = hashes == {key: _hash(path) for key, path in paths.items()}
    summary["implementation_unchanged"] = all(
        _hash(ROOT / path) == digest for path, digest in implementation.items()
    )
    summary["status"] = (
        "completed" if all(row["status"] == "completed" for row in summary["cases"]) else "failed"
    )
    summary["expectations_passed"] = (
        all(row.get("expectations_passed", False) for row in summary["cases"] if row["claim"] in expectations)
        if expectations
        else None
    )
    summary["ok"] = (
        summary["status"] == "completed"
        and summary["source_unchanged"]
        and summary["implementation_unchanged"]
        and summary["expectations_passed"] is not False
        and not any(row["failed_model_calls"] for row in summary["cases"])
    )
    _save(directory / "summary.json", summary)
    return directory, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Saved v2 run with materials/ and screening/")
    parser.add_argument("--claims", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/v2_experiments"))
    parser.add_argument(
        "--expectations", type=Path, help="JSON claim ID → supported/unsupported_conditions lists"
    )
    args = parser.parse_args(argv)
    load_env_file(ROOT / ".env")
    expected = json.loads(args.expectations.read_text(encoding="utf-8")) if args.expectations else None
    directory, summary = run_probe(args.source, args.claims, args.output_root, expectations=expected)
    print(str(directory), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
