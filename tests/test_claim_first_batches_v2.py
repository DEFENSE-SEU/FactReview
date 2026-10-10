"""Two public controls; semantic choices are explicit mocks, no model quality claim."""

import copy
import json

import pytest

from screening import claim_coverage as m
from tests.test_claim_coverage_v2 import CFG
from tests.test_claim_coverage_v3 import QUALIFIER, review
from tests.test_claim_recheck_batches_v2 import callback as later_callback
from tests.test_claim_recheck_batches_v2 import inputs, oracle, saved


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("External LLM forbidden"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("httpx.Client.send", lambda *a, **k: pytest.fail("HTTP forbidden"))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("Docker/process forbidden"))


def first_callback(calls, *, merge=None, mode=None, special=False):
    later = later_callback([], merge=merge)

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        schema = json.loads(kw["prompt"].split("OUTPUT_SCHEMA:\n", 1)[1].split("\nDATA_JSON:", 1)[0])
        calls.append((kw["module"], copy.deepcopy(p), schema))
        if kw["module"].endswith("coverage_discovery"):
            return {
                "schema_version": "claim-coverage-discovery-v1",
                "context_id": p["context_id"], "window_id": p["window_id"],
                "parent_digest": p["parent_digest"],
                "reviewed_block_ids": [b["id"] for b in p["blocks"] if mode != "unread" or b["id"] != "b1"],
                "new_findings": [], "explanation": "Explicit full-window omission judgment.",
            }
        if kw["module"].endswith("coverage_first_batch"):
            rows = oracle(p, merge=merge if any(r["claim_id"] == merge for r in p["required_claim_checks"]) else None)["original_claim_reviews"]
            if special and any(r["claim_id"] == "claim_004" for r in rows):
                expanded = m.unpack_scope_payload(p)
                selected = next(c for c in expanded["current_claims"] if c["claim_id"] == "claim_004")
                expanded.update(current_claims=[selected], required_claim_checks=[next(r for r in p["required_claim_checks"] if r["claim_id"] == "claim_004")],
                                scope_context=[r for r in expanded["scope_context"] if r["claim_id"] == "claim_004"])
                packed = m.pack_scope_payload(expanded)
                findings = [{"kind": "missing_needs", "condition_ids": ["c1"], "source_block_ids": ["b1"],
                             "needs": [need], "reason": "Mock absent routing " + need} for need in ("Theory", "Experiments")]
                replacement = review(packed, findings=[*findings, QUALIFIER])["claim_reviews"][0]
                rows = [replacement if r["claim_id"] == "claim_004" else r for r in rows]
            raw = {"schema_version": "claim-coverage-first-batch-v1", "context_id": p["context_id"],
                   "window_id": p["window_id"], "parent_digest": p["parent_digest"],
                   "batch_id": p["batch_id"], "claim_reviews": rows}
            if p["batch_id"] == "first_batch_002":
                if mode == "missing":
                    raw["claim_reviews"].pop()
                elif mode == "duplicate":
                    raw["claim_reviews"][1] = copy.deepcopy(rows[0])
                elif mode == "parent":
                    raw["parent_digest"] = "stale"
                elif mode == "context":
                    raw["context_id"] = "stale"
                elif mode == "source":
                    raw["claim_reviews"][0]["source_block_ids"] = ["foreign"]
                elif mode == "unresolved":
                    row = raw["claim_reviews"][0]
                    row.update(state="unresolved", other_findings=[], scope_atoms=[])
                    for group in row["scope_groups"]:
                        for dim in group["dimensions"].values():
                            dim.update(state="unresolved", atom_ids=[])
                elif mode == "service":
                    raise RuntimeError("Explicit mock service failure")
            return raw
        return later(**kw)
    return call


def test_complete_first_partition_single_adapter_links_and_zero_claim_discovery(tmp_path):
    paper, claims = inputs(5)
    before = [c.model_dump(mode="json") for c in claims]
    calls = []
    result = m.review_claim_coverage(paper, claims, first_review_batch_size=3,
        original_review_batch_size=2, call=first_callback(calls, merge="claim_004"), output_dir=tmp_path / "complete")
    discovery = [p for mod, p, _ in calls if mod.endswith("coverage_discovery")]
    batches = [p for mod, p, _ in calls if mod.endswith("coverage_first_batch")]
    assert len(discovery) == 1 and [len(p["required_claim_checks"]) for p in batches] == [3, 2]
    assert [r["claim_id"] for p in batches for r in p["required_claim_checks"]] == [c.id for c in claims]
    for p in batches:
        assert p["blocks"] == discovery[0]["blocks"] and p["current_claims"] == discovery[0]["current_claims"]
        assert p["review_only_block_ids"] == discovery[0]["review_only_block_ids"]
        assert p["previous_unresolved_observations"] == discovery[0]["previous_unresolved_observations"]
        assert m.unpack_scope_payload(p)["scope_context"] == m.unpack_scope_payload(discovery[0])["scope_context"]
        assert all(len(c["conditions"]) == 2 for c in p["current_claims"])
        assert m._digest({k: v for k, v in p.items() if k != "context_id"}) == p["context_id"]
    assert result.coverage["budget"]["review_calls"] == 3
    assert result.coverage["claim_checks_completed"] == 5 and result.coverage["status"] == "complete"
    assert len(result.claims) == 6 and len(result.coverage["revisions"]) == 1
    audit = saved(tmp_path / "complete")
    w = audit["coverage"]["windows"][0]
    adapter = w["first_batches"]["aggregate"]
    assert adapter["version"] == "program-first-review-adapter-v1" and adapter["context_id"] not in [p["context_id"] for p in batches]
    assert w["first_batches"]["remaining_required_claim_ids"] == []
    assert adapter["path_mappings"][3]["aggregate_path"] == "/claim_reviews/3"
    generated = set(w["first_batches"]["lowering"]["observation_origins"])
    follow = next(p for mod, p, _ in calls if mod.endswith("coverage_followup"))
    assert {o["id"] for o in follow["observations"]} == generated
    assert audit["original_claims"] == before and [c.model_dump(mode="json") for c in claims] == before
    assert all("response_digest" in a for a in audit["attempts"] if a["module"].endswith(("coverage_discovery", "coverage_first_batch")))
    for mod, p, schema in calls:
        if mod.endswith("coverage_first_batch"):
            assert schema["properties"]["claim_reviews"]["minItems"] == schema["properties"]["claim_reviews"]["maxItems"] == len(p["required_claim_checks"])
            assert schema["properties"]["parent_digest"]["const"] == p["parent_digest"]
    special_calls = []
    special = m.review_claim_coverage(paper, claims, first_review_batch_size=3, max_followup_calls=0,
        max_validation_calls=0, call=first_callback(special_calls, special=True), output_dir=tmp_path / "atom")
    lowered = special.coverage["windows"][0]["first_batches"]["lowering"]
    mapping = next(r for r in lowered["wire_mappings"] if r["claim_id"] == "claim_004")
    assert any(r["raw_path"] == "/scope_atoms/0" and r["generated_path"] == "/findings/2" for r in mapping["fields"])
    assert len(special.coverage["windows"][0]["observations"]) == 3
    assert not special.coverage["revisions"] and special.coverage["status"] == "partial"
    zero_calls = []
    zero = m.review_claim_coverage(paper, [], first_review_batch_size=3,
        original_review_batch_size=3, call=first_callback(zero_calls), output_dir=tmp_path / "zero")
    assert [mod for mod, _, _ in zero_calls] == ["screening.claims.coverage_discovery"]
    assert zero.coverage["budget"]["review_calls"] == 1
    assert zero.coverage["windows"][0]["first_batches"]["planned_claim_batches"] == 0
    assert zero.coverage["claim_checks_required"] == zero.coverage["claim_checks_completed"] == 0


def test_first_batch_failures_budget_unresolved_and_later_window_preserve_partial(tmp_path):
    for mode in ("missing", "duplicate", "parent", "context", "source", "unread", "unresolved", "service", "budget", "zero_budget"):
        paper, claims = inputs(5)
        before = [c.model_dump(mode="json") for c in claims]
        calls = []
        cap = 0 if mode == "zero_budget" else 1 if mode == "budget" else 12
        result = m.review_claim_coverage(paper, claims, first_review_batch_size=3, max_review_calls=cap,
            call=first_callback(calls, mode=mode), output_dir=tmp_path / mode)
        assert result.coverage["status"] == "partial" and not result.coverage["revisions"]
        assert [c.model_dump(mode="json") for c in result.claims] == before
        assert not any(mod.endswith(("coverage_followup", "coverage_validation", "coverage_original_batch", "coverage_batch_decisions")) for mod, _, _ in calls)
        w = result.coverage["windows"][0]
        assert w["first_batches"]["planned_claim_batches"] == 2 and w["first_batches"]["planned_review_calls"] == 3
        assert w["first_batches"]["remaining_required_claim_ids"]
        assert result.coverage["budget"]["review_calls"] <= cap
        assert w["original_recheck"]["status"] == "not_run_incomplete_first"
        assert saved(tmp_path / mode)["original_claims"] == before
    # Split actual original blocks into two natural windows. Window2's true discovery
    # remains independent and can advance after the first window's failed child.
    paper, claims = inputs(5)
    calls = []
    result = m.review_claim_coverage(paper, claims, first_review_batch_size=3, window_chars=60,
        call=first_callback(calls, mode="missing"), output_dir=tmp_path / "later")
    discoveries = [p for mod, p, _ in calls if mod.endswith("coverage_discovery")]
    assert len(discoveries) == 2 and discoveries[0]["window_id"] != discoveries[1]["window_id"]
    assert result.coverage["windows"][0]["first_batches"]["remaining_required_claim_ids"]
    assert result.coverage["windows"][1]["first_batches"]["planned_claim_batches"] == 0
    assert result.coverage["budget"]["review_calls"] == 4 and not result.coverage["revisions"]


@pytest.mark.parametrize("module,schema_type,version", [
    ("coverage_discovery", m.CoverageDiscoveryV1, "claim-coverage-discovery-v1"),
    ("coverage_first_batch", m.CoverageFirstBatchV1, "claim-coverage-first-batch-v1"),
])
def test_first_request_requires_explicit_version_and_rejects_missing_response(tmp_path, module, schema_type, version):
    paper, claims = inputs(1)
    before = [c.model_dump(mode="json") for c in claims]
    calls = []
    valid = first_callback(calls)

    def missing_version(**kw):
        raw = valid(**kw)
        if kw["module"].endswith(module):
            schema = calls[-1][2]
            assert "schema_version" in schema["required"]
            assert schema["properties"]["schema_version"]["const"] == version
            assert set(schema["required"]) == set(schema_type.model_json_schema()["required"]) | {"schema_version"}
            assert f'schema_version="{version}"' in kw["system"]
            assert schema_type.model_fields["schema_version"].default == version
            assert raw.pop("schema_version") == version
        return raw

    result = m.review_claim_coverage(paper, claims, first_review_batch_size=3,
        call=missing_version, output_dir=tmp_path)
    assert result.coverage["status"] == "partial" and not result.coverage["revisions"]
    assert result.coverage["claim_checks_required"] == 1 and result.coverage["claim_checks_completed"] == 0
    assert [c.model_dump(mode="json") for c in result.claims] == before
    assert [c.model_dump(mode="json") for c in claims] == before
    assert not any(mod.endswith(("coverage_followup", "coverage_validation", "coverage_original_batch", "coverage_batch_decisions")) for mod, _, _ in calls)
    audit = saved(tmp_path)
    failed = [a for a in audit["attempts"] if a["module"].endswith(module)]
    assert len(failed) == 1 and failed[0]["status"] == "failed"
    assert "schema_version" not in failed[0]["response"]
    assert failed[0]["error"] == "ValueError: First response must declare the exact requested protocol"
    assert audit["original_claims"] == before
