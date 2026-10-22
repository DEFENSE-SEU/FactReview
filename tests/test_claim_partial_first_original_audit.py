"""One offline control for independent audit progress and unchanged adoption gates."""

import copy
import json

import pytest

from screening import claim_coverage as m
from screening import stage
from tests.test_claim_coverage import CFG
from tests.test_claim_first_batches import first_callback
from tests.test_claim_recheck_batches import inputs, oracle, saved


def unresolved(row):
    row.update(state="unresolved", other_findings=[], scope_atoms=[])
    for group in row["scope_groups"]:
        for dimension in group["dimensions"].values():
            dimension.update(state="unresolved", atom_ids=[])


def test_partial_first_independent_original_audit_contract(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External LLM, retrieval, HTTP and Docker are forbidden")

    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)

    def run(name, *, first_mode="unresolved", original_mode="mixed", enabled=True,
            cap=12, window_chars=24000, original_size=3, discovery_finding=False):
        paper, claims = inputs(5)
        before_paper = paper.model_dump(mode="json")
        before_claims = [c.model_dump(mode="json") for c in claims]
        calls, raw_responses = [], []
        base = first_callback(calls, mode=first_mode)

        def call(**kw):
            module = kw["module"]
            payload = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
            if module.endswith("coverage_original_batch"):
                calls.append((module, copy.deepcopy(payload), None))
                raw = oracle(payload, merge="claim_001" if original_mode == "mixed"
                             and any(r["claim_id"] == "claim_001"
                                     for r in payload["required_original_claim_reviews"]) else None)
                raw = {k: raw[k] for k in
                       ("context_id", "window_id", "original_claim_reviews", "observation_links")}
                raw.update(schema_version="claim-original-recheck-batch-v1",
                           parent_digest=payload["parent_digest"], batch_id=payload["batch_id"])
                for row in raw["original_claim_reviews"]:
                    if row["claim_id"] == "claim_004":
                        if original_mode == "mechanical":
                            row["source_block_ids"] = ["b1", "b1"]
                        else:
                            unresolved(row)
            elif module.endswith("coverage_validation"):
                calls.append((module, copy.deepcopy(payload), None))
                raw = oracle(payload)
            else:
                raw = base(**kw)
                if module.endswith("coverage_discovery") and discovery_finding:
                    raw["new_findings"] = [{"kind": "uncertain", "source_block_ids": ["b2"],
                                            "reason": "Explicit offline discovery uncertainty."}]
                if module.endswith("coverage_first_batch") and payload["batch_id"] == "first_batch_002":
                    if first_mode == "ack":
                        raw["claim_reviews"][0]["source_review_groups"] = []
                    elif first_mode == "mechanical":
                        raw["claim_reviews"][0]["state"] = "invalid"
            raw_responses.append((module, copy.deepcopy(raw)))
            return raw

        result = m.review_claim_coverage(
            paper, claims, call=call, output_dir=tmp_path / name,
            first_review_batch_size=3, original_review_batch_size=original_size,
            audit_incomplete_first=enabled, max_validation_calls=cap,
            window_chars=window_chars,
        )
        assert paper.model_dump(mode="json") == before_paper
        assert [c.model_dump(mode="json") for c in claims] == before_claims
        assert [c.model_dump(mode="json") for c in result.claims] == before_claims
        assert not result.coverage["revisions"]
        audit = saved(tmp_path / name)
        assert audit["original_claims"] == before_claims and not audit["changes"]
        returned = [(a["module"], a["response"]) for a in audit["attempts"] if "response" in a]
        assert returned == raw_responses
        return result, calls, audit

    result, calls, audit = run("semantic")
    window = result.coverage["windows"][0]
    partial = window["partial_first_original_audit"]
    assert result.coverage["status"] == "partial" and window["status"] == "partially_reviewed"
    assert result.coverage["claim_checks_completed"] == 4
    assert result.coverage["original_claim_reviews_completed"] == 0
    assert window["original_recheck"]["status"] == "not_run_incomplete_first"
    assert window["followup_status"] == "not_run_incomplete_first"
    assert window["first_batches"]["remaining_required_claim_ids"] == ["claim_004"]
    assert partial["audit_only"] is True and partial["adoption_authorized"] is False
    assert partial["required"] == 5 and partial["completed"] == 4
    assert {f["kind"] for f in partial["findings"]} >= {"merged_conclusions", "uncertain"}
    assert "remaining_duty_retry" not in partial["batch_review"]
    summary = m.coverage_summary(result.coverage)["partial_first_original_audit"]
    assert summary == {"audit_only": True, "adoption_authorized": False,
                       "windows": 1, "required": 5, "completed": 4}
    originals = [p for mod, p, _ in calls if mod.endswith("coverage_original_batch")]
    assert [r["claim_id"] for p in originals for r in p["required_original_claim_reviews"]] == [
        f"claim_{i:03d}" for i in range(1, 6)]
    first_parent = window["first_batches"]["parent_manifest"]
    for payload in originals:
        expanded = m.unpack_scope_payload(payload)
        assert expanded["current_claims"] == first_parent["current_claims"]
        assert expanded["scope_context"] == first_parent["scope_context"]
        assert expanded["blocks"] == first_parent["blocks"]
        assert expanded["observations"] == []
        assert payload["parent_digest"] == partial["batch_review"]["parent_digest"]
        assert "candidates" not in payload or payload["candidates"] == []
    assert partial["batch_review"]["parent_manifest"]["candidates"] == []
    for link in partial["batch_review"]["batches"]:
        attempt = audit["attempts"][int(link["raw_audit_path"].split("/")[2])]
        assert link["response_digest"] == m._digest(attempt["response"])
        assert link["input_digest"] == m._digest(attempt["input"])
        assert attempt["input"]["context_id"] == m._digest(
            {k: v for k, v in attempt["input"].items() if k != "context_id"})
    assert not any(mod.endswith(("coverage_followup", "coverage_first_duty_retry",
                                "coverage_batch_decisions")) for mod, _, _ in calls)

    # Discovery mappings coexist with the complete ordered original mappings.
    # Their distinct schema has no claim_id; the valid finding remains in the
    # audit while first uncertainty and all scientific adoption gates stay closed.
    result, calls, audit = run("discovery_finding", discovery_finding=True)
    window = result.coverage["windows"][0]
    assert result.coverage["status"] == "partial" and not result.coverage.get("adoption_rolled_back")
    assert window["status"] == "partially_reviewed"
    first = window["first_batches"]
    assert first["requests_complete"] and not first["errors"]
    assert not first["lowering"]["errors"] and not window["claim_check_errors"]
    assert all(context["complete"] for context in window["scope_context"])
    claim_mappings = [row for row in first["lowering"]["mappings"] if "claim_id" in row]
    assert [row["claim_id"] for row in claim_mappings] == [r["claim_id"] for r in window["required_claim_checks"]]
    discovery_mapping = next(row for row in first["lowering"]["mappings"]
                             if row["raw_path"] == "/new_findings/0")
    assert "claim_id" not in discovery_mapping
    discovery = next(a for a in audit["attempts"] if a["module"].endswith("coverage_discovery"))
    finding = {"kind": "uncertain", "source_block_ids": ["b2"],
               "reason": "Explicit offline discovery uncertainty."}
    assert discovery["response"]["new_findings"] == [finding]
    assert first["aggregate"]["new_findings"] == [finding]
    assert discovery_mapping["raw_digest"] == m._digest(finding)
    observed = next(o for o in window["observations"] if o["id"] in discovery_mapping["observation_ids"])
    assert observed["state"] == "unresolved" and observed["target_claim_id"] is None
    assert window["partial_first_original_audit"]["required"] == 5
    assert window["partial_first_original_audit"]["completed"] == 4
    original_parent = window["partial_first_original_audit"]["batch_review"]["parent_manifest"]
    assert original_parent["observations"] == original_parent["candidates"] == []
    assert original_parent["required_original_claim_reviews"] == window["required_claim_checks"]
    assert window["original_recheck"]["status"] == "not_run_incomplete_first"
    assert window["original_recheck"]["completed"] == result.coverage["original_claim_reviews_completed"] == 0
    assert window["followup_status"] == "not_run_incomplete_first"
    assert first["remaining_required_claim_ids"] == ["claim_004"]
    assert result.coverage["claim_checks_completed"] == 4
    assert result.coverage["unresolved_observations"] == 3
    assert not any(mod.endswith(("coverage_followup", "coverage_first_duty_retry",
                                "coverage_batch_decisions")) for mod, _, _ in calls)

    for mode in ("missing", "parent", "context", "source", "unread", "service", "ack", "mechanical"):
        result, calls, _ = run(mode, first_mode=mode)
        assert result.coverage["status"] == "partial"
        assert not any(mod.endswith("coverage_original_batch") for mod, _, _ in calls)
        assert "partial_first_original_audit" not in result.coverage["windows"][0]
    result, calls, _ = run("disabled", enabled=False)
    assert not any(mod.endswith("coverage_original_batch") for mod, _, _ in calls)
    assert "partial_first_original_audit" not in result.coverage["windows"][0]

    # A later independent window keeps its original/global capacity. An audit row
    # with a mechanical defect may not consume those reserved calls for recovery.
    result, calls, _ = run("future", original_mode="mechanical", cap=4,
                           window_chars=60, original_size="auto")
    first, later = result.coverage["windows"]
    partial = first["partial_first_original_audit"]
    assert partial["reserved_future_validation_calls"] == 1
    assert partial["reserved_current_global_validation_calls"] == 1
    assert partial["validation_call_ceiling"] == 2
    assert partial["batch_review"]["remaining_duty_retry"]["status"] == "not_run_budget"
    assert len([p for mod, p, _ in calls if mod.endswith("coverage_original_batch")]) == 2
    assert later["original_recheck"]["status"] == "returned"
    assert result.coverage["budget"]["validation_calls"] == 3
    assert any(mod.endswith("coverage_validation") and p["window_id"] == later["id"]
               for mod, p, _ in calls)
    result, calls, _ = run("whole_wave_budget", cap=2, original_size=2)
    assert result.coverage["windows"][0]["partial_first_original_audit"]["status"] == "not_run_budget"
    assert not any(mod.endswith("coverage_original_batch") for mod, _, _ in calls)

    paper, claims = inputs(1)
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="audit_incomplete_first must be a boolean"):
            m.review_claim_coverage(paper, claims, audit_incomplete_first=value, call=forbidden)
        with pytest.raises(ValueError, match="claim_coverage_partial_first_audit must be a boolean"):
            stage.screen_paper(paper, tmp_path / "invalid", claim_coverage_partial_first_audit=value,
                               call=forbidden)
    forwarded = []
    monkeypatch.setattr(stage, "extract_claims", lambda *a, **k: claims)

    def capture(*args, **kwargs):
        forwarded.append(kwargs["audit_incomplete_first"])
        return m.ClaimCoverageResult(claims=claims, coverage={}, blocked_claim_ids=[], issues=[])

    monkeypatch.setattr(stage, "review_claim_coverage", capture)
    monkeypatch.setattr(stage, "check_writing", lambda *a, **k: [])
    monkeypatch.setattr(stage, "check_tables", lambda *a, **k: [])
    for name in ("check_figures", "check_visual_tables", "check_bibliography"):
        monkeypatch.setattr(stage, name, lambda *a, **k: ([], []))
    stage.screen_paper(paper, tmp_path / "stage_default")
    stage.screen_paper(paper, tmp_path / "stage_off", claim_coverage_partial_first_audit=False)
    assert forwarded == [True, False]
