"""Public first-batch admission/source acknowledgement; all external calls are mocked."""

import copy
import json

import pytest

from screening import claim_coverage as m
from screening.claim_scope import fingerprint
from tests.test_claim_coverage_v2 import CFG
from tests.test_claim_recheck_batches_v2 import inputs, oracle, saved


def test_discovery_admission_and_explicit_child_source_ack_preserve_public_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("External LLM forbidden"))
    monkeypatch.setattr("requests.sessions.Session.request", lambda *a, **k: pytest.fail("Network forbidden"))
    monkeypatch.setattr("httpx.Client.send", lambda *a, **k: pytest.fail("HTTP forbidden"))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("Docker/process forbidden"))
    paper, claims = inputs(1)
    before_paper = paper.model_dump(mode="json")
    before_claims = [c.model_dump(mode="json") for c in claims]

    def run(mode):
        calls, raw_responses = [], []

        def call(**kw):
            payload = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
            calls.append((kw["module"], copy.deepcopy(payload)))
            ids = [b["id"] for b in payload["blocks"]]
            if kw["module"].endswith("coverage_discovery"):
                reviewed = list(ids)
                if mode == "partial":
                    reviewed.remove("b1")
                elif mode == "foreign":
                    reviewed.append("foreign")
                elif mode == "duplicate":
                    reviewed.append(ids[0])
                elif mode == "order":
                    reviewed.reverse()
                raw = {
                    "schema_version": "claim-coverage-discovery-v1",
                    "context_id": payload["context_id"], "window_id": payload["window_id"],
                    "parent_digest": payload["parent_digest"],
                    "reviewed_block_ids": reviewed, "new_findings": [],
                    "explanation": "Mock full-window omission judgment with explicit read IDs.",
                }
            elif kw["module"].endswith("coverage_first_batch"):
                rows = oracle(payload)["original_claim_reviews"]
                claim = payload["current_claims"][0]
                block = next(b for b in payload["blocks"] if b["id"] == "b1")
                row = rows[0]
                condition_ids = [c["id"] for c in claim["conditions"]]
                row["scope_atoms"] = [{
                    "id": "preserved-original-b1", "dimension": "other_boundary",
                    "condition_ids": condition_ids, "state": "preserved",
                    "restriction": block["text"],
                    "sources": [{"block_id": "b1", "start": 0, "end": len(block["text"])}],
                    "carriers": [{"claim_path": "/text", "claim_value": claim["text"]}],
                    "effect": None, "reason": "Mock confirms this unchanged claim preserves the original restriction.",
                }]
                for group in row["scope_groups"]:
                    group["dimensions"]["other_boundary"] = {
                        "state": "preserved", "atom_ids": ["preserved-original-b1"],
                        "reason": "Explicit full-condition original carrier review.",
                    }
                source_group = next(g for g in row["source_review_groups"] if "b1" in g["source_ids"])
                source_group["state"] = "considered"
                if mode == "unavailable":
                    source_group["state"] = "unavailable"
                elif mode == "missing":
                    row["source_review_groups"].remove(source_group)
                raw = {
                    "schema_version": "claim-coverage-first-batch-v1",
                    "context_id": payload["context_id"], "window_id": payload["window_id"],
                    "parent_digest": payload["parent_digest"], "batch_id": payload["batch_id"],
                    "claim_reviews": rows,
                }
            else:
                assert kw["module"] == "screening.claims.coverage_original_batch"
                original = oracle(payload)
                raw = {k: original[k] for k in (
                    "context_id", "window_id", "original_claim_reviews", "observation_links")}
                raw.update(schema_version="claim-original-recheck-batch-v1",
                           parent_digest=payload["parent_digest"], batch_id=payload["batch_id"])
            raw_responses.append(copy.deepcopy(raw))
            return raw

        path = tmp_path / mode
        result = m.review_claim_coverage(paper, claims, first_review_batch_size=3,
                                        original_review_batch_size=3, call=call, output_dir=path)
        audit = saved(path)
        assert [a["response"] for a in audit["attempts"]] == raw_responses
        assert paper.model_dump(mode="json") == before_paper
        assert [c.model_dump(mode="json") for c in claims] == before_claims
        assert audit["original_claims"] == audit["result_claims"] == before_claims
        return result, calls, audit

    partial, partial_calls, partial_audit = run("partial")
    assert [mod for mod, _ in partial_calls] == ["screening.claims.coverage_discovery"]
    partial_window = partial.coverage["windows"][0]
    assert partial.coverage["status"] == "partial" and not partial.coverage["revisions"]
    assert partial_window["first_batches"]["discovery_remaining_block_ids"] == ["b1"]
    assert not partial_window["first_batches"]["requests_complete"]
    assert partial_window["first_batches"]["remaining_required_claim_ids"] == [claims[0].id]
    assert partial_window["original_recheck"]["status"] == "not_run_incomplete_first"
    assert partial_audit["attempts"][0]["response"]["reviewed_block_ids"] == ["b2"]

    healthy, calls, audit = run("healthy")
    assert [mod for mod, _ in calls] == ["screening.claims.coverage_discovery",
                                       "screening.claims.coverage_first_batch",
                                       "screening.claims.coverage_original_batch"]
    parent = healthy.coverage["windows"][0]["first_batches"]["parent_manifest"]
    assert parent["discovery_required_block_ids"] == [b.id for b in paper.blocks]
    assert parent["review_only_block_ids"] == []
    assert all(p["blocks"] == calls[0][1]["blocks"] and p["current_claims"] == calls[0][1]["current_claims"]
               for mod, p in calls if mod.endswith(("coverage_discovery", "coverage_first_batch")))
    assert healthy.coverage["claim_checks_completed"] == healthy.coverage["claim_checks_required"] == 1
    assert healthy.coverage["status"] == "complete" and not healthy.coverage["revisions"]
    window = healthy.coverage["windows"][0]
    first = window["first_batches"]
    assert first["discovery_remaining_block_ids"] == [] and first["requests_complete"]
    assert not first["lowering"]["errors"] and first["lowering"]["wire_mappings"]
    assert window["original_recheck"]["completed"] == window["original_recheck"]["required"] == 1
    assert window["original_recheck"]["status"] == "returned"
    assert healthy.coverage["budget"]["review_calls"] == 2 and healthy.coverage["budget"]["validation_calls"] == 1
    adapter = first["aggregate"]
    assert adapter["reviewed_block_ids"] == [b.id for b in paper.blocks]
    mappings = adapter["reviewed_source_mappings"]
    raw_group = audit["attempts"][1]["response"]["claim_reviews"][0]["source_review_groups"][0]
    assert any(r["raw_audit_path"] == "/attempts/1"
               and r["raw_path"] == "/claim_reviews/0/source_review_groups/0"
               and r["raw_digest"] == m._digest(raw_group) for r in mappings)
    ack = next(r for r in mappings if r["block_id"] == "b1")
    assert ack["block_digest"] == fingerprint(paper.blocks[0].model_dump(mode="json"))
    assert ack["source_path"] == ack["raw_path"] + "/source_ids/0"
    assert ack["condition_ids"] == [c.id for c in claims[0].conditions] and ack["state"] == "considered"
    assert ack["claim_id"] == claims[0].id and ack["aggregate_path"] == "/reviewed_block_ids/0"

    for mode in ("foreign", "duplicate", "order", "unavailable", "missing"):
        failed, failed_calls, failed_audit = run(mode)
        assert failed.coverage["status"] == "partial" and not failed.coverage["revisions"]
        assert not any(mod.endswith(("coverage_followup", "coverage_original_batch", "coverage_batch_decisions"))
                       for mod, _ in failed_calls)
        failed_window = failed.coverage["windows"][0]
        assert failed_window["first_batches"]["remaining_required_claim_ids"] == [claims[0].id]
        assert failed_window["original_recheck"]["status"] == "not_run_incomplete_first"
        if mode in {"foreign", "duplicate", "order"}:
            assert len(failed_calls) == 1 and not failed_window["first_batches"]["requests_complete"]
            assert failed_audit["attempts"][0]["response"]["reviewed_block_ids"] != ["b1", "b2"]
        else:
            assert len(failed_calls) == 2
            assert failed.coverage["claim_checks_completed"] == 0
            assert failed_window["first_batches"]["errors"] or failed_window.get("v3_lowering_errors")
            assert all(r["state"] in {"considered", "irrelevant"}
                       for r in failed_window["first_batches"].get("aggregate", {}).get("reviewed_source_mappings", []))
