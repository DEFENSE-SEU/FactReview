"""One fully offline public control for exact, bounded original-duty recovery."""

import copy
import json
import subprocess
from unittest.mock import Mock

import requests

from fact_generation.execution.tools import docker
from fact_generation.positioning.paper_search import PaperSearchAdapter
from screening import claim_coverage as m
from tests.test_claim_coverage_v2 import CFG
from tests.test_claim_recheck_batches_v2 import callback, inputs, saved


def test_mechanically_invalid_original_duties_get_one_exact_bounded_new_review(tmp_path, monkeypatch):
    external = Mock(side_effect=AssertionError("Unexpected external boundary"))
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", external)
    monkeypatch.setattr(requests.sessions.Session, "request", external)
    monkeypatch.setattr(subprocess, "run", external)
    monkeypatch.setattr(subprocess, "Popen", external)
    monkeypatch.setattr(docker, "docker_ensure_paper_image", external)
    monkeypatch.setattr(docker, "docker_run_paper_image", external)
    for method in ("search", "search_structured", "read_papers"):
        monkeypatch.setattr(PaperSearchAdapter, method, external)
    modes = ("accepted", "late_recovery", "late_retry_exhausted", "valid_unresolved", "zero_budget", "retry_budget_exhausted",
             "decision_budget_reserved", "failed_retry", "stale_retry", "duplicate_retry", "wrong_parent_retry",
             "invalid_scope_retry", "source_changed_retry", "initial_request_failure")
    for mode in modes:
        paper, claims = inputs(11)
        directory = tmp_path / mode
        directory.mkdir()
        source = directory / "source.pdf"
        source.write_bytes(b"operator-owned source identity fixture")
        paper.source_pdf = str(source)
        original_claims = [claim.model_dump(mode="json") for claim in claims]
        original_materials = paper.model_dump(mode="json")
        registry = m._claim_registry(claims)
        calls, retries, regular = [], [], []
        merge = "claim_009" if mode in {"late_recovery", "late_retry_exhausted", "decision_budget_reserved"} else None
        normal = callback(calls, merge=merge)
        initial_first = callback(calls, merge=merge if mode == "decision_budget_reserved" else None)

        def call(**kwargs):
            payload = json.loads(kwargs["prompt"].split("DATA_JSON:\n", 1)[1])
            if kwargs["module"] == "screening.claims.coverage":
                return initial_first(**kwargs)
            is_original = kwargs["module"] == "screening.claims.coverage_original_batch"
            is_retry = is_original and payload["batch_id"].startswith("original_retry_")
            if is_retry:
                retries.append(copy.deepcopy(payload))
                assert len(retries) == 1
                assert payload["current_claims"] == registry
                first = calls[0][1]
                assert payload["blocks"] == first["blocks"]
                assert payload["visible_source_spans"] == first["visible_source_spans"]
                assert payload["source_catalog"] == first["source_catalog"]
                assert m.unpack_scope_payload(payload)["scope_context"] == m.unpack_scope_payload(first)["scope_context"]
                assert payload["parent_provenance"]["original_claims_digest"] == m._digest(original_claims)
                if mode == "failed_retry":
                    calls.append((kwargs["module"], copy.deepcopy(payload)))
                    raise RuntimeError("mock remaining request unavailable")
            elif is_original:
                regular.append(copy.deepcopy(payload))
                if mode == "initial_request_failure" and len(regular) == 2:
                    calls.append((kwargs["module"], copy.deepcopy(payload)))
                    raise RuntimeError("mock initial request unavailable")
            raw = normal(**kwargs)
            if not is_original:
                return raw
            if is_retry:
                if mode == "stale_retry":
                    raw["context_id"] = "stale-context"
                elif mode == "wrong_parent_retry":
                    raw["parent_digest"] = "stale-parent"
                elif mode == "duplicate_retry":
                    raw["original_claim_reviews"][1] = copy.deepcopy(raw["original_claim_reviews"][0])
                elif mode == "invalid_scope_retry":
                    raw["original_claim_reviews"][0]["scope_groups"][0]["condition_ids"].pop()
                elif mode == "source_changed_retry":
                    source.write_bytes(b"source changed during new request")
                return raw
            if len(regular) <= 4:
                for row in raw["original_claim_reviews"]:
                    if row["claim_id"] in {"claim_006", "claim_008"}:
                        if mode == "valid_unresolved" and row["claim_id"] == "claim_008":
                            row["state"] = "unresolved"
                            for group in row["scope_groups"]:
                                for dimension in group["dimensions"].values():
                                    dimension.update(state="unresolved", atom_ids=[])
                        else:
                            row["scope_groups"][0]["condition_ids"].pop()
                    if mode in {"late_recovery", "late_retry_exhausted"} and row["claim_id"] == "claim_009":
                        group = row["assertion_groups"][0]
                        row["assertion_groups"] = [{**copy.deepcopy(group), "condition_ids": [key]}
                                                   for key in ("c1", "c2")]
            elif mode == "late_retry_exhausted":
                for row in raw["original_claim_reviews"]:
                    if row["claim_id"] == "claim_006":
                        row["scope_groups"][0]["condition_ids"].pop()
            return raw

        budget_limit = (0 if mode == "zero_budget" else 4 if mode == "retry_budget_exhausted"
                        else 5 if mode == "decision_budget_reserved" else 12 if mode == "late_retry_exhausted" else 10)
        result = m.review_claim_coverage(paper, claims, call=call, output_dir=directory,
            first_review_batch_size=None, original_review_batch_size=3,
            max_validation_calls=budget_limit, max_followup_calls=1)
        audit = saved(directory)
        window = result.coverage["windows"][0]
        budget = result.coverage["budget"]
        assert budget["validation_calls"] <= budget_limit
        assert [claim.model_dump(mode="json") for claim in claims] == original_claims
        assert paper.model_dump(mode="json") == original_materials
        assert len(retries) <= 1
        assert audit["original_claims"] == original_claims
        if mode in {"zero_budget", "retry_budget_exhausted", "decision_budget_reserved", "initial_request_failure"}:
            assert not retries
            assert result.coverage["status"] == "partial" and not result.coverage["revisions"]
            assert not any(module.endswith("coverage_batch_decisions") for module, _ in calls)
            assert budget["validation_calls"] == (0 if mode == "zero_budget" else 4)
            continue
        assert len(retries) == 1
        assert window["original_duty_retry_attempted"] is True
        required = [row["claim_id"] for row in retries[0]["required_original_claim_reviews"]]
        assert required == (["claim_006"] if mode == "valid_unresolved" else ["claim_006", "claim_008"])
        first_record = (audit["coverage"]["windows"][0]["late_original_recovery"]["initial_original_batches"]
                        if mode in {"late_recovery", "late_retry_exhausted"} else window["original_batches"])
        retry_audit = first_record["remaining_duty_retry"]
        assert retry_audit["initial_precheck"]["errors"]
        assert retry_audit["initial_aggregate"]["original_claim_reviews"]
        assert retries[0]["parent_digest"] == m._digest(first_record["parent_manifest"])
        assert first_record["parent_manifest"]["required_original_claim_reviews"] == [
            {"claim_id": row["claim_id"], "digest": row["digest"], "source_block_ids": [row["source_block_id"]]}
            for row in registry]
        assert retries[0]["batch_id"] not in {payload["batch_id"] for payload in regular}
        if mode in {"accepted", "late_recovery", "late_retry_exhausted", "valid_unresolved"}:
            aggregate = first_record["aggregate"]
            assert len(aggregate["original_claim_reviews"]) == 11
            assert len(aggregate["row_selections"]) == 11
            assert retry_audit["final_aggregate_digest"] == m._digest(aggregate)
            original_rows = retry_audit["initial_aggregate"]["original_claim_reviews"]
            final_rows = {row["claim_id"]: row for row in aggregate["original_claim_reviews"]}
            for row in original_rows:
                if row["claim_id"] not in required:
                    assert final_rows[row["claim_id"]] == row
            assert len(aggregate["raw_receipts"]) == 5 and not aggregate["aggregation_errors"]
            if mode == "late_retry_exhausted":
                assert result.coverage["status"] == "partial" and not result.coverage["revisions"]
                assert len(result.claims) == 11 and window["original_recheck"]["completed"] == 10
                assert budget["validation_calls"] == 9
                assert not any(module.endswith("coverage_batch_decisions") for module, _ in calls)
            elif mode == "valid_unresolved":
                assert final_rows["claim_008"]["state"] == "unresolved"
                assert result.coverage["status"] == "partial" and not result.coverage["revisions"]
                assert window["original_recheck"]["completed"] == 10
            else:
                assert result.coverage["status"] == "complete" and not result.blocked_claim_ids
                assert window["original_recheck"]["completed"] == window["original_recheck"]["required"] == 11
                assert len(result.coverage["revisions"]) == (1 if mode == "late_recovery" else 0)
                assert len(result.claims) == (12 if mode == "late_recovery" else 11)
                assert budget["validation_calls"] == (10 if mode == "late_recovery" else 5)
        else:
            assert result.coverage["status"] == ("failed" if mode == "source_changed_retry" else "partial")
            assert not result.coverage["revisions"]
            assert [claim.model_dump(mode="json") for claim in result.claims] == original_claims
            assert not any(module.endswith("coverage_batch_decisions") for module, _ in calls)
            assert budget["validation_calls"] == 5
            assert retry_audit["errors"]
    assert external.call_count == 0
