"""One public all-mock control for fixed first slots and one complete recovery wave."""

import copy
import inspect
import json
import subprocess
from unittest.mock import Mock

import httpx
import pytest
import requests

from fact_generation.execution.tools import docker
from fact_generation.positioning.paper_search import PaperSearchAdapter
from screening import claim_coverage as m
from screening import stage
from tests.test_claim_coverage_v2 import CFG
from tests.test_claim_recheck_batches_v2 import callback, inputs, oracle, saved


def test_first_partial_recovery_fixed_slots_full_context_and_single_wave(tmp_path, monkeypatch):
    external = Mock(side_effect=AssertionError("External model/retrieval/network/Docker forbidden"))
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", external)
    monkeypatch.setattr(requests.sessions.Session, "request", external)
    monkeypatch.setattr(httpx.Client, "send", external)
    monkeypatch.setattr(httpx.AsyncClient, "send", external)
    monkeypatch.setattr(subprocess, "run", external)
    monkeypatch.setattr(subprocess, "Popen", external)
    monkeypatch.setattr(docker, "docker_ensure_paper_image", external)
    monkeypatch.setattr(docker, "docker_run_paper_image", external)
    for method in ("search", "search_structured", "read_papers"):
        monkeypatch.setattr(PaperSearchAdapter, method, external)
    supported = "first_partial_recovery_rounds" in inspect.signature(m.review_claim_coverage).parameters
    modes = (
        "order", "schema", "mixed", "initial_unresolved", "fresh_unresolved", "fresh_failure",
        "fresh_invalid", "fresh_stale", "fresh_foreign", "initial_stale", "initial_foreign",
        "unknown", "drift", "review_budget", "validation_budget", "future_budget", "healthy", "default0",
        "schema_identity", "schema_source", "missing_identity", "nonjson", "schema_ack", "schema_empty",
    )
    for mode in modes:
        paper, claims = inputs(7)
        before = [claim.model_dump(mode="json") for claim in claims]
        calls, returned, fresh = [], [], []
        later = callback([])

        def unresolved(row):
            row.update(state="unresolved", other_findings=[], scope_atoms=[])
            for group in row["scope_groups"]:
                for dimension in group["dimensions"].values():
                    dimension.update(state="unresolved", atom_ids=[])

        def call(**kwargs):
            payload = json.loads(kwargs["prompt"].split("DATA_JSON:\n", 1)[1])
            module = kwargs["module"]
            calls.append((module, copy.deepcopy(payload)))
            if module.endswith("coverage_discovery"):
                raw = {
                    "schema_version": "claim-coverage-discovery-v1", "context_id": payload["context_id"],
                    "window_id": payload["window_id"], "parent_digest": payload["parent_digest"],
                    "reviewed_block_ids": [block["id"] for block in payload["blocks"]],
                    "new_findings": [], "explanation": "Explicit injected full-window reading.",
                }
            elif module.endswith(("coverage_first_batch", "coverage_first_partial_recheck")):
                is_fresh = module.endswith("coverage_first_partial_recheck")
                if is_fresh:
                    fresh.append(copy.deepcopy(payload))
                    if mode == "fresh_failure":
                        raise RuntimeError("Explicit mock delegate unavailable")
                raw = {
                    "schema_version": "claim-coverage-first-batch-v1", "context_id": payload["context_id"],
                    "window_id": payload["window_id"], "parent_digest": payload["parent_digest"],
                    "batch_id": payload["batch_id"], "claim_reviews": oracle(payload)["original_claim_reviews"],
                }
                if not is_fresh and payload["batch_id"] == "first_batch_002":
                    if mode == "unknown":
                        raise RuntimeError("ValidationError required order budget usage unknown; untrusted text")
                    if mode == "drift":
                        paper.markdown += "\nChanged input by mock."
                    if mode in {"schema", "schema_identity", "schema_source", "schema_ack", "schema_empty"}:
                        raw["claim_reviews"][0]["state"] = "invalid"
                        if mode == "schema_identity":
                            raw["claim_reviews"][0]["claim_id"] = "foreign"
                        elif mode == "schema_source":
                            raw["claim_reviews"][0]["source_block_ids"] = ["foreign"]
                        elif mode == "schema_ack":
                            raw["claim_reviews"][0]["source_review_groups"] = []
                        elif mode == "schema_empty":
                            raw["claim_reviews"][0]["source_block_ids"] = []
                    elif mode == "missing_identity":
                        raw.pop("parent_digest")
                    elif mode == "nonjson":
                        raw["claim_reviews"][0]["reason"] = float("inf")
                    elif mode == "initial_unresolved":
                        unresolved(raw["claim_reviews"][0])
                    elif mode == "initial_stale":
                        raw["context_id"] = "stale"
                    elif mode == "initial_foreign":
                        raw["claim_reviews"][0]["source_block_ids"] = ["foreign"]
                    elif mode != "healthy":
                        raw["claim_reviews"].pop()
                if not is_fresh and mode == "mixed" and payload["batch_id"] == "first_batch_003":
                    unresolved(raw["claim_reviews"][0])
                if is_fresh:
                    if mode == "fresh_unresolved":
                        unresolved(raw["claim_reviews"][0])
                    elif mode == "fresh_invalid":
                        raw["claim_reviews"][0]["source_block_ids"] = ["b1", "b1"]
                    elif mode == "fresh_stale":
                        raw["context_id"] = "stale"
                    elif mode == "fresh_foreign":
                        raw["claim_reviews"][0]["source_block_ids"] = ["foreign"]
            else:
                raw = later(**kwargs)
            returned.append((len(calls) - 1, copy.deepcopy(raw)))
            return raw

        options = {"first_review_batch_size": 3, "original_review_batch_size": 3,
                   "max_review_calls": 12, "max_validation_calls": 12}
        if supported:
            options["first_partial_recovery_rounds"] = 0 if mode == "default0" else 1
        if mode == "review_budget":
            options["max_review_calls"] = 4
        elif mode == "validation_budget":
            options["max_validation_calls"] = 2
        elif mode == "future_budget":
            options.update(window_chars=60, max_review_calls=5)
        path = tmp_path / mode
        result = m.review_claim_coverage(paper, claims, call=call, output_dir=path, **options)
        audit = saved(path)
        window = audit["coverage"]["windows"][0]
        first = window["first_batches"]
        initial = [p for module, p in calls if module.endswith("coverage_first_batch")
                   and p["window_id"] == window["id"]]
        assert [claim.model_dump(mode="json") for claim in claims] == before
        assert [claim.model_dump(mode="json") for claim in result.claims] == before
        assert audit["original_claims"] == before and not result.coverage["revisions"]
        for index, raw in returned:
            assert audit["attempts"][index]["response"] == raw
        stopped = {"default0", "initial_stale", "unknown", "drift", "schema_identity", "schema_source", "missing_identity", "nonjson", "schema_ack", "schema_empty"}
        if mode in stopped:
            assert len(initial) == 2 and not fresh
        else:
            assert [p["batch_id"] for p in initial] == ["first_batch_001", "first_batch_002", "first_batch_003"]
        if mode == "default0":
            assert "partial_recovery" not in first and "initial_batch_slots" not in first
            assert result.coverage["budget"]["review_calls"] == 3
        elif mode not in stopped | {"initial_foreign", "healthy"}:
            slots = first["initial_batch_slots"]
            assert [slot["batch_id"] for slot in slots] == [p["batch_id"] for p in initial]
            assert [slot["result"]["status"] for slot in slots] == (
                ["returned"] * 3 if mode == "initial_unresolved" else ["returned", "failed", "returned"])
            assert all(slot["payload_digest"] == m._digest(slot["payload"]) for slot in slots)
            recovery = first["partial_recovery"]
            assert recovery["initial_aggregate"] and recovery["initial_errors"] == first["errors"]
            if mode in {"review_budget", "validation_budget", "future_budget"}:
                assert not fresh and recovery["status"] == "not_run_budget"
            else:
                assert window["first_duty_retry_attempted"] is True
                assert len(fresh) == (2 if mode == "mixed" else 1)
                assert [r["claim_id"] for p in fresh for r in p["required_claim_checks"]] == recovery["eligible_claim_ids"]
                for payload in fresh:
                    expanded = m.unpack_scope_payload(payload)
                    parent = first["parent_manifest"]
                    for key in ("current_claims", "blocks", "scope_context", "supplemental_sources",
                                "previous_unresolved_observations", "discovery_required_block_ids", "parent_provenance"):
                        assert expanded[key] == parent[key]
                    assert payload["parent_digest"] == first["parent_digest"]
                    assert payload["context_id"] == m._digest({key: value for key, value in payload.items() if key != "context_id"})
        if mode in {"order", "schema", "mixed", "initial_unresolved"}:
            assert result.coverage["status"] == "complete"
            assert result.coverage["claim_checks_completed"] == result.coverage["original_claim_reviews_completed"] == 7
            assert first["selected_requests_complete"] is True and first["selected_errors"] == []
            selected = first["aggregate"]
            assert [row["claim_id"] for row in selected["claim_reviews"]] == [claim.id for claim in claims]
            initial_rows = {row["claim_id"]: row for row in first["partial_recovery"]["initial_aggregate"]["claim_reviews"]}
            for selection, row in zip(selected["row_selections"], selected["claim_reviews"], strict=True):
                origin = audit["attempts"][int(selection["raw_audit_path"].split("/")[2])]
                source_row = origin["response"]["claim_reviews"][int(selection["raw_path"].split("/")[2])]
                assert row == source_row and selection["raw_digest"] == m._digest(source_row)
                assert selection["input_digest"] == m._digest(origin["input"])
                if row["claim_id"] not in first["partial_recovery"]["eligible_claim_ids"]:
                    assert row == initial_rows[row["claim_id"]]
                else:
                    assert origin["module"].endswith("coverage_first_partial_recheck")
            assert selected["new_findings"] == first["partial_recovery"]["initial_aggregate"]["new_findings"]
            assert selected["discovery_mappings"] == first["partial_recovery"]["initial_aggregate"]["discovery_mappings"]
            assert selected["reviewed_block_ids"] == first["partial_recovery"]["initial_aggregate"]["reviewed_block_ids"]
            if mode in {"mixed", "initial_unresolved"}:
                assert selected["review_differences"]
                assert all(difference["previous_row"]["state"] == "unresolved" for difference in selected["review_differences"])
            if mode in {"order", "schema", "mixed"}:
                assert not first["requests_complete"] and first["errors"]
                assert first["initial_batch_slots"][1]["result"]["failure_stage"] == ("schema" if mode == "schema" else "required_order")
        elif mode == "healthy":
            assert not fresh and "partial_recovery" not in first
            assert result.coverage["status"] == "complete" and result.coverage["budget"]["review_calls"] == 4
        else:
            assert result.coverage["status"] in {"partial", "failed"}
            assert window["original_recheck"]["status"] == "not_run_incomplete_first"
            assert not any(module.endswith(("coverage_original_batch", "coverage_batch_decisions", "coverage_followup"))
                           for module, _ in calls)
            if mode.startswith("fresh_"):
                recovery = first["partial_recovery"]
                if mode == "fresh_unresolved":
                    assert recovery["status"] == "returned" and first["remaining_required_claim_ids"]
                else:
                    assert recovery["status"] == "failed" and first["aggregate"] == recovery["initial_aggregate"]
        assert result.coverage["budget"]["review_calls"] <= options["max_review_calls"]
        assert result.coverage["budget"]["validation_calls"] <= options["max_validation_calls"]
        if mode == "unknown":
            assert first["initial_batch_slots"][1]["result"]["failure_stage"] == "unclassified"
            assert "partial_recovery" not in first

    if supported:
        for value in (True, 2, -1, None):
            paper, claims = inputs(7)
            with pytest.raises(ValueError):
                m.review_claim_coverage(paper, claims, call=external, first_partial_recovery_rounds=value)
    captured = []
    paper, claims = inputs(7)
    monkeypatch.setattr(stage, "extract_claims", lambda *args, **kwargs: claims)

    def coverage(*args, **kwargs):
        captured.append(kwargs)
        return m.ClaimCoverageResult(claims, {"status": "not_run"}, [], [])

    monkeypatch.setattr(stage, "review_claim_coverage", coverage)
    monkeypatch.setattr(stage, "check_writing", lambda *args, **kwargs: [])
    monkeypatch.setattr(stage, "check_tables", lambda *args, **kwargs: [])
    for name in ("check_figures", "check_visual_tables", "check_bibliography"):
        monkeypatch.setattr(stage, name, lambda *args, **kwargs: ([], []))
    stage.screen_paper(paper, tmp_path / "framework-default", call=external)
    assert captured[-1]["first_partial_recovery_rounds"] == 1
    stage.screen_paper(paper, tmp_path / "framework-disabled", call=external,
                       claim_coverage_first_partial_recovery_rounds=0)
    assert captured[-1]["first_partial_recovery_rounds"] == 0
    assert external.call_count == 0
