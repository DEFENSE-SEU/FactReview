"""One offline control for a single atomic first-duty recovery wave."""

import copy
import json
import subprocess
from unittest.mock import Mock

import httpx
import requests

from fact_generation.execution.tools import docker
from fact_generation.positioning.paper_search import PaperSearchAdapter
from screening import claim_coverage as m
from tests.test_claim_coverage_v2 import CFG
from tests.test_claim_recheck_batches_v2 import callback, inputs, oracle, saved


def test_first_mechanical_recovery_preserves_closed_duties_raw_and_budgets(tmp_path, monkeypatch):
    external = Mock(side_effect=AssertionError("External boundary forbidden in public mock"))
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

    modes = (
        "accepted", "fresh_invalid", "fresh_stale", "fresh_parent", "fresh_duties", "fresh_ack", "fresh_service",
        "fresh_partial", "review_budget", "future_budget", "validation_budget",
        "valid_unresolved", "initial_incomplete", "discovery_incomplete",
        "discovery_identity", "discovery_lower_invalid", "initial_identity",
        "initial_foreign", "raw_parsed_changed",
    )
    for mode in modes:
        paper, claims = inputs(5)
        for claim in claims:
            claim.conditions[0].settings.update(count=16, fraction=0.25, enabled=True, label="public mock")
        before_paper = paper.model_dump(mode="json")
        before_claims = [claim.model_dump(mode="json") for claim in claims]
        calls, responses, fresh, initial = [], [], [], []
        bad_ids = {"claim_002"} if mode != "fresh_partial" else {f"claim_{i:03d}" for i in range(2, 6)}
        later = callback([])

        def rows_for(payload):
            rows = oracle(payload)["original_claim_reviews"]
            originals = {c["claim_id"]: c for c in payload["current_claims"]}
            span = next(s["block_local_whole_span"] for s in payload["visible_source_spans"] if s["block_id"] == "b1")
            for row in rows:
                # These are injected semantic decisions for artificial settings.
                atoms = []
                for name, value in originals[row["claim_id"]]["conditions"][0]["settings"].items():
                    atoms.append({
                        "id": "setting_" + name, "dimension": "other_boundary", "condition_ids": ["c1"],
                        "state": "preserved", "restriction": "Explicit injected setting " + name,
                        "sources": [{"block_id": "b1", **span}],
                        "carriers": [{"claim_path": "/conditions/0/settings/" + name, "claim_value": value}],
                        "effect": None, "reason": "Mock relation to the unchanged supplied source.",
                    })
                row["scope_atoms"] = atoms
                row["scope_groups"] = [
                    {"condition_ids": [condition], "dimensions": {
                        axis: {"state": "preserved" if axis == "other_boundary" and condition == "c1" else "not_governing",
                               "atom_ids": [a["id"] for a in atoms] if axis == "other_boundary" and condition == "c1" else [],
                               "reason": "Explicit injected dimension judgment."}
                        for axis in m.SCOPE_DIMENSIONS}}
                    for condition in ("c1", "c2")
                ]
                for group in row["source_review_groups"]:
                    if "b1" in group["source_ids"]:
                        group["state"] = "considered"
            return rows

        def call(**kwargs):
            payload = json.loads(kwargs["prompt"].split("DATA_JSON:\n", 1)[1])
            module = kwargs["module"]
            calls.append((module, copy.deepcopy(payload)))
            if module.endswith("coverage_discovery"):
                raw = {
                    "schema_version": "claim-coverage-discovery-v1", "context_id": payload["context_id"],
                    "window_id": payload["window_id"], "parent_digest": payload["parent_digest"],
                    "reviewed_block_ids": [b["id"] for b in payload["blocks"]], "new_findings": [],
                    "explanation": "Explicit full-window omission judgment.",
                }
                if mode == "discovery_incomplete":
                    raw["reviewed_block_ids"].pop()
                elif mode == "discovery_identity":
                    raw["context_id"] = "stale"
                elif mode == "discovery_lower_invalid":
                    raw["new_findings"] = [{"kind": "uncertain", "source_block_ids": ["b1", "b1"],
                                            "reason": "Explicit invalid discovery control."}]
                elif mode == "raw_parsed_changed":
                    raw["explanation"] = "  Explicit full-window omission judgment.  "
                elif mode == "valid_unresolved":
                    raw["new_findings"] = [{"kind": "uncertain", "source_block_ids": ["b2"],
                                            "reason": "Explicit public discovery uncertainty."}]
            elif module.endswith(("coverage_first_batch", "coverage_first_duty_retry")):
                is_fresh = module.endswith("coverage_first_duty_retry")
                (fresh if is_fresh else initial).append(copy.deepcopy(payload))
                if mode == "initial_incomplete" and not is_fresh and len(initial) == 2:
                    raise RuntimeError("Explicit mock initial delegate unavailable")
                if mode == "fresh_service" and is_fresh:
                    raise RuntimeError("Explicit mock fresh delegate unavailable")
                rows = rows_for(payload)
                raw = {
                    "schema_version": "claim-coverage-first-batch-v1", "context_id": payload["context_id"],
                    "window_id": payload["window_id"], "parent_digest": payload["parent_digest"],
                    "batch_id": payload["batch_id"], "claim_reviews": rows,
                }
                if not is_fresh:
                    for row in rows:
                        if row["claim_id"] in bad_ids:
                            row["source_block_ids"] = ["b1", "b1"]
                        if mode == "valid_unresolved" and row["claim_id"] == "claim_004":
                            row.update(state="unresolved", scope_atoms=[])
                            for group in row["scope_groups"]:
                                for dimension in group["dimensions"].values():
                                    dimension.update(state="unresolved", atom_ids=[])
                    if mode == "initial_foreign":
                        for row in rows:
                            if row["claim_id"] == "claim_002":
                                row["source_block_ids"] = ["foreign"]
                    elif mode == "initial_identity" and len(initial) == 2:
                        raw["parent_digest"] = "stale"
                else:
                    if mode == "fresh_invalid" or (mode == "fresh_partial" and len(fresh) == 2):
                        rows[0]["source_block_ids"] = ["b1", "b1"]
                    elif mode == "fresh_stale":
                        raw["context_id"] = "stale"
                    elif mode == "fresh_parent":
                        raw["parent_digest"] = "stale"
                    elif mode == "fresh_duties":
                        rows.pop()
                    elif mode == "fresh_ack":
                        rows[0]["source_review_groups"][0]["state"] = "unavailable"
            else:
                raw = later(**kwargs)
            responses.append((len(calls) - 1, copy.deepcopy(raw)))
            return raw

        options = {"first_review_batch_size": 2, "original_review_batch_size": 2,
                   "max_review_calls": 12, "max_validation_calls": 16}
        if mode == "review_budget":
            options["max_review_calls"] = 4
        elif mode == "future_budget":
            options.update(window_chars=60, max_review_calls=5)
        elif mode == "validation_budget":
            options["max_validation_calls"] = 3
        path = tmp_path / mode
        result = m.review_claim_coverage(paper, claims, call=call, output_dir=path, **options)
        audit = saved(path)
        window = audit["coverage"]["windows"][0]
        first = window["first_batches"]
        retry = first.get("remaining_duty_retry")
        assert paper.model_dump(mode="json") == before_paper
        assert [c.model_dump(mode="json") for c in claims] == before_claims
        assert audit["original_claims"] == audit["result_claims"] == before_claims
        for index, raw in responses:
            assert m._digest(audit["attempts"][index]["response"]) == m._digest(raw)
        assert result.coverage["budget"]["review_calls"] == len([
            module for module, _ in calls if module.endswith(("coverage_discovery", "coverage_first_batch", "coverage_first_duty_retry"))])
        assert result.coverage["budget"]["review_calls"] <= options["max_review_calls"]
        assert not result.coverage["revisions"]

        eligible = mode in {"accepted", "fresh_invalid", "fresh_stale", "fresh_parent", "fresh_duties", "fresh_ack", "fresh_service",
                            "fresh_partial", "review_budget", "future_budget", "validation_budget", "valid_unresolved"}
        if not eligible:
            assert not fresh and retry is None
        else:
            assert first["requests_complete"]
            assert len(first["results"]) == 4 and all(r["status"] == "returned" for r in first["results"])
            assert retry is not None
            assert retry["eligible_claim_ids"] == [c.id for c in claims if c.id in bad_ids]
            assert retry["initial_lowering"]["errors"]
            assert retry["initial_aggregate"]["new_findings"] == first["aggregate"]["new_findings"]
            if mode in {"review_budget", "future_budget", "validation_budget"}:
                assert not fresh and retry["status"] == "not_run_budget"
            else:
                assert window["first_duty_retry_attempted"] is True
                assert len(fresh) == (2 if mode == "fresh_partial" else 1)
                assert [r["claim_id"] for p in fresh for r in p["required_claim_checks"]] == retry["eligible_claim_ids"]
                parent = first["parent_manifest"]
                for payload in fresh:
                    expanded = m.unpack_scope_payload(payload)
                    assert payload["batch_id"].startswith("first_retry_001_")
                    assert payload["parent_digest"] == first["parent_digest"]
                    assert payload["current_claims"] == parent["current_claims"]
                    assert payload["blocks"] == parent["blocks"]
                    assert expanded["scope_context"] == parent["scope_context"]
                    assert expanded["supplemental_sources"] == parent["supplemental_sources"]
                    assert payload["discovery_required_block_ids"] == parent["discovery_required_block_ids"]
                    assert payload["previous_unresolved_observations"] == parent["previous_unresolved_observations"]
                    assert payload["parent_provenance"] == parent["parent_provenance"]
                    assert m._digest({k: v for k, v in payload.items() if k != "context_id"}) == payload["context_id"]
                assert len(retry["batches"]) == len(fresh)
                assert retry["budget_before"]["review_calls"] == 4
                assert retry["budget_after"]["review_calls"] == 4 + len(fresh)

        successful = mode in {"accepted", "valid_unresolved"}
        if successful:
            assert retry["status"] == "returned"
            adapter = first["aggregate"]
            assert adapter["version"] == "program-first-row-selection-adapter-v1"
            assert adapter["context_id"] != retry["initial_aggregate"]["context_id"]
            assert [r["claim_id"] for r in adapter["claim_reviews"]] == [c.id for c in claims]
            initial_rows = {r["claim_id"]: r for r in retry["initial_aggregate"]["claim_reviews"]}
            for row in adapter["claim_reviews"]:
                if row["claim_id"] not in bad_ids:
                    assert m._digest(row) == m._digest(initial_rows[row["claim_id"]])
            kept = adapter["claim_reviews"][0]
            assert [type(a["carriers"][0]["claim_value"]) for a in kept["scope_atoms"]] == [int, float, bool, str]
            assert adapter["discovery_mappings"] == retry["initial_aggregate"]["discovery_mappings"]
            assert adapter["reviewed_block_ids"] == retry["initial_aggregate"]["reviewed_block_ids"]
            assert adapter["explanation"] == retry["initial_aggregate"]["explanation"]
            for selection in adapter["row_selections"]:
                assert selection["raw_audit_path"] and selection["raw_digest"]
                origin = audit["attempts"][int(selection["raw_audit_path"].split("/")[2])]["response"]
                row = origin["claim_reviews"][int(selection["raw_path"].split("/")[2])]
                assert m._digest(row) == selection["raw_digest"]
                assert m._digest(row) == m._digest(adapter["claim_reviews"][int(selection["aggregate_path"].split("/")[2])])
            for mapping in adapter["reviewed_source_mappings"]:
                origin = audit["attempts"][int(mapping["raw_audit_path"].split("/")[2])]["response"]
                parts = mapping["raw_path"].split("/")
                group = origin["claim_reviews"][int(parts[2])]["source_review_groups"][int(parts[4])]
                assert m._digest(group) == mapping["raw_digest"]
            assert not first["lowering"]["errors"]
            assert len([r for r in first["lowering"]["mappings"] if "claim_id" in r]) == 5
        elif retry is not None:
            assert first["aggregate"] == retry["initial_aggregate"]
            assert first["lowering"]["errors"] == retry["initial_lowering"]["errors"]
            if fresh:
                assert retry["status"] == "failed" and retry["errors"]

        if mode == "accepted":
            assert result.coverage["status"] == "complete"
            assert result.coverage["claim_checks_completed"] == result.coverage["claim_checks_required"] == 5
            assert result.coverage["original_claim_reviews_completed"] == result.coverage["original_claim_reviews_required"] == 5
            assert result.coverage["budget"]["validation_calls"] == 3
            assert [module for module, _ in calls[-3:]] == ["screening.claims.coverage_original_batch"] * 3
        else:
            assert result.coverage["status"] == "partial"
            assert window["original_recheck"]["status"] == "not_run_incomplete_first"
            assert not any(module.endswith(("coverage_followup", "coverage_original_batch", "coverage_batch_decisions"))
                           for module, _ in calls)
        if mode == "valid_unresolved":
            assert "claim_004" in first["remaining_required_claim_ids"]
            assert all(r["claim_id"] != "claim_004" for p in fresh for r in p["required_claim_checks"])
            assert first["aggregate"]["claim_reviews"][3]["state"] == "unresolved"
        assert result.coverage["budget"]["followup_calls"] == 0
    assert external.call_count == 0
