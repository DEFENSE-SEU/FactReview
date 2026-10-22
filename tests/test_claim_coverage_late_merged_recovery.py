"""One public offline control for bounded recovery of a late atomicity finding."""

import copy
import json
import subprocess
from unittest.mock import Mock

import requests

from fact_generation.execution.tools import docker
from fact_generation.positioning.paper_search import PaperSearchAdapter
from screening import claim_coverage as m
from tests.test_claim_coverage import (
    CFG,
    action,
    candidate,
    extracted,
    followup,
    projection,
    review,
    validation,
)
from tests.test_claim_coverage_v3 import case


def test_late_merged_recovery_preserves_full_duties_and_budgets(tmp_path, monkeypatch):
    external = Mock(side_effect=AssertionError("Unexpected external boundary"))
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", external)
    monkeypatch.setattr(requests.sessions.Session, "request", external)
    monkeypatch.setattr(subprocess, "run", external)
    monkeypatch.setattr(subprocess, "Popen", external)
    monkeypatch.setattr(docker, "docker_ensure_paper_image", external)
    monkeypatch.setattr(docker, "docker_run_paper_image", external)
    monkeypatch.setattr(PaperSearchAdapter, "search", external)
    monkeypatch.setattr(PaperSearchAdapter, "search_structured", external)
    monkeypatch.setattr(PaperSearchAdapter, "read_papers", external)

    modes = (
        "accepted",
        "followup_budget_zero",
        "validation_budget_exhausted",
        "incomplete_originals",
        "invalid_recheck",
        "invalid_candidate",
    )
    for mode in modes:
        paper, first = case()
        raw = [first[0].model_dump(mode="json"), {**candidate(paper.blocks[1]), "id": "claim_002"}]
        paper, claims = projection([b.model_dump(mode="json") for b in paper.blocks], raw)
        originals = [c.model_dump(mode="json") for c in claims]
        materials = paper.model_dump(mode="json")
        registry = m._claim_registry(claims)
        calls, validations, followups = [], [], []

        def call(**kw):
            p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
            calls.append(kw["module"])
            assert p["current_claims"] == registry
            assert paper.model_dump(mode="json") == materials
            assert [c.model_dump(mode="json") for c in claims] == originals
            if kw["module"] == "screening.claims.coverage":
                return review(p, [])
            if kw["module"] == "screening.claims.coverage_followup":
                followups.append(copy.deepcopy(p))
                assert len(p["observations"]) == 1
                assert p["observations"][0]["kind"] == "merged_conclusions"
                assert p["observations"][0]["target_claim_id"] == "claim_001"
                children = []
                for i, text in enumerate(("The method reaches accuracy 90.", "The method uses batch size 16.")):
                    child = extracted(originals[0])
                    child.update(text=text, conditions=[child["conditions"][i]])
                    children.append(child)
                return followup(p, [action(p, p["observations"], "split", children)])
            assert kw["module"] == "screening.claims.coverage_validation"
            validations.append(copy.deepcopy(p))
            assert p["required_original_claim_reviews"] == [
                {"claim_id": r["claim_id"], "digest": r["digest"], "source_block_ids": [r["source_block_id"]]}
                for r in registry
            ]
            raw_validation = validation(p)
            if len(validations) == 1:
                assert not p["observations"] and not p["candidates"]
                group = raw_validation["original_claim_reviews"][0]["assertion_groups"][0]
                raw_validation["original_claim_reviews"][0]["assertion_groups"] = [
                    {**group, "condition_ids": ["c1"], "proposition": "Reported accuracy is 90."},
                    {**group, "condition_ids": ["c2"], "proposition": "Training uses batch size 16."},
                ]
                if mode == "incomplete_originals":
                    raw_validation["original_claim_reviews"].pop()
            elif mode == "invalid_recheck":
                raw_validation["original_claim_reviews"][1]["claim_digest"] = "stale-digest"
            elif mode == "invalid_candidate":
                raw_validation["change_decisions"][0]["new_claim_checks"].pop()
            return raw_validation

        result = m.review_claim_coverage(
            paper,
            claims,
            call=call,
            output_dir=tmp_path / mode,
            max_followup_calls=0 if mode == "followup_budget_zero" else 1,
            max_validation_calls=1 if mode == "validation_budget_exhausted" else 2,
        )
        budget = result.coverage["budget"]
        assert budget["review_calls"] == 1
        assert budget["followup_calls"] <= budget["max_followup_calls"]
        assert budget["validation_calls"] <= budget["max_validation_calls"]
        assert result.coverage["claim_checks_completed"] == result.coverage["claim_checks_required"] == 2
        assert [c.model_dump(mode="json") for c in claims] == originals
        assert paper.model_dump(mode="json") == materials
        if mode in {"accepted", "invalid_recheck", "invalid_candidate"}:
            assert calls == [
                "screening.claims.coverage",
                "screening.claims.coverage_validation",
                "screening.claims.coverage_followup",
                "screening.claims.coverage_validation",
            ]
            assert budget["followup_calls"] == 1 and budget["validation_calls"] == 2
            assert validations[1]["required_original_claim_reviews"] == validations[0]["required_original_claim_reviews"]
            assert validations[1]["scope_context"] == validations[0]["scope_context"] == followups[0]["scope_context"]
            assert len(validations[1]["required_new_claim_checks"]) == 2
            assert len(validations[1]["candidates"]) == 1
            assert validations[1]["observations"][0]["observation"] == followups[0]["observations"][0]
        else:
            assert calls == ["screening.claims.coverage", "screening.claims.coverage_validation"]
            assert budget["followup_calls"] == 0 and budget["validation_calls"] == 1
        if mode == "accepted":
            assert result.coverage["status"] == "complete" and not result.blocked_claim_ids
            assert result.coverage["unresolved_observations"] == 0
            assert result.coverage["original_claim_reviews_completed"] == result.coverage["original_claim_reviews_required"] == 2
            assert result.coverage["candidate_claim_checks_passed"] == result.coverage["candidate_claim_checks_required"] == 2
            assert len(result.claims) == 3 and len(result.coverage["revisions"]) == 1
            assert [c.id for c in result.claims] == ["claim_001", "claim_003", "claim_002"]
            assert [c.text for c in result.claims[:2]] == ["The method reaches accuracy 90.", "The method uses batch size 16."]
            assert result.coverage["revisions"][0]["original_digest"] == registry[0]["digest"]
        else:
            assert result.coverage["status"] == "partial"
            assert result.blocked_claim_ids == ["claim_001"]
            assert [c.model_dump(mode="json") for c in result.claims] == originals
            assert not result.coverage["revisions"]
            assert result.coverage["unresolved_observations"] == 1
            if mode in {"incomplete_originals", "invalid_recheck"}:
                assert result.coverage["original_claim_reviews_completed"] == 1
                assert result.coverage["original_claim_reviews_required"] == 2
    assert external.call_count == 0
