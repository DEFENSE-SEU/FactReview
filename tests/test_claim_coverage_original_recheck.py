"""Offline controls for independent original-claim recheck, never model-recall tests."""

import copy
import json

import pytest

from screening import claim_coverage as m
from tests.test_claim_coverage_v2 import CFG, action, extracted, followup, validation
from tests.test_claim_coverage_v3 import QUALIFIER, case, review


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("Unexpected external request"))


def revalidated(p, *, groups=1, findings=(), state="resolved", link=False):
    raw = validation(p)
    row = review(p, groups=groups, findings=findings, state=state)["claim_reviews"][0]
    raw.update(
        schema_version="claim-coverage-validation-v4"
        if "scope_context" in p
        else "claim-coverage-validation-v3",
        original_claim_reviews=[row],
        observation_links=[],
    )
    if link:
        for existing in p["observations"]:
            kind = existing["observation"]["kind"]
            suffix = (
                "/assertion_groups"
                if kind == "merged_conclusions"
                else "/state"
                if kind == "uncertain"
                else "/findings/0"
            )
            raw["observation_links"].append(
                {
                    "review_path": "/original_claim_reviews/0" + suffix,
                    "observation_id": existing["observation_id"],
                    "observation_digest": existing["observation_digest"],
                    "reason": "Independent mock identifies the exact same original assertion problem.",
                }
            )
    return raw


def run(tmp_path, *, first_groups=1, first_findings=(), validate=None, budget=12, first_selected=None):
    paper, claims = case()
    originals = [c.model_dump(mode="json") for c in claims]
    modules = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        modules.append(kw["module"])
        if kw["module"] == "screening.claims.coverage":
            raw = review(p, groups=first_groups, findings=first_findings)
            if first_selected is not None:
                raw["reviewed_block_ids"] = first_selected
            return raw
        if kw["module"] == "screening.claims.coverage_followup":
            assert first_groups == 2
            children = []
            for i, text in enumerate(["The method reaches accuracy 90.", "The method uses batch size 16."]):
                child = extracted(originals[0])
                child.update(text=text, conditions=[child["conditions"][i]])
                children.append(child)
            return followup(p, [action(p, p["observations"], "split", children)])
        return validate(p)

    result = m.review_claim_coverage(
        paper, claims, call=call, output_dir=tmp_path, max_validation_calls=budget
    )
    assert [c.model_dump(mode="json") for c in claims] == originals
    audit = json.loads((tmp_path / "coverage.json").read_text())
    return result, modules, audit


def test_clean_first_review_still_rechecks_and_new_qualifier_blocks(tmp_path):
    result, modules, audit = run(tmp_path, validate=lambda p: revalidated(p, findings=[QUALIFIER]))
    assert modules == ["screening.claims.coverage", "screening.claims.coverage_validation"]
    assert result.blocked_claim_ids == ["claim_001"]
    assert result.coverage["status"] == "partial" and not result.coverage["revisions"]
    check = result.coverage["windows"][0]["original_recheck"]
    assert check["completed"] == 1 and len(check["findings"]) == 1
    assert check["findings"][0]["blocked_claim_ids"] == ["claim_001"]
    assert audit["attempts"][1]["response"]["original_claim_reviews"][0]["findings"] == [QUALIFIER]
    assert audit["original_claims"] == audit["result_claims"]


def test_same_problem_link_allows_validated_split_without_duplicate_pending(tmp_path):
    result, modules, _ = run(tmp_path, first_groups=2, validate=lambda p: revalidated(p, groups=2, link=True))
    assert len(modules) == 3 and len(result.claims) == 2
    assert not result.blocked_claim_ids and result.coverage["unresolved_observations"] == 0
    assert result.coverage["status"] == "complete"
    summary = m.coverage_summary(result.coverage)
    assert summary["original_claim_reviews_completed"] == summary["original_claim_reviews_required"] == 1
    from review.report.v2 import claim_coverage_lines

    assert "Independent original-claim reviews: 1 / 1." in claim_coverage_lines(summary)
    assert result.coverage["windows"][0]["original_recheck"]["findings"][0]["linked_observation"]


def test_new_unlinked_problem_survives_old_action_and_propagates_to_split_children(tmp_path):
    result, _, _ = run(
        tmp_path, first_groups=2, validate=lambda p: revalidated(p, groups=2, findings=[QUALIFIER], link=True)
    )
    assert len(result.claims) == 2 and len(result.coverage["revisions"]) == 1
    assert result.blocked_claim_ids == ["claim_001", "claim_002"]
    assert result.coverage["unresolved_observations"] == 1


@pytest.mark.parametrize("corrupt", ["digest", "closed_pointer", "kind", "duplicate", "dismiss"])
def test_invalid_or_conflicting_link_cannot_clear_pending(tmp_path, corrupt):
    def invalid(p):
        raw = revalidated(p, groups=2, link=True)
        link = raw["observation_links"][0]
        if corrupt == "digest":
            link["observation_digest"] = "wrong"
        elif corrupt == "closed_pointer":
            link["review_path"] = "/original_claim_reviews/0/preserved_qualifiers/0"
        elif corrupt == "kind":
            raw["original_claim_reviews"][0]["findings"] = [QUALIFIER]
            link["review_path"] = "/original_claim_reviews/0/findings/0"
        elif corrupt == "duplicate":
            raw["observation_links"].append(copy.deepcopy(link))
        else:
            raw["observation_decisions"][0]["verdict"] = "dismiss_observation"
        return raw

    result, _, _ = run(tmp_path, first_groups=2, validate=invalid)
    assert len(result.claims) == 1 and not result.coverage["revisions"]
    assert result.blocked_claim_ids == ["claim_001"]
    assert result.coverage["windows"][0]["original_recheck"]["link_errors"]
    assert result.coverage["status"] == "partial"


def test_legacy_validation_readable_but_cannot_complete_original_recheck(tmp_path):
    def legacy(p):
        raw = validation(p)
        raw.update(schema_version="claim-coverage-validation-v2")
        raw.pop("original_claim_reviews")
        raw.pop("observation_links")
        return raw

    result, modules, _ = run(tmp_path, validate=legacy)
    assert len(modules) == 2 and not result.blocked_claim_ids
    assert result.coverage["status"] == "partial"
    assert result.coverage["windows"][0]["original_recheck"]["status"] == "legacy_unreviewed"


def test_budget_exhaustion_is_incomplete_without_inventing_defect(tmp_path):
    result, modules, _ = run(tmp_path, budget=0)
    assert len(modules) == 1 and not result.blocked_claim_ids
    assert result.coverage["status"] == "partial" and result.coverage["original_claim_reviews_completed"] == 0
    assert result.coverage["windows"][0]["validation_status"] == "not_run_budget"


def test_unresolved_recheck_records_uncertainty_without_blocking(tmp_path):
    result, _, _ = run(tmp_path, validate=lambda p: revalidated(p, state="unresolved"))
    assert not result.blocked_claim_ids and not result.coverage["revisions"]
    assert result.coverage["status"] == "partial"
    check = result.coverage["windows"][0]["original_recheck"]
    assert check["completed"] == 0 and check["findings"][0]["kind"] == "uncertain"


def test_original_recheck_does_not_backfill_first_review_range_or_checks(tmp_path):
    result, modules, _ = run(tmp_path, first_selected=["b2"], validate=revalidated)
    assert len(modules) == 2
    assert result.coverage["claim_checks_completed"] == 0
    assert result.coverage["original_claim_reviews_completed"] == 1
    assert result.coverage["windows"][0]["reviewed_block_ids"] == ["b2"]
    assert result.coverage["windows"][0]["status"] == "partially_reviewed"
    assert result.coverage["status"] == "partial" and not result.blocked_claim_ids


@pytest.mark.parametrize("mode", ["stale_digest", "legacy_validation"])
def test_invalid_original_recheck_cannot_authorize_split_or_clear_old_pending(tmp_path, mode):
    def invalid(p):
        raw = revalidated(p, groups=2)
        if mode == "stale_digest":
            raw["original_claim_reviews"][0]["claim_digest"] = "stale-independent-digest"
        else:
            raw.update(schema_version="claim-coverage-validation-v2")
            raw.pop("original_claim_reviews")
            raw.pop("observation_links")
        return raw

    result, modules, _ = run(tmp_path, first_groups=2, validate=invalid)
    assert len(modules) == 3
    assert len(result.claims) == 1 and not result.coverage["revisions"]
    assert result.blocked_claim_ids == ["claim_001"] and result.coverage["status"] == "partial"
    assert result.coverage["windows"][0]["original_recheck"]["completed"] == 0
