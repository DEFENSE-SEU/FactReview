"""Necessary source and semantic-contract controls; all judgments are mocks."""

import copy
import json

import pytest

from screening import claim_coverage as m
from tests.claim_scope_mock import with_scope
from tests.test_claim_coverage_v2 import CFG, validation
from tests.test_claim_coverage_v3 import case, review


def catalog():
    paper, claims = case()
    registry = m._claim_registry(claims)
    required = [
        {"claim_id": registry[0]["claim_id"], "digest": registry[0]["digest"], "source_block_ids": ["b1"]}
    ]
    return paper, registry, required


def scope_row():
    paper, registry, required = catalog()
    context, _extras = m._scope_context(
        registry, required, ["b1", "b2"], {b.id: b for b in paper.blocks}, paper.markdown, 200
    )
    payload = {
        "current_claims": registry,
        "context_id": "mock",
        "window_id": "mock",
        "blocks": [b.model_dump(mode="json") for b in paper.blocks],
    }
    row = review(payload)["claim_reviews"][0]
    dimensions = {
        axis: {"state": "not_governing", "atom_ids": [], "reason": "Injected no-governing-scope judgment"}
        for axis in m.SCOPE_DIMENSIONS
    }
    row.update(
        scope_groups=[{"condition_ids": ["c1", "c2"], "dimensions": dimensions}],
        scope_atoms=[],
        source_reviews=[
            {
                "block_id": item["block_id"],
                "state": "irrelevant",
                "condition_ids": ["c1", "c2"],
                "reason": "Injected source relation",
            }
            for item in context[0]["sources"]
        ],
    )
    return row, registry[0], context[0], {b.id: b for b in paper.blocks}


def test_v4_group_closes_two_conditions_without_repeated_atoms():
    row, claim, context, blocks = scope_row()
    checked = m.CurrentClaimReviewV4.model_validate(row)
    m._check_scope(checked, claim, context, blocks)
    assert len(checked.scope_groups) == 1 and checked.scope_atoms == []


@pytest.mark.parametrize("corrupt", ["dimension", "condition", "source"])
def test_missing_scope_obligation_rejected(corrupt):
    row, claim, context, blocks = scope_row()
    if corrupt == "dimension":
        row["scope_groups"][0]["dimensions"].pop(m.SCOPE_DIMENSIONS[0])
    elif corrupt == "condition":
        row["scope_groups"][0]["condition_ids"] = ["c1"]
    else:
        row["source_reviews"].pop()
    with pytest.raises(ValueError):
        m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, blocks)


def test_bound_source_budget_unavailable_is_explicit_not_missing_qualifier():
    paper, registry, required = catalog()
    registry[0]["source_refs"] = [{"source_block_id": "b2", "covered": ["c1"]}]
    context, extras = m._scope_context(
        registry, required, ["b1"], {b.id: b for b in paper.blocks}, paper.markdown, 1
    )
    bound = next(s for s in context[0]["sources"] if s["block_id"] == "b2")
    assert bound["availability"] == "unavailable" and bound["unavailable_reason"] == "budget"
    assert extras == [] and context[0]["complete"] is False


def test_legacy_v3_recheck_is_readable_without_new_completion():
    paper, registry, required = catalog()
    payload = {
        "current_claims": registry,
        "context_id": "mock",
        "window_id": "mock",
        "blocks": [b.model_dump(mode="json") for b in paper.blocks],
    }
    raw = review(payload)
    raw.update(
        schema_version="claim-coverage-validation-v3",
        change_decisions=[],
        observation_decisions=[],
        original_claim_reviews=raw.pop("claim_reviews"),
        observation_links=[],
    )
    raw.pop("reviewed_block_ids")
    raw.pop("new_findings")
    raw.pop("explanation")
    fresh, check, links = m._original_recheck(
        m.CoverageValidationV3.model_validate(copy.deepcopy(raw)),
        required,
        registry,
        [],
        {},
        {b.id: b for b in paper.blocks},
        paper.markdown,
    )
    assert fresh == [] and links == {} and check["completed"] == 0
    assert check["status"] == "legacy_scope_unreviewed"


def missing_row():
    row, claim, context, blocks = scope_row()
    row["findings"] = [
        {
            "kind": "missing_qualifier_or_condition",
            "condition_ids": ["c1"],
            "source_block_ids": ["b2"],
            "restriction": "one 250-label split",
            "material_effect": "Different label budget changes the verification setting.",
            "reason": "Injected governing scope.",
        }
    ]
    payload = {"scope_context": [context], "blocks": [b.model_dump(mode="json") for b in blocks.values()]}
    return with_scope(row, claim, payload), claim, context, blocks


@pytest.mark.parametrize(
    "corrupt",
    [
        "background",
        "provenance",
        "same_alternative",
        "excluded_alternative",
        "assertion_metadata",
        "source_as_carrier",
        "unindexed",
        "duplicate_index",
        "reason_conflict",
    ],
)
def test_materiality_and_indexed_missing_declaration_are_closed(corrupt):
    row, claim, context, blocks = missing_row()
    atom = row["scope_atoms"][0]
    effect = atom["effect"]
    if corrupt in {"background", "provenance"}:
        effect["kind"] = "background_fact" if corrupt == "background" else "provenance_only"
    elif corrupt == "same_alternative":
        effect["alternative_setting"] = effect["source_setting"]
    elif corrupt == "excluded_alternative":
        effect["original_permits_alternative"] = False
    elif corrupt == "assertion_metadata":
        effect.update(assertion_path="/conditions/0/id", assertion_value="c1")
    elif corrupt == "source_as_carrier":
        effect.update(assertion_path="/source_block_id", assertion_value=claim["source_block_id"])
    elif corrupt == "unindexed":
        row["findings"].append(copy.deepcopy(row["findings"][0]))
    elif corrupt == "reason_conflict":
        row["findings"][0]["reason"] = "An independently contradicted decision."
    else:
        second = copy.deepcopy(atom)
        second["id"] = "duplicate_problem"
        row["scope_atoms"].append(second)
    with pytest.raises(ValueError):
        m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, blocks)


def test_missing_scope_passes_once_without_fabricated_preservation():
    row, claim, context, blocks = missing_row()
    m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, blocks)
    assert row["scope_atoms"][0]["finding_index"] == 0 and not row["preserved_qualifiers"]


def test_atom_source_review_must_cover_its_conditions():
    row, claim, context, blocks = missing_row()
    context["sources"] = [s for s in context["sources"] if s["block_id"] != "b2"]
    next(s for s in row["source_reviews"] if s["block_id"] == "b2")["condition_ids"] = ["c2"]
    with pytest.raises(ValueError, match="condition"):
        m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, blocks)


def preserved_row():
    row, claim, context, blocks = scope_row()
    row["preserved_qualifiers"] = [
        {
            "restriction": "Dataset D",
            "source_block_ids": ["b1"],
            "claim_path": "/conditions/0/dataset",
            "claim_value": "D",
        }
    ]
    row["scope_atoms"] = [
        {
            "id": "dataset",
            "dimension": "dataset_population",
            "condition_ids": ["c1"],
            "state": "preserved",
            "restriction": "Dataset D",
            "preserved_indices": [0],
            "finding_index": None,
            "effect": None,
            "reason": "Injected carrier entailment.",
            "sources": [{"block_id": "b1", "start": 0, "end": len(blocks["b1"].text)}],
        }
    ]
    row["source_reviews"][0]["state"] = "considered"
    row["scope_groups"] = [
        {"condition_ids": [key], "dimensions": copy.deepcopy(row["scope_groups"][0]["dimensions"])}
        for key in ("c1", "c2")
    ]
    row["scope_groups"][0]["dimensions"]["dataset_population"].update(state="preserved", atom_ids=["dataset"])
    return row, claim, context, blocks


@pytest.mark.parametrize(
    "corrupt", [None, "foreign_condition", "whole_condition", "wrong_type", "span", "unindexed"]
)
def test_preserved_carriers_are_actual_semantic_leaves_for_affected_conditions(corrupt):
    row, claim, context, blocks = preserved_row()
    if corrupt == "foreign_condition":
        row["scope_atoms"][0]["condition_ids"] = ["c1", "c2"]
    elif corrupt == "whole_condition":
        row["preserved_qualifiers"][0].update(claim_path="/conditions/0", claim_value=claim["conditions"][0])
    elif corrupt == "wrong_type":
        row["preserved_qualifiers"][0]["claim_value"] = 1
    elif corrupt == "span":
        row["scope_atoms"][0]["sources"][0]["end"] += 1
    elif corrupt == "unindexed":
        row["preserved_qualifiers"].append(copy.deepcopy(row["preserved_qualifiers"][0]))
    if corrupt:
        with pytest.raises(ValueError):
            m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, blocks)
    else:
        m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, blocks)
        assert len(row["scope_atoms"]) == 1


def test_low_shared_budget_cannot_clear_old_pending_or_backfill_completion(monkeypatch, tmp_path):
    paper, claims = case()
    claims[0].source_refs = []
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    modules, payloads = [], []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        modules.append(kw["module"])
        payloads.append(p)
        if kw["module"] == "screening.claims.coverage":
            return review(p)
        if kw["module"] == "screening.claims.coverage_validation":
            return validation(p)
        pytest.fail("Unexpected follow-up or external service")

    result = m.review_claim_coverage(
        paper, claims, call=call, output_dir=tmp_path, window_chars=len(paper.blocks[0].text)
    )
    assert (
        result.coverage["budget"]["review_calls"] == 2 and result.coverage["budget"]["validation_calls"] == 2
    )
    assert all(
        w["supplemental_sources"]["characters"] <= len(paper.blocks[0].text)
        for w in result.coverage["windows"]
    )
    assert all(
        p["scope_context"] == payloads[i - 1]["scope_context"]
        for i, p in enumerate(payloads)
        if modules[i].endswith("validation")
    )


def test_candidate_heading_ancestry_requires_exact_span_and_complete_prefix_chain():
    from screening.claim_scope import _source_candidates
    from tests.test_claim_coverage_v2 import projection

    raw = [
        {"id": "h1", "text": "4 Experiments", "kind": "heading"},
        {"id": "intro", "text": "All results use the same governing setting."},
        {"id": "h2", "text": "4.1 Results", "kind": "heading"},
        {"id": "local", "text": "This is the local section introduction."},
        {"id": "target", "text": "The method improves the reported result."},
    ]
    paper, _ = projection(raw)
    blocks = {b.id: b for b in paper.blocks}
    candidates = _source_candidates(blocks, paper.markdown)["target"]
    assert ("intro", "numbered_heading_ancestor_candidate:h1") in candidates
    paper.blocks[2].text = "4.2.1 Results"
    candidates = _source_candidates(blocks, paper.markdown)["target"]
    assert all("ancestor" not in reason for _, reason in candidates)
    assert ("local", "ordered-nearby") in candidates


def test_binding_history_candidates_share_budget_and_full_blocks_only():
    paper, registry, required = catalog()
    blocks = {b.id: b for b in paper.blocks}
    used = []

    def history(remaining, loaded):
        used.append((remaining, loaded))
        return []

    context, extras = m._scope_context(
        registry, required, ["b1"], blocks, paper.markdown, len(blocks["b2"].text), history_loader=history
    )
    context2, extras2 = m._scope_context(
        registry, required, ["b1"], blocks, paper.markdown, len(blocks["b2"].text), already_loaded=extras
    )
    assert context == context2 and extras == extras2 and extras[0]["block"]["text"] == blocks["b2"].text
    assert used[0][0] == len(blocks["b2"].text) and used[0][1] == {"b1"}


@pytest.mark.parametrize("visible", [True, False])
def test_additional_remote_footnote_requires_actual_visible_original_body(visible):
    row, claim, context, blocks = missing_row()
    context["sources"] = [s for s in context["sources"] if s["block_id"] != "b2"]
    if visible:
        m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, blocks)
        assert row["scope_atoms"][0]["sources"][0]["block_id"] == "b2"
    else:
        with pytest.raises(ValueError, match="visible"):
            m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, {"b1": blocks["b1"]})


def test_third_only_target_cannot_be_omitted_or_backfilled_into_first(monkeypatch, tmp_path):
    from tests.test_claim_coverage_v2 import observation, small
    from tests.test_claim_coverage_v2 import review as legacy_review

    paper, claims = small()
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    seen = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        seen.append(p)
        if kw["module"] == "screening.claims.coverage":
            obs = (
                [observation(paper.blocks[1], "merged_conclusions")] if p["window_id"] == "window_002" else []
            )
            return legacy_review(p, obs, legacy_v2=True)
        assert kw["module"] == "screening.claims.coverage_validation"
        raw = validation(p)
        if p["window_id"] == "window_002":
            raw["original_claim_reviews"] = []
        return raw

    result = m.review_claim_coverage(
        paper, claims, call=call, output_dir=tmp_path, window_chars=65, max_followup_calls=0
    )
    window = result.coverage["windows"][1]
    assert window["required_claim_checks"] == [] and window["claim_checks"] == []
    assert len(window["validation_required_original_claim_reviews"]) == 1
    assert window["original_recheck"]["required"] == 1 and window["original_recheck"]["completed"] == 0
    assert result.blocked_claim_ids == ["claim_007"] and not result.coverage["revisions"]
    assert result.coverage["status"] == "partial"


def test_group_cannot_hide_distinct_condition_scope_by_aggregating_one_atom():
    row, claim, context, blocks = preserved_row()
    row["scope_groups"] = [row["scope_groups"][0]]
    row["scope_groups"][0]["condition_ids"] = ["c1", "c2"]
    with pytest.raises(ValueError, match="different governing atoms"):
        m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, blocks)


def test_unavailable_binding_keeps_native_stages_partial_without_invented_missing(monkeypatch, tmp_path):
    from schemas.claim import ClaimLocation, ClaimSourceRef
    from schemas.materials import MaterialBlock

    paper, claims = case()
    first = paper.blocks[0]
    body = "A governing original binding is unavailable under this tiny character budget. " * 2
    block = MaterialBlock(
        id="long_binding",
        text=body,
        loc=ClaimLocation(page=2, char_start=len(first.text) + 2, char_end=len(first.text) + 2 + len(body)),
    )
    paper.blocks = [first, block]
    paper.markdown = first.text + "\n\n" + body
    claims[0].source_refs = [
        ClaimSourceRef(source_block_id=block.id, source_quote=body, loc=block.loc, covered=["c1"])
    ]
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    modules = []

    def call(**kw):
        modules.append(kw["module"])
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        if kw["module"] == "screening.claims.coverage":
            return review(p, state="unresolved")
        assert kw["module"] == "screening.claims.coverage_validation"
        return validation(p)

    result = m.review_claim_coverage(
        paper, claims, call=call, output_dir=tmp_path, window_chars=len(first.text), max_followup_calls=0
    )
    assert modules == ["screening.claims.coverage", "screening.claims.coverage_validation"]
    assert result.coverage["original_claim_reviews_completed"] == 0 and result.coverage["status"] == "partial"
    assert result.blocked_claim_ids == [] and not result.coverage["revisions"]
    window = result.coverage["windows"][0]
    assert (
        window["scope_context"][0]["complete"] is False and window["supplemental_sources"]["characters"] == 0
    )
    assert all(o["kind"] == "uncertain" for o in window["observations"])


def test_mock_scope_groups_share_identical_condition_obligations():
    row, claim, context, blocks = scope_row()
    row = with_scope(
        row,
        claim,
        {"scope_context": [context], "blocks": [b.model_dump(mode="json") for b in blocks.values()]},
    )
    assert len(row["scope_groups"]) == 1 and row["scope_groups"][0]["condition_ids"] == ["c1", "c2"]
    m._check_scope(m.CurrentClaimReviewV4.model_validate(row), claim, context, blocks)
