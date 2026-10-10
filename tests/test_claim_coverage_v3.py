"""Necessary explicit-v3 controls. All scientific judgments are injected mocks."""

import copy
import json

import pytest

from schemas.materials import MaterialBlock
from screening import claim_coverage as m
from tests.test_claim_coverage_v2 import (
    CFG,
    action,
    candidate,
    extracted,
    followup,
    projection,
    validation,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(m, "resolve_llm_config", lambda: CFG)
    monkeypatch.setattr(m, "llm_json", lambda **_: pytest.fail("Unexpected external request"))


def case():
    blocks = [
        {"id": "b1", "text": "The method reaches accuracy 90 and uses batch size 16.", "loc": {"page": 1}},
        {"id": "b2", "text": "The accuracy evaluation uses one 250-label split.", "loc": {"page": 2}},
    ]
    b = MaterialBlock.model_validate(blocks[0])
    raw = {
        **candidate(b),
        "id": "claim_001",
        "conditions": [
            {"id": "c1", "dataset": "D", "metric": "accuracy", "description": "Accuracy is 90."},
            {"id": "c2", "description": "Batch size is 16."},
        ],
    }
    return projection(blocks, [raw])


def review(payload, *, groups=1, findings=(), state="resolved"):
    c = payload["current_claims"][0]
    all_ids = [x["id"] for x in c["conditions"]]
    partitions = [all_ids] if groups == 1 else [[i] for i in all_ids]
    raw = {
        "schema_version": "claim-coverage-v3",
        "context_id": payload["context_id"],
        "window_id": payload["window_id"],
        "reviewed_block_ids": [b["id"] for b in payload["blocks"]],
        "claim_reviews": [
            {
                "claim_id": c["claim_id"],
                "claim_digest": c["digest"],
                "state": state,
                "assertion_groups": [
                    {
                        "proposition": "Explicit mock assertion " + str(i),
                        "condition_ids": ids,
                        "source_block_ids": ["b1"],
                    }
                    for i, ids in enumerate(partitions)
                ],
                "findings": list(findings),
                "preserved_qualifiers": [],
                "source_block_ids": ["b1"],
                "reason": "Explicit source-grounded mock judgment.",
            }
        ],
        "new_findings": [],
        "explanation": "Offline explicit-v3 review.",
    }
    if "scope_context" in payload:
        from tests.claim_scope_mock import with_scope

        raw["schema_version"] = "claim-coverage-v4"
        # A no-required-claim window is used by explicit untargeted finding controls.
        if any(s["claim_id"] == c["claim_id"] for s in payload["scope_context"]):
            raw["claim_reviews"] = [with_scope(raw["claim_reviews"][0], c, payload)]
        else:
            raw["claim_reviews"] = []
    return raw


QUALIFIER = {
    "kind": "missing_qualifier_or_condition",
    "condition_ids": ["c1"],
    "source_block_ids": ["b2"],
    "restriction": "one 250-label split",
    "material_effect": "Without the restriction the result could mean a different label budget.",
    "reason": "Original b2 governs this accuracy assertion.",
}


@pytest.mark.parametrize("with_qualifier", [False, True])
def test_explicit_groups_lower_once_and_split_through_three_stages(tmp_path, with_qualifier):
    paper, claims = case()
    before = [c.model_dump(mode="json") for c in claims]
    calls, first = [], []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        calls.append((kw["module"], p))
        if kw["module"] == "screening.claims.coverage":
            r = review(p, groups=2, findings=[QUALIFIER] if with_qualifier else [])
            first.append(copy.deepcopy(r))
            return r
        if kw["module"] == "screening.claims.coverage_followup":
            kinds = [o["kind"] for o in p["observations"]]
            assert kinds.count("merged_conclusions") == 1
            assert kinds.count("missing_qualifier_or_condition") == int(with_qualifier)
            merge = next(o for o in p["observations"] if o["kind"] == "merged_conclusions")
            assert (
                json.loads(merge["reason"])["assertion_groups"]
                == first[0]["claim_reviews"][0]["assertion_groups"]
            )
            props = []
            for i, text in enumerate(["The method reaches accuracy 90.", "The method uses batch size 16."]):
                raw = extracted(before[0])
                raw.update(text=text, conditions=[raw["conditions"][i]])
                if i == 0 and with_qualifier:
                    raw["conditions"][0]["settings"] = {"labels": 250, "split": "single split"}
                    raw["source_refs"] = [
                        {"source_block_id": "b2", "source_quote": paper.blocks[1].text, "covered": ["c1"]}
                    ]
                props.append(raw)
            return followup(p, [action(p, p["observations"], "split", props)])
        return validation(p)

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert len(result.claims) == 2 and not result.blocked_claim_ids
    assert [name for name, _ in calls] == [
        "screening.claims.coverage",
        "screening.claims.coverage_followup",
        "screening.claims.coverage_validation",
    ]
    saved = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert saved["attempts"][0]["response"] == first[0]
    lowering = saved["attempts"][0]["v3_lowering"]
    assert len(lowering["mappings"][0]["observation_ids"]) == 1 + int(with_qualifier)
    assert len(saved["changes"]) == 1 and saved["changes"][0]["action"] == "split"
    assert [c.model_dump(mode="json") for c in claims] == before


def test_legacy_v2_missing_merge_stays_unreviewed(tmp_path):
    paper, claims = case()
    from tests.test_claim_coverage_v2 import review as old_review

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        r = old_review(p, [], legacy_v2=True)
        row = r["claim_checks"][0]
        row.update(
            atomicity="independent_conclusions",
            assertion_groups=[
                {"proposition": "First independent assertion", "condition_ids": ["c1"]},
                {"proposition": "Second independent assertion", "condition_ids": ["c2"]},
            ],
        )
        return r

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    checked = result.coverage["windows"][0]["claim_checks"][0]
    assert checked["state"] == "unreviewed" and "merged observation" in checked["error"]
    assert result.coverage["windows"][0]["observations"] == []
    assert not result.coverage["revisions"]


def test_missing_needs_uses_existing_revision_route(tmp_path):
    paper, claims = case()
    finding = {
        "kind": "missing_needs",
        "condition_ids": ["c1"],
        "source_block_ids": ["b1"],
        "needs": ["Experiments"],
        "reason": "Accuracy requires experimental evidence.",
    }
    modules = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        modules.append(kw["module"])
        if kw["module"] == "screening.claims.coverage":
            return review(p, findings=[finding])
        if kw["module"] == "screening.claims.coverage_followup":
            raw = extracted(claims[0].model_dump(mode="json"))
            raw["needs"].append("Experiments")
            return followup(p, [action(p, p["observations"], "revise", [raw])])
        return validation(p)

    result = m.review_claim_coverage(paper, claims, call=call, output_dir=tmp_path)
    assert result.claims[0].needs == ["Code", "Experiments"] and claims[0].needs == ["Code"]
    assert len(modules) == 3 and not result.blocked_claim_ids
    assert result.coverage["windows"][0]["observations"][0]["kind"] == "missing_needs"


def test_unresolved_never_derives_merge_qualifier_or_block(tmp_path):
    paper, claims = case()
    finding = {
        "kind": "uncertain",
        "condition_ids": ["c1"],
        "source_block_ids": ["b2"],
        "reason": "The precise governing relationship remains unclear.",
    }

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        return review(p, groups=2, state="unresolved", findings=[finding])

    result = m.review_claim_coverage(
        paper, claims, call=call, output_dir=tmp_path, max_followup_calls=0, max_validation_calls=0
    )
    rows = result.coverage["windows"][0]["observations"]
    assert len(rows) == 1 and rows[0]["kind"] == "uncertain"
    assert json.loads(rows[0]["reason"])["findings"] == [finding]
    assert not result.blocked_claim_ids and result.coverage["claim_checks_unreviewed"] == 1
    assert not result.coverage["revisions"]


def test_historical_source_rebind_is_bounded_and_not_reviewed(tmp_path):
    paper, claims = case()
    blocks = {b.id: b for b in paper.blocks}
    frozen = {b.id: b.model_dump(mode="json") for b in paper.blocks}
    obs = m.CoverageObservation(
        id="old",
        kind="missing_qualifier_or_condition",
        target_claim_id="claim_001",
        sources=[{"block_id": "b2", "quote": blocks["b2"].text}],
        reason="Original required scope.",
    )
    binding = m._observation_bindings(obs, blocks, paper.markdown, "materials-hash", {})
    old = {
        **obs.model_dump(mode="json"),
        "state": "unresolved",
        "window_id": "old_window",
        "source_bindings": binding,
    }
    loaded, unavailable = m._supplemental_sources(
        [old], {"claim_001"}, ["b1"], blocks, frozen, paper.markdown, "materials-hash", {}, 1000
    )
    assert [x["block"]["id"] for x in loaded] == ["b2"] and not unavailable
    assert loaded[0]["bindings"][0]["history_window_id"] == "old_window"
    assert old["state"] == "unresolved"
    unassigned = copy.deepcopy(old)
    unassigned.update(target_claim_id=None, kind="missing_conclusion")
    background, _ = m._supplemental_sources(
        [unassigned], set(), ["b1"], blocks, frozen, paper.markdown, "materials-hash", {}, 1000
    )
    assert background[0]["bindings"][0]["association"] == "unassigned_background"
    assert unassigned["target_claim_id"] is None and unassigned["state"] == "unresolved"
    for mode in ("invalid", "unrelated", "quote", "location", "changed", "budget"):
        altered, changed = copy.deepcopy(old), copy.deepcopy(blocks)
        if mode == "invalid":
            altered["state"] = "invalid"
        if mode == "unrelated":
            altered["target_claim_id"] = "other"
        if mode == "quote":
            altered["sources"][0]["quote"] = "Not original."
        if mode == "location":
            altered["source_bindings"][0]["loc"]["page"] = 99
        if mode == "changed":
            changed["b2"].text += " Changed."
        got, _ = m._supplemental_sources(
            [altered],
            {"claim_001"},
            ["b1"],
            changed,
            frozen,
            paper.markdown,
            "materials-hash",
            {},
            1 if mode == "budget" else 1000,
        )
        assert got == []
    payload = {
        "window_id": "w",
        "context_id": "ctx",
        "current_claims": m._claim_registry(claims),
        "blocks": [frozen["b1"]],
    }
    raw = review(payload)
    raw["reviewed_block_ids"] = []
    parsed = m.CoverageReviewV3.model_validate(raw)
    lowered, audit = m._lower_v3(
        parsed,
        [
            {
                "claim_id": "claim_001",
                "digest": payload["current_claims"][0]["digest"],
                "source_block_ids": ["b1"],
            }
        ],
        payload["current_claims"],
        blocks,
        paper.markdown,
        {"b2"},
    )
    assert lowered.reviewed_block_ids == [] and not lowered.observations
    assert audit["errors"]  # Loading b2 did not authorize the unreviewed b1.


def test_qualifier_carrier_and_duplicate_claim_identity_are_exact():
    paper, claims = case()
    registry = m._claim_registry(claims)
    required = [{"claim_id": "claim_001", "digest": registry[0]["digest"], "source_block_ids": ["b1"]}]
    blocks = {b.id: b for b in paper.blocks}
    payload = {
        "window_id": "w",
        "context_id": "ctx",
        "current_claims": registry,
        "blocks": [b.model_dump(mode="json") for b in paper.blocks],
    }
    raw = review(payload)
    raw["claim_reviews"][0]["preserved_qualifiers"] = [
        {
            "source_block_ids": ["b1"],
            "restriction": "Dataset D",
            "claim_path": "/conditions/0/dataset",
            "claim_value": "D",
        }
    ]

    def lower(value):
        return m._lower_v3(
            m.CoverageReviewV3.model_validate(value), required, registry, blocks, paper.markdown, set()
        )

    normalized, audit = lower(raw)
    assert len(normalized.claim_checks) == 1 and not audit["errors"]
    for mode in ("value", "source_path", "duplicate"):
        bad = copy.deepcopy(raw)
        if mode == "value":
            bad["claim_reviews"][0]["preserved_qualifiers"][0]["claim_value"] = "Other"
        if mode == "source_path":
            bad["claim_reviews"][0]["preserved_qualifiers"][0]["claim_path"] = "/source_block_id"
        if mode == "duplicate":
            duplicate = copy.deepcopy(bad["claim_reviews"][0])
            duplicate["claim_id"] = " claim_001 "
            bad["claim_reviews"].append(duplicate)
        normalized, audit = lower(bad)
        assert not normalized.claim_checks and not normalized.observations and audit["errors"]


@pytest.mark.parametrize("legacy_second", [False, True])
def test_historical_background_public_context_and_version_boundary(tmp_path, legacy_second):
    paper, claims = case()
    paper.blocks.reverse()
    first_response = []

    def call(**kw):
        p = json.loads(kw["prompt"].split("DATA_JSON:\n", 1)[1])
        if p["window_id"] == "window_001":
            r = review(p)
            r["claim_reviews"] = []
            r["new_findings"] = [
                {
                    "kind": "missing_conclusion",
                    "source_block_ids": ["b2"],
                    "reason": "Original unassigned source fact.",
                }
            ]
            first_response.append(copy.deepcopy(r))
            return r
        assert p["supplemental_sources"][0]["block"] == paper.blocks[0].model_dump(mode="json")
        assert p["supplemental_sources"][0]["bindings"][0]["association"] == "unassigned_background"
        if legacy_second:
            from tests.test_claim_coverage_v2 import review as old_review

            r = old_review(
                p,
                [
                    {
                        "id": "foreign",
                        "kind": "missing_qualifier_or_condition",
                        "target_claim_id": "claim_001",
                        "sources": [{"block_id": "b2"}],
                        "reason": "Source outside original v2 window.",
                    }
                ],
                legacy_v2=True,
            )
            return r
        return review(p, findings=[QUALIFIER])

    result = m.review_claim_coverage(
        paper,
        claims,
        call=call,
        output_dir=tmp_path,
        window_chars=60,
        max_followup_calls=0,
        max_validation_calls=0,
    )
    old, current = result.coverage["windows"]
    assert old["observations"][0]["target_claim_id"] is None
    assert old["observations"][0]["state"] == "unresolved"
    assert current["reviewed_block_ids"] == ["b1"]
    assert current["supplemental_sources"]["loaded_block_ids"] == ["b2"]
    assert current["observations"][0]["state"] == ("invalid" if legacy_second else "unresolved")
    assert result.blocked_claim_ids == ([] if legacy_second else ["claim_001"])
    saved = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert saved["attempts"][0]["response"] == first_response[0]
    if not legacy_second:
        assert saved["attempts"][1]["v3_lowering"]["consumed_supplemental_ids"] == ["b2"]
