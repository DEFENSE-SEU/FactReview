"""Explicit mock traces exercise routing, without measuring mathematical model accuracy."""

import copy
import json
from pathlib import Path

import pytest

from assessment import assess_claim
from schemas.materials import PageImage
from tests.test_theory_visual_recheck_v2 import case, offline, recovered
from verification.theory import verify_theory

__all__ = ["offline"]


def partial_case(tmp_path, conditions=1):
    claim, materials, raw = case(tmp_path, conditions=conditions)
    raw["items"][0].update(kind="derivation", direction="support", detail="The trace retains a specific gap.")
    raw["derivations"][0]["trace"]["outcome"] = "partial"
    return claim, materials, raw


def invoke(tmp_path, claim, materials, raw, *, rounds=1, visual=None, after_first=None):
    calls, payloads = [], []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == "verification.theory":
            if after_first:
                after_first()
            return copy.deepcopy(raw)
        if kw["module"] == "verification.theory.notation":
            return {
                "classification": "parser_artifact",
                "explanation": "Explicit mock original-page reading.",
            }
        if kw["module"] == "verification.theory.visual_recheck":
            payload = json.loads(kw["prompt"])
            payloads.append(payload)
            return visual(payload) if visual else recovered(payload)
        raise AssertionError("Unexpected stage " + kw["module"])

    result = verify_theory(
        claim, materials, call=model, visual_recheck_rounds=rounds, output_dir=tmp_path / "audit"
    )
    enriched = claim.model_copy(deep=True)
    enriched.evidence.extend(result.evidence)
    return result, assess_claim(enriched), calls, payloads


def test_partial_proof_routes_once_without_notation_or_upgrading_original(tmp_path):
    claim, materials, raw = partial_case(tmp_path)
    before = copy.deepcopy((claim.model_dump(), materials.model_dump(), raw))
    result, assessed, calls, payloads = invoke(tmp_path, claim, materials, raw)
    assert calls == ["verification.theory", "verification.theory.visual_recheck"]
    assert assessed.status == "supported"
    target = payloads[0]["targets"][0]
    assert target["trigger"] == "partial_proof" and target["notation_confirmation"] is None
    assert target["previous_trace"]["state"] == "validated"
    assert target["previous_trace"]["trace"]["outcome"] == "partial"
    assert result.theory_derivations[0].trace.outcome == "partial"
    assert not result.evidence[0].sufficient and result.evidence[-1].sufficient
    assert (
        json.loads(Path(result.theory_derivations[-1].audit_pointer).read_text())["request"]["targets"]
        == payloads[0]["targets"]
    )
    assert (claim.model_dump(), materials.model_dump(), raw) == before


@pytest.mark.parametrize(
    "mode", ["rounds_zero", "legacy", "invalid", "unable", "no_proof", "wrong_theorem", "healthy_complete"]
)
def test_ineligible_original_records_do_not_trigger(tmp_path, mode):
    claim, materials, raw = partial_case(tmp_path)
    if mode == "legacy":
        raw.pop("schema_version")
        raw.pop("derivations")
    elif mode == "invalid":
        raw["derivations"][0]["trace"]["gaps"][0]["sources"][0]["quote"] = "not in the source"
    elif mode in {"unable", "no_proof"}:
        raw["derivations"][0]["trace"]["outcome"] = "unable"
        if mode == "no_proof":
            raw["items"][0].update(kind="no_proof", direction="flaw")
    elif mode == "wrong_theorem":
        claim.source_quote = claim.text = "Theorem 2. For every real x, x*x is nonnegative."
    elif mode == "healthy_complete":
        raw["items"][0]["fully_supported_conditions"] = ["c1"]
        trace = raw["derivations"][0]["trace"]
        trace.update(
            outcome="completed",
            gaps=[],
            steps=[
                {
                    "id": "s1",
                    "statement": "The square is nonnegative.",
                    "reason": "Explicit fixed proof control.",
                    "assumption_ids": [],
                    "previous_step_ids": [],
                    "sources": [{"block_id": "b2", "quote": materials.blocks[1].text}],
                }
            ],
        )
    result, assessed, calls, _ = invoke(
        tmp_path, claim, materials, raw, rounds=0 if mode == "rounds_zero" else 1
    )
    assert "verification.theory.visual_recheck" not in calls
    if mode != "healthy_complete":
        assert not any(e.sufficient for e in result.evidence)
    else:
        assert assessed.status == "supported"


@pytest.mark.parametrize("mode", ["no_step", "foreign_condition", "unloaded_proof"])
def test_bad_original_support_contract_still_rejects(tmp_path, mode):
    claim, materials, raw = partial_case(tmp_path)
    if mode == "no_step":
        raw["items"][0]["step_quote"] = ""
    elif mode == "foreign_condition":
        raw["items"][0]["covered"] = ["foreign"]
    else:
        materials.blocks[1].loc.section = "Appendix A. Proof of Theorem 1"
    with pytest.raises(ValueError):
        invoke(tmp_path, claim, materials, raw)


@pytest.mark.parametrize(
    "mode", ["empty", "full_false", "partial", "required_unstated", "duplicate", "unknown_target"]
)
def test_new_partial_or_invalid_visual_never_becomes_sufficient(tmp_path, mode):
    claim, materials, raw = partial_case(tmp_path)

    def visual(payload):
        value = recovered(payload)
        if mode == "empty":
            value["items"] = []
        elif mode == "full_false":
            value["items"][0]["fully_supported"] = False
        elif mode in {"partial", "required_unstated"}:
            trace = value["items"][0]["trace"]
            trace.update(
                outcome="partial",
                gaps=[
                    {
                        "at": "goal",
                        "reason": "An unresolved case",
                        "needed": "Proof of the case",
                        "sources": [],
                    }
                ],
            )
            if mode == "required_unstated":
                trace["assumptions"][0]["status"] = "required_unstated"
        elif mode == "duplicate":
            value["items"].append(copy.deepcopy(value["items"][0]))
        else:
            value["items"][0]["target_id"] = "unknown"
        return value

    result, assessed, calls, _ = invoke(tmp_path, claim, materials, raw, visual=visual)
    assert calls.count("verification.theory.visual_recheck") == 1
    assert assessed.status == "unverified" and not any(e.sufficient for e in result.evidence)


def test_two_trigger_types_share_one_batch_and_distinct_audited_identities(tmp_path):
    claim, materials, raw = partial_case(tmp_path, conditions=2)
    raw["items"][0]["covered"] = ["c1"]
    notation = copy.deepcopy(raw["items"][0])
    notation.update(kind="notation", direction="flaw", covered=["c2"])
    raw["items"].append(notation)
    raw["derivations"].append(copy.deepcopy(raw["derivations"][0]))
    raw["derivations"][1]["item_index"] = 1
    _result, assessed, calls, payloads = invoke(tmp_path, claim, materials, raw)
    assert calls.count("verification.theory.visual_recheck") == 1
    targets = payloads[0]["targets"]
    assert {t["trigger"] for t in targets} == {"parser_confirmed", "partial_proof"}
    assert len({(t["condition_id"], t["original_record_index"]) for t in targets}) == 2
    assert all((t["notation_confirmation"] is None) == (t["trigger"] == "partial_proof") for t in targets)
    assert assessed.status == "supported"


@pytest.mark.parametrize("all_missing", [False, True])
def test_initial_missing_proof_page_is_local_and_never_makes_empty_call(tmp_path, all_missing):
    claim, materials, raw = partial_case(tmp_path, conditions=2)
    # b3 is a separately located duplicate of the proof on initially unrendered page 2.
    proof = materials.blocks[1].model_copy(deep=True)
    proof.id = "b3"
    proof.loc.page = 2
    materials.blocks.append(proof)
    materials.pages.append(
        PageImage(page=2, path=str(tmp_path / "missing.png"), width_points=595, height_points=842)
    )
    raw["items"][0]["covered"] = ["c1"]
    second = copy.deepcopy(raw["items"][0])
    second.update(block_id="b3", covered=["c2"])
    raw["items"].append(second)
    raw["derivations"].append(copy.deepcopy(raw["derivations"][0]))
    raw["derivations"][1]["item_index"] = 1
    raw["derivations"][1]["trace"]["gaps"][0]["sources"][0]["block_id"] = "b3"
    if all_missing:
        Path(materials.pages[0].path).unlink()
    result, _assessed, calls, payloads = invoke(tmp_path, claim, materials, raw)
    assert calls.count("verification.theory.visual_recheck") == (0 if all_missing else 1)
    assert any(
        limit.kind == "source_context_unavailable" and limit.condition_ids == ["c2"]
        for limit in result.verification_limitations
    )
    assert not any(e.sufficient and "c2" in e.covered for e in result.evidence)
    if not all_missing:
        assert [t["condition_id"] for t in payloads[0]["targets"]] == ["c1"]
        assert any(e.sufficient and e.covered == ["c1"] for e in result.evidence)


@pytest.mark.parametrize("mode", ["delete", "metadata", "replace", "claim", "block"])
def test_source_mutation_after_initial_freeze_rejects_before_new_call(tmp_path, mode):
    claim, materials, raw = partial_case(tmp_path)

    def mutate():
        if mode == "delete":
            Path(materials.pages[0].path).unlink()
        elif mode == "metadata":
            materials.pages[0].dpi += 1
        elif mode == "replace":
            Path(materials.pages[0].path).write_bytes(b"changed pixels")
        elif mode == "claim":
            claim.text += " changed"
        else:
            materials.blocks[1].text += " changed"

    with pytest.raises(ValueError, match=r"changed|identity"):
        invoke(tmp_path, claim, materials, raw, after_first=mutate)


def test_superseded_main_partial_cannot_trigger_recovery(tmp_path):
    claim, materials, raw = partial_case(tmp_path)
    appendix = materials.blocks[1].model_copy(deep=True)
    appendix.id = "appendix"
    appendix.loc.section = "Appendix A. Proof of Theorem 1"
    materials.blocks.append(appendix)
    raw["appendix_block_ids"] = ["appendix"]
    calls = []

    def model(**kw):
        calls.append(kw["module"])
        if kw["module"] == "verification.theory":
            return copy.deepcopy(raw)
        assert kw["module"] == "verification.theory.appendix"
        return {
            "schema_version": "theory-derivation-v1",
            "appendix_block_ids": [],
            "items": [],
            "derivations": [],
        }

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert calls == ["verification.theory", "verification.theory.appendix"]
    assert result.theory_derivations[0].state == "validated"
    assert result.theory_derivations[0].trace.outcome == "partial"
    assert not result.theory_derivations[0].adopted and not result.evidence


@pytest.mark.parametrize("disposition", ["outside_scope", "closed_disproof"])
def test_new_visual_flaw_still_requires_independent_concern_scope(tmp_path, disposition):
    claim, materials, raw = partial_case(tmp_path)
    calls = []

    def model(**kw):
        calls.append(kw["module"])
        payload = json.loads(kw["prompt"])
        if kw["module"] == "verification.theory":
            return copy.deepcopy(raw)
        if kw["module"] == "verification.theory.visual_recheck":
            value = recovered(payload)
            value["items"][0].update(
                direction="flaw",
                detail="Explicit mock purported counterexample; semantics are supplied by the test.",
            )
            return value
        assert kw["module"] == "verification.theory.concern_scope"
        return {
            "schema_version": "theory-concern-v1",
            "items": [
                {
                    **pair,
                    "disposition": disposition,
                    "target_sources": [{"block_id": "b1", "quote": "For every real x"}],
                    "trace_step_ids": ["s1"],
                    "trace_gap_indices": [],
                    "scope_reason": "Explicit controlled mock applicability judgment.",
                    "resolution": "Independent mocked decision, not mathematical accuracy evidence.",
                }
                for pair in payload["allowed_pairs"]
            ],
        }

    result = verify_theory(claim, materials, call=model, output_dir=tmp_path / "audit")
    assert calls == [
        "verification.theory",
        "verification.theory.visual_recheck",
        "verification.theory.concern_scope",
    ]
    enriched = claim.model_copy(deep=True)
    enriched.evidence.extend(result.evidence)
    assert assess_claim(enriched).status == ("flawed" if disposition == "closed_disproof" else "unverified")
    assert result.theory_derivations[-1].concern_reviews[0].state == "validated"


def test_healthy_other_item_excludes_partial_target_for_same_condition(tmp_path):
    claim, materials, raw = partial_case(tmp_path)
    healthy = copy.deepcopy(raw["items"][0])
    healthy["fully_supported_conditions"] = ["c1"]
    raw["items"].append(healthy)
    trace = copy.deepcopy(raw["derivations"][0]["trace"])
    trace.update(
        outcome="completed",
        gaps=[],
        steps=[
            {
                "id": "s1",
                "statement": "Square nonnegative.",
                "reason": "Explicit complete mock control.",
                "assumption_ids": [],
                "previous_step_ids": [],
                "sources": [{"block_id": "b2", "quote": materials.blocks[1].text}],
            }
        ],
    )
    raw["derivations"].append({"item_index": 1, "trace": trace})
    result, assessed, calls, _ = invoke(tmp_path, claim, materials, raw)
    assert calls == ["verification.theory"] and assessed.status == "supported"
    assert result.theory_derivations[0].trace.outcome == "partial"
