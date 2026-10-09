"""Original v4 obligations and finite composition; every external boundary is mocked."""

import copy

import pytest

from schemas.claim import ClaimSourceRef, Condition, SemanticPredictionProjection
from tests.test_execution_choice_finite_language_v2 import actual006, offline  # noqa: F401
from tests.test_execution_projection_semantics_v2 import confirm
from verification.execution_projection import ProjectionError, entry_recipe, field_inventory
from verification.execution_projection_choices import build_choice_registry
from verification.execution_projection_semantics import (
    _interpret_text,
    _require_fixed_sample_count,
    validate_semantic_obligations,
)
from verification.experiment_catalog import build_catalog

TEXT = (
    "On MiniSet test, model ExactMatch has accuracy 0.75, computed as a fraction by exact equality "
    "of each prediction and label, over four examples, with no ranking against other models."
)
CONDITION = {
    "id": "c1",
    "dataset": "MiniSet test",
    "metric": "accuracy",
    "settings": {
        "model": "ExactMatch",
        "accuracy": 0.75,
        "examples": 4,
        "computation": "fraction computed by exact equality of each prediction and label",
        "qualifiers": [
            "fixed predictions",
            "no repeated-run uncertainty claimed",
            "no population-performance conclusion claimed",
            "no ranking against other models",
        ],
    },
    "description": "Measured MiniSet test accuracy for ExactMatch on four fixed examples.",
}


def inputs(tmp_path):
    claim, materials = actual006(tmp_path)
    claim.text = TEXT
    claim.conditions = [Condition.model_validate(CONDITION)]
    return claim, materials


def validate(claim, materials, candidate, proposal=None):
    proposal = proposal or SemanticPredictionProjection.model_validate(candidate["proposal"])
    return validate_semantic_obligations(
        claim,
        claim.conditions[0],
        materials,
        proposal,
        confirm(proposal),
        entry_recipe(materials, candidate["entry_script"], candidate["config_path"], candidate["data_path"]),
        build_catalog(claim, materials),
        candidate["selector"]["number_id"],
        candidate["value"],
    )


def test_original_v4_complete_claim_and_eleven_leaves_survive_builder_and_consumer(tmp_path):
    claim, materials = inputs(tmp_path)
    before = copy.deepcopy((claim.model_dump(), materials.model_dump()))
    registry = build_choice_registry(claim, materials)
    assert len(registry["candidates"]) == 1, registry["unavailable"]
    candidate = next(iter(registry["candidates"].values()))
    assert candidate["condition"] == CONDITION
    assert len(field_inventory(claim.conditions[0])) == 11
    assert candidate["proposal"]["claim_bindings"][0]["end"] == len(TEXT)
    assert len(validate(claim, materials, candidate)["field_consumption"]) == 11
    assert before == (claim.model_dump(), materials.model_dump())


@pytest.mark.parametrize("witness", ["field", "whole", "none", "other_condition", "other_dataset"])
def test_fixed_population_requires_explicit_count_from_this_original_condition(tmp_path, witness):
    claim, materials = inputs(tmp_path)
    claim.conditions[0].description = "Measured MiniSet test accuracy for ExactMatch."
    if witness != "field":
        del claim.conditions[0].settings["examples"]
    if witness != "whole":
        claim.text = TEXT.replace(", over four examples", "")
    if witness == "other_condition":
        claim.conditions.append(
            Condition(id="c2", dataset="Other", metric="accuracy", settings={"examples": 4})
        )
    if witness == "other_dataset":
        claim.text = TEXT.replace("MiniSet test", "Other test")
    registry = build_choice_registry(claim, materials)
    candidates = [r for r in registry["candidates"].values() if r["condition_id"] == "c1"]
    assert bool(candidates) is (witness in {"field", "whole"})


def test_consumer_rejects_qualitative_count_borrowing_despite_confirmed_scope(tmp_path):
    claim, materials = inputs(tmp_path)
    candidate = next(iter(build_choice_registry(claim, materials)["candidates"].values()))
    proposal = SemanticPredictionProjection.model_validate(candidate["proposal"])
    del claim.conditions[0].settings["examples"]
    claim.conditions[0].description = "Measured MiniSet test accuracy for ExactMatch."
    claim.text = TEXT.replace(", over four examples", "")
    sample_id = next(a.id for a in proposal.atoms if a.kind == "sample_scope")
    proposal.field_bindings = [r for r in proposal.field_bindings if r.path != "/settings/examples"]
    for r in proposal.field_bindings:
        if r.path == "/description":
            r.atom_ids.remove(sample_id)
    proposal.claim_bindings[0].end = len(claim.text)
    proposal.claim_bindings[0].atom_ids.remove(sample_id)
    with pytest.raises(ProjectionError, match="explicit original sample count"):
        validate(claim, materials, candidate, proposal)
    # Identity names must keep their dispatch role; a numerical-looking name
    # cannot supply a count witness to the shared builder/consumer check.
    cfg = {"dataset": "MiniSet", "metric": "accuracy", "settings": {"split": "test", "model": "ExactMatch"}}
    for field in ("dataset", "metric"):
        changed = claim.model_copy(deep=True)
        setattr(changed.conditions[0], field, "four fixed examples")
        changed.conditions[0].description = ""
        bound = copy.deepcopy(cfg)
        if field == "dataset":
            bound[field] = "four fixed examples"
        changed.text = f"On {bound['dataset']} test, model ExactMatch has accuracy 0.75."
        with pytest.raises(ProjectionError, match="explicit original sample count"):
            _require_fixed_sample_count(
                changed,
                changed.conditions[0],
                bound,
                Condition(id="c1", **bound),
                0.75,
                4,
                claim_texts=[changed.text],
            )


def test_parameterized_composition_and_closed_adjuncts():
    cfg = {"dataset": "D[2].+", "metric": "accuracy", "settings": {"split": "held-out", "model": "R+G(v2)"}}
    runtime = Condition(id="c1", **cfg)
    head = "On D[2].+ held-out, model R+G(v2) has accuracy 0.5"
    definition = ", computed as a fraction by exact equality of each prediction and label"
    sample = ", over seven examples"
    boundary = ", with no ranking against other models"
    expected = {"dataset", "runtime:model", "value", "definition", "sample", "boundary:cross_model_ranking"}
    for tail in (
        definition + sample + boundary,
        sample + boundary + definition,
        boundary + definition + sample,
    ):
        assert _interpret_text(head + tail, cfg, runtime, 0.5, 7) == expected
    for tail in (
        definition + sample + sample,
        definition + sample + boundary + boundary,
        definition + definition + sample,
        definition + ", over eight examples" + boundary,
        definition + ", over seven train examples" + boundary,
        definition + sample + ", with no ranking against other models except R",
        definition + sample + ", with population-performance conclusion claimed",
        definition.replace("exact equality", "weighted exact equality") + sample,
        definition.replace("exact equality", "top-k exact equality") + sample,
        definition + sample + ", for arbitrary unseen data",
    ):
        assert _interpret_text(head + tail, cfg, runtime, 0.5, 7) is None, tail
    for phrase in (
        "not fixed predictions",
        "fixed weighted predictions",
        "fixed train predictions",
        "fixed predictions except errors",
    ):
        assert _interpret_text(phrase, cfg, runtime, 0.5, 7) is None


def test_fixed_population_still_needs_original_population_source(tmp_path):
    claim, materials = inputs(tmp_path)
    # Exclude only the population sentence from authorized ranges. The other
    # obligations retain their original complete quotes and all blocks stay intact.
    claim.source_quote = claim.source_quote.split(" Three of the four")[0]
    claim.source_refs.append(
        ClaimSourceRef(
            source_block_id=claim.source_block_id,
            source_quote="No repeated-run uncertainty or population-performance conclusion is claimed.",
            loc=claim.loc,
            covered=["c1"],
        )
    )
    registry = build_choice_registry(claim, materials)
    assert not registry["candidates"]
    assert any(
        "Obligation sample lacks one complete authorized source" in r["reason"]
        for r in registry["unavailable"]
    )
