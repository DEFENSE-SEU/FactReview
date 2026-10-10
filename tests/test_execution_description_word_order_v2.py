"""Closed scope descriptions retain every original execution obligation."""

import copy

from tests.test_execution_choice_finite_language_v2 import offline  # noqa: F401
from tests.test_execution_fixed_population_composition_v2 import inputs, validate
from verification.execution_projection import field_inventory
from verification.execution_projection_choices import build_choice_registry
from verification.execution_projection_semantics import _description

CFG = {
    "dataset": "MiniSet",
    "metric": "accuracy",
    "settings": {"split": "test", "model": "ExactMatch"},
}
DESCRIPTION = "ExactMatch measured accuracy on the MiniSet test set."


def test_measured_scope_word_order_retains_the_original_meaning():
    expected = _description("Measured ExactMatch accuracy on MiniSet test.", CFG, 4)
    assert expected == {"dataset", "definition", "runtime:model"}
    for text in (
        DESCRIPTION,
        "ExactMatch reported accuracy on MiniSet test.",
        "ExactMatch reported exact-match accuracy on the MiniSet test set.",
    ):
        assert _description(text, CFG, 4) == expected


def test_description_cannot_hide_changed_scope_or_additional_assertions():
    for text in (
        DESCRIPTION.replace("ExactMatch", "OtherModel"),
        DESCRIPTION.replace("MiniSet", "OtherSet"),
        DESCRIPTION.replace("test set", "train set"),
        DESCRIPTION.replace("accuracy", "recall"),
        DESCRIPTION.replace("MiniSet", "balanced MiniSet"),
        DESCRIPTION.replace("test set", "unseen test set"),
        DESCRIPTION.replace("set.", "set after tuning."),
    ):
        assert _description(text, CFG, 4) is None, text
    no_split = copy.deepcopy(CFG)
    del no_split["settings"]["split"]
    assert _description("ExactMatch measured accuracy on the MiniSet set.", no_split, 4) is None


def test_description_requires_an_existing_exact_sample_count():
    text = DESCRIPTION.rstrip(".") + " over four fixed test predictions."
    assert _description(text, CFG, 4) == {"dataset", "definition", "runtime:model", "sample"}
    assert _description(text, CFG, 3) is None
    assert _description(text, CFG) is None
    assert _description(DESCRIPTION.rstrip(".") + " over fixed predictions.", CFG, 4) is None
    assert "sample" not in _description(DESCRIPTION, CFG, 4)


def test_builder_and_final_consumer_keep_the_complete_original_fields(tmp_path):
    claim, materials = inputs(tmp_path)
    claim.conditions[0].description = DESCRIPTION
    before = copy.deepcopy((claim.model_dump(), materials.model_dump()))
    registry = build_choice_registry(claim, materials)
    assert len(registry["candidates"]) == 1, registry["unavailable"]
    candidate = next(iter(registry["candidates"].values()))
    assert candidate["condition"] == claim.conditions[0].model_dump(mode="json")
    consumed = validate(claim, materials, candidate)["field_consumption"]
    assert len(consumed) == len(field_inventory(claim.conditions[0]))
    assert "/description" in consumed
    assert before == (claim.model_dump(), materials.model_dump())
