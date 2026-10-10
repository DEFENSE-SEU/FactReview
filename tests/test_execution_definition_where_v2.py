"""Regression for the source-bound v7 definition clause; all services mocked."""

import copy

import pytest

from llm.client import LLMConfig
from tests.test_execution_choice_finite_language_v2 import CFG, RUNTIME, actual006
from verification.execution_projection import field_inventory
from verification.execution_projection_choices import build_choice_registry
from verification.execution_projection_semantics import _interpret_text


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External/model/process boundary must be explicitly mocked")

    for target in (
        "requests.sessions.Session.request",
        "httpx.Client.send",
        "httpx.AsyncClient.send",
        "subprocess.run",
        "subprocess.Popen",
        "screening.checks.llm_json",
    ):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr(
        "screening.checks.resolve_llm_config",
        lambda: LLMConfig(provider="mock", model="mock", base_url=None, api_key=None),
    )


def test_where_definition_retains_every_structural_choice_obligation(tmp_path):
    claim, materials = actual006(tmp_path)
    # Exact text/condition meanings of native v7 claim008; original run is untouched.
    claim.text = (
        "On MiniSet test, model ExactMatch has accuracy 0.75, where accuracy is a fraction "
        "computed by exact equality of each prediction and label, over four fixed test "
        "predictions, with no repeated-run uncertainty or population-performance conclusion claimed."
    )
    claim.conditions[0].settings = {
        "model": "ExactMatch",
        "accuracy": 0.75,
        "examples": 4,
        "computation": "fraction computed by exact equality of each prediction and label",
        "uncertainty": "no repeated-run uncertainty claimed",
        "scope": "no population-performance conclusion claimed",
        "ranking": "no ranking against other models",
    }
    claim.conditions[0].description = (
        "Reported MiniSet test accuracy for ExactMatch on four fixed test predictions."
    )
    before = copy.deepcopy((claim.model_dump(), materials.model_dump()))
    registry = build_choice_registry(claim, materials)
    assert len(registry["candidates"]) == 1, registry["unavailable"]
    candidate = next(iter(registry["candidates"].values()))
    assert candidate["condition"] == claim.conditions[0].model_dump(mode="json")
    assert {b["path"] for b in candidate["proposal"]["field_bindings"]} == set(
        field_inventory(claim.conditions[0])
    )
    assert candidate["proposal"]["claim_bindings"][0]["start"] == 0
    assert candidate["proposal"]["claim_bindings"][0]["end"] == len(claim.text)
    assert before == (claim.model_dump(), materials.model_dump())


def test_where_clause_cannot_drop_changed_definition_or_unknown_scope():
    head = "On MiniSet test, model ExactMatch has accuracy 0.75, where "
    for tail in (
        "accuracy is a fraction computed by weighted exact equality of each prediction and label",
        "accuracy is a fraction computed by exact equality of each prediction and label except hard examples",
        "model ExactMatch was trained on all available labels",
        "accuracy is a fraction computed by exact equality of each prediction and label, "
        "computed by exact equality of each prediction and label",
    ):
        assert _interpret_text(head + tail + ".", CFG, RUNTIME, 0.75, 4) is None, tail
