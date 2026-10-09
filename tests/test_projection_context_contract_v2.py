"""Projection context instructions reflect the existing full-sentence source gate."""

import json

import pytest

from tests import test_execution_projection_v2 as fixtures

offline = fixtures.offline


@pytest.mark.parametrize("context", ["empty", "complete", "trimmed"])
def test_existing_public_projection_gate_requires_complete_sentence(tmp_path, context):
    def change(stage, first, scope, claim, materials):
        if stage != "first":
            return
        reported = first["plans"][0]["targets"][0]["reported"]
        if context == "empty":
            reported["value_context"] = ""
        elif context == "trimmed":
            reported["value_context"] = "MiniSet test, model ExactMatch has accuracy 0.75"

    claim, _materials, result, calls = fixtures.public_verify(tmp_path, change=change)
    assert len(calls) == 2 and not result.evidence
    for request in calls:
        payload = json.loads(request["prompt"].split("\nINPUT_JSON:\n", 1)[-1])
        menu = payload["execution_projection_context"]["request_choices"]
        instruction = menu["target_contract"]["value_context"]
        assert "complete" in instruction and "verbatim" in instruction and "empty" in instruction
    if context == "trimmed":
        assert not any(p.feasibility == "ready" for p in result.plans)
        assert any("uniquely contain the selected scalar sentence" in s for s in result.issues)
    else:
        assert len(result.plans) == 1 and result.plans[0].feasibility == "ready"
        assert result.plans[0].target_conditions == claim.conditions
        assert result.plans[0].y_paper == {"c1": 0.75}
