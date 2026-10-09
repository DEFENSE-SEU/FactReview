"""Rejected joint bindings stay operational limitations with unchanged evidence."""

import copy

import pytest

from assessment import assess_claim
from review.report.advice import advice_input, validate_items
from schemas.claim import AdviceItem
from tests.test_experiment_joint_sources_v2 import offline, paper, review_for, run

__all__ = ["offline"]


@pytest.mark.parametrize("failure", ["missing_use", "unused_role", "bad_member"])
def test_rejected_joint_binding_is_condition_scoped_system_limitation(tmp_path, failure):
    inputs = paper(tmp_path)
    review = review_for(inputs[4])
    if failure == "missing_use":
        review["items"][0]["source_uses"].pop()
    elif failure == "unused_role":
        review["items"][0]["source_uses"][0]["roles"].append("metric_definition")
    else:
        inputs[2].additional_sources[0].quote = "This quote is absent from the supplied source."
    before = copy.deepcopy((inputs[0].model_dump(), inputs[1].model_dump(), review))
    result, _ = run(inputs, review)
    assert not any(e.sufficient for e in result.evidence)
    assert not result.questions
    assert len(result.verification_limitations) == 1
    limitation = result.verification_limitations[0]
    assert limitation.claim_id == inputs[0].id
    assert limitation.condition_ids == ["c1"]
    assert limitation.stage == "Experiments"
    assert limitation.kind == "evidence_validation_failed"
    assert limitation.responsibility == "system"
    assert "Joint candidate 0" in limitation.reason
    assert (inputs[0].model_dump(), inputs[1].model_dump(), review) == before


@pytest.mark.parametrize("complete", [True, False])
def test_healthy_or_semantically_partial_joint_has_no_protocol_limitation(tmp_path, complete):
    inputs = paper(tmp_path)
    review = review_for(inputs[4])
    review["items"][0]["full_support"] = complete
    result, _ = run(inputs, review)
    assert not result.verification_limitations
    assert len(result.evidence) == 1
    assert result.evidence[0].sufficient is complete


def test_advice_for_failed_binding_requires_operator_followup(tmp_path):
    inputs = paper(tmp_path)
    review = review_for(inputs[4])
    review["items"][0]["source_uses"].pop()
    result, _ = run(inputs, review)
    enriched = inputs[0].model_copy(deep=True)
    enriched.evidence.extend(result.evidence)
    enriched.verification_limitations.extend(result.verification_limitations)
    assessed = assess_claim(enriched)
    assert assessed.status == "unverified"
    data = advice_input(assessed, [])
    assert "/verification_limitations/0" in data["basis"]
    item = AdviceItem(
        action="verification_followup",
        condition_ids=["c1"],
        basis_refs=["/coverage_gaps/c1", "/verification_limitations/0"],
        text="Repair the rejected source binding before using this condition in review.",
    )
    validate_items(assessed, data, [item])
    item.action = "author_question"
    with pytest.raises(ValueError, match=r"verification_followup|operational"):
        validate_items(assessed, data, [item])
