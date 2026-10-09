"""Missing validated scope is a verifier limitation, without changing evidence."""

import copy

import pytest

from common import run_stats
from schemas.claim import Condition
from tests.test_code_binding_v2 import inputs as inputs
from tests.test_code_binding_v2 import offline as offline
from tests.test_code_binding_v2 import review, run


@pytest.mark.parametrize("missing", ["condition", "decision"])
def test_rejected_scope_is_local_system_limitation_and_keeps_healthy_neighbor(inputs, tmp_path, missing):
    claim, materials, item = inputs()
    claim.conditions.append(Condition(id="c2", description=claim.text))
    second = {**copy.deepcopy(item), "covered": ["c2"], "fully_supported_conditions": ["c2"]}
    response = review(claim, [item, second])
    if missing == "condition":
        response["conditions"][0]["claim_source_ids"] = ["foreign"]
    else:
        response["items"][0]["source_uses"] = [
            {"source_index": 0, "role": "implementation", "rationale": "Exact source implements Adam."}
        ]
    original = copy.deepcopy(response)
    with run_stats.run_scope(tmp_path / "run_stats.json"):
        result, assessed, calls = run(claim, materials, [item, second], response)
    assert calls == ["verification.code", "verification.code.scope"]
    assert response == original
    assert assessed.status == "unverified"
    assert {cid for e in result.evidence if e.sufficient for cid in e.covered} == {"c2"}
    assert len(result.verification_limitations) == 1
    limitation = result.verification_limitations[0]
    assert limitation.claim_id == claim.id and limitation.condition_ids == ["c1"]
    assert (limitation.stage, limitation.kind, limitation.responsibility) == (
        "Code",
        "evidence_validation_failed",
        "system",
    )
    assert "candidate 0" in limitation.reason and "condition c1" in limitation.reason
    audit = next((tmp_path / "code_scope_reviews").glob("*.json"))
    assert str(audit) in limitation.reason
    assert not result.questions


@pytest.mark.parametrize("relation", ["partial", "irrelevant", "contradicts_implementation"])
def test_validated_non_support_is_not_scope_protocol_failure(inputs, relation):
    claim, materials, item = inputs()
    response = review(claim, [item], relation=relation)
    if relation == "partial":
        response["items"][0]["missing_qualifiers"] = ["Implementation detail remains unresolved."]
    result, assessed, calls = run(claim, materials, [item], response)
    assert calls == ["verification.code", "verification.code.scope"]
    assert result.evidence and not any(e.sufficient for e in result.evidence)
    assert assessed.status == "unverified"
    assert not result.verification_limitations
