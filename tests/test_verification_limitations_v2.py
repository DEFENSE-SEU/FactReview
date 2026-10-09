"""System verification failures retain responsibility and claim scope."""

import pytest

from schemas.claim import AdviceItem, AuthorQuestion, Claim, ClaimAdvice, Evidence, EvidencePointer
from schemas.limitations import VerificationLimitation
from tests.test_dispatch_v2 import claim, materials
from verification.contracts import BranchResult, RejectedPlan
from verification.dispatch import verify_claims


def limitation(**changes):
    return VerificationLimitation(
        **{
            "claim_id": "c1",
            "condition_ids": ["d1"],
            "stage": "Code",
            "kind": "branch_failed",
            "reason": "An internal source check failed",
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes", [{"claim_id": "other"}, {"condition_ids": ["other"]}, {"condition_ids": [" d1 "]}]
)
async def test_foreign_or_padded_limitation_is_rejected_without_author_blame(tmp_path, changes):
    result = await verify_claims(
        [claim(["Code"])],
        materials(tmp_path),
        tmp_path,
        branches={
            "Code": lambda c, m: BranchResult(verification_limitations=[limitation(**changes)]),
        },
    )
    output = result.claims[0]
    assert not output.evidence and not output.questions
    assert len(output.verification_limitations) == 1
    assert output.verification_limitations[0].claim_id == "c1"
    assert output.verification_limitations[0].condition_ids == ["d1"]
    assert "enclosing claim" in output.verification_limitations[0].reason


async def test_rejected_plan_preserves_valid_observation_and_explicit_author_question(tmp_path):
    evidence = Evidence(
        source="paper_internal",
        pointer=EvidencePointer(locator="paper.pdf", page=1, quote="Claim"),
        covered=["d1"],
        direction="support",
        sufficient=True,
    )
    question = AuthorQuestion(
        claim_id="c1", text="Which seed was used?", reason="The manuscript does not specify a seed."
    )

    def reject(c, m):
        raise RejectedPlan(
            "Invalid proposed runtime target", BranchResult(evidence=[evidence], questions=[question])
        )

    result = await verify_claims(
        [claim(["Experiments"])], materials(tmp_path), tmp_path, branches={"Experiments": reject}
    )
    output = result.claims[0]
    assert result.plans == [] and output.evidence == [evidence]
    assert output.questions == [question]
    failure = output.verification_limitations[0]
    assert failure.kind == "plan_rejected" and failure.stage == "Experiments"
    assert failure.responsibility == "system" and failure.condition_ids == ["d1"]


async def test_failed_peer_does_not_hide_healthy_branch_or_existing_question(tmp_path):
    original = claim(["Theory", "Code"])
    original.questions = [AuthorQuestion(text="An existing source question", claim_id="c1")]
    evidence = Evidence(
        source="code",
        pointer=EvidencePointer(locator="x.py", line=1),
        covered=["d1"],
        direction="support",
        sufficient=False,
    )

    def fail(c, m):
        raise RuntimeError("Internal parser failure")

    result = await verify_claims(
        [original],
        materials(tmp_path),
        tmp_path,
        branches={
            "Theory": fail,
            "Code": lambda c, m: BranchResult(evidence=[evidence]),
        },
    )
    output = result.claims[0]
    assert output.evidence == [evidence] and output.questions == original.questions
    assert output.verification_limitations[0].stage == "Theory"
    assert original.verification_limitations == []


async def test_global_literature_cannot_attach_claim_specific_limitations(tmp_path):
    result = await verify_claims(
        [claim([])],
        materials(tmp_path),
        tmp_path,
        branches={},
        global_literature=lambda c, m: BranchResult(verification_limitations=[limitation()]),
    )
    assert result.claims[0].verification_limitations == []
    assert "findings/issues only" in result.issues[0]


def test_old_claim_and_advice_defaults_remain_readable():
    old = claim([]).model_dump(exclude={"theory_derivations", "verification_limitations"})
    restored = Claim.model_validate(old)
    assert restored.theory_derivations == [] and restored.verification_limitations == []
    item = AdviceItem(text="Review this claim", condition_ids=["d1"], basis_refs=["/text"])
    advice = ClaimAdvice(state="generated", items=[item], input_sha256="a" * 64)
    assert advice.input_version == "advice-v1" and item.action == "reviewer_guidance"
