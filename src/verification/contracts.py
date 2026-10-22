"""Shared results from the four peer verification branches."""

from pydantic import Field

from schemas.claim import AuthorQuestion, Contract, Evidence, ExecutionPlan, Finding, TheoryRecord
from schemas.limitations import VerificationLimitation
from schemas.review import DeliveryCheck


class BranchResult(Contract):
    evidence: list[Evidence] = Field(default_factory=list)
    plans: list[ExecutionPlan] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    questions: list[AuthorQuestion] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    theory_derivations: list[TheoryRecord] = Field(default_factory=list)
    verification_limitations: list[VerificationLimitation] = Field(default_factory=list)
    delivery_checks: list[DeliveryCheck] = Field(default_factory=list)


class RejectedPlan(ValueError):
    """An invalid execution plan, with independently validated paper observations."""

    def __init__(self, reason: str, observations: BranchResult):
        if observations.plans:
            raise ValueError("Rejected-plan observations must not contain execution plans")
        super().__init__(reason)
        self.observations = observations.model_copy(deep=True)
