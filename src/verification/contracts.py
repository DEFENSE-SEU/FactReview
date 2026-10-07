"""Shared results from the four peer verification branches."""

from pydantic import Field

from schemas.claim import AuthorQuestion, Contract, Evidence, ExecutionPlan, Finding


class BranchResult(Contract):
    evidence: list[Evidence] = Field(default_factory=list)
    plans: list[ExecutionPlan] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    questions: list[AuthorQuestion] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
