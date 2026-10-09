"""V2 review artifacts; historical report contracts live in legacy_review."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from schemas.claim import Claim, ClaimStatus, Contract, Finding, NonEmpty


class FinalReview(Contract):
    paper_key: NonEmpty
    run_id: NonEmpty
    claims: list[Claim] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    ledger: list[dict[str, Any]] = Field(default_factory=list)
    review_markdown: str = ""
    run_status: Literal["completed", "partial"] = "completed"
    incomplete_stages: list[str] = Field(default_factory=list)
    # None preserves the unspecified context of historical artifacts.
    execution_requested: bool | None = None

    @property
    def summary_counts(self) -> dict[ClaimStatus, int]:
        return {status: sum(claim.status == status for claim in self.claims) for status in ClaimStatus}
