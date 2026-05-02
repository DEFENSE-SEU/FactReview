"""Review schemas — the output of §3.4 review (report + teaser)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from schemas.claim import ClaimLabel


class EvidenceLink(BaseModel):
    """A pointer back to the evidence that justifies an assessment."""

    kind: str  # "paper" | "literature" | "execution"
    locator: str  # section_id / neighbor.name / task_id…
    snippet: str = ""  # short human-readable excerpt
    score: float | None = None  # optional confidence / similarity


class ClaimAssessment(BaseModel):
    """A single claim's verdict with linked evidence."""

    model_config = ConfigDict(extra="ignore")

    claim_id: str
    label: ClaimLabel
    rationale: str
    evidence: list[EvidenceLink] = Field(default_factory=list)
    confidence: float | None = None  # 0..1 when available
    subclaim_labels: dict[str, ClaimLabel] = Field(default_factory=dict)


class FinalReview(BaseModel):
    """The full §3.4 output: review text + per-claim assessments."""

    model_config = ConfigDict(extra="ignore")

    paper_key: str
    run_id: str
    review_markdown: str  # the concise review text
    evidence_markdown: str  # the linked evidence report
    assessments: list[ClaimAssessment] = Field(default_factory=list)
    summary_counts: dict[ClaimLabel, int] = Field(default_factory=dict)
