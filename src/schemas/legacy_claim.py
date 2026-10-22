"""Claim schemas used by the review and audit pipeline."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ClaimType(StrEnum):
    """Paper §3.1: claim taxonomy."""

    EMPIRICAL = "empirical"
    METHODOLOGICAL = "methodological"
    THEORETICAL = "theoretical"


class ClaimLabel(StrEnum):
    """The four verdicts for claim assessment."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    INCONCLUSIVE = "inconclusive"
    IN_CONFLICT = "in_conflict"


class ClaimLocation(BaseModel):
    """Where in the paper a claim lives."""

    section_id: str | None = None
    table_id: str | None = None
    figure_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    page: int | None = None


class SubClaim(BaseModel):
    """A decomposition of a broad claim by (task, dataset, metric).

    Example: the CompGCN claim "outperforms prior work across link prediction,
    node classification, and graph classification" decomposes into three
    :class:`SubClaim` entries, one per task.
    """

    id: str  # stable id, e.g. "claim_01.sub_02"
    text: str
    task: str | None = None
    dataset: str | None = None
    metric: str | None = None
    expected_value: float | None = None
    expected_tolerance: float | None = None


class Claim(BaseModel):
    """A review-relevant claim extracted from the paper."""

    model_config = ConfigDict(extra="ignore")

    id: str  # stable within one paper, e.g. "claim_01"
    text: str  # natural-language statement
    type: ClaimType
    scope: str = ""  # "broad" / "local" / free-form
    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    importance: str = ""  # "primary" / "secondary" when supplied by the extractor.
    location: ClaimLocation = Field(default_factory=ClaimLocation)
    subclaims: list[SubClaim] = Field(default_factory=list)
    evidence_targets: list[str] = Field(default_factory=list)
    # What evidence MUST we produce to move this claim off Inconclusive?
    # e.g. ["table_3.MRR.FB15k-237", "execution.eval_fb237_conve.mrr"]
