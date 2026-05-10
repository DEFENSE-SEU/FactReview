"""Schema contract tests.

One test per public schema covering: round-trip via ``model_dump`` /
``model_validate`` preserves the fields the rest of the pipeline reads, and
the ``extra="forbid"`` guard on ``StageResult`` catches typo'd kwargs (the
documented purpose of that config).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.claim import Claim, ClaimLabel, ClaimLocation, ClaimType
from schemas.paper import Paper, PaperMetadata, Section
from schemas.review import ClaimAssessment, EvidenceLink, FinalReview
from schemas.stage import StageResult


def test_stage_result_roundtrip_and_forbids_typos() -> None:
    result = StageResult(
        status="ok",
        outputs={"main": "/tmp/out.json"},
        extra={"job_id": "j-1"},
    )
    dumped = result.model_dump()
    rehydrated = StageResult.model_validate(dumped)
    assert rehydrated == result
    assert rehydrated.get_output("main") == "/tmp/out.json"
    assert rehydrated.get_output("missing", "fallback") == "fallback"

    # The forbid-extra guard is the whole reason ConfigDict(extra="forbid")
    # exists on this model — protect against silent metadata loss from a
    # typo like ``extras={...}``.
    with pytest.raises(ValidationError):
        StageResult(status="ok", extras={"job_id": "j-2"})  # type: ignore[call-arg]


def test_paper_and_claim_roundtrip_preserve_enums() -> None:
    paper = Paper(
        metadata=PaperMetadata(paper_key="k", title="T", year=2024),
        pdf_path=Path("/tmp/k.pdf"),
        sections=[Section(id="sec_1", title="Intro", text="x", char_start=0)],
    )
    claim = Claim(
        id="claim_01",
        text="x",
        type=ClaimType.EMPIRICAL,
        location=ClaimLocation(section_id="sec_1"),
    )
    assessment = ClaimAssessment(
        claim_id="claim_01",
        label=ClaimLabel.PARTIALLY_SUPPORTED,
        rationale="weak gap",
        evidence=[EvidenceLink(kind="paper", locator="table_1")],
    )

    rehydrated_paper = Paper.model_validate(paper.model_dump())
    rehydrated_claim = Claim.model_validate(claim.model_dump())
    rehydrated_assessment = ClaimAssessment.model_validate(assessment.model_dump())
    assert rehydrated_paper.metadata.paper_key == "k"
    assert rehydrated_claim.type is ClaimType.EMPIRICAL
    # StrEnum survives JSON-mode dump too (the path the pipeline uses on disk).
    assert rehydrated_assessment.label is ClaimLabel.PARTIALLY_SUPPORTED


def test_final_review_assessments_round_trip() -> None:
    review = FinalReview(
        paper_key="k",
        run_id="r-1",
        review_markdown="# Review\nbody",
        evidence_markdown="evidence body",
        assessments=[
            ClaimAssessment(
                claim_id="claim_01",
                label=ClaimLabel.SUPPORTED,
                rationale="ok",
            )
        ],
        summary_counts={ClaimLabel.SUPPORTED: 1},
    )
    rehydrated = FinalReview.model_validate(review.model_dump())
    assert rehydrated.paper_key == "k"
    assert rehydrated.assessments[0].label is ClaimLabel.SUPPORTED
    assert rehydrated.summary_counts.get(ClaimLabel.SUPPORTED) == 1
