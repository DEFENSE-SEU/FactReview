"""Closed wire contract for one original-page Theory recovery pass."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Exact = Annotated[str, Field(strict=True, min_length=1)]
Index = Annotated[int, Field(strict=True, ge=0)]


class VisualContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PaperRef(VisualContract):
    source_kind: Literal["paper"]
    block_id: Exact
    quote: Exact


class VisualRef(VisualContract):
    source_kind: Literal["visual"]
    visual_source_index: Index


SourceRef = Annotated[PaperRef | VisualRef, Field(discriminator="source_kind")]


class VisualReading(VisualContract):
    target_id: Exact
    page_id: Exact
    anchor_block_id: Exact
    printed_anchor: str = Field(strict=True)
    transcription: Exact = Field(
        description="Your reading of the original pixels; preserve symbols and qualifiers. This is model-transcribed text, never an exact parsed-source quote."
    )


class VisualAssumption(VisualContract):
    id: Exact
    text: Exact
    status: Literal["paper_explicit", "required_unstated"]
    sources: list[SourceRef]


class VisualStep(VisualContract):
    id: Exact
    statement: Exact
    reason: Exact
    assumption_ids: list[Exact]
    previous_step_ids: list[Exact]
    sources: list[SourceRef]


class VisualGap(VisualContract):
    at: Exact
    reason: Exact
    needed: Exact
    sources: list[SourceRef]


class VisualDerivation(VisualContract):
    goal: Exact
    assumptions: list[VisualAssumption]
    steps: list[VisualStep]
    gaps: list[VisualGap]
    outcome: Literal["completed", "partial", "unable"]
    completion_reason: Exact


class VisualItem(VisualContract):
    target_id: Exact
    direction: Literal["support", "flaw"]
    fully_supported: bool = Field(strict=True)
    detail: Exact
    trace: VisualDerivation


class VisualOutput(VisualContract):
    schema_version: Literal["theory-visual-derivation-v1"]
    visual_sources: list[VisualReading]
    items: list[VisualItem]
