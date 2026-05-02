"""Positioning schemas — the output of §3.2 literature positioning."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class NoveltyType(StrEnum):
    """Paper §3.2: the role the submission plays relative to nearby work."""

    NEW_MECHANISM = "new_mechanism"
    NEW_COMBINATION = "new_combination"
    EMPIRICAL_IMPROVEMENT = "empirical_improvement"
    UNCLEAR = "unclear"


class NeighborMethod(BaseModel):
    """A cited or semantically nearby method in the comparison set."""

    model_config = ConfigDict(extra="ignore")

    name: str
    family: str | None = None  # "knowledge-graph-embedding", "relational-gcn", …
    citation_key: str | None = None  # BibTeX key when known
    arxiv_id: str | None = None
    doi: str | None = None
    semantic_scholar_id: str | None = None
    short_summary: str = ""
    design_axes: dict[str, str] = Field(default_factory=dict)
    # e.g. {"relation_embedding": "yes", "message_passing": "yes"}


class LiteratureContext(BaseModel):
    """Full §3.2 positioning output."""

    model_config = ConfigDict(extra="ignore")

    neighbors: list[NeighborMethod] = Field(default_factory=list)
    design_axes: list[str] = Field(default_factory=list)
    # e.g. ["node_embedding", "relation_embedding", "message_passing", "parameter_efficiency"]
    novelty: NoveltyType = NoveltyType.UNCLEAR
    novelty_rationale: str = ""
    families: list[str] = Field(default_factory=list)
