"""V2 claim, evidence, and handoff contracts (method specification §2 and §6)."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    FLAWED = "flawed"
    QUESTIONED = "questioned"
    UNVERIFIED = "unverified"


class EvidenceNeed(StrEnum):
    LITERATURE = "Literature"
    THEORY = "Theory"
    CODE = "Code"
    EXPERIMENTS = "Experiments"


class ClaimLocation(Contract):
    """One-based page, section anchor, and/or offsets into the parsed paper."""

    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def localizable(self) -> Self:
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be supplied together")
        if self.char_start is not None and self.char_end <= self.char_start:
            raise ValueError("char_end must follow char_start")
        if self.page is None and not (self.section or "").strip() and self.char_start is None:
            raise ValueError("a claim location requires a page, section, or character span")
        return self


class Condition(Contract):
    """A stable coverage unit; settings can include a task, split, model, or seed."""

    id: NonEmpty
    dataset: NonEmpty | None = None
    metric: NonEmpty | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    @model_validator(mode="after")
    def specified(self) -> Self:
        if not (self.dataset or self.metric or self.settings or self.description.strip()):
            raise ValueError("a condition requires dataset, metric, settings, or a description")
        return self


class EvidencePointer(Contract):
    """Checkable artifact locator and passage/line/output key.

    Examples: paper.pdf + page/quote; arXiv/DOI URL + quote; config.yaml +
    line; runs/eval/metrics.json + key. The producing branch checks existence;
    this schema checks the information needed to locate the evidence.
    """

    locator: NonEmpty
    quote: str = ""
    page: int | None = Field(default=None, ge=1)
    line: int | None = Field(default=None, ge=1)
    key: str | None = None


class ExecutionProvenance(Contract):
    """Identify recomputation from the authors' released data or logs."""

    released_artifact: bool = False
    artifact_kind: Literal["data", "logs", "other"] = "other"
    environment_explanation_possible: bool = True
    run_id: str | None = None
    command: list[str] = Field(default_factory=list)
    runtime_conditions: list[Condition] = Field(default_factory=list)
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    repository: str | None = None
    recomputation_pointer: str | None = None


class Evidence(Contract):
    source: Literal["paper_internal", "literature", "theory", "code", "execution"]
    pointer: EvidencePointer
    covered: list[NonEmpty] = Field(default_factory=list)
    direction: Literal["support", "flaw"]
    sufficient: bool = False
    note: str = ""
    concern: bool = False
    affects_claim: bool = True
    overturnable: bool = True
    aligned: bool | None = None
    provenance: ExecutionProvenance | None = None

    @model_validator(mode="after")
    def check_pointer_and_alignment(self) -> Self:
        pointer = self.pointer
        if self.source in {"paper_internal", "theory"}:
            if not pointer.quote.strip() or not (pointer.page or (pointer.key or "").strip()):
                raise ValueError("paper/theory evidence requires a quote and page or section key")
        elif self.source == "literature":
            url = urlsplit(pointer.locator)
            valid_url = url.scheme in {"https", "http"} and bool(url.netloc)
            valid_identifier = re.fullmatch(
                r"(?i)(?:doi:)?10\.\d{4,9}/\S+|arxiv:(?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?",
                pointer.locator,
            )
            search_audit = pointer.locator.lower().endswith(".json") and pointer.key == "search_scope"
            if not (valid_url or valid_identifier or search_audit):
                raise ValueError(
                    "literature evidence requires a DOI, arXiv id, URL, or saved search-scope audit"
                )
            if not pointer.quote.strip():
                raise ValueError("literature evidence requires the retrieved passage")
        elif self.source == "code" and pointer.line is None:
            raise ValueError("code evidence requires a source/config file and line")
        elif self.source == "execution":
            if not (pointer.key or "").strip():
                raise ValueError("execution evidence requires a log/metric path and key")
            if self.sufficient and self.aligned is not True:
                raise ValueError("sufficient execution evidence must be aligned")
        if len(self.covered) != len(set(self.covered)):
            raise ValueError("evidence coverage ids must be unique")
        if self.sufficient and not self.covered:
            raise ValueError("sufficient evidence must identify covered conditions")
        return self


class AuthorQuestion(Contract):
    text: NonEmpty
    claim_id: str | None = None
    reason: str = ""


class ClaimSourceRef(Contract):
    """An original manuscript passage for specified conditions, not verification evidence."""

    source_block_id: NonEmpty
    source_quote: str = Field(min_length=1)
    loc: ClaimLocation
    covered: list[NonEmpty] = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> Self:
        if not self.source_quote.strip():
            raise ValueError("source_quote must contain original manuscript text")
        if len(self.covered) != len(set(self.covered)):
            raise ValueError("source reference coverage ids must be unique")
        return self


class Claim(Contract):
    """The same record is enriched from extraction through final assessment."""

    id: NonEmpty
    text: NonEmpty
    loc: ClaimLocation
    # Optional for previously saved records; extraction retains these together.
    source_block_id: NonEmpty | None = None
    source_quote: str | None = None
    source_refs: list[ClaimSourceRef] = Field(default_factory=list)
    conditions: list[Condition] = Field(min_length=1)
    needs: list[EvidenceNeed]
    importance: Literal["core", "secondary"] = "secondary"
    questions: list[AuthorQuestion] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_references(self) -> Self:
        if (self.source_block_id is None) != (self.source_quote is None):
            raise ValueError("source_block_id and source_quote must be supplied together")
        if self.source_quote is not None and not self.source_quote.strip():
            raise ValueError("source_quote must contain original manuscript text")
        ids = [condition.id for condition in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("condition ids must be unique within a claim")
        if len(self.needs) != len(set(self.needs)):
            raise ValueError("needs must not contain duplicate branches")
        source_keys = []
        for ref in self.source_refs:
            if not set(ref.covered).issubset(ids):
                raise ValueError("source reference coverage must refer to this claim's condition ids")
            source_keys.append((ref.source_block_id, ref.source_quote, tuple(sorted(ref.covered))))
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("source references must not be duplicated")
        for item in self.evidence:
            if not set(item.covered).issubset(ids):
                raise ValueError("evidence coverage must refer to this claim's condition ids")
        for question in self.questions:
            if question.claim_id is not None and question.claim_id != self.id:
                raise ValueError("an author question must refer to its enclosing claim")
        return self


class ExecutionTask(Contract):
    entry_script: str | None = None
    config: str | None = None
    command: list[str] = Field(default_factory=list)
    workdir: str = "."
    metric_output: str | None = None


class ExecutionPlan(Contract):
    id: NonEmpty
    claim_id: NonEmpty
    condition_ids: list[NonEmpty] = Field(min_length=1)
    target_conditions: list[Condition] = Field(min_length=1)
    task: ExecutionTask = Field(default_factory=ExecutionTask)
    run_mode: Literal["evaluation", "analysis", "training"]
    # Condition-id keys preserve two datasets reporting the same metric.
    y_paper: dict[str, FiniteNumber] = Field(min_length=1)
    feasibility: Literal["ready", "blocked"]
    blocker: str = ""
    priority: Literal["high", "medium", "low"]
    estimated_cost: str = "unknown"

    @model_validator(mode="after")
    def check_feasibility_and_conditions(self) -> Self:
        if self.feasibility == "blocked" and not self.blocker.strip():
            raise ValueError("blocked plans require a blocker reason")
        if self.feasibility == "ready" and self.blocker.strip():
            raise ValueError("ready plans cannot have an unresolved blocker")
        if self.feasibility == "ready" and not (self.task.entry_script or self.task.command):
            raise ValueError("ready plans require a candidate script or command")
        ids = [condition.id for condition in self.target_conditions]
        if len(self.condition_ids) != len(set(self.condition_ids)) or len(ids) != len(set(ids)):
            raise ValueError("plan condition ids must be unique")
        if set(self.condition_ids) != set(ids):
            raise ValueError("condition_ids must match target_conditions")
        if set(self.y_paper) != set(ids):
            raise ValueError("y_paper keys must match condition_ids")
        if any(not condition.metric for condition in self.target_conditions):
            raise ValueError("execution target conditions require a metric")
        return self


class Finding(Contract):
    kind: Literal["writing", "figure", "reference", "related_work", "baseline", "table"]
    loc: ClaimLocation
    evidence: list[Evidence] = Field(min_length=1)
    level: NonEmpty
    text: NonEmpty
