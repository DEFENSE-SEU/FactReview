"""Pydantic data model for the whole pipeline.

Reference          → a citation extracted from the user's input
ExternalRecord     → a single record returned by an arXiv / S2 lookup
MergedRecord       → arXiv + S2 merged with per-field provenance
Issue              → one detection (fake / outdated / incomplete)
CheckedReference   → reference + matches + issues + final verdict
Report             → top-level aggregate over all CheckedReferences
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    UNVERIFIED = "unverified"


class IssueCategory(str, Enum):
    FAKE = "fake"
    OUTDATED = "outdated"
    INCOMPLETE = "incomplete"
    NON_ACADEMIC = "non_academic"
    RETRACTED = "retracted"


class Verdict(str, Enum):
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    UNVERIFIED = "unverified"


class HallucinationVerdict(str, Enum):
    LIKELY = "LIKELY"
    UNLIKELY = "UNLIKELY"
    UNCERTAIN = "UNCERTAIN"


class SourceFormat(str, Enum):
    BIBTEX = "bibtex"
    PDF = "pdf"
    URL = "url"
    TEXT = "text"


class Reference(BaseModel):
    raw: str
    source_format: SourceFormat
    bibkey: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    arxiv_version: int | None = None
    url: str | None = None


class Backend(str, Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENREVIEW = "openreview"
    OPENALEX = "openalex"


class FoundBy(str, Enum):
    """How a backend retrieved a record. Informational only — never used for
    verdicts; lets reports show where each candidate came from."""

    ARXIV_ID = "arxiv_id"
    DOI = "doi"
    OPENREVIEW_ID = "openreview_id"
    TITLE = "title"
    LLM_SUGGESTION = "llm_suggestion"  # second-chance lookup with the LLM's suggested metadata


class ExternalRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: Backend
    record_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    publication_venue: str | None = None
    journal: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    arxiv_versions: list[int] = Field(default_factory=list)
    latest_arxiv_version: int | None = None
    is_retracted: bool = False
    s2_paper_id: str | None = None
    url: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)
    found_by: FoundBy | None = None


def tag_found_by(records: list[ExternalRecord], how: FoundBy) -> list[ExternalRecord]:
    """Copies of ``records`` marked with how they were retrieved."""
    return [r.model_copy(update={"found_by": how}) for r in records]


class MergedRecord(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    arxiv_versions: list[int] = Field(default_factory=list)
    latest_arxiv_version: int | None = None
    is_retracted: bool = False
    url: str = ""
    provenance: dict[str, Backend] = Field(default_factory=dict)
    sources: list[ExternalRecord] = Field(default_factory=list)


class Issue(BaseModel):
    severity: Severity
    category: IssueCategory
    code: str
    message: str
    suggestion: str | None = None
    confidence: float = 1.0


class CheckedReference(BaseModel):
    reference: Reference
    matches: list[ExternalRecord] = Field(default_factory=list)
    merged: MergedRecord | None = None
    hallucination_verdict: HallucinationVerdict | None = None
    verification_trace: str | None = None
    issues: list[Issue] = Field(default_factory=list)
    verdict: Verdict = Verdict.VALID


class ReportSummary(BaseModel):
    total_refs: int = 0
    errors: int = 0
    warnings: int = 0
    unverified: int = 0
    # References found in the input, before any ``max_refs`` cut-off
    # (``total_refs`` counts only the ones actually checked).
    refs_found: int = 0
    by_category: dict[str, int] = Field(default_factory=dict)


class Report(BaseModel):
    paper: dict[str, Any] | None = None
    checked: list[CheckedReference] = Field(default_factory=list)
    summary: ReportSummary = Field(default_factory=ReportSummary)
