"""Paper schemas — the output of §3.1a ingestion.

A :class:`Paper` is the structured representation produced by any ingestion
backend (MinerU, Grobid, Nougat, Science-Parse, Llama-Index). Downstream
stages never see the raw PDF or backend-native formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PaperMetadata(BaseModel):
    """Bibliographic + locator metadata for the submission."""

    model_config = ConfigDict(extra="ignore")

    paper_key: str  # URL-safe short id (folder name)
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    repo_url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None


class Section(BaseModel):
    """A section of the paper body."""

    id: str  # stable id, e.g. "sec_3_1"
    number: str | None = None  # "3.1", "A.1", …
    title: str
    level: int = 1  # 1=section, 2=subsection, …
    text: str  # plain-text body (markdown-safe)
    char_start: int = 0  # offset in the full ingested markdown
    char_end: int = 0


class Table(BaseModel):
    """A table lifted from the paper."""

    id: str  # e.g. "table_3"
    number: str | None = None  # "3", "4", "A.1"
    caption: str = ""
    html: str | None = None  # original HTML when available
    markdown: str | None = None  # markdown rendering
    rows: list[list[str]] = Field(default_factory=list)
    section_id: str | None = None  # back-reference into Paper.sections


class Figure(BaseModel):
    """A figure reference (image kept on disk)."""

    id: str
    number: str | None = None
    caption: str = ""
    image_path: str | None = None
    section_id: str | None = None


class ReportedResult(BaseModel):
    """A numeric result claimed by the paper — usually extracted from tables."""

    id: str
    metric: str  # "MRR", "Accuracy", "F1", …
    value: float
    unit: str | None = None  # "%", "ms", …
    dataset: str | None = None
    task: str | None = None
    method: str | None = None  # the method the number belongs to
    table_id: str | None = None
    row_index: int | None = None
    col_index: int | None = None
    context: str = ""  # short surrounding text


class Paper(BaseModel):
    """Structured paper — the output of §3.1a ingestion."""

    model_config = ConfigDict(extra="ignore")

    metadata: PaperMetadata
    pdf_path: Path
    markdown_path: Path | None = None  # canonical extracted markdown
    sections: list[Section] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    reported_results: list[ReportedResult] = Field(default_factory=list)

    # Provenance from the ingestion backend.
    backend: str = "unknown"
    backend_artifacts: dict[str, Any] = Field(default_factory=dict)

    def section_by_id(self, sid: str) -> Section | None:
        return next((s for s in self.sections if s.id == sid), None)

    def table_by_id(self, tid: str) -> Table | None:
        return next((t for t in self.tables if t.id == tid), None)
