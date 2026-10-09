"""Shared, traceable inputs prepared before v2 screening."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schemas.claim import ClaimLocation


class MaterialBlock(BaseModel):
    id: str
    text: str
    kind: str = "text"
    loc: ClaimLocation | None = None
    citations: list[str] = Field(default_factory=list)


class PageImage(BaseModel):
    page: int
    path: str
    width_points: float
    height_points: float
    dpi: int = 200


class FigureCaptionPart(BaseModel):
    """An unchanged parser element; visual role is deliberately not inferred here."""

    id: str
    field: str
    index: int
    text: str


class FigureMaterial(BaseModel):
    id: str
    block_id: str = ""
    parser_row_index: int | None = None
    parser_bbox_space: Literal["normalized_1000", "pdf_points"] | None = None
    caption_parts: list[FigureCaptionPart] = Field(default_factory=list)
    anchor: str = ""
    loc: ClaimLocation | None = None
    caption: str = ""
    caption_ambiguous: bool = False
    references: list[MaterialBlock] = Field(default_factory=list)
    bbox_points: tuple[float, float, float, float] | None = None
    crop_path: str = ""
    printed_crop_path: str = ""
    printed_dpi: int = 96


class TableMaterial(BaseModel):
    """One parser table and its original visual context; independent of figures."""

    id: str
    block_id: str
    anchor: str = ""
    loc: ClaimLocation | None = None
    caption: str = ""
    footnotes: str = ""
    caption_ambiguous: bool = False
    references: list[MaterialBlock] = Field(default_factory=list)
    bbox_points: tuple[float, float, float, float] | None = None
    crop_path: str = ""
    printed_crop_path: str = ""
    printed_dpi: int = 96
    issues: list[str] = Field(default_factory=list)


class RepositoryFile(BaseModel):
    path: str
    kind: Literal["documentation", "config", "entry", "source", "asset"]
    line_count: int | None = None
    sha256: str


class RepositoryIndex(BaseModel):
    root: str
    files: list[RepositoryFile] = Field(default_factory=list)
    entry_scripts: list[str] = Field(default_factory=list)
    configs: list[str] = Field(default_factory=list)


class SharedMaterials(BaseModel):
    paper_key: str
    title: str = ""
    abstract: str = ""
    source_pdf: str
    markdown: str
    markdown_path: str
    content_list_path: str
    provider: str
    blocks: list[MaterialBlock] = Field(default_factory=list)
    pages: list[PageImage] = Field(default_factory=list)
    figures: list[FigureMaterial] = Field(default_factory=list)
    tables: list[TableMaterial] = Field(default_factory=list)
    bibliography: list[MaterialBlock] = Field(default_factory=list)
    repository: RepositoryIndex | None = None
    issues: list[str] = Field(default_factory=list)
