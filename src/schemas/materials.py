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


class FigureMaterial(BaseModel):
    id: str
    anchor: str = ""
    loc: ClaimLocation | None = None
    caption: str = ""
    references: list[MaterialBlock] = Field(default_factory=list)
    bbox_points: tuple[float, float, float, float] | None = None
    crop_path: str = ""
    printed_crop_path: str = ""
    printed_dpi: int = 96


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
    bibliography: list[MaterialBlock] = Field(default_factory=list)
    repository: RepositoryIndex | None = None
    issues: list[str] = Field(default_factory=list)
