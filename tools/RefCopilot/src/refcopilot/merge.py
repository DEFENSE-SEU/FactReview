"""Merge :class:`ExternalRecord` results from multiple backends into one :class:`MergedRecord`.

Field priority:
  - title / authors / year       → arXiv (more authoritative for arXiv-cited papers).
  - venue / publication_venue    → Semantic Scholar (carries the published venue).
  - DOI                          → Semantic Scholar; arXiv DOI as fallback.
  - arxiv_id / arxiv_versions / latest_arxiv_version / withdrawn → arXiv.
  - s2_paper_id                  → Semantic Scholar.

Each merged field's provenance (``Backend``) is recorded so callers can trace
where a value came from.
"""

from __future__ import annotations

from refcopilot.models import Backend, ExternalRecord, MergedRecord


def merge_records(records: list[ExternalRecord]) -> MergedRecord | None:
    if not records:
        return None

    arxiv = next((r for r in records if r.backend == Backend.ARXIV), None)
    s2 = next((r for r in records if r.backend == Backend.SEMANTIC_SCHOLAR), None)

    provenance: dict[str, Backend] = {}

    title, prov_title = _pick("title", arxiv, s2, default="")
    authors, prov_authors = _pick_list("authors", arxiv, s2)
    year, prov_year = _pick("year", arxiv, s2)
    if title:
        provenance["title"] = prov_title
    if authors:
        provenance["authors"] = prov_authors
    if year is not None:
        provenance["year"] = prov_year

    venue: str | None = None
    venue_source: Backend | None = None
    if s2:
        venue = s2.publication_venue or s2.venue or s2.journal
        if venue:
            venue_source = Backend.SEMANTIC_SCHOLAR
    if not venue and arxiv:
        venue = arxiv.publication_venue or arxiv.venue or arxiv.journal
        if venue:
            venue_source = Backend.ARXIV
    if venue and venue_source:
        provenance["venue"] = venue_source

    doi: str | None = None
    if s2 and s2.doi:
        doi = s2.doi
        provenance["doi"] = Backend.SEMANTIC_SCHOLAR
    elif arxiv and arxiv.doi:
        doi = arxiv.doi
        provenance["doi"] = Backend.ARXIV

    arxiv_id: str | None = None
    arxiv_versions: list[int] = []
    latest_arxiv_version: int | None = None
    withdrawn = False
    if arxiv:
        arxiv_id = arxiv.arxiv_id
        arxiv_versions = list(arxiv.arxiv_versions)
        latest_arxiv_version = arxiv.latest_arxiv_version
        withdrawn = arxiv.withdrawn
        if arxiv_id:
            provenance["arxiv_id"] = Backend.ARXIV
    if not arxiv_id and s2 and s2.arxiv_id:
        arxiv_id = s2.arxiv_id
        provenance["arxiv_id"] = Backend.SEMANTIC_SCHOLAR

    url = ""
    if arxiv and arxiv.url:
        url = arxiv.url
    elif s2 and s2.url:
        url = s2.url

    return MergedRecord(
        title=title or "",
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        arxiv_versions=arxiv_versions,
        latest_arxiv_version=latest_arxiv_version,
        withdrawn=withdrawn,
        url=url,
        provenance=provenance,
        sources=list(records),
    )


def _pick(field: str, arxiv: ExternalRecord | None, s2: ExternalRecord | None, *, default=None):
    if arxiv:
        v = getattr(arxiv, field)
        if v not in (None, "", []):
            return v, Backend.ARXIV
    if s2:
        v = getattr(s2, field)
        if v not in (None, "", []):
            return v, Backend.SEMANTIC_SCHOLAR
    return default, Backend.ARXIV  # provenance is meaningless when value is empty


def _pick_list(field: str, arxiv: ExternalRecord | None, s2: ExternalRecord | None):
    if arxiv:
        v = getattr(arxiv, field) or []
        if v:
            return list(v), Backend.ARXIV
    if s2:
        v = getattr(s2, field) or []
        if v:
            return list(v), Backend.SEMANTIC_SCHOLAR
    return [], Backend.ARXIV
