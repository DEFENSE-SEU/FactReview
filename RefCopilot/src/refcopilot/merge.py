"""Merge :class:`ExternalRecord` results from multiple backends into one :class:`MergedRecord`.

Field priority (first non-empty wins):
  - title / authors / year       → arXiv > S2 > OpenReview.
  - venue / publication_venue    → Semantic Scholar > arXiv > OpenReview.
  - DOI                          → Semantic Scholar > arXiv (OpenReview rarely has it).
  - arxiv_id / arxiv_versions / latest_arxiv_version / withdrawn → arXiv > S2.
  - URL                          → arXiv > S2 > OpenReview.

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
    openreview = next((r for r in records if r.backend == Backend.OPENREVIEW), None)

    provenance: dict[str, Backend] = {}

    title, prov_title = _pick("title", arxiv, s2, openreview, default="")
    authors, prov_authors = _pick_list("authors", arxiv, s2, openreview)
    year, prov_year = _pick("year", arxiv, s2, openreview)
    if title:
        provenance["title"] = prov_title
    if authors:
        provenance["authors"] = prov_authors
    if year is not None:
        provenance["year"] = prov_year

    venue: str | None = None
    venue_source: Backend | None = None
    for source, backend in (
        (s2, Backend.SEMANTIC_SCHOLAR),
        (arxiv, Backend.ARXIV),
        (openreview, Backend.OPENREVIEW),
    ):
        if source is None:
            continue
        candidate = source.publication_venue or source.venue or source.journal
        if candidate:
            venue = candidate
            venue_source = backend
            break
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
    for source in (arxiv, s2, openreview):
        if source is not None and source.url:
            url = source.url
            break

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


def _pick(
    field: str,
    arxiv: ExternalRecord | None,
    s2: ExternalRecord | None,
    openreview: ExternalRecord | None,
    *,
    default=None,
):
    for source, backend in (
        (arxiv, Backend.ARXIV),
        (s2, Backend.SEMANTIC_SCHOLAR),
        (openreview, Backend.OPENREVIEW),
    ):
        if source is None:
            continue
        v = getattr(source, field)
        if v not in (None, "", []):
            return v, backend
    return default, Backend.ARXIV  # provenance is meaningless when value is empty


def _pick_list(
    field: str,
    arxiv: ExternalRecord | None,
    s2: ExternalRecord | None,
    openreview: ExternalRecord | None,
):
    for source, backend in (
        (arxiv, Backend.ARXIV),
        (s2, Backend.SEMANTIC_SCHOLAR),
        (openreview, Backend.OPENREVIEW),
    ):
        if source is None:
            continue
        v = getattr(source, field) or []
        if v:
            return list(v), backend
    return [], Backend.ARXIV
