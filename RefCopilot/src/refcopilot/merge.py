"""Merge :class:`ExternalRecord` results from multiple backends into one :class:`MergedRecord`.

Field priority (first non-empty wins) — earlier backends in each list are
considered more authoritative for that field:

  - title / authors / year       → arXiv > S2 > OpenAlex > OpenReview.
  - venue / publication_venue    → S2 > OpenAlex > arXiv > OpenReview.
  - DOI                          → S2 > OpenAlex > arXiv (OpenReview rarely has it).
  - arxiv_id / arxiv_versions / latest_arxiv_version → arXiv > S2.
  - is_retracted                 → OR across all backends (any positive wins).
  - URL                          → arXiv > S2 > OpenAlex > OpenReview.

Each merged field's provenance (``Backend``) is recorded so callers can trace
where a value came from.
"""

from __future__ import annotations

from typing import Callable

from refcopilot.models import Backend, ExternalRecord, MergedRecord


_TITLE_PRIORITY = (
    Backend.ARXIV,
    Backend.SEMANTIC_SCHOLAR,
    Backend.OPENALEX,
    Backend.OPENREVIEW,
)
_VENUE_PRIORITY = (
    Backend.SEMANTIC_SCHOLAR,
    Backend.OPENALEX,
    Backend.ARXIV,
    Backend.OPENREVIEW,
)
_DOI_PRIORITY = (
    Backend.SEMANTIC_SCHOLAR,
    Backend.OPENALEX,
    Backend.ARXIV,
)
_ARXIV_ID_PRIORITY = (Backend.ARXIV, Backend.SEMANTIC_SCHOLAR)
_URL_PRIORITY = (
    Backend.ARXIV,
    Backend.SEMANTIC_SCHOLAR,
    Backend.OPENALEX,
    Backend.OPENREVIEW,
)


def merge_records(records: list[ExternalRecord]) -> MergedRecord | None:
    if not records:
        return None

    # Index by backend; if a backend appears multiple times, keep the first.
    by_backend: dict[Backend, ExternalRecord] = {}
    for r in records:
        by_backend.setdefault(r.backend, r)

    provenance: dict[str, Backend] = {}

    title, prov_title = _pick(by_backend, _TITLE_PRIORITY, lambda r: r.title)
    authors, prov_authors = _pick(by_backend, _TITLE_PRIORITY, lambda r: r.authors)
    year, prov_year = _pick(by_backend, _TITLE_PRIORITY, lambda r: r.year)
    if title:
        provenance["title"] = prov_title  # type: ignore[assignment]
    if authors:
        provenance["authors"] = prov_authors  # type: ignore[assignment]
    if year is not None:
        provenance["year"] = prov_year  # type: ignore[assignment]

    venue, prov_venue = _pick(
        by_backend,
        _VENUE_PRIORITY,
        lambda r: r.publication_venue or r.venue or r.journal,
    )
    if venue:
        provenance["venue"] = prov_venue  # type: ignore[assignment]

    doi, prov_doi = _pick(by_backend, _DOI_PRIORITY, lambda r: r.doi)
    if doi:
        provenance["doi"] = prov_doi  # type: ignore[assignment]

    arxiv_rec = by_backend.get(Backend.ARXIV)
    arxiv_id: str | None = None
    arxiv_versions: list[int] = []
    latest_arxiv_version: int | None = None
    if arxiv_rec:
        arxiv_id = arxiv_rec.arxiv_id
        arxiv_versions = list(arxiv_rec.arxiv_versions)
        latest_arxiv_version = arxiv_rec.latest_arxiv_version
        if arxiv_id:
            provenance["arxiv_id"] = Backend.ARXIV
    if not arxiv_id:
        # Fall back to S2's externalIds.ArXiv.
        s2_rec = by_backend.get(Backend.SEMANTIC_SCHOLAR)
        if s2_rec and s2_rec.arxiv_id:
            arxiv_id = s2_rec.arxiv_id
            provenance["arxiv_id"] = Backend.SEMANTIC_SCHOLAR

    # Retraction is a positive signal — any backend that flags the work as
    # retracted (arXiv "withdrawn" string, OpenAlex is_retracted, future
    # CrossRef update-to) wins, regardless of backend priority.
    is_retracted = any(rec.is_retracted for rec in records)

    url_value, _ = _pick(by_backend, _URL_PRIORITY, lambda r: r.url)
    url = url_value or ""

    return MergedRecord(
        title=title or "",
        authors=authors or [],
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        arxiv_versions=arxiv_versions,
        latest_arxiv_version=latest_arxiv_version,
        is_retracted=is_retracted,
        url=url,
        provenance=provenance,
        sources=list(records),
    )


def _pick(
    by_backend: dict[Backend, ExternalRecord],
    priority: tuple[Backend, ...],
    getter: Callable[[ExternalRecord], object],
):
    """Return the first non-empty ``getter(record)`` walking ``priority``.

    Returns a ``(value, backend)`` tuple, or ``(None, None)`` if no backend in
    the priority list has a non-empty value.
    """
    for backend in priority:
        rec = by_backend.get(backend)
        if rec is None:
            continue
        value = getter(rec)
        if value not in (None, "", []):
            return value, backend
    return None, None
