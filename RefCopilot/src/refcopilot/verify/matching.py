"""Shared "is this citation matched, and by which records" decision.

A citation is MATCHED when there is a single retrieved candidate whose title
and authors are both consistent with the citation (the "anchor"). Fields are
never pooled across different candidates to decide this — if no single
candidate satisfies both at once, the citation is unmatched (fake).

Cited identifiers (doi/arxiv_id) deliberately play no part in this decision:
a paper can carry several DOIs (arXiv DataCite, conference, journal) while
each backend record exposes at most one, so an id mismatch between the
citation and a record says more about backend coverage than about whether
the cited paper exists.

Once an anchor is found, :func:`build_paper_cluster` collects the other
retrieved candidates that represent the SAME real-world paper as the anchor
(shared normalized arxiv_id/doi, or high title+author similarity between the
two clean records). This cluster — not the raw candidate pool — is what
should be merged and used for outdated/completeness/retraction checks.

Used by both :mod:`refcopilot.verify.hallucination` (to decide LIKELY /
UNLIKELY / UNCERTAIN) and :mod:`refcopilot.pipeline` (to build the
``MergedRecord``), so the "is this real" decision and the "which record(s)
do we merge from" decision never disagree about which paper was matched.
"""

from __future__ import annotations

from refcopilot.models import ExternalRecord, Reference
from refcopilot.verify.text_match import (
    author_overlap,
    normalize_arxiv_id,
    normalize_doi,
    title_similarity,
)
from refcopilot.verify.thresholds import (
    AUTHOR_FAKE_THRESHOLD,
    SAME_PAPER_AUTHOR_OVERLAP_MIN,
    SAME_PAPER_TITLE_SIM_THRESHOLD,
    TITLE_FAKE_THRESHOLD,
)


def find_anchor(reference: Reference, matches: list[ExternalRecord]) -> ExternalRecord | None:
    """Return the best candidate whose title and authors are both consistent
    with ``reference``, or ``None`` if no candidate qualifies.
    """
    if not matches or not reference.title:
        return None

    best: ExternalRecord | None = None
    best_sim = -1.0
    for m in matches:
        sim = title_similarity(reference.title, m.title)
        if sim < TITLE_FAKE_THRESHOLD:
            continue
        if _author_conflicts(reference, m):
            continue
        if sim > best_sim:
            best_sim = sim
            best = m
    return best


def build_paper_cluster(
    anchor: ExternalRecord, matches: list[ExternalRecord]
) -> list[ExternalRecord]:
    """Return ``[anchor]`` plus every other candidate confirmed to be the same paper."""
    cluster = [anchor]
    for m in matches:
        if m is anchor:
            continue
        if _same_paper(anchor, m):
            cluster.append(m)
    return cluster


def resolve_cluster(
    reference: Reference, matches: list[ExternalRecord]
) -> tuple[ExternalRecord | None, list[ExternalRecord]]:
    """``find_anchor`` + ``build_paper_cluster`` in one call.

    Returns ``(None, [])`` when no anchor is found.
    """
    anchor = find_anchor(reference, matches)
    if anchor is None:
        return None, []
    return anchor, build_paper_cluster(anchor, matches)


def _author_conflicts(reference: Reference, candidate: ExternalRecord) -> bool:
    overlap = author_overlap(reference.authors, candidate.authors)
    return overlap <= AUTHOR_FAKE_THRESHOLD and not reference.url


def _same_paper(a: ExternalRecord, b: ExternalRecord) -> bool:
    # An arXiv id is shared by all versions of a paper, so equal ids prove the
    # same paper and different ids prove different papers. DOIs only prove
    # sameness: one paper can have several (arXiv DataCite, conference, journal).
    a_arxiv = normalize_arxiv_id(a.arxiv_id)
    b_arxiv = normalize_arxiv_id(b.arxiv_id)
    if a_arxiv and b_arxiv:
        return a_arxiv == b_arxiv

    a_doi = normalize_doi(a.doi)
    b_doi = normalize_doi(b.doi)
    if a_doi and b_doi and a_doi == b_doi:
        return True

    sim = title_similarity(a.title, b.title)
    if sim < SAME_PAPER_TITLE_SIM_THRESHOLD:
        return False
    if not a.authors or not b.authors:
        return True
    return author_overlap(a.authors, b.authors) >= SAME_PAPER_AUTHOR_OVERLAP_MIN
