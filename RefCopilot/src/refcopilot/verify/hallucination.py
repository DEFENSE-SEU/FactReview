"""Hallucination (fake reference) detection.

Two-stage:

1.  :func:`pre_screen` — heuristic verdict (no LLM call) based on title
    similarity, author overlap, and OCR-garbled-title detection.
2.  :func:`to_issue` — given a final verdict, emit an :class:`Issue` only when
    the verdict is ``LIKELY``. ``UNLIKELY`` and ``UNCERTAIN`` produce no
    issue, so we never accuse a reference of being fake without confidence.
"""

from __future__ import annotations

from refcopilot.models import (
    HallucinationVerdict,
    Issue,
    IssueCategory,
    MergedRecord,
    Reference,
    Severity,
)
from refcopilot.verify.text_match import (
    author_overlap,
    is_garbled_title,
    title_similarity,
)
from refcopilot.verify.thresholds import (
    AUTHOR_FAKE_THRESHOLD,
    TITLE_FAKE_THRESHOLD,
    TITLE_SIMILARITY_THRESHOLD,
)


def pre_screen(
    reference: Reference,
    matches: list,
    merged: MergedRecord | None,
) -> HallucinationVerdict:
    """Return a tentative verdict before any LLM call."""
    if is_garbled_title(reference.title, reference.raw):
        return HallucinationVerdict.UNCERTAIN

    if not matches:
        # No candidates from any backend. If the cited paper has a working URL,
        # we treat as UNCERTAIN; otherwise LIKELY fake.
        if reference.url:
            return HallucinationVerdict.UNCERTAIN
        return HallucinationVerdict.LIKELY

    # We have at least one match; pick the best by title similarity.
    best = _best_by_title_similarity(reference.title, matches)
    if best is None:
        return HallucinationVerdict.UNCERTAIN

    sim = title_similarity(reference.title, best.title)
    if sim < TITLE_FAKE_THRESHOLD:
        return HallucinationVerdict.LIKELY

    overlap = author_overlap(reference.authors, best.authors)
    if overlap <= AUTHOR_FAKE_THRESHOLD and not reference.url:
        return HallucinationVerdict.LIKELY

    if sim >= TITLE_SIMILARITY_THRESHOLD:
        return HallucinationVerdict.UNLIKELY

    return HallucinationVerdict.UNCERTAIN


def _best_by_title_similarity(title: str | None, matches: list):
    if not title or not matches:
        return None
    best = None
    best_sim = -1.0
    for m in matches:
        sim = title_similarity(title, getattr(m, "title", None))
        if sim > best_sim:
            best_sim = sim
            best = m
    return best


def to_issue(verdict: HallucinationVerdict, reference: Reference, matches: list) -> Issue | None:
    """Convert a final verdict into an Issue (or None)."""
    if verdict != HallucinationVerdict.LIKELY:
        return None

    if not matches:
        return Issue(
            severity=Severity.ERROR,
            category=IssueCategory.FAKE,
            code="no_match",
            message="No matching paper found on arXiv or Semantic Scholar.",
            suggestion="Verify the citation; it may be fabricated.",
            confidence=0.9,
        )

    best = _best_by_title_similarity(reference.title, matches)
    sim = title_similarity(reference.title, best.title if best else None)
    overlap = author_overlap(reference.authors, best.authors if best else [])

    if sim < TITLE_FAKE_THRESHOLD:
        return Issue(
            severity=Severity.ERROR,
            category=IssueCategory.FAKE,
            code="title_mismatch",
            message=(
                f"Cited title does not match any retrieved record "
                f"(best similarity {sim:.2f} < {TITLE_FAKE_THRESHOLD})."
            ),
            suggestion=f"Closest match: {(best.title if best else 'n/a')[:160]}",
            confidence=0.9,
        )

    if overlap <= AUTHOR_FAKE_THRESHOLD:
        return Issue(
            severity=Severity.ERROR,
            category=IssueCategory.FAKE,
            code="author_mismatch",
            message=(
                f"Cited authors do not overlap with the retrieved record "
                f"(overlap {overlap:.2f} ≤ {AUTHOR_FAKE_THRESHOLD})."
            ),
            suggestion=f"Retrieved authors: {', '.join((best.authors if best else [])[:5])}",
            confidence=0.85,
        )

    return Issue(
        severity=Severity.ERROR,
        category=IssueCategory.FAKE,
        code="hallucination",
        message="Reference appears to be a hallucination.",
        confidence=0.7,
    )
