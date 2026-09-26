"""Hallucination (fake reference) detection.

Two-stage:

1.  :func:`pre_screen` — heuristic verdict (no LLM call) based on the shared
    anchor found by :mod:`refcopilot.verify.matching`, plus OCR-garbled-title
    detection.
2.  :func:`to_issue` — given a final verdict, emit an :class:`Issue` only when
    the verdict is ``LIKELY``. ``UNLIKELY`` and ``UNCERTAIN`` produce no
    issue, so we never accuse a reference of being fake without confidence.
"""

from __future__ import annotations

from refcopilot.models import (
    ExternalRecord,
    HallucinationVerdict,
    Issue,
    IssueCategory,
    Reference,
    Severity,
)
from refcopilot.verify.text_match import (
    is_garbled_title,
    title_similarity,
    titles_match,
)


def pre_screen(
    reference: Reference,
    matches: list,
    anchor: ExternalRecord | None,
) -> HallucinationVerdict:
    """Return a tentative verdict before any LLM call.

    ``anchor`` is the record :func:`refcopilot.verify.matching.find_anchor`
    picked for this reference (exact title and ordered authors). With
    candidates retrieved, the verdict is decided: an anchor means real, no
    anchor means fake.
    """
    if anchor is not None:
        return HallucinationVerdict.UNLIKELY

    if is_garbled_title(reference.title, reference.raw):
        return HallucinationVerdict.UNCERTAIN

    if not matches and reference.url:
        return HallucinationVerdict.UNCERTAIN
    return HallucinationVerdict.LIKELY


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

    same_title = next((m for m in matches if titles_match(reference.title, m.title)), None)
    if same_title is not None:
        return Issue(
            severity=Severity.ERROR,
            category=IssueCategory.FAKE,
            code="author_mismatch",
            message=(
                "A paper with this title exists, but no retrieved version has the "
                "cited author list (names and order)."
            ),
            suggestion=f"Retrieved authors: {', '.join(same_title.authors[:10])}",
            confidence=0.85,
        )

    closest = max(matches, key=lambda m: title_similarity(reference.title, m.title))
    return Issue(
        severity=Severity.ERROR,
        category=IssueCategory.FAKE,
        code="title_mismatch",
        message="No retrieved record has the cited title.",
        suggestion=f"Closest match: {closest.title[:160]}",
        confidence=0.9,
    )
