"""Incomplete reference detection.

Compares cited fields against the merged retrieved record and emits warnings
when a field is missing from the citation but present in the retrieved record.
"""

from __future__ import annotations

from refcopilot.models import (
    Issue,
    IssueCategory,
    MergedRecord,
    Reference,
    Severity,
)
from refcopilot.verify.thresholds import ET_AL_VARIANTS


def detect(reference: Reference, merged: MergedRecord | None) -> list[Issue]:
    issues: list[Issue] = []
    if merged is None:
        return issues

    if not reference.doi and merged.doi:
        issues.append(
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.INCOMPLETE,
                code="missing_doi",
                message="Citation is missing a DOI.",
                suggestion=f"Add doi: {merged.doi}",
                confidence=0.9,
            )
        )

    if not reference.arxiv_id and merged.arxiv_id:
        issues.append(
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.INCOMPLETE,
                code="missing_arxiv_id",
                message="Citation is missing an arXiv ID.",
                suggestion=f"Add arxiv: {merged.arxiv_id}",
                confidence=0.85,
            )
        )

    if not reference.year and merged.year:
        issues.append(
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.INCOMPLETE,
                code="missing_year",
                message="Citation is missing a year.",
                suggestion=f"Add year: {merged.year}",
                confidence=0.95,
            )
        )

    if not reference.venue and merged.venue:
        issues.append(
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.INCOMPLETE,
                code="missing_venue",
                message="Citation is missing a venue / journal.",
                suggestion=f"Add venue: {merged.venue}",
                confidence=0.85,
            )
        )

    if _truncated_authors(reference, merged):
        cited_count = len([a for a in reference.authors if not _is_et_al(a)])
        retrieved_count = len(merged.authors)
        issues.append(
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.INCOMPLETE,
                code="truncated_authors",
                message=(
                    f"Citation lists {cited_count} authors with et al.; "
                    f"the published record has {retrieved_count}."
                ),
                suggestion=f"List all {retrieved_count} authors.",
                confidence=0.7,
            )
        )

    if _abbreviated_venue(reference, merged):
        issues.append(
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.INCOMPLETE,
                code="abbreviated_venue",
                message=(
                    f"Cited venue '{reference.venue}' may be an abbreviation of "
                    f"'{merged.venue}'."
                ),
                suggestion=f"Use the full venue name: {merged.venue}.",
                confidence=0.6,
            )
        )

    return issues


def _is_et_al(value: str) -> bool:
    return value.strip().lower() in {v.lower() for v in ET_AL_VARIANTS}


def _truncated_authors(reference: Reference, merged: MergedRecord) -> bool:
    if not reference.authors or not merged.authors:
        return False
    has_et_al = any(_is_et_al(a) for a in reference.authors)
    cited_real = [a for a in reference.authors if not _is_et_al(a)]
    if not has_et_al:
        return False
    return len(merged.authors) >= len(cited_real) + 2


def _abbreviated_venue(reference: Reference, merged: MergedRecord) -> bool:
    cited = (reference.venue or "").strip()
    full = (merged.venue or "").strip()
    if not cited or not full:
        return False
    if cited.lower() == full.lower():
        return False
    if len(full) < len(cited) * 1.5:
        return False

    cited_tokens = {t for t in cited.lower().split() if len(t) > 1}
    full_tokens = {t for t in full.lower().split() if len(t) > 1}
    if cited_tokens and cited_tokens.issubset(full_tokens):
        return True

    # Subsequence test: 'neurips' is a subsequence of 'neuralinformationprocessingsystems'.
    cited_letters = "".join(c for c in cited.lower() if c.isalnum())
    full_letters = "".join(c for c in full.lower() if c.isalnum())
    if cited_letters and _is_subsequence(cited_letters, full_letters):
        return True

    return _initials_match(cited, full)


def _is_subsequence(short: str, long: str) -> bool:
    if not short or not long or len(short) > len(long):
        return False
    i = 0
    for ch in long:
        if i < len(short) and ch == short[i]:
            i += 1
    return i == len(short)


def _initials_match(short: str, long: str) -> bool:
    """e.g. 'ICML' matches 'International Conference on Machine Learning'."""
    initials = "".join(w[0].lower() for w in long.split() if w[:1].isalpha())
    short_alnum = "".join(c.lower() for c in short if c.isalnum())
    return bool(initials) and bool(short_alnum) and initials == short_alnum
