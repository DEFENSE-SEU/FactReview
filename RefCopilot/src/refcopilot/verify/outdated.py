"""Outdated reference detection.

Three categories:
  - arxiv_published   : cited as arXiv preprint but a real venue publication exists.
  - old_version       : cited arXiv vN, but a newer arXiv version exists.
  - workshop_promoted : cited workshop venue, but the same paper exists at a
                        full conference / journal.

Retraction (formerly the ``withdrawn`` case) lives in ``verify/retraction.py``
and is reported as an error rather than a warning.
"""

from __future__ import annotations

from refcopilot.models import (
    Issue,
    IssueCategory,
    MergedRecord,
    Reference,
    Severity,
)
from refcopilot.verify.thresholds import ARXIV_VENUE_ALIASES


def detect(reference: Reference, merged: MergedRecord | None) -> list[Issue]:
    issues: list[Issue] = []
    if merged is None:
        return issues

    if _cited_as_arxiv(reference) and _has_real_venue(merged):
        issues.append(
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.OUTDATED,
                code="arxiv_published",
                message=(
                    f"Paper was published at venue '{merged.venue}' but cited as an arXiv preprint."
                ),
                suggestion=f"Cite the published version: {merged.venue}.",
                confidence=0.85,
            )
        )

    if reference.arxiv_id and reference.arxiv_version and merged.latest_arxiv_version:
        if merged.latest_arxiv_version > reference.arxiv_version:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category=IssueCategory.OUTDATED,
                    code="old_version",
                    message=(
                        f"Cited arXiv version v{reference.arxiv_version} is older than "
                        f"the latest v{merged.latest_arxiv_version}."
                    ),
                    suggestion=f"Update to v{merged.latest_arxiv_version}.",
                    confidence=0.8,
                )
            )

    if _cited_workshop(reference) and _real_venue_is_full(merged, reference):
        issues.append(
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.OUTDATED,
                code="workshop_promoted",
                message=(
                    f"Cited workshop version '{reference.venue}', but a full version "
                    f"appears at '{merged.venue}'."
                ),
                suggestion=f"Cite the full version at {merged.venue}.",
                confidence=0.7,
            )
        )

    return issues


def _cited_as_arxiv(reference: Reference) -> bool:
    if reference.arxiv_id and not reference.doi:
        return True
    venue = (reference.venue or "").strip().lower()
    return venue in ARXIV_VENUE_ALIASES


def _has_real_venue(merged: MergedRecord) -> bool:
    venue = (merged.venue or "").strip().lower()
    if not venue:
        return False
    return venue not in ARXIV_VENUE_ALIASES


def _cited_workshop(reference: Reference) -> bool:
    return "workshop" in (reference.venue or "").lower()


def _real_venue_is_full(merged: MergedRecord, reference: Reference) -> bool:
    if not merged.venue or "workshop" in merged.venue.lower():
        return False
    cited_lower = (reference.venue or "").lower()
    venue_lower = merged.venue.lower()
    return cited_lower != venue_lower
