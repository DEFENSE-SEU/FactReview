"""Retraction detection.

A retracted citation is a credibility-destroying issue distinct from "outdated"
(where the cited paper still stands but a newer version exists). Sources that
populate ``MergedRecord.is_retracted``:

  - arXiv ``"this paper has been withdrawn"`` notice.
  - OpenAlex ``is_retracted`` field (covers publisher retractions sourced from
    CrossRef + Retraction Watch).
"""

from __future__ import annotations

from refcopilot.models import (
    Issue,
    IssueCategory,
    MergedRecord,
    Reference,
    Severity,
)


def detect(reference: Reference, merged: MergedRecord | None) -> list[Issue]:
    if merged is None or not merged.is_retracted:
        return []
    return [
        Issue(
            severity=Severity.ERROR,
            category=IssueCategory.RETRACTED,
            code="is_retracted",
            message="The cited paper has been retracted (or withdrawn by the author).",
            suggestion=(
                "Remove this citation or replace it with a non-retracted source. "
                "Verify the retraction notice on the publisher's site."
            ),
            confidence=0.95,
        )
    ]
