"""RefCopilot schema contract.

Pins down the shapes the FactReview adapter and on-disk reports depend on:
the per-field provenance map on ``MergedRecord``, the ``is_retracted`` flag
that carries the retraction signal across backends, and the
``to_factreview_dict`` output shape.
"""

from __future__ import annotations

from refcopilot.models import (
    Backend,
    CheckedReference,
    ExternalRecord,
    Issue,
    IssueCategory,
    MergedRecord,
    Reference,
    Report,
    ReportSummary,
    Severity,
    SourceFormat,
    Verdict,
)
from refcopilot.report import to_factreview_dict


def test_reference_and_issue_round_trip() -> None:
    ref = Reference(
        raw="@article{x, year={2020}}",
        source_format=SourceFormat.BIBTEX,
        bibkey="x",
        title="T",
        year=2020,
    )
    issue = Issue(
        severity=Severity.WARNING,
        category=IssueCategory.INCOMPLETE,
        code="missing_doi",
        message="DOI absent",
    )
    rehydrated_ref = Reference.model_validate(ref.model_dump())
    rehydrated_issue = Issue.model_validate(issue.model_dump())
    assert rehydrated_ref.source_format is SourceFormat.BIBTEX
    assert rehydrated_issue.severity is Severity.WARNING
    assert rehydrated_issue.category is IssueCategory.INCOMPLETE


def test_merged_record_carries_retraction_and_provenance() -> None:
    src = ExternalRecord(
        backend=Backend.OPENALEX,
        record_id="W123",
        title="A retracted paper",
        authors=["X"],
        year=2020,
        doi="10.x/retracted",
        is_retracted=True,
    )
    merged = MergedRecord(
        title="A retracted paper",
        authors=["X"],
        year=2020,
        doi="10.x/retracted",
        is_retracted=True,
        provenance={"title": Backend.OPENALEX, "doi": Backend.OPENALEX},
        sources=[src],
    )
    rehydrated = MergedRecord.model_validate(merged.model_dump())
    # is_retracted is the unified signal that ``verify.retraction`` keys off of;
    # losing it here means the retraction guard regresses silently.
    assert rehydrated.is_retracted is True
    assert rehydrated.provenance["title"] is Backend.OPENALEX
    assert rehydrated.sources[0].backend is Backend.OPENALEX


def test_factreview_dict_top_level_and_issue_keys() -> None:
    # Documents the shape FactReview's report stage_runner reads. Adding /
    # removing a top-level key without updating the FactReview side breaks
    # the embedded summary block; this test catches that immediately.
    report = Report(
        checked=[
            CheckedReference(
                reference=Reference(raw="x", source_format=SourceFormat.BIBTEX, title="T"),
                issues=[
                    Issue(
                        severity=Severity.ERROR,
                        category=IssueCategory.FAKE,
                        code="no_match",
                        message="No match.",
                    )
                ],
                verdict=Verdict.ERROR,
            ),
        ],
        summary=ReportSummary(total_refs=1, errors=1),
    )
    payload = to_factreview_dict(report, report_file="/tmp/x.txt")
    assert {
        "ok",
        "total_refs",
        "errors",
        "warnings",
        "unverified",
        "issues",
        "error_details",
        "warning_details",
        "unverified_details",
        "report_file",
    } <= set(payload)
    issue = payload["issues"][0]
    assert issue["severity"] == "error"
    assert "::" in issue["type"]  # "<category>::<code>"
    assert payload["report_file"] == "/tmp/x.txt"
