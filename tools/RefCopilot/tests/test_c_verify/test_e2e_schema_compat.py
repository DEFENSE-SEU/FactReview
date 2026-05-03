"""Test C.6 — output schema compatibility with FactReview's reference_check.json."""

from __future__ import annotations

from refcopilot.models import (
    Backend,
    CheckedReference,
    ExternalRecord,
    Issue,
    IssueCategory,
    Reference,
    Report,
    ReportSummary,
    Severity,
    SourceFormat,
    Verdict,
)
from refcopilot.report import to_legacy_dict


_LEGACY_TOP_KEYS = {
    "ok",
    "total_refs",
    "errors",
    "warnings",
    "unverified",
    "error_message",
    "issues",
    "error_details",
    "warning_details",
    "unverified_details",
    "report_file",
}

_LEGACY_ISSUE_KEYS = {
    "severity",
    "type",
    "reference_title",
    "reference_year",
    "cited_url",
    "verified_url",
    "details",
    "raw_reference",
    "corrected_plaintext",
    "corrected_bibtex",
    "corrected_bibitem",
}


def _sample_report() -> Report:
    ref = Reference(
        raw="@article{x, title={T}, year={2020}}",
        source_format=SourceFormat.BIBTEX,
        bibkey="x",
        title="T",
        authors=["A. B."],
        year=2020,
    )
    issue_err = Issue(
        severity=Severity.ERROR,
        category=IssueCategory.FAKE,
        code="no_match",
        message="No match found.",
    )
    issue_warn = Issue(
        severity=Severity.WARNING,
        category=IssueCategory.INCOMPLETE,
        code="missing_doi",
        message="Citation is missing a DOI.",
        suggestion="Add doi: 10.x/y",
    )
    return Report(
        paper={"input": "x.bib", "kind": "bibtex"},
        checked=[
            CheckedReference(reference=ref, issues=[issue_err], verdict=Verdict.ERROR),
            CheckedReference(
                reference=ref,
                matches=[
                    ExternalRecord(
                        backend=Backend.SEMANTIC_SCHOLAR,
                        record_id="abc",
                        title="T",
                        url="https://example.com",
                    )
                ],
                issues=[issue_warn],
                verdict=Verdict.WARNING,
            ),
        ],
        summary=ReportSummary(
            total_refs=2, errors=1, warnings=1, unverified=0,
            by_category={"fake": 1, "incomplete": 1},
        ),
    )


def test_legacy_top_keys_present():
    payload = to_legacy_dict(_sample_report(), report_file="/tmp/details.txt")
    assert _LEGACY_TOP_KEYS <= set(payload.keys())


def test_legacy_issue_keys_present():
    payload = to_legacy_dict(_sample_report())
    for issue in payload["issues"]:
        assert _LEGACY_ISSUE_KEYS <= set(issue.keys())


def test_legacy_severity_strings():
    payload = to_legacy_dict(_sample_report())
    severities = {i["severity"] for i in payload["issues"]}
    # Allowed values per stage_runner.py contract
    assert severities <= {"error", "warning", "unverified"}


def test_legacy_grouped_details_consistent():
    payload = to_legacy_dict(_sample_report())
    assert len(payload["error_details"]) == payload["errors"]
    assert len(payload["warning_details"]) == payload["warnings"]


def test_legacy_dict_includes_report_file():
    payload = to_legacy_dict(_sample_report(), report_file="/tmp/x.txt")
    assert payload["report_file"] == "/tmp/x.txt"


def test_legacy_type_label_format():
    payload = to_legacy_dict(_sample_report())
    types = {i["type"] for i in payload["issues"]}
    # type label format is "<category>::<code>"
    assert any("::" in t for t in types)
