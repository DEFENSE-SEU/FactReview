"""Minimal tests for the reference-checking adapter."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_refcheck_adapter_importable():
    """The adapter module must be importable without heavy refchecker deps."""
    from fact_generation.refcheck.refcheck import check_references

    assert callable(check_references)


_REFCOPILOT_FIXTURES = (
    Path(__file__).resolve().parents[4] / "tools" / "RefCopilot" / "tests" / "fixtures" / "inputs"
)


@pytest.mark.skipif(
    not (_REFCOPILOT_FIXTURES / "minimal.bib").exists(),
    reason="RefCopilot fixtures not present",
)
def test_refcheck_dispatches_to_refcopilot(monkeypatch):
    """Default behavior with FACTREVIEW_USE_REFCOPILOT=1 (or unset)."""
    monkeypatch.setenv("FACTREVIEW_USE_REFCOPILOT", "1")
    from fact_generation.refcheck.refcheck import check_references

    bib = _REFCOPILOT_FIXTURES / "minimal.bib"
    result = check_references(paper=str(bib), max_workers=1)

    expected_keys = {
        "ok", "total_refs", "errors", "warnings", "unverified",
        "error_message", "issues", "error_details", "warning_details",
        "unverified_details", "report_file",
    }
    assert expected_keys <= set(result.keys())
    assert isinstance(result["issues"], list)
    assert result["ok"] is True
    assert result["total_refs"] >= 0


def test_refcheck_legacy_flag_falls_back_to_refchecker(monkeypatch):
    """When FACTREVIEW_USE_REFCOPILOT=0, the adapter goes through the refchecker submodule path.

    The submodule may be uninitialized in CI; in that case the call should still
    return a graceful error dict (ok=False).
    """
    monkeypatch.setenv("FACTREVIEW_USE_REFCOPILOT", "0")
    from fact_generation.refcheck.refcheck import check_references

    result = check_references(paper="/nonexistent/paper.pdf")
    assert isinstance(result, dict)
    assert "ok" in result
    assert "error_message" in result


def test_refcheck_nonexistent_paper():
    """Passing a non-existent file should return ok=False gracefully."""
    try:
        import arxiv  # noqa: F401
    except ImportError:
        pytest.skip("refchecker deps (arxiv) not installed")

    from fact_generation.refcheck.refcheck import check_references

    result = check_references(paper="/nonexistent/paper.pdf")
    # The adapter calls refchecker which will fail; should not raise.
    assert isinstance(result, dict)
    assert "ok" in result
    assert "error_message" in result


def test_refchecker_package_importable_when_deps_installed():
    """The upstream refchecker package should import when its own deps are installed."""
    try:
        import refchecker
    except ModuleNotFoundError as exc:
        pytest.skip(f"refchecker optional dependency missing: {exc.name}")

    assert hasattr(refchecker, "__version__")
    assert refchecker.__version__


def test_refcheck_extracts_severity_from_original_issue_fields():
    from fact_generation.refcheck.refcheck import _extract_issues

    class Checker:
        def __init__(self):
            self.errors = [
                {
                    "ref_title": "Author mismatch",
                    "error_type": "author",
                    "error_details": "Author list differs.",
                    "_original_errors": [{"error_type": "author", "error_details": "Author list differs."}],
                },
                {
                    "ref_title": "Year mismatch",
                    "error_type": "year",
                    "error_details": "Year differs.",
                    "_original_errors": [{"warning_type": "year", "warning_details": "Year differs."}],
                },
                {
                    "ref_title": "Mixed issue",
                    "error_type": "multiple",
                    "error_details": "- DOI differs.\n- Venue differs.",
                    "_original_errors": [
                        {"error_type": "doi", "error_details": "DOI differs."},
                        {"warning_type": "venue", "warning_details": "Venue differs."},
                    ],
                },
                {
                    "ref_title": "Not found",
                    "error_type": "unverified",
                    "error_details": "Could not verify reference.",
                    "_original_errors": [
                        {"error_type": "unverified", "error_details": "Could not verify reference."}
                    ],
                },
            ]

    issues = _extract_issues(Checker())
    by_severity = {}
    for issue in issues:
        by_severity.setdefault(issue["severity"], []).append(issue["type"])

    assert by_severity["error"] == ["author", "doi"]
    assert by_severity["warning"] == ["year", "venue"]
    assert by_severity["unverified"] == ["unverified"]


def test_refcheck_markdown_includes_warning_and_error_details():
    from fact_generation.refcheck.refcheck import format_reference_check_markdown

    result = {
        "ok": True,
        "total_refs": 3,
        "errors": 1,
        "warnings": 1,
        "unverified": 0,
        "report_file": "/tmp/reference_check_details.txt",
        "issues": [
            {
                "severity": "error",
                "type": "title",
                "reference_title": "Wrong title",
                "details": "Title mismatch: cited A but verified B.",
                "raw_reference": "A. Wrong title. 2024.",
            },
            {
                "severity": "warning",
                "type": "year",
                "reference_title": "Year issue",
                "details": "Year mismatch: cited as 2023 but actually 2024.",
            },
        ],
    }

    markdown = format_reference_check_markdown(result)

    assert "## Reference Check" in markdown
    assert "### Errors" in markdown
    assert "Title mismatch" in markdown
    assert "### Warnings" in markdown
    assert "Year mismatch" in markdown
