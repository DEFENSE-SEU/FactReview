"""Tests for the reference-checking adapter."""

from __future__ import annotations

import sys
from pathlib import Path


def test_refcheck_adapter_importable():
    from fact_generation.refcheck.refcheck import check_references, format_reference_check_markdown

    assert callable(check_references)
    assert callable(format_reference_check_markdown)


_REFCOPILOT_SRC = Path(__file__).resolve().parents[4] / "RefCopilot" / "src"
if str(_REFCOPILOT_SRC) not in sys.path:
    sys.path.insert(0, str(_REFCOPILOT_SRC))


def test_refcheck_returns_expected_schema(monkeypatch):
    """The adapter forwards RefCopilot's output dict with the documented keys."""
    from refcopilot.models import (
        CheckedReference,
        Reference,
        Report,
        ReportSummary,
        SourceFormat,
        Verdict,
    )

    def _fake_run(self, spec, **_kwargs):
        ref = Reference(raw="x", source_format=SourceFormat.BIBTEX, title="T", authors=["A"])
        return Report(
            paper={"input": spec, "kind": "bibtex"},
            checked=[CheckedReference(reference=ref, verdict=Verdict.VALID)],
            summary=ReportSummary(total_refs=1, errors=0, warnings=0, unverified=0, by_category={}),
        )

    monkeypatch.setattr("refcopilot.pipeline.RefCopilotPipeline.run", _fake_run)
    from fact_generation.refcheck.refcheck import check_references

    result = check_references(paper="dummy.bib")

    expected_keys = {
        "ok", "total_refs", "errors", "warnings", "unverified",
        "error_message", "issues", "error_details", "warning_details",
        "unverified_details", "report_file",
    }
    assert expected_keys <= set(result.keys())
    assert result["ok"] is True
    assert result["total_refs"] == 1


def test_refcheck_empty_input_returns_failure_dict():
    """Genuinely unparseable input surfaces as ``ok=False`` with an error message."""
    from fact_generation.refcheck.refcheck import check_references

    result = check_references(paper="")
    assert isinstance(result, dict)
    assert result["ok"] is False
    assert result["error_message"]


def test_format_reference_check_markdown_renders_warnings_with_bibtex():
    """The embedded summary now lists errors AND warnings; warnings carry a
    corrected-BibTeX block so users can paste a fix straight into their .bib.
    Unverified entries are still suppressed."""
    from fact_generation.refcheck.refcheck import format_reference_check_markdown

    result = {
        "ok": True,
        "total_refs": 3,
        "errors": 1,
        "warnings": 1,
        "unverified": 1,
        "report_file": "/tmp/reference_check_details.txt",
        "issues": [
            {
                "severity": "error",
                "type": "hallucination::no_match",
                "reference_title": "Fake paper",
                "details": "No matching paper found.",
                "raw_reference": "[1] Fake paper. 2024.",
                "corrected_bibtex": "",
            },
            {
                "severity": "warning",
                "type": "incomplete::missing_doi",
                "reference_title": "Real paper",
                "details": "Citation is missing a DOI.",
                "corrected_bibtex": (
                    "% Suggested by RefCopilot. Field provenance:\n"
                    "%   semantic_scholar: doi — https://www.semanticscholar.org/paper/abc\n"
                    "@article{realpaper2024,\n"
                    "  title = {Real paper},\n"
                    "  doi = {10.1/x},\n"
                    "}"
                ),
            },
            {
                "severity": "unverified",
                "type": "unverified::no_match",
                "reference_title": "Mystery paper",
                "details": "Could not verify reference.",
                "corrected_bibtex": "",
            },
        ],
    }

    markdown = format_reference_check_markdown(result)

    assert "## Reference Check" in markdown
    assert "### Errors" in markdown
    assert "Fake paper" in markdown
    assert "### Warnings" in markdown
    assert "Real paper" in markdown
    assert "```bibtex" in markdown
    assert "Suggested by RefCopilot" in markdown
    assert "Mystery paper" not in markdown


def test_format_reference_check_markdown_handles_failure():
    from fact_generation.refcheck.refcheck import format_reference_check_markdown

    result = {"ok": False, "error_message": "boom", "report_file": ""}
    markdown = format_reference_check_markdown(result)
    assert "did not complete successfully" in markdown
    assert "boom" in markdown
