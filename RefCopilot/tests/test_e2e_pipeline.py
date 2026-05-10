"""End-to-end RefCopilot adapter smoke tests.

Marked ``e2e`` and skipped by default. Verifies the FactReview-facing
adapter (``refcopilot.factreview.check_references`` +
``format_factreview_markdown``) wraps the pipeline correctly and produces
the documented JSON / Markdown shapes that FactReview's refcheck stage
consumes.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


def test_check_references_runs_via_factreview_adapter(tmp_path, fixtures_dir, monkeypatch) -> None:
    # We don't want to spin up real backends here — that's exercised by the
    # individual verify tests with FakeBackend. Instead, swap in a stub
    # ``RefCopilotPipeline`` that returns a canned ``Report`` so the adapter
    # path (extract → run → to_factreview_dict → write text report) is
    # validated end-to-end.
    from refcopilot import factreview as fr_mod
    from refcopilot.models import (
        CheckedReference,
        Issue,
        IssueCategory,
        Reference,
        Report,
        ReportSummary,
        Severity,
        SourceFormat,
        Verdict,
    )

    captured: dict[str, object] = {}

    class _StubPipeline:
        def __init__(self, **kw):
            captured["init_kwargs"] = kw

        def run(self, spec, **kw):
            captured["spec"] = spec
            return Report(
                checked=[
                    CheckedReference(
                        reference=Reference(
                            raw="x",
                            source_format=SourceFormat.BIBTEX,
                            title="Fake et al. 2025",
                        ),
                        issues=[
                            Issue(
                                severity=Severity.ERROR,
                                category=IssueCategory.FAKE,
                                code="no_match",
                                message="no match found",
                            )
                        ],
                        verdict=Verdict.ERROR,
                    )
                ],
                summary=ReportSummary(total_refs=1, errors=1),
            )

    monkeypatch.setattr(fr_mod, "RefCopilotPipeline", _StubPipeline)

    output_text = tmp_path / "refs_out.txt"
    result = fr_mod.check_references(
        paper=str(fixtures_dir / "inputs" / "minimal.bib"),
        output_file=str(output_text),
    )

    assert result["ok"] is True
    assert result["errors"] == 1
    assert result["report_file"] == str(output_text)
    # The adapter must thread the file path through to the underlying pipeline.
    assert captured["spec"].endswith("minimal.bib")


def test_factreview_markdown_renders_errors_and_warnings_with_corrections() -> None:
    from refcopilot.factreview import format_factreview_markdown

    payload = {
        "ok": True,
        "total_refs": 2,
        "errors": 1,
        "warnings": 1,
        "unverified": 0,
        "issues": [
            {
                "severity": "error",
                "reference_title": "Fake 2024",
                "type": "fake::no_match",
                "details": "no academic match",
            },
            {
                "severity": "warning",
                "reference_title": "Real 2017",
                "type": "incomplete::missing_doi",
                "details": "DOI absent",
                "corrected_bibtex": "@inproceedings{real2017,\n  doi = {10.x/y}\n}",
            },
        ],
    }
    md = format_factreview_markdown(payload, include_warnings=True)
    assert "Reference Check" in md
    assert "Fake 2024" in md
    # FactReview-specific override: warnings appear with their corrected
    # BibTeX block. Without this, the embedded summary loses the actionable
    # part of the report.
    assert "Real 2017" in md
    assert "@inproceedings{real2017" in md
