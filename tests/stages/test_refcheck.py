"""Refcheck stage tests.

The refcheck stage delegates entirely to RefCopilot. The two tests here pin
down the FactReview-side adapter contract: ``check_references`` returns the
documented dict shape, and ``format_reference_check_markdown`` includes
warnings (the FactReview-specific override of RefCopilot's errors-only
default). The deeper RefCopilot pipeline behaviour lives in
``RefCopilot/tests``.
"""

from __future__ import annotations

from fact_generation.refcheck.refcheck import (
    check_references,
    format_reference_check_markdown,
)


def test_check_references_returns_documented_schema(tmp_path, monkeypatch) -> None:
    # Stub the underlying RefCopilot adapter so this test stays fast and
    # offline; we're verifying the FactReview shim's contract, not RefCopilot.
    fake_payload = {
        "ok": True,
        "total_refs": 3,
        "errors": 1,
        "warnings": 1,
        "unverified": 0,
        "error_message": "",
        "issues": [],
        "error_details": [{"reference": "Fake et al. 2024", "code": "no_match"}],
        "warning_details": [{"reference": "Real 2017", "code": "missing_doi", "corrected_bibtex": "@x{...}"}],
        "unverified_details": [],
        "report_file": "",
    }

    captured: dict[str, object] = {}

    def fake_check(
        paper, *, api_key=None, output_file=None, debug=False, enable_parallel=True, max_workers=4
    ):  # type: ignore[no-untyped-def]
        captured["paper"] = paper
        captured["max_workers"] = max_workers
        return fake_payload

    monkeypatch.setattr("refcopilot.factreview.check_references", fake_check)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    result = check_references(paper=str(pdf), max_workers=2)

    assert captured == {"paper": str(pdf), "max_workers": 2}
    # Keys the report_stage_runner reads when composing the final review.
    for key in (
        "ok",
        "total_refs",
        "errors",
        "warnings",
        "unverified",
        "error_details",
        "warning_details",
    ):
        assert key in result


def test_format_markdown_includes_warnings_for_factreview() -> None:
    payload = {
        "ok": True,
        "enabled": True,
        "total_refs": 2,
        "errors": 1,
        "warnings": 1,
        "unverified": 0,
        "issues": [
            {
                "severity": "error",
                "reference_title": "Fake et al. 2024",
                "type": "fake",
                "details": "no match found",
            },
            {
                "severity": "warning",
                "reference_title": "Real 2017",
                "type": "missing_doi",
                "details": "DOI absent",
                "corrected_bibtex": "@inproceedings{real2017,\n  doi = {10.x/y}\n}",
            },
        ],
    }

    md = format_reference_check_markdown(payload)
    # FactReview's adapter overrides RefCopilot's default to include warnings;
    # if this regresses the embedded summary loses the corrected-BibTeX block.
    assert "Reference Check" in md
    assert "Fake et al. 2024" in md
    assert "Real 2017" in md
    assert "@inproceedings{real2017" in md
