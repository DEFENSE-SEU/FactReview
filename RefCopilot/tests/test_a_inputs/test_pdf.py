"""PDF text extraction and bibliography section detection."""

from __future__ import annotations

from refcopilot.inputs import pdf


def test_find_bibliography_picks_last_after_20_percent():
    text = "x" * 100 + "\nReferences\n" + "y" * 1000 + "\n5. References\n[1] First ref\n[2] Second ref\n"
    bib = pdf.find_bibliography(text)
    assert "[1] First ref" in bib
    assert "[2] Second ref" in bib


def test_find_bibliography_returns_empty_when_missing():
    text = "intro\n\nbody without any bibliography header"
    assert pdf.find_bibliography(text) == ""


def test_find_bibliography_truncates_at_appendix():
    body = "filler " * 200
    text = (
        body
        + "\n5. References\n"
        + "[1] Smith. A paper. 2020.\n[2] Jones. Another paper. 2021.\n"
        + "\nA Appendix\n"
        + "appendix content here\n"
    )
    bib = pdf.find_bibliography(text)
    assert "Smith" in bib
    assert "appendix content here" not in bib


def test_find_bibliography_truncates_at_acknowledgments():
    body = "filler " * 200
    text = (
        body
        + "\n5. References\n"
        + "[1] Smith. A paper. 2020.\n"
        + "\nAcknowledgments\n"
        + "thanks reviewers\n"
    )
    bib = pdf.find_bibliography(text)
    assert "Smith" in bib
    assert "thanks reviewers" not in bib


def test_find_bibliography_skips_early_match():
    # "References" inside a TOC at the very start should be ignored.
    text = (
        "TOC\nReferences ............ 9\n"
        + "filler " * 500
        + "\nReferences\n[1] Paper A. 2020.\n"
    )
    bib = pdf.find_bibliography(text)
    assert "[1] Paper A. 2020." in bib


