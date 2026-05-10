"""Input parsing & detection.

Replaces the per-input-type test_a_inputs/* files. Covers the public
``detect`` entry point and the BibTeX parser's two most load-bearing
behaviours: arXiv ID extraction (from eprint or url field), and lenient
parsing that doesn't crash on a malformed entry.
"""

from __future__ import annotations

import pytest

from refcopilot.inputs import bibtex
from refcopilot.inputs.detector import detect
from refcopilot.models import SourceFormat


def test_detect_classifies_url_bibtex_and_text() -> None:
    assert detect("https://arxiv.org/abs/1706.03762") is SourceFormat.URL
    assert detect("2401.12345") is SourceFormat.URL  # bare arXiv id
    assert detect("@article{x, title={T}, author={A}}") is SourceFormat.BIBTEX
    assert detect("Just plain text without bibtex markers.") is SourceFormat.TEXT
    with pytest.raises(ValueError):
        detect("")


def test_detect_classifies_files_by_suffix(tmp_path, fixtures_dir) -> None:
    bib = fixtures_dir / "inputs" / "minimal.bib"
    assert detect(str(bib)) is SourceFormat.BIBTEX

    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert detect(str(pdf)) is SourceFormat.PDF

    txt = tmp_path / "notes.txt"
    txt.write_text("hello", encoding="utf-8")
    assert detect(str(txt)) is SourceFormat.TEXT


def test_bibtex_parses_minimal_fixture(fixtures_dir) -> None:
    refs = bibtex.parse_file(fixtures_dir / "inputs" / "minimal.bib")
    by_key = {r.bibkey: r for r in refs}
    vaswani = by_key["vaswani2017attention"]
    # The key fields downstream search backends key off of: title, year,
    # arxiv_id (when present). If any of these regress the lookup misses.
    assert vaswani.title == "Attention Is All You Need"
    assert vaswani.year == 2017
    assert vaswani.arxiv_id == "1706.03762"
    assert "Ashish Vaswani" in vaswani.authors


def test_bibtex_extracts_arxiv_id_from_eprint_or_url() -> None:
    # Two of the three places a BibTeX entry can hide an arXiv ID; both must
    # surface as the same canonical id + version split. Title / DOI fields
    # exercise other parser paths and are covered by the minimal fixture.
    eprint_raw = "@misc{x, title={T}, author={A. B.}, eprint={2401.12345v3}, archivePrefix={arXiv}}"
    url_raw = "@misc{x, title={T}, author={A. B.}, url={https://arxiv.org/abs/2401.12345v2}}"
    [eprint_ref] = bibtex.parse_string(eprint_raw)
    [url_ref] = bibtex.parse_string(url_raw)
    assert eprint_ref.arxiv_id == "2401.12345"
    assert eprint_ref.arxiv_version == 3
    assert url_ref.arxiv_id == "2401.12345"
    assert url_ref.arxiv_version == 2


def test_bibtex_lenient_mode_recovers_around_malformed_entry() -> None:
    # The parser must not crash on user input; at least the first valid
    # entry (before the broken one) must come back so the run can proceed.
    raw = """
    @article{good1, title={T1}, author={A. B.}, year={2020}}

    @article{broken,
      title = {missing closing brace
      author = {A. B.},

    @article{good2, title={T2}, author={C. D.}, year={2021}}
    """
    refs = bibtex.parse_string(raw)
    keys = {r.bibkey for r in refs}
    assert "good1" in keys
    assert refs  # at least one entry survives
