"""Input type detection."""

from __future__ import annotations

import pytest

from refcopilot.inputs.detector import detect
from refcopilot.models import SourceFormat


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("https://arxiv.org/abs/1706.03762", SourceFormat.URL),
        ("http://example.com/paper", SourceFormat.URL),
        ("arxiv:2401.12345", SourceFormat.URL),
        ("2401.12345", SourceFormat.URL),
        ("2401.12345v3", SourceFormat.URL),
        ("@article{x, title={T}, author={A}}", SourceFormat.BIBTEX),
        ("Just some plain text without bibtex markers.", SourceFormat.TEXT),
    ],
)
def test_detect_from_strings(spec, expected):
    assert detect(spec) == expected


def test_detect_from_files(tmp_path, fixtures_dir):
    bib = fixtures_dir / "inputs" / "minimal.bib"
    assert detect(str(bib)) == SourceFormat.BIBTEX

    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert detect(str(pdf)) == SourceFormat.PDF

    txt = tmp_path / "notes.txt"
    txt.write_text("hello", encoding="utf-8")
    assert detect(str(txt)) == SourceFormat.TEXT


def test_detect_rejects_empty():
    with pytest.raises(ValueError):
        detect("")
