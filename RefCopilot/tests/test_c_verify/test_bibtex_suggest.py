"""Suggested-BibTeX rendering."""

from __future__ import annotations

from refcopilot.bibtex_suggest import suggest_bibtex
from refcopilot.models import (
    Backend,
    ExternalRecord,
    MergedRecord,
    Reference,
    SourceFormat,
)


def _attention_merged() -> MergedRecord:
    return MergedRecord(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue="NeurIPS 2017",
        doi="10.5555/3295222.3295349",
        arxiv_id="1706.03762",
        latest_arxiv_version=7,
        url="https://arxiv.org/abs/1706.03762v7",
        provenance={
            "title": Backend.ARXIV,
            "authors": Backend.ARXIV,
            "year": Backend.ARXIV,
            "venue": Backend.SEMANTIC_SCHOLAR,
            "doi": Backend.SEMANTIC_SCHOLAR,
            "arxiv_id": Backend.ARXIV,
        },
        sources=[
            ExternalRecord(
                backend=Backend.ARXIV,
                record_id="1706.03762",
                title="Attention Is All You Need",
                url="https://arxiv.org/abs/1706.03762v7",
            ),
            ExternalRecord(
                backend=Backend.SEMANTIC_SCHOLAR,
                record_id="abc",
                title="Attention Is All You Need",
                url="https://www.semanticscholar.org/paper/abc",
            ),
        ],
    )


def test_returns_empty_when_no_merged_record():
    ref = Reference(raw="x", source_format=SourceFormat.TEXT, title="T")
    assert suggest_bibtex(ref, None) == ""


def test_includes_provenance_comment_per_backend():
    ref = Reference(raw="@article{vaswani2017,}", source_format=SourceFormat.BIBTEX, bibkey="vaswani2017")
    out = suggest_bibtex(ref, _attention_merged())
    assert "% Suggested by RefCopilot. Field provenance:" in out
    assert "%   arxiv:" in out
    assert "%   semantic_scholar:" in out
    # Each backend's URL is annotated for auditing.
    assert "https://arxiv.org/abs/1706.03762v7" in out
    assert "https://www.semanticscholar.org/paper/abc" in out


def test_preserves_bibkey_and_entry_type_from_original():
    ref = Reference(raw="@inproceedings{vaswani2017,}", source_format=SourceFormat.BIBTEX, bibkey="vaswani2017")
    out = suggest_bibtex(ref, _attention_merged())
    assert out.find("@inproceedings{vaswani2017,") >= 0


def test_picks_inproceedings_for_conference_venue_without_bibtex_hint():
    ref = Reference(raw="", source_format=SourceFormat.TEXT, title="X")
    merged = MergedRecord(
        title="X",
        authors=["A B"],
        year=2024,
        venue="Proceedings of the International Conference on ML",
        provenance={"title": Backend.ARXIV},
    )
    out = suggest_bibtex(ref, merged)
    assert out.startswith("% Suggested by RefCopilot")
    assert "@inproceedings{" in out
    assert "booktitle = {Proceedings of the International Conference on ML}" in out


def test_arxiv_only_uses_misc_with_eprint():
    ref = Reference(raw="", source_format=SourceFormat.TEXT, title="Y")
    merged = MergedRecord(
        title="Y",
        authors=["X Y"],
        year=2023,
        arxiv_id="2301.00001",
        url="https://arxiv.org/abs/2301.00001",
        provenance={"title": Backend.ARXIV, "arxiv_id": Backend.ARXIV},
    )
    out = suggest_bibtex(ref, merged)
    assert "@misc{" in out
    assert "eprint = {2301.00001}" in out
    assert "archivePrefix = {arXiv}" in out


def test_generated_bibkey_when_reference_has_none():
    ref = Reference(raw="", source_format=SourceFormat.TEXT, title="Y")
    out = suggest_bibtex(ref, _attention_merged())
    # bibkey defaults to lowercase surname+year+title-word
    assert "@article{vaswani2017" in out or "@inproceedings{vaswani2017" in out


def test_returns_empty_when_merged_has_no_usable_fields():
    ref = Reference(raw="", source_format=SourceFormat.TEXT)
    merged = MergedRecord(title="")
    assert suggest_bibtex(ref, merged) == ""
