"""Test B.5 — multi-source merge."""

from __future__ import annotations

from refcopilot.merge import merge_records
from refcopilot.models import Backend, ExternalRecord


def _arxiv_record(**kw):
    defaults = dict(
        backend=Backend.ARXIV,
        record_id="1706.03762",
        title="Attention Is All You Need (arXiv title)",
        authors=["A. Vaswani", "N. Shazeer"],
        year=2017,
        venue=None,
        publication_venue=None,
        journal=None,
        doi=None,
        arxiv_id="1706.03762",
        latest_arxiv_version=7,
        arxiv_versions=[7],
        withdrawn=False,
        url="https://arxiv.org/abs/1706.03762",
    )
    defaults.update(kw)
    return ExternalRecord(**defaults)


def _s2_record(**kw):
    defaults = dict(
        backend=Backend.SEMANTIC_SCHOLAR,
        record_id="abc123",
        title="Attention Is All You Need (S2 title)",
        authors=["Vaswani A.", "Shazeer N."],
        year=2017,
        venue="NIPS 2017",
        publication_venue="Conference on Neural Information Processing Systems",
        journal=None,
        doi="10.5555/3295222.3295349",
        arxiv_id="1706.03762",
        s2_paper_id="abc123",
        url="https://www.semanticscholar.org/paper/abc123",
    )
    defaults.update(kw)
    return ExternalRecord(**defaults)


def test_merge_arxiv_wins_title_authors_year():
    merged = merge_records([_arxiv_record(), _s2_record()])
    assert merged is not None
    assert merged.title == "Attention Is All You Need (arXiv title)"
    assert merged.authors == ["A. Vaswani", "N. Shazeer"]
    assert merged.year == 2017
    assert merged.provenance["title"] == Backend.ARXIV
    assert merged.provenance["authors"] == Backend.ARXIV
    assert merged.provenance["year"] == Backend.ARXIV


def test_merge_s2_wins_venue_and_doi():
    merged = merge_records([_arxiv_record(), _s2_record()])
    assert merged.venue == "Conference on Neural Information Processing Systems"
    assert merged.doi == "10.5555/3295222.3295349"
    assert merged.provenance["venue"] == Backend.SEMANTIC_SCHOLAR
    assert merged.provenance["doi"] == Backend.SEMANTIC_SCHOLAR


def test_merge_falls_back_to_arxiv_doi():
    arxiv = _arxiv_record(doi="10.1234/arxiv-doi")
    s2 = _s2_record(doi=None)
    merged = merge_records([arxiv, s2])
    assert merged.doi == "10.1234/arxiv-doi"
    assert merged.provenance["doi"] == Backend.ARXIV


def test_merge_arxiv_only():
    merged = merge_records([_arxiv_record()])
    assert merged.title.startswith("Attention")
    assert merged.venue is None
    assert merged.doi is None


def test_merge_s2_only():
    merged = merge_records([_s2_record()])
    assert merged.title == "Attention Is All You Need (S2 title)"
    assert merged.venue == "Conference on Neural Information Processing Systems"
    assert merged.provenance["title"] == Backend.SEMANTIC_SCHOLAR
    assert merged.arxiv_id == "1706.03762"  # comes from s2.externalIds
    assert merged.provenance["arxiv_id"] == Backend.SEMANTIC_SCHOLAR


def test_merge_empty_returns_none():
    assert merge_records([]) is None


def test_merge_propagates_arxiv_versioning():
    merged = merge_records([_arxiv_record(latest_arxiv_version=7, arxiv_versions=[7]), _s2_record()])
    assert merged.latest_arxiv_version == 7
    assert merged.arxiv_versions == [7]


def test_merge_propagates_withdrawn():
    merged = merge_records([_arxiv_record(withdrawn=True), _s2_record()])
    assert merged.withdrawn is True
