"""Multi-source merge."""

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


def _openreview_record(**kw):
    defaults = dict(
        backend=Backend.OPENREVIEW,
        record_id="BeFjjyzWOJ",
        title="A Technical Report on Erasing the Invisible (OR title)",
        authors=["Mucong Ding", "Bang An"],
        year=2025,
        venue="NeurIPS 2025 Datasets and Benchmarks Track poster",
        publication_venue="NeurIPS 2025 Datasets and Benchmarks Track poster",
        journal=None,
        doi=None,
        arxiv_id=None,
        url="https://openreview.net/forum?id=BeFjjyzWOJ",
    )
    defaults.update(kw)
    return ExternalRecord(**defaults)


def test_merge_openreview_only_supplies_metadata():
    """When the paper is OpenReview-exclusive, all fields come from there."""
    merged = merge_records([_openreview_record()])
    assert merged is not None
    assert "Erasing the Invisible" in merged.title
    assert merged.authors == ["Mucong Ding", "Bang An"]
    assert merged.year == 2025
    assert merged.venue == "NeurIPS 2025 Datasets and Benchmarks Track poster"
    assert merged.url == "https://openreview.net/forum?id=BeFjjyzWOJ"
    assert merged.provenance["title"] == Backend.OPENREVIEW
    assert merged.provenance["authors"] == Backend.OPENREVIEW
    assert merged.provenance["venue"] == Backend.OPENREVIEW


def test_merge_arxiv_outranks_openreview():
    """When all three backends respond, arXiv stays authoritative for title/authors."""
    merged = merge_records([_arxiv_record(), _s2_record(), _openreview_record()])
    assert merged.provenance["title"] == Backend.ARXIV
    assert merged.provenance["authors"] == Backend.ARXIV
    # Venue stays with S2 (the canonical published venue).
    assert merged.provenance["venue"] == Backend.SEMANTIC_SCHOLAR


def test_merge_openreview_fills_venue_when_others_missing():
    """When arXiv has no venue and S2 is absent, OpenReview's venue wins."""
    merged = merge_records([_arxiv_record(), _openreview_record()])
    assert merged.venue == "NeurIPS 2025 Datasets and Benchmarks Track poster"
    assert merged.provenance["venue"] == Backend.OPENREVIEW
