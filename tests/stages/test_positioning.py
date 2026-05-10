"""Positioning stage tests.

The positioning stage_runner is mostly JSON plumbing; the load-bearing logic
is the publication-cutoff filter that prevents anachronistic neighbour
papers from polluting the comparison. These tests pin down that filter
behaviour through its three public functions.
"""

from __future__ import annotations

import pytest

from util.cutoff_date import (
    derive_cutoff_from_source,
    filter_papers,
    parse_cutoff,
)


def test_arxiv_url_derives_month_precision_cutoff() -> None:
    # The positioning stage feeds this into Semantic Scholar's year filter
    # and into ``filter_papers``. New-scheme arXiv IDs (post-2007) and the
    # bare ``YYYYMM.NNNNN`` form must both resolve to the same month.
    for source in (
        "https://arxiv.org/abs/2210.12345",
        "https://arxiv.org/pdf/2210.12345v3",
        "2210.12345",
    ):
        cutoff = derive_cutoff_from_source(source)
        assert cutoff is not None
        assert (cutoff.year, cutoff.month, cutoff.precision) == (2022, 10, "month")

    # Non-arXiv inputs return None so the caller can decide whether to fall
    # back to a user-supplied cutoff. PDF-on-disk paths must NOT be parsed
    # as arXiv IDs by accident.
    assert derive_cutoff_from_source("/local/path/to/paper.pdf") is None
    assert derive_cutoff_from_source("https://example.com/paper") is None


def test_filter_papers_drops_post_cutoff_using_published_when_present() -> None:
    cutoff = parse_cutoff("2022-06")
    assert cutoff is not None
    papers = [
        {"title": "old", "year": 2020},
        {"title": "june-end", "year": 2022, "published": "2022-06-30T00:00:00Z"},
        {"title": "july-start", "year": 2022, "published": "2022-07-01T00:00:00Z"},
        {"title": "future-year-only", "year": 2024},
        {"title": "no-metadata", "year": None},
    ]
    kept, dropped = filter_papers(papers, cutoff)
    # ``published`` wins over ``year`` for day precision; ``no-metadata`` is
    # kept rather than silently filtered out.
    assert [p["title"] for p in kept] == ["old", "june-end", "no-metadata"]
    assert [p["title"] for p in dropped] == ["july-start", "future-year-only"]


def test_no_cutoff_keeps_everything_and_invalid_token_raises() -> None:
    papers = [{"year": 1990}, {"year": 2999}]
    kept, dropped = filter_papers(papers, None)
    assert kept == papers
    assert dropped == []

    # Non-empty malformed cutoff strings must surface as ValueError so users
    # get a clear error instead of a silent year-only fallback.
    with pytest.raises(ValueError):
        parse_cutoff("2022-13")
