from __future__ import annotations

import pytest

from util.cutoff_date import (
    CutoffDate,
    derive_cutoff_from_source,
    filter_papers,
    is_after_cutoff,
    parse_cutoff,
)


class TestParseCutoff:
    def test_year_only_expands_to_year_end(self) -> None:
        cutoff = parse_cutoff("2022")
        assert cutoff == CutoffDate(year=2022, month=12, day=31, precision="year")
        assert cutoff.s2_year_param() == "-2022"

    def test_year_month_expands_to_month_end(self) -> None:
        cutoff = parse_cutoff("2022-02")
        assert cutoff is not None
        assert (cutoff.year, cutoff.month, cutoff.day) == (2022, 2, 28)
        assert cutoff.precision == "month"

    def test_year_month_leap(self) -> None:
        cutoff = parse_cutoff("2024-02")
        assert cutoff is not None
        assert cutoff.day == 29

    def test_full_date(self) -> None:
        cutoff = parse_cutoff("2022-10-15")
        assert cutoff == CutoffDate(year=2022, month=10, day=15, precision="day")

    def test_empty_returns_none(self) -> None:
        assert parse_cutoff("") is None
        assert parse_cutoff(None) is None
        assert parse_cutoff("   ") is None

    @pytest.mark.parametrize(
        "value", ["abc", "2022-13", "2022-02-30", "1800", "2200-01", "2022-01-01-01"]
    )
    def test_invalid_raises(self, value: str) -> None:
        with pytest.raises(ValueError):
            parse_cutoff(value)


class TestDeriveCutoffFromSource:
    @pytest.mark.parametrize(
        "source,expected_year,expected_month",
        [
            ("https://arxiv.org/abs/2210.12345", 2022, 10),
            ("https://arxiv.org/pdf/2210.12345.pdf", 2022, 10),
            ("https://arxiv.org/pdf/2210.12345v3", 2022, 10),
            ("arxiv.org/abs/2210.12345", 2022, 10),
            ("2210.12345", 2022, 10),
            ("hep-th/0701005", 2007, 1),
        ],
    )
    def test_arxiv_inputs(self, source: str, expected_year: int, expected_month: int) -> None:
        cutoff = derive_cutoff_from_source(source)
        assert cutoff is not None
        assert cutoff.year == expected_year
        assert cutoff.month == expected_month
        assert cutoff.precision == "month"

    def test_old_arxiv_id_pre_2000(self) -> None:
        # 1999 January
        cutoff = derive_cutoff_from_source("hep-th/9901001")
        assert cutoff is not None
        assert cutoff.year == 1999
        assert cutoff.month == 1

    @pytest.mark.parametrize(
        "source",
        [
            "",
            None,
            "https://example.com/papers/foo.pdf",
            "/local/path/to/paper.pdf",
            "https://arxiv.org/abs/not-a-real-id",
        ],
    )
    def test_non_arxiv_returns_none(self, source: str | None) -> None:
        assert derive_cutoff_from_source(source) is None


class TestIsAfterCutoff:
    def test_published_date_takes_precedence_over_year(self) -> None:
        cutoff = parse_cutoff("2022-06")
        assert cutoff is not None
        # year=2022 alone would pass (cutoff is year=2022), but published is in July
        assert is_after_cutoff(
            paper_year=2022, paper_published="2022-07-01T00:00:00Z", cutoff=cutoff
        )
        assert not is_after_cutoff(
            paper_year=2022, paper_published="2022-06-30T00:00:00Z", cutoff=cutoff
        )

    def test_year_only_fallback(self) -> None:
        cutoff = parse_cutoff("2022")
        assert cutoff is not None
        assert is_after_cutoff(paper_year=2023, paper_published=None, cutoff=cutoff)
        assert not is_after_cutoff(paper_year=2022, paper_published=None, cutoff=cutoff)
        assert not is_after_cutoff(paper_year=2020, paper_published="", cutoff=cutoff)

    def test_unknown_year_kept(self) -> None:
        cutoff = parse_cutoff("2022")
        assert cutoff is not None
        assert not is_after_cutoff(paper_year=None, paper_published=None, cutoff=cutoff)

    def test_garbage_published_falls_back_to_year(self) -> None:
        cutoff = parse_cutoff("2022")
        assert cutoff is not None
        assert is_after_cutoff(paper_year=2023, paper_published="garbage", cutoff=cutoff)


class TestFilterPapers:
    def test_no_cutoff_keeps_everything(self) -> None:
        papers = [{"year": 2030}, {"year": 1990}]
        kept, dropped = filter_papers(papers, None)
        assert kept == papers
        assert dropped == []

    def test_filters_by_year(self) -> None:
        cutoff = parse_cutoff("2022")
        papers = [
            {"title": "old", "year": 2020},
            {"title": "borderline", "year": 2022},
            {"title": "future", "year": 2024},
            {"title": "no-year", "year": None},
        ]
        kept, dropped = filter_papers(papers, cutoff)
        kept_titles = [p["title"] for p in kept]
        dropped_titles = [p["title"] for p in dropped]
        assert kept_titles == ["old", "borderline", "no-year"]
        assert dropped_titles == ["future"]

    def test_uses_published_when_available(self) -> None:
        cutoff = parse_cutoff("2022-06")
        papers = [
            {"title": "june-end", "year": 2022, "published": "2022-06-30T00:00:00Z"},
            {"title": "july-start", "year": 2022, "published": "2022-07-01T00:00:00Z"},
        ]
        kept, dropped = filter_papers(papers, cutoff)
        assert [p["title"] for p in kept] == ["june-end"]
        assert [p["title"] for p in dropped] == ["july-start"]

    def test_skips_non_dict_entries(self) -> None:
        cutoff = parse_cutoff("2022")
        kept, dropped = filter_papers([{"year": 2020}, "garbage", None], cutoff)
        assert len(kept) == 1
        assert dropped == []
