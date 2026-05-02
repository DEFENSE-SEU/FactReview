"""Unit tests for :mod:`fact_extraction.results_parser`."""

from __future__ import annotations

from pathlib import Path

from preprocessing.claim_extract.results_parser import (
    _parse_number,
    extract_reported_results,
)
from schemas.paper import Paper, PaperMetadata, Table


def _paper_with_tables(tables: list[Table]) -> Paper:
    return Paper(
        metadata=PaperMetadata(paper_key="synthetic"),
        pdf_path=Path("synthetic.pdf"),
        tables=tables,
    )


class TestParseNumber:
    def test_plain_float(self) -> None:
        assert _parse_number("0.355") == 0.355

    def test_percentage(self) -> None:
        assert _parse_number("87.2%") == 87.2

    def test_bold_markdown(self) -> None:
        assert _parse_number("**53.1**") == 53.1

    def test_with_pm(self) -> None:
        assert _parse_number("0.355 \\pm 0.003") == 0.355

    def test_empty(self) -> None:
        assert _parse_number("") is None
        assert _parse_number("  ") is None
        assert _parse_number("-") is None


class TestExtractReportedResults:
    def test_simple_link_prediction_table(self) -> None:
        """A mock link-prediction table with metric headers on row 0."""
        table = Table(
            id="table_4",
            caption="Link prediction results on FB15k-237.",
            rows=[
                ["Method", "MRR", "Hits@10"],
                ["TransE", "0.294", "46.5"],
                ["ConvE", "0.325", "50.1"],
                ["OurModel", "0.355", "53.5"],
            ],
        )
        results = extract_reported_results(_paper_with_tables([table]))

        # 3 methods x 2 metrics = 6 results.
        assert len(results) == 6

        by_method = {(r.method, r.metric): r.value for r in results}
        assert by_method[("OurModel", "MRR")] == 0.355
        assert by_method[("OurModel", "Hits@10")] == 53.5

        # Dataset + task inferred from caption.
        for r in results:
            assert r.dataset == "FB15k-237"
            assert r.task == "link_prediction"
            assert r.table_id == "table_4"

    def test_two_level_header_per_column_dataset(self) -> None:
        """Multi-dataset table: per-column dataset must win over caption."""
        table = Table(
            id="table_5",
            caption="Link prediction results.",
            rows=[
                ["", "FB15k-237", "WN18RR"],
                ["Method", "MRR", "MRR"],
                ["OurModel", "0.355", "0.479"],
            ],
        )
        results = extract_reported_results(_paper_with_tables([table]))
        assert len(results) == 2
        by_ds = {r.dataset: r.value for r in results}
        assert by_ds["FB15k-237"] == 0.355
        assert by_ds["WN18RR"] == 0.479

    def test_table_without_metric_headers_is_skipped(self) -> None:
        table = Table(
            id="table_1",
            caption="Comparison of model properties.",
            rows=[
                ["Method", "NodeEmb", "RelEmb"],
                ["OurModel", "yes", "yes"],
            ],
        )
        assert extract_reported_results(_paper_with_tables([table])) == []

    def test_empty_table(self) -> None:
        assert extract_reported_results(_paper_with_tables([])) == []
        table = Table(id="t0", caption="", rows=[["MRR"]])
        assert extract_reported_results(_paper_with_tables([table])) == []

    def test_ids_are_unique_and_stable(self) -> None:
        table = Table(
            id="table_4",
            caption="Link prediction on FB15k-237.",
            rows=[
                ["Method", "MRR"],
                ["A", "0.1"],
                ["B", "0.2"],
            ],
        )
        results = extract_reported_results(_paper_with_tables([table]))
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("table_4.row") for i in ids)
