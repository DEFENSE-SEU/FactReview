"""Readable original evidence tables retain source records and never execute HTML."""

import json
import re
from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from pypdf import PdfReader

from review.report.v2 import render_markdown, write_review
from schemas.claim import Claim, ClaimLocation, Condition, Evidence, EvidencePointer
from schemas.review import FinalReview


def review(quote):
    return FinalReview(
        paper_key="safe-evidence-table",
        run_id="offline",
        claims=[
            Claim(
                id="claim_1",
                text="The source reports two original scores.",
                status="unverified",
                loc=ClaimLocation(page=3),
                conditions=[Condition(id="c1", dataset="D", metric="accuracy")],
                needs=["Experiments"],
                evidence=[
                    Evidence(
                        source="paper_internal",
                        pointer=EvidencePointer(
                            locator=r"C:\paper_source\paper.md", page=3, key="table_2", quote=quote
                        ),
                        covered=["c1"],
                        direction="support",
                        sufficient=False,
                        note="Original observations; broader claim coverage remains unverified.",
                    )
                ],
            )
        ],
    )


def table_rows(markdown):
    rows, current = [], None
    for token in MarkdownIt("gfm-like", {"linkify": False}).parse(markdown):
        if token.type == "tr_open":
            current = []
        elif token.type == "inline" and current is not None:
            current.append("".join(child.content for child in token.children or []))
        elif token.type == "tr_close":
            rows.append(current)
            current = None
    return rows


def assert_generated_navigation_only(tokens):
    children = [child for token in tokens for child in token.children or []]
    targets = []
    for child in children:
        assert child.type != "image"
        if child.type == "html_inline":
            match = re.fullmatch(
                r'<a id="(factreview-(?:source-[a-f0-9]{64}|evidence-[0-9]{6,}))">', child.content
            )
            assert match or child.content == "</a>"
            if match:
                targets.append(match.group(1))
    assert len(targets) == len(set(targets))
    assert all(
        child.attrGet("href") in {"#" + target for target in targets}
        for child in children
        if child.type == "link_open"
    )


def test_table_span_associations_and_all_original_records_survive_markdown_and_pdf(tmp_path):
    quote = (
        "Before table: original protocol.\n<table><caption>Table 2: D results</caption>"
        '<thead><tr><th rowspan="2">Model</th><th colspan="2">D test</th></tr>'
        "<tr><th>accuracy (%)</th><th>latency (ms)</th></tr></thead><tbody>"
        "<tr><td>A</td><td>90.1</td><td>5.0</td></tr>"
        "<tr><td>B</td><td>90.0</td><td>6.0</td></tr></tbody></table>\nAfter table: same settings."
    )
    original = review(quote)
    snapshot = original.model_dump(mode="json", exclude={"review_markdown"})
    outputs = write_review(original, tmp_path)
    assert "pdf_error" not in outputs
    saved = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    saved.pop("review_markdown")
    assert saved == snapshot
    assert original.model_dump(mode="json", exclude={"review_markdown"}) == snapshot
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    expected = [
        ["Model", "D test", "D test"],
        ["Model", "accuracy (%)", "latency (ms)"],
        ["A", "90.1", "5.0"],
        ["B", "90.0", "6.0"],
    ]
    assert table_rows(markdown)[-4:] == expected
    text = "\n".join(page.extract_text() for page in PdfReader(outputs["pdf"]).pages)
    for value in {cell for row in expected for cell in row} | {
        "Before table: original protocol.",
        "After table: same settings.",
        "Table 2: D results",
        r"C:\paper_source\paper.md",
        "table_2",
        "Original observations",
    }:
        assert value in text
    assert "<td>" not in text and "<table>" not in text
    assert all(f"## {n}." in markdown for n in range(1, 5))


def test_table_entities_formatting_and_special_characters_do_not_change_cell_count():
    quote = (
        "<table><tr><th>Model &amp; setting</th><th>Value</th></tr>"
        "<tr><td><strong>A|B</strong> &lt;C&gt; 'quoted' [x](https://example.invalid)</td>"
        "<td>384<sup>2</sup> and x<sub>1</sub><br/>90.1 ± 0.2</td></tr></table>"
    )
    markdown = render_markdown(review(quote))
    rows = table_rows(markdown)
    assert rows[-2:] == [
        ["Model & setting", "Value"],
        ["A|B <C> 'quoted' [x](https://example.invalid)", "384^(2) and x_(1) 90.1 ± 0.2"],
    ]
    tokens = MarkdownIt("gfm-like", {"linkify": False}).parse(markdown)
    assert_generated_navigation_only(tokens)


@pytest.mark.parametrize(
    "quote",
    [
        "<table><tr><td>A</td><td>90</td></tr>",
        "<table><tr><td>A<td>90</td></tr></table>",
        "<table><tr><td>A</td><td>90</td></tr><tr><td>B</td></tr></table>",
        '<table><tr><td rowspan="2">90</td></tr></table>',
        '<table><tr><td colspan="0">90</td></tr></table>',
        '<table><tr><td colspan="abc">90</td></tr></table>',
        "<table><tr><td>A<table><tr><td>90</td></tr></table></td></tr></table>",
        '<table><tr><td><img src="https://example.invalid/image"/>90</td></tr></table>',
        "<table><tr><td><script>alert(90)</script></td></tr></table>",
        "<table>outside-cell value 90<tr><td>A</td></tr></table>",
        "<table><tr><td>A<![arbitrary qualifier]></td><td>90</td></tr></table>",
        "<table><tr><td>A<![CDATA[ without augmentation ]]></td><td>90</td></tr></table>",
        "<table><tr><td>A<?audit original-qualifier?></td><td>90</td></tr></table>",
        '<table><tr><td colspan="1" colspan="2">A</td><td>90</td></tr></table>',
    ],
)
def test_malformed_or_unsupported_tables_preserve_explicit_safe_original_fallback(tmp_path, quote):
    record = review(quote)
    outputs = write_review(record, tmp_path, render_pdf=False)
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "Table layout unavailable; original passage follows." in markdown
    assert "&lt;table&gt;" in markdown
    assert (
        json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))["claims"][0]["evidence"][0]["pointer"][
            "quote"
        ]
        == quote
    )
    tokens = MarkdownIt("gfm-like", {"linkify": False}).parse(markdown)
    assert_generated_navigation_only(tokens)


def test_multiple_tables_preserve_each_table_and_surrounding_text():
    quote = "First. <table><tr><th>A</th></tr><tr><td>90</td></tr></table> Between. <table><tr><th>B</th></tr><tr><td>80</td></tr></table> Last."
    markdown = render_markdown(review(quote))
    assert table_rows(markdown)[-4:] == [["A"], ["90"], ["B"], ["80"]]
    assert all(word in markdown for word in ("First", "Between", "Last"))


def test_complete_row_excerpt_uses_explicit_column_positions_without_inventing_headers(tmp_path):
    quote = '<tr><td colspan="3">Variant group</td></tr><tr><td>A</td><td>90</td><td>80</td></tr>'
    outputs = write_review(review(quote), tmp_path)
    assert "pdf_error" not in outputs
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "column headings are not included" in markdown
    assert table_rows(markdown)[-3:] == [
        ["Column 1", "Column 2", "Column 3"],
        ["Variant group", "Variant group", "Variant group"],
        ["A", "90", "80"],
    ]
    text = "\n".join(page.extract_text() for page in PdfReader(outputs["pdf"]).pages)
    assert "column headings are not included" in text
    assert "<td>" not in text and "<tr>" not in text
    assert (
        json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))["claims"][0]["evidence"][0]["pointer"][
            "quote"
        ]
        == quote
    )


@pytest.mark.parametrize("quote", ["<td>A</td><td>90</td>", "<tr><td>A</td><td>90</td>"])
def test_incomplete_row_fragments_preserve_original_in_explicit_fallback(quote):
    markdown = render_markdown(review(quote))
    assert "Table layout unavailable; original passage follows." in markdown
    assert "&lt;td&gt;A&lt;/td&gt;" in markdown


def test_pdf_continuation_repeats_all_explicit_header_levels_and_keeps_original_rows(tmp_path):
    header = (
        '<tr><th rowspan="2">Model</th><th colspan="2">D test</th></tr>'
        "<tr><th>accuracy (%)</th><th>latency (ms)</th></tr>"
    )
    body = "".join(f"<tr><td>R{i}</td><td>{90 + i / 100:.2f}</td><td>{i + 1}</td></tr>" for i in range(100))
    record = review(f"<table><thead>{header}</thead><tbody>{body}</tbody></table>")
    output = write_review(record, tmp_path)
    assert "pdf_error" not in output
    markdown = Path(output["markdown"]).read_text(encoding="utf-8")
    rows = table_rows(markdown)[-103:]
    assert rows[:3] == [
        ["Model", "D test / accuracy (%)", "D test / latency (ms)"],
        ["Model", "D test", "D test"],
        ["Model", "accuracy (%)", "latency (ms)"],
    ]
    assert rows[3:] == [[f"R{i}", f"{90 + i / 100:.2f}", str(i + 1)] for i in range(100)]
    pages_with_data = []
    for page in PdfReader(output["pdf"]).pages:
        text = " ".join((page.extract_text() or "").split())
        if any(f"R{i}" in text for i in range(100)):
            pages_with_data.append(text)
            assert "D test / accuracy (%)" in text and "D test / latency (ms)" in text
    assert len(pages_with_data) > 1


def test_headerless_rows_do_not_gain_inferred_quantity_headers():
    quote = '<table><tr><td rowspan="2">Model</td><td colspan="2">D</td></tr><tr><td>90</td><td>80</td></tr></table>'
    rows = table_rows(render_markdown(review(quote)))
    assert rows[-2:] == [["Model", "D", "D"], ["Model", "90", "80"]]
