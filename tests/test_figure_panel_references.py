"""Body references to subpanels must reach their parent figure unchanged."""

from pathlib import Path

import fitz
import pytest

from preprocessing.materials import _anchors, build_materials
from preprocessing.parse.mineru_adapter import MineruParseResult


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See Fig. 1a for details.", {"1"}),
        ("Figures 1(a) and 2(b) show the results.", {"1", "2"}),
        ("Figs. 1a, 2b, and 10c use identical axes.", {"1", "2", "10"}),
        ("Figures 1 (a), 2 (b), and 3 (c) summarize the results.", {"1", "2", "3"}),
        ("Figure A.1a and Fig. S2(b) appear in the supplement.", {"a.1", "s2"}),
        ("Figures 1a and 1b share the legend.", {"1"}),
        ("Figure 1a shows 20 points and 3 classes.", {"1"}),
        ("Figure 1(a) is compared with Table 2 and Equation 3.", {"1"}),
        ("Figure A.1invalid is an invalid anchor.", set()),
        ("Figure 1ab and Figure 2extra are invalid anchors.", set()),
        ("Figures 1–3 retain their original range meaning.", {"1", "2", "3"}),
    ],
)
def test_panel_reference_grammar_and_boundaries(text, expected):
    assert _anchors(text) == expected


def test_panel_caption_and_body_linkage_preserve_source_spans(tmp_path: Path):
    paper = tmp_path / "panels.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.draw_rect(fitz.Rect(60, 80, 300, 400), color=(0, 0, 0))
        pdf.save(paper)
    sentences = [
        "See Fig. 1a for details.",
        "Figures 1(a) and 2(b) show the results.",
        "Figure 1a shows 20 points and 3 classes.",
    ]
    captions = ["Figure 1(a): First panel.", "Plot: Figure 2b shows the second panel.", "Figure 3: Unrelated."]
    rows = [{"type": "text", "text": text, "page_idx": 0} for text in sentences]
    rows.extend(
        {"type": "image", "image_caption": [caption], "page_idx": 0, "bbox": [100, 100, 500, 500]}
        for caption in captions
    )
    markdown = "\n\n".join(sentences + captions)
    parsed = MineruParseResult(
        markdown=markdown, content_list=rows, image_files=None, batch_id="panels",
        raw_result=None, provider="fixture",
    )
    result = build_materials(parsed, paper_pdf=paper, output_dir=tmp_path / "out", paper_key="panels")
    assert result.markdown == markdown
    assert Path(result.markdown_path).read_text(encoding="utf-8") == markdown
    assert [figure.anchor for figure in result.figures] == ["1", "2", "3"]
    assert [figure.caption for figure in result.figures] == captions
    assert [reference.text for reference in result.figures[0].references] == sentences
    assert [reference.text for reference in result.figures[1].references] == [sentences[1]]
    assert result.figures[2].references == []
    for figure in result.figures:
        for reference in figure.references:
            assert markdown[reference.loc.char_start:reference.loc.char_end] == reference.text
