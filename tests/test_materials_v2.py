"""Shared-material fixtures use real PDF rendering and a mocked MinerU boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock

import fitz
import pytest
from PIL import Image

from preprocessing.materials import _anchors, build_materials, index_repository, parse_materials
from preprocessing.parse.mineru_adapter import MineruParseResult
from schemas.materials import SharedMaterials


@pytest.fixture
def paper_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.draw_rect(fitz.Rect(60, 80, 300, 400), color=(0, 0, 0), fill=(0.1, 0.6, 0.8))
        page.insert_text((75, 110), "Panel (a)")
        pdf.save(path)
    return path


@pytest.fixture
def parsed() -> MineruParseResult:
    rows = [
        {"type": "text", "text": "Tiny Paper", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Abstract", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "A local test paper.", "page_idx": 0},
        {"type": "text", "text": "1 Method", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Figure 1 shows our model [1]. Figure 10 shows a baseline.", "page_idx": 0},
        {"type": "text", "text": "Figs. 1 and 10 share the same units.", "page_idx": 0},
        {"type": "equation", "text": "$$ y = W x $$", "page_idx": 0},
        {"type": "table", "table_body": "| Metric | Value |\n|---|---|\n| MRR | 0.4 |", "page_idx": 0},
        {
            "type": "image",
            "image_caption": ["Figure 1: Model with panel (a)."],
            "page_idx": 0,
            "bbox": [100, 100, 500, 500],
        },
        {
            "type": "image",
            "image_caption": ["Figure 10: Baseline."],
            "page_idx": 0,
            "bbox": [500, 500, 800, 750],
        },
        {"type": "text", "text": "References", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "[1] Author. Relevant work. 2020.", "page_idx": 0},
    ]
    markdown = "\n\n".join(
        ("# " if row.get("text_level") else "")
        + (row.get("text") or row.get("table_body") or "\n".join(row.get("image_caption", [])))
        for row in rows
    )
    return MineruParseResult(
        markdown=markdown,
        content_list=rows,
        image_files=None,
        batch_id="fixture",
        raw_result=None,
        provider="mineru_fixture",
    )


def test_materials_render_all_pages_crops_and_printed_size(paper_pdf, parsed, tmp_path):
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    assert len(result.pages) == 1
    assert Path(result.pages[0].path).is_file()
    assert len(result.figures) == 2
    for figure in result.figures:
        assert figure.caption and figure.loc.page == 1
        assert Path(figure.crop_path).is_file()
        assert Path(figure.printed_crop_path).is_file()
    assert result.figures[0].bbox_points == (60, 80, 300, 400)
    with Image.open(result.figures[0].printed_crop_path) as image:
        assert abs(image.width - 240 * 96 / 72) <= 1
        assert abs(image.height - 320 * 96 / 72) <= 1
    with Image.open(result.figures[0].crop_path) as image:
        assert image.width > 320
    assert result.issues == []


def test_materials_link_exact_figure_anchors_and_all_referencing_sentences(paper_pdf, parsed, tmp_path):
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    first, tenth = result.figures
    assert [row.text for row in first.references] == [
        "Figure 1 shows our model [1].",
        "Figs. 1 and 10 share the same units.",
    ]
    assert [row.text for row in tenth.references] == [
        "Figure 10 shows a baseline.",
        "Figs. 1 and 10 share the same units.",
    ]
    for figure in result.figures:
        for reference in figure.references:
            assert result.markdown[reference.loc.char_start : reference.loc.char_end] == reference.text


def test_materials_preserve_source_and_bibliography_with_verifiable_spans(paper_pdf, parsed, tmp_path):
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    assert Path(result.markdown_path).read_text(encoding="utf-8") == parsed.markdown
    assert result.markdown == parsed.markdown
    assert result.title == "Tiny Paper"
    assert result.abstract == "A local test paper."
    assert result.bibliography[0].text == "[1] Author. Relevant work. 2020."
    assert next(row for row in result.blocks if row.kind == "equation").text == "$$ y = W x $$"
    assert "| MRR | 0.4 |" in next(row for row in result.blocks if row.kind == "table").text
    for block in result.blocks:
        assert block.loc.page == 1
        assert block.loc.section
        assert result.markdown[block.loc.char_start : block.loc.char_end] == block.text
    round_trip = SharedMaterials.model_validate_json((tmp_path / "out" / "materials.json").read_text())
    assert round_trip == result


@pytest.mark.parametrize("bbox", [None, [-1, 0, 100, 100], [0, 0, 0, 100], [0, 0, 1001, 100]])
def test_materials_missing_or_invalid_bbox_retains_figure_and_reason(paper_pdf, parsed, tmp_path, bbox):
    parsed.content_list[8]["bbox"] = bbox
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    figure = result.figures[0]
    assert figure.caption and figure.references
    assert figure.crop_path == ""
    assert figure.printed_crop_path == ""
    assert any("figure_1: crop/printed-size input unavailable" in issue for issue in result.issues)


def test_materials_pdf_point_bbox_is_explicit(paper_pdf, parsed, tmp_path):
    parsed.content_list[8]["bbox"] = [60, 80, 300, 400]
    parsed.content_list[8]["bbox_space"] = "pdf_points"
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    assert result.figures[0].bbox_points == (60, 80, 300, 400)
    # The second row remains normalized even though its values also fit PDF points.
    assert result.figures[1].bbox_points == (300, 400, 480, 600)


def test_materials_missing_char_span_never_fabricates_offsets(paper_pdf, parsed, tmp_path):
    parsed.content_list.append({"type": "text", "text": "Missing from the markdown.", "page_idx": 0})
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    block = result.blocks[-1]
    assert block.loc.page == 1
    assert block.loc.char_start is None and block.loc.char_end is None
    assert any("could not be aligned" in issue for issue in result.issues)


def test_materials_empty_reference_list_is_valid(paper_pdf, parsed, tmp_path):
    parsed.content_list[8]["image_caption"] = ["Figure 2: Unreferenced drawing."]
    parsed.markdown = parsed.markdown.replace(
        "Figure 1: Model with panel (a).", "Figure 2: Unreferenced drawing."
    )
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    assert result.figures[0].references == []
    assert result.figures[0].crop_path and result.figures[0].caption


def test_materials_repeated_paragraphs_keep_distinct_exact_spans(paper_pdf, parsed, tmp_path):
    text = "Figure 1 has shared axes."
    parsed.markdown += f"\n\n{text}\n\n{text}"
    parsed.content_list.extend([{"type": "text", "text": text, "page_idx": 0}] * 2)
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    first, second = result.blocks[-2:]
    assert first.loc.char_start < second.loc.char_start
    assert result.markdown[first.loc.char_start : first.loc.char_end] == text
    assert result.markdown[second.loc.char_start : second.loc.char_end] == text


def test_materials_figure_reference_sentence_preserves_scientific_abbreviations(paper_pdf, parsed, tmp_path):
    original = parsed.content_list[4]["text"]
    replacement = "We use e.g. Fig.1 for a complete description. Figure 10 shows a baseline."
    parsed.content_list[4]["text"] = replacement
    parsed.markdown = parsed.markdown.replace(original, replacement)
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    assert result.figures[0].references[0].text == "We use e.g. Fig.1 for a complete description."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Figure A.1 shows the proof.", {"a.1"}),
        ("Figures 1, 2, and 3 summarize the results.", {"1", "2", "3"}),
        ("Figures A.1, A.2, and A.3 summarize the results.", {"a.1", "a.2", "a.3"}),
        ("Figs.1, 2 & 3 and Figure S1 agree.", {"1", "2", "3", "s1"}),
        ("Figures 1–3 are the ablations.", {"1", "2", "3"}),
        ("Figure A.10 and Figure 10 are distinct.", {"a.10", "10"}),
        ("Table A.1 and Eq. 3 are omitted.", set()),
        ("Figure A.1invalid is an invalid anchor.", set()),
    ],
)
def test_figure_anchor_grammar_and_exact_nonmatches(text, expected):
    assert _anchors(text) == expected


def test_materials_list_items_keep_claims_references_and_spans(paper_pdf, parsed, tmp_path):
    old_text = parsed.content_list[4]["text"]
    texts = ["Figure 1 shows the first advantage [1].", "Figure 10 shows the second advantage."]
    parsed.content_list[4] = {"type": "list", "list_items": [texts[0], {"text": texts[1]}], "page_idx": 0}
    parsed.markdown = parsed.markdown.replace(old_text, "\n".join(f"- {text}" for text in texts))
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    items = [block for block in result.blocks if block.kind == "list"]
    assert [block.text for block in items] == texts
    for block in items:
        assert block.loc.page == 1
        assert result.markdown[block.loc.char_start : block.loc.char_end] == block.text
    assert result.figures[0].references[0].text == texts[0]
    assert result.figures[1].references[0].text == texts[1]
    assert result.issues == []


def test_materials_unsupported_content_row_is_visible(paper_pdf, parsed, tmp_path):
    parsed.content_list.append(
        {"type": "custom_block", "content": "Important source content.", "page_idx": 0}
    )
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    assert any("unsupported parser row type 'custom_block'" in issue for issue in result.issues)
    assert any("parser content could not be converted to text" in issue for issue in result.issues)
    assert "Important source content." in Path(result.content_list_path).read_text(encoding="utf-8")


def test_repository_index_lists_entries_configs_and_never_writes(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "run.py").write_text("if __name__ == '__main__':\n    pass\n")
    (root / "README.md").write_text("Run python run.py\n")
    (root / "model.py").write_text("class Model: pass\n")
    (root / "config" / "train.yaml").write_text("epochs: 3\n")
    (root / ".git" / "config").write_text("private metadata")
    before = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    result = index_repository(root)
    assert result.entry_scripts == ["run.py"]
    assert result.configs == ["config/train.yaml"]
    assert {row.path for row in result.files} == {"run.py", "README.md", "model.py", "config/train.yaml"}
    assert next(row for row in result.files if row.path == "run.py").line_count == 2
    after = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.asyncio
async def test_parse_materials_uses_only_injected_mineru(paper_pdf, parsed, tmp_path):
    parser = AsyncMock()
    parser.parse_pdf.return_value = parsed
    result = await parse_materials(
        paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny", parser=parser
    )
    parser.parse_pdf.assert_awaited_once_with(pdf_path=paper_pdf, data_id="tiny")
    assert result.provider == "mineru_fixture"
    assert result.figures


def test_parser_assets_cannot_escape_materials_output(paper_pdf, parsed, tmp_path):
    parsed.image_files = {"../../escaped.png": b"bad", "images/asset.png": b"saved"}
    result = build_materials(parsed, paper_pdf=paper_pdf, output_dir=tmp_path / "out", paper_key="tiny")
    assert not (tmp_path / "escaped.png").exists()
    assert (tmp_path / "out" / "assets" / "images" / "asset.png").read_bytes() == b"saved"
    assert any("unsafe parser image path" in issue for issue in result.issues)
