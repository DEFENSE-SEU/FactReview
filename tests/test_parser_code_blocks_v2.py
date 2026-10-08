"""MinerU algorithm and fenced example rows remain located v2 source blocks."""

import json
from pathlib import Path

import pymupdf
import pytest

from preprocessing.materials import _blocks, _row_text, build_materials
from preprocessing.parse.mineru_adapter import MineruParseResult
from screening.claims import extract_claims
from verification.code import verify_code

ALGORITHM = (
    '<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">\n'
    "Algorithm 1 Blockwise Masking\n"
    "Input: image patches\n"
    "repeat\n"
    "    mask a sampled block\n"
    "until |M| &gt; 0.4N\n"
    "Masking ratio is 40%\n"
    "</div>"
)
NSP = (
    "```txt\n"
    "Input = [CLS] the man went to [MASK] store [SEP]\n"
    "    he bought a gallon [MASK] milk [SEP]\n"
    "Label = IsNext\n"
    "```"
)


@pytest.mark.parametrize(
    "subtype,body,quote,source_line",
    [
        ("algorithm", ALGORITHM, "Masking ratio is 40%", "mask_ratio = 0.4"),
        ("text", NSP, "Label = IsNext", "example_label = 'IsNext'"),
    ],
)
def test_code_rows_survive_materials_claim_extraction_and_code_verification(
    tmp_path, subtype, body, quote, source_line
):
    pdf_path = tmp_path / "paper.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((30, 40), quote)
        pdf.save(pdf_path)
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "config.py").write_text(source_line + "\n", encoding="utf-8")
    markdown = "# Method\n\n" + body + "\n"
    rows = [
        {"type": "text", "text": "# Method", "text_level": 1, "page_idx": 0},
        {"type": "code", "sub_type": subtype, "code_caption": [], "code_body": body, "page_idx": 0},
    ]
    materials = build_materials(
        MineruParseResult(markdown, rows, None, "fixture", {}, "fixture"),
        paper_pdf=pdf_path,
        output_dir=tmp_path / "materials",
        paper_key="fixture",
        repo_root=repository,
    )
    block = next(block for block in materials.blocks if block.kind == "code")
    assert block.id == "block_2"
    assert block.text == body
    assert block.loc.page == 1
    assert materials.markdown[block.loc.char_start : block.loc.char_end] == body
    assert Path(materials.markdown_path).read_text(encoding="utf-8") == markdown
    assert not materials.issues

    def call(**kwargs):
        if kwargs["module"] == "screening.claims":
            payload = json.loads(kwargs["prompt"].split("\nPAPER_DATA_JSON:\n", 1)[1])
            assert next(b for b in payload["blocks"] if b["kind"] == "code")["text"] == body
            return {
                "status": "ok",
                "claims": [
                    {
                        "text": quote,
                        "source_block_id": block.id,
                        "source_quote": quote,
                        "conditions": [{"id": "setting", "description": quote}],
                        "needs": ["Code"],
                        "importance": "core",
                    }
                ],
            }
        payload = json.loads(kwargs["prompt"])
        assert next(b for b in payload["paper_blocks"] if b["id"] == block.id)["text"] == body
        return {
            "items": [
                {
                    "file": "config.py",
                    "line": 1,
                    "quote": source_line,
                    "paper_block_id": block.id,
                    "paper_quote": quote,
                    "covered": ["setting"],
                    "fully_supported_conditions": ["setting"],
                    "direction": "support",
                    "aspect": "hyperparameters",
                    "detail": "The fixture source matches the located example setting.",
                }
            ]
        }

    claim = extract_claims(materials, call=call)[0]
    assert claim.source_block_id == block.id
    assert claim.loc.page == 1
    assert materials.markdown[claim.loc.char_start : claim.loc.char_end] == quote
    result = verify_code(claim, materials, call=call)
    assert len(result.evidence) == 1 and result.evidence[0].sufficient
    assert result.evidence[0].pointer.quote == source_line
    assert quote in result.evidence[0].note
    assert "chars:" in result.evidence[0].note


def test_code_caption_body_and_distinct_fallback_text_are_all_preserved_once():
    row = {
        "type": "code",
        "code_caption": ["Algorithm 1: Masking"],
        "code_body": ALGORITHM,
        "text": "Extra parser explanation.",
        "page_idx": 0,
    }
    text = "Algorithm 1: Masking\n" + ALGORITHM + "\nExtra parser explanation."
    assert _row_text(row) == text
    issues = []
    block = _blocks(text, [row], issues)[0]
    assert block.kind == "code" and block.text == text
    assert not issues
    assert _row_text({**row, "text": ALGORITHM}) == "Algorithm 1: Masking\n" + ALGORITHM


def test_unaligned_code_keeps_exact_parser_text_and_pdf_location():
    row = {"type": "code", "code_body": NSP, "page_idx": 2}
    issues = []
    block = _blocks("Markdown did not include the code.", [row], issues)[0]
    assert block.kind == "code" and block.text == NSP
    assert block.loc.page == 3 and block.loc.char_start is None
    assert any("char span unavailable" in issue for issue in issues)
    assert not any("unsupported parser row type" in issue for issue in issues)


def test_unknown_parser_row_types_still_produce_coverage_issue():
    issues = []
    _blocks("x", [{"type": "unknown-widget", "text": "x", "page_idx": 0}], issues)
    assert any("unsupported parser row type 'unknown-widget'" in issue for issue in issues)
