"""Table visuals use real local PDF crops and mocked model requests."""

import json
from pathlib import Path

import fitz
import pytest
from PIL import Image

from preprocessing.materials import build_materials
from preprocessing.parse.mineru_adapter import MineruParseResult
from schemas.materials import SharedMaterials


@pytest.fixture
def table_paper(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=360, height=300)
        page.insert_text((30, 30), "Table 1. Accuracy by method.")
        page.insert_text((30, 65), "Method       Accuracy (%)")
        page.insert_text((30, 95), "A                 90", fontname="hebo")
        page.insert_text((30, 125), "B                 80")
        pdf.save(pdf_path)
    rows = [
        {"type": "text", "text": "Method", "text_level": 1, "page_idx": 0},
        {
            "type": "text",
            "text": "Table 1 compares two methods. Figure 1 shows the architecture. Table 10 has other results.",
            "page_idx": 0,
        },
        {"type": "text", "text": "Tables 1 and 10 share units. Tab. 1 has two methods.", "page_idx": 0},
        {
            "type": "table",
            "table_caption": ["Table 1. Accuracy by method."],
            "table_footnote": ["Bold marks the highest accuracy."],
            "table_body": "<table><tr><th>Method</th><th>Accuracy (%)</th></tr><tr><td>A</td><td><b>90</b></td></tr><tr><td>B</td><td>80</td></tr></table>",
            "page_idx": 0,
            "bbox": [20, 20, 300, 145],
            "bbox_space": "pdf_points",
        },
        {
            "type": "table",
            "table_caption": ["Table 10. Other results."],
            "table_body": "<table><tr><th>Value</th></tr><tr><td>50</td></tr></table>",
            "page_idx": 0,
            "bbox": [20, 145, 300, 245],
            "bbox_space": "pdf_points",
        },
        {
            "type": "image",
            "image_caption": ["Figure 1. Architecture."],
            "page_idx": 0,
            "bbox": [300, 20, 350, 145],
            "bbox_space": "pdf_points",
        },
        {"type": "text", "text": "References", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "[1] Table 1 in a different work. 2020.", "page_idx": 0},
    ]
    markdown = "\n\n".join(
        ("# " if row.get("text_level") else "")
        + (
            row.get("text")
            or "\n".join(row.get("table_caption", row.get("image_caption", [])))
            + ("\n" + row["table_body"] if row.get("table_body") else "")
        )
        for row in rows
    )
    return pdf_path, MineruParseResult(markdown, rows, None, "fixture", None, "fixture")


def materialize(table_paper, tmp_path):
    pdf_path, parsed = table_paper
    return build_materials(parsed, paper_pdf=pdf_path, output_dir=tmp_path / "materials", paper_key="tables")


def test_table_crops_and_references_are_independent_of_figures(table_paper, tmp_path):
    materials = materialize(table_paper, tmp_path)
    assert len(materials.tables) == 2
    assert len(materials.figures) == 1
    first, tenth = materials.tables
    assert first.block_id == "block_4" and first.loc.page == 1
    assert first.caption == "Table 1. Accuracy by method."
    assert first.anchor == "1" and tenth.anchor == "10"
    assert first.bbox_points == (20, 20, 300, 145)
    with Image.open(first.printed_crop_path) as image:
        assert image.size == (round(280 * 96 / 72), round(125 * 96 / 72))
    with Image.open(first.crop_path) as image:
        assert image.width > 400
    assert [ref.text for ref in first.references] == [
        "Table 1 compares two methods.",
        "Tables 1 and 10 share units.",
        "Tab. 1 has two methods.",
    ]
    assert [ref.text for ref in tenth.references] == [
        "Table 10 has other results.",
        "Tables 1 and 10 share units.",
    ]
    assert [ref.text for ref in materials.figures[0].references] == ["Figure 1 shows the architecture."]
    for table in materials.tables:
        for reference in table.references:
            assert materials.markdown[reference.loc.char_start : reference.loc.char_end] == reference.text
    stored = json.loads((tmp_path / "materials/materials.json").read_text(encoding="utf-8"))
    assert len(stored["tables"]) == 2
    assert materials.markdown == table_paper[1].markdown


def test_legacy_materials_without_tables_remain_readable():
    material = SharedMaterials.model_validate(
        {
            "paper_key": "legacy",
            "source_pdf": "paper.pdf",
            "markdown": "",
            "markdown_path": "paper.md",
            "content_list_path": "",
            "provider": "legacy",
        }
    )
    assert material.tables == []


@pytest.fixture
def visual_materials(table_paper, tmp_path, monkeypatch):
    from llm.client import LLMConfig

    config = LLMConfig(provider="mock", model="mock", api_key=None, base_url=None)
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: config)
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda **_: config)
    return materialize(table_paper, tmp_path)


def issue(category="legibility"):
    return {
        "category": category,
        "disposition": "issue",
        "text": "Small row labels are unreadable in the supplied crop.",
    }


def test_visual_request_uses_real_crop_caption_all_references_and_source_pointer(visual_materials):
    from screening.tables import check_visual_tables

    seen = []
    before = visual_materials.model_dump(mode="json")

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_tables.context":
            return {
                "schema_version": "table-context-v1",
                "context_id": payload["context_id"],
                "table_id": payload["table_id"],
                "target": "matched",
                "caption_source_id": payload["caption_source"]["id"],
                "decisions": [
                    {
                        "candidate_id": c["candidate_id"],
                        "classification": "manuscript_issue",
                        "witness_span_ids": payload["caption_source"]["page_span_ids"],
                        "reason": "The original-page table retains the unexplained asterisk candidate.",
                    }
                    for c in payload["candidates"]
                ],
            }
        assert kwargs["module"] == "screening_tables.visual"
        table = next(table for table in visual_materials.tables if table.id == payload["table_id"])
        assert kwargs["images"] == [table.printed_crop_path]
        with Image.open(kwargs["images"][0]) as image:
            assert image.width > 0
        assert payload["caption"] == table.caption
        assert payload["footnotes"] == table.footnotes
        assert payload["references"] == [ref.model_dump() for ref in table.references]
        assert payload["printed_size_verified"] is True
        assert payload["printed_dpi"] == 96
        seen.append(table.id)
        if table.id == "table_1":
            return {
                "findings": [
                    {
                        "category": "self_containedness",
                        "disposition": "issue",
                        "text": "An asterisk attached to row A has no explained meaning in the table, caption, table footnotes or supplied body references.",
                    }
                ]
            }
        return {
            "findings": [
                {"category": "legibility", "disposition": "clear", "text": "The labels are readable."}
            ]
        }

    records = []
    findings, issues = check_visual_tables(visual_materials, call=call, records=records)
    assert seen == ["table_1", "table_2"]
    assert len(findings) == 1 and issues == []
    finding = findings[0]
    assert finding.kind == "table" and finding.level == "self_containedness"
    evidence = finding.evidence[0]
    assert evidence.source == "paper_internal" and not evidence.affects_claim
    # The caption's complete PDF span starts at y=18.175, above the crop's y=20.
    assert evidence.pointer.locator == visual_materials.source_pdf
    with fitz.open(visual_materials.source_pdf) as pdf:
        assert evidence.pointer.quote in pdf[0].get_text()
    assert evidence.pointer.page == 1 and evidence.pointer.key == "table_1"
    assert evidence.pointer.quote == visual_materials.tables[0].caption
    assert [row.status for row in records] == ["checked", "checked"]
    assert [row.finding_count for row in records] == [1, 0]
    assert visual_materials.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "failure",
    ["provider", "exception", "invalid_category", "invalid_disposition", "empty_text", "missing_findings"],
)
def test_failed_table_is_atomic_and_preserves_both_neighbors(visual_materials, failure):
    from screening.tables import check_visual_tables

    visual_materials.tables.append(visual_materials.tables[0].model_copy(update={"id": "table_3"}, deep=True))
    seen = []

    def call(**kwargs):
        table_id = json.loads(kwargs["prompt"])["table_id"]
        seen.append(table_id)
        if table_id != "table_2":
            return {"findings": [issue()]}
        if failure == "provider":
            return {"status": "error", "error": "fixed provider failure"}
        if failure == "exception":
            raise TimeoutError("fixed timeout")
        if failure == "missing_findings":
            return {}
        invalid = issue()
        invalid.update(
            {
                "invalid_category": {"category": "aesthetics"},
                "invalid_disposition": {"disposition": "supported"},
                "empty_text": {"text": " "},
            }[failure]
        )
        return {"findings": [issue(), invalid]}

    records = []
    findings, issues = check_visual_tables(visual_materials, call=call, recover_errors=True, records=records)
    assert seen == ["table_1", "table_2", "table_3"]
    assert [f.evidence[0].pointer.key for f in findings] == ["table_1", "table_3"]
    assert [record.status for record in records] == ["checked", "failed", "checked"]
    assert records[1].finding_count == 0 and records[1].issues
    assert any("table_2" in message and "failed" in message for message in issues)


def test_strict_direct_table_call_rejects_malformed_response(visual_materials):
    from screening.tables import check_visual_tables

    with pytest.raises(ValueError, match="categories"):
        check_visual_tables(visual_materials, call=lambda **_: {"findings": [issue("colour")]})


@pytest.mark.parametrize("invalid", ["missing", "corrupt", "pixel_size", "dpi", "bbox", "location"])
def test_invalid_table_image_is_unavailable_without_model_call(visual_materials, invalid):
    from screening.tables import check_visual_tables

    table = visual_materials.tables[0]
    image = Path(table.printed_crop_path)
    if invalid == "missing":
        image.unlink()
    elif invalid == "corrupt":
        image.write_bytes(b"invalid image")
    elif invalid == "pixel_size":
        Image.new("RGB", (400, 400)).save(image)
    elif invalid == "dpi":
        table.printed_dpi = 200
    elif invalid == "bbox":
        table.bbox_points = (0, 0, float("inf"), 20)
    else:
        table.loc = None
    seen = []

    def call(**kwargs):
        seen.append(json.loads(kwargs["prompt"])["table_id"])
        return {"findings": []}

    records = []
    findings, issues = check_visual_tables(visual_materials, call=call, records=records)
    assert seen == ["table_2"] and not findings
    assert [record.status for record in records] == ["unavailable", "checked"]
    assert any("table_1" in message and "unavailable" in message for message in issues)


def test_missing_physical_scale_never_confirms_legibility(visual_materials):
    from screening.tables import check_visual_tables

    visual_materials.tables = visual_materials.tables[:1]
    visual_materials.tables[0].bbox_points = None
    records = []
    findings, issues = check_visual_tables(
        visual_materials,
        call=lambda **_: {"findings": [issue(), issue("text_table_consistency")]},
        records=records,
    )
    assert findings == []
    assert any("printed-size legibility unavailable" in message for message in issues)
    assert records[0].printed_size_verified is False
    assert records[0].context_status == "unavailable"
    assert records[0].crop_response == {"findings": [issue(), issue("text_table_consistency")]}
    assert any(issue("text_table_consistency")["text"] in message for message in issues)


def test_missing_scale_is_visible_even_when_model_finds_no_issues(visual_materials):
    from screening.tables import check_visual_tables

    visual_materials.tables = visual_materials.tables[:1]
    visual_materials.tables[0].bbox_points = None
    records = []
    findings, issues = check_visual_tables(
        visual_materials, call=lambda **_: {"findings": []}, records=records
    )
    assert findings == [] and records[0].status == "checked"
    assert any("printed-size legibility unavailable" in message for message in issues)


def test_caption_assignment_ambiguity_downgrades_context_dependent_findings(visual_materials):
    from screening.tables import check_visual_tables

    visual_materials.tables = visual_materials.tables[:1]
    visual_materials.tables[0].caption_ambiguous = True
    records = []
    findings, issues = check_visual_tables(
        visual_materials,
        call=lambda **_: {
            "findings": [issue("self_containedness"), issue("text_table_consistency"), issue()]
        },
        records=records,
    )
    assert [f.level for f in findings] == ["legibility"]
    for category in ("self_containedness", "text_table_consistency"):
        assert (
            f"table_1: {category} unconfirmed because parser caption assignment is ambiguous: {issue(category)['text']}"
            in issues
        )
    assert records[0].context_status == "failed"
    assert any("original-page context failed" in message for message in issues)


def test_uncertain_observation_and_missing_caption_remain_visible(visual_materials):
    from screening.tables import check_visual_tables

    visual_materials.tables = visual_materials.tables[:1]
    table = visual_materials.tables[0]
    table.caption, table.references = "", []
    findings, issues = check_visual_tables(
        visual_materials,
        call=lambda **_: {"findings": [issue(), {**issue("self_containedness"), "disposition": "uncertain"}]},
    )
    assert not findings
    assert any("caption/body reference unavailable" in message for message in issues)
    assert any("check uncertain" in message for message in issues)


@pytest.mark.parametrize("missing", ["bbox", "page_idx"])
def test_parser_missing_crop_metadata_stays_visible_in_table_record(table_paper, tmp_path, missing):
    from screening.tables import check_visual_tables

    table_paper[1].content_list[3].pop(missing)
    materials = materialize(table_paper, tmp_path)
    materials.tables = materials.tables[:1]
    records = []
    findings, issues = check_visual_tables(
        materials, call=lambda **_: pytest.fail("unavailable crop must not call VLM"), records=records
    )
    assert not findings and records[0].status == "unavailable"
    assert any("crop/printed-size input unavailable" in message for message in issues)
    assert materials.issues == []  # New table diagnostics retain their own coverage boundary.


def test_table_normalized_bbox_and_caption_range_linkage(table_paper, tmp_path):
    table_paper[1].content_list[3].update(bbox=[100, 100, 900, 500], bbox_space="normalized_1000")
    table_paper[1].content_list[1]["text"] = "Tables 1–2 and 10 share units."
    # Source content-list row may have a PDF-only location; preserve that contract.
    materials = materialize(table_paper, tmp_path)
    assert materials.tables[0].bbox_points == (36, 30, 324, 150)
    assert any(ref.text == "Tables 1–2 and 10 share units." for ref in materials.tables[0].references)
    assert any(ref.text == "Tables 1–2 and 10 share units." for ref in materials.tables[1].references)


def test_parser_multiple_table_captions_is_explicitly_ambiguous(table_paper, tmp_path):
    table_paper[1].content_list[3]["table_caption"] = ["Table 1. Results.", "Table 2. Other results."]
    table = materialize(table_paper, tmp_path).tables[0]
    assert table.caption_ambiguous and table.anchor == ""
    assert any("assignment is ambiguous" in message for message in table.issues)


def test_zero_table_coverage_has_no_model_requests(visual_materials):
    from screening.tables import check_visual_tables

    visual_materials.tables = []
    records = []
    findings, issues = check_visual_tables(
        visual_materials, call=lambda **_: pytest.fail("no tables"), records=records
    )
    assert findings == issues == records == []


def test_original_text_table_check_still_receives_only_exact_parsed_text(visual_materials):
    from screening.checks import check_tables

    seen = []

    def call(**kwargs):
        assert kwargs["module"] == "screening_tables" and not kwargs.get("images")
        payload = json.loads(kwargs["prompt"])
        seen.extend(payload["tables"])
        return {"findings": []}

    assert check_tables(visual_materials, call=call) == []
    assert seen == [block.model_dump() for block in visual_materials.blocks if block.kind == "table"]


def test_parser_footnotes_reach_vlm_without_changing_parsed_table_block(visual_materials):
    from screening.tables import check_visual_tables

    visual_materials.tables = visual_materials.tables[:1]
    table = visual_materials.tables[0]
    assert table.footnotes == "Bold marks the highest accuracy."
    original = next(block for block in visual_materials.blocks if block.id == table.block_id).text
    assert "Bold marks" not in original

    def call(**kwargs):
        data = json.loads(kwargs["prompt"])
        assert data["footnotes"] == "Bold marks the highest accuracy."
        return {
            "findings": [
                {
                    "category": "self_containedness",
                    "disposition": "clear",
                    "text": "Bold values are explained by the supplied table footnote.",
                }
            ]
        }

    findings, issues = check_visual_tables(visual_materials, call=call)
    assert findings == issues == []
    assert next(block for block in visual_materials.blocks if block.id == table.block_id).text == original
