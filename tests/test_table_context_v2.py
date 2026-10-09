"""Table page context: real local PDF sources and strictly mocked visual calls."""

import copy
import json
from pathlib import Path

import fitz
import pytest

from llm.client import LLMConfig
from preprocessing.materials import build_materials
from preprocessing.parse.mineru_adapter import MineruParseResult
from screening import checks
from screening.table_context import TableSources
from screening.tables import check_visual_tables


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("External service/process forbidden")

    cfg = LLMConfig("mock", "table-context", None, None)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: cfg)
    monkeypatch.setattr(checks, "resolve_vlm_config", lambda **kw: cfg)
    monkeypatch.setattr(checks, "llm_json", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def paper(
    tmp_path,
    *,
    count=1,
    same_page=False,
    caption_mode="adjacent",
    wrap=False,
    above=False,
    duplicate=False,
    actual_caption=None,
    parser_caption=None,
):
    path, rows = tmp_path / "paper.pdf", []
    with fitz.open() as pdf:
        for index in range(count):
            page = pdf[0] if same_page and index else pdf.new_page(width=600, height=650)
            top = index * 290 if same_page else 0
            caption = f"Table {index + 7}. Model counts. M has six choices: box, nearest, cubic, linear, lanczos, hamming."
            caption = parser_caption or caption
            display = actual_caption or caption
            if wrap:
                display = display.replace("six choices:", "six\nchoices:")
            y = top + (50 if above else 230)
            page.insert_text((40, y), display, fontsize=9)
            if duplicate:
                page.insert_text((40, top + 270), display, fontsize=9)
            page.insert_text((60, top + 100), "Method       Accuracy (%)       M")
            page.insert_text((60, top + 135), "A                 90          see caption")
            page.draw_rect((40, top + 75, 560, top + 190))
            table = {
                "type": "table",
                "page_idx": 0 if same_page else index,
                "bbox": [40, top + 75, 560, top + 190],
                "bbox_space": "pdf_points",
                "table_caption": [caption] if caption_mode == "native" else [],
                "table_body": "<table><tr><td>A</td><td>90</td><td>see caption</td></tr></table>",
            }
            text = {
                "type": "text",
                "text": caption,
                "page_idx": 0 if same_page else index,
                "bbox": [35, y - 12, 570, y + (16 if wrap else 5)],
                "bbox_space": "pdf_points",
            }
            rows.extend(
                ([text, table] if above else [table, text]) if caption_mode == "adjacent" else [table]
            )
        pdf.save(path)
    md = "\n\n".join(
        r.get("text", "\n".join(r.get("table_caption", [])) + r.get("table_body", "")) for r in rows
    )
    parsed = MineruParseResult(md, rows, None, "fixture", {}, "fixture")
    return build_materials(
        parsed, paper_pdf=path, output_dir=tmp_path / "materials", paper_key="table-context"
    )


def crop():
    return {
        "findings": [
            {
                "category": "self_containedness",
                "disposition": "issue",
                "text": "M choices are missing from crop.",
            },
            {"category": "legibility", "disposition": "issue", "text": "Printed row labels are too small."},
        ]
    }


def decision(payload, classification="crop_artifact"):
    return {
        "schema_version": "table-context-v1",
        "context_id": payload["context_id"],
        "table_id": payload["table_id"],
        "target": "matched",
        "caption_source_id": payload["caption_source"]["id"],
        "decisions": [
            {
                "candidate_id": c["candidate_id"],
                "classification": classification,
                "witness_span_ids": payload["caption_source"]["page_span_ids"],
                "reason": "Original caption provides M choices.",
            }
            for c in payload["candidates"]
        ],
    }


def run(materials, transform=None, raw=None):
    calls, records = [], []

    def model(**kw):
        payload = json.loads(kw["prompt"])
        calls.append(kw)
        if kw["module"] == "screening_tables.visual":
            return copy.deepcopy(crop() if raw is None else raw)
        response = decision(payload)
        return transform(response, payload) if transform else response

    findings, issues = check_visual_tables(materials, call=model, records=records, recover_errors=True)
    return findings, issues, records, calls


@pytest.mark.parametrize(
    "mode,above,wrap",
    [
        ("native", False, False),
        ("adjacent", False, False),
        ("adjacent", True, False),
        ("adjacent", False, True),
    ],
)
def test_unique_complete_caption_is_sidecar_without_changing_crop_or_printed_judgment(
    tmp_path, mode, above, wrap
):
    materials = paper(tmp_path, caption_mode=mode, above=above, wrap=wrap)
    before = materials.model_dump(mode="json")
    findings, issues, records, calls = run(materials)
    assert len(calls) == 2 and [len(c["images"]) for c in calls] == [1, 2]
    assert records[0].context_status == "checked", issues
    assert records[0].crop_response == crop()
    assert [f.level for f in findings] == (["legibility"] if mode == "native" else [])
    assert "Table 7." in records[0].context_record["caption_source"]["text"]
    assert materials.tables[0].id == "table_1" and materials.model_dump(mode="json") == before
    assert records[0].context_record["caption_source"]["alignment"] == (
        "whitespace_layout" if wrap else "exact"
    )


@pytest.mark.parametrize("classification", ["manuscript_issue", "no_manuscript_issue", "uncertain"])
def test_confirmed_context_can_preserve_or_withhold_only_original_candidate(tmp_path, classification):
    materials = paper(tmp_path)
    findings, issues, records, _ = run(materials, lambda r, p: decision(p, classification))
    assert records[0].context_status == "checked", issues
    assert len(findings) == (1 if classification == "manuscript_issue" else 0)
    if findings:
        assert findings[0].level == "self_containedness" and all(
            not e.sufficient and not e.affects_claim for e in findings[0].evidence
        )
        assert findings[0].evidence[-1].pointer.locator == materials.source_pdf


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate",
        "symbols",
        "case",
        "missing",
        "foreign_label",
        "cross_page",
        "split_caption",
        "two_candidates",
    ],
)
def test_incomplete_or_wrong_original_caption_is_not_available(tmp_path, invalid):
    kwargs = {"duplicate": True} if invalid == "duplicate" else {}
    if invalid == "symbols":
        kwargs["actual_caption"] = (
            "Table 7. Model counts. M has seven choices: box, nearest, cubic, linear, lanczos, hamming."
        )
    if invalid == "case":
        kwargs["actual_caption"] = (
            "Table 7. Model counts. m has six choices: box, nearest, cubic, linear, lanczos, hamming."
        )
    if invalid == "missing":
        kwargs["actual_caption"] = "Results without a numbered caption."
    materials = paper(tmp_path, **kwargs)
    path = Path(materials.content_list_path)
    rows = json.loads(path.read_text())
    if invalid == "foreign_label":
        rows[1]["text"] = rows[1]["text"].replace("Table 7", "Table 8")
    if invalid == "cross_page":
        rows[1]["page_idx"] = 1
    if invalid == "split_caption":
        rows[1]["text"] = rows[1]["text"].split(" M has")[0]
    if invalid == "two_candidates":
        rows.insert(0, {**rows[1], "bbox": [35, 40, 570, 60]})
        materials.tables[0].block_id = "block_2"
        materials.tables[0].parser_row_index = 2
    path.write_text(json.dumps(rows))
    findings, issues, records, calls = run(materials)
    assert not findings and records[0].context_status == "unavailable", issues
    assert len(calls) == 1 and records[0].crop_response == crop()


@pytest.mark.parametrize(
    "invalid",
    [
        "version",
        "target_id",
        "context_id",
        "caption_id",
        "omitted",
        "extra",
        "duplicate",
        "witness",
        "empty_witness",
        "mismatch",
        "legibility",
    ],
)
def test_context_closed_response_rejects_unknown_or_unassigned_decisions(tmp_path, invalid):
    materials = paper(tmp_path, caption_mode="native")

    def mutate(r, p):
        if invalid == "version":
            r["schema_version"] = "unknown"
        elif invalid == "target_id":
            r["table_id"] = "table_7"
        elif invalid == "context_id":
            r["context_id"] = "other"
        elif invalid == "caption_id":
            r["caption_source_id"] = "neighbor"
        elif invalid == "omitted":
            r["decisions"] = []
        elif invalid == "extra":
            r["decisions"].append({**r["decisions"][0], "candidate_id": "extra"})
        elif invalid == "duplicate":
            r["decisions"] *= 2
        elif invalid == "witness":
            r["decisions"][0]["witness_span_ids"] = ["foreign"]
        elif invalid == "empty_witness":
            r["decisions"][0]["witness_span_ids"] = []
        elif invalid == "mismatch":
            r["target"] = "mismatch"
        elif invalid == "legibility":
            r["decisions"][0]["category"] = "legibility"
        return r

    findings, issues, records, calls = run(materials, mutate)
    assert [f.level for f in findings] == ["legibility"]
    assert records[0].context_status == "failed" and len(calls) == 2
    assert any("M choices are missing" in issue for issue in issues)


@pytest.mark.parametrize(
    "what,phase",
    [
        (x, y)
        for x in ("crop", "page", "pdf", "parser", "markdown", "blocks", "page_path", "table")
        for y in ("first", "context")
    ],
)
def test_callback_source_changes_revoke_all_affected_findings(tmp_path, what, phase):
    materials = paper(tmp_path, caption_mode="native")
    records = []

    def model(**kw):
        payload = json.loads(kw["prompt"])
        first = kw["module"] == "screening_tables.visual"
        if first == (phase == "first"):
            paths = {
                "crop": materials.tables[0].printed_crop_path,
                "page": materials.pages[0].path,
                "pdf": materials.source_pdf,
                "parser": materials.content_list_path,
                "markdown": materials.markdown_path,
            }
            if what in paths:
                Path(paths[what]).write_bytes(b"changed")
            elif what == "blocks":
                materials.blocks[0].text += " changed"
            elif what == "page_path":
                materials.pages[0].path = "changed.png"
            else:
                materials.tables[0] = materials.tables[0].model_copy(update={"caption": "changed"})
        return crop() if first else decision(payload, "manuscript_issue")

    findings, issues = check_visual_tables(materials, call=model, records=records, recover_errors=True)
    assert not findings and records[0].status == "failed", issues


@pytest.mark.parametrize("what,same_page", [("crop", False), ("page", False), ("page", True)])
def test_later_table_callback_revokes_prior_consumed_source(tmp_path, what, same_page):
    materials = paper(tmp_path, count=2, same_page=same_page, caption_mode="native")
    records = []

    def model(**kw):
        payload = json.loads(kw["prompt"])
        if kw["module"] == "screening_tables.visual":
            return crop()
        if payload["table_id"] == "table_2":
            Path(
                materials.tables[0].printed_crop_path if what == "crop" else materials.pages[0].path
            ).write_bytes(b"later change")
        return decision(payload, "manuscript_issue")

    findings, issues = check_visual_tables(materials, call=model, records=records, recover_errors=True)
    assert records[0].status == "failed" and records[0].finding_count == 0, issues
    assert records[1].status == ("failed" if same_page else "checked")
    assert all(f.evidence[0].pointer.key == "table_2" for f in findings)


def test_neighbor_caption_witness_cannot_be_borrowed(tmp_path):
    materials = paper(tmp_path, count=2, same_page=True, caption_mode="native")

    def mutate(r, p):
        foreign = next(
            s["id"]
            for s in p["page_spans"]
            if s["text"].startswith("Table 8." if p["table_id"] == "table_1" else "Table 7.")
        )
        r["decisions"][0]["witness_span_ids"] = [foreign]
        return r

    findings, _, records, _ = run(materials, mutate)
    assert [f.level for f in findings] == ["legibility", "legibility"]
    assert all(r.context_status == "failed" for r in records)


def test_crop_swap_is_rejected_before_any_model_call(tmp_path):
    materials = paper(tmp_path, count=2)
    # Give the second source different exact pixels without changing its image size.
    from PIL import Image

    Image.new("RGB", Image.open(materials.tables[0].printed_crop_path).size).save(
        materials.tables[0].printed_crop_path
    )
    _findings, _, records, calls = run(materials)
    assert records[0].status == "failed" and len(calls) == 2
    assert all(json.loads(c["prompt"])["table_id"] == "table_2" for c in calls)


def test_legacy_table_defaults_and_unavailable_origin_never_promote_context(tmp_path):
    materials = paper(tmp_path, caption_mode="native")
    materials.source_pdf = "missing.pdf"
    findings, issues, records, calls = run(materials)
    assert [f.level for f in findings] == ["legibility"]
    assert records[0].context_status == "unavailable" and len(calls) == 1
    assert any("M choices are missing" in issue for issue in issues)


def test_empty_candidate_context_only_does_not_create_findings(tmp_path):
    materials = paper(tmp_path)
    findings, _, records, calls = run(materials, raw={"findings": []})
    assert not findings and records[0].context_status == "checked" and len(calls) == 2
    assert records[0].context_record["response"]["decisions"] == []


def test_bound_sources_capture_current_page_and_parser_provenance(tmp_path):
    materials = paper(tmp_path)
    t = materials.tables[0]
    assert t.parser_row_index == 1 and t.parser_bbox_space == "normalized_1000"
    source = TableSources.capture(materials, t, {})
    context = source.prepare(materials)
    assert context is not None, source.context_error
    assert context["caption_source"]["parser_row_index"] == 2
    assert context["caption_source"]["markdown_loc"] is not None


@pytest.mark.parametrize(
    "caption",
    [
        "Table 7 and 8. Joint results.",
        "Table 7-8. Joint results.",
        "Table 7, 8. Joint results.",
        "Tables 7 and 8. Joint results.",
    ],
)
def test_shared_caption_cannot_establish_single_table_assignment(tmp_path, caption):
    materials = paper(tmp_path, caption_mode="native", parser_caption=caption)
    _, issues, records, calls = run(materials)
    assert records[0].context_status == "unavailable", issues
    assert len(calls) == 1


def test_valid_original_page_can_confirm_text_table_consistency_candidate(tmp_path):
    materials = paper(tmp_path)
    raw = {
        "findings": [
            {
                "category": "text_table_consistency",
                "disposition": "issue",
                "text": "Caption and row disagree about the displayed setting.",
            }
        ]
    }
    findings, issues, records, _ = run(materials, lambda r, p: decision(p, "manuscript_issue"), raw=raw)
    assert [f.level for f in findings] == ["text_table_consistency"], issues
    assert records[0].crop_response == raw and records[0].context_status == "checked"


def test_context_provider_failure_retains_valid_crop_legibility(tmp_path):
    materials = paper(tmp_path, caption_mode="native")

    def failure(r, p):
        raise TimeoutError("fixed context provider timeout")

    findings, issues, records, calls = run(materials, failure)
    assert [f.level for f in findings] == ["legibility"]
    assert records[0].status == "checked" and records[0].context_status == "failed"
    assert len(calls) == 2 and any("fixed context provider timeout" in i for i in issues)


def test_first_table_callback_cannot_replace_later_tables_baseline(tmp_path):
    materials = paper(tmp_path, count=2, caption_mode="native")
    records, calls = [], []

    def model(**kw):
        payload = json.loads(kw["prompt"])
        calls.append(payload["table_id"])
        if kw["module"] == "screening_tables.visual":
            Path(materials.tables[1].printed_crop_path).write_bytes(b"changed before second table")
            return crop()
        return decision(payload)

    findings, _issues = check_visual_tables(materials, call=model, records=records, recover_errors=True)
    assert calls == ["table_1", "table_1"]
    assert records[1].status == "failed" and records[1].finding_count == 0
    assert [f.evidence[0].pointer.key for f in findings] == ["table_1"]


def test_native_caption_cannot_be_reassigned_to_neighbor_table(tmp_path):
    materials = paper(tmp_path, count=2, same_page=True, caption_mode="native")
    rows_path = Path(materials.content_list_path)
    rows = json.loads(rows_path.read_text())
    rows[0]["table_caption"] = rows[1]["table_caption"]
    materials.tables[0].caption = materials.tables[1].caption
    materials.tables[0].anchor = materials.tables[1].anchor
    rows_path.write_text(json.dumps(rows))
    _, issues, records, calls = run(materials)
    assert records[0].context_status == "unavailable", issues
    assert len([c for c in calls if c["module"] == "screening_tables.context"]) == 1


def test_legacy_material_defaults_are_additive_and_original_source_remains_verifiable(tmp_path):
    from schemas.materials import SharedMaterials

    materials = paper(tmp_path)
    raw = materials.model_dump(mode="json")
    for t in raw["tables"]:
        t.pop("parser_row_index")
        t.pop("parser_bbox_space")
    restored = SharedMaterials.model_validate(raw)
    assert restored.tables[0].parser_row_index is None
    _, issues, records, _ = run(restored)
    assert records[0].context_status == "checked", issues


def test_neighbor_figure_pixels_cannot_be_used_as_table_witnesses(tmp_path):
    from schemas.claim import ClaimLocation
    from schemas.materials import FigureMaterial

    materials = paper(tmp_path, caption_mode="native")
    # The figure overlaps the proposed caption; no unique table assignment is available.
    materials.figures.append(
        FigureMaterial(
            id="figure_1",
            loc=ClaimLocation(page=1),
            caption="Figure 1. Other result.",
            bbox_points=(30, 215, 580, 250),
        )
    )
    _, issues, records, calls = run(materials)
    assert records[0].context_status == "unavailable", issues
    assert len(calls) == 1


def test_missing_page_image_does_not_bypass_available_pdf_crop_identity(tmp_path):
    from PIL import Image

    materials = paper(tmp_path)
    materials.pages = []
    crop_path = materials.tables[0].printed_crop_path
    with Image.open(crop_path) as image:
        size = image.size
    Image.new("RGB", size).save(crop_path)
    findings, issues, records, calls = run(materials)
    assert not findings and not calls
    assert records[0].status == "failed", issues


def test_recovered_caption_reference_lookup_excludes_bibliography(tmp_path):
    from schemas.claim import ClaimLocation
    from schemas.materials import MaterialBlock

    materials = paper(tmp_path)
    text = "Table 7 records the six transformation choices."
    start = len(materials.markdown) + 2
    materials.markdown += "\n\n" + text
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    block = MaterialBlock(
        id="body_reference",
        text=text,
        loc=ClaimLocation(page=1, char_start=start, char_end=start + len(text)),
    )
    materials.blocks.append(block)
    context = TableSources.capture(materials, materials.tables[0], {}).prepare(materials)
    assert any(r["text"] == text for r in context["located_references"])
    materials.bibliography.append(block.model_copy(deep=True))
    context = TableSources.capture(materials, materials.tables[0], {}).prepare(materials)
    assert not context["located_references"]
