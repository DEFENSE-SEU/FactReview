"""Original-page confirmation uses real local PDF pixels and mocked model calls."""

import copy
import json
from pathlib import Path

import fitz
import pytest

from llm.client import LLMConfig
from preprocessing.materials import build_materials
from preprocessing.parse.mineru_adapter import MineruParseResult
from screening import checks
from screening.figures import check_figures


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*a, **kw):
        raise AssertionError("External service/process forbidden")

    cfg = LLMConfig("mock", "figure-context", None, None)
    monkeypatch.setattr(checks, "resolve_llm_config", lambda: cfg)
    monkeypatch.setattr(checks, "resolve_vlm_config", lambda **kw: cfg)
    monkeypatch.setattr(checks, "llm_json", blocked)
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("requests.sessions.Session.request", blocked)
    monkeypatch.setattr("subprocess.run", blocked)


def paper(tmp_path, *, count=1, same_page=False, wrap=False, printed_caption=None, duplicate_caption=False):
    pdf_path = tmp_path / "paper.pdf"
    rows = []
    with fitz.open() as pdf:
        for index in range(count):
            page = pdf[0] if same_page and index else pdf.new_page(width=500, height=600)
            offset = 270 * index if same_page else 0
            label = f"Number {index + 1}"
            caption = f"Figure {index + 1}. Recorded counts for model {index + 1}."
            page.insert_text((40, offset + 55), label)
            page.draw_rect((50, offset + 70, 230, offset + 180))
            page.insert_text((90, offset + 110), str(index + 3))
            actual_caption = printed_caption or caption
            if wrap:
                page.insert_text((40, offset + 210), actual_caption.replace("counts for", "counts\nfor"))
            else:
                page.insert_text((40, offset + 210), actual_caption)
            if duplicate_caption:
                page.insert_text((40, offset + 250), actual_caption)
            rows.append(
                {
                    "type": "chart",
                    "chart_caption": [label, caption],
                    "page_idx": 0 if same_page else index,
                    "bbox": [50, offset + 70, 230, offset + 180],
                    "bbox_space": "pdf_points",
                }
            )
        pdf.save(pdf_path)
    markdown = "\n\n".join("\n".join(row["chart_caption"]) for row in rows)
    parsed = MineruParseResult(markdown, rows, None, "fixture", {}, "fixture")
    return build_materials(parsed, paper_pdf=pdf_path, output_dir=tmp_path / "materials", paper_key="figures")


def crop_response():
    return {
        "findings": [
            {
                "category": "self_containedness",
                "disposition": "issue",
                "text": "Axis meaning is absent from crop.",
            },
            {
                "category": "legibility",
                "disposition": "issue",
                "text": "Tick text is too small at printed size.",
            },
        ]
    }


def context_response(payload, *, classification="crop_artifact"):
    return {
        "schema_version": "figure-context-v1",
        "context_id": payload["context_id"],
        "figure_id": payload["figure_id"],
        "target": "matched",
        "part_roles": [
            {
                "part_id": part["id"],
                "role": "caption" if part["caption_label"] else "axis_label",
                "page_span_ids": part["matching_span_ids"],
                "reason": "Belongs to the selected figure.",
            }
            for part in payload["caption_parts"]
        ],
        "decisions": [
            {
                "candidate_id": candidate["candidate_id"],
                "classification": classification,
                "witness_span_ids": [payload["caption_parts"][0]["matching_span_ids"][0]],
                "reason": "Original page establishes the candidate's context.",
            }
            for candidate in payload["candidates"]
        ],
    }


def test_original_crop_and_ambiguity_preserved_with_bounded_context(tmp_path):
    materials = paper(tmp_path)
    before = materials.model_dump(mode="json")
    response = crop_response()
    calls, records = [], []

    def model(**kw):
        calls.append(kw)
        if kw["module"] == "screening_figures":
            return copy.deepcopy(response)
        return context_response(json.loads(kw["prompt"]))

    findings, issues = check_figures(materials, call=model, records=records)
    assert [c["module"] for c in calls] == ["screening_figures", "screening_figures.context"]
    assert len(calls[0]["images"]) == 1 and len(calls[1]["images"]) == 2
    assert [f.level for f in findings] == ["legibility"]
    assert any("crop_artifact" in issue for issue in issues)
    assert records[0].crop_response == response and records[0].context_status == "checked"
    assert materials.model_dump(mode="json") == before


def test_crop_mutation_after_callback_revokes_even_legibility(tmp_path):
    materials = paper(tmp_path)
    records = []

    def model(**kw):
        Path(materials.figures[0].printed_crop_path).write_bytes(b"changed")
        return crop_response()

    findings, issues = check_figures(materials, call=model, recover_errors=True, records=records)
    assert not findings and records[0].status == "failed"
    assert any("changed" in issue for issue in issues)


def test_later_figure_cannot_mutate_prior_crop_and_leave_its_findings(tmp_path):
    materials = paper(tmp_path, count=2)
    records = []

    def model(**kw):
        payload = json.loads(kw["prompt"])
        if kw["module"] == "screening_figures.context":
            return context_response(payload)
        if payload["figure_id"] == "figure_2":
            Path(materials.figures[0].printed_crop_path).write_bytes(b"changed later")
        return crop_response()

    findings, _ = check_figures(materials, call=model, recover_errors=True, records=records)
    assert [record.status for record in records] == ["failed", "checked"]
    assert [f.evidence[0].pointer.key for f in findings] == ["figure_2"]


def run_model(materials, transform=None, *, raw=None):
    calls, records = [], []

    def model(**kw):
        payload = json.loads(kw["prompt"])
        calls.append((kw["module"], payload))
        if kw["module"] == "screening_figures":
            return copy.deepcopy(crop_response() if raw is None else raw)
        response = context_response(payload)
        return transform(response, payload) if transform else response

    findings, issues = check_figures(materials, call=model, recover_errors=True, records=records)
    return findings, issues, records, calls


def test_layout_only_caption_alignment_retains_both_original_texts(tmp_path):
    materials = paper(tmp_path, wrap=True)
    findings, _, records, _ = run_model(materials)
    assert [f.level for f in findings] == ["legibility"]
    record = records[0]
    assert record.context_status == "checked"
    caption = record.context_record["caption_parts"][1]
    span = next(s for s in record.context_record["page_spans"] if s["id"] == caption["matching_span_ids"][0])
    assert caption["alignment"] == "whitespace_layout"
    assert "\n" not in caption["text"] and "\n" in span["text"]
    assert caption["text"].split() == span["text"].split()


@pytest.mark.parametrize(
    "actual",
    [
        "Figure 1. Recorded counts for model 9.",
        "Figure 1. Recorded counts for model +1.",
        "Figure 1. recorded counts for model 1.",
    ],
)
def test_caption_numeric_symbol_case_changes_are_not_layout_alignment(tmp_path, actual):
    _, issues, records, calls = run_model(paper(tmp_path, printed_caption=actual))
    assert records[0].context_status == "unavailable"
    assert len(calls) == 1 and any("unchanged tokens" in i for i in issues)


def test_duplicate_numbered_caption_cannot_establish_identity(tmp_path):
    _, _, records, calls = run_model(paper(tmp_path, duplicate_caption=True))
    assert records[0].context_status == "unavailable" and len(calls) == 1


def test_numbered_caption_cannot_be_assembled_from_neighbor_blocks(tmp_path):
    materials = paper(tmp_path, printed_caption="Figure 1. Recorded counts")
    source = Path(materials.source_pdf)
    with fitz.open(source) as pdf:
        pdf[0].insert_text((300, 410), "for model 1.")
        pdf.saveIncr()
        pdf[0].get_pixmap(dpi=200).save(materials.pages[0].path)
    _, _, records, calls = run_model(materials)
    assert records[0].context_status == "unavailable" and len(calls) == 1


@pytest.mark.parametrize("mismatch", ["crop", "page"])
def test_same_size_image_exchange_fails_original_pdf_pixel_binding(tmp_path, mismatch):
    materials = paper(tmp_path, count=2)
    if mismatch == "crop":
        Path(materials.figures[0].printed_crop_path).write_bytes(
            Path(materials.figures[1].printed_crop_path).read_bytes()
        )
    else:
        Path(materials.pages[0].path).write_bytes(Path(materials.pages[1].path).read_bytes())
    findings, _, records, calls = run_model(materials)
    assert records[0].status == "failed" and records[1].status == "checked"
    assert all(p["figure_id"] == "figure_2" for _, p in calls)
    assert [f.evidence[0].pointer.key for f in findings] == ["figure_2"]


@pytest.mark.parametrize(
    "bad",
    [
        "version",
        "context",
        "figure",
        "omitted",
        "duplicate",
        "unknown",
        "legibility",
        "part_duplicate",
        "part_unknown",
        "role",
        "witness",
        "empty_witness",
    ],
)
def test_context_response_closed_set_is_atomic(tmp_path, bad):
    def alter(response, payload):
        if bad == "version":
            response["schema_version"] = "future"
        elif bad == "context":
            response["context_id"] = "other"
        elif bad == "figure":
            response["figure_id"] = "figure_99"
        elif bad == "omitted":
            response["decisions"] = []
        elif bad == "duplicate":
            response["decisions"].append(copy.deepcopy(response["decisions"][0]))
        elif bad == "unknown":
            response["decisions"][0]["candidate_id"] = "candidate_999"
        elif bad == "legibility":
            response["decisions"][0]["category"] = "legibility"
        elif bad == "part_duplicate":
            response["part_roles"].append(copy.deepcopy(response["part_roles"][0]))
        elif bad == "part_unknown":
            response["part_roles"][0]["part_id"] = "unknown"
        elif bad == "role":
            response["part_roles"][0]["role"], response["part_roles"][1]["role"] = "caption", "axis_label"
        elif bad == "witness":
            response["decisions"][0]["witness_span_ids"] = ["page:99:block:0"]
        else:
            response["decisions"][0]["witness_span_ids"] = []
        return response

    findings, issues, records, calls = run_model(paper(tmp_path), alter)
    assert len(calls) == 2 and records[0].context_status == "failed"
    assert [f.level for f in findings] == ["legibility"]
    assert records[0].crop_response == crop_response()
    assert any("parser caption assignment is ambiguous" in issue for issue in issues)


def test_same_page_neighbor_label_cannot_confirm_target(tmp_path):
    def neighbor(response, payload):
        if payload["figure_id"] == "figure_1":
            foreign = next(s for s in payload["page_spans"] if s["text"] == "Number 2")
            response["decisions"][0]["witness_span_ids"] = [foreign["id"]]
        return response

    findings, _, records, _ = run_model(paper(tmp_path, count=2, same_page=True), neighbor)
    assert [r.context_status for r in records] == ["failed", "checked"]
    assert all(f.level == "legibility" for f in findings)


def test_original_manuscript_issue_has_both_crop_and_exact_pdf_witness(tmp_path):
    def issue(response, payload):
        response["decisions"][0]["classification"] = "manuscript_issue"
        return response

    materials = paper(tmp_path)
    findings, _, records, _ = run_model(materials, issue)
    issue_finding = next(f for f in findings if f.level == "self_containedness")
    assert len(issue_finding.evidence) == 2
    assert issue_finding.evidence[1].pointer.locator == materials.source_pdf
    assert issue_finding.evidence[1].pointer.quote == "Number 1"
    assert issue_finding.evidence[1].pointer.page == 1
    assert records[0].context_record["caption_assignment"] == "confirmed"
    assert materials.figures[0].caption_ambiguous


@pytest.mark.parametrize("phase", ["screening_figures", "screening_figures.context"])
@pytest.mark.parametrize(
    "change",
    ["crop", "page_bytes", "page_path", "pdf", "parser", "caption", "bbox", "references", "replacement"],
)
def test_callback_source_changes_revoke_all_figure_results(tmp_path, phase, change):
    materials = paper(tmp_path)
    records = []

    def model(**kw):
        payload = json.loads(kw["prompt"])
        if kw["module"] == phase:
            f = materials.figures[0]
            if change == "crop":
                Path(f.printed_crop_path).write_bytes(b"changed")
            elif change == "page_bytes":
                Path(materials.pages[0].path).write_bytes(b"changed")
            elif change == "page_path":
                materials.pages[0].path = str(tmp_path / "other.png")
            elif change == "pdf":
                Path(materials.source_pdf).write_bytes(b"changed")
            elif change == "parser":
                Path(materials.content_list_path).write_text("[]")
            elif change == "caption":
                f.caption = "Changed source caption"
            elif change == "bbox":
                f.bbox_points = (0, 0, 180, 110)
            elif change == "references":
                f.references = [materials.blocks[0]]
            else:
                materials.figures = [f.model_copy(update={"anchor": "99"})]
        return crop_response() if kw["module"] == "screening_figures" else context_response(payload)

    findings, issues = check_figures(materials, call=model, recover_errors=True, records=records)
    assert findings == [] and records[0].status == "failed"
    assert any("changed" in issue for issue in issues)


def test_failed_context_is_local_and_never_retried(tmp_path):
    def fail(response, payload):
        if payload["figure_id"] == "figure_1":
            raise TimeoutError("injected context failure")
        return response

    _, _, records, calls = run_model(paper(tmp_path, count=2), fail)
    assert [r.context_status for r in records] == ["failed", "checked"]
    assert len(calls) == 4


def test_later_figure_cannot_change_a_previously_consumed_page(tmp_path):
    materials = paper(tmp_path, count=2)

    def mutate(response, payload):
        if payload["figure_id"] == "figure_2":
            Path(materials.pages[0].path).write_bytes(b"changed previous page")
        return response

    findings, _, records, _ = run_model(materials, mutate)
    assert [r.status for r in records] == ["failed", "checked"]
    assert [r.context_status for r in records] == ["failed", "checked"]
    assert [f.evidence[0].pointer.key for f in findings] == ["figure_2"]


def test_future_figure_baseline_cannot_be_changed_by_first_callback(tmp_path):
    materials = paper(tmp_path, count=2)

    def mutate(response, payload):
        if payload["figure_id"] == "figure_1":
            Path(materials.figures[1].printed_crop_path).write_bytes(b"changed future crop")
        return response

    _, _, records, calls = run_model(materials, mutate)
    assert [r.status for r in records] == ["checked", "failed"]
    assert len(calls) == 2


def test_multiple_original_candidates_share_one_bounded_context_call(tmp_path):
    raw = crop_response()
    raw["findings"].append(
        {"category": "text_figure_consistency", "disposition": "uncertain", "text": "Caption link unclear."}
    )
    _, _, records, calls = run_model(paper(tmp_path), raw=raw)
    assert len(calls) == 2
    assert [d["candidate_id"] for d in records[0].context_record["response"]["decisions"]] == [
        "candidate_0",
        "candidate_2",
    ]


def test_legacy_json_builds_sidecar_without_mutating_saved_materials(tmp_path):
    from schemas.materials import SharedMaterials

    current = paper(tmp_path)
    raw = current.model_dump(mode="json")
    for f in raw["figures"]:
        for key in ("block_id", "parser_row_index", "caption_parts"):
            f.pop(key)
    legacy = SharedMaterials.model_validate(raw)
    before = legacy.model_dump(mode="json")
    _, _, records, _ = run_model(legacy)
    assert records[0].context_status == "checked"
    assert legacy.model_dump(mode="json") == before


def test_unindexed_neighbor_figure_cannot_donate_context(tmp_path):
    materials = paper(tmp_path, count=2, same_page=True)
    materials.figures = materials.figures[:1]
    _, _, records, calls = run_model(materials)
    assert records[0].context_status == "unavailable" and len(calls) == 1


def test_no_context_call_for_clear_unambiguous_crop(tmp_path):
    materials = paper(tmp_path)
    # Preserve real parser provenance while using a caption-only original row.
    content = Path(materials.content_list_path)
    rows = json.loads(content.read_text(encoding="utf-8"))
    rows[0]["chart_caption"] = rows[0]["chart_caption"][1:]
    content.write_text(json.dumps(rows), encoding="utf-8")
    from preprocessing.materials import figure_caption_parts

    figure = materials.figures[0]
    figure.caption_parts = figure_caption_parts(rows[0])
    figure.caption = rows[0]["chart_caption"][0]
    figure.caption_ambiguous = False
    figure.anchor = "1"
    _, _, records, calls = run_model(materials, raw={"findings": []})
    assert records[0].context_status == "not_requested" and len(calls) == 1


def test_unknown_context_role_and_uncertain_target_cannot_be_decisive(tmp_path):
    def response(value, payload):
        value["target"] = "uncertain"
        return value

    findings, _, records, _ = run_model(paper(tmp_path), response)
    assert records[0].context_status == "failed"
    assert all(f.level == "legibility" for f in findings)


def test_both_safe_record_and_findings_redact_configured_credentials(tmp_path, monkeypatch):
    cfg = LLMConfig(
        "mock", "figure-context", base_url="https://example.test", api_key="fake-secret-for-figure"
    )
    monkeypatch.setattr(checks, "resolve_vlm_config", lambda **kw: cfg)
    raw = crop_response()
    raw["findings"][1]["text"] += " fake-secret-for-figure"

    def change(value, payload):
        value["decisions"][0]["reason"] += " fake-secret-for-figure"
        return value

    findings, issues, records, _ = run_model(paper(tmp_path), change, raw=raw)
    serialized = json.dumps(
        {
            "findings": [f.model_dump() for f in findings],
            "issues": issues,
            "records": [r.model_dump() for r in records],
        }
    )
    assert "fake-secret-for-figure" not in serialized


def purpose_paper(tmp_path, *, gallery):
    pdf_path = tmp_path / "purpose.pdf"
    caption = (
        "Figure 1. The model attains 78% accuracy using these ten labeled training images."
        if gallery
        else "Figure 1. Model A obtains 78% accuracy on the test set."
    )
    reference = (
        "The displayed samples are the full labeled training set; using them gives 78% test accuracy."
        if gallery
        else "Test accuracy for Model A is plotted in Figure 1."
    )
    with fitz.open() as pdf:
        page = pdf.new_page(width=500, height=600)
        if gallery:
            for i in range(10):
                x, y = 50 + (i % 5) * 70, 80 + (i // 5) * 65
                page.draw_rect((x, y, x + 50, y + 45))
                page.insert_text((x + 3, y + 25), f"Sample {i + 1}", fontsize=7)
        else:
            page.draw_rect((65, 105, 115, 200), fill=(0.1, 0.4, 0.7))
            page.insert_text((60, 95), "Accuracy (%)")
            page.insert_text((75, 120), "65%")
        page.insert_textbox((40, 220, 460, 270), caption, fontsize=10)
        page.insert_textbox((40, 290, 460, 340), reference, fontsize=10)
        pdf.save(pdf_path)
    rows = [
        {"type": "text", "text": reference, "page_idx": 0},
        {
            "type": "chart",
            "chart_caption": [caption],
            "page_idx": 0,
            "bbox": [40, 70, 430, 210],
            "bbox_space": "pdf_points",
        },
    ]
    parsed = MineruParseResult(reference + "\n\n" + caption, rows, None, "fixture", {}, "fixture")
    return build_materials(
        parsed, paper_pdf=pdf_path, output_dir=tmp_path / "purpose_materials", paper_key="purpose"
    )


@pytest.mark.parametrize("classification", ["no_manuscript_issue", "manuscript_issue"])
def test_caption_context_decision_preserves_figure_purpose_boundary(tmp_path, classification):
    # Fixed model controls exercise adoption only; they do not measure VLM understanding.
    raw = {
        "findings": [
            {
                "category": "text_figure_consistency",
                "disposition": "issue",
                "text": "The figure does not display the caption-reported accuracy.",
            }
        ]
    }

    def decision(response, payload):
        response["decisions"][0]["classification"] = classification
        response["decisions"][0]["witness_span_ids"] = list(payload["caption_parts"][0]["matching_span_ids"])
        if classification == "manuscript_issue":
            response["decisions"][0]["witness_span_ids"].append(
                next(s["id"] for s in payload["page_spans"] if s["text"] == "65%")
            )
        response["decisions"][0]["reason"] = (
            "The caption supplies context to the example display."
            if classification == "no_manuscript_issue"
            else "The original plotted value contradicts the caption value."
        )
        return response

    materials = purpose_paper(tmp_path, gallery=classification == "no_manuscript_issue")
    findings, issues, records, calls = run_model(materials, decision, raw=raw)
    assert len(calls) == 2 and records[0].crop_response == raw
    assert records[0].context_record["response"]["decisions"][0]["classification"] == classification
    if classification == "no_manuscript_issue":
        assert not findings and any("no_manuscript_issue" in i for i in issues)
        assert not any("crop_artifact" in i or "crop boundary coverage" in i for i in issues)
    else:
        assert len(findings) == 1 and findings[0].level == "text_figure_consistency"


def test_first_model_cannot_override_program_candidate_ids(tmp_path):
    raw = crop_response()
    raw["findings"][0]["candidate_id"] = "candidate_99"
    _, _, records, calls = run_model(paper(tmp_path), raw=raw)
    assert calls[1][1]["candidates"][0]["candidate_id"] == "candidate_0"
    assert records[0].crop_response == raw


def test_page_confirmation_cannot_infer_missing_figure_purpose(tmp_path):
    def uncertain(response, payload):
        response["target"] = "uncertain"
        for item in response["decisions"]:
            item["classification"] = "uncertain"
        return response

    findings, issues, records, _ = run_model(paper(tmp_path), uncertain)
    assert records[0].context_status == "checked"
    assert records[0].context_record["caption_assignment"] == "uncertain"
    assert [f.level for f in findings] == ["legibility"]
    assert any("assignment remains uncertain" in issue for issue in issues)
