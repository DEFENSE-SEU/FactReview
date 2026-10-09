"""Writing sources and safe display remain independent of model decisions."""

import json
from pathlib import Path

import pytest

from common import run_stats
from llm.client import LLMConfig
from screening.checks import check_writing
from tests.test_writing_sections_v2 import candidate, confirmations, make_materials


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail("This design control permits no external operation")

    monkeypatch.setattr("socket.socket.connect", deny)
    monkeypatch.setattr("socket.create_connection", deny)
    monkeypatch.setattr("subprocess.Popen", deny)
    text = LLMConfig(
        "mock",
        "text",
        "https://text-user:text-pass@invalid.example/v1?token=text-token",
        "fake-text-key-9843",
    )
    vision = LLMConfig(
        "mock",
        "vision",
        "https://vision-user:vision-pass@invalid.example/v1?token=vision-token",
        "fake-vision-key-6271",
    )
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: text)
    monkeypatch.setattr("screening.checks.resolve_vlm_config", lambda **kwargs: vision)
    return text, vision


@pytest.mark.parametrize("when", ["first", "validation"])
@pytest.mark.parametrize("change", ["bytes", "path", "metadata"])
def test_consumed_page_frozen_before_first_callback(tmp_path, when, change):
    materials = make_materials(tmp_path, [("a", "These results is fixed.")])
    records, calls = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        first = kwargs["module"] == "screening_writing"
        calls.append(kwargs["module"])
        if first == (when == "first"):
            page = materials.pages[0]
            if change == "bytes":
                path = Path(page.path)
                path.write_bytes(path.read_bytes() + b"local-test-mutation")
            elif change == "path":
                replacement = tmp_path / "replacement.png"
                replacement.write_bytes(Path(page.path).read_bytes())
                page.path = str(replacement)
            else:
                page.dpi += 1
        return {"findings": [candidate(payload["blocks"][0])]} if first else confirmations(payload)

    findings = check_writing(materials, call=call, records=records, recover_errors=True)
    assert not findings
    assert records[0].status in {"failed", "unavailable"}
    assert len(calls) <= 2


@pytest.mark.parametrize("second_response", ["accepted", "empty", "exception"])
def test_later_section_cannot_leave_prior_page_finding_accepted(tmp_path, second_response):
    materials = make_materials(tmp_path, [("a", "These results is fixed."), ("b", "Those methods is clear.")])
    records = []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing":
            block = payload["blocks"][0]
            if block["id"] == "b2":
                page = Path(materials.pages[0].path)
                page.write_bytes(page.read_bytes() + b"changed-after-prior-consumption")
                if second_response == "exception":
                    raise RuntimeError("Independent later section service failure")
                if second_response == "empty":
                    return {"findings": []}
            return {"findings": [candidate(block)]}
        return confirmations(payload)

    findings = check_writing(materials, call=call, records=records, recover_errors=True)
    assert not any(f.loc.page == 1 for f in findings)
    assert records[0].status == "failed" and records[0].confirmed_count == 0
    if second_response == "accepted":
        assert len(findings) == 1 and findings[0].loc.page == 2


def test_healthy_call_count_and_unconsumed_missing_page(tmp_path):
    materials = make_materials(tmp_path, [("a", "These results is fixed."), ("b", "The procedure is clear.")])
    Path(materials.pages[1].path).unlink()
    records, calls = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(kwargs["module"])
        if kwargs["module"] == "screening_writing":
            block = payload["blocks"][0]
            return {"findings": [candidate(block)] if block["id"] == "b1" else []}
        return confirmations(payload)

    findings = check_writing(materials, call=call, records=records, recover_errors=True)
    assert len(findings) == 1
    assert calls == ["screening_writing", "screening_writing.validation", "screening_writing"]
    assert all(row.status == "checked" for row in records)


def test_success_narrative_and_visual_audit_redact_both_provider_configs(tmp_path, offline):
    text_cfg, visual_cfg = offline
    materials = make_materials(tmp_path, [("a", "These results is fixed.")])
    original = materials.model_dump(mode="json")

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing":
            row = candidate(payload["blocks"][0])
            row["text"] += (
                f" {text_cfg.api_key} {text_cfg.base_url} {visual_cfg.api_key} {visual_cfg.base_url}"
            )
            return {"findings": [row]}
        for secret in (
            text_cfg.api_key,
            visual_cfg.api_key,
            "text-pass",
            "vision-pass",
            "text-token",
            "vision-token",
        ):
            assert secret not in kwargs["prompt"]
        assert payload["candidates"][0]["quote"] == original["blocks"][0]["text"]
        result = confirmations(payload)
        result["results"][0]["explanation"] += f" {visual_cfg.api_key} {visual_cfg.base_url}"
        return result

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        findings = check_writing(materials, call=call, recover_errors=True)
    assert len(findings) == 1
    assert materials.model_dump(mode="json") == original
    assert findings[0].evidence[0].pointer.quote == original["blocks"][0]["text"]
    exposed = json.dumps([f.model_dump(mode="json") for f in findings])
    exposed += "".join(
        p.read_text("utf-8")
        for folder in ("writing_calls", "visual_calls")
        for p in (tmp_path / folder).glob("*.json")
    )
    for secret in (
        text_cfg.api_key,
        visual_cfg.api_key,
        "text-pass",
        "vision-pass",
        "text-token",
        "vision-token",
    ):
        assert secret not in exposed


def test_redaction_cannot_turn_raw_quote_mismatch_into_grounding(tmp_path, offline):
    text_cfg, vision_cfg = offline
    materials = make_materials(tmp_path, [("a", f"These results is fixed. Marker {text_cfg.api_key}.")])
    records, calls = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(kwargs["module"])
        assert kwargs["module"] == "screening_writing"
        row = candidate(payload["blocks"][0])
        row["quote"] = row["quote"].replace(text_cfg.api_key, vision_cfg.api_key)
        return {"findings": [row]}

    findings = check_writing(materials, call=call, records=records, recover_errors=True)
    assert not findings and records[0].status == "failed"
    assert calls == ["screening_writing"]


@pytest.mark.parametrize(
    "change", ["pdf_bytes", "markdown_bytes", "block_object", "block_location", "reference_directory"]
)
def test_original_text_and_directory_cannot_change_during_first_call(tmp_path, change):
    materials = make_materials(tmp_path, [("a", "These results is fixed.")])
    records, calls = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(kwargs["module"])
        assert kwargs["module"] == "screening_writing"
        if change == "pdf_bytes":
            path = Path(materials.source_pdf)
            path.write_bytes(path.read_bytes() + b"changed PDF")
        elif change == "markdown_bytes":
            Path(materials.markdown_path).write_text("Different source", encoding="utf-8")
        elif change == "block_object":
            materials.blocks[0] = materials.blocks[0].model_copy(update={"text": "Different source"})
        elif change == "block_location":
            materials.blocks[0].loc.page = 9
        else:
            materials.issues.append("Changed directory limitation")
        return {"findings": [candidate(payload["blocks"][0])]}

    assert not check_writing(materials, call=call, records=records, recover_errors=True)
    assert records[0].status == "failed" and calls == ["screening_writing"]


@pytest.mark.parametrize("change", ["duplicate_page", "late_created_page"])
def test_ambiguous_or_initially_unavailable_page_is_not_new_baseline(tmp_path, change):
    materials = make_materials(tmp_path, [("a", "These results is fixed.")])
    page = Path(materials.pages[0].path)
    pixels = page.read_bytes()
    if change == "duplicate_page":
        materials.pages.append(materials.pages[0].model_copy(deep=True))
    else:
        page.unlink()
    calls, records = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(kwargs["module"])
        assert kwargs["module"] == "screening_writing"
        if change == "late_created_page":
            page.write_bytes(pixels)
        return {"findings": [candidate(payload["blocks"][0])]}

    assert not check_writing(materials, call=call, records=records, recover_errors=True)
    assert records[0].status == "failed" and calls == ["screening_writing"]


def test_target_page_is_part_of_consumed_original_context(tmp_path):
    from schemas.materials import FigureMaterial

    materials = make_materials(
        tmp_path, [("a", "Figure 1 demonstrates the procedure."), ("b", "The procedure is clear.")]
    )
    materials.figures.append(
        FigureMaterial(
            id="f", anchor="1", caption="Figure 1: The procedure.", loc=materials.blocks[1].loc.model_copy()
        )
    )
    records = []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        if kwargs["module"] == "screening_writing":
            block = payload["blocks"][0]
            rows = (
                [
                    candidate(
                        block,
                        category="cross_reference",
                        target_kind="figure",
                        target_label="1",
                        reference_problem="inconsistent_target",
                    )
                ]
                if block["id"] == "b1"
                else []
            )
            return {"findings": rows}
        assert payload["additional_target_pages"] == [2]
        path = Path(materials.pages[1].path)
        path.write_bytes(path.read_bytes() + b"changed target")
        return confirmations(payload, kind="cross_reference")

    assert not check_writing(materials, call=call, records=records, recover_errors=True)
    assert records[0].status == "failed" and records[1].status == "checked"


def test_credential_in_true_quote_is_not_published_as_a_rewritten_source(tmp_path, offline):
    text_cfg, _ = offline
    materials = make_materials(tmp_path, [("a", f"These results is fixed. Marker {text_cfg.api_key}.")])
    records, calls = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(kwargs["module"])
        assert kwargs["module"] == "screening_writing"
        return {"findings": [candidate(payload["blocks"][0])]}

    assert not check_writing(materials, call=call, records=records, recover_errors=True)
    assert records[0].status == "failed" and calls == ["screening_writing"]
    assert "provenance contains configured credentials" in records[0].issues[0]
    assert text_cfg.api_key not in json.dumps(records[0].model_dump())


def test_redaction_scope_restores_on_exception_without_cross_call_leakage(offline):
    from screening.visual_audit import redacted_record, redaction_scope

    text_cfg, vision_cfg = offline
    unrelated = LLMConfig("mock", "other", None, None)
    value = {"original": f"{text_cfg.api_key} {vision_cfg.api_key}"}
    with pytest.raises(RuntimeError):
        with redaction_scope((text_cfg, vision_cfg)):
            assert text_cfg.api_key not in redacted_record(value, unrelated)["original"]
            assert vision_cfg.api_key not in redacted_record(value, unrelated)["original"]
            raise RuntimeError("Exit this writing run")
    assert redacted_record(value, unrelated) == value


def test_exactly_bound_target_with_credentials_cannot_be_forwarded(tmp_path, offline):
    from schemas.materials import FigureMaterial

    text_cfg, _ = offline
    materials = make_materials(tmp_path, [("a", "Figure 1 demonstrates the procedure.")])
    materials.figures.append(
        FigureMaterial(
            id="figure",
            anchor="1",
            caption=f"Figure 1: Procedure {text_cfg.api_key}.",
            loc=materials.blocks[0].loc.model_copy(),
        )
    )
    records, calls = [], []

    def call(**kwargs):
        payload = json.loads(kwargs["prompt"])
        calls.append(kwargs["module"])
        assert kwargs["module"] == "screening_writing"
        return {
            "findings": [
                candidate(
                    payload["blocks"][0],
                    category="cross_reference",
                    target_kind="figure",
                    target_label="1",
                    reference_problem="inconsistent_target",
                )
            ]
        }

    assert not check_writing(materials, call=call, records=records, recover_errors=True)
    assert records[0].status == "failed" and calls == ["screening_writing"]
    assert "cannot be forwarded" in records[0].issues[0]
    assert text_cfg.api_key not in json.dumps(records[0].model_dump())


def test_unused_invalid_visual_configuration_does_not_add_a_model_stage(tmp_path, monkeypatch):
    materials = make_materials(tmp_path, [("a", "The procedure is clear.")])
    calls, records = [], []

    def invalid_visual(**kwargs):
        raise ValueError("Visual provider is unavailable")

    monkeypatch.setattr("screening.checks.resolve_vlm_config", invalid_visual)

    def call(**kwargs):
        calls.append(kwargs["module"])
        return {"findings": []}

    assert not check_writing(materials, call=call, records=records, recover_errors=True)
    assert records[0].status == "checked" and calls == ["screening_writing"]
