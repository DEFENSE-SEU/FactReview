import hashlib
import json
import traceback

import pytest
from PIL import Image

from common import run_stats
from llm.client import LLMConfig
from screening.checks import ask


@pytest.fixture
def visual_input(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (120, 90), "white").save(image)
    cfg = LLMConfig(
        "openai", "text-model", "https://user:secret@example.invalid:1234/v1?key=secret", "secret"
    )
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)
    for key in ("VLM_MODEL_PROVIDER", "VLM_MODEL", "VLM_BASE_URL", "VLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLM_MODEL", "vision-model")
    return image


def test_visual_audit_records_actual_model_pixels_context_and_response(tmp_path, visual_input):
    def model(**request):
        assert request["cfg"].model == "vision-model"
        assert request["images"] == [str(visual_input)]
        return {"findings": []}

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        ask(
            "Inspect pixels",
            {"caption": "Known caption"},
            module="screening_figures",
            images=[str(visual_input)],
            call=model,
        )
    paths = list((tmp_path / "visual_calls").glob("*.json"))
    assert len(paths) == 1
    audit = json.loads(paths[0].read_text(encoding="utf-8"))
    assert audit["status"] == "ok" and audit["transport"] == "injected"
    assert audit["model"] == "vision-model"
    assert audit["endpoint"] == "https://example.invalid:1234/v1"
    assert "secret" not in paths[0].read_text(encoding="utf-8")
    assert audit["images"][0]["sha256"] == hashlib.sha256(visual_input.read_bytes()).hexdigest()
    assert audit["images"][0]["width"] == 120 and audit["images"][0]["height"] == 90
    assert audit["payload"]["caption"] == "Known caption" and audit["response"] == {"findings": []}


@pytest.mark.parametrize("failure", ["error_response", "exception"])
def test_failed_visual_calls_keep_inputs_and_explicit_failure(tmp_path, visual_input, failure):
    def model(**request):
        if failure == "exception":
            raise RuntimeError("transport unavailable")
        return {"status": "error", "error": "vision unsupported"}

    with run_stats.run_scope(tmp_path / "run_stats.json"), pytest.raises(RuntimeError):
        ask("Inspect pixels", {}, module="screening_figures", images=[str(visual_input)], call=model)
    audit = json.loads(next((tmp_path / "visual_calls").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["status"] == "failed" and audit["error"]
    assert len(audit["images"]) == 1 and audit["duration_seconds"] >= 0


def test_text_call_keeps_main_model_and_does_not_create_visual_audit(tmp_path, visual_input):
    def model(**request):
        assert request["cfg"].model == "text-model"
        assert "images" not in request
        return {"findings": []}

    with run_stats.run_scope(tmp_path / "run_stats.json"):
        ask("Read text", {}, module="screening_writing", call=model)
    assert not (tmp_path / "visual_calls").exists()


@pytest.mark.parametrize("failure", ["error_response", "exception"])
def test_failed_visual_diagnostics_redact_credentials_and_keep_endpoint_port(
    tmp_path, visual_input, monkeypatch, failure
):
    cfg = LLMConfig(
        "openai",
        "vision-model",
        "https://fixture-user:fixture-password@example.invalid:1234/v1?token=fixture-query-token",
        "fixture-api-key",
    )
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: cfg)
    source = {
        "status": "error",
        "error": f"Provider rejected {cfg.base_url}; Authorization: Bearer {cfg.api_key}; token=fixture-query-token",
        "base_url": cfg.base_url,
        "details": [{"api_key": cfg.api_key, "message": "normal diagnostic", "retry": False}],
    }

    def model(**request):
        if failure == "exception":
            raise ValueError(source["error"])
        return source

    with run_stats.run_scope(tmp_path / "run_stats.json"), pytest.raises(RuntimeError) as raised:
        ask("Inspect pixels", {}, module="screening_figures", images=[str(visual_input)], call=model)
    path = next((tmp_path / "visual_calls").glob("*.json"))
    serialized = path.read_text(encoding="utf-8")
    exception = "".join(traceback.format_exception(raised.value))
    for credential in ("fixture-user", "fixture-password", "fixture-query-token", "fixture-api-key"):
        assert credential not in serialized
        assert credential not in exception
    audit = json.loads(serialized)
    assert audit["status"] == "failed"
    assert audit["endpoint"] == "https://example.invalid:1234/v1"
    assert "https://example.invalid:1234/v1" in exception
    if failure == "error_response":
        assert audit["response"]["base_url"] == audit["endpoint"]
        assert audit["response"]["details"][0]["message"] == "normal diagnostic"
        assert audit["response"]["details"][0]["retry"] is False
    assert source["base_url"] == cfg.base_url
    assert source["details"][0]["api_key"] == cfg.api_key


def test_successful_visual_model_data_is_returned_unchanged(tmp_path, visual_input):
    result = {
        "findings": [{"category": "legibility", "disposition": "clear", "text": "Labels are readable."}]
    }
    with run_stats.run_scope(tmp_path / "run_stats.json"):
        returned = ask(
            "Inspect pixels",
            {"caption": "Known caption"},
            module="screening_figures",
            images=[str(visual_input)],
            call=lambda **request: result,
        )
    assert returned is result
    audit = json.loads(next((tmp_path / "visual_calls").glob("*.json")).read_text(encoding="utf-8"))
    assert audit["response"] == result
    assert audit["payload"] == {"caption": "Known caption"}
