"""Pixel payloads must reach the mocked provider transport for figure checks."""

import base64
import io
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from llm.client import LLMConfig, llm_json, resolve_llm_config, resolve_vlm_config
from llm.codex_auth import CodexAuth
from llm.codex_client import invoke_codex


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "printed.png"
    Image.new("RGB", (96, 48), "white").save(path)
    return path


def test_codex_receives_encoded_figure_pixels(monkeypatch, image_file):
    captured = {}

    def urlopen(request, **kwargs):
        captured.update(json.loads(request.data))
        events = [
            {"type": "response.output_text.delta", "delta": '{"findings": []}'},
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
        return io.BytesIO("".join("data: " + json.dumps(event) + "\n\n" for event in events).encode())

    monkeypatch.setattr("llm.client.get_codex_auth", lambda **kwargs: CodexAuth("test"))
    monkeypatch.setattr("llm.codex_client.urllib.request.urlopen", urlopen)
    cfg = LLMConfig("openai-codex", "fixture-model", "https://example.invalid", None)
    result = llm_json("Check the figure", "Use its pixels", cfg, images=[str(image_file)])
    assert result == {"findings": []}
    image = captured["input"][-1]["content"][1]
    assert image["type"] == "input_image"
    assert base64.b64decode(image["image_url"].split(",", 1)[1]) == image_file.read_bytes()


@pytest.mark.parametrize("provider", ["openai", "claude"])
def test_other_providers_receive_encoded_pixels(monkeypatch, image_file, provider):
    captured = {}
    constructor = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text='{"findings": []}')],
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"findings": []}'))],
            usage=None,
        )

    def fake(**kwargs):
        constructor.update(kwargs)
        return SimpleNamespace(
            messages=SimpleNamespace(create=create),
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )

    if provider == "openai":
        monkeypatch.setattr("openai.OpenAI", fake)
    else:
        monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=fake))
    cfg = LLMConfig(provider, "fixture-model", "https://example.invalid", "test")
    assert llm_json("Check", "Pixels", cfg, images=[str(image_file)]) == {"findings": []}
    image = captured["messages"][0 if provider == "claude" else 1]["content"][1]
    data = image["source"]["data"] if provider == "claude" else image["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(data) == image_file.read_bytes()
    assert constructor["base_url"] == "https://example.invalid"


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete", "response.cancelled", "error", None])
def test_codex_partial_json_cannot_hide_stream_failure(monkeypatch, image_file, terminal):
    events = [{"type": "response.output_text.delta", "delta": '{"findings": []}'}]
    if terminal:
        events.append({"type": terminal})
    stream = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    monkeypatch.setattr("llm.client.get_codex_auth", lambda **kwargs: CodexAuth("test"))
    monkeypatch.setattr("llm.codex_client.urllib.request.urlopen", lambda *a, **kw: io.BytesIO(stream.encode()))
    cfg = LLMConfig("openai-codex", "fixture-model", "https://example.invalid", None)
    result = llm_json("Check the figure", "Use its pixels", cfg, images=[str(image_file)])
    assert result["status"] == "error"
    assert "findings" not in result
    assert "Codex backend" in result["error"]


def test_codex_completed_response_supplies_final_text(monkeypatch, image_file):
    event = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [{"content": [{"type": "output_text", "text": '{"findings": []}'}]}],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
    }
    monkeypatch.setattr("llm.client.get_codex_auth", lambda **kwargs: CodexAuth("test"))
    monkeypatch.setattr("llm.codex_client.urllib.request.urlopen", lambda *a, **kw: io.BytesIO(("data: " + json.dumps(event) + "\n\n").encode()))
    cfg = LLMConfig("openai-codex", "fixture-model", "https://example.invalid", None)
    assert llm_json("Check", "Pixels", cfg, images=[str(image_file)]) == {"findings": []}


def test_visual_config_inherits_main_config_without_overrides():
    fallback = LLMConfig("openai", "shared-model", "https://shared.invalid", "shared-key", temperature=0.2)
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_vlm_config(fallback=fallback) is fallback


def test_visual_model_override_preserves_shared_provider_connection():
    fallback = LLMConfig("openai", "text-model", "https://shared.invalid", "shared-key")
    with patch.dict(os.environ, {"VLM_MODEL": "visual-model"}, clear=True):
        cfg = resolve_vlm_config(fallback=fallback)
    assert cfg == LLMConfig("openai", "visual-model", "https://shared.invalid", "shared-key")


def test_explicit_shared_visual_provider_retains_its_connection():
    fallback = LLMConfig("openai", "text-model", "https://shared.invalid", "shared-key")
    env = {"VLM_MODEL_PROVIDER": "openai", "VLM_MODEL": "visual-model"}
    with patch.dict(os.environ, env, clear=True):
        cfg = resolve_vlm_config(fallback=fallback)
    assert cfg == LLMConfig("openai", "visual-model", "https://shared.invalid", "shared-key")


def test_visual_provider_change_uses_separate_endpoint_and_credentials():
    fallback = LLMConfig("openai", "text-model", "https://text.invalid", "text-key")
    env = {"VLM_MODEL_PROVIDER": "claude", "VLM_MODEL": "visual-model", "VLM_BASE_URL": "https://visual.invalid", "VLM_API_KEY": "visual-key"}
    with patch.dict(os.environ, env, clear=True):
        cfg = resolve_vlm_config(fallback=fallback)
    assert cfg == LLMConfig("claude", "visual-model", "https://visual.invalid", "visual-key")


@pytest.mark.parametrize("provider", ["claude", "openai", "qwen", "deepseek"])
def test_visual_provider_without_its_credentials_fails(provider):
    fallback = LLMConfig("openai-codex", "text-model", "https://text.invalid", None)
    with patch.dict(os.environ, {"VLM_MODEL_PROVIDER": provider}, clear=True):
        with pytest.raises(ValueError, match="requires VLM_API_KEY"):
            resolve_vlm_config(fallback=fallback)


def test_visual_openai_override_does_not_fall_back_to_codex():
    env = {"MODEL_PROVIDER": "openai-codex", "VLM_MODEL_PROVIDER": "openai", "VLM_MODEL": "vision-model", "VLM_API_KEY": "visual-key"}
    with patch.dict(os.environ, env, clear=True):
        cfg = resolve_vlm_config()
    assert cfg.provider == "openai"
    assert cfg.model == "vision-model"
    assert cfg.api_key == "visual-key"
    assert cfg.base_url == "https://api.openai.com/v1"


def test_unknown_visual_provider_is_rejected():
    with patch.dict(os.environ, {"VLM_MODEL_PROVIDER": "typo-provider"}, clear=True):
        with pytest.raises(ValueError, match="unsupported visual model provider"):
            resolve_vlm_config()


@pytest.mark.parametrize("provider", ["codex", "chatgpt", "chatgpt-oauth", "openai_codex"])
def test_visual_codex_aliases_keep_subscription_authentication(provider):
    with patch.dict(os.environ, {"VLM_MODEL_PROVIDER": provider, "VLM_MODEL": "visual-model"}, clear=True):
        cfg = resolve_vlm_config()
    assert cfg.provider == "openai-codex"
    assert cfg.model == "visual-model"
    assert cfg.api_key is None


@pytest.mark.parametrize("provide_fallback", [False, True])
def test_switch_from_codex_to_openai_vision_uses_openai_connection(
    monkeypatch, image_file, provide_fallback
):
    env = {
        "MODEL_PROVIDER": "openai-codex",
        "EXECUTION_MODEL_PROVIDER": "openai-codex",
        "EXECUTION_OPENAI_BASE_URL": "https://chatgpt.com/backend-api/codex",
        "EXECUTION_OPENAI_MODEL": "execution-codex-model",
        "EXECUTION_OPENAI_API_KEY": "execution-key",
        "OPENAI_BASE_URL": "https://openai-compatible.invalid/v1",
        "OPENAI_MODEL": "openai-vision-model",
        "OPENAI_API_KEY": "openai-key",
        "VLM_MODEL_PROVIDER": "openai",
    }
    constructor, request = {}, {}

    def create(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"findings": []}'))], usage=None
        )

    def client(**kwargs):
        constructor.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("openai.OpenAI", client)
    with patch.dict(os.environ, env, clear=True):
        fallback = resolve_llm_config() if provide_fallback else None
        cfg = resolve_vlm_config(fallback=fallback)
        assert llm_json("Inspect pixels", "Check", cfg, images=[str(image_file)]) == {"findings": []}
    assert constructor == {"api_key": "openai-key", "base_url": "https://openai-compatible.invalid/v1"}
    assert request["model"] == "openai-vision-model"
    assert request["messages"][-1]["content"][1]["type"] == "image_url"


def test_cross_provider_visual_openai_cannot_borrow_execution_key():
    env = {
        "MODEL_PROVIDER": "openai-codex",
        "EXECUTION_OPENAI_API_KEY": "execution-key",
        "VLM_MODEL_PROVIDER": "openai",
    }
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError, match="requires VLM_API_KEY"):
            resolve_vlm_config()


def test_same_openai_visual_provider_preserves_execution_config():
    env = {
        "MODEL_PROVIDER": "openai",
        "EXECUTION_OPENAI_BASE_URL": "https://execution.invalid/v1",
        "EXECUTION_OPENAI_MODEL": "execution-model",
        "EXECUTION_OPENAI_API_KEY": "execution-key",
        "OPENAI_BASE_URL": "https://other.invalid/v1",
        "OPENAI_MODEL": "other-model",
        "OPENAI_API_KEY": "other-key",
        "VLM_MODEL_PROVIDER": "openai",
    }
    with patch.dict(os.environ, env, clear=True):
        assert resolve_vlm_config() == resolve_llm_config()


def test_codex_visual_switch_keeps_api_key_out_of_configuration():
    env = {"MODEL_PROVIDER": "openai", "OPENAI_API_KEY": "openai-key", "VLM_MODEL_PROVIDER": "codex"}
    with patch.dict(os.environ, env, clear=True):
        cfg = resolve_vlm_config()
    assert cfg.provider == "openai-codex" and cfg.api_key is None


@pytest.mark.parametrize(
    "usage,expected",
    [
        ({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
         {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
        ({}, {}),
        ({"input_tokens": "invalid", "output_tokens": 0, "total_tokens": 0}, {}),
        ({"input_tokens": False, "output_tokens": 0, "total_tokens": 0}, {}),
        ({"input_tokens": 0.5, "output_tokens": 0, "total_tokens": 0}, {}),
        ({"input_tokens": -1, "output_tokens": 0, "total_tokens": 0}, {}),
    ],
)
def test_codex_completion_usage_distinguishes_exact_zero_from_unavailable(monkeypatch, usage, expected):
    event = {
        "type": "response.completed",
        "response": {"status": "completed", "output_text": '{"findings": []}', "usage": usage},
    }
    monkeypatch.setattr(
        "llm.codex_client.urllib.request.urlopen",
        lambda *a, **kw: io.BytesIO(("data: " + json.dumps(event) + "\n\n").encode()),
    )
    response, measured = invoke_codex(
        "Check", "Pixels", auth=CodexAuth("fixture"), model="fixture-model",
        base_url="https://example.invalid", return_usage=True,
    )
    assert response == '{"findings": []}'
    assert measured == expected
