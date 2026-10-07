"""Pixel payloads must reach the mocked provider transport for figure checks."""

import base64
import io
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from llm.client import LLMConfig, llm_json
from llm.codex_auth import CodexAuth


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "printed.png"
    Image.new("RGB", (96, 48), "white").save(path)
    return path


def test_codex_receives_encoded_figure_pixels(monkeypatch, image_file):
    captured = {}

    def urlopen(request, **kwargs):
        captured.update(json.loads(request.data))
        event = {"type": "response.output_text.delta", "delta": '{"findings": []}'}
        return io.BytesIO(("data: " + json.dumps(event) + "\n\n").encode())

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

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text='{"findings": []}')],
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"findings": []}'))],
            usage=None,
        )

    def fake(**kwargs):
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
