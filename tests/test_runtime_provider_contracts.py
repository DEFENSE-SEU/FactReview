"""Runtime failures and CLI routes stay truthful without contacting providers."""

import io
import json
import os
import traceback
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from common import run_stats
from llm.client import LLMConfig, llm_json, resolve_llm_config, resolve_vlm_config
from llm.codex_auth import CodexAuth
from llm.codex_client import CodexResponseError, invoke_codex
from pipeline_full import _apply_cli_env_overrides
from schemas.claim import ClaimLocation
from schemas.materials import MaterialBlock, SharedMaterials
from screening.claims import ClaimExtractionError, extract_claims


@pytest.mark.parametrize("provider", ["openai-codex", "chatgpt-oauth", "openai", "claude", "qwen", "deepseek"])
def test_cli_model_override_reaches_selected_provider_and_inherited_vision(provider):
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-key"}, clear=True):
        _apply_cli_env_overrides(SimpleNamespace(llm_provider=provider, llm_model="requested-model"))
        cfg = resolve_llm_config()
        assert cfg.model == "requested-model"
        assert resolve_vlm_config(fallback=cfg).model == "requested-model"


def test_cli_model_uses_effective_execution_provider_and_preserves_visual_override():
    env = {"MODEL_PROVIDER": "openai-codex", "EXECUTION_MODEL_PROVIDER": "claude",
           "CLAUDE_API_KEY": "fixture-key", "VLM_MODEL": "visual-model"}
    with patch.dict(os.environ, env, clear=True):
        _apply_cli_env_overrides(SimpleNamespace(llm_provider="", llm_model="requested-model"))
        cfg = resolve_llm_config()
        assert cfg.provider == "claude" and cfg.model == "requested-model"
        assert resolve_vlm_config(fallback=cfg).model == "visual-model"


def test_claim_extraction_failure_never_exposes_provider_credentials(monkeypatch):
    cfg = LLMConfig("openai", "fixture", "https://fixture-user:fixture-password@example.invalid/v1?key=fixture-query", "fixture-key")
    text = "A located claim."
    materials = SharedMaterials(
        paper_key="fixture", source_pdf="paper.pdf", markdown=text, markdown_path="paper.md",
        content_list_path="content.json", provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=ClaimLocation(page=1))],
    )

    def fail(**kwargs):
        raise RuntimeError(f"Rejected key={cfg.api_key} at {cfg.base_url}")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    monkeypatch.setattr("screening.claims.resolve_llm_config", lambda: cfg)
    with pytest.raises(ClaimExtractionError) as raised:
        extract_claims(materials)
    formatted = "".join(traceback.format_exception(raised.value))
    for secret in ("fixture-key", "fixture-user", "fixture-password", "fixture-query"):
        assert secret not in formatted


@pytest.mark.parametrize("http_error", [False, True])
def test_codex_transport_failure_redacts_oauth_before_truncation(monkeypatch, http_error):
    token = "fixture-oauth-secret-" + "z" * 2400
    message = "Rejected Bearer " + token
    error = (
        urllib.error.HTTPError("https://example.invalid/responses", 401, "fixture", {}, io.BytesIO(message.encode()))
        if http_error else TimeoutError(message)
    )
    monkeypatch.setattr("llm.codex_client.urllib.request.urlopen", lambda *a, **kw: (_ for _ in ()).throw(error))
    with pytest.raises(CodexResponseError) as raised:
        invoke_codex("prompt", "system", auth=CodexAuth(token), model="fixture", base_url="https://example.invalid")
    formatted = "".join(traceback.format_exception(raised.value))
    assert "fixture-oauth-secret" not in formatted
    assert "z" * 100 not in formatted
    assert "[redacted]" in formatted
    assert raised.value.usage == {}


@pytest.mark.parametrize(
    "usage,total,unavailable",
    [({"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}, 120, 0),
     ({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, 0, 0),
     ({}, 0, 1), ({"input_tokens": "invalid"}, 0, 1)],
)
def test_failed_codex_response_retains_measured_usage(monkeypatch, tmp_path, usage, total, unavailable):
    events = [
        {"type": "response.output_text.delta", "delta": '{"findings": []}'},
        {"type": "response.failed", "response": {"status": "failed", "usage": usage}},
    ]
    stream = "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()
    monkeypatch.setattr("llm.client.get_codex_auth", lambda **kwargs: CodexAuth("fixture-oauth"))
    monkeypatch.setattr("llm.codex_client.urllib.request.urlopen", lambda *a, **kw: io.BytesIO(stream))
    with run_stats.run_scope(tmp_path / "stats.json"):
        result = llm_json("prompt", "system", LLMConfig("openai-codex", "fixture", "https://example.invalid", None), module="analysis")
        stats = run_stats.read()["modules"]["analysis"]
    assert result["status"] == "error" and "findings" not in result
    assert stats["failed_requests"] == 1 and stats["token_usage"]["requests"] == 1
    assert stats["token_usage"]["total_tokens"] == total
    assert stats["unavailable_usage_requests"] == unavailable
    assert stats["estimated"] is False
