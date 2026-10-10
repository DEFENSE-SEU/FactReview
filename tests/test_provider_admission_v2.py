"""SDK requests stay visible and invalid usage scopes fail before provider admission."""

import importlib

import openai
import pytest

from common import run_stats
from llm import client


def test_retryable_sdk_failure_has_one_send_and_invalid_module_has_none(tmp_path, monkeypatch):
    sdk_base = importlib.import_module("openai._base_client")
    sdk_http = getattr(sdk_base, "httpx", None) or getattr(sdk_base, "httpx2")
    sends, clients = [], []

    def forbidden(*args, **kwargs):
        pytest.fail("Provider admission control forbids real network, processes and Codex authentication")

    for name in ("subprocess.Popen", "subprocess.run", "requests.sessions.Session.request",
                 "httpx.HTTPTransport.handle_request", "llm.client.get_codex_auth", "llm.client.invoke_codex"):
        monkeypatch.setattr(name, forbidden)
    monkeypatch.setattr(sdk_http.HTTPTransport, "handle_request", forbidden)

    def reply(request):
        sends.append(request.url.path)
        return sdk_http.Response(429, json={"error": {"message": "offline rate limit", "type": "rate_limit_error"}})

    real_openai = openai.OpenAI

    def local_sdk(**kwargs):
        transport = sdk_http.Client(transport=sdk_http.MockTransport(reply))
        instance = real_openai(**kwargs, http_client=transport)
        clients.append(instance)
        return instance

    monkeypatch.setattr(openai, "OpenAI", local_sdk)
    config = client.LLMConfig(provider="openai", model="offline-model", base_url="https://offline.invalid/v1",
                              api_key="offline-key")
    stats_file = tmp_path / "run_stats.json"
    try:
        with run_stats.run_scope(stats_file):
            response = client.llm_json("Return JSON.", "Offline control.", config, module="execution")
            retryable_sends = len(sends)
            with pytest.raises(ValueError, match="unknown stats module"):
                client.llm_json("Return JSON.", "Offline control.", config, module="analysis.invalid")
            invalid_scope_sends = len(sends) - retryable_sends
        stats = run_stats.read(stats_file)["modules"]["execution"]
        assert response["status"] == "error"
        assert retryable_sends == 1
        assert invalid_scope_sends == 0
        assert stats["token_usage"]["requests"] == retryable_sends
        assert stats["failed_requests"] == 1
        assert stats["unavailable_usage_requests"] == 1
        assert stats["token_usage"]["estimated_requests"] == 0
    finally:
        for instance in clients:
            instance.close()
