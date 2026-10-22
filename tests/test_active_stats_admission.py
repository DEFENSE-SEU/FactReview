"""Unknown active accounting history cannot enter a model request or be replaced."""
import copy
import json

import pytest

from common import run_stats
from llm import client


def test_bad_active_stats_are_preserved_before_any_provider_admission(tmp_path, monkeypatch):
    admissions = []

    def forbidden(*args, **kwargs):
        pytest.fail("Active accounting control forbids real services and processes")

    for name in ("subprocess.Popen", "subprocess.run", "requests.sessions.Session.request",
                 "httpx.Client.send", "httpx.AsyncClient.send"):
        monkeypatch.setattr(name, forbidden)

    def auth(**kwargs):
        admissions.append("auth")
        return object()

    def provider(**kwargs):
        admissions.append("provider")
        return '{"status":"ok"}', {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}

    monkeypatch.setattr(client, "get_codex_auth", auth)
    monkeypatch.setattr(client, "invoke_codex", provider)
    config = client.LLMConfig(provider="openai-codex", model="offline-model", base_url="https://offline.invalid",
                              api_key="offline")
    stats_file = tmp_path / "stats.json"
    valid = run_stats.initialize(stats_file, activate=False)
    cases = [b"unknown old history", b"{}"]
    for field, value in (("version", True), ("missing_module", None), ("incomplete", None),
                         ("requests", True), ("total_tokens", -1), ("failed_requests", -1)):
        payload = copy.deepcopy(valid)
        if field == "version":
            payload[field] = value
        elif field == "missing_module":
            del payload["modules"]["parse"]
        elif field == "incomplete":
            del payload["modules"]["analysis"]["warnings"]
        elif field in ("requests", "total_tokens"):
            payload["modules"]["execution"]["token_usage"][field] = value
        else:
            payload["modules"]["execution"][field] = value
        cases.append(json.dumps(payload).encode())

    with run_stats.run_scope(stats_file):
        for raw in cases:
            stats_file.write_bytes(raw)
            admissions.clear()
            error = None
            try:
                client.llm_json("Refine command.", "Offline.", config, module="execution")
            except (ValueError, OSError) as exc:
                error = exc
            assert error is not None
            assert admissions == []
            assert stats_file.read_bytes() == raw

        stats_file.unlink()
        with pytest.raises(FileNotFoundError):
            client.llm_json("Refine command.", "Offline.", config, module="execution")
        assert admissions == []
        assert not stats_file.exists()

        run_stats.initialize(stats_file, activate=False)
        reply = client.llm_json("Refine command.", "Offline.", config, module="execution")
        assert reply == {"status": "ok"}
        assert admissions == ["auth", "provider"]
        assert run_stats.read(stats_file)["modules"]["execution"]["token_usage"]["total_tokens"] == 5
