"""Standalone execution owns one ledger before its first mocked provider call."""
import json
from pathlib import Path

import pytest

from common import run_stats
from fact_generation.execution import v2
from llm import client
from tests.test_execution_v2 import inputs  # noqa: F401 -- pytest fixture registration


def test_standalone_first_usage_inherited_history_and_exception_scope(inputs, tmp_path, monkeypatch):  # noqa: F811
    def forbidden(*args, **kwargs):
        pytest.fail("Standalone accounting control forbids real external calls")

    for name in ("subprocess.Popen", "subprocess.run", "requests.sessions.Session.request",
                 "httpx.Client.send", "httpx.AsyncClient.send",
                 "fact_generation.execution.v2.docker_runner", "fact_generation.execution.v2.run_command",
                 "fact_generation.execution.v2.docker_ensure_paper_image"):
        monkeypatch.setattr(name, forbidden)
    monkeypatch.delenv("FACTREVIEW_RUN_STATS_PATH", raising=False)
    assert run_stats.stats_path() is None
    admissions, usages = [], [{"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}]

    def auth(**kwargs):
        admissions.append("auth")
        return object()

    def provider(**kwargs):
        admissions.append("provider")
        return json.dumps({"command": ["python", "eval.py"], "metric_output": None}), usages[0]

    monkeypatch.setattr(client, "get_codex_auth", auth)
    monkeypatch.setattr(client, "invoke_codex", provider)
    monkeypatch.setattr(client, "resolve_llm_config", lambda: client.LLMConfig(
        provider="openai-codex", model="offline", base_url="https://offline.invalid", api_key=None))
    plan, claim, materials = inputs
    plan.task.command = []  # Real public command-refinement path, with original targets retained.
    initial_plan, initial_claim = plan.model_dump(mode="json"), claim.model_dump(mode="json")
    raw = "unchanged offline author output\n"

    def runner(request):
        assert request.command == ["python", "eval.py"]
        assert run_stats.read_initialized()["modules"]["execution"]["token_usage"]["total_tokens"] >= 5
        return v2.RunOutcome(returncode=0, stdout=raw)

    def invoke(name, run=runner):
        return v2.execute_plans([plan], [claim], materials, tmp_path / name,
                               config={"max_attempts": 0}, runner=run)

    result = invoke("standalone")
    path = tmp_path / "standalone" / "execution_stats.json"
    stats = run_stats.read_initialized(path)["modules"]["execution"]
    assert admissions == ["auth", "provider"]
    assert stats["token_usage"]["requests"] == 1 and stats["token_usage"]["total_tokens"] == 5
    assert stats["unavailable_usage_requests"] == stats["token_usage"]["estimated_requests"] == 0
    assert result.ledger[0]["refinement"]["tokens"] == 5
    assert Path(result.ledger[0]["refinement"]["token_source"]) == path
    assert result.ledger[0]["attempts"][0]["stdout"] == raw
    assert result.ledger[0]["attempts"][0]["returncode"] == 0
    assert not result.claims[0].evidence and result.claims[0].status == claim.status
    assert not list((tmp_path / "standalone").rglob("scientific_usage.json"))
    assert run_stats.stats_path() is None

    inherited = tmp_path / "inherited.json"
    with run_stats.run_scope(inherited):
        run_stats.record_llm_call(module="execution", provider="mock", model="prior",
                                 usage={"input_tokens": 7, "output_tokens": 4, "total_tokens": 11})
        inherited_result = invoke("inherited")
        current = run_stats.read_initialized(inherited)["modules"]["execution"]
        assert run_stats.stats_path() == inherited.resolve()
        assert current["token_usage"]["requests"] == 2 and current["token_usage"]["total_tokens"] == 16
        assert current["models"]["prior"] == 1
        assert inherited_result.ledger[0]["refinement"]["tokens"] == 5
        assert Path(inherited_result.ledger[0]["refinement"]["token_source"]) == inherited
        assert not (tmp_path / "inherited" / "execution_stats.json").exists()
    assert run_stats.stats_path() is None

    historical = tmp_path / "historical"
    historical.mkdir()
    history = historical / "execution_stats.json"
    history.write_bytes(b"unknown immutable accounting history")
    calls = list(admissions)
    with pytest.raises(FileExistsError):
        invoke("historical")
    assert history.read_bytes() == b"unknown immutable accounting history" and admissions == calls
    assert not (historical / "run_0000").exists() and run_stats.stats_path() is None

    class OfflineAbort(BaseException):
        pass

    def abort(request):
        raise OfflineAbort("Offline author process interruption")

    with pytest.raises(OfflineAbort):
        invoke("aborted", abort)
    assert run_stats.stats_path() is None
    assert run_stats.read_initialized(tmp_path / "aborted" / "execution_stats.json")["modules"]["execution"]["token_usage"]["total_tokens"] == 5

    usages[0] = {}
    unknown = invoke("unknown", lambda request: v2.RunOutcome(returncode=0, stdout=raw))
    unknown_stats = run_stats.read_initialized(tmp_path / "unknown" / "execution_stats.json")["modules"]["execution"]
    assert unknown_stats["token_usage"]["requests"] == 1
    assert unknown_stats["unavailable_usage_requests"] == 1
    # Existing client records its text estimate alongside explicit unavailable usage.
    assert unknown_stats["token_usage"]["estimated_requests"] == 1
    assert unknown_stats["estimated"] is True
    assert not unknown.claims[0].evidence and run_stats.stats_path() is None
    assert plan.model_dump(mode="json") == initial_plan and claim.model_dump(mode="json") == initial_claim
