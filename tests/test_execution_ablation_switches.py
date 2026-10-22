"""Alignment and repairer ablation switches; the LLM transport is mocked."""

from unittest.mock import Mock

import pytest

from fact_generation.execution.runner import Observation, RunOutcome
from tests.test_execution import inputs, run  # noqa: F401


def observation(**changes):
    values = {"dataset": "d1", "metric": "accuracy", "settings": {"split": "test", "seed": 1}, "value": 0.9}
    return Observation(**{**values, **changes})


def _ok(*observations):
    return lambda _: RunOutcome(returncode=0, observations=list(observations))


def test_strict_alignment_is_default_and_rejects_other_settings(inputs, tmp_path):  # noqa: F811
    result = run(inputs, tmp_path, _ok(observation(settings={"split": "dev"})))
    assert not result.claims[0].evidence
    assert result.ledger[0]["alignment"][0]["alignment_mode"] == "strict"


def test_relaxed_alignment_compares_on_dataset_only(inputs, tmp_path):  # noqa: F811
    obs = observation(settings={"split": "dev"}, metric="f1", value=0.5)
    result = run(inputs, tmp_path, _ok(obs), config={"alignment": "relaxed"})
    evidence = result.claims[0].evidence[0]
    assert evidence.aligned and evidence.direction == "flaw" and "alignment=relaxed" in evidence.note
    assert result.ledger[0]["alignment"][-1]["alignment_mode"] == "relaxed"


def test_relaxed_alignment_still_requires_the_claimed_dataset(inputs, tmp_path):  # noqa: F811
    result = run(inputs, tmp_path, _ok(observation(dataset="d2")), config={"alignment": "relaxed"})
    assert not result.claims[0].evidence


def test_relaxed_alignment_uses_metric_only_to_break_ties(inputs, tmp_path):  # noqa: F811
    observations = [observation(metric="f1", value=0.1), observation(settings={"split": "dev"})]
    result = run(inputs, tmp_path, _ok(*observations), config={"alignment": "relaxed"})
    evidence = result.claims[0].evidence[0]
    assert evidence.sufficient and evidence.direction == "support"


@pytest.fixture
def llm(monkeypatch):
    mock = Mock()
    monkeypatch.setattr("llm.client.llm_json", mock)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: "cfg")
    return mock


def _failing_then_ok(commands):
    def runner(request):
        commands.append(list(request.command))
        if request.repair_round == 0:
            return RunOutcome(returncode=1, stderr="RuntimeError: CUDA unavailable")
        return RunOutcome(returncode=0, observations=[observation()])

    return runner


def test_llm_repairer_proposal_passes_whitelist(inputs, tmp_path, llm):  # noqa: F811
    llm.return_value = {"repair": {"launch_arguments": {"--device": "cpu"}, "reason": "no GPU"}}
    commands = []
    result = run(inputs, tmp_path, _failing_then_ok(commands), config={"repairer": "llm"})
    assert commands[1] == ["python", "eval.py", "--device", "cpu"]
    assert result.ledger[0]["repairs"][0]["accepted"]
    assert result.claims[0].evidence[0].direction == "support"
    assert "CUDA unavailable" in llm.call_args.args[0]


def test_llm_repairer_proposal_outside_whitelist_is_rejected(inputs, tmp_path, llm):  # noqa: F811
    llm.return_value = {"repair": {"launch_arguments": {"--model": "small"}, "reason": "smaller model"}}
    commands = []
    result = run(inputs, tmp_path, _failing_then_ok(commands), config={"repairer": "llm"})
    assert len(commands) == 1
    record = result.ledger[0]["repairs"][0]
    assert not record["accepted"] and "protected" in record["reason"]


def test_llm_repairer_null_stops_and_dependency_fix_skips_llm(inputs, tmp_path, llm):  # noqa: F811
    llm.return_value = {"repair": None}
    result = run(inputs, tmp_path, lambda _: RunOutcome(returncode=1), config={"repairer": "llm"})
    assert len(result.ledger[0]["attempts"]) == 1
    assert result.ledger[0]["repairs"][0]["reason"] == "no permitted infrastructure repair available"

    llm.reset_mock()
    stderr = "ModuleNotFoundError: No module named 'yaml'"
    run(
        inputs,
        tmp_path / "dep",
        lambda _: RunOutcome(returncode=1, stderr=stderr),
        config={"repairer": "llm", "max_attempts": 1},
    )
    llm.assert_not_called()


def test_default_repairer_never_calls_llm(inputs, tmp_path, llm):  # noqa: F811
    run(inputs, tmp_path, lambda _: RunOutcome(returncode=1, stderr="RuntimeError"))
    llm.assert_not_called()
