"""Process-origin controls; no process, provider or Docker is started."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from fact_generation.execution import v2
from tests import test_execution_v2 as execution_fixtures
from tests.test_execution_v2 import observation
from util import subprocess_runner as process

execution_inputs = execution_fixtures.inputs


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External operation was not mocked")

    monkeypatch.setattr(process.subprocess, "Popen", forbidden)
    monkeypatch.setattr(process, "_create_windows_kill_on_close_job", lambda proc: None)
    monkeypatch.setattr(process, "_close_windows_job", lambda job: None)
    monkeypatch.setattr(process, "_kill_process_tree", forbidden)
    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr(v2, "docker_ensure_paper_image", forbidden)
    monkeypatch.setattr(v2, "run_command", forbidden)


def test_actual_helper_preserves_origin_and_original_exit_codes(tmp_path, monkeypatch):
    for mode, expected_kind, expected_code in [
        ("launch", "launch_failed", 127),
        ("communication", "communication_failed", 127),
        ("timeout", "timed_out", 124),
        ("unfinished", "unfinished", 124),
        ("author127", "completed", 127),
        ("author124", "completed", 124),
    ]:
        proc = Mock(returncode=expected_code if mode != "unfinished" else None)
        proc.communicate.return_value = ("author stdout", "author stderr")
        proc.communicate.side_effect = (
            OSError("fixture communication failure") if mode == "communication"
            else [subprocess.TimeoutExpired(["fixture"], 1), ("partial stdout", "partial stderr")]
            if mode == "timeout" else None
        )
        launch = Mock(
            side_effect=FileNotFoundError("fixture missing executable") if mode == "launch" else None,
            return_value=proc,
        )
        killed = Mock()
        monkeypatch.setattr(process.subprocess, "Popen", launch)
        monkeypatch.setattr(process, "_kill_process_tree", killed)
        result = process.run_command(["fixture"], str(tmp_path), timeout_sec=1)
        assert (result.termination, result.returncode) == (expected_kind, expected_code)
        assert result.exception_type == {
            "launch": "FileNotFoundError", "communication": "OSError", "timeout": "TimeoutExpired",
        }.get(mode)
        assert killed.call_count == int(mode in {"communication", "timeout", "unfinished"})
        assert launch.call_count == 1
        process.persist_command_result(result, tmp_path / mode, "run")
        saved = (tmp_path / mode / "run_command.txt").read_text("utf-8")
        assert f"termination: {expected_kind}" in saved
        assert (tmp_path / mode / "run_stdout.log").read_text("utf-8") == result.stdout
        assert (tmp_path / mode / "run_stderr.log").read_text("utf-8") == result.stderr
    old = process.CommandResult(["fixture"], str(tmp_path), 127, "", "old unknown origin", .01)
    assert old.termination is None and old.exception_type is None


def test_docker_consumer_uses_declared_origin_without_inferring_from_codes(
    execution_inputs, tmp_path, monkeypatch
):
    plan, claim, materials = execution_inputs
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *args, **kwargs: (True, "fixture-image"))
    for kind, code, failed in [
        ("launch_failed", 127, True), ("communication_failed", 127, True),
        ("unfinished", 124, True), ("completed", 127, False),
        ("completed", 124, False), ("timed_out", 124, False), (None, 127, False),
    ]:
        def transport(command, cwd, timeout_sec):
            if command[1] == "run":
                return process.CommandResult(
                    command, cwd, code, "", "fixture same opaque stderr", .01,
                    termination=kind, exception_type="OSError" if failed else None,
                )
            return process.CommandResult(command, cwd, 0, "", "", .01, termination="completed")

        monkeypatch.setattr(v2, "run_command", transport)
        result = v2.execute_plans(
            [plan], [claim], materials, tmp_path / f"{kind}-{code}",
            config={"max_attempts": 0}, runner=v2.docker_runner,
        )
        row = result.ledger[0]
        assert row["attempts"][0]["environment"]["process_termination"]["kind"] == (kind or "unknown")
        assert bool(result.delivery_checks) == bool(row["operation_failures"]) == failed
        assert not result.claims[0].evidence
        if failed:
            assert result.delivery_checks[0].component == "execution.runner"
            assert result.delivery_checks[0].responsibility == "system"
            assert not result.claims[0].questions
        else:
            assert result.claims[0].questions and result.delivery_checks == []
    assert claim.evidence == [] and claim.questions == []


def test_origin_is_history_after_real_infrastructure_repair(execution_inputs, tmp_path, monkeypatch):
    plan, claim, materials = execution_inputs
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *args, **kwargs: (True, "fixture-image"))
    runs = []

    def transport(command, cwd, timeout_sec):
        if command[1] == "run":
            runs.append(command)
            if len(runs) == 1:
                return process.CommandResult(
                    command, cwd, 127, "", "fixture launch failure", .01,
                    termination="launch_failed", exception_type="FileNotFoundError",
                )
            return process.CommandResult(
                command, cwd, 0, json.dumps({"observations": [observation().model_dump()]}), "", .01,
                termination="completed",
            )
        return process.CommandResult(command, cwd, 0, "", "", .01, termination="completed")

    monkeypatch.setattr(v2, "run_command", transport)
    result = v2.execute_plans(
        [plan], [claim], materials, tmp_path / "repair", runner=v2.docker_runner,
        repairer=lambda *args: v2.Repair(dependencies=["numpy"], reason="fixture infrastructure repair"),
    )
    row = result.ledger[0]
    assert len(runs) == len(row["attempts"]) == 2
    assert row["attempts"][0]["operation_failures"][0]["component"] == "execution.runner"
    assert row["attempts"][0]["environment"]["process_termination"]["kind"] == "launch_failed"
    assert row["operation_failures"] == result.delivery_checks == []
    assert result.claims[0].evidence[0].aligned
    saved = json.loads(Path(row["attempts"][0]["request"]["run_dir"], "ledger.json").read_text("utf-8"))
    assert saved["attempts"][0]["environment"]["process_termination"] == row["attempts"][0]["environment"]["process_termination"]
