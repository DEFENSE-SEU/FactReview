"""The formal file producer survives failed author execution and one repair."""
import json
from pathlib import Path

import pytest

from fact_generation.execution import runner as v2
from tests import test_execution as execution_fixtures
from tests.execution_science_fixtures import accuracy_fixture, record_metric_file, record_observer
from util.subprocess_runner import CommandResult

inputs = execution_fixtures.inputs


def test_formal_file_output_repair_preserves_failed_history_and_fresh_placeholder(inputs, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("File repair control forbids external services and processes")

    for target in ("subprocess.run", "subprocess.Popen", "requests.sessions.Session.request",
                   "httpx.Client.send", "httpx.AsyncClient.send", "urllib.request.urlopen"):
        monkeypatch.setattr(target, forbidden)
    plan, claim, materials = inputs
    plan.task.metric_output = "metrics.json"
    payload = accuracy_fixture(inputs, monkeypatch)
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *args, **kwargs: (True, "fixture-image"))
    commands, launches = [], []

    def transport(command, *, cwd, timeout_sec):
        commands.append(command)
        if command[1] == "rm":
            assert command == ["docker", "rm", "--force", launches[0]["container"]]
            return CommandResult(command, cwd, 0, launches[0]["container"], "", .01, termination="completed")
        assert command[:3] == ["docker", "run", "--rm"]
        attempt = len(launches)
        launch = json.loads((Path(cwd) / f"attempt_{attempt}" / "runtime_observer_launch.json").read_text(encoding="utf-8"))
        assert launch["status"] == "ready"
        placeholder = Path(launch["metric_file"]["workspace_placeholder"])
        assert placeholder.read_bytes() == b""
        launches.append({"launch": launch, "container": command[command.index("--name")+1]})
        if attempt == 0:
            # A real failed transport writes an incomplete output only into its
            # scratch slot. The read-only repository placeholder stays empty.
            Path(launch["metric_file"]["capture_path"]).write_bytes(b'{"partial":')
            return CommandResult(command, cwd, 1, "author failed before metric completion", "fixture dependency missing", .01,
                                 termination="completed")
        assert launches[0]["launch"]["metric_file"]["capture_path"] != launch["metric_file"]["capture_path"]
        record_metric_file(command, "metrics.json", payload)
        record_observer(command, cwd, payload)
        return CommandResult(command, cwd, 0, json.dumps({"ok": True, "out": "metrics.json"}), "", .01,
                             termination="completed")

    monkeypatch.setattr(v2, "run_command", transport)
    result = v2.execute_plans([plan], [claim], materials, tmp_path / "out", runner=v2.docker_runner,
        config=v2.ExecutionConfig(max_attempts=1),
        repairer=lambda *args: v2.Repair(dependencies=["numpy"], reason="fixture dependency repair"))
    row = result.ledger[0]
    assert len(row["attempts"]) == len(launches) == 2
    assert len(commands) == 3 and [cmd[1] for cmd in commands] == ["run", "rm", "run"]
    first, final = row["attempts"]
    assert first["returncode"] == 1 and final["returncode"] == 0
    assert first["observations"] == []
    assert Path(first["logs"]["stderr"]).read_text(encoding="utf-8") == "fixture dependency missing"
    assert row["repairs"][0]["proposal"]["dependencies"] == ["numpy"]
    cleanup = json.loads(Path(first["logs"]["container_cleanup"]).read_text(encoding="utf-8"))
    assert cleanup["status"] == "removed"
    discarded = json.loads(Path(first["logs"]["runtime_observer_launch"]).read_text(encoding="utf-8"))
    assert discarded["metric_placeholder_cleanup"]["status"] == "removed"
    assert Path(launches[0]["launch"]["metric_file"]["capture_path"]).read_bytes() == b'{"partial":'
    assert Path(launches[1]["launch"]["metric_file"]["workspace_placeholder"]).read_bytes() == b""
    assert launches[0]["container"] != launches[1]["container"]
    assert json.loads(Path(final["logs"]["raw_output"]).read_text(encoding="utf-8")) == payload
    assert Path(final["logs"]["metric_file_capture"]).is_file()
    evidence = result.claims[0].evidence[0]
    assert evidence.sufficient and evidence.aligned and evidence.direction == "support"
    assert row["reason"] == "" and len(row["repairs"]) == 1
    assert result.delivery_checks == []
