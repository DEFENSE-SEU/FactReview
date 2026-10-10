"""Existing refinement and Docker observer wiring; every transport is mocked."""

import json
from pathlib import Path

import pytest

from fact_generation.execution import v2
from tests import test_execution_v2 as original_fixtures
from util.subprocess_runner import CommandResult


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real external operation forbidden")

    for name in (
        "llm.client.llm_json", "requests.sessions.Session.request", "httpx.Client.send",
        "httpx.AsyncClient.send", "subprocess.run", "subprocess.Popen",
    ):
        monkeypatch.setattr(name, forbidden)
    monkeypatch.setattr(v2, "docker_ensure_paper_image", forbidden)
    monkeypatch.setattr(v2, "run_command", forbidden)


def test_existing_refinement_binds_only_supplied_source_sites_without_an_extra_call(tmp_path, monkeypatch):
    plan, _, materials = original_fixtures.inputs.__wrapped__(tmp_path)
    workspace = Path(materials.repository.root)
    source = "def measured(value):\n    return value\nmeasured(3)\n"
    (workspace / "eval.py").write_text(source, "utf-8")
    plan.task.command = []
    calls = []

    def refine(payload, prompt, **kwargs):
        calls.append((json.loads(payload), prompt, kwargs))
        return {"command": ["python", "eval.py"], "metric_output": None,
                "source_sites": [{"path": "eval.py", "qualname": "measured", "firstlineno": 1}]}

    monkeypatch.setattr("llm.client.llm_json", refine)
    command, output, audit = v2._refine(plan, workspace, v2.ExecutionConfig())
    assert len(calls) == 1 and calls[0][0]["files"]["eval.py"] == source
    assert command == ["python", "eval.py"] and output is None
    assert audit["source_sites"]["status"] == "bound"
    assert audit["source_sites"]["sites"][0]["path"] == "eval.py"
    assert "code_sha256" not in audit["source_sites"]["sites"][0]
    plan.task.command = list(command)
    _, _, existing = v2._refine(plan, workspace, v2.ExecutionConfig())
    assert len(calls) == 1 and existing["source_sites"]["status"] == "unresolved"


def test_mocked_docker_records_protected_observer_separately_and_rejects_missing_promised_report(tmp_path, monkeypatch):
    from fact_generation.execution.runtime_launch import bind_source_sites

    plan, _, materials = original_fixtures.inputs.__wrapped__(tmp_path)
    workspace = Path(materials.repository.root)
    source = "def measured(value):\n    return value\nmeasured(3)\n"
    (workspace / "eval.py").write_text(source, "utf-8")
    sites = bind_source_sites(
        [{"path": "eval.py", "qualname": "measured", "firstlineno": 1}],
        workspace=workspace, supplied_files={"eval.py": source},
    )
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *args, **kwargs: (True, "fixture-image"))
    supplied_report, commands = [True], []
    payload = {"observations": [original_fixtures.observation().model_dump()]}

    def transport(command, cwd, timeout_sec):
        commands.append(command)
        if command[1] == "run" and supplied_report[0]:
            trusted = next(Path(part.removesuffix(":/factreview-observer:ro"))
                           for part in command if part.endswith(":/factreview-observer:ro"))
            config = json.loads((trusted / "config.json").read_text("utf-8"))
            scratch = next(Path(part.removesuffix(":/workspace/run_dir"))
                           for part in command if part.endswith(":/workspace/run_dir"))
            target = scratch / Path(config["output"].removeprefix("/workspace/run_dir/"))
            target.write_text(json.dumps({
                "version": "python-source-events-v1", "events": [], "sites": [], "unresolved": ["site_not_observed"],
                "execution": {"status": "completed"},
            }), "utf-8")
        return CommandResult(command, str(cwd), 0, json.dumps(payload), "", .01, termination="completed")

    monkeypatch.setattr(v2, "run_command", transport)

    def request(name, source_sites):
        directory = tmp_path / name
        directory.mkdir()
        return v2.RunRequest(plan=plan, workspace=str(workspace), run_dir=str(directory),
            command=list(plan.task.command), workdir=".", metric_output=None, repair_round=0,
            config=v2.ExecutionConfig(), source_sites=source_sites)

    outcome = v2.docker_runner(request("observed", sites))
    assert outcome.returncode == 0 and outcome.observations[0].value == .9
    audit = outcome.environment["runtime_observer"]
    assert audit["status"] == "observed" and audit["report_status"] == "completed"
    assert audit["original_command"] == ["python", "eval.py"]
    assert any(part.endswith(":/app:ro") for part in commands[0])
    assert not any(part.endswith(":/app") for part in commands[0])
    assert Path(outcome.logs["runtime_observer"]).is_file()
    assert Path(outcome.logs["raw_output"]) != Path(outcome.logs["runtime_observer"])
    assert not {"aligned", "sufficient", "weights_used"} & audit.keys()

    supplied_report[0] = False
    missing = v2.docker_runner(request("missing", sites))
    assert missing.returncode != 0 and not missing.observations
    assert missing.operation_failures[0].component == "execution.runner"
    unbound = v2.docker_runner(request("unbound", None))
    assert unbound.returncode == 0 and unbound.observations[0].value == .9
    assert unbound.environment["runtime_observer"]["status"] == "unresolved"
    assert commands[-1][-2:] == ["python", "eval.py"]
