"""Receipt fixtures emulate trusted host capture; no container or author code runs."""

import hashlib
import json
from pathlib import Path

import pytest

from fact_generation.execution.runtime_launch import (
    bind_source_sites,
    prepare_observer_launch,
    protect_observer_docker_argv,
)
from fact_generation.execution.v2 import ExecutionConfig, ExecutionOperationFailure, RunOutcome, RunRequest
from tests import test_runtime_flow_v2 as flow_fixture


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Receipt controls forbid all external/process calls")
    for name in ("llm.client.llm_json", "requests.sessions.Session.request", "httpx.Client.send",
                 "httpx.AsyncClient.send", "subprocess.run", "subprocess.Popen",
                 "fact_generation.execution.v2.docker_runner"):
        monkeypatch.setattr(name, forbidden)


def fixture(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    input_dir = run / "fixture"
    input_dir.mkdir()
    source, _, args = flow_fixture.fixture(input_dir)
    workspace = run / "workspace"
    source.rename(workspace)
    plan = args["plan"]
    supplied = {"eval.py": (workspace / "eval.py").read_text()}
    proposals = [{key: row[key] for key in ("path", "qualname", "firstlineno")}
                 for row in args["source_sites"]["sites"]]
    sites = bind_source_sites(proposals, workspace=workspace, supplied_files=supplied)
    scratch = run / "runtime_scratch"
    scratch.mkdir()
    attempt = run / "attempt_0"
    attempt.mkdir()
    request = RunRequest(plan=plan, workspace=str(workspace), run_dir=str(run), command=["python", "eval.py"],
                         workdir=".", metric_output=None, repair_round=0,
                         config=ExecutionConfig(python_version="3.11"), source_sites=sites)
    launch = prepare_observer_launch(request.command, entry_script="eval.py", workdir=".", workspace=workspace,
        metric_output=None, source_sites=sites, trusted_dir=attempt / "observer_trusted", runtime_dir=scratch,
        repair_round=0, runtime_python="3.11")
    assert launch["status"] == "ready"
    env = {"EXECUTION_RUN_DIR": "/workspace/run_dir", "EXECUTION_ARTIFACT_DIR": "/workspace/run_dir/artifacts",
        "EXECUTION_PAPER_DIR": "/app", "EXECUTION_PAPER_ROOT": "/app", "PYTHONPATH": "/app",
        "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
        "PYTHONPYCACHEPREFIX": "/workspace/run_dir/.pycache", "XDG_CACHE_HOME": "/workspace/run_dir/.cache",
        "HF_HOME": "/workspace/run_dir/.cache/huggingface", "MPLCONFIGDIR": "/workspace/run_dir/.cache/matplotlib",
        "FACTREVIEW_REPAIR_ROUND": "0"}
    command = ["docker", "run", "--rm", "--name", "factreview-mock", "-v", str(workspace) + ":/app",
               "-v", str(scratch) + ":/workspace/run_dir", "-w", "/app/."]
    for key, value in env.items():
        command.extend(["-e", key + "=" + value])
    command.extend(["mock-image", *request.command])
    command = protect_observer_docker_argv(command, launch=launch)["argv"]
    expected = launch["source_sha256"]
    report = {"version": "python-source-events-v1", "unresolved": [], "execution": {"status": "completed"},
        "interpreter": {"version": "3.11.14 (mock producer)", "executable": "/usr/local/bin/python", "optimize": 0},
        "entry": {"argv": ["eval.py"], "cwd": "/app", "path": "/app/eval.py", "sha256": expected["eval.py"]},
        "sites": [dict(row, path="/app/" + row["path"], code_sha256=str(index + 1) * 64)
                  for index, row in enumerate(sites["sites"])],
        "events": [{"event": "call", "site": i, "invocation": i + 1, "parent_invocation": None,
                    "arguments": {}, "order": i * 2 + 1} if j == 0 else
                   {"event": "return", "site": i, "invocation": i + 1, "value": 1, "order": i * 2 + 2}
                   for i in range(4) for j in range(2)],
        "source_hashes_after": {"/app/" + key: value for key, value in expected.items()},
        "scope_limits": ["in_process_tampering_not_excluded", "events_do_not_prove_scientific_roles",
                         "native_and_child_execution_not_covered"]}
    raw = (json.dumps(report, indent=2) + "\n").encode()
    Path(launch["audit_output"]).write_bytes(raw)
    (attempt / "runtime_observer.json").write_bytes(raw)
    launch.update(status="observed", report_status="completed", report_unresolved=[],
                  report_sha256=hashlib.sha256(raw).hexdigest())
    (attempt / "runtime_observer_launch.json").write_text(json.dumps(launch), encoding="utf-8")
    (attempt / "run_stdout.log").write_bytes(b"1\n")
    manifest = {row.path: row.sha256 for row in args["materials"].repository.files}
    (run / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    outcome = RunOutcome(returncode=0, stdout="1\n", commands=[["docker", "image", "inspect", "mock-image"], command],
        environment={"transport": "docker", "image": "mock-image", "container_name": "factreview-mock",
                     "runtime_observer": launch, "process_termination": {"kind": "completed", "exception_type": None}},
        logs={"runtime_observer": str(attempt / "runtime_observer.json"),
              "runtime_observer_launch": str(attempt / "runtime_observer_launch.json"),
              "stdout": str(attempt / "run_stdout.log")})
    return request, outcome, attempt


def test_fixed_runtime_producer_receipt_never_host_compiles_or_grants_science(tmp_path, monkeypatch):
    from fact_generation.execution.runtime_receipt import read_observer_receipt
    request, outcome, _ = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Receipt must not host-compile runtime code identities")
    monkeypatch.setattr("fact_generation.execution.runtime_observer.prepare_site", forbidden)
    monkeypatch.setattr("fact_generation.execution.runtime_observer._code_sha", forbidden)
    result = read_observer_receipt(request, outcome)
    assert result["status"] == "received"
    assert result["report"]["interpreter"]["version"].startswith("3.11.")
    assert result["producer_code_identity"]["basis"] == "fixed_observer_actual_runtime_prepare_and_match"
    assert result["scientific_qualification"] is False and not result["alignment"] and not result["support"]
    assert result["run_identity"]["command"] == request.command and result["run_identity"]["runtime_root"] == "/app"
    assert result["evidence_refs"]["stdout"]["sha256"] == hashlib.sha256(b"1\n").hexdigest()


def test_receipt_modification_forgery_and_unknown_scope_refused(tmp_path):
    from fact_generation.execution.runtime_receipt import read_observer_receipt
    for kind in ("package", "config", "source", "report", "argv", "env", "stdout", "seal", "settings", "scope", "operation", "path", "resource", "duplicate", "readonly", "report_scope", "event_protocol", "env_unknown"):
        directory = tmp_path / kind
        directory.mkdir()
        request, outcome, attempt = fixture(directory)
        assert read_observer_receipt(request, outcome)["status"] == "received"
        if kind == "package":
            (attempt / "observer_trusted/observer.py").write_text("print('fake')", encoding="utf-8")
        elif kind == "config":
            path = attempt / "observer_trusted/config.json"
            value = json.loads(path.read_text())
            value["args"] = ["--changed"]
            path.write_text(json.dumps(value), encoding="utf-8")
        elif kind == "source":
            (Path(request.workspace) / "eval.py").write_text("print(1)", encoding="utf-8")
        elif kind == "report":
            path = attempt / "runtime_observer.json"
            value = json.loads(path.read_text())
            value["sites"][0]["code_sha256"] = "f" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
        elif kind == "argv":
            outcome.commands[-1][-1] = "/app/eval.py"
        elif kind == "env":
            index = outcome.commands[-1].index("PYTHONPATH=/app")
            outcome.commands[-1][index] = "PYTHONPATH=/outside"
        elif kind == "stdout":
            (attempt / "run_stdout.log").write_text("2\n", encoding="utf-8")
        elif kind == "seal":
            path = attempt / "runtime_observer_launch.json"
            value = json.loads(path.read_text())
            value["request_sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            outcome.environment["runtime_observer"] = value
        elif kind == "settings":
            request.plan.target_conditions[0].settings["split"] = "train"
        elif kind == "scope":
            outcome.environment["process_termination"]["kind"] = "unknown"
        elif kind == "operation":
            outcome.operation_failures = [ExecutionOperationFailure(component="execution.runner", reason="failed")]
        elif kind == "resource":
            request.plan.task.data_paths = ["other.json"]
        elif kind == "duplicate":
            outcome.commands.append(list(outcome.commands[-1]))
        elif kind == "readonly":
            index = outcome.commands[-1].index(str(Path(request.workspace)) + ":/app:ro")
            outcome.commands[-1][index] = str(Path(request.workspace)) + ":/app:rw"
        elif kind in {"report_scope", "event_protocol"}:
            report_path = attempt / "runtime_observer.json"
            report = json.loads(report_path.read_text())
            if kind == "report_scope":
                report["scope_limits"].append("unknown")
            else:
                report["events"][0]["unknown"] = True
            report_raw = json.dumps(report).encode()
            report_path.write_bytes(report_raw)
            launch_path = attempt / "runtime_observer_launch.json"
            launch = json.loads(launch_path.read_text())
            launch["report_sha256"] = hashlib.sha256(report_raw).hexdigest()
            launch_path.write_text(json.dumps(launch), encoding="utf-8")
            outcome.environment["runtime_observer"] = launch
        elif kind == "env_unknown":
            image_index = outcome.commands[-1].index("mock-image")
            outcome.commands[-1][image_index:image_index] = ["-e", "UNKNOWN_SCOPE=1"]
        else:
            outcome.logs["runtime_observer"] = str(attempt / "different.json")
        result = read_observer_receipt(request, outcome)
        assert result["status"] == "unresolved", kind
        assert not result["scientific_qualification"] and not result["alignment"] and not result["support"]


def test_receipt_snapshot_protocol_and_capacity_are_producer_bounded(tmp_path):
    from fact_generation.execution.runtime_observer import _snapshot
    from fact_generation.execution.runtime_receipt import read_observer_receipt
    request, outcome, attempt = fixture(tmp_path)
    report_path = attempt / "runtime_observer.json"
    original = json.loads(report_path.read_text(encoding="utf-8"))

    def captured(arguments, value):
        report = json.loads(json.dumps(original))
        report["events"][0]["arguments"] = arguments
        report["events"][1]["value"] = value
        raw = json.dumps(report, ensure_ascii=True).encode()
        report_path.write_bytes(raw)
        launch_path = attempt / "runtime_observer_launch.json"
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        launch["report_sha256"] = hashlib.sha256(raw).hexdigest()
        launch_path.write_text(json.dumps(launch), encoding="utf-8")
        outcome.environment["runtime_observer"] = launch
        return read_observer_receipt(request, outcome)

    legal = _snapshot((None, True, 1, 1.0, "文字", [False], {"key": (2,)}), 65536)
    valid = captured({"input": legal}, legal)
    assert valid["status"] == "received"
    assert not valid["scientific_qualification"] and not valid["alignment"] and not valid["support"]
    deep = 1
    for _ in range(13):
        deep = {"type": "list", "items": [deep]}
    malformed = [True, {"input": []}, {"input": {"type": "set", "items": []}},
        {"input": {"type": "dict", "items": [["x", 1], ["x", 2]]}},
        {"input": {"type": "tuple", "items": [1], "extra": True}},
        {"input": {"unresolved": "unsupported_snapshot_type"}}, {"input": float("inf")},
        {"input": 1 << 8192}, {"input": "文字" * 12000},
        {"input": {"type": "list", "items": [0] * 256}}, {"input": deep},
        {"a": "a" * 35000, "b": "b" * 35000}]
    for arguments in malformed:
        result = captured(arguments, 1)
        assert result["status"] == "unresolved"
        assert not result["scientific_qualification"] and not result["alignment"] and not result["support"]
    for value in ([], {"type": "dict", "items": [[1, 2]]}, deep):
        assert captured({}, value)["status"] == "unresolved"
