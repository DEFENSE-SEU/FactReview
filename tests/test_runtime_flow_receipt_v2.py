"""Production receipt/value-flow controls; all processes and author code are forbidden."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from fact_generation.execution.runtime_observer import _snapshot
from tests import test_runtime_flow_v2 as flow_fixture
from tests import test_runtime_receipt_v2 as receipt_fixture


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Receipt consumption controls forbid external and author execution")
    for name in ("llm.client.llm_json", "requests.sessions.Session.request", "httpx.Client.send",
                 "httpx.AsyncClient.send", "subprocess.run", "subprocess.Popen",
                 "fact_generation.execution.v2.docker_runner",
                 "fact_generation.execution.runtime_observer.run_observed",
                 "fact_generation.execution.runtime_observer.prepare_site"):
        monkeypatch.setattr(name, forbidden)


def captured(tmp_path, *, missing_command=False):
    from fact_generation.execution.runtime_flow import bind_builtin_flow
    original = tmp_path / "original"
    original.mkdir()
    _, proposal, args = flow_fixture.fixture(original)
    deployed = tmp_path / "deployed"
    deployed.mkdir()
    request, outcome, attempt = receipt_fixture.fixture(deployed)
    if missing_command:
        args["plan"].task.command = []
    request.plan = args["plan"].model_copy(deep=True)
    args["source_sites"] = request.source_sites
    binding = bind_builtin_flow(proposal, **args)
    assert binding["status"] == "bound"
    request.source_flow = copy.deepcopy(binding)
    request.source_flow_files = dict(args["supplied_files"])
    report_path = attempt / "runtime_observer.json"
    report = json.loads(report_path.read_text())
    data, weights = {"x": 3, "label": 5}, {"w": 2}
    calls = [{}, {"path": "data.json"}, {"path": "weights.json"},
             {"data": data, "weights": weights}, {"predictions": 6, "labels": 5}]
    returns = [1, data, weights, 6, 1]
    sites = [0, 1, 1, 2, 3]
    pattern = [("call", 0), ("call", 1), ("return", 1), ("call", 2), ("return", 2),
               ("call", 3), ("return", 3), ("call", 4), ("return", 4), ("return", 0)]
    report["events"] = []
    for order, (kind, invocation) in enumerate(pattern, 1):
        event = {"event": kind, "site": sites[invocation], "invocation": invocation + 1, "order": order}
        if kind == "call":
            event.update(parent_invocation=None if invocation == 0 else 1,
                         arguments={key: _snapshot(value, 65536) for key, value in calls[invocation].items()})
        else:
            event["value"] = _snapshot(returns[invocation], 65536)
        report["events"].append(event)
    raw = json.dumps(report).encode()
    report_path.write_bytes(raw)
    Path(outcome.environment["runtime_observer"]["audit_output"]).write_bytes(raw)
    outcome.environment["runtime_observer"]["report_sha256"] = hashlib.sha256(raw).hexdigest()
    (attempt / "runtime_observer_launch.json").write_text(
        json.dumps(outcome.environment["runtime_observer"]), encoding="utf-8")
    return binding, args, request, outcome, attempt


def test_authenticated_runtime_flow_binds_stdout_without_host_compilation_or_science(tmp_path, monkeypatch):
    from fact_generation.execution.runtime_flow import validate_builtin_flow
    def forbidden(*args, **kwargs):
        pytest.fail("Authenticated runtime identities cannot be host-compiled")
    monkeypatch.setattr("fact_generation.execution.runtime_flow.compile", forbidden, raising=False)
    monkeypatch.setattr("fact_generation.execution.runtime_flow._code_sha", forbidden)
    for missing in (False, True):
        directory = tmp_path / str(missing)
        directory.mkdir()
        binding, args, request, outcome, _ = captured(directory, missing_command=missing)
        before = args["plan"].model_dump(mode="json")
        result = validate_builtin_flow(binding, **args, runtime_request=request, runtime_outcome=outcome)
        assert result["status"] == "witnessed", result
        assert result["value"] == 1 and result["raw_output_binding"]["status"] == "bound"
        assert result["raw_output_binding"]["stdout"]["sha256"] == hashlib.sha256(b"1\n").hexdigest()
        assert result["raw_output_binding"]["driver_return_pointer"] == "/events/9/value"
        assert result["evidence_refs"]["producer_code_identity"]["interpreter"]["version"].startswith("3.11.")
        assert not result["scientific_qualification"] and not result["alignment"] and not result["support"]
        assert args["plan"].model_dump(mode="json") == before


def test_consumer_reauthenticates_receipt_and_rejects_mixed_or_changed_flow(tmp_path):
    from fact_generation.execution.runtime_flow import validate_builtin_flow
    for kind in ("partial", "mixed", "binding", "files", "plan", "argv", "report", "extra_stdout", "raw_type"):
        directory = tmp_path / kind
        directory.mkdir()
        binding, args, request, outcome, attempt = captured(directory)
        kwargs = dict(runtime_request=request, runtime_outcome=outcome)
        assert validate_builtin_flow(binding, **args, **kwargs)["status"] == "witnessed"
        if kind == "partial":
            kwargs.pop("runtime_outcome")
        elif kind == "mixed":
            kwargs["observer_report"] = json.loads((attempt / "runtime_observer.json").read_text())
        elif kind == "binding":
            request.source_flow["scientific_qualification"] = True
        elif kind == "files":
            request.source_flow_files["eval.py"] += "\n"
        elif kind == "plan":
            request.plan.priority = "low"
        elif kind == "argv":
            outcome.commands[-1][-1] = "/other/config.json"
        elif kind == "report":
            report_path = attempt / "runtime_observer.json"
            report = json.loads(report_path.read_text())
            report["events"][-1]["value"] = 2
            report_path.write_bytes(json.dumps(report).encode())
        elif kind == "extra_stdout":
            outcome.stdout = "1\n1\n"
            (attempt / "run_stdout.log").write_bytes(b"1\n1\n")
        else:
            kwargs["raw_value"] = True
        result = validate_builtin_flow(binding, **args, **kwargs)
        assert result["status"] == "unresolved", (kind, result)
        assert not result["scientific_qualification"] and not result["alignment"] and not result["support"]
