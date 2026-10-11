"""One combined operator-sandbox control; all external boundaries are mocked."""

import json

import pytest
from pydantic import ValidationError

from fact_generation.execution import v2
from fact_generation.execution.tools import docker
from fact_generation.execution.v2_config import validate_sandbox_options
from tests.execution_science_fixtures import accuracy_fixture
from tests.test_execution_workspace_readonly_v2 import (
    _ordinary_inputs,
)
from tests.test_execution_workspace_readonly_v2 import (
    test_default_runner_keeps_workspace_readonly_before_observer_qualification as _workspace_control,
)
from tests.test_torch_inference_execution_v2 import (
    test_full_cpu_inference_requires_actual_ids_loaded_state_and_whole_metric as _nn_control,
)


def test_public_execution_applies_immutable_operator_sandbox_to_all_native_routes(tmp_path, monkeypatch):
    default = v2.ExecutionConfig().sandbox
    policy = default.model_dump(mode="json")
    assert policy == {"version": "commercial-sandbox-v1", "cpus": 2.0, "memory_bytes": 4294967296,
        "pids_limit": 256, "network": "none", "uid": 1000, "gid": 1000,
        "read_only_rootfs": True, "cap_drop": "ALL", "no_new_privileges": True}
    for network in ("none", "bridge", "review-isolated_net.1"):
        assert type(default).model_validate({"network": network}).network == network
    for invalid in ({"network": "host"}, {"network": "container:author"}, {"uid": 0}, {"gid": 0},
                    {"read_only_rootfs": False}, {"cap_drop": "NET_ADMIN"}, {"no_new_privileges": False},
                    {"cpus": 0}, {"memory_bytes": 0}, {"pids_limit": 0}):
        with pytest.raises(ValidationError):
            type(default).model_validate(invalid)

    native_helper = docker.docker_run_paper_image
    helper_input = {"image": "ipc-control", "paper_root_host": str(tmp_path),
                    "run_dir_host": str(tmp_path), "cwd_container": "/app",
                    "cmd": ["python", "eval.py"], "env_passthrough": []}
    default_argv = native_helper(**helper_input, sandbox=default)
    prefix = default_argv[default_argv.index("run") + 1:default_argv.index("ipc-control")]
    assert "--ipc" not in default_argv
    validate_sandbox_options(prefix, default)
    private_argv = native_helper(**helper_input, sandbox=default, ipc="private")
    assert private_argv.count("--ipc") == 1 and private_argv[private_argv.index("--ipc") + 1] == "private"
    validate_sandbox_options(private_argv[private_argv.index("run") + 1:private_argv.index("ipc-control")], default)
    for invalid_ipc in ("host", "container:author", "shareable", "none", "", "private ", "PRIVATE"):
        with pytest.raises(ValueError, match="sandbox IPC"):
            native_helper(**helper_input, sandbox=default, ipc=invalid_ipc)
        with pytest.raises(ValueError, match="sandbox IPC"):
            validate_sandbox_options([*prefix, "--ipc", invalid_ipc], default)
    for duplicate_ipc in ("private", "host", "container:author"):
        with pytest.raises(ValueError, match="sandbox IPC"):
            validate_sandbox_options([*prefix, "--ipc", "private", "--ipc", duplicate_ipc], default)
    with pytest.raises(ValueError):
        validate_sandbox_options([*prefix, "--ipc"], default)
    supplied = []

    def checked_helper(**kwargs):
        selected = kwargs["sandbox"]
        assert selected.model_dump(mode="json") == policy
        argv = native_helper(**kwargs)
        expected = {"--cpus": "2.0", "--memory": "4294967296", "--memory-swap": "4294967296",
            "--pids-limit": "256", "--network": "none", "--user": "1000:1000",
            "--cap-drop": "ALL", "--security-opt": "no-new-privileges=true"}
        for option, value in expected.items():
            assert argv.count(option) == 1 and argv[argv.index(option) + 1] == value
        assert argv.count("--read-only") == 1
        assert "HOME=/workspace/run_dir/home" in argv and "TMPDIR=/workspace/run_dir/.tmp" in argv
        assert (v2.Path(kwargs["run_dir_host"]) / "home").is_dir()
        assert (v2.Path(kwargs["run_dir_host"]) / ".tmp").is_dir()
        supplied.append(selected.model_dump(mode="json"))
        return argv

    with monkeypatch.context() as boundary:
        boundary.setattr(v2, "docker_run_paper_image", checked_helper)
        root = tmp_path / "ordinary"
        root.mkdir()
        _workspace_control(root, boundary)
    with monkeypatch.context() as boundary:
        boundary.setattr(v2, "docker_run_paper_image", checked_helper)
        root = tmp_path / "nn"
        root.mkdir()
        _nn_control(root, boundary)
    assert len(supplied) == 10 and all(item == policy for item in supplied)

    with monkeypatch.context() as boundary:
        boundary.setattr(docker, "_docker_proxy_env", lambda _: {})
        boundary.setattr(docker, "_docker_run_user_args", lambda: ["--user", "123:456"])
        legacy = native_helper(image="legacy", paper_root_host=str(tmp_path), run_dir_host=str(tmp_path),
            cwd_container="/app", cmd=["python", "eval.py"], env_passthrough=[])
        assert legacy[legacy.index("--user") + 1] == "123:456"
        assert not any(option in legacy for option in ("--cpus", "--memory", "--pids-limit", "--network", "--read-only"))
        assert "HOME=/workspace/run_dir/home" not in legacy and "TMPDIR=/workspace/run_dir/.tmp" not in legacy
        for legacy_ipc in ("host", "container:author", "shareable", "none", "private", "private ", "PRIVATE"):
            legacy_argv = native_helper(**helper_input, ipc=legacy_ipc)
            assert legacy_argv.count("--ipc") == 1 and legacy_argv[legacy_argv.index("--ipc") + 1] == legacy_ipc
        assert native_helper(**helper_input, ipc="") == native_helper(**helper_input)

    root = tmp_path / "repair"
    root.mkdir()
    plan, claim, materials = _ordinary_inputs(root)
    accuracy_fixture((plan, claim, materials), monkeypatch)
    config = v2.ExecutionConfig(refine_with_llm=False, max_attempts=1)
    calls = []

    def runner(request):
        calls.append(request.config.sandbox.model_dump(mode="json"))
        assert len(calls) == 1
        return v2.RunOutcome(returncode=1, issue="environment fixture")

    def repairer(request, outcome):
        assert request.config.sandbox.model_dump(mode="json") == policy
        config.sandbox.cpus = 99.0
        return v2.Repair(dependencies=["numpy"], reason="mock environment repair")

    result = v2.execute_plans([plan], [claim], materials, root / "execute", config=config,
                              runner=runner, repairer=repairer)
    assert calls == [policy] and not result.claims[0].evidence
    assert result.ledger[0]["config"]["sandbox"] == policy
    assert "sandbox policy changed" in result.ledger[0]["reason"].lower()
    assert len(result.ledger[0]["attempts"]) == 1
    assert result.ledger[0]["training_budget"] == 0 and result.ledger[0]["config"]["max_attempts"] == 1
    assert json.loads((root / "execute/ledger.json").read_text(encoding="utf-8"))[0]["config"]["sandbox"] == policy
