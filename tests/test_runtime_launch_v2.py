"""Two local launcher boundaries; no execution or external transport is allowed."""

import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Launcher controls cannot execute author code or external operations")

    for name in (
        "llm.client.llm_json", "requests.sessions.Session.request", "httpx.Client.send",
        "httpx.AsyncClient.send", "subprocess.run", "subprocess.Popen",
        "fact_generation.execution.v2.docker_runner", "fact_generation.execution.runtime_observer.run_observed",
    ):
        monkeypatch.setattr(name, forbidden)


def test_visible_source_sites_are_program_bound_and_closed(tmp_path, monkeypatch):
    from fact_generation.execution.runtime_launch import bind_source_sites

    workspace = tmp_path / "released"
    workspace.mkdir()
    source = workspace / "eval.py"
    source.write_text("def measured(value):\n    return value\ndef stream():\n    yield 1\n", encoding="utf-8")
    supplied = {"eval.py": source.read_text(encoding="utf-8")}
    proposal = {"path": "eval.py", "qualname": "measured", "firstlineno": 1}
    bound = bind_source_sites([proposal], workspace=workspace, supplied_files=supplied)
    assert bound["status"] == "bound" and bound["unresolved"] == []
    assert bound["sites"] == [dict(proposal, sha256=hashlib.sha256(source.read_bytes()).hexdigest())]
    assert "code_sha256" not in bound["sites"][0]
    assert not {"aligned", "supported", "verified", "role"} & bound.keys()
    for proposals, visible in (
        (None, supplied), ([], supplied), ([proposal] * 9, supplied),
        ([dict(proposal, code_sha256="0" * 64)], supplied),
        ([dict(proposal, role="metric")], supplied), ([dict(proposal, verified=True)], supplied),
        ([proposal], {}), ([proposal], {"eval.py": "changed source"}),
        ([dict(proposal, path="../eval.py")], supplied),
        ([dict(proposal, path=str(source))], supplied),
        ([dict(proposal, qualname="stream", firstlineno=3)], supplied),
        ([dict(proposal, firstlineno=True)], supplied),
        ([dict(proposal, firstlineno=2)], supplied), ([proposal, proposal], supplied),
    ):
        rejected = bind_source_sites(proposals, workspace=workspace, supplied_files=visible)
        assert rejected["status"] == "unresolved" and rejected["unresolved"]
    original_is_symlink = Path.is_symlink
    with monkeypatch.context() as local:
        local.setattr(Path, "is_symlink", lambda path: path == workspace or original_is_symlink(path))
        assert bind_source_sites([proposal], workspace=workspace, supplied_files=supplied)["status"] == "unresolved"
    assert source.read_text(encoding="utf-8") == supplied["eval.py"]


def test_stdout_launcher_is_readonly_closed_and_preserves_docker_arguments(tmp_path):
    from fact_generation.execution.runtime_launch import (
        bind_source_sites,
        prepare_observer_launch,
        protect_observer_docker_argv,
    )

    workspace = tmp_path / "released"
    (workspace / "sub").mkdir(parents=True)
    entry = workspace / "sub/eval.py"
    entry.write_text("def measured(value):\n    return value\nmeasured(2)\n", encoding="utf-8")
    original = entry.read_bytes()
    sites = bind_source_sites([{"path": "sub/eval.py", "qualname": "measured", "firstlineno": 1}],
        workspace=workspace, supplied_files={"sub/eval.py": entry.read_text(encoding="utf-8")})
    runtime = tmp_path / "scratch"
    runtime.mkdir()
    trusted = tmp_path / "trusted"
    command = ["python3", "eval.py", "--config", "settings.json", "--split", "test"]
    inputs = dict(entry_script="sub/eval.py", workdir="sub", workspace=workspace, metric_output=None,
        source_sites=sites, trusted_dir=trusted, runtime_dir=runtime, repair_round=0)
    observer = Path(__file__).resolve().parents[1] / "src/fact_generation/execution/runtime_observer.py"
    observer_before = observer.read_bytes()
    launch = prepare_observer_launch(command, **inputs)
    assert launch["status"] == "ready" and launch["unresolved"] == []
    assert launch["original_command"] == command
    assert launch["launch_command"] == ["python3", "/factreview-observer/observer.py", "/factreview-observer/config.json"]
    config = json.loads((trusted / "config.json").read_text())
    assert config["entry"] == "eval.py" and config["args"] == command[2:] and config["cwd"] == "/app/sub"
    assert config["sites"] == [dict(sites["sites"][0], path="/app/sub/eval.py")]
    assert config["output"].startswith("/workspace/run_dir/observer_attempt_0_")
    assert not Path(launch["audit_output"]).exists()
    assert (trusted / "observer.py").read_bytes() == observer_before
    for name, digest in launch["package_files"].items():
        assert hashlib.sha256((trusted / name).read_bytes()).hexdigest() == digest
    assert launch["source_sha256"]["sub/eval.py"] == hashlib.sha256(original).hexdigest()
    assert launch["scope_limits"] and not {"aligned", "supported", "observations"} & launch.keys()
    argv = ["docker", "run", "--rm", "--name", "kept-container", "--user", "1000:1000", "--gpus", "all",
        "-v", f"{workspace}:/app", "-v", f"{runtime}:/workspace/run_dir", "-w", "/app/sub",
        "-e", "PYTHONPATH=/app", "-e", "CUSTOM=kept", "frozen-image", *command]
    protected = protect_observer_docker_argv(argv, launch=launch)
    assert protected["status"] == "ready" and argv[-len(command):] == command
    changed = protected["argv"]
    assert f"{workspace}:/app:ro" in changed and f"{workspace}:/app" not in changed
    assert changed.count(f"{trusted}:/factreview-observer:ro") == 1
    assert changed[-len(launch["launch_command"]):] == launch["launch_command"]
    expected = [token.replace(f"{workspace}:/app", f"{workspace}:/app:ro") for token in argv[:-len(command)]]
    expected[-1:-1] = ["-v", f"{trusted}:/factreview-observer:ro"]
    assert changed == [*expected, *launch["launch_command"]]
    for option, value in (("--name", "kept-container"), ("--user", "1000:1000"), ("--gpus", "all"), ("-e", "CUSTOM=kept")):
        assert any(changed[index:index+2] == [option, value] for index in range(len(changed)-1))
    assert protect_observer_docker_argv(changed, launch=launch)["argv"] == changed
    mount_form = argv.copy()
    mount_index = mount_form.index(f"{workspace}:/app")
    mount_form[mount_index-1:mount_index+1] = ["--mount", f"type=bind,source={workspace},target=/app"]
    typed_mount = protect_observer_docker_argv(mount_form, launch=launch)
    assert typed_mount["status"] == "ready"
    assert f"type=bind,source={workspace},target=/app,readonly" in typed_mount["argv"]
    for invalid in (
        [*argv[:-len(command)-1], "-v", f"{workspace}:/app", "frozen-image", *command],
        [*argv[:-len(command)-1], "-v", f"{workspace}:/app/sub", "frozen-image", *command],
        [*argv[:2], "--entrypoint", "other", *argv[2:]],
        [*argv[:-1], "changed"],
        [token.replace(f"{runtime}:/workspace/run_dir", f"{workspace}:/workspace/run_dir") for token in argv],
    ):
        refusal = protect_observer_docker_argv(invalid, launch=launch)
        assert refusal["status"] == "unresolved" and refusal["argv"] == invalid
    for index, changes in enumerate((
        {"metric_output": "metrics.json"}, {"entry_script": "other.py"}, {"source_sites": {"status": "unresolved"}},
        {"trusted_dir": workspace / "trusted"}, {"runtime_dir": workspace},
    )):
        candidate = dict(inputs, trusted_dir=tmp_path / f"unsupported-{index}")
        candidate.update(changes)
        refused = prepare_observer_launch(command, **candidate)
        assert refused["status"] == "unresolved" and refused["unresolved"]
        assert not (Path(candidate["trusted_dir"]) / "observer.py").exists()
    assert prepare_observer_launch(["bash", "eval.py"], **dict(inputs, trusted_dir=tmp_path / "shell"))["status"] == "unresolved"
    assert prepare_observer_launch(["python", "-I", "eval.py"], **dict(inputs, trusted_dir=tmp_path / "flags"))["status"] == "unresolved"
    assert prepare_observer_launch(command, **inputs)["status"] == "unresolved"
    wrapper = workspace / ".factreview/wrapper_1.py"
    wrapper.parent.mkdir()
    wrapper.write_text("import subprocess\nraise SystemExit(subprocess.call(['python', 'sub/eval.py']))\n", encoding="utf-8")
    wrapped = prepare_observer_launch(["python", ".factreview/wrapper_1.py"],
        **dict(inputs, entry_script=".factreview/wrapper_1.py", workdir=".", trusted_dir=tmp_path / "wrapper"))
    assert wrapped["status"] == "unresolved" and not (tmp_path / "wrapper").exists()
    entry.write_text("def measured(value):\n    return value + 1\n", encoding="utf-8")
    stale = prepare_observer_launch(command, **dict(inputs, trusted_dir=tmp_path / "stale"))
    assert stale["status"] == "unresolved" and not (tmp_path / "stale").exists()
    assert observer.read_bytes() == observer_before


def test_bootstrap_namespaces_and_import_environment_fail_closed(tmp_path, monkeypatch):
    import sys

    from fact_generation.execution import runtime_launch

    root = tmp_path / "released"
    parent = root / "work/sub"
    parent.mkdir(parents=True)
    entry = parent / "eval.py"
    entry.write_text("def measured(value):\n    return value\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source_sites = runtime_launch.bind_source_sites(
        [{"path": "work/sub/eval.py", "qualname": "measured", "firstlineno": 1}],
        workspace=root, supplied_files={"work/sub/eval.py": entry.read_text(encoding="utf-8")})
    inputs = dict(entry_script="work/sub/eval.py", workdir="work", workspace=root, metric_output=None,
        source_sites=source_sites, trusted_dir=tmp_path / "trusted", runtime_dir=scratch, repair_round=0)
    command = ["python", "sub/eval.py"]
    for collision in (root / "json.py", root / "JSON.py", parent / "email", root / "work/pathlib.pyc",
                      root / "_json.cpython-311-x86_64-linux-gnu.so", parent / "usercustomize.py",
                      root / "sitecustomize.py", parent / "imp.py"):
        if collision.name == "email":
            collision.mkdir()
        else:
            collision.write_text("AUTHOR_SHADOW_ONLY = True\n", encoding="utf-8")
        refused = runtime_launch.prepare_observer_launch(command, **inputs)
        assert refused["status"] == "unresolved" and not (tmp_path / "trusted").exists()
        assert refused["unresolved"][0]["reason"] == "author_stdlib_or_startup_shadow"
        if collision.is_dir():
            collision.rmdir()
        else:
            collision.unlink()
    unknown = runtime_launch.prepare_observer_launch(command, **inputs, runtime_python="3.99")
    assert unknown["status"] == "unresolved" and not (tmp_path / "trusted").exists()
    namespace = runtime_launch._CPYTHON_311_STDLIB
    assert {"imp", "distutils", "json", "_json", "subprocess", "pathlib", "dis", "email"} <= namespace
    assert len(namespace) == 305
    ready = runtime_launch.prepare_observer_launch(command, **inputs, runtime_python="3.11")
    assert ready["status"] == "ready" and ready["bootstrap_scope"]["declared_runtime_minor"] == "3.11"
    assert set(ready["bootstrap_scope"]["container_search_dirs"]) == {"/app", "/app/work", "/app/work/sub"}
    argv = ["docker", "run", "--rm", "-v", f"{root}:/app", "-v", f"{scratch}:/workspace/run_dir",
        "-w", "/app/work", "-e", "PYTHONPATH=/app", "-e", "PYTHONUNBUFFERED=1", "fixture-image", *command]
    assert runtime_launch.protect_observer_docker_argv(argv, launch=ready)["status"] == "ready"
    for environment in ("PYTHONPATH=/app:/extra", "PYTHONPATH=", "PYTHONHOME=/extra",
                        "PYTHONUSERBASE=/app", "PYTHONSAFEPATH=1", "PATH=/app", "LD_PRELOAD=/app/hook.so",
                        "PYTHONPATH", "PYTHONPYCACHEPREFIX=/extra"):
        override = [*argv[:-len(command)-1], "-e", environment, "fixture-image", *command]
        rejected = runtime_launch.protect_observer_docker_argv(override, launch=ready)
        assert rejected["status"] == "unresolved" and rejected["argv"] == override
    custom = root / "stdlib_future.py"
    custom.write_text("AUTHOR_FUTURE_SHADOW = True\n", encoding="utf-8")
    with monkeypatch.context() as local:
        local.setattr(sys, "stdlib_module_names", sys.stdlib_module_names | {"stdlib_future"})
        changed = runtime_launch.prepare_observer_launch(command, **dict(inputs, trusted_dir=tmp_path / "changed"))
        assert changed["status"] == "unresolved" and not (tmp_path / "changed").exists()
    custom.unlink()
    unrelated = parent / "unrelated"
    unrelated.write_text("no imports\n", encoding="utf-8")
    original_is_symlink = Path.is_symlink
    with monkeypatch.context() as local:
        local.setattr(Path, "is_symlink", lambda path: path == unrelated or original_is_symlink(path))
        linked = runtime_launch.prepare_observer_launch(command, **dict(inputs, trusted_dir=tmp_path / "linked"))
        assert linked["status"] == "unresolved" and linked["unresolved"][0]["reason"] == "linked_path"
        assert not (tmp_path / "linked").exists()
    cache = scratch / ".pycache"
    cache.mkdir()
    for nonempty in (False, True):
        if nonempty:
            (cache / "injected.pyc").write_bytes(b"prior author attempt bytes")
        cached = runtime_launch.prepare_observer_launch(command,
            **dict(inputs, trusted_dir=tmp_path / f"cache-{nonempty}"))
        assert cached["status"] == "unresolved" and cached["unresolved"][0]["reason"] == "bootstrap_cache_not_fresh"
        assert not (tmp_path / f"cache-{nonempty}").exists()
    assert (cache / "injected.pyc").read_bytes() == b"prior author attempt bytes"
    (cache / "injected.pyc").unlink()
    cache.rmdir()
    cache.write_bytes(b"prior file")
    file_cache = runtime_launch.prepare_observer_launch(command, **dict(inputs, trusted_dir=tmp_path / "file-cache"))
    assert file_cache["status"] == "unresolved" and cache.read_bytes() == b"prior file"
    cache.unlink()
    with monkeypatch.context() as local:
        local.setattr(Path, "is_symlink", lambda path: path == cache or original_is_symlink(path))
        link_cache = runtime_launch.prepare_observer_launch(command, **dict(inputs, trusted_dir=tmp_path / "link-cache"))
        assert link_cache["status"] == "unresolved" and link_cache["unresolved"][0]["reason"] == "bootstrap_cache_not_fresh"
    assert ready["bootstrap_scope"]["cache_absent_at_prepare"] is True
