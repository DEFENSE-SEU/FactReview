"""Exercise Linux ownership and real Docker observer/output mounts.

Run with the project's Python interpreter. Requires a Linux Docker daemon and
the cached python:3.11-slim image (override with --image). A temporary volume
holds Linux files and Pydantic installed for the controller; it is removed on
exit. Author containers have no network, Docker socket, or project checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

AUTHOR_SOURCE = '''import json
import os
from pathlib import Path

def evaluate():
    return int(Path("input.txt").read_text())

value = evaluate()
try:
    Path("input.txt").write_text("changed")
except OSError:
    readonly = True
else:
    raise RuntimeError("input mount allowed a write")
Path(os.environ["HOME"], "home.txt").write_text("ok")
Path(os.environ["TMPDIR"], "tmp.txt").write_text("ok")
Path("metrics.json").write_text(json.dumps({"value": value, "readonly": readonly, "uid": os.getuid()}))
'''


def prepare(case: str, image: str) -> dict:
    from fact_generation.execution.runtime_launch import (
        bind_source_sites,
        prepare_observer_launch,
        protect_observer_docker_argv,
    )
    from fact_generation.execution.runtime_permissions import prepare_runtime_permissions
    from fact_generation.execution.tools.docker import docker_run_paper_image
    from fact_generation.execution.v2_config import SandboxPolicy

    root = Path("/probe") / case
    root.mkdir()
    host_uid = 2345 if case == "user2345" else 0
    if host_uid:
        os.chown(root, host_uid, host_uid)
        os.setgroups([])
        os.setgid(host_uid)
        os.setuid(host_uid)
    os.umask(0o077)
    workspace, scratch = root / "workspace", root / "scratch"
    for directory in (workspace, scratch, scratch / "home", scratch / ".tmp"):
        directory.mkdir()
    (workspace / "eval.py").write_text(AUTHOR_SOURCE)
    (workspace / "input.txt").write_text("42")
    sites = bind_source_sites(
        [{"path": "eval.py", "qualname": "evaluate", "firstlineno": 5}],
        workspace=workspace, supplied_files={"eval.py": AUTHOR_SOURCE},
    )
    policy = SandboxPolicy()
    assert policy.uid == (host_uid or 1000) and policy.gid == (host_uid or 1000)
    if host_uid:
        try:
            prepare_runtime_permissions(workspace=workspace, runtime_dir=scratch,
                                        sandbox=SandboxPolicy(uid=3456, gid=3456))
        except ValueError:
            pass
        else:
            raise AssertionError("unprivileged host accepted an incompatible sandbox identity")
    launch = prepare_observer_launch(
        ["python", "eval.py"], entry_script="eval.py", workdir=".", workspace=workspace,
        metric_output="metrics.json", source_sites=sites, trusted_dir=root / "trusted",
        runtime_dir=scratch, repair_round=0, runtime_python="3.11",
    )
    assert launch["status"] == "ready", launch
    before = hashlib.sha256((workspace / "eval.py").read_bytes()).hexdigest()
    prepare_runtime_permissions(
        workspace=workspace, runtime_dir=scratch, trusted_dir=Path(launch["trusted_dir"]),
        audit_output=Path(launch["audit_output"]),
        metric_capture=Path(launch["metric_file"]["capture_path"]), sandbox=policy,
    )
    assert hashlib.sha256((workspace / "eval.py").read_bytes()).hexdigest() == before
    command = docker_run_paper_image(
        image=image, paper_root_host=str(workspace), run_dir_host=str(scratch),
        cwd_container="/app", cmd=["python", "eval.py"], env_passthrough=[], sandbox=policy,
    )
    protected = protect_observer_docker_argv(command, launch=launch, sandbox=policy)
    assert protected["status"] == "ready", protected
    record = {"case": case, "host_uid": host_uid, "sandbox_uid": policy.uid,
              "command": protected["argv"], "launch": launch}
    (root / "record.json").write_text(json.dumps(record))
    return record


def inspect_case(case: str) -> dict:
    from fact_generation.execution.runtime_launch import prepare_observer_launch

    root = Path("/probe") / case
    record = json.loads((root / "record.json").read_text())
    launch = record["launch"]
    metrics = json.loads(Path(launch["metric_file"]["capture_path"]).read_text())
    report = json.loads(Path(launch["audit_output"]).read_text())
    assert metrics == {"value": 42, "readonly": True, "uid": record["sandbox_uid"]}, metrics
    assert report["execution"]["status"] == "completed", report
    assert report["events"], report
    assert (root / "workspace/input.txt").read_text() == "42"
    assert (root / "scratch/home/home.txt").read_text() == "ok"
    assert (root / "scratch/.tmp/tmp.txt").read_text() == "ok"
    assert not (root / "scratch/.pycache").exists(), "ordinary execution poisoned repair bootstrap cache"
    if record["host_uid"]:
        os.setgroups([])
        os.setgid(record["host_uid"])
        os.setuid(record["host_uid"])
    os.umask(0o077)
    retry = prepare_observer_launch(
        ["python", "eval.py"], entry_script="eval.py", workdir=".", workspace=root / "workspace",
        metric_output=None, source_sites={"version": "python-source-sites-v1", "status": "bound",
                                        "sites": launch["source_sites"], "unresolved": []},
        trusted_dir=root / "trusted_retry", runtime_dir=root / "scratch", repair_round=1,
        runtime_python="3.11",
    )
    assert retry["status"] == "ready", retry
    return {"case": case, "host_uid": record["host_uid"], "sandbox_uid": record["sandbox_uid"],
            "umask": "077", "metrics": metrics, "observer_events": len(report["events"]),
            "home_write": True, "tmp_write": True, "input_readonly": True,
            "bytecode_cache_created": False, "repair_observer_preparation": retry["status"]}


def hostile_paths() -> dict:
    from fact_generation.execution.runtime_permissions import prepare_runtime_permissions
    from fact_generation.execution.v2_config import SandboxPolicy

    root = Path("/probe/links")
    root.mkdir()
    sentinel = root / "sentinel"
    sentinel.write_text("preserve")
    before = sentinel.stat()
    rejected = []
    for case in ("scratch_symlink", "workspace_symlink", "workspace_hardlink", "capture_hardlink"):
        base = root / case
        workspace, scratch = base / "workspace", base / "scratch"
        for directory in (base, workspace, scratch, scratch / "home", scratch / ".tmp"):
            directory.mkdir()
        capture = None
        if case == "scratch_symlink":
            (scratch / "home").rmdir()
            (scratch / "home").symlink_to(root, target_is_directory=True)
        elif case == "workspace_symlink":
            (workspace / "input").symlink_to(sentinel)
        elif case == "workspace_hardlink":
            os.link(sentinel, workspace / "input")
        else:
            audit = scratch / "audit"
            audit.mkdir()
            capture = audit / "metric.json"
            os.link(sentinel, capture)
        try:
            prepare_runtime_permissions(
                workspace=workspace, runtime_dir=scratch, sandbox=SandboxPolicy(),
                audit_output=capture.parent / "events.json" if capture else None,
                metric_capture=capture,
            )
        except (ValueError, OSError):
            rejected.append(case)
        else:
            raise AssertionError(f"accepted unsafe path: {case}")
        after = sentinel.stat()
        assert (before.st_uid, before.st_gid, before.st_mode) == (after.st_uid, after.st_gid, after.st_mode)
        assert sentinel.read_text() == "preserve"
    return {"rejected": rejected, "outside_file_unchanged": True}


def run(command: list[str], *, timeout: int = 180) -> str:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inside", choices=("prepare", "inspect", "links"), help=argparse.SUPPRESS)
    parser.add_argument("--case", default="root")
    args = parser.parse_args()
    if args.inside:
        result = (prepare(args.case, args.image) if args.inside == "prepare"
                  else inspect_case(args.case) if args.inside == "inspect" else hostile_paths())
        print(json.dumps(result))
        return
    assert run(["docker", "info", "--format", "{{.OSType}}"]).strip() == "linux", "Linux Docker required"
    run(["docker", "image", "inspect", args.image])
    volume = "factreview-sandbox-check-" + uuid.uuid4().hex
    repo = Path(__file__).resolve().parents[1]
    script = "/repo/scripts/check_linux_sandbox.py"
    run(["docker", "volume", "create", volume])
    try:
        base = ["docker", "run", "--rm", "--name", volume + "-controller",
                "-v", f"{volume}:/probe", "-v", f"{repo}:/repo:ro",
                "-e", "PYTHONPATH=/repo/src:/probe/deps"]
        run([*base, args.image, "python", "-m", "pip", "install", "--disable-pip-version-check",
             "--target", "/probe/deps", "pydantic>=2.5"], timeout=300)
        mountpoint = run(["docker", "volume", "inspect", "--format", "{{.Mountpoint}}", volume]).strip()
        cases = []
        for case in ("root", "user2345"):
            controller = [*base, "--network", "none", args.image, "python", script]
            record = json.loads(run([*controller, "--inside", "prepare", "--case", case, "--image", args.image]))
            command = record["command"]
            for index, token in enumerate(command[:-1]):
                if token == "-v":
                    assert command[index + 1].startswith("/probe/"), command
                    command[index + 1] = mountpoint + command[index + 1][len("/probe"):]
            container_name = volume + "-" + case
            command[2:2] = ["--name", container_name]
            try:
                run(command)
            finally:
                subprocess.run(["docker", "rm", "--force", container_name],
                               capture_output=True, timeout=30)
            cases.append(json.loads(run([*controller, "--inside", "inspect", "--case", case])))
        links = json.loads(run([*controller, "--inside", "links"]))
        report = {"status": "passed", "image": args.image, "cases": cases, "path_guards": links}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        subprocess.run(["docker", "rm", "--force", volume + "-controller"],
                       capture_output=True, timeout=30)
        run(["docker", "volume", "rm", volume])


if __name__ == "__main__":
    main()
