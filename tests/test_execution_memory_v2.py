"""One connected offline regression for bounded source and repair evidence."""

import difflib
import hashlib
import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

from common import run_stats
from fact_generation.execution import runtime_launch, v2
from tests import test_execution_v2 as fixtures

inputs = fixtures.inputs


def test_bounded_repair_retains_complete_raw_and_closes_unread_sources(inputs, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline boundary was not mocked")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(v2, "docker_runner", forbidden)
    monkeypatch.setattr(v2, "docker_ensure_paper_image", forbidden)
    monkeypatch.setattr(v2, "run_command", forbidden)
    monkeypatch.setattr(os, "getenv", lambda name, default=None: default)
    monkeypatch.setattr(os.environ, "get", lambda name, default=None: default)
    monkeypatch.setattr(run_stats, "stats_path", lambda: None)
    calls = []
    client = types.ModuleType("llm.client")
    client.llm_json = lambda *a, **kw: calls.append(a) or {}
    client.resolve_llm_config = lambda: object()
    monkeypatch.setitem(sys.modules, "llm.client", client)
    plan, claim, materials = inputs
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __init__(self, stream, path):
            self.stream, self.path = stream, path

        def read(self, size=-1):
            assert 0 <= size <= 1024 * 1024, "unbounded source/repair read"
            reads.append((str(self.path), size))
            return self.stream.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def guarded_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        monitored = {"workspace", "repair_evidence", "source_gap", "released"}
        if "r" in mode and monitored.intersection(Path(path).parts) and Path(path).is_relative_to(tmp_path):
            return BoundedReader(stream, path)
        return stream

    monkeypatch.setattr(Path, "open", guarded_open)

    def bytes_at(path):
        chunks = []
        with path.open("rb") as stream:
            while part := stream.read(1024 * 1024):
                chunks.append(part)
        return b"".join(chunks)

    attempts = []

    def runner(request):
        attempts.append(request.repair_round)
        workspace = Path(request.workspace)
        with (workspace / "checkpoint.bin").open("wb") as stream:
            stream.truncate(1024 * 1024 + 17)
        (workspace / "generated.bin").write_bytes(b"\x00\xff\x01")
        return v2.RunOutcome(returncode=1, stderr="fixture failed launch")

    def mutate(request, outcome):
        workspace = Path(request.workspace)
        with (workspace / "checkpoint.bin").open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"X")
        (workspace / "generated.bin").unlink()
        (workspace / "added.json").write_bytes(b'{"worker": 1}\n')
        return v2.Repair(dependencies=["numpy"], reason="fixture direct mutation")

    result = v2._execute_plans(
        [plan], [claim], materials, tmp_path / "mutation",
        config={"refine_with_llm": False}, runner=runner, repairer=mutate,
    )
    record = result.ledger[0]["repairs"][0]
    assert "unbounded source/repair read" not in record["reason"]
    assert not record["accepted"] and attempts == [0] and not result.claims[0].evidence
    changes = {row["path"]: row for row in record["unauthorized_file_changes"]}
    assert {name: row["kind"] for name, row in changes.items()} == {
        "checkpoint.bin": "modified", "generated.bin": "deleted", "added.json": "added",
    }
    for row in changes.values():
        for side in ("before", "after"):
            if row[side] is not None:
                artifact = Path(row[side]["artifact_locator"])
                raw = bytes_at(artifact)
                assert not artifact.is_relative_to(Path(result.ledger[0]["workspace"]))
                assert len(raw) == row[side]["size"]
                assert hashlib.sha256(raw).hexdigest() == row[side]["sha256"]
    checkpoint = changes["checkpoint.bin"]
    assert checkpoint["before"]["size"] == 1024 * 1024 + 17
    assert checkpoint["before"]["sha256"] != checkpoint["after"]["sha256"]
    assert bytes_at(Path(checkpoint["before"]["artifact_locator"]))[-1:] == b"\x00"
    assert bytes_at(Path(checkpoint["after"]["artifact_locator"]))[-1:] == b"X"
    assert record["file_diff_display"]["checkpoint.bin"]["status"] == "omitted"
    assert "capacity" in record["file_diff_display"]["checkpoint.bin"]["reason"]
    expected = "".join(difflib.unified_diff(
        [], ['{"worker": 1}\n'], fromfile="added.json.before", tofile="added.json.after",
    ))
    assert record["unauthorized_file_diffs"]["added.json"] == expected
    assert record["before_file_manifest"]["generated.bin"]["artifact_locator"]
    assert record["after_file_manifest"]["added.json"]["artifact_locator"]

    workspace = tmp_path / "source_gap"
    workspace.mkdir()
    large = workspace / "eval.py"
    with large.open("wb") as stream:
        stream.write(b"def driver():\n    return 1\n" + b"#" * 65537)
    missing_command = plan.model_copy(deep=True)
    missing_command.task.command = []
    with pytest.raises(ValueError, match=r"capacity|complete.*source") as capacity:
        v2._refine(missing_command, workspace, v2.ExecutionConfig())
    gap = capacity.value.source_read_scope
    assert "eval.py" not in gap["read"] and "eval.py" in gap["unavailable"]
    assert gap["sources"]["eval.py"]["complete"] is False
    assert gap["sources"]["eval.py"]["size"] == len(bytes_at(large))
    assert gap["sources"]["eval.py"]["sha256"] == hashlib.sha256(bytes_at(large)).hexdigest()
    assert calls == []
    known_command, _, _ = v2._refine(plan, workspace, v2.ExecutionConfig())
    assert known_command == plan.task.command
    site = {"path": "eval.py", "qualname": "driver", "firstlineno": 1}
    monkeypatch.setattr(runtime_launch, "prepare_site", forbidden)
    binding = runtime_launch.bind_source_sites([site], workspace=workspace, supplied_files={})
    assert binding["status"] == "unresolved" and binding["sites"] == []
    small = workspace / "small.py"
    small.write_bytes(b"def driver():\n    return 1\n")
    monkeypatch.setattr(runtime_launch, "prepare_site", lambda path, qualname, firstlineno: {
        "sha256": "0" * 64, "qualname": qualname, "firstlineno": firstlineno,
    })
    conflict = runtime_launch.bind_source_sites(
        [{**site, "path": "small.py"}], workspace=workspace,
        supplied_files={"small.py": bytes_at(small).decode()},
    )
    assert conflict["status"] == "unresolved" and conflict["sites"] == []
    assert conflict["unresolved"][0]["reason"] == "source_site_identity_changed_after_complete_read"

    alias = tmp_path / "mock_junction"
    real = tmp_path / "alias_target"
    alias.mkdir()
    real.mkdir()
    (real / "eval.py").write_bytes(b"def driver():\n    return 1\n")
    original_lstat, original_resolve = Path.lstat, Path.resolve
    resolved_alias = []

    def reparse_lstat(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if Path(path).absolute() == alias:
            return types.SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info

    def alias_resolve(path, *args, **kwargs):
        if Path(path).absolute().is_relative_to(alias):
            resolved_alias.append(str(path))
            return original_resolve(real / Path(path).absolute().relative_to(alias), *args, **kwargs)
        return original_resolve(path, *args, **kwargs)

    with monkeypatch.context() as linked:
        linked.setattr(Path, "lstat", reparse_lstat)
        linked.setattr(Path, "resolve", alias_resolve)
        linked.setattr(Path, "is_junction", lambda _: False, raising=False)
        with pytest.raises(ValueError, match="linked"):
            v2._refine(missing_command, alias, v2.ExecutionConfig())
        refused = runtime_launch.bind_source_sites(
            [site], workspace=alias, supplied_files={"eval.py": "def driver():\n    return 1\n"},
        )
        assert refused["status"] == "unresolved" and refused["sites"] == []
        assert "linked" in refused["unresolved"][0]["reason"]
        assert resolved_alias == [] and calls == []

    def wrapper_runner(request):
        if request.repair_round == 0:
            return v2.RunOutcome(returncode=1)
        raw = bytes_at(Path(request.workspace) / request.command[1])
        assert b"subprocess.call(['python', 'eval.py'])" in raw
        return v2.RunOutcome(returncode=0)

    wrapped = v2._execute_plans(
        [plan], [claim], materials, tmp_path / "wrapper",
        config={"refine_with_llm": False}, runner=wrapper_runner,
        repairer=lambda *_: v2.Repair(wrapper=True, reason="fixture forwarding"),
    )
    accepted = wrapped.ledger[0]["repairs"][0]
    text = "import subprocess\nraise SystemExit(subprocess.call(['python', 'eval.py']))\n"
    assert accepted["accepted"] and accepted["diff"]
    assert accepted["file_diffs"][".factreview/wrapper_1.py"] == "".join(
        difflib.unified_diff([], text.splitlines(keepends=True),
                             fromfile="/dev/null", tofile=".factreview/wrapper_1.py")
    )
    assert reads and all(0 <= size <= 1024 * 1024 for _, size in reads)
    assert calls == [] and v2.ExecutionConfig().max_attempts == 3

    # L2 may leave its entry unresolved; complete README launch instructions
    # still discover an existing script without granting unread source roles.
    (workspace / "README.md").write_text("Run python small.py.\n", encoding="utf-8")
    discovery_calls = []

    def discover(prompt, *args, **kwargs):
        discovery_calls.append(json.loads(prompt))
        return {"command": ["python", "small.py"], "metric_output": None}

    with monkeypatch.context() as discovery:
        discovery.setattr(client, "llm_json", discover)
        for entry, config in ((None, None), ("small.py", "missing-config.json")):
            candidate = plan.model_copy(deep=True)
            candidate.task.entry_script, candidate.task.config = entry, config
            candidate.task.command = []
            command, _, scope = v2._refine(candidate, workspace, v2.ExecutionConfig())
            assert command == ["python", "small.py"]
            assert scope["source_sites"]["status"] == "unresolved"
            assert scope["source_read_scope"]["sources"]["README.md"]["complete"] is True
    assert len(discovery_calls) == 2 and calls == []
