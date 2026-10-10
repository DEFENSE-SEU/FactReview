"""One local capacity control; external execution and author exec are forbidden."""

import hashlib
import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

from fact_generation.execution import runtime_launch, runtime_observer


def test_observer_compiles_only_complete_bounded_source_and_hashes_full_tail(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("external execution is forbidden")

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    for name in ("llm.client.llm_json", "requests.sessions.Session.request", "httpx.Client.send",
                 "httpx.AsyncClient.send", "fact_generation.execution.v2.docker_runner"):
        monkeypatch.setattr(name, forbidden)
    workspace, scratch = tmp_path / "author", tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir()
    trusted = tmp_path / "trusted"
    entry, config = workspace / "entry.py", workspace / "config.py"
    entry_bytes = b"# author entry\n" + b"#" * 65536
    config_bytes = b"# coding: latin-1\n# \xe9\ndef selected():\n    return 1\n"
    entry.write_bytes(entry_bytes)
    config.write_bytes(config_bytes)
    original_command = ["python", "entry.py", "actual-argument"]
    original_open, original_read_bytes = Path.open, Path.read_bytes
    proposal = {"path": "config.py", "qualname": "selected", "firstlineno": 3}
    # Encoding cookies are preserved by bytes compile. The host supply uses
    # the same replacement-text comparison already required by the binder.
    binding = runtime_launch.bind_source_sites(
        [proposal], workspace=workspace,
        supplied_files={"config.py": config_bytes.decode("utf-8", errors="replace")},
    )
    assert binding["status"] == "bound"
    launch = runtime_launch.prepare_observer_launch(
        original_command, entry_script="entry.py", workdir=".", workspace=workspace,
        metric_output=None, source_sites=binding, trusted_dir=trusted,
        runtime_dir=scratch, repair_round=0,
    )
    assert launch["status"] == "unresolved"
    assert "capacity" in launch["unresolved"][0]["reason"]
    assert not trusted.exists() and list(scratch.iterdir()) == []
    assert original_command == launch["original_command"] == ["python", "entry.py", "actual-argument"]
    assert original_read_bytes(entry) == entry_bytes and original_read_bytes(config) == config_bytes

    # Only author-source reads are instrumented. Full SHA may stream every
    # byte in 1 MiB chunks; no whole-file buffer or prefix compile is allowed.
    author_reads, compiled = [], []
    source_paths = {entry, config}

    class SourceStream:
        def __init__(self, stream, path):
            self.stream, self.path = stream, path

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            author_reads.append((str(self.path), size))
            assert 0 <= size <= 1048576, "unbounded observer source read"
            return self.stream.read(size)

    def bounded_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return SourceStream(stream, path) if path in source_paths and mode == "rb" else stream

    real_compile = compile

    def tracked_compile(source, filename, *args, **kwargs):
        compiled.append((source, filename))
        assert type(source) is bytes and len(source) <= 65536
        return real_compile(source, filename, *args, **kwargs)

    monkeypatch.setattr(Path, "open", bounded_open)
    monkeypatch.setattr(runtime_observer, "compile", tracked_compile, raising=False)
    with pytest.raises(ValueError, match="capacity"):
        runtime_observer.prepare_site(str(entry), "unused", 1)
    assert compiled == []
    site = runtime_observer.prepare_site(str(config), "selected", 3)
    expected_module = real_compile(config_bytes, str(config.resolve()), "exec",
                                   dont_inherit=True, optimize=sys.flags.optimize)
    expected_function = next(code for code in expected_module.co_consts if type(code) is types.CodeType)
    assert set(site) == {"path", "sha256", "qualname", "firstlineno", "code_sha256"}
    assert site["sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert site["code_sha256"] == runtime_observer._code_sha(expected_function)
    assert compiled == [(config_bytes, str(config.resolve()))]

    startup_exec = []
    monkeypatch.setattr(runtime_observer, "exec", lambda *_: startup_exec.append(True), raising=False)
    report = runtime_observer.run_observed(
        "entry.py", [], str(workspace), [site], str(tmp_path / "startup.json"),
    )
    assert report["execution"]["status"] == "not_started" and startup_exec == []
    assert any("capacity" in reason for reason in report["unresolved"])
    assert len(compiled) == 2 and all(source == config_bytes for source, _ in compiled)
    assert json.loads((tmp_path / "startup.json").read_text()) == report

    small_entry = b"# full entry\n" + b"#" * (65536 - len(b"# full entry\n"))
    entry.write_bytes(small_entry)
    tail = b"#" * (1048576 + 1)

    def mocked_author_exec(code, namespace):
        assert type(code) is types.CodeType and namespace["__file__"] == str(entry.resolve())
        # No author code runs. A controlled host write simulates source growth
        # after startup so the real post-run full-file hash must see its tail.
        with original_open(entry, "ab") as stream:
            stream.write(tail)

    monkeypatch.setattr(runtime_observer, "exec", mocked_author_exec)
    report = runtime_observer.run_observed(
        "entry.py", ["actual-argument"], str(workspace), [site], str(tmp_path / "tail.json"),
    )
    assert report["execution"]["status"] == "completed" and report["events"] == []
    assert report["entry"]["argv"] == ["entry.py", "actual-argument"]
    assert report["entry"]["sha256"] == hashlib.sha256(small_entry).hexdigest()
    assert report["source_hashes_after"][str(entry.resolve())] == hashlib.sha256(small_entry + tail).hexdigest()
    assert "source_changed_or_unavailable" in report["unresolved"]
    assert report["source_hashes_after"][str(config.resolve())] == site["sha256"]
    assert compiled[-1] == (small_entry, str(entry.resolve())) and len(small_entry) == 65536
    assert author_reads and all(0 <= size <= 1048576 for _, size in author_reads)
    assert json.loads((tmp_path / "tail.json").read_text()) == report
