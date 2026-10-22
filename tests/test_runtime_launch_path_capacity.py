from __future__ import annotations

import json
import re
from pathlib import Path

from fact_generation.execution import runtime_launch


def test_windows_long_scratch_keeps_full_nonce_and_refuses_over_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_launch.sys, "platform", "win32")
    workspace = tmp_path / "released"
    workspace.mkdir()
    entry = workspace / "eval.py"
    source = "def measured(value):\n    return value\nmeasured(2)\n"
    entry.write_bytes(source.encode())
    sites = runtime_launch.bind_source_sites(
        [{"path": "eval.py", "qualname": "measured", "firstlineno": 1}],
        workspace=workspace,
        supplied_files={"eval.py": source},
    )
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()

    def scratch_at_length(length):
        padding = length - len(str(scratch_parent)) - 1
        assert 0 < padding <= 200
        path = scratch_parent / ("s" * padding)
        path.mkdir()
        assert len(str(path)) == length
        return path

    scratch = scratch_at_length(196)
    inputs = dict(
        entry_script="eval.py", workdir=".", workspace=workspace,
        metric_output="metrics.json", source_sites=sites,
        trusted_dir=tmp_path / "trusted", runtime_dir=scratch, repair_round=0,
    )
    launch = runtime_launch.prepare_observer_launch(["python", "eval.py"], **inputs)
    assert launch["status"] == "ready" and launch["unresolved"] == []
    output = Path(launch["audit_output"])
    assert re.fullmatch(r"o0_[a-f0-9]{32}", output.parent.name)
    assert output.parent.parent == scratch and len(str(output.parent)) < 248
    assert len(str(Path(launch["metric_file"]["capture_path"]))) < 260
    assert not output.exists()
    assert Path(launch["metric_file"]["capture_path"]).read_bytes() == b""
    assert (workspace / "metrics.json").read_bytes() == b""
    config = json.loads((inputs["trusted_dir"] / "config.json").read_text("utf-8"))
    assert config["output"] == "/workspace/run_dir/" + output.relative_to(scratch).as_posix()
    assert launch["original_command"] == ["python", "eval.py"]
    assert launch["mounts"]["/app"] == {"source": str(workspace), "mode": "ro"}
    assert entry.read_bytes() == source.encode()

    second = runtime_launch.prepare_observer_launch(
        ["python", "eval.py"], **dict(inputs, metric_output=None, trusted_dir=tmp_path / "trusted-2")
    )
    assert second["status"] == "ready" and second["unresolved"] == []
    assert Path(second["audit_output"]).parent != output.parent
    assert re.fullmatch(r"o0_[a-f0-9]{32}", Path(second["audit_output"]).parent.name)
    assert not {"aligned", "supported", "observations"} & launch.keys()

    oversized = scratch_at_length(210)
    refused = runtime_launch.prepare_observer_launch(
        ["python", "eval.py"],
        **dict(inputs, runtime_dir=oversized, trusted_dir=tmp_path / "oversized-trusted",
               metric_output="other.json"),
    )
    assert refused["status"] == "unresolved"
    assert refused["unresolved"] == [{"reason": "observer_path_capacity_exceeded"}]
    assert not (tmp_path / "oversized-trusted").exists()
    assert list(oversized.iterdir()) == []
    assert not (workspace / "other.json").exists()
