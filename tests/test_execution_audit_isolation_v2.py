"""The Docker transport exposes writable runtime storage without host audit files."""

import json
from pathlib import Path

import pytest

from fact_generation.execution import v2
from fact_generation.execution.tools import docker
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, Condition, ExecutionPlan, ExecutionTask, PaperTargetPassage
from schemas.materials import MaterialBlock, SharedMaterials
from util.subprocess_runner import CommandResult
from verification.experiment_targets import bind_execution_target


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "eval.py").write_text("print('released evaluation')\n", encoding="utf-8")
    text = "d1 test seed 1 accuracy is 0.9."
    loc = ClaimLocation(page=1, char_start=0, char_end=len(text))
    paper = tmp_path / "paper.md"
    paper.write_text(text, encoding="utf-8")
    condition = Condition(id="c1", dataset="d1", metric="accuracy", settings={"split": "test", "seed": 1})
    claim = Claim(
        id="claim1",
        text=text,
        loc=loc,
        source_block_id="b1",
        source_quote=text,
        conditions=[condition],
        needs=["Experiments"],
    )
    materials = SharedMaterials(
        paper_key="fixture",
        source_pdf="unused.pdf",
        markdown=text,
        markdown_path=str(paper),
        content_list_path="unused.json",
        provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=loc)],
        repository=index_repository(repo),
    )
    binding = bind_execution_target(
        claim,
        condition,
        PaperTargetPassage(block_id="b1", quote=text, token="0.9", value_context=text),
        materials,
    )
    plan = ExecutionPlan(
        id="p1",
        claim_id=claim.id,
        condition_ids=["c1"],
        target_conditions=[condition],
        task=ExecutionTask(entry_script="eval.py", command=["python", "eval.py"]),
        run_mode="evaluation",
        y_paper={"c1": 0.9},
        target_bindings={"c1": binding},
        feasibility="ready",
        priority="high",
    )
    monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *a, **kw: (True, "fixture-image"))
    monkeypatch.setattr(docker, "_docker_proxy_env", lambda _: {})
    monkeypatch.setattr(docker, "_docker_run_user_args", lambda: [])
    monkeypatch.setattr(docker, "docker_cmd", lambda args: ["docker", *args])

    def forbidden(*a, **kw):
        pytest.fail("An offline execution isolation test attempted an external process")

    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    return plan, claim, materials


@pytest.mark.parametrize("metric_file", [False, True])
def test_container_writes_cannot_overwrite_host_audit_and_valid_observations_survive(
    inputs, tmp_path, monkeypatch, metric_file
):
    plan, claim, materials = inputs
    observed = {"dataset": "d1", "metric": "accuracy", "settings": {"split": "test", "seed": 1}, "value": 0.9}
    payload = {"observations": [observed]}
    if metric_file:
        plan.task.metric_output = "metrics.json"
    seen = {}

    def transport(command, *, cwd, timeout_sec):
        run_dir = Path(cwd)
        manifest = run_dir / "source_manifest.json"
        before = manifest.read_bytes()
        mounts = [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "-v"]
        run_mount = next(m for m in mounts if m.endswith(":/workspace/run_dir"))
        writable = Path(run_mount.removesuffix(":/workspace/run_dir"))
        workspace = Path(next(m for m in mounts if m.endswith(":/app")).removesuffix(":/app"))
        assert writable != run_dir
        assert writable.is_dir() and writable.is_relative_to(run_dir)
        assert {"source_manifest.json", "ledger.json", "attempt_0"}.isdisjoint(
            path.name for path in writable.iterdir()
        )
        assert set(mounts) == {f"{writable}:/workspace/run_dir", f"{workspace}:/app"}
        # Same allowed container-relative write as the original vulnerability,
        # now confined to scratch instead of replacing the host-owned manifest.
        (writable / "source_manifest.json").write_text('{"forged":true}', encoding="utf-8")
        (writable / "artifacts").mkdir(exist_ok=True)
        (writable / "artifacts" / "output.txt").write_text("runtime output", encoding="utf-8")
        if metric_file:
            (workspace / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        seen.update(manifest=manifest, before=before, mounts=mounts, workspace=workspace)
        return CommandResult(command, str(cwd), 0, "FACTREVIEW_OBSERVATIONS=" + json.dumps(payload), "", 0.01)

    monkeypatch.setattr(v2, "run_command", transport)
    result = v2.execute_plans(
        [plan],
        [claim],
        materials,
        tmp_path / "out",
        config=v2.ExecutionConfig(refine_with_llm=False, max_attempts=0),
        runner=v2.docker_runner,
    )
    assert seen["manifest"].read_bytes() == seen["before"]
    assert result.claims[0].evidence[0].sufficient and result.claims[0].evidence[0].aligned
    assert result.ledger[0]["reason"] == ""
    log = Path(result.ledger[0]["attempts"][0]["logs"]["stdout"])
    assert log.parent.name == "attempt_0" and log.name == "stdout.log" and log.is_file()


def test_linked_runtime_scratch_rejected_before_transport(inputs, tmp_path, monkeypatch):
    plan, claim, materials = inputs
    original_snapshot = v2._snapshot

    def snapshot(materials, workspace):
        manifest = original_snapshot(materials, workspace)
        scratch = workspace.parent / "runtime_scratch"
        scratch.symlink_to(workspace.parent, target_is_directory=True)
        return manifest

    monkeypatch.setattr(v2, "_snapshot", snapshot)
    calls = []
    monkeypatch.setattr(v2, "run_command", lambda *a, **kw: calls.append(True))
    result = v2.execute_plans(
        [plan],
        [claim],
        materials,
        tmp_path / "out",
        config=v2.ExecutionConfig(refine_with_llm=False, max_attempts=0),
        runner=v2.docker_runner,
    )
    assert calls == []
    assert not result.claims[0].evidence
    assert "linked execution path" in result.ledger[0]["reason"]
