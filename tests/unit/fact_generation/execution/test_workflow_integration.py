"""Integration smoke tests for the packaged FactReview layout."""

from __future__ import annotations


def test_workflow_importable():
    from fact_generation.execution.graph import ExecutionOrchestrator

    assert callable(ExecutionOrchestrator)


def test_all_nodes_importable():
    from fact_generation.execution.nodes.finalize import finalize_node
    from fact_generation.execution.nodes.fix import fix_node
    from fact_generation.execution.nodes.judge import judge_node
    from fact_generation.execution.nodes.plan import plan_node
    from fact_generation.execution.nodes.prepare import prepare_node
    from fact_generation.execution.nodes.run import run_node

    for fn in [prepare_node, plan_node, run_node, judge_node, fix_node, finalize_node]:
        assert callable(fn)


def test_tools_importable():
    from fact_generation.execution.tools.alignment import run_alignment
    from fact_generation.execution.tools.baseline_checks import Baseline
    from fact_generation.execution.tools.metrics import compute_check
    from fact_generation.positioning.bibtex import lookup_bibtex
    from fact_generation.refcheck.refcheck import check_references

    assert callable(lookup_bibtex)
    assert callable(check_references)
    assert callable(run_alignment)
    assert callable(compute_check)
    assert Baseline(raw={}).checks == []


def test_orchestrator_accepts_new_flags():
    """The orchestrator must accept optional integration flags."""
    from fact_generation.execution.graph import ExecutionOrchestrator

    o = ExecutionOrchestrator(
        run_root="/tmp/test_run",
        enable_refcheck=True,
        enable_bibtex=True,
        paper_extracted_dir="/tmp/extracted",
        run_dir="/tmp/run",
    )
    assert o.enable_refcheck is True
    assert o.enable_bibtex is True
    assert o.paper_extracted_dir == "/tmp/extracted"
    assert o.run_dir == "/tmp/run"


def test_run_layout_uses_single_run_folder():
    from pathlib import Path

    from util.run_layout import build_run_dir, slugify_run_key

    assert slugify_run_key("CompGCN Paper") == "compgcn_paper"
    expected = Path("/tmp/runs").resolve() / "compgcn_paper_2026-04-25_120000"
    assert build_run_dir("/tmp/runs", "CompGCN Paper", "2026-04-25_120000") == expected


def test_configured_demo_uses_slugified_key(tmp_path, monkeypatch):
    from fact_generation.execution.nodes import prepare

    demo_dir = tmp_path / "demos" / "compgcn_paper"
    demo_dir.mkdir(parents=True)
    monkeypatch.setattr(prepare, "_repo_root", lambda: tmp_path)

    assert prepare._configured_demo_dir("CompGCN Paper") == demo_dir.resolve()


def test_archive_prior_current_dir_preserves_old_attempt(tmp_path):
    from fact_generation.execution.stage_runner import _archive_prior_current_dir

    stage_root = tmp_path / "stages" / "fact_generation" / "execution"
    current_dir = stage_root / "current"
    stale_metric = current_dir / "artifacts" / "metrics" / "old.json"
    stale_metric.parent.mkdir(parents=True)
    stale_metric.write_text("{}", encoding="utf-8")
    sibling_execution_json = stage_root / "execution.json"
    sibling_execution_json.write_text("{}", encoding="utf-8")

    _archive_prior_current_dir(stage_root=stage_root, current_dir=current_dir)

    assert current_dir.exists()
    assert not (current_dir / "artifacts" / "metrics" / "old.json").exists()
    archives = sorted(p for p in stage_root.iterdir() if p.is_dir() and p.name.startswith("current."))
    assert len(archives) == 1
    assert (archives[0] / "artifacts" / "metrics" / "old.json").exists()
    assert sibling_execution_json.exists()


def test_paper_image_tag_uses_slugified_key():
    from fact_generation.execution.tools.docker import _paper_image_tag

    image = _paper_image_tag(cfg={}, paper_key="CompGCN Paper", payload="same")

    assert image.startswith("factreview-paper:compgcn_paper-")
    assert " " not in image


def test_fix_node_terminates_at_max_attempts(tmp_path):
    """Once ``attempt > max_attempts`` the fix loop must mark the run failed
    and route to finalize, so the orchestrator can never spin forever."""
    from fact_generation.execution.graph import _route_after_fix
    from fact_generation.execution.nodes.fix import fix_node

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {
        "status": "running",
        "attempt": 2,  # Will be incremented to 3, exceeding max_attempts=2.
        "max_attempts": 2,
        "config": {"paper_root": str(tmp_path), "docker_enabled": False},
        "run": {"dir": str(run_dir), "logs_dir": str(run_dir / "logs"), "fixes_dir": str(run_dir / "fixes")},
        "history": [],
    }

    result = fix_node(state)

    assert result["status"] == "failed"
    assert result["attempt"] == 3
    history_kinds = [h.get("kind") for h in result.get("history", []) if isinstance(h, dict)]
    assert "fix_stop" in history_kinds
    assert _route_after_fix(result) == "finalize"
