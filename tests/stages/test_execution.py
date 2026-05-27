"""Execution stage tests.

Code execution requires a Docker daemon, so this stage's only test is gated
behind ``@pytest.mark.requires_docker`` and skipped by default. The real
execution path is covered by manual runs of ``demos/`` papers (see the plan
doc's verification section).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from fact_generation.execution.graph import _route_after_judge, is_budget_exhausted
from fact_generation.execution.nodes.finalize import finalize_node
from fact_generation.execution.nodes.fix import (
    _container_cwd_to_host,
    _host_dependency_installs_allowed,
    _install_missing_module_in_run_venv,
    _is_host_dependency_install_command,
    _is_rerun_failed_task_command,
    _missing_module_looks_local,
    _normalize_container_path_text,
    _normalize_llm_cmd_for_platform,
    _pip_package_for_module,
    _resolve_max_attempts,
    _rewrite_container_path_leaks,
    _to_shell,
)
from fact_generation.execution.nodes.plan import _is_runtime_pip_install_cmd, _merge_auto_baseline
from fact_generation.execution.nodes.prepare import (
    DownloadIncompleteError,
    DownloadLimitError,
    _anonymous_4open_get_bytes,
    _anonymous_4open_repo_id,
    _download_anonymous_4open_repo,
    _download_openreview_candidate_source,
    _download_openreview_supplementary,
    _download_url_bytes,
    _extended_windows_path,
    _extract_archive_bytes,
    _git_clone_timeout_sec,
    _infer_python_spec_from_repo,
    _normalize_shell_script_line_endings,
    _openreview_candidate_source_urls,
    _openreview_forum_id,
    _patch_api_placeholders_for_env,
    _remove_tree_best_effort,
    _source_dir_has_payload,
    _source_dir_looks_partial_clone,
)
from fact_generation.execution.nodes.run import (
    _disable_embedded_command_timeouts,
    _effective_task_timeout,
    _repo_container_pythonpath,
    _resolve_host_python_cmd,
    _semantic_metric_failure,
    _semantic_runtime_failure,
    run_node,
)
from fact_generation.execution.tools.alignment import run_alignment
from fact_generation.execution.tools.docker import (
    _collect_repo_requirements_text,
    _docker_build_args,
    _docker_daemon_unavailable,
    _docker_env_passthrough,
    _docker_include_notebook_requirements,
    _docker_proxy_env,
    _docker_run_user_args,
    _normalize_container_proxy,
    _paper_dockerfile_text,
    _paper_install_deps_py_text,
    _select_python_image,
    docker_ensure_paper_image,
    docker_run_paper_image,
)
from fact_generation.execution.tools.evidence_summary import (
    build_execution_evidence_summary,
    classify_run_failure,
    classify_task_failure,
)
from fact_generation.execution.tools.log_metrics import extract_metrics_from_text, write_task_metric_artifact
from fact_generation.execution.tools.metrics import compute_check
from fact_generation.execution.tools.paper_tables import extract_paper_metric_targets
from fact_generation.execution.tools.task_infer import (
    _apply_external_api_policy,
    _apply_missing_entrypoint_policy,
    _apply_mode_policy,
    _apply_static_import_policy,
    infer_tasks_heuristic,
)
from util.subprocess_runner import CommandResult, run_command


def test_paper_budget_is_disabled_unless_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.delenv("EXECUTION_ENABLE_PAPER_BUDGET", raising=False)
    state = {"config": {"paper_budget_sec": 1, "t_start_monotonic": time.monotonic() - 10}, "history": []}

    assert is_budget_exhausted(state) is False

    monkeypatch.setenv("EXECUTION_ENABLE_PAPER_BUDGET", "1")
    assert is_budget_exhausted(state) is True
    assert state["history"][-1]["kind"] == "budget_exhausted"


def test_disable_embedded_notebook_timeouts_when_task_timeout_disabled() -> None:
    cmd = [
        "python",
        "-m",
        "jupyter",
        "nbconvert",
        "--ExecutePreprocessor.timeout=7200",
        "--ExecutePreprocessor.timeout",
        "1800",
        "demo.ipynb",
    ]

    assert _disable_embedded_command_timeouts(cmd) == [
        "python",
        "-m",
        "jupyter",
        "nbconvert",
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.timeout",
        "-1",
        "demo.ipynb",
    ]


def test_extracts_generic_paper_metric_table(tmp_path) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "index.json").write_text(
        json.dumps([{"id": "table_1", "path_md": "table_1.md"}]),
        encoding="utf-8",
    )
    (tables_dir / "table_1.md").write_text(
        "\n".join(
            [
                "| Method | Dataset | Accuracy | F1 |",
                "| --- | --- | --- | --- |",
                "| Ours | CIFAR-10 | 94.2 | 93.5 |",
            ]
        ),
        encoding="utf-8",
    )

    targets = extract_paper_metric_targets(tables_dir)

    assert len(targets) == 1
    target = targets[0]
    assert target.dataset == "CIFAR-10"
    assert target.method == "Ours"
    assert target.metric_source == "generic_table"
    assert target.metrics == {"accuracy": 94.2, "f1": 93.5}


def test_extracts_metric_row_paper_table(tmp_path) -> None:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    (tables_dir / "index.json").write_text(
        json.dumps([{"id": "table_2", "path_md": "table_2.md"}]),
        encoding="utf-8",
    )
    (tables_dir / "table_2.md").write_text(
        "\n".join(
            [
                "| Metric | Baseline | Ours |",
                "| --- | --- | --- |",
                "| Accuracy | 91.0 | 94.2 |",
                "| F1 | 90.5 | 93.5 |",
            ]
        ),
        encoding="utf-8",
    )

    targets = extract_paper_metric_targets(tables_dir)
    ours = next(t for t in targets if t.method == "Ours")

    assert ours.metric_source == "metric_row_table"
    assert ours.metrics == {"accuracy": 94.2, "f1": 93.5}


def test_alignment_matches_nested_metric_json_against_paper_target(tmp_path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    metrics_dir = artifacts_dir / "text" / "execution_ours"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": "CIFAR-10",
                "method": "Ours",
                "metrics": {"accuracy": 0.943, "f1": 0.934},
            }
        ),
        encoding="utf-8",
    )

    result = run_alignment(
        cfg={},
        run_dir=tmp_path,
        artifacts_dir=artifacts_dir,
        paper_extracted_tables_dir=tmp_path / "missing_tables",
        paper_metric_targets=[
            {
                "paper_table_id": "table_1",
                "paper_table_md_path": "table_1.md",
                "dataset": "CIFAR-10",
                "method": "Ours",
                "scoring_function": "",
                "metrics": {"accuracy": 94.2, "f1": 93.5},
            }
        ],
    )

    assert result.extracted_targets == 1
    assert result.matched == 1
    assert result.passed == 1
    assert result.failed == 0
    assert result.matches[0]["run_metrics_file"] == "text/execution_ours/metrics.json"
    assert len(result.comparisons) == 2
    accuracy_row = next(r for r in result.comparisons if r["metric"] == "accuracy")
    assert accuracy_row["dataset"] == "CIFAR-10"
    assert accuracy_row["paper_value"] == pytest.approx(0.942)
    assert accuracy_row["observed_value"] == pytest.approx(0.943)
    assert accuracy_row["delta"] == pytest.approx(0.001)
    assert accuracy_row["passed"] is True


def test_log_metrics_writes_standard_metric_artifact(tmp_path) -> None:
    rel = write_task_metric_artifact(
        artifacts_dir=tmp_path / "artifacts",
        task_id="eval_ours",
        task={
            "family": "eval",
            "dataset": "CIFAR-10",
            "method": "Ours",
            "expected_metrics": {"accuracy": 94.2, "f1": 93.5},
            "claims": ["Ours reaches 94.2 accuracy and 93.5 F1 on CIFAR-10."],
        },
        stdout="Epoch done\naccuracy: 94.3\nf1 = 93.4\n",
        stderr="",
    )

    assert rel == "metrics/eval_ours_metrics.json"
    payload = json.loads((tmp_path / "artifacts" / rel).read_text(encoding="utf-8"))
    assert payload["dataset"] == "CIFAR-10"
    assert payload["metrics"] == {"accuracy": 94.3, "f1": 93.4}


def test_finalize_publishes_alignment_comparisons_to_report_and_facts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    artifacts_dir = run_dir / "artifacts"
    alignment_dir = artifacts_dir / "alignment"
    alignment_dir.mkdir(parents=True)
    comparison = {
        "paper_key": "FB15k-237 / TinyMethod / mrr",
        "observed_key": "metrics/eval_tinymethod_metrics.json / mrr",
        "metric": "mrr",
        "dataset": "FB15k-237",
        "split": "",
        "paper_value": 0.355,
        "observed_value": 0.200,
        "paper_value_raw": 0.355,
        "observed_value_raw": 0.200,
        "delta": -0.155,
        "delta_abs": 0.155,
        "delta_pct": 43.66,
        "tolerance": 0.01,
        "within_tolerance": False,
        "passed": False,
        "direction": "higher_is_better",
        "run_metrics_file": "metrics/eval_tinymethod_metrics.json",
        "paper_table_id": "table_1",
        "paper_table_md_path": "table_1.md",
        "paper_row_label": "TinyMethod",
        "paper_scoring_function": "",
    }
    (alignment_dir / "alignment.json").write_text(
        json.dumps({"comparisons": [comparison], "matches": [], "critiques": []}),
        encoding="utf-8",
    )
    (alignment_dir / "alignment.md").write_text("# alignment\n", encoding="utf-8")

    state = {
        "config": {"paper_key": "tinymethod"},
        "run": {
            "id": "demo_run",
            "dir": str(run_dir),
            "artifacts_dir": str(artifacts_dir),
        },
        "status": "running",
        "attempt": 0,
        "run_result": {
            "success": True,
            "tasks": [
                {
                    "id": "eval_tinymethod",
                    "family": "eval",
                    "dataset": "FB15k-237",
                    "success": True,
                    "metric_artifact": "metrics/eval_tinymethod_metrics.json",
                }
            ],
        },
        "judge": {
            "passed": False,
            "results": [
                {
                    "type": "paper_metric_alignment",
                    "extracted_targets": 1,
                    "matched": 1,
                    "passed_n": 0,
                    "failed_n": 1,
                    "comparisons_n": 1,
                    "alignment_artifact": "alignment/alignment.json",
                }
            ],
        },
    }

    finalize_node(state)

    assert state["status"] == "deviated"
    report = (run_dir / "reports" / "demo_run.md").read_text(encoding="utf-8")
    assert "- Final status: `deviated`" in report
    assert "Paper-vs-Run Metric Comparison" in report
    assert "| FB15k-237 | mrr | 0.355 | 0.2 | -0.155 | 0.01 | FAIL |" in report
    assert "paper=0.355, reproduced=0.2, delta=-0.155 (outside tolerance)" in report

    facts = json.loads((run_dir / "review_pack" / "facts.json").read_text(encoding="utf-8"))
    assert facts["status"] == "deviated"
    assert facts["alignment_summary"]["comparisons"] == 1
    assert facts["alignment_comparisons"][0]["paper_value"] == pytest.approx(0.355)
    assert facts["alignment_comparisons"][0]["observed_value"] == pytest.approx(0.2)

    evidence = json.loads((artifacts_dir / "execution_evidence.json").read_text(encoding="utf-8"))
    assert evidence["funnel"]["alignment_comparisons"] == 1


def test_route_after_judge_finalizes_matched_metric_mismatch() -> None:
    mismatch_state = {
        "status": "running",
        "judge": {
            "passed": False,
            "results": [
                {
                    "type": "paper_metric_alignment",
                    "extracted_targets": 1,
                    "matched": 1,
                    "failed_n": 1,
                }
            ],
        },
    }
    unmatched_state = {
        "status": "running",
        "judge": {
            "passed": False,
            "results": [
                {
                    "type": "paper_metric_alignment",
                    "extracted_targets": 1,
                    "matched": 0,
                    "failed_n": 0,
                }
            ],
        },
    }

    assert _route_after_judge(mismatch_state) == "finalize"
    assert _route_after_judge(unmatched_state) == "fix"


def test_extract_metrics_from_json_log_line() -> None:
    metrics = extract_metrics_from_text(
        'prefix\n{"metrics": {"acc": 0.943, "loss": 0.12}}\n',
        expected_metrics={"accuracy": 94.2},
    )

    assert metrics["accuracy"] == 0.943
    assert metrics["loss"] == 0.12


def test_extract_metrics_from_plain_log_without_expected_metrics() -> None:
    metrics = extract_metrics_from_text("final report\nAccuracy: 94.3\nF1 = 93.4\n")

    assert metrics["accuracy"] == 94.3
    assert metrics["f1"] == 93.4


def test_plan_auto_baseline_generates_checks_from_expected_metrics(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tasks_p = tmp_path / "tasks.json"
    baseline_p = tmp_path / "baseline.json"
    baseline_p.write_text("{}", encoding="utf-8")
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "eval_ours",
                    "artifact_paths": ["metrics/eval_ours.json"],
                    "expected_metrics": {"accuracy": 0.942},
                }
            ]
        ),
        encoding="utf-8",
    )

    baseline = _merge_auto_baseline(
        baseline_p=baseline_p,
        tasks_p=tasks_p,
        paper_metric_targets=[
            {
                "paper_table_id": "table_1",
                "dataset": "CIFAR-10",
                "method": "Ours",
                "metrics": {"accuracy": 0.942},
            }
        ],
        run_dir=run_dir,
    )

    checks = baseline["checks"]
    assert baseline["paper_metric_targets"][0]["dataset"] == "CIFAR-10"
    assert any(chk["type"] == "file_exists" for chk in checks)
    assert any(
        chk["type"] == "json_value"
        and chk["path"] == "metrics/eval_ours_metrics.json"
        and chk["json_path"] == ["metrics", "accuracy"]
        for chk in checks
    )


def test_json_value_check_handles_metric_scaling_and_nested_paths(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / "metrics").mkdir(parents=True)
    (artifacts / "metrics" / "eval.json").write_text(
        json.dumps({"metrics": {"accuracy": 0.943}}),
        encoding="utf-8",
    )

    result = compute_check(
        str(artifacts),
        {
            "type": "json_value",
            "path": "metrics/eval.json",
            "json_path": ["metrics", "accuracy"],
            "metric": "accuracy",
            "expected": 94.2,
            "tolerance": 0.02,
        },
    )

    assert result["passed"] is True


def test_run_node_extracts_metrics_from_task_logs(tmp_path) -> None:
    paper_root = tmp_path / "repo"
    paper_root.mkdir()
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "eval_ours",
                    "family": "eval",
                    "cwd": "{paper_root}",
                    "cmd": ["python", "-c", "print('accuracy: 94.3')"],
                    "timeout_sec": 30,
                    "dataset": "CIFAR-10",
                    "method": "Ours",
                    "expected_metrics": {"accuracy": 94.2},
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    task_result = out["run_result"]["tasks"][0]
    assert task_result["metric_artifact"] == "metrics/eval_ours_metrics.json"
    metrics = json.loads((run_dir / "artifacts" / task_result["metric_artifact"]).read_text())
    assert metrics["dataset"] == "CIFAR-10"
    assert metrics["metrics"]["accuracy"] == 94.3


def test_run_node_adds_common_src_dir_to_pythonpath(tmp_path) -> None:
    paper_root = tmp_path / "repo"
    (paper_root / "src" / "agent").mkdir(parents=True)
    (paper_root / "src" / "agent" / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "import_src",
                    "family": "smoke",
                    "cwd": "{paper_root}",
                    "cmd": ["python", "-c", "import agent; print(agent.VALUE)"],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    task_result = out["run_result"]["tasks"][0]
    assert task_result["success"] is True
    assert Path(task_result["logs"]["stdout"]).read_text(encoding="utf-8").strip() == "7"


def test_run_node_uses_utf8_python_stdio_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    paper_root = tmp_path / "repo"
    paper_root.mkdir()
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "unicode_logs",
                    "family": "smoke",
                    "cwd": "{paper_root}",
                    "cmd": [
                        "python",
                        "-c",
                        (
                            "import os; "
                            "print(os.environ.get('PYTHONIOENCODING')); "
                            "print(os.environ.get('PYTHONUTF8')); "
                            "print('✅ ok')"
                        ),
                    ],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    task_result = out["run_result"]["tasks"][0]
    lines = Path(task_result["logs"]["stdout"]).read_text(encoding="utf-8").splitlines()
    assert task_result["success"] is True
    assert lines == ["utf-8", "1", "✅ ok"]


def test_run_node_prepends_host_venv_to_path_for_shell_tasks(tmp_path, monkeypatch) -> None:
    paper_root = tmp_path / "repo"
    paper_root.mkdir()
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    venv_dir = tmp_path / "short-venv"
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    run_dir.mkdir()
    (run_dir / ".host_venv_path").write_text(str(venv_dir), encoding="utf-8")
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "shell_python",
                    "family": "smoke",
                    "cwd": "{paper_root}",
                    "cmd": ["bash", "-lc", "python -c 'print(1)'"],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_command(cmd: list[str], cwd: str, timeout_sec: int | None = 3600, env=None) -> CommandResult:
        captured["cmd"] = cmd
        captured["env"] = dict(env or {})
        return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="1\n", stderr="", duration_sec=0.01)

    monkeypatch.setattr("fact_generation.execution.nodes.run.run_command", fake_run_command)
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    assert out["run_result"]["tasks"][0]["success"] is True
    env = captured["env"]
    assert isinstance(env, dict)
    path_head = str(env["PATH"]).split(os.pathsep)[0]
    assert Path(path_head) == venv_python.parent.resolve()
    assert env["VIRTUAL_ENV"] == str(venv_dir.resolve())


def test_run_node_adds_run_local_jupyter_path_for_host_tasks(tmp_path, monkeypatch) -> None:
    paper_root = tmp_path / "repo"
    paper_root.mkdir()
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    kernel_prefix = run_dir / "jupyter"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "notebook",
                    "family": "smoke",
                    "cwd": "{paper_root}",
                    "cmd": ["jupyter", "nbconvert", "--execute", "demo.ipynb"],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_command(cmd: list[str], cwd: str, timeout_sec: int | None = 3600, env=None) -> CommandResult:
        captured["env"] = dict(env or {})
        return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="", stderr="", duration_sec=0.01)

    monkeypatch.setattr("fact_generation.execution.nodes.run.run_command", fake_run_command)
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
            "jupyter_kernel_prefix": str(kernel_prefix),
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    assert out["run_result"]["tasks"][0]["success"] is True
    env = captured["env"]
    assert isinstance(env, dict)
    assert str(env["JUPYTER_PATH"]).split(os.pathsep)[0] == str((kernel_prefix / "share" / "jupyter").resolve())


def test_run_node_uses_extra_pythonpath_dirs_for_deep_repo_packages(tmp_path) -> None:
    paper_root = tmp_path / "repo"
    package = paper_root / "experiments" / "lib" / "deepagent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 11\n", encoding="utf-8")
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "import_deep",
                    "family": "smoke",
                    "cwd": "{paper_root}",
                    "cmd": ["python", "-c", "import deepagent; print(deepagent.VALUE)"],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
            "extra_pythonpath_dirs": ["experiments/lib"],
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    task_result = out["run_result"]["tasks"][0]
    assert task_result["success"] is True
    assert Path(task_result["logs"]["stdout"]).read_text(encoding="utf-8").strip() == "11"


def test_docker_pythonpath_includes_extra_repo_dirs(tmp_path) -> None:
    paper_root = tmp_path / "repo"
    (paper_root / "experiments" / "lib").mkdir(parents=True)

    value = _repo_container_pythonpath(paper_root, {"extra_pythonpath_dirs": ["experiments/lib"]})

    assert value == "/app:/app/experiments/lib"


def test_run_node_accepts_eval_metric_without_expected_metrics(tmp_path) -> None:
    paper_root = tmp_path / "repo"
    paper_root.mkdir()
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "eval_readme",
                    "family": "eval",
                    "cwd": "{paper_root}",
                    "cmd": ["python", "-c", "print('Accuracy: 94.3')"],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    task_result = out["run_result"]["tasks"][0]
    assert task_result["success"] is True
    assert task_result["metric_artifact"] == "metrics/eval_readme_metrics.json"


def test_run_node_marks_eval_without_metrics_inconclusive(tmp_path) -> None:
    paper_root = tmp_path / "repo"
    paper_root.mkdir()
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "eval_readme",
                    "family": "eval",
                    "cwd": "{paper_root}",
                    "cmd": [
                        "python",
                        "-c",
                        "print('Accumulated Accuracy Scores:\\n\\n-------------------------------------------------- Finished')",
                    ],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    assert out["status"] == "inconclusive"
    assert out["run_result"]["success"] is False
    assert out["run_result"]["inconclusive"] is True
    assert out["run_result"]["semantic_failure"] == "semantic_no_metrics"
    assert out["run_result"]["tasks"][0]["semantic_failure"] == "semantic_no_metrics"


def test_run_node_archives_failure_artifacts_before_fix(tmp_path) -> None:
    paper_root = tmp_path / "repo"
    paper_root.mkdir()
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "smoke_imports",
                    "family": "smoke",
                    "cwd": "{paper_root}",
                    "cmd": [
                        "python",
                        "-c",
                        (
                            "import pathlib; "
                            "pathlib.Path('metrics').mkdir(exist_ok=True); "
                            "pathlib.Path('metrics/smoke_imports_metrics.json').write_text('{\"success\": false}'); "
                            "raise SystemExit(1)"
                        ),
                    ],
                    "timeout_sec": 30,
                    "artifact_paths": ["metrics/smoke_imports_metrics.json"],
                    "metric_artifact_path": "metrics/smoke_imports_metrics.json",
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    task = out["run_result"]["tasks"][0]
    assert out["status"] == "failed"
    assert task["metric_artifact"] == "metrics/smoke_imports_metrics.json"
    assert (run_dir / "artifacts" / "metrics" / "smoke_imports_metrics.json").exists()
    assert "artifacts_archived" in (run_dir / "issues.jsonl").read_text(encoding="utf-8")


def test_run_node_accepts_smoke_task_even_if_llm_labeled_eval(tmp_path) -> None:
    paper_root = tmp_path / "repo"
    paper_root.mkdir()
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "smoke_list_entrypoints",
                    "family": "eval",
                    "cwd": "{paper_root}",
                    "cmd": ["python", "-c", "print('usage: train.py [--help]')"],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    assert out["run_result"]["success"] is True
    assert out["run_result"]["tasks"][0]["success"] is True
    assert "semantic_failure" not in out["run_result"]["tasks"][0]


def test_run_node_adds_paper_root_to_pythonpath_for_nested_scripts(tmp_path) -> None:
    paper_root = tmp_path / "repo"
    package = paper_root / "evaluate"
    package.mkdir(parents=True)
    (package / "_utils.py").write_text("VALUE = 42\n", encoding="utf-8")
    (package / "evaluate_code.py").write_text(
        "from evaluate._utils import VALUE\nprint(f'value={VALUE}')\n",
        encoding="utf-8",
    )
    tasks_p = tmp_path / "tasks.json"
    run_dir = tmp_path / "run"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "smoke_evaluate_code_help",
                    "family": "smoke",
                    "cwd": "{paper_root}",
                    "cmd": ["python", "evaluate/evaluate_code.py"],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    assert out["run_result"]["success"] is True
    assert out["run_result"]["tasks"][0]["success"] is True


def test_graph_exit_status_preserves_run_inconclusive() -> None:
    from fact_generation.execution.graph import _compute_exit_status

    assert _compute_exit_status({"status": "inconclusive", "run_result": {"success": False}}) == "inconclusive"


def test_execution_evidence_summary_classifies_metric_failure(tmp_path) -> None:
    task = {
        "id": "eval_readme",
        "family": "eval",
        "success": False,
        "semantic_failure": "semantic_no_metrics",
        "duration_sec": 12.5,
    }
    state = {
        "status": "inconclusive",
        "attempt": 1,
        "config": {"paper_key": "demo"},
        "run_result": {
            "success": False,
            "inconclusive": True,
            "semantic_failure": "semantic_no_metrics",
            "tasks": [task],
        },
        "judge": {},
        "node_timings": {"run": [{"duration_sec": 12.5, "attempt": 0}]},
    }

    summary = build_execution_evidence_summary(
        state=state,
        run_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )

    assert classify_task_failure(task) == "metric"
    assert classify_run_failure(state["run_result"], state["judge"]) == "metric"
    assert summary["failure_stage"] == "metric"
    assert summary["failure_stage_counts"]["metric"] == 1
    assert summary["cost"]["node_duration_sec"]["run"] == 12.5


def test_execution_evidence_summary_classifies_prepare_download_failure_as_access(tmp_path) -> None:
    (tmp_path / "issues.jsonl").write_text(
        json.dumps(
            {
                "ts": 1.0,
                "kind": "prepare_error",
                "data": {
                    "error": "anonymous_4open_download_failed: RuntimeError: anonymous_4open_no_files_downloaded: zip_http_404",
                    "repo_url": "https://anonymous.4open.science/r/context-aware-clustering-E90C",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = {
        "status": "failed",
        "attempt": 0,
        "config": {"paper_key": "cactus"},
        "run_result": {},
        "judge": {},
    }

    summary = build_execution_evidence_summary(
        state=state,
        run_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )

    assert summary["failure_stage"] == "access"
    assert summary["failure_stage_counts"]["access"] == 1


def test_execution_evidence_summary_synthesizes_failed_task_row(tmp_path) -> None:
    state = {
        "status": "failed",
        "attempt": 1,
        "config": {"paper_key": "fmp"},
        "run_result": {
            "success": False,
            "failed_task": "check_python_and_core_deps",
            "returncode": 1,
            "semantic_failure": "python_traceback_in_output",
            "stderr_tail": "ModuleNotFoundError: No module named 'torch'",
        },
        "judge": {},
    }

    summary = build_execution_evidence_summary(
        state=state,
        run_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )

    assert summary["funnel"]["tasks_total"] == 1
    assert summary["funnel"]["tasks_failed"] == 1
    assert summary["tasks"][0]["id"] == "check_python_and_core_deps"
    assert summary["failure_stage"] == "environment"
    assert summary["failure_stage_counts"]["environment"] == 1


def test_execution_evidence_summary_preserves_skipped_task_reason(tmp_path) -> None:
    task = {
        "id": "reproduce_notebook_demo",
        "family": "reproduce",
        "success": True,
        "skipped": True,
        "disabled_reason": "external_api_or_model_server_required",
        "requires_external_api": True,
        "static_import_issues": {"module": "demo", "missing_names": ["BasePolicy"]},
    }
    state = {
        "status": "running",
        "attempt": 0,
        "config": {"paper_key": "api_notebook"},
        "run_result": {"success": True, "tasks": [task]},
        "judge": {},
    }

    summary = build_execution_evidence_summary(
        state=state,
        run_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )

    row = summary["tasks"][0]
    assert row["status"] == "skipped"
    assert row["disabled_reason"] == "external_api_or_model_server_required"
    assert row["requires_external_api"] is True
    assert row["static_import_issues"]["missing_names"] == ["BasePolicy"]


def test_execution_evidence_summary_includes_planned_tasks_after_early_failure(tmp_path) -> None:
    (tmp_path / "tasks.yaml").write_text(
        json.dumps(
            [
                {"id": "smoke_import", "family": "smoke", "enabled": True},
                {
                    "id": "full_api_eval",
                    "family": "reproduce",
                    "enabled": False,
                    "disabled_reason": "external_api_or_model_server_required",
                    "requires_external_api": True,
                },
                {"id": "full_cpu_eval", "family": "eval", "enabled": True},
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "status": "failed",
        "attempt": 1,
        "config": {"paper_key": "early_fail"},
        "run_result": {
            "success": False,
            "failed_task": "smoke_import",
            "returncode": 1,
            "semantic_failure": "python_traceback_in_output",
            "stderr_tail": "ModuleNotFoundError: No module named 'gurobipy'",
        },
        "judge": {},
    }

    summary = build_execution_evidence_summary(
        state=state,
        run_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )
    rows = {task["id"]: task for task in summary["tasks"]}

    assert summary["funnel"]["tasks_total"] == 3
    assert summary["funnel"]["tasks_failed"] == 1
    assert summary["funnel"]["tasks_skipped"] == 1
    assert summary["funnel"]["tasks_not_run"] == 1
    assert rows["full_api_eval"]["status"] == "skipped"
    assert rows["full_api_eval"]["requires_external_api"] is True
    assert rows["full_cpu_eval"]["status"] == "not_run"


def test_runtime_traceback_output_is_treated_as_failure() -> None:
    reason = _semantic_runtime_failure(
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "/app/main.py", line 1, in <module>\n'
            "ModuleNotFoundError: No module named 'numpy'\n"
        ),
    )

    assert reason == "python_traceback_in_output"


def test_semantic_metric_failure_allows_smoke_evidence_json_without_numeric_metric() -> None:
    task = {
        "id": "import_smoke",
        "family": "smoke",
        "metric_artifact_path": "artifacts/metrics/import_smoke_metrics.json",
    }

    reason = _semantic_metric_failure(
        task=task,
        task_id="import_smoke",
        stdout='{"status": "ok", "imports": {"iemm.core": true}}\n',
        stderr="",
        metric_artifact="",
    )

    assert reason == ""


def test_semantic_metric_failure_requires_metrics_for_eval_contract() -> None:
    task = {
        "id": "eval_model",
        "family": "eval",
        "metric_artifact_path": "metrics/eval_model_metrics.json",
    }

    reason = _semantic_metric_failure(
        task=task,
        task_id="eval_model",
        stdout='{"status": "ok"}\n',
        stderr="",
        metric_artifact="",
    )

    assert reason == "semantic_no_metrics"


def test_missing_dotted_module_maps_to_root_package() -> None:
    assert _pip_package_for_module("evaluate._utils") == "evaluate"
    assert _pip_package_for_module("sklearn.metrics") == "scikit-learn"


def test_missing_module_local_path_is_not_treated_as_pypi_package(tmp_path) -> None:
    (tmp_path / "evaluate").mkdir()
    (tmp_path / "evaluate" / "_utils.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert _missing_module_looks_local(str(tmp_path), "evaluate._utils")
    assert not _missing_module_looks_local(str(tmp_path), "gurobipy")


def test_masked_startup_failure_output_is_treated_as_failure() -> None:
    reason = _semantic_runtime_failure(
        stdout=(
            "All trained models found! Can skip training.\n"
            "Failed to load MIMIC data: preprocessed_mimic_data.pkl missing\n"
            "System startup failed\n"
        ),
        stderr="",
    )

    assert reason == "semantic_system_startup_failed"


def test_run_result_carries_semantic_failure(tmp_path) -> None:
    paper_root = tmp_path / "paper"
    paper_root.mkdir()
    run_dir = tmp_path / "run"
    tasks_p = tmp_path / "tasks.json"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "masked_failure",
                    "cwd": "{paper_root}",
                    "cmd": ["python", "-c", "print('System startup failed')"],
                    "timeout_sec": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(paper_root),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)

    assert out["status"] == "failed"
    assert out["run_result"]["returncode"] == 0
    assert out["run_result"]["semantic_failure"] == "semantic_system_startup_failed"


def test_no_docker_python_tasks_use_current_interpreter() -> None:
    resolved = _resolve_host_python_cmd(["python", "-V"])

    assert resolved[0] == sys.executable
    assert resolved[1:] == ["-V"]


def test_no_docker_python_tasks_prefer_run_local_venv(tmp_path) -> None:
    venv_python = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    resolved = _resolve_host_python_cmd(["python", "-V"], run_dir=tmp_path)

    assert resolved == [str(venv_python), "-V"]


def test_no_docker_python_tasks_prefer_marked_short_venv(tmp_path) -> None:
    short_venv = tmp_path / "fv"
    venv_python = short_venv / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    (tmp_path / ".host_venv_path").write_text(str(short_venv), encoding="utf-8")

    resolved = _resolve_host_python_cmd(["python", "-V"], run_dir=tmp_path)

    assert resolved == [str(venv_python), "-V"]


def test_no_docker_uv_run_python_falls_back_when_uv_missing(monkeypatch) -> None:
    monkeypatch.setattr("fact_generation.execution.nodes.run.shutil.which", lambda name: None)

    resolved = _resolve_host_python_cmd(["uv", "run", "python", "-c", "print('ok')"])

    assert resolved == [sys.executable, "-c", "print('ok')"]


def test_no_docker_host_venv_install_prefers_surgical_package(tmp_path, monkeypatch) -> None:
    paper_root = tmp_path / "paper"
    paper_root.mkdir()
    (paper_root / "pyproject.toml").write_text("[project]\nname = 'paper'\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    short_root = tmp_path / "short_venvs"
    calls: list[list[str]] = []

    def fake_run_command(cmd: list[str], cwd: str, timeout_sec: int | None = 3600, env=None) -> CommandResult:
        calls.append(cmd)
        return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="ok", stderr="", duration_sec=0.01)

    monkeypatch.setattr("fact_generation.execution.nodes.fix.run_command", fake_run_command)

    ok = _install_missing_module_in_run_venv(
        cfg={"host_venv_dir": str(short_root)},
        run_dir=run_dir,
        logs_dir=logs_dir,
        paper_root=str(paper_root),
        module="litellm",
        attempt=1,
    )

    venv_dir = Path((run_dir / ".host_venv_path").read_text(encoding="utf-8").strip())
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    assert ok is True
    assert venv_dir.parent == short_root
    assert [sys.executable, "-m", "venv", str(venv_dir)] in calls
    assert [str(venv_python), "-m", "pip", "install", "litellm"] in calls
    assert [str(venv_python), "-m", "pip", "install", "-e", "."] not in calls


def test_no_docker_host_venv_install_follows_nested_verify_missing_module(tmp_path, monkeypatch) -> None:
    paper_root = tmp_path / "paper"
    paper_root.mkdir()
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    calls: list[list[str]] = []
    verify_count = 0

    def fake_run_command(cmd: list[str], cwd: str, timeout_sec: int | None = 3600, env=None) -> CommandResult:
        nonlocal verify_count
        calls.append(cmd)
        shell = " ".join(str(x) for x in cmd)
        if "-c import dgl" in shell:
            verify_count += 1
            if verify_count == 1:
                return CommandResult(
                    cmd=cmd,
                    cwd=cwd,
                    returncode=1,
                    stdout="",
                    stderr="ModuleNotFoundError: No module named 'packaging'",
                    duration_sec=0.01,
                )
        return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="ok", stderr="", duration_sec=0.01)

    monkeypatch.setattr("fact_generation.execution.nodes.fix.run_command", fake_run_command)

    ok = _install_missing_module_in_run_venv(
        cfg={},
        run_dir=run_dir,
        logs_dir=logs_dir,
        paper_root=str(paper_root),
        module="dgl",
        attempt=1,
    )

    venv_python = run_dir / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    assert ok is True
    assert [str(venv_python), "-m", "pip", "install", "dgl"] in calls
    assert [str(venv_python), "-m", "pip", "install", "packaging"] in calls


def test_no_docker_host_venv_install_repairs_dgl_graphbolt_torch_pin(tmp_path, monkeypatch) -> None:
    paper_root = tmp_path / "paper"
    graphbolt = paper_root / "site-packages" / "dgl" / "graphbolt"
    graphbolt.mkdir(parents=True)
    for version in ["2.1.0", "2.2.2", "2.3.0"]:
        (graphbolt / f"graphbolt_pytorch_{version}.dll").write_text("", encoding="utf-8")
    missing_dll = graphbolt / "graphbolt_pytorch_2.12.0.dll"
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    calls: list[list[str]] = []
    verify_count = 0

    def fake_run_command(cmd: list[str], cwd: str, timeout_sec: int | None = 3600, env=None) -> CommandResult:
        nonlocal verify_count
        calls.append(cmd)
        shell = " ".join(str(x) for x in cmd)
        if "-c import dgl" in shell:
            verify_count += 1
            if verify_count == 1:
                return CommandResult(
                    cmd=cmd,
                    cwd=cwd,
                    returncode=1,
                    stdout="",
                    stderr=f"FileNotFoundError: Cannot find DGL C++ graphbolt library at {missing_dll}",
                    duration_sec=0.01,
                )
        return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="ok", stderr="", duration_sec=0.01)

    monkeypatch.setattr("fact_generation.execution.nodes.fix.run_command", fake_run_command)

    ok = _install_missing_module_in_run_venv(
        cfg={},
        run_dir=run_dir,
        logs_dir=logs_dir,
        paper_root=str(paper_root),
        module="dgl",
        attempt=1,
    )

    venv_python = run_dir / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    assert ok is True
    assert [str(venv_python), "-m", "pip", "install", "torch==2.3.0"] in calls
    issues = (run_dir / "issues.jsonl").read_text(encoding="utf-8")
    assert "fix_host_venv_compat_package" in issues


def test_no_docker_host_venv_installs_pyg_native_package_from_torch_wheel_index(tmp_path, monkeypatch) -> None:
    paper_root = tmp_path / "paper"
    paper_root.mkdir()
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    calls: list[list[str]] = []

    def fake_run_command(cmd: list[str], cwd: str, timeout_sec: int | None = 3600, env=None) -> CommandResult:
        calls.append(cmd)
        shell = " ".join(str(x) for x in cmd)
        if "import torch" in shell and "torch.version" in shell:
            return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="2.3.0\n\n", stderr="", duration_sec=0.01)
        return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="ok", stderr="", duration_sec=0.01)

    monkeypatch.setattr("fact_generation.execution.nodes.fix.run_command", fake_run_command)

    ok = _install_missing_module_in_run_venv(
        cfg={},
        run_dir=run_dir,
        logs_dir=logs_dir,
        paper_root=str(paper_root),
        module="torch_sparse",
        attempt=1,
    )

    venv_python = run_dir / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    assert ok is True
    assert [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "torch-sparse",
        "-f",
        "https://data.pyg.org/whl/torch-2.3.0+cpu.html",
    ] in calls


def test_no_docker_host_venv_creates_case_variant_import_alias(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    paper_root = tmp_path / "paper"
    paper_root.mkdir()
    short_root = tmp_path / "venvs"
    calls: list[list[str]] = []
    alias_created = False

    def fake_run_command(cmd, cwd, timeout_sec=None, env=None):
        nonlocal alias_created
        calls.append(list(cmd))
        text = " ".join(str(part) for part in cmd)
        if len(cmd) >= 3 and cmd[1:3] == ["-m", "venv"]:
            venv_dir = Path(cmd[-1])
            venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / (
                "python.exe" if os.name == "nt" else "python"
            )
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")
            return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="", stderr="", duration_sec=0.01)
        if "-m pip install pyDOE2" in text:
            return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="installed pyDOE2", stderr="", duration_sec=0.01)
        if "alias_name='pyDOE'" in text or 'alias_name="pyDOE"' in text:
            assert "pyDOE2" in text
            alias_created = True
            return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="alias", stderr="", duration_sec=0.01)
        if "import pyDOE" in text or "importlib.import_module(mod)" in text:
            if alias_created:
                return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="module_ok", stderr="", duration_sec=0.01)
            return CommandResult(
                cmd=cmd,
                cwd=cwd,
                returncode=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'pyDOE'",
                duration_sec=0.01,
            )
        return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="", stderr="", duration_sec=0.01)

    monkeypatch.setattr("fact_generation.execution.nodes.fix.run_command", fake_run_command)

    ok = _install_missing_module_in_run_venv(
        cfg={"host_venv_dir": str(short_root)},
        run_dir=run_dir,
        logs_dir=logs_dir,
        paper_root=str(paper_root),
        module="pyDOE",
        context_modules=["pyDOE"],
        attempt=1,
    )

    assert ok is True
    assert alias_created is True
    issues = (run_dir / "issues.jsonl").read_text(encoding="utf-8")
    assert "fix_host_venv_case_variant_alias" in issues


def test_no_docker_host_venv_adds_imp_shim_for_pydoe2_on_python312(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    paper_root = tmp_path / "paper"
    paper_root.mkdir()
    calls: list[list[str]] = []
    alias_created = False
    imp_shim_created = False

    def fake_run_command(cmd, cwd, timeout_sec=None, env=None):
        nonlocal alias_created, imp_shim_created
        calls.append(list(cmd))
        text = " ".join(str(part) for part in cmd)
        if len(cmd) >= 3 and cmd[1:3] == ["-m", "venv"]:
            venv_dir = Path(cmd[-1])
            venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / (
                "python.exe" if os.name == "nt" else "python"
            )
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")
            return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="", stderr="", duration_sec=0.01)
        if "-m pip install pyDOE2" in text:
            return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="installed pyDOE2", stderr="", duration_sec=0.01)
        if "target_candidates=['pyDOE2', 'pydoe']" in text:
            alias_created = True
            return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="alias", stderr="", duration_sec=0.01)
        if "module='imp'" in text and "target.write_text" in text:
            imp_shim_created = True
            return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="shim", stderr="", duration_sec=0.01)
        if "import pyDOE" in text or "importlib.import_module(mod)" in text:
            if alias_created and imp_shim_created:
                return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="module_ok", stderr="", duration_sec=0.01)
            if alias_created:
                return CommandResult(
                    cmd=cmd,
                    cwd=cwd,
                    returncode=1,
                    stdout="",
                    stderr="ModuleNotFoundError: No module named 'imp'",
                    duration_sec=0.01,
                )
            return CommandResult(
                cmd=cmd,
                cwd=cwd,
                returncode=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'pyDOE'",
                duration_sec=0.01,
            )
        return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="", stderr="", duration_sec=0.01)

    monkeypatch.setattr("fact_generation.execution.nodes.fix.run_command", fake_run_command)

    ok = _install_missing_module_in_run_venv(
        cfg={},
        run_dir=run_dir,
        logs_dir=logs_dir,
        paper_root=str(paper_root),
        module="pyDOE",
        context_modules=["pyDOE"],
        attempt=1,
    )

    assert ok is True
    assert alias_created is True
    assert imp_shim_created is True
    issues = (run_dir / "issues.jsonl").read_text(encoding="utf-8")
    assert "fix_host_venv_stdlib_compat_shim" in issues


def test_no_docker_windows_bash_prefers_explicit_path(tmp_path, monkeypatch) -> None:
    fake_bash = tmp_path / "bash.exe"
    fake_bash.write_text("", encoding="utf-8")
    monkeypatch.setenv("EXECUTION_BASH_PATH", str(fake_bash))

    resolved = _resolve_host_python_cmd(["bash", "-lc", "echo ok"])

    if os.name == "nt":
        assert resolved == [str(fake_bash), "-lc", "echo ok"]
    else:
        assert resolved == ["bash", "-lc", "echo ok"]


def test_llm_fix_windows_bash_prefers_explicit_path(tmp_path, monkeypatch) -> None:
    fake_bash = tmp_path / "bash.exe"
    fake_bash.write_text("", encoding="utf-8")
    monkeypatch.setenv("EXECUTION_BASH_PATH", str(fake_bash))

    resolved = _normalize_llm_cmd_for_platform(["bash", "-lc", "mkdir -p metrics && echo ok"])

    if os.name == "nt":
        assert resolved == [str(fake_bash), "-lc", "mkdir -p metrics && echo ok"]
    else:
        assert resolved == ["bash", "-lc", "mkdir -p metrics && echo ok"]


def test_run_node_keeps_disabled_task_metadata(tmp_path) -> None:
    repo = tmp_path / "repo"
    run_dir = tmp_path / "run"
    repo.mkdir()
    tasks_p = tmp_path / "tasks.json"
    tasks_p.write_text(
        json.dumps(
            [
                {
                    "id": "api_notebook",
                    "family": "reproduce",
                    "enabled": False,
                    "disabled_reason": "external_api_or_model_server_required",
                    "requires_external_api": True,
                    "static_import_issues": {"module": "demo", "missing_names": ["Client"]},
                    "cmd": ["python", "-c", "print('should not run')"],
                }
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(repo),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)
    task = out["run_result"]["tasks"][0]

    assert task["skipped"] is True
    assert task["disabled_reason"] == "external_api_or_model_server_required"
    assert task["requires_external_api"] is True
    assert task["static_import_issues"]["module"] == "demo"


def test_run_node_failure_preserves_prior_task_results(tmp_path) -> None:
    repo = tmp_path / "repo"
    run_dir = tmp_path / "run"
    repo.mkdir()
    tasks_p = tmp_path / "tasks.json"
    tasks_p.write_text(
        json.dumps(
            [
                {"id": "env_smoke", "family": "smoke", "cmd": ["python", "-c", "print('ok')"]},
                {"id": "bad_import", "family": "smoke", "cmd": ["python", "-c", "import definitely_missing_pkg"]},
            ]
        ),
        encoding="utf-8",
    )
    state = {
        "config": {
            "paper_root": str(repo),
            "tasks_path": str(tasks_p),
            "docker_enabled": False,
        },
        "run": {
            "dir": str(run_dir),
            "logs_dir": str(run_dir / "logs"),
            "artifacts_dir": str(run_dir / "artifacts"),
        },
    }

    out = run_node(state)
    tasks = out["run_result"]["tasks"]

    assert out["status"] == "failed"
    assert [task["id"] for task in tasks] == ["env_smoke", "bad_import"]
    assert tasks[0]["success"] is True
    assert tasks[1]["success"] is False
    assert "ModuleNotFoundError" in tasks[1]["stderr_tail"]


def test_task_timeout_can_be_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DISABLE_TASK_TIMEOUT", "1")

    assert _effective_task_timeout({"timeout_sec": 1}, {}) == 0


def test_task_timeout_can_be_overridden_by_env(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_TASK_TIMEOUT_SEC", "0")

    assert _effective_task_timeout({"timeout_sec": 1}, {}) == 0


def test_run_command_reports_timeout_explicitly(tmp_path) -> None:
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=str(tmp_path),
        timeout_sec=1,
    )

    assert result.returncode == 124
    assert "TimeoutExpired" in result.stderr


def test_run_command_timeout_kills_windows_child_process_tree(tmp_path) -> None:
    if os.name != "nt":
        pytest.skip("Windows process-tree behavior")
    marker = tmp_path / "child_survived.txt"
    child = (
        "import pathlib, time; "
        "time.sleep(5); "
        f"pathlib.Path({str(marker)!r}).write_text('survived', encoding='utf-8')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(30)"
    )

    result = run_command([sys.executable, "-c", parent], cwd=str(tmp_path), timeout_sec=1)
    time.sleep(6)

    assert result.returncode == 124
    assert not marker.exists()


def test_run_command_zero_timeout_means_no_timeout(tmp_path) -> None:
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(0.1); print('ok')"],
        cwd=str(tmp_path),
        timeout_sec=0,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_docker_build_args_include_pip_index_and_extra_packages(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DOCKER_PIP_INDEX_URL", "https://mirror.example/simple")
    monkeypatch.setenv("EXECUTION_DOCKER_PIP_TRUSTED_HOST", "mirror.example")
    monkeypatch.setenv("EXECUTION_DOCKER_HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("EXECUTION_DOCKER_HTTPS_PROXY", "http://localhost:7897")

    args = _docker_build_args({"docker_extra_pip_packages": "numpy scikit-learn"})

    joined = " ".join(args)
    assert "--build-arg PIP_INDEX_URL=https://mirror.example/simple" in joined
    assert "--build-arg PIP_TRUSTED_HOST=mirror.example" in joined
    assert "--build-arg EXECUTION_DOCKER_EXTRA_PIP_PACKAGES=numpy scikit-learn" in joined
    assert "--build-arg HTTP_PROXY=http://host.docker.internal:7897" in joined
    assert "--build-arg HTTPS_PROXY=http://host.docker.internal:7897" in joined


def test_docker_daemon_unavailable_detects_windows_engine_pipe() -> None:
    result = CommandResult(
        cmd=["docker", "image", "inspect", "paper:test"],
        cwd=".",
        returncode=1,
        stdout="",
        stderr="open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.",
        duration_sec=0.1,
    )

    assert _docker_daemon_unavailable(result)


def test_docker_ensure_paper_image_fails_fast_when_daemon_unavailable(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run_command(cmd: list[str], cwd: str, timeout_sec: int | None = 3600, env=None) -> CommandResult:
        calls.append(cmd)
        return CommandResult(
            cmd=cmd,
            cwd=cwd,
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
            duration_sec=0.1,
        )

    monkeypatch.setattr("fact_generation.execution.tools.docker.run_command", fake_run_command)
    ok, detail = docker_ensure_paper_image({}, paper_key="demo", paper_root_host=str(tmp_path), python_spec="3.11")

    assert ok is False
    assert detail.startswith("docker_daemon_unavailable:")
    assert calls
    assert not any("build" in cmd for cmd in calls)


def test_docker_proxy_env_ignores_stale_inherited_loopback_proxy(monkeypatch) -> None:
    monkeypatch.setattr("fact_generation.execution.tools.docker._docker_info_field", lambda field: "")
    monkeypatch.delenv("EXECUTION_DOCKER_HTTP_PROXY", raising=False)
    monkeypatch.delenv("EXECUTION_DOCKER_HTTPS_PROXY", raising=False)
    monkeypatch.delenv("EXECUTION_DOCKER_NO_PROXY", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:1")
    monkeypatch.delenv("NO_PROXY", raising=False)

    env = _docker_proxy_env({})

    assert "HTTP_PROXY" not in env
    assert "HTTPS_PROXY" not in env


def test_docker_runtime_injects_host_proxy(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EXECUTION_DOCKER_HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("EXECUTION_DOCKER_HTTPS_PROXY", "http://localhost:7897")

    cmd = docker_run_paper_image(
        image="paper:test",
        paper_root_host=str(tmp_path),
        run_dir_host=str(tmp_path),
        cwd_container="/app",
        cmd=["python", "-V"],
    )

    joined = " ".join(cmd)
    assert "-e HTTP_PROXY=http://host.docker.internal:7897" in joined
    assert "-e HTTPS_PROXY=http://host.docker.internal:7897" in joined


def test_docker_runtime_maps_explicit_user_and_writable_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EXECUTION_DOCKER_USER", "123:456")

    cmd = docker_run_paper_image(
        image="paper:test",
        paper_root_host=str(tmp_path),
        run_dir_host=str(tmp_path),
        cwd_container="/app",
        cmd=["python", "-V"],
    )

    joined = " ".join(cmd)
    assert "--user 123:456" in joined
    assert "-e PYTHONPATH=/app" in joined
    assert "-e PYTHONUNBUFFERED=1" in joined
    assert "-e PYTHONIOENCODING=utf-8" in joined
    assert "-e PYTHONUTF8=1" in joined
    assert "-e PYTHONPYCACHEPREFIX=/workspace/run_dir/.pycache" in joined
    assert "-e XDG_CACHE_HOME=/workspace/run_dir/.cache" in joined


def test_docker_runtime_keeps_paper_root_on_custom_pythonpath(tmp_path) -> None:
    cmd = docker_run_paper_image(
        image="paper:test",
        paper_root_host=str(tmp_path),
        run_dir_host=str(tmp_path),
        cwd_container="/app",
        cmd=["python", "-V"],
        env={"PYTHONPATH": "/extra/path"},
    )

    joined = " ".join(cmd)
    assert "-e PYTHONPATH=/app:/extra/path" in joined


def test_docker_runtime_user_mapping_can_use_image_default(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DOCKER_USER", "image")

    assert _docker_run_user_args() == []


def test_docker_runtime_passes_selected_env_names_without_values(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("EXECUTION_DOCKER_ENV_PASSTHROUGH", "OPENAI_API_KEY,OPENAI_BASE_URL")

    cmd = docker_run_paper_image(
        image="paper:test",
        paper_root_host=str(tmp_path),
        run_dir_host=str(tmp_path),
        cwd_container="/app",
        cmd=["python", "-V"],
        env_passthrough=_docker_env_passthrough({}),
    )

    joined = " ".join(cmd)
    assert "-e OPENAI_API_KEY" in joined
    assert "secret-value" not in joined
    assert "OPENAI_BASE_URL" not in joined


def test_docker_runtime_passes_default_execution_llm_env_names(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EXECUTION_OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("EXECUTION_OPENAI_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("EXECUTION_OPENAI_MODEL", "gpt-test")

    cmd = docker_run_paper_image(
        image="paper:test",
        paper_root_host=str(tmp_path),
        run_dir_host=str(tmp_path),
        cwd_container="/app",
        cmd=["python", "-V"],
        env_passthrough=_docker_env_passthrough({}),
    )

    joined = " ".join(cmd)
    assert "-e EXECUTION_OPENAI_API_KEY" in joined
    assert "-e EXECUTION_OPENAI_BASE_URL" in joined
    assert "-e EXECUTION_OPENAI_MODEL" in joined
    assert "secret-value" not in joined


def test_normalize_container_proxy_rewrites_loopback() -> None:
    assert _normalize_container_proxy("http://127.0.0.1:7897") == "http://host.docker.internal:7897"
    assert _normalize_container_proxy("http://localhost:7897") == "http://host.docker.internal:7897"


def test_docker_install_deps_installs_numpy_before_torch() -> None:
    text = _paper_install_deps_py_text()

    assert text.index("if numpy_line:") < text.index("if torch_pin:")


def test_docker_install_deps_fallback_uses_requirements_file() -> None:
    text = _paper_install_deps_py_text()

    assert "requirements.codegen.single_rest.txt" in text
    assert "'-r', str(tmp)" in text
    assert "'--upgrade', dep" not in text
    assert "*shlex.split(dep)" not in text


def test_docker_install_deps_clears_old_torch_execstack() -> None:
    text = _paper_install_deps_py_text()

    assert "def _clear_torch_executable_stack()" in text
    assert "PT_GNU_STACK" in text
    assert "torch_execstack_cleared" in text


def test_docker_install_deps_repairs_pydantic_core_mismatch() -> None:
    text = _paper_install_deps_py_text()

    assert "def _repair_python_package_consistency()" in text
    assert "pydantic-core" in text
    assert "pip_check_pydantic_repaired" in text


def test_docker_python_image_does_not_fallback_to_wrong_minor(monkeypatch) -> None:
    monkeypatch.setattr("fact_generation.execution.tools.docker._image_exists", lambda image: False)

    assert _select_python_image({}, "3.12") == "python:3.12"
    assert _select_python_image({}, "3.11") == "python:3.11"


def test_dockerfile_installs_deps_globally_for_host_uid_runtime() -> None:
    text = _paper_install_deps_py_text()
    dockerfile = _paper_dockerfile_text(python_image="python:3.11")

    assert "USER user" not in dockerfile
    assert "python deployment/install_deps.py" in dockerfile
    assert "site.getsitepackages()" in text


def test_llm_fix_shell_wrapper_preserves_shell_form_commands() -> None:
    assert _to_shell(["sh", "-lc", "PYTHONDONTWRITEBYTECODE=1 python -m py_compile evaluate_acc.py"]) == (
        "PYTHONDONTWRITEBYTECODE=1 python -m py_compile evaluate_acc.py"
    )
    assert _to_shell(["python", "-m", "pip", "install", "numpy<2"]) == "python -m pip install 'numpy<2'"


def test_llm_fix_skips_commands_that_only_rerun_failed_task() -> None:
    failed = ["bash", "run_for_dataset.sh"]

    assert _is_rerun_failed_task_command(["bash", "run_for_dataset.sh"], failed)
    assert _is_rerun_failed_task_command(["bash", "-lc", "bash run_for_dataset.sh"], failed)
    assert not _is_rerun_failed_task_command(["python", "-m", "pip", "install", "numpy"], failed)


def test_llm_fix_detects_host_dependency_install_commands(monkeypatch) -> None:
    assert _is_host_dependency_install_command(["python", "-m", "pip", "install", "torch"])
    assert _is_host_dependency_install_command(["bash", "-lc", "python -m pip install -r requirements.txt"])
    assert _is_host_dependency_install_command(["conda", "install", "-y", "pytorch"])
    assert not _is_host_dependency_install_command(["python", "-m", "py_compile", "main.py"])

    monkeypatch.delenv("EXECUTION_ALLOW_HOST_DEP_INSTALLS", raising=False)
    monkeypatch.delenv("FACTREVIEW_ALLOW_HOST_DEP_INSTALLS", raising=False)
    assert not _host_dependency_installs_allowed({})
    monkeypatch.setenv("EXECUTION_ALLOW_HOST_DEP_INSTALLS", "1")
    assert _host_dependency_installs_allowed({})


def test_llm_fix_respects_zero_max_attempts() -> None:
    assert _resolve_max_attempts({"max_attempts": 0}, {}) == 0
    assert _resolve_max_attempts({}, {"max_attempts": "0"}) == 0
    assert _resolve_max_attempts({}, {}) == 5


def test_llm_fix_container_path_normalization_uses_mount_paths(tmp_path) -> None:
    run_dir = tmp_path / "run"
    paper_root = run_dir / "workspace" / "source"
    paper_root.mkdir(parents=True)

    shell = (
        f"mkdir -p {paper_root.resolve()}/.hf_cache && "
        f"cp {run_dir.resolve()}/logs/output.txt {paper_root.resolve()}/result.txt"
    )
    patched = _normalize_container_path_text(shell, str(paper_root), run_dir)

    assert f"{paper_root.resolve()}" not in patched
    assert f"{run_dir.resolve()}" not in patched
    assert "mkdir -p /app/.hf_cache" in patched
    assert "cp /workspace/run_dir/logs/output.txt /app/result.txt" in patched


def test_llm_fix_rewrites_container_path_leaks_in_workspace(tmp_path) -> None:
    run_dir = tmp_path / "run"
    paper_root = run_dir / "workspace" / "source"
    paper_root.mkdir(parents=True)
    script = paper_root / "run.sh"
    script.write_text(
        f'export HF_HOME="{paper_root.resolve()}/.hf_cache"\n'
        f'python eval.py --out "{run_dir.resolve()}/artifacts/metrics.json"\n',
        encoding="utf-8",
    )

    rewritten = _rewrite_container_path_leaks(str(paper_root), run_dir)

    assert rewritten == ["run.sh"]
    text = script.read_text(encoding="utf-8")
    assert f"{paper_root.resolve()}" not in text
    assert f"{run_dir.resolve()}" not in text
    assert 'HF_HOME="/app/.hf_cache"' in text
    assert '"/workspace/run_dir/artifacts/metrics.json"' in text


def test_no_docker_llm_fix_maps_container_cwd_to_host_paths(tmp_path) -> None:
    run_dir = tmp_path / "run"
    paper_root = run_dir / "workspace" / "source"
    paper_root.mkdir(parents=True)

    assert _container_cwd_to_host("/app", str(paper_root), run_dir) == str(paper_root)
    assert _container_cwd_to_host("/app/scripts", str(paper_root), run_dir) == str(paper_root / "scripts")
    assert _container_cwd_to_host("/workspace/run_dir/artifacts", str(paper_root), run_dir) == str(
        run_dir / "artifacts"
    )


def test_heuristic_tasks_use_paper_targets_and_readme_commands(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "```bash",
                "python train.py --dataset CIFAR-10 --model ours",
                "python eval.py --dataset CIFAR-10 --model ours",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "eval.py").write_text("print('accuracy: 94.3')\n", encoding="utf-8")

    result = infer_tasks_heuristic(
        str(tmp_path),
        mode="full",
        paper_metric_targets=[
            {
                "paper_table_id": "table_1",
                "dataset": "CIFAR-10",
                "method": "ours",
                "metrics": {"accuracy": 94.2},
                "paper_claim": "Ours reaches 94.2 accuracy on CIFAR-10.",
            }
        ],
    )

    target_tasks = [t for t in result.tasks if t.get("expected_metrics") == {"accuracy": 94.2}]
    assert target_tasks
    assert any(t.get("family") == "eval" for t in target_tasks)
    assert all(t.get("metric_artifact_path") for t in target_tasks)


def test_heuristic_tasks_do_not_install_missing_requirements(tmp_path) -> None:
    (tmp_path / "README.md").write_text("# Tiny repo\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="smoke")

    assert not any(t.get("id") == "install_deps" for t in result.tasks)
    assert any(t.get("id") == "repo_smoke" for t in result.tasks)


def test_heuristic_smoke_uses_unique_top_level_python_file(tmp_path) -> None:
    (tmp_path / "dinov2_moe_new.py").write_text("print('ok')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="smoke")

    task = next(t for t in result.tasks if t.get("id") == "repo_smoke")
    assert task.get("cmd") == ["python", "-m", "py_compile", "dinov2_moe_new.py"]


def test_heuristic_import_smoke_ignores_top_level_test_modules(tmp_path) -> None:
    pkg = tmp_path / "agent"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "test_litellm_agent_pattern.py").write_text(
        "model = 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0'\n",
        encoding="utf-8",
    )

    result = infer_tasks_heuristic(str(tmp_path), mode="smoke")

    assert "agent" in result.evidence["python_import_targets"]
    assert "test_litellm_agent_pattern" not in result.evidence["python_import_targets"]


def test_heuristic_smoke_imports_library_repo_and_records_notebook(tmp_path) -> None:
    pkg = tmp_path / "iemm"
    pkg.mkdir()
    (pkg / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (pkg / "utils.py").write_text("VALUE = 2\n", encoding="utf-8")
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "very_simple_example.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": "print('ok')"}]}),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "Start with `very_simple_example.ipynb` using `jupyter notebook very_simple_example.ipynb`.\n",
        encoding="utf-8",
    )

    result = infer_tasks_heuristic(str(tmp_path), mode="smoke")

    smoke = next(t for t in result.tasks if t.get("id") == "repo_smoke")
    assert smoke.get("cmd") == [
        "python",
        "-c",
        "import importlib; importlib.import_module('iemm.core'); print('import ok: iemm.core')",
    ]
    notebook = next(t for t in result.tasks if t.get("id") == "reproduce_notebook_experiments_very_simple_example_ipynb")
    assert notebook.get("enabled") is False
    assert notebook.get("disabled_reason") == "full_mode_required"
    assert "jupyter" in notebook.get("cmd", [])
    assert result.evidence["python_import_targets"] == ["iemm.core"]
    assert result.evidence["notebook_paths"] == ["experiments/very_simple_example.ipynb"]


def test_heuristic_full_adds_notebook_dependency_smoke_from_cells(tmp_path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "analysis.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "import numpy as np\n",
                            "import pandas as pd\n",
                            "from matplotlib import pyplot as plt\n",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("Run `experiments/analysis.ipynb` for results.\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")
    task_ids = [task.get("id") for task in result.tasks]
    dep_idx = task_ids.index("notebook_dependency_smoke")
    nb_idx = task_ids.index("reproduce_notebook_experiments_analysis_ipynb")
    dep_task = result.tasks[dep_idx]

    assert dep_idx < nb_idx
    assert dep_task["enabled"] is True
    assert dep_task["notebook_import_modules"][:3] == ["nbformat", "nbconvert", "IPython"]
    assert "pandas" in dep_task["notebook_import_modules"]
    assert "matplotlib" in dep_task["cmd"][2]


def test_heuristic_disables_api_notebook_only_when_api_tasks_disabled(tmp_path, monkeypatch) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "main.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "from openai import OpenAI\nclient = OpenAI(api_key='x')\n",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("Run `experiments/main.ipynb` for the paper results.\n", encoding="utf-8")

    monkeypatch.delenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", raising=False)
    allowed = infer_tasks_heuristic(str(tmp_path), mode="full")
    allowed_task = next(t for t in allowed.tasks if t.get("id") == "reproduce_notebook_experiments_main_ipynb")
    assert allowed_task.get("requires_external_api") is True
    assert allowed_task.get("enabled") is True
    assert allowed_task.get("disabled_reason") is None

    monkeypatch.setenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", "1")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("id") == "reproduce_notebook_experiments_main_ipynb")
    assert task.get("requires_external_api") is True
    assert task.get("enabled") is False
    assert task.get("disabled_reason") == "external_api_or_model_server_required"


def test_heuristic_ignores_vendored_notebooks(tmp_path) -> None:
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    (notebooks / "paper.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": "print('paper')\n"}]}),
        encoding="utf-8",
    )
    vendored = tmp_path / "simpletransformers" / "examples" / "t5"
    vendored.mkdir(parents=True)
    (vendored / "data_prep.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": "import openai\n"}]}),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("Run notebooks folder for the paper experiments.\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="smoke")

    assert result.evidence["notebook_paths"] == ["notebooks/paper.ipynb"]
    assert not any("simpletransformers" in str(t.get("id") or "") for t in result.tasks)


def test_task_mode_policy_keeps_smoke_fast_for_llm_generated_tasks() -> None:
    tasks = [
        {
            "id": "install_deps",
            "family": "prepare",
            "enabled": True,
            "cmd": ["python", "-m", "pip", "install", "iemm", "jupyter"],
        },
        {
            "id": "smoke_execute_notebook",
            "family": "smoke",
            "enabled": True,
            "cmd": [
                "python",
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "experiments/main.ipynb",
            ],
        },
        {
            "id": "smoke_import",
            "family": "smoke",
            "enabled": True,
            "cmd": ["python", "-c", "import iemm.core; print('ok')"],
        },
        {
            "id": "eval_empty_reason",
            "family": "eval",
            "enabled": False,
            "disabled_reason": "",
            "cmd": ["python", "eval.py"],
        },
    ]

    _apply_mode_policy(tasks, mode="smoke")

    assert tasks[0]["enabled"] is False
    assert tasks[0]["disabled_reason"] == "smoke_mode_prepare_disabled"
    assert tasks[1]["enabled"] is False
    assert tasks[1]["disabled_reason"] == "full_mode_required"
    assert tasks[2]["enabled"] is True
    assert tasks[3]["disabled_reason"] == "full_mode_required"


def test_static_import_policy_disables_namespace_package_root_import(tmp_path) -> None:
    pkg = tmp_path / "iemm"
    pkg.mkdir()
    (pkg / "core.py").write_text("class IEMM: pass\n", encoding="utf-8")
    tasks = [
        {
            "id": "bad_root_import",
            "family": "smoke",
            "enabled": True,
            "cmd": ["python", "-c", "from iemm import IEMM; print(IEMM)"],
        },
        {
            "id": "good_module_import",
            "family": "smoke",
            "enabled": True,
            "cmd": ["python", "-c", "import iemm.core; print('ok')"],
        },
    ]

    _apply_static_import_policy(tasks, tmp_path)

    assert tasks[0]["enabled"] is False
    assert tasks[0]["disabled_reason"] == "namespace_package_root_import_unavailable"
    assert tasks[1]["enabled"] is True


def test_static_import_policy_disables_import_module_list_with_top_level_data_load(tmp_path) -> None:
    (tmp_path / "adv_gnn.py").write_text(
        "def load_pokec(path):\n"
        "    return path\n"
        "dataset = 'pokec_n'\n"
        "data = load_pokec('/data/private/pokec.csv')\n",
        encoding="utf-8",
    )
    (tmp_path / "utils.py").write_text("VALUE = 1\n", encoding="utf-8")
    tasks = [
        {
            "id": "smoke_import_modules",
            "family": "smoke",
            "enabled": True,
            "cmd": [
                "python",
                "-c",
                "import importlib; mods=['adv_gnn','utils']; [importlib.import_module(m) for m in mods]",
            ],
        }
    ]

    _apply_static_import_policy(tasks, tmp_path)

    assert tasks[0]["enabled"] is False
    assert tasks[0]["disabled_reason"] == "module_import_has_top_level_side_effects"
    assert tasks[0]["static_import_issues"]["module"] == "adv_gnn"
    assert tasks[0]["static_import_issues"]["top_level_calls"] == ["load_pokec"]


def test_missing_entrypoint_policy_disables_nonexistent_script(tmp_path) -> None:
    tasks = [
        {
            "id": "eval_missing",
            "family": "eval",
            "enabled": True,
            "cmd": ["python", "compute_pareto_metrics.py", "--out_dir", "outputs/pareto_eval"],
        },
        {
            "id": "train_ok",
            "family": "train",
            "enabled": True,
            "cmd": ["python", "{paper_root}/run.py", "--help"],
        },
    ]
    (tmp_path / "run.py").write_text("print('ok')\n", encoding="utf-8")

    _apply_missing_entrypoint_policy(tasks, tmp_path)

    assert tasks[0]["enabled"] is False
    assert tasks[0]["disabled_reason"] == "script_entrypoint_not_found"
    assert tasks[0]["static_entrypoint_issues"] == {"missing_script": "compute_pareto_metrics.py"}
    assert tasks[1]["enabled"] is True


def test_plan_detects_runtime_pip_installs_for_paper_image_patch() -> None:
    assert _is_runtime_pip_install_cmd(["python", "-m", "pip", "install", "iemm"])
    assert _is_runtime_pip_install_cmd(["pip", "install", "-r", "requirements.txt"])
    assert _is_runtime_pip_install_cmd(["bash", "-lc", "python -m pip install -e ."])
    assert not _is_runtime_pip_install_cmd(["python", "-c", "import iemm"])


def test_heuristic_marks_shell_script_external_api_through_python_imports(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", "1")
    (tmp_path / "README.md").write_text("```bash\nbash run_for_dataset.sh\n```\n", encoding="utf-8")
    for idx in range(20):
        (tmp_path / f"helper_{idx}.py").write_text("print('helper')\n", encoding="utf-8")
    (tmp_path / "run_for_dataset.sh").write_text(
        "\n".join(
            [f"python helper_{idx}.py" for idx in range(20)]
            + ["output=$(python inferencer.py --dataset sst5)"]
        ),
        encoding="utf-8",
    )
    (tmp_path / "inferencer.py").write_text(
        "from src.models.api_client import run_api\nprint(run_api)\n",
        encoding="utf-8",
    )
    api_dir = tmp_path / "src" / "models"
    api_dir.mkdir(parents=True)
    (api_dir / "api_client.py").write_text("import openai\n\ndef run_api():\n    return openai\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("cmd") == ["bash", "run_for_dataset.sh"])
    assert task.get("requires_external_api") is True
    assert task.get("enabled") is False
    assert task.get("disabled_reason") == "external_api_or_model_server_required"


def test_heuristic_smoke_finds_nested_main_script(tmp_path) -> None:
    nested = tmp_path / "Code" / "Diagramdiff platform" / "Core algorithm code"
    nested.mkdir(parents=True)
    (nested / "main.py").write_text("print('ok')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="smoke")

    task = next(t for t in result.tasks if t.get("id") == "repo_smoke")
    assert task.get("cmd") == [
        "python",
        "-m",
        "py_compile",
        "Code/Diagramdiff platform/Core algorithm code/main.py",
    ]


def test_heuristic_full_adds_entrypoint_task_without_readme_command(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "working_main_system.py").write_text(
        "if __name__ == '__main__':\n    print('train')\n",
        encoding="utf-8",
    )

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("id") == "train_entrypoint_src_working_main_system_py")
    assert task.get("cmd") == ["python", "src/working_main_system.py"]
    assert task.get("enabled") is True


def test_heuristic_disables_entrypoint_with_required_args(tmp_path) -> None:
    (tmp_path / "main.py").write_text(
        "\n".join(
            [
                "import argparse",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--img_path', required=True)",
                "if __name__ == '__main__':",
                "    parser.parse_args()",
            ]
        ),
        encoding="utf-8",
    )

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("id") == "train_entrypoint_main_py")
    assert task.get("enabled") is False
    assert task.get("disabled_reason") == "required_cli_arguments_missing"


def test_heuristic_smoke_disables_readme_prepare_commands(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```bash\npip install -r requirements.txt\npython main.py --help\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text("tensorflow-gpu==1.14.0\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="smoke")

    prepare_tasks = [t for t in result.tasks if t.get("family") == "prepare"]
    assert prepare_tasks
    assert all(t.get("enabled") is False for t in prepare_tasks)


def test_heuristic_full_adds_conventional_preprocess_before_training(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```bash\npython run.py -score_func transe -opn sub\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "run.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "preprocess.sh").write_text("mkdir -p data\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    ids = [str(t.get("id") or "") for t in result.tasks]
    assert "prepare_preprocess_sh" in ids
    assert any(t.get("id") == "train_transe_sub" for t in result.tasks)
    assert ids.index("prepare_preprocess_sh") < ids.index("train_transe_sub")

    prep = next(t for t in result.tasks if t.get("id") == "prepare_preprocess_sh")
    assert prep.get("enabled") is True
    assert prep.get("cmd") == ["bash", "preprocess.sh"]
    assert not any((t.get("cmd") or [])[:2] == ["cmd", "/c"] for t in result.tasks)


def test_heuristic_skips_conda_environment_commands(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```bash\nconda create -n paper python=3.9\nconda activate paper\npython train.py\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    commands = [t.get("cmd") or [] for t in result.tasks]
    assert ["python", "train.py"] in commands
    assert not any(cmd and cmd[0] == "conda" for cmd in commands)


def test_heuristic_deduplicates_repeated_readme_commands(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```bash\nbash run_for_dataset.sh\nbash run_for_dataset.sh\nbash run_for_dataset.sh\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "run_for_dataset.sh").write_text("echo ok\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    commands = [t.get("cmd") for t in result.tasks]
    assert commands.count(["bash", "run_for_dataset.sh"]) == 1


def test_heuristic_infers_cwd_for_unique_readme_script(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```bash\npython generate_response.py\n```\n",
        encoding="utf-8",
    )
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "generate_response.py").write_text("print('ok')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("id") == "reproduce_readme_1")
    assert task.get("cwd") == "{paper_root}/Scripts"
    assert task.get("cmd") == ["python", "generate_response.py"]


def test_heuristic_marks_api_tasks_but_keeps_them_enabled_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", raising=False)
    (tmp_path / "README.md").write_text(
        "```bash\npython eval_model.py\npython plot_table.py\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_model.py").write_text(
        "from openai import OpenAI\nclient = OpenAI(api_key='xxx')\n",
        encoding="utf-8",
    )
    (tmp_path / "plot_table.py").write_text("print('accuracy: 23.9')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    api_task = next(t for t in result.tasks if t.get("id") == "eval_readme_1")
    table_task = next(t for t in result.tasks if t.get("id") == "reproduce_readme_2")
    assert api_task.get("requires_external_api") is True
    assert api_task.get("enabled") is True
    assert "disabled_reason" not in api_task
    assert table_task.get("requires_external_api") is not True
    assert table_task.get("enabled") is True


def test_heuristic_can_disable_api_tasks_for_server_queues(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", "1")
    (tmp_path / "README.md").write_text(
        "```bash\npython eval_model.py\npython plot_table.py\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_model.py").write_text(
        "from openai import OpenAI\nclient = OpenAI(api_key='xxx')\n",
        encoding="utf-8",
    )
    (tmp_path / "plot_table.py").write_text("print('accuracy: 23.9')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    api_task = next(t for t in result.tasks if t.get("id") == "eval_readme_1")
    table_task = next(t for t in result.tasks if t.get("id") == "reproduce_readme_2")
    assert api_task.get("requires_external_api") is True
    assert api_task.get("enabled") is False
    assert api_task.get("disabled_reason") == "external_api_or_model_server_required"
    assert table_task.get("enabled") is True


def test_external_api_policy_marks_llm_generated_tasks_for_server_queues(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", "1")
    script = tmp_path / "scripts" / "test" / "test_pipeline_gpt_4o_resume.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'export OPENAI_API_KEY="your api key"',
                'MODEL_NAME="gpt-4o"',
                "python -m tests.test_full_pipeline_resume",
            ]
        ),
        encoding="utf-8",
    )
    tasks = [
        {
            "id": "reproduce_gpt_4o_resume_pipeline",
            "family": "reproduce",
            "enabled": False,
            "disabled_reason": "full_mode_required",
            "cmd": ["bash", "scripts/test/test_pipeline_gpt_4o_resume.sh"],
            "method": "GPT-4o + ORGEval",
        }
    ]

    _apply_external_api_policy(tasks, tmp_path)

    assert tasks[0]["requires_external_api"] is True
    assert tasks[0]["enabled"] is False
    assert tasks[0]["disabled_reason"] == "external_api_or_model_server_required"


def test_external_api_policy_marks_uv_pytest_file_tasks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", "1")
    (tmp_path / "test_litellm_agent_pattern.py").write_text(
        "\n".join(
            [
                "from src.agent import Agent",
                "model = 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0'",
                "Agent(model)",
            ]
        ),
        encoding="utf-8",
    )
    tasks = [
        {
            "id": "smoke_pytest_litellm_agent_pattern",
            "family": "smoke",
            "enabled": True,
            "cmd": ["uv", "run", "python", "-m", "pytest", "test_litellm_agent_pattern.py", "-q"],
        }
    ]

    _apply_external_api_policy(tasks, tmp_path)

    assert tasks[0]["requires_external_api"] is True
    assert tasks[0]["enabled"] is False
    assert tasks[0]["disabled_reason"] == "external_api_or_model_server_required"


def test_external_api_policy_marks_inline_import_of_api_test_module(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", "1")
    (tmp_path / "test_litellm_agent_pattern.py").write_text(
        "MODEL = 'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0'\n",
        encoding="utf-8",
    )
    tasks = [
        {
            "id": "smoke_imports",
            "family": "smoke",
            "enabled": True,
            "cmd": [
                "uv",
                "run",
                "python",
                "-c",
                "import importlib; importlib.import_module('test_litellm_agent_pattern')",
            ],
        }
    ]

    _apply_external_api_policy(tasks, tmp_path)

    assert tasks[0]["requires_external_api"] is True
    assert tasks[0]["enabled"] is False
    assert tasks[0]["disabled_reason"] == "external_api_or_model_server_required"


def test_heuristic_marks_api_tasks_through_local_imports(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", "1")
    (tmp_path / "README.md").write_text("```bash\npython conversation.py\n```\n", encoding="utf-8")
    (tmp_path / "conversation.py").write_text("from agent import Agent\nAgent()\n", encoding="utf-8")
    (tmp_path / "agent.py").write_text(
        "from openai import OpenAI\nclient = OpenAI(api_key='Your API_KEY')\n",
        encoding="utf-8",
    )

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("id") == "reproduce_readme_1")
    assert task.get("requires_external_api") is True
    assert task.get("enabled") is False
    assert task.get("disabled_reason") == "external_api_or_model_server_required"


def test_heuristic_repairs_missing_readme_shell_script_to_existing_candidate(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```bash\nscripts/test/test_pipeline_gpt_4o_resume.sh\n```\n",
        encoding="utf-8",
    )
    scripts = tmp_path / "scripts" / "test"
    scripts.mkdir(parents=True)
    (scripts / "test_pipeline.sh").write_text("echo ok\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("id") == "reproduce_readme_1")
    assert task.get("cmd") == ["bash", "scripts/test/test_pipeline.sh"]


def test_heuristic_marks_shell_api_script(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DISABLE_EXTERNAL_API_TASKS", "1")
    (tmp_path / "README.md").write_text(
        "```bash\nbash run_eval.sh\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "run_eval.sh").write_text("export OPENAI_API_KEY=xxx\npython eval.py\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("id") == "eval_readme_1")
    assert task.get("requires_external_api") is True
    assert task.get("enabled") is False


def test_heuristic_does_not_treat_title_case_make_sentence_as_command(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```text\nMake predictions\nmake evaluate\n```\n",
        encoding="utf-8",
    )

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    commands = [t.get("cmd") for t in result.tasks]
    assert ["Make", "predictions"] not in commands
    assert ["make", "evaluate"] in commands


def test_heuristic_ignores_python_comments_that_look_like_make_commands(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```python\n# make batch size small enough so you do not run OOM\nprint('ok')\n```\n",
        encoding="utf-8",
    )

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    commands = [t.get("cmd") for t in result.tasks]
    assert ["make", "batch", "size", "small", "enough", "so", "you", "do", "not", "run", "OOM"] not in commands


def test_heuristic_ignores_readme_directory_paths(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```text\n./ckpt/domainnet126/\n```\n",
        encoding="utf-8",
    )

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    commands = [t.get("cmd") for t in result.tasks]
    assert ["./ckpt/domainnet126/"] not in commands


def test_heuristic_disables_unresolved_usage_placeholder_commands(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```bash\n./run.sh DATASET ADAPTER [CONFIG_FILE]\n./run.sh cifar10c petta\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "run.sh").write_text("echo ok\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    placeholder_task = next(t for t in result.tasks if t.get("id") == "reproduce_readme_1")
    concrete_task = next(t for t in result.tasks if t.get("id") == "reproduce_readme_2")
    assert placeholder_task.get("enabled") is False
    assert placeholder_task.get("disabled_reason") == "readme_placeholder_command"
    assert concrete_task.get("enabled") is True


def test_heuristic_detects_env_prefixed_training_command(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "```bash",
                "CUDA_VISIBLE_DEVICES=$GPU python main.py \\",
                "  --data_dir data/$1 \\",
                "  --do_train",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text("print('train')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("id") == "train_readme_1")
    assert task.get("cmd") == [
        "bash",
        "-lc",
        "CUDA_VISIBLE_DEVICES=$GPU python main.py --data_dir data/$1 --do_train",
    ]
    assert task.get("enabled") is False
    assert task.get("disabled_reason") == "readme_placeholder_command"


def test_heuristic_selects_first_readme_choice_placeholder(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "```bash\npython test_time_new.py --cfg cfgs/cifar10_c/[tent/cotta].yaml fed.fed_tech [fedavg/fedbn]\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "test_time_new.py").write_text("print('ok')\n", encoding="utf-8")

    result = infer_tasks_heuristic(str(tmp_path), mode="full")

    task = next(t for t in result.tasks if t.get("id") == "reproduce_readme_1")
    assert task.get("cmd") == [
        "python",
        "test_time_new.py",
        "--cfg",
        "cfgs/cifar10_c/tent.yaml",
        "fed.fed_tech",
        "fedavg",
    ]


def test_normalize_shell_script_line_endings(tmp_path) -> None:
    script = tmp_path / "preprocess.sh"
    script.write_bytes(b"#!/bin/bash\r\nmkdir data\r\n")

    changed = _normalize_shell_script_line_endings(tmp_path)

    assert changed == 1
    assert script.read_bytes() == b"#!/bin/bash\nmkdir data\n"


def test_collect_repo_requirements_uses_readme_and_imports(tmp_path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "## Python Packages",
                "* pytorch==1.12.1",
                "* dgl==1.0.1+cu113",
                "",
                "This benchmark evaluates several datasets but does not use the datasets package.",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "import torch",
                "from sklearn.metrics import accuracy_score",
                "import schemdraw",
                "from utils import local_helper",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "utils.py").write_text("def local_helper(): pass\n", encoding="utf-8")

    req_text = _collect_repo_requirements_text(tmp_path)

    assert "torch==1.12.1" in req_text
    assert "dgl==1.0.1" in req_text
    assert "numpy" in req_text
    assert "scikit-learn" in req_text
    assert "schemdraw" in req_text
    assert "utils" not in req_text
    assert "datasets" not in req_text
    assert _infer_python_spec_from_repo(tmp_path) == "3.10"


def test_infer_python_spec_prefers_pyproject_requires_python(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo"',
                'requires-python = ">=3.12"',
                "dependencies = [",
                '  "torch>=2.7.0",',
                '  "numpy>=2.2.6",',
                "]",
            ]
        ),
        encoding="utf-8",
    )

    assert _infer_python_spec_from_repo(tmp_path) == "3.12"
    req_text = _collect_repo_requirements_text(tmp_path)
    assert "torch>=2.7.0" in req_text
    assert "numpy>=2.2.6" in req_text


def test_collect_repo_requirements_reads_nested_requirements(tmp_path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "requirements.txt").write_text(
        "\n".join(
            [
                "cvxpy==1.7.1",
                "scikit_learn==1.7.1",
                "PyYAML==6.0.2",
            ]
        ),
        encoding="utf-8",
    )
    generated = tmp_path / "outputs"
    generated.mkdir()
    (generated / "requirements.txt").write_text("openai==1.0.0\n", encoding="utf-8")

    req_text = _collect_repo_requirements_text(tmp_path)

    assert "cvxpy==1.7.1" in req_text
    assert "scikit_learn==1.7.1" in req_text
    assert "PyYAML==6.0.2" in req_text
    assert "openai==1.0.0" not in req_text


def test_environment_yml_informs_python_spec_and_docker_requirements(tmp_path) -> None:
    (tmp_path / "environment.yml").write_text(
        "\n".join(
            [
                "name: demo",
                "dependencies:",
                "  - python=3.12.2",
                "  - pytorch=2.2.1=py3.12_cuda12.1",
                "  - torchvision=0.17.1",
                "  - pytorch-cuda=12.1",
                "  - libpng=1.6.39",
                "  - numpy=1.26.4",
                "  - pyyaml=6.0.1",
            ]
        ),
        encoding="utf-8",
    )

    assert _infer_python_spec_from_repo(tmp_path) == "3.12"
    req_text = _collect_repo_requirements_text(tmp_path)
    assert "torch" in req_text
    assert "torchvision" in req_text
    assert "numpy" in req_text
    assert "pyyaml" in req_text
    assert "python" not in req_text
    assert "pytorch-cuda" not in req_text
    assert "libpng" not in req_text
    assert "torch=" not in req_text


def test_collect_repo_requirements_adds_notebook_runtime_and_imports(tmp_path) -> None:
    (tmp_path / "iemm").mkdir()
    (tmp_path / "iemm" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "experiment.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "import numpy as np\n",
                            "import pandas as pd\n",
                            "from iemm.core import VALUE\n",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    req_text = _collect_repo_requirements_text(tmp_path)

    assert "nbformat" in req_text
    assert "nbconvert" in req_text
    assert "ipykernel" in req_text
    assert "numpy" in req_text
    assert "pandas" in req_text
    assert "iemm" not in req_text


def test_collect_repo_requirements_ignores_vendored_notebooks_and_sources(tmp_path) -> None:
    vendored = tmp_path / "simpletransformers" / "examples"
    vendored.mkdir(parents=True)
    (vendored / "data_prep.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": "import openai\nimport pandas as pd\n"}]}),
        encoding="utf-8",
    )
    transformers = tmp_path / "transformers"
    transformers.mkdir()
    (transformers / "client.py").write_text("import anthropic\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("import numpy as np\n", encoding="utf-8")

    req_text = _collect_repo_requirements_text(tmp_path)

    assert "numpy" in req_text
    assert "openai" not in req_text
    assert "pandas" not in req_text
    assert "anthropic" not in req_text


def test_collect_repo_requirements_can_skip_notebook_runtime_for_smoke(tmp_path) -> None:
    (tmp_path / "experiment.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": "import matplotlib.pyplot as plt\n"}]}),
        encoding="utf-8",
    )

    req_text = _collect_repo_requirements_text(tmp_path, include_notebook_runtime=False)

    assert "nbformat" in req_text
    assert "nbconvert" not in req_text
    assert "ipykernel" not in req_text
    assert "matplotlib" not in req_text
    assert _docker_include_notebook_requirements({"auto_tasks_mode": "smoke"}) is False
    assert _docker_include_notebook_requirements({"auto_tasks_mode": "full"}) is True


def test_patch_api_placeholders_reads_runtime_env_without_writing_secret(tmp_path) -> None:
    script = tmp_path / "generate_response.py"
    script.write_text(
        "\n".join(
            [
                "from openai import OpenAI",
                "API_KEY = 'YOUR API-KEY'",
                "BASE_URL = 'https://api.openai.com/v1'",
                "MODEL_NAME = 'o3-mini'",
            ]
        ),
        encoding="utf-8",
    )

    changed = _patch_api_placeholders_for_env(tmp_path)

    text = script.read_text(encoding="utf-8")
    assert changed == ["generate_response.py"]
    assert "EXECUTION_OPENAI_API_KEY" in text
    assert "EXECUTION_OPENAI_BASE_URL" in text
    assert "EXECUTION_OPENAI_MODEL" in text
    assert "sk-" not in text


def test_patch_api_placeholders_handles_lowercase_xxx_keys(tmp_path) -> None:
    script = tmp_path / "eval_model.py"
    script.write_text(
        "\n".join(
            [
                "from openai import OpenAI",
                "openai_api_key = 'xxx'",
                "openai_api_base = 'http://localhost:8000/v1'",
                "model_name = 'o3'",
                "client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)",
            ]
        ),
        encoding="utf-8",
    )

    changed = _patch_api_placeholders_for_env(tmp_path)

    text = script.read_text(encoding="utf-8")
    assert changed == ["eval_model.py"]
    assert "openai_api_key = __import__('os').environ.get('EXECUTION_OPENAI_API_KEY')" in text
    assert "openai_api_base = __import__('os').environ.get('EXECUTION_OPENAI_BASE_URL')" in text
    assert "model_name = __import__('os').environ.get('EXECUTION_OPENAI_MODEL')" in text


def test_patch_api_placeholders_handles_shell_exports(tmp_path) -> None:
    script = tmp_path / "run_eval.sh"
    script.write_text(
        "\n".join(
            [
                'export OPENAI_API_KEY="your api key"',
                'export OPENAI_BASE_URL="your api base url"',
                'MODEL_NAME="gpt-4o"',
                "python -m tests.test_full_pipeline_resume --model_name $MODEL_NAME",
            ]
        ),
        encoding="utf-8",
    )

    changed = _patch_api_placeholders_for_env(tmp_path)

    text = script.read_text(encoding="utf-8")
    assert changed == ["run_eval.sh"]
    assert 'export OPENAI_API_KEY="${EXECUTION_OPENAI_API_KEY:-' in text
    assert 'export OPENAI_BASE_URL="${EXECUTION_OPENAI_BASE_URL:-' in text
    assert 'MODEL_NAME="${EXECUTION_OPENAI_MODEL:-' in text


def test_anonymous_4open_repo_id_parses_repo_links() -> None:
    assert (
        _anonymous_4open_repo_id("https://anonymous.4open.science/r/FMP-AD84/README.md")
        == "FMP-AD84"
    )
    assert _anonymous_4open_repo_id("https://anonymous.4open.science/repository/BMAS-AAD0") == "BMAS-AAD0"
    assert _anonymous_4open_repo_id("https://github.com/mainlp/explaind") == ""


def test_openreview_forum_id_parses_forum_and_attachment_links() -> None:
    assert _openreview_forum_id("https://openreview.net/forum?id=wKPQXtVejB") == "wKPQXtVejB"
    assert (
        _openreview_forum_id("https://openreview.net/attachment?id=wKPQXtVejB&name=supplementary_material")
        == "wKPQXtVejB"
    )
    assert _openreview_forum_id("https://github.com/mainlp/explaind") == ""


def test_openreview_candidate_source_urls_extracts_code_and_attachments() -> None:
    urls = _openreview_candidate_source_urls(
        {
            "title": {"value": "A paper with https://example.com/not-code"},
            "code": {"value": "https://github.com/org/repo."},
            "supplementary_material": {"value": "/attachment?id=wKPQXtVejB&name=supplementary_material"},
            "software": {"value": "/attachment/abc123.zip"},
            "dataset": {"value": ["See https://huggingface.co/datasets/org/data,"]},
        }
    )

    assert urls == [
        "https://github.com/org/repo",
        "https://openreview.net/attachment?id=wKPQXtVejB&name=supplementary_material",
        "https://openreview.net/attachment/abc123.zip",
        "https://huggingface.co/datasets/org/data",
    ]


def test_openreview_download_logs_metadata_when_no_public_supplement(monkeypatch, tmp_path) -> None:
    def fake_metadata(forum_id: str, timeout_sec: int = 30) -> dict[str, object]:
        assert forum_id == "E8HGf11jTn"
        return {
            "title": {"value": "Ransomware Detection on Android"},
            "pdf": {"value": "/pdf?id=E8HGf11jTn"},
            "abstract": {"value": "No public source field is present."},
        }

    def fake_download(url: str, timeout_sec: int = 180) -> bytes:
        raise HTTPError(url, 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr("fact_generation.execution.nodes.prepare._openreview_note_metadata", fake_metadata)
    monkeypatch.setattr("fact_generation.execution.nodes.prepare._download_url_bytes", fake_download)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    with pytest.raises(RuntimeError, match="openreview_no_supplementary_or_code_url"):
        _download_openreview_supplementary(
            "https://openreview.net/forum?id=E8HGf11jTn",
            tmp_path / "source",
            logs_dir,
        )

    manifest = json.loads((logs_dir / "openreview_supplementary_metadata.json").read_text(encoding="utf-8"))
    assert manifest["forum_id"] == "E8HGf11jTn"
    assert manifest["http_status"] == 404
    assert manifest["candidate_source_urls"] == []
    assert manifest["content_keys"] == ["abstract", "pdf", "title"]


def test_openreview_download_reports_empty_metadata_explicitly(monkeypatch, tmp_path) -> None:
    def fake_metadata(forum_id: str, timeout_sec: int = 30) -> dict[str, object]:
        assert forum_id == "F5Cj26wfiu"
        return {}

    def fake_download(url: str, timeout_sec: int = 180) -> bytes:
        raise HTTPError(url, 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr("fact_generation.execution.nodes.prepare._openreview_note_metadata", fake_metadata)
    monkeypatch.setattr("fact_generation.execution.nodes.prepare._download_url_bytes", fake_download)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    with pytest.raises(RuntimeError, match="openreview_metadata_empty_or_unavailable"):
        _download_openreview_supplementary(
            "https://openreview.net/forum?id=F5Cj26wfiu",
            tmp_path / "source",
            logs_dir,
        )

    manifest = json.loads((logs_dir / "openreview_supplementary_metadata.json").read_text(encoding="utf-8"))
    assert manifest["forum_id"] == "F5Cj26wfiu"
    assert manifest["content_keys"] == []


def test_openreview_download_falls_back_to_candidate_repo(monkeypatch, tmp_path) -> None:
    def fake_metadata(forum_id: str, timeout_sec: int = 30) -> dict[str, object]:
        assert forum_id == "abc123"
        return {
            "code": {"value": "https://github.com/example/paper-code/tree/main"},
            "title": {"value": "Paper with separate code"},
        }

    def fake_download(url: str, timeout_sec: int = 180) -> bytes:
        assert "openreview.net/attachment" in url
        raise HTTPError(url, 404, "not found", hdrs=None, fp=None)

    def fake_run_command(cmd: list[str], cwd: str, timeout_sec: int | None = 3600, env=None) -> CommandResult:
        assert cmd[:3] == ["git", "clone", "--depth"]
        assert cmd[-2] == "https://github.com/example/paper-code"
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("# demo\n", encoding="utf-8")
        return CommandResult(cmd=cmd, cwd=cwd, returncode=0, stdout="", stderr="", duration_sec=0.01)

    monkeypatch.setattr("fact_generation.execution.nodes.prepare._openreview_note_metadata", fake_metadata)
    monkeypatch.setattr("fact_generation.execution.nodes.prepare._download_url_bytes", fake_download)
    monkeypatch.setattr("fact_generation.execution.nodes.prepare.run_command", fake_run_command)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    manifest = _download_openreview_supplementary(
        "https://openreview.net/forum?id=abc123",
        tmp_path / "source",
        logs_dir,
    )

    assert manifest["method"] == "candidate_git_clone"
    assert manifest["clone_url"] == "https://github.com/example/paper-code"
    assert manifest["attachment_http_status"] == 404
    assert manifest["candidate_source_urls"] == ["https://github.com/example/paper-code/tree/main"]
    assert (tmp_path / "source" / "README.md").exists()
    persisted = json.loads((logs_dir / "openreview_supplementary_download.json").read_text(encoding="utf-8"))
    assert persisted["method"] == "candidate_git_clone"


def test_openreview_candidate_clone_timeout_does_not_fallback(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run_command(cmd: list[str], cwd: str, timeout_sec: int | None = 3600, env=None) -> CommandResult:
        calls.append(cmd)
        return CommandResult(cmd=cmd, cwd=cwd, returncode=124, stdout="", stderr="TimeoutExpired", duration_sec=30)

    monkeypatch.setenv("EXECUTION_GIT_CLONE_TIMEOUT_SEC", "30")
    monkeypatch.setattr("fact_generation.execution.nodes.prepare.run_command", fake_run_command)

    with pytest.raises(RuntimeError, match="candidate_git_clone_timeout:30"):
        _download_openreview_candidate_source(
            "https://github.com/example/slow-repo",
            tmp_path / "source",
            tmp_path,
            1,
        )

    assert len(calls) == 1
    assert "--filter" in calls[0]


def test_openreview_download_falls_back_to_direct_attachment_after_timeout(monkeypatch, tmp_path) -> None:
    blob_io = BytesIO()
    with zipfile.ZipFile(blob_io, "w") as zf:
        zf.writestr("repo-main/README.md", "# demo\n")

    def fake_metadata(forum_id: str, timeout_sec: int = 30) -> dict[str, object]:
        assert forum_id == "slow123"
        return {
            "supplementary_material": {"value": "/attachment/direct-hash.zip"},
            "title": {"value": "Paper with direct attachment"},
        }

    def fake_download(url: str, timeout_sec: int = 180) -> bytes:
        if "attachment?id=slow123" in url:
            raise TimeoutError("download_total_timeout")
        if url == "https://openreview.net/attachment/direct-hash.zip":
            return blob_io.getvalue()
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("fact_generation.execution.nodes.prepare._openreview_note_metadata", fake_metadata)
    monkeypatch.setattr("fact_generation.execution.nodes.prepare._download_url_bytes", fake_download)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    manifest = _download_openreview_supplementary(
        "https://openreview.net/forum?id=slow123",
        tmp_path / "source",
        logs_dir,
    )

    assert manifest["method"] == "candidate_archive"
    assert manifest["attachment_http_status"] == 0
    assert manifest["candidate_url"] == "https://openreview.net/attachment/direct-hash.zip"
    assert (tmp_path / "source" / "README.md").exists()


def test_openreview_download_retries_canonical_attachment_after_transient_error(monkeypatch, tmp_path) -> None:
    blob_io = BytesIO()
    with zipfile.ZipFile(blob_io, "w") as zf:
        zf.writestr("repo-main/README.md", "# demo\n")
    calls: dict[str, int] = {}

    def fake_metadata(forum_id: str, timeout_sec: int = 30) -> dict[str, object]:
        assert forum_id == "retry123"
        return {
            "supplementary_material": {"value": "/attachment/stale-hash.zip"},
            "title": {"value": "Paper with transient attachment failure"},
        }

    def fake_download(url: str, timeout_sec: int = 180) -> bytes:
        calls[url] = calls.get(url, 0) + 1
        if "attachment?id=retry123" in url and calls[url] == 1:
            raise TimeoutError("download_total_timeout")
        if "attachment?id=retry123" in url:
            return blob_io.getvalue()
        raise HTTPError(url, 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr("fact_generation.execution.nodes.prepare._openreview_note_metadata", fake_metadata)
    monkeypatch.setattr("fact_generation.execution.nodes.prepare._download_url_bytes", fake_download)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    manifest = _download_openreview_supplementary(
        "https://openreview.net/forum?id=retry123",
        tmp_path / "source",
        logs_dir,
    )

    assert manifest["method"] == "candidate_archive"
    assert manifest["attachment_http_status"] == 0
    assert manifest["candidate_url"] == "https://openreview.net/attachment?id=retry123&name=supplementary_material"
    assert (tmp_path / "source" / "README.md").exists()


def test_download_url_bytes_retries_truncated_content_length(monkeypatch) -> None:
    calls = {"count": 0}

    class TruncatedResponse:
        def __init__(self) -> None:
            self.headers = {"Content-Length": "10"}

        def __enter__(self):
            self._chunks = [b"12345", b""]
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, n: int) -> bytes:
            return self._chunks.pop(0)

    def fake_urlopen(req, timeout: int):
        calls["count"] += 1
        return TruncatedResponse()

    monkeypatch.setattr("fact_generation.execution.nodes.prepare.urlopen", fake_urlopen)

    with pytest.raises(DownloadIncompleteError, match="download_incomplete"):
        _download_url_bytes("https://example.com/archive.zip", timeout_sec=30)

    assert calls["count"] == 3


def test_download_url_bytes_can_use_curl_backend(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DOWNLOAD_BACKEND", "curl")
    monkeypatch.setattr("fact_generation.execution.nodes.prepare.shutil.which", lambda name: "curl")

    def fake_run(cmd, capture_output, timeout, check):
        assert "--max-time" in cmd
        assert "--max-filesize" in cmd
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.write_bytes(b"archive-bytes")
        assert cmd[-1] == "https://example.com/archive.zip"
        assert capture_output is True
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("fact_generation.execution.nodes.prepare.subprocess.run", fake_run)

    assert _download_url_bytes("https://example.com/archive.zip", timeout_sec=30) == b"archive-bytes"


def test_download_url_bytes_curl_maps_max_filesize_to_limit_error(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DOWNLOAD_BACKEND", "curl")
    monkeypatch.setenv("EXECUTION_DOWNLOAD_MAX_BYTES", "10")
    monkeypatch.setattr("fact_generation.execution.nodes.prepare.shutil.which", lambda name: "curl")
    output_paths: list[Path] = []

    def fake_run(cmd, capture_output, timeout, check):
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_paths.append(output_path)
        output_path.write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, 63, stdout=b"", stderr=b"curl: (63) Maximum file size exceeded")

    monkeypatch.setattr("fact_generation.execution.nodes.prepare.subprocess.run", fake_run)

    with pytest.raises(DownloadLimitError, match="download_too_large"):
        _download_url_bytes("https://example.com/archive.zip", timeout_sec=30)

    assert output_paths
    assert not output_paths[0].exists()


def test_anonymous_4open_binary_download_uses_size_guard(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_DOWNLOAD_MAX_BYTES", "1000000")

    class LargeResponse:
        def __init__(self) -> None:
            self.headers = {"Content-Length": "1000001"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, n: int) -> bytes:
            return b""

    def fake_urlopen(req, timeout: int):
        return LargeResponse()

    monkeypatch.setattr("fact_generation.execution.nodes.prepare.urlopen", fake_urlopen)

    with pytest.raises(DownloadLimitError, match="download_too_large"):
        _anonymous_4open_get_bytes("/api/repo/demo/zip", timeout_sec=30)


def test_clone_timeout_uses_safe_default_and_env(monkeypatch) -> None:
    monkeypatch.delenv("EXECUTION_GIT_CLONE_TIMEOUT_SEC", raising=False)
    assert _git_clone_timeout_sec({}) == 900
    monkeypatch.setenv("EXECUTION_GIT_CLONE_TIMEOUT_SEC", "12")
    assert _git_clone_timeout_sec({}) == 12
    assert _git_clone_timeout_sec({"git_clone_timeout_sec": 7}) == 7


def test_partial_clone_source_dir_is_detected(tmp_path) -> None:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)

    assert not _source_dir_has_payload(source)
    assert _source_dir_looks_partial_clone(source)

    (source / "README.md").write_text("# ok\n", encoding="utf-8")
    assert _source_dir_has_payload(source)
    assert not _source_dir_looks_partial_clone(source)


def test_remove_tree_best_effort_handles_readonly_git_files(tmp_path) -> None:
    source = tmp_path / "source"
    objects = source / ".git" / "objects"
    objects.mkdir(parents=True)
    locked = objects / "pack.idx"
    locked.write_text("idx", encoding="utf-8")
    locked.chmod(stat.S_IREAD)

    assert _remove_tree_best_effort(source)
    assert not source.exists()


def test_extract_archive_bytes_flattens_single_root_and_blocks_traversal(tmp_path) -> None:
    blob_io = BytesIO()
    with zipfile.ZipFile(blob_io, "w") as zf:
        zf.writestr("repo-main/README.md", "# demo\n")
        zf.writestr("repo-main/main.py", "print('ok')\n")
        zf.writestr("repo-main/.DS_Store", "mac cruft\n")
        zf.writestr("repo-main/._README.md", "mac cruft\n")
        zf.writestr("__MACOSX/repo-main/README.md", "mac cruft\n")
        zf.writestr("../escape.txt", "bad\n")

    dest = tmp_path / "source"
    manifest = _extract_archive_bytes(blob_io.getvalue(), dest)

    assert manifest["files"] == 2
    assert (dest / "README.md").exists()
    assert (dest / "main.py").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_extract_archive_bytes_handles_deep_member_paths(tmp_path) -> None:
    blob_io = BytesIO()
    deep_dir = "/".join(["deep_path_segment_with_long_name"] * 8)
    member = f"repo-main/{deep_dir}/trainvalid_lasso_frac0p50_seed101_pred.png"
    with zipfile.ZipFile(blob_io, "w") as zf:
        zf.writestr("repo-main/README.md", "# demo\n")
        zf.writestr(member, b"png")

    dest = tmp_path / "source"
    manifest = _extract_archive_bytes(blob_io.getvalue(), dest)

    assert manifest["files"] == 2
    assert (dest / "README.md").exists()
    if manifest.get("path_rewrites"):
        long_member = dest / manifest["path_rewrites"][0]["to"]
    else:
        long_member = dest / deep_dir / "trainvalid_lasso_frac0p50_seed101_pred.png"
    with open(_extended_windows_path(long_member), "rb") as fh:
        assert fh.read() == b"png"


def test_extract_archive_bytes_skips_generated_output_dirs(tmp_path) -> None:
    blob_io = BytesIO()
    with zipfile.ZipFile(blob_io, "w") as zf:
        zf.writestr("repo-main/README.md", "# demo\n")
        zf.writestr("repo-main/code/outputs/custom_sweep/deep/pred.png", b"png")
        zf.writestr("repo-main/code/logs/train.log", b"log")
        zf.writestr("repo-main/logs-pv1/run.json", b"log")

    dest = tmp_path / "source"
    manifest = _extract_archive_bytes(blob_io.getvalue(), dest)

    assert manifest["files"] == 1
    assert (dest / "README.md").exists()
    assert not (dest / "code" / "outputs").exists()
    assert not (dest / "code" / "logs").exists()
    assert not (dest / "logs-pv1").exists()


def test_anonymous_4open_download_prefers_zip_archive(monkeypatch, tmp_path) -> None:
    blob_io = BytesIO()
    with zipfile.ZipFile(blob_io, "w") as zf:
        zf.writestr("repo-main/README.md", "# demo\n")

    def fake_get_bytes(path: str, timeout_sec: int = 120) -> bytes:
        assert path == "/api/repo/FMP-AD84/zip"
        return blob_io.getvalue()

    def fail_get_json(path: str, timeout_sec: int = 60):
        raise AssertionError(f"unexpected json API call: {path}")

    monkeypatch.setattr("fact_generation.execution.nodes.prepare._anonymous_4open_get_bytes", fake_get_bytes)
    monkeypatch.setattr("fact_generation.execution.nodes.prepare._anonymous_4open_get_json", fail_get_json)

    manifest = _download_anonymous_4open_repo(
        "https://anonymous.4open.science/r/FMP-AD84/README.md",
        tmp_path / "source",
        tmp_path,
    )

    assert manifest["method"] == "zip"
    assert manifest["files"] == 1
    assert (tmp_path / "source" / "README.md").exists()


def test_anonymous_4open_download_falls_back_to_files_api(monkeypatch, tmp_path) -> None:
    def fake_get_bytes(path: str, timeout_sec: int = 120) -> bytes:
        if path == "/api/repo/FMP-AD84/zip":
            raise HTTPError(path, 404, "not found", hdrs=None, fp=None)
        if path == "/api/repo/FMP-AD84/file/README.md?v=abc123":
            return b"# demo\n"
        raise AssertionError(f"unexpected bytes API call: {path}")

    def fake_get_json(path: str, timeout_sec: int = 60):
        if path == "/api/repo/FMP-AD84/options":
            return {"lastUpdateDate": "2024-01-01T00:00:00.000Z"}
        if path == "/api/repo/FMP-AD84/files/?path=&v=2024-01-01T00%3A00%3A00.000Z":
            return [
                {"name": "", "path": "", "size": 0},
                {"name": ".DS_Store", "path": "", "size": 1, "sha": "ignored"},
                {"name": "README.md", "path": "", "size": 7, "sha": "abc123"},
            ]
        raise AssertionError(f"unexpected json API call: {path}")

    monkeypatch.setattr("fact_generation.execution.nodes.prepare._anonymous_4open_get_bytes", fake_get_bytes)
    monkeypatch.setattr("fact_generation.execution.nodes.prepare._anonymous_4open_get_json", fake_get_json)

    manifest = _download_anonymous_4open_repo(
        "https://anonymous.4open.science/r/FMP-AD84/README.md",
        tmp_path / "source",
        tmp_path,
    )

    assert manifest["method"] == "files_api"
    assert manifest["files"] == 1
    assert (tmp_path / "source" / "README.md").read_text(encoding="utf-8") == "# demo\n"
    assert not (tmp_path / "source" / ".DS_Store").exists()


def test_run_execution_stage_accepts_repo_without_pdf(tmp_path) -> None:
    from fact_generation.execution.stage_runner import run_execution_stage

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Tiny repo\n", encoding="utf-8")
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")

    result = run_execution_stage(
        run_dir=tmp_path / "run",
        paper_key="tiny_repo",
        paper_root=str(repo),
        no_pdf_extract=True,
        no_llm=True,
        auto_tasks=True,
        auto_tasks_force=True,
        docker_enabled=False,
        max_attempts=1,
    )

    assert result.status in {"ok", "inconclusive"}
    execution_root = tmp_path / "run" / "stages" / "fact_generation" / "execution"
    assert (execution_root / "execution.json").exists()
    evidence = json.loads(
        (execution_root / "current" / "artifacts" / "execution_evidence.json").read_text(encoding="utf-8")
    )
    assert "node_duration_sec" in evidence["cost"]


def test_run_execution_stage_records_pdf_download_failure(monkeypatch, tmp_path) -> None:
    from fact_generation.execution.stage_runner import run_execution_stage

    def fail_materialize(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError("paper_pdf_download_timeout")

    monkeypatch.setattr("fact_generation.execution.nodes.prepare.materialize_paper_pdf", fail_materialize)

    result = run_execution_stage(
        run_dir=tmp_path / "run",
        paper_key="slow_pdf",
        paper_pdf="https://example.test/slow.pdf",
        no_pdf_extract=True,
        no_llm=True,
        auto_tasks=True,
        auto_tasks_force=True,
        docker_enabled=False,
        max_attempts=0,
    )

    assert result.status == "failed"
    execution_root = tmp_path / "run" / "stages" / "fact_generation" / "execution"
    assert (execution_root / "execution.json").exists()
    issues = (execution_root / "current" / "issues.jsonl").read_text(encoding="utf-8")
    assert "paper_pdf_unavailable" in issues


def test_run_execution_stage_treats_openreview_forum_as_source_locator(monkeypatch, tmp_path) -> None:
    from fact_generation.execution import stage_runner

    captured: dict[str, object] = {}

    async def fake_run_orchestrator_async(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"success": False, "exit_status": "skipped", "state": {"run": {"dir": ""}}}

    monkeypatch.setattr(stage_runner, "_run_orchestrator_async", fake_run_orchestrator_async)

    result = stage_runner.run_execution_stage(
        run_dir=tmp_path / "run",
        paper_pdf="https://openreview.net/forum?id=YqFLsI44vN",
        paper_key="bmas",
        no_pdf_extract=True,
        no_llm=True,
        auto_tasks=True,
        docker_enabled=False,
        max_attempts=0,
    )

    assert result.status == "skipped"
    assert captured["paper_pdf"] == ""
    assert captured["paper_repo_url"] == "https://openreview.net/forum?id=YqFLsI44vN"


def test_run_execution_stage_infers_key_from_source_locator(monkeypatch, tmp_path) -> None:
    from fact_generation.execution import stage_runner

    captured: dict[str, object] = {}

    async def fake_run_orchestrator_async(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"success": False, "exit_status": "skipped", "state": {"run": {"dir": ""}}}

    monkeypatch.setattr(stage_runner, "_run_orchestrator_async", fake_run_orchestrator_async)

    stage_runner.run_execution_stage(
        run_dir=tmp_path / "run",
        paper_pdf="https://openreview.net/forum?id=YqFLsI44vN",
        no_pdf_extract=True,
        no_llm=True,
        auto_tasks=True,
        docker_enabled=False,
        max_attempts=0,
    )

    assert captured["paper_key"] == "YqFLsI44vN"


def test_execution_stage_uses_active_dir_when_current_archive_is_locked(monkeypatch, tmp_path) -> None:
    from fact_generation.execution.stage_runner import _archive_prior_current_dir

    stage_root = tmp_path / "stage"
    current = stage_root / "current"
    current.mkdir(parents=True)
    (current / "README.md").write_text("# stale\n", encoding="utf-8")
    original_rename = Path.rename

    def locked_rename(self: Path, target: str | Path) -> Path:
        if self.resolve() == current.resolve():
            raise PermissionError("locked current")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", locked_rename)

    active = _archive_prior_current_dir(stage_root=stage_root, current_dir=current)

    assert active.name.startswith("current.active.")
    assert active.exists()
    assert current.exists()
    assert list(stage_root.glob("current_archive_warning.*.json"))


def test_run_execution_stage_passes_resolved_llm_identity(monkeypatch, tmp_path) -> None:
    from fact_generation.execution import stage_runner

    captured: dict[str, object] = {}

    async def fake_run_orchestrator_async(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"success": False, "exit_status": "skipped", "state": {"run": {"dir": ""}}}

    monkeypatch.setenv("MODEL_PROVIDER", "openai-codex")
    monkeypatch.setenv("EXECUTION_OPENAI_MODEL", "gpt-5.5")
    monkeypatch.setattr(stage_runner, "_run_orchestrator_async", fake_run_orchestrator_async)

    result = stage_runner.run_execution_stage(
        run_dir=tmp_path / "run",
        paper_key="identity",
        paper_pdf="https://example.test/paper.pdf",
        no_pdf_extract=True,
        no_llm=False,
        auto_tasks=True,
        docker_enabled=False,
        max_attempts=0,
    )

    assert result.status == "skipped"
    assert captured["llm_provider"] == "openai-codex"
    assert captured["llm_model"] == "gpt-5.5"


def test_explicit_source_does_not_load_same_key_demo_fixture(tmp_path) -> None:
    from fact_generation.execution.stage_runner import run_execution_stage

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Tiny repo\n", encoding="utf-8")
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")

    result = run_execution_stage(
        run_dir=tmp_path / "run",
        paper_key="compgcn",
        paper_root=str(repo),
        no_pdf_extract=True,
        no_llm=True,
        auto_tasks=True,
        auto_tasks_force=True,
        docker_enabled=False,
        max_attempts=1,
    )

    assert result.status == "inconclusive"


@pytest.mark.requires_docker
def test_docker_daemon_is_available_for_execution_stage() -> None:
    # Smoke check that an environment claiming to be Docker-capable actually
    # has the CLI available — protects against running the gated test on a
    # host where the orchestrator would crash in a confusing way.
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")
