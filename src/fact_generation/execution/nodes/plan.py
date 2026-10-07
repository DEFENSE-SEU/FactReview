from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from util.fs import write_text
from util.meta import collect_meta, write_meta
from util.recorder import append_event

from ..tools.paper_tables import extract_paper_metric_targets
from ..tools.task_infer import infer_tasks_heuristic, infer_tasks_llm
from ..v2_config import metric_tolerance
from .prepare import (
    _ensure_default_baseline,
    _read_text,
    _write_run_manifest,
    _write_tasks_risk_report,
    _write_yaml_or_json,
)


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_runtime_pip_install_cmd(cmd: Any) -> bool:
    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
        return False
    lowered = [x.strip().lower() for x in cmd]
    if len(lowered) >= 4 and lowered[0] in {"python", "python3"} and lowered[1:4] == ["-m", "pip", "install"]:
        return True
    if len(lowered) >= 2 and lowered[0] in {"pip", "pip3"} and lowered[1] == "install":
        return True
    if len(lowered) >= 3 and lowered[0] in {"bash", "sh"} and lowered[1] in {"-c", "-lc"}:
        return bool(re.search(r"\b(?:python(?:3)?\s+-m\s+)?pip(?:3)?\s+install\b", lowered[2]))
    return False


def _default_tolerance(metric: str, expected: Any) -> float:
    return metric_tolerance(metric, expected, profile="legacy_plan")


def _load_tasks_for_baseline(tasks_p: Path) -> list[dict[str, Any]]:
    if not tasks_p.exists():
        return []
    try:
        if tasks_p.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # type: ignore

            data = yaml.safe_load(tasks_p.read_text(encoding="utf-8", errors="ignore"))
            return data if isinstance(data, list) else []
        data = json.loads(tasks_p.read_text(encoding="utf-8", errors="ignore") or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _artifact_json_paths(task: dict[str, Any]) -> list[str]:
    out: list[str] = []
    raw_paths = task.get("artifact_paths") or []
    if not isinstance(raw_paths, list):
        return out
    for raw in raw_paths:
        token = str(raw or "").replace("\\", "/").strip()
        if not token or any(ch in token for ch in "*?[]"):
            continue
        if token.startswith("./"):
            token = token[2:]
        if token.lower().endswith(".json"):
            out.append(token)
    return out


def _expected_metric_artifact_paths(task: dict[str, Any]) -> list[str]:
    task_id = str(task.get("id") or "").strip()
    paths: list[str] = []
    explicit = str(task.get("metric_artifact_path") or "").replace("\\", "/").strip()
    if explicit:
        paths.append(explicit.lstrip("./"))
    if task_id:
        paths.append(f"metrics/{task_id}_metrics.json")
    paths.extend(_artifact_json_paths(task))
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _merge_auto_baseline(
    *,
    baseline_p: Path,
    tasks_p: Path,
    paper_metric_targets: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    try:
        raw = json.loads(_read_text(baseline_p) or "{}")
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    checks = raw.get("checks")
    if not isinstance(checks, list):
        checks = []

    if paper_metric_targets and not raw.get("paper_metric_targets"):
        raw["paper_metric_targets"] = paper_metric_targets
        raw["paper_metric_targets_source"] = "auto_extracted_from_paper"

    if not checks:
        generated: list[dict[str, Any]] = []
        for task in _load_tasks_for_baseline(tasks_p):
            if not isinstance(task, dict):
                continue
            expected = task.get("expected_metrics")
            if not isinstance(expected, dict) or not expected:
                continue
            paths = _expected_metric_artifact_paths(task)
            if not paths:
                continue
            rel_path = paths[0]
            generated.append(
                {
                    "type": "file_exists",
                    "path": rel_path,
                    "claim": f"{task.get('id') or rel_path} produced metric artifact",
                }
            )
            for metric, value in expected.items():
                try:
                    float(value)
                except Exception:
                    continue
                generated.append(
                    {
                        "type": "json_value",
                        "path": rel_path,
                        "json_path": ["metrics", str(metric)],
                        "json_path_alternatives": [[str(metric)]],
                        "metric": str(metric),
                        "expected": value,
                        "tolerance": _default_tolerance(str(metric), value),
                        "claim": f"{task.get('id') or rel_path}: {metric} matches paper target",
                    }
                )
        if generated:
            raw["checks"] = generated
            raw["checks_source"] = "auto_from_task_expected_metrics"
            append_event(
                run_dir,
                "baseline_auto_checks_written",
                {"path": str(baseline_p), "checks": len(generated)},
            )
        elif "checks" not in raw:
            raw["checks"] = []

    if paper_metric_targets or raw.get("checks"):
        baseline_p.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw


def plan_node(state: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = state.get("config", {}) or {}
    run_info = state.get("run", {})
    run_dir = Path(run_info.get("dir") or "")
    logs_dir = Path(run_info.get("logs_dir") or (run_dir / "logs"))

    paper_key = str(cfg.get("paper_key") or "paper").strip() or "paper"
    paper_root = Path(str(cfg.get("paper_root") or ".")).resolve()
    strategy = str(cfg.get("docker_strategy") or "").strip()
    run_id = str(run_info.get("id") or "")

    append_event(run_dir, "plan_start", {"paper_key": paper_key, "paper_root": str(paper_root)})
    state.setdefault("history", []).append(
        {"kind": "plan_start", "data": {"paper_key": paper_key, "paper_root": str(paper_root)}}
    )

    baseline_dir_raw = str(cfg.get("baseline_dir") or "").strip()
    if not baseline_dir_raw:
        raise RuntimeError(
            "plan_node requires cfg['baseline_dir'] to be set by prepare_node first; "
            f"got empty baseline_dir for paper_key={paper_key!r}."
        )
    baseline_dir = Path(baseline_dir_raw).resolve()
    tasks_path = str(cfg.get("tasks_path") or "").strip()
    baseline_path = str(cfg.get("baseline_path") or "").strip()
    if not tasks_path:
        tasks_path = str((baseline_dir / "tasks.yaml").resolve())
    if not baseline_path:
        baseline_path = str((baseline_dir / "baseline.json").resolve())
    cfg["tasks_path"] = tasks_path
    cfg["baseline_path"] = baseline_path

    paper_metric_targets: list[dict[str, Any]] = []
    try:
        tables_dir_raw = str(cfg.get("paper_extracted_tables_dir") or "").strip()
        md_path_raw = str(cfg.get("paper_pdf_extracted_md") or "").strip()
        targets = extract_paper_metric_targets(
            Path(tables_dir_raw) if tables_dir_raw else None,
            paper_markdown_path=Path(md_path_raw) if md_path_raw else None,
        )
        paper_metric_targets = [asdict(t) for t in targets]
        if paper_metric_targets:
            cfg["paper_metric_targets"] = paper_metric_targets
            write_text(
                logs_dir / "paper_metric_targets.json",
                json.dumps(paper_metric_targets, ensure_ascii=False, indent=2) + "\n",
            )
            append_event(
                run_dir,
                "paper_metric_targets_extracted",
                {"count": len(paper_metric_targets), "path": str(logs_dir / "paper_metric_targets.json")},
            )
    except Exception as exc:
        append_event(
            run_dir,
            "paper_metric_targets_extract_failed",
            {"error": f"{type(exc).__name__}: {exc}"},
        )

    tasks_p = Path(tasks_path)
    if (not tasks_p.exists()) or bool(cfg.get("auto_tasks")):
        mode = str(cfg.get("auto_tasks_mode") or "smoke").strip() or "smoke"
        force = bool(cfg.get("auto_tasks_force"))
        if tasks_p.exists() and (not force) and bool(cfg.get("auto_tasks")):
            append_event(run_dir, "tasks_keep_existing", {"path": tasks_path})
        else:
            paper_md_excerpt = ""
            try:
                mdp = str(cfg.get("paper_pdf_extracted_md") or "").strip()
                if mdp:
                    txt = _read_text(Path(mdp))
                    if len(txt) > 14000:
                        txt = txt[:14000] + "\n...(truncated)\n"
                    paper_md_excerpt = txt
            except Exception:
                paper_md_excerpt = ""

            use_llm = not bool(cfg.get("no_llm"))
            if use_llm:
                ir = infer_tasks_llm(
                    str(paper_root),
                    mode=mode,
                    cfg_provider=str(cfg.get("llm_provider") or ""),
                    cfg_model=str(cfg.get("llm_model") or ""),
                    cfg_base_url=str(cfg.get("llm_base_url") or ""),
                    paper_md_excerpt=paper_md_excerpt,
                    paper_metric_targets=paper_metric_targets,
                )
            else:
                ir = infer_tasks_heuristic(
                    str(paper_root),
                    mode=mode if bool(cfg.get("auto_tasks")) else "smoke",
                    paper_metric_targets=paper_metric_targets,
                )

            _write_yaml_or_json(tasks_p, ir.tasks)
            write_text(
                logs_dir / "tasks_infer_evidence.json",
                json.dumps(ir.evidence, ensure_ascii=False, indent=2) + "\n",
            )
            _write_tasks_risk_report(tasks_p, logs_dir)
            append_event(run_dir, "tasks_written", {"path": str(tasks_p), "count": len(ir.tasks)})

    # In per-paper image mode, dependencies are installed during image build.
    # Disable runtime pip tasks from auto-generated plans by default; they can
    # make a smoke run spend minutes in package resolution and mutate the image
    # after the reproducible build step. Set EXECUTION_ALLOW_RUNTIME_INSTALL_TASKS=1
    # for rare repos that truly need a run-time install command.
    if strategy == "paper_image" and not _truthy_env("EXECUTION_ALLOW_RUNTIME_INSTALL_TASKS"):
        try:
            import yaml  # type: ignore

            raw = tasks_p.read_text(encoding="utf-8", errors="ignore")
            data = yaml.safe_load(raw)
            if isinstance(data, list):
                changed = False
                for t in data:
                    if not isinstance(t, dict):
                        continue
                    cmd = t.get("cmd")
                    if _is_runtime_pip_install_cmd(cmd):
                        t["enabled"] = False
                        t["disabled_reason"] = "runtime_install_disabled_in_paper_image"
                        changed = True
                if changed:
                    tasks_p.write_text(
                        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                        errors="ignore",
                    )
                    append_event(run_dir, "tasks_patch_disable_runtime_installs", {"path": str(tasks_p)})
        except Exception:
            pass

    # Persist the effective tasks into the run directory so execution does not depend on baseline state.
    try:
        run_tasks_path = Path(run_dir) / "tasks.yaml"
        raw_tasks = tasks_p.read_text(encoding="utf-8", errors="ignore") if tasks_p.exists() else ""
        if raw_tasks.strip():
            write_text(run_tasks_path, raw_tasks)
            cfg["tasks_path"] = str(run_tasks_path)
            tasks_p = run_tasks_path
            append_event(run_dir, "tasks_persist_run_dir", {"path": str(run_tasks_path)})
    except Exception:
        pass

    baseline_p = Path(baseline_path)
    _ensure_default_baseline(baseline_p)
    state["baseline"] = _merge_auto_baseline(
        baseline_p=baseline_p,
        tasks_p=tasks_p,
        paper_metric_targets=paper_metric_targets,
        run_dir=run_dir,
    )

    state["config"] = cfg

    try:
        meta = collect_meta(
            run_id=run_id,
            paper_root=str(paper_root),
            tasks_path=str(tasks_p),
            baseline_path=str(baseline_p),
            llm_cfg={
                "provider": str(cfg.get("llm_provider") or ""),
                "model": str(cfg.get("llm_model") or ""),
                "base_url": str(cfg.get("llm_base_url") or ""),
                "no_llm": bool(cfg.get("no_llm")),
            },
        )
        write_meta(meta, run_dir)
    except Exception:
        pass

    _write_run_manifest(run_dir=run_dir, cfg=cfg, baseline_dir=baseline_dir)

    append_event(
        run_dir,
        "plan_ok",
        {
            "tasks_path": str(cfg.get("tasks_path") or ""),
            "baseline_path": str(cfg.get("baseline_path") or ""),
        },
    )
    state.setdefault("history", []).append(
        {
            "kind": "plan_ok",
            "data": {
                "tasks_path": str(cfg.get("tasks_path") or ""),
                "baseline_path": str(cfg.get("baseline_path") or ""),
            },
        }
    )
    state["status"] = "running"
    return state
