from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from util.fs import write_text
from util.meta import collect_meta, write_meta
from util.recorder import append_event

from ..tools.task_infer import infer_tasks_heuristic, infer_tasks_llm
from .prepare import (
    _ensure_default_baseline,
    _read_text,
    _write_run_manifest,
    _write_tasks_risk_report,
    _write_yaml_or_json,
)


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
                )
            else:
                ir = infer_tasks_heuristic(
                    str(paper_root), mode=mode if bool(cfg.get("auto_tasks")) else "smoke"
                )

            _write_yaml_or_json(tasks_p, ir.tasks)
            write_text(
                logs_dir / "tasks_infer_evidence.json",
                json.dumps(ir.evidence, ensure_ascii=False, indent=2) + "\n",
            )
            _write_tasks_risk_report(tasks_p, logs_dir)
            append_event(run_dir, "tasks_written", {"path": str(tasks_p), "count": len(ir.tasks)})

    # In per-paper image mode, dependencies are installed during image build.
    # Disable any generic "python -m pip install -r ..." task to avoid reinstalling
    # and mutating the environment at runtime.
    if strategy == "paper_image":
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
                    if (
                        isinstance(cmd, list)
                        and cmd[:4] == ["python", "-m", "pip", "install"]
                        and "-r" in cmd
                    ):
                        t["enabled"] = False
                        changed = True
                if changed:
                    tasks_p.write_text(
                        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                        errors="ignore",
                    )
                    append_event(run_dir, "tasks_patch_disable_install_deps", {"path": str(tasks_p)})
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
    try:
        baseline_raw = json.loads(_read_text(baseline_p) or "{}")
        state["baseline"] = baseline_raw if isinstance(baseline_raw, dict) else {}
    except Exception:
        state["baseline"] = {}

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
