from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.client import llm_json, resolve_llm_config


@dataclass(frozen=True)
class InferResult:
    tasks: list[dict[str, Any]]
    evidence: dict[str, Any]


def _read_optional(path: Path, max_chars: int = 12000) -> str:
    try:
        if not path.exists():
            return ""
        txt = path.read_text(encoding="utf-8", errors="ignore")
        if len(txt) > max_chars:
            return txt[:max_chars] + "\n...(truncated)\n"
        return txt
    except Exception:
        return ""


def _guess_entrypoints(repo_root: Path) -> list[str]:
    # Conservative: only look for common top-level scripts.
    cands = ["launcher.py", "run.py", "eval.py", "main.py", "app.py"]
    out: list[str] = []
    for c in cands:
        if (repo_root / c).exists():
            out.append(c)
    return out


def _extract_example_commands_from_readme(readme_text: str) -> list[str]:
    """
    Extract a few likely shell commands from README code fences.
    Keep it best-effort and small; this is only used as hinting.
    """
    txt = readme_text or ""
    cmds: list[str] = []
    # Grab fenced blocks ```bash ... ```
    for m in re.finditer(r"```(?:bash|sh|shell)\s+([\s\S]*?)```", txt, flags=re.IGNORECASE):
        block = (m.group(1) or "").strip()
        for line in block.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cmds.append(s)
            if len(cmds) >= 24:
                return cmds
    return cmds


def _detect_benchmark_datasets(repo_root: Path) -> list[str]:
    data_dir = repo_root / "data"
    if not data_dir.exists() or not data_dir.is_dir():
        return []
    out: list[str] = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name.strip()
        if not name:
            continue
        out.append(name)
        if len(out) >= 20:
            break
    return out


def _entrypoint_arg_hints(repo_root: Path, entrypoints: list[str], max_chars: int = 4000) -> dict[str, str]:
    hints: dict[str, str] = {}
    for ep in entrypoints[:5]:
        txt = _read_optional(repo_root / ep, max_chars=max_chars)
        if not txt.strip():
            continue
        lines: list[str] = []
        for ln in txt.splitlines():
            s = ln.strip()
            if "add_argument(" in s or "ArgumentParser(" in s:
                lines.append(s)
            if len(lines) >= 40:
                break
        if lines:
            hints[ep] = "\n".join(lines)
    return hints


def _cmd_flag_value(cmd: list[str], *flags: str) -> str:
    for i, tok in enumerate(cmd[:-1]):
        if tok in flags:
            return str(cmd[i + 1])
    return ""


def _strip_flag(cmd: list[str], *flags: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for i, tok in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if tok in flags:
            if i + 1 < len(cmd):
                skip_next = True
            continue
        out.append(tok)
    return out


def _safe_id_part(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "task"


def _build_readme_matrix_tasks(
    readme_example_cmds: list[str], datasets: list[str], mode: str
) -> list[dict[str, Any]]:
    parsed_runs: list[list[str]] = []
    for raw in readme_example_cmds:
        s = (raw or "").strip()
        if not s.startswith("python run.py "):
            continue
        try:
            parts = shlex.split(s, posix=True)
        except Exception:
            continue
        if len(parts) >= 2 and parts[0] == "python" and parts[1] == "run.py":
            parsed_runs.append(parts)

    if not parsed_runs:
        return []

    multi_dataset = len(datasets) > 1
    expanded_datasets = datasets if datasets else [""]
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for parts in parsed_runs:
        base_cmd = _strip_flag(_strip_flag(parts[2:], "-name", "--name"), "-data", "--data")
        explicit_name = _cmd_flag_value(parts, "-name", "--name").strip()
        score_func = _cmd_flag_value(parts, "-score_func", "--score_func").strip() or "run"
        opn = _cmd_flag_value(parts, "-opn", "--opn").strip()
        explicit_dataset = _cmd_flag_value(parts, "-data", "--data").strip()

        dataset_values = [explicit_dataset] if explicit_dataset else expanded_datasets
        for dataset in dataset_values:
            dataset_tag = _safe_id_part(dataset) if dataset else ""
            name_base = _safe_id_part(explicit_name or "_".join(x for x in [score_func, opn] if x))
            run_name = name_base
            if multi_dataset and dataset_tag:
                run_name = f"{name_base}_{dataset_tag}"
            task_id = f"train_{run_name}"
            if task_id in seen_ids:
                continue

            cmd = ["python", "run.py", "-name", run_name]
            if dataset:
                cmd.extend(["-data", dataset])
            cmd.extend(base_cmd)

            out.append(
                {
                    "id": task_id,
                    "enabled": mode == "full",
                    "cwd": "{paper_root}",
                    "cmd": cmd,
                    "timeout_sec": 86400,
                    "use_conda": True,
                    "artifact_paths": ["checkpoints/**", "log/**"],
                }
            )
            seen_ids.add(task_id)
    return out


def _is_training_task(task: dict[str, Any]) -> bool:
    cmd = task.get("cmd")
    if not isinstance(cmd, list) or len(cmd) < 2:
        return False
    if cmd[0] != "python":
        return False
    if cmd[1] != "run.py":
        return False
    return "-h" not in cmd and "--help" not in cmd


def _append_eval_export_tasks(repo_root: Path, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluator = repo_root / "codeeval_eval_ckpt.py"
    if not evaluator.exists():
        return tasks

    existing_ids = {str(t.get("id") or "").strip() for t in tasks if isinstance(t, dict)}
    out: list[dict[str, Any]] = list(tasks)
    for task in tasks:
        if not isinstance(task, dict) or not _is_training_task(task):
            continue
        cmd = task.get("cmd")
        assert isinstance(cmd, list)
        run_name = _cmd_flag_value(cmd, "-name", "--name").strip()
        task_id = str(task.get("id") or "").strip()
        if not run_name or not task_id:
            continue

        eval_id = f"eval_{task_id[6:]}" if task_id.startswith("train_") else f"eval_{task_id}"
        if eval_id in existing_ids:
            continue

        out_path = f"./metrics/{task_id}_test.json"
        out.append(
            {
                "id": eval_id,
                "enabled": bool(task.get("enabled", True)),
                "cwd": "{paper_root}",
                "cmd": [
                    "python",
                    "codeeval_eval_ckpt.py",
                    "--ckpt-dir",
                    "./checkpoints",
                    "--prefix",
                    run_name,
                    "--out",
                    out_path,
                    "--split",
                    "test",
                ],
                "timeout_sec": 1800,
                "use_conda": True,
                "artifact_paths": [out_path.lstrip("./")],
            }
        )
        existing_ids.add(eval_id)
    return out


def _finalize_tasks(
    *,
    repo_root: Path,
    tasks: list[dict[str, Any]],
    readme_example_cmds: list[str],
    datasets: list[str],
    mode: str,
) -> list[dict[str, Any]]:
    matrix_tasks = _build_readme_matrix_tasks(readme_example_cmds, datasets, mode=mode)
    if matrix_tasks:
        non_train = [t for t in tasks if isinstance(t, dict) and not _is_training_task(t)]
        tasks = non_train + matrix_tasks
    return _append_eval_export_tasks(repo_root, tasks)


def infer_tasks_heuristic(repo_root: str, mode: str = "smoke") -> InferResult:
    root = Path(repo_root)
    readme = _read_optional(root / "README.md")
    req = _read_optional(root / "requirements.txt", max_chars=8000)
    entrypoints = _guess_entrypoints(root)
    examples = _extract_example_commands_from_readme(readme)
    datasets = _detect_benchmark_datasets(root)

    # Default install step. We keep it lightweight and let the framework's prepare/fix deal with stdlib-in-req.
    tasks: list[dict[str, Any]] = [
        {
            "id": "install_deps",
            "cwd": "{paper_root}",
            "cmd": ["python", "-m", "pip", "install", "-r", "{paper_root}/requirements.txt"],
            "timeout_sec": 3600,
            "use_conda": True,
        }
    ]

    # Smoke: check --help for a chosen entrypoint.
    ep = entrypoints[0] if entrypoints else ""
    if not ep:
        # last resort: do nothing but print cwd (still validates the runner)
        tasks.append(
            {
                "id": "repo_smoke",
                "cwd": "{paper_root}",
                "cmd": ["python", "-c", "import os; print('cwd=', os.getcwd()); print('ok')"],
                "timeout_sec": 60,
                "use_conda": True,
            }
        )
    else:
        tasks.append(
            {
                "id": "repo_smoke",
                "cwd": "{paper_root}",
                "cmd": ["python", ep, "--help"],
                "timeout_sec": 600,
                "use_conda": True,
            }
        )
        if (root / "eval.py").exists() and ep != "eval.py":
            tasks.append(
                {
                    "id": "eval_smoke",
                    "cwd": "{paper_root}",
                    "cmd": ["python", "eval.py", "--help"],
                    "timeout_sec": 600,
                    "use_conda": True,
                }
            )

    # Full: propose heavier commands but disable them by default.
    if mode == "full":
        if examples:
            tasks.append(
                {
                    "id": "readme_example_1",
                    "enabled": False,
                    "cwd": "{paper_root}",
                    "cmd": ["cmd", "/c", examples[0]] if os.name == "nt" else ["bash", "-lc", examples[0]],
                    "timeout_sec": 3600,
                    "use_conda": True,
                    "artifact_paths": ["results/**", "logs/**"],
                }
            )

    tasks = _finalize_tasks(
        repo_root=root,
        tasks=tasks,
        readme_example_cmds=examples,
        datasets=datasets,
        mode=mode,
    )
    evidence = {
        "mode": mode,
        "entrypoints": entrypoints,
        "datasets_detected": datasets,
        "readme_has_content": bool(readme.strip()),
        "requirements_present": bool(req.strip()),
        "readme_example_cmds": examples,
    }
    return InferResult(tasks=tasks, evidence=evidence)


def infer_tasks_llm(
    repo_root: str,
    mode: str,
    cfg_provider: str,
    cfg_model: str,
    cfg_base_url: str,
    paper_md_excerpt: str = "",
) -> InferResult:
    """
    LLM-assisted task inference. Must be safe by design:
    - Prefer smoke tasks.
    - Heavy tasks must be generated with enabled=false unless explicitly requested by user.
    - Only wrapper commands (no source edits).
    """
    root = Path(repo_root)
    readme = _read_optional(root / "README.md", max_chars=14000)
    req = _read_optional(root / "requirements.txt", max_chars=8000)
    entrypoints = _guess_entrypoints(root)
    readme_example_cmds = _extract_example_commands_from_readme(readme)
    datasets = _detect_benchmark_datasets(root)
    entrypoint_hints = _entrypoint_arg_hints(root, entrypoints)

    # Keep prompt small but informative. The goal is to produce tasks that actually reflect the repo's README
    # (download/preprocess/train/eval) while staying safe by default.
    prompt = {
        "goal": "Generate tasks.yaml for running/evaluating this repo in a reproducible way.",
        "mode": mode,
        "platform": {
            "host_os": os.name,
            "execution_os": "linux",
            "execution_environment": "docker paper_image",
        },
        "repo_root": str(root),
        "files_top_level": [p.name for p in sorted(root.iterdir())][:200],
        "entrypoints_detected": entrypoints,
        "entrypoint_arg_hints": entrypoint_hints,
        "datasets_detected": datasets,
        "readme_example_commands": readme_example_cmds,
        "readme_md_excerpt": readme,
        "paper_mineru_md_excerpt": (paper_md_excerpt or ""),
        "requirements_txt_excerpt": req,
        "schema": {
            "tasks": [
                {
                    "id": "string",
                    "enabled": True,
                    "cwd": "{paper_root}",
                    "cmd": ["python", "run.py", "--help"],
                    "timeout_sec": 600,
                    "use_conda": True,
                    "artifact_paths": ["results/**"],
                }
            ],
            "notes": ["string"],
        },
        "constraints": [
            "Return JSON only, no prose outside JSON.",
            "You MUST derive commands from README when possible; do not output generic placeholder tasks if the README provides concrete steps.",
            "Prefer: install deps -> (optional) download/preprocess -> run/eval -> collect artifacts.",
            "Include at least one smoke task (help/print/version) as an early, fast validation step.",
            "If proposing any heavy task (downloads dataset, trains model), set enabled=false unless mode=='full'.",
            "Do not propose source code edits.",
            "Tasks execute inside a Linux Docker container even when the host machine is Windows. Do not emit Windows-only wrappers like ['cmd','/c', ...] unless the repo itself explicitly requires Windows shells.",
            "Commands must be compatible with shell=False: use argv arrays. For multi-step shell pipelines use ['bash','-lc','...'] because execution is Linux-based.",
            "When README lists multiple reproduction commands, preserve the full set of distinct commands instead of sampling only one or two representative examples.",
            "When datasets_detected contains multiple benchmark datasets and the training entrypoint supports a dataset flag, expand reproduction tasks across those datasets unless the README clearly restricts a command to one dataset.",
            "When a training command does not specify a run name but the CLI supports one, add a stable explicit run name so downstream checkpoints/logs can be located deterministically.",
            "If the repo contains a local evaluator/export script that can turn checkpoints into machine-readable metrics, add follow-up eval/export tasks for each training task.",
            "If you emit a pip install task, prefer id='install_deps'. In paper-image Docker mode, avoid redundant runtime installs unless they are clearly necessary beyond image build.",
            "Use {paper_root} in cwd/cmd paths instead of hardcoding absolute paths.",
        ],
    }
    system = "You are a senior engineer generating a safe, reproducible tasks.yaml for a research repo."
    llm_cfg = resolve_llm_config(cfg_provider, cfg_model, cfg_base_url)
    resp = llm_json(prompt=json.dumps(prompt, ensure_ascii=False), system=system, cfg=llm_cfg)
    if not isinstance(resp, dict) or resp.get("status") == "error":
        # fallback to heuristics if LLM fails
        hr = infer_tasks_heuristic(repo_root, mode=mode)
        ev = dict(hr.evidence)
        ev["llm_error"] = resp
        return InferResult(tasks=hr.tasks, evidence=ev)

    tasks = resp.get("tasks")
    if not isinstance(tasks, list):
        hr = infer_tasks_heuristic(repo_root, mode=mode)
        ev = dict(hr.evidence)
        ev["llm_bad_shape"] = resp
        return InferResult(tasks=hr.tasks, evidence=ev)

    cleaned: list[dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        cmd = t.get("cmd")
        if not isinstance(tid, str) or not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            continue
        cleaned.append(t)

    finalized = cleaned or infer_tasks_heuristic(repo_root, mode=mode).tasks
    finalized = _finalize_tasks(
        repo_root=root,
        tasks=finalized,
        readme_example_cmds=readme_example_cmds,
        datasets=datasets,
        mode=mode,
    )
    evidence = {
        "mode": mode,
        "llm_used": True,
        "llm_provider": llm_cfg.provider,
        "llm_model": llm_cfg.model,
        "raw": resp,
    }
    return InferResult(tasks=finalized, evidence=evidence)
