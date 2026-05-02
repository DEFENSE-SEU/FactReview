from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from llm.client import llm_json, resolve_llm_config
from util.fs import ensure_dir, write_text
from util.recorder import append_event
from util.subprocess_runner import persist_command_result, run_command

from ..tools.docker import docker_ensure_paper_image, docker_run_paper_image


def _extract_missing_module(stderr: str) -> str | None:
    # ModuleNotFoundError: No module named 'xxx'
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr or "")
    if m:
        return m.group(1)
    return None


_MODULE_TO_PIP = {
    # common mismatches
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
}


def _extract_missing_file(stderr: str) -> str | None:
    # Windows python: can't open file 'C:\\path\\to\\x.py': [Errno 2] No such file or directory
    m = re.search(r"can't open file ['\"]([^'\"]+)['\"]", stderr or "", flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _pick_smoke_entrypoint(paper_root: str) -> str:
    """
    Pick a best-effort entry script for `python <script> --help`.
    Keep it conservative: only choose from known common filenames.
    """
    pr = Path(paper_root or ".")
    for name in ["launcher.py", "run.py", "eval.py", "main.py", "app.py"]:
        if (pr / name).exists():
            return name
    return ""


def _normalize_llm_cmd_for_platform(cmd: list[str]) -> list[str]:
    """
    LLMs often emit a single-string shell command (a one-item list containing spaces).
    Our runner uses shell=False, so wrap these for the current platform.
    """
    if not cmd:
        return cmd
    # If cmd is a single string with spaces, treat it as a shell command.
    if len(cmd) == 1 and (" " in cmd[0].strip()):
        s = cmd[0].strip()
        if os.name == "nt":
            return ["cmd", "/c", s]
        return ["bash", "-lc", s]
    return cmd


def _to_shell(cmd: list[str]) -> str:
    if not cmd:
        return ""
    if len(cmd) == 1:
        return cmd[0].strip()
    if any((" " in (x or "").strip()) for x in cmd):
        return " && ".join([x.strip() for x in cmd if x.strip()])
    return " ".join([x.strip() for x in cmd if x.strip()])


def _normalize_shell_for_conda_env(shell: str) -> str:
    """
    In docker mode, commands are executed under `micromamba run ...`.
    Prefer `python -m pip` to avoid using the base image pip/python.
    """
    s = (shell or "").strip()
    if not s:
        return s
    s = re.sub(r"(^|&&\s*)pip3?\s+", r"\1python -m pip ", s)
    return s


def _torch_scatter_fallback_in_container_shell() -> str:
    """
    Inject a minimal `torch_scatter` python package into the current environment.
    This is a generic last resort when torch-scatter wheels/conda packages are unavailable.
    """
    return (
        "python - <<'PY'\n"
        "import os, site, pathlib\n"
        "sp = site.getsitepackages()[0]\n"
        "pkg = pathlib.Path(sp) / 'torch_scatter'\n"
        "pkg.mkdir(parents=True, exist_ok=True)\n"
        "(pkg / '__init__.py').write_text(\n"
        '    "import torch\\n\\n"\n'
        '    "def _expand_index(index, src, dim):\\n"\n'
        '    "    if index.dtype != torch.long: index = index.long()\\n"\n'
        '    "    if dim < 0: dim = src.dim() + dim\\n"\n'
        '    "    if index.dim() == 1 and src.dim() > 1:\\n"\n'
        '    "        shape = [1] * src.dim()\\n"\n'
        '    "        shape[dim] = index.numel()\\n"\n'
        '    "        index = index.view(*shape)\\n"\n'
        '    "    return index.expand_as(src)\\n\\n"\n'
        '    "def scatter_add(src, index, dim=0, out=None, dim_size=None):\\n"\n'
        '    "    if out is None:\\n"\n'
        '    "        if dim_size is None: dim_size = int(index.max().item()) + 1 if index.numel() else 0\\n"\n'
        '    "        out_shape = list(src.shape); out_shape[dim] = dim_size\\n"\n'
        '    "        out = torch.zeros(*out_shape, dtype=src.dtype, device=src.device)\\n"\n'
        '    "    idx = _expand_index(index, src, dim)\\n"\n'
        '    "    return out.scatter_add(dim, idx, src)\\n\\n"\n'
        '    "def scatter_max(src, index, dim=0, out=None, dim_size=None):\\n"\n'
        '    "    if index.dtype != torch.long: index = index.long()\\n"\n'
        '    "    if dim < 0: dim = src.dim() + dim\\n"\n'
        '    "    if dim_size is None: dim_size = int(index.max().item()) + 1 if index.numel() else 0\\n"\n'
        '    "    if src.dim() == 1:\\n"\n'
        "    \"        outv = torch.full((dim_size,), -float('inf'), dtype=src.dtype, device=src.device)\\n\"\n"
        '    "        arg = torch.full((dim_size,), -1, dtype=torch.long, device=src.device)\\n"\n'
        '    "        for i in range(src.numel()):\\n"\n'
        '    "            j = int(index[i].item()); v = src[i]\\n"\n'
        '    "            if v > outv[j]: outv[j] = v; arg[j] = i\\n"\n'
        '    "        return outv, arg\\n"\n'
        '    "    dims = list(range(src.dim())); dims[0], dims[dim] = dims[dim], dims[0]\\n"\n'
        '    "    inv = [0]*len(dims)\\n"\n'
        '    "    for i,d in enumerate(dims): inv[d]=i\\n"\n'
        '    "    srcp = src.permute(dims)\\n"\n'
        "    \"    outp = torch.full((dim_size, *srcp.shape[1:]), -float('inf'), dtype=src.dtype, device=src.device)\\n\"\n"
        '    "    argp = torch.full((dim_size, *srcp.shape[1:]), -1, dtype=torch.long, device=src.device)\\n"\n'
        '    "    for i in range(srcp.shape[0]):\\n"\n'
        '    "        j = int(index[i].item()); v = srcp[i]\\n"\n'
        '    "        better = v > outp[j]\\n"\n'
        '    "        outp[j] = torch.where(better, v, outp[j])\\n"\n'
        '    "        argp[j] = torch.where(better, torch.full_like(argp[j], i), argp[j])\\n"\n'
        '    "    return outp.permute(inv), argp.permute(inv)\\n\\n"\n'
        "    \"def scatter(src, index, dim=0, out=None, dim_size=None, reduce='sum'):\\n\"\n"
        "    \"    if reduce in {'sum','add'}: return scatter_add(src,index,dim=dim,out=out,dim_size=dim_size)\\n\"\n"
        "    \"    if reduce=='mean':\\n\"\n"
        '    "        outv = scatter_add(src,index,dim=dim,out=out,dim_size=dim_size)\\n"\n'
        '    "        cnt = scatter_add(torch.ones_like(src),index,dim=dim,out=None,dim_size=dim_size).clamp(min=1)\\n"\n'
        '    "        return outv/cnt\\n"\n'
        "    \"    if reduce=='max': return scatter_max(src,index,dim=dim,out=out,dim_size=dim_size)\\n\"\n"
        "    \"    raise ValueError('unsupported reduce')\\n\"\n"
        ")\n"
        "print('torch_scatter_fallback_installed', pkg)\n"
        "PY\n"
    )


def fix_node(state: dict[str, Any]) -> dict[str, Any]:
    cfg = state.get("config", {})
    run_info = state.get("run", {})
    run_dir = Path(run_info.get("dir") or "")
    logs_dir = Path(run_info.get("logs_dir") or (run_dir / "logs"))
    fixes_dir = ensure_dir(run_info.get("fixes_dir") or (run_dir / "fixes"))

    attempt = int(state.get("attempt") or 0) + 1
    state["attempt"] = attempt
    max_attempts = int(state.get("max_attempts") or cfg.get("max_attempts") or 5)

    # Stop condition
    if attempt > max_attempts:
        state["status"] = "failed"
        append_event(
            run_dir,
            "fix_stop",
            {"reason": "max_attempts_exceeded", "attempt": attempt, "max_attempts": max_attempts},
        )
        state.setdefault("history", []).append(
            {"kind": "fix_stop", "data": {"attempt": attempt, "max_attempts": max_attempts}}
        )
        return state

    run_result = state.get("run_result") or {}
    stderr_tail = str(run_result.get("stderr_tail") or "")
    stdout_tail = str(run_result.get("stdout_tail") or "")
    failed_task = run_result.get("failed_task")

    paper_root = (cfg.get("paper_root") or ".").strip() or "."
    docker_enabled = bool(cfg.get("docker_enabled", True))
    python_spec = str(cfg.get("python_spec") or "3.11").strip()

    append_event(run_dir, "fix_start", {"attempt": attempt, "failed_task": failed_task})
    state.setdefault("history", []).append(
        {"kind": "fix_start", "data": {"attempt": attempt, "failed_task": failed_task}}
    )

    missing = _extract_missing_module(stderr_tail)

    # Deterministic quick-fix: smoke task points to a missing script (common when we used a generic template).
    missing_file = _extract_missing_file(stderr_tail)
    if missing_file:
        # If the missing file is a known entrypoint name, rewrite tasks.yaml to use an existing entry.
        missing_name = Path(missing_file).name.lower()
        if missing_name in {"launcher.py", "run.py", "eval.py", "main.py", "app.py"}:
            entry = _pick_smoke_entrypoint(paper_root)
            if entry and entry.lower() != missing_name:
                try:
                    tasks_path = str(cfg.get("tasks_path") or "").strip()
                    if tasks_path and Path(tasks_path).exists():
                        txt = Path(tasks_path).read_text(encoding="utf-8", errors="ignore")
                        write_text(fixes_dir / f"fix_{attempt:03d}_tasks_before.txt", txt)
                        # Best-effort: replace the missing script with the discovered one.
                        patched = txt.replace(missing_name, entry)
                        if patched != txt:
                            write_text(Path(tasks_path), patched)
                            write_text(
                                fixes_dir / f"fix_{attempt:03d}_tasks_patch_entrypoint.txt",
                                f"Patched tasks file to fix missing entrypoint:\n- path: {tasks_path}\n- missing: {missing_name}\n- using: {entry}\n",
                            )
                            append_event(
                                run_dir,
                                "fix_edit_tasks_deterministic",
                                {"path": tasks_path, "ok": True, "missing": missing_name, "using": entry},
                            )
                            state.setdefault("history", []).append(
                                {
                                    "kind": "fix_edit_tasks_deterministic",
                                    "data": {"path": tasks_path, "missing": missing_name, "using": entry},
                                }
                            )
                            state["status"] = "running"
                            return state
                except Exception:
                    pass

    if missing:
        append_event(run_dir, "fix_missing_module", {"module": missing})
        state.setdefault("history", []).append({"kind": "fix_missing_module", "data": {"module": missing}})

    # Deterministic fix: missing torch_scatter.
    # Prefer installing in-container (wheel index), then fallback injection module. Avoid editing paper code.
    if missing == "torch_scatter" and docker_enabled:
        shell = (
            "set -e\n"
            "TV=$(python -c \"import torch; print((torch.__version__ or '').split('+')[0])\" 2>/dev/null || true)\n"
            'if [ -n "$TV" ]; then\n'
            "  python -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-${TV}+cpu.html -f https://data.pyg.org/whl/torch-${TV}.html || true\n"
            "fi\n"
            "python -c \"import torch_scatter; print('torch_scatter_ok')\" || true\n"
        )
        shell = (
            shell
            + "\n"
            + _torch_scatter_fallback_in_container_shell()
            + "\npython -c \"import torch_scatter; print('torch_scatter_ok_after_fallback')\""
        )
        ok_img, img_or_msg = docker_ensure_paper_image(
            cfg,
            paper_key=str(cfg.get("paper_key") or "paper"),
            paper_root_host=str(Path(paper_root).resolve()),
            python_spec=python_spec,
            timeout_sec=3600,
        )
        if ok_img:
            docker_cmd = docker_run_paper_image(
                image=img_or_msg,
                paper_root_host=str(Path(paper_root).resolve()),
                run_dir_host=str(run_dir),
                cwd_container="/app",
                cmd=["bash", "-lc", shell],
                env={},
            )
            res = run_command(cmd=docker_cmd, cwd=str(run_dir), timeout_sec=900)
            persist_command_result(res, logs_dir, prefix=f"fix_torch_scatter_{attempt}")
            if res.returncode == 0:
                append_event(run_dir, "fix_install_torch_scatter", {"ok": True, "strategy": "paper_image"})
                state.setdefault("history", []).append(
                    {"kind": "fix_install_torch_scatter", "data": {"ok": True, "strategy": "paper_image"}}
                )
                state["status"] = "running"
                return state

    # LLM triage: propose a fix plan (default enabled)
    if bool(cfg.get("no_llm")):
        state["status"] = "failed"
        append_event(run_dir, "fix_no_llm", {"reason": "no_deterministic_fix_matched"})
        state.setdefault("history", []).append(
            {"kind": "fix_no_llm", "data": {"reason": "no_deterministic_fix_matched"}}
        )
        return state

    llm_cfg = resolve_llm_config(
        cfg.get("llm_provider") or "", cfg.get("llm_model") or "", cfg.get("llm_base_url") or ""
    )
    system = (
        "You are a senior engineer doing rigorous paper-code reproduction.\n"
        "Produce a fix plan ONLY in JSON. Do not include prose outside JSON.\n"
        "The plan must be safe and reproducible, prefer environment/command fixes before source edits.\n"
    )
    prompt = {
        "attempt": attempt,
        "paper_root": paper_root,
        "failed_task": failed_task,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "constraints": {
            "prefer_wrapper_env_fixes": True,
            "avoid_core_source_changes": True,
            "must_be_reproducible": True,
        },
        "output_schema": {
            "category": "env|deps|path|encoding|data|runtime|other",
            "root_cause": "short string",
            "actions": [
                {"type": "command", "cmd": ["..."], "cwd": ".", "timeout_sec": 600, "why": "short"},
                {"type": "edit", "path": "relative/path", "content": "full new file content", "why": "short"},
            ],
            "confidence": 0.0,
        },
    }
    plan = llm_json(prompt=str(prompt), system=system, cfg=llm_cfg)
    write_text(
        fixes_dir / f"fix_{attempt:03d}_plan.json",
        __import__("json").dumps(plan, ensure_ascii=False, indent=2) + "\n",
    )
    append_event(run_dir, "fix_plan", {"plan": plan})
    state.setdefault("history", []).append({"kind": "fix_plan", "data": {"plan": plan}})

    # If LLM call failed, stop gracefully with a helpful record.
    if isinstance(plan, dict) and plan.get("status") == "error":
        state["status"] = "failed"
        append_event(
            run_dir,
            "fix_llm_error",
            {"error": plan.get("error"), "provider": plan.get("provider"), "model": plan.get("model")},
        )
        state.setdefault("history", []).append(
            {
                "kind": "fix_llm_error",
                "data": {
                    "error": plan.get("error"),
                    "provider": plan.get("provider"),
                    "model": plan.get("model"),
                },
            }
        )
        write_text(
            fixes_dir / f"fix_{attempt:03d}_llm_error.txt",
            "LLM call failed during fix triage.\n"
            f"provider: {plan.get('provider')}\n"
            f"model: {plan.get('model')}\n"
            f"error: {plan.get('error')}\n"
            "\n"
            "How to proceed:\n"
            "- rerun with --no-llm to disable LLM fixes, OR\n"
            "- set a valid provider/model in env (MODEL_PROVIDER / API_KEY / MODEL), OR\n"
            "- set MODEL_PROVIDER=openai-codex after `codex login`, OR\n"
            "- set OPENAI_MODEL to a model available to your account.\n",
        )
        return state

    # Apply only safe "command" actions automatically; record others for manual review.
    actions = plan.get("actions") if isinstance(plan, dict) else None
    applied_any = False
    if isinstance(actions, list):
        for j, act in enumerate(actions, 1):
            if not isinstance(act, dict):
                continue
            if act.get("type") == "command":
                cmd = act.get("cmd")
                if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
                    continue
                cwd = str(act.get("cwd") or "").strip()
                if not cwd or cwd in {".", "./"}:
                    cwd = paper_root or "."
                timeout = int(act.get("timeout_sec") or 600)
                if docker_enabled:
                    ok_img, img_or_msg = docker_ensure_paper_image(
                        cfg,
                        paper_key=str(cfg.get("paper_key") or "paper"),
                        paper_root_host=str(Path(paper_root).resolve()),
                        python_spec=python_spec,
                        timeout_sec=3600,
                    )
                    if not ok_img:
                        continue
                    # Per-paper image mode: run fixes inside /app.
                    # Avoid conda-specific wrappers.
                    shell = _to_shell(cmd)
                    docker_cmd = docker_run_paper_image(
                        image=img_or_msg,
                        paper_root_host=str(Path(paper_root).resolve()),
                        run_dir_host=str(run_dir),
                        cwd_container="/app",
                        cmd=["bash", "-lc", shell],
                        env={},
                    )
                    res = run_command(cmd=docker_cmd, cwd=str(run_dir), timeout_sec=timeout)
                else:
                    argv = _normalize_llm_cmd_for_platform(cmd)
                    res = run_command(cmd=argv, cwd=cwd, timeout_sec=timeout)
                persist_command_result(res, logs_dir, prefix=f"fix_cmd_{attempt}_{j}")
                ok = res.returncode == 0
                append_event(run_dir, "fix_command", {"cmd": cmd, "cwd": cwd, "ok": ok, "rc": res.returncode})
                state.setdefault("history", []).append(
                    {"kind": "fix_command", "data": {"cmd": cmd, "cwd": cwd, "ok": ok, "rc": res.returncode}}
                )
                applied_any = applied_any or ok
            elif act.get("type") == "edit":
                # Safety: only allow editing the tasks file (wrapper config), not paper code.
                tasks_path = str(cfg.get("tasks_path") or "").strip()
                path = str(act.get("path") or "").strip()
                content = act.get("content")
                if not tasks_path or not path or not isinstance(content, str):
                    continue
                try:
                    # allow absolute or relative path that resolves to the tasks_path
                    target = Path(path)
                    if not target.is_absolute():
                        target = Path(paper_root) / target
                    if str(target.resolve()).lower() != str(Path(tasks_path).resolve()).lower():
                        continue
                    write_text(target, content)
                    write_text(
                        fixes_dir / f"fix_{attempt:03d}_edit_tasks.txt", f"Edited tasks file: {tasks_path}\n"
                    )
                    append_event(run_dir, "fix_edit_tasks", {"path": tasks_path, "ok": True})
                    state.setdefault("history", []).append(
                        {"kind": "fix_edit_tasks", "data": {"path": tasks_path}}
                    )
                    applied_any = True
                except Exception:
                    continue

    # If nothing applied, stop (we still recorded the plan).
    if not applied_any:
        state["status"] = "failed"
        append_event(run_dir, "fix_not_applied", {"reason": "no_applicable_actions"})
        state.setdefault("history", []).append(
            {"kind": "fix_not_applied", "data": {"reason": "no_applicable_actions"}}
        )
        return state

    state["status"] = "running"
    return state
