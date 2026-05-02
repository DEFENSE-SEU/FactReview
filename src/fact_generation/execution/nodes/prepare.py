from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from preprocessing.parse.mineru import extract_with_mineru, mineru_available
from util.fs import copy_file_if_exists, ensure_dir, write_text
from util.paper_input import infer_paper_key, is_url, materialize_paper_pdf
from util.recorder import append_event
from util.run_layout import build_run_dir, ensure_run_subdirs, make_run_id, slugify_run_key
from util.subprocess_runner import persist_command_result, run_command

from ..tools.docker import docker_ensure_paper_image, docker_strategy


def _repo_root() -> Path:
    """Return the FactReview repository root (where ``demos/`` and ``runs/`` live)."""
    return Path(__file__).resolve().parents[4]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _write_yaml_or_json(path: Path, data: Any) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        write_text(path, text)
        return
    except Exception:
        pass
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _task_risk_level(task: dict[str, Any]) -> str:
    """
    Heuristic task risk classification for auditability.
    - smoke: fast, no real training/data downloads
    - heavy: likely training / large downloads / long runtimes
    - unknown: can't tell
    """
    cmd = task.get("cmd")
    if not isinstance(cmd, list):
        return "unknown"
    s = " ".join([str(x) for x in cmd]).lower()
    timeout = int(task.get("timeout_sec") or 0)
    if "--help" in s or " -h" in s or "print('ok')" in s or 'print("ok")' in s:
        return "smoke"
    heavy_tokens = [
        "train",
        "finetune",
        "fine-tune",
        "download",
        "wget",
        "curl",
        "pip install",
        "conda install",
        "make",
    ]
    if any(t in s for t in heavy_tokens):
        return "heavy"
    if timeout >= 3600:
        return "heavy"
    return "unknown"


def _write_tasks_risk_report(tasks_path: Path, logs_dir: Path) -> None:
    try:
        import yaml  # type: ignore

        raw = tasks_path.read_text(encoding="utf-8", errors="ignore")
        tasks = yaml.safe_load(raw)
        if not isinstance(tasks, list):
            return
        report = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            report.append(
                {
                    "id": str(t.get("id") or ""),
                    "enabled": bool(t.get("enabled", True)),
                    "timeout_sec": int(t.get("timeout_sec") or 0),
                    "risk": _task_risk_level(t),
                    "cmd": t.get("cmd"),
                }
            )
        write_text(
            logs_dir / "tasks_risk_report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    except Exception:
        return


def _parse_requirements_pins(req_text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (req_text or "").splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        if "==" not in s:
            continue
        name, ver = s.split("==", 1)
        name = name.strip()
        ver = ver.strip()
        if name and ver:
            pins[name] = ver
    return pins


def _infer_python_spec_from_requirements(req_path: Path) -> str:
    txt = _read_text(req_path) if req_path.exists() else ""
    pins = _parse_requirements_pins(txt)
    torch_ver = pins.get("torch") or pins.get("pytorch") or ""
    if torch_ver.startswith("1.4.") or torch_ver == "1.4.0":
        return "3.7"
    # Conservative: old numpy pins often imply Python <= 3.7 for many research repos.
    numpy_ver = pins.get("numpy") or ""
    if numpy_ver.startswith("1.16.") or numpy_ver.startswith("1.17."):
        return "3.7"
    return "3.11"


def _extract_repo_urls_from_pdf(pdf_path: Path, max_pages: int = 8) -> list[str]:
    """
    Extract GitHub repository URLs from a PDF using text extraction.
    Best-effort: returns candidates ordered by first appearance.
    """
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        texts: list[str] = []
        for page in reader.pages[: max_pages or 1]:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(texts)
    except Exception:
        text = ""

    if not text:
        return []

    pat = re.compile(r"(https?://)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", flags=re.IGNORECASE)
    seen = set()
    out: list[str] = []
    for m in pat.finditer(text):
        raw = (m.group(0) or "").strip()
        raw = raw.rstrip(").,;:]}'\"")
        if not raw:
            continue
        if not raw.lower().startswith("http"):
            raw = "https://" + raw
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def _copy_tree(src: Path, dst: Path) -> None:
    """
    Copy src -> dst in a Windows-friendly way.

    On Windows, `shutil.rmtree(..., ignore_errors=True)` can silently fail (file locks),
    leaving the destination directory behind and causing `copytree` to raise FileExistsError.
    Prefer an explicit delete; if it fails, fall back to merge-copy when possible.
    """
    if dst.exists():
        try:
            shutil.rmtree(dst, ignore_errors=False)
        except Exception:
            # Best-effort fallback: merge into existing dir (Python 3.8+).
            try:
                shutil.copytree(src, dst, ignore=_copy_ignore_patterns(src), dirs_exist_ok=True)
                return
            except Exception:
                # Re-raise the original intent: caller will record copy_source_failed.
                raise
    shutil.copytree(src, dst, ignore=_copy_ignore_patterns(src), dirs_exist_ok=True)


def _copy_ignore_patterns(src_root: Path):
    src_root = src_root.resolve()
    recursive_ignored = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "wandb",
    }
    root_generated_ignored = {
        "runs",
        "outputs",
        "output",
        "checkpoints",
        "checkpoint",
        "logs",
        "log",
    }

    def ignore(current: str, names: list[str]) -> set[str]:
        ignored = set(names).intersection(recursive_ignored)
        try:
            if Path(current).resolve() == src_root:
                ignored.update(set(names).intersection(root_generated_ignored))
        except Exception:
            pass
        return ignored

    return ignore


def _configured_demo_dir(paper_key: str) -> Path | None:
    """Locate a bundled demo fixture for a paper key."""
    raw_key = str(paper_key or "paper").strip()
    keys: list[str] = []
    for key in (raw_key, slugify_run_key(raw_key)):
        if key and key not in keys:
            keys.append(key)
    candidates = []
    for key in keys:
        candidates.append(_repo_root() / "demos" / key)
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def _select_demo_source(demo_dir: Path | None) -> Path | None:
    if demo_dir is None:
        return None
    for candidate in (demo_dir / "execution" / "repo",):
        if candidate.exists() and candidate.is_dir() and any(candidate.iterdir()):
            return candidate.resolve()
    return None


def _materialize_demo_fixture(demo_dir: Path | None, baseline_dir: Path) -> None:
    if demo_dir is None:
        return
    execution_dir = demo_dir / "execution"

    def _copy_first(dst_name: str, candidates: tuple[Path, ...]) -> None:
        dst = baseline_dir / dst_name
        if dst.exists():
            return
        for src in candidates:
            if copy_file_if_exists(src, dst):
                return

    _copy_first(
        "tasks.yaml",
        (
            execution_dir / "tasks.yaml",
            execution_dir / "tasks.yml",
            demo_dir / "tasks.yaml",
            demo_dir / "tasks.yml",
        ),
    )
    _copy_first(
        "baseline.json",
        (
            execution_dir / "checks.json",
            execution_dir / "baseline.json",
            demo_dir / "baseline.json",
        ),
    )
    copy_file_if_exists(demo_dir / "paper.pdf", baseline_dir / "paper.pdf")


def _copy_prepared_extract(prepared_extract_dir: str, baseline_dir: Path) -> str:
    token = str(prepared_extract_dir or "").strip()
    if not token:
        return ""
    src = Path(token).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        return ""
    dst = baseline_dir / "paper_extracted"
    if src.resolve() != dst.resolve():
        _copy_tree(src, dst)
    md = dst / "paper.mineru.md"
    return str(md.resolve()) if md.exists() else ""


def _ensure_default_baseline(baseline_path: Path) -> None:
    if baseline_path.exists():
        return
    write_text(baseline_path, json.dumps({"checks": []}, ensure_ascii=False, indent=2) + "\n")


def _git_reset_if_possible(repo_root: Path, logs_dir: Path) -> None:
    """
    Keep the repo folder reusable without carrying local patches across runs.
    If it is a git repo, reset to HEAD and clean untracked files.
    """
    if not (repo_root / ".git").exists():
        return
    try:
        r1 = run_command(["git", "reset", "--hard"], cwd=str(repo_root), timeout_sec=120)
        persist_command_result(r1, logs_dir, prefix="git_reset")
        r2 = run_command(["git", "clean", "-fd"], cwd=str(repo_root), timeout_sec=120)
        persist_command_result(r2, logs_dir, prefix="git_clean")
    except Exception:
        pass


def _git_head_sha(repo_root: Path) -> str:
    if not (repo_root / ".git").exists():
        return ""
    try:
        r = run_command(["git", "rev-parse", "HEAD"], cwd=str(repo_root), timeout_sec=30)
        if r.returncode != 0:
            return ""
        return (r.stdout or "").strip().splitlines()[0].strip()
    except Exception:
        return ""


def _write_run_manifest(*, run_dir: Path, cfg: dict[str, Any], baseline_dir: Path) -> None:
    """
    Write a compact, deterministic manifest for auditability and cross-run comparison.
    This intentionally duplicates some fields from meta.json, but adds paper/baseline pointers.
    """
    try:
        paper_key = str(cfg.get("paper_key") or "paper")
        paper_root = str(cfg.get("paper_root") or "")
        manifest = {
            "paper_key": paper_key,
            "paper_pdf": str(cfg.get("paper_pdf") or ""),
            "paper_repo_url": str(cfg.get("paper_repo_url") or ""),
            "paper_root": paper_root,
            "paper_git_head": _git_head_sha(Path(paper_root)) if paper_root else "",
            "paper_extracted": {
                "md_path": str(cfg.get("paper_pdf_extracted_md") or ""),
                "tables_dir": str((baseline_dir / "paper_extracted" / "tables").resolve()),
            },
            "wrapper_config": {
                "tasks_path": str(cfg.get("tasks_path") or ""),
                "baseline_path": str(cfg.get("baseline_path") or ""),
            },
            "docker": {
                "enabled": bool(cfg.get("docker_enabled", True)),
                "strategy": str(cfg.get("docker_strategy") or ""),
                "python_spec": str(cfg.get("python_spec") or ""),
                "paper_image": str(cfg.get("docker_paper_image") or ""),
                "gpus": str(cfg.get("docker_gpus") or os.environ.get("EXECUTION_DOCKER_GPUS") or ""),
                "shm_size": str(
                    cfg.get("docker_shm_size") or os.environ.get("EXECUTION_DOCKER_SHM_SIZE") or ""
                ),
                "ipc": str(cfg.get("docker_ipc") or os.environ.get("EXECUTION_DOCKER_IPC") or ""),
            },
            "llm": {
                "no_llm": bool(cfg.get("no_llm")),
                "provider": str(cfg.get("llm_provider") or ""),
                "model": str(cfg.get("llm_model") or ""),
                "base_url": str(cfg.get("llm_base_url") or ""),
                "judge_mode": str(cfg.get("llm_judge_mode") or ""),
            },
        }
        write_text(run_dir / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        return


def prepare_node(state: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = state.get("config", {}) or {}
    run_root = str(cfg.get("run_root") or (_repo_root() / "runs" / "execution"))

    paper_pdf = str(cfg.get("paper_pdf") or "").strip()
    paper_root_in = str(cfg.get("paper_root") or "").strip()
    paper_key = str(cfg.get("paper_key") or "").strip()
    local_source_path = str(cfg.get("local_source_path") or "").strip()
    no_pdf_extract = bool(cfg.get("no_pdf_extract"))
    dry_run = bool(cfg.get("dry_run"))
    strategy = docker_strategy(cfg)

    pdf_path = Path(paper_pdf).resolve() if (paper_pdf and not is_url(paper_pdf)) else None

    if not paper_key:
        if paper_pdf:
            paper_key = infer_paper_key(paper_pdf)
        elif paper_root_in:
            paper_key = Path(paper_root_in).resolve().name
        else:
            paper_key = "paper"

    run_id = str(cfg.get("run_id") or "").strip() or make_run_id()
    run_dir = (
        Path(str(cfg.get("run_dir") or "")).resolve()
        if str(cfg.get("run_dir") or "").strip()
        else build_run_dir(run_root, paper_key, run_id)
    )
    layout = ensure_run_subdirs(run_dir)
    logs_dir = ensure_dir(layout["logs"])
    artifacts_dir = ensure_dir(layout["artifacts"])
    fixes_dir = ensure_dir(run_dir / "fixes")
    inputs_dir = ensure_dir(layout["inputs"])
    workspace_dir = ensure_dir(layout["workspace"])

    state["run"] = {
        "id": run_id,
        "dir": str(run_dir),
        "logs_dir": str(logs_dir),
        "artifacts_dir": str(artifacts_dir),
        "fixes_dir": str(fixes_dir),
    }

    paper_pdf_source = paper_pdf
    if paper_pdf:
        try:
            materialized = materialize_paper_pdf(
                paper_pdf,
                inputs_dir / "source_pdf",
                paper_key=paper_key,
            )
            paper_pdf = str(materialized.path)
            pdf_path = materialized.path
        except Exception as exc:
            msg = f"paper_pdf_unavailable: {type(exc).__name__}: {exc}"
            append_event(
                run_dir,
                "prepare_error",
                {"error": msg, "paper_pdf": paper_pdf_source},
            )
            state.setdefault("history", []).append({"kind": "prepare_error", "data": {"error": msg}})
            state["status"] = "failed"
            return state

    append_event(
        run_dir,
        "prepare_start",
        {
            "paper_key": paper_key,
            "paper_pdf": paper_pdf,
            "paper_pdf_source": paper_pdf_source,
            "paper_root": paper_root_in,
        },
    )
    state.setdefault("history", []).append(
        {"kind": "prepare_start", "data": {"paper_key": paper_key, "paper_pdf": paper_pdf}}
    )

    demo_dir = _configured_demo_dir(paper_key)
    baseline_dir = (
        Path(str(cfg.get("baseline_dir") or "")).resolve()
        if str(cfg.get("baseline_dir") or "").strip()
        else (inputs_dir / "baseline" / slugify_run_key(paper_key)).resolve()
    )
    ensure_dir(baseline_dir)
    _materialize_demo_fixture(demo_dir, baseline_dir)

    source_dir = (workspace_dir / "source").resolve()
    demo_source_dir = _select_demo_source(demo_dir)

    if paper_root_in:
        source_origin = Path(paper_root_in).resolve()
    elif local_source_path:
        source_origin = Path(local_source_path).resolve()
    elif demo_source_dir is not None:
        source_origin = demo_source_dir
    else:
        source_origin = source_dir

    if pdf_path and pdf_path.exists():
        try:
            dst_pdf = baseline_dir / "paper.pdf"
            if not dst_pdf.exists():
                shutil.copy2(pdf_path, dst_pdf)
        except Exception:
            pass

    if paper_root_in or local_source_path or demo_source_dir is not None:
        if not source_origin.exists():
            msg = f"source_not_found: {source_origin}"
            append_event(run_dir, "prepare_error", {"error": msg})
            state.setdefault("history", []).append({"kind": "prepare_error", "data": {"error": msg}})
            state["status"] = "failed"
            return state
        try:
            _copy_tree(source_origin, source_dir)
        except Exception as exc:
            msg = f"copy_source_failed: {type(exc).__name__}: {exc}"
            append_event(
                run_dir,
                "prepare_error",
                {"error": msg, "source": str(source_origin), "dest": str(source_dir)},
            )
            state.setdefault("history", []).append({"kind": "prepare_error", "data": {"error": msg}})
            state["status"] = "failed"
            return state
        paper_root = source_dir.resolve()
        append_event(
            run_dir,
            "prepare_source_snapshot",
            {
                "source": str(source_origin),
                "dest": str(paper_root),
                "demo_fixture": str(demo_dir or ""),
            },
        )
        state.setdefault("history", []).append(
            {
                "kind": "prepare_source_snapshot",
                "data": {"source": str(source_origin), "dest": str(paper_root)},
            }
        )
    else:
        paper_root = source_dir.resolve()
        need_clone = (not source_dir.exists()) or (not any(source_dir.iterdir()))
        if need_clone:
            repo_url = str(cfg.get("paper_repo_url") or "").strip()
            candidates: list[str] = []
            if not repo_url and pdf_path and pdf_path.exists():
                candidates = _extract_repo_urls_from_pdf(pdf_path)
                write_text(
                    logs_dir / "repo_url_candidates.txt", "\n".join(candidates) + ("\n" if candidates else "")
                )
                repo_url = candidates[0] if candidates else ""

            if not repo_url:
                msg = "repo_url_not_found"
                append_event(run_dir, "prepare_error", {"error": msg})
                state.setdefault("history", []).append(
                    {"kind": "prepare_error", "data": {"error": msg, "candidates": candidates}}
                )
                state["status"] = "failed"
                return state

            ensure_dir(source_dir.parent)
            if source_dir.exists():
                shutil.rmtree(source_dir, ignore_errors=True)
            clone_cmd = ["git", "clone", "--depth", "1", repo_url, str(source_dir)]
            res = run_command(cmd=clone_cmd, cwd=str(baseline_dir), timeout_sec=3600)
            persist_command_result(res, logs_dir, prefix="clone")
            if res.returncode != 0:
                msg = "git_clone_failed"
                append_event(
                    run_dir, "prepare_error", {"error": msg, "repo_url": repo_url, "rc": res.returncode}
                )
                state.setdefault("history", []).append(
                    {"kind": "prepare_error", "data": {"error": msg, "repo_url": repo_url}}
                )
                state["status"] = "failed"
                return state
            cfg["paper_repo_url"] = repo_url
            append_event(run_dir, "prepare_clone_ok", {"repo_url": repo_url, "dest": str(source_dir)})

    _git_reset_if_possible(paper_root, logs_dir)

    prepared_md = _copy_prepared_extract(str(cfg.get("paper_extracted_dir") or ""), baseline_dir)
    if prepared_md:
        cfg["paper_pdf_extracted_md"] = prepared_md
        append_event(run_dir, "pdf_extract_reuse_pipeline_snapshot", {"output_md": prepared_md})

    if (not no_pdf_extract) and pdf_path and pdf_path.exists():
        out_dir = baseline_dir / "paper_extracted"
        existing_md = out_dir / "paper.mineru.md"
        if str(cfg.get("paper_pdf_extracted_md") or "").strip():
            append_event(
                run_dir,
                "pdf_extract_reuse_configured",
                {"output_md": str(cfg.get("paper_pdf_extracted_md"))},
            )
        elif existing_md.exists():
            cfg["paper_pdf_extracted_md"] = str(existing_md)
            append_event(run_dir, "pdf_extract_reuse_existing", {"output_md": str(existing_md)})
        else:
            if not mineru_available():
                msg = "pdf_extract_required_but_mineru_unavailable"
                append_event(
                    run_dir,
                    "prepare_error",
                    {
                        "error": msg,
                        "hint": (
                            "Install MinerU and ensure `mineru` is on PATH. "
                            "Or rerun with --no-pdf-extract to bypass."
                        ),
                    },
                )
                state.setdefault("history", []).append({"kind": "prepare_error", "data": {"error": msg}})
                state["status"] = "failed"
                return state

        if "paper_pdf_extracted_md" not in cfg:
            r = extract_with_mineru(
                pdf_path=str(pdf_path), out_dir=out_dir, logs_dir=logs_dir, timeout_sec=1800
            )
            append_event(
                run_dir,
                "pdf_extract_mineru",
                {"success": r.success, "output_md": r.output_md, "note": r.note},
            )
            if not r.success:
                msg = "pdf_extract_failed"
                append_event(
                    run_dir,
                    "prepare_error",
                    {
                        "error": msg,
                        "note": r.note,
                        "stdout_log": r.stdout_log,
                        "stderr_log": r.stderr_log,
                        "command_log": r.command_log,
                    },
                )
                state.setdefault("history", []).append(
                    {"kind": "prepare_error", "data": {"error": msg, "note": r.note}}
                )
                state["status"] = "failed"
                return state
            cfg["paper_pdf_extracted_md"] = r.output_md
    else:
        if pdf_path and pdf_path.exists():
            append_event(run_dir, "pdf_extract_skipped", {"reason": "disabled"})

    python_spec = str(cfg.get("python_spec") or os.getenv("EXECUTION_PYTHON_SPEC") or "").strip()
    if not python_spec:
        python_spec = _infer_python_spec_from_requirements(paper_root / "requirements.txt")
    cfg["python_spec"] = python_spec
    cfg["docker_enabled"] = True
    cfg["docker_strategy"] = strategy

    tasks_path = str(cfg.get("tasks_path") or "").strip()
    baseline_path = str(cfg.get("baseline_path") or "").strip()
    if not tasks_path:
        tasks_path = str((baseline_dir / "tasks.yaml").resolve())
    if not baseline_path:
        baseline_path = str((baseline_dir / "baseline.json").resolve())
    cfg["tasks_path"] = tasks_path
    cfg["baseline_path"] = baseline_path

    cfg["paper_key"] = paper_key
    cfg["paper_pdf"] = paper_pdf
    cfg["paper_root"] = str(paper_root)
    cfg["baseline_dir"] = str(baseline_dir)
    cfg["paper_extracted_dir"] = str((baseline_dir / "paper_extracted").resolve())
    cfg["paper_extracted_tables_dir"] = str((baseline_dir / "paper_extracted" / "tables").resolve())
    state["config"] = cfg

    if not dry_run:
        ok_img, img_or_msg = docker_ensure_paper_image(
            cfg,
            paper_key=paper_key,
            paper_root_host=str(paper_root),
            python_spec=python_spec,
            timeout_sec=3600,
        )
        if not ok_img:
            err = "docker_paper_image_build_failed"
            append_event(run_dir, "prepare_error", {"error": err, "detail": img_or_msg})
            state.setdefault("history", []).append(
                {"kind": "prepare_error", "data": {"error": err, "detail": img_or_msg}}
            )
            state["status"] = "failed"
            return state
        cfg["docker_paper_image"] = img_or_msg

    append_event(run_dir, "prepare_ok", {"paper_root": str(paper_root), "python_spec": python_spec})
    state.setdefault("history", []).append(
        {"kind": "prepare_ok", "data": {"paper_root": str(paper_root), "python_spec": python_spec}}
    )
    state["status"] = "running"
    return state
