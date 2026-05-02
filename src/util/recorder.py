from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .fs import ensure_dir, write_text
from .verbose import is_verbose


@dataclass
class Event:
    ts: float
    kind: str
    data: dict[str, Any]


def _short_path(path: str | Path, keep_parts: int = 3) -> str:
    try:
        p = Path(path)
        parts = list(p.parts)
        if len(parts) <= keep_parts:
            return str(p)
        return str(Path(*parts[-keep_parts:]))
    except Exception:
        return str(path)


def _format_cmd(cmd: Any, max_chars: int = 96) -> str:
    if not isinstance(cmd, list):
        return ""
    try:
        s = " ".join(str(x) for x in cmd)
    except Exception:
        s = str(cmd)
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "..."
    return s


def _format_duration(sec: Any) -> str:
    try:
        value = float(sec)
    except Exception:
        return ""
    if value < 60:
        return f"{value:.1f}s"
    mins = int(value // 60)
    rem = int(value % 60)
    return f"{mins}m {rem}s"


def _task_progress(data: dict[str, Any]) -> str:
    idx = data.get("task_index")
    total = data.get("task_total")
    if isinstance(idx, int) and isinstance(total, int) and total > 0:
        return f"{idx}/{total}"
    return "?"


def _run_summary(results: list[dict[str, Any]]) -> str:
    total = len(results)
    ok = sum(1 for r in results if r.get("success") and not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))
    failed = sum(1 for r in results if not r.get("success"))
    return f"{ok} passed, {skipped} skipped, {failed} failed, {total} total"


def _judge_label(data: dict[str, Any]) -> str:
    results = data.get("results") or []
    if bool(data.get("passed")):
        return "passed"
    if isinstance(results, list) and any(
        isinstance(r, dict) and r.get("type") == "inconclusive_no_baseline" for r in results
    ):
        return "inconclusive"
    return "failed"


def _console_event_line(kind: str, data: dict[str, Any], run_dir: Path) -> str:
    """
    Human-friendly event summary for console tracing.
    Keep it short; detailed stdout/stderr remains in logs/ files.
    """
    run_id = run_dir.name
    if kind == "prepare_start":
        paper_key = str(data.get("paper_key") or "paper")
        return f"[START] Run {run_id} | preparing paper '{paper_key}'"
    if kind == "prepare_ok":
        paper_root = _short_path(str(data.get("paper_root") or ""))
        py = str(data.get("python_spec") or "")
        return f"[OK] Prepare complete | paper_root={paper_root} | python={py}"
    if kind == "prepare_error":
        return f"[FAIL] Prepare failed | {data.get('error') or 'unknown error'!s}"
    if kind == "plan_start":
        return "[START] Planning execution tasks"
    if kind == "tasks_keep_existing":
        return f"[OK] Keeping existing tasks file | {_short_path(str(data.get('path') or ''))}"
    if kind == "tasks_written":
        return (
            f"[OK] Generated {int(data.get('count') or 0)} tasks | {_short_path(str(data.get('path') or ''))}"
        )
    if kind == "tasks_patch_disable_install_deps":
        return "[OK] Disabled runtime pip install tasks for Docker image mode"
    if kind == "tasks_persist_run_dir":
        return f"[OK] Snapshotted tasks into run directory | {_short_path(str(data.get('path') or ''))}"
    if kind == "plan_ok":
        return "[OK] Planning complete"
    if kind == "task_start":
        progress = _task_progress(data)
        task = str(data.get("task") or "task")
        cmd = _format_cmd(data.get("cmd"))
        return f"[RUN {progress}] {task} | {cmd}"
    if kind == "task_skipped":
        progress = _task_progress(data)
        task = str(data.get("task") or "task")
        reason = str(data.get("reason") or "skipped")
        return f"[SKIP {progress}] {task} | {reason}"
    if kind == "task_done":
        progress = _task_progress(data)
        task = str(data.get("task") or "task")
        if bool(data.get("dry_run")):
            return f"[OK {progress}] {task} | dry run"
        if bool(data.get("success")):
            dur = _format_duration(data.get("duration_sec"))
            extra = f" | {dur}" if dur else ""
            return f"[OK {progress}] {task}{extra}"
        rc = data.get("returncode")
        dur = _format_duration(data.get("duration_sec"))
        extra = f" | rc={rc}" if rc is not None else ""
        if dur:
            extra += f" | {dur}"
        return f"[FAIL {progress}] {task}{extra}"
    if kind == "artifacts_archived":
        task = str(data.get("task") or "task")
        count = int(data.get("count") or 0)
        return f"[OK] Archived {count} artifacts from {task}"
    if kind == "run_failed":
        task = str(data.get("failed_task") or "task")
        rc = data.get("returncode")
        extra = f" | rc={rc}" if rc is not None else ""
        return f"[FAIL] Run stopped at {task}{extra}"
    if kind == "run_ok":
        tasks = data.get("tasks") or []
        if isinstance(tasks, list):
            return f"[OK] Run stage complete | {_run_summary(tasks)}"
        return "[OK] Run stage complete"
    if kind == "judge":
        label = _judge_label(data)
        if label == "passed":
            return "[OK] Validation passed"
        if label == "inconclusive":
            return "[WARN] Validation inconclusive | no baseline checks defined"
        return "[FAIL] Validation failed"
    if kind == "finalize":
        report = _short_path(str(data.get("report") or ""))
        return f"[OK] Final report written | {report}"

    try:
        payload = json.dumps(data or {}, ensure_ascii=False)
    except Exception:
        payload = str(data)
    if len(payload) > 240:
        payload = payload[:240] + "...(truncated)"
    return f"[INFO] {kind} | {payload}"


def append_event(run_dir: str | Path, kind: str, data: dict[str, Any]) -> None:
    d = ensure_dir(run_dir)
    ev = Event(ts=time.time(), kind=kind, data=data)
    path = d / "issues.jsonl"
    line = json.dumps(asdict(ev), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if is_verbose():
        try:
            print(_console_event_line(kind, data, d), flush=True)
        except Exception:
            # never break workflow due to console printing issues
            pass


def write_issues_md(run_dir: str | Path, history: list[dict[str, Any]]) -> None:
    """
    Human-readable issue narrative. The 'history' is state-managed so it is always reproducible.
    """
    d = Path(run_dir)
    lines: list[str] = []
    lines.append("# Run Issues & Fix Log")
    lines.append("")

    # Prefer the event stream (issues.jsonl) because it provides a step-by-step timeline
    # including prepare sub-steps (clone/env) and detailed errors.
    events_path = d / "issues.jsonl"
    events: list[dict[str, Any]] = []
    if events_path.exists():
        try:
            for raw in events_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    events.append(json.loads(raw))
                except Exception:
                    continue
        except Exception:
            events = []

    # Quick summary
    if events:
        last = events[-1]
        last_kind = (last.get("kind") or "").strip()
        lines.append("## Summary")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "last_event": last_kind,
                    "last_event_data": last.get("data", {}),
                    "hint": "See logs/ for detailed command stdout/stderr. If a task failed, check the logs paths in run_failed.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("```")
        lines.append("")

    # Timeline
    if events:
        for i, ev in enumerate(events, 1):
            lines.append(f"## Step {i}: {ev.get('kind', 'event')}")
            lines.append("")
            payload = ev.get("data", {})
            lines.append("```json")
            lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    else:
        # Fallback: state history (older runs / tests)
        for i, step in enumerate(history, 1):
            lines.append(f"## Step {i}: {step.get('kind', 'event')}")
            lines.append("")
            payload = step.get("data", {})
            lines.append("```json")
            lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

    write_text(d / "issues.md", "\n".join(lines) + "\n")
