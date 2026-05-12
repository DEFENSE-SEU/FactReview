from __future__ import annotations

import contextlib
import contextvars
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except Exception:  # pragma: no cover - Windows fallback for local development.
    fcntl = None  # type: ignore[assignment]


MODULE_ORDER: tuple[str, ...] = (
    "parse",
    "analysis",
    "report_generation",
    "reference_check",
    "execution",
    "teaser_figure",
)

_ACTIVE_MODULE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "factreview_active_stats_module", default=None
)
_RUN_STATS_ENV = "FACTREVIEW_RUN_STATS_PATH"
_ACTIVE_MODULE_ENV = "FACTREVIEW_ACTIVE_STATS_MODULE"
_CLI_STATUS_ENV = "FACTREVIEW_CLI_STATUS"


def stats_path() -> Path | None:
    token = str(os.getenv(_RUN_STATS_ENV) or "").strip()
    return Path(token).expanduser().resolve() if token else None


def set_stats_path(path: Path) -> None:
    os.environ[_RUN_STATS_ENV] = str(path.expanduser().resolve())


def cli_status_enabled() -> bool:
    return str(os.getenv(_CLI_STATUS_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


def enable_cli_status() -> None:
    os.environ[_CLI_STATUS_ENV] = "1"


def log_status(message: str) -> None:
    if cli_status_enabled():
        print(message, file=sys.stderr, flush=True)


def validate_module(module: str) -> str:
    normalized = str(module or "").strip()
    if normalized not in MODULE_ORDER:
        raise ValueError(f"unknown stats module: {module!r}")
    return normalized


def current_module() -> str | None:
    value = _ACTIVE_MODULE.get()
    if value:
        return validate_module(value)
    env_value = str(os.getenv(_ACTIVE_MODULE_ENV) or "").strip()
    if env_value:
        return validate_module(env_value)
    return None


@contextlib.contextmanager
def module_scope(module: str) -> Iterator[None]:
    normalized = validate_module(module)
    token = _ACTIVE_MODULE.set(normalized)
    prior_env = os.environ.get(_ACTIVE_MODULE_ENV)
    os.environ[_ACTIVE_MODULE_ENV] = normalized
    try:
        yield
    finally:
        _ACTIVE_MODULE.reset(token)
        if prior_env is None:
            os.environ.pop(_ACTIVE_MODULE_ENV, None)
        else:
            os.environ[_ACTIVE_MODULE_ENV] = prior_env


@contextlib.contextmanager
def timed_module(module: str, *, status: str | None = None) -> Iterator[None]:
    normalized = validate_module(module)
    start = time.monotonic()
    try:
        with module_scope(normalized):
            yield
    finally:
        record_duration(normalized, time.monotonic() - start)
        if status:
            record_module_status(normalized, status)


def _empty_token_usage() -> dict[str, int]:
    return {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_requests": 0,
    }


def _empty_module_payload(status: str = "pending") -> dict[str, Any]:
    return {
        "status": status,
        "duration_sec": 0.0,
        "llm_duration_sec": 0.0,
        "token_usage": _empty_token_usage(),
        "estimated": False,
        "providers": {},
        "models": {},
        "warnings": [],
    }


def _empty_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "modules": {module: _empty_module_payload() for module in MODULE_ORDER},
        "pipeline": {"duration_sec": 0.0},
    }


def _normalize_payload(payload: Any) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    data.setdefault("version", 1)
    modules = data.get("modules") if isinstance(data.get("modules"), dict) else {}
    data["modules"] = modules
    for module in MODULE_ORDER:
        current = modules.get(module) if isinstance(modules.get(module), dict) else {}
        base = _empty_module_payload()
        base.update(current)
        token_usage = base.get("token_usage") if isinstance(base.get("token_usage"), dict) else {}
        merged_token = _empty_token_usage()
        for key in merged_token:
            merged_token[key] = max(0, int(token_usage.get(key) or 0))
        base["token_usage"] = merged_token
        base["duration_sec"] = float(base.get("duration_sec") or 0.0)
        base["llm_duration_sec"] = float(base.get("llm_duration_sec") or 0.0)
        base["estimated"] = bool(base.get("estimated"))
        base["providers"] = base.get("providers") if isinstance(base.get("providers"), dict) else {}
        base["models"] = base.get("models") if isinstance(base.get("models"), dict) else {}
        warnings = base.get("warnings") if isinstance(base.get("warnings"), list) else []
        base["warnings"] = [str(item) for item in warnings if str(item).strip()]
        modules[module] = base
    data["pipeline"] = data.get("pipeline") if isinstance(data.get("pipeline"), dict) else {}
    data["pipeline"]["duration_sec"] = float(data["pipeline"].get("duration_sec") or 0.0)
    return data


@contextlib.contextmanager
def _locked_payload(path: Path) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    lock_fh = lock_path.open("a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = {}
        normalized = _normalize_payload(payload)
        yield normalized
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        if fcntl is not None:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        lock_fh.close()


def initialize(path: Path) -> dict[str, Any]:
    set_stats_path(path)
    payload = _empty_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def read(path: Path | None = None) -> dict[str, Any]:
    target = path or stats_path()
    if target is None or not target.exists():
        return _normalize_payload({})
    try:
        return _normalize_payload(json.loads(target.read_text(encoding="utf-8")))
    except Exception:
        return _normalize_payload({})


def update(fn) -> None:
    target = stats_path()
    if target is None:
        return
    with _locked_payload(target) as payload:
        fn(payload)


def record_module_status(module: str, status: str, *, warning: str = "") -> None:
    normalized = validate_module(module)

    def apply(payload: dict[str, Any]) -> None:
        row = payload["modules"][normalized]
        row["status"] = str(status or "").strip() or row.get("status") or "pending"
        if warning:
            _append_warning(row, warning)

    update(apply)


def record_duration(module: str, duration_sec: float) -> None:
    normalized = validate_module(module)
    duration = max(0.0, float(duration_sec or 0.0))

    def apply(payload: dict[str, Any]) -> None:
        row = payload["modules"][normalized]
        row["duration_sec"] = float(row.get("duration_sec") or 0.0) + duration

    update(apply)


def set_pipeline_duration(duration_sec: float) -> None:
    duration = max(0.0, float(duration_sec or 0.0))

    def apply(payload: dict[str, Any]) -> None:
        payload["pipeline"]["duration_sec"] = duration

    update(apply)


def _append_warning(row: dict[str, Any], warning: str) -> None:
    text = str(warning or "").strip()
    if not text:
        return
    warnings = row.setdefault("warnings", [])
    if text not in warnings:
        warnings.append(text)


def _inc_counter(mapping: dict[str, Any], key: str) -> None:
    clean = str(key or "").strip() or "unknown"
    mapping[clean] = int(mapping.get(clean) or 0) + 1


def _coerce_usage(usage: Any) -> dict[str, int]:
    raw = usage if isinstance(usage, dict) else {}
    input_tokens = raw.get("input_tokens", raw.get("prompt_tokens"))
    output_tokens = raw.get("output_tokens", raw.get("completion_tokens"))
    total_tokens = raw.get("total_tokens")
    result = {
        "input_tokens": max(0, int(input_tokens or 0)),
        "output_tokens": max(0, int(output_tokens or 0)),
        "total_tokens": max(0, int(total_tokens or 0)),
    }
    if result["total_tokens"] <= 0:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def estimate_tokens(text: str) -> int:
    clean = str(text or "")
    if not clean:
        return 0
    # Rough cross-provider approximation for auditability when a backend does
    # not return usage. The warning path always marks these totals as estimated.
    return max(1, int(math.ceil(len(clean) / 4)))


def record_llm_call(
    *,
    module: str | None = None,
    provider: str = "",
    model: str = "",
    usage: dict[str, Any] | None = None,
    prompt: str = "",
    system: str = "",
    response_text: str = "",
    duration_sec: float = 0.0,
    estimated: bool | None = None,
    warning: str = "",
) -> None:
    resolved_module = module or current_module()
    if not resolved_module:
        raise RuntimeError("LLM usage recording requires an active stats module")
    normalized = validate_module(resolved_module)

    exact_usage = _coerce_usage(usage or {})
    has_usage = any(exact_usage[key] > 0 for key in ("input_tokens", "output_tokens", "total_tokens"))
    is_estimated = (not has_usage) if estimated is None else bool(estimated)
    if is_estimated:
        exact_usage = {
            "input_tokens": estimate_tokens("\n".join(part for part in (system, prompt) if part)),
            "output_tokens": estimate_tokens(response_text),
            "total_tokens": 0,
        }
        exact_usage["total_tokens"] = exact_usage["input_tokens"] + exact_usage["output_tokens"]
        if not warning:
            warning = (
                f"Estimated token usage for provider={provider or 'unknown'} "
                f"model={model or 'unknown'} because the response did not include usage."
            )

    def apply(payload: dict[str, Any]) -> None:
        row = payload["modules"][normalized]
        token_usage = row["token_usage"]
        token_usage["requests"] += 1
        token_usage["input_tokens"] += exact_usage["input_tokens"]
        token_usage["output_tokens"] += exact_usage["output_tokens"]
        token_usage["total_tokens"] += exact_usage["total_tokens"]
        if is_estimated:
            token_usage["estimated_requests"] += 1
            row["estimated"] = True
        row["llm_duration_sec"] = float(row.get("llm_duration_sec") or 0.0) + max(
            0.0, float(duration_sec or 0.0)
        )
        _inc_counter(row["providers"], provider or "unknown")
        _inc_counter(row["models"], model or "unknown")
        if warning:
            _append_warning(row, warning)

    update(apply)


def add_token_delta(
    *,
    module: str,
    provider: str = "",
    model: str = "",
    requests: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    estimated: bool = False,
    warning: str = "",
) -> None:
    normalized = validate_module(module)
    req = max(0, int(requests or 0))
    inp = max(0, int(input_tokens or 0))
    out = max(0, int(output_tokens or 0))
    total = max(0, int(total_tokens or 0)) or inp + out
    if req == 0 and inp == 0 and out == 0 and total == 0:
        return

    def apply(payload: dict[str, Any]) -> None:
        row = payload["modules"][normalized]
        token_usage = row["token_usage"]
        token_usage["requests"] += req
        token_usage["input_tokens"] += inp
        token_usage["output_tokens"] += out
        token_usage["total_tokens"] += total
        if estimated:
            token_usage["estimated_requests"] += req
            row["estimated"] = True
        if provider:
            _inc_counter(row["providers"], provider)
        if model:
            _inc_counter(row["models"], model)
        if warning:
            _append_warning(row, warning)

    update(apply)


def with_totals(payload: dict[str, Any]) -> dict[str, Any]:
    data = _normalize_payload(payload)
    total = _empty_token_usage()
    duration = 0.0
    llm_duration = 0.0
    estimated = False
    warnings: list[str] = []
    for module in MODULE_ORDER:
        row = data["modules"][module]
        usage = row["token_usage"]
        for key in total:
            total[key] += int(usage.get(key) or 0)
        duration += float(row.get("duration_sec") or 0.0)
        llm_duration += float(row.get("llm_duration_sec") or 0.0)
        estimated = estimated or bool(row.get("estimated"))
        warnings.extend(str(item) for item in row.get("warnings") or [] if str(item).strip())
    data["total"] = {
        "duration_sec": duration,
        "llm_duration_sec": llm_duration,
        "token_usage": total,
        "estimated": estimated,
        "warnings": warnings,
    }
    return data


def format_seconds(value: Any) -> str:
    seconds = max(0.0, float(value or 0.0))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def _fmt_int(value: Any) -> str:
    return f"{max(0, int(value or 0)):,}"


def format_summary_table(payload: dict[str, Any]) -> list[str]:
    data = with_totals(payload)
    lines = [
        "Token and runtime summary:",
        "  Module              Status        Time       Requests  Input       Output      Total       Notes",
        "  ------------------  ------------  ---------  --------  ----------  ----------  ----------  -----",
    ]
    for module in MODULE_ORDER:
        row = data["modules"][module]
        usage = row["token_usage"]
        notes: list[str] = []
        if row.get("estimated"):
            notes.append("estimated")
        if module == "parse" and int(usage.get("total_tokens") or 0) == 0:
            notes.append("external/no own LLM")
        line = (
            f"  {module:<18}  "
            f"{str(row.get('status') or 'pending'):<12}  "
            f"{format_seconds(row.get('duration_sec')):<9}  "
            f"{_fmt_int(usage.get('requests')):>8}  "
            f"{_fmt_int(usage.get('input_tokens')):>10}  "
            f"{_fmt_int(usage.get('output_tokens')):>10}  "
            f"{_fmt_int(usage.get('total_tokens')):>10}  "
            f"{', '.join(notes)}"
        )
        lines.append(line.rstrip())
    total = data["total"]
    usage = total["token_usage"]
    total_notes = "estimated" if total.get("estimated") else ""
    lines.append(
        (
            f"  {'TOTAL':<18}  {'':<12}  "
            f"{format_seconds(total.get('duration_sec')):<9}  "
            f"{_fmt_int(usage.get('requests')):>8}  "
            f"{_fmt_int(usage.get('input_tokens')):>10}  "
            f"{_fmt_int(usage.get('output_tokens')):>10}  "
            f"{_fmt_int(usage.get('total_tokens')):>10}  "
            f"{total_notes}"
        ).rstrip()
    )
    warnings = total.get("warnings") or []
    if warnings:
        lines.append("  Warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
    return lines


__all__ = [
    "MODULE_ORDER",
    "add_token_delta",
    "cli_status_enabled",
    "current_module",
    "enable_cli_status",
    "estimate_tokens",
    "format_seconds",
    "format_summary_table",
    "initialize",
    "log_status",
    "module_scope",
    "read",
    "record_duration",
    "record_llm_call",
    "record_module_status",
    "set_pipeline_duration",
    "set_stats_path",
    "stats_path",
    "timed_module",
    "validate_module",
    "with_totals",
]
