"""LLM JSON-output client.

Imports FactReview's :mod:`llm.client` lazily so callers can monkey-patch
:func:`call_json` without resolving the FactReview ``src/`` path.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LLM_JSON = None
_RESOLVE_LLM_CONFIG = None
_FACTREVIEW_SRC_HINT = Path(__file__).resolve().parents[4] / "src"


def _load_factreview_llm() -> None:
    """Locate FactReview's `src/llm/client.py` and import llm_json + resolve_llm_config."""
    global _LLM_JSON, _RESOLVE_LLM_CONFIG
    if _LLM_JSON is not None and _RESOLVE_LLM_CONFIG is not None:
        return

    candidates = [_FACTREVIEW_SRC_HINT]
    env_hint = os.environ.get("REFCOPILOT_FACTREVIEW_SRC")
    if env_hint:
        candidates.insert(0, Path(env_hint))

    for candidate in candidates:
        if (candidate / "llm" / "client.py").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    try:
        from llm.client import llm_json, resolve_llm_config  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "RefCopilot requires FactReview's llm.client; "
            "set REFCOPILOT_FACTREVIEW_SRC to FactReview/src or run from FactReview workspace"
        ) from exc

    _LLM_JSON = llm_json
    _RESOLVE_LLM_CONFIG = resolve_llm_config


def call_json(prompt: str, system: str, *, provider: str = "", model: str = "") -> dict[str, Any]:
    """Run an LLM call and return a parsed JSON dict.

    On any failure, FactReview's llm_json returns a dict with `status="error"`
    or `status="unknown"`. Callers should check the `status` key before using
    the payload.
    """
    _load_factreview_llm()
    cfg = _RESOLVE_LLM_CONFIG(provider=provider, model=model)  # type: ignore[misc]
    attempts = max(1, _env_int("REFCOPILOT_LLM_RETRIES", 4))
    delay = max(0.0, _env_float("REFCOPILOT_LLM_RETRY_DELAY_SEC", 3.0))
    last: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        payload = _LLM_JSON(prompt=prompt, system=system, cfg=cfg)  # type: ignore[misc]
        if isinstance(payload, dict):
            if payload.get("status") not in ("error", "unknown"):
                return payload
            last = dict(payload)
        else:
            last = {"status": "unknown", "raw": payload}
        if attempt < attempts:
            time.sleep(delay * attempt)
    last["_retry_attempts"] = attempts
    return last


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using default %s", name, raw, default)
        return default
