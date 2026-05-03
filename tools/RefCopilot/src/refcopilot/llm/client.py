"""Thin wrapper over FactReview's llm.client.llm_json + resolve_llm_config.

Importing FactReview's llm package is deferred until first use so that the
RefCopilot package is importable in environments where FactReview's src is not
on sys.path (e.g., unit tests using mocks).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

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
    return _LLM_JSON(prompt=prompt, system=system, cfg=cfg)  # type: ignore[misc]
