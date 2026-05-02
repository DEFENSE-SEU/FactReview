"""Shared verbose-flag helper for the execution stage's runner + recorder.

Both modules emit human-readable progress lines when ``EXECUTION_VERBOSE`` is
set. Centralising the env-var parse keeps the truthy-value list in one place.
"""

from __future__ import annotations

import os


def is_verbose() -> bool:
    v = (os.getenv("EXECUTION_VERBOSE") or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}
