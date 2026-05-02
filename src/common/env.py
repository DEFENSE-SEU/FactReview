from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.strip()


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = str(line or "").strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not _ENV_KEY_RE.match(key):
        return None

    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_file(env_path: str | Path, *, override: bool = False) -> dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        loaded[key] = value
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return loaded
