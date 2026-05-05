"""Filesystem disk cache with mtime-based TTL.

Cache key construction:
  - "arxiv:1706.03762"                → arxiv_1706.03762/
  - "https://..."                      → url_<sha256[:16]>/
  - local file path                    → file_<safe_basename>_<sha256[:8]>/
  - other (e.g. raw text)              → spec_<sha256[:16]>/

API call results land under <root>/api_cache/<source>/<key>.json.

Each backend stores RAW API responses (not pre-filtered records) so that
changes to filtering/parsing logic never require cache invalidation. The
on-disk format is bumped via ``API_CACHE_VERSION`` only when the API
response shape itself changes — old caches are auto-wiped on first use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_TTL_DAYS = 30

# Bumped only when the SHAPE of cached API payloads changes (e.g. switching
# from caching pre-filtered records to caching raw responses). Read/parse
# logic changes don't need a bump — they re-derive from the raw payload.
API_CACHE_VERSION = 2


def cache_key_for_paper(spec: str) -> str:
    s = (spec or "").strip()
    if not s:
        return "spec_empty"

    if s.lower().startswith("arxiv:"):
        return f"arxiv_{s[6:]}"

    m = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", s, re.IGNORECASE)
    if m:
        return f"arxiv_{m.group(1)}{m.group(2) or ''}"

    if s.startswith(("http://", "https://")):
        return f"url_{_short_hash(s)}"

    p = Path(s)
    if p.exists() and p.is_file():
        try:
            content_hash = _short_hash(p.read_bytes(), n=8)
        except Exception:
            content_hash = _short_hash(s, n=8)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", p.name)
        return f"file_{safe}_{content_hash}"

    return f"spec_{_short_hash(s)}"


def _short_hash(value: str | bytes, *, n: int = 16) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:n]


class DiskCache:
    def __init__(self, root: Path | str, *, ttl_days: int = _DEFAULT_TTL_DAYS, enabled: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.ttl_seconds = max(0, int(ttl_days)) * 86400
        self.enabled = enabled
        if self.enabled:
            self._ensure_compatible_api_cache()

    def paper_dir(self, spec: str) -> Path:
        d = self.root / cache_key_for_paper(spec)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_api(self, source: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._api_path(source, key)
        if not path.exists():
            return None
        if self._is_stale(path):
            logger.debug("cache stale: %s", path)
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("cache read failed: %s (%s)", path, exc)
            return None

    def set_api(self, source: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._api_path(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _ensure_compatible_api_cache(self) -> None:
        """Wipe ``api_cache/`` if its on-disk format version doesn't match.

        The marker file ``api_cache/.version`` holds the integer version that
        wrote the surrounding files. When it disagrees with the current
        ``API_CACHE_VERSION``, every cached entry is from an incompatible
        format (e.g. pre-filtered records vs raw payloads) and we wipe the
        directory so the next call refetches into the new format.
        """
        api_dir = self.root / "api_cache"
        marker = api_dir / ".version"
        if not api_dir.exists():
            api_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(API_CACHE_VERSION))
            return
        existing: int | None = None
        if marker.exists():
            try:
                existing = int(marker.read_text().strip())
            except (OSError, ValueError):
                existing = None
        if existing == API_CACHE_VERSION:
            return
        for child in api_dir.iterdir():
            if child.name == ".version":
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError as exc:
                logger.debug("could not remove stale cache entry %s: %s", child, exc)
        marker.write_text(str(API_CACHE_VERSION))
        logger.info(
            "wiped api_cache/ (was version %s, now version %s)",
            existing,
            API_CACHE_VERSION,
        )

    def _api_path(self, source: str, key: str) -> Path:
        safe_key = re.sub(r"[^A-Za-z0-9._-]+", "_", key)
        if len(safe_key) > 80:
            safe_key = safe_key[:40] + "_" + _short_hash(safe_key, n=16)
        return self.root / "api_cache" / source / f"{safe_key}.json"

    def _is_stale(self, path: Path) -> bool:
        if self.ttl_seconds <= 0:
            return False
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return True
        return (time.time() - mtime) > self.ttl_seconds

    def prune(self) -> int:
        """Delete entries older than TTL. Returns number of files removed."""
        if not self.root.exists() or self.ttl_seconds <= 0:
            return 0
        removed = 0
        for path in self.root.rglob("*.json"):
            if self._is_stale(path):
                try:
                    path.unlink()
                    removed += 1
                except OSError as exc:
                    logger.debug("could not unlink %s: %s", path, exc)
        return removed
