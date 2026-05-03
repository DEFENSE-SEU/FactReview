"""URL → downloaded PDF path. Special-cases arxiv abs/pdf URLs and bare arXiv IDs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


_ARXIV_BARE = re.compile(r"^(?:arxiv:)?(?P<id>\d{4}\.\d{4,5})(v(?P<v>\d+))?$", re.IGNORECASE)
_ARXIV_URL = re.compile(
    r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{4,5})(v(?P<v>\d+))?(?:\.pdf)?",
    re.IGNORECASE,
)


def normalize(spec: str) -> tuple[str, str | None, int | None]:
    """Return (canonical_pdf_url, arxiv_id_or_None, arxiv_version_or_None)."""
    s = spec.strip()
    m = _ARXIV_BARE.match(s)
    if m:
        aid = m.group("id")
        v = int(m.group("v")) if m.group("v") else None
        return _arxiv_pdf_url(aid, v), aid, v

    m = _ARXIV_URL.match(s)
    if m:
        aid = m.group("id")
        v = int(m.group("v")) if m.group("v") else None
        return _arxiv_pdf_url(aid, v), aid, v

    return s, None, None


def _arxiv_pdf_url(arxiv_id: str, version: int | None) -> str:
    suffix = f"v{version}" if version else ""
    return f"https://arxiv.org/pdf/{arxiv_id}{suffix}.pdf"


def download(spec: str, dest_dir: Path) -> Path:
    """Download to dest_dir; returns the local PDF path."""
    url, arxiv_id, version = normalize(spec)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"arxiv_{arxiv_id}{f'v{version}' if version else ''}.pdf"
        if arxiv_id
        else _safe_filename_from_url(url)
    )
    out_path = dest_dir / name

    if out_path.exists() and out_path.stat().st_size > 0:
        logger.debug("reusing cached PDF: %s", out_path)
        return out_path

    logger.info("downloading PDF: %s -> %s", url, out_path)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
    return out_path


def _safe_filename_from_url(url: str) -> str:
    tail = url.rstrip("/").rsplit("/", 1)[-1] or "download"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", tail)
    return safe if safe.lower().endswith(".pdf") else safe + ".pdf"
