from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterable
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8", errors="ignore")


def write_bytes(path: str | Path, data: bytes) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_bytes(data)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_relpath(path: str | Path, start: str | Path) -> str:
    try:
        return os.path.relpath(str(path), str(start))
    except Exception:
        return str(path)


def copy_file_if_exists(src: Path | None, dst: Path) -> bool:
    """Copy ``src`` to ``dst`` when ``src`` is an existing regular file.

    Returns ``True`` when the copy occurred, ``False`` when ``src`` was
    missing or not a file. Creates ``dst``'s parent directory as needed.
    """
    if src is None or (not src.exists()) or (not src.is_file()):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_dir_if_exists(src: Path | None, dst: Path) -> bool:
    """Copy directory tree ``src`` to ``dst`` when ``src`` is an existing dir.

    If ``dst`` already exists, it is removed first. Returns ``True`` on copy,
    ``False`` when ``src`` was missing or not a directory.
    """
    if src is None or (not src.exists()) or (not src.is_dir()):
        return False
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    return True


def copy_into(src: str | Path, dst_dir: str | Path, ignore_globs: Iterable[str] | None = None) -> Path:
    src_p = Path(src)
    dst_dir_p = ensure_dir(dst_dir)
    dst_p = dst_dir_p / src_p.name
    if src_p.is_dir():
        if dst_p.exists():
            shutil.rmtree(dst_p, ignore_errors=True)
        shutil.copytree(
            src_p,
            dst_p,
            ignore=shutil.ignore_patterns(*(ignore_globs or [])),
        )
    else:
        shutil.copy2(src_p, dst_p)
    return dst_p
