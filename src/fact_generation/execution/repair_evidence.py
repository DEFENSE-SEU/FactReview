"""Complete repair identities/raw evidence with bounded file-content memory."""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import stat
from pathlib import Path

CHUNK_BYTES = 1024 * 1024
SOURCE_BYTES = 65536
TEXT_BUDGET = 196608


class SourceReadUnavailable(ValueError):
    def __init__(self, message: str, scope: dict):
        super().__init__(message)
        self.source_read_scope = scope


def plain_path(value: Path) -> Path:
    """Reject every linked/reparse ancestor, including Windows 3.11 junctions."""
    path = Path(value).absolute()
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"linked repair/source path is forbidden: {part}")
    return path


def file_identity(path: Path, destination: Path | None = None) -> dict:
    path = plain_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"nonregular repair/source file: {path}")
    if before.st_nlink != 1:
        raise ValueError(f"linked repair/source file is forbidden: {path}")
    digest, size = hashlib.sha256(), 0
    with contextlib.ExitStack() as stack:
        source = stack.enter_context(path.open("rb"))
        target = None
        if destination is not None:
            destination = plain_path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            plain_path(destination)
            target = stack.enter_context(destination.open("xb"))
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
            if target is not None:
                target.write(chunk)
    after = plain_path(path).stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if size != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f"repair/source file changed during streaming read: {path}")
    result = {"sha256": digest.hexdigest(), "size": size}
    if destination is not None:
        result["artifact_locator"] = str(destination)
    return result


def bounded_source(path: Path) -> tuple[bytes | None, dict]:
    """No prefix can enter the complete source dictionary."""
    identity = file_identity(path)
    with plain_path(path).open("rb") as source:
        raw = source.read(SOURCE_BYTES + 1)
    if len(raw) > SOURCE_BYTES:
        return None, {**identity, "complete": False, "reason": "source_capacity"}
    if len(raw) != identity["size"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError("source changed during bounded read")
    return raw, {**identity, "complete": True}


def fresh_round(run_dir: Path, workspace: Path, number: int) -> Path:
    run_dir, workspace = plain_path(run_dir).resolve(), plain_path(workspace).resolve()
    if run_dir.is_relative_to(workspace):
        raise ValueError("repair evidence must be outside the execution workspace")
    result = plain_path(run_dir / "repair_evidence" / f"round_{number}")
    result.mkdir(parents=True, exist_ok=False)
    plain_path(result)
    return result


def inventory(workspace: Path, raw_dir: Path | None = None) -> dict[str, dict]:
    root = plain_path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repair workspace is unavailable")
    if raw_dir is not None:
        raw_dir = plain_path(raw_dir)
        if raw_dir.is_relative_to(root):
            raise ValueError("repair raw evidence must be outside workspace")
        raw_dir.mkdir(exist_ok=False)
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda exc: (_ for _ in ()).throw(exc)):
        for name in dirs:
            child = plain_path(Path(directory) / name)
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise ValueError(f"nonregular repair directory: {child}")
        for name in sorted(files):
            source = plain_path(Path(directory) / name)
            relative = source.relative_to(root).as_posix()
            result[relative] = file_identity(source, raw_dir / relative if raw_dir is not None else None)
    return result


def changes_with_raw(before: dict, after: dict, workspace: Path, raw_dir: Path) -> list[dict]:
    raw_dir = plain_path(raw_dir)
    raw_dir.mkdir(exist_ok=False)
    changes = []
    for name in sorted(before.keys() | after.keys()):
        left, right = before.get(name), after.get(name)
        if left is not None and right is not None and all(left[key] == right[key] for key in ("sha256", "size")):
            right["artifact_locator"] = left["artifact_locator"]
            continue
        if right is not None:
            copied = file_identity(Path(workspace) / name, raw_dir / name)
            if any(copied[key] != right[key] for key in ("sha256", "size")):
                raise ValueError("repair after evidence changed while being preserved")
            right.update(copied)
        changes.append({"path": name, "kind": "added" if left is None else "deleted" if right is None else "modified",
                        "before": left, "after": right})
    return changes


def bounded_diff(before: dict | None, after: dict | None, fromfile: str, tofile: str,
                 remaining: int, *, universal_newlines=False) -> tuple[str, dict]:
    sides = []
    for row in (before, after):
        if row is None:
            sides.append(b"")
            continue
        if row["size"] > SOURCE_BYTES:
            return "[text diff omitted: capacity; full raw evidence retained]", {"status": "omitted", "reason": "capacity"}
        with plain_path(Path(row["artifact_locator"])).open("rb") as source:
            raw = source.read(SOURCE_BYTES + 1)
        if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("repair raw evidence identity changed")
        if b"\x00" in raw:
            return "[text diff omitted: binary; full raw evidence retained]", {"status": "omitted", "reason": "binary"}
        sides.append(raw)
    decoded = [side.decode("utf-8", errors="replace") for side in sides]
    if universal_newlines:
        decoded = [text.replace("\r\n", "\n").replace("\r", "\n") for text in decoded]
    text = "".join(difflib.unified_diff(
        decoded[0].splitlines(keepends=True), decoded[1].splitlines(keepends=True),
        fromfile=fromfile, tofile=tofile,
    ))
    size = len(text.encode("utf-8"))
    if size > remaining:
        return "[text diff omitted: total capacity; full raw evidence retained]", {"status": "omitted", "reason": "total_capacity"}
    return text, {"status": "available", "utf8_bytes": size}


def file_diffs(changes: list[dict], *, added_from_null=False, remaining=TEXT_BUDGET) -> tuple[dict, dict]:
    diffs, display = {}, {}
    for row in changes:
        name = row["path"]
        diffs[name], display[name] = bounded_diff(
            row["before"], row["after"], "/dev/null" if added_from_null else name + ".before",
            name if added_from_null else name + ".after", remaining, universal_newlines=added_from_null,
        )
        if len(diffs[name].encode("utf-8")) > remaining:
            diffs[name] = ""  # The separate omitted status/reason remains explicit.
        remaining -= len(diffs[name].encode("utf-8"))
    return diffs, display


def request_diff(before: dict, after: dict, evidence: Path) -> tuple[str, dict, dict]:
    identities = {}
    for side, value in (("before", before), ("after", after)):
        path = plain_path(evidence / f"request.{side}.json")
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        identities[side] = {**file_identity(path), "artifact_locator": str(path)}
    diff, display = bounded_diff(identities["before"], identities["after"],
                                 "request.before.json", "request.after.json", TEXT_BUDGET)
    return diff, display, identities
