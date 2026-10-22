"""Preserve released-resource selections without claiming they were consumed.

Hashes are rebuilt from the original claim, independent task selections, repository
index and current file bytes. This is not an approval signature: callers must freeze
the original plan separately to reject simultaneous task/contract replacement.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from schemas.claim import Claim, ExecutionPlan, ExecutionResourceContract, ExecutionResourceSelection
from schemas.materials import SharedMaterials

_SCIENTIFIC_FIELDS = {
    "id", "text", "loc", "source_block_id", "source_quote", "source_refs",
    "conditions", "needs", "importance",
}


def _fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _relative(path: str) -> str:
    if (
        not isinstance(path, str) or not path or "\x00" in path or "\\" in path or ":" in path
        or PureWindowsPath(path).drive or PurePosixPath(path).is_absolute()
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("Resource selection requires an exact repository-relative file path")
    return path


def _actual_file(root: Path, relative: str) -> Path:
    path = root
    for part in _relative(relative).split("/"):
        path = path / part
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("Resource selection cannot follow a link or junction")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError("Selected indexed resource is missing or escapes the repository")
    return path


def _scientific_claim(claim: Claim) -> dict:
    # Reparse scientific fields so model_copy/model_construct cannot bypass closure.
    original = Claim.model_validate(claim.model_dump(mode="json", include=_SCIENTIFIC_FIELDS))
    return original.model_dump(mode="json", include=_SCIENTIFIC_FIELDS)


def build_resource_contract(
    claim: Claim,
    materials: SharedMaterials,
    *,
    condition_ids: list[str],
    entry_script: str | None,
    config: str | None,
    data_paths: list[str],
    weight_paths: list[str],
) -> ExecutionResourceContract | None:
    """Build identity from trusted inputs. Shared roles remain candidate proposals."""
    scientific = _scientific_claim(claim)
    conditions = {row["id"]: row for row in scientific["conditions"]}
    if (
        not isinstance(condition_ids, list) or not condition_ids
        or any(not isinstance(key, str) for key in condition_ids)
        or len(condition_ids) != len(set(condition_ids))
        or not set(condition_ids).issubset(conditions)
    ):
        raise ValueError("Resource selection conditions must identify original claim conditions")
    roles = [
        ("entry", [] if entry_script is None else [entry_script]),
        ("config", [] if config is None else [config]),
        ("data", data_paths), ("weights", weight_paths),
    ]
    for _, paths in roles:
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise ValueError("Resource selections must be exact path lists")
        if len(paths) != len(set(paths)):
            raise ValueError("A resource role cannot contain duplicate selections")
        for path in paths:
            _relative(path)
    index = materials.repository
    if index is None:
        if any(paths for _, paths in roles):
            raise ValueError("Selected resources require a released repository index")
        return None
    indexed = {}
    for item in index.files:
        path = _relative(item.path)
        if path in indexed:
            raise ValueError("Repository index contains duplicate resource identities")
        if not re.fullmatch(r"[0-9a-f]{64}", item.sha256):
            raise ValueError("Repository index contains an invalid resource hash")
        indexed[path] = item.sha256
    try:
        unresolved_root = Path(index.root).absolute()
        for ancestor in (unresolved_root, *unresolved_root.parents):
            if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
                raise ValueError("Released repository root cannot follow a link or junction")
        root = unresolved_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Released repository root is unavailable") from exc
    if not root.is_dir():
        raise ValueError("Released repository root is unavailable")
    resources, observed = [], {}
    for role, paths in roles:
        for path in paths:
            if path not in indexed:
                raise ValueError("Selected resource is absent from the repository index")
            if role == "entry" and path not in index.entry_scripts:
                raise ValueError("Selected entry is absent from the indexed entry scripts")
            if role == "config" and path not in index.configs:
                raise ValueError("Selected config is absent from the indexed configs")
            if path not in observed:
                actual = _actual_file(root, path)
                digest = hashlib.sha256()
                try:
                    with actual.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                except OSError as exc:
                    raise ValueError("Selected indexed resource cannot be read") from exc
                observed[path] = digest.hexdigest()
            if observed[path] != indexed[path]:
                raise ValueError("Selected resource bytes differ from the repository index")
            resources.append(ExecutionResourceSelection(role=role, path=path, sha256=indexed[path]))
    return ExecutionResourceContract(
        claim_id=scientific["id"], claim_sha256=_fingerprint(scientific),
        condition_ids=list(condition_ids),
        condition_sha256={key: _fingerprint(row) for key, row in conditions.items()},
        resources=resources,
    )


def validate_resource_contract(
    plan: ExecutionPlan, claim: Claim, materials: SharedMaterials
) -> ExecutionResourceContract | None:
    """None means unbound history; an invalid supplied contract raises ValueError.

    A matching result binds candidate identities only. It never authorizes execution,
    per-condition consumption, alignment, support, or replacing an approved plan.
    """
    if plan.task.resource_contract is None:
        return None
    raw = plan.task.resource_contract
    supplied = ExecutionResourceContract.model_validate(
        raw.model_dump(mode="json") if isinstance(raw, ExecutionResourceContract) else raw
    )
    scientific = _scientific_claim(claim)
    original = {row["id"]: row for row in scientific["conditions"]}
    target_ids = [row.id for row in plan.target_conditions]
    if (
        plan.claim_id != scientific["id"] or target_ids != plan.condition_ids
        or any(row.model_dump(mode="json") != original.get(row.id) for row in plan.target_conditions)
    ):
        raise ValueError("Resource contract plan differs from its original claim conditions")
    rebuilt = build_resource_contract(
        claim, materials, condition_ids=plan.condition_ids,
        entry_script=plan.task.entry_script, config=plan.task.config,
        data_paths=plan.task.data_paths, weight_paths=plan.task.weight_paths,
    )
    if rebuilt is None or supplied != rebuilt:
        raise ValueError("Resource contract differs from the independently rebuilt selection identity")
    return rebuilt
