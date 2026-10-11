"""Locate Torch candidate quotes in full frozen paper and normalized-LF sources."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

from schemas.inference_candidate import TorchInferenceCandidate
from schemas.runtime_science import ROLES, TorchInferenceProposal

from .repair_evidence import bounded_source, plain_path
from .runtime_science import read_inference_sources

CANDIDATE_VERSION = "torch-cpu-inference-candidate-v2"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def freeze_sources(*, plan, workspace, supplied_files, paper_sources, materials):
    """Snapshot the complete original closure before completion, never a prefix."""
    result = {"status": "unresolved", "snapshot_timing": "before_completion"}
    try:
        root = plain_path(Path(workspace)).resolve(strict=True)
        original = plain_path(Path(materials.repository.root)).resolve(strict=True)
        _require(root.is_dir() and original.is_dir(), "source_root_unavailable")
        _require(type(supplied_files) is dict and 0 < len(supplied_files) <= 16,
                 "source_scope_capacity")
        rebuilt, read_scope = read_inference_sources(plan, materials, workspace)
        _require(not read_scope["unavailable"] and rebuilt == supplied_files,
                 "complete_inference_closure_required")
        index = {row.path: row.sha256 for row in materials.repository.files}
        _require(len(index) == len(materials.repository.files), "source_index_duplicate")
        sources, total = {}, 0
        for name, displayed in supplied_files.items():
            _require(type(name) is str and type(displayed) is str, "source_type")
            relative = PurePosixPath(name)
            _require(name and "\\" not in name and not relative.is_absolute()
                     and ":" not in name and ".." not in relative.parts
                     and relative.as_posix() == name and name in index, "source_path")
            candidate = plain_path(root / name)
            author = plain_path(original / name)
            _require(candidate.resolve(strict=True).is_relative_to(root)
                     and author.resolve(strict=True).is_relative_to(original), "source_outside_repository")
            raw, identity = bounded_source(candidate)
            author_raw, author_identity = bounded_source(author)
            _require(raw is not None and author_raw is not None, "source_capacity")
            total += len(raw)
            _require(total <= 262144, "source_total_capacity")
            text = raw.decode("utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
            _require(identity == author_identity and identity["sha256"] == index[name]
                     and displayed == text, "source_index_or_display_changed")
            sources[name] = {**identity, "displayed_text": displayed}
        _require(type(paper_sources) is list and paper_sources, "paper_scope_missing")
        blocks = {block.id: block for block in materials.blocks}
        _require(len(blocks) == len(materials.blocks), "paper_block_duplicate")
        paper = {}
        for row in paper_sources:
            _require(type(row) is dict and set(row) == {"id", "text", "loc"}, "paper_source_shape")
            name = row["id"]
            _require(type(name) is str and name in blocks and name not in paper, "paper_source_id")
            block = blocks[name]
            _require(row["text"] == block.text and row["loc"] == block.loc.model_dump(mode="json"),
                     "paper_display_changed")
            paper[name] = copy.deepcopy(row)
        contents = {"workspace": str(root), "repository": str(original), "index": index,
                    "repository_index": materials.repository.model_dump(mode="json"),
                    "read_scope": read_scope, "sources": sources, "paper": paper}
        result.update(status="frozen", snapshot=contents, snapshot_digest=_digest(contents))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result["reason"] = "snapshot_unavailable:" + type(exc).__name__
    return result


def _unique(text, quote):
    start = text.find(quote)
    _require(start >= 0 and text.find(quote, start + 1) < 0, "quote_not_unique_exact")
    return start, start + len(quote)


def localize_candidate(wire, *, snapshot, plan, workspace, supplied_files, paper_sources, materials):
    """Only the explicit candidate version receives program-generated LF spans."""
    result = {"status": "unresolved", "canonical": None, "snapshot_timing": "before_completion",
              "scientific_qualification": False, "alignment": False, "support": False,
              "wire": copy.deepcopy(wire), "wire_digest": None,
              "snapshot": copy.deepcopy(snapshot), "locations": []}
    try:
        result["wire_digest"] = _digest(wire)
        if type(wire) is dict and wire.get("version") == "torch-cpu-inference-v1":
            result["status"] = "not_candidate"
            return result
        _require(type(wire) is dict and wire.get("version") == CANDIDATE_VERSION, "candidate_version")
        parsed = TorchInferenceCandidate.model_validate(wire)
        _require(snapshot.get("status") == "frozen", "precall_snapshot_unavailable")
        contents = snapshot["snapshot"]
        _require(_digest(contents) == snapshot["snapshot_digest"], "snapshot_changed")
        current = freeze_sources(plan=plan, workspace=workspace, supplied_files=supplied_files,
                                 paper_sources=paper_sources, materials=materials)
        _require(current.get("status") == "frozen" and current["snapshot"] == contents,
                 "source_or_paper_changed_after_completion")
        canonical = parsed.model_dump(mode="json")
        canonical["version"] = "torch-cpu-inference-v1"
        for role in ROLES:
            for group in ("paper", "sources"):
                row = canonical[group][role]
                name = row["block_id"] if group == "paper" else row["path"]
                _require(name in contents[group], "quote_outside_supplied_scope")
                text = contents[group][name]["text" if group == "paper" else "displayed_text"]
                start, end = _unique(text, row["quote"])
                row.update(start=start, end=end)
                result["locations"].append({"group": group, "role": role, "start": start, "end": end})
        canonical = TorchInferenceProposal.model_validate(canonical).model_dump(mode="json")
        result.update(status="localized", canonical=canonical, canonical_digest=_digest(canonical))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result["reason"] = "localization_unavailable:" + type(exc).__name__
    return result
