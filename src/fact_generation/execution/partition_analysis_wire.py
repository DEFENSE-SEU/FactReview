"""Exact quote localization against complete sources frozen before refinement."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

from schemas.partition_analysis import PartitionRuntimeCandidate, PartitionRuntimeProposal
from schemas.runtime_science import ROLES

from .repair_evidence import TEXT_BUDGET, bounded_source, plain_path

CANDIDATE_VERSION = "partition-runtime-candidate-v2"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _lf(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def freeze_sources(*, workspace, supplied_files, paper_sources, materials):
    """Read lexical regular paths before resolving; retain full bounded snapshots."""
    result = {"status": "unresolved", "snapshot_timing": "before_completion"}
    try:
        root = plain_path(Path(workspace)).resolve(strict=True)
        _require(root.is_dir(), "workspace_unavailable")
        _require(type(supplied_files) is dict and 0 < len(supplied_files) <= 6,
                 "source_scope_capacity")
        index = {row.path: row.sha256 for row in materials.repository.files}
        _require(len(index) == len(materials.repository.files), "source_index_duplicate")
        sources, total = {}, 0
        for name, displayed in supplied_files.items():
            _require(type(name) is str and type(displayed) is str, "source_type")
            relative = PurePosixPath(name)
            _require(name and "\\" not in name and not relative.is_absolute()
                     and ":" not in name and ".." not in relative.parts
                     and relative.as_posix() == name and name in index, "source_path")
            candidate = plain_path(Path(workspace) / name)
            _require(candidate.resolve(strict=True).is_relative_to(root), "source_outside_workspace")
            raw, identity = bounded_source(candidate)
            _require(raw is not None, "source_capacity")
            total += len(raw)
            _require(total <= TEXT_BUDGET, "source_total_capacity")
            text = raw.decode("utf-8", errors="strict")
            _require(identity["sha256"] == index[name] and displayed == _lf(text),
                     "source_index_or_display_changed")
            sources[name] = {**identity, "raw_text": text, "displayed_text": displayed}
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
        contents = {"sources": sources, "paper": paper}
        result.update(status="frozen", snapshot=contents, snapshot_digest=_digest(contents))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result["reason"] = "snapshot_unavailable:" + type(exc).__name__
    return result


def _unique(text, quote):
    first = text.find(quote)
    _require(first >= 0 and text.find(quote, first + 1) < 0, "quote_not_unique_exact")
    return first, first + len(quote)


def _boundaries(raw):
    positions, i = [0], 0
    while i < len(raw):
        i += 2 if raw[i:i+2] == "\r\n" else 1
        positions.append(i)
    return positions


def localize_candidate(wire, *, snapshot, workspace, supplied_files, paper_sources, materials):
    """One candidate-version dispatch. Original canonical input is never repaired."""
    result = {"status": "unresolved", "canonical": None, "snapshot_timing": "before_completion",
              "scientific_qualification": False, "alignment": False, "support": False,
              "wire": copy.deepcopy(wire), "wire_digest": None,
              "snapshot": copy.deepcopy(snapshot), "locations": []}
    try:
        result["wire_digest"] = _digest(wire)
        _require(type(wire) is dict and wire.get("version") == CANDIDATE_VERSION, "candidate_version")
        parsed = PartitionRuntimeCandidate.model_validate(wire)
        _require(snapshot.get("status") == "frozen", "precall_snapshot_unavailable")
        contents = snapshot["snapshot"]
        _require(_digest(contents) == snapshot["snapshot_digest"], "snapshot_changed")
        current = freeze_sources(workspace=workspace, supplied_files=supplied_files,
                                 paper_sources=paper_sources, materials=materials)
        _require(current.get("status") == "frozen" and current["snapshot"] == contents,
                 "source_or_paper_changed_after_completion")
        canonical = parsed.model_dump(mode="json")
        canonical["version"] = "partition-runtime-v1"
        analysis = canonical["analysis"]
        analysis["version"] = "released-partition-analysis-v1"
        for role in ROLES:
            for group in ("paper", "sources"):
                row = analysis[group][role]
                if group == "paper":
                    _require(row["block_id"] in contents["paper"], "paper_outside_supplied_scope")
                    text = contents["paper"][row["block_id"]]["text"]
                    start, end = _unique(text, row["quote"])
                    display_start, display_end = start, end
                else:
                    _require(row["path"] in contents["sources"], "source_outside_supplied_scope")
                    source = contents["sources"][row["path"]]
                    display_start, display_end = _unique(source["displayed_text"], row["quote"])
                    boundaries = _boundaries(source["raw_text"])
                    start, end = boundaries[display_start], boundaries[display_end]
                    raw_quote = source["raw_text"][start:end]
                    _require(_lf(raw_quote) == row["quote"], "raw_display_bridge")
                    _require(_unique(source["raw_text"], raw_quote) == (start, end), "raw_quote_ambiguous")
                    row["quote"] = raw_quote
                row.update(start=start, end=end)
                result["locations"].append({"group": group, "role": role,
                    "display_start": display_start, "display_end": display_end,
                    "raw_start": start, "raw_end": end})
        canonical = PartitionRuntimeProposal.model_validate(canonical).model_dump(mode="json")
        result.update(status="localized", canonical=canonical,
                      canonical_digest=_digest(canonical))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        result["reason"] = "localization_unavailable:" + type(exc).__name__
    return result
