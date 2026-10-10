"""Pure complete-data host measurement; no runtime or scientific qualification.

Original inputs must be captured independently by a future caller. This helper
rebuilds current identities, never authenticates simultaneous caller replacement,
author output, model inference or a source's scientific correspondence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from fact_generation.execution.resource_contract import (
    _actual_file,
    _scientific_claim,
    validate_resource_contract,
)
from schemas.partition_analysis import VERSION, PartitionAnalysisProposal
from schemas.runtime_science import ROLES


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root(path):
    unresolved = Path(path).absolute()
    _require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                     for p in (unresolved, *unresolved.parents)), "linked_analysis_root")
    root = unresolved.resolve(strict=True)
    _require(root.is_dir(), "analysis_root_unavailable")
    return root


def _read(path, capacity):
    with path.open("rb") as stream:
        raw = stream.read(capacity + 1)
    _require(len(raw) <= capacity, "analysis_byte_capacity_exceeded")
    return raw


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_analysis_json_key")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("nonfinite_analysis_json")


def _select(value, keys):
    for key in keys:
        _require((type(value) is dict and type(key) is str and key in value)
                 or (type(value) is list and type(key) is int and 0 <= key < len(value)),
                 "analysis_selector_missing_or_type")
        value = value[key]
    return value


def _scalar(value, *, numeric=False):
    kinds = (int, float) if numeric else (int, float, str)
    _require(type(value) in kinds, "analysis_value_type_invalid")
    if type(value) is str:
        _require(0 < len(value) <= 4096, "analysis_label_capacity")
    else:
        _require((type(value) is not int or value.bit_length() <= 256) and math.isfinite(value),
                 "nonfinite_analysis_numeric_value")
    return value


def _located(text, definition):
    _require(definition.end > definition.start and text[definition.start:definition.end] == definition.quote
             and text.count(definition.quote) == 1, "analysis_source_quote_span_invalid")


def _unresolved(reason):
    return {"version": VERSION, "status": "unresolved", "reason": reason,
            "host_measurement": None, "context": None, "scientific_qualification": False,
            "alignment": False, "support": False, "runtime_output_authenticated": False,
            "model_inference_performed": False, "derived_observations": [],
            "unresolved": ["independent_source_review_required", "model_inference_unproven", "actual_run_unauthenticated"]}


def analyze_partition(proposal, *, plan, claim, materials, workspace, max_bytes=1048576, max_rows=4096):
    """Return a measured candidate context, always unresolved and science-false.

    y_paper binds original targets only; no reported value enters measurement.
    No row filters, row-count proposals, model callbacks or author flags exist.
    """
    try:
        _require(type(max_bytes) is int and 0 < max_bytes <= 16777216
                 and type(max_rows) is int and 0 < max_rows <= 65536, "invalid_analysis_capacity")
        parsed = PartitionAnalysisProposal.model_validate(proposal)
        scientific = _scientific_claim(claim)
        _require(plan.run_mode == "analysis" and len(plan.condition_ids) == len(scientific["conditions"]) == 1,
                 "single_complete_analysis_condition_required")
        contract = validate_resource_contract(plan, claim, materials)
        _require(contract is not None, "unbound_analysis_resources")
        from verification.experiment_targets import validate_plan_targets
        targets = validate_plan_targets(plan, claim, materials)
        _require(all(row.version == 1 for row in targets.values()), "ordinary_analysis_target_required")
        condition = plan.target_conditions[0]
        _require(set(condition.settings).issubset({"model", "split", "sample_count"})
                 and {"model", "split"}.issubset(condition.settings), "unobserved_analysis_settings")
        _require(parsed.artifact_path in plan.task.data_paths, "analysis_artifact_outside_original_data_selection")
        _require(parsed.label_selector != parsed.prediction_selector, "analysis_label_prediction_selectors_identical")
        _require(type(parsed.partition_selector[-1]) is str
                 and type(condition.settings["split"]) is str
                 and parsed.partition_selector[-1] == condition.settings["split"], "actual_partition_conflicts_with_target")
        root, working = _root(materials.repository.root), _root(workspace)
        original_plan = plan.model_dump(mode="json")
        original_materials = materials.model_dump(mode="json")
        for resource in contract.resources:
            _require(_hash(_actual_file(working, resource.path)) == resource.sha256, "analysis_workspace_resource_changed")
        index = {row.path: row for row in materials.repository.files}
        original_raw = _read(_actual_file(root, parsed.artifact_path), max_bytes)
        raw = _read(_actual_file(working, parsed.artifact_path), max_bytes)
        _require(raw == original_raw and hashlib.sha256(raw).hexdigest() == index[parsed.artifact_path].sha256,
                 "analysis_artifact_changed")
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant)
        partition = _select(data, parsed.partition_selector)
        _require(type(partition) is list and 0 < len(partition) <= max_rows, "complete_partition_capacity_exceeded")
        if "sample_count" in condition.settings:
            count = condition.settings["sample_count"]
            _require(type(count) is int and count == len(partition), "analysis_complete_sample_count_conflicts")
        metadata = {role: _select(data, getattr(parsed, role + "_selector")) for role in ("dataset", "model", "metric")}
        _require(all(type(v) is str and v.strip() and len(v) <= 128 for v in metadata.values()), "analysis_metadata_type_invalid")
        _require(metadata["dataset"] == condition.dataset and metadata["model"] == condition.settings["model"],
                 "analysis_metadata_conflicts_with_original_condition")
        canonical = {"accuracy": "exact_match_fraction", "exact-match accuracy": "exact_match_fraction",
                     "mse": "mean_squared_error", "mean squared error": "mean_squared_error"}
        _require(canonical.get(metadata["metric"].lower()) == canonical.get((condition.metric or "").lower())
                 == parsed.metric_definition, "analysis_metric_definition_conflicts")
        rows, terms = [], []
        for i, row in enumerate(partition):
            label = _scalar(_select(row, parsed.label_selector), numeric=parsed.metric_definition == "mean_squared_error")
            prediction = _scalar(_select(row, parsed.prediction_selector), numeric=parsed.metric_definition == "mean_squared_error")
            if parsed.metric_definition == "exact_match_fraction":
                _require(type(label) is type(prediction), "exact_match_label_prediction_type_conflicts")
                term = int(label == prediction)
            else:
                term = (prediction - label) * (prediction - label)
                _scalar(term, numeric=True)
            rows.append({"index": i, "label": label, "prediction": prediction})
            terms.append(term)
        total = math.fsum(terms)
        _require(math.isfinite(total), "nonfinite_analysis_aggregate")
        measurement = {"metric_definition": parsed.metric_definition, "value": total / len(rows),
                       "sample_count": len(rows), "aggregate": total,
                       "unit": "fraction" if parsed.metric_definition == "exact_match_fraction" else None,
                       "measurement_authority": "independent host recomputation of complete released partition"}
        from verification.code_scope import _claim_sources
        authorized = _claim_sources(claim, materials)
        blocks = {row.id: row for row in materials.blocks}
        catalog, texts, source_hashes = {}, {}, {}
        paper_hashes = {}
        for ref in authorized.values():
            path = Path(ref["pointer"]["locator"])
            if path.is_file():
                paper_hashes[str(path.resolve())] = _hash(path)
        _require(bool(paper_hashes), "analysis_grounded_paper_artifact_unavailable")
        _require(len({row.path for row in parsed.sources.values()}) <= 6, "analysis_source_file_capacity")
        for role in ROLES:
            paper, source = parsed.paper[role], parsed.sources[role]
            _require(paper.block_id in blocks, "analysis_paper_definition_missing")
            block = blocks[paper.block_id]
            _located(block.text, paper)
            refs = [r for r in authorized.values() if r["block_id"] == block.id
                    and condition.id in r["covered"] and paper.quote in r["quote"]]
            _require(bool(refs), "analysis_paper_source_outside_original_condition")
            _require(source.path in index, "analysis_source_not_indexed")
            if source.path not in texts:
                source_raw = _read(_actual_file(root, source.path), 65536)
                _require(source_raw == _read(_actual_file(working, source.path), 65536)
                         and hashlib.sha256(source_raw).hexdigest() == index[source.path].sha256,
                         "analysis_definition_source_changed")
                texts[source.path] = source_raw.decode("utf-8")
                source_hashes[source.path] = index[source.path].sha256
            _located(texts[source.path], source)
            catalog[role] = {"paper": paper.model_dump(), "paper_block_digest": _digest(block.model_dump(mode="json")),
                             "authorized_sources": refs, "source": source.model_dump(), "source_sha256": source_hashes[source.path]}
        _require(sum(len(t.encode()) for t in texts.values()) <= 196608, "analysis_total_source_capacity")
        _require(validate_resource_contract(plan, claim, materials) == contract
                 and plan.model_dump(mode="json") == original_plan
                 and materials.model_dump(mode="json") == original_materials
                 and _scientific_claim(claim) == scientific, "analysis_original_inputs_changed")
        for resource in contract.resources:
            _require(_hash(_actual_file(working, resource.path)) == resource.sha256, "analysis_workspace_changed_during_measurement")
        for name, digest in source_hashes.items():
            _require(_hash(_actual_file(root, name)) == _hash(_actual_file(working, name)) == digest, "analysis_source_changed_during_measurement")
        _require(all(_hash(Path(path)) == digest for path, digest in paper_hashes.items()),
                 "analysis_paper_artifact_changed")
        context = {"version": VERSION, "scientific_claim": scientific, "original_plan": original_plan,
                   "resource_contract": contract.model_dump(mode="json"), "proposal": parsed.model_dump(mode="json"),
                   "catalog": catalog, "source_files": texts, "artifact_sha256": index[parsed.artifact_path].sha256,
                   "artifact_path": parsed.artifact_path, "rows": rows, "host_measurement": measurement,
                   "unreviewed_metadata": metadata, "paper_artifact_sha256": paper_hashes,
                   "limits": {"max_bytes": max_bytes, "max_rows": max_rows}}
        context["context_digest"] = _digest(context)
        _require(len(json.dumps(context, ensure_ascii=True, allow_nan=False).encode()) <= 2097152, "analysis_context_capacity")
        result = _unresolved("Host measurement available; source semantics and actual execution remain unreviewed")
        result.update(host_measurement=measurement, context=context)
        return result
    except (ValueError, OSError, TypeError, KeyError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        return _unresolved(str(exc) if type(exc) is ValueError else type(exc).__name__)
