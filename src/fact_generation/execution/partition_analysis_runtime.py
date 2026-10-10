"""Protected stdout identity plus complete released-data host measurement.

The receipt authenticates a limited producer under a trusted caller premise.
Equal output does not establish author data consumption or new model inference.
Paper correspondence requires the separate metered analysis-v1 semantic review.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from schemas.partition_analysis import PartitionRuntimeProposal
from schemas.runtime_science import ROLES

from .partition_analysis import _constant, _digest, _object, _read, _require, analyze_partition
from .runtime_receipt import read_observer_receipt
from .v2_outputs import _actual_metrics, _metric_key, _runtime_value, decode_output


def _audit_file(path, expected):
    path = Path(path).absolute()
    _require(path == expected.absolute() and not any(
        p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)),
        "analysis_runtime_audit_pointer_invalid")
    raw = _read(path, 1048576)
    return raw, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _json(raw):
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)


def build_partition_runtime_context(proposal, *, plan, claim, materials, request, outcome, source_files):
    """Rebuild original inputs and actual producer/output facts before review."""
    parsed = PartitionRuntimeProposal.model_validate(proposal)
    _require(request.plan.model_dump(mode="json") == plan.model_dump(mode="json"), "analysis_original_plan_changed")
    _require(request.metric_output is None and plan.run_mode == "analysis", "analysis_protected_stdout_required")
    measured = analyze_partition(parsed.analysis.model_dump(mode="json"), plan=plan, claim=claim,
                                 materials=materials, workspace=request.workspace)
    _require(measured["host_measurement"] is not None, measured["reason"])
    facts = measured["context"]
    _require(type(source_files) is dict and all(
        type(source_files.get(name)) is str and source_files[name] == text.replace("\r\n", "\n").replace("\r", "\n")
        for name, text in facts["source_files"].items()), "analysis_definition_not_supplied_to_refinement")
    receipt = read_observer_receipt(request, outcome)
    _require(receipt["status"] == "received", "analysis_runtime_receipt_unavailable: " + receipt.get("reason", "unknown"))
    attempt = Path(request.run_dir).absolute() / f"attempt_{request.repair_round}"
    raw, raw_ref = _audit_file(outcome.logs["raw_output"], attempt / "raw_output.json")
    _require(type(outcome.stdout) is str and len(outcome.stdout) <= 1048576
             and len(outcome.stdout.encode()) <= 1048576, "analysis_stdout_capacity")
    text = outcome.stdout.strip()
    prefix = "FACTREVIEW_OBSERVATIONS="
    if text.startswith(prefix):
        text = text[len(prefix):]
    payload = _json(text)
    _require(type(payload) is dict and _digest(payload) == _digest(_json(raw)), "analysis_stdout_raw_output_conflicts")
    selector = parsed.raw_value_selector
    _require(type(selector[-1]) is str and _metric_key(selector[-1]) == _metric_key(facts["unreviewed_metadata"]["metric"]),
             "analysis_stdout_metric_name_conflicts")
    value = _runtime_value(payload, selector)
    _require(type(value) in (int, float) and (type(value) is not int or value.bit_length() <= 256)
             and math.isfinite(value) and value == measured["host_measurement"]["value"], "analysis_stdout_host_measurement_conflicts")
    _require(all(actual == value for actual in _actual_metrics(payload).get(_metric_key(selector[-1]), [])),
             "analysis_stdout_identified_metric_conflicts")
    try:
        decoded, mapping = decode_output(payload, request.output_mapping)
    except ValueError as exc:
        # This subset permits a named scalar without observation metadata.
        # Explicit canonical observations and other decoder errors stay errors.
        _require(request.output_mapping is None and "observations" not in payload
                 and "dataset" not in payload
                 and str(exc) == "runtime dataset is missing at the configured JSON path",
                 "analysis_stdout_decoder_rejected: " + str(exc))
        decoded, mapping = [], None
    _require(_digest(decoded) == _digest([row.model_dump(mode="json") for row in outcome.observations])
             and not decoded, "analysis_decoded_observations_unsupported_or_changed")
    mapping_ref = None
    mapping_path = attempt / "output_mapping.json"
    if mapping is not None:
        mapping_raw, mapping_ref = _audit_file(outcome.logs["output_mapping"], mapping_path)
        _require(_digest(_json(mapping_raw)) == _digest({"source": str(Path(outcome.logs["raw_output"])), "selectors": mapping}),
                 "analysis_decoder_audit_changed")
    else:
        _require(not mapping_path.exists() and not mapping_path.is_symlink(), "analysis_decoder_failure_audit_conflicts")
    condition = plan.target_conditions[0]
    observed = {"dataset": facts["unreviewed_metadata"]["dataset"], "metric": facts["unreviewed_metadata"]["metric"],
        "settings": {"model": facts["unreviewed_metadata"]["model"], "split": parsed.analysis.partition_selector[-1]},
        "value": measured["host_measurement"]["value"]}
    if "sample_count" in condition.settings:
        observed["settings"]["sample_count"] = measured["host_measurement"]["sample_count"]
    if measured["host_measurement"]["unit"] is not None:
        observed["unit"] = measured["host_measurement"]["unit"]
    from .v2 import Observation, aligned
    _require(aligned(Observation.model_validate(observed), condition), "analysis_actual_metadata_conflicts")
    catalog = {}
    for role in ROLES:
        row = facts["catalog"][role]
        catalog["paper/"+role] = {**row["paper"], "paper_block_digest": row["paper_block_digest"],
            "authorized_sources": row["authorized_sources"]}
        catalog["source/"+role] = {**row["source"], "sha256": row["source_sha256"]}
    refs = {**receipt["evidence_refs"], "raw_output": raw_ref, "decoder_mapping": mapping_ref}
    catalog["actual/analysis"] = {"measurement": facts, "runtime_receipt": receipt, "raw_payload": payload,
        "raw_value_selector": selector, "evidence_refs": refs, "model_inference_performed": False,
        "author_data_to_metric_dynamic_consumption": "unproven",
        "measurement_authority": "independent host recomputation of complete released partition"}
    obligations = {role: ["paper/"+role, "source/"+role, "actual/analysis"] for role in ROLES}
    for key in ("dataset", "metric", "description", *["settings/"+name for name in condition.settings]):
        role = {"dataset": "dataset", "metric": "metric", "settings/model": "model", "settings/split": "split",
                "settings/sample_count": "population"}.get(key, "qualifiers")
        obligations["condition/"+key] = obligations[role]
    obligations["measurement_authority"] = ["paper/qualifiers", "paper/model", "source/model", "actual/analysis"]
    context = {"version": "analysis-v1", "scientific_claim": facts["scientific_claim"], "condition": condition.model_dump(mode="json"),
        "plan": facts["original_plan"], "resource_contract": facts["resource_contract"],
        "paper_artifact_sha256": facts["paper_artifact_sha256"], "catalog": catalog, "obligations": obligations,
        "actual_observation": observed, "limits": ["complete_released_partition_statistics_only", "protected_stdout_single_json_object",
        "no_raw_decoded_observation", "author_dynamic_consumption_unproven", "no_new_model_inference",
        "scientific_correspondence_requires_independent_review", "trusted_caller_and_runtime_receipt_premise"]}
    _require(len(json.dumps(context, ensure_ascii=True).encode()) <= 3145728, "analysis_runtime_context_capacity")
    context["context_digest"] = _digest(context)
    return context


def qualified_analysis_metadata(context):
    """Called only after the independent review and strict measured usage close."""
    facts = context["catalog"]["actual/analysis"]["measurement"]
    return {"protocol": "analysis-v1", "model_inference_performed": False,
        "measurement_authority": "independent host recomputation of complete released partition",
        "author_dynamic_consumption": "unproven", "author_artifact_provenance": {
            "released_artifact": True, "artifact_kind": "data", "environment_explanation_possible": False,
            "artifact_path": facts["artifact_path"], "artifact_sha256": facts["artifact_sha256"]}}
