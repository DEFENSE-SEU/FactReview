"""Protected stdout identity plus complete released-data host measurement.

The receipt authenticates a limited producer under a trusted caller premise.
Equal output does not establish author data consumption or new model inference.
Paper correspondence requires the separate metered analysis-v1 semantic review.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

from schemas.partition_analysis import PartitionRuntimeProposal
from schemas.runtime_science import ROLES

from .partition_analysis import _constant, _digest, _object, _read, _require, analyze_partition
from .resource_contract import _actual_file
from .runtime_observer import _snapshot
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


def _stdout_return(payload, receipt, request, source_files):
    """Finite source/observed-return/printed-output relation; never execute AST."""
    entry = request.plan.task.entry_script
    source = source_files.get(entry)
    _require(type(source) is str and len(source.encode()) <= 65536, "analysis_entry_not_supplied")
    raw = _read(_actual_file(Path(request.workspace).resolve(strict=True), entry), 65536)
    _require(hashlib.sha256(raw).hexdigest() == receipt["run_identity"]["source_sha256"][entry]
             and source == raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"),
             "analysis_entry_source_changed")
    tree = ast.parse(source)
    if request.metric_output is not None:
        writes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                  and n.func.value.id == "json" and n.func.attr == "dump"]
        _require(len(writes) == 1, "analysis_file_write_binding_unknown")
        written = writes[0]
        blocks = [n for n in tree.body if isinstance(n, ast.With) and len(n.items) == 1
                  and len(n.body) == 1 and isinstance(n.body[0], ast.Expr) and n.body[0].value is written]
        _require(len(blocks) == 1, "analysis_file_write_binding_unknown")
        item = blocks[0].items[0]
        opened = item.context_expr
        _require(isinstance(opened, ast.Call) and isinstance(opened.func, ast.Name)
                 and opened.func.id == "open" and len(opened.args) == 2
                 and all(isinstance(n, ast.Constant) and type(n.value) is str for n in opened.args)
                 and opened.args[0].value == request.metric_output and opened.args[1].value == "w"
                 and len(opened.keywords) == 1 and opened.keywords[0].arg == "encoding"
                 and isinstance(opened.keywords[0].value, ast.Constant) and opened.keywords[0].value.value == "utf-8"
                 and isinstance(item.optional_vars, ast.Name) and len(written.args) == 2 and not written.keywords
                 and isinstance(written.args[1], ast.Name) and written.args[1].id == item.optional_vars.id,
                 "analysis_file_literal_path_or_encoding_unknown")
        called = written.args[0]
    else:
        called = _printed_driver(tree)
    _require(isinstance(called, ast.Call) and isinstance(called.func, ast.Name)
             and not called.args and not called.keywords, "analysis_stdout_driver_unknown")
    report = receipt["report"]
    sites = [i for i, site in enumerate(report["sites"])
             if site["path"] == "/app/" + entry and site["qualname"] == called.func.id]
    _require(len(sites) == 1, "analysis_stdout_driver_not_observed")
    events = [event for event in report["events"] if event["site"] == sites[0]]
    _require(len(events) == 2 and events[0]["event"] == "call" and events[0]["arguments"] == {}
             and events[0]["parent_invocation"] is None and events[1]["event"] == "return"
             and events[0]["invocation"] == events[1]["invocation"]
             and _digest(events[1]["value"]) == _digest(_snapshot(payload, 65536)),
             "analysis_stdout_observed_return_conflicts")
    return {"site": sites[0], "invocation": events[0]["invocation"],
            "call_order": events[0]["order"], "return_order": events[1]["order"],
            "output_transport": "fresh_json_file" if request.metric_output is not None else "stdout"}


def _printed_driver(tree):
    prints = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == "print"]
    _require(len(prints) == 1 and any(isinstance(n, ast.Expr) and n.value is prints[0]
             for n in tree.body), "analysis_stdout_print_binding_unknown")
    printed = prints[0]
    _require(len(printed.args) == 1 and not printed.keywords, "analysis_stdout_print_binding_unknown")
    encoded = printed.args[0]
    _require(isinstance(encoded, ast.Call) and isinstance(encoded.func, ast.Attribute)
             and isinstance(encoded.func.value, ast.Name) and encoded.func.value.id == "json"
             and encoded.func.attr == "dumps" and len(encoded.args) == 1 and not encoded.keywords,
             "analysis_stdout_encoding_unknown")
    called = encoded.args[0]
    _require(isinstance(called, ast.Call) and isinstance(called.func, ast.Name)
             and not called.args and not called.keywords, "analysis_stdout_driver_unknown")
    return called


def build_partition_runtime_context(proposal, *, plan, claim, materials, request, outcome, source_files):
    """Rebuild original inputs and actual producer/output facts before review."""
    parsed = PartitionRuntimeProposal.model_validate(proposal)
    _require(request.plan.model_dump(mode="json") == plan.model_dump(mode="json"), "analysis_original_plan_changed")
    _require(plan.run_mode == "analysis", "analysis_run_mode_required")
    measured = analyze_partition(parsed.analysis.model_dump(mode="json"), plan=plan, claim=claim,
                                 materials=materials, workspace=request.workspace)
    _require(measured["host_measurement"] is not None, measured["reason"])
    facts = measured["context"]
    required_sources = dict(facts["source_files"])
    if facts["selected_config"] is not None:
        name = facts["selected_config"]["path"]
        config_raw = _read(Path(request.workspace) / name, 65536)
        _require(hashlib.sha256(config_raw).hexdigest() == facts["selected_config"]["sha256"],
                 "analysis_configuration_changed")
        required_sources[name] = config_raw.decode("utf-8")
    _require(type(source_files) is dict and all(
        type(source_files.get(name)) is str and source_files[name] == text.replace("\r\n", "\n").replace("\r", "\n")
        for name, text in required_sources.items()), "analysis_definition_not_supplied_to_refinement")
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
    if request.metric_output is None:
        payload = _json(text)
    else:
        _require("metric_file_output" in receipt, "analysis_metric_file_receipt_required")
        payload = receipt["metric_file_output"]["payload"]
    _require(type(payload) is dict and _digest(payload) == _digest(_json(raw)), "analysis_stdout_raw_output_conflicts")
    return_binding = _stdout_return(payload, receipt, request, source_files)
    selector = parsed.raw_value_selector
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
    from .v2 import Observation, aligned
    _require(_digest([Observation.model_validate(row).model_dump(mode="json") for row in decoded])
             == _digest([row.model_dump(mode="json") for row in outcome.observations]),
             "analysis_decoded_observations_unsupported_or_changed")
    selected_decoded = None
    if decoded:
        _require(type(mapping) is list and len(decoded) == len(mapping), "analysis_decoded_scope_ambiguous")
        selected = [
            (index, row, location)
            for index, (row, location) in enumerate(zip(decoded, mapping, strict=True))
            if selector == ([*location["canonical_path"], "value"] if "canonical_path" in location
                            else [*location["root_path"], *location["metric_path"]])
        ]
        _require(len(selected) == 1, "analysis_decoded_scope_ambiguous")
        index, row, location = selected[0]
        selected_decoded = {"index": index, "location": location, "observation": row}
        _require(row.get("dataset") == facts["unreviewed_metadata"]["dataset"]
                 and _metric_key(row.get("metric", "")) == _metric_key(facts["unreviewed_metadata"]["metric"])
                 and type(row.get("value")) in (int, float) and row["value"] == value,
                 "analysis_decoded_metric_conflicts")
        _require(type(row.get("settings")) is dict and all(name in facts["actual_settings"]
                 and _digest(setting) == _digest(facts["actual_settings"][name])
                 for name, setting in row["settings"].items()), "analysis_decoded_setting_conflicts")
        _require(row.get("reported_variance") is None and row.get("unit") in {None, measured["host_measurement"]["unit"]},
                 "analysis_decoded_units_or_variance_unknown")
    else:
        _require(type(selector[-1]) is str and _metric_key(selector[-1]) == _metric_key(facts["unreviewed_metadata"]["metric"]),
                 "analysis_stdout_metric_name_conflicts")
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
        "settings": facts["actual_settings"],
        "value": measured["host_measurement"]["value"]}
    if "sample_count" in condition.settings:
        observed["settings"]["sample_count"] = measured["host_measurement"]["sample_count"]
    if selected_decoded is not None and selected_decoded["observation"].get("unit") is not None:
        observed["unit"] = selected_decoded["observation"]["unit"]
    _require(not any(key in condition.settings for key in ("unit", "units", "reported_variance")),
             "analysis_explicit_units_or_variance_unobserved")
    _require(aligned(Observation.model_validate(observed), condition), "analysis_actual_metadata_conflicts")
    catalog = {}
    for role in ROLES:
        row = facts["catalog"][role]
        catalog["paper/"+role] = {**row["paper"], "paper_block_digest": row["paper_block_digest"],
            "authorized_sources": row["authorized_sources"]}
        catalog["source/"+role] = {**row["source"], "sha256": row["source_sha256"]}
    refs = {**receipt["evidence_refs"], "raw_output": raw_ref, "decoder_mapping": mapping_ref}
    if request.metric_output is not None:
        refs.update(receipt["metric_file_output"]["evidence_refs"])
    catalog["actual/analysis"] = {"measurement": facts, "runtime_receipt": receipt, "raw_payload": payload,
        "raw_value_selector": selector, "evidence_refs": refs, "model_inference_performed": False,
        "runtime_decoded_observations": decoded, "selected_decoded_metric": selected_decoded,
        "observed_return_binding": return_binding,
        "actual_entry_source": source_files[request.plan.task.entry_script],
        "actual_entry_sha256": receipt["run_identity"]["source_sha256"][request.plan.task.entry_script],
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
        "actual_observation": observed, "limits": ["complete_released_partition_statistics_only", "protected_stdout_single_json_object" if request.metric_output is None else "protected_fresh_literal_json_file",
        "decoded_raw_preserved_and_compared", "author_dynamic_consumption_unproven", "no_new_model_inference",
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
