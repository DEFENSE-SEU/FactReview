"""Pure host rebuilding for full numeric-partition CPU inference facts."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
from pathlib import Path

from pydantic import ValidationError

from schemas.runtime_science import ROLES, TorchInferenceProposal

from .config import TorchInferenceLimits
from .repair_evidence import bounded_source, plain_path
from .resource_contract import _actual_file, _fingerprint, _scientific_claim, validate_resource_contract

VERSION = "torch-cpu-inference-v1"

# Only existing literal requirements can become a public binding diagnostic.
_BINDING_GUARD_CODES = frozenset({
    'actual_batch_position_unknown',
    'actual_config_target_mismatch',
    'actual_getitem_different_official_sample',
    'actual_inference_condition_unaligned',
    'actual_metric_host_conflict',
    'actual_metric_return_raw_output_conflict',
    'actual_model_or_split_config_mismatch',
    'actual_prediction_adapter_conflict',
    'actual_role_call_mismatch',
    'actual_sample_tensor_binding_changed',
    'actual_source_span_mismatch',
    'actual_split_or_sample_mismatch',
    'actual_torch_seed_unaligned',
    'authenticated_inference_receipt_required',
    'classification_logits_adapter_unavailable',
    'complete_actual_model_metric_function_required',
    'complete_sample_denominator_required',
    'complete_source_changed_or_unsupplied',
    'complete_source_supply_capacity',
    'complete_source_supply_required',
    'duplicate_call_identity',
    'duplicate_json_key',
    'duplicate_sample_id_or_capacity',
    'empty_official_partition',
    'every_condition_setting_requires_actual_config_selector',
    'finite_actual_prediction_required',
    'finite_label_required',
    'host_metric_overflow',
    'inference_site_index_unbound',
    'inference_site_not_supplied',
    'inference_source_closure_unavailable_or_changed',
    'inference_source_site_not_unique_static_function',
    'inference_source_sites_unbound',
    'integer_actual_prediction_required',
    'integer_class_label_required',
    'integer_tensor_conversion_loss',
    'loaded_module_order_identity_mismatch',
    'nonfinite_json',
    'numeric_dtype_adapter_unavailable',
    'numeric_features_required',
    'ordinary_single_target_required',
    'ordinary_target_required',
    'paper_source_scope_mismatch',
    'partition_line_capacity_or_empty',
    'partition_row_not_object',
    'prediction_metric_actual_identity_missing',
    'resource_not_exclusive_regular_file',
    'runtime_binding_unresolved',
    'sample_id_type',
    'sample_ordinal_partition_incomplete',
    'scalar_model_output_required',
    'selected_binary_or_partition_changed',
    'selected_data_checkpoint_config_required',
    'source_capacity',
    'unsafe_actual_selector',
    'whole_partition_capacity',
    'whole_partition_single_metric_return_required',
})
_BINDING_FAILURE_TYPES = (
    (ValidationError, "invalid_schema"),
    (UnicodeError, "unicode_error"),
    (ValueError, "value_error"),
    (TypeError, "type_error"),
    (KeyError, "key_error"),
    (IndexError, "index_error"),
    (OSError, "io_error"),
    (OverflowError, "overflow_error"),
)

_PROPOSAL_SCHEMA_FIELDS = frozenset({
    "version", "condition_id", "checkpoint_path", "state_selector", "dataset_path", "codec",
    "id_key", "features_key", "label_key", "sites", "sample_id_selector", "sample_features_selector",
    "sample_label_selector", "batch_ids_selector", "batch_features_selector", "batch_labels_selector",
    "prediction_argument", "metric_prediction_argument", "metric_label_argument", "raw_value_selector",
    "prediction_adapter", "metric_adapter", "dataset_selector", "split_selector", "model_selector",
    "metric_selector", "settings_selectors", "paper", "sources",
})
_POLICY_SCHEMA_FIELDS = frozenset({
    "protocol", "torch_version", "image_id", "package_manifest_sha256", "max_tensor_elements",
    "max_events", "max_samples", "max_jsonl_line_bytes", "max_partition_bytes", "max_report_bytes",
})
_SELECTOR_SCHEMA_FIELDS = frozenset({
    "state_selector", "sample_id_selector", "sample_features_selector", "sample_label_selector",
    "batch_ids_selector", "batch_features_selector", "batch_labels_selector", "raw_value_selector",
    "dataset_selector", "split_selector", "model_selector", "metric_selector",
})
_SCHEMA_ERROR_CODES = frozenset({
    "missing", "literal_error", "extra_forbidden", "string_type", "string_too_short", "string_too_long",
    "string_pattern_mismatch", "int_type", "int_parsing", "int_from_float", "float_type", "float_parsing",
    "list_type", "dict_type", "too_short", "too_long", "greater_than_equal", "greater_than",
    "less_than_equal", "less_than", "value_error", "model_type",
})


def _schema_field_path(location, schema_id):
    """Return only finite schema paths, never dynamic keys or numeric indices."""
    if type(location) is not tuple or not location or type(location[0]) is not str:
        return "unknown_field"
    root = location[0]
    fields = _PROPOSAL_SCHEMA_FIELDS if schema_id == "proposal" else (
        _POLICY_SCHEMA_FIELDS if schema_id == "policy" else frozenset())
    if root not in fields:
        return "unknown_field"
    if len(location) == 1:
        return root
    if root in _SELECTOR_SCHEMA_FIELDS and len(location) in (2, 3) and type(location[1]) is int:
        if 0 <= location[1] < 12 and (len(location) == 2 or
                (type(location[2]) is str and location[2] in {"str", "int"})):
            return root + ".item"
    if root == "sites" and len(location) == 2 and type(location[1]) is str:
        return "sites." + location[1] if location[1] in {"getitem", "model", "prediction", "metric"} else "unknown_field"
    if root in {"paper", "sources"} and len(location) in (2, 3) and type(location[1]) is str:
        if location[1] in ROLES:
            prefix = root + "." + location[1]
            if len(location) == 2:
                return prefix
            fields = {"quote", "start", "end", "block_id" if root == "paper" else "path"}
            if type(location[2]) is str and location[2] in fields:
                return prefix + "." + location[2]
    return "unknown_field"


def _schema_validation_diagnostics(error):
    """Project fixed schema IDs, field paths and codes; discard all error text."""
    schema_id = {"TorchInferenceProposal": "proposal", "TorchInferenceLimits": "policy"}.get(
        error.title, "unknown_schema")
    errors = error.errors(include_url=False, include_context=False, include_input=False)
    rows = []
    for item in errors[:8]:
        code = item.get("type")
        rows.append({"field": _schema_field_path(item.get("loc"), schema_id),
                     "error_type": code if type(code) is str and code in _SCHEMA_ERROR_CODES else "unknown_error"})
    return {"schema": "torch-inference-schema-diagnostics-v1", "schema_id": schema_id,
            "errors": rows, "truncated": len(errors) > 8}


def _write_schema_validation_diagnostics(request, error, materials):
    """Write-once operator-output sidecar; failures cannot change qualification."""
    try:
        if type(request.repair_round) is not int or not 0 <= request.repair_round <= 3:
            return
        run_dir = plain_path(Path(request.run_dir)).resolve(strict=True)
        path = plain_path(run_dir / f"attempt_{request.repair_round}" / "inference_schema_diagnostics.json")
        if any(path.is_relative_to(plain_path(Path(root)).resolve(strict=True)) for root in
               (request.workspace, materials.repository.root)):
            return
        payload = _schema_validation_diagnostics(error)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError, AttributeError):
        pass


def _write_paper_scope_diagnostics(request, error, materials):
    """Write only fixed roles and short-circuit booleans outside author mounts."""
    try:
        detail = getattr(error, "paper_scope_diagnostics", None)
        fields = ("span_matches", "quote_unique", "authorized_source_matches")
        if (type(detail) is not dict or set(detail) != {"role", "checks"}
                or type(detail["role"]) is not str or detail["role"] not in ROLES
                or type(detail["checks"]) is not dict or set(detail["checks"]) != set(fields)
                or any(value is not None and type(value) is not bool for value in detail["checks"].values())):
            return
        if type(request.repair_round) is not int or not 0 <= request.repair_round <= 3:
            return
        run_dir = plain_path(Path(request.run_dir)).resolve(strict=True)
        path = plain_path(run_dir / f"attempt_{request.repair_round}" / "inference_paper_scope_diagnostics.json")
        if any(path.is_relative_to(plain_path(Path(root)).resolve(strict=True)) for root in
               (request.workspace, materials.repository.root)):
            return
        payload = {"schema": "torch-inference-paper-scope-diagnostics-v1", "role": detail["role"],
                   "checks": {field: detail["checks"][field] for field in fields}}
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError, AttributeError):
        pass


def bind_inference_request(request, *, claim, materials, source_sites_diagnostics=False, runtime_stage_diagnostic=None):
    """Program-owned attempt binding; errors retain the legal unobserved run."""
    if runtime_stage_diagnostic is not None:
        from .inference_source_sites_diagnostics import RuntimeStageDiagnostic

        if type(runtime_stage_diagnostic) is not RuntimeStageDiagnostic:
            runtime_stage_diagnostic = None
    try:
        diagnostics = {}
        if source_sites_diagnostics is True:
            from .inference_source_sites_diagnostics import SourceSitesDiagnostic

            diagnostics["source_sites_diagnostic"] = SourceSitesDiagnostic(request, materials)
        if runtime_stage_diagnostic is not None:
            diagnostics["runtime_stage_diagnostic"] = runtime_stage_diagnostic
            RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, phase="pre_run", stage="entry")
        binding = bind_inference_proposal(request.source_science, plan=request.plan, claim=claim, materials=materials,
            workspace=request.workspace, source_files=request.source_science_files,
            limits=request.config.docker_options.get("torch_inference"), source_sites=request.source_sites, **diagnostics)
        if runtime_stage_diagnostic is not None:
            RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="context", terminal="passed", pre_run_bound=True)
        return {"status":"bound", "request_sha256":_fingerprint(request.model_dump(mode="json")),"binding":binding}
    except (ValueError,TypeError,KeyError,IndexError,OSError,UnicodeError,OverflowError) as exc:
        if runtime_stage_diagnostic is not None:
            RuntimeStageDiagnostic.reject(runtime_stage_diagnostic, pre_run_bound=False)
        if type(exc) is ValidationError:
            _write_schema_validation_diagnostics(request, exc, materials)
        failure = next((code for error_type, code in _BINDING_FAILURE_TYPES
                        if isinstance(exc, error_type)), "binding_exception")
        if type(exc) is ValueError and len(exc.args) == 1 and type(exc.args[0]) is str:
            failure = exc.args[0] if exc.args[0] in _BINDING_GUARD_CODES else "local_requirement_failed"
            if failure == "paper_source_scope_mismatch":
                _write_paper_scope_diagnostics(request, exc, materials)
        return {"status":"unresolved","reason":"inference_request_binding_unavailable",
                "binding_failure": failure}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def _require_paper_source_scope(paper, block, condition_id, original_sources, role):
    checks = {"span_matches": block.text[paper.start:paper.end] == paper.quote,
              "quote_unique": None, "authorized_source_matches": None}
    if checks["span_matches"]:
        checks["quote_unique"] = block.text.count(paper.quote) == 1
        if checks["quote_unique"]:
            checks["authorized_source_matches"] = any(
                row["block_id"] == paper.block_id and condition_id in row["covered"]
                and paper.quote in row["quote"] for row in original_sources.values())
    if checks["authorized_source_matches"] is not True:
        error = ValueError("paper_source_scope_mismatch")
        error.paper_scope_diagnostics = {"role": role, "checks": checks}
        raise error


def file_sha(path):
    value = hashlib.sha256()
    path = plain_path(Path(path))
    require(path.is_file() and path.stat().st_nlink == 1, "resource_not_exclusive_regular_file")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def select(value, selector):
    for key in selector:
        require((type(value) is dict and type(key) is str and key in value) or
                (type(value) is list and type(key) is int and 0 <= key < len(value)), "unsafe_actual_selector")
        value = value[key]
    return value


def unique_json(raw):
    def pairs(rows):
        require(len(dict(rows)) == len(rows), "duplicate_json_key")
        return dict(rows)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda v: require(False, "nonfinite_json"))


def cast_numeric(values, dtype):
    """Independent standard numeric casting, with no Torch/author import."""
    formats = {"torch.float64":"d", "torch.float32":"f", "torch.float16":"e",
               "torch.int64":"q", "torch.int32":"i", "torch.int16":"h", "torch.int8":"b", "torch.uint8":"B"}
    require(dtype in formats, "numeric_dtype_adapter_unavailable")
    if dtype.startswith("torch.int") or dtype == "torch.uint8":
        require(all(type(v) is int for v in values), "integer_tensor_conversion_loss")
    return [struct.unpack("<"+formats[dtype], struct.pack("<"+formats[dtype], value))[0] for value in values]


def partition_rows(path, proposal, limits):
    """Yield every original row. A capacity failure never returns a prefix grant."""
    seen = set()
    byte_count = 0
    with plain_path(Path(path)).open("rb") as stream:
        index = 0
        while True:
            raw = stream.readline(limits.max_jsonl_line_bytes + 1)
            if not raw:
                break
            byte_count += len(raw)
            require(byte_count <= limits.max_partition_bytes, "whole_partition_capacity")
            require(len(raw) <= limits.max_jsonl_line_bytes and raw.strip(), "partition_line_capacity_or_empty")
            row = unique_json(raw.decode("utf-8"))
            require(type(row) is dict, "partition_row_not_object")
            identifier, features, label = (row[key] for key in (proposal.id_key, proposal.features_key, proposal.label_key))
            require(type(identifier) in (str, int) and identifier != "", "sample_id_type")
            key = (type(identifier).__name__, identifier)
            require(key not in seen and index < limits.max_samples, "duplicate_sample_id_or_capacity")
            seen.add(key)
            require(type(features) is list and features and all(type(v) in (int, float) and math.isfinite(v) for v in features), "numeric_features_required")
            require(type(label) in (int, float) and math.isfinite(label), "finite_label_required")
            if proposal.metric_adapter == "accuracy":
                require(type(label) is int, "integer_class_label_required")
            yield {"ordinal": index, "id": identifier, "features": features, "label": label}
            index += 1
    require(index > 0, "empty_official_partition")


def bind_inference_proposal(proposal, *, plan, claim, materials, workspace, source_files, limits, source_sites=None,
                            source_sites_diagnostic=None, runtime_stage_diagnostic=None):
    """Rebuild selected original resources and complete source, without running them."""
    if runtime_stage_diagnostic is not None:
        from .inference_source_sites_diagnostics import RuntimeStageDiagnostic

        if type(runtime_stage_diagnostic) is not RuntimeStageDiagnostic:
            runtime_stage_diagnostic = None
        else:
            RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="schema_resources")
    parsed = TorchInferenceProposal.model_validate(proposal)
    policy = TorchInferenceLimits.model_validate(limits)
    contract = validate_resource_contract(plan, claim, materials)
    require(contract is not None and plan.condition_ids == [parsed.condition_id] and
            len(plan.target_conditions) == 1 and plan.run_mode == "evaluation", "ordinary_single_target_required")
    require(parsed.dataset_path in plan.task.data_paths and parsed.checkpoint_path in plan.task.weight_paths
            and plan.task.config, "selected_data_checkpoint_config_required")
    root = plain_path(Path(materials.repository.root)).resolve(strict=True)
    running = plain_path(Path(workspace)).resolve(strict=True)
    indexed = {row.path: row.sha256 for row in materials.repository.files}
    if runtime_stage_diagnostic is not None:
        RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="source_closure")
    require(type(source_files) is dict, "complete_source_supply_required")
    from .runtime_science import read_inference_sources
    rebuilt_sources, read_scope = read_inference_sources(plan,materials,workspace)
    require(not read_scope["unavailable"] and rebuilt_sources == source_files, "inference_source_closure_unavailable_or_changed")
    require(len(source_files) <= 16 and all(type(text) is str and len(text) <= 65536 for text in source_files.values())
            and sum(len(text.encode("utf-8")) for text in source_files.values()) <= 262144, "complete_source_supply_capacity")
    files = {}
    for name in set(source_files) | set(row.path for row in parsed.sources.values()) | {plan.task.entry_script, plan.task.config}:
        raw, scope = bounded_source(_actual_file(root, name))
        require(raw is not None and scope["complete"], "source_capacity")
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        require(indexed.get(name) == scope["sha256"] == file_sha(_actual_file(running, name)) and
                source_files.get(name) == text, "complete_source_changed_or_unsupplied")
        files[name] = text
    if runtime_stage_diagnostic is not None:
        RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="source_sites")
    if source_sites_diagnostic is not None:
        from .inference_source_sites_diagnostics import SourceSitesDiagnostic

    if source_sites_diagnostic is None or type(source_sites_diagnostic) is not SourceSitesDiagnostic:
        require(type(source_sites) is dict and source_sites.get("status") == "bound"
                and source_sites.get("version") == "python-source-sites-v1" and source_sites.get("unresolved") == [],
                "inference_source_sites_unbound")
    else:
        SourceSitesDiagnostic.require_bound(source_sites_diagnostic, source_sites)
    if runtime_stage_diagnostic is not None:
        RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="source_sites", terminal="passed")
        RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="paper_config_partition")
    sites = source_sites["sites"]
    require(all(index < len(sites) for index in parsed.sites.values()), "inference_site_index_unbound")
    for role, index in parsed.sites.items():
        site = sites[index]
        require(site["path"] in files and site["sha256"] == indexed[site["path"]], "inference_site_not_supplied")
        located = []
        def visit(node, prefix, *, selection=site, selected=located, source=files[site["path"]]):
            for child in getattr(node,"body",[]):
                if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    name = prefix+child.name
                    first = min([child.lineno,*[row.lineno for row in child.decorator_list]])
                    if name == selection["qualname"] and first == selection["firstlineno"] and isinstance(child,ast.FunctionDef):
                        selected.append(ast.get_source_segment(source,child))
                    visit(child,name+(".<locals>." if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) else "."))
        visit(ast.parse(files[site["path"]]),"")
        require(len(located) == 1, "inference_source_site_not_unique_static_function")
        if role in ("model","metric"):
            require(parsed.sources[role].path == site["path"] and located[0] in parsed.sources[role].quote,
                    "complete_actual_model_metric_function_required")
    from verification.code_scope import _claim_sources
    from verification.experiment_targets import validate_plan_targets
    require(all(row.version == 1 for row in validate_plan_targets(plan, claim, materials).values()), "ordinary_target_required")
    original_sources = _claim_sources(claim, materials)
    blocks = {row.id: row for row in materials.blocks}
    catalog = {}
    for role in ROLES:
        paper, source = parsed.paper[role], parsed.sources[role]
        block = blocks[paper.block_id]
        _require_paper_source_scope(paper, block, parsed.condition_id, original_sources, role)
        require(files[source.path][source.start:source.end] == source.quote and
                files[source.path].count(source.quote) == 1, "actual_source_span_mismatch")
        catalog["paper/"+role] = {**paper.model_dump(), "block_sha256": hashlib.sha256(block.text.encode()).hexdigest()}
        catalog["source/"+role] = {**source.model_dump(), "sha256": indexed[source.path]}
    configuration = unique_json(files[plan.task.config])
    condition = plan.target_conditions[0]
    settings = {key: select(configuration, value) for key, value in parsed.settings_selectors.items()}
    require(set(settings) == set(condition.settings), "every_condition_setting_requires_actual_config_selector")
    actual = {"dataset": select(configuration, parsed.dataset_selector), "metric": select(configuration, parsed.metric_selector),
              "model": select(configuration, parsed.model_selector), "split": select(configuration, parsed.split_selector), "settings": settings}
    require(actual["dataset"] == condition.dataset and actual["metric"] == condition.metric and
            _fingerprint(settings) == _fingerprint(condition.settings), "actual_config_target_mismatch")
    require(all(actual[key] == condition.settings[key] for key in ("model","split") if key in condition.settings),
            "actual_model_or_split_config_mismatch")
    for name in (parsed.dataset_path, parsed.checkpoint_path):
        require(indexed.get(name) == file_sha(_actual_file(root, name)) == file_sha(_actual_file(running, name)), "selected_binary_or_partition_changed")
    rows_digest = hashlib.sha256()
    count = 0
    for row in partition_rows(_actual_file(root, parsed.dataset_path), parsed, policy):
        rows_digest.update(json.dumps(row, sort_keys=True, allow_nan=False).encode()+b"\n")
        count += 1
    binding = {"version": VERSION, "status": "bound", "proposal": parsed.model_dump(mode="json"),
               "policy": policy.model_dump(mode="json"), "scientific_claim": _scientific_claim(claim),
               "plan": plan.model_dump(mode="json"), "resource_contract": contract.model_dump(mode="json"),
               "source_files": files, "source_sha256": {name:indexed[name] for name in files}, "catalog": catalog,
               "source_sites":source_sites,
               "actual": actual, "partition_count": count, "partition_sha256": indexed[parsed.dataset_path],
               "partition_rows_sha256": rows_digest.hexdigest(), "checkpoint_sha256": indexed[parsed.checkpoint_path]}
    binding["binding_digest"] = _fingerprint(binding)
    return binding


def validate_inference_flow(binding, receipt, *, partition_path):
    """Actual identity edges plus independent full-data recomputation; no role grant."""
    require(receipt.get("status") == "received", "authenticated_inference_receipt_required")
    report = receipt["report"]
    require(report["binding_digest"] == binding["binding_digest"] and not report["unresolved"], "runtime_binding_unresolved")
    proposal = TorchInferenceProposal.model_validate(binding["proposal"])
    policy = TorchInferenceLimits.model_validate(binding["policy"])
    rows = report["samples"]
    require(type(rows) is list and len(rows) == binding["partition_count"], "complete_sample_denominator_required")
    require(all(type(row) is dict and type(row.get("ordinal")) is int for row in rows)
            and {row["ordinal"] for row in rows} == set(range(binding["partition_count"])), "sample_ordinal_partition_incomplete")
    rows = sorted(rows,key=lambda row:row["ordinal"])
    events = {row["invocation"]: row for row in report["calls"]}
    require(len(events) == len(report["calls"]), "duplicate_call_identity")
    if "seed" in binding["actual"]["settings"]:
        require(type(binding["actual"]["settings"]["seed"]) is int and all(row["torch_initial_seed"] == binding["actual"]["settings"]["seed"]
            for row in events.values() if row["role"] == "model"), "actual_torch_seed_unaligned")
    correct, squared, count = 0, 0.0, 0
    for original, observed in zip(partition_rows(partition_path, proposal, policy), rows, strict=True):
        require(_fingerprint({k:observed[k] for k in ("ordinal", "id", "features", "label")}) == _fingerprint(original), "actual_split_or_sample_mismatch")
        getitem, model, prediction, metric = (events[observed[key]] for key in ("getitem", "model", "prediction", "metric"))
        require([row["role"] for row in (getitem,model,prediction,metric)] == ["getitem","model","prediction","metric"], "actual_role_call_mismatch")
        require(getitem["order"] < model["order"] < prediction["order"] < metric["order"] and
                model["model_token"] == report["loaded_model_token"] and model["module_call_completed"] is True,
                "loaded_module_order_identity_mismatch")
        batch_index, row_index = observed["batch_index"], observed["row_index"]
        require(type(batch_index) is int and type(row_index) is int and batch_index >= 0 and row_index >= 0,
                "actual_batch_position_unknown")
        require(prediction["model_invocation"] == model["invocation"]
                and prediction["input_token"] == model["output_token"] and prediction["input"] == model["output"]
                and metric["predictions"][batch_index] == prediction["output"]
                and metric["labels"][batch_index] == model["batch_labels"]
                and model["batch_labels"]["token"] == getitem["label_batch_token"], "prediction_metric_actual_identity_missing")
        require(getitem["ordinal"] == original["ordinal"] and getitem["features"] == model["batch_features"]["values"][row_index]
                and _fingerprint(getitem["id"]) == _fingerprint(original["id"]) == _fingerprint(model["batch_ids"][row_index])
                and getitem["label"] == model["batch_labels"]["values"][row_index]
                and prediction["output"]["values"][row_index] == observed["prediction_value"], "actual_sample_tensor_binding_changed")
        require(_fingerprint(getitem["features"]) == _fingerprint(cast_numeric(original["features"], getitem["feature_snapshot"]["dtype"]))
                and _fingerprint(getitem["label"]) == _fingerprint(original["label"]), "actual_getitem_different_official_sample")
        value = observed["prediction_value"]
        require(type(value) in (int,float) and math.isfinite(value), "finite_actual_prediction_required")
        model_value = model["output"]["values"][row_index]
        if proposal.prediction_adapter == "argmax_last_axis":
            require(type(model_value) is list and model_value and all(type(v) in (int,float) and math.isfinite(v)
                for v in model_value), "classification_logits_adapter_unavailable")
            expected = max(range(len(model_value)),key=model_value.__getitem__)
        else:
            if proposal.prediction_adapter == "scalar_regression" and type(model_value) is list and len(model_value) == 1:
                model_value = model_value[0]
            require(type(model_value) in (int,float) and math.isfinite(model_value), "scalar_model_output_required")
            expected = model_value
        require(_fingerprint(expected) == _fingerprint(value), "actual_prediction_adapter_conflict")
        if proposal.metric_adapter == "accuracy":
            require(type(value) is int, "integer_actual_prediction_required")
            correct += value == original["label"]
        else:
            squared += (value-original["label"])**2
            require(math.isfinite(squared), "host_metric_overflow")
        count += 1
    value = correct/count if proposal.metric_adapter == "accuracy" else squared/count
    require(type(report["metric_value"]) in (int,float) and math.isfinite(report["metric_value"])
            and math.isclose(report["metric_value"], value, rel_tol=1e-7, abs_tol=1e-8), "actual_metric_host_conflict")
    require(len({row["metric"] for row in rows}) == 1, "whole_partition_single_metric_return_required")
    require(_fingerprint(select(receipt["raw_output"], proposal.raw_value_selector)) == _fingerprint(report["metric_value"]),
            "actual_metric_return_raw_output_conflict")
    return {"status":"witnessed", "value":value,"sample_count":count,"evidence_refs":receipt["evidence_refs"],
            "scientific_qualification":False,"alignment":False,"support":False}


def build_inference_context(proposal, *, plan, claim, materials, request, outcome, source_files,
                            runtime_stage_diagnostic=None):
    from .runtime_receipt import read_inference_receipt
    diagnostics = {}
    if runtime_stage_diagnostic is not None:
        from .inference_source_sites_diagnostics import RuntimeStageDiagnostic

        if type(runtime_stage_diagnostic) is not RuntimeStageDiagnostic:
            runtime_stage_diagnostic = None
        else:
            phase = "refreshed_context" if runtime_stage_diagnostic.phase == "refreshed_context" else "initial_context"
            RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, phase=phase, stage="entry")
            diagnostics["runtime_stage_diagnostic"] = runtime_stage_diagnostic
    try:
        binding = bind_inference_proposal(proposal, plan=plan, claim=claim, materials=materials,
            workspace=request.workspace, source_files=source_files, limits=request.config.docker_options.get("torch_inference"),
            source_sites=request.source_sites, **diagnostics)
        if runtime_stage_diagnostic is not None:
            RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="receipt")
        receipt = read_inference_receipt(request,outcome)
        if runtime_stage_diagnostic is not None:
            RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="flow")
        flow = validate_inference_flow(binding,receipt,partition_path=_actual_file(Path(request.workspace),proposal["dataset_path"]))
        observed = {"dataset":binding["actual"]["dataset"],"metric":binding["actual"]["metric"],
                    "settings":binding["actual"]["settings"],"value":flow["value"]}
        from .runner import Observation, aligned
        if runtime_stage_diagnostic is not None:
            RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="alignment")
        require(aligned(Observation.model_validate(observed), plan.target_conditions[0]), "actual_inference_condition_unaligned")
        if runtime_stage_diagnostic is not None:
            RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="context")
        catalog = dict(binding["catalog"], **{"actual/inference":{"binding_digest":binding["binding_digest"],
            "flow":flow,"runtime_framework":receipt["report"]["framework"], "producer_code_identity":receipt["producer_code_identity"],
            "loaded_state":receipt["report"]["loaded_state"], "actual":binding["actual"],
            "complete_samples":receipt["report"]["samples"],
            "complete_sample_count":flow["sample_count"],"source_files":binding["source_files"],
            "evidence_refs":receipt["evidence_refs"]}})
        obligations = {role:["paper/"+role,"source/"+role,"actual/inference"] for role in ROLES}
        for key in ("description","dataset","metric",*["settings/"+key for key in plan.target_conditions[0].settings]):
            role = {"dataset":"dataset","metric":"metric","settings/model":"model","settings/split":"split"}.get(key,"qualifiers")
            obligations["condition/"+key] = obligations[role]
        context = {"version":VERSION,"scientific_claim":binding["scientific_claim"],"condition":plan.target_conditions[0].model_dump(mode="json"),
                   "plan":plan.model_dump(mode="json"),"resource_contract":binding["resource_contract"],"catalog":catalog,
                   "obligations":obligations,"actual_observation":observed}
        context["context_digest"] = _fingerprint(context)
        if runtime_stage_diagnostic is not None:
            RuntimeStageDiagnostic.mark(runtime_stage_diagnostic, stage="context", terminal="passed")
        return context
    except Exception:
        if runtime_stage_diagnostic is not None:
            RuntimeStageDiagnostic.reject(runtime_stage_diagnostic)
        raise
