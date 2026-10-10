"""Finite source-linked JSON value flow, never a scientific alignment grant.

Only one-file straight-line Python with two JSON reads and bounded numeric return
expressions is interpreted. No author code, callback, getter, eval or exec runs here.
The consumer must run in the observer's exact interpreter to check full code identity;
cross-interpreter host compilation is explicitly unresolved. Caller must freeze the
original plan/request and authenticate the readonly observer receipt separately.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import operator
import re
import sys
import types
from pathlib import Path, PurePosixPath

from fact_generation.execution.resource_contract import _actual_file, validate_resource_contract
from fact_generation.execution.runtime_observer import _code_sha, _snapshot

VERSION = "builtin-json-flow-v1"
ROLES = ("driver", "reader", "inference", "metric")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _same(left, right):
    # Python equality merges bool/int/float; snapshots must retain exact types.
    return _sha(left) == _sha(right)


def _failure(reason):
    return {"status": "unresolved", "reason": reason, "scientific_qualification": False,
            "alignment": False, "support": False}


def _bounded_bytes(path, reason):
    with path.open("rb") as stream:
        raw = stream.read(65537)
    _require(len(raw) <= 65536, reason)
    return raw


def _params(node):
    args = node.args
    _require(not (node.decorator_list or args.posonlyargs or args.kwonlyargs or args.defaults
                  or args.kw_defaults or args.vararg or args.kwarg or node.returns)
             and all(not row.annotation for row in args.args), "unsupported_function_signature")
    return [row.arg for row in args.args]


def _selector(node):
    keys = []
    while isinstance(node, ast.Subscript):
        _require(isinstance(node.slice, ast.Constant) and type(node.slice.value) in (str, int), "unsafe_selector")
        key = node.slice.value
        _require((type(key) is str and 0 < len(key) <= 128) or (type(key) is int and 0 <= key < 256), "unsafe_selector")
        keys.insert(0, key)
        node = node.value
    _require(isinstance(node, ast.Name), "selector_requires_argument")
    return node.id, keys


def _expression(node, parameters):
    _require(sum(1 for _ in ast.walk(node)) <= 64, "expression_capacity")
    leaves = []

    def visit(item):
        if isinstance(item, (ast.Name, ast.Subscript)):
            name, keys = _selector(item)
            _require(name in parameters, "expression_unknown_name")
            leaves.append((name, keys))
        elif isinstance(item, ast.Constant):
            _number(item.value)
        elif isinstance(item, ast.BinOp) and type(item.op) in OPS:
            visit(item.left)
            visit(item.right)
        else:
            raise ValueError("unsupported_numeric_expression")
    visit(node)
    _require(set(name for name, _ in leaves) == set(parameters), "unused_return_parameter")
    return leaves


def _number(value):
    _require(type(value) in (int, float) and (type(value) is not int or value.bit_length() <= 256)
             and math.isfinite(value), "nonfinite_or_nonbuiltin_numeric_value")
    return value


def _get(value, keys):
    for key in keys:
        _require((type(value) is dict and type(key) is str and key in value)
                 or (type(value) is list and type(key) is int and key < len(value)), "selector_missing_or_type")
        value = value[key]
    return value


def _eval(node, arguments):
    if isinstance(node, (ast.Name, ast.Subscript)):
        name, keys = _selector(node)
        return _number(_get(arguments[name], keys))
    if isinstance(node, ast.Constant):
        return _number(node.value)
    _require(isinstance(node, ast.BinOp) and type(node.op) in OPS, "unsupported_numeric_expression")
    return _number(OPS[type(node.op)](_eval(node.left, arguments), _eval(node.right, arguments)))


def _tree(proposal, source, sites):
    _require(type(source) is str and len(source) <= 65536 and len(source.encode()) <= 65536, "source_capacity")
    tree = ast.parse(source)
    functions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            _require(node.name not in functions, "duplicate_static_call_target")
            functions[node.name] = node
    _require(len(functions) == 4 and len(tree.body) == 7
             and ast.dump(tree.body[0]) == ast.dump(ast.parse("import json").body[0])
             and ast.dump(tree.body[1]) == ast.dump(ast.parse("from pathlib import Path").body[0]),
             "unsupported_module_layout_or_rebinding")
    selected = {}
    for role in ROLES:
        row = proposal["roles"][role]
        _require(type(row) is dict and set(row) == {"site", "quote"} and type(row["site"]) is int
                 and 0 <= row["site"] < len(sites), "role_selector_protocol")
        site = sites[row["site"]]
        node = functions.get(site["qualname"])
        _require(node is not None and node.lineno == site["firstlineno"]
                 and type(row["quote"]) is str and ast.get_source_segment(source, node) == row["quote"]
                 and source.count(row["quote"]) == 1, "role_quote_or_site_mismatch")
        selected[role] = node
    _require(len({node.name for node in selected.values()}) == 4, "duplicate_role_site")
    driver, reader, infer, metric = (selected[role] for role in ROLES)
    _require(_params(driver) == [] and len(_params(reader)) == 1
             and len(_params(infer)) == len(_params(metric)) == 2, "unsupported_role_signature")
    expected = ast.parse(f"def f({_params(reader)[0]}):\n return json.loads(Path({_params(reader)[0]}).read_text())").body[0]
    _require(len(reader.body) == 1 and ast.dump(reader.body[0]) == ast.dump(expected.body[0]), "unsupported_json_reader")
    for node in (infer, metric):
        _require(len(node.body) == 1 and isinstance(node.body[0], ast.Return), "nonlinear_or_complex_role")
        _expression(node.body[0].value, _params(node))
    _require(len(driver.body) == 5 and all(isinstance(row, ast.Assign) and len(row.targets) == 1
             and isinstance(row.targets[0], ast.Name) for row in driver.body[:4]), "nonlinear_driver")
    names = [row.targets[0].id for row in driver.body[:4]]
    _require(len(set(names)) == 4 and not set(names) & (set(functions) | {"json", "Path", "print"}), "reassigned_driver_value")
    data, weights, prediction, value = names
    for index, role in enumerate(("reader", "reader", "inference", "metric")):
        call = driver.body[index].value
        _require(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                 and call.func.id == selected[role].name and not call.keywords, "nonunique_static_call_edge")
    resources = []
    for row in driver.body[:2]:
        _require(len(row.value.args) == 1 and isinstance(row.value.args[0], ast.Constant)
                 and type(row.value.args[0].value) is str, "reader_requires_exact_resource_literal")
        resources.append(row.value.args[0].value)
    _require(ast.dump(driver.body[2].value) == ast.dump(ast.parse(f"{infer.name}({data}, {weights})", mode="eval").body),
             "inference_inputs_not_reader_returns")
    call = driver.body[3].value
    _require(len(call.args) == 2 and isinstance(call.args[0], ast.Name) and call.args[0].id == prediction,
             "metric_prediction_not_inference_return")
    label_name, label_keys = _selector(call.args[1])
    _require(label_name == data and label_keys and isinstance(driver.body[4], ast.Return)
             and isinstance(driver.body[4].value, ast.Name) and driver.body[4].value.id == value,
             "metric_labels_or_driver_return_edge")
    _require(ast.dump(tree.body[-1]) == ast.dump(ast.parse(f"print({driver.name}())").body[0]), "unsupported_entry_output")
    return selected, resources, label_keys


def bind_builtin_flow(proposal, *, plan, claim, materials, supplied_files, source_sites):
    """Rebuild original scientific/resource identities; model input only selects sites/quotes."""
    try:
        _require(type(proposal) is dict and set(proposal) == {"version", "roles", "conditions"}
                 and proposal["version"] == VERSION and type(proposal["roles"]) is dict
                 and set(proposal["roles"]) == set(ROLES), "unknown_flow_protocol")
        contract = validate_resource_contract(plan, claim, materials)
        _require(contract is not None and len(plan.condition_ids) == 1 and plan.run_mode == "evaluation", "unbound_or_unsupported_plan")
        _require(len(plan.task.data_paths) == len(plan.task.weight_paths) == 1, "resource_scope_requires_one_data_and_weight")
        _require(type(source_sites) is dict and source_sites.get("status") == "bound"
                 and type(source_sites.get("sites")) is list and len(source_sites["sites"]) == 4, "unbound_source_sites")
        sites = source_sites["sites"]
        root = Path(materials.repository.root).resolve(strict=True)
        path = plan.task.entry_script
        _require(all(type(row) is dict and set(row) == {"path", "sha256", "qualname", "firstlineno"}
                     and row["path"] == path and type(row["qualname"]) is str
                     and type(row["firstlineno"]) is int for row in sites), "unsupported_or_unavailable_site")
        _require(type(supplied_files) is dict and type(supplied_files.get(path)) is str
                 and len(supplied_files[path]) <= 65536, "source_capacity_or_unread")
        actual = _actual_file(root, path)
        raw_source = _bounded_bytes(actual, "source_capacity")
        # Match read_text's universal-newline view; hash the same original bytes.
        source = raw_source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        source_sha = hashlib.sha256(raw_source).hexdigest()
        _require(supplied_files[path] == source
                 and all(row["sha256"] == source_sha for row in sites), "changed_or_unread_source")
        functions, resources, label_keys = _tree(proposal, source, sites)
        _require(resources == [plan.task.data_paths[0], plan.task.weight_paths[0]] and plan.task.workdir == ".",
                 "selected_resources_not_actual_reader_literals")
        _require(type(proposal["conditions"]) is list and len(proposal["conditions"]) == 1, "condition_coverage_incomplete")
        row, condition = proposal["conditions"][0], plan.target_conditions[0]
        _require(type(row) is dict and set(row) == {"condition", "paper_quote", "model_rationale", "metric_rationale"}
                 and _same(row["condition"], condition.model_dump(mode="json"))
                 and condition.dataset and condition.metric and condition.settings.get("model") and condition.settings.get("split")
                 and all(type(row[key]) is str and row[key].strip() for key in ("paper_quote", "model_rationale", "metric_rationale")),
                 "paper_model_or_all_condition_proposals_missing")
        blocks = [block for block in materials.blocks if block.id == claim.source_block_id]
        _require(len(blocks) == 1 and row["paper_quote"] == claim.source_quote
                 and blocks[0].text.count(claim.source_quote) == 1, "paper_source_unavailable_or_ambiguous")
        return {"status": "bound", "version": VERSION, "proposal": copy.deepcopy(proposal),
                "source_sites": copy.deepcopy(source_sites), "source_sha256": source_sha,
                "resource_contract": contract.model_dump(mode="json"), "plan_sha256": _sha(plan.model_dump(mode="json")),
                "label_selector": label_keys, "role_names": {key: node.name for key, node in functions.items()},
                "scientific_qualification": False, "alignment": False, "support": False}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, SyntaxError, OverflowError, RecursionError) as exc:
        return _failure(str(exc) if type(exc) is ValueError else type(exc).__name__)


def _decode(snapshot):
    def visit(item, depth=0):
        _require(depth <= 12, "snapshot_capacity")
        if type(item) in (type(None), bool, int, float, str):
            return item
        _require(type(item) is dict and set(item) == {"type", "items"} and type(item["items"]) is list
                 and len(item["items"]) <= 256, "invalid_builtin_snapshot")
        values = item["items"]
        if item["type"] == "dict":
            _require(all(type(pair) is list and len(pair) == 2 and type(pair[0]) is str for pair in values)
                     and len({pair[0] for pair in values}) == len(values), "invalid_dict_snapshot")
            return {key: visit(value, depth + 1) for key, value in values}
        _require(item["type"] in ("list", "tuple"), "invalid_builtin_snapshot")
        result = [visit(value, depth + 1) for value in values]
        return tuple(result) if item["type"] == "tuple" else result
    value = visit(snapshot)
    _require(_same(_snapshot(value, 65536), snapshot), "noncanonical_snapshot")
    return value


def _json_file(path):
    def unique(pairs):
        _require(len(dict(pairs)) == len(pairs), "duplicate_json_key")
        return dict(pairs)
    raw = _bounded_bytes(path, "resource_capacity")
    value = json.loads(raw, object_pairs_hook=unique)
    _snapshot(value, 65536)
    return value


def validate_builtin_flow(binding, *, plan, claim, materials, supplied_files, source_sites,
                          observer_report=None, run_request=None, raw_value=None):
    """Witness only finite source-flow; raw stdout and scientific qualification stay unresolved.

    run_request is the independently frozen runner request, not an author report.
    Same-interpreter validation can run in the trusted container; host/runtime
    version differences must not be accepted by a host-compiled code fingerprint.
    """
    try:
        _require(type(binding) is dict and binding.get("status") == "bound", "unbound_flow")
        rebuilt = bind_builtin_flow(binding["proposal"], plan=plan, claim=claim, materials=materials,
                                    supplied_files=supplied_files, source_sites=source_sites)
        _require(_same(rebuilt, binding) and rebuilt["status"] == "bound", "changed_binding")
        report, request = observer_report, run_request
        _require(type(report) is dict and report.get("version") == "python-source-events-v1"
                 and report.get("unresolved") == [] and report.get("execution", {}).get("status") == "completed", "incomplete_observation")
        _require(type(report["interpreter"]["version"]) is str and report["interpreter"]["version"] == sys.version
                 and type(report["interpreter"]["optimize"]) is int and report["interpreter"]["optimize"] == sys.flags.optimize,
                 "cross_interpreter_code_identity_unresolved")
        _require(type(request) is dict and set(request) == {"command", "cwd", "runtime_root", "repair_round", "launch_sha256", "config_sha256"}
                 and type(request["repair_round"]) is int and 0 <= request["repair_round"] <= 3
                 and all(type(request[key]) is str and re.fullmatch(r"[a-f0-9]{64}", request[key]) for key in ("launch_sha256", "config_sha256"))
                 and request["command"] == plan.task.command and request["command"] in (["python", plan.task.entry_script], ["python3", plan.task.entry_script]),
                 "missing_or_changed_run_request")
        runtime_root = request["runtime_root"]
        _require(type(runtime_root) is str and request["cwd"] == runtime_root and report["entry"]["cwd"] == runtime_root
                 and report["entry"]["argv"] == [plan.task.entry_script], "run_entry_or_cwd_mismatch")
        entry_path = str(Path(runtime_root) / plan.task.entry_script) if not runtime_root.startswith("/") else str(PurePosixPath(runtime_root) / plan.task.entry_script)
        _require(report["entry"]["path"] == entry_path and report["entry"]["sha256"] == binding["source_sha256"], "entry_identity_mismatch")
        source = supplied_files[plan.task.entry_script]
        functions, resources, label_keys = _tree(binding["proposal"], source, source_sites["sites"])
        compiled = compile(source.encode("utf-8"), entry_path, "exec", dont_inherit=True, optimize=sys.flags.optimize)
        codes = {code.co_name: code for code in compiled.co_consts if type(code) is types.CodeType}
        actual_sites = report["sites"]
        _require(type(actual_sites) is list and len(actual_sites) == 4, "actual_sites_missing")
        for index, selected in enumerate(source_sites["sites"]):
            expected = {"path": entry_path, "sha256": binding["source_sha256"], "qualname": selected["qualname"],
                        "firstlineno": selected["firstlineno"], "code_sha256": _code_sha(codes[selected["qualname"]])}
            _require(_same(actual_sites[index], expected) and report["source_hashes_after"].get(entry_path) == binding["source_sha256"], "actual_code_identity_mismatch")
        events = report["events"]
        roles = binding["proposal"]["roles"]
        indices = [roles[key]["site"] for key in ("driver", "reader", "reader", "inference", "metric")]
        pattern = [("call", 0), ("call", 1), ("return", 1), ("call", 2), ("return", 2),
                   ("call", 3), ("return", 3), ("call", 4), ("return", 4), ("return", 0)]
        _require(type(events) is list and len(events) == 10, "nonunique_or_missing_invocations")
        for order, (event, (kind, number)) in enumerate(zip(events, pattern, strict=True), 1):
            _require(type(event) is dict and set(event) == ({"event", "site", "invocation", "order", "parent_invocation", "arguments"}
                     if kind == "call" else {"event", "site", "invocation", "order", "value"})
                     and event["event"] == kind and type(event["site"]) is int and event["site"] == indices[number]
                     and type(event["order"]) is int and event["order"] == order
                     and type(event["invocation"]) is int and event["invocation"] == number + 1, "event_identity_or_order")
            if kind == "call":
                _require(event["parent_invocation"] is None if number == 0 else
                         type(event["parent_invocation"]) is int and event["parent_invocation"] == 1,
                         "wrong_parent_invocation")
        root = Path(materials.repository.root).resolve(strict=True)
        data, weights = [_json_file(_actual_file(root, path)) for path in resources]
        infer, metric = functions["inference"], functions["metric"]
        inference_args = dict(zip(_params(infer), (data, weights), strict=True))
        predicted = _eval(infer.body[0].value, inference_args)
        labels = _number(_get(data, label_keys))
        metric_args = dict(zip(_params(metric), (predicted, labels), strict=True))
        value = _eval(metric.body[0].value, metric_args)
        reader_param = _params(functions["reader"])[0]
        expected_calls = [{}, {reader_param: resources[0]}, {reader_param: resources[1]}, inference_args, metric_args]
        expected_returns = [value, data, weights, predicted, value]
        for event, (_, number) in zip(events, pattern, strict=True):
            if event["event"] == "call":
                _require(type(event["arguments"]) is dict and set(event["arguments"]) == set(expected_calls[number])
                         and all(_same(_snapshot(expected_calls[number][key], 65536), snapshot)
                                 for key, snapshot in event["arguments"].items()), "actual_argument_edge_mismatch")
            else:
                _require(_same(_snapshot(expected_returns[number], 65536), event["value"]), "actual_return_edge_mismatch")
                _decode(event["value"])
        # A syntactic name occurrence cannot establish value influence (w-w, x*0).
        # Require an independently recomputed changed result for every selected leaf.
        leaves = _expression(infer.body[0].value, _params(infer))
        for name, keys in leaves:
            _require(keys, "numeric_resource_selector_required")
            changed = copy.deepcopy(inference_args)
            original = _number(_get(changed[name], keys))
            sensitive = False
            for delta in (1, -1, 2):
                parent = _get(changed[name], keys[:-1])
                parent[keys[-1]] = _number(original + delta)
                altered_prediction = _eval(infer.body[0].value, changed)
                altered = _eval(metric.body[0].value, dict(zip(_params(metric), (altered_prediction, labels), strict=True)))
                sensitive |= type(altered) is type(value) and altered != value
            _require(sensitive, "input_or_weights_no_observed_numeric_influence")
        _require(any(_eval(metric.body[0].value, dict(zip(_params(metric), (predicted, labels + delta), strict=True))) != value
                     for delta in (1, -1, 2)), "labels_no_observed_numeric_influence")
        _require(type(raw_value) is type(value) and _number(raw_value) == value, "raw_value_mismatch")
        references = {
            "source_roles": {role: {"path": plan.task.entry_script, "sha256": binding["source_sha256"],
                "firstlineno": functions[role].lineno, "qualname": functions[role].name,
                "binding_pointer": f"/proposal/roles/{role}"} for role in ROLES},
            "runtime_events": {role: {"call": f"/events/{call}", "return": f"/events/{returned}"}
                for role, call, returned in (("driver", 0, 9), ("data_reader", 1, 2),
                    ("weights_reader", 3, 4), ("inference", 5, 6), ("metric", 7, 8))},
            "resources": binding["resource_contract"]["resources"],
            "conditions": binding["resource_contract"]["condition_sha256"],
            "request_identity": {key: request[key] for key in ("launch_sha256", "config_sha256", "repair_round")},
        }
        return {"status": "witnessed", "version": VERSION, "value": value,
                "data_participation": True, "weights_participation": True, "labels_participation": True,
                "request_sha256": _sha(request), "observer_sha256": _sha(report), "binding_sha256": _sha(binding),
                "evidence_refs": references,
                "raw_output_binding": "unresolved", "scientific_qualification": False, "alignment": False, "support": False,
                "limits": ["bounded_numeric_sensitivity_only", "paper_roles_remain_proposals",
                           "readonly_launcher_receipt_requires_independent_authentication", "in_process_tampering_not_excluded"]}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, SyntaxError, OverflowError,
            ZeroDivisionError, RecursionError) as exc:
        return _failure(str(exc) if type(exc) is ValueError else type(exc).__name__)
