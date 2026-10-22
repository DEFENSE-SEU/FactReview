"""Read a fixed observer's host-captured receipt; no scientific or adversarial grant.

Caller supplies independently frozen RunRequest and the trusted production
RunOutcome. Their origin cannot be authenticated from self-reported digests.
No host code compilation, author execution or environment/service discovery.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .resource_contract import _fingerprint, _relative
from .runtime_launch import _bootstrap_scope, _bounded_metric_bytes, _path, protect_observer_docker_argv
from .runtime_observer import _snapshot

_PREPARE = {"version", "status", "unresolved", "scope_limits", "original_command", "workspace", "trusted_dir",
            "runtime_dir", "original_workdir", "container_cwd", "source_sites", "bootstrap_scope", "source_sha256",
            "audit_output", "package_files", "launch_command", "mounts", "request_sha256"}
_REPORT_FACTS = {"report_status", "report_sha256", "report_unresolved"}
_DEFAULT_ENV = {"EXECUTION_RUN_DIR": "/workspace/run_dir", "EXECUTION_ARTIFACT_DIR": "/workspace/run_dir/artifacts",
    "EXECUTION_PAPER_DIR": "/app", "EXECUTION_PAPER_ROOT": "/app", "PYTHONPATH": "/app", "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONPYCACHEPREFIX": "/workspace/run_dir/.pycache",
    "XDG_CACHE_HOME": "/workspace/run_dir/.cache", "HF_HOME": "/workspace/run_dir/.cache/huggingface",
    "MPLCONFIGDIR": "/workspace/run_dir/.cache/matplotlib"}


def _check(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _same(left, right):
    return json.dumps(left, sort_keys=True, ensure_ascii=True, allow_nan=False) == json.dumps(right, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _checked_snapshot(wire, capacity):
    """Decode closed wire containers, then apply the actual producer's limits."""
    remaining = [256]

    def decode(item, depth):
        remaining[0] -= 1
        _check(remaining[0] >= 0 and depth <= 12, "snapshot_capacity_exceeded")
        if type(item) in (type(None), bool, int, float, str):
            _check(type(item) is not str or len(item) <= capacity, "snapshot_capacity_exceeded")
            return item
        _check(type(item) is dict and set(item) == {"type", "items"}
               and type(item["type"]) is str and item["type"] in {"list", "tuple", "dict"}
               and type(item["items"]) is list and len(item["items"]) <= remaining[0], "unknown_snapshot_protocol")
        values = item["items"]
        if item["type"] == "dict":
            result = {}
            for pair in values:
                _check(type(pair) is list and len(pair) == 2 and type(pair[0]) is str
                       and len(pair[0]) <= capacity and pair[0] not in result, "unknown_snapshot_key")
                result[pair[0]] = decode(pair[1], depth + 1)
            return result
        result = [decode(value, depth + 1) for value in values]
        return tuple(result) if item["type"] == "tuple" else result

    _check(_same(_snapshot(decode(wire, 0), capacity), wire), "noncanonical_snapshot_protocol")


def _read(path, limit=2 * 1024 * 1024):
    with _path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    _check(len(raw) <= limit, "receipt_artifact_capacity")
    return raw


def _json(raw):
    def unique(pairs):
        _check(len(dict(pairs)) == len(pairs), "duplicate_receipt_key")
        return dict(pairs)
    return json.loads(raw, object_pairs_hook=unique)


def _file_sha(path):
    value = hashlib.sha256()
    with _path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _inference_report(request, outcome, *, report, audit, raw, paths, runtime_sites, hashes,
                      bootstrap, argv, cwd, file_output, policy):
    """New protocol shape and host sources, with no host compilation or grant."""
    import math
    req, result = request.model_dump(mode="json"), outcome.model_dump(mode="json")
    _check(type(report) is dict and report.get("version") == "torch-cpu-inference-v1"
           and report.get("unresolved") == [] and report.get("execution") == {"status":"completed"}
           and _sha(raw["report"]) == audit["report_sha256"]
           and report["binding_digest"] == audit["inference"]["binding_digest"], "inference_report_incomplete_or_changed")
    framework = report["framework"]
    _check(set(framework) == {"version","package_manifest_sha256","package_manifest_sha256_after","device"} and framework["device"] == "cpu"
           and framework["version"].split("+")[0] == policy.torch_version
           and framework["package_manifest_sha256"] == framework["package_manifest_sha256_after"] == policy.package_manifest_sha256,
           "actual_framework_unapproved")
    expected_hashes = {"/app/"+key:value for key,value in hashes.items()}
    _check(_same(report["source_hashes_before"], expected_hashes) and _same(report["source_hashes_after"], expected_hashes)
           and report["checkpoint_sha256"] == report["checkpoint_sha256_after"] == audit["inference"]["checkpoint_sha256"]
           and report["partition_sha256_after"] == audit["inference"]["partition_sha256"], "inference_runtime_sources_changed")
    _check(_same(report["entry"], {"argv":req["command"][1:],"cwd":cwd,
        "path":cwd+"/"+req["command"][1],"sha256":hashes[req["plan"]["task"]["entry_script"]]}), "inference_runtime_entry_changed")
    interpreter = report["interpreter"]
    _check(set(interpreter) == {"version","optimize"} and type(interpreter["optimize"]) is int and interpreter["optimize"] == 0
           and interpreter["version"].split()[0].startswith(bootstrap["declared_runtime_minor"]+"."), "inference_interpreter_unknown")
    _check(len(report["sites"]) == len(runtime_sites), "inference_runtime_sites_missing")
    for actual, expected in zip(report["sites"],runtime_sites,strict=True):
        _check(type(actual) is dict and set(actual) == set(expected)|{"code_sha256"}
               and _same(expected,{key:actual[key] for key in expected})
               and re.fullmatch(r"[a-f0-9]{64}",actual["code_sha256"]), "inference_runtime_code_identity_unknown")
    proposal = audit["inference"]["proposal"]
    calls = report["calls"]
    _check(type(calls) is list and 0 < len(calls) <= policy.max_events
           and len({row["invocation"] for row in calls}) == len(calls), "inference_call_capacity_or_duplicates")

    def tensor(value, *, with_values=True):
        _check(type(value) is dict and set(value) == {"token","shape","dtype","sha256","tensor_version"}|({"values"} if with_values else set()),
               "inference_tensor_protocol")
        _check(type(value["token"]) is int and value["token"] > 0 and type(value["tensor_version"]) is int
               and type(value["shape"]) is list and len(value["shape"]) <= 8
               and all(type(n) is int and n >= 0 for n in value["shape"])
               and type(value["dtype"]) is str and re.fullmatch(r"[a-f0-9]{64}",value["sha256"]), "inference_tensor_identity")
        if with_values:
            size = math.prod(value["shape"])
            _check(size <= policy.max_tensor_elements, "inference_tensor_capacity")
            def finite(item, dims):
                if not dims:
                    _check(type(item) in (int,float) and math.isfinite(item), "inference_tensor_value_type")
                else:
                    _check(type(item) is list and len(item) == dims[0], "inference_tensor_shape")
                    for child in item:
                        finite(child,dims[1:])
            finite(value["values"],value["shape"])
    _check(type(report["loaded_model_token"]) is int and report["loaded_model_token"] > 0
           and type(report["loaded_state"]) is dict and len(report["loaded_state"]) <= policy.max_events,
           "inference_loaded_state_unknown")
    for name, value in report["loaded_state"].items():
        _check(type(name) is str and name and type(value) is dict and type(value.get("checkpoint_member")) is bool,
               "inference_registered_state_protocol")
        tensor({"token":1,**{key:item for key,item in value.items() if key != "checkpoint_member"}},with_values=False)
    for row in calls:
        _check(type(row) is dict and row.get("role") in proposal["sites"]
               and type(row.get("site")) is int and row["site"] == proposal["sites"][row["role"]]
               and type(row.get("order")) is int and row["order"] == row["invocation"] > 0
               and (row.get("parent_invocation") is None or type(row["parent_invocation"]) is int), "inference_call_protocol")
        if row["role"] == "getitem":
            tensor(row["feature_snapshot"])
            _check(type(row["ordinal"]) is int and row["ordinal"] >= 0 and type(row["label"]) in (int,float)
                   and type(row["id"]) in (str,int), "inference_getitem_protocol")
        elif row["role"] == "model":
            for key in ("output","batch_features","batch_labels"):
                tensor(row[key])
            _check(row["module_call_completed"] is True and row["model_token"] == report["loaded_model_token"], "inference_final_module_return_unknown")
            _check(type(row["batch_ids"]) is list and all(type(value) in (str,int) for value in row["batch_ids"])
                   and type(row["torch_initial_seed"]) is int, "inference_actual_batch_metadata_unknown")
        elif row["role"] == "prediction":
            _check(type(row.get("model_invocation")) is int and 0 < row["model_invocation"] < row["invocation"],
                   "inference_prediction_model_call_unknown")
            tensor(row["input"])
            tensor(row["output"])
        else:
            _check(type(row["predictions"]) is list and type(row["labels"]) is list
                   and len(row["predictions"]) == len(row["labels"]) <= policy.max_samples
                   and type(row["value"]) in (int,float) and math.isfinite(row["value"]), "inference_metric_protocol")
            for item in row["predictions"]+row["labels"]:
                tensor(item)
    _check(type(report["samples"]) is list and len(report["samples"]) == audit["inference"]["partition_count"] <= policy.max_samples,
           "inference_sample_denominator_unknown")
    # Independently captured host inspections, never an author report/image label.
    for phase in ("before","after"):
        identity_path = paths["launch"].parent / ("inference_image_"+phase+".json")
        identity_raw = _read(identity_path)
        identity = _json(identity_raw)
        expected_command = ["docker","image","inspect","--format","{{.Id}}",result["environment"]["image"]]
        _check(identity == {"command":expected_command,"returncode":0,"termination":"completed","stdout":policy.image_id+"\n"}
               and result["commands"].count(expected_command) == 2, "actual_inference_image_identity_unknown")
        paths["image_"+phase],raw["image_"+phase] = identity_path,identity_raw
    _check(raw["stdout"].decode("utf-8").replace("\r\n","\n") == result["stdout"], "inference_host_stdout_changed")
    for key,name in (("runtime_observer","report"),("runtime_observer_launch","launch")):
        _check(_path(result["logs"][key]) == _path(paths[name]), "inference_host_audit_pointer_changed")
    raw_path = _path(result["logs"]["raw_output"])
    _check(raw_path == paths["launch"].parent / "raw_output.json", "inference_raw_output_pointer_changed")
    raw_payload = _json(_read(raw_path))
    if file_output is not None:
        _check(_same(raw_payload,file_output["payload"]), "inference_metric_file_raw_conflict")
    else:
        lines = result["stdout"].strip().splitlines()
        _check(len(lines) == 1 and _same(_json(lines[0].removeprefix("FACTREVIEW_OBSERVATIONS=").encode()),raw_payload),
               "inference_stdout_raw_conflict")
    refs = {key:{"path":str(path),"sha256":_sha(raw[key])} for key,path in paths.items()}
    refs["raw_output"] = {"path":str(raw_path),"sha256":_file_sha(raw_path)}
    return {"status":"received","report":report,"raw_output":raw_payload,"evidence_refs":refs,
        "run_identity":{"request_sha256":_fingerprint(req),"actual_argv_sha256":_fingerprint(argv),"image_id":policy.image_id},
        "producer_code_identity":{"basis":"fixed_inference_observer_actual_runtime_prepare_and_match",
                                  "sites":report["sites"],"interpreter":interpreter},
        "scientific_qualification":False,"alignment":False,"support":False,
        "limits":["trusted_host_request_outcome_required","in_process_tampering_not_excluded",
                  "cpu_eager_numeric_only","image_text_tokenizer_uncovered"]}


def read_observer_receipt(request, outcome):
    return _read_receipt(request, outcome, inference=False)


def read_inference_receipt(request, outcome):
    """Authenticate only the new fixed producer; old observer reports cannot upgrade."""
    return _read_receipt(request, outcome, inference=True)


def _read_receipt(request, outcome, *, inference):
    """received/unresolved; actual-runtime CodeType facts remain producer observations.

    Arbitrary custom callers or in-process malicious author tampering are outside
    this receipt. Internal SHA values bind bytes, never provide authentication.
    """
    try:
        req, result = request.model_dump(mode="json"), outcome.model_dump(mode="json")
        env = result["environment"]
        _check(type(result["returncode"]) is int and result["returncode"] == 0 and not result["operation_failures"]
               and env.get("transport") == "docker" and env.get("process_termination", {}).get("kind") == "completed",
               "producer_process_not_known_completed")
        _check(type(req["repair_round"]) is int and 0 <= req["repair_round"] <= 3 and (req["metric_output"] is None or type(req["metric_output"]) is str),
               "unsupported_attempt_or_output")
        root, run = _path(req["workspace"]), _path(req["run_dir"])
        attempt, scratch = _path(run / f"attempt_{req['repair_round']}"), _path(run / "runtime_scratch")
        trusted = _path(attempt / "observer_trusted")
        _check(root == run / "workspace" and not attempt.is_relative_to(scratch)
               and {child.name for child in trusted.iterdir()} == ({"observer.py", "config.json", "source_binding.py"}
                   if inference else {"observer.py", "config.json"}), "unknown_audit_layout")
        paths = {"launch": attempt / "runtime_observer_launch.json", "report": attempt / "runtime_observer.json",
                 "config": trusted / "config.json", "observer": trusted / "observer.py", "stdout": attempt / "run_stdout.log"}
        if inference:
            paths["source_binding"] = trusted / "source_binding.py"
        raw = {key: _read(path, 32 * 1024 * 1024 if key in {"report", "stdout"} else 2 * 1024 * 1024)
               for key, path in paths.items()}
        audit, config, report = (_json(raw[key]) for key in ("launch", "config", "report"))
        prepare_fields = _PREPARE | ({"metric_file"} if req["metric_output"] is not None else set())
        if inference:
            prepare_fields |= {"inference"}
        observer_env = env.get("runtime_observer")
        _check(type(observer_env) is dict, "unknown_launch_environment")
        capture = observer_env.get("capture")
        _check(set(observer_env) == set(audit) | ({"capture"} if req["metric_output"] is not None else set()), "unknown_capture_environment")
        _check(type(audit) is dict and set(audit) == prepare_fields | _REPORT_FACTS and audit["status"] == "observed"
               and audit["unresolved"] == [] and audit["report_status"] == "completed" and audit["report_unresolved"] == []
               and _same(audit, {key: value for key, value in observer_env.items() if key != "capture"}), "unknown_or_changed_launch_audit")
        prepared = {key: audit[key] for key in prepare_fields}
        prepared["status"] = "ready"
        sealed = {key: value for key, value in prepared.items() if key not in {"unresolved", "request_sha256"}}
        _check(_sha(json.dumps(sealed, sort_keys=True, ensure_ascii=True).encode()) == audit["request_sha256"], "prepare_seal_mismatch")
        _check(audit["version"] == ("readonly-torch-inference-v1" if inference else "readonly-python-observer-v1") and audit["workspace"] == str(root)
               and audit["trusted_dir"] == str(trusted) and audit["runtime_dir"] == str(scratch)
               and audit["original_workdir"] == req["workdir"] and _same(audit["original_command"], req["command"]), "request_launch_identity_mismatch")
        entry = _relative(req["plan"]["task"]["entry_script"])
        workdir = req["workdir"]
        _check(workdir == "." or _relative(workdir), "invalid_workdir")
        cwd = "/app" + ("/" + workdir if workdir != "." else "")
        command = req["command"]
        _check(type(command) is list and len(command) >= 2 and command[0] in {"python", "python3"}
               and _path(root / workdir / _relative(command[1])) == _path(root / entry) and ".factreview" not in Path(entry).parts,
               "original_entry_or_wrapper_changed")
        sites = req["source_sites"]
        _check(type(sites) is dict and sites.get("version") == "python-source-sites-v1" and sites.get("status") == "bound"
               and sites.get("unresolved") == [] and type(sites.get("sites")) is list and 1 <= len(sites["sites"]) <= 8
               and _same(audit["source_sites"], sites["sites"]), "source_selectors_unbound_or_changed")
        hashes, runtime_sites, seen = {entry: _file_sha(root / entry)}, [], set()
        for site in sites["sites"]:
            _check(type(site) is dict and set(site) == {"path", "sha256", "qualname", "firstlineno"}
                   and type(site["qualname"]) is str and site["qualname"] and type(site["firstlineno"]) is int
                   and site["firstlineno"] > 0, "source_selector_protocol")
            relative = _relative(site["path"])
            key = (relative, site["qualname"], site["firstlineno"])
            _check(key not in seen and Path(relative).suffix == ".py", "duplicate_or_unknown_source_selector")
            seen.add(key)
            hashes[relative] = _file_sha(root / relative)
            _check(hashes[relative] == site["sha256"], "source_bytes_changed")
            runtime_sites.append(dict(site, path="/app/" + relative))
        manifest = _json(_read(run / "source_manifest.json", 16 * 1024 * 1024))
        _check(type(manifest) is dict and all(manifest.get(key) == value for key, value in hashes.items())
               and _same(hashes, audit["source_sha256"]), "source_manifest_or_launch_hash_changed")
        contract = req["plan"]["task"].get("resource_contract")
        _check(type(contract) is dict and contract["claim_id"] == req["plan"]["claim_id"]
               and _same(contract["condition_ids"], req["plan"]["condition_ids"]), "resource_condition_identity_unbound")
        task = req["plan"]["task"]
        expected_resources = [("entry", entry)] + ([] if task["config"] is None else [("config", task["config"])])
        expected_resources += [(role, path) for role, field in (("data", "data_paths"), ("weights", "weight_paths")) for path in task[field]]
        _check([(row["role"], row["path"]) for row in contract["resources"]] == expected_resources
               and len(set(expected_resources)) == len(expected_resources)
               and [row["id"] for row in req["plan"]["target_conditions"]] == req["plan"]["condition_ids"], "task_resource_or_conditions_changed")
        for condition in req["plan"]["target_conditions"]:
            _check(contract["condition_sha256"].get(condition["id"]) == _fingerprint(condition), "condition_settings_changed")
        for resource in contract["resources"]:
            relative = _relative(resource["path"])
            _check(_file_sha(root / relative) == resource["sha256"] == manifest.get(relative), "selected_resource_changed")
        output = _path(audit["audit_output"])
        relative_output = output.relative_to(scratch).as_posix()
        _check(re.fullmatch(f"(?:observer_attempt_{req['repair_round']}_|o{req['repair_round']}_)[a-f0-9]{{32}}/events.json",
                           relative_output), "unknown_producer_output_path")
        expected_config = {"entry": command[1], "args": command[2:], "cwd": cwd, "sites": runtime_sites,
                           "output": "/workspace/run_dir/" + relative_output}
        if inference:
            from schemas.runtime_science import TorchInferenceProposal

            from .config import TorchInferenceLimits
            nn = audit["inference"]
            _check(type(nn) is dict and set(nn) == {"binding_digest","proposal","policy","source_sha256",
                "partition_sha256","partition_count","checkpoint_sha256"}, "inference_config_protocol")
            TorchInferenceProposal.model_validate(nn["proposal"])
            policy = TorchInferenceLimits.model_validate(nn["policy"])
            _check(_same(nn["proposal"], TorchInferenceProposal.model_validate(req["source_science"]).model_dump(mode="json"))
                   and _same(nn["policy"],TorchInferenceLimits.model_validate(req["config"]["docker_options"].get("torch_inference")).model_dump(mode="json")),
                   "inference_proposal_or_policy_changed")
            for name, expected in nn["source_sha256"].items():
                _check(_file_sha(root / _relative(name)) == manifest.get(name) == expected, "inference_complete_source_changed")
            _check(nn["partition_sha256"] == manifest[nn["proposal"]["dataset_path"]]
                   and nn["checkpoint_sha256"] == manifest[nn["proposal"]["checkpoint_path"]], "inference_resources_changed")
            expected_config["inference"] = nn
        _check(_same(config, expected_config) and audit["container_cwd"] == cwd
               and audit["launch_command"] == [command[0], "/factreview-observer/observer.py", "/factreview-observer/config.json"], "actual_config_changed")
        expected_package = {"observer.py": _sha(raw["observer"]), "config.json": _sha(raw["config"])}
        if inference:
            _check(_sha(raw["source_binding"]) == _file_sha(Path(__file__).with_name("runtime_observer.py")), "inference_source_binding_changed")
            expected_package["source_binding.py"] = _sha(raw["source_binding"])
        _check(_sha(raw["observer"]) == _file_sha(Path(__file__).with_name("inference_observer.py" if inference else "runtime_observer.py"))
               and _same(audit["package_files"], expected_package), "observer_package_changed")
        bootstrap = _bootstrap_scope(root, _path(root / entry), _path(root / workdir), req["config"]["python_version"])
        bootstrap.update(cache_path="/workspace/run_dir/.pycache", cache_absent_at_prepare=True)
        expected_mounts = {
            "/app": {"source": str(root), "mode": "ro"}, "/factreview-observer": {"source": str(trusted), "mode": "ro"},
            "/workspace/run_dir": {"source": str(scratch), "mode": "rw"}}
        file_output = None
        if req["metric_output"] is not None:
            leaf = audit["metric_file"]
            placeholder = root / workdir / _relative(req["metric_output"])
            target = "/app/" + placeholder.relative_to(root).as_posix()
            source = output.parent / "metric_output.json"
            _check(type(leaf) is dict and _same(leaf, {"relative_path": req["metric_output"],
                "workspace_placeholder": str(placeholder), "capture_path": str(source), "container_path": target,
                "initial_sha256": _sha(b""), "initial_bytes": 0, "original_absent": True})
                and placeholder.relative_to(root).as_posix() not in manifest
                and _bounded_metric_bytes(placeholder) == b"", "metric_output_identity_or_source_changed")
            expected_mounts[target] = {"source": str(source), "mode": "rw"}
            metadata_path, host_copy = attempt / "metric_file_capture.json", attempt / "raw_metric_file.json"
            metadata_raw, captured_raw = _bounded_metric_bytes(metadata_path), _bounded_metric_bytes(host_copy)
            record = _json(metadata_raw)
            _check(type(capture) is dict and _same(capture, {"record": record,
                "metadata": {"path": str(metadata_path), "sha256": _sha(metadata_raw)}}), "metric_capture_environment_changed")
            _check(_same(record, {"version": "observer-json-file-capture-v1", "metric_output": req["metric_output"],
                "workdir": workdir, "repair_round": req["repair_round"], "launch_path": str(paths["launch"]),
                "launch_sha256": _sha(raw["launch"]), "prepare_request_sha256": audit["request_sha256"],
                "capture_path": str(source), "host_copy": str(host_copy), "sha256": _sha(captured_raw),
                "bytes": len(captured_raw), "original_absent": True})
                and captured_raw and captured_raw == _bounded_metric_bytes(source)
                and result["logs"].get("raw_metric_file") == str(host_copy)
                and result["logs"].get("metric_file_capture") == str(metadata_path), "metric_capture_bytes_or_identity_changed")
            file_output = {"payload": _json(captured_raw), "evidence_refs": {
                "raw_metric_file": {"path": str(host_copy), "sha256": _sha(captured_raw)},
                "metric_file_capture": {"path": str(metadata_path), "sha256": _sha(metadata_raw)}}}
        _check(_same(bootstrap, audit["bootstrap_scope"]) and _same(audit["mounts"], expected_mounts), "protection_identity_changed")
        runs = [argv for argv in result["commands"] if type(argv) is list and argv[:2] == ["docker", "run"]]
        _check(len(runs) == 1, "actual_run_argv_not_unique")
        argv = runs[0]
        sandbox = req["config"].get("sandbox")
        protected = protect_observer_docker_argv(argv, launch=prepared, sandbox=sandbox)
        _check(protected["status"] == "ready" and protected["argv"] == argv, "actual_argv_not_protected")
        image_index = len(argv) - len(audit["launch_command"]) - 1
        _check(argv[image_index] == env.get("image") and argv.count("--name") == 1
               and argv[argv.index("--name") + 1] == env.get("container_name"), "actual_container_identity_changed")
        actual_env = {}
        for index, token in enumerate(argv[:image_index]):
            if token in {"-e", "--env"}:
                name, separator, value = argv[index + 1].partition("=")
                _check(separator and name not in actual_env, "duplicate_or_inherited_environment")
                actual_env[name] = value
        expected_env = dict(_DEFAULT_ENV, FACTREVIEW_REPAIR_ROUND=str(req["repair_round"]))
        if sandbox is not None:
            from .config import SANDBOX_ENV, sandbox_policy

            approved = sandbox_policy(sandbox)
            expected_env.update(SANDBOX_ENV)
            _check(_same(env.get("sandbox"), {"policy": approved.model_dump(mode="json"),
                "user": f"{approved.uid}:{approved.gid}", "network": approved.network,
                "author_transport_invoked": True, "argv": argv, "runtime_enforcement": "unverified"}),
                "sandbox_policy_or_actual_argv_changed")
        _check(all(actual_env.get(key) == value for key, value in expected_env.items())
               and set(actual_env).issubset(set(expected_env) | {"HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"}), "declared_environment_changed_or_unknown")
        if inference:
            return _inference_report(request, outcome, report=report, audit=audit, raw=raw, paths=paths,
                runtime_sites=runtime_sites, hashes=nn["source_sha256"], bootstrap=bootstrap,
                argv=argv, cwd=cwd, file_output=file_output, policy=policy)
        _check(type(report) is dict and report.get("version") == "python-source-events-v1" and report.get("unresolved") == []
               and report.get("execution", {}).get("status") == "completed" and _sha(raw["report"]) == audit["report_sha256"], "report_changed_or_incomplete")
        _check(report.get("scope_limits") == ["in_process_tampering_not_excluded", "events_do_not_prove_scientific_roles",
               "native_and_child_execution_not_covered"], "unknown_producer_scope")
        _check(_same(report["entry"], {"argv": command[1:], "cwd": cwd, "path": "/app/" + entry, "sha256": hashes[entry]})
               and _same(report["source_hashes_after"], {"/app/" + key: value for key, value in hashes.items()}), "runtime_source_or_entry_changed")
        interpreter = report["interpreter"]
        _check(type(interpreter) is dict and set(interpreter) == {"version", "executable", "optimize"}
               and type(interpreter["version"]) is str and re.match(r"3\.\d+\.\d+(?:\s|$)", interpreter["version"])
               and interpreter["version"].split()[0].startswith(bootstrap["declared_runtime_minor"] + ".")
               and type(interpreter["executable"]) is str and interpreter["executable"].startswith("/")
               and type(interpreter["optimize"]) is int and interpreter["optimize"] == 0, "runtime_interpreter_unknown")
        _check(type(report["sites"]) is list and len(report["sites"]) == len(runtime_sites), "actual_runtime_sites_missing")
        for expected, actual in zip(runtime_sites, report["sites"], strict=True):
            _check(type(actual) is dict and set(actual) == set(expected) | {"code_sha256"}
                   and _same(expected, {key: actual[key] for key in expected})
                   and type(actual["code_sha256"]) is str and re.fullmatch(r"[a-f0-9]{64}", actual["code_sha256"]), "actual_runtime_site_identity_changed")
        _check(type(report["events"]) is list and 0 < len(report["events"]) <= 256
               and all(type(event) is dict and event.get("event") in {"call", "return"} and type(event.get("order")) is int
                       and event["order"] == index for index, event in enumerate(report["events"], 1)), "event_scope_unknown")
        active, invoked, seen_sites = [], set(), set()
        for event in report["events"]:
            site, invocation = event.get("site"), event.get("invocation")
            _check(type(site) is int and 0 <= site < len(runtime_sites) and type(invocation) is int and invocation > 0, "unknown_event_identity")
            _check(set(event) == ({"event", "site", "invocation", "order", "parent_invocation", "arguments"} if event["event"] == "call" else {"event", "site", "invocation", "order", "value"}), "unknown_event_protocol")
            if event["event"] == "call":
                _check(type(event["arguments"]) is dict, "unknown_call_arguments_protocol")
                capacity = 65536
                for name, value in event["arguments"].items():
                    _check(type(name) is str, "unknown_call_argument_name")
                    _checked_snapshot(value, max(0, capacity))
                    capacity -= len(json.dumps(value, ensure_ascii=True).encode()) + len(name) * 6 + 8
                    _check(capacity > 0, "argument_snapshot_capacity_exceeded")
                _check(invocation not in invoked and _same(event.get("parent_invocation"), active[-1][0] if active else None), "unknown_call_parent")
                active.append((invocation, site))
                invoked.add(invocation)
                seen_sites.add(site)
            else:
                _checked_snapshot(event["value"], 65536)
                _check(active and active.pop() == (invocation, site), "missing_or_unmatched_call_return")
        _check(not active and seen_sites == set(range(len(runtime_sites))), "unobserved_or_incomplete_sites")
        _check(raw["stdout"].decode("utf-8").replace("\r\n", "\n") == result["stdout"], "host_stdout_changed")
        for key, expected in (("runtime_observer", paths["report"]), ("runtime_observer_launch", paths["launch"])):
            _check(_path(result["logs"][key]) == _path(expected), "host_audit_pointer_changed")
        _check(_path(result["logs"]["stdout"]) in {_path(paths["stdout"]), attempt / "stdout.log"}
               and _read(result["logs"]["stdout"], 32 * 1024 * 1024).decode("utf-8").replace("\r\n", "\n") == result["stdout"], "stdout_pointer_or_bytes_changed")
        refs = {key: {"path": str(path), "sha256": _sha(raw[key])} for key, path in paths.items()}
        received = {"status": "received", "report": report, "run_identity": {"command": command, "cwd": cwd, "runtime_root": "/app",
            "repair_round": req["repair_round"], "launch_sha256": _sha(raw["launch"]), "config_sha256": _sha(raw["config"]),
            "request_sha256": _fingerprint(req), "actual_argv_sha256": _fingerprint(argv), "source_sha256": hashes}, "evidence_refs": refs,
            "producer_code_identity": {"basis": "fixed_observer_actual_runtime_prepare_and_match", "sites": report["sites"], "interpreter": interpreter},
            "scientific_qualification": False, "alignment": False, "support": False,
            "limits": ["caller_request_and_outcome_origin_must_be_trusted", "in_process_author_tampering_not_excluded",
                       "internal_hashes_are_not_authentication", "proxy_history_not_reconstructed", "scientific_roles_and_stdout_metric_binding_unproved"]}
        if file_output is not None:
            received["metric_file_output"] = file_output
        return received
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, OverflowError, RecursionError) as exc:
        return {"status": "unresolved", "reason": str(exc) if type(exc) is ValueError else type(exc).__name__,
                "scientific_qualification": False, "alignment": False, "support": False}
