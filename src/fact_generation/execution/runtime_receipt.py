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
from .runtime_launch import _bootstrap_scope, _path, protect_observer_docker_argv
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


def read_observer_receipt(request, outcome):
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
        _check(type(req["repair_round"]) is int and 0 <= req["repair_round"] <= 3 and req["metric_output"] is None,
               "unsupported_attempt_or_output")
        root, run = _path(req["workspace"]), _path(req["run_dir"])
        attempt, scratch = _path(run / f"attempt_{req['repair_round']}"), _path(run / "runtime_scratch")
        trusted = _path(attempt / "observer_trusted")
        _check(root == run / "workspace" and not attempt.is_relative_to(scratch)
               and {child.name for child in trusted.iterdir()} == {"observer.py", "config.json"}, "unknown_audit_layout")
        paths = {"launch": attempt / "runtime_observer_launch.json", "report": attempt / "runtime_observer.json",
                 "config": trusted / "config.json", "observer": trusted / "observer.py", "stdout": attempt / "run_stdout.log"}
        raw = {key: _read(path, 32 * 1024 * 1024 if key in {"report", "stdout"} else 2 * 1024 * 1024)
               for key, path in paths.items()}
        audit, config, report = (_json(raw[key]) for key in ("launch", "config", "report"))
        _check(type(audit) is dict and set(audit) == _PREPARE | _REPORT_FACTS and audit["status"] == "observed"
               and audit["unresolved"] == [] and audit["report_status"] == "completed" and audit["report_unresolved"] == []
               and _same(audit, env.get("runtime_observer")), "unknown_or_changed_launch_audit")
        prepared = {key: audit[key] for key in _PREPARE}
        prepared["status"] = "ready"
        sealed = {key: value for key, value in prepared.items() if key not in {"unresolved", "request_sha256"}}
        _check(_sha(json.dumps(sealed, sort_keys=True, ensure_ascii=True).encode()) == audit["request_sha256"], "prepare_seal_mismatch")
        _check(audit["version"] == "readonly-python-observer-v1" and audit["workspace"] == str(root)
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
        _check(re.fullmatch(f"observer_attempt_{req['repair_round']}_[a-f0-9]{{32}}/events.json", relative_output), "unknown_producer_output_path")
        expected_config = {"entry": command[1], "args": command[2:], "cwd": cwd, "sites": runtime_sites,
                           "output": "/workspace/run_dir/" + relative_output}
        _check(_same(config, expected_config) and audit["container_cwd"] == cwd
               and audit["launch_command"] == [command[0], "/factreview-observer/observer.py", "/factreview-observer/config.json"], "actual_config_changed")
        _check(_sha(raw["observer"]) == _file_sha(Path(__file__).with_name("runtime_observer.py"))
               and _same(audit["package_files"], {"observer.py": _sha(raw["observer"]), "config.json": _sha(raw["config"])}), "observer_package_changed")
        bootstrap = _bootstrap_scope(root, _path(root / entry), _path(root / workdir), req["config"]["python_version"])
        bootstrap.update(cache_path="/workspace/run_dir/.pycache", cache_absent_at_prepare=True)
        _check(_same(bootstrap, audit["bootstrap_scope"]) and _same(audit["mounts"], {
            "/app": {"source": str(root), "mode": "ro"}, "/factreview-observer": {"source": str(trusted), "mode": "ro"},
            "/workspace/run_dir": {"source": str(scratch), "mode": "rw"}}), "protection_identity_changed")
        runs = [argv for argv in result["commands"] if type(argv) is list and argv[:2] == ["docker", "run"]]
        _check(len(runs) == 1, "actual_run_argv_not_unique")
        argv = runs[0]
        protected = protect_observer_docker_argv(argv, launch=prepared)
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
        _check(all(actual_env.get(key) == value for key, value in expected_env.items())
               and set(actual_env).issubset(set(expected_env) | {"HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"}), "declared_environment_changed_or_unknown")
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
        return {"status": "received", "report": report, "run_identity": {"command": command, "cwd": cwd, "runtime_root": "/app",
            "repair_round": req["repair_round"], "launch_sha256": _sha(raw["launch"]), "config_sha256": _sha(raw["config"]),
            "request_sha256": _fingerprint(req), "actual_argv_sha256": _fingerprint(argv), "source_sha256": hashes}, "evidence_refs": refs,
            "producer_code_identity": {"basis": "fixed_observer_actual_runtime_prepare_and_match", "sites": report["sites"], "interpreter": interpreter},
            "scientific_qualification": False, "alignment": False, "support": False,
            "limits": ["caller_request_and_outcome_origin_must_be_trusted", "in_process_author_tampering_not_excluded",
                       "internal_hashes_are_not_authentication", "proxy_history_not_reconstructed", "scientific_roles_and_stdout_metric_binding_unproved"]}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, OverflowError, RecursionError) as exc:
        return {"status": "unresolved", "reason": str(exc) if type(exc) is ValueError else type(exc).__name__,
                "scientific_qualification": False, "alignment": False, "support": False}
