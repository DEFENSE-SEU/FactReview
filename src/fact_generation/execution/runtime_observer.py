"""Bounded source-linked Python events; no alignment or scientific-role judgment.

The disposable CLI additionally observes Python audit events. In-process callers
install only a temporary profile hook, and cannot exclude unobserved native work.
Neither mode is tamper-proof against arbitrary author code.
"""

from __future__ import annotations

import _thread
import dis
import hashlib
import importlib.machinery
import json
import math
import os
import struct
import subprocess
import sys
import time
import types
from pathlib import Path


def _source_identity(path):
    """Complete source SHA with bounded buffers; reject changes during the read."""
    before = path.stat()
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as source:
        while chunk := source.read(1048576):
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if size != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError("source_changed_during_identity_read")
    return {"sha256": digest.hexdigest(), "size": size}


def _compile_source(path):
    """Compile complete bytes in this interpreter/path; a prefix is never code."""
    identity = _source_identity(path)
    with path.open("rb") as source:
        raw = source.read(65537)
    if len(raw) > 65536:
        raise ValueError("source_capacity")
    if len(raw) != identity["size"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError("source_changed_during_bounded_read")
    return compile(raw, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize), identity


def _code_sha(code):
    def constant(value):
        kind = type(value)
        if kind is types.CodeType:
            return {"code": fields(value)}
        if kind in (tuple, frozenset):
            items = [constant(item) for item in value]
            if kind is frozenset:
                items.sort(key=lambda item: json.dumps(item, sort_keys=True))
            return {"tuple" if kind is tuple else "frozenset": items}
        if kind is bytes:
            return {"bytes": value.hex()}
        if kind is float:
            return {"float": struct.pack("!d", value).hex()}
        if kind is complex:
            return {"complex": [struct.pack("!d", value.real).hex(), struct.pack("!d", value.imag).hex()]}
        if kind is int:
            return {"int": hex(value)}
        if kind in (str, bool, type(None)):
            return value
        if value is Ellipsis:
            return {"ellipsis": True}
        raise ValueError("Unsupported code constant")

    def fields(item):
        names = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize",
                 "co_flags", "co_code", "co_consts", "co_names", "co_varnames", "co_filename", "co_name",
                 "co_qualname", "co_firstlineno", "co_linetable", "co_exceptiontable", "co_freevars", "co_cellvars")
        return {name: constant(getattr(item, name)) for name in names}

    return hashlib.sha256(json.dumps(fields(code), sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def prepare_site(path, qualname, firstlineno):
    """Program-generated identity; compile in the eventual execution interpreter/path."""
    if type(path) is not str or type(qualname) is not str or type(firstlineno) is not int or firstlineno < 1:
        raise ValueError("Site selectors require exact string paths/names and an integer source line")
    path = Path(path).resolve(strict=True)
    code, identity = _compile_source(path)
    candidates, pending = [], [code]
    while pending:
        item = pending.pop()
        if item.co_qualname == qualname and item.co_firstlineno == firstlineno:
            candidates.append(item)
        pending.extend(value for value in item.co_consts if type(value) is types.CodeType)
    if len(candidates) != 1 or candidates[0].co_flags & (0x20 | 0x80 | 0x200):
        raise ValueError("Site must identify one non-generator synchronous code object")
    return {
        "path": str(path), "sha256": identity["sha256"],
        "qualname": qualname, "firstlineno": firstlineno, "code_sha256": _code_sha(candidates[0]),
    }


def _snapshot(value, capacity):
    """Exact builtins only; typed containers preserve tuple/list/dict distinctions."""
    remaining, active = [256], set()

    def visit(item, depth):
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 12:
            raise ValueError("snapshot_capacity_exceeded")
        kind = type(item)
        if kind in (type(None), bool, int, float, str):
            if kind is float and not math.isfinite(item):
                raise ValueError("nonfinite_snapshot_value")
            if (kind is str and len(item) > capacity) or (kind is int and item.bit_length() > 8192):
                raise ValueError("snapshot_capacity_exceeded")
            return item
        if kind not in (list, tuple, dict):
            raise ValueError("unsupported_snapshot_type")
        if id(item) in active or len(item) > remaining[0]:
            raise ValueError("snapshot_cycle_or_capacity")
        active.add(id(item))
        try:
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("unsupported_snapshot_key")
                if any(len(key) > capacity for key in item):
                    raise ValueError("snapshot_capacity_exceeded")
                values = [[key, visit(value, depth + 1)] for key, value in dict.items(item)]
            else:
                values = [visit(value, depth + 1) for value in item]
            return {"type": {list: "list", tuple: "tuple", dict: "dict"}[kind], "items": values}
        finally:
            active.remove(id(item))

    projected = visit(value, 0)
    if len(json.dumps(projected, ensure_ascii=True, allow_nan=False).encode()) > capacity:
        raise ValueError("snapshot_capacity_exceeded")
    return projected


def _write_report(destination, report):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write("\n")


def run_observed(entry, args, cwd, sites, output, *, max_events=256, max_snapshot_bytes=65536,
                 _cli_audit=False, _cli_reraise=False):
    """Execute a direct script once; output must be fresh and outside author cwd.

    Sites must be generated from frozen source, never accepted as model hashes.
    No event metadata asserts resource consumption, full coverage or sufficiency.
    """
    report = {
        "version": "python-source-events-v1", "sites": [], "events": [], "unresolved": [],
        "execution": {"status": "not_started"},
        "scope_limits": ["in_process_tampering_not_excluded", "events_do_not_prove_scientific_roles",
                         "native_and_child_execution_not_covered"],
        "intrusion": ["profile_and_launcher_overhead", "performance_metrics_not_qualified",
                      "isolated_main_module_and_temporary_process_globals"],
        "interpreter": {"version": sys.version, "executable": sys.executable, "optimize": sys.flags.optimize},
    }

    def unresolved(reason):
        if reason not in report["unresolved"]:
            report["unresolved"].append(reason)

    destination = None
    try:
        if (type(entry) is not str or type(cwd) is not str or type(output) is not str
                or type(args) is not list or any(type(arg) is not str for arg in args)
                or type(sites) is not list or not sites
                or type(max_events) is not int or not 0 < max_events <= 4096
                or type(max_snapshot_bytes) is not int or not 0 < max_snapshot_bytes <= 1048576):
            raise ValueError("Invalid bounded observer request")
        working = Path(cwd).resolve(strict=True)
        script = (working / entry).resolve(strict=True)
        candidate_output = Path(output).absolute()
        if candidate_output.exists() or candidate_output.resolve().is_relative_to(working):
            raise ValueError("Observer output must be fresh outside author cwd")
        destination = candidate_output
        identities = {}
        for index, site in enumerate(sites):
            if type(site) is not dict or set(site) != {"path", "sha256", "qualname", "firstlineno", "code_sha256"}:
                raise ValueError("Invalid site identity fields")
            if any(type(site[key]) is not str for key in ("path", "sha256", "qualname", "code_sha256")):
                raise ValueError("Site identity strings cannot invoke custom projection")
            actual = prepare_site(site["path"], site["qualname"], site["firstlineno"])
            key = (actual["path"], actual["qualname"], actual["firstlineno"])
            if site != actual or key in identities:
                raise ValueError("Site identity differs from current compiled source")
            identities[key] = (index, actual)
            report["sites"].append(actual)
        executable, entry_identity = _compile_source(script)
        entry_sha = entry_identity["sha256"]
        getprofile, setprofile = sys.getprofile, sys.setprofile
        if getprofile() is not None:
            raise ValueError("An existing profile hook cannot be replaced")
    except (ValueError, OSError, TypeError, SyntaxError) as exc:
        if type(exc) is ValueError and exc.args == ("source_capacity",):
            unresolved("source_capacity")
        unresolved("startup_identity_or_request_invalid")
        if destination is not None:
            _write_report(destination, report)
        return report

    report["entry"] = {"argv": [entry, *args], "cwd": str(working), "path": str(script), "sha256": entry_sha}
    active, seen, sequence, enabled = {}, set(), [0], [True]
    thread_ident = _thread.get_ident()
    thread_starts = [_thread.start_new_thread]
    if hasattr(_thread, "start_joinable_thread"):
        thread_starts.append(_thread.start_joinable_thread)
    process_functions = [getattr(os, name) for name in ("fork", "forkpty", "posix_spawn", "posix_spawnp", "system") if hasattr(os, name)]
    process_code = subprocess.Popen.__init__.__code__ if type(subprocess.Popen) is type else None
    if len(sys._current_frames()) != 1:
        unresolved("threads_present_at_launch")

    def emit(row):
        if len(report["events"]) >= max_events:
            unresolved("event_capacity_exceeded")
            return
        row["order"] = len(report["events"]) + 1
        report["events"].append(row)

    def projected(value, capacity=max_snapshot_bytes):
        try:
            return _snapshot(value, capacity)
        except ValueError as exc:
            unresolved(exc.args[0])
            return {"unresolved": exc.args[0]}

    def profile(frame, event, arg):
        if not enabled[0]:
            return
        if _thread.get_ident() != thread_ident:
            unresolved("foreign_thread_event")
        if event == "c_call":
            if arg is setprofile:
                unresolved("profile_changed")
            if any(arg is function for function in thread_starts):
                unresolved("thread_start_observed")
            if any(arg is function for function in process_functions):
                unresolved("child_process_observed")
            return
        code = frame.f_code
        if event == "call" and code is process_code:
            unresolved("child_process_observed")
        key = (code.co_filename, code.co_qualname, code.co_firstlineno)
        if key not in identities:
            return
        index, site = identities[key]
        try:
            matching = _code_sha(code) == site["code_sha256"]
        except ValueError:
            matching = False
        if not matching:
            unresolved("runtime_code_identity_mismatch")
            return
        if event == "call":
            seen.add(index)
            sequence[0] += 1
            parent = frame.f_back
            while parent is not None and id(parent) not in active:
                parent = parent.f_back
            invocation = sequence[0]
            active[id(frame)] = invocation
            count = code.co_argcount + code.co_kwonlyargcount + bool(code.co_flags & 4) + bool(code.co_flags & 8)
            arguments, remaining = {}, max_snapshot_bytes
            for name in code.co_varnames[:count]:
                if name in frame.f_locals:
                    arguments[name] = projected(frame.f_locals[name], max(0, remaining))
                    remaining -= len(json.dumps(arguments[name], ensure_ascii=True).encode()) + len(name) * 6 + 8
                    if remaining <= 0:
                        unresolved("argument_snapshot_capacity_exceeded")
                        break
            emit({"event": "call", "site": index, "invocation": invocation,
                  "parent_invocation": active.get(id(parent)), "arguments": arguments})
        elif event == "return" and id(frame) in active:
            instruction = dis.opname[code.co_code[frame.f_lasti]] if frame.f_lasti >= 0 else ""
            returned = instruction in {"RETURN_VALUE", "RETURN_CONST"}
            emit({"event": "return" if returned else "unwind", "site": index,
                  "invocation": active.pop(id(frame)), **({"value": projected(arg)} if returned else {})})

    def audit(event, _args):
        if enabled[0] and (event.startswith("subprocess.") or event in {"os.fork", "os.forkpty", "os.posix_spawn", "os.system"}):
            unresolved("child_process_observed")

    if _cli_audit:
        # Only the disposable CLI opts in: Python audit hooks cannot be removed.
        sys.addaudithook(audit)
    old = os.getcwd(), sys.argv, sys.path, sys.modules.get("__main__")
    main = types.ModuleType("__main__")
    main.__dict__.update(__file__=str(script), __package__=None, __spec__=None, __cached__=None,
                         __loader__=importlib.machinery.SourceFileLoader("__main__", str(script)))
    started, author_exception = time.monotonic(), None
    try:
        os.chdir(working)
        sys.argv, sys.path, sys.modules["__main__"] = [entry, *args], [str(script.parent), *old[2][1:]], main
        setprofile(profile)
        report["execution"]["status"] = "running"
        try:
            exec(executable, main.__dict__)
            report["execution"]["status"] = "completed"
        except BaseException as exc:
            author_exception = exc
            report["execution"]["status"] = "raised"
            if type(exc) is SystemExit:
                report["execution"] = {"status": "system_exit", "code": projected(exc.code)}
    finally:
        if getprofile() is not profile:
            unresolved("profile_changed")
        enabled[0] = False
        setprofile(None)
        os.chdir(old[0])
        sys.argv, sys.path = old[1], old[2]
        if old[3] is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = old[3]
        report["execution"]["runtime_seconds"] = time.monotonic() - started
    if active:
        unresolved("missing_return_event")
    if seen != set(range(len(sites))):
        unresolved("site_not_observed")
    expected = {str(script): entry_sha, **{site["path"]: site["sha256"] for site in sites}}
    report["source_hashes_after"] = {}
    for path, sha in expected.items():
        try:
            actual = _source_identity(Path(path))["sha256"]
        except (OSError, ValueError):
            actual = None
        report["source_hashes_after"][path] = actual
        if actual != sha:
            unresolved("source_changed_or_unavailable")
    _write_report(destination, report)
    if _cli_reraise and author_exception is not None:
        # The disposable CLI preserves the original exception/exit code after
        # restoring process state and saving observations. API callers get JSON.
        raise author_exception
    return report


def main():
    """Trusted read-only JSON config; invoke this module in a disposable interpreter."""
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # Only this execution interpreter may produce code identities. Host/model
    # code hashes cannot authorize a different Python version or deployed path.
    prepared = []
    for site in request["sites"]:
        if set(site) != {"path", "sha256", "qualname", "firstlineno"}:
            raise ValueError("CLI requires source selectors without a supplied code hash")
        actual = prepare_site(site["path"], site["qualname"], site["firstlineno"])
        if actual["sha256"] != site["sha256"]:
            raise ValueError("CLI frozen source hash changed")
        prepared.append(actual)
    request["sites"] = prepared
    report = run_observed(**request, _cli_audit=True, _cli_reraise=True)
    if report["execution"]["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
