"""Finite readonly observer launch preparation; no scientific-consumption grant."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import uuid
from pathlib import Path, PurePosixPath

from .repair_evidence import SOURCE_BYTES, bounded_source, plain_path
from .runtime_observer import prepare_site

# Official CPython v3.11.0 Python/stdlib_module_names.h (305 names).
# https://github.com/python/cpython/blob/v3.11.0/Python/stdlib_module_names.h
# Raw source SHA256 f58f81abfc73c526744bbbde8fd110c6f97daa9b66737516b6b6ab1686bc6feb
_CPYTHON_311_STDLIB = frozenset({
    '__future__', '_abc', '_aix_support', '_ast', '_asyncio', '_bisect',
    '_blake2', '_bootsubprocess', '_bz2', '_codecs', '_codecs_cn', '_codecs_hk',
    '_codecs_iso2022', '_codecs_jp', '_codecs_kr', '_codecs_tw', '_collections', '_collections_abc',
    '_compat_pickle', '_compression', '_contextvars', '_crypt', '_csv', '_ctypes',
    '_curses', '_curses_panel', '_datetime', '_dbm', '_decimal', '_elementtree',
    '_frozen_importlib', '_frozen_importlib_external', '_functools', '_gdbm', '_hashlib', '_heapq',
    '_imp', '_io', '_json', '_locale', '_lsprof', '_lzma',
    '_markupbase', '_md5', '_msi', '_multibytecodec', '_multiprocessing', '_opcode',
    '_operator', '_osx_support', '_overlapped', '_pickle', '_posixshmem', '_posixsubprocess',
    '_py_abc', '_pydecimal', '_pyio', '_queue', '_random', '_scproxy',
    '_sha1', '_sha256', '_sha3', '_sha512', '_signal', '_sitebuiltins',
    '_socket', '_sqlite3', '_sre', '_ssl', '_stat', '_statistics',
    '_string', '_strptime', '_struct', '_symtable', '_thread', '_threading_local',
    '_tkinter', '_tokenize', '_tracemalloc', '_typing', '_uuid', '_warnings',
    '_weakref', '_weakrefset', '_winapi', '_zoneinfo', 'abc', 'aifc',
    'antigravity', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
    'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii',
    'bisect', 'builtins', 'bz2', 'cProfile', 'calendar', 'cgi',
    'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs',
    'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser',
    'contextlib', 'contextvars', 'copy', 'copyreg', 'crypt', 'csv',
    'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal',
    'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings',
    'ensurepip', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp',
    'fileinput', 'fnmatch', 'fractions', 'ftplib', 'functools', 'gc',
    'genericpath', 'getopt', 'getpass', 'gettext', 'glob', 'graphlib',
    'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html',
    'http', 'idlelib', 'imaplib', 'imghdr', 'imp', 'importlib',
    'inspect', 'io', 'ipaddress', 'itertools', 'json', 'keyword',
    'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox',
    'mailcap', 'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder',
    'msilib', 'msvcrt', 'multiprocessing', 'netrc', 'nis', 'nntplib',
    'nt', 'ntpath', 'nturl2path', 'numbers', 'opcode', 'operator',
    'optparse', 'os', 'ossaudiodev', 'pathlib', 'pdb', 'pickle',
    'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib',
    'posix', 'posixpath', 'pprint', 'profile', 'pstats', 'pty',
    'pwd', 'py_compile', 'pyclbr', 'pydoc', 'pydoc_data', 'pyexpat',
    'queue', 'quopri', 'random', 're', 'readline', 'reprlib',
    'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
    'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site',
    'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
    'sqlite3', 'sre_compile', 'sre_constants', 'sre_parse', 'ssl', 'stat',
    'statistics', 'string', 'stringprep', 'struct', 'subprocess', 'sunau',
    'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile',
    'telnetlib', 'tempfile', 'termios', 'textwrap', 'this', 'threading',
    'time', 'timeit', 'tkinter', 'token', 'tokenize', 'tomllib',
    'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo',
    'types', 'typing', 'unicodedata', 'unittest', 'urllib', 'uu',
    'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
    'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc',
    'zipapp', 'zipfile', 'zipimport', 'zlib', 'zoneinfo',
})

def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _path(value, *, exists=True):
    path = Path(value).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
            raise ValueError("linked_path")
    return path.resolve(strict=exists)


def _relative(value, *, dot=False):
    if type(value) is not str or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("invalid_relative_path")
    if dot and value == ".":
        return value
    if PurePosixPath(value).is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("invalid_relative_path")
    return value


def _site(site, root, *, hashed):
    fields = {"path", "qualname", "firstlineno"} | ({"sha256"} if hashed else set())
    if type(site) is not dict or set(site) != fields:
        raise ValueError("site_fields_not_closed")
    relative = _relative(site["path"])
    if Path(relative).suffix.lower() != ".py":
        raise ValueError("site_not_python_source")
    path = _path(root / relative)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("site_source_unavailable")
    actual = prepare_site(str(path), site["qualname"], site["firstlineno"])
    leaf = {key: actual[key] for key in ("sha256", "qualname", "firstlineno")}
    leaf["path"] = relative
    if hashed and site != leaf:
        raise ValueError("source_site_identity_changed")
    return leaf, path


def bind_source_sites(proposals, *, workspace, supplied_files):
    """Bind only exact proposals from the current refinement's actual file text."""
    result = {"version": "python-source-sites-v1", "status": "unresolved", "sites": [], "unresolved": []}
    try:
        root = _path(plain_path(Path(workspace)))
        if not root.is_dir() or type(supplied_files) is not dict:
            raise ValueError("supplied_source_unavailable")
        if type(proposals) is not list or not 1 <= len(proposals) <= 8:
            raise ValueError("source_sites_missing_or_over_capacity")
        seen = set()
        for index, proposal in enumerate(proposals):
            try:
                if type(proposal) is not dict or set(proposal) != {"path", "qualname", "firstlineno"}:
                    raise ValueError("site_fields_not_closed")
                relative = _relative(proposal["path"])
                supplied = supplied_files.get(relative)
                if type(supplied) is not str or len(supplied) > SOURCE_BYTES:
                    raise ValueError("source_site_not_supplied_or_over_capacity")
                candidate = _path(plain_path(root / relative))
                if not candidate.is_relative_to(root):
                    raise ValueError("site_source_unavailable")
                raw, scope = bounded_source(candidate)
                if raw is None or not scope["complete"]:
                    raise ValueError("source_site_over_capacity")
                if raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n") != supplied:
                    raise ValueError("supplied_source_changed")
                leaf, _path_unused = _site(proposal, root, hashed=False)
                if leaf["sha256"] != scope["sha256"]:
                    raise ValueError("source_site_identity_changed_after_complete_read")
                key = (leaf["path"], leaf["qualname"], leaf["firstlineno"])
                if key in seen:
                    raise ValueError("duplicate_source_site")
                seen.add(key)
                result["sites"].append(leaf)
            except (ValueError, OSError, TypeError, SyntaxError) as exc:
                result["unresolved"].append({"proposal_index": index, "reason":
                    exc.args[0] if type(exc) is ValueError else "source_site_unavailable"})
        if not result["unresolved"]:
            result["status"] = "bound"
    except (ValueError, OSError, TypeError) as exc:
        result["unresolved"].append({"reason": exc.args[0] if type(exc) is ValueError else "source_site_unavailable"})
    return result


def _bootstrap_scope(root, entry, working, runtime_python):
    host_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    declared = runtime_python if runtime_python is not None else host_minor
    if type(declared) is not str or not re.fullmatch(r"3\.\d+(?:\.\d+)?", declared):
        raise ValueError("bootstrap_runtime_minor_unknown")
    minor = ".".join(declared.split(".")[:2])
    if minor not in {host_minor, "3.11"}:
        raise ValueError("bootstrap_runtime_minor_unsupported")
    names = _CPYTHON_311_STDLIB | sys.stdlib_module_names | {"sitecustomize", "usercustomize"}
    folded_names = {name.casefold() for name in names}
    directories = sorted({root, entry.parent, working})
    for directory in directories:
        for child in directory.iterdir():
            actual = _path(child)
            if actual.is_dir():
                name = child.name
            elif child.suffix.lower() in {".py", ".pyc", ".pyo", ".so", ".pyd"}:
                name = child.name.split(".", 1)[0]
            else:
                continue
            if name.casefold() in folded_names:
                raise ValueError("author_stdlib_or_startup_shadow")
    return {"declared_runtime_minor": minor, "host_namespace_minor": host_minor,
            "catalog_source": "CPython-v3.11.0+host-sys.stdlib_module_names",
            "namespace_sha256": _sha(json.dumps(sorted(names)).encode()),
            "casefold_filesystem_guard": True,
            "container_search_dirs": ["/app" if directory == root else "/app/" + directory.relative_to(root).as_posix()
                                      for directory in directories],
            "limits": ["runtime_minor_declaration_is_not_interpreter_identity",
                       "image_standard_library_site_packages_and_hooks_must_be_trusted",
                       "dynamic_author_import_changes_are_not_excluded"]}


def prepare_observer_launch(original_command, *, entry_script, workdir, workspace, metric_output,
                            source_sites, trusted_dir, runtime_dir, repair_round, runtime_python=None):
    """Prepare a fresh stdout-only direct-script launch without changing author argv."""
    result = {"version": "readonly-python-observer-v1", "status": "unresolved", "unresolved": [],
              "scope_limits": ["source_linked_events_only", "scientific_consumption_not_established",
                               "in_process_tampering_not_excluded", "native_threads_children_uncovered",
                               "instrumentation_timing_not_qualified"]}
    try:
        if (type(original_command) is not list or len(original_command) < 2
                or any(type(item) is not str or not item or "\x00" in item for item in original_command)):
            raise ValueError("invalid_original_command")
        result["original_command"] = original_command.copy()
        if original_command[0] not in {"python", "python3"} or original_command[1].startswith("-"):
            raise ValueError("observer_direct_python_only")
        if metric_output is not None:
            raise ValueError("observer_stdout_only_protection_layout_unresolved")
        root, scratch, trusted = _path(workspace), _path(runtime_dir), _path(trusted_dir, exists=False)
        if not root.is_dir() or not scratch.is_dir() or trusted.exists() or not trusted.parent.is_dir():
            raise ValueError("observer_paths_not_fresh_or_available")
        for left, right in ((root, scratch), (root, trusted), (scratch, trusted)):
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("observer_paths_overlap")
        relative_entry, relative_workdir = _relative(entry_script), _relative(workdir, dot=True)
        entry, working = _path(root / relative_entry), _path(root / relative_workdir)
        script_token = original_command[1]
        if ("\\" in script_token or ":" in script_token or PurePosixPath(script_token).is_absolute()
                or ".." in script_token.split("/") or not working.is_dir()
                or _path(working / script_token) != entry or not entry.is_file() or entry.suffix.lower() != ".py"):
            raise ValueError("observer_entry_not_original_python_script")
        if ".factreview" in Path(relative_entry).parts:
            raise ValueError("observer_forwarding_wrapper_unsupported")
        cache = scratch / ".pycache"
        if cache.exists() or cache.is_symlink() or getattr(cache, "is_junction", lambda: False)():
            raise ValueError("bootstrap_cache_not_fresh")
        bootstrap = _bootstrap_scope(root, entry, working, runtime_python)
        bootstrap.update(cache_path="/workspace/run_dir/.pycache", cache_absent_at_prepare=True)
        if type(repair_round) is not int or repair_round < 0:
            raise ValueError("invalid_repair_round")
        if (type(source_sites) is not dict or source_sites.get("version") != "python-source-sites-v1"
                or source_sites.get("status") != "bound" or source_sites.get("unresolved") != []
                or type(source_sites.get("sites")) is not list or not 1 <= len(source_sites["sites"]) <= 8):
            raise ValueError("source_sites_unresolved")
        leaves, seen = [], set()
        for site in source_sites["sites"]:
            leaf, _ = _site(site, root, hashed=True)
            key = (leaf["path"], leaf["qualname"], leaf["firstlineno"])
            if key in seen:
                raise ValueError("duplicate_source_site")
            seen.add(key)
            leaves.append(leaf)
        container_cwd = "/app" + ("/" + relative_workdir if relative_workdir != "." else "")
        attempt = f"observer_attempt_{repair_round}_{uuid.uuid4().hex}"
        audit_dir = _path(scratch / attempt, exists=False)
        if audit_dir.exists():
            raise ValueError("observer_audit_path_not_fresh")
        config = {"entry": script_token, "args": original_command[2:], "cwd": container_cwd,
                  "sites": [dict(site, path="/app/" + site["path"]) for site in leaves],
                  "output": f"/workspace/run_dir/{attempt}/events.json"}
        observer_bytes = _path(Path(__file__).with_name("runtime_observer.py")).read_bytes()
        config_bytes = (json.dumps(config, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()
        source_hashes = {relative_entry: _sha(entry.read_bytes()), **{site["path"]: site["sha256"] for site in leaves}}
        # All checks precede any package creation; failed creation is retained for audit.
        trusted.mkdir(exist_ok=False)
        audit_dir.mkdir(exist_ok=False)
        for name, content in (("observer.py", observer_bytes), ("config.json", config_bytes)):
            with (trusted / name).open("xb") as stream:
                stream.write(content)
        result.update(status="ready", workspace=str(root), trusted_dir=str(trusted), runtime_dir=str(scratch),
            original_workdir=relative_workdir, container_cwd=container_cwd, source_sites=leaves, bootstrap_scope=bootstrap,
            source_sha256=source_hashes, audit_output=str(audit_dir / "events.json"),
            package_files={"observer.py": _sha(observer_bytes), "config.json": _sha(config_bytes)},
            launch_command=[original_command[0], "/factreview-observer/observer.py", "/factreview-observer/config.json"],
            mounts={"/app": {"source": str(root), "mode": "ro"},
                    "/factreview-observer": {"source": str(trusted), "mode": "ro"},
                    "/workspace/run_dir": {"source": str(scratch), "mode": "rw"}})
        sealed = {key: value for key, value in result.items() if key != "unresolved"}
        result["request_sha256"] = _sha(json.dumps(sealed, sort_keys=True, ensure_ascii=True).encode())
    except (ValueError, OSError, TypeError, SyntaxError) as exc:
        result["unresolved"].append({"reason": exc.args[0] if type(exc) is ValueError else "observer_preparation_unavailable"})
    return result


def _mount(value, option):
    if option in {"-v", "--volume"}:
        match = re.fullmatch(r"(.+):(/[^:]*)(?::(ro|rw))?", value)
        if not match:
            raise ValueError("unknown_volume_layout")
        source, target, mode = match.groups()
        return source, target, mode or "rw"
    fields = {}
    aliases = {"source": "source", "src": "source", "target": "target", "dst": "target", "destination": "target"}
    for piece in value.split(","):
        key, separator, item = piece.partition("=")
        key = aliases.get(key, key)
        if key in fields or key not in {"type", "source", "target", "readonly"}:
            raise ValueError("unknown_mount_layout")
        if key == "readonly":
            if separator and item not in {"true", "1"}:
                raise ValueError("unknown_mount_layout")
            fields[key] = True
        else:
            if not separator or not item:
                raise ValueError("unknown_mount_layout")
            fields[key] = item
    if set(fields) not in ({"type", "source", "target"}, {"type", "source", "target", "readonly"}) or fields["type"] != "bind":
        raise ValueError("unknown_mount_layout")
    return fields["source"], fields["target"], "ro" if fields.get("readonly") else "rw"


def protect_observer_docker_argv(argv, *, launch):
    """Pure transformation of the existing finite Docker helper's command layout."""
    result = {"status": "unresolved", "argv": argv.copy() if type(argv) is list else argv, "unresolved": []}
    try:
        if (type(argv) is not list or any(type(token) is not str for token in argv)
                or argv[:2] != ["docker", "run"] or type(launch) is not dict or launch.get("status") != "ready"):
            raise ValueError("unknown_docker_launch_layout")
        original, wrapper = launch["original_command"], launch["launch_command"]
        tails = [tail for tail in (original, wrapper) if tail and argv[-len(tail):] == tail]
        if len(tails) != 1 or len(argv) < len(tails[0]) + 3:
            raise ValueError("docker_command_tail_changed")
        image_index = len(argv) - len(tails[0]) - 1
        image = argv[image_index]
        if not image or image.startswith("-"):
            raise ValueError("unknown_docker_image_position")
        options, rewritten, mounts, cwd, index = argv[2:image_index], [], {}, [], 0
        pythonpath_present = False
        pairs = {"--name", "--user", "--gpus", "--shm-size", "--ipc", "-w", "--workdir", "-e", "--env"}
        while index < len(options):
            option = options[index]
            if option in {"--rm", "--read-only"}:
                rewritten.append(option)
                index += 1
                continue
            if option not in pairs | {"-v", "--volume", "--mount"} or index + 1 >= len(options):
                raise ValueError("unknown_docker_option_layout")
            value = options[index + 1]
            if option in {"-v", "--volume", "--mount"}:
                source, target, mode = _mount(value, option)
                expected = launch["mounts"].get(target)
                if target in mounts or expected is None or source != expected["source"]:
                    raise ValueError("duplicate_overlapping_or_unknown_mount")
                if target == "/workspace/run_dir" and mode != "rw":
                    raise ValueError("observer_scratch_not_writable")
                if target == "/factreview-observer" and mode != "ro":
                    raise ValueError("observer_package_not_readonly")
                mounts[target] = source
                if target == "/app":
                    value = f"{source}:/app:ro" if option != "--mount" else f"type=bind,source={source},target=/app,readonly"
            if option in {"-w", "--workdir"}:
                cwd.append(value)
            if option in {"-e", "--env"}:
                key, separator, content = value.partition("=")
                if not separator or key in {"PATH", "__PYVENV_LAUNCHER__"} or key.startswith(("LD_", "DYLD_")):
                    raise ValueError("bootstrap_import_environment_unknown_or_unsafe")
                if key.startswith("PYTHON"):
                    if key == "PYTHONPATH":
                        if content != "/app":
                            raise ValueError("bootstrap_pythonpath_not_closed")
                        pythonpath_present = True
                    elif key == "PYTHONPYCACHEPREFIX":
                        if content != "/workspace/run_dir/.pycache":
                            raise ValueError("bootstrap_cache_path_unknown")
                    elif key not in {"PYTHONUNBUFFERED", "PYTHONIOENCODING", "PYTHONUTF8"}:
                        raise ValueError("bootstrap_import_environment_unknown_or_unsafe")
            rewritten.extend([option, value])
            index += 2
        expected_cwd = launch["container_cwd"]
        valid_cwd = {expected_cwd} | ({"/app/.", "/app/"} if expected_cwd == "/app" else set())
        if not pythonpath_present:
            raise ValueError("bootstrap_pythonpath_not_declared")
        if set(mounts) not in ({"/app", "/workspace/run_dir"}, {"/app", "/workspace/run_dir", "/factreview-observer"}) or len(cwd) != 1 or cwd[0] not in valid_cwd:
            raise ValueError("incomplete_protection_layout")
        if "/factreview-observer" not in mounts:
            rewritten.extend(["-v", launch["trusted_dir"] + ":/factreview-observer:ro"])
        result.update(status="ready", argv=["docker", "run", *rewritten, image, *wrapper])
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        result["unresolved"].append({"reason": exc.args[0] if type(exc) is ValueError else "unknown_docker_launch_layout"})
    return result
