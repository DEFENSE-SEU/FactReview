"""Finite compiled callable selectors from a complete pre-completion snapshot.

Compilation never executes the author module or decorators. Eligible positions
are the intersection of original prepare_site and the inference consumer's
direct-body synchronous FunctionDef visitor; no scientific role is assigned.
"""
from __future__ import annotations

import ast
import json
import types
from collections import defaultdict
from pathlib import Path

from .inference_proposal_wire import _digest, freeze_sources
from .repair_evidence import plain_path
from .runtime_observer import _compile_source, prepare_site

VERSION = "inference-callable-catalog-v1"
MAX_LOCATIONS = 128
MAX_CATALOG_BYTES = 32768


def _static_functions(tree):
    """Match the original inference consumer's name/line traversal exactly."""
    found = defaultdict(list)

    def visit(node, prefix):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                first = min([child.lineno, *[row.lineno for row in child.decorator_list]])
                if isinstance(child, ast.FunctionDef):
                    found[(name, first)].append(child)
                visit(child, name + (".<locals>." if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else "."))

    visit(tree, "")
    return found


def build_callable_catalog(*, snapshot, plan, workspace, supplied_files, paper_sources, materials):
    """Publish a complete bounded catalog, or an empty fixed-reason observation.

    The input snapshot is already frozen before completion. Full re-freezes
    validate the original index, all sources and paper at both endpoints; the
    snapshot is never rebuilt from a prefix or granted alignment/support.
    """
    result = {"version": VERSION, "status": "unavailable", "locations": [], "reason": None}
    reason = "precall_snapshot_unavailable"
    try:
        if type(snapshot) is not dict or snapshot.get("status") != "frozen":
            raise ValueError
        reason = "snapshot_invalid"
        contents = snapshot["snapshot"]
        if _digest(contents) != snapshot["snapshot_digest"]:
            raise ValueError
        reason = "source_revalidation_failed"
        current = freeze_sources(plan=plan, workspace=workspace, supplied_files=supplied_files,
                                 paper_sources=paper_sources, materials=materials)
        if current.get("status") != "frozen" or current["snapshot"] != contents:
            raise ValueError
        root = plain_path(Path(workspace)).resolve(strict=True)
        locations = []
        for name, source in contents["sources"].items():
            if Path(name).suffix != ".py":
                continue
            path = plain_path(root / name).resolve(strict=True)
            reason = "source_compile_failed"
            code, identity = _compile_source(path)
            if (type(code) is not types.CodeType or identity["sha256"] != source["sha256"]
                    or identity["size"] != source["size"]):
                raise ValueError
            functions = _static_functions(ast.parse(source["displayed_text"]))
            compiled, pending = defaultdict(list), [code]
            while pending:
                item = pending.pop()
                compiled[(item.co_qualname, item.co_firstlineno)].append(item)
                pending.extend(value for value in item.co_consts if type(value) is types.CodeType)
            for (qualname, firstlineno), nodes in functions.items():
                reason = "callable_binding_unavailable"
                candidates = compiled[(qualname, firstlineno)]
                if not candidates:
                    raise ValueError
                reason = "callable_ambiguity"
                if len(nodes) != 1 or len(candidates) != 1:
                    raise ValueError
                if candidates[0].co_flags & (0x20 | 0x80 | 0x200):
                    continue
                reason = "site_binding_failed"
                bound = prepare_site(str(path), qualname, firstlineno)
                if (bound["path"] != str(path) or bound["sha256"] != source["sha256"]
                        or bound["qualname"] != qualname or bound["firstlineno"] != firstlineno):
                    raise ValueError
                locations.append({"path": name, "qualname": qualname, "firstlineno": firstlineno})
        locations.sort(key=lambda row: (row["path"], row["qualname"], row["firstlineno"]))
        reason = "catalog_count_capacity"
        if len(locations) > MAX_LOCATIONS:
            raise ValueError
        available = {"version": VERSION, "status": "available", "locations": locations, "reason": None}
        reason = "catalog_byte_capacity"
        # Match the original completion payload's JSON serializer, including escapes/spaces.
        if len(json.dumps(available).encode("utf-8")) > MAX_CATALOG_BYTES:
            raise ValueError
        reason = "source_endpoint_changed"
        after = freeze_sources(plan=plan, workspace=workspace, supplied_files=supplied_files,
                               paper_sources=paper_sources, materials=materials)
        if after.get("status") != "frozen" or after["snapshot"] != contents:
            raise ValueError
        return available
    except (OSError, ValueError, TypeError, KeyError, AttributeError, SyntaxError, RecursionError):
        result["reason"] = reason
        return result
