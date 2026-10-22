"""Opt-in fixed source-site gate observations outside author/material mounts."""
from __future__ import annotations

import json
from pathlib import Path

from .repair_evidence import plain_path

_FIELDS = ("is_dict", "status_bound", "version_current", "unresolved_empty")
_CODES = {
    "source_sites_not_dict": (False, None, None, None),
    "source_sites_status_unbound": (True, False, None, None),
    "source_sites_version_unbound": (True, True, False, None),
    "source_sites_unresolved": (True, True, True, False),
}


class SourceSitesDiagnostic:
    """Observe the original four predicates, with one exclusive write attempt."""

    def __init__(self, request, materials):
        self.request, self.materials = request, materials
        self.attempted = False

    def require_bound(self, source_sites):
        checks = dict.fromkeys(_FIELDS)
        # Retain a short-circuited comparison object until the original final require truth test.
        value = type(source_sites) is dict
        field, code = "is_dict", "source_sites_not_dict"
        if value:
            checks[field] = True
            value = source_sites.get("status") == "bound"
            field, code = "status_bound", "source_sites_status_unbound"
            if value:
                checks[field] = True
                value = source_sites.get("version") == "python-source-sites-v1"
                field, code = "version_current", "source_sites_version_unbound"
                if value:
                    checks[field] = True
                    value = source_sites.get("unresolved") == []
                    field, code = "unresolved_empty", "source_sites_unresolved"
        if not value:
            checks[field] = False
            self._record(code, checks)
            raise ValueError("inference_source_sites_unbound")

    def _record(self, code, checks):
        if self.attempted:
            return
        self.attempted = True
        try:
            if (type(code) is not str or code not in _CODES or type(checks) is not dict
                    or set(checks) != set(_FIELDS)
                    or any(value is not None and type(value) is not bool for value in checks.values())
                    or tuple(checks[field] for field in _FIELDS) != _CODES[code]):
                return
            request, materials = self.request, self.materials
            if type(request.repair_round) is not int or not 0 <= request.repair_round <= 3:
                return
            run_dir = plain_path(Path(request.run_dir)).resolve(strict=True)
            if not run_dir.is_dir():
                return
            parent = plain_path(run_dir / f"attempt_{request.repair_round}").resolve(strict=True)
            if not parent.is_dir() or not parent.is_relative_to(run_dir):
                return
            path = plain_path(parent / "inference_source_sites_diagnostics.json")
            source_paths = [materials.source_pdf, materials.markdown_path, materials.content_list_path]
            source_paths.extend(page.path for page in materials.pages)
            source_paths.extend(value for item in [*materials.figures, *materials.tables]
                                for value in (item.crop_path, item.printed_crop_path))
            excluded = [plain_path(Path(value)).resolve().parent for value in source_paths if value]
            excluded.append(plain_path(Path(request.workspace)).resolve(strict=True))
            if materials.repository is not None:
                excluded.append(plain_path(Path(materials.repository.root)).resolve(strict=True))
            if any(path.is_relative_to(root) for root in excluded):
                return
            payload = json.dumps({
                "schema": "torch-inference-source-sites-diagnostics-v1",
                "scope": "first_source_sites_binding_rejection", "code": code, "checks": checks,
            }, ensure_ascii=True, allow_nan=False) + "\n"
            if len(payload.encode("utf-8")) > 512 or plain_path(parent).resolve(strict=True) != parent:
                return
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except (OSError, ValueError, TypeError, AttributeError):
            pass


class RuntimeStageDiagnostic:
    """Canonical attempt-local program markers; never a scientific payload."""

    __slots__ = ("attempted", "materials", "phase", "pre_run_bound", "repair_round", "request", "run_dir",
                 "source_sites_passed", "source_sites_reached", "stage", "terminal", "workspace")
    _PHASES = ("pre_run", "initial_context", "refreshed_context", "semantic", "usage")
    _STAGES = ("entry", "schema_resources", "source_closure", "source_sites", "paper_config_partition",
               "receipt", "flow", "alignment", "context", "semantic_reply", "usage")
    _TERMINALS = ("entered", "passed", "rejected", "unresolved", "qualified", "failed", "skipped")

    def __init__(self, request, materials):
        self.request, self.materials = request, materials
        self.run_dir, self.repair_round, self.workspace = request.run_dir, request.repair_round, request.workspace
        self.attempted = False
        self.phase, self.stage, self.terminal = "pre_run", "entry", "entered"
        self.pre_run_bound = self.source_sites_reached = self.source_sites_passed = None

    @classmethod
    def mark(cls, value, *, phase=None, stage=None, terminal="entered", pre_run_bound=None):
        if type(value) is not RuntimeStageDiagnostic or value.attempted:
            return
        if ((phase is not None and (type(phase) is not str or phase not in RuntimeStageDiagnostic._PHASES))
                or (stage is not None and (type(stage) is not str or stage not in RuntimeStageDiagnostic._STAGES))
                or type(terminal) is not str or terminal not in RuntimeStageDiagnostic._TERMINALS
                or (pre_run_bound is not None and type(pre_run_bound) is not bool)):
            return
        if phase is not None:
            value.phase = phase
        if stage is not None:
            value.stage = stage
            if stage == "schema_resources":
                # These two fields describe the latest actual binding pass.
                value.source_sites_reached = value.source_sites_passed = None
            elif stage == "source_sites":
                value.source_sites_reached = True
                value.source_sites_passed = True if terminal == "passed" else None
        value.terminal = terminal
        if pre_run_bound is not None:
            value.pre_run_bound = pre_run_bound

    @classmethod
    def reject(cls, value, *, terminal="rejected", pre_run_bound=None):
        if type(value) is not RuntimeStageDiagnostic or value.attempted:
            return
        RuntimeStageDiagnostic.mark(value, terminal=terminal, pre_run_bound=pre_run_bound)
        if value.stage == "source_sites" and value.source_sites_reached is True:
            value.source_sites_passed = False

    @classmethod
    def close(cls, value):
        if type(value) is not RuntimeStageDiagnostic or value.attempted:
            return
        value.attempted = True
        try:
            if (type(value.phase) is not str or value.phase not in RuntimeStageDiagnostic._PHASES
                    or type(value.stage) is not str or value.stage not in RuntimeStageDiagnostic._STAGES
                    or type(value.terminal) is not str or value.terminal not in RuntimeStageDiagnostic._TERMINALS
                    or any(field is not None and type(field) is not bool for field in
                           (value.pre_run_bound, value.source_sites_reached, value.source_sites_passed))
                    or type(value.repair_round) is not int or not 0 <= value.repair_round <= 3):
                return
            run_dir = plain_path(Path(value.run_dir)).resolve(strict=True)
            parent = plain_path(run_dir / f"attempt_{value.repair_round}").resolve(strict=True)
            if not run_dir.is_dir() or not parent.is_dir() or not parent.is_relative_to(run_dir):
                return
            path = plain_path(parent / "inference_runtime_stage_diagnostics.json")
            materials = value.materials
            source_paths = [materials.source_pdf, materials.markdown_path, materials.content_list_path]
            source_paths.extend(page.path for page in materials.pages)
            source_paths.extend(item_path for item in [*materials.figures, *materials.tables]
                                for item_path in (item.crop_path, item.printed_crop_path))
            excluded = [plain_path(Path(item)).resolve().parent for item in source_paths if item]
            excluded.append(plain_path(Path(value.workspace)).resolve(strict=True))
            if materials.repository is not None:
                excluded.append(plain_path(Path(materials.repository.root)).resolve(strict=True))
            if any(path.is_relative_to(root) for root in excluded):
                return
            payload = json.dumps({
                "schema": "torch-inference-runtime-stage-v1", "scope": "attempt_program_stages",
                "audit_only": True, "adoption_authorized": False,
                "phase": value.phase, "stage": value.stage, "terminal": value.terminal,
                "pre_run_bound": value.pre_run_bound, "source_sites_reached": value.source_sites_reached,
                "source_sites_passed": value.source_sites_passed,
            }, ensure_ascii=True, allow_nan=False) + "\n"
            if len(payload.encode("utf-8")) > 512 or plain_path(parent).resolve(strict=True) != parent:
                return
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except Exception:
            # Diagnostic IO cannot change the original scientific return or cleanup.
            pass
