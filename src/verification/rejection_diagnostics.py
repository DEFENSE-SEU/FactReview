"""Fixed consumer-gate sidecars; writing cannot affect scientific verification."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import RLock

from fact_generation.execution.repair_evidence import plain_path

_CODES = {
    "Theory": frozenset({
        "theory_no_proof_trace_invalid", "theory_concern_trace_invalid",
        "theory_concern_target_source_missing", "theory_visual_recheck_invalid",
        "theory_concern_applicability_unresolved",
    }),
    "Experiments": frozenset({
        "experiments_choice_privacy_blocked", "experiments_joint_final_validation_failed",
    }),
}


_JOINT_SUBGUARDS = frozenset({
    "candidate_contract", "source_count", "member_pointer", "member_lookup",
    "member_quote_or_range", "member_record", "condition_candidate_uniqueness",
    "scope_item_schema", "scope_pair_identity", "scope_inner_contract", "scope_manifest_invalid",
    "source_uses_member_coverage", "source_uses_binding", "unused_mechanical_source_role",
    "scope_condition", "scope_operand", "scope_ground", "scope_pair_missing", "primary_pointer", "unknown",
})


_OUTCOME_CONTEXT = ContextVar("verification_write_outcomes", default=None)
_OUTCOME_STAGES = frozenset({
    "writer", "branch_lookup", "invoke_or_branch_contract", "rejected_plan_recovery",
    "result_validation", "unknown",
})
_OUTCOMES = frozenset({
    "constructed", "record_called", "attempted", "already_attempted",
    "disabled_or_invalid_arguments", "invalid_destination", "path_validation_failed",
    "input_overlap", "payload_capacity", "path_identity_changed", "written", "io_failed", "unknown", "failed",
}) | frozenset(f"joint_subguard_{tag}" for tag in _JOINT_SUBGUARDS)
_OUTCOME_CODES = frozenset().union(*_CODES.values(), {"experiments_dispatch_failed", "unknown"})
_OUTCOME_MAX_ROWS = 128
_OUTCOME_MAX_COUNT = 1_000_000


class _WriteOutcomes:
    """Bounded captured events, with no evidence or completeness authority."""

    def __init__(self):
        self._lock = RLock()
        self._counts = {}
        self._closed = self._overflow = self._unavailable = False

    def _record(self, branch, code, stage, outcome):
        branch = branch if type(branch) is str and branch in _CODES else "unknown"
        code = code if type(code) is str and code in _OUTCOME_CODES else "unknown"
        stage = stage if type(stage) is str and stage in _OUTCOME_STAGES else "unknown"
        outcome = outcome if type(outcome) is str and outcome in _OUTCOMES else "unknown"
        key = (branch, code, stage, outcome)
        with self._lock:
            if self._closed:
                return
            count = self._counts.get(key, 0)
            if ((key not in self._counts and len(self._counts) >= _OUTCOME_MAX_ROWS)
                    or count >= _OUTCOME_MAX_COUNT):
                self._overflow = True
                return
            self._counts[key] = count + 1

    def _mark_unavailable(self):
        try:
            with self._lock:
                if not self._closed:
                    self._unavailable = True
        except Exception:
            pass

    def close_snapshot(self):
        try:
            with self._lock:
                self._closed = True
                rows = [dict(zip(("branch", "code", "stage", "outcome"), key, strict=True), count=count)
                        for key, count in sorted(self._counts.items())]
                return {
                    "schema": "verification-write-outcomes-v1", "audit_only": True,
                    "adoption_authorized": False, "coverage": "captured_scope_events_only",
                    "closed": True, "overflow": self._overflow, "unavailable": self._unavailable,
                    "events": rows,
                }
        except Exception:
            return {
                "schema": "verification-write-outcomes-v1", "audit_only": True,
                "adoption_authorized": False, "coverage": "captured_scope_events_only",
                "closed": False, "overflow": False, "unavailable": True, "events": [],
            }


@contextmanager
def write_outcome_scope(*, enabled=False):
    """Explicit opt-in; copied async thread contexts share this scope's collector."""
    try:
        collector = _WriteOutcomes() if enabled is True else None
    except Exception:
        collector = None
    token = _OUTCOME_CONTEXT.set(collector)
    try:
        yield collector
    finally:
        try:
            if collector is not None:
                collector.close_snapshot()
        finally:
            _OUTCOME_CONTEXT.reset(token)


def _write_event(writer, code, outcome):
    collector = None
    try:
        collector = _OUTCOME_CONTEXT.get()
        if type(collector) is _WriteOutcomes:
            collector._record(writer.branch, code, "writer", outcome)
    except Exception:
        if type(collector) is _WriteOutcomes:
            collector._mark_unavailable()


def record_experiments_dispatch_failure(stage):
    """A fixed memory breadcrumb at the existing generic exception boundary."""
    collector = None
    try:
        collector = _OUTCOME_CONTEXT.get()
        if type(collector) is _WriteOutcomes:
            collector._record("Experiments", "experiments_dispatch_failed", stage, "failed")
    except Exception:
        if type(collector) is _WriteOutcomes:
            collector._mark_unavailable()


class JointSubguardCollector:
    """Program-only first rejection tags; never added to a scientific manifest."""

    def __init__(self):
        self._first = {}

    def record(self, index, code):
        try:
            if type(index) is int and index >= 0:
                value = code if type(code) is str and code in _JOINT_SUBGUARDS else "unknown"
                self._first.setdefault(index, value)
        except Exception:
            pass

    def clear(self, index):
        try:
            self._first.pop(index, None)
        except Exception:
            pass

    def first(self, index):
        try:
            return self._first.get(index, "unknown")
        except Exception:
            return "unknown"


class FirstRejectionDiagnostic:
    """At most one attempted fixed diagnostic per invocation of a peer branch."""

    def __init__(self, directory, materials, branch):
        self.directory, self.materials, self.branch = directory, materials, branch
        self.attempted = False
        _write_event(self, "unknown", "constructed")

    def record(self, code, *, first_subguard=None):
        _write_event(self, code, "record_called")
        if self.attempted:
            _write_event(self, code, "already_attempted")
            return
        self.attempted = True
        _write_event(self, code, "attempted")
        if (first_subguard is not None and type(self.branch) is str and self.branch == "Experiments"
                and type(code) is str and code == "experiments_joint_final_validation_failed"):
            tag = first_subguard if type(first_subguard) is str and first_subguard in _JOINT_SUBGUARDS else "unknown"
            _write_event(self, code, f"joint_subguard_{tag}")
        try:
            if (self.directory is None or type(self.branch) is not str or self.branch not in _CODES
                    or type(code) is not str or code not in _CODES[self.branch]):
                _write_event(self, code, "disabled_or_invalid_arguments")
                return
            directory = plain_path(Path(self.directory)).resolve()
            if directory.name != "diagnostics":
                _write_event(self, code, "invalid_destination")
                return
            operator_root = plain_path(directory.parent).resolve()
            if not directory.is_relative_to(operator_root):
                _write_event(self, code, "path_validation_failed")
                return
            materials = self.materials
            source_paths = [materials.source_pdf, materials.markdown_path, materials.content_list_path]
            source_paths.extend(page.path for page in materials.pages)
            source_paths.extend(path for item in [*materials.figures, *materials.tables]
                                for path in (item.crop_path, item.printed_crop_path))
            excluded = [plain_path(Path(path)).resolve().parent for path in source_paths if path]
            if materials.repository is not None:
                excluded.append(plain_path(Path(materials.repository.root)).resolve())
            if any(directory.is_relative_to(root) for root in excluded):
                _write_event(self, code, "input_overlap")
                return
            fields = {
                "schema": "verification-rejection-diagnostics-v1", "branch": self.branch,
                "scope": "first_evidence_validation_rejection", "code": code,
            }
            if (self.branch == "Experiments" and code == "experiments_joint_final_validation_failed"
                    and first_subguard is not None):
                fields.update(schema="verification-rejection-diagnostics-v2", first_subguard=(
                    first_subguard if type(first_subguard) is str and first_subguard in _JOINT_SUBGUARDS else "unknown"
                ))
            payload = json.dumps(fields, ensure_ascii=True, allow_nan=False) + "\n"
            if len(payload.encode("utf-8")) > 512:
                _write_event(self, code, "payload_capacity")
                return
            directory.mkdir(parents=True, exist_ok=True)
            if plain_path(directory).resolve(strict=True) != directory:
                _write_event(self, code, "path_identity_changed")
                return
            path = plain_path(directory / f"verification_rejection_{uuid.uuid4().hex}.json")
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
            _write_event(self, code, "written")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            _write_event(self, code, "io_failed" if isinstance(exc, OSError) else "unknown")
            pass
