"""Fixed consumer-gate sidecars; writing cannot affect scientific verification."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

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

    def record(self, code, *, first_subguard=None):
        if self.attempted:
            return
        self.attempted = True
        try:
            if (self.directory is None or type(self.branch) is not str or self.branch not in _CODES
                    or type(code) is not str or code not in _CODES[self.branch]):
                return
            directory = plain_path(Path(self.directory)).resolve()
            if directory.name != "diagnostics":
                return
            operator_root = plain_path(directory.parent).resolve()
            if not directory.is_relative_to(operator_root):
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
                return
            directory.mkdir(parents=True, exist_ok=True)
            if plain_path(directory).resolve(strict=True) != directory:
                return
            path = plain_path(directory / f"verification_rejection_{uuid.uuid4().hex}.json")
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
