"""Read completed host execution records after a stage failure, without resuming it."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .v2_config import ExecutionConfig


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _linked(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _validate_record(row, expected, directory):
    from .v2 import ExecutionOperationFailure, RunOutcome, RunRequest

    required = {
        "plan",
        "approval_mode",
        "training_budget",
        "training_used_before",
        "approved",
        "reason",
        "attempts",
        "repairs",
        "alignment",
        "paper_target_validation",
        "tolerance_profile",
        "config",
    }
    if (
        not required.issubset(row)
        or set(row) - required - {"workspace", "refinement", "operation_failures"}
        or _encoded(row["plan"]) != _encoded(expected)
    ):
        raise ValueError("incomplete record or changed plan snapshot")
    if type(row["approved"]) is not bool or not isinstance(row["reason"], str):
        raise ValueError("invalid approval or reason")
    config = ExecutionConfig.model_validate(row["config"]).model_dump(mode="json")
    if _encoded(config) != _encoded(row["config"]) or row["approval_mode"] != config["approval_mode"]:
        raise ValueError("invalid execution configuration")
    for field in ("training_budget", "training_used_before"):
        if type(row[field]) is not int or row[field] < 0:
            raise ValueError("invalid training count")
    if row["training_budget"] != config["training_budget"] or row["tolerance_profile"] != "alignment":
        raise ValueError("inconsistent execution policy")
    for field in ("attempts", "repairs", "alignment", "paper_target_validation"):
        if not isinstance(row[field], list) or any(not isinstance(item, dict) for item in row[field]):
            raise ValueError("invalid execution record list")
    if "operation_failures" in row:
        failures = row["operation_failures"]
        if not isinstance(failures, list) or any(
            not isinstance(item, dict)
            or _encoded(ExecutionOperationFailure.model_validate(item).model_dump()) != _encoded(item)
            for item in failures
        ):
            raise ValueError("invalid producer-declared operation failures")
    if row["attempts"] and not row["approved"]:
        raise ValueError("attempts recorded without approval")
    if not row["attempts"] and not row["reason"].strip():
        raise ValueError("record has no finalized attempt or blocker")
    if "workspace" in row and Path(row["workspace"]).resolve() != directory / "workspace":
        raise ValueError("workspace does not belong to this run")
    for ordinal, attempt in enumerate(row["attempts"]):
        request = RunRequest.model_validate(attempt.get("request"))
        outcome_data = {key: value for key, value in attempt.items() if key != "request"}
        outcome = RunOutcome.model_validate(outcome_data)
        serialized_outcome = outcome.model_dump(mode="json")
        if "operation_failures" not in outcome_data:
            # Historical outcomes predate this additive field. Every other key
            # remains mandatory and exact; recovery returns the original bytes.
            serialized_outcome.pop("operation_failures")
        if (
            _encoded(request.model_dump(mode="json")) != _encoded(attempt["request"])
            or _encoded(serialized_outcome) != _encoded(outcome_data)
            or _encoded(request.plan.model_dump(mode="json")) != _encoded(expected)
            or _encoded(request.config.model_dump(mode="json")) != _encoded(config)
            or Path(request.run_dir).resolve() != directory
            or Path(request.workspace).resolve() != directory / "workspace"
            or request.repair_round != ordinal
            or ordinal > config["max_attempts"]
        ):
            raise ValueError("attempt does not belong to this plan and run")


def recover_execution_records(output_dir, plans) -> tuple[list[dict], list[str]]:
    """Return original finalized ledger rows and recovery issues, never claim evidence.

    Only immediate run_NNNN/ledger.json files are eligible. A duplicate known ID
    invalidates every record for that ID, including an otherwise valid first row.
    Missing/invalid records cannot establish that no attempt executed.
    """
    issues = []
    root = Path(output_dir)
    expected_counts = Counter(plan.id for plan in plans)
    expected = {plan.id: plan.model_dump(mode="json") for plan in plans if expected_counts[plan.id] == 1}
    duplicate_ids = {identifier for identifier, count in expected_counts.items() if count > 1}
    if duplicate_ids:
        issues.append(
            "Execution recovery excluded duplicate input plan IDs: " + ", ".join(sorted(duplicate_ids))
        )
    records, seen = {}, Counter()
    if _linked(root):
        issues.append("Execution recovery rejected a linked output directory.")
    elif root.exists():
        try:
            directories = sorted(root.iterdir(), key=lambda path: path.name)
        except OSError:
            directories = []
            issues.append("Execution recovery could not list the output directory.")
        for directory in directories:
            if not re.fullmatch(r"run_\d{4,}", directory.name):
                continue
            path = directory / "ledger.json"
            try:
                if _linked(directory) or _linked(path) or not directory.is_dir():
                    raise ValueError("linked or invalid run directory")
                row = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
                _encoded(row)  # Reject NaN and other non-JSON numeric constants.
                raw_id = row.get("plan", {}).get("id") if isinstance(row, dict) else None
                if not isinstance(raw_id, str):
                    raise ValueError("missing plan ID")
                identifier = raw_id.strip()
                seen[identifier] += 1
                if identifier not in expected:
                    raise ValueError("unknown or ambiguous plan ID")
                _validate_record(row, expected[identifier], directory.resolve())
                records[identifier] = row
            except (OSError, ValueError, TypeError, AttributeError, KeyError):
                issues.append(
                    f"Execution recovery excluded {directory.name}/ledger.json: invalid, missing, or unbound record."
                )
    for identifier, count in seen.items():
        if count > 1:
            records.pop(identifier, None)
            issues.append(f"Execution recovery excluded duplicate records for plan {identifier}.")
    for identifier in expected:
        if identifier not in records:
            issues.append(
                f"No complete host ledger was recovered for plan {identifier}; whether attempts executed is unknown."
            )
    ledger = list(records.values())
    issues.append(
        f"Recovered {len(ledger)} finalized execution record(s) for audit only. Attempts may have executed "
        "without their evidence being adopted. Recovery does not resume execution or restore claim evidence."
    )
    return ledger, issues
