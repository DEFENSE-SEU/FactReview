"""Closed ordinary-condition wire and retained-source reconstruction; no model calls."""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from common import run_stats
from fact_generation.execution.repair_evidence import file_identity, plain_path
from schemas.claim import Contract, ExecutionOriginV4, NonEmpty
from screening.visual_audit import redacted_record
from verification.execution_projection import digest
from verification.experiments import ExperimentItem, PlanCandidate, PlanTarget

VERSION = "experiments-execution-v4"
DOMAIN = "ordinary-conditions-v1"
SCIENTIFIC_FIELDS = ("id", "text", "loc", "source_block_id", "source_quote", "source_refs", "conditions", "needs", "importance")


class OrdinaryPlanTargetV4(PlanTarget):
    projection: Literal[None] = None


class OrdinaryPlanCandidateV4(PlanCandidate):
    targets: list[OrdinaryPlanTargetV4] = Field(min_length=1, max_length=1)


class OrdinaryConditionDecisionV4(Contract):
    condition_id: NonEmpty
    condition_digest: NonEmpty
    kind: Literal["ordinary"]
    plan: OrdinaryPlanCandidateV4


class NoConditionExecutionV4(Contract):
    condition_id: NonEmpty
    condition_digest: NonEmpty
    kind: Literal["none"]
    rationale: NonEmpty


class ExperimentsOutputV4(Contract):
    schema_version: Literal["experiments-execution-v4"]
    execution_domain: Literal["ordinary-conditions-v1"]
    execution_context_digest: NonEmpty
    checked_aspects: list[Literal["correspondence", "fairness", "isolation", "stability", "consistency"]]
    items: list[ExperimentItem]
    execution: list[Annotated[OrdinaryConditionDecisionV4 | NoConditionExecutionV4, Field(discriminator="kind")]]
    issues: list[str] = Field(default_factory=list)


class ExperimentsOutputV4Normalized(Contract):
    checked_aspects: list[Literal["correspondence", "fairness", "isolation", "stability", "consistency"]]
    items: list[ExperimentItem]
    plans: list[OrdinaryPlanCandidateV4] = Field(default_factory=list)
    execution_choices: list = Field(default_factory=list, max_length=0)
    issues: list[str] = Field(default_factory=list)


def source_snapshot(claim, materials):
    scientific = {key: claim.model_dump(mode="json")[key] for key in SCIENTIFIC_FIELDS}
    files = {}
    for value in (materials.source_pdf, materials.markdown_path):
        files[str(plain_path(Path(value)).resolve(strict=True))] = file_identity(Path(value))
    if materials.repository:
        root = plain_path(Path(materials.repository.root)).resolve(strict=True)
        for row in materials.repository.files:
            path = plain_path(root / row.path).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("Ordinary-v4 resource is outside its original repository")
            identity = file_identity(path)
            if identity["sha256"] != row.sha256:
                raise ValueError("Ordinary-v4 indexed source changed")
            files[str(path)] = identity
    from verification import experiment_targets, experiments
    modules = {str(Path(module.__file__).resolve()): file_identity(Path(module.__file__))
               for module in (experiments, experiment_targets)}
    return {"claim": scientific, "materials": materials.model_dump(mode="json"), "files": files, "producer": modules}


def required_conditions(claim):
    return [{"condition_id": c.id, "condition_digest": digest(c.model_dump(mode="json"))} for c in claim.conditions]


def output_schema(claim, context):
    schema = ExperimentsOutputV4.model_json_schema()
    rows = schema["properties"]["execution"]
    rows.update(minItems=len(claim.conditions), maxItems=len(claim.conditions), prefixItems=[
        {"allOf": [copy.deepcopy(rows["items"])], "type": "object", "properties": {
            "condition_id": {"const": row["condition_id"]}, "condition_digest": {"const": row["condition_digest"]}},
         "required": ["condition_id", "condition_digest"]} for row in required_conditions(claim)])
    schema["properties"]["execution_context_digest"]["const"] = context
    return schema


def reserve_audit(directory, materials):
    if directory is None:
        stats = run_stats.stats_path()
        if stats is None:
            raise ValueError("Ordinary-v4 requires an auditable directory before delegation")
        directory = stats.parent / "execution_decisions"
    directory = plain_path(Path(directory))
    directory.mkdir(parents=True, exist_ok=True)
    directory = plain_path(directory).resolve(strict=True)
    if not directory.is_dir() or (materials.repository and directory.is_relative_to(
            plain_path(Path(materials.repository.root)).resolve(strict=True))):
        raise ValueError("Ordinary-v4 audit must be outside its author repository")
    bundle = directory / uuid.uuid4().hex
    bundle.mkdir(exist_ok=False)
    for name in ("raw.json", "origin.json"):
        with (bundle / name).open("xb"):
            pass
    return bundle


def save_raw(bundle, raw):
    # Accepted data must stay exact through the existing public privacy boundary.
    if digest(redacted_record(raw, cfg=None)) != digest(raw):
        raise ValueError("Ordinary-v4 raw response cannot be retained without changing its content")
    (bundle / "raw.json").write_text(json.dumps(raw, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def lower_v4(raw, claim, context):
    """Partition failures close all plans; well-identified invalid plans remain local."""
    normalized = {k: copy.deepcopy(raw[k]) for k in ("checked_aspects", "items", "issues") if k in raw}
    normalized.update(plans=[], execution_choices=[])
    states = [{**r, "ordinal": i, "state": "missing"} for i, r in enumerate(required_conditions(claim))]
    wire = {"version": VERSION, "domain": DOMAIN, "status": "rejected", "states": states,
            "raw_response_sha256": digest(raw), "errors": []}
    required = {"schema_version", "execution_domain", "execution_context_digest", "checked_aspects", "items", "execution"}
    rows = raw.get("execution")
    if (not required.issubset(raw) or set(raw) - required - {"issues"}
            or raw.get("schema_version") != VERSION or raw.get("execution_domain") != DOMAIN
            or raw.get("execution_context_digest") != context or type(rows) is not list
            or len(rows) != len(states) or any(type(row) is not dict or row.get("condition_id") != state["condition_id"]
                or row.get("condition_digest") != state["condition_digest"] or row.get("kind") not in {"ordinary", "none"}
                or set(row) != ({"condition_id", "condition_digest", "kind", "plan"} if row.get("kind") == "ordinary"
                                else {"condition_id", "condition_digest", "kind", "rationale"})
                for row, state in zip(rows or [], states, strict=True))):
        wire["errors"].append("Ordinary-v4 full ordered condition partition/version/context is invalid")
        return normalized, wire
    wire["status"] = "partition_closed"
    for ordinal, (row, state) in enumerate(zip(rows, states, strict=True)):
        state.update(raw_row_pointer=f"/execution/{ordinal}", raw_row_sha256=digest(row))
        try:
            if row["kind"] == "none":
                NoConditionExecutionV4.model_validate(row)
                state["state"] = "none"
                continue
            parsed = OrdinaryConditionDecisionV4.model_validate(row)
            if parsed.plan.targets[0].condition_id != row["condition_id"]:
                raise ValueError("Ordinary target does not match its original condition row")
            index = len(normalized["plans"])
            normalized["plans"].append(copy.deepcopy(row["plan"]))
            state.update(state="candidate", plan_index=index, raw_plan_pointer=f"/execution/{ordinal}/plan",
                         raw_plan_sha256=digest(row["plan"]), normalized_plan_pointer=f"/candidate_plans/{index}")
        except ValueError as exc:
            state.update(state="rejected", reason=str(exc))
            wire["errors"].append(f"{row['condition_id']}: {exc}")
    return normalized, wire


def finish_audit(bundle, raw, normalized, wire, snapshot, context, plans, *, scope_audit=None):
    decisions = copy.deepcopy(wire["states"])
    for state in decisions:
        plan = next((p for p in plans if p.condition_ids == [state["condition_id"]]), None)
        if plan is not None:
            state["state"] = plan.feasibility
        elif state["state"] == "candidate":
            state.update(state="rejected", reason="Candidate did not yield a retained original plan")
    audit = {"version": VERSION, "domain": DOMAIN, "execution_context_digest": context,
             "source_snapshot": snapshot, "raw_response": copy.deepcopy(raw),
             "raw_locator": str(bundle / "raw.json"), "raw_file_sha256": file_identity(bundle / "raw.json")["sha256"],
             "normalized": copy.deepcopy(normalized), "wire": copy.deepcopy(wire),
             "candidate_plans": copy.deepcopy(normalized["plans"]), "condition_decisions": decisions,
             "scope_audit": {"locator": str(scope_audit), "sha256": file_identity(scope_audit)["sha256"]}
                if scope_audit is not None else None,
             "adapter_sha256": file_identity(Path(__file__))["sha256"],
             "generated_plans": {plan.id: plan.model_dump(mode="json") for plan in plans}}
    path = bundle / "origin.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    sha = file_identity(path)["sha256"]
    for plan in plans:
        state = next(row for row in wire["states"] if row["condition_id"] == plan.condition_ids[0])
        plan.execution_origin = ExecutionOriginV4(version=VERSION, domain=DOMAIN, audit_locator=str(path),
            audit_sha256=sha, execution_context_digest=context, claim_digest=digest(snapshot["claim"]),
            condition_id=state["condition_id"], condition_digest=state["condition_digest"], ordinal=state["ordinal"],
            raw_response_sha256=digest(raw), raw_row_pointer=state["raw_row_pointer"], raw_row_sha256=state["raw_row_sha256"],
            raw_plan_pointer=state["raw_plan_pointer"], raw_plan_sha256=state["raw_plan_sha256"], plan_index=state["plan_index"],
            normalized_plan_pointer=state["normalized_plan_pointer"], normalized_response_sha256=digest(normalized),
            adapter_sha256=audit["adapter_sha256"])


def validate_execution_origin_v4(plan, claim, materials):
    from verification.experiments import _plan

    origin = plan.execution_origin
    if origin is None:
        if plan.id.startswith(claim.id + ".plan.v4."):
            raise ValueError("Ordinary-v4 plan is missing its retained original response")
        return
    path = plain_path(Path(origin.audit_locator)).resolve(strict=True)
    if file_identity(path)["sha256"] != origin.audit_sha256:
        raise ValueError("Ordinary-v4 original audit changed")
    audit = json.loads(path.read_text(encoding="utf-8"))
    snapshot = source_snapshot(claim, materials)
    context = digest({"version": VERSION, "domain": DOMAIN, "snapshot": snapshot})
    if (audit["source_snapshot"] != snapshot or origin.execution_context_digest != context
            or audit["execution_context_digest"] != context or origin.claim_digest != digest(snapshot["claim"])
            or origin.adapter_sha256 != file_identity(Path(__file__))["sha256"]
            or audit["adapter_sha256"] != origin.adapter_sha256):
        raise ValueError("Ordinary-v4 original scientific/source context changed")
    raw_path = plain_path(path.parent / "raw.json")
    if str(raw_path) != audit["raw_locator"] or file_identity(raw_path)["sha256"] != audit["raw_file_sha256"]:
        raise ValueError("Ordinary-v4 first raw file identity changed")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw != audit["raw_response"] or digest(raw) != origin.raw_response_sha256:
        raise ValueError("Ordinary-v4 retained raw response differs")
    normalized, wire = lower_v4(raw, claim, context)
    if (normalized != audit["normalized"] or normalized["plans"] != audit["candidate_plans"]
            or wire["status"] != "partition_closed"):
        raise ValueError("Ordinary-v4 original partition cannot be rebuilt")
    state = wire["states"][origin.ordinal]
    if (state["state"] != "candidate" or state["condition_id"] != origin.condition_id
            or state["condition_digest"] != origin.condition_digest or state["plan_index"] != origin.plan_index
            or any(state[key] != getattr(origin, key) for key in ("raw_row_pointer", "raw_row_sha256", "raw_plan_pointer",
                       "raw_plan_sha256", "normalized_plan_pointer"))
            or digest(normalized) != origin.normalized_response_sha256
            or plan.id != f"{claim.id}.plan.v4.{origin.ordinal}"):
        raise ValueError("Ordinary-v4 origin/ordinal/normalized pointer does not match its retained row")
    candidate = OrdinaryPlanCandidateV4.model_validate(normalized["plans"][origin.plan_index])
    rebuilt = _plan(claim, materials, candidate, plan_index=origin.plan_index, plan_id=plan.id)
    current = plan.model_dump(mode="json")
    current.pop("execution_origin", None)
    if current != rebuilt.model_dump(mode="json") or current != audit["generated_plans"].get(plan.id):
        raise ValueError("Ordinary-v4 plan differs from the retained raw source/target/resource reconstruction")
