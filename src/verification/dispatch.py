"""Run exactly the requested peer branches, retaining failed verification reasons."""

import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path

from pydantic import Field

from schemas.claim import Claim, Contract, EvidenceNeed, ExecutionPlan, Finding
from schemas.limitations import VerificationLimitation
from schemas.materials import SharedMaterials
from verification.contracts import BranchResult, RejectedPlan


class VerificationResult(Contract):
    claims: list[Claim]
    plans: list[ExecutionPlan] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    dispatched: dict[str, list[EvidenceNeed]] = Field(default_factory=dict)


async def _invoke(branch, claim, materials):
    if inspect.iscoroutinefunction(branch):
        raw = await branch(claim, materials)
    else:
        raw = await asyncio.to_thread(branch, claim, materials)
        if inspect.isawaitable(raw):
            raw = await raw
    return BranchResult.model_validate(raw)


def _validate_result(claim: Claim, name: EvidenceNeed, result: BranchResult):
    if name != EvidenceNeed.EXPERIMENTS and result.plans:
        raise ValueError("Only Experiments may emit execution plans")
    if name != EvidenceNeed.THEORY and result.theory_derivations:
        raise ValueError("Only Theory may emit derivation records")
    # Validate foreign coverage/questions before mutating the shared claim record.
    Claim.model_validate(
        {
            **claim.model_dump(),
            "evidence": result.evidence,
            "questions": result.questions,
            "theory_derivations": result.theory_derivations,
            "verification_limitations": result.verification_limitations,
        }
    )
    conditions = {condition.id: condition for condition in claim.conditions}
    for plan in result.plans:
        if plan.claim_id != claim.id or any(conditions.get(c.id) != c for c in plan.target_conditions):
            raise ValueError("Execution plan must target the enclosing claim's exact conditions")


async def verify_claims(
    claims: list[Claim],
    materials: SharedMaterials,
    output_dir: Path,
    *,
    submission_deadline=None,
    branches: dict[EvidenceNeed, Callable] | None = None,
    global_literature: Callable | None = None,
    call=None,
    blocked_claim_ids: list[str] | None = None,
) -> VerificationResult:
    if len({claim.id for claim in claims}) != len(claims):
        raise ValueError("Claim identifiers must be unique")
    blocked = set(blocked_claim_ids or [])
    if blocked - {claim.id for claim in claims}:
        raise ValueError("Extraction blocks must refer to retained claims")
    if branches is None:
        from verification.code import verify_code
        from verification.experiments import verify_experiments
        from verification.literature import _claim_source_excerpts, verify_literature
        from verification.theory import verify_theory

        global_targets = []
        for original in claims:
            try:
                global_targets.extend(
                    {**source, "claim_id": original.id}
                    for source in _claim_source_excerpts(original, materials)
                )
            except ValueError:
                # Preserve a visible unavailable target without borrowing another claim's source.
                global_targets.append({"claim_id": original.id})

        async def literature(claim, materials):
            return await verify_literature(
                claim,
                materials,
                submission_deadline=submission_deadline,
                call=call,
                output_dir=output_dir,
                manuscript_targets=global_targets if claim is None else None,
            )

        branches = {
            EvidenceNeed.LITERATURE: literature,
            EvidenceNeed.THEORY: lambda c, m: verify_theory(
                c, m, call=call, output_dir=output_dir / "theory_derivations"
            ),
            EvidenceNeed.CODE: lambda c, m: verify_code(c, m, call=call),
            EvidenceNeed.EXPERIMENTS: lambda c, m: verify_experiments(c, m, call=call),
        }
        global_literature = global_literature or literature
    result = VerificationResult(claims=[claim.model_copy(deep=True) for claim in claims])
    for claim in result.claims:
        if claim.id not in blocked:
            continue
        reason = (
            "Claim coverage review found an unresolved extraction problem. "
            "Repair the claim's meaning, independent conclusions or original conditions "
            "before verification; see screening/claim_coverage/coverage.json."
        )
        claim.verification_limitations.append(
            VerificationLimitation(
                claim_id=claim.id,
                condition_ids=[condition.id for condition in claim.conditions],
                stage="verification",
                kind="claim_extraction_incomplete",
                reason=reason,
            )
        )
        claim.notes.append(reason)
        result.issues.append(f"{claim.id}: {reason}")
        result.dispatched[claim.id] = []
    jobs = [(claim, name) for claim in result.claims if claim.id not in blocked for name in claim.needs]

    async def run(claim, name):
        try:
            branch = branches[name]
            try:
                value = await _invoke(branch, claim.model_copy(deep=True), materials)
            except RejectedPlan as exc:
                if name != EvidenceNeed.EXPERIMENTS or exc.observations.plans:
                    raise ValueError(
                        "Rejected-plan recovery requires Experiments observations without plans"
                    ) from exc
                value = exc.observations
                value.issues.append(f"Execution plan rejected: {exc}")
                value.verification_limitations.append(
                    VerificationLimitation(
                        claim_id=claim.id,
                        condition_ids=[c.id for c in claim.conditions],
                        stage=name.value,
                        kind="plan_rejected",
                        reason=str(exc),
                    )
                )
            _validate_result(claim, name, value)
            return value
        except Exception as exc:
            message = f"{name} verification failed: {exc}"
            return BranchResult(
                issues=[message],
                verification_limitations=[
                    VerificationLimitation(
                        claim_id=claim.id,
                        condition_ids=[c.id for c in claim.conditions],
                        stage=name.value,
                        kind="branch_failed",
                        reason=message,
                    )
                ],
            )

    outputs = await asyncio.gather(*(run(claim, name) for claim, name in jobs))
    for (claim, name), output in zip(jobs, outputs, strict=True):
        result.dispatched.setdefault(claim.id, []).append(name)
        claim.evidence.extend(output.evidence)
        claim.questions.extend(output.questions)
        claim.notes.extend(output.issues)
        claim.theory_derivations.extend(output.theory_derivations)
        claim.verification_limitations.extend(output.verification_limitations)
        result.plans.extend(output.plans)
        result.findings.extend(output.findings)
        result.issues.extend(f"{claim.id}: {issue}" for issue in output.issues)
    if global_literature is not None:
        try:
            global_result = await _invoke(global_literature, None, materials)
            if (
                global_result.plans
                or global_result.evidence
                or global_result.questions
                or global_result.theory_derivations
                or global_result.verification_limitations
            ):
                raise ValueError("Global literature produces findings/issues only")
            result.findings.extend(global_result.findings)
            result.issues.extend(global_result.issues)
        except Exception as exc:
            result.issues.append(f"Global literature search failed: {exc}")
    if len({plan.id for plan in result.plans}) != len(result.plans):
        raise ValueError("Execution plan identifiers must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "verification.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
