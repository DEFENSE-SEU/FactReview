"""Paper-only experiment checks and the sole claim-linked execution-plan producer."""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import Field

from schemas.claim import (
    AuthorQuestion,
    Claim,
    Contract,
    Evidence,
    ExecutionPlan,
    ExecutionTask,
    NonEmpty,
)
from schemas.materials import SharedMaterials
from screening.checks import ask
from verification.contracts import BranchResult
from verification.theory import _covered, _paper_pointer


class PaperNumber(Contract):
    block_id: NonEmpty
    quote: NonEmpty
    token: NonEmpty
    # A precise sentence/row/cell context may disambiguate a larger quoted table.
    value_context: str = ""


def _has_label(label: str | None, quote: str) -> bool:
    return bool(label and re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", quote, re.I))


def _number(materials: SharedMaterials, item: PaperNumber) -> float:
    _paper_pointer(materials, item.block_id, item.quote)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?", item.token):
        raise ValueError("A target must quote a finite numeric token from the paper")
    if not re.search(r"(?<![\w.])" + re.escape(item.token) + r"(?![\w%]|\.\d)", item.quote):
        raise ValueError("Reported number token is absent from the quoted paper passage")
    # Preserve the paper's metric scale: 89% remains 89, alongside its quote.
    value = float(item.token.rstrip("%"))
    if not math.isfinite(value):
        raise ValueError("Reported target must be finite")
    return value


def _target_ambiguity(claim: Claim, target: PlanTarget, value: float) -> str:
    """Conservatively block target assignment when a quote contains several values."""
    reported = target.reported
    context = reported.value_context or reported.quote
    if context not in reported.quote:
        raise ValueError("value_context must be an exact substring of the quoted paper passage")
    condition = next(item for item in claim.conditions if item.id == target.condition_id)
    if not _has_label(condition.metric, context) or (
        condition.dataset and not _has_label(condition.dataset, context)
    ):
        return "Exact value context does not identify the target dataset and metric"
    for setting, setting_value in condition.settings.items():
        if not _has_label(str(setting_value), context):
            return f"Exact value context does not identify the target setting {setting}={setting_value}"
    other_datasets = {item.dataset for item in claim.conditions if item.dataset != condition.dataset}
    if any(_has_label(dataset, context) for dataset in other_datasets):
        return "Exact value context contains multiple target datasets"
    # Dataset and metric labels can contain digits (FB15k-237, Hits@10).
    # Exclude those labels before counting independent reported numbers.
    numeric_context = context
    for label in (condition.dataset, condition.metric):
        if label:
            numeric_context = re.sub(
                r"(?<!\w)" + re.escape(label) + r"(?!\w)", " ", numeric_context, flags=re.I
            )
    for setting, setting_value in condition.settings.items():
        if isinstance(setting_value, (int, float)) and not isinstance(setting_value, bool):
            pattern = (
                r"(?<!\w)"
                + re.escape(setting)
                + r"\s*[:=]?\s*"
                + re.escape(str(setting_value))
                + r"(?![\w.]|\.\d)"
            )
            numeric_context, count = re.subn(pattern, " ", numeric_context, flags=re.I)
            if not count:
                return f"Numeric setting {setting}={setting_value} is not explicitly bound in the context"
    tokens = re.findall(
        r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?(?!\w|\.\d)", numeric_context
    )
    values = {float(token.rstrip("%")) for token in tokens}
    if values != {value}:
        return "Exact value context does not uniquely bind the reported number to this condition"
    return ""


class ExperimentItem(Contract):
    aspect: Literal["correspondence", "fairness", "isolation", "stability", "consistency"]
    kind: Literal[
        "paper_support",
        "missing_control",
        "missing_ablation",
        "missing_statistic",
        "small_gap_without_statistics",
        "large_gap_no_variance",
        "text_table_contradiction",
    ]
    block_id: NonEmpty
    quote: NonEmpty
    covered: list[NonEmpty]
    detail: NonEmpty
    # Contradiction pairs must be the same metric/setting; units are explicit.
    comparison: list[PaperNumber] = Field(default_factory=list, max_length=2)


class PlanTarget(Contract):
    condition_id: NonEmpty
    reported: PaperNumber


class PlanCandidate(Contract):
    targets: list[PlanTarget] = Field(min_length=1)
    entry_script: str | None = None
    config: str | None = None
    run_mode: Literal["evaluation", "analysis", "training"]
    feasibility: Literal["ready", "blocked"]
    blocker: str = ""
    priority: Literal["high", "medium", "low"]
    estimated_cost: str = "unknown"
    data_paths: list[str] = Field(default_factory=list)
    weight_paths: list[str] = Field(default_factory=list)


class ExperimentsOutput(Contract):
    checked_aspects: list[Literal["correspondence", "fairness", "isolation", "stability", "consistency"]]
    items: list[ExperimentItem]
    plans: list[PlanCandidate] = Field(default_factory=list, max_length=1)
    issues: list[str] = Field(default_factory=list)


def _plan(claim: Claim, materials: SharedMaterials, candidate: PlanCandidate) -> ExecutionPlan:
    conditions = {condition.id: condition for condition in claim.conditions}
    ids = _covered(claim, [target.condition_id for target in candidate.targets])
    values = {target.condition_id: _number(materials, target.reported) for target in candidate.targets}
    target_issues = []
    for target in candidate.targets:
        condition = conditions[target.condition_id]
        # Exact quotes must identify the target metric/dataset, possibly across
        # a full table containing the header and dataset row.
        if not _has_label(condition.metric, target.reported.quote):
            raise ValueError("Reported target quote must identify the condition's metric")
        if condition.dataset and not _has_label(condition.dataset, target.reported.quote):
            raise ValueError("Reported target quote must identify the condition's dataset")
        ambiguity = _target_ambiguity(claim, target, values[target.condition_id])
        if ambiguity:
            target_issues.append(f"Unresolved paper target {target.condition_id}: {ambiguity}")
    repository = materials.repository
    files = {item.path for item in repository.files} if repository else set()
    entries = set(repository.entry_scripts) if repository else set()
    configs = set(repository.configs) if repository else set()
    if candidate.entry_script and candidate.entry_script not in entries:
        raise ValueError("Execution-plan entry script must come from the repository index")
    if candidate.config and candidate.config not in configs:
        raise ValueError("Execution-plan config must come from the repository index")
    if not set(candidate.data_paths + candidate.weight_paths).issubset(files):
        raise ValueError("Execution resources must be listed in the repository index")
    blockers = ([candidate.blocker] if candidate.blocker.strip() else []) + target_issues
    if repository is None or not candidate.entry_script:
        blockers.append("Candidate released code is missing")
    if not candidate.data_paths:
        blockers.append("Released data location is missing")
    if candidate.run_mode == "evaluation" and not candidate.weight_paths:
        blockers.append("Released weights are missing")
    if candidate.feasibility == "blocked" and not blockers:
        blockers.append("Plan is blocked; runtime requirements need clarification")
    # Training approval/budget is enforced by L3; preserve its candidate here.
    return ExecutionPlan(
        id=f"{claim.id}.plan",
        claim_id=claim.id,
        condition_ids=ids,
        target_conditions=[conditions[key] for key in ids],
        y_paper=values,
        task=ExecutionTask(entry_script=candidate.entry_script, config=candidate.config),
        run_mode=candidate.run_mode,
        feasibility="blocked" if blockers else "ready",
        blocker="; ".join(blockers),
        priority=candidate.priority,
        estimated_cost=candidate.estimated_cost,
    )


def verify_experiments(claim: Claim, materials: SharedMaterials, *, call=None) -> BranchResult:
    output = ExperimentsOutput.model_validate(
        ask(
            "Check this experimental claim from the paper alone in five aspects: correspondence "
            "(an experiment for every assertion), fairness (same data/budget/tuning), isolation "
            "(credited components ablated), stability (variance/seeds/significance for small gaps), "
            "consistency (abstract/text/table numbers). List all five in checked_aspects. Return output_schema JSON. Every item needs "
            "an exact located paper quote and a concrete detail. paper_support stays paper-internal. "
            "large_gap_no_variance is a non-decisive note. text_table_contradiction requires two "
            "quoted numerical passages for the same target condition and metric, on the same scale. "
            "For each target claim with re-obtainable reported numbers, emit one plan. Include candidate "
            "entry script/config only from the supplied repository index; unknown commands/metric output "
            "remain for L3. Every plan target quotes the exact paper numeric token, metric, and dataset; "
            "copy full table headers when needed. For multi-value quotes, supply an exact value_context "
            "substring identifying the target dataset/metric and its unique reported value. Ambiguous "
            "whole-table references remain blocked until the target is resolved. Include actual indexed data/weight paths. Keep plans "
            "with missing code, data, weights, or budget as blocked with a reason. Priority follows "
            "the link to the paper's core contribution. Emit no execution evidence or final verdict.",
            {
                "claim": claim.model_dump(mode="json"),
                "paper_blocks": [b.model_dump() for b in materials.blocks],
                "repository_index": materials.repository.model_dump() if materials.repository else None,
                "output_schema": ExperimentsOutput.model_json_schema(),
            },
            module="verification.experiments",
            call=call,
        )
    )
    if set(output.checked_aspects) != {"correspondence", "fairness", "isolation", "stability", "consistency"}:
        raise ValueError("Experiments must inspect all five required aspects")
    result = BranchResult(issues=output.issues)
    for item in output.items:
        required_aspect = {
            "missing_ablation": "isolation",
            "missing_statistic": "stability",
            "small_gap_without_statistics": "stability",
            "large_gap_no_variance": "stability",
            "text_table_contradiction": "consistency",
        }.get(item.kind)
        if required_aspect and item.aspect != required_aspect:
            raise ValueError("Experimental concern is assigned to the wrong aspect")
        pointer = _paper_pointer(materials, item.block_id, item.quote)
        covered = _covered(claim, item.covered)
        contrary = item.kind != "paper_support"
        decisive = item.kind != "large_gap_no_variance"
        overturnable = True
        if item.kind == "text_table_contradiction":
            if len(item.comparison) != 2 or len(covered) != 1:
                raise ValueError("Text-table contradiction needs two values for one condition")
            left, right = [_number(materials, number) for number in item.comparison]
            condition = next(c for c in claim.conditions if c.id == covered[0])
            for quoted in item.comparison:
                if not _has_label(condition.metric, quoted.quote):
                    raise ValueError("Contradictory passages must identify the same metric")
                if condition.dataset and not _has_label(condition.dataset, quoted.quote):
                    raise ValueError("Contradictory passages must identify the same dataset")
            if left == right:
                raise ValueError("Equal paper values cannot establish a text-table contradiction")
            # Different numbers alone can have rounding/protocol explanations;
            # preserve both pointers for the author to clarify the claim.
            detail = (
                item.detail + "; compared " + "; ".join(f"{n.block_id}: {n.quote}" for n in item.comparison)
            )
        else:
            detail = item.detail
        result.evidence.append(
            Evidence(
                source="paper_internal",
                pointer=pointer,
                covered=covered,
                direction="flaw" if contrary else "support",
                sufficient=decisive,
                concern=contrary and decisive,
                affects_claim=decisive,
                overturnable=overturnable,
                note=f"{item.aspect}/{item.kind}: {detail}",
            )
        )
        if contrary and decisive:
            result.questions.append(
                AuthorQuestion(
                    claim_id=claim.id, text=f"Could you clarify the {item.aspect} concern?", reason=detail
                )
            )
    result.plans = [_plan(claim, materials, candidate) for candidate in output.plans]
    return result
