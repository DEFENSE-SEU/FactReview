"""Execution policy and preserved metric tolerance profiles."""

from typing import Annotated, Any, Literal

from pydantic import Field

from schemas.claim import Contract, FiniteNumber, NonEmpty

# Existing plan generation used .02 for MRR; the execution judge used .01.
# Keep both historical contracts explicit. V2 evidence uses alignment.
TOLERANCES = {
    "legacy_plan": {"mr": 30.0, "rate": 0.02, "percentage": 2.0, "absolute": 0.02, "relative": 0.05},
    "alignment": {"mr": 30.0, "mrr": 0.01, "rate": 0.02, "percentage": 2.0, "absolute": 0.05},
}


def metric_tolerance(metric: str, expected: Any = 0, *, profile="alignment", delta=None) -> float:
    values = TOLERANCES[profile]
    key = str(metric or "").strip().lower()
    try:
        magnitude = abs(float(expected))
    except (ValueError, TypeError):
        magnitude = 0.0
    if key == "mr":
        return values["mr"]
    if key == "mrr" and profile == "alignment":
        return values["mrr"]
    rates = {"mrr", "accuracy", "acc", "f1", "precision", "recall", "auc"}
    if profile == "alignment":
        rates = (rates - {"acc"}) | {"map", "ndcg"}
    if key.startswith("hits@") or key in rates:
        return values["percentage"] if profile == "legacy_plan" and magnitude > 1 else values["rate"]
    if key in {"bleu", "rouge-l", "rouge-1", "rouge-2"}:
        # Preserve the legacy alignment convention, including its delta-based scale.
        scale = magnitude if profile == "legacy_plan" else abs(float(delta or 0))
        return values["percentage"] if scale > 1 else values["rate"]
    if profile == "legacy_plan":
        return max(values["absolute"], magnitude * values["relative"])
    return values["absolute"]


class OutputMapping(Contract):
    """JSON selectors frozen before execution; every value comes from runtime output."""

    root_path: list[str | int] = Field(default_factory=list)
    dataset_path: list[str | int] = Field(default_factory=lambda: ["dataset"])
    settings_path: list[str | int] | None = Field(default_factory=lambda: ["settings"])
    settings_paths: dict[str, list[str | int]] = Field(default_factory=dict)
    metric_paths: dict[str, list[str | int]] = Field(default_factory=dict)


class ReleasedArtifactContract(Contract):
    """Operator-approved role and selectors, supplied before any run output exists."""

    artifact_path: NonEmpty
    artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    artifact_kind: Literal["data", "logs"]
    values_key: NonEmpty
    dataset_key: NonEmpty
    metric_key: NonEmpty
    settings_key: NonEmpty
    operation: Literal["mean", "sum", "min", "max", "identity"]


class PaperVariance(Contract):
    value: Annotated[FiniteNumber, Field(ge=0)]
    block_id: NonEmpty
    quote: NonEmpty


class ExecutionConfig(Contract):
    max_attempts: int = Field(default=3, ge=0, le=3, description="Maximum accepted repair rounds")
    approval_mode: Literal["auto", "interactive"] = "auto"
    training_budget: int = Field(default=0, ge=0, description="Maximum training runs, including retries")
    timeout_seconds: int = Field(default=3600, gt=0)
    python_version: str = "3.11"
    docker_build_timeout_seconds: int = Field(default=3600, gt=0)
    docker_options: dict[str, Any] = Field(default_factory=dict)
    refine_with_llm: bool = True
    output_mappings: dict[str, OutputMapping] = Field(default_factory=dict)
    author_artifacts: dict[str, list[ReleasedArtifactContract]] = Field(default_factory=dict)
    paper_variances: dict[str, dict[str, PaperVariance]] = Field(default_factory=dict)
    tolerance_overrides: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        default_factory=dict
    )
