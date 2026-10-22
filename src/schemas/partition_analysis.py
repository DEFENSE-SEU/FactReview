"""Untrusted source selections for complete released-partition analysis."""
from typing import Annotated, Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from schemas.claim import Contract
from schemas.runtime_science import ROLES, ObligationDecision, PaperDefinition, SourceDefinition

VERSION = "released-partition-analysis-v1"
Selector = Annotated[list[StrictStr | StrictInt], Field(min_length=1, max_length=12)]


class PartitionAnalysisProposal(Contract):
    version: Literal["released-partition-analysis-v1"]
    artifact_path: str = Field(min_length=1)
    partition_selector: Selector
    label_selector: Selector | None = None
    prediction_selector: Selector | None = None
    rank_selector: Selector | None = None
    config_settings_selector: dict[StrictStr, Selector] = Field(default_factory=dict)
    dataset_selector: Selector
    model_selector: Selector
    metric_selector: Selector
    metric_definition: Literal["exact_match_fraction", "mean_squared_error", "mean_reciprocal_rank"]
    paper: dict[str, PaperDefinition]
    sources: dict[str, SourceDefinition]

    @field_validator("partition_selector", "label_selector", "prediction_selector",
                     "dataset_selector", "model_selector", "metric_selector", "rank_selector")
    @classmethod
    def bounded_selector(cls, values):
        if values is None:
            return values
        if any((type(key) is str and (not key or len(key) > 128)) or
               (type(key) is int and not 0 <= key < 65536) for key in values):
            raise ValueError("unsafe_analysis_selector")
        return values

    @model_validator(mode="after")
    def selected_metric_inputs(self):
        rank = self.metric_definition == "mean_reciprocal_rank"
        if rank != (self.rank_selector is not None) or (rank and (
            self.label_selector is not None or self.prediction_selector is not None
        )) or (not rank and (self.label_selector is None or self.prediction_selector is None)):
            raise ValueError("analysis_metric_input_selectors_conflict")
        if len(self.config_settings_selector) > 24 or any(
            not key or len(key) > 128 for key in self.config_settings_selector
        ):
            raise ValueError("unsafe_analysis_setting_selector")
        for selector in self.config_settings_selector.values():
            self.bounded_selector(selector)
        return self

    @field_validator("paper", "sources")
    @classmethod
    def complete_roles(cls, values):
        if set(values) != set(ROLES):
            raise ValueError("all_analysis_source_roles_required")
        return values


class PartitionRuntimeProposal(Contract):
    """Untrusted selections; measured authority remains the complete host partition."""

    version: Literal["partition-runtime-v1"]
    analysis: PartitionAnalysisProposal
    raw_value_selector: Selector

    @field_validator("raw_value_selector")
    @classmethod
    def bounded_runtime_selector(cls, values):
        return PartitionAnalysisProposal.bounded_selector(values)


class PartitionAnalysisReview(Contract):
    version: Literal["analysis-v1"]
    context_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    condition_id: str = Field(min_length=1)
    measurement_scope: Literal["released_statistics", "requires_inference", "unresolved"]
    obligations: list[ObligationDecision] = Field(min_length=1)
    unresolved: list[str]


class CandidatePaperDefinition(Contract):
    block_id: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=65536)


class CandidateSourceDefinition(Contract):
    path: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=65536)


class CandidatePaperRoles(Contract):
    dataset: CandidatePaperDefinition
    split: CandidatePaperDefinition
    model: CandidatePaperDefinition
    metric: CandidatePaperDefinition
    population: CandidatePaperDefinition
    qualifiers: CandidatePaperDefinition


class CandidateSourceRoles(Contract):
    dataset: CandidateSourceDefinition
    split: CandidateSourceDefinition
    model: CandidateSourceDefinition
    metric: CandidateSourceDefinition
    population: CandidateSourceDefinition
    qualifiers: CandidateSourceDefinition


class PartitionAnalysisCandidate(Contract):
    version: Literal["released-partition-analysis-candidate-v2"]
    artifact_path: str = Field(min_length=1)
    partition_selector: Selector
    label_selector: Selector | None = None
    prediction_selector: Selector | None = None
    rank_selector: Selector | None = None
    config_settings_selector: dict[StrictStr, Selector] = Field(default_factory=dict)
    dataset_selector: Selector
    model_selector: Selector
    metric_selector: Selector
    metric_definition: Literal["exact_match_fraction", "mean_squared_error", "mean_reciprocal_rank"]
    paper: CandidatePaperRoles
    sources: CandidateSourceRoles

    @field_validator("partition_selector", "label_selector", "prediction_selector",
                     "dataset_selector", "model_selector", "metric_selector", "rank_selector")
    @classmethod
    def bounded_selector(cls, values):
        return PartitionAnalysisProposal.bounded_selector(values)

    @model_validator(mode="after")
    def selected_metric_inputs(self):
        # Apply the same selector contract without deriving source semantics.
        PartitionAnalysisProposal.selected_metric_inputs(self)
        return self


class PartitionRuntimeCandidate(Contract):
    """Untrusted quote choices; localization supplies positions, never semantics."""

    version: Literal["partition-runtime-candidate-v2"]
    analysis: PartitionAnalysisCandidate
    raw_value_selector: Selector

    @field_validator("raw_value_selector")
    @classmethod
    def bounded_runtime_selector(cls, values):
        return PartitionAnalysisProposal.bounded_selector(values)
