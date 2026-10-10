"""Untrusted source selections for complete released-partition analysis."""
from typing import Annotated, Literal

from pydantic import Field, StrictInt, StrictStr, field_validator

from schemas.claim import Contract
from schemas.runtime_science import ROLES, PaperDefinition, SourceDefinition

VERSION = "released-partition-analysis-v1"
Selector = Annotated[list[StrictStr | StrictInt], Field(min_length=1, max_length=12)]


class PartitionAnalysisProposal(Contract):
    version: Literal["released-partition-analysis-v1"]
    artifact_path: str = Field(min_length=1)
    partition_selector: Selector
    label_selector: Selector
    prediction_selector: Selector
    dataset_selector: Selector
    model_selector: Selector
    metric_selector: Selector
    metric_definition: Literal["exact_match_fraction", "mean_squared_error"]
    paper: dict[str, PaperDefinition]
    sources: dict[str, SourceDefinition]

    @field_validator("partition_selector", "label_selector", "prediction_selector",
                     "dataset_selector", "model_selector", "metric_selector")
    @classmethod
    def bounded_selector(cls, values):
        if any((type(key) is str and (not key or len(key) > 128)) or
               (type(key) is int and not 0 <= key < 65536) for key in values):
            raise ValueError("unsafe_analysis_selector")
        return values

    @field_validator("paper", "sources")
    @classmethod
    def complete_roles(cls, values):
        if set(values) != set(ROLES):
            raise ValueError("all_analysis_source_roles_required")
        return values
