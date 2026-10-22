"""Located proposals and independent decisions for finite scientific consumption."""
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import Field, StrictInt, StrictStr, field_validator

from schemas.claim import Contract

VERSION = "builtin-source-science-v1"
ROLES = ("dataset", "split", "model", "metric", "population", "qualifiers")
Selector = Annotated[list[StrictStr | StrictInt], Field(min_length=1, max_length=12)]


class Definition(Contract):
    quote: str = Field(min_length=1, max_length=65536)
    start: int = Field(strict=True, ge=0)
    end: int = Field(strict=True, gt=0)


class PaperDefinition(Definition):
    block_id: str = Field(min_length=1)


class SourceDefinition(Definition):
    path: str = Field(min_length=1)


class ScienceProposal(Contract):
    version: Literal["builtin-source-science-v1"]
    dataset_selector: Selector
    partition_selector: Selector
    model_selector: Selector
    metric_selector: Selector
    paper: dict[str, PaperDefinition]
    sources: dict[str, SourceDefinition]

    @field_validator("dataset_selector", "partition_selector", "model_selector", "metric_selector")
    @classmethod
    def bounded_selector(cls, value):
        if any((type(key) is str and (not key or len(key) > 128)) or
               (type(key) is int and not 0 <= key < 256) for key in value):
            raise ValueError("unsafe selector")
        return value

    @field_validator("paper", "sources")
    @classmethod
    def closed_roles(cls, value):
        if set(value) != set(ROLES):
            raise ValueError("all scientific roles are required")
        return value


class ObligationDecision(Contract):
    id: str = Field(min_length=1)
    decision: Literal["confirmed", "unresolved", "contradicted"]
    source_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)


_VersionT = TypeVar("_VersionT", bound=str)


class _ScienceReview(Contract, Generic[_VersionT]):
    version: _VersionT
    context_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    condition_id: str = Field(min_length=1)
    obligations: list[ObligationDecision] = Field(min_length=1)
    unresolved: list[str]


class ScienceReview(_ScienceReview[Literal["builtin-source-science-v1"]]):
    pass


_PaperDefinitionT = TypeVar("_PaperDefinitionT", bound=Contract)
_SourceDefinitionT = TypeVar("_SourceDefinitionT", bound=Contract)


class TorchInferenceProposalBase(Contract, Generic[_VersionT, _PaperDefinitionT, _SourceDefinitionT]):
    """Source selections only; runtime state and measurements are never proposals."""

    version: _VersionT
    condition_id: str = Field(min_length=1)
    checkpoint_path: str = Field(min_length=1)
    state_selector: list[StrictStr | StrictInt] = Field(default_factory=list, max_length=12)
    dataset_path: str = Field(min_length=1)
    codec: Literal["indexed-jsonl-numeric-v1"]
    id_key: str = Field(min_length=1)
    features_key: str = Field(min_length=1)
    label_key: str = Field(min_length=1)
    sites: dict[str, StrictInt]
    sample_id_selector: Selector
    sample_features_selector: Selector
    sample_label_selector: Selector
    batch_ids_selector: Selector
    batch_features_selector: Selector
    batch_labels_selector: Selector
    prediction_argument: str = Field(min_length=1)
    metric_prediction_argument: str = Field(min_length=1)
    metric_label_argument: str = Field(min_length=1)
    raw_value_selector: Selector
    prediction_adapter: Literal["class_indices", "argmax_last_axis", "scalar_regression"]
    metric_adapter: Literal["accuracy", "mean_squared_error"]
    dataset_selector: Selector
    split_selector: Selector
    model_selector: Selector
    metric_selector: Selector
    settings_selectors: dict[str, Selector]
    paper: dict[str, _PaperDefinitionT]
    sources: dict[str, _SourceDefinitionT]

    @field_validator("state_selector", "batch_features_selector", "batch_labels_selector",
                     "sample_id_selector", "sample_features_selector", "sample_label_selector", "batch_ids_selector",
                     "raw_value_selector", "dataset_selector", "split_selector", "model_selector", "metric_selector")
    @classmethod
    def safe_torch_selector(cls, value):
        return ScienceProposal.bounded_selector(value)

    @field_validator("settings_selectors")
    @classmethod
    def safe_setting_selectors(cls, value):
        for selector in value.values():
            ScienceProposal.bounded_selector(selector)
        return value

    @field_validator("sites")
    @classmethod
    def closed_sites(cls, value):
        if set(value) != {"getitem", "model", "prediction", "metric"} or any(
            type(i) is not int or not 0 <= i < 8 for i in value.values()
        ) or len(set(value.values())) != 4:
            raise ValueError("four distinct actual source sites required")
        return value

    @field_validator("paper", "sources")
    @classmethod
    def closed_torch_roles(cls, value):
        if set(value) != set(ROLES):
            raise ValueError("all scientific roles required")
        return value


class TorchInferenceProposal(TorchInferenceProposalBase[
    Literal["torch-cpu-inference-v1"], PaperDefinition, SourceDefinition,
]):
    """Source selections only; runtime state and measurements are never proposals."""


class TorchInferenceReview(_ScienceReview[Literal["torch-cpu-inference-v1"]]):
    pass
