"""Located proposals and independent decisions for finite scientific consumption."""
from typing import Annotated, Literal

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


class ScienceReview(Contract):
    version: Literal["builtin-source-science-v1"]
    context_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    condition_id: str = Field(min_length=1)
    obligations: list[ObligationDecision] = Field(min_length=1)
    unresolved: list[str]
