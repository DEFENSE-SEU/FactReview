"""Model-authored Theory concern reviews, without a formal-proof guarantee."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExactText = Annotated[str, Field(strict=True, min_length=1)]
Index = Annotated[int, Field(strict=True, ge=0)]


class ConcernContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConcernTargetSource(ConcernContract):
    block_id: ExactText
    quote: ExactText


class TheoryConcernDecision(ConcernContract):
    item_index: Index
    condition_id: ExactText
    disposition: Literal["outside_scope", "answerable_concern", "closed_disproof", "unresolved"]
    target_sources: list[ConcernTargetSource]
    trace_step_ids: list[ExactText]
    trace_gap_indices: list[Index]
    scope_reason: ExactText
    resolution: ExactText

    @model_validator(mode="after")
    def exact_identity_and_unique_refs(self):
        if self.condition_id != self.condition_id.strip():
            raise ValueError("Theory concern condition ID must be exact")
        if any(value != value.strip() for value in self.trace_step_ids):
            raise ValueError("Theory concern step IDs must be exact")
        if len(self.trace_step_ids) != len(set(self.trace_step_ids)) or len(self.trace_gap_indices) != len(
            set(self.trace_gap_indices)
        ):
            raise ValueError("Theory concern trace references must be distinct")
        sources = [(source.block_id, source.quote) for source in self.target_sources]
        if len(sources) != len(set(sources)):
            raise ValueError("Theory concern target sources must be distinct")
        if self.disposition != "unresolved" and (
            not self.target_sources or not (self.trace_step_ids or self.trace_gap_indices)
        ):
            raise ValueError("Deciding a Theory concern requires exact target sources and trace references")
        if not self.scope_reason.strip() or not self.resolution.strip():
            raise ValueError("Theory concern reasoning and resolution must be substantive")
        return self


class TheoryConcernOutput(ConcernContract):
    schema_version: Literal["theory-concern-v1"]
    items: list[TheoryConcernDecision]


class GroundedConcernSource(ConcernTargetSource):
    locator: ExactText
    key: str | None = None
    page: int | None = Field(default=None, ge=1)
    block_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class TheoryConcernReview(ConcernContract):
    schema_version: Literal["theory-concern-v1"] = "theory-concern-v1"
    item_index: Index
    condition_id: ExactText
    state: Literal["validated", "invalid", "unavailable"]
    validation_scope: Literal["structure_source_and_model_scope_judgment"] = (
        "structure_source_and_model_scope_judgment"
    )
    decision: TheoryConcernDecision | None = None
    target_sources: list[GroundedConcernSource] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    audit_pointer: str | None = None
    source_hashes: dict[str, str] = Field(default_factory=dict)
    provider: str = ""
    model: str = ""
    transport: Literal["live", "injected"] = "injected"

    @model_validator(mode="after")
    def bound_decision(self):
        if self.condition_id != self.condition_id.strip():
            raise ValueError("Theory concern record condition ID must be exact")
        if self.state == "validated":
            if self.decision is None or (self.item_index, self.condition_id) != (
                self.decision.item_index,
                self.decision.condition_id,
            ):
                raise ValueError("Validated Theory concern record requires its exact pair decision")
        elif self.decision is not None:
            raise ValueError("An invalid/unavailable concern cannot retain an accepted decision")
        return self
