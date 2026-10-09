"""Operational verification failures, kept distinct from author-artifact findings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class VerificationLimitation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    claim_id: StrictStr = Field(min_length=1)
    condition_ids: list[StrictStr] = Field(min_length=1)
    stage: Literal["Literature", "Theory", "Code", "Experiments", "verification", "execution"]
    kind: Literal["branch_failed", "plan_rejected", "stage_failed", "source_context_unavailable"]
    responsibility: Literal["system"] = "system"
    reason: StrictStr = Field(min_length=1)
    action: Literal["repair_or_retry_verification"] = "repair_or_retry_verification"

    @field_validator("condition_ids")
    @classmethod
    def unique_conditions(cls, value):
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("Verification limitations require distinct nonempty condition IDs")
        return value
