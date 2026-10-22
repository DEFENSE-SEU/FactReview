"""Recorded identity of explicitly reviewed, multi-file Code evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr


class CodeJointMember(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pointer_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class CodeJointBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["code-joint-v1"] = "code-joint-v1"
    claim_id: StrictStr = Field(min_length=1)
    condition_id: StrictStr = Field(min_length=1)
    candidate_index: int = Field(ge=0, strict=True)
    target_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    scope_audit_pointer: StrictStr = Field(min_length=1)
    scope_audit_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    members: list[CodeJointMember] = Field(min_length=2, max_length=9)
