"""Optional reference suggestions; metadata candidates never assert manuscript error."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReferenceFieldSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    value: str | int | list[str]
    source_field: str
    source_record_index: int = Field(ge=0)
    backend: str
    record_id: str
    url: str
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReferenceCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["metadata_candidate", "unavailable"]
    raw_reference: str
    corrected_bibtex: str = ""
    identity_identifier: str | None = None
    verified_url: str = ""
    raw_result_pointer: str
    records_pointer: str | None = None
    records_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fields: list[ReferenceFieldSource] = Field(default_factory=list)
    reason: str

    @model_validator(mode="after")
    def usable_candidate(self):
        if self.state == "metadata_candidate":
            if not all(
                (
                    self.corrected_bibtex,
                    self.identity_identifier,
                    self.records_pointer,
                    self.records_sha256,
                    self.fields,
                )
            ):
                raise ValueError(
                    "Reference metadata candidate requires complete identity and field provenance"
                )
        elif self.corrected_bibtex or self.fields:
            raise ValueError("Unavailable reference correction cannot carry a usable replacement")
        return self
