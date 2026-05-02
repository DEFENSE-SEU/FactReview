"""Per-stage configuration models.

The pipeline as a whole is configured through ``.env`` and CLI flags; the
models here are the small structured contracts that individual stages still
take as keyword arguments.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClaimExtractCfg(BaseModel):
    mode: str = "auto"  # auto | llm | heuristic
    decompose_broad_claims: bool = True


class LLMCfg(BaseModel):
    provider: str = "openai-codex"
    model: str = ""
    base_url: str = ""
    route: dict[str, str] = Field(default_factory=dict)
