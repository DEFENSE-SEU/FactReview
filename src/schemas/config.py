"""Per-stage configuration models.

The pipeline as a whole is configured through ``.env`` and CLI flags; the
models here are the small structured contracts that individual stages still
take as keyword arguments.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClaimExtractCfg(BaseModel):
    mode: str = "auto"  # auto = strict LLM extraction; llm | heuristic
    decompose_broad_claims: bool = False
    prompt_max_chars: int = 50_000


class LLMCfg(BaseModel):
    provider: str = "openai-codex"
    model: str = ""
    base_url: str = ""
    max_tokens: int | None = None
    route: dict[str, str] = Field(default_factory=dict)
