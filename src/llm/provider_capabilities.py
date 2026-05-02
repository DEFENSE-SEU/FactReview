from __future__ import annotations

from dataclasses import dataclass

_CODEX_PROVIDER_ALIASES = {
    "openai-codex",
    "codex",
    "chatgpt-oauth",
    "chatgpt",
}


@dataclass(frozen=True)
class LLMProviderCapabilities:
    provider: str
    uses_codex_subscription: bool = False
    requires_streaming: bool = False
    supports_max_output_tokens: bool = True
    supports_parallel_tool_calls: bool = True
    supports_response_include: bool = True
    supports_store: bool = True


def normalize_provider(provider: str | None, *, default: str = "openai-codex") -> str:
    normalized = str(provider or "").strip().lower().replace("_", "-")
    return normalized or default


def is_codex_provider(provider: str | None) -> bool:
    return normalize_provider(provider, default="") in _CODEX_PROVIDER_ALIASES


def provider_capabilities(provider: str | None) -> LLMProviderCapabilities:
    normalized = normalize_provider(provider)
    if is_codex_provider(normalized):
        return LLMProviderCapabilities(
            provider="openai-codex",
            uses_codex_subscription=True,
            requires_streaming=True,
            supports_max_output_tokens=False,
            supports_parallel_tool_calls=False,
            supports_response_include=True,
            supports_store=False,
        )
    return LLMProviderCapabilities(provider=normalized)
