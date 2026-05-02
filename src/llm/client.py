from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .codex_auth import get_codex_auth
from .codex_client import (
    invoke_codex,
    resolve_codex_base_url,
    resolve_codex_model,
)
from .provider_capabilities import is_codex_provider, normalize_provider


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    temperature: float = 0.1
    max_tokens: int = 1500


def _resolve_openai_codex_model(explicit_model: str = "") -> str:
    candidate = (os.getenv("OPENAI_CODEX_MODEL") or "").strip()
    if candidate:
        return candidate
    candidate = (os.getenv("EXECUTION_OPENAI_MODEL") or "").strip()
    if candidate:
        return candidate
    return resolve_codex_model(explicit_model)


def _resolve_provider(explicit_provider: str = "") -> str:
    for candidate in (
        explicit_provider,
        os.getenv("EXECUTION_MODEL_PROVIDER"),
        os.getenv("MODEL_PROVIDER"),
        os.getenv("AGENT_MODEL_PROVIDER"),
    ):
        normalized = normalize_provider(str(candidate or ""), default="")
        if normalized:
            return normalized
    return "openai-codex"


def resolve_llm_config(provider: str = "", model: str = "", base_url: str = "") -> LLMConfig:
    prov = _resolve_provider(provider)

    if prov == "deepseek":
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip() or None
        base = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        mdl = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return LLMConfig(provider=prov, model=mdl, base_url=base, api_key=api_key)

    if prov == "qwen":
        api_key = (os.getenv("QWEN_API_KEY") or "").strip() or None
        base = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        mdl = model or os.getenv("QWEN_MODEL", "qwen-3")
        return LLMConfig(provider=prov, model=mdl, base_url=base, api_key=api_key)

    if prov == "claude":
        api_key = (os.getenv("CLAUDE_API_KEY") or "").strip() or None
        base = base_url or os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com")
        mdl = model or os.getenv("CLAUDE_MODEL", "claude-4-sonnet")
        return LLMConfig(provider=prov, model=mdl, base_url=base, api_key=api_key)

    if is_codex_provider(prov):
        return LLMConfig(
            provider="openai-codex",
            model=model or _resolve_openai_codex_model(),
            base_url=resolve_codex_base_url(base_url or os.getenv("OPENAI_CODEX_BASE_URL", "")),
            api_key=None,
        )

    api_key = (
        os.getenv("EXECUTION_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
    ).strip() or None
    if api_key:
        return LLMConfig(
            provider="openai",
            model=model or os.getenv("EXECUTION_OPENAI_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5"),
            base_url=base_url
            or os.getenv("EXECUTION_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=api_key,
        )

    # No API key: automatically fall back to the Codex subscription backend.
    return LLMConfig(
        provider="openai-codex",
        model=_resolve_openai_codex_model(model),
        base_url=resolve_codex_base_url(base_url or os.getenv("OPENAI_CODEX_BASE_URL", "")),
        api_key=None,
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text or "")
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"status": "unknown", "raw": text}


def llm_json(
    prompt: str,
    system: str,
    cfg: LLMConfig,
) -> dict[str, Any]:
    """
    Minimal JSON response helper for the providers used in the execution stage.
    """
    try:
        if cfg.provider == "claude":
            from anthropic import Anthropic

            client = Anthropic(api_key=cfg.api_key)
            resp = client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            try:
                if resp.content and len(resp.content) > 0:
                    text = (resp.content[0].text or "").strip()
            except Exception:
                text = str(resp).strip()
        elif cfg.provider == "openai-codex":
            auth = get_codex_auth(allow_browser_login=True)
            text = invoke_codex(
                prompt=prompt,
                system=system,
                auth=auth,
                model=cfg.model,
                base_url=cfg.base_url or "https://chatgpt.com/backend-api/codex",
            )
        else:
            from openai import OpenAI

            client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "provider": cfg.provider,
            "model": cfg.model,
            "base_url": cfg.base_url,
        }

    return _parse_json_response(text)
