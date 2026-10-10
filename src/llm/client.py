from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from common import run_stats

from .codex_auth import get_codex_auth
from .codex_client import (
    CodexResponseError,
    invoke_codex,
    resolve_codex_base_url,
    resolve_codex_model,
)
from .diagnostics import redact_provider_details
from .provider_capabilities import is_codex_provider, normalize_provider


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    temperature: float = 0.1
    max_tokens: int | None = None


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


def _resolve_max_tokens(max_tokens: int | None) -> int | None:
    value = int(max_tokens or 0)
    return value if value > 0 else None


def resolve_llm_config(
    provider: str = "",
    model: str = "",
    base_url: str = "",
    max_tokens: int | None = None,
) -> LLMConfig:
    resolved_max_tokens = _resolve_max_tokens(max_tokens)
    prov = _resolve_provider(provider)

    if prov == "deepseek":
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip() or None
        base = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        mdl = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return LLMConfig(
            provider=prov, model=mdl, base_url=base, api_key=api_key, max_tokens=resolved_max_tokens
        )

    if prov == "qwen":
        api_key = (os.getenv("QWEN_API_KEY") or "").strip() or None
        base = base_url or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        mdl = model or os.getenv("QWEN_MODEL", "qwen-3")
        return LLMConfig(
            provider=prov, model=mdl, base_url=base, api_key=api_key, max_tokens=resolved_max_tokens
        )

    if prov == "claude":
        api_key = (os.getenv("CLAUDE_API_KEY") or "").strip() or None
        base = base_url or os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com")
        mdl = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        return LLMConfig(
            provider=prov, model=mdl, base_url=base, api_key=api_key, max_tokens=resolved_max_tokens
        )

    if is_codex_provider(prov):
        return LLMConfig(
            provider="openai-codex",
            model=model or _resolve_openai_codex_model(),
            base_url=resolve_codex_base_url(base_url or os.getenv("OPENAI_CODEX_BASE_URL", "")),
            api_key=None,
            max_tokens=resolved_max_tokens,
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
            max_tokens=resolved_max_tokens,
        )

    # No API key: automatically fall back to the Codex subscription backend.
    return LLMConfig(
        provider="openai-codex",
        model=_resolve_openai_codex_model(model),
        base_url=resolve_codex_base_url(base_url or os.getenv("OPENAI_CODEX_BASE_URL", "")),
        api_key=None,
        max_tokens=resolved_max_tokens,
    )


def resolve_vlm_config(*, fallback: LLMConfig | None = None) -> LLMConfig:
    """Resolve optional visual-model overrides without silently changing providers.

    With no visual overrides the regular model configuration is inherited.
    A different visual provider uses its own credentials and defaults; a key
    belonging to the main provider must not be sent to another provider.
    """
    provider = (os.getenv("VLM_MODEL_PROVIDER") or "").strip()
    model = (os.getenv("VLM_MODEL") or "").strip()
    base_url = (os.getenv("VLM_BASE_URL") or "").strip()
    api_key = (os.getenv("VLM_API_KEY") or "").strip()
    if not any((provider, model, base_url, api_key)) and fallback is not None:
        return fallback
    requested = _resolve_provider(provider or (fallback.provider if fallback else ""))
    if is_codex_provider(requested):
        requested = "openai-codex"
    if requested not in {"openai-codex", "openai", "claude", "deepseek", "qwen"}:
        raise ValueError(f"unsupported visual model provider: {requested}")

    main_config = fallback or resolve_llm_config()
    fallback_provider = normalize_provider(main_config.provider)
    if is_codex_provider(fallback_provider):
        fallback_provider = "openai-codex"
    inherited = (
        main_config
        if not provider or requested == fallback_provider
        else resolve_llm_config(provider=provider)
    )
    if not any((provider, model, base_url, api_key)):
        return inherited
    if requested == "openai-codex":
        if api_key:
            raise ValueError("The openai-codex visual provider uses Codex login; VLM_API_KEY is unsupported")
        return replace(inherited, model=model or inherited.model, base_url=base_url or inherited.base_url)

    # Execution settings may describe the Codex backend. A separate OpenAI
    # visual provider must use OpenAI's own connection and credentials.
    # Preserve the legacy resolver's priorities when sharing the main provider.
    if requested == "openai" and (requested != fallback_provider or inherited.provider != "openai"):
        inherited = LLMConfig(
            provider="openai",
            model=(os.getenv("OPENAI_MODEL") or "").strip() or "gpt-5",
            base_url=(os.getenv("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1",
            api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        )
    resolved = replace(
        inherited,
        model=model or inherited.model,
        base_url=base_url or inherited.base_url,
        api_key=api_key or inherited.api_key,
    )
    if not resolved.api_key:
        raise ValueError(f"Visual model provider {requested} requires VLM_API_KEY or its provider API key")
    return resolved


def _parse_json_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"status": "unknown", "raw": text}
    except Exception:
        pass

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    pos = 0
    raw_text = text or ""
    while pos < len(raw_text):
        start = raw_text.find("{", pos)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(raw_text[start:])
        except Exception:
            pos = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        pos = start + max(end, 1)

    for obj in objects:
        if isinstance(obj.get("tasks"), list):
            return obj
    for obj in objects:
        if obj:
            return obj

    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"status": "unknown", "raw": text}


def llm_json(
    prompt: str,
    system: str,
    cfg: LLMConfig,
    *,
    module: str | None = None,
    images: list[str] | None = None,
) -> dict[str, Any]:
    """
    Minimal JSON response helper for the providers used in the execution stage.
    """
    if run_stats.stats_path() is not None:
        module = run_stats.validate_module(module or run_stats.current_module() or "")
        run_stats.read_initialized()
    t0 = time.monotonic()
    usage: dict[str, Any] = {}
    text = ""
    retry_failure_recorded = False
    try:
        encoded_images = []
        for image_path in images or []:
            mime = mimetypes.guess_type(image_path)[0]
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ValueError(f"unsupported image format: {image_path}")
            encoded_images.append((mime, base64.b64encode(Path(image_path).read_bytes()).decode("ascii")))
        if cfg.provider == "claude":
            from anthropic import Anthropic

            client = Anthropic(api_key=cfg.api_key, base_url=cfg.base_url)
            resp = client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens or 8192,
                temperature=cfg.temperature,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": prompt}]
                            + [
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": mime, "data": data},
                                }
                                for mime, data in encoded_images
                            ]
                            if encoded_images
                            else prompt
                        ),
                    }
                ],
            )
            text = ""
            try:
                if resp.content and len(resp.content) > 0:
                    text = (resp.content[0].text or "").strip()
            except Exception:
                text = str(resp).strip()
            raw_usage = getattr(resp, "usage", None)
            usage = (
                {
                    "input_tokens": int(getattr(raw_usage, "input_tokens", 0) or 0),
                    "output_tokens": int(getattr(raw_usage, "output_tokens", 0) or 0),
                }
                if raw_usage is not None
                else {}
            )
            if usage:
                usage["total_tokens"] = int(usage["input_tokens"]) + int(usage["output_tokens"])
        elif cfg.provider == "openai-codex":
            auth = get_codex_auth(allow_browser_login=True)
            for attempt in range(2):
                try:
                    codex_result = invoke_codex(
                        prompt=prompt,
                        system=system,
                        auth=auth,
                        model=cfg.model,
                        base_url=cfg.base_url or "https://chatgpt.com/backend-api/codex",
                        return_usage=True,
                        **(
                            {"image_data": [f"data:{mime};base64,{data}" for mime, data in encoded_images]}
                            if encoded_images
                            else {}
                        ),
                    )
                    break
                except CodexResponseError as exc:
                    if attempt != 0 or not exc.pre_sse_transient:
                        raise
                    # Persist the failed physical request before another POST.
                    # If recording/backoff fails, no new request has begun.
                    retry_failure_recorded = True
                    if run_stats.stats_path() is not None:
                        run_stats.record_llm_call(
                            module=module,
                            provider=cfg.provider,
                            model=cfg.model,
                            usage=exc.usage,
                            duration_sec=time.monotonic() - t0,
                            failed=True,
                            image_count=len(images or []),
                            warning="Codex transport attempt 1 failed before SSE with EOF/reset; "
                            f"{redact_provider_details(str(exc), cfg)}. "
                            "One retry follows; missing usage remains unavailable.",
                        )
                    time.sleep(1.0)
                    t0 = time.monotonic()
                    usage = {}
                    retry_failure_recorded = False
            if isinstance(codex_result, tuple):
                text, usage = codex_result
            else:
                text = codex_result
        else:
            from openai import OpenAI

            client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, max_retries=0)
            kwargs: dict[str, Any] = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": prompt}]
                            + [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{data}", "detail": "high"},
                                }
                                for mime, data in encoded_images
                            ]
                            if encoded_images
                            else prompt
                        ),
                    },
                ],
                "temperature": cfg.temperature,
            }
            if cfg.max_tokens is not None:
                kwargs["max_tokens"] = cfg.max_tokens
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            raw_usage = getattr(resp, "usage", None)
            usage = (
                {
                    "input_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                    "output_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(raw_usage, "total_tokens", 0) or 0),
                }
                if raw_usage is not None
                else {}
            )
    except Exception as e:
        if isinstance(e, CodexResponseError):
            usage = e.usage
        if not retry_failure_recorded and run_stats.stats_path() is not None:
            run_stats.record_llm_call(
                module=module,
                provider=cfg.provider,
                model=cfg.model,
                usage=usage,
                duration_sec=time.monotonic() - t0,
                failed=True,
                image_count=len(images or []),
            )
        return redact_provider_details(
            {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "provider": cfg.provider,
                "model": cfg.model,
                "base_url": cfg.base_url,
            },
            cfg,
        )

    if run_stats.stats_path() is not None:
        run_stats.record_llm_call(
            module=module,
            provider=cfg.provider,
            model=cfg.model,
            usage=usage,
            prompt=prompt,
            system=system,
            response_text=text,
            duration_sec=time.monotonic() - t0,
            image_count=len(images or []),
        )

    return redact_provider_details(_parse_json_response(text), cfg)
