from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .codex_auth import CodexAuth
from .diagnostics import redact_provider_details
from .provider_capabilities import is_codex_provider as _is_codex_provider

_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
_DEFAULT_MODEL = "gpt-5.5"
_DEFAULT_INSTRUCTIONS = Path(__file__).resolve().parent / "providers" / "codex_instructions.txt"


class CodexResponseError(RuntimeError):
    """A failed request retaining any validated usage already reported by Codex."""

    def __init__(self, message: str, *, usage: dict[str, int] | None = None):
        super().__init__(message)
        self.usage = dict(usage or {})


def is_codex_provider(provider: str | None) -> bool:
    return _is_codex_provider(provider)


def resolve_codex_model(explicit_model: str = "") -> str:
    candidate = str(explicit_model or "").strip()
    if candidate:
        return candidate
    return _DEFAULT_MODEL


def resolve_codex_base_url(explicit_base_url: str = "") -> str:
    return str(explicit_base_url or "").strip() or _DEFAULT_BASE_URL


def codex_headers(auth: CodexAuth) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {auth.access_token}",
        "Content-Type": "application/json",
    }
    if auth.account_id:
        headers["ChatGPT-Account-Id"] = auth.account_id
    return headers


def load_codex_instructions() -> str:
    try:
        return _DEFAULT_INSTRUCTIONS.read_text(encoding="utf-8").strip()
    except Exception:
        return "You are Codex, based on GPT-5. You are running as a coding agent on a user's computer."


def _to_input_messages(system: str, prompt: str, image_data: list[str] | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if (system or "").strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system.strip()}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}]
            + [{"type": "input_image", "image_url": url, "detail": "high"} for url in (image_data or [])],
        }
    )
    return messages


def _extract_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if content.get("type") in {"output_text", "text"} and isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _iter_sse_data(response) -> list[str]:
    events: list[str] = []
    for raw_line in response:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="ignore").strip()
        else:
            line = str(raw_line).strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        if data:
            events.append(data)
    return events


def _coerce_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts = {}
    for key, alias in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        raw = value.get(key, value.get(alias))
        if raw is None:
            continue
        try:
            count = int(raw)
        except (ValueError, TypeError, OverflowError):
            return {}
        if isinstance(raw, bool) or count < 0 or (isinstance(raw, float) and raw != count):
            return {}
        counts[key] = count
    if not counts:
        return {}
    input_count = counts.get("input_tokens", 0)
    output_count = counts.get("output_tokens", 0)
    total_count = counts.get("total_tokens", 0)
    if total_count <= 0:
        total_count = input_count + output_count
    return {
        "input_tokens": input_count,
        "output_tokens": output_count,
        "total_tokens": total_count,
    }


def _extract_usage(payload: dict[str, Any]) -> dict[str, int]:
    direct = _coerce_usage(payload.get("usage"))
    if direct:
        return direct
    response = payload.get("response")
    if isinstance(response, dict):
        nested = _coerce_usage(response.get("usage"))
        if nested:
            return nested
    item = payload.get("item")
    if isinstance(item, dict):
        item_usage = _coerce_usage(item.get("usage"))
        if item_usage:
            return item_usage
    return {}


def invoke_codex(
    prompt: str,
    system: str,
    *,
    auth: CodexAuth,
    model: str,
    base_url: str,
    return_usage: bool = False,
    image_data: list[str] | None = None,
) -> str | tuple[str, dict[str, int]]:
    url = resolve_codex_base_url(base_url).rstrip("/") + "/responses"
    payload = {
        "model": resolve_codex_model(model),
        "input": _to_input_messages(system=system, prompt=prompt, image_data=image_data),
        "instructions": load_codex_instructions(),
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "reasoning": {"summary": "auto"},
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    for key, value in codex_headers(auth).items():
        request.add_header(key, value)
    request.add_header("Accept", "text/event-stream")
    request.add_header("User-Agent", "factreview/execution")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            events = _iter_sse_data(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        detail = redact_provider_details(detail, base_url=url, secrets=(auth.access_token,))
        raise CodexResponseError(f"Codex backend HTTP {exc.code}: {detail[:2000]}") from None
    except Exception as exc:
        detail = redact_provider_details(
            f"{type(exc).__name__}: {exc}", base_url=url, secrets=(auth.access_token,)
        )
        raise CodexResponseError(f"Codex request failed: {detail}") from None

    chunks: list[str] = []
    last_payload: dict[str, Any] = {}
    usage: dict[str, int] = {}
    completed = False
    for event in events:
        try:
            payload_item = json.loads(event)
        except Exception:
            continue
        if not isinstance(payload_item, dict):
            continue
        last_payload = payload_item
        event_usage = _extract_usage(payload_item)
        if event_usage:
            usage = event_usage
        event_type = payload_item.get("type")
        response_payload = payload_item.get("response")
        response_status = response_payload.get("status") if isinstance(response_payload, dict) else None
        if event_type in {"error", "response.failed", "response.incomplete", "response.cancelled"} or (
            response_status in {"failed", "incomplete", "cancelled"}
        ):
            raise CodexResponseError(
                f"Codex backend did not complete the response: {event_type or response_status}", usage=usage
            )
        if event_type == "response.completed":
            if response_status not in {None, "completed"}:
                raise CodexResponseError(
                    f"Codex backend returned an invalid completion status: {response_status}", usage=usage
                )
            completed = True
            if isinstance(response_payload, dict):
                final_text = _extract_output_text(response_payload)
                if final_text:
                    chunks = [final_text]
            continue
        if event_type == "response.output_text.delta" and isinstance(payload_item.get("delta"), str):
            chunks.append(payload_item["delta"])
            continue
        if (
            event_type == "response.output_text.done"
            and isinstance(payload_item.get("text"), str)
            and not chunks
        ):
            chunks.append(payload_item["text"])
            continue

        fallback_text = _extract_output_text(payload_item)
        if fallback_text and not chunks:
            chunks.append(fallback_text)

    if not completed:
        raise CodexResponseError("Codex backend stream ended before response.completed", usage=usage)
    text = "".join(chunks).strip()
    if text:
        if return_usage:
            return text, usage
        return text
    raise CodexResponseError(
        f"Codex backend returned no text. Response keys: {sorted(last_payload.keys())}", usage=usage
    )
