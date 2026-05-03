"""Plain-text input is a passthrough: the LLM extractor handles it directly."""

from __future__ import annotations


def normalize(text: str) -> str:
    return (text or "").strip()
