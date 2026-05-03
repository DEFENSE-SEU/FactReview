"""Shared test fixtures for RefCopilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_llm_json(monkeypatch):
    """Replace ``refcopilot.extract.llm_extractor.call_json`` with a canned-response stub.

    Tests should set responses with ``mock_llm_json.set(payload)`` or
    ``mock_llm_json.set_sequence([payload, ...])``.
    """

    class _Stub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._next: list[dict[str, Any]] = []

        def __call__(self, prompt: str, system: str, **kwargs) -> dict[str, Any]:
            self.calls.append({"prompt": prompt, "system": system})
            if not self._next:
                return {"references": []}
            return self._next.pop(0)

        def set(self, payload: dict[str, Any]) -> None:
            self._next = [payload]

        def set_sequence(self, payloads: list[dict[str, Any]]) -> None:
            self._next = list(payloads)

        def load_json(self, path: Path) -> None:
            self.set(json.loads(Path(path).read_text(encoding="utf-8")))

    stub = _Stub()
    from refcopilot.extract import llm_extractor

    monkeypatch.setattr(llm_extractor, "call_json", stub)
    return stub
