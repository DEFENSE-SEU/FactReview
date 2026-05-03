"""Shared test fixtures for RefCopilot.

Highlights:
  - `fixtures_dir` resolves to tests/fixtures/.
  - `mock_llm_json` monkey-patches refcopilot.extract.llm_extractor.call_json
    so unit tests don't need a real LLM.
  - `mock_arxiv_backend` and `mock_s2_backend` provide minimal in-memory
    search backends with fixed canned responses.
"""

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
    """Replace refcopilot.extract.llm_extractor.call_json with a canned-response stub.

    Tests should set responses with `mock_llm_json.set(payload_dict)` or
    `mock_llm_json.set_sequence([dict, dict, ...])`.
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


@pytest.fixture
def mock_arxiv_backend():
    from refcopilot.models import Backend, ExternalRecord

    class _Stub:
        def __init__(self) -> None:
            self._by_id: dict[str, ExternalRecord] = {}
            self._by_title: dict[str, ExternalRecord] = {}
            self.calls: list[dict[str, Any]] = []

        def add(self, record: ExternalRecord) -> None:
            self._by_id[record.record_id] = record
            if record.title:
                self._by_title[record.title.lower()] = record

        def lookup(self, *, arxiv_id=None, title=None, **kw):
            self.calls.append({"arxiv_id": arxiv_id, "title": title})
            if arxiv_id and arxiv_id in self._by_id:
                return [self._by_id[arxiv_id]]
            if title and title.lower() in self._by_title:
                return [self._by_title[title.lower()]]
            return []

        @property
        def name(self) -> str:
            return Backend.ARXIV.value

    return _Stub()


@pytest.fixture
def mock_s2_backend():
    from refcopilot.models import Backend, ExternalRecord

    class _Stub:
        def __init__(self) -> None:
            self._records: list[ExternalRecord] = []
            self.calls: list[dict[str, Any]] = []
            self.fail_with_429 = False

        def add(self, record: ExternalRecord) -> None:
            self._records.append(record)

        def lookup(self, *, doi=None, arxiv_id=None, title=None, year=None, authors=None, **kw):
            self.calls.append(
                {"doi": doi, "arxiv_id": arxiv_id, "title": title, "year": year, "authors": authors}
            )
            if doi:
                return [r for r in self._records if r.doi == doi]
            if arxiv_id:
                return [r for r in self._records if r.arxiv_id == arxiv_id]
            if title:
                t = title.lower()
                return [r for r in self._records if t in r.title.lower() or r.title.lower() in t]
            return []

        @property
        def name(self) -> str:
            return Backend.SEMANTIC_SCHOLAR.value

    return _Stub()
