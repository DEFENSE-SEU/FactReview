"""Non-academic recheck downgrades fake errors to warnings."""

from __future__ import annotations

from refcopilot.models import (
    Issue,
    IssueCategory,
    Reference,
    Severity,
    SourceFormat,
)
from refcopilot.verify import non_academic


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, *, prompt, system, **kw):
        self.calls.append({"prompt": prompt, "system": system})
        return self.payload


def _ref(**kw):
    defaults = dict(raw="x", source_format=SourceFormat.BIBTEX)
    defaults.update(kw)
    return Reference(**defaults)


def _fake_issue() -> Issue:
    return Issue(
        severity=Severity.ERROR,
        category=IssueCategory.FAKE,
        code="title_mismatch",
        message="Cited title does not match any retrieved record.",
    )


def test_recheck_downgrades_system_card(monkeypatch):
    stub = _StubLLM(
        {"is_non_academic": True, "citation_type": "system_card",
         "reasoning": "Anthropic system card published as a vendor PDF."}
    )
    monkeypatch.setattr(non_academic, "call_json", stub)

    out = non_academic.recheck(
        _ref(title="Claude opus 4.6 system card", authors=["Anthropic"]),
        matches=[],
        original_issue=_fake_issue(),
    )
    assert out.severity == Severity.WARNING
    assert out.category == IssueCategory.NON_ACADEMIC
    assert out.code.startswith("non_academic::system_card")


def test_recheck_downgrades_blog_post(monkeypatch):
    stub = _StubLLM(
        {"is_non_academic": True, "citation_type": "blog_post", "reasoning": "OpenAI announcement."}
    )
    monkeypatch.setattr(non_academic, "call_json", stub)

    out = non_academic.recheck(
        _ref(title="Introducing gpt-5.4", authors=["OpenAI"]),
        matches=[],
        original_issue=_fake_issue(),
    )
    assert out.severity == Severity.WARNING
    assert out.code == "non_academic::blog_post"


def test_recheck_keeps_error_when_llm_says_no(monkeypatch):
    stub = _StubLLM({"is_non_academic": False, "reasoning": "Looks fabricated."})
    monkeypatch.setattr(non_academic, "call_json", stub)

    original = _fake_issue()
    out = non_academic.recheck(_ref(title="Quantum Bagels"), matches=[], original_issue=original)
    assert out is original
    assert out.severity == Severity.ERROR
    assert out.category == IssueCategory.FAKE


def test_recheck_keeps_error_on_llm_failure(monkeypatch):
    stub = _StubLLM({"status": "error", "error": "TimeoutError"})
    monkeypatch.setattr(non_academic, "call_json", stub)

    original = _fake_issue()
    out = non_academic.recheck(_ref(title="x"), matches=[], original_issue=original)
    assert out is original


def test_recheck_keeps_error_on_malformed_payload(monkeypatch):
    stub = _StubLLM({"foo": "bar"})  # no is_non_academic key
    monkeypatch.setattr(non_academic, "call_json", stub)

    original = _fake_issue()
    out = non_academic.recheck(_ref(title="x"), matches=[], original_issue=original)
    assert out is original
