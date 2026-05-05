"""LLM-driven secondary verification."""

from __future__ import annotations

from refcopilot.models import (
    Backend,
    ExternalRecord,
    HallucinationVerdict,
    Reference,
    SourceFormat,
)
from refcopilot.verify import llm_verifier


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[dict] = []

    def __call__(self, *, prompt: str, system: str, **kw) -> dict:
        self.calls.append({"prompt": prompt, "system": system})
        return self.payload


def _ref(**kw):
    defaults = dict(raw="x", source_format=SourceFormat.BIBTEX)
    defaults.update(kw)
    return Reference(**defaults)


def _record(**kw):
    defaults = dict(
        backend=Backend.SEMANTIC_SCHOLAR,
        record_id="x",
        title="X",
        url="https://example.com",
    )
    defaults.update(kw)
    return ExternalRecord(**defaults)


def test_verifier_promotes_uncertain_to_unlikely(monkeypatch):
    stub = _StubLLM({"verdict": "UNLIKELY", "reason": "Real paper, well known."})
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    ref = _ref(title="Attention Is All You Need", authors=["A. V."])
    rec = _record(title="Attention Is All You Need", authors=["A. Vaswani"])

    out = llm_verifier.verify(ref, [rec], initial=HallucinationVerdict.UNCERTAIN)
    assert out.verdict == HallucinationVerdict.UNLIKELY
    assert out.suggestion is None


def test_verifier_can_keep_uncertain(monkeypatch):
    stub = _StubLLM({"verdict": "UNCERTAIN", "reason": "Cannot decide."})
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    out = llm_verifier.verify(_ref(title="x"), [], initial=HallucinationVerdict.UNCERTAIN)
    assert out.verdict == HallucinationVerdict.UNCERTAIN
    assert out.suggestion is None


def test_verifier_can_flip_likely_to_unlikely(monkeypatch):
    stub = _StubLLM({"verdict": "UNLIKELY", "reason": "Found landing page."})
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    out = llm_verifier.verify(_ref(title="x"), [], initial=HallucinationVerdict.LIKELY)
    assert out.verdict == HallucinationVerdict.UNLIKELY


def test_verifier_skips_when_already_unlikely(monkeypatch):
    """Don't waste an LLM call when pre-screen is already confident."""
    stub = _StubLLM({"verdict": "LIKELY"})
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    out = llm_verifier.verify(_ref(title="x"), [], initial=HallucinationVerdict.UNLIKELY)
    assert out.verdict == HallucinationVerdict.UNLIKELY
    assert stub.calls == []


def test_verifier_falls_back_on_error(monkeypatch):
    stub = _StubLLM({"status": "error", "error": "TimeoutError"})
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    out = llm_verifier.verify(_ref(title="x"), [], initial=HallucinationVerdict.LIKELY)
    assert out.verdict == HallucinationVerdict.LIKELY


def test_verifier_falls_back_on_malformed_payload(monkeypatch):
    stub = _StubLLM({"foo": "bar"})  # no "verdict"
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    out = llm_verifier.verify(_ref(title="x"), [], initial=HallucinationVerdict.UNCERTAIN)
    assert out.verdict == HallucinationVerdict.UNCERTAIN
    assert out.suggestion is None


def test_verifier_returns_suggestion_when_unlikely(monkeypatch):
    """When the LLM declares the paper real, it may attach corrected metadata."""
    stub = _StubLLM(
        {
            "verdict": "UNLIKELY",
            "reason": "Real paper; cited title has a typo.",
            "suggestion": {
                "title": "MathArena: Evaluating LLMs on Uncontaminated Math Competitions",
                "authors": ["Mislav Balunovic", "Jasper Dekoninck"],
                "year": 2025,
                "arxiv_id": "2505.23281",
            },
        }
    )
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    out = llm_verifier.verify(
        _ref(title="Math-arena: Evaluating llms on uncontaminated math competitions"),
        [],
        initial=HallucinationVerdict.LIKELY,
    )
    assert out.verdict == HallucinationVerdict.UNLIKELY
    assert out.suggestion is not None
    assert out.suggestion.title.startswith("MathArena")
    assert out.suggestion.arxiv_id == "2505.23281"


def test_verifier_drops_suggestion_when_verdict_not_unlikely(monkeypatch):
    """A suggestion is only meaningful alongside an UNLIKELY verdict."""
    stub = _StubLLM(
        {
            "verdict": "LIKELY",
            "reason": "Looks fabricated.",
            "suggestion": {"title": "Some other paper"},
        }
    )
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    out = llm_verifier.verify(_ref(title="x"), [], initial=HallucinationVerdict.LIKELY)
    assert out.verdict == HallucinationVerdict.LIKELY
    assert out.suggestion is None


def test_verifier_drops_empty_suggestion(monkeypatch):
    """Suggestion lacking any actionable identifier is discarded."""
    stub = _StubLLM(
        {
            "verdict": "UNLIKELY",
            "reason": "Real paper.",
            "suggestion": {"authors": ["Anon"]},  # no title/arxiv_id/doi
        }
    )
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    out = llm_verifier.verify(_ref(title="x"), [], initial=HallucinationVerdict.LIKELY)
    assert out.verdict == HallucinationVerdict.UNLIKELY
    assert out.suggestion is None


def test_verifier_drops_malformed_suggestion(monkeypatch):
    """Suggestion that isn't a dict is ignored."""
    stub = _StubLLM(
        {"verdict": "UNLIKELY", "reason": "Real.", "suggestion": "not a dict"}
    )
    monkeypatch.setattr(llm_verifier, "call_json", stub)

    out = llm_verifier.verify(_ref(title="x"), [], initial=HallucinationVerdict.LIKELY)
    assert out.verdict == HallucinationVerdict.UNLIKELY
    assert out.suggestion is None
