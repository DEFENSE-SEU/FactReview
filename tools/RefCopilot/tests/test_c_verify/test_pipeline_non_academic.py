"""Pipeline-level test: fake-error → non-academic warning downgrade."""

from __future__ import annotations

from refcopilot.models import SourceFormat, Verdict
from refcopilot.pipeline import RefCopilotPipeline
from refcopilot.verify import llm_verifier, non_academic


class _FakeBackend:
    def lookup(self, ref):
        return []


def test_pipeline_downgrades_system_card_to_warning(tmp_path, monkeypatch):
    # llm_verifier confirms LIKELY (so we'd normally emit an error).
    monkeypatch.setattr(llm_verifier, "call_json", lambda *, prompt, system, **kw: {
        "verdict": "LIKELY", "reason": "no academic match"
    })
    # non_academic recheck flips it to a warning.
    monkeypatch.setattr(non_academic, "call_json", lambda *, prompt, system, **kw: {
        "is_non_academic": True,
        "citation_type": "system_card",
        "reasoning": "Anthropic vendor system card.",
    })

    bib = """
    @misc{claude46,
      author = {Anthropic},
      title  = {Claude opus 4.6 system card},
      year   = {2026},
    }
    """
    pipeline = RefCopilotPipeline(
        cache_dir=tmp_path,
        arxiv_backend=_FakeBackend(),
        s2_backend=_FakeBackend(),
        use_llm_verify=True,
        max_workers=1,
    )
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)

    assert report.summary.total_refs == 1
    c = report.checked[0]
    assert c.verdict == Verdict.WARNING
    cats = {i.category.value for i in c.issues}
    assert "non_academic" in cats
    assert "fake" not in cats


def test_pipeline_keeps_error_when_recheck_says_no(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_verifier, "call_json", lambda *, prompt, system, **kw: {
        "verdict": "LIKELY", "reason": "no academic match"
    })
    monkeypatch.setattr(non_academic, "call_json", lambda *, prompt, system, **kw: {
        "is_non_academic": False,
        "reasoning": "looks fabricated",
    })

    bib = """
    @article{fake,
      author = {X. Fake and Y. Imaginary},
      title  = {Quantum Bagel Theorems},
      year   = {2025},
    }
    """
    pipeline = RefCopilotPipeline(
        cache_dir=tmp_path,
        arxiv_backend=_FakeBackend(),
        s2_backend=_FakeBackend(),
        use_llm_verify=True,
        max_workers=1,
    )
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)

    c = report.checked[0]
    assert c.verdict == Verdict.ERROR
    cats = {i.category.value for i in c.issues}
    assert "fake" in cats
    assert "non_academic" not in cats
