"""Verify-layer tests, exercised through the public ``RefCopilotPipeline``.

Replaces test_c_verify/test_{fake,outdated,incomplete,retraction,non_academic,
llm_verifier,bibtex_suggest}.py. Each test runs the full pipeline against
``_FakeBackend`` shims and pins down the verdict the pipeline emits for a
specific reference shape — these are the policy decisions FactReview reports
on, so they're the right granularity for regression coverage.
"""

from __future__ import annotations

from refcopilot.models import (
    Backend,
    ExternalRecord,
    SourceFormat,
    Verdict,
)
from refcopilot.pipeline import RefCopilotPipeline


class _FakeBackend:
    """Returns canned ``ExternalRecord`` lists keyed by arxiv_id or title."""

    def __init__(self, results: dict | None = None) -> None:
        self.results = results or {}
        self.calls: list = []

    def lookup(self, ref):  # type: ignore[no-untyped-def]
        self.calls.append(ref)
        if ref.arxiv_id and ref.arxiv_id in self.results:
            return self.results[ref.arxiv_id]
        if ref.title and ref.title.lower() in self.results:
            return self.results[ref.title.lower()]
        return []


def _attention_arxiv(**kw) -> ExternalRecord:
    base = dict(
        backend=Backend.ARXIV,
        record_id="1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        year=2017,
        arxiv_id="1706.03762",
        latest_arxiv_version=7,
        url="https://arxiv.org/abs/1706.03762v7",
    )
    base.update(kw)
    return ExternalRecord(**base)


def _attention_s2(**kw) -> ExternalRecord:
    base = dict(
        backend=Backend.SEMANTIC_SCHOLAR,
        record_id="abc",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        year=2017,
        venue="NeurIPS",
        publication_venue="NeurIPS",
        doi="10.5555/3295222.3295349",
        arxiv_id="1706.03762",
    )
    base.update(kw)
    return ExternalRecord(**base)


def _build_pipeline(tmp_path, *, arxiv=None, s2=None, openreview=None, use_llm_verify=False):
    return RefCopilotPipeline(
        cache_dir=tmp_path,
        arxiv_backend=_FakeBackend(arxiv or {}),
        s2_backend=_FakeBackend(s2 or {}),
        openreview_backend=_FakeBackend(openreview or {}),
        use_llm_verify=use_llm_verify,
        max_workers=1,
    )


def test_clean_bibtex_with_full_metadata_yields_no_errors(tmp_path, fixtures_dir) -> None:
    arxiv = {"1706.03762": [_attention_arxiv()]}
    s2 = {"1706.03762": [_attention_s2()]}
    pipeline = _build_pipeline(tmp_path, arxiv=arxiv, s2=s2)

    bib = (fixtures_dir / "inputs" / "minimal.bib").read_text(encoding="utf-8")
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)

    vas = next(c for c in report.checked if c.reference.bibkey == "vaswani2017attention")
    assert vas.verdict in (Verdict.VALID, Verdict.WARNING)
    # The hard contract: a fully-resolved citation must NOT be flagged as
    # fabricated. ``no_match`` here would mean the search/merge regressed.
    assert all(i.code != "no_match" for i in vas.issues)


def test_unmatched_reference_is_flagged_as_error(tmp_path, fixtures_dir) -> None:
    arxiv = {"1706.03762": [_attention_arxiv()]}
    s2 = {"1706.03762": [_attention_s2()]}
    pipeline = _build_pipeline(tmp_path, arxiv=arxiv, s2=s2)

    bib = (fixtures_dir / "inputs" / "one_fake.bib").read_text(encoding="utf-8")
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)

    fake = next(c for c in report.checked if c.reference.bibkey == "fake2025bagels")
    assert fake.verdict is Verdict.ERROR
    assert any(i.category.value == "fake" for i in fake.issues)


def test_retracted_reference_is_flagged_as_error(tmp_path) -> None:
    # Preserves the f63cb75 contract: any backend reporting is_retracted=True
    # results in a hard error, not a warning.
    bib = """
    @article{retracted,
      author = {X. Author},
      title  = {A retracted paper},
      year   = {2020},
      doi    = {10.1109/access.2020.3018326},
    }
    """
    retracted_record = ExternalRecord(
        backend=Backend.OPENALEX,
        record_id="W123",
        title="A retracted paper",
        authors=["X. Author"],
        year=2020,
        doi="10.1109/access.2020.3018326",
        is_retracted=True,
    )
    pipeline = RefCopilotPipeline(
        cache_dir=tmp_path,
        arxiv_backend=_FakeBackend({}),
        s2_backend=_FakeBackend({"a retracted paper": [retracted_record]}),
        openreview_backend=_FakeBackend({}),
        use_llm_verify=False,
        max_workers=1,
    )
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)
    c = report.checked[0]
    assert c.verdict is Verdict.ERROR
    codes = {i.code for i in c.issues}
    assert "is_retracted" in codes


def test_outdated_arxiv_reference_emits_published_warning(tmp_path) -> None:
    bib = """
    @misc{vas,
      author = {A. Vaswani},
      title  = {Attention Is All You Need},
      year   = {2017},
      eprint = {1706.03762},
      archivePrefix = {arXiv},
    }
    """
    arxiv = {"1706.03762": [_attention_arxiv()]}
    s2 = {"1706.03762": [_attention_s2()]}  # carries DOI/venue → "published"
    pipeline = _build_pipeline(tmp_path, arxiv=arxiv, s2=s2)

    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)
    c = report.checked[0]
    assert c.verdict is Verdict.WARNING
    codes = {i.code for i in c.issues}
    assert "arxiv_published" in codes


def test_incomplete_reference_missing_doi_is_warning(tmp_path) -> None:
    bib = """
    @inproceedings{vas,
      author = {A. Vaswani},
      title  = {Attention Is All You Need},
      booktitle = {NeurIPS},
      year   = {2017},
    }
    """
    arxiv = {"attention is all you need": [_attention_arxiv()]}
    s2 = {"attention is all you need": [_attention_s2()]}
    pipeline = _build_pipeline(tmp_path, arxiv=arxiv, s2=s2)

    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)
    c = report.checked[0]
    codes = {i.code for i in c.issues}
    assert "missing_doi" in codes


def test_non_academic_citation_downgrades_fake_to_warning(tmp_path, monkeypatch) -> None:
    # When the LLM verifier confirms the citation looks fake, the non-academic
    # recheck must downgrade vendor system cards / blog posts to a WARNING
    # with category=non_academic. This is the safety valve for legitimate
    # non-paper references that show up in modern bibliographies.
    from refcopilot.verify import llm_verifier, non_academic

    monkeypatch.setattr(
        llm_verifier,
        "call_json",
        lambda *, prompt, system, **kw: {"verdict": "LIKELY", "reason": "no academic match"},
    )
    monkeypatch.setattr(
        non_academic,
        "call_json",
        lambda *, prompt, system, **kw: {
            "is_non_academic": True,
            "citation_type": "system_card",
            "reasoning": "vendor system card.",
        },
    )

    bib = """
    @misc{claude46,
      author = {Anthropic},
      title  = {Claude opus 4.6 system card},
      year   = {2026},
    }
    """
    pipeline = _build_pipeline(tmp_path, use_llm_verify=True)
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)

    c = report.checked[0]
    assert c.verdict is Verdict.WARNING
    cats = {i.category.value for i in c.issues}
    assert "non_academic" in cats
    assert "fake" not in cats


def test_llm_verifier_suggestion_resolves_unverified(tmp_path, monkeypatch) -> None:
    # When the initial title-based search misses but the LLM proposes a
    # corrected title + arxiv_id, the pipeline retries by id and lifts the
    # citation out of UNVERIFIED.
    from refcopilot.verify import llm_verifier

    canonical_title = "MathArena: Evaluating LLMs on Uncontaminated Math Competitions"

    monkeypatch.setattr(
        llm_verifier,
        "call_json",
        lambda *, prompt, system, **kw: {
            "verdict": "UNLIKELY",
            "reason": "Real paper, cited with a typo.",
            "suggestion": {"title": canonical_title, "arxiv_id": "2505.23281", "year": 2025},
        },
    )

    canonical = ExternalRecord(
        backend=Backend.ARXIV,
        record_id="2505.23281",
        title=canonical_title,
        authors=["Mislav Balunovic"],
        year=2025,
        arxiv_id="2505.23281",
        url="https://arxiv.org/abs/2505.23281",
    )
    pipeline = _build_pipeline(tmp_path, arxiv={"2505.23281": [canonical]}, use_llm_verify=True)

    bib = """
    @inproceedings{matharena,
      author = {Mislav Balunovic},
      title  = {Math-arena: Evaluating llms on uncontaminated math competitions},
      year   = {2025},
    }
    """
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)
    c = report.checked[0]

    assert c.verdict in (Verdict.WARNING, Verdict.VALID)
    assert c.merged is not None
    assert c.merged.arxiv_id == "2505.23281"
    assert c.verification_trace and "retry" in c.verification_trace.lower()
