"""End-to-end pipeline (mocked backends + mocked LLM)."""

from __future__ import annotations

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.models import (
    Backend,
    ExternalRecord,
    SourceFormat,
    Verdict,
)
from refcopilot.pipeline import RefCopilotPipeline


class _FakeBackend:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def lookup(self, ref):
        self.calls.append(ref)
        if ref.arxiv_id and ref.arxiv_id in self.results:
            return self.results[ref.arxiv_id]
        if ref.title and ref.title.lower() in self.results:
            return self.results[ref.title.lower()]
        return []


def _attention_arxiv():
    return ExternalRecord(
        backend=Backend.ARXIV,
        record_id="1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        arxiv_id="1706.03762",
        latest_arxiv_version=7,
        url="https://arxiv.org/abs/1706.03762v7",
    )


def _attention_s2():
    return ExternalRecord(
        backend=Backend.SEMANTIC_SCHOLAR,
        record_id="abc123",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue="Advances in Neural Information Processing Systems",
        publication_venue="NeurIPS",
        doi="10.5555/3295222.3295349",
        arxiv_id="1706.03762",
        s2_paper_id="abc123",
        url="https://www.semanticscholar.org/paper/abc123",
    )


def _build_pipeline(
    tmp_path,
    *,
    arxiv_results=None,
    s2_results=None,
    openreview_results=None,
    use_llm_verify=False,
):
    arxiv = _FakeBackend(arxiv_results or {})
    s2 = _FakeBackend(s2_results or {})
    openreview = _FakeBackend(openreview_results or {})
    return RefCopilotPipeline(
        cache_dir=tmp_path,
        arxiv_backend=arxiv,
        s2_backend=s2,
        openreview_backend=openreview,
        use_llm_verify=use_llm_verify,
        max_workers=1,
    )


# ---- Bibtex inputs ---------------------------------------------------------


def test_pipeline_clean_bibtex_no_issues(tmp_path, fixtures_dir):
    arxiv = {"1706.03762": [_attention_arxiv()]}
    s2 = {"1706.03762": [_attention_s2()]}
    pipeline = _build_pipeline(tmp_path, arxiv_results=arxiv, s2_results=s2)

    bib_text = (fixtures_dir / "inputs" / "minimal.bib").read_text(encoding="utf-8")
    report = pipeline.run(bib_text, input_type=SourceFormat.BIBTEX)

    assert report.summary.total_refs == 3
    # Vaswani has both arxiv + s2 → at most "missing_doi" if DOI missing in cite
    vas = next(c for c in report.checked if c.reference.bibkey == "vaswani2017attention")
    assert vas.verdict in (Verdict.VALID, Verdict.WARNING)
    assert all(i.code != "no_match" for i in vas.issues)


def test_pipeline_one_fake_yields_error(tmp_path, fixtures_dir):
    # Set up backends that resolve Vaswani but NOT the fake bagels paper.
    arxiv = {"1706.03762": [_attention_arxiv()]}
    s2 = {"1706.03762": [_attention_s2()]}
    pipeline = _build_pipeline(tmp_path, arxiv_results=arxiv, s2_results=s2)

    bib_text = (fixtures_dir / "inputs" / "one_fake.bib").read_text(encoding="utf-8")
    report = pipeline.run(bib_text, input_type=SourceFormat.BIBTEX)

    assert report.summary.total_refs == 2
    fake = next(c for c in report.checked if c.reference.bibkey == "fake2025bagels")
    assert fake.verdict == Verdict.ERROR
    assert any(i.category.value == "fake" for i in fake.issues)


def test_pipeline_outdated_arxiv_published(tmp_path):
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
    s2 = {"1706.03762": [_attention_s2()]}
    pipeline = _build_pipeline(tmp_path, arxiv_results=arxiv, s2_results=s2)

    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)
    assert report.summary.total_refs == 1
    c = report.checked[0]
    assert c.verdict == Verdict.WARNING
    codes = {i.code for i in c.issues}
    assert "arxiv_published" in codes


def test_pipeline_incomplete_missing_doi(tmp_path):
    bib = """
    @inproceedings{vas,
      author = {A. Vaswani and N. Shazeer},
      title  = {Attention Is All You Need},
      booktitle = {NeurIPS},
      year   = {2017},
    }
    """
    arxiv = {"attention is all you need": [_attention_arxiv()]}
    s2 = {"attention is all you need": [_attention_s2()]}
    pipeline = _build_pipeline(tmp_path, arxiv_results=arxiv, s2_results=s2)

    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)
    c = report.checked[0]
    codes = {i.code for i in c.issues}
    assert "missing_doi" in codes


def test_pipeline_summary_counts(tmp_path, fixtures_dir):
    # Mix: 1 valid + 1 fake (no records for fake)
    arxiv = {"1706.03762": [_attention_arxiv()]}
    s2 = {"1706.03762": [_attention_s2()]}
    pipeline = _build_pipeline(tmp_path, arxiv_results=arxiv, s2_results=s2)

    bib_text = (fixtures_dir / "inputs" / "one_fake.bib").read_text(encoding="utf-8")
    report = pipeline.run(bib_text, input_type=SourceFormat.BIBTEX)

    assert report.summary.errors >= 1
    assert "fake" in report.summary.by_category


def test_pipeline_no_matches_yields_unverified_or_error(tmp_path):
    bib = """
    @article{x,
      author = {A. Smith},
      title  = {A paper that does not exist},
      year   = {2020},
    }
    """
    pipeline = _build_pipeline(tmp_path, arxiv_results={}, s2_results={})
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)
    c = report.checked[0]
    # Without URL → likely treated as fake; with URL → unverified.
    assert c.verdict in (Verdict.ERROR, Verdict.UNVERIFIED)
    # Trace string should always be populated.
    assert c.verification_trace is not None
    assert "arXiv: 0" in c.verification_trace


def test_pipeline_unverified_renders_trace_in_markdown(tmp_path):
    """Unverified entries get a `_Verification trace: ..._` italic line in the report."""
    from refcopilot.report import to_markdown

    bib = """
    @article{x,
      author = {A. Smith},
      title  = {A paper that does not exist},
      year   = {2020},
      url    = {https://example.com/x},
    }
    """
    pipeline = _build_pipeline(tmp_path, arxiv_results={}, s2_results={})
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)

    md = to_markdown(report)
    if any(c.verdict == Verdict.UNVERIFIED for c in report.checked):
        assert "_Verification trace:" in md


def test_pipeline_retries_with_llm_suggestion(tmp_path, monkeypatch):
    """When initial backends miss but the LLM suggests a corrected title, retry succeeds."""
    from refcopilot.verify import llm_verifier

    cited_title = "Math-arena: Evaluating llms on uncontaminated math competitions"
    canonical_title = "MathArena: Evaluating LLMs on Uncontaminated Math Competitions"

    def stub_call_json(*, prompt, system, **kw):
        return {
            "verdict": "UNLIKELY",
            "reason": "Real paper, cited with a typo.",
            "suggestion": {
                "title": canonical_title,
                "arxiv_id": "2505.23281",
                "year": 2025,
            },
        }

    monkeypatch.setattr(llm_verifier, "call_json", stub_call_json)

    canonical_record = ExternalRecord(
        backend=Backend.ARXIV,
        record_id="2505.23281",
        title=canonical_title,
        authors=["Mislav Balunovic", "Jasper Dekoninck", "Ivo Petrov"],
        year=2025,
        arxiv_id="2505.23281",
        latest_arxiv_version=3,
        url="https://arxiv.org/abs/2505.23281v3",
    )

    # Initial search by cited title misses; retry by arxiv_id hits.
    arxiv_results = {"2505.23281": [canonical_record]}
    s2_results: dict = {}
    pipeline = _build_pipeline(
        tmp_path,
        arxiv_results=arxiv_results,
        s2_results=s2_results,
        use_llm_verify=True,
    )

    bib = f"""
    @inproceedings{{matharena,
      author = {{Mislav Balunovic and Jasper Dekoninck}},
      title  = {{ {cited_title} }},
      year   = {{2025}},
    }}
    """
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)
    c = report.checked[0]

    # Retry path lifted us out of UNVERIFIED.
    assert c.verdict in (Verdict.WARNING, Verdict.VALID)
    # The matched record now backs the citation.
    assert c.merged is not None
    assert c.merged.arxiv_id == "2505.23281"
    # Trace records the retry happened.
    assert c.verification_trace is not None
    assert "retry" in c.verification_trace.lower()
    # The canonical_title_mismatch warning should fire.
    codes = {i.code for i in c.issues}
    assert "canonical_title_mismatch" in codes


def test_pipeline_no_retry_when_llm_has_no_suggestion(tmp_path, monkeypatch):
    """No suggestion → no retry; verdict stays UNVERIFIED with trace explaining."""
    from refcopilot.verify import llm_verifier

    arxiv = _FakeBackend({})
    s2 = _FakeBackend({})
    openreview = _FakeBackend({})

    def stub_call_json(*, prompt, system, **kw):
        return {"verdict": "UNLIKELY", "reason": "Real paper."}  # no suggestion

    monkeypatch.setattr(llm_verifier, "call_json", stub_call_json)

    pipeline = RefCopilotPipeline(
        cache_dir=tmp_path,
        arxiv_backend=arxiv,
        s2_backend=s2,
        openreview_backend=openreview,
        use_llm_verify=True,
        max_workers=1,
    )

    bib = """
    @article{x,
      author = {A. Smith},
      title  = {Some real-sounding paper},
      year   = {2020},
      url    = {https://example.com/x},
    }
    """
    report = pipeline.run(bib, input_type=SourceFormat.BIBTEX)
    c = report.checked[0]

    # arxiv was called once for the initial lookup; never again because no suggestion.
    assert len(arxiv.calls) == 1
    assert c.verdict == Verdict.UNVERIFIED
    assert c.verification_trace and "retry" not in c.verification_trace.lower()
