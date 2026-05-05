"""Main RefCopilot orchestrator.

Tying together:
  inputs (detector / bibtex / pdf / url / plain_text)
    → extract (LLM-only)
    → search (arxiv + semantic_scholar)
    → merge
    → verify (hallucination → optional LLM verifier → outdated → completeness)
    → report
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from refcopilot.cache.disk_cache import DiskCache
from refcopilot.extract.llm_extractor import extract_references
from refcopilot.inputs import bibtex as bibtex_input
from refcopilot.inputs import pdf as pdf_input
from refcopilot.inputs import plain_text as text_input
from refcopilot.inputs import url as url_input
from refcopilot.inputs.detector import detect
from refcopilot.merge import merge_records
from refcopilot.models import (
    CheckedReference,
    HallucinationVerdict,
    Issue,
    IssueCategory,
    Reference,
    Report,
    ReportSummary,
    SourceFormat,
    Verdict,
)
from refcopilot.search.arxiv import ArxivBackend
from refcopilot.search.openreview import OpenReviewBackend
from refcopilot.search.semantic_scholar import SemanticScholarBackend
from refcopilot.verify import completeness as completeness_verify
from refcopilot.verify import hallucination as hallu_verify
from refcopilot.verify import llm_verifier
from refcopilot.verify import non_academic
from refcopilot.verify import outdated as outdated_verify

logger = logging.getLogger(__name__)


_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "refcopilot"


class RefCopilotPipeline:
    def __init__(
        self,
        *,
        cache_dir: Path | str | None = None,
        cache_enabled: bool = True,
        cache_ttl_days: int = 30,
        s2_api_key: str | None = None,
        s2_base_url: str | None = None,
        arxiv_backend: ArxivBackend | None = None,
        s2_backend: SemanticScholarBackend | None = None,
        openreview_backend: OpenReviewBackend | None = None,
        use_llm_verify: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.cache = DiskCache(
            Path(cache_dir or _DEFAULT_CACHE_DIR),
            ttl_days=cache_ttl_days,
            enabled=cache_enabled,
        )
        self.arxiv = arxiv_backend or ArxivBackend(cache=self.cache)
        self.s2 = s2_backend or SemanticScholarBackend(
            api_key=s2_api_key,
            base_url=(s2_base_url or "https://api.semanticscholar.org/graph/v1"),
            cache=self.cache,
        )
        self.openreview = openreview_backend or OpenReviewBackend(cache=self.cache)
        self.use_llm_verify = use_llm_verify
        self.max_workers = max(1, int(max_workers))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        spec: str,
        *,
        input_type: SourceFormat | None = None,
        max_refs: int | None = None,
    ) -> Report:
        kind = input_type or detect(spec)
        references = self._extract_references(spec, kind)
        if max_refs and len(references) > max_refs:
            references = references[:max_refs]
        checked = self._check_all(references)
        return self._build_report(spec, kind, checked)

    # ------------------------------------------------------------------
    # Stage 1: input → references
    # ------------------------------------------------------------------

    def _extract_references(self, spec: str, kind: SourceFormat) -> list[Reference]:
        if kind == SourceFormat.BIBTEX:
            if _is_existing_path(spec):
                return bibtex_input.parse_file(spec)
            return bibtex_input.parse_string(spec)

        if kind == SourceFormat.PDF:
            bib_text = pdf_input.extract_bibliography(spec)
            return extract_references(bib_text, source_format=SourceFormat.PDF)

        if kind == SourceFormat.URL:
            cache_dir = self.cache.paper_dir(spec) if self.cache.enabled else Path.cwd() / "refcopilot-out"
            local_pdf = url_input.download(spec, cache_dir)
            bib_text = pdf_input.extract_bibliography(local_pdf)
            return extract_references(bib_text, source_format=SourceFormat.URL)

        if kind == SourceFormat.TEXT:
            normalized = text_input.normalize(spec)
            return extract_references(normalized, source_format=SourceFormat.TEXT)

        raise ValueError(f"unsupported input kind: {kind}")

    # ------------------------------------------------------------------
    # Stage 2: search + verify per reference
    # ------------------------------------------------------------------

    def _check_all(self, references: list[Reference]) -> list[CheckedReference]:
        if not references:
            return []
        if self.max_workers <= 1 or len(references) == 1:
            return [self._check_one(r) for r in references]

        results: list[CheckedReference | None] = [None] * len(references)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._check_one, ref): i for i, ref in enumerate(references)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    logger.warning("check_one failed for ref %d: %s", idx, exc)
                    results[idx] = CheckedReference(reference=references[idx], verdict=Verdict.UNVERIFIED)
        return [r for r in results if r is not None]

    def _check_one(self, ref: Reference) -> CheckedReference:
        arxiv_records = _safe_lookup(self.arxiv, ref, "arxiv")
        s2_records = _safe_lookup(self.s2, ref, "semantic_scholar")
        openreview_records = _safe_lookup(self.openreview, ref, "openreview")
        matches = list(arxiv_records) + list(s2_records) + list(openreview_records)
        merged = merge_records(matches) if matches else None

        initial_arxiv_count = len(arxiv_records)
        initial_s2_count = len(s2_records)
        initial_openreview_count = len(openreview_records)

        pre_verdict = hallu_verify.pre_screen(ref, matches, merged)
        verdict = pre_verdict
        llm_verdict: HallucinationVerdict | None = None
        suggestion: llm_verifier.LLMSuggestion | None = None
        retry_arxiv_count: int | None = None
        retry_s2_count: int | None = None
        retry_openreview_count: int | None = None
        retry_used = False

        if self.use_llm_verify and verdict != HallucinationVerdict.UNLIKELY:
            llm_result = llm_verifier.verify(ref, matches, initial=verdict)
            llm_verdict = llm_result.verdict
            verdict = llm_result.verdict
            suggestion = llm_result.suggestion

        # Second-chance lookup: when LLM thinks the cited paper is real but we
        # haven't found anything yet, re-run backends with the LLM-suggested
        # canonical metadata. Single retry, no recursion: we don't call the
        # LLM verifier again on the new matches.
        if (
            self.use_llm_verify
            and not matches
            and verdict == HallucinationVerdict.UNLIKELY
            and suggestion is not None
        ):
            synth_ref = _build_synth_reference(ref, suggestion)
            if synth_ref is not None:
                retry_used = True
                retry_arxiv = _safe_lookup(self.arxiv, synth_ref, "arxiv (retry)")
                retry_s2 = _safe_lookup(self.s2, synth_ref, "semantic_scholar (retry)")
                retry_openreview = _safe_lookup(
                    self.openreview, synth_ref, "openreview (retry)"
                )
                retry_arxiv_count = len(retry_arxiv)
                retry_s2_count = len(retry_s2)
                retry_openreview_count = len(retry_openreview)
                new_matches = (
                    list(retry_arxiv) + list(retry_s2) + list(retry_openreview)
                )
                if new_matches:
                    matches = new_matches
                    merged = merge_records(matches)
                    # Re-evaluate the heuristic against the new evidence.
                    # We deliberately skip llm_verifier on the retry path —
                    # the LLM already gave its opinion and we don't want to
                    # spend another call (or risk drifting).
                    verdict = hallu_verify.pre_screen(ref, matches, merged)

        issues: list[Issue] = []
        fake_issue = hallu_verify.to_issue(verdict, ref, matches)
        if fake_issue:
            if self.use_llm_verify:
                fake_issue = non_academic.recheck(ref, matches, fake_issue)
            issues.append(fake_issue)

        # Skip metadata checks only when the LLM-confirmed fake verdict still stands.
        suppress_metadata_checks = (
            verdict == HallucinationVerdict.LIKELY
            and fake_issue is not None
            and fake_issue.category == IssueCategory.FAKE
        )
        if not suppress_metadata_checks:
            issues.extend(outdated_verify.detect(ref, merged))
            issues.extend(completeness_verify.detect(ref, merged))

        final = _verdict_from_issues(issues, has_match=merged is not None)
        trace = _build_verification_trace(
            arxiv_count=initial_arxiv_count,
            s2_count=initial_s2_count,
            openreview_count=initial_openreview_count,
            pre_verdict=pre_verdict,
            llm_verdict=llm_verdict,
            suggestion=suggestion,
            retry_used=retry_used,
            retry_arxiv_count=retry_arxiv_count,
            retry_s2_count=retry_s2_count,
            retry_openreview_count=retry_openreview_count,
        )
        return CheckedReference(
            reference=ref,
            matches=matches,
            merged=merged,
            hallucination_verdict=verdict,
            verification_trace=trace,
            issues=issues,
            verdict=final,
        )

    # ------------------------------------------------------------------
    # Stage 3: report
    # ------------------------------------------------------------------

    def _build_report(self, spec: str, kind: SourceFormat, checked: list[CheckedReference]) -> Report:
        errors = sum(1 for c in checked if c.verdict == Verdict.ERROR)
        warnings = sum(1 for c in checked if c.verdict == Verdict.WARNING)
        unverified = sum(1 for c in checked if c.verdict == Verdict.UNVERIFIED)
        by_category: dict[str, int] = {}
        for c in checked:
            for issue in c.issues:
                by_category[issue.category.value] = by_category.get(issue.category.value, 0) + 1

        return Report(
            paper={"input": spec, "kind": kind.value},
            checked=checked,
            summary=ReportSummary(
                total_refs=len(checked),
                errors=errors,
                warnings=warnings,
                unverified=unverified,
                by_category=by_category,
            ),
        )


def _safe_lookup(backend, ref, name):
    if backend is None:
        return []
    try:
        return backend.lookup(ref) or []
    except Exception as exc:
        logger.warning("%s lookup failed for ref title=%r: %s", name, ref.title, exc)
        return []


def _is_existing_path(spec: str) -> bool:
    try:
        return Path(spec).exists()
    except (OSError, ValueError):
        return False


def _verdict_from_issues(issues: list[Issue], *, has_match: bool) -> Verdict:
    has_error = any(i.severity.value == "error" for i in issues)
    if has_error:
        return Verdict.ERROR
    has_warning = any(i.severity.value == "warning" for i in issues)
    if has_warning:
        return Verdict.WARNING
    if not has_match:
        return Verdict.UNVERIFIED
    return Verdict.VALID


def _build_synth_reference(
    ref: Reference, suggestion: llm_verifier.LLMSuggestion
) -> Reference | None:
    """Merge an LLM suggestion onto the original Reference for a re-lookup.

    Only fields the LLM filled in override the original. Returns ``None`` if
    the resulting reference would be identical to the input (no point in
    re-running the same query).
    """
    new_title = suggestion.title or ref.title
    new_authors = suggestion.authors if suggestion.authors else ref.authors
    new_year = suggestion.year if suggestion.year is not None else ref.year
    new_arxiv_id = suggestion.arxiv_id or ref.arxiv_id
    new_doi = suggestion.doi or ref.doi

    if (
        new_title == ref.title
        and new_authors == ref.authors
        and new_year == ref.year
        and new_arxiv_id == ref.arxiv_id
        and new_doi == ref.doi
    ):
        return None

    return Reference(
        raw=ref.raw,
        source_format=ref.source_format,
        bibkey=ref.bibkey,
        title=new_title,
        authors=new_authors,
        year=new_year,
        venue=ref.venue,
        doi=new_doi,
        arxiv_id=new_arxiv_id,
        arxiv_version=ref.arxiv_version,
        url=ref.url,
    )


def _build_verification_trace(
    *,
    arxiv_count: int,
    s2_count: int,
    openreview_count: int,
    pre_verdict: HallucinationVerdict,
    llm_verdict: HallucinationVerdict | None,
    suggestion: llm_verifier.LLMSuggestion | None,
    retry_used: bool,
    retry_arxiv_count: int | None,
    retry_s2_count: int | None,
    retry_openreview_count: int | None,
) -> str:
    """Single-line summary of which sources were tried and what they said."""
    parts = [
        f"arXiv: {arxiv_count}",
        f"S2: {s2_count}",
        f"OpenReview: {openreview_count}",
        f"pre-screen: {pre_verdict.value}",
    ]
    if llm_verdict is not None and llm_verdict != pre_verdict:
        parts.append(f"LLM: {llm_verdict.value}")
    if retry_used:
        suggested_title = (suggestion.title if suggestion else None) or "(no title)"
        title_short = suggested_title if len(suggested_title) <= 80 else suggested_title[:77] + "..."
        parts.append(
            f"retry with LLM suggestion '{title_short}' → arXiv: {retry_arxiv_count}, "
            f"S2: {retry_s2_count}, OpenReview: {retry_openreview_count}"
        )
    return "; ".join(parts)
