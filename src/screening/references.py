"""Entry-level RefCopilot check over the shared bibliography."""

import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import Field

from common import run_stats
from fact_generation.refcheck.refcheck import ReferenceCheckBundle
from fact_generation.refcheck.refcheck import check_references_with_records as check_references
from schemas.claim import Contract, Evidence, EvidencePointer, Finding, NonEmpty
from schemas.materials import SharedMaterials
from schemas.review import DeliveryCheck
from screening.checks import ask, paper_finding
from screening.reference_corrections import bibliographic_ids as _bibliographic_ids
from screening.reference_corrections import bound_record, build_correction, printed_publication_venue
from screening.reference_corrections import url_identifiers as _url_identifiers
from screening.reference_warnings import workshop_publication_metadata
from screening.visual_audit import redacted_record


class ReferenceDecision(Contract):
    candidate_id: NonEmpty
    page: int = Field(ge=1)
    classification: Literal["manuscript_error", "parser_artifact", "uncertain"]
    printed_quote: str = ""
    comparison_quote: str = ""
    mismatch_kind: Literal["title", "wrong_author", "author_order", "unknown"] = "unknown"
    reason: NonEmpty


class ReferencePageReview(Contract):
    results: list[ReferenceDecision]


def _identity_text(value: str) -> str:
    """Formatting alone cannot establish a different author or title identity."""
    return "".join(char for char in unicodedata.normalize("NFKC", value).casefold() if char.isalnum())


def _reference_identity(row, block):
    """The exported RefCopilot shape loses match provenance; require explicit shared identity."""
    cited_url, verified_url = str(row.get("cited_url") or ""), str(row.get("verified_url") or "")
    # cited_url is itself produced by RefCopilot's extractor; it cannot prove
    # that an identifier was printed in the manuscript.
    cited = _bibliographic_ids(block.text)
    verified = _url_identifiers(verified_url)
    common = cited & verified
    qualified = len(cited) == len(verified) == len(common) == 1
    identifier = next(iter(common)) if len(common) == 1 else None
    if identifier and identifier.startswith("arxiv:") and not re.search(r"v[1-9]\d*$", identifier):
        qualified = False
    reason = "Shared DOI or explicit arXiv version in original citation and verified_url."
    if not qualified:
        reason = "Same-work/version identity unavailable: require one shared DOI or explicit arXiv version."
    # A closest-match suggestion describes a retrieval candidate, without binding
    # that candidate's quoted metadata to the exported merged record URL.
    if "closest match:" in str(row.get("details") or "").casefold():
        qualified = False
        reason = "Closest-match metadata does not establish the identity of the cited work/version."
    return {
        "qualified": qualified,
        "identifier": identifier,
        "cited_identifiers": sorted(cited),
        "verified_identifiers": sorted(verified),
        "cited_url": cited_url,
        "verified_url": verified_url,
        "reason": reason,
    }


def check_bibliography(
    materials: SharedMaterials, output_dir: Path, *, checker=None, call=None, delivery_checks=None
) -> tuple[list[Finding], list[str]]:
    if not materials.bibliography:
        run_stats.record_module_status("reference_check", "skipped")
        return [], ["Reference check unavailable: parser supplied no bibliography entries."]
    with run_stats.timed_module("reference_check"):
        try:
            findings, issues, status = _check_bibliography(
                materials, output_dir, checker=checker, call=call, delivery_checks=delivery_checks
            )
        except Exception as exc:
            run_stats.record_module_status("reference_check", "failed", warning=str(exc))
            _delivery_failure(
                delivery_checks, "reference_checker", "failed",
                f"Reference checking raised {type(exc).__name__}. Audit: {output_dir / 'reference_check.json'}",
            )
            raise
        if status == "failed":
            _delivery_failure(
                delivery_checks, "reference_checker", "failed",
                f"Reference checker returned an unsuccessful or invalid result. Audit: {output_dir / 'reference_check.json'}",
            )
        run_stats.record_module_status("reference_check", status, warning="; ".join(issues))
        return findings, issues


def _delivery_failure(sink, component, state, reason):
    if sink is not None:
        sink.append(DeliveryCheck(
            stage="screening", component=component, state=state, reason=redacted_record(reason, cfg=None)
        ))


def _check_bibliography(materials, output_dir, *, checker=None, call=None, delivery_checks=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "bibliography.txt"
    path.write_text("\n\n".join(b.text for b in materials.bibliography), encoding="utf-8")
    checked = (checker if checker is not None else check_references)(paper=str(path))
    records_path = None
    if isinstance(checked, ReferenceCheckBundle):
        result = checked.payload
        if checked.records is not None:
            records_path = output_dir / "reference_records.json"
            records_path.write_text(
                json.dumps(checked.records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    else:
        result = checked
    result_path = output_dir / "reference_check.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result.get("ok"):
        return (
            [],
            [f"Reference check failed: {result.get('error_message', 'unknown error')}; {result_path}"],
            "failed",
        )
    if not isinstance(result.get("total_refs"), int) or result["total_refs"] <= 0:
        return [], [f"Reference check incomplete: no processed bibliography entries; {result_path}"], "failed"
    if not isinstance(result.get("issues"), list):
        return [], [f"Reference check returned no valid issues list; {result_path}"], "failed"
    findings, issues, candidates = [], [], []
    corrections = {}
    if result["total_refs"] != len(materials.bibliography):
        issues.append(
            f"Reference coverage needs review: checker processed {result['total_refs']} entries "
            f"from {len(materials.bibliography)} parser bibliography blocks; {result_path}"
        )
    for index, row in enumerate(result.get("issues", [])):
        title = str(row.get("reference_title") or row.get("reference") or "").strip()
        raw_reference = str(row.get("raw_reference") or "")
        if isinstance(checked, ReferenceCheckBundle) and checked.records is not None:
            try:
                reference_index = checked.records["bindings"][index]["reference_index"]
                raw_reference = checked.records["report"]["checked"][reference_index]["reference"]["raw"]
            except (KeyError, IndexError, TypeError):
                pass
        exact = [b for b in materials.bibliography if raw_reference and b.text == raw_reference]
        matches = exact or [
            b for b in materials.bibliography if title and title.casefold() in b.text.casefold()
        ]
        if len(matches) != 1 or matches[0].loc is None:
            issues.append(
                f"Reference finding lacks a unique paper location: {title}; {result_path}#issues.{index}"
            )
            continue
        block = matches[0]
        code = str(row.get("type") or row.get("code") or "").lower()
        if row.get("corrected_bibtex"):
            corrections[index] = build_correction(row, block, index, result_path, records_path)
            if corrections[index].state == "unavailable":
                issues.append(
                    f"Reference correction unavailable: {corrections[index].reason}; {result_path}#issues.{index}"
                )
        if "arxiv_published" in code:
            full_record = None
            try:
                full_record, _ = bound_record(row, block, index, result_path, records_path)
            except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
                pass
            venue = printed_publication_venue(row, block, full_record)
            if venue:
                issues.append(
                    f"Reference arxiv-only warning not confirmed: original entry already names publication venue {venue!r}; {result_path}#issues.{index}"
                )
                continue
            issues.append(
                f"Reference arxiv-only warning remains unconfirmed: retrieved publication metadata does not establish that the original citation omitted a venue; {result_path}#issues.{index}"
            )
            if index not in corrections or corrections[index].state != "metadata_candidate":
                continue
        if any(kind in code for kind in ("author_mismatch", "title_mismatch")):
            candidates.append((index, row, block))
            continue
        publication_text = None
        if "workshop_promoted" in code:
            publication_text, unavailable = workshop_publication_metadata(
                row, block, index, result_path, records_path, corrections.get(index)
            )
            if unavailable:
                issues.append(
                    f"Reference workshop publication warning unconfirmed: {unavailable}; {result_path}#issues.{index}"
                )
                continue
        text = str(
            row.get("details") or row.get("message") or row.get("code") or row.get("type") or "Entry check"
        )
        if publication_text:
            text = publication_text
        if "arxiv_published" in code:
            text = "Retrieved metadata provides a publication venue; verify whether the original preprint citation should be updated. Original manuscript omission is unconfirmed."
        text += f" [RefCopilot: {result_path}#issues.{index}]"
        finding = paper_finding(
            materials,
            block,
            quote=block.text,
            text=text,
            kind="reference",
            level="metadata_candidate" if publication_text else str(row.get("severity") or "unverified"),
        )
        if index in corrections:
            finding = finding.model_copy(update={"reference_correction": corrections[index]})
        findings.append(finding)
    checked_findings, validation_issues = _confirm_reference_candidates(
        materials, candidates, result_path, output_dir, call=call, corrections=corrections,
        delivery_checks=delivery_checks,
    )
    findings.extend(checked_findings)
    issues.extend(validation_issues)
    for index, _row, block in candidates:
        correction = corrections.get(index)
        if correction is None or correction.state != "metadata_candidate":
            continue
        if any(getattr(item, "reference_correction", None) == correction for item in checked_findings):
            continue
        finding = paper_finding(
            materials,
            block,
            quote=block.text,
            kind="reference",
            level="metadata_candidate",
            text="Identity-bound retrieved metadata is available as a citation candidate. "
            "The original-PDF author/title discrepancy was not confirmed; this suggestion does not establish a manuscript error.",
        )
        findings.append(finding.model_copy(update={"reference_correction": correction}))
    if corrections:
        (output_dir / "reference_corrections.json").write_text(
            json.dumps(
                {str(index): value.model_dump(mode="json") for index, value in corrections.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return findings, issues, "ok"


def _confirm_reference_candidates(
    materials, candidates, result_path, output_dir, *, call=None, corrections=None, delivery_checks=None
):
    """Keep parser-based metadata mismatches separate from PDF-confirmed defects."""
    findings, issues, audit = [], [], []
    by_page = {}
    for index, row, block in candidates:
        by_page.setdefault(block.loc.page, []).append((index, row, block))
    for page_number, page_candidates in by_page.items():
        records = [
            {
                "candidate_id": f"reference_{index + 1}",
                "page": page_number,
                "block_id": block.id,
                "parsed_reference": block.text,
                "refcopilot_pointer": f"{result_path}#issues.{index}",
                "refcopilot": row,
                "identity": _reference_identity(row, block),
                "comparison_metadata": "\n".join(
                    str(row.get(key) or "")
                    for key in (
                        "details",
                        "message",
                        "corrected_plaintext",
                        "corrected_bibtex",
                        "corrected_bibitem",
                    )
                ),
                "status": "unavailable",
            }
            for index, row, block in page_candidates
        ]
        audit.extend(records)
        page = next((page for page in materials.pages if page.page == page_number), None)
        if page is None or not Path(page.path).is_file():
            _delivery_failure(
                delivery_checks, "reference_pdf", "unavailable",
                f"Original PDF page {page_number} is unavailable for reference confirmation. "
                f"Audit: {output_dir / 'reference_validation.json'} (page {page_number}).",
            )
            for record in records:
                issues.append(
                    f"{record['candidate_id']}: reference mismatch unconfirmed; original PDF page image "
                    f"unavailable; {record['refcopilot_pointer']}"
                )
            continue
        try:
            review = ReferencePageReview.model_validate(
                ask(
                    "Verify each reference author/title mismatch against the attached ORIGINAL PDF page. "
                    "RefCopilot compared parser output with retrieved metadata; parser accent placement, OCR, "
                    "line wrapping and lost symbols can create false mismatches. Read the visible author/title "
                    "text from the pixels and compare it to the supplied comparison_metadata. Return "
                    "output_schema JSON with exactly one result for every candidate_id and its supplied page. "
                    "Use parser_artifact when the original printed reference agrees with the comparison "
                    "metadata and the discrepancy comes from parsing. Use manuscript_error only when the "
                    "specific author/title discrepancy is visibly present in the original manuscript. Copy "
                    "the relevant author/title fragment verbatim into printed_quote, and copy the differing "
                    "retrieved fragment verbatim from comparison_metadata into comparison_quote. Explain "
                    "the actual discrepancy in reason. A classification without both concrete fragments is "
                    "insufficient. The supplied identity is computed from existing citation and verified "
                    "identifiers; you cannot upgrade or invent it. When identity.qualified is false, use "
                    "uncertain for a visible metadata difference. You may still identify a parser_artifact "
                    "when the printed text agrees. Set mismatch_kind to title, wrong_author (different or "
                    "missing names), author_order (same authors, different order), or unknown. A closest "
                    "retrieval match cannot establish that the cited title or author is wrong. "
                    "Do not treat capitalization, punctuation, initials or formatting alone as "
                    "a wrong identity. Use uncertain when the entry is unreadable, spans missing pages, "
                    "retrieved metadata is ambiguous, or no concrete comparison can be made. Never invent "
                    "canonical metadata or infer an error from the parsed_reference alone.",
                    {
                        "page": page_number,
                        "candidates": records,
                        "output_schema": ReferencePageReview.model_json_schema(),
                    },
                    module="reference_check",
                    call=call,
                    images=[page.path],
                )
            )
            received = [decision.candidate_id for decision in review.results]
            if len(received) != len(set(received)) or set(received) != {
                record["candidate_id"] for record in records
            }:
                raise ValueError("Reference validation must cover each candidate exactly once")
            if any(decision.page != page_number for decision in review.results):
                raise ValueError("Reference validation returned an unrelated PDF page")
        except Exception as exc:
            _delivery_failure(
                delivery_checks, "reference_pdf", "failed",
                f"Original PDF reference confirmation raised {type(exc).__name__} on page {page_number}. "
                f"Audit: {output_dir / 'reference_validation.json'} (page {page_number}).",
            )
            for record in records:
                record.update(status="failed", error=str(exc))
                issues.append(
                    f"{record['candidate_id']}: original PDF reference validation failed: {exc}; "
                    f"{record['refcopilot_pointer']}"
                )
            continue
        decisions = {decision.candidate_id: decision for decision in review.results}
        for record, (_index, row, block) in zip(records, page_candidates, strict=True):
            decision = decisions[record["candidate_id"]]
            record.update(status=decision.classification, decision=decision.model_dump())
            if decision.classification != "manuscript_error":
                issues.append(
                    f"{record['candidate_id']}: reference mismatch {decision.classification}: "
                    f"{decision.reason}; {record['refcopilot_pointer']}"
                )
                continue
            printed, comparison = decision.printed_quote.strip(), decision.comparison_quote.strip()
            if not record["identity"]["qualified"]:
                record["status"] = "uncertain"
                issues.append(
                    f"{record['candidate_id']}: reference mismatch uncertain; {record['identity']['reason']} "
                    f"PDF observation ({decision.mismatch_kind}): {decision.reason}; {record['refcopilot_pointer']}"
                )
                continue
            expected_kinds = (
                {"wrong_author", "author_order"}
                if "author_mismatch" in str(row.get("type") or row.get("code") or "").lower()
                else {"title"}
            )
            if (
                not printed
                or not comparison
                or comparison not in record["comparison_metadata"]
                or _identity_text(printed) == _identity_text(comparison)
                or decision.mismatch_kind not in expected_kinds
            ):
                record["status"] = "uncertain"
                issues.append(
                    f"{record['candidate_id']}: reference mismatch uncertain; PDF confirmation lacks "
                    f"a visible differing quote and grounded comparison metadata; {record['refcopilot_pointer']}"
                )
                continue
            finding = paper_finding(
                materials,
                block,
                quote=block.text,
                text=f"{decision.mismatch_kind}: {decision.reason} "
                f"[Original PDF page {page_number}; identity: {record['identity']['identifier']}; "
                f"RefCopilot: {record['refcopilot_pointer']}]. "
                "Difference from the identified retrieved metadata; its external accuracy was not independently rechecked.",
                kind="reference",
                level=str(row.get("severity") or "unverified"),
            )
            finding.evidence.append(
                Evidence(
                    source="literature",
                    pointer=EvidencePointer(locator=record["identity"]["verified_url"], quote=comparison),
                    direction="flaw",
                    sufficient=False,
                    affects_claim=False,
                    note=f"Citation identity: {record['identity']['identifier']}; {record['identity']['reason']}",
                )
            )
            finding.evidence.append(
                Evidence(
                    source="paper_internal",
                    pointer=EvidencePointer(
                        locator=page.path, page=page_number, quote=printed, key=record["candidate_id"]
                    ),
                    direction="flaw",
                    sufficient=False,
                    affects_claim=False,
                    note=f"Original PDF read: {decision.reason}; retrieved comparison: {comparison}",
                )
            )
            if corrections and _index in corrections:
                finding = finding.model_copy(update={"reference_correction": corrections[_index]})
            findings.append(finding)
    if candidates:
        (output_dir / "reference_validation.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return findings, issues
