"""Section-scoped writing checks with explicit policy and original-PDF confirmation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from schemas.claim import Contract, Finding, NonEmpty
from schemas.materials import SharedMaterials
from screening.checks import ask, check_tables, paper_finding

AnonymityPolicy = Literal["unspecified", "required", "not_required"]


class WritingSectionRecord(Contract):
    section_id: str
    section: str | None = None
    block_ids: list[str] = Field(default_factory=list)
    status: Literal["checked", "failed", "unavailable"]
    candidate_count: int = Field(default=0, ge=0)
    confirmed_count: int = Field(default=0, ge=0)
    anonymity_policy: AnonymityPolicy = "unspecified"
    issues: list[str] = Field(default_factory=list)


class WritingCandidate(Contract):
    block_id: NonEmpty
    quote: str = Field(min_length=1)
    text: NonEmpty
    level: Literal["definite_error", "clarity_issue"]
    category: Literal["language", "cross_reference", "anonymity"] = "language"
    target_kind: Literal["figure", "table", "section", "appendix", "equation"] | None = None
    target_label: NonEmpty | None = None
    reference_problem: Literal["unresolved_placeholder", "missing_target", "inconsistent_target"] | None = (
        None
    )

    @model_validator(mode="after")
    def category_fields(self):
        values = (self.target_kind, self.target_label, self.reference_problem)
        if self.category == "cross_reference":
            if not all(values):
                raise ValueError("cross-reference candidates require target kind, label and problem")
        elif any(value is not None for value in values):
            raise ValueError("target fields require a cross-reference candidate")
        return self


class WritingDecision(Contract):
    candidate_id: NonEmpty
    classification: Literal[
        "manuscript_error",
        "clarity_issue",
        "parser_artifact",
        "style",
        "uncertain",
        "cross_reference_error",
        "anonymity_violation",
    ]
    explanation: NonEmpty


class WritingPageReview(Contract):
    results: list[WritingDecision]


_SYSTEM = (
    "Check this section for typos, grammar, consequential unclear sentences, internal cross-references, "
    "and the explicitly configured anonymity policy. Return JSON {findings: [candidate_schema objects]}. "
    "Quote an exact original sentence from one supplied block. Give a concrete correction or clarification. "
    "category is language, cross_reference, or anonymity. Language level is definite_error for a concrete "
    "typo or grammatical construction error, or clarity_issue for a specific ambiguity that changes "
    "scientific meaning or prevents understanding. Exclude discretionary style: optional hyphenation, "
    "capitalization, article conventions, rephrasing, concision, synonyms, and active/passive voice. "
    "A conventional dataset name plus split label is not automatically an article or grammar error. "
    "Explain the violated grammatical relation or the competing scientific interpretations; a preferred "
    "wording alone does not establish a defect. OCR may alter symbols, words or spacing; proposed errors "
    "require original PDF confirmation. Copy quotes without normalizing math or whitespace. "
    "Use the full-paper cross_reference_index for targets outside this section. Keep figure, table, section, "
    "appendix and equation namespaces distinct. A cross_reference candidate must state target_kind, "
    "target_label and reference_problem. The index is incomplete parser evidence; an absent target alone "
    "does not prove a missing manuscript object. unresolved_placeholder requires a visibly unexpanded "
    "reference; inconsistent_target requires a located target with a concrete conflicting description. "
    "For anonymity, only policy=required activates review: explicit manuscript authorship or direct "
    "self-identification may violate it. Cited authors, self-citations or institution/dataset names alone "
    "do not. unspecified and not_required permit no anonymity finding. Do not infer venue rules. "
    "policy_only_block_ids may be inspected solely for explicit anonymity violations, never language "
    "or cross-reference suggestions. Return findings=[] when no concrete applicable issue is found."
)

_VALIDATION_SYSTEM = (
    "Verify every supplied writing candidate against the attached original PDF page. Read the printed "
    "sentence, mathematical symbols, and surrounding context. Return output_schema JSON with exactly "
    "one result per candidate_id and no new IDs. Additional supplied pages show the original located "
    "cross-reference targets; compare those pixels as well. Confirm the proposed defect itself. Use manuscript_error "
    "only for a visible concrete typo or grammatical construction error. Explain the violated grammatical "
    "relation; conventional dataset/split names and optional articles do not alone establish an error. "
    "Use clarity_issue only for a specific ambiguity that materially changes scientific meaning or prevents "
    "understanding. Use parser_artifact for OCR errors or parsing-related spacing; style for discretionary "
    "hyphenation, capitalization, rephrasing, concision, synonyms, article conventions or voice preferences. "
    "For cross_reference candidates, use cross_reference_error only when the printed unresolved placeholder "
    "or concrete contradiction with the uniquely located supplied target is confirmed. Absence from a parser "
    "index cannot establish absence from the manuscript. For anonymity candidates use anonymity_violation "
    "only under policy=required with visible explicit authorship/self-identification; cited names or "
    "self-citations alone are insufficient. Each positive classification must match the candidate category. "
    "Use uncertain when the page/context cannot establish the specific issue. A parsed quote alone cannot "
    "confirm an error. Explain visible evidence and preserve the original material."
)


def _reference_index(materials):
    """Expose located original targets; this directory never claims complete parsing."""
    entries = []

    def add(kind, label, quote, loc, block_id=None, ambiguous=False):
        if not quote:
            return
        entries.append(
            {
                "kind": kind,
                "label": label,
                "quote": quote,
                "loc": loc.model_dump() if loc else None,
                "block_id": block_id,
                "ambiguous": ambiguous or not label or loc is None,
            }
        )

    for block in materials.blocks:
        if block.kind == "heading":
            match = re.match(r"\s*(?:section\s+)?(\d+(?:\.\d+)*)(?=[\s.:]|$)", block.text, re.I)
            appendix = re.match(r"\s*appendix\s+([A-Za-z0-9]+)\b", block.text, re.I)
            if match:
                add("section", match[1], block.text, block.loc, block.id)
            elif appendix:
                add("appendix", appendix[1], block.text, block.loc, block.id)
            else:
                add("section", block.text, block.text, block.loc, block.id)
        elif block.kind == "equation":
            match = re.search(r"\\tag\{([A-Za-z0-9.]+)\}", block.text)
            if match:
                add("equation", match[1], block.text, block.loc, block.id)
    for figure in materials.figures:
        add("figure", figure.anchor, figure.caption, figure.loc, ambiguous=figure.caption_ambiguous)
    represented = set()
    for table in materials.tables:
        represented.add(table.block_id)
        add("table", table.anchor, table.caption, table.loc, table.block_id, table.caption_ambiguous)
    for block in materials.blocks:
        if block.kind == "table" and block.id not in represented:
            match = re.match(r"\s*(?:Table|Tab\.)\s+([A-Za-z]?\d+[A-Za-z]?)\b", block.text, re.I)
            if match:
                add("table", match[1], block.text, block.loc, block.id)
    return {
        "complete": False,
        "entries": entries,
        "issues": list(materials.issues),
        "limit": "Located parser targets only; absence does not prove a missing manuscript target.",
    }


def _sections(materials, policy):
    groups = {}
    bibliography = {block.id for block in materials.bibliography}
    for block in materials.blocks:
        if block.id in bibliography or block.kind == "ref_text":
            continue
        language = block.kind in {"text", "list"}
        policy_only = policy == "required" and block.kind in {"heading", "author", "authors", "affiliation"}
        if not (language or policy_only):
            continue
        label = block.loc.section if block.loc else None
        group = groups.setdefault(label, {"blocks": [], "policy_only": []})
        group["blocks"].append(block)
        if policy_only:
            group["policy_only"].append(block.id)
    return groups


def _reference_context(row, index):
    """Bind the claimed reference to the original quote, preserving namespace."""
    kind = {
        "figure": r"(?:fig(?:ure)?\.?s?)",
        "table": r"(?:tab(?:le)?\.?s?)",
        "section": r"(?:sec(?:tion)?\.?)",
        "appendix": "appendix",
        "equation": r"(?:eq(?:uation)?\.?)",
    }[row.target_kind]
    label = re.escape(row.target_label)
    if row.target_kind == "equation":
        # Equation labels conventionally use a balanced pair of parentheses.
        # Keep the plain form, while rejecting partial or nested delimiters.
        label = rf"(?:\(\s*{label}\s*\)|{label})(?![\w()]|\.\d)"
    else:
        label += r"(?!\w|\.\d)"
    literal = re.search(rf"\b{kind}\s+{label}", row.quote, re.I)
    placeholder = bool(literal and row.target_label == "??") or any(
        match[1] == row.target_label
        for match in re.finditer(r"\\(?:ref|eqref|autoref)\{([^{}]+)\}", row.quote)
    )
    if not literal and not (row.reference_problem == "unresolved_placeholder" and placeholder):
        raise ValueError("cross-reference target is not bound to the quoted original sentence")
    if row.reference_problem == "unresolved_placeholder":
        if not placeholder:
            raise ValueError("unresolved reference candidate has no literal placeholder")
        return [], None
    matches = [
        entry
        for entry in index["entries"]
        if entry["kind"] == row.target_kind and entry["label"].casefold() == row.target_label.casefold()
    ]
    if row.reference_problem == "missing_target":
        return matches, "Parser directory absence cannot confirm a missing manuscript target."
    if len(matches) != 1 or matches[0]["ambiguous"] or not matches[0]["loc"]:
        return matches, "Cross-reference target is missing or ambiguous in the supplied directory."
    return matches, None


def _check_section(materials, group, record, index, *, call, first_candidate):
    blocks = {block.id: block for block in group["blocks"]}
    if len(blocks) != len(group["blocks"]):
        raise ValueError("writing section has duplicate block identities")
    if any(block.loc is None for block in group["blocks"]):
        record.status = "unavailable"
        record.issues.append(
            "Section includes unlocated text; original-source confirmation is unavailable for those blocks."
        )
    result = ask(
        _SYSTEM,
        {
            "section_id": record.section_id,
            "section": record.section,
            "blocks": [block.model_dump() for block in group["blocks"]],
            "policy_only_block_ids": group["policy_only"],
            "anonymity_policy": record.anonymity_policy,
            "cross_reference_index": index,
            "candidate_schema": WritingCandidate.model_json_schema(),
        },
        module="screening_writing",
        call=call,
    )
    if not isinstance(result.get("findings"), list):
        raise ValueError("writing response must contain a findings list")
    record.candidate_count = len(result["findings"])
    candidates = []
    for offset, raw in enumerate(result["findings"]):
        row = WritingCandidate.model_validate(raw)
        if row.block_id not in blocks:
            raise ValueError("writing candidate must quote a block in its own section")
        if row.block_id in group["policy_only"] and row.category != "anonymity":
            raise ValueError("policy-only blocks cannot authorize language/cross-reference findings")
        if blocks[row.block_id].text.count(row.quote) > 1:
            raise ValueError("writing quote must identify one exact sentence occurrence in its block")
        finding = paper_finding(
            materials, blocks[row.block_id], quote=row.quote, text=row.text, kind="writing", level=row.level
        )
        candidate_id = f"writing_{first_candidate + offset}"
        targets = []
        if row.category == "anonymity" and record.anonymity_policy != "required":
            record.issues.append(
                f"{candidate_id}: anonymity check not applicable under policy={record.anonymity_policy}."
            )
            continue
        if row.category == "cross_reference":
            targets, limitation = _reference_context(row, index)
            if limitation:
                record.issues.append(f"{candidate_id}: unconfirmed; {limitation}")
                record.status = "unavailable"
                continue
        candidates.append((candidate_id, row, finding, targets))
    # No image request occurs until every candidate in this section has validated.
    by_page = {}
    for candidate in candidates:
        by_page.setdefault(candidate[2].loc.page, []).append(candidate)
    findings = []
    for page_number, page_candidates in by_page.items():
        page = next((page for page in materials.pages if page.page == page_number), None)
        if page is None or not Path(page.path).is_file():
            record.issues.append(
                f"writing page {page_number}: original PDF page image unavailable; "
                f"unconfirmed candidates {[c[0] for c in page_candidates]}"
            )
            if record.status != "failed":
                record.status = "unavailable"
            continue
        image_paths, target_pages = [page.path], []
        missing_target_page = False
        for _candidate_id, _row, _finding, targets in page_candidates:
            for target in targets:
                target_page_number = target["loc"]["page"]
                target_page = next((p for p in materials.pages if p.page == target_page_number), None)
                if target_page is None or not Path(target_page.path).is_file():
                    record.issues.append(
                        f"writing page {page_number}: original cross-reference target page {target_page_number} unavailable"
                    )
                    missing_target_page = True
                    continue
                if target_page.path not in image_paths:
                    image_paths.append(target_page.path)
                    target_pages.append(target_page_number)
        if missing_target_page:
            if record.status != "failed":
                record.status = "unavailable"
            continue
        try:
            review = WritingPageReview.model_validate(
                ask(
                    _VALIDATION_SYSTEM,
                    {
                        "page": page_number,
                        "anonymity_policy": record.anonymity_policy,
                        "additional_target_pages": target_pages,
                        "candidates": [
                            {"candidate_id": candidate_id, **row.model_dump(), "targets": targets}
                            for candidate_id, row, _finding, targets in page_candidates
                        ],
                        "output_schema": WritingPageReview.model_json_schema(),
                    },
                    module="screening_writing.validation",
                    call=call,
                    images=image_paths,
                )
            )
            expected = {candidate_id for candidate_id, *_ in page_candidates}
            received = [item.candidate_id for item in review.results]
            if len(received) != len(set(received)) or set(received) != expected:
                raise ValueError(
                    f"Writing validation must cover each candidate exactly once; received={received!r}, expected={sorted(expected)!r}"
                )
            decisions = {item.candidate_id: item for item in review.results}
            positive = {
                "language": {"manuscript_error", "clarity_issue"},
                "cross_reference": {"cross_reference_error"},
                "anonymity": {"anonymity_violation"},
            }
            all_positive = set().union(*positive.values())
            for candidate_id, row, _finding, _targets in page_candidates:
                classification = decisions[candidate_id].classification
                if classification in all_positive and classification not in positive[row.category]:
                    raise ValueError("Writing validation classification does not match candidate category")
        except Exception as exc:
            record.issues.append(f"writing page {page_number}: original PDF validation failed: {exc}")
            record.status = "failed"
            continue
        for candidate_id, row, finding, targets in page_candidates:
            decision = decisions[candidate_id]
            if decision.classification not in positive[row.category]:
                record.issues.append(
                    f"{candidate_id} ({row.block_id}, page {page_number}): "
                    f"{decision.classification}; {decision.explanation}"
                )
                if decision.classification == "uncertain" and record.status != "failed":
                    record.status = "unavailable"
                continue
            finding.level = (
                "clarity_issue" if decision.classification == "clarity_issue" else "definite_error"
            )
            finding.evidence[0].note += f"; original PDF page {page_number} confirmed: {decision.explanation}"
            if row.category != "language":
                finding.evidence[
                    0
                ].note += f"; category={row.category}; anonymity_policy={record.anonymity_policy}"
            for target in targets:
                finding.evidence[
                    0
                ].note += f"; cross-reference target {target['kind']} {target['label']}: {target['quote']} ({target['loc']})"
            findings.append(finding)
    # A failed section never contributes a partial batch of confirmed findings.
    if record.status == "failed":
        return []
    record.confirmed_count = len(findings)
    return findings


def check_writing(
    materials: SharedMaterials,
    *,
    call=None,
    issues: list[str] | None = None,
    records: list[WritingSectionRecord] | None = None,
    anonymity_policy: AnonymityPolicy = "unspecified",
    recover_errors: bool = False,
) -> list[Finding]:
    """Inspect each original section; production may recover between sections."""
    if anonymity_policy not in {"unspecified", "required", "not_required"}:
        raise ValueError("anonymity_policy must be unspecified, required or not_required")
    issues = [] if issues is None else issues
    records = [] if records is None else records
    groups = _sections(materials, anonymity_policy)
    if not groups:
        issue = "Writing check unavailable: no applicable manuscript text blocks."
        issues.append(issue)
        records.append(
            WritingSectionRecord(
                section_id="writing_empty",
                status="unavailable",
                anonymity_policy=anonymity_policy,
                issues=[issue],
            )
        )
        return []
    index = _reference_index(materials)
    findings, first_candidate = [], 1
    for section, group in groups.items():
        identity = "unsectioned" if section is None else "section:" + section
        record = WritingSectionRecord(
            section_id="writing_" + hashlib.sha256(identity.encode()).hexdigest()[:16],
            section=section,
            block_ids=[b.id for b in group["blocks"]],
            status="checked",
            anonymity_policy=anonymity_policy,
        )
        try:
            findings.extend(
                _check_section(materials, group, record, index, call=call, first_candidate=first_candidate)
            )
        except Exception as exc:
            record.status = "failed"
            record.issues.append(f"writing check failed for section {section!r}: {exc}")
            if not recover_errors:
                raise
        finally:
            first_candidate += record.candidate_count
            records.append(record)
            issues.extend(record.issues)
    return findings


__all__ = ["WritingSectionRecord", "check_tables", "check_writing"]
