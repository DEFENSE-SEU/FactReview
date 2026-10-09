"""Visual table screening, retaining parsed-text checks as a separate boundary."""

import copy
from typing import Literal

from pydantic import Field

from schemas.claim import Contract, Evidence, EvidencePointer, Finding
from schemas.materials import SharedMaterials, TableMaterial
from screening.checks import ask
from screening.figure_context import safe_response
from screening.figures import _printed_size_verified
from screening.table_context import TableSources, confirm

CATEGORIES = {"self_containedness", "legibility", "text_table_consistency"}


class TableCheckRecord(Contract):
    table_id: str
    status: Literal["checked", "failed", "unavailable"]
    finding_count: int = Field(default=0, ge=0)
    printed_size_verified: bool = False
    issues: list[str] = Field(default_factory=list)
    crop_response: dict = Field(default_factory=dict)
    context_status: Literal["not_requested", "checked", "failed", "unavailable"] = "not_requested"
    context_record: dict = Field(default_factory=dict)


def check_visual_tables(
    materials: SharedMaterials,
    *,
    call=None,
    recover_errors: bool = False,
    records: list[TableCheckRecord] | None = None,
) -> tuple[list[Finding], list[str]]:
    """Retain crop legibility separately from source-bound original-page decisions."""
    findings, issues = [], []
    if records is None:
        records = []
    cache, pending = {}, []
    frozen = [(t.model_copy(deep=True), TableSources.capture(materials, t, cache)) for t in materials.tables]
    for table, guard in frozen:
        table_issues = list(table.issues)
        record = TableCheckRecord(table_id=table.id, status="checked")
        if not table.printed_crop_path or not guard.hashes.get(table.printed_crop_path) or table.loc is None:
            record.status = "unavailable"
            table_issues.append(f"{table.id}: table visual check unavailable; crop or location missing")
            record.issues = table_issues
            records.append(record)
            issues.extend(table_issues)
            continue
        try:
            guard.check(materials)
        except Exception as exc:
            if not recover_errors:
                raise
            record.status = "failed"
            table_issues.append(f"table visual check failed for {table.id}: {exc}")
            record.issues = table_issues
            records.append(record)
            issues.extend(table_issues)
            continue
        try:
            verified = _printed_size_verified(table)
        except (OSError, ValueError, OverflowError) as exc:
            record.status = "unavailable"
            table_issues.append(
                f"{table.id}: table visual check unavailable; invalid printed-size input: {exc}"
            )
            record.issues = table_issues
            records.append(record)
            issues.extend(table_issues)
            continue
        try:
            guard.check(materials)
            record.printed_size_verified = verified
            context = guard.prepare(materials)
            raw = _crop_response(table, call=call, printed_size_verified=verified)
            record.crop_response = copy.deepcopy(raw)
            guard.check(materials)
            table_findings, observed_issues = _crop_findings(
                table, {"findings": [r for r in raw["findings"] if r["category"] == "legibility"]}
            )
            table_issues.extend(observed_issues)
            candidates = [
                {**row, "candidate_id": f"candidate_{i}"}
                for i, row in enumerate(raw["findings"])
                if row["category"] != "legibility" and row["disposition"] != "clear"
            ]
            if candidates or table.caption_ambiguous or not table.caption:
                guard.check(materials, context=context is not None)
                if context is None:
                    record.context_status = "unavailable"
                    table_issues.append(
                        f"{table.id}: original-page context unavailable: {guard.context_error}"
                    )
                else:
                    try:
                        response, sidecar = confirm(context, candidates, call=call)
                        guard.check(materials, context=True)
                        guard.context_consumed = True
                        record.context_status, record.context_record = "checked", sidecar
                        if sidecar["caption_assignment"] != "confirmed":
                            table_issues.append(
                                f"{table.id}: original-page target/caption assignment remains uncertain."
                            )
                        by_id = {c["candidate_id"]: c for c in candidates}
                        spans = {s["id"]: s for s in context["page_spans"]}
                        caption_spans = [spans[sid] for sid in context["caption_source"]["page_span_ids"]]
                        box = table.bbox_points
                        caption_in_crop = all(
                            box[0] <= span["bbox_points"][0] < span["bbox_points"][2] <= box[2]
                            and box[1] <= span["bbox_points"][1] < span["bbox_points"][3] <= box[3]
                            for span in caption_spans
                        )
                        for decision in response.decisions:
                            candidate = by_id[decision.candidate_id]
                            if decision.classification != "manuscript_issue":
                                table_issues.append(
                                    f"{table.id}: {candidate['category']} {decision.classification}: {decision.reason}; original crop observation: {candidate['text']}"
                                )
                                continue
                            evidence = [
                                Evidence(
                                    source="paper_internal",
                                    direction="flaw",
                                    sufficient=False,
                                    affects_claim=False,
                                    pointer=EvidencePointer(
                                        locator=table.printed_crop_path
                                        if caption_in_crop
                                        else materials.source_pdf,
                                        quote="\n".join(span["text"] for span in caption_spans),
                                        page=table.loc.page,
                                        key=table.id,
                                    ),
                                    note=f"Original crop candidate: {candidate['text']}; crop={table.printed_crop_path}",
                                )
                            ]
                            for sid in decision.witness_span_ids:
                                evidence.append(
                                    Evidence(
                                        source="paper_internal",
                                        direction="flaw",
                                        sufficient=False,
                                        affects_claim=False,
                                        pointer=EvidencePointer(
                                            locator=materials.source_pdf,
                                            page=table.loc.page,
                                            key=f"{table.id}:bbox={table.bbox_points}:{sid}",
                                            quote=spans[sid]["text"],
                                        ),
                                        note=f"Original-page visual confirmation; context_id={context['context_id']}; model visual interpretation.",
                                    )
                                )
                            table_findings.append(
                                Finding(
                                    kind="table",
                                    loc=table.loc,
                                    level=candidate["category"],
                                    text=f"{candidate['text']} Original-page confirmation: {decision.reason}",
                                    evidence=evidence,
                                )
                            )
                    except Exception as exc:
                        record.context_status = "failed"
                        table_issues.append(f"{table.id}: original-page context failed: {exc}")
                    guard.check(materials, context=True)
                if record.context_status != "checked":
                    for candidate in candidates:
                        state = (
                            "check uncertain"
                            if candidate["disposition"] == "uncertain"
                            else (
                                "unconfirmed because parser caption assignment is ambiguous"
                                if table.caption_ambiguous
                                else "unconfirmed original crop observation"
                            )
                        )
                        table_issues.append(
                            f"{table.id}: {candidate['category']} {state}: {candidate['text']}"
                        )
            guard.check(materials)
            if not verified:
                table_issues.append(
                    f"{table.id}: printed-size legibility unavailable because physical bounding box metadata is missing."
                )
                for finding in table_findings:
                    if finding.level == "legibility":
                        table_issues.append(
                            f"{table.id}: legibility check uncertain because physical printed size is unverified: {finding.text}"
                        )
                table_findings = [f for f in table_findings if f.level != "legibility"]
        except Exception as exc:
            if not recover_errors:
                raise
            record.status = "failed"
            if record.context_status == "checked":
                record.context_status = "failed"
                record.context_record["source_integrity"] = "failed"
            table_issues.append(f"table visual check failed for {table.id}: {exc}")
            record.issues = table_issues
            records.append(record)
            issues.extend(table_issues)
            continue
        record.finding_count, record.issues = len(table_findings), table_issues
        records.append(record)
        pending.append((guard, record, table_findings))
    for guard, record, table_findings in pending:
        try:
            guard.check(materials)
        except Exception as exc:
            if not recover_errors:
                raise
            record.status, record.finding_count = "failed", 0
            if record.context_status == "checked":
                record.context_status = "failed"
                record.context_record["source_integrity"] = "failed"
            record.issues.append(f"{record.table_id}: final source integrity check failed: {exc}")
        else:
            findings.extend(table_findings)
        issues.extend(record.issues)
    return findings, issues


def _check_table(table: TableMaterial, *, call=None, printed_size_verified=False):
    return _crop_findings(
        table, _crop_response(table, call=call, printed_size_verified=printed_size_verified)
    )


def _crop_response(table, *, call=None, printed_size_verified=False):
    result = ask(
        "Inspect this original table crop, its caption, supplied table footnotes, and EVERY body reference. "
        "Use the pixels to check self_containedness (row/column headers, units, meanings of bold, "
        "asterisks or other marks), legibility, and text_table_consistency. Interpret marks using "
        "the visible table, caption, table footnotes and supplied context; report unexplained marks only when their "
        "meaning is ambiguous. Check concrete agreements or contradictions between this table and "
        "its caption/body references. Do not assess scientific claim validity or numerical significance. "
        "When printed_size_verified is true, judge legibility at the attached physical printed size "
        "of 96 dpi. When false, mark legibility uncertain. When caption_ambiguous is true, "
        "caption-dependent self_containedness and text_table_consistency are uncertain. "
        "The parsed assignment cannot establish a manuscript defect. Do not infer defects from "
        "style or aesthetics. Return JSON {findings: [{category, disposition, text}]}; category is "
        "self_containedness, legibility, or text_table_consistency; disposition is issue, clear, "
        "or uncertain. Give concrete visible evidence in text. Use clear for positive observations, "
        "uncertain for insufficient pixels/context, and findings=[] when no defect or uncertainty is found.",
        {
            "table_id": table.id,
            "block_id": table.block_id,
            "caption": table.caption,
            "footnotes": table.footnotes,
            "caption_ambiguous": table.caption_ambiguous,
            "references": [block.model_dump() for block in table.references],
            "printed_dpi": table.printed_dpi,
            "printed_size_verified": printed_size_verified,
        },
        module="screening_tables.visual",
        call=call,
        images=[table.printed_crop_path],
    )
    if not isinstance(result.get("findings"), list):
        raise ValueError("table visual response must contain a findings list")
    for row in result["findings"]:
        if not isinstance(row, dict) or row.get("category") not in CATEGORIES:
            raise ValueError("table visual findings require one of the three specified categories")
        if row.get("disposition") not in {"issue", "clear", "uncertain"}:
            raise ValueError("table visual disposition must be issue, clear, or uncertain")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("table visual findings require a nonempty explanation")
    return safe_response(result)


def _crop_findings(table, result):
    findings, issues = [], []
    for row in result["findings"]:
        if row["disposition"] == "clear":
            continue
        if row["disposition"] == "uncertain":
            issues.append(f"{table.id}: {row['category']} check uncertain: {row['text']}")
            continue
        if table.caption_ambiguous and row["category"] in {"self_containedness", "text_table_consistency"}:
            issues.append(
                f"{table.id}: {row['category']} unconfirmed because parser caption assignment is ambiguous: {row['text']}"
            )
            continue
        quote = table.caption or next((reference.text for reference in table.references), "")
        if not quote:
            issues.append(f"{table.id}: {row['text']} (caption/body reference unavailable)")
            continue
        findings.append(
            Finding(
                kind="table",
                loc=table.loc,
                level=row["category"],
                text=row["text"],
                evidence=[
                    Evidence(
                        source="paper_internal",
                        pointer=EvidencePointer(
                            locator=table.printed_crop_path, quote=quote, page=table.loc.page, key=table.id
                        ),
                        direction="flaw",
                        sufficient=False,
                        note=row["text"],
                        affects_claim=False,
                    )
                ],
            )
        )
    return findings, issues
