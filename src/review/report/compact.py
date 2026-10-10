"""An opt-in reading view of a checked review, with a complete linked appendix.

This module selects presentation entries by recorded metadata. It neither assesses
claims nor generates advice. The JSON and the appendix retain every original record.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from review.report import v2


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _anchor(path, *, main=False):
    return "factreview-" + ("main-" if main else "record-") + hashlib.sha256(path.encode()).hexdigest()


def _json(value):
    return v2._text(json.dumps(value, ensure_ascii=False))


def _paths(value, path=""):
    yield path
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _paths(child, path + "/" + escaped)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _paths(child, path + f"/{index}")


def _nearest(path, targets):
    while path not in targets:
        path = path.rsplit("/", 1)[0]
    return targets[path]


class _Navigation:
    def __init__(self, review):
        self.claim_paths = {id(claim): f"/claims/{i}" for i, claim in enumerate(review.claims)}
        self.appendix = {"": _anchor("")}
        self.main = {"": _anchor("", main=True)}

    def marker(self, path, main_path=None):
        anchor = self.appendix.setdefault(path, _anchor(path))
        main_target = _nearest(main_path if main_path is not None else path, self.main)
        # Prefix on a real paragraph lets the existing PDF parser keep the target
        # with its actual content, including split paragraphs and CJK text.
        return f'<a id="{anchor}"></a>[Main](final_review.md#{main_target})'

    def main_marker(self, path):
        return f'<a id="{self.main.setdefault(path, _anchor(path, main=True))}"></a>'

    def link(self, path, label="Full record"):
        target = _nearest(path, self.appendix)
        return f"[{label}](technical_appendix.md#{target})"

    def __call__(self, kind, owner, index=None):
        if kind == "claim":
            path = self.claim_paths[id(owner)]
        elif kind in {"finding", "finding_evidence"}:
            path = f"/findings/{owner}"
            if kind == "finding_evidence":
                path += f"/evidence/{index}"
        elif kind == "ledger":
            path = f"/ledger/{owner}"
        elif kind == "issues":
            path = "/delivery_context"
        else:
            path = self.claim_paths[id(owner)]
            if kind == "primary":
                path += "/source_quote"
                return [
                    "",
                    self.marker(path),
                    "Original manuscript primary source: "
                    + v2._text(owner.source_block_id or "block unavailable")
                    + "; "
                    + (v2._location(owner.loc) or "location unavailable")
                    + ".",
                    "Original quote: " + v2._text(owner.source_quote or "not recorded"),
                    "",
                ]
            path += f"/{kind}" + (f"/{index}" if index is not None else "")
        return [self.marker(path)]


def _attach_markers(markdown):
    """Attach our generated record markers to the following content paragraph."""
    result, pending = [], []
    pattern = re.compile(r'^<a id="factreview-record-[a-f0-9]{64}"></a>\[Main\]')
    for line in markdown.splitlines():
        if pattern.match(line):
            pending.append(line)
            continue
        if pending and line.strip():
            line += " " + " ".join(pending)
            pending = []
        result.append(line)
    if pending:
        result.append("Record navigation: " + " ".join(pending))
    return "\n".join(result).rstrip() + "\n"


def _selection(evidence):
    """Keep decisive entries and each direction/source/condition group's first entry."""
    groups = {}
    reasons = {}
    for index, item in enumerate(evidence):
        if item.affects_claim and item.sufficient:
            reasons[index] = ["recorded sufficient and affects_claim"]
        for condition in item.covered or [None]:
            key = (condition, item.direction, item.source, item.concern)
            groups.setdefault(key, []).append(index)
    for key, indices in groups.items():
        reasons.setdefault(indices[0], []).append("first in original order for group " + json.dumps(key))
    return reasons, [
        {
            "condition": key[0],
            "direction": key[1],
            "source": key[2],
            "concern": key[3],
            "count": len(indices),
            "representative_index": indices[0],
            "indices": indices,
        }
        for key, indices in groups.items()
    ]


def _evidence_entry(item, path, nav, reasons, *, neutral=False):
    direction = "reference source; discrepancy unconfirmed" if neutral else item.direction
    lines = [
        f"- {nav.main_marker(path)}**{v2._text(item.source)} / {v2._text(direction)}**; "
        f"covers: {v2._text(', '.join(item.covered) or 'no claim conditions')}; "
        f"sufficient: {str(item.sufficient).lower()}; affects claim: {str(item.affects_claim).lower()}; "
        f"concern: {str(item.concern).lower()}; aligned: {_json(item.aligned)}. {nav.link(path)}.",
        f"  - Primary: {v2._pointer_location(item.pointer)}. {nav.link(path + '/pointer', 'Passage')}.",
    ]
    lines.extend(
        f"  - Additional {i}: {v2._pointer_location(pointer)}. "
        + nav.link(path + f"/additional_pointers/{i - 1}", "Passage")
        + "."
        for i, pointer in enumerate(item.additional_pointers, 1)
    )
    lines.append("  - Display selection: " + v2._text("; ".join(reasons)) + ".")
    return lines


def _compact_markdown(review, nav, context):
    lines = [
        f"# FactReview reading report — {v2._text(review.paper_key)}",
        "",
        f"{nav.main_marker('')}Full records: {nav.link('', 'technical appendix')}. "
        "Package navigation: review_bundle.pdf; export availability is recorded in report_manifest.json. "
        "Evidence entries below are selected by recorded metadata and original order; judgments are unchanged.",
        "",
        "## 1. Overview",
        "",
        f"Run: {v2._text(review.run_id)}; delivery: {review.run_status}; "
        f"execution requested: {_json(review.execution_requested)}.",
        f"Incomplete stages: {v2._text(', '.join(review.incomplete_stages) or 'none recorded')}.",
        "Counts describe retained records. Missing checks cannot establish an absence of paper problems.",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {s.value} | {review.summary_counts[s]} |" for s in v2.STATUS_ORDER)
    from review.delivery import delivery_lines

    lines += delivery_lines(review)
    lines += v2.claim_coverage_lines(context.get("claim_coverage"))
    lines += ["", "Execution coverage: " + _json(v2.execution_summary(review)) + ".", ""]
    for key in (
        "writing_coverage",
        "figure_coverage",
        "figure_context_coverage",
        "table_coverage",
        "table_context_coverage",
        "anonymity_policy",
    ):
        if context.get(key) is not None:
            lines.append(f"- {v2._text(key)}: {_json(context[key])}.")
    usage = context.get("token_usage")
    if usage is not None:
        keys = (
            "requests",
            "failed_requests",
            "unavailable_usage_requests",
            "image_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated",
            "unavailable",
        )
        lines += ["", "Model call accounting: " + _json({k: usage[k] for k in keys if k in usage}) + "."]
    # Generated coverage/accounting limitations stay visible; arbitrary diagnostics
    # retain their exact strings in the appendix without a guessed severity.
    known_context = {**context, "issues": None}
    if usage is not None:
        known_context["token_usage"] = {k: v for k, v in usage.items() if k != "warnings"}
    known = v2.verification_limitations(**known_context)
    if known:
        lines += ["", "Coverage and accounting limitations:", *["- " + v2._text(x) for x in known]]
    lines += [
        "",
        f"Other delivery diagnostics: {len(context.get('issues') or [])}. "
        + nav.link("/delivery_context", "Complete delivery context")
        + ".",
        "",
    ]
    if usage is not None:
        warnings = usage.get("warnings", [])
        count = str(len(warnings)) if isinstance(warnings, list) else "unavailable (non-array record)"
        lines += [
            f"Accounting warnings: {count}. "
            + nav.link("/delivery_context", "Complete warning records")
            + ".",
            "",
        ]
    states = {
        state: sum(c.advice is not None and c.advice.state == state for c in review.claims)
        for state in ("generated", "unavailable")
    }
    states["not_recorded"] = sum(c.advice is None for c in review.claims)
    lines += ["Advice delivery: " + _json(states) + ".", "", "Claim index:", ""]
    for claim in v2.ordered_claims(review):
        path = nav.claim_paths[id(claim)]
        lines.append(
            f"- [{v2._text(claim.id)}](#{nav.main[path]}) — {claim.status.value}, {claim.importance}."
        )
    lines += ["", "## 2. Claim list", ""]
    selections = {}
    for claim in v2.ordered_claims(review):
        path = nav.claim_paths[id(claim)]
        lines += [
            f"### {nav.main_marker(path)}{v2._text(claim.id)} — {claim.status.value}",
            "",
            v2._text(claim.text),
            "",
            f"Location: {v2._location(claim.loc) or 'unavailable'}; "
            f"block: {v2._text(claim.source_block_id or 'unavailable')}; importance: {claim.importance}; "
            f"needs: {v2._text(', '.join(claim.needs) or 'none')}. {nav.link(path + '/source_quote', 'Original source')}.",
            "",
            "Conditions:",
            "",
        ]
        for i, condition in enumerate(claim.conditions):
            data = {
                k: v
                for k, v in condition.model_dump(mode="json").items()
                if v is not None and v != {} and v != []
            }
            lines.append(f"- {_json(data)}. {nav.link(path + f'/conditions/{i}')}.")
        for i, source in enumerate(claim.source_refs):
            lines.append(
                f"- Original source {v2._text(source.source_block_id)}; {v2._location(source.loc)}; "
                f"covers {v2._text(', '.join(source.covered))}. {nav.link(path + f'/source_refs/{i}')} ."
            )
        lines.extend(v2._code_joint_integrity(claim))
        reasons, groups = _selection(claim.evidence)
        selections[path] = {"groups": groups, "selected": {str(i): why for i, why in reasons.items()}}
        lines += [
            "",
            f"Evidence: {len(claim.evidence)} total; {len(reasons)} displayed entries. "
            + nav.link(path, "All evidence and diagnostics")
            + ".",
            "",
        ]
        for group in groups:
            index = group["representative_index"]
            lines.append(
                "- Group: "
                + _json({k: v for k, v in group.items() if k != "indices"})
                + ". "
                + nav.link(path + f"/evidence/{index}", "Representative")
                + "."
            )
        for index in sorted(reasons):
            lines.extend(
                _evidence_entry(claim.evidence[index], path + f"/evidence/{index}", nav, reasons[index])
            )
        if claim.verification_limitations:
            lines += ["", "System verification limitations:", ""]
            for index, limitation in enumerate(claim.verification_limitations):
                lines.append(
                    "- "
                    + _json(limitation.model_dump(mode="json"))
                    + ". "
                    + nav.link(path + f"/verification_limitations/{index}")
                    + "."
                )
        lines += ["", "Questions for authors:", ""]
        lines.extend("- " + v2._text(q.text) + " Reason: " + v2._text(q.reason) for q in claim.questions)
        if not claim.questions:
            lines.append("None recorded.")
        lines += ["", "Reviewer advice:", ""]
        if claim.advice is None:
            lines.append("No generated advice is recorded.")
        elif claim.advice.state == "unavailable":
            lines.append(
                "Advice unavailable: "
                + v2._text(claim.advice.failure_reason)
                + ". "
                + nav.link(path + "/advice")
            )
        else:
            for item in claim.advice.items:
                lines.append(
                    f"- {v2._text(item.text)} Conditions: {v2._text(', '.join(item.condition_ids))}; "
                    f"action: {v2._text(item.action)}."
                )
                for ref in item.basis_refs:
                    target = path + ref if ref.startswith("/evidence/") else path + "/advice"
                    lines.append("  - Basis " + v2._text(ref) + ": " + nav.link(target) + ".")
            lines.append(
                "Advice wording is a model interpretation; references and input consistency were checked."
            )
        if claim.advice and claim.advice.audit_pointer:
            lines.append("Advice audit: " + v2._text(claim.advice.audit_pointer) + ".")
        lines += [
            "",
            f"Retained notes: {len(claim.notes)}; Theory records: {len(claim.theory_derivations)}. "
            + nav.link(path, "Full diagnostic and derivation records")
            + ".",
            "",
        ]
    lines += ["## 3. Other findings", ""]
    if not review.findings:
        lines.append("No additional findings recorded.")
    for index, finding in enumerate(review.findings):
        path = f"/findings/{index}"
        lines += [
            f"### {nav.main_marker(path)}{finding.kind} — {v2._text(finding.level)}",
            "",
            v2._text(finding.text),
            "",
            "Location: " + v2._location(finding.loc) + ". " + nav.link(path) + ".",
            "",
        ]
        reasons, groups = _selection(finding.evidence)
        selections[path] = {"groups": groups, "selected": {str(i): why for i, why in reasons.items()}}
        lines.append(f"Evidence: {len(finding.evidence)} total; {len(reasons)} displayed entries.")
        for i in sorted(reasons):
            item = finding.evidence[i]
            neutral = (
                finding.kind == "reference"
                and finding.level == "metadata_candidate"
                and not (item.affects_claim or item.sufficient or item.covered)
            )
            lines.extend(_evidence_entry(item, path + f"/evidence/{i}", nav, reasons[i], neutral=neutral))
        if finding.reference_correction:
            lines.append(
                "Reference metadata suggestion: "
                + v2._text(finding.reference_correction.state)
                + ". "
                + v2._text(finding.reference_correction.reason)
                + ". "
                + nav.link(path, "Complete metadata and provenance")
                + "."
            )
        lines.append("")
    lines += ["## 4. Execution ledger", ""]
    if not review.ledger:
        lines.append(
            "No complete execution records are retained."
            if "execution" in review.incomplete_stages
            else "Execution was not run; no new execution evidence was produced."
        )
    for index, entry in enumerate(review.ledger):
        path = f"/ledger/{index}"
        plan = entry.get("plan") if isinstance(entry.get("plan"), dict) else {}
        keys = ("approval_mode", "approved", "reason", "training_budget", "training_used_before")
        summary = {key: entry[key] for key in keys if key in entry}
        summary["plan"] = {
            k: plan[k] for k in ("id", "claim_id", "task", "run_mode", "feasibility", "blocker") if k in plan
        }
        attempts = entry.get("attempts")
        summary["recorded_attempts"] = len(attempts) if isinstance(attempts, list) else None
        lines += [
            f"### {nav.main_marker(path)}Run {index + 1}",
            "",
            _json(summary),
            "",
            nav.link(path, "Complete ledger") + ".",
        ]
        if isinstance(attempts, list):
            for attempt_index, attempt in enumerate(attempts, 1):
                if isinstance(attempt, dict):
                    fields = (
                        "status",
                        "success",
                        "returncode",
                        "exit_code",
                        "issue",
                        "reason",
                        "run_mode",
                        "resource_mode",
                        "model_inference_performed",
                        "measurement",
                    )
                    lines.append(
                        f"- Attempt {attempt_index}: "
                        + _json({k: attempt[k] for k in fields if k in attempt})
                    )
        for item in entry.get("alignment", []):
            if isinstance(item, dict):
                # Only omit the large bound paper/source snapshots. Keep every
                # recorded comparison, actual measurement and execution qualifier.
                brief = {
                    k: v for k, v in item.items() if k not in {"paper_target_binding", "target_condition"}
                }
                lines.append("- Recorded alignment: " + _json(brief))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n", selections


_SIBLING_LINK = re.compile(
    r"\[([^\n]*?)\]\((final_review|technical_appendix)\.md#(factreview-(?:(?:record|main)-[a-f0-9]{64}|evidence-[0-9]{6,}(?:-source-[0-9]{2,})?))\)"
)


def _pdf_markdown(markdown, *, bundle=False, appendix_pages=None):
    def convert(match):
        label, document, anchor = match.groups()
        if bundle:
            return f"[{label}](#{anchor})"
        page = (appendix_pages or {}).get(anchor) if document == "technical_appendix" else None
        suffix = f", page {page}" if page is not None else ""
        record = (
            anchor.removeprefix("factreview-")
            if anchor.startswith("factreview-evidence-")
            else anchor.rsplit("-", 1)[1][:12]
        )
        return f"{label} ({document}.pdf{suffix}; record {record}; bundle navigation)"

    return _SIBLING_LINK.sub(convert, markdown)


def _delivery_context_markdown(context):
    """Keep exact context values in independently laid-out JSON entries.

    A single issues array can contain thousands of diagnostics. ReportLab must
    repeatedly wrap the remainder of an oversized paragraph as it crosses pages.
    Emit each original issue separately; the initial empty-array entry and each
    explicit index preserve empty lists, order and duplicate values without loss.
    """
    lines = ["", "Delivery context (complete; ordered JSON entries):", ""]
    for key, value in context.items():
        split_issues = key == "issues" and isinstance(value, list)
        entries = [{"field": key, "value": [] if split_issues else value}]
        if split_issues:
            entries.extend({"field": key, "index": i, "value": item} for i, item in enumerate(value))
        elif key == "token_usage" and isinstance(value, dict) and isinstance(value.get("warnings"), list):
            entries[0]["value"] = {**value, "warnings": []}
            entries.extend(
                {"field": key, "key": "warnings", "index": i, "value": item}
                for i, item in enumerate(value["warnings"])
            )
        for entry in entries:
            text = json.dumps(entry, ensure_ascii=False)
            # Keep JSON quote characters literal. Escaping apostrophes to a
            # numeric HTML entity before Markdown escaping would turn its '#'
            # into visible text and prevent exact JSON reconstruction.
            literal = re.sub(r"([\\`*_\[\]{}()#+.!~-])", r"\\\1", html.escape(text, quote=False))
            lines += [literal, ""]
    return "\n".join(lines)


def _build_pdf(review, markdown, context, targets, title):
    from review.report.pdf_renderer import build_review_report_pdf

    models = sorted(
        {
            f"{c.advice.provider}/{c.advice.model}"
            for c in review.claims
            if c.advice and c.advice.state == "generated"
        }
    )
    return build_review_report_pdf(
        workspace_title=f"FactReview {title}: {review.paper_key}",
        source_pdf_name=review.paper_key,
        run_id=review.run_id,
        status=review.run_status,
        decision=None,
        estimated_cost=0,
        actual_cost=None,
        exported_at=datetime.now(UTC),
        meta_review={},
        reviewers=[],
        raw_output=None,
        final_report_markdown=markdown,
        implicit_math=False,
        agent_model="deterministic layered report" + ("; advice: " + ", ".join(models) if models else ""),
        token_usage=context.get("token_usage") or {"unavailable": True},
        navigation_targets=targets,
    )


def write_layered_review(review, output_dir: Path, *, render_pdf=True, **context):
    output_dir = Path(output_dir)
    names = (
        "final_review.md",
        "final_review.json",
        "final_review.pdf",
        "technical_appendix.md",
        "technical_appendix.pdf",
        "review_bundle.md",
        "review_bundle.pdf",
        "report_manifest.json",
    )
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError(
            "Layered report requires a fresh output directory; existing artifacts are retained"
        )
    original = review.model_dump(mode="json", exclude={"review_markdown"})
    from review.delivery import checked_delivery

    checked = checked_delivery(v2._checked_report(review), **context)
    snapshot = checked.model_dump(mode="json", exclude={"review_markdown"})
    v2.validate_publication_language(
        [snapshot, context.get("issues") or [], (context.get("token_usage") or {}).get("warnings", [])]
    )
    nav = _Navigation(checked)
    # Reserve all main destinations before the appendix adds return links.
    for path in [
        *nav.claim_paths.values(),
        *[f"/findings/{i}" for i in range(len(checked.findings))],
        *[f"/ledger/{i}" for i in range(len(checked.ledger))],
    ]:
        nav.main[path] = _anchor(path, main=True)
    for path, items in [(nav.claim_paths[id(c)], c.evidence) for c in checked.claims] + [
        (f"/findings/{i}", f.evidence) for i, f in enumerate(checked.findings)
    ]:
        for index in _selection(items)[0]:
            target = path + f"/evidence/{index}"
            nav.main[target] = _anchor(target, main=True)
    appendix = v2.render_markdown(checked, **context, _checked=True, _navigation=nav)
    appendix = _attach_markers(appendix)
    source_usages = []
    occurrences = v2._source_index(v2.ordered_claims(checked), checked.findings)
    owners = [(nav.claim_paths[id(c)], c.evidence) for c in v2.ordered_claims(checked)]
    owners += [(f"/findings/{i}", f.evidence) for i, f in enumerate(checked.findings)]
    for owner, evidence in owners:
        for index, item in enumerate(evidence):
            occurrence = next(occurrences)
            for pointer_index, pointer in enumerate([item.pointer, *item.additional_pointers]):
                path = (
                    owner
                    + f"/evidence/{index}"
                    + ("/pointer" if pointer_index == 0 else f"/additional_pointers/{pointer_index - 1}")
                )
                usage = occurrence["usages"][pointer_index]
                nav.appendix[path] = usage["anchor"]
                source_usages.append(
                    {
                        "json_pointer": path,
                        "usage_anchor": usage["anchor"],
                        "source_anchor": usage["source"]["anchor"] if usage["source"] else None,
                        "exact_source": pointer.model_dump(mode="json"),
                    }
                )
    appendix = appendix.replace("# FactReview", "# FactReview technical appendix", 1)
    appendix = nav.marker("") + "\n" + appendix
    appendix = _attach_markers(appendix)
    # The context is outside FinalReview; retain its full exact JSON in this
    # appendix as well as in the manifest, including all usage diagnostics.
    if "/delivery_context" not in nav.appendix:
        appendix += "\n" + nav.marker("/delivery_context", "") + "\n"
    appendix += _delivery_context_markdown(context)
    appendix = _attach_markers(appendix)
    main, selection = _compact_markdown(checked, nav, context)
    bundle = _pdf_markdown(main, bundle=True) + "\n\n" + _pdf_markdown(appendix, bundle=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for key, name, content in (
        ("markdown", "final_review.md", main),
        ("appendix_markdown", "technical_appendix.md", appendix),
        ("bundle_markdown", "review_bundle.md", bundle),
    ):
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        outputs[key] = str(path)
    checked.review_markdown = main
    artifact = output_dir / "final_review.json"
    artifact.write_text(checked.model_dump_json(indent=2), encoding="utf-8")
    outputs["json"] = str(artifact)
    pages = {key: {} for key in ("main", "appendix", "bundle")}
    if render_pdf:
        for name, key, text, target, title in (
            ("technical_appendix.pdf", "appendix_pdf", appendix, "appendix", "Technical appendix"),
            ("final_review.pdf", "pdf", main, "main", "Reading report"),
            ("review_bundle.pdf", "bundle_pdf", bundle, "bundle", "Reading report and technical appendix"),
        ):
            try:
                content = (
                    text if target == "bundle" else _pdf_markdown(text, appendix_pages=pages["appendix"])
                )
                data = _build_pdf(checked, content, context, pages[target], title)
                path = output_dir / name
                path.write_bytes(data)
                outputs[key] = str(path)
            except Exception as exc:
                outputs[key + "_error"] = f"{type(exc).__name__}: {exc}"
                pages[target].clear()
    if review.model_dump(mode="json", exclude={"review_markdown"}) != original:
        raise ValueError("Layered rendering changed input review records")
    if checked.model_dump(mode="json", exclude={"review_markdown"}) != snapshot:
        raise ValueError("Layered rendering changed checked review records")
    export_errors = {key: value for key, value in outputs.items() if key.endswith("_error")}
    if export_errors:
        from review.delivery import delivery_lines
        from schemas.review import DeliveryCheck

        checks = [DeliveryCheck(stage="report", component=key.removesuffix("_error"),
                                state="failed", reason=reason) for key, reason in export_errors.items()]
        if any(key.endswith("pdf") for key in outputs):
            checks.append(DeliveryCheck(
                stage="report", component="pdf_delivery_finalization", state="incomplete",
                reason="Successful PDFs are preserved with pre-export delivery metadata; finalization is pending.",
            ))
        checked = checked_delivery(checked, additional_checks=checks)
        # Preserve evidence text, source anchors and existing page navigation.
        # Healthy PDFs remain traceable; their late-status finalization is explicit.
        detail = "\n".join(delivery_lines(checked))
        main = main.replace(f"delivery: {snapshot['run_status']};", f"delivery: {checked.run_status};", 1)
        main = main.replace(
            f"Incomplete stages: {v2._text(', '.join(snapshot['incomplete_stages']) or 'none recorded')}.",
            f"Incomplete stages: {v2._text(', '.join(checked.incomplete_stages))}.", 1,
        )
        main += "\n" + detail
        appendix += "\n**Partial review — report export incomplete.**\n" + detail
        bundle = _pdf_markdown(main, bundle=True) + "\n\n" + _pdf_markdown(appendix, bundle=True)
        for key, text in (("markdown", main), ("appendix_markdown", appendix), ("bundle_markdown", bundle)):
            Path(outputs[key]).write_text(text, encoding="utf-8")
        checked.review_markdown = main
        snapshot = checked.model_dump(mode="json", exclude={"review_markdown"})
        artifact.write_text(checked.model_dump_json(indent=2), encoding="utf-8")
    locations = {}
    for pointer in _paths(snapshot):
        main_target, appendix_target = _nearest(pointer, nav.main), _nearest(pointer, nav.appendix)
        locations[pointer] = {
            "main_anchor": main_target,
            "appendix_anchor": appendix_target,
            "main_pdf_page": pages["main"].get(main_target),
            "appendix_pdf_page": pages["appendix"].get(appendix_target),
            "bundle_main_page": pages["bundle"].get(main_target),
            "bundle_appendix_page": pages["bundle"].get(appendix_target),
        }
    all_evidence = [e for c in checked.claims for e in c.evidence] + [
        e for f in checked.findings for e in f.evidence
    ]
    manifest = {
        "version": "layered-report-v1",
        "model_calls": 0,
        "navigation": {
            "method": "Internal GoTo in bundle; standalone main gives appendix file and page; Markdown uses sibling links.",
            "bundle_available": "bundle_pdf" in outputs,
            "appendix_page_locations_available": "appendix_pdf" in outputs,
        },
        "input_records_sha256": _digest(original),
        "checked_records_sha256": _digest(snapshot),
        "checked_records_equal_input": original == snapshot,
        "checked_changes": {
            "delivery_fields": [
                key for key in ("run_status", "incomplete_stages", "delivery_checks")
                if original[key] != snapshot[key]
            ],
            "advice_claim_indices": [
                i for i, c in enumerate(snapshot["claims"]) if c["advice"] != original["claims"][i]["advice"]
            ],
            "reference_finding_indices": [
                i
                for i, f in enumerate(snapshot["findings"])
                if f["reference_correction"] != original["findings"][i]["reference_correction"]
            ],
        },
        "records_equal_saved_json": snapshot
        == {k: v for k, v in json.loads(artifact.read_text("utf-8")).items() if k != "review_markdown"},
        "counts": {
            "claims": len(checked.claims),
            "conditions": sum(len(c.conditions) for c in checked.claims),
            "findings": len(checked.findings),
            "evidence": len(all_evidence),
            "pointer_usages": sum(1 + len(e.additional_pointers) for e in all_evidence),
            "unique_passage_sources": len({x["source_anchor"] for x in source_usages if x["source_anchor"]}),
            "unique_pointer_sources": len({_digest(x["exact_source"]) for x in source_usages}),
            "notes": sum(len(c.notes) for c in checked.claims),
            "questions": sum(len(c.questions) for c in checked.claims),
            "ledger": len(checked.ledger),
        },
        "delivery_context": context,
        "selection": selection,
        "records": locations,
        "source_usages": source_usages,
        "pdf_targets": pages,
        "artifacts": {
            key: {"name": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for key, path in outputs.items()
            if not key.endswith("_error")
        },
        "render_errors": {key: value for key, value in outputs.items() if key.endswith("_error")},
    }
    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["manifest"] = str(manifest_path)
    return outputs
