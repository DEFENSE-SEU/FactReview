"""Independent, source-grounded applicability review for Code observations."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from common import run_stats
from llm.diagnostics import redact_provider_details, sanitized_endpoint
from schemas.claim import Claim, Contract, NonEmpty
from schemas.materials import SharedMaterials
from screening import checks
from screening.visual_audit import redacted_record

ExactID = Annotated[str, StringConstraints(min_length=1)]


class CodeConditionScope(Contract):
    condition_id: ExactID
    required_facets: list[
        Literal[
            "implementation",
            "repository_contents",
            "empirical_outcome",
            "novelty",
            "availability",
            "uncertain",
        ]
    ] = Field(min_length=1)
    claim_source_ids: list[ExactID] = Field(min_length=1)
    rationale: NonEmpty


class BridgeQuote(Contract):
    block_id: ExactID
    quote: NonEmpty


class CodeSourceUse(Contract):
    source_index: int = Field(ge=0, strict=True)
    role: Literal["implementation", "configuration", "artifact_contents", "link"]
    rationale: NonEmpty


class CodeScopeDecision(Contract):
    item_index: int = Field(ge=0, strict=True)
    condition_id: ExactID
    relation: Literal[
        "supports_implementation", "contradicts_implementation", "partial", "irrelevant", "uncertain"
    ]
    full_condition: bool = Field(strict=True)
    basis: Literal["direct_source", "absence", "uncertain"]
    bridge_quotes: list[BridgeQuote]
    missing_qualifiers: list[NonEmpty]
    rationale: NonEmpty
    source_uses: list[CodeSourceUse] = Field(
        default_factory=list,
        description="Use only for explicit joint candidates with additional_sources; single-source candidates require an empty list.",
    )


class CodeScopeOutput(Contract):
    conditions: list[CodeConditionScope]
    items: list[CodeScopeDecision]


def _claim_sources(claim, materials):
    blocks = {block.id: block for block in materials.blocks}
    condition_ids = [condition.id for condition in claim.conditions]
    primary_refs = [
        ref
        for ref in claim.source_refs
        if (ref.source_block_id, ref.source_quote) == (claim.source_block_id, claim.source_quote)
    ]
    primary_coverage = (
        [cid for cid in condition_ids if any(cid in ref.covered for ref in primary_refs)]
        if primary_refs
        else condition_ids
    )
    rows = [("primary", claim.source_block_id, claim.source_quote, claim.loc, primary_coverage)]
    rows.extend(
        (f"ref_{index}", ref.source_block_id, ref.source_quote, ref.loc, ref.covered)
        for index, ref in enumerate(claim.source_refs, 1)
    )
    catalog = {}
    for identifier, block_id, quote, loc, covered in rows:
        block = blocks.get(block_id)
        if block is None or block.loc is None or not quote or quote not in block.text:
            continue
        if loc.page is not None and loc.page != block.loc.page:
            continue
        if loc.char_start is not None and block.loc.char_start is not None:
            start = block.loc.char_start + block.text.find(quote)
            if (loc.char_start, loc.char_end) != (start, start + len(quote)):
                continue
        try:
            pointer = checks.grounded_paper_pointer(materials, block, quote)
        except ValueError:
            continue
        catalog[identifier] = {
            "source_id": identifier,
            "block_id": block_id,
            "quote": quote,
            "covered": covered,
            "pointer": pointer.model_dump(mode="json"),
        }
    return catalog


def review_code_scope(
    claim: Claim,
    materials: SharedMaterials,
    items,
    sources: dict[str, str],
    source_scope: dict,
    *,
    call=None,
    candidate_indices=None,
    joint_members=None,
    first_response=None,
):
    """Keep valid condition/item decisions when another decision is malformed."""
    cfg = checks.resolve_llm_config()
    indices = list(range(len(items))) if candidate_indices is None else candidate_indices
    members = joint_members or {}
    catalog = _claim_sources(claim, materials)
    relevant = {item.paper_block_id for item in items} | {row["block_id"] for row in catalog.values()}
    positions = {index for index, block in enumerate(materials.blocks) if block.id in relevant}
    positions = {near for index in positions for near in (index - 1, index, index + 1)}
    paper = {block.id: block for index, block in enumerate(materials.blocks) if index in positions}
    code_lines = {}
    for original_index, item in zip(indices, items, strict=True):
        if original_index in members:
            # Joint sufficiency can consume only the explicitly declared ranges.
            continue
        lines = sources[item.file].splitlines()
        selected = code_lines.setdefault(item.file, {})
        for index in range(
            max(0, item.line - 6), min(len(lines), item.line + len(item.quote.splitlines()) + 4)
        ):
            selected[index + 1] = lines[index]
    payload = {
        "claim": claim.model_dump(mode="json"),
        "claim_sources": list(catalog.values()),
        "candidate_items": [item.model_dump(mode="json") for item in items],
        "paper_blocks": [block.model_dump(mode="json") for block in paper.values()],
        "source_context": {
            name: [{"line": line, "text": text} for line, text in sorted(lines.items())]
            for name, lines in code_lines.items()
        },
        "source_scope": source_scope,
        "output_schema": CodeScopeOutput.model_json_schema(),
    }
    if members:
        payload.update(candidate_indices=indices, joint_members=members)
    elif indices != list(range(len(items))):
        payload["candidate_indices"] = indices
    system = (
        "Independently review the applicability and complete coverage of each Code candidate. "
        "Return exactly one conditions entry per claim condition, and one items entry for each "
        "candidate item_index and its covered condition ID. Treat candidate detail and full-support "
        "flags as proposals. Independently confirm the ENTIRE condition; do not inherit first-pass "
        "fully_supported_conditions. Classify ALL requirements of each condition in required_facets: "
        "implementation, empirical_outcome, novelty, availability, or uncertain. Preserve combined "
        "requirements such as using Adam AND achieving MRR 0.4. A real Adam line cannot establish "
        "a measured MRR score, historical novelty, or working public downloads. "
        "claim_source_ids must name supplied exact claim_sources covering that condition. "
        "An abstract assertion can correspond to a Methods paragraph without shared exact wording. "
        "Explain that correspondence using the actual method/component/settings, and provide exact "
        "bridge_quotes from supplied paper_blocks when additional passages establish the link. "
        "Never borrow a different setting, optional/dead code path, component, or empirical result. "
        "Use supports_implementation only for genuine implementation correspondence; full_condition "
        "requires every qualifier. Use contradicts_implementation for a concrete conflicting "
        "implementation under the claimed setting. Partial, irrelevant and uncertain observations "
        "cannot decide a claim. Record missing_qualifiers explicitly. "
        "basis=direct_source requires the quoted actual source to show that agreement/conflict. "
        "Code context contains selected snippets; omitted files or unseen code cannot establish "
        "implementation absence. Mark absence-based proposals with basis=absence and unresolved "
        "context with uncertain. Do not invent execution evidence or claim statuses. "
        "Copy bridge quotes verbatim, preserving whitespace and mathematical markup. "
        "For every single-source candidate, return source_uses=[]. This field is reserved for "
        "explicit joint candidates with additional_sources."
    )
    if members:
        system += (
            " Joint candidates explicitly declare all their code members in joint_members, keyed by "
            "original candidate index; candidate_indices maps candidate_items order to those indices. "
            "Judge each joint as one observation, never union separate partial candidates. Its decisive "
            "code sources are only those exact declared member lines, not another candidate or nearby "
            "context. Give source_uses naming every source_index exactly once, its role and why it "
            "contributes to this condition. Verify the actual cross-file links and selected configuration; "
            "do not substitute unused/dead/optional paths. repository_contents means only the supplied "
            "frozen repository's local artifact contents. availability still includes public URL access "
            "or external release completeness and cannot be established by local files. Keep empirical "
            "outcomes and novelty separate."
        )
    stats = run_stats.stats_path()
    audit_path = stats.parent / "code_scope_reviews" / f"{uuid.uuid4().hex}.json" if stats else None
    audit = {
        "claim_id": claim.id,
        "provider": cfg.provider,
        "model": cfg.model,
        "endpoint": sanitized_endpoint(cfg.base_url),
        "system": system,
        "request": payload,
    }
    if members:
        import hashlib

        audit.update(
            repository_root=str(Path(materials.repository.root).resolve()),
            paper_source_hashes={
                str(Path(name).resolve()): hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in (materials.markdown_path, materials.source_pdf)
                if Path(name).is_file()
            },
        )
    if first_response is not None:
        audit["first_response"] = first_response
    scopes, decisions, issues = {}, {}, []
    condition_ids = {condition.id for condition in claim.conditions}
    expected = {(index, cid) for index, item in zip(indices, items, strict=True) for cid in item.covered}
    seen_conditions, seen_items = set(), set()
    try:
        response = checks.ask(system, payload, module="verification.code.scope", call=call)
        audit["response"] = response
        if set(response) - {"conditions", "items", "status"}:
            raise ValueError("Unknown Code scope response fields")
        if not isinstance(response.get("conditions"), list) or not isinstance(response.get("items"), list):
            raise ValueError("Code scope response requires conditions and items lists")
        for row in response["conditions"]:
            cid = row.get("condition_id") if isinstance(row, dict) else None
            # Detect conflicting aliases before schema checks. Malformed or
            # padded duplicates permanently revoke their canonical condition.
            cid = cid.strip() if isinstance(cid, str) else cid
            try:
                if isinstance(cid, str) and cid in condition_ids:
                    if cid in seen_conditions:
                        scopes.pop(cid, None)
                        raise ValueError("Duplicate condition scope")
                    seen_conditions.add(cid)
                parsed = CodeConditionScope.model_validate(row)
                if parsed.condition_id not in condition_ids:
                    raise ValueError("Scope references a foreign condition")
                if len(set(parsed.required_facets)) != len(parsed.required_facets):
                    raise ValueError("Duplicate requirement facets")
                if len(set(parsed.claim_source_ids)) != len(parsed.claim_source_ids) or any(
                    identifier not in catalog or cid not in catalog[identifier]["covered"]
                    for identifier in parsed.claim_source_ids
                ):
                    raise ValueError("Scope source is unavailable or outside this condition")
                parsed.rationale = redact_provider_details(parsed.rationale, cfg)
                scopes[cid] = parsed
            except (ValueError, TypeError) as exc:
                issues.append(f"Code condition scope unconfirmed for {cid}: {exc}")
        for row in response["items"]:
            key = None
            try:
                if (
                    isinstance(row, dict)
                    and type(row.get("item_index")) is int
                    and isinstance(row.get("condition_id"), str)
                ):
                    key = (row["item_index"], row["condition_id"].strip())
                    if key in expected:
                        if key in seen_items:
                            decisions.pop(key, None)
                            raise ValueError("Duplicate candidate scope decision")
                        seen_items.add(key)
                parsed = CodeScopeDecision.model_validate(row)
                key = (parsed.item_index, parsed.condition_id)
                if key not in expected:
                    raise ValueError("Scope decision references a foreign candidate/condition pair")
                if parsed.item_index in members:
                    supplied = [use.source_index for use in parsed.source_uses]
                    if sorted(supplied) != list(range(len(members[parsed.item_index]))):
                        raise ValueError("Joint Code scope must consume every declared member exactly once")
                elif parsed.source_uses:
                    raise ValueError("Single-source Code scope cannot consume joint members")
                for bridge in parsed.bridge_quotes:
                    if bridge.block_id not in paper:
                        raise ValueError("Bridge references an unsupplied manuscript block")
                    checks.grounded_paper_pointer(materials, paper[bridge.block_id], bridge.quote)
                parsed.rationale = redact_provider_details(parsed.rationale, cfg)
                for use in parsed.source_uses:
                    use.rationale = redact_provider_details(use.rationale, cfg)
                decisions[key] = parsed
            except (ValueError, TypeError) as exc:
                issues.append(f"Code candidate scope unconfirmed for {key}: {exc}")
        for cid in condition_ids - scopes.keys():
            issues.append(f"Code condition scope unavailable: {cid}")
        for key in expected - decisions.keys():
            issues.append(f"Code candidate scope unavailable: {key}")
    except Exception as exc:
        scopes, decisions = {}, {}
        issues.append(f"Code scope review unconfirmed: {type(exc).__name__}: {exc}")
    issues = redact_provider_details(issues, cfg)
    audit.update(
        issues=issues,
        validated_conditions={cid: row.model_dump(mode="json") for cid, row in scopes.items()},
        validated_items=[row.model_dump(mode="json") for row in decisions.values()],
    )
    if audit_path:
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(redacted_record(audit, cfg), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            if first_response is None:
                raise
            issues.append(redact_provider_details(f"Joint Code scope audit unavailable: {exc}", cfg))
            audit_path = None
    return scopes, decisions, issues, audit_path
