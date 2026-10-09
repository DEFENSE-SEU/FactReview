"""Compare quoted paper descriptions with an immutable, indexed repository snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import Field

from common import run_stats
from llm.diagnostics import redact_provider_details
from schemas.claim import AuthorQuestion, Claim, Contract, Evidence, EvidencePointer, NonEmpty
from schemas.limitations import VerificationLimitation
from schemas.materials import SharedMaterials
from screening import checks
from screening.checks import ask
from screening.visual_audit import redacted_record
from verification.code_joint import checked_joint_sources, make_binding
from verification.code_scope import review_code_scope
from verification.code_sources import CodeSourceContext
from verification.contracts import BranchResult
from verification.theory import (
    FULL_SUPPORT_DESCRIPTION,
    _covered,
    _fully_supported,
    _paper_pointer,
    _support_note,
)


def _indexed_sources(materials: SharedMaterials, *, claim: Claim, budget: int) -> tuple[dict[str, str], dict]:
    """Select complete files within a serialized source budget; retain omitted scope."""
    scope = {"budget_bytes": budget, "selected_bytes": 2, "selected_files": [], "omitted_files": []}
    if materials.repository is None:
        return {}, scope
    root = Path(materials.repository.root).resolve(strict=True)
    sources = {}
    terms = set(re.findall(r"[a-z][a-z0-9_]{2,}", claim.text.lower()))
    candidates = [item for item in materials.repository.files if item.kind in {"config", "source", "entry"}]
    # Prefer claim-named paths, then configs and entry points; tie-breaking is stable.
    candidates.sort(
        key=lambda item: (
            -len(terms & set(re.findall(r"[a-z][a-z0-9_]{2,}", item.path.lower()))),
            {"config": 0, "entry": 1, "source": 2}[item.kind],
            item.path,
        )
    )
    for item in candidates:
        path = root / item.path
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Indexed file escapes repository: {item.path}")
        if path.stat().st_size > budget:
            scope["omitted_files"].append({"path": item.path, "reason": "source_budget"})
            continue
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != item.sha256:
            raise ValueError(f"Indexed repository file changed: {item.path}")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            scope["omitted_files"].append({"path": item.path, "reason": "not_utf8"})
            continue
        encoded = json.dumps({item.path: _numbered_lines(text)}, ensure_ascii=False).encode("utf-8")
        # Include separators and JSON line-number overhead in the actual payload bound.
        if scope["selected_bytes"] + len(encoded) > budget:
            scope["omitted_files"].append({"path": item.path, "reason": "source_budget"})
            continue
        sources[item.path] = text
        scope["selected_bytes"] += len(encoded)
        scope["selected_files"].append(item.path)
    return sources, scope


def _numbered_lines(text: str) -> list[dict]:
    return [{"line": i, "text": line} for i, line in enumerate(text.splitlines(), 1)]


def _scope_summary(scope: dict, claim: Claim) -> dict:
    """Keep prompt/report metadata bounded; full manifests remain in the run."""
    summary = {key: scope[key] for key in ("budget_bytes", "selected_bytes")}
    summary.update(
        selected_count=len(scope["selected_files"]),
        omitted_count=len(scope["omitted_files"]),
        omission_reasons=dict(Counter(row["reason"] for row in scope["omitted_files"])),
        selected_sample=[path[:160] for path in scope["selected_files"][:10]],
        omitted_sample=[{**row, "path": row["path"][:160]} for row in scope["omitted_files"][:10]],
        sample_limit=10,
    )
    stats = run_stats.stats_path()
    if stats is not None:
        directory = stats.parent / "code_scopes"
        directory.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", claim.id)[:80]
        path = directory / f"{name}_{uuid.uuid4().hex}.json"
        path.write_text(
            json.dumps({"claim_id": claim.id, **scope}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["manifest"] = str(path)
    return summary


class CodeSourceSpan(Contract):
    file: NonEmpty
    line: int = Field(ge=1)
    quote: str = Field(
        min_length=1,
        description="Exact contiguous full source lines starting at line; preserve indentation and whitespace.",
    )


class CodeItem(CodeSourceSpan):
    additional_sources: list[CodeSourceSpan] = Field(default_factory=list, max_length=8)
    paper_block_id: NonEmpty
    paper_quote: NonEmpty = Field(
        description="Verbatim contiguous substring of paper_block_id's text, preserving math and whitespace."
    )
    covered: list[NonEmpty] = Field(
        min_length=1,
        description="Distinct exact condition IDs from allowed_condition_ids for this claim. No claim IDs, block IDs, or labels.",
        json_schema_extra={"uniqueItems": True},
    )
    fully_supported_conditions: list[NonEmpty] = Field(
        default_factory=list, description=FULL_SUPPORT_DESCRIPTION, json_schema_extra={"uniqueItems": True}
    )
    direction: Literal["support", "flaw"]
    aspect: Literal["architecture", "loss", "optimizer", "hyperparameters", "data_processing", "evaluation"]
    detail: NonEmpty


class CodeOutput(Contract):
    items: list[CodeItem]
    issues: list[str] = Field(default_factory=list)


def verify_code(claim: Claim, materials: SharedMaterials, *, call=None, scope_call=None) -> BranchResult:
    budget = int(os.environ.get("CODE_SOURCE_MAX_BYTES", "200000"))
    if budget <= 0:
        raise ValueError("CODE_SOURCE_MAX_BYTES must be a positive integer")
    sources, scope = _indexed_sources(materials, claim=claim, budget=budget)
    summary = _scope_summary(scope, claim)
    scope_issues = []
    if scope["omitted_files"]:
        scope_issues.append(
            "Code verification source coverage is partial: " + json.dumps(summary, ensure_ascii=False)
        )
    if not sources:
        reason = "No readable released-repository source fits the configured inspection scope."
        if scope["omitted_files"]:
            return BranchResult(
                issues=[reason, *scope_issues],
                verification_limitations=[
                    VerificationLimitation(
                        claim_id=claim.id,
                        condition_ids=[condition.id for condition in claim.conditions],
                        stage="Code",
                        kind="source_context_unavailable",
                        reason=" ".join([reason, *scope_issues]),
                    )
                ],
            )
        return BranchResult(
            issues=[reason, *scope_issues],
            questions=[
                AuthorQuestion(
                    claim_id=claim.id,
                    text="Can the released source/configuration needed to check this claim be provided?",
                    reason=reason,
                )
            ],
        )
    cfg = checks.resolve_llm_config()
    frozen = CodeSourceContext.capture(claim, materials, sources)
    response = ask(
        "Compare paper descriptions to indexed configs and source: architecture, loss, optimizer, "
        "hyperparameters, data processing, and evaluation protocol. Return output_schema JSON. "
        "Each item requires a specific source/config file, its one-based first line and exact contiguous "
        "line quote, plus an exact located paper quote. Explain the agreement or mismatch in detail. "
        "covered must be a nonempty list of distinct exact strings from allowed_condition_ids, the IDs "
        "in claim.conditions. Never use claim.id, block IDs, datasets, or metric names. Include only "
        "conditions the item actually addresses. Omit items that cannot be tied to an allowed condition "
        "and explain the limitation in issues; return items=[] when none can be grounded. "
        "For support, explicitly list fully_supported_conditions only when this one item establishes "
        "the ENTIRE condition and every relevant qualifier of the claim. Leave the list empty for "
        "partial agreement and describe the missing parts in detail. Source presence, loading code, "
        "or a download hook does not prove a public URL works or that all claimed data is released. "
        "Architecture and implementation do not establish historical novelty, reported performance, "
        "or an extension absent from the implementation. Do not fully support such conditions using "
        "only relevant source lines. "
        "Copy paper_quote verbatim as one contiguous substring of the selected paper block's text. "
        "Copy quote as complete contiguous source lines starting at line, preserving indentation. "
        "Preserve all mathematical markup, whitespace, punctuation, and spelling; do not normalize "
        "math, paraphrase, or join disjoint passages. "
        "source_scope lists omitted files and the source budget. An omitted file has not been inspected; "
        "do not infer missing implementations or repository-wide agreement from this selection. "
        "Only cite supplied files. Never execute or change code. Do not invent status or sufficiency fields. "
        "When one condition requires multiple files, explicitly propose ONE support item with primary "
        "file/line/quote and additional_sources containing every other exact source range needed. "
        "Such a joint item must cover exactly one condition. Establish the actual cross-file links "
        "and all original qualifiers; do not automatically merge separate partial observations. "
        "additional_sources=[] retains single-source behavior. Local file contents do not establish "
        "public URL availability, a measured execution result, or external release completeness.",
        {
            "claim": claim.model_dump(mode="json"),
            "allowed_condition_ids": [condition.id for condition in claim.conditions],
            "paper_blocks": [b.model_dump() for b in materials.blocks],
            "files": {name: _numbered_lines(text) for name, text in sources.items()},
            "source_scope": summary,
            "output_schema": CodeOutput.model_json_schema(),
        },
        module="verification.code",
        call=call,
    )
    frozen.check(claim, materials)
    response = deepcopy(response)
    parse_issues = []
    invalid_joint_rows = []
    parsed_items = []
    try:
        raw_items = response.get("items") if isinstance(response, dict) else None
        has_joint = isinstance(raw_items, list) and any(
            isinstance(row, dict) and "additional_sources" in row and row["additional_sources"] != []
            for row in raw_items
        )
        if has_joint:
            if set(response) - {"items", "issues"}:
                raise ValueError("Unknown Code response fields")
            output = CodeOutput(items=[], issues=response.get("issues", []))
            for index, row in enumerate(raw_items):
                try:
                    parsed_items.append((index, CodeItem.model_validate(row)))
                except ValueError as exc:
                    if (
                        not isinstance(row, dict)
                        or "additional_sources" not in row
                        or row["additional_sources"] == []
                    ):
                        raise
                    parse_issues.append(
                        f"Joint Code candidate {index} invalid: {redact_provider_details(str(exc), cfg)}"
                    )
                    invalid_joint_rows.append((index, row))
        else:
            output = CodeOutput.model_validate(response)
            parsed_items = list(enumerate(output.items))
    except ValueError as exc:
        raise ValueError(redact_provider_details(str(exc), cfg)) from None
    result = BranchResult(issues=[*scope_issues, *parse_issues, *redact_provider_details(output.issues, cfg)])
    validated = []
    joint_members = {}
    root = Path(materials.repository.root).resolve(strict=True)
    hashes = {entry.path: entry.sha256 for entry in materials.repository.files}

    def joint_failure(index, covered, error):
        reason = f"Joint Code candidate {index} unavailable: {redact_provider_details(str(error), cfg)}"
        result.issues.append(reason)
        if covered:
            result.verification_limitations.append(
                VerificationLimitation(
                    claim_id=claim.id,
                    condition_ids=covered,
                    stage="Code",
                    kind="evidence_validation_failed",
                    reason=reason,
                )
            )

    for index, row in invalid_joint_rows:
        try:
            covered = _covered(claim, row.get("covered", []))
        except (ValueError, TypeError):
            covered = []
        joint_failure(index, covered, "The explicit joint candidate failed schema validation")
    for index, item in parsed_items:
        covered = []
        try:
            covered = _covered(claim, item.covered)
            _fully_supported(covered, item.fully_supported_conditions)
            if item.additional_sources and (len(covered) != 1 or item.direction != "support"):
                raise ValueError("Joint Code requires one condition and support direction")
            members, seen = [], set()
            for ordinal, span in enumerate([item, *item.additional_sources]):
                if span.file not in sources:
                    raise ValueError("Code evidence references a file outside the repository index")
                lines, quoted_lines = sources[span.file].splitlines(), span.quote.splitlines()
                if (
                    not span.quote.strip()
                    or lines[span.line - 1 : span.line - 1 + len(quoted_lines)] != quoted_lines
                ):
                    raise ValueError("Code evidence quote does not match its indexed source lines")
                key = (span.file, span.line, span.quote)
                if key in seen:
                    raise ValueError("Joint Code contains a duplicate source")
                seen.add(key)
                pointer = EvidencePointer(
                    locator=str((root / span.file).resolve()), line=span.line, quote=span.quote
                )
                members.append(
                    {
                        "source_index": ordinal,
                        "file": span.file,
                        "pointer": pointer.model_dump(mode="json"),
                        "artifact_sha256": hashes[span.file],
                    }
                )
            paper = _paper_pointer(materials, item.paper_block_id, item.paper_quote)
            validated.append((index, item, paper, covered, members))
            if item.additional_sources:
                joint_members[index] = members
        except (ValueError, KeyError) as exc:
            if not item.additional_sources:
                raise
            joint_failure(index, covered, exc)
    if not validated:
        if has_joint and run_stats.stats_path() is not None:
            path = run_stats.stats_path().parent / "code_scope_reviews" / f"{uuid.uuid4().hex}.json"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        redacted_record(
                            {
                                "claim_id": claim.id,
                                "first_response": response,
                                "state": "no_grounded_candidate",
                                "issues": result.issues,
                            },
                            cfg,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                result.issues.append(f"Joint Code rejected candidate audit: {path}")
            except OSError as exc:
                result.issues.append(
                    redact_provider_details(f"Joint Code candidate audit unavailable: {exc}", cfg)
                )
        frozen.check(claim, materials)
        return result
    scopes, decisions, issues, audit = review_code_scope(
        claim,
        materials,
        [row[1] for row in validated],
        sources,
        summary,
        call=scope_call or call,
        candidate_indices=[row[0] for row in validated],
        joint_members=joint_members,
        first_response=response if has_joint else None,
    )
    frozen.check(claim, materials)
    result.issues.extend(issues)
    for name in {
        span.file
        for _, item in parsed_items
        for span in [item, *item.additional_sources]
        if span.file in sources
    }:
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Indexed file escapes repository: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"Indexed repository file changed during verification: {name}")
    for index, item, paper, covered, members in validated:
        detail = redact_provider_details(item.detail, cfg)
        note = f"{item.aspect}: {detail}; paper {paper.locator} [{paper.key}]: {paper.quote}"
        if item.direction == "support":
            note = _support_note(note, item.fully_supported_conditions)
        confirmed = []
        for cid in covered:
            scope, decision = scopes.get(cid), decisions.get((index, cid))
            valid = bool(
                scope
                and decision
                and decision.basis == "direct_source"
                and not decision.missing_qualifiers
                and "uncertain" not in scope.required_facets
            )
            if valid and item.direction == "support":
                valid = (
                    (
                        bool(scope.required_facets)
                        and set(scope.required_facets) <= {"implementation", "repository_contents"}
                        if item.additional_sources
                        else scope.required_facets == ["implementation"]
                    )
                    and decision.relation == "supports_implementation"
                    and decision.full_condition
                    and cid in item.fully_supported_conditions
                )
            elif valid:
                valid = (
                    "implementation" in scope.required_facets
                    and decision.relation == "contradicts_implementation"
                )
            if valid:
                confirmed.append(cid)
            else:
                result.issues.append(f"Code candidate {index} is non-decisive for condition {cid}.")
            if scope and decision:
                note += (
                    f"; {cid} independent_scope={scope.required_facets!r}/{decision.relation}: "
                    f"{decision.rationale}"
                )
                for bridge in decision.bridge_quotes:
                    pointer = _paper_pointer(materials, bridge.block_id, bridge.quote)
                    note += f"; bridge {pointer.locator} [{pointer.key}]: {pointer.quote}"
        if audit:
            note += f"; scope_audit={audit}; candidate_index={index}"
        binding = None
        if item.additional_sources:
            try:
                binding = make_binding(claim, covered[0], index, members, audit)
            except (OSError, ValueError) as exc:
                joint_failure(index, covered, exc)
                continue
        evidence = Evidence(
            source="code",
            pointer=EvidencePointer(
                locator=str((Path(materials.repository.root) / item.file).resolve()),
                line=item.line,
                quote=item.quote,
            ),
            covered=covered,
            direction=item.direction,
            sufficient=confirmed == covered,
            note=note,
            concern=item.direction == "flaw" and confirmed == covered,
            affects_claim=confirmed == covered,
            overturnable=True,
            additional_pointers=[EvidencePointer.model_validate(row["pointer"]) for row in members[1:]],
            code_joint_binding=binding,
        )
        if binding:
            try:
                checked_joint_sources(evidence, claim)
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                joint_failure(index, covered, exc)
                continue
        result.evidence.append(evidence)
        if confirmed and confirmed != covered:
            # Retain the original observation's scope as non-decisive, alongside
            # the independently confirmed subset. Other conditions stay open.
            result.evidence.append(
                evidence.model_copy(
                    update={
                        "covered": confirmed,
                        "sufficient": True,
                        "concern": item.direction == "flaw",
                        "affects_claim": True,
                        "note": note + f"; independent_scope_confirmed_conditions={confirmed!r}",
                    }
                )
            )
    frozen.check(claim, materials)
    retained = []
    for evidence in result.evidence:
        try:
            checked_joint_sources(evidence, claim)
            retained.append(evidence)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            joint_failure(evidence.code_joint_binding.candidate_index, evidence.covered, exc)
    result.evidence = retained
    return result
