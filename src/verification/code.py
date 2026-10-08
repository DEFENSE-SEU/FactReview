"""Compare quoted paper descriptions with an immutable, indexed repository snapshot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from schemas.claim import AuthorQuestion, Claim, Contract, Evidence, EvidencePointer, NonEmpty
from schemas.materials import SharedMaterials
from screening.checks import ask
from verification.contracts import BranchResult
from verification.theory import (
    FULL_SUPPORT_DESCRIPTION,
    _covered,
    _fully_supported,
    _paper_pointer,
    _support_note,
)


def _indexed_sources(materials: SharedMaterials) -> dict[str, str]:
    if materials.repository is None:
        return {}
    root = Path(materials.repository.root).resolve(strict=True)
    sources = {}
    for item in materials.repository.files:
        if item.kind not in {"config", "source", "entry"}:
            continue
        path = root / item.path
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Indexed file escapes repository: {item.path}")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != item.sha256:
            raise ValueError(f"Indexed repository file changed: {item.path}")
        try:
            sources[item.path] = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return sources


class CodeItem(Contract):
    file: NonEmpty
    line: int = Field(ge=1)
    quote: str = Field(
        min_length=1,
        description="Exact contiguous full source lines starting at line; preserve indentation and whitespace.",
    )
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


def verify_code(claim: Claim, materials: SharedMaterials, *, call=None) -> BranchResult:
    sources = _indexed_sources(materials)
    if not sources:
        reason = "No readable released-repository index is available."
        return BranchResult(
            issues=[reason],
            questions=[
                AuthorQuestion(
                    claim_id=claim.id,
                    text="Can the released source/configuration needed to check this claim be provided?",
                    reason=reason,
                )
            ],
        )
    output = CodeOutput.model_validate(
        ask(
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
            "Only cite supplied files. Never execute or change code. Do not invent status or sufficiency fields.",
            {
                "claim": claim.model_dump(mode="json"),
                "allowed_condition_ids": [condition.id for condition in claim.conditions],
                "paper_blocks": [b.model_dump() for b in materials.blocks],
                "files": {
                    name: [{"line": i, "text": line} for i, line in enumerate(text.splitlines(), 1)]
                    for name, text in sources.items()
                },
                "output_schema": CodeOutput.model_json_schema(),
            },
            module="verification.code",
            call=call,
        )
    )
    result = BranchResult(issues=output.issues)
    for item in output.items:
        if item.file not in sources:
            raise ValueError("Code evidence references a file outside the repository index")
        lines = sources[item.file].splitlines()
        quoted_lines = item.quote.splitlines()
        if not item.quote.strip() or lines[item.line - 1 : item.line - 1 + len(quoted_lines)] != quoted_lines:
            raise ValueError("Code evidence quote does not match its indexed source lines")
        paper = _paper_pointer(materials, item.paper_block_id, item.paper_quote)
        covered = _covered(claim, item.covered)
        full_support = _fully_supported(covered, item.fully_supported_conditions)
        note = f"{item.aspect}: {item.detail}; paper {paper.locator} [{paper.key}]: {paper.quote}"
        if item.direction == "support":
            note = _support_note(note, item.fully_supported_conditions)
        result.evidence.append(
            Evidence(
                source="code",
                pointer=EvidencePointer(
                    locator=str((Path(materials.repository.root) / item.file).resolve()),
                    line=item.line,
                    quote=item.quote,
                ),
                covered=covered,
                direction=item.direction,
                sufficient=item.direction == "flaw" or full_support,
                note=note,
                concern=item.direction == "flaw",
                overturnable=True,
            )
        )
    return result
