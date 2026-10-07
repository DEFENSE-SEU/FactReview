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
from verification.theory import _covered, _paper_pointer


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
    quote: str = Field(min_length=1)
    paper_block_id: NonEmpty
    paper_quote: NonEmpty
    covered: list[NonEmpty]
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
            "Only cite supplied files. Never execute or change code. Do not invent status or sufficiency fields.",
            {
                "claim": claim.model_dump(mode="json"),
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
        result.evidence.append(
            Evidence(
                source="code",
                pointer=EvidencePointer(
                    locator=str((Path(materials.repository.root) / item.file).resolve()),
                    line=item.line,
                    quote=item.quote,
                ),
                covered=_covered(claim, item.covered),
                direction=item.direction,
                sufficient=True,
                note=f"{item.aspect}: {item.detail}; paper {paper.locator} [{paper.key}]: {paper.quote}",
                concern=item.direction == "flaw",
                overturnable=True,
            )
        )
    return result
