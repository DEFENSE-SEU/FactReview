"""Freeze the target and consumed Code inputs across both model boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


def _hash(path):
    candidate = Path(path)
    try:
        return hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() else None
    except OSError:
        # Unavailable, unused artifacts retain the existing grounding contract.
        return None


def _materials(materials):
    return {
        "blocks": [block.model_dump(mode="json") for block in materials.blocks],
        "markdown": materials.markdown,
        "markdown_path": materials.markdown_path,
        "source_pdf": materials.source_pdf,
        "repository": materials.repository.model_dump(mode="json") if materials.repository else None,
    }


@dataclass
class CodeSourceContext:
    claim: dict
    materials: dict
    paper_hashes: dict
    code_hashes: dict
    repository_root: str | None

    @classmethod
    def capture(cls, claim, materials, selected_sources):
        paper = {path: _hash(path) for path in (materials.markdown_path, materials.source_pdf)}
        repository = materials.repository
        files = {row.path: row.sha256 for row in repository.files} if repository else {}
        root = Path(repository.root).resolve(strict=True) if repository else None
        code = {str(root / name): files[name] for name in selected_sources}
        context = cls(
            claim.model_dump(mode="json"), _materials(materials), paper, code, str(root) if root else None
        )
        # The index was validated while selecting files; never adopt changed
        # bytes here as a fresh baseline after constructing the model payload.
        context.check(claim, materials)
        return context

    def check(self, claim, materials):
        if claim.model_dump(mode="json") != self.claim:
            raise ValueError("Code target claim/conditions changed during verification")
        if _materials(materials) != self.materials:
            raise ValueError("Code paper or repository context changed during verification")
        if self.repository_root and str(Path(materials.repository.root).resolve()) != self.repository_root:
            raise ValueError("Code repository location changed during verification")
        if any(_hash(path) != digest for path, digest in self.paper_hashes.items()):
            raise ValueError("Code paper source artifact changed during verification")
        for path, digest in self.code_hashes.items():
            candidate = Path(path)
            if (
                candidate.is_symlink()
                or not candidate.resolve().is_relative_to(self.repository_root)
                or _hash(path) != digest
            ):
                raise ValueError("Code indexed source artifact changed during verification")
