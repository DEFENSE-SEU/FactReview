"""Frozen inputs for section writing checks and their consumed original pages."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path


def _file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
    except OSError:
        return None


class WritingSourceError(ValueError):
    """The current source no longer matches the snapshot preceding every callback."""


@dataclass
class WritingSources:
    artifacts: dict
    hashes: dict
    markdown: str
    block_ids: list
    sections: dict
    reference_index: dict
    pages: list
    image_hashes: dict
    consumed: dict = field(default_factory=dict)

    @classmethod
    def capture(cls, materials, groups, index):
        artifacts = {
            name: getattr(materials, name) for name in ("markdown_path", "source_pdf", "content_list_path")
        }
        return cls(
            artifacts,
            {path: _file_hash(path) for path in artifacts.values()},
            materials.markdown,
            [block.id for block in materials.blocks],
            {
                section: [b.model_dump(mode="json") for b in group["blocks"]]
                for section, group in groups.items()
            },
            copy.deepcopy(index),
            [page.model_dump(mode="json") for page in materials.pages],
            {page.path: _file_hash(page.path) for page in materials.pages},
        )

    def check_section(self, materials, section, index):
        if (
            any(getattr(materials, name) != path for name, path in self.artifacts.items())
            or materials.markdown != self.markdown
            or any(_file_hash(path) != value for path, value in self.hashes.items())
        ):
            raise WritingSourceError("Writing original manuscript artifact changed during screening")
        current_ids = [block.id for block in materials.blocks]
        if current_ids != self.block_ids or len(current_ids) != len(set(current_ids)):
            raise WritingSourceError("Writing source block identities changed or are duplicated")
        blocks = {block.id: block for block in materials.blocks}
        if any(blocks[row["id"]].model_dump(mode="json") != row for row in self.sections[section]):
            raise WritingSourceError("Writing section source text or location changed during screening")
        if index != self.reference_index:
            raise WritingSourceError("Writing original cross-reference directory changed during screening")
        self.check_pages(materials, self.consumed.get(section, ()))

    def check_pages(self, materials, numbers, *, section=None, consume=False):
        for number in numbers:
            original = [page for page in self.pages if page["page"] == number]
            current = [page.model_dump(mode="json") for page in materials.pages if page.page == number]
            if len(original) != 1 or current != original:
                raise WritingSourceError("Writing original page identity or metadata changed or is ambiguous")
            path = original[0]["path"]
            expected = self.image_hashes[path]
            if expected is None or _file_hash(path) != expected:
                raise WritingSourceError(
                    "Writing original page pixels changed or were unavailable before screening"
                )
        if consume:
            self.consumed.setdefault(section, set()).update(numbers)
