"""Search backend protocol — both arXiv and Semantic Scholar implement this."""

from __future__ import annotations

from typing import Protocol

from refcopilot.models import ExternalRecord, Reference


class SearchBackend(Protocol):
    name: str

    def lookup(self, ref: Reference) -> list[ExternalRecord]:
        """Return zero or more candidate records for the given reference."""
        ...
