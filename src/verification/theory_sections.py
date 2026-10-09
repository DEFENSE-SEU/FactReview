"""Conservative, source-preserving roles for Theory's two reading phases."""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.materials import SharedMaterials

_LABEL = re.compile(r"^([A-Z](?:\.\d+)*)(?:[.:]?\s+|$)")
_EXPLICIT = re.compile(r"^Appendix\s+([A-Z](?:\.\d+)*)(?:[.:]?\s+.*)?$", re.I)
_REFERENCES = re.compile(r"^(?:References|Bibliography)$", re.I)
_APPENDIX_REF = re.compile(r"\bappendi(?:x|ces)\s+([A-Z](?:\.\d+)*)\b", re.I)
_STATEMENT = re.compile(
    r"\b(theorem|lemma|proposition|corollary)\s+((?:[A-Z]\.)?\d+(?:\.\d+)*|[A-Z])\b", re.I
)
_PROOF = re.compile(
    r"\b(?:proof|derivation)\s+(?:of|for)\s+(?:the\s+)?((?:theorem|lemma|proposition|corollary)\s+(?:[A-Z]\.)?\d+(?:\.\d+)*|(?:theorem|lemma|proposition|corollary)\s+[A-Z])\b",
    re.I,
)


def _title(section: str) -> str:
    return re.sub(r"^(?:section_\d+|block_[\w.]+):\s*", "", section).strip().lstrip("# ")


def _explicit(title: str) -> tuple[bool, str | None]:
    match = _EXPLICIT.fullmatch(title)
    if match:
        return True, match.group(1).upper()
    return bool(
        re.fullmatch(
            r"Appendix|Appendices|Appendix(?:\s*:\s*|\s+)(?:Proofs?|Derivations?)|Supplement(?:ary)?(?: Materials?| Information)?",
            title,
            re.I,
        )
    ), None


def _unresolved_appendix_title(title: str) -> bool:
    return bool(
        re.match(
            r"^(?:Appendices\b|Appendix\s*:|Appendix\s+[A-Z](?:\.\d+)*(?:[.:]|\s|$)|Supplement(?:ary)?\s*:)",
            title,
        )
    )


def _located_heading(block, markdown: str) -> bool:
    if not block.loc or block.loc.char_start is None or block.loc.char_end is None:
        return False
    start, end = block.loc.char_start, block.loc.char_end
    return 0 <= start < end <= len(markdown) and markdown[start:end] == block.text


def _local_references(text):
    for match in _APPENDIX_REF.finditer(text):
        before, after = text[: match.start()], text[match.end() :]
        if re.search(r"(?:\b(?:their|his|her|external|another|prior)\s+|['’]s\s+)$", before, re.I):
            continue
        if re.match(r"\s+(?:of|in)\s+(?:(?:the|a)\s+)?(?:\[|reference\b|paper\b|work\b)", after, re.I):
            continue
        yield match


@dataclass
class TheorySections:
    roles: dict[str, str]
    regions: list[dict]
    references: list[dict]
    issues: list[str]

    def is_main(self, identifier: str) -> bool:
        return self.roles.get(identifier) == "main"

    def is_appendix(self, identifier: str) -> bool:
        return self.roles.get(identifier) == "appendix"

    def audit(self) -> dict:
        return {
            "version": "theory-sections-v1",
            "roles": self.roles,
            "regions": self.regions,
            "author_references": self.references,
            "issues": self.issues,
        }

    def no_proof_issue(
        self, claim, covered, cited_block_id, cited_quote, materials, loaded_ids
    ) -> str | None:
        """An unread related region cannot establish absence of an author proof."""
        blocks = {b.id: b for b in materials.blocks}
        passages = [(cited_block_id, cited_quote)]
        primary_refs = [
            ref
            for ref in claim.source_refs
            if (ref.source_block_id, ref.source_quote) == (claim.source_block_id, claim.source_quote)
        ]
        if claim.source_block_id and (
            not primary_refs or any(set(ref.covered) & set(covered) for ref in primary_refs)
        ):
            passages.append((claim.source_block_id, claim.source_quote))
        passages.extend(
            (ref.source_block_id, ref.source_quote)
            for ref in claim.source_refs
            if set(ref.covered) & set(covered)
        )
        targets = {key for key, _ in passages}
        texts = [quote for key, quote in passages if key in blocks and quote and quote in blocks[key].text]
        labels = {m.group(1).upper() for text in texts for m in _local_references(text)}
        identities = {
            " ".join(m.group().casefold().split()) for text in texts for m in _STATEMENT.finditer(text)
        }
        related = {key for key in targets if self.roles.get(key) in {"appendix", "unknown"}}
        unresolved = []
        if any(re.search(r"\bsupplement(?:ary)?\s+(?:material|information)\b", text, re.I) for text in texts):
            if any(
                role in {"appendix", "unknown"} and key not in loaded_ids for key, role in self.roles.items()
            ):
                unresolved.append(
                    "the target refers to supplementary material whose relevant extent has not been confirmed/read"
                )
        for label in sorted(labels):
            matching = [
                r
                for r in self.regions
                if r["label"] == label or (r["label"] and r["label"].startswith(label + "."))
            ]
            if not matching:
                unresolved.append(f"appendix {label} has no uniquely classified source region")
            for region in matching:
                related.update(region["block_ids"])
        for block in materials.blocks:
            if self.roles.get(block.id) not in {"appendix", "unknown"}:
                continue
            if any(
                " ".join(m.group(1).casefold().split()) in identities for m in _PROOF.finditer(block.text)
            ):
                related.add(block.id)
                # Reading a proof's heading is not reading the proof body.
                # Explicit numbered descendants inherit this exact region;
                # stop before a sibling/parent or an unrelated proof region.
                for index, region in enumerate(self.regions):
                    if region["heading_id"] != block.id:
                        continue
                    related.update(region["block_ids"])
                    label = region["label"]
                    if label:
                        for descendant in self.regions[index + 1 :]:
                            child_label = descendant["label"]
                            if not child_label or not child_label.startswith(label + "."):
                                break
                            related.update(descendant["block_ids"])
        unknown = sorted(key for key in related if self.roles.get(key) == "unknown")
        unread = sorted(key for key in related if self.roles.get(key) == "appendix" and key not in loaded_ids)
        if unknown:
            unresolved.append(f"related source roles are unresolved: {unknown}")
        if unread:
            unresolved.append(f"related appendix sources have not been read: {unread}")
        return "; ".join(unresolved) or None


def partition_theory_sections(materials: SharedMaterials) -> TheorySections:
    blocks = materials.blocks
    roles = {b.id: "main" for b in blocks}
    if len(roles) != len(blocks):
        raise ValueError("Theory material block IDs must be unique")
    headings = []
    for index, block in enumerate(blocks):
        if block.kind != "heading":
            continue
        title = _title(block.text)
        explicit, explicit_label = _explicit(title)
        label = _LABEL.match(title)
        headings.append(
            {
                "index": index,
                "heading_id": block.id,
                "title": title,
                "label": explicit_label or (label.group(1) if label else None),
                "explicit": explicit,
                "located": _located_heading(block, materials.markdown),
                "loc": block.loc.model_dump(mode="json") if block.loc else None,
            }
        )
    # A paper title such as "A Study ..." is not the first appendix A.
    # Derive the candidate terminal chain after normal body/front-matter boundaries.
    first_numbered = next((h["index"] for h in headings if re.match(r"^\d+(?:\.\d+)*\b", h["title"])), None)
    references_start = next((h["index"] for h in headings if _REFERENCES.fullmatch(h["title"])), None)
    tail_after_references = references_start is not None and any(
        h["label"] and h["index"] > references_start for h in headings
    )
    letter_headings = [
        h
        for h in headings
        if h["label"]
        and not h["explicit"]
        and h["title"] != materials.title
        and (first_numbered is None or h["index"] > first_numbered)
        and (not tail_after_references or h["index"] > references_start)
    ]
    references = []
    first_letter = min((h["index"] for h in letter_headings), default=len(blocks))
    bibliography_positions = set()
    for index, heading in enumerate(headings):
        if _REFERENCES.fullmatch(heading["title"]):
            end = next(
                (h["index"] for h in headings[index + 1 :] if h["explicit"] or h["label"]), len(blocks)
            )
            bibliography_positions.update(range(heading["index"], end))
    for index, block in enumerate(blocks[:first_letter]):
        if (
            block.kind == "heading"
            or index in bibliography_positions
            or not _located_heading(block, materials.markdown)
        ):
            continue
        for match in _local_references(block.text):
            references.append(
                {
                    "block_id": block.id,
                    "quote": match.group(),
                    "label": match.group(1).upper(),
                    "loc": block.loc.model_dump(mode="json") if block.loc else None,
                }
            )
    labels = {row["label"] for row in references}
    top = [h for h in letter_headings if "." not in h["label"]]
    sequence = [h["label"] for h in top]
    ordered_chain = bool(sequence) and sequence == [chr(ord("A") + i) for i in range(len(sequence))]
    end_region = bool(top) and not any(
        re.match(r"^\d+(?:\.\d+)*\b", h["title"]) for h in headings if h["index"] > top[0]["index"]
    )
    reference_boundary = bool(top) and any(
        _REFERENCES.fullmatch(h["title"]) and h["located"] and h["index"] < top[0]["index"] for h in headings
    )
    numeric_main = bool(top) and any(
        re.match(r"^\d+(?:\.\d+)*\b", h["title"]) and h["index"] < top[0]["index"] for h in headings
    )
    heading_labels = {h["label"] for h in headings if h["located"]}
    cross_roots = {label.split(".")[0] for label in labels if label in heading_labels} & set(sequence)
    chain_confirmed = (
        ordered_chain
        and end_region
        and all(h["located"] for h in letter_headings)
        and bool(cross_roots)
        and (reference_boundary or (numeric_main and len(cross_roots) >= 2))
    )
    identity_headings = [*letter_headings, *(h for h in headings if h["explicit"])]
    duplicate_labels = {
        label
        for label in [h["label"] for h in identity_headings if h["label"]]
        if sum(h["label"] == label for h in identity_headings) > 1
    }
    regions, issues = [], []
    active = "main"
    explicit_root = None
    current_letter_root = None
    for number, heading in enumerate(headings):
        label = heading["label"]
        start = heading["index"]
        end = headings[number + 1]["index"] if number + 1 < len(headings) else len(blocks)
        if _REFERENCES.fullmatch(heading["title"]):
            role = "references" if heading["located"] else "unknown"
            explicit_root = None
        elif heading["explicit"]:
            role = "appendix" if heading["located"] and label not in duplicate_labels else "unknown"
            explicit_root = label.split(".")[0] if label else "*"
        elif _unresolved_appendix_title(heading["title"]):
            role, explicit_root = "unknown", None
        elif label:
            inherited = active == "appendix" and (
                explicit_root == "*" or label.split(".")[0] == explicit_root
            )
            chain_member = (
                chain_confirmed
                and heading in letter_headings
                and label.split(".")[0] in sequence
                and ("." not in label or label.split(".")[0] == current_letter_root)
            )
            if heading["located"] and label not in duplicate_labels and (chain_member or inherited):
                role = "appendix"
            elif heading["title"] == materials.title or (
                first_numbered is not None and start < first_numbered
            ):
                role = "main"
            elif labels or reference_boundary or active in {"appendix", "unknown"}:
                role = "unknown"
            else:
                role = "main"
        elif re.match(r"^\d+(?:\.\d+)*\b", heading["title"]):
            role, explicit_root = ("unknown" if active == "references" else "main"), None
        else:
            role = active if active in {"appendix", "references", "unknown"} else "main"
        if not heading["located"] and role != "main":
            role = "unknown"
        for block in blocks[start:end]:
            roles[block.id] = role
        region = {
            **heading,
            "role": role,
            "parent_label": label.rsplit(".", 1)[0] if label and "." in label else None,
            "block_ids": [b.id for b in blocks[start:end]],
            "basis": "explicit_heading"
            if heading["explicit"]
            else "author_references_and_terminal_heading_chain"
            if role == "appendix" and chain_confirmed
            else "parent_region"
            if role == "appendix"
            else "heading_region",
        }
        regions.append(region)
        if role == "unknown":
            issues.append(f"Unresolved Theory source region: {heading['heading_id']} ({heading['title']})")
        active = role
        if label and "." not in label:
            current_letter_root = label
    # Old materials can carry explicit section identity without heading blocks.
    # Accept only the narrow, exact legacy label, never a keyword substring.
    if not headings:
        legacy_root = None
        for block in blocks:
            title = _title(block.loc.section or "") if block.loc else ""
            explicit, label = _explicit(title)
            if explicit:
                legacy_root = label.split(".")[0] if label else "*"
                roles[block.id] = "appendix"
                regions.append(
                    {
                        "heading_id": None,
                        "title": title,
                        "label": label,
                        "parent_label": None,
                        "role": "appendix",
                        "block_ids": [block.id],
                        "basis": "legacy_explicit_section",
                        "loc": block.loc.model_dump(mode="json"),
                    }
                )
            elif _REFERENCES.fullmatch(title):
                roles[block.id] = "references"
                legacy_root = None
            elif _unresolved_appendix_title(title):
                roles[block.id] = "unknown"
                issues.append(f"Unresolved Theory source section: {block.id} ({title})")
                legacy_root = None
            elif (
                (match := _LABEL.match(title)) and legacy_root and match.group(1).split(".")[0] == legacy_root
            ):
                roles[block.id] = "appendix"
                regions.append(
                    {
                        "heading_id": None,
                        "title": title,
                        "label": match.group(1),
                        "parent_label": legacy_root,
                        "role": "appendix",
                        "block_ids": [block.id],
                        "basis": "legacy_explicit_parent_section",
                        "loc": block.loc.model_dump(mode="json"),
                    }
                )
            else:
                legacy_root = None
    return TheorySections(roles, regions, references, issues)
