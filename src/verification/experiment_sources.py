"""Exact, candidate-local evidence views over unchanged authoritative materials.

Derived choices retain original block hashes/spans and parent identities. Numeric
occurrences keep their original IDs and complete governing sentences. No clipped
material is reparsed as a new paper, and no source union grants wider scope.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
from pathlib import Path

from verification.experiment_catalog import _CAPTION, _stable_id, resolve_source
from verification.theory import _paper_pointer


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact(pointer):
    path = Path(pointer.locator)
    if not path.is_file():
        raise ValueError("Joint evidence artifact is unavailable")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def joint_limit():
    value = int(os.getenv("EXPERIMENT_JOINT_MAX_SOURCES", "6"))
    if value < 2:
        raise ValueError("EXPERIMENT_JOINT_MAX_SOURCES must be at least two")
    return value


def containing_member(view, block_id, start, end):
    return next(
        (
            member
            for member in sorted(view["members"], key=lambda m: (m["end"] - m["start"], m["ordinal"]))
            if member["block_id"] == block_id and member["start"] <= start < end <= member["end"]
        ),
        None,
    )


def require_passage(catalog, materials, block_id, quote, *, purpose):
    """Check every derived/implicit passage against one original member."""
    view = catalog.get("joint_view")
    if view is None:
        return None
    block = next((b for b in materials.blocks if b.id == block_id), None)
    if block is None or not quote or block.text.count(quote) != 1:
        raise ValueError(f"Joint {purpose} needs one exact unique original passage")
    start = block.text.index(quote)
    member = containing_member(view, block_id, start, start + len(quote))
    if member is None:
        raise ValueError(f"Joint {purpose} reads outside its declared source ranges")
    if _hash(block.text) != member["block_sha256"]:
        raise ValueError("Joint source block changed")
    if block.loc is None or block.loc.model_dump(exclude_none=True) != member["loc"]:
        raise ValueError("Joint source block location changed")
    member_quote = block.text[member["start"] : member["end"]]
    if (
        not 0 <= member["start"] < member["end"] <= len(block.text)
        or _hash(member_quote) != member["quote_sha256"]
        or member_quote != member["pointer"]["quote"]
    ):
        raise ValueError("Joint source member range changed")
    # A parser-only full member can point to a PDF page while an exact child
    # also occurs in Markdown. Preserve the declared member's artifact identity.
    pointer = _paper_pointer(materials, block_id, member_quote)
    if pointer.model_dump() != member["pointer"]:
        raise ValueError("Joint source pointer changed")
    if _artifact(pointer) != member["artifact"]:
        raise ValueError("Joint source artifact changed")
    record = {
        "member_source_id": member["source_id"],
        "block_id": block_id,
        "start": start,
        "end": start + len(quote),
        "purpose": purpose,
    }
    if record not in view["consumption"]:
        view["consumption"].append(record)
    return member["source_id"]


def prepare_joint_candidate(claim, materials, catalog, item, index, *, max_sources=None):
    """Return a candidate view, including validated pointers when a member fails."""
    condition_id = item.covered[0] if len(item.covered) == 1 else ""
    identity = _stable_id(
        "candidate",
        catalog["paper_key"],
        catalog["source_pdf"],
        claim.model_dump(mode="json"),
        index,
        item.model_dump(mode="json"),
    )
    view = {
        "candidate_id": identity,
        "candidate_index": index,
        "condition_id": condition_id,
        "members": [],
        "errors": [],
        "consumption": [],
        "semantic_only": [],
    }
    limit = joint_limit() if max_sources is None else max_sources
    view["max_sources"] = limit
    if item.kind != "paper_support" or len(item.covered) != 1:
        view["errors"].append("Joint sources require paper_support for one original condition")
    passages = [(item.block_id, item.quote), *((p.block_id, p.quote) for p in item.additional_sources)]
    if len(passages) > limit:
        view["errors"].append(f"Joint source count exceeds configured limit {limit}; no sources truncated")
    seen = set()
    for ordinal, (block_id, quote) in enumerate(passages):
        try:
            pointer = _paper_pointer(materials, block_id, quote)
            block = next(b for b in materials.blocks if b.id == block_id)
            if block.text.count(quote) != 1:
                raise ValueError("Joint member must be a unique contiguous original passage")
            start = block.text.index(quote)
            span = (block_id, start, start + len(quote))
            if span in seen:
                raise ValueError("Duplicate joint member range")
            seen.add(span)
            parents = [
                key
                for key, value in catalog["sources"].items()
                if value.get("block_id") == block_id
                and value.get("start", -1) <= start
                and value.get("end", -1) >= start + len(quote)
            ]
            record = {
                "block_id": block_id,
                "block_sha256": _hash(block.text),
                "start": start,
                "end": start + len(quote),
                "loc": block.loc.model_dump(exclude_none=True),
                "kind": "candidate_source",
                "covered": [condition_id],
                "candidate_id": identity,
                "parent_source_ids": parents,
            }
            source_id = _stable_id("src", catalog["paper_key"], catalog["source_pdf"], record)
            view["members"].append(
                {
                    **record,
                    "source_id": source_id,
                    "ordinal": ordinal,
                    "quote_sha256": _hash(quote),
                    "pointer": pointer.model_dump(),
                    "artifact": _artifact(pointer),
                }
            )
        except (ValueError, OSError, StopIteration) as exc:
            view["errors"].append(f"member {ordinal}: {exc}")
    view["primary_source_id"] = next((m["source_id"] for m in view["members"] if m["ordinal"] == 0), None)
    view["member_source_ids"] = [m["source_id"] for m in view["members"]]
    bounded = {
        key: copy.deepcopy(value)
        for key, value in catalog.items()
        if key not in {"sources", "cells", "numbers", "tables"}
    }
    bounded.update(sources={}, cells={}, numbers={}, tables={}, joint_view=view)
    # Assertion selectors are a separate domain. They may explain a claimed
    # endpoint/difference, and never authorize observation or context reads.
    bounded["assertion_numbers"] = {}
    for key, number in catalog.get("numbers", {}).items():
        if any(
            source.get("kind") == "claim_source"
            and condition_id in source["covered"]
            and source["block_id"] == number["block_id"]
            and source["start"] <= number["sentence_start"] < number["sentence_end"] <= source["end"]
            for source in catalog["sources"].values()
        ):
            bounded["assertion_numbers"][key] = copy.deepcopy(number)
    if view["errors"]:
        return bounded
    for member in view["members"]:
        record = {
            key: member[key]
            for key in (
                "block_id",
                "block_sha256",
                "start",
                "end",
                "loc",
                "kind",
                "covered",
                "candidate_id",
                "parent_source_ids",
            )
        }
        bounded["sources"][member["source_id"]] = record

    def derived_source(parent_id, start=None, end=None):
        parent = catalog["sources"][parent_id]
        if parent["kind"] == "claim_text":
            return None
        begin, finish = (parent["start"] if start is None else start, parent["end"] if end is None else end)
        if not containing_member(view, parent["block_id"], begin, finish):
            return None
        record = {
            **copy.deepcopy(parent),
            "start": begin,
            "end": finish,
            "candidate_id": identity,
            "parent_source_ids": [parent_id],
        }
        identifier = _stable_id("src", catalog["paper_key"], catalog["source_pdf"], record)
        bounded["sources"][identifier] = record
        return identifier

    source_map = {key: derived_source(key) for key in catalog["sources"]}
    for table_id, table in catalog["tables"].items():
        source = resolve_source(catalog, table["source_id"], materials)
        spans = list(re.finditer(r"<table\b.*?</table>", source["quote"], re.I | re.S))
        span = spans[table["table"]]
        start, end = source["start"] + span.start(), source["start"] + span.end()
        if not containing_member(view, source["block_id"], start, end):
            continue
        caption_id = source_map.get(table["caption_source_id"])
        # A complete adjacent prefix may accompany the table only when the
        # combined contiguous range belongs to ONE declared member.
        if caption_id:
            caption_start = bounded["sources"][caption_id]["start"]
            if containing_member(view, source["block_id"], caption_start, end):
                start = caption_start
        source_id = derived_source(table["source_id"], start, end)
        derived_table_id = _stable_id("table", source_id, 0)
        caption_ambiguous = table["caption_ambiguous"]
        if caption_id is None and table["caption_source_id"]:
            prefix = resolve_source(catalog, table["caption_source_id"], materials)
            headings = list(_CAPTION.finditer(prefix["quote"]))
            if headings:
                # The parser prefix may contain an earlier table's caption.
                # Keep the complete final caption, with only separator whitespace
                # excluded, when one declared member authorizes all of it. The
                # already-derived table body and its selector IDs stay unchanged.
                caption_start = prefix["start"] + headings[-1].start(1)
                caption_end = prefix["start"] + len(prefix["quote"].rstrip())
                caption_id = derived_source(table["caption_source_id"], caption_start, caption_end)
                if caption_id:
                    caption_ambiguous = False
        bounded["tables"][derived_table_id] = {
            **copy.deepcopy(table),
            "source_id": source_id,
            "table": 0,
            "caption_source_id": caption_id,
            "caption_ambiguous": caption_ambiguous,
            "parent_table_id": table_id,
            "origin_table_index": table["table"],
            "reference_source_ids": [
                source_map[key] for key in table["reference_source_ids"] if source_map.get(key)
            ],
        }
        for parent_id, cell in catalog["cells"].items():
            if cell["table_id"] != table_id:
                continue
            record = {
                **copy.deepcopy(cell),
                "source_id": source_id,
                "table": 0,
                "table_id": derived_table_id,
                "parent_cell_id": parent_id,
                "origin_table_id": table_id,
                "origin_table_index": cell["table"],
            }
            key = _stable_id("cell", source_id, 0, cell["row"], cell["column"], cell["token"])
            bounded["cells"][key] = record
    bounded["numbers"] = {
        key: copy.deepcopy(number)
        for key, number in catalog.get("numbers", {}).items()
        if containing_member(view, number["block_id"], number["sentence_start"], number["sentence_end"])
    }
    return bounded


def validate_source_uses(catalog, row, materials, *, own_table_reference, has_label, condition):
    """Record declared roles and concrete consumers; semantic-only roles stay explicit."""
    view = catalog["joint_view"]
    uses = {use.source_id: use for use in row.source_uses}
    if len(uses) != len(row.source_uses) or set(uses) != set(view["member_source_ids"]):
        raise ValueError("Joint source_uses must cover each declared member exactly once")
    consumed = {key: set() for key in uses}

    def mark(source_id, role):
        source = resolve_source(catalog, source_id, materials)
        member = require_passage(catalog, materials, source["block_id"], source["quote"], purpose=role)
        consumed[member].add(role)

    grounds = set()
    for source_id in row.grounds_source_ids:
        source = resolve_source(catalog, source_id, materials)
        grounds.add(
            require_passage(catalog, materials, source["block_id"], source["quote"], purpose="ground")
        )
    for comparison in row.comparisons:
        for side in (comparison.left, comparison.right):
            if side.kind == "cell":
                mark(catalog["cells"][side.cell_id]["source_id"], "result")
            else:
                record = catalog["numbers"][side.number_id]
                member = require_passage(
                    catalog, materials, record["block_id"], record["sentence"], purpose="result"
                )
                consumed[member].add("result")
        for bridge in comparison.bridges:
            from verification.experiment_catalog import resolve_cell

            selector = comparison.left if bridge.applies_to != "comparator" else comparison.right
            if selector.kind != "cell":
                raise ValueError("Joint source bridges require table operands")
            cell = resolve_cell(catalog, selector.cell_id, materials)
            names = re.findall(r"(?:^|\n)\s*(Table\s+\d+)\s*[:.]", cell["caption"], re.I)
            for source_id in bridge.source_ids:
                source = resolve_source(catalog, source_id, materials)
                if names and own_table_reference(source["quote"], names[-1]):
                    mark(source_id, "table_reference")
                if (
                    bridge.kind == "metric"
                    and has_label(condition.dataset, source["quote"])
                    and has_label(bridge.paper_label, source["quote"])
                ):
                    mark(source_id, "metric_definition")
                if bridge.kind == "setting" and has_label(bridge.paper_label, source["quote"]):
                    mark(source_id, "setup_definition")
    for source_id, use in uses.items():
        if not use.roles or len(use.roles) != len(set(use.roles)):
            raise ValueError("Joint source use roles must be distinct and nonempty")
        for role in use.roles:
            if role in {"protocol", "other_qualifier"} or (role == "result" and not row.comparisons):
                if source_id not in grounds:
                    raise ValueError("Semantic-only joint use requires an explicit grounded passage")
                view["semantic_only"].append(
                    {"source_id": source_id, "role": role, "rationale": use.rationale}
                )
            elif role not in consumed[source_id]:
                raise ValueError(f"Joint source role {role} has no corresponding numerical/bridge consumer")
    view["source_uses"] = [use.model_dump() for use in row.source_uses]
    view["role_consumption"] = {key: sorted(value) for key, value in consumed.items()}


def revalidate_members(catalog, materials):
    for member in catalog["joint_view"]["members"]:
        require_passage(
            catalog,
            materials,
            member["block_id"],
            member["pointer"]["quote"],
            purpose="final_source_revalidation",
        )
    view = catalog["joint_view"]
    for assertion in view.get("assertion_consumption", []):
        artifact = assertion["artifact"]
        path = Path(artifact["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("Joint condition assertion artifact changed")
    view["revalidated_artifacts"] = {
        member["artifact"]["path"]: member["artifact"]["sha256"] for member in view["members"]
    }


def record_assertion(catalog, materials, number, *, purpose):
    """Audit a separately condition-validated assertion without granting member access."""
    if catalog is None or "joint_view" not in catalog:
        return
    pointer = _paper_pointer(materials, number.block_id, number.quote)
    block = next(b for b in materials.blocks if b.id == number.block_id)
    record = {
        "domain": "condition_assertion",
        "purpose": purpose,
        "condition_id": catalog["joint_view"]["condition_id"],
        "block_id": block.id,
        "block_sha256": _hash(block.text),
        "pointer": pointer.model_dump(),
        "artifact": _artifact(pointer),
        "token": number.token,
        "grants_evidence_access": False,
    }
    records = catalog["joint_view"].setdefault("assertion_consumption", [])
    if record not in records:
        records.append(record)
