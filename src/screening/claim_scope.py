"""Source visibility and closed scope obligations; semantic judgments remain model output."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr

from schemas.claim import Contract

SCOPE_DIMENSIONS = (
    "dataset_population",
    "split_sampling",
    "label_budget",
    "selection_protocol",
    "model_training_augmentation",
    "other_boundary",
)
Dimension = Literal[
    "dataset_population",
    "split_sampling",
    "label_budget",
    "selection_protocol",
    "model_training_augmentation",
    "other_boundary",
]
ScopeState = Literal["preserved", "missing", "not_governing", "unresolved"]


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class ScopeSpan(Contract):
    block_id: StrictStr
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)


class ScopeEffect(Contract):
    kind: Literal["verification_setting", "conclusion_boundary", "background_fact", "provenance_only"]
    source_setting: Any
    alternative_setting: Any
    original_permits_alternative: StrictBool
    assertion_path: StrictStr
    assertion_value: Any
    explanation: StrictStr = Field(min_length=1)


class ScopeAtom(Contract):
    id: StrictStr = Field(min_length=1)
    dimension: Dimension
    condition_ids: list[StrictStr] = Field(min_length=1)
    state: ScopeState
    restriction: StrictStr = Field(
        min_length=1, description="One governing restriction, never an unexamined conjunction."
    )
    sources: list[ScopeSpan] = Field(min_length=1)
    preserved_indices: list[StrictInt] = Field(
        description="Closed indices of unchanged preserved_qualifiers carriers; each declares this same atom."
    )
    finding_index: StrictInt | None
    effect: ScopeEffect | None
    reason: StrictStr = Field(min_length=1)


class ScopeDimensionReview(Contract):
    state: ScopeState
    atom_ids: list[StrictStr]
    reason: StrictStr = Field(min_length=1)


class ScopeGroup(Contract):
    condition_ids: list[StrictStr] = Field(min_length=1)
    dimensions: dict[Dimension, ScopeDimensionReview]


class ScopeSourceReview(Contract):
    block_id: StrictStr
    state: Literal["considered", "irrelevant", "unavailable"]
    condition_ids: list[StrictStr] = Field(min_length=1)
    reason: StrictStr = Field(min_length=1)


def _trusted_span(block, markdown):
    if block.loc and block.loc.char_start is not None:
        a, b = block.loc.char_start, block.loc.char_end
        return (
            (a, b)
            if type(a) is int
            and type(b) is int
            and 0 <= a < b <= len(markdown)
            and markdown[a:b] == block.text
            else None
        )
    if len(block.text) >= 16 and markdown.count(block.text) == 1:
        a = markdown.index(block.text)
        return a, a + len(block.text)
    return None


def _source_candidates(blocks, markdown):
    """Numbered headings provide candidate ancestry only with a complete exact-span chain."""
    ordered = list(blocks.values())
    headings = []
    for i, block in enumerate(ordered):
        if block.kind == "heading":
            match = re.match(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\s+\S", block.text)
            headings.append(
                (i, block, _trusted_span(block, markdown), tuple(match[1].split(".")) if match else None)
            )
    counts = Counter(h[3] for h in headings if h[3])
    chains, stack, previous_end = {}, [], -1
    for i, _block, span, number in headings:
        reliable = span is not None and number is not None and counts[number] == 1 and span[0] >= previous_end
        if span:
            previous_end = max(previous_end, span[1])
        if not reliable:
            stack = []
            chains[i] = None
            continue
        while stack and stack[-1][1] != number[:-1]:
            stack.pop()
        if len(number) > 1 and not stack:
            chains[i] = None
            continue
        chains[i] = [entry[0] for entry in stack]
        stack.append((i, number))
    result = {}
    heading_indices = [h[0] for h in headings]
    for i, block in enumerate(ordered):
        preceding = [h for h in heading_indices if h < i]
        heading = preceding[-1] if preceding else None
        stop = next((h for h in heading_indices if h > i), len(ordered))
        start = heading + 1 if heading is not None else 0
        target_span = _trusted_span(block, markdown)
        heading_span = _trusted_span(ordered[heading], markdown) if heading is not None else None
        next_span = _trusted_span(ordered[stop], markdown) if stop < len(ordered) else None
        reliable = (
            heading is not None
            and chains.get(heading) is not None
            and target_span is not None
            and heading_span is not None
            and target_span[0] >= heading_span[1]
            and (stop == len(ordered) or (next_span is not None and target_span[1] <= next_span[0]))
        )
        candidates = []
        nearby = [
            j
            for j in (range(start, stop) if reliable else range(len(ordered)))
            if ordered[j].kind == "text" and ordered[j].text.strip() and j != i
        ]
        for j in ([max(j for j in nearby if j < i)] if any(j < i for j in nearby) else []) + (
            [min(j for j in nearby if j > i)] if any(j > i for j in nearby) else []
        ):
            candidates.append((ordered[j].id, "same_section_nearby" if reliable else "ordered-nearby"))
        for j in (i - 1, i + 1):
            if (
                0 <= j < len(ordered)
                and ordered[j].kind in {"caption", "footnote"}
                and ordered[j].text.strip()
            ):
                candidates.append((ordered[j].id, "ordered-nearby"))
        if reliable:
            for h in [heading, *reversed(chains[heading])]:
                end = next((k for k in heading_indices if k > h), len(ordered))
                intro = next(
                    (j for j in range(h + 1, end) if ordered[j].kind == "text" and ordered[j].text.strip()),
                    None,
                )
                intro_span = _trusted_span(ordered[intro], markdown) if intro is not None else None
                parent_span = _trusted_span(ordered[h], markdown)
                boundary_span = _trusted_span(ordered[end], markdown) if end < len(ordered) else None
                if (
                    intro is not None
                    and intro != i
                    and intro_span is not None
                    and parent_span is not None
                    and intro_span[0] >= parent_span[1]
                    and (
                        end == len(ordered)
                        or (boundary_span is not None and intro_span[1] <= boundary_span[0])
                    )
                ):
                    candidates.append(
                        (
                            ordered[intro].id,
                            "section_intro_candidate"
                            if h == heading
                            else "numbered_heading_ancestor_candidate:" + ordered[h].id,
                        )
                    )
        result[block.id] = list(dict.fromkeys(candidates))
    return result


def scope_context(
    registry,
    required,
    window_ids,
    blocks,
    markdown,
    budget,
    *,
    validate=None,
    already_loaded=(),
    history_loader=None,
    background_loader=None,
):
    """One shared extra-body budget for bindings, history and candidate context."""
    if type(budget) is not int or budget < 0:
        raise ValueError("Source budget must be a nonnegative integer")
    limit = min(budget, 24_000)
    extras = copy.deepcopy(list(already_loaded))
    loaded = {row["block"]["id"] for row in extras}
    size = sum(len(row["block"]["text"]) for row in extras)
    if size > limit or len(loaded) != len(extras):
        raise ValueError("Previously loaded source context exceeds the shared budget or repeats blocks")
    candidates = _source_candidates(blocks, markdown)
    claims = {row["claim_id"]: row for row in registry}
    contexts, queue = [], []
    for subject_index, subject in enumerate(required):
        claim = claims[subject["claim_id"]]
        conditions = [c["id"] for c in claim["conditions"]]
        selected = {}
        bindings = [
            (claim["source_block_id"], conditions),
            *[(ref["source_block_id"], ref["covered"]) for ref in claim["source_refs"]],
        ]
        for key, covered in bindings:
            selected.setdefault(key, {"block_id": key, "reasons": [], "condition_ids": []})
            selected[key]["reasons"].append("original_binding")
            selected[key]["condition_ids"] = list(dict.fromkeys([*selected[key]["condition_ids"], *covered]))
            queue.append((0, subject_index, key))
        for bound, _ in bindings:
            for key, reason in candidates.get(bound, []):
                selected.setdefault(key, {"block_id": key, "reasons": [], "condition_ids": conditions})
                selected[key]["reasons"].append(reason)
                selected[key]["condition_ids"] = list(
                    dict.fromkeys([*selected[key]["condition_ids"], *conditions])
                )
                selected[key].setdefault("candidate_for", []).append(
                    {"bound_block_id": bound, "reason": reason}
                )
                queue.append((1 if "intro" in reason or "ancestor" in reason else 2, subject_index, key))
        contexts.append(
            {
                "claim_id": claim["claim_id"],
                "claim_digest": claim["digest"],
                "sources": list(selected.values()),
            }
        )
    availability = {}

    def load(key):
        nonlocal size
        try:
            block = blocks[key]
            if validate:
                validate(key)
            elif block.loc is None or not block.text.strip():
                raise ValueError("Source has no recorded original location/text")
            if key in window_ids or key in loaded:
                availability[key] = ("in_window" if key in window_ids else "loaded", None)
            elif size + len(block.text) > limit:
                availability[key] = ("unavailable", "budget")
            else:
                extras.append({"block": copy.deepcopy(block.model_dump(mode="json")), "bindings": []})
                loaded.add(key)
                size += len(block.text)
                availability[key] = ("loaded", None)
        except (ValueError, KeyError, TypeError):
            availability[key] = ("unavailable", "invalid_or_missing_source")

    order = {key: i for i, key in enumerate(blocks)}
    unique = sorted(set(queue), key=lambda row: (row[0], row[1], order.get(row[2], len(order))))
    for priority, _, key in unique:
        if priority == 0 and key not in availability:
            load(key)

    def history(load_history):
        nonlocal size
        if not load_history:
            return
        for item in load_history(max(0, limit - size), set(window_ids) | loaded):
            key = item["block"]["id"]
            existing = next((r for r in extras if r["block"]["id"] == key), None)
            if existing is not None:
                for binding in item["bindings"]:
                    if binding not in existing["bindings"]:
                        existing["bindings"].append(copy.deepcopy(binding))
            if key not in window_ids and key not in loaded:
                if size + len(item["block"]["text"]) > limit:
                    raise ValueError("Historical loader exceeded the shared character budget")
                extras.append(copy.deepcopy(item))
                loaded.add(key)
                size += len(item["block"]["text"])

    history(history_loader)
    for _, _, key in unique:
        if key not in availability:
            load(key)
    history(background_loader)
    for context in contexts:
        for source in context["sources"]:
            key = source["block_id"]
            state, reason = availability[key]
            source.update(
                availability=state,
                unavailable_reason=reason,
                block_digest=fingerprint(blocks[key].model_dump(mode="json")) if key in blocks else None,
                trusted_span=list(_trusted_span(blocks[key], markdown))
                if key in blocks and _trusted_span(blocks[key], markdown) is not None
                else None,
            )
        context["complete"] = all(s["availability"] != "unavailable" for s in context["sources"])
    return contexts, extras


def check_scope(row, claim, context, blocks, carrier):
    """Close identities and indexed declarations; entailment is the model's responsibility."""
    if context is None or context["claim_id"] != row.claim_id or context["claim_digest"] != row.claim_digest:
        raise ValueError("V4 scope context identity differs from the reviewed claim")
    ids = {c["id"] for c in claim["conditions"]}

    def closed(values, allowed, name):
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError("V4 repeated, missing or foreign " + name)

    partition = [key for group in row.scope_groups for key in group.condition_ids]
    if len(set(partition)) != len(partition) or set(partition) != ids:
        raise ValueError("V4 scope groups must partition every original condition exactly once")
    sources = {s["block_id"]: s for s in context["sources"]}
    reviews = {s.block_id: s for s in row.source_reviews}
    # Additional explicit sources must already be visible in this request. Candidate
    # selection is conservative and does not forbid a model discovering another relation.
    for key in set(reviews) - set(sources):
        if key not in blocks:
            raise ValueError("V4 additional scope source has no visible original block")
        sources[key] = {
            "block_id": key,
            "availability": "in_window",
            "condition_ids": reviews[key].condition_ids,
            "block_digest": fingerprint(blocks[key].model_dump(mode="json")),
        }
    if len(reviews) != len(row.source_reviews) or set(reviews) != set(sources):
        raise ValueError("V4 source reviews must close the complete source directory")
    for key, review in reviews.items():
        closed(review.condition_ids, ids, "source conditions")
        if not set(sources[key]["condition_ids"]) <= set(review.condition_ids):
            raise ValueError("V4 source review omits affected source conditions")
        unavailable = sources[key]["availability"] == "unavailable"
        if (review.state == "unavailable") != unavailable:
            raise ValueError("V4 source review contradicts mechanical source availability")
    if not context["complete"] and row.state != "unresolved":
        raise ValueError("V4 unavailable sources cannot authorize resolved scope")
    atoms = {a.id: a for a in row.scope_atoms}
    if len(atoms) != len(row.scope_atoms):
        raise ValueError("V4 scope atom IDs must be unique")
    preserved_used, finding_used, referenced = [], [], set()

    def semantic_value(path, value, affected):
        actual = carrier(claim, path)
        if fingerprint(actual) != fingerprint(value):
            raise ValueError("V4 semantic carrier differs from original value/type")
        if path != "/text":
            tokens = path[1:].split("/")
            if len(tokens) < 3 or tokens[2] not in {"dataset", "metric", "description", "settings"}:
                raise ValueError("V4 carrier must locate a semantic field, never source/identity metadata")
            condition = claim["conditions"][int(tokens[1])]["id"]
            if condition not in affected or set(affected) != {condition}:
                raise ValueError("V4 condition carrier cannot supply another condition's scope")
        if actual is None or isinstance(actual, (dict, list)) or actual == "":
            raise ValueError("V4 carrier must be an actual nonempty semantic leaf")

    for atom in row.scope_atoms:
        closed(atom.condition_ids, ids, "atom conditions")
        if atom.state == "not_governing":
            raise ValueError("Non-governing dimensions have no governing atoms")
        span_ids = set()
        span_keys = set()
        for span in atom.sources:
            key = span.block_id
            marker = (key, span.start, span.end)
            if marker in span_keys or key not in sources or reviews[key].state != "considered":
                raise ValueError("V4 atom source must be unique, visible and explicitly considered")
            if not set(atom.condition_ids) <= set(reviews[key].condition_ids):
                raise ValueError("V4 atom conditions exceed its source review conditions")
            span_keys.add(marker)
            if sources[key]["availability"] == "unavailable" or key not in blocks:
                raise ValueError("Unavailable source cannot supply a V4 scope atom")
            block = blocks[key]
            if sources[key]["block_digest"] != fingerprint(block.model_dump(mode="json")):
                raise ValueError("V4 scope source digest changed")
            if (
                not 0 <= span.start < span.end <= len(block.text)
                or not block.text[span.start : span.end].strip()
            ):
                raise ValueError("V4 atom span is outside the unchanged original block")
            span_ids.add(key)
        if atom.state == "preserved":
            if not atom.preserved_indices or atom.finding_index is not None or atom.effect is not None:
                raise ValueError("V4 preserved atom must use only indexed original carriers")
            carrier_conditions = set()
            for index in atom.preserved_indices:
                if index < 0 or index >= len(row.preserved_qualifiers):
                    raise ValueError("V4 preserved carrier index is out of range")
                preserved_used.append(index)
                raw = row.preserved_qualifiers[index]
                if raw.restriction != atom.restriction or set(raw.source_block_ids) != span_ids:
                    raise ValueError(
                        "V4 indexed preserved declaration differs from its atomic restriction/source"
                    )
                if raw.claim_path == "/text":
                    affected = atom.condition_ids
                else:
                    token = raw.claim_path.split("/")[2]
                    affected = [claim["conditions"][int(token)]["id"]]
                semantic_value(raw.claim_path, raw.claim_value, affected)
                carrier_conditions.update(affected)
            if carrier_conditions != set(atom.condition_ids):
                raise ValueError("V4 preserved carriers do not cover the atom's affected conditions")
        elif atom.state == "missing":
            index, effect = atom.finding_index, atom.effect
            if atom.preserved_indices or index is None or index < 0 or index >= len(row.findings):
                raise ValueError("V4 missing atom requires one existing qualifier finding index")
            raw = row.findings[index]
            if (
                raw.kind != "missing_qualifier_or_condition"
                or raw.restriction != atom.restriction
                or set(raw.source_block_ids) != span_ids
                or set(raw.condition_ids) != set(atom.condition_ids)
                or raw.reason != atom.reason
            ):
                raise ValueError(
                    "V4 indexed missing declaration differs from its atomic restriction/source/conditions"
                )
            if (
                effect is None
                or effect.kind not in {"verification_setting", "conclusion_boundary"}
                or not effect.original_permits_alternative
                or effect.source_setting is None
                or effect.alternative_setting is None
                or fingerprint(effect.source_setting) == fingerprint(effect.alternative_setting)
                or effect.explanation != raw.material_effect
            ):
                raise ValueError(
                    "V4 missing qualifier needs a closed material verification/conclusion alternative"
                )
            semantic_value(effect.assertion_path, effect.assertion_value, atom.condition_ids)
            finding_used.append(index)
        elif atom.preserved_indices or atom.finding_index is not None or atom.effect is not None:
            raise ValueError("V4 unresolved atom cannot declare preserved or missing scope")
    if sorted(preserved_used) != list(range(len(row.preserved_qualifiers))):
        raise ValueError("V4 preserved fields must be indexed once in the atomic table")
    qualifier_indices = [i for i, f in enumerate(row.findings) if f.kind == "missing_qualifier_or_condition"]
    if sorted(finding_used) != qualifier_indices:
        raise ValueError("V4 missing qualifier fields must be indexed once in the atomic table")
    for group in row.scope_groups:
        if set(group.dimensions) != set(SCOPE_DIMENSIONS):
            raise ValueError("V4 each condition group must close all scope dimensions")
        for dimension, review in group.dimensions.items():
            expected = {
                a.id
                for a in row.scope_atoms
                if a.dimension == dimension and set(a.condition_ids) & set(group.condition_ids)
            }
            if len(set(review.atom_ids)) != len(review.atom_ids) or set(review.atom_ids) != expected:
                raise ValueError("V4 dimension atom references are omitted, repeated or foreign")
            if any(not set(group.condition_ids) <= set(atoms[key].condition_ids) for key in expected):
                raise ValueError("V4 same-scope group combines conditions with different governing atoms")
            referenced.update(review.atom_ids)
            states = {atoms[key].state for key in expected}
            state = next((s for s in ("unresolved", "missing", "preserved") if s in states), "not_governing")
            if review.state != state and not (review.state == "unresolved" and not expected):
                raise ValueError("V4 dimension state contradicts its atomic closure")
            if review.state == "unresolved" and row.state != "unresolved":
                raise ValueError("V4 unresolved scope cannot authorize a resolved claim review")
    if referenced != set(atoms):
        raise ValueError("V4 scope table contains an unclosed atom")
