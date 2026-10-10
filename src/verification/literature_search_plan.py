"""Small deterministic source-bound seed plan; roles grant no scientific sufficiency."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy

from fact_generation.positioning.structured_query import (
    INTENTS,
    SEEDS,
    QueryTerm,
    StructuredPaperQuery,
    digest,
)


def plan_digest(plan):
    value = deepcopy({key: item for key, item in plan.items() if key != "digest"})
    for query in value["queries"]:
        query.pop("plan_digest", None)
    return digest(value)


def build_grounded_plan(claim, materials, *, novelty_ids):
    from verification.literature import _claim_source_excerpts

    ids = sorted(novelty_ids)
    concepts, errors, omitted = [], [], []
    try:
        excerpts = _claim_source_excerpts(claim, materials)
    except ValueError as exc:
        excerpts = []
        errors.append(str(exc))
    bibliography_ids = {block.id for block in materials.bibliography}
    for excerpt in excerpts:
        covered = sorted(set(excerpt["covered"]) & set(ids))
        if not covered:
            continue
        block = next(b for b in materials.blocks if b.id == excerpt["source_block_id"])
        if block.id in bibliography_ids or block.kind in {
            "heading",
            "section_header",
            "title",
            "metadata",
            "author",
            "authors",
            "affiliation",
            "affiliations",
            "bibliography",
            "reference",
            "references",
        }:
            continue
        text = excerpt["source_quote"]
        # A one-codepoint hyphen substitution preserves Python source offsets.
        normalized = text.translate(str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2212"}))
        source_start = block.text.find(text)
        for family, (role, variants) in SEEDS.items():
            pattern = (
                r"(?<!\w)(?:"
                + "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
                + r")(?!\w)"
            )
            match = re.search(pattern, normalized, re.I)
            if not match:
                continue
            # Reject matches inside a URL or an explicit field/operator payload.
            token_start = max(text.rfind(" ", 0, match.start()), text.rfind("\n", 0, match.start())) + 1
            token_end = text.find(" ", match.end())
            token_end = len(text) if token_end < 0 else token_end
            token = text[token_start:token_end]
            if any(char in token for char in ":/\\"):
                continue
            quote = text[match.start() : match.end()]
            concepts.append(
                {
                    "concept_id": f"s{len(concepts) + 1}",
                    "family": family,
                    "role": role,
                    "phrase": quote,
                    "condition_ids": covered,
                    "source_block_id": block.id,
                    "loc": deepcopy(excerpt["loc"]),
                    "block_local_span": {
                        "start": source_start + match.start(),
                        "end": source_start + match.end(),
                    },
                    "quote": quote,
                    "block_sha256": hashlib.sha256(block.text.encode()).hexdigest(),
                    "excerpt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "term_sha256": hashlib.sha256(quote.encode()).hexdigest(),
                }
            )
    # Keep the compact six-family budget explicit. All dropped conditions remain ineligible.
    families = []
    for role in INTENTS:
        for concept in concepts:
            if concept["role"] == role and concept["family"] not in families:
                families.append(concept["family"])
    retained_families = set(families[:6])
    omitted = [c["concept_id"] for c in concepts if c["family"] not in retained_families]
    selected = [c for c in concepts if c["family"] in retained_families]
    groups = {}
    for role in INTENTS:
        unique = {}
        for c in selected:
            if c["role"] == role:
                key = c["phrase"].casefold()
                if key in unique:
                    previous = unique[key]
                    unique[key] = QueryTerm(
                        concept_id=previous.concept_id,
                        family=previous.family,
                        phrase=previous.phrase,
                        source_concept_ids=[*previous.source_concept_ids, c["concept_id"]],
                    )
                else:
                    unique[key] = QueryTerm(
                        concept_id=c["concept_id"],
                        family=c["family"],
                        phrase=c["phrase"],
                        source_concept_ids=[c["concept_id"]],
                    )
        groups[role] = list(unique.values())
    queries = []
    for role in INTENTS:
        if not groups[role]:
            continue
        roles = [role] if role == "mechanism" or not groups["mechanism"] else ["mechanism", role]
        bound = [
            cid
            for cid in ids
            if all(any(c["role"] == r and cid in c["condition_ids"] for c in selected) for r in roles)
        ]
        if not bound:
            continue
        queries.append(
            StructuredPaperQuery(
                query_id=f"q{len(queries) + 1}",
                intent=role,
                condition_ids=bound,
                groups=[groups[r] for r in roles],
                participation={
                    cid: {
                        r: [c["concept_id"] for c in selected if c["role"] == r and cid in c["condition_ids"]]
                        for r in roles
                    }
                    for cid in bound
                },
                plan_digest="0" * 64,
            ).model_dump(mode="json")
        )
    expressions = [
        StructuredPaperQuery.model_validate(q).compile(start=0, limit=1).expression for q in queries
    ]
    distinct = bool(expressions) and len(expressions) == len(set(expressions))
    conditions = {}
    for cid in ids:
        available = [
            r for r in INTENTS if any(c["role"] == r and cid in c["condition_ids"] for c in selected)
        ]
        missing = [r for r in INTENTS if r not in available]
        truncated = any(cid in c["condition_ids"] and c["concept_id"] in omitted for c in concepts)
        conditions[cid] = {
            "available_roles": available,
            "missing_roles": missing,
            "truncated": truncated,
            "eligible": bool(not errors and not missing and not truncated and distinct and len(queries) == 3),
        }
    plan = {
        "version": "grounded-search-plan-v1",
        "claim_id": claim.id,
        "claim_sha256": digest(claim.model_dump(mode="json")),
        "concepts": concepts,
        "queries": queries,
        "conditions": conditions,
        "source_errors": errors,
        "omitted_concepts": omitted,
        "distinct_queries": distinct,
        "limitations": [
            "Closed source seed vocabulary; differently named mechanisms and unsearched indexes remain outside this scope."
        ],
    }
    plan["digest"] = plan_digest(plan)
    for q in plan["queries"]:
        q["plan_digest"] = plan["digest"]
    return plan


def query_strings(plan):
    return [
        StructuredPaperQuery.model_validate(q).compile(start=0, limit=1).expression for q in plan["queries"]
    ]


def free_text_queries(plan):
    return [
        " ".join(dict.fromkeys(term["phrase"] for group in q["groups"] for term in group))
        for q in plan["queries"]
    ]


def condition_eligible(plan, cid):
    try:
        return plan_closed(plan) and plan["conditions"][cid]["eligible"] is True
    except (KeyError, TypeError, ValueError):
        return False


def plan_closed(plan):
    """Every condition witness must be an actual original-covered concept in its query."""
    try:
        if plan["digest"] != plan_digest(plan) or not plan["queries"]:
            return False
        concepts = {c["concept_id"]: c for c in plan["concepts"]}
        if len(concepts) != len(plan["concepts"]):
            return False
        for raw in plan["queries"]:
            query = StructuredPaperQuery.model_validate(raw)
            if query.plan_digest != plan["digest"]:
                return False
            query.compile(start=0, limit=1)
            for group in query.groups:
                for term in group:
                    for sid in term.source_concept_ids:
                        concept = concepts[sid]
                        if (
                            sid in plan["omitted_concepts"]
                            or concept["family"] != term.family
                            or concept["phrase"].casefold() != term.phrase.casefold()
                            or concept["role"] != SEEDS[term.family][0]
                        ):
                            return False
            for cid, roles in query.participation.items():
                if cid not in plan["conditions"]:
                    return False
                for role, ids in roles.items():
                    actual = {
                        sid
                        for group in query.groups
                        for term in group
                        for sid in term.source_concept_ids
                        if concepts[sid]["role"] == role and cid in concepts[sid]["condition_ids"]
                    }
                    if set(ids) != actual:
                        return False
        return True
    except (ValueError, TypeError, KeyError):
        return False
