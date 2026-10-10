"""Closed source-term query grammar; no raw expression or author field enters transport."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict, Field, model_validator

INTENTS = ("mechanism", "target setting", "evaluation protocol baseline")
# Families are general scientific seeds. Every transmitted spelling must occur in a source.
SEEDS = {
    "pseudo_label": (
        "mechanism",
        ("pseudo-label", "pseudo-labels", "pseudo-labeling", "pseudo-labelling", "pseudo labels"),
    ),
    "self_training": ("mechanism", ("self-training", "self training")),
    "consistency_regularization": ("mechanism", ("consistency regularization",)),
    "graph_convolution": ("mechanism", ("graph convolution", "graph convolutional network")),
    "graph_neural_network": ("mechanism", ("graph neural network",)),
    "message_passing": ("mechanism", ("message passing",)),
    "attention": ("mechanism", ("attention", "self-attention")),
    "transformer": ("mechanism", ("transformer", "transformers")),
    "semi_supervised": (
        "target setting",
        ("semi-supervised learning", "semi-supervised", "semi supervised learning"),
    ),
    "unlabeled_image": ("target setting", ("unlabeled images", "unlabelled images")),
    "image_classification": ("target setting", ("image classification",)),
    "node_classification": ("target setting", ("node classification",)),
    "link_prediction": ("target setting", ("link prediction",)),
    "text_classification": ("target setting", ("text classification",)),
    "transductive": ("evaluation protocol baseline", ("transductive",)),
    "inductive": ("evaluation protocol baseline", ("inductive",)),
    "ablation": ("evaluation protocol baseline", ("ablation", "ablations")),
    "accuracy": ("evaluation protocol baseline", ("accuracy",)),
    "f1": ("evaluation protocol baseline", ("f1", "f1-score", "f1 score")),
    "precision": ("evaluation protocol baseline", ("precision",)),
    "recall": ("evaluation protocol baseline", ("recall",)),
    "cifar10": ("evaluation protocol baseline", ("cifar-10", "cifar10")),
    "cifar100": ("evaluation protocol baseline", ("cifar-100", "cifar100")),
    "svhn": ("evaluation protocol baseline", ("svhn",)),
    "stl10": ("evaluation protocol baseline", ("stl-10", "stl10")),
}
_HYPHENS = str.maketrans({char: "-" for char in "\u2010\u2011\u2012\u2013\u2212"})


def normalized_phrase(text):
    return " ".join(text.translate(_HYPHENS).casefold().split())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class QueryTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    concept_id: str = Field(min_length=1)
    family: str
    phrase: str
    source_concept_ids: list[str]

    @model_validator(mode="after")
    def safe_phrase(self):
        if (
            not self.source_concept_ids
            or self.concept_id not in self.source_concept_ids
            or len(set(self.source_concept_ids)) != len(self.source_concept_ids)
        ):
            raise ValueError("Term source references must be explicit and distinct")
        if self.family not in SEEDS or normalized_phrase(self.phrase) not in SEEDS[self.family][1]:
            raise ValueError("Term is outside the closed scientific seed vocabulary")
        if any(char in self.phrase for char in ':"()[]{}\\/') or re.search(
            r"\b(?:and|or|andnot)\b", self.phrase, re.I
        ):
            raise ValueError("Raw query syntax, URLs and author fields are forbidden")
        return self


@dataclass(frozen=True)
class CompiledQuery:
    expression: str
    params: dict
    url: str
    digest: str


class StructuredPaperQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["structured-arxiv-query-v1"] = "structured-arxiv-query-v1"
    query_id: str = Field(pattern=r"^q[1-3]$")
    intent: Literal["mechanism", "target setting", "evaluation protocol baseline"]
    condition_ids: list[str]
    groups: list[list[QueryTerm]]
    participation: dict[str, dict[str, list[str]]]
    plan_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def scientific_groups(self):
        expected = [self.intent] if len(self.groups) == 1 else ["mechanism", self.intent]
        if self.intent == "mechanism" and len(self.groups) != 1:
            raise ValueError("Mechanism query requires one actual mechanism group")
        if (
            len(self.groups) != len(expected)
            or not self.condition_ids
            or len(set(self.condition_ids)) != len(self.condition_ids)
        ):
            raise ValueError("Query needs distinct condition IDs and an actual dimension")
        if set(self.participation) != set(self.condition_ids):
            raise ValueError("Query condition participation is not closed")
        for group, role in zip(self.groups, expected, strict=True):
            if not group or any(SEEDS[term.family][0] != role for term in group):
                raise ValueError("Query group differs from its scientific role")
            if len({normalized_phrase(term.phrase) for term in group}) != len(group):
                raise ValueError("Duplicate term alternatives cannot create query coverage")
        for cid in self.condition_ids:
            row = self.participation[cid]
            if set(row) != set(expected):
                raise ValueError("Each queried condition needs the actual group roles")
            for group, role in zip(self.groups, expected, strict=True):
                available = {sid for term in group for sid in term.source_concept_ids}
                ids = row[role]
                if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(available):
                    raise ValueError("Condition references are outside actual query sources")
        return self

    def compile(self, *, start: int, limit: int) -> CompiledQuery:
        # Revalidate even model_copy/model_construct or mutated list contents.
        query = type(self).model_validate(self.model_dump(mode="json"))
        if type(start) is not int or start < 0 or type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError("Invalid arXiv page bounds")
        groups = []
        for group in query.groups:
            leaves = sorted({f'{field}:"{term.phrase}"' for term in group for field in ("ti", "abs")})
            groups.append("(" + " OR ".join(leaves) + ")")
        expression = " AND ".join(groups)
        params = {
            "search_query": expression,
            "start": start,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        return CompiledQuery(
            expression,
            params,
            "https://export.arxiv.org/api/query?" + urlencode(params),
            digest(query.model_dump(mode="json")),
        )
