"""Deterministic, compact choices for original experiment passages and table cells.

The catalog contains references to manuscript bytes, never inferred metric/role
bindings. Resolvers recover exact quotes locally; models need not copy tables or
count their rows. Neither construction nor resolution calls external services.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from html.parser import HTMLParser
from itertools import product

from schemas.claim import Claim
from schemas.materials import SharedMaterials

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?")
_TABLE = re.compile(r"<table\b.*?</table>", re.I | re.S)
_CAPTION = re.compile(r"(?:^|\n)\s*(Table\s+(\d+))\s*[:.](?!\d)", re.I)


class TableGrid(HTMLParser):
    """Expand the existing native-HTML table coordinates, without rendering HTML."""

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tables, self.rows, self.row, self.cell = [], None, None, None
        self.headers, self.origins, self.metadata, self.in_thead = [], [], [], False
        self.feed(text)
        self.close()
        if self.rows is not None or self.row is not None or self.cell is not None:
            raise ValueError("Unclosed table has ambiguous numerical coordinates")

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.rows is not None:
                raise ValueError("Nested tables have ambiguous numerical coordinates")
            self.rows = []
            self.in_thead = False
        elif tag == "thead" and self.rows is not None:
            self.in_thead = True
        elif tag == "tr" and self.rows is not None:
            if self.row is not None:
                raise ValueError("Unclosed table row")
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            if self.cell is not None:
                raise ValueError("Unclosed table cell")
            attrs = dict(attrs)
            try:
                spans = [int(attrs.get(name, "1")) for name in ("rowspan", "colspan")]
            except (TypeError, ValueError):
                raise ValueError("Unsupported table span") from None
            if any(value < 1 or value > 100 for value in spans):
                raise ValueError("Unsupported table span")
            self.cell = [
                [],
                *spans,
                {
                    "tag": tag,
                    "thead": self.in_thead,
                    "scope": (attrs.get("scope") or "").lower(),
                    "rowspan": spans[0],
                    "colspan": spans[1],
                },
            ]

    def handle_data(self, data):
        if self.cell is not None:
            self.cell[0].append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(("".join(self.cell[0]).strip(), *self.cell[1:]))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.cell is not None:
                raise ValueError("Unclosed table cell")
            self.rows.append(self.row)
            self.row = None
        elif tag == "thead":
            self.in_thead = False
        elif tag == "table" and self.rows is not None:
            if self.row is not None:
                raise ValueError("Unclosed table row")
            grid, headers, origins, metadata = {}, {}, {}, {}
            for row_index, row in enumerate(self.rows):
                column = 0
                for text, rowspan, colspan, meta in row:
                    while (row_index, column) in grid:
                        column += 1
                    for y in range(row_index, row_index + rowspan):
                        for x in range(column, column + colspan):
                            if (y, x) in grid:
                                raise ValueError("Overlapping table spans")
                            grid[y, x] = text
                            headers[y, x] = meta["tag"] == "th" or meta["thead"]
                            origins[y, x] = (row_index, column)
                            metadata[y, x] = meta.copy()
                    column += colspan
            self.tables.append(grid)
            self.headers.append(headers)
            self.origins.append(origins)
            self.metadata.append(metadata)
            self.rows = None


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(kind, *parts):
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{kind}_{_hash(payload)[:24]}"


def _put(records, key, value):
    if key in records and records[key] != value:
        raise ValueError(f"Catalog ID collision: {key}")
    records[key] = value
    return key


def _blocks(materials):
    blocks = {block.id: block for block in materials.blocks}
    if len(blocks) != len(materials.blocks):
        raise ValueError("Catalog requires unique material block IDs")
    return blocks


def _location(block):
    return block.loc.model_dump(exclude_none=True) if block.loc is not None else None


def _source(catalog, block, start, end, kind, covered=()):
    if block.loc is None or not 0 <= start < end <= len(block.text):
        raise ValueError("Catalog source needs an exact located nonempty span")
    record = {
        "block_id": block.id,
        "block_sha256": _hash(block.text),
        "start": start,
        "end": end,
        "loc": _location(block),
        "kind": kind,
        "covered": sorted(covered),
    }
    key = _stable_id("src", catalog["paper_key"], catalog["source_pdf"], record)
    return _put(catalog["sources"], key, record)


def _exact_source(catalog, blocks, block_id, quote, kind, covered):
    block = blocks.get(block_id)
    if block is None or not quote or block.text.count(quote) != 1:
        raise ValueError("Claim source must identify one exact original passage")
    start = block.text.index(quote)
    return _source(catalog, block, start, start + len(quote), kind, covered)


def _labels(values):
    # These are literal axis strings, without guessing which one is a metric.
    return " | ".join(
        dict.fromkeys(value for value in values if value and value != "-" and not _NUMBER.fullmatch(value))
    )


def _cases(condition, max_cases):
    dimensions = {key: values for key, values in condition.settings.items() if isinstance(values, list)}
    if any(
        not values or any(isinstance(value, (dict, list)) for value in values)
        for values in dimensions.values()
    ):
        raise ValueError("Case dimensions must be nonempty lists of scalar input values")
    if math.prod(len(values) for values in dimensions.values()) > max_cases:
        raise ValueError(f"Condition exceeds the {max_cases} catalog case limit")
    cases = []
    seen = set()
    for values in product(*dimensions.values()):
        settings = {key: str(value) for key, value in zip(dimensions, values, strict=True)}
        key = _stable_id("case", condition.model_dump(mode="json"), settings)
        if key in seen:
            raise ValueError("Duplicate or colliding condition cases")
        seen.add(key)
        cases.append({"id": key, "settings": settings})
    return cases


def _table_references(catalog, blocks):
    """Index literal table mentions, without asserting a manuscript relationship.

    A reference may describe another paper's table. Every returned ID is only
    a source candidate; the scope reviewer must still establish applicability.
    Reuse whole-block source IDs so neither quotes nor locations are duplicated.
    """
    sources = [(key, source) for key, source in catalog["sources"].items() if source["kind"] == "paper_block"]
    for table in catalog["tables"].values():
        table["reference_source_ids"] = []
        if not table["caption_source_id"]:
            continue
        caption = catalog["sources"][table["caption_source_id"]]
        prefix = blocks[caption["block_id"]].text[caption["start"] : caption["end"]]
        labels = list(_CAPTION.finditer(prefix))
        if not labels:
            continue
        # MinerU can prepend a previous table's caption. Only the last caption
        # immediately preceding this HTML table names the selected table.
        mention = re.compile(r"(?<!\w)Table\s+" + labels[-1].group(2) + r"(?!\w|\.\d)", re.I)
        table["reference_source_ids"] = sorted(
            key
            for key, source in sources
            if source["block_id"] != caption["block_id"] and mention.search(blocks[source["block_id"]].text)
        )


def build_catalog(claim: Claim, materials: SharedMaterials, *, max_cases: int = 256) -> dict:
    """Return compact exact-source choices, with no inferred aliases or values.

    ``cases`` enumerate the existing list-valued condition settings contract;
    they do not add a claim quantifier. Scalar conditions have one empty case.
    Source and cell records intentionally omit full manuscript quote strings.
    """
    from verification.prose_numbers import index_numbers

    blocks = _blocks(materials)
    ids = [condition.id for condition in claim.conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalog requires unique condition IDs")
    catalog = {
        "version": 1,
        "paper_key": materials.paper_key,
        "source_pdf": materials.source_pdf,
        "claim_id": claim.id,
        "sources": {},
        "cells": {},
        "numbers": index_numbers(materials),
        "tables": {},
        "conditions": {},
        "issues": [],
    }
    claim_source = {
        "kind": "claim_text",
        "text": claim.text,
        "text_sha256": _hash(claim.text),
        "loc": claim.loc.model_dump(exclude_none=True),
        "covered": ids,
    }
    catalog["claim_source_id"] = _put(
        catalog["sources"], _stable_id("src", claim.id, claim_source), claim_source
    )
    for condition in claim.conditions:
        try:
            cases = _cases(condition, max_cases)
        except ValueError as exc:
            cases = []
            catalog["issues"].append(f"{condition.id}: {exc}")
        catalog["conditions"][condition.id] = {
            "field_values": {
                "dataset": condition.dataset,
                "metric": condition.metric,
                "settings": copy.deepcopy(condition.settings),
            },
            "cases": cases,
        }
    explicit_primary = [
        ref
        for ref in claim.source_refs
        if ref.source_block_id == claim.source_block_id and ref.source_quote == claim.source_quote
    ]
    if claim.source_block_id is not None:
        coverage = (
            sorted({value for ref in explicit_primary for value in ref.covered}) if explicit_primary else ids
        )
        _exact_source(catalog, blocks, claim.source_block_id, claim.source_quote, "claim_source", coverage)
    for ref in claim.source_refs:
        if not set(ref.covered).issubset(ids):
            raise ValueError("Claim source covers an unknown condition")
        _exact_source(catalog, blocks, ref.source_block_id, ref.source_quote, "claim_source", ref.covered)
    for block in blocks.values():
        if block.loc is None or not block.text.strip():
            continue
        source_id = _source(catalog, block, 0, len(block.text), "paper_block")
        if not re.search(r"<table\b", block.text, re.I):
            continue
        try:
            tables, spans = TableGrid(block.text).tables, list(_TABLE.finditer(block.text))
            if len(tables) != len(spans):
                raise ValueError("Unresolved original table boundaries")
        except ValueError as exc:
            catalog["issues"].append(f"{block.id}: {exc}")
            continue
        for line in re.finditer(r"[^\r\n]+", block.text):
            if line.group().strip() and not any(
                line.start() < span.end() and line.end() > span.start() for span in spans
            ):
                _source(catalog, block, line.start(), line.end(), "paper_line")
        for index, (grid, span) in enumerate(zip(tables, spans, strict=True)):
            prefix_start = spans[index - 1].end() if index else 0
            prefix = block.text[prefix_start : span.start()]
            caption_id = (
                _source(catalog, block, prefix_start, span.start(), "table_prefix")
                if prefix.strip()
                else None
            )
            table_id = _stable_id("table", source_id, index)
            _put(
                catalog["tables"],
                table_id,
                {
                    "source_id": source_id,
                    "table": index,
                    "caption_source_id": caption_id,
                    "caption_ambiguous": len(_CAPTION.findall(prefix)) > 1,
                },
            )
            for (row, column), token in grid.items():
                if not token.strip():
                    continue
                numeric = bool(_NUMBER.fullmatch(token))
                if numeric and (
                    not math.isfinite(float(token.rstrip("%")))
                    or not re.search(r"(?<![\w.])" + re.escape(token) + r"(?![\w%]|\.\d)", block.text)
                ):
                    continue
                record = {
                    "source_id": source_id,
                    "table_id": table_id,
                    "token": token,
                    "cell_type": "number" if numeric else "label",
                    "table": index,
                    "row": row,
                    "column": column,
                    "row_labels": _labels(v for (r, c), v in grid.items() if r == row and c < column),
                    "column_labels": _labels(v for (r, c), v in grid.items() if c == column and r < row),
                }
                _put(catalog["cells"], _stable_id("cell", source_id, index, row, column, token), record)
    _table_references(catalog, blocks)
    return catalog


def resolve_source(catalog: dict, source_id: str, materials: SharedMaterials) -> dict:
    """Recover the original contiguous quote, rejecting unknown or stale sources."""
    if catalog.get("paper_key") != materials.paper_key or catalog.get("source_pdf") != materials.source_pdf:
        raise ValueError("Catalog belongs to different materials")
    source = catalog["sources"].get(source_id)
    if source is None:
        raise ValueError(f"Unknown catalog source ID: {source_id}")
    if source["kind"] == "claim_text":
        if _hash(source["text"]) != source["text_sha256"]:
            raise ValueError("Catalog claim text changed")
        return {"block_id": None, "quote": source["text"], **copy.deepcopy(source)}
    block = _blocks(materials).get(source["block_id"])
    if block is None or _hash(block.text) != source["block_sha256"] or _location(block) != source["loc"]:
        raise ValueError("Catalog material source changed or is unavailable")
    if _stable_id("src", catalog["paper_key"], catalog["source_pdf"], source) != source_id:
        raise ValueError("Catalog source record changed")
    start, end = source["start"], source["end"]
    if not 0 <= start < end <= len(block.text):
        raise ValueError("Invalid catalog source span")
    result = {**copy.deepcopy(source), "quote": block.text[start:end]}
    if catalog.get("joint_view") is not None:
        from verification.experiment_sources import require_passage

        require_passage(catalog, materials, result["block_id"], result["quote"], purpose="catalog_source")
    return result


def resolve_cell(catalog: dict, cell_id: str, materials: SharedMaterials) -> dict:
    """Return PaperNumber fields and zero-based TableCell coordinates from a chosen ID."""
    cell = catalog["cells"].get(cell_id)
    if cell is None:
        raise ValueError(f"Unknown catalog cell ID: {cell_id}")
    source = resolve_source(catalog, cell["source_id"], materials)
    expected_id = _stable_id(
        "cell", cell["source_id"], cell["table"], cell["row"], cell["column"], cell["token"]
    )
    if expected_id != cell_id:
        raise ValueError("Catalog cell record changed")
    tables = TableGrid(source["quote"]).tables
    if (
        cell["table"] >= len(tables)
        or tables[cell["table"]].get((cell["row"], cell["column"])) != cell["token"]
    ):
        raise ValueError("Catalog cell does not match the original table")
    table = catalog["tables"][cell["table_id"]]
    caption = (
        resolve_source(catalog, table["caption_source_id"], materials)["quote"]
        if table["caption_source_id"]
        else ""
    )
    if "joint_view" in catalog:
        record = {
            "kind": "cell",
            "selected_id": cell_id,
            "parent_cell_id": cell["parent_cell_id"],
            "origin_table_id": cell["origin_table_id"],
            "origin_table_index": cell["origin_table_index"],
            "quote_table_index": cell["table"],
            "row": cell["row"],
            "column": cell["column"],
            "token": cell["token"],
            "source_id": cell["source_id"],
            "caption_source_id": table["caption_source_id"],
        }
        trace = catalog["joint_view"].setdefault("selector_consumption", [])
        if record not in trace:
            trace.append(record)
    return {
        **copy.deepcopy(cell),
        "block_id": source["block_id"],
        "quote": source["quote"],
        "caption": caption,
        "caption_ambiguous": table["caption_ambiguous"],
    }


def resolve_case(catalog: dict, condition_id: str, case_id: str) -> dict:
    condition = catalog["conditions"].get(condition_id)
    if condition is None:
        raise ValueError(f"Unknown catalog condition ID: {condition_id}")
    case = next((case for case in condition["cases"] if case["id"] == case_id), None)
    if case is None:
        raise ValueError(f"Unknown case ID for condition {condition_id}: {case_id}")
    return copy.deepcopy(case)


def catalog_prompt(catalog: dict) -> dict:
    """Small model-facing index; resolve returned IDs against the full local catalog.

    Field names are shared once instead of repeated for hundreds of cells. The
    original claim and paper blocks already supply the text. Hashes and full
    locations stay in the local catalog used by the resolvers and audit.
    """
    from verification.prose_numbers import sentence_id

    source_fields = ["block_id", "start", "end", "kind", "covered"]
    cell_fields = ["table_id", "row", "column", "token", "cell_type", "row_axis_id", "column_axis_id"]
    number_fields = ["block_id", "token", "unit_suffix", "sentence_id", "ordinal"]
    sentences, numbers, ordinals = {}, {}, {}
    for identifier, record in catalog.get("numbers", {}).items():
        sentence_key = sentence_id(record)
        sentences[sentence_key] = record["sentence"]
        ordinals[sentence_key] = ordinals.get(sentence_key, 0) + 1
        numbers[identifier] = [
            record["block_id"],
            record["token"],
            record["unit_suffix"],
            sentence_key,
            ordinals[sentence_key],
        ]
    assertion_numbers = {}
    for identifier, record in catalog.get("assertion_numbers", {}).items():
        sentence_key = sentence_id(record)
        sentences[sentence_key] = record["sentence"]
        assertion_numbers[identifier] = [
            record["block_id"],
            record["token"],
            record["unit_suffix"],
            sentence_key,
            record["start"],
            record["end"],
        ]
    axes = {}

    def axis_id(text):
        if not text:
            return None
        return _put(axes, "axis_" + _hash(text)[:12], text)

    cells = {
        key: [
            cell["table_id"],
            cell["row"],
            cell["column"],
            cell["token"],
            cell["cell_type"],
            axis_id(cell["row_labels"]) if cell["cell_type"] == "number" else None,
            axis_id(cell["column_labels"]) if cell["cell_type"] == "number" else None,
        ]
        for key, cell in catalog["cells"].items()
    }
    return {
        "version": catalog["version"],
        "claim_id": catalog["claim_id"],
        "claim_source_id": catalog["claim_source_id"],
        "source_fields": source_fields,
        "sources": {
            key: [copy.deepcopy(source.get(field)) for field in source_fields]
            for key, source in catalog["sources"].items()
        },
        "cell_fields": cell_fields,
        "cells": cells,
        "number_fields": number_fields,
        "numbers": numbers,
        **(
            {
                "assertion_number_fields": [
                    "block_id",
                    "token",
                    "unit_suffix",
                    "sentence_id",
                    "original_start",
                    "original_end",
                ],
                "assertion_numbers": assertion_numbers,
            }
            if "joint_view" in catalog
            else {}
        ),
        "sentences": sentences,
        "axes": axes,
        "tables": copy.deepcopy(catalog["tables"]),
        "conditions": copy.deepcopy(catalog["conditions"]),
        "issues": list(catalog["issues"]),
        **(
            {
                "candidate_id": catalog["joint_view"]["candidate_id"],
                "member_source_ids": catalog["joint_view"]["member_source_ids"],
            }
            if "joint_view" in catalog
            else {}
        ),
    }
