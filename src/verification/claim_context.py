"""Claim-local paper context for per-claim verification requests.

Used by the experiment scope review and the Code check. The scope reviewer
selects program-generated source/cell/number IDs that the validator resolves
back to original text. Restricting the catalog and blocks to one claim's
neighbourhood shrinks the request; IDs and blocks outside it are unknown to the
resolvers, so trimmed material cannot be selected.
"""

from __future__ import annotations

import re

_SKIP_KINDS = {"page_number", "header", "footer", "ref_text"}
_MENTION = re.compile(r"\b(Table|Tab\.|Figure|Fig\.)\s*(\d+)", re.I)
_CAPTION = re.compile(r"^\s*(Table|Tab\.|Figure|Fig\.)\s*(\d+)\s*[:.|]", re.I)
_MAX_DEFINITION_BLOCKS = 8
# Small tables are sent whole; larger ones are cut to the condition's slice.
_WHOLE_TABLE_CELLS = 60
_SLACK = 0


def _kind(label: str) -> str:
    return "table" if label.lower().startswith("tab") else "figure"


def _mentions(text: str) -> set[tuple[str, str]]:
    return {(_kind(kind), number) for kind, number in _MENTION.findall(text)}


def condition_terms(claim) -> set[str]:
    terms = set()
    for condition in claim.conditions:
        values = [condition.dataset, condition.metric]
        for value in condition.settings.values():
            values.extend(value if isinstance(value, list) else [value])
        terms.update(str(v).strip().casefold() for v in values if isinstance(v, str) and len(v.strip()) >= 3)
    return terms


def scope_block_ids(claim, materials, seed_block_ids=()) -> set[str]:
    """Blocks for one claim: its sources, the candidates' passages, their
    neighbours, the tables/figures they cite and the passages that cite those,
    plus a few blocks naming the condition's dataset/metric/settings."""
    everything = list(materials.blocks)
    position = {block.id: index for index, block in enumerate(everything)}
    blocks = [block for block in everything if block.kind not in _SKIP_KINDS]
    seeds = {claim.source_block_id, *(ref.source_block_id for ref in claim.source_refs), *seed_block_ids}
    seeds = {block_id for block_id in seeds if block_id in position}
    if not seeds:
        # An unanchored claim has no neighbourhood; keep every evidence block.
        return {block.id for block in blocks}
    selected = set(seeds)
    for block_id in seeds:
        for step in (-1, 1):
            index = position[block_id] + step
            while 0 <= index < len(everything) and everything[index].kind in _SKIP_KINDS:
                index += step
            if 0 <= index < len(everything):
                selected.add(everything[index].id)
    by_id = {block.id: block for block in everything}
    captions = {}
    for block in blocks:
        match = _CAPTION.match(block.text)
        if match:
            captions.setdefault((_kind(match.group(1)), match.group(2)), set()).add(block.id)
    cited = set().union(*(_mentions(by_id[block_id].text) for block_id in seeds))
    # A selected caption block is itself a cited table/figure.
    cited |= {key for key, ids in captions.items() if ids & seeds}
    for key in cited:
        selected |= captions.get(key, set())
        # Passages referencing a cited table/figure carry its metric/setup definitions.
        selected |= {block.id for block in blocks if key in _mentions(block.text)}
    terms = condition_terms(claim)
    if terms:
        anchors = [position[block_id] for block_id in seeds]
        matches = [
            block for block in blocks
            if block.id not in selected
            and "<table" not in block.text.casefold()
            and any(term in block.text.casefold() for term in terms)
        ]
        # ponytail: plain substring match ranked by distance to the claim; a
        # definition far from every seed and beyond the cap is not sent.
        matches.sort(key=lambda block: min(abs(position[block.id] - anchor) for anchor in anchors))
        selected.update(block.id for block in matches[:_MAX_DEFINITION_BLOCKS])
    return selected


def condition_groups(claim) -> list[list[set[str]]]:
    """Per condition, one term set per field (dataset, metric, each setting)."""
    result = []
    for condition in claim.conditions:
        fields = [condition.dataset, condition.metric, *condition.settings.values()]
        groups = []
        for value in fields:
            values = value if isinstance(value, list) else [value]
            terms = {
                part.strip().casefold()
                for item in values
                if isinstance(item, str)
                for part in item.split(" / ")
                if len(part.strip()) >= 3
            }
            if terms:
                groups.append(terms)
        result.append(groups)
    return result


def _relevant_cells(cells: dict, groups: list[list[set[str]]]) -> dict:
    """Per table: every label cell plus, for each condition, the number cells
    whose row/column axes name the most condition fields. A table no condition
    names at all stays whole."""
    by_table = {}
    for key, cell in cells.items():
        by_table.setdefault(cell["table_id"], {})[key] = cell
    kept = {}
    for table_cells in by_table.values():
        numbers = {key: cell for key, cell in table_cells.items() if cell["cell_type"] == "number"}
        if len(numbers) <= _WHOLE_TABLE_CELLS:
            kept.update(table_cells)
            continue
        chosen = set()
        for condition_groups in groups:
            scores = {
                key: sum(
                    any(term in f"{cell['row_labels']} | {cell['column_labels']}".casefold() for term in terms)
                    for terms in condition_groups
                )
                for key, cell in numbers.items()
            }
            best = max(scores.values(), default=0)
            # ponytail: substring match on axis labels; a condition whose names
            # differ from the table's spelling falls back to its broader slice.
            if best:
                chosen.update(key for key, score in scores.items() if score >= best - _SLACK)
        if not chosen:
            kept.update(table_cells)
            continue
        kept.update((key, cell) for key, cell in table_cells.items() if key in chosen or key not in numbers)
    return kept


def local_catalog(catalog: dict, block_ids: set[str], groups: list | None = None) -> dict:
    """Restrict sources/cells/numbers/tables to the selected blocks, and the
    number cells of each kept table to the condition's slice (see _relevant_cells)."""
    sources = {
        key: source
        for key, source in catalog["sources"].items()
        if source["kind"] == "claim_text" or source.get("block_id") in block_ids
    }
    tables = {
        key: {
            **table,
            "reference_source_ids": [ref for ref in table.get("reference_source_ids", []) if ref in sources],
        }
        for key, table in catalog["tables"].items()
        if table["source_id"] in sources
    }
    result = dict(catalog)
    result["sources"] = sources
    result["tables"] = tables
    result["cells"] = _relevant_cells(
        {key: cell for key, cell in catalog["cells"].items() if cell["table_id"] in tables}, groups or []
    )
    result["numbers"] = {
        key: number for key, number in catalog.get("numbers", {}).items() if number["block_id"] in block_ids
    }
    if "assertion_numbers" in catalog:
        result["assertion_numbers"] = {
            key: number for key, number in catalog["assertion_numbers"].items() if number["block_id"] in block_ids
        }
    return result


def local_materials(materials, block_ids: set[str]):
    return materials.model_copy(update={"blocks": [b for b in materials.blocks if b.id in block_ids]})


def candidate_block_ids(output) -> set[str]:
    ids = set()
    for item in output.items:
        ids.add(item.block_id)
        ids.update(passage.block_id for passage in item.additional_sources)
        ids.update(number.block_id for number in item.comparison)
    for plan in output.plans:
        ids.update(target.reported.block_id for target in plan.targets)
    return ids


_LONG_ID = re.compile(r"(?:src|cell|table|axis|number|sentence|case)_[0-9a-f]{12,}")


def alias_ids(payload):
    """Replace long hash IDs with short handles (#0, #1, ...) for the model.

    Returns the aliased payload and the alias -> ID map that ``dealias``
    applies to the response, so the resolvers still see the original IDs."""
    aliases, handles = {}, {}

    def handle(value):
        if not _LONG_ID.fullmatch(value):
            return value
        if value not in handles:
            handles[value] = f"#{len(handles)}"
            aliases[handles[value]] = value
        return handles[value]

    def walk(value):
        if isinstance(value, dict):
            return {handle(key) if isinstance(key, str) else key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return handle(value) if isinstance(value, str) else value

    return walk(payload), aliases


def dealias(value, aliases):
    if isinstance(value, dict):
        return {aliases.get(key, key) if isinstance(key, str) else key: dealias(v, aliases) for key, v in value.items()}
    if isinstance(value, list):
        return [dealias(item, aliases) for item in value]
    return aliases.get(value, value) if isinstance(value, str) else value


_EXPERIMENT_SECTION = re.compile(
    r"experiment|evaluat|result|setup|setting|implementation|ablation|analys|benchmark|dataset|baseline"
    r"|training|hyper-?param|comparison|study|studies|detail",
    re.I,
)
_CONTRIBUTION = re.compile(r"contribution|we (?:propose|introduce|present|show|demonstrate)", re.I)


def _section_title(block) -> str:
    section = (block.loc.section if block.loc else "") or ""
    return section.split(":", 1)[-1]


def experiment_block_ids(claim, materials) -> set[str]:
    """Blocks for the experiment check's first pass.

    Kept: the claim's own context (scope_block_ids); every block of a section
    whose heading names experiments, evaluation, results, setup/settings,
    implementation, ablations, analysis, benchmarks, datasets, baselines,
    training/hyperparameters, comparisons/studies or details, and of its numbered
    subsections (5 -> 5.3); every block of a section
    holding a table; the abstract; and introduction paragraphs stating
    contributions ("contribution", "we propose/introduce/present/show/demonstrate").
    Setup and ablation passages needed for fairness/isolation/consistency
    checks live in those sections. A paper whose headings match nothing keeps
    every evidence block.
    """
    blocks = [block for block in materials.blocks if block.kind not in _SKIP_KINDS]
    table_sections = {
        _section_title(block) for block in blocks if block.kind == "table" or "<table" in block.text.casefold()
    }
    titles = {_section_title(block) for block in blocks}
    experimental = {title for title in titles if _EXPERIMENT_SECTION.search(title)}
    # Numbered subsections of an experimental section (5 -> 5.3) belong to it.
    numbers = {m[1] for title in experimental if (m := re.match(r"\s*([A-Z]?\d+(?:\.\d+)*)\b", title))}
    experimental |= {
        title
        for title in titles
        if (m := re.match(r"\s*([A-Z]?\d+(?:\.\d+)*)\b", title))
        and any(m[1].startswith(number + ".") for number in numbers)
    }
    if not experimental and not table_sections:
        return {block.id for block in blocks}
    selected = scope_block_ids(claim, materials)
    for block in blocks:
        title = _section_title(block)
        if (
            title in experimental
            or title in table_sections
            or re.search(r"abstract", title, re.I)
            or (re.search(r"introduction", title, re.I) and _CONTRIBUTION.search(block.text))
        ):
            selected.add(block.id)
    return selected
