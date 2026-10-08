"""Original numeric occurrences and conservative, explicit prose bindings."""

from __future__ import annotations

import hashlib
import json
import math
import re

from screening.checks import grounded_paper_pointer

TOKEN = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?"
NUMBER = re.compile(rf"(?<![\w.]){TOKEN}(?![\w%]|\.\d)")
_SUFFIX = r"(?:percentage\s+points|percent|percentage|milliseconds?|microseconds?|seconds?|sec|minutes?|hours?|ms|us|s|min|h|pp|points)"


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _id(materials, record):
    value = [materials.paper_key, materials.source_pdf, record]
    return "number_" + _hash(json.dumps(value, sort_keys=True, ensure_ascii=False))[:24]


def _sentences(text):
    start = 0
    for boundary in re.finditer(r"[.!?](?=\s|$)|\n", text):
        end = boundary.end()
        if text[start:end].strip():
            leading = len(text[start:end]) - len(text[start:end].lstrip())
            yield start + leading, end
        start = end
    if text[start:].strip():
        yield start + len(text[start:]) - len(text[start:].lstrip()), len(text)


def index_numbers(materials):
    """Index source bytes only; HTML numbers retain their separate cell contract."""
    result = {}
    seen = set()
    for block in materials.blocks:
        if block.id in seen:
            raise ValueError("Numeric catalog requires unique original block IDs")
        seen.add(block.id)
        if block.loc is None or re.search(r"<table\b", block.text, re.I):
            continue
        for start, end in _sentences(block.text):
            for match in NUMBER.finditer(block.text, start, end):
                if not math.isfinite(float(match.group().rstrip("%"))):
                    continue
                suffix = re.match(rf"\s*({_SUFFIX})(?!\w)", block.text[match.end() : end], re.I)
                record = {
                    "block_id": block.id,
                    "block_sha256": _hash(block.text),
                    "loc": block.loc.model_dump(exclude_none=True),
                    "start": match.start(),
                    "end": match.end(),
                    "sentence_start": start,
                    "sentence_end": end,
                    "sentence": block.text[start:end],
                    "token": match.group(),
                    "unit_suffix": "%"
                    if match.group().endswith("%")
                    else (suffix.group(1) if suffix else ""),
                }
                result[_id(materials, record)] = record
    return result


def resolve_number(numbers, identifier, materials):
    record = numbers.get(identifier)
    if record is None or _id(materials, record) != identifier:
        raise ValueError("Unknown or modified original number occurrence")
    block = next((b for b in materials.blocks if b.id == record["block_id"]), None)
    if (
        block is None
        or block.loc is None
        or _hash(block.text) != record["block_sha256"]
        or block.loc.model_dump(exclude_none=True) != record["loc"]
        or block.text[record["start"] : record["end"]] != record["token"]
    ):
        raise ValueError("Original number source changed or is unavailable")
    quote = block.text[record["sentence_start"] : record["sentence_end"]]
    grounded_paper_pointer(materials, block, quote)
    return {**record, "number_id": identifier, "quote": quote}


def sentence_id(record):
    """Share the same original sentence identity across every model-facing index."""
    key = (record["block_id"], record["sentence_start"], record["sentence_end"])
    return "sentence_" + _hash(json.dumps(key))[:16]


def _prefix(dataset, split):
    if not isinstance(dataset, str) or not dataset or not isinstance(split, str) or not split:
        raise ValueError("Prose scope requires an explicit dataset and split")
    return rf"(?:On|For)\s+the\s+{re.escape(split)}\s+split\s+of\s+(?:dataset\s+)?{re.escape(dataset)},\s*"


def _quantity(name, metric):
    return rf"(?P<{name}>{TOKEN})(?:\s+(?P<{name}_unit>{_SUFFIX}))?\s+{re.escape(metric)}"


def _actor(label):
    if not label:
        raise ValueError("Prose operand requires its original subject/comparator identity")
    return rf"(?:(?:method|model)\s+)?{re.escape(label)}\s+(?:has|achieves|records|reports)\s+"


def _match_pair(text, dataset, metric, settings, first_label, second_label, *, transition=False):
    prefix = _prefix(dataset, settings.get("split"))
    if transition:
        body = (
            _actor(settings["method"])
            + _quantity("first", metric)
            + rf"\s+in\s+setting\s+{re.escape(settings['from_setting'])}\s+and\s+"
            + _quantity("second", metric)
            + rf"\s+in\s+setting\s+{re.escape(settings['to_setting'])}"
        )
    else:
        body = (
            _actor(first_label)
            + _quantity("first", metric)
            + r"\s*,?\s+(?:and|while|whereas)\s+"
            + _actor(second_label)
            + _quantity("second", metric)
        )
    return re.fullmatch(prefix + body + r"\s*[.!]?", text, re.I)


def bind_pair(
    materials,
    numbers,
    left_id,
    right_id,
    *,
    dataset,
    metric,
    settings,
    left_label,
    right_label,
    transition=False,
):
    """Bind a finite explicit sentence form; unknown/contrastive scope stays unresolved."""
    left, right = (resolve_number(numbers, identifier, materials) for identifier in (left_id, right_id))
    if left_id == right_id or (left["block_id"], left["sentence_start"], left["sentence_end"]) != (
        right["block_id"],
        right["sentence_start"],
        right["sentence_end"],
    ):
        raise ValueError("Prose pair requires two distinct occurrences in one explicit sentence")
    allowed = {"method", "model", "subject", "comparison", "comparator", "baseline", "split", "unit", "units"}
    if transition:
        allowed |= {"from_setting", "to_setting"}
    if set(settings) - allowed or any(isinstance(value, (dict, list)) for value in settings.values()):
        raise ValueError("Prose sentence does not establish every original condition setting")
    ordered = sorted((left, right), key=lambda row: row["start"])
    labels = (left_label, right_label) if ordered[0] is left else (right_label, left_label)
    match = _match_pair(left["quote"], dataset, metric, settings, *labels, transition=transition)
    if match is None:
        raise ValueError("Prose has no directly scoped subject-predicate numerical pair")
    for name, record in zip(("first", "second"), ordered, strict=True):
        start, end = match.span(name)
        if (record["start"], record["end"]) != (
            record["sentence_start"] + start,
            record["sentence_start"] + end,
        ):
            raise ValueError("Selected occurrence belongs to a different operand role")
    if transition and (ordered[0] is not right or ordered[1] is not left):
        raise ValueError("Transition operands must retain subject=to and comparator=from")
    return {"left": left, "right": right, "kind": "named_transition" if transition else "explicit_pair"}


def direct_pair_candidates(
    materials, numbers, *, dataset, metric, settings, left_label, right_label, transition=False
):
    """Discover exact grammatical pairs; callers must still validate the condition and units."""
    if (
        not dataset
        or not metric
        or not left_label
        or not right_label
        or (left_label == right_label and not transition)
    ):
        return
    sentences = {}
    for identifier, record in numbers.items():
        group = sentences.setdefault(sentence_id(record), {"record": record, "positions": {}})
        group["positions"][record["start"], record["end"]] = identifier
    orders = (
        [(right_label, left_label, True)]
        if transition
        else [(left_label, right_label, False), (right_label, left_label, True)]
    )
    for group in sentences.values():
        record, positions = group["record"], group["positions"]
        for first, second, reverse in orders:
            match = _match_pair(
                record["sentence"], dataset, metric, settings, first, second, transition=transition
            )
            if match is None:
                continue
            identifiers = [
                positions.get(tuple(record["sentence_start"] + offset for offset in match.span(name)))
                for name in ("first", "second")
            ]
            if any(identifier is None for identifier in identifiers):
                continue
            left, right = reversed(identifiers) if reverse else identifiers
            yield bind_pair(
                materials,
                numbers,
                left,
                right,
                dataset=dataset,
                metric=metric,
                settings=settings,
                left_label=left_label,
                right_label=right_label,
                transition=transition,
            )


def _transition_text(text, condition):
    settings, metric = condition.settings, condition.metric
    if not all(
        isinstance(settings.get(key), str) and settings[key]
        for key in ("method", "from_setting", "to_setting")
    ):
        return None
    if settings["from_setting"] == settings["to_setting"] or not metric:
        return None
    sentences = list(_sentences(text))
    matches = []
    for index, (start, end) in enumerate(sentences[:-1]):
        pair = _match_pair(text[start:end], condition.dataset, metric, settings, "", "", transition=True)
        if pair is None:
            continue
        following = text[sentences[index + 1][0] : sentences[index + 1][1]]
        before, after = pair.group("first"), pair.group("second")
        statement = (
            rf"The\s+recorded\s+{re.escape(metric)}\s+(?P<verb>decreases|increases|changes)\s+"
            rf"from\s+{re.escape(before)}\s+in\s+{re.escape(settings['from_setting'])}\s+"
            rf"to\s+{re.escape(after)}\s+in\s+{re.escape(settings['to_setting'])}"
            rf"(?:,\s+an?\s+(?P<direction>decrease|increase)\s+of\s+(?P<gap>{TOKEN})\s+(?P<unit>{_SUFFIX}))?\s*[.!]?"
        )
        assertion = re.fullmatch(statement, following, re.I)
        if assertion:
            before_value, after_value = (float(value.rstrip("%")) for value in (before, after))
            delta = after_value - before_value
            direction = "increase" if delta > 0 else "decrease"
            verb = assertion.group("verb").casefold()
            if not delta or (verb != "changes" and verb != direction + "s"):
                raise ValueError("Original transition direction conflicts with its asserted endpoints")
            if assertion.group("gap"):
                if assertion.group("direction").casefold() != direction:
                    raise ValueError("Original transition gap direction conflicts with its endpoints")
                unit = re.sub(r"\s+", " ", assertion.group("unit").casefold())
                endpoint_units = [
                    "%" if token.endswith("%") else (pair.group(name + "_unit") or "").casefold()
                    for name, token in (("first", before), ("second", after))
                ]
                endpoint_units = [
                    "%" if value in {"percent", "percentage"} else value for value in endpoint_units
                ]
                if endpoint_units[0] != endpoint_units[1]:
                    raise ValueError("Original transition endpoint units are inconsistent")
                if unit in {"percent", "percentage", "%"}:
                    if before_value == 0:
                        raise ValueError("Original relative transition gap has a zero baseline")
                    actual = abs(delta) / abs(before_value) * 100
                elif unit in {"percentage points", "pp", "points"}:
                    if endpoint_units != ["%", "%"]:
                        raise ValueError("Original percentage-point gap requires explicit percent endpoints")
                    actual = abs(delta)
                elif endpoint_units == [unit, unit]:
                    actual = abs(delta)
                else:
                    raise ValueError("Original transition gap has unresolved units")
                expected = float(assertion.group("gap").rstrip("%"))
                if not math.isfinite(expected) or not math.isclose(
                    actual, expected, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(
                        "Original transition gap conflicts with its asserted endpoints and scale"
                    )
            matches.append((after, before))
    return matches[0] if len(matches) == 1 else None


def transition_endpoints(claim, condition, materials):
    """Use an original named method transition, never infer a missing comparator."""
    expected = _transition_text(claim.text, condition)
    if expected is None:
        return None
    explicit = [
        ref
        for ref in claim.source_refs
        if (ref.source_block_id, ref.source_quote) == (claim.source_block_id, claim.source_quote)
    ]
    sources = [
        (ref.source_block_id, ref.source_quote) for ref in claim.source_refs if condition.id in ref.covered
    ]
    if not explicit or any(condition.id in ref.covered for ref in explicit):
        sources.append((claim.source_block_id, claim.source_quote))
    for block_id, quote in sources:
        block = next((b for b in materials.blocks if b.id == block_id), None)
        if block is None or not quote or quote not in block.text:
            continue
        try:
            observed = _transition_text(quote, condition)
        except ValueError:
            continue
        if observed == expected:
            grounded_paper_pointer(materials, block, quote)
            return expected
    return None
