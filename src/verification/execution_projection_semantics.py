"""Finite v2 interpretations of unchanged released-prediction obligations.

This grammar recognizes a small exact-match measurement language. Independent
scope review is still required; an unrecognized clause remains unavailable.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from schemas.claim import Condition, SemanticPredictionProjection, SemanticProjectionScopeDecision
from verification.execution_projection import (
    ProjectionError,
    _pointer,
    _words_count,
    digest,
    field_inventory,
    file_hash,
    indexed_file,
)

VERSION = "released-predictions-v2"
BOUNDARIES = {
    "repeated_run_uncertainty": "repeated run uncertainty",
    "population_performance": "population performance conclusion",
    "cross_model_ranking": "ranking against other models",
}


def _same(a, b):
    return type(a) is type(b) and a == b


def pointer_tokens(path):
    if not isinstance(path, str) or not path.startswith("/") or re.search(r"~(?![01])", path):
        raise ProjectionError("Invalid original field JSON pointer")
    return [s.replace("~1", "/").replace("~0", "~") for s in path[1:].split("/")]


def _unique(values, label):
    if len(values) != len(set(values)):
        raise ProjectionError(f"Duplicate {label} cannot discharge an obligation")
    return set(values)


def _normal(text):
    return re.sub(r"\s+", " ", text.strip().rstrip(".! ")).casefold()


def _definition(text, *, require_fraction=False):
    text = _normal(text)
    prefix = r"(?:accuracy (?:is |computed (?:as |by ))?(?:a )?)?"
    fraction = r"fraction,? (?:computed (?:by|as) )?"
    equality = r"(?:the )?exact equality of (?:each |every )?prediction and (?:its )?label"
    if re.fullmatch(prefix + fraction + equality, text):
        return True
    return not require_fraction and bool(re.fullmatch(r"(?:accuracy computed (?:as|by) )?" + equality, text))


def _negative(text):
    text = _normal(text).replace("-", " ")
    text = re.sub(r" (?:is|are) claimed$", "", text)
    if not text.startswith("no "):
        return None
    parts = re.split(r"\s+(?:or|and)\s+", text[3:])
    inverse = {v: k for k, v in BOUNDARIES.items()}
    if any(part.removeprefix("no ") not in inverse for part in parts):
        return None
    keys = [inverse[part.removeprefix("no ")] for part in parts]
    if len(keys) != len(set(keys)):
        return None
    return {"boundary:" + key for key in keys}


def _sample(text, count, split):
    words = rf"(?:{count}|{re.escape(_words_count(count))})"
    scoped = rf"(?:{re.escape(split)} )?" if split else ""
    match = re.fullmatch(rf"(?:the )?{words} (?:fixed )?{scoped}(?:examples|predictions)", _normal(text))
    if not match:
        return None
    keys = {"sample"}
    if split and re.search(r"\b" + re.escape(split.casefold()) + r"\b", _normal(text)):
        keys.add("dataset")
    return keys


def _metric(text, cfg):
    bare = r"(?:exact[- ]match )?accuracy"
    if re.fullmatch(bare, _normal(text)):
        return {"definition"}
    split = cfg["settings"].get("split")
    if isinstance(split, str) and re.fullmatch(re.escape(split.casefold()) + " " + bare, _normal(text)):
        return {"dataset", "definition"}
    return None


def _fragments(text):
    # HTML bodies never become free prose. Their caption can contain a boundary.
    text = text.split("<table", 1)[0]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n|;\s*", text) if s.strip()]


def _statement(text, runtime, value, count):
    from verification.experiment_targets import _scalar_match

    normalized = text.strip().rstrip(".! ")
    pieces = re.split(r",\s*(?:with|computed\s+as)\s+", normalized, flags=re.I)
    if len(pieces) > 2:
        return None
    keys = set()
    if len(pieces) == 2:
        if not _definition(pieces[1]):
            return None
        keys.add("definition")
    head = pieces[0]
    over = re.search(r"\s+over\s+(.+)$", head, re.I)
    if over:
        sample = _sample(over.group(1), count, runtime.settings.get("split"))
        if sample is None:
            return None
        keys.update(sample)
        head = head[: over.start()]
    scalar = _scalar_match(head, runtime)
    if scalar is None or scalar[1] != value or scalar[2] is not None:
        return None
    keys.update({"dataset", "value"})
    keys.update("runtime:" + key for key in runtime.settings if key != "split")
    return keys


def _description(text, cfg, count=None):
    roles = [str(cfg["settings"][k]) for k in ("model", "method") if k in cfg["settings"]]
    if len(roles) != 1:
        return None
    scope = re.escape(cfg["dataset"])
    if cfg["settings"].get("split"):
        scope += " " + re.escape(str(cfg["settings"]["split"]))
    if re.fullmatch(
        r"Measured " + re.escape(roles[0]) + r" (?:exact[- ]match )?accuracy on " + scope + r"[.]?",
        text,
        re.I,
    ):
        return {"dataset", "definition"} | {
            "runtime:" + k for k in ("model", "method") if k in cfg["settings"]
        }
    if (
        type(count) is int
        and count > 0
        and re.fullmatch(
            r"Reported "
            + scope
            + r" (?:exact[- ]match )?accuracy for "
            + re.escape(roles[0])
            + rf" on (?:{count}|{re.escape(_words_count(count))}) (?:fixed )?examples[.]?",
            text,
            re.I,
        )
    ):
        return {"dataset", "definition", "sample"} | {
            "runtime:" + k for k in ("model", "method") if k in cfg["settings"]
        }
    return None


def _interpret_text(text, cfg, runtime, value, count):
    if not isinstance(text, str) or not text.strip():
        return None
    if _definition(text):
        return {"definition"}
    negative = _negative(text)
    if negative is not None:
        return negative
    sample = _sample(text, count, cfg["settings"].get("split"))
    if sample is not None:
        return sample
    description = _description(text, cfg, count)
    if description is not None:
        return description
    return _statement(text, runtime, value, count)


def _source_authorized(catalog, source, condition_id):
    return any(
        record["kind"] == "claim_source"
        and condition_id in record.get("covered", [])
        and record.get("block_id") == source.get("block_id")
        and record["start"] <= source["start"] < source["end"] <= record["end"]
        for record in catalog["sources"].values()
    )


def _source_supports(atom, quote, runtime, value, count, number):
    kind = atom.kind
    fragments = _fragments(quote)
    if kind == "measurement_definition":
        return any(_definition(f, require_fraction=True) for f in fragments)
    if kind == "conclusion_boundary":
        found = set()
        for fragment in fragments:
            choices = [fragment]
            # The complete negative tail remains anchored in its caption sentence.
            if ", with no " in fragment:
                choices.append("no " + fragment.split(", with no ", 1)[1])
            for choice in choices:
                found.update(_negative(choice) or ())
        return {"boundary:" + k for k in atom.excludes}.issubset(found)
    if kind == "sample_scope":
        split = runtime.settings.get("split")
        words = rf"(?:{count}|{re.escape(_words_count(count))})"
        scope = (re.escape(split) + " ") if split else ""
        correct = r"(?:[0-9]+|" + "|".join(_words_count(n) for n in range(21)) + ")"
        pattern = (
            rf"(?:{correct} of (?:the )?)?{words} fixed {scope}(?:examples|predictions)(?: are correct)?"
        )
        return any(re.fullmatch(pattern, _normal(fragment), re.I) for fragment in fragments)
    if kind == "reported_value":
        return number["quote"] in quote
    if kind in {"dataset_identity", "runtime_setting"}:
        key = "dataset" if kind == "dataset_identity" else "runtime:" + atom.key
        for fragment in fragments:
            keys = _statement(fragment, runtime, value, count)
            if keys is not None and (key in keys or (kind == "runtime_setting" and atom.key == "split")):
                return True
    return False


def _atom_keys(atom, cfg, number_id, count):
    if atom.kind == "dataset_identity":
        if atom.dataset != cfg["dataset"] or not _same(atom.split, cfg["settings"].get("split")):
            raise ProjectionError("Dataset/split atom differs from the actual configuration")
        return {"dataset"}
    if atom.kind == "runtime_setting":
        if atom.key not in cfg["settings"] or not _same(atom.value, cfg["settings"][atom.key]):
            raise ProjectionError("Runtime atom differs from the actual configuration")
        return {"runtime:" + atom.key}
    if atom.kind == "reported_value":
        if atom.number_id != number_id:
            raise ProjectionError("Reported atom selects another original numeric occurrence")
        return {"value"}
    if atom.kind == "sample_scope":
        if atom.count != count or not _same(atom.split, cfg["settings"].get("split")):
            raise ProjectionError("Sample atom differs from the complete released population")
        return {"sample"}
    if atom.kind == "measurement_definition":
        return {"definition"}
    if atom.kind == "conclusion_boundary":
        _unique(atom.excludes, "conclusion boundary")
        return {"boundary:" + k for k in atom.excludes}
    raise ProjectionError("Unknown semantic obligation remains unresolved")


def _check_review(proposal, review):
    if (
        review.proposal_sha256 != digest(proposal.model_dump(mode="json"))
        or review.classification != "absolute_fixed_predictions"
        or not all(
            (
                review.dataset_identity_confirmed,
                review.measurement_definition_confirmed,
                review.all_original_qualifiers_preserved,
            )
        )
        or review.unresolved
    ):
        raise ProjectionError("Independent v2 scope leaves original obligations unresolved")
    sources = {sid for atom in proposal.atoms for sid in atom.source_ids}
    if _unique(review.source_ids, "scope source") != sources:
        raise ProjectionError("Scope must retain the exact proposed source set")
    if _unique([r.atom_id for r in review.atom_reviews], "atom review") != {a.id for a in proposal.atoms}:
        raise ProjectionError("Scope must review each original atom exactly once")
    for atom in proposal.atoms:
        row = next(r for r in review.atom_reviews if r.atom_id == atom.id)
        if row.decision != "confirmed" or _unique(row.source_ids, "atom review source") != set(
            atom.source_ids
        ):
            raise ProjectionError("Independent atom decision changes or leaves its sources unresolved")
    for proposed, reviews, identity in (
        (proposal.field_bindings, review.field_reviews, lambda r: r.path),
        (proposal.claim_bindings, review.claim_reviews, lambda r: (r.start, r.end)),
    ):
        if _unique([identity(r) for r in reviews], "binding review") != {identity(r) for r in proposed}:
            raise ProjectionError("Scope must review every unchanged field/claim binding")
        for row in proposed:
            checked = next(r for r in reviews if identity(r) == identity(row))
            if checked.decision != "confirmed" or checked.atom_ids != row.atom_ids:
                raise ProjectionError("Scope changed the proposed obligation mapping")


def validate_semantic_obligations(
    claim, condition, materials, proposal, review, recipe, catalog, number_id, value
):
    from verification.experiment_catalog import resolve_source
    from verification.prose_numbers import resolve_number

    proposal = SemanticPredictionProjection.model_validate(
        proposal.model_dump() if hasattr(proposal, "model_dump") else proposal
    )
    review = SemanticProjectionScopeDecision.model_validate(
        review.model_dump() if hasattr(review, "model_dump") else review
    )
    if (
        next((c for c in claim.conditions if c.id == condition.id), None) != condition
        or review.condition_id != condition.id
    ):
        raise ProjectionError("Semantic projection must retain the exact original condition")
    _check_review(proposal, review)
    cfg, count = recipe["configuration"], recipe["sample_count"]
    if (
        cfg["metric"] not in {"accuracy", "exact-match accuracy"}
        or not isinstance(cfg["dataset"], str)
        or not cfg["dataset"]
    ):
        raise ProjectionError("Semantic recipe requires a canonical absolute exact-match metric")
    runtime = Condition(
        id=condition.id, dataset=cfg["dataset"], metric=cfg["metric"], settings=cfg["settings"]
    )
    names = [cfg["dataset"]]
    if isinstance(cfg["settings"].get("split"), str):
        names.append(cfg["dataset"] + " " + cfg["settings"]["split"])
    if condition.dataset not in names:
        raise ProjectionError("Original dataset/split differs from the canonical configuration")
    decompositions = set()
    for path in materials.repository.configs:
        if Path(path).suffix.lower() != ".json":
            continue
        try:
            other = json.loads(indexed_file(materials, path).read_text("utf-8"))
        except (ValueError, UnicodeError):
            continue
        if (
            not isinstance(other, dict)
            or set(other) != {"dataset", "metric", "settings"}
            or not isinstance(other["settings"], dict)
            or not isinstance(other["dataset"], str)
        ):
            continue
        other_names = [other["dataset"]]
        if isinstance(other["settings"].get("split"), str):
            other_names.append(other["dataset"] + " " + other["settings"]["split"])
        if (
            other["metric"] in {"accuracy", "exact-match accuracy"}
            and condition.dataset in other_names
            and all(
                _same(other["settings"].get(k), v)
                for k, v in cfg["settings"].items()
                if k in {"model", "method"}
            )
        ):
            decompositions.add(digest({"dataset": other["dataset"], "split": other["settings"].get("split")}))
    if len(decompositions) != 1:
        raise ProjectionError("Dataset/split configuration decomposition is ambiguous or unavailable")
    inventory = field_inventory(condition)
    fields = _unique([r.path for r in proposal.field_bindings], "original field binding")
    if fields != set(inventory) or _unique(review.confirmed_field_paths, "confirmed field") != fields:
        raise ProjectionError("Every original semantic leaf must be retained exactly once")
    _unique([a.id for a in proposal.atoms], "atom identity")
    atom_keys, pointers, seen_keys = {}, {}, set()
    number = resolve_number(catalog["numbers"], number_id, materials)
    for atom in proposal.atoms:
        _unique(atom.source_ids, "atom source")
        keys = _atom_keys(atom, cfg, number_id, count)
        if seen_keys & keys:
            raise ProjectionError("Duplicate semantic atoms cannot hide conflicting obligations")
        seen_keys.update(keys)
        atom_keys[atom.id] = keys
        quotes = []
        for sid in atom.source_ids:
            source = resolve_source(catalog, sid, materials)
            raw = catalog["sources"][sid]
            if not _source_authorized(
                catalog, {**source, "start": raw["start"], "end": raw["end"]}, condition.id
            ):
                raise ProjectionError("Semantic atom source is outside the original condition's authority")
            block = next(b for b in materials.blocks if b.id == source["block_id"])
            pointers[sid] = _pointer(materials, block, source["quote"])
            quotes.append(source["quote"])
        if not any(_source_supports(atom, q, runtime, value, count, number) for q in quotes):
            raise ProjectionError(f"Atom {atom.id} has no complete source-grounded finite interpretation")

    def selected(ids):
        _unique(ids, "binding atom")
        if not set(ids).issubset(atom_keys):
            raise ProjectionError("Binding refers to an unknown atom")
        return set().union(*(atom_keys[i] for i in ids))

    consumed, known_runtime = {}, set()
    for row in proposal.field_bindings:
        actual, tokens = inventory[row.path], pointer_tokens(row.path)
        keys = selected(row.atom_ids)
        expected = None
        if len(tokens) == 2 and tokens[0] == "settings" and tokens[1] in cfg["settings"]:
            key = tokens[1]
            if not _same(actual, cfg["settings"][key]):
                raise ProjectionError("Original runtime field conflicts with the actual configuration")
            expected = {"runtime:" + key}
            known_runtime.add(key)
        elif row.path == "/dataset":
            expected = {"dataset"}
        elif row.path == "/metric":
            expected = _metric(actual, cfg) if isinstance(actual, str) else None
        elif type(actual) in {int, float}:
            if (keys == {"value"} and actual == value) or (
                keys == {"sample"} and type(actual) is int and actual == count
            ):
                expected = keys
        elif isinstance(actual, str):
            expected = _interpret_text(actual, cfg, runtime, value, count)
        if expected is None or keys != expected:
            raise ProjectionError(f"Original field {row.path} has an unresolved or changed finite meaning")
        consumed[row.path] = {"value": actual, "atom_ids": row.atom_ids, "obligations": sorted(keys)}
    if "split" in cfg["settings"] and any("dataset" in row["obligations"] for row in consumed.values()):
        known_runtime.add("split")
    if set(cfg["settings"]) != known_runtime:
        raise ProjectionError("Configuration contains an extra unbound runtime obligation")
    cursor, claim_consumed = 0, []
    for row in proposal.claim_bindings:
        if (
            row.start < cursor
            or row.end > len(claim.text)
            or row.end <= row.start
            or claim.text[cursor : row.start].strip(" \t\r\n,;.!?")
        ):
            raise ProjectionError("Claim bindings overlap or omit original text")
        text = claim.text[row.start : row.end]
        expected = _interpret_text(text, cfg, runtime, value, count)
        if expected is None or selected(row.atom_ids) != expected:
            raise ProjectionError("Original claim has unresolved or changed finite meaning")
        claim_consumed.append({"start": row.start, "end": row.end, "text": text, "atom_ids": row.atom_ids})
        cursor = row.end
    if claim.text[cursor:].strip(" \t\r\n,;.!?"):
        raise ProjectionError("Original claim suffix was omitted")
    used = {i for r in [*proposal.field_bindings, *proposal.claim_bindings] for i in r.atom_ids}
    if used != set(atom_keys) or not {"dataset", "value", "sample", "definition"}.issubset(seen_keys):
        raise ProjectionError("Projection contains unused atoms or lacks a full measurement obligation")
    return {
        "runtime": runtime,
        "sample_count": count,
        "source_pointers": pointers,
        "field_consumption": consumed,
        "claim_consumption": claim_consumed,
    }


def semantic_request_context(claim, materials):
    """Separate version domain; never add keys to the historical v1 context."""
    from verification.execution_projection import repository_context_files
    from verification.execution_projection_catalog import semantic_request_choices

    files = repository_context_files(materials)
    return {
        "version": VERSION,
        "field_inventory": {c.id: field_inventory(c) for c in claim.conditions},
        "claim_text": claim.text,
        "claim_text_length": len(claim.text),
        "repository_files": files,
        "request_choices": semantic_request_choices(claim, materials, repository_files=files),
        "limit": "Closed exact-match scalar, optional sample and definition clauses, fixed sample phrases and explicit negative conclusion boundaries only; unknown residue remains unavailable. Every field and claim fragment needs a finite interpretation and independent scope review. No automatic conversion of null or v1 proposals.",
        "implementation_sha256": file_hash(__file__),
    }


def bind_semantic_target(
    claim,
    condition,
    reported,
    materials,
    *,
    selector,
    proposal,
    entry_script,
    config_path,
    review,
    audit_path,
):
    """Version-specific reconstruction; all historical v1 fields stay in their domain."""
    from schemas.claim import (
        Claim,
        PaperTargetPassage,
        PaperTargetSelector,
        ProjectedExecutionTargetBinding,
        SemanticProjectionRecord,
    )
    from verification.execution_projection import decode_projection_reviews, entry_recipe, projection_context
    from verification.experiment_catalog import build_catalog
    from verification.experiment_targets import _claim_fingerprint, _fingerprint, _scalar_match
    from verification.prose_numbers import resolve_number

    proposal = SemanticPredictionProjection.model_validate(
        proposal.model_dump() if hasattr(proposal, "model_dump") else proposal
    )
    review = SemanticProjectionScopeDecision.model_validate(
        review.model_dump() if hasattr(review, "model_dump") else review
    )
    reported = PaperTargetPassage.model_validate(
        reported.model_dump() if hasattr(reported, "model_dump") else reported
    )
    selector = PaperTargetSelector.model_validate(
        selector.model_dump() if hasattr(selector, "model_dump") else selector
    )
    if not entry_script or not config_path or not selector.number_id:
        raise ProjectionError("Semantic recipe requires an indexed entry/config and prose number occurrence")
    recipe = entry_recipe(materials, entry_script, config_path, proposal.data_path)
    cfg = recipe["configuration"]
    runtime = Condition(
        id=condition.id, dataset=cfg["dataset"], metric=cfg["metric"], settings=cfg["settings"]
    )
    catalog = build_catalog(claim, materials)
    number = resolve_number(catalog["numbers"], selector.number_id, materials)
    block = next(b for b in materials.blocks if b.id == reported.block_id)
    pointer = _pointer(materials, block, reported.quote)
    if (
        number["block_id"] != block.id
        or reported.token != number["token"]
        or number["quote"] not in reported.quote
    ):
        raise ProjectionError("Semantic target selector is outside its exact reported passage")
    scalar = _scalar_match(number["quote"], runtime)
    if scalar is None or scalar[1] != float(reported.token) or scalar[2] is not None:
        raise ProjectionError("Reported occurrence does not establish the canonical absolute quantity")
    leading = len(number["quote"]) - len(number["quote"].lstrip())
    if scalar[0].span("value") != (
        number["start"] - number["sentence_start"] - leading,
        number["end"] - number["sentence_start"] - leading,
    ):
        raise ProjectionError("Selected number is not the canonical result slot")
    context = reported.value_context or reported.quote
    if (
        block.text.count(reported.quote) != 1
        or reported.quote.count(context) != 1
        or number["quote"] not in context
    ):
        raise ProjectionError("Reported context lacks one complete original result sentence")
    value = scalar[1]
    if not 0 <= value <= 1:
        raise ProjectionError("Fraction target lies outside the recipe's unit scale")
    resolved = validate_semantic_obligations(
        claim, condition, materials, proposal, review, recipe, catalog, selector.number_id, value
    )
    audit_path = Path(audit_path) if audit_path else None
    if audit_path is None or not audit_path.is_file():
        raise ProjectionError("Independent v2 plan audit is required")
    audit = json.loads(audit_path.read_text("utf-8"))
    data = audit["input"]
    if _claim_fingerprint(Claim.model_validate(data["claim"])) != _claim_fingerprint(claim) or data.get(
        "paper_blocks"
    ) != [b.model_dump(mode="json") for b in materials.blocks]:
        raise ProjectionError("Semantic plan audit refers to changed original paper/claim")
    expected_context = semantic_request_context(claim, materials)
    expected_context["proposals"] = [
        {
            "plan_index": index,
            "condition_id": target["condition_id"],
            "proposal_sha256": digest(target["projection"]),
        }
        for index, plan in enumerate(data["candidate_plans"])
        for target in plan["targets"]
        if target.get("projection") is not None
    ]
    if (
        data.get("execution_projection_context") != projection_context(claim, materials)
        or data.get("execution_projection_semantics_context") != expected_context
    ):
        raise ProjectionError("Semantic plan resources, implementation or versioned context changed")
    raw_plan = data["candidate_plans"][review.plan_index]
    targets = [row for row in raw_plan["targets"] if row.get("condition_id") == condition.id]
    if (
        len(targets) != 1
        or targets[0].get("projection") != proposal.model_dump(mode="json")
        or PaperTargetPassage.model_validate(targets[0]["reported"]) != reported
        or PaperTargetSelector.model_validate(targets[0]["selector"]) != selector
        or raw_plan.get("entry_script") != entry_script
        or raw_plan.get("config") != config_path
        or raw_plan.get("run_mode") != "evaluation"
        or raw_plan.get("data_paths") != [proposal.data_path]
        or raw_plan.get("weight_paths")
    ):
        raise ProjectionError("Semantic plan differs from the independently reviewed proposal/resources")
    from verification.experiments import PlanCandidate

    decisions, _ = decode_projection_reviews(
        audit.get("response", {}), [PlanCandidate.model_validate(p) for p in data["candidate_plans"]]
    )
    if (
        audit.get("response", {}).get("schema_version") != "catalog-v2"
        or decisions.get((review.plan_index, condition.id)) != review
    ):
        raise ProjectionError("Semantic plan lacks one retained independent raw review")
    sources = resolved["source_pointers"]
    return ProjectedExecutionTargetBinding(
        condition_id=condition.id,
        reported=reported,
        selector=selector,
        pointer=pointer,
        block_sha256=hashlib.sha256(block.text.encode()).hexdigest(),
        artifact_sha256=file_hash(pointer.locator),
        claim_sha256=_claim_fingerprint(claim),
        condition_sha256=_fingerprint(condition.model_dump(mode="json")),
        value=value,
        subject=scalar[3],
        unit="fraction",
        projection=SemanticProjectionRecord(
            proposal=proposal,
            runtime_target=resolved["runtime"],
            sample_count=resolved["sample_count"],
            source_pointers=sources,
            source_hashes={key: file_hash(ptr.locator) for key, ptr in sources.items()},
            paper_hashes={str(p): file_hash(p) for p in (materials.source_pdf, materials.markdown_path)},
            repository_hashes=recipe["repository_hashes"],
            repository_root=str(Path(materials.repository.root).resolve()),
            entry_script=entry_script,
            config_path=config_path,
            label_key=recipe["label_key"],
            prediction_key=recipe["prediction_key"],
            scope_review=review,
            scope_audit=str(audit_path.resolve()),
            scope_audit_sha256=file_hash(audit_path),
            recipe_sha256=file_hash(Path(__file__).with_name("execution_projection.py")),
            field_consumption=resolved["field_consumption"],
            claim_consumption=resolved["claim_consumption"],
            semantics_sha256=file_hash(__file__),
        ),
    )
