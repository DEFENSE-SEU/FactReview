"""Closed structural choices for released predictions; semantic review remains explicit.

No author program runs here. The builder never uses the recipe's measured
numerator to decide whether a paper target is eligible.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from schemas.claim import (
    Claim,
    Condition,
    ExecutionChoice,
    ExecutionChoiceReview,
    SemanticPredictionProjection,
    SemanticProjectionScopeDecision,
)
from verification.execution_projection import (
    ProjectionError,
    _pointer,
    digest,
    entry_recipe,
    field_inventory,
    file_hash,
    projection_snapshot,
    repository_context_files,
)
from verification.execution_projection_semantics import (
    _interpret_text,
    _metric,
    _require_fixed_sample_count,
    _same,
    _source_authorized,
    _source_supports,
    pointer_tokens,
    validate_semantic_obligations,
)

VERSION = "released-predictions-choice-v1"
MAX_CHOICES = 16
MAX_COMBINATION_CHECKS = 128
MAX_RESOURCE_BYTES = 8 * 1024 * 1024
_DERIVED = {
    "questions",
    "evidence",
    "status",
    "notes",
    "advice",
    "theory_derivations",
    "verification_limitations",
}
_WIRE = TypeAdapter(ExecutionChoice)
_PRIVACY_MARKER = "_choice_privacy_rejected"


class ChoicePrivacy:
    """Capture actual call configs; reject sensitive rows without changing model objects."""

    def __init__(self):
        self.configs = []
        self.original_rows = {}
        self.blocked = False

    def safe(self, value):
        from screening.visual_audit import redacted_record

        result = copy.deepcopy(value)
        for cfg in self.configs:
            result = redacted_record(result, cfg)
        return result

    def capture(self, call):
        def invoke(**kwargs):
            from screening import checks

            self.configs.append(copy.deepcopy(kwargs["cfg"]))
            if kwargs.get("module") == "verification.experiments.scope":
                payload = json.loads(kwargs["prompt"])
                outgoing = {
                    key: payload[key]
                    for key in ("execution_choices", "selection_expansions")
                    if key in payload
                }
                if self.safe(outgoing) != outgoing:
                    self.blocked = True
                    raise ProjectionError("Provider-sensitive choice scope blocked before model invocation")
            return (call if call is not None else checks.llm_json)(**kwargs)

        return invoke

    def rows(self, rows, phase):
        self.original_rows[phase] = copy.deepcopy(rows)
        if not isinstance(rows, list):
            return self.safe(rows), []
        output, rejected = [], []
        for index, row in enumerate(rows):
            safe = self.safe(row)
            if safe == row:
                output.append(copy.deepcopy(row))
                continue
            rejected.append(index)
            output.append(
                {
                    "condition_id": self.safe(row.get("condition_id")) if isinstance(row, dict) else None,
                    _PRIVACY_MARKER: {"phase": phase, "index": index, "raw_sha256": digest(row)},
                    "redacted_row": safe,
                }
            )
        return output, rejected

    def audit(self, audit):
        result = copy.deepcopy(audit)
        phases = {}
        for phase, original in list(self.original_rows.items()):
            rows, rejected = self.rows(original, phase)
            if not rejected:
                continue
            phases[phase] = {
                "rejected_indices": rejected,
                "original_rows_sha256": digest(original),
                "safe_rows_sha256": digest(rows),
                "row_sha256": {str(i): digest(original[i]) for i in rejected},
            }
            if phase == "selection":
                if "first_pass_response" in result:
                    first = result["first_pass_response"]
                    if "execution_wire" in result:
                        if first.get("execution", {}).get("kind") == "choice":
                            first["execution"]["selection"] = rows[0]
                        result["first_pass_normalized"]["execution_choices"] = copy.deepcopy(rows)
                    else:
                        first["execution_choices"] = rows
                if "execution_choices" in result.get("input", {}):
                    result["input"]["execution_choices"] = copy.deepcopy(rows)
            elif "response" in result:
                result["response"]["execution_choice_reviews"] = rows
        if phases:
            result["choice_privacy"] = {
                "version": "choice-privacy-v1",
                "wire_representation": "redacted_copy",
                "phases": phases,
            }
        if self.blocked:
            result["choice_scope_delivery"] = "blocked_before_model_invocation"
        result = self.safe(result)
        if result != audit:
            result["choice_audit_representation"] = {
                "wire_representation": "redacted_copy",
                "original_record_sha256": digest(audit),
            }
        if "execution_wire" in result:
            wire = result["execution_wire"]
            wire["saved_response_sha256"] = digest(result["first_pass_response"])
            wire["saved_normalized_sha256"] = digest(result["first_pass_normalized"])
            wire["wire_representation"] = (
                "original"
                if wire["raw_response_sha256"] == wire["saved_response_sha256"]
                and wire.get("normalized_response_sha256", wire["saved_normalized_sha256"])
                == wire["saved_normalized_sha256"]
                else "redacted_copy"
            )
        return result


def privacy_indices(rows, metadata, phase):
    """Validate persisted rejection tombstones before decoding safe audit copies."""
    marked = [i for i, row in enumerate(rows or []) if isinstance(row, dict) and _PRIVACY_MARKER in row]
    if metadata is None:
        if marked:
            raise ProjectionError("Choice privacy metadata is missing")
        return []
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"version", "wire_representation", "phases"}
        or metadata["version"] != "choice-privacy-v1"
        or metadata["wire_representation"] != "redacted_copy"
        or not isinstance(metadata["phases"], dict)
        or not metadata["phases"]
        or set(metadata["phases"]) - {"selection", "review"}
    ):
        raise ProjectionError("Invalid choice privacy metadata")
    entry = metadata["phases"].get(phase)
    if entry is None:
        if marked:
            raise ProjectionError("Choice privacy phase is missing")
        return []
    if not isinstance(entry, dict) or set(entry) != {
        "rejected_indices",
        "original_rows_sha256",
        "safe_rows_sha256",
        "row_sha256",
    }:
        raise ProjectionError("Invalid choice privacy phase")
    indices = entry["rejected_indices"]
    if (
        not isinstance(rows, list)
        or not isinstance(indices, list)
        or not indices
        or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
        or indices != sorted(set(indices))
        or indices != marked
        or entry["safe_rows_sha256"] != digest(rows)
        or not isinstance(entry["row_sha256"], dict)
        or set(entry["row_sha256"]) != {str(i) for i in indices}
    ):
        raise ProjectionError("Choice privacy indices or safe rows changed")
    hashes = [entry["original_rows_sha256"], *entry["row_sha256"].values()]
    if any(not isinstance(h, str) or len(h) != 64 or set(h) - set("0123456789abcdef") for h in hashes):
        raise ProjectionError("Invalid choice privacy original hash")
    for index in indices:
        row = rows[index]
        marker = row[_PRIVACY_MARKER]
        if (
            set(row) != {"condition_id", _PRIVACY_MARKER, "redacted_row"}
            or not isinstance(marker, dict)
            or type(marker.get("index")) is not int
            or marker
            != {
                "phase": phase,
                "index": index,
                "raw_sha256": entry["row_sha256"][str(index)],
            }
        ):
            raise ProjectionError("Choice privacy row marker changed")
    return indices


class _ChoiceBudget(ProjectionError):
    pass


def _identity(claim):
    return digest(claim.model_dump(mode="json", exclude=_DERIVED))


def _code_hashes():
    here = Path(__file__).parent
    return {
        str(p): file_hash(p)
        for p in (
            Path(__file__),
            here / "execution_projection.py",
            here / "execution_projection_semantics.py",
            here / "execution_projection_catalog.py",
            here / "experiment_catalog.py",
            here / "experiment_targets.py",
            here / "experiments.py",
            here / "prose_numbers.py",
        )
    }


def _field_keys(path, actual, cfg, runtime, value, count):
    tokens = pointer_tokens(path)
    if len(tokens) == 2 and tokens[0] == "settings" and tokens[1] in cfg["settings"]:
        if not _same(actual, cfg["settings"][tokens[1]]):
            raise ProjectionError(f"Runtime field {path} conflicts with the actual configuration")
        return {"runtime:" + tokens[1]}
    if path == "/dataset":
        names = [cfg["dataset"]]
        if isinstance(cfg["settings"].get("split"), str):
            names.append(cfg["dataset"] + " " + cfg["settings"]["split"])
        return {"dataset"} if actual in names else None
    if path == "/metric":
        return _metric(actual, cfg) if isinstance(actual, str) else None
    if type(actual) in {int, float}:
        if path in {"/settings/accuracy", "/settings/reported_value"} and actual == value:
            return {"value"}
        if (
            path in {"/settings/examples", "/settings/sample_count"}
            and type(actual) is int
            and actual == count
        ):
            return {"sample"}
        return None
    return _interpret_text(actual, cfg, runtime, value, count) if isinstance(actual, str) else None


def _canonical_sources(catalog, materials, condition):
    from verification.experiment_catalog import resolve_source

    groups = {}
    for sid, raw in catalog["sources"].items():
        if not raw.get("block_id") or not _source_authorized(catalog, raw, condition.id):
            continue
        source = resolve_source(catalog, sid, materials)
        key = (raw["block_id"], raw["start"], raw["end"], source["quote"])
        preference = (0 if raw["kind"] == "claim_source" else 1, sid)
        if key not in groups or preference < groups[key][0]:
            groups[key] = (preference, sid, source)
    return {sid: source for _, sid, source in sorted(groups.values())}


def _atom(key, cfg, number_id, count):
    if key == "dataset":
        return dict(kind="dataset_identity", dataset=cfg["dataset"], split=cfg["settings"].get("split"))
    if key.startswith("runtime:"):
        name = key[len("runtime:") :]
        return dict(kind="runtime_setting", key=name, value=cfg["settings"][name])
    if key == "value":
        return dict(kind="reported_value", number_id=number_id)
    if key == "sample":
        return dict(
            kind="sample_scope",
            count=count,
            population="fixed_released_predictions",
            split=cfg["settings"].get("split"),
            all_records=True,
        )
    if key == "definition":
        return dict(
            kind="measurement_definition",
            measure="exact_match_accuracy",
            predicate="prediction_equals_label",
            aggregation="fraction_of_all_records",
            unit="fraction",
        )
    if key.startswith("boundary:"):
        return dict(kind="conclusion_boundary", excludes=[key[len("boundary:") :]])
    raise ProjectionError("Unknown structural obligation")


def _candidate(claim, condition, materials, catalog, sources, recipe, entry, config, data, number_id):
    from schemas.claim import SemanticProjectionAtom
    from verification.experiment_targets import _scalar_match
    from verification.prose_numbers import resolve_number

    cfg, count = recipe["configuration"], recipe["sample_count"]
    if cfg["metric"] not in {"accuracy", "exact-match accuracy"}:
        raise ProjectionError("Recipe metric is outside the canonical exact-match contract")
    runtime = Condition(
        id=condition.id, dataset=cfg["dataset"], metric=cfg["metric"], settings=cfg["settings"]
    )
    number = resolve_number(catalog["numbers"], number_id, materials)
    scalar = _scalar_match(number["quote"], runtime)
    leading = len(number["quote"]) - len(number["quote"].lstrip())
    if (
        scalar is None
        or scalar[2] is not None
        or scalar[0].span("value")
        != (
            number["start"] - number["sentence_start"] - leading,
            number["end"] - number["sentence_start"] - leading,
        )
    ):
        raise ProjectionError("Number is not an original absolute result slot")
    value = scalar[1]
    if not 0 <= value <= 1:
        raise ProjectionError("Target has a non-fraction scale")
    fields = {}
    for path, actual in field_inventory(condition).items():
        keys = _field_keys(path, actual, cfg, runtime, value, count)
        if not keys:
            raise ProjectionError(f"Original field {path} has no finite choice meaning")
        fields[path] = keys
    whole = _interpret_text(claim.text, cfg, runtime, value, count)
    if not whole:
        raise ProjectionError("Whole claim is outside the finite choice grammar")
    _require_fixed_sample_count(claim, condition, cfg, runtime, value, count, claim_texts=[claim.text])
    keys = set().union(whole, *fields.values())
    if not {"dataset", "value", "sample", "definition"}.issubset(keys):
        raise ProjectionError("Choice lacks a complete fixed exact-match measurement")
    runtime_keys = {k[len("runtime:") :] for k in keys if k.startswith("runtime:")}
    if "dataset" in keys and "split" in cfg["settings"]:
        runtime_keys.add("split")
    if runtime_keys != set(cfg["settings"]):
        raise ProjectionError("Unconsumed runtime configuration")
    atoms, ids = [], {}
    adapter = TypeAdapter(SemanticProjectionAtom)
    for key in sorted(keys):
        raw = _atom(key, cfg, number_id, count)
        temporary = adapter.validate_python(dict(raw, id="pending", source_ids=["pending"]))
        sids = [
            sid
            for sid, source in sources.items()
            if _source_supports(temporary, source["quote"], runtime, value, count, number)
        ]
        if not sids:
            raise ProjectionError(f"Obligation {key} lacks one complete authorized source")
        identifier = "atom_" + digest({"meaning": raw, "sources": sids})
        ids[key] = identifier
        atoms.append(dict(raw, id=identifier, source_ids=sids))
    proposal = SemanticPredictionProjection(
        version="released-predictions-v2",
        recipe="exact_match_accuracy",
        data_path=data,
        atoms=atoms,
        field_bindings=[dict(path=p, atom_ids=[ids[k] for k in sorted(v)]) for p, v in fields.items()],
        claim_bindings=[dict(start=0, end=len(claim.text), atom_ids=[ids[k] for k in sorted(whole)])],
    ).model_dump(mode="json")
    block = next(b for b in materials.blocks if b.id == number["block_id"])
    if block.text.count(number["quote"]) != 1:
        raise ProjectionError("Result sentence is not unique in its block")
    pointer = _pointer(materials, block, number["quote"])
    used = {sid for atom in atoms for sid in atom["source_ids"]}
    return dict(
        condition_id=condition.id,
        condition=condition.model_dump(mode="json"),
        entry_script=entry,
        config_path=config,
        data_path=data,
        proposal=proposal,
        selector={"number_id": number_id, "cell_id": None},
        reported=dict(block_id=block.id, quote=number["quote"], token=number["token"], value_context=""),
        value=value,
        subject=scalar[3],
        pointer=pointer.model_dump(mode="json"),
        runtime_target=runtime.model_dump(mode="json"),
        sample_count=count,
        repository_hashes=recipe["repository_hashes"],
        sources={sid: sources[sid] for sid in sorted(used)},
    )


def build_choice_registry(claim, materials):
    """Finite candidates only; empty/unavailable never becomes an execution plan."""
    from verification.execution_projection_catalog import semantic_request_choices
    from verification.experiment_catalog import build_catalog

    snapshot = dict(
        claim=claim.model_dump(mode="json"),
        stable_claim_sha256=_identity(claim),
        materials=materials.model_dump(mode="json"),
        original_sha256=projection_snapshot(claim, materials),
        code_hashes=_code_hashes(),
    )
    registry = dict(
        version=VERSION,
        status="structural_choices_only",
        snapshot=snapshot,
        candidates={},
        unavailable=[],
        budget={
            "max_choices_per_condition": MAX_CHOICES,
            "max_combination_checks": MAX_COMBINATION_CHECKS,
            "checks_used": 0,
            "max_resource_bytes": MAX_RESOURCE_BYTES,
            "resource_bytes": 0,
        },
    )
    menu = semantic_request_choices(claim, materials, repository_files=repository_context_files(materials))
    catalog = build_catalog(claim, materials)
    for condition in claim.conditions:
        found, errors = [], set()
        sources = _canonical_sources(catalog, materials, condition)
        configs = menu["conditions"].get(condition.id, {}).get("by_config", {})
        decompositions = {
            digest(
                {
                    "dataset": r["configuration"]["dataset"],
                    "split": r["configuration"]["settings"].get("split"),
                }
            )
            for r in configs.values()
            if r["finite_metric_scope_compatible"]
        }
        if len(decompositions) != 1:
            if not configs:
                reason = "No usable configuration in the finite released-prediction context"
            elif not any(
                isinstance(condition.metric, str)
                and row["configuration"]["metric"] in {"accuracy", "exact-match accuracy"}
                and _metric(condition.metric, row["configuration"]) is not None
                for row in configs.values()
            ):
                reason = "No configuration matches the claimed metric under the finite exact-match accuracy recipe"
            elif not decompositions:
                reason = "No metric-compatible configuration matches the claimed dataset/split"
            else:
                reason = "Multiple dataset/split decompositions match the claimed scope"
            registry["unavailable"].append(
                dict(
                    condition_id=condition.id,
                    reason=reason,
                )
            )
            continue

        def consume_check():
            if registry["budget"]["checks_used"] >= MAX_COMBINATION_CHECKS:
                raise _ChoiceBudget(
                    "Registry combination-check budget exhausted; current condition unavailable"
                )
            registry["budget"]["checks_used"] += 1

        try:
            for config, row in configs.items():
                if not row["finite_metric_scope_compatible"]:
                    continue
                for entry in menu["entries"]:
                    for data in menu["data_candidates"]:
                        consume_check()
                        try:
                            root = Path(materials.repository.root).resolve()
                            paths = [(root / name).resolve() for name in (entry, config, data)]
                            if any(not path.is_relative_to(root) for path in paths):
                                raise ProjectionError("Choice resource is outside the repository")
                            size = sum(path.stat().st_size for path in paths)
                            if registry["budget"]["resource_bytes"] + size > MAX_RESOURCE_BYTES:
                                raise _ChoiceBudget(
                                    "Registry resource-size budget exhausted; current condition unavailable"
                                )
                            registry["budget"]["resource_bytes"] += size
                            recipe = entry_recipe(materials, entry, config, data)
                        except _ChoiceBudget:
                            raise
                        except (ValueError, OSError, KeyError, TypeError, SyntaxError) as exc:
                            errors.add(str(exc))
                            continue
                        for occurrence in row["prose_scalar_candidates"]:
                            consume_check()
                            try:
                                candidate = _candidate(
                                    claim,
                                    condition,
                                    materials,
                                    catalog,
                                    sources,
                                    recipe,
                                    entry,
                                    config,
                                    data,
                                    occurrence["number_id"],
                                )
                            except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                                errors.add(str(exc))
                                continue
                            found.append(candidate)
                            if len(found) > MAX_CHOICES:
                                raise _ChoiceBudget(
                                    f"Candidate bound exceeded: at least {len(found)} > {MAX_CHOICES}; none selected"
                                )
        except _ChoiceBudget as exc:
            registry["unavailable"].append(dict(condition_id=condition.id, reason=str(exc)))
            continue
        for candidate in found:
            cid = "choice_" + digest({"snapshot": snapshot, "candidate": candidate})
            candidate["candidate_id"] = cid
            obligations = []
            for kind, rows in (
                ("atom", candidate["proposal"]["atoms"]),
                ("field", candidate["proposal"]["field_bindings"]),
                ("claim", candidate["proposal"]["claim_bindings"]),
            ):
                for binding in rows:
                    o = dict(kind=kind, binding=binding)
                    if kind == "field":
                        o["original_value"] = field_inventory(condition)[binding["path"]]
                    if kind == "claim":
                        o["original_text"] = claim.text[binding["start"] : binding["end"]]
                    o["obligation_id"] = "ob_" + digest({"candidate_id": cid, "obligation": o})
                    obligations.append(o)
            candidate["obligations"] = obligations
            registry["candidates"][cid] = candidate
        if not found:
            registry["unavailable"].append(
                dict(
                    condition_id=condition.id,
                    reason="; ".join(sorted(errors))
                    or "No complete original result and indexed recipe combination",
                )
            )
    registry["registry_sha256"] = digest(registry)
    return registry


def choice_context(registry):
    return copy.deepcopy(
        {
            k: registry[k]
            for k in ("version", "status", "registry_sha256", "candidates", "unavailable", "budget")
        }
    )


def revalidate_registry(registry, claim, materials, *, downstream=False):
    body = {k: v for k, v in registry.items() if k != "registry_sha256"}
    if digest(body) != registry.get("registry_sha256"):
        raise ProjectionError("Choice registry content changed")
    snapshot = registry["snapshot"]
    if _identity(claim) != snapshot["stable_claim_sha256"]:
        raise ProjectionError("Original choice claim changed")
    original = Claim.model_validate(snapshot["claim"]) if downstream else claim
    try:
        if (
            projection_snapshot(original, materials) != snapshot["original_sha256"]
            or _code_hashes() != snapshot["code_hashes"]
        ):
            raise ProjectionError("Choice paper, resources or implementation changed")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ProjectionError(f"Choice source snapshot unavailable: {exc}") from exc
    if build_choice_registry(original, materials) != registry:
        raise ProjectionError("Choice registry cannot be reproduced from the original inputs")


def decode_choices(rows, registry, plans, *, privacy_rejected_indices=()):
    known = {c["id"] for c in registry["snapshot"]["claim"]["conditions"]}
    accepted, invalid, seen, errors = {}, set(), set(), []
    if not isinstance(rows, list):
        return {}, ["Execution choices must be a list"]
    collisions = {t.condition_id for p in plans for t in p.targets}
    for index, raw in enumerate(rows):
        raw_id = raw.get("condition_id") if isinstance(raw, dict) else None
        key = raw_id.strip() if isinstance(raw_id, str) else None
        try:
            if key not in known:
                invalid.update(known)
                raise ProjectionError("Choice has no known condition identity")
            if key in seen:
                raise ProjectionError("Duplicate choice cannot restore a condition")
            seen.add(key)
            if index in privacy_rejected_indices:
                raise ProjectionError("Provider-sensitive choice narrative rejected")
            row = _WIRE.validate_python(raw)
            if key in collisions:
                raise ProjectionError("Legacy plan and choice collide for one condition")
            if row.decision == "unresolved":
                raise ProjectionError("Choice explicitly unresolved: " + row.rationale)
            candidate = registry["candidates"].get(row.candidate_id)
            if candidate is None or candidate["condition_id"] != key:
                raise ProjectionError("Unknown or cross-condition choice")
            accepted[key] = {"index": index, "selection": row.model_dump(mode="json")}
        except (ValueError, TypeError) as exc:
            if key in known:
                invalid.add(key)
            errors.append(str(exc))
    if len(accepted) > 1:
        invalid.update(accepted)
        errors.append("One original claim permits at most one selected execution choice")
    return {k: v for k, v in accepted.items() if k not in invalid}, errors


def _valid_review(raw, selection, registry):
    row = ExecutionChoiceReview.model_validate(raw)
    selected = selection["selection"]
    if row.condition_id != selected["condition_id"] or row.candidate_id != selected["candidate_id"]:
        raise ProjectionError("Choice review changed the explicit selection")
    if row.decision != "confirmed" or row.classification != "absolute_fixed_predictions":
        raise ProjectionError("Choice semantics remain unresolved")
    expected = {o["obligation_id"] for o in registry["candidates"][row.candidate_id]["obligations"]}
    ids = [r.obligation_id for r in row.reviews]
    if (
        len(ids) != len(set(ids))
        or set(ids) != expected
        or any(r.decision != "confirmed" for r in row.reviews)
    ):
        raise ProjectionError("Every unchanged choice obligation needs one explicit confirmed review")
    return row


def decode_choice_reviews(rows, selections, registry, *, privacy_rejected_indices=()):
    accepted, invalid, seen, errors = {}, set(), set(), []
    if not isinstance(rows, list):
        return {}, ["Choice reviews must be a list"]
    for index, raw in enumerate(rows):
        raw_id = raw.get("condition_id") if isinstance(raw, dict) else None
        key = raw_id.strip() if isinstance(raw_id, str) else None
        try:
            if key not in selections:
                invalid.update(selections)
                raise ProjectionError("Review has no known selected condition")
            if key in seen:
                raise ProjectionError("Duplicate review cannot restore a choice")
            seen.add(key)
            if index in privacy_rejected_indices:
                raise ProjectionError("Provider-sensitive choice review narrative rejected")
            row = _valid_review(raw, selections[key], registry)
            accepted[key] = {"index": index, "review": row.model_dump(mode="json")}
        except (ValueError, TypeError) as exc:
            if key in selections:
                invalid.add(key)
            errors.append(str(exc))
    for key in selections.keys() - accepted.keys():
        errors.append(f"Missing healthy choice review: {key}")
    return {k: v for k, v in accepted.items() if k not in invalid}, errors


def _normalized_review(candidate, actual):
    """One-to-one expansion of actual confirmed new-wire obligations, never a model raw response."""
    by_id = {r.obligation_id: r for r in actual.reviews}
    atom_reviews, field_reviews, claim_reviews = [], [], []
    for o in candidate["obligations"]:
        decision = by_id[o["obligation_id"]]
        b = o["binding"]
        base = dict(decision=decision.decision, rationale=decision.rationale)
        if o["kind"] == "atom":
            atom_reviews.append(dict(base, atom_id=b["id"], source_ids=b["source_ids"]))
        elif o["kind"] == "field":
            field_reviews.append(dict(base, **b))
        else:
            claim_reviews.append(dict(base, **b))
    proposal = candidate["proposal"]
    return SemanticProjectionScopeDecision(
        version="released-predictions-v2",
        plan_index=0,
        condition_id=candidate["condition_id"],
        classification=actual.classification,
        dataset_identity_confirmed=True,
        measurement_definition_confirmed=True,
        all_original_qualifiers_preserved=True,
        confirmed_field_paths=[r["path"] for r in proposal["field_bindings"]],
        source_ids=sorted({s for a in proposal["atoms"] for s in a["source_ids"]}),
        unresolved=[],
        rationale=actual.rationale,
        proposal_sha256=digest(proposal),
        atom_reviews=atom_reviews,
        field_reviews=field_reviews,
        claim_reviews=claim_reviews,
    )


def retained_choice_first_response(audit):
    """Rebuild only the new choice wire; historical audit interpretation is unchanged."""
    from verification.experiments import lower_execution_response

    raw = audit.get("first_pass_response", {})
    if not any(key in raw for key in ("schema_version", "execution")) and not any(
        key in audit for key in ("execution_wire", "first_pass_normalized")
    ):
        return raw
    normalized, expected = lower_execution_response(raw)
    wire = audit.get("execution_wire")
    if (
        expected["kind"] != "choice"
        or not isinstance(wire, dict)
        or set(wire)
        != set(expected) | {"saved_response_sha256", "saved_normalized_sha256", "wire_representation"}
        or normalized != audit.get("first_pass_normalized")
        or wire["saved_response_sha256"] != expected["raw_response_sha256"]
        or wire["saved_normalized_sha256"] != expected["normalized_response_sha256"]
        or any(
            wire[key] != value
            for key, value in expected.items()
            if key not in {"raw_response_sha256", "normalized_response_sha256"}
        )
    ):
        raise ProjectionError("Choice raw execution origin or audited lowering changed")
    # Original hashes describe the pre-redaction response, never reconstructed text.
    # The accepted origin itself remains exact; polluted origins fail the strict union.
    if wire["wire_representation"] == "original":
        if any(wire[key] != expected[key] for key in ("raw_response_sha256", "normalized_response_sha256")):
            raise ProjectionError("Choice original response fingerprint changed")
    elif wire["wire_representation"] == "redacted_copy":
        marker = audit.get("choice_audit_representation")
        hashes = [wire["raw_response_sha256"], wire["normalized_response_sha256"]]
        if (
            not isinstance(marker, dict)
            or set(marker) != {"wire_representation", "original_record_sha256"}
            or marker["wire_representation"] != "redacted_copy"
        ):
            raise ProjectionError("Choice safe-copy provenance is missing")
        hashes.append(marker["original_record_sha256"])
        if any(not isinstance(h, str) or len(h) != 64 or set(h) - set("0123456789abcdef") for h in hashes):
            raise ProjectionError("Choice original safe-copy fingerprint is invalid")
    else:
        raise ProjectionError("Unknown choice wire representation")
    return normalized


def bind_choice_target(claim, materials, registry, selection, review, audit_path, *, downstream=False):
    from schemas.claim import (
        ChoiceProjectionRecord,
        PaperTargetPassage,
        PaperTargetSelector,
        ProjectedExecutionTargetBinding,
    )
    from verification.experiment_catalog import build_catalog
    from verification.experiment_targets import _claim_fingerprint, _fingerprint

    revalidate_registry(registry, claim, materials, downstream=downstream)
    original = Claim.model_validate(registry["snapshot"]["claim"])
    candidate = registry["candidates"][selection["selection"]["candidate_id"]]
    condition = next(c for c in original.conditions if c.id == candidate["condition_id"])
    actual = _valid_review(review["review"], selection, registry)
    normalized = _normalized_review(candidate, actual)
    recipe = entry_recipe(
        materials, candidate["entry_script"], candidate["config_path"], candidate["data_path"]
    )
    proposal = SemanticPredictionProjection.model_validate(candidate["proposal"])
    resolved = validate_semantic_obligations(
        original,
        condition,
        materials,
        proposal,
        normalized,
        recipe,
        build_catalog(original, materials),
        candidate["selector"]["number_id"],
        candidate["value"],
    )
    path = Path(audit_path) if audit_path else None
    if path is None or not path.is_file():
        raise ProjectionError("Independent choice audit is required")
    audit = json.loads(path.read_text("utf-8"))
    if audit.get("choice_registry") != registry or audit["input"].get(
        "execution_choice_context"
    ) != choice_context(registry):
        raise ProjectionError("Choice audit registry differs from the original request")
    if audit["input"].get("claim") != registry["snapshot"]["claim"] or audit["input"].get("paper_blocks") != [
        b.model_dump(mode="json") for b in materials.blocks
    ]:
        raise ProjectionError("Choice audit changed original claim or paper context")
    retained_first = retained_choice_first_response(audit)
    if retained_first.get("execution_choices", []) != audit["input"].get("execution_choices"):
        raise ProjectionError("Choice scope selection differs from the preserved first response")
    from verification.experiments import PlanCandidate

    accepted, _ = decode_choices(
        audit["input"].get("execution_choices"),
        registry,
        [PlanCandidate.model_validate(p) for p in audit["input"]["candidate_plans"]],
        privacy_rejected_indices=privacy_indices(
            audit["input"].get("execution_choices"), audit.get("choice_privacy"), "selection"
        ),
    )
    decisions, _ = decode_choice_reviews(
        audit.get("response", {}).get("execution_choice_reviews", []),
        accepted,
        registry,
        privacy_rejected_indices=privacy_indices(
            audit.get("response", {}).get("execution_choice_reviews", []),
            audit.get("choice_privacy"),
            "review",
        ),
    )
    if (
        audit.get("response", {}).get("schema_version") != "catalog-v2"
        or accepted.get(condition.id) != selection
        or decisions.get(condition.id) != review
    ):
        raise ProjectionError("Choice lacks unchanged retained actual selection and independent review")
    wire = audit.get("execution_wire")
    saved_response_hash = digest(audit["first_pass_response"])
    origin = {
        "wire_version": wire["version"] if wire is not None else "legacy",
        "saved_response_sha256": saved_response_hash,
        "selected_origin_sha256": digest(retained_first["execution_choices"][selection["index"]]),
        "original_response_sha256": (
            wire["raw_response_sha256"]
            if wire is not None
            else (None if audit.get("choice_audit_representation") else saved_response_hash)
        ),
        "adapter_sha256": wire["adapter_sha256"] if wire is not None else None,
    }
    block = next(b for b in materials.blocks if b.id == candidate["reported"]["block_id"])
    pointer = _pointer(materials, block, candidate["reported"]["quote"])
    sources = resolved["source_pointers"]
    binding = ProjectedExecutionTargetBinding(
        condition_id=condition.id,
        reported=PaperTargetPassage.model_validate(candidate["reported"]),
        selector=PaperTargetSelector.model_validate(candidate["selector"]),
        pointer=pointer,
        block_sha256=hashlib.sha256(block.text.encode()).hexdigest(),
        artifact_sha256=file_hash(pointer.locator),
        claim_sha256=_claim_fingerprint(original),
        condition_sha256=_fingerprint(condition.model_dump(mode="json")),
        value=candidate["value"],
        subject=candidate["subject"],
        unit="fraction",
        projection=ChoiceProjectionRecord(
            record_version=VERSION,
            proposal=proposal,
            runtime_target=resolved["runtime"],
            sample_count=resolved["sample_count"],
            source_pointers=sources,
            source_hashes={sid: file_hash(ptr.locator) for sid, ptr in sources.items()},
            paper_hashes={str(p): file_hash(p) for p in (materials.source_pdf, materials.markdown_path)},
            repository_hashes=recipe["repository_hashes"],
            repository_root=str(Path(materials.repository.root).resolve()),
            entry_script=candidate["entry_script"],
            config_path=candidate["config_path"],
            label_key=recipe["label_key"],
            prediction_key=recipe["prediction_key"],
            scope_review=normalized,
            scope_audit=str(path.resolve()),
            scope_audit_sha256=file_hash(path),
            recipe_sha256=file_hash(Path(__file__).with_name("execution_projection.py")),
            semantics_sha256=file_hash(Path(__file__).with_name("execution_projection_semantics.py")),
            field_consumption=resolved["field_consumption"],
            claim_consumption=resolved["claim_consumption"],
            selection=selection["selection"],
            choice_review=actual,
            selection_index=selection["index"],
            review_index=review["index"],
            candidate_id=candidate["candidate_id"],
            registry_sha256=registry["registry_sha256"],
            registry_snapshot=copy.deepcopy(registry),
            choice_builder_sha256=file_hash(__file__),
            execution_origin=origin,
        ),
    )
    revalidate_registry(registry, claim, materials, downstream=downstream)
    return binding


def choice_plan(claim, materials, registry, selection, review, audit_path):
    from schemas.claim import ExecutionPlan, ExecutionTask

    binding = bind_choice_target(claim, materials, registry, selection, review, audit_path)
    candidate = registry["candidates"][binding.projection.candidate_id]
    return ExecutionPlan(
        id=f"{claim.id}.plan",
        claim_id=claim.id,
        condition_ids=[binding.condition_id],
        target_conditions=[
            next(c for c in claim.conditions if c.id == binding.condition_id).model_copy(deep=True)
        ],
        y_paper={binding.condition_id: binding.value},
        target_bindings={binding.condition_id: binding},
        task=ExecutionTask(
            entry_script=candidate["entry_script"],
            config=candidate["config_path"],
            command=["python", "-I", "-S", candidate["entry_script"]],
        ),
        run_mode="evaluation",
        feasibility="ready",
        priority="high" if claim.importance == "core" else "medium",
        estimated_cost="unknown",
    )
