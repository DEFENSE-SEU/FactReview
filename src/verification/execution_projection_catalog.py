"""Finite request choices; these hints never authorize a plan or execute a resource."""

from __future__ import annotations

import json
from pathlib import PurePosixPath, PureWindowsPath

from schemas.claim import Condition
from verification.experiment_catalog import build_catalog, resolve_source

VERSION = "released-prediction-choices-v1"


def request_choices(claim, materials, *, repository_files):
    from verification.execution_projection import (
        expected_field_role,
        field_inventory,
        qualifier_category,
        source_hits,
        source_requirements,
    )
    from verification.experiment_targets import TargetBindingError, _scalar_match

    repo = materials.repository
    result = {
        "schema_version": VERSION,
        "recipe": "exact_match_accuracy",
        "status": "structural_choices_only",
        "scope": "Released full-list categorical prediction evaluation; no inference/training or sufficient-support decision.",
        "resource_contract": {
            "path_format": "Choose exact repository-relative indexed identifiers; never prepend repository.root.",
            "data_paths": "Exactly [projection.data_path]; never include config or entry_script here.",
            "config": "One config identifier in the separate config field.",
            "entry_script": "One entry identifier in the separate entry_script field.",
            "weight_paths": [],
            "run_mode": "evaluation",
        },
        "target_contract": {
            "selector": "number_id only; cell_id is unsupported by this recipe.",
            "reported": "Select an original prose occurrence explicitly. Preserve its original source, full sentence and numeric token; do not replace an existing returned target.",
            "value_context": "For this projection recipe, reported.value_context must contain the complete prose_scalar_candidates sentence verbatim, including its leading words and punctuation, inside reported.quote. An empty value_context uses the complete reported.quote. Narrowed substrings accepted by other target types cannot satisfy this recipe's complete-sentence check.",
        },
        "entries": [],
        "configs": [],
        "data_candidates": [],
        "conditions": {},
        "issues": [],
    }
    if repo is None:
        result["issues"].append("No indexed repository; projection unavailable.")
        return result
    # Reuse existing model-visible context. Do not open files or measure predictions here.
    indexed = set()
    for file in repo.files:
        posix, windows = PurePosixPath(file.path), PureWindowsPath(file.path)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
            result["issues"].append(f"Indexed resource is not a bounded relative identifier: {file.path}")
        else:
            indexed.add(file.path)
    result["entries"] = [p for p in repo.entry_scripts if p in indexed]
    try:
        catalog = build_catalog(claim, materials)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result["issues"].append(f"Original source catalog unavailable: {exc}")
        catalog = {"sources": {}, "numbers": {}}
    sources = {}
    for sid, row in catalog["sources"].items():
        if row["kind"] == "claim_text":
            continue
        try:
            sources[sid] = resolve_source(catalog, sid, materials)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result["issues"].append(f"Source {sid} unavailable: {exc}")
    configurations = {}
    visible = {row["path"]: row for row in repository_files}
    for name in (p for p in repo.configs if p in indexed):
        try:
            if name not in visible:
                raise ValueError("Resource has no existing request-context text")
            cfg = json.loads(visible[name]["text"])
            if (
                not isinstance(cfg, dict)
                or set(cfg) != {"dataset", "metric", "settings"}
                or not isinstance(cfg["settings"], dict)
                or not isinstance(cfg["dataset"], str)
                or not cfg["dataset"]
                or not isinstance(cfg["metric"], str)
                or not cfg["metric"].strip()
            ):
                raise ValueError("Config does not match the finite recipe shape")
            configurations[name] = cfg
        except (ValueError, OSError, UnicodeError, KeyError, TypeError) as exc:
            result["issues"].append(f"Config {name} unavailable: {exc}")
    result["configs"] = list(configurations)
    result["data_candidates"] = [
        f.path
        for f in repo.files
        if f.path in indexed and f.path not in set(repo.entry_scripts) | set(configurations)
    ]
    for condition in claim.conditions:
        inventory = field_inventory(condition)
        eligible = {
            sid: source
            for sid, source in sources.items()
            if not source["covered"] or condition.id in source["covered"]
        }
        condition_choices = {
            "original_fields": inventory,
            "condition_source_ids": list(eligible),
            "by_config": {},
            "unsupported_without_config": not bool(configurations),
        }
        result["conditions"][condition.id] = condition_choices
        for name, cfg in configurations.items():
            rows = []
            for field, value in inventory.items():
                role = expected_field_role(field, value, cfg)
                reason = ""
                if role is None:
                    reason = "Unclassified original semantic obligation"
                elif (
                    role == "conclusion_boundary"
                    and field.startswith("/settings/qualifiers/")
                    and qualifier_category(value) is None
                ):
                    reason = "Unknown qualifier family"
                elif role == "sample_scope" and (type(value) is not int or value <= 0):
                    reason = "Sample scope requires a positive integer"
                elif role == "measurement_definition" and not isinstance(value, str):
                    reason = "Measurement definition requires original text"
                requirements = source_requirements(field, value, role)
                hits = {
                    sid: [k for k, matched in source_hits(requirements, source["quote"]).items() if matched]
                    for sid, source in eligible.items()
                }
                complete = [
                    sid for sid, names in hits.items() if requirements and set(names) == set(requirements)
                ]
                rows.append(
                    {
                        "path": field,
                        "allowed_roles": [] if reason else [role],
                        "unavailable_reason": reason,
                        "source_requirements": list(requirements),
                        "complete_source_ids": complete,
                        "partial_source_hits": {
                            sid: names for sid, names in hits.items() if names and sid not in complete
                        },
                        "source_selection": "Explicitly select original condition_source_ids. Required lexical predicates are rechecked on the newline-joined selected quotes; single-source completeness is a hint, not semantic support.",
                    }
                )
            # These exact sentences are structural candidates under this config, not ready targets.
            numbers = []
            runtime = Condition(
                id=condition.id, dataset=cfg["dataset"], metric=cfg["metric"], settings=cfg["settings"]
            )
            scope_names = [cfg["dataset"]]
            if isinstance(cfg["settings"].get("split"), str):
                scope_names.append(cfg["dataset"] + " " + cfg["settings"]["split"])
            compatible = (
                condition.dataset in scope_names
                and condition.metric == cfg["metric"]
                and condition.metric in {"accuracy", "exact-match accuracy"}
            )
            if compatible:
                for identifier, number in catalog["numbers"].items():
                    sentence = number["sentence"]
                    try:
                        scalar = _scalar_match(sentence, runtime)
                    except TargetBindingError:
                        continue
                    if scalar is None or scalar[2] is not None or not 0 <= scalar[1] <= 1:
                        continue
                    leading = len(sentence) - len(sentence.lstrip())
                    if scalar[0].span("value") != (
                        number["start"] - number["sentence_start"] - leading,
                        number["end"] - number["sentence_start"] - leading,
                    ):
                        continue
                    sids = [
                        sid
                        for sid, source in eligible.items()
                        if source["block_id"] == number["block_id"]
                        and source["start"] <= number["sentence_start"]
                        and source["end"] >= number["sentence_end"]
                    ]
                    if sids:
                        numbers.append(
                            {
                                "number_id": identifier,
                                "block_id": number["block_id"],
                                "source_ids": sids,
                                "sentence": sentence,
                                "token": number["token"],
                                "loc": number["loc"],
                            }
                        )
            condition_choices["by_config"][name] = {
                "configuration": cfg,
                "dataset_metric_identity_matches": compatible,
                "fields": rows,
                "prose_scalar_candidates": numbers,
                "consumer_still_checks": "Full claim, every setting/qualifier, exact source location, selected occurrence, resource AST, all data, independent scope and immutable hashes.",
            }
    return result
