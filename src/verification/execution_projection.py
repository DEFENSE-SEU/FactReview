"""Bounded, source-grounded projections for released exact-match predictions.

The author program is never executed here. Its accepted AST is a finite data
dependency contract; natural-language applicability remains an independent
scope decision. Unknown programs and qualifiers remain unavailable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from pypdf.errors import PdfReadError

from schemas.claim import (
    Condition,
    PaperTargetPassage,
    PaperTargetSelector,
    PredictionProjection,
    ProjectedExecutionTargetBinding,
    ProjectionRecord,
    ProjectionScopeDecision,
)
from screening.checks import grounded_paper_pointer
from verification.experiment_catalog import build_catalog, resolve_source
from verification.prose_numbers import resolve_number


class ProjectionError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pointer(materials, block, quote):
    try:
        return grounded_paper_pointer(materials, block, quote)
    except (PdfReadError, OSError, ValueError) as exc:
        raise ProjectionError(f"Original projection passage cannot be located: {type(exc).__name__}") from exc


def projection_snapshot(claim, materials):
    """Freeze original identities and actual paper/repository bytes across both calls."""
    files = {str(path): file_hash(path) for path in (materials.source_pdf, materials.markdown_path)}
    if materials.repository:
        files.update(
            {str(indexed_file(materials, row.path)): row.sha256 for row in materials.repository.files}
        )
    return digest(
        {
            "claim": claim.model_dump(mode="json"),
            "materials": materials.model_dump(mode="json"),
            "files": files,
        }
    )


def field_inventory(condition):
    """Semantic leaves, including empty containers; identity is fixed separately."""
    result = {}

    def visit(value, path):
        if isinstance(value, dict) and value:
            for key, item in value.items():
                visit(item, path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list) and value:
            for index, item in enumerate(value):
                visit(item, path + "/" + str(index))
        else:
            result[path] = value

    for key in ("dataset", "metric", "settings", "description"):
        visit(getattr(condition, key), "/" + key)
    return result


def indexed_file(materials, relative):
    repo = materials.repository
    if repo is None:
        raise ProjectionError("Released repository is missing")
    indexed = [row for row in repo.files if row.path == relative]
    root = Path(repo.root).resolve()
    path = (root / relative).resolve()
    if len(indexed) != 1 or not path.is_relative_to(root) or not path.is_file():
        raise ProjectionError("Projection resource lacks one indexed repository path")
    if file_hash(path) != indexed[0].sha256:
        raise ProjectionError("Indexed projection resource changed")
    return path


def entry_recipe(materials, entry, config, data):
    """Accept one transparent standard-library JSON evaluation program shape.

    Variable and resource names are arbitrary. Additional statements, filters,
    transformed values, alternate output expressions and dynamic paths fail.
    """
    entry_path = indexed_file(materials, entry)
    config_path = indexed_file(materials, config)
    data_path = indexed_file(materials, data)
    for row in materials.repository.files:
        parts = Path(row.path.replace("\\", "/")).parts
        if (
            Path(row.path).name in {"json.py", "pathlib.py", "sitecustomize.py", "usercustomize.py"}
            or any(part in {"json", "pathlib"} for part in parts[:-1])
            or Path(row.path).suffix == ".pth"
        ):
            raise ProjectionError(
                "Released files can shadow the recipe's trusted standard-library dependencies"
            )
    if entry not in materials.repository.entry_scripts or config not in materials.repository.configs:
        raise ProjectionError("Entry/config are absent from their repository index categories")
    try:
        tree = ast.parse(entry_path.read_text("utf-8"))
    except SyntaxError as exc:
        raise ProjectionError("Released entry cannot be parsed as the finite Python recipe") from exc
    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 8:
        raise ProjectionError("Entry has statements outside the released-predictions recipe")
    if ast.dump(body[0]) != ast.dump(ast.parse("import json").body[0]) or ast.dump(body[1]) != ast.dump(
        ast.parse("from pathlib import Path").body[0]
    ):
        raise ProjectionError("Recipe requires explicit standard-library json/Path imports")
    assignments = body[2:7]
    if any(
        not isinstance(row, ast.Assign) or len(row.targets) != 1 or not isinstance(row.targets[0], ast.Name)
        for row in assignments
    ):
        raise ProjectionError("Recipe assignments are not simple unique data dependencies")
    names = [row.targets[0].id for row in assignments]
    if len(set(names)) != len(names) or set(names) & {"json", "Path", "sum", "len", "print", "__file__"}:
        raise ProjectionError("Recipe variables overwrite a dependency")
    root_name, cfg_name, rows_name, count_name, accuracy_name = names
    root_expr = assignments[0].value
    if (
        not isinstance(root_expr, ast.Subscript)
        or not isinstance(root_expr.slice, ast.Constant)
        or type(root_expr.slice.value) is not int
    ):
        raise ProjectionError("Recipe root must be a fixed parent of its entry file")
    depth = root_expr.slice.value
    if (
        depth < 0
        or depth >= len(entry_path.parents)
        or entry_path.parents[depth] != Path(materials.repository.root).resolve()
    ):
        raise ProjectionError("Recipe file parent does not identify the indexed repository")
    template_root = ast.parse(f"Path(__file__).resolve().parents[{depth}]", mode="eval").body
    if ast.dump(root_expr) != ast.dump(template_root):
        raise ProjectionError("Recipe root contains an unverified operation")
    for assignment, path in ((assignments[1], config), (assignments[2], data)):
        allowed = [
            f"json.loads(({root_name} / {path!r}).read_text())",
            f"json.loads(({root_name} / {path!r}).read_text(encoding='utf-8'))",
        ]
        if ast.dump(assignment.value) not in {
            ast.dump(ast.parse(code, mode="eval").body) for code in allowed
        }:
            raise ProjectionError("Entry does not load the complete bound JSON resource")
    count = assignments[3].value
    if (
        not isinstance(count, ast.Call)
        or len(count.args) != 1
        or not isinstance(count.args[0], ast.GeneratorExp)
    ):
        raise ProjectionError("Entry must sum exact pairwise equality over all rows")
    generator = count.args[0]
    if len(generator.generators) != 1 or not isinstance(generator.generators[0].target, ast.Name):
        raise ProjectionError("Recipe generator must visit the unfiltered full row list")
    item = generator.generators[0].target.id
    if item in set(names) | {"json", "Path", "sum", "len", "print"}:
        raise ProjectionError("Generator variable shadows a recipe dependency")
    comparison = generator.elt
    if not isinstance(comparison, ast.Compare) or len(comparison.comparators) != 1:
        raise ProjectionError("Recipe requires one label/prediction equality")
    operands = [comparison.left, comparison.comparators[0]]
    if any(
        not isinstance(node, ast.Subscript)
        or not isinstance(node.slice, ast.Constant)
        or not isinstance(node.slice.value, str)
        for node in operands
    ):
        raise ProjectionError("Recipe equality selects fixed JSON field names")
    label, prediction = [node.slice.value for node in operands]
    if label == prediction:
        raise ProjectionError("A label cannot be compared with itself as a prediction")
    expected = f"sum({item}[{label!r}] == {item}[{prediction!r}] for {item} in {rows_name})"
    if ast.dump(count) != ast.dump(ast.parse(expected, mode="eval").body):
        raise ProjectionError("Entry changes the full-row exact-match operation")
    expected_value = ast.parse(f"{count_name} / len({rows_name})", mode="eval").body
    if ast.dump(assignments[4].value) != ast.dump(expected_value):
        raise ProjectionError("Entry does not divide the computed count by the same full row count")
    expected_output = ast.parse(
        f"print(json.dumps({{'observations': [{{**{cfg_name}, 'value': {accuracy_name}}}]}}))"
    ).body[0]
    if ast.dump(body[7]) != ast.dump(expected_output):
        raise ProjectionError("Author output lacks direct config and computed-value data dependencies")
    configuration = json.loads(config_path.read_text("utf-8"))
    rows = json.loads(data_path.read_text("utf-8"))
    if (
        not isinstance(configuration, dict)
        or set(configuration) != {"dataset", "metric", "settings"}
        or not isinstance(configuration["settings"], dict)
    ):
        raise ProjectionError("Released config must declare exactly dataset, metric and runtime settings")
    if not isinstance(rows, list) or not rows:
        raise ProjectionError("Released prediction data must be a nonempty complete JSON list")
    for row in rows:
        if not isinstance(row, dict) or label not in row or prediction not in row:
            raise ProjectionError("Every released row must contain its label and prediction")
        a, b = row[label], row[prediction]
        if type(a) is not type(b) or type(a) not in {str, int, bool}:
            raise ProjectionError(
                "Exact-match recipe requires same-type categorical string/integer/boolean pairs"
            )
    return {
        "configuration": configuration,
        "sample_count": len(rows),
        "label_key": label,
        "prediction_key": prediction,
        "numerator": sum(row[label] == row[prediction] for row in rows),
        "denominator": len(rows),
        "unit": "fraction",
        "repository_hashes": {
            name: file_hash(indexed_file(materials, name)) for name in (entry, config, data)
        },
    }


def projection_context(claim, materials):
    files = []
    if materials.repository:
        for row in materials.repository.files:
            if row.kind in {"entry", "config", "documentation"}:
                path = indexed_file(materials, row.path)
                if path.stat().st_size <= 100_000:
                    files.append({"path": row.path, "sha256": row.sha256, "text": path.read_text("utf-8")})
    return {
        "field_inventory": {c.id: field_inventory(c) for c in claim.conditions},
        "repository_files": files,
        "limit": "Only finite full-list JSON exact-match released-predictions evaluation. No model inference or training proof.",
    }


def decode_projection_reviews(response, plans):
    """Plan-local tombstones; malformed plan reviews never alter paper evidence."""
    expected = {
        (i, target.condition_id)
        for i, plan in enumerate(plans)
        for target in plan.targets
        if target.projection is not None
    }
    accepted, invalid, seen, errors = {}, set(), set(), []
    rows = response.get("plan_projection_reviews", [])
    if not isinstance(rows, list):
        return {}, ["Plan projection reviews must be a list"]
    for raw in rows:
        index = raw.get("plan_index") if isinstance(raw, dict) else None
        condition = raw.get("condition_id") if isinstance(raw, dict) else None
        key = (index, condition.strip()) if type(index) is int and isinstance(condition, str) else None
        try:
            if key is None or key not in expected:
                invalid.update(expected)
                raise ProjectionError("Projection review has no strict known plan/condition identity")
            if key in seen:
                raise ProjectionError("Duplicate projection decision cannot restore an earlier decision")
            seen.add(key)
            row = ProjectionScopeDecision.model_validate(raw)
            accepted[key] = row
        except (ValueError, TypeError) as exc:
            if key in expected:
                invalid.add(key)
            errors.append(str(exc))
    for key in expected - accepted.keys():
        errors.append(f"Missing independent plan projection decision: {key}")
    return {key: value for key, value in accepted.items() if key not in invalid}, errors


def _words_count(value):
    names = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ]
    return names[value] if 0 <= value < len(names) else str(value)


def bind_projection_target(
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
    """Reconstruct source, resource, field-role and independent scope obligations."""
    from verification.experiment_targets import _claim_fingerprint, _fingerprint, _scalar_match

    proposal = PredictionProjection.model_validate(
        proposal.model_dump() if hasattr(proposal, "model_dump") else proposal
    )
    review = ProjectionScopeDecision.model_validate(
        review.model_dump() if hasattr(review, "model_dump") else review
    )
    if (
        next((c for c in claim.conditions if c.id == condition.id), None) != condition
        or review.condition_id != condition.id
    ):
        raise ProjectionError("Projection must retain its exact original condition")
    if (
        review.classification != "absolute_fixed_predictions"
        or not all(
            (
                review.dataset_identity_confirmed,
                review.measurement_definition_confirmed,
                review.all_original_qualifiers_preserved,
            )
        )
        or review.unresolved
    ):
        raise ProjectionError("Independent projection review leaves original obligations unresolved")
    inventory = field_inventory(condition)
    paths = [row.path for row in proposal.field_roles]
    if (
        len(paths) != len(set(paths))
        or set(paths) != set(inventory)
        or len(review.confirmed_field_paths) != len(set(review.confirmed_field_paths))
        or set(review.confirmed_field_paths) != set(paths)
    ):
        raise ProjectionError(
            "Projection must classify and independently confirm every original semantic leaf once"
        )
    if condition.metric not in {"accuracy", "exact-match accuracy"}:
        raise ProjectionError(
            "Projection recipe only establishes absolute exact-match accuracy; compound counts are outside its contract"
        )
    if not entry_script or not config_path:
        raise ProjectionError("Released-predictions entry and config are required")
    recipe = entry_recipe(materials, entry_script, config_path, proposal.data_path)
    cfg = recipe["configuration"]
    if cfg["metric"] != condition.metric or not isinstance(cfg["dataset"], str) or not cfg["dataset"]:
        raise ProjectionError("Bound config does not name the original measured quantity")
    runtime = Condition(
        id=condition.id, dataset=cfg["dataset"], metric=cfg["metric"], settings=cfg["settings"]
    )
    scope_names = [cfg["dataset"]]
    if isinstance(cfg["settings"].get("split"), str):
        scope_names.append(cfg["dataset"] + " " + cfg["settings"]["split"])
    if condition.dataset not in scope_names:
        raise ProjectionError("Original dataset/split has no exact bound config identity")
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
        ):
            continue
        names = [other["dataset"]]
        if isinstance(other["dataset"], str) and isinstance(other["settings"].get("split"), str):
            names.append(other["dataset"] + " " + other["settings"]["split"])
        matching_roles = all(
            other["settings"].get(key) == condition.settings[key]
            for key in ("model", "method")
            if key in condition.settings
        )
        if other["metric"] == condition.metric and condition.dataset in names and matching_roles:
            decompositions.add(digest({"dataset": other["dataset"], "split": other["settings"].get("split")}))
    if len(decompositions) > 1:
        raise ProjectionError("Multiple indexed configurations give unresolved dataset/split decompositions")
    catalog = build_catalog(claim, materials)
    sources = {}
    for sid in dict.fromkeys(
        [sid for row in proposal.field_roles for sid in row.source_ids] + review.source_ids
    ):
        source = resolve_source(catalog, sid, materials)
        if not source.get("block_id") or (source["covered"] and condition.id not in source["covered"]):
            raise ProjectionError("Projection requires located original paper sources")
        block = next(b for b in materials.blocks if b.id == source["block_id"])
        sources[sid] = _pointer(materials, block, source["quote"])
    if not set(review.source_ids).issuperset(sid for row in proposal.field_roles for sid in row.source_ids):
        raise ProjectionError("Independent projection review omitted a proposed source")
    reported = PaperTargetPassage.model_validate(
        reported.model_dump() if hasattr(reported, "model_dump") else reported
    )
    selector = PaperTargetSelector.model_validate(
        selector.model_dump() if hasattr(selector, "model_dump") else selector
    )
    if selector.number_id is None:
        raise ProjectionError("The first projection recipe requires a prose scalar number occurrence")
    number = resolve_number(catalog["numbers"], selector.number_id, materials)
    block = next(b for b in materials.blocks if b.id == reported.block_id)
    pointer = _pointer(materials, block, reported.quote)
    if (
        number["block_id"] != block.id
        or reported.token != number["token"]
        or number["quote"] not in reported.quote
    ):
        raise ProjectionError("Projected target selector is outside its exact reported passage")
    scalar = _scalar_match(number["quote"], runtime)
    if scalar is None or scalar[1] != float(reported.token) or scalar[2] is not None:
        raise ProjectionError("Reported number does not identify the bound subject and absolute accuracy")
    leading = len(number["quote"]) - len(number["quote"].lstrip())
    if scalar[0].span("value") != (
        number["start"] - number["sentence_start"] - leading,
        number["end"] - number["sentence_start"] - leading,
    ):
        raise ProjectionError("Selected occurrence is a setting or another numeric role")
    context = reported.value_context or reported.quote
    if (
        block.text.count(reported.quote) != 1
        or reported.quote.count(context) != 1
        or number["quote"] not in context
    ):
        raise ProjectionError("Reported context does not uniquely contain the selected scalar sentence")
    value = scalar[1]
    if not 0 <= value <= 1:
        raise ProjectionError("Fraction target is outside its explicit scale")
    sample_values = []
    known_runtime = set()
    for row in proposal.field_roles:
        field, actual = row.path, inventory[row.path]
        text = "\n".join(sources[sid].quote for sid in row.source_ids)
        key = field.removeprefix("/settings/")
        if field == "/dataset":
            expected_role = "dataset_identity"
        elif field == "/metric":
            expected_role = "metric"
        elif field == "/description":
            expected_role = "conclusion_boundary"
        elif key in cfg["settings"] and not isinstance(actual, (dict, list)):
            expected_role = "runtime_setting"
            if json.dumps(actual) != json.dumps(cfg["settings"][key]):
                raise ProjectionError("Original runtime setting differs from the actual config")
            known_runtime.add(key)
        elif key in {"accuracy", "reported_value"}:
            expected_role = "reported_value"
            if type(actual) not in {int, float} or actual != value:
                raise ProjectionError("Original reported-value field differs from its scalar selector")
        elif key in {"examples", "sample_count"}:
            expected_role = "sample_scope"
            if type(actual) is not int or actual <= 0:
                raise ProjectionError("Original sample count must be one positive integer")
            words = rf"(?:{actual}|{_words_count(actual)})"
            if not re.search(
                rf"\b{words}\s+(?:(?:fixed|test|validation|training)\s+)*(?:examples|predictions)\b",
                text,
                re.I,
            ):
                raise ProjectionError("Sample count lacks an explicit scoped paper passage")
            sample_values.append(actual)
        elif key in {"accuracy_definition", "measurement_definition"}:
            expected_role = "measurement_definition"
            if not isinstance(actual, str) or not all(
                re.search(term, text, re.I)
                for term in (r"\bfraction\b", r"\bpredictions?\b", r"\blabels?\b", r"\bequal(?:ity)?\b")
            ):
                raise ProjectionError("Exact-match fraction definition lacks original source grounding")
        elif re.fullmatch(r"qualifiers/[0-9]+", key):
            expected_role = "conclusion_boundary"
            families = (
                r"no repeated.run uncertainty(?: claimed)?",
                r"no population.performance conclusion(?: claimed)?",
                r"no ranking against other models",
            )
            if not isinstance(actual, str) or not any(
                re.fullmatch(pattern, actual, re.I) for pattern in families
            ):
                raise ProjectionError("Unknown qualifier cannot be discharged as a conclusion boundary")
            category = (
                "ranking"
                if "ranking" in actual.lower()
                else ("population" if "population" in actual.lower() else "repeated")
            )
            if not re.search(rf"\bno\b[^.]*\b{category}", text, re.I):
                raise ProjectionError("Conclusion boundary lacks its original negative qualifier")
        else:
            raise ProjectionError("Unclassified original runtime or semantic obligation")
        if row.role != expected_role:
            raise ProjectionError("Proposed field role would drop or change an original obligation")
    if len(sample_values) != 1:
        raise ProjectionError("Released-predictions projection requires one explicit original sample scope")
    # A split carried in the original combined dataset remains a runtime obligation.
    if "split" in cfg["settings"] and condition.dataset == cfg["dataset"] + " " + str(
        cfg["settings"]["split"]
    ):
        known_runtime.add("split")
    if set(cfg["settings"]) != known_runtime:
        raise ProjectionError("Config contains an additional unbound runtime setting")
    claim_text = re.sub(
        rf"\s+over\s+(?:{sample_values[0]}|{_words_count(sample_values[0])})\s+examples(?=[.!]?$)",
        "",
        claim.text,
        flags=re.I,
    )
    claim_scalar = _scalar_match(claim_text, runtime, claim_statement=True)
    if claim_scalar is None or claim_scalar[1] != value:
        raise ProjectionError(
            "Original claim includes a comparative, inference or unsupported non-scalar assertion"
        )
    audit_path = Path(audit_path) if audit_path else None
    if audit_path is None or not audit_path.is_file():
        raise ProjectionError("Independent plan scope audit is required before execution")
    audit = json.loads(audit_path.read_text("utf-8"))
    from schemas.claim import Claim

    if _claim_fingerprint(Claim.model_validate(audit["input"]["claim"])) != _claim_fingerprint(claim):
        raise ProjectionError("Plan review was made for a different original claim")
    if audit["input"].get("paper_blocks") != [b.model_dump(mode="json") for b in materials.blocks]:
        raise ProjectionError("Original paper blocks differ from the independently reviewed materials")
    if audit["input"].get("execution_projection_context") != projection_context(claim, materials):
        raise ProjectionError("Released resources differ from the independently reviewed context")
    raw_plan = audit["input"]["candidate_plans"][review.plan_index]
    raw_targets = [row for row in raw_plan["targets"] if row.get("condition_id") == condition.id]
    if (
        len(raw_targets) != 1
        or raw_targets[0].get("projection") != proposal.model_dump(mode="json")
        or PaperTargetPassage.model_validate(raw_targets[0]["reported"]) != reported
        or PaperTargetSelector.model_validate(raw_targets[0]["selector"]) != selector
        or raw_plan.get("entry_script") != entry_script
        or raw_plan.get("config") != config_path
        or raw_plan.get("run_mode") != "evaluation"
        or raw_plan.get("data_paths") != [proposal.data_path]
        or raw_plan.get("weight_paths")
    ):
        raise ProjectionError("Plan projection/resources differ from the independently reviewed proposal")
    from verification.experiments import PlanCandidate

    reviews, _ = decode_projection_reviews(
        audit.get("response", {}),
        [PlanCandidate.model_validate(p) for p in audit["input"]["candidate_plans"]],
    )
    if (
        audit.get("response", {}).get("schema_version") != "catalog-v2"
        or reviews.get((review.plan_index, condition.id)) != review
    ):
        raise ProjectionError("Projection review does not identify one retained independent raw decision")
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
        projection=ProjectionRecord(
            proposal=proposal,
            runtime_target=runtime,
            sample_count=sample_values[0],
            source_pointers=sources,
            source_hashes={key: file_hash(ptr.locator) for key, ptr in sources.items()},
            paper_hashes={
                str(path): file_hash(path) for path in (materials.source_pdf, materials.markdown_path)
            },
            repository_hashes=recipe["repository_hashes"],
            repository_root=str(Path(materials.repository.root).resolve()),
            entry_script=entry_script,
            config_path=config_path,
            label_key=recipe["label_key"],
            prediction_key=recipe["prediction_key"],
            scope_review=review,
            scope_audit=str(audit_path.resolve()),
            scope_audit_sha256=file_hash(audit_path),
            recipe_sha256=file_hash(__file__),
        ),
    )
