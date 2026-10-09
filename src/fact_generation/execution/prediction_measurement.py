"""Host-side measurement of immutable released predictions, separate from stdout.

The accepted author program has a finite, checked data dependency. This module
independently reads the entire frozen data/config again; expected paper values
never enter the measurement. No inference or training is performed here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from verification.execution_projection import ProjectionError, entry_recipe, file_hash


def check_request(request, binding):
    projection = binding.projection
    if (
        request.command != ["python", "-I", "-S", projection.entry_script]
        or request.workdir != "."
        or request.metric_output is not None
        or request.output_mapping is not None
    ):
        raise ProjectionError(
            "Released-predictions request changed the verified direct author command/output contract"
        )


def measure_predictions(request, binding, materials, observation, output, *, runtime_environment=None):
    """Return independently computed output and provenance, preserving raw output."""
    check_request(request, binding)
    projection = binding.projection
    if observation.reported_variance is not None or observation.released_recomputation is not None:
        raise ProjectionError(
            "Released-predictions output contains an unverified alternative measurement contract"
        )
    if observation.unit not in (None, "fraction"):
        raise ProjectionError("Author output unit conflicts with the exact-match fraction recipe")
    snapshot = materials.model_copy(deep=True)
    snapshot.repository.root = request.workspace
    recipe = entry_recipe(
        snapshot, projection.entry_script, projection.config_path, projection.proposal.data_path
    )
    if recipe["repository_hashes"] != projection.repository_hashes:
        raise ProjectionError("Frozen measurement resources changed")
    # This result derives exclusively from the actual frozen resources. Paper
    # sample scope is checked later as an obligation, never used as denominator.
    cfg = recipe["configuration"]
    value = recipe["numerator"] / recipe["denominator"]
    if json.dumps(
        {"dataset": observation.dataset, "metric": observation.metric, "settings": observation.settings},
        sort_keys=True,
    ) != json.dumps(cfg, sort_keys=True):
        raise ProjectionError("Author output identity differs from the released config actually read")
    if not math.isclose(observation.value, value, rel_tol=1e-12, abs_tol=1e-12):
        raise ProjectionError("Author output value differs from independent full-data recomputation")
    if recipe["sample_count"] != projection.sample_count:
        raise ProjectionError("Actual complete sample count differs from the original paper's sample scope")
    if (recipe["label_key"], recipe["prediction_key"]) != (projection.label_key, projection.prediction_key):
        raise ProjectionError("Actual measurement fields differ from the independently bound definition")
    measured = {**cfg, "value": value, "unit": recipe["unit"]}
    artifact = Path(request.workspace) / projection.proposal.data_path
    audit = {
        "version": "host-exact-match-v1",
        "resource_mode": "released_predictions",
        "measurement_authority": "host full-data recomputation of frozen released resources",
        "model_inference_performed": False,
        "training_performed": False,
        "runtime_environment": runtime_environment or {"transport": "injected_or_unreported"},
        "execution_assumptions": "Fixed python -I -S command; operator-trusted CPython base image and Docker engine. This is a finite recipe proof and does not establish general resistance to a compromised interpreter/container.",
        "command": request.command,
        "author_observation": observation.model_dump(mode="json"),
        "author_observation_artifact": str(
            Path(request.run_dir) / f"attempt_{request.repair_round}" / "observations.json"
        ),
        "measurement": measured,
        "numerator": recipe["numerator"],
        "sample_count": recipe["sample_count"],
        "definition": "sum(label == prediction for every released row) / len(all released rows)",
        "label_key": recipe["label_key"],
        "prediction_key": recipe["prediction_key"],
        "repository_hashes": recipe["repository_hashes"],
        "artifact": str(artifact),
        "recipe_sha256": file_hash(__file__),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), "utf-8")
    return measured, {
        "released_artifact": True,
        "artifact_kind": "data",
        "environment_explanation_possible": False,
        "artifact_path": str(artifact),
        "artifact_sha256": recipe["repository_hashes"][projection.proposal.data_path],
        "repository": request.workspace,
        "recomputation_pointer": str(output),
    }
