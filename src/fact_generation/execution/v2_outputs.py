"""Read ordinary author JSON without supplying any missing values from a plan."""

import re
from typing import Any

from .tools.log_metrics import _collect_metrics_from_obj
from .tools.metrics import _get_json_path
from .tools.paper_tables import _is_metric_header, _metric_key
from .v2_config import OutputMapping


def _target_field(name: str | int) -> bool:
    token = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return token.startswith(("expected", "target", "paper", "ypaper"))


def _reject_failure(value: Any) -> None:
    if isinstance(value, dict) and (
        value.get("ok") is False
        or value.get("success") is False
        or str(value.get("status", "")).lower() in {"failed", "failure", "error"}
    ):
        raise ValueError("author metric output reports failure")


def _runtime_value(value: Any, path: list[str | int]) -> Any:
    """Check every selector ancestor before reading a runtime value."""
    _reject_failure(value)
    for component in path:
        if _target_field(component):
            raise ValueError("output mapping cannot select expected/target/paper values")
        value = _get_json_path(value, [component])
        _reject_failure(value)
    return value


def _actual_metrics(payload: Any) -> dict[str, list[float]]:
    if not isinstance(payload, dict):
        return {}
    actual: dict[str, list[float]] = {}
    for container in (payload, payload.get("metrics")):
        if not isinstance(container, dict):
            continue
        for name, value in container.items():
            if _target_field(name) or not _is_metric_header(name) or isinstance(value, (dict, list)):
                continue
            parsed = _collect_metrics_from_obj({name: value})
            key = _metric_key(name)
            if key in parsed:
                actual.setdefault(key, []).append(parsed[key])
    return actual


def decode_output(payload: Any, mapping: OutputMapping | None = None) -> tuple[list[dict], list[dict]]:
    if isinstance(payload, list):
        observations, audit = [], []
        for index, row in enumerate(payload):
            decoded, locations = decode_output(row, mapping)
            observations.extend(decoded)
            audit.extend({"row": index, **item} for item in locations)
        return observations, audit
    if not isinstance(payload, dict):
        raise ValueError("metric output must contain a JSON object or rows")
    _reject_failure(payload)
    if "observations" in payload and mapping is None:
        if not isinstance(payload["observations"], list):
            raise ValueError("observations must be a list")
        for observation in payload["observations"]:
            _reject_failure(observation)
            if isinstance(observation, dict):
                key = _metric_key(str(observation.get("metric", "")))
                if any(
                    actual != observation.get("value") for actual in _actual_metrics(payload).get(key, [])
                ):
                    raise ValueError(
                        f"canonical observation conflicts with an identified actual metric: {key}"
                    )
        return payload["observations"], [
            {"canonical_path": ["observations", index]} for index in range(len(payload["observations"]))
        ]
    mapping = mapping.model_copy(deep=True) if mapping else _infer_mapping(payload)
    root = _runtime_value(payload, mapping.root_path)
    dataset = _runtime_value(root, mapping.dataset_path)
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("runtime dataset is missing at the configured JSON path")
    settings = _runtime_value(root, mapping.settings_path) if mapping.settings_path is not None else {}
    if not isinstance(settings, dict):
        raise ValueError("runtime settings object is missing at the configured JSON path")
    settings = dict(settings)
    for name, path in mapping.settings_paths.items():
        value = _runtime_value(root, path)
        if value is None:
            raise ValueError(f"runtime setting is missing: {name} at {path}")
        settings[name] = value
    if mapping.settings_path is None and not mapping.settings_paths:
        raise ValueError("runtime settings require an actual object or metadata selectors")
    observations, audit = [], []
    for metric, path in mapping.metric_paths.items():
        if not path or _metric_key(str(path[-1])) != _metric_key(metric):
            raise ValueError("metric mapping must retain the metric named by the output field")
        value = _runtime_value(root, path)
        # Existing parser handles metric aliases and numeric strings. A one-field
        # object preserves the source path and prevents nested rows overwriting metrics.
        parsed = _collect_metrics_from_obj({metric: value})
        key = _metric_key(metric)
        if key not in parsed:
            raise ValueError(f"runtime metric is missing or nonnumeric: {path}")
        for container in (payload, root):
            if any(actual != parsed[key] for actual in _actual_metrics(container).get(key, [])):
                raise ValueError(f"output mapping conflicts with an identified actual metric: {key}")
        observations.append({"dataset": dataset, "metric": key, "settings": settings, "value": parsed[key]})
        audit.append(
            {
                "dataset_path": mapping.dataset_path,
                "metric_path": path,
                "settings_path": mapping.settings_path,
                "settings_paths": mapping.settings_paths,
                "root_path": mapping.root_path,
            }
        )
    if not observations:
        raise ValueError("runtime output has no named numeric metrics")
    return observations, audit


def _infer_mapping(payload: dict) -> OutputMapping:
    """Recognize common author layouts, including CompGCN's released evaluator."""
    mapping = OutputMapping()
    if not isinstance(payload.get("settings"), dict):
        mapping.settings_path = None
        metadata = {
            "split",
            "seed",
            "seeds",
            "model",
            "method",
            "score_func",
            "opn",
            "protocol",
            "optimizer",
            "batch_size",
            "epochs",
            "architecture",
            "loss",
            "aggregation",
        }
        mapping.settings_paths = {name: [name] for name in payload if name in metadata}
    metric_root = payload.get("metrics")
    if isinstance(metric_root, dict):
        mapping.metric_paths = {name: ["metrics", name] for name in metric_root if not _target_field(name)}
    else:
        mapping.metric_paths = {
            name: [name] for name in payload if _is_metric_header(name) and not _target_field(name)
        }
    return mapping
