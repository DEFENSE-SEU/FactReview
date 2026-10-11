"""Closed schema sidecars through the existing public, fully mocked L3 fixture."""
import json

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from fact_generation.execution import inference_runtime
from tests.test_inference_binding_diagnostics_v2 import (
    test_public_binding_diagnostics_are_closed_and_preserve_scientific_qualification as public_binding_fixture,
)


def test_public_schema_diagnostics_are_bounded_closed_and_preserve_binding(tmp_path, monkeypatch):
    marker = "PRIVATE_BINDING_TOKEN_never_publish_7281"
    original = inference_runtime.bind_inference_proposal
    original_request = inference_runtime.bind_inference_request
    saved_request = {}

    def invalid_proposal(proposal, **kwargs):
        if proposal.get("metric_adapter") == marker:
            # Each defect comes from the mocked model response, with real schema validation.
            del proposal["checkpoint_path"]
            proposal["sample_id_selector"] = [{marker: marker}]
            proposal["sources"]["qualifiers"][marker] = marker
            proposal["paper"]["dataset"][marker] = marker
            for index in range(20):
                proposal[f"{marker}_{index}"] = marker
        return original(proposal, **kwargs)

    def capture_request(request, **kwargs):
        if request.source_science.get("metric_adapter") == marker:
            saved_request.update(request=request, kwargs=kwargs)
        return original_request(request, **kwargs)

    monkeypatch.setattr(inference_runtime, "bind_inference_proposal", invalid_proposal)
    monkeypatch.setattr(inference_runtime, "bind_inference_request", capture_request)
    public_binding_fixture(tmp_path, monkeypatch)
    sidecar_path = tmp_path / "schema/run_0000/attempt_0/inference_schema_diagnostics.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert set(sidecar) == {"schema", "schema_id", "errors", "truncated"}
    assert sidecar["schema"] == "torch-inference-schema-diagnostics-v1"
    assert sidecar["schema_id"] == "proposal"
    assert sidecar["truncated"] is True and len(sidecar["errors"]) == 8
    pairs = {(row["field"], row["error_type"]) for row in sidecar["errors"]}
    assert ("checkpoint_path", "missing") in pairs
    assert ("metric_adapter", "literal_error") in pairs
    assert ("sample_id_selector.item", "string_type") in pairs
    assert ("sample_id_selector.item", "int_type") in pairs
    assert ("unknown_field", "extra_forbidden") in pairs
    assert all(set(row) == {"field", "error_type"} for row in sidecar["errors"])
    assert marker not in sidecar_path.read_text(encoding="utf-8")
    assert not (tmp_path / "guard/run_0000/attempt_0/inference_schema_diagnostics.json").exists()
    assert not (tmp_path / "healthy/run_0000/attempt_0/inference_schema_diagnostics.json").exists()

    # Unknown titles, dynamic role paths and custom error codes publish fixed fallbacks.
    error = ValidationError.from_exception_data(marker, [{
        "type": PydanticCustomError(marker, marker),
        "loc": ("sources", marker, "quote"), "input": {marker: marker},
    }])
    assert inference_runtime._schema_validation_diagnostics(error) == {
        "schema": "torch-inference-schema-diagnostics-v1", "schema_id": "unknown_schema",
        "errors": [{"field": "unknown_field", "error_type": "unknown_error"}], "truncated": False,
    }
    policy_error = ValidationError.from_exception_data("TorchInferenceLimits", [{
        "type": "missing", "loc": ("image_id",), "input": {marker: marker},
    }])
    assert inference_runtime._schema_validation_diagnostics(policy_error) == {
        "schema": "torch-inference-schema-diagnostics-v1", "schema_id": "policy",
        "errors": [{"field": "image_id", "error_type": "missing"}], "truncated": False,
    }

    # The sidecar is write-once; diagnostic failure retains the original unresolved result.
    original_sidecar = sidecar_path.read_bytes()
    def rejected_schema(*args, **kwargs):
        raise error

    monkeypatch.setattr(inference_runtime, "bind_inference_proposal", rejected_schema)
    assert original_request(saved_request["request"], **saved_request["kwargs"]) == {
        "status": "unresolved", "reason": "inference_request_binding_unavailable", "binding_failure": "invalid_schema",
    }
    assert sidecar_path.read_bytes() == original_sidecar
