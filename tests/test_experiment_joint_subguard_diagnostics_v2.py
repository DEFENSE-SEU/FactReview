"""Fixed joint diagnostics preserve original consumer results and model inputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from llm.client import LLMConfig
from tests import test_experiment_binding_repair_v2 as binding_fixture
from tests.experiment_authority_fixtures import authority_scope_fixture
from verification import experiments
from verification.experiment_catalog import build_catalog
from verification.experiment_sources import prepare_joint_candidate

MARKER = "PRIVATE_MOCK_JOINT_SUBGUARD_e94ad3"


@pytest.mark.parametrize("scenario", ["coverage_rejected", "repair_accepted", "repair_then_primary_failure", "healthy"])
def test_joint_subguard_preserves_results_and_prompts(tmp_path, monkeypatch, scenario):
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))

    def forbidden(*args, **kwargs):
        pytest.fail("Unmocked external boundary")

    for target in ("screening.checks.llm_json", "httpx.Client.send", "httpx.AsyncClient.send",
                   "requests.sessions.Session.request", "subprocess.run", "subprocess.Popen"):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr("common.run_stats.stats_path", lambda: None)
    sources = tmp_path / "author_sources"
    sources.mkdir()
    original_paper = binding_fixture.paper

    def source_local_paper(path):
        claim, materials, item, _, _ = original_paper(path)
        materials.source_pdf = str(path / "paper.pdf")
        materials.content_list_path = str(path / "fixture.json")
        catalog = build_catalog(claim, materials)
        bounded = prepare_joint_candidate(claim, materials, catalog, item, 0)
        return claim, materials, item, catalog, bounded

    monkeypatch.setattr(binding_fixture, "paper", source_local_paper)
    claim, materials, first, scope, repair = binding_fixture.responses(sources, "roles")
    if scenario == "coverage_rejected":
        scope["items"][0]["source_uses"][1]["source_id"] = f"foreign_{MARKER}"
        scope["items"][0]["rationale"] = f"{MARKER} primary_pointer scope_item_schema"
    elif scenario == "healthy":
        scope["items"][0]["source_uses"][1]["roles"] = ["metric_definition"]
    original = copy.deepcopy([claim.model_dump(mode="json"), materials.model_dump(mode="json"), first, scope, repair])
    original_pointer = experiments._paper_pointer
    original_repair = experiments.repair_bindings
    original_open = Path.open
    repaired = False

    def controlled_repair(**kwargs):
        nonlocal repaired
        result = original_repair(**kwargs)
        repaired = bool(result[0])
        return result

    def controlled_pointer(*args, **kwargs):
        if scenario == "repair_then_primary_failure" and repaired:
            raise ValueError(f"{MARKER} source_uses_member_coverage")
        return original_pointer(*args, **kwargs)

    monkeypatch.setattr(experiments, "repair_bindings", controlled_repair)
    monkeypatch.setattr(experiments, "_paper_pointer", controlled_pointer)
    snapshots = []
    for mode in ("baseline", "enabled", "diagnostic_io_failure"):
        repaired = False
        calls = []

        def model(**kwargs):
            calls.append({"module": kwargs["module"], "prompt": kwargs["prompt"], "system": kwargs["system"]})
            if kwargs["module"] == "verification.experiments":
                return copy.deepcopy(first)
            if kwargs["module"] == "verification.experiments.scope":
                return authority_scope_fixture(json.loads(kwargs["prompt"]), scope,
                    requirement="manuscript_reported_result", basis="reported_measurement", rationale=MARKER)
            assert kwargs["module"] == "verification.experiments.binding_repair"
            return copy.deepcopy(repair)

        directory = tmp_path / "operator" / mode / "diagnostics"

        def controlled_open(path, open_mode="r", *args, **kwargs):
            open_mode = kwargs.pop("mode", open_mode)
            if mode == "diagnostic_io_failure" and path.parent == directory and open_mode == "x":
                raise OSError(MARKER)
            return original_open(path, open_mode, *args, **kwargs)

        options = {"call": model, "diagnostics_dir": directory,
                   "scope_binding_repair_rounds": 0 if scenario == "coverage_rejected" else 1}
        if mode != "baseline":
            options["joint_subguard_diagnostics"] = True
        with monkeypatch.context() as control:
            control.setattr(Path, "open", controlled_open)
            result = experiments.verify_experiments(claim, materials, **options).model_dump(mode="json")
        snapshots.append((result, calls))
        assert [claim.model_dump(mode="json"), materials.model_dump(mode="json"), first, scope, repair] == original
        expected_modules = ["verification.experiments", "verification.experiments.scope"]
        if scenario in {"repair_accepted", "repair_then_primary_failure"}:
            expected_modules.append("verification.experiments.binding_repair")
        assert [call["module"] for call in calls] == expected_modules
        assert result["plans"] == []
        rejected = scenario in {"coverage_rejected", "repair_then_primary_failure"}
        assert any(row["kind"] == "evidence_validation_failed" for row in result["verification_limitations"]) == rejected
        assert any(row["sufficient"] for row in result["evidence"]) == (not rejected)
        if scenario == "repair_accepted":
            assert repaired and "binding_repair=accepted" in result["evidence"][0]["note"]
        if scenario == "repair_then_primary_failure":
            assert repaired and result["evidence"] == []
        paths = list(directory.glob("*.json"))
        assert len(paths) == int(rejected and mode != "diagnostic_io_failure")
        for path in paths:
            raw = path.read_bytes()
            assert len(raw) <= 512 and MARKER.encode() not in raw
            assert str(tmp_path).encode() not in raw
            payload = {"schema": "verification-rejection-diagnostics-v1", "branch": "Experiments",
                       "scope": "first_evidence_validation_rejection", "code": "experiments_joint_final_validation_failed"}
            if mode == "enabled":
                payload.update(schema="verification-rejection-diagnostics-v2",
                    first_subguard="source_uses_member_coverage" if scenario == "coverage_rejected" else "primary_pointer")
            assert json.loads(raw) == payload
    assert snapshots[1:] == [snapshots[0], snapshots[0]]
