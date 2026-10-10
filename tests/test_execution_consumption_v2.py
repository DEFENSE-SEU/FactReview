"""Matching output labels and candidate files cannot prove resource consumption."""

from pathlib import Path

import pytest

from fact_generation.execution import v2
from fact_generation.execution.resource_contract import build_resource_contract
from preprocessing.materials import index_repository
from tests import test_execution_v2 as original_fixtures

original_inputs = original_fixtures.inputs
observation = original_fixtures.observation


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real execution and external services are forbidden")

    for name in (
        "llm.client.llm_json", "subprocess.Popen", "subprocess.run",
        "requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
        "fact_generation.execution.v2.docker_runner", "fact_generation.execution.v2.run_command",
        "fact_generation.execution.v2.docker_ensure_paper_image",
    ):
        monkeypatch.setattr(name, forbidden)


def test_matching_metadata_and_observed_tag_do_not_qualify_selected_resources(original_inputs, tmp_path):
    plan, claim, materials = original_inputs
    root = Path(materials.repository.root)
    (root / "data.json").write_text('[{"feature": 1, "label": 0}]', "utf-8")
    (root / "weights.json").write_text('{"bias": 0}', "utf-8")
    materials.repository = index_repository(root)
    plan.task.data_paths, plan.task.weight_paths = ["data.json"], ["weights.json"]
    plan.task.resource_contract = build_resource_contract(
        claim, materials, condition_ids=plan.condition_ids,
        entry_script=plan.task.entry_script, config=None,
        data_paths=plan.task.data_paths, weight_paths=plan.task.weight_paths,
    )
    original_claim = claim.model_dump(mode="json")
    original_files = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    calls = []

    def metadata_only(request):
        calls.append(request.command)
        return v2.RunOutcome(
            returncode=0, observations=[observation()],
            environment={"runtime_observer": {"status": "observed", "report_status": "completed"}},
        )

    assert v2.aligned(observation(), claim.conditions[0])
    result = v2.execute_plans(
        [plan], [claim], materials, tmp_path / "out", runner=metadata_only,
        config={"max_attempts": 0},
    )
    assert calls == [["python", "eval.py"]]
    assert not result.claims[0].evidence
    row = result.ledger[0]
    assert row["attempts"][0]["returncode"] == 0
    assert row["attempts"][0]["observations"][0]["value"] == 0.9
    assert row["alignment"][0]["aligned"] is False
    assert row["alignment"][0]["consumption"]["status"] == "unresolved"
    assert "consumption" in row["reason"].lower()
    assert result.delivery_checks == []
    assert claim.model_dump(mode="json") == original_claim
    assert {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()} == original_files
