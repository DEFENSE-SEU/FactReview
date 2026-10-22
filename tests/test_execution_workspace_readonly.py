"""One public runner control: input mounts stay readonly without observation."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from assessment.rules import assess_claim
from fact_generation.execution import runner as v2
from fact_generation.execution.resource_contract import _scientific_claim
from fact_generation.execution.tools import docker
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, Condition, ExecutionPlan, ExecutionTask, PaperTargetPassage
from schemas.materials import MaterialBlock, SharedMaterials
from tests.execution_science_fixtures import accuracy_fixture, record_metric_file, record_observer
from tests.test_execution_projection import bind, original_inputs, proposal_input
from util.subprocess_runner import CommandResult
from verification.experiment_targets import bind_execution_target


def _ordinary_inputs(root):
    repository = root / "repository"
    repository.mkdir()
    (repository / "eval.py").write_text("print('released evaluation')\n", encoding="utf-8")
    (repository / "weights.bin").write_bytes(b"released checkpoint bytes")
    text = "d1 test seed 1 accuracy is 0.9."
    location = ClaimLocation(page=1, char_start=0, char_end=len(text))
    paper = root / "paper.md"
    paper.write_text(text, encoding="utf-8")
    condition = Condition(id="c1", dataset="d1", metric="accuracy", settings={"split": "test", "seed": 1})
    claim = Claim(id="claim1", text=text, loc=location, source_block_id="b1", source_quote=text,
                  conditions=[condition], needs=["Experiments"])
    materials = SharedMaterials(paper_key="fixture", source_pdf="unused.pdf", markdown=text,
        markdown_path=str(paper), content_list_path="unused.json", provider="fixture",
        blocks=[MaterialBlock(id="b1", text=text, loc=location)], repository=index_repository(repository))
    binding = bind_execution_target(claim, condition,
        PaperTargetPassage(block_id="b1", quote=text, token="0.9", value_context=text), materials)
    plan = ExecutionPlan(id="p1", claim_id=claim.id, condition_ids=["c1"], target_conditions=[condition],
        task=ExecutionTask(entry_script="eval.py", command=["python", "eval.py"]), run_mode="evaluation",
        y_paper={"c1": 0.9}, target_bindings={"c1": binding}, feasibility="ready", priority="high")
    return plan, claim, materials


def test_default_runner_keeps_workspace_readonly_before_observer_qualification(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The readonly workspace control attempted an external operation")

    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr(docker, "_docker_proxy_env", lambda _: {})
    monkeypatch.setattr(docker, "_docker_run_user_args", lambda: [])
    monkeypatch.setattr(docker, "docker_cmd", lambda args: ["docker", *args])
    for name in ("EXECUTION_DOCKER_PAPER_PYTHON_IMAGE", "EXECUTION_DOCKER_EXTRA_PIP_PACKAGES"):
        monkeypatch.delenv(name, raising=False)
    helper = v2.docker_run_paper_image
    image_id = "sha256:" + "a" * 64
    for mode in ("ready_file", "unresolved", "unresolved_file", "projected", "duplicate_source"):
        root = tmp_path / mode
        root.mkdir()
        if mode == "projected":
            from verification.experiments import PlanCandidate, _plan

            claim, materials = original_inputs(root)
            data = proposal_input(claim, materials, root)
            binding = bind(claim, materials, data)
            plan = _plan(claim, materials, PlanCandidate.model_validate(data[4]),
                         projection_reviews={(0, "c1"): data[1]}, scope_audit=data[5])
            assert plan.target_bindings["c1"] == binding and plan.feasibility == "ready"
            payload = {"observations": [{"dataset": "MiniSet", "metric": "accuracy",
                "settings": {"split": "test", "model": "ExactMatch"}, "value": 0.75}]}
            monkeypatch.setattr(v2, "docker_ensure_paper_image", forbidden)
        else:
            plan, claim, materials = _ordinary_inputs(root)
            if mode.endswith("file"):
                plan.task.metric_output = "metrics.json"
            payload = accuracy_fixture((plan, claim, materials), monkeypatch)
            monkeypatch.setattr(v2, "docker_ensure_paper_image", lambda *a, **kw: (True, "fixture-image"))
        scientific_before = _scientific_claim(claim)
        plan_before = plan.model_dump(mode="json")
        original_command = list(plan.task.command)
        source_before = {row.path: (Path(materials.repository.root) / row.path).read_bytes()
                         for row in materials.repository.files}
        transports = []
        inspections = []
        helper_calls = []

        def command_builder(**kwargs):
            helper_calls.append(copy.deepcopy(kwargs))
            argv = helper(**kwargs)
            if mode == "duplicate_source":
                argv[2:2] = ["-v", str(Path(kwargs["paper_root_host"]).resolve()) + ":/app:rw"]
            return argv

        def transport(argv, *, cwd, timeout_sec):
            if argv[1:3] == ["image", "inspect"]:
                assert mode == "projected" and timeout_sec == 30
                inspections.append(argv)
                return CommandResult(argv, str(cwd), 0, json.dumps([{"Id": image_id,
                    "RepoTags": ["python:3.11"], "Config": {"Entrypoint": None}}]), "", .001,
                    termination="completed")
            assert argv[:2] == ["docker", "run"]
            transports.append(argv)
            assert mode not in {"unresolved_file", "duplicate_source"}
            mounts = [argv[index + 1] for index, token in enumerate(argv[:-1]) if token == "-v"]
            app = [value for value in mounts if value.split(":/app", 1)[-1] in {"", ":ro", ":rw"}]
            workspace = Path(cwd) / "workspace"
            scratch = Path(cwd) / "runtime_scratch"
            assert app == [f"{workspace.resolve()}:/app:ro"]
            assert f"{scratch.resolve()}:/workspace/run_dir" in mounts
            assert not any(value.startswith(str(Path(cwd).resolve()) + ":") for value in mounts)
            assert helper_calls[-1]["env_passthrough"] == []
            assert helper_calls[-1]["cmd"] == original_command
            assert helper_calls[-1]["cwd_container"] == "/app/."
            manifest = json.loads((Path(cwd) / "source_manifest.json").read_text(encoding="utf-8"))
            assert set(manifest) == set(source_before)
            assert all(manifest[name] == hashlib.sha256(raw).hexdigest() for name, raw in source_before.items())
            if mode == "ready_file":
                slot = next(value for value in mounts if value.endswith(":/app/metrics.json:rw"))
                assert Path(slot.removesuffix(":/app/metrics.json:rw")).is_relative_to(scratch)
                assert set(mounts) == {app[0], f"{scratch.resolve()}:/workspace/run_dir", slot,
                    f"{Path(cwd) / 'attempt_0/observer_trusted'}:/factreview-observer:ro"}
                record_metric_file(argv, "metrics.json", payload)
                record_observer(argv, cwd, payload)
                stdout = json.dumps({"ok": True, "out": "metrics.json"})
            else:
                assert set(mounts) == {app[0], f"{scratch.resolve()}:/workspace/run_dir"}
                assert argv[-len(original_command):] == original_command
                stdout = json.dumps(payload)
            return CommandResult(argv, str(cwd), 0, stdout, "", .002, termination="completed")

        monkeypatch.setattr(v2, "docker_run_paper_image", command_builder)
        monkeypatch.setattr(v2, "run_command", transport)
        result = v2.execute_plans([plan], [claim], materials, root / "execute",
            config=v2.ExecutionConfig(refine_with_llm=mode == "ready_file", max_attempts=0))
        ledger = result.ledger[0]
        assert len(ledger["attempts"]) == 1 and not ledger["repairs"]
        assert _scientific_claim(result.claims[0]) == scientific_before
        assert plan.model_dump(mode="json") == plan_before
        assert all((Path(materials.repository.root) / name).read_bytes() == raw for name, raw in source_before.items())
        workspace = Path(ledger["workspace"])
        assert all((workspace / name).read_bytes() == raw for name, raw in source_before.items())
        if mode in {"ready_file", "projected"}:
            assert len(transports) == 1 and ledger["reason"] == ""
            evidence = result.claims[0].evidence
            assert len(evidence) == 1 and evidence[0].sufficient and evidence[0].aligned
            assert evidence[0].direction == "support" and assess_claim(result.claims[0]).status.value == "supported"
            if mode == "projected":
                assert len(inspections) == 2 and inspections[-1][-1] == image_id
                measurement = json.loads(Path(evidence[0].pointer.locator).read_text(encoding="utf-8"))
                assert measurement["sample_count"] == 4 and measurement["numerator"] == 3
                assert measurement["measurement"]["value"] == .75
        else:
            assert not result.claims[0].evidence and assess_claim(result.claims[0]).status.value == "unverified"
            if mode == "unresolved":
                assert len(transports) == 1
                assert ledger["attempts"][0]["returncode"] == 0
                assert ledger["attempts"][0]["environment"]["runtime_observer"]["status"] == "unresolved"
            else:
                assert transports == [] and ledger["attempts"][0]["returncode"] == 1
                assert ledger["attempts"][0]["operation_failures"][0]["component"] == "execution.environment"
                assert "readonly" in ledger["reason"].lower()
