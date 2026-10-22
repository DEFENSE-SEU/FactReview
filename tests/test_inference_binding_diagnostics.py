"""Public binding diagnostics: external boundaries mocked, scientific guards real."""
import ast
import hashlib
import json
import struct
from pathlib import Path

import pytest

from assessment.rules import assess_claims
from common import run_stats
from fact_generation.execution import runner as v2
from fact_generation.execution.resource_contract import build_resource_contract
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials
from tests.test_torch_inference_execution import SOURCE
from util.subprocess_runner import CommandResult
from verification.experiment_targets import bind_execution_target


def test_public_binding_diagnostics_are_closed_and_preserve_scientific_qualification(tmp_path, monkeypatch):
    marker = "PRIVATE_BINDING_TOKEN_never_publish_7281"
    calls = {"refinement": 0, "semantic": 0, "author": 0, "inspect": 0, "external_real": 0}

    def forbidden(*args, **kwargs):
        calls["external_real"] += 1
        pytest.fail("Real LLM, retrieval, network, Docker and subprocess are forbidden")

    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "subprocess.Popen", "subprocess.run", "socket.create_connection"):
        monkeypatch.setattr(name, forbidden)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    monkeypatch.setattr("fact_generation.execution.tools.docker._docker_proxy_env", lambda _: {})
    monkeypatch.setattr(v2, "docker_ensure_paper_image", forbidden)
    repo = tmp_path / "repo"
    repo.mkdir()
    definition = ("Numeric test is exactly the two held-out IDs a and b with one feature and labels 0 and 1. "
        "Affine uses the released state weight 1 and bias 0 in one linear layer; it predicts x. "
        "Both samples belong to the complete test split. MSE is the mean of both squared prediction errors. "
        "Use seed 1, CPU evaluation, no preprocessing and no training-history claim.")
    quote = "On Numeric test seed 1 sample_count 2, model Affine reports mse 1."
    paper = quote + "\n" + definition
    configuration = {"dataset": "Numeric", "model": "Affine", "split": "test", "metric": "mse", "seed": 1, "sample_count": 2}
    rows = [{"id": "a", "features": [1.0], "label": 0}, {"id": "b", "features": [2.0], "label": 1}]
    for name, raw in {"eval.py": SOURCE.encode(), "README.md": definition.encode(),
            "config.json": json.dumps(configuration).encode(), "weights.pt": b"mock actual binary checkpoint boundary",
            "test.jsonl": ("\n".join(json.dumps(row) for row in rows) + "\n").encode()}.items():
        (repo / name).write_bytes(raw)
    markdown = tmp_path / "paper.md"
    markdown.write_bytes(paper.encode())
    condition = Condition(id="c1", dataset="Numeric", metric="mse",
        settings={"model": "Affine", "split": "test", "seed": 1, "sample_count": 2})
    claim = Claim(id="c", text=quote, source_quote=quote, source_block_id="paper",
        loc=ClaimLocation(page=1, char_start=0, char_end=len(quote)), conditions=[condition], needs=["Experiments"],
        source_refs=[ClaimSourceRef(source_block_id="paper", source_quote=definition, covered=["c1"],
            loc=ClaimLocation(page=1, char_start=paper.index(definition), char_end=len(paper)))])
    materials = SharedMaterials(paper_key="numeric", source_pdf=str(markdown), markdown=paper,
        markdown_path=str(markdown), content_list_path="content.json", provider="fixture", repository=index_repository(repo),
        blocks=[MaterialBlock(id="paper", text=paper, loc=ClaimLocation(page=1, char_start=0, char_end=len(paper)))])
    task = ExecutionTask(entry_script="eval.py", config="config.json", command=["python", "eval.py"],
        data_paths=["test.jsonl"], weight_paths=["weights.pt"])
    task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"], entry_script="eval.py",
        config="config.json", data_paths=task.data_paths, weight_paths=task.weight_paths)
    plan = ExecutionPlan(id="p", claim_id="c", condition_ids=["c1"], target_conditions=[condition], task=task,
        y_paper={"c1": 1}, run_mode="evaluation", feasibility="ready", priority="high",
        target_bindings={"c1": bind_execution_target(claim, condition,
            {"block_id": "paper", "quote": quote, "token": "1", "value_context": "mse 1"}, materials)})
    originals = [item.model_dump(mode="json") for item in (claim, materials, plan)]
    file_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in repo.iterdir()}
    policy = {"image_id": "sha256:" + "a" * 64, "package_manifest_sha256": "b" * 64}
    roles = ("dataset", "split", "model", "metric", "population", "qualifiers")
    nodes = {}
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.FunctionDef):
            nodes[node.name] = node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    nodes[node.name + "." + child.name] = child
    current = {"mode": "guard"}

    def completion(prompt, system, cfg, *, module):
        assert module == "execution"
        payload = json.loads(prompt)
        run_stats.record_llm_call(module=module, provider="mock", model="binding-diagnostics",
            usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
        if "plan" in payload:
            calls["refinement"] += 1
            assert payload["scientific_claim"]["conditions"] == originals[0]["conditions"]
            sources = {}
            for role in roles:
                path = "eval.py" if role in ("model", "metric") else "README.md"
                text = ast.get_source_segment(SOURCE, nodes["Model.forward" if role == "model" else "metric"]) if path == "eval.py" else definition
                whole = payload["science_files"][path]
                sources[role] = {"path": path, "quote": text, "start": whole.index(text), "end": whole.index(text) + len(text)}
            proposal = {"version": "torch-cpu-inference-v1", "condition_id": "c1", "checkpoint_path": "weights.pt",
                "dataset_path": "test.jsonl", "codec": "indexed-jsonl-numeric-v1", "id_key": "id", "features_key": "features", "label_key": "label",
                "sites": {"getitem": 0, "model": 1, "prediction": 2, "metric": 3},
                "sample_id_selector": ["id"], "sample_features_selector": ["features"], "sample_label_selector": ["label"],
                "batch_ids_selector": ["id"], "batch_features_selector": ["features"], "batch_labels_selector": ["label"],
                "prediction_argument": "output", "metric_prediction_argument": "predictions", "metric_label_argument": "labels",
                "raw_value_selector": ["mse"], "prediction_adapter": "scalar_regression", "metric_adapter": "mean_squared_error",
                "dataset_selector": ["dataset"], "model_selector": ["model"], "split_selector": ["split"], "metric_selector": ["metric"],
                "settings_selectors": {key: [key] for key in condition.settings}, "sources": sources,
                "paper": {role: {"block_id": "paper", "quote": definition, "start": paper.index(definition), "end": len(paper)} for role in roles}}
            if current["mode"] == "guard":
                proposal["sources"]["qualifiers"]["start"] += 1
            if current["mode"] == "schema":
                proposal["metric_adapter"] = marker
            return {"command": ["python", "eval.py"], "metric_output": None, "science_proposal": proposal,
                "source_sites": [{"path": "eval.py", "qualname": name, "firstlineno": nodes[name].lineno}
                    for name in ("Samples.__getitem__", "Model.forward", "prediction", "metric")]}
        calls["semantic"] += 1
        context = payload["context"]
        assert current["mode"] == "healthy"
        assert context["catalog"]["actual/inference"]["complete_sample_count"] == 2
        assert context["scientific_claim"]["conditions"] == originals[0]["conditions"]
        return {"version": context["version"], "context_digest": context["context_digest"], "condition_id": "c1",
            "obligations": [{"id": key, "decision": "confirmed", "source_ids": source_ids,
                "rationale": "Complete original selected source correspondence."} for key, source_ids in context["obligations"].items()], "unresolved": []}

    def tensor(token, values, shape, *, dtype="torch.float64"):
        flat = values if len(shape) == 1 else [item for row in values for item in row]
        raw = b"".join(struct.pack("<q" if dtype == "torch.int64" else "<d", item) for item in flat)
        return {"token": token, "values": values, "shape": shape, "dtype": dtype,
            "sha256": hashlib.sha256(raw).hexdigest(), "tensor_version": 0}

    def transport(command, cwd, timeout_sec):
        if command[:3] == ["docker", "image", "inspect"]:
            calls["inspect"] += 1
            assert current["mode"] == "healthy" and timeout_sec == 30
            return CommandResult(command, cwd, 0, policy["image_id"] + "\n", "", .01, termination="completed")
        assert command[:2] == ["docker", "run"] and policy["image_id"] in command
        calls["author"] += 1
        launch = json.loads((Path(cwd) / "attempt_0/runtime_observer_launch.json").read_text(encoding="utf-8"))
        if current["mode"] != "healthy":
            assert launch["status"] == "unresolved" and command[-2:] == ["python", "eval.py"]
            return CommandResult(command, cwd, 0, '{"mse":1.0}\n', "", .01, termination="completed")
        assert launch["version"] == "readonly-torch-inference-v1" and launch["mounts"]["/app"]["mode"] == "ro"
        config = json.loads((Path(launch["trusted_dir"]) / "config.json").read_text(encoding="utf-8"))
        features = tensor(3, [[1.0], [2.0]], [2, 1])
        labels = tensor(4, [0, 1], [2], dtype="torch.int64")
        output = tensor(5, [1.0, 2.0], [2])
        events = [{"role": "getitem", "site": 0, "invocation": i + 1, "order": i + 1, "parent_invocation": None,
            "ordinal": i, "id": row["id"], "feature_snapshot": tensor(i + 10, row["features"], [1]),
            "features": row["features"], "label": row["label"], "label_batch_token": 4} for i, row in enumerate(rows)]
        events += [{"role": "model", "site": 1, "invocation": 3, "order": 3, "parent_invocation": None, "model_token": 1,
            "torch_initial_seed": 1, "batch_features": features, "batch_labels": labels, "batch_ids": ["a", "b"],
            "output_token": 5, "output": output, "module_call_completed": True},
            {"role": "prediction", "site": 2, "invocation": 4, "order": 4, "parent_invocation": None,
                "input_token": 5, "input": output, "output_token": 5, "output": output, "model_invocation": 3},
            {"role": "metric", "site": 3, "invocation": 5, "order": 5, "parent_invocation": None,
                "predictions": [output], "labels": [labels], "value": 1.0}]
        report = {"version": "torch-cpu-inference-v1", "binding_digest": launch["inference"]["binding_digest"],
            "unresolved": [], "execution": {"status": "completed"}, "calls": events,
            "samples": [{"ordinal": i, **row, "batch_index": 0, "row_index": i, "getitem": i + 1,
                "model": 3, "prediction": 4, "metric": 5, "prediction_value": float(i + 1)} for i, row in enumerate(rows)],
            "metric_value": 1.0, "loaded_model_token": 1,
            "loaded_state": {name: {**{key: value for key, value in snap.items() if key not in {"token", "values"}},
                "checkpoint_member": True} for name, snap in {"layer.weight": tensor(20, [[1.0]], [1, 1]), "layer.bias": tensor(21, [0.0], [1])}.items()},
            "framework": {"version": "2.14.0+cpu", "device": "cpu", "package_manifest_sha256": "b" * 64, "package_manifest_sha256_after": "b" * 64},
            "interpreter": {"version": "3.11.14 (mock)", "optimize": 0},
            "entry": {"argv": [config["entry"], *config["args"]], "cwd": config["cwd"], "path": config["cwd"] + "/" + config["entry"], "sha256": file_hashes["eval.py"]},
            "source_hashes_before": {"/app/" + key: value for key, value in launch["inference"]["source_sha256"].items()},
            "source_hashes_after": {"/app/" + key: value for key, value in launch["inference"]["source_sha256"].items()},
            "partition_sha256_after": file_hashes["test.jsonl"], "checkpoint_sha256": file_hashes["weights.pt"],
            "checkpoint_sha256_after": file_hashes["weights.pt"], "sites": [dict(row, code_sha256="c" * 64) for row in config["sites"]]}
        Path(launch["audit_output"]).write_text(json.dumps(report), encoding="utf-8")
        return CommandResult(command, cwd, 0, '{"mse":1.0}\n', "", .01, termination="completed")

    monkeypatch.setattr("llm.client.llm_json", completion)
    monkeypatch.setattr(v2, "run_command", transport)
    saved = {}
    for mode in ("guard", "schema", "healthy"):
        current["mode"] = mode
        before_semantic = calls["semantic"]
        result = v2.execute_plans([plan], [claim], materials, tmp_path / mode,
            config={"max_attempts": 0, "approval_mode": "auto", "training_budget": 0, "docker_options": {"torch_inference": policy}})
        binding = json.loads((tmp_path / mode / "run_0000/attempt_0/inference_binding.json").read_text(encoding="utf-8"))
        assessed = assess_claims(result.claims)
        saved[mode] = {"binding": binding, "assessment": assessed[0].status, "semantic_calls": calls["semantic"] - before_semantic}
        assert [item.model_dump(mode="json") for item in (claim, materials, plan)] == originals
        assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in repo.iterdir()} == file_hashes
        assert result.ledger[0]["attempts"][0]["returncode"] == 0
        if mode != "healthy":
            assert binding == {"status": "unresolved", "reason": "inference_request_binding_unavailable",
                "binding_failure": "actual_source_span_mismatch" if mode == "guard" else "invalid_schema"}
            assert marker not in json.dumps(binding)
            assert result.claims[0].evidence == [] and assessed[0].status == "unverified"
            assert calls["semantic"] == before_semantic
        else:
            assert binding["status"] == "bound" and set(binding) == {"status", "request_sha256", "binding"}
            assert len(result.claims[0].evidence) == 1
            assert result.claims[0].evidence[0].sufficient and result.claims[0].evidence[0].aligned
            assert result.claims[0].evidence[0].covered == ["c1"] and assessed[0].status == "supported"
            assert result.ledger[0]["alignment"][0]["observed"]["value"] == 1
    assert calls == {"refinement": 3, "semantic": 1, "author": 3, "inspect": 2, "external_real": 0}
    (tmp_path / "binding-diagnostics-public-audit.json").write_text(json.dumps({"calls": calls, "modes": saved}), encoding="utf-8")
