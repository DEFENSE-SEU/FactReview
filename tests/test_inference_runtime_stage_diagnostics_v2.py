"""One offline stage-control node; scientific guards and graph routing stay real."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from common import run_stats
from fact_generation.execution import (
    inference_runtime,
    inference_source_sites_diagnostics,
    runtime_science,
    v2,
)
from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_launch import bind_source_sites
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials
from tests.test_torch_inference_execution_v2 import SOURCE
from verification.experiment_targets import bind_execution_target


def _public_inputs(tmp_path):
    repository, workspace, material_dir = (tmp_path / name for name in ("repository", "workspace", "materials"))
    for directory in (repository, workspace, material_dir):
        directory.mkdir()
    definition = (
        "Numeric test consists of held-out IDs a and b, with features 1 and 2 and labels 0 and 1. "
        "Affine is one linear layer with released weight 1 and bias 0, predicting x without preprocessing. "
        "MSE averages both squared prediction errors on the complete test split. Seed is 1; CPU evaluation."
    )
    quote = "On Numeric test seed 1 sample_count 2, model Affine reports mse 1."
    paper = quote + "\n" + definition
    configuration = {"dataset": "Numeric", "model": "Affine", "split": "test", "metric": "mse", "seed": 1, "sample_count": 2}
    rows = [{"id": "a", "features": [1.0], "label": 0}, {"id": "b", "features": [2.0], "label": 1}]
    files = {"eval.py": SOURCE.encode(), "README.md": definition.encode(), "config.json": json.dumps(configuration).encode(),
             "weights.pt": b"public mock checkpoint", "test.jsonl": ("\n".join(json.dumps(row) for row in rows) + "\n").encode()}
    for name, raw in files.items():
        for directory in (repository, workspace):
            (directory / name).write_bytes(raw)
    markdown = material_dir / "paper.md"
    markdown.write_text(paper, encoding="utf-8")
    condition = Condition(id="c1", dataset="Numeric", metric="mse",
                          settings={"model": "Affine", "split": "test", "seed": 1, "sample_count": 2})
    claim = Claim(id="c", text=quote, source_quote=quote, source_block_id="paper", needs=["Experiments"],
                  loc=ClaimLocation(page=1, char_start=0, char_end=len(quote)), conditions=[condition],
                  source_refs=[ClaimSourceRef(source_block_id="paper", source_quote=definition, covered=["c1"],
                               loc=ClaimLocation(page=1, char_start=paper.index(definition), char_end=len(paper)))])
    materials = SharedMaterials(paper_key="public-runtime-stage", source_pdf=str(markdown), markdown=paper,
        markdown_path=str(markdown), content_list_path=str(material_dir / "content.json"), provider="fixture",
        repository=index_repository(repository), blocks=[MaterialBlock(id="paper", text=paper,
        loc=ClaimLocation(page=1, char_start=0, char_end=len(paper)))])
    task = ExecutionTask(entry_script="eval.py", config="config.json", command=["python", "eval.py"],
                         data_paths=["test.jsonl"], weight_paths=["weights.pt"])
    task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"], entry_script="eval.py",
        config="config.json", data_paths=task.data_paths, weight_paths=task.weight_paths)
    plan = ExecutionPlan(id="p", claim_id="c", condition_ids=["c1"], target_conditions=[condition], task=task,
        y_paper={"c1": 1}, run_mode="evaluation", feasibility="ready", priority="high", target_bindings={"c1":
        bind_execution_target(claim, condition, {"block_id": "paper", "quote": quote, "token": "1", "value_context": "mse 1"}, materials)})
    nodes = {}
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.FunctionDef):
            nodes[node.name] = node
        elif isinstance(node, ast.ClassDef):
            nodes.update({node.name + "." + child.name: child for child in node.body if isinstance(child, ast.FunctionDef)})
    source_files, scope = runtime_science.read_inference_sources(plan, materials, workspace)
    assert scope["unavailable"] == []
    sites = bind_source_sites([{"path": "eval.py", "qualname": name, "firstlineno": nodes[name].lineno}
        for name in ("Samples.__getitem__", "Model.forward", "prediction", "metric")], workspace=workspace,
        supplied_files=source_files)
    sources = {}
    roles = ("dataset", "split", "model", "metric", "population", "qualifiers")
    for role in roles:
        name = "eval.py" if role in ("model", "metric") else "README.md"
        text = ast.get_source_segment(SOURCE, nodes["Model.forward" if role == "model" else "metric"]) if name == "eval.py" else definition
        sources[role] = {"path": name, "quote": text, "start": source_files[name].index(text), "end": source_files[name].index(text) + len(text)}
    proposal = {"version": "torch-cpu-inference-v1", "condition_id": "c1", "checkpoint_path": "weights.pt",
        "dataset_path": "test.jsonl", "codec": "indexed-jsonl-numeric-v1", "id_key": "id", "features_key": "features", "label_key": "label",
        "sites": {"getitem": 0, "model": 1, "prediction": 2, "metric": 3}, "sample_id_selector": ["id"],
        "sample_features_selector": ["features"], "sample_label_selector": ["label"], "batch_ids_selector": ["id"],
        "batch_features_selector": ["features"], "batch_labels_selector": ["label"], "prediction_argument": "output",
        "metric_prediction_argument": "predictions", "metric_label_argument": "labels", "raw_value_selector": ["mse"],
        "prediction_adapter": "scalar_regression", "metric_adapter": "mean_squared_error", "dataset_selector": ["dataset"],
        "model_selector": ["model"], "split_selector": ["split"], "metric_selector": ["metric"],
        "settings_selectors": {key: [key] for key in condition.settings}, "sources": sources,
        "paper": {role: {"block_id": "paper", "quote": definition, "start": paper.index(definition), "end": len(paper)} for role in roles}}
    config = v2.ExecutionConfig(max_attempts=1, docker_options={"torch_inference": {
        "image_id": "sha256:" + "a" * 64, "package_manifest_sha256": "b" * 64}})
    request = v2.RunRequest(plan=plan, workspace=str(workspace), run_dir=str(tmp_path / "operator"), command=list(task.command),
        workdir=task.workdir, metric_output=None, repair_round=0, config=config, source_sites=sites,
        source_science=proposal, source_science_files=source_files)
    return request, claim, materials, files


def test_runtime_stage_diagnostics_preserve_original_decisions_and_attempts(tmp_path, monkeypatch):
    marker = "PRIVATE_RUNTIME_STAGE_DETAIL_9461"
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(True)
        pytest.fail("Real LLM, retrieval, network, Docker, subprocess and author execution are forbidden")

    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send", "socket.create_connection",
                 "subprocess.Popen", "subprocess.run", "urllib.request.urlopen"):
        monkeypatch.setattr(name, forbidden)
    monkeypatch.setitem(sys.modules, "docker", SimpleNamespace(from_env=forbidden, DockerClient=forbidden))
    for name in ("run_command", "docker_runner", "docker_ensure_paper_image", "docker_run_paper_image"):
        monkeypatch.setattr(v2, name, forbidden)
    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    monkeypatch.setattr(v2.time, "monotonic", lambda: 10.0)
    request, claim, materials, raw_files = _public_inputs(tmp_path)
    original_inputs = [item.model_dump(mode="json") for item in (request, claim, materials)]
    Runtime = inference_source_sites_diagnostics.RuntimeStageDiagnostic
    filename = "inference_runtime_stage_diagnostics.json"
    canonical = dict(mark=Runtime.mark, reject=Runtime.reject, close=Runtime.close)
    original_bind = inference_runtime.bind_inference_request
    original_build = inference_runtime.build_inference_context
    original_flow = inference_runtime.validate_inference_flow
    original_qualify = runtime_science.qualify_consumption
    base_kwargs = dict(claim=claim, materials=materials)

    def selected(name, *, repair_round=0, run_dir=None, **updates):
        directory = run_dir or tmp_path / "operator" / name
        (directory / f"attempt_{repair_round}").mkdir(parents=True, exist_ok=True)
        return request.model_copy(update={"run_dir": str(directory), "repair_round": repair_round, **updates})

    def path_for(current):
        return Path(current.run_dir) / f"attempt_{current.repair_round}" / filename

    def payload(current, phase, stage, terminal, pre_run_bound, reached, passed):
        raw = path_for(current).read_bytes()
        assert len(raw) <= 512 and marker.encode() not in raw
        assert json.loads(raw) == {"schema": "torch-inference-runtime-stage-v1", "scope": "attempt_program_stages",
            "audit_only": True, "adoption_authorized": False, "phase": phase, "stage": stage, "terminal": terminal,
            "pre_run_bound": pre_run_bound, "source_sites_reached": reached, "source_sites_passed": passed}
        assert path_for(current).stat().st_nlink == 1 and not path_for(current).is_symlink()
        return raw

    expected = original_bind(selected("baseline"), **base_kwargs)
    assert expected["status"] == "bound"
    for name, updates, stage, reached, passed, code in (
        ("healthy", {}, "context", True, True, None),
        ("schema", {"source_science": {**request.source_science, "metric_adapter": marker}}, "schema_resources", None, None, "invalid_schema"),
        ("closure", {"source_science_files": {}}, "source_closure", None, None, "inference_source_closure_unavailable_or_changed"),
        ("sites", {"source_sites": None}, "source_sites", True, False, "inference_source_sites_unbound"),
        ("later", {"source_sites": {**request.source_sites, "sites": []}}, "paper_config_partition", True, True, "inference_site_index_unbound"),
    ):
        current = selected(name, **updates)
        before = original_bind(current, **base_kwargs)
        assert not path_for(current).exists()
        collector = Runtime(current, materials)
        result = original_bind(current, **base_kwargs, runtime_stage_diagnostic=collector)
        assert result == before
        assert result["status"] == ("bound" if code is None else "unresolved")
        if code is not None:
            assert result["binding_failure"] == code
        Runtime.close(collector)
        snapshot = payload(current, "pre_run", stage, "passed" if code is None else "rejected", code is None, reached, passed)
        Runtime.mark(collector, phase="semantic", stage="semantic_reply", terminal="qualified")
        Runtime.close(collector)
        assert path_for(current).read_bytes() == snapshot

    # Native context construction rebinds the original resources. Only unavailable
    # receipt/flow leaf observations are mocked; they never create evidence.
    receipt = {"status": "received", "report": {"framework": {}, "loaded_state": {}, "samples": []},
               "producer_code_identity": {}, "evidence_refs": []}
    flow = {"status": "witnessed", "value": 1.0, "sample_count": 2, "evidence_refs": [],
            "scientific_qualification": False, "alignment": False, "support": False}

    def build_kwargs(current):
        return dict(plan=current.plan, claim=claim, materials=materials, request=current,
                    outcome=v2.RunOutcome(returncode=0), source_files=current.source_science_files)

    def healthy_receipt(*args, **kwargs):
        return copy.deepcopy(receipt)

    def healthy_flow(*args, **kwargs):
        return copy.deepcopy(flow)

    with monkeypatch.context() as local:
        local.setattr("fact_generation.execution.runtime_receipt.read_inference_receipt", healthy_receipt)
        local.setattr(inference_runtime, "validate_inference_flow", healthy_flow)
        current = selected("initial_context")
        reference_context = original_build(current.source_science, **build_kwargs(current))
        collector = Runtime(current, materials)
        assert original_build(current.source_science, **build_kwargs(current), runtime_stage_diagnostic=collector) == reference_context
        Runtime.close(collector)
        payload(current, "initial_context", "context", "passed", None, True, True)
        for name, updates, phase, stage, code in (
            ("initial_rebind", {"source_sites": None}, "initial_context", "source_sites", "inference_source_sites_unbound"),
            ("refreshed_rebind", {"source_sites": None}, "refreshed_context", "source_sites", "inference_source_sites_unbound"),
        ):
            current = selected(name, **updates)
            collector = Runtime(current, materials)
            Runtime.mark(collector, phase=phase, stage="entry", terminal="entered")
            with pytest.raises(ValueError, match="^" + code + "$"):
                original_build(current.source_science, **build_kwargs(current), runtime_stage_diagnostic=collector)
            Runtime.close(collector)
            payload(current, phase, stage, "rejected", None, True, False)
        for name, stage in (("receipt_reject", "receipt"), ("flow_reject", "flow"), ("alignment_reject", "alignment")):
            current = selected(name)
            collector = Runtime(current, materials)
            with monkeypatch.context() as case:
                if stage == "receipt":
                    def reject_receipt(*args, **kwargs):
                        raise ValueError(marker)
                    case.setattr("fact_generation.execution.runtime_receipt.read_inference_receipt", reject_receipt)
                elif stage == "flow":
                    case.setattr("fact_generation.execution.runtime_receipt.read_inference_receipt", lambda *a, **k: {"status": "unreceived"})
                    case.setattr(inference_runtime, "validate_inference_flow", original_flow)
                else:
                    case.setattr(inference_runtime, "validate_inference_flow", lambda *a, **k: {**flow, "value": 1.0})
                    case.setattr(v2, "aligned", lambda *a, **k: False)
                with pytest.raises(ValueError):
                    original_build(current.source_science, **build_kwargs(current), runtime_stage_diagnostic=collector)
            Runtime.close(collector)
            payload(current, "initial_context", stage, "rejected", None, True, True)

        # The real reviewer invokes the context builder a second time. A mock
        # response consumes the supplied obligations and records genuine counters.
        for mode in ("qualified", "unresolved", "semantic_failure", "usage_incomplete", "refreshed_failure"):
            current = selected("review_" + mode)
            collector = Runtime(current, materials)
            context_calls = []

            def traced_build(proposal, **kwargs):
                context_calls.append(kwargs)
                if mode == "refreshed_failure" and len(context_calls) == 2:
                    kwargs = {**kwargs, "request": kwargs["request"].model_copy(update={"source_sites": None})}
                return original_build(proposal, **kwargs)

            def completion(prompt, system, cfg, *, module):
                assert module == "execution"
                if mode != "usage_incomplete":
                    run_stats.record_llm_call(module=module, provider="mock", model="runtime-stage-control",
                                             usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
                if mode == "semantic_failure":
                    raise RuntimeError(marker)
                context = json.loads(prompt)["context"]
                return {"version": context["version"], "context_digest": context["context_digest"], "condition_id": "c1",
                    "obligations": [{"id": key, "decision": "unresolved" if mode == "unresolved" else "confirmed",
                                     "source_ids": ids, "rationale": "Public mock source correspondence."}
                                    for key, ids in context["obligations"].items()], "unresolved": []}

            with monkeypatch.context() as case:
                case.setattr(inference_runtime, "build_inference_context", traced_build)
                case.setattr("llm.client.llm_json", completion)
                with run_stats.run_scope(Path(current.run_dir) / "real_mock_usage.json"):
                    result = original_qualify(current.source_science, **build_kwargs(current), runtime_stage_diagnostic=collector)
            Runtime.close(collector)
            assert result["status"] == ("qualified" if mode == "qualified" else "unresolved" if mode == "unresolved" else "failed")
            assert result["scientific_qualification"] is (mode == "qualified")
            assert len(context_calls) == (1 if mode == "semantic_failure" else 2)
            assert all(row.get("runtime_stage_diagnostic") is collector for row in context_calls)
            if mode == "refreshed_failure":
                payload(current, "refreshed_context", "source_sites", "rejected", None, True, False)
            elif mode == "semantic_failure":
                payload(current, "semantic", "semantic_reply", "failed", None, True, True)
            else:
                payload(current, "usage", "usage", result["status"], None, True, True)

    # Canonical calls cannot execute a foreign or instance-overridden observer.
    observer_calls = []

    def override(*args, **kwargs):
        observer_calls.append(True)
        raise RuntimeError(marker)

    class Foreign:
        mark = reject = close = staticmethod(override)

    class Subclass(Runtime):
        mark = reject = close = staticmethod(override)

    current = selected("foreign", source_sites=None)
    for collector in (Foreign(), Subclass(current, materials)):
        observed = original_bind(current, **base_kwargs, runtime_stage_diagnostic=collector)
        assert observed["binding_failure"] == "inference_source_sites_unbound"
        for operation in (Runtime.reject, Runtime.close):
            operation(collector)
        Runtime.mark(collector, phase="pre_run", stage="entry", terminal="entered")
        assert not path_for(current).exists()
    assert observer_calls == []
    concrete = Runtime(selected("slots"), materials)
    assert not hasattr(concrete, "__dict__")
    for name in ("mark", "reject", "close"):
        with pytest.raises(AttributeError):
            setattr(concrete, name, override)

    # Exclusive IO is soft, including failed writes; retries use a new collector.
    current = selected("io_failure")
    collector = Runtime(current, materials)
    Runtime.mark(collector, phase="pre_run", stage="entry", terminal="passed")
    original_open = Path.open
    io_attempts = []

    def broken_open(self, *args, **kwargs):
        if self == path_for(current):
            io_attempts.append(args[0])
            raise OSError(marker)
        return original_open(self, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(Path, "open", broken_open)
        Runtime.close(collector)
        Runtime.close(collector)
    assert io_attempts == ["x"] and not path_for(current).exists()
    for label, directory in (("workspace", Path(request.workspace) / "probe"),
                             ("repository", Path(materials.repository.root) / "probe"),
                             ("materials", Path(materials.source_pdf).parent / "probe")):
        current = selected(label, run_dir=directory)
        collector = Runtime(current, materials)
        Runtime.mark(collector, phase="pre_run", stage="entry", terminal="entered")
        Runtime.close(collector)
        assert not path_for(current).exists()

    # Default routing preserves kwargs, returns and exceptions at each native API.
    default_calls = []
    with monkeypatch.context() as local:
        def proposal_stub(*args, **kwargs):
            default_calls.append(kwargs)
            return {"public": "unchanged"}
        local.setattr(inference_runtime, "bind_inference_proposal", proposal_stub)
        for operation in canonical:
            local.setattr(Runtime, operation, override)
        result = original_bind(request, **base_kwargs)
        assert result["binding"] == {"public": "unchanged"}
        assert all("runtime_stage_diagnostic" not in row for row in default_calls)
        def raising_stub(*args, **kwargs):
            raise RuntimeError(marker)
        local.setattr(inference_runtime, "bind_inference_proposal", raising_stub)
        with pytest.raises(RuntimeError, match=marker):
            original_bind(request, **base_kwargs)
        def unavailable_context(*args, **kwargs):
            default_calls.append(kwargs)
            raise ValueError(marker)
        local.setattr(runtime_science, "_build_review_context", unavailable_context)
        result = original_qualify(request.source_science, **build_kwargs(request))
        assert result["status"] == "unresolved" and result["reason"] == marker
        assert all("runtime_stage_diagnostic" not in row for row in default_calls)
    assert observer_calls == []

    # Run the production graph with mocked external/scientific consumers. No
    # qualification or scientific evidence is supplied by the graph mocks.
    graph_bindings, graph_reviews = [], []

    def graph_bind(current, **kwargs):
        graph_bindings.append((current.repair_round, kwargs))
        return original_bind(current, **kwargs)

    def graph_review(proposal, **kwargs):
        graph_reviews.append(kwargs)
        return {"status": "unresolved", "scientific_qualification": False, "derived_observations": [],
                "reason": "Public mock scientific scope unresolved"}

    manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in raw_files.items()}
    with monkeypatch.context() as local:
        local.setattr(inference_runtime, "bind_inference_request", graph_bind)
        local.setattr(runtime_science, "qualify_consumption", graph_review)
        canonical_calls = []
        for operation, implementation in canonical.items():
            def traced_operation(obj, *, _name=operation, _implementation=implementation, **kwargs):
                canonical_calls.append((_name, obj, kwargs))
                return _implementation(obj, **kwargs)
            local.setattr(Runtime, operation, traced_operation)
        for enabled in (False, 1, True):
            graph_bindings.clear()
            graph_reviews.clear()
            canonical_calls.clear()
            current = selected("graph_" + str(enabled))
            ledger = {"config": current.config.model_dump(mode="json"), "attempts": [], "repairs": [], "alignment": [],
                      "paper_target_validation": [], "resource_validation": [], "operation_failures": [],
                      "resource_origin_sha256": v2._resource_origin(current.plan, materials)}
            runners = []
            def runner(current):
                runners.append(current.model_dump(mode="json"))
                return v2.RunOutcome(returncode=1 if current.repair_round == 0 else 0, stdout="public mock stdout")
            reason = v2._execute_graph(current, claim.model_copy(deep=True), materials, ledger, runner,
                lambda *args: v2.Repair(dependencies=["public-mock-repair"], reason="Public infrastructure repair"),
                [], manifest, {"used": 0}, source_sites_diagnostics=enabled)
            assert reason == "runtime conditions do not uniquely align with every requested claim condition"
            assert len(runners) == len(ledger["attempts"]) == len(graph_bindings) == 2
            assert len(graph_reviews) == 1 and ledger["repairs"][0]["accepted"] is True
            if enabled is True:
                collectors = [row[1]["runtime_stage_diagnostic"] for row in graph_bindings]
                assert type(collectors[0]) is type(collectors[1]) is Runtime and collectors[0] is not collectors[1]
                assert graph_reviews[0]["runtime_stage_diagnostic"] is collectors[1]
                assert [row[1] for row in canonical_calls if row[0] == "close"] == collectors
                for round_number in (0, 1):
                    assert path_for(current.model_copy(update={"repair_round": round_number})).is_file()
            else:
                assert canonical_calls == []
                assert all("runtime_stage_diagnostic" not in row[1] for row in graph_bindings)
                assert "runtime_stage_diagnostic" not in graph_reviews[0]
                assert not path_for(current).exists()
            assert [row["request"] for row in ledger["attempts"]] == runners
    assert observer_calls == forbidden_calls == []
    assert [item.model_dump(mode="json") for item in (request, claim, materials)] == original_inputs
    assert all((Path(request.workspace) / name).read_bytes() == raw for name, raw in raw_files.items())
    assert all((Path(materials.repository.root) / name).read_bytes() == raw for name, raw in raw_files.items())
