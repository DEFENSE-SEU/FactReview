"""One public control of frozen locations and opt-in API diagnostics; no author execution."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from common import run_stats
from fact_generation.execution import inference_callable_catalog as catalog
from fact_generation.execution import inference_runtime
from fact_generation.execution import runner as v2
from fact_generation.execution.inference_proposal_wire import freeze_sources
from fact_generation.execution.inference_source_sites_diagnostics import RuntimeStageDiagnostic
from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_launch import bind_source_sites
from fact_generation.execution.runtime_observer import prepare_site
from fact_generation.execution.runtime_science import read_inference_sources
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials
from schemas.runtime_science import ROLES
from tests.test_torch_inference_execution import SOURCE
from verification.experiment_targets import bind_execution_target


def test_complete_callable_catalog_and_public_diagnostic_seam(tmp_path, monkeypatch):
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(True)
        pytest.fail("Real model/retrieval/network/Docker/process/author execution forbidden")

    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "socket.create_connection", "subprocess.Popen", "subprocess.run", "urllib.request.urlopen"):
        monkeypatch.setattr(name, forbidden)
    monkeypatch.setitem(sys.modules, "docker", SimpleNamespace(from_env=forbidden, DockerClient=forbidden))
    for name in ("run_command", "docker_runner", "docker_ensure_paper_image", "docker_run_paper_image"):
        monkeypatch.setattr(v2, name, forbidden)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    monkeypatch.setattr(v2.time, "monotonic", lambda: 10.0)

    before_path = Path(__file__).resolve().parent / "fixtures/original_sources/before-catalog.py"
    # Keep the frozen source bytes intact while resolving its historical imports.
    from fact_generation.execution import config, outputs

    monkeypatch.setitem(sys.modules, "fact_generation.execution.v2_config", config)
    monkeypatch.setitem(sys.modules, "fact_generation.execution.v2_outputs", outputs)
    spec = importlib.util.spec_from_file_location("fact_generation.execution._catalog_before_v2", before_path)
    before = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, before)
    spec.loader.exec_module(before)
    selected = before if os.environ.get("FACTREVIEW_CATALOG_CONTROL_VARIANT") == "before" else v2
    for name in ("run_command", "docker_runner", "docker_ensure_paper_image", "docker_run_paper_image"):
        monkeypatch.setattr(before, name, forbidden)

    repo = tmp_path / "repo"
    repo.mkdir()
    definition = ("Numeric test contains exactly two held-out IDs a and b, each with one feature and labels 0 and 1.\n"
        "Affine uses weight 1 and bias 0 in one linear layer; it predicts x.\n"
        "MSE is the mean of both squared prediction errors. Use seed 1, CPU evaluation and no preprocessing.")
    quote = "On Numeric test seed 1 sample_count 2, model Affine reports mse 1."
    paper = "惟馃榾 complete paper\n" + quote + "\n" + definition
    source = "# 惟馃榾 complete source\nimport locations\n" + SOURCE
    extras = '''def decorate(fn):
    raise RuntimeError("Author decorator must never execute")
@decorate
def decorated(x):
    return x
def outer():
    def nested(x):
        return x
    return nested
class Local:
    @decorate
    def method(self):
        return 1
async def asynchronous():
    return 1
def generator():
    yield 1
async def async_generator():
    yield 1
anonymous = lambda x: x
if True:
    def conditional():
        return 1
raise RuntimeError("Author module and decorators must never execute")
'''
    configuration = {"dataset": "Numeric", "model": "Affine", "split": "test", "metric": "mse", "seed": 1, "sample_count": 2}
    rows = [{"id": "a", "features": [1.0], "label": 0}, {"id": "b", "features": [2.0], "label": 1}]
    raw_files = {"eval.py": source.replace("\n", "\r\n").encode(),
        "locations.py": extras.replace("\n", "\r\n").encode(), "README.md": definition.encode(),
        "config.json": json.dumps(configuration).encode(), "weights.pt": b"public checkpoint bytes",
        "test.jsonl": ("\n".join(json.dumps(row) for row in rows)+"\n").encode()}
    for name, raw in raw_files.items():
        (repo / name).write_bytes(raw)
    material_dir = tmp_path / "materials"
    material_dir.mkdir()
    markdown = material_dir / "paper.md"
    markdown.write_text(paper, encoding="utf-8")
    condition = Condition(id="c1", dataset="Numeric", metric="mse",
        settings={"model": "Affine", "split": "test", "seed": 1, "sample_count": 2})
    claim = Claim(id="c", text=quote, source_quote=quote, source_block_id="paper",
        loc=ClaimLocation(page=1, char_start=paper.index(quote), char_end=paper.index(quote)+len(quote)),
        conditions=[condition], needs=["Experiments"], source_refs=[ClaimSourceRef(source_block_id="paper",
            source_quote=definition, covered=["c1"], loc=ClaimLocation(page=1, char_start=paper.index(definition), char_end=len(paper)))])
    materials = SharedMaterials(paper_key="numeric", source_pdf=str(markdown), markdown=paper,
        markdown_path=str(markdown), content_list_path=str(material_dir / "content.json"), provider="fixture",
        repository=index_repository(repo), blocks=[MaterialBlock(id="paper", text=paper,
            loc=ClaimLocation(page=1, char_start=0, char_end=len(paper)))])
    task = ExecutionTask(entry_script="eval.py", config="config.json", command=["python", "eval.py"],
        data_paths=["test.jsonl"], weight_paths=["weights.pt"])
    task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"], entry_script="eval.py",
        config="config.json", data_paths=task.data_paths, weight_paths=task.weight_paths)
    plan = ExecutionPlan(id="p", claim_id="c", condition_ids=["c1"], target_conditions=[condition], task=task,
        y_paper={"c1": 1}, run_mode="evaluation", feasibility="ready", priority="high",
        target_bindings={"c1": bind_execution_target(claim, condition,
            {"block_id": "paper", "quote": quote, "token": "1", "value_context": "mse 1"}, materials)})
    policy = {"image_id": "sha256:"+"a"*64, "package_manifest_sha256": "b"*64}
    config = v2.ExecutionConfig(docker_options={"torch_inference": policy})
    paper_sources = [{"id": "paper", "text": paper, "loc": materials.blocks[0].loc.model_dump(mode="json")}]

    def frozen():
        files, _ = read_inference_sources(plan, materials, repo)
        snapshot = freeze_sources(plan=plan, workspace=repo, supplied_files=files,
            paper_sources=paper_sources, materials=materials)
        args = dict(snapshot=snapshot, plan=plan, workspace=repo, supplied_files=files,
            paper_sources=paper_sources, materials=materials)
        return files, snapshot, args

    files, snapshot, args = frozen()
    complete = catalog.build_callable_catalog(**args)
    assert complete["status"] == "available" and complete["reason"] is None
    assert set(complete) == {"version", "status", "locations", "reason"}
    locations = complete["locations"]
    names = {row["qualname"] for row in locations}
    assert {"decorated", "outer.<locals>.nested", "Local.method", "Samples.__getitem__", "Model.forward", "prediction", "metric"} <= names
    assert not names & {"<module>", "Local", "<lambda>", "asynchronous", "generator", "async_generator", "conditional"}
    for row in locations:
        assert set(row) == {"path", "qualname", "firstlineno"}
        bound = prepare_site(str(repo / row["path"]), row["qualname"], row["firstlineno"])
        assert bound["sha256"] == snapshot["snapshot"]["sources"][row["path"]]["sha256"]
        assert bound["sha256"] == hashlib.sha256((repo / row["path"]).read_bytes()).hexdigest()
    decorated = next(row for row in locations if row["qualname"] == "decorated")
    assert decorated["firstlineno"] == extras.splitlines().index("@decorate")+1
    assert not (repo / "__pycache__").exists()
    role_names = ("Samples.__getitem__", "Model.forward", "prediction", "metric")
    sites = [next(row for row in locations if row["qualname"] == name) for name in role_names]
    assert bind_source_sites(sites*3, workspace=repo, supplied_files=files)["status"] == "unresolved"
    nodes = {node.name: node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)}
    proposal = {"version": "torch-cpu-inference-v1", "condition_id": "c1", "checkpoint_path": "weights.pt",
        "state_selector": [], "dataset_path": "test.jsonl", "codec": "indexed-jsonl-numeric-v1",
        "id_key": "id", "features_key": "features", "label_key": "label",
        "sites": {"getitem": 0, "model": 1, "prediction": 2, "metric": 3},
        "sample_id_selector": ["id"], "sample_features_selector": ["features"], "sample_label_selector": ["label"],
        "batch_ids_selector": ["id"], "batch_features_selector": ["features"], "batch_labels_selector": ["label"],
        "prediction_argument": "output", "metric_prediction_argument": "predictions", "metric_label_argument": "labels",
        "raw_value_selector": ["mse"], "prediction_adapter": "scalar_regression", "metric_adapter": "mean_squared_error",
        "dataset_selector": ["dataset"], "model_selector": ["model"], "split_selector": ["split"], "metric_selector": ["metric"],
        "settings_selectors": {key: [key] for key in condition.settings},
        "paper": {role: {"block_id": "paper", "quote": definition, "start": paper.index(definition), "end": len(paper)} for role in ROLES},
        "sources": {}}
    for role in ROLES:
        name = "eval.py" if role in ("model", "metric") else "README.md"
        text = ast.get_source_segment(source, nodes["forward" if role == "model" else "metric"]) if name == "eval.py" else definition
        proposal["sources"][role] = {"path": name, "quote": text, "start": files[name].index(text), "end": files[name].index(text)+len(text)}
    current = {"inference": True, "sites": sites, "mutation": None}
    completions = []

    def completion(prompt, system, cfg, *, module):
        payload = json.loads(prompt)
        completions.append((payload, system))
        assert module == "execution" and cfg is None
        if current["inference"]:
            assert payload["inference_callable_catalog"] == complete
            assert "science_proposal.sites indexes your returned list, never the catalog order" in system
            assert "zero-based Unicode codepoint" in system and "candidate paper leaf has only block_id and quote" in system
        else:
            assert "inference_callable_catalog" not in payload
        if current["mutation"]:
            current["mutation"]()
        return {"command": ["python", "eval.py"], "metric_output": None,
                "source_sites": copy.deepcopy(current["sites"]),
                "science_proposal": copy.deepcopy(proposal) if current["inference"] else None}

    monkeypatch.setattr("llm.client.llm_json", completion)
    originals = [item.model_dump(mode="json") for item in (plan, claim, materials)]
    with monkeypatch.context() as local:
        local.setattr("common.run_stats.stats_path", lambda: None)
        _, _, telemetry = selected._refine(plan, repo, config, claim=claim, materials=materials)
        audit = telemetry["inference_callable_catalog"]
        assert audit["catalog"] == complete and audit["snapshot_digest"] == snapshot["snapshot_digest"]
        assert audit["snapshot_timing"] == "before_completion" and all(audit[k] is False for k in ("scientific_qualification", "alignment", "support"))
        assert telemetry["science_proposal"] == proposal and telemetry["source_sites"]["status"] == "bound"
        assert inference_runtime.bind_inference_proposal(proposal, plan=plan, claim=claim, materials=materials,
            workspace=repo, source_files=files, limits=policy, source_sites=telemetry["source_sites"])["status"] == "bound"
        current["sites"] = None
        assert selected._refine(plan, repo, config, claim=claim, materials=materials)[2]["source_sites"]["status"] == "unresolved"
        current["sites"] = sites
        current["mutation"] = lambda: (repo / "locations.py").write_bytes(raw_files["locations.py"]+b"# changed\n")
        changed = selected._refine(plan, repo, config, claim=claim, materials=materials)[2]
        assert changed["inference_callable_catalog"] == audit and changed["source_sites"]["status"] == "bound"
        with pytest.raises(ValueError, match="inference_source_closure_unavailable_or_changed"):
            inference_runtime.bind_inference_proposal(proposal, plan=plan, claim=claim, materials=materials,
                workspace=repo, source_files=files, limits=policy, source_sites=changed["source_sites"])
        (repo / "locations.py").write_bytes(raw_files["locations.py"])
        current.update(inference=False, sites=None, mutation=None)
        for mode in ("none", "analysis", "noninference", "deterministic"):
            altered = plan.model_copy(update={"run_mode": "analysis"}) if mode == "analysis" else plan
            if mode == "none":
                altered = plan.model_copy(update={"task": task.model_copy(update={"command": []})})
            settings = v2.ExecutionConfig(refine_with_llm=mode != "deterministic")
            kwargs = {} if mode in ("none", "deterministic") else {"claim": claim, "materials": materials}
            pairs = []
            for implementation in (before, selected):
                mark = len(completions)
                result = implementation._refine(altered, repo, settings, **kwargs)
                pairs.append((result, completions[mark:]))
            assert pairs[0] == pairs[1]
    assert [item.model_dump(mode="json") for item in (plan, claim, materials)] == originals

    def unavailable(arguments, reason=None):
        result = catalog.build_callable_catalog(**arguments)
        assert result["status"] == "unavailable" and result["locations"] == []
        assert result["reason"] is not None and (reason is None or result["reason"] == reason)

    bad = copy.deepcopy(snapshot)
    bad["snapshot"]["sources"]["eval.py"]["displayed_text"] += "# prefix\n"
    unavailable({**args, "snapshot": bad}, "snapshot_invalid")
    unavailable({**args, "snapshot": {"status": "unresolved"}}, "precall_snapshot_unavailable")
    unavailable({**args, "supplied_files": {**files, "eval.py": files["eval.py"][:10]}}, "source_revalidation_failed")
    for change in ("missing", "unindexed", "changed", "metadata"):
        saved_index = materials.repository.model_copy(deep=True)
        if change == "missing":
            (repo / "locations.py").unlink()
        elif change == "unindexed":
            materials.repository.files = [row for row in materials.repository.files if row.path != "locations.py"]
        elif change == "changed":
            (repo / "locations.py").write_bytes(raw_files["locations.py"]+b"# changed\n")
        else:
            materials.repository.files[0].line_count = 999
        unavailable(args, "source_revalidation_failed")
        (repo / "locations.py").write_bytes(raw_files["locations.py"])
        materials.repository = saved_index
    with monkeypatch.context() as local:
        local.setattr(catalog, "_compile_source", lambda path: (_ for _ in ()).throw(SyntaxError("public syntax failure")))
        unavailable(args, "source_compile_failed")
    real_compile = catalog._compile_source

    def duplicate(path):
        code, identity = real_compile(path)
        function = next(item for item in code.co_consts if type(item).__name__ == "code")
        return code.replace(co_consts=(*code.co_consts, function)), identity

    with monkeypatch.context() as local:
        local.setattr(catalog, "_compile_source", duplicate)
        unavailable(args, "callable_ambiguity")
    real_prepare = catalog.prepare_site

    def changed_endpoint(*values):
        bound = real_prepare(*values)
        (repo / "README.md").write_bytes(raw_files["README.md"]+b" changed")
        return bound

    with monkeypatch.context() as local:
        local.setattr(catalog, "prepare_site", changed_endpoint)
        unavailable(args, "source_endpoint_changed")
    (repo / "README.md").write_bytes(raw_files["README.md"])
    original_extra = raw_files["locations.py"]
    baseline_count = sum(row["path"] != "locations.py" for row in locations)
    for count in (128-baseline_count, 129-baseline_count):
        (repo / "locations.py").write_text("".join(f"def f{i}():\n    return {i}\n" for i in range(count)))
        materials.repository = index_repository(repo)
        _, _, bounded_args = frozen()
        result = catalog.build_callable_catalog(**bounded_args)
        if count+baseline_count == 128:
            assert result["status"] == "available" and len(result["locations"]) == 128
        else:
            assert result["reason"] == "catalog_count_capacity" and result["locations"] == []
    (repo / "locations.py").write_text("".join(f"def {'x'*350}{i}():\n    return {i}\n" for i in range(100)))
    materials.repository = index_repository(repo)
    unavailable(frozen()[2], "catalog_byte_capacity")
    (repo / "locations.py").write_text("".join(f"def {'中'*90}{i}():\n    return {i}\n" for i in range(100)), encoding="utf-8")
    materials.repository = index_repository(repo)
    unavailable(frozen()[2], "catalog_byte_capacity")
    for text in ("#"+"x"*65536, "def syntax(:\n    pass\n"):
        (repo / "locations.py").write_text(text)
        materials.repository = index_repository(repo)
        unavailable(frozen()[2], "precall_snapshot_unavailable")
    added = []
    for count, size in ((16, 0), (6, 60000)):
        for i in range(count):
            name = f"extra{i}.py"
            added.append(name)
            (repo / name).write_text("#"+"x"*size+"\n")
        (repo / "locations.py").write_text("\n".join(f"import extra{i}" for i in range(count)))
        materials.repository = index_repository(repo)
        unavailable(frozen()[2], "precall_snapshot_unavailable")
        for name in added:
            (repo / name).unlink()
        added.clear()
    (repo / "locations.py").write_bytes(original_extra)
    materials.repository = index_repository(repo)
    assert catalog.build_callable_catalog(**frozen()[2]) == complete

    current.update(inference=True, sites=sites, mutation=None)
    real_refine = selected._refine
    real_inner, real_graph, real_bind = selected._execute_plans, selected._execute_graph, inference_runtime.bind_inference_request
    dispatch = {"inner": [], "graph": [], "bind": []}
    site_choice = {"value": None}

    def refined(*values, **kwargs):
        command, output, record = real_refine(*values, **kwargs)
        record["source_sites"] = copy.deepcopy(site_choice["value"])
        return command, output, record

    def inner(*values, **kwargs):
        if "source_sites_diagnostics" in kwargs:
            assert kwargs["source_sites_diagnostics"] is True
        dispatch["inner"].append(set(kwargs))
        return real_inner(*values, **kwargs)

    def graph(*values, **kwargs):
        if "source_sites_diagnostics" in kwargs:
            assert kwargs["source_sites_diagnostics"] is True
        dispatch["graph"].append(set(kwargs))
        return real_graph(*values, **kwargs)

    def binding(*values, **kwargs):
        if "source_sites_diagnostics" in kwargs:
            assert kwargs["source_sites_diagnostics"] is True
            assert type(kwargs["runtime_stage_diagnostic"]) is RuntimeStageDiagnostic
        dispatch["bind"].append(set(kwargs))
        return real_bind(*values, **kwargs)

    monkeypatch.setattr(selected, "_refine", refined)
    monkeypatch.setattr(selected, "_execute_plans", inner)
    monkeypatch.setattr(selected, "_execute_graph", graph)
    monkeypatch.setattr(inference_runtime, "bind_inference_request", binding)
    filename = "inference_source_sites_diagnostics.json"
    variants = [(None, "source_sites_not_dict"), ({"status": "unresolved"}, "source_sites_status_unbound"),
        ({"status": "bound", "version": "wrong"}, "source_sites_version_unbound"),
        ({"status": "bound", "version": "python-source-sites-v1", "unresolved": ["public gap"]}, "source_sites_unresolved")]
    for index, (sites_value, code) in enumerate(variants):
        site_choice["value"] = sites_value
        output = tmp_path / "operator" / str(index)
        scope = run_stats.run_scope(tmp_path / "operator/shared_stats.json") if index == 0 else contextlib.nullcontext()
        with scope:
            result = selected.execute_plans([plan], [claim], materials, output, source_sites_diagnostics=True,
                config={"max_attempts": 0, "docker_options": {"torch_inference": policy}},
                runner=lambda request: selected.RunOutcome(returncode=1, issue="public mock unobserved"))
        assert result.claims[0].evidence == []
        payload = (output / "run_0000/attempt_0" / filename).read_bytes()
        observation = json.loads(payload)
        assert len(payload) <= 512 and observation["code"] == code
        assert all(value is None or type(value) is bool for value in observation["checks"].values())
        assert dispatch["inner"][-1] == {"config", "runner", "approver", "repairer", "source_sites_diagnostics"}
        assert dispatch["graph"][-1] == {"source_sites_diagnostics"}
        assert dispatch["bind"][-1] == {
            "claim", "materials", "source_sites_diagnostics", "runtime_stage_diagnostic",
        }
        stage_raw = (output / "run_0000/attempt_0/inference_runtime_stage_diagnostics.json").read_bytes()
        assert len(stage_raw) <= 512
        assert json.loads(stage_raw) == {
            "schema": "torch-inference-runtime-stage-v1", "scope": "attempt_program_stages",
            "audit_only": True, "adoption_authorized": False, "phase": "pre_run", "stage": "source_sites",
            "terminal": "rejected", "pre_run_bound": False, "source_sites_reached": True, "source_sites_passed": False,
        }
    for index, choice in enumerate((None, False, 1, "true")):
        output = tmp_path / "operator" / ("default"+str(index))
        options = {} if choice is None else {"source_sites_diagnostics": choice}
        result = selected.execute_plans([plan], [claim], materials, output, **options,
            config={"max_attempts": 0, "docker_options": {"torch_inference": policy}},
            runner=lambda request: selected.RunOutcome(returncode=1, issue="public mock unobserved"))
        assert result.claims[0].evidence == [] and not (output / "run_0000/attempt_0" / filename).exists()
        assert dispatch["inner"][-1] == {"config", "runner", "approver", "repairer"}
        assert dispatch["graph"][-1] == set() and dispatch["bind"][-1] == {"claim", "materials"}
    assert {p.name: p.read_bytes() for p in repo.iterdir()} == raw_files
    assert forbidden_calls == []
