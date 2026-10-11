"""One public quote-wire control; original binder stays the scientific authority."""
import ast
import copy
import hashlib
import json

import pytest

from fact_generation.execution import inference_runtime, v2
from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_launch import bind_source_sites
from fact_generation.execution.runtime_science import read_inference_sources
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials
from schemas.runtime_science import ROLES, TorchInferenceProposal
from tests.test_torch_inference_execution_v2 import SOURCE
from verification.experiment_targets import bind_execution_target


def test_quote_only_refinement_preserves_canonical_and_original_inference_guards(tmp_path, monkeypatch):
    calls = {"completion": 0, "external": 0}

    def forbidden(*args, **kwargs):
        calls["external"] += 1
        pytest.fail("Real LLM, retrieval, network, Docker, process and author execution forbidden")

    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "subprocess.Popen", "subprocess.run", "socket.create_connection", "llm.client.llm_json"):
        monkeypatch.setattr(name, forbidden)
    for name in ("docker_runner", "run_command", "docker_ensure_paper_image", "docker_run_paper_image"):
        monkeypatch.setattr(v2, name, forbidden)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    monkeypatch.setattr("common.run_stats.stats_path", lambda: None)

    repo = tmp_path / "repo"
    repo.mkdir()
    definition = ("Numeric test contains exactly two held-out IDs a and b, each with one feature and labels 0 and 1.\n"
        "Affine uses weight 1 and bias 0 in one linear layer; it predicts x.\n"
        "MSE is the mean of both squared prediction errors. Use seed 1, CPU evaluation and no preprocessing.")
    quote = "On Numeric test seed 1 sample_count 2, model Affine reports mse 1."
    paper = "Ω😀 supplied paper\n" + quote + "\n" + definition
    source = "# Ω😀 complete source\n# LF coordinates\n" + SOURCE
    # Source indices are in LF text, despite the original bytes using CRLF and a lone CR.
    raw_source = source.replace("\n", "\r\n", 1).replace("# LF coordinates\n", "# LF coordinates\r")
    raw_source = raw_source.replace("import json\n", "import json\r\n")
    configuration = {"dataset": "Numeric", "model": "Affine", "split": "test", "metric": "mse", "seed": 1, "sample_count": 2}
    rows = [{"id": "a", "features": [1.0], "label": 0}, {"id": "b", "features": [2.0], "label": 1}]
    raw_files = {"eval.py": raw_source.encode(), "README.md": ("Ω😀 source\r\n" + definition.replace("\n", "\r\n")).encode(),
                 "config.json": json.dumps(configuration).encode(), "weights.pt": b"public mock checkpoint bytes",
                 "test.jsonl": ("\n".join(json.dumps(row) for row in rows) + "\n").encode()}
    for name, raw in raw_files.items():
        (repo / name).write_bytes(raw)
    markdown = tmp_path / "paper.md"
    markdown.write_bytes(paper.encode())
    condition = Condition(id="c1", dataset="Numeric", metric="mse",
        settings={"model": "Affine", "split": "test", "seed": 1, "sample_count": 2})
    claim = Claim(id="c", text=quote, source_quote=quote, source_block_id="paper",
        loc=ClaimLocation(page=1, char_start=paper.index(quote), char_end=paper.index(quote)+len(quote)),
        conditions=[condition], needs=["Experiments"], source_refs=[ClaimSourceRef(source_block_id="paper",
            source_quote=definition, covered=["c1"], loc=ClaimLocation(page=1, char_start=paper.index(definition), char_end=len(paper)))])
    materials = SharedMaterials(paper_key="numeric", source_pdf=str(markdown), markdown=paper,
        markdown_path=str(markdown), content_list_path="content.json", provider="fixture", repository=index_repository(repo),
        blocks=[MaterialBlock(id="paper", text=paper, loc=ClaimLocation(page=1, char_start=0, char_end=len(paper)))])
    task = ExecutionTask(entry_script="eval.py", config="config.json", command=["python", "eval.py"],
        metric_output="metrics.json", data_paths=["test.jsonl"], weight_paths=["weights.pt"])
    task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"], entry_script="eval.py",
        config="config.json", data_paths=task.data_paths, weight_paths=task.weight_paths)
    plan = ExecutionPlan(id="p", claim_id="c", condition_ids=["c1"], target_conditions=[condition], task=task,
        y_paper={"c1": 1}, run_mode="evaluation", feasibility="ready", priority="high",
        target_bindings={"c1": bind_execution_target(claim, condition,
            {"block_id": "paper", "quote": quote, "token": "1", "value_context": "mse 1"}, materials)})
    policy = {"image_id": "sha256:" + "a" * 64, "package_manifest_sha256": "b" * 64}
    config = v2.ExecutionConfig(docker_options={"torch_inference": policy}, output_mappings={"p": v2.OutputMapping(metric_paths={"mse": ["mse"]})})
    files, read_scope = read_inference_sources(plan, materials, repo)
    assert read_scope["unavailable"] == [] and files["eval.py"] == source
    nodes = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            nodes[node.name] = node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    nodes[node.name + "." + child.name] = child
    sites = [{"path": "eval.py", "qualname": name, "firstlineno": nodes[name].lineno}
             for name in ("Samples.__getitem__", "Model.forward", "prediction", "metric")]
    original = {"version": "torch-cpu-inference-v1", "condition_id": "c1", "checkpoint_path": "weights.pt",
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
        text = ast.get_source_segment(source, nodes["Model.forward" if role == "model" else "metric"]) if name == "eval.py" else definition
        original["sources"][role] = {"path": name, "quote": text, "start": files[name].index(text), "end": files[name].index(text)+len(text)}
    original = TorchInferenceProposal.model_validate(original).model_dump(mode="json")
    wire = copy.deepcopy(original)
    wire["version"] = "torch-cpu-inference-candidate-v2"
    for group in ("paper", "sources"):
        for row in wire[group].values():
            row.pop("start")
            row.pop("end")
    originals = [item.model_dump(mode="json") for item in (plan, claim, materials)]
    current = {"wire": wire, "mutation": None}

    def completion(prompt, system, cfg, *, module):
        calls["completion"] += 1
        assert module == "execution" and cfg is None
        payload = json.loads(prompt)
        assert payload["science_proposal_schema"]["properties"]["version"]["const"] == "torch-cpu-inference-v1"
        schema = payload["science_proposal_candidate_schema"]
        assert schema["properties"]["version"]["const"] == "torch-cpu-inference-candidate-v2"
        for name, keys in (("CandidatePaperDefinition", {"block_id", "quote"}), ("CandidateSourceDefinition", {"path", "quote"})):
            assert set(schema["$defs"][name]["properties"]) == keys
            assert schema["$defs"][name]["additionalProperties"] is False
        assert payload["science_files"] == files
        for phrase in ("zero-based Unicode codepoint", "block-local", "[start,end)", "decoded block.text",
                       "unique", "paper_claim_sources", "covered", "candidate paper leaf has only block_id and quote"):
            assert phrase in system
        if current["mutation"]:
            current["mutation"]()
        return {"command": ["python", "changed.py"], "metric_output": "changed.json",
                "output_mapping": {"metric_paths": {"mse": ["changed"]}},
                "source_sites": copy.deepcopy(sites), "science_proposal": copy.deepcopy(current["wire"])}

    monkeypatch.setattr("llm.client.llm_json", completion)

    def refine(value=wire, mutation=None):
        current.update(wire=value, mutation=mutation)
        before = calls["completion"]
        command, metric_output, telemetry = v2._refine(plan, repo, config, claim=claim, materials=materials)
        assert calls["completion"] == before + 1
        assert command == plan.task.command and metric_output == "metrics.json"
        assert telemetry["output_mapping"] == config.output_mappings["p"].model_dump()
        assert telemetry["science_proposal_wire"] == value
        record = telemetry["science_proposal_localization"]
        assert record["wire"] == value and record["wire_digest"] == hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        assert all(record[key] is False for key in ("scientific_qualification", "alignment", "support"))
        return telemetry, record

    def bind(proposal, source_files=files, source_sites=None):
        return inference_runtime.bind_inference_proposal(proposal, plan=plan, claim=claim, materials=materials,
            workspace=repo, source_files=source_files, limits=policy,
            source_sites=source_sites or bind_source_sites(sites, workspace=repo, supplied_files=source_files))

    telemetry, record = refine()
    assert record["status"] == "localized" and record["canonical"] == original
    assert len(record["locations"]) == 12 and record["snapshot_timing"] == "before_completion"
    frozen = record["snapshot"]
    assert frozen["status"] == "frozen"
    assert frozen["snapshot"]["sources"]["eval.py"]["sha256"] == hashlib.sha256(raw_files["eval.py"]).hexdigest()
    assert frozen["snapshot"]["sources"]["eval.py"]["displayed_text"] == source
    assert frozen["snapshot"]["paper"]["paper"]["text"] == paper
    for group in ("paper", "sources"):
        for role in ROLES:
            row = record["canonical"][group][role]
            whole = paper if group == "paper" else files[row["path"]]
            assert whole[row["start"]:row["end"]] == row["quote"]
    model_quote = original["sources"]["model"]["quote"]
    assert raw_source.index(model_quote) != source.index(model_quote)
    assert bind(telemetry["science_proposal"], source_sites=telemetry["source_sites"])["status"] == "bound"
    assert [item.model_dump(mode="json") for item in (plan, claim, materials)] == originals

    # Canonical coordinates and validation errors retain their original meaning.
    legacy = copy.deepcopy(original)
    legacy["paper"]["dataset"]["start"] += 1
    telemetry, record = refine(legacy)
    assert record["status"] == "not_candidate" and record["canonical"] is None
    assert telemetry["science_proposal"] == legacy
    with pytest.raises(ValueError, match=r"^paper_source_scope_mismatch$"):
        bind(telemetry["science_proposal"])
    assert refine(original)[0]["science_proposal"] == original

    mutations = [lambda w: w.update(version="unknown"), lambda w: w["paper"].pop("metric"),
        lambda w: w["sources"]["model"].update(verified=True), lambda w: w["paper"]["metric"].update(start=0),
        lambda w: w["sources"]["metric"].update(hash="a"*64), lambda w: w.update(sites={**w["sites"], "model": True}),
        lambda w: w.update(raw_value_selector=[True]), lambda w: w.update(settings_selectors={"seed": [""]}),
        lambda w: w["sources"]["dataset"].update(path="../README.md"),
        lambda w: w["paper"]["dataset"].update(block_id="unsupplied"),
        lambda w: w["sources"]["metric"].update(quote="absent"),
        lambda w: w["paper"]["dataset"].update(quote="Numeric")]
    for mutation in mutations:
        bad = copy.deepcopy(wire)
        mutation(bad)
        telemetry, record = refine(bad)
        assert record["status"] == "unresolved" and record["canonical"] is None
        assert telemetry["science_proposal"] is None
    assert refine(None)[0]["science_proposal"] is None

    # Localization does not authorize a condition or incomplete model/metric body.
    for role, quote_value, reason in (("dataset", paper, "paper_source_scope_mismatch"),
                                    ("model", "return self.layer(features).squeeze(-1)", "complete_actual_model_metric_function_required"),
                                    ("metric", "return numerator / sum(y.numel() for y in labels)", "complete_actual_model_metric_function_required")):
        bad = copy.deepcopy(wire)
        bad["paper" if role == "dataset" else "sources"][role]["quote"] = quote_value
        telemetry, record = refine(bad)
        assert record["status"] == "localized"
        with pytest.raises(ValueError, match="^"+reason+"$"):
            bind(telemetry["science_proposal"])

    # Return-time changes preserve the pre-completion snapshot and never grant canonical.
    for mutation, restore in (
        (lambda: (repo/"README.md").write_bytes(raw_files["README.md"]+b" changed"),
         lambda: (repo/"README.md").write_bytes(raw_files["README.md"])),
        (lambda: setattr(materials.blocks[0], "text", paper+" changed"), lambda: setattr(materials.blocks[0], "text", paper)),
        (lambda: setattr(materials.repository.files[0], "sha256", "0"*64),
         lambda: setattr(materials, "repository", index_repository(repo))),
    ):
        telemetry, record = refine(mutation=mutation)
        assert record["snapshot"] == frozen and record["status"] == "unresolved"
        assert record["canonical"] is None and telemetry["science_proposal"] is None
        restore()

    from fact_generation.execution.inference_proposal_wire import freeze_sources, localize_candidate
    papers = [{"id": b.id, "text": b.text, "loc": b.loc.model_dump(mode="json")} for b in materials.blocks]

    def freeze(supplied=files, paper_sources=papers):
        return freeze_sources(plan=plan, workspace=repo, supplied_files=supplied, paper_sources=paper_sources, materials=materials)

    def local(value=wire, snapshot=frozen, supplied=files, paper_sources=papers):
        return localize_candidate(value, snapshot=snapshot, plan=plan, workspace=repo,
            supplied_files=supplied, paper_sources=paper_sources, materials=materials)

    tampered = copy.deepcopy(frozen)
    tampered["snapshot"]["sources"]["README.md"]["displayed_text"] += " changed"
    assert local(snapshot=tampered)["canonical"] is None
    assert freeze({"eval.py": files["eval.py"]})["status"] == "unresolved"  # No healthy subset.
    assert freeze({**files, **{f"extra{i}.py": "" for i in range(14)}})["status"] == "unresolved"
    assert freeze(paper_sources=[{**papers[0], "text": paper[:20]}])["status"] == "unresolved"
    # Overlapping exact matches are ambiguous even when str.count reports one.
    overlap_text = definition + "\naaa"
    (repo/"README.md").write_bytes(overlap_text.encode())
    materials.repository = index_repository(repo)
    overlap_files, scope = read_inference_sources(plan, materials, repo)
    assert not scope["unavailable"]
    overlap_frozen = freeze(overlap_files)
    overlap = copy.deepcopy(wire)
    overlap["sources"]["dataset"]["quote"] = "aa"
    assert local(overlap, overlap_frozen, overlap_files)["canonical"] is None
    (repo/"README.md").write_bytes(raw_files["README.md"])
    materials.repository = index_repository(repo)
    assert local()["canonical"] == original

    # Complete source capacities keep their original raw byte limits.
    for name, raw in (("README.md", b"x"*65537), ("README.md", b"\xff")):
        (repo/name).write_bytes(raw)
        materials.repository = index_repository(repo)
        assert freeze()["status"] == "unresolved"
        (repo/name).write_bytes(raw_files[name])
    materials.repository = index_repository(repo)
    # The complete closure can use all 16 sources; a seventeenth stays unavailable.
    imports = "".join(f"import extra{i}\n" for i in range(13))
    (repo/"eval.py").write_bytes((imports+source).encode())
    for i in range(13):
        (repo/f"extra{i}.py").write_bytes(b"# public complete source\n")
    materials.repository = index_repository(repo)
    full_files, scope = read_inference_sources(plan, materials, repo)
    assert len(full_files) == 16 and not scope["unavailable"] and freeze(full_files)["status"] == "frozen"
    (repo/"extra13.py").write_bytes(b"# seventeenth source\n")
    (repo/"eval.py").write_bytes((imports+"import extra13\n"+source).encode())
    materials.repository = index_repository(repo)
    partial_files, scope = read_inference_sources(plan, materials, repo)
    assert scope["unavailable"] and freeze(partial_files)["status"] == "unresolved"
    for i in range(14):
        (repo/f"extra{i}.py").unlink()
    imports = "".join(f"import extra{i}\n" for i in range(5))
    (repo/"eval.py").write_bytes((imports+source).encode())
    for i in range(5):
        (repo/f"extra{i}.py").write_bytes(b"#"+b"x"*59998+b"\n")
    materials.repository = index_repository(repo)
    overfull_files, scope = read_inference_sources(plan, materials, repo)
    assert scope["unavailable"] and freeze(overfull_files)["status"] == "unresolved"
    for i in range(5):
        (repo/f"extra{i}.py").unlink()
    (repo/"eval.py").write_bytes(raw_files["eval.py"])
    materials.repository = index_repository(repo)
    assert [item.model_dump(mode="json") for item in (plan, claim, materials)] == originals
    assert all((repo/name).read_bytes() == raw for name, raw in raw_files.items())
    assert local()["canonical"] == original and calls["external"] == 0
