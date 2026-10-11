"""One public CPU/numeric inference protocol control; no installed Torch required.

Only external completion/image/process boundaries supply mock observations.
Author source is never executed. Production receipt, host measurement and judge run.
"""
import ast
import hashlib
import json
import struct
from pathlib import Path

import pytest

from common import run_stats
from fact_generation.execution import v2
from fact_generation.execution.resource_contract import build_resource_contract
from preprocessing.materials import index_repository
from schemas.claim import Claim, ClaimLocation, ClaimSourceRef, Condition, ExecutionPlan, ExecutionTask
from schemas.materials import MaterialBlock, SharedMaterials
from util.subprocess_runner import CommandResult
from verification.experiment_targets import bind_execution_target

SOURCE = '''import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
class Samples(Dataset):
    def __init__(self):
        with open('test.jsonl', encoding='utf-8') as stream:
            self.rows = [json.loads(line) for line in stream]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return {'id': row['id'], 'features': torch.tensor(row['features'], dtype=torch.float64), 'label': row['label']}
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1).double()
    def forward(self, features):
        return self.layer(features).squeeze(-1)
def prediction(output):
    return output
def metric(predictions, labels):
    numerator = sum(((p-y)**2).sum().item() for p,y in zip(predictions, labels))
    return numerator / sum(y.numel() for y in labels)
with open('config.json', encoding='utf-8') as stream:
    config = json.load(stream)
torch.manual_seed(config['seed'])
model = Model()
model.load_state_dict(torch.load('weights.pt', weights_only=True, map_location='cpu'))
model.eval()
predictions, labels = [], []
with torch.no_grad():
    for batch in DataLoader(Samples(), batch_size=2, shuffle=False, num_workers=0):
        predictions.append(prediction(model(batch['features'])))
        labels.append(batch['label'])
print(json.dumps({'mse': metric(predictions, labels)}))
'''


def test_full_cpu_inference_requires_actual_ids_loaded_state_and_whole_metric(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External network, SDK, Docker and subprocess calls are forbidden")
    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "subprocess.Popen", "subprocess.run"):
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
    configuration = {"dataset":"Numeric", "model":"Affine", "split":"test", "metric":"mse", "seed":1, "sample_count":2}
    rows = [{"id":"a", "features":[1.0], "label":0}, {"id":"b", "features":[2.0], "label":1}]
    for name, raw in {"eval.py":SOURCE.encode(), "README.md":definition.encode(),
        "config.json":json.dumps(configuration).encode(), "weights.pt":b"mock actual binary checkpoint boundary",
        "test.jsonl":("\n".join(json.dumps(row) for row in rows)+"\n").encode()}.items():
        (repo / name).write_bytes(raw)
    md = tmp_path / "paper.md"
    md.write_text(paper, encoding="utf-8")
    condition = Condition(id="c1", dataset="Numeric", metric="mse",
        settings={"model":"Affine", "split":"test", "seed":1, "sample_count":2})
    claim = Claim(id="c", text=quote, source_quote=quote, source_block_id="paper",
        loc=ClaimLocation(page=1,char_start=0,char_end=len(quote)), conditions=[condition], needs=["Experiments"],
        source_refs=[ClaimSourceRef(source_block_id="paper",source_quote=definition,covered=["c1"],
            loc=ClaimLocation(page=1,char_start=paper.index(definition),char_end=len(paper)))])
    materials = SharedMaterials(paper_key="numeric", source_pdf=str(md), markdown=paper, markdown_path=str(md),
        content_list_path="content.json", provider="fixture", repository=index_repository(repo),
        blocks=[MaterialBlock(id="paper",text=paper,loc=ClaimLocation(page=1,char_start=0,char_end=len(paper)))])
    task = ExecutionTask(entry_script="eval.py", config="config.json", command=["python","eval.py"],
        data_paths=["test.jsonl"], weight_paths=["weights.pt"])
    task.resource_contract = build_resource_contract(claim,materials,condition_ids=["c1"],entry_script="eval.py",
        config="config.json",data_paths=task.data_paths,weight_paths=task.weight_paths)
    plan = ExecutionPlan(id="p",claim_id="c",condition_ids=["c1"],target_conditions=[condition],task=task,
        y_paper={"c1":1},run_mode="evaluation",feasibility="ready",priority="high",
        target_bindings={"c1":bind_execution_target(claim,condition,
            {"block_id":"paper","quote":quote,"token":"1","value_context":"mse 1"},materials)})
    originals = (claim.model_dump(mode="json"),materials.model_dump(mode="json"),plan.model_dump(mode="json"))
    original_files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.iterdir()}
    policy = {"image_id":"sha256:"+"a"*64,"package_manifest_sha256":"b"*64}
    current = {"mode":"healthy"}
    source = SOURCE
    semantic = []
    roles = ("dataset","split","model","metric","population","qualifiers")
    nodes = {"prediction":None,"metric":None,"Samples.__getitem__":None,"Model.forward":None}
    for node in ast.parse(SOURCE).body:
        if isinstance(node,ast.FunctionDef) and node.name in nodes:
            nodes[node.name] = node
        if isinstance(node,ast.ClassDef):
            for child in node.body:
                if node.name+"."+getattr(child,"name","") in nodes:
                    nodes[node.name+"."+child.name] = child

    def model(prompt, system, cfg, *, module):
        assert module == "execution"
        payload = json.loads(prompt)
        run_stats.record_llm_call(module=module,provider="mock",model="cpu-protocol",
            usage={"input_tokens":10,"output_tokens":4,"total_tokens":14})
        if "plan" in payload:
            assert payload["scientific_claim"]["text"] == quote
            assert payload["science_proposal_schema"]["properties"]["version"]["const"] == "torch-cpu-inference-v1"
            assert "weights.pt" not in payload["science_files"]
            assert payload["science_read_scope"]["selected_resource_metadata"]["weights.pt"]["sha256"] == original_files["weights.pt"]
            sources = {}
            source_nodes = {}
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.FunctionDef) and node.name in {"forward","metric"}:
                    source_nodes[node.name] = node
            for role in roles:
                path = "eval.py" if role in ("model","metric") else "README.md"
                text = ast.get_source_segment(source,source_nodes["forward" if role=="model" else "metric"]) if path=="eval.py" else definition
                whole = payload["science_files"][path]
                sources[role] = {"path":path,"quote":text,"start":whole.index(text),"end":whole.index(text)+len(text)}
            proposal = {"version":"torch-cpu-inference-v1","condition_id":"c1","checkpoint_path":"weights.pt",
                "dataset_path":"test.jsonl","codec":"indexed-jsonl-numeric-v1","id_key":"id","features_key":"features","label_key":"label",
                "sites":{"getitem":0,"model":1,"prediction":2,"metric":3},
                "sample_id_selector":["id"],"sample_features_selector":["features"],"sample_label_selector":["label"],
                "batch_ids_selector":["id"],"batch_features_selector":["features"],"batch_labels_selector":["label"],
                "prediction_argument":"output","metric_prediction_argument":"predictions","metric_label_argument":"labels",
                "raw_value_selector":["mse"],"prediction_adapter":"scalar_regression","metric_adapter":"mean_squared_error",
                "dataset_selector":["dataset"],"model_selector":["model"],"split_selector":["split"],"metric_selector":["metric"],
                "settings_selectors":{key:[key] for key in condition.settings},"sources":sources,
                "paper":{role:{"block_id":"paper","quote":definition,"start":paper.index(definition),"end":len(paper)} for role in roles}}
            return {"command":["python","eval.py"],"metric_output":None,"science_proposal":proposal,
                "source_sites":[{"path":"eval.py","qualname":name,"firstlineno":nodes[name].lineno}
                    for name in ("Samples.__getitem__","Model.forward","prediction","metric")]}
        context = payload["context"]
        semantic.append(context)
        assert context["catalog"]["actual/inference"]["complete_sample_count"] == 2
        assert context["scientific_claim"]["conditions"] == originals[0]["conditions"]
        return {"version":context["version"],"context_digest":context["context_digest"],"condition_id":"c1",
            "obligations":[{"id":key,"decision":"confirmed","source_ids":sources,"rationale":"Complete selected original sources."}
                for key,sources in context["obligations"].items()],"unresolved":[]}

    def tensor(token, values, shape, *, dtype="torch.float64"):
        flat = values if len(shape)==1 else [item for row in values for item in row]
        raw = b"".join(struct.pack("<q" if dtype=="torch.int64" else "<d",item) for item in flat)
        return {"token":token,"values":values,"shape":shape,"dtype":dtype,
                "sha256":hashlib.sha256(raw).hexdigest(),"tensor_version":0}

    def transport(command, cwd, timeout_sec):
        if command[:3] == ["docker","image","inspect"]:
            assert timeout_sec == 30
            return CommandResult(command,cwd,0,policy["image_id"]+"\n","",.01,termination="completed")
        assert command[:2] == ["docker","run"]
        assert policy["image_id"] in command
        launch = json.loads((Path(cwd)/"attempt_0/runtime_observer_launch.json").read_text(encoding="utf-8"))
        if current["mode"] == "compiled":
            assert launch["status"] == "unresolved"
            assert command[-2:] == ["python","eval.py"]
            assert "/factreview-observer/observer.py" not in command
            binding = json.loads((Path(cwd)/"attempt_0/inference_binding.json").read_text(encoding="utf-8"))
            assert binding["status"] == "unresolved"
            current["compiled_author_runs"] = current.get("compiled_author_runs",0)+1
            return CommandResult(command,cwd,0,'{"mse":1.0}\n',"",.01,termination="completed")
        assert launch["version"] == "readonly-torch-inference-v1"
        assert launch["mounts"]["/app"]["mode"] == "ro"
        config = json.loads((Path(launch["trusted_dir"])/"config.json").read_text(encoding="utf-8"))
        features, labels, output = tensor(3,[[1.0],[2.0]],[2,1]),tensor(4,[0,1],[2],dtype="torch.int64"),tensor(5,[1.0,2.0],[2])
        calls = [{"role":"getitem","site":0,"invocation":i+1,"order":i+1,"parent_invocation":None,
            "ordinal":i,"id":row["id"],"feature_snapshot":tensor(i+10,row["features"],[1]),
            "features":row["features"],"label":row["label"],"label_batch_token":4} for i,row in enumerate(rows)]
        calls += [{"role":"model","site":1,"invocation":3,"order":3,"parent_invocation":None,"model_token":1,
            "torch_initial_seed":1,"batch_features":features,"batch_labels":labels,"batch_ids":["a","b"],
            "output_token":5,"output":output,"module_call_completed":True},
            {"role":"prediction","site":2,"invocation":4,"order":4,"parent_invocation":None,
                "input_token":5,"input":output,"output_token":5,"output":output,"model_invocation":3},
            {"role":"metric","site":3,"invocation":5,"order":5,"parent_invocation":None,
                "predictions":[output],"labels":[labels],"value":1.0}]
        samples = [{"ordinal":i,**row,"batch_index":0,"row_index":i,"getitem":i+1,"model":3,"prediction":4,"metric":5,
            "prediction_value":float(i+1)} for i,row in enumerate(rows)]
        if current["mode"] == "same-buffer":
            output = tensor(5,[1.0],[1])
            calls, samples = [], []
            metric_labels = []
            for i,item in enumerate(rows):
                getitem, model_call, predict = 1+3*i, 2+3*i, 3+3*i
                features = tensor(30+i,[item["features"]],[1,1])
                labels = tensor(40+i,[item["label"]],[1],dtype="torch.int64")
                metric_labels.append(labels)
                calls += [{"role":"getitem","site":0,"invocation":getitem,"order":getitem,"parent_invocation":None,
                    "ordinal":i,"id":item["id"],"feature_snapshot":tensor(i+10,item["features"],[1]),
                    "features":item["features"],"label":item["label"],"label_batch_token":40+i},
                    {"role":"model","site":1,"invocation":model_call,"order":model_call,"parent_invocation":None,
                        "model_token":1,"torch_initial_seed":1,"batch_features":features,"batch_labels":labels,
                        "batch_ids":[item["id"]],"output_token":5,"output":output,"module_call_completed":True},
                    {"role":"prediction","site":2,"invocation":predict,"order":predict,"parent_invocation":None,
                        "input_token":5,"input":output,"output_token":5,"output":output,"model_invocation":model_call}]
                samples.append({"ordinal":i,**item,"batch_index":i,"row_index":0,"getitem":getitem,
                    "model":model_call,"prediction":predict,"metric":7,"prediction_value":1.0})
            calls.append({"role":"metric","site":3,"invocation":7,"order":7,"parent_invocation":None,
                "predictions":[output,output],"labels":metric_labels,"value":1.0})
            assert samples[0]["model"] != samples[1]["model"]
            assert calls[1]["output"] == calls[4]["output"]
        report = {"version":"torch-cpu-inference-v1","binding_digest":launch["inference"]["binding_digest"],
            "unresolved":[],"execution":{"status":"completed"},"calls":calls,"samples":samples,"metric_value":1.0,
            "loaded_model_token":1,"loaded_state":{name:{**{k:v for k,v in snapshot.items() if k not in {"token","values"}},
                "checkpoint_member":True} for name,snapshot in {
                    "layer.weight":tensor(20,[[1.0]],[1,1]),"layer.bias":tensor(21,[0.0],[1])}.items()},
            "framework":{"version":"2.14.0+cpu","device":"cpu",
                "package_manifest_sha256":"b"*64,"package_manifest_sha256_after":"b"*64},
            "interpreter":{"version":"3.11.14 (mock)","optimize":0},
            "entry":{"argv":[config["entry"],*config["args"]],"cwd":config["cwd"],"path":config["cwd"]+"/"+config["entry"],
                "sha256":original_files["eval.py"]},"source_hashes_before":{"/app/"+k:v for k,v in launch["inference"]["source_sha256"].items()},
            "source_hashes_after":{"/app/"+k:v for k,v in launch["inference"]["source_sha256"].items()},
            "partition_sha256_after":original_files["test.jsonl"],"checkpoint_sha256":original_files["weights.pt"],
            "checkpoint_sha256_after":original_files["weights.pt"],
            "sites":[dict(row,code_sha256="c"*64) for row in config["sites"]]}
        if current["mode"] == "same-buffer":
            report["loaded_state"] = {"constant":{**{key:value for key,value in output.items()
                if key not in {"token","values"}},"checkpoint_member":True}}
        if current["mode"]=="unknown-state":
            report["unresolved"]=["checkpoint_loaded_state_mismatch"]
        if current["mode"]=="wrong-id":
            samples[1]["id"]="different-official-sample"
        Path(launch["audit_output"]).write_text(json.dumps(report),encoding="utf-8")
        return CommandResult(command,cwd,0,'{"mse":1.0}\n',"",.01,termination="completed")

    monkeypatch.setattr("llm.client.llm_json",model)
    monkeypatch.setattr(v2,"run_command",transport)
    results = {}
    for mode in ("healthy","unknown-state","wrong-id","compiled","same-buffer"):
        current["mode"]=mode
        if mode == "compiled":
            source = SOURCE.replace("return self.layer(features).squeeze(-1)",
                "return torch._C.CompilationUnit('def f(x: Tensor) -> Tensor:\\n    return x\\n').f(features).squeeze(-1)")
            (repo/"eval.py").write_text(source,encoding="utf-8")
            materials.repository = index_repository(repo)
            plan.task.resource_contract = build_resource_contract(claim,materials,condition_ids=["c1"],entry_script="eval.py",
                config="config.json",data_paths=task.data_paths,weight_paths=task.weight_paths)
            originals = (claim.model_dump(mode="json"),materials.model_dump(mode="json"),plan.model_dump(mode="json"))
            original_files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.iterdir()}
        if mode == "same-buffer":
            assert len(semantic) == 1
            source = SOURCE.replace("self.layer = nn.Linear(1, 1).double()",
                "self.register_buffer('constant', torch.ones(1, dtype=torch.float64))").replace(
                    "return self.layer(features).squeeze(-1)","return self.constant").replace("batch_size=2","batch_size=1")
            definition = definition.replace("labels 0 and 1","labels 0 and 2").replace(
                "Affine uses the released state weight 1 and bias 0 in one linear layer; it predicts x.",
                "Affine loads the released registered constant buffer 1; it predicts 1 for both samples.")
            rows[1]["label"] = 2
            paper = quote+"\n"+definition
            for name,raw in {"eval.py":source,"README.md":definition,
                    "test.jsonl":"\n".join(json.dumps(row) for row in rows)+"\n"}.items():
                (repo/name).write_text(raw,encoding="utf-8")
            md.write_text(paper,encoding="utf-8")
            claim.source_refs[0].source_quote = definition
            claim.source_refs[0].loc.char_end = len(paper)
            materials.markdown = materials.blocks[0].text = paper
            materials.blocks[0].loc.char_end = len(paper)
            materials.repository = index_repository(repo)
            plan.task.resource_contract = build_resource_contract(claim,materials,condition_ids=["c1"],entry_script="eval.py",
                config="config.json",data_paths=task.data_paths,weight_paths=task.weight_paths)
            plan.target_bindings = {"c1":bind_execution_target(claim,condition,
                {"block_id":"paper","quote":quote,"token":"1","value_context":"mse 1"},materials)}
            originals = (claim.model_dump(mode="json"),materials.model_dump(mode="json"),plan.model_dump(mode="json"))
            original_files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.iterdir()}
        results[mode]=v2.execute_plans([plan],[claim],materials,tmp_path/mode,
            config={"max_attempts":0,"approval_mode":"auto","training_budget":0,
                "docker_options":{"torch_inference":policy}})
        assert (claim.model_dump(mode="json"),materials.model_dump(mode="json"),plan.model_dump(mode="json")) == originals
        assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in repo.iterdir()} == original_files
        assert results[mode].ledger[0]["attempts"][0]["returncode"] == 0
    assert len(results["healthy"].claims[0].evidence) == 1
    evidence = results["healthy"].claims[0].evidence[0]
    assert evidence.sufficient and evidence.aligned and evidence.covered == ["c1"]
    assert results["healthy"].ledger[0]["alignment"][0]["observed"]["value"] == 1
    assert len(results["same-buffer"].claims[0].evidence) == 1
    assert results["same-buffer"].claims[0].evidence[0].sufficient
    assert len(semantic) == 2
    assert current["compiled_author_runs"] == 1
    assert results["compiled"].claims[0].evidence == []
    compiled_ledger = results["compiled"].ledger[0]
    assert "eval.py" in compiled_ledger["refinement"]["science_read_scope"]["unavailable"]
    compiled_audit = json.loads(Path(compiled_ledger["attempts"][0]["logs"]["scientific_consumption"]).read_text(encoding="utf-8"))
    assert not compiled_audit["scientific_qualification"] and compiled_audit["model_calls"] == 0
    for mode in ("unknown-state","wrong-id"):
        assert results[mode].claims[0].evidence == []
        audit = json.loads(Path(results[mode].ledger[0]["attempts"][0]["logs"]["scientific_consumption"]).read_text(encoding="utf-8"))
        assert not audit["scientific_qualification"] and audit["model_calls"] == 0
    (tmp_path/"public-cpu-inference-audit.json").write_text(json.dumps({"modes":list(results),
        "semantic_calls":len(semantic),"external_real_calls":0,"installed_torch_executed":False}),encoding="utf-8")
