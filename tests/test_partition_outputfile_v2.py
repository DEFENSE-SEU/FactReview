"""Fresh protected JSON output; process/LLM/HTTP boundaries are forbidden."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from fact_generation.execution import runtime_launch as launch_api
from fact_generation.execution import v2
from fact_generation.execution.partition_analysis_runtime import build_partition_runtime_context
from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_observer import _snapshot
from fact_generation.execution.runtime_receipt import _DEFAULT_ENV, read_observer_receipt
from preprocessing.materials import index_repository
from tests.execution_runtime_fixtures import sandbox_argv, sandbox_audit
from tests.test_partition_rank_analysis_v2 import fixture


def test_fresh_metric_file_capture_receipt_and_full_rank_context(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Outputfile controls forbid process, service and author execution")
    for name in ("llm.client.llm_json", "subprocess.Popen", "subprocess.run",
                 "requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send"):
        monkeypatch.setattr(name, forbidden)
    plan, claim, materials, workspace, proposal, source = fixture(tmp_path, [2,2,2,4,4], .4)
    source = source.replace('print(json.dumps(driver()))',
        'with open("metrics/result.json", "w", encoding="utf-8") as stream:\n    json.dump(driver(), stream)\nprint(json.dumps({"ok": True, "out": "metrics/result.json"}))')
    root = Path(materials.repository.root)
    (root/"eval.py").write_text(source, encoding="utf-8")
    materials.repository = index_repository(root)
    plan.task.command = ["python", "eval.py", "--out", "metrics/result.json"]
    plan.task.metric_output = "metrics/result.json"
    plan.task.resource_contract = build_resource_contract(claim, materials, condition_ids=["c1"],
        entry_script="eval.py", config="config.json", data_paths=["data.json"], weight_paths=[])
    shutil.rmtree(workspace)
    run = tmp_path/"run"
    run.mkdir()
    shutil.copytree(root, run/"workspace")
    workspace = run/"workspace"
    manifest = {row.path: row.sha256 for row in materials.repository.files}
    (run/"source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    sites = launch_api.bind_source_sites([{"path":"eval.py", "qualname":"driver", "firstlineno":node.lineno}],
        workspace=workspace, supplied_files={"eval.py":source})
    request = v2.RunRequest(plan=plan, workspace=str(workspace), run_dir=str(run), command=plan.task.command,
        workdir=".", metric_output=plan.task.metric_output, repair_round=0, config=v2.ExecutionConfig(), source_sites=sites)
    attempt, scratch = run/"attempt_0", run/"runtime_scratch"
    attempt.mkdir()
    scratch.mkdir()
    launch = launch_api.prepare_observer_launch(request.command, entry_script="eval.py", workdir=".", workspace=workspace,
        metric_output=request.metric_output, source_sites=sites, trusted_dir=attempt/"observer_trusted", runtime_dir=scratch,
        repair_round=0, runtime_python="3.11")
    assert launch["status"] == "ready", launch
    assert (workspace/"metrics/result.json").read_bytes() == b""
    env = dict(_DEFAULT_ENV, FACTREVIEW_REPAIR_ROUND="0")
    argv = ["docker","run","--rm","--name","mock", "-v",str(workspace)+":/app", "-v",str(scratch)+":/workspace/run_dir", "-w","/app/."]
    for key, value in env.items():
        argv += ["-e",key+"="+value]
    argv += ["image",*request.command]
    protected = launch_api.protect_observer_docker_argv(sandbox_argv(argv, request), launch=launch, sandbox=request.config.sandbox)
    assert protected["status"] == "ready"
    assert launch_api.protect_observer_docker_argv(protected["argv"], launch=launch, sandbox=request.config.sandbox)["argv"] == protected["argv"]
    payload = {"metrics":{"MRR":.4}, "observations":[{"dataset":"Benchmark","metric":"MRR","settings":{"split":"test","score_func":"transe","opn":"sub"},"value":.4}]}
    report = {"version":"python-source-events-v1", "unresolved":[], "execution":{"status":"completed"},
        "interpreter":{"version":"3.11.14 (mock producer)","executable":"/usr/local/bin/python","optimize":0},
        "entry":{"argv":request.command[1:],"cwd":"/app","path":"/app/eval.py","sha256":launch["source_sha256"]["eval.py"]},
        "sites":[dict(row,path="/app/"+row["path"],code_sha256="1"*64) for row in sites["sites"]],
        "events":[{"event":"call","site":0,"invocation":1,"order":1,"parent_invocation":None,"arguments":{}},
                  {"event":"return","site":0,"invocation":1,"order":2,"value":_snapshot(payload,65536)}],
        "source_hashes_after":{"/app/"+key:value for key,value in launch["source_sha256"].items()},
        "scope_limits":["in_process_tampering_not_excluded","events_do_not_prove_scientific_roles","native_and_child_execution_not_covered"]}
    report_raw = json.dumps(report).encode()
    Path(launch["audit_output"]).write_bytes(report_raw)
    (attempt/"runtime_observer.json").write_bytes(report_raw)
    launch.update(status="observed",report_status="completed",report_unresolved=[],report_sha256=hashlib.sha256(report_raw).hexdigest())
    (attempt/"runtime_observer_launch.json").write_text(json.dumps(launch),encoding="utf-8")
    exact = json.dumps(payload, indent=2).encode()
    Path(launch["metric_file"]["capture_path"]).write_bytes(exact)
    raw, capture = launch_api.capture_metric_file_output(launch,metric_output=request.metric_output,workdir=".",repair_round=0,runtime_dir=scratch,attempt_dir=attempt)
    assert raw == exact and Path(capture["record"]["host_copy"]).read_bytes() == exact
    launch_env = dict(launch,capture=capture)
    stdout = '{"ok":true,"out":"metrics/result.json"}\n'
    (attempt/"run_stdout.log").write_bytes(stdout.encode())
    (attempt/"raw_output.json").write_text(json.dumps(payload),encoding="utf-8")
    decoded, mapping = v2.decode_output(payload)
    (attempt/"output_mapping.json").write_text(json.dumps({"source":str(attempt/"raw_output.json"),"selectors":mapping}),encoding="utf-8")
    outcome = v2.RunOutcome(returncode=0,stdout=stdout,commands=[protected["argv"]],observations=[v2.Observation.model_validate(row) for row in decoded],
        environment={"transport":"docker","image":"image","container_name":"mock","runtime_observer":launch_env,"sandbox":sandbox_audit(protected["argv"], request),"process_termination":{"kind":"completed","exception_type":None}},
        logs={"runtime_observer":str(attempt/"runtime_observer.json"),"runtime_observer_launch":str(attempt/"runtime_observer_launch.json"),"stdout":str(attempt/"run_stdout.log"),
              "raw_output":str(attempt/"raw_output.json"),"output_mapping":str(attempt/"output_mapping.json"),"raw_metric_file":capture["record"]["host_copy"],"metric_file_capture":capture["metadata"]["path"]})
    receipt = read_observer_receipt(request,outcome)
    assert receipt["status"] == "received", receipt
    assert receipt["metric_file_output"]["payload"] == payload
    assert not any(receipt[key] for key in ("scientific_qualification","alignment","support"))
    files = {name:(workspace/name).read_text(encoding="utf-8") for name in ("eval.py","data.json","config.json","README.md")}
    before = outcome.model_dump(mode="json")
    context = build_partition_runtime_context({"version":"partition-runtime-v1","analysis":proposal,"raw_value_selector":["observations",0,"value"]},
        plan=plan,claim=claim,materials=materials,request=request,outcome=outcome,source_files=files)
    assert context["actual_observation"]["value"] == .4 and context["actual_observation"]["settings"] == claim.conditions[0].settings
    assert outcome.model_dump(mode="json") == before
    assert context["catalog"]["actual/analysis"]["model_inference_performed"] is False
    changed = outcome.model_copy(deep=True)
    changed.commands[0].append("extra")
    assert read_observer_receipt(request,changed)["status"] == "unresolved"
    original = Path(capture["record"]["host_copy"]).read_bytes()
    Path(capture["record"]["host_copy"]).write_bytes(b'{}')
    assert read_observer_receipt(request,outcome)["status"] == "unresolved"
    Path(capture["record"]["host_copy"]).write_bytes(original)
    (attempt/"raw_output.json").write_text('{}',encoding="utf-8")
    with pytest.raises(ValueError):
        build_partition_runtime_context({"version":"partition-runtime-v1","analysis":proposal,"raw_value_selector":["observations",0,"value"]},
            plan=plan,claim=claim,materials=materials,request=request,outcome=outcome,source_files=files)
    existing = launch_api.prepare_observer_launch(request.command,entry_script="eval.py",workdir=".",workspace=workspace,metric_output=request.metric_output,
        source_sites=sites,trusted_dir=attempt/"unused",runtime_dir=scratch,repair_round=1,runtime_python="3.11")
    assert existing["status"] == "unresolved" and not (attempt/"unused").exists()
    assert launch_api.discard_empty_metric_placeholder(launch,source_manifest={**manifest,"metrics/result.json":"0"*64})["status"] == "unresolved"
    assert (workspace/"metrics/result.json").exists()
    assert launch_api.discard_empty_metric_placeholder(launch,source_manifest=manifest)["status"] == "removed"
    assert not (workspace/"metrics/result.json").exists() and Path(launch["metric_file"]["capture_path"]).read_bytes() == exact
    next_attempt = run/"attempt_1"
    next_attempt.mkdir()
    retry = launch_api.prepare_observer_launch(request.command,entry_script="eval.py",workdir=".",workspace=workspace,metric_output=request.metric_output,
        source_sites=sites,trusted_dir=next_attempt/"observer_trusted",runtime_dir=scratch,repair_round=1,runtime_python="3.11")
    assert retry["status"] == "ready" and retry["original_command"] == request.command
