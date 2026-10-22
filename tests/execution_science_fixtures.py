"""Full released-statistics fixtures; services and the runtime producer stay mocked.

These fixtures supply source facts and producer events to the real qualification
and judge. They never return a qualification flag or copy a paper target into the
host's input data or returned measurement.
"""

import ast
import hashlib
import json
from pathlib import Path

from common import run_stats
from fact_generation.execution import runner as v2
from fact_generation.execution.resource_contract import build_resource_contract
from fact_generation.execution.runtime_observer import _snapshot
from preprocessing.materials import index_repository
from schemas.claim import ClaimLocation, ClaimSourceRef
from verification.experiment_targets import bind_execution_target

DOCKER_PRODUCER = v2.docker_runner


def statistics_definition(artifact):
    settings = artifact["settings"]
    rows = artifact["partitions"][settings["split"]]
    return (
        f"{artifact['dataset']} {settings['split']} is the complete released partition of {len(rows)} rows. "
        f"The released {artifact['model']} record contains all predictions or ranks for these rows. "
        "No new inference, training history or population-performance result is claimed. "
        + ("MRR is the arithmetic mean of reciprocal positive integer ranks over every row, with no filtering. "
           if artifact["metric"].lower() == "mrr" else
           "Accuracy is the fraction of all rows whose categorical prediction exactly equals the label. ")
        + "The original configuration settings are " + json.dumps(settings, sort_keys=True) + "."
    )


def released_statistics(values, *, rows, model="ReleasedModel", data_path="statistics.json"):
    """Author complete fixture resources before freezing targets and task identity."""
    plan, claim, materials = values
    condition = claim.conditions[0]
    reported = plan.target_bindings[condition.id].reported
    condition.settings = {**condition.settings, "model": model}
    plan.target_conditions = [condition.model_copy(deep=True)]
    root = Path(materials.repository.root)
    artifact = {"dataset": condition.dataset, "metric": condition.metric, "model": model,
                "settings": dict(condition.settings),
                "partitions": {condition.settings["split"]: rows}}
    path = root / data_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    config_path = plan.task.config or "configuration.json"
    cfg = root / config_path
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(dict(condition.settings)), encoding="utf-8")
    definition = statistics_definition(artifact)
    (root / "README.md").write_text(definition, encoding="utf-8")
    # Retain the original sentence and number in the paper. A separately
    # located scalar sentence supplies the newly explicit released-model role.
    block = next(row for row in materials.blocks if row.id == claim.source_block_id)
    start = len(block.text) + 1
    extras = "".join(f" {key} {value}" for key, value in condition.settings.items()
                     if key not in {"model", "split"})
    statement = (f"On {condition.dataset} {condition.settings['split']}{extras}, "
                 f"model {model} reports {condition.metric} is {reported.token}.")
    block.text += "\n" + statement + "\n" + definition
    block.loc.char_end = (block.loc.char_start or 0) + len(block.text)
    claim.text = claim.source_quote = statement
    claim.loc = ClaimLocation(page=block.loc.page, char_start=(block.loc.char_start or 0)+start,
                              char_end=(block.loc.char_start or 0)+start+len(statement))
    definition_start = start + len(statement) + 1
    claim.source_refs.append(ClaimSourceRef(source_block_id=block.id, source_quote=definition,
        loc=ClaimLocation(page=block.loc.page, char_start=(block.loc.char_start or 0)+definition_start,
                          char_end=block.loc.char_end), covered=[condition.id]))
    materials.markdown = "\n\n".join(row.text for row in materials.blocks)
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    plan.task.config = config_path
    plan.task.data_paths = [data_path]
    plan.task.weight_paths = []
    plan.run_mode = "analysis"
    materials.repository = index_repository(root)
    plan.task.resource_contract = build_resource_contract(claim, materials,
        condition_ids=plan.condition_ids, entry_script=plan.task.entry_script,
        config=config_path, data_paths=[data_path], weight_paths=[])
    reported = reported.model_copy(update={"quote": statement, "value_context": statement})
    plan.target_bindings = {condition.id: bind_execution_target(claim, condition, reported, materials)}
    return artifact, definition


def analysis_source(entry, data_path, split, *, payload_expression, metric="accuracy", file_output=False):
    """Source emitter computes its metric from every released row."""
    operation = ("sum(1 / row['rank'] for row in rows) / len(rows)" if metric.lower() == "mrr"
                 else "sum(row['label'] == row['prediction'] for row in rows) / len(rows)")
    prefix = "../" * (len(Path(entry).parts) - 1)
    source = ("import json\nfrom pathlib import Path\n"
              "def driver():\n"
              f"    data = json.loads((Path(__file__).resolve().parent / {prefix+data_path!r}).read_text(encoding='utf-8'))\n"
              f"    rows = data['partitions'][{split!r}]\n"
              f"    value = {operation}\n"
              f"    return {payload_expression}\n")
    if file_output:
        assert isinstance(file_output, str)
        source += (f"with open({file_output!r}, 'w', encoding='utf-8') as stream:\n"
                   "    json.dump(driver(), stream)\n"
                   f"print(json.dumps({{'ok': True, 'out': {file_output!r}}}))\n")
    else:
        source += "print(json.dumps(driver()))\n"
    return source


def science_model():
    """Mock only refinement and independent semantic decisions, recording usage."""
    calls = []

    def llm(prompt, system, cfg, *, module):
        assert module == "execution"
        payload = json.loads(prompt)
        calls.append(payload)
        run_stats.record_llm_call(module=module, provider="mock", model="source-statistics",
            usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
        if "plan" not in payload:
            context = payload["context"]
            assert context["version"] == "analysis-v1"
            return {"version": "analysis-v1", "context_digest": context["context_digest"],
                    "condition_id": context["condition"]["id"], "measurement_scope": "released_statistics",
                    "obligations": [{"id": key, "decision": "confirmed", "source_ids": value,
                        "rationale": "The complete released rows and explicit located source definitions agree."}
                        for key, value in context["obligations"].items()], "unresolved": []}
        task = payload["plan"]["task"]
        files = payload["science_files"]
        assert payload["science_read_scope"]["unavailable"] == []
        data = json.loads(files[task["data_paths"][0]])
        definition = files["README.md"]
        block = next(row for row in payload["paper_sources"] if definition in row["text"])
        start = block["text"].index(definition)
        roles = ("dataset", "split", "model", "metric", "population", "qualifiers")
        analysis = {"version": "released-partition-analysis-v1", "artifact_path": task["data_paths"][0],
            "partition_selector": ["partitions", data["settings"]["split"]],
            "dataset_selector": ["dataset"], "model_selector": ["model"], "metric_selector": ["metric"],
            "metric_definition": "mean_reciprocal_rank" if data["metric"].lower() == "mrr" else "exact_match_fraction",
            "paper": {role: {"block_id": block["id"], "quote": definition,
                              "start": start, "end": start+len(definition)} for role in roles},
            "sources": {role: {"path": "README.md", "quote": definition, "start": 0,
                                "end": len(definition)} for role in roles},
            "config_settings_selector": {key: [key] for key in data["settings"]
                                         if key not in {"model", "split", "sample_count"}}}
        if data["metric"].lower() == "mrr":
            analysis.update(rank_selector=["rank"], label_selector=None, prediction_selector=None)
        else:
            analysis.update(label_selector=["label"], prediction_selector=["prediction"])
        source = files[task["entry_script"]]
        node = next(row for row in ast.parse(source).body if isinstance(row, ast.FunctionDef))
        mapping = payload["plan"]["task"].get("output_mapping")
        if data["metric"].lower() == "mrr":
            selector = ["mrr"] if task.get("metric_output") else ["observations", 0, "value"]
        elif "measurements" in source:
            selector = ["measurements", "accuracy"]
        else:
            selector = ["observations", 0, "value"]
        return {"command": task["command"] or ["python", task["entry_script"]], "metric_output": task.get("metric_output"),
                "source_sites": [{"path": task["entry_script"], "qualname": "driver", "firstlineno": node.lineno}],
                "science_proposal": {"version": "partition-runtime-v1", "analysis": analysis,
                                     "raw_value_selector": selector},
                **({"output_mapping": mapping} if mapping else {})}

    llm.calls = calls
    return llm


def install_science_model(monkeypatch):
    llm = science_model()
    monkeypatch.setattr("llm.client.llm_json", llm)
    monkeypatch.setattr("llm.client.resolve_llm_config", lambda: None)
    return llm.calls


def record_observer(command, cwd, payload):
    """The mocked Docker boundary writes its actual selected producer events."""
    token = next(value for value in command if value.startswith("FACTREVIEW_REPAIR_ROUND="))
    attempt = Path(cwd) / f"attempt_{token.partition('=')[2]}"
    launch = json.loads((attempt / "runtime_observer_launch.json").read_text(encoding="utf-8"))
    assert launch["status"] == "ready", launch
    config = json.loads((Path(launch["trusted_dir"]) / "config.json").read_text(encoding="utf-8"))
    entry = next(name for name in launch["source_sha256"] if name.endswith(config["entry"]))
    report = {"version": "python-source-events-v1", "unresolved": [],
        "execution": {"status": "completed"},
        "interpreter": {"version": "3.11.14 (mock producer)", "executable": "/usr/local/bin/python", "optimize": 0},
        "entry": {"argv": [config["entry"], *config["args"]], "cwd": config["cwd"],
                  "path": "/app/"+entry, "sha256": launch["source_sha256"][entry]},
        "sites": [dict(row, code_sha256=hashlib.sha256((row["qualname"]+str(index)).encode()).hexdigest())
                  for index, row in enumerate(config["sites"])],
        "events": [{"event": "call", "site": 0, "invocation": 1, "order": 1,
                    "parent_invocation": None, "arguments": {}},
                   {"event": "return", "site": 0, "invocation": 1, "order": 2,
                    "value": _snapshot(payload, 65536)}],
        "source_hashes_after": {"/app/"+key: value for key, value in launch["source_sha256"].items()},
        "scope_limits": ["in_process_tampering_not_excluded", "events_do_not_prove_scientific_roles",
                         "native_and_child_execution_not_covered"]}
    Path(launch["audit_output"]).write_text(json.dumps(report), encoding="utf-8")


def record_metric_file(command, relative, payload):
    """Write through the mocked transport's sole protected author-output mount."""
    suffix = f":/app/{relative}:rw"
    mounts = [command[index+1] for index, token in enumerate(command[:-1]) if token == "-v"]
    selected = [mount for mount in mounts if mount.endswith(suffix)]
    assert len(selected) == 1
    target = Path(selected[0].removesuffix(suffix))
    assert target.is_file()
    target.write_text(json.dumps(payload), encoding="utf-8")


def finish_source(values, source):
    """Bind freshly authored source only after its complete content is fixed."""
    plan, claim, materials = values
    path = Path(materials.repository.root) / plan.task.entry_script
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    materials.repository = index_repository(Path(materials.repository.root))
    plan.task.resource_contract = build_resource_contract(claim, materials,
        condition_ids=plan.condition_ids, entry_script=plan.task.entry_script,
        config=plan.task.config, data_paths=plan.task.data_paths, weight_paths=[])


def accuracy_fixture(values, monkeypatch):
    """Nine correct categorical predictions and one error, independent of y_paper."""
    plan, claim, _materials = values
    rows = [{"label": 1, "prediction": 1} for _ in range(9)] + [{"label": 0, "prediction": 1}]
    artifact, _definition = released_statistics(values, rows=rows)
    payload = {"observations": [{"dataset": artifact["dataset"], "metric": artifact["metric"],
                                "settings": artifact["settings"], "value": .9}]}
    expression = "{'observations': [{'dataset': data['dataset'], 'metric': data['metric'], 'settings': data['settings'], 'value': value}]}"
    source = analysis_source(plan.task.entry_script, plan.task.data_paths[0], claim.conditions[0].settings["split"],
                             payload_expression=expression, file_output=plan.task.metric_output or False)
    finish_source(values, source)
    install_science_model(monkeypatch)
    monkeypatch.setattr("fact_generation.execution.tools.docker._docker_proxy_env", lambda _: {})
    return payload
