"""Required operational facts, exact raw locator and deterministic PDF navigation."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from review.report import v2
from schemas.review import FinalReview


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Ledger presentation must remain offline")
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("llm.client.llm_json", forbidden)
    monkeypatch.setattr("review.report.advice.llm_json", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)


def fixture():
    binding = {"version": 2, "condition_id": "c1", "value": .75, "unit": "fraction",
               "projection": {"runtime_target": {"model": "M", "split": "test"},
                              "registry_snapshot": {"catalog": "MACHINE_PAYLOAD " * 40},
                              "scope_review": {"reason": "Exact source and metric scope retained"}}}
    attempt = {"request": {"plan": {"target_bindings": {"c1": copy.deepcopy(binding)}},
                           "command": ["python", "eval.py", "--seed", "42"],
                           "config": {"max_attempts": 3}, "workdir": "original_workdir"},
               "commands": [["docker", "inspect", "original_image"]], "returncode": 0,
               "environment": {"image": "original_image", "python": "3.11"},
               "logs": {"stdout": "original_stdout.log"}, "stdout": "STDOUT_FULL",
               "stderr": "STDERR_FULL", "runtime_seconds": 2.5, "tokens": 17,
               "observations": [{"value": .75, "metric": "accuracy", "sample_count": 4}]}
    return FinalReview(paper_key="ledger", run_id="offline", execution_requested=True,
        ledger=[{"plan": {"id": "p1", "target_bindings": {"c1": binding}},
                 "approval_mode": "auto", "approved": True, "training_budget": 0,
                 "attempts": [attempt], "alignment": [{"condition_id": "c1", "aligned": True,
                     "expected": .75, "observed": .75, "gap": 0, "tolerance": .02,
                     "consistent": True, "paper_target_binding": copy.deepcopy(binding),
                     "resource_mode": "released_predictions", "model_inference_performed": False}],
                 "paper_target_validation": [{"phase": "before_run", "verified": True,
                                               "bindings": {"c1": copy.deepcopy(binding)}}],
                 "repairs": [{"round": 1, "accepted": True, "diff": "REPAIR_DIFF_FULL"}],
                 "unknown_field": {"future_contract": "UNKNOWN_FULL"}}])


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def resolve(value, pointer):
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def test_complete_facts_exact_locators_and_full_contract(tmp_path):
    record = fixture()
    original = record.model_dump(mode="json")
    output = v2.write_review(record, tmp_path / "layered", presentation="layered", render_pdf=False)
    saved = json.loads(Path(output["json"]).read_text("utf-8"))
    appendix = Path(output["appendix_markdown"]).read_text("utf-8")
    main = Path(output["markdown"]).read_text("utf-8")
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    assert saved["ledger"] == original["ledger"] and record.model_dump(mode="json") == original
    assert "MACHINE_PAYLOAD" not in appendix
    assert appendix.count("Exact source and metric scope retained") == 1
    for text in ("original_workdir", "original_image", "original_stdout.log", "STDOUT_FULL",
                 "STDERR_FULL", "runtime_seconds", "max_attempts", "sample_count", "tolerance",
                 "gap", "approved", "training_budget", "tokens", "REPAIR_DIFF_FULL", "UNKNOWN_FULL"):
        assert text in appendix
    assert v2._text("eval.py") in main and "returncode" in main
    for row in manifest["ledger_rendering"]:
        assert digest(resolve(saved, row["json_pointer"])) == row["sha256"]
        assert row["canonical_artifact"] == "final_review.json"
        assert row["raw_value_printed"] is False
        assert row["appendix_anchor"] in appendix
        location = manifest["records"][row["json_pointer"]]
        assert location["pdf_location_role"] == "raw_json_locator"
        assert manifest["ledger_raw_artifact"] == row["canonical_artifact"]
    raw = v2.write_review(record, tmp_path / "full", render_pdf=False)
    assert '"approval_mode": "auto"' in Path(raw["markdown"]).read_text("utf-8")
    assert "MACHINE_PAYLOAD" in Path(raw["markdown"]).read_text("utf-8")


def test_changed_binding_legacy_and_escaped_pointer(tmp_path):
    record = fixture()
    record.ledger[0]["alignment"][0]["paper_target_binding"]["value"] = .5
    binding = record.ledger[0]["plan"]["target_bindings"].pop("c1")
    record.ledger[0]["plan"]["target_bindings"]["c/1~"] = binding
    record.ledger.append({"plan_id": "legacy", "command": ["python", "legacy.py"], "logs": "legacy.log"})
    output = v2.write_review(record, tmp_path, presentation="layered", render_pdf=False)
    appendix = Path(output["appendix_markdown"]).read_text("utf-8")
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    assert appendix.count("Exact source and metric scope retained") == 2
    assert "legacy.py" in appendix and "legacy.log" in appendix
    assert any("c~11~0" in row["json_pointer"] for row in manifest["ledger_rendering"])
    saved = json.loads(Path(output["json"]).read_text("utf-8"))
    assert saved["ledger"] == record.model_dump(mode="json")["ledger"]
    assert len({row["sha256"] for row in manifest["ledger_rendering"]
                if row["kind"] == "binding"}) == 2


def test_real_pdf_navigation_and_artifact_hashes(tmp_path):
    output = v2.write_review(fixture(), tmp_path, presentation="layered")
    assert not any(k.endswith("_error") for k in output)
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    bundle = PdfReader(output["bundle_pdf"])
    page_ids = {page.indirect_reference.idnum for page in bundle.pages}
    links = []
    for page in bundle.pages:
        for ref in page.get("/Annots", []):
            ann = ref.get_object()
            if ann.get("/Subtype") == "/Link":
                action = ann.get("/A")
                assert action is None or action["/S"] == "/GoTo"
                dest = action["/D"] if action else ann["/Dest"]
                assert dest[0].idnum in page_ids
                links.append(dest)
    assert links
    for row in manifest["ledger_rendering"]:
        assert row["appendix_anchor"] in manifest["pdf_targets"]["appendix"]
        assert row["appendix_anchor"] in manifest["pdf_targets"]["bundle"]
    for row in manifest["artifacts"].values():
        assert hashlib.sha256((tmp_path / row["name"]).read_bytes()).hexdigest() == row["sha256"]
    assert "MACHINE_PAYLOAD" not in "\n".join(page.extract_text() for page in bundle.pages)


def test_static_uses_same_projection_without_models(tmp_path):
    record = fixture()
    output = v2.write_static_review(record, tmp_path, presentation="layered", pdf_keys=())
    saved = json.loads(Path(output["json"]).read_text("utf-8"))
    manifest = json.loads(Path(output["manifest"]).read_text("utf-8"))
    assert saved["ledger"] == record.model_dump(mode="json")["ledger"]
    assert manifest["static_finalization"] and manifest["model_calls"] == 0
    assert manifest["ledger_rendering"]
    assert "MACHINE_PAYLOAD" not in Path(output["appendix_markdown"]).read_text("utf-8")


def test_unknown_nested_registry_is_not_a_machine_catalog(tmp_path):
    record = fixture()
    record.ledger[0]["plan"]["target_bindings"]["c1"]["unknown"] = {
        "projection": {"registry_snapshot": {"scientific_detail": "UNKNOWN_NESTED_FULL"}}}
    output = v2.write_review(record, tmp_path, presentation="layered", render_pdf=False)
    assert "UNKNOWN_NESTED_FULL" in Path(output["appendix_markdown"]).read_text("utf-8")
    saved = json.loads(Path(output["json"]).read_text("utf-8"))
    assert saved["ledger"] == record.model_dump(mode="json")["ledger"]
