"""Paper-source contract through public mocked boundaries and the pure host binder."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import llm.client
from fact_generation.execution import inference_runtime, v2
from tests.test_inference_binding_diagnostics_v2 import (
    test_public_binding_diagnostics_are_closed_and_preserve_scientific_qualification as public_binding_fixture,
)
from verification.code_scope import _claim_sources


def test_public_inference_paper_contract_and_closed_diagnostics(tmp_path, monkeypatch):
    original_refine = v2._refine
    original_request = inference_runtime.bind_inference_request
    saved = {}
    prompt_calls = []

    def refine(plan, workspace, config, *, claim=None, materials=None):
        completion = llm.client.llm_json  # The public fixture installed its mock.

        def inspect_completion(prompt, system, cfg, *, module):
            payload = json.loads(prompt)
            assert payload["paper_claim_sources"] == list(_claim_sources(claim, materials).values())
            assert payload["scientific_claim"]["source_refs"] == claim.model_dump(mode="json")["source_refs"]
            for phrase in ("zero-based Unicode codepoint", "block-local", "[start,end)",
                           "decoded block.text", "unique", "paper_claim_sources", "covered"):
                assert phrase in system
            prompt_calls.append(payload["paper_claim_sources"])
            return completion(prompt, system, cfg, module=module)

        monkeypatch.setattr(llm.client, "llm_json", inspect_completion)
        try:
            return original_refine(plan, workspace, config, claim=claim, materials=materials)
        finally:
            monkeypatch.setattr(llm.client, "llm_json", completion)

    def capture_request(request, **kwargs):
        if Path(request.run_dir).parent.name == "healthy":
            saved.update(request=request.model_copy(deep=True), kwargs=kwargs)
        return original_request(request, **kwargs)

    monkeypatch.setattr(v2, "_refine", refine)
    monkeypatch.setattr(inference_runtime, "bind_inference_request", capture_request)
    public_binding_fixture(tmp_path, monkeypatch)
    assert len(prompt_calls) == 3
    request, kwargs = saved["request"], saved["kwargs"]
    healthy = original_request(request, **kwargs)
    assert healthy["status"] == "bound" and set(healthy) == {"status", "request_sha256", "binding"}
    sidecar_name = "inference_paper_scope_diagnostics.json"
    assert not (Path(request.run_dir) / "attempt_0" / sidecar_name).exists()
    paper = kwargs["materials"].blocks[0].text
    unresolved = {"status": "unresolved", "reason": "inference_request_binding_unavailable",
                  "binding_failure": "paper_source_scope_mismatch"}
    mutations = {
        "span": ({**request.source_science["paper"]["dataset"],
                  "start": request.source_science["paper"]["dataset"]["start"] + 1},
                 {"span_matches": False, "quote_unique": None, "authorized_source_matches": None}),
        "unique": ({"block_id": "paper", "quote": "Numeric", "start": paper.index("Numeric"),
                    "end": paper.index("Numeric") + len("Numeric")},
                   {"span_matches": True, "quote_unique": False, "authorized_source_matches": None}),
        "authorization": ({"block_id": "paper", "quote": paper, "start": 0, "end": len(paper)},
                          {"span_matches": True, "quote_unique": True, "authorized_source_matches": False}),
    }
    for name, (definition, checks) in mutations.items():
        case = request.model_copy(deep=True)
        case.source_science["paper"]["dataset"] = definition
        run_dir = tmp_path / name
        (run_dir / "attempt_0").mkdir(parents=True)
        case.run_dir = str(run_dir)
        assert original_request(case, **kwargs) == unresolved
        sidecar_path = run_dir / "attempt_0" / sidecar_name
        payload = json.loads(sidecar_path.read_bytes())
        assert payload == {"schema": "torch-inference-paper-scope-diagnostics-v1",
                           "role": "dataset", "checks": checks}
        assert set(payload) == {"schema", "role", "checks"}
        assert definition["quote"] not in sidecar_path.read_text(encoding="utf-8")
        with pytest.raises(ValueError) as captured:
            inference_runtime.bind_inference_proposal(case.source_science, plan=case.plan,
                claim=kwargs["claim"], materials=kwargs["materials"], workspace=case.workspace,
                source_files=case.source_science_files,
                limits=case.config.docker_options["torch_inference"], source_sites=case.source_sites)
        assert type(captured.value) is ValueError
        assert captured.value.args == ("paper_source_scope_mismatch",)

        # Existing sidecars cannot be replaced, including non-JSON operator data.
        sidecar_path.write_bytes(b"preexisting operator record")
        assert original_request(case, **kwargs) == unresolved
        assert sidecar_path.read_bytes() == b"preexisting operator record"

    failing = request.model_copy(deep=True)
    failing.source_science["paper"]["dataset"] = mutations["span"][0]
    failing.run_dir = str(tmp_path / "write-failure")
    (Path(failing.run_dir) / "attempt_0").mkdir(parents=True)
    original_open = Path.open

    def reject_sidecar(path, *args, **kwargs):
        if path.name == sidecar_name:
            raise OSError("private writer failure must not affect qualification")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", reject_sidecar)
        assert original_request(failing, **kwargs) == unresolved
    assert not (Path(failing.run_dir) / "attempt_0" / sidecar_name).exists()
    assert original_request(request, **kwargs) == healthy


def test_paper_scope_diagnostic_preserves_short_circuit_and_output_boundary(tmp_path):
    class NoCount(str):
        def count(self, *args):
            pytest.fail("quote uniqueness must stay unevaluated after a span mismatch")

    class NoSources(dict):
        def values(self):
            pytest.fail("authorization must stay unevaluated after an earlier mismatch")

    for text, quote, end, expected in (
        (NoCount("actual"), "wrong", 5,
         {"span_matches": False, "quote_unique": None, "authorized_source_matches": None}),
        ("same same", "same", 4,
         {"span_matches": True, "quote_unique": False, "authorized_source_matches": None}),
    ):
        with pytest.raises(ValueError) as captured:
            inference_runtime._require_paper_source_scope(
                SimpleNamespace(start=0, end=end, quote=quote, block_id="secret-block"),
                SimpleNamespace(text=text), "secret-condition", NoSources(), "qualifiers")
        error = captured.value
        assert type(error) is ValueError and error.args == ("paper_source_scope_mismatch",)
        assert error.paper_scope_diagnostics == {"role": "qualifiers", "checks": expected}

    # Invalid rounds/fields and output paths inside author workspaces never write.
    repo = tmp_path / "author"
    repo.mkdir()
    materials = SimpleNamespace(repository=SimpleNamespace(root=str(repo)))
    error = ValueError("paper_source_scope_mismatch")
    error.paper_scope_diagnostics = {"role": "dataset", "checks": {
        "span_matches": False, "quote_unique": None, "authorized_source_matches": None}}
    for round_number in (-1, 4, True):
        (tmp_path / f"attempt_{round_number}").mkdir()
        inference_runtime._write_paper_scope_diagnostics(
            SimpleNamespace(run_dir=str(tmp_path), workspace=str(repo), repair_round=round_number), error, materials)
    author_attempt = repo / "attempt_0"
    author_attempt.mkdir()
    inference_runtime._write_paper_scope_diagnostics(
        SimpleNamespace(run_dir=str(repo), workspace=str(repo), repair_round=0), error, materials)
    assert not (author_attempt / "inference_paper_scope_diagnostics.json").exists()
    (tmp_path / "attempt_0").mkdir()
    for invalid_detail in (
        {"role": "untrusted-role", "checks": error.paper_scope_diagnostics["checks"]},
        {"role": "dataset", "checks": {"span_matches": 0, "quote_unique": None,
                                         "authorized_source_matches": None}},
        {**error.paper_scope_diagnostics, "untrusted-field": "private input"},
    ):
        error.paper_scope_diagnostics = invalid_detail
        inference_runtime._write_paper_scope_diagnostics(
            SimpleNamespace(run_dir=str(tmp_path), workspace=str(repo), repair_round=0), error, materials)
    assert not list(tmp_path.rglob("inference_paper_scope_diagnostics.json"))
