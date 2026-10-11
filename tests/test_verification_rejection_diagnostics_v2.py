"""Actual consumer rejection gates retain their outputs without external services."""

from __future__ import annotations

import copy
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llm.client import LLMConfig
from schemas.claim import Claim, ClaimLocation, Condition
from schemas.materials import MaterialBlock, RepositoryIndex, SharedMaterials
from tests.experiment_authority_fixtures import authority_scope_fixture
from verification.experiments import verify_experiments
from verification.theory import verify_theory

MARKER = "PRIVATE_MOCK_CONTENT_76bd9b"
ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]


def _fixture(tmp_path):
    sources = tmp_path / "author_sources"
    sources.mkdir()
    repository = tmp_path / "author_repository"
    repository.mkdir()
    text = f"For real x and y, (x+y)^2 = x*x + 2*x*y + y*y by distributivity. {MARKER}"
    markdown = sources / "paper.md"
    secondary = f"Original source context {MARKER}"
    full_text = text + "\n" + secondary
    markdown.write_text(full_text, encoding="utf-8")
    loc = ClaimLocation(page=1, section="Theory", char_start=0, char_end=len(text))
    materials = SharedMaterials(
        paper_key=MARKER, source_pdf=str(sources / "paper.pdf"), markdown=full_text,
        markdown_path=str(markdown), content_list_path=str(sources / "content.json"),
        provider="mock", blocks=[MaterialBlock(id=MARKER, text=text, loc=loc),
            MaterialBlock(id=f"second_{MARKER}", text=secondary,
                loc=ClaimLocation(page=1, section="Theory", char_start=len(text) + 1,
                    char_end=len(full_text)))],
        repository=RepositoryIndex(root=str(repository)),
    )
    claim = Claim(
        id=f"claim_{MARKER}", text=text, loc=loc, source_block_id=MARKER,
        source_quote=text, conditions=[Condition(id=f"condition_{MARKER}",
            description=f"The stated identity for real inputs {MARKER}")],
        needs=["Theory", "Experiments"],
    )
    condition = claim.conditions[0].id
    item = {"block_id": MARKER, "quote": text, "step_quote": "(x+y)^2 = x*x + 2*x*y + y*y",
        "covered": [condition], "fully_supported_conditions": [], "kind": "no_proof",
        "direction": "flaw", "detail": f"{MARKER} experiments_choice_privacy_blocked"}
    trace = {"goal": MARKER, "assumptions": [], "steps": [
        {"id": "s1", "statement": MARKER, "reason": MARKER, "assumption_ids": [],
         "previous_step_ids": ["s2"], "sources": [{"block_id": MARKER, "quote": text}]},
        {"id": "s2", "statement": MARKER, "reason": MARKER, "assumption_ids": [],
         "previous_step_ids": ["s1"], "sources": []}],
        "gaps": [], "outcome": "completed", "completion_reason": MARKER}
    theory = {"schema_version": "theory-derivation-v1", "items": [item, {**item, "kind": "missing_assumption"}],
        "appendix_block_ids": [], "derivations": [
            {"item_index": index, "trace": copy.deepcopy(trace)} for index in range(2)]}
    experiments = {"checked_aspects": ASPECTS, "items": [{
        "aspect": "correspondence", "kind": "paper_support", "block_id": MARKER,
        "quote": text, "covered": [condition], "fully_supported_conditions": [],
        "detail": f"{MARKER} theory_no_proof_trace_invalid",
        "additional_sources": [{"block_id": f"second_{MARKER}", "quote": secondary}]}], "plans": []}
    return claim, materials, theory, experiments


@pytest.mark.parametrize("consumer_modes", [("normal", "diagnostic_io_failure")], ids=["controlled-consumers"])
def test_rejection_sidecars_preserve_consumer_results(tmp_path, monkeypatch, consumer_modes):
    monkeypatch.setattr("screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "mock", None, None))
    def fail(*args, **kwargs):
        pytest.fail("Unmocked external boundary")
    monkeypatch.setattr("screening.checks.llm_json", fail)
    monkeypatch.setattr("httpx.Client.send", fail)
    monkeypatch.setattr("httpx.AsyncClient.send", fail)
    monkeypatch.setattr("requests.sessions.Session.request", fail)
    monkeypatch.setattr("subprocess.run", fail)
    monkeypatch.setattr("subprocess.Popen", fail)
    monkeypatch.setattr("common.run_stats.stats_path", lambda: None)
    claim, materials, theory, experiments = _fixture(tmp_path)
    original_inputs = (claim.model_dump(mode="json"), materials.model_dump(mode="json"))
    original_open = Path.open
    comparisons, pending = [], []

    for branch, function, response, expected_code in (
        ("Theory", verify_theory, theory, "theory_no_proof_trace_invalid"),
        ("Experiments", verify_experiments, experiments, "experiments_joint_final_validation_failed"),
    ):
        for mode in consumer_modes:
            calls = []

            def model(**kwargs):
                calls.append(kwargs["module"])
                if kwargs["module"] != "verification.experiments.scope":
                    return copy.deepcopy(response)
                payload = json.loads(kwargs["prompt"])
                review = {"conditions": [{"condition_id": claim.conditions[0].id,
                    "claim_quote": claim.text, "assertion": "descriptive", "rationale": MARKER,
                    "matched_controls_required": False, "uncertainty_sensitive": False,
                    "relation": "none", "subject": "x", "comparator": ""}],
                    "items": [{"item_index": 0, "condition_id": claim.conditions[0].id,
                        "applicability": "applicable", "grounds": [{"block_id": MARKER,
                            "quote": materials.blocks[0].text}], "rationale": MARKER, "full_support": False,
                        "qualifiers_complete": False, "comparison_objects": "unresolved", "comparisons": [],
                        "source_uses": [{"source_id": f"foreign_{MARKER}", "roles": ["other_qualifier"],
                            "rationale": MARKER}]}]}
                return authority_scope_fixture(payload, review, requirement="manuscript_reported_result",
                    basis="reported_measurement", rationale=MARKER)

            options = {"call": model}
            if branch == "Theory":
                options["visual_recheck_rounds"] = 0
            else:
                options.update(scope_binding_repair_rounds=0, scope_paper_format="catalog-v2")
            before = function(claim, materials, **options).model_dump(mode="json")
            before_calls = list(calls)
            calls.clear()
            directory = tmp_path / "operator_output" / f"{branch}_{mode}" / "diagnostics"

            def controlled_open(path, open_mode="r", *args, **kwargs):
                open_mode = kwargs.pop("mode", open_mode)
                if mode == "diagnostic_io_failure" and open_mode == "x" and path.parent == directory:
                    raise OSError(MARKER)
                return original_open(path, open_mode, *args, **kwargs)

            with monkeypatch.context() as control:
                control.setattr(Path, "open", controlled_open)
                enabled_options = dict(options)
                if "diagnostics_dir" in inspect.signature(function).parameters:
                    enabled_options["diagnostics_dir"] = directory
                after = function(claim, materials, **enabled_options).model_dump(mode="json")
                ordinary = directory.parent / "ordinary_audit.json"
                ordinary.parent.mkdir(parents=True, exist_ok=True)
                ordinary.write_text(json.dumps({"mock": MARKER}), encoding="utf-8")
                assert json.loads(ordinary.read_text(encoding="utf-8")) == {"mock": MARKER}
            assert before == after
            assert before_calls == calls
            assert before["plans"] == []
            assert any(row["kind"] == "evidence_validation_failed" for row in before["verification_limitations"])
            assert not any(row["sufficient"] for row in before["evidence"])
            if branch == "Theory":
                assert len(before["theory_derivations"]) == 2
                assert all(row["state"] == "invalid" for row in before["theory_derivations"])
                assert before_calls == ["verification.theory"]
            else:
                assert before_calls == ["verification.experiments", "verification.experiments.scope"]
            assert original_inputs == (claim.model_dump(mode="json"), materials.model_dump(mode="json"))
            comparisons.append({"branch": branch, "mode": mode, "before": before, "after": after,
                "before_calls": before_calls, "after_calls": list(calls), "inputs_unchanged": True})
            pending.append((directory, branch, mode, expected_code))
    evidence = os.environ.get("L2_DIAGNOSTIC_EVIDENCE_PATH")
    if evidence:
        Path(evidence).write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    for directory, branch, mode, expected_code in pending:
        paths = list(directory.glob("*.json"))
        assert len(paths) == (1 if mode == "normal" else 0)
        for path in paths:
            raw = path.read_bytes()
            assert len(raw) <= 512 and MARKER.encode() not in raw
            assert str(tmp_path).encode() not in raw
            assert json.loads(raw) == {"schema": "verification-rejection-diagnostics-v1", "branch": branch,
                "scope": "first_evidence_validation_rejection", "code": expected_code}

    from verification.rejection_diagnostics import FirstRejectionDiagnostic

    shared = tmp_path / "operator_output" / "concurrent" / "diagnostics"

    def independent_call(_):
        writer = FirstRejectionDiagnostic(shared, materials, "Theory")
        writer.record("theory_no_proof_trace_invalid")
        writer.record("theory_concern_trace_invalid")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(independent_call, range(8)))
    paths = list(shared.glob("*.json"))
    assert len(paths) == 8 and len({p.name for p in paths}) == 8
    assert all(json.loads(path.read_text(encoding="utf-8"))["code"] == "theory_no_proof_trace_invalid" for path in paths)
    denied = [Path(materials.markdown_path).parent / "diagnostics",
        Path(materials.repository.root) / "diagnostics", shared.parent / "wrong-directory"]
    for directory in denied:
        FirstRejectionDiagnostic(directory, materials, "Theory").record("theory_no_proof_trace_invalid")
        assert not directory.exists()
    FirstRejectionDiagnostic(shared, materials, "Theory").record(f"unknown_{MARKER}")
    FirstRejectionDiagnostic(shared, materials, MARKER).record("theory_no_proof_trace_invalid")
    assert len(list(shared.glob("*.json"))) == 8
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "operator_output" / "linked"
    # Construct a public filesystem-control fixture without executing a subprocess.
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        # Windows may require a privilege unavailable in a unit-test process.
        # The fixed mock lstat control below still exercises the reparse rejection.
        from types import SimpleNamespace
        original_lstat = Path.lstat
        with monkeypatch.context() as control:
            control.setattr(Path, "lstat", lambda path, *a, **kw:
                SimpleNamespace(st_mode=0, st_file_attributes=0x400) if path == link
                else original_lstat(path, *a, **kw))
            FirstRejectionDiagnostic(link / "diagnostics", materials, "Theory").record("theory_no_proof_trace_invalid")
    else:
        FirstRejectionDiagnostic(link / "diagnostics", materials, "Theory").record("theory_no_proof_trace_invalid")
    assert not (outside / "diagnostics").exists()
