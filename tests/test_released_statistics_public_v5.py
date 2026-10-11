"""One public control; semantic model and execution transport are external mocks."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import fitz
import pytest

from assessment.rules import assess_claim
from common.run_stats import run_scope
from fact_generation.execution.v2 import ExecutionConfig, Observation, RunOutcome, execute_plans
from llm.client import LLMConfig
from review.report.v2 import render_markdown
from schemas.review import FinalReview
from screening.claims import extract_claims
from tests.experiment_authority_fixtures import authority_scope_fixture
from tests.test_execution_projection_v2 import original_inputs
from verification.contracts import BranchResult
from verification.dispatch import verify_claims
from verification.execution_projection import digest


def _materials(path, correct):
    path.mkdir()
    _, materials = original_inputs(path)
    text = (
        "On MiniSet test, model ExactMatch has accuracy 0.75. "
        "Accuracy is a fraction, computed by exact equality of each prediction and label.\n"
        "75 of the 100 fixed test predictions are correct.\n"
        "25 fixed test predictions are incorrect.\n"
        "Table 1 reports the result and Figure 1 shows the corresponding counts. "
        "No repeated-run uncertainty or population-performance conclusion is claimed."
    )
    block = next(b for b in materials.blocks if b.id == "block_10")
    materials.markdown = materials.markdown.replace(block.text, text)
    block.text = text
    for block_id, old, new in (("block_3", "four fixed predictions", "100 fixed predictions"),
                              ("block_7", "four predictions", "100 predictions"),
                              ("block_11", "<td>4</td>", "<td>100</td>"),
                              ("block_13", "three correct and one incorrect", "75 correct and 25 incorrect")):
        row = next(b for b in materials.blocks if b.id == block_id)
        replacement = row.text.replace(old, new)
        materials.markdown = materials.markdown.replace(row.text, replacement)
        row.text = replacement
    materials.abstract = materials.abstract.replace("four fixed predictions", "100 fixed predictions")
    materials.markdown = materials.markdown.replace("three correct and one incorrect", "75 correct and 25 incorrect")
    for row in materials.blocks:
        if row.loc is not None and row.loc.char_start is not None and materials.markdown.count(row.text) == 1:
            row.loc.char_start = materials.markdown.index(row.text)
            row.loc.char_end = row.loc.char_start + len(row.text)
    reference = "Table 1 reports the result and Figure 1 shows the corresponding counts."
    for item in [*materials.figures, *materials.tables]:
        for row in item.references:
            row.loc.char_start = materials.markdown.index(reference)
            row.loc.char_end = row.loc.char_start + len(reference)
    for figure in materials.figures:
        figure.caption = next(b.text for b in materials.blocks if b.id == "block_13")
    materials.markdown_path = str(path / "paper.md")
    Path(materials.markdown_path).write_text(materials.markdown, encoding="utf-8")
    # These are new synthetic assets. No historical crop/source is read or modified.
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 35, 555, 790), "\n".join(b.text for b in materials.blocks if b.loc and b.loc.page == 1), fontsize=8)
        table_image = path / "table.png"
        page.get_pixmap().save(table_image)
        page = doc.new_page()
        page.insert_text((50, 40), "Fixed prediction counts: 75 correct; 25 incorrect; total 100", fontsize=12)
        page.draw_rect(fitz.Rect(80, 70, 155, 220), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
        page.draw_rect(fitz.Rect(200, 170, 275, 220), color=(0, 0, 0), fill=(0.8, 0.3, 0.2))
        page.insert_text((80, 240), "75 correct       25 incorrect", fontsize=12)
        page.insert_textbox(fitz.Rect(40, 275, 555, 790), "\n".join(b.text for b in materials.blocks if b.loc and b.loc.page == 2), fontsize=8)
        figure_image = path / "figure.png"
        page.get_pixmap().save(figure_image)
        pdf = path / "statistics-paper.pdf"
        doc.save(pdf)
    materials.source_pdf = str(pdf)
    for page in materials.pages:
        page.path = str(table_image if page.page == 1 else figure_image)
        page.dpi = 72
    for item in materials.tables:
        item.crop_path = item.printed_crop_path = str(table_image)
        item.bbox_points = (0, 0, 595, 842)
        item.printed_dpi = 72
    for item in materials.figures:
        item.crop_path = item.printed_crop_path = str(figure_image)
        item.bbox_points = (0, 0, 595, 842)
        item.printed_dpi = 72
    content_list = path / "synthetic-content-list.json"
    content_list.write_text(json.dumps([b.model_dump(mode="json") for b in materials.blocks], ensure_ascii=False), encoding="utf-8")
    materials.content_list_path = str(content_list)
    materials.provider = "synthetic-unit-input"
    repo = Path(materials.repository.root)
    readme = repo / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace("the four released", "the 100 released"), encoding="utf-8")
    data = [{"label": 0, "prediction": 0 if n < correct else 1} for n in range(100)]
    (repo / "data/test_predictions.json").write_text(json.dumps(data), encoding="utf-8")
    for row in materials.repository.files:
        row.sha256 = hashlib.sha256((repo / row.path).read_bytes()).hexdigest()
    return materials, text


def _selection(candidate, text, *, bad_fragment=False, partial_count=False):
    source = next(sid for sid, row in candidate["sources"].items() if row["quote"] == text)
    ranking = next(sid for sid, row in candidate["sources"].items() if row["block_id"] == "block_11")
    def atom(id, kind, **kw):
        return {"id": id, "kind": kind, "source_ids": [source], **kw}
    atoms = [
        atom("dataset", "dataset_identity", dataset="MiniSet", split="test"),
        atom("model", "runtime_setting", key="model", value="ExactMatch"),
        atom("split", "runtime_setting", key="split", value="test"),
        atom("value", "reported_value", number_id=candidate["number_id"]),
        atom("sample", "sample_scope", count=100, population="fixed_released_predictions", split="test", all_records=True),
        atom("definition", "measurement_definition", measure="exact_match_accuracy", predicate="prediction_equals_label",
             aggregation="fraction_of_all_records", unit="fraction"),
        atom("boundaries", "conclusion_boundary", excludes=["repeated_run_uncertainty", "population_performance"]),
        {"id": "ranking", "kind": "conclusion_boundary", "source_ids": [ranking], "excludes": ["cross_model_ranking"]},
    ]
    for outcome, token in (("correct", "75"), ("incorrect", "25")):
        start = text.index(token + (" of" if outcome == "correct" else " fixed"))
        atoms.append(atom(outcome, "outcome_count", outcome=outcome, expected_count=int(token),
                          count_binding={"source_id": source, "start": start, "end": start+len(token), "token": token}))
    if partial_count:
        start = text.index("0.75") + 2
        atoms[-2]["count_binding"].update(start=start, end=start+2)
    ids = {"/dataset": ["dataset"], "/metric": ["definition"], "/description": ["dataset", "model", "definition"],
           "/settings/model": ["model"], "/settings/accuracy": ["value"], "/settings/examples": ["sample"],
           "/settings/correct_predictions": ["correct"], "/settings/incorrect_predictions": ["incorrect"],
           "/settings/accuracy_definition": ["definition"], "/settings/qualifiers/0": ["boundaries"],
           "/settings/qualifiers/1": ["ranking"]}
    fields = []
    for key, value in candidate["field_inventory"].items():
        empty = value is None or value == "" or (type(value) in {dict, list} and not value)
        mapped = [] if empty else ids[key]
        fields.append({"path": key, "value_digest": digest(value), "original_empty": empty, "atom_ids": mapped,
                       "fragments": [{"start": 0, "end": len(value), "text_digest": digest(value),
                                      "separator": False, "atom_ids": mapped}] if isinstance(value, str) and value else []})
    claim_ids = [a["id"] for a in atoms if a["id"] != "ranking"]
    fragment = {"start": 0, "end": len(text), "text_digest": digest(text), "separator": False, "atom_ids": claim_ids}
    if bad_fragment:
        fragment["end"] -= 1
    return {"version": "released-predictions-choice-v2", "condition_id": candidate["condition_id"],
            "candidate_id": candidate["candidate_id"], "decision": "select", "rationale": "Fixed released-row statistics.",
            "proposal": {"version": "released-predictions-v3", "domain": "full-list-exact-match-statistics-v1",
                         "recipe": "exact_match_accuracy", "candidate_id": candidate["candidate_id"],
                         "data_path": candidate["data_path"], "reported_number_id": candidate["number_id"],
                         "atoms": atoms, "field_bindings": fields, "claim_bindings": [fragment]}}


def test_public_statistics_original_fields_independent_review_and_host_counts(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Unmocked external boundary")

    for name in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send", "subprocess.run",
                 "screening.checks.llm_json"):
        monkeypatch.setattr(name, forbidden)
    for name in ("screening.claims.resolve_llm_config", "screening.checks.resolve_llm_config", "verification.literature.resolve_llm_config"):
        monkeypatch.setattr(name, lambda: LLMConfig("mock", "mock", None, None))
    results = {}
    for mode, correct in (("healthy", 75), ("count_mismatch", 74), ("unresolved", 75), ("missing_fragment", 75), ("partial_count_token", 75)):
        case = tmp_path / mode
        materials, text = _materials(case, correct)
        calls, raw, runs = [], [], []

        def model(**kw):
            module = kw["module"]
            calls.append(module)
            if module == "screening.claims":
                response = {"status": "ok", "claims": [{"text": text, "source_block_id": "block_10", "source_quote": text,
                    "source_refs": [{"source_block_id": "block_11", "source_quote": next(b.text for b in materials.blocks if b.id == "block_11"), "covered": ["c1"]}],
                    "conditions": [{"id": "c1", "dataset": "MiniSet", "metric": "accuracy", "settings": {
                        "model": "ExactMatch", "accuracy": 0.75, "examples": 100, "correct_predictions": 75,
                        "incorrect_predictions": 25, "accuracy_definition": "fraction computed by exact equality of each prediction and label",
                        "qualifiers": ["No repeated-run uncertainty or population-performance conclusion claimed", "No ranking against other models"],
                        "empty_list": [], "empty_object": {}, "empty_value": None},
                        "description": "ExactMatch measured test accuracy on MiniSet test."}], "needs": ["Experiments"], "importance": "core"}]}
            else:
                payload = json.loads(kw["prompt"])
                if module == "verification.experiments":
                    registry = payload["released_statistics_context"]
                    assert payload["output_schema"]["properties"]["schema_version"]["const"] == "experiments-execution-v5"
                    assert not registry["previous_registry"]["candidates"]
                    assert registry["route_state"]["enumeration_complete"] is True
                    assert registry["budget"]["checks_used"] > registry["previous_registry"]["budget"]["checks_used"]
                    assert len(registry["candidates"]) == 1
                    c = next(iter(registry["candidates"].values()))
                    assert "numerator" not in c["recipe_facts"]
                    response = {"schema_version": "experiments-execution-v5", "execution_domain": registry["domain"],
                        "execution_context_digest": registry["context_digest"],
                        "checked_aspects": ["correspondence", "fairness", "isolation", "stability", "consistency"], "items": [],
                        "execution": {"kind": "statistics_choice", "selection": _selection(c, text,
                            bad_fragment=mode == "missing_fragment", partial_count=mode == "partial_count_token")}}
                else:
                    assert module == "verification.experiments.scope"
                    context = payload["released_statistics_context"]
                    selection = context["selection"]
                    assert payload["claim"]["text"] == text
                    assert set(context["registry"]["candidates"][selection["candidate_id"]]["field_inventory"]) >= {
                        "/settings/empty_list", "/settings/empty_object", "/settings/empty_value", "/settings/correct_predictions"}
                    assert any(row["kind"] == "composition" and row["path"] == "/text" for row in context["obligations"])
                    reviews = [{"obligation_id": row["obligation_id"], "decision": "confirmed",
                                "rationale": "The supplied original field and complete fragment explicitly express this finite released-statistics meaning."}
                               for row in context["obligations"]]
                    if mode == "unresolved":
                        reviews[-1]["decision"] = "unresolved"
                    condition = {"condition_id": selection["condition_id"], "assertion": "descriptive",
                        "matched_controls_required": False, "uncertainty_sensitive": False, "relation": "eq",
                        "rationale": "Absolute fixed released-predictions assertion, with all negative boundaries retained."}
                    response = authority_scope_fixture(payload, {"conditions": [condition], "items": []}, requirement="unresolved", basis="unresolved",
                        rationale="Execution-only review does not assert independent paper-internal support.")
                    response.update(execution_scope_domain=context["domain"], released_statistics_reviews=[{
                        "version": "statistics-review-v1", "condition_id": selection["condition_id"],
                        "candidate_id": selection["candidate_id"], "proposal_sha256": context["proposal_sha256"],
                        "context_digest": context["context_digest"], "decision": "confirmed", "reviews": reviews,
                        "rationale": "Independent full original statement, fragments and field composition review."}])
            raw.append(copy.deepcopy(response))
            return response

        async def external_literature(*a, **kw):
            return BranchResult()

        with run_scope(case / "stats.json"):
            extracted = extract_claims(materials, call=model)
            original = extracted[0].model_dump(mode="json")
            verified = asyncio.run(verify_claims(extracted, materials, case / "verification", call=model, global_literature=external_literature))
        (case / "public-debug.json").write_text(json.dumps({"calls": calls, "issues": verified.issues, "raw": raw}, indent=2), encoding="utf-8")
        assert len(extracted) == 1 and extracted[0].model_dump(mode="json") == original
        assert verified.claims[0].conditions == extracted[0].conditions
        assert verified.claims[0].text == text
        assert calls[0:2] == ["screening.claims", "verification.experiments"]
        if mode in {"unresolved", "missing_fragment", "partial_count_token"}:
            assert not verified.plans
            assert assess_claim(verified.claims[0]).status.value == "unverified"
            assert verified.issues
            results[mode] = {"calls": calls, "status": "unverified", "issues": verified.issues}
            continue
        assert calls == ["screening.claims", "verification.experiments", "verification.experiments.scope"]
        assert len(verified.plans) == 1
        plan = verified.plans[0]
        binding = plan.target_bindings["c1"]
        assert binding.projection.record_version == "released-statistics-choice-v1"
        assert plan.target_conditions == extracted[0].conditions
        assert json.loads(Path(binding.projection.scope_audit).read_text(encoding="utf-8"))["first_pass_response"] == raw[1]

        def runner(request):
            runs.append(request)
            observation = Observation(dataset="MiniSet", metric="accuracy", settings={"split": "test", "model": "ExactMatch"}, value=correct/100)
            return RunOutcome(returncode=0, observations=[observation], stdout=json.dumps({"observations": [observation.model_dump(mode="json")]}))

        executed = execute_plans(verified.plans, verified.claims, materials, case / "execution",
                                 config=ExecutionConfig(refine_with_llm=False, training_budget=0), runner=runner)
        assert len(runs) == 1, executed.issues
        assert len(executed.claims[0].evidence) == 1, executed.issues
        assessed = assess_claim(executed.claims[0])
        expected = "supported" if mode == "healthy" else "flawed"
        assert assessed.status.value == expected
        ledger = executed.ledger[0]
        alignment = ledger["alignment"][0]
        audit = json.loads(Path(alignment["measurement"]).read_text(encoding="utf-8"))
        assert audit["statistics"] == {"total": 100, "correct": correct, "incorrect": 100-correct, "accuracy": correct/100}
        assert audit["all_records"] is True and audit["training_performed"] is False
        assert audit["author_observation"]["value"] == correct/100
        assert Path(audit["base_measurement"]["locator"]).is_file()
        assert alignment["released_statistics"]["scalar_consistent"] is True
        assert alignment["released_statistics"]["consistent"] is (mode == "healthy")
        assert alignment["tolerance"] == 0.02
        review = FinalReview(paper_key=materials.paper_key, run_id=mode, claims=[assessed], ledger=[ledger])
        assert f"| {expected} | 1 |" in render_markdown(review)
        assert assessed.conditions == extracted[0].conditions
        results[mode] = {"calls": calls, "status": expected, "alignment": alignment, "host": audit}
    (tmp_path / "public-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
