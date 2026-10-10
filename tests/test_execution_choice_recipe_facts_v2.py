"""Public choice recipe facts keep paper semantics independently reviewed."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from common.run_stats import run_scope
from llm.client import LLMConfig
from tests.test_execution_projection_choices_v2 import review, select
from tests.test_execution_projection_semantics_v2 import inputs
from verification.execution_projection import ProjectionError, digest
from verification.execution_projection_choices import choice_context, revalidate_registry
from verification.experiment_targets import validate_plan_targets
from verification.experiments import verify_experiments


def test_public_recipe_facts_require_all_original_reviews_and_rebuild(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("External/model/process boundary must be explicitly mocked")

    for target in (
        "requests.sessions.Session.request",
        "httpx.Client.send",
        "httpx.AsyncClient.send",
        "subprocess.run",
        "subprocess.Popen",
        "screening.checks.llm_json",
    ):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr(
        "screening.checks.resolve_llm_config",
        lambda: LLMConfig(provider="mock", model="mock", base_url=None, api_key=None),
    )
    captures = Path(os.environ.get("FACTREVIEW_RECIPE_FACTS_CAPTURE", str(tmp_path / "capture")))
    captures.mkdir(parents=True, exist_ok=True)
    for mode in ("confirmed", "sample_unresolved", "resource_changed"):
        work = tmp_path / mode
        work.mkdir()
        claim, materials, _, _ = inputs(work)
        original = copy.deepcopy((claim.model_dump(mode="json"), materials.model_dump(mode="json")))
        calls, selected, responses = [], [], []

        def model(**kw):
            payload = json.loads(kw["prompt"])
            calls.append({"module": kw["module"], "payload": copy.deepcopy(payload)})
            (captures / f"{mode}-{len(calls)}-request.json").write_text(
                json.dumps(calls[-1], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            context = payload["execution_choice_context"]
            candidate = next(iter(context["candidates"].values()))
            assert context["version"] == "released-predictions-choice-context-v2"
            assert context["registry_version"] == "released-predictions-choice-registry-v2"
            assert context["choice_wire_version"] == "released-predictions-choice-v1"
            facts = candidate["repository_recipe_facts"]
            assert facts["version"] == "released-predictions-recipe-facts-v2"
            assert facts["recipe"] == "exact_match_accuracy"
            assert facts["rule_version"] == "entry-recipe-full-list-v1"
            assert facts["recipe_sha256"] == hashlib.sha256(
                Path("src/verification/execution_projection.py").read_bytes()
            ).hexdigest()
            assert set(facts["resources"]) == {"entry", "configuration", "data"}
            root = Path(materials.repository.root)
            for role, key in (("entry", "entry_script"), ("configuration", "config_path"), ("data", "data_path")):
                path = candidate[key]
                assert facts["resources"][role] == {
                    "path": path,
                    "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
                }
            assert facts["configuration"] == json.loads((root / candidate["config_path"]).read_text("utf-8"))
            assert facts["whole_rows_count"] == facts["denominator"] == len(
                json.loads((root / candidate["data_path"]).read_text("utf-8"))
            ) == 4
            assert facts["unfiltered"] is True
            assert (facts["label_key"], facts["prediction_key"]) == ("label", "prediction")
            assert not {"numerator", "accuracy", "measured_accuracy", "value"} & set(facts)
            assert len(candidate["obligations"]) == 19
            assert candidate["condition"] == original[0]["conditions"][0]
            assert candidate["proposal"]["claim_bindings"][0]["end"] == len(claim.text)
            if kw["module"] == "verification.experiments":
                selected.append(candidate)
                raw = dict(
                    schema_version="experiments-execution-v3",
                    checked_aspects=["correspondence", "fairness", "isolation", "stability", "consistency"],
                    items=[],
                    execution={"kind": "choice", "selection": select(candidate)},
                )
            else:
                assert kw["module"] == "verification.experiments.scope"
                assert candidate == selected[0]
                decision = review(candidate)  # Explicit mock semantic decisions, independent of facts.
                if mode == "sample_unresolved":
                    sample_id = next(o["obligation_id"] for o in candidate["obligations"] if o["kind"] == "atom" and o["binding"]["kind"] == "sample_scope")
                    next(r for r in decision["reviews"] if r["obligation_id"] == sample_id)["decision"] = "unresolved"
                if mode == "resource_changed":
                    path = root / candidate["data_path"]
                    rows = json.loads(path.read_text("utf-8"))
                    rows[-1]["prediction"] = "changed"
                    path.write_text(json.dumps(rows), encoding="utf-8")
                raw = dict(schema_version="catalog-v2", conditions=[], items=[], execution_choice_reviews=[decision])
            responses.append(copy.deepcopy(raw))
            return raw

        with run_scope(work / "stats.json"):
            result = verify_experiments(claim, materials, call=model, scope_binding_repair_rounds=0)
        assert len(calls) == 2
        assert original == (claim.model_dump(mode="json"), materials.model_dump(mode="json"))
        assert bool(result.plans) is (mode == "confirmed"), result.issues
        assert result.evidence == []
        if mode == "confirmed":
            plan = result.plans[0]
            assert plan.target_conditions == claim.conditions and plan.y_paper == {"c1": 0.75}
            assert plan.target_bindings["c1"].projection.record_version == "released-predictions-choice-v1"
            validate_plan_targets(plan, claim, materials)
            audit = json.loads(Path(plan.target_bindings["c1"].projection.scope_audit).read_text("utf-8"))
            assert audit["first_pass_response"] == responses[0] and audit["response"] == responses[1]
            registry = audit["choice_registry"]
            assert registry["version"] == "released-predictions-choice-registry-v2"
            revalidate_registry(registry, claim, materials)
            changed = copy.deepcopy(registry)
            next(iter(changed["candidates"].values()))["repository_recipe_facts"]["unfiltered"] = False
            changed["registry_sha256"] = digest({k: v for k, v in changed.items() if k != "registry_sha256"})
            with pytest.raises(ProjectionError):
                revalidate_registry(changed, claim, materials)
            historical = copy.deepcopy(registry)
            historical["version"] = "released-predictions-choice-v1"
            for row in historical["candidates"].values():
                row.pop("repository_recipe_facts")
            historical["registry_sha256"] = digest({k: v for k, v in historical.items() if k != "registry_sha256"})
            assert choice_context(historical)["version"] == "released-predictions-choice-v1"
            with pytest.raises(ProjectionError, match="Historical"):
                revalidate_registry(historical, claim, materials)
