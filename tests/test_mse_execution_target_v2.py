"""Source-grounded absolute regression targets retain scalar identity guards."""

import pytest

from schemas.claim import Claim, ClaimLocation, Condition, PaperTargetPassage
from schemas.materials import MaterialBlock, SharedMaterials
from verification.experiment_targets import TargetBindingError, bind_execution_target


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("MSE target controls forbid external services, processes and author execution")
    for name in ("llm.client.llm_json", "subprocess.Popen", "subprocess.run",
                 "requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send",
                 "fact_generation.execution.v2.docker_runner"):
        monkeypatch.setattr(name, forbidden)


def test_source_grounded_mse_names_preserve_provenance_quantity_and_condition_guards(tmp_path):
    def inputs(metric="mse", prose=None):
        quote = prose or f"On Toy test, model Linear reports {metric} 1."
        loc = ClaimLocation(page=1, char_start=0, char_end=len(quote))
        condition = Condition(id="c1", dataset="Toy", metric=metric,
                              settings={"model": "Linear", "split": "test"})
        claim = Claim(id="c", text=quote, loc=loc, source_block_id="b", source_quote=quote,
                      conditions=[condition], needs=["Experiments"])
        paper = tmp_path / "paper.md"
        paper.write_text(quote, encoding="utf-8")
        materials = SharedMaterials(paper_key="p", source_pdf=str(tmp_path / "absent.pdf"), markdown=quote,
            markdown_path=str(paper), content_list_path="content.json", provider="fixture",
            blocks=[MaterialBlock(id="b", text=quote, loc=loc)])
        passage = PaperTargetPassage(block_id="b", quote=quote, token="1", value_context=quote)
        return claim, condition, passage, materials

    for metric in ("mse", "mean squared error"):
        claim, condition, passage, materials = inputs(metric)
        before = claim.model_dump(mode="json")
        binding = bind_execution_target(claim, condition, passage, materials)
        assert binding.value == 1 and binding.condition_id == "c1"
        assert binding.reported == passage and binding.pointer.quote == claim.source_quote
        assert binding.pointer.locator == materials.markdown_path
        assert claim.model_dump(mode="json") == before
    for failure in ("missing_source", "unknown_residual", "comparative", "split", "model", "dataset", "different_quantity"):
        claim, condition, passage, materials = inputs(
            "residual" if failure == "unknown_residual" else "mse",
            "On Toy test, model Linear reports mse improvement 1." if failure == "comparative" else None,
        )
        if failure == "missing_source":
            (tmp_path / "paper.md").unlink()
        elif failure == "split":
            condition.settings["split"] = "train"
        elif failure == "model":
            condition.settings["model"] = "Other"
        elif failure == "dataset":
            condition.dataset = "Other"
        elif failure == "different_quantity":
            condition.metric = "accuracy"
        with pytest.raises(TargetBindingError):
            bind_execution_target(claim, condition, passage, materials)
