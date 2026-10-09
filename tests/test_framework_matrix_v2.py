"""Run the reusable matrix with real v2 stages and fixed external boundaries."""

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pymupdf
import pytest

from fact_generation.execution import v2 as execution
from util.subprocess_runner import CommandResult


@pytest.fixture(scope="module")
def script():
    path = Path(__file__).resolve().parents[1] / "scripts/check_v2_framework.py"
    spec = importlib.util.spec_from_file_location("framework_matrix_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def matrix(script, tmp_path_factory):
    # Running once retains actual report/PDF/image files for every assertion.
    return script.run_matrix(tmp_path_factory.mktemp("framework-matrix"))


@pytest.mark.parametrize(
    "name",
    [
        "theory_appendix",
        "figures_partial",
        "missing_repository",
        "mapped_runtime",
        "mapped_runtime_misaligned",
    ],
)
def test_complete_case_manifest_and_exported_artifacts(matrix, name):
    case = next(row for row in matrix["cases"] if row["scenario"] == name)
    assert matrix["passed"] and case["passed"], case
    assert all(case["checks"].values())
    assert case["observed_statuses"] == case["expected_statuses"]
    summary = json.loads(Path(case["paths"]["summary"]).read_text(encoding="utf-8"))
    assert list(summary["stages"]) == [
        "materials",
        "screening",
        "verification",
        "execution",
        "assessment",
        "report",
        "teaser",
    ]
    assert set(summary["stages"].values()) == {"ok"}
    assert summary["stage_errors"] == {}
    assert case["boundaries"]["LLM_VLM"] == "fixed responses"
    assert case["boundaries"]["Docker"].startswith("mocked")
    assert "accuracy" in case["boundary_note"]
    for key in (
        "input_pdf",
        "parser_fixture",
        "model_calls",
        "retrieval_calls",
        "report_json",
        "report_markdown",
        "report_pdf",
        "teaser_image",
    ):
        assert Path(case["paths"][key]).is_file()
    with pymupdf.open(case["paths"]["report_pdf"]) as pdf:
        assert len(pdf) > 0 and "FactReview" in pdf[0].get_text()
    assert Path(case["paths"]["teaser_image"]).read_text(encoding="utf-8").startswith("<svg")


def test_theory_uses_appendix_without_supporting_the_unproved_claim(matrix):
    case = next(row for row in matrix["cases"] if row["scenario"] == "theory_appendix")
    review = json.loads(Path(case["paths"]["report_json"]).read_text(encoding="utf-8"))
    claims = {row["id"]: row for row in review["claims"]}
    assert claims["claim_001"]["status"] == "supported"
    assert claims["claim_001"]["evidence"][0]["pointer"]["page"] == 2
    assert claims["claim_002"]["status"] == "unverified"
    assert not claims["claim_002"]["evidence"] and claims["claim_002"]["questions"]
    calls = json.loads(Path(case["paths"]["model_calls"]).read_text(encoding="utf-8"))
    first = next(call for call in calls if call["module"] == "verification.theory")
    second = next(call for call in calls if call["module"] == "verification.theory.appendix")
    assert "appendix_proofs" not in first["input"] and second["input"]["appendix_proofs"]


def test_visual_failure_and_ambiguous_caption_reach_report_without_losing_later_findings(matrix):
    case = next(row for row in matrix["cases"] if row["scenario"] == "figures_partial")
    screen = json.loads(Path(case["paths"]["screening"]).read_text(encoding="utf-8"))
    assert [row["status"] for row in screen["figure_checks"]] == ["failed", "checked", "checked"]
    assert all(row["printed_size_verified"] for row in screen["figure_checks"])
    assert {finding["level"] for finding in screen["findings"]} == {"legibility", "self_containedness"}
    calls = json.loads(Path(case["paths"]["model_calls"]).read_text(encoding="utf-8"))
    visual = [call for call in calls if call["module"] == "screening_figures"]
    assert len(visual) == 3 and all(len(call["images"]) == 1 for call in visual)
    assert "error" in visual[0] and visual[1]["input"]["caption_ambiguous"]
    assert all(call["input"]["references"] for call in visual)
    report = Path(case["paths"]["report_markdown"]).read_text(encoding="utf-8")
    assert "Figure screening is incomplete" in report
    review = json.loads(Path(case["paths"]["report_json"]).read_text(encoding="utf-8"))
    assert review["claims"][0]["status"] == "unverified"
    assert review["claims"][0]["evidence"][0]["covered"] == ["alpha"]


def test_figure_fixture_prints_the_original_caption_and_uses_one_bound_context(matrix):
    case = next(row for row in matrix["cases"] if row["scenario"] == "figures_partial")
    assert case["fixture_version"] == "printed-caption-v2"
    assert "omitted" in case["fixture_source_correction"]
    fixture = json.loads(Path(case["paths"]["parser_fixture"]).read_text(encoding="utf-8"))
    figure_rows = [row for row in fixture["content_list"] if row["type"] == "image"]
    assert figure_rows[-1]["bbox"] == [680, 450, 880, 650]
    with pymupdf.open(case["paths"]["input_pdf"]) as pdf:
        blocks = pdf[0].get_text("blocks")
        caption = figure_rows[-1]["image_caption"]
        matches = [block for block in blocks if block[4].split() == caption.split()]
        assert len(matches) == 1
        assert matches[0][0] >= 408 and matches[0][2] <= 528 and matches[0][1] > 520
        page_tokens = " ".join(pdf[0].get_text().split())
        assert all(
            text in page_tokens
            for row in figure_rows
            for text in (
                row["image_caption"] if isinstance(row["image_caption"], list) else [row["image_caption"]]
            )
        )
    calls = json.loads(Path(case["paths"]["model_calls"]).read_text(encoding="utf-8"))
    context = [call for call in calls if call["module"] == "screening_figures.context"]
    assert len(context) == 1 and len(context[0]["images"]) == 2
    assert context[0]["input"]["figure_id"] == "figure_3"
    assert context[0]["response"]["decisions"][0]["classification"] == "manuscript_issue"


def _figure_materials_with_sources(script, tmp_path, *, omit_pdf_captions=False):
    from preprocessing.materials import build_materials

    pdf_path, parsed, claims, _ = script.make_inputs("figures_partial", tmp_path)
    if omit_pdf_captions:
        # Recreate the earlier source inconsistency locally; generated historical
        # PDFs are never edited. Crop panels and parser/Markdown stay unchanged.
        with pymupdf.open(pdf_path) as pdf:
            page = pdf[0]
            page.add_redact_annot((0, 525, 600, 600))
            page.apply_redactions()
            old = tmp_path / "legacy_without_printed_captions.pdf"
            pdf.save(old)
        pdf_path = old
    return build_materials(
        parsed, paper_pdf=pdf_path, output_dir=tmp_path / "materials", paper_key="figures_partial"
    ), claims


def test_old_missing_pdf_caption_cannot_be_rescued_by_new_context_mock(script, tmp_path, monkeypatch):
    from llm.client import LLMConfig
    from screening.figures import check_figures

    monkeypatch.setattr(
        "screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    materials, claims = _figure_materials_with_sources(script, tmp_path, omit_pdf_captions=True)
    model = script.FixedModel("figures_partial", claims)
    records = []
    findings, _ = check_figures(materials, call=model, records=records, recover_errors=True)
    assert {finding.level for finding in findings} == {"legibility"}
    assert records[-1].context_status == "unavailable"
    assert "absent" in " ".join(records[-1].issues)
    assert all(call["module"] != "screening_figures.context" for call in model.calls)


def test_context_fixture_cannot_borrow_the_neighbor_panel_caption(script, tmp_path, monkeypatch):
    from llm.client import LLMConfig
    from screening.figures import check_figures

    monkeypatch.setattr(
        "screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    materials, claims = _figure_materials_with_sources(script, tmp_path)
    model = script.FixedModel("figures_partial", claims)

    def call(**kwargs):
        response = model(**kwargs)
        if kwargs["module"] == "screening_figures.context":
            payload = json.loads(kwargs["prompt"])
            foreign = next(
                span["id"] for span in payload["page_spans"] if span["text"].startswith("Figure 1:")
            )
            response["decisions"][0]["witness_span_ids"] = [foreign]
        return response

    records = []
    findings, _ = check_figures(materials, call=call, records=records, recover_errors=True)
    assert {finding.level for finding in findings} == {"legibility"}
    assert records[-1].context_status == "failed"
    assert "another figure" in " ".join(records[-1].issues)


def test_missing_repository_retains_blocked_plan_and_nondeciding_reason(matrix):
    case = next(row for row in matrix["cases"] if row["scenario"] == "missing_repository")
    review = json.loads(Path(case["paths"]["report_json"]).read_text(encoding="utf-8"))
    assert all(claim["status"] == "unverified" and claim["questions"] for claim in review["claims"])
    ledger = review["ledger"][0]
    assert ledger["plan"]["feasibility"] == "blocked" and not ledger["approved"] and not ledger["attempts"]
    explanatory = [e for claim in review["claims"] for e in claim["evidence"]]
    assert explanatory and all(not e["sufficient"] and not e["affects_claim"] for e in explanatory)


@pytest.mark.parametrize(
    "name,split,status",
    [
        ("mapped_runtime", "test", "supported"),
        ("mapped_runtime_misaligned", "validation", "unverified"),
    ],
)
def test_native_runtime_metadata_controls_alignment_and_training_never_runs(matrix, name, split, status):
    case = next(row for row in matrix["cases"] if row["scenario"] == name)
    review = json.loads(Path(case["paths"]["report_json"]).read_text(encoding="utf-8"))
    analysis, training = review["ledger"]
    attempt = analysis["attempts"][0]
    raw = json.loads(Path(attempt["logs"]["raw_output"]).read_text())
    mapping = json.loads(Path(attempt["logs"]["output_mapping"]).read_text())
    assert raw == {
        "context": {"dataset": "FixtureVision", "split": split},
        "measurements": {"accuracy": 0.75},
    }
    assert mapping["selectors"][0]["metric_path"] == ["measurements", "accuracy"]
    assert attempt["observations"][0]["settings"] == {"split": split}
    assert attempt["commands"][0][:3] == ["docker", "run", "--rm"]
    assert next(claim for claim in review["claims"] if claim["id"] == "claim_001")["status"] == status
    assert not training["attempts"] and training["reason"] == "training budget exhausted"
    assert training["training_budget"] == 0


def test_wrong_runtime_value_fails_the_matrix_instead_of_copying_expected_result(
    script, tmp_path, monkeypatch
):
    original = script.mock_docker_command

    def changed(command, **kwargs):
        result = original(command, **kwargs)
        payload = json.loads(result.stdout)
        payload["measurements"]["accuracy"] = 0.1
        return CommandResult(
            result.cmd, result.cwd, result.returncode, json.dumps(payload), result.stderr, result.duration_sec
        )

    monkeypatch.setattr(script, "mock_docker_command", changed)
    decoder = Mock(wraps=execution.decode_output)
    monkeypatch.setattr(execution, "decode_output", decoder)
    manifest = script.run_matrix(tmp_path, scenarios=["mapped_runtime"])
    assert not manifest["passed"]
    assert decoder.call_count == 1
    assert decoder.call_args.args[0]["measurements"]["accuracy"] == 0.1
    case = manifest["cases"][0]
    assert case["observed_statuses"]["claim_001"] == "questioned"
    assert not case["checks"]["claim_statuses"]
    assert Path(manifest["root"], "manifest.json").is_file()


def test_cli_reports_failure_exit_code_and_preserves_manifest_location(script, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        script, "run_matrix", Mock(return_value={"root": str(tmp_path), "passed": False, "cases": []})
    )
    assert script.main(["--scenario", "mapped_runtime", "--docker"]) == 1
    assert "MATRIX_MANIFEST=" in capsys.readouterr().out
    assert script.run_matrix.call_args.kwargs == {"scenarios": ["mapped_runtime"], "real_docker": True}


def test_user_visual_configuration_cannot_change_fixed_offline_boundaries(script, tmp_path, monkeypatch):
    visual = {
        "VLM_MODEL_PROVIDER": "openai-codex",
        "VLM_MODEL": "external-vision-model",
        "VLM_BASE_URL": "https://example.invalid/v1",
        "VLM_API_KEY": "fixture-key-not-a-credential",
    }
    for name, value in visual.items():
        monkeypatch.setenv(name, value)
    result = script.run_matrix(tmp_path, scenarios=["figures_partial"])
    assert result["passed"]
    calls = json.loads(Path(result["cases"][0]["paths"]["model_calls"]).read_text(encoding="utf-8"))
    assert all(call["model_config"] == {"provider": "mock", "model": "framework-fixture"} for call in calls)
    assert {name: os.environ[name] for name in visual} == visual
