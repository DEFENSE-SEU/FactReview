"""Offline full-stack v2 tests with real stages and mocked service boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pymupdf
import pytest

import pipeline_full
import pipeline_v2
from common import run_stats
from fact_generation.execution.v2 import Observation, RunOutcome
from llm.client import LLMConfig
from preprocessing.parse.mineru_adapter import MineruParseResult
from verification import literature

ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]


@pytest.fixture(autouse=True)
def offline_boundaries(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline pipeline test attempted network access or an external process")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)

    def cfg():
        return LLMConfig("mock", "fixture", None, None)

    monkeypatch.setattr("screening.claims.resolve_llm_config", cfg)
    monkeypatch.setattr("screening.checks.resolve_llm_config", cfg)
    monkeypatch.setattr(literature, "resolve_llm_config", cfg)
    monkeypatch.setattr("screening.claims.llm_json", forbidden)
    monkeypatch.setattr("screening.checks.llm_json", forbidden)
    monkeypatch.setattr(literature, "llm_json", forbidden)
    monkeypatch.setattr("fact_generation.execution.v2.docker_runner", forbidden)


class MockMinerU:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def parse_pdf(self, *, pdf_path, data_id):
        assert Path(pdf_path).is_file()
        self.calls.append((pdf_path, data_id))
        return self.result


@pytest.fixture
def tiny_inputs(tmp_path):
    body = (
        "A test MRR is 0.4. We use Adam. Our novel graph neural network improves link prediction. "
        "The identity theorem states x = x. Proof. x = x by reflexivity. Figure 1 shows our model."
    )
    rows = [
        {
            "type": "text",
            "text": "# Graph neural network for link prediction",
            "text_level": 1,
            "page_idx": 0,
        },
        {"type": "text", "text": "## Abstract", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": body, "page_idx": 0},
        {"type": "table", "text": "Method | Time (s)\nTiny | 0.1", "page_idx": 0},
        {
            "type": "image",
            "image_caption": "Figure 1: Tiny model.",
            "page_idx": 0,
            "bbox": [75, 400, 400, 520],
        },
        {"type": "text", "text": "## References", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "A. Author. Foundational prior. 2020.", "page_idx": 0},
    ]
    markdown = "\n\n".join(row.get("text", row.get("image_caption", "")) for row in rows)
    paper = tmp_path / "paper.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=400, height=500)
        page.insert_textbox((25, 25, 375, 180), body, fontsize=10)
        page.draw_rect((30, 200, 160, 260))
        page.insert_text((40, 230), "Tiny model")
        pdf.save(paper)
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "eval.py").write_text("optimizer = 'Adam'\nprint('fixture evaluation')\n", encoding="utf-8")
    (repository / "data.csv").write_text("id,value\n1,0.4\n", encoding="utf-8")
    (repository / "weights.pt").write_bytes(b"fixture weights; never loaded")
    parsed = MineruParseResult(markdown, rows, None, "mock", {"fixture": True}, "mock-mineru")
    args = SimpleNamespace(
        paper_pdf=str(paper),
        paper_key="tiny",
        run_root=str(tmp_path / "runs"),
        repository_root=str(repository),
        submission_deadline="2021-01-31",
        run_execution=True,
        execution_no_llm=True,
        approval_mode="auto",
        training_budget=0,
        max_attempts=3,
    )
    return args, MockMinerU(parsed)


class ModelBoundary:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        module = kwargs["module"]
        self.calls.append(module)
        if module == "screening.claims":
            payload = json.loads(kwargs["prompt"].split("\nPAPER_DATA_JSON:\n", 1)[1])
            block = next(b for b in payload["blocks"] if "A test MRR" in b["text"])
            rows = [
                (
                    "A test MRR is 0.4.",
                    [{"id": "a", "dataset": "A", "metric": "MRR", "settings": {"split": "test"}}],
                    ["Experiments"],
                ),
                ("We use Adam.", [{"id": "optimizer", "description": "Configured optimizer"}], ["Code"]),
                (
                    "Our novel graph neural network improves link prediction.",
                    [{"id": "novelty", "description": "Novel mechanism"}],
                    ["Literature"],
                ),
                (
                    "The identity theorem states x = x.",
                    [{"id": "identity", "description": "Reflexive identity"}],
                    ["Theory"],
                ),
            ]
            return {
                "status": "ok",
                "claims": [
                    {
                        "text": text,
                        "source_block_id": block["id"],
                        "source_quote": text,
                        "conditions": conditions,
                        "needs": needs,
                        "importance": "core",
                    }
                    for text, conditions, needs in rows
                ],
            }
        if module in {"screening_writing", "screening_tables", "screening_figures"}:
            if module == "screening_figures":
                assert kwargs["images"] and Path(kwargs["images"][0]).is_file()
                assert "Figure 1 shows our model." in kwargs["prompt"]
            return {"findings": []}
        if module == "verification_literature":
            data = json.loads(kwargs["prompt"].split("\nDATA_JSON:\n", 1)[1])
            return {
                "status": "ok",
                "comparisons": [
                    {
                        "paper_id": "2001.00001",
                        "purpose": "novelty",
                        "relation": "different",
                        "quote": "This fixture source studies relational database joins.",
                        "covered": ["novelty"] if data["claim"] else [],
                        "fully_supported_conditions": [],
                        "mechanism": "Database joins versus message passing.",
                        "setting": "Relational database queries versus link prediction.",
                        "protocol": "Query latency versus predictive accuracy.",
                        "note": "Fixture distinct mechanism.",
                    }
                ],
            }
        data = json.loads(kwargs["prompt"])
        blocks = data.get("paper_blocks", data.get("main_text", []))
        block = next(b for b in blocks if "A test MRR" in b["text"])
        if module == "verification.code":
            return {
                "items": [
                    {
                        "file": "eval.py",
                        "line": 1,
                        "quote": "optimizer = 'Adam'",
                        "paper_block_id": block["id"],
                        "paper_quote": "We use Adam.",
                        "covered": ["optimizer"],
                        "fully_supported_conditions": ["optimizer"],
                        "direction": "support",
                        "aspect": "optimizer",
                        "detail": "The source uses the reported optimizer.",
                    }
                ]
            }
        if module == "verification.theory":
            return {
                "items": [
                    {
                        "block_id": block["id"],
                        "quote": "Proof. x = x by reflexivity.",
                        "covered": ["identity"],
                        "fully_supported_conditions": ["identity"],
                        "kind": "derivation",
                        "direction": "support",
                        "detail": "Reflexive equality.",
                        "step_quote": "x = x",
                    }
                ]
            }
        if module == "verification.experiments":
            return {
                "checked_aspects": ASPECTS,
                "items": [
                    {
                        "aspect": "correspondence",
                        "kind": "paper_support",
                        "block_id": block["id"],
                        "quote": "A test MRR is 0.4.",
                        "covered": ["a"],
                        "fully_supported_conditions": ["a"],
                        "detail": "The paper reports the target value.",
                    }
                ],
                "plans": [
                    {
                        "targets": [
                            {
                                "condition_id": "a",
                                "reported": {
                                    "block_id": block["id"],
                                    "quote": "A test MRR is 0.4.",
                                    "token": "0.4",
                                },
                            }
                        ],
                        "entry_script": "eval.py",
                        "run_mode": "evaluation",
                        "feasibility": "ready",
                        "priority": "high",
                        "data_paths": ["data.csv"],
                        "weight_paths": ["weights.pt"],
                    }
                ],
            }
        raise AssertionError(f"Unexpected LLM module: {module}")


class RetrievalBoundary:
    def __init__(self):
        self.queries = []
        self.paper = {
            "id": "2001.00001",
            "arxiv_id": "2001.00001",
            "title": "Fixture relational database joins",
            "url": "https://arxiv.org/abs/2001.00001",
            "published": "2020-01-01",
            "abstract": "This fixture describes relational database query operations.",
        }

    async def search(self, *, query, cutoff_date):
        self.queries.append((query, cutoff_date.to_string()))
        return {"success": True, "provider": "fixture", "complete": True, "papers": [self.paper], "count": 1}

    async def read_papers(self, *, items):
        assert items[0]["id"] == self.paper["id"]
        return {
            "success": True,
            "items": [
                {
                    "id": self.paper["id"],
                    "success": True,
                    "paper": self.paper,
                    "evidence": [
                        {"page": 1, "text": "This fixture source studies relational database joins."}
                    ],
                }
            ],
        }


def run_tiny(tiny_inputs, monkeypatch, *, call=None, runner=None, render_pdf=True):
    args, parser = tiny_inputs
    model = call or ModelBoundary()
    retrieval = RetrievalBoundary()
    monkeypatch.setattr(literature, "_default_adapter", lambda: retrieval)
    runner = runner or Mock(
        return_value=RunOutcome(
            returncode=0,
            stdout="fixture execution",
            observations=[Observation(dataset="A", metric="MRR", settings={"split": "test"}, value=0.4)],
            environment={"transport": "mock-docker"},
        )
    )
    summary = pipeline_v2.run_v2_pipeline(
        args,
        parser=parser,
        call=model,
        reference_checker=lambda **kwargs: {"ok": True, "total_refs": 1, "issues": []},
        runner=runner,
        render_pdf=render_pdf,
    )
    return summary, model, retrieval, runner


def test_real_v2_stages_with_mocked_external_services_render_pdf_and_svg(tiny_inputs, monkeypatch):
    summary, model, retrieval, runner = run_tiny(tiny_inputs, monkeypatch)
    assert list(summary["stages"]) == list(pipeline_v2.STAGES)
    assert set(summary["stages"].values()) == {"ok"}
    assert summary["stage_errors"] == {}
    assert summary["counts"] == {"supported": 4, "flawed": 0, "questioned": 0, "unverified": 0}
    stats = summary["run_stats"]["modules"]
    assert stats["reference_check"]["status"] == "ok"
    assert stats["reference_check"]["duration_sec"] > 0
    assert stats["analysis"]["duration_sec"] + stats["reference_check"]["duration_sec"] == pytest.approx(
        sum(summary["stage_durations_sec"][stage] for stage in ("screening", "verification", "assessment"))
    )
    assert summary["submission_deadline"] == "2021-01-31"
    assert summary["concurrent_start"] == "2020-10-31"
    assert len(tiny_inputs[1].calls) == 1 and runner.call_count == 1
    prepared = json.loads(Path(summary["outputs"]["materials"]).read_text(encoding="utf-8"))
    assert prepared["figures"][0]["bbox_points"] == [30, 200, 160, 260]
    assert {
        "verification.theory",
        "verification.code",
        "verification.experiments",
        "verification_literature",
    }.issubset(model.calls)
    assert all(deadline == "2021-01-31" for _, deadline in retrieval.queries)
    assert model.calls.count("screening.claims") == 1
    assert not any("report" in name for name in model.calls)
    review = json.loads(Path(summary["outputs"]["report_json"]).read_text(encoding="utf-8"))
    execution_claim = next(c for c in review["claims"] if c["id"] == "claim_001")
    assert {e["source"] for e in execution_claim["evidence"]} == {"paper_internal", "execution"}
    assert next(e for e in execution_claim["evidence"] if e["source"] == "execution")["aligned"] is True
    assert review["ledger"][0]["approval_mode"] == "auto"
    assert review["ledger"][0]["training_budget"] == 0
    assert review["ledger"][0]["config"]["max_attempts"] == 3
    verification = json.loads(Path(summary["outputs"]["verification"]).read_text(encoding="utf-8"))
    assert verification["dispatched"] == {
        "claim_001": ["Experiments"],
        "claim_002": ["Code"],
        "claim_003": ["Literature"],
        "claim_004": ["Theory"],
    }
    assert Path(summary["outputs"]["teaser_image"]).read_text(encoding="utf-8").startswith("<svg")
    with pymupdf.open(summary["outputs"]["report_pdf"]) as pdf:
        assert len(pdf) >= 1 and "FactReview" in pdf[0].get_text()


def test_wrong_runtime_metadata_never_becomes_execution_support(tiny_inputs, monkeypatch):
    runner = Mock(
        return_value=RunOutcome(
            returncode=0,
            observations=[Observation(dataset="B", metric="MRR", settings={"split": "test"}, value=0.4)],
        )
    )
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, runner=runner, render_pdf=False)
    review = json.loads(Path(summary["outputs"]["report_json"]).read_text(encoding="utf-8"))
    claim = next(c for c in review["claims"] if c["id"] == "claim_001")
    assert all(e["source"] != "execution" for e in claim["evidence"])
    assert claim["status"] == "supported"  # The separate paper-internal evidence is still visible.
    assert claim["questions"] and review["ledger"][0]["alignment"]


def test_extraction_failure_stops_before_verification_and_report(tiny_inputs, monkeypatch):
    summary, _, retrieval, runner = run_tiny(
        tiny_inputs,
        monkeypatch,
        call=lambda **kwargs: {"status": "error", "error": "fixture provider offline"},
        render_pdf=False,
    )
    assert summary["stages"]["screening"] == "failed"
    assert summary["stages"]["verification"] == summary["stages"]["report"] == "skipped"
    assert "fixture provider offline" in summary["stage_errors"]["screening"]
    assert retrieval.queries == [] and runner.call_count == 0
    assert "report_json" not in summary["outputs"]


def test_same_second_runs_have_distinct_directories_and_do_not_overwrite(tiny_inputs, monkeypatch):
    monkeypatch.setattr(pipeline_v2, "make_run_id", lambda: "2026-10-08_120000")
    first, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    path = Path(first["outputs"]["report_json"])
    original = path.read_bytes()
    second, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert first["run_dir"] != second["run_dir"]
    assert path.read_bytes() == original
    assert Path(second["outputs"]["report_json"]).is_file()


def test_cli_cutoff_requires_explicit_deadline_and_never_defaults_to_arxiv(monkeypatch):
    source = "https://arxiv.org/abs/1911.03082"
    monkeypatch.setattr(sys, "argv", ["factreview", source])
    args = pipeline_full.parse_args()
    assert args.submission_deadline == ""
    assert pipeline_full._resolve_cutoff(args=args, paper_source=source) is None
    monkeypatch.setattr(sys, "argv", ["factreview", source, "--submission-deadline", "2021-01-31"])
    args = pipeline_full.parse_args()
    assert pipeline_full._resolve_cutoff(args=args, paper_source=source).to_string() == "2021-01-31"
    with pytest.raises(ValueError):
        args.submission_deadline = "2021-02-30"
        pipeline_full._resolve_cutoff(args=args, paper_source=source)


def test_arxiv_fallback_reaches_retrieval_and_report_with_saved_provenance(tiny_inputs, monkeypatch):
    from fact_generation.positioning.paper_search import PaperSearchAdapter

    args, _ = tiny_inputs
    args.submission_deadline = ""
    args.derive_cutoff_from_arxiv = True
    args.arxiv_id = "2101.01234v3"
    lookup = AsyncMock(
        return_value={
            "success": True,
            "paper": {
                "arxiv_id": "2101.01234v4",
                "published": "2021-01-07T00:00:00Z",
                "updated": "2025-08-01T00:00:00Z",
            },
        }
    )
    monkeypatch.setattr(PaperSearchAdapter, "lookup_metadata", lookup)
    summary, _, retrieval, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert set(summary["stages"].values()) == {"ok"}
    assert summary["submission_deadline"] == "2021-01-07"
    assert summary["concurrent_start"] == "2020-10-07"
    assert {date for _, date in retrieval.queries} == {"2021-01-07"}
    lookup.assert_awaited_once_with(identifier="2101.01234")
    assert json.loads(Path(summary["outputs"]["cutoff"]).read_text(encoding="utf-8")) == summary["cutoff"]
    assert summary["cutoff"]["source"] == "arxiv_first_submission"
    assert "venue deadline is unknown" in Path(summary["outputs"]["report_markdown"]).read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "explicit,expected,source",
    [
        ("submission_deadline", "2021-02-02", "submission_deadline"),
        ("cutoff_date", "2021-03-04", "explicit_cutoff"),
    ],
)
def test_explicit_cutoff_precedes_arxiv_fallback(tiny_inputs, monkeypatch, explicit, expected, source):
    args, _ = tiny_inputs
    args.submission_deadline = ""
    setattr(args, explicit, expected)
    args.derive_cutoff_from_arxiv = True
    lookup = AsyncMock(side_effect=AssertionError("An explicit cutoff must prevent metadata lookup"))
    monkeypatch.setattr(pipeline_v2, "resolve_arxiv_first_submission", lookup)
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert set(summary["stages"].values()) == {"ok"}
    assert summary["cutoff"]["source"] == source
    assert summary["submission_deadline"] == expected
    lookup.assert_not_called()


@pytest.mark.parametrize("opt_in", [True, False])
def test_missing_cutoff_keeps_other_stages_and_reports_limits(tiny_inputs, monkeypatch, opt_in):
    args, _ = tiny_inputs
    args.submission_deadline = ""
    args.derive_cutoff_from_arxiv = opt_in
    lookup = AsyncMock(side_effect=RuntimeError("metadata unavailable"))
    monkeypatch.setattr(pipeline_v2, "resolve_arxiv_first_submission", lookup)
    summary, _, retrieval, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert set(summary["stages"].values()) == {"ok"}
    assert "submission_deadline" not in summary
    assert summary["cutoff"]["source"] == "unresolved"
    assert not retrieval.queries
    assert lookup.await_count == int(opt_in)
    assert any("Submission deadline missing" in issue for issue in summary["issues"])
    if opt_in:
        assert "metadata unavailable" in summary["cutoff"]["error"]


def test_invalid_explicit_cutoff_is_a_durable_pipeline_failure(tiny_inputs, monkeypatch):
    args, _ = tiny_inputs
    args.submission_deadline = "2021-02-30"
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["stages"]["materials"] == "failed"
    assert all(value == "skipped" for name, value in summary["stages"].items() if name != "materials")
    saved = json.loads((Path(summary["run_dir"]) / "full_pipeline_summary.json").read_text(encoding="utf-8"))
    assert saved["stage_errors"] == summary["stage_errors"]


def test_compgcn_fixture_replay_preserves_reference_artifacts_and_records_limits(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts" / "check_v2_compgcn.py"
    spec = importlib.util.spec_from_file_location("compgcn_fixture_script", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparison = module.replay_compgcn(tmp_path / "compgcn")
    assert comparison["live_run_completed"] is False
    assert comparison["reference_hashes_unchanged"]
    assert "fixtures" in module.__doc__
    assert comparison["comparison_limits"] and comparison["submission_deadline"] is None
    assert comparison["fixture_counts"] == {"supported": 2, "flawed": 0, "questioned": 0, "unverified": 1}
    assert set(comparison["summary"]["stages"].values()) == {"ok"}
    root = Path(comparison["summary"]["run_dir"])
    assert (root / "comparison.json").is_file() and (root / "comparison.md").is_file()
    ledger = json.loads((root / "execution" / "ledger.json").read_text(encoding="utf-8"))
    assert len(ledger) == 1 and ledger[0]["approved"] is False
    assert ledger[0]["attempts"] == []
    assert "Unresolved paper target" in ledger[0]["reason"]
    assert len(comparison["old_report"]) == 3
    actual_pdf = module.ROOT / "demos" / "Graph" / "compgcn" / "paper.pdf"
    assert comparison["reference_hashes"]["paper.pdf"] == hashlib.sha256(actual_pdf.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("file_repairs", "cli_flags", "expected_repairs", "expected_budget"),
    [
        (0, [], 0, 2),
        (3, ["--max-attempts", "0", "--training-budget", "0"], 0, 0),
    ],
)
def test_execution_config_loads_and_only_explicit_cli_options_override_it(
    tiny_inputs,
    monkeypatch,
    tmp_path,
    file_repairs,
    cli_flags,
    expected_repairs,
    expected_budget,
):
    original, parser = tiny_inputs
    config_path = tmp_path / "execution-config.json"
    config_path.write_text(
        json.dumps(
            {
                "max_attempts": file_repairs,
                "training_budget": 2,
                "approval_mode": "auto",
                "refine_with_llm": False,
                "timeout_seconds": 41,
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "factreview",
        original.paper_pdf,
        "--run-root",
        original.run_root,
        "--paper-key",
        "config",
        "--repository-root",
        original.repository_root,
        "--submission-deadline",
        original.submission_deadline,
        "--run-execution",
        "--execution-config",
        str(config_path),
        *cli_flags,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = pipeline_full.parse_args()
    summary, _, _, runner = run_tiny((args, parser), monkeypatch, render_pdf=False)
    assert summary["stage_errors"] == {}
    request = runner.call_args.args[0]
    assert request.config.max_attempts == expected_repairs
    assert request.config.training_budget == expected_budget
    assert request.config.timeout_seconds == 41
    assert request.config.refine_with_llm is False
    ledger = json.loads((Path(summary["run_dir"]) / "execution" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger[0]["config"]["max_attempts"] == expected_repairs
    assert ledger[0]["training_budget"] == expected_budget


def test_invalid_execution_config_is_recorded_as_execution_failure(tiny_inputs, monkeypatch, tmp_path):
    args, _ = tiny_inputs
    config = tmp_path / "invalid-config.json"
    config.write_text('{"max_attempts":4}', encoding="utf-8")
    args.execution_config = str(config)
    summary, _, _, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["stages"]["execution"] == "failed"
    assert "max_attempts" in summary["stage_errors"]["execution"]
    assert runner.call_count == 0 and summary["stages"]["report"] == "skipped"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("teaser_mode", "api"),
        ("reuse_job_id", "old-job"),
        ("execution_auto_tasks", True),
        ("execution_auto_tasks_force", True),
        ("execution_paper_budget_sec", 100),
        ("no_cutoff", True),
    ],
)
def test_unsupported_legacy_options_are_saved_as_failures(tiny_inputs, monkeypatch, option, value):
    args, parser = tiny_inputs
    setattr(args, option, value)
    summary, _, _, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["stages"]["materials"] == "failed"
    assert "Unsupported v2 options" in summary["stage_errors"]["materials"]
    assert option in summary["stage_errors"]["materials"]
    assert parser.calls == [] and runner.call_count == 0
    saved = json.loads((Path(summary["run_dir"]) / "full_pipeline_summary.json").read_text(encoding="utf-8"))
    assert saved["stage_errors"] == summary["stage_errors"]


def test_run_statistics_environment_is_restored_after_success_and_failure(tiny_inputs, monkeypatch, tmp_path):
    parent = tmp_path / "parent-stats.json"
    parent.write_text('{"parent":true}', encoding="utf-8")
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(parent))
    summary, _, _, _ = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["stage_errors"] == {}
    assert os.environ["FACTREVIEW_RUN_STATS_PATH"] == str(parent)
    assert parent.read_text(encoding="utf-8") == '{"parent":true}'
    monkeypatch.delenv("FACTREVIEW_RUN_STATS_PATH")
    summary, _, _, _ = run_tiny(
        tiny_inputs,
        monkeypatch,
        call=lambda **kwargs: {"status": "error", "error": "intentional fixture failure"},
        render_pdf=False,
    )
    assert summary["stages"]["screening"] == "failed"
    assert "FACTREVIEW_RUN_STATS_PATH" not in os.environ
    assert summary["run_stats"]["modules"]["analysis"]["status"] == "failed"
    assert summary["run_stats"]["modules"]["report_generation"]["status"] == "skipped"


def test_missing_parser_failure_is_consistent_in_summary_and_stats(tiny_inputs, monkeypatch):
    _args, parser = tiny_inputs

    async def failed_parse(**kwargs):
        raise RuntimeError("MinerU credentials unavailable (fixture)")

    parser.parse_pdf = failed_parse
    summary, _, _, runner = run_tiny(tiny_inputs, monkeypatch, render_pdf=False)
    assert summary["stages"]["materials"] == "failed"
    assert summary["run_stats"]["modules"]["parse"]["status"] == "failed"
    assert summary["run_stats"]["modules"]["parse"]["duration_sec"] > 0
    assert summary["run_stats"]["modules"]["analysis"]["status"] == "skipped"
    assert summary["stage_durations_sec"]["materials"] > 0
    assert runner.call_count == 0


@pytest.mark.parametrize("first_fails", [False, True])
def test_concurrent_pipelines_keep_usage_and_parent_context_isolated(
    tiny_inputs, monkeypatch, tmp_path, first_fails
):
    parent = tmp_path / "parent-stats.json"
    parent.write_text('{"parent":true}', encoding="utf-8")
    monkeypatch.setenv("FACTREVIEW_RUN_STATS_PATH", str(parent))
    monkeypatch.setenv("FACTREVIEW_ACTIVE_STATS_MODULE", "parse")
    first_parsing, second_parsing, first_finished = Event(), Event(), Event()
    original_args, original_parser = tiny_inputs

    def run(index):
        args = SimpleNamespace(**vars(original_args))
        args.paper_key = f"parallel-{index}"
        args.run_execution = False
        boundary = ModelBoundary()

        class InterleavedParser:
            async def parse_pdf(self, **kwargs):
                if index == 11:
                    first_parsing.set()
                    assert second_parsing.wait(timeout=10)
                else:
                    second_parsing.set()
                    assert first_finished.wait(timeout=20)
                return original_parser.result

        def model(**kwargs):
            if kwargs["module"] == "screening.claims":
                run_stats.record_llm_call(usage={"input_tokens": index}, model=f"run-{index}")
                if first_fails and index == 11:
                    return {"status": "error", "error": "intentional first-run failure"}
            return boundary(**kwargs)

        try:
            return pipeline_v2.run_v2_pipeline(
                args,
                parser=InterleavedParser(),
                call=model,
                reference_checker=lambda **kwargs: {"ok": True, "total_refs": 1, "issues": []},
                branches={name: lambda *args: {} for name in ("Literature", "Theory", "Code", "Experiments")},
                global_literature=lambda *args: {},
                render_pdf=False,
            )
        finally:
            if index == 11:
                first_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, 11)
        assert first_parsing.wait(timeout=10)
        second = pool.submit(run, 22)
        results = (first.result(timeout=30), second.result(timeout=30))
    for index, summary in zip((11, 22), results, strict=True):
        assert summary["run_stats"]["total"]["token_usage"]["input_tokens"] == index
        assert summary["run_stats"]["modules"]["analysis"]["models"] == {f"run-{index}": 1}
        failed = first_fails and index == 11
        assert summary["stages"]["screening"] == ("failed" if failed else "ok")
        assert summary["stages"]["report"] == ("skipped" if failed else "ok")
        if not failed:
            assert Path(summary["outputs"]["report_json"]).is_file()
    assert parent.read_text(encoding="utf-8") == '{"parent":true}'
    assert os.environ["FACTREVIEW_RUN_STATS_PATH"] == str(parent)
    assert os.environ["FACTREVIEW_ACTIVE_STATS_MODULE"] == "parse"
