"""Manuscript dependency URLs require explicit released-repository binding."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf
import pytest

import pipeline_v2
from llm.client import LLMConfig
from preprocessing.parse.mineru_adapter import MineruParseResult
from verification.contracts import BranchResult


@pytest.fixture
def run_bound_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "screening.claims.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    monkeypatch.setattr(
        "screening.checks.resolve_llm_config", lambda: LLMConfig("mock", "fixture", None, None)
    )
    pdf_path = tmp_path / "paper.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 50), "The implementation uses Adam.")
        pdf.save(pdf_path)

    def run(urls, *, repository_root="", repository_url=""):
        text = "The implementation uses Adam."
        rows = [
            {"type": "text", "text": "# Fixture", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": text, "page_idx": 0},
            {"type": "text", "text": "Dependencies and baselines: " + " ".join(urls), "page_idx": 0},
        ]
        markdown = "\n\n".join(row["text"] for row in rows)

        class Parser:
            async def parse_pdf(self, **kwargs):
                return MineruParseResult(markdown, rows, None, "fixture", {}, "fixture")

        def call(**kwargs):
            if kwargs["module"] == "screening.claims":
                return {
                    "status": "ok",
                    "claims": [
                        {
                            "text": text,
                            "source_block_id": "block_2",
                            "source_quote": text,
                            "conditions": [{"id": "optimizer", "description": "Adam optimizer"}],
                            "needs": ["Code"],
                            "importance": "core",
                        }
                    ],
                }
            return {"findings": []}

        received = []

        def code_branch(claim, materials):
            received.append(materials.repository)
            return BranchResult()

        summary = pipeline_v2.run_v2_pipeline(
            SimpleNamespace(
                paper_pdf=str(pdf_path),
                paper_key="repo-binding",
                run_root=str(tmp_path / "runs"),
                submission_deadline="2021-01-31",
                run_execution=False,
                repository_root=str(repository_root),
                repository_url=repository_url,
            ),
            parser=Parser(),
            call=call,
            branches={"Code": code_branch},
            global_literature=lambda *args: BranchResult(),
            render_pdf=False,
        )
        assert summary["stages"]["report"] == "ok", summary["stage_errors"]
        materials = json.loads(Path(summary["outputs"]["materials"]).read_text(encoding="utf-8"))
        assert len(received) == 1
        return summary, materials, received[0]

    return run


@pytest.mark.parametrize(
    "urls,expected",
    [
        (["https://github.com/third-party/tokenizer"], ["https://github.com/third-party/tokenizer"]),
        (
            [
                "https://github.com/baseline/model.",
                "https://github.com/third-party/tokenizer",
                "https://github.com/third-party/tokenizer",
            ],
            ["https://github.com/baseline/model", "https://github.com/third-party/tokenizer"],
        ),
    ],
)
def test_manuscript_links_remain_unbound_candidates_without_any_clone(
    run_bound_paper, monkeypatch, urls, expected
):
    process = Mock(side_effect=AssertionError("Unbound repository links cannot trigger a clone"))
    monkeypatch.setattr(pipeline_v2.subprocess, "run", process)
    summary, materials, bound = run_bound_paper(urls)
    process.assert_not_called()
    assert summary["repository_candidates"] == expected
    assert "repository_url" not in summary
    assert materials["repository"] is None and bound is None
    assert any("have not been bound" in issue for issue in summary["issues"])


def test_explicit_repository_url_is_the_only_remote_cloned(run_bound_paper, monkeypatch):
    explicit = "https://github.com/authors/released-code"

    def clone(command, **kwargs):
        assert command[:4] == ["git", "clone", "--depth", "1"]
        assert command[-2] == explicit
        target = Path(command[-1])
        target.mkdir(parents=True)
        (target / "model.py").write_text("optimizer = 'Adam'\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    process = Mock(side_effect=clone)
    monkeypatch.setattr(pipeline_v2.subprocess, "run", process)
    summary, materials, bound = run_bound_paper(
        ["https://github.com/third-party/tokenizer"], repository_url=explicit
    )
    assert process.call_count == 1
    assert summary["repository_url"] == explicit
    assert materials["repository"]["root"] == bound.root
    assert [item.path for item in bound.files] == ["model.py"]


def test_explicit_local_repository_takes_priority_over_url(run_bound_paper, tmp_path, monkeypatch):
    repository = tmp_path / "author-repo"
    repository.mkdir()
    (repository / "local.py").write_text("optimizer = 'Adam'\n", encoding="utf-8")
    process = Mock(side_effect=AssertionError("Explicit local binding already supplies the repository"))
    monkeypatch.setattr(pipeline_v2.subprocess, "run", process)
    summary, materials, bound = run_bound_paper(
        ["https://github.com/third-party/tokenizer"],
        repository_root=repository,
        repository_url="https://github.com/authors/remote",
    )
    process.assert_not_called()
    assert materials["repository"]["root"] == str(repository.resolve()) == bound.root
    assert "repository_url" not in summary
    assert [item.path for item in bound.files] == ["local.py"]
