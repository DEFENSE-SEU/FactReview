"""Replay v2 locally with explicit fixtures over the real, unchanged CompGCN PDF.

This exercises contracts and rendering. Fixed model/retrieval responses do not
measure model quality, live MinerU behavior, or independent reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pymupdf

from pipeline_v2 import run_v2_pipeline
from preprocessing.parse.mineru_adapter import MineruParseResult
from schemas.claim import EvidenceNeed
from schemas.v1_adapter import read_v1_artifact
from verification.code import verify_code
from verification.experiments import verify_experiments
from verification.literature import verify_literature
from verification.theory import verify_theory

ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]
BOUNDARY = (
    "Offline fixture replay: PDF text extraction replaces MinerU; fixed responses replace LLM and "
    "literature services; Docker is mocked. This checks data flow and artifacts. It does not establish "
    "live parser/model accuracy or independent numerical reproduction."
)


class FixtureParser:
    def __init__(self, result: MineruParseResult, pdf_sha256: str):
        self.result, self.pdf_sha256 = result, pdf_sha256

    async def parse_pdf(self, *, pdf_path: Path, data_id: str) -> MineruParseResult:
        if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != self.pdf_sha256:
            raise ValueError("Fixture parser received a different source PDF")
        return self.result


def _pdf_fixture(pdf_path: Path) -> tuple[FixtureParser, dict[int, str]]:
    with pymupdf.open(pdf_path) as pdf:
        pages = {index + 1: page.get_text() for index, page in enumerate(pdf)}
    rows = [
        {
            "type": "text",
            "text": "# COMPOSITION-BASED MULTI-RELATIONAL GRAPH CONVOLUTIONAL NETWORKS",
            "text_level": 1,
            "page_idx": 0,
        }
    ]
    for page, text in pages.items():
        rows.extend(
            [
                {"type": "text", "text": f"## Page {page}", "text_level": 1, "page_idx": page - 1},
                {"type": "text", "text": text, "page_idx": page - 1},
            ]
        )
    parsed = MineruParseResult(
        markdown="\n\n".join(row["text"] for row in rows),
        content_list=rows,
        image_files=None,
        batch_id="offline-fixture",
        raw_result={"fixture": True},
        provider="fixture-pymupdf",
        warning=BOUNDARY + " Figure boxes and parsed bibliography are unavailable in this fixture.",
    )
    return FixtureParser(parsed, hashlib.sha256(pdf_path.read_bytes()).hexdigest()), pages


def _between(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end, text.index(start))].strip()


class CompGCNModelFixture:
    def __init__(self, pages: dict[int, str], repository: Path):
        self.relation = "hr = Wrelzr,"
        if self.relation not in pages[5]:
            raise ValueError("Expected Eq. 4 is absent from the real PDF")
        self.theorem = _between(pages[5], "Proposition 4.1.", "Proof.")
        self.proof = pages[5][pages[5].index("Proof.") :].rsplit("\n5", 1)[0].strip()
        self.table = _between(pages[6], "FB15k-237", "5\nEXPERIMENTAL SETUP")
        if ".355" not in self.table or ".479" not in self.table:
            raise ValueError("Expected Table 3 targets are absent from the real PDF")
        source = repository / "model" / "compgcn_conv.py"
        lines = source.read_text(encoding="utf-8").splitlines()
        self.code_line, self.code_quote = next(
            (i, line)
            for i, line in enumerate(lines, 1)
            if "return self.act(out), torch.matmul(rel_embed, self.w_rel)" in line
        )
        self.calls = []

    def __call__(self, **kwargs):
        name = kwargs["module"]
        self.calls.append(name)
        if name == "screening.claims":
            payload = json.loads(kwargs["prompt"].split("\nPAPER_DATA_JSON:\n", 1)[1])

            def row(text, quote, conditions, needs):
                block = next(b for b in payload["blocks"] if quote in b["text"])
                return {
                    "text": text,
                    "source_block_id": block["id"],
                    "source_quote": quote,
                    "conditions": conditions,
                    "needs": needs,
                    "importance": "core",
                }

            return {
                "status": "ok",
                "claims": [
                    row(
                        "COMPGCN transforms relation embeddings with a learned Wrel matrix.",
                        self.relation,
                        [{"id": "relation_update", "description": "Equation 4 relation embedding update"}],
                        ["Code"],
                    ),
                    row(
                        "COMPGCN generalizes Kipf-GCN, Relational-GCN, Directed-GCN, and Weighted-GCN.",
                        self.theorem,
                        [
                            {"id": name, "description": f"Reduction to {name}"}
                            for name in ("Kipf-GCN", "Relational-GCN", "Directed-GCN", "Weighted-GCN")
                        ],
                        ["Theory"],
                    ),
                    row(
                        "COMPGCN reports link prediction MRR .355 on FB15k-237 and .479 on WN18RR.",
                        self.table,
                        [
                            {"id": dataset, "dataset": dataset, "metric": "MRR"}
                            for dataset in ("FB15k-237", "WN18RR")
                        ],
                        ["Literature", "Experiments"],
                    ),
                ],
            }
        if name in {"screening_writing", "screening_tables", "screening_figures"}:
            return {"findings": []}
        payload = json.loads(kwargs["prompt"])
        blocks = payload.get("paper_blocks", payload.get("main_text", []))

        def block_id(quote):
            return next(block["id"] for block in blocks if quote in block["text"])

        if name == "verification.code":
            return {
                "items": [
                    {
                        "file": "model/compgcn_conv.py",
                        "line": self.code_line,
                        "quote": self.code_quote,
                        "paper_block_id": block_id(self.relation),
                        "paper_quote": self.relation,
                        "covered": ["relation_update"],
                        "fully_supported_conditions": ["relation_update"],
                        "direction": "support",
                        "aspect": "architecture",
                        "detail": "Fixture comparison: the released forward pass transforms relation embeddings with w_rel.",
                    }
                ]
            }
        if name == "verification.theory":
            return {
                "items": [
                    {
                        "block_id": block_id(self.proof),
                        "quote": self.proof,
                        "covered": ["Kipf-GCN"],
                        "fully_supported_conditions": ["Kipf-GCN"],
                        "kind": "derivation",
                        "direction": "support",
                        "detail": "Fixture checks the explicit Kipf-GCN reduction; the other three reductions remain unassessed.",
                        "step_quote": self.proof,
                    }
                ]
            }
        if name == "verification.experiments.scope":
            return {
                "conditions": [
                    {
                        "condition_id": condition["id"],
                        "claim_quote": payload["claim"]["text"],
                        "assertion": "descriptive",
                        "matched_controls_required": False,
                        "uncertainty_sensitive": False,
                        "relation": "none",
                        "subject": "",
                        "comparator": "",
                        "rationale": "The fixed claim reports two absolute MRR scores, with no superiority or causal qualifier.",
                    }
                    for condition in payload["claim"]["conditions"]
                ],
                "items": [
                    {
                        "item_index": index,
                        "condition_id": condition,
                        "applicability": "applicable",
                        "grounds": [{"block_id": block_id(self.table), "quote": self.table}],
                        "rationale": "The fixed Table 3 passage reports the named dataset's MRR; runtime cell targeting remains blocked separately.",
                        "full_support": True,
                        "qualifiers_complete": True,
                        "comparison_objects": "not_comparative",
                    }
                    for index, item in enumerate(payload["candidate_items"])
                    for condition in item["covered"]
                ],
            }
        if name == "verification.experiments":
            source = block_id(self.table)
            return {
                "checked_aspects": ASPECTS,
                "items": [
                    {
                        "aspect": "correspondence",
                        "kind": "paper_support",
                        "block_id": source,
                        "quote": self.table,
                        "covered": ["FB15k-237", "WN18RR"],
                        "fully_supported_conditions": ["FB15k-237", "WN18RR"],
                        "detail": "Fixture reads the proposed-method MRR row in Table 3; this is paper-internal support.",
                    }
                ],
                "plans": [
                    {
                        "targets": [
                            {
                                "condition_id": dataset,
                                "reported": {"block_id": source, "quote": self.table, "token": value},
                            }
                            for dataset, value in (("FB15k-237", ".355"), ("WN18RR", ".479"))
                        ],
                        "entry_script": "run.py",
                        "run_mode": "evaluation",
                        "feasibility": "blocked",
                        "blocker": "Fixture table extraction does not preserve unambiguous target cells or released weights.",
                        "priority": "high",
                    }
                ],
            }
        raise AssertionError(f"Unexpected fixture model call: {name}")


async def _offline_search(**kwargs):
    return {"success": True, "provider": "offline-fixture", "complete": False, "papers": [], "count": 0}


async def _offline_read(**kwargs):
    raise AssertionError("The empty retrieval fixture must never read an external source")


def replay_compgcn(
    output_root: Path, *, submission_deadline: str = "", demo_root: Path | None = None
) -> dict:
    demo = (demo_root or ROOT / "demos" / "Graph" / "compgcn").resolve()
    hashes = {
        str(path.relative_to(demo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in demo.rglob("*")
        if path.is_file()
    }
    parser, pages = _pdf_fixture(demo / "paper.pdf")
    model = CompGCNModelFixture(pages, demo / "execution" / "repo")
    args = SimpleNamespace(
        paper_pdf=str(demo / "paper.pdf"),
        paper_key="compgcn_fixture",
        run_root=str(output_root),
        repository_root=str(demo / "execution" / "repo"),
        submission_deadline=submission_deadline,
        run_execution=True,
        execution_no_llm=True,
        approval_mode="auto",
        training_budget=0,
        max_attempts=3,
    )

    async def literature(claim, materials):
        return await verify_literature(
            claim,
            materials,
            submission_deadline=submission_deadline,
            searcher=_offline_search,
            reader=_offline_read,
            call=model,
        )

    branches = {
        EvidenceNeed.CODE: lambda c, m: verify_code(c, m, call=model),
        EvidenceNeed.THEORY: lambda c, m: verify_theory(c, m, call=model),
        EvidenceNeed.EXPERIMENTS: lambda c, m: verify_experiments(c, m, call=model),
        EvidenceNeed.LITERATURE: literature,
    }

    def forbidden_runner(request):
        raise AssertionError("Blocked fixture plans must never invoke Docker")

    def no_network(*args, **kwargs):
        raise AssertionError("Offline replay attempted an external service or process")

    with (
        patch("requests.sessions.Session.request", no_network),
        patch("httpx.Client.send", no_network),
        patch("httpx.AsyncClient.send", no_network),
        patch("urllib.request.urlopen", no_network),
        patch("subprocess.run", no_network),
    ):
        summary = run_v2_pipeline(
            args,
            parser=parser,
            call=model,
            reference_checker=lambda **kwargs: {"ok": True, "total_refs": 0, "issues": []},
            branches=branches,
            global_literature=literature,
            runner=forbidden_runner,
        )
    after = {
        str(path.relative_to(demo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in demo.rglob("*")
        if path.is_file()
    }
    if hashes != after:
        raise AssertionError("CompGCN reference artifacts changed during fixture replay")
    legacy = read_v1_artifact(demo / "report.md")
    final_path = summary["outputs"].get("report_json")
    final = json.loads(Path(final_path).read_text(encoding="utf-8")) if final_path else {"claims": []}
    comparison = {
        "kind": "offline_fixture_replay",
        "boundary": BOUNDARY,
        "live_baseline": "Unavailable: no successful live MinerU baseline was recorded in Phase 0.",
        "live_mineru_token_configured": bool(os.getenv("MINERU_API_TOKEN")),
        "live_run_completed": False,
        "submission_deadline": submission_deadline or None,
        "deadline_note": "Explicit fixture input; no historical venue deadline is asserted; no arXiv-derived cutoff.",
        "reference_hashes_unchanged": hashes == after,
        "reference_hashes": hashes,
        "old_report": [
            {"text": c.text, "original_label": c.original_label, "adapted_status": c.status.value}
            for c in legacy.claims
        ],
        "old_adapted_counts": dict(Counter(c.status.value for c in legacy.claims)),
        "fixture_claims": [
            {
                "id": c["id"],
                "text": c["text"],
                "status": c["status"],
                "sources": sorted({e["source"] for e in c["evidence"]}),
            }
            for c in final["claims"]
        ],
        "fixture_counts": summary.get("counts", {}),
        "model_calls": model.calls,
        "comparison_limits": [
            "The legacy and fixture claim sets differ; no paired accuracy comparison is valid.",
            "Theory fixture intentionally covers only the explicit Kipf-GCN reduction.",
            "Multi-value table targets remain blocked; there is no independent execution evidence.",
            "Figure boxes and bibliography parsing require live MinerU validation.",
        ],
        "summary": summary,
    }
    destination = Path(summary["run_dir"])
    (destination / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = "# CompGCN v2 fixture replay\n\n" + BOUNDARY + "\n\n"
    markdown += f"Live baseline: {comparison['live_baseline']}\n\n"
    markdown += "| Source | Status counts |\n|---|---|\n"
    markdown += f"| Legacy report, adapted labels | {comparison['old_adapted_counts']} |\n"
    markdown += f"| v2 fixed-response fixture | {comparison['fixture_counts']} |\n\n"
    markdown += "\n".join("- " + item for item in comparison["comparison_limits"]) + "\n"
    (destination / "comparison.md").write_text(markdown, encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "v2_compgcn")
    parser.add_argument("--submission-deadline", default="")
    args = parser.parse_args()
    comparison = replay_compgcn(args.output_root, submission_deadline=args.submission_deadline)
    print(
        json.dumps(
            {
                "run_dir": comparison["summary"]["run_dir"],
                "stages": comparison["summary"]["stages"],
                "fixture_counts": comparison["fixture_counts"],
                "live_run_completed": False,
            },
            indent=2,
        )
    )
    if comparison["summary"]["stage_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
