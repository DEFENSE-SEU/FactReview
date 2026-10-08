"""Reproducible synthetic v2 integration matrix with explicit external boundaries.

Default runs are offline. --docker enables the production Docker builder/runner
for the two runtime cases; parsing, model judgments and retrieval stay fixtures.
These cases measure integration contracts, not scientific-review accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pymupdf

from llm.client import LLMConfig
from pipeline_v2 import STAGES, run_v2_pipeline
from preprocessing.parse.mineru_adapter import MineruParseResult
from util.subprocess_runner import CommandResult

SCENARIOS = (
    "theory_appendix",
    "figures_partial",
    "missing_repository",
    "mapped_runtime",
    "mapped_runtime_misaligned",
)
ASPECTS = ["correspondence", "fairness", "isolation", "stability", "consistency"]
LIMIT = "Synthetic contract fixtures; no model accuracy or independent scientific reproduction is measured."
RUNTIME_SCRIPT = """import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
config = json.loads((root / "configs/runtime.json").read_text())
correct = json.loads((root / "data/correct.json").read_text())
print(json.dumps({"context": config, "measurements": {"accuracy": sum(correct) / len(correct)}}))
"""


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def text_row(text, page=0, heading=False):
    return {"type": "text", "text": text, "page_idx": page, **({"text_level": 1} if heading else {})}


def condition(identifier, dataset=None, split=None, description=""):
    if dataset:
        return {"id": identifier, "dataset": dataset, "metric": "accuracy", "settings": {"split": split}}
    return {"id": identifier, "description": description}


def make_inputs(name: str, directory: Path):
    rows = [text_row("# Synthetic framework validation", heading=True)]
    claims = []

    def claim(text, conditions, needs, expected):
        rows.append(text_row(text))
        claims.append({"text": text, "conditions": conditions, "needs": needs, "expected": expected})

    repo = None
    if name == "theory_appendix":
        rows[0]["text"] = "# Synthetic optimization theorem"
        claim(
            "Theorem: for every real x, x squared is nonnegative.",
            [condition("nonnegative", description="Nonnegativity for every real x")],
            ["Theory"],
            "supported",
        )
        claim(
            "Conjecture: every update converges.",
            [condition("convergence", description="Every update converges")],
            ["Theory"],
            "unverified",
        )
        rows += [
            text_row("## Appendix proof", page=1, heading=True),
            text_row("Proof. x squared = x * x >= 0 for every real x, hence the theorem holds.", page=1),
        ]
    elif name == "figures_partial":
        rows[0]["text"] = "# Synthetic image classification figures"
        claim(
            "Accuracy improves on both Alpha and Beta.",
            [condition("alpha", "Alpha", "test"), condition("beta", "Beta", "test")],
            ["Experiments"],
            "unverified",
        )
        rows += [
            text_row("Alpha test accuracy is 0.75."),
            text_row("Figures 1, 2, 3 and 4 describe the result."),
        ]
        for index, caption in enumerate(
            (
                "Figure 1: Unavailable visual response.",
                ["Figure 2: First panel.", "Figure 3: Neighbor caption."],
                "Figure 4: A plotted observation.",
            )
        ):
            rows.append(
                {
                    "type": "image",
                    "image_caption": caption,
                    "page_idx": 0,
                    "bbox": [80 + index * 300, 450, 280 + index * 300, 650],
                }
            )
    elif name == "missing_repository":
        rows[0]["text"] = "# Synthetic missing-artifact image classification"
        claim(
            "The implementation uses the Adam optimizer.",
            [condition("optimizer", description="Implementation uses Adam")],
            ["Code"],
            "unverified",
        )
        claim(
            "MissingSet test accuracy is 0.75.",
            [condition("missing", "MissingSet", "test")],
            ["Experiments"],
            "unverified",
        )
    else:
        rows[0]["text"] = "# Synthetic image classification analysis"
        claim(
            "FixtureVision test accuracy is 0.75.",
            [condition("test", "FixtureVision", "test")],
            ["Experiments"],
            "unverified" if name.endswith("misaligned") else "supported",
        )
        claim(
            "FixtureVision train accuracy is 0.75.",
            [condition("train", "FixtureVision", "train")],
            ["Experiments"],
            "unverified",
        )
        repo = directory / "repository"
        (repo / "tools").mkdir(parents=True)
        (repo / "tools/evaluate.py").write_text(RUNTIME_SCRIPT, encoding="utf-8")
        save(
            repo / "configs/runtime.json",
            {"dataset": "FixtureVision", "split": "validation" if name.endswith("misaligned") else "test"},
        )
        save(repo / "data/correct.json", [1, 1, 0, 1])
    markdown_parts = []
    for row in rows:
        caption = row.get("image_caption", "")
        markdown_parts.append(
            row.get("text") or ("\n".join(caption) if isinstance(caption, list) else caption)
        )
    markdown = "\n\n".join(markdown_parts)
    pdf_path = directory / "paper.pdf"
    with pymupdf.open() as pdf:
        for page_number in range(max(row["page_idx"] for row in rows) + 1):
            page = pdf.new_page(width=600, height=800)
            body = "\n\n".join(
                row["text"] for row in rows if row["page_idx"] == page_number and "text" in row
            )
            page.insert_textbox((30, 30, 570, 330), body, fontsize=11)
            for row in rows:
                if row["page_idx"] != page_number or row["type"] != "image":
                    continue
                x1, y1, x2, y2 = row["bbox"]
                box = pymupdf.Rect(x1 * 0.6, y1 * 0.8, x2 * 0.6, y2 * 0.8)
                page.draw_rect(box)
                page.draw_line((box.x0, box.y1), (box.x1, box.y0))
                page.insert_text((box.x0 + 4, box.y0 + 14), "synthetic panel", fontsize=8)
        pdf.save(pdf_path)
    save(directory / "parser_fixture.json", {"markdown": markdown, "content_list": rows, "boundary": LIMIT})
    parsed = MineruParseResult(
        markdown, rows, None, "framework-matrix", {"fixture": True}, "fixed-parser", LIMIT
    )
    return pdf_path, parsed, claims, repo


class FixedParser:
    def __init__(self, parsed, pdf):
        self.parsed = parsed
        self.sha256 = hashlib.sha256(pdf.read_bytes()).hexdigest()
        self.calls = 0

    async def parse_pdf(self, *, pdf_path, data_id):
        if hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest() != self.sha256:
            raise AssertionError("Fixture parser received a different PDF")
        self.calls += 1
        return self.parsed


class FixedRetrieval:
    def __init__(self):
        self.calls = []

    async def search(self, *, query, cutoff_date):
        self.calls.append({"query": query, "cutoff": cutoff_date.to_string()})
        return {"success": True, "provider": "fixed-empty-search", "complete": False, "papers": []}

    async def read_papers(self, **kwargs):
        raise AssertionError("Empty search fixture must never read external works")


class FixedModel:
    def __init__(self, scenario, claims):
        self.scenario, self.claims, self.calls = scenario, claims, []

    def __call__(self, **kwargs):
        module = kwargs["module"]
        prompt = kwargs["prompt"].split("\nPAPER_DATA_JSON:\n", 1)[-1]
        payload = json.loads(prompt)
        record = {
            "module": module,
            "input": payload,
            "images": kwargs.get("images", []),
            "model_config": {"provider": kwargs["cfg"].provider, "model": kwargs["cfg"].model},
        }
        self.calls.append(record)
        try:
            record["response"] = self.respond(module, payload)
            return record["response"]
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def respond(self, module, payload):
        if module == "screening.claims":
            return {
                "status": "ok",
                "claims": [
                    {
                        "text": row["text"],
                        "source_block_id": next(
                            b["id"] for b in payload["blocks"] if b["text"] == row["text"]
                        ),
                        "source_quote": row["text"],
                        "conditions": row["conditions"],
                        "needs": row["needs"],
                        "importance": "core",
                    }
                    for row in self.claims
                ],
            }
        if module in {"screening_writing", "screening_tables"}:
            return {"findings": []}
        if module == "screening_figures":
            if payload["figure_id"] == "figure_1":
                raise TimeoutError("Injected visual-service timeout for the first figure")
            if payload["caption_ambiguous"]:
                return {
                    "findings": [
                        {
                            "category": "text_figure_consistency",
                            "disposition": "issue",
                            "text": "Fixture ambiguity must suppress this context-dependent item.",
                        },
                        {
                            "category": "legibility",
                            "disposition": "issue",
                            "text": "Fixture small panel text is unreadable at its printed size.",
                        },
                    ]
                }
            return {
                "findings": [
                    {
                        "category": "self_containedness",
                        "disposition": "issue",
                        "text": "Fixture plotted axis has no unit label.",
                    }
                ]
            }
        if module.startswith("verification.theory"):
            blocks = payload["main_text"]
            source = next(b for b in blocks if b["text"] == payload["claim"]["text"])
            identifier = payload["allowed_condition_ids"][0]
            if identifier == "convergence":
                return {
                    "items": [
                        {
                            "block_id": source["id"],
                            "quote": source["text"],
                            "covered": [identifier],
                            "kind": "no_proof",
                            "direction": "support",
                            "detail": "The fixture contains no proof for this conjecture.",
                        }
                    ]
                }
            if module == "verification.theory":
                return {"items": [], "appendix_block_ids": [b["id"] for b in payload["appendix_index"]]}
            proof = next(b for b in payload["appendix_proofs"] if b["text"].startswith("Proof."))
            return {
                "items": [
                    {
                        "block_id": proof["id"],
                        "main_block_id": source["id"],
                        "quote": proof["text"],
                        "covered": [identifier],
                        "fully_supported_conditions": [identifier],
                        "kind": "derivation",
                        "direction": "support",
                        "detail": "Fixed proof comparison exercises appendix coverage.",
                        "step_quote": proof["text"],
                    }
                ]
            }
        if module == "verification.experiments.scope":
            return {
                "conditions": [
                    {
                        "condition_id": condition["id"],
                        "claim_quote": payload["claim"]["text"],
                        "assertion": "controlled_comparison",
                        "matched_controls_required": True,
                        "uncertainty_sensitive": True,
                        "relation": "gt",
                        "subject": "",
                        "comparator": "",
                        "rationale": "The fixture claims improvement on both datasets.",
                    }
                    for condition in payload["claim"]["conditions"]
                ],
                "items": [
                    {
                        "item_index": index,
                        "condition_id": condition,
                        "applicability": "applicable",
                        "grounds": [{"block_id": item["block_id"], "quote": item["quote"]}],
                        "rationale": "An absolute Alpha score does not establish improvement over a baseline.",
                        "full_support": False,
                        "qualifiers_complete": False,
                        "comparison_objects": "unresolved",
                        "unresolved_qualifiers": ["Baseline result absent"],
                    }
                    for index, item in enumerate(payload["candidate_items"])
                    for condition in item["covered"]
                ],
            }
        if module == "verification.experiments":
            blocks = payload["paper_blocks"]
            claim = payload["claim"]
            if self.scenario == "figures_partial":
                row = next(b for b in blocks if b["text"] == "Alpha test accuracy is 0.75.")
                return {
                    "checked_aspects": ASPECTS,
                    "items": [
                        {
                            "aspect": "correspondence",
                            "kind": "paper_support",
                            "block_id": row["id"],
                            "quote": row["text"],
                            "covered": ["alpha"],
                            "fully_supported_conditions": ["alpha"],
                            "detail": "Only the Alpha condition is checked by this fixture.",
                        }
                    ],
                    "plans": [],
                }
            row = next(b for b in blocks if b["text"] == claim["text"])
            identifier = claim["conditions"][0]["id"]
            missing = self.scenario == "missing_repository"
            return {
                "checked_aspects": ASPECTS,
                "items": [],
                "plans": [
                    {
                        "targets": [
                            {
                                "condition_id": identifier,
                                "reported": {"block_id": row["id"], "quote": row["text"], "token": "0.75"},
                            }
                        ],
                        "entry_script": None if missing else "tools/evaluate.py",
                        "run_mode": "training" if identifier == "train" else "analysis",
                        "feasibility": "blocked" if missing else "ready",
                        "blocker": "Released repository absent" if missing else "",
                        "priority": "high",
                        "data_paths": [] if missing else ["data/correct.json"],
                        "estimated_cost": "Synthetic CPU analysis; training remains disabled",
                    }
                ],
            }
        raise AssertionError(f"Unexpected fixture model request: {module}")


def no_external(*args, **kwargs):
    raise AssertionError("Synthetic matrix attempted an unconfigured external service or process")


def mock_docker_command(command, *, cwd, timeout_sec):
    if command[:2] != ["docker", "run"]:
        raise AssertionError(f"Unexpected mocked Docker operation: {command[:2]}")
    mount = next(value for value in command if value.endswith(":/app"))
    repository = Path(mount.removesuffix(":/app"))
    config = json.loads((repository / "configs/runtime.json").read_text())
    values = json.loads((repository / "data/correct.json").read_text())
    output = {"context": config, "measurements": {"accuracy": sum(values) / len(values)}}
    return CommandResult(command, cwd, 0, json.dumps(output), "", 0.01)


def validate_case(name, summary, claims, parser, model):
    outputs = summary["outputs"]
    review = (
        json.loads(Path(outputs["report_json"]).read_text(encoding="utf-8"))
        if "report_json" in outputs
        else {}
    )
    observed = {c["id"]: c["status"] for c in review.get("claims", [])}
    expected = {f"claim_{i:03d}": row["expected"] for i, row in enumerate(claims, 1)}
    checks = {
        "pipeline_stages": list(summary["stages"]) == list(STAGES)
        and set(summary["stages"].values()) == {"ok"},
        "no_stage_errors": not summary["stage_errors"],
        "parse_once": parser.calls == 1,
        "claim_statuses": observed == expected,
        "complete_artifacts": all(
            key in outputs and Path(outputs[key]).is_file()
            for key in ("report_json", "report_markdown", "report_pdf", "teaser_image")
        ),
        "all_claims_have_source": all(
            c.get("source_block_id") and c.get("source_quote") for c in review.get("claims", [])
        ),
    }
    if "report_markdown" in outputs:
        report = Path(outputs["report_markdown"]).read_text(encoding="utf-8")
        checks["four_report_sections"] = all(
            f"## {index}. {title}" in report
            for index, title in enumerate(("Overview", "Claim list", "Other findings", "Execution ledger"), 1)
        )
    if name == "theory_appendix":
        checks["appendix_was_consulted"] = any(
            row["module"] == "verification.theory.appendix" for row in model.calls
        )
        target = next((c for c in review.get("claims", []) if c["id"] == "claim_001"), {})
        checks["appendix_evidence_located"] = any(
            e["source"] == "theory" and e["pointer"]["page"] == 2 for e in target.get("evidence", [])
        )
        missing = next((c for c in review.get("claims", []) if c["id"] == "claim_002"), {})
        checks["missing_proof_question"] = bool(missing.get("questions"))
    if name == "figures_partial":
        checks["figure_failure_isolated"] = summary.get("figure_coverage") == {
            "total": 3,
            "checked": 2,
            "failed": 1,
            "unavailable": 0,
        }
        findings = [f for f in review.get("findings", []) if f["kind"] == "figure"]
        checks["valid_figures_retained"] = {f["level"] for f in findings} == {
            "legibility",
            "self_containedness",
        } and len(findings) == 2
        checks["ambiguous_caption_recorded"] = any("ambiguous" in issue for issue in summary["issues"])
        checks["visual_coverage_in_report"] = "Figure screening is incomplete" in report
    if name == "missing_repository":
        ledger = review.get("ledger", [])
        checks["blocked_plan_retained"] = (
            len(ledger) == 1
            and not ledger[0]["approved"]
            and not ledger[0]["attempts"]
            and "missing" in ledger[0]["reason"].lower()
        )
        checks["no_code_model_without_repository"] = all(
            row["module"] != "verification.code" for row in model.calls
        )
        checks["resource_questions_visible"] = all(c["questions"] for c in review.get("claims", []))
    if name.startswith("mapped_runtime"):
        ledger = review.get("ledger", [])
        analysis = next((entry for entry in ledger if entry["plan"]["id"] == "claim_001.plan"), {})
        training = next((entry for entry in ledger if entry["plan"]["id"] == "claim_002.plan"), {})
        checks["training_budget_enforced"] = (
            not training.get("approved", True)
            and "training budget" in training.get("reason", "")
            and not training.get("attempts")
        )
        checks["single_analysis_attempt"] = len(analysis.get("attempts", [])) == 1
        if analysis.get("attempts"):
            logs = analysis["attempts"][0]["logs"]
            mapping = json.loads(Path(logs["output_mapping"]).read_text(encoding="utf-8"))
            raw = json.loads(Path(logs["raw_output"]).read_text(encoding="utf-8"))
            checks["native_output_decoded"] = (
                bool(mapping["selectors"])
                and raw["measurements"]["accuracy"] == 0.75
                and "observations" not in raw
            )
        checks["approval_recorded"] = bool(ledger) and all(
            entry["approval_mode"] == "auto" and entry["training_budget"] == 0 for entry in ledger
        )
        target = next((c for c in review.get("claims", []) if c["id"] == "claim_001"), {})
        deciding = [e for e in target.get("evidence", []) if e["source"] == "execution" and e["sufficient"]]
        checks["runtime_alignment_gates_evidence"] = (
            (not deciding and bool(target.get("questions")))
            if name.endswith("misaligned")
            else (len(deciding) == 1 and deciding[0]["aligned"] is True)
        )
    return {
        "expected_statuses": expected,
        "observed_statuses": observed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_case(name: str, root: Path, *, real_docker=False):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=False)
    pdf, parsed, claims, repo = make_inputs(name, directory)
    parser, model, retrieval = FixedParser(parsed, pdf), FixedModel(name, claims), FixedRetrieval()
    config = {
        "max_attempts": 0,
        "approval_mode": "auto",
        "training_budget": 0,
        "refine_with_llm": False,
        "timeout_seconds": 45,
        "docker_build_timeout_seconds": 180,
        "docker_options": {
            "docker_paper_python_image": "python:3.11-slim",
            "docker_include_notebook_requirements": False,
        },
        "output_mappings": {
            "claim_001.plan": {
                "dataset_path": ["context", "dataset"],
                "settings_path": None,
                "settings_paths": {"split": ["context", "split"]},
                "metric_paths": {"accuracy": ["measurements", "accuracy"]},
            }
        },
    }
    save(directory / "execution_config.json", config)
    args = SimpleNamespace(
        paper_pdf=str(pdf),
        paper_key=name,
        run_root=str(directory / "runs"),
        repository_root=str(repo) if repo else "",
        submission_deadline="2021-01-01",
        run_execution=True,
        execution_config=str(directory / "execution_config.json"),
    )
    docker_enabled = real_docker and name.startswith("mapped_runtime")
    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                os.environ,
                {name: "" for name in ("VLM_MODEL_PROVIDER", "VLM_MODEL", "VLM_BASE_URL", "VLM_API_KEY")},
            )
        )
        for target in (
            "requests.sessions.Session.request",
            "httpx.Client.send",
            "httpx.AsyncClient.send",
            "urllib.request.urlopen",
        ):
            stack.enter_context(patch(target, no_external))
        for target in (
            "screening.claims.resolve_llm_config",
            "screening.checks.resolve_llm_config",
            "verification.literature.resolve_llm_config",
        ):
            stack.enter_context(patch(target, lambda: LLMConfig("mock", "framework-fixture", None, None)))
        stack.enter_context(patch("verification.literature._default_adapter", lambda: retrieval))
        if not docker_enabled:
            stack.enter_context(patch("subprocess.run", no_external))
            stack.enter_context(patch("subprocess.Popen", no_external))
            stack.enter_context(
                patch("fact_generation.execution.tools.docker._docker_info_field", lambda *a, **k: "")
            )
            stack.enter_context(
                patch(
                    "fact_generation.execution.v2.docker_ensure_paper_image",
                    return_value=(True, "mock-framework-image"),
                )
            )
            stack.enter_context(patch("fact_generation.execution.v2.run_command", mock_docker_command))
        summary = run_v2_pipeline(
            args,
            parser=parser,
            call=model,
            reference_checker=lambda **kwargs: {"ok": True, "total_refs": 0, "issues": []},
        )
    save(directory / "model_calls.json", model.calls)
    save(directory / "retrieval_calls.json", retrieval.calls)
    try:
        result = validate_case(name, summary, claims, parser, model)
    except Exception as exc:
        result = {"passed": False, "validation_error": f"{type(exc).__name__}: {exc}"}
    result.update(
        {
            "scenario": name,
            "boundary_note": LIMIT,
            "boundaries": {
                "MinerU": "fixed parser",
                "LLM_VLM": "fixed responses",
                "retrieval": "fixed incomplete empty results",
                "reference_check": "fixed empty bibliography",
                "Docker": "real default builder/runner"
                if docker_enabled
                else "mocked image build and command transport; production output decoder",
            },
            "paths": {
                "input_pdf": str(pdf),
                "parser_fixture": str(directory / "parser_fixture.json"),
                "model_calls": str(directory / "model_calls.json"),
                "retrieval_calls": str(directory / "retrieval_calls.json"),
                "summary": str(Path(summary["run_dir"]) / "full_pipeline_summary.json"),
                **summary["outputs"],
            },
            "stage_errors": summary["stage_errors"],
            "issues": summary["issues"],
        }
    )
    save(directory / "case_manifest.json", result)
    return result


def run_matrix(output_root: Path, *, scenarios=None, real_docker=False):
    selected = list(scenarios or SCENARIOS)
    if not selected or len(selected) != len(set(selected)) or any(name not in SCENARIOS for name in selected):
        raise ValueError("Select distinct known matrix scenarios")
    root = (output_root / ("matrix-" + uuid.uuid4().hex[:10])).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = [run_case(name, root, real_docker=real_docker) for name in selected]
    manifest = {
        "kind": "synthetic_v2_framework_matrix",
        "boundary_note": LIMIT,
        "training_budget": 0,
        "root": str(root),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
    save(root / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "v2_framework_matrix")
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use real Docker only for runtime scenarios; other services stay mocked",
    )
    args = parser.parse_args(argv)
    manifest = run_matrix(args.output_root, scenarios=args.scenario, real_docker=args.docker)
    print("MATRIX_MANIFEST=" + str(Path(manifest["root"]) / "manifest.json"))
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "cases": [
                    {"scenario": row["scenario"], "passed": row["passed"]} for row in manifest["cases"]
                ],
            }
        )
    )
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
