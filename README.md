# FactReview <a href="https://arxiv.org/abs/2604.04074"><img src="https://img.shields.io/badge/arXiv-2604.04074-b31b1b.svg" alt="Paper"></a> <img src="https://img.shields.io/badge/license-AGPL--3.0-green.svg" alt="License">

<p align="center">
  <img src="docs/pipeline.png" alt="FactReview pipeline: L1 screening and claim extraction, L2 typed verification, L3 execution, evidence-linked review" width="900">
</p>

<p align="center"><strong>Evidence-grounded reviews for ML papers — every claim traced back to the literature, the paper, or actually running the code.</strong></p>

You give FactReview a paper PDF (or an arXiv URL). It returns a Markdown + PDF review where every major claim is tagged with one of four verdicts and linked to a paper section, a literature neighbor, or a number it reproduced by running the paper's code. The pipeline above shows L1 screening and claim extraction, L2 typed verification, optional L3 execution, and the evidence-linked review.

> **Two tools in this repo.**
> **[FactReview](#quick-start--factreview)** takes a paper and returns a full evidence-grounded review.
> **[RefCopilot](#quick-start--refcopilot)** takes any bibliography (`.bib`, PDF, or URL) and returns a list of fabricated, retracted, outdated, or incomplete citations — plus copy-paste-ready BibTeX corrections. They share infrastructure but run independently.

## Why FactReview

Generic LLM reviewers — pasting a PDF into ChatGPT, Gemini, or Claude — fail in four predictable ways. FactReview is built around fixing each of them.

| Generic LLM reviewer | FactReview |
|---|---|
| **Hallucinates citations** ("As shown by Smith et al., 2021…" — no such paper). | Every reference is verified in parallel against arXiv, Semantic Scholar, OpenReview, and (optional) OpenAlex. Fabrications, retractions, and arXiv withdrawals are flagged. See [RefCopilot](#what-refcopilot-produces). |
| **Calls Δ = 0.3% a "significant improvement"** because the paper said so. | The claim-audit pass auto-downgrades comparative claims when Δ < 2σ and flags missing ablations. See [`src/review/report/claim_audit.py`](src/review/report/claim_audit.py). |
| **Faults the paper for not citing related work that was published after it.** | When the input is an arXiv URL, FactReview derives a publication-date cutoff from the arXiv ID and applies it to retrieval (server-side at Semantic Scholar and client-side in the agent's `paper_search`). |
| **Cannot verify a single reported number.** | Optional Docker-based execution stage runs the paper's repository on its claimed benchmarks via a `prepare → plan → run → judge → fix → finalize` loop, then reports paper-vs-reproduced deltas. See [`src/fact_generation/execution/stage_runner.py`](src/fact_generation/execution/stage_runner.py). |

## See it in Action

Open the [CompGCN review](demos/Graph/compgcn/report.pdf) for an example of the rendered report.

**More demos** (each contains the full run artifacts and rendered review):

| Domain | Papers |
|---|---|
| Graph | [CompGCN](demos/Graph/compgcn) · [Graphormer](demos/Graph/graphormer) · [SACN](demos/Graph/sacn) |
| Image | [BEiT](demos/Image/beit) · [FixMatch](demos/Image/fixmatch) · [LRCN](demos/Image/lrcn) · [UDA](demos/Image/uda) |
| Text | [BERT](demos/Text/bert) · [Prefix-Tuning](demos/Text/Prefix-Tuning) |

## Quick Start — FactReview

Requirements: Python 3.11+, a local Codex login. Docker is only needed if you enable code execution.

```bash
git clone https://github.com/DEFENSE-SEU/FactReview.git && cd FactReview
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[runtime]"
codex login                                          # ChatGPT sign-in flow
cp .env.example .env                                 # then set MINERU_API_TOKEN
python scripts/execute_review_pipeline.py demos/Graph/compgcn/paper.pdf --submission-deadline 2019-09-25
```

When the run finishes, open the headline output:

```
runs/<paper_key>_<timestamp>_<id>/review/report/final_review.pdf
```

`--submission-deadline` sets the date Literature checks against; without it, Literature claims stay unverified. Pass `--repository-url` or `--repository-root` to bind the authors' released code for the Code and Experiments checks.

That is your review. To run on your own paper:

**Template** — replace the path and key with your own:
```bash
python scripts/execute_review_pipeline.py path/to/paper.pdf --paper-key my_paper
```

**Example** — fetch directly from arXiv:
```bash
python scripts/execute_review_pipeline.py https://arxiv.org/abs/1911.03082
```

If `codex` is not on your PATH, install OpenAI's Codex CLI (`npm install -g @openai/codex`) and rerun `codex login`. Get a free MinerU token from <https://mineru.net> (the free tier covers most papers). For all other configuration knobs, see [Configuration](#configuration). For all CLI flags and single-stage reruns, see [CLI Reference](#cli-reference).

## Quick Start — RefCopilot

RefCopilot stands alone — no PDF parsing, no Docker, no MinerU token. Give it a `.bib`, a PDF, an arXiv URL, or a plain-text bibliography:

```bash
pip install -e "./RefCopilot[dev]"
codex login                                          # if you have not already
refcopilot check path/to/paper.pdf                   # or .bib, arXiv URL, plain text
```

You get a Markdown + JSON report listing fabricated, retracted, outdated, and incomplete citations, with a copy-paste-ready corrected BibTeX entry for each fixable warning. See [`RefCopilot/README.md`](RefCopilot/README.md) for the full library API, cache management, and all flags.

## What FactReview Produces

Every extracted claim gets one of four statuses, decided by fixed rules over the collected evidence:

| Status | Evidence rule |
|---|---|
| **Supported** | Sufficient support covers every condition of the claim, with no concern affecting it. |
| **Questioned** | An author-resolvable concern, or sufficient support and flaw evidence overlap. |
| **Flawed** | Sufficient flaw evidence survives reasonable author explanations. |
| **Unverified** | Available evidence cannot decide the claim. |

A gate then checks Supported claims against the evidence their type needs (`--gate lenient|default|strict`); a Supported claim without that evidence is reported as Unverified, and all three tiers are kept in `gate_status`. The demos in this README were produced by v1, whose labels map as `in_conflict` → Questioned.

What makes the report distinctive (beyond the verdicts):

- **Design-axis positioning matrix** — neighbor papers retrieved from Semantic Scholar (and optional OpenAlex) are placed on a small set of design dimensions specific to the paper's domain, so reviewers can see which dimensions the paper genuinely innovates on. See [`src/fact_generation/positioning/stage_runner.py`](src/fact_generation/positioning/stage_runner.py).
- **Statistical-rigor downgrades** — comparative claims with Δ < 2σ are auto-downgraded; missing ablations are flagged. See [`src/review/report/claim_audit.py`](src/review/report/claim_audit.py).
- **Publication-date cutoff** — when the input is an arXiv URL or ID, FactReview derives a `YYYY-MM` cutoff from the arXiv identifier so the manuscript is not penalized for missing citations to work that was published after it. Override with `--cutoff-date` or disable with `--no-cutoff`.
- **Optional code execution** — L3 runs confirmed plans in a Docker sandbox, checks that dataset, metric and settings match the claim, allows at most 3 bounded repairs (`--max-attempts`), and records commands, logs and metrics under `execution/`. See [`src/fact_generation/execution/v2.py`](src/fact_generation/execution/v2.py).

## What RefCopilot Produces

RefCopilot extracts every reference from the input and verifies it in parallel against arXiv, Semantic Scholar, OpenReview, and (optionally) OpenAlex. Each finding falls into one of these buckets:

| Severity | Type | What it means |
|---|---|---|
| **Error** | `fake / no_match` | No matching record on any backend. Likely fabricated. |
| **Error** | `retracted` | Publisher retraction (via OpenAlex `is_retracted` + Retraction Watch) or arXiv author-withdrawn preprint. |
| **Warning** | `outdated / arxiv_published` | Cited as an arXiv preprint, but a published version exists at a venue. |
| **Warning** | `outdated / arxiv_version` | An older arXiv version is cited; a newer revision exists. |
| **Warning** | `outdated / workshop_to_full` | Cited as a workshop paper, but a full-conference version exists. |
| **Warning** | `incomplete` | Missing DOI / arXiv ID / venue / year, truncated authors, abbreviated venue name. |
| **Warning** | `non_academic_downgrade` | Title-mismatch heuristic flagged it, but LLM verification recognised it as a system card / blog post / dataset / standard / white paper — downgraded from error. |

For each fixable warning, RefCopilot emits a corrected BibTeX entry with a leading provenance comment listing **which backend supplied each field**, and the lookup URL — so a reviewer can audit any fix in one click:

```bibtex
% Suggested by RefCopilot. Field provenance:
%   semantic_scholar: title, authors, year, journal, doi — https://api.semanticscholar.org/...
%   arxiv: arxiv_id — https://arxiv.org/abs/...
@article{Smith2017Real,
  title   = {Real: A Title},
  author  = {Smith, John and Jones, Jane},
  year    = {2017},
  journal = {NeurIPS 2017},
  doi     = {10.1234/example},
}
```

What's behind the accuracy:

- **Four-backend parallel lookup** — arXiv, Semantic Scholar, OpenReview, and (when `OPENALEX_API_KEY` is set) OpenAlex are queried concurrently; results are merged with per-field provenance. See [`RefCopilot/src/refcopilot/pipeline.py`](RefCopilot/src/refcopilot/pipeline.py).
- **Two-stage hallucination detection** — a fast offline heuristic (title similarity + author overlap + OCR-garbled-title detection) handles the easy cases; the LLM is only invoked on the ambiguous ones. Saves tokens, raises precision.
- **"Second-chance" lookup** — if no backend matches but the LLM judges the paper as real, RefCopilot asks the LLM for canonical metadata (corrected title, DOI, arXiv ID) and retries all four backends once with the suggested values.
- **Retraction detection** — unified `is_retracted` signal from OpenAlex (publisher retractions + Retraction Watch) plus arXiv withdrawn-preprint notices. See [`RefCopilot/src/refcopilot/verify/retraction.py`](RefCopilot/src/refcopilot/verify/retraction.py).

When RefCopilot is invoked from FactReview's pipeline (`--enable-refcheck`), the full result is written to `stages/fact_generation/refcheck/reference_check.json` and a fabricated-references summary is appended to `final_review.md`.

## Configuration

FactReview keeps routine configuration in two places: `.env` for secrets and runtime choices, CLI flags for one-off overrides. The four settings most users touch:

**LLM backend (default: Codex login).** `.env.example` ships with `MODEL_PROVIDER=openai-codex` and the Codex model alias pre-filled — copy to `.env` and run `codex login` once. The Codex model alias is *not* a public OpenAI Platform model id; do not try to use it with `OPENAI_API_KEY` against `api.openai.com`.

**MinerU PDF parsing (required).** `MINERU_API_TOKEN` must be set. FactReview uses MinerU's cloud API by default — free tier, no local CUDA / GPU / model download. Get a token at <https://mineru.net>. You can also pass `--mineru-api-token` per-run.

**Semantic Scholar related-work retrieval (on by default).** The positioning stage uses Semantic Scholar for objective related-work neighbors unless `SEMANTIC_SCHOLAR_ENABLED=false` is set. Set `SEMANTIC_SCHOLAR_API_KEY` to avoid rate limits during positioning retrieval. Free key at <https://www.semanticscholar.org/product/api>. Disable it for one run with `--disable-semantic-scholar`.

**Agent `paper_search/read_paper` tools (on by default).** FactReview lets the review agent run claim-driven literature searches and requires a `read_paper` deep-read pass after effective search results before writing Technical Positioning. Set `PAPER_SEARCH_ENABLED=false` for manuscript/Semantic-Scholar-only runs. Available providers include `arxiv` (no key), `semantic_scholar`, `openalex`, and `remote`. The `semantic_scholar` provider reuses `SEMANTIC_SCHOLAR_API_KEY`, and `openalex` can use `OPENALEX_API_KEY` if set. `remote` keeps compatibility with advanced `/pasa/search` services. When `read_paper` is called without `PAPER_READ_BASE_URL`, FactReview uses a built-in arXiv PDF full-text fallback; set `PAPER_READ_BASE_URL` only if you run a stronger compatible reader service.

For OpenAlex, local MinerU fallback, agent-tracing knobs, and other rarely-touched variables see [Advanced Configuration](#advanced-configuration).

## CLI Reference

Full default pipeline on a local PDF:

```bash
python scripts/execute_review_pipeline.py path/to/paper.pdf --paper-key my_paper
```

You can also pass an arXiv URL — abstract links are normalized to the PDF download:

```bash
python scripts/execute_review_pipeline.py https://arxiv.org/abs/1911.03082 --paper-key compgcn
```

When the input is an arXiv link, FactReview auto-derives a publication-date cutoff (`YYYY-MM`) from the arXiv identifier so positioning retrieval only considers prior work. Override with `--cutoff-date YYYY[-MM[-DD]]`, or disable entirely with `--no-cutoff`. Local PDFs default to no cutoff unless `--cutoff-date` is supplied.

`refcheck` and `execution` are off by default. Enable code execution (Docker daemon required, no extra Python deps):

```bash
python scripts/execute_review_pipeline.py path/to/paper.pdf --run-execution
```

Enable RefCopilot inside the pipeline:

```bash
pip install -e ".[refcheck]"
python scripts/execute_review_pipeline.py path/to/paper.pdf --enable-refcheck
```

Or globally via `FACTREVIEW_ENABLE_REFCHECK=true`. The full result lands in `stages/fact_generation/refcheck/reference_check.json`; the Markdown summary appended to `final_review.md` lists fabricated references only. For the complete breakdown, run RefCopilot's standalone CLI or read the JSON directly.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--llm-provider` | `openai-codex` | Switches the LLM provider. Mirrors to `MODEL_PROVIDER`. |
| `--llm-model` | provider default | Mirrors to `AGENT_MODEL`, `EXECUTION_OPENAI_MODEL`, and `OPENAI_CODEX_MODEL` (when the provider is Codex). |
| `--mineru-api-token` | from `.env` | One-off override for `MINERU_API_TOKEN`. |
| `--disable-semantic-scholar` | off | Disable Semantic Scholar objective related-work retrieval for this run. Equivalent to setting `SEMANTIC_SCHOLAR_ENABLED=false` before loading settings. |
| `--enable-refcheck` | off | Run RefCopilot as the refcheck stage. |
| `--run-execution` | off | Enables the code-execution stage. Requires Docker. |
| `--max-attempts` | `3` | Repair rounds in L3 execution, 0–3. |
| `--submission-deadline` | – | Venue deadline (`YYYY-MM-DD`) that bounds Literature retrieval. |
| `--repository-url` / `--repository-root` | – | The authors' released code for Code and Experiments checks. |
| `--gate` | `default` | Claim gate tier reported as each claim's status: `lenient`, `default`, `strict`. Also `CLAIM_GATE`. |
| `--alignment` | `strict` | Execution alignment; `relaxed` matches the dataset only (ablation). |
| `--repairer` | `default` | Execution repair proposer; `llm` proposes whitelisted repairs (ablation). See [execution ablations](docs/execution_ablations.md). |
| `--report-presentation` | `full` | `layered` writes a reading report plus a technical appendix and a linked PDF bundle. |
| `--no-pdf-extract` | off | Skip MinerU re-extraction inside the execution `prepare` node when the parse stage already produced the snapshot. |
| `--reuse-job-id` | – | Reuse a prior agent-runtime job, skipping the parse-stage agent run. Accepts either an absolute path to a `runtime/jobs/<id>` directory (taken as-is) or a bare job id (looked up under the current run dir, then under `<run-root>/**/runtime/jobs/<id>`). Useful for re-rendering the report after a downstream-stage tweak without paying the parse cost again. |
| `--run-root` | `runs` | Override the root output directory. |
| `--cutoff-date` | auto | Inclusive publication-date cutoff for positioning retrieval, as `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`. When omitted, an arXiv URL/ID is used to auto-derive `YYYY-MM` from the arXiv identifier; for non-arXiv inputs no cutoff is applied. Both Semantic Scholar (server-side `year=` filter) and the agent's `paper_search` calls (client-side filter) are constrained to papers at or before the cutoff, so the agent does not penalise the manuscript for not citing later work. |
| `--no-cutoff` | off | Disable the publication-date cutoff entirely (overrides `--cutoff-date` and arXiv auto-derivation). Useful for analysing how the paper compares against later work. |

### Single-Stage Reruns

Each stage has a standalone script that reads the same per-run layout. `parse` takes the original PDF (because the bridge state may not exist yet); the rest work off the run dir alone:

```bash
python scripts/execute_stage_parse.py          path/to/paper.pdf --run-dir runs/<run>
python scripts/execute_stage_refcheck.py       --run-dir runs/<run>
python scripts/execute_stage_positioning.py    --run-dir runs/<run>
python scripts/execute_stage_execution.py      --run-dir runs/<run>
python scripts/execute_stage_report.py         --run-dir runs/<run>
```

### Outputs

Each run writes to `runs/<paper_key>_<timestamp>_<id>/`. Primary artifacts:

- `full_pipeline_summary.json`, `run_stats.json` — stage results, errors and token usage.
- `materials/materials.json` — parsed paper, page and figure images, repository index.
- `screening/screening.json` — L1 claims and basic checks; `screening/claim_coverage/` holds the extraction coverage review.
- `verification/verification.json` — L2 evidence, findings and execution plans.
- `execution/` — L3 commands, logs, alignment and repair records (only with `--run-execution`).
- `review/report/final_review.{json,md,pdf}` — **the headline review.**

## Pipeline Architecture

```text
materials → L1 screening → L2 typed verification → L3 execution (optional) → assessment → report
```

- **L1 screening & extraction** — writing, figure and reference checks; claims extracted with their location, conditions and evidence needs.
- **L2 typed verification** — each claim goes only to the branches it needs: Literature (prior work), Theory (equations and derivations), Code (code and config), Experiments (evaluation and run plan).
- **L3 execution** — runs confirmed plans in a Docker sandbox, aligns dataset, metric and settings with the claim, and allows at most 3 bounded repairs (dependencies, paths, launch arguments; never model, data or evaluation).
- **Assessment and report** — fixed status rules and the claim gate set each status; the report links every judgement to related work, code or a run log.

The full design is in [`docs/method_v2_spec.md`](docs/method_v2_spec.md).

## Benchmark

The evaluation uses 300 peer-reviewed papers in six groups from Agents4Science, ICLR, TMLR, Nature Communications, COLM and ICML. PDFs, public OpenReview review threads and per-paper licenses are on Hugging Face at [kabuda777/factreview-bench](https://huggingface.co/datasets/kabuda777/factreview-bench); [`benchmark/papers.csv`](benchmark/papers.csv) is the same index, described in [`benchmark/README.md`](benchmark/README.md).

```python
from datasets import load_dataset
papers = load_dataset("kabuda777/factreview-bench", split="train")
```

## Troubleshooting

- **`codex login` fails or is not on PATH** — install OpenAI's Codex CLI (`npm install -g @openai/codex`), then rerun `codex login` and pick the ChatGPT sign-in flow.
- **`MINERU_API_TOKEN` missing** — the parse stage will raise on the first run. Get a token from <https://mineru.net> (free tier is sufficient for most papers) and set it in `.env` or pass `--mineru-api-token`.
- **`--enable-refcheck` errors with missing deps** — install with `pip install -e ".[refcheck]"`.
- **Positioning stage is slow or returns sparse results** — unauthenticated Semantic Scholar requests are rate-limited. Set `SEMANTIC_SCHOLAR_API_KEY` in `.env` (free key from <https://www.semanticscholar.org/product/api>).

## Advanced Configuration

Less common environment variables — set in `.env` or via the shell. `.env.example` is the authoritative list; the table below covers the ones most users will touch.

| Variable | Purpose |
|---|---|
| `FACTREVIEW_ENABLE_REFCHECK` | Enable reference checking globally (equivalent to the `--enable-refcheck` flag). |
| `FACTREVIEW_EXECUTION_ENABLE_REFCHECK` | Enable a refcheck sweep *inside* the execution stage's refcheck node. Independent from the global gate above. |
| `OPENALEX_API_KEY` | Optional OpenAlex API key. When set, OpenAlex is queried as a fourth cross-check signal in RefCopilot and used by `PAPER_SEARCH_PROVIDER=openalex`; when empty, OpenAlex paper search still works unauthenticated. Free key at <https://openalex.org/settings/api>. |
| `MINERU_BASE_URL` | Override the MinerU cloud API endpoint (default: `https://mineru.net/api/v4`). |
| `MINERU_ALLOW_LOCAL_FALLBACK` | Set to `true` to let the execution stage's `prepare` node fall back to the local `mineru` CLI when the cloud snapshot is unavailable. |
| `MINERU_LOCAL_BACKEND` / `MINERU_LOCAL_DEVICE` / `MINERU_LOCAL_SOURCE` | Tune the local `mineru` CLI's pipeline backend, device, and source mirror. Only consulted when `MINERU_ALLOW_LOCAL_FALLBACK=true` and a local MinerU install is present. |
| `OPENAI_AGENTS_DISABLE_TRACING` | Set to `0` to enable the openai-agents SDK trace exporter. Disabled (`1`) by default to avoid POSTing traces to the Agents tracing endpoint. |
| `OPENAI_CODEX_BASE_URL` | Point Codex at a different Codex-compatible endpoint (default: `https://chatgpt.com/backend-api/codex`). |
| `SEMANTIC_SCHOLAR_ENABLED` | Enable or disable Semantic Scholar objective related-work retrieval. Defaults to `true`; set `false` to skip it persistently. |
| `SEMANTIC_SCHOLAR_API_KEY` | Recommended. Free API key from [Semantic Scholar](https://www.semanticscholar.org/product/api) for the positioning stage and `PAPER_SEARCH_PROVIDER=semantic_scholar`. Without it, unauthenticated requests may be rate-limited. |
| `PAPER_SEARCH_ENABLED` | Enable the agent `paper_search` tool. Defaults to `true`; set `false` to force manuscript/Semantic-Scholar-only runs. |
| `PAPER_SEARCH_PROVIDER` | Retrieval backend for `paper_search`: `arxiv` (zero-key), `semantic_scholar`, `openalex`, or `remote`. |
| `PAPER_SEARCH_BASE_URL` / `PAPER_SEARCH_ENDPOINT` | Only needed for `PAPER_SEARCH_PROVIDER=remote`; points to a compatible `/pasa/search` service. |
| `PAPER_SEARCH_API_KEY` / `PAPER_SEARCH_HEALTH_ENDPOINT` | Optional bearer token and health endpoint for `remote` paper search. Default health endpoint: `/health`. |
| `PAPER_READ_BASE_URL` / `PAPER_READ_ENDPOINT` | Optional external `read_paper` service URL and endpoint. Leave base URL empty to use the built-in arXiv PDF full-text fallback when `read_paper` is called. Default endpoint: `/read`. |
| `PAPER_READ_API_KEY` | Optional bearer token for the external `read_paper` service. |

## Development

```bash
pip install -e ".[runtime,dev]"

ruff check .
ruff format --check .
# Narrow CI smoke check (the contracts most likely to break consumers). For a
# full pass, run `mypy` with no args — it picks up the broader package list
# from pyproject.toml's [tool.mypy] section.
mypy src/schemas src/util src/common
pytest                          # default: ~50 fast tests, gated markers off
pytest -m e2e                   # end-to-end integration tests
pytest -m requires_docker       # execution stage (needs Docker daemon)
pytest -m ""                    # full set, including all gated tests
```

## Paper

Read the paper on <https://arxiv.org/abs/2604.04074> or from the local PDF at [`factreview.pdf`](factreview.pdf).

If FactReview helped your work, please ⭐ the repo and cite:

```bibtex
@article{yue2026factreview,
  title={FactReview: Evidence-Grounded Peer Review with Execution-Based Claim Verification},
  author={Yue, Ling and Ouyang, Chaoqian and Xu, Hang and Huang, Ruijun and Liu, Yuchen and Zheng, Libin and Liu, Wei and Pan, Shaowu and Di, Shimin and Zhang, Min-Ling},
  journal={arXiv preprint arXiv:2604.04074},
  year={2026}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=defense-seu/factreview&type=timeline&legend=top-left)](https://www.star-history.com/#defense-seu/factreview&type=timeline&legend=top-left)

## License

AGPL-3.0-only.
