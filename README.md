# FactReview <a href="https://arxiv.org/abs/2604.04074"><img src="https://img.shields.io/badge/arXiv-2604.04074-b31b1b.svg" alt="Paper"></a> <img src="https://img.shields.io/badge/license-AGPL--3.0-green.svg" alt="License">

<p align="center">
  <img src="demos/Graph/compgcn/teaser_figure.png" alt="A FactReview output for the CompGCN paper: technical positioning, claim verdicts, reproduced experimental numbers, strengths and weaknesses — all on one page." width="900">
</p>

<p align="center"><strong>Evidence-grounded reviews for ML papers — every claim traced back to the literature, the paper, or actually running the code.</strong></p>

You give FactReview a paper PDF (or an arXiv URL). It returns a Markdown + PDF review where every major claim is tagged with one of five verdicts and linked to a paper section, a literature neighbor, or a number it reproduced by running the paper's code. The image above is a real FactReview output for [CompGCN](https://arxiv.org/abs/1911.03082) — design-axis positioning, color-coded verdicts, paper-vs-reproduced numbers with Δ, and auto-synthesized strengths and weaknesses, on one page. It is **deliberately designed as a one-minute review aid**: as ML submission volumes outrun reviewer capacity and per-paper attention shrinks, the bottleneck is no longer "can a reviewer read the paper" but "can they triage ten of them this week."

> **Two tools in this repo.**
> **[FactReview](#quick-start--factreview)** takes a paper and returns a full evidence-grounded review.
> **[RefCopilot](#quick-start--refcopilot)** takes any bibliography (`.bib`, PDF, or URL) and returns a list of fabricated, retracted, outdated, or incomplete citations — plus copy-paste-ready BibTeX corrections. They share infrastructure but run independently.

## Why FactReview

Generic LLM reviewers — pasting a PDF into ChatGPT, Gemini, or Claude — fail in five predictable ways. FactReview is built around fixing each of them.

| Generic LLM reviewer | FactReview |
|---|---|
| **Hallucinates citations** ("As shown by Smith et al., 2021…" — no such paper). | Every reference is verified in parallel against arXiv, Semantic Scholar, OpenReview, and (optional) OpenAlex. Fabrications, retractions, and arXiv withdrawals are flagged. See [RefCopilot](#what-refcopilot-produces). |
| **Calls Δ = 0.3% a "significant improvement"** because the paper said so. | The claim-audit pass auto-downgrades comparative claims when Δ < 2σ and flags missing ablations. See [`src/review/report/claim_audit.py`](src/review/report/claim_audit.py). |
| **Faults the paper for not citing related work that was published after it.** | When the input is an arXiv URL, FactReview derives a publication-date cutoff from the arXiv ID and applies it to retrieval (server-side at Semantic Scholar and client-side in the agent's `paper_search`). |
| **Cannot verify a single reported number.** | Optional Docker-based execution stage runs the paper's repository on its claimed benchmarks via a `prepare → plan → run → judge → fix → finalize` loop, then reports paper-vs-reproduced deltas. See [`src/fact_generation/execution/stage_runner.py`](src/fact_generation/execution/stage_runner.py). |
| **Returns a wall of prose** that a tired reviewer still has to read end-to-end before they can decide if the paper is worth deeper attention. | A fixed-layout **one-page teaser figure** (the hero image above) lets a reviewer or area chair triage a paper in roughly a minute — positioning, verdicts, paper-vs-reproduced numbers, and weaknesses always in the same on-screen regions. The full review is one click away when the teaser raises a question. Increasingly the deciding factor as ML submission volumes outrun reviewer capacity. |

## See it in Action

The hero image is FactReview's most condensed deliverable: a one-page **teaser** purpose-built for triage. The layout is fixed — verdicts and positioning in the top half, reproduced numbers and synthesized strengths/weaknesses in the bottom half — so a reviewer's eye lands on the same regions across every paper, and the whole thing is legible without scrolling. Each panel earns its place:

- **Technical Positioning** (top-left) — The paper is placed against neighbor methods on a small set of design axes pulled from related-work retrieval. Reviewers can see at a glance which dimensions the paper actually innovates on.
- **Claims** (top-center) — Every major claim the paper makes is tagged with a verdict (✓ Supported, ☑ Paper-supported, ⚠ Partially supported, ✗ In conflict, ? Inconclusive) and linked back to the section it came from.
- **Experiment / Ablation** (bottom-left) — Paper-reported numbers and FactReview's reproduced numbers are shown side by side with Δ. The CompGCN demo includes a row where the paper's "outperforms baselines" claim is downgraded — PACHYSAN actually beats CompGCN 92.6% vs 89.0% on Graph Classification (MUTAG).
- **Summary / Strengths / Weaknesses** (right) — Auto-synthesized, including weaknesses the paper itself does not own up to (e.g., "random seeds and significance testing not reported").

For the full review behind that image, open [`demos/Graph/compgcn/report.pdf`](demos/Graph/compgcn/report.pdf).

**More demos** (each contains the full run artifacts and rendered review):

| Domain | Papers |
|---|---|
| Graph | [CompGCN](demos/Graph/compgcn) · [Graphormer](demos/Graph/graphormer) · [SACN](demos/Graph/sacn) |
| Image | [BEiT](demos/Image/beit) · [FixMatch](demos/Image/fixmatch) · [LRCN](demos/Image/lrcn) · [UDA](demos/Image/uda) |
| Text | [BERT](demos/Text/bert) · [Prefix-Tuning](demos/Text/Prefix-Tuning) |

> The teaser figure is the format we recommend for first-pass triage; the full Markdown + PDF review at `final_review.{md,pdf}` is what reviewers should open when a teaser raises questions. Rendering the figure itself requires Gemini (or pasting the saved prompt into the Gemini web app — FactReview writes the prompt to disk and copies it to your clipboard if no key is set). The Markdown + PDF review is generated unconditionally; only the teaser figure is gated on Gemini. See [Configuration](#configuration).

## Quick Start — FactReview

Requirements: Python 3.11+, a local Codex login. Docker is only needed if you enable code execution.

```bash
git clone https://github.com/DEFENSE-SEU/FactReview.git && cd FactReview
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[runtime]"
codex login                                          # ChatGPT sign-in flow
cp .env.example .env                                 # then set MINERU_API_TOKEN
python scripts/execute_review_pipeline.py demos/Graph/compgcn/paper.pdf
```

When the run finishes, open the headline output:

```
runs/<paper_key>_<timestamp>/stages/review/report/final_review.pdf
```

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

Every major claim in the paper is tagged with one of five verdicts:

| Verdict | What it means | From the CompGCN demo |
|---|---|---|
| **✓ Supported** | Independent literature evidence (or a reproduced number) agrees with the claim. | *"Scales with relations via basis decomposition"* — verified that performance is stable while parameter count scales linearly with `B` (Section 6.3, Figure 3). |
| **☑ Paper-supported** | The paper itself supports the claim and the supporting argument is sound; no external corroboration was attempted (e.g., a mathematical proof). | *"Generalizes prior multi-relational GCNs"* — the mathematical reduction to R-GCN / Kipf-GCN is logically sound (Proposition 4.1). |
| **⚠ Partially supported** | Evidence agrees with part of the claim and disagrees with or fails to address the rest. | *"Outperforms baselines in Link Prediction, Node Classification, and Graph Classification"* — verified on the first two; on Graph Classification, PACHYSAN beats CompGCN 92.6% vs 89.0% (Tables 3 and 5). |
| **✗ In conflict** | Independent evidence directly contradicts the claim. | — |
| **? Inconclusive** | Neither external nor in-paper evidence is sufficient to judge. | — |

What makes the report distinctive (beyond the verdicts):

- **Design-axis positioning matrix** — neighbor papers retrieved from Semantic Scholar (and optional OpenAlex) are placed on a small set of design dimensions specific to the paper's domain, so reviewers can see which dimensions the paper genuinely innovates on. See [`src/fact_generation/positioning/stage_runner.py`](src/fact_generation/positioning/stage_runner.py).
- **Statistical-rigor downgrades** — comparative claims with Δ < 2σ are auto-downgraded; missing ablations are flagged. See [`src/review/report/claim_audit.py`](src/review/report/claim_audit.py).
- **Publication-date cutoff** — when the input is an arXiv URL or ID, FactReview derives a `YYYY-MM` cutoff from the arXiv identifier so the manuscript is not penalized for missing citations to work that was published after it. Override with `--cutoff-date` or disable with `--no-cutoff`.
- **Optional code execution** — the execution stage runs a bounded `prepare → plan → run → judge → fix → finalize` Docker loop (default `--max-attempts 5`) and writes its verdict into `stages/fact_generation/execution/execution.json`. See [`src/fact_generation/execution/stage_runner.py`](src/fact_generation/execution/stage_runner.py).
- **One-page teaser figure for triage** — alongside the Markdown + PDF, FactReview generates a layout-constrained one-page figure (the hero image of this README) that compresses the positioning matrix, claim verdicts, paper-vs-reproduced numbers, and strengths/weaknesses into a single screen. The layout is fixed across every paper so a reviewer's eye lands on the same regions every time — designed for the realistic per-paper budget reviewers and area chairs actually have.

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

**Gemini teaser figure (optional).** If `GEMINI_API_KEY` is empty, FactReview writes the prompt to `teaser_figure_prompt.txt`, copies it to your clipboard, and tells you to paste it into the Gemini web app. If `GEMINI_API_KEY` is set, FactReview uses it automatically. Force prompt-only with `--teaser-mode prompt` or `TEASER_USE_GEMINI=false`.

<details>
<summary><strong>Prompt-only Gemini workflow (manual upload)</strong></summary>

The prompt refers to "the attached reference image" — when you paste it into Gemini / ChatGPT / any image-model web UI, **also upload a layout reference image in the same message**. The recommended reference is `demos/Graph/compgcn/teaser_figure.png`, which the prompt's geometry constraints are written against. Without this image the model has nothing to anchor the layout to and tends to leave panels empty. Override which file is used with `TEASER_TEMPLATE_REFERENCE_PNG=path/to/your_template.png` in `.env`.

</details>

**Semantic Scholar (recommended).** Set `SEMANTIC_SCHOLAR_API_KEY` to avoid rate limits during positioning retrieval. Free key at <https://www.semanticscholar.org/product/api>.

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

Or globally via `FACTREVIEW_ENABLE_REFCHECK=true`. The full result lands in `stages/fact_generation/refcheck/reference_check.json`; the Markdown summary appended to `final_review.md` lists fabricated references only. For the complete breakdown, run RefCopilot's standalone CLI or read the JSON directly. The report sub-stage also writes `final_review_clean.md` (without the refcheck section) for the teaser sub-stage.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--llm-provider` | `openai-codex` | Switches the LLM provider. Mirrors to `MODEL_PROVIDER`. |
| `--llm-model` | provider default | Mirrors to `AGENT_MODEL`, `EXECUTION_OPENAI_MODEL`, and `OPENAI_CODEX_MODEL` (when the provider is Codex). |
| `--mineru-api-token` | from `.env` | One-off override for `MINERU_API_TOKEN`. |
| `--gemini-api-key` | from `.env` | One-off override for `GEMINI_API_KEY`. |
| `--teaser-mode` | `auto` | `auto` = use Gemini when `GEMINI_API_KEY` is set, otherwise prompt-only. `prompt` = always prompt-only. `api` = always attempt the Gemini image API. |
| `--enable-refcheck` | off | Run RefCopilot as the refcheck stage. |
| `--run-execution` | off | Enables the code-execution stage. Requires Docker. |
| `--max-attempts` | `5` | Max iterations of the execution stage's `judge → fix` loop. |
| `--no-pdf-extract` | off | Skip MinerU re-extraction inside the execution `prepare` node when the parse stage already produced the snapshot. |
| `--reuse-job-id` | – | Reuse a prior agent-runtime job, skipping the parse-stage agent run. Accepts either an absolute path to a `runtime/jobs/<id>` directory (taken as-is) or a bare job id (looked up under the current run dir, then under `<run-root>/**/runtime/jobs/<id>`). Useful for re-rendering the report after a downstream-stage tweak without paying the parse cost again. |
| `--run-root` | `runs` | Override the root output directory. |
| `--cutoff-date` | auto | Inclusive publication-date cutoff for positioning retrieval, as `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`. When omitted, an arXiv URL/ID is used to auto-derive `YYYY-MM` from the arXiv identifier; for non-arXiv inputs no cutoff is applied. Both Semantic Scholar (server-side `year=` filter) and the agent's `paper_search` calls (client-side filter) are constrained to papers at or before the cutoff, so the agent does not penalise the manuscript for not citing later work. |
| `--no-cutoff` | off | Disable the publication-date cutoff entirely (overrides `--cutoff-date` and arXiv auto-derivation). Useful for analysing how the paper compares against later work. |

### Single-Stage Reruns

Each stage has a standalone script that reads the same per-run layout. `parse` takes the original PDF (because the bridge state may not exist yet); the rest work off the run dir alone:

```bash
python scripts/execute_stage_parse.py          path/to/paper.pdf --run-dir runs/<run>
python scripts/execute_stage_claim_extract.py  --run-dir runs/<run>
python scripts/execute_stage_refcheck.py       --run-dir runs/<run>
python scripts/execute_stage_positioning.py    --run-dir runs/<run>
python scripts/execute_stage_execution.py      --run-dir runs/<run>
python scripts/execute_stage_report.py         --run-dir runs/<run>
python scripts/execute_stage_teaser.py         --run-dir runs/<run>
```

### Outputs

Each run writes to `runs/<paper_key>_<timestamp>/`. Primary artifacts:

- `full_pipeline_summary.json` — per-stage status, error reasons, and output paths.
- `inputs/source_pdf/` — copy of the input paper PDF.
- `runtime/jobs/<job_id>/` — raw runtime job state, MinerU output, prompts, and agent traces.
- `stages/preprocessing/parse/paper.json` — parse-stage outputs and bridge state.
- `stages/preprocessing/claim_extract/` — extracted claim list.
- `stages/fact_generation/refcheck/` — reference check report (only when `--enable-refcheck`).
- `stages/fact_generation/positioning/` — literature neighbours and design-axis table.
- `stages/fact_generation/execution/current/` — in-place workspace for the latest execution attempt; the prior attempt is archived alongside as `current.<timestamp>` (only when `--run-execution`).
- `stages/fact_generation/execution/history/` — per-attempt orchestrator outputs (only when `--run-execution`).
- `stages/review/report/final_review.{json,md,pdf}` — **the headline review.**
- `stages/review/report/final_review_clean.md` — same review without the refcheck section, used by the teaser.
- `stages/review/teaser/teaser_figure_prompt.txt` — teaser figure prompt.
- `stages/review/teaser/teaser_figure.png` — teaser image (only when Gemini is enabled).

`workspace/`, `logs/`, and `debug/` are intermediate; you usually do not need to look at them.

## Pipeline Architecture

<p align="center">
  <img src="overview.png" alt="FactReview pipeline overview" width="800">
</p>

The pipeline runs seven sub-stages, grouped into three phases. `refcheck` and `execution` are skipped by default.

```text
preprocessing                fact_generation                        review
parse → claim_extract  →  refcheck? → positioning → execution? → report → teaser
```

- **parse** — PDF → structured `Paper` (MinerU cloud).
- **claim_extract** — `Paper` → list of major claims (LLM + decomposer).
- **refcheck** — bibliography validation via [RefCopilot](RefCopilot/) (off by default; `--enable-refcheck`).
- **positioning** — neighbour papers, design axes, novelty verdict.
- **execution** — optional Docker-based code-running stage (off by default; `--run-execution`).
- **report** — synthesises the final review Markdown / PDF, runs the claim audit.
- **teaser** — teaser figure prompt and (optionally) image.

## Troubleshooting

- **`codex login` fails or is not on PATH** — install OpenAI's Codex CLI (`npm install -g @openai/codex`), then rerun `codex login` and pick the ChatGPT sign-in flow.
- **`MINERU_API_TOKEN` missing** — the parse stage will raise on the first run. Get a token from <https://mineru.net> (free tier is sufficient for most papers) and set it in `.env` or pass `--mineru-api-token`.
- **`--enable-refcheck` errors with missing deps** — install with `pip install -e ".[refcheck]"`.
- **Positioning stage is slow or returns sparse results** — unauthenticated Semantic Scholar requests are rate-limited. Set `SEMANTIC_SCHOLAR_API_KEY` in `.env` (free key from <https://www.semanticscholar.org/product/api>).
- **Teaser stage skips silently / no `teaser_figure.png`** — `GEMINI_API_KEY` is unset (this is the default). The prompt is still written to `stages/review/teaser/teaser_figure_prompt.txt` and copied to your clipboard; paste it into the Gemini web app to generate the image manually.

## Advanced Configuration

Less common environment variables — set in `.env` or via the shell. `.env.example` is the authoritative list; the table below covers the ones most users will touch.

| Variable | Purpose |
|---|---|
| `FACTREVIEW_ENABLE_REFCHECK` | Enable reference checking globally (equivalent to the `--enable-refcheck` flag). |
| `FACTREVIEW_EXECUTION_ENABLE_REFCHECK` | Enable a refcheck sweep *inside* the execution stage's refcheck node. Independent from the global gate above. |
| `OPENALEX_API_KEY` | Optional OpenAlex API key. When set, OpenAlex is queried as a fourth cross-check signal in RefCopilot; when empty, OpenAlex is skipped. Free key at <https://openalex.org/settings/api>. |
| `MINERU_BASE_URL` | Override the MinerU cloud API endpoint (default: `https://mineru.net/api/v4`). |
| `MINERU_ALLOW_LOCAL_FALLBACK` | Set to `true` to let the execution stage's `prepare` node fall back to the local `mineru` CLI when the cloud snapshot is unavailable. |
| `MINERU_LOCAL_BACKEND` / `MINERU_LOCAL_DEVICE` / `MINERU_LOCAL_SOURCE` | Tune the local `mineru` CLI's pipeline backend, device, and source mirror. Only consulted when `MINERU_ALLOW_LOCAL_FALLBACK=true` and a local MinerU install is present. |
| `OPENAI_AGENTS_DISABLE_TRACING` | Set to `0` to enable the openai-agents SDK trace exporter. Disabled (`1`) by default to avoid POSTing traces to the Agents tracing endpoint. |
| `TEASER_USE_GEMINI` | Force prompt-only teaser output (`false`) even when a Gemini key is configured. Equivalent to `--teaser-mode prompt`. |
| `OPENAI_CODEX_BASE_URL` | Point Codex at a different Codex-compatible endpoint (default: `https://chatgpt.com/backend-api/codex`). |
| `SEMANTIC_SCHOLAR_API_KEY` | Recommended. Free API key from [Semantic Scholar](https://www.semanticscholar.org/product/api) for the positioning stage. Without it, unauthenticated requests may be rate-limited. |

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
pytest -m e2e                   # report-audit + teaser tail integration
pytest -m requires_docker       # execution stage (needs Docker daemon)
pytest -m ""                    # full set, including all gated tests
```

## Paper

Read the paper on <https://arxiv.org/abs/2604.04074> or from the local PDF at [`factreview.pdf`](factreview.pdf).

If FactReview helped your work, please ⭐ the repo and cite:

```bibtex
@misc{xu2026factreview,
  title = {FactReview: Evidence-Grounded Reviews with Literature Positioning and Execution-Based Claim Verification},
  author = {Xu, Hang and Yue, Ling and Ouyang, Chaoqian and Liu, Yuchen and Zheng, Libin and Pan, Shaowu and Di, Shimin and Zhang, Min-Ling},
  year = {2026},
  eprint = {2604.04074},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  doi = {10.48550/arXiv.2604.04074},
  url = {https://arxiv.org/abs/2604.04074}
}
```

## License

AGPL-3.0-only.
