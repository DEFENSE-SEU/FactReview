# FactReview <a href="https://arxiv.org/abs/2604.04074"><img src="https://img.shields.io/badge/arXiv-2604.04074-b31b1b.svg" alt="Paper"></a>

<p align="center">
  <img src="overview.png" alt="FactReview Overview" width="800">
</p>

Evidence-grounded AI reviewing for empirical ML papers. Given a paper PDF,
FactReview extracts the major claims, positions the paper against nearby
literature, and writes a concise review linked to evidence. Repository/code
execution is available, but disabled by default.

## Quick Start

**Requirements:** Python 3.11+ and a local Codex login.
Docker is only needed if you explicitly enable code execution.

```bash
git clone https://github.com/DEFENSE-SEU/FactReview.git
cd FactReview

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[runtime]"
codex login

cp .env.example .env
# Then open .env and set MINERU_API_TOKEN (free tier from https://mineru.net is
# enough for most papers). See "Required Configuration" below if you want to
# override the LLM backend or enable Gemini.
```

If `codex` is not on your PATH, install the Codex CLI first, then rerun
`codex login`.

Run the pipeline against the bundled CompGCN PDF to verify your install — no
need to supply your own paper:

```bash
python scripts/execute_review_pipeline.py demos/compgcn/paper.pdf
```

Once the demo works, see [Running](#running) below for arbitrary PDFs and
arXiv URLs. The headline output is always:

```text
runs/<paper_key>_<timestamp>/stages/review/report/final_review.md
```

## Required Configuration

FactReview intentionally keeps routine configuration to two places:

- `.env` / environment variables for secrets and normal runtime choices.
- CLI flags for one-off overrides.

### LLM Backend

The default backend is Codex login. The model id and provider are pre-filled
in `.env.example` (currently `MODEL_PROVIDER=openai-codex` with the Codex
backend's expected model alias) — copy that file to `.env` and you're done.
The Codex model alias is *not* a public OpenAI Platform model id, so don't try
to use it with `OPENAI_API_KEY` against `api.openai.com`.

Run `codex login` once and choose the ChatGPT sign-in flow. FactReview reads
the local Codex OAuth cache, so no OpenAI Platform API key is needed for the
default path.

### MinerU PDF Parsing

`MINERU_API_TOKEN` is required. FactReview uses MinerU's cloud API by default
because it is free to start with generous quota and avoids local CUDA/GPU and
MinerU model setup.

```bash
MINERU_API_TOKEN=your_mineru_token
```

You can also pass it once from CLI:

```bash
python scripts/execute_review_pipeline.py paper.pdf --mineru-api-token your_mineru_token
```

### Gemini Teaser Figure

Gemini is optional. If `GEMINI_API_KEY` is empty, FactReview treats teaser
generation as successful prompt-only output: it writes
`teaser_figure_prompt.txt`, copies the prompt to the clipboard when possible,
and tells you to paste it into the Gemini web app. If `GEMINI_API_KEY` is set,
FactReview uses it automatically.

```bash
GEMINI_API_KEY=
```

To force prompt-only output even when a Gemini key is configured, pass
`--teaser-mode prompt` on the CLI (or set `TEASER_USE_GEMINI=false` in `.env`).

## Running

Full default pipeline on a local PDF:

```bash
python scripts/execute_review_pipeline.py path/to/paper.pdf --paper-key my_paper
```

You can also pass a PDF URL. arXiv abstract links are normalized to the PDF
download automatically:

```bash
python scripts/execute_review_pipeline.py https://arxiv.org/abs/1911.03082 --paper-key compgcn
```

This runs the pipeline's seven sub-stages, grouped into three phases. `refcheck`
and `execution` are skipped by default; see the flags below to enable them.

```text
preprocessing                fact_generation                        review
parse → claim_extract  →  refcheck? → positioning → execution? → report → teaser
```

- **parse** — PDF → structured `Paper`.
- **claim_extract** — `Paper` → list of major claims.
- **refcheck** — bibliography validation (off by default; see `--enable-refcheck`).
- **positioning** — neighbour papers, design axes, novelty verdict.
- **execution?** — optional code-running stage (off by default; see `--run-execution`).
- **report** — synthesises the final review markdown / PDF.
- **teaser** — teaser figure prompt and (optionally) image.

Code execution is off by default. Enable it only when you want repository/code
evaluation. The execution stage drives Docker via the `docker` CLI, so the only
prerequisite beyond the `runtime` extra is a working Docker daemon — no
additional Python packages need to be installed:

```bash
python scripts/execute_review_pipeline.py path/to/paper.pdf --run-execution
```

Reference-accuracy checking is also off by default. Enable it when you want
RefChecker to validate bibliography entries and append warning/error details to
the final report. RefChecker ships as a git submodule, so fresh clones need a
one-time setup:

```bash
git submodule update --init --recursive
pip install -e ".[refcheck]"
python scripts/execute_review_pipeline.py path/to/paper.pdf --enable-refcheck
```

You can also enable it through the environment with `FACTREVIEW_ENABLE_REFCHECK=true`.
Results are written under `stages/fact_generation/refcheck/reference_check.*` and
appended to `stages/review/report/final_review.md`. The report sub-stage also writes
`final_review_clean.md` (without the refcheck section); the teaser sub-stage reads
that clean copy so refcheck warnings don't pollute teaser prompts.

### CLI Flags

Common one-off overrides:

| Flag | Default | Notes |
|---|---|---|
| `--llm-provider` | `openai-codex` | Switches the LLM provider. Mirrors to `MODEL_PROVIDER`. |
| `--llm-model` | provider default | Mirrors to `AGENT_MODEL`, `EXECUTION_OPENAI_MODEL`, and `OPENAI_CODEX_MODEL` (when the provider is Codex). |
| `--mineru-api-token` | from `.env` | One-off override for `MINERU_API_TOKEN`. |
| `--gemini-api-key` | from `.env` | One-off override for `GEMINI_API_KEY`. |
| `--teaser-mode` | `auto` | `auto` = use Gemini when `GEMINI_API_KEY` is set, otherwise prompt-only. `prompt` = always prompt-only. `api` = always attempt the Gemini image API. |
| `--enable-refcheck` | off | See above. |
| `--run-execution` | off | Enables the code-execution stage. Requires Docker. |
| `--max-attempts` | `5` | Max iterations of the execution stage's `judge → fix` loop. |
| `--no-pdf-extract` | off | Skip MinerU re-extraction inside the execution `prepare` node when the parse stage already produced the snapshot. |
| `--reuse-job-id` | – | Reuse a prior agent-runtime job, skipping the parse-stage agent run. Accepts either an absolute path to a `runtime/jobs/<id>` directory (taken as-is) or a bare job id (looked up under the current run dir, then under `<run-root>/**/runtime/jobs/<id>`). Useful for re-rendering the report after a downstream-stage tweak without paying the parse cost again. |
| `--run-root` | `runs` | Override the root output directory. |

### Single-Stage Reruns

Each stage has a standalone script that reads the same per-run layout. `parse`
takes the original PDF (because the bridge state may not exist yet); the rest
work off the run dir alone:

```bash
python scripts/execute_stage_parse.py          path/to/paper.pdf --run-dir runs/<run>
python scripts/execute_stage_claim_extract.py  --run-dir runs/<run>
python scripts/execute_stage_refcheck.py       --run-dir runs/<run>
python scripts/execute_stage_positioning.py    --run-dir runs/<run>
python scripts/execute_stage_execution.py      --run-dir runs/<run>
python scripts/execute_stage_report.py         --run-dir runs/<run>
python scripts/execute_stage_teaser.py         --run-dir runs/<run>
```

## Outputs

Each run writes to:

```text
runs/<paper_key>_<timestamp>/
```

Primary artifacts:

- `full_pipeline_summary.json` — per-stage status, error reasons, and output paths.
- `inputs/source_pdf/` — copy of the input paper PDF.
- `runtime/jobs/<job_id>/` — raw runtime job state, MinerU output, prompts, and agent traces.
- `stages/preprocessing/parse/paper.json` — parse-stage outputs and bridge state.
- `stages/preprocessing/claim_extract/` — extracted claim list.
- `stages/fact_generation/refcheck/` — reference check report (only when `--enable-refcheck`).
- `stages/fact_generation/positioning/` — literature neighbours and design-axis table.
- `stages/fact_generation/execution/current/` — in-place workspace for the latest execution attempt; the prior attempt is archived alongside as `current.<timestamp>` (only when `--run-execution`).
- `stages/fact_generation/execution/history/` — per-attempt orchestrator outputs (only when `--run-execution`).
- `stages/review/report/final_review.{json,md,pdf}` — the headline review.
- `stages/review/report/final_review_clean.md` — same review without the refcheck section, used by the teaser.
- `stages/review/teaser/teaser_figure_prompt.txt` — teaser figure prompt.
- `stages/review/teaser/teaser_figure.png` — teaser image (only when Gemini is enabled).

The run dir also contains `workspace/`, `logs/`, and `debug/` directories used
for intermediate artefacts; you usually don't need to look at these.

## What the Pipeline Produces

The final review tags every claim with one of five judgments:

- **Supported** — independent literature evidence agrees with the claim.
- **Supported by the paper** — only the paper itself supports the claim; no external corroboration was found.
- **Partially supported** — evidence agrees with part of the claim and disagrees with or fails to address the rest.
- **In conflict** — independent evidence contradicts the claim.
- **Inconclusive** — neither external nor in-paper evidence is sufficient to judge.

When `--run-execution` is on, the execution stage runs a bounded
`prepare → plan → run → judge → fix → finalize` loop (default `--max-attempts 5`)
and writes its verdict into `stages/fact_generation/execution/execution.json`.

## Troubleshooting

- **`codex login` fails or is not on PATH** — install OpenAI's Codex CLI
  (`npm install -g @openai/codex`), then rerun `codex login` and pick the
  ChatGPT sign-in flow.
- **`MINERU_API_TOKEN` missing** — the parse stage will raise on the first run.
  Get a token from <https://mineru.net> (free tier is sufficient for most
  papers) and set it in `.env` or pass `--mineru-api-token`.
- **`--enable-refcheck` errors with "tools/refchecker not found"** — RefChecker
  is a git submodule that fresh clones don't pull by default. Run
  `git submodule update --init --recursive` and `pip install -e ".[refcheck]"`.
- **Positioning stage is slow or returns sparse results** — unauthenticated Semantic Scholar requests are rate-limited. Set `SEMANTIC_SCHOLAR_API_KEY` in `.env` (free key from <https://www.semanticscholar.org/product/api>).
- **Teaser stage skips silently / no `teaser_figure.png`** — `GEMINI_API_KEY`
  is unset (this is the default). The prompt is still written to
  `stages/review/teaser/teaser_figure_prompt.txt` and copied to your
  clipboard; paste it into the Gemini web app to generate the image
  manually.

## Advanced Configuration

Less common environment variables — set in `.env` or via the shell. `.env.example`
is the authoritative list; the table below covers the ones most users will
touch.

| Variable | Purpose |
|---|---|
| `FACTREVIEW_ENABLE_REFCHECK` | Enable RefChecker globally (equivalent to the `--enable-refcheck` flag). |
| `FACTREVIEW_EXECUTION_ENABLE_REFCHECK` | Enable a refcheck sweep *inside* the execution stage's refcheck node. Independent from the global gate above. |
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
pytest tests/unit -m "not slow and not e2e and not requires_docker and not requires_llm and not requires_mineru"
```

## Paper

Read the paper on https://arxiv.org/abs/2604.04074 or from the local
PDF at [`factreview.pdf`](factreview.pdf).

If you use FactReview, please cite:

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
