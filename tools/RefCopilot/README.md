# RefCopilot

A focused reference-accuracy checker for academic papers. Given a PDF, URL,
BibTeX file, or plain text bibliography, RefCopilot extracts each citation and
verifies it against arXiv and Semantic Scholar, emitting:

- **Errors** for fabricated / hallucinated references that don't match any
  retrievable record.
- **Warnings** for outdated references (an arXiv preprint that's since been
  published, an older arXiv version, withdrawn papers, workshop → full upgrades).
- **Warnings** for incomplete references (missing DOI / arXiv ID / venue / year,
  truncated authors, abbreviated venue names).

Suspected hallucinations are rechecked by an LLM pass that downgrades
recognisable non-academic citations (system cards, vendor blog posts, technical
reports, dataset cards, standards, white papers) to a warning instead of a
hard error.

## Install

From the FactReview workspace root:

```bash
pip install -e "./tools/RefCopilot[dev]"
```

RefCopilot reuses FactReview's LLM client (`src/llm/client.py`). LLM
extraction is **always on** — there is no regex fallback — so the
`openai-codex` provider must be configured before running. By default the LLM
client is loaded from `<repo>/src/`; set `REFCOPILOT_FACTREVIEW_SRC` to point
at a different FactReview checkout if needed.

## CLI

```bash
refcopilot check <input>
```

`<input>` may be:
- a `.bib` file
- a `.pdf` file
- an arXiv URL or paper URL (auto-downloaded)
- plain bibliography text

Useful flags: `--output-dir DIR`, `--no-llm-verify`, `--no-cache`,
`--cache-dir DIR`, `--cache-ttl-days N`, `--max-refs N`, `--debug`.
Run `refcopilot check --help` for the full list.

Cache management:

```bash
refcopilot cache prune --ttl-days 30
```

## Library

```python
from refcopilot import RefCopilotPipeline

pipeline = RefCopilotPipeline()
report = pipeline.run("path/to/paper.pdf")
print(report.summary)
```

## FactReview integration

The `refcopilot.factreview` module is what FactReview's refcheck stage calls.
It exposes:

```python
from refcopilot.factreview import check_references, format_factreview_markdown
```

`check_references()` returns the dict written to `reference_check.json`.
`format_factreview_markdown()` renders the embedded Markdown summary that
appears in the final review report — errors only by default, since the
embedded summary is meant to flag fabrications. Pass `include_warnings=True`
or `include_unverified=True` for a full listing.

## Configuration

| Environment variable | Purpose |
|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | Optional Semantic Scholar API key (recommended to avoid rate limits). |
| `SEMANTIC_SCHOLAR_BASE_URL` | Override the S2 base URL. |
| `REFCOPILOT_FACTREVIEW_SRC` | Override the FactReview `src/` path used to load the LLM client. |

## Running the tests

```bash
pytest
```

The `pytest` configuration deselects `slow` tests by default. Run
`pytest -m slow` to include them.
