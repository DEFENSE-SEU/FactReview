# RefCopilot

A focused reference-accuracy checker for academic papers. Given a PDF, URL,
BibTeX file, or plain text bibliography, it extracts each citation and verifies
it against arXiv and Semantic Scholar, emitting:

- **Errors** for fake/hallucinated references that don't match any retrievable
  record.
- **Warnings** for outdated references (arXiv preprint that's been published,
  older arXiv version cited, withdrawn papers, workshop→full upgrades).
- **Warnings** for incomplete references (missing DOI / arXiv ID / venue / year /
  truncated authors / abbreviated venue names).

## Install

```bash
pip install -e ./tools/RefCopilot[dev]
```

RefCopilot is designed to live inside FactReview's `tools/` directory and to
reuse FactReview's LLM client (`src/llm/client.py`). LLM extraction is **always
on** — there is no regex fallback — so the FactReview `openai-codex` provider
must be configured before running.

## CLI

```bash
refcopilot check <input>
```

`<input>` may be:
- `.bib` file
- `.pdf` file
- arXiv URL or paper URL (auto-downloaded)
- plain text

Run `refcopilot check --help` for all flags.

## Library

```python
from refcopilot import RefCopilotPipeline

pipeline = RefCopilotPipeline()
report = pipeline.run("path/to/paper.pdf")
print(report.summary)
```

For backward compatibility with FactReview's existing `refcheck` stage:

```python
from refcopilot.compat import check_references, format_reference_check_markdown
```

These match the signatures of the original `refchecker` adapter.
