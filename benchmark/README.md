# Benchmark papers

`papers.csv` lists the 300 papers used to evaluate FactReview. Every paper has public peer reviews. PDFs, OpenReview review threads and each paper's license are in the Hugging Face dataset [kabuda777/factreview-bench](https://huggingface.co/datasets/kabuda777/factreview-bench); `platform_id` (OpenReview forum id) and `doi` (Nature Communications) point to the originals.

| Group | Source | Papers |
|---|---|---|
| `ai_generated` | Agents4Science 2025 | 52 |
| `top_conf_accepted` | ICLR 2025, accepted | 52 |
| `top_conf_rejected` | ICLR 2025, rejected | 52 |
| `journal` | TMLR, reviews public 2025-01-01 to 2026-06-30 | 52 |
| `cross_discipline` | Nature Communications with peer review files | 26 |
| `fresh` | COLM 2026 accepted, ICML 2026 rejected (public), TMLR after 2026-06-30 | 66 |

`depth` is `precise` for the 60 papers annotated in full and `light` for the other 240. `fresh` papers were reviewed after the training cutoff of the evaluated models and measure contamination.

Columns:

- `paper_id`: `or_<forum id>` for OpenReview, `nc_<article id>` for Nature Communications.
- `decision`, `venue_tier`: venue outcome; `venue_tier` is oral/spotlight/poster where the venue reports it.
- `domain_label`: venue primary area or first keyword; labels are not comparable across groups and `unknown` means missing.
- `has_code`, `code_executability`: whether a repository was found, and a keyword-based first guess at what its README supports.
- `pages`: page count of the reviewed PDF.
- `review_public_date`: date the reviews became public.

The sample was drawn with seed 20261009, one stream per group.
