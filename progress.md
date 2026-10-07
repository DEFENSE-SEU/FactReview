# FactReview v2 progress

## Task

Implement `docs/method_v2_spec.md` through Phases 0–8 on `refactor/method-v2`, following the adopted prompt and maintainer decisions. Use independent automatic review at each phase and continue without phase pauses. Preserve tests and mock LLM/retrieval/Docker in unit tests. Keep RefCopilot internals, demo reference outputs and LICENSE unchanged. The documentation PR is published; refactor commits remain local. Publishing this refactor would require the separate approval specified by the adopted prompt.

## Outputs

Workspace: `E:\kabuda\FactReview`.

- Documentation PR: https://github.com/DEFENSE-SEU/FactReview/pull/12 — requested title/body, base `main`, exactly two documentation files.
- Docs branch: `docs/method-v2-spec`, patch `ab983b9`, continuous automatic-review amendment `281e0f5`; amendment applied locally as `84c2aeb`.
- Refactor branch: `refactor/method-v2`, created from docs branch. No remote branch or refactor PR has been created.
- Authoritative specification and procedure: `docs/method_v2_spec.md`, `docs/refactor_v2_prompt.md`.
- Final acceptance evidence: `checks.md`; durable handoff: this file.
- Schemas/compatibility: `src/schemas/{claim,review,materials,legacy_claim,legacy_review,v1_adapter}.py`.
- Shared materials: `src/preprocessing/materials.py`.
- Screening: `src/screening/`; typed verification: `src/verification/`; assessment: `src/assessment/`.
- Execution: `src/fact_generation/execution/{v2,v2_config,v2_outputs}.py` and reused Docker/alignment/plan infrastructure.
- Report and teaser: `src/review/report/v2.py`, `src/review/teaser/v2.py`, existing PDF renderer.
- Default entry: `src/pipeline_full.py` → `src/pipeline_v2.py`; `scripts/execute_review_pipeline.py` remains the CLI wrapper.
- Integration fixture: `scripts/check_v2_compgcn.py`; regression tests under `tests/`, including `test_pipeline_v2.py` and `test_run_stats_v2.py`.
- Local evidence: `runs/v2_baseline/`, `runs/v2_materials/`, `runs/v2_screening/`, `runs/v2_verification/`, `runs/v2_execution/`, `runs/v2_report/`, `runs/v2_final/`.
- Current CompGCN fixture: `runs/v2_compgcn/compgcn_fixture_2026-10-08_050820_50ebee36/`, including `comparison.{json,md}`, `full_pipeline_summary.json`, `review/report/final_review.{json,md,pdf}`, `review/teaser/teaser.{json,svg}`.
- Current wheel: `runs/v2_packaging_final/factreview-0.1.0-py3-none-any.whl`; final build includes the PDF literal-pointer correction.

Logs and generated artifacts in `runs/` are local and ignored by Git. Saved commands and outcomes here allow another checkout to reproduce the verification.

## Completed

Fetched and fast-forwarded main from `87dad71` to `5dc71d9`, applied the attached patch with `git am`, then opened documentation PR #12 with `PR_DESCRIPTION.md`. No code is included in that PR. The maintainer replaced phase pauses with independent automatic review; that amendment is the only change to the supplied procedure.

| Phase | Work and verification | Commit |
|---|---|---|
| 0 | Existing baseline: 256 passed, 3 original deselections after the Windows short-temp correction. Credential exception recorded for live CompGCN. Independent `baseline_review`. | `ac5840f` |
| 1 | Strict claim/evidence/plan/finding/question contracts, four statuses, one-way historical adapter. 62 schema cases and 2 mocked legacy integration cases passed. Independent `phase_reviewer`. | `f2888b0` |
| 2 | Direct MinerU materials, original text/blocks, page/figure rendering, reference links, bibliography and read-only repository index. 25 cases passed. Independent `phase_reviewer`. | `f15dbd7` |
| 3 | Upfront claim extraction, writing/figure/reference checks and image transport. Removed report extraction duties and C1–C3 cap. 45 affected cases passed. Independent `phase_reviewer`. | `f055600` |
| 4 | Exact needs dispatch, four parallel peer branches, global Literature findings and retrieval constraints. 202 affected cases passed. Durable offline/Windows pytest fixtures. Independent `phase_reviewer`. | `6fe09d2` |
| 5 | Approved execution, ordering/budgets, exact observed alignment, bounded repairs, released artifact provenance, output normalization and ledger. 61 v2 cases; 218 affected cases passed with 1 original Docker deselection. Independent `integration_map`. | `4502aca` |
| 6 | Pure ordered assessment, conflict/flaw/concern/coverage rules and non-decisive notes. 16 cases passed. Independent `phase_reviewer`. | `e2504db` |
| 7 | Four-part report and all-claim four-status teaser, source/pointer preservation, recommendation guards and literal Markdown escaping. 11 cases passed at phase checkpoint. Independent `phase_reviewer`. | `32e1785` |
| 8 | Default v2 integration, explicit deadline/config CLI, usage/error propagation, README, wheel packaging and CompGCN comparison. Final PDF pointer correction, full regression and independent artifact review passed. | Final Phase 8 checkpoint commit (`refactor(v2): phase 8 — integrate and verify the complete claim review pipeline`) |

Final full default suite: **615 passed, 3 original deselections, exit 0** in `runs/v2_final/pytest-full.log`. The two deselected legacy e2e cases passed separately (`pytest-legacy-e2e.log`); the remaining live Docker case was not run. No new skips or relaxed assertions. Statistics/pipeline focus: 18 passed. Ruff passed across new implementation modules and Phase 8 files. Formatting of three new Phase 8 files retained identical ASTs. The wheel contains both pipeline entry modules and all v2 packages; 110 packaged Python modules parsed with the Python 3.11 grammar, and required packaged files matched workspace bytes (`wheel-inspection.json`).

### Corrections and preserved failures

- Phase 0: initial 252 passed/4 failed. Existing production logic routes long Windows venv paths to `C:\frv-venvs`, while four tests require a short run-local path. A fresh short `--basetemp` passed all original assertions. Phase 4 made that fixture durable under `runs/pytest/<unique-id>` without changing production behavior.
- Phase 1: rejected ambiguous/non-finite execution targets; fixed historical Markdown escaped pipes, malformed rows and emphasized labels. The adapter never invents missing locations or sufficient evidence.
- Phase 2: fixed appendix/Oxford-list figure references, list-item loss and crop rounding. Unknown parser content remains an explicit issue.
- Phase 3: rejected zero-reference-processing success and model failure containers with empty findings. Original retrieval assertions retained.
- Phase 4: fixed reader payloads promoting abstracts into sufficient evidence, unknown identity before reading, concurrent citations, malformed retrieval results, duplicate paper quotes, missing source files, code indentation and ambiguous numeric target binding. Initial network isolation blocked Windows asyncio's internal wakeup socketpair (137 passed/65 setup errors); the fixture now allows only that scoped internal connection. Same assertions passed. Logs retain initial command-path and isolation errors.
- Phase 5: fixed expected/paper target selectors masquerading as runtime values, failed output containers/canonical rows, conflicting actual metric mappings and runtime selection of supposed author artifacts. Operator contracts are frozen before execution; tests reject each invalid path.
- Phase 7: strengthened recommendation guards across fields/whitespace and preserved literal Markdown content. PDF extraction whitespace was normalized for line wrapping while retaining the required text values. Final artifact review found PDF implicit-formula detection corrupting Windows paths and snake_case identifiers. V2 now disables that heuristic for ordinary text and inline code; explicit formula tokens still render and v1 defaults stay unchanged. Two exact PDF/literal-mode regressions were added (13 report cases in the final suite; 28 report/legacy focused cases passed). The fresh five-page PDF preserves all evidence locator paths and claim IDs. Independent rendered-page inspection found no clipping/overlap; `runs/v2_final/pdf-literal-inspection.json` records the root-agent check. Acceptance check 12 passes.
- Phase 8: corrected wheel entry-module omissions, outdated CLI help, failure statistics, explicit UTF-8 fixture reads and high-resolution timing for immediate failures. The 80-call parallel statistics test exposed a Windows replacement error: write locking alone and read locking did not cover public path resolution. The third correction puts `Path.resolve()` under the same RLock; original 80-request/160-input/240-output/400-total assertions and the full suite pass. No evidence assigns this failure to antivirus. Prior failures remain in `pytest-full-before-path-lock.log` and Phase 8 local logs.
- Packaging: `python -m build` was unavailable; non-isolated pip build also lacked the hatchling backend. Existing pip isolated build installed declared build dependencies and passed. Both errors and successful output are saved separately. An unused test variable was renamed without changing assertions; final formatting was AST-equivalent.

## Decisions

- Confirmed by the maintainer: repair default/cap **3** (initial run plus at most 3 repaired runs); default rule-based **auto** approval with optional interactive approval and the actual mode recorded; training budget **0 runs**, configurable, high-priority training only, retries counted.
- Confirmed cutoff: explicit `--submission-deadline YYYY-MM-DD`, with **3 calendar months** before it labelled concurrent. No cutoff derived from the paper's own arXiv ID. No historical venue deadline is invented for the fixture.
- Confirmed tolerances: central `v2_config.TOLERANCES` preserves both existing plan and alignment profiles, including their different MRR defaults.
- Confirmed v1 compatibility: historical `in_conflict` maps to `questioned`; original records/labels are retained. Historical display does not reassess evidence or create a v2 claim lacking required locations.
- `--execution-config` loads a validated `ExecutionConfig`; only explicitly supplied CLI options override corresponding fields, including an explicit zero budget. Unsupported legacy options produce recorded errors. Docker is required when execution is enabled.
- Use the existing MinerU adapter directly for shared materials because the legacy parse runtime also performs downstream review/retrieval. Reuse RefCopilot, Docker helpers and PDF infrastructure. Legacy entry/functions remain isolated for compatibility.
- Repository indexing is read-only and hashes actual files. Approved L3 execution creates a separate workspace; source mutations outside the allowlist are rejected and accepted repair diffs saved. Reproduction uses exact observed dataset/metric/settings.
- Ambiguous paper targets remain blocked. Blocked plans attach explanatory evidence with `affects_claim=False`. Failed or unaligned attempts retain their ledger reasons and author questions without adding deciding execution evidence. Aligned mismatches normally remain explainable concerns.
- A decisive released-artifact inconsistency requires a pre-run operator contract: indexed path, SHA-256, released data/log role, metadata selectors and deterministic recomputation. Runtime output cannot choose an artifact or replace its recomputed value. The operator confirms the artifact's role.
- Optional `paper_variances` require a real block, exact quote and nonnegative value. The operator confirms statistic meaning, units and condition binding. These semantics are not established by quote matching; automatic branches leave this contract empty.
- Theory, Code and paper-only Experiments retain explainable differences as concerns. No theorem prover was added. Default Literature retrieval does not certify complete search, so absence alone cannot support novelty; restricted technical vocabulary leaves uncovered domains visibly unresolved.
- Figure boxes have explicit units; locations use 1-based PDF pages. Page/crop rendering uses 200 dpi, printed-size checks 96 dpi. Missing/invalid images remain issues. Tables use parsed text.
- Reports consume deterministic assessed records. Teasers render locally to SVG; no image service. Unknown usage is marked unavailable; estimates are labelled. No claim is made about model semantic accuracy from mocked responses.
- Independent agents inherit configured model/effort without overrides. Exact runtime identifiers were unavailable. Git commits use per-command identity `ChaoqianO <224349230+ChaoqianO@users.noreply.github.com>`, taken from the authenticated account; global Git config was unchanged. Dependencies are isolated in `.venv`.

## CompGCN comparison and open issues

The latest offline replay uses the real 15-page PDF and read-only released repository, with explicitly fixed parser/model/retrieval/Docker boundaries. All seven stages completed. It yields **2 supported, 1 unverified**; the historical adapter reads **2 supported, 1 questioned**. Claim sets differ, so these counts do not establish improved accuracy. The Theory fixture covers one explicit reduction; other conditions stay unverified. The multi-value results table remains blocked for L3: **zero Docker runner calls**, no independent numerical reproduction. Paper-internal support retains its source label. All 27 demo-tree file hashes and the Phase 0 seven reference hashes match. There are no Git changes to RefCopilot, demos or LICENSE.

- **Live verification blocker:** repository `.env` and `MINERU_API_TOKEN` were absent at baseline and remain unconfigured. No successful live MinerU/LLM/retrieval/Docker run or live accuracy measurement is claimed. Criterion 13 explicitly permits a reported blocker. Configure credentials locally; do not put secrets in Git or chat.
- Actual historical experiment approval mode/training budget were not supplied. Confirmed defaults describe new runs only.
- Live extraction, figure/bibliography parsing and retrieval quality require a real-service run. The deterministic tests establish contracts, routing, aggregation and saved artifacts.
- No implementation blocker remains. All 14 acceptance criteria pass, with the live-run exception explicitly recorded for criterion 13.

## Reproduction commands

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest tests/test_e2e_pipeline.py -m e2e -q
.\.venv\Scripts\python.exe scripts/check_v2_compgcn.py
.\.venv\Scripts\python.exe -m pip wheel . --no-deps --wheel-dir runs/v2_packaging_final
```

Original Phase 0 command used the saved ignored plugin: `python -m pytest -p baseline_isolation --basetemp runs/v2_baseline/t1`, with `src` and `runs/v2_baseline` on PYTHONPATH. Never reuse that explicit directory blindly because pytest clears `--basetemp`; use a fresh verified path. Current durable fixtures choose fresh short paths automatically.

## Next action

The requested implementation and automatic reviews are complete. For additional live evidence, configure MinerU and the desired model/services locally, supply the actual venue submission deadline, and run the v2 CLI. Refactor publishing remains a separate approval boundary; all implementation commits stay local. The open documentation PR is ready for maintainer review.
