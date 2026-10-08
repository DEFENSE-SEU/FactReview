# FactReview v2 progress

## Task

Implement `docs/method_v2_spec.md` through Phases 0–8 on `refactor/method-v2`, following the adopted prompt and maintainer decisions. Use independent automatic review at each phase and continue without phase pauses. Preserve tests and mock LLM/retrieval/Docker in unit tests. Keep RefCopilot internals, demo reference outputs and LICENSE unchanged. The maintainer rejected a separate documentation PR on 2026-10-08; PR #12 is closed and delivery focuses on the code refactor. The maintainer explicitly requested the push on 2026-10-08. The completed phase commits are on origin/refactor/method-v2; the validated live follow-up is checkpointed for that same authorized branch. No refactor PR has been requested or opened.

## Outputs

Workspace: `E:\kabuda\FactReview`.

- Documentation PR: https://github.com/DEFENSE-SEU/FactReview/pull/12 — requested title/body, base `main`, exactly two documentation files; closed on 2026-10-08 after the maintainer rejected a separate documentation PR.
- Former docs branch: `docs/method-v2-spec`, patch `ab983b9`, continuous automatic-review amendment `281e0f5`; amendment retained locally as `84c2aeb`. Local and remote docs branches were deleted after verifying identical document contents and equivalent commits in `refactor/method-v2`.
- Refactor branch: `refactor/method-v2`, created from docs branch. Remote branch: https://github.com/DEFENSE-SEU/FactReview/tree/refactor/method-v2 . Initial push verified at `7e7630a`; no refactor PR is open.
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

Phase 8 checkpoint full default suite: **615 passed, 3 original deselections, exit 0** in `runs/v2_final/pytest-full.log`. The two deselected legacy e2e cases passed separately (`pytest-legacy-e2e.log`); the remaining live Docker case was not run. No new skips or relaxed assertions. Statistics/pipeline focus: 18 passed. Ruff passed across new implementation modules and Phase 8 files. Formatting of three new Phase 8 files retained identical ASTs. The wheel contains both pipeline entry modules and all v2 packages; 110 packaged Python modules parsed with the Python 3.11 grammar, and required packaged files matched workspace bytes (`wheel-inspection.json`).

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
- Confirmed cutoff: explicit `--submission-deadline YYYY-MM-DD`, with **3 calendar months** before it labelled concurrent. On 2026-10-08 the maintainer authorized arXiv fallback when a venue deadline is unavailable. CompGCN live checks use first submission **2019-11-08** from https://arxiv.org/abs/1911.03082 and concurrent start **2019-08-08**. Provenance records `cutoff_source=arxiv_first_submission`; no venue deadline is asserted. The earlier offline fixture remains unchanged.
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

The Phase 8 offline replay remains in `runs/v2_compgcn/compgcn_fixture_2026-10-08_050820_50ebee36/`. It has 2 supported/1 unverified versus the historical adapter's 2 supported/1 questioned. The claim sets differ, so their label counts are not an accuracy comparison. All 27 demo hashes remain unchanged; there is no Git diff to RefCopilot, demos or LICENSE.

The replacement MinerU token worked: the real 15-page PDF produced 210 content-list rows. Credentials remain only in ignored `.env`; no supplied secret was found in tracked files. The active LLM route is local Codex OAuth, model `gpt-5.5`.

Live observations led to these corrections:

- Materials now include all 5 image/chart crops, exclude page metadata from bibliography and text cursor alignment, and preserve original Markdown bytes while restoring its image links. There are 49 bibliography blocks. Two captions were combined by MinerU; their ambiguity is explicit and cannot create caption-dependent figure flaws.
- RefCopilot receives the actual text-file contents and the configured Semantic Scholar key. Its real run processed 48 entries from 49 parser blocks. Three findings lack unique original-text locations and remain issues. The preserved log records Semantic Scholar timeout/SSL failures and a negative-sleep error inside the protected RefCopilot library. Its internals were not changed.
- Structured condition IDs and verbatim quotes are explicit in branch prompts. Invalid execution plans remain rejected; independently validated experiment observations survive a plan-only rejection after dispatcher revalidation.
- Writing candidates are checked against original PDF page pixels. OCR artifacts, discretionary style and unconfirmed candidates stay in diagnostics. Theory notation flaws also require original-page confirmation. The first real audit found the printed dir(r) and mapping arrow had been corrupted in parsed text; raw parsing is preserved.
- Positive evidence defaults to insufficient. A model must explicitly declare `fully_supported_conditions`, a distinct subset of the conditions addressed, and establish the entire covered condition including the claim's qualifiers. Contextual ConvE references cannot alone establish COMPGCN extensibility; source hooks cannot establish a public URL or all datasets. The claim prompt explicitly separates novelty from independent architecture conclusions. Semantic completeness remains a model judgment and requires audit.
- Citation-only arXiv identifiers resolve through metadata before self/date/review-page checks and full-text reading. Search completeness is never invented; the current provider does not certify exhaustive search. Novelty support by absence remains unavailable.

Full follow-up regression: **733 passed, 3 original deselections** (`runs/v2_live/pytest-semantic-final.log`, exit 0). The 2 legacy e2e cases also passed separately. Existing assertions were retained; complete-positive mocks explicitly supply the new coverage declaration. The new wheel contains 110 Python 3.11-compatible modules, with all 15 affected packaged modules matching source (`wheel-verified-inspection.json`). Independent agents cross-reviewed material, reference, visual and support-coverage changes.

Remaining limits: Docker availability was restored in the follow-up below. The CompGCN plan still lacks unambiguous metric/settings bindings and required released artifacts, so no L3 numerical reproduction is claimed; training remains 0. Literature search coverage, reference coverage and model semantic accuracy are incomplete. Model-generated statuses are system outputs, not measured accuracy or independently established ground truth.

## Docker recovery, 2026-10-08

- Docker Desktop 4.87 failed while removing old Windows socket entries: first `Docker/run/sailor-ingest.sock`, then `docker-secrets-engine/engine.sock`. Its stopped WSL engine could not serve Docker requests. Independent agent `docker_diagnosis_review` confirmed the failure from logs and file metadata. Reparse-point attributes alone do not establish corruption.
- Preserving the old `Docker/run` directory under a unique sibling name temporarily restored Engine 29.7.2, a real Python container, E: bind-mount reads/writes and a synthetic v2 analysis. An orderly Desktop restart reproduced error 1920. A second runtime relocation exposed the Secrets Engine socket failure. These failed attempts remain in `runs/v2_docker_recovery/`.
- Upgraded in place to official Docker Desktop **4.94.0.241994**, Engine **29.8.2**, after matching the installer SHA-256 to Docker's published checksum and verifying its valid Docker Inc signature. Installer exit code was 0. The offline 17,498,636,288-byte data disk backup matched its source hash; immediately after installation, both the data disk and settings were byte-identical to their backups. Backup and installer files remain local under `runs/v2_docker_recovery/`, ignored by Git. No factory reset, WSL unregister, volume deletion or Windows reboot was used.
- The new version started successfully using the existing endpoints. Its normal `docker desktop restart` completed; `engine-after-upgraded-restart.json` records a responding Linux Engine 29.8.2. Before/after inventories preserve both original container IDs, all 5 volumes and all 4 images present before upgrade. A further synthetic image was created by validation.
- Real `execute_plans` runs used the default Docker builder/runner, fresh workspaces and actual runtime observations. `v2-smoke-d9c846ec/` passed after upgrade; `v2-smoke-d5afd169/` passed again after the normal restart. Both use explicitly synthetic `[1,2,3]` input, observed sum 6, automatic approval, analysis mode, training budget 0 and deterministic refinement with no LLM calls. These infrastructure checks establish no CompGCN numerical result; they stop at execution evidence without final assessment. Independent review is saved in `independent-audit.md`.
- The first real build exposed generated `deployment/install_deps.py` scanning itself and falsely requesting `torch_scatter`. The scanner now excludes only its own resolved path. Author imports in the repository root, `deployment/`, and another same-named script remain detectable. Four offline regressions reproduced the old failure and verify the fix. The corrected real build emitted `install_deps_ok` with no torch-scatter installation or fallback.
- Strengthened the existing explicit `requires_docker` integration test to require a real server-version response within 30 seconds. It retains the original missing-CLI handling; default unit tests still mock Docker. Latest full suite: **737 passed, 3 original deselections** (`pytest-full.log`); the real Docker test separately **1 passed** (`pytest-docker.log`). The earlier two legacy e2e passes remain recorded above. Ruff and diff checks passed. The rebuilt wheel has 110 Python 3.11-compatible modules and exactly matches the corrected Docker source (`wheel-inspection.json`).
- Network failures during the installer download were handled by retaining completed ranges and validating the final complete file; no TLS verification was disabled. The unavailable `python -m build` attempt is retained in `wheel.log`; the existing `pip wheel` workflow then succeeded in `wheel-pip.log`.

Official references: [Docker release notes](https://docs.docker.com/desktop/release-notes/), [Windows installer](https://docs.docker.com/desktop/setup/install/windows-install/), and [upstream socket issue](https://github.com/docker/desktop-feedback/issues/625). Recovery is verified across one normal Desktop restart; no Windows reboot was performed.

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

Resolve the blocked CompGCN plan’s metric/settings bindings and required released artifacts before running an approved evaluation/analysis job. Docker is available and verified after a normal Desktop restart. Training remains disabled by the confirmed zero budget. The implementation and live follow-up are ready for use on the authorized remote branch; no refactor PR has been requested.

## Preserved live diagnostics

- `runs/v2_live/compgcn_live_2026-10-08_213558_5e0e658d/`: first real service run, 43 claims, known parser/contract defects; diagnostic output.
- `runs/v2_live/references_20261008_215355_a9200c8a/`: real RefCopilot response and usage; 48 processed entries. `reference-repaired-round2.log` retains service failures.
- `runs/v2_live/probe_20261008_215439_49fdd8c8/`: real representative branch responses. Code produced 6 grounded evidence items; a qualitative experimental condition generated an invalid MRR plan, motivating explicit rejection with retained observations.
- `runs/v2_live/compgcn_live_repaired_2026-10-08_220730_2373040b/`: 40 claims, 6 supported/30 questioned/4 unverified, no execution. Independent audit found remaining Writing/figure parser false alarms and three overbroad support cases. These counts are retained as diagnostic history.
- `runs/v2_live/compgcn_live_repaired_2026-10-08_222956_c40e836f/`: completed corrected run (563.39 s): **54 claims, 5 supported / 0 flawed / 33 questioned / 16 unverified**. All six non-execution stages are `ok`; execution is `skipped`, approval mode `not_run`. One training plan remains blocked. It reuses actual MinerU bytes, byte-identical bibliography-check responses and exact query/deadline/id/question retrieval responses; all model judgments run again. New retrieval keys use the service. Provenance and original source runs are retained.
- Local driver failures are preserved: the first standalone reference probe omitted the active statistics scope; one replay logger attempted to parse the extraction prompt as plain JSON. Both driver errors were corrected without changing production validation. Their logs and failed summaries remain available.
- First branch push failed at optional LFS lock verification (EOF). Retrying with per-command `lfs.locksverify=false` succeeded; LFS uploads were not bypassed and persistent Git configuration was unchanged.

## Final live audit

`runs/v2_live/compgcn_live_repaired_2026-10-08_222956_c40e836f/independent-audit.json` records the distinct reviewer pass. The reviewer checked 335 local evidence pointers with no mismatches. All 29 sufficient positive evidence items explicitly declare full condition coverage; 175 contextual/partial positive items remain insufficient. The previously over-supported novelty and public-availability claims are now unverified; extensibility is questioned. The figure-caption and known Writing OCR false alarms were suppressed. These are specific observed corrections, not an accuracy estimate.

The corrected report is `review/report/final_review.{json,md,pdf}` under that run; teaser is `review/teaser/teaser.svg`. The PDF has 152 pages; root-agent visual samples of the first and last pages show no clipping/overlap. Full-document visual perfection is not claimed. `run_stats.json` contains 92 model requests and labels two requests' usage as estimates; cached parsing, bibliography and retrieval provenance is explicit. All source model responses and unsuccessful prior runs remain available locally.

Live follow-up checkpoint: `fix(v2): validate live evidence and original PDF findings`. The maintainer authorized pushing this work to `refactor/method-v2`; the commit contains source, regression tests and these audit records, with all runtime artifacts and credentials excluded.
