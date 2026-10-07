# FactReview v2 progress

## Task

Open the documentation-only PR, then implement the authoritative v2 method specification through Phases 0–8 on `refactor/method-v2`. Use an independent agent to review each phase, address findings, verify and commit, then continue automatically. The maintainer removed the phase-pause requirement on 2026-10-08 and requested continuous work until completion or an explicit stop. Keep all 14 acceptance checks observable. Never weaken or add skips to tests. Unit tests must mock LLM, retrieval, and Docker. Do not modify RefCopilot internals, demo reference outputs, or LICENSE. Keep refactor commits local until all checks pass or the user explicitly requests a push; the adopted prompt also requires approval before publishing the refactor.

## Outputs

- Documentation PR: https://github.com/DEFENSE-SEU/FactReview/pull/12
- Documentation branch: `docs/method-v2-spec`; initial patch commit `ab983b9`, automatic-review amendment `281e0f5` (also applied to the refactor branch as `84c2aeb`).
- Current branch: `refactor/method-v2`, created directly from the documentation branch.
- `docs/method_v2_spec.md`: authoritative specification, unchanged from the patch.
- `docs/refactor_v2_prompt.md`: adopted procedure, amended to use automatic phase reviews and continuous execution per the maintainer's request.
- `checks.md`: all 14 acceptance requirements.
- `progress.md`: this handoff.
- `runs/v2_baseline/`: local baseline logs (ignored by Git).
- `runs/v2_baseline/dependency-install.log`, `dependencies.txt`: installation output and exact installed versions.
- `runs/v2_baseline/pytest-baseline.log`, `pytest-baseline.exitcode`: first baseline (exit 1).
- `runs/v2_baseline/pytest-short-temp.log`, `pytest-short-temp.exitcode`: passing baseline retry (exit 0).
- `runs/v2_baseline/baseline_isolation.py`: explicit local pytest plugin for indirect Docker and loopback probes.
- `runs/v2_baseline/preflight.json`, `compgcn-baseline.txt`: credential readiness and live-demo skip reason.
- `runs/v2_baseline/baseline-manifest.json`: source commit identities and checksums of 7 unchanged demo reference artifacts.
- Phase 1: `src/schemas/claim.py`, `review.py`, `__init__.py`, `legacy_claim.py`, `legacy_review.py`, `v1_adapter.py`; `tests/test_schemas_v2.py` and import-only legacy adjustments in `tests/test_schemas.py`.
- Phase 1 evidence: `runs/v2_baseline/phase1-schema.log` (62 passed), `phase1-schema.exitcode` (0), `phase1-import-tail.log` (2 mocked legacy integration tests passed).
- Phase 2: `src/preprocessing/materials.py`, `src/schemas/materials.py`, `tests/test_materials_v2.py`; `runs/v2_materials/pytest.log` (25 passed).

## Completed

- Fetched origin and fast-forwarded main from `87dad71` to `5dc71d9` before creating the documentation branch.
- Applied the attached patch with `git am`; no fallback copying needed.
- Pushed only the documentation branch and opened PR #12 with the requested title and attached `PR_DESCRIPTION.md` as the body. The PR changes exactly two documentation files.
- Phase 0 complete: installed existing extras into `.venv` using Python 3.12.10. No application code, test assertions, or project configuration changed.
- Initial baseline: 252 passed, 4 failed, 3 deselected in 29.41s (exit 1). Failures: `test_no_docker_host_venv_install_follows_nested_verify_missing_module`, `test_no_docker_host_venv_install_repairs_dgl_graphbolt_torch_pin`, `test_no_docker_host_venv_installs_pyg_native_package_from_torch_wheel_index`, `test_install_run_venv_jupyter_kernel_uses_run_local_prefix`.
- Diagnosis: Windows default pytest temporary paths exceed 100 characters; `src/fact_generation/execution/nodes/fix.py::_host_venv_dir` then selects `C:\\frv-venvs/<hash>`, while these four tests expect run-local `.venv`. Preserved the initial failure output.
- One environment correction: selected a new shorter pytest `--basetemp` under `runs/v2_baseline/t1`. Reran the same 256 selected tests with unchanged assertions: **256 passed, 3 deselected in 21.51s**, exit 0. The 3 default deselections are 2 `e2e` tests and 1 `requires_docker` test.
- CompGCN live no-execution baseline: skipped under the prompt's explicit credential exception. `MINERU_API_TOKEN` and repository `.env` are absent; strict parsing requires the token (`src/preprocessing/parse/mineru_adapter.py:49–57`). No live pipeline request or parser fallback used. Codex cached credentials are present; validity/reachability remains untested. No external source reachability claim is made.
- Independently reviewed the local harness, selected external-call paths, failure cause, checkpoint files, and all 7 unchanged artifact hashes with reviewer agent `baseline_review`.
- Phase 1 complete: strict v2 claim/evidence/plan/finding/question schemas and v2 review container; historical schemas isolated in explicit legacy modules. Schema checks: 62 passed. Legacy stage-import/report-tail integration: 2 passed with mocked services. Existing schema assertions retained; only their legacy imports changed.
- Phase 1 automatic review (`phase_reviewer`) found ambiguous execution target keys/non-finite targets and two legacy Markdown parsing defects. Corrections enforce exact condition-id targets with finite values, preserve escaped pipes, flag malformed rows, and normalize emphasized status labels. Regression tests cover every finding. Ruff passes on Phase 1 files.
- Real CompGCN v1 `report.md` adapter smoke: 3 historical display records (`questioned`, `supported`, `supported`), zero issues. Historical labels remain explicitly unassessed; no new evidence or missing coordinates fabricated.
- Phase 1 committed as `f2888b0`.
- Phase 2 complete: direct MinerU adapter entry, original Markdown/content list, located blocks, bibliography, 200 dpi pages/crops, 96 dpi printed crops and a read-only repository index. 25 fixture tests passed; Ruff check/format passed.
- Phase 2 automatic review (`phase_reviewer`) found missing appendix/Oxford-list figure anchors and dropped MinerU list items. Both fixed with regression coverage; unrecognized parser content leaves explicit issues. Python 3.11 compatibility for junction checks verified. Printed crop dimensions use PDF physical dimensions; the initial clip-rounding failure was fixed without changing assertions.
- Phase 2 committed as `f15dbd7`.
- Phase 3 complete: upfront claim extraction, writing/table/figure/reference checks, durable screening results, actual image transport for all three providers. Removed report-agent extraction/splitting/merging duties and the C1–C3 limit.
- Phase 3 independent review (`phase_reviewer`) identified reference checks silently succeeding with zero processed entries and model failure payloads carrying empty findings. Fixed both with regression tests. `runs/v2_screening/pytest.log`: 45 passed; original retrieval-policy assertions preserved. Ruff passes on the new screening/image files.
- Refactor branch remains local. Phases 0–5 are complete; independently prepared Phase 6–8 files await their separate commits and final integration checks.

### Baseline command

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = "$PWD\src;$PWD\runs\v2_baseline"
.\.venv\Scripts\python.exe -m pytest -p baseline_isolation --basetemp runs/v2_baseline/t1
```

The first run used the same command without `--basetemp`. For a repeat, choose a **new** short directory under `runs/v2_baseline/` and save a new log; pytest clears an existing explicit base-temp directory. Verify the resolved path stays within that baseline directory. Inspect previous results before repeating operations that could write outside `runs/`. The local ignored logs/plugin survive in this workspace; this tracked note preserves the command, outcomes, limitations, and next action for other checkouts.

## Decisions

- Use per-command Git committer identity `ChaoqianO <224349230+ChaoqianO@users.noreply.github.com>`, derived from the authenticated GitHub account. Git had no configured identity; global Git configuration was left unchanged. Patch author/message retained by `git am`.
- Use `.venv` for repository dependencies; preserve global Python packages. Install the existing runtime/dev/refcheck/positioning extras needed to collect and exercise the current suite.
- Preserve the existing default pytest marker selection; record deselections explicitly. Phase 0 left tests unchanged. Phase 1 moved old schema imports to the explicit compatibility modules while preserving every assertion.
- Apply the user's required unit-test mocks via the explicit local baseline plugin. A separate review found existing Docker argument-construction tests indirectly call `docker info` through `_docker_info_field`; the stale-loopback test calls `socket.create_connection`. The plugin mocks these two probes, preserving test-specific overrides. Existing selected LLM/retrieval paths were reviewed for their own mocks. The plugin is scoped baseline tooling and provides no universal network-access guarantee. Incorporate durable unit-test isolation in the affected implementation phase.
- Use shorter test temporary paths for this Windows baseline. No production path handling or assertion was changed. The long-default-path failures remain documented as test portability work.
- The directly requested root files `checks.md` and `progress.md`, plus explicitly requested baseline outputs in `runs/`, are authorized outputs alongside the prompt's source-path list.
- Phase review uses a separate reviewer agent with inherited configured model/effort and no override. Exact runtime model/effort identifiers are unavailable to this task; no claim is made that prompt wording changes them.
- The maintainer removed phase pauses on 2026-10-08. Documentation PR #12 was updated with automatic reviews; implementation continues immediately after each phase's review, checks and commit. Prepare independent components in parallel, keeping phase commits and integration validation in dependency order.
- Canonical v2 schemas forbid unknown fields; evidence coverage and execution targets use stable condition IDs. Each target value is keyed by condition ID, so two datasets using the same metric remain distinct. New records require an actual page, section or character span; missing legacy coordinates stay missing in a separate historical envelope.
- The v1 adapter migrates display labels conservatively and retains original records. Reading a historical artifact performs no v2 reassessment; the adapter never creates sufficient evidence from old prose. Unknown labels and malformed rows remain visible as issues.
- Shared materials use the original MinerU adapter directly because the legacy parse stage also runs retrieval and report generation internally. Its old runtime remains available for compatibility; v2 integration will use the new material entry point.
- Bounding-box units are explicit (`normalized_1000` by default, or `pdf_points`), with 1-based PDF page locations. Missing/invalid boxes and unmatched text remain visible issues. Figure checks receive the physically downscaled 96 dpi image. Repository indexing reads and hashes files without executing or modifying them; L3 creates a separate execution workspace.
- **Maintainer-confirmed configuration (2026-10-08):** configurable repair limit defaults to 3, capped at 3 per spec; rule-based automatic approval by default with optional interactive approval and actual mode recorded; training budget defaults to **0 runs**, configurable, with training eligible only for high-priority plans within budget; explicit `--submission-deadline YYYY-MM-DD` with a 3-month concurrent window and no default arXiv-derived cutoff; existing per-metric tolerances centralized in one table; one-way v1 artifact adapter maps `in_conflict` to `questioned` unless re-assessed. The user confirmed the bundled choices in Chinese. These choices are authorized for the corresponding implementation phases.

## Open issues

- All five configuration choices and a default training budget of 0 runs are confirmed. Continue user-facing communication in Chinese.
- The actual historical experimental approval mode and training budget have not been supplied. Do not describe proposed defaults as historical experiment settings.
- Live baseline unavailable until MinerU is configured locally. Never store credentials in tracked notes or ask for secret values in chat.
- The durable Phase 4 pytest fixture now supplies fresh short Windows paths and isolates indirect Docker/socket probes; full-suite integration revalidation is in progress.
- Phase 6–8 files are prepared and reviewed separately; their phase checkpoints and final integration validation remain due.
- Claim splitting tests validate the prompt contract, mocked model outputs and grounding. Live model semantic extraction accuracy is not established by these tests.

## Next action

Commit the reviewed Phase 5 execution checkpoint, then commit and validate assessment, report and final integration in order.

## Phase 4 checkpoint

- Phase 3 committed as `f055600`.
- Phase 4 implements four peer branches, exact multi-label dispatch, global uncited-work findings, and explicit day-level submission deadlines with a three-calendar-month concurrent window.
- Independent review: `phase_reviewer`. Fixed invalid reader output upgrading abstracts to decisive evidence, failed reader containers, uncertain paper identity before reading, concurrent citation handling, malformed retrieval payloads, duplicate manuscript-quote locations, missing source artifacts, code indentation, ambiguous numeric target assignment, and numeric setting disambiguation.
- `runs/v2_verification/pytest.log`: **202 passed** (schemas, all 16 dispatch combinations, literature, three other branches, screening regressions and legacy positioning). Ruff checks/format pass on the phase files.
- Preserved initial command-path error as `pytest-command-error.log`; corrected `tests/stages/test_positioning.py`. Preserved initial unit isolation error as `pytest-isolation-error.log` (137 passed, 65 setup errors): blocking every socket connect prevented Windows asyncio from constructing its internal wakeup pair. The fixture now allows only the scoped internal socketpair connection; external connections remain blocked. Same assertions pass after this harness correction.
- Durable `tests/conftest.py` isolation replaces the ignored baseline probe plugin for ordinary unit tests. Live-marked tests retain their explicit external boundary. Windows test temp roots now use fresh short paths inside `runs/pytest`; this retains the original production path-switch behavior and all original test assertions. Full original-suite revalidation remains due after integration.
- Theory, Code and paper-only Experiments treat unresolved differences as explainable concerns (`overturnable=True`). No theorem prover or symbolic counterexample evaluator was added. Literature same-mechanism/same-setting evidence and verified author artifacts can still supply decisive flaws; the deterministic assessment layer handles its supplied evidence flags.
- The default search adapter does not certify complete retrieval. Such runs retain the actual search scope and remain unverified for support-by-absence; actual read passages still support direct comparison. The restricted technical-query vocabulary deliberately leaves unknown domains unresolved instead of sending author-name queries.
- Execution targets must bind a single paper value to the named dataset, metric and declared settings. Ambiguous whole-table values remain blocked, with the candidate and reason preserved for correction.
- Phase 4 committed as `6fe09d2`. Phase 6–8 preparation is uncommitted. All phase commits stay local.

## Phase 5 checkpoint

- Implemented `src/fact_generation/execution/v2.py`, `v2_config.py`, `v2_outputs.py`, generic alignment, shared legacy tolerance lookup, Docker helpers and execution provenance schema fields. `tests/test_execution_v2.py` contains 61 cases.
- Independent review (`integration_map`) passed. `runs/v2_execution/pytest-review-round2.log`: **218 passed, 1 original live-Docker test deselected**. External service calls are mocked. Review corrections reject expected-value selectors, failed containers/canonical rows, conflicting mappings and output-controlled author-artifact selection.
- Approval precedes workspace creation. Every ledger records the mode and budget. Ordering is priority, readiness, then evaluation/analysis before training. Training defaults to 0; positive budgets count retries. Default/cap is 3 accepted repairs, allowing the initial run plus 3 repaired runs. Source hashes and declarative infrastructure repairs protect model/loss/data/evaluation/baselines; accepted diffs are saved.
- Only exact observed dataset, metric and settings permit execution judging. Aligned matches support; aligned mismatches become author-resolvable concerns. Failed, blocked or unaligned runs retain reasons with `affects_claim=False`; they do not supply deciding evidence.
- Decisive author-artifact evidence requires a pre-run contract binding an indexed released data/log path, SHA-256, metadata and recomputation selector, with actual recomputation agreeing. Operator approval of the released artifact's role is an explicit trust boundary. Runtime output cannot choose the artifact or overwrite its value.
- Raw JSON metric formats use selectors into actual output. Both legacy tolerance profiles remain in one table. Optional paper variance must match an exact paper quote and number; the operator confirms statistic type, units and condition binding. Automated branches leave this override empty.

Next action: commit the reviewed assessment and report phases, finish integrated regression and the CompGCN comparison, then complete Phase 8.
