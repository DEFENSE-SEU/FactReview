# Reproducing the v2 framework checks

The acceptance gate remains the 14 requirements in `checks.md`. The commands below add complementary integration coverage through the current v2 pipeline and save explicit service boundaries. They use synthetic inputs and do not estimate review accuracy, false accusations, false support, or performance on published benchmarks.

## Offline integration matrix

Install the development/runtime dependencies described in the README, then run:

```sh
python scripts/check_v2_framework.py
python -m pytest tests/test_framework_matrix_v2.py
```

The CLI writes a fresh directory under `runs/v2_framework_matrix/`, containing `manifest.json`, source PDFs, parser fixtures, recorded model/retrieval calls, per-case manifests and the pipeline's normal artifacts. It exits unsuccessfully if any scenario fails. The pytest path mocks all external services.

| Scenario | Contract exercised | Expected result |
|---|---|---|
| `theory_appendix` | Main-text theorem, appendix lookup, precise proof pointer; second statement without proof | Grounded theorem support; unproved statement remains unverified with a question |
| `figures_partial` | Several figures, one failed VLM call, combined ambiguous captions, body links, partial condition coverage | Independent valid findings survive; failure and ambiguity are visible; partially supported claim remains unverified |
| `missing_repository` | Code need and experimental plan without released source | No Code model call; explicit author questions and retained blocked plan |
| `mapped_runtime` | Nested entry script, native runtime JSON, explicit output selectors, alignment, assessment and reporting | Aligned analysis supports its condition; training plan remains blocked by budget 0 |
| `mapped_runtime_misaligned` | Identical numerical value with a different observed data split | No deciding execution evidence; claim remains unverified with the alignment reason |

The default matrix uses production materials, screening, verification, execution orchestration, output decoding, assessment, report and teaser code. MinerU, LLM/VLM, reference checking and retrieval use fixed fixtures. Docker image building and command transport are mocked; the runtime fixtures still pass native JSON through the production decoder. A fixture's proof or visual judgment is predefined test input and is not a measured model capability.

## Real Docker route

```sh
python scripts/check_v2_framework.py --scenario mapped_runtime --scenario mapped_runtime_misaligned --docker
```

This enables the production Docker builder and runner for the selected runtime scenarios. It uses a small synthetic CPU calculation, automatic approval, training budget 0, no repair attempts, a 45-second run timeout and a 180-second image-build timeout. The model/parser/retrieval fixtures remain fixed. The resulting execution ledger retains commands, output mappings, raw output, alignment decisions and evidence. Both cases continue through final assessment, PDF/JSON/Markdown report and SVG teaser. The older `validate_execution_real_repos.py` script exercises the legacy v1 execution entry and does not validate this route.

## Real visual route

Configure the usual model or the optional `VLM_*` overrides, then run:

```sh
python scripts/check_v2_visual.py
```

This makes five real model calls: two random six-digit images and three controlled figures. The code exists only in pixels. The figure cases exercise consistent context, a body reference to a nonexistent panel, and missing axis labels. A figure passes only after a successful check with verified physical size, no unresolved issues and the expected finding categories. Parser content for these controlled PDFs is fixed.

The output under `runs/v2_visual/` records each input image, its hash and dimensions, actual provider/model, prompt/context, response/failure, duration and usage. Provider credentials are removed from diagnostics. This verifies limited visual transport and checking behavior; broader scientific figure accuracy needs independent labels and an evaluation protocol.

## Selected experimental claims with real model calls

For a saved v2 run containing `materials/materials.json` and `screening/screening.json`, run:

```sh
python scripts/check_v2_experiments.py path/to/saved/run --claims claim_039 claim_042
```

This reuses the saved extraction and original parsed materials, then calls the configured model for experimental observations and independent scope review. It saves requests/responses, actual provider/model, usage, exact source and implementation hashes, retained observations after rejected plans, and the deterministic assessment. Source artifacts remain unchanged. It does not run retrieval, Docker or training.

An optional `--expectations expected.json` supplies an independent coverage oracle:

```json
{
  "claim_039": {"supported_conditions": ["c1"]},
  "claim_042": {"supported_conditions": ["c1"], "unsupported_conditions": ["c2"]}
}
```

These IDs illustrate the format; choose expected outcomes by inspecting the actual original sources. A completed call with no support fails a required positive expectation. Without expectations, successful exit confirms that the selected verification calls and artifact checks completed; it makes no accuracy claim. Each output under `runs/v2_experiments/` records this distinction, and source or implementation changes during the probe invalidate its result. `tests/test_experiment_probe_driver.py` mocks model, network and process boundaries.

## Limits retained in artifacts

- Experimental numerical scope selects immutable source/table-cell IDs. Prose comparisons select an exact original passage, token and operand context. Claimed from/to endpoints are derived from the claim and condition; table values cannot supply their own expected targets. Cross-paragraph metric/setup definitions must link to the selected table. The numbered-table link recognizer accepts explicit current-manuscript result references and rejects explicit foreign-work ownership; unfamiliar reference wording stays unconfirmed.
- Code candidates receive a separate applicability review after their source lines and paper quotations are validated. Complete support requires all implementation qualifiers. Empirical outcomes, novelty and public-artifact availability require their own evidence; a matching configuration line cannot establish them. The scope response, source selection and unresolved conditions remain in `code_scope_reviews/` and the report. This second model judgment reduces a demonstrated first-pass failure but does not provide a formal guarantee of semantic correctness.
- Theory checks bind numbered theorem/proposition identities to the target assertion and reject proofs that merely cite another statement. Unnumbered derivations retain exact source and step validation. These checks establish source identity; they do not constitute a formal proof checker.
- Default retrieval does not certify exhaustive search. Unknown technical domains, missing identity/date metadata, unavailable source text and failed services remain explicit limitations.
- Code source inspection has a configurable serialized-source budget. Full selection manifests live in `code_scopes/`; bounded samples and counts enter model requests and reports. Paper text has no corresponding global budget, so unusually large manuscripts can still exceed a provider's context window.
- VLM failures do not stop other figures. Coverage distinguishes checked, failed and unavailable input; checked figures may retain uncertain observations. Missing physical dimensions prevent a confirmed printed-size legibility finding.
- Failed calls and unknown usage are counted. Text-only estimates cannot establish image token cost.
- Independent semantic accuracy, comprehensive benchmark reproduction and human-review benefits remain separate research evaluations. The matrix establishes the listed engineering contracts only.
