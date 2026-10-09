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

The runtime matrix also checks the exact original paper target and artifact hash, plus target revalidation before approval, before launch and after the run. The same paper target must survive JSON serialization and the execution ledger. A correct output value with the wrong split still provides no deciding execution evidence.

Execution plans retain `target_bindings` with original number/cell selectors, quotations, pointers and source/claim/condition hashes. L3 reconstructs them before approval and execution, and after successful or failed runs. A changed source stops that plan and retains its logs without starting infrastructure repair. Historical plans without bindings remain readable; executing them requires a new L2 plan from the original material. An invalid plan keeps its local reason while healthy plans continue.

The current target binder supports a bounded set of complete scalar statements and native tables with explicit dimensions. It uses complete registered metric names for absolute measurements. Unknown metrics, derived improvements/ratios, list/composite conditions, unparsed qualifiers, incomplete table rows and cross-paragraph definitions remain blocked. Nonempty condition descriptions also need an unambiguous scalar interpretation. This limits executable coverage on real manuscripts. Explicit paper units require actual runtime unit metadata; unspecified bare values retain their original scale, and no percentage conversion is inferred. Generic condition alignment and existing tolerance profiles remain unchanged.

## Claim source repair

The initial extraction still requires an exact original quotation. When source validation fails and repair is enabled, the repair request offers `source-block-v1` choices for complete original blocks within the existing repair context. The model selects an opaque source ID; code restores the original block text and location, then runs the original grounding checks. The first extraction request and claim schema are unchanged.

Repair can change source bindings only. Claim text, conditions, needs, importance, order and reference coverage stay fixed. An explicit null keeps the original binding; unresolved bindings remain failures within the configured cap of three repair rounds. The response must cover every invalid candidate exactly once. Unknown versions, unknown or out-of-context IDs, changed source materials and malformed selections fail visibly. Unversioned saved quote repairs retain their strict decoder.

Repair audits record the original response, source catalog, whole-block granularity and resolved bindings separately. Blocks without a valid location, inconsistent character offsets or a lossless representation in the existing quote field are unavailable. In particular, this version does not trim outer whitespace to make a block selectable. Selecting whole blocks can broaden displayed quotations or produce duplicate references that still fail validation. A valid source ID does not establish entailment, extraction quality, sufficient evidence or a claim status.

## Real visual route

Configure the usual model or the optional `VLM_*` overrides, then run:

```sh
python scripts/check_v2_visual.py
```

This makes six real model calls: two random six-digit images and four controlled figures. The code exists only in pixels. The figure cases exercise consistent context, a body reference to a nonexistent panel, missing axis labels, and present but unreadably small printed labels. The latter uses 2 pt axis/tick/legend text rendered at 96 dpi; the caption and body do not disclose the expected defect. A figure passes only after a successful check with verified physical size, no unresolved issues and the expected finding categories. Parser content for these controlled PDFs is fixed.

To check only printed-size legibility, use `python scripts/check_v2_visual.py --mode figures --figure-cases tiny_labels`. The `--figure-cases` option selects from the named controlled figures and leaves transport probes governed by `--mode`. Saved results distinguish each run's selected cases; an earlier five-case run does not establish the newly added legibility case.

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

## Joint experimental sources

An experimental candidate can declare `additional_sources` when its table, measured quantity, setup or connecting passage occupy different original locations. The first implementation permits one explicit joint `paper_support` candidate for one original condition. It retains the primary passage and every separately proposed partial observation. `EXPERIMENT_JOINT_MAX_SOURCES` defaults to six including the primary source; exceeding the configured limit leaves the candidate unconfirmed and never truncates its source list.

The first pass must declare the complete joint member set. When a table depends on external metric or setup definitions, that set also needs its own exact manuscript passage connecting the numbered table. A reference emitted as a separate partial candidate grants no access to the joint. The second pass cannot add omitted members.

The independent second pass selects candidate-specific catalog entries and declares `source_uses` for every joint member. Single-source candidates must return `source_uses=[]`; their indices are identified explicitly in the input. Each selected passage must fit inside one declared continuous original range. Prose operands require the complete governing sentence; table operands require the complete native table and its headers. Captions, definitions and references must also be within declared ranges. Every source retains its original block/span, parent identity, pointer and artifact fingerprint. Original Claim and SharedMaterials objects remain authoritative and unchanged.

Expected endpoints and claimed differences use a separate, condition-scoped original assertion lookup. That lookup does not authorize the assertion as an observed result or as extra context for a candidate. Numerical roles, metric/setup bridges, explicit manuscript table references, units, relation, controls and required setting cases still pass their existing checks. Source count and declared roles alone cannot establish support. Mechanically checked uses and semantic-only protocol/qualifier judgments are recorded separately in the scope audit.

Artifact revalidation uses each complete declared member's original pointer, location and file hash. A consumed substring remains bounded by that member even when the substring also occurs in another prepared representation, such as Markdown text within a PDF-backed member. Changes to the original member's pointer, block, range or artifact remain invalid.

Table units come from the selected measurement headers, the value's own explicit suffix, or an authorized caption for that exact table. Header scope and span origins constrain inheritance in both orientations; implicit all-`td` headers use a bounded structural rule. A parent header grants access only to its next connected header level. Conflicting units, unknown unit expressions and compound operations remain unconfirmed, with no scale conversion. A caption must have a fully recognized quantity/unit declaration; any dataset and setting qualifiers must match the current operand's original condition. Unknown qualifiers, another setting, negation, disjunction and mixed quantities cannot donate units. Explicit axis/token units remain independently usable. A complete adjacent caption can be projected from one declared member while preserving existing table/cell IDs; separate fragments cannot be joined to authorize it. Unsupported layout or wording stays visible as a limitation.

Only uniquely identified new joint pairs have local source/selector failures. A duplicate or malformed pair cannot later be restored by another row. Invalid condition rows and existing single-source decoding failures retain their condition-wide behavior. A valid primary source may remain an insufficient observation when an additional member is invalid; an invalid primary cannot create an evidence pointer.

An accepted source group produces one Evidence with `additional_pointers`, preserving one sufficiency decision and the original condition coverage. Old JSON remains readable with an empty default list. Complete original quotations and positions survive dispatch, assessment and FinalReview JSON/Markdown/PDF. The teaser retains its compact claim/status/source-type summary and report link. The base joint-source route uses the existing two model passes and does not start execution or training.

## Optional experimental binding repair

`EXPERIMENT_SCOPE_BINDING_REPAIR_ROUNDS` defaults to `0`. Setting it to `1` permits one additional model request per claim, combining eligible joint candidates into a single bounded request. The normal pipeline reads this setting through its existing configuration. It is separate from L3's execution repair budget and does not start retrieval, Docker or training.

This path handles two validation errors: a missing operand-specific setting bridge, or a declared mechanical source role with no corresponding consumer. A candidate must already have complete support judgments from both original passes, valid condition identity and an intact declared source set. Partial candidates, missing joint candidates, other validation errors and permanently invalidated condition/pair identifiers remain ineligible.

The model must explicitly append a bridge for the diagnosed setting or remove a diagnosed unused mechanical role. Existing source members, conditions, coverage, semantic judgments, numeric selectors, relations, existing bridges, grounds and their order stay fixed. Mixed semantic/mechanical role records retain their original rationale and are excluded from role repair. Every accepted patch passes the original source, identity, numerical, unit and setting checks again on a fresh catalog snapshot. A later failure cannot trigger a second repair. Ineffective patches and service failures remain visible; healthy original evidence survives local rejection. Input changes invalidate the repair, and adopted evidence links to the new effective response while retaining the original response and errors.

The current positive repair checks use mocked synthetic candidates. Fixed original-paper replays remain unsuccessful: adding the missing setup bridge exposes an unbound comparator in one case; the other has a separate metric/reference error and receives no repair request. The original positive expectations and responses are unchanged. The option remains disabled by default, and these checks establish neither real-model repair success nor scientific accuracy.

## Selected Literature citations

`scripts/check_v2_literature.py` runs declared citation cases with explicit service boundaries:

```sh
python scripts/check_v2_literature.py --plan path/to/probe-plan.json --case selected-case --mode live
```

A version 1 plan contains `protected_files` (input path to SHA256) and named `cases`. Each case declares its original saved run and unchanged `raw_claim`, or separately labelled synthetic claim/materials, its submission deadline, allowed modes and independent expectations. Expectations partition every original condition into supported and unsupported sets, and identify the required cited work or exact unresolved bibliography entry. Original cases require protected `materials/materials.json` and `screening/screening.json`. The driver checks the saved claim against the declared original before making requests.

`live` uses the configured retrieval, reading and model services. `original-cache` requires a protected original search audit and exact request matches; it never falls back to network calls. Reconstructed cached model envelopes are labelled, and original comparison flags remain unchanged. Cache modification times do not establish original acquisition times.

Positive and negative citation cases both require healthy source identity, page-located text actually supplied to the model, and a valid comparison against that passage. An unavailable service or rejected comparison cannot satisfy a negative expectation. A separately declared unresolved-cache case checks the precise missing identity and explanation; unrelated historical reader failures remain visible and do not establish reader health.

Literature quotations must be exact substrings of the matching source's `passages[].text`, preserving original characters, hyphenation and whitespace. The model-visible `paper` metadata omits `abstract`; original metadata remains available to identity, self-exclusion and audit processing. Abstract text remains eligible when explicitly included in the source's passages by the existing fallback, with its existing sufficiency restrictions. Text normalization or another reader item cannot silently grant a quotation evidence status.

Each output saves requests, responses, actual boundary counts, provider/model, recorded usage, result/assessment and input/implementation hashes. Adapter calls and HTTP requests are distinct counts. Current readers do not expose original PDF bytes for an independent PDF hash; the driver records that limitation and does not download the PDF again. These selected checks measure citation contracts and service integration; they provide no broad scientific accuracy estimate.

## Evidence report navigation

The report renders supported HTML table structures as readable tables, retaining cell associations, explicit multirow headers and units on continued PDF pages. Complete row excerpts without headers use neutral column positions and identify the missing headings. Malformed or unsupported markup falls back to the escaped original passage. Original quotes and evidence judgments remain in JSON.

Repeated evidence passages share a source only when `(locator, page, line, key, quote)` matches exactly. Each evidence occurrence keeps its own direction, condition coverage, sufficiency, pointer, note and execution provenance. The first occurrence displays the full source and links to every use; later occurrences link back. Markdown and PDF contain actual internal destinations, including across page and paragraph splits. Source text cannot create its own anchors or links. Report notes, questions and execution records remain complete; this presentation change does not reassess claims.

Joint evidence has ordered primary and additional source usages under the same evidence judgment. Each additional location has its own destination and backlink. A source shared between an additional pointer and another evidence item's primary pointer is rendered once with links to both uses. Empty additional-pointer lists retain the existing single-source Markdown presentation; canonical JSON serialization adds the empty default field.

## Limits retained in artifacts

- Upfront extraction checks exact original quotes, locations and structured contracts. Claim recall, semantic entailment, preservation of all qualifiers and selection of every necessary `needs` branch remain model judgments. Existing splitting and dispatch unit tests use injected answers; they do not measure those judgments. Historical full-paper location audits and selected excerpt checks have no independent whole-paper recall denominator. An empty needs subset is structurally permitted and must not be interpreted as proof that the claim needs no verification.
- A fixed nine-excerpt evaluation uses two prior regression excerpts, six additional excerpts from two local papers, and one acknowledgement control. Source text and independent agent reference labels were frozen before the calls. Eight cases returned 35 claims; one exhausted all three exact-source repair attempts. Semantic review also identified an omitted dataset/table linkage and a ranking claim missing Literature. A merged footnote/parameter number is an input-ambiguity case: these excerpts use exact local PDF text extraction and bypass MinerU. The original failure and labels remain saved under `runs/v2_binding_followup/claim-extraction-quality-set-v1/`. Conclusion coverage counts do not establish qualifier accuracy, human expert agreement or whole-paper recall.
- The initial extraction prompt explicitly covers shared conclusions across tasks, each claim's own unambiguous table/caption scope, and Literature needs for external records or ranking claims. A later fixed evaluation combines four unchanged original excerpts with six disclosed synthetic paired controls. The synthetic controls preserve their nine required units; the original excerpts retain an omitted independent parameter-reuse assertion and a qualifier present only in its source quote. The full semantic evaluation therefore remains failed. Exact source text does not substitute for a missing extracted conclusion, and one output per version cannot establish why an omission occurred. The new run and unchanged reference-label bundle are saved under `runs/v2_binding_followup/claim-semantic-prompt-evaluation-v1/`; the original run stays unchanged under `runs/v2_binding_followup/claim-extraction-quality-set-v1/live-runs/`.
- A subsequent paragraph asks the same initial extraction request to check omitted independent assertions and each claim's governing qualifiers. It introduces no separate model stage or completeness guarantee. A single frozen 13-case evaluation retains the preceding ten cases and adds three independently agent-labelled synthetic controls. The four original regressions preserve all 14 required units; two synthetic cases have provider failures, and another retains incomplete claim-local qualifier provenance. Overall, 24 reference units are complete, one has a provenance limitation and five are unavailable. The batch remains failed. Only 12 of 14 requests return usage, so its 33,358 measured tokens omit the two failed requests' unknown cost. Inputs, labels and failed responses remain under `runs/v2_binding_followup/claim-completeness-evaluation-v1/`; independent semantic review is in `claim-completeness-implementation-review/evaluation-extract-219bf7d0f8f8/` under the same follow-up root. There was no resampling to replace failures. These selected results establish neither whole-paper recall nor a causal effect of the paragraph.
- A scope-only fairness probe with a frozen first-pass candidate tests the second-stage applicability judgment. Its three predeclared synthetic controls distinguish an unrecorded training duration, irrelevant missing power telemetry and an explicit epoch mismatch. The three actual scope responses match those expectations. The original complete numerical preflight failed on unsupported prose grammar and remains saved. After the bounded nominal-pair grammar extension, a separate fixed-response replay binds the three original result pairs and removes that grammar error. Its six original hand-authored responses, partial judgments and concern flags remain unchanged; full claim statuses stay Questioned / Unverified / Questioned. Only structural pair choices change in the scope payload. This replay makes no actual model calls, and the earlier scoped result establishes neither full claim support nor candidate-generation quality or general scientific fairness accuracy.
- The default experimental scope schema (`catalog-v2`) selects immutable table-cell or original prose-number IDs. Numeric tokens, source positions and units are restored locally; the model does not supply numeric offsets or concatenate operand contexts. Shared sentence text appears once in the model catalog, with a separate choice for every numeric occurrence. Legacy `catalog-v1` and unversioned saved responses retain their original strict decoding; unknown versions are rejected.
- Prose comparisons require an explicitly scoped subject/predicate pair. Ordinary pairs accept complete forms such as `On the test split of dataset D, R has 80% accuracy and S has 84% accuracy` and `On the D test set, R accuracy is 80%, and S accuracy is 84%`; either explicit prefix can accompany either complete predicate form. Dataset, split, actor and complete metric names match literally, and selected numbers retain their exact original occurrence offsets. Mixed predicate forms, uncertain or changing scope, negation and approximate values remain outside this extension. Named transitions keep their existing grammar and additionally require the original method, from/to settings and matching source assertion. This bounded language support leaves unfamiliar phrasing unconfirmed. Claimed endpoints come from the original claim/source; observed table values cannot supply their own expected targets. Cross-paragraph metric/setup definitions must link to the selected table. The numbered-table link recognizer accepts explicit current-manuscript references and rejects explicit foreign-work ownership; unfamiliar reference wording stays unconfirmed.
- Model-facing prose pair choices provide existing occurrence IDs, condition/case identity and canonical setting roles after structural validation. They carry no support judgment: the model must independently assess the entire original candidate and condition, and the numerical gates still validate its response. Identical original source ranges can carry several explicit condition scopes; different ranges retain separate coverage boundaries.
- Complete numerical support must cover every required setting and establish the stated relation. A statistical concern may concern a subset of settings, but each cited comparison still requires valid source, role, metric, split and unit bindings. This contract does not compute an unreported significance test or introduce new numerical tolerances.
- Code candidates receive a separate applicability review after their source lines and paper quotations are validated. Complete support requires all implementation qualifiers. Empirical outcomes, novelty and public-artifact availability require their own evidence; a matching configuration line cannot establish them. The scope response, source selection and unresolved conditions remain in `code_scope_reviews/` and the report. This second model judgment reduces a demonstrated first-pass failure but does not provide a formal guarantee of semantic correctness.
- Theory checks bind numbered theorem/proposition identities to the target assertion and reject proofs that merely cite another statement. Unnumbered derivations retain exact source and step validation. These checks establish source identity; they do not constitute a formal proof checker.
- Default retrieval does not certify exhaustive search. Unknown technical domains, missing identity/date metadata, unavailable source text and failed services remain explicit limitations.
- Citation matching uses original bibliography labels locally, including exact alphanumeric labels and grouped citations. Ambiguous labels are unresolved; ordinary bracketed words do not become author queries. Reader identity conflicts reject that paper's returned content, and reader metadata cannot replace a previously resolved evidence locator. A fallback abstract after failed identity binding cannot establish sufficient citation support. Other healthy papers continue through verification.
- Code source inspection has a configurable serialized-source budget. Full selection manifests live in `code_scopes/`; bounded samples and counts enter model requests and reports. Paper text has no corresponding global budget, so unusually large manuscripts can still exceed a provider's context window.
- VLM failures do not stop other figures. Coverage distinguishes checked, failed and unavailable input; checked figures may retain uncertain observations. Missing physical dimensions prevent a confirmed printed-size legibility finding.
- Failed calls and unknown usage are counted. Text-only estimates cannot establish image token cost.
- Independent semantic accuracy, comprehensive benchmark reproduction and human-review benefits remain separate research evaluations. The matrix establishes the listed engineering contracts only.
