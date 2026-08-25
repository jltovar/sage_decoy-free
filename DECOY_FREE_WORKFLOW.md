# Native Decoy-Free validation workflow

The native workflow is designed first for ordinary public validation datasets. It does not
require biological sample-role metadata. Optional LCM roles (filopodia/TNT ROI, whole-cell,
empty-dish, preparation blank, and instrument blank) remain a later experimental layer.

Run a workflow with:

```bash
sage workflow workflow.json
```

Use `--plan-only` to validate the manifest and materialize resolved plans without searching.

## Required inputs

The minimum inputs are a target FASTA, spectra, a normal Sage search JSON, candidate foreign
species FASTAs, and a workflow JSON. The workflow will:

1. hash and freeze declared baseline material;
2. digest target and foreign FASTAs with Sage's own digestion rules;
3. exclude foreign proteins sharing target peptides;
4. select a foreign proteome deterministically by peptide/peptidoform ratio proximity;
5. construct a seeded protein-level 1:1 entrapment FASTA and write mappings/exclusions/hashes;
6. search the entrapment database once per strict search fingerprint while retaining all requested
   ranks in an immutable candidate pool shared by compatible models;
7. search requested null-window bounds adaptively or exhaustively (or replay an explicit frozen
   list) in memory from those retained candidates;
8. lock the highest-yield window satisfying PSM, peptide, and protein entrapment-FDP limits;
9. measure MS2Rescore and retain it only when its configured gain/calibration gates pass;
10. run the target-only search under an explicit calibration policy, defaulting to a locked
    dataset-local window with nuisance parameters refit in the target-only candidate space;
11. assemble the JSON-requested Ensemble roster from technically valid, independently selected
    expert windows and stage-matched frozen artifacts;
12. emit raw-q and Level-4/reportable counts, FDP, direct optimized/MS2Rescore/target-only
    comparisons, missing-run reports, transfer-stability results, parity checks, constrained TDC
    comparisons, nonblocking Ensemble validation diagnostics, technical roster decisions, and a
    separate statistical release/default report.

The primary invariant is **dataset-local optimization**. Every dataset and every individual model
fit creates or selects its own entrapment FASTA, measures its own ratios, and selects its own null
window. A window or fitted artifact selected on ISB18 is never used as PXD001468's normal search
configuration. The same rule applies to any future dataset pair.

## FASTA generation and optimizer isolation

FASTA-generation parity and null-window optimizer parity are deliberately separate experiments.
To test the optimizer against a frozen legacy search, set `database_mode` to `frozen_legacy` and
provide `frozen_legacy_fasta`. Sage inspects that exact combined target-plus-entrapment FASTA and
measures its ratios, but does not regenerate it. A frozen optimizer-input audit cannot also declare
a FASTA-generation parity reference.

To test native FASTA generation without loading spectra, run:

```bash
sage audit-entrapment entrapment.audit.json
```

The command supports three `foreign_source_mode` values:

- `automatic`: evaluate every candidate foreign FASTA after seeded protein selection and choose
  the source whose Sage-measured peptide and peptidoform ratios are jointly closest to the requested
  protein ratio;
- `explicit`: evaluate and use only `selected_foreign_fasta`; and
- `automatic_with_override`: report the automatic recommendation but generate the database from
  the user-declared `selected_foreign_fasta`.

`shared_peptide_exclusion_mode` defaults to `sage_search_space`, which excludes shared peptides in
the actual Sage searchable mass/modification space. `fdrbench004_compatibility` exists for frozen
engineering parity: it reproduces FDRBench 0.0.4's length-only shared-peptide filter and its Java
HashMap/`Collections.shuffle` seed behavior. It is not the production default. In both modes, the
final protein, peptide, and peptidoform ratios are measured with Sage's configured digestion and
search space.

An audit writes `entrapment.generation.json`, `entrapment.audit.json`, and, when a legacy reference
is supplied, `entrapment.fasta_parity.json`. The parity report compares the selected source,
accession set and order, exclusions, target/entrapment header order, normalized source mapping,
counts, ratios, and hashes. FDRBench header spelling is normalized for semantic comparison, while
the raw file hash is still recorded. The generation report records the seed, exact selection
algorithm, source and output hashes, mappings, exclusions, order hashes, and a deterministic
selection hash so a repeated run can be verified.

Normal workflow execution requires `validation.use_generated_entrapment_ratios: true` (the
default) and does not require literal ratio values in the manifest. Sage fills the effective ratios
from the active dataset's combined FASTA. Therefore an ISB peptide ratio can never be silently
reused for PXD.

## Shared candidate pool and resumable null-window optimization

The workflow stores native pre-FDR candidates under
`OUTPUT_ROOT/candidate_pools/SEARCH_FINGERPRINT/`. The pool is written after native Sage candidate
scoring and configured RT/IMS prediction, but before any Decoy-Free model, q-value method,
entrapment threshold, rescue layer, or MS2Rescore annotation is applied. Compatible Moments, MLE,
Lower Order, MSFDR, MSFDR2-SMIX, Nokoi, and Ensemble optimization stages therefore consume the
same candidate identities and scores. MSFDR1-SMIX remains rank-1-only but may consume the same
pool.

The strict search fingerprint contains ordered spectrum content hashes, the active FASTA content
hash, digestion/modification settings, mass tolerances, charge/isotope settings, native scoring
and preprocessing settings, candidate schema, Sage version, and retained search depth. A changed
search fingerprint automatically selects a separate pool. The pool manifest also declares its
rank-depth and feature capabilities; an analysis cannot request a rank that was not retained.

The separate analysis fingerprint contains the Decoy-Free model/window, evidence and p-value
combination settings, q-value methods/covariates, Storey settings, FDR thresholds, protein
grouping, RT/IMS gates, reproducibility rescue, hierarchical reporting, entrapment ratios, and
external-feature policy. Changing these settings refits/re-evaluates the fixed pool but does not
reread or rescore spectra.

Every cached record carries a stable SHA-256 candidate ID derived from the search fingerprint,
spectrum identity, peptidoform, charge, rank, label, precursor mass, and isotope error. Process-
local `psm_id` values are not cache identity. Both candidate pools and the layered raw MS2Rescore
prediction cache use this stable ID.

New datasets normally declare a compact `window_optimizer` with inclusive `min_rank_range` and
`max_rank_range`. `strategy: landscape_adaptive` uses a compact coarse probe to classify the
observed surface as frontier, interior, or irregular. It applies row-wise boundary search to a
frontier, top-three multi-start hill search with diamond polish to an interior surface, and an
automatic exhaustive fail-safe to irregular or contradicted surfaces. Frontier and interior
results are reported as heuristic; an exhaustive fallback is exact over the bounded universe.
`strategy: adaptive` retains the original sparse-probe heuristic for reproducibility, and
`strategy: exhaustive` always generates and evaluates the complete bounded universe.
The older `candidate_windows` field remains an exact ordered replay interface for frozen
experiments; it is not necessary to enumerate windows manually for new datasets.

Every enabled statistical and Level-4 stage is executed for each visited trial, but INFO-level
per-trial diagnostics are suppressed, only a compact `NullWindowEvaluation` row is retained, and
discarded trials immediately drop their feature/artifact payloads. The selected window is
materialized once more with normal diagnostics and becomes the ordinary fitted artifact.
`elapsed_milliseconds` records trial time. `null_window_optimizer.checkpoint.json` is written after
each new fit and reused only when its candidate-population and analysis fingerprint matches.
`null_window_optimization.json` records the versioned algorithm, declared universe, visit order,
adaptive decision, selected window, timing, and exactness semantics.

Candidate-pool reuse currently targets identification/statistical stages. A cached pool does not
contain MS1/TMT data or matched-fragment payloads, so LFQ, TMT, or `--annotate-matches` execution
fails closed rather than silently emitting incomplete output. The final annotated target-only
stage remains a fresh target-database search. MS2Rescore annotations are cached separately; they
are not mixed into this immutable native pool.

MSFDR1-SMIX is always rank-1-only. A manifest that gives it a null window is rejected.
Ensemble windows are not optimized as one combined window; constituent experts must be
optimized and locked independently. If an `ensemble` model is listed, the workflow runs it after
the individual experts and writes `ensemble.lock.json`; manually copying windows into an Ensemble
JSON is no longer required. Native and MS2Rescore artifacts are kept separate so an artifact fit
after rescoring cannot be used silently in the native comparison.

The broader `parameter_optimizer` workflow section builds on this machinery. It resolves and
optimizes each expert independently, then runs separate final Ensemble combination/aggregation
blocks. Trials require exact existing candidate pools and layered raw caches; search and raw
annotation generation have no fallback. Only +entrapment development evidence can rank trials.
Target-only results are reporting-only after the winner is frozen. The full interface and portable
resume/fingerprint contract are documented in [PARAMETER_OPTIMIZER.md](PARAMETER_OPTIMIZER.md).
Machine-readable binding coverage prevents a catalogued parameter from becoming a silent no-op.
The bounded production-smoke mode evaluates real +entrapment trials, materializes winners, and
then stops before ordinary post-winner or target-only stages.
For complete development optimization, schema v2 provides
`execution_mode: optimization_only`. Unlike the bounded smoke flag, this mode accepts Ensemble and
normal trial budgets. It preflights only +entrapment resources, materializes each expert winner and
the final Ensemble winner through the ordinary production optimizer, records downstream stages as
`not_run_by_execution_scope`, and returns before target-only resource resolution. The default
`optimization_and_post_selection` retains existing workflow behavior for manifests that omit the
field.

Schema v3 independently controls underpowered entrapment evidence. The backward-compatible
`underpowered_trial_policy: not_evaluable` keeps the previous selection block. The explicit
development-only `development_eligible` policy treats
`minimum_entrapment_observations_for_power` as a power/precision diagnostic rather than an FDP
point-estimate ceiling: within-ceiling trials remain rankable, but their validation status is
`not_evaluable_underpowered` and cannot support a statistical-default claim. The policy does not
alter expert participation, objective order, target-only isolation, technical fail-closed rules,
or the adjusted-FDP ceiling.

Schema v4 adds a separate dataset-local entrapment-label holdout. With
`entrapment_validation.mode: selection_audit`, Sage digests the active target and entrapment FASTAs,
links foreign proteins that share any searchable canonical peptide, and cryptographically assigns
whole connected components to selection or audit populations from the frozen seed, salt, and
fractions. Target-shared foreign peptides remain excluded, and the workflow fails closed on an
empty partition, cross-partition peptide/peptidoform/protein group, or identity/ratio mismatch.
Record, candidate, path, thread, and resume ordering cannot alter component assignment.

Both populations remain in the immutable +entrapment candidate pool because production fitting
must see the real score population. Fitting and q-calibration receive the scores but no selection
or audit entrapment role. During optimization, audit identities are ignored—not relabeled as
targets—by both the model-local null-window objective and the outer parameter objective, while
selection identities are used only for those declared development FDP/objective calculations.
Audit labels and metrics are absent from trial checkpoints. Once all individual and
final Ensemble winners are frozen, the workflow reads each retained result table once and writes a
separate immutable audit result with measured audit protein/peptide/peptidoform ratios, adjusted
FDP, a 95% interval, and explicit powered/underpowered status. A sparse or zero audit count remains
not evaluable for statistical validation. Audit findings do not retune winners, run target-only,
or alter the JSON-selected Ensemble roster.
Selection/audit mode is paired with `execution_mode: optimization_only`. Target-only reporting is
a later invocation over frozen settings and cannot be interleaved with audit or optimization.
Materialize the prospective partition first with
`sage materialize-entrapment-partition workflow.json`, record its hash, then set
`require_existing_partition: true` for strict preflight and execution. The materialization command
does not resolve candidate or annotation caches and cannot run searches, fitting, optimizer trials,
or target-only stages.
Run the same command with `--inputs-only` first to freeze the portable input identities without
constructing a component assignment or writing the artifact.

## Layered MS2Rescore prediction and calibration caches

An MS2Rescore stage consumes the same immutable native candidate pool as its optimized stage, so
Sage does not reread or rescore the spectra. Expensive MS2PIP and DeepLC inference is stored under
`RAW_ROOT/raw_predictions/RAW_FINGERPRINT/`. The native pool continues to declare
`external_annotations: false`.

The portable raw identity contains the strict search/candidate-population fingerprint, stable
candidate-ID schema, requested rank depth, candidate count, mapped spectrum-source content,
generator configuration, wrapper and Python content, relevant package versions, selected logical
model names/files/sizes/content hashes, and raw schema. Absolute paths remain provenance only. The
raw identity excludes the Decoy-Free model, selected null window, preliminary q/PEP stream,
target-only policy, Ensemble roster, fitted artifact, and analysis fingerprint.

DeepLC must declare a positive `calibration_set_size` when it participates in the raw layer. An
implicit q-value-selected calibration set is rejected because it would couple raw prediction to a
model/window-specific preliminary stream.

Sage joins raw predictions only by `sage-candidate-id-v1`, then deterministically derives each
model/window-specific external empirical profile in Rust. A separate compact stage-calibration
identity binds the raw fingerprint, preliminary calibration stream, model/stage, and analysis
fingerprint. Process-local `psm_id` values may be used while parsing one newly generated table, but
are never durable identity. A dataset ordinarily needs one raw +entrapment cache and one raw
target-only cache even when many models or windows are evaluated.

Raw manifests and compressed payloads are count-, schema-, identity-, finite-value-, and SHA-256-
checked. A genuine raw miss is generated by default; an existing corrupt or mismatched cache fails
closed. After a new raw cache is written and verified, workflow-managed candidate exports, wrapper
configuration, and feature TSVs are removed.

Workflow manifests may set `require_existing_annotation_cache: true` to prohibit raw prediction
generation. Strict mode accepts only a complete compatible raw hit; absence, schema or identity
mismatch, candidate-population/count mismatch, duplicate or missing stable IDs,
manifest/payload disagreement, corruption, nonfinite required values, or unavailable durable
package/model provenance fails closed before export, temporary files, Python, the wrapper, MS2PIP,
or DeepLC. The option is independent of `require_existing_candidate_pool`; exact replay normally
enables both and may use external candidate and raw-cache roots.

`migrate_schema_v2_annotation_cache_only` is an explicit compatibility path. With
`require_existing_candidate_pool: true`, it may extract the raw payload from an exact,
integrity-valid schema-v2 cache, but it never invokes an external generator. It is mutually
exclusive with strict read-only annotation reuse. Schema-v2 caches are never silently
reinterpreted or overwritten.

When either strict-reuse option is enabled, `workflow --plan-only` performs a read-only preflight
of both +entrapment and target-only resources. Candidate-pool compatibility is portable: original
and current source URIs remain provenance, while equality uses the portable digest, FASTA content,
ordered spectrum ordinals and content hashes, normalized search settings, schema, retained depth,
counts, and manifest/payload integrity. Paths and filenames alone never establish equivalence.

Layered preflight distinguishes raw inference from stage calibration:

1. **Phase A — static read-only preflight.** Validate FASTAs, spectra, search identity, portable
   candidate pools, counts, rank depth, durable generator/model provenance, and the exact raw
   manifest/payload/full stable-ID join. A strict hit is reported as `validated_exact`.
2. **Phase B — stage-local calibration.** Native Rust optimization/fitting resolves the window and
   preliminary stream, derives the stage-calibration identity, and fits the compact external
   profile from the joined raw values. Plan-only reports these stages as
   `deferred_until_calibration`; this is not a missing raw cache and never authorizes generation.

Static plan-only creates no workflow output directory or temporary files and starts no search or
annotation child process. Strict reuse changes workflow execution provenance and checkpoints, but
not the search fingerprint, stable IDs, candidate-pool identity, raw-cache identity, or raw payload.

On macOS, the MS2PIP/XGBoost environment must be able to load `libomp.dylib`. Verify this with an
`import xgboost` using the configured Python executable before a long workflow. If LLVM provides
OpenMP outside the loader's default path, launch Sage with `DYLD_LIBRARY_PATH` set to LLVM's `lib`
directory; this affects runtime loading only and does not weaken the annotation fingerprint.

The target-only stage remains a distinct target-FASTA candidate population. It performs a fresh
search when no exact pool exists or when matched-fragment annotation is requested; otherwise the
immutable target-only pool can be reused. Its raw predictions are cached separately under the
target-only search fingerprint. No raw cache can cross a FASTA, dataset, candidate population,
generator environment, or rank-depth boundary.

## Explicit target-only calibration

`target_only_calibration_policy` replaces the ambiguous term "locked" and defaults to
`refit_with_locked_window`. The setting is workflow-wide and can be overridden on an individual
model subject to that model's declared target-only capability. The policies are:

- `refit_with_locked_window`: retain the window selected on the dataset's +entrapment search, but
  re-estimate nuisance parameters in the target-only candidate space. Target-only outcomes never
  retune the window. This is the initial legacy-parity policy and the default.
- `reuse_dataset_artifact`: apply the complete fitted +entrapment artifact from the same dataset
  without refitting its nuisance state. Artifact model, dataset, search configuration, source
  candidate fingerprint, and `sage-candidate-id-v1` schema must match or the stage fails closed.
- `compare_both`: materialize both interpretations in separate directories and validation rows.
  The refit result is the release candidate; reuse is retained as a diagnostic comparison and
  cannot veto that release candidate.

Lower Order supports only `refit_with_locked_window` across the +entrapment and target-only search
spaces. Its nuisance parameters and candidate-count normalization are search-space dependent and
must be refitted after the FASTA/candidate space changes. For Lower Order, `reuse_dataset_artifact`
fails closed and `compare_both` records the reuse branch as `not_evaluable`; neither path silently
omits or substitutes the unsupported result. The +entrapment-selected window is still reused, and
target-only outcomes never retune it.

Every target-only checkpoint records the policy plus a separate `window_provenance` object with
the source dataset, model, +entrapment stage, selected ranks, source artifact hash, candidate
fingerprint, and candidate-ID schema. Fitted nuisance-state provenance remains in
`fitted_model_artifacts.json`; it is not conflated with window provenance.

The target FASTA produces a strict search fingerprint distinct from +entrapment. If no exact
target-only pool exists, the first interpretation performs a fresh spectrum search and writes one;
subsequent models and `compare_both` interpretations reuse that exact target population. Thus the
policy comparison cannot be confounded by a second spectrum search. The target-only raw prediction
cache is reused across model/window-specific preliminary calibration streams; each stream still
produces a distinct compact stage-calibration identity and fitted profile. If matched-fragment output is
requested, the release interpretation performs a fresh search because immutable candidate pools
deliberately do not persist fragment payloads; the diagnostic second stage remains unannotated and
may reuse the pool.

Legacy validation manifests with a stage named `target_only` remain readable. For engineering
parity, that legacy name maps only to `target_only_refit_with_locked_window`, because the frozen
legacy shell workflow retained the window and refit the target-only nuisance parameters.

## Portable Nokoi

Nokoi uses `sage-nokoi-crossfit-portable-v2`. The artifact freezes the exact ordered feature
contract, imputation and normalization state, deterministic stable-ID fold construction,
fold-specific and final weights/intercepts, complete lambda-evaluation and early-stopping state,
the lambda grid, class construction and sampling rules, the sorted out-of-fold null-score
distribution, pi0, complete Grenander blocks,
the monotone p-to-PEP mapping, candidate-count reference state, source/configuration identities,
and hashes for every state block and the complete canonical payload. Artifact floating-point state
is serialized as hexadecimal IEEE-754 bits; legacy decimal v1 payloads remain readable for
diagnosis but cannot pass v2 portability validation. Absolute paths never participate in identity.

Training, sampling, fold assignment, lambda tie-breaking, model initialization, early stopping,
and calibration use stable deterministic ordering. Applying v2 extracts the frozen feature order,
uses only the frozen model and calibration, and performs no training, cross-validation, lambda
selection, or calibration refit. Missing, corrupt, nonfinite, dimensionally invalid, nonmonotone,
wrong-window, wrong-population, or provenance-incompatible state fails closed.

`refit_with_locked_window` retains the +entrapment-selected window but fits a separate complete v2
artifact in target-only candidate space. `reuse_dataset_artifact` applies the complete +entrapment
artifact without refitting and is restricted to the corresponding target-only population of the
same parent dataset. `compare_both` preserves the two interpretations in separate stages. A
separate dataset must fit its own Nokoi artifact. The historical ISB `2-15` window remains
diagnostic evidence: its seeded shuffle and contiguous folds were still keyed to process-local row
order, while v2 uses stable-candidate-ID sampling and fold assignment. V2 selection follows that
corrected deterministic optimizer contract and is not forced to the historical value.

## Development, holdout, and artifact scope

Both `development` and `holdout` datasets run the same predeclared, dataset-local optimization
procedure. A holdout locks the *procedure*—candidate-grid rules, feasibility criteria,
tie-breaking, model settings, and MS2Rescore decision rules—not another dataset's selected window
or fitted parameters. Candidate windows and `ms2rescore: measure` are therefore valid on a
holdout when they are part of the predeclared procedure.

`artifact_reuse_policy` defaults to `dataset_local_only`. Workflow artifacts and stage
checkpoints are stamped with a content fingerprint derived from the target FASTA and spectra plus
the search-configuration hash. Supplying an artifact without matching provenance fails closed.

`cross_dataset_diagnostic` exists only to reproduce historical diagnostic experiments. It
requires `validation.diagnostic_only: true`, logs the mismatch, and can never satisfy the release
gate. No biological sample-role metadata is required for either normal development or holdout
workflows.

## Lower Order transfer

The entrapment search writes `lower_order_model_artifact.json`. It contains the complete fitted
charge-stratified model, rank window, TEV transformation, candidate-count power/scale,
extrapolation strength, version, and reference candidate-count distribution. The artifact remains
valid for exact same-pool replay and +entrapment Ensemble application. Across a target-only search
space, Lower Order retains only its selected ranks and refits the nuisance state under
`refit_with_locked_window`; complete-artifact reuse is unsupported and fails closed.

The ISB diagnostic established decision-equivalent numerical grid behavior, exact same-pool
artifact replay, deterministic schema-v2 annotation reuse, and controlled target-only refitting at
the locked `6-9` window. Bitwise historical grid parity and complete frozen annotated parity are
`not_evaluable` because the raw historical search payload differs by pre-existing one-ULP values
and the required historical non-rank-1 Linux annotations were not preserved. Preserved rank-1
annotations match exactly.

Lower Order is available for JSON-selected Ensemble voting after dataset-local optimization and
technical fail-closed validation. Its target-only policy remains `refit_with_locked_window`;
statistical diagnostics do not authorize complete cross-space nuisance-artifact reuse. This does
not establish statistical superiority over TDC or eligibility for the statistical default.

## Frozen ISB parity status

The frozen ISB run completed all 22 planned stages and an exact resume reused all stages in about
10 seconds with no new searches. Moments (`9-18`), MLE (`8-25`), Lower Order (`6-9`), seeded
MSFDR (`9-13`), and MSFDR2-SMIX (`9-17`) reproduced every visited legacy grid point. MSFDR1-SMIX
remained fixed at rank `1-1`. Moments, MLE, Lower Order under refit semantics, and MSFDR1-SMIX
also passed the applicable downstream count comparisons; the Moments MS2Rescore result differed
by only two PSMs with exact peptide and protein counts.

Seeded MSFDR and MSFDR2-SMIX later passed individual +entrapment and target-only parity after the
frozen Linux annotation environment was reproduced under WSL. Both are available as JSON-selected
Ensemble voters after technical validation. Parity, holdout, calibration, and yield measurements
remain validation diagnostics rather than voter-admission controls.
The historical Nokoi v1 comparison selected `2-12` rather than legacy `2-15` and failed its old
target-only checks. That result remains diagnostic evidence and is not relabeled. Portable v2
repairs the incomplete, nondeterministic implementation contract; JSON selection plus current
technical validation now controls whether Nokoi votes. Full historical evidence is recorded in
`validation/reports/phase6_isb_model_parity_2026-08-07.json`.

## Independent PXD001468 Moments parity status

PXD001468 was optimized as an independent holdout dataset; it did not import ISB18 windows or
fitted parameters. The native workflow measured protein, peptide, and peptidoform ratios of
`1.0`, `0.5948800028882344`, and `0.6316190823137978` from the active frozen PXD FASTA. It
accounted for all 53 frozen legacy trace rows, excluded the same six invalid rank-1 setup probes,
evaluated all 47 valid windows, matched every comparable count/FDP and feasibility decision, and
selected the exact legacy window `10-10`.

The optimized Level-4 result was exact: 185,310 target PSMs, 24,482 canonical peptides, and 4,397
proteins. The native MS2Rescore stage reported 216,922 PSMs, 32,094 peptides, and 5,089 proteins,
all within the predeclared 0.5% platform tolerance of the frozen 217,296/32,185/5,100 baseline.
Its Level-4 peptide gain was 7,612 versus the frozen gain of 7,703. Target-only
`refit_with_locked_window` retained `10-10`, refit only the nuisance state, and reported
221,561/34,084/5,276 versus 221,847/34,143/5,283; transfer was stable and within tolerance.

Both the +entrapment and target-only MS2Rescore caches joined exactly one annotation per stable
candidate ID (12,033,536 and 12,181,311 rows, respectively). A complete rerun resumed all three
stages in about 92 seconds without another spectrum search or Python feature-generation process.
The optimizer's 47 trial evaluations took 3,111 seconds in aggregate. Peak observed resident
memory was about 38.6 GiB while loading/evaluating the large candidate pool, so further streaming
or compact-record work remains a useful engineering optimization.

This is an engineering-parity pass, not evidence for a statistical default change. The generic
release report still requests a matched TDC benchmark, and both the legacy and native
post-MS2Rescore protein FDP are slightly above 1% (about 1.09% and 1.13%). Those facts are recorded
rather than mislabeled as a failed PXD parity run. Full evidence is in
`validation/reports/phase7_pxd001468_moments_parity_2026-08-08.json` and the complete grid report
is in `validation/reports/phase7_pxd001468_moments_null_window_parity_2026-08-08.json`.

PXD Moments is the only required PXD model for this refactor. Do not automatically run additional
PXD models; decide later whether one is scientifically or technically necessary.

## Secondary-model and Ensemble release policy

The core Ensemble is continuous and PSM-first. Every requested, technically valid model supplies
its continuous PSM p-value and PEP-like streams. The configured Ensemble combiners operate on
those streams, Ensemble PSM q-values are calculated, peptide q-values are derived from the combined
PSM stream, and protein q-values are derived downstream. No accepted-list masking or model-level
peptide/protein voting is performed.

Each enabled individual model may declare `ensemble_participation: auto` to request one vote, or
`ensemble_participation: excluded` with a required `ensemble_exclusion_reason`. Technical
fail-closed validation produces a separate actual roster. Canonical sorting exists only for
serialization and reproducibility; it never assigns credit by discovery count. Duplicate canonical
models and duplicate artifact votes fail closed.

`minimum_incremental_ensemble_peptides` and `maximum_transfer_fraction_loss` remain accepted for
backward-compatible JSON loading and validation reports, but are deprecated as runtime admission
controls. They cannot change `expert.enabled`, the requested roster, or the actual roster.

The current production participation policy is:

- Moments, MLE, rank-1-only MSFDR1-SMIX, seeded MSFDR, MSFDR2-SMIX, Lower Order, and Nokoi are
  selectable voters. MSFDR1-SMIX remains fixed at 1-1; every variable-window model independently
  optimizes its own dataset-local window.
- Lower Order supports target-only `refit_with_locked_window` only; cross-space
  `reuse_dataset_artifact` is explicitly unsupported, and target-only outcomes never tune its
  window.
- Nokoi requires a complete `sage-nokoi-crossfit-portable-v2` artifact. Legacy v1 artifacts remain
  readable as historical diagnostics but cannot enter an Ensemble or target-only reuse path.

The schema-v9 Ensemble lock copies every actual voter's independently optimized dataset-local
window, complete effective expert-local production configuration, configuration hash, fitted
artifact, and dataset/search/candidate/annotation identities; it never optimizes one combined
window. Windows alone are insufficient because purification, robust fitting, mixture/classifier,
and per-expert calibration settings can all change the reconstructed stream. Target-only
`refit_with_locked_window` resolves each expert from its own locked configuration and refits only
the nuisance state permitted by that model. Workflow defaults cannot replace locked values. The
separate final Ensemble configuration controls combination and downstream aggregation only.
Schema-v8 and older incomplete locks fail closed for target-only refit or `compare_both`; users
must materialize a new lock from the frozen single-value development configuration rather than
inferring omitted values from contemporary defaults.

A requested voter is excluded only for technical
failures such as a missing/corrupt artifact, model or dataset/search/analysis mismatch, prohibited
fallback, incompatible candidate/annotation/fitted-profile provenance, unsupported target-only
state, nonfinite or invalid fitted state, or duplicate model/artifact vote. Statistical measures—
including entrapment FDP, observation counts, transfer-loss percentage, parity, unique/incremental
yield, interaction calibration, holdout outcome, and release/default eligibility—are nonblocking
diagnostics.

For newly eligible experts, `ensemble_interaction_baseline` distinguishes the established
counterfactual Ensemble from the final assembled Ensemble without changing any gate. The workflow
reuses the same candidate pool and annotation cache to report baseline and final raw-q and Level-4
PSM, canonical-peptide, and peptidoform entrapment FDP, including measured-ratio numerators,
denominators, and absolute/relative changes. Raw-q deterioration greater than `0.01` is emitted as
a structured informational warning and is never mislabeled as a passing gate. Level-4 interaction
calibration is also reported but does not change the roster or suppress results. The report is stored in
`validation.ensemble_interaction.json`, the selected Ensemble stage checkpoint, and workflow state;
the schema-v8 lock records requested and actual rosters, explicit exclusions, technical failures,
constituent identities, independently selected windows, target policies, combiner settings, each
expert's fitted-profile and annotation-cache provenance, and a separate canonical shared-profile
contract identity that binds only the shared dataset/search/candidate and explicit calibration
inputs. The complete lock analysis fingerprint separately binds the roster and every expert's
artifact and annotation provenance. The
shared dataset-local profile is fitted once at the explicit 9-18 window and cannot be selected by
expert ordering or overwritten by an expert artifact.

Ensemble remains optional and cannot block the core refactor. If fewer than
`minimum_ensemble_experts` are technically valid, `ensemble.lock.json` is still written with `evaluable: false` and
the reasons, and Ensemble stages are skipped without invalidating completed individual-model
stages. Applying a non-evaluable lock fails closed. The historical Phase 8 policy remains preserved
in `validation/policies/phase8_isb_ensemble_expert_policy_2026-08-08.json`. The current status is in
`validation/policies/current_ensemble_expert_policy_2026-08-15.json`; implementation evidence for
the original secondary-expert repair is in
`validation/reports/phase8_secondary_model_repairs_2026-08-08.json`.

No additional PXD model search was run for the secondary-model audit. PXD Moments remains the only required PXD
parity run for this refactor.

`precursor_fdr`, `peptide_fdr`, and `protein_fdr` control reported PSM/precursor, peptide, and
protein identifications respectively. They are downstream reporting thresholds and never select
Ensemble voters. Entrapment FDP and related validation measurements may be written to reports, but
they cannot alter the requested or actual roster.

## Final release evaluation and resumption integrity

The final release gate has three explicit outcomes:

- `eligible`: all required evidence is evaluable and every release criterion passes;
- `not_eligible`: the evidence is complete, but at least one scientific or engineering criterion
  fails; or
- `not_evaluable`: required evidence is missing, invalid, unreadable, or cannot be linked to its
  declared calibration source.

Missing result files and malformed tables are collected in `validation.missing_runs.json` and
`validation.invalid_runs.json`; they no longer abort before an audit can describe the problem.
External parity evidence remains supplementary and cannot replace a declared dataset-local
baseline/native comparison. Every declared parity stage/layer must have a local comparison.
Likewise, a target-only result must identify an existing calibration stage and have an explicit
transfer comparison. The validator never falls back to another calibration stage when the
declared source is absent.

All validation layers now carry four consistently defined identification counts:

- PSM: a distinct rank-1, label-1 result-table PSM identity;
- peptide: the unmodified sequence with bracketed modifications removed and I/L canonicalized;
- peptidoform: the sequence with bracketed modification annotations retained and unmodified I/L
  canonicalized; and
- protein: one unambiguous inferred protein key.

Contaminants and ambiguous target/entrapment mappings are excluded from all four definitions.
Peptidoform FDP uses the measured peptidoform ratio already stored as the workflow's PSM ratio.

Stage checkpoint schema 2 hashes both the result table and resolved search configuration. A cache
hit also verifies candidate-pool schema, identity, capability, count, and payload hash; annotated
stages additionally verify the layered raw MS2Rescore prediction cache and stage-calibration
identity. A checkpoint left `running`
by interruption is never resumed as complete, and a changed durable output invalidates the stage.
Compatible schema-1 checkpoints are migrated once after their existing dataset, input, output,
candidate-pool, and annotation identities pass.

Final cache-hit validation resumed all 22 frozen ISB stages twice with zero new searches, then
resumed the three required PXD Moments stages with zero new searches. All 25 checkpoints were
migrated to hashed outputs; both workflows reported zero missing and zero invalid runs. PXD kept
all six declared parity comparisons within tolerance. Full evidence is recorded in
`validation/reports/phase9_release_finalization_2026-08-09.json`.

## Outputs and resumption

Each stage has a resolved search configuration and a hash checkpoint. A completed stage is reused
only if its manifest, model, FASTA, spectra, search configuration, and frozen artifact hashes still
match. Important workflow reports include:

- `entrapment.generation.json`
- `entrapment.fasta_parity.json` when a legacy generation reference is declared
- `entrapment.input.json` containing the Sage-measured ratios applied to the searches
- `null_window_evaluations.json` in an optimized model directory
- `candidate_pools/<search fingerprint>/candidate_pool.json`
- `candidate_pools/<search fingerprint>/candidate_pool.bin.zst`
- `workflow.candidate_pools.json` with per-stage search and analysis fingerprints and reuse status
- `ms2rescore_annotations/raw_predictions/<raw fingerprint>/raw_external_predictions.json`
- `ms2rescore_annotations/raw_predictions/<raw fingerprint>/raw_external_predictions.bin.zst`
- `workflow.ms2rescore_annotations.json` with per-stage generation/reuse and joined counts
- `validation.summary.json`
- `validation.stage_comparisons.json`
- `validation.transfer_stability.json`
- `validation.missing_runs.json`
- `validation.invalid_runs.json`
- `validation.ensemble_expert_gates.json`
- `validation.parity.json`
- `validation.tdc_benchmarks.json`
- `validation.release_gate.json`
- `ensemble.lock.json`
- `workflow.state.json`

For frozen FDRBench parity, compare every visited null window under the same canonical counting
definition rather than comparing only the winner or the legacy log's modification-sensitive
peptide counter:

```bash
python3 tools/compare_null_window_evaluations.py \
  LEGACY/cumulative_results_long.csv \
  NATIVE/optimized/null_window_evaluations.json \
  --legacy-results-root LEGACY/outdirs \
  --psm-ratio 0.7754795035727717 \
  --peptide-ratio 0.7667638483965015 \
  --protein-ratio 1.0 \
  --expected-window 9-18 \
  --output null-window-parity.json
```

Both the legacy long and transposed wide cumulative CSV layouts are accepted. Supplying the
legacy results root recalculates PSM, unmodified I/L-canonical peptide, protein, FDP, and Level-4
counts directly from each frozen result table.

Concrete target-only results are written under
`MODEL/target_only/refit_with_locked_window/` and/or
`MODEL/target_only/reuse_dataset_artifact/`. Their stage names and validation rows are likewise
explicit; `compare_both` never overwrites one interpretation with the other.

Expert reports mark an accepted entrapment count below the configured stability minimum as
underpowered even when the point FDP estimate is zero. The warning is not an automatic veto,
because requiring several accepted entrapments while also requiring FDP <=1% can be mathematically
impossible in a genuinely sparse experiment. This is important for ultra-low-input validation:
zero observed errors is not automatically evidence of exact zero FDP, so development/holdout and
replicate evidence remain necessary.

Freeze corrected reference outputs independently with:

```bash
sage freeze-baseline --output validation/baseline.json /path/to/results /path/to/plots
```

Use `--status` when a snapshot is intentionally partial; a dataless cloud placeholder must never
be labeled complete. Completed result tables can also be audited without rerunning spectra:

```bash
sage validate-results validation/manifests/isb18_corrected_results_audit_2026-08-05.json
```

The audit distinguishes Decoy-Free from TDC methods, compares Decoy-Free Level 4 with TDC's
reportable q-value layer, and links a final target-only result to the +entrapment calibration stage
whose frozen artifact it actually used. Peptide counts are unmodified I/L-canonical sequences;
modified peptide strings are not silently reported as peptide counts.

For independent validation, engineering parity and later statistical-performance studies are
separate gates. External parity evidence can supplement a report, but it cannot replace a
dataset-local baseline/native parity pair. ISB18 must reproduce the frozen ISB18 workflow, and
PXD001468 Moments must reproduce the frozen PXD001468 Moments workflow after independently
optimizing PXD001468.

The refactor parity sequence is:

1. preserve the complete corrected ISB18 baseline and the corrected PXD001468 Moments baseline;
2. validate native FASTA generation separately from search/optimizer parity;
3. run ISB18 with the internal optimizer and require the dataset/model-specific selected windows
   and outputs to match the frozen ISB18 workflow;
4. run only PXD001468 Moments initially, with the same optimizer procedure but a PXD-specific
   search and selected window; and
5. decide later whether the runtime cost of additional PXD001468 model fits is scientifically or
   technically necessary.

The historical ISB-artifact-on-PXD run is retained only as a diagnostic demonstrating why
cross-dataset artifact reuse is prohibited. `tools/prepare_pxd_moments_holdout.sh` is disabled
unless an explicit diagnostic environment variable is set. Never replace a frozen baseline after
seeing new behavior; record a new baseline instead.
