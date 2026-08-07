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
7. evaluate requested null windows in memory from those retained candidates;
8. lock the highest-yield window satisfying PSM, peptide, and protein entrapment-FDP limits;
9. measure MS2Rescore and retain it only when its configured gain/calibration gates pass;
10. run the target-only search under an explicit calibration policy, defaulting to a locked
    dataset-local window with nuisance parameters refit in the target-only candidate space;
11. assemble Ensemble automatically from the independently selected expert windows and
    stage-matched frozen artifacts;
12. emit raw-q and Level-4/reportable counts, FDP, direct optimized/MS2Rescore/target-only
    comparisons, missing-run reports, transfer-stability results, parity checks, constrained TDC
    comparisons, Ensemble expert-quality gates, and a release gate.

The primary invariant is **dataset-local optimization**. Every dataset and every individual model
fit creates or selects its own entrapment FASTA, measures its own ratios, and selects its own null
window. A window or fitted artifact selected on ISB18 is never used as PXD001468's normal search
configuration. The same rule applies to any future dataset pair.

## Phase 2: FASTA generation and optimizer isolation

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

## Phase 3: shared candidate pool and exact lean optimization

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
local `psm_id` values are not cache identity. Phase 4 uses this stable ID for the separate
MS2Rescore annotation cache.

Null-window grids run in `lean_exact` mode by default. Every enabled statistical and Level-4
stage is executed for every trial, but INFO-level per-trial diagnostics are suppressed, only a
compact `NullWindowEvaluation` row is retained per trial, and discarded trials immediately drop
their feature/artifact payloads. The selected window is materialized once more with normal
diagnostics and becomes the ordinary fitted artifact. `elapsed_milliseconds` records trial time.
No approximate or coarse-to-fine statistical search is used.

Candidate-pool reuse currently targets identification/statistical stages. A cached pool does not
contain MS1/TMT data or matched-fragment payloads, so LFQ, TMT, or `--annotate-matches` execution
fails closed rather than silently emitting incomplete output. The final annotated target-only
stage remains a fresh target-database search. Phase 4 caches MS2Rescore annotations separately;
it does not mix them into this immutable native pool.

MSFDR1-SMIX is always rank-1-only. A manifest that gives it a null window is rejected.
Ensemble windows are not optimized as one combined window; constituent experts must be
optimized and locked independently. If an `ensemble` model is listed, the workflow runs it after
the individual experts and writes `ensemble.lock.json`; manually copying windows into an Ensemble
JSON is no longer required. Native and MS2Rescore artifacts are kept separate so an artifact fit
after rescoring cannot be used silently in the native comparison.

## Phase 4: separately cached MS2Rescore annotations

An MS2Rescore stage now consumes the same immutable native candidate pool as its optimized stage,
so Sage does not reread or rescore the spectra. External feature generation still reads the
spectra it needs, but the resulting annotations are stored independently under
`OUTPUT_ROOT/ms2rescore_annotations/ANNOTATION_FINGERPRINT/`. The native pool continues to declare
`external_annotations: false`.

The annotation fingerprint contains the strict search fingerprint, ordered stable candidate IDs,
the exact preliminary score/q-value/PEP calibration input exported to MS2Rescore, the requested
rank depth, generator configuration, mapped spectrum-source hashes, wrapper and Python hashes,
detected Python/package versions, and annotation schema. Temp/output locations, failure policy,
and downstream evidence-use policy are not generator identity. Changing a model, selected null
window, or another setting that changes preliminary q/PEP values selects a new annotation cache.
This model/window-local behavior preserves the pre-refactor DeepLC calibration semantics; sharing
only model-independent MS2PIP components can be considered later as a separately validated
optimization.

Annotations are joined only by `sage-candidate-id-v1`. Process-local `psm_id` values may still be
used while parsing one freshly generated external table, but they are never persisted as cache
identity. Cache manifests and compressed payloads are count-, schema-, identity-, and SHA-256-
checked. A missing annotation set is generated; an existing corrupt or mismatched set fails
closed. Workflow stages that claim MS2Rescore must record at least one joined annotation. After a
new cache has been written and verified, the workflow removes its temporary exported candidate
table, wrapper configuration, and feature-rich TSV; direct one-off searches without a cache retain
their historical temporary-file behavior, as do configurations that explicitly request an
external output directory.

On macOS, the MS2PIP/XGBoost environment must be able to load `libomp.dylib`. Verify this with an
`import xgboost` using the configured Python executable before a long workflow. If LLVM provides
OpenMP outside the loader's default path, launch Sage with `DYLD_LIBRARY_PATH` set to LLVM's `lib`
directory; this affects runtime loading only and does not weaken the annotation fingerprint.

The target-only stage remains a distinct target-FASTA candidate population. It performs a fresh
search when no exact pool exists or when matched-fragment annotation is requested; otherwise the
immutable target-only pool can be reused. If MS2Rescore is selected for that stage, its external
annotations are still independently cacheable under the target-only search fingerprint. No
annotation cache can cross a FASTA, dataset, candidate calibration, generator environment, or
rank-depth boundary.

## Phase 5: explicit target-only calibration

`target_only_calibration_policy` replaces the ambiguous term "locked" and defaults to
`refit_with_locked_window`. The setting is workflow-wide and can be overridden on an individual
model while Lower Order is being evaluated. The policies are:

- `refit_with_locked_window`: retain the window selected on the dataset's +entrapment search, but
  re-estimate nuisance parameters in the target-only candidate space. Target-only outcomes never
  retune the window. This is the initial legacy-parity policy and the default.
- `reuse_dataset_artifact`: apply the complete fitted +entrapment artifact from the same dataset
  without refitting its nuisance state. Artifact model, dataset, search configuration, source
  candidate fingerprint, and `sage-candidate-id-v1` schema must match or the stage fails closed.
- `compare_both`: materialize both interpretations in separate directories and validation rows.
  The refit result is the release candidate; reuse is retained as a diagnostic comparison and
  cannot veto that release candidate.

Every target-only checkpoint records the policy plus a separate `window_provenance` object with
the source dataset, model, +entrapment stage, selected ranks, source artifact hash, candidate
fingerprint, and candidate-ID schema. Fitted nuisance-state provenance remains in
`fitted_model_artifacts.json`; it is not conflated with window provenance.

The target FASTA produces a strict search fingerprint distinct from +entrapment. If no exact
target-only pool exists, the first interpretation performs a fresh spectrum search and writes one;
subsequent models and `compare_both` interpretations reuse that exact target population. Thus the
policy comparison cannot be confounded by a second spectrum search. An MS2Rescore annotation cache
is reused only when the preliminary calibration inputs are identical; otherwise Phase 4's
fingerprint correctly generates a policy-specific annotation set. If matched-fragment output is
requested, the release interpretation performs a fresh search because immutable candidate pools
deliberately do not persist fragment payloads; the diagnostic second stage remains unannotated and
may reuse the pool.

Legacy validation manifests with a stage named `target_only` remain readable. For engineering
parity, that legacy name maps only to `target_only_refit_with_locked_window`, because the frozen
legacy shell workflow retained the window and refit the target-only nuisance parameters.

## Portable Nokoi

Nokoi now writes its final logistic model, exact feature schema, imputation medians,
normalization means/standard deviations, selected L1 penalty, deterministic fold metadata,
out-of-fold null-score distribution, dataset-local pi0, and frozen monotone p-to-PEP calibration.
The `reuse_dataset_artifact` target-only policy may evaluate the artifact fitted earlier in the
same dataset workflow without training a new classifier or re-estimating the null/PEP calibration.
The default `refit_with_locked_window` interpretation instead refits Nokoi in the target-only
candidate space and must be evaluated during model-specific parity. A separate holdout
dataset fits its own Nokoi model and null window under the same predeclared procedure. Missing,
cross-dataset, incompatible, or incomplete artifacts fail closed.

The Phase 6 ISB audit found that this is not yet a complete portable artifact for release. It does
not retain fold-specific weights/intercepts, an explicit fold-membership reconstruction rule,
complete training-rule and Grenander state, or source hashes. Its native frozen-grid result also
selected `2-12` rather than legacy `2-15`. Nokoi is therefore explicitly deferred; target-only
must not silently retrain it for the first refactor release.

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
extrapolation strength, version, and reference candidate-count distribution. Under
`reuse_dataset_artifact`, the target-only stage validates and loads this artifact without
refitting. Candidate counts are empirically quantile-normalized to the reference +entrapment
distribution. Under `refit_with_locked_window`, Lower Order retains only its selected ranks and
fits target-only nuisance state. Missing or incompatible reuse artifacts fail closed.

This normalization is intentionally auditable and must first pass the complete ISB18 parity and
same-dataset target-only tests before Lower Order is restored as an Ensemble default. A costly
PXD001468 Lower Order run is optional and will be considered only after PXD001468 Moments parity.

Phase 6 established exact ISB grid, MS2Rescore, and target-only parity for
`refit_with_locked_window`. The diagnostic `reuse_dataset_artifact` interpretation did not match:
it produced 6,479 Level-4 PSMs, 291 peptides, and 17 proteins instead of the legacy/refit
558/44/10. Lower Order remains excluded from the production Ensemble until that normalization
behavior is repaired; the exact refit result does not make artifact reuse release-eligible.

## Phase 6: frozen ISB parity status

The frozen ISB run completed all 22 planned stages and an exact resume reused all stages in about
10 seconds with no new searches. Moments (`9-18`), MLE (`8-25`), Lower Order (`6-9`), seeded
MSFDR (`9-13`), and MSFDR2-SMIX (`9-17`) reproduced every visited legacy grid point. MSFDR1-SMIX
remained fixed at rank `1-1`. Moments, MLE, Lower Order under refit semantics, and MSFDR1-SMIX
also passed the applicable downstream count comparisons; the Moments MS2Rescore result differed
by only two PSMs with exact peptide and protein counts.

Seeded MSFDR and MSFDR2-SMIX matched their unannotated fits but exceeded downstream tolerance
after regenerating MS2PIP/DeepLC features on macOS. Candidate identities and hyperscores matched;
the external features did not match the frozen Linux environment. These annotated variants are
deferred until their environment can be reproduced or their platform robustness is repaired.
Nokoi failed the frozen fit, selected-window, and target-only checks and is deferred as described
above. The production Ensemble is not assembled from these incomplete experts. Full evidence is
recorded in `validation/reports/phase6_isb_model_parity_2026-08-07.json`.

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
- `ms2rescore_annotations/<annotation fingerprint>/ms2rescore_annotations.json`
- `ms2rescore_annotations/<annotation fingerprint>/ms2rescore_annotations.bin.zst`
- `workflow.ms2rescore_annotations.json` with per-stage generation/reuse and joined counts
- `validation.summary.json`
- `validation.stage_comparisons.json`
- `validation.transfer_stability.json`
- `validation.missing_runs.json`
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
