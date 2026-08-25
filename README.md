# Sage Decoy-Free

> **Experimental research fork.** This repository adds Decoy-Free false-discovery-rate (FDR)
> estimation to the Sage search engine. It is not an official Sage release and is not currently
> validated as a drop-in statistical replacement for conventional target-decoy competition (TDC).

This fork is designed for validation-first analysis of low-input and ultra-low-input proteomics,
where sparse decoy counts can limit empirical FDR resolution. It provides both direct Decoy-Free
searches and a native, reproducible workflow that performs the dataset-specific work required to
evaluate them responsibly.

The native workflow can:

1. select a foreign-species FASTA deterministically;
2. generate a seeded protein-level 1:1 target-plus-entrapment FASTA with Sage digestion rules;
3. measure the active FASTA's protein, peptide, and peptidoform ratios rather than using hard-coded
   ratios from another dataset;
4. search spectra once per strict search fingerprint and retain an immutable candidate pool;
5. optimize each model's null-rank window independently from that shared pool;
6. cache MS2Rescore annotations separately and join them by stable candidate identity;
7. run a target-only search under an explicit calibration policy;
8. assemble an optional Ensemble only from dataset-local experts that pass every declared gate;
9. resume verified stages without silently accepting interrupted, modified, or corrupt outputs; and
10. emit provenance, parity, transfer, TDC-comparison, validation, and release-gate reports.

The central statistical pipeline still separates:

1. **Base Decoy-Free evidence** from a fitted score/null model.
2. **Optional physical evidence updates** from retention time (RT) and ion mobility (IMS).
3. **Optional reproducibility updates** from expert agreement and cross-run recurrence.
4. **Reporting-only hierarchical inference**, which may identify protein-supported peptides and
   PSMs without overwriting the active Decoy-Free q-value streams.

Some standard Sage output paths are not validated with the Decoy-Free workflow, including LFQ,
TMT, PIN, parquet, and matched-fragment annotation from a reused candidate pool. These modes fail
closed where the cache lacks the required payload, but they should still be independently validated
before scientific use.

## Required validation-first workflow

Decoy-Free FDR estimation is model- and dataset-dependent. There is no universal null-rank window,
fitted artifact, or model choice that can safely be copied between datasets. Every dataset and every
model must optimize its own window using that dataset's active entrapment FASTA and measured ratios.

The analysis still contains two search spaces, but `sage workflow` orchestrates them as one
resumable workflow:

1. **Target plus entrapment:** generate or select the entrapment FASTA, search spectra once per
   strict fingerprint, and choose each model's highest-yield feasible null window.
2. **Target only:** search the biological target FASTA and apply the chosen target-only calibration
   policy without using target-only outcomes to retune the entrapment-selected window.

Null windows are evaluated in memory from a shared candidate pool. For a new dataset, declare
compact rank bounds and use the landscape-adaptive search; Sage generates and visits windows
internally. The workflow does not edit JSON repeatedly, launch a new spectrum search for every
window, or require a user to copy a selected window into another configuration file.

Do not transfer a window or fitted artifact from one dataset to another for normal analysis.
Cross-dataset reuse is permitted only when explicitly declared diagnostic-only and can never satisfy
a release gate.


## Current workflow capabilities

### Native entrapment generation

`sage workflow` replaces the former manual `make_entrapment.sh` step. Given a target FASTA and one
or more candidate foreign FASTAs, Sage digests each source in its own search space, excludes shared
peptides, selects proteins deterministically, writes explicit `Ent_` headers, and measures the final
protein, peptide, and peptidoform ratios.

Foreign-source selection supports:

- `automatic`: choose the source whose measured peptide and peptidoform ratios best approach the
  requested protein ratio;
- `explicit`: use only `selected_foreign_fasta`; and
- `automatic_with_override`: record the automatic recommendation but generate from the declared
  override.

The production shared-peptide mode is `sage_search_space`. The
`fdrbench004_compatibility` mode exists only to reproduce frozen FDRBench 0.0.4 behavior during
legacy engineering comparisons. `make_entrapment.sh` is not part of the current repository and is
not required for a new workflow.

To audit FASTA construction without searching spectra:

```bash
sage audit-entrapment entrapment.audit.json
```

See [`entrapment.audit.example.json`](entrapment.audit.example.json) for the audit schema.

### Shared candidate pool and in-memory optimization

`sage workflow` performs one native spectrum search per strict search fingerprint. Compatible
models and bounded or explicit window searches reuse a compressed, immutable pre-FDR candidate pool. Changes to
statistical settings create a new analysis fingerprint and refit the fixed candidates; changes to
the FASTA, spectra, digestion, modifications, tolerances, scoring, preprocessing, or retained rank
depth create a different search fingerprint and therefore a different pool.

Candidate pools are portable across filesystem relocation. The manifest retains both its original
spectrum URIs and the current resolved URIs as provenance, but absolute path spelling is not
scientific identity. Reuse instead requires the same portable fingerprint digest, FASTA hash,
ordered spectrum ordinals and content hashes, normalized search configuration, candidate schema,
rank depth, candidate count, and payload hash. A macOS path and a WSL path can therefore reference
the same pool content; reordering or changing any input still fails closed.

The optimizer has four interfaces:

- `window_optimizer.strategy: "landscape_adaptive"` is the recommended new-dataset mode. It uses
  a compact coarse probe to classify the observed surface as `frontier`, `interior`, or
  `irregular`. Frontier surfaces use row-wise boundary search; interior surfaces use top-three
  multi-start hill search plus a radius-two diamond polish. Irregular surfaces—and surfaces that
  contradict their initial classification—automatically fall back to exhaustive evaluation.
- `window_optimizer.strategy: "adaptive"` preserves the original deterministic sparse-probe
  heuristic for reproducibility. It does not use the new landscape classifier or exhaustive
  fail-safe.
- `window_optimizer.strategy: "exhaustive"` generates every valid window inside the same compact
  bounds and is exact over that bounded universe.
- `candidate_windows` evaluates an explicit ordered list exactly. It remains useful for frozen
  historical replay; users do not need to enumerate windows for a new dataset.

Every trial retains compact metrics, and the winning artifact is materialized once with full
diagnostics. `null_window_optimizer.checkpoint.json` is updated after each new trial, allowing an
interrupted optimization to resume without refitting completed windows. The final
`null_window_optimization.json` records the algorithm version, strategy, bounds, adaptive-mode
decision, complete visit order, selected window, timing, and whether the result is exact over the
declared universe. MSFDR1-SMIX is always rank `1-1`. Ensemble does not optimize a combined window:
every constituent expert must supply its own independently optimized dataset-local window and
artifact.

The versioned workflow `parameter_optimizer` is a separate development-only optimizer for bounded
analysis parameters. It supports independent per-expert grids or staged-coordinate blocks,
per-expert q/calibration settings, final Ensemble-only combination and aggregation blocks, exact
checkpoint resume, and strict immutable candidate/raw-cache reuse. It never treats reporting FDR
thresholds as tunable variables and never consumes target-only outcomes. See
[PARAMETER_OPTIMIZER.md](PARAMETER_OPTIMIZER.md) for the schema, ownership and precedence rules,
objective semantics, portable fingerprint, and failure classifications.
Expert identities are canonical across manifests, checkpoints, artifacts, Ensemble locks, and
target-only reconstruction: `moments`, `mle`, `lower_order`, `msfdr`, `msfdr1_smix`,
`msfdr2_smix`, and `nokoi`. Legacy input aliases `msfdr_seeded`, `msfdr_1smix`, and
`msfdr_2smix` normalize before validation and fingerprinting; durable output uses canonical names,
and duplicate canonical/alias map entries fail closed.
Every executable candidate has serialized production-binding coverage. A bounded
production-smoke mode verifies real +entrapment fitting and objective selection without entering
target-only reporting.
Final-Ensemble winners use transactional schema-v10 locks: the exact selected final configuration
and exact expert configuration/artifact mapping are validated before atomic replacement, reopened,
and validated again before the workflow can succeed. Effective scientific identities hash fully
resolved settings, making omitted, `null`, and explicit defaults equivalent without dropping any
active setting. Schema-v9 optimizer locks fail closed for target-only use and must be regenerated
by frozen replay; completed checkpoints can recover lock materialization without reevaluating
trials.
Frozen expert hashes are prepared inputs-only with
`sage resolve-frozen-expert-configurations WORKFLOW.json --output RESOLUTION.json`. The command
uses the production stage projection and effective `FdrSettings` resolver for every single-valued
expert block, emits one immutable canonical artifact, and performs no spectra, cache, fitting,
target-only, or optimizer-trial work. Strict preflight independently resolves and compares the
complete roster before any production trial. Do not manufacture current hashes by editing older
stage records or replacing only their implementation identity.
Schema-v2 optimizer manifests can additionally set `execution_mode: optimization_only` for a full
development run that materializes all individual and final Ensemble +entrapment winners and then
returns successfully without resolving or executing any target-only resource or stage. Omitting
the field preserves the historical `optimization_and_post_selection` behavior.
Schema v3 also adds an explicit `underpowered_trial_policy`. Its default `not_evaluable` preserves
historical blocking behavior. `development_eligible` may be requested only for development-only
runs: a technically valid, within-ceiling zero/sparse-entrapment trial may enter the unchanged
development objective, but remains `not_evaluable_underpowered` for empirical validation and
`not_evaluated` for statistical-default eligibility.
Schema v4 adds dataset-local `entrapment_validation`. The backward-compatible
`full_population_development` mode exposes the complete entrapment population to development
selection and makes no independent calibration claim. The prospective `selection_audit` mode
partitions foreign proteins by content-stable shared-searchable-peptide components before fitting.
Production fitting and q-calibration see every candidate score but no entrapment partition role.
Only the workflow's development FDP/objective sees selection labels and measured selection ratios;
audit labels remain hidden until every expert and final Ensemble winner is frozen, then each winner
is evaluated once with separately measured audit ratios and uncertainty. The partition changes the
analysis/optimizer identity, never the spectrum-search candidate pool or raw prediction-cache
identity, and never controls Ensemble participation. Every new dataset constructs its own
partition and reruns local parameter/window optimization; previously selected ISB settings are not
portable defaults.

Create and freeze the prospective artifact before any optimizer trial with
`sage materialize-entrapment-partition workflow.json`. This dedicated path reads only the manifest,
digestion/search-space configuration, dataset identities, target and active +entrapment FASTAs, and
the existing entrapment-construction report. It does not resolve candidate or annotation caches,
search spectra, fit a model, evaluate a trial, or access target-only resources.
Its `--inputs-only` form reports the prospective identities without assigning components or
writing the artifact.

### Layered MS2Rescore prediction cache

Expensive MS2PIP and DeepLC outputs are stored outside the immutable Sage candidate pool and joined
with `sage-candidate-id-v1`, never with a process-local `psm_id`. The raw-cache identity includes
the exact candidate population and retained depth, spectra, portable generator/model-file content,
wrapper and relevant Python/package identities, and the raw feature schema. It deliberately
excludes the Decoy-Free model, null window, preliminary q/PEP stream, target-only policy, and
Ensemble roster. Sage derives each model/window-specific external empirical profile from this raw
layer. Consequently a dataset ordinarily needs one raw +entrapment cache and one raw target-only
cache, even when many models or windows are evaluated.

When DeepLC is enabled, layered caching requires an explicit positive `calibration_set_size`.
This makes its dataset-level calibration selection deterministic from the native candidate score
instead of allowing an implicit q-value filter to make raw predictions stage-dependent.

`require_existing_annotation_cache` defaults to `false`, preserving ordinary from-scratch
generation after a genuine cache miss. Set it to `true` for exact cache-only replay: every required
+entrapment and target-only raw cache must pass schema, identity, population, stable-ID, payload,
and durable package/model-provenance checks. Static `workflow --plan-only` can validate the raw
prediction identity before fitting because it has no preliminary-calibration dependency. Native
Rust then computes a separate deterministic stage-calibration identity and fits the compact
external empirical profile. A raw hit starts no Python, wrapper, MS2PIP, or DeepLC process.

This option is independent of `require_existing_candidate_pool`; enable both for cache-only
execution. Strict mode guarantees zero external prediction-generation fallback, while native
model/profile calibration remains ordinary inexpensive scientific computation. Set
`migrate_schema_v2_annotation_cache_only: true` for an explicit one-time conversion of an exact,
valid schema-v2 cache into the raw layer without permitting external generation; strict mode itself
never writes or migrates. Execution controls are recorded in workflow provenance but excluded from
raw scientific identity.

The target-only FASTA has a different candidate population and therefore uses a distinct candidate
pool and raw prediction cache.

### Explicit target-only calibration

The workflow supports three target-only policies:

- `refit_with_locked_window` (default): keep the entrapment-selected window and re-estimate nuisance
  parameters in the target-only candidate space;
- `reuse_dataset_artifact`: reuse the complete fitted same-dataset entrapment artifact without
  refitting; and
- `compare_both`: produce both interpretations with separate results and provenance.

No policy may use target-only outcomes to retune the window. Cross-dataset artifact reuse is
prohibited by default.

### Verified resumption

Completed checkpoints hash both `results.sage.tsv` and the resolved search configuration. Resume
also verifies dataset identity, model and stage identity, target-only policy, candidate-pool schema
and payload, and MS2Rescore cache integrity. A checkpoint left `running`, a modified output, or an
incompatible cache is rebuilt rather than silently accepted.

## Validation and model status

The engineering refactor and its required parity scope are complete. Frozen ISB model-by-model
comparisons and the independent PXD001468 Moments comparison were completed. PXD contained 53
legacy trace rows, 47 valid comparable windows, and selected the same `10-10` window; optimized
counts were exact, while MS2Rescore and target-only counts remained within the predeclared 0.5%
platform tolerance. PXD Moments was the only required PXD model for this refactor.

This is engineering parity, not evidence that Decoy-Free should replace TDC as the statistical
default. The current release evidence is `not_evaluable` for that decision because a matched TDC
benchmark is absent. In addition, both frozen legacy and native PXD post-MS2Rescore protein FDP
slightly exceeded 1%; this does not invalidate parity, but it prevents a broader calibration claim.

| Model | Current workflow status |
|---|---|
| Moments | Selectable Ensemble voter after dataset-local fitting and technical validation. |
| MLE | Selectable Ensemble voter after dataset-local fitting and technical validation. |
| MSFDR1-SMIX | Selectable rank-1-only Ensemble voter; its window remains fixed at 1-1. |
| Lower Order | Selectable Ensemble voter after dataset-local optimization. Target-only supports `refit_with_locked_window` only; cross-space `reuse_dataset_artifact` is unsupported. |
| MSFDR | Repaired, selectable Ensemble voter after dataset-local optimization and technical validation. |
| MSFDR2-SMIX | Repaired, selectable Ensemble voter after dataset-local optimization and technical validation. |
| Nokoi | Selectable Ensemble voter after deterministic dataset-local fitting and portable-artifact v2 technical validation. |
| Ensemble | Optional; JSON selects the requested voters, technical fail-closed checks select the actual roster, and statistical validation measurements remain nonblocking diagnostics. |

Release evaluation has three states:

- `eligible`: required evidence is evaluable and every criterion passes;
- `not_eligible`: evidence is complete, but one or more criteria fail; and
- `not_evaluable`: required evidence is missing, invalid, unreadable, or cannot be linked to its
  declared calibration source.

Missing or invalid evidence is never reported as stable or passed. See
[`DECOY_FREE_WORKFLOW.md`](DECOY_FREE_WORKFLOW.md),
[`validation/reports/phase9_release_finalization_2026-08-09.json`](validation/reports/phase9_release_finalization_2026-08-09.json),
and
[`validation/policies/current_ensemble_expert_policy_2026-08-15.json`](validation/policies/current_ensemble_expert_policy_2026-08-15.json)
for the detailed engineering record.


## Practical implication

Decoy-free mode should not be treated as a “set once and forget” option. It is a model-fitting workflow. The null window, model fit, Storey/BH q-value behavior, physical rescue settings, and hierarchical inference settings must be validated together. Higher target counts alone are not sufficient evidence of improvement. A configuration is only useful if it increases target identifications while maintaining acceptable entrapment behavior, especially at the protein level.

---
---
## Core concept: decoy-free evidence inference

Decoy-free mode treats the search results as a model-based evidence problem rather than a single target-decoy counting problem. Instead of assuming that physical reverse decoys provide a sufficiently dense empirical null in every dataset, DF models estimate false-match behavior from lower-ranked target-database candidates and/or fitted score distributions.

The branch can be run in either **single-model mode** or **ensemble mode**.

In **single-model mode**, `model_fit` selects one DF expert as the primary base evidence model. This is useful when you want to isolate or validate one statistical assumption at a time, tune a specific null-rank window, or avoid ensemble behavior during benchmarking. Supported single-model paths include:

- **Moments**: Gumbel null fit by method of moments.
- **MLE**: Gumbel null fit by maximum likelihood.
- **LowerOrder**: lower-order-statistics model inspired by lower-order decoy-free FDR estimation.
- **MSFDR seeded**: seeded mixture-model path.
- **MSFDR1_SMIX**: rank-1 semi-mixture path.
- **MSFDR2_SMIX**: pooled lower-rank semi-mixture path.
- **NOKOI**: cross-fitted L1-regularized discriminant model using target-like positives and lower-ranked null-like examples.

In **ensemble mode**, the branch can run a “jury” of up to seven enabled experts and combine their evidence streams. Each expert can produce p-value-like or PEP-like evidence. Ensemble mode combines enabled expert streams using either a p-value combiner or a PEP combiner, depending on `final_evidence_space`.

The ensemble is optional. It should not be interpreted as the only decoy-free workflow. For many validation runs, single-model mode is preferable because it makes model-specific behavior, calibration, and failure modes easier to interpret.

The branch also supports bounded physical evidence updates. RT and IMS evidence are not intended to overwrite the base score model. They are applied as capped auxiliary confidence shifts with anchor/reliability requirements, regardless of whether the base evidence came from a single model or from the ensemble.

## Important status notes

This branch is experimental. The DF code is designed to fail closed whenever the mandatory base model cannot produce a valid rank-1 stream. Optional post-base stages are nonfatal: if RT, IMS, peptide reproducibility rescue, or protein reproducibility rescue cannot produce a valid update, the previous last-good active stream is kept.

Direct one-off Ensemble searches honor only the static enable/disable flags and weights in the
search JSON; their code defaults enable all seven experts. They do not apply the workflow's current
secondary-model exclusions or workflow roster policy automatically. The native `sage workflow`
path adds fail-closed artifact/provenance validation and writes the requested and actual
dataset-local expert rosters. Holdout
datasets run the same predeclared optimization procedure; they do not import another dataset's
selected windows or fitted experts.

Direct one-off searches use configured null-rank windows. The native `sage workflow` path can
search bounded or explicit candidate windows in memory from one retained candidate set, select the
highest-yield feasible window, and lock it for later stages.

See [`DECOY_FREE_WORKFLOW.md`](DECOY_FREE_WORKFLOW.md) and [`workflow.example.json`](workflow.example.json) for the resumable development/holdout workflow and validation-only audits of completed result tables.

The current implementation exposes `msfdr2_smix_min_null_rank` and `msfdr2_smix_max_null_rank`, but there are no `msfdr1_smix_min_null_rank` or `msfdr1_smix_max_null_rank` options in `FdrOptions`. MSFDR1_SMIX is a rank-1 semi-mixture path controlled by initialization fractions and pi clamps. Do **not** put `msfdr1_smix_min_null_rank` or `msfdr1_smix_max_null_rank` in the JSON config; they are not accepted by the current code.

For `model_fit = "ensemble"`, `final_evidence_space` must be explicitly set to either `"p_value"` or `"pep"`. The code rejects `final_evidence_space = "auto"` in ensemble mode.

---

## Installation and basic usage

This fork is built from source with Rust. Python is not required for native Sage searching,
entrapment generation, null-window optimization, validation reporting, or cache resumption. A
compatible Python environment is required only when generating MS2Rescore/MS2PIP/DeepLC
annotations or when using optional analysis and plotting tools.

Install Rust using [rustup](https://rustup.rs/) and then clone and build the fork:

```bash
git clone https://github.com/jltovar/sage_decoy-free sage_decoy_free_github
cd sage_decoy_free_github
git switch decoy-free
cargo build --release
./target/release/sage --help
```

The executable is written to:

```bash
./target/release/sage
```

The repository's declared minimum supported Rust version is exercised by GitHub Actions. Use the
toolchain declared by the repository rather than assuming an older system Rust installation is
sufficient.

---

## Running Sage

For a new dataset, use the native workflow rather than a direct one-off Decoy-Free search. Start
from [`workflow.example.json`](workflow.example.json) and provide:

- a normal Sage search JSON;
- the target FASTA;
- the spectra;
- one or more candidate foreign-species FASTAs;
- an output root;
- each model's compact adaptive/exhaustive rank bounds (or an explicit frozen replay list) and
  MS2Rescore policy; and
- any frozen parity or matched-TDC evidence required by the declared validation scope.

A normal new-dataset model declaration is short:

```json
{
  "model": "moments",
  "window_optimizer": {
    "strategy": "landscape_adaptive",
    "min_rank_range": [2, 10],
    "max_rank_range": [2, 25]
  },
  "ms2rescore": "always"
}
```

The ranges are inclusive. Sage retains candidates through at least the largest allowed `max_rank`
(and farther when a later MS2Rescore stage requires it), chooses a path from the observed
entrapment behavior, and records every window it actually visits. A frontier or interior result is
a deterministic heuristic; an irregular fallback and an explicit `exhaustive` run are exact over
the bounded universe. Use `exhaustive` directly when proof of the bounded global optimum is
required regardless of the observed landscape.
Use `candidate_windows` only when the exact ordered list is itself part of the experiment.

Validate and materialize the plan without searching:

```bash
./target/release/sage workflow workflow.json --plan-only
```

Run or resume the workflow:

```bash
RUST_BACKTRACE=1 SAGE_LOG=info \
  ./target/release/sage workflow workflow.json
```

Re-running the same command resumes only stages whose complete identity and durable outputs still
pass verification. If interruption occurs during null-window fitting, the per-trial optimizer
checkpoint is also reused after its candidate-population and analysis fingerprint is verified.

### Workflow outputs

The output root contains the generated FASTA and provenance, immutable candidate pools, layered
raw MS2Rescore prediction caches, per-model stage calibrations, fitted artifacts, and validation
reports. Important top-level reports include:

```text
workflow.dataset.json
workflow.manifest.resolved.json
workflow.candidate_pools.json
workflow.ms2rescore_annotations.json
workflow.state.json
validation.audit.json
validation.missing_runs.json
validation.invalid_runs.json
validation.stage_comparisons.json
validation.transfer_stability.json
validation.parity.json
validation.tdc_benchmarks.json
validation.release_gate.json
```

Each optimized model stage also contains `null_window_evaluations.json`,
`null_window_optimization.json`, and `null_window_optimizer.checkpoint.json`.

### Entrapment-only audit

Use the native audit when you want to inspect or compare FASTA construction without loading
spectra:

```bash
./target/release/sage audit-entrapment entrapment.audit.json
```

This is the current replacement for the old external `make_entrapment.sh` workflow. A legacy
FDRBench reference is optional and should be supplied only when performing an engineering-parity
audit.

### Direct search mode

The binary still accepts an ordinary Sage search JSON directly:

```bash
RUST_BACKTRACE=1 SAGE_LOG=info \
  ./target/release/sage search.json
```

Direct TDC searches use `fdr.mode = "tdc"` (the default). Direct Decoy-Free searches use
`fdr.mode = "decoy_free"` and a `model_fit` value of:

```text
moments
mle
lower_order
msfdr
msfdr1_smix
msfdr2_smix
nokoi
ensemble
```

A direct Decoy-Free search uses the windows and artifacts already present in its search JSON. It
does not generate an entrapment FASTA, optimize windows, enforce the workflow's dataset-local expert
policy, or create a release gate. Direct mode is therefore appropriate for replaying a previously
validated configuration or for diagnostics, not for calibrating a new dataset.


## Direct Decoy-Free configuration example

The native workflow is the recommended starting point for a new dataset. The following `fdr`
block only demonstrates the syntax for a direct single-model replay after a Moments window has
already been validated for the same dataset. The `9-18` window shown here is the frozen ISB parity
value and must not be transferred to another dataset.

```json
{
  "fdr": {
    "mode": "decoy_free",
    "model_fit": "moments",
    "moments_min_null_rank": 9,
    "moments_max_null_rank": 18,
    "psm_q_method": "storey",
    "peptide_q_method": "storey",
    "protein_q_method": "storey",
    "precursor_fdr": 0.01,
    "peptide_fdr": 0.01,
    "protein_fdr": 0.01
  }
}
```

Decoy-free protein grouping is opt-in. To evaluate IDPicker-style parsimonious
groups while retaining decoy-free protein calibration, add:

```json
"decoy_free_protein_grouping": true
```

This changes the protein hypothesis from a single raw accession to a single
inferred group. Indistinguishable accessions are reported with `/` inside one
group; peptides mapping to multiple groups remain excluded from protein-level
evidence. The setting does not run picked target-decoy group FDR. Leave it off
(the default) when reproducing previously validated decoy-free results.

---

## Advanced direct-search configuration surface

The following block illustrates the broad direct-search configuration surface. It is not a
workflow manifest, a recommended production configuration, or a validated parameter set for a new
dataset. Direct Ensemble mode uses static flags and does not construct a provenance-bearing
workflow roster. Use [`workflow.example.json`](workflow.example.json) for independently optimized
dataset-local artifacts, technical fail-closed validation, and requested/actual roster provenance.

The example requests Nokoi as a normal voter. Its deterministic portable v2 artifact must pass the
same technical fail-closed checks as every other requested voter; statistical validation
diagnostics do not select Ensemble voters.

```json
{
  "fdr": {
    "mode": "decoy_free",
    "entrapment_report": "auto",

    "model_fit": "ensemble",
    "final_evidence_space": "p_value",

    "peptide_p_combine": "cauchy",
    "protein_p_combine": "cauchy",

    "psm_q_method": "storey",
    "peptide_q_method": "storey",
    "protein_q_method": "storey",

    "precursor_fdr": 0.01,
    "peptide_fdr": 0.01,
    "protein_fdr": 0.01,

    "report_psms_by_peptide_q": false,

    "min_null_rank": 1,
    "max_null_rank": 50,
    "min_null_size": 100,
    "min_rank_count": 10,

    "enable_rt_confidence_adjustment": true,
    "enable_ims_confidence_adjustment": false,
    "enable_peptide_reproducibility_rescue": true,
    "enable_protein_reproducibility_rescue": true,

    "min_storey_n": 100,
    "storey_pi0_clamp_min": 0.5,
    "storey_pi0_clamp_max": 1.0,
    "storey_lambda_min": 0.05,
    "storey_lambda_max": 0.95,
    "storey_lambda_step": 0.05,
    "storey_lambda_min_for_agg": 0.5,
    "storey_pi0_agg": "median",
    "storey_degen_same_as_median_frac": 0.9,
    "storey_degen_eps": 1e-6,
    "storey_degen_pi0_eps": 0.001,
    "storey_degen_fallback": "bh",

    "moments_min_null_rank": 7,
    "moments_max_null_rank": 24,
    "moments_purification_factor": 0.25,
    "moments_robust_fit": true,
    "moments_winsor_lower_q": 0.01,
    "moments_winsor_upper_q": 0.90,

    "mle_min_null_rank": 10,
    "mle_max_null_rank": 25,
    "mle_purification_factor": 0.25,
    "mle_robust_fit": true,
    "mle_winsor_lower_q": 0.01,
    "mle_winsor_upper_q": 0.90,

    "lower_order_min_null_rank": 6,
    "lower_order_max_null_rank": 7,
    "lower_order_purification_factor": 0.25,
    "lo_min_count_per_rank": 10,
    "lo_stratify": "global",
    "lo_evalue_candidate_count_power": 1.0,
    "lo_evalue_scale": 1.0,
    "lo_tev_transform": "neg_log_e",
    "lo_tnm_extrapolation_strength": 1.85,

    "msfdr_min_null_rank": 10,
    "msfdr_max_null_rank": 20,
    "msfdr_seeded_purification_factor": 0.25,
    "msfdr_seeded_top_frac_init": 0.2,
    "msfdr_multistart": 3,
    "msfdr_pi_clamp_min": 0.01,
    "msfdr_pi_clamp_max": 0.99,

    "mix_em_max_iter": 200,
    "mix_em_tol": 1e-6,

    "msfdr1_bottom_frac_init": 0.5,
    "msfdr1_top_frac_init": 0.2,
    "msfdr1_pi_clamp_min": 0.01,
    "msfdr1_pi_clamp_max": 0.99,

    "msfdr2_smix_min_null_rank": 9,
    "msfdr2_smix_max_null_rank": 17,
    "msfdr2_bottom_frac_init": 0.5,
    "msfdr2_top_frac_init": 0.2,
    "msfdr2_pi_clamp_min": 0.01,
    "msfdr2_pi_clamp_max": 0.99,

    "nokoi_min_null_rank": 3,
    "nokoi_max_null_rank": 4,
    "nokoi_null_purification_factor": 0.25,
    "nokoi_positive_top_fraction": 0.10,
    "nokoi_k_folds": 5,
    "nokoi_l1_lambda_min": 0.2,
    "nokoi_l1_lambda_max": 1.0,
    "nokoi_l1_lambda_steps": 10,

    "enable_moments": true,
    "enable_mle": true,
    "enable_lower_order": true,
    "enable_msfdr_seeded": true,
    "enable_msfdr_1smix": true,
    "enable_msfdr_2smix": true,
    "enable_nokoi": true,

    "ensemble_p_combiner": "second_best",
    "ensemble_cauchy_penalty": 1.0224,
    "ensemble_pep_combiner": "median",
    "ensemble_pep_trim_frac": 0.2,
    "ensemble_pep_quantile": 0.5,
    "ensemble_pep_top_k": 5,
    "ensemble_pep_logit_eps": 1e-6,

    "ensemble_weight_moments": 1.0,
    "ensemble_weight_mle": 1.0,
    "ensemble_weight_lower_order": 1.0,
    "ensemble_weight_msfdr_seeded": 1.0,
    "ensemble_weight_msfdr_1smix": 1.0,
    "ensemble_weight_msfdr_2smix": 1.0,
    "ensemble_weight_nokoi": 1.0,

    "physical_rescue": {
      "rt_mode": "bounded_aux",
      "ims_mode": "bounded_aux",
      "anchor_mode": "default",
      "anchor_max_pep": 0.01,
      "anchor_max_q": 0.01,
      "min_anchor_count_per_run": 10,
      "min_anchor_count_per_charge": 5,
      "joint_mode": "independent",
      "reliability_floor": 0.5,
      "missing_penalty": 0.0,
      "rt_region_bins": 10,
      "use_local_rt_scale": false,
      "cov_shrinkage": 0.1,
      "dart_cfg": {
        "dart_use_bootstrap": true,
        "dart_bootstrap_method": "parametric_mixture",
        "dart_mu_estimation": "median",
        "dart_bootstrap_iters": 100,
        "dart_leave_one_run_out": false,
        "dart_null_rt_model": "uniform",
        "dart_true_rt_model": "laplace",
        "dart_recalc_q_from_posterior": true
      },
      "bounded_cfg": {
        "update_space": "logit_confidence",
        "max_rescue_shift": 0.5,
        "max_penalty_shift": 0.5
      }
    },

    "reproducibility": {
      "enabled": true,
      "max_total_shift": 0.5,
      "max_agreement_shift": 0.5,
      "use_expert_agreement": true,
      "use_cross_run_recurrence": true,
      "max_recurrence_shift": 0.25,
      "redundancy_discount": 0.8,

      "peptide_eligibility": {
        "min_run_fraction": 0.0,
        "min_run_count": 2,
        "strong_reference_q_threshold_physical": 0.05,
        "strong_reference_pep_threshold_physical": 0.05,
        "min_strong_run_fraction": 0.0,
        "min_strong_run_count": 1
      },

      "protein_eligibility": {
        "enabled": true,
        "q_threshold_physical": 0.01,
        "min_unique_passing_peptides": 2,
        "min_unique_passing_fraction": 0.4
      },

      "anchor": {
        "mode": "second_best",
        "trim_fraction": 0.1
      },

      "rescue_band": {
        "rescue_mode": "bounded_shrinkage",
        "strong_cutoff_pep": 0.01,
        "weak_cutoff_pep": 0.25,
        "max_rescue_fraction": 0.5
      }
    },

    "hierarchical_inference": {
      "enabled": true,
      "entrapment_validation": true,
      "mode": "protein_anchored"
    }
  }
}
```

---

## Configuration reference

### Global DF controls

| Key | Accepted values / type | Current default | Notes |
|---|---:|---:|---|
| `mode` | `tdc`, `decoy_free` | `tdc` | Selects conventional TDC or Decoy-Free mode. |
| `entrapment_report` | `off`, `auto`, `on` | `auto` | Controls entrapment reporting behavior. |
| `model_fit` | `moments`, `mle`, `lower_order`, `msfdr`, `msfdr1_smix`, `msfdr2_smix`, `nokoi`, `ensemble` | `ensemble` in Decoy-Free mode | Selects the base Decoy-Free expert or Ensemble. |
| `final_evidence_space` | `auto`, `p_value`, `pep` | `auto` | Ensemble requires explicit `p_value` or `pep`. |
| `peptide_p_combine` | `fisher`, `cauchy`, `acat`, `sidak_min_p`, `bonferroni_min_p`, `tippett`, `best`, `second_best`, `hmp`, `brown`, `mudholkar_george`, `edgington`, `t_fisher`, `g_fisher`, `ihw`, `exchangeable_e_value`, `vovk_wang_generalized_mean`, `ordmeta_w_fisher`, `mcm`, `cmc` | `cauchy` | Peptide-level p-value combination. Some advanced modes require or use the separate calibration controls described in code. |
| `protein_p_combine` | Same values as `peptide_p_combine` | `cauchy` | Protein-level p-value combination. |
| `psm_q_method` | `auto`, `bh`, `storey`, `by`, `bky`, `sfdr`, `covariate_weighted_bh`, `cummean` | `storey` | PSM-level q-value method. `cummean` is only valid for PEP-native evidence. |
| `peptide_q_method` | Same values as `psm_q_method` | `auto` | Peptide-level q-value method. |
| `protein_q_method` | Same values as `psm_q_method` | `auto` | Protein-level q-value method. |
| `precursor_fdr` | float | `0.01` | PSM/precursor threshold. |
| `peptide_fdr` | float | `0.01` | Peptide threshold. |
| `protein_fdr` | float | `0.01` | Protein threshold. |
| `decoy_free_protein_grouping` | bool | `false` | Opt-in parsimonious protein hypotheses; q-values remain Decoy-Free, not picked TDC. |
| `report_psms_by_peptide_q` | bool | `false` | Reporting-only. Does not change native p/PEP streams. |

### Rank-null pool controls

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `min_null_rank` | integer | `2` | Global lower bound for null-rank pool. |
| `max_null_rank` | integer | `50` | Global upper bound for null-rank pool. |
| `min_null_size` | integer | `300` | Minimum null pool size target. |
| `min_rank_count` | integer | `10` | Minimum support per rank. |

Method-specific null windows are resolved inside the global `min_null_rank..=max_null_rank` range.  If a method-specific minimum and maximum are reversed, the code swaps them.

### Stage gates

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `enable_rt_confidence_adjustment` | bool | `false` | Enables RT post-base update. |
| `enable_ims_confidence_adjustment` | bool | `false` | Enables IMS post-base update. |
| `enable_peptide_reproducibility_rescue` | bool | `reproducibility.enabled` | Enables peptide recurrence/expert-agreement rescue. |
| `enable_protein_reproducibility_rescue` | bool | `reproducibility.enabled` | Enables protein-backed recurrence rescue. |

### Storey q-value controls

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `min_storey_n` | integer | `300` | Safety threshold for Storey calculations. |
| `storey_pi0_clamp_min` | float | `0.50` | Lower clamp for pi0. |
| `storey_pi0_clamp_max` | float | `1.00` | Upper clamp for pi0. |
| `storey_lambda_min` | float | `0.05` | Lambda grid minimum. |
| `storey_lambda_max` | float | `0.95` | Lambda grid maximum. |
| `storey_lambda_step` | float | `0.05` | Lambda grid step. |
| `storey_lambda_min_for_agg` | float | `0.50` | Lower lambda bound for aggregating pi0. |
| `storey_pi0_agg` | `median`, `trimmed_mean` | `median` | Aggregator over pi0(lambda). |
| `storey_degen_same_as_median_frac` | float | `0.90` | Degenerate shelf detector. |
| `storey_degen_eps` | float | `1e-6` | q-value equality tolerance. |
| `storey_degen_pi0_eps` | float | `1e-3` | pi0 degeneracy tolerance. |
| `storey_degen_fallback` | `bh`, `none` | `bh` | Fallback when Storey degeneracy is detected. |

### Moments model

The Moments model fits a Gumbel null using method-of-moments from lower-ranked null evidence.

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `moments_min_null_rank` | integer | `4` | Method-specific lower null rank. |
| `moments_max_null_rank` | integer | `50` | Method-specific upper null rank. |
| `moments_purification_factor` | float | `0.25` | Null-pool purification factor, clamped to `[0, 0.9]`. |
| `moments_robust_fit` | bool | `false` | Winsorizes null scores before moments fitting. |
| `moments_winsor_lower_q` | float | `0.01` | Lower winsor quantile. |
| `moments_winsor_upper_q` | float | `0.95` | Upper winsor quantile. |

### MLE model

The MLE model fits a Gumbel null by maximum likelihood.

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `mle_min_null_rank` | integer | `4` | Method-specific lower null rank. |
| `mle_max_null_rank` | integer | `50` | Method-specific upper null rank. |
| `mle_purification_factor` | float | `0.25` | Null-pool purification factor, clamped to `[0, 0.9]`. |
| `mle_robust_fit` | bool | `false` | Winsorizes before MLE; useful only as a sensitivity/contamination-control mode. |
| `mle_winsor_lower_q` | float | `0.01` | Lower winsor quantile. |
| `mle_winsor_upper_q` | float | `0.975` | Upper winsor quantile. |

### LowerOrder model

The LowerOrder model uses non-rank-1 lower-order evidence to infer a rank-1 target-null model (TNM).  Rank 1 is excluded from lower-order null fitting because it is target contaminated.  The implementation requires at least two usable lower-order ranks; invalid windows fail closed during fitting.

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `lower_order_min_null_rank` | integer | `6` | Method-specific lower null rank. |
| `lower_order_max_null_rank` | integer | `12` | Method-specific upper null rank. |
| `lower_order_purification_factor` | float | `0.15` | Null-pool purification factor, clamped to `[0, 0.9]`. |
| `lo_min_count_per_rank` | integer | `10` | Minimum support per LO rank. |
| `lo_stratify` | `global`, `charge` | `charge` | LO stratification mode. |
| `lo_evalue_candidate_count_power` | float | `0.75` | Candidate-count power for LO E-value construction, clamped to `[0, 1]`. |
| `lo_evalue_scale` | float | `1.0` | Compatibility field for a fitted-location reparameterization. Keep at canonical `1.0`; it is not eligible for yield optimization. |
| `lo_tev_transform` | `neg_log_e`, `log1000_over_e`, `scaled_log1000_over_e` | `neg_log_e` | Positive-affine TEV representation. Canonicalize to `neg_log_e`; legacy spellings with an underscore before `1000` remain accepted as loading aliases. |
| `lo_tnm_extrapolation_strength` | float | `1.0` | Rank-1 TNM extrapolation strength, clamped to `[0.25, 5.0]`. |

LO E-values are constructed as:

```text
E_LO = lo_spectrum_tail_p
     * lo_spectrum_candidate_count.powf(lo_evalue_candidate_count_power)
     * lo_evalue_scale
```

Then the configured TEV transform is applied:

```text
neg_log_e              => TEV = -ln(E_LO)
log1000_over_e        => TEV = ln(1000 / E_LO)
scaled_log1000_over_e => TEV = 0.02 * ln(1000 / E_LO)
```

### MSFDR seeded model

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `msfdr_min_null_rank` | integer | `4` | Method-specific lower null rank. |
| `msfdr_max_null_rank` | integer | `50` | Method-specific upper null rank. |
| `msfdr_seeded_purification_factor` | float | `0.25` | Null-pool purification factor. |
| `msfdr_seeded_top_frac_init` | float | `0.20` | Initial top fraction, clamped by the generic fraction helper. |
| `msfdr_multistart` | integer | `3` | Compatibility field. The current seeded MSFDR fitter uses one deterministic initialization, so this field is not eligible for yield optimization. |
| `msfdr_pi_clamp_min` | float | `0.01` | Minimum mixture weight clamp. |
| `msfdr_pi_clamp_max` | float | `0.565` | Maximum mixture weight clamp. |

### Shared mixture controls for MSFDR1_SMIX and MSFDR2_SMIX

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `mix_em_max_iter` | integer | `200` | EM iteration cap, clamped to `[1, 10000]`. |
| `mix_em_tol` | float | `1e-6` | EM convergence tolerance; must be positive. |

### MSFDR1_SMIX

MSFDR1_SMIX is a rank-1 mixture path.  The current code does not expose `msfdr1_smix_min_null_rank` or `msfdr1_smix_max_null_rank`.

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `msfdr1_bottom_frac_init` | float | `0.50` | Initial bottom fraction. |
| `msfdr1_top_frac_init` | float | `0.20` | Initial top fraction. |
| `msfdr1_pi_clamp_min` | float | `0.01` | Minimum mixture weight clamp. |
| `msfdr1_pi_clamp_max` | float | `0.65` | Maximum mixture weight clamp. |

### MSFDR2_SMIX

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `msfdr2_smix_min_null_rank` | integer | `4` | Lower pooled-rank null bound. |
| `msfdr2_smix_max_null_rank` | integer | `50` | Upper pooled-rank null bound. |
| `msfdr2_bottom_frac_init` | float | `msfdr1_bottom_frac_init` | Initial bottom fraction. |
| `msfdr2_top_frac_init` | float | `msfdr1_top_frac_init` | Initial top fraction. |
| `msfdr2_pi_clamp_min` | float | `0.01` | Minimum mixture weight clamp. |
| `msfdr2_pi_clamp_max` | float | `0.568` | Maximum mixture weight clamp. |

### NOKOI model

NOKOI uses lower-ranked null evidence and a rank-1 positive training class.  The positive training class is controlled by `nokoi_positive_top_fraction` and internal provisional evidence.

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `nokoi_min_null_rank` | integer | `2` | Method-specific lower null rank. |
| `nokoi_max_null_rank` | integer | `7` | Method-specific upper null rank. |
| `nokoi_null_purification_factor` | float | `0.20` | Null-pool purification factor. |
| `nokoi_positive_top_fraction` | float | `0.10` | High-scoring rank-1 fraction for positive class construction. |
| `nokoi_k_folds` | integer | `2` | Cross-fit folds, clamped to `[2, 20]`. |
| `nokoi_l1_lambda_min` | float | `1e-4` | Minimum L1 lambda. |
| `nokoi_l1_lambda_max` | float | `1e-1` | Maximum L1 lambda. |
| `nokoi_l1_lambda_steps` | integer | `10` | Lambda grid steps, clamped to `[1, 100]`. |

### Ensemble controls

Direct Ensemble mode combines the expert streams enabled in the search JSON. Explicit single-model
modes ignore these enable flags because the selected model is mandatory and fails closed if it
cannot produce a valid stream. The defaults below describe direct-search configuration only; they
do not override `sage workflow` participation policy. The workflow records the requested JSON
roster separately from the actual technically valid roster. Artifact/provenance integrity,
supported target-only state, nonfallback finite fitted state, and duplicate-vote checks fail
closed; parity, calibration, transfer, overlap, holdout, and yield measurements are nonblocking
diagnostics and do not select voters.

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `enable_moments` | bool | `true` | Include Moments expert in ensemble. |
| `enable_mle` | bool | `true` | Include MLE expert in ensemble. |
| `enable_lower_order` | bool | `true` | Include LowerOrder expert in ensemble. |
| `enable_msfdr_seeded` | bool | `true` | Include seeded MSFDR expert in ensemble. |
| `enable_msfdr_1smix` | bool | `true` | Include MSFDR1_SMIX expert in ensemble. |
| `enable_msfdr_2smix` | bool | `true` | Include MSFDR2_SMIX expert in ensemble. |
| `enable_nokoi` | bool | `true` | Include NOKOI expert in ensemble. |
| `ensemble_p_combiner` | `fisher`, `cauchy`, `sidak_min_p`, `best`, `second_best` | `cauchy` | p-value combiner. |
| `ensemble_pep_combiner` | `median`, `trimmed_mean`, `max`, `mean`, `weighted_mean`, `weighted_median`, `winsorized_mean`, `quantile`, `top_k_mean`, `geometric_mean`, `logit_mean` | `median` | PEP-like combiner. |
| `ensemble_cauchy_penalty` | float | `1.0` | Multiplies Cauchy-combined p-values; clamped to `[1, 100]`. |
| `ensemble_pep_trim_frac` | float | `0.20` | Trim fraction, clamped to `[0, 0.49]`. |
| `ensemble_pep_quantile` | float | `0.50` | Quantile for quantile combiner. |
| `ensemble_pep_top_k` | integer | `2` | Top-k count for `top_k_mean`. |
| `ensemble_pep_logit_eps` | float | `1e-6` | Epsilon for logit combiner. |
| `ensemble_weight_*` | float | `1.0` | Static nonnegative expert weights for weighted combiners. |

The parameter optimizer applies these controls conditionally. P-value combiner and Cauchy-penalty
trials require a p-value final stream. PEP combiner and shape trials require a PEP final stream,
and expert-weight trials additionally require `weighted_mean` or `weighted_median`. Dormant
settings remain at positive canonical defaults, so they cannot masquerade as optimized values or
silently remove a selected voter. A median PEP produced beside a p-value final stream remains an
auxiliary stored consensus statistic and does not drive the final p-value decisions.

### Physical rescue / auxiliary evidence

Physical rescue can use RT and/or IMS evidence.  Current accepted modes are:

```text
off
dart_bayes
bounded_aux
```

Current accepted physical anchor modes are:

```text
strict
default
relaxed
evidence_only
```

Current accepted joint modes are:

```text
min
product
independent
```

The currently supported bounded update space is:

```text
logit_confidence
```

The `physical_rescue` block supports:

| Key | Type | Notes |
|---|---:|---|
| `rt_mode` | `off`, `dart_bayes`, `bounded_aux` | RT adjustment mode. |
| `ims_mode` | `off`, `dart_bayes`, `bounded_aux` | IMS adjustment mode. |
| `anchor_mode` | `strict`, `default`, `relaxed`, `evidence_only` | Anchor selection mode. |
| `anchor_max_pep` | float | Maximum anchor PEP. |
| `anchor_max_q` | float | Maximum anchor q-value. |
| `min_anchor_count_per_run` | integer | Minimum RT anchors per run. |
| `min_anchor_count_per_charge` | integer | Minimum IMS anchors per charge. |
| `joint_mode` | `min`, `product`, `independent` | How RT/IMS evidence is combined. |
| `reliability_floor` | float | Minimum reliability retained in bounded evidence. |
| `missing_penalty` | float | Penalty for missing auxiliary evidence. |
| `rt_region_bins` | integer | Local RT region bins. |
| `use_local_rt_scale` | bool | Whether local RT scale is used. |
| `cov_shrinkage` | float | Covariance shrinkage. |
| `dart_cfg` | object or null | DART-like model options. |
| `bounded_cfg` | object or null | Bounded logit-shift options. |

### DART-like RT/IMS options

| Key | Accepted values / type | Notes |
|---|---:|---|
| `dart_use_bootstrap` | bool | Enables bootstrap support. |
| `dart_bootstrap_method` | `none`, `parametric`, `parametric_mixture`, `non_parametric` | Bootstrap method. |
| `dart_mu_estimation` | `mean`, `median`, `weighted_mean` | Center estimator. |
| `dart_bootstrap_iters` | integer | Bootstrap iterations. |
| `dart_leave_one_run_out` | bool | Leave-one-run-out mode. |
| `dart_null_rt_model` | `normal`, `uniform` | Null RT model. |
| `dart_true_rt_model` | `normal`, `laplace` | True RT model. |
| `dart_recalc_q_from_posterior` | bool | Recalculate q from posterior. |

### Reproducibility rescue

Reproducibility rescue can use expert agreement, cross-run recurrence, and protein/peptide eligibility.  It is bounded by logit-space shifts and does not have unlimited rescue authority.

| Key | Type | Notes |
|---|---:|---|
| `enabled` | bool | Global reproducibility enable. |
| `max_total_shift` | float | Maximum total reproducibility shift. |
| `max_agreement_shift` | float | Maximum expert-agreement shift. |
| `max_recurrence_shift` | float | Maximum cross-run recurrence shift. |
| `use_expert_agreement` | bool | Enables expert-agreement support. |
| `use_cross_run_recurrence` | bool | Enables cross-run recurrence support. |
| `redundancy_discount` | float | Discount for redundant recurrence evidence. |
| `protein_eligibility` | object | Protein-level gating. |
| `peptide_eligibility` | object | Peptide-level gating. |
| `anchor` | object | Anchor construction. |
| `rescue_band` | object | Strong/weak PEP rescue band. |

Accepted reproducibility anchor modes are:

```text
best
second_best
mean
median
trimmed_mean
```

Accepted rescue modes are:

```text
replace
bounded_shrinkage
```

### Hierarchical inference / Level 4 reporting

The JSON-facing block is `hierarchical_inference`:

```json
"hierarchical_inference": {
  "enabled": true,
  "entrapment_validation": true,
  "mode": "protein_anchored"
}
```

Accepted JSON modes are:

```text
off
protein_anchored
```

When `enabled = true` and `mode = "protein_anchored"`, the code maps this to internal `HierarchicalReportingMode::Strict`.  This produces protein-supported reporting flags but must not overwrite the active final fields:

```text
decoy_free_p_value
decoy_free_pep
decoy_free_score
decoy_free_q_value
decoy_free_peptide_q
decoy_free_protein_q
```

The reporting flags are:

```text
decoy_free_protein_supported_peptide
decoy_free_peptide_supported_psm
```

These flags are reporting-only.  They mean that a rank-1 target PSM/peptide supports an accepted protein under the configured hierarchical mode.  They are not independent PSM-level or peptide-level FDR claims.

---

## Statistical rationale in ultra-low-input data

Conventional TDA/TDC is highly effective when the dataset is large enough that decoy counts provide a stable empirical null. In ultra-low-input data, the number of confidently observed peptides and proteins can be small. At protein level, this creates coarse empirical FDR resolution: one observed decoy protein can dominate the attainable q-value floor, while no observed decoys can create large tied blocks of similar q-values.

The DF branch is designed to address this sparse-count regime by replacing pure decoy counting with continuous evidence models. It estimates false-match behavior from score distributions, lower-ranked candidate behavior, physical evidence such as RT/IMS, and optional reproducibility/protein hierarchy. The goal is not to declare TDA invalid. The goal is to provide an auditable alternative when TDA/TDC becomes count-starved, especially for protein-level reporting in ultra-low-input experiments.

Important limitations remain:

- DF model calibration is model- and dataset-dependent.
- Higher discovery power must be checked against entrapment or other external validation.
- Physical evidence should be bounded and gated; RT/IMS/mass-error agreement should not rescue arbitrary low-quality PSMs without base evidence support.
- Direct Ensemble mode uses static weights; the native validation workflow adds automatic inclusion gates but does not learn weights from holdout data.


## Output field semantics

### Final active DF fields

The final active DF answer is represented by:

```text
decoy_free_p_value
decoy_free_pep
decoy_free_score
decoy_free_q_value
decoy_free_peptide_q
decoy_free_protein_q
```

Only these fields represent the selected last-good active DF stream.  They are the fields to use for final filtering and downstream interpretation.

The DF TSV also reports `protein_groups` and `num_protein_groups`. When
`decoy_free_protein_grouping` is enabled, `decoy_free_protein_q` is the q-value
of the single inferred `protein_groups` hypothesis on that row. Rows mapping to
more than one inferred group receive a protein q-value of 1.0. When grouping is
disabled, these columns mirror raw protein assignments and preserve the prior
single-accession rule.

### Stage snapshots

The code also writes stage snapshots when the corresponding stage runs:

```text
decoy_free_p_value_base
decoy_free_pep_base
decoy_free_score_base
decoy_free_q_base

decoy_free_p_value_rt
decoy_free_pep_rt
decoy_free_score_rt
decoy_free_q_rt

decoy_free_p_value_ims
decoy_free_pep_ims
decoy_free_score_ims
decoy_free_q_ims

decoy_free_p_value_peptide_rescue
decoy_free_pep_peptide_rescue
decoy_free_score_peptide_rescue
decoy_free_q_peptide_rescue

decoy_free_p_value_protein_rescue
decoy_free_pep_protein_rescue
decoy_free_score_protein_rescue
decoy_free_q_protein_rescue
```

These are audit fields.  They should not be used as the final active stream unless intentionally diagnosing a specific stage.

### Transitional internal fields

The current `DfFeature` structure still contains transitional internal fields:

```text
decoy_free_p_value_l2
decoy_free_pep_l2
decoy_free_score_l2
decoy_free_q_l2

decoy_free_pep_l3
decoy_free_score_l3
decoy_free_q_l3
```

They are marked as transitional in the code and should not be treated as final public reporting fields.

### Rank scope

DF PSM-level outputs are defined only for rank-1 PSMs.  For non-rank-1 rows, final, stage-specific, and method-specific DF fields are scrubbed to `None`.  This prevents stale values from lower-ranked candidates from leaking into TSV output, peptide inference, protein inference, or diagnostics.

---

## Evidence-space semantics

Base DF experts may be p-value-native or PEP-like/PEP-native.  The active evidence space determines how q-values are calculated and how downstream inference should interpret fields.

### p-value-native streams

For p-value-native active streams, q-values are computed from active p-values using the configured p-value FDR method, such as BH or Storey.  Peptide and protein inference may use configured p-value combiners such as Cauchy, Fisher, Sidak-min-p, best, or second-best.

### PEP-native streams

For PEP-native active streams, q-values are cumulative means of PEP-like values after best-first sorting, followed by monotonic correction.  PEP-native streams must not use BH or Storey directly unless first converted into a valid p-value stream.

### Ensemble streams

The ensemble combines expert p-value streams and/or expert PEP-like streams depending on `final_evidence_space`.  In the current code, ensemble mode is not allowed to infer the final evidence space automatically; `final_evidence_space` must be explicit.

---

## Model-specific interpretation

### Moments / MLE / LowerOrder

These models produce fitted-null tail p-value streams and calibrated local-FDR/PEP-like streams derived from those p-values.  They are useful for decoy-free modeling of lower-rank null evidence, but their calibration should be checked with entrapment or other external validation.

Lower Order is available for JSON-selected Ensemble voting after dataset-local window optimization
and technical validation. Its target-only contract is `refit_with_locked_window`: the selected
+entrapment window is retained, nuisance state is refitted in the target-only candidate space, and
target-only outcomes never retune the window. Complete-artifact cross-space reuse remains
unsupported and fails closed.

### MSFDR variants

The MSFDR family produces fitted or empirical null-survival p-like streams and derived PEP-like
streams. The SMIX variants use mixture-model fitting and can be sensitive to initialization and
rank-window choices. Seeded MSFDR and MSFDR2-SMIX passed individual +entrapment and target-only
parity in the reproduced WSL annotation environment and are available as JSON-selected Ensemble
voters. Their validation and holdout measurements remain reportable diagnostics rather than roster
controls.

### NOKOI

NOKOI uses a cross-fit classifier-like approach with lower-rank null evidence and high-scoring
rank-1 positives. Portable artifact v2 stores the complete feature, fold, model, sampling,
normalization, lambda-evaluation, convergence, empirical-null, pi0, Grenander, p-to-PEP,
candidate-count, and integrity state needed for scoring without retraining. Stable candidate
identities determine sampling and folds, numeric artifact state uses exact hexadecimal IEEE-754
bits, and absolute paths do not affect identity.
Wrong schemas, dimensions, windows, populations, provenance, nonfinite/nonmonotone calibration, or
hash mismatches fail closed. Nokoi is a selectable continuous PSM p-value/PEP Ensemble voter; it has
no statistical admission requirement beyond JSON selection and technical validity.

### Ensemble

The Ensemble combines continuous PSM-level p-values and PEP-like values from every actual voter
using the configured combiners. It then calculates Ensemble PSM q-values, derives peptide q-values
from that combined PSM stream, and derives protein q-values downstream. The native workflow
optimizes each constituent window independently within the current dataset; it never optimizes a
combined Ensemble window. JSON configuration defines the requested roster. Artifact integrity,
provenance, dataset/search/analysis identity, supported target-only semantics, fallback, fitted
external-profile provenance, finite fitted state, and duplicate-vote checks define the actual
roster. Schema-v9 locks additionally bind each expert's complete effective production
configuration and a separate final Ensemble configuration. A target-only refit therefore uses
the frozen expert-local scientific and calibration settings as well as its window; present-day
workflow defaults cannot replace locked values. Old locks without complete resolved expert
configurations cannot support target-only refit or `compare_both` and fail closed instead of
guessing missing values. Expert configurations, artifact identities, policy-specific streams, and
the final Ensemble settings remain distinct.

The locks preserve each expert's fitted-profile provenance separately from the one canonical
dataset-local Ensemble profile contract. The Ensemble profile is fitted once at the
explicit 9-18 window, independent of expert order; expert-specific profiles cannot overwrite it.
Statistical diagnostics do not remove a technically valid requested voter. Native and
MS2Rescore-fitted artifacts remain separate.

Optimizer manifests retain immutable all-expert root provenance. Production orchestration derives
typed stage-local views: one expert and one expected configuration hash for an individual stage,
then the complete all-expert map for final Ensemble evaluation. Shared scientific and cache
settings are not filtered, root configuration bytes are never mutated, and resolved hashes are
checked before fitting. Root/stage lineage hashes enter optimizer checkpoints without changing
search, candidate-pool, or raw-annotation identities.

Each distinct canonical model contributes one continuous PSM-level vote. Canonical ordering is
used only for deterministic serialization; models are not ordered by discovery counts. Duplicate
canonical models and duplicate artifact instances fail closed. `precursor_fdr`, `peptide_fdr`, and
`protein_fdr` control reported PSM/precursor, peptide, and protein identifications respectively;
they do not select voters.

The legacy workflow fields `minimum_incremental_ensemble_peptides` and
`maximum_transfer_fraction_loss` remain readable for compatibility and validation reporting, but
are deprecated as runtime admission controls. Parity pairs, entrapment observation minima, holdout
status, and release/default eligibility likewise do not alter the roster.

When an automatically eligible expert is outside the declared established interaction baseline,
the workflow also reports the baseline-to-final raw-q and Level-4 entrapment FDP changes for PSMs,
canonical peptides, and peptidoforms. An absolute raw-q deterioration above `0.01` produces a
structured informational warning. Raw-q and Level-4 interaction measurements are explicitly
nonblocking validation diagnostics: they neither change the roster nor suppress target-only
execution. The diagnostic is deterministic, provenance bearing, and does not use target-only
outcomes to decide participation.

---

## Interpretation of current validation evidence

The completed validation establishes engineering behavior, not general statistical superiority:

1. Frozen ISB model-by-model comparisons exercise the implementation on a development dataset.
2. The required independent PXD001468 Moments comparison reproduced the legacy selected window and
   stayed within its predeclared platform tolerances.
3. Both frozen legacy and native PXD post-MS2Rescore protein FDP slightly exceeded 1%, so parity
   must not be described as proof of calibrated 1% protein FDR.
4. A matched TDC benchmark is still absent from the final release evidence. The repository therefore
   cannot yet evaluate whether Decoy-Free should replace TDC as a statistical default.
5. Lower Order supports target-only `refit_with_locked_window` only. Lower Order, seeded MSFDR,
   MSFDR2-SMIX, and Nokoi are selectable voters subject to technical fail-closed validation.
   Statistical-default eligibility remains a separate, unevaluated question.
6. The present evidence does not select a universally best Decoy-Free model. Model suitability must
   be assessed within each dataset using entrapment calibration and matched comparisons.

---

## Known limitations and recommended next improvements

### 1. Matched TDC release benchmark

The highest-priority missing release evidence is a matched conventional TDC analysis using the same
dataset, FASTA/search assumptions, thresholds, validation layers, and PSM/peptide/peptidoform/protein
counting definitions. Until that comparison exists, `not_evaluable` is the correct statistical
release state.

### 2. Additional null-window diagnostics

The native workflow now searches compact bounds adaptively, exhaustively, or by exact explicit
replay without rereading spectra, then selects the highest-yield visited window satisfying PSM,
peptide, and protein entrapment-FDP constraints. Further diagnostics can still be added, such as:

```text
candidate count per rank
score stability
mass-error distribution broadness
RT-error broadness
depletion of target-like delta-mass peak
pi0 stability
EM convergence stability
```

### 3. Additional Ensemble diagnostics and weighting

The native workflow now vetoes a requested voter only for technical failures such as missing,
corrupt, mismatched, fallback, unsupported, or duplicate artifacts/state. Entrapment calibration,
observation counts, transfer-loss percentages, parity, unique/incremental yield, holdout outcomes,
and interaction changes remain nonblocking diagnostics. Additional diagnostics could include:

```text
finite-value fraction
pi0 relative to expert median/IQR
q-value saturation at 1.0
q-value floor saturation
number of discoveries at 1%
rank correlation to consensus
EM convergence status
boundary-solution flags
entrapment behavior when available
```

Experts could then be retained, downweighted, or assigned weight 0 before ensemble combination.

### 4. Entrapment-guided monotone recalibration

For models that are high-power but liberal, such as some Moments/MSFDR configurations, a monotone recalibration layer could map reported q-values onto entrapment-estimated FDR:

```text
q_reported -> q_entrapment_calibrated
```

Isotonic regression is a natural first implementation because it preserves rank order while correcting calibration.

### 5. Joint physical local-FDR modeling

RT and delta-mass evidence are currently used through optional bounded stages.  A future model could estimate local FDR over joint physical evidence:

```text
x = (score, delta_mass, delta_rt, ims, recurrence)
lfdr(x) = pi0 * f0(x) / (pi0 * f0(x) + pi1 * f1(x))
```

This should be bounded and gated in low-input data to prevent overfitting or uncontrolled rescue.

### 6. Clearer public/private field separation

The final output should eventually avoid transitional internal fields in public TSV output, or mark them clearly as diagnostic-only.  Public documentation should emphasize that only the final active fields and documented stage snapshots are intended for downstream use.

---

## Suggested logging for future validation

For publication-grade validation, log the following for each model fit:

```text
selected model_fit
active evidence space
resolved null windows
null pool size
rank counts
model convergence status
pi0 estimates
Storey fallback status
number of finite p/PEP values
PSM/peptide/protein targets at configured thresholds
PSM/peptide/protein entrapments at configured thresholds
first entrapment rank
RT/IMS anchor counts by run/charge
reproducibility eligible proteins/peptides
rescued PSM counts
expert weights in ensemble mode
```

For NOKOI and ensemble mode, also log the top-ranked entrapment peptides with:

```text
peptide
protein
run/file
charge
rank
base score
active p-value
active PEP
active q-value
peptide q
protein q
delta mass
RT error
IMS error when available
stage-specific rescue shifts
protein-supported peptide flag
peptide-supported PSM flag
```

---

## Recommended publication framing

The current evidence supports a narrow engineering statement:

> This experimental Sage fork implements a reproducible, dataset-local Decoy-Free workflow with
> native protein-level entrapment generation, shared candidate searches, model-specific null-window
> optimization, separate MS2Rescore annotation caching, explicit target-only calibration, and
> fail-closed validation. Frozen ISB comparisons and the required independent PXD001468 Moments
> comparison establish implementation parity within declared tolerances. They do not establish that
> Decoy-Free is generally better calibrated or more powerful than matched TDC.

Claims about identification gains, biological enrichment, or replacement of the statistical
default require the missing matched TDC evaluation and any model-specific portability/calibration
work relevant to the claimed workflow.

---

## Unsupported or removed keys

The following keys appeared in earlier configs or drafts but are not accepted by the current
`FdrOptions` structure:

```text
msfdr1_smix_min_null_rank
msfdr1_smix_max_null_rank
```

If they are present in a JSON file, they should be removed to avoid confusion.  MSFDR1_SMIX is rank-1 based in the current code and is configured with:

```text
msfdr1_bottom_frac_init
msfdr1_top_frac_init
msfdr1_pi_clamp_min
msfdr1_pi_clamp_max
```

---

## File-level implementation map

| File | Role |
|---|---|
| [`crates/sage-cli/src/workflow.rs`](crates/sage-cli/src/workflow.rs) | Native orchestration, stage checkpoints, calibration policies, Ensemble locks, and release evaluation. |
| [`crates/sage-cli/src/entrapment.rs`](crates/sage-cli/src/entrapment.rs) | Native foreign-source selection, protein entrapment generation, ratio measurement, and FASTA provenance. |
| [`crates/sage-cli/src/candidate_pool.rs`](crates/sage-cli/src/candidate_pool.rs) | Immutable candidate-pool persistence, fingerprints, stable IDs, and integrity checks. |
| [`crates/sage-cli/src/external_feature_cache.rs`](crates/sage-cli/src/external_feature_cache.rs) | Layered raw MS2Rescore prediction caches, stage-calibration identities, schema-v2 migration, and stable-ID joins. |
| [`crates/sage-cli/src/validation.rs`](crates/sage-cli/src/validation.rs) | Identification counting, parity, transfer, TDC, and expert-quality comparisons. |
| [`crates/sage/src/input.rs`](crates/sage/src/input.rs) | Search JSON surface, enums, defaults, clamping, and resolved `FdrSettings`. |
| [`crates/sage/src/decoy_free_fdr.rs`](crates/sage/src/decoy_free_fdr.rs) | Decoy-Free model fitting, active evidence pipeline, null-window evaluation, inference, and reporting. |
| [`crates/sage/src/ml/lower_order.rs`](crates/sage/src/ml/lower_order.rs) | Lower Order model and portable fitted state. |
| [`crates/sage/src/ml/msfdr.rs`](crates/sage/src/ml/msfdr.rs) | Seeded MSFDR, MSFDR1-SMIX, and MSFDR2-SMIX models. |
| [`crates/sage/src/ml/nokoi.rs`](crates/sage/src/ml/nokoi.rs) | Nokoi-style fitting, scoring, and artifact schema. |
| [`crates/sage/src/ml/stats.rs`](crates/sage/src/ml/stats.rs) | Statistical helpers, q-value utilities, p-value combiners, and soft caps. |

---

## Future improvements

The current code supports a large configuration surface, but several improvements would make the workflow more robust and easier to validate:

1. **Broader null-window diagnostics.** Extend the implemented in-memory constrained scanner with mass/RT null-likeness, pi0-stability, and EM-convergence diagnostics.
2. **Adaptive Ensemble weighting.** The workflow now excludes failed experts; future work can predeclare a weighting procedure and evaluate it with dataset-local holdout optimization.
3. **Bounded joint physical evidence.** RT, IMS, and delta-mass evidence should eventually be represented as local-FDR or empirical-Bayes evidence terms, but with caps, minimum support, reliability gates, and clear audit fields.
4. **Entrapment-guided calibration.** Where validation entrapments are available, use monotone recalibration, such as isotonic regression, to map reported q-values onto observed entrapment-estimated FDR.
5. **Layer-separated output fields.** Future output should preserve base, physical, reproducibility, and final hierarchical evidence separately so that every rescue/demotion is auditable.

The workflow already provides exact explicit/exhaustive and deterministic adaptive in-memory window
selection plus fail-closed expert exclusion;
the diagnostics and adaptive weighting described here are possible extensions to those conservative
implementations.

---

## References

Core decoy-free and lower-order modeling:

- **Modeling Lower-Order Statistics to Enable Decoy-Free FDR Estimation in Proteomics**  
  Dominik Madej and Henry Lam  
  *Journal of Proteome Research* 2023, 22 (4), 1159–1171  
  https://doi.org/10.1021/acs.jproteome.2c00604  
  https://pubs.acs.org/doi/10.1021/acs.jproteome.2c00604  
  https://github.com/dommad/pylord

Mixture modeling (MSFDR):

- **New mixture models for decoy-free false discovery rate estimation in mass spectrometry proteomics**  
  Yisu Peng, Shantanu Jain, Yong Fuga Li, Michal Greguš, Alexander R. Ivanov,  
  Olga Vitek, Predrag Radivojac  
  *Bioinformatics* 2020, 36(Supplement_2), i745–i753  
  https://doi.org/10.1093/bioinformatics/btaa807  
  https://academic.oup.com/bioinformatics/article/36/Supplement_2/i745/6055912  
  https://github.com/shawn-peng/DecoyFree-MSFDR

Early decoy-free ideas and Nokoi:

- **A Decoy-Free Approach to the Identification of Peptides**  
  Giulia Gonnelli, Michiel Stock, Jan Verwaeren, Davy Maddelein,  
  Bernard De Baets, Lennart Martens, Sven Degroeve  
  *Journal of Proteome Research* 2015, 14 (4), 1792–1798  
  https://doi.org/10.1021/pr501164r  
  https://pubs.acs.org/doi/10.1021/pr501164r  
  https://bio.tools/nokoi

Protein-level FDR:

- **Decoy-free protein-level false discovery rate estimation**  
  Ben Teng, Ting Huang, Zengyou He  
  *Bioinformatics* 2014, 30(5), 675–681  
  https://doi.org/10.1093/bioinformatics/btt431  
  https://academic.oup.com/bioinformatics/article/30/5/675/244620

Classical combination and FDR methods:

- **Statistical Methods for Research Workers** (Fisher’s Method)  
  R.A. Fisher, Oliver and Boyd, Edinburgh, 1925

- **The harmonic mean p-value for combining dependent tests**  
  Daniel J. Wilson  
  *PNAS* 2019, 116(4), 1195–1200  
  https://doi.org/10.1073/pnas.1814092116

- **Statistical significance for genomewide studies** (Storey Q-value)  
  John D. Storey and Robert Tibshirani  
  *PNAS* 2003, 100(16), 9440–9445  
  https://doi.org/10.1073/pnas.1530509100

## Upstream Sage and citation

This experimental fork is based on [Sage](https://github.com/lazear/sage). Consult the
[official Sage documentation](https://sage-docs.vercel.app/docs) for the underlying search engine,
spectrum processing, supported file formats, and standard TDC workflows. Decoy-Free workflow
behavior documented in this repository is fork-specific.

If you use Sage in a scientific publication, cite:

[Sage: An Open-Source Tool for Fast Proteomics Searching and Quantification at
Scale](https://doi.org/10.1021/acs.jproteome.3c00486).
