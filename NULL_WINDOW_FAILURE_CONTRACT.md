# Null-window failure and single-trial diagnostic contract

## Failure lifecycle

`WorkflowTrialEvaluator` invokes `run_search_stage`, which installs the model-local
window policy and calls `Runner::run_with_workflow_caches`. The runner loads the
exact candidate pool, excludes explicitly decoy-labelled rows, converts them to
DF features, and evaluates native windows **before** the external-feature join.
No spectra are searched on strict reuse. Window selection failure prevents the
raw-feature join and all subsequent stages of that trial.

The previous core API discarded the actual fitted artifacts at each evaluated
window and returned one string if no window passed every FDP limit. The workflow
mapped any stage error to `TechnicalFailure`; cleanup removed nonwinner trial
directories. This establishes a classification/evidence defect, not proof that a
particular historical fit was valid. Deleted window evidence cannot be recovered
by interpreting a terminal error or counting repeated log warnings.

The detailed API now preserves each evaluated window separately:

| Outcome | Evidence | Outer classification |
| --- | --- | --- |
| Empirical rejection | Valid fitted state; defined FDP above a declared limit | EmpiricallyInfeasible |
| Unavailable metric | Valid fitted state; undefined FDP (for example empty accepted population) | NotEvaluable |
| Technical failure | Missing/invalid fitted artifact, invalid observed probabilities, or structural/provenance error | TechnicalFailure |
| Mixed evaluated outcomes | All individual reasons retained | NotEvaluable, mixed_window_outcomes |
| Defined underpowered estimate | Nonempty population, few entrapments; point-limit decision remains independent | No new power gate |
| Dependency pruning | Existing pre-production dependency reason; no window evaluation | Existing dependency diagnostic retained |

Zero observed entrapments with nonempty target evidence gives a defined zero FDP;
it is not an empty population. It remains underpowered. No failed constraint is
converted into a pass and no baseline is substituted. A valid model artifact is
checked with the same validator used for workflow winners. Purified-null fallback
and the actual q-method/fallback reason are observed separately: a configured
Storey-to-BH fallback is not synonymous with failure to fit a Moments model.

Adaptive terminal wording is **no feasible evaluated window**. The coverage,
candidate-universe size, adaptive branch and exhaustive-fallback flag delimit
what was tested; no claim about unvisited windows follows from adaptive failure.

## Scientific contract unchanged

`settings_for_null_window` fixes the requested rank window and reporting
thresholds. Level4 applies independently calculated PSM, peptide and protein
q-values plus protein-supported peptide/PSM flags; it does not replace raw q-values.
The Level4/hierarchy compatibility guard remains in preflight and at runtime.

For each constrained level, the empirical calculation remains
`clamp(E * (1 + 1 / measured_ratio) / (T + E), 0, 1)`. Empty populations or invalid
ratios remain undefined. The inner workflow policy constrains PSM, canonical
peptide and protein FDP at `validation.fdr_threshold`. It uses the verified
selection partition's peptidoform, canonical-peptide and protein ratios,
respectively. Its low-count warning threshold is three observations. The outer
optimizer applies its separately declared empirical constraints and development
power policy; these are not interchangeable with the inner three-level guard.

The window counter uses rank-1, label-1, noncontaminant rows. Audit-only and mixed
target/entrapment mappings enter neither selection FDP numerator nor denominator.
Selection membership is applied before counts. Peptides are canonical I/L-normalized
sequences; proteins use the inferred hypothesis. The final validation reporter
uses the same Level4 support flags and measured-ratio equation. It deduplicates
PSM identifiers and fails on cross-partition groups; the in-memory counter counts
rank-1 rows and conservatively skips mixed selection/audit mappings. Agreement
therefore relies on the already-required unique candidate/PSM population and
component-disjoint partition. These existing distinctions are not changed here.
Production fitting/q calibration remains label-blind in selection/audit mode;
that calibration reference population is not the empirical target denominator.

The preserved ISB18 frozen parity and PXD001468 locked-holdout manifests specify
Level4 and a 1% threshold. They do not contain the later staged parameter optimizer.
The archived PXD001468 parity report records an explicit visited grid and selected
10–10 window; its historical legacy peptide counters are marked diagnostic-only.
Those successes establish neither fit validity nor feasibility on a different
dataset/adaptive trajectory. No historical dataset was rerun for this correction.

Database-fragment construction builds an in-memory peptide index, not a spectrum
search. The workflow shares that index across parameter trials with the same
strict search fingerprint. Strict preflight has its own index construction;
each production trial still reopens/verifies/deserializes the candidate pool.
Sharing a verified immutable candidate representation is a separate performance
opportunity, not part of this correction.

Staged-coordinate control flow is unchanged: failure to select a winner in the
model-fit block stops that expert before its subsequent q/calibration blocks and
stops later experts. Whether later q settings would help is not tested by a failed
earlier block and is not grounds to change the preregistered schedule.

## Durable evidence and replay

The runner writes native settings, schema-v2 window checkpoints and terminal
failure reports atomically, syncs and reopens their bytes, and writes content-hash
sidecars. A checkpoint with absent evidence, wrong schema/fingerprint or content
hash fails closed; it is not silently overwritten as a new search. Each completed
window records its settings hash, actual artifact content hash and scalar state
(large arrays are count/hash summaries), hierarchy, annotation-join
state, fit validity, q diagnostics, fallback events, constrained counts, ratios,
numerators/denominators, values, limits and individual predicates.

Failed trials and trials with window evidence are retained by cleanup. This
conservative policy also retains large payloads; a future compactor must preserve
and verify the evidence before removing anything. An interrupted window itself
may not have completed evidence, but earlier completed windows and the exact
settings remain. Historical artifacts and classifications are not rewritten.
Exact compatible checkpoint replay reuses completed window evaluations. Normal
successful searches still rematerialize the selected model; the diagnostic is
not a replacement production winner.

## Bounded diagnostic interface

```sh
sage diagnose-null-window-trial FROZEN_WORKFLOW.json \
  --checkpoint HISTORICAL_OPTIMIZER.checkpoint.json \
  --checkpoint-sha256 EXPECTED_FILE_SHA256 \
  --trial-id EXACT_TRIAL_ID \
  --output NEW_DIAGNOSTIC_DIRECTORY
```

This fresh-output-only command verifies the immutable checkpoint, historical
proposal payload, unchanged manifest/search/proposal content, and exact effective
options and window policy. It verifies strict existing entrapment, partition,
pool and raw-cache resources; derives selection ratios from that verified
partition; and installs the same production window policy. Current implementation
identity is recorded separately. It cannot run the outer parameter optimizer,
join/calibrate external features, generate annotations, evaluate audit outcomes,
run target-only/TDC, or publish production winners. It retains all evaluated
windows even if none is acceptable. The raw cache is verified, **not joined**, to
faithfully reproduce the original native-before-annotation phase.

Exit zero means the diagnostic report was written, not that a feasible window
exists. Read its status, failure classification, coverage and per-window evidence.
Do not use this report as a new optimizer result or silently resume another run.
