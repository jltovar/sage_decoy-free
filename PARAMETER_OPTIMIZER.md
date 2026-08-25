# Internal Decoy-Free parameter optimizer

The workflow `parameter_optimizer` section is a schema-versioned, development-only interface for
bounded optimization of analysis parameters. It never changes a Sage search input. Every trial
must reuse an existing immutable candidate pool and the existing layered raw MS2PIP/DeepLC cache;
a missing or incompatible resource fails closed. The optimizer refits only inexpensive statistical
calibration and model state.

The repository inventory in
`validation/statistical_conformance/parameter_catalog.json` is the authoring and review record.
Runtime validation uses the same embedded Rust contracts, so a portable workflow never depends on
a repository checkout or machine-specific catalog path.

Every runtime result also serializes `parameter_binding_coverage`. Each entry names its supported
scope, setter path, production `FdrOptions`/`FdrSettings` field, dependency predicate, and
executable, conditional, or deliberately deferred status. Enabled spaces fail closed when a
binding is deferred; catalog exposure can never silently become a no-op.

## Ownership, scope, and precedence

Every block declares one scope: `default`, `per_expert`, `ensemble_final`, `physical`,
`reproducibility`, `hierarchical_or_reporting`, or `numerical_only`. Model-local blocks also name
exactly one of `moments`, `mle`, `lower_order`, `msfdr_seeded`, `msfdr_1smix`, `msfdr_2smix`, or
`nokoi`; final Ensemble blocks name `ensemble`. A parameter owned by a different expert is rejected.
Physical, reproducibility, and hierarchy blocks name the stream they modify and remain separate
because their applicability depends on run, group, sample-role, instrument, and IMS metadata.

Values resolve in this order, with the later source winning:

1. compiled default;
2. workflow default;
3. fixed baseline/per-expert override;
4. optimizer trial value;
5. final Ensemble-only value.

Resolved maps are kept separately by expert and scope. Sharing candidate rows or raw predictions
does not share null windows, fitted nuisance state, calibration, aggregation, or artifacts. The
final Ensemble retains its existing shared external-profile contract; it does not reinstate
last-expert-wins behavior. MSFDR1-SMIX is always rank 1–1 and Ensemble has no null window.

The seven `ensemble_weight_*` values are independently configurable. Every JSON-selected expert
must remain present and must have a finite strictly positive effective weight. Statistical
diagnostics, identification yield, parity, transfer loss, entrapment FDP, or holdout performance
never become voter-admission rules.

## Search strategies

`exhaustive_grid` sorts parameter names and canonical JSON values, evaluates every valid declared
combination, and refuses a grid larger than its block or workflow trial budget. It is called a
bounded optimum only for a single completely evaluated declared block. Multiple exhaustive blocks
are a deterministic staged/local result, not a global Cartesian optimum.

`staged_coordinate` visits declared blocks in `block_order`, parameters in canonical name order,
and values in canonical JSON order. It repeats up to `maximum_optimization_passes`, stopping after
a pass without improvement or at the trial budget. It is always classified as deterministic
heuristic/local optimization.

Candidate values are type- and domain-validated when the manifest is loaded. A combination that
is individually valid but violates a relational dependency is still assigned its deterministic
trial identity and consumes one declared trial-budget position, then is recorded as
`parameter_dependency_invalid_before_production` without calling the production evaluator. Such
points are therefore counted in conservative upper bounds but pruned before fitting.

A model block may include `window_search` with `explicit_grid` or `landscape_adaptive`. The window
is selected inside that model trial using the existing bounded null-window implementation and the
same declared development objective. Every expert retains its selected dataset-local window and
artifact. Both optimized and explicitly fixed windows are serialized in winner provenance. There
is no combined Ensemble window.

## Objective and validity

The objective is an explicit ordered list. The provisional low-input default is Level-4 proteins,
canonical peptides, peptidoforms, and PSMs (all maximize), then measured-ratio-adjusted
entrapment FDP, model complexity, and canonical parameter order (all minimize). The workflow's
fixed evaluation threshold is recorded and must equal its validation threshold.
`precursor_fdr`, `peptide_fdr`, and `protein_fdr` are fixed reporting thresholds and are rejected
as yield-optimization variables.

Technical validity, empirical feasibility, ranking, complexity, and deterministic ties are
separate. A failed fit—including the retained MSFDR mixture-identifiability rules—is an infeasible
trial with its technical reason. No fallback model is substituted. A valid trial that violates a
declared entrapment FDP ceiling is `empirically_infeasible`.

Schema v3 separates empirical point-estimate compliance from empirical power with
`underpowered_trial_policy`. Omitting it, or setting `not_evaluable`, preserves the historical
behavior: fewer than `minimum_entrapment_observations_for_power` accepted entrapments makes the
trial non-selectable. The explicit `development_eligible` policy keeps a technically valid trial
selectable when its finite adjusted-FDP point estimate is within the declared ceiling, while
recording `empirical_calibration_power: underpowered`,
`statistical_validation_status: not_evaluable_underpowered`, and
`statistical_default_eligibility: not_evaluated`. Zero observed entrapments remain zero—no
pseudocount is invented—and never establish calibration. The policy is accepted only with
`classification: development_only`; holdout, release, statistical-default, and production-default
claims remain prohibited. Failure of every empirical trial never changes the Ensemble roster.

Structural method-family comparisons set `structural_comparison: true`. Score/evidence-related
q-value covariates and other conditional choices marked by the catalog additionally require an
explicit named statistical-validity contract. In particular, JSON exposure does not establish
the null-independence or cross-fitting validity of `matched_peaks`, `best_longest_y_pct`,
`nsaf_observable_length`, or any other covariate.

Post-optimization review must distinguish active scientific settings from dormant,
reparameterization-equivalent, numerical, and provenance-only settings. Lower Order
`lo_evalue_scale` and `lo_tev_transform` are not eligible yield objectives: with the production
`neg_log_e` family they induce only a positive affine change that the fitted location/scale
absorbs. Their canonical values are `1.0` and `neg_log_e`; legacy spellings
`log_1000_over_e` and `scaled_log_1000_over_e` load only as compatibility aliases and serialize
canonically as `log1000_over_e` and `scaled_log1000_over_e`. Seeded MSFDR
`msfdr_multistart` is retained for configuration compatibility but is not consumed by the current
production estimator and is therefore ineligible until a production binding and numerical
convergence contract are implemented. A declared active optimization block containing any of
these fields fails closed rather than recording a no-op winner.

Final-Ensemble dependencies are also fail-closed. P-value combiner choices and Cauchy penalty are
active yield variables only for `final_evidence_space: p_value`; PEP combiner choices and their
shape parameters are active only for `final_evidence_space: pep`. Expert weights are active only
for a PEP final stream using `weighted_mean` or `weighted_median`. Otherwise those values remain
positive canonical defaults and may still describe auxiliary stored evidence, but the optimizer
does not present them as identification-yield winners.

## Leakage, identity, and resume

Only the +entrapment development population contributes feasibility, objective values, ranking,
early stopping, window selection, or parameter selection. Each trial request records
`target_only_outcomes_allowed: false`.

### Dataset-local entrapment selection/audit holdout

Schema v4 adds this explicit configuration:

```json
"entrapment_validation": {
  "mode": "selection_audit",
  "partition_schema_version": 1,
  "seed": 42,
  "salt": "prospectively-frozen-dataset-salt",
  "selection_fraction": 0.5,
  "audit_fraction": 0.5,
  "require_existing_partition": true
}
```

The workflow's `entrapment.partition_artifact` names the immutable artifact. Missing configuration
defaults to `full_population_development`, preserving historical development behavior and making
no independent calibration claim. Selection/audit mode requires schema v4, a nonempty salt,
positive fractions summing to one, and nonempty realized populations. A first prospective run may
create the artifact; a formal replay should set `require_existing_partition: true` so any missing,
corrupt, mismatched, or incompatible artifact fails closed during strict preflight.
The mode currently requires `execution_mode: optimization_only`; downstream target-only reporting
is a separate invocation using already frozen winners, never part of the selection/audit optimizer.

Prospectively create the artifact before any trial with
`sage materialize-entrapment-partition workflow.json`. The command reads only the workflow,
digestion/search-space configuration, dataset identities, target and active +entrapment FASTAs, and
the existing entrapment-construction report. It does not resolve candidate or annotation caches,
search spectra, fit models, evaluate trials, or access target-only resources. Freeze the resulting
hash, change `require_existing_partition` to `true`, and then run strict workflow preflight.
Use `--inputs-only` first to record the exact portable dataset, FASTA, digestion, and
entrapment-construction identities without assigning components or writing an artifact.

Partitioning occurs at the foreign-protein connected-component level. Sage applies its configured
digestion and modification search space, canonicalizes I/L, links all entrapment proteins sharing
any searchable peptide, and assigns the whole content-identified component by a cryptographic hash
of the schema, seed, salt, and component identity. Scores, accepted discoveries, FDP outcomes, file
paths, FASTA/candidate ordering, thread scheduling, and prior trials never enter assignment.
Selection and audit protein, canonical-peptide, peptidoform, and protein-group populations must be
disjoint. Protein, peptide, and peptidoform ratios are measured separately from the realized
observable components; no requested-fraction multiplication is used.

All candidates remain physically present while models fit, but production fitting and q-calibration
see no selection or audit entrapment role. The selection identity list is consumed only by
model-local window metrics and outer trial summaries: genuine targets count normally, selection
entrapments supply the development FDP numerator, and audit entrapments are ignored rather than
treated as targets. Trial records contain the partition identity and `selection_only`
population marker but no audit label or metric. After every expert and the final Ensemble winner is
frozen, each retained winner result table receives exactly one immutable audit read. Its separate
record reports target/audit counts, realized audit ratios, adjusted FDP and Wilson-derived 95%
intervals by level, power, and validation classification. Exact resume verifies and reuses that
record. Audit results cannot change the winner, checkpoint, convergence, fallback, or voter roster;
zero audit entrapments never imply statistical validation.

The partition identity participates in analysis, optimizer, trial, and checkpoint fingerprints.
The strict spectrum-search fingerprint, immutable candidate-pool identity, and layered raw
MS2PIP/DeepLC prediction identity remain unchanged. All experts and Ensemble use the same artifact.
Target-only remains downstream of the frozen dataset-local winners and cannot retune any partition,
parameter, window, or Ensemble setting. Each new dataset must create its own partition and perform
its own complete expert-local and final-Ensemble optimization; prior ISB winners may be candidate
values but are not transferred answers or statistical defaults.

Schema v2 adds an explicit `execution_mode`. `optimization_and_post_selection` is the default,
including for schema-v1 manifests that omit the field, and preserves the historical behavior of
continuing into ordinary post-selection and target-only reporting after winners are frozen.
`optimization_only` runs the same production evaluator, optimizer, objective, bindings, and
checkpoint engine, but its execution boundary ends after all +entrapment winners—including the
final Ensemble winner—are materialized. In that mode strict preflight resolves only the
+entrapment pool and layered raw cache; target-only pool/cache paths are not opened or hashed, and
ordinary MS2Rescore, target-only, interaction-diagnostic, and validation stages are explicitly
reported as `not_run_by_execution_scope` rather than failed or completed.

The portable optimizer fingerprint binds execution mode, dataset, candidate pool, raw annotation cache, optional
calibrated-annotation identity, model/artifact and optimizer schemas, requested/resolved settings,
parameter spaces, scopes, precedence, strategy, block order, objective, constraints, budgets,
seed, source configuration, and catalog contract. Absolute paths, usernames, hostnames, wall-clock
timestamps, and process-local `psm_id` values do not participate.

The identity also embeds a build-time SHA-256 over the optimizer engine and production workflow
evaluator sources. A binary whose trial-evaluation implementation changes therefore rejects an
older checkpoint instead of silently replaying results under a different evaluator.

Trial IDs are immutable hashes of that fingerprint, block, pass, ordinal, and canonical parameter
map. An atomic checkpoint stores every compact trial record and a payload SHA-256. Resume validates
schema, fingerprint, and payload integrity, then reuses completed trial identities exactly.
Nonwinner result/artifact payloads are removed after completion; full diagnostics and fitted state
remain only for each block winner.

When independently optimized experts are assembled, the schema-v9 Ensemble lock stores the
complete canonical effective `FdrOptions` and fully resolved `FdrSettings` for each expert, not
only optimizer deltas or selected windows. It also binds the fitted artifact and the
dataset/search/candidate/annotation identities under which that configuration was consumed. The
final Ensemble configuration has its own hash and cannot overwrite expert-local settings.
Target-only `refit_with_locked_window` reconstructs each expert from this locked configuration;
`reuse_dataset_artifact` additionally verifies the artifact/configuration identity. Old partial
locks are deliberately incompatible with refit and `compare_both`.

Legacy outcomes distinguish `exhaustive_bounded_optimum`, `completed_heuristic_local`,
`trial_budget_exhausted`, `no_technically_valid_solution`, `no_empirically_feasible_solution`,
`interrupted_resumable`, and `not_evaluable`. Schema-v3 development-eligible runs additionally
use `underpowered_development_winner` when the selected development winner is underpowered and
`completed_development_optimization` when the run completes without that condition. Powered,
underpowered, and not-assessed trial counts are serialized separately. Neither completion status
asserts statistical-default eligibility.

`implementation_smoke_only: true` is a bounded infrastructure-test mode limited to 16 trials and
non-Ensemble blocks. It writes optimizer/checkpoint evidence but skips ordinary and target-only
workflow stages. It is not a scientific optimization mode and must be false in a full development
manifest.

`production_smoke_only: true` is likewise bounded and non-Ensemble, but it uses the ordinary
production trial evaluator and real +entrapment Level-4 metrics before stopping after winner
materialization. It exists only for cache-only integration acceptance. It never runs an ordinary
post-winner or target-only stage and is mutually exclusive with `implementation_smoke_only`.
It is not a substitute for `execution_mode: optimization_only`: the latter supports Ensemble,
normal scientific trial budgets, every production search strategy, and exact resume.

## Parameter classes

The catalog classifies fields as scientific numeric candidates (A), structural method-family
choices (B), numerical convergence/precision controls (C), fixed reporting thresholds (D),
validation/reporting/provenance controls (E), or unsafe/unsupported candidates (F). Classes C, D,
and E cannot be optimized for identification yield. Numerical controls such as EM iterations and
tolerance, bootstrap iteration count, logit epsilon, and Storey degeneracy epsilons are tested for
adequate convergence rather than selected by identifications.

The physical, recurrence, and hierarchical blocks are implemented as explicit conditional scopes
but are intentionally absent from the initial blocks 1–4 development run. Recurrence must respect
declared run/group/sample-role boundaries and must never use unrelated biological groups, blanks,
or negative controls as generic rescue.
