# Null-window selection policies

`strict_v1` (also the default when omitted) retains historical empirical hard
ceilings and navigation. `reporting_guided_v1` is an explicit, scientifically
different development-selection policy, not a correction to historical results.
Changing policy requires a new analysis/proposal/checkpoint identity. Do not
resume an old checkpoint by editing its identity or reinterpret its terminal
no-feasible-window result.

## Workflow configuration

```json
{
  "validation": {
    "null_window_selection_policy": "reporting_guided_v1",
    "null_window_fdp_references": {"psm": 0.01, "peptide": 0.01, "protein": 0.01}
  },
  "parameter_optimizer": {
    "schema_version": 6,
    "empirical_selection_policy": "reporting_guided_v1"
  }
}
```

These are additions to a complete workflow, not a standalone executable
manifest. The outer and inner policy must explicitly agree. Schema 6 is the
optimizer **configuration** schema; existing checkpoint layout 5 remains
supported, with policy bound into its configuration fingerprint.

Reporting thresholds `precursor_fdr`, `peptide_fdr`, `protein_fdr` remain fixed
q-value reporting controls. Existing default/per-expert/final-Ensemble ownership
and override precedence applies. Guided window evaluation retains these three
effective settings independently; it does not overwrite them with the shared
optimizer threshold. Historical strict evaluation retains its historical shared
threshold behavior. No reporting threshold is optimized by this policy.

Empirical references are separate. A complete triplet in
`validation.null_window_fdp_references_by_expert` overrides the global triplet
for that canonical expert; the global triplet overrides the legacy shared
`validation.fdr_threshold` fallback. All references must be finite in [0,1].
There is no Ensemble null window. Outer empirical constraints keep their
independently declared levels/references; the outer objective is unchanged.
For direct native null-window options, the corresponding fields are
`selection_policy` and `fdp_references` inside `null_window_optimizer`.

## Guided ranking and eligibility

Only numerically valid evaluations with defined finite FDPs and nonempty target
PSM, canonical-peptide and protein populations are rankable. Undefined/empty
metrics are explicitly unavailable, not numerical fit failures and not zero FDP.
Provenance failures and prohibited fit fallback remain fail-closed. Configured
q-method fallback is recorded separately and is not invented positive evidence.

Lexicographic ranking, best among **evaluated** windows:

1. Prefer zero observed selection-entrapment proteins.
2. Minimize `max(0, FDP-reference)`: protein, canonical peptide, PSM.
3. Maximize target proteins, canonical peptides, PSMs.
4. Minimize FDP: protein, canonical peptide, PSM.
5. Prefer narrower, then lower-start, then lower-end windows.

No absolute distance to a reference is used. Above-reference and underpowered
results can be development-selected without being called reference-passing or
statistically validated. The report retains numerical validity, rankability,
each reference result, power and uncertainty separately. Underpowered status
remains `not_evaluable_underpowered`; default eligibility is `not_evaluated`.
Outer trial selection and later blocks can progress using such results under
the original outer objective. The window ordering is **not** the Ensemble
objective or a voter-admission gate.

Level 4 itself does not require zero entrapment proteins. It still uses the
independent q-values, accepted protein anchors and peptide/PSM support flags.
No row, entrapment label, fitted probability or reporting flag is removed or
modified by ranking. Audit labels never enter this selection key. Zero observed
selection entrapments are an observation, not proof of zero error. Raw-q metrics
are separately labeled diagnostics and cannot veto Level-4 window selection.

## Adaptive navigation and coverage

Guided `adaptive` uses rankability, not a protein-FDP hard gate, for navigation;
its best-neighbor comparisons use the same production ranking key.
Guided `landscape_adaptive` uses the bounded sparse-probe/hill-or-boundary route
(`reporting_guided_sparse_*_v1`) because the legacy monotone hard-FDP frontier
classification is inapplicable to soft references. The existing sparse offsets,
row step, eligible-fraction switch, stride, dead-row bound and hill-step limit
control this route. Legacy landscape-specific coarse/frontier heuristic controls
are not used by this guided route. Bounds and seed are unchanged, but visited
windows can differ. This is a policy-specific navigation change, not a claim
of identical adaptive coverage. Reports record the route and actual coverage;
unvisited zero-entrapment windows cannot be ruled out. Exhaustive requests still
evaluate the complete declared universe. `strict_v1` keeps its original route.

## Saved-evidence analysis

```sh
sage rerank-null-window-evidence CHECKPOINT.json \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --policy POLICY.json --policy-sha256 POLICY_SHA256 \
  --output NEW_RERANKING.json
```

The policy file is:

```json
{"schema_version":1,"selection_policy":"reporting_guided_v1","fdp_references":{"psm":0.01,"peptide":0.01,"protein":0.01}}
```

The command verifies hash-bound saved numerical evidence, counts and FDP
calculations, ranks all supplied windows using production logic, atomically
publishes and reopens a new report, and never constructs a Runner or launches
external processes. It cannot access spectra, pools, raw caches, fit models or
create a production winner lock. Original evidence is unchanged. The report
contains the full key, observed zero-entrapment count, measurements, each
alternative's decisive losing priority and target-yield tradeoffs.

Reranking is a new analysis of old measurements, not an exact optimizer resume,
a new fitted artifact or independent validation. A data-informed policy amendment
must disclose that history. Frozen searches, candidate pools and raw predictions
can retain their identities; new analysis/optimizer/checkpoint identities cannot.
