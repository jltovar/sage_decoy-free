# Sage Decoy-Free statistical methods

This document specifies the current method at commit
`8dacdb43dffc33e3cb05553c8311813ee540bff2`. It is a conformance contract, not
a claim that every Sage-specific method is a literal port of the cited paper.
The companion machine contracts are
`validation/statistical_conformance/method_contracts.json`; the predeclared
synthetic evaluation is `validation/statistical_conformance/simulation_plan.json`.

## Audit status

The single Step 2 pass found one demonstrated production defect and two
important interpretation limits. The defect correction below was preregistered
before inspecting the existing ISB fitted artifacts, implemented as a generic
post-fit gate, and verified without changing any valid ISB fit:

- MSFDR1-SMIX and MSFDR2-SMIX formerly accepted coincident zero-variance
  components as fitted models. Such mixtures are nonidentifiable. They now
  fail closed with an explicit technical reason before evidence or artifacts
  are emitted; the active regression suite covers the gate.
- Ensemble `second_best` is the raw second order statistic. It is continuous,
  deterministic, order invariant, and robust to one optimistic expert, but it
  is not a universally calibrated combined p-value. For three independent
  null uniforms, `Pr(P_(2) <= 0.9)=0.972 > 0.9`.
- Ensemble median PEP is an operational PEP-like consensus. It is not a
  posterior probability without a joint model for expert errors and
  dependence.

The active correction tests and targeted cache-only MSFDR1/MSFDR2 regression
passed. Identification counts were not a validity criterion and were not used
to choose the gate. The project is ready for Step 3 with the named
interpretation and source-method limitations in this document.

## Fixed project decisions

This specification preserves internal entrapment-FASTA generation; measured
protein, canonical-peptide, and peptidoform ratios; one immutable candidate
search per strict search fingerprint; model-local, dataset-local null-window
selection; fixed rank-1/one-incorrect-component MSFDR1-SMIX; layered raw
MS²PIP/DeepLC caching; JSON-selected Ensemble participation; technical-only
fail-closed voter exclusion; continuous PSM-first Ensemble combination; and
the current target-only policies. `precursor_fdr`, `peptide_fdr`, and
`protein_fdr` remain reporting error-rate thresholds and are excluded from the
Step 3 parameter catalog.

## Conformance vocabulary

| Class | Meaning |
|---|---|
| A | Exact implementation detail |
| B | Numerically equivalent implementation |
| C | Intentional ultra-low-input modification |
| D | Engineering or determinism change without statistical effect |
| E | Statistically meaningful project-specific extension |
| F | Unintentional or unjustified divergence |
| G | Not evaluable from available evidence |

One model may contain several classes because its source equation, estimator,
and low-input extension can have different classifications.

## Primary-source inventory

| Primary source | Equation or algorithm used | Required assumptions | Sage use |
|---|---|---|---|
| E. J. Gumbel, *Statistics of Extremes*, Columbia University Press, 1958, DOI [10.7312/gumb92958](https://doi.org/10.7312/gumb92958) | Type-I extreme-value location/scale distribution; `E[X]=mu+gamma beta`, `Var[X]=pi² beta²/6`; likelihood score equation for beta | observations follow one nondegenerate Gumbel location-scale population | Moments, MLE, seeded MSFDR null, Lower Order rank-1 tail |
| D. Madej and H. Lam, “Modeling Lower-Order Statistics to Enable Decoy-Free FDR Estimation in Proteomics,” *J. Proteome Res.* 22 (2023) 1159–1171, DOI [10.1021/acs.jproteome.2c00604](https://doi.org/10.1021/acs.jproteome.2c00604) | supplement’s k-th lower-order extreme-value density/CDF, moment equations, MLE, finite-N/asymptotic relation, and lower-order-to-top-null extrapolation | lower ranks are predominantly incorrect; comparable candidate-count/search populations; stable relation among order distributions; sufficient ranks | Lower Order |
| Madej and Lam author code, [PyLord](https://github.com/dommad/pylord), audited at commit `3678451c50fa449dcbf67cf84ba203f813e831dc` | `exp(-kz-exp(-z))/(beta (k-1)!)`; asymptotic CDF; TEV-k moments and MLE | same as the paper; PyLord recommends at least ten hits per spectrum | Lower Order equation parity |
| Y. Peng et al., “New mixture models for decoy-free false discovery rate estimation in mass spectrometry proteomics,” *Bioinformatics* 36 Suppl. 2 (2020) i745–i753, DOI [10.1093/bioinformatics/btaa807](https://doi.org/10.1093/bioinformatics/btaa807) | `S1=a SN(C)+(1-a)SN(I1)`; `S2=a SN(I1)+(1-a-b)SN(I2)+b SN(C)`; latent truncated-normal EM; Eq. 4 tail-area FDR | identifiable skew-normal components; one first and one second score per spectrum for 2SMix; correct model family; stable likelihood optimum | MSFDR1-SMIX and structural basis of MSFDR2-SMIX |
| Peng et al. author code, [DecoyFree-MSFDR](https://github.com/shawn-peng/DecoyFree-MSFDR), audited at commit `e480daffdb6a3ad3ca37221f2e4ceb2a393b2419` | constrained joint mixture weights, skew-normal EM, multi-initialization, paper FDR curve | same as paper; supplied initialization and constraints | comparison target for MSFDR family |
| G. Gonnelli et al., “A Decoy-Free Approach to the Identification of Peptides,” *J. Proteome Res.* 14 (2015) 1792–1798, DOI [10.1021/pr501164r](https://doi.org/10.1021/pr501164r) | externally trained L1 logistic classifier on 47 published features using Mascot-rank-derived labels | heterogeneous external training population transfers to applications; feature/search-engine contract matches | historical basis for Nokoi; not an exact specification of Nokoi v2 |
| Y. Benjamini and Y. Hochberg, “Controlling the False Discovery Rate,” *JRSS B* 57 (1995) 289–300, DOI [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) | ordered `p_(i)m/i` followed by reverse cumulative minimum | valid null p-values; independence or the paper’s applicable dependence conditions | BH q-values and conservative fallbacks |
| J. D. Storey, “A Direct Approach to False Discovery Rates,” *JRSS B* 64 (2002) 479–498, DOI [10.1111/1467-9868.00346](https://doi.org/10.1111/1467-9868.00346); J. D. Storey and R. Tibshirani, *PNAS* 100 (2003) 9440–9445, DOI [10.1073/pnas.1530509100](https://doi.org/10.1073/pnas.1530509100) | `pi0(lambda)=#{p>lambda}/(m(1-lambda))`; `q_(i)=min_{j>=i} pi0*m*p_(j)/j` | valid null p-values and an estimable null-dominated upper tail; applicable dependence/large-sample conditions | pi0 and Storey q-values |
| K. Strimmer, “A unified approach to false discovery rate estimation,” *BMC Bioinformatics* 9 (2008) 303, DOI [10.1186/1471-2105-9-303](https://doi.org/10.1186/1471-2105-9-303) | modified Grenander decreasing-density estimate and local FDR `pi0 f0/f` | p-null density is uniform; mixture density can be represented by the monotone estimator; reference population is appropriate | PEP-like calibration for p-native experts and Nokoi |
| Y. Liu and J. Xie, “Cauchy Combination Test,” *JASA* 115 (2020) 393–402, DOI [10.1080/01621459.2018.1554485](https://doi.org/10.1080/01621459.2018.1554485) | average Cauchy transform of p-values and inverse-Cauchy tail | individually valid p-values; tail approximation conditions under dependence | default Ensemble and aggregation p combiner |
| B. Wen et al., “Assessment of false discovery rate control in tandem mass spectrometry analysis using entrapment,” *Nature Methods* 22 (2025) 1454–1463, DOI [10.1038/s41592-025-02719-x](https://doi.org/10.1038/s41592-025-02719-x) | combined-list upper-bound estimator `E(1+1/r)/(T+E)` | effective entrapment/target size ratio is correct; incorrect matches have the required equal-chance analogue; sufficiently large or replicated evaluation | measured-ratio window constraints and validation summaries |
| B. Teng, T. Huang, and Z. He, “Decoy-free protein-level false discovery rate estimation,” *Bioinformatics* 30 (2014) 675–681, DOI [10.1093/bioinformatics/btt431](https://doi.org/10.1093/bioinformatics/btt431) | degree-preserving randomized peptide-protein graphs, permutation protein p-values, Storey pFDR | randomized graph is the protein null and adequately preserves graph structure; enough permutations | historical protein-level reference only; current Sage aggregation is not this algorithm |

Fisher’s 1925 product method and other optional combiners are implemented as
documented utilities, but they are not the defining sources for the default
audited path. Repository citations, history, the two pinned author-code
snapshots, and preserved Phase 5/8 reports were inspected before external
sources.

## Shared statistical contract

### Candidate, label, and rank populations

The candidate search is immutable for one strict search fingerprint. Models
may select different lower-rank views of that retained population, but analysis
knobs do not trigger a different search. Final Decoy-Free PSM evidence is
defined only for finite rank-1 candidates. Lower ranks are null observations
for models that declare them; all final and model-specific fields on non-rank-1
rows are scrubbed.

Entrapment labels may evaluate a fit or select a null window, but they are not
positive/negative training labels for Moments, MLE, Lower Order, MSFDR, or
Nokoi. Target/entrapment mappings that are mixed or ambiguous are excluded
from the corresponding measured-FDP counts. A target-only result cannot
reoptimize a window because it has no valid entrapment objective.

### Probability and error-rate semantics

- Native p-values are finite, clamped to `[1e-300,1]`, and smaller means
  stronger evidence. Nonfinite input fails to 1.
- PEP/logit values are finite and bounded away from exact 0 and 1 for numerical
  transforms. Model posteriors and empirical PEP-like calibrations are kept
  distinct in the method contracts.
- P-native q-values use the configured BH/Storey-family method. Storey reports
  its actual method and fallback reason; sparse or failed pi0 estimation falls
  back to BH.
- PEP-native q-values are best-first cumulative means of PEP followed by the
  monotone reverse-minimum correction. A cumulative mean is an expected false
  fraction only to the extent that its inputs retain a PEP interpretation.
- Zero observations produce no q-values. Zero or invalid entrapment denominator
  is `None`, not zero. A failed fit is unavailable and never presented as a
  fitted model. Workflow `not_evaluable` is never converted into a pass.

### pi0 and Grenander calibration

For a declared reference p-value population, Sage evaluates a high-lambda
Storey grid, clamps admissible estimates, and uses the median or trimmed mean.
At least three usable lambda estimates are required. The Grenander PAVA fit
pools exact ties, estimates a nonincreasing mixture density, computes
`PEP-like(p)=pi0/f_hat(p)`, and enforces nondecreasing PEP with worsening p.
When the fit population and application population differ, the curve is fitted
only on the declared reference rows and monotonically interpolated to all
rows. This prevents target/entrapment membership from entering the density fit.

### Aggregation and Level 4

Only the finalized rank-1 active stream enters aggregation. Canonical peptide
keys remove bracketed modifications and canonicalize I/L. Peptidoform keys
retain bracketed modifications and canonicalize unmodified I/L. Protein
evidence uses the inferred protein key, with optional parsimonious grouping a
separate setting. P-native evidence is combined only by declared p combiners;
PEP-native evidence uses the declared best/support rule and cumulative-mean
interpretation.

Level 4 is a hierarchical reporting layer. In strict mode, reportable PSMs and
peptides must independently pass their native q thresholds and the documented
peptide/protein support conditions. In protein-primary mode, lower-level rows
are protein-supported observations rather than independent lower-level
discoveries. Reporting thresholds are not fit parameters.

The combined entrapment estimate at every level is

`FDP_combined = E * (1 + 1/r) / (T + E)`.

The measured peptidoform ratio is stored in the workflow’s `psm` ratio field
and is used for PSM/peptidoform space; canonical peptide and protein use their
own measured ratios. Missing, zero, or nonfinite ratios are not evaluable.

### Null-window objective

Each candidate window is evaluated at a fixed optimizer evaluation threshold.
It is feasible only when PSM, peptide, and protein combined entrapment FDP are
all evaluable and below the declared maximum. The deterministic objective is
lexicographic: maximize target proteins, then target peptides, then target PSMs;
then minimize protein, peptide, and PSM FDP; then prefer narrower/lower windows.
The evaluation threshold does not tune the reporting FDR thresholds. Once
chosen, the window is immutable for target-only application.

## Model contracts and findings

### Moments — A/B with an optional C mode

Input is the model-local purified lower-rank hyperscore pool. The ordinary fit
uses population variance:

`beta = sqrt(6 Var(X))/pi`, `mu = mean(X)-gamma beta`.

Rank-1 p-values are upper-tail Gumbel probabilities. Companion PEP-like values
use reference-population Storey/Grenander calibration; q-values are p-native.
The optional winsorized fit is an intentional contamination-control mode and
analytically corrects the standard-Gumbel winsorization moments. It is not the
ordinary unmodified method. Nondegenerate variance and `min_null_size` are
required. Synthetic recovery passed for `mu=7.5`, `beta=1.8`, `n=20,000`.

Target-only locked-window refit is coherent when lower-rank observations remain
the declared null. Identity-bound artifact application is a distinct coherent
fixed-model question. Neither can change the selected window.

### MLE — A/B with an optional C mode

Input and output populations match Moments. Sage solves the standard Gumbel
profile-likelihood equation

`beta-xbar + sum(x exp(-x/beta))/sum(exp(-x/beta)) = 0`

with centered log-sum-exp weights, then applies the closed-form location update.
The centering is class B numerical stabilization. Invalid beta, nonconvergence,
or degenerate input makes the expert unavailable. Optional preprocessing by
winsorization changes the likelihood target and is correctly declared as a
sensitivity mode, not canonical MLE. The same synthetic recovery criterion
passed.

### Lower Order — A equations, C/D/E Sage estimator

For each spectrum Sage constructs

`E_LO = p_tail * N_candidates^candidate_count_power * evalue_scale`

and canonically `TEV=-ln(E_LO)`. For rank `k`, `z=(x-mu_k)/beta_k` and the
asymptotic density is

`f_k(x)=exp(-kz-exp(-z))/(beta_k (k-1)!)`.

The TEV-k density, CDF, moments, and likelihood match the paper/PyLord (A/B).
Sage’s production top-null estimator is not PyLord’s exact procedure: it fits
per-rank LOMs and performs deterministic local linear extrapolation of the
nearest supported `(mu_k,beta_k)` values to rank 1 (E). Charge stratification,
stable ordering, complete portable state, and fail-closed missing-charge
behavior are C/D modifications for small datasets. At least two supported
ranks and `lo_min_count_per_rank` observations per rank are mandatory.

The predeclared TEV simulations for ranks 2, 3, 5, and 8 recovered known
parameters within the finite-sample tolerances. Candidate-count scaling exactly
matched the declared equation.

`refit_with_locked_window` is a valid interpretation: the method’s nuisance
state is re-estimated in the changed target candidate space while the
entrapment-selected window remains fixed. Complete-artifact reuse across that
space is not valid because `p_tail`, candidate count `N`, and therefore the
TEV population change with the searched database. The workflow correctly
rejects that policy.

### MSFDR seeded — C/E, not the Peng 1SMix algorithm

The lower-rank window seeds a fixed Gumbel null; rank-1 scores fit a
skew-normal target component with deterministic EM-like responsibilities and
weighted-moment updates. Native p-values are Gumbel null survival; PEP is the
fitted posterior null responsibility. This is a coherent project-specific
two-component extension, but it is not an exact implementation of either Peng
1SMix or 2SMix. No primary source was found for its exact fixed-Gumbel plus
skew-normal update sequence (G). It must therefore be described by this
current-method contract, not by paper parity.

### MSFDR1-SMIX — A structure, C/E fitting, corrected F identifiability

The rank-1-only model is fixed at one correct and one incorrect component:

`S1 ~ a SN(C) + (1-a) SN(I1)`.

Its local PEP, incorrect-component survival p-value, and paper Eq. 4 tail-area
FDR are algebraically correct (A). Unlike the paper’s latent truncated-normal
EM, Sage searches deterministic bottom/top fractions and skew signs, then uses
weighted moments and upper-tail orientation (C/E). This can reasonably be
conservative: the single incorrect component must absorb all rank-1 incorrect
heterogeneity, bounded `a` limits optimistic fits, and upper-tail sanity
penalizes reversed components. Conservatism is not itself misspecification.

The original conformance fixture proved a defect: 100 identical rank-1 scores
returned a model. Component separation was zero, so mixture weights/posteriors
were not identified (F). The shared gate specified below now rejects that fit
before trial ranking, artifact serialization, or probability production.

### MSFDR2-SMIX — A structure, C/E pooled extension, corrected F identifiability

The paper uses one second-best score per spectrum. Sage uses a selectable
pooled lower-rank population and an `S2` balance factor:

`S1 ~ a C + (1-a) I1`

`pooled S2 ~ (a*s) I1 + (1-a*s-b) I2 + b C`, `s=min(n1/n2,1)`.

The rank-1 PEP and `SF_I1` p-value meanings remain coherent. Pooling ranks and
diluting the shared I1 prior are project-specific low-input extensions (C/E),
not Peng’s exact likelihood. Exact fourfold replication of the same pooled S2
rows changed fitted weights by less than the predeclared 0.03 tolerance.

Lower power can be expected when I1/I2 overlap, pooled ranks are heterogeneous,
or the additional component consumes limited information; it is not evidence
of misspecification by itself. Coincident rank-1 and S2 populations formerly
returned a model, reproducing the same nonidentifiability defect as 1SMix (F).
Each pooled-window trial now passes the common gate before it is eligible for
window ranking; failure of one trial does not invalidate another prospectively
configured trial.

### Nokoi v2 — C/D/E, not a 47-feature source-method port

The source Nokoi is an externally trained 47-feature L1 logistic classifier
using Mascot ranks. Sage Nokoi v2 is a dataset-local 12-feature model with
pseudo-positive rank-1 examples, purified lower-rank null examples,
portable normalization, L1 FISTA, deterministic lambda selection, and stable-ID
fold assignment. This is an intentional extension/name lineage (E), motivated
by the absence of a compatible external training corpus and the variance/leakage
risks of ultra-low-input data. Stable-ID folding is D.

Each held-out null is scored by a model that did not train on it. OOF null
scores define a smoothed empirical upper-tail p-value. Storey pi0 and frozen
Grenander blocks define the PEP-like curve. The artifact includes feature order,
normalization, all fold/final weights and intercepts, lambda evaluations,
stable population identity, OOF scores, pi0, calibration curve, and block
hashes. Existing deterministic fixtures verify fold assignment, permutation
invariance, exact replay, relocation, integrity failure, and explicit
application modes.

Locked-window refit is coherent as a new local crossfit. Complete-artifact
reuse is also coherent as a different fixed-model interpretation only for the
same parent dataset, with the exact parent fingerprint and declared
`SameDatasetTargetOnly` mode. Neither interpretation may use target-only
results to retune the window. Counts alone do not select between them.

### Ensemble — D/E consensus, not expert admission

For finite expert p-values sorted `p_(1)<=...<=p_(m)`, `second_best` returns
`p_(2)` for `m>=2`, `p_(1)` for `m=1`, and 1 for `m=0`. Median PEP returns the
ordinary sample median of finite PEP-like values and 1 when none exist. Both
are deterministic and order invariant. All JSON-selected technically valid
experts contribute; invalid/missing values are omitted; no biological-count or
calibration gate changes participation.

Under exact duplicate experts, second-best and median reproduce the duplicate
value, so duplication increases that evidence stream’s effective influence.
Under independent experts, raw second-best has a beta order-statistic null,
not a uniform null. Under correlated experts its null depends on the joint
copula. Median PEP similarly has no posterior meaning without a joint error
model. Absence of evidence maps to 1; a weak expert changes the consensus
continuously according to the selected combiner. The methods are valid as
explicitly named consensus scores and must not be advertised as calibrated
probabilities without rank-null or external calibration.

## Ultra-low-input modifications

| Modification | Problem | Mathematical change | Expected advantage | Calibration risk / assumptions | Executable evidence | Status |
|---|---|---|---|---|---|---|
| Model-local purified null windows | target contamination and heterogeneous ranks | select/purify each model’s retained lower-rank population | lower bias and model-specific support | selection must be entrapment-evaluable and locked | null-window deterministic tests | retained |
| Bias-corrected winsorized Moments | upper-tail contamination | clamp fixed quantiles and invert the winsorized standard-Gumbel moments | robust location/scale at low input | contamination model and fixed quantiles must be appropriate | `winsorized_moments_remove_gumbel_location_scale_bias` | retained, optional |
| Winsorized MLE sensitivity mode | contamination | fit the likelihood to clamped observations | robustness diagnostic | no longer canonical MLE likelihood | configuration contract | questionable as a primary fit; retained as explicit sensitivity |
| Lower Order spectrum-local E-value | search-space-dependent candidate burden | `p_tail*N^power*scale` before TEV | comparable spectrum evidence | power/scale require calibration; state cannot transfer blindly | candidate-count scaling test | retained |
| Lower Order local LOM extrapolation | sparse ranks make global relations unstable | extrapolate nearest supported `(mu_k,beta_k)` to rank 1 | lower variance/locality | extrapolation bias and beta boundary | TEV recovery and artifact tests | retained |
| Charge-stratified Lower Order with deterministic fill | charge-dependent score null and sparse strata | separate fits; nearest fitted charge fallback | reduces pooling bias | sparse-stratum variance; fallback provenance | artifact/fail-closed tests | retained |
| Seeded MSFDR fixed null | weak joint identifiability | freeze lower-rank Gumbel while fitting target | reduces free mixture dimensions | misspecified seed biases posterior | bounds/monotonicity tests | retained with source ambiguity |
| Bounded mixture weights, multistarts, and post-fit validity | boundary collapse/local optima/nonidentifiability | clamps plus deterministic initialization grid and a scale-aware validity gate | stable low-input optimization with fail-closed invalid trials | imposed prior-like bounds can bias calibration | active mixture validity and provenance tests | retained; gate verified |
| Pooled-rank 2SMix | rank 2 alone may be sparse | pool selected ranks and dilute I1 prior by `n1/n2` | more S2 observations | heterogeneous ranks, non-paper likelihood | replication-invariance test | retained, project extension |
| Dataset-local Nokoi crossfit | no compatible external training artifact | stable-ID K-fold OOF null scoring and local pseudo-labels | portable and leakage-controlled | pseudo-label selection bias; enough fold support | Nokoi fold/replay tests | retained |
| Robust Ensemble consensus | correlated experts and one optimistic expert | median PEP / second order p statistic | continuity and outlier resistance | outputs lose formal posterior/uniform meaning | analytic second-best test | retained with explicit PEP-like/p-like wording |
| Storey high-lambda median and clamps | unstable pi0 at small n | robust grid aggregation and bounded pi0 | lower variance/fail-safe | bias from clamps and sparse tail | fallback/bounds tests | retained |
| Measured-ratio entrapment correction | different effective hypothesis counts by level | level-specific `E(1+1/r)/(T+E)` | corrects unequal target/entrapment space | equal-chance analogue and sufficient counts | ratio-level test | retained |

## Executable conformance evidence

The focused test name prefix is `statistical_conformance`. The deterministic
plan fixes seeds, population sizes, effect distributions, and tolerances before
execution. New passing tests cover Gumbel and TEV-k recovery, bounds,
q monotonicity and ties, Storey fallback provenance, sparse/zero-null behavior,
candidate-count scaling, measured-ratio correction, pooled-S2 replication,
Ensemble order/duplicate/continuity behavior, and the analytic second-best
counterexample. Existing Nokoi portable-v2 tests cover fold determinism,
train/application separation, artifact replay, integrity, and population
identity. Existing null-window/workflow tests cover deterministic selection,
locked-window provenance, unsupported Lower Order reuse, and non-evaluable
handling.

The regression test
`statistical_conformance_coincident_mixture_components_fail_closed` is the
minimal reproduction of the behavior-changing defect. Its original ignored
execution failed with both `1SMix_returned_model=true` and
`2SMix_returned_model=true`; it is now active and passes. Additional active
tests cover nonfinite input, constant nonzero input, invalid scales and
weights, ineffective support, coincident and numerically indistinguishable
components, valid separated and low-variance fits, determinism, canonical
labels, explicit provenance, and absence of probability output after failure.

The targeted ISB cache-only regression ran six MSFDR1/2 stages with no external
execution, no search, and no fallback. All six fitted artifacts and all six
`results.sage.tsv` tables were byte-identical to the frozen baseline. The
MSFDR2 selected window remained ranks 9--17; only elapsed-time metadata and
output-directory URIs differed in workflow summaries.

## Per-model final classification

| Model | Source-method conformance | Ultra-low-input validity | Probability/calibration interpretation | Target-only interpretation | Remaining ambiguity / defect | Step 3 |
|---|---|---|---|---|---|---|
| Moments | exact standard Gumbel moments | valid; robust mode explicit | valid fitted-null p; empirical PEP-like | refit/reuse distinct, window locked | null exchangeability is empirical | ready |
| MLE | numerically stable standard Gumbel MLE | valid; winsor mode only sensitivity | valid fitted-null p; empirical PEP-like | same as Moments | optimizer/model adequacy is empirical | ready |
| Lower Order | exact TEV-k equations, Sage-specific TNM estimator | coherent and tested under stated assumptions | fitted extrapolated-null p; empirical PEP-like | locked refit valid; cross-space complete reuse invalid | local extrapolation lacks direct source validation | ready with named limitation |
| MSFDR seeded | project-specific, not Peng exact | coherent fixed-null dimension reduction | model p/posterior conditional on seed/model | both explicit locked interpretations coherent | exact source rationale unavailable | ready with limitation |
| MSFDR1-SMIX | paper mixture/evidence equations; different estimator | deterministic changes coherent; validity gate active | p and posterior conditional on identifiable fit | explicit locked interpretations coherent | estimator differs from author EM | ready with named limitation |
| MSFDR2-SMIX | paper structure; pooled-rank extension | plausible, replication-stable, validity gate active | p/posterior conditional on pooled model | window must remain locked | pooled likelihood is a Sage extension | ready with named limitation |
| Nokoi v2 | intentional extension, not paper port | coherent leakage-controlled redesign | empirical OOF p and PEP-like calibration | both modes coherent with identity guards | pseudo-label calibration remains empirical | ready with named limitation |
| Ensemble | project-specific consensus | coherent robust ranking | second-best is p-like, median is PEP-like; neither automatically calibrated | reuses/refits constituent states per declared policy | joint dependence not modeled | ready with interpretation limits |

## Preregistered MSFDR mixture-validity correction

Let `r = 64 * f64::EPSILON * max(max_abs_score, score_range, 1)`. The multiplier
covers the fixed binary64 subtraction, moment, square-root, and skew-normal
transforms. It is a numerical-resolution allowance, not a biological or
identification-count-derived cutoff.

A valid fit requires all supplied scores to be finite; at least 20 observations
in every required population; observed range and stable population standard
deviation both greater than `r`; finite component and weight parameters; every
skew-normal scale greater than `r`; and all required mixture weights greater
than `64*f64::EPSILON`. Every three-parameter skew-normal component must have
expected aggregate support of at least three observations. Every semantic
component pair must differ in location, scale, or shape by more than its
corresponding 64-epsilon parameter-resolution tolerance. 1SMix retains its
existing deterministic upper-tail orientation. The 2SMix C/I1/I2 labels are
fixed by their non-exchangeable joint equations and deterministic
initialization. A mean-order restriction is deliberately not added: the author
implementation defaults to unconstrained component locations, so such a rule
would change the estimator rather than merely validate it.

No likelihood-ratio threshold is added. Neither Peng et al. nor the author
implementation specifies one, and a data-tuned threshold would change the
estimator. A valid mixture passes all rules and is returned unchanged. The two
declared methods do not emit reduced models: a boundary-weight solution is a
technical failure. A nonidentifiable fit, an optimizer failure, and workflow
fallback are distinct provenance outcomes. Nonidentifiable fits produce no
artifact or p-value/PEP/q-value stream, and no silent alternate model is used.

## Step 3 parameter boundary

`validation/statistical_conformance/parameter_catalog.json` records current
values, mathematical domains, bounded future spaces, identities, cache reuse,
dependencies, risks, and cost. The three reporting FDR thresholds are excluded.
MSFDR mixture parameters are eligible now that the gate and targeted regression
pass. Eligibility is catalog metadata, not optimization; this pass implements
no optimizer and changes no reporting threshold.
