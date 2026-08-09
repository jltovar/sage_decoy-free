# Sage: Decoy-Free Edition (Experimental)

> **Experimental research fork.** This is a decoy-free FDR branch of the Sage search engine. APIs, configuration options, output semantics, and statistical behavior may change as the workflow is refined.

This branch adds an explicit **Decoy-Free (DF) FDR** mode to Sage. It is intended for low-input and ultra-low-input proteomics experiments where conventional target-decoy competition (TDC/TDA) can become statistically coarse or underpowered, especially at the peptide-to-protein inference stage.

The central goal is to increase biologically useful identifications, especially proteins, without allowing uncontrolled false identifications. The implementation therefore separates:

1. **Base DF evidence** from fitted score/null models.
2. **Optional physical evidence updates** from retention time (RT) and ion mobility (IMS).
3. **Optional reproducibility rescue** from expert agreement and cross-run recurrence.
4. **Reporting-only hierarchical inference**, which can report protein-supported peptides/PSMs without overwriting the active DF q-value streams.

The current implementation should be read as a research/statistical validation branch, not as a drop-in replacement for all Sage production workflows. In this fork, some standard Sage output modes may be incomplete or not validated with the decoy-free path, including LFQ, TMT, PIN output, annotate-matches output, and parquet output. Re-validate these modes before relying on them.

--
## Required validation-first workflow

Decoy-free FDR estimation is model- and dataset-dependent. There is no universally correct null-rank window, model-fit choice, or rescue setting that can be assumed to work across datasets. Ultra-low-input proteomics is especially sensitive to these choices because the number of accepted PSMs, peptides, and proteins is small, and a poorly calibrated null model can either suppress true identifications or inflate false discoveries.

For this reason, the decoy-free workflow should be treated as a validation-first, two-run process:

1. Optimization / validation run with entrapment enabled
    First, run the dataset against a target-plus-foreign-entrapment FASTA and use the optimization scripts to search null-rank windows and model-fit settings. The goal is not simply to maximize target identifications. The goal is to find the highest-yielding configuration that keeps entrapment identifications controlled, especially at the protein level.
    
2. Final production run without entrapment
    After the null window and model settings have been selected using entrapment validation, rerun the analysis on the intended production FASTA without the foreign-entrapment sequences. The production run should use the validated parameters from the entrapment optimization run.

This makes decoy-free analysis more computationally expensive than a single standard TDA/TDC run. The first stage may require many Sage runs across candidate null-rank windows and model fits. This cost is intentional: higher discovery power is only meaningful if it is checked against entrapment or another external validation strategy.

Do not assume that a null-rank window optimized for one dataset, instrument, acquisition method, sample amount, or model fit will transfer to another dataset. A configuration that is well calibrated for one ultra-low-input experiment may be too conservative or too liberal for another. Every new dataset should be validated before decoy-free results are interpreted biologically.


## Entrapment and null-window optimization scripts

This repository includes helper scripts for the validation-first decoy-free workflow.

**make_entrapment.sh**

make_entrapment.sh builds a foreign-species entrapment FASTA using **FDRBench** (https://github.com/Noble-Lab/FDRBench/tree/main/src/main/java/FDR). It combines a target FASTA with a foreign FASTA and labels the foreign entries with an entrapment prefix, such as Ent_, so that Sage output can later separate true target identifications from forward-entrapment identifications.

The script:

1. runs the FDRBench JAR using the supplied target and foreign FASTA files;
2. supports protein-level or peptide-level entrapment generation;
3. applies digestion parameters such as enzyme, missed cleavages, minimum peptide length, and maximum peptide length;
4. optionally applies I/L normalization and removal of shared peptides;
5. post-processes FASTA headers so entrapment entries are consistently labeled;
6. writes both the raw and final entrapment FASTA plus a log file.

The resulting FASTA is used only for validation and optimization. It should not be used for the final biological production run unless the purpose of the run is continued entrapment validation.

**optimize_null_window.sh**

optimize_null_window.sh searches candidate null-rank windows for a selected decoy-free model fit. It repeatedly edits a base Sage JSON file, runs Sage for each candidate window, parses the resulting logs and results.sage.tsv, and records target and entrapment counts at the PSM, peptide, and protein levels.

The script is designed for entrapment-first optimization. It tracks both raw decoy-free discovery counts and Level 4 hierarchical reporting metrics, including:

decoy_free_is_entrapment
decoy_free_protein_supported_peptide
decoy_free_peptide_supported_psm

It writes cumulative result tables and a best-result summary, including:

cumulative_results_long.csv
cumulative_results_wide.csv
best_result.txt
search_trace.txt

The current optimizer is protein-primary. By default, feasibility and optimum eligibility are driven by protein-level entrapment control, while Level 4 peptide and PSM counts are used to rank yield among protein-safe configurations. This matches the intended use case for ultra-low-input proteomics, where the primary biological objective is a reliable protein list rather than maximal raw PSM count.

In practical terms, the optimizer asks:

Among the null-rank windows that keep entrapment proteins at zero or below the configured threshold, which window gives the best protein-supported peptide and PSM recovery?

The selected null window should then be copied into the final Sage JSON and used for the production run without entrapment sequences.

### Native Phase 2 workflow

The Rust workflow now internalizes the FASTA-generation and null-window-optimization boundary.
These two parts have independent parity gates: optimizer tests can consume the exact frozen legacy
entrapment FASTA, while `sage audit-entrapment entrapment.audit.json` compares native FASTA
generation with a `make_entrapment.sh`/FDRBench reference without running a spectral search.

Native generation supports automatic foreign-source selection, an explicit source, and automatic
selection with a user override. It records selected accessions, shared-peptide exclusions, source
mappings, order-sensitive hashes, seed reproducibility, and Sage-measured protein, peptide, and
peptidoform ratios. Workflow searches require the ratios measured from their own active FASTA;
hard-coded ratios from another dataset are rejected.

For legacy comparison only, `fdrbench004_compatibility` reproduces FDRBench 0.0.4's length-only
shared-peptide exclusion and Java seeded selection. New production workflows default to
`sage_search_space`, which uses the peptides Sage can actually search. See
[`DECOY_FREE_WORKFLOW.md`](DECOY_FREE_WORKFLOW.md) and
[`entrapment.audit.example.json`](entrapment.audit.example.json).

### Native Phase 3 candidate sharing and Phase 4 MS2Rescore caching

`sage workflow` now performs one native spectrum search per strict search fingerprint and persists
the pre-FDR candidates in a compressed, immutable pool. Compatible model fits and null-window
grids reuse that pool while receiving separate analysis fingerprints. Statistical changes such as
q-value methods, covariates, Storey parameters, FDR thresholds, RT/IMS adjustments, rescue gates,
or null windows therefore trigger a refit but not another spectrum search. FASTA, spectrum,
digestion/modification, tolerance, scoring, preprocessing, or retained-depth changes select a new
pool automatically.

The exact lean optimizer retains compact metrics and timing for every trial, drops nonselected
feature/artifact payloads immediately, and materializes the selected window once with full normal
diagnostics. Stable candidate IDs now support a separate, compressed MS2Rescore annotation cache.
The MS2Rescore stage reuses the native candidate pool, while its external annotations remain
outside that immutable pool and are reused only when the search, preliminary q/PEP calibration
input, rank depth, generator settings, mapped spectra, wrapper, and Python/package environment all
match. A changed model/window calibration receives a different annotation cache, preserving exact
DeepLC behavior. See [`DECOY_FREE_WORKFLOW.md`](DECOY_FREE_WORKFLOW.md) for fingerprint,
capability, integrity, and fail-closed rules.

### Phase 5-7 calibration and parity status

Target-only calibration is now explicit. The default `refit_with_locked_window` policy keeps the
dataset-local window selected with entrapment and refits nuisance parameters in the target-only
candidate space; `reuse_dataset_artifact` reuses the complete same-dataset fit, and `compare_both`
keeps both interpretations separate. Cross-dataset artifact reuse remains prohibited by default.

Frozen ISB model-by-model parity and the required independent PXD001468 Moments parity have been
completed. PXD evaluated all 47 valid frozen Moments windows and selected the exact legacy
`10-10` window. Optimized counts were exact, MS2Rescore and target-only counts were within the
predeclared 0.5% platform tolerance, and an end-to-end cache-hit rerun performed no new spectrum
search or external feature generation. PXD Moments is the only required PXD model for this
refactor; additional PXD models are optional. See [`DECOY_FREE_WORKFLOW.md`](DECOY_FREE_WORKFLOW.md)
and `validation/reports/phase7_pxd001468_moments_parity_2026-08-08.json` for the caveats and full
evidence.


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

Direct one-off Ensemble searches still honor the static enable/disable flags and weights in the search JSON. The native `sage workflow` path adds automatic, fail-closed expert-quality gates and writes a dataset-local expert set for development or holdout validation. Holdout datasets run the same predeclared optimization procedure; they do not import another dataset's selected windows or fitted experts.

Direct one-off searches use configured null-rank windows. The native `sage workflow` path can scan declared candidate windows in memory from one retained candidate set, select the highest-yield feasible window, and lock it for later stages.

See [`DECOY_FREE_WORKFLOW.md`](DECOY_FREE_WORKFLOW.md) and [`workflow.example.json`](workflow.example.json) for the resumable development/holdout workflow and validation-only audits of completed result tables.

The current implementation exposes `msfdr2_smix_min_null_rank` and `msfdr2_smix_max_null_rank`, but there are no `msfdr1_smix_min_null_rank` or `msfdr1_smix_max_null_rank` options in `FdrOptions`. MSFDR1_SMIX is a rank-1 semi-mixture path controlled by initialization fractions and pi clamps. Do **not** put `msfdr1_smix_min_null_rank` or `msfdr1_smix_max_null_rank` in the JSON config; they are not accepted by the current code.

For `model_fit = "ensemble"`, `final_evidence_space` must be explicitly set to either `"p_value"` or `"pep"`. The code rejects `final_evidence_space = "auto"` in ensemble mode.

---

## Installation and basic usage

This fork is built from source using the Rust toolchain. The Python environment is optional for running Sage itself, but it is necessary for running the validation scripts, plotting, and downstream analysis.

## Optional: installing `pyenv` on Ubuntu / WSL

Sage itself is built with Rust and does not require Python. A Python environment is useful for validation scripts, plotting, and downstream analysis. One convenient option is `pyenv`.

### 1. Install build dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev \
  libxmlsec1-dev libffi-dev liblzma-dev
```

### 2. Install `pyenv`

```bash
curl https://pyenv.run | bash
```

### 3. Add `pyenv` to your shell

For Bash/WSL, add the following lines to `~/.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

# Initialize pyenv and pyenv-virtualenv.
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Then reload the shell:

```bash
source ~/.bashrc
exec "$SHELL"
```

### 4. Create the Sage validation Python environment

```bash
pyenv install 3.12.0
pyenv virtualenv 3.12.0 SAGE_decoy_free_github
pyenv activate SAGE_decoy_free_github
```

Confirm Python is active:

```bash
python --version
which python
```

Do not run `sudo apt-get upgrade -y` as part of this README setup unless you intentionally want to upgrade system packages. It is not required for installing `pyenv`.

### 1. Set up the local environment

```bash
# Optional Python environment for validation/plotting scripts
pyenv install 3.12.0
pyenv virtualenv 3.12.0 SAGE_decoy_free_github
pyenv activate SAGE_decoy_free_github

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Confirm toolchain
rustc --version
cargo --version
```

### 2. Clone and build the decoy-free fork

```bash
git clone https://github.com/jltovar/sage_decoy-free sage_decoy_free_github
cd sage_decoy_free_github

# Switch to the decoy-free development branch
git checkout decoy-free

# Format and build
cargo fmt
cargo build --release

# Confirm executable
./target/release/sage --help
```

After building, the executable is located at:

```bash
./target/release/sage
```

A typical absolute path on WSL/Linux is:

```bash
$HOME/sage_decoy_free_github/target/release/sage
```

---

## Running Sage

Sage is run by passing a JSON parameter file to the compiled executable:

```bash
RUST_BACKTRACE=1 SAGE_LOG=trace /path/to/sage /path/to/search_config.json
```

For deeper Rust panic diagnostics, use:

```bash
RUST_BACKTRACE=full SAGE_LOG=trace /path/to/sage /path/to/search_config.json
```

### Standard Sage / vanilla TDA-TDC run

Use the upstream or vanilla Sage executable when generating a standard TDA/TDC baseline:

```bash
RUST_BACKTRACE=1 SAGE_LOG=trace \
  "$HOME/sage_master/target/release/sage" \
  /path/to/vanilla_validation_ISB18_ent_decoy.json
```

```bash
RUST_BACKTRACE=1 SAGE_LOG=trace \
  "$HOME/sage_master/target/release/sage" \
  /path/to/vanilla_validation_PXD001468_ent_decoy.json
```

### Target-decoy mode using the decoy-free fork

The decoy-free fork can also be run in standard target-decoy mode, depending on the `fdr.mode` value in the JSON file:

```bash
RUST_BACKTRACE=full SAGE_LOG=trace \
  $HOME/sage_decoy_free_github/target/release/sage \
  /path/to/VANILLA_SAGE/validation_ISB18_ent_decoy.json
```

### Decoy-free model-fit runs

Each decoy-free run is controlled by the `fdr.model_fit` setting in the JSON file. Common model-fit values include:

```text
moments
mle
lower_order
msfdr
msfdr_1smix
msfdr_2smix
nokoi
ensemble
```

Example Moments run:

```bash
RUST_BACKTRACE=1 SAGE_LOG=trace \
  $HOME/sage_decoy_free_github/target/release/sage \
  /path/to/MOMENTS/validation_ISB18_ent_decoyfree.json
```

Example MLE run:

```bash
RUST_BACKTRACE=1 SAGE_LOG=trace \
  $HOME/sage_decoy_free_github/target/release/sage \
  /path/to/MLE/validation_ISB18_ent_decoyfree.json
```

Example LowerOrder run:

```bash
RUST_BACKTRACE=1 SAGE_LOG=trace \
  $HOME/sage_decoy_free_github/target/release/sage \
  /path/to/LOWER_ORDER/validation_ISB18_ent_decoyfree.json
```

Example MSFDR run:

```bash
RUST_BACKTRACE=1 SAGE_LOG=trace \
  $HOME/sage_decoy_free_github/target/release/sage \
  /path/to/MSFDR/validation_ISB18_ent_decoyfree.json
```

Example NOKOI run:

```bash
RUST_BACKTRACE=1 SAGE_LOG=trace \
  $HOME/sage_decoy_free_github/target/release/sage \
  /path/to//NOKOI/validation_ISB18_ent_decoyfree.json
```

Example Ensemble run:

```bash
RUST_BACKTRACE=1 SAGE_LOG=trace \
  $HOME/sage_decoy_free_github/target/release/sage \
  /path/to//ENSEMBLE/validation_ISB18_ent_decoyfree.json
```

The same pattern can be repeated for other validation datasets by changing the JSON path.

---

## Validation helper scripts

The repository includes helper scripts for entrapment generation and null-window optimization. These scripts are intended for validation and parameter selection before final production runs.

### Normalize script line endings

If the scripts were edited on Windows, first normalize line endings:

```bash
sed -i 's/\r$//' /path/to/optimize_null_window.sh
sed -i 's/\r$//' /path/to/make_entrapment.sh
```

Then make them executable:

```bash
chmod +x /path/to/optimize_null_window.sh
chmod +x /path/to/VALIDATION/make_entrapment.sh
```

---

## Building a foreign-entrapment FASTA

Use `make_entrapment.sh` to build a target-plus-foreign-entrapment FASTA for validation.

Example for ISB18:

```bash
/path/to/make_entrapment.sh \
  --dataset isb18 \
  --target-db /path/to/ISB18.fasta \
  --foreign-db /path/to/foreign_entrapment/uniprotkb_proteome_UP000008311_2026_04_12.fasta \
  --output-dir /path/to/target_entrapment \
  --output-prefix ISB18_foreign_species_entrapment
```

The resulting FASTA should be used for entrapment validation and null-window optimization. After a model fit and null window are selected, rerun the final production analysis without the foreign-entrapment sequences.

---

## Optimizing the null-rank window

Use `optimize_null_window.sh` to search null-rank windows for a given decoy-free model fit. This is the recommended first-pass validation step before running decoy-free mode on a production FASTA.

Example for ISB18 Moments:

```bash
/path/to/optimize_null_window.sh \
  --model-fit moments \
  --base-json /path/to/MOMENTS/validation_ISB18_ent_decoyfree.json \
  --results-root /path/to/ISB18/DECOYFREE/MOMENTS
```

The optimizer repeatedly modifies the decoy-free null-window parameters, runs Sage, parses the output, and records target and entrapment recovery. The goal is to find a model-specific null window that maximizes useful target recovery while keeping entrapment controlled, especially at the protein level.

Typical outputs include:

```text
cumulative_results_long.csv
cumulative_results_wide.csv
best_result.txt
search_trace.txt
```

Use the selected null-window parameters from `best_result.txt` in the final production JSON.

---

## Recommended two-run workflow

Decoy-free analysis should be run as a two-stage workflow:

1. **Entrapment optimization run**  
   Use a target-plus-entrapment FASTA and run `optimize_null_window.sh` to identify a calibrated model/null-window configuration.

2. **Production run**  
   Remove the entrapment sequences, keep the validated parameters, and rerun Sage on the intended biological FASTA.

This is more computationally expensive than a single TDA/TDC search, but it is necessary because decoy-free model calibration is dataset-dependent. Higher target recovery is only meaningful if it remains controlled by entrapment or another external validation strategy.


## Quick start

A minimal DF configuration uses the `fdr` block inside the Sage JSON configuration:

```json
{
  "fdr": {
    "mode": "decoy_free",
    "model_fit": "ensemble",
    "final_evidence_space": "p_value",
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

## Recommended full example configuration

The following block is a cleaned, code-consistent broad configuration for ultra-low-input validation runs. It uses only fields accepted by the current `FdrOptions` structure. It also shows a conservative ensemble choice in which the two SMIX experts are enabled for diagnostics but assigned zero ensemble weight, because these mixture paths can be conservative or unstable on sparse datasets. This is a **recommendation**, not the code default: if the `ensemble_weight_*` keys are omitted, the current defaults are `1.0`.

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
    "msfdr_multistart": 2,
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

    "ensemble_p_combiner": "cauchy",
    "ensemble_cauchy_penalty": 1.0224,
    "ensemble_pep_combiner": "median",
    "ensemble_pep_trim_frac": 0.5,
    "ensemble_pep_quantile": 0.5,
    "ensemble_pep_top_k": 5,
    "ensemble_pep_logit_eps": 1e-6,

    "ensemble_weight_moments": 1.0,
    "ensemble_weight_mle": 1.0,
    "ensemble_weight_lower_order": 1.0,
    "ensemble_weight_msfdr_seeded": 1.0,
    "ensemble_weight_msfdr_1smix": 0.0,
    "ensemble_weight_msfdr_2smix": 0.0,
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
| `mode` | `tdc`, `decoy_free` | `decoy_free` | Selects conventional TDC or DF mode. |
| `entrapment_report` | `off`, `auto`, `on` | `auto` | Controls entrapment reporting behavior. |
| `model_fit` | `moments`, `mle`, `lower_order`, `msfdr`, `msfdr1_smix`, `msfdr2_smix`, `nokoi`, `ensemble` | `ensemble` | Selects the base DF expert or ensemble. |
| `final_evidence_space` | `auto`, `p_value`, `pep` | `auto` | Ensemble requires explicit `p_value` or `pep`. |
| `peptide_p_combine` | `fisher`, `cauchy`, `sidak_min_p`, `best`, `second_best` | `cauchy` | Peptide-level p-value combination. |
| `protein_p_combine` | `fisher`, `cauchy`, `sidak_min_p`, `best`, `second_best` | `cauchy` | Protein-level p-value combination. |
| `psm_q_method` | `auto`, `bh`, `storey`, `cummean` | `storey` | PSM-level q-value method. |
| `peptide_q_method` | `auto`, `bh`, `storey`, `cummean` | `auto` | Peptide-level q-value method. |
| `protein_q_method` | `auto`, `bh`, `storey`, `cummean` | `auto` | Protein-level q-value method. |
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
| `lo_evalue_scale` | float | `1.0` | Multiplicative E-value scale, clamped to `[1e-6, 1e6]`. |
| `lo_tev_transform` | `neg_log_e`, `log_1000_over_e`, `scaled_log_1000_over_e` | `neg_log_e` | TEV transform. |
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
log_1000_over_e        => TEV = ln(1000 / E_LO)
scaled_log_1000_over_e => TEV = 0.02 * ln(1000 / E_LO)
```

### MSFDR seeded model

| Key | Type | Current default | Notes |
|---|---:|---:|---|
| `msfdr_min_null_rank` | integer | `4` | Method-specific lower null rank. |
| `msfdr_max_null_rank` | integer | `50` | Method-specific upper null rank. |
| `msfdr_seeded_purification_factor` | float | `0.25` | Null-pool purification factor. |
| `msfdr_seeded_top_frac_init` | float | `0.20` | Initial top fraction, clamped by the generic fraction helper. |
| `msfdr_multistart` | integer | `3` | Number of multistart fits, clamped to `[1, 25]`. |
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

Ensemble mode combines enabled expert streams.  Explicit model-fit modes ignore the enable flags because the selected model is required to run and fail closed if it cannot produce a valid stream.

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

### MSFDR variants

The MSFDR family produces fitted or empirical null-survival p-like streams and derived PEP-like streams.  The SMIX variants use mixture-model fitting and can be sensitive to initialization and rank-window choices.

### NOKOI

NOKOI uses a cross-fit classifier-like approach with lower-rank null evidence and high-scoring rank-1 positives.  It is often powerful but should be audited carefully for early entrapment behavior and for feature-driven overconfidence.

### Ensemble

The ensemble is designed to combine evidence from multiple experts. Direct searches use static flags and weights. The native workflow optimizes each constituent window independently within the current dataset, rejects experts that fail calibration/transfer/artifact/support gates, and creates `ensemble.lock.json` automatically. It keeps native and MS2Rescore-fitted artifacts separate. A holdout dataset applies the same locked expert-selection procedure to its own independently optimized experts; it never imports another dataset's expert windows or fitted models in normal operation.

---

## Practical guidance from current validation plots

The current ISB18 validation figures suggest the following practical interpretation:

1. **TDA/TDC remains a strong peptide-level benchmark.**  The decoy-free branch should not be described as universally dominating TDA at every level.
2. **The strongest DF advantage is protein-level inference in ultra-low-input conditions.**  The data show the same unfiltered proteins observed by TDA/TDC and DF, but TDA/TDC reports zero proteins at 1% protein q while DF hierarchical inference reports 16–18 proteins with zero entrapment proteins.
3. **NOKOI is currently the strongest single candidate model by discovery power and protein recovery, but it needs top-ranked entrapment auditing.**
4. **Moments and MSFDR are high-power but can be liberal by peptide-level entrapment calibration.**
5. **MSFDR1_SMIX is underpowered under the current configuration.**
6. **The equal-weight ensemble is not yet optimal.**  It should be improved with expert QC, adaptive weighting, or outlier rejection.

---

## Known limitations and recommended next improvements

### 1. Dynamic null-window selection

The native workflow now scans declared candidate windows without rereading spectra and selects the highest-yield window satisfying PSM, peptide, and protein entrapment-FDP constraints. Further diagnostics can still be added, such as:

```text
candidate count per rank
score stability
mass-error distribution broadness
RT-error broadness
depletion of target-like delta-mass peak
pi0 stability
EM convergence stability
```

### 2. Ensemble expert QC and outlier rejection

The native workflow now vetoes experts with missing/fallback artifacts, failed entrapment calibration, underpowered accepted-entrapment counts, unstable target-only transfer, or no incremental Level-4 peptide yield. Additional diagnostics could include:

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

### 3. Entrapment-guided monotone recalibration

For models that are high-power but liberal, such as some Moments/MSFDR configurations, a monotone recalibration layer could map reported q-values onto entrapment-estimated FDR:

```text
q_reported -> q_entrapment_calibrated
```

Isotonic regression is a natural first implementation because it preserves rank order while correcting calibration.

### 4. Joint physical local-FDR modeling

RT and delta-mass evidence are currently used through optional bounded stages.  A future model could estimate local FDR over joint physical evidence:

```text
x = (score, delta_mass, delta_rt, ims, recurrence)
lfdr(x) = pi0 * f0(x) / (pi0 * f0(x) + pi1 * f1(x))
```

This should be bounded and gated in low-input data to prevent overfitting or uncontrolled rescue.

### 5. Clearer public/private field separation

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

The safest interpretation of the current data is:

> Conventional TDA/TDC remains an effective peptide-level benchmark, but in ultra-low-input proteomics it can become count-starved at the protein level.  The decoy-free framework addresses this by replacing sparse decoy counting with continuous evidence modeling and protein-supported hierarchical inference.  In ISB18, TDA/TDC and DF observe the same unfiltered protein set, but TDA/TDC reports zero significant proteins at protein q ≤ 1%, whereas DF reports 16–18 proteins with zero entrapment proteins.  The current challenge is no longer whether DF can recover proteins, but how to make the DF models uniformly calibrated, auditable, and robust across datasets.

---

## Unsupported or removed keys

The following keys appeared in earlier configs or drafts but are not accepted by the current uploaded `FdrOptions` structure:

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

The current uploaded code is organized roughly as follows:

| File | Role |
|---|---|
| `input.rs` | JSON configuration surface, enums, defaults, clamping, and resolved `FdrSettings`. |
| `decoy_free_fdr.rs` | Main DF active-stream pipeline, base model selection, q-value computation, peptide/protein inference, physical rescue, reproducibility rescue, hierarchical reporting. |
| `lower_order.rs` | LowerOrder Gumbel/TEV/TNM model fitting utilities. |
| `msfdr.rs` | MSFDR seeded, MSFDR1_SMIX, and MSFDR2_SMIX mixture models. |
| `nokoi.rs` | NOKOI-style model fitting and scoring. |
| `retention_model.rs` | RT model fitting and RT error support. |
| `mobility_model.rs` | IMS/mobility model fitting and mobility error support. |
| `stats.rs` | Statistical helpers, q-value utilities, p-value combiners, soft caps. |
| `scoring.rs` | Core feature and DF output-field structures. |
| `linear_discriminant.rs` | Linear discriminant helper logic. |

---

## Future improvements

The current code supports a large configuration surface, but several improvements would make the workflow more robust and easier to validate:

1. **Broader null-window diagnostics.** Extend the implemented in-memory constrained scanner with mass/RT null-likeness, pi0-stability, and EM-convergence diagnostics.
2. **Adaptive Ensemble weighting.** The workflow now excludes failed experts; future work can predeclare a weighting procedure and evaluate it with dataset-local holdout optimization.
3. **Bounded joint physical evidence.** RT, IMS, and delta-mass evidence should eventually be represented as local-FDR or empirical-Bayes evidence terms, but with caps, minimum support, reliability gates, and clear audit fields.
4. **Entrapment-guided calibration.** Where validation entrapments are available, use monotone recalibration, such as isotonic regression, to map reported q-values onto observed entrapment-estimated FDR.
5. **Layer-separated output fields.** Future output should preserve base, physical, reproducibility, and final hierarchical evidence separately so that every rescue/demotion is auditable.

The first two items now have conservative workflow implementations as described above; the listed extensions remain future work.

---

## 5. References

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

---

**Happy Hunting!**

---

<img src="figures/logo.png" width="300">

# Sage: proteomics searching so fast it seems like magic

[![Rust](https://github.com/lazear/sage/actions/workflows/rust.yml/badge.svg)](https://github.com/lazear/sage/actions/workflows/rust.yml) [![Anaconda-Server Badge](https://anaconda.org/bioconda/sage-proteomics/badges/version.svg)](https://anaconda.org/bioconda/sage-proteomics)


For more information please read [the online documentation!](https://sage-docs.vercel.app/docs)


# Introduction
 
Sage is, at it's core, a proteomics database search engine - 
    a tool that transforms raw mass spectra from proteomics experiments into peptide identifications 
    via database searching & spectral matching. 

However, Sage includes a variety of advanced features that make it a one-stop shop: retention time prediction, quantification (both isobaric & LFQ), peptide-spectrum match rescoring, and FDR control. You can directly use results from Sage without needing to use other tools for these tasks.

Additionally, Sage was designed with cloud computing in mind - massively parallel processing and the ability to directly stream compressed mass spectrometry data to/from AWS S3 enables unprecedented search speeds with minimal cost. 

 Sage also runs just as well reading local files from your Mac/PC/Linux device!

## Why use Sage instead of other tools?

Sage is **simple to configure**, **powerful** and **flexible**. 
It also happens to be well-tested, **mind-boggingly fast**, open-source (MIT-licensed) and free.

## Citation

If you use Sage in a scientific publication, please cite the following paper:

[Sage: An Open-Source Tool for Fast Proteomics Searching and Quantification at Scale](https://doi.org/10.1021/acs.jproteome.3c00486)


## Features

- Incredible performance out of the box
- [Effortlessly cross-platform](https://sage-docs.vercel.app/docs/started#download-the-latest-binary-release) (Linux/MacOS/Windows), effortlessly parallel (uses all of your CPU cores)
- [Fragment indexing strategy](https://sage-docs.vercel.app/docs/how_it_works) allows for blazing fast narrow and open searches (> 500 Da precursor tolerance)
- [Isobaric quantification](https://sage-docs.vercel.app/docs/how_it_works#tmt-based) (MS2/MS3-TMT, or custom reporter ions)
- [Label-free quantification](https://sage-docs.vercel.app/docs/how_it_works#label-free): consider all charge states & isotopologues *a la* FlashLFQ
- Capable of searching for [chimeric/co-fragmenting spectra](https://sage-docs.vercel.app/docs/configuration/additional)
- Wide-window (dynamic precursor tolerance) search mode - [enables WWA/PRM/DIA searches](https://sage-docs.vercel.app/docs/configuration/tolerance#wide-window-mode)
- Retention time prediction models fit to each LC/MS run
- [PSM rescoring](https://sage-docs.vercel.app/docs/how_it_works#machine-learning-for-psm-rescoring) using built-in linear discriminant analysis (LDA)
- PEP calculation using a non-parametric model (KDE)
- FDR calculation using target-decoy competition and picked-peptide & picked-protein approaches
- Percolator/Mokapot [compatible output](https://sage-docs.vercel.app/docs/configuration#env)
- Configuration by [JSON file](https://sage-docs.vercel.app/docs/configuration#file)
- Built-in support for reading gzipped-mzML files
- Support for reading/writing directly from [AWS S3](https://sage-docs.vercel.app/docs/configuration/aws), Google Cloud, or Azure.

## Interoperability

Sage is well-integrated into the open-source proteomics ecosystem. The following projects support analyzing results from Sage (typically in addition to other tools), or redistribute Sage binaries for use in their pipelines. 

- [SearchGUI](http://compomics.github.io/projects/searchgui): a graphical user interface for running searches
- [PeptideShaker](http://compomics.github.io/projects/peptide-shaker): visualize peptide-spectrum matches
- [MS2Rescore](http://compomics.github.io/projects/ms2rescore): AI-assisted rescoring of results
- [Picked group FDR](https://github.com/kusterlab/picked_group_fdr): scalable protein group FDR for large-scale experiments
- [sagepy](https://github.com/theGreatHerrLebert/sagepy): Python bindings to the sage-core library
- [quantms](https://github.com/bigbio/quantms): nextflow pipeline for running searches with Sage
- [OpenMS](https://github.com/OpenMS/OpenMS): Sage is included as a "TOPP" tool in OpenMS
- [sager](https://github.com/UCLouvain-CBIO/sager): R package for analyzing results from Sage searches
- [Sage results to mzIdentML](https://github.com/magnuspalmblad/shic/blob/main/shims/Peptide_identification_in_TSV_to_Peptide_identification_in_mzIdentML.sh): Bash script to convert `results.sage.tsv` files to mzIdentML
- [i2MassChroQ](http://pappso.inrae.fr/bioinfo/i2masschroq/): a graphical user interface for proteomics analysis
- [annotator](https://github.com/snijderlab/annotator): a graphical user interface for visualizing peptide-spectrum matches
- [rustyms](https://github.com/snijderlab/rustyms): a Rust library (with Python bindings) to handle peptides and identified peptide files
- If your project supports Sage and it's not listed, please open a pull request! If you need help integrating or interfacing with Sage in some way, please reach out.

Check out the (now outdated) [blog post introducing the first version of Sage](https://lazear.github.io/sage/) for more information and full benchmarks!
