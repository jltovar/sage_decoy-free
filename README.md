# Sage: Decoy-Free Edition (EXPERIMENTAL)

> ⚠️ **Experimental fork.**
> This is a research fork of the original [Sage search engine]().
> APIs, configuration options, and statistical behavior may change as the decoy-free workflow is refined (**Currently not working LFQ, TMT, PIN output, annotate-matches output, parquet output**).

This fork implements an explicit **decoy-free false discovery rate (FDR)** workflow. Instead of relying on the target–decoy competition (TDC) assumption that "decoys behave like targets," this mode models **noise** using the lower-ranked PSMs from the target database itself (Rank ).

The primary motivation is **increased sensitivity and statistical power** for ultra-low-input proteomics (e.g., single-cell), where constructing a balanced decoy database is statistically difficult and every confident peptide matters.

---

## 1. Core Concept: The "Jury" System

The Decoy-Free mode treats the search engine as a **multi-model jury**. Instead of relying on a single statistical assumption, it employs a **consensus ensemble** of up to seven distinct experts.

### 1.1 The Null Model (Noise)

The engine uses lower-ranked matches (Rank , default ) to estimate the null distribution (the score distribution of random matches).

### 1.2 The Experts

The ensemble combines evidence from three classes of models:

1. **Parametric Base Models (Gumbel):**
* **Moments:** Fits Gumbel parameters using the method of moments. Fast and conservative.
* **MLE:** Fits Gumbel parameters using Maximum Likelihood Estimation. Robust to outliers in the noise tail.
* **Lower-Order (LO):** A regression-based model (Madej & Lam, 2023) that exploits the theoretical decay of order statistics to predict Rank-1 behavior from lower ranks. This implementation uses a **relative multiplicity shift** and **slope stabilization** (blending with Moments) to handle open searches and sparse data.


2. **Mixture Models (MSFDR Family):**
Derived from the work of Peng et al. (2020), this fork implements **three distinct variants** to handle different data regimes:
* **Seeded (Legacy):** Uses a fixed null derived from the pool or LO model. Only the target component (Skew-Normal) is updated via EM.
* **1SMix (Unanchored):** Initializes from the bottom/top slices of Rank-1 data and allows both the Null (Gumbel) and Target (Skew-Normal) to drift during EM training.
* **2SMix (Anchored):** Uses the pure null pool (Ranks ) to strictly anchor the Null component, preventing it from absorbing signal in high-quality datasets.


3. **Machine Learning (Nokoi 2.0):**
* **Algorithm:** L1-regularized Logistic Regression (Lasso) optimized via **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm).
* **Scoring:** Generates probabilities .
* **Calibration:** Calculates empirical p-values by comparing the ML score against the null distribution of lower-ranked matches.



---

## 2. Statistical Architecture

### 2.1 Ensemble Combination

The outputs from all active experts are combined to form a single consensus result:

* **P-Value Combination:** The default strategy is the **Cauchy Combination Test**, which is robust to strong dependencies between the expert models. Other options include Fisher, Brown (empirical covariance), and Sidak.
* **PEP Combination:** Posterior Error Probabilities (PEPs) are combined using the **Geometric Mean**, which penalizes experts that are uncertain (near 0.5) while rewarding strong consensus.

### 2.2 Calibration & Q-Values

* **Isotonic Calibration (PAVA):** Raw p-values from the ensemble and individual methods are calibrated using Isotonic Regression to enforce monotonicity (better scores  lower p-values).
* **Q-Value Calculation:**
* **Standard:** Uses the **Storey-Tibshirani** method (or Benjamini-Hochberg) to convert p-values to q-values.
* **Mixture Models:** For the MSFDR variants (1SMix/2SMix), q-values are computed directly as the cumulative mean of the PEPs, sorted by model confidence.



### 2.3 Fail-Closed Design

The engine is designed to **fail closed**. If a model cannot be fit (e.g., due to sparse data or mathematical instability), it outputs `None` (missing) or a conservative default (p=1.0). It never "guesses" or extrapolates wildly.

---

## 3. Configuration

Decoy-free mode is configured in your JSON file under the `fdr` key.

### 3.1 Recommended Defaults (JSON)

These defaults enable the full ensemble (Seeded, 1SMix, 2SMix, Nokoi, and Base Models) with Cauchy combination.

```json
"fdr": {
  "mode": "decoy_free",
    
  // Global Strategy
  "model_fit": "ensemble",
  "type": "storey",
  "protein_p_combine": "cauchy",
  "ensemble_p_combiner": "cauchy",
  "ensemble_pep_combiner": "geometric_mean",
  "calibrate_per_method": true,
  
  // Thresholds
  "peptide_fdr": 0.01,
  "protein_fdr": 0.01,
  "precursor_fdr": 0.05,

  // Global Null Window (Ranks used for noise modeling)
  "min_null_rank": 2,
  "max_null_rank": 50,
  "min_null_size": 300,
  "purification_factor": 0.5,
  "min_rank_count": 10,
  
  // Mixture EM Settings (Global)
  "mix_em_max_iter": 200,
  "mix_em_tol": 1e-6,
  "mix_anchor_incorrect": true,

  // Storey (Pi0) Tuning
  "min_storey_n": 300,
  "storey_pi0_clamp_min": 0.5,
  "storey_pi0_clamp_max": 1.0,
  "storey_lambda_min": 0.05,
  "storey_lambda_max": 0.95,
  "storey_lambda_step": 0.05,
  "storey_lambda_min_for_agg": 0.5,
  "storey_pi0_agg": "median",
  "storey_degen_fallback": "bh",

  // Moments
  "moments_min_null_rank": 8,
  "moments_max_null_rank": 8,
  
  // MLE
  "mle_min_null_rank": 11,
  "mle_max_null_rank": 11,

  // Lower-Order
  "lower_order_min_null_rank": 4,
  "lower_order_max_null_rank": 10,
  "lo_rank_key": "lo_adjusted",
  "lo_mode": "auto",
  "lo_lom_estimator": "auto",
  "lo_mean_beta_mode": "consecutive",
  "lo_mean_beta_min_rank": 8,
  "lo_mean_beta_count": 3,
  "lo_lr_window_size": null,
  "lo_stratify": "global",
  "lo_score": "per_spectrum",

  // MSFDR
  "enable_msfdr_seeded": true,
  "msfdr_min_null_rank": 4,
  "msfdr_max_null_rank": 50,
  "msfdr_use_canonical_pep": true,

  // MSFDR: 1SMix (Unanchored Mixture)
  "enable_msfdr_1smix": true,
  "msfdr1_smix_min_null_rank": 5,
  "msfdr1_smix_max_null_rank": 50,
  "msfdr1_top_frac_init": 0.2,
  "msfdr1_bottom_frac_init": 0.5,
  "msfdr1_beta_drift_mult": [0.9, 1.1],
  "msfdr1_pi_clamp_min": 0.01,
  "msfdr1_pi_clamp_max": 0.65,

  // MSFDR: 2SMix (Anchored Mixture)
  "enable_msfdr_2smix": true,
  "msfdr2_smix_min_null_rank": 4,
  "msfdr2_smix_max_null_rank": 50,
  "msfdr2_beta_drift_mult": [0.5, 2.0],
  "msfdr2_pi_clamp_min": 0.01,
  "msfdr2_pi_clamp_max": 0.568,

  // Nokoi (Machine Learning)
  "nokoi_min_null_rank": 2,
  "nokoi_max_null_rank": 7,
  "nokoi_k_folds": 2,
  "nokoi_pos_p_thresh": 0.000001,
  "nokoi_pos_rule": "and"
}

```

### 3.2 Configuration Options Explained

#### Global parameters

* **`model_fit`**:
* `"ensemble"` (Recommended): Runs all enabled models and combines them.
* `"moments"`, `"mle"`, `"lower_order"`, `"msfdr"`, `"nokoi"`: Runs only that specific model.
* `"msfdr1_smix"`, `"msfdr2_smix"`: Runs specific mixture variants.


* **`type`**: `storey` (adaptive) or `bh` (conservative Benjamini-Hochberg).
* **`ensemble_p_combiner`**: Strategy for combining p-values.
* `"cauchy"`: Robust to correlation (Default).
* `"fisher"`: Assumes independence.
* `"brown"`: Adjusts for covariance.
* `"sidak_minp"`, `"median_beta"`, `"stouffer"`.


* **`protein_p_combine`**: Aggregation for protein inference.
* `"cauchy"` (Default), `"fisher"`, `"sidak_minp"`.


#### Lower-Order parameters

* **`lower_order_min_null_rank, lower_order_max_null_rank`**: Rank window (k = hit_rank) used to fit the lower-order TEV(k) series and derive the Rank-1 null parameters.
* **`lo_rank_key`**: Which score stream LO models (`lo_adjusted` vs `hyperscore`).
* **`lo_mode`**: Bridge from lower ranks (k≥2) to Rank-1 null (`auto`, `linear_regression`, `mean_beta`).
* **`lo_lom_estimator`**: Lower-order per-rank estimator (`auto`, `mm`, `mle`) used to build the TEV(k) series.
* **`lo_mean_beta_mode`**: How β is pooled for the Mean-β bridge (`consecutive`, `from_min_rank`).
* **`lo_mean_beta_min_rank`**: First rank included in β pooling.
* **`lo_mean_beta_count`**: Number of ranks included when `lo_mean_beta_mode="consecutive"`.
* **`lo_lr_window_size`**: Optional window size for LR mode (set `null` to disable).
* **`lo_stratify`**: LO stratification (`charge` fits per-charge buckets; `global` fits one shared bucket).
* **`lo_score`**: LO score normalization (`raw` uses the selected score directly; `per_spectrum` centers each spectrum using the median score within the LO rank window).

#### MSFDR parameters

* **`enable_msfdr_seeded`**: Uses a fixed null derived from LO or Moments. Best for stability.
* **`enable_msfdr_1smix`**: Allows the null to drift. Good for datasets where the null pool might slightly mismatch the Rank-1 noise.
* **`enable_msfdr_2smix`**: Uses the pure rank-null pool to anchor the model. Often the most accurate for high-quality data.

#### Nokoi (ML)

* **`nokoi_pos_rule`**: How to select positive training examples.
* `"and"`: Must be top-rank AND have low provisional p-value (Default).
* `"or"`: Top-rank OR low p-value.
* `"top_only"`, `"p_only"`.



---

## 4. Output Columns

In `decoy_free` mode, the `results.sage.tsv` file includes specific columns for the ensemble results and detailed diagnostics for every expert method.

### 4.1 Consensus Outputs

| Column | Description |
| --- | --- |
| `decoy_free_p_value` | The final combined p-value (e.g., via Cauchy combination). |
| `decoy_free_pep` | The final combined PEP (e.g., via Geometric Mean). |
| `decoy_free_q_value` | The PSM-level False Discovery Rate. |
| `decoy_free_score` | A transformed score derived from the PEP (higher is better). |
| `decoy_free_peptide_q` | Peptide-level FDR (min q-value for the sequence). |
| `decoy_free_protein_q` | Protein-level FDR (aggregated via `protein_p_combine`). |

### 4.2 Expert Diagnostics

The output also contains the raw P, Q, and PEP values for every individual expert. These are useful for debugging why a specific spectrum was accepted or rejected.

* **P-values:** `p_mom`, `p_mle`, `p_lo`, `p_msfdr` (Seeded), `p_1smix`, `p_2smix`, `p_nokoi`.
* **Q-values:** `q_mom`, `q_mle`, `q_lo`, `q_msfdr`, `q_1smix`, `q_2smix`, `q_nokoi`.
* **PEPs:** `pep_mom`, `pep_mle`, `pep_lo`, `pep_msfdr`, `pep_1smix`, `pep_2smix`, `pep_nokoi`.

> **Note:** If a method is disabled or fails to fit (fail-closed), its columns will contain `NaN` or empty values.

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
- Support for reading/writing directly from [AWS S3](https://sage-docs.vercel.app/docs/configuration/aws)

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
- [rustyms](https://gtihub.com/snijderlab/rustyme): a Rust library (with Python bindings) to handle peptides and identified peptide files
- If your project supports Sage and it's not listed, please open a pull request! If you need help integrating or interfacing with Sage in some way, please reach out.

Check out the (now outdated) [blog post introducing the first version of Sage](https://lazear.github.io/sage/) for more information and full benchmarks!
