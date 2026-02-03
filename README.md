# Sage: Decoy-Free Edition (EXPERIMENTAL)

> ⚠️ **Experimental fork.**
> This is a research fork of the original [Sage search engine by Michael Lazear](https://github.com/lazear/sage).
> APIs, configuration options, and statistical behavior may change as the decoy-free workflow is refined.

This fork adds an explicit **decoy-free false discovery rate (FDR)** workflow alongside standard target–decoy competition (TDC). Instead of relying on the assumption that "decoys behave like targets," the decoy-free mode models **noise** using several statistical theories and **signal** using regularized machine learning and robust mixture modeling.

The primary motivation is **increased sensitivity and statistical power** for **ultra-low-input proteomics**, including single-cell and subcellular assays, where every additional confident peptide matters.

---

## 1. Decoy-Free Search Mode: Concept

In classic TDC, FDR is estimated by searching both **target** and **decoy** sequences and counting how often decoys "win." This works well but has limitations:

- It assumes decoys mimic the null distribution perfectly.
- It can be fragile in **small databases** or **extreme low-input** settings.
- It ties discovery power directly to decoy design.

The **decoy-free mode** in this fork instead uses **lower-ranked PSMs from the target database itself** to build a null model:

- Rank 1 = candidate "signal" (best PSM per spectrum).
- Ranks 2, 3, …, K = candidate "noise" used to estimate the null score distribution.

From this null, the engine computes **p-values, q-values, and posterior error probabilities (PEPs)** without ever consulting a decoy database.

---

## 2. High-Level Implementation

### 2.1 Augmented Ensemble Scoring

To reduce dependence on any single statistical assumption, the decoy-free workflow employs a **consensus ensemble** of four distinct estimators. The final significance (P-value) is derived by combining outputs from the first three models (Moments, MLE, Lower-Order) using the **Harmonic Mean P-value (HMP)**. The fourth model (MSFDR) contributes its probability estimate to the ensemble. The final Posterior Error Probability (PEP) is calculated utilizing an "Consensus Strategy", where the consensus Ensemble P-value serves directly as the PEP (PEP ≈ P-value). This ensures that if the machine learning expert (Nokoi) identifies a strong match, the error probability reflects that confidence, preventing pessimistic mixture models from overruling high-quality signal.

1.  **Method of Moments (Gumbel)**
    - Fits a Gumbel distribution to noise scores (ranks 2+) using empirical mean and variance.
    - Fast, stable, and serves as a conservative "anchor" for the ensemble.

2.  **Maximum Likelihood Estimation (MLE)**
    - Fits Gumbel parameters via likelihood maximization.
    - More robust to outliers in the tail of the noise distribution.

3.  **Lower-Order Statistics (Madej & Lam, 2023)**
    - Regresses score against $-\psi(\text{rank})$ (negative Digamma) to exploit the exact theoretical decay of Gumbel order statistics.
    - Fits an anchored intercept and slope from ranks $k = 2 \dots K$ to model the null governing rank-1.
    -  Applies a relative multiplicity shift to account for spectrum-specific search space size without reconstructing an absolute location parameter.
    -  Includes multiplicity attenuation, slope shrinkage, and safety caps to ensure numerical stability and prevent over-penalization in correlated open-search regimes.

4.  **Robust MSFDR Mixture Model (Peng et al., 2020)**
    - A stability-hardened implementation of the Mix-Max-Score framework.
    - Models the score distribution as a two-component mixture: **Gumbel (Null)** + **Skew-Normal (Target)**.
    - Features **smart data-driven initialization**, **scale-invariant EM convergence** checks using relative (1e-4) and absolute (1e-6) tolerances on the average log-likelihood per point, and **safety clamps** to prevent model collapse on sparse data.

> **Scientific Idea:** The engine acts as a **multi-model jury**, preventing false discoveries if one model fits poorly while boosting sensitivity when models agree.

#### 2.1.1 Lower-Order Multiplicity Correction & Stabilization

The Lower-Order model fits a regression line to the scores of lower-ranked matches (ranks 2, 3, etc.) to predict the expected distribution of the top rank.
- **Anchored Relative Shift** Instead of recalculating parameters from scratch for every spectrum, this fork calculates a "global reference" search space size (the geometric median of candidate counts). For each individual spectrum, it then applies a relative shift to the score baseline. This shift is proportional to the difference between that spectrum's candidate count and the global reference.
- **Multiplicity Attenuation** In open searches, candidate peptides are often highly correlated (e.g., the same peptide with different PTMs). Treating them as independent trials would over-penalize the score. This fork allows you to "dampen" this penalty using an attenuation factor ().
- **Slope Stabilization** To prevent the model from becoming unstable on sparse data, the regression slope is partially blended with the more conservative "Moments" slope. Additionally, a "safety belt" cap is applied to ensure the slope never exceeds a safe multiple of the reference Moments slope.
- **Ensemble Hijack Protection** In the ensemble mode, if the Lower-Order model produces a P-value that is drastically more optimistic (e.g., >1000x smaller) than the standard statistical models for a saturated spectrum, the system automatically falls back to the more conservative estimate to prevent false discoveries.

---

> **Scientific Note (Decoy-Free Lower-Order Calibration)**  
> This fork applies a relative, anchored multiplicity correction rather than reconstructing an absolute location parameter from rank statistics.  
> This avoids coordinate mismatches between digamma regression and log-multiplicity correction that can otherwise cause severe miscalibration.  
> The implementation is validated using mirror tests, entrapment calibration, and ensemble stability diagnostics.

### 2.2 Nokoi 2.0 Rescoring (Lasso/FISTA)

The engine includes a native implementation of **Nokoi 2.0**, an on-the-fly machine learning rescoring engine tailored for decoy-free analysis:

- **Algorithm:** L1-regularized Logistic Regression (Lasso) optimized via **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm).
- **Feature Selection:** Automatically selects relevant features (e.g., retention time alignment, ion mobility delta, intensity coverage) and zeros out noise features using L1 sparsity.
- **Adaptive Training:** Uses **K-fold cross-validation** with **early stopping** to prevent overfitting on small datasets.
- **Integration:** Nokoi probabilities are fed into the ensemble as an additional high-quality evidence stream.
- **Robust Training:** Implements balanced sampling (1:1 positive/negative ratio) to prevent class imbalance from biasing the model, and utilizes the specific wide regularization grid proposed in the original Nokoi paper.

### 2.3 Isotonic Calibration (PAVA)

Raw probabilities from the ensemble can sometimes be "jittery" due to local noise. This fork implements Isotonic Regression (PAVA) to enforce monotonicity on final P-values.

PAVA is applied for all ensemble and parametric modes, but is **skipped in pure Lower-Order mode** to preserve the intrinsic ordering of the digamma regression.

- **Function:** Enforces monotonicity on the final P-values.
- **Guarantee:** Ensures that a better matching score *always* results in an equal or better P-value/PEP.
- **Result:** Smoother, more statistically valid FDR curves that respect the natural ordering of data.

---

### 2.4 Adaptive FDR Control & Protein Inference

#### Adaptive FDR (Storey–Tibshirani)
The fork supports two procedures for converting P-values to Q-values:
- **`bh` (Benjamini–Hochberg):** Standard, conservative ($\pi_0 = 1$).
- **`storey` (Storey–Tibshirani):** Estimates the fraction of true nulls ($\hat{\pi}_0$) from the P-value distribution. This increases power in high-quality datasets by "reclaiming" true positives that BH would discard.

#### Protein Inference (Fisher's Method)
Peptide-level evidence is aggregated into protein-level confidence using **Fisher's Combined Probability Test**. This method sums the natural logarithms of the individual peptide p-values to create a single score that reflects the aggregate evidence for a protein. This allows proteins supported by multiple moderate-confidence peptides to be identified confidently.

---

## 3. Configuration & Usage

Decoy-free mode is configured in your JSON file under the `fdr` key:

```json
"fdr": {
    "mode": "decoy_free",
    "peptide_fdr": 0.01,
    "protein_fdr": 0.01,
    "precursor_fdr": 0.05,
    "min_null_rank": 4,
    "max_null_rank": 50,
    "min_null_size": 300,
    "model_fit": "ensemble",
    "type": "storey",
    "min_storey_n": 300,
    "kde_samples": 20000,
    "lo_multiplicity_alpha": 0.50,
    "lo_ln_ratio_cap": 6.9,
    "lo_beta_blend_moments": 0.30,
    "lo_beta_safety_mult": 0.60,
    "purification_factor": 0.50,
	"min_rank_count": 10
}
```

### 3.1 Core Strategy (`mode`)

- `"tdc"` (**Default**): Standard Target–Decoy Competition.
- `"decoy_free"`: **Decoy-free rank-null mode.** No decoy database is required.

### 3.2 Thresholds (`*_fdr`)

- `peptide_fdr` (default: `0.01`): The primary gatekeeper. It serves two functions:
1. Spectrum Filter: Discards any individual Spectrum Match (PSM) with a q-value above this threshold before peptide or protein inference begins.
2. Peptide Reporting: Sets the maximum allowable q-value for a unique peptide sequence to be considered "discovered" in the final report. (Note: Setting this too low (e.g., 0.01) may aggressively prune "mediocre" spectra that could have otherwise supported protein inference.)

- `protein_fdr` (default: `0.01`): The protein-level cutoff. After mapping the surviving PSMs (those that passed the peptide_fdr filter) to proteins, this threshold determines which proteins are statistically significant enough to be reported.
- `precursor_fdr` (default: `0.01`): The MS1 noise filter (LFQ only). This validates chromatographic peaks by comparing them to "shadow" (decoy) peaks. Only MS1 features with a probability of being random noise lower than this threshold will be quantified.

### 3.3 Tuning Parameters

- `min_null_rank` (default: `2`): First rank used for null modeling.
- `max_null_rank` (default: `5`): Last rank used for null modeling.
- `min_null_size` (default: `100`): Minimum number of null scores required to attempt a fit.
- `kde_samples` (default: `20000`): Controls the maximum number of data points used for Kernel Density Estimation (KDE) during P-value calculation in non-parametric modes (e.g., Moments, MLE).  Adjustment: Increase this value (e.g., to 50,000) for marginally higher precision at the cost of speed, or decrease it (e.g., to 5,000) for faster processing on low-memory systems.
- `model_fit`:
    - `"moments"`: Uses the Gumbel Method of Moments (fast, conservative).
    - `"mle"`: Uses Gumbel Maximum Likelihood Estimation (robust to outliers).
    - `"lower_order"`: Uses the Lower-Order Statistics regression (good for heavy tails).
    - `"msfdr"`: Uses the Robust Mixture Model (Gumbel + Skew-Normal).
    - `"nokoi"`: Uses the Linear Discriminant Analysis (LDA) p-value and q-value derived from ML-based rescoring.
    - `"ensemble"`: (Recommended) Runs Moments, MLE, Lower-Order, and Robust MSFDR.
- `type`:
    - `"bh"`: Benjamini-Hochberg.
    - `"storey"`: Storey-Tibshirani (requires `min_storey_n` samples).

#### 3.3.1 Lower-Order Stabilization Parameters (Decoy-Free only)

These parameters control multiplicity correction and stabilization of the Lower-Order model. They are most relevant for open searches and other highly correlated candidate spaces.

- `lo_multiplicity_alpha` (default: `0.50`): Attenuates the multiplicity shift applied to spectra with unusually large candidate sets.  
  - `1.0` = full theoretical shift  
  - `< 1.0` = damped shift (recommended for correlated searches)

- `lo_ln_ratio_cap` (default: `6.9`): Caps the multiplicity shift magnitude to prevent a single spectrum from dominating calibration.  
  `6.9 ≈ ln(1000)`.

- `lo_beta_blend_moments` (default: `0.30`): Shrinks the Lower-Order slope toward the Moments slope for stability on sparse or heavy-tailed nulls.  
  - `0.0` = pure Lower-Order  
  - higher values = more stabilization

- `lo_beta_safety_mult` (default: `0.60`): **Safety belt on the effective LO scale** relative to the Moments scale.  

  **Why the default is < 1**: In open-search or PTM-heavy workflows, candidate matches are often highly correlated rather than independent trials. Without a safety cap, the Lower-Order model can become overly conservative, inflating P-values and killing true discoveries. The safety belt effectively clamps the regression slope so it never exceeds a specific multiple (default 0.60x) of the global Moments slope.  
  The default `0.60` was selected because it yields stable entrapment calibration behavior across ISB18 and PXD001468 in this fork.  
  Increase toward `1.0–1.5` only if your candidate sets behave closer to independent trials.
  
- `purification_factor` (default: `0.50`): Sensitivity Unlock. Excludes the top-tier Rank-1 PSMs from the null distribution fit to prevent real signal from contaminating the background model.

- `min_rank_count` (default: `10`): The minimum PSMs required at a specific rank for inclusion in the Lower-Order regression. Lowering to 4–6 helps stabilize models in sparse datasets.

---



---

## 4. Decoy-Free Output & Column Definitions

When running in `decoy_free` mode, Sage maps its internal statistical calculations to the standard Sage output columns. This ensures that the results files (`results.sage.tsv`) remain compatible with existing downstream analysis tools.

### 4.1 Output Columns (Decoy-Free Mode)

In Decoy-Free mode, the output TSV replaces standard Sage columns with explicit decoy-free metrics to avoid ambiguity.

| Decoy-Free Column | Description |
| --- | --- |
| `decoy_free_score` | A "Phred-scaled" score derived from the PEP (). Used for ranking. Higher is better. |
| `decoy_free_pep` | The Posterior Error Probability (Local FDR). In Optimistic mode, this equals the Ensemble P-value. |
| `decoy_free_p_value` | The raw consensus P-value derived from the 5-way ensemble. |
| `decoy_free_q_value` | The PSM-level False Discovery Rate (FDR). |
| `decoy_free_peptide_q` | The Peptide-level FDR. |
| `p_mom` / `p_mle` / `p_lo` | Individual P-values from the Moments, MLE, and Lower-Order statistical experts. |
| `p_msfdr` / `p_nokoi` | Individual P-values from the Robust Mixture Model and Nokoi Machine Learning experts. |

> **Note on Compatibility:** While the output CSV uses these specific headers, the engine internally maps these values to standard structures during runtime. This ensures that internal Sage modules (Retention Time Prediction, Ion Mobility, and LFQ) automatically train on your Decoy-Free results without requiring external converters.


### 4.2 False Discovery Rate (FDR) Calculations

The FDR columns in Sage Decoy-Free are dynamic. Their values change depending on the `model_fit` strategy selected in your configuration (e.g., `moments`, `mle`, `lower_order`, `msfdr`, `ensemble`).

#### `spectrum_q` (PSM-level FDR)

- **Source:** Derived directly from the `decoy_free_p_value`.
- **Dependency:** The underlying P-value changes based on the selected model:

  - `ModelFit::Moments`: Uses the Gumbel Moments p-value.  
  - `ModelFit::Mle`: Uses the Gumbel MLE p-value.
  - `ModelFit::LowerOrder`: Uses the Lower-Order Statistics p-value (optimized for small sample sizes).
  - `ModelFit::Msfdr`: Uses the **seeded null survival p-value** from the MSFDR model’s null component (Gumbel), while the mixture model is used to compute **PEP**.
  - `ModelFit::Nokoi`: Uses the Linear Discriminant Analysis (LDA) p-value derived from ML-based rescoring.
  - `ModelFit::Ensemble`: Calculates the Harmonic Mean of the parametric null models (Moments, MLE, Lower-Order). The MSFDR mixture model is **not included in the ensemble P-value** to avoid double-counting the seeded null. Instead, MSFDR is used exclusively to compute the Posterior Error Probability (PEP).

- **Calculation:** After determining the raw P-value, the Benjamini–Hochberg (or Storey) procedure is applied globally to convert it into a Q-value (`spectrum_q`).

#### `peptide_q` (Peptide-level FDR)

- **Calculation:** Computed by taking the best (minimum) decoy_free_q_value observed for that peptide sequence across all scans. This is now calculated unconditionally for every search, ensuring valid peptide-level statistics are always available for LFQ.
- **Dependency:** Improvements in spectrum-level modeling (e.g., Ensemble vs Moments) directly propagate to peptide-level confidence.

#### `protein_q` (Protein-level FDR)

- **Calculation:** Aggregates the `decoy_free_p_values` of all unique peptides assigned to a protein using **Fisher’s Method** for combining independent p-values.
- **Dependency:** Strongly influenced by model choice. Sharper p-values from better models (e.g., Ensemble) improve discrimination during protein inference.

---


## 5. Scientific Summary of This Fork

In this experimental decoy-free fork, Sage is being developed into a **multi-model consensus engine** for proteomics discovery:

- It **models noise** using multiple statistical frameworks:
  - Extreme value theory (Gumbel via Moments and MLE),
  - Rank-order theory (Lower-Order Statistics),
  - Two-component mixtures (Robust MSFDR).

- It **models signal** using:
  - **Nokoi 2.0 (Lasso)**: Sparse, regularized machine learning.
  - **Robust MSFDR**: Mixture modeling with Skew-Normal targets.

- It **ensures validity** using:
  - **Isotonic Calibration (PAVA)**: Enforcing monotonic probabilities.
  - **Harmonic Mean P-values (HMP)**: Robust evidence combination.

The overarching goal is a workflow that is **statistically principled**, **honest in outputs** (explicit NaNs for missing data), and **highly sensitive** for ultra-low-input regimes.

---

## 6. References

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

## 7. Status & Caveats

- This fork is **experimental** and intended for method development and research.
- Always inspect log messages and output columns to confirm which models were applied.
- **Fail-Safe Design:** If statistical models cannot be fit (e.g., due to sparse data), the engine defaults to a probability of 1.0 (Fail-Closed). Additionally, Nokoi ML includes a graceful fallback for datasets with fewer than 50 confident PSMs, defaulting to normalized hyperscores to prevent unstable training.
- If opening an issue, please include your `config.json` and a log excerpt showing the active `fdr.mode`.
- Log messages report LO saturation diagnostics (frac_ln_clipped, frac_beta_capped) to assist calibration tuning.

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
