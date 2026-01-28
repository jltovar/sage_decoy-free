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

To reduce dependence on any single statistical assumption, the decoy-free workflow employs a **consensus ensemble** of four distinct estimators. The final significance (P-value) is derived by combining outputs from the first three models (Moments, MLE, Lower-Order) using the **Harmonic Mean P-value (HMP)**. The fourth model (MSFDR) is used to calculate the Posterior Error Probability (PEP), providing a robust, data-driven measure of local FDR that informs the final score.

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

The Lower-Order model exploits the theory of Gumbel order statistics by fitting mean score versus rank:

$$
\mathrm{Score}_k \approx \mu_{\mathrm{intercept}} + \beta_{\mathrm{LO}} \, \bigl[-\psi(k)\bigr]
$$

where $\psi(\cdot)$ is the Digamma function and $k$ is the rank.

In practice, each spectrum is searched against a different effective number of candidates (`scored_candidates`). Treating all spectra as having identical multiplicity leads to systematic miscalibration, especially when candidate counts vary widely across spectra.

**Anchored Relative Shift (coordinate-consistent mapping)**  
This fork applies multiplicity correction as a *relative* shift anchored to a global reference multiplicity, rather than reconstructing an absolute location parameter from incompatible formulas.

---

##### Step 1 — Define a reference multiplicity ($n_{\mathrm{global}}$)

Let $n_i$ denote the effective candidate count for spectrum $i$:

$$
n_i = \mathrm{scored\_candidates}_i.
$$

We define the global reference multiplicity using the **median on the log scale** (i.e., $\exp(\mathrm{median}(\log n))$) over rank-1 spectra:

$$
n_{\mathrm{global}}
=
\exp\!\left(
\operatorname{median}\!\left[
\log\!\Big(
\operatorname{clamp}(n_i,\;10,\;10^{7})
\Big)
\right]
\right).
$$

This is a scale-robust anchor for multiplicative search-space effects: the median is taken in log-space and exponentiated.

---

##### Step 2 — Apply a bounded, attenuated multiplicity shift

For a spectrum with local multiplicity $n_i$, we compute a bounded log-ratio:

$$
\Delta_i
=
\operatorname{clamp}\!\left(
\ln n_i - \ln n_{\mathrm{global}},\;
-c,\;
c
\right),
$$

where $c$ corresponds to `lo_ln_ratio_cap`.

We then shift only *relative* to the reference anchor:

$$
\mu_i
=
\mu_{\mathrm{intercept}}
+
\beta_{\mathrm{LO,eff}} \, \alpha \, \Delta_i,
$$

where:

- $\alpha \in [0,1]$ is `lo_multiplicity_alpha` (attenuates multiplicity in correlated candidate spaces),
- $\beta_{\mathrm{LO,eff}}$ is the stabilized Lower-Order scale used for the dynamic Gumbel null.

Key properties:
- preserves the digamma regression coordinate system,
- avoids reconstructing an absolute $\mu$ using mixed coordinate formulas,
- penalizes only deviations from the typical search space,
- keeps behavior bounded and monotone.

---

##### Multiplicity attenuation (correlation control)

In open-search and PTM-heavy workflows, candidates can be highly correlated. To avoid over-penalization, the multiplicity shift is damped by `lo_multiplicity_alpha`:

- $\alpha = 1.0$ → full theoretical penalty,
- $\alpha = 0.5$ → attenuated penalty (default in this fork).

---

##### Slope stabilization & safety belt

Lower-Order slopes can become unstable in heavy tails or sparse nulls. Three safeguards are applied:

**(1) Shrinkage toward Moments**

$$
\beta_{\mathrm{shrunk}}
=
(1-w)\,\beta_{\mathrm{LO}}
+
w\,\beta_{\mathrm{Moments}},
$$

where $w$ is `lo_beta_blend_moments`.

**(2) Hard safety cap relative to a reference scale**

Let $\beta_{\mathrm{ref}}$ denote the reference scale (Moments beta). Then:

$$
\beta_{\mathrm{LO,eff}}
=
\operatorname{clamp}\!\left(
\beta_{\mathrm{shrunk}},\;
\epsilon,\;
\text{safety}_{\mathrm{mult}} \cdot \beta_{\mathrm{ref}}
\right),
$$

where `safety_mult` is `lo_beta_safety_mult` and $\epsilon$ is a small positive constant to keep the Gumbel scale valid.

---

##### (3) Ensemble hijack protection (ensemble modes only)

To prevent pathological dominance by the Lower-Order model during *ensemble* combination, soft hijack protection is applied **only** in modes that compute an ensemble P-value (e.g., `ensemble`, `ensemble_debug`, and Nokoi-augmented ensemble paths). It is **not** applied in pure `lower_order` mode.

Define the conservative base:

$$
p_{\mathrm{base}} = \min\!\left(p_{\mathrm{Moments}},\,p_{\mathrm{MLE}}\right).
$$

For each spectrum, define a "saturated regime" indicator:

- the multiplicity shift was clipped (i.e., $|\Delta_i| = c$), **or**
- the LO scale hit the safety cap (i.e., $\beta_{\mathrm{LO,eff}} = \beta_{\mathrm{cap}}$).

Hijack protection triggers only if LO is saturated **and** LO is extremely more liberal than the base:

$$
p_{\mathrm{LO}} < \frac{p_{\mathrm{base}}}{1000}.
$$

If both conditions hold, LO is guarded:

$$
p_{\mathrm{LO,guarded}} = p_{\mathrm{base}},
$$

otherwise:

$$
p_{\mathrm{LO,guarded}} = p_{\mathrm{LO}}.
$$

The ensemble P-value is then computed as:

$$
p_{\mathrm{ens}} = \operatorname{HMP}\!\left(p_{\mathrm{Moments}},\,p_{\mathrm{MLE}},\,p_{\mathrm{LO,guarded}}\right).
$$

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
Peptide-level evidence is aggregated into protein-level confidence using **Fisher's Combined Probability Test**:
$$X = -2 \sum_{i=1}^{n} \ln(p_i)$$
This allows proteins supported by multiple moderate-confidence peptides to be identified confidently, even if no single peptide reaches strict significance on its own.

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
    - `"ensemble_debug"`: Same as above but writes **validation columns** (`p_moments`, `p_mle`, `p_msfdr`, `p_lower_order`, `p_nokoi`, `q_nokoi`) to the output.
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
  In code, the dynamic LO null uses:

$$
\beta_{\mathrm{cap}} = \text{safety}_{\text{mult}} \cdot \beta_{\text{Moments}},
\qquad
\beta_{\text{LO,eff}} = \operatorname{clamp}\!\left(\beta_{\text{shrunk}},\;10^{-9},\;\beta_{\mathrm{cap}}\right)
$$

  **Why the default is < 1:** in open-search / PTM-heavy workflows, `scored_candidates` are highly correlated (not independent trials).  
  Without a tight cap, LO can enter an overly heavy-tailed regime and become **off-the-chart conservative** (inflated survival p-values), collapsing discoveries.  
  The default `0.60` was selected because it yields stable entrapment calibration behavior across ISB18 and PXD001468 in this fork.  
  Increase toward `1.0–1.5` only if your candidate sets behave closer to independent trials.
  
- `purification_factor` (default: `0.50`): Sensitivity Unlock. Excludes the top-tier Rank-1 PSMs from the null distribution fit to prevent real signal from contaminating the background model.

- `min_rank_count` (default: `10`): The minimum PSMs required at a specific rank for inclusion in the Lower-Order regression. Lowering to 4–6 helps stabilize models in sparse datasets.

---



---

## 4. Decoy-Free Output & Column Definitions

When running in `decoy_free` mode, Sage maps its internal statistical calculations to the standard Sage output columns. This ensures that the results files (`results.sage.tsv`) remain compatible with existing downstream analysis tools.

### 4.1 Column Mapping (Decoy-Free Mode)

To maintain compatibility with TDA workflows, Decoy-Free statistics are mapped to standard Sage headers as follows:

| Standard Sage Header | Decoy-Free Source Value | Description |
|---------------------|------------------------|-------------|
| `sage_discriminant_score` | `decoy_free_score` | A "Phred-scaled" score derived from the Posterior Error Probability (PEP). Calculated as `-10 * log10(PEP)`. Higher is better. |
| `posterior_error` | `decoy_free_pep` | The probability that this specific match is incorrect (Local FDR). Calculated as `null_density / total_density`. |
| `spectrum_q` | `decoy_free_q_value` | The final PSM-level False Discovery Rate (FDR). |

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

- **Calculation:** Computed by taking the best (minimum) `spectrum_q` observed for that peptide sequence across all scans.
- **Dependency:** Improvements in spectrum-level modeling (e.g., Ensemble vs Moments) directly propagate to peptide-level confidence.

#### `protein_q` (Protein-level FDR)

- **Calculation:** Aggregates the `decoy_free_p_values` of all unique peptides assigned to a protein using **Fisher’s Method** for combining independent p-values.
- **Dependency:** Strongly influenced by model choice. Sharper p-values from better models (e.g., Ensemble) improve discrimination during protein inference.

### 4.3 Debug Columns (`ensemble_debug`)

Sage includes a special configuration mode for analyzing the internal behavior of the Decoy-Free models.

If you set:

```json
"model_fit": "ensemble_debug"
```

in your JSON configuration, the output CSV will additionally include the following raw statistical columns (hidden in standard runs to reduce file size and clutter):

- `decoy_free_p_value` — Raw probability that a hit is a random false match (direct model output).  
- `p_moments` — P-value from Gumbel Method of Moments.  
- `p_mle` — P-value from Gumbel Maximum Likelihood Estimation.  
- `p_lower_order` — P-value from Lower-Order Statistics regression.  
- `p_msfdr` — P-value from the MSFDR (Skew-Normal mixture) model.  
- `p_nokoi` / `q_nokoi` — Statistics from the Nokoi rescoring integration.

These columns are intended for **method validation, diagnostics, and calibration studies**, and should not normally be used directly for reporting discoveries.

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
- **Fail-Safe Design:** If statistical models cannot be fit (e.g., due to sparse data), the engine defaults to a probability of 1.0 (Fail-Closed) rather than crashing.
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
