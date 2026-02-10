//! Statistics and probability theory helpers

use log::warn;
use statrs::distribution::{ChiSquared, ContinuousCDF};

pub fn mean(data: &[f64]) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

pub fn std_dev(data: &[f64]) -> f64 {
    let m = mean(data);
    let variance = data
        .iter()
        .map(|value| {
            let diff = m - (*value);
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64;
    variance.sqrt()
}

/// Benjamini-Hochberg FDR Control
/// Returns q-values for the input p-values.
pub fn bh_q_value(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..m).collect();
    // Sort indices by p-value ascending
    indices.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

    let mut q_values = vec![0.0; m];
    let mut min_q = 1.0;

    // BH Step-up procedure
    for (i, &idx) in indices.iter().enumerate().rev() {
        let p = p_values[idx];
        let rank = (i + 1) as f64;
        let m_f64 = m as f64;

        // BH Formula: q = p * m / rank
        // We clamp to 1.0 because probability cannot exceed 1
        let q = (p * m_f64 / rank).min(1.0);

        // Enforce monotonicity (q_i <= q_{i+1})
        min_q = q.min(min_q);
        q_values[idx] = min_q;
    }

    q_values
}

/// Storey-Tibshirani Adaptive FDR Control
///
/// More powerful than BH when a significant fraction of hypotheses are true targets.
/// Includes safety checks and *input validation* (p-values must be in [0, 1]).
///
/// If the input looks invalid (NaN/inf, negatives, or many values > 1),
/// we clamp to [eps, 1] and warn with diagnostics, because those cases almost
/// always indicate the caller passed scores/log-values instead of p-values.
pub fn storey_q_value(p_values: &[f64], min_n: usize) -> Vec<f64> {
    use log::{info, warn};

    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    // Safety Net: Fallback to BH for small datasets
    if m < min_n {
        warn!(
            "Storey: N < {} ({}), falling back to Benjamini-Hochberg for stability.",
            min_n, m
        );
        return bh_q_value(p_values);
    }

    // ---------- Input validation + sanitation ----------
    // We will clamp p-values into [eps, 1.0] and run Storey on the sanitized vector.
    // If many values are out-of-range, that's a strong signal the caller provided
    // non-p-values (e.g., -log10(p), PEP in log space, discriminant score, etc.).
    let eps = 1e-300_f64; // tiny, avoids zeros without distorting normal p-values

    let mut n_nonfinite = 0usize;
    let mut n_neg = 0usize;
    let mut n_gt1 = 0usize;

    // Copy + sanitize
    let mut p: Vec<f64> = Vec::with_capacity(m);
    for &x in p_values {
        if !x.is_finite() {
            n_nonfinite += 1;
            p.push(1.0); // treat as uninformative / null-like
            continue;
        }
        if x < 0.0 {
            n_neg += 1;
        }
        if x > 1.0 {
            n_gt1 += 1;
        }
        p.push(x.clamp(eps, 1.0));
    }

    if n_nonfinite > 0 || n_neg > 0 || n_gt1 > 0 {
        warn!(
            "Storey: input p-values out of range; nonfinite={}, neg={}, >1={}. \
             Clamping to [eps, 1]. This usually means the caller passed non-p-values.",
            n_nonfinite, n_neg, n_gt1
        );
    }

    // Optional: a small diagnostic to catch the classic "everything > 0.5" case
    // without spamming. This is useful when debugging data flow.
    if log::log_enabled!(log::Level::Info) {
        let lambda_dbg = 0.5;
        let count_gt = p.iter().filter(|&&v| v > lambda_dbg).count();
        // Only log if it's extreme-ish
        if count_gt == 0 || count_gt == m {
            info!(
                "Storey DEBUG: after clamping, count(p > 0.5) = {} of {} (extreme).",
                count_gt, m
            );
        }
    }

    // ---------- Storey pi0 estimation ----------
    // Standard robust choice
    let lambda = 0.5_f64;

    // Use sanitized p-values here
    let count_gt_lambda = p.iter().filter(|&&pv| pv > lambda).count() as f64;

    // If count_gt_lambda is extreme, Storey reduces to BH anyway (pi0 ~ 1),
    // but we keep a fallback for numerical stability / clarity.
    if count_gt_lambda == 0.0 || count_gt_lambda == m as f64 {
        warn!(
            "Storey: extreme pi0 estimate (count_gt_lambda = {} of {}), falling back to BH.",
            count_gt_lambda, m
        );
        return bh_q_value(&p);
    }

    // Standard Storey estimator: pi0 = #{p > lambda} / (m * (1 - lambda))
    // IMPORTANT: Do NOT force a hard lower clamp like 0.5; it can make you
    // artificially conservative. Keep it in [0, 1].
    let mut pi0 = count_gt_lambda / (m as f64 * (1.0 - lambda));
    pi0 = pi0.clamp(0.0, 1.0);

    // ---------- Q-value computation ----------
    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&a, &b| p[a].total_cmp(&p[b]));

    let mut q_values: Vec<f64> = vec![0.0_f64; m];
    let mut min_q: f64 = 1.0_f64;

    for (i, &idx) in indices.iter().enumerate().rev() {
        let pv = p[idx]; // already clamped into [eps, 1]
        let rank = (i + 1) as f64;

        // Storey: q = (pi0 * p * m) / rank
        let q = (pi0 * pv * (m as f64) / rank).min(1.0);

        min_q = min_q.min(q);
        q_values[idx] = min_q;
    }

    q_values
}

/// Combine P-values using Harmonic Mean P-value (HMP)
/// Robust to dependency between tests.
pub fn combine_hmp(p_values: &[f64]) -> f64 {
    if p_values.is_empty() {
        return 1.0;
    }
    let k = p_values.len() as f64;
    // FIX: Relaxed clamp from 1e-15 to 1e-100 to preserve high-confidence scores
    let sum_inverse: f64 = p_values.iter().map(|&p| 1.0 / p.max(1e-100)).sum();

    // Landau's correction factor usually applied for dependent tests,
    // here simplified to the asymptotic HMP bound.
    // HMP = (Sum(1/p) / k)^-1
    // We adjust by * e * ln(k) for rigorous control under arbitrary dependence,
    // but for consensus scoring, the raw harmonic mean is often used as a ranking score.
    // We'll use the raw harmonic mean * k (Standard HMP) then typically homogenized.
    // Here we implement the simple Harmonic Mean: k / sum(1/p)
    // Wait, HMP for combining p-values usually roughly follows:
    // p_combined = (w_1 + ... + w_k) / (w_1/p_1 + ... + w_k/p_k)

    let hmp = k / sum_inverse;

    // Often we penalize slightly for ensemble agreement.
    // HMP is asymptotically valid.
    hmp.min(1.0)
}

/// Combine P-values using Fisher's Method
/// Assumes independence (used for Peptide -> Protein aggregation)
pub fn combine_fisher(p_values: &[f64]) -> f64 {
    if p_values.is_empty() {
        return 1.0;
    }
    let k = p_values.len() as f64;
    // X = -2 * sum(ln(p))
    // FIX: Relaxed clamp here too
    let chi_sq_stat: f64 = -2.0 * p_values.iter().map(|&p| p.max(1e-100).ln()).sum::<f64>();

    // Degrees of freedom = 2k
    let dof = 2.0 * k;

    match ChiSquared::new(dof) {
        Ok(dist) => 1.0 - dist.cdf(chi_sq_stat),
        Err(_) => 1.0,
    }
}

/// Parameters for Brown’s method approximation:
/// S = -2 * sum ln(p_i) is approximated as: S ~ scale * ChiSquare(dof)
#[derive(Clone, Copy, Debug)]
pub struct BrownParams {
    pub dof: f64,
    pub scale: f64,
}

/// Fit Brown parameters from an observation matrix of p-values:
/// - rows = observations (e.g., many PSMs)
/// - cols = component tests (e.g., per-method p-value streams)
///
/// This is the "Empirical Brown’s Method" style fit:
/// - transform X_ij = -2 ln(p_ij)
/// - estimate Var(S) using covariance of columns
/// - match moments: mean(S)=2k, var(S)=Var(S)
pub fn fit_brown_params(p_matrix: &[Vec<f64>]) -> Option<BrownParams> {
    if p_matrix.is_empty() {
        return None;
    }
    let k = p_matrix[0].len();
    if k < 2 {
        return None;
    }
    // ensure rectangular
    if p_matrix.iter().any(|row| row.len() != k) {
        warn!("Brown fit: non-rectangular p-matrix; skipping Brown fit.");
        return None;
    }

    let eps = 1e-300_f64;
    let n = p_matrix.len() as f64;

    // Compute transformed means per column: X = -2 ln(p)
    let mut col_mean = vec![0.0_f64; k];
    for row in p_matrix {
        for (j, &p) in row.iter().enumerate() {
            let pj = p.clamp(eps, 1.0);
            let x = -2.0 * pj.ln();
            col_mean[j] += x;
        }
    }
    for j in 0..k {
        col_mean[j] /= n;
    }

    // Sample covariance matrix of columns (unbiased with n-1).
    // cov[j][l] = cov(X_j, X_l)
    if p_matrix.len() < 2 {
        return None;
    }
    let denom = (p_matrix.len() as f64) - 1.0;

    let mut cov = vec![vec![0.0_f64; k]; k];
    for row in p_matrix {
        // centered X for this row
        let mut centered = vec![0.0_f64; k];
        for (j, &p) in row.iter().enumerate() {
            let pj = p.clamp(eps, 1.0);
            let x = -2.0 * pj.ln();
            centered[j] = x - col_mean[j];
        }
        for j in 0..k {
            for l in 0..=j {
                cov[j][l] += centered[j] * centered[l];
            }
        }
    }
    for j in 0..k {
        for l in 0..=j {
            cov[j][l] /= denom;
            cov[l][j] = cov[j][l];
        }
    }

    // Var(S) where S = sum_j X_j  is: 1^T Cov 1  = sum_{j,l} cov[j][l]
    let mut var_s = 0.0_f64;
    for j in 0..k {
        for l in 0..k {
            var_s += cov[j][l];
        }
    }

    // Brown moment matching:
    // mean_s = 2k
    // dof = 2 * mean_s^2 / var_s
    // scale = var_s / (2 * mean_s)
    let mean_s = 2.0 * (k as f64);

    if !var_s.is_finite() || var_s <= 0.0 {
        warn!(
            "Brown fit: non-positive or non-finite var_s={}; skipping Brown.",
            var_s
        );
        return None;
    }

    let dof = 2.0 * mean_s * mean_s / var_s;
    let scale = var_s / (2.0 * mean_s);

    if !dof.is_finite() || !scale.is_finite() || dof <= 0.0 || scale <= 0.0 {
        warn!(
            "Brown fit: invalid params dof={} scale={}; skipping Brown.",
            dof, scale
        );
        return None;
    }

    Some(BrownParams { dof, scale })
}

/// Combine p-values with Brown’s method using fitted BrownParams.
/// Falls back to Fisher if parameters are missing/invalid.
pub fn combine_brown(p_values: &[f64], params: Option<BrownParams>) -> f64 {
    if p_values.is_empty() {
        return 1.0;
    }
    let eps = 1e-300_f64;

    // Fisher statistic
    let s: f64 = -2.0
        * p_values
            .iter()
            .map(|&p| p.clamp(eps, 1.0).ln())
            .sum::<f64>();

    let Some(bp) = params else {
        // No dependency model; Fisher is the appropriate fallback
        return combine_fisher(p_values);
    };

    // Brown: S ~ scale * ChiSquare(dof)
    // => p = P(ChiSquare(dof) >= S/scale)
    let x = s / bp.scale;

    match ChiSquared::new(bp.dof) {
        Ok(dist) => (1.0 - dist.cdf(x)).clamp(0.0, 1.0),
        Err(_) => 1.0,
    }
}
