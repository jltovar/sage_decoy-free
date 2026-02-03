//! Statistics and probability theory helpers

use log::warn;

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
/// Includes strict safety checks for small/pathological datasets.
pub fn storey_q_value(p_values: &[f64], min_n: usize) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    // Safety Net: Fallback to BH for small datasets (e.g. Single Cell)
    // Storey's pi0 estimator is unstable/noisy for N < min_n
    if m < min_n {
        warn!(
            "Storey: N < {} ({}), falling back to Benjamini-Hochberg for stability.",
            min_n, m
        );
        return bh_q_value(p_values);
    }

    // 1. Estimate Pi0 (Proportion of True Nulls)
    // We use lambda = 0.5, a standard robust choice
    let lambda = 0.5;
    let count_gt_lambda = p_values.iter().filter(|&&p| p > lambda).count() as f64;

    // Additional Safety: If count_gt_lambda is extreme (0 or all), fallback to BH.
    // 0 means "everything is a target" (impossible), m means "everything is noise" (BH is fine).
    if count_gt_lambda == 0.0 || count_gt_lambda == m as f64 {
        warn!(
            "Storey: Extreme pi0 estimate (count_gt_lambda = {}), falling back to Benjamini-Hochberg.",
            count_gt_lambda
        );
        return bh_q_value(p_values);
    }

    // Standard Storey estimator: pi0 = #{p > lambda} / (m * (1 - lambda))
    // We clamp pi0 to [0.5, 1.0].
    // - Lower bound 0.5: Prevents hyper-optimism (assuming >50% targets is risky in low input).
    // - Upper bound 1.0: Probability cannot exceed 1.
    let pi0 = (count_gt_lambda / (m as f64 * (1.0 - lambda))).clamp(0.5, 1.0);

    // 2. Calculate Q-values using the estimated pi0
    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

    let mut q_values = vec![0.0; m];
    let mut min_q = 1.0;

    for (i, &idx) in indices.iter().enumerate().rev() {
        let p = p_values[idx].max(f64::MIN_POSITIVE); // Safety clamp
        let rank = (i + 1) as f64;
        let m_f64 = m as f64;

        // Storey Formula: q = (pi0 * p * m) / rank
        let q = (pi0 * p * m_f64 / rank).min(1.0);

        min_q = q.min(min_q);
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

    // Chi-squared CDF (upper tail)
    // Using statrs for gamma functions if available, or approximation
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    match ChiSquared::new(dof) {
        Ok(dist) => 1.0 - dist.cdf(chi_sq_stat),
        Err(_) => 1.0,
    }
}
