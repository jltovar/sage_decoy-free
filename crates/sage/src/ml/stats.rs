//! Statistics and probability theory helpers

/// Calculate the harmonic mean p-value of a set of p-values
pub fn combine_hmp(p_values: &[f64]) -> f64 {
    let w = 1.0 / p_values.len() as f64;
    let sum_recip: f64 = p_values.iter().map(|&p| 1.0 / p).sum();
    (w / sum_recip).min(1.0)
}

/// Fisher's Method for combining independent p-values
/// X^2 ~ -2 * sum(ln(p))
pub fn combine_fisher(p_values: &[f64]) -> f64 {
    if p_values.is_empty() {
        return 1.0;
    }

    // Clamp small p-values to avoid infinity
    let sum_log_p: f64 = p_values.iter().map(|&p| p.max(1e-300).ln()).sum();

    let chi_sq = -2.0 * sum_log_p;
    let df = 2.0 * p_values.len() as f64;

    // Chi-squared survival function (1 - CDF)
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    match ChiSquared::new(df) {
        Ok(dist) => dist.sf(chi_sq),
        Err(_) => 1.0,
    }
}

/// Calculate the arithmetic mean of a slice. Returns 0.0 if empty.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

/// Calculate the sample standard deviation (using Bessel's correction).
pub fn std_dev(data: &[f64]) -> f64 {
    let len = data.len();
    if len < 2 {
        return 0.0; // Variance undefined/zero for < 2 samples
    }
    let mean = mean(data);
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (len - 1) as f64;
    variance.sqrt()
}

/// Benjamini-Hochberg FDR correction
/// Returns a vector of q-values corresponding to the input p-values
pub fn bh_q_value(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..m).collect();
    // Sort indices by p-value (ascending)
    indices.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

    let mut q_values = vec![0.0; m];
    let mut min_q = 1.0;

    // Iterate in reverse (largest p-value first)
    for (i, &idx) in indices.iter().enumerate().rev() {
        let p = p_values[idx];
        let rank = (i + 1) as f64;
        let m_f64 = m as f64;

        // BH Formula: q = p * m / rank
        let q = (p * m_f64 / rank).min(1.0);

        // Enforce monotonicity: q_i <= q_{i+1}
        min_q = q.min(min_q);
        q_values[idx] = min_q;
    }

    q_values
}

/// Storey's Q-value method (Least Conservative).
/// Estimates pi0 (proportion of true nulls) to boost power.
/// Includes SAFETY NETS for small N.
pub fn storey_q_value(p_values: &[f64], min_n: usize) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    // Safety Net: Fallback to BH for small datasets (e.g. Single Cell)
    // Storey's pi0 estimator is unstable/noisy for N < min_n
    if m < min_n {
        log::warn!(
            "Storey: N < {} ({}), falling back to Benjamini-Hochberg for stability.",
            min_n,
            m
        );
        return bh_q_value(p_values);
    }

    // 1. Estimate Pi0 (Proportion of True Nulls)
    let lambda = 0.5;
    let count_gt_lambda = p_values.iter().filter(|&&p| p > lambda).count() as f64;

    // Clamp pi0 to (0, 1] to prevent math errors
    let pi0 = (count_gt_lambda / (m as f64 * (1.0 - lambda))).clamp(0.0001, 1.0);

    // 2. Calculate Q-values using the estimated pi0
    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

    let mut q_values = vec![0.0; m];
    let mut min_q = 1.0;

    for (i, &idx) in indices.iter().enumerate().rev() {
        let p = p_values[idx].clamp(1e-15, 1.0); // Safety clamp
        let rank = (i + 1) as f64;
        let m_f64 = m as f64;

        // Storey Formula: q = (pi0 * p * m) / rank
        let q = (pi0 * p * m_f64 / rank).min(1.0);

        min_q = q.min(min_q);
        q_values[idx] = min_q;
    }

    q_values
}
