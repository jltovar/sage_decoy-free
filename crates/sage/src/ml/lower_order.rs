//! Decoy-free Lower-Order (LO) model fitting utilities.
//!
//! This module implements Lower-Order statistics from the rank-specific
//! transformed extreme-value likelihood.  For a selected null-rank window
//! [min_null_rank, max_null_rank], each observed lower-order score keeps its
//! actual hit rank k and contributes through the k-specific TEV likelihood.
//!
//! The fitted Target Null Model (TNM) is a single shared rank-1 null model
//! obtained by minimizing the joint negative log-likelihood across all usable
//! lower-order ranks in the configured window.  The window is therefore an
//! actual likelihood window, not a menu from which one rank is selected.
//!
//! The methods in this module are inspired by the work of Dominik Madej and Henry Lam published here:
//!
//! Modeling Lower-Order Statistics to Enable Decoy-Free FDR Estimation in Proteomics
//! Dominik Madej and Henry Lam
//! Journal of Proteome Research 2023 22 (4), 1159-1171
//! DOI: 10.1021/acs.jproteome.2c00604
//! https://pubs.acs.org/doi/full/10.1021/acs.jproteome.2c00604
//!
//! and implemented on GitHub here:
//! https://github.com/dommad/pylord

use fnv::FnvHashMap;
use statrs::consts::EULER_MASCHERONI;
use statrs::function::gamma::ln_gamma;

/// Method-of-moments Gumbel fit (mu, beta).
pub(crate) fn fit_gumbel_moments(scores: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.len() < 2 {
        return (f64::NAN, f64::NAN);
    }

    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let var = finite.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;

    if !var.is_finite() || var < 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let beta = (var * 6.0 / std::f64::consts::PI.powi(2)).sqrt();
    if !beta.is_finite() || beta <= 0.0 {
        (f64::NAN, f64::NAN)
    } else {
        (mean - EULER_MASCHERONI * beta, beta)
    }
}

/// Gumbel MLE fit (mu, beta) using a Newton–Raphson update for beta
/// (PyLord-family estimator), then closed-form mu.
///
/// Solves for beta in:
///   f(beta) = beta - x̄ + A(beta)/B(beta) = 0
/// where:
///   B(beta) = Σ exp(-x/beta)
///   A(beta) = Σ x exp(-x/beta)
///
/// Newton step:
///   beta <- beta - f(beta) / f'(beta)
pub(crate) fn fit_gumbel_mle(scores: &[f64]) -> Option<(f64, f64)> {
    let finite: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.len() < 2 {
        return None;
    }

    let n = finite.len() as f64;
    let x_bar = finite.iter().sum::<f64>() / n;

    // Initialize with moments beta.
    let (_, mut beta) = fit_gumbel_moments(&finite);
    if !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    const MAX_ITERS: usize = 50;
    const TOL_ABS: f64 = 1e-8;

    for _ in 0..MAX_ITERS {
        // B = Σ e^{-x/beta}
        // A = Σ x e^{-x/beta}
        // C = Σ x^2 e^{-x/beta}   (for derivative)
        let mut b_sum = 0.0f64;
        let mut a_sum = 0.0f64;
        let mut c_sum = 0.0f64;

        for &x in &finite {
            let z = -x / beta;
            let e = z.exp();
            if !e.is_finite() {
                return None;
            }
            b_sum += e;
            a_sum += x * e;
            c_sum += x * x * e;
        }

        if !b_sum.is_finite() || b_sum <= 0.0 {
            return None;
        }

        let a_over_b = a_sum / b_sum;

        // f(beta) = beta - x_bar + A/B
        let f = beta - x_bar + a_over_b;

        // dA/dbeta = (1/beta^2) * Σ x^2 e^{-x/beta} = c_sum / beta^2
        // dB/dbeta = (1/beta^2) * Σ x   e^{-x/beta} = a_sum / beta^2
        // d(A/B)   = (dA*B - A*dB)/B^2
        let beta2 = beta * beta;
        if !beta2.is_finite() || beta2 <= 0.0 {
            return None;
        }

        let d_a = c_sum / beta2;
        let d_b = a_sum / beta2;
        let d_a_over_b = (d_a * b_sum - a_sum * d_b) / (b_sum * b_sum);

        // f'(beta) = 1 + d(A/B)
        let fp = 1.0 + d_a_over_b;

        if !fp.is_finite() || fp.abs() < 1e-12 {
            return None;
        }

        let next_beta = beta - (f / fp);

        if !next_beta.is_finite() || next_beta <= 0.0 {
            return None;
        }

        if (next_beta - beta).abs() < TOL_ABS {
            beta = next_beta;
            break;
        }

        beta = next_beta;
    }

    // Closed-form mu: mu = -beta * ln( (1/n) Σ exp(-x/beta) ) = beta * ln(n / Σ exp(-x/beta))
    let sum_exp = finite.iter().map(|&x| (-x / beta).exp()).sum::<f64>();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return None;
    }
    let mu = beta * (n / sum_exp).ln();

    if mu.is_finite() && beta.is_finite() && beta > 0.0 {
        Some((mu, beta))
    } else {
        None
    }
}

#[inline]
fn log_factorial_k_minus_1(k: u32) -> f64 {
    // (k-1)! = Γ(k)  => ln((k-1)!) = lnΓ(k)
    if k == 0 {
        return f64::NAN;
    }
    ln_gamma(k as f64)
}

// -----------------------------------------------------------------------------
// Per-k LO estimators (PyLord-inspired / PyLord-consistent)
// -----------------------------------------------------------------------------
//
// Matches PyLord stat.py:
//   MethodOfMoments().estimate_parameters(scores, hit_rank=k)
//
// scale  = sqrt( (E[X^2] - E[X]^2) / psi(k-1) )
// location = E[X] - scale * (EulerGamma - H_{k-1})
//
// where:
//   psi(m) = pi^2/6 - sum_{i=1..m} 1/i^2
//   H_m    = sum_{i=1..m} 1/i
//

#[inline]
fn nll_tev_k(mu: f64, beta: f64, scores: &[f64], k: u32) -> f64 {
    // Matches PyLord AsymptoticGumbelMLE.get_log_likelihood:
    //
    // likelihood = -n*log(beta*(k-1)!) - k*sum(z) - sum(exp(-z))
    // returns -likelihood
    //
    // z = (x - mu)/beta
    //
    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 || k < 1 {
        return f64::INFINITY;
    }

    let lf = log_factorial_k_minus_1(k);
    if !lf.is_finite() {
        return f64::INFINITY;
    }

    let n = scores.len();
    if n == 0 {
        return f64::INFINITY;
    }

    let ln_beta = beta.ln();
    if !ln_beta.is_finite() {
        return f64::INFINITY;
    }

    let mut sum_z = 0.0f64;
    let mut sum_exp = 0.0f64;

    for &x in scores {
        if !x.is_finite() {
            return f64::INFINITY;
        }
        let z = (x - mu) / beta;
        if !z.is_finite() {
            return f64::INFINITY;
        }
        sum_z += z;

        // exp(-z) can overflow for large negative z, clamp exponent.
        let t = (-z).clamp(-745.0, 745.0);
        let ez = t.exp();
        if !ez.is_finite() {
            return f64::INFINITY;
        }
        sum_exp += ez;
    }

    // -likelihood:
    // n * ln(beta*(k-1)!) + k*sum(z) + sum(exp(-z))
    let n_f = n as f64;
    n_f * (ln_beta + lf) + (k as f64) * sum_z + sum_exp
}

#[inline]
fn tev_cdf_asymptotic(z: f64, k: u32) -> f64 {
    // Asymptotic TEV CDF consistent with nll_tev_k():
    // cdf_asymptotic(z, k) = exp(-exp(-z)) * Σ_{m=0..k-1} exp(-m z)/m!
    //
    // For k = 1 this reduces to the standard Gumbel-max CDF:
    //   exp(-exp(-z))
    //
    // Implemented via a stable recurrence:
    // term_0 = 1
    // term_m = term_{m-1} * exp(-z) / m
    //
    if !z.is_finite() || k < 1 {
        return f64::NAN;
    }

    // t = exp(-z) with exponent clamp to avoid overflow
    let t = (-z).clamp(-745.0, 745.0).exp();
    if !t.is_finite() {
        return f64::NAN;
    }

    // sum_{m=0..k-1} t^m / m!
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    for m in 1..k {
        term *= t / (m as f64);
        sum += term;
    }

    // exp(-t) * sum
    let cdf = (-t).clamp(-745.0, 745.0).exp() * sum;
    cdf.clamp(0.0, 1.0)
}

#[derive(Debug, Clone)]
pub(crate) struct RankBucket {
    pub k: u32,
    pub scores: Vec<f64>,
}

// -----------------------------------------------------------------------------
// Charge filling rules
//
//
// Rules implemented:
// 1) If charge 1 is missing but charge 2 exists: copy charge 2 into charge 1.
// 2) If a charge ABOVE the max fitted charge is requested: copy max fitted charge.
// 3) For gaps INSIDE the fitted range:
//    - MinimalDelta: do NOT nearest-neighbor fill; rely on fallback_params.
//    - ClosestAvailable: fill ONLY at query-time (p_value) using closest fitted charge.
//
// Default below is MinimalDelta (explicit, no silent interpolation for internal gaps).

#[derive(Clone, Copy, Debug)]
pub enum ChargeFillMode {
    MinimalDelta,
    ClosestAvailable,
}

// -----------------------------------------------------------------------------
// LowerOrderModel (per-charge TNM parameters)
// -----------------------------------------------------------------------------
//
// Contract:
// - Stores per-charge TNM (mu, beta) pairs.
// - Provides p_value(score, charge) using the existing TEV-normalization path:
//       tev_norm = (score - mu) / beta
//       p = 1 - TEV_CDF_asymptotic(z, k=1)
// - Uses fail-closed fallback parameters when charge-specific params are absent.
//   In the current LO configuration these fallback params are NaN, causing p_value()
//   to return 1.0 unless an explicit charge-sharing rule supplies parameters.
//

#[derive(Clone, Debug)]
pub struct LowerOrderModel {
    pub params_by_charge: FnvHashMap<u8, (f64, f64)>,
    pub fallback_params: (f64, f64), // (mu, beta)

    // metadata (computed once at fit-time)
    pub charge_fill_mode: ChargeFillMode,
    pub fitted_charges_sorted: Vec<u8>,
    pub max_fitted_charge: u8, // 0 means "none fitted"
}

impl LowerOrderModel {
    #[inline]
    fn resolve_params(&self, charge: u8) -> (f64, f64) {
        // Fast path: exact charge fit exists
        if let Some(p) = self.params_by_charge.get(&charge).copied() {
            return p;
        }

        // Rule: charges above max fitted => copy max fitted
        if self.max_fitted_charge > 0 && charge > self.max_fitted_charge {
            if let Some(p) = self.params_by_charge.get(&self.max_fitted_charge).copied() {
                return p;
            }
        }

        // Internal gaps (or below-min / unknown charges)
        match self.charge_fill_mode {
            ChargeFillMode::MinimalDelta => self.fallback_params,

            // MinimalDelta: do NOT nearest-neighbor fill; rely on fallback_params.
            // ClosestAvailable: only fill at query-time using closest fitted charge
            ChargeFillMode::ClosestAvailable => {
                if self.fitted_charges_sorted.is_empty() {
                    return self.fallback_params;
                }

                let mut best = self.fitted_charges_sorted[0];
                let mut best_d = (best as i16 - charge as i16).abs();

                for &c in &self.fitted_charges_sorted[1..] {
                    let d = (c as i16 - charge as i16).abs();
                    if d < best_d {
                        best = c;
                        best_d = d;
                    }
                }

                self.params_by_charge
                    .get(&best)
                    .copied()
                    .unwrap_or(self.fallback_params)
            }
        }
    }

    #[inline]
    pub fn p_value(&self, score: f64, charge: u8) -> f64 {
        let (mu, beta) = self.resolve_params(charge);

        // Fail-closed semantics
        if !score.is_finite() || !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
            return 1.0;
        }

        let z = (score - mu) / beta;

        // Rank-1 asymptotic TEV CDF: for k = 1 this is the standard Gumbel-max CDF, exp(-exp(-z))
        let cdf = tev_cdf_asymptotic(z, 1);
        if !cdf.is_finite() {
            return 1.0; // fail-closed
        }

        let p = (1.0 - cdf).clamp(0.0, 1.0);
        p.max(1e-300)
    }
}

#[inline]
fn joint_nll_tev(mu: f64, beta: f64, buckets: &[RankBucket]) -> f64 {
    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 || buckets.is_empty() {
        return f64::INFINITY;
    }

    let mut total = 0.0f64;

    for b in buckets {
        if b.k < 2 || b.scores.is_empty() {
            return f64::INFINITY;
        }

        let nll = nll_tev_k(mu, beta, &b.scores, b.k);
        if !nll.is_finite() {
            return f64::INFINITY;
        }

        total += nll;
    }

    total
}

fn fit_joint_tev_mle(buckets: &[RankBucket]) -> Option<(f64, f64, f64)> {
    if buckets.is_empty() {
        return None;
    }

    let mut obs: Vec<(u32, f64)> = Vec::new();

    for b in buckets {
        if b.k < 2 || b.scores.is_empty() {
            continue;
        }

        for &x in &b.scores {
            if x.is_finite() {
                obs.push((b.k, x));
            }
        }
    }

    if obs.len() < 2 {
        return None;
    }

    let x_min = obs.iter().map(|&(_, x)| x).fold(f64::INFINITY, f64::min);

    if !x_min.is_finite() {
        return None;
    }

    let n_total = obs.len();
    let n_f64 = n_total as f64;

    let mut k_total_f64 = 0.0f64;
    let mut y_wk_sum = 0.0f64;
    let mut ys = Vec::with_capacity(obs.len());

    for &(k, x) in &obs {
        let y = x - x_min;

        if !y.is_finite() || y < 0.0 {
            return None;
        }

        let k_f64 = k as f64;
        k_total_f64 += k_f64;
        y_wk_sum += k_f64 * y;
        ys.push(y);
    }

    if k_total_f64 <= 0.0 {
        return None;
    }

    let r_factor = k_total_f64 / n_f64;
    let y_wk_bar = y_wk_sum / n_f64;

    let y_mean = ys.iter().sum::<f64>() / n_f64;
    let y_var = ys
        .iter()
        .map(|&y| {
            let d = y - y_mean;
            d * d
        })
        .sum::<f64>()
        / n_f64;

    let mut beta = (y_var * 6.0 / std::f64::consts::PI.powi(2))
        .sqrt()
        .max(1e-6);

    const MAX_ITERS: usize = 100;
    const TOL_ABS: f64 = 1e-8;

    let mut final_b_sum = 0.0f64;

    for _ in 0..MAX_ITERS {
        let mut b_sum = 0.0f64;
        let mut a_sum = 0.0f64;
        let mut c_sum = 0.0f64;

        for &y in &ys {
            let z = -y / beta;
            let e = z.exp();

            if !e.is_finite() {
                return None;
            }

            b_sum += e;
            a_sum += y * e;
            c_sum += y * y * e;
        }

        if b_sum <= 0.0 || !b_sum.is_finite() {
            return None;
        }

        final_b_sum = b_sum;

        let a_over_b = a_sum / b_sum;

        // Joint LO score equation in shifted coordinates.
        let f = beta - y_wk_bar + r_factor * a_over_b;

        let beta2 = beta * beta;

        // d/dβ Σ y exp(-y/β) = Σ y^2 exp(-y/β) / β^2
        let d_a = c_sum / beta2;

        // d/dβ Σ exp(-y/β) = Σ y exp(-y/β) / β^2
        let d_b = a_sum / beta2;

        let d_a_over_b = (d_a * b_sum - a_sum * d_b) / (b_sum * b_sum);
        let fp = 1.0 + r_factor * d_a_over_b;

        if !fp.is_finite() || fp.abs() < 1e-12 {
            return None;
        }

        let mut next_beta = beta - f / fp;

        if !next_beta.is_finite() {
            return None;
        }

        if next_beta <= 0.0 {
            next_beta = beta * 0.5;
        }

        if (next_beta - beta).abs() < TOL_ABS.max(TOL_ABS * beta.abs()) {
            beta = next_beta;
            break;
        }

        beta = next_beta;
    }

    if final_b_sum <= 0.0 || !final_b_sum.is_finite() {
        return None;
    }

    // For shifted y = x - x_min:
    // exp(mu / beta) * Σ exp(-x / beta) = K
    // Σ exp(-x / beta) = exp(-x_min / beta) * Σ exp(-y / beta)
    // so:
    // mu = x_min + beta * ln(K / Σ exp(-y / beta))
    let mu = x_min + beta * (k_total_f64 / final_b_sum).ln();

    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    let nll = joint_nll_tev(mu, beta, buckets);

    if !nll.is_finite() {
        return None;
    }

    log::info!(
        "LO joint-MLE solved: mu={:.6} beta={:.6} nll={:.4} total_obs={} k_total={:.1}",
        mu,
        beta,
        nll,
        n_total,
        k_total_f64
    );

    Some((mu, beta, nll))
}

/// Fits a charge-stratified Lower Order Model.
///
/// LO fitting contract
/// -------------------
/// - Rank 1 is never used for fitting.
/// - Every usable lower-order rank k in the configured window contributes to
///   one joint TEV likelihood.
/// - Each score is evaluated under its own k-specific order-statistic density.
/// - No autonomous single-rank selection is performed.
/// - No rank-1 target-contaminated scores are used to choose the TNM.
/// - Missing charge buckets fail closed unless covered by explicit charge-sharing.
pub fn fit_decoy_free_model(
    rank_null_stream: &[(u32, f64, u8)],
    rank1_stream: &[(f64, u8)],
    min_null_rank: u32,
    max_null_rank: u32,
    lo_min_count_per_rank: usize,
) -> Option<LowerOrderModel> {
    let effective_min_null_rank = min_null_rank.max(2);

    if effective_min_null_rank > max_null_rank {
        return None;
    }

    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "LO joint-MLE null-rank window: requested_min={} effective_min={} max={}",
            min_null_rank,
            effective_min_null_rank,
            max_null_rank
        );
    }

    // null_by_rank[z][k] -> Vec<f64>
    let mut null_by_rank: FnvHashMap<u8, FnvHashMap<u32, Vec<f64>>> = FnvHashMap::default();

    for &(rank, score, charge) in rank_null_stream {
        if rank < effective_min_null_rank || rank > max_null_rank {
            continue;
        }

        if !score.is_finite() {
            continue;
        }

        null_by_rank
            .entry(charge)
            .or_default()
            .entry(rank)
            .or_default()
            .push(score);
    }

    // Track rank-1 charges only to avoid fitting buckets that are never evaluated.
    let mut rank1_counts_by_charge: FnvHashMap<u8, usize> = FnvHashMap::default();
    for &(score, charge) in rank1_stream {
        if score.is_finite() {
            *rank1_counts_by_charge.entry(charge).or_insert(0) += 1;
        }
    }

    if log::log_enabled!(log::Level::Debug) {
        let mut summaries: Vec<String> = Vec::new();

        for (&charge, by_rank) in &null_by_rank {
            let mut ranks: Vec<u32> = by_rank.keys().copied().collect();
            ranks.sort_unstable();

            let rank_summary = ranks
                .into_iter()
                .map(|k| {
                    let n = by_rank.get(&k).map(|v| v.len()).unwrap_or(0);
                    format!("{k}:{n}")
                })
                .collect::<Vec<_>>()
                .join(",");

            summaries.push(format!("z{charge}[{rank_summary}]"));
        }

        summaries.sort();

        log::debug!(
            "LO joint-MLE usable null ranks by bucket before support gate: {}",
            summaries.join(" ")
        );
    }

    let mut params_by_charge: FnvHashMap<u8, (f64, f64)> = FnvHashMap::default();

    for (&charge, by_rank) in &null_by_rank {
        let rank1_n = rank1_counts_by_charge.get(&charge).copied().unwrap_or(0);
        if rank1_n == 0 {
            continue;
        }

        let mut buckets: Vec<RankBucket> = Vec::new();

        for k in effective_min_null_rank..=max_null_rank {
            let Some(scores_k) = by_rank.get(&k) else {
                continue;
            };

            let scores: Vec<f64> = scores_k.iter().copied().filter(|x| x.is_finite()).collect();

            if scores.len() < lo_min_count_per_rank {
                continue;
            }

            buckets.push(RankBucket { k, scores });
        }

        if buckets.is_empty() {
            log::debug!(
                "LO joint-MLE bucket {charge}: no ranks passed support gate min_count={}",
                lo_min_count_per_rank
            );
            continue;
        }

        let total_n = buckets.iter().map(|b| b.scores.len()).sum::<usize>();
        let rank_summary = buckets
            .iter()
            .map(|b| format!("{}:{}", b.k, b.scores.len()))
            .collect::<Vec<_>>()
            .join(",");

        let Some((mu, beta, nll)) = fit_joint_tev_mle(&buckets) else {
            log::warn!(
                "LO joint-MLE bucket {charge}: fit failed | ranks=[{}] total_null={}",
                rank_summary,
                total_n
            );
            continue;
        };

        log::info!(
            "LO joint-MLE bucket {charge}: mu={:.6} beta={:.6} joint_nll={:.4} ranks=[{}] total_null={} rank1_eval={}",
            mu,
            beta,
            nll,
            rank_summary,
            total_n,
            rank1_n
        );

        params_by_charge.insert(charge, (mu, beta));
    }

    let fallback_params = (f64::NAN, f64::NAN);

    // Explicit charge-sharing rule: if charge 1 is absent but charge 2 exists,
    // copy charge 2.  Other internal gaps remain fail-closed.
    if !params_by_charge.contains_key(&1) {
        if let Some(p2) = params_by_charge.get(&2).copied() {
            params_by_charge.insert(1, p2);
        }
    }

    if params_by_charge.is_empty() {
        return None;
    }

    let mut fitted_charges_sorted: Vec<u8> = params_by_charge.keys().copied().collect();
    fitted_charges_sorted.sort_unstable();
    let max_fitted_charge: u8 = fitted_charges_sorted.last().copied().unwrap_or(0);

    Some(LowerOrderModel {
        params_by_charge,
        fallback_params,
        charge_fill_mode: ChargeFillMode::MinimalDelta,
        fitted_charges_sorted,
        max_fitted_charge,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple deterministic RNG (no external deps)
    #[derive(Clone)]
    struct XorShift64 {
        state: u64,
    }
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self { state: seed.max(1) }
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }
        fn next_f64(&mut self) -> f64 {
            // (0,1)
            let u = self.next_u64();
            let v = ((u >> 11) as f64) / ((1u64 << 53) as f64);
            v.clamp(1e-12, 1.0 - 1e-12)
        }
    }

    fn sample_gumbel(mu: f64, beta: f64, u: f64) -> f64 {
        // inverse CDF: x = mu - beta * ln(-ln(u))
        mu - beta * (-u.ln()).ln()
    }

    fn sample_kth_largest_gumbel(
        mu: f64,
        beta: f64,
        n: usize,
        k: usize,
        rng: &mut XorShift64,
    ) -> f64 {
        // Draw n iid Gumbel, return k-th largest (1-indexed k)
        let mut xs: Vec<f64> = (0..n)
            .map(|_| sample_gumbel(mu, beta, rng.next_f64()))
            .collect();
        xs.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
        xs[(k - 1).min(xs.len() - 1)]
    }

    #[test]
    fn lo_estimators_recover_reasonable_params_and_joint_fit_is_finite() {
        // Synthetic parameters
        let mu_true = 15.0;
        let beta_true = 3.0;

        // Simulate "null" scores at a given hit rank k from N candidates
        let n_candidates = 2000usize;
        let k = 8usize;
        let n_samples = 4000usize;

        let mut rng = XorShift64::new(0xC0FFEE);

        let mut scores: Vec<f64> = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            scores.push(sample_kth_largest_gumbel(
                mu_true,
                beta_true,
                n_candidates,
                k,
                &mut rng,
            ));
        }

        // Moments estimator should be finite
        let (mu_mm, beta_mm) = fit_gumbel_moments(&scores);
        assert!(mu_mm.is_finite());
        assert!(beta_mm.is_finite());
        assert!(beta_mm > 0.0);

        // TEV-k moments/MLE should be finite (paper/PyLord helpers)
        let (mu_k_mm, beta_k_mm) = fit_tev_k_moments(&scores, k as u32).expect("tev-k moments");
        assert!(mu_k_mm.is_finite());
        assert!(beta_k_mm.is_finite());
        assert!(beta_k_mm > 0.0);

        let (mu_k_mle, beta_k_mle) = fit_tev_k_mle(&scores, k as u32).expect("tev-k mle");
        assert!(mu_k_mle.is_finite());
        assert!(beta_k_mle.is_finite());
        assert!(beta_k_mle > 0.0);

        // Loose sanity: we shouldn't be orders of magnitude off
        assert!((beta_k_mm / beta_true) > 0.2 && (beta_k_mm / beta_true) < 5.0);
        assert!((beta_k_mle / beta_true) > 0.2 && (beta_k_mle / beta_true) < 5.0);
    }

    #[test]
    fn pylord_parity_tev_cdf_asymptotic_matches_known_values() {
        // These expected values are from PyLord's:
        //   cdf_asymptotic(x, mu=0, beta=1, hit_rank=k)
        // evaluated at x=z (i.e., z = (x-mu)/beta).
        //
        // Asymptotic TEV CDF used by this implementation:
        //   cdf = exp(-exp(-z)) * Σ_{m=0..k-1} exp(-m z) / m!
        //
        // We hardcode a few representative (z,k) points.
        let cases: &[(f64, u32, f64)] = &[
            (-1.0, 1, 0.06598803584531254),
            (0.0, 1, 0.36787944117144233),
            (0.0, 3, 0.9196986029286058),
            (1.0, 1, 0.6922006275553464),
            (1.0, 3, 0.9936865915923109),
        ];

        for &(z, k, expected) in cases {
            let got = tev_cdf_asymptotic(z, k);
            assert!(
                got.is_finite(),
                "tev_cdf_asymptotic produced non-finite value for z={z}, k={k}"
            );
            let err = (got - expected).abs();
            assert!(
                err < 1e-12,
                "tev_cdf_asymptotic mismatch for z={z}, k={k}: got={got:.16e}, expected={expected:.16e}, err={err:.3e}",
            );
        }
    }

    #[test]
    fn pylord_parity_nll_tev_k_matches_asymptotic_gumbel_mle_k1() {
        // PyLord stat.py:
        // AsymptoticGumbelMLE.get_log_likelihood(log_params):
        //
        //   location, scale = exp(log_params) + 1e-10
        //   z = (scores - location)/scale
        //   factorial_term = factorial(hit_rank - 1)
        //   likelihood = -n*log(scale*factorial) - hit_rank*sum(z) - sum(exp(-z))
        //   return -likelihood     # negative log-likelihood
        //
        // Our nll_tev_k(mu,beta,ts,k) should match for k=1.
        let scores: [f64; 4] = [10.0, 11.0, 12.0, 13.5];
        let mu = 11.2;
        let beta = 2.3;
        let k = 1u32;

        // Expected value computed from the PyLord formula above (k=1 => factorial(0)=1).
        let expected_nll = 7.920672765212197_f64;

        let got = nll_tev_k(mu, beta, &scores, k);
        assert!(
            got.is_finite(),
            "nll_tev_k returned non-finite value: {got}"
        );
        let err = (got - expected_nll).abs();
        assert!(
            err < 1e-12,
            "nll_tev_k mismatch: got={got:.16e}, expected={expected_nll:.16e}, err={err:.3e}",
        );

        // Extra guard: re-compute the PyLord closed-form here and compare again.
        let n = scores.len() as f64;
        let factorial = 1.0_f64; // factorial(k-1) = factorial(0) = 1
        let mut sum_z = 0.0;
        let mut sum_exp_neg_z = 0.0;
        for &x in &scores {
            let z = (x - mu) / beta;
            sum_z += z;
            sum_exp_neg_z += (-z).exp();
        }
        let pylord_nll = n * (beta * factorial).ln() + (k as f64) * sum_z + sum_exp_neg_z;

        let err2 = (got - pylord_nll).abs();
        assert!(
            err2 < 1e-12,
            "nll_tev_k does not match reconstructed PyLord NLL: got={got:.16e}, pylord={pylord_nll:.16e}, err={err2:.3e}"
        );
    }
}
