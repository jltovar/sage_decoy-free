//! Decoy-free Lower-Order (LO) model fitting utilities.
//!
//! This module implements a Sage-specific deterministic LowerOrder estimator
//! inspired by the lower-order statistics framework of Madej and Lam:
//!
//!   Modeling Lower-Order Statistics to Enable Decoy-Free FDR Estimation in Proteomics
//!   Dominik Madej and Henry Lam
//!   Journal of Proteome Research 2023 22 (4), 1159-1171
//!   DOI: 10.1021/acs.jproteome.2c00604
//!
//!   Implemented on GitHub here:
//!   https://github.com/dommad/pylord
//!
//! The production path is not a direct PyLord port. It uses Sage-provided
//! spectrum-local E-values, a configurable TEV transform, per-rank LOM MLEs,
//! and one deterministic joint-MLE rank-1 TNM fit over the supported
//! lower-order rank buckets.

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

/// Gumbel MLE fit (mu, beta) using a Newton-Raphson update for beta,
/// followed by the closed-form mu update.
///
/// Solves for beta in:
///
///   f(beta) = beta - x_bar + A(beta) / B(beta) = 0
///
/// where:
///
///   B(beta) = Σ exp(-x / beta)
///   A(beta) = Σ x exp(-x / beta)
///
/// The implementation evaluates A/B and the derivative with centered
/// log-sum-exp weights so shifted scores or small beta values do not overflow.
pub(crate) fn fit_gumbel_mle(scores: &[f64]) -> Option<(f64, f64)> {
    let finite: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.len() < 2 {
        return None;
    }

    let n = finite.len() as f64;
    let x_bar = finite.iter().sum::<f64>() / n;
    if !x_bar.is_finite() {
        return None;
    }

    // Initialize with moments beta.
    let (_, mut beta) = fit_gumbel_moments(&finite);
    if !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    const MAX_ITERS: usize = 50;
    const TOL_ABS: f64 = 1e-8;
    const MIN_BETA: f64 = 1e-12;

    for _ in 0..MAX_ITERS {
        if !beta.is_finite() || beta <= MIN_BETA {
            return None;
        }

        // Stable evaluation of:
        //
        //   B = Σ exp(-x / beta)
        //   A = Σ x exp(-x / beta)
        //   C = Σ x² exp(-x / beta)
        //
        // Direct exp(-x / beta) can overflow when scores are shifted negative
        // or beta is small. Use centered weights:
        //
        //   z_i = -x_i / beta
        //   z_max = max_i z_i
        //   w_i = exp(z_i - z_max)
        //
        // The common exp(z_max) factor cancels in A/B, C/B, and d(A/B).
        let mut z_max = f64::NEG_INFINITY;
        for &x in &finite {
            let z = -x / beta;
            if !z.is_finite() {
                return None;
            }
            z_max = z_max.max(z);
        }

        if !z_max.is_finite() {
            return None;
        }

        let mut w_sum = 0.0f64;
        let mut x_w_sum = 0.0f64;
        let mut x2_w_sum = 0.0f64;

        for &x in &finite {
            let z = -x / beta;
            let w = (z - z_max).exp();

            if !w.is_finite() {
                return None;
            }

            w_sum += w;
            x_w_sum += x * w;
            x2_w_sum += x * x * w;
        }

        if !w_sum.is_finite() || w_sum <= 0.0 {
            return None;
        }

        let weighted_mean_x = x_w_sum / w_sum;
        let weighted_mean_x2 = x2_w_sum / w_sum;

        if !weighted_mean_x.is_finite() || !weighted_mean_x2.is_finite() {
            return None;
        }

        // f(beta) = beta - x_bar + A/B
        let f = beta - x_bar + weighted_mean_x;

        let beta2 = beta * beta;
        if !beta2.is_finite() || beta2 <= 0.0 {
            return None;
        }

        // d(A/B)/d(beta)
        //
        // Original expression:
        //
        //   d(A/B) = (dA*B - A*dB) / B²
        //
        // with:
        //
        //   dA = C / beta²
        //   dB = A / beta²
        //
        // Therefore:
        //
        //   d(A/B) = (C/B - (A/B)²) / beta²
        //
        // This weighted-moment form is equivalent and avoids overflow.
        let d_a_over_b = (weighted_mean_x2 - weighted_mean_x * weighted_mean_x) / beta2;

        // f'(beta) = 1 + d(A/B)
        let fp = 1.0 + d_a_over_b;

        if !f.is_finite() || !fp.is_finite() || fp.abs() < 1e-12 {
            return None;
        }

        let next_beta = beta - (f / fp);

        if !next_beta.is_finite() || next_beta <= MIN_BETA {
            return None;
        }

        if (next_beta - beta).abs() < TOL_ABS {
            beta = next_beta;
            break;
        }

        beta = next_beta;
    }

    // Closed-form mu:
    //
    //   mu = -beta * ln( (1/n) Σ exp(-x / beta) )
    //      = beta * (ln(n) - logsumexp(-x / beta))
    //
    // Compute logsumexp stably using the same centering trick.
    let mut z_max = f64::NEG_INFINITY;
    for &x in &finite {
        let z = -x / beta;
        if !z.is_finite() {
            return None;
        }
        z_max = z_max.max(z);
    }

    if !z_max.is_finite() {
        return None;
    }

    let mut w_sum = 0.0f64;
    for &x in &finite {
        let z = -x / beta;
        let w = (z - z_max).exp();

        if !w.is_finite() {
            return None;
        }

        w_sum += w;
    }

    if !w_sum.is_finite() || w_sum <= 0.0 {
        return None;
    }

    let log_sum_exp = z_max + w_sum.ln();
    if !log_sum_exp.is_finite() {
        return None;
    }

    let mu = beta * (n.ln() - log_sum_exp);

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
// Per-k LO estimators
// -----------------------------------------------------------------------------
//
// These helpers implement the asymptotic k-th lower-order Gumbel formulas used
// by the Madej/Lam lower-order framework. The formulas are compatible with the
// corresponding PyLord statistical expressions, but the production Sage LO path
// around them is a deterministic Sage-specific implementation.
//
// Method-of-moments form:
//
//   scale    = sqrt((E[X^2] - E[X]^2) / psi(k - 1))
//   location = E[X] - scale * (EulerGamma - H_{k - 1})
//
// where:
//
//   psi(m) = pi^2/6 - sum_{i=1..m} 1/i^2
//   H_m    = sum_{i=1..m} 1/i
//

#[inline]
fn harmonic(m: u32) -> f64 {
    if m == 0 {
        return 0.0;
    }
    let mut s = 0.0f64;
    for i in 1..=m {
        s += 1.0 / (i as f64);
    }
    s
}

#[inline]
fn psi_tail(m: u32) -> f64 {
    // psi(m) here is the finite trigamma-tail identity used by the
    // lower-order Gumbel moment formula:
    //
    //   psi(m) = pi^2/6 - sum_{i=1..m} 1/i^2
    //
    // It is not the digamma function.
    let mut s = 0.0f64;
    for i in 1..=m {
        let ii = i as f64;
        s += 1.0 / (ii * ii);
    }
    (std::f64::consts::PI * std::f64::consts::PI) / 6.0 - s
}

#[inline]
pub(crate) fn fit_tev_k_moments(scores: &[f64], k: u32) -> Option<(f64, f64)> {
    // k is the selected lower-order rank. Production LO uses k >= 2.
    if k < 2 {
        return None;
    }
    let xs: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
    if xs.len() < 2 {
        return None;
    }

    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let second = xs.iter().map(|x| x * x).sum::<f64>() / n;
    let var = second - mean * mean;
    if !var.is_finite() || var <= 0.0 {
        return None;
    }

    let denom = psi_tail(k - 1);
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }

    let beta = (var / denom).sqrt();
    if !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    // EulerGamma in PyLord: euler_m = -digamma(1) = EulerMascheroni
    let location = mean - beta * (EULER_MASCHERONI - harmonic(k - 1));
    if location.is_finite() {
        Some((location, beta))
    } else {
        None
    }
}

#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    if !a.is_finite() {
        return b;
    }
    if !b.is_finite() {
        return a;
    }

    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[inline]
fn nll_tev_k(mu: f64, beta: f64, scores: &[f64], k: u32) -> f64 {
    // Negative log-likelihood for the asymptotic k-th lower-order Gumbel model:
    //
    //   NLL = n * ln(beta * (k - 1)!) + k * Σz_i + Σexp(-z_i)
    //
    // where:
    //
    //   z_i = (x_i - mu) / beta
    //
    // This is the same likelihood family used by the Madej/Lam/PyLord
    // lower-order formulation.
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
    //
    //   F_k(z) = exp(-exp(-z)) * Σ_{m=0..k-1} exp(-m z) / m!
    //
    // Equivalently, with t = exp(-z):
    //
    //   F_k(z) = exp(-t) * Σ_{m=0..k-1} t^m / m!
    //
    // This is the Poisson CDF P[N <= k-1] for N ~ Poisson(t).
    //
    // The direct recurrence can overflow for large t and k:
    //
    //   term_m = term_{m-1} * t / m
    //
    // Therefore compute in log-space:
    //
    //   log F_k(z) = -t + logsumexp_{m=0..k-1}(m log(t) - log(m!)).
    //
    if !z.is_finite() || k < 1 {
        return f64::NAN;
    }

    let log_t = -z;

    // If log_t is so large that t = exp(log_t) would overflow, then
    // -t dominates the polynomial log-sum and the CDF is numerically zero.
    if log_t > f64::MAX.ln() {
        return 0.0;
    }

    // If log_t is very negative, then t ≈ 0 and F_k(z) ≈ 1 for all k >= 1.
    if log_t < -745.0 {
        return 1.0;
    }

    let t = log_t.exp();
    if !t.is_finite() || t < 0.0 {
        return f64::NAN;
    }

    let mut log_sum = f64::NEG_INFINITY;

    for m in 0..k {
        let m_f = m as f64;

        let log_term = if m == 0 {
            0.0
        } else {
            m_f * log_t - ln_gamma(m_f + 1.0)
        };

        log_sum = log_add_exp(log_sum, log_term);
    }

    let log_cdf = -t + log_sum;

    if log_cdf >= 0.0 {
        return 1.0;
    }

    if log_cdf < -745.0 {
        return 0.0;
    }

    log_cdf.exp().clamp(0.0, 1.0)
}

#[inline]
fn mu_bounds_from_scores(beta0: f64, scores: &[f64]) -> Option<(f64, f64)> {
    // Same spirit as your mu_search_bounds(): build a robust mu window on the score scale.
    let mut xs: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
    if xs.len() < 10 || !beta0.is_finite() || beta0 <= 0.0 {
        return None;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = xs.len();
    let p05 = xs[((0.05 * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    let p95 = xs[((0.95 * (n as f64 - 1.0)).round() as usize).min(n - 1)];

    let lo = p05 - 4.0 * beta0;
    let hi = p95 + 4.0 * beta0;

    (lo.is_finite() && hi.is_finite() && lo < hi).then_some((lo, hi))
}

pub(crate) fn fit_tev_k_mle(scores: &[f64], k: u32) -> Option<(f64, f64)> {
    if k < 2 {
        return None;
    }
    let xs: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
    if xs.len() < 2 {
        return None;
    }

    // Init from MM (PyLord uses mean/std, but MM is the best deterministic analog and
    // keeps us in a sane region).
    let (mu0, beta0) = fit_tev_k_moments(&xs, k)?;
    if !mu0.is_finite() || !beta0.is_finite() || beta0 <= 0.0 {
        return None;
    }

    let (mu_min, mu_max) = mu_bounds_from_scores(beta0, &xs).unwrap_or_else(|| {
        // fallback to min/max if percentiles unavailable
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for &v in &xs {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        (mn, mx)
    });

    // ----- coarse search -----
    let mut best: Option<(f64, f64, f64)> = None; // (mu, beta, nll)

    // beta grid around beta0 in log2 space (stable, scale-aware)
    for p in -6..=6 {
        let beta = beta0 * 2.0f64.powi(p);
        if !beta.is_finite() || beta <= 0.0 {
            continue;
        }

        const MU_N: usize = 96;
        let step = (mu_max - mu_min) / ((MU_N - 1) as f64);
        if !step.is_finite() || step <= 0.0 {
            continue;
        }

        for i in 0..MU_N {
            let mu = mu_min + (i as f64) * step;
            let nll = nll_tev_k(mu, beta, &xs, k);
            if !nll.is_finite() {
                continue;
            }
            match best {
                None => best = Some((mu, beta, nll)),
                Some((_, _, best_nll)) if nll < best_nll => best = Some((mu, beta, nll)),
                _ => {}
            }
        }
    }

    let (mut mu_best, mut beta_best, mut nll_best) = best?;

    // ----- refinement -----
    // refine beta log2 step=0.25 around current best, and mu local window
    for q in -8..=8 {
        let beta = beta_best * 2.0f64.powf((q as f64) / 4.0);
        if !beta.is_finite() || beta <= 0.0 {
            continue;
        }

        const MU_LOCAL_HALF_WIDTH: f64 = 4.0; // score units; local refinement
        let lo = (mu_best - MU_LOCAL_HALF_WIDTH).clamp(mu_min, mu_max);
        let hi = (mu_best + MU_LOCAL_HALF_WIDTH).clamp(mu_min, mu_max);

        const MU_N_LOCAL: usize = 81;
        let step = if MU_N_LOCAL > 1 {
            (hi - lo) / ((MU_N_LOCAL - 1) as f64)
        } else {
            0.0
        };
        if !step.is_finite() {
            continue;
        }

        for i in 0..MU_N_LOCAL {
            let mu = lo + (i as f64) * step;
            let nll = nll_tev_k(mu, beta, &xs, k);
            if !nll.is_finite() {
                continue;
            }
            if nll < nll_best {
                mu_best = mu;
                beta_best = beta;
                nll_best = nll;
            }
        }
    }

    (mu_best.is_finite() && beta_best.is_finite() && beta_best > 0.0)
        .then_some((mu_best, beta_best))
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

// -----------------------------------------------------------------------------
// Charge-stratified TNM fitter
// -----------------------------------------------------------------------------
//
// Inputs:
// - rank-null stream: (rank, TEV score, charge) for selected lower-order ranks.
// - rank-1 stream:    (TEV score, charge) for top-scoring PSMs.
//
// Output:
// - LowerOrderModel with params_by_charge[z] = (mu, beta) for the selected
//   top null model (TNM) per charge.
//
// Deterministic LO contract:
// - LowerOrder is fit on TEV scores derived from Sage spectrum-local E-values.
// - The TEV scale is selected upstream by LoTevTransform.
// - Rank 1 is never used as lower-order null evidence.
// - Each selected lower-order rank k is modeled with its k-specific TEV likelihood.
// - Only MLE LOMs are used for the production TNM path.
// - The β(μ) linear trend is reported as a diagnostic across supported LOMs.
// - The rank-1 TNM is fit by one deterministic joint likelihood over all
//   supported lower-order rank buckets.
// - No rank-1 score likelihood is used to fit or select the null.
// - No PyLord-style autonomous selection is performed:
//     no best 5-rank LR window,
//     no MM/MLE switching,
//     no mean-beta fallback,
//     no candidate-family competition.
//
// Support gating:
// - lo_min_count_per_rank is the minimum number of finite observations required
//   for an individual selected lower-order rank k to contribute.
// - A charge is fit only if enough selected ranks satisfy this support threshold
//   and yield finite MLE LOM parameters.
//
// Fail-closed behavior:
// - No pooled/global fallback TNM is used. Missing charges return p=1.0 unless
//   covered by explicit charge-sharing rules.

#[inline]
fn finite_quantiles(xs: &[f64]) -> Option<(f64, f64, f64, f64, f64, f64, f64)> {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }

    v.sort_by(|a, b| a.total_cmp(b));

    let q = |p: f64| -> f64 {
        let idx = (p.clamp(0.0, 1.0) * ((v.len() - 1) as f64)).round() as usize;
        v[idx.min(v.len() - 1)]
    };

    Some((
        q(0.00),
        q(0.01),
        q(0.10),
        q(0.50),
        q(0.90),
        q(0.99),
        q(1.00),
    ))
}

#[inline]
fn median_f64(mut xs: Vec<f64>) -> Option<f64> {
    xs.retain(|x| x.is_finite());
    if xs.is_empty() {
        return None;
    }

    xs.sort_by(|a, b| a.total_cmp(b));

    let n = xs.len();
    if n % 2 == 1 {
        Some(xs[n / 2])
    } else {
        Some(0.5 * (xs[n / 2 - 1] + xs[n / 2]))
    }
}

const LO_MIN_LOM_RANKS: usize = 2;
const LO_TNM_BETA_REL_FLOOR: f64 = 1e-4;
const LO_TNM_BETA_ABS_FLOOR: f64 = 1e-8;

#[inline]
fn ols_beta_on_mu_all_supported_ranks(loms: &[(u32, f64, f64)]) -> Option<(f64, f64, f64)> {
    let rows: Vec<(f64, f64)> = loms
        .iter()
        .filter_map(|(_, mu, beta)| {
            if mu.is_finite() && beta.is_finite() && *beta > 0.0 {
                Some((*mu, *beta))
            } else {
                None
            }
        })
        .collect();

    if rows.len() < 2 {
        return None;
    }

    let n = rows.len() as f64;
    let mean_x = rows.iter().map(|(mu, _)| *mu).sum::<f64>() / n;
    let mean_y = rows.iter().map(|(_, beta)| *beta).sum::<f64>() / n;

    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;

    for (x, y) in &rows {
        let dx = *x - mean_x;
        let dy = *y - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    if !sxx.is_finite() || sxx <= 0.0 {
        return None;
    }

    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;

    if !slope.is_finite() || !intercept.is_finite() {
        return None;
    }

    let r = if sxx > 0.0 && syy > 0.0 {
        (sxy / (sxx.sqrt() * syy.sqrt())).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    Some((slope, intercept, r))
}

#[inline]
fn joint_nll_tev_buckets(mu: f64, beta: f64, buckets: &[RankBucket]) -> f64 {
    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 || buckets.is_empty() {
        return f64::INFINITY;
    }

    let mut total = 0.0f64;

    for b in buckets {
        let nll = nll_tev_k(mu, beta, &b.scores, b.k);
        if !nll.is_finite() {
            return f64::INFINITY;
        }
        total += nll;
    }

    total
}

fn fit_joint_tnm_mle_from_loms(
    buckets: &[RankBucket],
    lom_mle: &[(u32, f64, f64)],
) -> Option<(f64, f64, f64)> {
    if buckets.is_empty() || lom_mle.len() < LO_MIN_LOM_RANKS {
        return None;
    }

    let mu0 = median_f64(lom_mle.iter().map(|(_, mu, _)| *mu).collect())?;
    let beta0 = median_f64(lom_mle.iter().map(|(_, _, beta)| *beta).collect())?;

    if !mu0.is_finite() || !beta0.is_finite() || beta0 <= 0.0 {
        return None;
    }

    let beta_floor = (beta0 * LO_TNM_BETA_REL_FLOOR).max(LO_TNM_BETA_ABS_FLOOR);
    let mu_half_width = (6.0 * beta0).max(1.0);

    let mut best: Option<(f64, f64, f64)> = None;

    for b_step in -12..=12 {
        let beta = beta0 * 2.0f64.powf((b_step as f64) / 4.0);
        if !beta.is_finite() || beta <= beta_floor {
            continue;
        }

        let mu_lo = mu0 - mu_half_width;
        let mu_hi = mu0 + mu_half_width;

        const MU_N: usize = 161;
        let step = (mu_hi - mu_lo) / ((MU_N - 1) as f64);

        if !step.is_finite() || step <= 0.0 {
            continue;
        }

        for i in 0..MU_N {
            let mu = mu_lo + (i as f64) * step;
            let nll = joint_nll_tev_buckets(mu, beta, buckets);

            if !nll.is_finite() {
                continue;
            }

            match best {
                None => best = Some((mu, beta, nll)),
                Some((_, _, best_nll)) if nll < best_nll => best = Some((mu, beta, nll)),
                _ => {}
            }
        }
    }

    let (mut mu_best, mut beta_best, mut nll_best) = best?;

    for round in 0..3 {
        let mu_half_width = beta_best
            * match round {
                0 => 2.0,
                1 => 0.75,
                _ => 0.25,
            };

        let beta_log_half_width = match round {
            0 => 1.0,
            1 => 0.5,
            _ => 0.25,
        };

        const MU_N_LOCAL: usize = 81;
        const BETA_N_LOCAL: usize = 41;

        let mu_lo = mu_best - mu_half_width;
        let mu_hi = mu_best + mu_half_width;

        for bi in 0..BETA_N_LOCAL {
            let frac_b = (bi as f64) / ((BETA_N_LOCAL - 1) as f64);
            let log2_delta = -beta_log_half_width + 2.0 * beta_log_half_width * frac_b;
            let beta = beta_best * 2.0f64.powf(log2_delta);

            if !beta.is_finite() || beta <= beta_floor {
                continue;
            }

            for mi in 0..MU_N_LOCAL {
                let frac_m = (mi as f64) / ((MU_N_LOCAL - 1) as f64);
                let mu = mu_lo + (mu_hi - mu_lo) * frac_m;

                let nll = joint_nll_tev_buckets(mu, beta, buckets);
                if !nll.is_finite() {
                    continue;
                }

                if nll < nll_best {
                    mu_best = mu;
                    beta_best = beta;
                    nll_best = nll;
                }
            }
        }
    }

    if mu_best.is_finite()
        && beta_best.is_finite()
        && beta_best > beta_floor
        && nll_best.is_finite()
    {
        Some((mu_best, beta_best, nll_best))
    } else {
        None
    }
}

/// Fits a charge-stratified Lower Order Model.
///
/// Production LO path:
/// - Uses every supported selected lower-order rank in the configured window.
/// - Fits MLE LOM parameters for each supported lower-order rank.
/// - Fits one deterministic β(μ) trend across those LOMs for diagnostics.
/// - Fits the rank-1 TNM by one deterministic joint likelihood over all
///   supported lower-order rank buckets.
/// - Never uses rank-1 scores to fit or select the null.
/// - Does not perform PyLord-style autonomous candidate selection.
/// - Preserves one external null-rank window -> one deterministic LO model.
///
/// Fail-closed by default, with explicit charge-sharing rules for selected
/// missing-charge cases.
pub fn fit_decoy_free_model(
    rank_null_stream: &[(u32, f64, u8)],
    rank1_stream: &[(f64, u8)],
    min_null_rank: u32,
    max_null_rank: u32,
    lo_min_count_per_rank: usize,
) -> Option<LowerOrderModel> {
    // LowerOrder null evidence must come from non-top hits only.
    // Rank 1 is the target-contaminated top-hit mixture and is never a valid
    // lower-order null rank.
    let effective_min_null_rank = min_null_rank.max(2);

    if effective_min_null_rank > max_null_rank {
        log::error!(
			"LO INVALID WINDOW: requested null-rank window [{}..={}] contains no usable lower-order ranks. \
			 LowerOrder never uses rank 1 because rank 1 is the target-contaminated top-hit mixture. \
			 LowerOrder failed closed; no LO model was fit.",
			min_null_rank,
			max_null_rank
		);
        return None;
    }

    if effective_min_null_rank == max_null_rank {
        log::error!(
			"LO INVALID WINDOW: effective null-rank window [{}..={}] contains only one usable lower-order rank. \
			 This implementation requires at least two usable lower-order ranks to estimate the rank-to-rank β(μ) trend. \
			 LowerOrder failed closed; no LO model was fit.",
			effective_min_null_rank,
			max_null_rank
		);
        return None;
    }

    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "LO DEBUG null-rank window: requested_min={} effective_min={} max={}",
            min_null_rank,
            effective_min_null_rank,
            max_null_rank
        );
    }

    // -------------------------
    // (A) Build per-charge datasets
    // -------------------------
    // null_by_rank[z][k] -> Vec<f64>
    let mut null_by_rank: FnvHashMap<u8, FnvHashMap<u32, Vec<f64>>> = FnvHashMap::default();
    let mut pooled_null_scores: Vec<f64> = Vec::new();

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

        pooled_null_scores.push(score);
    }

    if log::log_enabled!(log::Level::Debug) {
        let mut v: Vec<(u8, usize)> = null_by_rank
            .iter()
            .map(|(z, by_rank)| {
                let n = by_rank.values().map(|scores| scores.len()).sum::<usize>();
                (*z, n)
            })
            .collect();

        v.sort_by_key(|(z, _)| *z);

        let mut s = String::new();
        for (z, n) in v {
            if !s.is_empty() {
                s.push_str(", ");
            }
            s.push_str(&format!("z{}={}", z, n));
        }

        log::debug!("LO DEBUG null totals by charge (post rank filter): {}", s);
        log::debug!(
            "LO DEBUG pooled_null_scores.len()={}",
            pooled_null_scores.len()
        );
    }

    // top_scores[z] = Vec<f64>
    let mut top_scores: FnvHashMap<u8, Vec<f64>> = FnvHashMap::default();
    for &(score, charge) in rank1_stream {
        if score.is_finite() {
            top_scores.entry(charge).or_default().push(score);
        }
    }

    // -------------------------
    // (B)(C)(D) Fit & select TNM per charge
    // -------------------------
    let mut params_by_charge: FnvHashMap<u8, (f64, f64)> = FnvHashMap::default();

    for (&charge, by_rank) in &null_by_rank {
        // Require rank-1 scores for this charge so the fitted model is only built
        // for charge states observed in the rank-1 evaluation stream.
        if !top_scores
            .get(&charge)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            continue;
        }

        // Build rank buckets for k in [min_null_rank..=max_null_rank]
        // using the per-rank null scores for this charge.
        let mut buckets: Vec<RankBucket> = Vec::new();

        for k in effective_min_null_rank..=max_null_rank {
            let scores_k = match by_rank.get(&k) {
                Some(v) if v.len() >= lo_min_count_per_rank => v,
                _ => continue,
            };

            // Collect finite values only. A selected lower-order rank contributes
            // only if it has enough finite observations for this charge.
            let scores: Vec<f64> = scores_k.iter().copied().filter(|x| x.is_finite()).collect();
            if scores.len() < lo_min_count_per_rank {
                continue;
            }

            buckets.push(RankBucket { k, scores });
        }

        // LowerOrder needs at least two usable selected ranks to estimate the
        // rank-to-rank β(μ) trend used by the TNM candidates.
        if buckets.len() < 2 {
            log::warn!(
				"LO charge {} skipped: only {} usable lower-order rank bucket(s) survived filtering in effective window [{}..={}]; need at least 2. \
				 This charge failed closed.",
				charge,
				buckets.len(),
				effective_min_null_rank,
				max_null_rank
			);
            continue;
        }

        if log::log_enabled!(log::Level::Info) {
            for b in &buckets {
                if let Some((q0, q1, q10, q50, q90, q99, q100)) = finite_quantiles(&b.scores) {
                    log::info!(
                "LO bucket diagnostics charge={} rank={} n={} tev_q=[{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}]",
                charge,
                b.k,
                b.scores.len(),
                q0,
                q1,
                q10,
                q50,
                q90,
                q99,
                q100
            );
                }
            }
        }

        // ---------------------------------------------------------------------
        // Deterministic joint-MLE TNM construction
        // ---------------------------------------------------------------------
        //
        // One external null-rank window produces one deterministic LO fit.
        // Rank-1 scores are not used to fit or select the null. The final rank-1
        // TNM is fit by one joint likelihood over all supported lower-order rank
        // buckets.

        let mut lom_mle: Vec<(u32, f64, f64)> = Vec::new();

        for b in &buckets {
            match fit_tev_k_mle(&b.scores, b.k) {
                Some((mu_k, beta_k)) if mu_k.is_finite() && beta_k.is_finite() && beta_k > 0.0 => {
                    let nll_k = nll_tev_k(mu_k, beta_k, &b.scores, b.k);
                    log::info!(
                        "LO LOM MLE diagnostics charge={} rank={} n={} mu_k={:.6} beta_k={:.6} nll_k={:.4}",
                        charge,
                        b.k,
                        b.scores.len(),
                        mu_k,
                        beta_k,
                        nll_k
                    );
                    lom_mle.push((b.k, mu_k, beta_k));
                }
                _ => {
                    log::warn!(
                        "LO LOM MLE diagnostics charge={} rank={} n={} fit_failed",
                        charge,
                        b.k,
                        b.scores.len()
                    );
                }
            }
        }

        if lom_mle.len() < LO_MIN_LOM_RANKS {
            log::warn!(
                "LO joint-MLE charge {charge}: insufficient supported LOM ranks: have={} need={}",
                lom_mle.len(),
                LO_MIN_LOM_RANKS
            );
            continue;
        }

        let Some((slope, intercept, r)) = ols_beta_on_mu_all_supported_ranks(&lom_mle) else {
            log::warn!(
                "LO joint-MLE charge {charge}: failed diagnostic β(μ) regression across {} LOM ranks",
                lom_mle.len()
            );
            continue;
        };

        let rank_summary = lom_mle
            .iter()
            .map(|(k, mu, beta)| format!("{}:{:.5}/{:.5}", k, mu, beta))
            .collect::<Vec<_>>()
            .join(",");

        let Some((mu_final, beta_final, joint_nll)) =
            fit_joint_tnm_mle_from_loms(&buckets, &lom_mle)
        else {
            log::warn!(
                "LO joint-MLE charge {charge}: TNM joint fit failed | lom_ranks={} loms=[{}]",
                lom_mle.len(),
                rank_summary
            );
            continue;
        };

        log::info!(
			"LO joint-MLE charge={} mu={:.6} beta={:.6} joint_nll={:.4} beta_mu_slope={:.6} beta_mu_intercept={:.6} beta_mu_r={:.4} lom_ranks={} loms=[{}]",
			charge,
			mu_final,
			beta_final,
			joint_nll,
			slope,
			intercept,
			r,
			lom_mle.len(),
			rank_summary
		);

        params_by_charge.insert(charge, (mu_final, beta_final));
    }

    // -------------------------
    // Fallback params (global)
    // -------------------------
    // Fail-closed: LO must not fall back to any other model (including pooled moments).
    // If a charge is missing params, resolve_params() will return fallback_params, and
    // p_value() will return 1.0 when mu/beta are NaN.
    let fallback_params = (f64::NAN, f64::NAN);

    // -------------------------
    // Charge filling rules (fit-time + query-time)
    // -------------------------

    // Rule: charges above max fitted => copy max fitted
    if !params_by_charge.contains_key(&1) {
        if let Some(p2) = params_by_charge.get(&2).copied() {
            params_by_charge.insert(1, p2);
        }
    }

    // Cache fitted charges + max fitted charge
    let mut fitted_charges_sorted: Vec<u8> = params_by_charge.keys().copied().collect();
    fitted_charges_sorted.sort_unstable();
    let max_fitted_charge: u8 = fitted_charges_sorted.last().copied().unwrap_or(0);

    if params_by_charge.is_empty() {
        log::error!(
            "LO failed closed: no charge state produced a valid LowerOrder model. \
			 No LO p-values, PEPs, or q-values should be interpreted as valid."
        );
        return None;
    }

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
    fn lo_estimators_recover_reasonable_params_and_selection_is_finite() {
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

    #[cfg(test)]
    mod tests {
        use super::*;

        // existing tests above...

        #[test]
        fn fit_gumbel_mle_is_stable_for_shifted_scores() {
            let scores = vec![
                -10_000.0, -9_999.5, -9_999.0, -9_998.4, -9_997.9, -9_997.2, -9_996.8, -9_996.1,
                -9_995.5, -9_995.0,
            ];

            let Some((mu, beta)) = fit_gumbel_mle(&scores) else {
                panic!("fit_gumbel_mle returned None for finite shifted scores");
            };

            assert!(mu.is_finite(), "mu is not finite: {mu}");
            assert!(beta.is_finite(), "beta is not finite: {beta}");
            assert!(beta > 0.0, "beta must be positive: {beta}");
        }

        // existing tests below...
    }

    #[test]
    fn pylord_parity_tev_cdf_asymptotic_matches_known_values() {
        // These expected values exercise the asymptotic k-th lower-order Gumbel CDF:
        //   F_k(z) = exp(-exp(-z)) * Σ_{m=0..k-1} exp(-m z) / m!
        // evaluated directly on standardized z values.
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
    fn tev_cdf_asymptotic_is_finite_for_extreme_inputs() {
        let cases: &[(f64, u32)] = &[
            (-1_000.0, 1),
            (-1_000.0, 2),
            (-1_000.0, 10),
            (-100.0, 50),
            (100.0, 1),
            (100.0, 10),
        ];

        for &(z, k) in cases {
            let got = tev_cdf_asymptotic(z, k);
            assert!(
                got.is_finite(),
                "tev_cdf_asymptotic produced non-finite value for z={z}, k={k}: {got}"
            );
            assert!(
                (0.0..=1.0).contains(&got),
                "tev_cdf_asymptotic outside [0,1] for z={z}, k={k}: {got}"
            );
        }

        assert_eq!(tev_cdf_asymptotic(-1_000.0, 10), 0.0);
        assert_eq!(tev_cdf_asymptotic(1_000.0, 10), 1.0);
    }

    #[test]
    fn pylord_parity_nll_tev_k_matches_asymptotic_gumbel_mle_k1() {
        // Validate the asymptotic k-th lower-order Gumbel NLL for k=1:
        //
        //   NLL = n * ln(beta * (k - 1)!) + k * Σz_i + Σexp(-z_i)
        //
        // with:
        //
        //   z_i = (x_i - mu) / beta
        //
        // For k=1, (k - 1)! = 1.
        let scores: [f64; 4] = [10.0, 11.0, 12.0, 13.5];
        let mu = 11.2;
        let beta = 2.3;
        let k = 1u32;

        // Expected value computed from the closed-form NLL above.
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

        // Extra guard: re-compute the closed-form NLL here and compare again.
        let n = scores.len() as f64;
        let factorial = 1.0_f64; // factorial(k-1) = factorial(0) = 1
        let mut sum_z = 0.0;
        let mut sum_exp_neg_z = 0.0;
        for &x in &scores {
            let z = (x - mu) / beta;
            sum_z += z;
            sum_exp_neg_z += (-z).exp();
        }
        let closed_form_nll = n * (beta * factorial).ln() + (k as f64) * sum_z + sum_exp_neg_z;

        let err2 = (got - closed_form_nll).abs();
        assert!(
            err2 < 1e-12,
            "nll_tev_k does not match reconstructed closed-form NLL: got={got:.16e}, closed_form={closed_form_nll:.16e}, err={err2:.3e}"
        );
    }
}
