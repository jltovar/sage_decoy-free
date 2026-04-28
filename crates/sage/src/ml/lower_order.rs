//! Decoy-free Lower-Order (LO) model fitting utilities.
//!
//! The methods in this module are based on the work of Dominik Madej and Henry Lam published here:
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
    // psi(m) in PyLord = pi^2/6 - sum_{i=1..m} 1/i^2
    // NOTE: This is NOT digamma; it's the trigamma tail identity used in the paper/PyLord.
    let mut s = 0.0f64;
    for i in 1..=m {
        let ii = i as f64;
        s += 1.0 / (ii * ii);
    }
    (std::f64::consts::PI * std::f64::consts::PI) / 6.0 - s
}

#[inline]
pub(crate) fn fit_tev_k_moments(scores: &[f64], k: u32) -> Option<(f64, f64)> {
    // k is hit_rank in PyLord. LO uses k>=2.
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
// - rank-null pool stream: (rank, score, charge) for ranks in
//   [lower_order_min_null_rank..=lower_order_max_null_rank] (clamped within global window)
// - rank1 score stream:    (score, charge) for rank 1
//
// Output:
// - LowerOrderModel with params_by_charge[z] = (mu, beta) for the selected TNM
//   per charge, chosen by minimum TEV negative log-likelihood across 4 candidates:
//     (LR vs mean-β) × (MLE vs moments)
//
// Notes:
// - This function is intentionally self-contained (no FdrSettings dependency).
// - All support gating is per charge and per selected lower-order rank:
//     lo_min_count_per_rank     minimum finite observations required for an
//                               individual selected rank k to contribute a LOM fit.
// - There is no separate total bucket-size threshold here. A charge is fit only
//   if at least two selected ranks each satisfy lo_min_count_per_rank and yield
//   finite LOM parameters.
// - μ scan range is data-driven:
//     * PyLord-style [0.1 * mean, 1.5 * mean] when the score scale is positive
//     * score-range fallback when the score scale is centered / near-zero / negative
//

#[inline]
fn median_of_finite(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut xs: Vec<f64> = values.into_iter().filter(|x| x.is_finite()).collect();
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

#[inline]
fn median_lom_params(loms: &[(u32, f64, f64)]) -> Option<(f64, f64)> {
    if loms.len() < 2 {
        return None;
    }

    let mu = median_of_finite(loms.iter().map(|(_, mu, _)| *mu))?;
    let beta = median_of_finite(loms.iter().map(|(_, _, beta)| *beta))?;

    if mu.is_finite() && beta.is_finite() && beta > 0.0 {
        Some((mu, beta))
    } else {
        None
    }
}

#[inline]
fn lower_order_total_nll(buckets: &[RankBucket], mu: f64, beta: f64) -> Option<f64> {
    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    let mut total = 0.0f64;

    for b in buckets {
        if b.k < 2 || b.scores.len() < 2 {
            continue;
        }

        let nll = nll_tev_k(mu, beta, &b.scores, b.k);
        if !nll.is_finite() {
            return None;
        }

        total += nll;
    }

    total.is_finite().then_some(total)
}

/// Fits a charge-stratified Lower Order Model.
///
/// LO fitting contract (Decoy-Free)
/// --------------------------------
/// This module fits a charge-stratified Target Null Model (TNM) and uses it to
/// produce rank-1 p-values.
///
/// Implementation Details (PyLord-Inspired LO):
/// - Fits LOM parameters (μ_k, β_k) per charge for ranks k within the
///   configured LO null-rank window using Method of Moments (MM) and/or MLE.
/// - Derives Rank 1 TNM candidates using Linear Regression and/or Mean-β modes.
/// - Evaluates candidates over a configured μ scan range.
/// - Selects the best candidate by minimum negative log-likelihood (NLL).
///   For the candidate TNMs compared here, this is equivalent to BIC up to an additive constant.
/// - Fail-closed by default, with explicit charge-sharing rules for selected
///   missing-charge cases (e.g. charge 1 inherits charge 2 when available, and
///   charges above the fitted range inherit the highest fitted charge).
///
pub fn fit_decoy_free_model(
    rank_null_stream: &[(u32, f64, u8)],
    rank1_stream: &[(f64, u8)],
    min_null_rank: u32,
    max_null_rank: u32,
    lo_min_count_per_rank: usize,
    lo_mode: crate::input::LoMode,
    lo_lom_estimator: crate::input::LoLomEstimator,
    lo_mean_beta_mode: crate::input::LoMeanBetaMode,
) -> Option<LowerOrderModel> {
    let lo_mean_beta_min_rank = min_null_rank;
    let lo_mean_beta_count = max_null_rank
        .saturating_sub(min_null_rank)
        .saturating_add(1);
    let lo_lr_window_size = Some(lo_mean_beta_count);

    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "LO DEBUG derived subwindow controls: mean_beta_start={} mean_beta_count={} lr_window_size={:?}",
            lo_mean_beta_min_rank,
            lo_mean_beta_count,
            lo_lr_window_size
        );
    }

    // -------------------------
    // (A) Build per-charge datasets
    // -------------------------
    // null_by_rank[z][k] -> Vec<f64>
    let mut null_by_rank: FnvHashMap<u8, FnvHashMap<u32, Vec<f64>>> = FnvHashMap::default();
    let mut pooled_null_scores: Vec<f64> = Vec::new();

    for &(rank, score, charge) in rank_null_stream {
        if rank < min_null_rank || rank > max_null_rank {
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

        for k in min_null_rank..=max_null_rank {
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
        // rank-to-rank trend used by the TNM candidates.
        if buckets.len() < 2 {
            continue;
        }

        // ---------------------------------------------------------------------
        // LowerOrder TNM construction
        // ---------------------------------------------------------------------
        //
        // The per-rank LowerOrder estimators already convert rank-k score samples
        // into estimates of the rank-1 target-null model parameters (mu, beta).
        //
        // Therefore the TNM should be derived from the lower-order LOM estimates
        // themselves. Do not re-fit mu against rank-1 scores here: rank-1 scores
        // are target-contaminated, and KDE-based rank-1 filtering makes the model
        // discontinuous across nearby null-rank windows.
        //
        // We build MM and/or MLE LOM tables, robustly aggregate their implied TNM
        // parameters, and select the candidate that best explains the lower-order
        // buckets by their own rank-k likelihood.
        let _ = lo_mode;
        let _ = &lo_mean_beta_mode;

        // 1) Build LOM tables for each estimator family.
        // Each entry is the rank-1 TNM estimate implied by lower-order rank k:
        //     (k, mu, beta)
        let mut lom_mm: Vec<(u32, f64, f64)> = Vec::new();
        let mut lom_mle: Vec<(u32, f64, f64)> = Vec::new();

        for b in &buckets {
            let k = b.k;
            if k < min_null_rank || k > max_null_rank {
                continue;
            }
            if b.scores.len() < lo_min_count_per_rank {
                continue;
            }

            if let Some((mu_k, beta_k)) = fit_tev_k_moments(&b.scores, k) {
                if mu_k.is_finite() && beta_k.is_finite() && beta_k > 0.0 {
                    lom_mm.push((k, mu_k, beta_k));
                }
            }

            if let Some((mu_k, beta_k)) = fit_tev_k_mle(&b.scores, k) {
                if mu_k.is_finite() && beta_k.is_finite() && beta_k > 0.0 {
                    lom_mle.push((k, mu_k, beta_k));
                }
            }
        }

        if lom_mm.len() < 2 && lom_mle.len() < 2 {
            continue;
        }

        let use_mm = matches!(
            lo_lom_estimator,
            crate::input::LoLomEstimator::Auto | crate::input::LoLomEstimator::Mm
        );
        let use_mle = matches!(
            lo_lom_estimator,
            crate::input::LoLomEstimator::Auto | crate::input::LoLomEstimator::Mle
        );

        let mut candidates: Vec<(&'static str, (f64, f64, f64))> = Vec::new();

        if use_mm {
            if let Some((mu, beta)) = median_lom_params(&lom_mm) {
                if let Some(nll) = lower_order_total_nll(&buckets, mu, beta) {
                    candidates.push(("LO/MM", (mu, beta, nll)));
                }
            }
        }

        if use_mle {
            if let Some((mu, beta)) = median_lom_params(&lom_mle) {
                if let Some(nll) = lower_order_total_nll(&buckets, mu, beta) {
                    candidates.push(("LO/MLE", (mu, beta, nll)));
                }
            }
        }

        let best_cand = candidates
            .into_iter()
            .min_by(|(_, (_, _, nll_a)), (_, (_, _, nll_b))| {
                nll_a
                    .partial_cmp(nll_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let (mu_final, beta_final) = match best_cand {
            Some((name, (mu, beta, nll))) => {
                log::info!(
                    "LO DEBUG charge {charge}: best TNM candidate = {name} (mu={:.4}, beta={:.4}, lower_order_nll={:.4}) | ranks: MM={}, MLE={}",
                    mu,
                    beta,
                    nll,
                    lom_mm.len(),
                    lom_mle.len()
                );
                (mu, beta)
            }
            None => {
                log::warn!("LO DEBUG charge {charge}: all lower-order TNM candidates failed.");
                continue;
            }
        };

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

        // LR/Meanβ selection should produce finite candidates
        let loms = vec![
            (8u32, mu_k_mm, beta_k_mm),
            (9u32, mu_k_mm + 0.1, beta_k_mm * 1.02),
            (10u32, mu_k_mm + 0.2, beta_k_mm * 1.05),
        ];
        let top_scores = scores.clone(); // stand-in; just need non-empty

        let lr = eval_candidate_lr_range(&loms, &top_scores, 0.05, 0.40, None);
        assert!(lr.is_some());

        let mb = compute_mean_beta(&loms, 8, 3, &crate::input::LoMeanBetaMode::Consecutive);
        assert!(mb.is_some());
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
