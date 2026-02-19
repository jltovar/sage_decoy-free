//! Decoy-free Lower-Order (LO) model fitting utilities.

use fnv::FnvHashMap;
use statrs::consts::EULER_MASCHERONI;
use statrs::distribution::{ContinuousCDF, Gumbel};

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

/// BIC helper for TNM selection with fixed beta.
///
/// Formula: BIC = p * ln(N) - 2 * ln(L)
/// Here, p = 1 because beta is fixed and we optimize mu only.
///
/// Uses a stable log-likelihood implementation; returns +INF on invalid inputs.
#[inline]
pub(crate) fn calculate_bic(mu: f64, beta: f64, top_scores: &[f64]) -> f64 {
    const P: f64 = 1.0;

    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return f64::INFINITY;
    }
    let n = top_scores.len();
    if n == 0 {
        return f64::INFINITY;
    }

    let ln_beta = beta.ln();
    if !ln_beta.is_finite() {
        return f64::INFINITY;
    }

    // logpdf = -ln(beta) - z - exp(-z), z=(x-mu)/beta
    let mut log_l = 0.0f64;
    for &x in top_scores {
        if !x.is_finite() {
            return f64::INFINITY;
        }
        let z = (x - mu) / beta;
        let ez = (-z).exp();
        if !ez.is_finite() {
            return f64::INFINITY;
        }
        let logpdf = -ln_beta - z - ez;
        if !logpdf.is_finite() {
            return f64::INFINITY;
        }
        log_l += logpdf;
    }

    let n_f64 = n as f64;
    if !n_f64.is_finite() || n_f64 <= 0.0 {
        return f64::INFINITY;
    }

    (P * n_f64.ln()) - (2.0 * log_l)
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
//       p = Gumbel(0,1).sf(tev_norm)
// - Falls back to a global (mu, beta) when charge-specific params are absent.
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

        let tev_norm = (score - mu) / beta;

        let std = match Gumbel::new(0.0, 1.0) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };

        std.sf(tev_norm).clamp(0.0, 1.0).max(1e-300)
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
//   per charge, chosen by min-BIC across 4 candidates:
//     (LR vs mean-β) × (MLE vs moments)
//
// Notes:
// - This function is intentionally self-contained (no FdrSettings dependency).
// - All gating is per-charge using:
//     min_null_size_per_charge  (total null size across ranks
//                               [lower_order_min_null_rank..=lower_order_max_null_rank]
//                               (clamped within global window))
//     min_rank_count            (minimum per-rank count to fit a LOM at rank k)
// - μ scan range is fixed to [0.05, 0.4].
//

#[derive(Clone, Copy, Debug)]
struct GumbelParams {
    mu: f64,
    beta: f64,
}

#[inline]
fn ols_beta_on_mu(pairs: &[GumbelParams]) -> Option<(f64, f64)> {
    // Fit beta = a * mu + b by ordinary least squares.
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;

    let sum_x = pairs.iter().map(|p| p.mu).sum::<f64>();
    let sum_y = pairs.iter().map(|p| p.beta).sum::<f64>();
    let sum_xx = pairs.iter().map(|p| p.mu * p.mu).sum::<f64>();
    let sum_xy = pairs.iter().map(|p| p.mu * p.beta).sum::<f64>();

    let denom = n * sum_xx - sum_x * sum_x;
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return None;
    }

    let a = (n * sum_xy - sum_x * sum_y) / denom;
    let b = (sum_y - a * sum_x) / n;

    if a.is_finite() && b.is_finite() {
        Some((a, b))
    } else {
        None
    }
}

#[inline]
fn mu_grid_best_bic(beta: f64, top_scores: &[f64]) -> Option<(f64, f64)> {
    // Scan mu in [0.05, 0.4] and pick mu minimizing BIC (fixed beta).
    if top_scores.is_empty() {
        return None;
    }
    if !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    const MU_MIN: f64 = 0.05;
    const MU_MAX: f64 = 0.40;
    const MU_N: usize = 256;

    let step = (MU_MAX - MU_MIN) / ((MU_N - 1) as f64);
    if !step.is_finite() || step <= 0.0 {
        return None;
    }

    let mut best_mu = MU_MIN;
    let mut best_bic = f64::INFINITY;

    for i in 0..MU_N {
        let mu = MU_MIN + (i as f64) * step;
        let bic = calculate_bic(mu, beta, top_scores);
        if bic < best_bic {
            best_bic = bic;
            best_mu = mu;
        }
    }

    best_bic.is_finite().then_some((best_mu, best_bic))
}

#[inline]
fn mean_beta_highest_three(available: &[(u32, GumbelParams)]) -> Option<f64> {
    // Mean of the three highest available ranks.
    // Input is (rank_k, params) for ranks that were fit.
    let mut v: Vec<(u32, f64)> = available.iter().map(|(k, p)| (*k, p.beta)).collect();
    v.retain(|(_, b)| b.is_finite() && *b > 0.0);
    if v.is_empty() {
        return None;
    }
    // sort by k descending and take up to 3
    v.sort_by(|a, b| b.0.cmp(&a.0));
    let take = v.len().min(3);
    let mean = v.iter().take(take).map(|(_, b)| *b).sum::<f64>() / (take as f64);
    (mean.is_finite() && mean > 0.0).then_some(mean)
}

/// Fits a charge-stratified Lower Order Model.
///
/// Calibration knobs
/// -----------------
/// These are optional, explicit post-selection transforms applied PER CHARGE
/// AFTER the min-BIC TNM candidate is selected:
/// - lo_beta_blend_moments:  blend selected beta toward global moments beta
/// - lo_beta_safety_mult:    multiply beta by a safety factor
pub fn fit_decoy_free_model(
    rank_null_stream: &[(u32, f64, u8)],
    rank1_stream: &[(f64, u8)],
    min_null_rank: u32,
    max_null_rank: u32,
    min_null_size_per_charge: usize,
    min_rank_count: usize,
    lo_beta_blend_moments: f64,
    lo_beta_safety_mult: f64,
) -> LowerOrderModel {
    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "LO DEBUG fit: null-rank window [{min_null_rank}..={max_null_rank}]. rank_null_stream.len()={} rank1_stream.len()={} min_null_size_per_charge={} min_rank_count={} lo_beta_blend_moments={} lo_beta_safety_mult={}",
            rank_null_stream.len(),
            rank1_stream.len(),
            min_null_size_per_charge,
            min_rank_count,
            lo_beta_blend_moments,
            lo_beta_safety_mult
        );
    }

    // -------------------------
    // (A) Build per-charge datasets
    // -------------------------
    // null_by_rank[z][k] -> Vec<f64>
    let mut null_by_rank: FnvHashMap<u8, FnvHashMap<u32, Vec<f64>>> = FnvHashMap::default();
    let mut null_total_by_charge: FnvHashMap<u8, usize> = FnvHashMap::default();
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

        *null_total_by_charge.entry(charge).or_insert(0) += 1;
        pooled_null_scores.push(score);
    }

    if log::log_enabled!(log::Level::Debug) {
        let mut v: Vec<(u8, usize)> = null_total_by_charge.iter().map(|(z, n)| (*z, *n)).collect();
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
        let n_null = *null_total_by_charge.get(&charge).unwrap_or(&0);
        if n_null < min_null_size_per_charge {
            continue;
        }
        let ts = match top_scores.get(&charge) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };

        // Fit LOMs for k in [min_null_rank..=max_null_rank] (two estimator families)
        // Here, per plan: LOM(k) is fit on the null scores for rank==k.
        let mut lom_mle: Vec<(u32, GumbelParams)> = Vec::new();
        let mut lom_mom: Vec<(u32, GumbelParams)> = Vec::new();

        for k in min_null_rank..=max_null_rank {
            let scores_k = match by_rank.get(&k) {
                Some(v) if v.len() >= min_rank_count => v,
                _ => continue,
            };

            // Moments LOM
            let (mu_m, beta_m) = fit_gumbel_moments(scores_k);
            if mu_m.is_finite() && beta_m.is_finite() && beta_m > 0.0 {
                lom_mom.push((
                    k,
                    GumbelParams {
                        mu: mu_m,
                        beta: beta_m,
                    },
                ));
            }

            // MLE LOM
            if let Some((mu_e, beta_e)) = fit_gumbel_mle(scores_k) {
                if mu_e.is_finite() && beta_e.is_finite() && beta_e > 0.0 {
                    lom_mle.push((
                        k,
                        GumbelParams {
                            mu: mu_e,
                            beta: beta_e,
                        },
                    ));
                }
            }
        }

        // Need at least something to proceed; otherwise skip this charge
        if lom_mle.is_empty() && lom_mom.is_empty() {
            continue;
        }

        // Build 4 TNM candidates:
        // (1) LR-based (MLE)
        // (2) Mean-β (MLE)
        // (3) LR-based (Moments)
        // (4) Mean-β (Moments)
        let mut best_charge: Option<(f64, f64, f64)> = None; // (mu, beta, bic)

        // ---- Candidate 1: LR-based (MLE) ----
        {
            let pairs: Vec<GumbelParams> = lom_mle.iter().map(|(_, p)| *p).collect();
            if let Some((a, b)) = ols_beta_on_mu(&pairs) {
                let mut best_local: Option<(f64, f64, f64)> = None; // (mu, beta, bic)
                                                                    // scan mu in [0.05, 0.4]
                const MU_MIN: f64 = 0.05;
                const MU_MAX: f64 = 0.40;
                const MU_N: usize = 256;
                let step = (MU_MAX - MU_MIN) / ((MU_N - 1) as f64);

                if step.is_finite() && step > 0.0 {
                    for i in 0..MU_N {
                        let mu = MU_MIN + (i as f64) * step;
                        let beta = a * mu + b;
                        if !beta.is_finite() || beta <= 0.0 {
                            continue;
                        }
                        let bic = calculate_bic(mu, beta, ts);
                        if bic.is_finite() {
                            match best_local {
                                None => best_local = Some((mu, beta, bic)),
                                Some((_, _, best_bic)) if bic < best_bic => {
                                    best_local = Some((mu, beta, bic))
                                }
                                _ => {}
                            }
                        }
                    }
                }

                if let Some((mu, beta, bic)) = best_local {
                    match best_charge {
                        None => best_charge = Some((mu, beta, bic)),
                        Some((_, _, best_bic)) if bic < best_bic => {
                            best_charge = Some((mu, beta, bic))
                        }
                        _ => {}
                    }
                }
            }
        }

        // ---- Candidate 2: Mean-β (MLE) ----
        {
            if let Some(beta_mean) = mean_beta_highest_three(&lom_mle) {
                if let Some((mu, bic)) = mu_grid_best_bic(beta_mean, ts) {
                    match best_charge {
                        None => best_charge = Some((mu, beta_mean, bic)),
                        Some((_, _, best_bic)) if bic < best_bic => {
                            best_charge = Some((mu, beta_mean, bic))
                        }
                        _ => {}
                    }
                }
            }
        }

        // ---- Candidate 3: LR-based (Moments) ----
        {
            let pairs: Vec<GumbelParams> = lom_mom.iter().map(|(_, p)| *p).collect();
            if let Some((a, b)) = ols_beta_on_mu(&pairs) {
                let mut best_local: Option<(f64, f64, f64)> = None; // (mu, beta, bic)
                const MU_MIN: f64 = 0.05;
                const MU_MAX: f64 = 0.40;
                const MU_N: usize = 256;
                let step = (MU_MAX - MU_MIN) / ((MU_N - 1) as f64);

                if step.is_finite() && step > 0.0 {
                    for i in 0..MU_N {
                        let mu = MU_MIN + (i as f64) * step;
                        let beta = a * mu + b;
                        if !beta.is_finite() || beta <= 0.0 {
                            continue;
                        }
                        let bic = calculate_bic(mu, beta, ts);
                        if bic.is_finite() {
                            match best_local {
                                None => best_local = Some((mu, beta, bic)),
                                Some((_, _, best_bic)) if bic < best_bic => {
                                    best_local = Some((mu, beta, bic))
                                }
                                _ => {}
                            }
                        }
                    }
                }

                if let Some((mu, beta, bic)) = best_local {
                    match best_charge {
                        None => best_charge = Some((mu, beta, bic)),
                        Some((_, _, best_bic)) if bic < best_bic => {
                            best_charge = Some((mu, beta, bic))
                        }
                        _ => {}
                    }
                }
            }
        }

        // ---- Candidate 4: Mean-β (Moments) ----
        {
            if let Some(beta_mean) = mean_beta_highest_three(&lom_mom) {
                if let Some((mu, bic)) = mu_grid_best_bic(beta_mean, ts) {
                    match best_charge {
                        None => best_charge = Some((mu, beta_mean, bic)),
                        Some((_, _, best_bic)) if bic < best_bic => {
                            best_charge = Some((mu, beta_mean, bic))
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some((mu, mut beta, _bic)) = best_charge {
            // ---------------------------------------------------------
            // Calibration knobs (explicit, per-charge)
            // Applied AFTER TNM selection (min-BIC) and BEFORE storage.
            // ---------------------------------------------------------

            // (1) Blend selected beta toward GLOBAL moments beta
            // Note: lo_beta_blend_moments is a scalar in [0,1]; 0 => no blend.
            let w = lo_beta_blend_moments.clamp(0.0, 1.0);
            if w > 0.0 {
                // pooled_null_scores is already built in (A); use it as the global null pool
                let (_mu_mom_g, beta_mom_g) = fit_gumbel_moments(&pooled_null_scores);
                if beta_mom_g.is_finite() && beta_mom_g > 0.0 && beta.is_finite() && beta > 0.0 {
                    beta = (1.0 - w) * beta + w * beta_mom_g;
                }
            }

            // Apply optional beta safety scaling (no-op when == 1.0)
            if (lo_beta_safety_mult - 1.0).abs() > 1e-12 {
                beta *= lo_beta_safety_mult;
            }

            // Final guard
            if mu.is_finite() && beta.is_finite() && beta > 0.0 {
                params_by_charge.insert(charge, (mu, beta));
            }
        }
    }

    // -------------------------
    // Fallback params (global)
    // -------------------------
    let fallback_params = {
        // Prefer a pooled moments fit on the pooled null scores.
        let (mu, beta) = fit_gumbel_moments(&pooled_null_scores);
        if mu.is_finite() && beta.is_finite() && beta > 0.0 {
            (mu, beta)
        } else {
            (f64::NAN, f64::NAN)
        }
    };

    // -------------------------
    // Charge filling rules (fit-time + query-time)
    // -------------------------

    // Rule: If charge 1 missing, copy charge 2 (if present)
    if !params_by_charge.contains_key(&1) {
        if let Some(p2) = params_by_charge.get(&2).copied() {
            params_by_charge.insert(1, p2);
        }
    }

    // Cache fitted charges + max fitted charge
    let mut fitted_charges_sorted: Vec<u8> = params_by_charge.keys().copied().collect();
    fitted_charges_sorted.sort_unstable();
    let max_fitted_charge: u8 = fitted_charges_sorted.last().copied().unwrap_or(0);

    LowerOrderModel {
        params_by_charge,
        fallback_params,

        // Default policy: MinimalDelta (no internal gap nearest-neighbor fill)
        charge_fill_mode: ChargeFillMode::MinimalDelta,
        fitted_charges_sorted,
        max_fitted_charge,
    }
}
