//! Decoy-free Lower-Order (LO) model fitting utilities.

use fnv::FnvHashMap;
use statrs::consts::EULER_MASCHERONI;
use statrs::distribution::{ContinuousCDF, Gumbel};
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

/// k-order TEV (Lower-Order) likelihood primitives.
///
/// We work in standardized coordinates z = (x - μ) / β.
///
/// For the k-th order statistic in the Gumbel max-domain (Poisson-process / TEV
/// asymptotic form), define λ = exp(-z).
///
/// CDF  (paper Eq. 5 form):
///   F_k(z) = exp(-λ) * Σ_{j=0..k-1} λ^j / j!
///
/// PDF  (paper Eq. 6 form; derivative of the above):
///   f_k(z) = exp(-λ) * λ^k / (k-1)!
///         = exp(-exp(-z) - k*z) / (k-1)!
///
/// These are used for joint (μ,β) fitting across ranks and for diagnostics.
/// Numerical policy: return -INF on invalid inputs; clamp exponentials.
#[inline]
fn log_factorial_k_minus_1(k: u32) -> f64 {
    // (k-1)! = Γ(k)  => ln((k-1)!) = lnΓ(k)
    if k == 0 {
        return f64::NAN;
    }
    ln_gamma(k as f64)
}

#[inline]
fn log_factorial(n: u32) -> f64 {
    // n! = Γ(n+1) => ln(n!) = lnΓ(n+1)
    ln_gamma((n as f64) + 1.0)
}

#[inline]
fn log_sum_exp(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut m = f64::NEG_INFINITY;
    for &v in vals {
        if v.is_finite() && v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        return f64::NEG_INFINITY;
    }
    let mut s = 0.0f64;
    for &v in vals {
        if v.is_finite() {
            s += (v - m).exp();
        }
    }
    if s <= 0.0 || !s.is_finite() {
        f64::NEG_INFINITY
    } else {
        m + s.ln()
    }
}

#[inline]
pub(crate) fn tev_k_order_logpdf(k: u32, z: f64) -> f64 {
    if k == 0 || !z.is_finite() {
        return f64::NEG_INFINITY;
    }

    // exp(-z) can overflow if z << 0; clamp exponent argument.
    let t = (-z).clamp(-745.0, 745.0);
    let lambda = t.exp();
    if !lambda.is_finite() {
        return f64::NEG_INFINITY;
    }

    // log f_k(z) = -ln((k-1)!) - k*z - exp(-z)
    let lf = log_factorial_k_minus_1(k);
    if !lf.is_finite() {
        return f64::NEG_INFINITY;
    }

    let logpdf = -lf - (k as f64) * z - lambda;
    if logpdf.is_finite() {
        logpdf
    } else {
        f64::NEG_INFINITY
    }
}

#[inline]
pub(crate) fn tev_k_order_logcdf(k: u32, z: f64) -> f64 {
    if k == 0 || !z.is_finite() {
        return f64::NEG_INFINITY;
    }

    // λ = exp(-z)
    let t = (-z).clamp(-745.0, 745.0);
    let lambda = t.exp();
    if !lambda.is_finite() {
        return f64::NEG_INFINITY;
    }

    // log Σ_{j=0..k-1} exp( j*ln(λ) - ln(j!) )
    // But ln(λ) = -z, so term_j = -j*z - ln(j!)
    let mut terms: Vec<f64> = Vec::with_capacity(k as usize);
    for j in 0..k {
        let lf = log_factorial(j);
        if !lf.is_finite() {
            return f64::NEG_INFINITY;
        }
        let term = -(j as f64) * z - lf;
        terms.push(term);
    }
    let lse = log_sum_exp(&terms);
    if !lse.is_finite() {
        return f64::NEG_INFINITY;
    }

    // log F_k(z) = -λ + logsumexp(terms)
    let logcdf = -lambda + lse;
    if logcdf.is_finite() {
        logcdf
    } else {
        f64::NEG_INFINITY
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RankBucket {
    pub k: u32,
    pub scores: Vec<f64>,
    pub weight: f64,
}

#[inline]
pub(crate) fn nll_joint(mu: f64, beta: f64, buckets: &[RankBucket]) -> f64 {
    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return f64::INFINITY;
    }
    if buckets.is_empty() {
        return f64::INFINITY;
    }

    let mut nll = 0.0f64;

    for b in buckets {
        if b.k == 0 || b.scores.is_empty() {
            continue;
        }
        if !b.weight.is_finite() || b.weight <= 0.0 {
            // Non-positive/invalid weights are treated as "ignore this bucket".
            continue;
        }

        // weight_k * Σ_i [-log f_k(z_i)]
        let mut bucket_nll = 0.0f64;
        for &x in &b.scores {
            if !x.is_finite() {
                continue;
            }
            let z = (x - mu) / beta;
            let lp = tev_k_order_logpdf(b.k, z);
            if !lp.is_finite() {
                return f64::INFINITY; // fail-closed for invalid regions
            }
            bucket_nll += -lp;
        }

        // If everything was non-finite, bucket_nll stays 0; that's ok.
        nll += b.weight * bucket_nll;
        if !nll.is_finite() {
            return f64::INFINITY;
        }
    }

    nll
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct JointFitDiag {
    pub n_buckets: usize,
    pub n_points: usize,
    pub best_nll: f64,
}

/// Stable, dependency-free optimizer for joint (μ,β) TEV-k fitting.
///
/// Strategy:
/// - coarse scan over β in log-space around beta_init
/// - for each β, grid-search μ (bounded)
/// - local refinement around best (μ,β)
///
/// Returns (μ_hat, β_hat, se_beta, diag)
pub(crate) fn fit_joint_tev(
    mu_init: f64,
    beta_init: f64,
    buckets: &[RankBucket],
) -> Option<(f64, f64, f64, JointFitDiag)> {
    if buckets.is_empty() {
        return None;
    }

    // Hard bounds aligned with your TNM scan domain.
    const MU_MIN: f64 = 0.05;
    const MU_MAX: f64 = 0.40;

    // Count points (for diagnostics only).
    let mut n_points = 0usize;
    for b in buckets {
        n_points += b.scores.len();
    }

    let beta0 = if beta_init.is_finite() && beta_init > 0.0 {
        beta_init
    } else {
        1.0
    };
    let _mu0 = if mu_init.is_finite() {
        mu_init.clamp(MU_MIN, MU_MAX)
    } else {
        0.2
    };

    // ----- coarse beta scan (2^[-6..+6]) -----
    let mut best: Option<(f64, f64, f64)> = None; // (mu, beta, nll)
    for p in -6..=6 {
        let beta = beta0 * 2.0f64.powi(p);
        if !beta.is_finite() || beta <= 0.0 {
            continue;
        }

        // μ grid across full bounds (coarse)
        const MU_N: usize = 96;
        let mu_step = (MU_MAX - MU_MIN) / ((MU_N - 1) as f64);

        for i in 0..MU_N {
            let mu = MU_MIN + (i as f64) * mu_step;
            let nll = nll_joint(mu, beta, buckets);
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

    // ----- refinement around best -----
    // Refine beta in log2 steps of 0.25 within ±2.0 (i.e., 2^(±2) = 4x)
    for q in -8..=8 {
        let beta = beta_best * 2.0f64.powf((q as f64) / 4.0);
        if !beta.is_finite() || beta <= 0.0 {
            continue;
        }

        // Refine μ in a local window around current best.
        const MU_LOCAL_HALF_WIDTH: f64 = 0.05;
        let lo = (mu_best - MU_LOCAL_HALF_WIDTH).clamp(MU_MIN, MU_MAX);
        let hi = (mu_best + MU_LOCAL_HALF_WIDTH).clamp(MU_MIN, MU_MAX);

        const MU_N_LOCAL: usize = 81;
        let mu_step = if MU_N_LOCAL > 1 {
            (hi - lo) / ((MU_N_LOCAL - 1) as f64)
        } else {
            0.0
        };

        for i in 0..MU_N_LOCAL {
            let mu = lo + (i as f64) * mu_step;
            let nll = nll_joint(mu, beta, buckets);
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

    // ----- crude se_beta from curvature (finite-difference in β) -----
    // se_beta ≈ sqrt(1 / d2/dβ2 NLL) at optimum, if curvature > 0
    let eps = (beta_best.abs() * 0.01).max(1e-6);
    let nll_m = nll_joint(mu_best, beta_best - eps, buckets);
    let nll_0 = nll_joint(mu_best, beta_best, buckets);
    let nll_p = nll_joint(mu_best, beta_best + eps, buckets);

    let se_beta = if nll_m.is_finite() && nll_0.is_finite() && nll_p.is_finite() {
        let d2 = (nll_p - 2.0 * nll_0 + nll_m) / (eps * eps);
        if d2.is_finite() && d2 > 0.0 {
            (1.0 / d2).sqrt()
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    };

    let diag = JointFitDiag {
        n_buckets: buckets.len(),
        n_points,
        best_nll: nll_best,
    };

    Some((mu_best, beta_best, se_beta, diag))
}

#[inline]
fn approx_pp_distance_tev_k(k: u32, scores: &[f64], mu: f64, beta: f64) -> f64 {
    if k == 0 || scores.is_empty() || !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return 1.0;
    }

    // Empirical CDF vs model CDF at sorted sample points.
    // Use max |F_model(x_i) - i/n| as a KS-like P–P distance.
    let mut xs: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
    if xs.is_empty() {
        return 1.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = xs.len() as f64;
    let mut d = 0.0f64;

    for (i, &x) in xs.iter().enumerate() {
        let z = (x - mu) / beta;
        let logcdf = tev_k_order_logcdf(k, z);
        if !logcdf.is_finite() {
            return 1.0;
        }
        let f_model = logcdf.exp().clamp(0.0, 1.0);
        let f_emp = ((i + 1) as f64) / n;
        let diff = (f_model - f_emp).abs();
        if diff > d {
            d = diff;
        }
    }
    d.clamp(0.0, 1.0)
}

#[inline]
fn median_finite(mut v: Vec<f64>) -> Option<f64> {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        Some(v[mid])
    } else {
        Some(0.5 * (v[mid - 1] + v[mid]))
    }
}

#[inline]
fn approx_squeeze_score(k: u32, buckets: &[RankBucket], mu: f64, beta: f64) -> Option<f64> {
    // Monotone squeeze indicator:
    // compare observed median gap between rank k and k+1 null scores
    // to expected gap under the fitted TEV-k / TEV-(k+1) medians.
    //
    // squeeze_score in (0, +inf); smaller => more compression (more squeeze).
    let b_k = buckets.iter().find(|b| b.k == k)?;
    let b_k1 = buckets.iter().find(|b| b.k == k + 1)?;

    let med_k = median_finite(b_k.scores.clone())?;
    let med_k1 = median_finite(b_k1.scores.clone())?;
    let gap_obs = (med_k - med_k1).abs();

    // Expected medians (approx): use model quantiles at p=0.5 via inverse by bisection.
    // We only need a monotone indicator, so coarse bisection is fine.
    let q = 0.5;
    let z_k = approx_quantile_z(k, q)?;
    let z_k1 = approx_quantile_z(k + 1, q)?;
    let gap_exp = ((mu + beta * z_k) - (mu + beta * z_k1)).abs();

    if gap_exp <= 0.0 || !gap_exp.is_finite() {
        return None;
    }
    Some((gap_obs / gap_exp).clamp(0.0, 10.0))
}

#[inline]
fn approx_quantile_z(k: u32, p: f64) -> Option<f64> {
    if k == 0 || !(0.0..=1.0).contains(&p) {
        return None;
    }
    // Solve F_k(z) = p by bisection on z in a safe range.
    // For order statistics in Gumbel domain, z typically lives in [-20, +20].
    let mut lo = -20.0f64;
    let mut hi = 20.0f64;

    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let logcdf = tev_k_order_logcdf(k, mid);
        if !logcdf.is_finite() {
            return None;
        }
        let f = logcdf.exp();
        if f < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

#[inline]
fn weight_from_gof_and_squeeze(k: u32, gof: f64, squeeze: Option<f64>) -> f64 {
    // GOF: d in [0,1], smaller is better.
    // Map to weight via an exponential falloff: w_gof = exp(-a * d)
    // Choose a moderate a so d=0.2 reduces weight meaningfully, but not to zero.
    let a = 6.0;
    let w_gof = (-a * gof.clamp(0.0, 1.0)).exp();

    // Squeeze: ratio gap_obs/gap_exp. <1 => compression.
    // Penalize early ranks if compressed. If squeeze is None, don't penalize.
    let w_sq = match squeeze {
        None => 1.0,
        Some(r) => {
            // If r >= 1, no squeeze penalty. If r < 1, downweight.
            // Use r^b with b>1 to penalize strong compression.
            let b = 2.5;
            if r >= 1.0 {
                1.0
            } else {
                r.clamp(0.0, 1.0).powf(b)
            }
        }
    };

    // Mild extra penalty for smallest admissible k (still within window)
    // to reduce correlation squeeze influence.
    let w_k = if k <= 6 { 0.85 } else { 1.0 };

    (w_gof * w_sq * w_k).clamp(0.05, 1.0)
}

#[inline]
fn squeeze_factor_from_diags(diags: &[(u32, f64, Option<f64>)]) -> f64 {
    // diags entries are (k, gof, squeeze_ratio)
    // squeeze_ratio ~ gap_obs / gap_exp; < 1 => compression.
    //
    // We aggregate using a robust statistic:
    // - take the median of available squeeze ratios
    // - clamp to [0.6..1.0] (never inflate beta by > ~1/0.6 ≈ 1.67x here)
    //
    // If no squeeze ratios exist, return 1.0 (no correction).
    let mut rs: Vec<f64> = diags
        .iter()
        .filter_map(|(_k, _gof, sq)| *sq)
        .filter(|r| r.is_finite() && *r > 0.0)
        .collect();

    if rs.is_empty() {
        return 1.0;
    }

    rs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = rs.len() / 2;
    let med = if rs.len() % 2 == 1 {
        rs[mid]
    } else {
        0.5 * (rs[mid - 1] + rs[mid])
    };

    med.clamp(0.6, 1.0)
}

/// Fits a charge-stratified Lower Order Model.
///
/// LO fitting contract (Decoy-Free)
/// --------------------------------
/// This module fits a charge-stratified Target Null Model (TNM) and uses it to
/// produce rank-1 p-values (downstream interface remains unchanged: TNM is a
/// Gumbel(μ, β) used only for rank-1 scoring).
///
/// Refactor target (for correctness):
/// - Fit TNM parameters (μ, β) *per charge* by jointly fitting k-order TEV
///   distributions for ranks k within the configured LO null-rank window.
/// - Use moderate k with adaptive weighting to downweight ranks exhibiting
///   correlation squeeze and/or poor goodness-of-fit.
/// - Replace any fixed "blend with moments" behavior with Empirical-Bayes
///   shrinkage on β using uncertainty from the joint fit.
/// - Optionally apply an N_eff / correlation-squeeze correction that inflates β
///   only when diagnostics indicate rank compression.
/// - Preserve the external behavior: output TNM as Gumbel(μ, β) for rank-1
///   p-values; no changes to downstream consumers.
///
pub fn fit_decoy_free_model(
    rank_null_stream: &[(u32, f64, u8)],
    rank1_stream: &[(f64, u8)],
    min_null_rank: u32,
    max_null_rank: u32,
    min_null_size_per_charge: usize,
    min_rank_count: usize,
) -> LowerOrderModel {
    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
			"LO DEBUG fit: null-rank window [{min_null_rank}..={max_null_rank}]. rank_null_stream.len()={} rank1_stream.len()={} min_null_size_per_charge={} min_rank_count={}",
			rank_null_stream.len(),
			rank1_stream.len(),
			min_null_size_per_charge,
			min_rank_count
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

        // Build rank buckets for k in [min_null_rank..=max_null_rank]
        // using the per-rank null scores for this charge.
        let mut buckets: Vec<RankBucket> = Vec::new();
        let mut pooled_charge_null: Vec<f64> = Vec::new();

        for k in min_null_rank..=max_null_rank {
            let scores_k = match by_rank.get(&k) {
                Some(v) if v.len() >= min_rank_count => v,
                _ => continue,
            };

            // Collect finite values only (fail-closed on NaNs/Infs)
            let scores: Vec<f64> = scores_k.iter().copied().filter(|x| x.is_finite()).collect();
            if scores.len() < min_rank_count {
                continue;
            }

            pooled_charge_null.extend_from_slice(&scores);

            buckets.push(RankBucket {
                k,
                scores,
                weight: 1.0,
            });
        }

        // Need at least something to proceed; otherwise skip this charge
        if buckets.is_empty() {
            continue;
        }

        // ---- Adaptive k admissibility (bounded) ----
        // Default policy:
        // - exclude k <= 3 (contamination / correlation squeeze zone)
        // - avoid very large k when support is sparse (conditioning bias / thin buckets)
        const K_EXCLUDE_LEQ_DEFAULT: u32 = 3;

        // Define "very large k" relative to the available bucket count.
        // If we only have a few usable buckets, we restrict to a modest upper tail.
        let bucket_count_pre = buckets.len();

        // Remove k <= 3 by default
        buckets.retain(|b| b.k > K_EXCLUDE_LEQ_DEFAULT);

        if buckets.is_empty() {
            continue;
        }

        // If buckets are sparse, cap k to avoid large-k conditioning artifacts.
        // Heuristic: if we have < 6 buckets, cap at min(K_max, 12) (unless the user's max is smaller).
        // If we have < 4 buckets, cap at min(K_max, 10).
        let k_cap = if bucket_count_pre < 4 {
            10u32
        } else if bucket_count_pre < 6 {
            12u32
        } else {
            // If we have decent support, do not impose an extra cap here.
            max_null_rank
        };

        if k_cap < max_null_rank {
            buckets.retain(|b| b.k <= k_cap);
        }

        if buckets.len() < 2 {
            // Not enough ranks to identify a joint (μ,β) fit robustly.
            continue;
        }

        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "LO DEBUG charge={}: buckets used (k:count) = [{}]",
                charge,
                buckets
                    .iter()
                    .map(|b| format!("{}:{}", b.k, b.scores.len()))
                    .collect::<Vec<_>>()
                    .join(",")
            );
        };

        // ---- Joint (μ,β) fit via TEV-k NLL (stable grid + refinement) ----
        let (_mu0, beta0) = fit_gumbel_moments(&pooled_charge_null);
        let beta0 = if beta0.is_finite() && beta0 > 0.0 {
            beta0
        } else {
            1.0
        };

        // μ init: prefer rank-1 TS μ-grid at beta0, else midpoint.
        let mu0 = mu_grid_best_bic(beta0, ts)
            .map(|(m, _bic)| m)
            .unwrap_or(0.2);

        // Pass 1: uniform weights
        for b in buckets.iter_mut() {
            b.weight = 1.0;
        }

        let (mu1, beta1, _se1, _diag1) = match fit_joint_tev(mu0, beta0, &buckets) {
            Some(v) => v,
            None => continue,
        };

        if !beta1.is_finite() || beta1 <= 0.0 {
            continue;
        }

        // Compute diagnostics with an immutable borrow first (avoids E0502).
        let diags: Vec<(u32, f64, Option<f64>)> = buckets
            .iter()
            .map(|b| {
                let gof = approx_pp_distance_tev_k(b.k, &b.scores, mu1, beta1);
                let squeeze = approx_squeeze_score(b.k, &buckets, mu1, beta1);
                (b.k, gof, squeeze)
            })
            .collect();

        // Now assign weights with a mutable borrow.
        for b in buckets.iter_mut() {
            if let Some((_, gof, squeeze)) = diags.iter().find(|(k, _, _)| *k == b.k) {
                b.weight = weight_from_gof_and_squeeze(b.k, *gof, *squeeze);
            } else {
                b.weight = 1.0;
            }
        }

        // Pass 2: refit with diagnostic weights
        let (_mu2, beta2, se2, diag2) = match fit_joint_tev(mu1, beta1, &buckets) {
            Some(v) => v,
            None => continue,
        };

        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "LO DEBUG charge={}: joint-fit diag: n_buckets={} n_points={} best_nll={:.3}",
                charge,
                diag2.n_buckets,
                diag2.n_points,
                diag2.best_nll
            );
        }

        if !beta2.is_finite() || beta2 <= 0.0 {
            continue;
        }

        // Data-driven squeeze correction (inflate β only when ranks are compressed)
        let squeeze_factor = squeeze_factor_from_diags(&diags); // in (0,1], clamped
        let beta_corr = beta2 / squeeze_factor;

        if !beta_corr.is_finite() || beta_corr <= 0.0 {
            continue;
        }

        let beta_hat = beta_corr;

        let max_pp = diags
            .iter()
            .map(|(_k, gof, _sq)| *gof)
            .fold(0.0_f64, |a, b| a.max(b));

        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
				"LO DEBUG charge={}: beta pass1={:.6} pass2={:.6} max_pp={:.4} squeeze_factor={:.4} beta_corr={:.6}",
				charge,
				beta1,
				beta2,
				max_pp,
				squeeze_factor,
				beta_corr
			);
        }

        // Use existing 1-DOF BIC μ-grid using beta = beta_hat
        // (this preserves your downstream TNM-selection semantics on rank-1 TS).
        let beta_prior = beta0; // per-charge pooled-null prior (moments)
        let se_beta = se2;

        let best_charge: Option<(f64, f64, f64, f64, f64)> = mu_grid_best_bic(beta_hat, ts)
            .map(|(mu_best, bic)| (mu_best, beta_hat, se_beta, beta_prior, bic));
        if best_charge.is_none() {
            continue;
        }

        let (mu, mut beta, se_beta, beta_prior, _bic) = match best_charge {
            Some(t) => t,
            None => continue,
        };
        {
            // ---------------------------------------------------------
            // Calibration knobs (explicit, per-charge)
            // Applied AFTER TNM selection (min-BIC) and BEFORE storage.
            // ---------------------------------------------------------

            // ---- Empirical-Bayes β shrinkage (variance-weighted) ----
            // beta is currently β_corr (already squeeze-corrected).
            // beta_prior is the pooled-null moments β for this charge.
            //
            // w = se^2 / (se^2 + tau^2)
            // β_final = (1-w)*β_corr + w*β_prior
            //
            // tau: shrink strength (exposed in JSON). For now, use a conservative default.
            const LO_EB_TAU_DEFAULT: f64 = 0.03;

            let tau = LO_EB_TAU_DEFAULT;

            let se2 = if se_beta.is_finite() && se_beta > 0.0 {
                se_beta
            } else {
                // If curvature is unusable, shrink strongly toward prior (fail-closed)
                f64::INFINITY
            };

            let w = if se2.is_infinite() {
                1.0
            } else {
                let se2_sq = se2 * se2;
                let tau_sq = tau * tau;
                (se2_sq / (se2_sq + tau_sq)).clamp(0.0, 1.0)
            };

            let beta_corr = beta;
            let mut beta_final = (1.0 - w) * beta_corr + w * beta_prior;

            if !beta_final.is_finite() || beta_final <= 0.0 {
                beta_final = beta_corr; // last-resort fallback
            }

            beta = beta_final;

            if log::log_enabled!(log::Level::Debug) {
                log::debug!(
                    "LO DEBUG charge={}: EB shrink w={:.4} beta_prior={:.6} beta_final={:.6}",
                    charge,
                    w,
                    beta_prior,
                    beta
                );
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
