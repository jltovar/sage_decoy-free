//! Decoy-free MSFDR model fitting utilities.
//!
//! The methods in this module are based on the work of Yisu Peng, et al. published here:
//! 
//! New mixture models for decoy-free false discovery rate estimation in mass spectrometry proteomics
//! Yisu Peng, Shantanu Jain, Yong Fuga Li, Michal Greguš, Alexander R. Ivanov, Olga Vitek, Predrag Radivojac, 
//! Bioinformatics, Volume 36, Issue Supplement_2, December 2020, Pages i745–i753, 
//! DOI: 10.1093/bioinformatics/btaa8074
//! https://academic.oup.com/bioinformatics/article/36/Supplement_2/i745/6055912
//!
//! and implemented on GitHub here:
//! https://github.com/shawn-peng/DecoyFree-MSFDR

use crate::ml::skew_normal::SkewNormal;
use serde::{Deserialize, Serialize};
use statrs::consts::EULER_MASCHERONI;
use statrs::distribution::{Continuous, ContinuousCDF, Gumbel};
use std::f64::consts::PI;

/// Small floor to prevent log(0) and divide-by-zero cascades.
const TINY: f64 = 1e-300;

/// Minimum Gumbel scale used for MSFDR1 null initialization to avoid var/scale collapse.
const MSFDR1_MIN_NULL_SCALE: f64 = 1e-6;

// --- Formatting helpers for parameter summaries ---
#[inline]
fn fmt_f64(x: f64) -> String {
    if x.is_finite() {
        // Stable, compact, and unambiguous across scales.
        format!("{:.6e}", x)
    } else if x.is_nan() {
        "NaN".to_string()
    } else if x.is_sign_negative() {
        "-Inf".to_string()
    } else {
        "Inf".to_string()
    }
}

/// Returns a compact parameter summary in the form
/// `pi=<...>, null=(<loc>,<scale>), target=(...)`.
pub trait MsfdrParamTuple {
    fn param_tuple(&self) -> String;
}

/// Stable log-sum-exp for two terms.
#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Clamp to [0,1] with a tiny floor on the open interval for downstream log safety.
#[inline]
fn clamp_p01(p: f64) -> f64 {
    if !p.is_finite() {
        return 1.0;
    }
    p.clamp(0.0, 1.0).max(TINY)
}

/// Weighted mean/var/skew (skew is standardized 3rd central moment).
fn weighted_moments(x: &[f64], w: &[f64]) -> Option<(f64, f64, f64)> {
    debug_assert_eq!(x.len(), w.len());
    if x.len() < 5 {
        return None;
    }

    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    for (&xi, &wi) in x.iter().zip(w.iter()) {
        if !xi.is_finite() || !wi.is_finite() || wi <= 0.0 {
            continue;
        }
        sum_w += wi;
        sum_wx += wi * xi;
    }
    if sum_w <= 0.0 {
        return None;
    }
    let mean = sum_wx / sum_w;

    let mut sum_wv = 0.0;
    let mut sum_wm3 = 0.0;
    for (&xi, &wi) in x.iter().zip(w.iter()) {
        if !xi.is_finite() || !wi.is_finite() || wi <= 0.0 {
            continue;
        }
        let d = xi - mean;
        sum_wv += wi * d * d;
    }
    let var = (sum_wv / sum_w).max(0.0);
    let std = var.sqrt().max(1e-12);

    for (&xi, &wi) in x.iter().zip(w.iter()) {
        if !xi.is_finite() || !wi.is_finite() || wi <= 0.0 {
            continue;
        }
        let z = (xi - mean) / std;
        sum_wm3 += wi * z * z * z;
    }
    let skew = sum_wm3 / sum_w;

    Some((mean, var, skew))
}

/// Gumbel moments inversion:
/// mean = mu + gamma*beta, var = (pi^2 * beta^2)/6
fn gumbel_from_mean_var(mean: f64, var: f64) -> Option<(f64, f64)> {
    if !mean.is_finite() || !var.is_finite() || var <= 0.0 {
        return None;
    }
    let beta = ((6.0 * var).sqrt() / PI).max(1e-9);
    let mu = mean - EULER_MASCHERONI * beta;
    if mu.is_finite() && beta.is_finite() && beta > 0.0 {
        Some((mu, beta))
    } else {
        None
    }
}

/// Seeded two-component mixture model for rank-1 scores.
///
/// The null component is a Gumbel distribution with externally supplied
/// location and scale parameters. The target component is a skew-normal
/// distribution initialized from the upper tail of the rank-1 score
/// distribution. During expectation-maximization, the null component remains
/// fixed and only the mixture weight and target-component parameters are
/// updated.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MsfdrSeededModel {
    pub null_loc: f64,
    pub null_scale: f64,
    pub target_mean: f64,
    pub target_std: f64,
    pub target_alpha: f64,
    pub pi: f64,
}

impl MsfdrSeededModel {
    /// Fit on rank1 only with a fixed null seed (mu_in, beta_in).
    ///
    /// Notes:
    /// - Null params remain fixed in the EM loop (same as your current "seeded" path).
    /// - Target skew-normal is updated by weighted moments each iteration.
    pub fn fit_rank1_seeded(
        rank1_scores: &[f64],
        mu_in: f64,
        beta_in: f64,
        iters: usize,
        em_tol: f64,
        pi_clamp: (f64, f64),
        top_frac_init: f64,
    ) -> Option<Self> {
        let xs: Vec<f64> = rank1_scores
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if xs.len() < 10 {
            return None;
        }

        let null_loc = mu_in;
        let null_scale = beta_in.max(1e-6);

        let mut sorted = xs.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));

        // Target init from top slice.
        let top_frac = top_frac_init.clamp(0.05, 0.5);
        let top_n = ((sorted.len() as f64) * top_frac).round() as usize;
        let top_n = top_n.max(5).min(sorted.len());
        let top = &sorted[..top_n];

        let t_mean = top.iter().sum::<f64>() / (top.len() as f64);
        let t_var = top.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / (top.len() as f64);
        let t_std = t_var.sqrt().max(1e-6);

        // pi init from fraction above null mean proxy
        let null_mean_proxy = null_loc + EULER_MASCHERONI * null_scale;
        let frac_above = (sorted.iter().filter(|&&v| v > null_mean_proxy).count() as f64)
            / (sorted.len() as f64);
        let mut pi = frac_above.clamp(pi_clamp.0, pi_clamp.1);

        let mut target_mean = t_mean;
        let mut target_std = t_std;
        let mut target_alpha = 2.0f64; // stable default

        let null_dist = Gumbel::new(null_loc, null_scale).ok()?;

        let mut prev_ll = -f64::INFINITY;
        let iters = iters.max(5).min(200);

        for _ in 0..iters {
            let pi0 = pi.clamp(1e-6, 1.0 - 1e-6);
            let log_pi = pi0.ln();
            let log_1m_pi = (1.0 - pi0).ln();

            // E-step: responsibilities for target component
            let mut resp: Vec<f64> = Vec::with_capacity(sorted.len());
            let mut ll = 0.0;

            let sn = SkewNormal::new(target_mean, target_std.max(1e-9), target_alpha);

            for &x in &sorted {
                let f0 = null_dist.pdf(x).max(TINY);
                let f1 = sn.pdf(x).max(TINY);

                let log_f0 = f0.ln();
                let log_f1 = f1.ln();

                let log_num = log_pi + log_f1;
                let log_den = log_add_exp(log_1m_pi + log_f0, log_num);

                ll += log_den;
                let r = (log_num - log_den).exp();
                resp.push(if r.is_finite() { r } else { 0.0 });
            }

            let avg_ll = ll / (sorted.len() as f64);
            if prev_ll.is_finite() && (avg_ll - prev_ll).abs() < em_tol {
                break;
            }
            prev_ll = avg_ll;

            // M-step: update pi + target moments
            let sum_r = resp.iter().sum::<f64>();
            if sum_r < 1e-8 {
                break;
            }

            pi = (sum_r / (sorted.len() as f64)).clamp(pi_clamp.0, pi_clamp.1);

            // Weighted moments for target using r as weights
            if let Some((m, v, s)) = weighted_moments(&sorted, &resp) {
                // Fit skew-normal from moments; if fails, keep last parameters.
                if let Some(dist) = SkewNormal::from_moments(m, v, s) {
                    target_mean = dist.location;
                    target_std = dist.scale.max(1e-6);
                    target_alpha = dist.shape;
                }
            }
        }

        Some(Self {
            null_loc,
            null_scale,
            target_mean,
            target_std,
            target_alpha,
            pi,
        })
    }

    /// Model-derived PEP = P(null | x).
    pub fn pep(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }
        let null_dist = match Gumbel::new(self.null_loc, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };
        let sn = SkewNormal::new(
            self.target_mean,
            self.target_std.max(1e-9),
            self.target_alpha,
        );

        let f0 = null_dist.pdf(x).max(TINY);
        let f1 = sn.pdf(x).max(TINY);

        let pi = self.pi.clamp(1e-6, 1.0 - 1e-6);
        let num = (1.0 - pi) * f0;
        let den = num + pi * f1;
        if den > 0.0 && den.is_finite() {
            (num / den).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Null tail p-value under the fitted null (equivalent to your TEV-normalized sf path).
    pub fn p_value(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }
        let null_dist = match Gumbel::new(self.null_loc, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };
        clamp_p01(null_dist.sf(x))
    }
}

impl MsfdrParamTuple for MsfdrSeededModel {
    fn param_tuple(&self) -> String {
        format!(
            "pi={}, null=({},{}), target=({},{},{})",
            fmt_f64(self.pi),
            fmt_f64(self.null_loc),
            fmt_f64(self.null_scale),
            // Seeded target is stored as moments-like params
            fmt_f64(self.target_mean),
            fmt_f64(self.target_std),
            fmt_f64(self.target_alpha),
        )
    }
}

/// Unanchored two-component mixture model for rank-1 scores.
///
/// This variant fits a Gumbel null component and a skew-normal target
/// component directly from the rank-1 score distribution. Both components,
/// together with the target mixture proportion, are updated by
/// expectation-maximization. Initialization uses lower-score observations for
/// the null component and upper-score observations for the target component.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Msfdr1SmixModel {
    pub null_loc: f64,
    pub null_scale: f64,
    pub target: SkewNormal,
    pub pi: f64, // mixture weight of target
}

impl Msfdr1SmixModel {
    /// Fit on rank1 only.
    ///
    /// Initialization:
    /// - null init from bottom slice (bottom_frac_init)
    /// - target init from top slice (top_frac_init)
    /// - pi init from fraction above null mean proxy, clamped by pi_clamp
    pub fn fit_rank1(
        rank1_scores: &[f64],
        iters: usize,
        em_tol: f64,
        pi_clamp: (f64, f64),
        bottom_frac_init: f64,
        top_frac_init: f64,
        mu_drift_abs: f64,
        beta_drift_mult: (f64, f64),
    ) -> Option<Self> {
        let mut xs: Vec<f64> = rank1_scores
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if xs.len() < 20 {
            return None;
        }
        xs.sort_by(|a, b| a.total_cmp(b)); // ascending for slices

        let n = xs.len();

        // --- init null from bottom slice ---
        let bfrac = bottom_frac_init.clamp(0.5, 0.9);
        let b_n = ((n as f64) * bfrac).round() as usize;
        let b_n = b_n.max(10).min(n);
        let bottom = &xs[..b_n];

        let b_mean = bottom.iter().sum::<f64>() / (bottom.len() as f64);
        let b_var_raw =
            bottom.iter().map(|v| (v - b_mean).powi(2)).sum::<f64>() / (bottom.len() as f64);

        // Clamp variance so gumbel init cannot fail on var<=0 (tie-heavy / discretized bottoms).
        // var_min derived from: var = (pi^2 / 6) * scale^2  =>  var_min = (pi*scale_min)^2 / 6
        let var_min = (PI * MSFDR1_MIN_NULL_SCALE).powi(2) / 6.0;
        let b_var = if b_var_raw < var_min {
            if log::log_enabled!(log::Level::Debug) {
                log::debug!(
                    "MSFDR1 DEBUG init clamp: b_var_raw={:.6e} < var_min={:.6e}; clamping (scale_min={:.3e})",
                    b_var_raw,
                    var_min,
                    MSFDR1_MIN_NULL_SCALE
                );
            }
            var_min
        } else {
            b_var_raw
        };

        let (mut null_loc, mut null_scale) = gumbel_from_mean_var(b_mean, b_var)?;

        // Optional second clamp: ensure the initialized scale is not absurdly tiny.
        if null_scale < MSFDR1_MIN_NULL_SCALE {
            if log::log_enabled!(log::Level::Debug) {
                log::debug!(
                    "MSFDR1 DEBUG init clamp: null_scale_raw={:.6e} < scale_min={:.3e}; clamping",
                    null_scale,
                    MSFDR1_MIN_NULL_SCALE
                );
            }
            null_scale = MSFDR1_MIN_NULL_SCALE;
        }

        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "MSFDR1 DEBUG init: n={} bfrac={:.3} b_n={} b_mean={:.6} b_var_raw={:.6e} b_var_used={:.6e} null_loc={:.6} null_scale_used={:.6e} top_frac_init={:.3}",
                n,
                bfrac,
                b_n,
                b_mean,
                b_var_raw,
                b_var,
                null_loc,
                null_scale,
                top_frac_init
            );
        }

        // --- init target from top slice ---
        let mut desc = xs.clone();
        desc.sort_by(|a, b| b.total_cmp(a));
        let tfrac = top_frac_init.clamp(0.05, 0.5);
        let t_n = ((n as f64) * tfrac).round() as usize;
        let t_n = t_n.max(10).min(n);
        let top = &desc[..t_n];

        let t_mean = top.iter().sum::<f64>() / (top.len() as f64);
        let t_var = top.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / (top.len() as f64);
        let t_std = t_var.sqrt().max(1e-6);

        let mut target = SkewNormal::new(t_mean, t_std, 2.0);

        // --- init pi from fraction above null mean proxy ---
        let null_mean_proxy = null_loc + EULER_MASCHERONI * null_scale;
        let frac_above = (xs.iter().filter(|&&v| v > null_mean_proxy).count() as f64) / (n as f64);
        let mut pi = frac_above.clamp(pi_clamp.0, pi_clamp.1);

        // Capture initialization constraints to prevent runaway null drift.
        let init_null_loc = null_loc;
        let init_null_scale = null_scale;

        // EM
        let iters = iters.max(10).min(500);
        let mut prev_ll = -f64::INFINITY;

        for _ in 0..iters {
            let null_dist = match Gumbel::new(null_loc, null_scale.max(1e-9)) {
                Ok(d) => d,
                Err(_) => return None,
            };

            let pi0 = pi.clamp(1e-6, 1.0 - 1e-6);
            let log_pi = pi0.ln();
            let log_1m_pi = (1.0 - pi0).ln();

            // E-step: r_i = P(target | x_i)
            let mut r: Vec<f64> = Vec::with_capacity(n);
            let mut ll = 0.0;

            for &x in &xs {
                let f0 = null_dist.pdf(x).max(TINY);
                let f1 = target.pdf(x).max(TINY);

                let log_f0 = f0.ln();
                let log_f1 = f1.ln();

                let log_num = log_pi + log_f1;
                let log_den = log_add_exp(log_1m_pi + log_f0, log_num);

                ll += log_den;
                let ri = (log_num - log_den).exp();
                r.push(if ri.is_finite() { ri } else { 0.0 });
            }

            let avg_ll = ll / (n as f64);
            if prev_ll.is_finite() && (avg_ll - prev_ll).abs() < em_tol {
                break;
            }
            prev_ll = avg_ll;

            // M-step: update pi
            let sum_r = r.iter().sum::<f64>();
            if sum_r < 1e-8 || sum_r > (n as f64 - 1e-8) {
                break;
            }
            pi = (sum_r / (n as f64)).clamp(pi_clamp.0, pi_clamp.1);

            // Update target skew-normal by weighted moments with weights r
            if let Some((m, v, s)) = weighted_moments(&xs, &r) {
                if let Some(sn) = SkewNormal::from_moments(m, v, s) {
                    target = sn;
                }
            }

            // Update null gumbel by weighted moments with weights (1-r)
            let w0: Vec<f64> = r.iter().map(|&ri| (1.0 - ri).max(0.0)).collect();
            if let Some((m0, v0, _s0)) = weighted_moments(&xs, &w0) {
                if let Some((mu0, beta0)) = gumbel_from_mean_var(m0, v0) {
                    // Tighten constraints relative to initialization using passed params
                    let clamped_mu =
                        mu0.clamp(init_null_loc - mu_drift_abs, init_null_loc + mu_drift_abs);
                    let clamped_beta = beta0.clamp(
                        init_null_scale * beta_drift_mult.0,
                        init_null_scale * beta_drift_mult.1,
                    );

                    null_loc = clamped_mu;
                    null_scale = clamped_beta.max(1e-9);
                }
            }
        }

        Some(Self {
            null_loc,
            null_scale,
            target,
            pi,
        })
    }

    /// Fit on rank1 only, but initialize the null from an external seed
    /// (e.g. rank-null pool window), rather than bottom slice of rank1.
    pub fn fit_rank1_with_null_seed(
        rank1_scores: &[f64],
        iters: usize,
        em_tol: f64,
        pi_clamp: (f64, f64),
        null_loc_seed: f64,
        null_scale_seed: f64,
        top_frac_init: f64,
        mu_drift_abs: f64,
        beta_drift_mult: (f64, f64),
    ) -> Option<Self> {
        let mut xs: Vec<f64> = rank1_scores
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if xs.len() < 20 {
            return None;
        }
        xs.sort_by(|a, b| a.total_cmp(b)); // ascending for slices

        let n = xs.len();

        // --- init null from external seed ---
        let mut null_loc = null_loc_seed;
        let mut null_scale = null_scale_seed;

        if !null_loc.is_finite() || !null_scale.is_finite() {
            return None;
        }

        if null_scale < MSFDR1_MIN_NULL_SCALE {
            null_scale = MSFDR1_MIN_NULL_SCALE;
        }

        // --- init target from top slice ---
        let mut desc = xs.clone();
        desc.sort_by(|a, b| b.total_cmp(a));
        let tfrac = top_frac_init.clamp(0.05, 0.5);
        let t_n = ((n as f64) * tfrac).round() as usize;
        let t_n = t_n.max(10).min(n);
        let top = &desc[..t_n];

        let t_mean = top.iter().sum::<f64>() / (top.len() as f64);
        let t_var = top.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / (top.len() as f64);
        let t_std = t_var.sqrt().max(1e-6);

        let mut target = SkewNormal::new(t_mean, t_std, 2.0);

        // --- init pi from fraction above null mean proxy ---
        let null_mean_proxy = null_loc + EULER_MASCHERONI * null_scale;
        let frac_above = (xs.iter().filter(|&&v| v > null_mean_proxy).count() as f64) / (n as f64);
        let mut pi = frac_above.clamp(pi_clamp.0, pi_clamp.1);

        // EM (same as fit_rank1)
        let iters = iters.max(10).min(500);
        let mut prev_ll = -f64::INFINITY;

        for _ in 0..iters {
            let null_dist = match Gumbel::new(null_loc, null_scale.max(1e-9)) {
                Ok(d) => d,
                Err(_) => return None,
            };

            let pi0 = pi.clamp(1e-6, 1.0 - 1e-6);
            let log_pi = pi0.ln();
            let log_1m_pi = (1.0 - pi0).ln();

            let mut r: Vec<f64> = Vec::with_capacity(n);
            let mut ll = 0.0;

            for &x in &xs {
                let f0 = null_dist.pdf(x).max(TINY);
                let f1 = target.pdf(x).max(TINY);

                let log_f0 = f0.ln();
                let log_f1 = f1.ln();

                let log_num = log_pi + log_f1;
                let log_den = log_add_exp(log_1m_pi + log_f0, log_num);

                ll += log_den;
                let ri = (log_num - log_den).exp();
                r.push(if ri.is_finite() { ri } else { 0.0 });
            }

            let avg_ll = ll / (n as f64);
            if prev_ll.is_finite() && (avg_ll - prev_ll).abs() < em_tol {
                break;
            }
            prev_ll = avg_ll;

            let sum_r = r.iter().sum::<f64>();
            if sum_r < 1e-8 || sum_r > (n as f64 - 1e-8) {
                break;
            }
            pi = (sum_r / (n as f64)).clamp(pi_clamp.0, pi_clamp.1);

            if let Some((m, v, s)) = weighted_moments(&xs, &r) {
                if let Some(sn) = SkewNormal::from_moments(m, v, s) {
                    target = sn;
                }
            }

            // Update null gumbel by weighted moments with weights (1-r)
            let w0: Vec<f64> = r.iter().map(|&ri| (1.0 - ri).max(0.0)).collect();
            if let Some((m0, v0, _s0)) = weighted_moments(&xs, &w0) {
                if let Some((mu0, beta0)) = gumbel_from_mean_var(m0, v0) {
                    // Tighten constraints relative to seed using passed params
                    let clamped_mu =
                        mu0.clamp(null_loc_seed - mu_drift_abs, null_loc_seed + mu_drift_abs);
                    let clamped_beta = beta0.clamp(
                        null_scale_seed * beta_drift_mult.0,
                        null_scale_seed * beta_drift_mult.1,
                    );

                    null_loc = clamped_mu;
                    null_scale = clamped_beta.max(1e-9);
                }
            }
        }

        Some(Self {
            null_loc,
            null_scale,
            target,
            pi,
        })
    }

    /// PEP = P(null | x)
    pub fn pep(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }
        let null_dist = match Gumbel::new(self.null_loc, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };
        let f0 = null_dist.pdf(x).max(TINY);
        let f1 = self.target.pdf(x).max(TINY);

        let pi = self.pi.clamp(1e-6, 1.0 - 1e-6);
        let num = (1.0 - pi) * f0;
        let den = num + pi * f1;
        if den > 0.0 && den.is_finite() {
            (num / den).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Null tail p-value under the learned null.
    pub fn p_value(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }
        let null_dist = match Gumbel::new(self.null_loc, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };
        clamp_p01(null_dist.sf(x))
    }
}

impl MsfdrParamTuple for Msfdr1SmixModel {
    fn param_tuple(&self) -> String {
        format!(
            "pi={}, null=({},{}), target=({},{},{})",
            fmt_f64(self.pi),
            fmt_f64(self.null_loc),
            fmt_f64(self.null_scale),
            fmt_f64(self.target.location),
            fmt_f64(self.target.scale),
            fmt_f64(self.target.shape),
        )
    }
}

/// Anchored two-component mixture model for rank-1 scores.
///
/// This variant fits the target mixture on rank-1 observations while using an
/// external null-score pool to estimate the Gumbel null component. The pooled
/// null fit provides the initial null location and scale. When
/// `mix_anchor_incorrect` is `true`, the null component is held fixed during
/// expectation-maximization. Otherwise, the null component may adapt from the
/// pooled estimate, subject to explicit drift limits on location and scale.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Msfdr2SmixModel {
    pub null_seed_loc: f64,
    pub null_seed_scale: f64,

    pub null_loc: f64,
    pub null_scale: f64,

    pub target: SkewNormal,
    pub pi: f64,
    pub mix_anchor_incorrect: bool,

    // drift clamps (only used when mix_anchor_incorrect == false)
    pub beta_drift_mult: (f64, f64),
    pub mu_drift_abs: f64,
}

impl Msfdr2SmixModel {
    /// Fit with a pure-null pool.
    ///
    /// pool_scores are assumed to be null-only and are used to seed the null Gumbel.
    pub fn fit_rank1_with_pool(
        rank1_scores: &[f64],
        pool_scores: &[f64],
        iters: usize,
        em_tol: f64,
        pi_clamp: (f64, f64),
        top_frac_init: f64,
        mix_anchor_incorrect: bool,
        beta_drift_mult: (f64, f64),
        mu_drift_abs: f64,
    ) -> Option<Self> {
        let mut xs: Vec<f64> = rank1_scores
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if xs.len() < 20 {
            return None;
        }
        xs.sort_by(|a, b| a.total_cmp(b));
        let n = xs.len();

        let pool: Vec<f64> = pool_scores
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if pool.len() < 20 {
            return None;
        }

        // Seed null from pool moments (consistent, robust).
        let pool_mean = pool.iter().sum::<f64>() / (pool.len() as f64);
        let pool_var =
            pool.iter().map(|v| (v - pool_mean).powi(2)).sum::<f64>() / (pool.len() as f64);
        let (seed_loc, seed_scale) = gumbel_from_mean_var(pool_mean, pool_var)?;

        let mut null_loc = seed_loc;
        let mut null_scale = seed_scale.max(1e-9);

        // Target init from top slice of rank1.
        let mut desc = xs.clone();
        desc.sort_by(|a, b| b.total_cmp(a));
        let tfrac = top_frac_init.clamp(0.05, 0.5);
        let t_n = ((n as f64) * tfrac).round() as usize;
        let t_n = t_n.max(10).min(n);
        let top = &desc[..t_n];

        let t_mean = top.iter().sum::<f64>() / (top.len() as f64);
        let t_var = top.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / (top.len() as f64);
        let t_std = t_var.sqrt().max(1e-6);
        let mut target = SkewNormal::new(t_mean, t_std, 2.0);

        // pi init based on seed null mean proxy.
        let null_mean_proxy = seed_loc + EULER_MASCHERONI * seed_scale;
        let frac_above = (xs.iter().filter(|&&v| v > null_mean_proxy).count() as f64) / (n as f64);
        let mut pi = frac_above.clamp(pi_clamp.0, pi_clamp.1);

        let iters = iters.max(10).min(500);
        let mut prev_ll = -f64::INFINITY;

        for _ in 0..iters {
            let null_dist = match Gumbel::new(null_loc, null_scale.max(1e-9)) {
                Ok(d) => d,
                Err(_) => return None,
            };

            let pi0 = pi.clamp(1e-6, 1.0 - 1e-6);
            let log_pi = pi0.ln();
            let log_1m_pi = (1.0 - pi0).ln();

            // E-step: r_i = P(target | x_i)
            let mut r: Vec<f64> = Vec::with_capacity(n);
            let mut ll = 0.0;

            for &x in &xs {
                let f0 = null_dist.pdf(x).max(TINY);
                let f1 = target.pdf(x).max(TINY);

                let log_f0 = f0.ln();
                let log_f1 = f1.ln();

                let log_num = log_pi + log_f1;
                let log_den = log_add_exp(log_1m_pi + log_f0, log_num);

                ll += log_den;
                let ri = (log_num - log_den).exp();
                r.push(if ri.is_finite() { ri } else { 0.0 });
            }

            let avg_ll = ll / (n as f64);
            if prev_ll.is_finite() && (avg_ll - prev_ll).abs() < em_tol {
                break;
            }
            prev_ll = avg_ll;

            // M-step: update pi
            let sum_r = r.iter().sum::<f64>();
            if sum_r < 1e-8 || sum_r > (n as f64 - 1e-8) {
                break;
            }
            pi = (sum_r / (n as f64)).clamp(pi_clamp.0, pi_clamp.1);

            // Update target by weighted moments with weights r
            if let Some((m, v, s)) = weighted_moments(&xs, &r) {
                if let Some(sn) = SkewNormal::from_moments(m, v, s) {
                    target = sn;
                }
            }

            // Update null only if not anchored
            if !mix_anchor_incorrect {
                let w0: Vec<f64> = r.iter().map(|&ri| (1.0 - ri).max(0.0)).collect();
                if let Some((m0, v0, _s0)) = weighted_moments(&xs, &w0) {
                    if let Some((mu0, beta0)) = gumbel_from_mean_var(m0, v0) {
                        // clamp drift
                        let min_beta = seed_scale * beta_drift_mult.0.max(0.1);
                        let max_beta = seed_scale * beta_drift_mult.1.max(beta_drift_mult.0 + 0.01);
                        let beta_clamped = beta0.clamp(min_beta, max_beta);

                        let mu_clamped =
                            mu0.clamp(seed_loc - mu_drift_abs.abs(), seed_loc + mu_drift_abs.abs());

                        null_loc = mu_clamped;
                        null_scale = beta_clamped.max(1e-9);
                    }
                }
            } else {
                null_loc = seed_loc;
                null_scale = seed_scale.max(1e-9);
            }
        }

        Some(Self {
            null_seed_loc: seed_loc,
            null_seed_scale: seed_scale,
            null_loc,
            null_scale,
            target,
            pi,
            mix_anchor_incorrect,
            beta_drift_mult,
            mu_drift_abs,
        })
    }

    pub fn pep(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }
        let null_dist = match Gumbel::new(self.null_loc, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };
        let f0 = null_dist.pdf(x).max(TINY);
        let f1 = self.target.pdf(x).max(TINY);

        let pi = self.pi.clamp(1e-6, 1.0 - 1e-6);
        let num = (1.0 - pi) * f0;
        let den = num + pi * f1;
        if den > 0.0 && den.is_finite() {
            (num / den).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    pub fn p_value(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }
        let null_dist = match Gumbel::new(self.null_loc, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };
        clamp_p01(null_dist.sf(x))
    }
}

impl MsfdrParamTuple for Msfdr2SmixModel {
    fn param_tuple(&self) -> String {
        format!(
            "pi={}, null=({},{}), target=({},{},{})",
            fmt_f64(self.pi),
            fmt_f64(self.null_loc),
            fmt_f64(self.null_scale),
            fmt_f64(self.target.location),
            fmt_f64(self.target.scale),
            fmt_f64(self.target.shape),
        )
    }
}

// -----------------------------------------------------------------------------
// Backward-compatibility adapter preserving the legacy `MsfdrModel` interface.
// -----------------------------------------------------------------------------
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MsfdrModel {
    /// Proportion of true targets (pi)
    pub target_weight: f64,
    /// Distribution of True Targets (Skew-Normal)
    pub target_dist: SkewNormal,
    /// Distribution of Null/Decoys (Gumbel)
    pub null_location: f64,
    pub null_scale: f64,
}

impl MsfdrModel {
    pub fn new(
        target_weight: f64,
        target_dist: SkewNormal,
        null_location: f64,
        null_scale: f64,
    ) -> Self {
        Self {
            target_weight,
            target_dist,
            null_location,
            null_scale,
        }
    }

    pub fn posterior_probability(&self, score: f64) -> f64 {
        let f_target = self.target_dist.pdf(score).max(TINY);
        let gumbel = match Gumbel::new(self.null_location, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 0.0,
        };
        let f_null = gumbel.pdf(score).max(TINY);

        let prob_target = self.target_weight * f_target;
        let prob_null = (1.0 - self.target_weight) * f_null;
        let total = prob_target + prob_null;

        if total > 0.0 && total.is_finite() {
            (prob_target / total).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn calculate_pep(&self, score: f64) -> f64 {
        1.0 - self.posterior_probability(score)
    }
}

impl MsfdrParamTuple for MsfdrModel {
    fn param_tuple(&self) -> String {
        format!(
            "pi={}, null=({},{}), target=({},{},{})",
            fmt_f64(self.target_weight),
            fmt_f64(self.null_location),
            fmt_f64(self.null_scale),
            fmt_f64(self.target_dist.location),
            fmt_f64(self.target_dist.scale),
            fmt_f64(self.target_dist.shape),
        )
    }
}

// =============================================================================
// Validation (Tests)
// Unit tests for MSFDR math invariants (not calibration performance)
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------
    // Helpers (deterministic)
    // -----------------------

    fn grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        assert!(n >= 2);
        let step = (hi - lo) / ((n - 1) as f64);
        (0..n).map(|i| lo + (i as f64) * step).collect()
    }

    fn assert_p01(name: &str, p: f64) {
        assert!(p.is_finite(), "{name} is not finite: {p}");
        assert!((0.0..=1.0).contains(&p), "{name} out of [0,1]: {p}");
    }

    // Synthetic rank1 data: mostly "null-ish" plus some "target-ish" high scores.
    // No RNG; fixed values -> deterministic across platforms.
    fn synthetic_rank1_scores() -> Vec<f64> {
        let mut xs = Vec::new();

        // null-ish bulk: [-1.0, 1.0] step 0.02 -> 101 values
        xs.extend(grid(-1.0, 1.0, 101));

        // target-ish tail: [3.0, 6.0] step ~0.06 -> 51 values
        xs.extend(grid(3.0, 6.0, 51));

        xs
    }

    // Pure-null pool: tight-ish around [-1.5, 1.5]
    fn synthetic_pool_scores() -> Vec<f64> {
        grid(-1.5, 1.5, 121)
    }

    // -----------------------
    // 1) Bounds: pep(x), p_value(x) in [0,1] and finite on a grid
    // -----------------------

    #[test]
    fn seeded_bounds_pep_and_p_value_are_finite_in_unit_interval() {
        let xs = synthetic_rank1_scores();

        // fixed null seed (mu, beta) with reasonable scale
        let m = MsfdrSeededModel::fit_rank1_seeded(
            &xs,
            /*mu_in*/ 0.0,
            /*beta_in*/ 1.0,
            /*iters*/ 50,
            /*em_tol*/ 1e-6,
            /*pi_clamp*/ (0.01, 0.99),
            /*top_frac_init*/ 0.2,
        )
        .expect("seeded model should fit on synthetic input");

        for x in grid(-5.0, 10.0, 301) {
            let pep = m.pep(x);
            let pv = m.p_value(x);
            assert_p01("seeded pep", pep);
            assert_p01("seeded p_value", pv);
        }

        // non-finite x should fail-closed to 1.0
        assert_eq!(m.pep(f64::NAN), 1.0);
        assert_eq!(m.pep(f64::INFINITY), 1.0);
        assert_eq!(m.p_value(f64::NAN), 1.0);
        assert_eq!(m.p_value(f64::INFINITY), 1.0);
    }

    #[test]
    fn onesmix_bounds_pep_and_p_value_are_finite_in_unit_interval() {
        let xs = synthetic_rank1_scores();

        let m = Msfdr1SmixModel::fit_rank1(
            &xs,
            /*iters*/ 100,
            /*em_tol*/ 1e-6,
            /*pi_clamp*/ (0.01, 0.99),
            /*bottom_frac_init*/ 0.7,
            /*top_frac_init*/ 0.2,
            /*mu_drift*/ 0.5,
            /*beta_drift*/ (0.8, 1.25),
        )
        .expect("1Smix should fit on synthetic input");

        for x in grid(-5.0, 10.0, 301) {
            let pep = m.pep(x);
            let pv = m.p_value(x);
            assert_p01("1Smix pep", pep);
            assert_p01("1Smix p_value", pv);
        }

        assert_eq!(m.pep(f64::NAN), 1.0);
        assert_eq!(m.pep(f64::INFINITY), 1.0);
        assert_eq!(m.p_value(f64::NAN), 1.0);
        assert_eq!(m.p_value(f64::INFINITY), 1.0);
    }

    #[test]
    fn twosmix_bounds_pep_and_p_value_are_finite_in_unit_interval() {
        let xs = synthetic_rank1_scores();
        let pool = synthetic_pool_scores();

        let m = Msfdr2SmixModel::fit_rank1_with_pool(
            &xs,
            &pool,
            /*iters*/ 100,
            /*em_tol*/ 1e-6,
            /*pi_clamp*/ (0.01, 0.99),
            /*top_frac_init*/ 0.2,
            /*mix_anchor_incorrect*/ true, // anchored (null fixed to pool)
            /*beta_drift_mult*/ (0.8, 1.25),
            /*mu_drift_abs*/ 0.5,
        )
        .expect("2Smix should fit on synthetic input");

        for x in grid(-5.0, 10.0, 301) {
            let pep = m.pep(x);
            let pv = m.p_value(x);
            assert_p01("2Smix pep", pep);
            assert_p01("2Smix p_value", pv);
        }

        assert_eq!(m.pep(f64::NAN), 1.0);
        assert_eq!(m.pep(f64::INFINITY), 1.0);
        assert_eq!(m.p_value(f64::NAN), 1.0);
        assert_eq!(m.p_value(f64::INFINITY), 1.0);
    }

    // -----------------------
    // 2) Sanity monotonic trend: p_value(x) should generally decrease as x increases under null sf
    //    (allow a small number of violations for numeric wiggles)
    // -----------------------

    #[test]
    fn p_value_is_generally_nonincreasing_in_x_for_seeded() {
        let xs = synthetic_rank1_scores();
        let m = MsfdrSeededModel::fit_rank1_seeded(&xs, 0.0, 1.0, 50, 1e-6, (0.01, 0.99), 0.2)
            .expect("seeded model should fit");

        let g = grid(-5.0, 10.0, 801);
        let mut prev = m.p_value(g[0]);
        let mut violations = 0usize;

        for &x in &g[1..] {
            let cur = m.p_value(x);
            // allow tiny epsilon increases as numerical wiggles
            if cur > prev + 1e-12 {
                violations += 1;
            }
            prev = cur;
        }

        // Very permissive: <= 1% violations across a dense grid
        let max_viol = (g.len() / 100).max(3);
        assert!(
            violations <= max_viol,
            "too many monotonicity violations: {violations} > {max_viol}"
        );
    }

    #[test]
    fn p_value_is_generally_nonincreasing_in_x_for_onesmix() {
        let xs = synthetic_rank1_scores();
        let m =
            Msfdr1SmixModel::fit_rank1(&xs, 100, 1e-6, (0.01, 0.99), 0.7, 0.2, 0.5, (0.8, 1.25))
                .expect("1Smix model should fit");

        let g = grid(-5.0, 10.0, 801);
        let mut prev = m.p_value(g[0]);
        let mut violations = 0usize;

        for &x in &g[1..] {
            let cur = m.p_value(x);
            if cur > prev + 1e-12 {
                violations += 1;
            }
            prev = cur;
        }

        let max_viol = (g.len() / 100).max(3);
        assert!(
            violations <= max_viol,
            "too many monotonicity violations: {violations} > {max_viol}"
        );
    }

    #[test]
    fn p_value_is_generally_nonincreasing_in_x_for_twosmix() {
        let xs = synthetic_rank1_scores();
        let pool = synthetic_pool_scores();
        let m = Msfdr2SmixModel::fit_rank1_with_pool(
            &xs,
            &pool,
            100,
            1e-6,
            (0.01, 0.99),
            0.2,
            true,
            (0.8, 1.25),
            0.5,
        )
        .expect("2Smix model should fit");

        let g = grid(-5.0, 10.0, 801);
        let mut prev = m.p_value(g[0]);
        let mut violations = 0usize;

        for &x in &g[1..] {
            let cur = m.p_value(x);
            if cur > prev + 1e-12 {
                violations += 1;
            }
            prev = cur;
        }

        let max_viol = (g.len() / 100).max(3);
        assert!(
            violations <= max_viol,
            "too many monotonicity violations: {violations} > {max_viol}"
        );
    }

    // -----------------------
    // 3) Fit fail-closed: too-small input returns None
    // -----------------------

    #[test]
    fn fit_fail_closed_on_too_small_input() {
        // Seeded requires xs.len() >= 10 (after finite filtering)
        let too_small_9: Vec<f64> = (0..9).map(|i| i as f64).collect();
        assert!(
            MsfdrSeededModel::fit_rank1_seeded(&too_small_9, 0.0, 1.0, 50, 1e-6, (0.01, 0.99), 0.2)
                .is_none(),
            "seeded fit should return None for <10 rank1 scores"
        );

        // 1Smix requires xs.len() >= 20
        let too_small_19: Vec<f64> = (0..19).map(|i| i as f64).collect();
        assert!(
            Msfdr1SmixModel::fit_rank1(
                &too_small_19,
                50,
                1e-6,
                (0.01, 0.99),
                0.7,
                0.2,
                0.5,
                (0.8, 1.25)
            )
            .is_none(),
            "1Smix fit should return None for <20 rank1 scores"
        );

        // 2Smix requires rank1 >= 20 and pool >= 20
        let rank1_19: Vec<f64> = (0..19).map(|i| i as f64).collect();
        let pool_50: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        assert!(
            Msfdr2SmixModel::fit_rank1_with_pool(
                &rank1_19,
                &pool_50,
                50,
                1e-6,
                (0.01, 0.99),
                0.2,
                true,
                (0.8, 1.25),
                0.5
            )
            .is_none(),
            "2Smix fit should return None for rank1 <20"
        );

        let rank1_50: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        let pool_19: Vec<f64> = (0..19).map(|i| i as f64).collect();
        assert!(
            Msfdr2SmixModel::fit_rank1_with_pool(
                &rank1_50,
                &pool_19,
                50,
                1e-6,
                (0.01, 0.99),
                0.2,
                true,
                (0.8, 1.25),
                0.5
            )
            .is_none(),
            "2Smix fit should return None for pool <20"
        );
    }

    // -----------------------
    // 4) No NaN propagation: fitted models never emit NaN for typical finite inputs
    // -----------------------

    #[test]
    fn no_nan_propagation_for_fitted_models() {
        let xs = synthetic_rank1_scores();
        let pool = synthetic_pool_scores();

        let seeded =
            MsfdrSeededModel::fit_rank1_seeded(&xs, 0.0, 1.0, 50, 1e-6, (0.01, 0.99), 0.2).unwrap();
        let onesmix =
            Msfdr1SmixModel::fit_rank1(&xs, 100, 1e-6, (0.01, 0.99), 0.7, 0.2, 0.5, (0.8, 1.25))
                .unwrap();
        let twosmix = Msfdr2SmixModel::fit_rank1_with_pool(
            &xs,
            &pool,
            100,
            1e-6,
            (0.01, 0.99),
            0.2,
            true,
            (0.8, 1.25),
            0.5,
        )
        .unwrap();

        for x in grid(-10.0, 15.0, 501) {
            // seeded
            let a = seeded.pep(x);
            let b = seeded.p_value(x);
            assert!(!a.is_nan(), "seeded pep NaN at x={x}");
            assert!(!b.is_nan(), "seeded p_value NaN at x={x}");

            // onesmix
            let c = onesmix.pep(x);
            let d = onesmix.p_value(x);
            assert!(!c.is_nan(), "1Smix pep NaN at x={x}");
            assert!(!d.is_nan(), "1Smix p_value NaN at x={x}");

            // twosmix
            let e = twosmix.pep(x);
            let f = twosmix.p_value(x);
            assert!(!e.is_nan(), "2Smix pep NaN at x={x}");
            assert!(!f.is_nan(), "2Smix p_value NaN at x={x}");
        }
    }
}
