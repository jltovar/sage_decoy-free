//! Decoy-free MSFDR model fitting utilities.
//!
//! The methods in this module are based on the work of Yisu Peng, et al. published here:
//!
//! New mixture models for decoy-free false discovery rate estimation in mass spectrometry proteomics
//! Yisu Peng, Shantanu Jain, Yong Fuga Li, Michal Greguš, Alexander R. Ivanov, Olga Vitek, Predrag Radivojac,
//! Bioinformatics, Volume 36, Issue Supplement_2, December 2020, Pages i745–i753,
//! DOI: 10.1093/bioinformatics/btaa807
//! https://academic.oup.com/bioinformatics/article/36/Supplement_2/i745/6055912
//!
//! and implemented on GitHub here:
//! https://github.com/shawn-peng/DecoyFree-MSFDR

use crate::ml::skew_normal::SkewNormal;
use serde::{Deserialize, Serialize};
use statrs::consts::EULER_MASCHERONI;
use statrs::distribution::{Continuous, ContinuousCDF, Gumbel};
use std::fmt;

/// Small floor to prevent log(0) and divide-by-zero cascades.
const TINY: f64 = 1e-300;

/// Binary64 guard used only to distinguish representable score/component
/// variation from roundoff. This is not a biological or power threshold.
const IDENTIFIABILITY_ULPS: f64 = 64.0;
const MIN_EFFECTIVE_COMPONENT_SUPPORT: f64 = 3.0;

/// Explicit technical failure emitted by the MSFDR1/2 post-fit validity gate.
///
/// These failures are distinct from a valid reduced model (which the declared
/// MSFDR1/2 methods do not emit), optimizer non-convergence, and any workflow
/// fallback. An invalid fit never becomes a probability-producing model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum MsfdrMixtureFitFailure {
    NonFiniteInput {
        population: String,
        index: usize,
        value: f64,
    },
    InsufficientData {
        population: String,
        observed: usize,
        minimum: usize,
    },
    DegenerateScoreVariance {
        population: String,
        observed_range: f64,
        observed_std: f64,
        numerical_resolution: f64,
    },
    NonFiniteParameter {
        component: String,
        parameter: String,
        value: f64,
    },
    NonPositiveScale {
        component: String,
        scale: f64,
        numerical_resolution: f64,
    },
    InvalidMixtureWeight {
        component: String,
        weight: f64,
        numerical_resolution: f64,
    },
    IneffectiveComponentSupport {
        component: String,
        effective_support: f64,
        minimum: f64,
    },
    CoincidentComponents {
        left: String,
        right: String,
    },
    NoFeasibleTrial {
        model: String,
        attempted: usize,
        last_failure: String,
    },
}

impl MsfdrMixtureFitFailure {
    pub fn code(&self) -> &'static str {
        match self {
            Self::NonFiniteInput { .. } => "non_finite_input",
            Self::InsufficientData { .. } => "insufficient_data",
            Self::DegenerateScoreVariance { .. } => "degenerate_score_variance",
            Self::NonFiniteParameter { .. } => "non_finite_parameter",
            Self::NonPositiveScale { .. } => "non_positive_scale",
            Self::InvalidMixtureWeight { .. } => "invalid_mixture_weight",
            Self::IneffectiveComponentSupport { .. } => "ineffective_component_support",
            Self::CoincidentComponents { .. } => "coincident_components",
            Self::NoFeasibleTrial { .. } => "no_feasible_trial",
        }
    }
}

impl fmt::Display for MsfdrMixtureFitFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.code())?;
        match self {
            Self::NonFiniteInput {
                population,
                index,
                value,
            } => write!(f, "{population}[{index}]={value:?}"),
            Self::InsufficientData {
                population,
                observed,
                minimum,
            } => write!(f, "{population} n={observed} < {minimum}"),
            Self::DegenerateScoreVariance {
                population,
                observed_range,
                observed_std,
                numerical_resolution,
            } => write!(
                f,
                "{population} range={observed_range:.6e} std={observed_std:.6e} resolution={numerical_resolution:.6e}"
            ),
            Self::NonFiniteParameter {
                component,
                parameter,
                value,
            } => write!(f, "{component}.{parameter}={value:?}"),
            Self::NonPositiveScale {
                component,
                scale,
                numerical_resolution,
            } => write!(
                f,
                "{component}.scale={scale:.6e} <= resolution={numerical_resolution:.6e}"
            ),
            Self::InvalidMixtureWeight {
                component,
                weight,
                numerical_resolution,
            } => write!(
                f,
                "{component} weight={weight:.6e} <= resolution={numerical_resolution:.6e} or is outside the declared simplex"
            ),
            Self::IneffectiveComponentSupport {
                component,
                effective_support,
                minimum,
            } => write!(
                f,
                "{component} expected support={effective_support:.6e} < {minimum:.6e}"
            ),
            Self::CoincidentComponents { left, right } => {
                write!(f, "{left} and {right} are numerically indistinguishable")
            }
            Self::NoFeasibleTrial {
                model,
                attempted,
                last_failure,
            } => write!(
                f,
                "{model} had no technically valid fit among {attempted} deterministic trials; last failure: {last_failure}"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ScoreValidity {
    n: usize,
    resolution: f64,
}

impl ScoreValidity {
    fn new(
        population: &str,
        scores: &[f64],
        minimum: usize,
    ) -> Result<Self, MsfdrMixtureFitFailure> {
        if scores.len() < minimum {
            return Err(MsfdrMixtureFitFailure::InsufficientData {
                population: population.into(),
                observed: scores.len(),
                minimum,
            });
        }

        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut max_abs = 0.0f64;
        for (index, &value) in scores.iter().enumerate() {
            if !value.is_finite() {
                return Err(MsfdrMixtureFitFailure::NonFiniteInput {
                    population: population.into(),
                    index,
                    value,
                });
            }
            let count = index as f64 + 1.0;
            let delta = value - mean;
            mean += delta / count;
            m2 += delta * (value - mean);
            min = min.min(value);
            max = max.max(value);
            max_abs = max_abs.max(value.abs());
        }

        let range = max - min;
        let std = (m2 / scores.len() as f64).max(0.0).sqrt();
        let resolution = IDENTIFIABILITY_ULPS * f64::EPSILON * max_abs.max(range.abs()).max(1.0);
        if !range.is_finite() || !std.is_finite() || range <= resolution || std <= resolution {
            return Err(MsfdrMixtureFitFailure::DegenerateScoreVariance {
                population: population.into(),
                observed_range: range,
                observed_std: std,
                numerical_resolution: resolution,
            });
        }

        Ok(Self {
            n: scores.len(),
            resolution,
        })
    }
}

fn validate_component(
    name: &str,
    component: &SkewNormal,
    score_resolution: f64,
) -> Result<(), MsfdrMixtureFitFailure> {
    for (parameter, value) in [
        ("location", component.location),
        ("scale", component.scale),
        ("shape", component.shape),
    ] {
        if !value.is_finite() {
            return Err(MsfdrMixtureFitFailure::NonFiniteParameter {
                component: name.into(),
                parameter: parameter.into(),
                value,
            });
        }
    }
    if component.scale <= score_resolution {
        return Err(MsfdrMixtureFitFailure::NonPositiveScale {
            component: name.into(),
            scale: component.scale,
            numerical_resolution: score_resolution,
        });
    }
    Ok(())
}

fn validate_weight(name: &str, weight: f64) -> Result<(), MsfdrMixtureFitFailure> {
    let resolution = IDENTIFIABILITY_ULPS * f64::EPSILON;
    if !weight.is_finite() || weight <= resolution || weight >= 1.0 - resolution {
        return Err(MsfdrMixtureFitFailure::InvalidMixtureWeight {
            component: name.into(),
            weight,
            numerical_resolution: resolution,
        });
    }
    Ok(())
}

fn validate_effective_support(name: &str, support: f64) -> Result<(), MsfdrMixtureFitFailure> {
    if !support.is_finite() || support < MIN_EFFECTIVE_COMPONENT_SUPPORT {
        return Err(MsfdrMixtureFitFailure::IneffectiveComponentSupport {
            component: name.into(),
            effective_support: support,
            minimum: MIN_EFFECTIVE_COMPONENT_SUPPORT,
        });
    }
    Ok(())
}

fn components_distinct(
    left_name: &str,
    left: &SkewNormal,
    right_name: &str,
    right: &SkewNormal,
    score_resolution: f64,
) -> Result<(), MsfdrMixtureFitFailure> {
    let scale_resolution =
        (IDENTIFIABILITY_ULPS * f64::EPSILON * left.scale.abs().max(right.scale.abs()).max(1.0))
            .max(score_resolution);
    let shape_resolution =
        IDENTIFIABILITY_ULPS * f64::EPSILON * left.shape.abs().max(right.shape.abs()).max(1.0);
    if (left.location - right.location).abs() <= score_resolution
        && (left.scale - right.scale).abs() <= scale_resolution
        && (left.shape - right.shape).abs() <= shape_resolution
    {
        return Err(MsfdrMixtureFitFailure::CoincidentComponents {
            left: left_name.into(),
            right: right_name.into(),
        });
    }
    Ok(())
}

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

fn sample_skewness(x: &[f64], mean: f64, var: f64) -> f64 {
    if x.len() < 3 || !mean.is_finite() || !var.is_finite() || var <= 0.0 {
        return 0.0;
    }

    let sd = var.sqrt().max(1e-12);
    let mut m3 = 0.0;

    for &xi in x {
        if xi.is_finite() {
            let z = (xi - mean) / sd;
            m3 += z * z * z;
        }
    }

    (m3 / x.len() as f64).clamp(-0.99, 0.99)
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

/// One-sample skew-normal mixture model for rank-1 scores.
///
/// This is the Sage Decoy-Free implementation of MSFDR 1SMix:
///
///   S1 ~ a * SN(hc) + (1 - a) * SN(h1)
///
/// where:
/// - `correct` is the high-score correct-PSM component C
/// - `incorrect1` is the rank-1 incorrect-PSM component I1
/// - `a` is the mixture weight of the correct component
///
/// This model is rank-1-only. It does not use a lower-rank null pool and does
/// not use an externally seeded null.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Msfdr1SmixModel {
    pub correct: SkewNormal,
    pub incorrect1: SkewNormal,
    pub a: f64,
}

impl Msfdr1SmixModel {
    #[inline]
    fn skew_normal_mean(sn: &SkewNormal) -> f64 {
        let delta = sn.shape / (1.0 + sn.shape * sn.shape).sqrt();
        sn.location + sn.scale * delta * (2.0 / std::f64::consts::PI).sqrt()
    }

    /// Validate a fitted 1SMix model against the population it models.
    /// Successful validation never mutates the fitted parameters.
    pub fn validate_for_scores(&self, rank1_scores: &[f64]) -> Result<(), MsfdrMixtureFitFailure> {
        let scores = ScoreValidity::new("rank1", rank1_scores, 20)?;
        validate_component("correct", &self.correct, scores.resolution)?;
        validate_component("incorrect1", &self.incorrect1, scores.resolution)?;
        validate_weight("correct", self.a)?;
        validate_weight("incorrect1", 1.0 - self.a)?;
        validate_effective_support("correct", scores.n as f64 * self.a)?;
        validate_effective_support("incorrect1", scores.n as f64 * (1.0 - self.a))?;
        components_distinct(
            "correct",
            &self.correct,
            "incorrect1",
            &self.incorrect1,
            scores.resolution,
        )?;
        Ok(())
    }

    #[inline]
    fn component_strength(sn: &SkewNormal, q90: f64, q95: f64) -> f64 {
        let mean = Self::skew_normal_mean(sn);
        let sf90 = sn.sf(q90).max(TINY);
        let sf95 = sn.sf(q95).max(TINY);

        // Higher-is-better score orientation: prefer the component with the
        // stronger upper tail, not merely the larger location parameter.
        mean + 0.5 * (-sf90.log10()) + 0.5 * (-sf95.log10())
    }

    fn avg_log_likelihood_for(
        xs: &[f64],
        correct: &SkewNormal,
        incorrect1: &SkewNormal,
        a: f64,
    ) -> f64 {
        if xs.is_empty() {
            return f64::NEG_INFINITY;
        }

        let a = a.clamp(1e-6, 1.0 - 1e-6);
        let log_a = a.ln();
        let log_1ma = (1.0 - a).ln();

        let mut ll = 0.0;
        for &x in xs {
            let fc = correct.pdf(x).max(TINY);
            let f1 = incorrect1.pdf(x).max(TINY);

            ll += log_add_exp(log_a + fc.ln(), log_1ma + f1.ln());
        }

        ll / xs.len() as f64
    }

    fn orient_by_upper_tail(mut self, xs: &[f64]) -> Self {
        if xs.len() < 5 {
            return self;
        }

        let q90 = xs[((xs.len() as f64 * 0.90).floor() as usize).min(xs.len() - 1)];
        let q95 = xs[((xs.len() as f64 * 0.95).floor() as usize).min(xs.len() - 1)];

        let c_strength = Self::component_strength(&self.correct, q90, q95);
        let i_strength = Self::component_strength(&self.incorrect1, q90, q95);

        if i_strength > c_strength {
            std::mem::swap(&mut self.correct, &mut self.incorrect1);
            self.a = (1.0 - self.a).clamp(1e-6, 1.0 - 1e-6);
        }

        self
    }

    fn tail_sanity_penalty(&self, xs: &[f64]) -> f64 {
        if xs.len() < 5 {
            return 0.0;
        }

        let q90 = xs[((xs.len() as f64 * 0.90).floor() as usize).min(xs.len() - 1)];
        let q95 = xs[((xs.len() as f64 * 0.95).floor() as usize).min(xs.len() - 1)];

        let c_mean = Self::skew_normal_mean(&self.correct);
        let i_mean = Self::skew_normal_mean(&self.incorrect1);

        let c_sf90 = self.correct.sf(q90).max(TINY);
        let i_sf90 = self.incorrect1.sf(q90).max(TINY);
        let c_sf95 = self.correct.sf(q95).max(TINY);
        let i_sf95 = self.incorrect1.sf(q95).max(TINY);

        let mut penalty = 0.0;

        if c_mean <= i_mean {
            penalty += 10.0;
        }
        if c_sf90 <= i_sf90 {
            penalty += 5.0;
        }
        if c_sf95 <= i_sf95 {
            penalty += 5.0;
        }

        penalty
    }

    fn fit_rank1_once(
        xs: &[f64],
        iters: usize,
        em_tol: f64,
        pi_clamp: (f64, f64),
        bottom_frac_init: f64,
        top_frac_init: f64,
        bottom_skew_seed: f64,
        top_skew_seed: f64,
    ) -> Option<(Self, f64)> {
        let n = xs.len();
        if n < 20 {
            return None;
        }

        let min_seed_n = if n < 100 {
            10usize
        } else if n < 1000 {
            25usize
        } else {
            50usize
        };

        // Incorrect I1 initialization from the lower part of S1.
        let bottom_frac = bottom_frac_init.clamp(0.10, 0.90);
        let bottom_n = ((n as f64) * bottom_frac).round() as usize;
        let bottom_n = bottom_n.max(min_seed_n).min(n);
        let bottom = &xs[..bottom_n];

        // Correct C initialization from the upper part of S1.
        let top_frac = top_frac_init.clamp(0.05, 0.50);
        let top_n = ((n as f64) * top_frac).round() as usize;
        let top_n = top_n.max(min_seed_n).min(n);
        let top = &xs[(n - top_n)..];

        if bottom.len() < 10 || top.len() < 10 {
            return None;
        }

        let b_mean = bottom.iter().sum::<f64>() / bottom.len() as f64;
        let b_var = bottom.iter().map(|v| (v - b_mean).powi(2)).sum::<f64>() / bottom.len() as f64;
        let b_skew_emp = sample_skewness(bottom, b_mean, b_var);
        let b_skew = if bottom_skew_seed == 0.0 {
            0.0
        } else {
            (bottom_skew_seed * b_skew_emp).clamp(-0.99, 0.99)
        };

        let mut incorrect1 = SkewNormal::from_moments(b_mean, b_var.max(1e-12), b_skew)
            .unwrap_or_else(|| SkewNormal::new(b_mean, b_var.sqrt().max(1e-6), 0.0));

        let t_mean = top.iter().sum::<f64>() / top.len() as f64;
        let t_var = top.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / top.len() as f64;
        let t_skew_emp = sample_skewness(top, t_mean, t_var);
        let t_skew = if top_skew_seed == 0.0 {
            0.0
        } else {
            (top_skew_seed * t_skew_emp).clamp(-0.99, 0.99)
        };

        let mut correct = SkewNormal::from_moments(t_mean, t_var.max(1e-12), t_skew)
            .unwrap_or_else(|| SkewNormal::new(t_mean, t_var.sqrt().max(1e-6), 0.0));

        let mut a_mix = 0.5f64.clamp(pi_clamp.0, pi_clamp.1);

        let iters = iters.max(10).min(500);
        let mut prev_ll = f64::NEG_INFINITY;

        for _ in 0..iters {
            let a0 = a_mix.clamp(1e-6, 1.0 - 1e-6);
            let log_a = a0.ln();
            let log_1ma = (1.0 - a0).ln();

            let mut pc = Vec::with_capacity(n);
            let mut p1 = Vec::with_capacity(n);
            let mut ll = 0.0;

            for &x in xs {
                let fc = correct.pdf(x).max(TINY);
                let f1 = incorrect1.pdf(x).max(TINY);

                let lc = log_a + fc.ln();
                let l1 = log_1ma + f1.ln();
                let den = log_add_exp(lc, l1);

                ll += den;

                let rc = (lc - den).exp();
                let ri = (l1 - den).exp();

                pc.push(if rc.is_finite() { rc } else { 0.0 });
                p1.push(if ri.is_finite() { ri } else { 1.0 });
            }

            let avg_ll = ll / n as f64;
            if prev_ll.is_finite() && (avg_ll - prev_ll).abs() < em_tol {
                break;
            }
            prev_ll = avg_ll;

            let sum_pc = pc.iter().sum::<f64>();
            if sum_pc <= 1e-8 || sum_pc >= n as f64 - 1e-8 {
                break;
            }

            a_mix = (sum_pc / n as f64).clamp(pi_clamp.0, pi_clamp.1);

            if let Some((m, v, s)) = weighted_moments(xs, &pc) {
                if let Some(sn) = SkewNormal::from_moments(m, v.max(1e-12), s.clamp(-0.99, 0.99)) {
                    correct = sn;
                }
            }

            if let Some((m, v, s)) = weighted_moments(xs, &p1) {
                if let Some(sn) = SkewNormal::from_moments(m, v.max(1e-12), s.clamp(-0.99, 0.99)) {
                    incorrect1 = sn;
                }
            }
        }

        let model = Self {
            correct,
            incorrect1,
            a: a_mix,
        }
        .orient_by_upper_tail(xs);

        let ll = Self::avg_log_likelihood_for(xs, &model.correct, &model.incorrect1, model.a);
        let penalty = model.tail_sanity_penalty(xs);

        Some((model, ll - penalty))
    }

    pub fn fit_rank1_checked(
        rank1_scores: &[f64],
        iters: usize,
        em_tol: f64,
        pi_clamp: (f64, f64),
        bottom_frac_init: f64,
        top_frac_init: f64,
    ) -> Result<Self, MsfdrMixtureFitFailure> {
        ScoreValidity::new("rank1", rank1_scores, 20)?;
        let mut xs = rank1_scores.to_vec();

        xs.sort_by(|a, b| a.total_cmp(b));
        let n = xs.len();

        let mut bottom_fracs = vec![
            bottom_frac_init.clamp(0.10, 0.90),
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
        ];
        bottom_fracs.sort_by(|a, b| a.total_cmp(b));
        bottom_fracs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let mut top_fracs = vec![top_frac_init.clamp(0.05, 0.50), 0.10, 0.20, 0.30, 0.40];
        top_fracs.sort_by(|a, b| a.total_cmp(b));
        top_fracs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        // Small-dataset fallback: keep the search compact and close to the
        // user-provided/paper-like initialization so seeds are not too sparse.
        if n < 100 {
            bottom_fracs = vec![bottom_frac_init.clamp(0.10, 0.90), 0.50];
            top_fracs = vec![top_frac_init.clamp(0.05, 0.50), 0.20];
            bottom_fracs.sort_by(|a, b| a.total_cmp(b));
            bottom_fracs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
            top_fracs.sort_by(|a, b| a.total_cmp(b));
            top_fracs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        }

        let skew_signs = [1.0, -1.0, 0.0];

        let mut best: Option<(Self, f64, f64, f64)> = None;
        let mut attempted = 0usize;
        let mut last_failure = None;

        for &bottom_frac in &bottom_fracs {
            for &top_frac in &top_fracs {
                for &bottom_skew_seed in &skew_signs {
                    for &top_skew_seed in &skew_signs {
                        attempted += 1;
                        let Some((model, score)) = Self::fit_rank1_once(
                            &xs,
                            iters,
                            em_tol,
                            pi_clamp,
                            bottom_frac,
                            top_frac,
                            bottom_skew_seed,
                            top_skew_seed,
                        ) else {
                            continue;
                        };

                        if let Err(error) = model.validate_for_scores(&xs) {
                            last_failure = Some(error.to_string());
                            continue;
                        }

                        let replace = best
                            .as_ref()
                            .map(|(_, best_score, _, _)| score > *best_score)
                            .unwrap_or(true);

                        if replace {
                            best = Some((model, score, bottom_frac, top_frac));
                        }
                    }
                }
            }
        }

        let Some((model, score, bottom_frac, top_frac)) = best else {
            return Err(MsfdrMixtureFitFailure::NoFeasibleTrial {
                model: "MSFDR1-SMIX".into(),
                attempted,
                last_failure: last_failure
                    .unwrap_or_else(|| "optimizer did not return a finite candidate model".into()),
            });
        };

        log::info!(
            "DF MSFDR 1smix selected init: bottom_frac={:.3} top_frac={:.3} score={:.6e} correct_mean={:.6e} incorrect_mean={:.6e}",
            bottom_frac,
            top_frac,
            score,
            Self::skew_normal_mean(&model.correct),
            Self::skew_normal_mean(&model.incorrect1),
        );

        Ok(model)
    }

    /// Backward-compatible optional fit. Production callers that need
    /// provenance must use [`Self::fit_rank1_checked`].
    pub fn fit_rank1(
        rank1_scores: &[f64],
        iters: usize,
        em_tol: f64,
        pi_clamp: (f64, f64),
        bottom_frac_init: f64,
        top_frac_init: f64,
    ) -> Option<Self> {
        Self::fit_rank1_checked(
            rank1_scores,
            iters,
            em_tol,
            pi_clamp,
            bottom_frac_init,
            top_frac_init,
        )
        .ok()
    }

    /// Local posterior error probability: P(I1 | x).
    pub fn pep(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }

        let a = self.a.clamp(1e-6, 1.0 - 1e-6);

        let fc = self.correct.pdf(x).max(TINY);
        let f1 = self.incorrect1.pdf(x).max(TINY);

        let num = (1.0 - a) * f1;
        let den = a * fc + num;

        if den > 0.0 && den.is_finite() {
            (num / den).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Paper-style FDR estimate at score threshold x:
    ///
    /// FDR(x) = (1-a) * P(I1 > x) / P(S1 > x)
    pub fn fdr_at_score(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }

        let a = self.a.clamp(1e-6, 1.0 - 1e-6);

        let sf_c = self.correct.sf(x).max(TINY);
        let sf_i1 = self.incorrect1.sf(x).max(TINY);

        let num = (1.0 - a) * sf_i1;
        let den = a * sf_c + (1.0 - a) * sf_i1;

        if den > 0.0 && den.is_finite() {
            (num / den).clamp(0.0, 1.0).max(TINY)
        } else {
            1.0
        }
    }

    /// Native p-like tail probability under the rank-1 incorrect component.
    ///
    /// This is the correct stream for `decoy_free_p_value`.
    /// The paper-style threshold FDR curve is available separately as
    /// `fdr_at_score(x)` and must not be used as a per-PSM p-value.
    pub fn p_value(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }

        self.incorrect1.sf(x).clamp(0.0, 1.0).max(TINY)
    }
}

impl MsfdrParamTuple for Msfdr1SmixModel {
    fn param_tuple(&self) -> String {
        format!(
            "a={}, correct=({},{},{}), incorrect1=({},{},{})",
            fmt_f64(self.a),
            fmt_f64(self.correct.location),
            fmt_f64(self.correct.scale),
            fmt_f64(self.correct.shape),
            fmt_f64(self.incorrect1.location),
            fmt_f64(self.incorrect1.scale),
            fmt_f64(self.incorrect1.shape),
        )
    }
}

/// Pooled-rank two-sample skew-normal mixture model.
///
/// This is the Sage adaptation of MSFDR 2SMix:
///
///   S1 ~ a * SN(hc) + (1 - a) * SN(h1)
///   S2 ~ a * SN(h1) + (1 - a - b) * SN(h2) + b * SN(hc)
///
/// where:
/// - S1 is rank-1 scores
/// - S2 is the pooled lower-rank score sample selected by
///   msfdr2_smix_min_null_rank..=msfdr2_smix_max_null_rank
/// - C  is the correct component
/// - I1 is the first-incorrect component
/// - I2 is the second/lower-rank incorrect component
///
/// If S2 is rank 2 only, this matches the paper more closely. If S2 includes
/// ranks 2..K, this is a pooled-rank generalization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Msfdr2SmixModel {
    pub correct: SkewNormal,
    pub incorrect1: SkewNormal,
    pub incorrect2: SkewNormal,
    pub a: f64,
    pub b: f64,
}

impl Msfdr2SmixModel {
    /// Validate a fitted pooled 2SMix model against both modeled populations.
    /// Successful validation never mutates the fitted parameters.
    pub fn validate_for_scores(
        &self,
        rank1_scores: &[f64],
        pooled_rank_scores: &[f64],
    ) -> Result<(), MsfdrMixtureFitFailure> {
        let s1 = ScoreValidity::new("rank1", rank1_scores, 20)?;
        let s2 = ScoreValidity::new("pooled_rank", pooled_rank_scores, 20)?;
        let resolution = s1.resolution.max(s2.resolution);
        validate_component("correct", &self.correct, resolution)?;
        validate_component("incorrect1", &self.incorrect1, resolution)?;
        validate_component("incorrect2", &self.incorrect2, resolution)?;

        let s2_balance = (s1.n as f64 / s2.n as f64).clamp(1e-6, 1.0);
        let a_s2 = self.a * s2_balance;
        let i2_weight = 1.0 - a_s2 - self.b;
        for (name, weight) in [
            ("correct_s1", self.a),
            ("incorrect1_s1", 1.0 - self.a),
            ("incorrect1_s2", a_s2),
            ("correct_s2", self.b),
            ("incorrect2_s2", i2_weight),
        ] {
            validate_weight(name, weight)?;
        }

        validate_effective_support("correct", s1.n as f64 * self.a + s2.n as f64 * self.b)?;
        validate_effective_support(
            "incorrect1",
            s1.n as f64 * (1.0 - self.a) + s2.n as f64 * a_s2,
        )?;
        validate_effective_support("incorrect2", s2.n as f64 * i2_weight)?;

        components_distinct(
            "correct",
            &self.correct,
            "incorrect1",
            &self.incorrect1,
            resolution,
        )?;
        components_distinct(
            "correct",
            &self.correct,
            "incorrect2",
            &self.incorrect2,
            resolution,
        )?;
        components_distinct(
            "incorrect1",
            &self.incorrect1,
            "incorrect2",
            &self.incorrect2,
            resolution,
        )?;

        Ok(())
    }

    pub fn fit_top_two_pooled_checked(
        rank1_scores: &[f64],
        pooled_rank_scores: &[f64],
        iters: usize,
        em_tol: f64,
        pi_clamp: (f64, f64),
        bottom_frac_init: f64,
        top_frac_init: f64,
    ) -> Result<Self, MsfdrMixtureFitFailure> {
        ScoreValidity::new("rank1", rank1_scores, 20)?;
        ScoreValidity::new("pooled_rank", pooled_rank_scores, 20)?;
        let mut s1 = rank1_scores.to_vec();
        let mut s2 = pooled_rank_scores.to_vec();

        s1.sort_by(|a, b| a.total_cmp(b));
        s2.sort_by(|a, b| a.total_cmp(b));

        let n1 = s1.len();
        let n2 = s2.len();

        // In strict paper 2SMix, S2 is one second-best score per spectrum,
        // so |S2| is comparable to |S1|. In the pooled-rank Sage variant,
        // S2 may contain many lower-rank scores per spectrum. Without this
        // balancing factor, broad S2 rank windows dominate the EM updates and
        // can drive over-permissive, entrapment-heavy fits.
        let s2_balance = ((n1 as f64) / (n2 as f64)).clamp(1e-6, 1.0);

        // Initialize I1 from the lower part of S1.
        let bottom_frac = bottom_frac_init.clamp(0.10, 0.90);
        let bottom_n = ((n1 as f64) * bottom_frac).round() as usize;
        let bottom_n = bottom_n.max(10).min(n1);
        let s1_low = &s1[..bottom_n];

        let i1_mean = s1_low.iter().sum::<f64>() / s1_low.len() as f64;
        let i1_var =
            s1_low.iter().map(|v| (v - i1_mean).powi(2)).sum::<f64>() / s1_low.len() as f64;
        let i1_skew = sample_skewness(s1_low, i1_mean, i1_var);

        let mut incorrect1 = SkewNormal::from_moments(i1_mean, i1_var.max(1e-12), i1_skew)
            .unwrap_or_else(|| SkewNormal::new(i1_mean, i1_var.sqrt().max(1e-6), 0.0));

        // Initialize C from top fraction of S1.
        let top_frac = top_frac_init.clamp(0.05, 0.50);
        let top_n = ((n1 as f64) * top_frac).round() as usize;
        let top_n = top_n.max(10).min(n1);
        let s1_top = &s1[(n1 - top_n)..];

        let c_mean = s1_top.iter().sum::<f64>() / s1_top.len() as f64;
        let c_var = s1_top.iter().map(|v| (v - c_mean).powi(2)).sum::<f64>() / s1_top.len() as f64;
        let c_skew = sample_skewness(s1_top, c_mean, c_var);

        let mut correct = SkewNormal::from_moments(c_mean, c_var.max(1e-12), c_skew)
            .unwrap_or_else(|| SkewNormal::new(c_mean, c_var.sqrt().max(1e-6), 0.0));

        // Initialize I2 from pooled lower-rank scores.
        let i2_mean = s2.iter().sum::<f64>() / n2 as f64;
        let i2_var = s2.iter().map(|v| (v - i2_mean).powi(2)).sum::<f64>() / n2 as f64;
        let i2_skew = sample_skewness(&s2, i2_mean, i2_var);

        let mut incorrect2 = SkewNormal::from_moments(i2_mean, i2_var.max(1e-12), i2_skew)
            .unwrap_or_else(|| SkewNormal::new(i2_mean, i2_var.sqrt().max(1e-6), 0.0));

        // Paper initializes a and b at 0.5, subject to a+b<=1.
        // Use conservative valid starting point to avoid immediate degeneracy.
        let mut a_mix = 0.45f64.clamp(pi_clamp.0, pi_clamp.1);
        let mut b_mix = 0.05f64;
        if a_mix + b_mix >= 0.95 {
            b_mix = (0.95 - a_mix).max(1e-6);
        }

        let iters = iters.max(10).min(500);
        let mut prev_ll = f64::NEG_INFINITY;

        for _ in 0..iters {
            let a0 = a_mix.clamp(1e-6, 1.0 - 1e-6);

            // In strict paper 2SMix, S2 is one rank-2 score per spectrum, so the
            // I1 prior in S2 is `a`. In pooled-rank Sage 2SMix, S2 contains many
            // lower-rank scores per rank-1 score. Therefore the I1 prior inside S2
            // must be diluted by the effective S1/S2 sampling ratio.
            let a_s2 = (a0 * s2_balance).clamp(1e-6, 1.0 - 1e-6);

            let b0 = b_mix.clamp(1e-6, 1.0 - a_s2 - 1e-6);
            let i2_weight = (1.0 - a_s2 - b0).clamp(1e-6, 1.0);

            // ---------- E-step for S1 ----------
            let mut pc_s1 = Vec::with_capacity(n1);
            let mut p1_s1 = Vec::with_capacity(n1);

            let mut ll = 0.0;

            for &x in &s1 {
                let fc = correct.pdf(x).max(TINY);
                let f1 = incorrect1.pdf(x).max(TINY);

                let lc = a0.ln() + fc.ln();
                let l1 = (1.0 - a0).ln() + f1.ln();
                let den = log_add_exp(lc, l1);

                ll += den;

                pc_s1.push((lc - den).exp().clamp(0.0, 1.0));
                p1_s1.push((l1 - den).exp().clamp(0.0, 1.0));
            }

            // ---------- E-step for pooled S2 ----------
            let mut rc_s2 = Vec::with_capacity(n2);
            let mut r1_s2 = Vec::with_capacity(n2);
            let mut r2_s2 = Vec::with_capacity(n2);

            for &x in &s2 {
                let fc = correct.pdf(x).max(TINY);
                let f1 = incorrect1.pdf(x).max(TINY);
                let f2 = incorrect2.pdf(x).max(TINY);

                let lc = b0.ln() + fc.ln();
                let l1 = a_s2.ln() + f1.ln();
                let l2 = i2_weight.ln() + f2.ln();

                let den12 = log_add_exp(l1, l2);
                let den = log_add_exp(lc, den12);

                ll += den;

                rc_s2.push((lc - den).exp().clamp(0.0, 1.0));
                r1_s2.push((l1 - den).exp().clamp(0.0, 1.0));
                r2_s2.push((l2 - den).exp().clamp(0.0, 1.0));
            }

            let avg_ll = ll / (n1 + n2) as f64;
            if prev_ll.is_finite() && (avg_ll - prev_ll).abs() < em_tol {
                break;
            }
            prev_ll = avg_ll;

            // ---------- M-step mixture weights ----------
            let sum_pc_s1 = pc_s1.iter().sum::<f64>();
            let sum_r1_s2 = r1_s2.iter().sum::<f64>();
            let sum_rc_s2 = rc_s2.iter().sum::<f64>();

            // Effective S2 size is balanced to approximately match S1. This prevents
            // pooled lower-rank depth from dominating the estimate of `a`.
            let effective_n2 = (n2 as f64) * s2_balance;

            let new_a = ((sum_pc_s1 + sum_r1_s2) / (n1 as f64 + effective_n2))
                .clamp(pi_clamp.0, pi_clamp.1);

            let new_a_s2 = (new_a * s2_balance).clamp(1e-6, 1.0 - 1e-6);

            let mut new_b = (sum_rc_s2 / n2 as f64).clamp(1e-6, 1.0 - new_a_s2 - 1e-6);

            if new_a_s2 + new_b >= 0.999 {
                new_b = (0.999 - new_a_s2).max(1e-6);
            }

            a_mix = new_a;
            b_mix = new_b;

            // ---------- M-step component parameters ----------
            // C is updated from S1 pc + S2 rc.
            let mut c_x = Vec::with_capacity(n1 + n2);
            let mut c_w = Vec::with_capacity(n1 + n2);
            c_x.extend_from_slice(&s1);
            c_w.extend_from_slice(&pc_s1);
            c_x.extend_from_slice(&s2);
            c_w.extend_from_slice(&rc_s2);

            if let Some((m, v, s)) = weighted_moments(&c_x, &c_w) {
                if let Some(sn) = SkewNormal::from_moments(m, v.max(1e-12), s) {
                    correct = sn;
                }
            }

            // I1 is updated from S1 p1 + S2 r1. The S2 I1 responsibilities are already
            // diluted in the E-step by `a_s2`, so do not multiply them again here.
            let mut i1_x = Vec::with_capacity(n1 + n2);
            let mut i1_w = Vec::with_capacity(n1 + n2);
            i1_x.extend_from_slice(&s1);
            i1_w.extend_from_slice(&p1_s1);
            i1_x.extend_from_slice(&s2);
            i1_w.extend_from_slice(&r1_s2);

            if let Some((m, v, s)) = weighted_moments(&i1_x, &i1_w) {
                if let Some(sn) = SkewNormal::from_moments(m, v.max(1e-12), s) {
                    incorrect1 = sn;
                }
            }

            // I2 is updated from S2 r2 only.
            if let Some((m, v, s)) = weighted_moments(&s2, &r2_s2) {
                if let Some(sn) = SkewNormal::from_moments(m, v.max(1e-12), s) {
                    incorrect2 = sn;
                }
            }
        }

        let model = Self {
            correct,
            incorrect1,
            incorrect2,
            a: a_mix,
            b: b_mix,
        };
        model.validate_for_scores(&s1, &s2)?;
        Ok(model)
    }

    /// Backward-compatible optional fit. Production callers that need
    /// provenance must use [`Self::fit_top_two_pooled_checked`].
    pub fn fit_top_two_pooled(
        rank1_scores: &[f64],
        pooled_rank_scores: &[f64],
        iters: usize,
        em_tol: f64,
        pi_clamp: (f64, f64),
        bottom_frac_init: f64,
        top_frac_init: f64,
    ) -> Option<Self> {
        Self::fit_top_two_pooled_checked(
            rank1_scores,
            pooled_rank_scores,
            iters,
            em_tol,
            pi_clamp,
            bottom_frac_init,
            top_frac_init,
        )
        .ok()
    }

    /// Local posterior error probability for an S1/rank1 score: P(I1 | S1=x).
    pub fn pep(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }

        let a = self.a.clamp(1e-6, 1.0 - 1e-6);

        let fc = self.correct.pdf(x).max(TINY);
        let f1 = self.incorrect1.pdf(x).max(TINY);

        let num = (1.0 - a) * f1;
        let den = a * fc + num;

        if den > 0.0 && den.is_finite() {
            (num / den).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Paper-style FDR estimate for accepting rank1 scores above x.
    pub fn fdr_at_score(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }

        let a = self.a.clamp(1e-6, 1.0 - 1e-6);

        let sf_c = self.correct.sf(x).max(TINY);
        let sf_i1 = self.incorrect1.sf(x).max(TINY);

        let num = (1.0 - a) * sf_i1;
        let den = a * sf_c + (1.0 - a) * sf_i1;

        if den > 0.0 && den.is_finite() {
            (num / den).clamp(0.0, 1.0).max(TINY)
        } else {
            1.0
        }
    }

    /// Native p-like tail probability under the rank-1 incorrect component.
    ///
    /// For rank-1 acceptance, the null/incorrect comparator is I1, not I2.
    /// The paper-style threshold FDR curve is available separately as
    /// `fdr_at_score(x)` and must not be used as a per-PSM p-value.
    pub fn p_value(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }

        self.incorrect1.sf(x).clamp(0.0, 1.0).max(TINY)
    }
}

impl MsfdrParamTuple for Msfdr2SmixModel {
    fn param_tuple(&self) -> String {
        format!(
            "a={}, b={}, correct=({},{},{}), incorrect1=({},{},{}), incorrect2=({},{},{})",
            fmt_f64(self.a),
            fmt_f64(self.b),
            fmt_f64(self.correct.location),
            fmt_f64(self.correct.scale),
            fmt_f64(self.correct.shape),
            fmt_f64(self.incorrect1.location),
            fmt_f64(self.incorrect1.scale),
            fmt_f64(self.incorrect1.shape),
            fmt_f64(self.incorrect2.location),
            fmt_f64(self.incorrect2.scale),
            fmt_f64(self.incorrect2.shape),
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

        let m = Msfdr1SmixModel::fit_rank1_checked(
            &xs,
            /*iters*/ 100,
            /*em_tol*/ 1e-6,
            /*pi_clamp*/ (0.01, 0.99),
            /*bottom_frac_init*/ 0.7,
            /*top_frac_init*/ 0.2,
        )
        .unwrap_or_else(|error| panic!("1Smix should fit on synthetic input: {error}"));

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
    fn pooled_rank_twosmix_bounds_pep_and_p_value_are_finite_in_unit_interval() {
        let xs = synthetic_rank1_scores();
        let pool = synthetic_pool_scores();

        let m = Msfdr2SmixModel::fit_top_two_pooled_checked(
            &xs,
            &pool,
            /*iters*/ 100,
            /*em_tol*/ 1e-6,
            /*pi_clamp*/ (0.01, 0.99),
            /*bottom_frac_init*/ 0.5,
            /*top_frac_init*/ 0.2,
        )
        .unwrap_or_else(|error| panic!("2Smix should fit on synthetic input: {error}"));

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
        let m = Msfdr1SmixModel::fit_rank1(&xs, 100, 1e-6, (0.01, 0.99), 0.7, 0.2)
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
        let m = Msfdr2SmixModel::fit_top_two_pooled(&xs, &pool, 100, 1e-6, (0.01, 0.99), 0.5, 0.2)
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
            Msfdr1SmixModel::fit_rank1(&too_small_19, 50, 1e-6, (0.01, 0.99), 0.7, 0.2).is_none(),
            "1Smix fit should return None for <20 rank1 scores"
        );

        // 2Smix requires rank1 >= 20 and pool >= 20
        let rank1_19: Vec<f64> = (0..19).map(|i| i as f64).collect();
        let pool_50: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        assert!(
            Msfdr2SmixModel::fit_top_two_pooled(
                &rank1_19,
                &pool_50,
                50,
                1e-6,
                (0.01, 0.99),
                0.5,
                0.2,
            )
            .is_none(),
            "2Smix fit should return None for rank1 <20"
        );

        let rank1_50: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        let pool_19: Vec<f64> = (0..19).map(|i| i as f64).collect();
        assert!(
            Msfdr2SmixModel::fit_top_two_pooled(
                &rank1_50,
                &pool_19,
                50,
                1e-6,
                (0.01, 0.99),
                0.5,
                0.2,
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
        let onesmix = Msfdr1SmixModel::fit_rank1(&xs, 100, 1e-6, (0.01, 0.99), 0.7, 0.2).unwrap();
        let twosmix =
            Msfdr2SmixModel::fit_top_two_pooled(&xs, &pool, 100, 1e-6, (0.01, 0.99), 0.5, 0.2)
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

    #[test]
    fn statistical_conformance_pooled_s2_exact_replication_is_fit_invariant() {
        // Exact row replication carries no new information. The pooled-rank
        // balancing extension should therefore preserve its fitted state.
        let rank1 = synthetic_rank1_scores();
        let pool = synthetic_pool_scores();
        let replicated: Vec<f64> = pool
            .iter()
            .flat_map(|&x| std::iter::repeat_n(x, 4))
            .collect();
        let fit = |s2: &[f64]| {
            Msfdr2SmixModel::fit_top_two_pooled(&rank1, s2, 200, 1e-8, (0.01, 0.99), 0.5, 0.2)
                .expect("2SMix fit")
        };
        let one = fit(&pool);
        let four = fit(&replicated);
        assert!(
            (one.a - four.a).abs() <= 0.03,
            "pooled-S2 replication changed a: one={} four={}",
            one.a,
            four.a
        );
        assert!(
            (one.b - four.b).abs() <= 0.03,
            "pooled-S2 replication changed b: one={} four={}",
            one.b,
            four.b
        );
    }

    #[test]
    fn statistical_conformance_coincident_mixture_components_fail_closed() {
        let rank1 = vec![0.0; 100];
        let pool = vec![0.0; 100];
        let onesmix = Msfdr1SmixModel::fit_rank1_checked(&rank1, 100, 1e-8, (0.01, 0.99), 0.5, 0.2);
        let twosmix = Msfdr2SmixModel::fit_top_two_pooled_checked(
            &rank1,
            &pool,
            100,
            1e-8,
            (0.01, 0.99),
            0.5,
            0.2,
        );
        assert!(
            matches!(
                onesmix,
                Err(MsfdrMixtureFitFailure::DegenerateScoreVariance { .. })
            ) && matches!(
                twosmix,
                Err(MsfdrMixtureFitFailure::DegenerateScoreVariance { .. })
            ),
            "coincident mixtures must be unavailable: 1SMix_returned_model={} 2SMix_returned_model={}",
            onesmix.is_ok(),
            twosmix.is_ok()
        );
    }

    fn gate_rank1_scores() -> Vec<f64> {
        grid(-2.0, 6.0, 100)
    }

    fn gate_s2_scores() -> Vec<f64> {
        grid(-3.0, 2.0, 100)
    }

    fn valid_onesmix_artifact() -> Msfdr1SmixModel {
        Msfdr1SmixModel {
            correct: SkewNormal::new(4.0, 0.7, 0.5),
            incorrect1: SkewNormal::new(0.0, 0.8, -0.2),
            a: 0.4,
        }
    }

    fn valid_twosmix_artifact() -> Msfdr2SmixModel {
        Msfdr2SmixModel {
            correct: SkewNormal::new(4.0, 0.7, 0.5),
            incorrect1: SkewNormal::new(0.0, 0.8, -0.2),
            incorrect2: SkewNormal::new(-2.0, 0.9, 0.1),
            a: 0.4,
            b: 0.1,
        }
    }

    #[test]
    fn statistical_conformance_nonzero_constants_and_nonfinite_inputs_fail_closed() {
        for constant in [0.0, 37.0] {
            let rank1 = vec![constant; 100];
            let pool = vec![constant; 100];
            assert!(matches!(
                Msfdr1SmixModel::fit_rank1_checked(&rank1, 100, 1e-8, (0.01, 0.99), 0.5, 0.2),
                Err(MsfdrMixtureFitFailure::DegenerateScoreVariance { .. })
            ));
            assert!(matches!(
                Msfdr2SmixModel::fit_top_two_pooled_checked(
                    &rank1,
                    &pool,
                    100,
                    1e-8,
                    (0.01, 0.99),
                    0.5,
                    0.2
                ),
                Err(MsfdrMixtureFitFailure::DegenerateScoreVariance { .. })
            ));
        }

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut rank1 = gate_rank1_scores();
            rank1[17] = bad;
            assert!(matches!(
                Msfdr1SmixModel::fit_rank1_checked(&rank1, 100, 1e-8, (0.01, 0.99), 0.5, 0.2),
                Err(MsfdrMixtureFitFailure::NonFiniteInput { index: 17, .. })
            ));

            let mut pool = gate_s2_scores();
            pool[23] = bad;
            assert!(matches!(
                Msfdr2SmixModel::fit_top_two_pooled_checked(
                    &gate_rank1_scores(),
                    &pool,
                    100,
                    1e-8,
                    (0.01, 0.99),
                    0.5,
                    0.2
                ),
                Err(MsfdrMixtureFitFailure::NonFiniteInput { index: 23, .. })
            ));
        }
    }

    #[test]
    fn statistical_conformance_invalid_scales_weights_and_support_fail_closed() {
        let rank1 = gate_rank1_scores();
        let pool = gate_s2_scores();

        for scale in [0.0, f64::NAN, f64::INFINITY] {
            let mut one = valid_onesmix_artifact();
            one.correct.scale = scale;
            assert!(matches!(
                one.validate_for_scores(&rank1),
                Err(MsfdrMixtureFitFailure::NonPositiveScale { .. })
                    | Err(MsfdrMixtureFitFailure::NonFiniteParameter { .. })
            ));

            let mut two = valid_twosmix_artifact();
            two.incorrect2.scale = scale;
            assert!(matches!(
                two.validate_for_scores(&rank1, &pool),
                Err(MsfdrMixtureFitFailure::NonPositiveScale { .. })
                    | Err(MsfdrMixtureFitFailure::NonFiniteParameter { .. })
            ));
        }

        for a in [0.0, 1.0, f64::NAN] {
            let mut one = valid_onesmix_artifact();
            one.a = a;
            assert!(matches!(
                one.validate_for_scores(&rank1),
                Err(MsfdrMixtureFitFailure::InvalidMixtureWeight { .. })
            ));
        }

        let mut ineffective_one = valid_onesmix_artifact();
        ineffective_one.a = 0.02;
        assert!(matches!(
            ineffective_one.validate_for_scores(&rank1),
            Err(MsfdrMixtureFitFailure::IneffectiveComponentSupport { .. })
        ));

        let mut boundary_two = valid_twosmix_artifact();
        boundary_two.b = 0.0;
        assert!(matches!(
            boundary_two.validate_for_scores(&rank1, &pool),
            Err(MsfdrMixtureFitFailure::InvalidMixtureWeight { .. })
        ));

        let mut ineffective_two = valid_twosmix_artifact();
        ineffective_two.a = 0.01;
        ineffective_two.b = 0.97;
        assert!(matches!(
            ineffective_two.validate_for_scores(&rank1, &pool),
            Err(MsfdrMixtureFitFailure::IneffectiveComponentSupport { .. })
        ));
    }

    #[test]
    fn statistical_conformance_coincident_and_numerically_indistinguishable_components_fail() {
        let rank1 = gate_rank1_scores();
        let pool = gate_s2_scores();

        let mut exact_one = valid_onesmix_artifact();
        exact_one.incorrect1 = exact_one.correct.clone();
        assert!(matches!(
            exact_one.validate_for_scores(&rank1),
            Err(MsfdrMixtureFitFailure::CoincidentComponents { .. })
        ));

        let mut numeric_one = valid_onesmix_artifact();
        numeric_one.incorrect1 = numeric_one.correct.clone();
        numeric_one.incorrect1.location += 1e-14;
        assert!(matches!(
            numeric_one.validate_for_scores(&rank1),
            Err(MsfdrMixtureFitFailure::CoincidentComponents { .. })
        ));

        let mut exact_two = valid_twosmix_artifact();
        exact_two.incorrect2 = exact_two.incorrect1.clone();
        assert!(matches!(
            exact_two.validate_for_scores(&rank1, &pool),
            Err(MsfdrMixtureFitFailure::CoincidentComponents { .. })
        ));

        let mut numeric_two = valid_twosmix_artifact();
        numeric_two.incorrect2 = numeric_two.incorrect1.clone();
        numeric_two.incorrect2.shape += 1e-15;
        assert!(matches!(
            numeric_two.validate_for_scores(&rank1, &pool),
            Err(MsfdrMixtureFitFailure::CoincidentComponents { .. })
        ));
    }

    #[test]
    fn statistical_conformance_valid_well_separated_and_low_variance_mixtures_pass() {
        let rank1 = gate_rank1_scores();
        let pool = gate_s2_scores();
        valid_onesmix_artifact()
            .validate_for_scores(&rank1)
            .expect("well-separated 1SMix artifact");
        valid_twosmix_artifact()
            .validate_for_scores(&rank1, &pool)
            .expect("well-separated 2SMix artifact");

        let low_rank1: Vec<f64> = (0..100).map(|i| 1.0 + (i as f64 - 50.0) * 2e-9).collect();
        let low_pool: Vec<f64> = (0..100)
            .map(|i| 1.0 - 6e-8 + (i as f64 - 50.0) * 2e-9)
            .collect();
        let low_one = Msfdr1SmixModel {
            correct: SkewNormal::new(1.0 + 8e-8, 1e-8, 0.25),
            incorrect1: SkewNormal::new(1.0, 1e-8, -0.25),
            a: 0.5,
        };
        low_one
            .validate_for_scores(&low_rank1)
            .expect("representable low-variance 1SMix artifact");
        let low_two = Msfdr2SmixModel {
            correct: SkewNormal::new(1.0 + 12e-8, 1e-8, 0.3),
            incorrect1: SkewNormal::new(1.0 + 6e-8, 1e-8, 0.0),
            incorrect2: SkewNormal::new(1.0, 1e-8, -0.3),
            a: 0.4,
            b: 0.1,
        };
        low_two
            .validate_for_scores(&low_rank1, &low_pool)
            .expect("representable low-variance 2SMix artifact");
    }

    #[test]
    fn statistical_conformance_repeated_fitting_and_label_roles_are_deterministic() {
        let rank1 = synthetic_rank1_scores();
        let pool = synthetic_pool_scores();
        let fit_one = || {
            Msfdr1SmixModel::fit_rank1_checked(&rank1, 100, 1e-8, (0.01, 0.99), 0.5, 0.2)
                .expect("valid 1SMix")
        };
        let first_one = fit_one();
        let second_one = fit_one();
        assert_eq!(first_one.param_tuple(), second_one.param_tuple());

        let mut swapped_one = first_one.clone();
        std::mem::swap(&mut swapped_one.correct, &mut swapped_one.incorrect1);
        swapped_one.a = 1.0 - swapped_one.a;
        let mut sorted = rank1.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let recanonicalized = swapped_one.orient_by_upper_tail(&sorted);
        assert_eq!(first_one.param_tuple(), recanonicalized.param_tuple());

        let fit_two = || {
            Msfdr2SmixModel::fit_top_two_pooled_checked(
                &rank1,
                &pool,
                100,
                1e-8,
                (0.01, 0.99),
                0.5,
                0.2,
            )
            .expect("valid fixed-role 2SMix")
        };
        assert_eq!(fit_two().param_tuple(), fit_two().param_tuple());
    }

    #[test]
    fn statistical_conformance_invalid_fit_has_explicit_serializable_provenance_and_no_model() {
        let fit =
            Msfdr1SmixModel::fit_rank1_checked(&vec![37.0; 100], 100, 1e-8, (0.01, 0.99), 0.5, 0.2);
        let error = fit.expect_err("invalid fit must not expose a probability-producing model");
        assert_eq!(error.code(), "degenerate_score_variance");
        let serialized = serde_json::to_string(&error).expect("failure provenance serializes");
        assert!(serialized.contains("degenerate_score_variance"));
        assert!(!serialized.contains("p_value"));
        assert!(!serialized.contains("pep"));
        assert!(!serialized.contains("q_value"));
    }
}
