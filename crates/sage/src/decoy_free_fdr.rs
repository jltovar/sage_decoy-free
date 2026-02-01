use crate::database::IndexedDatabase;
use crate::input::FdrMode;
use crate::input::{FdrSettings, FdrType, ModelFit};
use crate::lfq::{Peak, PrecursorId};
use crate::ml::nokoi;
use crate::ml::stats;
use crate::scoring::Feature;
use fnv::{FnvHashMap, FnvHashSet};
use rayon::prelude::*;
use statrs::consts::EULER_MASCHERONI;
use statrs::distribution::{Continuous, ContinuousCDF, Gumbel};
use statrs::function::gamma::digamma;
use std::sync::atomic::{AtomicU64, Ordering};

// --- HELPER MATH ---
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };

    // --- SAFETY: clamp to avoid overflow; erf(|x|>=10) is ~1.0 anyway ---
    let abs_x = x.abs().min(10.0);

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x).exp();
    sign * y
}

fn skew_normal_pdf(x: f64, loc: f64, scale: f64, alpha: f64) -> f64 {
    let scale = scale.max(1e-9); // Safety clamp
    let z = (x - loc) / scale;
    let phi = (-(z * z) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let big_phi = 0.5 * (1.0 + erf_approx(alpha * z / std::f64::consts::SQRT_2));
    (2.0 / scale) * phi * big_phi
}

#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    // Handle -inf cases
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }

    // Handle +inf cases explicitly (prevents inf - inf => NaN)
    if a.is_infinite() && a.is_sign_positive() {
        return a;
    }
    if b.is_infinite() && b.is_sign_positive() {
        return b;
    }

    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Silverman's Rule for KDE Bandwidth (adaptive to data std and n)
fn silverman_bw(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    if n < 2.0 {
        return 1.0; // Fallback width for single points to prevent crash
    }
    let sigma = stats::std_dev(samples);
    if sigma == 0.0 {
        return 1.0; // Fallback if all scores are identical
    }
    1.06 * sigma * n.powf(-0.2)
}

/// Isotonic Regression (INCREASING)
/// Ensures P-values increase as Score quality decreases (High Score -> Low P-value)
fn isotonic_regression_increasing(p_values: &mut [f64]) {
    if p_values.is_empty() {
        return;
    }

    // blocks: (value, weight_count)
    // Using usize for weights prevents floating point drift
    let mut blocks: Vec<(f64, usize)> = p_values.iter().map(|&p| (p, 1)).collect();
    let mut i = 0;
    while i < blocks.len() - 1 {
        // Violator: Current P > Next P (should be <=)
        if blocks[i].0 > blocks[i + 1].0 {
            // Merge
            let w_prev = blocks[i].1;
            let w_next = blocks[i + 1].1;
            let w_new = w_prev + w_next;

            // Weighted average of values
            let val_new =
                (blocks[i].0 * w_prev as f64 + blocks[i + 1].0 * w_next as f64) / w_new as f64;

            blocks[i] = (val_new, w_new);
            blocks.remove(i + 1);
            if i > 0 {
                i -= 1;
            }
        } else {
            i += 1;
        }
    }

    // Flatten back
    let mut idx = 0;
    for (val, weight) in blocks {
        for _ in 0..weight {
            if idx < p_values.len() {
                p_values[idx] = val;
                idx += 1;
            }
        }
    }
}

// --- Helper: median and clamp (for LO anchoring) ---
fn median_f64(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        Some(v[n / 2])
    } else {
        let a = v[n / 2 - 1];
        let b = v[n / 2];
        Some((a + b) / 2.0)
    }
}

#[inline]
fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    if !x.is_finite() {
        return lo;
    }
    x.max(lo).min(hi)
}

// --- ROBUST MSFDR MODEL ---
#[derive(Clone, Debug)]
struct RobustMsfdrModel {
    null_loc: f64,
    null_scale: f64,
    target_mean: f64,
    target_std: f64,
    target_alpha: f64,
    pi: f64,
}

impl RobustMsfdrModel {
    pub fn fit(rank1_scores: &[f64], mu_in: f64, beta_in: f64) -> Option<Self> {
        if rank1_scores.len() < 10 {
            return None;
        }

        // 1. Initialization
        let init_null_loc = mu_in;
        let init_null_scale = beta_in.max(1e-6);
        let null_mean_approx = init_null_loc + EULER_MASCHERONI * init_null_scale;

        // Target (Top 20% Heuristic)
        let mut sorted_targets: Vec<f64> = rank1_scores
            .iter()
            .cloned()
            .filter(|x| x.is_finite())
            .collect();

        if sorted_targets.len() < 10 {
            return None;
        }

        sorted_targets.sort_by(|a, b| b.total_cmp(a));

        let top_20 = (sorted_targets.len() as f64 * 0.2) as usize;
        let top_slice = &sorted_targets[0..top_20.max(5).min(sorted_targets.len())];

        let t_mean = top_slice.iter().sum::<f64>() / top_slice.len() as f64;
        let t_var =
            top_slice.iter().map(|s| (s - t_mean).powi(2)).sum::<f64>() / top_slice.len() as f64;
        let t_std = t_var.sqrt().max(1e-6);

        // Pi (Data-Driven Smart Start)
        // IMPORTANT: use the *finite* denominator, otherwise NaNs reduce pi spuriously.
        let n_total_finite = sorted_targets.len().max(1) as f64;
        let n_better = sorted_targets
            .iter()
            .filter(|&&s| s > null_mean_approx)
            .count();
        let init_pi = (n_better as f64 / n_total_finite).clamp(0.05, 0.95);

        let mut params = Self {
            null_loc: init_null_loc,
            null_scale: init_null_scale,
            target_mean: t_mean,
            target_std: t_std,
            target_alpha: 2.0,
            pi: init_pi,
        };

        // 2. EM Loop
        let max_iters = 25;
        let mut old_ll = -f64::INFINITY;

        for iter in 0..max_iters {
            let mut sum_z = 0.0;
            let mut sum_z_x = 0.0;
            let mut sum_z_xx = 0.0;
            let mut new_ll = 0.0;
            let mut n_used = 0usize;

            // --- Guard parameters before constructing the null distribution ---
            if !params.null_loc.is_finite()
                || !params.null_scale.is_finite()
                || params.null_scale <= 0.0
            {
                return None;
            }

            let null_dist = match Gumbel::new(params.null_loc, params.null_scale) {
                Ok(d) => d,
                Err(_) => return None,
            };

            // Clamp pi locally during E-step to prevent transient degeneracy
            let pi = params.pi.clamp(0.01, 0.99);

            // ------------------ E-STEP (log-space; stable) ------------------
            let log_pi = pi.ln();
            let log_1m_pi = (1.0 - pi).ln();

            for &x in rank1_scores {
                if !x.is_finite() {
                    continue;
                }

                // Compute pdfs, floor away from 0 to avoid underflow->drop
                let f0 = null_dist.pdf(x).max(1e-300);
                let f1 = skew_normal_pdf(
                    x,
                    params.target_mean,
                    params.target_std,
                    params.target_alpha,
                )
                .max(1e-300);

                if !f0.is_finite() || !f1.is_finite() {
                    continue;
                }

                let log_f0 = f0.ln();
                let log_f1 = f1.ln();

                // log den = log((1-pi)f0 + pi f1)
                let log_num = log_pi + log_f1;
                let log_den = log_add_exp(log_1m_pi + log_f0, log_num);

                if !log_den.is_finite() {
                    continue;
                }

                // z = exp(log_num - log_den)
                let z = (log_num - log_den).exp();
                if !z.is_finite() {
                    continue;
                }

                sum_z += z;
                sum_z_x += z * x;
                sum_z_xx += z * x * x;

                new_ll += log_den;
                n_used += 1;
            }

            // --- HARD GUARDS AFTER E-STEP ---

            // If no usable points, model cannot be fit
            if n_used < 10 {
                return None;
            }

            // If essentially no posteriors are assigned to the alternative, EM is collapsing.
            // Returning None forces caller to fall back to KDE/other conservative behavior.
            if sum_z < 1e-8 {
                if iter == 0 {
                    return None; // immediate collapse => incompatible
                } else {
                    break; // later collapse => keep last stable params
                }
            }

            // Prevent NaN / Inf propagation
            if !new_ll.is_finite() {
                return None;
            }

            // --- SCALE-INVARIANT CONVERGENCE CHECK ---
            // Use average log-likelihood per point to avoid large-N stagnation
            let avg_ll = new_ll / (n_used as f64);
            if !avg_ll.is_finite() {
                return None;
            }

            // Convergence: relative improvement in average log-likelihood
            // This is stable across dataset sizes and prevents needless iterations.
            let tol_rel = 1e-4;
            let tol_abs = 1e-6;

            if old_ll.is_finite() {
                let delta = (avg_ll - old_ll).abs();
                let scale = old_ll.abs().max(1.0); // avoid division by tiny numbers
                if delta < tol_abs || (delta / scale) < tol_rel {
                    break;
                }
            }

            old_ll = avg_ll;

            // ------------------ M-STEP ------------------

            let n_total = n_used as f64;

            // Update pi (clamped away from 0/1)
            params.pi = (sum_z / n_total).clamp(0.01, 0.99);
            if !params.pi.is_finite() {
                return None;
            }

            // Update target mean
            params.target_mean = sum_z_x / sum_z;
            if !params.target_mean.is_finite() {
                return None;
            }

            // Update target variance
            let var = (sum_z_xx / sum_z) - params.target_mean.powi(2);
            if !var.is_finite() || var < 0.0 {
                return None;
            }

            params.target_std = var.sqrt().max(1e-6);
            if !params.target_std.is_finite() {
                return None;
            }

            // Gentle skew adaptation (keeps tail stable)
            if params.target_mean > t_mean {
                params.target_alpha = (params.target_alpha + 0.1).min(10.0);
            }
        }

        log::info!(
            "Robust MSFDR: pi={:.2}, mean={:.2}",
            params.pi,
            params.target_mean
        );
        Some(params)
    }

    pub fn calculate_pep(&self, x: f64) -> f64 {
        // --- Guard against invalid inputs ---
        if !x.is_finite() {
            return 1.0;
        }

        // --- Guard against invalid Gumbel params ---
        let null_dist = match Gumbel::new(self.null_loc, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 1.0, // Conservative fallback
        };

        // Clamp pi away from exact 0/1 to avoid degeneracy
        let pi = self.pi.clamp(0.01, 0.99);

        let f0 = null_dist.pdf(x).max(1e-300);
        let f1 =
            skew_normal_pdf(x, self.target_mean, self.target_std, self.target_alpha).max(1e-300);

        let den = (1.0 - pi) * f0 + pi * f1;

        // --- Guard against divide-by-zero / NaN ---
        if !den.is_finite() || den <= 0.0 {
            return 1.0;
        }

        // PEP = P(null | x) = (1-pi)*f0 / ((1-pi)*f0 + pi*f1)
        (((1.0 - pi) * f0) / den).clamp(0.0, 1.0)
    }

    // Explicitly named to indicate this uses the Seed parameters (fixed null)
    pub fn calculate_seeded_null_p(&self, x: f64) -> f64 {
        // --- Guard against invalid inputs ---
        if !x.is_finite() {
            return 1.0;
        }

        // --- Guard against invalid Gumbel params ---
        let null_dist = match Gumbel::new(self.null_loc, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 1.0, // Conservative fallback (P=1.0)
        };

        // Avoid exact 0.0 p-values (later log / combination safety)
        null_dist.sf(x).clamp(0.0, 1.0).max(1e-300)
    }
}

// --- MAIN FUNCTION ---
/// Calculate spectrum-level q-values using Gumbel-based decoy-free methods.
pub fn calculate_q_values(psms: &[Feature], settings: &FdrSettings) -> Vec<Feature> {
    let mut new_features = psms.to_vec();
    let min_rank = settings.min_null_rank;
    let max_rank = settings.max_null_rank;
    let fit_method = settings.model_fit.clone();
    let min_null_size = settings.min_null_size;
    let min_storey_n = settings.min_storey_n;

    log::info!(
        "Building null distribution [Rank {}..={}] using {:?} fit...",
        min_rank,
        max_rank,
        fit_method
    );

    // --- PHASE 1: SOFT PURIFIED NULL ---
    // 1. Identify "High Confidence" Rank 1 peptides to exclude from Null
    let mut rank1_scores: Vec<(u32, f64)> = new_features
        .iter()
        .filter(|f| f.rank == 1)
        .map(|f| (f.peptide_idx.0, f.hyperscore as f64))
        .filter(|(_, s)| s.is_finite())
        .collect();

    // --- FIX: use TOP 20% threshold (not median) ---
    // We want to exclude only very high-confidence rank1 peptides from the null.
    // Using the median would exclude ~50% and can collapse the null.
    let purification_threshold = if rank1_scores.len() >= 10 {
        // Sort by score descending
        rank1_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Use the materialized setting from your new input.rs
        let p_factor = settings.purification_factor;

        let top_k = ((rank1_scores.len() as f64) * p_factor).round() as usize;
        let top_k = top_k.max(5).min(rank1_scores.len());

        // The score at this index becomes the cutoff for "too good to be null"
        rank1_scores[top_k - 1].1
    } else {
        1000.0
    };

    let purified_peptides: FnvHashSet<u32> = rank1_scores
        .iter()
        .filter(|(_, score)| *score >= purification_threshold)
        .map(|(idx, _)| *idx)
        .collect();

    // 2. Build Null Scores (Try Purified First)
    let mut fit_data: Vec<(u32, f64)> = new_features
        .iter()
        .filter_map(|psm| {
            if psm.rank >= min_rank && psm.rank <= max_rank {
                if purified_peptides.contains(&psm.peptide_idx.0) {
                    return None;
                }
                let s = psm.hyperscore as f64;
                if !s.is_finite() {
                    return None;
                }
                Some((psm.rank, s))
            } else {
                None
            }
        })
        .collect();

    // 3. Fallback
    if fit_data.len() < min_null_size {
        log::warn!("Purified null too small, falling back to unpurified null.");
        fit_data = new_features
            .iter()
            .filter_map(|psm| {
                if psm.rank >= min_rank && psm.rank <= max_rank {
                    let s = psm.hyperscore as f64;
                    if !s.is_finite() {
                        return None;
                    }
                    Some((psm.rank, s))
                } else {
                    None
                }
            })
            .collect();

        if fit_data.len() < min_null_size {
            log::error!("Null distribution too small. Aborting FDR.");
            new_features.par_iter_mut().for_each(|psm| {
                psm.spectrum_q = 1.0;
                if psm.rank == 1 {
                    psm.decoy_free_p_value = Some(1.0);
                    psm.decoy_free_pep = Some(1.0);
                    psm.decoy_free_score = Some(0.0);
                    psm.decoy_free_q_value = Some(1.0);
                } else {
                    psm.decoy_free_p_value = None;
                    psm.decoy_free_pep = None;
                    psm.decoy_free_score = None;
                    psm.decoy_free_q_value = None;
                }
            });
            return new_features;
        }
    }

    // --- SAFETY: filter non-finite scores in fit_data (protect LO regression) ---
    let fit_data_before = fit_data.len();
    fit_data.retain(|&(_, s)| s.is_finite());
    let fit_data_dropped = fit_data_before - fit_data.len();
    if fit_data_dropped > 0 {
        log::warn!(
            "Dropped {} non-finite entries from fit_data before LO regression (fit_data {}).",
            fit_data_dropped,
            fit_data_before
        );
    }

    // If filtering made the null too small, fail-closed.
    if fit_data.len() < min_null_size {
        log::error!(
            "Null distribution too small after filtering non-finite fit_data ({} < {}). Aborting FDR.",
            fit_data.len(),
            min_null_size
        );
        new_features.par_iter_mut().for_each(|psm| {
            psm.spectrum_q = 1.0;
            if psm.rank == 1 {
                psm.decoy_free_p_value = Some(1.0);
                psm.decoy_free_pep = Some(1.0);
                psm.decoy_free_score = Some(0.0);
                psm.decoy_free_q_value = Some(1.0);
            } else {
                psm.decoy_free_p_value = None;
                psm.decoy_free_pep = None;
                psm.decoy_free_score = None;
                psm.decoy_free_q_value = None;
            }
        });
        return new_features;
    }

    // --- SAFETY: filter non-finite scores before fitting ---
    let scores: Vec<f64> = fit_data
        .iter()
        .map(|(_, s)| *s)
        .filter(|s| s.is_finite())
        .collect();

    let dropped = fit_data.len() - scores.len();
    if dropped > 0 {
        log::warn!(
            "Dropped {} non-finite null scores before fitting (fit_data {}).",
            dropped,
            fit_data.len()
        );
    }

    // If filtering made the null too small, fail-closed.
    if scores.len() < min_null_size {
        log::error!(
            "Null distribution too small after filtering non-finite scores ({} < {}). Aborting FDR.",
            scores.len(),
            min_null_size
        );
        new_features.par_iter_mut().for_each(|psm| {
            psm.spectrum_q = 1.0;
            if psm.rank == 1 {
                psm.decoy_free_p_value = Some(1.0);
                psm.decoy_free_pep = Some(1.0);
                psm.decoy_free_score = Some(0.0);
                psm.decoy_free_q_value = Some(1.0);
            } else {
                psm.decoy_free_p_value = None;
                psm.decoy_free_pep = None;
                psm.decoy_free_score = None;
                psm.decoy_free_q_value = None;
            }
        });
        return new_features;
    }

    // Calculate Global Fallback n_eff
    let total_candidates: u64 = new_features
        .iter()
        .filter(|f| f.rank == 1 && (f.hyperscore as f64).is_finite())
        .map(|f| f.scored_candidates as u64)
        .sum();

    let num_spectra = new_features
        .iter()
        .filter(|f| f.rank == 1 && (f.hyperscore as f64).is_finite())
        .count()
        .max(1) as f64;

    let n_eff_global = (total_candidates as f64 / num_spectra).max(2.0);

    log::info!("Global Effective Search Space (n_eff): {:.1}", n_eff_global);

    // --- NEW: LO anchoring reference (n_global / n_ref) ---
    // IMPORTANT:
    // - Our LO regression is fit in digamma-space: Score ~ intercept + beta * (-digamma(rank))
    // - Therefore the fitted intercept already corresponds to the *typical* multiplicity seen in the fit dataset.
    // - We should NOT back-calculate an absolute μ with (ln(n)+gamma) here (that mixes coordinate systems).
    // - Instead, we anchor to a reference n_global (median scored_candidates of rank-1 spectra),
    //   and apply ONLY a relative shift in ln-space: +beta*(ln(n_local)-ln(n_global)).
    //
    // REFINEMENT (Multiplicity attenuation):
    // - In open-search contexts, "scored_candidates" are highly correlated (same peptide with small mass shifts),
    //   so ln(n_local/n_global) can over-penalize if treated as independent trials.
    // - We therefore support a damping exponent alpha in [0,1]:
    //       shift = beta * alpha * ln(n_local/n_global)
    //   alpha=1.0 => original (full multiplicity penalty), alpha<1.0 => attenuated penalty.
    // --- NEW: geometric reference for n_global ---
    // n_global = exp(median(log(n_i))) with n_i clamped and filtered
    let log_n_global_vec: Vec<f64> = new_features
        .iter()
        .filter(|f| f.rank == 1 && (f.hyperscore as f64).is_finite())
        .filter_map(|f| {
            let n = f.scored_candidates as f64;
            if !n.is_finite() || n < 2.0 {
                return None;
            }

            // Clamp BEFORE log to avoid log(0), log(inf), extreme leverage
            let n_clamped = n.max(10.0).min(1e7);
            let ln = n_clamped.ln();
            if ln.is_finite() {
                Some(ln)
            } else {
                None
            }
        })
        .collect();

    let n_global = if let Some(med_ln) = median_f64(log_n_global_vec) {
        med_ln.exp()
    } else {
        // fallback stays coherent with your existing behavior
        clamp_f64(n_eff_global, 10.0, 1e7)
    };

    let n_global = clamp_f64(n_global, 10.0, 1e7);

    log::info!(
        "LO reference search space (n_global, geometric median rank-1 scored_candidates): {:.1}",
        n_global
    );

    // --- SPLIT EXECUTION FLAGS ---
    let debug_mode = matches!(fit_method, ModelFit::EnsembleDebug);
    let run_parametric = matches!(
        fit_method,
        ModelFit::Ensemble | ModelFit::EnsembleDebug | ModelFit::Nokoi
    );
    let run_nokoi = matches!(fit_method, ModelFit::EnsembleDebug | ModelFit::Nokoi);

    // --- OUTPUT MAPPING MODE ---
    // map_to_standard_output controls ownership of standard Sage columns.
    // Only DecoyFree mode may write posterior_error / discriminant_score.
    // TDC mode must leave these untouched.
    let map_to_standard_output = matches!(settings.mode, FdrMode::DecoyFree);

    debug_assert!(
        !(settings.mode == FdrMode::Tdc && map_to_standard_output),
        "TDC must never map decoy-free stats into standard columns"
    );

    // 1. Fit Moments
    let (mu_mom, beta_mom) = fit_gumbel_moments(&scores);

    // --- SAFETY: explicit validity check for fitted Moments params ---
    // We do NOT rely on Gumbel::new(...) to detect NaN/Inf consistently.
    let moments_params_ok = mu_mom.is_finite() && beta_mom.is_finite() && beta_mom > 0.0;
    if !moments_params_ok {
        log::warn!(
            "Moments fit returned invalid params (mu={}, beta={}). Marking moments invalid.",
            mu_mom,
            beta_mom
        );
    }

    // 2. Fit Lower Order
    // --- SAFETY: LO fallback MUST NOT use invalid Moments params ---
    // If LO regression fails and Moments is invalid, use a conservative last-ditch fallback.
    //
    // NOTE ON FALLBACK:
    // In digamma LO, an "intercept" is an anchored quantity. If we must synthesize it from Moments,
    // we do so coherently by anchoring the intercept at n_global:
    //   intercept ≈ μ_mom + β_mom * ln(n_global)
    // (We do NOT use +EULER_MASCHERONI here, because that is part of the digamma/Gumbel order-statistic
    // mapping that should not be mixed into the intercept after the fact.)
    let min_count = settings.min_rank_count;

    let (lo_intercept_raw, lo_beta_raw) = if matches!(fit_method, ModelFit::LowerOrder)
        || run_parametric
    {
        // UPDATED: Now passing min_count to the regression function
        fit_lower_order_regression(&fit_data, min_rank, max_rank, min_count).unwrap_or_else(|| {
            if moments_params_ok {
                let intercept = mu_mom + beta_mom * n_global.ln();
                (intercept, beta_mom)
            } else {
                // Last-ditch fallback (safe but not statistically meaningful)
                (0.0, 1.0)
            }
        })
    } else {
        if moments_params_ok {
            let intercept = mu_mom + beta_mom * n_global.ln();
            (intercept, beta_mom)
        } else {
            // Last-ditch fallback (safe but not statistically meaningful)
            (0.0, 1.0)
        }
    };

    // --- Harden LO parameters immediately ---
    // Ensure finite and positive before using it in any calculations
    let lo_beta = if lo_beta_raw.is_finite() && lo_beta_raw > 0.0 {
        lo_beta_raw
    } else if moments_params_ok {
        beta_mom.max(1e-9)
    } else {
        1.0
    };

    // Intercept also needs hardening (can become NaN if fallback path was NaN before)
    let lo_intercept = if lo_intercept_raw.is_finite() {
        lo_intercept_raw
    } else {
        0.0
    };

    // --- OPTIONAL ROBUSTNESS: shrink LO beta toward Moments beta ---
    // This reduces parametric fragility in heavy tails without changing the LO functional form.
    //
    // REFINEMENT:
    // - We make the blending weight configurable (default: 0.30 toward Moments).
    // - Some datasets benefit from stronger shrinkage (e.g., 0.50) when LO slope is tail-noisy.
    //
    // If settings.lo_beta_blend_moments is present, it should be in [0,1] and represents the fraction of beta_mom.
    let w_mom = settings.lo_beta_blend_moments.clamp(0.0, 1.0);
    let lo_beta_shrunk = if moments_params_ok {
        ((1.0 - w_mom) * lo_beta) + (w_mom * beta_mom.max(1e-9))
    } else {
        lo_beta
    };

    log::info!(
        "LO anchor: intercept={:.4}, beta_raw={:.4}, beta_shrunk={:.4}, n_global={:.1}, w_mom={:.2}",
        lo_intercept,
        lo_beta,
        lo_beta_shrunk,
        n_global,
        w_mom
    );

    // 3. Fit MLE
    // --- SAFETY: If Moments invalid, do not "fallback" MLE to Moments (would be NaN/NaN) ---
    let (mu_mle, beta_mle) = if matches!(fit_method, ModelFit::Mle) || run_parametric {
        fit_gumbel_mle(&scores).unwrap_or_else(|| {
            if moments_params_ok {
                (mu_mom, beta_mom)
            } else {
                // Safe fallback only (will not be used if fail-closed triggers)
                (0.0, 1.0)
            }
        })
    } else {
        if moments_params_ok {
            (mu_mom, beta_mom)
        } else {
            (0.0, 1.0)
        }
    };

    let beta_mle = if beta_mle.is_finite() && beta_mle > 0.0 {
        beta_mle
    } else if moments_params_ok {
        beta_mom.max(1e-9)
    } else {
        1.0
    };

    let mu_mle = if mu_mle.is_finite() { mu_mle } else { 0.0 };

    // 4. Fit Robust MSFDR
    // MSFDR needs a single starting mu. Use the global average n_eff (seeded null).
    //
    // IMPORTANT (Consistency):
    // - LO intercept is anchored at n_global.
    // - Therefore the consistent global seeded μ is:
    //     μ_seed = intercept + β * (ln(n_eff_global) - ln(n_global))
    // - We do NOT use μ = intercept - β*(ln(n)+gamma) here; that is the coordinate mismatch bug.
    let beta_lo_seed = lo_beta_shrunk.max(1e-9);
    let mu_lo_global = lo_intercept + beta_lo_seed * (n_eff_global.ln() - n_global.ln());

    let msfdr_model = if matches!(fit_method, ModelFit::Msfdr) || run_parametric {
        let (mu_in, beta_in) = if run_parametric {
            (mu_lo_global, beta_lo_seed)
        } else {
            (mu_mom, beta_mom)
        };
        let target_scores: Vec<f64> = new_features
            .iter()
            .filter(|f| f.rank == 1)
            .map(|f| f.hyperscore as f64)
            .filter(|x| x.is_finite())
            .collect();
        log::info!("Fitting Robust MSFDR mixture model...");
        RobustMsfdrModel::fit(&target_scores, mu_in, beta_in)
    } else {
        None
    };

    // --- Fail-Closed Logic for Null Distributions ---
    // --- SAFETY: Moments validity is driven by explicit param check (not constructor behavior) ---
    let moments_valid = moments_params_ok;

    // Build Moments distribution ONLY if valid (otherwise dummy for type-checker; not used)
    let dist_mom = if moments_valid {
        Gumbel::new(mu_mom, beta_mom).unwrap()
    } else {
        log::warn!(
            "Moments fit yielded invalid params (mu={}, beta={}). FDR will fail closed.",
            mu_mom,
            beta_mom
        );
        Gumbel::new(0.0, 1.0).unwrap()
    };

    // Guard MLE (fallback to Moments if MLE fails)
    let dist_mle = match Gumbel::new(mu_mle, beta_mle) {
        Ok(d) => d,
        Err(_) => {
            if moments_valid {
                Gumbel::new(mu_mom, beta_mom).unwrap()
            } else {
                // dummy; will not be used because we fail-closed immediately after
                Gumbel::new(0.0, 1.0).unwrap()
            }
        }
    };

    // --- Enforce Fail-Closed ---
    if !moments_valid {
        log::error!("Invalid null fit (Moments). FDR will fail closed (all q=1).");
        new_features.par_iter_mut().for_each(|psm| {
            psm.spectrum_q = 1.0;
            if psm.rank == 1 {
                psm.decoy_free_p_value = Some(1.0);
                psm.decoy_free_pep = Some(1.0);
                psm.decoy_free_score = Some(0.0);
                psm.decoy_free_q_value = Some(1.0);
            } else {
                psm.decoy_free_p_value = None;
                psm.decoy_free_pep = None;
                psm.decoy_free_score = None;
                psm.decoy_free_q_value = None;
            }
        });
        return new_features;
    }

    // KDE Setup
    let kde_limit = if settings.kde_samples > 0 {
        settings.kde_samples
    } else {
        20_000
    };
    let mut target_scores_kde: Vec<f64> = new_features
        .iter()
        .filter(|f| f.rank == 1)
        .map(|f| f.hyperscore as f64)
        .filter(|x| x.is_finite())
        .collect();

    if target_scores_kde.len() > kde_limit {
        let step = (target_scores_kde.len() / kde_limit).max(1); // --- SAFETY: step_by(0) protection ---
        target_scores_kde = target_scores_kde.into_iter().step_by(step).collect();
    }
    let bandwidth = silverman_bw(&target_scores_kde).max(1e-9);

    // Guard against empty KDE data
    if target_scores_kde.is_empty() {
        log::warn!("No rank-1 scores available for KDE; PEP will default to 1.0.");
    }

    // Conservative KDE mixture weight if MSFDR is not available
    // We prefer a conservative null weight so PEP does not become overly optimistic.
    let pi0_kde = 0.90_f64;

    // --- NEW: multiplicity attenuation controls (synergistic fix) ---
    //
    // alpha in [0,1] attenuates the ln(n_local/n_global) shift:
    //   alpha=1.0 => full multiplicity penalty
    //   alpha=0.5 => square-root-like damping (reasonable open-search default)
    //
    // ln_ratio_cap bounds extreme multiplicity shifts so one pathological spectrum cannot dominate.
    //
    // These default conservatively and can be tuned with entrapment calibration.
    // --- DYNAMIC ALPHA IMPLEMENTATION ---
    let n_rank1_est = new_features.iter().filter(|f| f.rank == 1).count();
    let low_input_scaling = if n_rank1_est < 1000 {
        // Smoothly scale alpha down for low-input samples
        // to be less punishing when competition is low.
        0.75
    } else {
        1.0
    };

    let lo_alpha = (settings.lo_multiplicity_alpha.clamp(0.0, 1.0)) * low_input_scaling;
    let lo_ln_ratio_cap = settings.lo_ln_ratio_cap.max(0.0);

    log::info!(
        "LO multiplicity attenuation: alpha={:.2}, ln_ratio_cap={:.2}",
        lo_alpha,
        lo_ln_ratio_cap
    );

    // --- LO saturation diagnostics counters (thread-safe) ---
    let clipped_ln = AtomicU64::new(0);
    let capped_beta = AtomicU64::new(0);
    let n_rank1 = AtomicU64::new(0);

    // --- CALCULATION LOOP ---
    new_features.par_iter_mut().for_each(|psm| {
        // IMPORTANT:
        // Do NOT blank these here. In decoy-free mode we will map decoy-free stats into these
        // fields for standard Sage output compatibility. In TDC paths, these may already be set.

        if psm.rank == 1 {
            let x = psm.hyperscore as f64;

            // --- SAFETY: hyperscore must be finite ---
            if !x.is_finite() {
                psm.decoy_free_p_value = Some(1.0);
                psm.decoy_free_pep = Some(1.0);

                // Map to standard output columns too (fail-closed)
                if map_to_standard_output {
                    psm.posterior_error = 1.0;
                    psm.discriminant_score = 0.0;
                }

                psm.spectrum_q = 1.0;
                psm.decoy_free_score = Some(0.0);

                if debug_mode {
                    psm.p_moments = Some(1.0_f32);
                    psm.p_mle = Some(1.0_f32);
                    psm.p_lower_order = Some(1.0_f32);
                    psm.p_msfdr = None;
                }
                return;
            }

            // Static models
            let p_mom = dist_mom.sf(x).clamp(0.0, 1.0).max(1e-300);
            let p_mle = dist_mle.sf(x).clamp(0.0, 1.0).max(1e-300);

            // Dynamic Lower Order
            let n_eff = if psm.scored_candidates >= 2 {
                (psm.scored_candidates as f64).max(2.0).min(1e9)
            } else {
                n_eff_global
            };

            // --- CORE FIX: LO dynamic mapping via RELATIVE SHIFT (consistent with digamma fit) ---
            // We treat the fitted LO intercept as an anchored score level corresponding to n_global.
            // For each spectrum, we shift only by the deviation in search-space complexity:
            //   mu_local = intercept + beta * (ln(n_local) - ln(n_global))
            //
            // REFINEMENT (Multiplicity attenuation):
            // In open-search, candidates are correlated, so ln(n_local/n_global) can over-penalize.
            // We therefore attenuate the shift by alpha in [0,1]:
            //   mu_local = intercept + beta * alpha * clamp(ln(n_local/n_global), +/- ln_ratio_cap)
            //
            // This:
            //   - preserves the correct direction of multiplicity adjustment,
            //   - prevents “0 discoveries” regimes caused by treating correlated candidates as independent,
            //   - keeps behavior tunable and stable.
            // Reference beta (your "βrank"/βref). Pick the right one—see below.
            let beta_ref = beta_mom.max(1e-9);

            let safety_mult =
                if settings.lo_beta_safety_mult.is_finite() && settings.lo_beta_safety_mult > 0.0 {
                    settings.lo_beta_safety_mult
                } else {
                    0.60
                };

            // Safety belt: keep LO beta in a sane range relative to reference beta.
            let beta_cap = (safety_mult * beta_ref).max(1e-9);

            // Enforce 0 ≤ beta_lo ≤ beta_cap (and keep >0 for Gumbel validity).
            let beta_lo = lo_beta_shrunk.clamp(1e-9, beta_cap);
            let mut ln_ratio = n_eff.ln() - n_global.ln();
            let ln_ratio_raw = ln_ratio; // optional but helpful for exact “did clamp happen?”

            if lo_ln_ratio_cap > 0.0 {
                ln_ratio = ln_ratio.clamp(-lo_ln_ratio_cap, lo_ln_ratio_cap);
            }

            // --- LO saturation diagnostics ---
            n_rank1.fetch_add(1, Ordering::Relaxed);

            if lo_ln_ratio_cap > 0.0 && (ln_ratio != ln_ratio_raw) {
                clipped_ln.fetch_add(1, Ordering::Relaxed);
            }

            // Count how often beta_lo is at the cap (within eps)
            if beta_lo >= beta_cap * (1.0 - 1e-12) {
                capped_beta.fetch_add(1, Ordering::Relaxed);
            }

            let mu_i = lo_intercept + beta_lo * lo_alpha * ln_ratio;

            // --- Guard against Unwrap Panic in Dynamic LO ---
            let dist_lo_dynamic = Gumbel::new(mu_i, beta_lo).unwrap_or_else(|_| dist_mom.clone());
            let p_lo = dist_lo_dynamic.sf(x).clamp(0.0, 1.0).max(1e-300);

            let p_final = match fit_method {
                ModelFit::Moments => p_mom,

                // MSFDR mode should use seeded null P-value, not Moments
                ModelFit::Msfdr => {
                    if let Some(m) = &msfdr_model {
                        m.calculate_seeded_null_p(x)
                    } else {
                        p_mom
                    }
                }

                ModelFit::Mle => p_mle,
                ModelFit::LowerOrder => p_lo,

                // Ensemble
                ModelFit::Ensemble | ModelFit::EnsembleDebug | ModelFit::Nokoi => {
                    // Ensemble excludes MSFDR p-value to avoid double-counting seeded null
                    //
                    // --- Guard against "LO hijacking" ---
                    // HMP is min-like; a single overly-liberal p-value can dominate.
                    // If LO is >100x smaller than BOTH Moments and MLE, cap it at the best of those two.
                    let p_base_min = p_mom.min(p_mle);

                    // Detect whether LO is operating in a "saturated" regime for this spectrum:
                    // - ln_ratio got clipped (extreme multiplicity shift)
                    // - or beta hit the safety cap
                    let ln_was_clipped =
                        lo_ln_ratio_cap > 0.0 && (ln_ratio.abs() >= lo_ln_ratio_cap - 1e-12);

                    let beta_was_capped = beta_lo >= beta_cap - 1e-12;

                    let lo_saturated = ln_was_clipped || beta_was_capped;

                    // Soft guard:
                    // - If LO is massively smaller than the conservative base AND LO is saturated,
                    //   treat LO as unreliable and fall back to the base.
                    // - Otherwise, allow LO to contribute (this is the main sensitivity unlock).
                    let p_lo_guarded = if lo_saturated && (p_lo < (p_base_min / 1000.0)) {
                        p_base_min
                    } else {
                        p_lo
                    };

                    stats::combine_hmp(&[p_mom, p_mle, p_lo_guarded])
                }
            };

            // Safety Clamp P-value [0, 1] and avoid exact 0
            let p_final = p_final.clamp(0.0, 1.0).max(1e-300);

            // Calculate PEP
            // DESIGN CHOICE: We intentionally prefer the MSFDR mixture model for PEP
            // if it fits successfully, even if the primary P-value comes from another model.
            let pep_raw = if let Some(model) = &msfdr_model {
                model.calculate_pep(x)
            } else if !target_scores_kde.is_empty() {
                // KDE fallback: approximate mixture posterior with conservative pi0
                let f0 = match fit_method {
                    ModelFit::Mle => dist_mle.pdf(x).max(1e-300),
                    ModelFit::LowerOrder => dist_lo_dynamic.pdf(x).max(1e-300),
                    _ => dist_mom.pdf(x).max(1e-300),
                };
                let f1 = kde_density(x, &target_scores_kde, bandwidth).max(1e-300);

                let den = pi0_kde * f0 + (1.0 - pi0_kde) * f1;
                if !den.is_finite() || den <= 0.0 {
                    1.0
                } else {
                    (pi0_kde * f0 / den).clamp(0.0, 1.0)
                }
            } else {
                1.0
            };

            // --- Safety Clamp PEP to ensure finite range ---
            let pep = if pep_raw.is_finite() {
                pep_raw.clamp(0.0, 1.0)
            } else {
                1.0
            };

            psm.decoy_free_p_value = Some(p_final as f32);
            psm.decoy_free_pep = Some(pep as f32);
            psm.spectrum_q = p_final as f32;

            let safe_pep = pep.max(1e-15);
            let df_score = (-10.0 * safe_pep.log10()) as f32;
            psm.decoy_free_score = Some(df_score);

            // Only map into standard columns in decoy-free mode
            if map_to_standard_output {
                // Standard Sage TSV columns:
                //   - posterior_error => decoy-free PEP
                //   - sage_discriminant_score => -10 * log10(PEP)
                psm.posterior_error = pep as f32;
                psm.discriminant_score = df_score;
            }

            if debug_mode {
                psm.p_moments = Some(p_mom as f32);
                psm.p_mle = Some(p_mle as f32);
                psm.p_lower_order = Some(p_lo as f32);
                if let Some(m) = &msfdr_model {
                    psm.p_msfdr = Some(m.calculate_seeded_null_p(x) as f32);
                }
            }
        } else {
            psm.spectrum_q = 1.0;
            psm.decoy_free_p_value = None;
            psm.decoy_free_pep = None;
            psm.decoy_free_score = None;
            psm.decoy_free_q_value = None;

            if map_to_standard_output {
                psm.posterior_error = 1.0;
                psm.discriminant_score = 0.0;
            }
        }
    });

    // Apply PAVA to enforce monotonicity only when NOT running pure LowerOrder mode.
    // But: q-values must ALWAYS be computed, regardless of PAVA.
    // --- PAVA SOFT GUARD IMPLEMENTATION ---
    // We skip PAVA if we are in pure LowerOrder mode OR if the user
    // explicitly wants to trust the raw dynamic nulls in an Ensemble.
    let apply_pava = match fit_method {
        ModelFit::LowerOrder => false,
        // If you want Ensemble to also skip PAVA for ultra-low input testing:
        ModelFit::Ensemble | ModelFit::EnsembleDebug => {
            if n_rank1_est < 500 {
                false
            } else {
                true
            }
        }
        _ => true,
    };

    // --- Collect Rank-1 indices and p-values (always) ---
    let mut rank1_indices = Vec::with_capacity(new_features.len() / 2);
    let mut rank1_scores_for_pava = Vec::with_capacity(new_features.len() / 2);
    let mut rank1_pvalues = Vec::with_capacity(new_features.len() / 2);

    for (i, psm) in new_features.iter().enumerate() {
        if psm.rank == 1 {
            let s = psm.hyperscore as f64;
            if s.is_finite() {
                rank1_indices.push(i);
                rank1_scores_for_pava.push(s);
                rank1_pvalues.push(psm.spectrum_q as f64); // currently holds p_final from loop
            }
        }
    }

    // --- Guard against empty Rank-1 set (always) ---
    if rank1_indices.is_empty() {
        log::warn!("No finite rank-1 hyperscores found; failing closed (all q=1).");
        new_features.par_iter_mut().for_each(|psm| {
            psm.spectrum_q = 1.0;
            if psm.rank == 1 {
                psm.decoy_free_p_value = Some(1.0);
                psm.decoy_free_pep = Some(1.0);
                psm.decoy_free_score = Some(0.0);
                psm.decoy_free_q_value = Some(1.0);
            } else {
                psm.decoy_free_p_value = None;
                psm.decoy_free_pep = None;
                psm.decoy_free_score = None;
                psm.decoy_free_q_value = None;
            }
        });
        return new_features;
    }

    // --- LO SATURATION DIAGNOSTICS ---
    // These counters measure how often the stabilizers are actively constraining the LO model.
    //
    // Interpretation:
    //   frac_ln_clipped  = fraction of rank-1 spectra whose ln(n_i / n_global) shift hit the cap
    //                      → high values mean multiplicity correction is saturating frequently
    //                      → suggests ln_ratio_cap is too small or search-space variability is extreme
    //
    //   frac_beta_capped = fraction of spectra where beta_lo was clamped to safety_mult * beta_ref
    //                      → high values mean the LO regression slope is too aggressive
    //                      → indicates tail regression instability or insufficient shrinkage
    //
    // In a healthy, well-behaved LO model:
    //   • frac_ln_clipped  should typically be < 0.10–0.20
    //   • frac_beta_capped should typically be < 0.05–0.10
    //
    // Values significantly above these ranges indicate the model is operating in a constrained regime
    // and that calibration is being driven by guards rather than the statistical fit itself.

    let n_seen = n_rank1.load(Ordering::Relaxed);
    if n_seen > 0 {
        let n = n_seen as f64;
        let frac_ln = clipped_ln.load(Ordering::Relaxed) as f64 / n;
        let frac_beta = capped_beta.load(Ordering::Relaxed) as f64 / n;

        log::info!(
            "LO saturation diagnostics: n_rank1={}, frac_ln_clipped={:.3}, frac_beta_capped={:.3}",
            n_seen,
            frac_ln,
            frac_beta
        );
    } else {
        // This should only occur if no rank-1 spectra were processed in the scoring loop
        // (e.g., pathological dataset or early failure paths).
        log::info!("LO saturation diagnostics: n_rank1=0 (no rank-1 spectra processed)");
    }

    // --- PAVA CALIBRATION ---
    // Assumes inputs are sorted by score descending; enforces p-values non-decreasing along that order.
    if apply_pava {
        // Sort by score descending (high score = should be low p)
        let mut pava_data: Vec<(f64, usize, f64)> = rank1_scores_for_pava
            .iter()
            .zip(rank1_indices.iter())
            .zip(rank1_pvalues.iter())
            .map(|((&s, &idx), &p)| (s, idx, p))
            .collect();

        pava_data.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        let mut sorted_pvalues: Vec<f64> = pava_data.iter().map(|&(_, _, p)| p).collect();

        // Sanitize input to PAVA
        for p in &mut sorted_pvalues {
            if !p.is_finite() {
                *p = 1.0;
            } else {
                *p = p.clamp(0.0, 1.0).max(1e-300);
            }
        }

        isotonic_regression_increasing(&mut sorted_pvalues);

        // Write calibrated p-values back to the corresponding features and rank1_pvalues vector
        for (i, &(_, idx, _)) in pava_data.iter().enumerate() {
            let cal_p = sorted_pvalues[i].clamp(0.0, 1.0).max(1e-300);
            new_features[idx].spectrum_q = cal_p as f32;
            new_features[idx].decoy_free_p_value = Some(cal_p as f32);
        }

        // Refresh rank1_pvalues from updated spectrum_q after PAVA
        for (k, &idx) in rank1_indices.iter().enumerate() {
            rank1_pvalues[k] = new_features[idx].spectrum_q as f64;
        }
    }

    // --- Compute q-values (always) ---
    let q_values = match settings.type_ {
        FdrType::Storey => stats::storey_q_value(&rank1_pvalues, min_storey_n),
        FdrType::Bh => stats::bh_q_value(&rank1_pvalues),
    };

    for (idx, q) in rank1_indices.into_iter().zip(q_values) {
        let feat = &mut new_features[idx];
        feat.spectrum_q = q as f32;
        feat.decoy_free_q_value = Some(q as f32);
    }

    // --- NOKOI RESCORING ---
    if run_nokoi {
        log::info!("Running Nokoi Rescoring...");
        if let Some(probs) = nokoi::rescore(&new_features, 0.01) {
            let nokoi_p_values = nokoi::calc_empirical_p_values(&new_features, &probs);

            // A) Independent Nokoi Q-values
            let nokoi_rank1_p: Vec<f64> = new_features
                .iter()
                .zip(&nokoi_p_values)
                .filter(|(f, _)| f.rank == 1)
                .map(|(_, &p)| p)
                .collect();

            let nokoi_rank1_q = match settings.type_ {
                FdrType::Storey => stats::storey_q_value(&nokoi_rank1_p, min_storey_n),
                FdrType::Bh => stats::bh_q_value(&nokoi_rank1_p),
            };

            // B) Combine & Assign
            let mut final_p_values = Vec::new();
            let mut final_indices = Vec::new();
            let mut q_iter = nokoi_rank1_q.into_iter();

            for (i, feat) in new_features.iter_mut().enumerate() {
                if feat.rank == 1 {
                    let nokoi_p = nokoi_p_values[i];

                    if debug_mode {
                        feat.p_nokoi = Some(nokoi_p as f32);
                        if let Some(q) = q_iter.next() {
                            feat.q_nokoi = Some(q as f32);
                        }
                    } else {
                        q_iter.next();
                    }

                    // Ensure this uses seeded null if you want consistency, or just Moments as base.
                    // Sticking to Moments as the base for Nokoi HMP is safe and standard.
                    let old_p = feat.decoy_free_p_value.unwrap_or(1.0) as f64;
                    let final_p = stats::combine_hmp(&[old_p, nokoi_p]);

                    feat.decoy_free_p_value = Some(final_p as f32);
                    feat.spectrum_q = final_p as f32;

                    let proxy_pep = final_p.max(1e-15);
                    let df_score = (-10.0 * proxy_pep.log10()) as f32;
                    feat.decoy_free_score = Some(df_score);

                    if map_to_standard_output {
                        feat.posterior_error = proxy_pep as f32;
                        feat.discriminant_score = df_score;
                    }

                    final_p_values.push(final_p);
                    final_indices.push(i);
                }
            }

            // Re-calc Q-values on the Super Ensemble
            let final_qs = match settings.type_ {
                FdrType::Storey => stats::storey_q_value(&final_p_values, min_storey_n),
                FdrType::Bh => stats::bh_q_value(&final_p_values),
            };

            for (idx, q) in final_indices.into_iter().zip(final_qs) {
                let feat = &mut new_features[idx];
                feat.decoy_free_q_value = Some(q as f32);
                feat.spectrum_q = q as f32;
            }
        }
    }

    new_features.sort_unstable_by(|a, b| a.spectrum_q.total_cmp(&b.spectrum_q));
    new_features
}

pub fn calculate_peptide_q(
    features: &mut [Feature],
    db: &IndexedDatabase,
    threshold: f32,
) -> usize {
    let mut best_q: FnvHashMap<String, f32> = FnvHashMap::default();
    for feat in features.iter().filter(|f| f.rank == 1) {
        let peptide = db[feat.peptide_idx].to_string();
        best_q
            .entry(peptide)
            .and_modify(|q| *q = q.min(feat.spectrum_q))
            .or_insert(feat.spectrum_q);
    }
    for feat in features.iter_mut() {
        let peptide = db[feat.peptide_idx].to_string();
        if let Some(q) = best_q.get(&peptide) {
            feat.peptide_q = *q;
        }
    }
    best_q.values().filter(|&&q| q <= threshold).count()
}

pub fn calculate_protein_q(
    features: &mut [Feature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
) -> usize {
    let mut protein_peptide_map: FnvHashMap<String, FnvHashMap<String, f64>> =
        FnvHashMap::default();
    for feat in features.iter() {
        if let Some(p_val) = feat.decoy_free_p_value {
            let protein_key = db[feat.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            let peptide_seq = db[feat.peptide_idx].to_string();
            let peptide_map = protein_peptide_map.entry(protein_key).or_default();
            peptide_map
                .entry(peptide_seq)
                .and_modify(|p| *p = p.min(p_val as f64))
                .or_insert(p_val as f64);
        }
    }

    let mut protein_p_values = Vec::new();
    let mut protein_keys = Vec::new();
    for (key, peptide_map) in protein_peptide_map {
        let p_values: Vec<f64> = peptide_map.values().cloned().collect();
        let combined_p = stats::combine_fisher(&p_values);
        protein_keys.push(key);
        protein_p_values.push(combined_p);
    }

    let protein_q_values = match settings.type_ {
        FdrType::Storey => stats::storey_q_value(&protein_p_values, settings.min_storey_n),
        FdrType::Bh => stats::bh_q_value(&protein_p_values),
    };

    let mut best_q: FnvHashMap<String, f32> = FnvHashMap::default();
    for (key, q) in protein_keys.into_iter().zip(protein_q_values) {
        best_q.insert(key, q as f32);
    }

    for feat in features.iter_mut() {
        let protein_key = db[feat.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        if let Some(q) = best_q.get(&protein_key) {
            feat.protein_q = *q;
        } else {
            feat.protein_q = 1.0;
        }
    }
    best_q
        .values()
        .filter(|&&q| q <= settings.protein_fdr)
        .count()
}

pub fn decoy_free_precursor(
    peaks: &mut FnvHashMap<(PrecursorId, bool), (Peak, Vec<f64>)>,
    threshold: f32,
) -> usize {
    let decoy_scores: Vec<f64> = peaks
        .iter()
        .filter_map(|((_, is_decoy), (peak, _))| if *is_decoy { Some(peak.score) } else { None })
        .collect();

    // Filter out NaNs before fitting to protect mean/variance calculations
    let valid_scores: Vec<f64> = decoy_scores.into_iter().filter(|s| s.is_finite()).collect();

    if valid_scores.len() < 50 {
        return 0;
    }

    let (mu, beta_raw) = fit_gumbel_moments(&valid_scores);

    // --- HARD GUARD: mu/beta must be finite and beta > 0 ---
    if !mu.is_finite() || !beta_raw.is_finite() || beta_raw <= 0.0 {
        return 0;
    }

    // NaN-safe clamp
    let beta = beta_raw.max(1e-9);

    let gumbel = match Gumbel::new(mu, beta) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    let mut target_keys = Vec::new();
    let mut target_pvalues = Vec::new();

    for (key, (peak, _)) in peaks.iter() {
        if !key.1 {
            if !peak.score.is_finite() {
                continue;
            }
            let p = gumbel.sf(peak.score).clamp(0.0, 1.0).max(1e-300);
            target_keys.push(*key);
            target_pvalues.push(p);
        }
    }

    if target_keys.is_empty() {
        return 0;
    }

    let q_values = stats::bh_q_value(&target_pvalues);

    for (key, q) in target_keys.into_iter().zip(q_values) {
        if let Some((peak, _)) = peaks.get_mut(&key) {
            peak.q_value = q as f32;
        }
    }

    // Explicitly set q=1.0 for skipped (NaN) targets
    for ((_, is_decoy), (peak, _)) in peaks.iter_mut() {
        if !*is_decoy && !peak.score.is_finite() {
            peak.q_value = 1.0;
        }
    }

    peaks
        .values()
        .filter(|(peak, _)| peak.score.is_finite() && peak.q_value <= threshold)
        .count()
}

fn fit_lower_order_regression(
    data: &[(u32, f64)],
    min_rank: u32,
    max_rank: u32,
    min_count: usize,
) -> Option<(f64, f64)> {
    let span = (max_rank - min_rank + 1) as usize;
    let mut rank_sums = vec![0.0f64; span];
    let mut rank_counts = vec![0usize; span];

    for &(rank, score) in data {
        if rank < min_rank || rank > max_rank {
            continue;
        }
        let idx = (rank - min_rank) as usize;
        rank_sums[idx] += score;
        rank_counts[idx] += 1;
    }

    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();

    for r in min_rank..=max_rank {
        let idx = (r - min_rank) as usize;
        let count = rank_counts[idx];
        if count >= min_count {
            let mean = rank_sums[idx] / count as f64;

            // --- REFINEMENT: Use Digamma instead of ln(r) ---
            let neg_psi = -digamma(r as f64);

            x_vec.push(neg_psi);
            y_vec.push(mean);
        }
    }

    if x_vec.len() < 2 {
        return None;
    }

    let n_points = x_vec.len() as f64;
    let sum_x: f64 = x_vec.iter().sum();
    let sum_y: f64 = y_vec.iter().sum();
    let sum_xy: f64 = x_vec.iter().zip(&y_vec).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = x_vec.iter().map(|x| x * x).sum();

    // --- Guard against degenerate regression (denom ~ 0) ---
    let denom = n_points * sum_xx - sum_x.powi(2);
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return None;
    }

    // Linear Regression: y = Intercept + Slope * (-psi(r))
    let slope = (n_points * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n_points;

    // Slope ~= beta
    let beta = slope;

    // Check for finiteness explicitly
    if !beta.is_finite() || beta <= 0.0 || !intercept.is_finite() {
        None
    } else {
        log::info!(
            "LowerOrder Fit (Digamma): beta={:.4}, intercept={:.4}",
            beta,
            intercept
        );
        Some((intercept, beta))
    }
}

fn fit_gumbel_moments(scores: &[f64]) -> (f64, f64) {
    // Require finite inputs
    let finite: Vec<f64> = scores.iter().cloned().filter(|x| x.is_finite()).collect();
    if finite.len() < 2 {
        return (f64::NAN, f64::NAN);
    }

    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;

    let variance = finite
        .iter()
        .map(|s| {
            let d = s - mean;
            d * d
        })
        .sum::<f64>()
        / n;

    if !variance.is_finite() || variance < 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let beta = (variance * 6.0 / std::f64::consts::PI.powi(2)).sqrt();
    if !beta.is_finite() || beta <= 0.0 {
        return (f64::NAN, f64::NAN);
    }

    let mu = mean - EULER_MASCHERONI * beta;
    if !mu.is_finite() {
        return (f64::NAN, f64::NAN);
    }

    (mu, beta)
}

fn fit_gumbel_mle(scores: &[f64]) -> Option<(f64, f64)> {
    let finite: Vec<f64> = scores.iter().cloned().filter(|x| x.is_finite()).collect();
    if finite.len() < 2 {
        return None;
    }

    let n = finite.len() as f64;
    let x_bar = finite.iter().sum::<f64>() / n;

    let (_, mut beta) = fit_gumbel_moments(&finite);
    if !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    for _ in 0..20 {
        let mut num = 0.0;
        let mut den = 0.0;

        for &x in &finite {
            let z = x / beta;
            if !z.is_finite() {
                continue;
            }
            let exp_neg_z = (-z).exp();
            if !exp_neg_z.is_finite() {
                continue;
            }
            num += x * exp_neg_z;
            den += exp_neg_z;
        }

        if !den.is_finite() || den <= 0.0 {
            return None;
        }

        let next_beta = x_bar - (num / den);
        if !next_beta.is_finite() || next_beta <= 0.0 {
            return None;
        }

        if (next_beta - beta).abs() < 1e-5 {
            beta = next_beta;
            break;
        }
        beta = next_beta;
    }

    let sum_exp = finite.iter().map(|&x| (-x / beta).exp()).sum::<f64>();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return None;
    }

    let mu = -beta * (sum_exp / n).ln();
    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        None
    } else {
        Some((mu, beta))
    }
}

fn kde_density(x: f64, samples: &[f64], bw: f64) -> f64 {
    // --- Internal Guard against bad inputs ---
    // Added !x.is_finite() check as requested
    if !x.is_finite() || samples.is_empty() || !bw.is_finite() || bw <= 0.0 {
        return 1e-300;
    }

    let n = samples.len() as f64;
    let norm = 1.0 / (n * bw * (2.0 * std::f64::consts::PI).sqrt());
    let sum_kernel: f64 = samples
        .iter()
        .map(|&xi| {
            let u = (x - xi) / bw;
            (-0.5 * u * u).exp()
        })
        .sum();
    norm * sum_kernel
}
