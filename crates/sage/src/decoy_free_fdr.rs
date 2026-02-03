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

// --- HELPER MATH ---

fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs().min(10.0);

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x).exp();
    sign * y
}

fn skew_normal_pdf(x: f64, loc: f64, scale: f64, alpha: f64) -> f64 {
    let scale = scale.max(1e-9);
    let z = (x - loc) / scale;
    let phi = (-(z * z) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let big_phi = 0.5 * (1.0 + erf_approx(alpha * z / std::f64::consts::SQRT_2));
    (2.0 / scale) * phi * big_phi
}

#[inline]
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }
    if a.is_infinite() && a.is_sign_positive() {
        return a;
    }
    if b.is_infinite() && b.is_sign_positive() {
        return b;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Isotonic Regression (INCREASING)
/// Ensures P-values increase as Score quality decreases (High Score -> Low P-value)
fn isotonic_regression_increasing(p_values: &mut [f64]) {
    if p_values.is_empty() {
        return;
    }

    // blocks: (value, weight_count)
    let mut blocks: Vec<(f64, usize)> = p_values.iter().map(|&p| (p, 1)).collect();
    let mut i = 0;
    while i < blocks.len() - 1 {
        // Violator: Current P > Next P (should be <=)
        if blocks[i].0 > blocks[i + 1].0 {
            // Merge
            let w_prev = blocks[i].1;
            let w_next = blocks[i + 1].1;
            let w_new = w_prev + w_next;
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
        Some((v[n / 2 - 1] + v[n / 2]) / 2.0)
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

        let init_null_loc = mu_in;
        let init_null_scale = beta_in.max(1e-6);
        let null_mean_approx = init_null_loc + EULER_MASCHERONI * init_null_scale;

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

        let max_iters = 25;
        let mut old_ll = -f64::INFINITY;

        for iter in 0..max_iters {
            let mut sum_z = 0.0;
            let mut sum_z_x = 0.0;
            let mut sum_z_xx = 0.0;
            let mut new_ll = 0.0;
            let mut n_used = 0usize;

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
            let pi = params.pi.clamp(0.01, 0.99);
            let log_pi = pi.ln();
            let log_1m_pi = (1.0 - pi).ln();

            for &x in rank1_scores {
                if !x.is_finite() {
                    continue;
                }
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
                let log_num = log_pi + log_f1;
                let log_den = log_add_exp(log_1m_pi + log_f0, log_num);
                if !log_den.is_finite() {
                    continue;
                }
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

            if n_used < 10 {
                return None;
            }
            if sum_z < 1e-8 {
                if iter == 0 {
                    return None;
                } else {
                    break;
                }
            }
            if !new_ll.is_finite() {
                return None;
            }

            let avg_ll = new_ll / (n_used as f64);
            let tol_rel = 1e-4;
            let tol_abs = 1e-6;
            if old_ll.is_finite() {
                let delta = (avg_ll - old_ll).abs();
                let scale = old_ll.abs().max(1.0);
                if delta < tol_abs || (delta / scale) < tol_rel {
                    break;
                }
            }
            old_ll = avg_ll;

            let n_total = n_used as f64;
            params.pi = (sum_z / n_total).clamp(0.01, 0.99);
            params.target_mean = sum_z_x / sum_z;
            let var = (sum_z_xx / sum_z) - params.target_mean.powi(2);
            if !var.is_finite() || var < 0.0 {
                return None;
            }
            params.target_std = var.sqrt().max(1e-6);
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
pub fn calculate_q_values(psms: &[Feature], settings: &FdrSettings) -> Vec<Feature> {
    let mut new_features = psms.to_vec();
    let min_rank = settings.min_null_rank;
    let max_rank = settings.max_null_rank;
    let min_null_size = settings.min_null_size;
    let min_storey_n = settings.min_storey_n;

    let use_ensemble = matches!(settings.model_fit, ModelFit::Ensemble);

    log::info!(
        "Building null distribution [Rank {}..={}] using {:?} fit (Ensemble={})",
        min_rank,
        max_rank,
        settings.model_fit,
        use_ensemble
    );

    // --- PHASE 1: SOFT PURIFIED NULL ---
    let mut rank1_scores: Vec<(u32, f64)> = new_features
        .iter()
        .filter(|f| f.rank == 1)
        .map(|f| (f.peptide_idx.0, f.hyperscore as f64))
        .filter(|(_, s)| s.is_finite())
        .collect();

    let purification_threshold = if rank1_scores.len() >= 10 {
        rank1_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let p_factor = settings.purification_factor;
        let top_k = ((rank1_scores.len() as f64) * p_factor).round() as usize;
        let top_k = top_k.max(5).min(rank1_scores.len());
        rank1_scores[top_k - 1].1
    } else {
        1000.0
    };

    let purified_peptides: FnvHashSet<u32> = rank1_scores
        .iter()
        .filter(|(_, score)| *score >= purification_threshold)
        .map(|(idx, _)| *idx)
        .collect();

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
                psm.posterior_error = 1.0;
                psm.discriminant_score = 0.0;
                if psm.rank == 1 {
                    psm.decoy_free_p_value = Some(1.0);
                    psm.decoy_free_pep = Some(1.0);
                    psm.decoy_free_score = Some(0.0);
                    psm.decoy_free_q_value = Some(1.0);
                    psm.decoy_free_peptide_q = Some(1.0);
                }
            });
            return new_features;
        }
    }

    // --- SAFETY: filter non-finite scores in fit_data ---
    fit_data.retain(|&(_, s)| s.is_finite());
    if fit_data.len() < min_null_size {
        log::error!("Null distribution too small after filtering non-finite scores.");
        new_features.par_iter_mut().for_each(|psm| {
            psm.spectrum_q = 1.0;
            psm.posterior_error = 1.0;
            psm.discriminant_score = 0.0;
        });
        return new_features;
    }

    let scores: Vec<f64> = fit_data.iter().map(|(_, s)| *s).collect();

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

    let log_n_global_vec: Vec<f64> = new_features
        .iter()
        .filter(|f| f.rank == 1 && (f.hyperscore as f64).is_finite())
        .filter_map(|f| {
            let n = f.scored_candidates as f64;
            if !n.is_finite() || n < 2.0 {
                return None;
            }
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
        clamp_f64(n_eff_global, 10.0, 1e7)
    };
    let n_global = clamp_f64(n_global, 10.0, 1e7);

    log::info!("Effective search space (n_global): {:.1}", n_global);

    let map_to_standard_output = matches!(settings.mode, FdrMode::DecoyFree);

    // 1. Fit Moments
    let (mu_mom, beta_mom) = fit_gumbel_moments(&scores);
    let moments_params_ok = mu_mom.is_finite() && beta_mom.is_finite() && beta_mom > 0.0;

    // 2. Fit Lower Order
    let min_count = settings.min_rank_count;
    let (lo_intercept_raw, lo_beta_raw) =
        fit_lower_order_regression(&fit_data, min_rank, max_rank, min_count).unwrap_or_else(|| {
            if moments_params_ok {
                (mu_mom + beta_mom * n_global.ln(), beta_mom)
            } else {
                (0.0, 1.0)
            }
        });

    let lo_beta = if lo_beta_raw.is_finite() && lo_beta_raw > 0.0 {
        lo_beta_raw
    } else if moments_params_ok {
        beta_mom.max(1e-9)
    } else {
        1.0
    };
    let lo_intercept = if lo_intercept_raw.is_finite() {
        lo_intercept_raw
    } else {
        0.0
    };
    let lo_beta_shrunk = if moments_params_ok {
        (0.95 * lo_beta) + (0.05 * beta_mom.max(1e-9))
    } else {
        lo_beta
    };

    // 3. Fit MLE
    let (mu_mle, beta_mle) = fit_gumbel_mle(&scores).unwrap_or_else(|| (mu_mom, beta_mom));

    // 4. Fit Robust MSFDR
    let beta_lo_seed = lo_beta_shrunk.max(1e-9);
    let mu_lo_global = lo_intercept + beta_lo_seed * (n_eff_global.ln() - n_global.ln());

    let run_msfdr = use_ensemble || matches!(settings.model_fit, ModelFit::Msfdr);
    let msfdr_model = if run_msfdr {
        let target_scores: Vec<f64> = new_features
            .iter()
            .filter(|f| f.rank == 1)
            .map(|f| f.hyperscore as f64)
            .filter(|x| x.is_finite())
            .collect();
        RobustMsfdrModel::fit(&target_scores, mu_lo_global, beta_lo_seed)
    } else {
        None
    };

    if !moments_params_ok {
        log::error!("Invalid null fit. FDR will fail closed.");
        new_features.par_iter_mut().for_each(|psm| {
            psm.spectrum_q = 1.0;
            psm.posterior_error = 1.0;
            psm.discriminant_score = 0.0;
        });
        return new_features;
    }

    let dist_mom = Gumbel::new(mu_mom, beta_mom).unwrap();
    let dist_mle = Gumbel::new(mu_mle, beta_mle).unwrap_or(dist_mom.clone());

    let lo_alpha = settings.lo_multiplicity_alpha.clamp(0.0, 1.0);
    let lo_ln_ratio_cap = settings.lo_ln_ratio_cap.max(0.0);

    let run_nokoi = use_ensemble || matches!(settings.model_fit, ModelFit::Nokoi);
    let nokoi_probs = if run_nokoi {
        log::info!("Running Nokoi Rescoring...");
        nokoi::rescore(&new_features, 0.01)
    } else {
        None
    };
    let nokoi_p_values = if let Some(ref probs) = nokoi_probs {
        Some(nokoi::calc_empirical_p_values(&new_features, probs))
    } else {
        None
    };

    let nokoi_p_ref = &nokoi_p_values;

    // --- CALCULATION LOOP ---
    new_features
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, psm)| {
            if psm.rank == 1 {
                let x = psm.hyperscore as f64;
                if !x.is_finite() {
                    psm.spectrum_q = 1.0;
                    psm.posterior_error = 1.0;
                    psm.discriminant_score = 0.0;
                    return;
                }

                let p_mom = dist_mom.sf(x).clamp(0.0, 1.0).max(1e-300);
                let p_mle = dist_mle.sf(x).clamp(0.0, 1.0).max(1e-300);

                let n_eff = if psm.scored_candidates >= 2 {
                    (psm.scored_candidates as f64).max(2.0).min(1e9)
                } else {
                    n_eff_global
                };
                let ln_ratio =
                    (n_eff.ln() - n_global.ln()).clamp(-lo_ln_ratio_cap, lo_ln_ratio_cap);
                let mu_i = lo_intercept + lo_beta_shrunk * lo_alpha * ln_ratio;
                let dist_lo = Gumbel::new(mu_i, lo_beta_shrunk).unwrap_or(dist_mom.clone());
                let p_lo = dist_lo.sf(x).clamp(0.0, 1.0).max(1e-300);

                let p_msfdr = msfdr_model
                    .as_ref()
                    .map(|m| m.calculate_seeded_null_p(x).clamp(0.0, 1.0).max(1e-300));
                let p_nokoi = if let Some(ref p_vec) = nokoi_p_ref {
                    Some(p_vec[idx].clamp(0.0, 1.0).max(1e-300))
                } else {
                    None
                };

                psm.p_mom = Some(p_mom as f32);
                psm.p_mle = Some(p_mle as f32);
                psm.p_lo = Some(p_lo as f32);
                psm.p_msfdr = p_msfdr.map(|v| v as f32);
                psm.p_nokoi = p_nokoi.map(|v| v as f32);

                let mut experts = vec![p_mom, p_mle, p_lo];
                if let Some(p) = p_msfdr {
                    experts.push(p);
                }
                if let Some(p) = p_nokoi {
                    experts.push(p);
                }

                let p_final = if use_ensemble {
                    stats::combine_hmp(&experts)
                } else {
                    match settings.model_fit {
                        ModelFit::Moments => p_mom,
                        ModelFit::Mle => p_mle,
                        ModelFit::LowerOrder => p_lo,
                        ModelFit::Msfdr => p_msfdr.unwrap_or(p_mom),
                        ModelFit::Nokoi => p_nokoi.unwrap_or(p_mom),
                        _ => p_mom,
                    }
                };

                let p_final = p_final.clamp(0.0, 1.0).max(1e-300);

                // Use p_final directly as the PEP (Error Probability) because it aggregates all experts.
                // This prevents MSFDR's pessimism from dominating when ML sees a strong match.
                let pep = p_final;

                psm.decoy_free_p_value = Some(p_final as f32);
                psm.decoy_free_pep = Some(pep as f32);

                let df_score = (-10.0 * (pep as f64).max(1e-15).log10()) as f32;
                psm.decoy_free_score = Some(df_score);
                psm.spectrum_q = p_final as f32;

                if map_to_standard_output {
                    psm.posterior_error = pep as f32;
                    // AGGRESSIVE: Overwrite Discriminant Score immediately for internal sorting
                    psm.discriminant_score = df_score;
                }
            } else {
                psm.spectrum_q = 1.0;
                psm.posterior_error = 1.0;
                psm.discriminant_score = 0.0;
                psm.decoy_free_p_value = None;
                psm.decoy_free_pep = None;
                psm.decoy_free_score = None;
                psm.decoy_free_q_value = None;
                psm.decoy_free_peptide_q = None;
                psm.p_mom = None;
                psm.p_mle = None;
                psm.p_lo = None;
                psm.p_msfdr = None;
                psm.p_nokoi = None;
            }
        });

    // --- PAVA CALIBRATION ---
    let mut pava_data: Vec<(f64, usize, f64)> = new_features
        .iter()
        .enumerate()
        .filter(|(_, f)| f.rank == 1)
        .map(|(idx, f)| (f.hyperscore as f64, idx, f.spectrum_q as f64))
        .collect();

    pava_data.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut sorted_pvalues: Vec<f64> = pava_data.iter().map(|&(_, _, p)| p).collect();
    isotonic_regression_increasing(&mut sorted_pvalues);

    for (i, &(_, idx, _)) in pava_data.iter().enumerate() {
        let cal_p = sorted_pvalues[i].clamp(0.0, 1.0).max(1e-300);
        new_features[idx].spectrum_q = cal_p as f32;
        new_features[idx].decoy_free_p_value = Some(cal_p as f32);
    }

    // --- Q-VALUES ---
    let rank1_indices: Vec<usize> = new_features
        .iter()
        .enumerate()
        .filter(|(_, f)| f.rank == 1)
        .map(|(i, _)| i)
        .collect();
    let rank1_p: Vec<f64> = rank1_indices
        .iter()
        .map(|&i| new_features[i].spectrum_q as f64)
        .collect();

    let q_values = match settings.type_ {
        FdrType::Storey => stats::storey_q_value(&rank1_p, min_storey_n),
        FdrType::Bh => stats::bh_q_value(&rank1_p),
    };

    for (&idx, q) in rank1_indices.iter().zip(q_values) {
        new_features[idx].spectrum_q = q as f32;
        new_features[idx].decoy_free_q_value = Some(q as f32);
    }

    if map_to_standard_output {
        new_features.par_iter_mut().for_each(|psm| {
            if psm.rank == 1 {
                if psm.posterior_error >= 0.99 && psm.spectrum_q < 0.1 {
                    psm.posterior_error = psm.spectrum_q;
                    psm.discriminant_score =
                        (-10.0 * (psm.spectrum_q as f64).max(1e-15).log10()) as f32;
                }
            }
        });
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

    // Use spectrum_q here which has been synced with decoy_free_q_value
    for feat in features.iter().filter(|f| f.rank == 1) {
        let peptide = db[feat.peptide_idx].to_string();
        best_q
            .entry(peptide)
            .and_modify(|q| *q = q.min(feat.spectrum_q))
            .or_insert(feat.spectrum_q);
    }

    // Update both peptide_q (standard) AND decoy_free_peptide_q (new)
    for feat in features.iter_mut() {
        let peptide = db[feat.peptide_idx].to_string();
        if let Some(q) = best_q.get(&peptide) {
            feat.peptide_q = *q;
            feat.decoy_free_peptide_q = Some(*q);
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
        let combined_p =
            stats::combine_fisher(&peptide_map.values().cloned().collect::<Vec<f64>>());
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
        feat.protein_q = *best_q.get(&protein_key).unwrap_or(&1.0);
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
    let valid_scores: Vec<f64> = decoy_scores.into_iter().filter(|s| s.is_finite()).collect();
    if valid_scores.len() < 50 {
        return 0;
    }
    let (mu, beta_raw) = fit_gumbel_moments(&valid_scores);
    if !mu.is_finite() || !beta_raw.is_finite() || beta_raw <= 0.0 {
        return 0;
    }
    let gumbel = Gumbel::new(mu, beta_raw.max(1e-9)).unwrap();
    let mut target_keys = Vec::new();
    let mut target_pvalues = Vec::new();
    for (key, (peak, _)) in peaks.iter() {
        if !key.1 && peak.score.is_finite() {
            target_keys.push(*key);
            target_pvalues.push(gumbel.sf(peak.score).clamp(0.0, 1.0).max(1e-300));
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
        if rank >= min_rank && rank <= max_rank {
            let idx = (rank - min_rank) as usize;
            rank_sums[idx] += score;
            rank_counts[idx] += 1;
        }
    }
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    for r in min_rank..=max_rank {
        let idx = (r - min_rank) as usize;
        if rank_counts[idx] >= min_count {
            x_vec.push(-digamma(r as f64));
            y_vec.push(rank_sums[idx] / rank_counts[idx] as f64);
        }
    }
    if x_vec.len() < 2 {
        return None;
    }
    let n = x_vec.len() as f64;
    let sum_x: f64 = x_vec.iter().sum();
    let sum_y: f64 = y_vec.iter().sum();
    let sum_xy: f64 = x_vec.iter().zip(&y_vec).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = x_vec.iter().map(|x| x * x).sum();
    let denom = n * sum_xx - sum_x.powi(2);
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return None;
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    if !slope.is_finite() || slope <= 0.0 || !intercept.is_finite() {
        None
    } else {
        Some((intercept, slope))
    }
}

fn fit_gumbel_moments(scores: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = scores.iter().cloned().filter(|x| x.is_finite()).collect();
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
            let exp_neg_z = (-z).exp();
            if exp_neg_z.is_finite() {
                num += x * exp_neg_z;
                den += exp_neg_z;
            }
        }
        if den <= 0.0 {
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
    let mu = -beta * (sum_exp / n).ln();
    if mu.is_finite() {
        Some((mu, beta))
    } else {
        None
    }
}
