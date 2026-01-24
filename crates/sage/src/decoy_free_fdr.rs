use crate::database::IndexedDatabase;
use crate::input::{FdrSettings, FdrType, ModelFit};
use crate::lfq::{Peak, PrecursorId};
use crate::ml::nokoi;
use crate::ml::stats;
use crate::scoring::Feature;
use fnv::{FnvHashMap, FnvHashSet};
use rayon::prelude::*;
use statrs::consts::EULER_MASCHERONI;
use statrs::distribution::{Continuous, ContinuousCDF, Gumbel};

// --- HELPER MATH (Phase 3 Dependencies) ---

fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();
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

/// Silverman's Rule for KDE Bandwidth (adaptive to data std and n)
/// CRITICAL for low-input: Prevents Div/0 crashes when n < 2
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

/// Phase 3.5: Isotonic Regression (INCREASING)
/// Ensures P-values increase as Score quality decreases (High Score -> Low P-value)
fn isotonic_regression_increasing(p_values: &mut [f64]) {
    if p_values.is_empty() {
        return;
    }

    // blocks: (value, weight)
    let mut blocks: Vec<(f64, f64)> = p_values.iter().map(|&p| (p, 1.0)).collect();
    let mut i = 0;
    while i < blocks.len() - 1 {
        // Violator: Current P > Next P (should be <=)
        if blocks[i].0 > blocks[i + 1].0 {
            // Merge
            let w_new = blocks[i].1 + blocks[i + 1].1;
            let val_new = (blocks[i].0 * blocks[i].1 + blocks[i + 1].0 * blocks[i + 1].1) / w_new;
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
        let count = weight as usize;
        for _ in 0..count {
            if idx < p_values.len() {
                p_values[idx] = val;
                idx += 1;
            }
        }
    }
}

// --- PHASE 3: ROBUST MSFDR MODEL ---

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
        let mut sorted_targets = rank1_scores.to_vec();
        sorted_targets.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending
        let top_20 = (sorted_targets.len() as f64 * 0.2) as usize;
        let top_slice = &sorted_targets[0..top_20.max(5).min(sorted_targets.len())];

        let t_mean = top_slice.iter().sum::<f64>() / top_slice.len() as f64;
        let t_var =
            top_slice.iter().map(|s| (s - t_mean).powi(2)).sum::<f64>() / top_slice.len() as f64;
        let t_std = t_var.sqrt().max(1e-6);

        // Pi (Data-Driven Smart Start: Ratio of Rank1 > Approx Null Mean)
        let n_better = rank1_scores
            .iter()
            .filter(|&&s| s > null_mean_approx)
            .count();
        let init_pi = (n_better as f64 / rank1_scores.len() as f64).clamp(0.05, 0.95);

        let mut params = Self {
            null_loc: init_null_loc,
            null_scale: init_null_scale,
            target_mean: t_mean,
            target_std: t_std,
            target_alpha: 2.0,
            pi: init_pi,
        };

        // 2. EM Loop (Convergence Checked)
        let max_iters = 25;
        let mut old_ll = -f64::INFINITY;

        for _iter in 0..max_iters {
            let mut sum_z = 0.0;
            let mut sum_z_x = 0.0;
            let mut sum_z_xx = 0.0;
            let mut new_ll = 0.0;

            let null_dist = match Gumbel::new(params.null_loc, params.null_scale) {
                Ok(d) => d,
                Err(_) => return None,
            };

            for &x in rank1_scores {
                let f0 = null_dist.pdf(x).max(1e-10);
                let f1 = skew_normal_pdf(
                    x,
                    params.target_mean,
                    params.target_std,
                    params.target_alpha,
                )
                .max(1e-10);

                let num = params.pi * f1;
                let den = (1.0 - params.pi) * f0 + num;
                let z = num / den;

                sum_z += z;
                sum_z_x += z * x;
                sum_z_xx += z * x * x;
                new_ll += den.ln();
            }

            // Convergence Check (Phase 3 Requirement)
            if (new_ll - old_ll).abs() < 1e-5 {
                break;
            }
            old_ll = new_ll;

            // M-Step
            let n_total = rank1_scores.len() as f64;
            params.pi = sum_z / n_total;
            params.target_mean = sum_z_x / sum_z;
            let var = (sum_z_xx / sum_z) - params.target_mean.powi(2);
            params.target_std = var.sqrt().max(1e-6);

            // Heuristic alpha update
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
        let null_dist = Gumbel::new(self.null_loc, self.null_scale).unwrap();
        let f0 = null_dist.pdf(x).max(1e-10);
        let f1 =
            skew_normal_pdf(x, self.target_mean, self.target_std, self.target_alpha).max(1e-10);
        let den = (1.0 - self.pi) * f0 + self.pi * f1;
        ((1.0 - self.pi) * f0 / den).clamp(0.0, 1.0)
    }
}

// --- MAIN FUNCTION ---

/// Calculate spectrum-level q-values using Gumbel-based decoy-free methods.
pub fn calculate_q_values(psms: &[Feature], settings: &FdrSettings) -> Vec<Feature> {
    let mut new_features = psms.to_vec();

    let min_rank = settings.min_null_rank;
    let max_rank = settings.max_null_rank;
    let fit_method = &settings.model_fit;

    // Grab config values
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
        .collect();

    let purification_threshold = if !rank1_scores.is_empty() {
        let mid = rank1_scores.len() / 2;
        rank1_scores.select_nth_unstable_by(mid, |a, b| b.1.total_cmp(&a.1));
        rank1_scores[mid].1
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
                Some((psm.rank, psm.hyperscore as f64))
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
                    Some((psm.rank, psm.hyperscore as f64))
                } else {
                    None
                }
            })
            .collect();

        if fit_data.len() < min_null_size {
            log::error!("Null distribution too small. Aborting FDR.");
            new_features.par_iter_mut().for_each(|psm| {
                psm.spectrum_q = 1.0;
                psm.decoy_free_p_value = Some(1.0);
            });
            return new_features;
        }
    }

    let scores: Vec<f64> = fit_data.iter().map(|(_, s)| *s).collect();

    // --- LOGIC UPDATE: SPLIT EXECUTION FLAGS ---
    let debug_mode = matches!(fit_method, ModelFit::EnsembleDebug);

    // run_parametric: "Teacher" models. Needed for Ensemble, Debug, AND Nokoi (as baseline)
    let run_parametric = matches!(
        fit_method,
        ModelFit::Ensemble | ModelFit::EnsembleDebug | ModelFit::Nokoi
    );

    // run_nokoi: "Student" model. Only run if explicitly requested (Slow!)
    let run_nokoi = matches!(fit_method, ModelFit::EnsembleDebug | ModelFit::Nokoi);

    let (mu_mom, beta_mom) = fit_gumbel_moments(&scores);

    let (mu_mle, beta_mle) = if matches!(fit_method, ModelFit::Mle) || run_parametric {
        fit_gumbel_mle(&scores).unwrap_or((mu_mom, beta_mom))
    } else {
        (mu_mom, beta_mom)
    };

    let (mu_lo, beta_lo) = if matches!(fit_method, ModelFit::LowerOrder) || run_parametric {
        fit_lower_order_regression(&fit_data, min_rank, max_rank).unwrap_or((mu_mom, beta_mom))
    } else {
        (mu_mom, beta_mom)
    };

    // Robust MSFDR Model
    let msfdr_model = if matches!(fit_method, ModelFit::Msfdr) || run_parametric {
        let (mu_in, beta_in) = if run_parametric {
            (mu_lo, beta_lo)
        } else {
            (mu_mom, beta_mom)
        };
        let target_scores: Vec<f64> = new_features
            .iter()
            .filter(|f| f.rank == 1)
            .map(|f| f.hyperscore as f64)
            .collect();
        log::info!("Fitting Robust MSFDR mixture model...");
        RobustMsfdrModel::fit(&target_scores, mu_in, beta_in)
    } else {
        None
    };

    let dist_mom = Gumbel::new(mu_mom, beta_mom).unwrap();
    let dist_mle = Gumbel::new(mu_mle, beta_mle).unwrap();
    let dist_lo = Gumbel::new(mu_lo, beta_lo).unwrap();

    // --- FIX 1: Downsample the KDE reference set ---
    // With 1M+ PSMs, the naive KDE (N^2) hangs the system.
    // We downsample to the user-configured limit (default 20k) which is sufficient for density estimation.
    let mut target_scores_kde: Vec<f64> = new_features
        .iter()
        .filter(|f| f.rank == 1)
        .map(|f| f.hyperscore as f64)
        .collect();

    // Use the setting from JSON, falling back to 20,000 if 0 is accidentally passed
    let kde_limit = if settings.kde_samples > 0 {
        settings.kde_samples
    } else {
        20_000
    };

    if target_scores_kde.len() > kde_limit {
        let step = target_scores_kde.len() / kde_limit;
        target_scores_kde = target_scores_kde.into_iter().step_by(step).collect();

        log::debug!(
            "Downsampled KDE reference from {} to {} points (step size: {})",
            new_features.len(),
            target_scores_kde.len(),
            step
        );
    } else {
        log::info!(
            "Using full KDE reference set ({} points)",
            target_scores_kde.len()
        );
    }

    let bandwidth = silverman_bw(&target_scores_kde);

    // --- FIX 2: Parallelize the calculation loop ---
    // Note: We cannot push to vectors inside par_iter_mut, so we calculate fields in parallel,
    // then collect the indices/scores for PAVA in a second (fast, linear) pass.
    new_features.par_iter_mut().for_each(|psm| {
        psm.discriminant_score = f32::NAN;
        psm.posterior_error = f32::NAN;

        if psm.rank == 1 {
            let x = psm.hyperscore as f64;

            let p_mom = dist_mom.sf(x);
            let p_mle = dist_mle.sf(x);
            let p_lo = dist_lo.sf(x);

            let p_final = match fit_method {
                ModelFit::Moments => p_mom,
                ModelFit::Msfdr => p_mom, // MSFDR handled in PEP, fallback to Mom for P-value if not overridden
                ModelFit::Mle => p_mle,
                ModelFit::LowerOrder => p_lo,
                // Ensemble, Debug, OR Nokoi all start with the Harmonic Mean of Parametric models
                ModelFit::Ensemble | ModelFit::EnsembleDebug | ModelFit::Nokoi => {
                    if let Some(m) = &msfdr_model {
                        let p_msfdr = m.calculate_pep(x);
                        stats::combine_hmp(&[p_mom, p_mle, p_lo, p_msfdr])
                    } else {
                        stats::combine_hmp(&[p_mom, p_mle, p_lo])
                    }
                }
            };

            // FIX: This path (else block) was the bottleneck.
            // KDE over 1M points -> 1 Trillion ops.
            // Downsampling target_scores_kde makes this tractable.
            let pep = if let Some(model) = &msfdr_model {
                model.calculate_pep(x)
            } else {
                let dist_active = match fit_method {
                    ModelFit::Mle => &dist_mle,
                    ModelFit::LowerOrder => &dist_lo,
                    _ => &dist_mom,
                };
                let f0 = dist_active.pdf(x);
                let f_target = kde_density(x, &target_scores_kde, bandwidth);
                (f0 / f_target).min(1.0)
            };

            psm.decoy_free_p_value = Some(p_final as f32);
            psm.decoy_free_pep = Some(pep as f32);
            psm.spectrum_q = p_final as f32;

            let safe_pep = pep.max(1e-15);
            psm.decoy_free_score = Some((-10.0 * safe_pep.log10()) as f32);

            if debug_mode {
                psm.p_moments = Some(p_mom as f32);
                psm.p_mle = Some(p_mle as f32);
                psm.p_lower_order = Some(p_lo as f32);
                if let Some(m) = &msfdr_model {
                    psm.p_msfdr = Some(m.calculate_pep(x) as f32);
                }
            }
        } else {
            psm.spectrum_q = 1.0;
            psm.decoy_free_p_value = None;
            psm.decoy_free_pep = None;
            psm.decoy_free_score = None;
            psm.decoy_free_q_value = None;
        }
    });

    // --- Collect vectors for PAVA (Sequential Pass) ---
    // This is very fast (linear memory access) compared to the calc loop.
    let mut rank1_indices = Vec::with_capacity(new_features.len() / 2);
    let mut rank1_scores_for_pava = Vec::with_capacity(new_features.len() / 2);

    for (i, psm) in new_features.iter().enumerate() {
        if psm.rank == 1 {
            rank1_indices.push(i);
            rank1_scores_for_pava.push(psm.hyperscore as f64);
        }
    }

    // --- PAVA CALIBRATION ---
    let mut pava_data: Vec<(f64, usize)> = rank1_scores_for_pava
        .iter()
        .zip(rank1_indices.iter())
        .map(|(&s, &idx)| (s, idx))
        .collect();

    pava_data.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut sorted_pvalues: Vec<f64> = pava_data
        .iter()
        .map(|&(_, idx)| new_features[idx].spectrum_q as f64)
        .collect();

    isotonic_regression_increasing(&mut sorted_pvalues);

    let mut calibrated_pvalues_map = FnvHashMap::default();
    for (i, &(_, idx)) in pava_data.iter().enumerate() {
        let cal_p = sorted_pvalues[i];
        new_features[idx].spectrum_q = cal_p as f32;
        calibrated_pvalues_map.insert(idx, cal_p);
    }

    let final_pvalues_for_q: Vec<f64> = rank1_indices
        .iter()
        .map(|idx| *calibrated_pvalues_map.get(idx).unwrap())
        .collect();

    let q_values = match settings.type_ {
        FdrType::Storey => stats::storey_q_value(&final_pvalues_for_q, min_storey_n),
        FdrType::Bh => stats::bh_q_value(&final_pvalues_for_q),
    };

    for (idx, q) in rank1_indices.into_iter().zip(q_values) {
        let feat = &mut new_features[idx];
        feat.spectrum_q = q as f32;
        feat.decoy_free_q_value = Some(q as f32);
    }

    // --- NOKOI RESCORING ---
    // Optimization: ONLY run if we specifically asked for Nokoi or Debug
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

                    // Combine old ensemble + nokoi using HMP
                    let old_p = feat.decoy_free_p_value.unwrap_or(1.0) as f64;
                    let final_p = stats::combine_hmp(&[old_p, nokoi_p]);

                    feat.decoy_free_p_value = Some(final_p as f32);
                    feat.spectrum_q = final_p as f32;

                    let proxy_pep = final_p.max(1e-15);
                    feat.decoy_free_score = Some((-10.0 * proxy_pep.log10()) as f32);

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

/// Calculate peptide-level q-values (Best Rank-1 PSM per peptide)
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

    if decoy_scores.len() < 50 {
        return 0;
    }

    let (mu, beta) = fit_gumbel_moments(&decoy_scores);
    let gumbel = Gumbel::new(mu, beta).unwrap();

    let mut target_keys = Vec::new();
    let mut target_pvalues = Vec::new();

    for (key, (peak, _)) in peaks.iter() {
        if !key.1 {
            let p = gumbel.sf(peak.score);
            target_keys.push(*key);
            target_pvalues.push(p);
        }
    }

    let q_values = stats::bh_q_value(&target_pvalues);

    for (key, q) in target_keys.into_iter().zip(q_values) {
        if let Some((peak, _)) = peaks.get_mut(&key) {
            peak.q_value = q as f32;
        }
    }
    peaks
        .values()
        .filter(|(peak, _)| !peak.score.is_nan() && peak.q_value <= threshold)
        .count()
}

fn fit_lower_order_regression(
    data: &[(u32, f64)],
    min_rank: u32,
    max_rank: u32,
) -> Option<(f64, f64)> {
    let mut rank_sums = FnvHashMap::default();
    let mut rank_counts = FnvHashMap::default();
    for &(rank, score) in data {
        *rank_sums.entry(rank).or_insert(0.0) += score;
        *rank_counts.entry(rank).or_insert(0) += 1;
    }
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    for r in min_rank..=max_rank {
        if let Some(&sum) = rank_sums.get(&r) {
            let count = *rank_counts.get(&r).unwrap();
            if count > 10 {
                let mean = sum / count as f64;
                x_vec.push((r as f64).ln());
                y_vec.push(mean);
            }
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
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x.powi(2));
    let intercept = (sum_y - slope * sum_x) / n;
    let beta = -slope;
    let mu = intercept;
    if beta <= 0.0 || mu.is_nan() {
        None
    } else {
        Some((mu, beta))
    }
}

fn fit_gumbel_moments(scores: &[f64]) -> (f64, f64) {
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let beta = (variance * 6.0 / std::f64::consts::PI.powi(2)).sqrt();
    let mu = mean - EULER_MASCHERONI * beta;
    (mu, beta)
}

fn fit_gumbel_mle(scores: &[f64]) -> Option<(f64, f64)> {
    let n = scores.len() as f64;
    let x_bar = scores.iter().sum::<f64>() / n;
    let (_, mut beta) = fit_gumbel_moments(scores);
    for _ in 0..20 {
        let mut num = 0.0;
        let mut den = 0.0;
        for &x in scores {
            let z = x / beta;
            let exp_neg_z = (-z).exp();
            num += x * exp_neg_z;
            den += exp_neg_z;
        }
        if den == 0.0 {
            return None;
        }
        let next_beta = x_bar - (num / den);
        if (next_beta - beta).abs() < 1e-5 {
            beta = next_beta;
            break;
        }
        beta = next_beta;
    }
    let sum_exp = scores.iter().map(|&x| (-x / beta).exp()).sum::<f64>();
    let mu = -beta * (sum_exp / n).ln();
    if mu.is_nan() || beta.is_nan() || beta <= 0.0 {
        None
    } else {
        Some((mu, beta))
    }
}

fn kde_density(x: f64, samples: &[f64], bw: f64) -> f64 {
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
