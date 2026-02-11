use crate::database::IndexedDatabase;
use crate::input::{EnsemblePCombiner, EnsemblePepCombiner, LoRankKey};
use crate::input::{FdrSettings, FdrType, ModelFit};
use crate::lfq::{Peak, PrecursorId};
use crate::ml::lower_order::{
    fit_decoy_free_model, fit_gumbel_mle, fit_gumbel_moments, LowerOrderModel,
};
use crate::ml::nokoi;
use crate::ml::stats;
use crate::scoring::DfFeature;
use fnv::{FnvHashMap, FnvHashSet};
use rayon::prelude::*;
use statrs::consts::EULER_MASCHERONI;
use statrs::distribution::{Continuous, ContinuousCDF, Gumbel};
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Rank1Computed {
    idx: usize,
    // per-method p's
    p_mom: f64,
    p_mle: f64,
    p_lo: f64,
    p_msfdr: Option<f64>,
    p_nokoi: Option<f64>,
    // per-method peps
    pep_mom: f64,
    pep_mle: f64,
    pep_lo: f64,
    pep_msfdr: Option<f64>,
    pep_nokoi: Option<f64>,
    // final DF outputs
    p_final: f64,
    pep_final: f64,
    df_score: f32,
}

// =============================================================================
// Decoy-Free Lower-Order (LO) contract (Phase 0 / Step 0.1 — comment-only)
// =============================================================================
//
// Purpose
// -------
// LO implements a charge-stratified “Top Null Model” (TNM) selection via minimum
// BIC, and a set of per-charge Lower-Order Models (LOMs) fit on rank-null pools.
//
// Definitions (paper-aligned)
// ---------------------------
// - TNM (Top Null Model): a Gumbel(mu, beta) intended to model the top-scoring
//   distribution (rank==1) for a given charge state.
// - LOM (Lower-Order Model): regression fit using rank-null pool scores over
//   ranks k=2..10 (paper default), per charge state.
//
// Selection / fitting rules (invariants)
// --------------------------------------
// 1) LO is charge-stratified.
//    - All TNM/LOM fitting and selection are done independently per charge.
//
// 2) TNM selection considers exactly 4 candidates:
//      (LR vs mean-β) × (MLE vs moments)
//    - “LR”      : beta derived from lower-order regression strategy.
//    - “mean-β”  : beta derived as the mean of β estimates from multiple LOM fits.
//    - “MLE”     : mu (and beta when applicable) fitted by maximum likelihood.
//    - “moments” : mu/beta fitted by method of moments.
//
// 3) TNM BIC is computed using ONLY rank==1 scores for that charge.
//    - Candidate TNMs are compared by BIC on the charge-specific rank==1 set.
//    - The selected TNM is the candidate with minimum BIC (for that charge).
//
// 4) LOMs are fit from the rank-null pool for that charge using ranks k=2..10
//    (paper default), per charge.
//    - The rank-null pool is charge-specific and excludes rank==1.
//    - LOM fitting uses only ranks within the configured null-rank window.
// =============================================================================

// =============================================================================
// Helpers (math, calibration, parsing, diagnostics, and model fitting)
// =============================================================================

// -----------------------------------------------------------------------------
// 1) Low-level special functions + stable numeric primitives
// -----------------------------------------------------------------------------

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

#[inline]
fn neg_log10_p(p: f64) -> f64 {
    // “quality”: bigger is better
    -p.max(1e-300).log10()
}

// -----------------------------------------------------------------------------
// 1b) TEV normalization helper
// -----------------------------------------------------------------------------
//
// We keep core.hyperscore unchanged.
// For any Gumbel(mu,beta).sf(hyperscore), we compute TEV_norm = (hs - mu)/beta
// and evaluate sf(TEV_norm) under a standard Gumbel(0,1).
//
// This makes the sf(x) input a normalized TEV scale without changing outputs.
//
#[inline]
fn tev_norm_from_hyperscore(hs: f64, mu: f64, beta: f64) -> f64 {
    if !hs.is_finite() || !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return f64::NEG_INFINITY; // yields sf ~ 1.0 (fail-closed)
    }
    (hs - mu) / beta
}

// -----------------------------------------------------------------------------
// 2) Distributions / densities used by models (MSFDR target skew-normal)
// -----------------------------------------------------------------------------

fn skew_normal_pdf(x: f64, loc: f64, scale: f64, alpha: f64) -> f64 {
    let scale = scale.max(1e-9);
    let z = (x - loc) / scale;
    let phi = (-(z * z) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let big_phi = 0.5 * (1.0 + erf_approx(alpha * z / std::f64::consts::SQRT_2));
    (2.0 / scale) * phi * big_phi
}

// -----------------------------------------------------------------------------
// 3) PAVA / isotonic calibration utilities
// -----------------------------------------------------------------------------

fn isotonic_regression_increasing(p_values: &mut [f64]) {
    if p_values.is_empty() {
        return;
    }

    let mut blocks: Vec<(f64, usize)> = p_values.iter().map(|&p| (p, 1)).collect();
    let mut i = 0;
    while i < blocks.len() - 1 {
        if blocks[i].0 > blocks[i + 1].0 {
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

// -----------------------------------------------------------------------------
// 4) Simple statistics helpers (median / trimmed mean)
// -----------------------------------------------------------------------------

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

fn trimmed_mean(v: &mut [f64], trim_frac: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        return None;
    }
    let t = ((trim_frac.clamp(0.0, 0.49)) * (n as f64)).floor() as usize;
    let lo = t;
    let hi = n.saturating_sub(t);
    if lo >= hi {
        return None;
    }
    let slice = &v[lo..hi];
    let mean = slice.iter().sum::<f64>() / (slice.len() as f64);
    mean.is_finite().then_some(mean)
}

// -----------------------------------------------------------------------------
// 5) Ensemble combiners (p-values + PEPs)
// -----------------------------------------------------------------------------
fn combine_p_values(
    pvals: &[f64],
    how: EnsemblePCombiner,
    brown_params: Option<stats::BrownParams>,
) -> f64 {
    match how {
        EnsemblePCombiner::Hmp => stats::combine_hmp(pvals),
        EnsemblePCombiner::Fisher => stats::combine_fisher(pvals),
        EnsemblePCombiner::Brown => {
            if brown_params.is_none() {
                log::warn!("Brown requested but params missing; using Fisher fallback.");
            }
            stats::combine_brown(pvals, brown_params)
        }
    }
}

fn combine_peps(peps: &[f64], how: EnsemblePepCombiner) -> f64 {
    if peps.is_empty() {
        return 1.0;
    }
    match how {
        EnsemblePepCombiner::Mean => {
            let m = peps.iter().sum::<f64>() / (peps.len() as f64);
            m.clamp(0.0, 1.0)
        }
        EnsemblePepCombiner::GeometricMean => {
            let eps = 1e-12;
            let mean_log = peps
                .iter()
                .map(|&p| p.clamp(eps, 1.0 - eps).ln())
                .sum::<f64>()
                / (peps.len() as f64);
            mean_log.exp().clamp(0.0, 1.0)
        }
        EnsemblePepCombiner::LogitMean => {
            let eps = 1e-12;
            let logits: f64 = peps
                .iter()
                .map(|&p| {
                    let p = p.clamp(eps, 1.0 - eps);
                    (p / (1.0 - p)).ln()
                })
                .sum();
            let avg = logits / (peps.len() as f64);
            let odds = avg.exp();
            (odds / (1.0 + odds)).clamp(0.0, 1.0)
        }
    }
}

// -----------------------------------------------------------------------------
// 6) Feature field helpers (tiny setters/getters for DF streams)
// -----------------------------------------------------------------------------

#[inline(always)]
fn set_df_p_value(psm: &mut DfFeature, p: f32) {
    psm.decoy_free_p_value = Some(p);
}

#[inline(always)]
fn set_df_q_value(psm: &mut DfFeature, q: f32) {
    psm.decoy_free_q_value = Some(q);
}

#[inline(always)]
fn df_p_value(psm: &DfFeature) -> f32 {
    psm.decoy_free_p_value.unwrap_or(1.0)
}

#[inline(always)]
fn df_q_value(psm: &DfFeature) -> f32 {
    psm.decoy_free_q_value.unwrap_or(1.0)
}

// -----------------------------------------------------------------------------
// 6b) Canonical evidence accessor (raw hyperscore) + TEV normalization
// -----------------------------------------------------------------------------
//
// Contract:
// - We keep `core.hyperscore` UNCHANGED and treat it as the raw evidence score
//   produced by vanilla Sage ("hyperscore").
// - The decoy-free null is modeled as Gumbel(mu, beta) over this raw hyperscore
//   (fitted from the rank-null pool).
// - Whenever we need a "TEV-normalized" input for survival evaluation, we compute:
//       tev_norm = (hyperscore - mu) / beta
//   and evaluate `sf(tev_norm)` under a STANDARD Gumbel(0, 1).
//
// Invariants:
// - Null pool scores == raw hyperscore values returned by `tev(f)` below.
// - Moments/MLE/LO fit parameters are fit on those same raw values.
// - All p-value computations use `tev_norm_from_hyperscore(hs, mu, beta)`
#[inline(always)]
fn tev(f: &DfFeature) -> Option<f64> {
    let x = f.core.hyperscore as f64;
    if x.is_finite() {
        Some(x)
    } else {
        None
    }
}

// -----------------------------------------------------------------------------
// 7) Protein-string classification (contam / entrapment)
// -----------------------------------------------------------------------------

#[inline]
fn is_contam_str(proteins: &str) -> bool {
    proteins.contains("Cont_")
}

#[inline]
fn is_entrapment_str(proteins: &str) -> bool {
    proteins.contains("|Ent_") || proteins.contains("Ent_")
}

// -----------------------------------------------------------------------------
// 8) Empirical null tail p-values (used for Nokoi null mapping)
// -----------------------------------------------------------------------------

fn empirical_p_from_null_ge(null_sorted: &[f64], x: f64) -> f64 {
    if null_sorted.len() < 10 || !x.is_finite() {
        return 1.0;
    }
    let n = null_sorted.len();
    let idx = null_sorted.partition_point(|&v| v < x); // first >= x
    let count_ge = (n - idx) as f64;
    ((count_ge + 1.0) / ((n as f64) + 1.0))
        .clamp(0.0, 1.0)
        .max(1e-300)
}

// -----------------------------------------------------------------------------
// 9) Debug / diagnostics helpers
// -----------------------------------------------------------------------------

fn summarize_pvec(name: &str, p: &[f64]) {
    if p.is_empty() {
        log::warn!("DF DEBUG {}: empty p-vector", name);
        return;
    }

    let mut v: Vec<f64> = p
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .map(|x| x.clamp(0.0, 1.0).max(1e-300))
        .collect();

    if v.is_empty() {
        log::warn!("DF DEBUG {}: no finite p-values", name);
        return;
    }

    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len();
    let min_p = v[0];
    let max_p = v[m - 1];
    let median_p = v[m / 2];
    let p05 = v[((m - 1) as f64 * 0.05).round() as usize];

    let n_le_05 = v.iter().filter(|&&x| x <= 0.5).count();
    let n_le_01 = v.iter().filter(|&&x| x <= 0.01).count();

    log::info!(
        "DF DEBUG {}: m={} min={:.3e} p05={:.3e} median={:.3e} max={:.3e}  frac<=0.5={:.4}  frac<=0.01={:.6}",
        name,
        m,
        min_p,
        p05,
        median_p,
        max_p,
        (n_le_05 as f64) / (m as f64),
        (n_le_01 as f64) / (m as f64),
    );
}

fn summarize_q(label: &str, qs_in: impl Iterator<Item = f32>) {
    let mut qs: Vec<f32> = qs_in.filter(|q| q.is_finite()).collect();

    if qs.is_empty() {
        log::info!("DF DEBUG {}: n=0 (no finite q)", label);
        return;
    }

    qs.sort_by(|a, b| a.total_cmp(b));

    let n = qs.len();
    let idx = |p: f64| -> usize {
        let i = ((p * (n as f64 - 1.0)).round() as isize).clamp(0, n as isize - 1);
        i as usize
    };

    let min = qs[0];
    let p05 = qs[idx(0.05)];
    let median = qs[idx(0.50)];
    let p95 = qs[idx(0.95)];
    let max = qs[n - 1];

    let frac_le_001 = qs.iter().filter(|&&q| q <= 0.01).count() as f64 / n as f64;
    let frac_eq_pi0ish = {
        let med = median;
        qs.iter().filter(|&&q| q == med).count() as f64 / n as f64
    };

    log::info!(
        "DF DEBUG {}: n={} min={:.4e} p05={:.4e} med={:.4e} p95={:.4e} max={:.4e} frac<=0.01={:.4} frac==med={:.4}",
        label, n, min, p05, median, p95, max, frac_le_001, frac_eq_pi0ish
    );
}

// -----------------------------------------------------------------------------
// 10) Storey π0 estimation + q-value computation with fixed π0
// -----------------------------------------------------------------------------

fn estimate_pi0_from_reference_grid(p_ref: &[f64], settings: &FdrSettings) -> Option<f64> {
    if p_ref.is_empty() {
        return None;
    }

    let m = p_ref.len() as f64;
    if m < 10.0 {
        return None;
    }

    let mut pi0s: Vec<f64> = Vec::new();

    let mut lambda = settings.storey_lambda_min;
    while lambda <= settings.storey_lambda_max + 1e-12 {
        if !(0.0..1.0).contains(&lambda) {
            lambda += settings.storey_lambda_step;
            continue;
        }
        if lambda < settings.storey_lambda_min_for_agg {
            lambda += settings.storey_lambda_step;
            continue;
        }

        let count_gt = p_ref.iter().filter(|&&p| p > lambda).count() as f64;

        if count_gt == 0.0 || count_gt == m {
            lambda += settings.storey_lambda_step;
            continue;
        }

        let denom = m * (1.0 - lambda);
        if denom <= 0.0 {
            lambda += settings.storey_lambda_step;
            continue;
        }

        let pi0 =
            (count_gt / denom).clamp(settings.storey_pi0_clamp_min, settings.storey_pi0_clamp_max);

        if pi0.is_finite() {
            pi0s.push(pi0);
        }

        lambda += settings.storey_lambda_step;
    }

    if pi0s.len() < 3 {
        return None;
    }

    match settings.storey_pi0_agg {
        crate::input::StoreyPi0Agg::Median => median_f64(pi0s),
        crate::input::StoreyPi0Agg::TrimmedMean => {
            let mut tmp = pi0s;
            trimmed_mean(&mut tmp, 0.20)
        }
    }
}

fn storey_q_value_with_pi0(p_values: &[f64], pi0: f64, settings: &FdrSettings) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    let pi0 = pi0.clamp(0.0, 1.0);

    let mut pv: Vec<(f64, usize)> = p_values
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let p = if p.is_finite() {
                p.clamp(0.0, 1.0)
            } else {
                1.0
            };
            (p, i)
        })
        .collect();

    pv.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let m_f64 = m as f64;

    let mut q_sorted = vec![1.0f64; m];
    for (rank0, (p, _orig_idx)) in pv.iter().enumerate() {
        let rank = (rank0 + 1) as f64;
        let q = (pi0 * (*p) * m_f64 / rank).min(1.0);
        q_sorted[rank0] = q;
    }

    for i in (0..(m - 1)).rev() {
        q_sorted[i] = q_sorted[i].min(q_sorted[i + 1]);
    }

    let mut out = vec![1.0f64; m];
    for (rank0, (_p, orig_idx)) in pv.into_iter().enumerate() {
        out[orig_idx] = q_sorted[rank0];
    }

    let mut out_sorted = out.clone();
    out_sorted.sort_by(|a, b| a.total_cmp(b));
    let med = out_sorted[m / 2];

    let same_as_med = out_sorted
        .iter()
        .filter(|&&q| (q - med).abs() <= settings.storey_degen_eps)
        .count() as f64
        / (m as f64);

    let med_near_pi0 = (med - pi0).abs() <= settings.storey_degen_pi0_eps;

    if same_as_med >= settings.storey_degen_same_as_median_frac && med_near_pi0 {
        log::warn!(
            "DF DEBUG Storey: q-vector degenerate (same_as_med={:.3}, med={:.6} ~ pi0={:.6}); fallback={:?}.",
            same_as_med,
            med,
            pi0,
            settings.storey_degen_fallback
        );

        match settings.storey_degen_fallback {
            crate::input::StoreyDegeneracyFallback::Bh => {
                return crate::ml::stats::bh_q_value(p_values)
            }
            crate::input::StoreyDegeneracyFallback::None => {}
        }
    }

    out
}

// -----------------------------------------------------------------------------
// 11) Work set helper (rank-1 index list)
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct WorkSet {
    rank1_indices: Vec<usize>, // rank==1 AND finite hyperscore
}

impl WorkSet {
    fn build(features: &[DfFeature]) -> Self {
        let rank1_indices: Vec<usize> = features
            .iter()
            .enumerate()
            .filter(|(_, f)| f.core.rank == 1 && tev(f).is_some())
            .map(|(i, _)| i)
            .collect();
        Self { rank1_indices }
    }

    #[inline]
    fn n_rank1(&self) -> usize {
        self.rank1_indices.len()
    }
}

// -----------------------------------------------------------------------------
// 12) Debug helper: summarize rank-1 composition (label/entrap/contam)
// -----------------------------------------------------------------------------

fn log_rank1_composition(features: &[DfFeature], work: &WorkSet, db: &IndexedDatabase) {
    let mut n_rank1 = 0usize;
    let mut n_label1 = 0usize;
    let mut n_ent = 0usize;
    let mut n_cont = 0usize;
    let mut n_ent_label1 = 0usize;
    let mut n_cont_label1 = 0usize;

    for &i in &work.rank1_indices {
        let f = &features[i];
        n_rank1 += 1;

        let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        let ent = is_entrapment_str(&prot);
        let cont = is_contam_str(&prot);

        if f.core.label == 1 {
            n_label1 += 1;
            if ent {
                n_ent_label1 += 1;
            }
            if cont {
                n_cont_label1 += 1;
            }
        }
        if ent {
            n_ent += 1;
        }
        if cont {
            n_cont += 1;
        }
    }

    log::info!(
        "DF DEBUG rank1 composition: rank1={} label1={} ent={} cont={} ent∩label1={} cont∩label1={}",
        n_rank1, n_label1, n_ent, n_cont, n_ent_label1, n_cont_label1
    );
}

// --- STAGE STRUCTS ---
#[derive(Clone, Debug)]
struct RankNullPool {
    // Null pool members (purified) are rank in [min_rank..=max_rank]
    // We keep indices so other models (e.g., Nokoi) can reuse the same null pool.
    fit_data: Vec<(u32, f64, u8)>, // (rank, hyperscore, charge)
    null_indices: Vec<usize>,      // indices into `features`, aligned with fit_data/scores
    scores: Vec<f64>,              // hyperscore scores for global moments/mle fit (aligned)
}

#[derive(Clone)]
struct Engines {
    // Fitted parameters for TEV normalization (no need to carry Gumbel objects)
    mom_mu: f64,
    mom_beta: f64,

    mle_mu: f64,
    mle_beta: f64,

    // LO parameters (charge-stratified TNM model; Step 2.2)
    lo_model: LowerOrderModel,

    // optional engines
    msfdr_model: Option<RobustMsfdrModel>,

    // Nokoi outputs:
    // - nokoi_prob_target: raw model probability P(target) per feature index
    // - nokoi_p_values: p-values derived from empirical null built from rank-null pool
    nokoi_prob_target: Option<Arc<Vec<f64>>>,
    nokoi_p_values: Option<Arc<Vec<f64>>>,
}

// --- BUILD RANK-NULL POOL ---
fn build_rank_null_pool(
    features: &[DfFeature],
    work: &WorkSet,
    settings: &FdrSettings,
) -> Option<RankNullPool> {
    let min_null_size = settings.min_null_size;

    // Null-rank window (used throughout this function)
    let min_rank = settings.min_null_rank;
    let max_rank = settings.max_null_rank;

    // --- PHASE 1: SOFT PURIFIED NULL (same logic we already wrote) ---

    // Build rank1_scores as (peptide_idx, hyperscore)
    let mut rank1_scores: Vec<(u32, f64)> = work
        .rank1_indices
        .iter()
        .filter_map(|&i| {
            let f = &features[i];
            Some((f.core.peptide_idx.0, tev(f)?))
        })
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

    let mut fit_data: Vec<(u32, f64, u8)> = Vec::new();
    let mut null_indices: Vec<usize> = Vec::new();

    for (idx, psm) in features.iter().enumerate() {
        let r: u32 = psm.core.rank as u32;
        if r < min_rank || r > max_rank {
            continue;
        }
        if purified_peptides.contains(&psm.core.peptide_idx.0) {
            continue;
        }
        let s = match tev(psm) {
            Some(v) => v,
            None => continue,
        };
        fit_data.push((r, s, psm.core.charge));
        null_indices.push(idx);
    }

    if fit_data.len() < min_null_size {
        log::warn!("Purified null too small, falling back to unpurified null.");
        fit_data.clear();
        null_indices.clear();

        for (idx, psm) in features.iter().enumerate() {
            let r: u32 = psm.core.rank as u32;
            if r < min_rank || r > max_rank {
                continue;
            }
            let s = match tev(psm) {
                Some(v) => v,
                None => continue,
            };
            fit_data.push((r, s, psm.core.charge));
            null_indices.push(idx);
        }
    }

    // final safety (keep alignment between fit_data and null_indices)
    let mut fit2: Vec<(u32, f64, u8)> = Vec::with_capacity(fit_data.len());
    let mut idx2: Vec<usize> = Vec::with_capacity(null_indices.len());

    for (k, (r, s, z)) in fit_data.into_iter().enumerate() {
        if s.is_finite() {
            fit2.push((r, s, z));
            idx2.push(null_indices[k]);
        }
    }
    let fit_data = fit2;
    let null_indices = idx2;

    if fit_data.len() < min_null_size {
        return None;
    }

    let scores: Vec<f64> = fit_data.iter().map(|(_, s, _)| *s).collect();

    Some(RankNullPool {
        fit_data,
        null_indices,
        scores,
    })
}

#[derive(Clone, Copy, Debug)]
struct RunGates {
    run_msfdr: bool,
    run_nokoi: bool,
}

// --- FIT/PREPARE ENGINES ---
fn fit_engines(
    features: &[DfFeature],
    work: &WorkSet,
    pool: &RankNullPool,
    settings: &FdrSettings,
    gates: RunGates,
) -> Option<Engines> {
    // 1) Moments
    let (mu_mom, beta_mom) = fit_gumbel_moments(&pool.scores);
    let moments_ok = mu_mom.is_finite() && beta_mom.is_finite() && beta_mom > 0.0;
    if !moments_ok {
        return None;
    }

    // Standard Gumbel for TEV-normalized sf inputs (used in Nokoi provisional p gate).
    let std_gumbel = Gumbel::new(0.0, 1.0).expect("standard gumbel");

    // 2) LO: call the new fitter directly
    //
    // Required by the new fitter:
    // - pool.fit_data:           full rank-null pool stream (rank, hyperscore, charge)
    // - rank1_scores_by_charge:  rank-1 stream (hyperscore, charge)
    let mut rank1_scores_by_charge: Vec<(f64, u8)> = Vec::with_capacity(work.rank1_indices.len());
    for &i in &work.rank1_indices {
        let f = &features[i];
        let x = match tev(f) {
            Some(v) => v,
            None => continue,
        };
        rank1_scores_by_charge.push((x, f.core.charge));
    }

    // IMPORTANT: preserve your existing knobs, but apply them INSIDE the LO module
    // as explicit (mu,beta) post-selection transforms (non-paper calibration levers).
    let lo_model = fit_decoy_free_model(
        &pool.fit_data,
        &rank1_scores_by_charge,
        settings.min_null_size,
        settings.min_rank_count,
        settings.lo_beta_blend_moments,
        settings.lo_beta_safety_mult,
    );

    // 3) MLE
    let (mu_mle, beta_mle) = fit_gumbel_mle(&pool.scores).unwrap_or((mu_mom, beta_mom));

    // 4) MSFDR seeding (Phase 4.2)
    // Minimal-delta option: keep MSFDR global, but choose the null seed source via settings.
    let (seed_mu, seed_beta) = match settings.msfdr_seed_mode {
        crate::input::MsfdrSeedMode::Lo => lo_model.fallback_params,
        crate::input::MsfdrSeedMode::PoolMoments => (mu_mom, beta_mom),
        crate::input::MsfdrSeedMode::PoolMle => (mu_mle, beta_mle),
    };
    let seed_beta = seed_beta.max(1e-9);

    let msfdr_model = if gates.run_msfdr {
        let target_scores: Vec<f64> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| tev(&features[i]))
            .collect();
        RobustMsfdrModel::fit(&target_scores, seed_mu, seed_beta)
    } else {
        None
    };

    // 5) Nokoi: pep from probability, p from rank-null pool empirical survival
    let run_nokoi = gates.run_nokoi;

    let mut nokoi_prob_target: Option<Arc<Vec<f64>>> = None;
    let mut nokoi_p_values: Option<Arc<Vec<f64>>> = None;

    if run_nokoi {
        log::info!("Running Nokoi Rescoring ...");

        // Positives:
        // rank==1 AND (hyperscore high enough OR provisional p-value check using moments null)
        //
        // "high enough" threshold uses the same purification logic style:
        // top_k = round(purification_factor * n_rank1), clamped to [5, n_rank1]
        let mut rank1_hs: Vec<f64> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| tev(&features[i]))
            .collect();

        let pos_hyperscore_threshold: f64 = if rank1_hs.len() >= 10 {
            rank1_hs.sort_by(|a, b| b.total_cmp(a));
            let top_k = ((rank1_hs.len() as f64) * settings.purification_factor).round() as usize;
            let top_k = top_k.max(5).min(rank1_hs.len());
            rank1_hs[top_k - 1]
        } else {
            // If too few rank1, rely on provisional p-value only
            f64::INFINITY
        };

        // Provisional p-value threshold (moments null), used when hyperscore isn't in top slice.
        let nokoi_pos_p_thresh: f64 = 0.01;

        let probs = nokoi::rescore_df(
            features,
            0.01, // epsilon (same as before)
            settings.min_null_rank,
            settings.max_null_rank,
            move |f: &DfFeature| {
                if f.core.rank != 1 {
                    return false;
                }
                let x = match tev(f) {
                    Some(v) => v,
                    None => return false,
                };
                if x >= pos_hyperscore_threshold {
                    return true;
                }
                // provisional p-value check (moments null)
                let tev = tev_norm_from_hyperscore(x, mu_mom, beta_mom);
                let p0 = std_gumbel.sf(tev).clamp(0.0, 1.0).max(1e-300);
                p0 <= nokoi_pos_p_thresh
            },
        )?; // returns P(target) per PSM index

        // Contract: probs must align 1:1 with `features`.
        if probs.len() != features.len() {
            log::error!(
                "Nokoi probabilities not aligned: probs.len()={} features.len()={}. Disabling Nokoi.",
                probs.len(),
                features.len()
            );
        } else {
            // 1) Save raw P(target) (for pep_nokoi = 1 - P(target))
            let mut prob = probs;
            for v in &mut prob {
                let vv = if v.is_finite() { *v } else { 0.0 };
                *v = vv.clamp(0.0, 1.0);
            }
            let prob_arc = Arc::new(prob);

            // 2) Build null score distribution from the *rank-null pool* (purified)
            let mut null_scores: Vec<f64> = Vec::with_capacity(pool.null_indices.len());
            for &j in &pool.null_indices {
                // Use Nokoi score = P(target) for null pool members
                null_scores.push(prob_arc[j]);
            }
            null_scores.retain(|x| x.is_finite());
            null_scores.sort_by(|a, b| a.total_cmp(b));

            if null_scores.len() < 10 {
                log::warn!(
                    "Nokoi: rank-null pool too small for null calibration (n_null={}); disabling Nokoi p-values.",
                    null_scores.len()
                );
                // still keep probabilities for pep
                nokoi_prob_target = Some(prob_arc);
            } else {
                // 3) Compute p-values for all PSMs using that null
                // Higher prob_target = better => smaller p (tail on >=)
                let mut p_all = vec![1.0f64; features.len()];
                for (i, &pt) in prob_arc.iter().enumerate() {
                    p_all[i] = empirical_p_from_null_ge(&null_scores, pt);
                }

                // sanitize/clamp once
                for v in &mut p_all {
                    let vv = if v.is_finite() { *v } else { 1.0 };
                    *v = vv.clamp(0.0, 1.0).max(1e-300);
                }

                nokoi_prob_target = Some(prob_arc);
                nokoi_p_values = Some(Arc::new(p_all));
            }
        }
    }

    Some(Engines {
        mom_mu: mu_mom,
        mom_beta: beta_mom,

        mle_mu: mu_mle,
        mle_beta: beta_mle,

        lo_model,

        msfdr_model,
        nokoi_prob_target,
        nokoi_p_values,
    })
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

        // Avoid exact 0.0 p-values (later log / combination safety)
        let std = Gumbel::new(0.0, 1.0).expect("standard gumbel");
        let tev = tev_norm_from_hyperscore(x, self.null_loc, self.null_scale.max(1e-9));
        std.sf(tev).clamp(0.0, 1.0).max(1e-300)
    }

    /// Mixture-model PEP: P(null | x)
    /// Uses the fitted Gumbel null (f0) and skew-normal target (f1) with mixing pi.
    pub fn calculate_pep(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 1.0;
        }

        let null_dist = match Gumbel::new(self.null_loc, self.null_scale.max(1e-9)) {
            Ok(d) => d,
            Err(_) => return 1.0,
        };

        let f0 = null_dist.pdf(x).max(1e-300);
        let f1 =
            skew_normal_pdf(x, self.target_mean, self.target_std, self.target_alpha).max(1e-300);

        let pi = self.pi.clamp(0.01, 0.99);
        let num = (1.0 - pi) * f0;
        let den = num + pi * f1;

        if den.is_finite() && den > 0.0 {
            (num / den).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }
}

// --- MAIN FUNCTION ---
pub fn calculate_q_values(
    psms: &[DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> Vec<DfFeature> {
    let mut new_features = psms.to_vec();

    // ==============================
    // Stage 0 — work set
    // ==============================
    let work = WorkSet::build(&new_features);
    let min_storey_n = settings.min_storey_n;
    let use_ensemble = matches!(settings.model_fit, ModelFit::Ensemble);

    // Centralized gating: only these methods are considered "ran" for output population.
    // (We may still compute intermediate values for internal fallbacks, but we do NOT populate
    // per-method columns unless the method is enabled here.)
    let run_mom = use_ensemble || matches!(settings.model_fit, ModelFit::Moments);
    let run_mle = use_ensemble || matches!(settings.model_fit, ModelFit::Mle);
    let run_lo = use_ensemble || matches!(settings.model_fit, ModelFit::LowerOrder);
    let run_msfdr = use_ensemble || matches!(settings.model_fit, ModelFit::Msfdr);
    let run_nokoi = use_ensemble || matches!(settings.model_fit, ModelFit::Nokoi);

    let gates = RunGates {
        run_msfdr,
        run_nokoi,
    };

    log::info!(
        "DF: rank1_work={} model_fit={:?} ensemble={}",
        work.n_rank1(),
        settings.model_fit,
        use_ensemble
    );

    // ==============================
    // Stage 1 — build rank-null pool
    // ==============================
    let pool = match build_rank_null_pool(&new_features, &work, settings) {
        Some(p) => p,
        None => {
            log::error!("Null distribution too small. Aborting FDR.");
            new_features.par_iter_mut().for_each(|psm| {
                // clear DF fields (do this for *all* ranks to avoid stale values)
                psm.decoy_free_p_value = None;
                psm.decoy_free_pep = None;
                psm.decoy_free_score = None;
                psm.decoy_free_q_value = None;
                psm.decoy_free_peptide_q = None;
                psm.decoy_free_protein_q = None;

                // clear per-method outputs too
                psm.p_mom = None;
                psm.p_mle = None;
                psm.p_lo = None;
                psm.p_msfdr = None;
                psm.p_nokoi = None;

                psm.pep_mom = None;
                psm.pep_mle = None;
                psm.pep_lo = None;
                psm.pep_msfdr = None;
                psm.pep_nokoi = None;

                psm.q_mom = None;
                psm.q_mle = None;
                psm.q_lo = None;
                psm.q_msfdr = None;
                psm.q_nokoi = None;

                // If you prefer “rank1 gets explicit 1.0 Some(...)”, we can re-add that here:
                if psm.core.rank == 1 {
                    psm.decoy_free_p_value = Some(1.0);
                    psm.decoy_free_pep = Some(1.0);
                    psm.decoy_free_score = Some(0.0);
                    psm.decoy_free_q_value = Some(1.0);
                    psm.decoy_free_peptide_q = Some(1.0);
                }
            });
            return new_features;
        }
    };

    log::info!("DF: pool_size={}", pool.fit_data.len());

    // ==============================
    // Stage 2 — fit engines
    // ==============================
    let engines = match fit_engines(&new_features, &work, &pool, settings, gates) {
        Some(e) => e,
        None => {
            log::error!("Invalid null fit. FDR will fail closed.");
            new_features.par_iter_mut().for_each(|psm| {
                set_df_q_value(psm, 1.0);
            });
            return new_features;
        }
    };

    // --- CALCULATION LOOP ---
    // --- capture small config once for rayon closure ---
    let ensemble_p_combiner = settings.ensemble_p_combiner.clone();
    let ensemble_pep_combiner = settings.ensemble_pep_combiner.clone();

    let mom_mu = engines.mom_mu;
    let mom_beta = engines.mom_beta;

    let mle_mu = engines.mle_mu;
    let mle_beta = engines.mle_beta;

    let lo_model = engines.lo_model.clone();

    // Standard Gumbel for TEV-normalized sf inputs
    let std_gumbel = Gumbel::new(0.0, 1.0).expect("standard gumbel");

    let msfdr_model = engines.msfdr_model.clone(); // Option<RobustMsfdrModel>
    let nokoi_prob_target = engines.nokoi_prob_target.clone(); // Option<Arc<Vec<f64>>>
    let nokoi_p_values = engines.nokoi_p_values.clone(); // Option<Arc<Vec<f64>>>

    // Run-level "expert present" gates (keeps Brown-fit and PART A consistent)
    let use_msfdr_expert = run_msfdr && msfdr_model.is_some();
    let use_nokoi_expert = run_nokoi && nokoi_p_values.is_some();

    // ---------------------------------------------------------
    // Brown fit (run-level): estimate dependency across experts
    // ---------------------------------------------------------
    let brown_params: Option<stats::BrownParams> =
        if use_ensemble && matches!(ensemble_p_combiner, EnsemblePCombiner::Brown) {
            // How many observations to use for the covariance fit.
            // Reuse an existing knob; kde_samples is a reasonable proxy.
            let max_obs = settings.kde_samples.max(2000); // keep sane minimum
            let n_rank1 = work.rank1_indices.len();

            if n_rank1 < 50 {
                log::warn!(
                "Brown fit: too few rank1 observations (n={}); skipping Brown and falling back.",
                n_rank1
            );
                None
            } else {
                // Deterministic sampling without RNG:
                // take ~max_obs points evenly spaced through rank1_indices
                let step = ((n_rank1 as f64) / (max_obs as f64)).ceil() as usize;
                let step = step.max(1);

                let mut p_matrix: Vec<Vec<f64>> = Vec::with_capacity((n_rank1 / step).max(1));

                for &psm_idx in work.rank1_indices.iter().step_by(step) {
                    let psm = &new_features[psm_idx];
                    let x = match tev(psm) {
                        Some(v) => v,
                        None => continue,
                    };

                    // Build experts using the SAME gates as PART A (enabled + present)
                    let mut experts: Vec<f64> = Vec::new();

                    // Moments
                    if run_mom {
                        let tev = tev_norm_from_hyperscore(x, mom_mu, mom_beta);
                        experts.push(std_gumbel.sf(tev).clamp(0.0, 1.0).max(1e-300));
                    }

                    // MLE
                    if run_mle {
                        let tev = tev_norm_from_hyperscore(x, mle_mu, mle_beta);
                        experts.push(std_gumbel.sf(tev).clamp(0.0, 1.0).max(1e-300));
                    }

                    // Lower-Order (LO) — global TNM only (no per-PSM ln_ratio shift)
                    if run_lo {
                        let charge = psm.core.charge;
                        experts.push(lo_model.p_value(x, charge).clamp(0.0, 1.0).max(1e-300));
                    }

                    // MSFDR (only if present for this run)
                    if use_msfdr_expert {
                        let m = msfdr_model
                            .as_ref()
                            .expect("use_msfdr_expert implies msfdr_model.is_some()");
                        experts.push(m.calculate_seeded_null_p(x).clamp(0.0, 1.0).max(1e-300));
                    }

                    // Nokoi (only if present for this run)
                    if use_nokoi_expert {
                        let p_vec = nokoi_p_values
                            .as_ref()
                            .expect("use_nokoi_expert implies nokoi_p_values.is_some()");
                        experts.push(
                            p_vec
                                .get(psm_idx)
                                .copied()
                                .unwrap_or(1.0)
                                .clamp(0.0, 1.0)
                                .max(1e-300),
                        );
                    }

                    if !experts.is_empty() {
                        p_matrix.push(experts);
                    }
                }

                let bp = stats::fit_brown_params(&p_matrix);
                if bp.is_none() {
                    log::warn!("Brown fit: failed; using Fisher fallback.");
                }
                bp
            }
        } else {
            None
        };

    // --- PART A: Compute in Parallel (Read-Only) ---
    // We compute everything first and store it in a temporary list (rank1_out)
    let rank1_out: Vec<Rank1Computed> = work
        .rank1_indices
        .par_iter()
        .filter_map(|&idx| {
            let psm = &new_features[idx]; // Immutable borrow - safe for Rayon
            let x = tev(psm)?;

            // 1. Calculate base Null P-values (STRICT compute gating)
            let p_mom = if run_mom {
                let tev = tev_norm_from_hyperscore(x, mom_mu, mom_beta);
                std_gumbel.sf(tev).clamp(0.0, 1.0).max(1e-300)
            } else {
                1.0
            };

            let p_mle = if run_mle {
                let tev = tev_norm_from_hyperscore(x, mle_mu, mle_beta);
                std_gumbel.sf(tev).clamp(0.0, 1.0).max(1e-300)
            } else {
                1.0
            };

            // 2. Lower-Order (LO) adjustment (STRICT compute gating)
            let p_lo = if run_lo {
                let charge = psm.core.charge;
                lo_model.p_value(x, charge).clamp(0.0, 1.0).max(1e-300)
            } else {
                1.0
            };

            // 3. Optional Models (MSFDR and Nokoi)
            // STRICT compute gating: only compute when the method is enabled for this run
            // AND the required model/vector is present.
            let (p_msfdr, pep_msfdr) = if use_msfdr_expert {
                let m = msfdr_model
                    .as_ref()
                    .expect("use_msfdr_expert implies msfdr_model.is_some()");
                let p = m.calculate_seeded_null_p(x).clamp(0.0, 1.0).max(1e-300);
                let pep = m.calculate_pep(x).clamp(0.0, 1.0).max(1e-300);
                (Some(p), Some(pep))
            } else {
                (None, None)
            };

            let (p_nokoi, pep_nokoi) = if run_nokoi {
                // pep from probability if available (still gated by run_nokoi)
                let pep = if let Some(prob) = nokoi_prob_target.as_ref() {
                    let pt = prob.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    Some((1.0 - pt).clamp(0.0, 1.0).max(1e-300))
                } else {
                    None
                };

                // p from rank-null-calibrated p-values ONLY if this expert is usable this run
                let p = if use_nokoi_expert {
                    let p_vec = nokoi_p_values
                        .as_ref()
                        .expect("use_nokoi_expert implies nokoi_p_values.is_some()");
                    Some(
                        p_vec
                            .get(idx)
                            .copied()
                            .unwrap_or(1.0)
                            .clamp(0.0, 1.0)
                            .max(1e-300),
                    )
                } else {
                    None
                };

                (p, pep)
            } else {
                (None, None)
            };

            // 4. Ensemble Logic: Combine Experts (build list from run flags)
            let mut experts: Vec<f64> = Vec::new();
            if run_mom {
                experts.push(p_mom);
            }
            if run_mle {
                experts.push(p_mle);
            }
            if run_lo {
                experts.push(p_lo);
            }
            if use_msfdr_expert {
                experts.push(p_msfdr.unwrap_or(1.0));
            }
            if use_nokoi_expert {
                experts.push(p_nokoi.unwrap_or(1.0));
            }

            let p_final = if use_ensemble {
                // If an expert didn't run / didn't produce p, it is simply absent.
                // If somehow empty (shouldn't happen), fail-closed.
                if experts.is_empty() {
                    1.0
                } else {
                    combine_p_values(&experts, ensemble_p_combiner.clone(), brown_params)
                }
            } else {
                match settings.model_fit {
                    ModelFit::Moments => p_mom,
                    ModelFit::Mle => p_mle,
                    ModelFit::LowerOrder => p_lo,
                    ModelFit::Msfdr => p_msfdr.unwrap_or(1.0), // FAIL-CLOSED
                    ModelFit::Nokoi => p_nokoi.unwrap_or(1.0), // FAIL-CLOSED
                    _ => 1.0,                                  // FAIL-CLOSED
                }
            }
            .clamp(0.0, 1.0)
            .max(1e-300);

            // 5. PEP Calculation
            // Step 3 contract:
            // - null-only methods: pep == p (until we implement a real pep model)
            // - MSFDR: use mixture pep
            // - Nokoi: pep == p for now (until Step 6)
            let pep_mom = p_mom;
            let pep_mle = p_mle;
            let pep_lo = p_lo;

            let pep_final = if use_ensemble {
                let mut pep_experts: Vec<f64> = Vec::new();
                if run_mom {
                    pep_experts.push(pep_mom);
                }
                if run_mle {
                    pep_experts.push(pep_mle);
                }
                if run_lo {
                    pep_experts.push(pep_lo);
                }
                if use_msfdr_expert {
                    // pep_msfdr is only computed when use_msfdr_expert is true
                    pep_experts.push(pep_msfdr.unwrap_or(1.0));
                }
                if run_nokoi {
                    if let Some(v) = pep_nokoi {
                        pep_experts.push(v);
                    }
                }

                if pep_experts.is_empty() {
                    1.0
                } else {
                    combine_peps(&pep_experts, ensemble_pep_combiner.clone())
                }
            } else {
                match settings.model_fit {
                    ModelFit::Moments => pep_mom,
                    ModelFit::Mle => pep_mle,
                    ModelFit::LowerOrder => pep_lo,
                    ModelFit::Msfdr => pep_msfdr.unwrap_or(1.0), // FAIL-CLOSED
                    ModelFit::Nokoi => pep_nokoi.unwrap_or(1.0), // FAIL-CLOSED
                    _ => 1.0,                                    // FAIL-CLOSED
                }
            }
            .clamp(0.0, 1.0)
            .max(1e-300);

            let df_score = (-10.0 * (pep_final).max(1e-15).log10()) as f32;

            Some(Rank1Computed {
                idx,
                p_mom,
                p_mle,
                p_lo,
                p_msfdr,
                p_nokoi,
                pep_mom,
                pep_mle,
                pep_lo,
                pep_msfdr,
                pep_nokoi,
                p_final,
                pep_final,
                df_score,
            })
        })
        .collect();

    // --- PART B: Write back to new_features ---
    // Now that Part A is finished, we have a list of updates.
    // We can iterate and apply them without borrow conflicts.
    for r in rank1_out {
        let psm = &mut new_features[r.idx];

        // Populate per-method outputs ONLY if the method is enabled by the centralized gate.
        // (Otherwise leave as None to make "didn't run" unambiguous.)
        psm.p_mom = if run_mom { Some(r.p_mom as f32) } else { None };
        psm.p_mle = if run_mle { Some(r.p_mle as f32) } else { None };
        psm.p_lo = if run_lo { Some(r.p_lo as f32) } else { None };

        psm.p_msfdr = if run_msfdr {
            r.p_msfdr.map(|v| v as f32)
        } else {
            None
        };
        psm.p_nokoi = if run_nokoi {
            r.p_nokoi.map(|v| v as f32)
        } else {
            None
        };

        psm.pep_mom = if run_mom {
            Some(r.pep_mom as f32)
        } else {
            None
        };
        psm.pep_mle = if run_mle {
            Some(r.pep_mle as f32)
        } else {
            None
        };
        psm.pep_lo = if run_lo { Some(r.pep_lo as f32) } else { None };

        psm.pep_msfdr = if run_msfdr {
            r.pep_msfdr.map(|v| v as f32)
        } else {
            None
        };
        psm.pep_nokoi = if run_nokoi {
            r.pep_nokoi.map(|v| v as f32)
        } else {
            None
        };

        // Final DF outputs ALWAYS populated (rank1-only by design)
        set_df_p_value(psm, r.p_final as f32); // sets decoy_free_p_value
        psm.decoy_free_pep = Some(r.pep_final as f32);
        psm.decoy_free_score = Some(r.df_score);
    }

    // --- PART C: Clear Non-Rank1 (Cheap safety pass) ---
    // We only compute/define DF outputs for rank==1. Anything else must be scrubbed
    // to avoid stale values leaking into downstream tables / plots.
    new_features.par_iter_mut().for_each(|psm| {
        if psm.core.rank != 1 {
            // Rank!=1: DF outputs are undefined by contract; scrub to None (fail-closed via accessors).
            psm.decoy_free_p_value = None;
            psm.decoy_free_pep = None;
            psm.decoy_free_score = None;
            psm.decoy_free_q_value = None;
            psm.decoy_free_peptide_q = None;
            psm.decoy_free_protein_q = None;

            // --- per-method p/pep/q outputs (rank1-only by design in Step 3) ---
            psm.p_mom = None;
            psm.p_mle = None;
            psm.p_lo = None;
            psm.p_msfdr = None;
            psm.p_nokoi = None;

            psm.pep_mom = None;
            psm.pep_mle = None;
            psm.pep_lo = None;
            psm.pep_msfdr = None;
            psm.pep_nokoi = None;

            psm.q_mom = None;
            psm.q_mle = None;
            psm.q_lo = None;
            psm.q_msfdr = None;
            psm.q_nokoi = None;
        }
    });

    // --- CALIBRATION (replaces old global hyperscore PAVA) ---
    // We calibrate the *chosen* p-value (currently stored in spectrum_q / decoy_free_p_value)
    // using a ranking key appropriate to the chosen method.
    // This is the structural fix for LO conservatism when LO != hyperscore-ranked.
    {
        // Helper: apply isotonic regression to a given (quality, idx, p) stream.
        // quality: larger = better (sorted descending)
        let calibrate = |mut rows: Vec<(f64, usize, f64)>| -> Vec<(usize, f64)> {
            if rows.is_empty() {
                return Vec::new();
            }
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut p_sorted: Vec<f64> = rows.iter().map(|r| r.2).collect();
            isotonic_regression_increasing(&mut p_sorted);

            rows.iter()
                .enumerate()
                .map(|(k, r)| (r.1, p_sorted[k].clamp(0.0, 1.0).max(1e-300)))
                .collect()
        };

        // LO rank-key helper aligned to LO p-value computation.
        // Current LO implementation uses TEV ordering (larger TEV = better).
        // If LO ranking ever changes, implement that change here.
        let lo_rank_key = |f: &DfFeature| -> f64 { tev(f).unwrap_or(f64::NEG_INFINITY) };

        // Build rows for chosen-output calibration.
        let rows: Vec<(f64, usize, f64)> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                let f = &new_features[i];

                // p_used must reflect the chosen method’s p-value (pre-q-value)
                let p_used: f64 = match settings.model_fit {
                    ModelFit::Moments => f.p_mom.map(|v| v as f64)?,
                    ModelFit::Mle => f.p_mle.map(|v| v as f64)?,
                    ModelFit::LowerOrder => f.p_lo.map(|v| v as f64)?,
                    ModelFit::Msfdr => f.p_msfdr.map(|v| v as f64)?,
                    ModelFit::Nokoi => f.p_nokoi.map(|v| v as f64)?,
                    ModelFit::Ensemble => f.decoy_free_p_value.map(|v| v as f64)?, // p_final written earlier
                };

                // quality key: larger = better (sorted descending)
                let quality: f64 = match settings.model_fit {
                    ModelFit::Moments | ModelFit::Mle => tev(f).unwrap_or(f64::NEG_INFINITY),

                    ModelFit::LowerOrder => match settings.lo_rank_key {
                        LoRankKey::Hyperscore => f.core.hyperscore as f64,
                        LoRankKey::LoAdjusted => lo_rank_key(f),
                    },

                    // method-aligned ranking:
                    ModelFit::Msfdr => neg_log10_p(p_used),
                    ModelFit::Nokoi => neg_log10_p(p_used),

                    // ensemble: use the ensemble’s own evidence scale (df_score is monotone w/ pep_final)
                    ModelFit::Ensemble => f.decoy_free_score.unwrap_or(0.0) as f64,
                };

                Some((quality, i, p_used.clamp(0.0, 1.0).max(1e-300)))
            })
            .collect();

        // Write calibrated chosen p-values back into DF p-value stream
        for (i, pcal) in calibrate(rows) {
            set_df_p_value(&mut new_features[i], pcal as f32); // sets decoy_free_p_value
        }
    }

    // --- OPTIONAL: PER-METHOD PAVA CALIBRATION ---
    // Calibrate p_* independently using a method-appropriate sort key.
    if settings.calibrate_per_method {
        // Helper closure: apply isotonic regression to a given (quality, idx, p) stream
        let calibrate = |mut rows: Vec<(f64, usize, f64)>| -> Vec<(usize, f64)> {
            if rows.is_empty() {
                return Vec::new();
            }
            // Sort by quality descending (best first)
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut p_sorted: Vec<f64> = rows.iter().map(|r| r.2).collect();
            isotonic_regression_increasing(&mut p_sorted);

            rows.iter()
                .enumerate()
                .map(|(k, r)| (r.1, p_sorted[k].clamp(0.0, 1.0).max(1e-300)))
                .collect()
        };

        // LO rank-key helper aligned to LO p-value computation (TEV ordering).
        let lo_rank_key = |f: &DfFeature| -> f64 { tev(f).unwrap_or(f64::NEG_INFINITY) };

        // Moments: rank by hyperscore
        {
            let rows: Vec<(f64, usize, f64)> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| {
                    let f = &new_features[i];
                    let p = f.p_mom? as f64;
                    Some((tev(f).unwrap_or(f64::NEG_INFINITY), i, p))
                })
                .collect();

            for (i, pcal) in calibrate(rows) {
                new_features[i].p_mom = Some(pcal as f32);
                // null-only pep follows p (PepEqualsP for now)
                new_features[i].pep_mom = Some(pcal as f32);
            }
        }

        // MLE: rank by hyperscore
        {
            let rows: Vec<(f64, usize, f64)> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| {
                    let f = &new_features[i];
                    let p = f.p_mle? as f64;
                    Some((tev(f).unwrap_or(f64::NEG_INFINITY), i, p))
                })
                .collect();

            for (i, pcal) in calibrate(rows) {
                new_features[i].p_mle = Some(pcal as f32);
                new_features[i].pep_mle = Some(pcal as f32);
            }
        }

        // LO: rank key is configurable (Hyperscore vs LO-adjusted)
        {
            let rows: Vec<(f64, usize, f64)> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| {
                    let f = &new_features[i];
                    let p = f.p_lo? as f64;
                    let quality = match settings.lo_rank_key {
                        LoRankKey::Hyperscore => f.core.hyperscore as f64,
                        LoRankKey::LoAdjusted => lo_rank_key(f),
                    };
                    Some((quality, i, p))
                })
                .collect();

            for (i, pcal) in calibrate(rows) {
                new_features[i].p_lo = Some(pcal as f32);
                new_features[i].pep_lo = Some(pcal as f32);
            }
        }

        // MSFDR: rank by -log10(p_msfdr) (method-aligned)
        {
            let rows: Vec<(f64, usize, f64)> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| {
                    let f = &new_features[i];
                    let p = f.p_msfdr? as f64;
                    Some((neg_log10_p(p), i, p))
                })
                .collect();

            for (i, pcal) in calibrate(rows) {
                new_features[i].p_msfdr = Some(pcal as f32);
                // Keep msfdr PEP as the mixture PEP (do not overwrite pep_msfdr with pcal)
            }
        }

        // Nokoi: rank by -log10(p_nokoi)
        {
            let rows: Vec<(f64, usize, f64)> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| {
                    let f = &new_features[i];
                    let p = f.p_nokoi? as f64;
                    Some((neg_log10_p(p), i, p))
                })
                .collect();

            for (i, pcal) in calibrate(rows) {
                new_features[i].p_nokoi = Some(pcal as f32);
                // IMPORTANT: keep pep_nokoi from Nokoi probability (do NOT overwrite)
            }
        }
    }

    // --- Q-VALUES ---
    let rank1_p: Vec<f64> = work
        .rank1_indices
        .iter()
        .map(|&i| {
            (df_p_value(&new_features[i]) as f64)
                .clamp(0.0, 1.0)
                .max(1e-300)
        })
        .collect();

    // Diagnostics (3 logs)
    log_rank1_composition(&new_features, &work, db);
    summarize_pvec("rank1_p (chosen stream, pre-q)", &rank1_p);

    // Build a clean reference set for pi0: target label==1, excluding ENT + CONT.
    // This does NOT change entrapment validation later; it only makes pi0 estimation sane.
    let rank1_p_ref: Vec<f64> = work
        .rank1_indices
        .iter()
        .filter_map(|&i| {
            let f = &new_features[i];
            if f.core.label != 1 {
                return None;
            }
            let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            if is_contam_str(&prot) {
                return None;
            }
            if is_entrapment_str(&prot) {
                return None;
            }
            let p = (df_p_value(f) as f64).clamp(0.0, 1.0).max(1e-300);
            p.is_finite().then_some(p)
        })
        .collect();

    summarize_pvec("rank1_p_ref (targets, non-ENT, non-CONT)", &rank1_p_ref);

    summarize_q(
        "PREQ rank1 (chosen p-stream)",
        work.rank1_indices
            .iter()
            .map(|&i| df_p_value(&new_features[i])),
    );

    let q_values = match settings.type_ {
        FdrType::Bh => stats::bh_q_value(&rank1_p),

        FdrType::Storey => {
            let pi0_opt = estimate_pi0_from_reference_grid(&rank1_p_ref, settings);

            match pi0_opt {
                Some(pi0) => {
                    log::info!(
                        "DF DEBUG Storey(grid): using pi0={:.4} from rank1_p_ref (m_ref={})",
                        pi0,
                        rank1_p_ref.len()
                    );
                    storey_q_value_with_pi0(&rank1_p, pi0, settings)
                }
                None => {
                    log::warn!(
                        "DF DEBUG Storey(grid): degenerate pi0 on reference set (m_ref={}), falling back to BH for chosen stream.",
                        rank1_p_ref.len()
                    );
                    stats::bh_q_value(&rank1_p)
                }
            }
        }
    };

    for (&idx, q) in work.rank1_indices.iter().zip(q_values) {
        set_df_q_value(&mut new_features[idx], q as f32);
        new_features[idx].decoy_free_q_value = Some(q as f32);
    }

    // --- POST-Q SUMMARIES (diagnostic) ---
    // Summary over rank-1 targets (whatever you treat as “targets” in DF mode)
    summarize_q(
        "POSTQ rank1(label==1)",
        new_features
            .iter()
            .filter(|f| f.core.rank == 1 && f.core.label == 1)
            .map(|f| df_q_value(f)),
    );

    // Summary over ALL rank-1 (includes entrap/contam; useful to spot mass-flatlines)
    summarize_q(
        "POSTQ rank1(all labels)",
        new_features
            .iter()
            .filter(|f| f.core.rank == 1)
            .map(|f| df_q_value(f)),
    );

    // --- POST-Q SUMMARY: Pi0 Reference Set ---
    // This uses the EXACT same logic used to build rank1_p_ref for the Storey pi0 estimate.
    summarize_q(
        "POSTQ rank1_p_ref (same subset as pi0)",
        new_features
            .iter()
            .filter(|f| {
                if f.core.rank != 1 || f.core.label != 1 {
                    return false;
                }
                let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
                if is_contam_str(&prot) || is_entrapment_str(&prot) {
                    return false;
                }
                true
            })
            .map(|f| df_q_value(f)),
    );

    // --- PER-METHOD Q-VALUES (rank-1 only) ---
    // Compute q_* from each method's p_* vector.
    // IMPORTANT: Do NOT fabricate p-values for disabled methods (MSFDR/Nokoi).
    // If a method never produced p-values (all None), skip q computation entirely for it.

    // Build p-vectors only for methods that actually produced p-values.
    // No defaults, no "pretend it ran".
    let mut mom_pos: Vec<usize> = Vec::new();
    let mut p_mom_present: Vec<f64> = Vec::new();

    let mut mle_pos: Vec<usize> = Vec::new();
    let mut p_mle_present: Vec<f64> = Vec::new();

    let mut lo_pos: Vec<usize> = Vec::new();
    let mut p_lo_present: Vec<f64> = Vec::new();

    // Optional methods: store only present values + where to write them back.
    let mut msfdr_pos: Vec<usize> = Vec::new();
    let mut p_msfdr_present: Vec<f64> = Vec::new();

    let mut nokoi_pos: Vec<usize> = Vec::new();
    let mut p_nokoi_present: Vec<f64> = Vec::new();

    // Per-method pi0 reference sets (label==1, non-ENT, non-CONT), aligned to each method's p-values.
    let mut p_mom_ref: Vec<f64> = Vec::new();
    let mut p_mle_ref: Vec<f64> = Vec::new();
    let mut p_lo_ref: Vec<f64> = Vec::new();
    let mut p_msfdr_ref: Vec<f64> = Vec::new();
    let mut p_nokoi_ref: Vec<f64> = Vec::new();

    for (k, &i) in work.rank1_indices.iter().enumerate() {
        let f = &new_features[i];

        // Reference-set membership for pi0 estimation
        let is_ref = if f.core.label != 1 {
            false
        } else {
            let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            !is_contam_str(&prot) && !is_entrapment_str(&prot)
        };

        if let Some(v) = f.p_mom {
            let p = (v as f64).clamp(0.0, 1.0).max(1e-300);
            mom_pos.push(k);
            p_mom_present.push(p);
            if is_ref {
                p_mom_ref.push(p);
            }
        }
        if let Some(v) = f.p_mle {
            let p = (v as f64).clamp(0.0, 1.0).max(1e-300);
            mle_pos.push(k);
            p_mle_present.push(p);
            if is_ref {
                p_mle_ref.push(p);
            }
        }
        if let Some(v) = f.p_lo {
            let p = (v as f64).clamp(0.0, 1.0).max(1e-300);
            lo_pos.push(k);
            p_lo_present.push(p);
            if is_ref {
                p_lo_ref.push(p);
            }
        }

        if let Some(v) = f.p_msfdr {
            let p = (v as f64).clamp(0.0, 1.0).max(1e-300);
            msfdr_pos.push(k);
            p_msfdr_present.push(p);
            if is_ref {
                p_msfdr_ref.push(p);
            }
        }
        if let Some(v) = f.p_nokoi {
            let p = (v as f64).clamp(0.0, 1.0).max(1e-300);
            nokoi_pos.push(k);
            p_nokoi_present.push(p);
            if is_ref {
                p_nokoi_ref.push(p);
            }
        }
    }

    // Compute q-vectors only for non-empty present sets.
    // IMPORTANT: Storey uses the SAME grid-π0 codepath as the chosen DF stream.
    let compute_q_present = |p_present: &Vec<f64>, p_ref: &Vec<f64>| -> Vec<f64> {
        if p_present.is_empty() {
            return Vec::new();
        }
        match settings.type_ {
            FdrType::Bh => stats::bh_q_value(p_present),

            FdrType::Storey => {
                // Enforce minimum reference size (old API used min_storey_n as a safety brake).
                if p_ref.len() < min_storey_n {
                    return stats::bh_q_value(p_present);
                }
                match estimate_pi0_from_reference_grid(p_ref, settings) {
                    Some(pi0) => storey_q_value_with_pi0(p_present, pi0, settings),
                    None => stats::bh_q_value(p_present),
                }
            }
        }
    };

    let q_mom_present: Vec<f64> = compute_q_present(&p_mom_present, &p_mom_ref);
    let q_mle_present: Vec<f64> = compute_q_present(&p_mle_present, &p_mle_ref);
    let q_lo_present: Vec<f64> = compute_q_present(&p_lo_present, &p_lo_ref);
    let q_msfdr_present: Vec<f64> = compute_q_present(&p_msfdr_present, &p_msfdr_ref);
    let q_nokoi_present: Vec<f64> = compute_q_present(&p_nokoi_present, &p_nokoi_ref);

    // Sparse write-back: only where p_* existed.
    for (j, &k) in mom_pos.iter().enumerate() {
        let i = work.rank1_indices[k];
        new_features[i].q_mom = Some(q_mom_present[j] as f32);
    }
    for (j, &k) in mle_pos.iter().enumerate() {
        let i = work.rank1_indices[k];
        new_features[i].q_mle = Some(q_mle_present[j] as f32);
    }
    for (j, &k) in lo_pos.iter().enumerate() {
        let i = work.rank1_indices[k];
        new_features[i].q_lo = Some(q_lo_present[j] as f32);
    }
    for (j, &k) in msfdr_pos.iter().enumerate() {
        let i = work.rank1_indices[k];
        new_features[i].q_msfdr = Some(q_msfdr_present[j] as f32);
    }
    for (j, &k) in nokoi_pos.iter().enumerate() {
        let i = work.rank1_indices[k];
        new_features[i].q_nokoi = Some(q_nokoi_present[j] as f32);
    }

    new_features
}

pub fn calculate_peptide_q_df(
    features: &mut [DfFeature],
    db: &IndexedDatabase,
    threshold: f32,
) -> usize {
    let mut best_q: FnvHashMap<String, f32> = FnvHashMap::default();

    // DF aggregation contract:
    // - rank1 only
    // - read ONLY decoy-free q stream
    for feat in features.iter().filter(|f| f.core.rank == 1) {
        let q = df_q_value(feat); // <-- DF stream accessor (decoy_free_q_value)
        let peptide = db[feat.core.peptide_idx].to_string();

        best_q
            .entry(peptide)
            .and_modify(|v| *v = v.min(q))
            .or_insert(q);
    }

    // Write ONLY DF peptide q (rank1-only by contract)
    for feat in features.iter_mut() {
        if feat.core.rank != 1 {
            // DF peptide_q is undefined for rank!=1; scrub to avoid leakage.
            feat.decoy_free_peptide_q = None;
            continue;
        }

        let peptide = db[feat.core.peptide_idx].to_string();
        if let Some(q) = best_q.get(&peptide) {
            feat.decoy_free_peptide_q = Some(*q);
        } else {
            // If peptide never seen in rank1 (should be rare), fail-closed.
            feat.decoy_free_peptide_q = Some(1.0);
        }
    }

    best_q.values().filter(|&&q| q <= threshold).count()
}

// DF aggregation/write-back contract: rank1-only; rank!=1 must be None to prevent stale leakage.
pub fn calculate_protein_q_df(
    features: &mut [DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
) -> usize {
    // Protein -> (peptide -> best_p) using DF-only p stream
    let mut protein_peptide_map: FnvHashMap<String, FnvHashMap<String, f64>> =
        FnvHashMap::default();

    for feat in features.iter().filter(|f| f.core.rank == 1) {
        // DF aggregation contract: read ONLY decoy_free_p_value stream
        // Use accessor (fail-closed to 1.0), but SKIP if the underlying field is absent,
        // so we don't inject conservative 1.0s into Fisher combining.
        if feat.decoy_free_p_value.is_none() {
            continue;
        }
        let p = (df_p_value(feat) as f64).clamp(0.0, 1.0).max(1e-300);

        let protein_key = db[feat.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        let peptide_seq = db[feat.core.peptide_idx].to_string();

        let peptide_map = protein_peptide_map.entry(protein_key).or_default();
        peptide_map
            .entry(peptide_seq)
            .and_modify(|v| *v = v.min(p))
            .or_insert(p);
    }

    // Combine per protein -> p-value
    let mut protein_keys: Vec<String> = Vec::new();
    let mut protein_p_values: Vec<f64> = Vec::new();

    for (key, peptide_map) in protein_peptide_map {
        // Fisher combine across best-per-peptide p-values
        let p_vec: Vec<f64> = peptide_map.values().copied().collect();
        if p_vec.is_empty() {
            continue;
        }

        let combined_p = stats::combine_fisher(&p_vec).clamp(0.0, 1.0).max(1e-300);
        protein_keys.push(key);
        protein_p_values.push(combined_p);
    }

    // If no proteins, write fail-closed (rank1-only by contract) and return 0.
    if protein_p_values.is_empty() {
        for feat in features.iter_mut() {
            if feat.core.rank != 1 {
                // DF protein_q undefined for rank!=1; scrub to avoid leakage.
                feat.decoy_free_protein_q = None;
            } else {
                feat.decoy_free_protein_q = Some(1.0);
            }
        }
        return 0;
    }

    // Convert protein p-values -> protein q-values (DF-only output)
    //
    // Consistency fix: use the SAME Storey path as the DF PSM stream:
    // - estimate pi0 from a cleaner reference subset
    // - compute Storey q-values with fixed pi0
    // - detect degeneracy and fallback (handled inside storey_q_value_with_pi0)
    let protein_q_values = match settings.type_ {
        FdrType::Bh => stats::bh_q_value(&protein_p_values),

        FdrType::Storey => {
            // Reference set for pi0 estimation: exclude contaminants + entrapment proteins.
            // (Do NOT change entrapment validation—this only stabilizes pi0 estimation.)
            let mut protein_p_ref: Vec<f64> = Vec::new();
            for (key, &p) in protein_keys.iter().zip(protein_p_values.iter()) {
                if is_contam_str(key) {
                    continue;
                }
                if is_entrapment_str(key) {
                    continue;
                }
                if p.is_finite() {
                    protein_p_ref.push(p.clamp(0.0, 1.0).max(1e-300));
                }
            }

            // Enforce the same minimum reference size guard you use elsewhere.
            if protein_p_ref.len() < settings.min_storey_n {
                stats::bh_q_value(&protein_p_values)
            } else {
                match estimate_pi0_from_reference_grid(&protein_p_ref, settings) {
                    Some(pi0) => storey_q_value_with_pi0(&protein_p_values, pi0, settings),
                    None => stats::bh_q_value(&protein_p_values),
                }
            }
        }
    };

    // Map back: protein_key -> q
    let mut best_q: FnvHashMap<String, f32> = FnvHashMap::default();
    for (key, q) in protein_keys.into_iter().zip(protein_q_values) {
        best_q.insert(key, q as f32);
    }

    // Write ONLY DF protein q (rank1-only by contract; scrub others)
    for feat in features.iter_mut() {
        if feat.core.rank != 1 {
            // DF protein_q undefined for rank!=1; scrub to avoid leakage.
            feat.decoy_free_protein_q = None;
            continue;
        }

        let protein_key = db[feat.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        let q = *best_q.get(&protein_key).unwrap_or(&1.0);
        feat.decoy_free_protein_q = Some(q);
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
    // In decoy-free LFQ, (is_decoy==true) are *shadow/off-target* samples (target-derived null).
    let mut null_scores: Vec<f64> = peaks
        .iter()
        .filter_map(|((_, is_decoy), (peak, _))| {
            if *is_decoy && peak.score.is_finite() {
                Some(peak.score)
            } else {
                None
            }
        })
        .collect();

    if null_scores.len() < 200 {
        // Too few shadow samples -> do not assign q-values (fail closed)
        return 0;
    }

    // Robustify: winsorize upper tail so occasional real peaks in shadow channel
    // don't dominate the null fit.
    null_scores.sort_by(|a, b| a.total_cmp(b));
    let n = null_scores.len();
    let p95_idx = ((n as f64 - 1.0) * 0.95).round() as usize;
    let cap = null_scores[p95_idx].max(1e-12);

    // Apply cap (winsorization)
    for v in &mut null_scores {
        if *v > cap {
            *v = cap;
        }
    }

    let (mu, beta_raw) = fit_gumbel_moments(&null_scores);
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
