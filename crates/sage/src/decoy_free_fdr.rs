/* =============================================================================
Decoy-Free (DF) FDR Contract — AUTHORITATIVE ORACLE (Phase 0 / Step 0.1)

This comment is the invariants contract for Decoy-Free mode. If code and this
contract disagree, the code is wrong.

A) Rank constraint (HARD)
-------------------------
1) DF is rank==1 only.
   - Any DF output (decoy_free_*) is defined ONLY for rank==1 PSMs.
   - For rank!=1: ALL DF outputs AND ALL DF per-method outputs MUST be scrubbed
     to None (no stale leakage).

B) Output separation (HARD)
---------------------------
2) DF “final selected outputs” are ONLY the decoy_free_* streams:
     decoy_free_p_value / decoy_free_pep / decoy_free_score / decoy_free_q_value
     decoy_free_peptide_q / decoy_free_protein_q
   All other DF family outputs are auxiliary streams and MUST NOT be treated as
   “the answer”:
     - p_msfdr / pep_msfdr / q_msfdr         (seeded / legacy stream)
     - p_1smix / pep_1smix / q_1smix         (aux stream)
     - p_2smix / pep_2smix / q_2smix         (aux stream)
     - p_mom / pep_mom / q_mom, etc.         (aux expert streams)

C) Q-value semantics for MSFDR variants (HARD)
----------------------------------------------
3) MSFDR variant q-values are ALWAYS PEP-derived cumulative means.
   - q_1smix and q_2smix MUST be computed as:
        q[k] = mean(pep[0..k]) after sorting by “quality” (best-first),
     followed by a monotone (non-decreasing with worsening quality) pass.
   - They MUST NOT use BH or Storey (regardless of settings.type_).
   - (“seeded/legacy” q_msfdr is allowed to follow global settings.type_ only if
      explicitly intended; q_1smix/q_2smix are NEVER BH/Storey.)

D) Failure mode (FAIL-CLOSED; matches current implementation)
-------------------------------------------------------------
4) If DF engines do not exist (None) / fail to fit / do not run:
   - All DF outputs must be cleared (None) to prevent stale leakage.
   - Current implementation CLARIFICATION (intentional):
       When engine fitting fails, rank==1 may be populated with conservative
       defaults (p=1, pep=1, score=0, q=1, peptide_q=1, protein_q=1) so downstream
       consumers fail-closed rather than treating missing values as permissive.
     Rank!=1 MUST remain None always.

E) Sorting key definition (“sorted by score”) (HARD)
----------------------------------------------------
5) Anywhere this module says “sorted by score/quality (best-first)”, the key is:
   - For mixture q (q_1smix / q_2smix): sort by descending SCORE where:
        SCORE = tev(feature) if present, else decoy_free_score.
     (Higher SCORE means better evidence / earlier in the list.)
   - For PAVA calibration of a chosen p-stream: sort by the method-aligned
     “quality” key used in the calibration block (see the calibration section),
     NOT by raw hyperscore unless that is explicitly selected (e.g., LO rank key).

============================================================================= */

use crate::database::IndexedDatabase;
use crate::input::{EnsemblePCombiner, EnsemblePepCombiner, LoRankKey};
use crate::input::{FdrSettings, FdrType, ModelFit};
use crate::input::{LoScore, LoStratify};
use crate::lfq::{Peak, PrecursorId};
use crate::ml::lower_order::{
    fit_decoy_free_model, fit_gumbel_mle, fit_gumbel_moments, LowerOrderModel,
};
use crate::ml::msfdr::MsfdrParamTuple;
use crate::ml::msfdr::{Msfdr1SmixModel, Msfdr2SmixModel, MsfdrSeededModel};
use crate::ml::nokoi;
use crate::ml::stats;
use crate::scoring::DfFeature;
use fnv::{FnvHashMap, FnvHashSet};
use rayon::prelude::*;
use statrs::distribution::{Beta, ContinuousCDF, Gumbel, Normal};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Rank1Computed {
    idx: usize,

    // per-method p's
    p_mom: f64,
    p_mle: f64,
    p_lo: f64,

    // MSFDR family p's (independent streams)
    p_msfdr: Option<f64>, // seeded/legacy stream
    p_1smix: Option<f64>,
    p_2smix: Option<f64>,

    p_nokoi: Option<f64>,

    // MSFDR family peps (model-derived)
    pep_msfdr: Option<f64>, // seeded/legacy stream
    pep_1smix: Option<f64>,
    pep_2smix: Option<f64>,

    // Nokoi pep (from 1 - P(target))
    pep_nokoi: Option<f64>,

    // final DF p output (pep/score computed later, after lfdr mapping)
    p_final: f64,
}

// =============================================================================
// Helpers (math, calibration, parsing, diagnostics, and model fitting)
// =============================================================================

// -----------------------------------------------------------------------------
// 1) Low-level special functions + stable numeric primitives
// -----------------------------------------------------------------------------

#[inline]
fn neg_log10_p(p: f64) -> f64 {
    // “quality”: bigger is better
    -p.max(1e-300).log10()
}

// -----------------------------------------------------------------------------
// 2) TEV normalization helper
// -----------------------------------------------------------------------------
//
// Keep core.hyperscore unchanged.
// For any Gumbel(mu,beta).sf(hyperscore), compute TEV_norm = (hs - mu)/beta
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
    if pvals.is_empty() {
        return 1.0;
    }

    match how {
        EnsemblePCombiner::Hmp => stats::combine_hmp(pvals),
        EnsemblePCombiner::Fisher => stats::combine_fisher(pvals),
        EnsemblePCombiner::Brown => {
            if brown_params.is_none() {
                log::warn!("Brown requested but params missing; using Fisher fallback.");
            }
            stats::combine_brown(pvals, brown_params)
        }

        EnsemblePCombiner::Cauchy => combine_cauchy(pvals),
        EnsemblePCombiner::MedianBeta => combine_median_beta(pvals),
        EnsemblePCombiner::Stouffer => combine_stouffer(pvals),
        EnsemblePCombiner::SidakMinP => combine_sidak_minp(pvals),
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
// - Keep `core.hyperscore` UNCHANGED and treat it as the raw evidence score
//   produced by vanilla Sage ("hyperscore").
// - The decoy-free null is modeled as Gumbel(mu, beta) over this raw hyperscore
//   (fitted from the rank-null pool).
// - Whenever "TEV-normalized" input is needed for survival evaluation, compute:
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
// 6c) DF reset helper (critical): clear all DF + per-method outputs
// -----------------------------------------------------------------------------

#[inline(always)]
fn clear_all_df_outputs(psm: &mut DfFeature, fail_closed_rank1: bool) {
    // clear DF streams
    psm.decoy_free_p_value = None;
    psm.decoy_free_pep = None;
    psm.decoy_free_score = None;
    psm.decoy_free_q_value = None;
    psm.decoy_free_peptide_q = None;
    psm.decoy_free_protein_q = None;

    // clear per-method p streams
    psm.p_mom = None;
    psm.p_mle = None;
    psm.p_lo = None;
    psm.p_msfdr = None;
    psm.p_1smix = None;
    psm.p_2smix = None;
    psm.p_nokoi = None;

    // clear per-method pep streams
    psm.pep_mom = None;
    psm.pep_mle = None;
    psm.pep_lo = None;
    psm.pep_msfdr = None;
    psm.pep_1smix = None;
    psm.pep_2smix = None;
    psm.pep_nokoi = None;

    // clear per-method q streams
    psm.q_mom = None;
    psm.q_mle = None;
    psm.q_lo = None;
    psm.q_msfdr = None;
    psm.q_1smix = None;
    psm.q_2smix = None;
    psm.q_nokoi = None;

    // If you want rank1 to be explicitly fail-closed (rather than None),
    // populate rank1 with conservative defaults.
    if fail_closed_rank1 && psm.core.rank == 1 {
        psm.decoy_free_p_value = Some(1.0);
        psm.decoy_free_pep = Some(1.0);
        psm.decoy_free_score = Some(0.0);
        psm.decoy_free_q_value = Some(1.0);
        psm.decoy_free_peptide_q = Some(1.0);
        psm.decoy_free_protein_q = Some(1.0);
    }
}

// -----------------------------------------------------------------------------
// 6d) DF rank-order “score” key (authoritative for "sorted by score")
// -----------------------------------------------------------------------------

#[inline(always)]
fn df_rank_score(f: &DfFeature) -> f64 {
    tev(f).unwrap_or_else(|| f.decoy_free_score.unwrap_or(0.0) as f64)
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
// 9b) Model-fit diagnostics helpers (Step 4.2)
// -----------------------------------------------------------------------------

#[inline]
fn log_fit_ok<T: std::fmt::Debug>(label: &str, model: &T) {
    // Use Debug formatting so we don't depend on model field visibility.
    // If msfdr.rs Debug includes pi/null/target params, you'll see them here.
    log::info!("DF {} fit OK: {:?}", label, model);
}

#[inline]
fn log_fit_failed_closed(label: &str) {
    // "failed-closed" means the variant did not produce an engine (None)
    // and the pipeline will proceed without writing that stream.
    log::warn!(
        "DF {} fit FAILED (None) — failing closed for this variant.",
        label
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
// 10b) P-value -> PEP proxy via local-FDR (Storey pi0 + histogram density)
// -----------------------------------------------------------------------------
//
// Under the mixture model for p-values:
//   f(p) = pi0 * 1 + (1 - pi0) * g(p)
// local-fdr / PEP proxy:  lfdr(p) = P(null | p) = (pi0 * 1) / f(p) = pi0 / f(p)
//
// Estimate f(p) with a simple histogram density estimator on the method’s
// rank-1 p-values (optionally filtered to a “reference set” for pi0 only).
//
// This produces a monotone-ish, bounded PEP proxy that is NOT equal to p.
//
fn hist_density_01(p: &[f64], bins: usize) -> Vec<f64> {
    let bins = bins.max(20).min(2000);
    let mut counts = vec![0usize; bins];
    let mut n = 0usize;

    for &x in p {
        if !x.is_finite() {
            continue;
        }
        let xx = x.clamp(0.0, 1.0);
        // put p==1.0 into last bin
        let mut b = (xx * (bins as f64)) as usize;
        if b >= bins {
            b = bins - 1;
        }
        counts[b] += 1;
        n += 1;
    }

    if n == 0 {
        return vec![1.0; bins]; // fail-closed: uniform density
    }

    let bin_w = 1.0 / (bins as f64);
    counts
        .into_iter()
        .map(|c| {
            // density estimate: count / (n * bin_w)
            let d = (c as f64) / ((n as f64) * bin_w);
            if d.is_finite() && d > 0.0 {
                d
            } else {
                1.0
            }
        })
        .collect()
}

#[inline]
fn pep_from_p_lfdr(p: f64, pi0: f64, dens: &[f64]) -> f64 {
    let bins = dens.len().max(1);
    let pp = p.clamp(0.0, 1.0);
    let mut b = (pp * (bins as f64)) as usize;
    if b >= bins {
        b = bins - 1;
    }
    let fhat = dens[b].max(1e-12); // avoid div-by-zero
    (pi0 / fhat).clamp(0.0, 1.0).max(1e-300)
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

// -----------------------------------------------------------------------------
// 13)
// -----------------------------------------------------------------------------

#[inline]
fn lo_bucket_id(settings: &FdrSettings, charge: u8) -> u8 {
    match settings.lo_stratify {
        LoStratify::Charge => charge,
        LoStratify::Global => 0u8,
    }
}

// --- STAGE STRUCTS ---
#[derive(Clone, Debug)]
struct RankNullPool {
    // Null pool members (purified) are rank in [min_rank..=max_rank]
    // Keep indices so other models (e.g., Nokoi) can reuse the same null pool.
    fit_data: Vec<(u32, f64, u8, usize, String)>, // (rank, score, charge, file_id, spec_id)
    null_indices: Vec<usize>, // indices into `features`, aligned with fit_data/scores
    scores: Vec<f64>,         // hyperscore scores for global moments/mle fit (aligned)
}

impl RankNullPool {
    /// Return hyperscore values for pool members whose rank is in [min..=max].
    /// Note: ranks are the original per-PSM hit rank used to build the global pool.
    fn scores_in_window(&self, min: u32, max: u32) -> Vec<f64> {
        self.fit_data
            .iter()
            .filter_map(|(rank, score, _charge, _file_id, _spec_id)| {
                if *rank >= min && *rank <= max {
                    Some(*score)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return (rank, hyperscore, charge) tuples for pool members whose rank is in [min..=max].
    fn fit_data_in_window(&self, min: u32, max: u32) -> Vec<(u32, f64, u8, usize, String)> {
        self.fit_data
            .iter()
            .cloned()
            .filter(|(rank, _score, _charge, _file_id, _spec_id)| *rank >= min && *rank <= max)
            .collect()
    }

    /// Return indices into the original `features` slice for pool members whose rank is in [min..=max].
    /// This is aligned with `fit_data`: `null_indices[i]` corresponds to `fit_data[i]`.
    #[allow(dead_code)]
    fn null_indices_in_window(&self, min: u32, max: u32) -> Vec<usize> {
        self.fit_data
            .iter()
            .zip(self.null_indices.iter())
            .filter_map(|((rank, _score, _charge, _file_id, _spec_id), idx)| {
                if *rank >= min && *rank <= max {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Clone)]
struct Engines {
    // Fitted parameters for TEV normalization (no need to carry Gumbel objects)
    mom_mu: f64,
    mom_beta: f64,

    // MLE fit (may fall back to moments if fail-closed)
    mle_mu: f64,
    mle_beta: f64,
    mle_fit_ok: bool,

    // LO parameters (bucket-stratified TNM model)
    lo_model: Option<LowerOrderModel>,

    // Only populated when lo_score == PerSpectrum; key = (file_id, spec_id)
    lo_centers: Option<FnvHashMap<(usize, String), f64>>,

    lo_fit_ok: bool,

    // MSFDR variants (Phase 4.1)
    msfdr_seeded: Option<MsfdrSeededModel>,
    msfdr_1smix: Option<Msfdr1SmixModel>,
    msfdr_2smix: Option<Msfdr2SmixModel>,

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

    // --- SOFT PURIFIED NULL (same logic) ---

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

    let mut fit_data: Vec<(u32, f64, u8, usize, String)> = Vec::new();
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
        fit_data.push((
            r,
            s,
            psm.core.charge,
            psm.core.file_id,
            psm.core.spec_id.clone(),
        ));
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
            fit_data.push((
                r,
                s,
                psm.core.charge,
                psm.core.file_id,
                psm.core.spec_id.clone(),
            ));
            null_indices.push(idx);
        }
    }

    // final safety (keep alignment between fit_data and null_indices)
    let mut fit2: Vec<(u32, f64, u8, usize, String)> = Vec::with_capacity(fit_data.len());
    let mut idx2: Vec<usize> = Vec::with_capacity(null_indices.len());

    for (k, (r, s, z, file_id, spec_id)) in fit_data.into_iter().enumerate() {
        if s.is_finite() {
            fit2.push((r, s, z, file_id, spec_id));
            idx2.push(null_indices[k]);
        }
    }
    let fit_data = fit2;
    let null_indices = idx2;

    if fit_data.len() < min_null_size {
        return None;
    }

    let scores: Vec<f64> = fit_data.iter().map(|(_, s, _, _, _)| *s).collect();

    Some(RankNullPool {
        fit_data,
        null_indices,
        scores,
    })
}

#[derive(Clone, Copy, Debug)]
struct RunGates {
    run_msfdr_seeded: bool,
    run_msfdr_1smix: bool,
    run_msfdr_2smix: bool,
    run_nokoi: bool,
}

// --- MSFDR variant fit helpers ---

#[inline]
fn fit_msfdr_seeded(
    rank1_scores: &[f64],
    pool_scores: &[f64],
    settings: &FdrSettings,
) -> Option<MsfdrSeededModel> {
    let iters = settings.mix_em_max_iter;
    let pi_clamp = (settings.msfdr_pi_clamp_min, settings.msfdr_pi_clamp_max);
    let top_frac_init = settings.msfdr1_top_frac_init;

    // Pool-based moments seed (same math pattern as 1smix seed block)
    let xs: Vec<f64> = pool_scores
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if xs.len() < 20 {
        return None; // fail-closed
    }

    let mean = xs.iter().sum::<f64>() / (xs.len() as f64);
    let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (xs.len() as f64);

    let beta = ((6.0 * var).sqrt() / std::f64::consts::PI).max(1e-9);
    let mu = mean - 0.5772156649015329_f64 * beta;

    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    MsfdrSeededModel::fit_rank1_seeded(rank1_scores, mu, beta, iters, pi_clamp, top_frac_init)
}

#[inline]
fn fit_msfdr_1smix(
    rank1_scores: &[f64],
    pool_scores: &[f64],
    settings: &FdrSettings,
) -> Option<Msfdr1SmixModel> {
    let iters = settings.mix_em_max_iter;
    let pi_clamp = (settings.msfdr1_pi_clamp_min, settings.msfdr1_pi_clamp_max);
    let top_frac_init = settings.msfdr1_top_frac_init;

    // Capture drift settings
    let mu_drift_abs = settings.msfdr1_mu_drift_abs;
    let beta_drift_mult = settings.msfdr1_beta_drift_mult;

    // Pool-based seed: fit gumbel moments from pool window.
    let seed = {
        let xs: Vec<f64> = pool_scores
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        if xs.len() < 20 {
            None
        } else {
            let mean = xs.iter().sum::<f64>() / (xs.len() as f64);
            let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (xs.len() as f64);
            let beta = ((6.0 * var).sqrt() / std::f64::consts::PI).max(1e-6);
            let mu = mean - 0.5772156649015329_f64 * beta;

            if mu.is_finite() && beta.is_finite() && beta > 0.0 {
                Some((mu, beta))
            } else {
                None
            }
        }
    };

    if let Some((mu, beta)) = seed {
        Msfdr1SmixModel::fit_rank1_with_null_seed(
            rank1_scores,
            iters,
            pi_clamp,
            mu,
            beta,
            top_frac_init,
            mu_drift_abs,
            beta_drift_mult,
        )
    } else {
        // Fail-closed fallback: keep old behavior if pool seed is too small/noisy
        let bottom_frac_init = settings.msfdr1_bottom_frac_init;
        Msfdr1SmixModel::fit_rank1(
            rank1_scores,
            iters,
            pi_clamp,
            bottom_frac_init,
            top_frac_init,
            mu_drift_abs,
            beta_drift_mult,
        )
    }
}

#[inline]
fn fit_msfdr_2smix(
    rank1_scores: &[f64],
    pool_scores: &[f64],
    settings: &FdrSettings,
) -> Option<Msfdr2SmixModel> {
    let iters = settings.mix_em_max_iter;
    let pi_clamp = (settings.msfdr2_pi_clamp_min, settings.msfdr2_pi_clamp_max);

    let top_frac_init = settings.msfdr1_top_frac_init;

    let mix_anchor_incorrect = settings.mix_anchor_incorrect;
    let beta_drift_mult = settings.msfdr2_beta_drift_mult;
    let mu_drift_abs = settings.msfdr2_mu_drift_abs;

    Msfdr2SmixModel::fit_rank1_with_pool(
        rank1_scores,
        pool_scores,
        iters,
        pi_clamp,
        top_frac_init,
        mix_anchor_incorrect,
        beta_drift_mult,
        mu_drift_abs,
    )
}

// --- FIT/PREPARE ENGINES ---
fn fit_engines(
    features: &[DfFeature],
    work: &WorkSet,
    pool: &RankNullPool,
    settings: &FdrSettings,
    gates: RunGates,
) -> Option<Engines> {
    // ---------------------------------------------------------------------
    // Fail-closed gating for per-method null windows (global threshold).
    // ---------------------------------------------------------------------
    let min_null_size = settings.min_null_size;

    let window_ok = |method: &str, window_min: u32, window_max: u32, count: usize| -> bool {
        if count < min_null_size {
            log::warn!(
                "DF fail-closed: {method}: null window [{window_min}..={window_max}] too small (n={count} < min_null_size={min_null_size}). Skipping.",
            );
            false
        } else {
            true
        }
    };

    // 1) Moments
    let mom_scores = pool.scores_in_window(
        settings.moments_min_null_rank,
        settings.moments_max_null_rank,
    );
    if !window_ok(
        "Moments",
        settings.moments_min_null_rank,
        settings.moments_max_null_rank,
        mom_scores.len(),
    ) {
        // Moments are required for TEV normalization + Nokoi provisional p-gate.
        return None;
    }

    let (mu_mom, beta_mom) = fit_gumbel_moments(&mom_scores);
    let moments_ok = mu_mom.is_finite() && beta_mom.is_finite() && beta_mom > 0.0;
    if !moments_ok {
        log::warn!(
            "DF fail-closed: Moments produced invalid params (mu={mu_mom}, beta={beta_mom}) using window [{}..={}].",
            settings.moments_min_null_rank,
            settings.moments_max_null_rank
        );
        return None;
    }

    // 2) LO: call the Lower Order fitter

    fn build_lo_per_spectrum_center(
        features: &[DfFeature],
        settings: &FdrSettings,
    ) -> FnvHashMap<(usize, String), f64> {
        // Key: (file_id, spec_id) -> median score in [min_rank..=max_rank] for that spectrum.
        let min_k = settings.lower_order_min_null_rank;
        let max_k = settings.lower_order_max_null_rank;

        let mut buckets: FnvHashMap<(usize, String), Vec<f64>> = FnvHashMap::default();
        for f in features {
            let k = f.core.rank;
            if k < min_k || k > max_k {
                continue;
            }
            if let Some(x) = tev(f) {
                buckets
                    .entry((f.core.file_id, f.core.spec_id.clone()))
                    .or_default()
                    .push(x);
            }
        }

        let mut centers: FnvHashMap<(usize, String), f64> = FnvHashMap::default();
        for (key, mut xs) in buckets {
            if xs.is_empty() {
                continue;
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            let med = xs[xs.len() / 2];
            centers.insert(key, med);
        }
        centers
    }

    let lo_centers = if settings.lo_score == LoScore::PerSpectrum {
        Some(build_lo_per_spectrum_center(features, settings))
    } else {
        None
    };

    //
    // Required by the LO fitter:
    // - pool.fit_data:           full rank-null pool stream (rank, hyperscore, charge)
    // - rank1_scores_by_charge:  rank-1 stream (hyperscore, charge)
    let mut rank1_scores_by_charge: Vec<(f64, u8)> = Vec::with_capacity(work.rank1_indices.len());
    for &i in &work.rank1_indices {
        let f = &features[i];
        let x = match tev(f) {
            Some(v) => v,
            None => continue,
        };
        let bid = lo_bucket_id(settings, f.core.charge);
        let x_lo = if let Some(ref centers) = lo_centers {
            let key = (f.core.file_id, f.core.spec_id.clone());
            if let Some(c) = centers.get(&key) {
                x - *c
            } else {
                x
            }
        } else {
            x
        };
        rank1_scores_by_charge.push((x_lo, bid));
    }

    let lo_raw_fit_data = pool.fit_data_in_window(
        settings.lower_order_min_null_rank,
        settings.lower_order_max_null_rank,
    );

    // Window-size gate (data availability only)
    let lo_window_ok = window_ok(
        "LowerOrder",
        settings.lower_order_min_null_rank,
        settings.lower_order_max_null_rank,
        lo_raw_fit_data.len(),
    );

    let lo_fit_data_for_fit: Vec<(u32, f64, u8)> = if lo_window_ok {
        lo_raw_fit_data
            .into_iter()
            .map(|(k, x, charge, file_id, spec_id)| {
                let x2 = if let Some(ref centers) = lo_centers {
                    if let Some(c) = centers.get(&(file_id, spec_id)) {
                        x - *c
                    } else {
                        x
                    }
                } else {
                    x
                };
                (k, x2, lo_bucket_id(settings, charge))
            })
            .collect()
    } else {
        // Fail-closed: do not fit LO on an undersized window.
        Vec::new()
    };

    // Fit LO
    let lo_model: Option<LowerOrderModel> = fit_decoy_free_model(
        &lo_fit_data_for_fit,
        &rank1_scores_by_charge,
        settings.lower_order_min_null_rank,
        settings.lower_order_max_null_rank,
        settings.min_null_size,
        settings.min_rank_count,
        settings.lo_mode.clone(),
        settings.lo_lom_estimator.clone(),
        settings.lo_mean_beta_mode.clone(),
        settings.lo_mean_beta_min_rank,
        settings.lo_mean_beta_count,
        settings.lo_lr_window_size,
    );

    // Real “LO is usable” gate: must have enough data AND the model must exist
    let lo_fit_ok = lo_window_ok && lo_model.is_some();

    // 3) MLE
    let mle_scores = pool.scores_in_window(settings.mle_min_null_rank, settings.mle_max_null_rank);

    let mut mle_fit_ok = false;
    let (mu_mle, beta_mle) = if window_ok(
        "MLE",
        settings.mle_min_null_rank,
        settings.mle_max_null_rank,
        mle_scores.len(),
    ) {
        match fit_gumbel_mle(&mle_scores) {
            Some((mu, beta)) if mu.is_finite() && beta.is_finite() && beta > 0.0 => {
                mle_fit_ok = true;
                (mu, beta)
            }
            _ => {
                log::warn!(
                    "DF fail-closed: MLE fit invalid using window [{}..={}]; falling back to Moments (and excluding MLE from Ensemble).",
                    settings.mle_min_null_rank,
                    settings.mle_max_null_rank
                );
                (mu_mom, beta_mom)
            }
        }
    } else {
        // Fail-closed: do not fit MLE; keep pipeline runnable via Moments fallback.
        (mu_mom, beta_mom)
    };

    // 4) MSFDR: compute rank1_scores + pool_scores ONCE, then fit gated
    //
    // rank1_scores = rank1 hyperscores (finite)
    // pool_scores  = pool.scores (already)
    let rank1_scores: Vec<f64> = work
        .rank1_indices
        .iter()
        .filter_map(|&i| tev(&features[i]))
        .collect();

    let msfdr_seed_pool =
        pool.scores_in_window(settings.msfdr_min_null_rank, settings.msfdr_max_null_rank);

    let msfdr2_pool = pool.scores_in_window(
        settings.msfdr2_smix_min_null_rank,
        settings.msfdr2_smix_max_null_rank,
    );

    // Fit variants gated (independent slots) + diagnostics (Step 4.2)
    let msfdr_seeded = if gates.run_msfdr_seeded {
        let m = if window_ok(
            "MSFDR seeded",
            settings.msfdr_min_null_rank,
            settings.msfdr_max_null_rank,
            msfdr_seed_pool.len(),
        ) {
            fit_msfdr_seeded(&rank1_scores, &msfdr_seed_pool, settings)
        } else {
            None
        };

        match &m {
            Some(model) => {
                log_fit_ok("MSFDR seeded", model);
                log::info!(
                    "MSFDR(seed) pool window [{}..={}] n={}",
                    settings.msfdr_min_null_rank,
                    settings.msfdr_max_null_rank,
                    msfdr_seed_pool.len()
                );
                log::info!("MSFDR(seed) params: {}", model.param_tuple());
            }
            None => {
                log_fit_failed_closed("MSFDR seeded");
            }
        }
        m
    } else {
        None
    };

    let msfdr_1smix = if gates.run_msfdr_1smix {
        let pool_scores_1smix = pool.scores_in_window(
            settings.msfdr1_smix_min_null_rank,
            settings.msfdr1_smix_max_null_rank,
        );

        let m = fit_msfdr_1smix(&rank1_scores, &pool_scores_1smix, settings);
        match &m {
            Some(model) => {
                log_fit_ok("MSFDR 1smix", model);
                log::info!("MSFDR(1smix) params: {}", model.param_tuple());
            }
            None => {
                log_fit_failed_closed("MSFDR 1smix");
            }
        }
        m
    } else {
        None
    };

    let msfdr_2smix = if gates.run_msfdr_2smix {
        let m = if window_ok(
            "MSFDR 2smix",
            settings.msfdr2_smix_min_null_rank,
            settings.msfdr2_smix_max_null_rank,
            msfdr2_pool.len(),
        ) {
            fit_msfdr_2smix(&rank1_scores, &msfdr2_pool, settings)
        } else {
            None
        };

        match &m {
            Some(model) => {
                log_fit_ok("MSFDR 2smix", model);
                log::info!("MSFDR(2smix) params: {}", model.param_tuple());
            }
            None => {
                log_fit_failed_closed("MSFDR 2smix");
            }
        }
        m
    } else {
        None
    };

    // 5) Nokoi: pep from probability, p from rank-null pool empirical survival
    let run_nokoi = gates.run_nokoi;

    let mut nokoi_prob_target: Option<Arc<Vec<f64>>> = None;
    let mut nokoi_p_values: Option<Arc<Vec<f64>>> = None;

    if run_nokoi {
        log::info!("Running Nokoi Rescoring ...");

        let nokoi_window_fit_data =
            pool.fit_data_in_window(settings.nokoi_min_null_rank, settings.nokoi_max_null_rank);

        let nokoi_window_ok = window_ok(
            "Nokoi",
            settings.nokoi_min_null_rank,
            settings.nokoi_max_null_rank,
            nokoi_window_fit_data.len(),
        );

        if !nokoi_window_ok {
            log::warn!(
                "Nokoi disabled: invalid/too-small null window [{}..={}], count={}",
                settings.nokoi_min_null_rank,
                settings.nokoi_max_null_rank,
                nokoi_window_fit_data.len()
            );
        } else {
            // Independent Nokoi positives:
            // rank==1 AND hyperscore in purified top slice
            let mut rank1_hs: Vec<f64> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| tev(&features[i]))
                .collect();

            let pos_hyperscore_threshold: f64 = if rank1_hs.len() >= 10 {
                rank1_hs.sort_by(|a, b| b.total_cmp(a));
                let top_k =
                    ((rank1_hs.len() as f64) * settings.purification_factor).round() as usize;
                let top_k = top_k.max(5).min(rank1_hs.len());
                rank1_hs[top_k - 1]
            } else {
                f64::INFINITY
            };

            let is_positive = |f: &DfFeature| -> bool {
                if f.core.rank != 1 {
                    return false;
                }

                let x = match tev(f) {
                    Some(v) => v,
                    None => return false,
                };

                x >= pos_hyperscore_threshold
            };

            if log::log_enabled!(log::Level::Debug) {
                let n_pos = features.iter().filter(|f| is_positive(f)).count();
                let n_neg = features
                    .iter()
                    .filter(|f| {
                        if is_positive(f) {
                            return false;
                        }
                        let r = f.core.rank as u32;
                        r >= settings.nokoi_min_null_rank && r <= settings.nokoi_max_null_rank
                    })
                    .count();

                log::debug!(
                    "DF DEBUG Nokoi: pos_hyperscore_threshold={:.6} nokoi_min_null_rank={} nokoi_max_null_rank={} n_pos={} n_neg={} l1_lambda_min={:.6e} l1_lambda_max={:.6e} l1_lambda_steps={}",
                    pos_hyperscore_threshold,
                    settings.nokoi_min_null_rank,
                    settings.nokoi_max_null_rank,
                    n_pos,
                    n_neg,
                    settings.nokoi_l1_lambda_min,
                    settings.nokoi_l1_lambda_max,
                    settings.nokoi_l1_lambda_steps
                );
            }

            let nokoi_config = nokoi::NokoiConfig {
                enabled: true,
                train_fdr: 0.01,
                learning_rate: 0.1,
                epochs: 500,
                patience: 15,
                l1_lambda: settings.nokoi_l1_lambda_min,
                l1_lambda_min: settings.nokoi_l1_lambda_min,
                l1_lambda_max: settings.nokoi_l1_lambda_max,
                l1_lambda_steps: settings.nokoi_l1_lambda_steps,
            };

            if let Some((probs, null_scores_oof)) = nokoi::rescore_df_crossfit(
                features,
                &nokoi_config,
                settings.nokoi_min_null_rank,
                settings.nokoi_max_null_rank,
                settings.nokoi_k_folds,
                is_positive,
                &pool.null_indices,
            ) {
                if probs.len() != features.len() {
                    log::error!(
                        "Nokoi probabilities not aligned: probs.len()={} features.len()={}. Disabling Nokoi.",
                        probs.len(),
                        features.len()
                    );
                } else {
                    let mut prob = probs;
                    for v in &mut prob {
                        let vv = if v.is_finite() { *v } else { 0.0 };
                        *v = vv.clamp(0.0, 1.0);
                    }
                    let prob_arc = Arc::new(prob);

                    let mut null_scores: Vec<f64> = null_scores_oof;
                    null_scores.retain(|x| x.is_finite());
                    null_scores.sort_by(|a, b| a.total_cmp(b));

                    if log::log_enabled!(log::Level::Debug) && !null_scores.is_empty() {
                        let n = null_scores.len();
                        let idx01 = ((n as f64 - 1.0) * 0.01).round() as usize;
                        let idx50 = ((n as f64 - 1.0) * 0.50).round() as usize;
                        let idx99 = ((n as f64 - 1.0) * 0.99).round() as usize;

                        let p01 = null_scores[idx01.min(n - 1)];
                        let p50 = null_scores[idx50.min(n - 1)];
                        let p99 = null_scores[idx99.min(n - 1)];

                        log::debug!(
                            "DF DEBUG Nokoi null_scores quantiles: n_null={} p01={:.6} p50={:.6} p99={:.6}",
                            n,
                            p01,
                            p50,
                            p99
                        );
                    }

                    if null_scores.len() < 10 {
                        log::warn!(
                            "Nokoi: rank-null pool too small for null calibration (n_null={}); disabling Nokoi p-values.",
                            null_scores.len()
                        );
                        nokoi_prob_target = Some(prob_arc);
                    } else {
                        let mut p_all = vec![1.0f64; features.len()];
                        for (i, &pt) in prob_arc.iter().enumerate() {
                            p_all[i] = empirical_p_from_null_ge(&null_scores, pt);
                        }

                        for v in &mut p_all {
                            let vv = if v.is_finite() { *v } else { 1.0 };
                            *v = vv.clamp(0.0, 1.0).max(1e-300);
                        }

                        nokoi_prob_target = Some(prob_arc);
                        nokoi_p_values = Some(Arc::new(p_all));
                    }
                }
            } else {
                log::warn!("Nokoi disabled: crossfit training/calibration failed closed.");
            }
        }
    }

    Some(Engines {
        mom_mu: mu_mom,
        mom_beta: beta_mom,

        mle_mu: mu_mle,
        mle_beta: beta_mle,

        mle_fit_ok,
        lo_model,
        lo_centers,
        lo_fit_ok,

        msfdr_seeded,
        msfdr_1smix,
        msfdr_2smix,

        nokoi_prob_target,
        nokoi_p_values,
    })
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
    // (Still compute intermediate values for internal fallbacks, but do NOT populate
    // per-method columns unless the method is enabled here.)
    let run_mom = use_ensemble || matches!(settings.model_fit, ModelFit::Moments);
    let run_mle = use_ensemble || matches!(settings.model_fit, ModelFit::Mle);
    let run_lo = use_ensemble || matches!(settings.model_fit, ModelFit::LowerOrder);

    // MSFDR variants (independent gating; replaces any single run_msfdr flag)
    let run_msfdr_seeded = matches!(settings.model_fit, ModelFit::Msfdr)
        || (use_ensemble && settings.enable_msfdr_seeded);

    let run_msfdr_1smix = matches!(settings.model_fit, ModelFit::Msfdr1Smix)
        || (use_ensemble && settings.enable_msfdr_1smix);

    let run_msfdr_2smix = matches!(settings.model_fit, ModelFit::Msfdr2Smix)
        || (use_ensemble && settings.enable_msfdr_2smix);

    let run_nokoi = use_ensemble || matches!(settings.model_fit, ModelFit::Nokoi);

    let gates = RunGates {
        run_msfdr_seeded,
        run_msfdr_1smix,
        run_msfdr_2smix,
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
                psm.p_1smix = None;
                psm.p_2smix = None;
                psm.p_nokoi = None;

                psm.pep_mom = None;
                psm.pep_mle = None;
                psm.pep_lo = None;
                psm.pep_msfdr = None;
                psm.pep_1smix = None;
                psm.pep_2smix = None;
                psm.pep_nokoi = None;

                psm.q_mom = None;
                psm.q_mle = None;
                psm.q_lo = None;
                psm.q_msfdr = None;
                psm.q_1smix = None;
                psm.q_2smix = None;
                psm.q_nokoi = None;

                if psm.core.rank == 1 {
                    psm.decoy_free_p_value = Some(1.0);
                    psm.decoy_free_pep = Some(1.0);
                    psm.decoy_free_score = Some(0.0);
                    psm.decoy_free_q_value = Some(1.0);
                    psm.decoy_free_peptide_q = Some(1.0);
                    psm.decoy_free_protein_q = Some(1.0);
                }
            });
            return new_features;
        }
    };

    log::info!("DF: pool_size={}", pool.fit_data.len());

    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "DF DEBUG settings: min_null_rank={} max_null_rank={} min_null_size={} purification_factor={}",
            settings.min_null_rank,
            settings.max_null_rank,
            settings.min_null_size,
            settings.purification_factor
        );

        log::debug!(
            "DF DEBUG pool sizes: fit_data.len()={} null_indices.len()={} scores.len()={}",
            pool.fit_data.len(),
            pool.null_indices.len(),
            pool.scores.len()
        );

        // rank histogram over pool.fit_data
        let mut rank_hist: BTreeMap<u32, usize> = BTreeMap::new();
        for (r, _, _, _, _) in &pool.fit_data {
            *rank_hist.entry(*r).or_insert(0) += 1;
        }

        // compact string: "4:123 5:456 ..."
        let mut hist_s = String::new();
        for (r, c) in rank_hist {
            if !hist_s.is_empty() {
                hist_s.push(' ');
            }
            hist_s.push_str(&format!("{}:{}", r, c));
        }
        log::debug!("DF DEBUG pool.fit_data rank histogram: {}", hist_s);
    }

    // ==============================
    // Stage 2 — fit engines
    // ==============================
    let engines = match fit_engines(&new_features, &work, &pool, settings, gates) {
        Some(e) => e,
        None => {
            log::error!("Invalid null fit. FDR will fail closed.");
            new_features.par_iter_mut().for_each(|psm| {
                // CRITICAL: clear DF + per-method outputs to prevent stale leakage.
                // Rank1 gets explicit fail-closed defaults.
                clear_all_df_outputs(psm, true);
            });
            return new_features;
        }
    };

    // ==============================
    // Diagnostics: log model-fit success + key parameters
    // ==============================
    log::info!(
        "DF fit summary: moments_null=(mu={:.6}, beta={:.6}) mle_null=(mu={:.6}, beta={:.6})",
        engines.mom_mu,
        engines.mom_beta,
        engines.mle_mu,
        engines.mle_beta
    );

    // LO: you at least have fallback params available; log them.
    match &engines.lo_model {
        Some(m) => log::info!(
            "DF fit summary: LO fallback_params=(mu={:.6}, beta={:.6})",
            m.fallback_params.0,
            m.fallback_params.1
        ),
        None => log::warn!("DF fail-closed: LO failed to fit (no fitted charges)."),
    }

    // MSFDR variants: summarize presence + debug dump if present
    match &engines.msfdr_seeded {
        Some(m) => log_fit_ok("MSFDR seeded (engine present)", m),
        None => log::info!("DF MSFDR seeded: engine absent (None)"),
    }
    match &engines.msfdr_1smix {
        Some(m) => log_fit_ok("MSFDR 1smix (engine present)", m),
        None => log::info!("DF MSFDR 1smix: engine absent (None)"),
    }
    match &engines.msfdr_2smix {
        Some(m) => log_fit_ok("MSFDR 2smix (engine present)", m),
        None => log::info!("DF MSFDR 2smix: engine absent (None)"),
    }

    // --- CALCULATION LOOP ---
    // --- capture small config once for rayon closure ---
    let ensemble_p_combiner = settings.ensemble_p_combiner.clone();
    let ensemble_pep_combiner = settings.ensemble_pep_combiner.clone();

    let mom_mu = engines.mom_mu;
    let mom_beta = engines.mom_beta;

    let mle_mu = engines.mle_mu;
    let mle_beta = engines.mle_beta;

    let lo_model = engines.lo_model.clone(); // Option<LowerOrderModel>
    let lo_centers = engines.lo_centers.clone(); // Option<FnvHashMap<(usize, String), f64>>

    // Standard Gumbel for TEV-normalized sf inputs
    let std_gumbel = Gumbel::new(0.0, 1.0).expect("standard gumbel");

    let msfdr_seeded = engines.msfdr_seeded.clone();
    let msfdr_1smix = engines.msfdr_1smix.clone();
    let msfdr_2smix = engines.msfdr_2smix.clone();

    let nokoi_prob_target = engines.nokoi_prob_target.clone();
    let nokoi_p_values = engines.nokoi_p_values.clone();

    // Expert inclusion gates for Ensemble:
    // - Moments is required; if it fails, fit_engines() returns None earlier.
    // - MLE/LO are included ONLY if they actually fit (not fallback-only).
    let use_mle_expert = run_mle && engines.mle_fit_ok;
    let use_lo_expert = run_lo && engines.lo_fit_ok;

    // “expert present” gates (must match Brown fitting + Part A expert lists)
    let use_seeded_expert = run_msfdr_seeded && msfdr_seeded.is_some();
    let use_1smix_expert = run_msfdr_1smix && msfdr_1smix.is_some();
    let use_2smix_expert = run_msfdr_2smix && msfdr_2smix.is_some();

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
                    if use_mle_expert {
                        let tev = tev_norm_from_hyperscore(x, mle_mu, mle_beta);
                        experts.push(std_gumbel.sf(tev).clamp(0.0, 1.0).max(1e-300));
                    }

                    // Lower-Order (LO) — global TNM only (no per-PSM ln_ratio shift)
                    if use_lo_expert {
                        let bid = lo_bucket_id(settings, psm.core.charge);

                        let x_eval = if settings.lo_score == LoScore::PerSpectrum {
                            if let Some(ref centers) = lo_centers {
                                let key = (psm.core.file_id, psm.core.spec_id.clone());
                                if let Some(c) = centers.get(&key) {
                                    x - *c
                                } else {
                                    x
                                }
                            } else {
                                x
                            }
                        } else {
                            x
                        };

                        if let Some(ref m) = lo_model {
                            experts.push(m.p_value(x_eval, bid).max(1e-300));
                        } else {
                            // LO requested but failed to fit => fail-closed
                            experts.push(1.0);
                        }
                    }

                    // MSFDR seeded / 1smix / 2smix (only if present for this run)
                    if use_seeded_expert {
                        let m = msfdr_seeded
                            .as_ref()
                            .expect("use_seeded_expert implies msfdr_seeded.is_some()");
                        experts.push(m.p_value(x).clamp(0.0, 1.0).max(1e-300));
                    }
                    if use_1smix_expert {
                        let m = msfdr_1smix
                            .as_ref()
                            .expect("use_1smix_expert implies msfdr_1smix.is_some()");
                        experts.push(m.p_value(x).clamp(0.0, 1.0).max(1e-300));
                    }
                    if use_2smix_expert {
                        let m = msfdr_2smix
                            .as_ref()
                            .expect("use_2smix_expert implies msfdr_2smix.is_some()");
                        experts.push(m.p_value(x).clamp(0.0, 1.0).max(1e-300));
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
    // Compute everything first and store it in a temporary list (rank1_out)
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
            let p_lo = if let Some(ref m) = lo_model {
				let bid = lo_bucket_id(settings, psm.core.charge);

				let x_eval = if settings.lo_score == LoScore::PerSpectrum {
					if let Some(ref centers) = lo_centers {
						let key = (psm.core.file_id, psm.core.spec_id.clone());
						if let Some(c) = centers.get(&key) { x - *c } else { x }
					} else {
						x
					}
				} else {
					x
				};

				m.p_value(x_eval, bid).max(1e-300)
			} else {
				// LO requested but failed -> fail-closed
				1.0
			};

            // 3. Optional Models (MSFDR and Nokoi)
            // MSFDR (seeded / 1smix / 2smix): compute per-variant p/pep only if enabled+present.
            let (p_msfdr, pep_msfdr) = if use_seeded_expert {
                let m = msfdr_seeded
                    .as_ref()
                    .expect("use_seeded_expert implies msfdr_seeded.is_some()");
                let pep = m.pep(x).clamp(0.0, 1.0).max(1e-300);
                // Use actual null p-value for the p-stream, not PEP
                let p = m.p_value(x).clamp(0.0, 1.0).max(1e-300);
                (Some(p), Some(pep))
            } else {
                (None, None)
            };

            let (p_1smix, pep_1smix) = if use_1smix_expert {
                let m = msfdr_1smix
                    .as_ref()
                    .expect("use_1smix_expert implies msfdr_1smix.is_some()");
                let pep = m.pep(x).clamp(0.0, 1.0).max(1e-300);
                // Use actual null p-value for the p-stream, not PEP
                let p = m.p_value(x).clamp(0.0, 1.0).max(1e-300);
                (Some(p), Some(pep))
            } else {
                (None, None)
            };

            let (p_2smix, pep_2smix) = if use_2smix_expert {
                let m = msfdr_2smix
                    .as_ref()
                    .expect("use_2smix_expert implies msfdr_2smix.is_some()");
                let pep = m.pep(x).clamp(0.0, 1.0).max(1e-300);
                // Use actual null p-value for the p-stream, not PEP
                let p = m.p_value(x).clamp(0.0, 1.0).max(1e-300);
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
            if use_mle_expert {
				experts.push(p_mle);
			}
            if use_lo_expert {
				experts.push(p_lo);
			}
            // MSFDR experts:
            // - Single-method mode selects one variant (fail-closed if missing)
            // - Ensemble can include any/all variants as independent experts
            match settings.model_fit {
                ModelFit::Msfdr => {
                    if let Some(p) = p_msfdr {
                        experts.push(p);
                    }
                }
                ModelFit::Msfdr1Smix => {
                    if let Some(p) = p_1smix {
                        experts.push(p);
                    }
                }
                ModelFit::Msfdr2Smix => {
                    if let Some(p) = p_2smix {
                        experts.push(p);
                    }
                }
                ModelFit::Ensemble => {
                    if use_seeded_expert {
                        if let Some(p) = p_msfdr {
                            experts.push(p);
                        }
                    }
                    if use_1smix_expert {
                        if let Some(p) = p_1smix {
                            experts.push(p);
                        }
                    }
                    if use_2smix_expert {
                        if let Some(p) = p_2smix {
                            experts.push(p);
                        }
                    }
                }
                _ => {}
            }
            if use_nokoi_expert {
                if let Some(p) = p_nokoi {
                    experts.push(p);
                }
            }

            let p_final = if use_ensemble {
                // If an expert didn't run / didn't produce p, it is simply absent.
                // If somehow empty (shouldn't happen), fail-closed.
                if experts.is_empty() {
                    log::error!(
                        "DF Ensemble fail-closed: no experts available after fit gating; returning p=1.0 for this PSM."
                    );
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
                    ModelFit::Msfdr1Smix => p_1smix.unwrap_or(1.0), // FAIL-CLOSED
                    ModelFit::Msfdr2Smix => p_2smix.unwrap_or(1.0), // FAIL-CLOSED
                    ModelFit::Nokoi => p_nokoi.unwrap_or(1.0), // FAIL-CLOSED
                    _ => 1.0,                                  // FAIL-CLOSED
                }
            }
            .clamp(0.0, 1.0)
            .max(1e-300);

            // 5) Final DF p (already computed above). PEP/score are computed later after lfdr mapping.
            Some(Rank1Computed {
                idx,
                p_mom,
                p_mle,
                p_lo,

                // MSFDR family
                p_msfdr,
                p_1smix,
                p_2smix,

                p_nokoi,

                // MSFDR family peps (model-derived)
                pep_msfdr,
                pep_1smix,
                pep_2smix,

                // Nokoi pep
                pep_nokoi,

                p_final,
            })
        })
        .collect();

    // ---------------------------------------------------------
    // PART A.5: Build PEP proxies for Moments/MLE/LO via local-FDR
    // ---------------------------------------------------------
    // Compute:
    //  - pi0 from a clean reference subset (label==1, non-ENT, non-CONT)
    //  - density f(p) from the method’s rank1 p-values (all rank1 for that method)
    //  - pep(p) = pi0 / f(p)
    //
    // NOTE: This is intentionally *not* tied to PAVA calibration; PEP is a posterior proxy.
    let pep_bins: usize = 200; // fixed, simple, stable

    // Collect method p-vectors from rank1_out (aligned 1:1 with rank1_out)
    let mut p_mom_all: Vec<f64> = Vec::new();
    let mut p_mle_all: Vec<f64> = Vec::new();
    let mut p_lo_all: Vec<f64> = Vec::new();

    p_mom_all.reserve(rank1_out.len());
    p_mle_all.reserve(rank1_out.len());
    p_lo_all.reserve(rank1_out.len());

    for r in &rank1_out {
        p_mom_all.push(r.p_mom.clamp(0.0, 1.0).max(1e-300));
        p_mle_all.push(r.p_mle.clamp(0.0, 1.0).max(1e-300));
        p_lo_all.push(r.p_lo.clamp(0.0, 1.0).max(1e-300));
    }

    // Build reference membership per rank1_out entry (for pi0 only)
    let mut is_ref: Vec<bool> = Vec::with_capacity(rank1_out.len());
    for r in &rank1_out {
        let f = &new_features[r.idx];
        if f.core.label != 1 {
            is_ref.push(false);
            continue;
        }
        let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        is_ref.push(!is_contam_str(&prot) && !is_entrapment_str(&prot));
    }

    // Build reference p vectors for pi0 (method-specific)
    let mut p_mom_ref: Vec<f64> = Vec::new();
    let mut p_mle_ref: Vec<f64> = Vec::new();
    let mut p_lo_ref: Vec<f64> = Vec::new();

    for (k, &ref_ok) in is_ref.iter().enumerate() {
        if !ref_ok {
            continue;
        }
        p_mom_ref.push(p_mom_all[k]);
        p_mle_ref.push(p_mle_all[k]);
        p_lo_ref.push(p_lo_all[k]);
    }

    // Estimate pi0 per method (fallback to 1.0 if degenerate)
    let pi0_mom = estimate_pi0_from_reference_grid(&p_mom_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let pi0_mle = estimate_pi0_from_reference_grid(&p_mle_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let pi0_lo = estimate_pi0_from_reference_grid(&p_lo_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    // Density estimates from all rank1 p-values (method-specific)
    let dens_mom = hist_density_01(&p_mom_all, pep_bins);
    let dens_mle = hist_density_01(&p_mle_all, pep_bins);
    let dens_lo = hist_density_01(&p_lo_all, pep_bins);

    // PEP proxies aligned to rank1_out order
    let pep_mom_vec: Vec<f64> = p_mom_all
        .iter()
        .map(|&p| pep_from_p_lfdr(p, pi0_mom, &dens_mom))
        .collect();
    let pep_mle_vec: Vec<f64> = p_mle_all
        .iter()
        .map(|&p| pep_from_p_lfdr(p, pi0_mle, &dens_mle))
        .collect();
    let pep_lo_vec: Vec<f64> = p_lo_all
        .iter()
        .map(|&p| pep_from_p_lfdr(p, pi0_lo, &dens_lo))
        .collect();

    // --- PART B: Write back to new_features ---
    for (j, r) in rank1_out.into_iter().enumerate() {
        let psm = &mut new_features[r.idx];

        // Write per-variant MSFDR fields ONLY if that variant is enabled+present this run.
        // Reuse the same run-level gates as Part A (expert present gates).
        let write_seeded = use_seeded_expert;
        let write_1smix = use_1smix_expert;
        let write_2smix = use_2smix_expert;

        // --- per-method p outputs (unchanged) ---
        psm.p_mom = if run_mom { Some(r.p_mom as f32) } else { None };
        psm.p_mle = if run_mle { Some(r.p_mle as f32) } else { None };
        psm.p_lo = if run_lo { Some(r.p_lo as f32) } else { None };

        psm.p_msfdr = if write_seeded {
            r.p_msfdr.map(|v| v as f32)
        } else {
            None
        };
        psm.p_1smix = if write_1smix {
            r.p_1smix.map(|v| v as f32)
        } else {
            None
        };
        psm.p_2smix = if write_2smix {
            r.p_2smix.map(|v| v as f32)
        } else {
            None
        };

        psm.p_nokoi = if run_nokoi {
            r.p_nokoi.map(|v| v as f32)
        } else {
            None
        };

        // --- per-method PEP outputs ---
        // Moments/MLE/LO now get a local-FDR PEP proxy (NOT p-value reuse).
        psm.pep_mom = if run_mom {
            Some(pep_mom_vec[j] as f32)
        } else {
            None
        };
        psm.pep_mle = if run_mle {
            Some(pep_mle_vec[j] as f32)
        } else {
            None
        };
        psm.pep_lo = if run_lo {
            Some(pep_lo_vec[j] as f32)
        } else {
            None
        };

        // MSFDR family peps remain model-derived
        psm.pep_msfdr = if write_seeded {
            r.pep_msfdr.map(|v| v as f32)
        } else {
            None
        };
        psm.pep_1smix = if write_1smix {
            r.pep_1smix.map(|v| v as f32)
        } else {
            None
        };
        psm.pep_2smix = if write_2smix {
            r.pep_2smix.map(|v| v as f32)
        } else {
            None
        };

        // Nokoi pep remains 1 - P(target)
        psm.pep_nokoi = if run_nokoi {
            r.pep_nokoi.map(|v| v as f32)
        } else {
            None
        };

        // --- final DF outputs ---
        // p_final computed in Part A, pep_final computed here using correct pep streams.
        set_df_p_value(psm, r.p_final as f32);

        let pep_final: f64 = if use_ensemble {
            let mut pep_experts: Vec<f64> = Vec::new();

            if run_mom {
                pep_experts.push(pep_mom_vec[j]);
            }
            if run_mle {
                pep_experts.push(pep_mle_vec[j]);
            }
            if run_lo {
                pep_experts.push(pep_lo_vec[j]);
            }

            if use_seeded_expert {
                if let Some(v) = r.pep_msfdr {
                    pep_experts.push(v);
                }
            }
            if use_1smix_expert {
                if let Some(v) = r.pep_1smix {
                    pep_experts.push(v);
                }
            }
            if use_2smix_expert {
                if let Some(v) = r.pep_2smix {
                    pep_experts.push(v);
                }
            }

            if run_nokoi {
                if let Some(v) = r.pep_nokoi {
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
                ModelFit::Moments => pep_mom_vec[j],
                ModelFit::Mle => pep_mle_vec[j],
                ModelFit::LowerOrder => pep_lo_vec[j],

                ModelFit::Msfdr => r.pep_msfdr.unwrap_or(1.0),
                ModelFit::Msfdr1Smix => r.pep_1smix.unwrap_or(1.0),
                ModelFit::Msfdr2Smix => r.pep_2smix.unwrap_or(1.0),

                ModelFit::Nokoi => r.pep_nokoi.unwrap_or(1.0),

                // Ensemble handled above; everything else fail-closed
                _ => 1.0,
            }
        }
        .clamp(0.0, 1.0)
        .max(1e-300);

        let df_score = (-10.0 * pep_final.max(1e-15).log10()) as f32;

        psm.decoy_free_pep = Some(pep_final as f32);
        psm.decoy_free_score = Some(df_score);

        // Debug-build assertions for PEP streams (present-only)
        #[cfg(debug_assertions)]
        {
            // Helper: assert pep is finite and in (0, 1]
            let assert_pep = |name: &str, v: Option<f32>, idx: usize| {
                if let Some(x) = v {
                    debug_assert!(
                        x.is_finite() && x > 0.0 && x <= 1.0,
                        "DF ASSERT {} invalid at feature_idx={}: {}",
                        name,
                        idx,
                        x
                    );
                }
            };

            let idx = r.idx;
            assert_pep("decoy_free_pep", psm.decoy_free_pep, idx);
            assert_pep("pep_mom", psm.pep_mom, idx);
            assert_pep("pep_mle", psm.pep_mle, idx);
            assert_pep("pep_lo", psm.pep_lo, idx);
            assert_pep("pep_msfdr", psm.pep_msfdr, idx);
            assert_pep("pep_1smix", psm.pep_1smix, idx);
            assert_pep("pep_2smix", psm.pep_2smix, idx);
            assert_pep("pep_nokoi", psm.pep_nokoi, idx);
        }
    }

    // --- PART C: Clear Non-Rank1 (Cheap safety pass) ---
    // Only compute/define DF outputs for rank==1. Anything else must be scrubbed
    // to avoid stale values leaking into downstream tables / plots.
    new_features.par_iter_mut().for_each(|psm| {
        if psm.core.rank != 1 {
            // Rank!=1: DF outputs are undefined by contract; scrub EVERYTHING to None.

            // --- final DF outputs (rank1-only) ---
            psm.decoy_free_p_value = None;
            psm.decoy_free_pep = None;
            psm.decoy_free_score = None;
            psm.decoy_free_q_value = None;
            psm.decoy_free_peptide_q = None;
            psm.decoy_free_protein_q = None;

            // --- per-method p streams (rank1-only) ---
            psm.p_mom = None;
            psm.p_mle = None;
            psm.p_lo = None;

            // --- MSFDR family p streams (HARD rank1-only) ---
            psm.p_msfdr = None;
            psm.p_1smix = None;
            psm.p_2smix = None;

            psm.p_nokoi = None;

            // --- per-method pep streams (rank1-only) ---
            psm.pep_mom = None;
            psm.pep_mle = None;
            psm.pep_lo = None;

            // --- MSFDR family pep streams (HARD rank1-only) ---
            psm.pep_msfdr = None;
            psm.pep_1smix = None;
            psm.pep_2smix = None;

            psm.pep_nokoi = None;

            // --- per-method q streams (rank1-only) ---
            psm.q_mom = None;
            psm.q_mle = None;
            psm.q_lo = None;

            // --- MSFDR family q streams (HARD rank1-only) ---
            psm.q_msfdr = None;
            psm.q_1smix = None;
            psm.q_2smix = None;

            psm.q_nokoi = None;
        }
    });

    // --- CALIBRATION (replaces old global hyperscore PAVA) ---
    // Calibrate the *chosen* p-value (rank1-only; stored in decoy_free_p_value)
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

        // LO-adjusted ranking key: use LO evidence scale (-log10(p_lo)), larger = better.
        // Fail-closed if p_lo is missing/non-finite.
        let lo_rank_key = |f: &DfFeature| -> f64 {
            match f.p_lo {
                Some(p) => {
                    let p = (p as f64).clamp(1e-300, 1.0);
                    if p.is_finite() {
                        -p.log10()
                    } else {
                        f64::NEG_INFINITY
                    }
                }
                None => f64::NEG_INFINITY,
            }
        };

        if log::log_enabled!(log::Level::Debug) {
            // What the calibration block will actually use as its quality key.
            let quality_desc = match settings.model_fit {
                ModelFit::Moments | ModelFit::Mle => {
                    "quality=tev(feature) (TEV-normalized hyperscore)"
                }
                ModelFit::LowerOrder => match settings.lo_rank_key {
                    LoRankKey::Hyperscore => "quality=hyperscore",
                    LoRankKey::LoAdjusted => "quality=-log10(p_lo) (LO evidence)",
                },
                ModelFit::Msfdr | ModelFit::Msfdr1Smix | ModelFit::Msfdr2Smix | ModelFit::Nokoi => {
                    "quality=-log10(p_used)"
                }
                ModelFit::Ensemble => "quality=decoy_free_score",
            };

            log::debug!(
                "DF DEBUG calibration(chosen): model_fit={:?} lo_rank_key={:?} {}",
                settings.model_fit,
                settings.lo_rank_key,
                quality_desc
            );
        }

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
                    ModelFit::Msfdr1Smix | ModelFit::Msfdr2Smix | ModelFit::Ensemble => {
                        f.decoy_free_p_value.map(|v| v as f64)?
                    }
                    ModelFit::Nokoi => f.p_nokoi.map(|v| v as f64)?,
                };

                // quality key: larger = better (sorted descending)
                let quality: f64 = match settings.model_fit {
                    ModelFit::Moments | ModelFit::Mle => tev(f).unwrap_or(f64::NEG_INFINITY),

                    ModelFit::LowerOrder => match settings.lo_rank_key {
                        LoRankKey::Hyperscore => f.core.hyperscore as f64,
                        LoRankKey::LoAdjusted => lo_rank_key(f),
                    },

                    // method-aligned ranking:
                    ModelFit::Msfdr | ModelFit::Msfdr1Smix | ModelFit::Msfdr2Smix => {
                        neg_log10_p(p_used)
                    }
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

        // LO-adjusted ranking key: use LO evidence scale (-log10(p_lo)), larger = better.
        // Fail-closed if p_lo is missing/non-finite.
        let lo_rank_key = |f: &DfFeature| -> f64 {
            match f.p_lo {
                Some(p) => {
                    let p = (p as f64).clamp(1e-300, 1.0);
                    if p.is_finite() {
                        -p.log10()
                    } else {
                        f64::NEG_INFINITY
                    }
                }
                None => f64::NEG_INFINITY,
            }
        };

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
                // IMPORTANT: do NOT overwrite pep_mom here; pep_mom is a local-FDR proxy, not p.
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
                // IMPORTANT: do NOT overwrite pep_mle here; pep_mle is a local-FDR proxy, not p.
            }
        }

        if log::log_enabled!(log::Level::Debug) {
            let desc = match settings.lo_rank_key {
                LoRankKey::Hyperscore => "quality=hyperscore",
                LoRankKey::LoAdjusted => "quality=-log10(p_lo) (LO evidence)",
            };
            log::debug!(
                "DF DEBUG calibration(per-method LO): lo_rank_key={:?} {}",
                settings.lo_rank_key,
                desc
            );
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
                // IMPORTANT: do NOT overwrite pep_lo here; pep_lo is a local-FDR proxy, not p.;
            }
        }

        // MSFDR (seeded stream): rank by -log10(p_msfdr) (method-aligned)
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

        // MSFDR 1SMix: rank by -log10(p_1smix) (method-aligned)
        {
            let rows: Vec<(f64, usize, f64)> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| {
                    let f = &new_features[i];
                    let p = f.p_1smix? as f64;
                    Some((neg_log10_p(p), i, p))
                })
                .collect();

            for (i, pcal) in calibrate(rows) {
                new_features[i].p_1smix = Some(pcal as f32);
                // Calibration Rule:
                // - OK to calibrate p_1smix
                // - DO NOT overwrite pep_1smix (posterior probability)
            }
        }

        // MSFDR 2SMix: rank by -log10(p_2smix) (method-aligned)
        {
            let rows: Vec<(f64, usize, f64)> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| {
                    let f = &new_features[i];
                    let p = f.p_2smix? as f64;
                    Some((neg_log10_p(p), i, p))
                })
                .collect();

            for (i, pcal) in calibrate(rows) {
                new_features[i].p_2smix = Some(pcal as f32);
                // PHASE 6 RULE:
                // - OK to calibrate p_2smix
                // - DO NOT overwrite pep_2smix (posterior probability)
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

    // Diagnostics for MSFDR 1SMix / 2SMix p streams (only if present)
    {
        let rank1_p_1smix: Vec<f64> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                new_features[i]
                    .p_1smix
                    .map(|v| (v as f64).clamp(0.0, 1.0).max(1e-300))
            })
            .collect();
        if !rank1_p_1smix.is_empty() {
            summarize_pvec("rank1_p_1smix (pre-q)", &rank1_p_1smix);
        }

        let rank1_p_2smix: Vec<f64> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                new_features[i]
                    .p_2smix
                    .map(|v| (v as f64).clamp(0.0, 1.0).max(1e-300))
            })
            .collect();
        if !rank1_p_2smix.is_empty() {
            summarize_pvec("rank1_p_2smix (pre-q)", &rank1_p_2smix);
        }
    }

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

    // ============================================================
    // Chosen DF q-values for MSFDR variants are PEP-derived
    // ============================================================
    if matches!(
        settings.model_fit,
        ModelFit::Msfdr | ModelFit::Msfdr1Smix | ModelFit::Msfdr2Smix | ModelFit::Nokoi
    ) {
        // Build rows sorted by "quality" (best-first), then q[k]=mean(pep[0..k]) with monotone pass.
        let mut rows: Vec<(f64, usize, f64)> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                let f = &new_features[i];
                let pep = f.decoy_free_pep? as f64;
                if !pep.is_finite() {
                    return None;
                }
                let score_key = df_rank_score(f); // already defined/used elsewhere
                Some((score_key, i, pep.clamp(0.0, 1.0).max(1e-300)))
            })
            .collect();

        // best first
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));

        // cumulative mean
        let mut q_sorted: Vec<f64> = Vec::with_capacity(rows.len());
        let mut cum = 0.0f64;
        for (k, (_, _, pep)) in rows.iter().enumerate() {
            cum += *pep;
            q_sorted.push((cum / ((k + 1) as f64)).clamp(0.0, 1.0));
        }

        // enforce monotone non-decreasing with worsening quality
        for i in (0..q_sorted.len().saturating_sub(1)).rev() {
            q_sorted[i] = q_sorted[i].min(q_sorted[i + 1]);
        }

        // write back
        for (k, (_score, feat_idx, _pep)) in rows.into_iter().enumerate() {
            set_df_q_value(&mut new_features[feat_idx], q_sorted[k] as f32);
        }
    } else {
        // Existing BH/Storey path for all other methods (including Ensemble)
        let q_values = match settings.type_ {
            FdrType::Bh => stats::bh_q_value(&rank1_p),

            FdrType::Storey => {
                let pi0_opt = estimate_pi0_from_reference_grid(&rank1_p_ref, settings);
                match pi0_opt {
                    Some(pi0) => storey_q_value_with_pi0(&rank1_p, pi0, settings),
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
        }
    }

    // Phase 8: debug-build assertions for chosen DF q stream (present-only)
    #[cfg(debug_assertions)]
    {
        for &i in &work.rank1_indices {
            if let Some(q) = new_features[i].decoy_free_q_value {
                debug_assert!(
                    q.is_finite() && q > 0.0 && q <= 1.0,
                    "DF ASSERT decoy_free_q_value invalid at feature_idx={}: {}",
                    i,
                    q
                );
            }
        }
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

    // =========================================================================
    // MSFDR 1SMix / 2SMix q-values are ALWAYS PEP-derived
    // -------------------------------------------------------------------------
    // Contract:
    // - q_1smix / q_2smix NEVER use BH/Storey, regardless of settings.type_.
    // - q_1smix / q_2smix are computed ONLY from the model-derived PEP streams
    //   via cumulative mean after sorting by quality (best first).
    // - Clear rank1 q_1smix/q_2smix first to prevent stale leakage on runs where
    //   pep_1smix/pep_2smix are absent.
    // =========================================================================
    for &i in &work.rank1_indices {
        new_features[i].q_1smix = None;
        new_features[i].q_2smix = None;
    }

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

    // Phase 8: debug-build assertions for per-method q streams (present-only)
    #[cfg(debug_assertions)]
    {
        let assert_q = |name: &str, v: Option<f32>, idx: usize| {
            if let Some(x) = v {
                debug_assert!(
                    x.is_finite() && x > 0.0 && x <= 1.0,
                    "DF ASSERT {} invalid at feature_idx={}: {}",
                    name,
                    idx,
                    x
                );
            }
        };

        for &i in &work.rank1_indices {
            let f = &new_features[i];
            assert_q("q_mom", f.q_mom, i);
            assert_q("q_mle", f.q_mle, i);
            assert_q("q_lo", f.q_lo, i);
            assert_q("q_msfdr", f.q_msfdr, i);
            assert_q("q_nokoi", f.q_nokoi, i);
            // q_1smix/q_2smix asserted after mixture-q write below (next block)
        }
    }

    // =========================================================================
    // Mixture q-values for MSFDR 1SMix / 2SMix (PEP path)
    //   q[k] = mean(pep[0..k]) after sorting by quality (best first)
    //   sorted by score (best first): tev(feature) primary; fallback decoy_free_scoree
    //   This computation is intentionally independent of settings.type_.
    // =========================================================================
    {
        // Helper: compute cumulative-mean q over PEPs, sorted by quality desc.
        // IMPORTANT: THIS MUST REMAIN PEP-ONLY.
        // Do NOT replace with BH/Storey (even if settings.type_ changes).
        // Returns (feature_idx, q) pairs..
        let mixture_q_from_pep = |mut rows: Vec<(f64, usize, f64)>| -> Vec<(usize, f64)> {
            if rows.is_empty() {
                return Vec::new();
            }

            // best first
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));

            // cumulative mean
            let mut q_sorted: Vec<f64> = Vec::with_capacity(rows.len());
            let mut cum = 0.0f64;
            for (k, (_, _, pep)) in rows.iter().enumerate() {
                cum += *pep;
                let q = (cum / ((k + 1) as f64)).clamp(0.0, 1.0);
                q_sorted.push(q);
            }

            // enforce monotone non-decreasing with worsening quality
            for i in (0..q_sorted.len().saturating_sub(1)).rev() {
                q_sorted[i] = q_sorted[i].min(q_sorted[i + 1]);
            }

            rows.into_iter()
                .enumerate()
                .map(|(k, (_quality, feat_idx, _pep))| (feat_idx, q_sorted[k]))
                .collect()
        };

        // ---- 1SMix ----
        let rows_1smix: Vec<(f64, usize, f64)> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                let f = &new_features[i];

                // only where pep exists
                let pep = f.pep_1smix? as f64;
                if !pep.is_finite() {
                    return None;
                }

                // Step 3.5: "sorted by score" means TEV (primary) else decoy_free_score
                let score_key = df_rank_score(f);

                Some((score_key, i, pep.clamp(0.0, 1.0)))
            })
            .collect();

        for (i, q) in mixture_q_from_pep(rows_1smix) {
            if new_features[i].core.rank == 1 && new_features[i].pep_1smix.is_some() {
                new_features[i].q_1smix = Some(q as f32);
            }
        }

        // ---- 2SMix ----
        let rows_2smix: Vec<(f64, usize, f64)> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                let f = &new_features[i];

                // only where pep exists
                let pep = f.pep_2smix? as f64;
                if !pep.is_finite() {
                    return None;
                }

                // Step 3.5: "sorted by score" means TEV (primary) else decoy_free_score
                let score_key = df_rank_score(f);

                Some((score_key, i, pep.clamp(0.0, 1.0)))
            })
            .collect();

        for (i, q) in mixture_q_from_pep(rows_2smix) {
            if new_features[i].core.rank == 1 && new_features[i].pep_2smix.is_some() {
                new_features[i].q_2smix = Some(q as f32);
            }
        }

        // Phase 8: debug-build assertions for mixture q streams (present-only)
        #[cfg(debug_assertions)]
        {
            for &i in &work.rank1_indices {
                if let Some(q) = new_features[i].q_1smix {
                    debug_assert!(
                        q.is_finite() && q > 0.0 && q <= 1.0,
                        "DF ASSERT q_1smix invalid at feature_idx={}: {}",
                        i,
                        q
                    );
                }
                if let Some(q) = new_features[i].q_2smix {
                    debug_assert!(
                        q.is_finite() && q > 0.0 && q <= 1.0,
                        "DF ASSERT q_2smix invalid at feature_idx={}: {}",
                        i,
                        q
                    );
                }
            }
        }
    }

    // Diagnostics for MSFDR 1SMix / 2SMix q streams (only if present)
    {
        let any_q_1smix = work
            .rank1_indices
            .iter()
            .any(|&i| new_features[i].q_1smix.is_some());
        if any_q_1smix {
            summarize_q(
                "POSTQ q_1smix (rank1, present-only)",
                work.rank1_indices
                    .iter()
                    .filter_map(|&i| new_features[i].q_1smix),
            );
        }

        let any_q_2smix = work
            .rank1_indices
            .iter()
            .any(|&i| new_features[i].q_2smix.is_some());
        if any_q_2smix {
            summarize_q(
                "POSTQ q_2smix (rank1, present-only)",
                work.rank1_indices
                    .iter()
                    .filter_map(|&i| new_features[i].q_2smix),
            );
        }
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
    // - rank==1 only
    // - TARGETS ONLY for aggregation
    // - read ONLY decoy-free q stream
    for feat in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let q = df_q_value(feat); // <-- DF stream accessor (decoy_free_q_value)
        let peptide = db[feat.core.peptide_idx].to_string();

        best_q
            .entry(peptide)
            .and_modify(|v| *v = v.min(q))
            .or_insert(q);
    }

    // Write ONLY DF peptide q (rank1-only by contract)
    // - rank==1: populate decoy_free_peptide_q from best_q map (fail-closed to 1.0 if missing)
    // - rank!=1: scrub decoy_free_peptide_q AND MSFDR family streams (hard rank1-only)
    for feat in features.iter_mut() {
        // Only rank==1 AND target rows get DF peptide q.
        // Everything else must be scrubbed to None to prevent leakage / miscounting.
        if feat.core.rank != 1 || feat.core.label != 1 {
            feat.decoy_free_peptide_q = None;

            // MSFDR family (p/pep/q) must be None for non-rank1 (and we also keep them
            // off non-target rows to avoid confusion/leakage in DF mode).
            feat.p_msfdr = None;
            feat.pep_msfdr = None;
            feat.q_msfdr = None;

            feat.p_1smix = None;
            feat.pep_1smix = None;
            feat.q_1smix = None;

            feat.p_2smix = None;
            feat.pep_2smix = None;
            feat.q_2smix = None;

            continue;
        }

        // rank==1, target
        let peptide = db[feat.core.peptide_idx].to_string();
        let q = best_q.get(&peptide).copied().unwrap_or(1.0);
        feat.decoy_free_peptide_q = Some(q);
    }

    best_q.values().filter(|&&q| q <= threshold).count()
}

fn combine_cauchy(p: &[f64]) -> f64 {
    // Cauchy combination (robust under dependence)
    // p_i in (0,1); clamp for numerical stability.
    let m = p.len() as f64;
    let mut t_sum = 0.0;
    for &pi in p {
        let pi = pi.clamp(0.0, 1.0).max(1e-300).min(1.0 - 1e-16);
        t_sum += ((0.5 - pi) * std::f64::consts::PI).tan();
    }
    let t = t_sum / m;
    let p_c = 0.5 - (t.atan() / std::f64::consts::PI);
    p_c.clamp(0.0, 1.0).max(1e-300)
}

fn combine_sidak_minp(p: &[f64]) -> f64 {
    // Sidak adjustment on min p: 1 - (1 - minP)^m
    let m = p.len() as f64;
    let pmin = p
        .iter()
        .copied()
        .fold(1.0_f64, |a, b| a.min(b.clamp(0.0, 1.0).max(1e-300)));
    let p = 1.0 - (1.0 - pmin).powf(m);
    p.clamp(0.0, 1.0).max(1e-300)
}

fn combine_median_beta(p: &[f64]) -> f64 {
    // Order-statistic consensus:
    // Take the median p (lower-median for even m), then convert to a valid p-value via Beta CDF:
    // U_(k) ~ Beta(k, m-k+1) under i.i.d. Uniform(0,1) null.
    let m = p.len();
    if m == 0 {
        return 1.0;
    }

    let mut v: Vec<f64> = p
        .iter()
        .copied()
        .map(|x| x.clamp(0.0, 1.0).max(1e-300).min(1.0 - 1e-16))
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // lower-median for even m: k = (m+1)/2
    let k: usize = (m + 1) / 2; // 1-based
    let p_k = v[k - 1];

    let a = k as f64;
    let b = (m - k + 1) as f64;
    let beta = Beta::new(a, b).unwrap();
    beta.cdf(p_k).clamp(0.0, 1.0).max(1e-300)
}

fn combine_stouffer(p: &[f64]) -> f64 {
    // Stouffer Z (equal weights):
    // z_i = Phi^{-1}(1 - p_i), Z = mean(z_i) * sqrt(m), p = 1 - Phi(Z)
    let m = p.len();
    if m == 0 {
        return 1.0;
    }

    let n01 = Normal::new(0.0, 1.0).unwrap();

    let mut z_sum = 0.0;
    for &pi in p {
        let pi = pi.clamp(0.0, 1.0).max(1e-300).min(1.0 - 1e-16);
        z_sum += n01.inverse_cdf(1.0 - pi);
    }

    let z = z_sum / (m as f64).sqrt();
    (1.0 - n01.cdf(z)).clamp(0.0, 1.0).max(1e-300)
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

    for feat in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        // DF aggregation contract: read ONLY decoy_free_p_value stream
        // Use accessor (fail-closed to 1.0), but SKIP if the underlying field is absent,
        // so don't inject conservative 1.0s into Fisher combining.
        if feat.decoy_free_p_value.is_none() {
            continue;
        }
        let p = (df_p_value(feat) as f64).clamp(0.0, 1.0).max(1e-300);

        // Unique-only protein inference using the same source as TSV `num_proteins`
        let peptide = &db[feat.core.peptide_idx];
        if peptide.proteins.len() != 1 {
            continue;
        }

        let protein_key = peptide.proteins(&db.decoy_tag, db.generate_decoys);
        let peptide_seq = peptide.to_string();

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

        let combined_p = match settings.protein_p_combine {
            crate::input::ProteinPCombine::Fisher => {
                stats::combine_fisher(&p_vec).clamp(0.0, 1.0).max(1e-300)
            }
            crate::input::ProteinPCombine::Cauchy => combine_cauchy(&p_vec),
            crate::input::ProteinPCombine::SidakMinP => combine_sidak_minp(&p_vec),
        };
        protein_keys.push(key);
        protein_p_values.push(combined_p);
    }

    // If no proteins, write fail-closed (rank1-only by contract) and return 0.
    if protein_p_values.is_empty() {
        for feat in features.iter_mut() {
            if feat.core.rank != 1 || feat.core.label != 1 {
                feat.decoy_free_protein_q = None;

                feat.p_msfdr = None;
                feat.pep_msfdr = None;
                feat.q_msfdr = None;

                feat.p_1smix = None;
                feat.pep_1smix = None;
                feat.q_1smix = None;

                feat.p_2smix = None;
                feat.pep_2smix = None;
                feat.q_2smix = None;
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
        if feat.core.rank != 1 || feat.core.label != 1 {
            feat.decoy_free_protein_q = None;

            feat.p_msfdr = None;
            feat.pep_msfdr = None;
            feat.q_msfdr = None;

            feat.p_1smix = None;
            feat.pep_1smix = None;
            feat.q_1smix = None;

            feat.p_2smix = None;
            feat.pep_2smix = None;
            feat.q_2smix = None;

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
