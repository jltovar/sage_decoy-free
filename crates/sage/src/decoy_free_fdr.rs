/* =============================================================================
Decoy-Free (DF) FDR Logic Contract

This comment defines the invariants for Decoy-Free mode.

1) Scope: DF is rank==1 only
-----------------------------------
- All Decoy-Free outputs are defined only for rank==1 PSMs.
- For rank!=1, every DF output and every DF auxiliary/per-method output must be
  scrubbed to None.
- No stale DF values may survive on non-rank-1 rows.

2) Final-output boundary
-------------------------------
The only DF outputs that represent the selected final answer are:

  decoy_free_p_value
  decoy_free_pep
  decoy_free_score
  decoy_free_q_value
  decoy_free_peptide_q
  decoy_free_protein_q

All other DF-family fields are auxiliary stage streams or per-method streams and
must never be treated as the final answer. This includes, for example:

  p_msfdr / pep_msfdr / q_msfdr
  p_1smix / pep_1smix / q_1smix
  p_2smix / pep_2smix / q_2smix
  p_mom / pep_mom / q_mom
  p_nokoi / pep_nokoi / q_nokoi
  and any other expert-specific DF fields

3) Evidence-stream contract
-----------------------------------
Decoy-Free engines may produce either p-like evidence streams or PEP-like
streams, but the base writeback normalizes the selected evidence into the
active decoy_free_* output fields before peptide/protein inference.

Current intended stream semantics:

- Moments / MLE:
    p_mom / p_mle are Gumbel-tail p-value streams.
    pep_mom / pep_mle are calibrated local-FDR/PEP-like streams derived from
    those p-values.

- LowerOrder:
    p_lo is a lower-order p-like stream evaluated on the LO TEV scale.
    pep_lo is the calibrated local-FDR/PEP-like stream derived from p_lo.

- MSFDR variants:
    p_msfdr / p_1smix / p_2smix are fitted-null tail p-like streams.
    pep_msfdr / pep_1smix / pep_2smix are calibrated PEP-like streams derived
    from those p-like streams.
    Raw mixture posterior probabilities must not be treated as final calibrated
    PEPs unless explicitly validated.

- Nokoi:
    p_nokoi is an empirical null-survival p-like stream from Nokoi classifier
    scores.
    pep_nokoi is a calibrated PEP-like stream derived from p_nokoi.
    Raw 1 - P(target) from the logistic classifier must not be treated as PEP.

- Ensemble:
    Ensemble combines expert PEP-like streams according to the configured
    ensemble_pep_combiner and expert weights.
    The result is an empirical consensus PEP-like stream, not a raw p-value
    stream.

4) Final-stream contract before peptide/protein inference
----------------------------------------------------------------
Before peptide or protein inference runs, the active final DF PSM stream must be
fully activated into the decoy_free_* fields.

Layer definitions:
- Layer 1 = base DF consensus/model stream
- Layer 2 = physical/DART-updated DF stream
- Layer 3 = reproducibility-updated DF stream

Required final-stream behavior:
- If the active final stream is p-value-native:
    decoy_free_p_value is populated.
    decoy_free_pep and decoy_free_score may also be populated as calibrated
    auxiliary posterior-like evidence, but decoy_free_q_value is computed from
    the active p-value stream using the configured p-value FDR procedure.

- If the active final stream is PEP-native:
    decoy_free_p_value must be None unless a valid aligned p-value stream is
    explicitly available and explicitly selected.
    decoy_free_pep / decoy_free_score / decoy_free_q_value must come from the
    active PEP-native stream.

- Ensemble, Layer 2, and Layer 3 are PEP-native by design.
- Standalone MSFDR, MSFDR-1smix, MSFDR-2smix, and Nokoi are treated as
  PEP-native final streams in the current implementation.
- Standalone Moments, MLE, and LowerOrder remain p-value-native final streams
  in the current implementation.

Peptide and protein inference consume only this finalized active stream.
They must not infer the active layer from stale auxiliary fields.

5) Q-value semantics
----------------------------
For PEP-native final DF streams, q-values must be computed from cumulative means
of PEP after sorting best-first by the stream’s defined quality key, followed by
a monotone pass.

Formally:
  q[k] = mean(pep[0..k])   after best-first sorting

This applies to:
- Ensemble final q-values
- standalone MSFDR / MSFDR-1smix / MSFDR-2smix / Nokoi final q-values
- any final physical-rescue or reproducibility-updated PEP-native stream
- auxiliary q_1smix and q_2smix streams in the current implementation

PEP-native final streams must not use BH or Storey directly.

For p-value-native DF streams:
- q-values may use the configured p-value FDR procedure, such as BH or Storey.
- In the current implementation, auxiliary q_mom, q_mle, q_lo, q_msfdr, and
  q_nokoi are computed from their p_* streams using the configured p-value FDR
  procedure.
- Any calibrated pep_* field written alongside a p-value-native stream is
  auxiliary unless that stream is explicitly activated as PEP-native.

6) Peptide/protein inference
-----------------------------------
Peptide and protein inference consume only the finalized active decoy_free_*
stream.

For PEP-native peptide inference:
- The peptide evidence is derived from the best supporting PSM-level PEP, with
  at most bounded support from additional strong PSMs.
- Repeated spectra for the same peptide are corroborating evidence and must not
  be penalized with a count-based Šidák selected-min correction.

For p-value-native peptide/protein inference:
- Method-specific p-value combination rules may be used only when the active
  stream is genuinely p-value-native.

Protein inference must respect the selected p-native vs PEP-native semantics of
the finalized active stream.

7) Failure behavior is fail-closed
-----------------------------------------
If DF engines do not exist, fail to fit, or do not run:
- DF outputs must not remain permissively populated from stale values.
- The implementation may either clear DF outputs to None or write conservative
  fail-closed rank==1 defaults.

Current allowed fail-closed rank==1 defaults are:
  p=1, pep=1, score=0, q=1, peptide_q=1, protein_q=1

Rank!=1 must always remain None.

8) Best-first sorting key must be explicit
-------------------------------------------------
Whenever this module says “sorted by score” or “sorted by quality (best-first)”,
the sorting key must be the method-appropriate key for that block.

Examples:
- For PEP-native q-values:
    sort by the stream’s defined quality key or by increasing PEP if no
    independent quality key is available.

- For LowerOrder:
    use the configured LO quality key, such as LO-adjusted p-like evidence or
    hyperscore, according to lo_rank_key.

- For Grenander calibration of a p-like stream:
    sort by the p-like value itself. The p-like stream must already encode the
    method-specific evidence ordering.

Never rely on an implicit or ambiguous “score” definition.

============================================================================= */

use crate::database::IndexedDatabase;
use crate::input::LoStratify;
use crate::input::{
    BoundedAuxUpdateSpace, DartNullRtModel, DartTrueRtModel, EnsemblePepCombiner, FdrSettings,
    FdrType, JointMode, L3AnchorMode, L3RescueMode, ModelFit, PeptidePCombine, PhysicalAnchorMode,
    PhysicalRescueMode,
};
use crate::lfq::{Peak, PrecursorId};
use crate::ml::lower_order::{
    fit_decoy_free_model, fit_gumbel_mle, fit_gumbel_moments, LowerOrderModel,
};
use crate::ml::msfdr::{Msfdr1SmixModel, Msfdr2SmixModel, MsfdrSeededModel};
use crate::ml::nokoi;
use crate::ml::stats;
use crate::scoring::{DfFeature, TdcFeature};
use fnv::{FnvHashMap, FnvHashSet};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Gumbel};
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Rank1Computed {
    idx: usize,

    // per-method p's
    p_mom: f64,
    p_mle: f64,
    p_lo: f64,

    // MSFDR family p's (independent streams)
    p_msfdr: Option<f64>, // seeded MSFDR fitted-null tail p-like stream
    p_1smix: Option<f64>,
    p_2smix: Option<f64>,

    p_nokoi: Option<f64>,

    // final DF p output (pep/score computed later, after lfdr mapping)
    p_final: f64,
}

#[derive(Clone, Debug)]
struct RtReliabilitySummary {
    pub rt_sigma_global: Option<f64>,
    pub runwise_rt_sigma: Vec<(usize, f64)>,
    pub reliability: f64,
    pub fail_closed_hint: bool,
}

#[derive(Clone, Debug)]
struct ImsReliabilitySummary {
    pub ims_sigma_global: Option<f64>,
    pub runwise_ims_sigma: Vec<(usize, f64)>,
    pub reliability: f64,
    pub fail_closed_hint: bool,
}

#[derive(Clone, Debug)]
struct JointPhysicalSummary {
    pub joint_reliability: f64,
    pub fail_closed_hint: bool,
}

// =============================================================================
// Helpers (math, calibration, parsing, diagnostics, and model fitting)
// =============================================================================

// -----------------------------------------------------------------------------
// 1) TEV normalization helper
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
// 2) Simple statistics helpers (median / trimmed / winsor / quantile / weights)
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

fn mean_f64(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let m = v.iter().sum::<f64>() / (v.len() as f64);
    m.is_finite().then_some(m)
}

fn trimmed_mean(v: &mut [f64], trim_frac: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let t = ((trim_frac.clamp(0.0, 0.49)) * (n as f64)).floor() as usize;
    let lo = t;
    let hi = n.saturating_sub(t);
    if lo >= hi {
        return None;
    }
    let slice = &v[lo..hi];
    mean_f64(slice)
}

fn second_best_f64(v: &mut [f64]) -> Option<f64> {
    if v.len() < 2 {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[1])
}

fn winsorized_mean(v: &mut [f64], trim_frac: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let t = ((trim_frac.clamp(0.0, 0.49)) * (n as f64)).floor() as usize;
    if t == 0 {
        return mean_f64(v);
    }
    if 2 * t >= n {
        return None;
    }

    let lo_val = v[t];
    let hi_val = v[n - t - 1];

    let mut acc = 0.0;
    for (i, &x) in v.iter().enumerate() {
        let y = if i < t {
            lo_val
        } else if i >= n - t {
            hi_val
        } else {
            x
        };
        acc += y;
    }

    let m = acc / (n as f64);
    m.is_finite().then_some(m)
}

fn quantile_f64(mut v: Vec<f64>, q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));

    let n = v.len();
    if n == 1 {
        return Some(v[0]);
    }

    let q = q.clamp(0.0, 1.0);
    let pos = q * ((n - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    if lo == hi {
        Some(v[lo])
    } else {
        let w = pos - (lo as f64);
        Some(v[lo] * (1.0 - w) + v[hi] * w)
    }
}

fn top_k_mean(mut v: Vec<f64>, k: usize) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b)); // lower PEP is better
    let kk = k.max(1).min(v.len());
    mean_f64(&v[..kk])
}

fn geometric_mean(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut acc = 0.0;
    for &x in v {
        if !x.is_finite() || x <= 0.0 {
            return None;
        }
        acc += x.ln();
    }
    let g = (acc / (v.len() as f64)).exp();
    g.is_finite().then_some(g)
}

fn logit_mean(v: &[f64], eps: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let eps = eps.clamp(1e-12, 1e-2);
    let mut acc = 0.0;
    for &x in v {
        if !x.is_finite() {
            return None;
        }
        let p = x.clamp(eps, 1.0 - eps);
        acc += (p / (1.0 - p)).ln();
    }
    let z = acc / (v.len() as f64);
    let p = 1.0 / (1.0 + (-z).exp());
    p.is_finite().then_some(p)
}

fn normalize_weighted_pairs(peps: &[f64], weights: &[f64]) -> Vec<(f64, f64)> {
    let mut pairs: Vec<(f64, f64)> = peps
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .filter(|(p, w)| p.is_finite() && w.is_finite() && *w >= 0.0)
        .map(|(p, w)| (p.clamp(1e-300, 1.0), w))
        .collect();

    if pairs.is_empty() {
        return Vec::new();
    }

    let sum_w: f64 = pairs.iter().map(|(_, w)| *w).sum();
    if sum_w <= 0.0 || !sum_w.is_finite() {
        let eq = 1.0 / (pairs.len() as f64);
        for (_, w) in &mut pairs {
            *w = eq;
        }
        return pairs;
    }

    for (_, w) in &mut pairs {
        *w /= sum_w;
    }
    pairs
}

fn weighted_mean(peps: &[f64], weights: &[f64]) -> Option<f64> {
    let pairs = normalize_weighted_pairs(peps, weights);
    if pairs.is_empty() {
        return None;
    }
    let m = pairs.iter().map(|(p, w)| p * w).sum::<f64>();
    m.is_finite().then_some(m)
}

fn weighted_median(peps: &[f64], weights: &[f64]) -> Option<f64> {
    let mut pairs = normalize_weighted_pairs(peps, weights);
    if pairs.is_empty() {
        return None;
    }

    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut cdf = 0.0;
    for (p, w) in pairs {
        cdf += w;
        if cdf >= 0.5 {
            return Some(p);
        }
    }
    Some(1.0)
}

// -----------------------------------------------------------------------------
// 3) Ensemble combiner for expert PEPs
//
// IMPORTANT:
// This function produces an empirical consensus score on the PEP scale by
// combining expert-specific PEP estimates. The result is used operationally
// as a PEP-like quantity for ranking and cumulative-mean q-value estimation,
// but it should not be described as a formally calibrated posterior
// probability unless that calibration is validated externally.
// -----------------------------------------------------------------------------

/// Combine multiple expert PEP estimates into a single empirical consensus
/// score on the PEP scale.
///
/// The output is bounded to [1e-300, 1.0] and is suitable for downstream
/// rank ordering and cumulative-mean q-value construction in the ensemble
/// path. It is intentionally treated as a PEP-like consensus score rather
/// than a guaranteed calibrated posterior.

fn combine_peps(
    peps: &[f64],
    weights: &[f64],
    how: EnsemblePepCombiner,
    trim_frac: f64,
    quantile: f64,
    top_k: usize,
    logit_eps: f64,
) -> f64 {
    if peps.is_empty() {
        return 1.0;
    }

    let valid_peps: Vec<f64> = peps
        .iter()
        .copied()
        .filter(|p| p.is_finite())
        .map(|p| p.clamp(1e-300, 1.0))
        .collect();

    if valid_peps.is_empty() {
        return 1.0;
    }

    match how {
        EnsemblePepCombiner::Median => median_f64(valid_peps).unwrap_or(1.0),

        EnsemblePepCombiner::TrimmedMean => {
            let mut tmp = valid_peps;
            trimmed_mean(&mut tmp, trim_frac).unwrap_or(1.0)
        }

        EnsemblePepCombiner::Max => valid_peps.into_iter().fold(0.0_f64, |a, b| a.max(b)),

        EnsemblePepCombiner::Mean => mean_f64(&valid_peps).unwrap_or(1.0).clamp(1e-300, 1.0),

        EnsemblePepCombiner::WeightedMean => weighted_mean(&valid_peps, weights)
            .unwrap_or(1.0)
            .clamp(1e-300, 1.0),

        EnsemblePepCombiner::WeightedMedian => weighted_median(&valid_peps, weights)
            .unwrap_or(1.0)
            .clamp(1e-300, 1.0),

        EnsemblePepCombiner::WinsorizedMean => {
            let mut tmp = valid_peps;
            winsorized_mean(&mut tmp, trim_frac)
                .unwrap_or(1.0)
                .clamp(1e-300, 1.0)
        }

        EnsemblePepCombiner::Quantile => quantile_f64(valid_peps, quantile)
            .unwrap_or(1.0)
            .clamp(1e-300, 1.0),

        EnsemblePepCombiner::TopKMean => top_k_mean(valid_peps, top_k)
            .unwrap_or(1.0)
            .clamp(1e-300, 1.0),

        EnsemblePepCombiner::GeometricMean => geometric_mean(&valid_peps)
            .unwrap_or(1.0)
            .clamp(1e-300, 1.0),

        EnsemblePepCombiner::LogitMean => logit_mean(&valid_peps, logit_eps)
            .unwrap_or(1.0)
            .clamp(1e-300, 1.0),
    }
}

// -----------------------------------------------------------------------------
// 4) Feature field helpers (tiny setters/getters for DF streams)
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
// 5) Canonical evidence accessors
// -----------------------------------------------------------------------------
//
// IMPORTANT:
// - `tev(...)` below remains the raw hyperscore accessor used by the existing
//   non-LO DF code paths (Moments / MLE / MSFDR / sorting that already expect
//   raw hyperscore semantics).
// - `lo_tev(...)` is the PyLord-style LO score accessor and is what LO should
//   use for both fit-time and eval-time.
// - PyLord's Sage parser constructed:
//
//       tev = par_a * ln(p_value * num_candidates / par_n0)
//
//   For LO fitting, `par_a` and `par_n0` only induce an affine transform if
//   applied consistently to both fit and eval. So the drop-in Rust equivalent
//   below uses the monotone / affine-equivalent form:
//
//       lo_tev = -ln(p_value * num_candidates)
//
//   which preserves the PyLord score ordering and gives the LO fitter the
//   intended parser-style TEV scale.
//
// NOTE:
// If your `DfFeature` field names differ, map the two lines below to the fields
// that hold the original Sage `spectrum_p_value` and `scored_candidates`.
//
#[inline(always)]
fn tev(f: &DfFeature) -> Option<f64> {
    let x = f.core.hyperscore as f64;
    if x.is_finite() {
        Some(x)
    } else {
        None
    }
}

#[inline(always)]
fn lo_tev(f: &DfFeature) -> Option<f64> {
    // LowerOrder must preserve ordering and dynamic range in the high-confidence
    // tail. Avoid deriving the LO score from spectrum_p_value because that field
    // may already have lost tail resolution through f32 underflow or quantization.
    //
    // Use the original hyperscore scale as the LO score input; the LowerOrder
    // fitter estimates the null-tail transformation from the selected lower-order
    // ranks.
    let x = f.core.hyperscore as f64;
    if x.is_finite() {
        Some(x)
    } else {
        None
    }
}

// -----------------------------------------------------------------------------
// 6) DF reset helper (critical): clear all DF + per-method outputs
// -----------------------------------------------------------------------------

#[inline(always)]
fn clear_all_df_outputs(psm: &mut DfFeature, fail_closed_rank1: bool) {
    // clear final DF streams
    psm.decoy_free_p_value = None;
    psm.decoy_free_pep = None;
    psm.decoy_free_score = None;
    psm.decoy_free_q_value = None;
    psm.decoy_free_peptide_q = None;
    psm.decoy_free_protein_q = None;

    // clear Layer 1/base snapshots
    psm.decoy_free_p_value_base = None;
    psm.decoy_free_pep_base = None;
    psm.decoy_free_score_base = None;
    psm.decoy_free_q_base = None;

    // clear Layer 2 physical/DART streams
    psm.decoy_free_p_value_l2 = None;
    psm.decoy_free_pep_l2 = None;
    psm.decoy_free_score_l2 = None;
    psm.decoy_free_q_l2 = None;

    // clear Layer 3 reproducibility streams
    psm.decoy_free_pep_l3 = None;
    psm.decoy_free_score_l3 = None;
    psm.decoy_free_q_l3 = None;

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
// 7) DF rank-order score helpers
// -----------------------------------------------------------------------------

#[inline(always)]
fn df_rank_score(f: &DfFeature) -> f64 {
    tev(f).unwrap_or_else(|| f.decoy_free_score.unwrap_or(0.0) as f64)
}

#[inline(always)]
fn df_rank_score_base_pep(f: &DfFeature) -> f64 {
    f.decoy_free_score
        .map(|x| x as f64)
        .unwrap_or_else(|| tev(f).unwrap_or(0.0))
}

#[inline(always)]
fn df_rank_score_l2_pep(f: &DfFeature) -> f64 {
    f.decoy_free_score_l2.map(|x| x as f64).unwrap_or_else(|| {
        f.decoy_free_score
            .map(|x| x as f64)
            .unwrap_or_else(|| tev(f).unwrap_or(0.0))
    })
}

#[inline(always)]
fn df_rank_score_l3_pep(f: &DfFeature) -> f64 {
    f.decoy_free_score_l3.map(|x| x as f64).unwrap_or_else(|| {
        f.decoy_free_score_l2
            .map(|x| x as f64)
            .or_else(|| f.decoy_free_score.map(|x| x as f64))
            .unwrap_or_else(|| tev(f).unwrap_or(0.0))
    })
}

// -----------------------------------------------------------------------------
// 8) Protein-string classification (contam / entrapment)
// -----------------------------------------------------------------------------

#[inline]
fn is_contam_str(proteins: &str) -> bool {
    proteins.contains("Cont_")
}

#[inline]
fn is_entrapment_str(proteins: &str) -> bool {
    proteins.contains("|Ent_") || proteins.contains("Ent_")
}

#[inline]
fn df_unique_protein_key_for_feature(f: &DfFeature, db: &IndexedDatabase) -> Option<String> {
    let peptide = &db[f.core.peptide_idx];
    if peptide.proteins.len() != 1 {
        return None;
    }
    Some(peptide.proteins(&db.decoy_tag, db.generate_decoys))
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EntrapmentCounts {
    pub psms: usize,
    pub peptides: usize,
    pub proteins: usize,
}

pub fn calculate_entrapment_counts_df(
    features: &[DfFeature],
    db: &IndexedDatabase,
    peptide_fdr: f32,
    protein_fdr: f32,
) -> EntrapmentCounts {
    let mut counts = EntrapmentCounts::default();
    let mut peptide_set: FnvHashSet<String> = FnvHashSet::default();
    let mut protein_set: FnvHashSet<String> = FnvHashSet::default();

    for feat in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let peptide = &db[feat.core.peptide_idx];
        let protein_key = peptide.proteins(&db.decoy_tag, db.generate_decoys);

        if !is_entrapment_str(&protein_key) {
            continue;
        }

        if feat.decoy_free_q_value.unwrap_or(1.0) <= peptide_fdr {
            counts.psms += 1;
        }

        if feat.decoy_free_peptide_q.unwrap_or(1.0) <= peptide_fdr {
            peptide_set.insert(peptide.to_string());
        }

        if feat.decoy_free_protein_q.unwrap_or(1.0) <= protein_fdr {
            protein_set.insert(protein_key);
        }
    }

    counts.peptides = peptide_set.len();
    counts.proteins = protein_set.len();
    counts
}

pub fn calculate_entrapment_counts_tdc(
    features: &[TdcFeature],
    db: &IndexedDatabase,
    spectrum_fdr: f32,
    peptide_fdr: f32,
    protein_fdr: f32,
) -> EntrapmentCounts {
    let mut counts = EntrapmentCounts::default();
    let mut peptide_set: FnvHashSet<String> = FnvHashSet::default();
    let mut protein_set: FnvHashSet<String> = FnvHashSet::default();

    for feat in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let peptide = &db[feat.core.peptide_idx];
        let protein_key = peptide.proteins(&db.decoy_tag, db.generate_decoys);

        if !is_entrapment_str(&protein_key) {
            continue;
        }

        if feat.spectrum_q <= spectrum_fdr {
            counts.psms += 1;
        }

        if feat.peptide_q <= peptide_fdr {
            peptide_set.insert(peptide.to_string());
        }

        if feat.protein_q <= protein_fdr {
            protein_set.insert(protein_key);
        }
    }

    counts.peptides = peptide_set.len();
    counts.proteins = protein_set.len();
    counts
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DfEntrapmentCounts {
    pub psms: usize,
    pub peptides: usize,
    pub proteins: usize,
}

pub fn has_entrapment_proteins(db: &IndexedDatabase) -> bool {
    db.peptides.iter().any(|pep| {
        let protein_key = pep.proteins(&db.decoy_tag, db.generate_decoys);
        is_entrapment_str(&protein_key)
    })
}

// -----------------------------------------------------------------------------
// 9) Empirical null tail p-values (used for Nokoi null mapping)
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
// 10) Debug / diagnostics helpers
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
// 11) Model-fit diagnostics helpers
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
// 12) Storey π0 estimation + q-value computation with fixed π0
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
// 13) P-value -> PEP proxy via local-FDR using a Grenander estimator.
//
// The p-native experts produce p-values under an approximately uniform null.
// For target-search p-values, the mixture density is expected to be monotone
// decreasing on [0, 1]. The Grenander estimator estimates this decreasing
// density by taking the slopes of the least-concave-majorant of the empirical
// CDF. We implement the equivalent PAVA form on adjacent probability intervals.
//
// The returned value is a PEP-like local-FDR estimate:
//     pep(p) = pi0 / f_hat(p)
// where f_hat is the monotone decreasing mixture density.
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct GrenanderBlock {
    p_start: f64,
    p_end: f64,
    count: usize,
}

impl GrenanderBlock {
    #[inline]
    fn density_raw(&self) -> f64 {
        let width = self.p_end - self.p_start;
        if width <= 0.0 {
            f64::INFINITY
        } else {
            (self.count as f64) / width
        }
    }
}

fn grenander_pep_from_p(p_values: &[f64], pi0: f64) -> Vec<f64> {
    const EPS: f64 = 1e-300;

    if p_values.is_empty() {
        return Vec::new();
    }

    let mut pv: Vec<(f64, usize)> = p_values
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| {
            if p.is_finite() {
                Some((p.clamp(EPS, 1.0), i))
            } else {
                None
            }
        })
        .collect();

    if pv.is_empty() {
        return vec![1.0; p_values.len()];
    }

    pv.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    // Pool identical p-values first. This avoids zero-width blocks and gives
    // ties the correct multiplicity.
    let mut pooled: Vec<(f64, usize)> = Vec::new();
    for &(p, _) in &pv {
        if let Some(last) = pooled.last_mut() {
            if last.0 == p {
                last.1 += 1;
                continue;
            }
        }
        pooled.push((p, 1));
    }

    let mut stack: Vec<GrenanderBlock> = Vec::with_capacity(pooled.len());
    let mut p_prev = 0.0f64;

    for &(p_end, count) in &pooled {
        if p_end <= p_prev {
            continue;
        }

        let mut block = GrenanderBlock {
            p_start: p_prev,
            p_end,
            count,
        };

        // Enforce monotone decreasing density as p increases.
        // A violation occurs when the previous block has lower density than
        // the new block.
        while let Some(prev) = stack.last() {
            if prev.density_raw() <= block.density_raw() {
                let prev = stack.pop().unwrap();
                block = GrenanderBlock {
                    p_start: prev.p_start,
                    p_end: block.p_end,
                    count: prev.count + block.count,
                };
            } else {
                break;
            }
        }

        stack.push(block);
        p_prev = p_end;
    }

    if stack.is_empty() {
        return vec![1.0; p_values.len()];
    }

    let n = pv.len() as f64;
    let pi0 = pi0.clamp(0.0, 1.0);

    let mut sorted_peps: Vec<(f64, usize, f64)> = Vec::with_capacity(pv.len());
    let mut block_idx = 0usize;

    for &(p, orig_idx) in &pv {
        while block_idx + 1 < stack.len() && p > stack[block_idx].p_end {
            block_idx += 1;
        }

        let f_hat = stack[block_idx].density_raw() / n;
        let pep = if f_hat.is_finite() && f_hat > 0.0 {
            (pi0 / f_hat).clamp(EPS, 1.0)
        } else if f_hat.is_infinite() {
            EPS
        } else {
            1.0
        };

        sorted_peps.push((p, orig_idx, pep));
    }

    // Numerical safety: worse p-values must not receive better PEPs.
    sorted_peps.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut running_max = EPS;
    for (_, _, pep) in sorted_peps.iter_mut() {
        running_max = running_max.max(*pep);
        *pep = running_max.clamp(EPS, 1.0);
    }

    let mut out = vec![1.0; p_values.len()];
    for (_, orig_idx, pep) in sorted_peps {
        out[orig_idx] = pep;
    }

    out
}

// -----------------------------------------------------------------------------
// 14) Q-values from PEP cumulative mean
// -----------------------------------------------------------------------------
fn q_from_pep_cummean(mut rows: Vec<(f64, usize, f64)>) -> Vec<(usize, f64)> {
    if rows.is_empty() {
        return Vec::new();
    }

    // sort by quality descending (best first)
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));

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
}

// -----------------------------------------------------------------------------
// 15) Work set helper (rank-1 index list)
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
// 16) Debug helper: summarize rank-1 composition (label/entrap/contam)
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
// 17) LowerOrder stratification helper
// -----------------------------------------------------------------------------
//
// LowerOrder can fit either a single global model or separate charge-stratified
// models. This helper maps the observed precursor charge to the bucket ID used
// by the LO fitter and evaluator.
//
// In charge-stratified mode, each charge state receives its own bucket.
// In global mode, all charges are collapsed into bucket 0.
//
// The same mapping must be used during both LO fitting and LO scoring; otherwise
// rank-1 PSMs may be evaluated against a different null model than the one used
// to fit their lower-rank evidence.
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
    fit_data: Vec<(u32, f64, u8, usize, String)>,
    null_indices: Vec<usize>,
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
}

#[derive(Clone)]
struct Engines {
    mom_params: Option<(f64, f64)>,
    mle_params: Option<(f64, f64)>,

    lo_model: Option<LowerOrderModel>,

    msfdr_seeded: Option<MsfdrSeededModel>,
    msfdr_1smix: Option<Msfdr1SmixModel>,
    msfdr_2smix: Option<Msfdr2SmixModel>,

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

    Some(RankNullPool {
        fit_data,
        null_indices,
    })
}

#[derive(Clone, Copy, Debug)]
struct RunGates {
    run_mom: bool,
    run_mle: bool,
    run_lo: bool,
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
    let em_tol = settings.mix_em_tol;
    let pi_clamp = (settings.msfdr_pi_clamp_min, settings.msfdr_pi_clamp_max);
    let top_frac_init = settings.msfdr_seeded_top_frac_init;

    let xs: Vec<f64> = pool_scores
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if xs.len() < 20 {
        return None;
    }

    let mean = xs.iter().sum::<f64>() / (xs.len() as f64);
    let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (xs.len() as f64);
    let beta = ((6.0 * var).sqrt() / std::f64::consts::PI).max(1e-9);
    let mu = mean - 0.5772156649015329_f64 * beta;

    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    MsfdrSeededModel::fit_rank1_seeded(
        rank1_scores,
        mu,
        beta,
        iters,
        em_tol,
        pi_clamp,
        top_frac_init,
    )
}

#[inline]
fn fit_msfdr_1smix(
    rank1_scores: &[f64],
    pool_scores: &[f64],
    settings: &FdrSettings,
) -> Option<Msfdr1SmixModel> {
    let iters = settings.mix_em_max_iter;
    let em_tol = settings.mix_em_tol;
    let pi_clamp = (settings.msfdr1_pi_clamp_min, settings.msfdr1_pi_clamp_max);
    let top_frac_init = settings.msfdr1_top_frac_init;
    let mu_drift_abs = settings.msfdr1_mu_drift_abs;
    let beta_drift_mult = settings.msfdr1_beta_drift_mult;

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

    let (mu, beta) = seed?;

    Msfdr1SmixModel::fit_rank1_with_null_seed(
        rank1_scores,
        iters,
        em_tol,
        pi_clamp,
        mu,
        beta,
        top_frac_init,
        mu_drift_abs,
        beta_drift_mult,
    )
}

#[inline]
fn fit_msfdr_2smix(
    rank1_scores: &[f64],
    pool_scores: &[f64],
    settings: &FdrSettings,
) -> Option<Msfdr2SmixModel> {
    let iters = settings.mix_em_max_iter;
    let em_tol = settings.mix_em_tol;
    let pi_clamp = (settings.msfdr2_pi_clamp_min, settings.msfdr2_pi_clamp_max);
    let top_frac_init = settings.msfdr2_top_frac_init;
    let mix_anchor_incorrect = settings.mix_anchor_incorrect;
    let beta_drift_mult = settings.msfdr2_beta_drift_mult;
    let mu_drift_abs = settings.msfdr2_mu_drift_abs;

    Msfdr2SmixModel::fit_rank1_with_pool(
        rank1_scores,
        pool_scores,
        iters,
        em_tol,
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
    let min_null_size = settings.min_null_size;

    let window_ok = |method: &str, window_min: u32, window_max: u32, count: usize| -> bool {
        if count < min_null_size {
            log::warn!("DF fail-closed: {method}: null window [{window_min}..={window_max}] too small (n={count} < min_null_size={min_null_size}). Skipping.");
            false
        } else {
            true
        }
    };

    // 1) Moments
    let mom_params = if gates.run_mom {
        let scores = pool.scores_in_window(
            settings.moments_min_null_rank,
            settings.moments_max_null_rank,
        );
        if window_ok(
            "Moments",
            settings.moments_min_null_rank,
            settings.moments_max_null_rank,
            scores.len(),
        ) {
            let (mu, beta) = fit_gumbel_moments(&scores);
            if mu.is_finite() && beta.is_finite() && beta > 0.0 {
                Some((mu, beta))
            } else {
                log::warn!("DF fail-closed: Moments produced invalid params.");
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // 2) LO
    let mut lo_model = None;

    if gates.run_lo {
        let mut rank1_scores_by_charge: Vec<(f64, u8)> =
            Vec::with_capacity(work.rank1_indices.len());
        for &i in &work.rank1_indices {
            let f = &features[i];
            if let Some(x_lo) = lo_tev(f) {
                let bid = lo_bucket_id(settings, f.core.charge);
                rank1_scores_by_charge.push((x_lo, bid));
            }
        }

        let lo_raw = pool.fit_data_in_window(
            settings.lower_order_min_null_rank,
            settings.lower_order_max_null_rank,
        );

        if window_ok(
            "LowerOrder",
            settings.lower_order_min_null_rank,
            settings.lower_order_max_null_rank,
            lo_raw.len(),
        ) {
            // Build a direct lookup once, instead of scanning `features` for every pooled row.
            let mut lo_tev_by_key: FnvHashMap<(u32, u8, usize, String), f64> =
                FnvHashMap::default();

            for f in features.iter() {
                if let Some(x_lo) = lo_tev(f) {
                    lo_tev_by_key.insert(
                        (
                            f.core.rank,
                            f.core.charge,
                            f.core.file_id,
                            f.core.spec_id.clone(),
                        ),
                        x_lo,
                    );
                }
            }

            let lo_fit_data: Vec<(u32, f64, u8)> = lo_raw
                .into_iter()
                .filter_map(|(k, _x_raw, charge, file_id, spec_id)| {
                    let x_lo = lo_tev_by_key.get(&(k, charge, file_id, spec_id.clone()))?;
                    Some((k, *x_lo, lo_bucket_id(settings, charge)))
                })
                .collect();

            // LowerOrder rank-window selection is controlled only by
            // lower_order_min_null_rank..=lower_order_max_null_rank.
            //
            // lo_min_count_per_rank is a fixed support threshold for each selected
            // lower-order rank within the active LO bucket. It is not multiplied by
            // the number of selected ranks.
            let lo_min_count_per_rank = settings.lo_min_count_per_rank;

            lo_model = fit_decoy_free_model(
                &lo_fit_data,
                &rank1_scores_by_charge,
                settings.lower_order_min_null_rank,
                settings.lower_order_max_null_rank,
                lo_min_count_per_rank,
                settings.lo_mode.clone(),
                settings.lo_lom_estimator.clone(),
                settings.lo_mean_beta_mode.clone(),
            );
        }
    }

    // 3) MLE
    let mle_params = if gates.run_mle {
        let scores = pool.scores_in_window(settings.mle_min_null_rank, settings.mle_max_null_rank);
        if window_ok(
            "MLE",
            settings.mle_min_null_rank,
            settings.mle_max_null_rank,
            scores.len(),
        ) {
            match fit_gumbel_mle(&scores) {
                Some((mu, beta)) if mu.is_finite() && beta.is_finite() && beta > 0.0 => {
                    Some((mu, beta))
                }
                _ => {
                    log::warn!("DF fail-closed: MLE fit invalid.");
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // 4) MSFDR Variants
    let rank1_scores: Vec<f64> = work
        .rank1_indices
        .iter()
        .filter_map(|&i| tev(&features[i]))
        .collect();

    let msfdr_seeded = if gates.run_msfdr_seeded {
        let seed_pool =
            pool.scores_in_window(settings.msfdr_min_null_rank, settings.msfdr_max_null_rank);
        if window_ok(
            "MSFDR seeded",
            settings.msfdr_min_null_rank,
            settings.msfdr_max_null_rank,
            seed_pool.len(),
        ) {
            let m = fit_msfdr_seeded(&rank1_scores, &seed_pool, settings);
            if let Some(ref model) = m {
                log_fit_ok("MSFDR seeded", model);
            } else {
                log_fit_failed_closed("MSFDR seeded");
            }
            m
        } else {
            None
        }
    } else {
        None
    };

    let msfdr_1smix = if gates.run_msfdr_1smix {
        let pool_1smix = pool.scores_in_window(
            settings.msfdr1_smix_min_null_rank,
            settings.msfdr1_smix_max_null_rank,
        );
        let m = fit_msfdr_1smix(&rank1_scores, &pool_1smix, settings);
        if let Some(ref model) = m {
            log_fit_ok("MSFDR 1smix", model);
        } else {
            log_fit_failed_closed("MSFDR 1smix");
        }
        m
    } else {
        None
    };

    let msfdr_2smix = if gates.run_msfdr_2smix {
        let pool_2smix = pool.scores_in_window(
            settings.msfdr2_smix_min_null_rank,
            settings.msfdr2_smix_max_null_rank,
        );
        if window_ok(
            "MSFDR 2smix",
            settings.msfdr2_smix_min_null_rank,
            settings.msfdr2_smix_max_null_rank,
            pool_2smix.len(),
        ) {
            let m = fit_msfdr_2smix(&rank1_scores, &pool_2smix, settings);
            if let Some(ref model) = m {
                log_fit_ok("MSFDR 2smix", model);
            } else {
                log_fit_failed_closed("MSFDR 2smix");
            }
            m
        } else {
            None
        }
    } else {
        None
    };

    // 5) Nokoi
    let mut nokoi_p_values = None;

    if gates.run_nokoi {
        log::info!("Running Nokoi Rescoring ...");
        let nokoi_data =
            pool.fit_data_in_window(settings.nokoi_min_null_rank, settings.nokoi_max_null_rank);

        if window_ok(
            "Nokoi",
            settings.nokoi_min_null_rank,
            settings.nokoi_max_null_rank,
            nokoi_data.len(),
        ) {
            let mut rank1_hs: Vec<f64> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| tev(&features[i]))
                .collect();
            let threshold = if rank1_hs.len() >= 10 {
                rank1_hs.sort_by(|a, b| b.total_cmp(a));
                let top_k =
                    ((rank1_hs.len() as f64) * settings.purification_factor).round() as usize;
                rank1_hs[top_k.max(5).min(rank1_hs.len()) - 1]
            } else {
                f64::INFINITY
            };

            let is_positive = |f: &DfFeature| -> bool {
                if f.core.rank != 1 {
                    return false;
                }
                tev(f).map(|v| v >= threshold).unwrap_or(false)
            };

            let config = nokoi::NokoiConfig {
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

            if let Some((probs, mut null_scores)) = nokoi::rescore_df_crossfit(
                features,
                &config,
                settings.nokoi_min_null_rank,
                settings.nokoi_max_null_rank,
                settings.nokoi_k_folds,
                is_positive,
                &pool.null_indices,
            ) {
                let prob_arc = Arc::new(probs);
                null_scores.retain(|x| x.is_finite());
                null_scores.sort_by(|a, b| a.total_cmp(b));
                if null_scores.len() < 10 {
                    log::warn!("Nokoi: pool too small for calibration.");
                } else {
                    let mut p_all = vec![1.0; features.len()];
                    for (i, &pt) in prob_arc.iter().enumerate() {
                        p_all[i] = empirical_p_from_null_ge(&null_scores, pt)
                            .clamp(0.0, 1.0)
                            .max(1e-300);
                    }
                    nokoi_p_values = Some(Arc::new(p_all));
                }
            } else {
                log::warn!("Nokoi disabled: crossfit failed.");
            }
        }
    }

    Some(Engines {
        mom_params,
        mle_params,
        lo_model,
        msfdr_seeded,
        msfdr_1smix,
        msfdr_2smix,
        nokoi_p_values,
    })
}

// --- MAIN FUNCTION ---
// =============================================================================
// LAYER 1: Base discovery helpers
// =============================================================================

#[derive(Clone)]
struct BaseDiscoveryResult {
    // Stage summaries
    pub workset: WorkSet,
    pub engines: Option<Engines>,

    // Pre-writeback rank-1 outputs and expert mappings
    pub rank1_out: Vec<Rank1Computed>,

    // Aligned rank-1 vectors for calibrated PEP-like streams.
    // Moments/MLE/LO consume genuine p-value streams.
    // MSFDR consumes fitted-null tail p-values, not raw mixture posterior PEPs.
    // Nokoi consumes empirical null-survival p-values, not raw 1 - P(target).
    pub pep_mom_vec: Vec<f64>,
    pub pep_mle_vec: Vec<f64>,
    pub pep_lo_vec: Vec<f64>,
    pub pep_msfdr_vec: Vec<f64>,
    pub pep_1smix_vec: Vec<f64>,
    pub pep_2smix_vec: Vec<f64>,
    pub pep_nokoi_vec: Vec<f64>,
}

#[derive(Clone, Debug)]
struct PhysicalRescueResult {
    pub fail_closed: bool,

    pub anchor_count_total: usize,
    pub anchor_count_after_filters: usize,

    pub rt_reliability: f64,
    pub ims_reliability: f64,
    pub joint_reliability: f64,

    pub rt_sigma_global: Option<f64>,
    pub ims_sigma_global: Option<f64>,

    pub dropped_runs: Vec<usize>,
    pub dropped_charge_bins: Vec<(i32, usize)>,
}

#[derive(Clone, Debug, Default)]
struct ReproducibilityResult {
    pub enabled: bool,
    pub fail_closed: bool,

    pub n_rescue_eligible_proteins: usize,
    pub n_rescue_eligible_peptides: usize,
    pub n_anchor_peptides: usize,

    pub n_rescued_psms: usize,
    pub n_strong_unchanged_psms: usize,
    pub n_too_weak_unrescued_psms: usize,

    pub agreement_support_mean: f64,
    pub max_shift_applied: f64,
}

fn build_base_workset(features: &[DfFeature]) -> WorkSet {
    WorkSet::build(features)
}

fn build_base_null_pool(
    features: &[DfFeature],
    work: &WorkSet,
    settings: &FdrSettings,
) -> Option<RankNullPool> {
    build_rank_null_pool(features, work, settings)
}

fn fit_base_experts(
    features: &[DfFeature],
    work: &WorkSet,
    pool: &RankNullPool,
    settings: &FdrSettings,
    gates: RunGates,
) -> Option<Engines> {
    fit_engines(features, work, pool, settings, gates)
}

fn score_base_rank1(
    features: &[DfFeature],
    workset: WorkSet,
    engines_opt: Option<Engines>,
    settings: &FdrSettings,
    gates: RunGates,
    db: &IndexedDatabase,
) -> BaseDiscoveryResult {
    let use_ensemble = matches!(settings.model_fit, ModelFit::Ensemble);

    let engines = engines_opt
        .as_ref()
        .expect("engines must be present to score");

    let mom_params = engines.mom_params;
    let mle_params = engines.mle_params;
    let lo_model = engines.lo_model.clone();
    let msfdr_seeded = engines.msfdr_seeded.clone();
    let msfdr_1smix = engines.msfdr_1smix.clone();
    let msfdr_2smix = engines.msfdr_2smix.clone();
    let nokoi_p_values = engines.nokoi_p_values.clone();

    let use_mom_expert = gates.run_mom && mom_params.is_some();
    let use_mle_expert = gates.run_mle && mle_params.is_some();
    let use_lo_expert = gates.run_lo && lo_model.is_some();
    let use_seeded_expert = gates.run_msfdr_seeded && msfdr_seeded.is_some();
    let use_1smix_expert = gates.run_msfdr_1smix && msfdr_1smix.is_some();
    let use_2smix_expert = gates.run_msfdr_2smix && msfdr_2smix.is_some();
    let use_nokoi_expert = gates.run_nokoi && nokoi_p_values.is_some();

    let std_gumbel = Gumbel::new(0.0, 1.0).expect("standard gumbel");

    let rank1_out: Vec<Rank1Computed> = workset
        .rank1_indices
        .par_iter()
        .filter_map(|&idx| {
            let psm = &features[idx];
            let x = tev(psm)?;

            let p_mom = if let Some((mu, beta)) = mom_params {
                let tev = tev_norm_from_hyperscore(x, mu, beta);
                std_gumbel.sf(tev).clamp(0.0, 1.0).max(1e-300)
            } else {
                1.0
            };

            let p_mle = if let Some((mu, beta)) = mle_params {
                let tev = tev_norm_from_hyperscore(x, mu, beta);
                std_gumbel.sf(tev).clamp(0.0, 1.0).max(1e-300)
            } else {
                1.0
            };

            let p_lo = if let Some(ref m) = lo_model {
                let bid = lo_bucket_id(settings, psm.core.charge);
                let x_eval = lo_tev(psm)?;

                let p = m.p_value(x_eval, bid);
                if p.is_finite() {
                    p.clamp(0.0, 1.0).max(1e-300)
                } else {
                    return None;
                }
            } else {
                1.0
            };

            let p_msfdr = if use_seeded_expert {
                let m = msfdr_seeded.as_ref().unwrap();
                Some(m.p_value(x).clamp(0.0, 1.0).max(1e-300))
            } else {
                None
            };

            let p_1smix = if use_1smix_expert {
                let m = msfdr_1smix.as_ref().unwrap();
                Some(m.p_value(x).clamp(0.0, 1.0).max(1e-300))
            } else {
                None
            };

            let p_2smix = if use_2smix_expert {
                let m = msfdr_2smix.as_ref().unwrap();
                Some(m.p_value(x).clamp(0.0, 1.0).max(1e-300))
            } else {
                None
            };

            let p_nokoi = if use_nokoi_expert {
                let p_vec = nokoi_p_values.as_ref().unwrap();
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

            let p_final = if use_ensemble {
                1.0
            } else {
                match settings.model_fit {
                    ModelFit::Moments => {
                        if use_mom_expert {
                            p_mom
                        } else {
                            1.0
                        }
                    }
                    ModelFit::Mle => {
                        if use_mle_expert {
                            p_mle
                        } else {
                            1.0
                        }
                    }
                    ModelFit::LowerOrder => {
                        if use_lo_expert {
                            p_lo
                        } else {
                            1.0
                        }
                    }
                    ModelFit::Msfdr => p_msfdr.unwrap_or(1.0),
                    ModelFit::Msfdr1Smix => p_1smix.unwrap_or(1.0),
                    ModelFit::Msfdr2Smix => p_2smix.unwrap_or(1.0),
                    ModelFit::Nokoi => p_nokoi.unwrap_or(1.0),
                    ModelFit::Ensemble => 1.0,
                }
            }
            .clamp(0.0, 1.0)
            .max(1e-300);

            Some(Rank1Computed {
                idx,
                p_mom,
                p_mle,
                p_lo,
                p_msfdr,
                p_1smix,
                p_2smix,
                p_nokoi,
                p_final,
            })
        })
        .collect();

    let p_mom_all: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.p_mom.clamp(0.0, 1.0).max(1e-300))
        .collect();

    let p_mle_all: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.p_mle.clamp(0.0, 1.0).max(1e-300))
        .collect();

    let p_lo_all: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.p_lo.clamp(0.0, 1.0).max(1e-300))
        .collect();

    let p_msfdr_all: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.p_msfdr.unwrap_or(1.0).clamp(0.0, 1.0).max(1e-300))
        .collect();

    let p_1smix_all: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.p_1smix.unwrap_or(1.0).clamp(0.0, 1.0).max(1e-300))
        .collect();

    let p_2smix_all: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.p_2smix.unwrap_or(1.0).clamp(0.0, 1.0).max(1e-300))
        .collect();

    let p_nokoi_all: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.p_nokoi.unwrap_or(1.0).clamp(0.0, 1.0).max(1e-300))
        .collect();

    let mut is_ref: Vec<bool> = Vec::with_capacity(rank1_out.len());
    for r in &rank1_out {
        let f = &features[r.idx];
        if f.core.label != 1 {
            is_ref.push(false);
            continue;
        }
        let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        is_ref.push(!is_contam_str(&prot) && !is_entrapment_str(&prot));
    }

    let mut p_mom_ref: Vec<f64> = Vec::new();
    let mut p_mle_ref: Vec<f64> = Vec::new();
    let mut p_lo_ref: Vec<f64> = Vec::new();
    let mut p_msfdr_ref: Vec<f64> = Vec::new();
    let mut p_1smix_ref: Vec<f64> = Vec::new();
    let mut p_2smix_ref: Vec<f64> = Vec::new();
    let mut p_nokoi_ref: Vec<f64> = Vec::new();

    for (k, &ref_ok) in is_ref.iter().enumerate() {
        if !ref_ok {
            continue;
        }

        p_mom_ref.push(p_mom_all[k]);
        p_mle_ref.push(p_mle_all[k]);
        p_lo_ref.push(p_lo_all[k]);

        let r = &rank1_out[k];

        if r.p_msfdr.is_some() {
            p_msfdr_ref.push(p_msfdr_all[k]);
        }
        if r.p_1smix.is_some() {
            p_1smix_ref.push(p_1smix_all[k]);
        }
        if r.p_2smix.is_some() {
            p_2smix_ref.push(p_2smix_all[k]);
        }
        if r.p_nokoi.is_some() {
            p_nokoi_ref.push(p_nokoi_all[k]);
        }
    }

    let pi0_mom = estimate_pi0_from_reference_grid(&p_mom_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let pi0_mle = estimate_pi0_from_reference_grid(&p_mle_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let pi0_lo = estimate_pi0_from_reference_grid(&p_lo_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    let pi0_msfdr = estimate_pi0_from_reference_grid(&p_msfdr_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let pi0_1smix = estimate_pi0_from_reference_grid(&p_1smix_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let pi0_2smix = estimate_pi0_from_reference_grid(&p_2smix_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let pi0_nokoi = estimate_pi0_from_reference_grid(&p_nokoi_ref, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    let pep_mom_vec = grenander_pep_from_p(&p_mom_all, pi0_mom);
    let pep_mle_vec = grenander_pep_from_p(&p_mle_all, pi0_mle);
    let pep_lo_vec = grenander_pep_from_p(&p_lo_all, pi0_lo);

    // MSFDR: calibrate fitted-null tail p-values, not raw mixture posterior PEPs.
    let pep_msfdr_vec = grenander_pep_from_p(&p_msfdr_all, pi0_msfdr);
    let pep_1smix_vec = grenander_pep_from_p(&p_1smix_all, pi0_1smix);
    let pep_2smix_vec = grenander_pep_from_p(&p_2smix_all, pi0_2smix);

    // Nokoi: calibrate empirical null-survival p-values from classifier scores,
    // not raw 1 - P(target).
    let pep_nokoi_vec = grenander_pep_from_p(&p_nokoi_all, pi0_nokoi);

    BaseDiscoveryResult {
        workset,
        engines: engines_opt,
        rank1_out,
        pep_mom_vec,
        pep_mle_vec,
        pep_lo_vec,
        pep_msfdr_vec,
        pep_1smix_vec,
        pep_2smix_vec,
        pep_nokoi_vec,
    }
}

fn write_base_method_outputs(
    features: &mut [DfFeature],
    base_res: &BaseDiscoveryResult,
    settings: &FdrSettings,
    gates: RunGates,
) {
    let use_ensemble = matches!(settings.model_fit, ModelFit::Ensemble);

    let ensemble_pep_combiner = settings.ensemble_pep_combiner.clone();
    let ensemble_pep_trim_frac = settings.ensemble_pep_trim_frac;
    let ensemble_pep_quantile = settings.ensemble_pep_quantile;
    let ensemble_pep_top_k = settings.ensemble_pep_top_k;
    let ensemble_pep_logit_eps = settings.ensemble_pep_logit_eps;

    let ensemble_weight_moments = settings.ensemble_weight_moments;
    let ensemble_weight_mle = settings.ensemble_weight_mle;
    let ensemble_weight_lower_order = settings.ensemble_weight_lower_order;
    let ensemble_weight_msfdr_seeded = settings.ensemble_weight_msfdr_seeded;
    let ensemble_weight_msfdr_1smix = settings.ensemble_weight_msfdr_1smix;
    let ensemble_weight_msfdr_2smix = settings.ensemble_weight_msfdr_2smix;
    let ensemble_weight_nokoi = settings.ensemble_weight_nokoi;

    let engines = base_res
        .engines
        .as_ref()
        .expect("engines must be present to write outputs");

    let use_mom_expert = gates.run_mom && engines.mom_params.is_some();
    let use_mle_expert = gates.run_mle && engines.mle_params.is_some();
    let use_lo_expert = gates.run_lo && engines.lo_model.is_some();
    let use_seeded_expert = gates.run_msfdr_seeded && engines.msfdr_seeded.is_some();
    let use_1smix_expert = gates.run_msfdr_1smix && engines.msfdr_1smix.is_some();
    let use_2smix_expert = gates.run_msfdr_2smix && engines.msfdr_2smix.is_some();
    let use_nokoi_expert = gates.run_nokoi && engines.nokoi_p_values.is_some();

    for (j, r) in base_res.rank1_out.iter().enumerate() {
        let psm = &mut features[r.idx];

        psm.p_mom = if use_mom_expert {
            Some(r.p_mom as f32)
        } else {
            None
        };
        psm.p_mle = if use_mle_expert {
            Some(r.p_mle as f32)
        } else {
            None
        };
        psm.p_lo = if use_lo_expert {
            Some(r.p_lo as f32)
        } else {
            None
        };
        psm.p_msfdr = if use_seeded_expert {
            r.p_msfdr.map(|v| v as f32)
        } else {
            None
        };
        psm.p_1smix = if use_1smix_expert {
            r.p_1smix.map(|v| v as f32)
        } else {
            None
        };
        psm.p_2smix = if use_2smix_expert {
            r.p_2smix.map(|v| v as f32)
        } else {
            None
        };
        psm.p_nokoi = if use_nokoi_expert {
            r.p_nokoi.map(|v| v as f32)
        } else {
            None
        };

        psm.pep_mom = if use_mom_expert {
            Some(base_res.pep_mom_vec[j] as f32)
        } else {
            None
        };
        psm.pep_mle = if use_mle_expert {
            Some(base_res.pep_mle_vec[j] as f32)
        } else {
            None
        };
        psm.pep_lo = if use_lo_expert {
            Some(base_res.pep_lo_vec[j] as f32)
        } else {
            None
        };
        psm.pep_msfdr = if use_seeded_expert {
            Some(base_res.pep_msfdr_vec[j] as f32)
        } else {
            None
        };
        psm.pep_1smix = if use_1smix_expert {
            Some(base_res.pep_1smix_vec[j] as f32)
        } else {
            None
        };
        psm.pep_2smix = if use_2smix_expert {
            Some(base_res.pep_2smix_vec[j] as f32)
        } else {
            None
        };
        psm.pep_nokoi = if use_nokoi_expert {
            Some(base_res.pep_nokoi_vec[j] as f32)
        } else {
            None
        };

        if !use_ensemble {
            set_df_p_value(psm, r.p_final as f32);
        } else {
            psm.decoy_free_p_value = None;
        }

        let pep_consensus: f64 = if use_ensemble {
            let mut pep_experts: Vec<f64> = Vec::new();
            let mut pep_weights: Vec<f64> = Vec::new();

            if use_mom_expert {
                pep_experts.push(base_res.pep_mom_vec[j]);
                pep_weights.push(ensemble_weight_moments);
            }
            if use_mle_expert {
                pep_experts.push(base_res.pep_mle_vec[j]);
                pep_weights.push(ensemble_weight_mle);
            }
            if use_lo_expert {
                pep_experts.push(base_res.pep_lo_vec[j]);
                pep_weights.push(ensemble_weight_lower_order);
            }
            if use_seeded_expert {
                pep_experts.push(base_res.pep_msfdr_vec[j]);
                pep_weights.push(ensemble_weight_msfdr_seeded);
            }
            if use_1smix_expert {
                pep_experts.push(base_res.pep_1smix_vec[j]);
                pep_weights.push(ensemble_weight_msfdr_1smix);
            }
            if use_2smix_expert {
                pep_experts.push(base_res.pep_2smix_vec[j]);
                pep_weights.push(ensemble_weight_msfdr_2smix);
            }
            if use_nokoi_expert {
                pep_experts.push(base_res.pep_nokoi_vec[j]);
                pep_weights.push(ensemble_weight_nokoi);
            }

            combine_peps(
                &pep_experts,
                &pep_weights,
                ensemble_pep_combiner.clone(),
                ensemble_pep_trim_frac,
                ensemble_pep_quantile,
                ensemble_pep_top_k,
                ensemble_pep_logit_eps,
            )
        } else {
            match settings.model_fit {
                ModelFit::Moments => {
                    if use_mom_expert {
                        base_res.pep_mom_vec[j]
                    } else {
                        1.0
                    }
                }
                ModelFit::Mle => {
                    if use_mle_expert {
                        base_res.pep_mle_vec[j]
                    } else {
                        1.0
                    }
                }
                ModelFit::LowerOrder => {
                    if use_lo_expert {
                        base_res.pep_lo_vec[j]
                    } else {
                        1.0
                    }
                }
                ModelFit::Msfdr => {
                    if use_seeded_expert {
                        base_res.pep_msfdr_vec[j]
                    } else {
                        1.0
                    }
                }
                ModelFit::Msfdr1Smix => {
                    if use_1smix_expert {
                        base_res.pep_1smix_vec[j]
                    } else {
                        1.0
                    }
                }
                ModelFit::Msfdr2Smix => {
                    if use_2smix_expert {
                        base_res.pep_2smix_vec[j]
                    } else {
                        1.0
                    }
                }
                ModelFit::Nokoi => {
                    if use_nokoi_expert {
                        base_res.pep_nokoi_vec[j]
                    } else {
                        1.0
                    }
                }
                _ => 1.0,
            }
        }
        .clamp(0.0, 1.0)
        .max(1e-300);

        let df_score = (-10.0 * pep_consensus.max(1e-15).log10()) as f32;
        psm.decoy_free_pep = Some(pep_consensus as f32);
        psm.decoy_free_score = Some(df_score);
    }
}

fn scrub_non_rank1_df_outputs(features: &mut [DfFeature]) {
    features.par_iter_mut().for_each(|psm| {
        if psm.core.rank != 1 {
            psm.decoy_free_p_value = None;
            psm.decoy_free_pep = None;
            psm.decoy_free_score = None;
            psm.decoy_free_q_value = None;
            psm.decoy_free_peptide_q = None;
            psm.decoy_free_protein_q = None;
            psm.decoy_free_p_value_base = None;
            psm.decoy_free_pep_base = None;
            psm.decoy_free_score_base = None;
            psm.decoy_free_q_base = None;
            psm.decoy_free_p_value_l2 = None;
            psm.decoy_free_pep_l2 = None;
            psm.decoy_free_score_l2 = None;
            psm.decoy_free_q_l2 = None;
            psm.decoy_free_pep_l3 = None;
            psm.decoy_free_score_l3 = None;
            psm.decoy_free_q_l3 = None;
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
        }
    });
}

fn finalize_base_q_values(
    features: &mut [DfFeature],
    work: &WorkSet,
    settings: &FdrSettings,
    db: &IndexedDatabase,
) {
    let use_ensemble = matches!(settings.model_fit, ModelFit::Ensemble);
    let min_storey_n = settings.min_storey_n;

    let rank1_p: Vec<f64> = work
        .rank1_indices
        .iter()
        .filter_map(|&i| {
            features[i]
                .decoy_free_p_value
                .map(|p| (p as f64).clamp(0.0, 1.0).max(1e-300))
        })
        .collect();

    log_rank1_composition(features, work, db);
    if !use_ensemble {
        summarize_pvec("rank1_p (chosen stream, pre-q)", &rank1_p);
    } else {
        let rank1_pep: Vec<f64> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                features[i]
                    .decoy_free_pep
                    .map(|p| (p as f64).clamp(0.0, 1.0).max(1e-300))
            })
            .collect();
        summarize_pvec("rank1_pep (chosen stream, pre-q)", &rank1_pep);
    }

    {
        let rank1_p_1smix: Vec<f64> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                features[i]
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
                features[i]
                    .p_2smix
                    .map(|v| (v as f64).clamp(0.0, 1.0).max(1e-300))
            })
            .collect();
        if !rank1_p_2smix.is_empty() {
            summarize_pvec("rank1_p_2smix (pre-q)", &rank1_p_2smix);
        }
    }

    let rank1_p_ref: Vec<f64> = work
        .rank1_indices
        .iter()
        .filter_map(|&i| {
            let f = &features[i];
            if f.core.label != 1 {
                return None;
            }
            let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            if is_contam_str(&prot) || is_entrapment_str(&prot) {
                return None;
            }
            let p = f.decoy_free_p_value?;
            let p = (p as f64).clamp(0.0, 1.0).max(1e-300);
            p.is_finite().then_some(p)
        })
        .collect();

    if !use_ensemble {
        summarize_pvec("rank1_p_ref (targets, non-ENT, non-CONT)", &rank1_p_ref);
    }

    summarize_q(
        "PREQ rank1 (chosen p-stream/pep)",
        work.rank1_indices.iter().filter_map(|&i| {
            if use_ensemble {
                features[i].decoy_free_pep
            } else {
                Some(df_p_value(&features[i]))
            }
        }),
    );

    if matches!(
        settings.model_fit,
        ModelFit::Msfdr
            | ModelFit::Msfdr1Smix
            | ModelFit::Msfdr2Smix
            | ModelFit::Nokoi
            | ModelFit::Ensemble
    ) {
        let rows: Vec<(f64, usize, f64)> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                let f = &features[i];
                let pep = f.decoy_free_pep? as f64;
                if !pep.is_finite() {
                    return None;
                }
                let score_key = df_rank_score_base_pep(f);
                Some((score_key, i, pep.clamp(0.0, 1.0).max(1e-300)))
            })
            .collect();

        for (feat_idx, q) in q_from_pep_cummean(rows) {
            set_df_q_value(&mut features[feat_idx], q as f32);
        }
    } else {
        let q_values = match settings.type_ {
            FdrType::Bh => stats::bh_q_value(&rank1_p),
            FdrType::Storey => {
                let pi0_opt = estimate_pi0_from_reference_grid(&rank1_p_ref, settings);
                match pi0_opt {
                    Some(pi0) => storey_q_value_with_pi0(&rank1_p, pi0, settings),
                    None => {
                        log::warn!("DF DEBUG Storey(grid): degenerate pi0 on reference set, falling back to BH.");
                        stats::bh_q_value(&rank1_p)
                    }
                }
            }
        };

        for (&idx, q) in work.rank1_indices.iter().zip(q_values) {
            set_df_q_value(&mut features[idx], q as f32);
        }
    }

    #[cfg(debug_assertions)]
    {
        for &i in &work.rank1_indices {
            if let Some(q) = features[i].decoy_free_q_value {
                debug_assert!(
                    q.is_finite() && q > 0.0 && q <= 1.0,
                    "DF ASSERT decoy_free_q_value invalid"
                );
            }
        }
    }

    summarize_q(
        "POSTQ rank1(label==1)",
        features
            .iter()
            .filter(|f| f.core.rank == 1 && f.core.label == 1)
            .map(|f| df_q_value(f)),
    );
    summarize_q(
        "POSTQ rank1(all labels)",
        features
            .iter()
            .filter(|f| f.core.rank == 1)
            .map(|f| df_q_value(f)),
    );
    summarize_q(
        "POSTQ rank1_p_ref (same subset as pi0)",
        features
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

    for &i in &work.rank1_indices {
        features[i].q_1smix = None;
        features[i].q_2smix = None;
    }

    let mut mom_pos: Vec<usize> = Vec::new();
    let mut p_mom_present: Vec<f64> = Vec::new();
    let mut mle_pos: Vec<usize> = Vec::new();
    let mut p_mle_present: Vec<f64> = Vec::new();
    let mut lo_pos: Vec<usize> = Vec::new();
    let mut p_lo_present: Vec<f64> = Vec::new();
    let mut msfdr_pos: Vec<usize> = Vec::new();
    let mut p_msfdr_present: Vec<f64> = Vec::new();
    let mut nokoi_pos: Vec<usize> = Vec::new();
    let mut p_nokoi_present: Vec<f64> = Vec::new();

    let mut p_mom_ref: Vec<f64> = Vec::new();
    let mut p_mle_ref: Vec<f64> = Vec::new();
    let mut p_lo_ref: Vec<f64> = Vec::new();
    let mut p_msfdr_ref: Vec<f64> = Vec::new();
    let mut p_nokoi_ref: Vec<f64> = Vec::new();

    for (k, &i) in work.rank1_indices.iter().enumerate() {
        let f = &features[i];
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

    let compute_q_present = |p_present: &Vec<f64>, p_ref: &Vec<f64>| -> Vec<f64> {
        if p_present.is_empty() {
            return Vec::new();
        }
        match settings.type_ {
            FdrType::Bh => stats::bh_q_value(p_present),
            FdrType::Storey => {
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

    let q_mom_present = compute_q_present(&p_mom_present, &p_mom_ref);
    let q_mle_present = compute_q_present(&p_mle_present, &p_mle_ref);
    let q_lo_present = compute_q_present(&p_lo_present, &p_lo_ref);
    let q_msfdr_present = compute_q_present(&p_msfdr_present, &p_msfdr_ref);
    let q_nokoi_present = compute_q_present(&p_nokoi_present, &p_nokoi_ref);

    for (j, &k) in mom_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_mom = Some(q_mom_present[j] as f32);
    }
    for (j, &k) in mle_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_mle = Some(q_mle_present[j] as f32);
    }
    for (j, &k) in lo_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_lo = Some(q_lo_present[j] as f32);
    }
    for (j, &k) in msfdr_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_msfdr = Some(q_msfdr_present[j] as f32);
    }
    for (j, &k) in nokoi_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_nokoi = Some(q_nokoi_present[j] as f32);
    }

    {
        let rows_1smix: Vec<(f64, usize, f64)> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                let f = &features[i];
                let pep = f.pep_1smix? as f64;
                if !pep.is_finite() {
                    return None;
                }
                Some((df_rank_score(f), i, pep.clamp(0.0, 1.0)))
            })
            .collect();
        for (i, q) in q_from_pep_cummean(rows_1smix) {
            if features[i].core.rank == 1 && features[i].pep_1smix.is_some() {
                features[i].q_1smix = Some(q as f32);
            }
        }

        let rows_2smix: Vec<(f64, usize, f64)> = work
            .rank1_indices
            .iter()
            .filter_map(|&i| {
                let f = &features[i];
                let pep = f.pep_2smix? as f64;
                if !pep.is_finite() {
                    return None;
                }
                Some((df_rank_score(f), i, pep.clamp(0.0, 1.0)))
            })
            .collect();
        for (i, q) in q_from_pep_cummean(rows_2smix) {
            if features[i].core.rank == 1 && features[i].pep_2smix.is_some() {
                features[i].q_2smix = Some(q as f32);
            }
        }
    }
}

// =============================================================================
// LAYER 2: Shared scaffolding and dispatcher
// =============================================================================

fn build_physical_anchor_set(
    features: &[DfFeature],
    settings: &FdrSettings,
    _db: &crate::database::IndexedDatabase,
) -> Vec<usize> {
    let max_pep = settings.physical_rescue.anchor_max_pep as f32;
    let max_q = settings.physical_rescue.anchor_max_q;

    features
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            if f.core.rank != 1 {
                return false;
            }

            // Evidence Floor (PEP)
            let pep = f.decoy_free_pep_base.unwrap_or(1.0);
            if !pep.is_finite() || pep > max_pep {
                return false;
            }

            // Evidence Floor (Q-value)
            let q = f.decoy_free_q_base.unwrap_or(1.0);
            if q > max_q as f32 {
                return false;
            }

            true
        })
        .map(|(i, _)| i)
        .collect()
}

fn filter_anchor_candidates_by_run(
    features: &[DfFeature],
    candidate_indices: Vec<usize>,
    settings: &FdrSettings,
) -> (Vec<usize>, Vec<usize>) {
    use std::collections::HashMap;

    let min_per_run = settings.physical_rescue.min_anchor_count_per_run;

    let mut run_counts: HashMap<usize, usize> = HashMap::new();
    for &idx in &candidate_indices {
        *run_counts.entry(features[idx].core.file_id).or_insert(0) += 1;
    }

    let mut dropped_runs: Vec<usize> = run_counts
        .iter()
        .filter_map(|(&file_id, &count)| {
            if count < min_per_run {
                Some(file_id)
            } else {
                None
            }
        })
        .collect();
    dropped_runs.sort_unstable();

    let kept: Vec<usize> = candidate_indices
        .into_iter()
        .filter(|&idx| {
            run_counts
                .get(&features[idx].core.file_id)
                .copied()
                .unwrap_or(0)
                >= min_per_run
        })
        .collect();

    (kept, dropped_runs)
}

fn filter_anchor_candidates_by_charge(
    features: &[DfFeature],
    candidate_indices: Vec<usize>,
    settings: &FdrSettings,
) -> (Vec<usize>, Vec<(i32, usize)>) {
    use std::collections::HashMap;

    let min_per_charge = settings.physical_rescue.min_anchor_count_per_charge;
    if min_per_charge <= 1 {
        return (candidate_indices, Vec::new());
    }

    let mut charge_counts: HashMap<i32, usize> = HashMap::new();
    for &idx in &candidate_indices {
        *charge_counts
            .entry(features[idx].core.charge as i32)
            .or_insert(0) += 1;
    }

    let mut dropped_charge_bins: Vec<(i32, usize)> = charge_counts
        .iter()
        .filter_map(|(&charge, &count)| {
            if count < min_per_charge {
                Some((charge, count))
            } else {
                None
            }
        })
        .collect();
    dropped_charge_bins.sort_unstable_by_key(|(charge, _)| *charge);

    let kept: Vec<usize> = candidate_indices
        .into_iter()
        .filter(|&idx| {
            charge_counts
                .get(&(features[idx].core.charge as i32))
                .copied()
                .unwrap_or(0)
                >= min_per_charge
        })
        .collect();

    (kept, dropped_charge_bins)
}

fn exclude_non_rescue_safe_anchors(
    features: &[DfFeature],
    candidate_indices: Vec<usize>,
    settings: &FdrSettings,
    db: &crate::database::IndexedDatabase,
) -> (Vec<usize>, Vec<usize>) {
    let anchor_mode = &settings.physical_rescue.anchor_mode;

    // Supported modes:
    //   Strict / Default -> require finite aligned/predicted/delta RT and exclude contam+entrapment
    //   Relaxed          -> require finite aligned/predicted RT, ignore delta, still exclude contam+entrapment
    //   EvidenceOnly     -> no extra anchor safety exclusions beyond evidence floor
    let require_aligned_rt = !matches!(anchor_mode, PhysicalAnchorMode::EvidenceOnly);
    let require_predicted_rt = !matches!(anchor_mode, PhysicalAnchorMode::EvidenceOnly);
    let require_delta_rt = matches!(
        anchor_mode,
        PhysicalAnchorMode::Strict | PhysicalAnchorMode::Default
    );
    let exclude_unsafe_proteins = !matches!(anchor_mode, PhysicalAnchorMode::EvidenceOnly);

    let mut kept: Vec<usize> = Vec::new();
    let mut dropped: Vec<usize> = Vec::new();

    for idx in candidate_indices {
        let f = &features[idx];

        let rt_ok = (!require_aligned_rt || f.core.aligned_rt.is_finite())
            && (!require_predicted_rt || f.core.predicted_rt.is_finite())
            && (!require_delta_rt || f.core.delta_rt_model.is_finite());

        let protein_ok = if exclude_unsafe_proteins {
            let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            !is_entrapment_str(&prot) && !is_contam_str(&prot)
        } else {
            true
        };

        if rt_ok && protein_ok {
            kept.push(idx);
        } else {
            dropped.push(idx);
        }
    }

    (kept, dropped)
}

fn summarize_anchor_coverage(
    features: &[DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
    anchors: &[usize],
) {
    use std::collections::{HashMap, HashSet};

    let max_pep = settings.physical_rescue.anchor_max_pep as f32;
    let max_q = settings.physical_rescue.anchor_max_q;
    let min_per_run = settings.physical_rescue.min_anchor_count_per_run;
    let min_per_charge = settings.physical_rescue.min_anchor_count_per_charge;
    let anchor_mode = &settings.physical_rescue.anchor_mode;
    let require_aligned_rt = !matches!(anchor_mode, PhysicalAnchorMode::EvidenceOnly);
    let require_predicted_rt = !matches!(anchor_mode, PhysicalAnchorMode::EvidenceOnly);
    let require_delta_rt = matches!(
        anchor_mode,
        PhysicalAnchorMode::Strict | PhysicalAnchorMode::Default
    );
    let exclude_unsafe_proteins = !matches!(anchor_mode, PhysicalAnchorMode::EvidenceOnly);

    // ---------------------------------------------------------------------
    // 1) Total candidates before filtering
    // ---------------------------------------------------------------------
    let total_candidates_before_filtering = features.iter().filter(|f| f.core.rank == 1).count();

    // ---------------------------------------------------------------------
    // 2) After evidence filtering
    //    Match current builder semantics: rank 1 + strong base evidence
    // ---------------------------------------------------------------------
    let evidence_filtered: Vec<usize> = features
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            if f.core.rank != 1 {
                return false;
            }
            let pep = f.decoy_free_pep_base.unwrap_or(1.0);
            let q = f.decoy_free_q_base.unwrap_or(1.0);

            pep.is_finite() && pep <= max_pep && q <= max_q as f32
        })
        .map(|(i, _)| i)
        .collect();

    let after_evidence_filtering = evidence_filtered.len();

    // ---------------------------------------------------------------------
    // 3) Excluded for missing / invalid physical evidence under anchor_mode
    // ---------------------------------------------------------------------
    let excluded_for_missing_rt = evidence_filtered
        .iter()
        .filter(|&&idx| {
            let f = &features[idx];
            let missing_aligned = require_aligned_rt && !f.core.aligned_rt.is_finite();
            let missing_predicted = require_predicted_rt && !f.core.predicted_rt.is_finite();
            let missing_delta = require_delta_rt && !f.core.delta_rt_model.is_finite();
            missing_aligned || missing_predicted || missing_delta
        })
        .count();

    // ---------------------------------------------------------------------
    // 4) Excluded for contaminant / entrapment under anchor_mode
    // ---------------------------------------------------------------------
    let excluded_for_contam_or_entrapment = if exclude_unsafe_proteins {
        evidence_filtered
            .iter()
            .filter(|&&idx| {
                let f = &features[idx];
                let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
                is_entrapment_str(&prot) || is_contam_str(&prot)
            })
            .count()
    } else {
        0
    };

    // ---------------------------------------------------------------------
    // 5) After mode-aware unsafe-anchor exclusion
    // ---------------------------------------------------------------------
    let safe_after_exclusion: Vec<usize> = evidence_filtered
        .into_iter()
        .filter(|&idx| {
            let f = &features[idx];

            let rt_ok = (!require_aligned_rt || f.core.aligned_rt.is_finite())
                && (!require_predicted_rt || f.core.predicted_rt.is_finite())
                && (!require_delta_rt || f.core.delta_rt_model.is_finite());

            let safe_protein = if exclude_unsafe_proteins {
                let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
                !is_entrapment_str(&prot) && !is_contam_str(&prot)
            } else {
                true
            };

            rt_ok && safe_protein
        })
        .collect();

    // ---------------------------------------------------------------------
    // 6) After run filtering
    // ---------------------------------------------------------------------
    let mut run_counts: HashMap<usize, usize> = HashMap::new();
    for &idx in &safe_after_exclusion {
        *run_counts.entry(features[idx].core.file_id).or_insert(0) += 1;
    }

    let after_run_filtering_vec: Vec<usize> = safe_after_exclusion
        .iter()
        .copied()
        .filter(|&idx| {
            run_counts
                .get(&features[idx].core.file_id)
                .copied()
                .unwrap_or(0)
                >= min_per_run
        })
        .collect();

    let after_run_filtering = after_run_filtering_vec.len();

    // ---------------------------------------------------------------------
    // 7) After charge filtering
    // ---------------------------------------------------------------------
    let after_charge_filtering = if min_per_charge <= 1 {
        after_run_filtering
    } else {
        let mut charge_counts: HashMap<i32, usize> = HashMap::new();
        for &idx in &after_run_filtering_vec {
            *charge_counts
                .entry(features[idx].core.charge as i32)
                .or_insert(0) += 1;
        }

        after_run_filtering_vec
            .iter()
            .filter(|&&idx| {
                charge_counts
                    .get(&(features[idx].core.charge as i32))
                    .copied()
                    .unwrap_or(0)
                    >= min_per_charge
            })
            .count()
    };

    // ---------------------------------------------------------------------
    // 8) Final accepted anchors
    // ---------------------------------------------------------------------
    let final_accepted_anchor_count = anchors.len();

    let mut runs: HashSet<usize> = HashSet::new();
    let mut charges: HashSet<i32> = HashSet::new();
    for &idx in anchors {
        let f = &features[idx];
        runs.insert(f.core.file_id);
        charges.insert(f.core.charge as i32);
    }

    log::debug!(
        "L2 Anchor Diagnostics: total_before={} after_evidence={} after_run={} after_charge={} excluded_missing_rt={} excluded_contam_or_entrapment={} final_accepted={} runs={} charges={}",
        total_candidates_before_filtering,
        after_evidence_filtering,
        after_run_filtering,
        after_charge_filtering,
        excluded_for_missing_rt,
        excluded_for_contam_or_entrapment,
        final_accepted_anchor_count,
        runs.len(),
        charges.len()
    );
}

fn compute_rt_reliability(
    features: &[DfFeature],
    anchors: &[usize],
    settings: &FdrSettings,
) -> RtReliabilitySummary {
    use std::collections::HashMap;

    let mut deltas: Vec<f64> = anchors
        .iter()
        .map(|&i| features[i].core.delta_rt_model as f64)
        .filter(|x| x.is_finite())
        .collect();

    deltas.sort_by(|a, b| a.total_cmp(b));

    let rt_sigma_global = if deltas.is_empty() {
        None
    } else {
        let med = deltas[deltas.len() / 2].abs();
        Some((med * 1.4826).clamp(0.05, 0.25))
    };

    let mut by_file: HashMap<usize, Vec<f64>> = HashMap::new();
    for &i in anchors {
        let d = features[i].core.delta_rt_model as f64;
        if d.is_finite() {
            by_file
                .entry(features[i].core.file_id)
                .or_default()
                .push(d.abs());
        }
    }

    let mut runwise_rt_sigma: Vec<(usize, f64)> = Vec::new();
    for (file_id, vals) in by_file.into_iter() {
        if vals.is_empty() {
            continue;
        }
        let mut vals = vals;
        vals.sort_by(|a, b| a.total_cmp(b));
        let med = vals[vals.len() / 2];
        let sigma = (med * 1.4826).clamp(0.05, 0.25);
        runwise_rt_sigma.push((file_id, sigma));
    }
    runwise_rt_sigma.sort_by_key(|(file_id, _)| *file_id);

    let reliability = match rt_sigma_global {
        Some(sig) => {
            let raw = 1.0 / (1.0 + sig / 0.10);
            raw.clamp(0.0, 1.0)
        }
        None => 0.0,
    };

    let fail_closed_hint = anchors.len() < settings.physical_rescue.min_anchor_count_per_run
        || rt_sigma_global.is_none()
        || reliability < settings.physical_rescue.reliability_floor;

    RtReliabilitySummary {
        rt_sigma_global,
        runwise_rt_sigma,
        reliability,
        fail_closed_hint,
    }
}

fn compute_ims_reliability(
    features: &[DfFeature],
    anchors: &[usize],
    settings: &FdrSettings,
) -> ImsReliabilitySummary {
    if !settings.physical_rescue.ims_enabled {
        return ImsReliabilitySummary {
            ims_sigma_global: None,
            runwise_ims_sigma: Vec::new(),
            reliability: 0.0,
            fail_closed_hint: false,
        };
    }
    use std::collections::HashMap;

    let mut ims_deltas: Vec<f64> = anchors
        .iter()
        .filter_map(|&i| {
            let obs = features[i].core.ims;
            let pred = features[i].core.predicted_ims;
            if obs.is_finite() && pred.is_finite() {
                Some((obs - pred).abs() as f64)
            } else {
                None
            }
        })
        .collect();

    ims_deltas.sort_by(|a, b| a.total_cmp(b));

    let ims_sigma_global = if ims_deltas.is_empty() {
        None
    } else {
        let med = ims_deltas[ims_deltas.len() / 2];
        Some((med * 1.4826).clamp(0.01, 0.25))
    };

    let mut by_file: HashMap<usize, Vec<f64>> = HashMap::new();
    for &i in anchors {
        let obs = features[i].core.ims;
        let pred = features[i].core.predicted_ims;
        if obs.is_finite() && pred.is_finite() {
            by_file
                .entry(features[i].core.file_id)
                .or_default()
                .push((obs - pred).abs() as f64);
        }
    }

    let mut runwise_ims_sigma: Vec<(usize, f64)> = Vec::new();
    for (file_id, vals) in by_file.into_iter() {
        if vals.is_empty() {
            continue;
        }
        let mut vals = vals;
        vals.sort_by(|a, b| a.total_cmp(b));
        let med = vals[vals.len() / 2];
        let sigma = (med * 1.4826).clamp(0.01, 0.25);
        runwise_ims_sigma.push((file_id, sigma));
    }
    runwise_ims_sigma.sort_by_key(|(file_id, _)| *file_id);

    let reliability = match ims_sigma_global {
        Some(sig) => {
            let raw = 1.0 / (1.0 + sig / 0.05);
            raw.clamp(0.0, 1.0)
        }
        None => 0.0,
    };

    let fail_closed_hint = settings.physical_rescue.ims_enabled
        && (ims_sigma_global.is_none() || reliability < settings.physical_rescue.reliability_floor);

    ImsReliabilitySummary {
        ims_sigma_global,
        runwise_ims_sigma,
        reliability,
        fail_closed_hint,
    }
}

fn compute_joint_physical_summary(
    rt: &RtReliabilitySummary,
    ims: &ImsReliabilitySummary,
    settings: &FdrSettings,
) -> JointPhysicalSummary {
    let joint_mode = &settings.physical_rescue.joint_mode;
    let ims_active = settings.physical_rescue.ims_enabled && ims.ims_sigma_global.is_some();

    let joint_reliability = match joint_mode {
        JointMode::Min => {
            if ims_active {
                rt.reliability.min(ims.reliability)
            } else {
                rt.reliability
            }
        }
        JointMode::Product => {
            if ims_active {
                (rt.reliability * ims.reliability).clamp(0.0, 1.0)
            } else {
                rt.reliability
            }
        }
        JointMode::Independent => {
            if ims_active {
                0.5 * rt.reliability + 0.5 * ims.reliability
            } else {
                rt.reliability
            }
        }
    };

    let fail_closed_hint = rt.fail_closed_hint || (ims_active && ims.fail_closed_hint);

    JointPhysicalSummary {
        joint_reliability,
        fail_closed_hint,
    }
}

fn should_fail_closed_l2(
    anchor_count_after_filters: usize,
    joint: &JointPhysicalSummary,
    settings: &FdrSettings,
    rt_scale_invalid: bool,
    missing_critical_diagnostics: bool,
) -> bool {
    if rt_scale_invalid {
        return true;
    }
    if missing_critical_diagnostics {
        return true;
    }
    if anchor_count_after_filters < settings.physical_rescue.min_anchor_count_per_run {
        return true;
    }
    if joint.fail_closed_hint {
        return true;
    }
    if joint.joint_reliability < settings.physical_rescue.reliability_floor {
        return true;
    }
    false
}

fn compute_dart_null_rt_params(features: &[DfFeature], anchors: &[usize]) -> (f64, f64) {
    let mut rts: Vec<f64> = anchors
        .iter()
        .filter_map(|&i| {
            let x = features[i].core.aligned_rt as f64;
            x.is_finite().then_some(x)
        })
        .collect();

    if rts.len() < 8 {
        rts = features
            .iter()
            .filter(|f| f.core.rank == 1)
            .filter_map(|f| {
                let x = f.core.aligned_rt as f64;
                x.is_finite().then_some(x)
            })
            .collect();
    }

    if rts.len() < 2 {
        return (0.5, 0.2);
    }

    let center = stats::mean(&rts);
    let spread = stats::std_dev(&rts).clamp(0.05, 0.30);
    (center, spread)
}

#[derive(Clone, Debug)]
struct L2PhysicalContext {
    pub anchors: Vec<usize>,
    pub rt_rel: RtReliabilitySummary,
    pub ims_rel: ImsReliabilitySummary,
    pub joint_rel: JointPhysicalSummary,
    pub is_unreliable: bool,
    pub rt_sigma: f64,
    pub null_rt_center: f64,
    pub null_rt_spread: f64,
    pub anchor_count_total: usize,
    pub dropped_runs: Vec<usize>,
    pub dropped_charge_bins: Vec<(i32, usize)>,
}

fn prepare_l2_physical_context(
    features: &[DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
    rt_scale_invalid: bool,
    missing_critical_diagnostics: bool,
) -> L2PhysicalContext {
    let candidates = build_physical_anchor_set(features, settings, db);
    let anchor_count_total = candidates.len();

    let (safe_anchors, _) = exclude_non_rescue_safe_anchors(features, candidates, settings, db);
    let (run_vetted, dropped_runs) =
        filter_anchor_candidates_by_run(features, safe_anchors, settings);
    let (final_anchors, dropped_charge_bins) =
        filter_anchor_candidates_by_charge(features, run_vetted, settings);

    summarize_anchor_coverage(features, settings, db, &final_anchors);

    let rt_rel = compute_rt_reliability(features, &final_anchors, settings);
    let ims_rel = compute_ims_reliability(features, &final_anchors, settings);
    let joint_rel = compute_joint_physical_summary(&rt_rel, &ims_rel, settings);

    log::debug!(
        "L2 RT reliability: global_sigma={:?}, reliability={:.3}, fail_closed_hint={}",
        rt_rel.rt_sigma_global,
        rt_rel.reliability,
        rt_rel.fail_closed_hint
    );

    if !rt_rel.runwise_rt_sigma.is_empty() {
        log::debug!("L2 RT runwise sigma: {:?}", rt_rel.runwise_rt_sigma);
    }

    if settings.physical_rescue.ims_enabled {
        log::debug!(
            "L2 IMS reliability: global_sigma={:?}, reliability={:.3}, fail_closed_hint={}",
            ims_rel.ims_sigma_global,
            ims_rel.reliability,
            ims_rel.fail_closed_hint
        );

        if !ims_rel.runwise_ims_sigma.is_empty() {
            log::debug!("L2 IMS runwise sigma: {:?}", ims_rel.runwise_ims_sigma);
        }
    } else {
        log::debug!("L2 IMS reliability: disabled");
    }

    log::debug!(
        "L2 joint physical reliability: joint_reliability={:.3}, fail_closed_hint={}",
        joint_rel.joint_reliability,
        joint_rel.fail_closed_hint
    );

    let is_unreliable = should_fail_closed_l2(
        final_anchors.len(),
        &joint_rel,
        settings,
        rt_scale_invalid,
        missing_critical_diagnostics || rt_rel.rt_sigma_global.is_none(),
    );

    let rt_sigma = if is_unreliable {
        1.0
    } else {
        rt_rel.rt_sigma_global.unwrap_or(1.0)
    };

    let (null_rt_center, null_rt_spread) = compute_dart_null_rt_params(features, &final_anchors);

    L2PhysicalContext {
        anchors: final_anchors,
        rt_rel,
        ims_rel,
        joint_rel,
        is_unreliable,
        rt_sigma,
        null_rt_center,
        null_rt_spread,
        anchor_count_total,
        dropped_runs,
        dropped_charge_bins,
    }
}

fn apply_physical_rescue(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> PhysicalRescueResult {
    use crate::input::PhysicalRescueMode;

    // 1. Capture Base Layer active outputs into explicit base fields
    for f in features.iter_mut() {
        if f.core.rank == 1 {
            f.decoy_free_p_value_base = f.decoy_free_p_value;
            f.decoy_free_pep_base = f.decoy_free_pep;
            f.decoy_free_score_base = f.decoy_free_score;
            f.decoy_free_q_base = f.decoy_free_q_value;
        }
    }

    // 2. Dispatcher
    match settings.physical_rescue.mode {
        PhysicalRescueMode::Off => {
            for f in features.iter_mut() {
                if f.core.rank == 1 {
                    f.decoy_free_p_value_l2 = f.decoy_free_p_value_base;
                    f.decoy_free_pep_l2 = f.decoy_free_pep_base;
                    f.decoy_free_score_l2 = f.decoy_free_score_base;
                    f.decoy_free_q_l2 = f.decoy_free_q_base;
                    f.physical_mode_used = Some("off".to_string());
                }
            }
            PhysicalRescueResult {
                fail_closed: false,
                anchor_count_total: 0,
                anchor_count_after_filters: 0,
                rt_reliability: 0.0,
                ims_reliability: 0.0,
                joint_reliability: 0.0,
                rt_sigma_global: None,
                ims_sigma_global: None,
                dropped_runs: Vec::new(),
                dropped_charge_bins: Vec::new(),
            }
        }
        _ => {
            let l2_ctx = prepare_l2_physical_context(features, settings, db, false, false);

            match settings.physical_rescue.mode {
                PhysicalRescueMode::DartBayes => {
                    apply_dart_bayes_update(features, settings, &l2_ctx);
                }
                PhysicalRescueMode::BoundedAux => {
                    apply_bounded_physical_shift(features, settings, &l2_ctx);
                }
                PhysicalRescueMode::Off => unreachable!(),
            }

            PhysicalRescueResult {
                fail_closed: l2_ctx.is_unreliable,
                anchor_count_total: l2_ctx.anchor_count_total,
                anchor_count_after_filters: l2_ctx.anchors.len(),
                rt_reliability: l2_ctx.rt_rel.reliability,
                ims_reliability: l2_ctx.ims_rel.reliability,
                joint_reliability: l2_ctx.joint_rel.joint_reliability,
                rt_sigma_global: l2_ctx.rt_rel.rt_sigma_global,
                ims_sigma_global: l2_ctx.ims_rel.ims_sigma_global,
                dropped_runs: l2_ctx.dropped_runs.clone(),
                dropped_charge_bins: l2_ctx.dropped_charge_bins.clone(),
            }
        }
    }
}

fn verify_dart_rt_scale_consistency(
    observed_rt: f64,
    reference_rt: f64,
    delta_rt: f64,
    null_center: f64,
    null_spread: f64,
) -> (bool, f64, f64, f64, Option<String>) {
    #[inline]
    fn classify_scale(x: f64, spread: Option<f64>) -> Option<&'static str> {
        if !x.is_finite() {
            return None;
        }
        if (-0.1..=1.1).contains(&x) && spread.map(|s| s <= 1.0).unwrap_or(true) {
            Some("normalized")
        } else if x >= 0.0 {
            Some("native")
        } else {
            None
        }
    }

    if !observed_rt.is_finite()
        || !reference_rt.is_finite()
        || !delta_rt.is_finite()
        || !null_center.is_finite()
        || !null_spread.is_finite()
        || null_spread <= 0.0
    {
        return (
            false,
            observed_rt,
            reference_rt,
            delta_rt,
            Some("non-finite-or-invalid-null".to_string()),
        );
    }

    let observed_scale = classify_scale(observed_rt, None);
    let reference_scale = classify_scale(reference_rt, None);
    let null_scale = classify_scale(null_center, Some(null_spread));

    if observed_scale.is_none() || reference_scale.is_none() || null_scale.is_none() {
        return (
            false,
            observed_rt,
            reference_rt,
            delta_rt,
            Some("unclassified-scale".to_string()),
        );
    }

    if observed_scale != reference_scale || observed_scale != null_scale {
        return (
            false,
            observed_rt,
            reference_rt,
            delta_rt,
            Some("mixed-coordinate-systems".to_string()),
        );
    }

    let internal_diff = (observed_rt - reference_rt) - delta_rt;
    if internal_diff.abs() > 1e-4 {
        return (
            false,
            observed_rt,
            reference_rt,
            delta_rt,
            Some("internal-delta-inconsistent".to_string()),
        );
    }

    (true, observed_rt, reference_rt, delta_rt, None)
}

// =============================================================================
// LAYER 2: DART-Bayes mode kernels
// =============================================================================

fn compute_dart_reference_rt(
    f: &DfFeature,
    features: &[DfFeature],
    settings: &FdrSettings,
    l2_ctx: &L2PhysicalContext,
) -> (f64, f64, bool) {
    let predicted_rt = f.core.predicted_rt as f64;
    let observed_rt = f.core.aligned_rt as f64;
    let delta_rt = f.core.delta_rt_model as f64;

    let reference_rt = if observed_rt.is_finite() && delta_rt.is_finite() {
        (observed_rt - delta_rt).clamp(0.0, 1.0)
    } else {
        predicted_rt
    };

    let reference_rt_sigma = compute_dart_bootstrap_uncertainty(f, features, settings, l2_ctx);

    let reference_rt_valid = reference_rt.is_finite()
        && (0.0..=1.0).contains(&reference_rt)
        && predicted_rt.is_finite()
        && reference_rt_sigma.is_finite()
        && reference_rt_sigma > 0.0
        && (reference_rt - predicted_rt).abs() <= 0.10;

    (reference_rt, reference_rt_sigma, reference_rt_valid)
}

fn compute_dart_true_rt_likelihood(
    observed_rt: f64,
    reference_rt: f64,
    rt_sigma: f64,
    model_type: &DartTrueRtModel,
) -> f64 {
    match model_type {
        DartTrueRtModel::Laplace => {
            crate::ml::stats::laplace_logpdf(observed_rt, reference_rt, rt_sigma)
        }
        DartTrueRtModel::Normal => {
            crate::ml::stats::normal_logpdf(observed_rt, reference_rt, rt_sigma)
        }
    }
}

fn compute_dart_null_rt_likelihood(
    observed_rt: f64,
    null_center: f64,
    null_spread: f64,
    null_model_type: &DartNullRtModel,
) -> f64 {
    match null_model_type {
        DartNullRtModel::Normal => {
            crate::ml::stats::normal_logpdf(observed_rt, null_center, null_spread)
        }
        DartNullRtModel::Uniform => {
            // A truly uniform null over the observed data spread.
            // Spread here represents the standard deviation; for a run-aware uniform
            // null, we approximate the density as 1.0 (log-lik 0.0) in the normalized [0,1] space.
            0.0
        }
    }
}

fn compute_dart_posterior_pep(prior_pep: f64, log_lik_true: f64, log_lik_null: f64) -> f64 {
    crate::ml::stats::dart_posterior_pep(prior_pep, log_lik_true, log_lik_null)
}

fn compute_dart_bootstrap_uncertainty(
    f: &DfFeature,
    features: &[DfFeature],
    settings: &FdrSettings,
    l2_ctx: &L2PhysicalContext,
) -> f64 {
    let dart_cfg = settings
        .physical_rescue
        .dart_cfg
        .as_ref()
        .expect("DART-Bayes config missing");

    let global_sigma = l2_ctx.rt_sigma.clamp(0.02, 0.30);

    if !settings.physical_rescue.use_local_rt_scale && !dart_cfg.dart_use_bootstrap {
        return global_sigma;
    }

    let target_pred_rt = f.core.predicted_rt as f64;
    if !target_pred_rt.is_finite() {
        return global_sigma;
    }

    let bins = settings.physical_rescue.rt_region_bins.max(2) as f64;
    let half_width = (1.0 / bins).clamp(0.02, 0.25);

    let mut local_abs_deltas: Vec<f64> = l2_ctx
        .anchors
        .iter()
        .filter_map(|&idx| {
            let a = &features[idx];
            let pred = a.core.predicted_rt as f64;
            let delta = a.core.delta_rt_model as f64;
            if pred.is_finite() && delta.is_finite() && (pred - target_pred_rt).abs() <= half_width
            {
                Some(delta.abs())
            } else {
                None
            }
        })
        .collect();

    if local_abs_deltas.len() < 5 {
        return global_sigma;
    }

    local_abs_deltas.sort_by(|a, b| a.total_cmp(b));
    let med = local_abs_deltas[local_abs_deltas.len() / 2];
    let local_sigma = (med * 1.4826).clamp(0.02, 0.30);

    let shrink = settings.physical_rescue.cov_shrinkage.clamp(0.0, 1.0);
    let sigma = if settings.physical_rescue.use_local_rt_scale || dart_cfg.dart_use_bootstrap {
        ((1.0 - shrink) * local_sigma + shrink * global_sigma).clamp(0.02, 0.30)
    } else {
        global_sigma
    };

    sigma
}

fn finalize_dart_l2_q_values(features: &mut [DfFeature]) {
    let mut rows: Vec<(f64, usize, f64)> = Vec::new();

    for (i, f) in features.iter().enumerate() {
        if f.core.rank == 1 {
            if let Some(pep) = f.decoy_free_pep_l2 {
                if pep.is_finite() {
                    let score_key = df_rank_score_l2_pep(f);
                    rows.push((score_key, i, (pep as f64).clamp(0.0, 1.0).max(1e-300)));
                }
            }
        }
    }

    for (feat_idx, q) in q_from_pep_cummean(rows) {
        features[feat_idx].decoy_free_q_l2 = Some(q as f32);
    }
}

fn apply_dart_bayes_update(
    // This function writes stage-local Layer 2 PEP-native outputs only.
    // It does not activate the final DF stream.
    // Final activation happens only in finalize_df_psm_stream(...).
    features: &mut [DfFeature],
    settings: &FdrSettings,
    l2_ctx: &L2PhysicalContext,
) {
    let is_unreliable = l2_ctx.is_unreliable;

    let dart_cfg = settings
        .physical_rescue
        .dart_cfg
        .as_ref()
        .expect("DART-Bayes config must be provided");

    let null_center = l2_ctx.null_rt_center;
    let null_spread = l2_ctx.null_rt_spread;

    let snapshot = features.to_vec();

    for f in features.iter_mut() {
        if f.core.rank != 1 {
            continue;
        }

        f.decoy_free_p_value_l2 = f.decoy_free_p_value_base;
        f.decoy_free_pep_l2 = f.decoy_free_pep_base;
        f.decoy_free_score_l2 = f.decoy_free_score_base;
        f.decoy_free_q_l2 = f.decoy_free_q_base;

        f.physical_mode_used = Some(
            if is_unreliable {
                "dart_bayes_fail_closed"
            } else {
                "dart_bayes"
            }
            .to_string(),
        );

        if is_unreliable {
            f.dart_posterior_used = Some(false);
            continue;
        }

        let observed_rt = f.core.aligned_rt as f64;
        let predicted_rt = f.core.predicted_rt as f64;
        let delta_rt = f.core.delta_rt_model as f64;

        let (rt_ok, _, _, _, reason) = verify_dart_rt_scale_consistency(
            observed_rt,
            predicted_rt,
            delta_rt,
            null_center,
            null_spread,
        );

        let (reference_rt, reference_sigma, reference_valid) =
            compute_dart_reference_rt(f, &snapshot, settings, l2_ctx);

        if !rt_ok || !reference_valid {
            f.dart_posterior_used = Some(false);
            if let Some(reason) = reason {
                f.physical_mode_used = Some(format!("dart_bayes_fail_closed:{reason}"));
            }
            continue;
        }

        let prior_pep = f.decoy_free_pep_base.unwrap_or(1.0) as f64;

        let log_lik_true = compute_dart_true_rt_likelihood(
            observed_rt,
            reference_rt,
            reference_sigma,
            &dart_cfg.dart_true_rt_model,
        );

        let log_lik_null = compute_dart_null_rt_likelihood(
            observed_rt,
            null_center,
            null_spread,
            &dart_cfg.dart_null_rt_model,
        );

        let posterior_pep = compute_dart_posterior_pep(prior_pep, log_lik_true, log_lik_null);

        f.decoy_free_pep_l2 = Some(posterior_pep as f32);
        f.decoy_free_score_l2 = Some((-10.0 * posterior_pep.max(1e-15).log10()) as f32);

        f.dart_posterior_used = Some(true);
        f.dart_rt_lik_correct = Some(log_lik_true as f32);
        f.dart_rt_lik_incorrect = Some(log_lik_null as f32);
    }

    if !is_unreliable && dart_cfg.dart_recalc_q_from_posterior {
        finalize_dart_l2_q_values(features);
    }
}

// =============================================================================
// LAYER 2: Bounded auxiliary mode kernels
// =============================================================================

/// Compute a bounded auxiliary rescue shift on the logit-confidence scale.
///
/// Positive values support rescue; negative values support demotion.
/// The returned value is not a probability-space increment.

fn compute_physical_shift(f: &DfFeature, rt_sigma: f64) -> f64 {
    let delta = f.core.delta_rt_model as f64;
    let predicted = f.core.predicted_rt as f64;

    // Missing or invalid evidence -> zero shift/penalty
    if !delta.is_finite() || !predicted.is_finite() || predicted <= 0.0 || predicted >= 1.0 {
        return 0.0;
    }

    let z = delta / rt_sigma.max(1e-9);

    // Base log-odds shift: Reward tight RT (Z ~ 0), mildly penalize wide RT.
    // Yields +2.0 at z=0, 0.0 at z=1.414, and goes negative for z > 1.414.
    // The exact min/max bounds are strictly enforced downstream by the soft_cap.
    2.0 - z.powi(2)
}

fn finalize_bounded_l2_q_values(features: &mut [DfFeature]) {
    let mut rows: Vec<(f64, usize, f64)> = Vec::new();

    for (i, f) in features.iter().enumerate() {
        if f.core.rank == 1 {
            if let Some(pep) = f.decoy_free_pep_l2 {
                if pep.is_finite() {
                    let score_key = df_rank_score_l2_pep(f);
                    rows.push((score_key, i, (pep as f64).clamp(0.0, 1.0).max(1e-300)));
                }
            }
        }
    }

    for (feat_idx, q) in q_from_pep_cummean(rows) {
        features[feat_idx].decoy_free_q_l2 = Some(q as f32);
    }
}

fn apply_bounded_physical_shift(
    // This function writes stage-local Layer 2 PEP-native outputs only.
    // It does not activate the final DF stream.
    // Final activation happens only in finalize_df_psm_stream(...).
    features: &mut [DfFeature],
    settings: &FdrSettings,
    l2_ctx: &L2PhysicalContext,
) {
    let is_unreliable = l2_ctx.is_unreliable;
    let rt_sigma = l2_ctx.rt_sigma;

    let cfg = settings
        .physical_rescue
        .bounded_cfg
        .as_ref()
        .expect("BoundedAux config must be provided");

    for f in features.iter_mut() {
        if f.core.rank != 1 {
            continue;
        }

        f.decoy_free_p_value_l2 = f.decoy_free_p_value_base;
        f.decoy_free_pep_l2 = f.decoy_free_pep_base;
        f.decoy_free_score_l2 = f.decoy_free_score_base;
        f.decoy_free_q_l2 = f.decoy_free_q_base;

        let mode_str = if is_unreliable {
            "bounded_aux_fail_closed"
        } else {
            "bounded_aux"
        };
        f.physical_mode_used = Some(mode_str.to_string());

        if is_unreliable {
            continue;
        }

        let prior_pep = f.decoy_free_pep_base.unwrap_or(1.0) as f64;

        let missing_rt = !f.core.aligned_rt.is_finite()
            || !f.core.predicted_rt.is_finite()
            || !f.core.delta_rt_model.is_finite();

        let missing_ims = settings.physical_rescue.ims_enabled
            && (!f.core.ims.is_finite() || !f.core.predicted_ims.is_finite());

        let missing_penalty = if missing_rt || missing_ims {
            settings.physical_rescue.missing_penalty.max(0.0)
        } else {
            0.0
        };

        // The physical rescue shift is defined on the logit-confidence scale.
        // It is capped before being added to the prior logit-confidence.
        let raw_shift = compute_physical_shift(f, rt_sigma) - missing_penalty;
        let bounded_shift =
            crate::ml::stats::capped_shift(raw_shift, cfg.max_rescue_shift, cfg.max_penalty_shift);

        let posterior_pep = match cfg.update_space {
            BoundedAuxUpdateSpace::LogitConfidence => {
                let logit_prior = crate::ml::stats::safe_logit_confidence(prior_pep);
                let logit_post = logit_prior + bounded_shift;
                crate::ml::stats::safe_inv_logit_confidence(logit_post)
            }
        };

        f.decoy_free_pep_l2 = Some(posterior_pep as f32);
        let df_score = (-10.0 * posterior_pep.max(1e-15).log10()) as f32;
        f.decoy_free_score_l2 = Some(df_score);

        f.physical_shift_total = Some(bounded_shift as f32);

        if bounded_shift > 0.0 && (cfg.max_rescue_shift - bounded_shift).abs() < 0.05 {
            f.physical_cap_hit_pos = Some(true);
        } else if bounded_shift < 0.0 && (cfg.max_penalty_shift - (-bounded_shift)).abs() < 0.05 {
            f.physical_cap_hit_neg = Some(true);
        }
    }

    if !is_unreliable {
        finalize_bounded_l2_q_values(features);
    }
}

#[derive(Clone, Debug, Default)]
struct L3ProteinSupportSummary {
    pub n_unique_observed: usize,
    pub n_unique_passing_l2: usize,
    pub is_rescue_eligible: bool,
}

#[derive(Clone, Debug, Default)]
struct L3PeptideEligibilitySummary {
    pub n_runs_observed: usize,
    pub n_runs_strong_l2: usize,
    pub observed_run_fraction: f64,
    pub strong_run_fraction: f64,
    pub is_rescue_eligible: bool,
    pub protein_rescue_eligible: bool,
}

#[derive(Clone, Debug, Default)]
struct L3AnchorSummary {
    pub anchor_value: f64,
    pub n_anchor_observations: usize,
}

// =============================================================================
// LAYER 3: Reproducibility and agreement
// =============================================================================

fn compute_expert_agreement_support(f: &DfFeature, settings: &FdrSettings) -> f64 {
    if !settings.reproducibility.use_expert_agreement {
        return 0.0;
    }
    let mut strong_experts = 0;
    let threshold = 0.05; // P-value threshold for "strong" agreement

    // Base experts
    if f.p_mom.unwrap_or(1.0) < threshold {
        strong_experts += 1;
    }
    if f.p_mle.unwrap_or(1.0) < threshold {
        strong_experts += 1;
    }
    if f.p_lo.unwrap_or(1.0) < threshold {
        strong_experts += 1;
    }
    if f.p_nokoi.unwrap_or(1.0) < threshold {
        strong_experts += 1;
    }

    // Mixture experts (highly redundant, count them as max 1 independent vote)
    let mut mix_experts = 0;
    if f.p_msfdr.unwrap_or(1.0) < threshold {
        mix_experts += 1;
    }
    if f.p_1smix.unwrap_or(1.0) < threshold {
        mix_experts += 1;
    }
    if f.p_2smix.unwrap_or(1.0) < threshold {
        mix_experts += 1;
    }
    if mix_experts > 0 {
        strong_experts += 1;
    }

    if strong_experts <= 1 {
        return 0.0;
    }

    let reward = (strong_experts as f64 - 1.0) * 0.5;
    crate::ml::stats::soft_cap(reward, settings.reproducibility.max_agreement_shift)
}

fn compute_cross_run_recurrence_support(
    elig: &L3PeptideEligibilitySummary,
    settings: &FdrSettings,
) -> f64 {
    if !settings.reproducibility.use_cross_run_recurrence {
        return 0.0;
    }

    if !elig.is_rescue_eligible {
        return 0.0;
    }

    let pep_cfg = &settings.reproducibility.peptide_eligibility;

    let observed_excess = (elig.observed_run_fraction - pep_cfg.min_run_fraction).max(0.0);
    let strong_excess = (elig.strong_run_fraction - pep_cfg.min_strong_run_fraction).max(0.0);

    // Recurrence support is based on how far the peptide exceeds the minimum
    // recurrence requirements used for L3 eligibility. Strong-run excess is
    // weighted more heavily than observed-run excess.
    let raw_support = observed_excess + 2.0 * strong_excess;

    crate::ml::stats::soft_cap(raw_support, settings.reproducibility.max_recurrence_shift)
}

fn compute_redundancy_discount(_f: &DfFeature, settings: &FdrSettings) -> f64 {
    settings.reproducibility.redundancy_discount.clamp(0.0, 1.0)
}

fn build_l3_protein_support_map(
    features: &[DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
) -> FnvHashMap<String, L3ProteinSupportSummary> {
    let cfg = &settings.reproducibility.protein_eligibility;

    let mut observed: FnvHashMap<String, FnvHashSet<String>> = FnvHashMap::default();
    let mut passing: FnvHashMap<String, FnvHashSet<String>> = FnvHashMap::default();

    for f in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let protein_key = match df_unique_protein_key_for_feature(f, db) {
            Some(k) => k,
            None => continue,
        };

        let peptide_seq = db[f.core.peptide_idx].to_string();
        observed
            .entry(protein_key.clone())
            .or_default()
            .insert(peptide_seq.clone());

        if f.decoy_free_q_l2
            .map(|q| q <= cfg.q_threshold_l2 as f32)
            .unwrap_or(false)
        {
            passing.entry(protein_key).or_default().insert(peptide_seq);
        }
    }

    let mut out: FnvHashMap<String, L3ProteinSupportSummary> = FnvHashMap::default();
    for (protein_key, obs_set) in observed {
        let n_unique_observed = obs_set.len();
        let n_unique_passing_l2 = passing.get(&protein_key).map(|s| s.len()).unwrap_or(0);

        let frac_ok = match cfg.min_unique_passing_fraction {
            Some(frac) if n_unique_observed > 0 => {
                (n_unique_passing_l2 as f64) / (n_unique_observed as f64) >= frac
            }
            Some(_) => false,
            None => true,
        };

        let is_rescue_eligible = if cfg.enabled {
            n_unique_passing_l2 >= cfg.min_unique_passing_peptides && frac_ok
        } else {
            true
        };

        out.insert(
            protein_key,
            L3ProteinSupportSummary {
                n_unique_observed,
                n_unique_passing_l2,
                is_rescue_eligible,
            },
        );
    }

    out
}

fn build_l3_peptide_eligibility_map(
    features: &[DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
    protein_support_map: &FnvHashMap<String, L3ProteinSupportSummary>,
) -> FnvHashMap<u32, L3PeptideEligibilitySummary> {
    let cfg = &settings.reproducibility.peptide_eligibility;

    let total_runs = features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
        .map(|f| f.core.file_id)
        .collect::<FnvHashSet<_>>()
        .len()
        .max(1);

    let mut observed_runs: FnvHashMap<u32, FnvHashSet<usize>> = FnvHashMap::default();
    let mut strong_runs: FnvHashMap<u32, FnvHashSet<usize>> = FnvHashMap::default();
    let mut protein_ok_by_peptide: FnvHashMap<u32, bool> = FnvHashMap::default();

    for f in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let pep_id = f.core.peptide_idx.0;

        observed_runs
            .entry(pep_id)
            .or_default()
            .insert(f.core.file_id);

        let protein_rescue_eligible = df_unique_protein_key_for_feature(f, db)
            .and_then(|k| protein_support_map.get(&k).map(|s| s.is_rescue_eligible))
            .unwrap_or(false);

        protein_ok_by_peptide
            .entry(pep_id)
            .and_modify(|v| *v |= protein_rescue_eligible)
            .or_insert(protein_rescue_eligible);

        let q_ok = f
            .decoy_free_q_l2
            .map(|q| q <= cfg.strong_reference_q_threshold_l2 as f32)
            .unwrap_or(false);

        let pep_ok = match cfg.strong_reference_pep_threshold_l2 {
            Some(thr) => f
                .decoy_free_pep_l2
                .map(|p| p <= thr as f32)
                .unwrap_or(false),
            None => true,
        };

        if q_ok && pep_ok {
            strong_runs
                .entry(pep_id)
                .or_default()
                .insert(f.core.file_id);
        }
    }

    let mut out: FnvHashMap<u32, L3PeptideEligibilitySummary> = FnvHashMap::default();
    for (pep_id, obs_set) in observed_runs {
        let n_runs_observed = obs_set.len();
        let n_runs_strong_l2 = strong_runs.get(&pep_id).map(|s| s.len()).unwrap_or(0);
        let protein_rescue_eligible = protein_ok_by_peptide.get(&pep_id).copied().unwrap_or(false);

        let observed_run_fraction = (n_runs_observed as f64) / (total_runs as f64);
        let strong_run_fraction = (n_runs_strong_l2 as f64) / (total_runs as f64);

        let observed_frac_ok = observed_run_fraction >= cfg.min_run_fraction;
        let observed_count_ok = n_runs_observed >= cfg.min_run_count;

        let strong_frac_ok = strong_run_fraction >= cfg.min_strong_run_fraction;
        let strong_count_ok = n_runs_strong_l2 >= cfg.min_strong_run_count;

        let is_rescue_eligible = protein_rescue_eligible
            && observed_frac_ok
            && observed_count_ok
            && strong_frac_ok
            && strong_count_ok;

        out.insert(
            pep_id,
            L3PeptideEligibilitySummary {
                n_runs_observed,
                n_runs_strong_l2,
                observed_run_fraction,
                strong_run_fraction,
                is_rescue_eligible,
                protein_rescue_eligible,
            },
        );
    }

    out
}

fn build_l3_anchor_map(
    features: &[DfFeature],
    settings: &FdrSettings,
    peptide_eligibility_map: &FnvHashMap<u32, L3PeptideEligibilitySummary>,
) -> FnvHashMap<u32, L3AnchorSummary> {
    let pep_cfg = &settings.reproducibility.peptide_eligibility;
    let anchor_cfg = &settings.reproducibility.anchor;

    let mut strong_peps: FnvHashMap<u32, Vec<f64>> = FnvHashMap::default();

    for f in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let pep_id = f.core.peptide_idx.0;
        let elig = match peptide_eligibility_map.get(&pep_id) {
            Some(x) if x.is_rescue_eligible => x,
            _ => continue,
        };

        let _ = elig;

        let q_ok = f
            .decoy_free_q_l2
            .map(|q| q <= pep_cfg.strong_reference_q_threshold_l2 as f32)
            .unwrap_or(false);

        let pep_ok = match pep_cfg.strong_reference_pep_threshold_l2 {
            Some(thr) => f
                .decoy_free_pep_l2
                .map(|p| p <= thr as f32)
                .unwrap_or(false),
            None => true,
        };

        if !(q_ok && pep_ok) {
            continue;
        }

        if let Some(pep) = f.decoy_free_pep_l2 {
            let pep = (pep as f64).clamp(0.0, 1.0).max(1e-300);
            strong_peps.entry(pep_id).or_default().push(pep);
        }
    }

    let mut out: FnvHashMap<u32, L3AnchorSummary> = FnvHashMap::default();
    for (pep_id, vals) in strong_peps {
        if vals.is_empty() {
            continue;
        }

        let mut tmp = vals.clone();
        let anchor_value = match anchor_cfg.mode {
            L3AnchorMode::Best => tmp.iter().copied().min_by(|a, b| a.total_cmp(b)),
            L3AnchorMode::SecondBest => second_best_f64(&mut tmp),
            L3AnchorMode::Mean => Some(tmp.iter().sum::<f64>() / (tmp.len() as f64)),
            L3AnchorMode::Median => median_f64(tmp),
            L3AnchorMode::TrimmedMean => {
                let trim = anchor_cfg.trim_fraction.unwrap_or(0.1);
                trimmed_mean(&mut tmp, trim)
            }
        };

        if let Some(anchor_value) = anchor_value {
            out.insert(
                pep_id,
                L3AnchorSummary {
                    anchor_value: anchor_value.clamp(0.0, 1.0).max(1e-300),
                    n_anchor_observations: vals.len(),
                },
            );
        }
    }

    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum L3RescueBand {
    Strong,
    RescueEligible,
    TooWeak,
}

fn classify_l3_rescue_band(prior_pep_l2: f64, settings: &FdrSettings) -> L3RescueBand {
    let band = &settings.reproducibility.rescue_band;
    let strong_cutoff = band.strong_cutoff_pep_l2;
    let weak_cutoff = band.weak_cutoff_pep_l2;

    if prior_pep_l2 <= strong_cutoff {
        L3RescueBand::Strong
    } else if prior_pep_l2 < weak_cutoff {
        L3RescueBand::RescueEligible
    } else {
        L3RescueBand::TooWeak
    }
}

fn apply_l3_anchor_rescue(
    prior_pep_l2: f64,
    anchor_pep: f64,
    expert_support_shift: f64,
    recurrence_support_shift: f64,
    settings: &FdrSettings,
) -> (f64, f64) {
    let band = &settings.reproducibility.rescue_band;
    let max_frac = band.max_rescue_fraction.clamp(0.0, 1.0);

    let prior = prior_pep_l2.clamp(0.0, 1.0).max(1e-300);
    let anchor = anchor_pep.clamp(0.0, 1.0).max(1e-300);

    let improved_target = anchor.min(prior);
    let rescued_pep = match band.rescue_mode {
        L3RescueMode::Replace => improved_target,
        L3RescueMode::BoundedShrinkage => prior + max_frac * (improved_target - prior),
    }
    .clamp(0.0, 1.0)
    .max(1e-300);

    let prior_logit = crate::ml::stats::safe_logit_confidence(prior);
    let rescue_logit = crate::ml::stats::safe_logit_confidence(rescued_pep);

    let recurrence_support_shift =
        recurrence_support_shift.clamp(0.0, settings.reproducibility.max_recurrence_shift);

    let combined_shift =
        (rescue_logit - prior_logit) + expert_support_shift + recurrence_support_shift;

    let bounded_shift = crate::ml::stats::capped_shift(
        combined_shift,
        settings.reproducibility.max_total_shift,
        settings.reproducibility.max_total_shift,
    );

    let post_pep =
        crate::ml::stats::safe_inv_logit_confidence(prior_logit + bounded_shift).clamp(0.0, 1.0);

    (post_pep, bounded_shift.abs())
}

fn finalize_repro_l3_q_values(features: &mut [DfFeature]) {
    let mut rows: Vec<(f64, usize, f64)> = Vec::new();

    for (i, f) in features.iter().enumerate() {
        if f.core.rank == 1 {
            if let Some(pep) = f.decoy_free_pep_l3 {
                if pep.is_finite() {
                    let score_key = df_rank_score_l3_pep(f);
                    rows.push((score_key, i, (pep as f64).clamp(0.0, 1.0).max(1e-300)));
                }
            }
        }
    }

    for (feat_idx, q) in q_from_pep_cummean(rows) {
        features[feat_idx].decoy_free_q_l3 = Some(q as f32);
    }
}

fn clear_l3_stage_outputs(features: &mut [DfFeature]) {
    for f in features.iter_mut() {
        if f.core.rank == 1 {
            f.decoy_free_pep_l3 = None;
            f.decoy_free_score_l3 = None;
            f.decoy_free_q_l3 = None;
        }
    }
}

fn apply_bounded_repro_shift(
    // This function writes stage-local Layer 3 PEP-native outputs only.
    // It does not activate the final DF stream.
    // Final activation happens only in finalize_df_psm_stream(...).
    //
    // Layer 3 does not grant a universal peptide-level recurrence bonus.
    // Rescue is allowed only for peptides that pass:
    //   1) protein-support eligibility from Layer 2,
    //   2) peptide recurrence eligibility,
    //   3) strong-reference eligibility.
    //
    // Only weaker-but-plausible PSMs are rescue candidates.
    // Strong PSMs are left unchanged.
    // Very weak PSMs are not rescued.
    // Rescue-eligible PSMs move in a bounded way toward a Layer 2-derived anchor.
    // Cross-run recurrence, when enabled, contributes only as an additional
    // bounded support term within that rescue path.
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> ReproducibilityResult {
    if !settings.reproducibility.enabled {
        for f in features.iter_mut() {
            if f.core.rank == 1 {
                f.decoy_free_pep_l3 = f.decoy_free_pep_l2;
                f.decoy_free_score_l3 = f.decoy_free_score_l2;
                f.decoy_free_q_l3 = f.decoy_free_q_l2;
            }
        }
        return ReproducibilityResult {
            enabled: false,
            ..Default::default()
        };
    }

    let protein_support_map = build_l3_protein_support_map(features, db, settings);

    let total_unique_observed: usize = protein_support_map
        .values()
        .map(|s| s.n_unique_observed)
        .sum();

    let total_unique_passing_l2: usize = protein_support_map
        .values()
        .map(|s| s.n_unique_passing_l2)
        .sum();

    log::debug!(
		"L3 protein support: proteins={} eligible={} total_unique_observed={} total_unique_passing_l2={}",
		protein_support_map.len(),
		protein_support_map.values().filter(|s| s.is_rescue_eligible).count(),
		total_unique_observed,
		total_unique_passing_l2
	);

    let peptide_eligibility_map =
        build_l3_peptide_eligibility_map(features, db, settings, &protein_support_map);

    let total_runs_observed: usize = peptide_eligibility_map
        .values()
        .map(|s| s.n_runs_observed)
        .sum();

    let total_runs_strong_l2: usize = peptide_eligibility_map
        .values()
        .map(|s| s.n_runs_strong_l2)
        .sum();

    let protein_backed_peptides = peptide_eligibility_map
        .values()
        .filter(|s| s.protein_rescue_eligible)
        .count();

    log::debug!(
		"L3 peptide eligibility: peptides={} eligible={} protein_backed={} total_runs_observed={} total_runs_strong_l2={}",
		peptide_eligibility_map.len(),
		peptide_eligibility_map.values().filter(|s| s.is_rescue_eligible).count(),
		protein_backed_peptides,
		total_runs_observed,
		total_runs_strong_l2
	);

    let anchor_map = build_l3_anchor_map(features, settings, &peptide_eligibility_map);

    let total_anchor_observations: usize =
        anchor_map.values().map(|s| s.n_anchor_observations).sum();

    log::debug!(
        "L3 anchors: peptides_with_anchor={} total_anchor_observations={}",
        anchor_map.len(),
        total_anchor_observations
    );

    let n_rescue_eligible_proteins = protein_support_map
        .values()
        .filter(|s| s.is_rescue_eligible)
        .count();
    let n_rescue_eligible_peptides = peptide_eligibility_map
        .values()
        .filter(|s| s.is_rescue_eligible)
        .count();
    let n_anchor_peptides = anchor_map.len();

    let mut n_rescued_psms = 0usize;
    let mut n_strong_unchanged_psms = 0usize;
    let mut n_too_weak_unrescued_psms = 0usize;

    let mut agreement_sum = 0.0f64;
    let mut max_shift_applied = 0.0f64;
    let mut cnt = 0usize;

    for f in features.iter_mut() {
        if f.core.rank != 1 {
            continue;
        }

        cnt += 1;

        let prior_pep = f.decoy_free_pep_l2.unwrap_or(1.0) as f64;
        let pep_id = f.core.peptide_idx.0;

        let expert_s = compute_expert_agreement_support(f, settings)
            * compute_redundancy_discount(f, settings);
        agreement_sum += expert_s;

        let eligibility = peptide_eligibility_map.get(&pep_id);
        let recurrence_s = eligibility
            .map(|elig| compute_cross_run_recurrence_support(elig, settings))
            .unwrap_or(0.0);

        let band = classify_l3_rescue_band(prior_pep, settings);

        let (post_pep, shift_abs) = match (eligibility, band) {
            (Some(elig), L3RescueBand::RescueEligible) if elig.is_rescue_eligible => {
                if let Some(anchor) = anchor_map.get(&pep_id) {
                    let (post, shift_abs) = apply_l3_anchor_rescue(
                        prior_pep,
                        anchor.anchor_value,
                        expert_s,
                        recurrence_s,
                        settings,
                    );
                    if post + 1e-12 < prior_pep {
                        n_rescued_psms += 1;
                    }
                    (post, shift_abs)
                } else {
                    (prior_pep, 0.0)
                }
            }
            (_, L3RescueBand::Strong) => {
                n_strong_unchanged_psms += 1;
                (prior_pep, 0.0)
            }
            (_, L3RescueBand::TooWeak) => {
                n_too_weak_unrescued_psms += 1;
                (prior_pep, 0.0)
            }
            _ => {
                // Not eligible for peptide-context rescue.
                // Leave unchanged rather than applying a generic recurrence bonus.
                (prior_pep, 0.0)
            }
        };

        max_shift_applied = max_shift_applied.max(shift_abs);

        f.decoy_free_pep_l3 = Some(post_pep as f32);
        f.decoy_free_score_l3 = Some((-10.0 * post_pep.max(1e-15).log10()) as f32);
    }

    finalize_repro_l3_q_values(features);

    ReproducibilityResult {
        enabled: true,
        fail_closed: false,
        n_rescue_eligible_proteins,
        n_rescue_eligible_peptides,
        n_anchor_peptides,
        n_rescued_psms,
        n_strong_unchanged_psms,
        n_too_weak_unrescued_psms,
        agreement_support_mean: agreement_sum / cnt.max(1) as f64,
        max_shift_applied,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FinalDfStream {
    Base,
    Layer2,
    Layer3,
}

#[inline]
fn activate_final_base_stream(f: &mut DfFeature) {
    f.decoy_free_p_value = f.decoy_free_p_value_base;
    f.decoy_free_pep = f.decoy_free_pep_base;
    f.decoy_free_score = f.decoy_free_score_base;
    f.decoy_free_q_value = f.decoy_free_q_base;
}

#[inline]
fn activate_final_pep_stream(
    f: &mut DfFeature,
    pep: Option<f32>,
    score: Option<f32>,
    q: Option<f32>,
) {
    f.decoy_free_p_value = None;
    f.decoy_free_pep = pep;
    f.decoy_free_score = score;
    f.decoy_free_q_value = q;
}

#[inline]
fn detect_final_df_stream(features: &[DfFeature], settings: &FdrSettings) -> FinalDfStream {
    use crate::input::ModelFit;

    // Phase 1 / Step 1 contract:
    // For p-value-native selected-model runs, keep the canonical exported
    // decoy_free_* columns on the Base stream.
    //
    // L2/L3 stage-local fields may still be computed and logged, but they must
    // not become the canonical final stream for these model fits.
    match settings.model_fit {
        ModelFit::Moments
        | ModelFit::Mle
        | ModelFit::LowerOrder
        | ModelFit::Msfdr
        | ModelFit::Msfdr1Smix
        | ModelFit::Msfdr2Smix => {
            return FinalDfStream::Base;
        }
        _ => {}
    }

    let l3_enabled = settings.reproducibility.enabled;
    let l2_enabled = settings.physical_rescue.mode != PhysicalRescueMode::Off;

    if l3_enabled
        && features
            .iter()
            .any(|f| f.core.rank == 1 && f.decoy_free_pep_l3.is_some())
    {
        FinalDfStream::Layer3
    } else if l2_enabled
        && features
            .iter()
            .any(|f| f.core.rank == 1 && f.decoy_free_pep_l2.is_some())
    {
        FinalDfStream::Layer2
    } else {
        FinalDfStream::Base
    }
}

fn finalize_df_psm_stream(features: &mut [DfFeature], settings: &FdrSettings) -> FinalDfStream {
    let final_stream = detect_final_df_stream(features, settings);

    for f in features.iter_mut() {
        if f.core.rank != 1 {
            continue;
        }

        match final_stream {
            FinalDfStream::Layer3 => {
                activate_final_pep_stream(
                    f,
                    f.decoy_free_pep_l3,
                    f.decoy_free_score_l3,
                    f.decoy_free_q_l3,
                );
            }
            FinalDfStream::Layer2 => {
                activate_final_pep_stream(
                    f,
                    f.decoy_free_pep_l2,
                    f.decoy_free_score_l2,
                    f.decoy_free_q_l2,
                );
            }
            FinalDfStream::Base => {
                activate_final_base_stream(f);
            }
        }
    }

    final_stream
}

fn validate_final_df_stream_contract(features: &[DfFeature], final_stream: FinalDfStream) {
    let mut n_rank1 = 0usize;
    let mut n_p_native = 0usize;
    let mut n_pep_native = 0usize;
    let mut n_invalid = 0usize;

    for f in features.iter().filter(|f| f.core.rank == 1) {
        n_rank1 += 1;

        let has_p = f.decoy_free_p_value.is_some();
        let has_pep = f.decoy_free_pep.is_some();
        let has_q = f.decoy_free_q_value.is_some();

        match final_stream {
            FinalDfStream::Base => {
                if has_p {
                    n_p_native += 1;
                }
                if !has_p || !has_pep || !has_q {
                    n_invalid += 1;
                }
            }
            FinalDfStream::Layer2 | FinalDfStream::Layer3 => {
                if !has_p {
                    n_pep_native += 1;
                }
                if has_p || !has_pep || !has_q {
                    n_invalid += 1;
                }
            }
        }
    }

    if n_invalid > 0 {
        let stream_name = match final_stream {
            FinalDfStream::Base => "base / p-value-native",
            FinalDfStream::Layer2 => "layer2 / pep-native",
            FinalDfStream::Layer3 => "layer3 / pep-native",
        };

        let msg = format!(
            "DF final stream contract violated: stream={} rank1={} p_native={} pep_native={} invalid_rows={}",
            stream_name, n_rank1, n_p_native, n_pep_native, n_invalid
        );
        log::error!("{}", msg);
        panic!("{}", msg);
    }

    log::debug!(
        "DF final stream contract OK: stream={:?} rank1={} p_native={} pep_native={}",
        final_stream,
        n_rank1,
        n_p_native,
        n_pep_native
    );
}

pub fn run_df_layers(
    psms: &[DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> Vec<DfFeature> {
    let mut new_features = psms.to_vec();

    let use_ensemble = matches!(settings.model_fit, ModelFit::Ensemble);

    let run_mom = if use_ensemble {
        settings.enable_moments
    } else {
        matches!(settings.model_fit, ModelFit::Moments)
    };
    let run_mle = if use_ensemble {
        settings.enable_mle
    } else {
        matches!(settings.model_fit, ModelFit::Mle)
    };
    let run_lo = if use_ensemble {
        settings.enable_lower_order
    } else {
        matches!(settings.model_fit, ModelFit::LowerOrder)
    };
    let run_msfdr_seeded = if use_ensemble {
        settings.enable_msfdr_seeded
    } else {
        matches!(settings.model_fit, ModelFit::Msfdr)
    };
    let run_msfdr_1smix = if use_ensemble {
        settings.enable_msfdr_1smix
    } else {
        matches!(settings.model_fit, ModelFit::Msfdr1Smix)
    };
    let run_msfdr_2smix = if use_ensemble {
        settings.enable_msfdr_2smix
    } else {
        matches!(settings.model_fit, ModelFit::Msfdr2Smix)
    };
    let run_nokoi = use_ensemble || matches!(settings.model_fit, ModelFit::Nokoi);

    let gates = RunGates {
        run_mom,
        run_mle,
        run_lo,
        run_msfdr_seeded,
        run_msfdr_1smix,
        run_msfdr_2smix,
        run_nokoi,
    };

    // 1A. Work Set
    let work = build_base_workset(&new_features);

    log::info!(
        "DF: rank1_work={} model_fit={:?} ensemble={}",
        work.n_rank1(),
        settings.model_fit,
        use_ensemble
    );

    // 1B. Null Pool
    let pool = match build_base_null_pool(&new_features, &work, settings) {
        Some(p) => p,
        None => {
            log::error!("Null distribution too small. Aborting FDR.");
            new_features.par_iter_mut().for_each(|psm| {
                clear_all_df_outputs(psm, true);
            });
            return new_features;
        }
    };

    log::info!("DF: pool_size={}", pool.fit_data.len());

    // 1C. Experts
    let engines = match fit_base_experts(&new_features, &work, &pool, settings, gates) {
        Some(e) => e,
        None => {
            log::error!("Invalid null fit. FDR will fail closed.");
            new_features.par_iter_mut().for_each(|psm| {
                clear_all_df_outputs(psm, true);
            });
            return new_features;
        }
    };

    // Diagnostic Logging before moving engines
    let (mom_mu, mom_beta) = engines.mom_params.unwrap_or((f64::NAN, f64::NAN));
    let (mle_mu, mle_beta) = engines.mle_params.unwrap_or((f64::NAN, f64::NAN));
    log::info!(
        "DF fit summary: moments_null=(mu={:.6}, beta={:.6}) mle_null=(mu={:.6}, beta={:.6})",
        mom_mu,
        mom_beta,
        mle_mu,
        mle_beta
    );

    match &engines.lo_model {
        Some(m) => log::info!(
            "DF fit summary: LO fallback_params=(mu={:.6}, beta={:.6})",
            m.fallback_params.0,
            m.fallback_params.1
        ),
        None => log::warn!("DF fail-closed: LO failed to fit (no fitted charges)."),
    }
    match &engines.msfdr_seeded {
        Some(m) => log_fit_ok("MSFDR seeded", m),
        None => log::info!("DF MSFDR seeded: absent"),
    }
    match &engines.msfdr_1smix {
        Some(m) => log_fit_ok("MSFDR 1smix", m),
        None => log::info!("DF MSFDR 1smix: absent"),
    }
    match &engines.msfdr_2smix {
        Some(m) => log_fit_ok("MSFDR 2smix", m),
        None => log::info!("DF MSFDR 2smix: absent"),
    }

    // 1D. Score Base Rank 1
    let base_res = score_base_rank1(&new_features, work, Some(engines), settings, gates, db);

    // 1E. Write Method Outputs
    write_base_method_outputs(&mut new_features, &base_res, settings, gates);

    // 1F. Scrub Rank!=1
    scrub_non_rank1_df_outputs(&mut new_features);

    // 1G. Finalize Base Q-Values
    finalize_base_q_values(&mut new_features, &base_res.workset, settings, db);

    // 2. Apply Layer 2 Physical Rescue
    let rescue_res = apply_physical_rescue(&mut new_features, settings, db);
    if settings.physical_rescue.ims_enabled {
        log::info!(
			"DF Layer 2: mode={:?} (anchors={}/{}) JointRel={:.4} (RT={:.4}, IMS={:.4}) FailClosed={}",
			settings.physical_rescue.mode,
			rescue_res.anchor_count_after_filters,
			rescue_res.anchor_count_total,
			rescue_res.joint_reliability,
			rescue_res.rt_reliability,
			rescue_res.ims_reliability,
			rescue_res.fail_closed
		);
    } else {
        log::info!(
			"DF Layer 2: mode={:?} (anchors={}/{}) JointRel={:.4} (RT-only; IMS disabled) FailClosed={}",
			settings.physical_rescue.mode,
			rescue_res.anchor_count_after_filters,
			rescue_res.anchor_count_total,
			rescue_res.joint_reliability,
			rescue_res.fail_closed
		);
    }

    // Optional: Log detailed run/charge diagnostics at DEBUG level to avoid spamming INFO
    log::debug!(
        "DF Layer 2 Details: rt_sigma={:?}, ims_sigma={:?}, dropped_runs={:?}, dropped_charges={:?}",
        rescue_res.rt_sigma_global,
        rescue_res.ims_sigma_global,
        rescue_res.dropped_runs,
        rescue_res.dropped_charge_bins
    );

    // 3. Apply Layer 3 Reproducibility & Agreement
    let repro_res = if rescue_res.fail_closed {
        clear_l3_stage_outputs(&mut new_features);
        log::warn!(
            "DF Layer 3 skipped: Layer 2 fail-closed; Layer 3 will not activate as the final stream."
        );
        ReproducibilityResult {
            enabled: false,
            fail_closed: true,
            ..Default::default()
        }
    } else {
        apply_bounded_repro_shift(&mut new_features, settings, db)
    };

    log::info!(
        "DF Layer 3: enabled={} eligible_proteins={} eligible_peptides={} anchor_peptides={} rescued_psms={} strong_unchanged={} too_weak_unrescued={} agree_mean={:.4} max_shift={:.4} fail_closed={}",
        repro_res.enabled,
        repro_res.n_rescue_eligible_proteins,
        repro_res.n_rescue_eligible_peptides,
        repro_res.n_anchor_peptides,
        repro_res.n_rescued_psms,
        repro_res.n_strong_unchanged_psms,
        repro_res.n_too_weak_unrescued_psms,
        repro_res.agreement_support_mean,
        repro_res.max_shift_applied,
        repro_res.fail_closed
    );

    // 4. Finalize the active PSM stream from the completed DF layers.
    let final_stream = finalize_df_psm_stream(&mut new_features, settings);
    validate_final_df_stream_contract(&new_features, final_stream);

    let stream_kind = match final_stream {
        FinalDfStream::Base => "base / p-value-native",
        FinalDfStream::Layer2 => "layer2 / pep-native",
        FinalDfStream::Layer3 => "layer3 / pep-native",
    };
    log::info!("DF final active stream: {}", stream_kind);

    new_features
}

pub fn calculate_q_values(
    psms: &[DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> Vec<DfFeature> {
    run_df_layers(psms, settings, db)
}

pub fn calculate_peptide_q_df(
    features: &mut [DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
    threshold: f32,
) -> (usize, usize) {
    // Peptide inference consumes the finalized active DF PSM stream.
    // PEP-native final streams use decoy_free_pep; p-value-native streams use
    // decoy_free_p_value. Peptide-level aggregation uses the best supporting PSM,
    // with only bounded support from additional strong observations. Repeated
    // spectra for the same peptide are treated as corroborating evidence, not as a
    // count-based selected-min penalty.
    let is_pep_native = features
        .iter()
        .find(|f| f.core.rank == 1)
        .map(|f| f.decoy_free_p_value.is_none())
        .unwrap_or(false);

    #[derive(Default)]
    struct PepEvidence {
        vals: Vec<f64>,
        is_entrapment: bool,
    }

    let mut peptide_evidence_map: FnvHashMap<String, PepEvidence> = FnvHashMap::default();
    let mut finite_psm_count = 0usize;

    for feat in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let val = if is_pep_native {
            feat.decoy_free_pep
        } else {
            feat.decoy_free_p_value
        };

        let v = match val {
            Some(x) if x.is_finite() => (x as f64).clamp(0.0, 1.0).max(1e-300),
            _ => continue,
        };

        finite_psm_count += 1;

        let peptide = db[feat.core.peptide_idx].to_string();
        let proteins = db[feat.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        let is_ent = is_entrapment_str(&proteins);

        let entry = peptide_evidence_map.entry(peptide).or_default();
        entry.vals.push(v);
        entry.is_entrapment |= is_ent;
    }

    log::debug!(
        "DF peptide inference pool: finite_rank1_psms={} unique_peptides={}",
        finite_psm_count,
        peptide_evidence_map.len()
    );

    let mut peptide_keys = Vec::with_capacity(peptide_evidence_map.len());
    let mut peptide_combined_vals = Vec::with_capacity(peptide_evidence_map.len());
    let mut is_ent_flags = Vec::with_capacity(peptide_evidence_map.len());

    for (peptide, mut ev) in peptide_evidence_map {
        ev.vals.retain(|v| v.is_finite());
        if ev.vals.is_empty() {
            continue;
        }

        ev.vals.sort_by(|a, b| a.total_cmp(b));

        let best = ev.vals[0].clamp(1e-300, 1.0);

        let combined = if is_pep_native {
            // PEP-native path:
            // Repeated spectra for the same peptide are corroborating evidence, not
            // independent penalties against the peptide. Use the best PSM-level PEP as
            // the peptide evidence, with only a bounded support bonus from additional
            // strong observations.
            let support_factor = if ev.vals.len() >= 2 {
                let second = ev.vals[1].clamp(1e-300, 1.0);
                if second <= 0.01 {
                    0.50
                } else if second <= 0.05 {
                    0.75
                } else {
                    1.00
                }
            } else {
                1.00
            };

            (best * support_factor).clamp(1e-300, 1.0)
        } else {
            // P-value-native path:
            // Do not apply the PEP support bonus to p-values. Instead combine the
            // peptide's PSM-level p-values with the configured p-value combiner.
            match settings.peptide_p_combine {
                PeptidePCombine::Fisher => {
                    stats::combine_fisher(&ev.vals).clamp(0.0, 1.0).max(1e-300)
                }
                PeptidePCombine::Cauchy => combine_cauchy(&ev.vals),
                PeptidePCombine::SidakMinP => {
                    let n = ev.vals.len() as f64;
                    (1.0 - (1.0 - best).powf(n)).clamp(0.0, 1.0).max(1e-300)
                }
                PeptidePCombine::Best => best,
            }
        };

        peptide_keys.push(peptide);
        peptide_combined_vals.push(combined);
        is_ent_flags.push(ev.is_entrapment);
    }

    let q_values = if is_pep_native {
        // PEP-native path: cumulative mean of peptide-level PEP-like values.
        let mut rows: Vec<(f64, usize)> = peptide_combined_vals
            .iter()
            .copied()
            .enumerate()
            .map(|(i, pep)| (pep, i))
            .collect();

        rows.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut q_sorted: Vec<f64> = Vec::with_capacity(rows.len());
        let mut cum = 0.0f64;

        for (k, &(pep, _)) in rows.iter().enumerate() {
            cum += pep.clamp(0.0, 1.0);
            q_sorted.push((cum / ((k + 1) as f64)).clamp(0.0, 1.0));
        }

        for i in (0..q_sorted.len().saturating_sub(1)).rev() {
            q_sorted[i] = q_sorted[i].min(q_sorted[i + 1]);
        }

        let mut out = vec![1.0f64; rows.len()];
        for (k, &(_, orig_idx)) in rows.iter().enumerate() {
            out[orig_idx] = q_sorted[k];
        }

        out
    } else {
        // P-value-native path: BH over configured-combined peptide-level p-values.
        crate::ml::stats::bh_q_value(&peptide_combined_vals)
    };

    let mut best_q: FnvHashMap<String, (f32, bool)> = FnvHashMap::default();

    for i in 0..peptide_keys.len() {
        best_q.insert(
            peptide_keys[i].clone(),
            (q_values[i] as f32, is_ent_flags[i]),
        );
    }

    for feat in features.iter_mut() {
        if feat.core.rank != 1 || feat.core.label != 1 {
            feat.decoy_free_peptide_q = None;
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

        let peptide = db[feat.core.peptide_idx].to_string();
        let q = best_q.get(&peptide).map(|v| v.0).unwrap_or(1.0);
        feat.decoy_free_peptide_q = Some(q);
    }

    let mut passing_total = 0usize;
    let mut passing_entrapments = 0usize;

    for &(q, is_ent) in best_q.values() {
        if q <= threshold {
            passing_total += 1;
            if is_ent {
                passing_entrapments += 1;
            }
        }
    }

    (passing_total, passing_entrapments)
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

// DF aggregation/write-back contract: rank1-only; rank!=1 must be None to prevent stale leakage.
pub fn calculate_protein_q_df(
    features: &mut [DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
) -> usize {
    // Protein inference consumes the peptide-passing pool derived from the
    // finalized active DF stream.
    // Layer 3 may change which final PSMs survive into peptide inference,
    // but it does not change the downstream protein-inference contract.
    // Base-only remains p-value-native; Layer 2 and Layer 3 final streams are
    // PEP-native.
    let is_pep_native = features
        .iter()
        .find(|f| f.core.rank == 1)
        .map(|f| f.decoy_free_p_value.is_none())
        .unwrap_or(false);

    let mut peptide_passing_psm_count = 0usize;

    // Protein -> (peptide -> best_evidence)
    let mut protein_peptide_map: FnvHashMap<String, FnvHashMap<String, f64>> =
        FnvHashMap::default();

    for feat in features.iter().filter(|f| {
        f.core.rank == 1
            && f.core.label == 1
            && f.decoy_free_peptide_q
                .map(|q| q <= settings.peptide_fdr)
                .unwrap_or(false)
    }) {
        peptide_passing_psm_count += 1;

        // Read the appropriate evidence stream based on the contract
        let val = if is_pep_native {
            feat.decoy_free_pep
        } else {
            feat.decoy_free_p_value
        };

        let v = match val {
            Some(x) => (x as f64).clamp(0.0, 1.0).max(1e-300),
            None => continue,
        };

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
            .and_modify(|prev| *prev = prev.min(v))
            .or_insert(v);
    }

    log::debug!(
        "DF protein inference pool: peptide_passing_psms={} proteins_with_evidence={}",
        peptide_passing_psm_count,
        protein_peptide_map.len()
    );

    // Combine unique-peptide evidence per protein into a single protein-level
    // evidence value from the finalized DF stream.
    let mut protein_keys: Vec<String> = Vec::new();
    let mut protein_combined_vals: Vec<f64> = Vec::new();

    for (key, peptide_map) in protein_peptide_map {
        let mut vals: Vec<f64> = peptide_map.values().copied().collect();
        if vals.len() < 2 {
            continue;
        }

        let combined = if is_pep_native {
            // - unique-only peptides already enforced above
            // - require at least 2 unique peptides
            // - protein evidence = second-best (2nd smallest) peptide PEP
            vals.sort_by(|a, b| a.total_cmp(b));
            vals[1].clamp(0.0, 1.0).max(1e-300)
        } else {
            // P-value native combiners over at least two unique peptides
            match settings.protein_p_combine {
                crate::input::ProteinPCombine::Fisher => {
                    stats::combine_fisher(&vals).clamp(0.0, 1.0).max(1e-300)
                }
                crate::input::ProteinPCombine::Cauchy => combine_cauchy(&vals),
                crate::input::ProteinPCombine::SidakMinP => combine_sidak_minp(&vals),
            }
        };

        protein_keys.push(key);
        protein_combined_vals.push(combined);
    }

    // If no proteins, write fail-closed (rank1-only by contract) and return 0.
    if protein_combined_vals.is_empty() {
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

    // Convert protein evidence -> protein q-values (DF-only output)
    let protein_q_values = if is_pep_native {
        // PEP-native path: cumulative mean of protein-level PEP / PEP-like values
        let mut rows: Vec<(f64, usize)> = protein_combined_vals
            .iter()
            .copied()
            .enumerate()
            .map(|(i, pep)| (pep, i))
            .collect();

        // Sort ascending by PEP (best/lowest first)
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut q_sorted: Vec<f64> = Vec::with_capacity(rows.len());
        let mut cum = 0.0f64;
        for (k, &(pep, _)) in rows.iter().enumerate() {
            cum += pep;
            q_sorted.push((cum / ((k + 1) as f64)).clamp(0.0, 1.0));
        }

        // Enforce monotone non-decreasing
        for i in (0..q_sorted.len().saturating_sub(1)).rev() {
            q_sorted[i] = q_sorted[i].min(q_sorted[i + 1]);
        }

        // Reconstruct original order
        let mut out = vec![1.0f64; rows.len()];
        for (k, &(_, orig_idx)) in rows.iter().enumerate() {
            out[orig_idx] = q_sorted[k];
        }
        out
    } else {
        // P-value native path: Storey / BH
        match settings.type_ {
            FdrType::Bh => stats::bh_q_value(&protein_combined_vals),
            FdrType::Storey => {
                let mut protein_p_ref: Vec<f64> = Vec::new();
                for (key, &p) in protein_keys.iter().zip(protein_combined_vals.iter()) {
                    if !is_contam_str(key) && !is_entrapment_str(key) && p.is_finite() {
                        protein_p_ref.push(p.clamp(0.0, 1.0).max(1e-300));
                    }
                }

                if protein_p_ref.len() < settings.min_storey_n {
                    stats::bh_q_value(&protein_combined_vals)
                } else {
                    match estimate_pi0_from_reference_grid(&protein_p_ref, settings) {
                        Some(pi0) => storey_q_value_with_pi0(&protein_combined_vals, pi0, settings),
                        None => stats::bh_q_value(&protein_combined_vals),
                    }
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
