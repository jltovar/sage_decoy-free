/* =============================================================================
Decoy-Free (DF) FDR Logic Contract

This module implements Decoy-Free FDR scoring, optional post-model evidence
updates, and peptide/protein-level DF aggregation. The central invariant is that
`decoy_free_*` always represents the current last-good active PSM stream.

1) Rank scope
-------------
DF PSM-level outputs are defined only for rank==1 PSMs. For rank!=1 rows, all
DF final, stage-specific, and method-specific fields must be scrubbed to None.
This prevents stale values from lower-ranked candidates from leaking into TSV
output, peptide inference, protein inference, or diagnostics.

2) Final-output boundary
------------------------
The selected active DF answer is represented only by:

  decoy_free_p_value
  decoy_free_pep
  decoy_free_score
  decoy_free_q_value
  decoy_free_peptide_q
  decoy_free_protein_q

All other DF-family fields are audit snapshots, candidate buffers, or
method-specific diagnostic streams. They must not be used to infer the final
active stream.

3) Active-stream pipeline
-------------------------
The active-stream order is:

  base model fit
  -> optional RT confidence adjustment
  -> optional IMS confidence adjustment
  -> optional peptide reproducibility rescue
  -> optional protein reproducibility rescue

The mandatory base model creates the first valid active stream. Optional stages
may replace the active stream only after producing a finite, validated,
stage-specific update. If an optional stage fails locally or produces no
productive update, the previous active stream must remain unchanged.

4) Failure semantics
--------------------
The full DF run fails closed only if the mandatory base DF model cannot produce
a valid rank-1 active stream.

RT, IMS, peptide rescue, and protein rescue are optional nonfatal stages. Their
local failure must never erase, invalidate, or reinitialize the existing
`decoy_free_*` active stream. A local optional-stage failure means: keep the
last-good active stream and continue.

5) Evidence-stream semantics
----------------------------
Base DF experts may be p-value-native or PEP-native.

- Moments / MLE / LowerOrder:
    p_* fields are fitted-null tail p-value streams.
    pep_* fields are calibrated local-FDR/PEP-like streams derived from those
    p-value streams.

- MSFDR variants and Nokoi:
    p_* fields are fitted or empirical null-survival p-like streams.
    pep_* fields are calibrated PEP-like streams derived from those p-like
    streams. Raw classifier or mixture posteriors are not final calibrated PEPs
    unless explicitly validated.

- Ensemble:
    combines expert PEP-like streams into an empirical consensus PEP-like
    stream. It should be treated as operational PEP-like evidence, not as a
    formally calibrated posterior unless externally validated.

Physical RT/IMS adjustments and reproducibility rescue operate on PEP-like
confidence evidence and update the active stream only through validated stage
wrappers.

6) Q-value semantics
--------------------
For PEP-native active streams, q-values are cumulative means of PEP after
best-first sorting by the stream’s quality key, followed by monotonic
correction.

For p-value-native active streams, q-values are computed from the active p-value
stream using the configured p-value FDR procedure.

PEP-native streams must not use BH or Storey directly unless they have first
been converted into a valid p-value stream.

7) Peptide/protein inference
----------------------------
Peptide and protein inference consume only the finalized active `decoy_free_*`
stream.

For PEP-native peptide inference, peptide evidence is derived from the best
supporting PSM-level PEP with bounded support from additional strong PSMs.
Repeated spectra for the same peptide are corroborating evidence and must not
be penalized with a count-based selected-min correction.

For p-value-native peptide/protein inference, method-appropriate p-value
combination rules may be used only when the active stream is genuinely
p-value-native.

8) Stage snapshots
------------------
`decoy_free_*_base`, `decoy_free_*_rt`, `decoy_free_*_ims`,
`decoy_free_*_peptide_rescue`, and `decoy_free_*_protein_rescue` are audit
snapshots of the active stream after a stage is successfully applied. They must
not be used to infer control flow. Legacy internal L2/L3 candidate fields have
been entirely removed in favor of direct active-stream tracking.

9) Sorting keys
---------------
Whenever this module says “best-first,” the sorting key must be explicit and
stage-appropriate.

Examples:
- PEP-native q-values sort by a defined quality score, or by increasing PEP if
  no independent quality score is available.
- LowerOrder uses the configured LO quality key.
- Grenander calibration sorts by the p-like stream being calibrated.

Never rely on an implicit or ambiguous “score” definition.

============================================================================= */

use crate::database::IndexedDatabase;
use crate::input::LoStratify;
use crate::input::{
    BoundedAuxUpdateSpace, DartNullRtModel, DartTrueRtModel, EnsemblePCombiner,
    EnsemblePepCombiner, FdrSettings, FinalEvidenceSpace, HierarchicalReportingMode, JointMode,
    LoTevTransform, ModelFit, PCombineCalibrationMode, PeptidePCombine, PhysicalAnchorMode,
    ProteinPCombine, QCovariate, QMethod,
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
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Rank1Computed {
    idx: usize,

    // per-method p's
    p_mom: f64,
    p_mle: f64,
    p_lo: f64,

    // MSFDR family p-value streams.
    p_msfdr: Option<f64>, // seeded MSFDR fitted-null tail p-like stream
    p_1smix: Option<f64>, // 1SMix I1 survival p-like stream
    p_2smix: Option<f64>, // 2SMix I1 survival p-like stream

    // MSFDR family native posterior-error streams.
    pep_msfdr: Option<f64>,
    pep_1smix: Option<f64>,
    pep_2smix: Option<f64>,

    p_nokoi: Option<f64>,
    pep_nokoi: Option<f64>,

    // final DF p output; companion PEP streams are computed/stored separately.
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

        // Keep values paired with their original weights while filtering. Filtering
        // `valid_peps` first would shift every weight after a non-finite expert.
        EnsemblePepCombiner::WeightedMean => weighted_mean(peps, weights)
            .unwrap_or(1.0)
            .clamp(1e-300, 1.0),

        EnsemblePepCombiner::WeightedMedian => weighted_median(peps, weights)
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
// 3B) DF numeric boundary helpers
// -----------------------------------------------------------------------------
//
// Storage policy:
//   - DF p-values are stored as f64 so MSFDR values down to 1e-300 survive.
//   - Do not globally floor p-values at 1e-15.
//
// Transform policy:
//   - p-value log-space transforms may use values down to 1e-300.
//   - PEP/logit transforms must not see 0.0 or 1.0.

const DF_PVALUE_FLOOR: f64 = 1.0e-300;
const DF_PEP_FLOOR: f64 = 1.0e-15;
const DF_PEP_CEIL: f64 = 1.0 - 1.0e-15;

#[inline(always)]
fn finite_df_p_value(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(DF_PVALUE_FLOOR, 1.0)
    } else {
        1.0
    }
}

#[inline(always)]
fn finite_df_probability_for_logit(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(DF_PEP_FLOOR, DF_PEP_CEIL)
    } else {
        1.0
    }
}

// -----------------------------------------------------------------------------
// 4) Feature field helpers (tiny setters/getters for DF streams)
// -----------------------------------------------------------------------------

#[inline(always)]
fn set_df_q_value(psm: &mut DfFeature, q: f64) {
    psm.decoy_free_q_value = Some(finite_df_p_value(q));
}

#[inline(always)]
fn df_q_value(psm: &DfFeature) -> f64 {
    psm.decoy_free_q_value.unwrap_or(1.0)
}

#[inline(always)]
fn df_score_from_p_value(p: f64) -> f64 {
    -10.0 * finite_df_p_value(p).log10()
}

#[inline(always)]
fn df_score_from_active(active: ActiveEvidenceSpace, p_value: f64, pep: f64) -> f64 {
    match active {
        ActiveEvidenceSpace::PValue => df_score_from_p_value(p_value),
        ActiveEvidenceSpace::Pep => df_score_from_pep(pep),
    }
}

#[inline(always)]
fn set_df_evidence_pair(psm: &mut DfFeature, active: ActiveEvidenceSpace, p_value: f64, pep: f64) {
    let p_value = finite_df_p_value(p_value);
    let pep = finite_df_probability_for_logit(pep);

    psm.decoy_free_p_value = Some(p_value);
    psm.decoy_free_pep = Some(pep);
    psm.decoy_free_score = Some(df_score_from_active(active, p_value, pep));
}

// -----------------------------------------------------------------------------
// 5) Canonical evidence accessors
// -----------------------------------------------------------------------------
//
// IMPORTANT:
// - `tev(...)` remains the raw hyperscore accessor used by existing non-LO DF
//   code paths.
// - LowerOrder does not derive TEV from raw hyperscore downstream.
// - LowerOrder consumes raw spectrum-local components computed upstream
//   during scoring while the full per-spectrum candidate hyperscore distribution
//   is still available:
//
//       local p_tail = P(local spectrum null >= observed hyperscore)
//       n_candidates = number of scored candidates for the spectrum
//
//   DF constructs the LO E-value exactly once:
//
//       E = local p_tail
//         * n_candidates.powf(lo_evalue_candidate_count_power)
//         * lo_evalue_scale
//
//   The final LO TEV score is selected explicitly by `lo_tev_transform`:
//
//       neg_log_e              => TEV = -ln(E)
//       log_1000_over_e        => TEV = ln(1000 / E)
//       scaled_log_1000_over_e => TEV = 0.02 * ln(1000 / E)
//
//   `neg_log_e` is the default canonical uncompressed LO scale.
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
fn lo_spectrum_tail_p(f: &DfFeature) -> Option<f64> {
    let p = f.core.lo_spectrum_tail_p;

    if p.is_finite() && p > 0.0 {
        Some(p.clamp(1e-300, 1.0))
    } else {
        None
    }
}

#[inline(always)]
fn lo_spectrum_candidate_count(f: &DfFeature) -> Option<f64> {
    let n = f.core.lo_spectrum_candidate_count as f64;

    if n.is_finite() && n >= 1.0 {
        Some(n)
    } else {
        None
    }
}

#[inline(always)]
fn lo_e_value_from_tail_and_candidates(
    tail_p: f64,
    candidate_count: f64,
    candidate_count_power: f64,
    evalue_scale: f64,
) -> Option<f64> {
    if !tail_p.is_finite()
        || tail_p <= 0.0
        || !candidate_count.is_finite()
        || candidate_count < 1.0
        || !candidate_count_power.is_finite()
        || !evalue_scale.is_finite()
        || evalue_scale <= 0.0
    {
        return None;
    }

    let e = tail_p.clamp(1e-300, 1.0)
        * candidate_count.powf(candidate_count_power.clamp(0.0, 1.0))
        * evalue_scale.clamp(1e-6, 1e6);

    if e.is_finite() && e > 0.0 {
        Some(e.clamp(1e-300, 1e300))
    } else {
        None
    }
}

#[inline(always)]
fn lo_constructed_e_value(f: &DfFeature, settings: &FdrSettings) -> Option<f64> {
    let tail_p = lo_spectrum_tail_p(f)?;
    let candidate_count = lo_spectrum_candidate_count(f)?;

    lo_e_value_from_tail_and_candidates(
        tail_p,
        candidate_count,
        settings.lo_evalue_candidate_count_power,
        settings.lo_evalue_scale,
    )
}

#[derive(Clone, Debug)]
struct LoTevByIndex {
    by_index: Vec<Option<f64>>,
    valid: usize,
    invalid: usize,
}

const LO_TEV_REFERENCE_EVALUE: f64 = 1000.0;
const LO_HISTORICAL_TEV_SCALE: f64 = 0.02;

#[inline(always)]
fn lo_tev_from_e_value(e_value: f64, transform: LoTevTransform) -> Option<f64> {
    if !e_value.is_finite() || e_value <= 0.0 {
        return None;
    }

    let e = e_value.clamp(1e-300, 1e300);

    let tev = match transform {
        LoTevTransform::NegLogE => -e.ln(),

        LoTevTransform::Log1000OverE => LO_TEV_REFERENCE_EVALUE.ln() - e.ln(),

        LoTevTransform::ScaledLog1000OverE => {
            LO_HISTORICAL_TEV_SCALE * (LO_TEV_REFERENCE_EVALUE.ln() - e.ln())
        }
    };

    tev.is_finite().then_some(tev)
}

fn build_lo_tev_from_spectrum_tail_components(
    features: &[DfFeature],
    settings: &FdrSettings,
    _pool: &RankNullPool,
) -> LoTevByIndex {
    let mut by_index = vec![None; features.len()];
    let mut valid = 0usize;
    let mut invalid = 0usize;

    for (idx, f) in features.iter().enumerate() {
        if f.core.rank < 1 {
            invalid += 1;
            continue;
        }

        let Some(e_value) = lo_constructed_e_value(f, settings) else {
            invalid += 1;
            continue;
        };

        let Some(x_lo) =
            lo_tev_from_e_value(e_value.clamp(1e-300, 1e300), settings.lo_tev_transform)
        else {
            invalid += 1;
            continue;
        };

        by_index[idx] = Some(x_lo);
        valid += 1;
    }

    log::info!(
        "LO TEV diagnostics: valid={} invalid={} source=spectrum_local_tail_components candidate_count_power={:.3} evalue_scale={:.3} tev_transform={:?}",
        valid,
        invalid,
        settings.lo_evalue_candidate_count_power,
        settings.lo_evalue_scale,
        settings.lo_tev_transform
    );

    LoTevByIndex {
        by_index,
        valid,
        invalid,
    }
}

// -----------------------------------------------------------------------------
// 6) DF reset helper (critical): clear all DF + per-method outputs
// -----------------------------------------------------------------------------

#[inline(always)]
fn clear_all_df_outputs(psm: &mut DfFeature, fail_closed_rank1: bool) {
    // Clear the active final DF stream.
    psm.decoy_free_p_value = None;
    psm.decoy_free_pep = None;
    psm.decoy_free_score = None;
    psm.decoy_free_q_value = None;
    psm.decoy_free_peptide_q = None;
    psm.decoy_free_protein_q = None;
    psm.decoy_free_protein_supported_peptide = None;
    psm.decoy_free_peptide_supported_psm = None;
    psm.rt_residual = None;
    psm.abs_rt_residual = None;
    psm.rt_z = None;
    psm.rt_within_1sigma = None;
    psm.rt_within_2sigma = None;
    psm.rt_within_3sigma = None;

    psm.ims_residual = None;
    psm.abs_ims_residual = None;
    psm.ims_z = None;
    psm.ims_within_1sigma = None;
    psm.ims_within_2sigma = None;
    psm.ims_within_3sigma = None;

    psm.physical_rescue_source = None;
    psm.rescued_by_rt = None;
    psm.rescued_by_ims = None;
    psm.rescued_by_recurrence = None;

    psm.rt_local_z = None;
    psm.rt_local_outlier = None;
    psm.rt_training_eligible = None;

    // Clear the frozen base-stage audit snapshot.
    psm.decoy_free_p_value_base = None;
    psm.decoy_free_pep_base = None;
    psm.decoy_free_score_base = None;
    psm.decoy_free_q_base = None;

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

    // If the mandatory base DF model fails, rank-1 rows may be populated with
    // conservative fail-closed defaults. Optional-stage local failures must not call
    // this path; they preserve the previous active stream instead.
    if fail_closed_rank1 && psm.core.rank == 1 {
        psm.decoy_free_p_value = Some(1.0);
        psm.decoy_free_pep = Some(1.0);
        psm.decoy_free_score = Some(0.0);
        psm.decoy_free_q_value = Some(1.0);
        psm.decoy_free_peptide_q = Some(1.0);
        psm.decoy_free_protein_q = Some(1.0);
        psm.decoy_free_protein_supported_peptide = Some(false);
        psm.decoy_free_peptide_supported_psm = Some(false);
    }
}

// -----------------------------------------------------------------------------
// 7) DF rank-order score helpers
// -----------------------------------------------------------------------------

#[inline(always)]
fn df_rank_score(f: &DfFeature) -> f64 {
    tev(f).unwrap_or_else(|| f.decoy_free_score.unwrap_or(0.0) as f64)
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

/// Return the single protein hypothesis assigned to a Decoy-Free feature.
///
/// Once group annotation has run, a slash-delimited indistinguishable group is
/// one hypothesis, while a semicolon-delimited assignment remains ambiguous and
/// is excluded. If annotation has not run, fall back to the historical
/// unique-protein rule so direct callers retain fail-safe behavior.
#[inline]
fn df_inferred_protein_key_for_feature<'a>(
    f: &'a DfFeature,
    db: &IndexedDatabase,
) -> Option<Cow<'a, str>> {
    match (f.protein_groups.as_deref(), f.num_protein_groups) {
        (Some(group), 1) if !group.is_empty() => Some(Cow::Borrowed(group)),
        (Some(_), _) => None,
        (None, _) => df_unique_protein_key_for_feature(f, db).map(Cow::Owned),
    }
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
    let peptide_fdr = peptide_fdr as f64;
    let protein_fdr = protein_fdr as f64;
    let mut counts = EntrapmentCounts::default();
    let mut peptide_set: FnvHashSet<String> = FnvHashSet::default();
    let mut protein_set: FnvHashSet<String> = FnvHashSet::default();

    for feat in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let peptide = &db[feat.core.peptide_idx];
        let raw_protein_key = peptide.proteins(&db.decoy_tag, db.generate_decoys);
        let is_entrapment_peptide = is_entrapment_str(&raw_protein_key);

        if is_entrapment_peptide && feat.decoy_free_q_value.unwrap_or(1.0) <= peptide_fdr {
            counts.psms += 1;
        }

        if is_entrapment_peptide && feat.decoy_free_peptide_q.unwrap_or(1.0) <= peptide_fdr {
            peptide_set.insert(peptide.to_string());
        }

        if feat.decoy_free_protein_q.unwrap_or(1.0) <= protein_fdr {
            if let Some(protein_key) = df_inferred_protein_key_for_feature(feat, db) {
                if is_entrapment_str(protein_key.as_ref()) {
                    protein_set.insert(protein_key.into_owned());
                }
            }
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

fn summarize_q(label: &str, qs_in: impl Iterator<Item = f64>) {
    let mut qs: Vec<f64> = qs_in.filter(|q| q.is_finite()).collect();

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
// 11B) Final evidence-space helpers
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveEvidenceSpace {
    PValue,
    Pep,
}

#[inline]
fn active_evidence_space(settings: &FdrSettings) -> ActiveEvidenceSpace {
    match settings.final_evidence_space {
        FinalEvidenceSpace::PValue => ActiveEvidenceSpace::PValue,
        FinalEvidenceSpace::Pep => ActiveEvidenceSpace::Pep,
        FinalEvidenceSpace::Auto => match settings.model_fit {
            ModelFit::Moments | ModelFit::Mle | ModelFit::LowerOrder => ActiveEvidenceSpace::PValue,

            ModelFit::Msfdr | ModelFit::Msfdr1Smix | ModelFit::Msfdr2Smix | ModelFit::Nokoi => {
                ActiveEvidenceSpace::PValue
            }

            ModelFit::Ensemble => ActiveEvidenceSpace::PValue,
        },
    }
}

#[inline]
fn effective_psm_q_method(settings: &FdrSettings) -> QMethod {
    match settings.psm_q_method {
        QMethod::Auto => QMethod::Storey,
        method => method,
    }
}

#[inline]
fn effective_peptide_q_method(settings: &FdrSettings) -> QMethod {
    match settings.peptide_q_method {
        QMethod::Auto => effective_psm_q_method(settings),
        method => method,
    }
}

#[inline]
fn effective_protein_q_method(settings: &FdrSettings) -> QMethod {
    match settings.protein_q_method {
        QMethod::Auto => effective_psm_q_method(settings),
        method => method,
    }
}

#[inline]
fn finite_sorted(values: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    out.sort_by(|a, b| a.total_cmp(b));
    out
}

#[inline]
fn quantile_from_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }

    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[inline]
fn canonical_f64_bits(x: f64) -> u64 {
    if x == 0.0 {
        0.0f64.to_bits()
    } else {
        x.to_bits()
    }
}

fn n_unique_finite_values(values: &[f64]) -> usize {
    let mut seen: FnvHashSet<u64> = FnvHashSet::default();

    for &v in values {
        if v.is_finite() {
            seen.insert(canonical_f64_bits(v));
        }
    }

    seen.len()
}

fn most_common_finite_values(values: &[f64], n: usize) -> String {
    let mut counts: HashMap<u64, (f64, usize)> = HashMap::new();

    for &v in values {
        if !v.is_finite() {
            continue;
        }

        let key = canonical_f64_bits(v);
        counts
            .entry(key)
            .and_modify(|(_, c)| *c += 1)
            .or_insert((v, 1));
    }

    let mut rows: Vec<(f64, usize)> = counts.into_values().collect();

    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));

    rows.into_iter()
        .take(n)
        .map(|(v, c)| format!("{:.6e}x{}", v, c))
        .collect::<Vec<_>>()
        .join(", ")
}

fn log_peptide_q_diagnostics(
    peptide_combined_vals: &[f64],
    peptide_ref_vals: &[f64],
    q_report: &QValueComputation,
    settings: &FdrSettings,
    is_pep_native: bool,
) {
    let input_sorted = finite_sorted(peptide_combined_vals);
    let output_sorted = finite_sorted(&q_report.q_values);

    if input_sorted.is_empty() || output_sorted.is_empty() {
        log::warn!(
            "DF peptide q diagnostics: empty input/output; n_input={} n_output={}",
            peptide_combined_vals.len(),
            q_report.q_values.len()
        );
        return;
    }

    let effective_q_method = if is_pep_native {
        QMethod::Cummean
    } else {
        q_report.effective_method
    };

    let min_input = input_sorted[0];
    let min_output_q = output_sorted[0];

    let pi0_peptide = if is_pep_native { None } else { q_report.pi0 };

    let expected_min_q = pi0_peptide
        .map(|pi0| (pi0 * min_input * peptide_combined_vals.len() as f64).clamp(0.0, 1.0));

    log::info!(
        concat!(
            "DF peptide q diagnostics: ",
            "model_fit={:?} final_evidence_space={:?} active_space={:?} ",
            "peptide_q_method_requested={:?} peptide_q_method_effective={:?} peptide_q_method_actual={} fallback_reason={} ",
            "peptide_p_combine={:?} ",
            "n_peptides={} n_reference_peptides={} ",
            "n_unique_peptide_input_values={} n_unique_output_peptide_q={} ",
            "min_input_value={:.6e} p001_input_value={:.6e} p01_input_value={:.6e} median_input_value={:.6e} ",
            "pi0_peptide={} expected_min_q={} min_output_peptide_q={:.6e} ",
            "most_common_input_values=[{}] most_common_output_q_values=[{}]"
        ),
        settings.model_fit,
        settings.final_evidence_space,
        active_evidence_space(settings),
        q_report.requested_method,
		effective_q_method,
		q_report.actual_method,
		q_report.fallback_reason.unwrap_or("none"),
        settings.peptide_p_combine,
        peptide_combined_vals.len(),
        peptide_ref_vals.len(),
        n_unique_finite_values(peptide_combined_vals),
        n_unique_finite_values(&q_report.q_values),
        min_input,
        quantile_from_sorted(&input_sorted, 0.001),
        quantile_from_sorted(&input_sorted, 0.01),
        quantile_from_sorted(&input_sorted, 0.50),
        pi0_peptide
            .map(|v| format!("{:.6e}", v))
            .unwrap_or_else(|| "NA".to_string()),
        expected_min_q
            .map(|v| format!("{:.6e}", v))
            .unwrap_or_else(|| "NA".to_string()),
        min_output_q,
        most_common_finite_values(peptide_combined_vals, 10),
        most_common_finite_values(&q_report.q_values, 10),
    );
}

#[derive(Debug)]
struct QValueComputation {
    q_values: Vec<f64>,
    requested_method: QMethod,
    effective_method: QMethod,
    actual_method: &'static str,
    pi0: Option<f64>,
    fallback_reason: Option<&'static str>,
}

fn q_values_from_p_values_with_method_report(
    p_values: &[f64],
    p_ref: &[f64],
    settings: &FdrSettings,
    method: QMethod,
    level_name: &str,
) -> QValueComputation {
    let effective_method = match method {
        QMethod::Auto => effective_psm_q_method(settings),
        other => other,
    };

    match effective_method {
        QMethod::Bh => QValueComputation {
            q_values: stats::bh_q_value(p_values),
            requested_method: method,
            effective_method,
            actual_method: "BH",
            pi0: None,
            fallback_reason: None,
        },

        QMethod::By => QValueComputation {
            q_values: stats::by_q_value(p_values),
            requested_method: method,
            effective_method,
            actual_method: "BY",
            pi0: None,
            fallback_reason: None,
        },

        QMethod::Bky => QValueComputation {
            q_values: stats::bky_q_value(p_values, settings.bky_alpha),
            requested_method: method,
            effective_method,
            actual_method: "BKY",
            pi0: None,
            fallback_reason: None,
        },

        QMethod::Sfdr => QValueComputation {
            q_values: stats::sfdr_q_value(p_values, settings.sfdr_gamma),
            requested_method: method,
            effective_method,
            actual_method: "sFDR",
            pi0: None,
            fallback_reason: None,
        },

        QMethod::CovariateWeightedBh => {
            log::warn!(
                "DF {} q_method=covariate_weighted_bh reached generic p-value path; using BH. \
                 Level-specific covariate paths should intercept this first.",
                level_name
            );

            QValueComputation {
                q_values: stats::bh_q_value(p_values),
                requested_method: method,
                effective_method,
                actual_method: "BH",
                pi0: None,
                fallback_reason: Some("covariate_weighted_bh_without_level_covariates"),
            }
        }

        QMethod::Storey => {
            if p_ref.len() < settings.min_storey_n {
                log::warn!(
                    "DF {} Storey: reference count {} < min_storey_n {}; falling back to BH.",
                    level_name,
                    p_ref.len(),
                    settings.min_storey_n
                );

                QValueComputation {
                    q_values: stats::bh_q_value(p_values),
                    requested_method: method,
                    effective_method,
                    actual_method: "BH",
                    pi0: None,
                    fallback_reason: Some("storey_reference_count_below_min_storey_n"),
                }
            } else {
                match estimate_pi0_from_reference_grid(p_ref, settings) {
                    Some(pi0) => {
                        let storey_report = storey_q_value_with_pi0_report(p_values, pi0, settings);

                        QValueComputation {
                            q_values: storey_report.q_values,
                            requested_method: method,
                            effective_method,
                            actual_method: storey_report.actual_method,
                            pi0: Some(pi0),
                            fallback_reason: storey_report.fallback_reason,
                        }
                    }
                    None => {
                        log::warn!(
                            "DF {} Storey: failed to estimate pi0; falling back to BH.",
                            level_name
                        );

                        QValueComputation {
                            q_values: stats::bh_q_value(p_values),
                            requested_method: method,
                            effective_method,
                            actual_method: "BH",
                            pi0: None,
                            fallback_reason: Some("storey_pi0_estimation_failed"),
                        }
                    }
                }
            }
        }

        QMethod::Auto => unreachable!("QMethod::Auto should have been resolved above."),

        QMethod::Cummean => {
            log::warn!(
                "DF {} q_method=cummean requested on p-value-native evidence; using BH instead.",
                level_name
            );

            QValueComputation {
                q_values: stats::bh_q_value(p_values),
                requested_method: method,
                effective_method,
                actual_method: "BH",
                pi0: None,
                fallback_reason: Some("cummean_requested_on_p_value_native_evidence"),
            }
        }
    }
}

#[derive(Clone, Copy)]
enum QLevel {
    Psm,
    Peptide,
    Protein,
}

fn covariate_higher_is_better(cov: QCovariate) -> bool {
    !matches!(
        cov,
        QCovariate::DeltaBest
            | QCovariate::BestDeltaRtModel
            | QCovariate::ScoredCandidates
            | QCovariate::MissedCleavages
            | QCovariate::ProteinLength
            | QCovariate::ObservableProteinPeptides
    )
}

fn weights_from_covariates(
    values: &[Option<f64>],
    cov: QCovariate,
    bins: usize,
    strength: f64,
    level_name: &str,
) -> Option<Vec<f64>> {
    if matches!(cov, QCovariate::None) || values.is_empty() {
        return None;
    }

    let bins = bins.clamp(2, 20);
    let strength = strength.clamp(0.0, 5.0);

    if strength == 0.0 {
        return Some(vec![1.0; values.len()]);
    }

    let higher_is_better = covariate_higher_is_better(cov);

    let mut usable: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let x = (*v)?;
            x.is_finite().then_some((i, x))
        })
        .collect();

    if usable.len() < bins * 5 {
        log::warn!(
            "DF {} covariate q: covariate={:?} usable_values={} too small for bins={}; using BH.",
            level_name,
            cov,
            usable.len(),
            bins
        );
        return None;
    }

    usable.sort_by(|a, b| a.1.total_cmp(&b.1));

    let n = usable.len();
    let mut weights = vec![1.0f64; values.len()];

    for (rank0, &(idx, _)) in usable.iter().enumerate() {
        // Assign one shared weight per covariate bin.
        // This avoids hypothesis-specific continuous weights from tiny covariate
        // fluctuations and makes this closer to an IHW-style stratified mode.
        let bin_idx = ((rank0 * bins) / n).min(bins - 1);
        let frac = (bin_idx as f64 + 0.5) / bins as f64;

        let oriented = if higher_is_better { frac } else { 1.0 - frac };
        let centered = oriented - 0.5;

        weights[idx] = (strength * centered).exp();
    }

    log::info!(
        "DF {} covariate q: covariate={:?} higher_is_better={} usable={} bins={} strength={:.3} weighting=binned_exponential",
        level_name,
        cov,
        higher_is_better,
        usable.len(),
        bins,
        strength
    );

    Some(weights)
}

fn q_values_from_level_covariates(
    p_values: &[f64],
    p_ref: &[f64],
    cov_values: &[Option<f64>],
    settings: &FdrSettings,
    method: QMethod,
    level: QLevel,
    level_name: &str,
) -> QValueComputation {
    let effective_method = match method {
        QMethod::Auto => match level {
            QLevel::Psm => effective_psm_q_method(settings),
            QLevel::Peptide => effective_peptide_q_method(settings),
            QLevel::Protein => effective_protein_q_method(settings),
        },
        other => other,
    };

    if !matches!(effective_method, QMethod::CovariateWeightedBh) {
        return q_values_from_p_values_with_method_report(
            p_values, p_ref, settings, method, level_name,
        );
    }

    let (cov, bins, strength) = match level {
        QLevel::Psm => (
            settings.psm_q_covariate,
            settings.psm_q_covariate_bins,
            settings.psm_q_covariate_weight_strength,
        ),
        QLevel::Peptide => (
            settings.peptide_q_covariate,
            settings.peptide_q_covariate_bins,
            settings.peptide_q_covariate_weight_strength,
        ),
        QLevel::Protein => (
            settings.protein_q_covariate,
            settings.protein_q_covariate_bins,
            settings.protein_q_covariate_weight_strength,
        ),
    };

    let Some(weights) = weights_from_covariates(cov_values, cov, bins, strength, level_name) else {
        return QValueComputation {
            q_values: stats::bh_q_value(p_values),
            requested_method: method,
            effective_method,
            actual_method: "BH",
            pi0: None,
            fallback_reason: Some("covariate_weighting_unavailable"),
        };
    };

    QValueComputation {
        q_values: stats::weighted_bh_q_value(p_values, &weights),
        requested_method: method,
        effective_method,
        actual_method: "WeightedBH",
        pi0: None,
        fallback_reason: None,
    }
}

fn psm_covariate_value(f: &DfFeature, cov: QCovariate) -> Option<f64> {
    let v = match cov {
        QCovariate::None => return None,
        QCovariate::Hyperscore => f.core.hyperscore as f64,
        QCovariate::DeltaNext => f.core.delta_next as f64,
        QCovariate::DeltaBest => f.core.delta_best as f64,
        QCovariate::MatchedPeaks => f.core.matched_peaks as f64,
        QCovariate::LongestB => f.core.longest_b as f64,
        QCovariate::LongestY => f.core.longest_y as f64,
        QCovariate::LongestYPct => {
            let denom = (f.core.peptide_len as f64 - 1.0).max(1.0);
            f.core.longest_y as f64 / denom
        }
        QCovariate::MatchedIntensityPct => f.core.matched_intensity_pct as f64,
        QCovariate::ScoredCandidates => f.core.scored_candidates as f64,
        QCovariate::Ms2Intensity => {
            let x = f.core.ms2_intensity as f64;
            if x > 0.0 {
                x.ln_1p()
            } else {
                x
            }
        }
        QCovariate::PeptideLen => f.core.peptide_len as f64,
        QCovariate::Charge => f.core.charge as f64,
        QCovariate::MissedCleavages => f.core.missed_cleavages as f64,

        _ => return None,
    };

    v.is_finite().then_some(v)
}

fn q_values_from_p_values_with_method(
    p_values: &[f64],
    p_ref: &[f64],
    settings: &FdrSettings,
    method: QMethod,
    level_name: &str,
) -> Vec<f64> {
    q_values_from_p_values_with_method_report(p_values, p_ref, settings, method, level_name)
        .q_values
}

#[inline]
fn combine_p_values_for_ensemble(p_values: &[f64], settings: &FdrSettings) -> f64 {
    if p_values.is_empty() {
        return 1.0;
    }

    match settings.ensemble_p_combiner {
        EnsemblePCombiner::Fisher => stats::combine_fisher(p_values).clamp(0.0, 1.0).max(1e-300),

        EnsemblePCombiner::Cauchy => {
            let p = combine_cauchy(p_values);
            let penalty = settings.ensemble_cauchy_penalty;
            (p * penalty).clamp(0.0, 1.0).max(1e-300)
        }

        EnsemblePCombiner::SidakMinP => combine_sidak_minp(p_values),

        EnsemblePCombiner::Best => p_values
            .iter()
            .copied()
            .filter(|p| p.is_finite())
            .fold(1.0_f64, |a, b| a.min(b.clamp(0.0, 1.0).max(1e-300))),

        EnsemblePCombiner::SecondBest => combine_second_best_p(p_values),
    }
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

fn storey_q_value_with_pi0_report(
    p_values: &[f64],
    pi0: f64,
    settings: &FdrSettings,
) -> QValueComputation {
    let m = p_values.len();
    if m == 0 {
        return QValueComputation {
            q_values: Vec::new(),
            requested_method: QMethod::Storey,
            effective_method: QMethod::Storey,
            actual_method: "Storey",
            pi0: Some(pi0),
            fallback_reason: None,
        };
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
                return QValueComputation {
                    q_values: crate::ml::stats::bh_q_value(p_values),
                    requested_method: QMethod::Storey,
                    effective_method: QMethod::Storey,
                    actual_method: "BH",
                    pi0: Some(pi0),
                    fallback_reason: Some("storey_degenerate_q_vector"),
                };
            }
            crate::input::StoreyDegeneracyFallback::None => {}
        }
    }

    QValueComputation {
        q_values: out,
        requested_method: QMethod::Storey,
        effective_method: QMethod::Storey,
        actual_method: "Storey",
        pi0: Some(pi0),
        fallback_reason: None,
    }
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

fn recalibrate_companion_pep_from_active_p_values(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    context: &str,
) {
    let mut rows: Vec<(usize, f64)> = Vec::new();

    for (idx, f) in features.iter().enumerate() {
        if f.core.rank != 1 {
            continue;
        }

        let Some(p) = f.decoy_free_p_value else {
            continue;
        };

        let p = (p as f64).clamp(0.0, 1.0).max(1e-300);

        if p.is_finite() {
            rows.push((idx, p));
        }
    }

    if rows.is_empty() {
        log::warn!(
            "DF {}: companion PEP recalibration skipped because no rank-1 active p-values were available.",
            context
        );
        return;
    }

    let p_values: Vec<f64> = rows.iter().map(|&(_, p)| p).collect();

    let pi0 = estimate_pi0_from_reference_grid(&p_values, settings)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    let peps = grenander_pep_from_p(&p_values, pi0);

    if peps.len() != rows.len() {
        log::warn!(
            "DF {}: companion PEP recalibration skipped because p/PEP lengths mismatched: p_values={} peps={}.",
            context,
            rows.len(),
            peps.len()
        );
        return;
    }

    for ((idx, p), pep) in rows.into_iter().zip(peps.into_iter()) {
        set_df_evidence_pair(
            &mut features[idx],
            ActiveEvidenceSpace::PValue,
            p,
            pep.clamp(0.0, 1.0).max(1e-300),
        );
    }

    log::debug!(
        "DF {}: recalibrated companion PEP stream from {} active p-values using pi0={:.6}.",
        context,
        p_values.len(),
        pi0
    );
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

fn grenander_pep_from_reference_population(p_all: &[f64], is_ref: &[bool], pi0: f64) -> Vec<f64> {
    const EPS: f64 = 1e-300;

    if p_all.len() != is_ref.len() {
        log::warn!(
            "Grenander reference-population PEP calibration skipped because p/ref lengths differ: p_all={} is_ref={}.",
            p_all.len(),
            is_ref.len()
        );
        return vec![1.0; p_all.len()];
    }

    let mut ref_rows: Vec<(usize, f64)> = Vec::new();

    for (idx, (&p, &ref_ok)) in p_all.iter().zip(is_ref.iter()).enumerate() {
        if ref_ok && p.is_finite() {
            ref_rows.push((idx, p.clamp(EPS, 1.0)));
        }
    }

    if ref_rows.is_empty() {
        log::warn!(
            "Grenander reference-population PEP calibration skipped because reference population is empty."
        );
        return vec![1.0; p_all.len()];
    }

    let p_ref: Vec<f64> = ref_rows.iter().map(|&(_, p)| p).collect();
    let pep_ref = grenander_pep_from_p(&p_ref, pi0);

    if pep_ref.len() != ref_rows.len() {
        log::warn!(
            "Grenander reference-population PEP calibration skipped because calibrated length mismatched: ref_rows={} pep_ref={}.",
            ref_rows.len(),
            pep_ref.len()
        );
        return vec![1.0; p_all.len()];
    }

    let mut calibration_pairs: Vec<(f64, f64)> = p_ref
        .into_iter()
        .zip(pep_ref.into_iter())
        .filter(|(p, pep)| p.is_finite() && pep.is_finite())
        .map(|(p, pep)| (p.clamp(EPS, 1.0), pep.clamp(EPS, 1.0)))
        .collect();

    if calibration_pairs.is_empty() {
        return vec![1.0; p_all.len()];
    }

    calibration_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Collapse duplicate p-values conservatively by keeping the worst PEP.
    let mut collapsed: Vec<(f64, f64)> = Vec::new();
    for (p, pep) in calibration_pairs {
        if let Some(last) = collapsed.last_mut() {
            if last.0 == p {
                last.1 = last.1.max(pep);
                continue;
            }
        }
        collapsed.push((p, pep));
    }

    // Enforce monotone non-decreasing PEP with worsening p-value.
    let mut running_max = EPS;
    for (_, pep) in collapsed.iter_mut() {
        running_max = running_max.max(*pep);
        *pep = running_max.clamp(EPS, 1.0);
    }

    p_all
        .iter()
        .map(|&p| {
            if !p.is_finite() {
                return 1.0;
            }

            let p = p.clamp(EPS, 1.0);

            match collapsed.binary_search_by(|(p_ref, _)| p_ref.total_cmp(&p)) {
                Ok(pos) => collapsed[pos].1,
                Err(0) => collapsed[0].1,
                Err(pos) if pos >= collapsed.len() => collapsed[collapsed.len() - 1].1,
                Err(pos) => {
                    let (p_lo, pep_lo) = collapsed[pos - 1];
                    let (p_hi, pep_hi) = collapsed[pos];

                    if p_hi <= p_lo {
                        pep_hi.max(pep_lo).clamp(EPS, 1.0)
                    } else {
                        let w = ((p - p_lo) / (p_hi - p_lo)).clamp(0.0, 1.0);
                        let pep = pep_lo * (1.0 - w) + pep_hi * w;
                        pep.clamp(EPS, 1.0)
                    }
                }
            }
        })
        .collect()
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
// 16) Debug helpers: rank-1 composition and LowerOrder score diagnostics
// -----------------------------------------------------------------------------

#[inline]
fn log_f64_distribution(label: &str, values_in: &[f64]) {
    let mut values: Vec<f64> = values_in
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();

    if values.is_empty() {
        log::info!("{label}: n=0");
        return;
    }

    values.sort_by(|a, b| a.total_cmp(b));

    let q_at = |frac: f64| -> f64 {
        let idx = (frac.clamp(0.0, 1.0) * ((values.len() - 1) as f64)).round() as usize;
        values[idx.min(values.len() - 1)]
    };

    let n = values.len();
    let min = q_at(0.00);
    let p01 = q_at(0.01);
    let p05 = q_at(0.05);
    let p10 = q_at(0.10);
    let med = q_at(0.50);
    let p90 = q_at(0.90);
    let p95 = q_at(0.95);
    let p99 = q_at(0.99);
    let max = q_at(1.00);

    log::info!(
        "{}: n={} q=[min={:.6e},p01={:.6e},p05={:.6e},p10={:.6e},med={:.6e},p90={:.6e},p95={:.6e},p99={:.6e},max={:.6e}]",
        label,
        n,
        min,
        p01,
        p05,
        p10,
        med,
        p90,
        p95,
        p99,
        max
    );
}

fn log_lower_order_rank1_score_diagnostics(
    features: &[DfFeature],
    work: &WorkSet,
    db: &IndexedDatabase,
    tev_by_index: &[Option<f64>],
    settings: &FdrSettings,
) {
    let mut tev_all = Vec::new();
    let mut tev_label1 = Vec::new();
    let mut tev_ref = Vec::new();
    let mut tev_ent = Vec::new();
    let mut tev_cont = Vec::new();

    let mut tail_all = Vec::new();
    let mut tail_ref = Vec::new();
    let mut tail_ent = Vec::new();

    let mut cand_all = Vec::new();
    let mut cand_ref = Vec::new();
    let mut cand_ent = Vec::new();

    let mut eval_all = Vec::new();
    let mut eval_ref = Vec::new();
    let mut eval_ent = Vec::new();

    let mut missing_tev = 0usize;
    let mut missing_components = 0usize;

    for &idx in &work.rank1_indices {
        let f = &features[idx];

        let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        let is_ent = is_entrapment_str(&prot);
        let is_cont = is_contam_str(&prot);
        let is_ref = f.core.label == 1 && !is_ent && !is_cont;

        match tev_by_index.get(idx).copied().flatten() {
            Some(x) if x.is_finite() => {
                tev_all.push(x);

                if f.core.label == 1 {
                    tev_label1.push(x);
                }
                if is_ref {
                    tev_ref.push(x);
                }
                if is_ent {
                    tev_ent.push(x);
                }
                if is_cont {
                    tev_cont.push(x);
                }
            }
            _ => {
                missing_tev += 1;
            }
        }

        let tail_p = lo_spectrum_tail_p(f);
        let cand = lo_spectrum_candidate_count(f);
        let eval = lo_constructed_e_value(f, settings);

        match (tail_p, cand, eval) {
            (Some(tp), Some(cc), Some(ev))
                if tp.is_finite() && cc.is_finite() && ev.is_finite() =>
            {
                tail_all.push(tp);
                cand_all.push(cc);
                eval_all.push(ev);

                if is_ref {
                    tail_ref.push(tp);
                    cand_ref.push(cc);
                    eval_ref.push(ev);
                }

                if is_ent {
                    tail_ent.push(tp);
                    cand_ent.push(cc);
                    eval_ent.push(ev);
                }
            }
            _ => {
                missing_components += 1;
            }
        }
    }

    log::info!(
        "LO rank1 TEV/component diagnostics: missing_tev={} missing_components={} candidate_count_power={:.3} evalue_scale={:.3} tev_transform={:?}",
        missing_tev,
        missing_components,
        settings.lo_evalue_candidate_count_power,
        settings.lo_evalue_scale,
        settings.lo_tev_transform
    );

    log_f64_distribution("LO rank1 TEV all", &tev_all);
    log_f64_distribution("LO rank1 TEV label1", &tev_label1);
    log_f64_distribution(
        "LO rank1 TEV reference_target_noncontam_nonentrap",
        &tev_ref,
    );
    log_f64_distribution("LO rank1 TEV entrapment", &tev_ent);
    log_f64_distribution("LO rank1 TEV contaminant", &tev_cont);

    log_f64_distribution("LO rank1 tail_p all", &tail_all);
    log_f64_distribution(
        "LO rank1 tail_p reference_target_noncontam_nonentrap",
        &tail_ref,
    );
    log_f64_distribution("LO rank1 tail_p entrapment", &tail_ent);

    log_f64_distribution("LO rank1 candidate_count all", &cand_all);
    log_f64_distribution(
        "LO rank1 candidate_count reference_target_noncontam_nonentrap",
        &cand_ref,
    );
    log_f64_distribution("LO rank1 candidate_count entrapment", &cand_ent);

    log_f64_distribution("LO rank1 E_LO all", &eval_all);
    log_f64_distribution(
        "LO rank1 E_LO reference_target_noncontam_nonentrap",
        &eval_ref,
    );
    log_f64_distribution("LO rank1 E_LO entrapment", &eval_ent);
}

// Rank-1 composition summary.
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
#[derive(Clone, Copy, Debug)]
struct RankNullRow {
    feature_idx: usize,
    peptide_idx: u32,
    rank: u32,
    score: f64,
    charge: u8,
}

#[derive(Clone, Debug)]
struct RankNullSource {
    rows: Arc<[RankNullRow]>,
    rank1_scores_desc: Arc<[(u32, f64)]>,
}

impl RankNullSource {
    fn build(features: &[DfFeature], work: &WorkSet, settings: &FdrSettings) -> Self {
        let mut rank1_scores_desc: Vec<(u32, f64)> = work
            .rank1_indices
            .iter()
            .filter_map(|&idx| {
                let feature = &features[idx];
                Some((feature.core.peptide_idx.0, tev(feature)?))
            })
            .collect();
        rank1_scores_desc.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let rows: Vec<RankNullRow> = features
            .iter()
            .enumerate()
            .filter_map(|(feature_idx, feature)| {
                let rank = feature.core.rank;
                if rank < settings.min_null_rank || rank > settings.max_null_rank {
                    return None;
                }

                Some(RankNullRow {
                    feature_idx,
                    peptide_idx: feature.core.peptide_idx.0,
                    rank,
                    score: tev(feature)?,
                    charge: feature.core.charge,
                })
            })
            .collect();

        Self {
            rows: rows.into(),
            rank1_scores_desc: rank1_scores_desc.into(),
        }
    }
}

#[derive(Clone, Debug)]
struct RankNullPool {
    source: Arc<[RankNullRow]>,
    selected_rows: Vec<usize>,
}

impl RankNullPool {
    fn rows(&self) -> impl Iterator<Item = RankNullRow> + '_ {
        self.selected_rows.iter().map(|&idx| self.source[idx])
    }

    #[cfg(any(test, feature = "bench"))]
    fn len(&self) -> usize {
        self.selected_rows.len()
    }

    /// Return hyperscore values for pool members whose rank is in [min..=max].
    /// Note: ranks are the original per-PSM hit rank used to build the global pool.
    fn scores_in_window(&self, min: u32, max: u32) -> Vec<f64> {
        self.rows()
            .filter(|row| row.rank >= min && row.rank <= max)
            .map(|row| row.score)
            .collect()
    }

    fn indexed_fit_data_in_window(&self, min: u32, max: u32) -> Vec<(usize, u32, f64, u8)> {
        self.rows()
            .filter(|row| row.rank >= min && row.rank <= max)
            .map(|row| (row.feature_idx, row.rank, row.score, row.charge))
            .collect()
    }

    fn count_in_window(&self, min: u32, max: u32) -> usize {
        self.rows()
            .filter(|row| row.rank >= min && row.rank <= max)
            .count()
    }

    fn feature_indices(&self) -> Vec<usize> {
        self.rows().map(|row| row.feature_idx).collect()
    }
}

#[derive(Clone)]
struct Engines {
    mom_params: Option<(f64, f64)>,
    mle_params: Option<(f64, f64)>,

    lo_model: Option<LowerOrderModel>,
    lo_tev_by_index: Option<Arc<[Option<f64>]>>,

    msfdr_seeded: Option<MsfdrSeededModel>,
    msfdr_1smix: Option<Msfdr1SmixModel>,
    msfdr_2smix: Option<Msfdr2SmixModel>,

    nokoi_p_values: Option<Arc<Vec<f64>>>,
    nokoi_peps: Option<Arc<Vec<f64>>>,
}

// --- BUILD RANK-NULL POOL ---
fn build_rank_null_pool(
    source: &RankNullSource,
    settings: &FdrSettings,
    null_purification_factor: f64,
    label: &str,
) -> Option<RankNullPool> {
    let min_null_size = settings.min_null_size;
    let p_factor = null_purification_factor.clamp(0.0, 0.9);

    let purification_threshold = if source.rank1_scores_desc.len() >= 10 && p_factor > 0.0 {
        let top_k = ((source.rank1_scores_desc.len() as f64) * p_factor).round() as usize;
        let top_k = top_k.max(5).min(source.rank1_scores_desc.len());
        source.rank1_scores_desc[top_k - 1].1
    } else {
        f64::INFINITY
    };

    let purified_peptides: FnvHashSet<u32> = source
        .rank1_scores_desc
        .iter()
        .filter(|(_, score)| *score >= purification_threshold)
        .map(|(idx, _)| *idx)
        .collect();

    let mut selected_rows: Vec<usize> = source
        .rows
        .iter()
        .enumerate()
        .filter(|(_, row)| !purified_peptides.contains(&row.peptide_idx))
        .map(|(idx, _)| idx)
        .collect();

    if selected_rows.len() < min_null_size {
        log::warn!(
            "{label}: purified null too small with purification_factor={:.3}; falling back to unpurified null.",
            p_factor
        );
        selected_rows = (0..source.rows.len()).collect();
    }

    if selected_rows.len() < min_null_size {
        log::warn!(
            "{label}: null pool too small after fallback: n={} < min_null_size={}",
            selected_rows.len(),
            min_null_size
        );
        return None;
    }

    log::info!(
        "{label}: rank-null pool built with purification_factor={:.3}; rows={}",
        p_factor,
        selected_rows.len()
    );

    Some(RankNullPool {
        source: Arc::clone(&source.rows),
        selected_rows,
    })
}

#[cfg(feature = "bench")]
#[doc(hidden)]
pub struct NullPoolBenchmark {
    features: Vec<DfFeature>,
    work: WorkSet,
    settings: FdrSettings,
}

#[cfg(feature = "bench")]
impl NullPoolBenchmark {
    pub fn new(spectra: usize, ranks_per_spectrum: u32) -> Self {
        use crate::database::PeptideIx;
        use crate::input::{FdrMode, FdrOptions};
        use crate::scoring::{ExternalPsmFeatures, FeatureCore};

        assert!(ranks_per_spectrum >= 2);

        let mut features = Vec::with_capacity(spectra * ranks_per_spectrum as usize);
        for spectrum in 0..spectra {
            let spec_id = format!("controllerType=0 controllerNumber=1 scan={spectrum}");
            for rank in 1..=ranks_per_spectrum {
                let peptide = spectrum * ranks_per_spectrum as usize + rank as usize;
                features.push(
                    FeatureCore {
                        peptide_idx: PeptideIx(peptide as u32),
                        psm_id: features.len(),
                        peptide_len: 12,
                        spec_id: spec_id.clone(),
                        file_id: spectrum % 8,
                        rank,
                        label: 1,
                        expmass: 1_000.0,
                        calcmass: 1_000.0,
                        charge: 2 + (spectrum % 3) as u8,
                        rt: 0.5,
                        aligned_rt: 0.5,
                        predicted_rt: 0.5,
                        delta_rt_model: 0.0,
                        ims: 1.0,
                        predicted_ims: 1.0,
                        delta_ims_model: 0.0,
                        delta_mass: 0.0,
                        isotope_error: 0.0,
                        average_ppm: 0.0,
                        hyperscore: 100.0 - rank as f64 + (spectrum % 17) as f64 * 0.01,
                        delta_next: 1.0,
                        delta_best: rank.saturating_sub(1) as f64,
                        matched_peaks: 20,
                        longest_b: 5,
                        longest_y: 5,
                        longest_y_pct: 0.5,
                        missed_cleavages: 0,
                        matched_intensity_pct: 50.0,
                        scored_candidates: 100,
                        poisson_log10_p_value: -3.0,
                        lo_spectrum_tail_p: 0.01,
                        lo_spectrum_candidate_count: 100,
                        ms2_intensity: 1_000.0,
                        external_features: ExternalPsmFeatures::default(),
                        fragments: None,
                    }
                    .to_df(),
                );
            }
        }

        let mut options = FdrOptions::default();
        options.mode = Some(FdrMode::DecoyFree);
        options.model_fit = Some(ModelFit::Moments);
        options.min_null_rank = Some(2);
        options.max_null_rank = Some(ranks_per_spectrum);
        options.min_null_size = Some(10);
        let settings = FdrSettings::from(options);
        let work = WorkSet::build(&features);

        Self {
            features,
            work,
            settings,
        }
    }

    pub fn build_all(&self) -> usize {
        let source = RankNullSource::build(&self.features, &self.work, &self.settings);
        [0.25, 0.25, 0.15, 0.25, 0.20]
            .into_iter()
            .filter_map(|factor| build_rank_null_pool(&source, &self.settings, factor, "benchmark"))
            .map(|pool| pool.len())
            .sum()
    }
}

fn winsorize_scores_for_fit(scores: &[f64], lower_q: f64, upper_q: f64) -> Vec<f64> {
    let mut sorted: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();

    if sorted.is_empty() {
        return Vec::new();
    }

    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();
    let lower_q = lower_q.clamp(0.0, 1.0);
    let upper_q = upper_q.clamp(lower_q, 1.0);

    let lo_idx = ((n.saturating_sub(1) as f64) * lower_q).round() as usize;
    let hi_idx = ((n.saturating_sub(1) as f64) * upper_q).round() as usize;

    let lo = sorted[lo_idx.min(n - 1)];
    let hi = sorted[hi_idx.min(n - 1)];

    scores
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .map(|x| x.clamp(lo, hi))
        .collect()
}

/// Mean and variance of a standard Gumbel after quantile winsorization.
///
/// If `Z ~ Gumbel(0, 1)` and `W = clamp(Z, Q(lower_q), Q(upper_q))`, these
/// moments let us undo the location/scale bias introduced by clamping.  A
/// deterministic midpoint rule is used over the standard-Gumbel quantile
/// function.  This is evaluated once per model fit and is negligible next to
/// sorting the rank-null pool.
fn standard_gumbel_winsorized_moments(lower_q: f64, upper_q: f64) -> Option<(f64, f64)> {
    let lower_q = lower_q.clamp(0.0, 1.0);
    let upper_q = upper_q.clamp(lower_q, 1.0);

    if lower_q == 0.0 && upper_q == 1.0 {
        return Some((
            statrs::consts::EULER_MASCHERONI,
            std::f64::consts::PI.powi(2) / 6.0,
        ));
    }

    const GRID_SIZE: usize = 1 << 16;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for index in 0..GRID_SIZE {
        let probability = (index as f64 + 0.5) / GRID_SIZE as f64;
        let clipped_probability = probability.clamp(lower_q, upper_q);
        let quantile = -(-clipped_probability.ln()).ln();
        sum += quantile;
        sum_sq += quantile * quantile;
    }

    let n = GRID_SIZE as f64;
    let mean = sum / n;
    let variance = sum_sq / n - mean * mean;
    if mean.is_finite() && variance.is_finite() && variance > 0.0 {
        Some((mean, variance))
    } else {
        None
    }
}

/// Bias-corrected Gumbel method-of-moments fit for already winsorized scores.
///
/// For `X = mu + beta * Z`, winsorizing at fixed quantiles preserves the
/// location-scale form: `E[X_w] = mu + beta E[Z_w]` and
/// `Var[X_w] = beta^2 Var[Z_w]`.  Ordinary Gumbel moments applied directly to
/// `X_w` underestimate beta, especially with an upper clamp at 0.90.
fn fit_gumbel_winsorized_moments(
    winsorized_scores: &[f64],
    lower_q: f64,
    upper_q: f64,
) -> (f64, f64) {
    let finite: Vec<f64> = winsorized_scores
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.len() < 2 {
        return (f64::NAN, f64::NAN);
    }

    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let variance = finite
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / n;
    let Some((standard_mean, standard_variance)) =
        standard_gumbel_winsorized_moments(lower_q, upper_q)
    else {
        return (f64::NAN, f64::NAN);
    };

    if !variance.is_finite() || variance <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let beta = (variance / standard_variance).sqrt();
    let mu = mean - beta * standard_mean;
    if mu.is_finite() && beta.is_finite() && beta > 0.0 {
        (mu, beta)
    } else {
        (f64::NAN, f64::NAN)
    }
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
fn fit_msfdr_1smix(rank1_scores: &[f64], settings: &FdrSettings) -> Option<Msfdr1SmixModel> {
    Msfdr1SmixModel::fit_rank1(
        rank1_scores,
        settings.mix_em_max_iter,
        settings.mix_em_tol,
        (settings.msfdr1_pi_clamp_min, settings.msfdr1_pi_clamp_max),
        settings.msfdr1_bottom_frac_init,
        settings.msfdr1_top_frac_init,
    )
}

#[inline]
fn fit_msfdr_2smix(
    rank1_scores: &[f64],
    pooled_rank_scores: &[f64],
    settings: &FdrSettings,
) -> Option<Msfdr2SmixModel> {
    Msfdr2SmixModel::fit_top_two_pooled(
        rank1_scores,
        pooled_rank_scores,
        settings.mix_em_max_iter,
        settings.mix_em_tol,
        (settings.msfdr2_pi_clamp_min, settings.msfdr2_pi_clamp_max),
        settings.msfdr2_bottom_frac_init,
        settings.msfdr2_top_frac_init,
    )
}

// --- FIT/PREPARE ENGINES ---
fn fit_engines(
    features: &[DfFeature],
    work: &WorkSet,
    settings: &FdrSettings,
    gates: RunGates,
) -> Option<Engines> {
    let min_null_size = settings.min_null_size;
    let null_source = RankNullSource::build(features, work, settings);

    let moments_pool = if gates.run_mom {
        build_rank_null_pool(
            &null_source,
            settings,
            settings.moments_purification_factor,
            "Moments",
        )
    } else {
        None
    };

    let mle_pool = if gates.run_mle {
        build_rank_null_pool(
            &null_source,
            settings,
            settings.mle_purification_factor,
            "MLE",
        )
    } else {
        None
    };

    let lower_order_pool = if gates.run_lo {
        build_rank_null_pool(
            &null_source,
            settings,
            settings.lower_order_purification_factor,
            "LowerOrder",
        )
    } else {
        None
    };

    let msfdr_seeded_pool = if gates.run_msfdr_seeded {
        build_rank_null_pool(
            &null_source,
            settings,
            settings.msfdr_seeded_purification_factor,
            "MSFDR seeded",
        )
    } else {
        None
    };

    let nokoi_pool = if gates.run_nokoi {
        build_rank_null_pool(
            &null_source,
            settings,
            settings.nokoi_null_purification_factor,
            "Nokoi",
        )
    } else {
        None
    };

    let window_ok = |method: &str, window_min: u32, window_max: u32, count: usize| -> bool {
        if count < min_null_size {
            log::warn!(
                "{method}: null window [{window_min}..={window_max}] too small \
				 (n={count} < min_null_size={min_null_size}); expert will be unavailable. \
				 If this is the selected non-ensemble model, the selected-model guard will fail closed."
            );
            false
        } else {
            true
        }
    };

    // 1) Moments
    let mom_params = if gates.run_mom {
        let Some(moments_pool) = moments_pool.as_ref() else {
            log::warn!("Moments unavailable: method-specific null pool could not be built.");
            return None;
        };

        let scores = moments_pool.scores_in_window(
            settings.moments_min_null_rank,
            settings.moments_max_null_rank,
        );

        let fit_scores = if settings.moments_robust_fit {
            let x = winsorize_scores_for_fit(
                &scores,
                settings.moments_winsor_lower_q,
                settings.moments_winsor_upper_q,
            );

            log::info!(
                "Moments robust fit: enabled=true bias_corrected=true raw_n={} fit_n={} winsor_q=[{:.3}, {:.3}]",
                scores.len(),
                x.len(),
                settings.moments_winsor_lower_q,
                settings.moments_winsor_upper_q
            );

            x
        } else {
            scores.clone()
        };

        if window_ok(
            "Moments",
            settings.moments_min_null_rank,
            settings.moments_max_null_rank,
            fit_scores.len(),
        ) {
            let (mu, beta) = if settings.moments_robust_fit {
                fit_gumbel_winsorized_moments(
                    &fit_scores,
                    settings.moments_winsor_lower_q,
                    settings.moments_winsor_upper_q,
                )
            } else {
                fit_gumbel_moments(&fit_scores)
            };
            if mu.is_finite() && beta.is_finite() && beta > 0.0 {
                Some((mu, beta))
            } else {
                log::warn!(
                    "Moments fit produced invalid parameters; Moments expert will be unavailable. \
					 If model_fit=Moments, the selected-model guard will fail closed."
                );
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
    let mut lo_tev_by_index: Option<Arc<[Option<f64>]>> = None;

    if gates.run_lo {
        let Some(lower_order_pool) = lower_order_pool.as_ref() else {
            log::warn!("LowerOrder unavailable: method-specific null pool could not be built.");
            return None;
        };

        let lo_tev_map =
            build_lo_tev_from_spectrum_tail_components(features, settings, lower_order_pool);
        let lo_tev_valid = lo_tev_map.valid;
        let lo_tev_invalid = lo_tev_map.invalid;
        let tev_by_index: Arc<[Option<f64>]> = lo_tev_map.by_index.into();
        lo_tev_by_index = Some(Arc::clone(&tev_by_index));

        log::info!(
            "LO spectrum-local TEV diagnostics: valid={} invalid={} source=core.lo_spectrum_tail_p+core.lo_spectrum_candidate_count candidate_count_power={:.3} evalue_scale={:.3} tev_transform={:?}",
            lo_tev_valid,
            lo_tev_invalid,
            settings.lo_evalue_candidate_count_power,
            settings.lo_evalue_scale,
            settings.lo_tev_transform
        );

        let mut rank1_scores_by_charge: Vec<(f64, u8)> =
            Vec::with_capacity(work.rank1_indices.len());

        for &i in &work.rank1_indices {
            let f = &features[i];
            if let Some(x_lo) = tev_by_index.get(i).copied().flatten() {
                let bid = lo_bucket_id(settings, f.core.charge);
                rank1_scores_by_charge.push((x_lo, bid));
            }
        }

        let lo_raw = lower_order_pool.indexed_fit_data_in_window(
            settings.lower_order_min_null_rank,
            settings.lower_order_max_null_rank,
        );

        if window_ok(
            "LowerOrder",
            settings.lower_order_min_null_rank,
            settings.lower_order_max_null_rank,
            lo_raw.len(),
        ) {
            let lo_fit_data: Vec<(u32, f64, u8)> = lo_raw
                .into_iter()
                .filter_map(|(feature_idx, k, _x_raw, charge)| {
                    let x_lo = tev_by_index.get(feature_idx).copied().flatten()?;
                    Some((k, x_lo, lo_bucket_id(settings, charge)))
                })
                .collect();

            if log::log_enabled!(log::Level::Info) {
                let mut by_rank: FnvHashMap<u32, (usize, f64, f64)> = FnvHashMap::default();

                for &(k, x, _) in &lo_fit_data {
                    let entry =
                        by_rank
                            .entry(k)
                            .or_insert((0usize, f64::INFINITY, f64::NEG_INFINITY));
                    entry.0 += 1;
                    entry.1 = entry.1.min(x);
                    entry.2 = entry.2.max(x);
                }

                let mut ranks: Vec<u32> = by_rank.keys().copied().collect();
                ranks.sort_unstable();

                let summary = ranks
                    .into_iter()
                    .map(|k| {
                        let (n, lo, hi) = by_rank.get(&k).copied().unwrap();
                        format!("k{}:n{}:[{:.4},{:.4}]", k, n, lo, hi)
                    })
                    .collect::<Vec<_>>()
                    .join(" ");

                log::info!(
					"LO TEV fit-data diagnostics: window=[{}..={}] rank1_rows={} fit_rows={} ranks={}",
					settings.lower_order_min_null_rank,
					settings.lower_order_max_null_rank,
					rank1_scores_by_charge.len(),
					lo_fit_data.len(),
					summary
				);

                let mut by_rank_values: FnvHashMap<u32, Vec<f64>> = FnvHashMap::default();

                for &(k, x, _) in &lo_fit_data {
                    if x.is_finite() {
                        by_rank_values.entry(k).or_default().push(x);
                    }
                }

                let mut rank_keys: Vec<u32> = by_rank_values.keys().copied().collect();
                rank_keys.sort_unstable();

                for k in rank_keys {
                    if let Some(xs) = by_rank_values.get(&k) {
                        log_f64_distribution(&format!("LO fit-data TEV rank={}", k), xs);
                    }
                }
            }

            // LowerOrder uses TEV scores derived from Sage spectrum-local E-values.
            // The selected LoTevTransform determines the final TEV score scale.
            let lo_min_count_per_rank = settings.lo_min_count_per_rank;

            log::info!(
				"LO TNM fit mode: local_lom_extrapolated source=supported_lower_order_lom_mles tev_transform={:?} extrapolation_strength={:.3}",
				settings.lo_tev_transform,
				settings.lo_tnm_extrapolation_strength
			);

            lo_model = fit_decoy_free_model(
                &lo_fit_data,
                &rank1_scores_by_charge,
                settings.lower_order_min_null_rank,
                settings.lower_order_max_null_rank,
                lo_min_count_per_rank,
                settings.lo_tnm_extrapolation_strength,
            );

            if lo_model.is_none() {
                log::error!(
					"LO failed closed after local-extrapolated TNM fit attempt: requested_window=[{}..={}] effective_min={} fit_rows={} rank1_rows={}. \
					 LO diagnostic fields and LowerOrder-selected DF fields will be left blank.",
					settings.lower_order_min_null_rank,
					settings.lower_order_max_null_rank,
					settings.lower_order_min_null_rank.max(2),
					lo_fit_data.len(),
					rank1_scores_by_charge.len()
				);
            }
        }
    }

    // 3) MLE
    let mle_params = if gates.run_mle {
        let Some(mle_pool) = mle_pool.as_ref() else {
            log::warn!("MLE unavailable: method-specific null pool could not be built.");
            return None;
        };

        let scores =
            mle_pool.scores_in_window(settings.mle_min_null_rank, settings.mle_max_null_rank);

        let fit_scores = if settings.mle_robust_fit {
            let x = winsorize_scores_for_fit(
                &scores,
                settings.mle_winsor_lower_q,
                settings.mle_winsor_upper_q,
            );

            log::info!(
                "MLE robust preprocessing: enabled=true raw_n={} fit_n={} winsor_q=[{:.3}, {:.3}]",
                scores.len(),
                x.len(),
                settings.mle_winsor_lower_q,
                settings.mle_winsor_upper_q
            );

            x
        } else {
            scores.clone()
        };

        if window_ok(
            "MLE",
            settings.mle_min_null_rank,
            settings.mle_max_null_rank,
            fit_scores.len(),
        ) {
            match fit_gumbel_mle(&fit_scores) {
                Some((mu, beta)) if mu.is_finite() && beta.is_finite() && beta > 0.0 => {
                    Some((mu, beta))
                }
                _ => {
                    log::warn!(
                        "MLE fit produced invalid parameters; MLE expert will be unavailable. \
						 If model_fit=Mle, the selected-model guard will fail closed."
                    );
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
        let Some(msfdr_seeded_pool) = msfdr_seeded_pool.as_ref() else {
            log::warn!("MSFDR seeded unavailable: method-specific null pool could not be built.");
            return None;
        };

        let seed_pool = msfdr_seeded_pool
            .scores_in_window(settings.msfdr_min_null_rank, settings.msfdr_max_null_rank);
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
        let m = fit_msfdr_1smix(&rank1_scores, settings);
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
        // MSFDR2 is a joint S1/S2 mixture model. Unlike Moments/MLE/LO,
        // it should not receive the purified rank-null pool because the
        // model explicitly includes a correct-in-S2 component (`b`).
        //
        // Use raw lower-rank scores directly from features. Also enforce
        // rank >= 2 because S2 must not contain the rank-1 S1 scores.
        let effective_min_rank = settings.msfdr2_smix_min_null_rank.max(2);
        let effective_max_rank = settings.msfdr2_smix_max_null_rank.max(effective_min_rank);

        if settings.msfdr2_smix_min_null_rank < 2 {
            log::warn!(
				"MSFDR pooled-rank 2smix: requested min rank {} includes S1; using effective S2 min rank {}",
				settings.msfdr2_smix_min_null_rank,
				effective_min_rank
			);
        }

        let unpurified_s2_scores: Vec<f64> = features
            .iter()
            .filter(|f| {
                let r = f.core.rank as u32;
                r >= effective_min_rank && r <= effective_max_rank
            })
            .filter_map(|f| tev(f))
            .filter(|x| x.is_finite())
            .collect();

        log::info!(
			"DF MSFDR pooled-rank 2smix S2 source: unpurified_features ranks={}..{} n_s1={} n_s2={}",
			effective_min_rank,
			effective_max_rank,
			rank1_scores.len(),
			unpurified_s2_scores.len()
		);

        if window_ok(
            "MSFDR pooled-rank 2smix",
            effective_min_rank,
            effective_max_rank,
            unpurified_s2_scores.len(),
        ) {
            let m = fit_msfdr_2smix(&rank1_scores, &unpurified_s2_scores, settings);
            if let Some(ref model) = m {
                log_fit_ok("MSFDR pooled-rank 2smix", model);
            } else {
                log_fit_failed_closed("MSFDR pooled-rank 2smix");
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
    let mut nokoi_peps = None;

    if gates.run_nokoi {
        log::info!("Running Nokoi Rescoring ...");

        let Some(nokoi_pool) = nokoi_pool.as_ref() else {
            log::warn!("Nokoi unavailable: method-specific null pool could not be built.");
            return None;
        };

        let nokoi_count =
            nokoi_pool.count_in_window(settings.nokoi_min_null_rank, settings.nokoi_max_null_rank);

        if window_ok(
            "Nokoi",
            settings.nokoi_min_null_rank,
            settings.nokoi_max_null_rank,
            nokoi_count,
        ) {
            let mut rank1_hs: Vec<f64> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| tev(&features[i]))
                .collect();
            let threshold = if rank1_hs.len() >= 10 {
                rank1_hs.sort_by(|a, b| b.total_cmp(a));
                let top_k = ((rank1_hs.len() as f64) * settings.nokoi_positive_top_fraction).round()
                    as usize;
                rank1_hs[top_k.max(5).min(rank1_hs.len()) - 1]
            } else {
                f64::INFINITY
            };

            log::info!(
                "Nokoi training split: null_purification_factor={:.3} positive_top_fraction={:.3} positive_threshold={:.6e}",
                settings.nokoi_null_purification_factor,
                settings.nokoi_positive_top_fraction,
                threshold
            );

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

            let nokoi_null_indices = nokoi_pool.feature_indices();
            if let Some((probs, null_scores_oof)) = nokoi::rescore_df_crossfit(
                features,
                &config,
                settings.nokoi_min_null_rank,
                settings.nokoi_max_null_rank,
                settings.nokoi_k_folds,
                is_positive,
                &nokoi_null_indices,
            ) {
                let nokoi_evidence =
                    nokoi::build_nokoi_evidence_from_crossfit_null(&probs, &null_scores_oof);

                if nokoi_evidence.p_values.len() != features.len()
                    || nokoi_evidence.peps.len() != features.len()
                {
                    log::warn!(
						"Nokoi disabled: paired evidence length mismatch p_values={} peps={} features={}",
						nokoi_evidence.p_values.len(),
						nokoi_evidence.peps.len(),
						features.len()
					);
                } else {
                    nokoi_p_values = Some(Arc::new(
                        nokoi_evidence
                            .p_values
                            .into_iter()
                            .map(|p| p.clamp(0.0, 1.0).max(1e-300))
                            .collect(),
                    ));

                    nokoi_peps = Some(Arc::new(
                        nokoi_evidence
                            .peps
                            .into_iter()
                            .map(|pep| pep.clamp(0.0, 1.0).max(1e-300))
                            .collect(),
                    ));
                }
            } else {
                log::warn!("Nokoi disabled: crossfit failed.");
            }
        }
    }

    // Fail closed when a non-ensemble selected base model was requested but did not fit.
    //
    // In single-model mode, the selected model is the mandatory base DF stream.
    // Returning Some(Engines { selected_model: None, ... }) would allow
    // score_base_rank1() to silently substitute p=1.0 / PEP=1.0 for every PSM,
    // producing a superficially valid but uninformative DF run.
    //
    // Ensemble mode is different: unavailable experts are simply excluded from
    // the ensemble, so missing optional expert fits are allowed there.
    if !matches!(settings.model_fit, ModelFit::Ensemble) {
        let selected_fit_ok = match settings.model_fit {
            ModelFit::Moments => mom_params.is_some(),
            ModelFit::Mle => mle_params.is_some(),
            ModelFit::LowerOrder => lo_model.is_some(),
            ModelFit::Msfdr => msfdr_seeded.is_some(),
            ModelFit::Msfdr1Smix => msfdr_1smix.is_some(),
            ModelFit::Msfdr2Smix => msfdr_2smix.is_some(),
            ModelFit::Nokoi => nokoi_p_values.is_some() && nokoi_peps.is_some(),
            ModelFit::Ensemble => true,
        };

        if !selected_fit_ok {
            log::error!(
                "DF fail-closed: selected model_fit={:?} did not produce a valid fitted engine. \
                 Clearing selected DF outputs instead of substituting p=1.0/PEP=1.0.",
                settings.model_fit
            );
            return None;
        }
    }

    Some(Engines {
        mom_params,
        mle_params,
        lo_model,
        lo_tev_by_index,
        msfdr_seeded,
        msfdr_1smix,
        msfdr_2smix,
        nokoi_p_values,
        nokoi_peps,
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

#[derive(Clone, Debug, Default)]
struct PhysicalRescueResult {
    pub enabled: bool,
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

fn fit_base_experts(
    features: &[DfFeature],
    work: &WorkSet,
    settings: &FdrSettings,
    gates: RunGates,
) -> Option<Engines> {
    fit_engines(features, work, settings, gates)
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
    let lo_tev_by_index = engines.lo_tev_by_index.clone();
    let msfdr_seeded = engines.msfdr_seeded.clone();
    let msfdr_1smix = engines.msfdr_1smix.clone();
    let msfdr_2smix = engines.msfdr_2smix.clone();
    let nokoi_p_values = engines.nokoi_p_values.clone();
    let nokoi_peps = engines.nokoi_peps.clone();

    let use_mom_expert = gates.run_mom && mom_params.is_some();
    let use_mle_expert = gates.run_mle && mle_params.is_some();
    let use_lo_expert = gates.run_lo && lo_model.is_some();
    let use_seeded_expert = gates.run_msfdr_seeded && msfdr_seeded.is_some();
    let use_1smix_expert = gates.run_msfdr_1smix && msfdr_1smix.is_some();
    let use_2smix_expert = gates.run_msfdr_2smix && msfdr_2smix.is_some();
    let use_nokoi_expert = gates.run_nokoi && nokoi_p_values.is_some() && nokoi_peps.is_some();

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

            let p_lo = if let (Some(ref m), Some(ref tev_by_index)) = (&lo_model, &lo_tev_by_index)
            {
                let bid = lo_bucket_id(settings, psm.core.charge);

                match tev_by_index.get(idx).copied().flatten() {
                    Some(x_eval) => {
                        let p = m.p_value(x_eval, bid);
                        if p.is_finite() {
                            p.clamp(0.0, 1.0).max(1e-300)
                        } else {
                            // Fail closed for malformed LO evaluation, but keep the PSM.
                            1.0
                        }
                    }
                    None => {
                        // Global-hyperscore LO TEV should exist when the relevant lower-rank
                        // bucket fit succeeds. Missing LO evidence still fails closed, not by
                        // removing the row.
                        1.0
                    }
                }
            } else {
                1.0
            };

            let (p_msfdr, pep_msfdr) = if use_seeded_expert {
                let m = msfdr_seeded.as_ref().unwrap();
                (
                    Some(m.p_value(x).clamp(0.0, 1.0).max(1e-300)),
                    Some(m.pep(x).clamp(0.0, 1.0).max(1e-300)),
                )
            } else {
                (None, None)
            };

            let (p_1smix, pep_1smix) = if use_1smix_expert {
                let m = msfdr_1smix.as_ref().unwrap();
                (
                    Some(m.p_value(x).clamp(0.0, 1.0).max(1e-300)),
                    Some(m.pep(x).clamp(0.0, 1.0).max(1e-300)),
                )
            } else {
                (None, None)
            };

            let (p_2smix, pep_2smix) = if use_2smix_expert {
                let m = msfdr_2smix.as_ref().unwrap();
                (
                    Some(m.p_value(x).clamp(0.0, 1.0).max(1e-300)),
                    Some(m.pep(x).clamp(0.0, 1.0).max(1e-300)),
                )
            } else {
                (None, None)
            };

            let (p_nokoi, pep_nokoi) = if use_nokoi_expert {
                let p_vec = nokoi_p_values.as_ref().unwrap();
                let pep_vec = nokoi_peps.as_ref().unwrap();

                (
                    Some(
                        p_vec
                            .get(idx)
                            .copied()
                            .unwrap_or(1.0)
                            .clamp(0.0, 1.0)
                            .max(1e-300),
                    ),
                    Some(
                        pep_vec
                            .get(idx)
                            .copied()
                            .unwrap_or(1.0)
                            .clamp(0.0, 1.0)
                            .max(1e-300),
                    ),
                )
            } else {
                (None, None)
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
                pep_msfdr,
                pep_1smix,
                pep_2smix,
                p_nokoi,
                pep_nokoi,
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

    if matches!(settings.model_fit, ModelFit::LowerOrder) {
        let finite_lo: Vec<f64> = p_lo_all.iter().copied().filter(|p| p.is_finite()).collect();

        if !finite_lo.is_empty() {
            let n_total = finite_lo.len();
            let n_one = finite_lo.iter().filter(|&&p| p >= 0.999999).count();
            let n_floor = finite_lo.iter().filter(|&&p| p <= 1e-250).count();

            let mut sorted = finite_lo.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));

            let q_at = |frac: f64| -> f64 {
                let idx = (frac.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64)).round() as usize;
                sorted[idx.min(sorted.len() - 1)]
            };

            log::info!(
				"LO rank1 p-value diagnostics: n={} floor_like={} one_like={} q=[{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}]",
				n_total,
				n_floor,
				n_one,
				q_at(0.00),
				q_at(0.01),
				q_at(0.10),
				q_at(0.50),
				q_at(0.90),
				q_at(0.99),
				q_at(1.00)
			);
        }

        if let Some(ref tev_by_index) = lo_tev_by_index {
            log_lower_order_rank1_score_diagnostics(features, &workset, db, tev_by_index, settings);
        } else {
            log::warn!("LO rank1 TEV/component diagnostics skipped: no LO TEV map was available.");
        }
    }

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

    // Companion PEP-like streams for p-native experts are calibrated on the same
    // reference population used to estimate pi0, then mapped back to all rank-1
    // rows by monotone interpolation on the p-value scale.
    //
    // This avoids mixing populations between:
    //   numerator: pi0 estimated from reference target/non-contam/non-entrapment PSMs
    //   denominator: Grenander density fitted on all rank-1 PSMs
    //
    // The resulting pep_* streams remain empirical PEP-like calibration aids, not
    // formally validated posterior probabilities.
    let pep_mom_vec = grenander_pep_from_reference_population(&p_mom_all, &is_ref, pi0_mom);
    let pep_mle_vec = grenander_pep_from_reference_population(&p_mle_all, &is_ref, pi0_mle);
    let pep_lo_vec = grenander_pep_from_reference_population(&p_lo_all, &is_ref, pi0_lo);

    // MSFDR-family models expose both streams:
    // - p_*     = native p-like null/incorrect tail probability
    // - pep_*   = native model posterior error probability
    //
    // Do not derive 1SMix/2SMix PEPs from the threshold-FDR curve, and do not
    // overwrite native model posteriors with Grenander unless a separate
    // calibration experiment explicitly justifies that.
    let pep_msfdr_vec: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.pep_msfdr.unwrap_or(1.0).clamp(0.0, 1.0).max(1e-300))
        .collect();

    let pep_1smix_vec: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.pep_1smix.unwrap_or(1.0).clamp(0.0, 1.0).max(1e-300))
        .collect();

    let pep_2smix_vec: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.pep_2smix.unwrap_or(1.0).clamp(0.0, 1.0).max(1e-300))
        .collect();

    // Nokoi paired evidence:
    // - p_nokoi   = empirical null-survival p-value from Nokoi score
    // - pep_nokoi = paired posterior-error / PEP-like stream from Nokoi evidence builder
    let pep_nokoi_vec: Vec<f64> = rank1_out
        .iter()
        .map(|r| r.pep_nokoi.unwrap_or(1.0).clamp(0.0, 1.0).max(1e-300))
        .collect();

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
            Some(finite_df_p_value(r.p_mom))
        } else {
            None
        };
        psm.p_mle = if use_mle_expert {
            Some(finite_df_p_value(r.p_mle))
        } else {
            None
        };
        psm.p_lo = if use_lo_expert {
            Some(finite_df_p_value(r.p_lo))
        } else {
            None
        };
        psm.p_msfdr = if use_seeded_expert {
            r.p_msfdr.map(finite_df_p_value)
        } else {
            None
        };
        psm.p_1smix = if use_1smix_expert {
            r.p_1smix.map(finite_df_p_value)
        } else {
            None
        };
        psm.p_2smix = if use_2smix_expert {
            r.p_2smix.map(finite_df_p_value)
        } else {
            None
        };
        psm.p_nokoi = if use_nokoi_expert {
            r.p_nokoi.map(finite_df_p_value)
        } else {
            None
        };

        psm.pep_mom = if use_mom_expert {
            Some(finite_df_probability_for_logit(base_res.pep_mom_vec[j]))
        } else {
            None
        };
        psm.pep_mle = if use_mle_expert {
            Some(finite_df_probability_for_logit(base_res.pep_mle_vec[j]))
        } else {
            None
        };
        psm.pep_lo = if use_lo_expert {
            Some(finite_df_probability_for_logit(base_res.pep_lo_vec[j]))
        } else {
            None
        };
        psm.pep_msfdr = if use_seeded_expert {
            Some(finite_df_probability_for_logit(base_res.pep_msfdr_vec[j]))
        } else {
            None
        };
        psm.pep_1smix = if use_1smix_expert {
            Some(finite_df_probability_for_logit(base_res.pep_1smix_vec[j]))
        } else {
            None
        };
        psm.pep_2smix = if use_2smix_expert {
            Some(finite_df_probability_for_logit(base_res.pep_2smix_vec[j]))
        } else {
            None
        };
        psm.pep_nokoi = if use_nokoi_expert {
            Some(finite_df_probability_for_logit(base_res.pep_nokoi_vec[j]))
        } else {
            None
        };

        let active_space = active_evidence_space(settings);

        let p_consensus: f64 = if use_ensemble {
            let mut p_experts: Vec<f64> = Vec::new();

            if use_mom_expert {
                p_experts.push(r.p_mom);
            }
            if use_mle_expert {
                p_experts.push(r.p_mle);
            }
            if use_lo_expert {
                p_experts.push(r.p_lo);
            }
            if use_seeded_expert {
                if let Some(p) = r.p_msfdr {
                    p_experts.push(p);
                }
            }
            if use_1smix_expert {
                if let Some(p) = r.p_1smix {
                    p_experts.push(p);
                }
            }
            if use_2smix_expert {
                if let Some(p) = r.p_2smix {
                    p_experts.push(p);
                }
            }
            if use_nokoi_expert {
                if let Some(p) = r.p_nokoi {
                    p_experts.push(p);
                }
            }

            combine_p_values_for_ensemble(&p_experts, settings)
        } else {
            r.p_final
        }
        .clamp(0.0, 1.0)
        .max(1e-300);

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

        set_df_evidence_pair(psm, active_space, p_consensus, pep_consensus);
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
            psm.decoy_free_protein_supported_peptide = None;
            psm.decoy_free_peptide_supported_psm = None;
            psm.rt_residual = None;
            psm.abs_rt_residual = None;
            psm.rt_z = None;
            psm.rt_within_1sigma = None;
            psm.rt_within_2sigma = None;
            psm.rt_within_3sigma = None;

            psm.ims_residual = None;
            psm.abs_ims_residual = None;
            psm.ims_z = None;
            psm.ims_within_1sigma = None;
            psm.ims_within_2sigma = None;
            psm.ims_within_3sigma = None;

            psm.physical_rescue_source = None;
            psm.rescued_by_rt = None;
            psm.rescued_by_ims = None;
            psm.rescued_by_recurrence = None;

            psm.rt_local_z = None;
            psm.rt_local_outlier = None;
            psm.rt_training_eligible = None;
            psm.decoy_free_p_value_base = None;
            psm.decoy_free_pep_base = None;
            psm.decoy_free_score_base = None;
            psm.decoy_free_q_base = None;
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

            psm.decoy_free_p_value_rt = None;
            psm.decoy_free_pep_rt = None;
            psm.decoy_free_score_rt = None;
            psm.decoy_free_q_rt = None;

            psm.decoy_free_p_value_ims = None;
            psm.decoy_free_pep_ims = None;
            psm.decoy_free_score_ims = None;
            psm.decoy_free_q_ims = None;

            psm.decoy_free_p_value_peptide_rescue = None;
            psm.decoy_free_pep_peptide_rescue = None;
            psm.decoy_free_score_peptide_rescue = None;
            psm.decoy_free_q_peptide_rescue = None;

            psm.decoy_free_p_value_protein_rescue = None;
            psm.decoy_free_pep_protein_rescue = None;
            psm.decoy_free_score_protein_rescue = None;
            psm.decoy_free_q_protein_rescue = None;

            psm.p_ensemble = None;
            psm.q_ensemble = None;
            psm.pep_ensemble = None;
            psm.score_ensemble = None;

            psm.rt_adjust_p_ensemble = None;
            psm.rt_adjust_q_ensemble = None;
            psm.rt_adjust_pep_ensemble = None;

            psm.ims_adjust_p_ensemble = None;
            psm.ims_adjust_q_ensemble = None;
            psm.ims_adjust_pep_ensemble = None;

            psm.peptide_rescue_p_ensemble = None;
            psm.peptide_rescue_q_ensemble = None;
            psm.peptide_rescue_pep_ensemble = None;

            psm.protein_rescue_p_ensemble = None;
            psm.protein_rescue_q_ensemble = None;
            psm.protein_rescue_pep_ensemble = None;
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

    let rank1_p: Vec<f64> = work
        .rank1_indices
        .iter()
        .filter_map(|&i| features[i].decoy_free_p_value.map(finite_df_p_value))
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
                    .map(finite_df_probability_for_logit)
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

    let active_space = active_evidence_space(settings);

    match active_space {
        ActiveEvidenceSpace::Pep => {
            summarize_q(
                "PREQ rank1 active PEP stream",
                work.rank1_indices
                    .iter()
                    .filter_map(|&i| features[i].decoy_free_pep),
            );

            let rows: Vec<(f64, usize, f64)> = work
                .rank1_indices
                .iter()
                .filter_map(|&i| {
                    let f = &features[i];
                    let pep = f.decoy_free_pep?;
                    if !pep.is_finite() {
                        return None;
                    }

                    let score_key = f
                        .decoy_free_score
                        .map(|s| s as f64)
                        .unwrap_or_else(|| df_score_from_pep(pep) as f64);

                    Some((score_key, i, pep.clamp(0.0, 1.0).max(1e-300)))
                })
                .collect();

            for (feat_idx, q) in q_from_pep_cummean(rows) {
                set_df_q_value(&mut features[feat_idx], q);
            }
        }

        ActiveEvidenceSpace::PValue => {
            summarize_q(
                "PREQ rank1 active p-value stream",
                work.rank1_indices
                    .iter()
                    .filter_map(|&i| features[i].decoy_free_p_value),
            );

            let psm_cov_values: Vec<Option<f64>> = work
                .rank1_indices
                .iter()
                .map(|&idx| psm_covariate_value(&features[idx], settings.psm_q_covariate))
                .collect();

            let q_report = q_values_from_level_covariates(
                &rank1_p,
                &rank1_p_ref,
                &psm_cov_values,
                settings,
                effective_psm_q_method(settings),
                QLevel::Psm,
                "PSM active p-value",
            );

            let q_values = q_report.q_values;

            for (&idx, q) in work.rank1_indices.iter().zip(q_values) {
                set_df_q_value(&mut features[idx], q);
            }
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

    // If the active model fit is Ensemble, preserve the finalized base ensemble
    // stream in explicit ensemble audit columns before any later RT/IMS/rescue
    // stage is allowed to replace the live decoy_free_* controlling stream.
    //
    // At this point:
    //   decoy_free_p_value = ensemble base p-like stream, if populated
    //   decoy_free_pep     = ensemble base PEP stream
    //   decoy_free_score   = ensemble base score
    //   decoy_free_q_value = finalized ensemble base q-value
    if use_ensemble {
        for &i in &work.rank1_indices {
            let f = &mut features[i];

            if f.core.rank != 1 {
                continue;
            }

            f.p_ensemble = f.decoy_free_p_value;
            f.q_ensemble = f.decoy_free_q_value;
            f.pep_ensemble = f.decoy_free_pep;
            f.score_ensemble = f.decoy_free_score;
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

        q_values_from_p_values_with_method(
            p_present,
            p_ref,
            settings,
            effective_psm_q_method(settings),
            "expert diagnostic",
        )
    };

    let q_mom_present = compute_q_present(&p_mom_present, &p_mom_ref);
    let q_mle_present = compute_q_present(&p_mle_present, &p_mle_ref);
    let q_lo_present = compute_q_present(&p_lo_present, &p_lo_ref);
    let q_msfdr_present = compute_q_present(&p_msfdr_present, &p_msfdr_ref);
    let q_nokoi_present = compute_q_present(&p_nokoi_present, &p_nokoi_ref);

    for (j, &k) in mom_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_mom = Some(finite_df_p_value(q_mom_present[j]));
    }
    for (j, &k) in mle_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_mle = Some(finite_df_p_value(q_mle_present[j]));
    }
    for (j, &k) in lo_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_lo = Some(finite_df_p_value(q_lo_present[j]));
    }
    for (j, &k) in msfdr_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_msfdr = Some(finite_df_p_value(q_msfdr_present[j]));
    }
    for (j, &k) in nokoi_pos.iter().enumerate() {
        features[work.rank1_indices[k]].q_nokoi = Some(finite_df_p_value(q_nokoi_present[j]));
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
                features[i].q_1smix = Some(finite_df_p_value(q));
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
                features[i].q_2smix = Some(finite_df_p_value(q));
            }
        }
    }
}

// =============================================================================
// Optional physical evidence stages: shared anchor and reliability scaffolding
// =============================================================================

fn build_physical_anchor_set(
    features: &[DfFeature],
    settings: &FdrSettings,
    _db: &crate::database::IndexedDatabase,
) -> Vec<usize> {
    let max_pep = settings.physical_rescue.anchor_max_pep as f64;
    let max_q = settings.physical_rescue.anchor_max_q as f64;

    features
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            if f.core.rank != 1 {
                return false;
            }

            // Physical-stage anchors are selected from the frozen base stream, not from the
            // mutable active stream. This prevents an RT/IMS stage from using evidence that
            // was already modified by an earlier optional stage.
            let pep = f.decoy_free_pep_base.unwrap_or(1.0);
            if !pep.is_finite() || pep > max_pep {
                return false;
            }

            // The q-value threshold enforces global base-stream confidence before a PSM can
            // contribute to RT/IMS reliability estimation.
            let q = f.decoy_free_q_base.unwrap_or(1.0);
            if q > max_q {
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

    let max_pep = settings.physical_rescue.anchor_max_pep as f64;
    let max_q = settings.physical_rescue.anchor_max_q as f64;
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

            pep.is_finite() && pep <= max_pep && q <= max_q
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
        "DF physical anchor diagnostics: total_before={} after_evidence={} after_run={} after_charge={} excluded_missing_rt={} excluded_contam_or_entrapment={} final_accepted={} runs={} charges={}",
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
    if !settings.enable_ims_confidence_adjustment {
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

    let fail_closed_hint = settings.enable_ims_confidence_adjustment
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
    let ims_active = settings.enable_ims_confidence_adjustment && ims.ims_sigma_global.is_some();

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

// Local optional-stage failure predicate for physical evidence updates.
// A true result means the RT/IMS stage must not modify the active stream.
fn should_fail_closed_physical(
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

// Physical-stage context shared by legacy DART and bounded auxiliary kernels.
// Semantically, this is the reliability and anchor context for optional RT/IMS updates.
#[derive(Clone, Debug)]
struct PhysicalContext {
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

fn prepare_physical_context(
    features: &[DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
    rt_scale_invalid: bool,
    missing_critical_diagnostics: bool,
) -> PhysicalContext {
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
        "DF physical RT reliability: global_sigma={:?}, reliability={:.3}, fail_closed_hint={}",
        rt_rel.rt_sigma_global,
        rt_rel.reliability,
        rt_rel.fail_closed_hint
    );

    if !rt_rel.runwise_rt_sigma.is_empty() {
        log::debug!(
            "DF physical RT runwise sigma: {:?}",
            rt_rel.runwise_rt_sigma
        );
    }

    if settings.enable_ims_confidence_adjustment {
        log::debug!(
            "DF physical IMS reliability: global_sigma={:?}, reliability={:.3}, fail_closed_hint={}",
            ims_rel.ims_sigma_global,
            ims_rel.reliability,
            ims_rel.fail_closed_hint
        );

        if !ims_rel.runwise_ims_sigma.is_empty() {
            log::debug!(
                "DF physical IMS runwise sigma: {:?}",
                ims_rel.runwise_ims_sigma
            );
        }
    } else {
        log::debug!("DF physical IMS reliability: disabled");
    }

    log::debug!(
        "DF physical joint reliability: joint_reliability={:.3}, fail_closed_hint={}",
        joint_rel.joint_reliability,
        joint_rel.fail_closed_hint
    );

    let is_unreliable = should_fail_closed_physical(
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

    PhysicalContext {
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

    snapshot_base_stream_once(features);

    match settings.physical_rescue.rt_mode {
        PhysicalRescueMode::Off => {
            for f in features.iter_mut() {
                if f.core.rank == 1 {
                    f.physical_mode_used = Some("off".to_string());
                }
            }
            PhysicalRescueResult {
                enabled: false,
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
            let phys_ctx = prepare_physical_context(features, settings, db, false, false);

            match settings.physical_rescue.rt_mode {
                PhysicalRescueMode::DartBayes => {
                    apply_dart_bayes_update(features, settings, &phys_ctx);
                }
                PhysicalRescueMode::BoundedAux => {
                    apply_bounded_physical_shift(features, settings, &phys_ctx);
                }
                PhysicalRescueMode::Off => unreachable!(),
            }

            PhysicalRescueResult {
                enabled: true,
                fail_closed: phys_ctx.is_unreliable,
                anchor_count_total: phys_ctx.anchor_count_total,
                anchor_count_after_filters: phys_ctx.anchors.len(),
                rt_reliability: phys_ctx.rt_rel.reliability,
                ims_reliability: phys_ctx.ims_rel.reliability,
                joint_reliability: phys_ctx.joint_rel.joint_reliability,
                rt_sigma_global: phys_ctx.rt_rel.rt_sigma_global,
                ims_sigma_global: phys_ctx.ims_rel.ims_sigma_global,
                dropped_runs: phys_ctx.dropped_runs.clone(),
                dropped_charge_bins: phys_ctx.dropped_charge_bins.clone(),
            }
        }
    }
}

// =============================================================================
// Independent optional physical stages: RT-only and IMS-only wrappers
// =============================================================================

fn apply_rt_dart_bayes_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> PhysicalRescueResult {
    let mut rt_settings = settings.clone();

    // Force the legacy DART implementation to behave as an RT-only optional stage.
    // The DART kernel proposes posterior PEP/score/q updates; the wrapper-level
    // stage outcome determines whether those values are accepted as the new active
    // stream.
    rt_settings.enable_ims_confidence_adjustment = false;
    rt_settings.physical_rescue.rt_mode = crate::input::PhysicalRescueMode::DartBayes;
    rt_settings.physical_rescue.ims_mode = crate::input::PhysicalRescueMode::Off;

    apply_physical_rescue(features, &rt_settings, db)
}

fn apply_physical_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
    stage: PhysicalEvidenceStage,
) -> PhysicalRescueResult {
    use crate::input::PhysicalRescueMode;

    match stage {
        PhysicalEvidenceStage::RtOnly => match settings.physical_rescue.rt_mode {
            PhysicalRescueMode::Off => PhysicalRescueResult {
                enabled: false,
                fail_closed: false,
                ..Default::default()
            },

            PhysicalRescueMode::BoundedAux | PhysicalRescueMode::DartBayes => {
                match active_evidence_space(settings) {
                    ActiveEvidenceSpace::Pep => match settings.physical_rescue.rt_mode {
                        PhysicalRescueMode::BoundedAux => {
                            apply_rt_bounded_update_to_active_stream(features, settings, db)
                        }
                        PhysicalRescueMode::DartBayes => {
                            apply_rt_dart_bayes_update_to_active_stream(features, settings, db)
                        }
                        PhysicalRescueMode::Off => unreachable!(),
                    },

                    ActiveEvidenceSpace::PValue => {
                        apply_rt_null_pvalue_update_to_active_stream(features, settings, db)
                    }
                }
            }
        },

        PhysicalEvidenceStage::ImsOnly => match settings.physical_rescue.ims_mode {
            PhysicalRescueMode::Off => PhysicalRescueResult {
                enabled: false,
                fail_closed: false,
                ..Default::default()
            },

            PhysicalRescueMode::BoundedAux | PhysicalRescueMode::DartBayes => {
                match active_evidence_space(settings) {
                    ActiveEvidenceSpace::Pep => match settings.physical_rescue.ims_mode {
                        PhysicalRescueMode::BoundedAux => {
                            apply_ims_bounded_update_to_active_stream(features, settings, db)
                        }
                        PhysicalRescueMode::DartBayes => {
                            apply_ims_dart_bayes_update_to_active_stream(features, settings, db)
                        }
                        PhysicalRescueMode::Off => unreachable!(),
                    },

                    ActiveEvidenceSpace::PValue => {
                        apply_ims_null_pvalue_update_to_active_stream(features, settings, db)
                    }
                }
            }
        },
    }
}

fn apply_rt_only_physical_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> PhysicalRescueResult {
    apply_physical_update_to_active_stream(features, settings, db, PhysicalEvidenceStage::RtOnly)
}

fn apply_ims_only_physical_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> PhysicalRescueResult {
    apply_physical_update_to_active_stream(features, settings, db, PhysicalEvidenceStage::ImsOnly)
}

#[inline]
fn rt_delta_for_feature(f: &DfFeature) -> Option<f64> {
    let delta = f.core.delta_rt_model as f64;

    if delta.is_finite() {
        Some(delta)
    } else {
        None
    }
}

struct RtAuxNullModel {
    global: Vec<f64>,
    by_file: HashMap<usize, Vec<f64>>,
    by_region: HashMap<i32, Vec<f64>>,
    by_file_region: HashMap<(usize, i32), Vec<f64>>,
    rt_sigma_global: Option<f64>,
    reliability: f64,
}

#[inline]
fn rt_region_bin_for_feature(f: &DfFeature, settings: &FdrSettings) -> Option<i32> {
    let pred = f.core.predicted_rt as f64;
    if !pred.is_finite() {
        return None;
    }

    let bins = settings.physical_rescue.rt_region_bins.max(2) as f64;
    let b = (pred.clamp(0.0, 1.0) * bins).floor() as i32;

    Some(b.clamp(0, bins as i32 - 1))
}

#[inline]
fn aux_null_p_from_sorted_agreements(null_sorted: &[f64], agreement_obs: f64) -> f64 {
    if null_sorted.is_empty() {
        return 1.0;
    }

    let idx = null_sorted.partition_point(|&x| x < agreement_obs);
    let count_ge = null_sorted.len().saturating_sub(idx);

    ((count_ge as f64 + 1.0) / (null_sorted.len() as f64 + 1.0)).clamp(1e-300, 1.0)
}

const DF_RT_LOCAL_OUTLIER_Z: f64 = 5.0;

#[inline]
fn robust_sigma_from_abs_residuals(mut vals: Vec<f64>, lo: f64, hi: f64) -> Option<f64> {
    vals.retain(|x| x.is_finite() && *x >= 0.0);
    if vals.is_empty() {
        return None;
    }

    vals.sort_by(|a, b| a.total_cmp(b));
    let med = vals[vals.len() / 2];

    Some((1.4826 * med).clamp(lo, hi))
}

#[inline]
fn rt_residual_for_feature(f: &DfFeature) -> Option<f64> {
    let aligned = f.core.aligned_rt as f64;
    let predicted = f.core.predicted_rt as f64;

    if aligned.is_finite() && predicted.is_finite() {
        Some(aligned - predicted)
    } else {
        None
    }
}

#[inline]
fn ims_residual_for_feature(f: &DfFeature) -> Option<f64> {
    let observed = f.core.ims as f64;
    let predicted = f.core.predicted_ims as f64;

    if observed.is_finite() && predicted.is_finite() {
        Some(observed - predicted)
    } else {
        None
    }
}

fn annotate_physical_residual_columns(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    rt_sigma: Option<f64>,
    ims_sigma: Option<f64>,
) {
    for f in features.iter_mut().filter(|f| f.core.rank == 1) {
        if let (Some(delta), Some(sig)) = (rt_residual_for_feature(f), rt_sigma) {
            if sig.is_finite() && sig > 0.0 {
                let abs_delta = delta.abs();
                let z = abs_delta / sig;

                f.rt_residual = Some(delta as f32);
                f.abs_rt_residual = Some(abs_delta as f32);
                f.rt_z = Some(z as f32);
                f.rt_within_1sigma = Some(z <= 1.0);
                f.rt_within_2sigma = Some(z <= 2.0);
                f.rt_within_3sigma = Some(z <= 3.0);
            }
        }

        if let (Some(delta), Some(sig)) = (ims_residual_for_feature(f), ims_sigma) {
            if sig.is_finite() && sig > 0.0 {
                let abs_delta = delta.abs();
                let z = abs_delta / sig;

                f.ims_residual = Some(delta as f32);
                f.abs_ims_residual = Some(abs_delta as f32);
                f.ims_z = Some(z as f32);
                f.ims_within_1sigma = Some(z <= 1.0);
                f.ims_within_2sigma = Some(z <= 2.0);
                f.ims_within_3sigma = Some(z <= 3.0);
            }
        }

        if f.rescued_by_rt.is_none() {
            f.rescued_by_rt = Some(false);
        }
        if f.rescued_by_ims.is_none() {
            f.rescued_by_ims = Some(false);
        }
        if f.rescued_by_recurrence.is_none() {
            f.rescued_by_recurrence = Some(false);
        }

        if f.physical_rescue_source.is_none() {
            f.physical_rescue_source = Some("none".to_string());
        }

        // Default to eligible unless local diagnostics prove otherwise.
        if f.rt_training_eligible.is_none() {
            f.rt_training_eligible = Some(true);
        }

        // Avoid warning for currently unused settings argument if future cfg wiring
        // is added later.
        let _ = settings;
    }
}

#[derive(Default, Clone, Copy)]
struct SigmaBinCounts {
    n: usize,

    le025: usize,
    le050: usize,
    le075: usize,
    le1: usize,

    le150: usize,
    le2: usize,
    le3: usize,
    gt3: usize,
}

impl SigmaBinCounts {
    fn observe(&mut self, z: Option<f32>) {
        let Some(z) = z else {
            return;
        };
        let z = z as f64;
        if !z.is_finite() {
            return;
        }

        self.n += 1;

        if z <= 0.25 {
            self.le025 += 1;
        }
        if z <= 0.50 {
            self.le050 += 1;
        }
        if z <= 0.75 {
            self.le075 += 1;
        }
        if z <= 1.0 {
            self.le1 += 1;
        }

        if z <= 1.50 {
            self.le150 += 1;
        }
        if z <= 2.0 {
            self.le2 += 1;
        }
        if z <= 3.0 {
            self.le3 += 1;
        } else {
            self.gt3 += 1;
        }
    }
}

#[inline]
fn fmt_sigma_bins(c: SigmaBinCounts) -> String {
    if c.n == 0 {
        return concat!(
            "n=0 ",
            "<=0.25σ=0(0.0%) ",
            "<=0.50σ=0(0.0%) ",
            "<=0.75σ=0(0.0%) ",
            "<=1σ=0(0.0%) ",
            "<=1.5σ=0(0.0%) ",
            "<=2σ=0(0.0%) ",
            "<=3σ=0(0.0%) ",
            ">3σ=0(0.0%)"
        )
        .to_string();
    }

    let n = c.n as f64;
    format!(
        concat!(
            "n={} ",
            "<=0.25σ={}({:.1}%) ",
            "<=0.50σ={}({:.1}%) ",
            "<=0.75σ={}({:.1}%) ",
            "<=1σ={}({:.1}%) ",
            "<=1.5σ={}({:.1}%) ",
            "<=2σ={}({:.1}%) ",
            "<=3σ={}({:.1}%) ",
            ">3σ={}({:.1}%)"
        ),
        c.n,
        c.le025,
        100.0 * c.le025 as f64 / n,
        c.le050,
        100.0 * c.le050 as f64 / n,
        c.le075,
        100.0 * c.le075 as f64 / n,
        c.le1,
        100.0 * c.le1 as f64 / n,
        c.le150,
        100.0 * c.le150 as f64 / n,
        c.le2,
        100.0 * c.le2 as f64 / n,
        c.le3,
        100.0 * c.le3 as f64 / n,
        c.gt3,
        100.0 * c.gt3 as f64 / n
    )
}

#[inline]
fn sigma_rate_le025(c: SigmaBinCounts) -> f64 {
    if c.n == 0 {
        0.0
    } else {
        c.le025 as f64 / c.n as f64
    }
}

#[inline]
fn sigma_rate_le050(c: SigmaBinCounts) -> f64 {
    if c.n == 0 {
        0.0
    } else {
        c.le050 as f64 / c.n as f64
    }
}

#[inline]
fn sigma_rate_le075(c: SigmaBinCounts) -> f64 {
    if c.n == 0 {
        0.0
    } else {
        c.le075 as f64 / c.n as f64
    }
}

#[inline]
fn sigma_rate_le150(c: SigmaBinCounts) -> f64 {
    if c.n == 0 {
        0.0
    } else {
        c.le150 as f64 / c.n as f64
    }
}

#[inline]
fn sigma_rate_le1(c: SigmaBinCounts) -> f64 {
    if c.n == 0 {
        0.0
    } else {
        c.le1 as f64 / c.n as f64
    }
}

#[inline]
fn sigma_rate_le2(c: SigmaBinCounts) -> f64 {
    if c.n == 0 {
        0.0
    } else {
        c.le2 as f64 / c.n as f64
    }
}

#[inline]
fn sigma_rate_le3(c: SigmaBinCounts) -> f64 {
    if c.n == 0 {
        0.0
    } else {
        c.le3 as f64 / c.n as f64
    }
}

#[derive(Default, Clone, Copy)]
struct RescueCompositionCounts {
    total: usize,
    target_ref: usize,
    entrapment: usize,
    contaminant: usize,
    accepted_psm_q_1pct: usize,
}

impl RescueCompositionCounts {
    fn observe(&mut self, f: &DfFeature, db: &IndexedDatabase) {
        let proteins = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        let is_ent = is_entrapment_str(&proteins);
        let is_cont = is_contam_str(&proteins);
        let is_ref = f.core.label == 1 && !is_ent && !is_cont;

        self.total += 1;

        if is_ref {
            self.target_ref += 1;
        }
        if is_ent {
            self.entrapment += 1;
        }
        if is_cont {
            self.contaminant += 1;
        }
        if f.decoy_free_q_value.unwrap_or(1.0) <= 0.01 {
            self.accepted_psm_q_1pct += 1;
        }
    }
}

#[inline]
fn pct_part(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

fn fmt_rescue_composition(c: RescueCompositionCounts) -> String {
    format!(
        "total={} target_ref={}({:.2}%) entrapment={}({:.2}%) contaminant={}({:.2}%) accepted_psm_q_1pct={}({:.2}%)",
        c.total,
        c.target_ref,
        pct_part(c.target_ref, c.total),
        c.entrapment,
        pct_part(c.entrapment, c.total),
        c.contaminant,
        pct_part(c.contaminant, c.total),
        c.accepted_psm_q_1pct,
        pct_part(c.accepted_psm_q_1pct, c.total)
    )
}

fn log_physical_rescue_enrichment_diagnostics(features: &[DfFeature], db: &IndexedDatabase) {
    let mut all_rank1 = RescueCompositionCounts::default();
    let mut accepted_psm_q_1pct = RescueCompositionCounts::default();
    let mut rescued_by_rt = RescueCompositionCounts::default();
    let mut rescued_by_ims = RescueCompositionCounts::default();
    let mut rescued_by_recurrence = RescueCompositionCounts::default();

    for f in features.iter().filter(|f| f.core.rank == 1) {
        all_rank1.observe(f, db);

        if f.decoy_free_q_value.unwrap_or(1.0) <= 0.01 {
            accepted_psm_q_1pct.observe(f, db);
        }

        if f.rescued_by_rt.unwrap_or(false) {
            rescued_by_rt.observe(f, db);
        }

        if f.rescued_by_ims.unwrap_or(false) {
            rescued_by_ims.observe(f, db);
        }

        if f.rescued_by_recurrence.unwrap_or(false) {
            rescued_by_recurrence.observe(f, db);
        }
    }

    log::info!(
        "DF physical rescue enrichment: all_rank1 {}",
        fmt_rescue_composition(all_rank1)
    );
    log::info!(
        "DF physical rescue enrichment: accepted_psm_q_1pct {}",
        fmt_rescue_composition(accepted_psm_q_1pct)
    );
    log::info!(
        "DF physical rescue enrichment: rescued_by_rt {}",
        fmt_rescue_composition(rescued_by_rt)
    );
    log::info!(
        "DF physical rescue enrichment: rescued_by_ims {}",
        fmt_rescue_composition(rescued_by_ims)
    );
    log::info!(
        "DF physical rescue enrichment: rescued_by_recurrence {}",
        fmt_rescue_composition(rescued_by_recurrence)
    );

    if all_rank1.total > 0 {
        let baseline_ent_rate = all_rank1.entrapment as f64 / all_rank1.total as f64;

        for (name, c) in [
            ("rescued_by_rt", rescued_by_rt),
            ("rescued_by_ims", rescued_by_ims),
            ("rescued_by_recurrence", rescued_by_recurrence),
        ] {
            if c.total == 0 {
                continue;
            }

            let rescued_ent_rate = c.entrapment as f64 / c.total as f64;
            let delta = rescued_ent_rate - baseline_ent_rate;

            log::info!(
                "DF physical rescue enrichment: {} entrapment_rate_delta_vs_all_rank1={:.4} baseline={:.4} rescued={:.4}",
                name,
                delta,
                baseline_ent_rate,
                rescued_ent_rate
            );

            if delta > 0.01 {
                log::warn!(
                    "DF physical rescue enrichment warning: {} is entrapment-enriched versus all_rank1 by {:.2} percentage points.",
                    name,
                    100.0 * delta
                );
            }
        }
    }
}

fn log_df_pvalue_bin_diagnostics(features: &[DfFeature], db: &IndexedDatabase) {
    const THRESHOLDS: [f64; 6] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2];

    for thr in THRESHOLDS {
        let mut counts = RescueCompositionCounts::default();

        for f in features.iter().filter(|f| f.core.rank == 1) {
            let Some(p) = f.decoy_free_p_value else {
                continue;
            };

            if p.is_finite() && (p as f64) <= thr {
                counts.observe(f, db);
            }
        }

        log::info!(
            "DF evidence separation: decoy_free_p_value<={:.0e} {}",
            thr,
            fmt_rescue_composition(counts)
        );
    }
}

type DfMetricFn = fn(&DfFeature) -> Option<f64>;

#[derive(Clone, Copy)]
struct DfMetricSpec {
    name: &'static str,
    higher_is_better: bool,
    value: DfMetricFn,
}

#[inline]
fn metric_hyperscore(f: &DfFeature) -> Option<f64> {
    f.core.hyperscore.is_finite().then_some(f.core.hyperscore)
}

#[inline]
fn metric_delta_next(f: &DfFeature) -> Option<f64> {
    f.core.delta_next.is_finite().then_some(f.core.delta_next)
}

#[inline]
fn metric_delta_best(f: &DfFeature) -> Option<f64> {
    f.core.delta_best.is_finite().then_some(f.core.delta_best)
}

#[inline]
fn metric_matched_peaks(f: &DfFeature) -> Option<f64> {
    Some(f.core.matched_peaks as f64)
}

#[inline]
fn metric_longest_b(f: &DfFeature) -> Option<f64> {
    Some(f.core.longest_b as f64)
}

#[inline]
fn metric_longest_y(f: &DfFeature) -> Option<f64> {
    Some(f.core.longest_y as f64)
}

#[inline]
fn metric_matched_intensity_pct(f: &DfFeature) -> Option<f64> {
    let x = f.core.matched_intensity_pct as f64;
    x.is_finite().then_some(x)
}

#[inline]
fn metric_ms2_intensity(f: &DfFeature) -> Option<f64> {
    let x = f.core.ms2_intensity as f64;
    x.is_finite().then_some(x)
}

#[inline]
fn metric_scored_candidates(f: &DfFeature) -> Option<f64> {
    Some(f.core.scored_candidates as f64)
}

fn quantile_sorted(vals: &[f64], q: f64) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }

    let q = q.clamp(0.0, 1.0);
    let pos = q * (vals.len().saturating_sub(1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;

    if lo == hi {
        Some(vals[lo])
    } else {
        let w = pos - lo as f64;
        Some(vals[lo] * (1.0 - w) + vals[hi] * w)
    }
}

fn count_metric_subset(
    features: &[DfFeature],
    db: &IndexedDatabase,
    metric: DfMetricFn,
    higher_is_better: bool,
    threshold: f64,
) -> RescueCompositionCounts {
    let mut counts = RescueCompositionCounts::default();

    for f in features.iter().filter(|f| f.core.rank == 1) {
        let Some(v) = metric(f) else {
            continue;
        };

        if !v.is_finite() {
            continue;
        }

        let pass = if higher_is_better {
            v >= threshold
        } else {
            v <= threshold
        };

        if pass {
            counts.observe(f, db);
        }
    }

    counts
}

fn log_spectral_feature_separation_diagnostics(features: &[DfFeature], db: &IndexedDatabase) {
    let metrics: [DfMetricSpec; 9] = [
        DfMetricSpec {
            name: "hyperscore",
            higher_is_better: true,
            value: metric_hyperscore,
        },
        DfMetricSpec {
            name: "delta_next",
            higher_is_better: true,
            value: metric_delta_next,
        },
        DfMetricSpec {
            name: "delta_best",
            higher_is_better: true,
            value: metric_delta_best,
        },
        DfMetricSpec {
            name: "matched_peaks",
            higher_is_better: true,
            value: metric_matched_peaks,
        },
        DfMetricSpec {
            name: "longest_b",
            higher_is_better: true,
            value: metric_longest_b,
        },
        DfMetricSpec {
            name: "longest_y",
            higher_is_better: true,
            value: metric_longest_y,
        },
        DfMetricSpec {
            name: "matched_intensity_pct",
            higher_is_better: true,
            value: metric_matched_intensity_pct,
        },
        DfMetricSpec {
            name: "ms2_intensity",
            higher_is_better: true,
            value: metric_ms2_intensity,
        },
        DfMetricSpec {
            name: "scored_candidates",
            higher_is_better: false,
            value: metric_scored_candidates,
        },
    ];

    for spec in metrics {
        let mut vals: Vec<f64> = features
            .iter()
            .filter(|f| f.core.rank == 1)
            .filter_map(|f| (spec.value)(f))
            .filter(|x| x.is_finite())
            .collect();

        if vals.len() < 100 {
            log::warn!(
                "DF spectral feature separation: metric={} skipped; finite_n={}",
                spec.name,
                vals.len()
            );
            continue;
        }

        vals.sort_by(|a, b| a.total_cmp(b));

        let quantiles: [(f64, &'static str); 3] = if spec.higher_is_better {
            [(0.90, "top10pct"), (0.95, "top5pct"), (0.99, "top1pct")]
        } else {
            [
                (0.10, "bottom10pct"),
                (0.05, "bottom5pct"),
                (0.01, "bottom1pct"),
            ]
        };

        for (q, label) in quantiles {
            let Some(thr) = quantile_sorted(&vals, q) else {
                continue;
            };

            let counts = count_metric_subset(features, db, spec.value, spec.higher_is_better, thr);

            log::info!(
                "DF spectral feature separation: metric={} subset={} threshold={:.6e} direction={} {}",
                spec.name,
                label,
                thr,
                if spec.higher_is_better { "higher" } else { "lower" },
                fmt_rescue_composition(counts)
            );
        }
    }
}

fn log_delta_next_focused_diagnostics(features: &[DfFeature], db: &IndexedDatabase) {
    const THRESHOLDS: [f64; 7] = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00];

    for thr in THRESHOLDS {
        let counts = count_metric_subset(features, db, metric_delta_next, true, thr);

        log::info!(
            "DF delta_next separation: delta_next>={:.2} {}",
            thr,
            fmt_rescue_composition(counts)
        );
    }
}

#[derive(Default)]
struct PeptideRunEvidence {
    runs: FnvHashSet<usize>,
    is_entrapment: bool,
    is_contaminant: bool,
    is_reference: bool,
}

fn fmt_run_count_bins(label: &str, bins: &[usize; 6]) -> String {
    let total: usize = bins.iter().sum();

    format!(
        "{} n={} runs1={}({:.1}%) runs2={}({:.1}%) runs3={}({:.1}%) runs4={}({:.1}%) runs5plus={}({:.1}%)",
        label,
        total,
        bins[1],
        pct_part(bins[1], total),
        bins[2],
        pct_part(bins[2], total),
        bins[3],
        pct_part(bins[3], total),
        bins[4],
        pct_part(bins[4], total),
        bins[5],
        pct_part(bins[5], total)
    )
}

fn log_peptide_recurrence_separation_diagnostics(features: &[DfFeature], db: &IndexedDatabase) {
    let mut peptide_runs: FnvHashMap<u32, PeptideRunEvidence> = FnvHashMap::default();

    for f in features.iter().filter(|f| f.core.rank == 1) {
        let proteins = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        let is_ent = is_entrapment_str(&proteins);
        let is_cont = is_contam_str(&proteins);
        let is_ref = f.core.label == 1 && !is_ent && !is_cont;

        let entry = peptide_runs.entry(f.core.peptide_idx.0).or_default();
        entry.runs.insert(f.core.file_id);
        entry.is_entrapment |= is_ent;
        entry.is_contaminant |= is_cont;
        entry.is_reference |= is_ref;
    }

    let mut target_bins = [0usize; 6];
    let mut ent_bins = [0usize; 6];
    let mut cont_bins = [0usize; 6];

    for ev in peptide_runs.values() {
        let n_runs = ev.runs.len().clamp(1, 5);

        if ev.is_reference && !ev.is_entrapment && !ev.is_contaminant {
            target_bins[n_runs] += 1;
        }

        if ev.is_entrapment {
            ent_bins[n_runs] += 1;
        }

        if ev.is_contaminant {
            cont_bins[n_runs] += 1;
        }
    }

    log::info!(
        "DF peptide recurrence separation: {}",
        fmt_run_count_bins("target_ref", &target_bins)
    );
    log::info!(
        "DF peptide recurrence separation: {}",
        fmt_run_count_bins("entrapment", &ent_bins)
    );
    log::info!(
        "DF peptide recurrence separation: {}",
        fmt_run_count_bins("contaminant", &cont_bins)
    );

    for min_runs in 2..=5 {
        let mut counts = RescueCompositionCounts::default();

        for f in features.iter().filter(|f| f.core.rank == 1) {
            let n = peptide_runs
                .get(&f.core.peptide_idx.0)
                .map(|ev| ev.runs.len())
                .unwrap_or(1);

            if n >= min_runs {
                counts.observe(f, db);
            }
        }

        log::info!(
            "DF peptide recurrence PSM enrichment: peptide_observed_in_{}plus_runs {}",
            min_runs,
            fmt_rescue_composition(counts)
        );
    }
}

fn log_mass_error_separation_diagnostics(features: &[DfFeature], db: &IndexedDatabase) {
    const PPM_THRESHOLDS: [f64; 6] = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0];

    for thr in PPM_THRESHOLDS {
        let mut precursor_counts = RescueCompositionCounts::default();
        let mut fragment_counts = RescueCompositionCounts::default();

        for f in features.iter().filter(|f| f.core.rank == 1) {
            let precursor_ppm_like = f.core.delta_mass as f64;
            if precursor_ppm_like.is_finite() && precursor_ppm_like.abs() <= thr {
                precursor_counts.observe(f, db);
            }

            let fragment_ppm_like = f.core.average_ppm as f64;
            if fragment_ppm_like.is_finite() && fragment_ppm_like.abs() <= thr {
                fragment_counts.observe(f, db);
            }
        }

        log::info!(
            "DF mass-error separation: abs_precursor_error<={:.1} {}",
            thr,
            fmt_rescue_composition(precursor_counts)
        );
        log::info!(
            "DF mass-error separation: abs_fragment_ppm<={:.1} {}",
            thr,
            fmt_rescue_composition(fragment_counts)
        );
    }
}

fn log_support_layer_separation_diagnostics(features: &[DfFeature], db: &IndexedDatabase) {
    let mut peptide_supported = RescueCompositionCounts::default();
    let mut protein_supported = RescueCompositionCounts::default();
    let mut peptide_q_1pct = RescueCompositionCounts::default();
    let mut protein_q_1pct = RescueCompositionCounts::default();

    for f in features.iter().filter(|f| f.core.rank == 1) {
        if f.decoy_free_peptide_supported_psm.unwrap_or(false) {
            peptide_supported.observe(f, db);
        }

        if f.decoy_free_protein_supported_peptide.unwrap_or(false) {
            protein_supported.observe(f, db);
        }

        if f.decoy_free_peptide_q.unwrap_or(1.0) <= 0.01 {
            peptide_q_1pct.observe(f, db);
        }

        if f.decoy_free_protein_q.unwrap_or(1.0) <= 0.01 {
            protein_q_1pct.observe(f, db);
        }
    }

    log::info!(
        "DF support-layer separation: peptide_supported_psm {}",
        fmt_rescue_composition(peptide_supported)
    );
    log::info!(
        "DF support-layer separation: protein_supported_peptide {}",
        fmt_rescue_composition(protein_supported)
    );
    log::info!(
        "DF support-layer separation: peptide_q<=0.01 {}",
        fmt_rescue_composition(peptide_q_1pct)
    );
    log::info!(
        "DF support-layer separation: protein_q<=0.01 {}",
        fmt_rescue_composition(protein_q_1pct)
    );
}

fn log_physical_sigma_diagnostics(features: &[DfFeature], db: &IndexedDatabase) {
    #[derive(Default)]
    struct Row {
        rt: SigmaBinCounts,
        ims: SigmaBinCounts,
    }

    let mut all_rank1 = Row::default();
    let mut target_ref = Row::default();
    let mut entrapment = Row::default();
    let mut accepted_psm_q_1pct = Row::default();
    let mut rescued_by_rt = Row::default();
    let mut rescued_by_ims = Row::default();
    let mut rescued_by_recurrence = Row::default();

    for f in features.iter().filter(|f| f.core.rank == 1) {
        let proteins = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        let is_ent = is_entrapment_str(&proteins);
        let is_cont = is_contam_str(&proteins);
        let is_ref = f.core.label == 1 && !is_ent && !is_cont;

        all_rank1.rt.observe(f.rt_z);
        all_rank1.ims.observe(f.ims_z);

        if is_ref {
            target_ref.rt.observe(f.rt_z);
            target_ref.ims.observe(f.ims_z);
        }

        if is_ent {
            entrapment.rt.observe(f.rt_z);
            entrapment.ims.observe(f.ims_z);
        }

        if f.decoy_free_q_value.unwrap_or(1.0) <= 0.01 {
            accepted_psm_q_1pct.rt.observe(f.rt_z);
            accepted_psm_q_1pct.ims.observe(f.ims_z);
        }

        if f.rescued_by_rt.unwrap_or(false) {
            rescued_by_rt.rt.observe(f.rt_z);
            rescued_by_rt.ims.observe(f.ims_z);
        }

        if f.rescued_by_ims.unwrap_or(false) {
            rescued_by_ims.rt.observe(f.rt_z);
            rescued_by_ims.ims.observe(f.ims_z);
        }

        if f.rescued_by_recurrence.unwrap_or(false) {
            rescued_by_recurrence.rt.observe(f.rt_z);
            rescued_by_recurrence.ims.observe(f.ims_z);
        }
    }

    log::info!("DF RT residual diagnostics:");
    log::info!(
        "DF RT residual diagnostics: all_rank1 {}",
        fmt_sigma_bins(all_rank1.rt)
    );
    log::info!(
        "DF RT residual diagnostics: target_ref {}",
        fmt_sigma_bins(target_ref.rt)
    );
    log::info!(
        "DF RT residual diagnostics: entrapment {}",
        fmt_sigma_bins(entrapment.rt)
    );
    log::info!(
        "DF RT residual diagnostics: accepted_psm_q_1pct {}",
        fmt_sigma_bins(accepted_psm_q_1pct.rt)
    );
    log::info!(
        "DF RT residual diagnostics: rescued_by_rt {}",
        fmt_sigma_bins(rescued_by_rt.rt)
    );
    log::info!(
        "DF RT residual diagnostics: rescued_by_ims {}",
        fmt_sigma_bins(rescued_by_ims.rt)
    );
    log::info!(
        "DF RT residual diagnostics: rescued_by_recurrence {}",
        fmt_sigma_bins(rescued_by_recurrence.rt)
    );

    log::info!("DF IMS residual diagnostics:");
    log::info!(
        "DF IMS residual diagnostics: all_rank1 {}",
        fmt_sigma_bins(all_rank1.ims)
    );
    log::info!(
        "DF IMS residual diagnostics: target_ref {}",
        fmt_sigma_bins(target_ref.ims)
    );
    log::info!(
        "DF IMS residual diagnostics: entrapment {}",
        fmt_sigma_bins(entrapment.ims)
    );
    log::info!(
        "DF IMS residual diagnostics: accepted_psm_q_1pct {}",
        fmt_sigma_bins(accepted_psm_q_1pct.ims)
    );
    log::info!(
        "DF IMS residual diagnostics: rescued_by_rt {}",
        fmt_sigma_bins(rescued_by_rt.ims)
    );
    log::info!(
        "DF IMS residual diagnostics: rescued_by_ims {}",
        fmt_sigma_bins(rescued_by_ims.ims)
    );
    log::info!(
        "DF IMS residual diagnostics: rescued_by_recurrence {}",
        fmt_sigma_bins(rescued_by_recurrence.ims)
    );

    let rt_d025 = sigma_rate_le025(target_ref.rt) - sigma_rate_le025(entrapment.rt);
    let rt_d050 = sigma_rate_le050(target_ref.rt) - sigma_rate_le050(entrapment.rt);
    let rt_d075 = sigma_rate_le075(target_ref.rt) - sigma_rate_le075(entrapment.rt);
    let rt_d1 = sigma_rate_le1(target_ref.rt) - sigma_rate_le1(entrapment.rt);
    let rt_d150 = sigma_rate_le150(target_ref.rt) - sigma_rate_le150(entrapment.rt);
    let rt_d2 = sigma_rate_le2(target_ref.rt) - sigma_rate_le2(entrapment.rt);
    let rt_d3 = sigma_rate_le3(target_ref.rt) - sigma_rate_le3(entrapment.rt);

    let ims_d025 = sigma_rate_le025(target_ref.ims) - sigma_rate_le025(entrapment.ims);
    let ims_d050 = sigma_rate_le050(target_ref.ims) - sigma_rate_le050(entrapment.ims);
    let ims_d075 = sigma_rate_le075(target_ref.ims) - sigma_rate_le075(entrapment.ims);
    let ims_d1 = sigma_rate_le1(target_ref.ims) - sigma_rate_le1(entrapment.ims);
    let ims_d150 = sigma_rate_le150(target_ref.ims) - sigma_rate_le150(entrapment.ims);
    let ims_d2 = sigma_rate_le2(target_ref.ims) - sigma_rate_le2(entrapment.ims);
    let ims_d3 = sigma_rate_le3(target_ref.ims) - sigma_rate_le3(entrapment.ims);

    log::info!(
        concat!(
            "DF physical target-entrapment separation: ",
            "rt_delta_le0.25σ={:.4} ",
            "rt_delta_le0.50σ={:.4} ",
            "rt_delta_le0.75σ={:.4} ",
            "rt_delta_le1σ={:.4} ",
            "rt_delta_le1.5σ={:.4} ",
            "rt_delta_le2σ={:.4} ",
            "rt_delta_le3σ={:.4} ",
            "ims_delta_le0.25σ={:.4} ",
            "ims_delta_le0.50σ={:.4} ",
            "ims_delta_le0.75σ={:.4} ",
            "ims_delta_le1σ={:.4} ",
            "ims_delta_le1.5σ={:.4} ",
            "ims_delta_le2σ={:.4} ",
            "ims_delta_le3σ={:.4}"
        ),
        rt_d025,
        rt_d050,
        rt_d075,
        rt_d1,
        rt_d150,
        rt_d2,
        rt_d3,
        ims_d025,
        ims_d050,
        ims_d075,
        ims_d1,
        ims_d150,
        ims_d2,
        ims_d3
    );

    if rt_d025.abs() < 0.02 && rt_d050.abs() < 0.02 && rt_d075.abs() < 0.02 && rt_d1.abs() < 0.02 {
        log::warn!(
            "DF RT separation warning: target_ref and entrapment RT residual distributions are very similar from <=0.25σ through <=1σ; rt_reliability reflects internal anchor/sigma stability, not target-entrapment discrimination."
        );
    }

    if ims_d025.abs() < 0.02
        && ims_d050.abs() < 0.02
        && ims_d075.abs() < 0.02
        && ims_d1.abs() < 0.02
    {
        log::warn!(
            "DF IMS separation warning: target_ref and entrapment IMS residual distributions are very similar from <=0.25σ through <=1σ; ims_reliability reflects internal anchor/sigma stability, not target-entrapment discrimination."
        );
    }
}

fn annotate_local_rt_training_guardrail(
    features: &mut [DfFeature],
    settings: &FdrSettings,
) -> usize {
    let min_n = settings
        .physical_rescue
        .min_anchor_count_per_run
        .max(settings.min_null_size)
        .max(10);

    let mut by_file_region: HashMap<(usize, i32), Vec<f64>> = HashMap::new();
    let mut by_file: HashMap<usize, Vec<f64>> = HashMap::new();

    for f in features.iter().filter(|f| f.core.rank == 1) {
        let Some(delta) = rt_residual_for_feature(f) else {
            continue;
        };

        let abs_delta = delta.abs();
        by_file.entry(f.core.file_id).or_default().push(abs_delta);

        if let Some(region) = rt_region_bin_for_feature(f, settings) {
            by_file_region
                .entry((f.core.file_id, region))
                .or_default()
                .push(abs_delta);
        }
    }

    let mut sigma_by_file_region: HashMap<(usize, i32), f64> = HashMap::new();
    for (key, vals) in by_file_region {
        if vals.len() >= min_n {
            if let Some(sig) = robust_sigma_from_abs_residuals(vals, 0.05, 0.25) {
                sigma_by_file_region.insert(key, sig);
            }
        }
    }

    let mut sigma_by_file: HashMap<usize, f64> = HashMap::new();
    for (key, vals) in by_file {
        if vals.len() >= min_n {
            if let Some(sig) = robust_sigma_from_abs_residuals(vals, 0.05, 0.25) {
                sigma_by_file.insert(key, sig);
            }
        }
    }

    let mut excluded = 0usize;
    let mut excluded_by_file_region: HashMap<(usize, i32), usize> = HashMap::new();

    for f in features.iter_mut().filter(|f| f.core.rank == 1) {
        let Some(delta) = rt_residual_for_feature(f) else {
            f.rt_training_eligible = Some(false);
            f.rt_local_outlier = Some(true);
            excluded += 1;
            continue;
        };

        let region = rt_region_bin_for_feature(f, settings);
        let sigma = region
            .and_then(|r| sigma_by_file_region.get(&(f.core.file_id, r)).copied())
            .or_else(|| sigma_by_file.get(&f.core.file_id).copied());

        let Some(sigma) = sigma else {
            // Fail open for sparse local bins: keep the PSM, but mark that no local
            // outlier decision was possible.
            f.rt_training_eligible = Some(true);
            f.rt_local_outlier = Some(false);
            continue;
        };

        let z = delta.abs() / sigma;
        let outlier = z > DF_RT_LOCAL_OUTLIER_Z;

        f.rt_local_z = Some(z as f32);
        f.rt_local_outlier = Some(outlier);
        f.rt_training_eligible = Some(!outlier);

        if outlier {
            excluded += 1;
            if let Some(r) = region {
                *excluded_by_file_region
                    .entry((f.core.file_id, r))
                    .or_default() += 1;
            }
        }
    }

    log::info!(
        "DF RT local guardrail: threshold_z={:.1} min_local_n={} excluded_psms={}",
        DF_RT_LOCAL_OUTLIER_Z,
        min_n,
        excluded
    );

    let mut rows: Vec<_> = excluded_by_file_region.into_iter().collect();
    rows.sort_by_key(|((file_id, region), _)| (*file_id, *region));
    for ((file_id, region), n) in rows {
        log::debug!(
            "DF RT local guardrail: file_id={} rt_region_bin={} excluded_psms={}",
            file_id,
            region,
            n
        );
    }

    excluded
}

#[inline]
fn rt_training_eligible(f: &DfFeature) -> bool {
    f.rt_training_eligible.unwrap_or(true) && !f.rt_local_outlier.unwrap_or(false)
}

fn aux_sigma_from_agreements(null_agreements: &[f64]) -> Option<f64> {
    let vals: Vec<f64> = null_agreements
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .collect();

    median_f64(vals).map(|m| (1.4826 * m).clamp(1e-6, 1.0))
}

#[inline]
fn aux_reliability_from_null_count(n: usize, settings: &FdrSettings) -> f64 {
    let min_n = settings.min_null_size.max(1) as f64;
    ((n as f64) / (2.0 * min_n)).clamp(0.0, 1.0)
}

fn build_rt_aux_null_model(
    features: &[DfFeature],
    settings: &FdrSettings,
    _db: &IndexedDatabase,
) -> Option<RtAuxNullModel> {
    let mut global: Vec<f64> = Vec::new();
    let mut by_file: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut by_region: HashMap<i32, Vec<f64>> = HashMap::new();
    let mut by_file_region: HashMap<(usize, i32), Vec<f64>> = HashMap::new();

    for f in features.iter() {
        if f.core.rank < settings.min_null_rank || f.core.rank > settings.max_null_rank {
            continue;
        }

        if !rt_training_eligible(f) {
            continue;
        }

        let Some(delta) = rt_delta_for_feature(f) else {
            continue;
        };

        let agreement = -delta.abs();

        global.push(agreement);
        by_file.entry(f.core.file_id).or_default().push(agreement);

        if let Some(region) = rt_region_bin_for_feature(f, settings) {
            by_region.entry(region).or_default().push(agreement);
            by_file_region
                .entry((f.core.file_id, region))
                .or_default()
                .push(agreement);
        }
    }

    if global.len() < settings.min_null_size {
        return None;
    }

    global.sort_by(|a, b| a.total_cmp(b));
    for vals in by_file.values_mut() {
        vals.sort_by(|a, b| a.total_cmp(b));
    }
    for vals in by_region.values_mut() {
        vals.sort_by(|a, b| a.total_cmp(b));
    }
    for vals in by_file_region.values_mut() {
        vals.sort_by(|a, b| a.total_cmp(b));
    }

    let rt_sigma_global = aux_sigma_from_agreements(&global);
    let reliability = aux_reliability_from_null_count(global.len(), settings);

    Some(RtAuxNullModel {
        global,
        by_file,
        by_region,
        by_file_region,
        rt_sigma_global,
        reliability,
    })
}

fn rt_aux_null_p_value(model: &RtAuxNullModel, settings: &FdrSettings, f: &DfFeature) -> f64 {
    let Some(delta_rt) = rt_delta_for_feature(f) else {
        return 1.0;
    };

    let agreement_obs = -delta_rt.abs();
    let min_n = settings.min_null_size;

    if let Some(region) = rt_region_bin_for_feature(f, settings) {
        if let Some(vals) = model.by_file_region.get(&(f.core.file_id, region)) {
            if vals.len() >= min_n {
                return aux_null_p_from_sorted_agreements(vals, agreement_obs);
            }
        }
    }

    if let Some(vals) = model.by_file.get(&f.core.file_id) {
        if vals.len() >= min_n {
            return aux_null_p_from_sorted_agreements(vals, agreement_obs);
        }
    }

    if let Some(region) = rt_region_bin_for_feature(f, settings) {
        if let Some(vals) = model.by_region.get(&region) {
            if vals.len() >= min_n {
                return aux_null_p_from_sorted_agreements(vals, agreement_obs);
            }
        }
    }

    aux_null_p_from_sorted_agreements(&model.global, agreement_obs)
}

fn apply_rt_null_pvalue_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> PhysicalRescueResult {
    let Some(model) = build_rt_aux_null_model(features, settings, db) else {
        return PhysicalRescueResult {
            enabled: true,
            fail_closed: true,
            ..Default::default()
        };
    };

    let is_unreliable = model.rt_sigma_global.is_none()
        || model.reliability < settings.physical_rescue.reliability_floor
        || model.global.len() < settings.min_null_size;

    if is_unreliable {
        log::warn!(
            "RT p-value rescue failed closed: null_n={} reliability={:.4} floor={:.4} sigma={:?}",
            model.global.len(),
            model.reliability,
            settings.physical_rescue.reliability_floor,
            model.rt_sigma_global
        );

        return PhysicalRescueResult {
            enabled: true,
            fail_closed: true,
            anchor_count_total: model.global.len(),
            anchor_count_after_filters: model.global.len(),
            rt_reliability: model.reliability,
            rt_sigma_global: model.rt_sigma_global,
            ..Default::default()
        };
    }

    for f in features.iter_mut().filter(|f| f.core.rank == 1) {
        let base_p = finite_df_p_value(f.decoy_free_p_value.unwrap_or(1.0));

        let p_rt = if rt_training_eligible(f) {
            rt_aux_null_p_value(&model, settings, f)
        } else {
            1.0
        };

        let p_new = finite_df_p_value(combine_cauchy(&[base_p, p_rt]));

        /*
         * Placeholder PEP is overwritten immediately below by
         * recalibrate_companion_pep_from_active_p_values(...).
         * The score is already correct because active evidence is p-value.
         */
        let placeholder_pep = finite_df_probability_for_logit(f.decoy_free_pep.unwrap_or(1.0));

        set_df_evidence_pair(f, ActiveEvidenceSpace::PValue, p_new, placeholder_pep);

        let rescued = p_new < base_p;
        f.rescued_by_rt = Some(rescued);
        if rescued {
            f.physical_rescue_source = Some("rt".to_string());
        }
    }

    recalibrate_companion_pep_from_active_p_values(features, settings, "RT p-value rescue");
    recalculate_active_q_values(features, settings);

    PhysicalRescueResult {
        enabled: true,
        fail_closed: false,
        anchor_count_total: model.global.len(),
        anchor_count_after_filters: model.global.len(),
        rt_reliability: model.reliability,
        ims_reliability: 0.0,
        joint_reliability: model.reliability,
        rt_sigma_global: model.rt_sigma_global,
        ims_sigma_global: None,
        dropped_runs: Vec::new(),
        dropped_charge_bins: Vec::new(),
    }
}

fn apply_rt_bounded_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> PhysicalRescueResult {
    use crate::input::PhysicalRescueMode;

    if matches!(settings.physical_rescue.rt_mode, PhysicalRescueMode::Off) {
        return PhysicalRescueResult {
            enabled: false,
            fail_closed: false,
            ..Default::default()
        };
    }

    let candidates = build_physical_anchor_set(features, settings, db);
    let anchor_count_total = candidates.len();

    let (safe_anchors, _) = exclude_non_rescue_safe_anchors(features, candidates, settings, db);
    let (run_vetted, dropped_runs) =
        filter_anchor_candidates_by_run(features, safe_anchors, settings);
    let (final_anchors, dropped_charge_bins) =
        filter_anchor_candidates_by_charge(features, run_vetted, settings);

    let rt_rel = compute_rt_reliability(features, &final_anchors, settings);

    let is_unreliable = final_anchors.len() < settings.physical_rescue.min_anchor_count_per_run
        || rt_rel.rt_sigma_global.is_none()
        || rt_rel.reliability < settings.physical_rescue.reliability_floor
        || rt_rel.fail_closed_hint;

    let rt_sigma = if is_unreliable {
        1.0
    } else {
        rt_rel.rt_sigma_global.unwrap_or(1.0)
    };

    let cfg = settings.physical_rescue.bounded_cfg.as_ref().expect(
        "Invalid DF config: RT bounded auxiliary adjustment requires physical_rescue.bounded_cfg.",
    );

    let mut rows_for_q: Vec<(f64, usize, f64)> = Vec::new();

    for (i, f) in features.iter_mut().enumerate() {
        if f.core.rank != 1 {
            continue;
        }

        f.physical_mode_used = Some(if is_unreliable {
            "rt_bounded_aux_fail_closed".to_string()
        } else {
            "rt_bounded_aux".to_string()
        });

        if is_unreliable {
            continue;
        }

        let prior_pep = f.decoy_free_pep.unwrap_or(1.0);

        let missing_rt = !f.core.aligned_rt.is_finite()
            || !f.core.predicted_rt.is_finite()
            || !f.core.delta_rt_model.is_finite();

        let missing_penalty = if missing_rt {
            settings.physical_rescue.missing_penalty.max(0.0)
        } else {
            0.0
        };

        let raw_shift = if missing_rt || !rt_training_eligible(f) {
            0.0
        } else {
            compute_physical_shift(f, rt_sigma)
        } - missing_penalty;

        let bounded_shift =
            crate::ml::stats::capped_shift(raw_shift, cfg.max_rescue_shift, cfg.max_penalty_shift);

        let posterior_pep = match cfg.update_space {
            BoundedAuxUpdateSpace::LogitConfidence => {
                let logit_prior = crate::ml::stats::safe_logit_confidence(prior_pep);
                let logit_post = logit_prior + bounded_shift;
                crate::ml::stats::safe_inv_logit_confidence(logit_post)
            }
        };

        f.decoy_free_pep = Some(finite_df_probability_for_logit(posterior_pep));

        let rescued = posterior_pep < prior_pep;
        f.rescued_by_rt = Some(rescued);
        if rescued {
            f.physical_rescue_source = Some("rt".to_string());
        }
        f.rt_rescue_delta = Some(bounded_shift as f32);

        let df_score = df_score_from_pep(posterior_pep);
        f.decoy_free_score = Some(df_score);

        rows_for_q.push((df_score, i, posterior_pep.clamp(0.0, 1.0).max(1e-300)));
    }

    if !is_unreliable {
        for (feat_idx, q) in q_from_pep_cummean(rows_for_q) {
            features[feat_idx].decoy_free_q_value = Some(finite_df_p_value(q));
        }
    }

    PhysicalRescueResult {
        enabled: true,
        fail_closed: is_unreliable,
        anchor_count_total,
        anchor_count_after_filters: final_anchors.len(),
        rt_reliability: rt_rel.reliability,
        ims_reliability: 0.0,
        joint_reliability: rt_rel.reliability,
        rt_sigma_global: rt_rel.rt_sigma_global,
        ims_sigma_global: None,
        dropped_runs,
        dropped_charge_bins,
    }
}

#[inline]
fn ims_delta_for_feature(f: &DfFeature) -> Option<f64> {
    let obs = f.core.ims as f64;
    let pred = f.core.predicted_ims as f64;

    if obs.is_finite() && pred.is_finite() {
        Some(obs - pred)
    } else {
        None
    }
}

struct ImsAuxNullModel {
    global: Vec<f64>,
    by_file: HashMap<usize, Vec<f64>>,
    by_charge: HashMap<i32, Vec<f64>>,
    by_file_charge: HashMap<(usize, i32), Vec<f64>>,
    ims_sigma_global: Option<f64>,
    reliability: f64,
}

fn build_ims_aux_null_model(
    features: &[DfFeature],
    settings: &FdrSettings,
    _db: &IndexedDatabase,
) -> Option<ImsAuxNullModel> {
    let mut global: Vec<f64> = Vec::new();
    let mut by_file: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut by_charge: HashMap<i32, Vec<f64>> = HashMap::new();
    let mut by_file_charge: HashMap<(usize, i32), Vec<f64>> = HashMap::new();

    for f in features.iter() {
        if f.core.rank < settings.min_null_rank || f.core.rank > settings.max_null_rank {
            continue;
        }

        let Some(delta) = ims_delta_for_feature(f) else {
            continue;
        };

        let agreement = -delta.abs();
        let charge = f.core.charge as i32;

        global.push(agreement);
        by_file.entry(f.core.file_id).or_default().push(agreement);
        by_charge.entry(charge).or_default().push(agreement);
        by_file_charge
            .entry((f.core.file_id, charge))
            .or_default()
            .push(agreement);
    }

    if global.len() < settings.min_null_size {
        return None;
    }

    global.sort_by(|a, b| a.total_cmp(b));
    for vals in by_file.values_mut() {
        vals.sort_by(|a, b| a.total_cmp(b));
    }
    for vals in by_charge.values_mut() {
        vals.sort_by(|a, b| a.total_cmp(b));
    }
    for vals in by_file_charge.values_mut() {
        vals.sort_by(|a, b| a.total_cmp(b));
    }

    let ims_sigma_global = aux_sigma_from_agreements(&global);
    let reliability = aux_reliability_from_null_count(global.len(), settings);

    Some(ImsAuxNullModel {
        global,
        by_file,
        by_charge,
        by_file_charge,
        ims_sigma_global,
        reliability,
    })
}

fn ims_aux_null_p_value(model: &ImsAuxNullModel, settings: &FdrSettings, f: &DfFeature) -> f64 {
    let Some(delta_ims) = ims_delta_for_feature(f) else {
        return 1.0;
    };

    let agreement_obs = -delta_ims.abs();
    let min_n = settings.min_null_size;
    let charge = f.core.charge as i32;

    if let Some(vals) = model.by_file_charge.get(&(f.core.file_id, charge)) {
        if vals.len() >= min_n {
            return aux_null_p_from_sorted_agreements(vals, agreement_obs);
        }
    }

    if let Some(vals) = model.by_file.get(&f.core.file_id) {
        if vals.len() >= min_n {
            return aux_null_p_from_sorted_agreements(vals, agreement_obs);
        }
    }

    if let Some(vals) = model.by_charge.get(&charge) {
        if vals.len() >= min_n {
            return aux_null_p_from_sorted_agreements(vals, agreement_obs);
        }
    }

    aux_null_p_from_sorted_agreements(&model.global, agreement_obs)
}

fn apply_ims_null_pvalue_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> PhysicalRescueResult {
    let Some(model) = build_ims_aux_null_model(features, settings, db) else {
        return PhysicalRescueResult {
            enabled: true,
            fail_closed: true,
            ..Default::default()
        };
    };

    let is_unreliable = model.ims_sigma_global.is_none()
        || model.reliability < settings.physical_rescue.reliability_floor
        || model.global.len() < settings.min_null_size;

    if is_unreliable {
        log::warn!(
            "IMS p-value rescue failed closed: null_n={} reliability={:.4} floor={:.4} sigma={:?}",
            model.global.len(),
            model.reliability,
            settings.physical_rescue.reliability_floor,
            model.ims_sigma_global
        );

        return PhysicalRescueResult {
            enabled: true,
            fail_closed: true,
            anchor_count_total: model.global.len(),
            anchor_count_after_filters: model.global.len(),
            ims_reliability: model.reliability,
            ims_sigma_global: model.ims_sigma_global,
            ..Default::default()
        };
    }

    for f in features.iter_mut().filter(|f| f.core.rank == 1) {
        let base_p = f.decoy_free_p_value.unwrap_or(1.0) as f64;

        let p_ims = ims_aux_null_p_value(&model, settings, f);
        let p_new = combine_cauchy(&[base_p, p_ims]).clamp(1e-300, 1.0);

        /*
         * Placeholder PEP is overwritten immediately below by
         * recalibrate_companion_pep_from_active_p_values(...).
         * The score is already correct because active evidence is p-value.
         */
        let placeholder_pep = f.decoy_free_pep.unwrap_or(1.0) as f64;

        set_df_evidence_pair(f, ActiveEvidenceSpace::PValue, p_new, placeholder_pep);

        let rescued = p_new < base_p;
        f.rescued_by_ims = Some(rescued);
        if rescued {
            f.physical_rescue_source = Some("ims".to_string());
        }
    }

    recalibrate_companion_pep_from_active_p_values(features, settings, "IMS p-value rescue");
    recalculate_active_q_values(features, settings);

    PhysicalRescueResult {
        enabled: true,
        fail_closed: false,
        anchor_count_total: model.global.len(),
        anchor_count_after_filters: model.global.len(),
        rt_reliability: 0.0,
        ims_reliability: model.reliability,
        joint_reliability: model.reliability,
        rt_sigma_global: None,
        ims_sigma_global: model.ims_sigma_global,
        dropped_runs: Vec::new(),
        dropped_charge_bins: Vec::new(),
    }
}

fn apply_ims_bounded_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> PhysicalRescueResult {
    use crate::input::PhysicalRescueMode;

    if matches!(settings.physical_rescue.ims_mode, PhysicalRescueMode::Off) {
        return PhysicalRescueResult {
            enabled: false,
            fail_closed: false,
            ..Default::default()
        };
    }

    let candidates = build_physical_anchor_set(features, settings, db);
    let anchor_count_total = candidates.len();

    let anchor_mode = &settings.physical_rescue.anchor_mode;
    let exclude_unsafe_proteins = !matches!(anchor_mode, PhysicalAnchorMode::EvidenceOnly);

    let mut safe_anchors = Vec::new();

    for idx in candidates.iter().copied() {
        let f = &features[idx];

        let ims_ok = f.core.ims.is_finite() && f.core.predicted_ims.is_finite();

        let protein_ok = if exclude_unsafe_proteins {
            let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            !is_entrapment_str(&prot) && !is_contam_str(&prot)
        } else {
            true
        };

        if ims_ok && protein_ok {
            safe_anchors.push(idx);
        }
    }

    let (run_vetted, dropped_runs) =
        filter_anchor_candidates_by_run(features, safe_anchors, settings);
    let (final_anchors, dropped_charge_bins) =
        filter_anchor_candidates_by_charge(features, run_vetted, settings);

    let ims_rel = compute_ims_reliability(features, &final_anchors, settings);

    let is_unreliable = final_anchors.len() < settings.physical_rescue.min_anchor_count_per_run
        || ims_rel.ims_sigma_global.is_none()
        || ims_rel.reliability < settings.physical_rescue.reliability_floor
        || ims_rel.fail_closed_hint;

    let ims_sigma = if is_unreliable {
        1.0
    } else {
        ims_rel.ims_sigma_global.unwrap_or(1.0)
    };

    let cfg = settings
        .physical_rescue
        .bounded_cfg
        .as_ref()
        .expect("BoundedAux config required for IMS");

    let mut rows_for_q: Vec<(f64, usize, f64)> = Vec::new();

    for (i, f) in features.iter_mut().enumerate() {
        if f.core.rank != 1 {
            continue;
        }

        f.physical_mode_used = Some(if is_unreliable {
            "ims_bounded_aux_fail_closed".to_string()
        } else {
            "ims_bounded_aux".to_string()
        });

        if is_unreliable {
            continue;
        }

        let prior_pep = f.decoy_free_pep.unwrap_or(1.0) as f64;

        let missing_ims = !f.core.ims.is_finite() || !f.core.predicted_ims.is_finite();

        let missing_penalty = if missing_ims {
            settings.physical_rescue.missing_penalty.max(0.0)
        } else {
            0.0
        };

        let raw_shift = if missing_ims {
            0.0
        } else {
            let delta = (f.core.ims - f.core.predicted_ims).abs() as f64;
            let z = delta / ims_sigma.max(1e-9);
            2.0 - z.powi(2)
        } - missing_penalty;

        let bounded_shift =
            crate::ml::stats::capped_shift(raw_shift, cfg.max_rescue_shift, cfg.max_penalty_shift);

        let posterior_pep = match cfg.update_space {
            BoundedAuxUpdateSpace::LogitConfidence => {
                let logit_prior = crate::ml::stats::safe_logit_confidence(prior_pep);
                let logit_post = logit_prior + bounded_shift;
                crate::ml::stats::safe_inv_logit_confidence(logit_post)
            }
        };

        f.decoy_free_pep = Some(finite_df_probability_for_logit(posterior_pep));

        let rescued = posterior_pep < prior_pep;
        f.rescued_by_ims = Some(rescued);
        if rescued {
            f.physical_rescue_source = Some("ims".to_string());
        }
        f.ims_rescue_delta = Some(bounded_shift as f32);

        let df_score = df_score_from_pep(posterior_pep);
        f.decoy_free_score = Some(df_score);

        rows_for_q.push((df_score, i, posterior_pep.clamp(0.0, 1.0).max(1e-300)));
    }

    if !is_unreliable {
        for (feat_idx, q) in q_from_pep_cummean(rows_for_q) {
            features[feat_idx].decoy_free_q_value = Some(finite_df_p_value(q));
        }
    }

    PhysicalRescueResult {
        enabled: true,
        fail_closed: is_unreliable,
        anchor_count_total,
        anchor_count_after_filters: final_anchors.len(),
        rt_reliability: 0.0,
        ims_reliability: ims_rel.reliability,
        joint_reliability: ims_rel.reliability,
        rt_sigma_global: None,
        ims_sigma_global: ims_rel.ims_sigma_global,
        dropped_runs,
        dropped_charge_bins,
    }
}

fn compute_dart_null_ims_params(features: &[DfFeature], anchors: &[usize]) -> (f64, f64) {
    let mut values: Vec<f64> = anchors
        .iter()
        .filter_map(|&i| {
            let x = features[i].core.ims as f64;
            x.is_finite().then_some(x)
        })
        .collect();

    if values.len() < 8 {
        values = features
            .iter()
            .filter(|f| f.core.rank == 1)
            .filter_map(|f| {
                let x = f.core.ims as f64;
                x.is_finite().then_some(x)
            })
            .collect();
    }

    if values.len() < 2 {
        return (0.0, 1.0);
    }

    let center = stats::mean(&values);
    let spread = stats::std_dev(&values).clamp(0.01, 0.25);

    (center, spread)
}

#[derive(Clone, Debug)]
struct ImsDartContext {
    pub anchors: Vec<usize>,
    pub ims_rel: ImsReliabilitySummary,
    pub is_unreliable: bool,
    pub ims_sigma: f64,
    pub null_ims_center: f64,
    pub null_ims_spread: f64,
    pub anchor_count_total: usize,
    pub dropped_runs: Vec<usize>,
    pub dropped_charge_bins: Vec<(i32, usize)>,
}

fn prepare_ims_dart_context(
    features: &[DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> ImsDartContext {
    let candidates = build_physical_anchor_set(features, settings, db);
    let anchor_count_total = candidates.len();

    let anchor_mode = &settings.physical_rescue.anchor_mode;
    let exclude_unsafe_proteins = !matches!(anchor_mode, PhysicalAnchorMode::EvidenceOnly);

    let mut safe_anchors = Vec::new();

    for idx in candidates.iter().copied() {
        let f = &features[idx];

        let ims_ok = f.core.ims.is_finite() && f.core.predicted_ims.is_finite();

        let protein_ok = if exclude_unsafe_proteins {
            let prot = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            !is_entrapment_str(&prot) && !is_contam_str(&prot)
        } else {
            true
        };

        if ims_ok && protein_ok {
            safe_anchors.push(idx);
        }
    }

    let (run_vetted, dropped_runs) =
        filter_anchor_candidates_by_run(features, safe_anchors, settings);

    let (final_anchors, dropped_charge_bins) =
        filter_anchor_candidates_by_charge(features, run_vetted, settings);

    let ims_rel = compute_ims_reliability(features, &final_anchors, settings);

    let is_unreliable = final_anchors.len() < settings.physical_rescue.min_anchor_count_per_run
        || ims_rel.ims_sigma_global.is_none()
        || ims_rel.reliability < settings.physical_rescue.reliability_floor
        || ims_rel.fail_closed_hint;

    let ims_sigma = if is_unreliable {
        1.0
    } else {
        ims_rel.ims_sigma_global.unwrap_or(1.0)
    };

    let (null_ims_center, null_ims_spread) = compute_dart_null_ims_params(features, &final_anchors);

    ImsDartContext {
        anchors: final_anchors,
        ims_rel,
        is_unreliable,
        ims_sigma,
        null_ims_center,
        null_ims_spread,
        anchor_count_total,
        dropped_runs,
        dropped_charge_bins,
    }
}

fn compute_dart_true_ims_likelihood(
    observed_ims: f64,
    reference_ims: f64,
    ims_sigma: f64,
    model_type: &DartTrueRtModel,
) -> f64 {
    match model_type {
        DartTrueRtModel::Laplace => {
            crate::ml::stats::laplace_logpdf(observed_ims, reference_ims, ims_sigma)
        }
        DartTrueRtModel::Normal => {
            crate::ml::stats::normal_logpdf(observed_ims, reference_ims, ims_sigma)
        }
    }
}

fn compute_dart_null_ims_likelihood(
    observed_ims: f64,
    null_center: f64,
    null_spread: f64,
    null_model_type: &DartNullRtModel,
) -> f64 {
    match null_model_type {
        DartNullRtModel::Normal => {
            crate::ml::stats::normal_logpdf(observed_ims, null_center, null_spread)
        }
        DartNullRtModel::Uniform => 0.0,
    }
}

fn apply_ims_dart_bayes_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> PhysicalRescueResult {
    let dart_cfg = settings
        .physical_rescue
        .dart_cfg
        .as_ref()
        .expect("DART-Bayes config required when physical_rescue.ims_mode='dart_bayes'");

    let ims_ctx = prepare_ims_dart_context(features, settings, db);
    let is_unreliable = ims_ctx.is_unreliable;

    let snapshot = features.to_vec();

    let mut peptide_ims_map: FnvHashMap<u32, Vec<f64>> = FnvHashMap::default();

    if !is_unreliable && dart_cfg.dart_use_bootstrap {
        for a in snapshot.iter() {
            if a.core.ims.is_finite() && a.core.predicted_ims.is_finite() {
                peptide_ims_map
                    .entry(a.core.peptide_idx.0)
                    .or_default()
                    .push(a.core.ims as f64);
            }
        }
    }

    let empty_ims = Vec::new();

    for f in features.iter_mut() {
        if f.core.rank != 1 {
            continue;
        }

        f.physical_mode_used = Some(
            if is_unreliable {
                "ims_dart_bayes_fail_closed"
            } else {
                "ims_dart_bayes"
            }
            .to_string(),
        );

        if is_unreliable {
            f.dart_posterior_used = Some(false);
            continue;
        }

        let observed_ims = f.core.ims as f64;
        let predicted_ims = f.core.predicted_ims as f64;

        if !observed_ims.is_finite() || !predicted_ims.is_finite() {
            f.dart_posterior_used = Some(false);
            f.physical_mode_used = Some("ims_dart_bayes_fail_closed:missing-ims".to_string());
            continue;
        }

        let peptide_ims = peptide_ims_map
            .get(&f.core.peptide_idx.0)
            .unwrap_or(&empty_ims);

        let mut reference_ims = predicted_ims;
        let mut reference_sigma = ims_ctx.ims_sigma.clamp(0.01, 0.25);

        if dart_cfg.dart_use_bootstrap && peptide_ims.len() > 1 && dart_cfg.dart_bootstrap_iters > 0
        {
            let iters = dart_cfg.dart_bootstrap_iters;
            let mut prng = FastRng((f.core.spec_id.len() + f.core.file_id * 7331) as u64 + 211);
            let mut boot_mus = Vec::with_capacity(iters);

            for _ in 0..iters {
                let mut sample = Vec::with_capacity(peptide_ims.len());

                for _ in 0..peptide_ims.len() {
                    sample.push(peptide_ims[prng.next_usize(peptide_ims.len())]);
                }

                let mu_b = match dart_cfg.dart_bootstrap_method {
                    crate::input::DartBootstrapMethod::None
                    | crate::input::DartBootstrapMethod::NonParametric => {
                        aggregate_mu(&mut sample, &dart_cfg.dart_mu_estimation)
                    }

                    crate::input::DartBootstrapMethod::Parametric
                    | crate::input::DartBootstrapMethod::ParametricMixture => {
                        let mut weights = Vec::with_capacity(sample.len());

                        for &ims_cand in &sample {
                            let log_lik_true = compute_dart_true_ims_likelihood(
                                ims_cand,
                                predicted_ims,
                                reference_sigma,
                                &dart_cfg.dart_true_rt_model,
                            );

                            let log_lik_null = compute_dart_null_ims_likelihood(
                                ims_cand,
                                ims_ctx.null_ims_center,
                                ims_ctx.null_ims_spread,
                                &dart_cfg.dart_null_rt_model,
                            );

                            let weight = if dart_cfg.dart_bootstrap_method
                                == crate::input::DartBootstrapMethod::ParametricMixture
                            {
                                let p_true = log_lik_true.exp();
                                let p_null = log_lik_null.exp();
                                p_true / (p_true + p_null + 1e-300)
                            } else {
                                1.0
                            };

                            weights.push(weight);
                        }

                        aggregate_weighted_mu(&mut sample, &weights, &dart_cfg.dart_mu_estimation)
                    }
                };

                boot_mus.push(mu_b);
            }

            reference_ims = aggregate_mu(&mut boot_mus, &dart_cfg.dart_mu_estimation);

            if boot_mus.len() > 1 {
                let mean = boot_mus.iter().sum::<f64>() / boot_mus.len() as f64;
                let var = boot_mus.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (boot_mus.len() - 1) as f64;

                let boot_sigma = var.sqrt().clamp(0.01, 0.25);
                let shrink = settings.physical_rescue.cov_shrinkage.clamp(0.0, 1.0);

                reference_sigma =
                    ((1.0 - shrink) * boot_sigma + shrink * reference_sigma).clamp(0.01, 0.25);
            }
        }

        if !reference_ims.is_finite() || !reference_sigma.is_finite() || reference_sigma <= 0.0 {
            f.dart_posterior_used = Some(false);
            f.physical_mode_used = Some("ims_dart_bayes_fail_closed:invalid-reference".to_string());
            continue;
        }

        let prior_pep = f.decoy_free_pep.unwrap_or(1.0) as f64;

        let log_lik_true = compute_dart_true_ims_likelihood(
            observed_ims,
            reference_ims,
            reference_sigma,
            &dart_cfg.dart_true_rt_model,
        );

        let log_lik_null = compute_dart_null_ims_likelihood(
            observed_ims,
            ims_ctx.null_ims_center,
            ims_ctx.null_ims_spread,
            &dart_cfg.dart_null_rt_model,
        );

        let posterior_pep = compute_dart_posterior_pep(prior_pep, log_lik_true, log_lik_null)
            .clamp(0.0, 1.0)
            .max(1e-300);

        f.decoy_free_pep = Some(finite_df_probability_for_logit(posterior_pep));
        f.decoy_free_score = Some(df_score_from_pep(posterior_pep));

        f.dart_posterior_used = Some(true);

        // Existing TSV fields are RT-named, but they are generic DART likelihood
        // diagnostics in this implementation.
        f.dart_rt_lik_correct = Some(log_lik_true as f32);
        f.dart_rt_lik_incorrect = Some(log_lik_null as f32);
    }

    if !is_unreliable && dart_cfg.dart_recalc_q_from_posterior {
        recalculate_active_q_values(features, settings);
    }

    PhysicalRescueResult {
        enabled: true,
        fail_closed: is_unreliable,
        anchor_count_total: ims_ctx.anchor_count_total,
        anchor_count_after_filters: ims_ctx.anchors.len(),
        rt_reliability: 0.0,
        ims_reliability: ims_ctx.ims_rel.reliability,
        joint_reliability: ims_ctx.ims_rel.reliability,
        rt_sigma_global: None,
        ims_sigma_global: ims_ctx.ims_rel.ims_sigma_global,
        dropped_runs: ims_ctx.dropped_runs,
        dropped_charge_bins: ims_ctx.dropped_charge_bins,
    }
}

fn apply_rt_confidence_adjustment(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> (PhysicalRescueResult, DfStageOutcome) {
    if !settings.enable_rt_confidence_adjustment {
        return (
            PhysicalRescueResult {
                enabled: false,
                ..Default::default()
            },
            DfStageOutcome::Skipped,
        );
    }

    let res = apply_rt_only_physical_update_to_active_stream(features, settings, db);

    let outcome = if res.fail_closed || res.anchor_count_total == 0 {
        DfStageOutcome::FailedClosed
    } else {
        finalize_stage_snapshot(features, settings, DfAdjustmentStage::Rt);
        DfStageOutcome::Applied
    };

    (res, outcome)
}

fn apply_ims_confidence_adjustment(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> (PhysicalRescueResult, DfStageOutcome) {
    if !settings.enable_ims_confidence_adjustment {
        return (
            PhysicalRescueResult {
                enabled: false,
                ..Default::default()
            },
            DfStageOutcome::Skipped,
        );
    }

    let res = apply_ims_only_physical_update_to_active_stream(features, settings, db);

    let outcome = if res.fail_closed || res.anchor_count_total == 0 {
        DfStageOutcome::FailedClosed
    } else {
        finalize_stage_snapshot(features, settings, DfAdjustmentStage::Ims);
        DfStageOutcome::Applied
    };

    (res, outcome)
}

fn apply_protein_recurrence_null_pvalue_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> ReproducibilityResult {
    let protein_support_map = build_protein_support_map(features, db, settings);

    let n_rescue_eligible_proteins = protein_support_map
        .values()
        .filter(|s| s.is_rescue_eligible)
        .count();

    if n_rescue_eligible_proteins == 0 {
        log::warn!(
            "DF protein reproducibility rescue skipped in p-value mode: eligible_proteins=0; leaving active stream unchanged."
        );
    } else {
        log::warn!(
            "DF protein reproducibility rescue skipped in p-value mode: eligible_proteins={} but protein-specific p-value recurrence rescue is not implemented; leaving active stream unchanged.",
            n_rescue_eligible_proteins
        );
    }

    for f in features.iter_mut().filter(|f| f.core.rank == 1) {
        if f.rescued_by_recurrence.is_none() {
            f.rescued_by_recurrence = Some(false);
        }
    }

    ReproducibilityResult {
        enabled: true,
        fail_closed: false,
        n_rescue_eligible_proteins,
        n_rescue_eligible_peptides: 0,
        n_anchor_peptides: 0,
        n_rescued_psms: 0,
        n_strong_unchanged_psms: 0,
        n_too_weak_unrescued_psms: 0,
        agreement_support_mean: 0.0,
        max_shift_applied: 0.0,
    }
}

fn apply_peptide_reproducibility_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> ReproducibilityResult {
    match active_evidence_space(settings) {
        ActiveEvidenceSpace::Pep => apply_bounded_repro_shift(features, settings, db),

        ActiveEvidenceSpace::PValue => {
            apply_recurrence_null_pvalue_update_to_active_stream(features, settings, db)
        }
    }
}

fn apply_protein_reproducibility_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> ReproducibilityResult {
    match active_evidence_space(settings) {
        ActiveEvidenceSpace::Pep => apply_bounded_repro_shift(features, settings, db),

        ActiveEvidenceSpace::PValue => {
            apply_protein_recurrence_null_pvalue_update_to_active_stream(features, settings, db)
        }
    }
}

fn apply_peptide_reproducibility_rescue(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> (ReproducibilityResult, DfStageOutcome) {
    if !settings.enable_peptide_reproducibility_rescue {
        return (
            ReproducibilityResult {
                enabled: false,
                ..Default::default()
            },
            DfStageOutcome::Skipped,
        );
    }

    let res = apply_peptide_reproducibility_update_to_active_stream(features, settings, db);

    let outcome = if res.fail_closed {
        DfStageOutcome::FailedClosed
    } else if res.n_rescued_psms == 0 {
        DfStageOutcome::Skipped
    } else {
        finalize_stage_snapshot(features, settings, DfAdjustmentStage::PeptideRescue);
        DfStageOutcome::Applied
    };

    (res, outcome)
}

fn apply_protein_reproducibility_rescue(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> (ReproducibilityResult, DfStageOutcome) {
    if !settings.enable_protein_reproducibility_rescue {
        return (
            ReproducibilityResult {
                enabled: false,
                ..Default::default()
            },
            DfStageOutcome::Skipped,
        );
    }

    let res = apply_protein_reproducibility_update_to_active_stream(features, settings, db);

    let outcome = if res.fail_closed {
        DfStageOutcome::FailedClosed
    } else if res.n_rescued_psms == 0 {
        DfStageOutcome::Skipped
    } else {
        finalize_stage_snapshot(features, settings, DfAdjustmentStage::ProteinRescue);
        DfStageOutcome::Applied
    };

    (res, outcome)
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
// Legacy DART-Bayes physical evidence kernels
// =============================================================================

fn compute_dart_reference_rt(
    f: &DfFeature,
    features: &[DfFeature],
    settings: &FdrSettings,
    phys_ctx: &PhysicalContext,
    peptide_rts: &[f64],
) -> (f64, f64, bool) {
    use crate::input::DartBootstrapMethod;
    let dart_cfg = settings
        .physical_rescue
        .dart_cfg
        .as_ref()
        .expect("DART-Bayes config missing");

    let predicted_rt = f.core.predicted_rt as f64;
    let observed_rt = f.core.aligned_rt as f64;
    let delta_rt = f.core.delta_rt_model as f64;

    let mut reference_rt = if observed_rt.is_finite() && delta_rt.is_finite() {
        (observed_rt - delta_rt).clamp(0.0, 1.0)
    } else {
        predicted_rt
    };

    let mut reference_rt_sigma =
        compute_dart_bootstrap_uncertainty(f, features, settings, phys_ctx);

    if dart_cfg.dart_use_bootstrap && peptide_rts.len() > 1 && dart_cfg.dart_bootstrap_iters > 0 {
        let iters = dart_cfg.dart_bootstrap_iters;
        let mut prng = FastRng((f.core.spec_id.len() + f.core.file_id * 1337) as u64 + 101);
        let mut boot_mus = Vec::with_capacity(iters);

        for _ in 0..iters {
            let mut sample = Vec::with_capacity(peptide_rts.len());
            for _ in 0..peptide_rts.len() {
                sample.push(peptide_rts[prng.next_usize(peptide_rts.len())]);
            }

            let mu_b = match dart_cfg.dart_bootstrap_method {
                DartBootstrapMethod::None | DartBootstrapMethod::NonParametric => {
                    aggregate_mu(&mut sample, &dart_cfg.dart_mu_estimation)
                }
                DartBootstrapMethod::Parametric | DartBootstrapMethod::ParametricMixture => {
                    let mut weights = Vec::with_capacity(sample.len());
                    for &rt_cand in &sample {
                        let log_lik_true = compute_dart_true_rt_likelihood(
                            rt_cand,
                            predicted_rt,
                            reference_rt_sigma,
                            &dart_cfg.dart_true_rt_model,
                        );
                        let log_lik_null = compute_dart_null_rt_likelihood(
                            rt_cand,
                            phys_ctx.null_rt_center,
                            phys_ctx.null_rt_spread,
                            &dart_cfg.dart_null_rt_model,
                        );

                        let weight = if dart_cfg.dart_bootstrap_method
                            == DartBootstrapMethod::ParametricMixture
                        {
                            let p_true = log_lik_true.exp();
                            let p_null = log_lik_null.exp();
                            p_true / (p_true + p_null + 1e-300)
                        } else {
                            1.0
                        };
                        weights.push(weight);
                    }
                    aggregate_weighted_mu(&mut sample, &weights, &dart_cfg.dart_mu_estimation)
                }
            };
            boot_mus.push(mu_b);
        }

        reference_rt = aggregate_mu(&mut boot_mus, &dart_cfg.dart_mu_estimation).clamp(0.0, 1.0);

        if boot_mus.len() > 1 {
            let mean = boot_mus.iter().sum::<f64>() / boot_mus.len() as f64;
            let var = boot_mus.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (boot_mus.len() - 1) as f64;
            let boot_sigma = var.sqrt().clamp(0.02, 0.30);
            let shrink = settings.physical_rescue.cov_shrinkage;
            reference_rt_sigma =
                ((1.0 - shrink) * boot_sigma + shrink * reference_rt_sigma).clamp(0.02, 0.30);
        }
    }

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

struct FastRng(u64);
impl FastRng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 21;
        self.0 ^= self.0 >> 35;
        self.0 ^= self.0 << 4;
        (self.0 as f64) / (u64::MAX as f64)
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_f64() * (max as f64)) as usize
    }
}

fn aggregate_mu(vals: &mut [f64], method: &crate::input::DartMuEstimation) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    match method {
        crate::input::DartMuEstimation::Mean | crate::input::DartMuEstimation::WeightedMean => {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
        crate::input::DartMuEstimation::Median => {
            vals.sort_by(|a, b| a.total_cmp(b));
            let mid = vals.len() / 2;
            if vals.len() % 2 == 0 {
                (vals[mid - 1] + vals[mid]) / 2.0
            } else {
                vals[mid]
            }
        }
    }
}

fn aggregate_weighted_mu(
    vals: &mut [f64],
    weights: &[f64],
    method: &crate::input::DartMuEstimation,
) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let sum_w: f64 = weights.iter().sum();
    if sum_w <= 1e-12 {
        return aggregate_mu(vals, &crate::input::DartMuEstimation::Median);
    }
    match method {
        crate::input::DartMuEstimation::Mean | crate::input::DartMuEstimation::WeightedMean => {
            vals.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / sum_w
        }
        crate::input::DartMuEstimation::Median => {
            let mut pairs: Vec<(f64, f64)> =
                vals.iter().copied().zip(weights.iter().copied()).collect();
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut cum = 0.0;
            let target = sum_w / 2.0;
            for (v, w) in pairs {
                cum += w;
                if cum >= target {
                    return v;
                }
            }
            vals.last().copied().unwrap_or(0.0)
        }
    }
}

fn compute_dart_bootstrap_uncertainty(
    f: &DfFeature,
    features: &[DfFeature],
    settings: &FdrSettings,
    phys_ctx: &PhysicalContext,
) -> f64 {
    let dart_cfg = settings
        .physical_rescue
        .dart_cfg
        .as_ref()
        .expect("DART-Bayes config missing");

    let global_sigma = phys_ctx.rt_sigma.clamp(0.02, 0.30);

    if !settings.physical_rescue.use_local_rt_scale && !dart_cfg.dart_use_bootstrap {
        return global_sigma;
    }

    let target_pred_rt = f.core.predicted_rt as f64;
    if !target_pred_rt.is_finite() {
        return global_sigma;
    }

    let bins = settings.physical_rescue.rt_region_bins.max(2) as f64;
    let half_width = (1.0 / bins).clamp(0.02, 0.25);

    let mut local_abs_deltas: Vec<f64> = phys_ctx
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

fn apply_dart_bayes_update(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    phys_ctx: &PhysicalContext,
) {
    let is_unreliable = phys_ctx.is_unreliable;

    let dart_cfg = settings
        .physical_rescue
        .dart_cfg
        .as_ref()
        .expect("DART-Bayes config must be provided");

    if is_unreliable {
        for f in features.iter_mut().filter(|f| f.core.rank == 1) {
            f.physical_mode_used = Some("dart_bayes_fail_closed".to_string());
            f.dart_posterior_used = Some(false);
        }
        return;
    }

    let null_center = phys_ctx.null_rt_center;
    let null_spread = phys_ctx.null_rt_spread;

    let snapshot = features.to_vec();

    let mut pep_rt_map: FnvHashMap<u32, Vec<f64>> = FnvHashMap::default();
    if dart_cfg.dart_use_bootstrap {
        for a in snapshot.iter() {
            if a.core.aligned_rt.is_finite() && a.core.delta_rt_model.is_finite() {
                pep_rt_map.entry(a.core.peptide_idx.0).or_default().push(
                    (a.core.aligned_rt as f64 - a.core.delta_rt_model as f64).clamp(0.0, 1.0),
                );
            }
        }
    }
    let empty_rts = Vec::new();

    for f in features.iter_mut() {
        if f.core.rank != 1 {
            continue;
        }

        f.physical_mode_used = Some("dart_bayes".to_string());

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

        let peptide_rts = pep_rt_map.get(&f.core.peptide_idx.0).unwrap_or(&empty_rts);

        let (reference_rt, reference_sigma, reference_valid) =
            compute_dart_reference_rt(f, &snapshot, settings, phys_ctx, peptide_rts);

        if !rt_ok || !reference_valid {
            f.dart_posterior_used = Some(false);
            if let Some(reason) = reason {
                f.physical_mode_used = Some(format!("dart_bayes_fail_closed:{reason}"));
            }
            continue;
        }

        let prior_pep = f.decoy_free_pep.unwrap_or(1.0) as f64;

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

        f.decoy_free_pep = Some(finite_df_probability_for_logit(posterior_pep));
        f.decoy_free_score = Some(df_score_from_pep(posterior_pep));

        f.dart_posterior_used = Some(true);
        f.dart_rt_lik_correct = Some(log_lik_true as f32);
        f.dart_rt_lik_incorrect = Some(log_lik_null as f32);
    }

    if dart_cfg.dart_recalc_q_from_posterior {
        recalculate_active_q_values(features, settings);
    }
}

// =============================================================================
// Bounded auxiliary physical evidence kernels
// =============================================================================

/// Compute a bounded RT/IMS auxiliary shift on the logit-confidence scale.
///
/// Positive values increase confidence, negative values decrease confidence.
/// The returned value is a bounded logit-space shift, not a probability-space
/// increment. Conversion back to PEP/confidence space happens downstream.

fn compute_physical_shift(f: &DfFeature, rt_sigma: f64) -> f64 {
    let delta = f.core.delta_rt_model as f64;
    let predicted = f.core.predicted_rt as f64;

    // Missing or invalid physical evidence is neutral. It must not create either a
    // rescue bonus or a demotion penalty.
    if !delta.is_finite() || !predicted.is_finite() || predicted <= 0.0 || predicted >= 1.0 {
        return 0.0;
    }

    let z = delta / rt_sigma.max(1e-9);

    // Quadratic logit-confidence shift: tight agreement with the physical model
    // supports rescue, while large deviations support demotion. The exact positive
    // and negative bounds are enforced downstream by the configured caps.
    2.0 - z.powi(2)
}

fn apply_bounded_physical_shift(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    phys_ctx: &PhysicalContext,
) {
    let is_unreliable = phys_ctx.is_unreliable;
    let rt_sigma = phys_ctx.rt_sigma;

    let cfg = settings
        .physical_rescue
        .bounded_cfg
        .as_ref()
        .expect("BoundedAux config must be provided");

    if is_unreliable {
        for f in features.iter_mut().filter(|f| f.core.rank == 1) {
            f.physical_mode_used = Some("bounded_aux_fail_closed".to_string());
        }
        return;
    }

    for f in features.iter_mut() {
        if f.core.rank != 1 {
            continue;
        }

        f.physical_mode_used = Some("bounded_aux".to_string());

        let prior_pep = f.decoy_free_pep.unwrap_or(1.0) as f64;

        let missing_rt = !f.core.aligned_rt.is_finite()
            || !f.core.predicted_rt.is_finite()
            || !f.core.delta_rt_model.is_finite();

        let missing_ims = settings.enable_ims_confidence_adjustment
            && (!f.core.ims.is_finite() || !f.core.predicted_ims.is_finite());

        let missing_penalty = if missing_rt || missing_ims {
            settings.physical_rescue.missing_penalty.max(0.0)
        } else {
            0.0
        };

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

        f.decoy_free_pep = Some(finite_df_probability_for_logit(posterior_pep));
        let df_score = df_score_from_pep(posterior_pep);
        f.decoy_free_score = Some(df_score);

        f.physical_shift_total = Some(bounded_shift as f32);

        if bounded_shift > 0.0 && (cfg.max_rescue_shift - bounded_shift).abs() < 0.05 {
            f.physical_cap_hit_pos = Some(true);
        } else if bounded_shift < 0.0 && (cfg.max_penalty_shift - (-bounded_shift)).abs() < 0.05 {
            f.physical_cap_hit_neg = Some(true);
        }
    }

    recalculate_active_q_values(features, settings);
}

#[derive(Clone, Debug, Default)]
struct ProteinSupportSummary {
    pub n_unique_observed: usize,
    pub n_unique_passing_prior: usize,
    pub is_rescue_eligible: bool,
}

#[derive(Clone, Debug, Default)]
struct PeptideEligibilitySummary {
    pub n_runs_observed: usize,
    pub n_runs_strong_prior: usize,
    pub observed_run_fraction: f64,
    pub strong_run_fraction: f64,
    pub is_rescue_eligible: bool,
    pub protein_rescue_eligible: bool,
}

#[derive(Clone, Debug, Default)]
struct ReproducibilityAnchorSummary {
    pub anchor_value: f64,
    pub n_anchor_observations: usize,
}

// =============================================================================
// Optional reproducibility rescue: peptide/protein recurrence and expert agreement
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
    elig: &PeptideEligibilitySummary,
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

    let raw_support = observed_excess + 2.0 * strong_excess;

    crate::ml::stats::soft_cap(raw_support, settings.reproducibility.max_recurrence_shift)
}

fn compute_redundancy_discount(_f: &DfFeature, settings: &FdrSettings) -> f64 {
    settings.reproducibility.redundancy_discount.clamp(0.0, 1.0)
}

fn build_protein_support_map(
    features: &[DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
) -> FnvHashMap<String, ProteinSupportSummary> {
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

        if f.decoy_free_q_value
            .map(|q| q <= cfg.q_threshold_physical)
            .unwrap_or(false)
        {
            passing.entry(protein_key).or_default().insert(peptide_seq);
        }
    }

    let mut out: FnvHashMap<String, ProteinSupportSummary> = FnvHashMap::default();

    for (protein_key, obs_set) in observed {
        let n_unique_observed = obs_set.len();
        let n_unique_passing_prior = passing.get(&protein_key).map(|s| s.len()).unwrap_or(0);

        let frac_ok = match cfg.min_unique_passing_fraction {
            Some(frac) if n_unique_observed > 0 => {
                (n_unique_passing_prior as f64) / (n_unique_observed as f64) >= frac
            }
            Some(_) => false,
            None => true,
        };

        let is_rescue_eligible = if cfg.enabled {
            n_unique_passing_prior >= cfg.min_unique_passing_peptides && frac_ok
        } else {
            true
        };

        out.insert(
            protein_key,
            ProteinSupportSummary {
                n_unique_observed,
                n_unique_passing_prior,
                is_rescue_eligible,
            },
        );
    }

    out
}

fn build_peptide_eligibility_map(
    features: &[DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
    protein_support_map: &FnvHashMap<String, ProteinSupportSummary>,
) -> FnvHashMap<u32, PeptideEligibilitySummary> {
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
            .decoy_free_q_value
            .map(|q| q <= cfg.strong_reference_q_threshold_physical as f64)
            .unwrap_or(false);

        let pep_ok = match cfg.strong_reference_pep_threshold_physical {
            Some(thr) => f.decoy_free_pep.map(|p| p <= thr).unwrap_or(false),
            None => true,
        };

        if q_ok && pep_ok {
            strong_runs
                .entry(pep_id)
                .or_default()
                .insert(f.core.file_id);
        }
    }

    let mut out: FnvHashMap<u32, PeptideEligibilitySummary> = FnvHashMap::default();

    for (pep_id, obs_set) in observed_runs {
        let n_runs_observed = obs_set.len();
        let n_runs_strong_prior = strong_runs.get(&pep_id).map(|s| s.len()).unwrap_or(0);
        let protein_rescue_eligible = protein_ok_by_peptide.get(&pep_id).copied().unwrap_or(false);

        let observed_run_fraction = (n_runs_observed as f64) / (total_runs as f64);
        let strong_run_fraction = (n_runs_strong_prior as f64) / (total_runs as f64);

        let observed_frac_ok = observed_run_fraction >= cfg.min_run_fraction;
        let observed_count_ok = n_runs_observed >= cfg.min_run_count;

        let strong_frac_ok = strong_run_fraction >= cfg.min_strong_run_fraction;
        let strong_count_ok = n_runs_strong_prior >= cfg.min_strong_run_count;

        let is_rescue_eligible = protein_rescue_eligible
            && observed_frac_ok
            && observed_count_ok
            && strong_frac_ok
            && strong_count_ok;

        out.insert(
            pep_id,
            PeptideEligibilitySummary {
                n_runs_observed,
                n_runs_strong_prior,
                observed_run_fraction,
                strong_run_fraction,
                is_rescue_eligible,
                protein_rescue_eligible,
            },
        );
    }

    out
}

fn build_reproducibility_anchor_map(
    features: &[DfFeature],
    settings: &FdrSettings,
    peptide_eligibility_map: &FnvHashMap<u32, PeptideEligibilitySummary>,
) -> FnvHashMap<u32, ReproducibilityAnchorSummary> {
    use crate::input::ReproducibilityAnchorMode;

    let pep_cfg = &settings.reproducibility.peptide_eligibility;
    let anchor_cfg = &settings.reproducibility.anchor;

    let mut strong_peps: FnvHashMap<u32, Vec<f64>> = FnvHashMap::default();

    for f in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let pep_id = f.core.peptide_idx.0;

        match peptide_eligibility_map.get(&pep_id) {
            Some(x) if x.is_rescue_eligible => {}
            _ => continue,
        }

        let q_ok = f
            .decoy_free_q_value
            .map(|q| q <= pep_cfg.strong_reference_q_threshold_physical as f64)
            .unwrap_or(false);

        let pep_ok = match pep_cfg.strong_reference_pep_threshold_physical {
            Some(thr) => f.decoy_free_pep.map(|p| p <= thr).unwrap_or(false),
            None => true,
        };

        if !(q_ok && pep_ok) {
            continue;
        }

        if let Some(pep) = f.decoy_free_pep {
            let pep = finite_df_probability_for_logit(pep);
            strong_peps.entry(pep_id).or_default().push(pep);
        }
    }

    let mut out: FnvHashMap<u32, ReproducibilityAnchorSummary> = FnvHashMap::default();

    for (pep_id, vals) in strong_peps {
        if vals.is_empty() {
            continue;
        }

        let mut tmp = vals.clone();

        let anchor_value = match anchor_cfg.mode {
            ReproducibilityAnchorMode::Best => tmp.iter().copied().min_by(|a, b| a.total_cmp(b)),
            ReproducibilityAnchorMode::SecondBest => second_best_f64(&mut tmp),
            ReproducibilityAnchorMode::Mean => Some(tmp.iter().sum::<f64>() / (tmp.len() as f64)),
            ReproducibilityAnchorMode::Median => median_f64(tmp),
            ReproducibilityAnchorMode::TrimmedMean => {
                let trim = anchor_cfg.trim_fraction.unwrap_or(0.1);
                trimmed_mean(&mut tmp, trim)
            }
        };

        if let Some(anchor_value) = anchor_value {
            out.insert(
                pep_id,
                ReproducibilityAnchorSummary {
                    anchor_value: finite_df_probability_for_logit(anchor_value),
                    n_anchor_observations: vals.len(),
                },
            );
        }
    }

    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RescueBand {
    Strong,
    RescueEligible,
    TooWeak,
}

fn classify_rescue_band(prior_pep: f64, settings: &FdrSettings) -> RescueBand {
    let band = &settings.reproducibility.rescue_band;

    let strong_cutoff = band.strong_cutoff_pep;
    let weak_cutoff = band.weak_cutoff_pep;

    if prior_pep <= strong_cutoff {
        RescueBand::Strong
    } else if prior_pep < weak_cutoff {
        RescueBand::RescueEligible
    } else {
        RescueBand::TooWeak
    }
}

fn apply_reproducibility_anchor_rescue(
    prior_pep: f64,
    anchor_pep: f64,
    expert_support_shift: f64,
    recurrence_support_shift: f64,
    settings: &FdrSettings,
) -> (f64, f64) {
    use crate::input::RescueMode;

    let band = &settings.reproducibility.rescue_band;
    let max_frac = band.max_rescue_fraction.clamp(0.0, 1.0);

    let prior = finite_df_probability_for_logit(prior_pep);
    let anchor = finite_df_probability_for_logit(anchor_pep);

    let improved_target = anchor.min(prior);

    let rescued_pep = finite_df_probability_for_logit(match band.rescue_mode {
        RescueMode::Replace => improved_target,
        RescueMode::BoundedShrinkage => prior + max_frac * (improved_target - prior),
    });

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

    let post_pep = finite_df_probability_for_logit(crate::ml::stats::safe_inv_logit_confidence(
        prior_logit + bounded_shift,
    ));

    (post_pep, bounded_shift.abs())
}

#[inline]
fn peptide_key_for_feature(f: &DfFeature) -> String {
    f.core.peptide_idx.0.to_string()
}

struct RecurrenceNullModel {
    null_support: Vec<f64>,
}

fn build_recurrence_null_model(
    features: &[DfFeature],
    settings: &FdrSettings,
    _db: &IndexedDatabase,
) -> Option<RecurrenceNullModel> {
    let mut peptide_to_support: FnvHashMap<String, f64> = FnvHashMap::default();

    for f in features.iter() {
        if f.core.rank < settings.min_null_rank || f.core.rank > settings.max_null_rank {
            continue;
        }

        let peptide = peptide_key_for_feature(f);
        *peptide_to_support.entry(peptide).or_insert(0.0) += 1.0;
    }

    let mut vals: Vec<f64> = peptide_to_support.into_values().collect();

    if vals.len() < settings.min_null_size {
        return None;
    }

    vals.sort_by(|a, b| a.total_cmp(b));

    Some(RecurrenceNullModel { null_support: vals })
}

fn recurrence_aux_null_p_value(model: &RecurrenceNullModel, support: f64) -> f64 {
    let n = model.null_support.len();
    if n == 0 {
        return 1.0;
    }

    let idx = model.null_support.partition_point(|&x| x < support);
    let count_ge = n - idx;

    ((count_ge as f64 + 1.0) / (n as f64 + 1.0)).clamp(1e-300, 1.0)
}

fn apply_recurrence_null_pvalue_update_to_active_stream(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> ReproducibilityResult {
    let Some(model) = build_recurrence_null_model(features, settings, db) else {
        return ReproducibilityResult {
            enabled: true,
            fail_closed: true,
            ..Default::default()
        };
    };

    let mut peptide_support: FnvHashMap<String, f64> = FnvHashMap::default();

    for f in features.iter().filter(|f| f.core.rank == 1) {
        let peptide = peptide_key_for_feature(f);
        *peptide_support.entry(peptide).or_insert(0.0) += 1.0;
    }

    let n_rescue_eligible_peptides = peptide_support.len();
    let mut n_rescued_psms = 0usize;
    let mut max_shift_applied = 0.0f64;

    for f in features.iter_mut().filter(|f| f.core.rank == 1) {
        let peptide = peptide_key_for_feature(f);
        let support = peptide_support.get(&peptide).copied().unwrap_or(1.0);

        let p_recur = finite_df_p_value(recurrence_aux_null_p_value(&model, support));
        let base_p = finite_df_p_value(f.decoy_free_p_value.unwrap_or(1.0));

        let combined_p = finite_df_p_value(combine_cauchy(&[base_p, p_recur]));

        // Bound p-value-active recurrence rescue on the -log10(p) scale.
        // This preserves MSFDR dynamic range but prevents unbounded auxiliary
        // evidence stacking.
        let base_score = -base_p.log10();
        let combined_score = -combined_p.log10();

        let max_total_shift = settings.reproducibility.max_total_shift.max(0.0);
        let max_recurrence_shift = settings.reproducibility.max_recurrence_shift.max(0.0);
        let max_shift = max_total_shift.min(max_recurrence_shift);

        let raw_delta = combined_score - base_score;
        let bounded_delta = raw_delta.clamp(-max_shift, max_shift);
        let bounded_score = (base_score + bounded_delta).max(0.0);

        let p_new = finite_df_p_value(10.0_f64.powf(-bounded_score));

        let rescued = p_new < base_p;
        f.rescued_by_recurrence = Some(rescued);
        if rescued {
            n_rescued_psms += 1;
            f.physical_rescue_source = Some("recurrence".to_string());
        }

        let shift_abs = (bounded_score - base_score).abs();
        if shift_abs.is_finite() {
            max_shift_applied = max_shift_applied.max(shift_abs);
        }

        let placeholder_pep = finite_df_probability_for_logit(f.decoy_free_pep.unwrap_or(1.0));

        set_df_evidence_pair(f, ActiveEvidenceSpace::PValue, p_new, placeholder_pep);
    }

    recalibrate_companion_pep_from_active_p_values(features, settings, "recurrence p-value rescue");
    recalculate_active_q_values(features, settings);

    let configured_shift_cap = settings
        .reproducibility
        .max_total_shift
        .max(0.0)
        .min(settings.reproducibility.max_recurrence_shift.max(0.0));

    if max_shift_applied > configured_shift_cap + 1e-9 {
        log::warn!(
            "DF recurrence p-value rescue exceeded configured cap: observed={:.6e} cap={:.6e}",
            max_shift_applied,
            configured_shift_cap
        );
    }

    ReproducibilityResult {
        enabled: true,
        fail_closed: false,
        n_rescue_eligible_proteins: 0,
        n_rescue_eligible_peptides,
        n_anchor_peptides: model.null_support.len(),
        n_rescued_psms,
        n_strong_unchanged_psms: 0,
        n_too_weak_unrescued_psms: 0,
        agreement_support_mean: if n_rescue_eligible_peptides > 0 {
            peptide_support.values().sum::<f64>() / n_rescue_eligible_peptides as f64
        } else {
            0.0
        },
        max_shift_applied,
    }
}

fn apply_bounded_repro_shift(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> ReproducibilityResult {
    if !settings.reproducibility.enabled {
        return ReproducibilityResult {
            enabled: false,
            ..Default::default()
        };
    }

    let protein_support_map = build_protein_support_map(features, db, settings);

    let total_unique_observed: usize = protein_support_map
        .values()
        .map(|s| s.n_unique_observed)
        .sum();

    let total_unique_passing_prior: usize = protein_support_map
        .values()
        .map(|s| s.n_unique_passing_prior)
        .sum();

    log::debug!(
        "DF reproducibility protein support: proteins={} eligible={} total_unique_observed={} total_unique_passing_prior={}",
        protein_support_map.len(),
        protein_support_map.values().filter(|s| s.is_rescue_eligible).count(),
        total_unique_observed,
        total_unique_passing_prior
    );

    let peptide_eligibility_map =
        build_peptide_eligibility_map(features, db, settings, &protein_support_map);

    let total_runs_observed: usize = peptide_eligibility_map
        .values()
        .map(|s| s.n_runs_observed)
        .sum();

    let total_runs_strong_prior: usize = peptide_eligibility_map
        .values()
        .map(|s| s.n_runs_strong_prior)
        .sum();

    let protein_backed_peptides = peptide_eligibility_map
        .values()
        .filter(|s| s.protein_rescue_eligible)
        .count();

    log::debug!(
        "DF reproducibility peptide eligibility: peptides={} eligible={} protein_backed={} total_runs_observed={} total_runs_strong_prior={}",
        peptide_eligibility_map.len(),
        peptide_eligibility_map.values().filter(|s| s.is_rescue_eligible).count(),
        protein_backed_peptides,
        total_runs_observed,
        total_runs_strong_prior
    );

    let anchor_map = build_reproducibility_anchor_map(features, settings, &peptide_eligibility_map);

    let total_anchor_observations: usize =
        anchor_map.values().map(|s| s.n_anchor_observations).sum();

    log::debug!(
        "DF reproducibility anchors: peptides_with_anchor={} total_anchor_observations={}",
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

        let prior_pep = f.decoy_free_pep.unwrap_or(1.0);
        let pep_id = f.core.peptide_idx.0;

        let expert_s = compute_expert_agreement_support(f, settings)
            * compute_redundancy_discount(f, settings);
        agreement_sum += expert_s;

        let eligibility = peptide_eligibility_map.get(&pep_id);
        let recurrence_s = eligibility
            .map(|elig| compute_cross_run_recurrence_support(elig, settings))
            .unwrap_or(0.0);

        let band = classify_rescue_band(prior_pep, settings);

        let (post_pep, shift_abs) = match (eligibility, band) {
            (Some(elig), RescueBand::RescueEligible) if elig.is_rescue_eligible => {
                if let Some(anchor) = anchor_map.get(&pep_id) {
                    let (post, shift_abs) = apply_reproducibility_anchor_rescue(
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
            (_, RescueBand::Strong) => {
                n_strong_unchanged_psms += 1;
                (prior_pep, 0.0)
            }
            (_, RescueBand::TooWeak) => {
                n_too_weak_unrescued_psms += 1;
                (prior_pep, 0.0)
            }
            _ => (prior_pep, 0.0),
        };

        max_shift_applied = max_shift_applied.max(shift_abs);

        let rescued = post_pep + 1e-12 < prior_pep;
        f.rescued_by_recurrence = Some(rescued);
        if rescued {
            f.physical_rescue_source = Some("recurrence".to_string());
        }

        f.decoy_free_pep = Some(post_pep);
        f.decoy_free_score = Some(df_score_from_pep(post_pep));
    }

    recalculate_active_q_values(features, settings);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfStageOutcome {
    Applied,
    Skipped,
    FailedClosed,
}

impl DfStageOutcome {
    #[inline]
    pub fn applied(self) -> bool {
        matches!(self, DfStageOutcome::Applied)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveDfStream {
    Base,
    Rt,
    Ims,
    PeptideRescue,
    ProteinRescue,
}

// Validate that the active stream is complete before peptide/protein inference.
// This function checks the mutable `decoy_free_*` stream, not stage snapshots.
// Optional-stage failures should already have preserved the last-good active
// stream before this point.
fn validate_final_df_stream_contract(
    features: &[DfFeature],
    final_stream: ActiveDfStream,
    settings: &FdrSettings,
) {
    let active = active_evidence_space(settings);

    let mut n_rank1 = 0usize;
    let mut n_invalid = 0usize;

    for f in features.iter() {
        if f.core.rank == 1 {
            n_rank1 += 1;

            let p = f.decoy_free_p_value;
            let pep = f.decoy_free_pep;
            let score = f.decoy_free_score;
            let q = f.decoy_free_q_value;

            let valid_complete = p.is_some() && pep.is_some() && score.is_some() && q.is_some();

            let valid_score = match (active, p, pep, score) {
                (ActiveEvidenceSpace::PValue, Some(p), _, Some(s)) => {
                    let expected = df_score_from_p_value(p as f64);
                    ((s - expected).abs() as f64) < 1e-3
                }
                (ActiveEvidenceSpace::Pep, _, Some(pep), Some(s)) => {
                    let expected = df_score_from_pep(pep as f64);
                    ((s - expected).abs() as f64) < 1e-3
                }
                _ => false,
            };

            if !valid_complete || !valid_score {
                n_invalid += 1;
            }
        } else {
            let leaked = f.decoy_free_p_value.is_some()
                || f.decoy_free_pep.is_some()
                || f.decoy_free_score.is_some()
                || f.decoy_free_q_value.is_some()
                || f.decoy_free_peptide_q.is_some()
                || f.decoy_free_protein_q.is_some()
                || f.decoy_free_protein_supported_peptide.is_some()
                || f.decoy_free_peptide_supported_psm.is_some();

            if leaked {
                n_invalid += 1;
            }
        }
    }

    if n_invalid > 0 {
        let msg = format!(
            "DF final stream contract violated: stream={:?} active={:?} rank1={} invalid_rows={}",
            final_stream, active, n_rank1, n_invalid
        );
        log::error!("{}", msg);
        panic!("{}", msg);
    }

    log::debug!(
        "DF final stream contract OK: stream={:?} active={:?} rank1={}",
        final_stream,
        active,
        n_rank1
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DfAdjustmentStage {
    Rt,
    Ims,
    PeptideRescue,
    ProteinRescue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PhysicalEvidenceStage {
    RtOnly,
    ImsOnly,
}

#[inline]
fn df_score_from_pep(pep: f64) -> f64 {
    -10.0 * finite_df_probability_for_logit(pep).log10()
}

// Copy the current active stream into the audit columns for a successfully
// applied optional stage. This function must not be used for disabled,
// failed-closed, or no-op stages.
#[inline]
fn snapshot_current_stream_to_stage(f: &mut DfFeature, stage: DfAdjustmentStage) {
    let p = f.decoy_free_p_value;
    let pep = f.decoy_free_pep;
    let score = f.decoy_free_score;
    let q = f.decoy_free_q_value;

    match stage {
        DfAdjustmentStage::Rt => {
            f.decoy_free_p_value_rt = p;
            f.decoy_free_pep_rt = pep;
            f.decoy_free_score_rt = score;
            f.decoy_free_q_rt = q;
        }
        DfAdjustmentStage::Ims => {
            f.decoy_free_p_value_ims = p;
            f.decoy_free_pep_ims = pep;
            f.decoy_free_score_ims = score;
            f.decoy_free_q_ims = q;
        }
        DfAdjustmentStage::PeptideRescue => {
            f.decoy_free_p_value_peptide_rescue = p;
            f.decoy_free_pep_peptide_rescue = pep;
            f.decoy_free_score_peptide_rescue = score;
            f.decoy_free_q_peptide_rescue = q;
        }
        DfAdjustmentStage::ProteinRescue => {
            f.decoy_free_p_value_protein_rescue = p;
            f.decoy_free_pep_protein_rescue = pep;
            f.decoy_free_score_protein_rescue = score;
            f.decoy_free_q_protein_rescue = q;
        }
    }
}

// Recompute q-values for a PEP-native active stream using cumulative mean PEP
// after best-first sorting. `decoy_free_score` is larger-is-better; do not
// negate it before calling q_from_pep_cummean.
fn recalculate_active_q_values(features: &mut [DfFeature], settings: &FdrSettings) {
    match active_evidence_space(settings) {
        ActiveEvidenceSpace::Pep => {
            let mut rows: Vec<(f64, usize, f64)> = Vec::new();

            for (i, f) in features.iter().enumerate() {
                if f.core.rank != 1 {
                    continue;
                }

                let pep = match f.decoy_free_pep {
                    Some(x) if x.is_finite() => (x as f64).clamp(0.0, 1.0).max(1e-300),
                    _ => continue,
                };

                let score_key = f.decoy_free_score.unwrap_or_else(|| df_score_from_pep(pep)) as f64;
                rows.push((score_key, i, pep));
            }

            for (feat_idx, q) in q_from_pep_cummean(rows) {
                features[feat_idx].decoy_free_q_value = Some(finite_df_p_value(q));
            }
        }

        ActiveEvidenceSpace::PValue => {
            let mut rows: Vec<(f64, usize, f64)> = Vec::new();
            let mut p_values: Vec<f64> = Vec::new();

            for (i, f) in features.iter().enumerate() {
                if f.core.rank != 1 {
                    continue;
                }

                let p = match f.decoy_free_p_value {
                    Some(x) if x.is_finite() => (x as f64).clamp(0.0, 1.0).max(1e-300),
                    _ => continue,
                };

                rows.push((p, i, p));
                p_values.push(p);
            }

            let feature_indices: Vec<usize> = rows.iter().map(|(_, idx, _)| *idx).collect();

            let psm_cov_values: Vec<Option<f64>> = feature_indices
                .iter()
                .map(|&idx| psm_covariate_value(&features[idx], settings.psm_q_covariate))
                .collect();

            let q_report = q_values_from_level_covariates(
                &p_values,
                &p_values,
                &psm_cov_values,
                settings,
                effective_psm_q_method(settings),
                QLevel::Psm,
                "PSM active p-value",
            );

            let q_values = q_report.q_values;

            for ((_, feat_idx, _), q) in rows.into_iter().zip(q_values.into_iter()) {
                features[feat_idx].decoy_free_q_value = Some(finite_df_p_value(q));
            }
        }
    }
}

// Freeze the mandatory base stream exactly once. RT/IMS anchor selection uses
// this base snapshot so that physical reliability is estimated from unmodified
// base evidence. This function is idempotent and must not overwrite an existing
// base snapshot after optional stages have run.
fn snapshot_base_stream_once(features: &mut [DfFeature]) {
    for f in features.iter_mut().filter(|f| f.core.rank == 1) {
        if f.decoy_free_pep_base.is_none()
            && f.decoy_free_q_base.is_none()
            && f.decoy_free_score_base.is_none()
            && f.decoy_free_p_value_base.is_none()
        {
            f.decoy_free_p_value_base = f.decoy_free_p_value;
            f.decoy_free_pep_base = f.decoy_free_pep;
            f.decoy_free_score_base = f.decoy_free_score;
            f.decoy_free_q_base = f.decoy_free_q_value;
        }
    }
}

fn write_model_stage_snapshot(f: &mut DfFeature, settings: &FdrSettings, stage: DfAdjustmentStage) {
    let p = f.decoy_free_p_value;
    let q = f.decoy_free_q_value;
    let pep = f.decoy_free_pep;

    match settings.model_fit {
        ModelFit::Moments => match stage {
            DfAdjustmentStage::Rt => {
                f.rt_adjust_p_mom = p;
                f.rt_adjust_q_mom = q;
                f.rt_adjust_pep_mom = pep;
            }
            DfAdjustmentStage::Ims => {
                f.ims_adjust_p_mom = p;
                f.ims_adjust_q_mom = q;
                f.ims_adjust_pep_mom = pep;
            }
            DfAdjustmentStage::PeptideRescue => {
                f.peptide_rescue_p_mom = p;
                f.peptide_rescue_q_mom = q;
                f.peptide_rescue_pep_mom = pep;
            }
            DfAdjustmentStage::ProteinRescue => {
                f.protein_rescue_p_mom = p;
                f.protein_rescue_q_mom = q;
                f.protein_rescue_pep_mom = pep;
            }
        },

        ModelFit::Mle => match stage {
            DfAdjustmentStage::Rt => {
                f.rt_adjust_p_mle = p;
                f.rt_adjust_q_mle = q;
                f.rt_adjust_pep_mle = pep;
            }
            DfAdjustmentStage::Ims => {
                f.ims_adjust_p_mle = p;
                f.ims_adjust_q_mle = q;
                f.ims_adjust_pep_mle = pep;
            }
            DfAdjustmentStage::PeptideRescue => {
                f.peptide_rescue_p_mle = p;
                f.peptide_rescue_q_mle = q;
                f.peptide_rescue_pep_mle = pep;
            }
            DfAdjustmentStage::ProteinRescue => {
                f.protein_rescue_p_mle = p;
                f.protein_rescue_q_mle = q;
                f.protein_rescue_pep_mle = pep;
            }
        },

        ModelFit::LowerOrder => match stage {
            DfAdjustmentStage::Rt => {
                f.rt_adjust_p_lo = p;
                f.rt_adjust_q_lo = q;
                f.rt_adjust_pep_lo = pep;
            }
            DfAdjustmentStage::Ims => {
                f.ims_adjust_p_lo = p;
                f.ims_adjust_q_lo = q;
                f.ims_adjust_pep_lo = pep;
            }
            DfAdjustmentStage::PeptideRescue => {
                f.peptide_rescue_p_lo = p;
                f.peptide_rescue_q_lo = q;
                f.peptide_rescue_pep_lo = pep;
            }
            DfAdjustmentStage::ProteinRescue => {
                f.protein_rescue_p_lo = p;
                f.protein_rescue_q_lo = q;
                f.protein_rescue_pep_lo = pep;
            }
        },

        ModelFit::Msfdr => match stage {
            DfAdjustmentStage::Rt => {
                f.rt_adjust_p_msfdr = p;
                f.rt_adjust_q_msfdr = q;
                f.rt_adjust_pep_msfdr = pep;
            }
            DfAdjustmentStage::Ims => {
                f.ims_adjust_p_msfdr = p;
                f.ims_adjust_q_msfdr = q;
                f.ims_adjust_pep_msfdr = pep;
            }
            DfAdjustmentStage::PeptideRescue => {
                f.peptide_rescue_p_msfdr = p;
                f.peptide_rescue_q_msfdr = q;
                f.peptide_rescue_pep_msfdr = pep;
            }
            DfAdjustmentStage::ProteinRescue => {
                f.protein_rescue_p_msfdr = p;
                f.protein_rescue_q_msfdr = q;
                f.protein_rescue_pep_msfdr = pep;
            }
        },

        ModelFit::Msfdr1Smix => match stage {
            DfAdjustmentStage::Rt => {
                f.rt_adjust_p_1smix = p;
                f.rt_adjust_q_1smix = q;
                f.rt_adjust_pep_1smix = pep;
            }
            DfAdjustmentStage::Ims => {
                f.ims_adjust_p_1smix = p;
                f.ims_adjust_q_1smix = q;
                f.ims_adjust_pep_1smix = pep;
            }
            DfAdjustmentStage::PeptideRescue => {
                f.peptide_rescue_p_1smix = p;
                f.peptide_rescue_q_1smix = q;
                f.peptide_rescue_pep_1smix = pep;
            }
            DfAdjustmentStage::ProteinRescue => {
                f.protein_rescue_p_1smix = p;
                f.protein_rescue_q_1smix = q;
                f.protein_rescue_pep_1smix = pep;
            }
        },

        ModelFit::Msfdr2Smix => match stage {
            DfAdjustmentStage::Rt => {
                f.rt_adjust_p_2smix = p;
                f.rt_adjust_q_2smix = q;
                f.rt_adjust_pep_2smix = pep;
            }
            DfAdjustmentStage::Ims => {
                f.ims_adjust_p_2smix = p;
                f.ims_adjust_q_2smix = q;
                f.ims_adjust_pep_2smix = pep;
            }
            DfAdjustmentStage::PeptideRescue => {
                f.peptide_rescue_p_2smix = p;
                f.peptide_rescue_q_2smix = q;
                f.peptide_rescue_pep_2smix = pep;
            }
            DfAdjustmentStage::ProteinRescue => {
                f.protein_rescue_p_2smix = p;
                f.protein_rescue_q_2smix = q;
                f.protein_rescue_pep_2smix = pep;
            }
        },

        ModelFit::Nokoi => match stage {
            DfAdjustmentStage::Rt => {
                f.rt_adjust_p_nokoi = p;
                f.rt_adjust_q_nokoi = q;
                f.rt_adjust_pep_nokoi = pep;
            }
            DfAdjustmentStage::Ims => {
                f.ims_adjust_p_nokoi = p;
                f.ims_adjust_q_nokoi = q;
                f.ims_adjust_pep_nokoi = pep;
            }
            DfAdjustmentStage::PeptideRescue => {
                f.peptide_rescue_p_nokoi = p;
                f.peptide_rescue_q_nokoi = q;
                f.peptide_rescue_pep_nokoi = pep;
            }
            DfAdjustmentStage::ProteinRescue => {
                f.protein_rescue_p_nokoi = p;
                f.protein_rescue_q_nokoi = q;
                f.protein_rescue_pep_nokoi = pep;
            }
        },

        ModelFit::Ensemble => match stage {
            DfAdjustmentStage::Rt => {
                f.rt_adjust_p_ensemble = p;
                f.rt_adjust_q_ensemble = q;
                f.rt_adjust_pep_ensemble = pep;
            }
            DfAdjustmentStage::Ims => {
                f.ims_adjust_p_ensemble = p;
                f.ims_adjust_q_ensemble = q;
                f.ims_adjust_pep_ensemble = pep;
            }
            DfAdjustmentStage::PeptideRescue => {
                f.peptide_rescue_p_ensemble = p;
                f.peptide_rescue_q_ensemble = q;
                f.peptide_rescue_pep_ensemble = pep;
            }
            DfAdjustmentStage::ProteinRescue => {
                f.protein_rescue_p_ensemble = p;
                f.protein_rescue_q_ensemble = q;
                f.protein_rescue_pep_ensemble = pep;
            }
        },
    }
}

// Finalize a successfully applied PEP-native optional stage by recalculating
// active q-values and writing both generic stage snapshots and model-specific
// audit fields. Call this only after the stage has already updated the active
// `decoy_free_*` stream.
fn finalize_stage_snapshot(
    features: &mut [DfFeature],
    settings: &FdrSettings,
    stage: DfAdjustmentStage,
) {
    recalculate_active_q_values(features, settings);

    for f in features.iter_mut().filter(|f| f.core.rank == 1) {
        snapshot_current_stream_to_stage(f, stage);
        write_model_stage_snapshot(f, settings, stage);
    }
}

fn log_external_feature_diagnostics(features: &[DfFeature]) {
    let joined = features
        .iter()
        .filter(|f| f.core.external_features.ms2rescore_feature_joined)
        .count();

    if joined == 0 {
        return;
    }

    log::info!(
        "external TIMS2/MS2Rescore features present on {}/{} candidate PSMs",
        joined,
        features.len()
    );

    log_external_feature_one(features, "ms2rescore_ms2pip_pcc", |f| {
        f.core.external_features.ms2rescore_ms2pip_pcc as f64
    });

    log_external_feature_one(features, "ms2rescore_spectral_angle", |f| {
        f.core.external_features.ms2rescore_spectral_angle as f64
    });

    log_external_feature_one(features, "ms2rescore_deeplc_abs_rt_error", |f| {
        f.core.external_features.ms2rescore_deeplc_abs_rt_error as f64
    });

    log_external_feature_one(features, "tims2rescore_abs_ccs_error", |f| {
        f.core.external_features.tims2rescore_abs_ccs_error as f64
    });

    log_external_feature_one(features, "tims2rescore_pct_ccs_error", |f| {
        f.core.external_features.tims2rescore_pct_ccs_error as f64
    });
}

fn log_external_feature_one<F>(features: &[DfFeature], name: &str, getter: F)
where
    F: Fn(&DfFeature) -> f64,
{
    let rank1: Vec<f64> = features
        .iter()
        .filter(|f| f.core.rank == 1)
        .map(&getter)
        .filter(|x| x.is_finite())
        .collect();

    let rank_null: Vec<f64> = features
        .iter()
        .filter(|f| f.core.rank > 1)
        .map(&getter)
        .filter(|x| x.is_finite())
        .collect();

    if rank1.len() < 10 || rank_null.len() < 10 {
        log::info!(
            "external feature diagnostic {name}: insufficient finite values rank1={} rank_null={}",
            rank1.len(),
            rank_null.len()
        );
        return;
    }

    let rank1_med = external_feature_median(rank1);
    let null_med = external_feature_median(rank_null);

    let hyp_corr =
        external_feature_pearson_pairwise(features, |f| getter(f), |f| f.core.hyperscore as f64);

    let p_corr = external_feature_pearson_pairwise(
        features,
        |f| getter(f),
        |f| f.decoy_free_p_value.unwrap_or(f64::NAN),
    );

    log::info!(
        "external feature diagnostic {name}: rank1_median={:.6} rank_null_median={:.6} corr_hyperscore={:.4} corr_df_p={:.4}",
        rank1_med,
        null_med,
        hyp_corr,
        p_corr
    );
}

fn external_feature_median(mut xs: Vec<f64>) -> f64 {
    xs.retain(|x| x.is_finite());
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn external_feature_pearson_pairwise<F, G>(features: &[DfFeature], x_getter: F, y_getter: G) -> f64
where
    F: Fn(&DfFeature) -> f64,
    G: Fn(&DfFeature) -> f64,
{
    let pairs: Vec<(f64, f64)> = features
        .iter()
        .map(|f| (x_getter(f), y_getter(f)))
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();

    if pairs.len() < 10 {
        return f64::NAN;
    }

    let n = pairs.len() as f64;
    let mean_x = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|p| p.1).sum::<f64>() / n;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    if den_x <= 0.0 || den_y <= 0.0 {
        f64::NAN
    } else {
        num / (den_x.sqrt() * den_y.sqrt())
    }
}

pub fn run_df_layers(
    psms: &[DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> Vec<DfFeature> {
    let mut new_features = psms.to_vec();

    log_external_feature_diagnostics(&new_features);

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
    let run_nokoi = if use_ensemble {
        settings.enable_nokoi
    } else {
        matches!(settings.model_fit, ModelFit::Nokoi)
    };

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

    // 1B. Experts
    //
    // Each expert now builds its own method-specific rank-null pool using its own
    // purification factor. There is intentionally no shared/global null pool here.
    let engines = match fit_base_experts(&new_features, &work, settings, gates) {
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

    // 1H. Freeze the finalized base DF stream before any RT/IMS/rescue stage.
    // RT/IMS anchor selection reads decoy_free_*_base, so these fields must be
    // populated immediately after base q-values are computed.
    snapshot_base_stream_once(&mut new_features);

    // 1I. Physical residual diagnostics and conservative local RT guardrail.
    //
    // This does not delete any PSMs. It only:
    //   - writes RT/IMS residual diagnostics to the TSV,
    //   - marks extreme local RT outliers,
    //   - prevents those outliers from positive RT rescue / RT physical training.
    //
    // The sigma values used here are intentionally derived from the current DF
    // physical context, not from final peptide/protein inference.
    let rt_diag_anchors = build_physical_anchor_set(&new_features, settings, db);
    let (rt_diag_safe_anchors, _) =
        exclude_non_rescue_safe_anchors(&new_features, rt_diag_anchors, settings, db);
    let rt_diag_rel = compute_rt_reliability(&new_features, &rt_diag_safe_anchors, settings);
    let ims_diag_rel = compute_ims_reliability(&new_features, &rt_diag_safe_anchors, settings);

    annotate_physical_residual_columns(
        &mut new_features,
        settings,
        rt_diag_rel.rt_sigma_global,
        ims_diag_rel.ims_sigma_global,
    );

    let _rt_local_excluded = annotate_local_rt_training_guardrail(&mut new_features, settings);

    let base_snapshot_missing = new_features
        .iter()
        .filter(|f| f.core.rank == 1)
        .filter(|f| {
            f.decoy_free_pep_base.is_none()
                || f.decoy_free_q_base.is_none()
                || f.decoy_free_score_base.is_none()
        })
        .count();

    if base_snapshot_missing > 0 {
        panic!(
        "DF base snapshot failed: {} rank-1 rows are missing base pep/q/score after finalize_base_q_values",
        base_snapshot_missing
    );
    }

    // Track the last-good active stream explicitly. Do not infer the final stream
    // from populated audit/scratch fields.
    let mut active_stream = ActiveDfStream::Base;

    // 2A. Optional RT confidence adjustment.
    if settings.enable_rt_confidence_adjustment {
        let (rt_res, outcome) = apply_rt_confidence_adjustment(&mut new_features, settings, db);
        if outcome.applied() {
            active_stream = ActiveDfStream::Rt;
        }
        log::info!(
            "DF RT adjustment: enabled={} anchors={}/{} rt_reliability={:.4} joint_reliability={:.4} rt_sigma={:?} ims_sigma={:?} dropped_runs={} dropped_charge_bins={} fail_closed={}",
            rt_res.enabled, rt_res.anchor_count_after_filters, rt_res.anchor_count_total,
            rt_res.rt_reliability, rt_res.joint_reliability, rt_res.rt_sigma_global,
            rt_res.ims_sigma_global, rt_res.dropped_runs.len(), rt_res.dropped_charge_bins.len(),
            rt_res.fail_closed
        );
    } else {
        log::info!("DF RT adjustment: enabled=false");
    }

    // 2B. Optional IMS confidence adjustment.
    if settings.enable_ims_confidence_adjustment {
        let (ims_res, outcome) = apply_ims_confidence_adjustment(&mut new_features, settings, db);
        if outcome.applied() {
            active_stream = ActiveDfStream::Ims;
        }
        log::info!(
            "DF IMS adjustment: enabled={} anchors={}/{} ims_reliability={:.4} joint_reliability={:.4} rt_sigma={:?} ims_sigma={:?} dropped_runs={} dropped_charge_bins={} fail_closed={}",
            ims_res.enabled, ims_res.anchor_count_after_filters, ims_res.anchor_count_total,
            ims_res.ims_reliability, ims_res.joint_reliability, ims_res.rt_sigma_global,
            ims_res.ims_sigma_global, ims_res.dropped_runs.len(), ims_res.dropped_charge_bins.len(),
            ims_res.fail_closed
        );
    } else {
        log::info!("DF IMS adjustment: enabled=false");
    }

    // 3A. Optional peptide reproducibility rescue.
    if settings.enable_peptide_reproducibility_rescue {
        let (pep_repro_res, outcome) =
            apply_peptide_reproducibility_rescue(&mut new_features, settings, db);
        if outcome.applied() {
            active_stream = ActiveDfStream::PeptideRescue;
        }
        log::info!(
            "DF peptide reproducibility rescue: enabled={} eligible_peptides={} anchor_peptides={} rescued_psms={} strong_unchanged={} too_weak_unrescued={} agree_mean={:.4} max_shift={:.4} fail_closed={}",
            pep_repro_res.enabled, pep_repro_res.n_rescue_eligible_peptides, pep_repro_res.n_anchor_peptides,
            pep_repro_res.n_rescued_psms, pep_repro_res.n_strong_unchanged_psms, pep_repro_res.n_too_weak_unrescued_psms,
            pep_repro_res.agreement_support_mean, pep_repro_res.max_shift_applied, pep_repro_res.fail_closed
        );
    } else {
        log::info!("DF peptide reproducibility rescue: enabled=false");
    }

    // 3B. Optional protein reproducibility rescue.
    if settings.enable_protein_reproducibility_rescue {
        let (prot_repro_res, outcome) =
            apply_protein_reproducibility_rescue(&mut new_features, settings, db);
        if outcome.applied() {
            active_stream = ActiveDfStream::ProteinRescue;
        }
        log::info!(
            "DF protein reproducibility rescue: enabled={} eligible_proteins={} rescued_psms={} max_shift={:.4} fail_closed={}",
            prot_repro_res.enabled, prot_repro_res.n_rescue_eligible_proteins, prot_repro_res.n_rescued_psms,
            prot_repro_res.max_shift_applied, prot_repro_res.fail_closed
        );
    } else {
        log::info!("DF protein reproducibility rescue: enabled=false");
    }

    // 4. Validate the active PSM stream from the completed DF layers.
    validate_final_df_stream_contract(&new_features, active_stream, settings);

    let stream_kind = match active_stream {
        ActiveDfStream::Base => "base",
        ActiveDfStream::Rt => "rt_adjusted",
        ActiveDfStream::Ims => "ims_adjusted",
        ActiveDfStream::PeptideRescue => "peptide_reproducibility_rescue",
        ActiveDfStream::ProteinRescue => "protein_reproducibility_rescue",
    };
    log::info!("DF final active stream: {}", stream_kind);

    log_physical_sigma_diagnostics(&new_features, db);
    log_physical_rescue_enrichment_diagnostics(&new_features, db);

    log_df_pvalue_bin_diagnostics(&new_features, db);
    log_spectral_feature_separation_diagnostics(&new_features, db);
    log_delta_next_focused_diagnostics(&new_features, db);
    log_peptide_recurrence_separation_diagnostics(&new_features, db);
    log_mass_error_separation_diagnostics(&new_features, db);

    new_features
}

pub fn apply_external_ms2rescore_bounded_experts(
    features: &mut [DfFeature],
    settings: &FdrSettings,
) {
    let cfg = match settings.physical_rescue.bounded_cfg.as_ref() {
        Some(cfg) => cfg,
        None => {
            log::warn!(
                "external bounded DF experts requested, but physical_rescue.bounded_cfg is missing; no external feature update applied"
            );
            return;
        }
    };

    let profiles = ExternalMs2RescoreProfiles::learn(features, settings);

    log::info!(
        "external empirical bounded expert profiles: ms2pip_pcc={} spectral_angle={} fragment_agreement={} deeplc_abs_rt={} ccs_pct={} ccs_abs={}",
        profiles.ms2pip_pcc.summary(),
        profiles.spectral_angle.summary(),
        profiles.fragment_intensity_agreement.summary(),
        profiles.deeplc_abs_rt_error.summary(),
        profiles.ccs_pct_error.summary(),
        profiles.ccs_abs_error.summary(),
    );

    let mut n_joined = 0usize;
    let mut n_used = 0usize;
    let mut n_rescued = 0usize;
    let mut n_penalized = 0usize;
    let mut max_abs_shift = 0.0f64;

    for f in features.iter_mut().filter(|f| f.core.rank == 1) {
        let ext = f.core.external_features;

        if !ext.ms2rescore_feature_joined {
            continue;
        }

        n_joined += 1;

        let mut evidence = Vec::new();

        evidence.push(profiles.ms2pip_pcc.score(ext.ms2rescore_ms2pip_pcc as f64));
        evidence.push(
            profiles
                .spectral_angle
                .score(ext.ms2rescore_spectral_angle as f64),
        );
        evidence.push(
            profiles
                .fragment_intensity_agreement
                .score(ext.ms2rescore_fragment_intensity_agreement as f64),
        );
        evidence.push(
            profiles
                .deeplc_abs_rt_error
                .score(ext.ms2rescore_deeplc_abs_rt_error as f64),
        );
        evidence.push(
            profiles
                .ccs_pct_error
                .score(ext.tims2rescore_pct_ccs_error as f64),
        );
        evidence.push(
            profiles
                .ccs_abs_error
                .score(ext.tims2rescore_abs_ccs_error as f64),
        );

        evidence.retain(|x| x.is_finite());

        if evidence.is_empty() {
            continue;
        }

        n_used += 1;

        let mean_evidence = evidence.iter().sum::<f64>() / evidence.len() as f64;

        let raw_shift = if mean_evidence >= 0.0 {
            mean_evidence * cfg.max_rescue_shift
        } else {
            mean_evidence * cfg.max_penalty_shift
        };

        let bounded_shift =
            stats::capped_shift(raw_shift, cfg.max_rescue_shift, cfg.max_penalty_shift);

        if bounded_shift.abs() <= 1e-12 {
            continue;
        }

        max_abs_shift = max_abs_shift.max(bounded_shift.abs());

        match active_evidence_space(settings) {
            ActiveEvidenceSpace::Pep => {
                let prior_pep = f.decoy_free_pep.unwrap_or(1.0);
                let logit_prior = stats::safe_logit_confidence(prior_pep);
                let posterior_pep = stats::safe_inv_logit_confidence(logit_prior + bounded_shift);

                f.decoy_free_pep = Some(finite_df_probability_for_logit(posterior_pep));
                f.decoy_free_score = Some(df_score_from_pep(posterior_pep));
            }

            ActiveEvidenceSpace::PValue => {
                let prior_p = f.decoy_free_p_value.unwrap_or(1.0);
                let logit_prior = stats::safe_logit_confidence(prior_p);
                let posterior_p = stats::safe_inv_logit_confidence(logit_prior + bounded_shift);

                let posterior_p = finite_df_p_value(posterior_p);

                f.decoy_free_p_value = Some(posterior_p);
                f.decoy_free_pep = Some(finite_df_probability_for_logit(posterior_p));
                f.decoy_free_score = Some(df_score_from_pep(posterior_p));
            }
        }

        if bounded_shift > 0.0 {
            n_rescued += 1;
        } else {
            n_penalized += 1;
        }
    }

    recalculate_active_q_values(features, settings);

    log::info!(
        "external empirical bounded DF expert update: joined={} used={} rescued={} penalized={} max_abs_shift={:.4} rescue_cap={:.4} penalty_cap={:.4} good_anchor_q<={:.4} good_anchor_pep<={:.4} null_ranks={}..={}",
        n_joined,
        n_used,
        n_rescued,
        n_penalized,
        max_abs_shift,
        cfg.max_rescue_shift,
        cfg.max_penalty_shift,
        settings.physical_rescue.anchor_max_q,
        settings.physical_rescue.anchor_max_pep,
        settings.moments_min_null_rank,
        settings.moments_max_null_rank,
    );
}

#[derive(Clone, Debug)]
struct ExternalMs2RescoreProfiles {
    ms2pip_pcc: ExternalEmpiricalFeatureProfile,
    spectral_angle: ExternalEmpiricalFeatureProfile,
    fragment_intensity_agreement: ExternalEmpiricalFeatureProfile,
    deeplc_abs_rt_error: ExternalEmpiricalFeatureProfile,
    ccs_pct_error: ExternalEmpiricalFeatureProfile,
    ccs_abs_error: ExternalEmpiricalFeatureProfile,
}

impl ExternalMs2RescoreProfiles {
    fn learn(features: &[DfFeature], settings: &FdrSettings) -> Self {
        Self {
            ms2pip_pcc: ExternalEmpiricalFeatureProfile::learn(
                "ms2rescore_ms2pip_pcc",
                true,
                features,
                settings,
                |f| f.core.external_features.ms2rescore_ms2pip_pcc as f64,
            ),
            spectral_angle: ExternalEmpiricalFeatureProfile::learn(
                "ms2rescore_spectral_angle",
                true,
                features,
                settings,
                |f| f.core.external_features.ms2rescore_spectral_angle as f64,
            ),
            fragment_intensity_agreement: ExternalEmpiricalFeatureProfile::learn(
                "ms2rescore_fragment_intensity_agreement",
                true,
                features,
                settings,
                |f| {
                    f.core
                        .external_features
                        .ms2rescore_fragment_intensity_agreement as f64
                },
            ),
            deeplc_abs_rt_error: ExternalEmpiricalFeatureProfile::learn(
                "ms2rescore_deeplc_abs_rt_error",
                false,
                features,
                settings,
                |f| f.core.external_features.ms2rescore_deeplc_abs_rt_error as f64,
            ),
            ccs_pct_error: ExternalEmpiricalFeatureProfile::learn(
                "tims2rescore_pct_ccs_error",
                false,
                features,
                settings,
                |f| f.core.external_features.tims2rescore_pct_ccs_error as f64,
            ),
            ccs_abs_error: ExternalEmpiricalFeatureProfile::learn(
                "tims2rescore_abs_ccs_error",
                false,
                features,
                settings,
                |f| f.core.external_features.tims2rescore_abs_ccs_error as f64,
            ),
        }
    }
}

#[derive(Clone, Debug)]
struct ExternalEmpiricalFeatureProfile {
    name: &'static str,
    enabled: bool,
    higher_is_better: bool,
    good_median: f64,
    null_median: f64,
    separation: f64,
    auc: f64,
    good_n: usize,
    null_n: usize,
}

impl ExternalEmpiricalFeatureProfile {
    fn learn<F>(
        name: &'static str,
        higher_is_better: bool,
        features: &[DfFeature],
        settings: &FdrSettings,
        getter: F,
    ) -> Self
    where
        F: Fn(&DfFeature) -> f64,
    {
        const MIN_GOOD_ANCHORS: usize = 25;
        const MIN_NULL_ANCHORS: usize = 100;

        // Do not allow tiny, technically-positive differences to become active evidence.
        const MIN_EMPIRICAL_AUC: f64 = 0.58;
        const MIN_ABS_SEPARATION_FLOOR: f64 = 1e-4;
        const MIN_RELATIVE_SEPARATION_FRAC: f64 = 0.05;

        let mut good = Vec::new();
        let mut null = Vec::new();

        for f in features {
            if !f.core.external_features.ms2rescore_feature_joined {
                continue;
            }

            let x = getter(f);
            if !x.is_finite() {
                continue;
            }

            if f.core.rank == 1 {
                let q = f.decoy_free_q_value.unwrap_or(1.0);
                let pep = f.decoy_free_pep.unwrap_or(1.0);

                if q <= settings.physical_rescue.anchor_max_q
                    && pep <= settings.physical_rescue.anchor_max_pep
                {
                    good.push(x);
                }
            } else if f.core.rank >= settings.moments_min_null_rank
                && f.core.rank <= settings.moments_max_null_rank
            {
                null.push(x);
            }
        }

        let good_n = good.len();
        let null_n = null.len();

        let good_median = external_empirical_median(&good);
        let null_median = external_empirical_median(&null);

        let separation = if good_median.is_finite() && null_median.is_finite() {
            if higher_is_better {
                good_median - null_median
            } else {
                null_median - good_median
            }
        } else {
            f64::NAN
        };

        let auc = external_empirical_auc(&good, &null, higher_is_better);

        let scale_floor =
            good_median.abs().max(null_median.abs()).max(1.0) * MIN_RELATIVE_SEPARATION_FRAC;

        let min_required_separation = MIN_ABS_SEPARATION_FLOOR.max(scale_floor);

        let enabled = good_n >= MIN_GOOD_ANCHORS
            && null_n >= MIN_NULL_ANCHORS
            && separation.is_finite()
            && separation >= min_required_separation
            && auc.is_finite()
            && auc >= MIN_EMPIRICAL_AUC;

        if !enabled {
            log::warn!(
				"external empirical feature {name} disabled: good_n={} null_n={} good_median={:.6} null_median={:.6} separation={:.6} min_required_separation={:.6} auc={:.4} min_auc={:.4} higher_is_better={}",
				good_n,
				null_n,
				good_median,
				null_median,
				separation,
				min_required_separation,
				auc,
				MIN_EMPIRICAL_AUC,
				higher_is_better
			);
        }

        Self {
            name,
            enabled,
            higher_is_better,
            good_median,
            null_median,
            separation,
            auc,
            good_n,
            null_n,
        }
    }

    fn score(&self, x: f64) -> f64 {
        if !self.enabled || !x.is_finite() || !self.separation.is_finite() || self.separation <= 0.0
        {
            return f64::NAN;
        }

        let s = if self.higher_is_better {
            // null_median maps to -1; good_median maps to +1.
            2.0 * ((x - self.null_median) / self.separation) - 1.0
        } else {
            // null_median maps to -1; good_median maps to +1.
            2.0 * ((self.null_median - x) / self.separation) - 1.0
        };

        s.clamp(-1.0, 1.0)
    }

    fn summary(&self) -> String {
        format!(
			"{}:enabled={} good_n={} null_n={} good_med={:.6} null_med={:.6} sep={:.6} auc={:.4} hib={}",
			self.name,
			self.enabled,
			self.good_n,
			self.null_n,
			self.good_median,
			self.null_median,
			self.separation,
			self.auc,
			self.higher_is_better
		)
    }
}

fn external_empirical_auc(good: &[f64], null: &[f64], higher_is_better: bool) -> f64 {
    if good.is_empty() || null.is_empty() {
        return f64::NAN;
    }

    let mut wins = 0.0f64;
    let mut total = 0.0f64;

    for &g in good {
        if !g.is_finite() {
            continue;
        }

        for &n in null {
            if !n.is_finite() {
                continue;
            }

            total += 1.0;

            if higher_is_better {
                if g > n {
                    wins += 1.0;
                } else if g == n {
                    wins += 0.5;
                }
            } else if g < n {
                wins += 1.0;
            } else if g == n {
                wins += 0.5;
            }
        }
    }

    if total <= 0.0 {
        f64::NAN
    } else {
        wins / total
    }
}

fn external_empirical_median(xs: &[f64]) -> f64 {
    let mut xs = xs
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect::<Vec<_>>();

    if xs.is_empty() {
        return f64::NAN;
    }

    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

pub fn calculate_q_values(
    psms: &[DfFeature],
    settings: &FdrSettings,
    db: &IndexedDatabase,
) -> Vec<DfFeature> {
    let mut features = run_df_layers(psms, settings, db);

    let _ = calculate_peptide_q_df(&mut features, db, settings, settings.peptide_fdr);

    apply_peptide_q_to_psm_reporting_df(&mut features, settings);

    let _ = calculate_protein_q_df(&mut features, db, settings);

    let _ = apply_hierarchical_reporting_df(&mut features, db, settings);

    log_support_layer_separation_diagnostics(&features, db);

    features
}

pub fn calculate_peptide_q_df(
    features: &mut [DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
    threshold: f32,
) -> (usize, usize) {
    let threshold = threshold as f64;
    // Peptide inference consumes the finalized active DF PSM stream.
    // PEP-native final streams use decoy_free_pep; p-value-native streams use
    // decoy_free_p_value. Peptide-level aggregation uses the best supporting PSM,
    // with only bounded support from additional strong observations. Repeated
    // spectra for the same peptide are treated as corroborating evidence, not as a
    // count-based selected-min penalty.
    let is_pep_native = matches!(active_evidence_space(settings), ActiveEvidenceSpace::Pep);

    let peptide_pcombine_calibration = if is_pep_native {
        None
    } else {
        let null_pool = rank_null_p_value_pool(features, settings);
        build_empirical_pcombine_calibration(&null_pool, settings)
    };

    #[derive(Default)]
    struct PepEvidence {
        vals: Vec<f64>,
        is_entrapment: bool,
        is_reference: bool,

        best_matched_peaks: f64,
        best_longest_y_pct: f64,
        best_delta_rt_model: f64,
        best_hyperscore: f64,
        psm_count: usize,
        peptide_len: Option<f64>,
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
        let is_contam = is_contam_str(&proteins);
        let is_ref = !is_ent && !is_contam;

        let entry = peptide_evidence_map.entry(peptide).or_default();
        entry.vals.push(v);
        entry.is_entrapment |= is_ent;
        entry.is_reference |= is_ref;

        entry.psm_count += 1;
        entry
            .peptide_len
            .get_or_insert(feat.core.peptide_len as f64);

        entry.best_matched_peaks = entry.best_matched_peaks.max(feat.core.matched_peaks as f64);

        let denom = (feat.core.peptide_len as f64 - 1.0).max(1.0);
        let longest_y_pct = feat.core.longest_y as f64 / denom;
        if longest_y_pct.is_finite() {
            entry.best_longest_y_pct = entry.best_longest_y_pct.max(longest_y_pct);
        }

        let x = feat.core.delta_rt_model as f64;
        if x.is_finite() {
            if entry.best_delta_rt_model == 0.0 {
                entry.best_delta_rt_model = x.abs();
            } else {
                entry.best_delta_rt_model = entry.best_delta_rt_model.min(x.abs());
            }
        }

        let hs = feat.core.hyperscore as f64;
        if hs.is_finite() {
            entry.best_hyperscore = entry.best_hyperscore.max(hs);
        }
    }

    log::debug!(
        "DF peptide inference pool: finite_rank1_psms={} unique_peptides={}",
        finite_psm_count,
        peptide_evidence_map.len()
    );

    let mut peptide_keys = Vec::with_capacity(peptide_evidence_map.len());
    let mut peptide_combined_vals = Vec::with_capacity(peptide_evidence_map.len());
    let mut peptide_ref_vals = Vec::new();
    let mut is_ent_flags = Vec::with_capacity(peptide_evidence_map.len());

    let mut peptide_cov_values = Vec::with_capacity(peptide_evidence_map.len());

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
            combine_df_peptide_p_values(
                &ev.vals,
                settings.peptide_p_combine.clone(),
                peptide_pcombine_calibration.as_ref(),
                settings,
            )
        };

        if !is_pep_native && ev.is_reference {
            peptide_ref_vals.push(combined);
        }

        let peptide_cov = match settings.peptide_q_covariate {
            QCovariate::PeptideLen => ev.peptide_len,
            QCovariate::BestMatchedPeaks => Some(ev.best_matched_peaks),
            QCovariate::BestLongestYPct => Some(ev.best_longest_y_pct),
            QCovariate::BestDeltaRtModel => Some(ev.best_delta_rt_model),
            QCovariate::BestHyperscore => Some(ev.best_hyperscore),
            QCovariate::PsmCount => Some(ev.psm_count as f64),
            _ => None,
        };

        peptide_cov_values.push(peptide_cov);

        peptide_keys.push(peptide);
        peptide_combined_vals.push(combined);
        is_ent_flags.push(ev.is_entrapment);
    }

    let q_report = if is_pep_native {
        let q_values = match settings.peptide_q_method {
            QMethod::Auto | QMethod::Cummean => {
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
            }
            QMethod::Bh
            | QMethod::Storey
            | QMethod::By
            | QMethod::Bky
            | QMethod::Sfdr
            | QMethod::CovariateWeightedBh => {
                log::warn!(
                "DF peptide_q_method={:?} requested on PEP-native peptide evidence; using cumulative-mean PEP q-values.",
                settings.peptide_q_method
            );

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
            }
        };

        QValueComputation {
            q_values,
            requested_method: settings.peptide_q_method,
            effective_method: QMethod::Cummean,
            actual_method: "CummeanPEP",
            pi0: None,
            fallback_reason: None,
        }
    } else {
        q_values_from_level_covariates(
            &peptide_combined_vals,
            &peptide_ref_vals,
            &peptide_cov_values,
            settings,
            effective_peptide_q_method(settings),
            QLevel::Peptide,
            "peptide",
        )
    };

    log_peptide_q_diagnostics(
        &peptide_combined_vals,
        &peptide_ref_vals,
        &q_report,
        settings,
        is_pep_native,
    );

    let q_values = q_report.q_values;

    let mut best_q: FnvHashMap<String, (f64, bool)> = FnvHashMap::default();

    for i in 0..peptide_keys.len() {
        best_q.insert(
            peptide_keys[i].clone(),
            (finite_df_p_value(q_values[i]), is_ent_flags[i]),
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
        feat.decoy_free_peptide_q = Some(finite_df_p_value(q));
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

/// Optional reporting-only PSM expansion from accepted peptide discoveries.
///
/// This does not alter model-native p-values, PEPs, peptide q-values, or protein
/// q-values. It only adjusts `decoy_free_q_value` when
/// `report_psms_by_peptide_q=true`.
///
/// When `report_psms_by_peptide_q=false`, this function is a strict no-op.
/// In that default mode, downstream reporting/Level 4 logic must not assume that
/// PSM q-values were rewritten from peptide q-values.
pub fn apply_peptide_q_to_psm_reporting_df(features: &mut [DfFeature], settings: &FdrSettings) {
    if !settings.report_psms_by_peptide_q {
        log::debug!(
            "DF reporting: peptide-q PSM reporting disabled; leaving decoy_free_q_value unchanged."
        );
        return;
    }

    let mut adjusted = 0usize;

    for feat in features.iter_mut() {
        if feat.core.rank != 1 || feat.core.label != 1 {
            continue;
        }

        let Some(peptide_q) = feat.decoy_free_peptide_q else {
            continue;
        };

        if peptide_q > settings.peptide_fdr as f64 {
            continue;
        }

        let old_q = feat.decoy_free_q_value.unwrap_or(1.0);
        let new_q = old_q.min(peptide_q);

        if new_q < old_q {
            feat.decoy_free_q_value = Some(new_q);
            adjusted += 1;
        }
    }

    log::info!(
        "DF reporting: peptide-q PSM reporting enabled adjusted_rank1_target_psms={}",
        adjusted
    );
}

fn combine_second_best_p(p: &[f64]) -> f64 {
    let mut vals: Vec<f64> = p
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .map(|x| x.clamp(0.0, 1.0).max(1e-300))
        .collect();

    if vals.is_empty() {
        return 1.0;
    }

    vals.sort_by(|a, b| a.total_cmp(b));

    if vals.len() == 1 {
        vals[0]
    } else {
        vals[1]
    }
}

fn combine_cauchy(p: &[f64]) -> f64 {
    stats::combine_cauchy_acat(p).clamp(0.0, 1.0).max(1e-300)
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

fn rank_null_p_value_pool(features: &[DfFeature], settings: &FdrSettings) -> Vec<f64> {
    features
        .iter()
        .filter(|f| f.core.rank >= settings.min_null_rank && f.core.rank <= settings.max_null_rank)
        .filter_map(|f| f.decoy_free_p_value)
        .map(|p| (p as f64).clamp(1e-300, 1.0))
        .filter(|p| p.is_finite())
        .collect()
}

fn build_empirical_pcombine_calibration(
    null_pool: &[f64],
    settings: &FdrSettings,
) -> Option<stats::EmpiricalCombinerCalibration> {
    if !matches!(
        settings.p_combine_calibration_mode,
        PCombineCalibrationMode::RankNull
    ) {
        return None;
    }

    let null_pool: Vec<f64> = null_pool
        .iter()
        .copied()
        .filter(|p| p.is_finite() && *p > 0.0 && *p <= 1.0)
        .map(|p| p.clamp(1e-300, 1.0))
        .collect();

    if null_pool.len() < settings.min_null_size {
        log::warn!(
            "DF p-combine calibration skipped: null_pool={} < min_null_size={}",
            null_pool.len(),
            settings.min_null_size
        );
        return None;
    }

    let min_k = settings.p_combine_calibration_min_k.max(1);
    let max_k = settings.p_combine_calibration_max_k.max(min_k).min(100);

    let reps = settings
        .p_combine_calibration_null_replicates
        .clamp(100, 100_000);

    let tau = settings.p_combine_tfisher_tau.clamp(1e-12, 1.0);

    let mut cal = stats::EmpiricalCombinerCalibration::default();

    for k in min_k..=max_k {
        let mut p_matrix: Vec<Vec<f64>> = Vec::with_capacity(reps);
        let mut tfisher_stats = Vec::with_capacity(reps);
        let mut gfisher_stats = Vec::with_capacity(reps);
        let mut ordmeta_stats = Vec::with_capacity(reps);
        let mut evalue_stats = Vec::with_capacity(reps);

        for r in 0..reps {
            let mut row = Vec::with_capacity(k);

            // Deterministic pseudo-resampling without adding an RNG dependency.
            // The stride is prime and co-prime to most pool sizes.
            let stride = 7919usize;
            let offset = (r
                .wrapping_mul(stride)
                .wrapping_add(k.wrapping_mul(104_729usize)))
                % null_pool.len();

            for j in 0..k {
                let idx = (offset + j.wrapping_mul(stride)) % null_pool.len();
                row.push(null_pool[idx]);
            }

            tfisher_stats.push(stats::tfisher_stat(&row, tau));
            gfisher_stats.push(stats::fisher_stat(&row));
            ordmeta_stats.push(stats::ordmeta_stat(&row));
            evalue_stats.push(stats::exchangeable_evalue_stat(&row));

            p_matrix.push(row);
        }

        tfisher_stats.sort_by(|a, b| a.total_cmp(b));
        gfisher_stats.sort_by(|a, b| a.total_cmp(b));
        ordmeta_stats.sort_by(|a, b| a.total_cmp(b));
        evalue_stats.sort_by(|a, b| a.total_cmp(b));

        if let Some(bp) = stats::fit_brown_params(&p_matrix) {
            cal.brown_by_k.insert(k, bp);
        }

        cal.tfisher_by_k.insert(k, tfisher_stats);
        cal.gfisher_by_k.insert(k, gfisher_stats);
        cal.ordmeta_by_k.insert(k, ordmeta_stats);
        cal.evalue_by_k.insert(k, evalue_stats);
    }

    log::info!(
        "DF p-combine calibration built: null_pool={} k={}..{} reps={} tau={:.4}",
        null_pool.len(),
        min_k,
        max_k,
        reps,
        tau
    );

    Some(cal)
}

fn combine_df_peptide_p_values(
    vals: &[f64],
    method: PeptidePCombine,
    calibration: Option<&stats::EmpiricalCombinerCalibration>,
    settings: &FdrSettings,
) -> f64 {
    let vals: Vec<f64> = vals
        .iter()
        .copied()
        .filter(|p| p.is_finite())
        .map(|p| p.clamp(0.0, 1.0).max(1e-300))
        .collect();

    if vals.is_empty() {
        return 1.0;
    }

    match method {
        PeptidePCombine::Fisher => stats::combine_fisher(&vals),
        PeptidePCombine::Cauchy | PeptidePCombine::Acat => stats::combine_cauchy_acat(&vals),
        PeptidePCombine::SidakMinP => combine_sidak_minp(&vals),
        PeptidePCombine::BonferroniMinP => stats::combine_bonferroni_minp(&vals),
        PeptidePCombine::Tippett => stats::combine_tippett(&vals),
        PeptidePCombine::Best => vals.iter().copied().fold(1.0_f64, f64::min),
        PeptidePCombine::SecondBest => combine_second_best_p(&vals),
        PeptidePCombine::Hmp => stats::combine_hmp(&vals),
        PeptidePCombine::Brown => stats::empirical_brown_p(&vals, calibration),
        PeptidePCombine::MudholkarGeorge => stats::combine_mudholkar_george(&vals),
        PeptidePCombine::Edgington => stats::combine_edgington(&vals),
        PeptidePCombine::TFisher => {
            stats::empirical_tfisher_p(&vals, settings.p_combine_tfisher_tau, calibration)
        }
        PeptidePCombine::GFisher => stats::empirical_gfisher_p(&vals, calibration),
        PeptidePCombine::Ihw => {
            log::warn!(
                "peptide_p_combine=ihw is not a valid within-peptide p-value combiner; \
                 use peptide_q_method=covariate_weighted_bh instead. Falling back to Fisher."
            );
            stats::combine_fisher(&vals)
        }
        PeptidePCombine::ExchangeableEValue => {
            stats::empirical_exchangeable_evalue_p(&vals, calibration)
        }
        PeptidePCombine::VovkWangGeneralizedMean => stats::combine_vovk_wang_harmonic(&vals),
        PeptidePCombine::OrdmetaWFisher => stats::empirical_ordmeta_p(&vals, calibration),
        PeptidePCombine::Mcm => stats::combine_mcm(&vals),
        PeptidePCombine::Cmc => stats::combine_cmc(&vals),
    }
    .clamp(0.0, 1.0)
    .max(1e-300)
}

fn combine_df_protein_p_values(
    vals: &[f64],
    method: ProteinPCombine,
    calibration: Option<&stats::EmpiricalCombinerCalibration>,
    settings: &FdrSettings,
) -> f64 {
    let vals: Vec<f64> = vals
        .iter()
        .copied()
        .filter(|p| p.is_finite())
        .map(|p| p.clamp(0.0, 1.0).max(1e-300))
        .collect();

    if vals.is_empty() {
        return 1.0;
    }

    match method {
        ProteinPCombine::Fisher => stats::combine_fisher(&vals),
        ProteinPCombine::Cauchy | ProteinPCombine::Acat => stats::combine_cauchy_acat(&vals),
        ProteinPCombine::SidakMinP => combine_sidak_minp(&vals),
        ProteinPCombine::BonferroniMinP => stats::combine_bonferroni_minp(&vals),
        ProteinPCombine::Tippett => stats::combine_tippett(&vals),
        ProteinPCombine::Best => vals.iter().copied().fold(1.0_f64, f64::min),
        ProteinPCombine::SecondBest => combine_second_best_p(&vals),
        ProteinPCombine::Hmp => stats::combine_hmp(&vals),
        ProteinPCombine::Brown => stats::empirical_brown_p(&vals, calibration),
        ProteinPCombine::MudholkarGeorge => stats::combine_mudholkar_george(&vals),
        ProteinPCombine::Edgington => stats::combine_edgington(&vals),
        ProteinPCombine::TFisher => {
            stats::empirical_tfisher_p(&vals, settings.p_combine_tfisher_tau, calibration)
        }
        ProteinPCombine::GFisher => stats::empirical_gfisher_p(&vals, calibration),
        ProteinPCombine::Ihw => {
            log::warn!(
                "protein_p_combine=ihw is not a valid within-protein p-value combiner; \
                 use protein_q_method=covariate_weighted_bh instead. Falling back to Fisher."
            );
            stats::combine_fisher(&vals)
        }
        ProteinPCombine::ExchangeableEValue => {
            stats::empirical_exchangeable_evalue_p(&vals, calibration)
        }
        ProteinPCombine::VovkWangGeneralizedMean => stats::combine_vovk_wang_harmonic(&vals),
        ProteinPCombine::OrdmetaWFisher => stats::empirical_ordmeta_p(&vals, calibration),
        ProteinPCombine::Mcm => stats::combine_mcm(&vals),
        ProteinPCombine::Cmc => stats::combine_cmc(&vals),
    }
    .clamp(0.0, 1.0)
    .max(1e-300)
}

// DF protein aggregation/write-back contract:
// - consume only rank-1 PSMs from the finalized active stream;
// - write protein q-values only to rank-1 rows;
// - leave rank!=1 rows as None to prevent stale leakage.
pub fn calculate_protein_q_df(
    features: &mut [DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
) -> usize {
    // Protein grouping defines hypotheses only; Decoy-Free evidence aggregation
    // and q-value calibration below remain unchanged. The mode is opt-in so
    // previously validated configurations retain their raw unique-protein path.
    crate::protein_grouping::generate_protein_groups_df(
        db,
        features,
        settings.decoy_free_protein_grouping,
        Some(settings.peptide_fdr as f64),
    );

    // Protein inference consumes the peptide-passing pool derived from the finalized
    // active DF stream. Optional stages may change which PSMs pass peptide-level DF,
    // but they must not change the downstream aggregation contract.
    //
    // Base-only streams may be p-value-native. RT/IMS and reproducibility-adjusted
    // streams are PEP-native unless a valid aligned p-value stream is explicitly
    // introduced.
    let is_pep_native = matches!(active_evidence_space(settings), ActiveEvidenceSpace::Pep);

    let protein_pcombine_calibration = if is_pep_native {
        None
    } else {
        let null_pool = rank_null_p_value_pool(features, settings);
        build_empirical_pcombine_calibration(&null_pool, settings)
    };

    let mut peptide_passing_psm_count = 0usize;

    // Protein -> (peptide -> best_evidence)
    let mut protein_peptide_map: FnvHashMap<String, FnvHashMap<String, f64>> =
        FnvHashMap::default();

    for feat in features.iter().filter(|f| {
        f.core.rank == 1
            && f.core.label == 1
            && f.decoy_free_peptide_q
                .map(|q| q <= settings.peptide_fdr as f64)
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

        // Use exactly one inferred protein hypothesis. With grouping enabled,
        // an indistinguishable slash-delimited group is one hypothesis and a
        // peptide spanning multiple semicolon-delimited groups is excluded.
        // With grouping disabled, this reduces to the historical unique-protein
        // rule because raw assignments were written as fallback groups.
        let peptide = &db[feat.core.peptide_idx];
        let Some(protein_key) = df_inferred_protein_key_for_feature(feat, db) else {
            continue;
        };
        let peptide_seq = peptide.to_string();

        let peptide_map = protein_peptide_map
            .entry(protein_key.into_owned())
            .or_default();
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
    let mut protein_cov_values: Vec<Option<f64>> = Vec::new();

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
            combine_df_protein_p_values(
                &vals,
                settings.protein_p_combine.clone(),
                protein_pcombine_calibration.as_ref(),
                settings,
            )
        };

        let observed_unique_peptides = vals.len() as f64;

        let protein_cov = match settings.protein_q_covariate {
            QCovariate::ObservedUniquePeptides => Some(observed_unique_peptides),
            QCovariate::ObservedPeptideSupport => Some(observed_unique_peptides),

            QCovariate::ProteinLength => db
                .protein_metadata
                .get(&key)
                .map(|meta| meta.length as f64)
                .filter(|x| x.is_finite() && *x > 0.0),

            QCovariate::ObservableProteinPeptides => db
                .protein_metadata
                .get(&key)
                .map(|meta| meta.observable_peptides as f64)
                .filter(|x| x.is_finite() && *x > 0.0),

            QCovariate::NsafObservableLength => {
                let theoretical = db
                    .protein_metadata
                    .get(&key)
                    .map(|meta| meta.observable_peptides as f64)
                    .unwrap_or(0.0);

                if theoretical.is_finite() && theoretical > 0.0 {
                    Some(observed_unique_peptides / theoretical)
                } else {
                    None
                }
            }

            _ => None,
        };

        protein_keys.push(key);
        protein_combined_vals.push(combined);
        protein_cov_values.push(protein_cov);
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
        // P-value-native protein path.
        let mut protein_p_ref: Vec<f64> = Vec::new();
        for (key, &p) in protein_keys.iter().zip(protein_combined_vals.iter()) {
            if !is_contam_str(key) && !is_entrapment_str(key) && p.is_finite() {
                protein_p_ref.push(p.clamp(0.0, 1.0).max(1e-300));
            }
        }

        q_values_from_level_covariates(
            &protein_combined_vals,
            &protein_p_ref,
            &protein_cov_values,
            settings,
            effective_protein_q_method(settings),
            QLevel::Protein,
            "protein",
        )
        .q_values
    };

    // Map back: protein_key -> q
    let mut best_q: FnvHashMap<String, f64> = FnvHashMap::default();
    for (key, q) in protein_keys.into_iter().zip(protein_q_values) {
        best_q.insert(key, finite_df_p_value(q));
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

        let q = df_inferred_protein_key_for_feature(feat, db)
            .and_then(|protein_key| best_q.get(protein_key.as_ref()).copied())
            .unwrap_or(1.0);
        feat.decoy_free_protein_q = Some(finite_df_p_value(q));
    }

    best_q
        .iter()
        .filter(|(protein_key, &q)| {
            !is_contam_str(protein_key)
                && !is_entrapment_str(protein_key)
                && q <= settings.protein_fdr as f64
        })
        .count()
}

#[inline]
fn df_row_single_protein_hypothesis_key<'a>(
    feat: &'a DfFeature,
    db: &IndexedDatabase,
) -> Option<Cow<'a, str>> {
    // Keep Level 4 aligned with protein inference: only peptides assigned to one
    // inferred protein hypothesis can define a protein-supported row.
    df_inferred_protein_key_for_feature(feat, db)
}

#[inline]
fn df_row_passes_strict_psm_threshold(feat: &DfFeature, settings: &FdrSettings) -> bool {
    if feat.core.rank != 1 || feat.core.label != 1 {
        return false;
    }

    feat.decoy_free_q_value
        .map(|q| q <= settings.precursor_fdr as f64)
        .unwrap_or(false)
}

/// Level 4 protein-anchored reporting.
///
/// This layer is reporting-only. It does not overwrite:
/// - decoy_free_q_value
/// - decoy_free_peptide_q
/// - decoy_free_protein_q
///
/// It only writes:
/// - decoy_free_protein_supported_peptide
/// - decoy_free_peptide_supported_psm
///
/// Intended use:
///
///   The optimizer first identifies null windows with protein_ent_target_ratio == 0.
///   For such windows, the accepted protein set is treated as the trusted anchor.
///   Peptides and PSMs are then reported only if they support the accepted protein set.
///
/// Critical validation rule:
///
///   Entrapment status must not be used to decide whether a row is accepted.
///   Ent_ is only used downstream to split already accepted/reportable rows into
///   target versus entrapment counts.
pub fn apply_hierarchical_reporting_df(
    features: &mut [DfFeature],
    db: &IndexedDatabase,
    settings: &FdrSettings,
) -> (usize, usize) {
    for feat in features.iter_mut() {
        feat.decoy_free_protein_supported_peptide = None;
        feat.decoy_free_peptide_supported_psm = None;
    }

    if settings.hierarchical_reporting == HierarchicalReportingMode::Off {
        return (0, 0);
    }

    // From this point forward, Level 4 is active for this run.
    // Mark all rank-1 rows as evaluated. Non-rank-1 rows remain None.
    for feat in features.iter_mut().filter(|f| f.core.rank == 1) {
        feat.decoy_free_protein_supported_peptide = Some(false);
        feat.decoy_free_peptide_supported_psm = Some(false);
    }

    // Step 1:
    // Build the accepted protein anchor set using protein q-values only.
    //
    // Do not filter out entrapments here. If entrapment proteins pass, they must
    // remain visible so validation can detect the failure.
    let mut accepted_proteins: FnvHashSet<String> = FnvHashSet::default();

    for feat in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let protein_q = feat.decoy_free_protein_q.unwrap_or(1.0);
        if protein_q > settings.protein_fdr as f64 {
            continue;
        }

        let Some(protein_key) = df_row_single_protein_hypothesis_key(feat, db) else {
            continue;
        };

        accepted_proteins.insert(protein_key.into_owned());
    }

    if accepted_proteins.is_empty() {
        log::info!(
            "DF Level 4 protein-anchored reporting: mode={:?} accepted_proteins=0 protein_supported_peptides=0 protein_supported_psms=0",
            settings.hierarchical_reporting
        );
        return (0, 0);
    }

    let accepted_entrapment_proteins = accepted_proteins
        .iter()
        .filter(|protein_key| is_entrapment_str(protein_key))
        .count();

    // Step 2:
    // Define protein-supported peptides.
    //
    // This is the peptide-cleaning layer:
    // a peptide is reportable if it is independently accepted at peptide level
    // and maps to exactly one accepted protein hypothesis.
    //
    // Do not use Ent_ here.
    let mut protein_supported_peptides: FnvHashSet<String> = FnvHashSet::default();

    for feat in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.label == 1)
    {
        let peptide_q = feat.decoy_free_peptide_q.unwrap_or(1.0);
        if peptide_q > settings.peptide_fdr as f64 {
            continue;
        }

        let Some(protein_key) = df_row_single_protein_hypothesis_key(feat, db) else {
            continue;
        };

        if !accepted_proteins.contains(protein_key.as_ref()) {
            continue;
        }

        let peptide = &db[feat.core.peptide_idx];
        protein_supported_peptides.insert(peptide.to_string());
    }

    // Step 3:
    // Define protein-supported PSMs.
    //
    // This is intentionally protein-supported, not merely peptide-supported:
    // the PSM must independently pass the PSM threshold and map to exactly one
    // accepted protein hypothesis.
    //
    // Do not use Ent_ here.
    let mut protein_supported_psm_count = 0usize;

    for feat in features.iter_mut() {
        if feat.core.rank != 1 {
            feat.decoy_free_protein_supported_peptide = None;
            feat.decoy_free_peptide_supported_psm = None;
            continue;
        }

        if feat.core.label != 1 {
            feat.decoy_free_protein_supported_peptide = Some(false);
            feat.decoy_free_peptide_supported_psm = Some(false);
            continue;
        }

        let peptide = &db[feat.core.peptide_idx];
        let peptide_key = peptide.to_string();

        let peptide_is_protein_supported = protein_supported_peptides.contains(&peptide_key);

        let psm_is_protein_supported = df_row_passes_strict_psm_threshold(feat, settings)
            && df_row_single_protein_hypothesis_key(feat, db)
                .map(|protein_key| accepted_proteins.contains(protein_key.as_ref()))
                .unwrap_or(false);

        feat.decoy_free_protein_supported_peptide = Some(peptide_is_protein_supported);
        feat.decoy_free_peptide_supported_psm = Some(psm_is_protein_supported);

        if psm_is_protein_supported {
            protein_supported_psm_count += 1;
        }
    }

    log::info!(
        "DF Level 4 protein-anchored reporting: mode={:?} accepted_proteins={} accepted_entrapment_proteins={} protein_supported_peptides={} protein_supported_psms={}",
        settings.hierarchical_reporting,
        accepted_proteins.len(),
        accepted_entrapment_proteins,
        protein_supported_peptides.len(),
        protein_supported_psm_count
    );

    (
        protein_supported_peptides.len(),
        protein_supported_psm_count,
    )
}

pub fn decoy_free_precursor(
    peaks: &mut FnvHashMap<(PrecursorId, bool), (Peak, Vec<f64>)>,
    threshold: f32,
) -> usize {
    // Peak::default() starts at q=0. Reset every target before any fallible fit so
    // an early return cannot leak a permissive, stale q-value into output.
    for ((_, is_decoy), (peak, _)) in peaks.iter_mut() {
        if !*is_decoy {
            peak.q_value = 1.0;
        }
    }

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
    let n = null_scores.len();
    let p95_idx = ((n as f64 - 1.0) * 0.95).round() as usize;
    let (_, cap, _) = null_scores.select_nth_unstable_by(p95_idx, |a, b| a.total_cmp(b));
    let cap = (*cap).max(1e-12);

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
        .iter()
        .filter(|((_, is_decoy), (peak, _))| {
            !*is_decoy && peak.score.is_finite() && peak.q_value <= threshold
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::PeptideIx;
    use crate::peptide::Peptide;

    fn test_peak(score: f64) -> (Peak, Vec<f64>) {
        (
            Peak {
                score,
                ..Peak::default()
            },
            Vec::new(),
        )
    }

    #[test]
    fn weighted_pep_combiners_preserve_weight_alignment() {
        let peps = [f64::NAN, 0.2, 0.8];
        let weights = [100.0, 1.0, 9.0];

        let mean = combine_peps(
            &peps,
            &weights,
            EnsemblePepCombiner::WeightedMean,
            0.0,
            0.5,
            1,
            1e-12,
        );
        let median = combine_peps(
            &peps,
            &weights,
            EnsemblePepCombiner::WeightedMedian,
            0.0,
            0.5,
            1,
            1e-12,
        );

        assert!((mean - 0.74).abs() < 1e-12);
        assert_eq!(median, 0.8);
    }

    #[test]
    fn winsorized_moments_remove_gumbel_location_scale_bias() {
        const N: usize = 1 << 16;
        let expected_mu = 12.25;
        let expected_beta = 2.4;
        let scores: Vec<f64> = (0..N)
            .map(|index| {
                let probability = (index as f64 + 0.5) / N as f64;
                let standard_quantile = -(-probability.ln()).ln();
                expected_mu + expected_beta * standard_quantile
            })
            .collect();
        let winsorized = winsorize_scores_for_fit(&scores, 0.01, 0.90);

        let (corrected_mu, corrected_beta) = fit_gumbel_winsorized_moments(&winsorized, 0.01, 0.90);
        let (_, uncorrected_beta) = fit_gumbel_moments(&winsorized);

        assert!((corrected_mu - expected_mu).abs() < 1e-3);
        assert!((corrected_beta - expected_beta).abs() < 1e-3);
        assert!(uncorrected_beta < expected_beta * 0.82);
    }

    #[test]
    fn winsorized_moments_fail_closed_for_degenerate_quantiles() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let winsorized = winsorize_scores_for_fit(&scores, 0.5, 0.5);
        let (mu, beta) = fit_gumbel_winsorized_moments(&winsorized, 0.5, 0.5);
        assert!(mu.is_nan());
        assert!(beta.is_nan());
    }

    #[test]
    fn decoy_free_precursor_fails_closed_without_enough_nulls() {
        let mut peaks = FnvHashMap::default();
        peaks.insert(
            (PrecursorId::Combined(PeptideIx(0)), false),
            test_peak(10.0),
        );

        assert_eq!(decoy_free_precursor(&mut peaks, 0.01), 0);
        assert_eq!(
            peaks[&(PrecursorId::Combined(PeptideIx(0)), false)]
                .0
                .q_value,
            1.0
        );
    }

    #[test]
    fn decoy_free_precursor_counts_only_targets() {
        let mut peaks = FnvHashMap::default();
        peaks.insert(
            (PrecursorId::Combined(PeptideIx(0)), false),
            test_peak(10.0),
        );

        for i in 0..200 {
            peaks.insert(
                (PrecursorId::Combined(PeptideIx(i + 1)), true),
                test_peak(i as f64 / 200.0),
            );
        }

        assert_eq!(decoy_free_precursor(&mut peaks, 1.0), 1);
    }

    fn indistinguishable_group_fixture() -> (IndexedDatabase, Vec<DfFeature>) {
        let peptides = ["PEPTIDEA", "PEPTIDEB"]
            .into_iter()
            .map(|sequence| Peptide {
                sequence: sequence.as_bytes().to_vec().into(),
                modifications: vec![0.0; sequence.len()],
                proteins: vec![Arc::from("protA"), Arc::from("protB")],
                ..Default::default()
            })
            .collect();

        let db = IndexedDatabase {
            peptides,
            decoy_tag: "rev_".to_string(),
            generate_decoys: false,
            ..IndexedDatabase::default()
        };

        let features = [0.001, 0.002]
            .into_iter()
            .enumerate()
            .map(|(ix, pep)| {
                let mut feature = crate::scoring::FeatureCore {
                    peptide_idx: PeptideIx(ix as u32),
                    rank: 1,
                    label: 1,
                    ..Default::default()
                }
                .to_df();
                feature.decoy_free_pep = Some(pep);
                feature.decoy_free_p_value = Some(pep);
                feature.decoy_free_q_value = Some(pep);
                feature.decoy_free_peptide_q = Some(pep);
                feature
            })
            .collect();

        (db, features)
    }

    #[test]
    fn decoy_free_protein_grouping_uses_one_group_hypothesis() {
        use crate::input::{FdrMode, FdrOptions};

        let (db, mut features) = indistinguishable_group_fixture();
        let mut settings = FdrSettings::from(FdrOptions {
            mode: Some(FdrMode::DecoyFree),
            final_evidence_space: Some(FinalEvidenceSpace::Pep),
            decoy_free_protein_grouping: Some(true),
            peptide_fdr: Some(0.01),
            protein_fdr: Some(0.01),
            ..Default::default()
        });
        settings.hierarchical_reporting = HierarchicalReportingMode::Strict;

        let passing = calculate_protein_q_df(&mut features, &db, &settings);

        for feature in &features {
            assert_eq!(feature.protein_groups.as_deref(), Some("protA/protB"));
            assert_eq!(feature.num_protein_groups, 1);
            let protein_q = feature.decoy_free_protein_q.unwrap();
            assert!(
                (protein_q - 0.002).abs() < 1e-12,
                "unexpected protein q-value: {protein_q}"
            );
        }
        assert_eq!(passing, 1);

        let (protein_supported_peptides, protein_supported_psms) =
            apply_hierarchical_reporting_df(&mut features, &db, &settings);
        assert_eq!(protein_supported_peptides, 2);
        assert_eq!(protein_supported_psms, 2);
        assert!(features.iter().all(|feature| {
            feature.decoy_free_protein_supported_peptide == Some(true)
                && feature.decoy_free_peptide_supported_psm == Some(true)
        }));
    }

    #[test]
    fn decoy_free_protein_grouping_is_opt_in() {
        use crate::input::{FdrMode, FdrOptions};

        let (db, mut features) = indistinguishable_group_fixture();
        let settings = FdrSettings::from(FdrOptions {
            mode: Some(FdrMode::DecoyFree),
            final_evidence_space: Some(FinalEvidenceSpace::Pep),
            decoy_free_protein_grouping: Some(false),
            peptide_fdr: Some(0.01),
            protein_fdr: Some(0.01),
            ..Default::default()
        });

        let passing = calculate_protein_q_df(&mut features, &db, &settings);

        assert_eq!(passing, 0);
        for feature in features {
            assert_eq!(feature.protein_groups.as_deref(), Some("protA;protB"));
            assert_eq!(feature.num_protein_groups, 2);
            assert_eq!(feature.decoy_free_protein_q, Some(1.0));
        }
    }

    #[test]
    fn decoy_free_group_entrapment_counts_use_the_group_hypothesis() {
        use crate::input::{FdrMode, FdrOptions};

        let (mut db, mut features) = indistinguishable_group_fixture();
        for peptide in &mut db.peptides {
            peptide.proteins = vec![Arc::from("Ent_protA"), Arc::from("Ent_protB")];
        }

        let settings = FdrSettings::from(FdrOptions {
            mode: Some(FdrMode::DecoyFree),
            final_evidence_space: Some(FinalEvidenceSpace::Pep),
            decoy_free_protein_grouping: Some(true),
            peptide_fdr: Some(0.01),
            protein_fdr: Some(0.01),
            ..Default::default()
        });

        // Entrapment groups are evaluated and receive q-values, but are not
        // reported as target protein discoveries.
        assert_eq!(calculate_protein_q_df(&mut features, &db, &settings), 0);
        let counts = calculate_entrapment_counts_df(&features, &db, 0.01, 0.01);

        assert_eq!(counts.psms, 2);
        assert_eq!(counts.peptides, 2);
        assert_eq!(counts.proteins, 1);
        assert_eq!(
            features[0].protein_groups.as_deref(),
            Some("Ent_protA/Ent_protB")
        );
    }

    #[test]
    fn rank_null_pools_share_compact_source_and_preserve_fallback() {
        use crate::input::{FdrMode, FdrOptions};

        let rows: Vec<RankNullRow> = (0..20)
            .map(|idx| RankNullRow {
                feature_idx: idx,
                peptide_idx: idx as u32,
                rank: 2,
                score: 20.0 - idx as f64,
                charge: 2,
            })
            .collect();
        let rank1_scores_desc: Vec<(u32, f64)> =
            (0..10).map(|idx| (idx, 100.0 - idx as f64)).collect();
        let source = RankNullSource {
            rows: rows.into(),
            rank1_scores_desc: rank1_scores_desc.into(),
        };

        let mut options = FdrOptions::default();
        options.mode = Some(FdrMode::DecoyFree);
        options.model_fit = Some(ModelFit::Moments);
        options.min_null_size = Some(1);
        let settings = FdrSettings::from(options.clone());

        let purified = build_rank_null_pool(&source, &settings, 0.2, "test").unwrap();
        assert!(Arc::ptr_eq(&source.rows, &purified.source));
        assert_eq!(purified.len(), 15);
        assert!(purified.rows().all(|row| row.peptide_idx >= 5));

        options.min_null_size = Some(18);
        let fallback_settings = FdrSettings::from(options);
        let fallback = build_rank_null_pool(&source, &fallback_settings, 0.2, "test").unwrap();
        assert!(Arc::ptr_eq(&source.rows, &fallback.source));
        assert_eq!(fallback.len(), 20);
    }
}
