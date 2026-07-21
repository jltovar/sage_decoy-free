use crate::database::{IndexedDatabase, PeptideIx};
use crate::heap::bounded_min_heapify;
use crate::ion_series::{IonSeries, Kind};
use crate::mass::{Tolerance, NEUTRON, PROTON};
use crate::spectrum::{Precursor, ProcessedSpectrum};
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// --- DEBUG HELPERS ---
static DBG_EXPMASS_PRINTED: AtomicBool = AtomicBool::new(false);
fn dbg_extract_scan(spec_id: &str) -> Option<u32> {
    let key = "scan=";
    let pos = spec_id.rfind(key)? + key.len();
    let tail = &spec_id[pos..];
    let end = tail
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(tail.len());
    tail[..end].parse::<u32>().ok()
}
fn dbg_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse::<usize>().ok()
}
fn dbg_env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok()?.parse::<u32>().ok()
}
fn dbg_expmass_match(query: &ProcessedSpectrum) -> bool {
    if std::env::var("SAGE_DBG_EXPMASS").is_err() {
        return false;
    }
    let tf = match dbg_env_usize("SAGE_DBG_FILE_ID") {
        Some(v) => v,
        None => return false,
    };
    let ts = match dbg_env_u32("SAGE_DBG_SCAN") {
        Some(v) => v,
        None => return false,
    };
    if query.file_id != tf {
        return false;
    }
    match dbg_extract_scan(&query.id) {
        Some(scan) => scan == ts,
        None => false,
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum ScoreType {
    SageHyperScore,
    OpenMSHyperScore,
}

#[derive(Copy, Clone, Default, Debug, PartialEq)]
struct Score {
    peptide: PeptideIx,
    matched_b: u16,
    matched_y: u16,
    summed_b: f32,
    summed_y: f32,
    longest_b: usize,
    longest_y: usize,
    hyperscore: f64,
    ppm_difference: f32,
    precursor_charge: u8,
    isotope_error: i8,
}
impl Eq for Score {}
impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hyperscore
            .partial_cmp(&other.hyperscore)
            .unwrap_or(std::cmp::Ordering::Less)
    }
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PreScore {
    matched: u16,
    peptide: PeptideIx,
    precursor_charge: u8,
    isotope_error: i8,
}

#[derive(Clone, Default)]
struct InitialHits {
    matched_peaks: usize,
    scored_candidates: usize,
    preliminary: Vec<PreScore>,
}
impl AddAssign<InitialHits> for InitialHits {
    fn add_assign(&mut self, rhs: InitialHits) {
        self.matched_peaks += rhs.matched_peaks;
        self.scored_candidates += rhs.scored_candidates;
        self.preliminary.extend(rhs.preliminary);
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ExternalPsmFeatures {
    pub ms2rescore_ms2pip_pcc: f32,
    pub ms2rescore_spectral_angle: f32,
    pub ms2rescore_fragment_intensity_agreement: f32,

    pub ms2rescore_deeplc_predicted_rt: f32,
    pub ms2rescore_deeplc_calibrated_rt: f32,
    pub ms2rescore_deeplc_rt_error: f32,
    pub ms2rescore_deeplc_abs_rt_error: f32,

    pub tims2rescore_im2deep_predicted_ccs: f32,
    pub tims2rescore_observed_ccs: f32,
    pub tims2rescore_abs_ccs_error: f32,
    pub tims2rescore_pct_ccs_error: f32,

    pub tims2rescore_predicted_ion_mobility: f32,
    pub tims2rescore_observed_ion_mobility: f32,
    pub tims2rescore_abs_ion_mobility_error: f32,
    pub tims2rescore_pct_ion_mobility_error: f32,

    pub ms2rescore_feature_joined: bool,
}

impl Default for ExternalPsmFeatures {
    fn default() -> Self {
        Self {
            ms2rescore_ms2pip_pcc: f32::NAN,
            ms2rescore_spectral_angle: f32::NAN,
            ms2rescore_fragment_intensity_agreement: f32::NAN,

            ms2rescore_deeplc_predicted_rt: f32::NAN,
            ms2rescore_deeplc_calibrated_rt: f32::NAN,
            ms2rescore_deeplc_rt_error: f32::NAN,
            ms2rescore_deeplc_abs_rt_error: f32::NAN,

            tims2rescore_im2deep_predicted_ccs: f32::NAN,
            tims2rescore_observed_ccs: f32::NAN,
            tims2rescore_abs_ccs_error: f32::NAN,
            tims2rescore_pct_ccs_error: f32::NAN,

            tims2rescore_predicted_ion_mobility: f32::NAN,
            tims2rescore_observed_ion_mobility: f32::NAN,
            tims2rescore_abs_ion_mobility_error: f32::NAN,
            tims2rescore_pct_ion_mobility_error: f32::NAN,

            ms2rescore_feature_joined: false,
        }
    }
}

/// The core identification data produced by the search engine.
/// This struct contains NO FDR information (neither TDC nor DF).
/// It is the raw material that enters the FDR pipeline.
#[derive(Serialize, Clone, Debug, Default)]
pub struct FeatureCore {
    #[serde(skip_serializing)]
    pub peptide_idx: PeptideIx,
    pub psm_id: usize,
    pub peptide_len: usize,
    pub spec_id: String,
    pub file_id: usize,
    pub rank: u32,
    pub label: i32,
    pub expmass: f32,
    pub calcmass: f32,
    pub charge: u8,
    pub rt: f32,
    pub aligned_rt: f32,
    pub predicted_rt: f32,
    pub delta_rt_model: f32,
    pub ims: f32,
    pub predicted_ims: f32,
    pub delta_ims_model: f32,
    pub delta_mass: f32,
    pub isotope_error: f32,
    pub average_ppm: f32,
    pub hyperscore: f64,
    pub delta_next: f64,
    pub delta_best: f64,
    pub matched_peaks: u32,
    pub longest_b: u32,
    pub longest_y: u32,
    pub longest_y_pct: f32,
    pub missed_cleavages: u8,
    pub matched_intensity_pct: f32,
    pub scored_candidates: u32,
    pub poisson_log10_p_value: f64,

    // Spectrum-local LowerOrder calibration.
    //
    // These raw components are computed during scoring, while the full
    // per-spectrum candidate hyperscore distribution is still available.
    //
    // Important:
    // - `lo_spectrum_tail_p` is only the local tail probability:
    //       P(local spectrum null >= observed hyperscore)
    //
    // - `lo_spectrum_candidate_count` is the number of scored candidates for
    //   this spectrum after the same filtering used for output ranking.
    //
    // Decoy-Free LowerOrder constructs the E-value exactly once downstream:
    //
    //   E = lo_spectrum_tail_p
    //     * lo_spectrum_candidate_count.powf(lo_evalue_candidate_count_power)
    //     * lo_evalue_scale
    //
    // Do not pre-multiply by candidate count here.
    pub lo_spectrum_tail_p: f64,
    pub lo_spectrum_candidate_count: u32,

    pub ms2_intensity: f32,
    pub external_features: ExternalPsmFeatures,
    pub fragments: Option<Fragments>,
}

/// A feature augmented with standard Target-Decoy Competition (TDC) outputs.
#[derive(Serialize, Clone, Debug)]
pub struct TdcFeature {
    #[serde(flatten)]
    pub core: FeatureCore,

    // Vanilla Sage FDR columns
    pub discriminant_score: f32,
    pub posterior_error: f32,
    pub spectrum_q: f32,
    pub peptide_q: f32,
    pub protein_q: f32,
    pub protein_group_q: f32,
    pub protein_groups: Option<String>,
    pub num_protein_groups: u32,
}

/// A feature augmented with Decoy-Free (DF) outputs.
/// Vanilla TDC FDR fields are not included in this representation.
#[derive(Serialize, Clone, Debug)]
pub struct DfFeature {
    #[serde(flatten)]
    pub core: FeatureCore,

    /// Parsimonious protein-group assignment used by Decoy-Free protein
    /// inference. A slash joins indistinguishable proteins within one group;
    /// a semicolon separates multiple groups for an ambiguous peptide.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_groups: Option<String>,
    pub num_protein_groups: u32,

    // --- DECOY-FREE: Core Columns ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_peptide_q: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_protein_q: Option<f64>,

    /// Level 4 reporting-only flag.
    ///
    /// True means this rank-1 target PSM's peptide supports at least one accepted
    /// protein under the configured hierarchical reporting mode.
    ///
    /// This is not an independent peptide-level FDR claim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_protein_supported_peptide: Option<bool>,

    /// Level 4 reporting-only flag.
    ///
    /// True means this rank-1 target PSM is reportable because it supports a
    /// Level-4 protein-supported peptide.
    ///
    /// This is not an independent PSM-level FDR claim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_peptide_supported_psm: Option<bool>,

    // =========================================================================
    // Decoy-Free explicit stage snapshots
    // =========================================================================
    //
    // The live controlling stream is always:
    //   decoy_free_p_value
    //   decoy_free_pep
    //   decoy_free_score
    //   decoy_free_q_value
    //   decoy_free_peptide_q
    //   decoy_free_protein_q
    //
    // These fields preserve stage-local snapshots. They should only be populated
    // when the corresponding model/stage actually ran.

    // Base Sage Decoy-Free post-model-fit snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_base: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_base: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_base: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_base: Option<f64>,

    // RT confidence adjustment snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_rt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_rt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_rt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_rt: Option<f64>,

    // IMS confidence adjustment snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_ims: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_ims: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_ims: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_ims: Option<f64>,

    // Peptide reproducibility rescue snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_peptide_rescue: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_peptide_rescue: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_peptide_rescue: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_peptide_rescue: Option<f64>,

    // Protein reproducibility rescue snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_protein_rescue: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_protein_rescue: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_protein_rescue: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_protein_rescue: Option<f64>,

    // Transitional internal fields.
    // TODO: remove after apply_physical_rescue/apply_bounded_repro_shift are rewritten
    // to operate directly on the active decoy_free_* stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_l2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_l2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_l2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_l2: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_l3: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_l3: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_l3: Option<f64>,

    // 5D. Layer 2 diagnostics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_mode_used: Option<String>,

    // PSM-level RT/IMS residual diagnostics.
    //
    // Existing core columns already include:
    //   aligned_rt
    //   predicted_rt
    //   ims
    //   predicted_ims
    //
    // These normalized columns are written by the DF physical-diagnostics layer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_residual: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abs_rt_residual: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_z: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_within_1sigma: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_within_2sigma: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_within_3sigma: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_residual: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abs_ims_residual: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_z: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_within_1sigma: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_within_2sigma: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_within_3sigma: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_rescue_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rescued_by_rt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rescued_by_ims: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rescued_by_recurrence: Option<bool>,

    // Local RT guardrail diagnostics.
    //
    // rt_local_z is computed against a local file+RT-bin robust sigma when possible.
    // rt_training_eligible=false means the row is retained in output, but excluded
    // from RT physical training/positive RT rescue.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_local_z: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_local_outlier: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_training_eligible: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_rescue_delta: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_rescue_delta: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_shift_total: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_reliability_rt: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_reliability_ims: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_reliability_joint: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_cap_hit_pos: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_cap_hit_neg: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_anchor_eligible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dart_posterior_used: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dart_rt_lik_correct: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dart_rt_lik_incorrect: Option<f32>,

    // 5E. Layer 3 diagnostics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agreement_support: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recurrence_support: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub within_run_support: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redundancy_discount_applied: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repro_shift_total: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repro_cap_hit: Option<bool>,

    // --- DECOY-FREE: Per-method outputs (p / q / pep) ---
    // Moments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_mom: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_mom: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_mom: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_mom: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_mom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_mom: Option<f64>,

    // MLE
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_mle: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_mle: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_mle: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_mle: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_mle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_mle: Option<f64>,

    // Lower Order
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_lo: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_lo: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_lo: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_lo: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_lo: Option<f64>,

    // MSFDR (seeded / legacy)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_msfdr: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_msfdr: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_msfdr: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_msfdr: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_msfdr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_msfdr: Option<f64>,

    // MSFDR (1-state mixture)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_1smix: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_1smix: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_1smix: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_1smix: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_1smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_1smix: Option<f64>,

    // MSFDR (2-state mixture)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_2smix: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_2smix: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_2smix: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_2smix: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_2smix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_2smix: Option<f64>,

    // Nokoi
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_nokoi: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_nokoi: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_nokoi: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_nokoi: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_nokoi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_nokoi: Option<f64>,

    // Ensemble consensus stream
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_ensemble: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_ensemble: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_ensemble: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_ensemble: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_ensemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_ensemble: Option<f64>,
}

// Conversion helpers
impl FeatureCore {
    pub fn to_tdc(self) -> TdcFeature {
        TdcFeature {
            core: self,
            discriminant_score: 0.0,
            posterior_error: 1.0,
            spectrum_q: 1.0,
            peptide_q: 1.0,
            protein_q: 1.0,
            protein_group_q: 1.0,
            protein_groups: None,
            num_protein_groups: 0,
        }
    }

    pub fn to_df(self) -> DfFeature {
        DfFeature {
            core: self,
            protein_groups: None,
            num_protein_groups: 0,
            decoy_free_p_value: None,
            decoy_free_pep: None,
            decoy_free_score: None,
            decoy_free_q_value: None,
            decoy_free_peptide_q: None,
            decoy_free_protein_q: None,
            decoy_free_protein_supported_peptide: None,
            decoy_free_peptide_supported_psm: None,

            p_mom: None,
            q_mom: None,
            pep_mom: None,

            p_mle: None,
            q_mle: None,
            pep_mle: None,

            p_lo: None,
            q_lo: None,
            pep_lo: None,

            p_msfdr: None,
            q_msfdr: None,
            pep_msfdr: None,

            p_1smix: None,
            q_1smix: None,
            pep_1smix: None,

            p_2smix: None,
            q_2smix: None,
            pep_2smix: None,

            p_nokoi: None,
            q_nokoi: None,
            pep_nokoi: None,

            rt_adjust_p_mom: None,
            rt_adjust_q_mom: None,
            rt_adjust_pep_mom: None,

            ims_adjust_p_mom: None,
            ims_adjust_q_mom: None,
            ims_adjust_pep_mom: None,

            peptide_rescue_p_mom: None,
            peptide_rescue_q_mom: None,
            peptide_rescue_pep_mom: None,

            protein_rescue_p_mom: None,
            protein_rescue_q_mom: None,
            protein_rescue_pep_mom: None,

            rt_adjust_p_mle: None,
            rt_adjust_q_mle: None,
            rt_adjust_pep_mle: None,

            ims_adjust_p_mle: None,
            ims_adjust_q_mle: None,
            ims_adjust_pep_mle: None,

            peptide_rescue_p_mle: None,
            peptide_rescue_q_mle: None,
            peptide_rescue_pep_mle: None,

            protein_rescue_p_mle: None,
            protein_rescue_q_mle: None,
            protein_rescue_pep_mle: None,

            rt_adjust_p_lo: None,
            rt_adjust_q_lo: None,
            rt_adjust_pep_lo: None,

            ims_adjust_p_lo: None,
            ims_adjust_q_lo: None,
            ims_adjust_pep_lo: None,

            peptide_rescue_p_lo: None,
            peptide_rescue_q_lo: None,
            peptide_rescue_pep_lo: None,

            protein_rescue_p_lo: None,
            protein_rescue_q_lo: None,
            protein_rescue_pep_lo: None,

            rt_adjust_p_msfdr: None,
            rt_adjust_q_msfdr: None,
            rt_adjust_pep_msfdr: None,

            ims_adjust_p_msfdr: None,
            ims_adjust_q_msfdr: None,
            ims_adjust_pep_msfdr: None,

            peptide_rescue_p_msfdr: None,
            peptide_rescue_q_msfdr: None,
            peptide_rescue_pep_msfdr: None,

            protein_rescue_p_msfdr: None,
            protein_rescue_q_msfdr: None,
            protein_rescue_pep_msfdr: None,

            rt_adjust_p_1smix: None,
            rt_adjust_q_1smix: None,
            rt_adjust_pep_1smix: None,

            ims_adjust_p_1smix: None,
            ims_adjust_q_1smix: None,
            ims_adjust_pep_1smix: None,

            peptide_rescue_p_1smix: None,
            peptide_rescue_q_1smix: None,
            peptide_rescue_pep_1smix: None,

            protein_rescue_p_1smix: None,
            protein_rescue_q_1smix: None,
            protein_rescue_pep_1smix: None,

            rt_adjust_p_2smix: None,
            rt_adjust_q_2smix: None,
            rt_adjust_pep_2smix: None,

            ims_adjust_p_2smix: None,
            ims_adjust_q_2smix: None,
            ims_adjust_pep_2smix: None,

            peptide_rescue_p_2smix: None,
            peptide_rescue_q_2smix: None,
            peptide_rescue_pep_2smix: None,

            protein_rescue_p_2smix: None,
            protein_rescue_q_2smix: None,
            protein_rescue_pep_2smix: None,

            rt_adjust_p_nokoi: None,
            rt_adjust_q_nokoi: None,
            rt_adjust_pep_nokoi: None,

            ims_adjust_p_nokoi: None,
            ims_adjust_q_nokoi: None,
            ims_adjust_pep_nokoi: None,

            peptide_rescue_p_nokoi: None,
            peptide_rescue_q_nokoi: None,
            peptide_rescue_pep_nokoi: None,

            protein_rescue_p_nokoi: None,
            protein_rescue_q_nokoi: None,
            protein_rescue_pep_nokoi: None,

            p_ensemble: None,
            q_ensemble: None,
            pep_ensemble: None,
            score_ensemble: None,

            rt_adjust_p_ensemble: None,
            rt_adjust_q_ensemble: None,
            rt_adjust_pep_ensemble: None,

            ims_adjust_p_ensemble: None,
            ims_adjust_q_ensemble: None,
            ims_adjust_pep_ensemble: None,

            peptide_rescue_p_ensemble: None,
            peptide_rescue_q_ensemble: None,
            peptide_rescue_pep_ensemble: None,

            protein_rescue_p_ensemble: None,
            protein_rescue_q_ensemble: None,
            protein_rescue_pep_ensemble: None,

            // Initialize additional Decoy-Free layer fields
            decoy_free_p_value_base: None,
            decoy_free_pep_base: None,
            decoy_free_score_base: None,
            decoy_free_q_base: None,

            decoy_free_p_value_rt: None,
            decoy_free_pep_rt: None,
            decoy_free_score_rt: None,
            decoy_free_q_rt: None,

            decoy_free_p_value_ims: None,
            decoy_free_pep_ims: None,
            decoy_free_score_ims: None,
            decoy_free_q_ims: None,

            decoy_free_p_value_peptide_rescue: None,
            decoy_free_pep_peptide_rescue: None,
            decoy_free_score_peptide_rescue: None,
            decoy_free_q_peptide_rescue: None,

            decoy_free_p_value_protein_rescue: None,
            decoy_free_pep_protein_rescue: None,
            decoy_free_score_protein_rescue: None,
            decoy_free_q_protein_rescue: None,

            decoy_free_p_value_l2: None,
            decoy_free_pep_l2: None,
            decoy_free_score_l2: None,
            decoy_free_q_l2: None,

            decoy_free_pep_l3: None,
            decoy_free_score_l3: None,
            decoy_free_q_l3: None,

            physical_mode_used: None,

            rt_residual: None,
            abs_rt_residual: None,
            rt_z: None,
            rt_within_1sigma: None,
            rt_within_2sigma: None,
            rt_within_3sigma: None,

            ims_residual: None,
            abs_ims_residual: None,
            ims_z: None,
            ims_within_1sigma: None,
            ims_within_2sigma: None,
            ims_within_3sigma: None,

            physical_rescue_source: None,
            rescued_by_rt: None,
            rescued_by_ims: None,
            rescued_by_recurrence: None,

            rt_local_z: None,
            rt_local_outlier: None,
            rt_training_eligible: None,

            rt_rescue_delta: None,
            ims_rescue_delta: None,
            physical_shift_total: None,
            physical_reliability_rt: None,
            physical_reliability_ims: None,
            physical_reliability_joint: None,
            physical_cap_hit_pos: None,
            physical_cap_hit_neg: None,
            physical_anchor_eligible: None,
            dart_posterior_used: None,
            dart_rt_lik_correct: None,
            dart_rt_lik_incorrect: None,

            agreement_support: None,
            recurrence_support: None,
            within_run_support: None,
            redundancy_discount_applied: None,
            repro_shift_total: None,
            repro_cap_hit: None,
        }
    }
}

#[derive(Serialize, Default, Clone, Debug)]
pub struct Fragments {
    /// Observed fragment charge state.
    #[serde(skip_serializing)]
    pub charges: Vec<i32>,
    pub kinds: Vec<Kind>,
    pub fragment_ordinals: Vec<i32>,
    pub intensities: Vec<f32>,
    pub mz_calculated: Vec<f32>,
    pub mz_experimental: Vec<f32>,
}

static PSM_COUNTER: AtomicUsize = AtomicUsize::new(1);
fn increment_psm_counter() -> usize {
    PSM_COUNTER.fetch_add(1, Ordering::Relaxed)
}
fn lnfact(n: u16) -> f64 {
    if n <= 1 {
        0.0
    } else {
        (2..=n as u32).map(|i| (i as f64).ln()).sum()
    }
}

fn poisson_sf_geq(k: u16, lambda: f64) -> f64 {
    if !lambda.is_finite() || lambda < 0.0 {
        return 1.0;
    }
    if k == 0 {
        return 1.0;
    }
    if lambda == 0.0 {
        return 0.0;
    }

    let mut term = (k as f64 * lambda.ln() - lambda - lnfact(k)).exp();
    let mut tail = term;
    let mut i = k as u32;

    loop {
        i += 1;
        term *= lambda / i as f64;
        let next = tail + term;
        if next == tail {
            break;
        }
        tail = next;
        if i > 100_000 {
            break;
        }
    }

    tail.clamp(0.0, 1.0)
}

impl ScoreType {
    pub fn score(&self, matched_b: u16, matched_y: u16, summed_b: f32, summed_y: f32) -> f64 {
        let score = match self {
            Self::SageHyperScore => {
                let i = (summed_b + 1.0) as f64 * (summed_y + 1.0) as f64;
                i.ln() + lnfact(matched_b) + lnfact(matched_y)
            }
            Self::OpenMSHyperScore => {
                let summed_intensity = summed_b + summed_y;
                summed_intensity.ln_1p() as f64 + lnfact(matched_b) + lnfact(matched_y)
            }
        };
        if score.is_finite() {
            score
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl Score {
    fn hyperscore(&self, score_type: ScoreType) -> f64 {
        score_type.score(self.matched_b, self.matched_y, self.summed_b, self.summed_y)
    }
}

pub struct Scorer<'db> {
    pub db: &'db IndexedDatabase,
    pub precursor_tol: Tolerance,
    pub fragment_tol: Tolerance,
    pub min_matched_peaks: u16,
    pub min_isotope_err: i8,
    pub max_isotope_err: i8,
    pub min_precursor_charge: u8,
    pub max_precursor_charge: u8,
    pub override_precursor_charge: bool,
    pub max_fragment_charge: Option<u8>,
    pub chimera: bool,
    pub report_psms: usize,
    pub wide_window: bool,
    pub annotate_matches: bool,
    pub score_type: ScoreType,
}

#[inline(always)]
fn first_precursor(query: &ProcessedSpectrum) -> Option<&Precursor> {
    let precursor = query.precursors.first();
    if precursor.is_none() {
        eprintln!(
            "[sage] skipping spectrum without MS1 precursor metadata: {}",
            query.id
        );
    }
    precursor
}

const LO_LOCAL_MIN_CANDIDATES: usize = 10;
const LO_LOCAL_GUMBEL_EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

#[inline(always)]
fn lo_local_gumbel_nll(mu: f64, beta: f64, scores: &[f64]) -> f64 {
    if !mu.is_finite() || !beta.is_finite() || beta <= 0.0 || scores.is_empty() {
        return f64::INFINITY;
    }

    let log_beta = beta.ln();
    let mut nll = 0.0f64;

    for &x in scores {
        if !x.is_finite() {
            return f64::INFINITY;
        }

        let z = (x - mu) / beta;
        if !z.is_finite() {
            return f64::INFINITY;
        }

        let ez = (-z).clamp(-745.0, 745.0).exp();
        nll += log_beta + z + ez;
    }

    nll
}

fn fit_lo_local_gumbel_mle(scores: &[f64]) -> Option<(f64, f64)> {
    if scores.len() < LO_LOCAL_MIN_CANDIDATES {
        return None;
    }

    let n = scores.len() as f64;
    let mean = scores.iter().copied().sum::<f64>() / n;

    let var = scores
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>()
        / n.max(1.0);

    if !mean.is_finite() || !var.is_finite() || var <= 0.0 {
        return None;
    }

    let beta0 = (var.sqrt() * (6.0_f64).sqrt() / std::f64::consts::PI).max(1e-6);
    let mu0 = mean - LO_LOCAL_GUMBEL_EULER_GAMMA * beta0;

    let mut best_mu = mu0;
    let mut best_log_beta = beta0.ln();
    let mut best_nll = lo_local_gumbel_nll(best_mu, best_log_beta.exp(), scores);

    if !best_nll.is_finite() {
        return None;
    }

    let score_min = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let score_max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (score_max - score_min).abs().max(1.0);

    let mut step_mu = 0.25 * span;
    let mut step_log_beta = 0.25f64;

    for _ in 0..80 {
        let mut improved = false;

        let candidates = [
            (best_mu - step_mu, best_log_beta),
            (best_mu + step_mu, best_log_beta),
            (best_mu, best_log_beta - step_log_beta),
            (best_mu, best_log_beta + step_log_beta),
            (best_mu - step_mu, best_log_beta - step_log_beta),
            (best_mu - step_mu, best_log_beta + step_log_beta),
            (best_mu + step_mu, best_log_beta - step_log_beta),
            (best_mu + step_mu, best_log_beta + step_log_beta),
        ];

        for (mu, log_beta) in candidates {
            let beta = log_beta.exp();
            let nll = lo_local_gumbel_nll(mu, beta, scores);

            if nll.is_finite() && nll < best_nll {
                best_mu = mu;
                best_log_beta = log_beta;
                best_nll = nll;
                improved = true;
            }
        }

        if !improved {
            step_mu *= 0.5;
            step_log_beta *= 0.5;

            if step_mu < 1e-8 && step_log_beta < 1e-8 {
                break;
            }
        }
    }

    let beta = best_log_beta.exp();

    if best_mu.is_finite() && beta.is_finite() && beta > 0.0 {
        Some((best_mu, beta))
    } else {
        None
    }
}

#[inline(always)]
fn lo_local_gumbel_survival(score: f64, mu: f64, beta: f64) -> Option<f64> {
    if !score.is_finite() || !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    let z = (score - mu) / beta;
    if !z.is_finite() {
        return None;
    }

    let t = (-z).clamp(-745.0, 745.0).exp();
    let p = (-(-t).exp_m1()).clamp(1e-300, 1.0);

    p.is_finite().then_some(p)
}

fn assign_lo_spectrum_tail_components(
    score_vector: &[(Score, Option<Fragments>)],
) -> Vec<(f64, u32)> {
    let n_candidates = score_vector.len().max(1);
    let fail_closed = || vec![(1.0f64, n_candidates as u32); score_vector.len()];

    if score_vector.len() < LO_LOCAL_MIN_CANDIDATES {
        return fail_closed();
    }

    // Exclude rank 1 from the local null fit. Rank 1 is the candidate most likely
    // to contain true target signal.
    let null_scores: Vec<f64> = score_vector
        .iter()
        .enumerate()
        .filter_map(|(idx, (score, _))| {
            if idx == 0 {
                None
            } else if score.hyperscore.is_finite() {
                Some(score.hyperscore)
            } else {
                None
            }
        })
        .collect();

    if null_scores.len() < LO_LOCAL_MIN_CANDIDATES {
        return fail_closed();
    }

    let Some((mu, beta)) = fit_lo_local_gumbel_mle(&null_scores) else {
        return fail_closed();
    };

    score_vector
        .iter()
        .map(|(score, _)| {
            let tail_p = lo_local_gumbel_survival(score.hyperscore, mu, beta)
                .unwrap_or(1.0)
                .clamp(1e-300, 1.0);

            (tail_p, n_candidates as u32)
        })
        .collect()
}

impl<'db> Scorer<'db> {
    #[inline(always)]
    fn resolved_max_fragment_charge(&self, precursor_charge: u8) -> u8 {
        precursor_charge
            .min(
                self.max_fragment_charge
                    .map(|c| c + 1)
                    .unwrap_or(precursor_charge),
            )
            .max(2)
    }

    pub fn quick_score(
        &self,
        query: &ProcessedSpectrum,
        prefilter_low_memory: bool,
    ) -> Vec<PeptideIx> {
        assert_eq!(
            query.level, 2,
            "internal bug, trying to score a non-MS2 scan!"
        );
        let Some(precursor) = first_precursor(query) else {
            return Vec::new();
        };
        let hits = self.initial_hits(query, precursor);
        if prefilter_low_memory {
            let mut score_vector = hits
                .preliminary
                .iter()
                .filter_map(|pre| {
                    if pre.peptide == PeptideIx::default() {
                        return None;
                    }
                    let (score, _) = self.score_candidate(query, pre);
                    if (score.matched_b + score.matched_y) < self.min_matched_peaks {
                        return None;
                    }
                    Some(score)
                })
                .collect::<Vec<_>>();
            let k = self.report_psms.min(score_vector.len()) + 1;
            bounded_min_heapify(&mut score_vector, k);
            score_vector.iter().map(|x| x.peptide).collect()
        } else {
            hits.preliminary
                .iter()
                .map(|x| x.peptide)
                .filter(|&peptide| peptide != PeptideIx::default())
                .collect()
        }
    }

    pub fn score(&self, query: &ProcessedSpectrum) -> Vec<FeatureCore> {
        assert_eq!(
            query.level, 2,
            "internal bug, trying to score a non-MS2 scan!"
        );
        match self.chimera {
            true => self.score_chimera_fast(query),
            false => self.score_standard(query),
        }
    }

    fn trim_hits(&self, hits: &mut InitialHits) {
        let k = 50.clamp(
            (self.report_psms * 2).min(hits.preliminary.len()),
            hits.preliminary.len(),
        );
        bounded_min_heapify(&mut hits.preliminary, k);
        hits.preliminary.truncate(k);
    }

    fn matched_peaks_with_isotope(
        &self,
        query: &ProcessedSpectrum,
        precursor_mass: f32,
        precursor_charge: u8,
        precursor_tol: Tolerance,
        isotope_error: i8,
    ) -> InitialHits {
        let candidates = self.db.query(
            precursor_mass - isotope_error as f32 * NEUTRON,
            precursor_tol,
            self.fragment_tol,
        );
        let max_fragment_charge = self.resolved_max_fragment_charge(precursor_charge);
        let potential = candidates.pre_idx_hi - candidates.pre_idx_lo + 1;
        let mut hits = InitialHits {
            matched_peaks: 0,
            scored_candidates: 0,
            preliminary: vec![PreScore::default(); potential],
        };
        for peak_mass in query.masses.iter() {
            for charge in 1..max_fragment_charge {
                let mass = peak_mass * charge as f32;
                for frag in candidates.page_search(mass) {
                    let idx = frag.peptide_index.0 as usize - candidates.pre_idx_lo;
                    let sc = &mut hits.preliminary[idx];
                    if sc.matched == 0 {
                        hits.scored_candidates += 1;
                        sc.precursor_charge = precursor_charge;
                        sc.peptide = frag.peptide_index;
                        sc.isotope_error = isotope_error;
                    }
                    sc.matched += 1;
                    hits.matched_peaks += 1;
                }
            }
        }
        if hits.matched_peaks == 0 {
            return hits;
        }
        self.trim_hits(&mut hits);
        hits
    }

    fn matched_peaks(
        &self,
        query: &ProcessedSpectrum,
        precursor_mass: f32,
        precursor_charge: u8,
        precursor_tol: Tolerance,
    ) -> InitialHits {
        if self.min_isotope_err != self.max_isotope_err {
            let mut hits = (self.min_isotope_err..=self.max_isotope_err).fold(
                InitialHits::default(),
                |mut hits, isotope| {
                    hits += self.matched_peaks_with_isotope(
                        query,
                        precursor_mass,
                        precursor_charge,
                        precursor_tol,
                        isotope,
                    );
                    hits
                },
            );
            self.trim_hits(&mut hits);
            hits
        } else {
            self.matched_peaks_with_isotope(
                query,
                precursor_mass,
                precursor_charge,
                precursor_tol,
                0,
            )
        }
    }

    fn initial_hits(&self, query: &ProcessedSpectrum, precursor: &Precursor) -> InitialHits {
        // Sage operates on masses without protons; [M] instead of [MH+]
        let mz = precursor.mz - PROTON;
        if dbg_expmass_match(query) && !DBG_EXPMASS_PRINTED.load(Ordering::Relaxed) {
            eprintln!("[DBG_PRECURSOR] spec_id={} ...", query.id);
        }

        let mut hits = if self.wide_window {
            (self.min_precursor_charge..=self.max_precursor_charge).fold(
                InitialHits::default(),
                |mut hits, precursor_charge| {
                    let precursor_mass = mz * precursor_charge as f32;
                    let precursor_tol = precursor
                        .isolation_window
                        .unwrap_or(Tolerance::Da(-2.4, 2.4))
                        * precursor_charge as f32;
                    hits +=
                        self.matched_peaks(query, precursor_mass, precursor_charge, precursor_tol);
                    hits
                },
            )
        } else if precursor.charge.is_some() && !self.override_precursor_charge {
            let charge = precursor.charge.unwrap();
            let precursor_mass = mz * charge as f32;
            self.matched_peaks(query, precursor_mass, charge, self.precursor_tol)
        } else {
            (self.min_precursor_charge..=self.max_precursor_charge).fold(
                InitialHits::default(),
                |mut hits, precursor_charge| {
                    let precursor_mass = mz * precursor_charge as f32;
                    hits += self.matched_peaks(
                        query,
                        precursor_mass,
                        precursor_charge,
                        self.precursor_tol,
                    );
                    hits
                },
            )
        };
        self.trim_hits(&mut hits);
        hits
    }

    pub fn score_standard(&self, query: &ProcessedSpectrum) -> Vec<FeatureCore> {
        let Some(precursor) = first_precursor(query) else {
            return Vec::new();
        };
        let hits = self.initial_hits(query, precursor);
        let mut features = Vec::with_capacity(self.report_psms);
        self.build_features(query, precursor, &hits, self.report_psms, &mut features);
        features
    }

    fn build_features(
        &self,
        query: &ProcessedSpectrum,
        precursor: &Precursor,
        hits: &InitialHits,
        report_psms: usize,
        features: &mut Vec<FeatureCore>,
    ) {
        if dbg_expmass_match(query) && !DBG_EXPMASS_PRINTED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "\n[SAGE_DBG] HIT file_id={} scan={}",
                query.file_id,
                dbg_extract_scan(&query.id).unwrap_or(0)
            );
        }

        let mut score_vector = hits
            .preliminary
            .iter()
            .filter(|score| score.peptide != PeptideIx::default())
            .map(|pre| self.score_candidate(query, pre))
            .filter(|s| (s.0.matched_b + s.0.matched_y) >= self.min_matched_peaks)
            .collect::<Vec<_>>();

        score_vector.sort_by(|a, b| b.0.hyperscore.total_cmp(&a.0.hyperscore));

        // Compute spectrum-local LO tail components before output pruning.
        // This is the only point where the full per-spectrum scored candidate
        // hyperscore distribution is still available.
        let lo_spectrum_tail_components = assign_lo_spectrum_tail_components(&score_vector);

        let scored_len = score_vector.len().max(1) as f64;

        // Fix the old lambda fragility: estimate the matched-peak-count background
        // from the same fully scored, min_matched_peaks-filtered candidate population
        // that is used for ranking/output, not from preliminary hit bookkeeping.
        let filtered_matched_peaks: u32 = score_vector
            .iter()
            .map(|(score, _)| (score.matched_b + score.matched_y) as u32)
            .sum();

        let lambda = filtered_matched_peaks as f64 / scored_len;

        let mz = precursor.mz - PROTON;

        for idx in 0..report_psms.min(score_vector.len()) {
            let score = score_vector[idx].0;

            let (lo_spectrum_tail_p, lo_spectrum_candidate_count) = lo_spectrum_tail_components
                .get(idx)
                .copied()
                .unwrap_or((1.0, score_vector.len().max(1) as u32));

            let fragments: Option<Fragments> = score_vector[idx].1.take();
            let psm_id = increment_psm_counter();
            let peptide = &self.db[score.peptide];
            let precursor_mass = mz * score.precursor_charge as f32;

            let next = score_vector
                .get(idx + 1)
                .map(|score| score.0.hyperscore)
                .unwrap_or_default();
            let best = score_vector
                .first()
                .map(|score| score.0.hyperscore)
                .expect("valid index 0");
            let k = score.matched_b + score.matched_y;

            // Keep the matched-peak Poisson value for the existing diagnostic column only.
            let matched_peak_poisson_p_value = poisson_sf_geq(k, lambda).max(1e-325);

            let isotope_error = score.isotope_error as f32 * NEUTRON;
            let delta_mass = (precursor_mass - peptide.monoisotopic - isotope_error) * 2E6
                / (precursor_mass - isotope_error + peptide.monoisotopic);

            let matched_intensity = score.summed_b + score.summed_y;
            let matched_intensity_pct = if query.total_ion_current > 0.0 {
                100.0 * matched_intensity / query.total_ion_current
            } else {
                0.0
            };

            let fragment_positions = peptide.sequence.len().saturating_sub(1);
            let longest_y_pct = if fragment_positions > 0 {
                score.longest_y as f32 / fragment_positions as f32
            } else {
                0.0
            };

            features.push(FeatureCore {
                psm_id,
                peptide_idx: score.peptide,
                spec_id: query.id.clone(),
                file_id: query.file_id,
                rank: idx as u32 + 1,
                label: peptide.label(),
                expmass: precursor_mass,
                calcmass: peptide.monoisotopic,
                charge: score.precursor_charge,
                rt: query.scan_start_time,
                ims: precursor.inverse_ion_mobility.unwrap_or(0.0),
                delta_mass,
                isotope_error,
                average_ppm: score.ppm_difference,
                hyperscore: score.hyperscore,
                delta_next: score.hyperscore - next,
                delta_best: best - score.hyperscore,
                matched_peaks: k as u32,
                matched_intensity_pct,
                poisson_log10_p_value: matched_peak_poisson_p_value.log10(),
                longest_b: score.longest_b as u32,
                longest_y: score.longest_y as u32,
                longest_y_pct,
                peptide_len: peptide.sequence.len(),
                scored_candidates: hits.scored_candidates as u32,
                lo_spectrum_tail_p,
                lo_spectrum_candidate_count,
                missed_cleavages: peptide.missed_cleavages,
                predicted_rt: 0.0,
                predicted_ims: 0.0,
                aligned_rt: query.scan_start_time,
                delta_rt_model: 0.999,
                delta_ims_model: 0.999,
                ms2_intensity: matched_intensity,
                external_features: ExternalPsmFeatures::default(),
                fragments,
            })
        }
    }

    /// Remove peaks matching a PSM from a query spectrum
    fn remove_matched_peaks(&self, query: &mut ProcessedSpectrum, psm: &FeatureCore) {
        let peptide = &self.db[psm.peptide_idx];
        let fragments = self
            .db
            .ion_kinds
            .iter()
            .flat_map(|kind| IonSeries::new(peptide, *kind));
        let max_fragment_charge = self.resolved_max_fragment_charge(psm.charge);
        let mut to_remove = Vec::new();
        for frag in fragments {
            for charge in 1..max_fragment_charge {
                // Experimental peaks are multipled by charge, therefore theoretical are divided
                if let Some(peak_idx) = crate::spectrum::select_most_intense_peak(
                    &query.masses,
                    &query.intensities,
                    frag.monoisotopic_mass / charge as f32,
                    self.fragment_tol,
                    None,
                ) {
                    to_remove.push((
                        query.masses[peak_idx],
                        query.intensities[peak_idx],
                        query.charges[peak_idx],
                    ));
                }
            }
        }
        let mut masses = Vec::with_capacity(query.masses.len());
        let mut intensities = Vec::with_capacity(query.intensities.len());
        let mut charges = Vec::with_capacity(query.charges.len());
        let mut mobilities = Vec::with_capacity(query.mobilities.len());

        for idx in 0..query.masses.len() {
            let peak = (
                query.masses[idx],
                query.intensities[idx],
                query.charges[idx],
            );
            if !to_remove.contains(&peak) {
                masses.push(query.masses[idx]);
                intensities.push(query.intensities[idx]);
                charges.push(query.charges[idx]);
                if !query.mobilities.is_empty() {
                    mobilities.push(query.mobilities[idx]);
                }
            }
        }

        query.masses = masses;
        query.intensities = intensities;
        query.charges = charges;
        query.mobilities = mobilities;
        query.total_ion_current = query.intensities.iter().sum::<f32>();
    }

    /// Return multiple PSMs for each spectra - first is the best match, second PSM is the best match
    /// after all theoretical peaks assigned to the best match are removed, etc
    pub fn score_chimera_fast(&self, query: &ProcessedSpectrum) -> Vec<FeatureCore> {
        let Some(precursor) = first_precursor(query) else {
            return Vec::new();
        };
        let mut query = query.clone();
        let hits = self.initial_hits(&query, precursor);
        let mut candidates: Vec<FeatureCore> = Vec::with_capacity(self.report_psms);
        let mut prev = 0;
        while candidates.len() < self.report_psms {
            self.build_features(&query, precursor, &hits, 1, &mut candidates);
            if candidates.len() > prev {
                if let Some(feat) = candidates.get_mut(prev) {
                    self.remove_matched_peaks(&mut query, feat);
                    feat.rank = prev as u32 + 1;
                }
                prev = candidates.len()
            } else {
                break;
            }
        }
        candidates
    }

    fn score_candidate(
        &self,
        query: &ProcessedSpectrum,
        pre_score: &PreScore,
    ) -> (Score, Option<Fragments>) {
        let mut score = Score {
            peptide: pre_score.peptide,
            precursor_charge: pre_score.precursor_charge,
            isotope_error: pre_score.isotope_error,
            ..Default::default()
        };
        let peptide = &self.db[score.peptide];
        let max_fragment_charge = self.resolved_max_fragment_charge(score.precursor_charge);
        let fragments = self
            .db
            .ion_kinds
            .iter()
            .flat_map(|kind| IonSeries::new(peptide, *kind).enumerate());
        let mut b_run = Run::default();
        let mut y_run = Run::default();
        let mut fragments_details = Fragments::default();

        for (idx, frag) in fragments {
            for charge in 1..max_fragment_charge {
                let mz = frag.monoisotopic_mass / charge as f32;
                if let Some(peak_idx) = crate::spectrum::select_most_intense_peak(
                    &query.masses,
                    &query.intensities,
                    mz,
                    self.fragment_tol,
                    None,
                ) {
                    let peak_mass = query.masses[peak_idx];
                    let peak_intensity = query.intensities[peak_idx];
                    let fragment_charge = query.charges[peak_idx].max(charge);

                    score.ppm_difference +=
                        peak_intensity * (mz - peak_mass).abs() * 2E6 / (mz + peak_mass);

                    let exp_mz = query.peak_mz(peak_idx);
                    let calc_mz = frag.monoisotopic_mass / fragment_charge as f32 + PROTON;
                    match frag.kind {
                        Kind::A | Kind::B | Kind::C => {
                            score.matched_b += 1;
                            score.summed_b += peak_intensity;
                            b_run.matched(idx);
                        }
                        Kind::X | Kind::Y | Kind::Z => {
                            score.matched_y += 1;
                            score.summed_y += peak_intensity;
                            y_run.matched(idx);
                        }
                    }
                    if self.annotate_matches {
                        let idx = match frag.kind {
                            Kind::A | Kind::B | Kind::C => idx as i32 + 1,
                            Kind::X | Kind::Y | Kind::Z => {
                                peptide.sequence.len().saturating_sub(1) as i32 - idx as i32
                            }
                        };
                        fragments_details.kinds.push(frag.kind);
                        fragments_details.charges.push(fragment_charge as i32);
                        fragments_details.mz_experimental.push(exp_mz);
                        fragments_details.mz_calculated.push(calc_mz);
                        fragments_details.fragment_ordinals.push(idx);
                        fragments_details.intensities.push(peak_intensity);
                    }
                }
            }
        }
        score.hyperscore = score.hyperscore(self.score_type);
        score.longest_b = b_run.longest;
        score.longest_y = y_run.longest;
        let matched_intensity = score.summed_b + score.summed_y;
        if matched_intensity > 0.0 {
            score.ppm_difference /= matched_intensity;
        } else {
            score.ppm_difference = 0.0;
        }

        if self.annotate_matches {
            (score, Some(fragments_details))
        } else {
            (score, None)
        }
    }
}

#[derive(Default)]
struct Run {
    start: usize,
    length: usize,
    last: usize,
    pub longest: usize,
}
impl Run {
    pub fn matched(&mut self, index: usize) {
        if self.last == index {
            return;
        } else if self.start + self.length == index {
            self.length += 1;
            self.longest = self.longest.max(self.length);
        } else {
            self.start = index;
            self.length = 1;
            self.longest = self.longest.max(self.length);
        }
        self.last = index;
    }
}
