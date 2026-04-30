use crate::database::{IndexedDatabase, PeptideIx};
use crate::heap::bounded_min_heapify;
use crate::ion_series::{IonSeries, Kind};
use crate::mass::{Tolerance, NEUTRON, PROTON};
use crate::spectrum::{Peak, Precursor, ProcessedSpectrum};
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
fn dbg_expmass_match(query: &ProcessedSpectrum<Peak>) -> bool {
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

#[derive(Copy, Clone, Default, Debug, PartialEq, PartialOrd)]
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

/// The core identification data produced by the search engine.
/// This struct contains NO FDR information (neither TDC nor DF).
/// It is the raw material that enters the FDR pipeline.
#[derive(Serialize, Clone, Debug)]
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
    pub spectrum_p_value: f64,
    pub poisson_log10_p_value: f64,
    pub ms2_intensity: f32,
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
}

/// A feature augmented with Decoy-Free (DF) outputs.
/// Vanilla TDC FDR fields are not included in this representation.
#[derive(Serialize, Clone, Debug)]
pub struct DfFeature {
    #[serde(flatten)]
    pub core: FeatureCore,

    // --- DECOY-FREE: Core Columns ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_peptide_q: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_protein_q: Option<f32>,

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
    pub decoy_free_p_value_base: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_base: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_base: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_base: Option<f32>,

    // RT confidence adjustment snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_rt: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_rt: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_rt: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_rt: Option<f32>,

    // IMS confidence adjustment snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_ims: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_ims: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_ims: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_ims: Option<f32>,

    // Peptide reproducibility rescue snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_peptide_rescue: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_peptide_rescue: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_peptide_rescue: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_peptide_rescue: Option<f32>,

    // Protein reproducibility rescue snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_protein_rescue: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_protein_rescue: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_protein_rescue: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_protein_rescue: Option<f32>,

    // Transitional internal fields.
    // TODO: remove after apply_physical_rescue/apply_bounded_repro_shift are rewritten
    // to operate directly on the active decoy_free_* stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_p_value_l2: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_l2: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_l2: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_l2: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_pep_l3: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_score_l3: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoy_free_q_l3: Option<f32>,

    // 5D. Layer 2 diagnostics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_mode_used: Option<String>,
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
    pub p_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_mom: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_mom: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_mom: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_mom: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_mom: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_mom: Option<f32>,

    // MLE
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_mle: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_mle: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_mle: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_mle: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_mle: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_mle: Option<f32>,

    // Lower Order
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_lo: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_lo: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_lo: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_lo: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_lo: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_lo: Option<f32>,

    // MSFDR (seeded / legacy)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_msfdr: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_msfdr: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_msfdr: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_msfdr: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_msfdr: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_msfdr: Option<f32>,

    // MSFDR (1-state mixture)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_1smix: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_1smix: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_1smix: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_1smix: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_1smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_1smix: Option<f32>,

    // MSFDR (2-state mixture)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_2smix: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_2smix: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_2smix: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_2smix: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_2smix: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_2smix: Option<f32>,

    // Nokoi
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_nokoi: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_nokoi: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_nokoi: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_nokoi: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_nokoi: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_nokoi: Option<f32>,

    // Ensemble consensus stream
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pep_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_ensemble: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_p_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_q_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rt_adjust_pep_ensemble: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_p_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_q_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ims_adjust_pep_ensemble: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_p_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_q_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peptide_rescue_pep_ensemble: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_p_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_q_ensemble: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protein_rescue_pep_ensemble: Option<f32>,
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
        }
    }

    pub fn to_df(self) -> DfFeature {
        DfFeature {
            core: self,
            decoy_free_p_value: None,
            decoy_free_pep: None,
            decoy_free_score: None,
            decoy_free_q_value: None,
            decoy_free_peptide_q: None,
            decoy_free_protein_q: None,

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

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Method-of-moments Gumbel fit to the per-spectrum candidate hyperscore
/// distribution. The fit is intentionally local to one spectrum/query.
///
/// We fit the retained, fully scored candidate hyperscores rather than the
/// preliminary matched-peak counts. This gives LowerOrder a Sage-native
/// score-tail p-value tied to the same statistic used for ranking.
fn gumbel_moments_from_scores(scores: &[f64]) -> Option<(f64, f64)> {
    let xs: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();

    if xs.len() < 3 {
        return None;
    }

    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>()
        / n;

    if !var.is_finite() || var <= 0.0 {
        return None;
    }

    let beta = (6.0 * var).sqrt() / std::f64::consts::PI;
    if !beta.is_finite() || beta <= 0.0 {
        return None;
    }

    let mu = mean - EULER_GAMMA * beta;

    if mu.is_finite() {
        Some((mu, beta))
    } else {
        None
    }
}

#[inline]
fn gumbel_sf(score: f64, mu: f64, beta: f64) -> f64 {
    if !score.is_finite() || !mu.is_finite() || !beta.is_finite() || beta <= 0.0 {
        return 1.0;
    }

    let z = (score - mu) / beta;

    if z >= 36.0 {
        // For very high scores, SF ≈ exp(-z).
        (-z).exp().clamp(1e-325, 1.0)
    } else if z <= -36.0 {
        1.0
    } else {
        let t = (-z).exp();
        (-(-t).exp_m1()).clamp(1e-325, 1.0)
    }
}

#[inline]
fn empirical_sf_from_sorted_desc(sorted_scores: &[f64], score: f64) -> f64 {
    if sorted_scores.is_empty() || !score.is_finite() {
        return 1.0;
    }

    let ge = sorted_scores
        .iter()
        .filter(|x| x.is_finite() && **x >= score)
        .count();
    ((ge as f64 + 0.5) / (sorted_scores.len() as f64 + 1.0)).clamp(1e-325, 1.0)
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
fn first_precursor<'a>(query: &'a ProcessedSpectrum<Peak>) -> Option<&'a Precursor> {
    let precursor = query.precursors.first();
    if precursor.is_none() {
        eprintln!(
            "[sage] skipping spectrum without MS1 precursor metadata: {}",
            query.id
        );
    }
    precursor
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
        query: &ProcessedSpectrum<Peak>,
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

    pub fn score(&self, query: &ProcessedSpectrum<crate::spectrum::Peak>) -> Vec<FeatureCore> {
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
        query: &ProcessedSpectrum<crate::spectrum::Peak>,
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
        for peak in query.peaks.iter() {
            for charge in 1..max_fragment_charge {
                for frag in candidates.page_search(peak.mass, charge) {
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
        query: &ProcessedSpectrum<Peak>,
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

    fn initial_hits(&self, query: &ProcessedSpectrum<Peak>, precursor: &Precursor) -> InitialHits {
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
        } else if precursor.charge.is_some() && self.override_precursor_charge == false {
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

    pub fn score_standard(&self, query: &ProcessedSpectrum<Peak>) -> Vec<FeatureCore> {
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
        query: &ProcessedSpectrum<Peak>,
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

        let scored_len = score_vector.len().max(1) as f64;

        // Fix the old lambda fragility: estimate the matched-peak-count background
        // from the same fully scored, min_matched_peaks-filtered candidate population
        // that is used for ranking/output, not from preliminary hit bookkeeping.
        let filtered_matched_peaks: u32 = score_vector
            .iter()
            .map(|(score, _)| (score.matched_b + score.matched_y) as u32)
            .sum();

        let lambda = filtered_matched_peaks as f64 / scored_len;

        // Sage-native per-spectrum hyperscore tail model.
        // Exclude rank 1 when possible so a true target does not dominate the local
        // null fit. If too few lower candidates exist, use all retained candidates.
        // This p-value is tied to hyperscore, unlike the old matched-peak Poisson SF.
        let tail_fit_scores: Vec<f64> = if score_vector.len() >= 6 {
            score_vector
                .iter()
                .skip(1)
                .map(|(score, _)| score.hyperscore)
                .filter(|x| x.is_finite())
                .collect()
        } else {
            score_vector
                .iter()
                .map(|(score, _)| score.hyperscore)
                .filter(|x| x.is_finite())
                .collect()
        };

        let gumbel_tail = gumbel_moments_from_scores(&tail_fit_scores);

        let sorted_hyperscores: Vec<f64> = score_vector
            .iter()
            .map(|(score, _)| score.hyperscore)
            .filter(|x| x.is_finite())
            .collect();

        let mz = precursor.mz - PROTON;

        for idx in 0..report_psms.min(score_vector.len()) {
            let score = score_vector[idx].0;
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

            // Main spectrum_p_value is now the Sage-native hyperscore-tail p-value.
            // This is the value LowerOrder should convert to e-value/TEV.
            let spectrum_p_value = match gumbel_tail {
                Some((mu, beta)) => gumbel_sf(score.hyperscore, mu, beta),
                None => empirical_sf_from_sorted_desc(&sorted_hyperscores, score.hyperscore),
            }
            .max(1e-325);

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
                spectrum_p_value,
                missed_cleavages: peptide.missed_cleavages,
                predicted_rt: 0.0,
                predicted_ims: 0.0,
                aligned_rt: query.scan_start_time,
                delta_rt_model: 0.999,
                delta_ims_model: 0.999,
                ms2_intensity: matched_intensity,
                fragments,
            })
        }
    }

    fn remove_matched_peaks(&self, query: &mut ProcessedSpectrum<Peak>, psm: &FeatureCore) {
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
                if let Some(peak) = crate::spectrum::select_most_intense_peak(
                    &query.peaks,
                    frag.monoisotopic_mass / charge as f32,
                    self.fragment_tol,
                    None,
                ) {
                    to_remove.push(*peak);
                }
            }
        }
        query.peaks = query
            .peaks
            .drain(..)
            .filter(|peak| !to_remove.contains(peak))
            .collect();
        query.total_ion_current = query.peaks.iter().map(|peak| peak.intensity).sum::<f32>();
    }

    pub fn score_chimera_fast(&self, query: &ProcessedSpectrum<Peak>) -> Vec<FeatureCore> {
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
        query: &ProcessedSpectrum<Peak>,
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
                if let Some(peak) = crate::spectrum::select_most_intense_peak(
                    &query.peaks,
                    mz,
                    self.fragment_tol,
                    None,
                ) {
                    score.ppm_difference +=
                        peak.intensity * (mz - peak.mass).abs() * 2E6 / (mz + peak.mass);
                    let exp_mz = peak.mass + PROTON;
                    let calc_mz = mz + PROTON;
                    match frag.kind {
                        Kind::A | Kind::B | Kind::C => {
                            score.matched_b += 1;
                            score.summed_b += peak.intensity;
                            b_run.matched(idx);
                        }
                        Kind::X | Kind::Y | Kind::Z => {
                            score.matched_y += 1;
                            score.summed_y += peak.intensity;
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
                        fragments_details.charges.push(charge as i32);
                        fragments_details.mz_experimental.push(exp_mz);
                        fragments_details.mz_calculated.push(calc_mz);
                        fragments_details.fragment_ordinals.push(idx);
                        fragments_details.intensities.push(peak.intensity);
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
