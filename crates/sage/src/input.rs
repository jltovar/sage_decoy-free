use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FdrMode {
    #[default]
    Tdc,
    DecoyFree,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelFit {
    #[default]
    Moments,
    Mle,
    LowerOrder,

    // MSFDR family
    Msfdr,
    Msfdr1Smix,
    Msfdr2Smix,

    Nokoi,
    Ensemble,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProteinPCombine {
    Fisher,
    #[default]
    Cauchy,
    SidakMinP,
    Best,
    SecondBest,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PeptidePCombine {
    Fisher,
    #[default]
    Cauchy,
    SidakMinP,
    Best,
    SecondBest,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnsemblePCombiner {
    Fisher,
    #[default]
    Cauchy,
    SidakMinP,
    Best,
    SecondBest,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinalEvidenceSpace {
    /// Explicitly choose the active Decoy-Free evidence stream.
    /// Auto uses model-specific defaults but must never infer evidence space
    /// from missing output fields.
    #[default]
    Auto,
    /// Force final selected Decoy-Free evidence to be p-value-native.
    PValue,
    /// Force final selected Decoy-Free evidence to be PEP-native.
    Pep,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QMethod {
    /// Use the level-appropriate default.
    #[default]
    Auto,
    /// Benjamini-Hochberg over p-values.
    Bh,
    /// Storey q-values over p-values.
    Storey,
    /// Cumulative mean over PEP-like values.
    /// Only valid for PEP-native evidence.
    Cummean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntrapmentReportMode {
    Off,
    Auto,
    On,
}

impl Default for EntrapmentReportMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MsfdrSeedMode {
    /// Seed MSFDR null directly from the rank-null pool window
    /// [msfdr_min_null_rank..=msfdr_max_null_rank].
    Pool,
}

impl Default for MsfdrSeedMode {
    fn default() -> Self {
        MsfdrSeedMode::Pool
    }
}

// ---------------------------------------------------------------------------
// Decoy-free tuning knobs (configuration surface)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoStratify {
    Global,
    #[default]
    Charge,
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoTevTransform {
    /// Canonical LowerOrder transformed E-value:
    ///   TEV = -ln(E)
    ///
    /// This is the default publication-facing LO scale.
    #[default]
    NegLogE,

    /// Reference-shifted transformed E-value:
    ///   TEV = ln(1000 / E)
    ///
    /// This preserves the historical 1000/E reference without the 0.02 compression.
    Log1000OverE,

    /// Historical compressed Tide/Comet-style scale:
    ///   TEV = 0.02 * ln(1000 / E)
    ///
    /// Retained only for backward-compatible comparisons.
    ScaledLog1000OverE,
}

/// How Nokoi defines the "positive" class in DF mode.
///
/// - or:      top-slice OR provisional p-threshold  (current behavior)
/// - and:     top-slice AND provisional p-threshold
/// - top_only: only top-slice
/// - p_only:   only provisional p-threshold
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NokoiPosRule {
    #[default]
    Or,
    And,
    TopOnly,
    POnly,
}

/// How to combine PEPs in ensemble mode.
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnsemblePepCombiner {
    #[default]
    Median,
    TrimmedMean,
    Max,
    Mean,
    WeightedMean,
    WeightedMedian,
    WinsorizedMean,
    Quantile,
    TopKMean,
    GeometricMean,
    LogitMean,
}

/// How to aggregate pi0(lambda) values over a lambda grid.
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StoreyPi0Agg {
    /// Robust default.
    #[default]
    Median,
    /// Trimmed mean over pi0(lambda) (after trimming tails).
    TrimmedMean,
}

/// What to do when Storey q-values collapse to a shelf (degenerate).
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StoreyDegeneracyFallback {
    #[default]
    Bh,
    None,
}

// =========================================================================
// Layer 2 / Layer 3 configuration
// =========================================================================

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalRescueMode {
    #[default]
    Off,
    DartBayes,
    BoundedAux,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DartNullRtModel {
    Normal,
    #[default]
    Uniform,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DartTrueRtModel {
    Normal,
    #[default]
    Laplace,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BoundedAuxUpdateSpace {
    #[default]
    LogitConfidence,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalAnchorMode {
    Strict,
    #[default]
    Default,
    Relaxed,
    EvidenceOnly,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JointMode {
    Min,
    Product,
    #[default]
    Independent,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReproducibilityAnchorMode {
    Best,
    #[default]
    SecondBest,
    Mean,
    Median,
    TrimmedMean,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RescueMode {
    Replace,
    #[default]
    BoundedShrinkage,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DartBootstrapMethod {
    None,
    Parametric,
    #[default]
    ParametricMixture,
    NonParametric,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DartMuEstimation {
    Mean,
    #[default]
    Median,
    WeightedMean,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct DartBayesConfig {
    pub dart_use_bootstrap: bool,
    pub dart_bootstrap_method: DartBootstrapMethod,
    pub dart_mu_estimation: DartMuEstimation,
    pub dart_bootstrap_iters: usize,
    pub dart_leave_one_run_out: bool,
    pub dart_null_rt_model: DartNullRtModel,
    pub dart_true_rt_model: DartTrueRtModel,
    pub dart_recalc_q_from_posterior: bool,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct BoundedAuxConfig {
    /// Update space for bounded auxiliary rescue.
    /// Currently only logit-confidence space is supported.
    pub update_space: BoundedAuxUpdateSpace,
    pub max_rescue_shift: f64,
    pub max_penalty_shift: f64,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct PhysicalRescueConfig {
    pub rt_mode: PhysicalRescueMode,
    pub ims_mode: PhysicalRescueMode,
    pub anchor_mode: PhysicalAnchorMode,
    pub anchor_max_pep: f64,
    pub anchor_max_q: f64,
    pub min_anchor_count_per_run: usize,
    pub min_anchor_count_per_charge: usize,
    pub joint_mode: JointMode,
    pub reliability_floor: f64,
    pub missing_penalty: f64,
    pub rt_region_bins: usize,
    pub use_local_rt_scale: bool,
    pub cov_shrinkage: f64,
    pub dart_cfg: Option<DartBayesConfig>,
    pub bounded_cfg: Option<BoundedAuxConfig>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ProteinEligibilityConfig {
    pub enabled: bool,
    pub q_threshold_physical: f64,
    pub min_unique_passing_peptides: usize,
    pub min_unique_passing_fraction: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct PeptideEligibilityConfig {
    pub min_run_fraction: f64,
    pub min_run_count: usize,
    pub strong_reference_q_threshold_physical: f64,
    pub strong_reference_pep_threshold_physical: Option<f64>,
    pub min_strong_run_fraction: f64,
    pub min_strong_run_count: usize,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ReproducibilityAnchorConfig {
    pub mode: ReproducibilityAnchorMode,
    pub trim_fraction: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct RescueBandConfig {
    #[serde(alias = "strong_cutoff_pep_l2")]
    pub strong_cutoff_pep: f64,
    #[serde(alias = "weak_cutoff_pep_l2")]
    pub weak_cutoff_pep: f64,
    pub max_rescue_fraction: f64,
    pub rescue_mode: RescueMode,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ReproducibilityConfig {
    // Global Reproducibility controls
    pub enabled: bool,
    pub max_total_shift: f64,
    pub max_agreement_shift: f64,
    pub max_recurrence_shift: f64,
    pub use_expert_agreement: bool,
    pub use_cross_run_recurrence: bool,
    pub redundancy_discount: f64,

    // Eligibility controls
    pub protein_eligibility: ProteinEligibilityConfig,
    pub peptide_eligibility: PeptideEligibilityConfig,

    // Anchor / rescue controls
    pub anchor: ReproducibilityAnchorConfig,
    pub rescue_band: RescueBandConfig,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct FdrOptions {
    // =========================================================================
    // A) Global knobs
    // =========================================================================
    pub mode: Option<FdrMode>,
    pub entrapment_report: Option<EntrapmentReportMode>,

    // Model selection
    pub model_fit: Option<ModelFit>,

    // Final active evidence-space controls.
    pub final_evidence_space: Option<FinalEvidenceSpace>,

    pub protein_p_combine: Option<ProteinPCombine>,
    pub peptide_p_combine: Option<PeptidePCombine>,

    // Explicit q-value method controls.
    pub psm_q_method: Option<QMethod>,
    pub peptide_q_method: Option<QMethod>,
    pub protein_q_method: Option<QMethod>,

    pub peptide_fdr: Option<f32>,
    pub protein_fdr: Option<f32>,
    pub precursor_fdr: Option<f32>,

    /// Reporting-only option.
    /// If true, accepted peptide-level discoveries may report all rank-1 PSMs
    /// supporting accepted target peptides. This does not change the native
    /// p-value stream or peptide/protein inference; it only changes the PSM-level
    /// reporting q-value used by the runner's reported PSM count.
    pub report_psms_by_peptide_q: Option<bool>,

    // Global null window (superset pool builder)
    pub min_null_rank: Option<u32>,
    pub max_null_rank: Option<u32>,

    // Rank-null pool construction controls
    pub min_null_size: Option<usize>,  // default 300
    pub min_rank_count: Option<usize>, // default 10

    // Explicit post-base Decoy-Free stage gates.
    // These control whether stage-specific TSV snapshots are produced and whether
    // the stage is allowed to replace the active decoy_free_* stream.
    pub enable_rt_confidence_adjustment: Option<bool>,
    pub enable_ims_confidence_adjustment: Option<bool>,
    pub enable_peptide_reproducibility_rescue: Option<bool>,
    pub enable_protein_reproducibility_rescue: Option<bool>,

    // Configurable Safety Brakes (global)
    pub min_storey_n: Option<usize>,

    // Storey/π0 tuning knobs (decoy-free and general)
    pub storey_pi0_clamp_min: Option<f64>,
    pub storey_pi0_clamp_max: Option<f64>,
    pub storey_lambda_min: Option<f64>,
    pub storey_lambda_max: Option<f64>,
    pub storey_lambda_step: Option<f64>,
    pub storey_lambda_min_for_agg: Option<f64>,
    pub storey_pi0_agg: Option<StoreyPi0Agg>,

    // Storey degeneracy detector knobs
    pub storey_degen_same_as_median_frac: Option<f64>,
    pub storey_degen_eps: Option<f64>,
    pub storey_degen_pi0_eps: Option<f64>,
    pub storey_degen_fallback: Option<StoreyDegeneracyFallback>,

    // =========================================================================
    // B) Moments specific knobs
    // =========================================================================
    pub moments_min_null_rank: Option<u32>,
    pub moments_max_null_rank: Option<u32>,
    pub moments_purification_factor: Option<f64>,

    // =========================================================================
    // C) MLE specific knobs
    // =========================================================================
    pub mle_min_null_rank: Option<u32>,
    pub mle_max_null_rank: Option<u32>,
    pub mle_purification_factor: Option<f64>,

    // =========================================================================
    // D) LowerOrder specific knobs
    // =========================================================================
    pub lower_order_min_null_rank: Option<u32>,
    pub lower_order_max_null_rank: Option<u32>,
    pub lower_order_purification_factor: Option<f64>,

    // LowerOrder support threshold.
    // This is not a rank-window selector. The selected LO ranks are controlled
    // only by lower_order_min_null_rank..=lower_order_max_null_rank.
    pub lo_min_count_per_rank: Option<usize>,

    // LowerOrder controls.
    //
    // LO uses only non-rank-1 lower-order null evidence. Rank 1 is excluded because
    // it is the target-contaminated top-hit mixture.
    //
    // This implementation requires at least two usable lower-order ranks. Each
    // supported rank contributes a k-specific LOM MLE; those LOMs are also used to
    // report the diagnostic β(μ) trend. The production rank-1 TNM is fit by one
    // deterministic joint likelihood over the supported lower-order rank buckets.
    //
    // Invalid windows such as 1..1 or 2..2 are allowed through config parsing but
    // fail closed during LO fitting with explicit logs.
    //
    // The TEV transform is configurable through lo_tev_transform.
    pub lo_stratify: Option<LoStratify>,

    // LowerOrder TEV construction.
    //
    // Upstream scoring stores spectrum-local tail evidence:
    //
    //   core.lo_spectrum_tail_p
    //   core.lo_spectrum_candidate_count
    //
    // Decoy-Free constructs the LO E-value once:
    //
    //   E_LO = lo_spectrum_tail_p
    //        * lo_spectrum_candidate_count.powf(lo_evalue_candidate_count_power)
    //        * lo_evalue_scale
    //
    // Then lo_tev_transform selects the TEV score scale:
    //
    //   neg_log_e              => TEV = -ln(E_LO)
    //   log_1000_over_e        => TEV = ln(1000 / E_LO)
    //   scaled_log_1000_over_e => TEV = 0.02 * ln(1000 / E_LO)
    //
    pub lo_evalue_candidate_count_power: Option<f64>,
    pub lo_evalue_scale: Option<f64>,
    pub lo_tev_transform: Option<LoTevTransform>,

    // LowerOrder rank-1 TNM extrapolation strength.
    //
    // The production LO path fits per-rank lower-order LOMs, then infers the
    // rank-1 TNM by local extrapolation from the nearest supported lower-order
    // ranks. This scalar controls how far the rank-1 TNM is extrapolated beyond
    // the nearest lower-order LOMs.
    //
    // strength = 1.0 means one local rank step beyond rank 2.
    // Larger values produce a more aggressive rank-1 null and smaller rank-1
    // p-values. Entrapment is not used to fit this value; it is an external
    // validation readout only.
    pub lo_tnm_extrapolation_strength: Option<f64>,

    // LowerOrder TNM estimator.
    //
    // There is no TNM mode knob. The production LO path infers the rank-1 TNM by
    // deterministic local extrapolation from supported lower-order LOMs. Rank-1
    // scores are never used to fit or select the null.

    // =========================================================================
    // E) MSFDR specific knobs
    // =========================================================================
    pub msfdr_min_null_rank: Option<u32>,
    pub msfdr_max_null_rank: Option<u32>,
    pub msfdr_seeded_purification_factor: Option<f64>,

    // MSFDR init/drift knobs (needed by real models)
    pub msfdr_seeded_top_frac_init: Option<f64>, // default 0.2
    pub msfdr_multistart: Option<usize>,

    // --- Specific clamps (overrides) ---
    pub msfdr_pi_clamp_min: Option<f64>,
    pub msfdr_pi_clamp_max: Option<f64>,

    // =========================================================================
    // Mixture knobs (MSFDR 1smix / pooled-rank 2smix)
    // =========================================================================
    pub mix_em_max_iter: Option<usize>, // default 200; clamp 1..10_000
    pub mix_em_tol: Option<f64>,        // default 1e-6; must be >0

    // =========================================================================
    // F) MSFDR1_Smix specific knobs
    // =========================================================================

    // MSFDR1 initialization knobs.
    pub msfdr1_bottom_frac_init: Option<f64>, // default 0.50
    pub msfdr1_top_frac_init: Option<f64>,    // default 0.20

    // MSFDR1 mixture-weight clamps.
    pub msfdr1_pi_clamp_min: Option<f64>,
    pub msfdr1_pi_clamp_max: Option<f64>,

    // =========================================================================
    // G) MSFDR2_Smix specific knobs
    // =========================================================================
    pub msfdr2_smix_min_null_rank: Option<u32>,
    pub msfdr2_smix_max_null_rank: Option<u32>,

    // MSFDR2 initialization knob.
    pub msfdr2_bottom_frac_init: Option<f64>,
    pub msfdr2_top_frac_init: Option<f64>,

    // --- Specific clamps (overrides) ---
    pub msfdr2_pi_clamp_min: Option<f64>,
    pub msfdr2_pi_clamp_max: Option<f64>,

    // =========================================================================
    // H) Nokoi specific knobs
    // =========================================================================
    pub nokoi_min_null_rank: Option<u32>,
    pub nokoi_max_null_rank: Option<u32>,

    /// Nokoi-specific controls.
    ///
    /// `nokoi_null_purification_factor` controls lower-rank null-pool
    /// construction.
    ///
    /// `nokoi_positive_top_fraction` controls the high-scoring rank-1 fraction
    /// used to define Nokoi's positive training class.
    pub nokoi_null_purification_factor: Option<f64>, // default 0.20; clamp 0..0.9
    pub nokoi_positive_top_fraction: Option<f64>, // default 0.10; clamp 0..0.9

    // Nokoi DF cross-fit calibration
    pub nokoi_k_folds: Option<usize>,

    // Nokoi L1 lambda grid (JSON-exposed)
    pub nokoi_l1_lambda_min: Option<f64>,
    pub nokoi_l1_lambda_max: Option<f64>,
    pub nokoi_l1_lambda_steps: Option<usize>,

    // =========================================================================
    // I) Ensemble specific knobs
    // =========================================================================
    // Ensemble expert gates (Ensemble uses these; explicit model_fit variants override gates)
    pub enable_moments: Option<bool>,      // default true
    pub enable_mle: Option<bool>,          // default true
    pub enable_lower_order: Option<bool>,  // default true
    pub enable_msfdr_seeded: Option<bool>, // default true
    pub enable_msfdr_1smix: Option<bool>,  // default true
    pub enable_msfdr_2smix: Option<bool>,  // default true
    pub enable_nokoi: Option<bool>,        // default true

    // Ensemble combination choices (global controls; used by ModelFit::Ensemble)
    pub ensemble_p_combiner: Option<EnsemblePCombiner>,
    pub ensemble_pep_combiner: Option<EnsemblePepCombiner>,

    // Shared robust-combiner knobs
    pub ensemble_pep_trim_frac: Option<f64>, // default 0.20, clamp [0.0, 0.49]
    pub ensemble_pep_quantile: Option<f64>,  // default 0.50, clamp [0.0, 1.0]
    pub ensemble_pep_top_k: Option<usize>,   // default 2, clamp >= 1
    pub ensemble_pep_logit_eps: Option<f64>, // default 1e-6, clamp [1e-12, 1e-2]

    // Static per-expert weights (used by weighted_mean / weighted_median)
    pub ensemble_weight_moments: Option<f64>, // default 1.0
    pub ensemble_weight_mle: Option<f64>,     // default 1.0
    pub ensemble_weight_lower_order: Option<f64>, // default 1.0
    pub ensemble_weight_msfdr_seeded: Option<f64>, // default 1.0
    pub ensemble_weight_msfdr_1smix: Option<f64>, // default 1.0
    pub ensemble_weight_msfdr_2smix: Option<f64>, // default 1.0
    pub ensemble_weight_nokoi: Option<f64>,   // default 1.0

    // =========================================================================
    // J) Layer 2: Physical confidence adjustment: RT and IMS Knobs
    //    Layer 3: Reproducibility rescue: peptide and protein
    // =========================================================================
    pub physical_rescue: Option<PhysicalRescueConfig>,
    pub reproducibility: Option<ReproducibilityConfig>,
}

#[derive(Clone, Serialize, Debug)]
pub struct FdrSettings {
    // =========================================================================
    // A) Global knobs
    // =========================================================================
    pub mode: FdrMode,
    pub entrapment_report: EntrapmentReportMode,

    // Model selection
    pub model_fit: ModelFit,

    // Final active evidence-space controls.
    pub final_evidence_space: FinalEvidenceSpace,

    // Protein/peptide p-value combiners
    pub protein_p_combine: ProteinPCombine,
    pub peptide_p_combine: PeptidePCombine,

    // Explicit q-value method controls
    pub psm_q_method: QMethod,
    pub peptide_q_method: QMethod,
    pub protein_q_method: QMethod,

    pub peptide_fdr: f32,
    pub protein_fdr: f32,
    pub precursor_fdr: f32,

    /// Reporting-only option.
    /// When true, peptide-accepted rank-1 target PSMs receive a reporting q-value
    /// no worse than their peptide q-value. The original model p-value and PEP
    /// streams remain unchanged.
    pub report_psms_by_peptide_q: bool,

    // Global null window (superset pool builder)
    pub min_null_rank: u32,
    pub max_null_rank: u32,

    // Rank-null pool construction controls
    pub min_null_size: usize,
    pub min_rank_count: usize,

    // Explicit post-base Decoy-Free stage gates.
    pub enable_rt_confidence_adjustment: bool,
    pub enable_ims_confidence_adjustment: bool,
    pub enable_peptide_reproducibility_rescue: bool,
    pub enable_protein_reproducibility_rescue: bool,

    // Configurable Safety Brakes
    pub min_storey_n: usize,

    // Storey/π0 tuning knobs
    pub storey_pi0_clamp_min: f64,
    pub storey_pi0_clamp_max: f64,
    pub storey_lambda_min: f64,
    pub storey_lambda_max: f64,
    pub storey_lambda_step: f64,
    pub storey_lambda_min_for_agg: f64,
    pub storey_pi0_agg: StoreyPi0Agg,

    // Storey degeneracy detector knobs
    pub storey_degen_same_as_median_frac: f64,
    pub storey_degen_eps: f64,
    pub storey_degen_pi0_eps: f64,
    pub storey_degen_fallback: StoreyDegeneracyFallback,

    // =========================================================================
    // B) Moments specific resolved null window
    // =========================================================================
    pub moments_min_null_rank: u32,
    pub moments_max_null_rank: u32,
    pub moments_purification_factor: f64,

    // =========================================================================
    // C) MLE specific resolved null window
    // =========================================================================
    pub mle_min_null_rank: u32,
    pub mle_max_null_rank: u32,
    pub mle_purification_factor: f64,

    // =========================================================================
    // D) LowerOrder specific resolved null window + knobs
    // =========================================================================
    pub lower_order_min_null_rank: u32,
    pub lower_order_max_null_rank: u32,
    pub lower_order_purification_factor: f64,

    pub lo_min_count_per_rank: usize,

    pub lo_stratify: LoStratify,

    pub lo_evalue_candidate_count_power: f64,
    pub lo_evalue_scale: f64,
    pub lo_tev_transform: LoTevTransform,
    pub lo_tnm_extrapolation_strength: f64,

    // =========================================================================
    // E) MSFDR specific resolved null window + knobs
    // =========================================================================
    pub msfdr_min_null_rank: u32,
    pub msfdr_max_null_rank: u32,
    pub msfdr_seeded_purification_factor: f64,

    pub msfdr_seeded_top_frac_init: f64,
    pub msfdr_multistart: usize,

    pub msfdr_pi_clamp_min: f64,
    pub msfdr_pi_clamp_max: f64,

    // =========================================================================
    // Mixture knobs (MSFDR 1smix / pooled-rank 2smix)
    // =========================================================================
    pub mix_em_max_iter: usize,
    pub mix_em_tol: f64,

    // =========================================================================
    // F) MSFDR1_Smix specific resolved null window + knobs
    // =========================================================================
    pub msfdr1_bottom_frac_init: f64,
    pub msfdr1_top_frac_init: f64,

    pub msfdr1_pi_clamp_min: f64,
    pub msfdr1_pi_clamp_max: f64,

    // =========================================================================
    // G) MSFDR2_Smix specific resolved null window + knobs
    // =========================================================================
    pub msfdr2_smix_min_null_rank: u32,
    pub msfdr2_smix_max_null_rank: u32,

    pub msfdr2_bottom_frac_init: f64,
    pub msfdr2_top_frac_init: f64,

    pub msfdr2_pi_clamp_min: f64,
    pub msfdr2_pi_clamp_max: f64,

    // =========================================================================
    // H) Nokoi specific resolved null window + knobs
    // =========================================================================
    pub nokoi_min_null_rank: u32,
    pub nokoi_max_null_rank: u32,

    /// Resolved Nokoi-specific controls.
    pub nokoi_null_purification_factor: f64,
    pub nokoi_positive_top_fraction: f64,

    pub nokoi_k_folds: usize,

    pub nokoi_l1_lambda_min: f64,
    pub nokoi_l1_lambda_max: f64,
    pub nokoi_l1_lambda_steps: usize,

    // =========================================================================
    // I) Ensemble specific knobs
    // =========================================================================
    pub enable_moments: bool,
    pub enable_mle: bool,
    pub enable_lower_order: bool,
    pub enable_msfdr_seeded: bool,
    pub enable_msfdr_1smix: bool,
    pub enable_msfdr_2smix: bool,
    pub enable_nokoi: bool,

    pub ensemble_p_combiner: EnsemblePCombiner,
    pub ensemble_pep_combiner: EnsemblePepCombiner,

    pub ensemble_pep_trim_frac: f64,
    pub ensemble_pep_quantile: f64,
    pub ensemble_pep_top_k: usize,
    pub ensemble_pep_logit_eps: f64,

    pub ensemble_weight_moments: f64,
    pub ensemble_weight_mle: f64,
    pub ensemble_weight_lower_order: f64,
    pub ensemble_weight_msfdr_seeded: f64,
    pub ensemble_weight_msfdr_1smix: f64,
    pub ensemble_weight_msfdr_2smix: f64,
    pub ensemble_weight_nokoi: f64,

    // =========================================================================
    // J) Layer 2: Physical confidence adjustment: RT and IMS Knobs
    //    Layer 3: Reproducibility rescue: peptide and protein
    // =========================================================================
    pub physical_rescue: PhysicalRescueConfig,
    pub reproducibility: ReproducibilityConfig,
}

impl From<FdrOptions> for FdrSettings {
    fn from(options: FdrOptions) -> Self {
        // ---------------------------------------------------------------------
        // Local helpers
        // ---------------------------------------------------------------------
        let clamp_weight = |w: Option<f64>| -> f64 {
            match w {
                Some(x) if x.is_finite() && x >= 0.0 => x,
                _ => 1.0,
            }
        };

        let clamp_frac = |x: f64, default: f64| -> f64 {
            if x.is_finite() {
                x.clamp(0.01, 0.99)
            } else {
                default
            }
        };

        // ---------------------------------------------------------------------
        // A) Global knobs
        // ---------------------------------------------------------------------
        let mode = options.mode.unwrap_or(FdrMode::DecoyFree);

        let physical_rescue = options
            .physical_rescue
            .unwrap_or_else(|| PhysicalRescueConfig {
                rt_mode: PhysicalRescueMode::Off,
                ims_mode: PhysicalRescueMode::Off,
                anchor_mode: PhysicalAnchorMode::Default,
                anchor_max_pep: 0.1,
                anchor_max_q: 0.01,
                min_anchor_count_per_run: 10,
                min_anchor_count_per_charge: 5,
                joint_mode: JointMode::Independent,
                reliability_floor: 0.5,
                missing_penalty: 0.0,
                rt_region_bins: 10,
                use_local_rt_scale: true,
                cov_shrinkage: 0.1,
                dart_cfg: None,
                bounded_cfg: None,
            });

        let reproducibility = options
            .reproducibility
            .unwrap_or_else(|| ReproducibilityConfig {
                enabled: false,
                max_total_shift: 0.0,
                max_agreement_shift: 0.0,
                max_recurrence_shift: 0.0,
                use_expert_agreement: false,
                use_cross_run_recurrence: false,
                redundancy_discount: 1.0,

                protein_eligibility: ProteinEligibilityConfig {
                    enabled: true,
                    q_threshold_physical: 0.01,
                    min_unique_passing_peptides: 2,
                    min_unique_passing_fraction: None,
                },

                peptide_eligibility: PeptideEligibilityConfig {
                    min_run_fraction: 0.6,
                    min_run_count: 2,
                    strong_reference_q_threshold_physical: 0.01,
                    strong_reference_pep_threshold_physical: None,
                    min_strong_run_fraction: 0.2,
                    min_strong_run_count: 1,
                },

                anchor: ReproducibilityAnchorConfig {
                    mode: ReproducibilityAnchorMode::SecondBest,
                    trim_fraction: Some(0.1),
                },

                rescue_band: RescueBandConfig {
                    strong_cutoff_pep: 0.01,
                    weak_cutoff_pep: 0.25,
                    max_rescue_fraction: 0.5,
                    rescue_mode: RescueMode::BoundedShrinkage,
                },
            });

        // New explicit post-base stage gates.
        //
        // Backward-compatible behavior:
        // - RT defaults to disabled unless enable_rt_confidence_adjustment is true.
        // - IMS defaults to disabled unless enable_ims_confidence_adjustment is true.
        // - Peptide/protein reproducibility default to reproducibility.enabled.
        //
        // This lets old JSONs continue to work, while allowing the new independent
        // switches to be used directly.
        let enable_rt_confidence_adjustment =
            options.enable_rt_confidence_adjustment.unwrap_or(false);

        let enable_ims_confidence_adjustment =
            options.enable_ims_confidence_adjustment.unwrap_or(false);

        let enable_peptide_reproducibility_rescue = options
            .enable_peptide_reproducibility_rescue
            .unwrap_or(reproducibility.enabled);

        let enable_protein_reproducibility_rescue = options
            .enable_protein_reproducibility_rescue
            .unwrap_or(reproducibility.enabled);

        let precursor_fdr = options.precursor_fdr.unwrap_or(0.01);
        let peptide_fdr = options.peptide_fdr.unwrap_or(0.01);
        let protein_fdr = options.protein_fdr.unwrap_or(0.01);
        let report_psms_by_peptide_q = options.report_psms_by_peptide_q.unwrap_or(false);
        let entrapment_report = options
            .entrapment_report
            .unwrap_or(EntrapmentReportMode::Auto);

        let model_fit = options.model_fit.unwrap_or(ModelFit::Ensemble);
        let protein_p_combine = options.protein_p_combine.unwrap_or(ProteinPCombine::Cauchy);
        let peptide_p_combine = options.peptide_p_combine.unwrap_or(PeptidePCombine::Cauchy);
        let final_evidence_space = options
            .final_evidence_space
            .unwrap_or(FinalEvidenceSpace::Auto);

        let psm_q_method = options.psm_q_method.unwrap_or(QMethod::Storey);
        let peptide_q_method = options.peptide_q_method.unwrap_or(QMethod::Auto);
        let protein_q_method = options.protein_q_method.unwrap_or(QMethod::Auto);

        let min_storey_n = options.min_storey_n.unwrap_or(300);
        let min_null_size = options.min_null_size.unwrap_or(300);

        let moments_purification_factor = options
            .moments_purification_factor
            .unwrap_or(0.25)
            .clamp(0.0, 0.9);

        let mle_purification_factor = options
            .mle_purification_factor
            .unwrap_or(0.25)
            .clamp(0.0, 0.9);

        let lower_order_purification_factor = options
            .lower_order_purification_factor
            .unwrap_or(0.15)
            .clamp(0.0, 0.9);

        let msfdr_seeded_purification_factor = options
            .msfdr_seeded_purification_factor
            .unwrap_or(0.25)
            .clamp(0.0, 0.9);

        let nokoi_null_purification_factor = options
            .nokoi_null_purification_factor
            .unwrap_or(0.20)
            .clamp(0.0, 0.9);

        let nokoi_positive_top_fraction = options
            .nokoi_positive_top_fraction
            .unwrap_or(0.10)
            .clamp(0.0, 0.9);

        let min_rank_count = options.min_rank_count.unwrap_or(10);

        // ---------------------------------------------------------------------
        // A.1) Storey / pi0 tuning knobs
        // ---------------------------------------------------------------------
        let storey_pi0_clamp_min = options.storey_pi0_clamp_min.unwrap_or(0.50).clamp(0.0, 1.0);

        let storey_pi0_clamp_max = options
            .storey_pi0_clamp_max
            .unwrap_or(1.00)
            .clamp(0.0, 1.0)
            .max(storey_pi0_clamp_min);

        let storey_lambda_min = options.storey_lambda_min.unwrap_or(0.05).clamp(0.0, 0.99);

        let storey_lambda_max = options
            .storey_lambda_max
            .unwrap_or(0.95)
            .clamp(0.01, 1.0)
            .max(storey_lambda_min);

        let storey_lambda_step = options.storey_lambda_step.unwrap_or(0.05).max(1e-6);

        let storey_lambda_min_for_agg = options
            .storey_lambda_min_for_agg
            .unwrap_or(0.50)
            .clamp(storey_lambda_min, storey_lambda_max);

        let storey_pi0_agg = options.storey_pi0_agg.unwrap_or(StoreyPi0Agg::Median);

        // ---------------------------------------------------------------------
        // A.2) Storey degeneracy knobs
        // ---------------------------------------------------------------------
        let storey_degen_same_as_median_frac = options
            .storey_degen_same_as_median_frac
            .unwrap_or(0.90)
            .clamp(0.0, 1.0);

        let storey_degen_eps = options.storey_degen_eps.unwrap_or(1e-6).max(0.0);

        let storey_degen_pi0_eps = options.storey_degen_pi0_eps.unwrap_or(1e-3).max(0.0);

        let storey_degen_fallback = options
            .storey_degen_fallback
            .unwrap_or(StoreyDegeneracyFallback::Bh);

        // ---------------------------------------------------------------------
        // A.3) Global null window (superset pool builder)
        // ---------------------------------------------------------------------
        let (min_null_rank, max_null_rank) = {
            let a = options.min_null_rank.unwrap_or(2);
            let b = options.max_null_rank.unwrap_or(50);
            if a <= b {
                (a, b)
            } else {
                (b, a)
            }
        };

        // Local helper to resolve a method window within the global window.
        let resolve_window = |opt_min: Option<u32>,
                              opt_max: Option<u32>,
                              default_min: u32,
                              default_max: u32|
         -> (u32, u32) {
            let mut mn = opt_min.unwrap_or(default_min);
            let mut mx = opt_max.unwrap_or(default_max);

            if mn > mx {
                std::mem::swap(&mut mn, &mut mx);
            }

            mn = mn.clamp(min_null_rank, max_null_rank);
            mx = mx.clamp(min_null_rank, max_null_rank);

            if mn > mx {
                mn = mn.clamp(min_null_rank, max_null_rank);
                mx = mn;
            }

            (mn, mx)
        };

        // ---------------------------------------------------------------------
        // B) Moments specific resolved null window
        // ---------------------------------------------------------------------
        let enable_moments = options.enable_moments.unwrap_or(true);

        let (moments_min_null_rank, moments_max_null_rank) = resolve_window(
            options.moments_min_null_rank,
            options.moments_max_null_rank,
            4,
            50,
        );

        // ---------------------------------------------------------------------
        // C) MLE specific resolved null window
        // ---------------------------------------------------------------------
        let enable_mle = options.enable_mle.unwrap_or(true);

        let (mle_min_null_rank, mle_max_null_rank) =
            resolve_window(options.mle_min_null_rank, options.mle_max_null_rank, 4, 50);

        // ---------------------------------------------------------------------
        // D) LowerOrder specific resolved null window + LO knobs
        // ---------------------------------------------------------------------
        let enable_lower_order = options.enable_lower_order.unwrap_or(true);

        let (lower_order_min_null_rank, lower_order_max_null_rank) = resolve_window(
            options.lower_order_min_null_rank,
            options.lower_order_max_null_rank,
            6,
            12,
        );

        let lo_min_count_per_rank = options.lo_min_count_per_rank.unwrap_or(10).max(1);
        let lo_stratify = options.lo_stratify.unwrap_or(LoStratify::Charge);

        let lo_evalue_candidate_count_power = options
            .lo_evalue_candidate_count_power
            .unwrap_or(0.75)
            .clamp(0.0, 1.0);

        let lo_evalue_scale = options.lo_evalue_scale.unwrap_or(1.0).clamp(1e-6, 1e6);

        let lo_tev_transform = options.lo_tev_transform.unwrap_or(LoTevTransform::NegLogE);

        let lo_tnm_extrapolation_strength = options
            .lo_tnm_extrapolation_strength
            .unwrap_or(1.0)
            .clamp(0.25, 5.0);

        // ---------------------------------------------------------------------
        // E) MSFDR specific resolved null window + knobs
        // ---------------------------------------------------------------------
        let enable_msfdr_seeded = options.enable_msfdr_seeded.unwrap_or(true);

        let (msfdr_min_null_rank, msfdr_max_null_rank) = resolve_window(
            options.msfdr_min_null_rank,
            options.msfdr_max_null_rank,
            4,
            50,
        );

        let msfdr_multistart = options.msfdr_multistart.unwrap_or(3).clamp(1, 25);

        let msfdr_seeded_top_frac_init =
            clamp_frac(options.msfdr_seeded_top_frac_init.unwrap_or(0.20), 0.20);

        // ---------------------------------------------------------------------
        // Mixture knobs (shared)
        // ---------------------------------------------------------------------
        let mix_em_max_iter = options.mix_em_max_iter.unwrap_or(200).clamp(1, 10_000);

        let mix_em_tol = match options.mix_em_tol {
            Some(x) if x.is_finite() && x > 0.0 => x,
            _ => 1e-6,
        };

        // ---------------------------------------------------------------------
        // Specific pi clamps
        // ---------------------------------------------------------------------
        let msfdr_pi_clamp_min = options.msfdr_pi_clamp_min.unwrap_or(0.01).clamp(0.0, 1.0);

        let msfdr_pi_clamp_max = options
            .msfdr_pi_clamp_max
            .unwrap_or(0.565)
            .clamp(0.0, 1.0)
            .max(msfdr_pi_clamp_min);

        let msfdr1_pi_clamp_min = options.msfdr1_pi_clamp_min.unwrap_or(0.01).clamp(0.0, 1.0);

        let msfdr1_pi_clamp_max = options
            .msfdr1_pi_clamp_max
            .unwrap_or(0.65)
            .clamp(0.0, 1.0)
            .max(msfdr1_pi_clamp_min);

        let msfdr2_pi_clamp_min = options.msfdr2_pi_clamp_min.unwrap_or(0.01).clamp(0.0, 1.0);

        let msfdr2_pi_clamp_max = options
            .msfdr2_pi_clamp_max
            .unwrap_or(0.568)
            .clamp(0.0, 1.0)
            .max(msfdr2_pi_clamp_min);

        // ---------------------------------------------------------------------
        // F) MSFDR1_Smix specific resolved null window + knobs
        // ---------------------------------------------------------------------
        let enable_msfdr_1smix = options.enable_msfdr_1smix.unwrap_or(true);

        let msfdr1_bottom_frac_init =
            clamp_frac(options.msfdr1_bottom_frac_init.unwrap_or(0.50), 0.50);

        let msfdr1_top_frac_init = clamp_frac(options.msfdr1_top_frac_init.unwrap_or(0.20), 0.20);

        // ---------------------------------------------------------------------
        // G) MSFDR2_Smix specific resolved null window + knobs
        // ---------------------------------------------------------------------
        let enable_msfdr_2smix = options.enable_msfdr_2smix.unwrap_or(true);

        let (msfdr2_smix_min_null_rank, msfdr2_smix_max_null_rank) = resolve_window(
            options.msfdr2_smix_min_null_rank,
            options.msfdr2_smix_max_null_rank,
            4,
            50,
        );

        let msfdr2_bottom_frac_init = clamp_frac(
            options
                .msfdr2_bottom_frac_init
                .unwrap_or(msfdr1_bottom_frac_init),
            msfdr1_bottom_frac_init,
        );

        let msfdr2_top_frac_init = clamp_frac(
            options.msfdr2_top_frac_init.unwrap_or(msfdr1_top_frac_init),
            msfdr1_top_frac_init,
        );

        // ---------------------------------------------------------------------
        // H) Nokoi specific resolved null window + knobs
        // ---------------------------------------------------------------------
        let enable_nokoi = options.enable_nokoi.unwrap_or(true);

        let (nokoi_min_null_rank, nokoi_max_null_rank) = resolve_window(
            options.nokoi_min_null_rank,
            options.nokoi_max_null_rank,
            2,
            7,
        );

        let nokoi_k_folds = options.nokoi_k_folds.unwrap_or(2).clamp(2, 20);
        let nokoi_l1_lambda_min = options.nokoi_l1_lambda_min.unwrap_or(1e-4).max(1e-12);

        let nokoi_l1_lambda_max = options
            .nokoi_l1_lambda_max
            .unwrap_or(1e-1)
            .max(nokoi_l1_lambda_min);

        let nokoi_l1_lambda_steps = options.nokoi_l1_lambda_steps.unwrap_or(10).clamp(1, 100);

        // ---------------------------------------------------------------------
        // Ensemble combination choices + weights
        // ---------------------------------------------------------------------
        let ensemble_p_combiner = options
            .ensemble_p_combiner
            .unwrap_or(EnsemblePCombiner::Cauchy);

        let ensemble_pep_combiner = options
            .ensemble_pep_combiner
            .unwrap_or(EnsemblePepCombiner::Median);

        if matches!(model_fit, ModelFit::Ensemble)
            && matches!(final_evidence_space, FinalEvidenceSpace::Auto)
        {
            panic!(
                "Invalid DF configuration: model_fit=ensemble requires explicit \
				 final_evidence_space='p_value' or final_evidence_space='pep'. \
				 Ensemble evidence routing must not use auto."
            );
        }

        let ensemble_pep_trim_frac = options
            .ensemble_pep_trim_frac
            .unwrap_or(0.20)
            .clamp(0.0, 0.49);

        let ensemble_pep_quantile = options
            .ensemble_pep_quantile
            .unwrap_or(0.50)
            .clamp(0.0, 1.0);

        let ensemble_pep_top_k = options.ensemble_pep_top_k.unwrap_or(2).max(1);

        let ensemble_pep_logit_eps = options
            .ensemble_pep_logit_eps
            .unwrap_or(1e-6)
            .clamp(1e-12, 1e-2);

        let ensemble_weight_moments = clamp_weight(options.ensemble_weight_moments);
        let ensemble_weight_mle = clamp_weight(options.ensemble_weight_mle);
        let ensemble_weight_lower_order = clamp_weight(options.ensemble_weight_lower_order);
        let ensemble_weight_msfdr_seeded = clamp_weight(options.ensemble_weight_msfdr_seeded);
        let ensemble_weight_msfdr_1smix = clamp_weight(options.ensemble_weight_msfdr_1smix);
        let ensemble_weight_msfdr_2smix = clamp_weight(options.ensemble_weight_msfdr_2smix);
        let ensemble_weight_nokoi = clamp_weight(options.ensemble_weight_nokoi);

        // ---------------------------------------------------------------------
        // Build resolved settings in the same exact order as FdrSettings
        // ---------------------------------------------------------------------
        Self {
            // =========================================================================
            // A) Global knobs
            // =========================================================================
            mode,
            entrapment_report,

            // Model selection
            model_fit,

            // Final active evidence-space controls
            final_evidence_space,

            // Protein/peptide p-value combiners
            protein_p_combine,
            peptide_p_combine,

            // Explicit q-value method controls
            psm_q_method,
            peptide_q_method,
            protein_q_method,

            peptide_fdr,
            protein_fdr,
            precursor_fdr,

            report_psms_by_peptide_q,

            // Global null window (superset pool builder)
            min_null_rank,
            max_null_rank,

            // Rank-null pool construction controls
            min_null_size,
            min_rank_count,

            // Explicit post-base Decoy-Free stage gates
            enable_rt_confidence_adjustment,
            enable_ims_confidence_adjustment,
            enable_peptide_reproducibility_rescue,
            enable_protein_reproducibility_rescue,

            // Configurable Safety Brakes
            min_storey_n,

            // Storey/π0 tuning knobs
            storey_pi0_clamp_min,
            storey_pi0_clamp_max,
            storey_lambda_min,
            storey_lambda_max,
            storey_lambda_step,
            storey_lambda_min_for_agg,
            storey_pi0_agg,

            // Storey degeneracy detector knobs
            storey_degen_same_as_median_frac,
            storey_degen_eps,
            storey_degen_pi0_eps,
            storey_degen_fallback,

            // =========================================================================
            // B) Moments specific resolved null window
            // =========================================================================
            moments_min_null_rank,
            moments_max_null_rank,
            moments_purification_factor,

            // =========================================================================
            // C) MLE specific resolved null window
            // =========================================================================
            mle_min_null_rank,
            mle_max_null_rank,
            mle_purification_factor,

            // =========================================================================
            // D) LowerOrder specific resolved null window + knobs
            // =========================================================================
            lower_order_min_null_rank,
            lower_order_max_null_rank,
            lower_order_purification_factor,

            lo_min_count_per_rank,

            lo_stratify,

            lo_evalue_candidate_count_power,
            lo_evalue_scale,
            lo_tev_transform,
            lo_tnm_extrapolation_strength,

            // =========================================================================
            // E) MSFDR specific resolved null window + knobs
            // =========================================================================
            msfdr_min_null_rank,
            msfdr_max_null_rank,
            msfdr_seeded_purification_factor,

            msfdr_seeded_top_frac_init,
            msfdr_multistart,

            msfdr_pi_clamp_min,
            msfdr_pi_clamp_max,

            // =========================================================================
            // Mixture knobs (MSFDR 1smix / pooled-rank 2smix)
            // =========================================================================
            mix_em_max_iter,
            mix_em_tol,

            // =========================================================================
            // F) MSFDR1_Smix specific resolved null window + knobs
            // =========================================================================
            msfdr1_bottom_frac_init,
            msfdr1_top_frac_init,

            msfdr1_pi_clamp_min,
            msfdr1_pi_clamp_max,

            // =========================================================================
            // G) MSFDR2_Smix specific resolved null window + knobs
            // =========================================================================
            msfdr2_smix_min_null_rank,
            msfdr2_smix_max_null_rank,

            msfdr2_bottom_frac_init,
            msfdr2_top_frac_init,

            msfdr2_pi_clamp_min,
            msfdr2_pi_clamp_max,

            // =========================================================================
            // H) Nokoi specific resolved null window + knobs
            // =========================================================================
            nokoi_min_null_rank,
            nokoi_max_null_rank,

            nokoi_null_purification_factor,
            nokoi_positive_top_fraction,

            nokoi_k_folds,

            nokoi_l1_lambda_min,
            nokoi_l1_lambda_max,
            nokoi_l1_lambda_steps,

            // =========================================================================
            // I) Ensemble specific knobs
            // =========================================================================
            enable_moments,
            enable_mle,
            enable_lower_order,
            enable_msfdr_seeded,
            enable_msfdr_1smix,
            enable_msfdr_2smix,
            enable_nokoi,

            ensemble_p_combiner,
            ensemble_pep_combiner,

            ensemble_pep_trim_frac,
            ensemble_pep_quantile,
            ensemble_pep_top_k,
            ensemble_pep_logit_eps,

            ensemble_weight_moments,
            ensemble_weight_mle,
            ensemble_weight_lower_order,
            ensemble_weight_msfdr_seeded,
            ensemble_weight_msfdr_1smix,
            ensemble_weight_msfdr_2smix,
            ensemble_weight_nokoi,

            // =========================================================================
            // J) Layer 2: Physical confidence adjustment: RT and IMS Knobs
            //    Layer 3: Reproducibility rescue: peptide and protein
            // =========================================================================
            physical_rescue,
            reproducibility,
        }
    }
}
