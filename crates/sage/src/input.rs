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
pub enum FdrType {
    #[default]
    Bh, // Benjamini-Hochberg
    Storey, // Storey-Tibshirani
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProteinPCombine {
    #[default]
    Fisher,
    Cauchy,
    SidakMinP,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PeptidePCombine {
    Fisher,
    #[default]
    Cauchy,
    SidakMinP,
    Best,
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
pub enum LoMeanBetaMode {
    #[default]
    Consecutive, // Original paper behavior (n consecutive ranks)
    All, // PyLord behavior (all ranks >= min_rank)
}

/// How to rank/monotonize LO-derived values.
/// - hyperscore: sort by hyperscore (legacy)
/// - lo_adjusted: sort by LO-adjusted evidence (recommended; fixes LO + PAVA mismatch)
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoRankKey {
    #[default]
    Hyperscore,
    LoAdjusted,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoMode {
    #[default]
    Auto, // evaluate both TNM constructions; pick best BIC
    LinearRegression,
    MeanBeta,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoLomEstimator {
    #[default]
    Auto, // evaluate MM and MLE; pick best BIC
    Mm,
    Mle,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoStratify {
    Global,
    #[default]
    Charge,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoScore {
    /// Use existing tev(f) selection (hyperscore vs lo_adjusted) — no extra normalization.
    #[default]
    Raw,

    /// PyLord-style per-spectrum normalization (implemented in decoy_free_fdr.rs).
    PerSpectrum,
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
pub enum L3AnchorMode {
    Best,
    #[default]
    SecondBest,
    Mean,
    Median,
    TrimmedMean,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum L3RescueMode {
    Replace,
    #[default]
    BoundedShrinkage,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct DartBayesConfig {
    pub dart_use_bootstrap: bool,
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
    pub mode: PhysicalRescueMode,
    pub anchor_mode: PhysicalAnchorMode,
    pub anchor_max_pep: f64,
    pub anchor_max_q: f64,
    pub min_anchor_count_per_run: usize,
    pub min_anchor_count_per_charge: usize,
    pub rt_enabled: bool,
    pub ims_enabled: bool,
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
pub struct L3ProteinEligibilityConfig {
    pub enabled: bool,
    pub q_threshold_l2: f64,
    pub min_unique_passing_peptides: usize,
    pub min_unique_passing_fraction: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct L3PeptideEligibilityConfig {
    pub min_run_fraction: f64,
    pub min_run_count: usize,
    pub strong_reference_q_threshold_l2: f64,
    pub strong_reference_pep_threshold_l2: Option<f64>,
    pub min_strong_run_fraction: f64,
    pub min_strong_run_count: usize,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct L3AnchorConfig {
    pub mode: L3AnchorMode,
    pub trim_fraction: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct L3RescueBandConfig {
    pub strong_cutoff_pep_l2: f64,
    pub weak_cutoff_pep_l2: f64,
    pub max_rescue_fraction: f64,
    pub rescue_mode: L3RescueMode,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ReproducibilityConfig {
    // Global Layer 3 controls
    pub enabled: bool,
    pub max_total_shift: f64,
    pub max_agreement_shift: f64,
    pub max_recurrence_shift: f64,
    pub use_expert_agreement: bool,
    pub use_cross_run_recurrence: bool,
    pub redundancy_discount: f64,

    // Eligibility controls
    pub protein_eligibility: L3ProteinEligibilityConfig,
    pub peptide_eligibility: L3PeptideEligibilityConfig,

    // Anchor / rescue controls
    pub anchor: L3AnchorConfig,
    pub rescue_band: L3RescueBandConfig,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct FdrOptions {
    // =========================================================================
    // A) Global knobs
    // =========================================================================
    pub mode: Option<FdrMode>,
    pub physical_rescue: Option<PhysicalRescueConfig>,
    pub reproducibility: Option<ReproducibilityConfig>,
    pub peptide_fdr: Option<f32>,
    pub protein_fdr: Option<f32>,
    pub precursor_fdr: Option<f32>,
    pub entrapment_report: Option<EntrapmentReportMode>,

    // Global null window (superset pool builder)
    pub min_null_rank: Option<u32>,
    pub max_null_rank: Option<u32>,

    // Rank-null pool construction controls
    pub purification_factor: Option<f64>, // default 0.20; clamp 0..0.9
    pub min_rank_count: Option<usize>,    // default 10

    // Model selection + global FDR type
    pub model_fit: Option<ModelFit>,
    #[serde(alias = "type")]
    pub type_: Option<FdrType>,
    pub protein_p_combine: Option<ProteinPCombine>,
    pub peptide_p_combine: Option<PeptidePCombine>,

    // Configurable Safety Brakes (global)
    pub min_storey_n: Option<usize>,
    pub min_null_size: Option<usize>,
    pub kde_samples: Option<usize>,

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

    // Ensemble combination choices (global controls; used by ModelFit::Ensemble)
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
    // B) Moments specific knobs
    // =========================================================================
    pub moments_min_null_rank: Option<u32>,
    pub moments_max_null_rank: Option<u32>,

    // =========================================================================
    // C) MLE specific knobs
    // =========================================================================
    pub mle_min_null_rank: Option<u32>,
    pub mle_max_null_rank: Option<u32>,

    // =========================================================================
    // D) LowerOrder specific knobs
    // =========================================================================
    pub lower_order_min_null_rank: Option<u32>,
    pub lower_order_max_null_rank: Option<u32>,

    // LO (paper/PyLord) controls
    pub lo_rank_key: Option<LoRankKey>, // unchanged: output rank key (lo_adjusted vs hyperscore)
    pub lo_mode: Option<LoMode>,        // auto | linear_regression | mean_beta
    pub lo_lom_estimator: Option<LoLomEstimator>, // auto | mm | mle

    // LowerOrder support threshold.
    // This is not a rank-window selector. The selected LO ranks are controlled
    // only by lower_order_min_null_rank..=lower_order_max_null_rank.
    pub lo_min_count_per_rank: Option<usize>,

    // PyLord parity knobs
    pub lo_stratify: Option<LoStratify>, // default Charge
    pub lo_score: Option<LoScore>,       // default Raw
    pub lo_tev_cutoff: Option<f64>,      // default 0.18

    // Mean-β scheme controls (paper defaults: min_rank=8, count=3)
    pub lo_mean_beta_mode: Option<LoMeanBetaMode>, // default consecutive

    // =========================================================================
    // E) MSFDR specific knobs
    // =========================================================================
    pub msfdr_min_null_rank: Option<u32>,
    pub msfdr_max_null_rank: Option<u32>,
    pub msfdr_multistart: Option<usize>,

    // =========================================================================
    // Mixture knobs (MSFDR 1smix / 2smix)
    // =========================================================================
    pub mix_em_max_iter: Option<usize>, // default 200; clamp 1..10_000
    pub mix_em_tol: Option<f64>,        // default 1e-6; must be >0
    pub mix_pi_clamp_min: Option<f64>,  // default 0.01
    pub mix_pi_clamp_max: Option<f64>,  // default 0.99
    pub mix_anchor_incorrect: Option<bool>, // default true (for 2smix)

    // --- Specific clamps (overrides) ---
    pub msfdr_pi_clamp_min: Option<f64>,
    pub msfdr_pi_clamp_max: Option<f64>,
    pub msfdr1_pi_clamp_min: Option<f64>,
    pub msfdr1_pi_clamp_max: Option<f64>,
    pub msfdr2_pi_clamp_min: Option<f64>,
    pub msfdr2_pi_clamp_max: Option<f64>,

    // =========================================================================
    // F) MSFDR1_Smix specific knobs
    // =========================================================================
    pub msfdr1_smix_min_null_rank: Option<u32>,
    pub msfdr1_smix_max_null_rank: Option<u32>,

    // MSFDR init/drift knobs (needed by real models)
    pub msfdr1_bottom_frac_init: Option<f64>, // default 0.7
    pub msfdr_seeded_top_frac_init: Option<f64>, // default 0.2
    pub msfdr1_top_frac_init: Option<f64>,    // default 0.2

    // Expose drift clamps for MSFDR1
    pub msfdr1_beta_drift_mult: Option<(f64, f64)>, // default (0.8, 1.25)
    pub msfdr1_mu_drift_abs: Option<f64>,           // default 0.5

    // =========================================================================
    // G) MSFDR2_Smix specific knobs
    // =========================================================================
    pub msfdr2_smix_min_null_rank: Option<u32>,
    pub msfdr2_smix_max_null_rank: Option<u32>,

    pub msfdr2_beta_drift_mult: Option<(f64, f64)>, // default (0.5, 2.0)
    pub msfdr2_mu_drift_abs: Option<f64>,           // default 5.0
    pub msfdr2_top_frac_init: Option<f64>, // default = msfdr1_top_frac_init (optional but clean)

    // Ensemble expert gates (Ensemble uses these; explicit model_fit variants override gates)
    pub enable_moments: Option<bool>,      // default true
    pub enable_mle: Option<bool>,          // default true
    pub enable_lower_order: Option<bool>,  // default true
    pub enable_msfdr_seeded: Option<bool>, // default true
    pub enable_msfdr_1smix: Option<bool>,  // default true
    pub enable_msfdr_2smix: Option<bool>,  // default true
    pub enable_nokoi: Option<bool>,        // default true

    // =========================================================================
    // H) Nokoi specific knobs
    // =========================================================================
    pub nokoi_min_null_rank: Option<u32>,
    pub nokoi_max_null_rank: Option<u32>,

    // Nokoi DF cross-fit calibration
    pub nokoi_k_folds: Option<usize>,

    // Nokoi L1 lambda grid (JSON-exposed)
    pub nokoi_l1_lambda_min: Option<f64>,
    pub nokoi_l1_lambda_max: Option<f64>,
    pub nokoi_l1_lambda_steps: Option<usize>,
}

#[derive(Clone, Serialize, Debug)]
pub struct FdrSettings {
    // =========================================================================
    // A) Global knobs
    // =========================================================================
    pub mode: FdrMode,
    pub physical_rescue: PhysicalRescueConfig,
    pub reproducibility: ReproducibilityConfig,
    pub peptide_fdr: f32,
    pub protein_fdr: f32,
    pub precursor_fdr: f32,

    #[serde(default)]
    pub entrapment_report: EntrapmentReportMode,

    // Global null window (superset pool builder)
    pub min_null_rank: u32,
    pub max_null_rank: u32,

    // =========================================================================
    // B) Moments specific resolved null window
    // =========================================================================
    pub moments_min_null_rank: u32,
    pub moments_max_null_rank: u32,

    // =========================================================================
    // C) MLE specific resolved null window
    // =========================================================================
    pub mle_min_null_rank: u32,
    pub mle_max_null_rank: u32,

    // =========================================================================
    // D) LowerOrder specific resolved null window
    // =========================================================================
    pub lower_order_min_null_rank: u32,
    pub lower_order_max_null_rank: u32,
    pub lo_rank_key: LoRankKey,

    // LO (paper/PyLord) settings
    pub lo_mode: LoMode,
    pub lo_lom_estimator: LoLomEstimator,

    /// Minimum number of observations required for an individual selected
    /// lower-order rank to contribute to the LowerOrder fit.
    ///
    /// This is not a rank-window selector. The selected LO ranks are controlled
    /// only by lower_order_min_null_rank..=lower_order_max_null_rank.
    pub lo_min_count_per_rank: usize,

    // PyLord parity settings
    pub lo_stratify: LoStratify, // Charge | Global
    pub lo_score: LoScore,       // Raw | PerSpectrum
    pub lo_tev_cutoff: f64,

    pub lo_mean_beta_mode: LoMeanBetaMode,

    // =========================================================================
    // E) MSFDR specific resolved null window
    // =========================================================================
    pub msfdr_min_null_rank: u32,
    pub msfdr_max_null_rank: u32,

    // =========================================================================
    // F) MSFDR1_Smix specific resolved null window
    // =========================================================================
    pub msfdr1_smix_min_null_rank: u32,
    pub msfdr1_smix_max_null_rank: u32,

    // =========================================================================
    // G) MSFDR2_Smix specific resolved null window
    // =========================================================================
    pub msfdr2_smix_min_null_rank: u32,
    pub msfdr2_smix_max_null_rank: u32,

    // =========================================================================
    // H) Nokoi specific resolved null window
    // =========================================================================
    pub nokoi_min_null_rank: u32,
    pub nokoi_max_null_rank: u32,

    // Global model selection + FDR type
    pub model_fit: ModelFit,
    pub type_: FdrType,
    pub protein_p_combine: ProteinPCombine,
    pub peptide_p_combine: PeptidePCombine,

    // Configurable Safety Brakes
    pub min_storey_n: usize,
    pub min_null_size: usize,
    pub kde_samples: usize,

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

    // Nokoi DF cross-fit calibration
    pub nokoi_k_folds: usize,

    // Nokoi L1 lambda grid
    pub nokoi_l1_lambda_min: f64,
    pub nokoi_l1_lambda_max: f64,
    pub nokoi_l1_lambda_steps: usize,

    // Ensemble combination choices
    pub ensemble_pep_combiner: EnsemblePepCombiner,

    // Shared robust-combiner knobs
    pub ensemble_pep_trim_frac: f64,
    pub ensemble_pep_quantile: f64,
    pub ensemble_pep_top_k: usize,
    pub ensemble_pep_logit_eps: f64,

    // Static per-expert weights
    pub ensemble_weight_moments: f64,
    pub ensemble_weight_mle: f64,
    pub ensemble_weight_lower_order: f64,
    pub ensemble_weight_msfdr_seeded: f64,
    pub ensemble_weight_msfdr_1smix: f64,
    pub ensemble_weight_msfdr_2smix: f64,
    pub ensemble_weight_nokoi: f64,

    // MSFDR controls
    pub msfdr_multistart: usize,

    // MSFDR init/drift knobs (needed by real models)
    pub msfdr_seeded_top_frac_init: f64,
    pub msfdr1_top_frac_init: f64,
    pub msfdr1_bottom_frac_init: f64,
    pub msfdr1_beta_drift_mult: (f64, f64),
    pub msfdr1_mu_drift_abs: f64,
    pub msfdr2_beta_drift_mult: (f64, f64),
    pub msfdr2_mu_drift_abs: f64,
    pub msfdr2_top_frac_init: f64, // optional but clean

    // Ensemble expert gates (Ensemble uses these; explicit model_fit variants override gates)
    pub enable_moments: bool,      // default true
    pub enable_mle: bool,          // default true
    pub enable_lower_order: bool,  // default true
    pub enable_msfdr_seeded: bool, // default true
    pub enable_msfdr_1smix: bool,  // default true
    pub enable_msfdr_2smix: bool,  // default true
    pub enable_nokoi: bool,        // default true

    // Mixture knobs (MSFDR 1smix / 2smix)
    pub mix_em_max_iter: usize,
    pub mix_em_tol: f64,
    pub mix_pi_clamp_min: f64,
    pub mix_pi_clamp_max: f64,
    pub mix_anchor_incorrect: bool,

    // --- Specific clamps ---
    pub msfdr_pi_clamp_min: f64,
    pub msfdr_pi_clamp_max: f64,
    pub msfdr1_pi_clamp_min: f64,
    pub msfdr1_pi_clamp_max: f64,
    pub msfdr2_pi_clamp_min: f64,
    pub msfdr2_pi_clamp_max: f64,

    // Rank-null pool construction controls
    pub purification_factor: f64,
    pub min_rank_count: usize,
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
                mode: PhysicalRescueMode::Off,
                anchor_mode: PhysicalAnchorMode::Default,
                anchor_max_pep: 0.1,
                anchor_max_q: 0.01,
                min_anchor_count_per_run: 10,
                min_anchor_count_per_charge: 5,
                rt_enabled: false,
                ims_enabled: false,
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

                protein_eligibility: L3ProteinEligibilityConfig {
                    enabled: true,
                    q_threshold_l2: 0.01,
                    min_unique_passing_peptides: 2,
                    min_unique_passing_fraction: None,
                },

                peptide_eligibility: L3PeptideEligibilityConfig {
                    min_run_fraction: 0.6,
                    min_run_count: 2,
                    strong_reference_q_threshold_l2: 0.01,
                    strong_reference_pep_threshold_l2: None,
                    min_strong_run_fraction: 0.2,
                    min_strong_run_count: 1,
                },

                anchor: L3AnchorConfig {
                    mode: L3AnchorMode::SecondBest,
                    trim_fraction: Some(0.1),
                },

                rescue_band: L3RescueBandConfig {
                    strong_cutoff_pep_l2: 0.01,
                    weak_cutoff_pep_l2: 0.25,
                    max_rescue_fraction: 0.5,
                    rescue_mode: L3RescueMode::BoundedShrinkage,
                },
            });

        let precursor_fdr = options.precursor_fdr.unwrap_or(0.01);
        let peptide_fdr = options.peptide_fdr.unwrap_or(0.01);
        let protein_fdr = options.protein_fdr.unwrap_or(0.01);
        let entrapment_report = options
            .entrapment_report
            .unwrap_or(EntrapmentReportMode::Auto);

        let model_fit = options.model_fit.unwrap_or(ModelFit::Ensemble);
        let type_ = options.type_.unwrap_or(FdrType::Storey);
        let protein_p_combine = options.protein_p_combine.unwrap_or(ProteinPCombine::Cauchy);
        let peptide_p_combine = options.peptide_p_combine.unwrap_or(PeptidePCombine::Cauchy);

        let min_storey_n = options.min_storey_n.unwrap_or(300);
        let min_null_size = options.min_null_size.unwrap_or(300);
        let kde_samples = options.kde_samples.unwrap_or(50_000);

        let purification_factor = options.purification_factor.unwrap_or(0.50).clamp(0.0, 0.9);
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

        let lo_rank_key = options.lo_rank_key.unwrap_or(LoRankKey::LoAdjusted);
        let lo_mode = options.lo_mode.unwrap_or(LoMode::Auto);
        let lo_lom_estimator = options.lo_lom_estimator.unwrap_or(LoLomEstimator::Auto);
        let lo_min_count_per_rank = options.lo_min_count_per_rank.unwrap_or(10).max(1);
        let lo_stratify = options.lo_stratify.unwrap_or(LoStratify::Charge);
        let lo_score = options.lo_score.unwrap_or(LoScore::Raw);
        let lo_tev_cutoff = options.lo_tev_cutoff.unwrap_or(0.18).clamp(0.01, 1.0);

        let lo_mean_beta_mode = options
            .lo_mean_beta_mode
            .unwrap_or(LoMeanBetaMode::Consecutive);

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

        let mix_pi_clamp_min = options.mix_pi_clamp_min.unwrap_or(0.01).clamp(0.0, 1.0);

        let mix_pi_clamp_max = options
            .mix_pi_clamp_max
            .unwrap_or(0.565)
            .clamp(0.0, 1.0)
            .max(mix_pi_clamp_min);

        let mix_anchor_incorrect = options.mix_anchor_incorrect.unwrap_or(true);

        // ---------------------------------------------------------------------
        // Specific pi clamps
        // ---------------------------------------------------------------------
        let msfdr_pi_clamp_min = options
            .msfdr_pi_clamp_min
            .unwrap_or(mix_pi_clamp_min)
            .clamp(0.0, 1.0);

        let msfdr_pi_clamp_max = options
            .msfdr_pi_clamp_max
            .unwrap_or(mix_pi_clamp_max)
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

        let (msfdr1_smix_min_null_rank, msfdr1_smix_max_null_rank) = resolve_window(
            options.msfdr1_smix_min_null_rank,
            options.msfdr1_smix_max_null_rank,
            5,
            50,
        );

        let msfdr1_bottom_frac_init =
            clamp_frac(options.msfdr1_bottom_frac_init.unwrap_or(0.50), 0.50);

        let msfdr1_top_frac_init = clamp_frac(options.msfdr1_top_frac_init.unwrap_or(0.20), 0.20);

        let msfdr1_beta_drift_mult = match options.msfdr1_beta_drift_mult {
            Some((a, b)) if a.is_finite() && b.is_finite() && a > 0.0 && b >= a => (a, b),
            _ => (0.9, 1.1),
        };

        let msfdr1_mu_drift_abs = match options.msfdr1_mu_drift_abs {
            Some(x) if x.is_finite() && x >= 0.0 => x,
            _ => 0.5,
        };

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

        let msfdr2_top_frac_init = clamp_frac(
            options.msfdr2_top_frac_init.unwrap_or(msfdr1_top_frac_init),
            msfdr1_top_frac_init,
        );

        let msfdr2_beta_drift_mult = match options.msfdr2_beta_drift_mult {
            Some((a, b)) if a.is_finite() && b.is_finite() && a > 0.0 && b >= a => (a, b),
            _ => (0.5, 2.0),
        };

        let msfdr2_mu_drift_abs = match options.msfdr2_mu_drift_abs {
            Some(x) if x.is_finite() && x >= 0.0 => x,
            _ => 0.5,
        };

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
        let ensemble_pep_combiner = options
            .ensemble_pep_combiner
            .unwrap_or(EnsemblePepCombiner::Median);

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
            physical_rescue,
            reproducibility,
            peptide_fdr,
            protein_fdr,
            precursor_fdr,
            entrapment_report,

            // Global null window (superset pool builder)
            min_null_rank,
            max_null_rank,

            // =========================================================================
            // B) Moments specific resolved null window
            // =========================================================================
            moments_min_null_rank,
            moments_max_null_rank,

            // =========================================================================
            // C) MLE specific resolved null window
            // =========================================================================
            mle_min_null_rank,
            mle_max_null_rank,

            // =========================================================================
            // D) LowerOrder specific resolved null window
            // =========================================================================
            lower_order_min_null_rank,
            lower_order_max_null_rank,
            lo_rank_key,

            // LO (paper/PyLord) settings
            lo_mode,
            lo_lom_estimator,
            lo_min_count_per_rank,

            // PyLord parity settings
            lo_stratify,
            lo_score,
            lo_tev_cutoff,

            lo_mean_beta_mode,

            // =========================================================================
            // E) MSFDR specific resolved null window
            // =========================================================================
            msfdr_min_null_rank,
            msfdr_max_null_rank,

            // =========================================================================
            // F) MSFDR1_Smix specific resolved null window
            // =========================================================================
            msfdr1_smix_min_null_rank,
            msfdr1_smix_max_null_rank,

            // =========================================================================
            // G) MSFDR2_Smix specific resolved null window
            // =========================================================================
            msfdr2_smix_min_null_rank,
            msfdr2_smix_max_null_rank,

            // =========================================================================
            // H) Nokoi specific resolved null window
            // =========================================================================
            nokoi_min_null_rank,
            nokoi_max_null_rank,

            // Global model selection + FDR type
            model_fit,
            type_,
            protein_p_combine,
            peptide_p_combine,

            // Configurable Safety Brakes
            min_storey_n,
            min_null_size,
            kde_samples,

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

            // Nokoi DF cross-fit calibration
            nokoi_k_folds,

            // Nokoi L1 lambda grid
            nokoi_l1_lambda_min,
            nokoi_l1_lambda_max,
            nokoi_l1_lambda_steps,

            // Ensemble combination choices
            ensemble_pep_combiner,

            // Shared robust-combiner knobs
            ensemble_pep_trim_frac,
            ensemble_pep_quantile,
            ensemble_pep_top_k,
            ensemble_pep_logit_eps,

            // Static per-expert weights
            ensemble_weight_moments,
            ensemble_weight_mle,
            ensemble_weight_lower_order,
            ensemble_weight_msfdr_seeded,
            ensemble_weight_msfdr_1smix,
            ensemble_weight_msfdr_2smix,
            ensemble_weight_nokoi,

            // MSFDR controls
            msfdr_multistart,

            // MSFDR init/drift knobs (needed by real models)
            msfdr_seeded_top_frac_init,
            msfdr1_top_frac_init,
            msfdr1_bottom_frac_init,
            msfdr1_beta_drift_mult,
            msfdr1_mu_drift_abs,
            msfdr2_beta_drift_mult,
            msfdr2_mu_drift_abs,
            msfdr2_top_frac_init,

            // Ensemble expert gates (Ensemble uses these; explicit model_fit variants override gates)
            enable_moments,
            enable_mle,
            enable_lower_order,
            enable_msfdr_seeded,
            enable_msfdr_1smix,
            enable_msfdr_2smix,
            enable_nokoi,

            // Mixture knobs (MSFDR 1smix / 2smix)
            mix_em_max_iter,
            mix_em_tol,
            mix_pi_clamp_min,
            mix_pi_clamp_max,
            mix_anchor_incorrect,

            // --- Specific clamps ---
            msfdr_pi_clamp_min,
            msfdr_pi_clamp_max,
            msfdr1_pi_clamp_min,
            msfdr1_pi_clamp_max,
            msfdr2_pi_clamp_min,
            msfdr2_pi_clamp_max,

            // Rank-null pool construction controls
            purification_factor,
            min_rank_count,
        }
    }
}
