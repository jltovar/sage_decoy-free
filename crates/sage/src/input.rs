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
}

/// Mandatory PEP derivation for null-only methods (Moments/MLE/LO).
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NullOnlyPepMode {
    /// Approximate but simple: treat the method’s canonical p-value proxy as PEP.
    #[default]
    PepEqualsP,
    /// Optional: derive an approximate PEP from q-value heuristics (later step; may remain unused).
    PepFromQHeuristic,
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

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct FdrOptions {
    // =========================================================================
    // A) Global knobs
    // =========================================================================
    pub mode: Option<FdrMode>,
    pub peptide_fdr: Option<f32>,
    pub protein_fdr: Option<f32>,
    pub precursor_fdr: Option<f32>,

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

    // Null-only PEP strategy (Moments/MLE/LO)
    pub null_only_pep_mode: Option<NullOnlyPepMode>,

    // Per-method calibration / ranking controls
    pub calibrate_per_method: Option<bool>,

    // Ensemble combination choices (global controls; used by ModelFit::Ensemble)
    pub ensemble_pep_combiner: Option<EnsemblePepCombiner>,

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

    // PyLord parity knobs
    pub lo_stratify: Option<LoStratify>, // default Charge
    pub lo_score: Option<LoScore>,       // default Raw

    // Mean-β scheme controls (paper defaults: min_rank=8, count=3)
    pub lo_mean_beta_mode: Option<LoMeanBetaMode>, // default consecutive
    pub lo_mean_beta_min_rank: Option<u32>,        // default 8
    pub lo_mean_beta_count: Option<u32>,           // default 3

    // (Optional) if you later implement PyLord sliding-window LR selection
    pub lo_lr_window_size: Option<u32>, // default None (disabled)

    // =========================================================================
    // E) MSFDR specific knobs
    // =========================================================================
    pub msfdr_min_null_rank: Option<u32>,
    pub msfdr_max_null_rank: Option<u32>,

    pub msfdr_use_canonical_pep: Option<bool>,
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
    pub peptide_fdr: f32,
    pub protein_fdr: f32,
    pub precursor_fdr: f32,

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

    // PyLord parity settings
    pub lo_stratify: LoStratify, // Charge | Global
    pub lo_score: LoScore,       // Raw | PerSpectrum

    pub lo_mean_beta_mode: LoMeanBetaMode,
    pub lo_mean_beta_min_rank: u32,
    pub lo_mean_beta_count: u32,
    pub lo_lr_window_size: Option<u32>,

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

    // Per-method calibration / ranking controls
    pub calibrate_per_method: bool,

    // Nokoi DF cross-fit calibration
    pub nokoi_k_folds: usize,

    // Nokoi L1 lambda grid
    pub nokoi_l1_lambda_min: f64,
    pub nokoi_l1_lambda_max: f64,
    pub nokoi_l1_lambda_steps: usize,

    // Null-only PEP strategy (Moments/MLE/LO)
    pub null_only_pep_mode: NullOnlyPepMode,

    // Ensemble combination choices
    pub ensemble_pep_combiner: EnsemblePepCombiner,

    // MSFDR controls
    pub msfdr_use_canonical_pep: bool,
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
        // --- Storey / pi0 knobs ---
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

        // --- Storey degeneracy knobs ---
        let storey_degen_same_as_median_frac = options
            .storey_degen_same_as_median_frac
            .unwrap_or(0.90)
            .clamp(0.0, 1.0);

        let storey_degen_eps = options.storey_degen_eps.unwrap_or(1e-6).max(0.0);

        let storey_degen_pi0_eps = options.storey_degen_pi0_eps.unwrap_or(1e-3).max(0.0);

        let storey_degen_fallback = options
            .storey_degen_fallback
            .unwrap_or(StoreyDegeneracyFallback::Bh);

        let calibrate_per_method = options.calibrate_per_method.unwrap_or(true);
        let lo_rank_key = options.lo_rank_key.unwrap_or(LoRankKey::LoAdjusted);
        let lo_mode = options.lo_mode.unwrap_or(LoMode::Auto);
        let lo_lom_estimator = options.lo_lom_estimator.unwrap_or(LoLomEstimator::Auto);

        // PyLord parity knobs (defaults)
        let lo_stratify = options.lo_stratify.unwrap_or(LoStratify::Charge);
        let lo_score = options.lo_score.unwrap_or(LoScore::Raw);

        let lo_mean_beta_mode = options
            .lo_mean_beta_mode
            .unwrap_or(LoMeanBetaMode::Consecutive);
        let lo_mean_beta_min_rank = options.lo_mean_beta_min_rank.unwrap_or(8).max(2);
        let lo_mean_beta_count = options.lo_mean_beta_count.unwrap_or(3).clamp(1, 10);

        let lo_lr_window_size = options.lo_lr_window_size;

        let nokoi_k_folds = options.nokoi_k_folds.unwrap_or(2).max(2).min(20);

        let nokoi_l1_lambda_min = options.nokoi_l1_lambda_min.unwrap_or(1e-4).max(1e-12);

        let nokoi_l1_lambda_max = options
            .nokoi_l1_lambda_max
            .unwrap_or(1e-1)
            .max(nokoi_l1_lambda_min);

        let nokoi_l1_lambda_steps = options.nokoi_l1_lambda_steps.unwrap_or(10).clamp(1, 100);

        let null_only_pep_mode = options
            .null_only_pep_mode
            .unwrap_or(NullOnlyPepMode::PepFromQHeuristic);

        let ensemble_pep_combiner = options
            .ensemble_pep_combiner
            .unwrap_or(EnsemblePepCombiner::Median);

        let msfdr_use_canonical_pep = options.msfdr_use_canonical_pep.unwrap_or(true);

        // Keep small/safe; avoids pathological huge multistarts.
        let msfdr_multistart = options.msfdr_multistart.unwrap_or(3).clamp(1, 25);

        // --- MSFDR init/drift knobs ---
        let clamp_frac = |x: f64, default: f64| -> f64 {
            if x.is_finite() {
                x.clamp(0.01, 0.99)
            } else {
                default
            }
        };

        let msfdr_seeded_top_frac_init =
            clamp_frac(options.msfdr_seeded_top_frac_init.unwrap_or(0.20), 0.20);

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

        // Ensemble expert gates (Ensemble uses these; explicit model_fit variants override gates)
        let enable_moments = options.enable_moments.unwrap_or(true);
        let enable_mle = options.enable_mle.unwrap_or(true);
        let enable_lower_order = options.enable_lower_order.unwrap_or(true);
        let enable_msfdr_seeded = options.enable_msfdr_seeded.unwrap_or(true);
        let enable_msfdr_1smix = options.enable_msfdr_1smix.unwrap_or(true);
        let enable_msfdr_2smix = options.enable_msfdr_2smix.unwrap_or(true);

        // Mixture knobs
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

        // --- Specific clamps (fallback to global mix_pi_clamp) ---
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

        // Default true; only meaningful for 2smix
        let mix_anchor_incorrect = options.mix_anchor_incorrect.unwrap_or(true);

        let purification_factor = options.purification_factor.unwrap_or(0.50).clamp(0.0, 0.9);
        let min_rank_count = options.min_rank_count.unwrap_or(10);

        // ---------------------------------------------------------------------
        // Per-method null windows: resolve defaults, swap if inverted, clamp to
        // the global [min_null_rank..=max_null_rank] window.
        // ---------------------------------------------------------------------

        // Global superset pool window (also normalized if user inverted it)
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

            // Swap if user inverted
            if mn > mx {
                std::mem::swap(&mut mn, &mut mx);
            }

            // Clamp to global window
            mn = mn.clamp(min_null_rank, max_null_rank);
            mx = mx.clamp(min_null_rank, max_null_rank);

            // Repair if clamp caused inversion
            if mn > mx {
                mn = mn.clamp(min_null_rank, max_null_rank);
                mx = mn;
            }

            (mn, mx)
        };

        // Paper-default windows (per your spec)
        let (moments_min_null_rank, moments_max_null_rank) = resolve_window(
            options.moments_min_null_rank,
            options.moments_max_null_rank,
            4,
            50,
        );

        let (mle_min_null_rank, mle_max_null_rank) =
            resolve_window(options.mle_min_null_rank, options.mle_max_null_rank, 4, 50);

        let (lower_order_min_null_rank, lower_order_max_null_rank) = resolve_window(
            options.lower_order_min_null_rank,
            options.lower_order_max_null_rank,
            6,
            12,
        );

        // MSFDR seeded null window defaults (clamped to global [min_null_rank..=max_null_rank])
        let (msfdr_min_null_rank, msfdr_max_null_rank) = resolve_window(
            options.msfdr_min_null_rank,
            options.msfdr_max_null_rank,
            4,
            50,
        );

        let (msfdr1_smix_min_null_rank, msfdr1_smix_max_null_rank) = resolve_window(
            options.msfdr1_smix_min_null_rank,
            options.msfdr1_smix_max_null_rank,
            5,
            50,
        );

        let (msfdr2_smix_min_null_rank, msfdr2_smix_max_null_rank) = resolve_window(
            options.msfdr2_smix_min_null_rank,
            options.msfdr2_smix_max_null_rank,
            4,
            50,
        );

        // Nokoi negative class defaults: ranks 2..5
        let (nokoi_min_null_rank, nokoi_max_null_rank) = resolve_window(
            options.nokoi_min_null_rank,
            options.nokoi_max_null_rank,
            2,
            7,
        );

        Self {
            mode: options.mode.unwrap_or(FdrMode::DecoyFree),

            precursor_fdr: options.precursor_fdr.unwrap_or(0.01),
            peptide_fdr: options.peptide_fdr.unwrap_or(0.01),
            protein_fdr: options.protein_fdr.unwrap_or(0.01),

            // Global null window (superset pool builder)
            min_null_rank,
            max_null_rank,

            // Global parameters
            purification_factor,
            min_rank_count,

            mix_em_max_iter,
            mix_em_tol,
            mix_pi_clamp_min,
            mix_pi_clamp_max,
            mix_anchor_incorrect,

            model_fit: options.model_fit.unwrap_or(ModelFit::Ensemble),
            type_: options.type_.unwrap_or(FdrType::Storey),
            protein_p_combine: options.protein_p_combine.unwrap_or(ProteinPCombine::Cauchy),

            min_storey_n: options.min_storey_n.unwrap_or(300),
            min_null_size: options.min_null_size.unwrap_or(300),
            kde_samples: options.kde_samples.unwrap_or(50_000),

            storey_pi0_clamp_min,
            storey_pi0_clamp_max,
            storey_lambda_min,
            storey_lambda_max,
            storey_lambda_step,
            storey_lambda_min_for_agg,
            storey_pi0_agg,

            storey_degen_same_as_median_frac,
            storey_degen_eps,
            storey_degen_pi0_eps,
            storey_degen_fallback,

            calibrate_per_method,
            null_only_pep_mode,

            // Moments-specific resolved null window
            enable_moments,
            moments_min_null_rank,
            moments_max_null_rank,

            // MLE-specific resolved null window
            enable_mle,
            mle_min_null_rank,
            mle_max_null_rank,

            // LowerOrder-specific resolved null window
            enable_lower_order,
            lower_order_min_null_rank,
            lower_order_max_null_rank,
            lo_rank_key,
            lo_mode,
            lo_lom_estimator,
            lo_stratify,
            lo_score,
            lo_mean_beta_mode,
            lo_mean_beta_min_rank,
            lo_mean_beta_count,
            lo_lr_window_size,

            // MSFDR-specific resolved null window
            enable_msfdr_seeded,
            msfdr_seeded_top_frac_init,
            msfdr_min_null_rank,
            msfdr_max_null_rank,
            msfdr_use_canonical_pep,
            msfdr_multistart,
            msfdr_pi_clamp_min,
            msfdr_pi_clamp_max,

            // MSFDR1_Smix-specific resolved null window
            enable_msfdr_1smix,
            msfdr1_smix_min_null_rank,
            msfdr1_smix_max_null_rank,
            msfdr1_bottom_frac_init,
            msfdr1_top_frac_init,
            msfdr1_beta_drift_mult,
            msfdr1_mu_drift_abs,
            msfdr1_pi_clamp_min,
            msfdr1_pi_clamp_max,

            // MSFDR2_Smix-specific resolved null window
            enable_msfdr_2smix,
            msfdr2_smix_min_null_rank,
            msfdr2_smix_max_null_rank,
            msfdr2_beta_drift_mult,
            msfdr2_mu_drift_abs,
            msfdr2_top_frac_init,
            msfdr2_pi_clamp_min,
            msfdr2_pi_clamp_max,

            // Nokoi-specific resolved null window
            nokoi_min_null_rank,
            nokoi_max_null_rank,
            nokoi_k_folds,
            nokoi_l1_lambda_min,
            nokoi_l1_lambda_max,
            nokoi_l1_lambda_steps,

            ensemble_pep_combiner,
        }
    }
}
