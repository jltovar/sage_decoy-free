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
    Msfdr,
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

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MsfdrSeedMode {
    /// Default: seed MSFDR null from LO (best sensitivity when multiplicity matters)
    Lo,
    /// Seed MSFDR null from rank-null pool moments fit (more independent, often more conservative)
    PoolMoments,
    /// Seed MSFDR null from rank-null pool MLE fit (more independent, can be conservative/unstable)
    PoolMle,
}

impl Default for MsfdrSeedMode {
    fn default() -> Self {
        MsfdrSeedMode::Lo
    }
}

// ---------------------------------------------------------------------------
// Decoy-free tuning knobs (configuration surface)
// ---------------------------------------------------------------------------

/// How to rank/monotonize LO-derived values.
/// - hyperscore: sort by hyperscore (legacy)
/// - lo_adjusted: sort by LO-adjusted evidence (recommended; fixes LO + PAVA mismatch)
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoRankKey {
    Hyperscore,
    #[default]
    LoAdjusted,
}

/// How to combine p-values in ensemble mode.
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnsemblePCombiner {
    #[default]
    Hmp,
    Fisher,
    Brown,
}

/// How to combine PEPs in ensemble mode.
#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnsemblePepCombiner {
    #[default]
    LogitMean,
    Mean,
    GeometricMean,
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
    pub mode: Option<FdrMode>,
    pub peptide_fdr: Option<f32>,
    pub protein_fdr: Option<f32>,
    pub precursor_fdr: Option<f32>,
    pub min_null_rank: Option<u32>,
    pub max_null_rank: Option<u32>,
    pub model_fit: Option<ModelFit>,
    #[serde(alias = "type")]
    pub type_: Option<FdrType>,

    // Configurable Safety Brakes
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

    // Decoy-Free Lower-Order (LO) robustness controls
    pub lo_multiplicity_alpha: Option<f64>,
    pub lo_ln_ratio_cap: Option<f64>,
    pub lo_beta_blend_moments: Option<f64>,
    pub lo_beta_safety_mult: Option<f64>,

    // Per-method calibration / ranking controls
    pub calibrate_per_method: Option<bool>,
    pub lo_rank_key: Option<LoRankKey>,

    // Null-only PEP strategy (Moments/MLE/LO)
    pub null_only_pep_mode: Option<NullOnlyPepMode>,

    // Ensemble combination choices
    pub ensemble_p_combiner: Option<EnsemblePCombiner>,
    pub ensemble_pep_combiner: Option<EnsemblePepCombiner>,

    // MSFDR controls
    pub msfdr_use_canonical_pep: Option<bool>,
    pub msfdr_multistart: Option<usize>,
    pub msfdr_seed_mode: Option<MsfdrSeedMode>,

    // Rank-null pool construction controls
    pub purification_factor: Option<f64>,
    pub min_rank_count: Option<usize>,
}

#[derive(Clone, Serialize, Debug)]
pub struct FdrSettings {
    pub mode: FdrMode,
    pub peptide_fdr: f32,
    pub protein_fdr: f32,
    pub precursor_fdr: f32,
    pub min_null_rank: u32,
    pub max_null_rank: u32,
    pub model_fit: ModelFit,
    pub type_: FdrType,

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

    // Decoy-Free Lower-Order (LO) robustness controls
    pub lo_multiplicity_alpha: f64,
    pub lo_ln_ratio_cap: f64,
    pub lo_beta_blend_moments: f64,
    pub lo_beta_safety_mult: f64,

    // Per-method calibration / ranking controls
    pub calibrate_per_method: bool,
    pub lo_rank_key: LoRankKey,

    // Null-only PEP strategy (Moments/MLE/LO)
    pub null_only_pep_mode: NullOnlyPepMode,

    // Ensemble combination choices
    pub ensemble_p_combiner: EnsemblePCombiner,
    pub ensemble_pep_combiner: EnsemblePepCombiner,

    // MSFDR controls
    pub msfdr_use_canonical_pep: bool,
    pub msfdr_multistart: usize,
    pub msfdr_seed_mode: MsfdrSeedMode,

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

        let lo_multiplicity_alpha = options
            .lo_multiplicity_alpha
            .unwrap_or(0.50)
            .clamp(0.0, 1.0);

        let lo_ln_ratio_cap = options.lo_ln_ratio_cap.unwrap_or(6.9).max(0.0);

        // Independence default: do NOT blend LO toward Moments unless explicitly requested.
        let lo_beta_blend_moments = options.lo_beta_blend_moments.unwrap_or(0.0).clamp(0.0, 1.0);

        let lo_beta_safety_mult = match options.lo_beta_safety_mult {
            Some(x) if x.is_finite() && x > 0.0 => x.clamp(0.1, 10.0),
            _ => 0.60,
        };

        let calibrate_per_method = options.calibrate_per_method.unwrap_or(true);
        let lo_rank_key = options.lo_rank_key.unwrap_or(LoRankKey::LoAdjusted);

        let null_only_pep_mode = options
            .null_only_pep_mode
            .unwrap_or(NullOnlyPepMode::PepEqualsP);

        let ensemble_p_combiner = options
            .ensemble_p_combiner
            .unwrap_or(EnsemblePCombiner::Hmp);

        let ensemble_pep_combiner = options
            .ensemble_pep_combiner
            .unwrap_or(EnsemblePepCombiner::LogitMean);

        let msfdr_use_canonical_pep = options.msfdr_use_canonical_pep.unwrap_or(true);

        // Keep small/safe; avoids pathological huge multistarts.
        let msfdr_multistart = options.msfdr_multistart.unwrap_or(3).clamp(1, 25);

        let msfdr_seed_mode = options.msfdr_seed_mode.unwrap_or(MsfdrSeedMode::Lo);

        let purification_factor = options.purification_factor.unwrap_or(0.20).clamp(0.0, 0.9);
        let min_rank_count = options.min_rank_count.unwrap_or(10);

        Self {
            mode: options.mode.unwrap_or(FdrMode::Tdc),
            peptide_fdr: options.peptide_fdr.unwrap_or(0.01),
            protein_fdr: options.protein_fdr.unwrap_or(0.01),
            precursor_fdr: options.precursor_fdr.unwrap_or(0.01),
            min_null_rank: options.min_null_rank.unwrap_or(2),
            max_null_rank: options.max_null_rank.unwrap_or(10),
            model_fit: options.model_fit.unwrap_or(ModelFit::Moments),
            type_: options.type_.unwrap_or(FdrType::Bh),

            min_storey_n: options.min_storey_n.unwrap_or(500),
            min_null_size: options.min_null_size.unwrap_or(150),
            kde_samples: options.kde_samples.unwrap_or(20_000),

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

            lo_multiplicity_alpha,
            lo_ln_ratio_cap,
            lo_beta_blend_moments,
            lo_beta_safety_mult,

            calibrate_per_method,
            lo_rank_key,
            null_only_pep_mode,

            ensemble_p_combiner,
            ensemble_pep_combiner,

            msfdr_use_canonical_pep,
            msfdr_multistart,
            msfdr_seed_mode,

            purification_factor,
            min_rank_count,
        }
    }
}
