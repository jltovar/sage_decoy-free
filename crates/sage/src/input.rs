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

    // Decoy-Free Lower-Order (LO) robustness controls
    pub lo_multiplicity_alpha: Option<f64>,
    pub lo_ln_ratio_cap: Option<f64>,
    pub lo_beta_blend_moments: Option<f64>,
    pub lo_beta_safety_mult: Option<f64>,

    // New Controls
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

    pub min_storey_n: usize,
    pub min_null_size: usize,
    pub kde_samples: usize,

    pub lo_multiplicity_alpha: f64,
    pub lo_ln_ratio_cap: f64,
    pub lo_beta_blend_moments: f64,
    pub lo_beta_safety_mult: f64,

    pub purification_factor: f64,
    pub min_rank_count: usize,
}

impl From<FdrOptions> for FdrSettings {
    fn from(options: FdrOptions) -> Self {
        let lo_multiplicity_alpha = options
            .lo_multiplicity_alpha
            .unwrap_or(0.50)
            .clamp(0.0, 1.0);
        let lo_ln_ratio_cap = options.lo_ln_ratio_cap.unwrap_or(6.9).max(0.0);
        let lo_beta_blend_moments = options
            .lo_beta_blend_moments
            .unwrap_or(0.30)
            .clamp(0.0, 1.0);

        let lo_beta_safety_mult = match options.lo_beta_safety_mult {
            Some(x) if x.is_finite() && x > 0.0 => x.clamp(0.1, 10.0),
            _ => 0.60,
        };

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
            lo_multiplicity_alpha,
            lo_ln_ratio_cap,
            lo_beta_blend_moments,
            lo_beta_safety_mult,
            purification_factor,
            min_rank_count,
        }
    }
}
