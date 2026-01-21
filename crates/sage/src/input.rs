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
    EnsembleDebug,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FdrType {
    #[default]
    Bh, // Benjamini-Hochberg
    Storey, // Storey-Tibshirani
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct FdrOptions {
    pub mode: Option<FdrMode>,
    pub peptide_fdr: Option<f32>,
    pub protein_fdr: Option<f32>,
    pub precursor_fdr: Option<f32>,
    pub min_null_rank: Option<u32>,
    pub max_null_rank: Option<u32>,
    pub model_fit: Option<ModelFit>,
    pub type_: Option<FdrType>,

    // Configurable Safety Brakes
    pub min_storey_n: Option<usize>,
    pub min_null_size: Option<usize>,
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
}

impl From<FdrOptions> for FdrSettings {
    fn from(options: FdrOptions) -> Self {
        Self {
            mode: options.mode.unwrap_or(FdrMode::Tdc),
            peptide_fdr: options.peptide_fdr.unwrap_or(0.01),
            protein_fdr: options.protein_fdr.unwrap_or(0.01),
            precursor_fdr: options.precursor_fdr.unwrap_or(0.01),
            min_null_rank: options.min_null_rank.unwrap_or(2),
            max_null_rank: options.max_null_rank.unwrap_or(10),
            model_fit: options.model_fit.unwrap_or(ModelFit::Moments),
            type_: options.type_.unwrap_or(FdrType::Bh),

            // Apply Defaults Here (500 and 150)
            min_storey_n: options.min_storey_n.unwrap_or(500),
            min_null_size: options.min_null_size.unwrap_or(150),
        }
    }
}