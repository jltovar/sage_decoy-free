//! Deterministic, dataset-local Decoy-Free parameter optimization.
//!
//! This module owns only analysis parameters. Search inputs and raw external
//! predictions are immutable inputs identified by their content fingerprints.
//! The optimizer is development-only and never consumes target-only outcomes.

use crate::provenance::write_json_atomic;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const PARAMETER_OPTIMIZER_SCHEMA_VERSION: u32 = 4;
pub const PARAMETER_CATALOG_SCHEMA_VERSION: u32 = 2;
pub const PARAMETER_OPTIMIZER_IMPLEMENTATION_SOURCE_SHA256: &str =
    env!("SAGE_PARAMETER_OPTIMIZER_SOURCE_SHA256");

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ParameterScope {
    Default,
    PerExpert,
    EnsembleFinal,
    Physical,
    Reproducibility,
    HierarchicalOrReporting,
    NumericalOnly,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParameterClass {
    ScientificOptimizationCandidate,
    StructuralMethodFamilyChoice,
    NumericalConvergenceOrPrecision,
    FixedReportingThreshold,
    ValidationReportingOrProvenanceOnly,
    UnsafeOrUnsupported,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParameterKind {
    Boolean,
    Integer,
    Float,
    Enumeration,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerExpert {
    Moments,
    Mle,
    LowerOrder,
    MsfdrSeeded,
    Msfdr1Smix,
    Msfdr2Smix,
    Nokoi,
    Ensemble,
}

impl OptimizerExpert {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Moments => "moments",
            Self::Mle => "mle",
            Self::LowerOrder => "lower_order",
            Self::MsfdrSeeded => "msfdr_seeded",
            Self::Msfdr1Smix => "msfdr_1smix",
            Self::Msfdr2Smix => "msfdr_2smix",
            Self::Nokoi => "nokoi",
            Self::Ensemble => "ensemble",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParameterContract {
    pub name: &'static str,
    pub owner: &'static str,
    pub scopes: &'static [ParameterScope],
    pub kind: ParameterKind,
    pub class: ParameterClass,
    pub json_exposed: bool,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub enum_values: &'static [&'static str],
    pub affects_identifications: bool,
    pub eligible: bool,
    pub validity_contract_required: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProductionBindingStatus {
    Executable,
    ConditionallyExecutable,
    DeliberatelyDeferred,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParameterProductionBinding {
    pub canonical_name: String,
    pub supported_scopes: Vec<ParameterScope>,
    pub setter_path: String,
    pub production_structure: String,
    pub dependency_predicate: String,
    pub status: ProductionBindingStatus,
    pub currently_executable: bool,
    pub conditionally_executable: bool,
    pub deliberately_deferred: bool,
    pub reason: String,
}

const PER_EXPERT: &[ParameterScope] = &[ParameterScope::PerExpert];
const PER_EXPERT_AND_FINAL: &[ParameterScope] =
    &[ParameterScope::PerExpert, ParameterScope::EnsembleFinal];
const ENSEMBLE_FINAL: &[ParameterScope] = &[ParameterScope::EnsembleFinal];
const PHYSICAL: &[ParameterScope] = &[ParameterScope::Physical];
const REPRODUCIBILITY: &[ParameterScope] = &[ParameterScope::Reproducibility];
const HIERARCHICAL: &[ParameterScope] = &[ParameterScope::HierarchicalOrReporting];
const NUMERICAL: &[ParameterScope] = &[ParameterScope::NumericalOnly];
const DEFAULT_ONLY: &[ParameterScope] = &[ParameterScope::Default];

macro_rules! contract {
    ($name:literal,$owner:literal,$scopes:expr,$kind:ident,$class:ident,$min:expr,$max:expr,$enums:expr,$affects:expr,$eligible:expr,$validity:expr) => {
        ParameterContract {
            name: $name,
            owner: $owner,
            scopes: $scopes,
            kind: ParameterKind::$kind,
            class: ParameterClass::$class,
            json_exposed: true,
            minimum: $min,
            maximum: $max,
            enum_values: $enums,
            affects_identifications: $affects,
            eligible: $eligible,
            validity_contract_required: $validity,
        }
    };
}

/// Runtime validation catalog. The more detailed audit inventory is kept in
/// `validation/statistical_conformance/parameter_catalog.json`; this registry
/// intentionally contains the same canonical names without becoming a runtime
/// filesystem dependency.
pub fn parameter_contracts() -> Vec<ParameterContract> {
    vec![
        // Shared evidence, calibration, and aggregation. These may be resolved
        // independently for an expert or for the final Ensemble stream.
        contract!(
            "final_evidence_space",
            "evidence",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["auto", "p_value", "pep"],
            true,
            true,
            false
        ),
        contract!(
            "peptide_p_combine",
            "aggregation",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[
                "fisher",
                "cauchy",
                "acat",
                "sidak_min_p",
                "bonferroni_min_p",
                "tippett",
                "best",
                "second_best",
                "hmp",
                "brown",
                "mudholkar_george",
                "edgington",
                "t_fisher",
                "g_fisher",
                "ihw",
                "exchangeable_e_value",
                "vovk_wang_generalized_mean",
                "ordmeta_w_fisher",
                "mcm",
                "cmc"
            ],
            true,
            true,
            true
        ),
        contract!(
            "protein_p_combine",
            "aggregation",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[
                "fisher",
                "cauchy",
                "acat",
                "sidak_min_p",
                "bonferroni_min_p",
                "tippett",
                "best",
                "second_best",
                "hmp",
                "brown",
                "mudholkar_george",
                "edgington",
                "t_fisher",
                "g_fisher",
                "ihw",
                "exchangeable_e_value",
                "vovk_wang_generalized_mean",
                "ordmeta_w_fisher",
                "mcm",
                "cmc"
            ],
            true,
            true,
            true
        ),
        contract!(
            "p_combine_calibration_mode",
            "aggregation",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["off", "rank_null"],
            true,
            true,
            true
        ),
        contract!(
            "p_combine_calibration_min_k",
            "aggregation",
            PER_EXPERT_AND_FINAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            Some(100.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "p_combine_calibration_max_k",
            "aggregation",
            PER_EXPERT_AND_FINAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            Some(100.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "p_combine_calibration_null_replicates",
            "aggregation",
            NUMERICAL,
            Integer,
            NumericalConvergenceOrPrecision,
            Some(100.0),
            Some(100000.0),
            &[],
            true,
            false,
            false
        ),
        contract!(
            "p_combine_tfisher_tau",
            "aggregation",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(1e-12),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "psm_q_method",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[
                "auto",
                "bh",
                "storey",
                "by",
                "bky",
                "sfdr",
                "covariate_weighted_bh",
                "cummean"
            ],
            true,
            true,
            false
        ),
        contract!(
            "peptide_q_method",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[
                "auto",
                "bh",
                "storey",
                "by",
                "bky",
                "sfdr",
                "covariate_weighted_bh",
                "cummean"
            ],
            true,
            true,
            false
        ),
        contract!(
            "protein_q_method",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[
                "auto",
                "bh",
                "storey",
                "by",
                "bky",
                "sfdr",
                "covariate_weighted_bh",
                "cummean"
            ],
            true,
            true,
            false
        ),
        contract!(
            "bky_alpha",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(1e-6),
            Some(0.5),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "sfdr_gamma",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.1),
            Some(3.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "psm_q_covariate",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[
                "none",
                "hyperscore",
                "delta_next",
                "delta_best",
                "matched_peaks",
                "longest_b",
                "longest_y",
                "longest_y_pct",
                "matched_intensity_pct",
                "scored_candidates",
                "ms2_intensity",
                "peptide_len",
                "charge",
                "missed_cleavages",
                "best_matched_peaks",
                "best_longest_y_pct",
                "best_delta_rt_model",
                "best_hyperscore",
                "psm_count",
                "peptide_observed_run_count",
                "observed_unique_peptides",
                "observed_peptide_support",
                "protein_length",
                "observable_protein_peptides",
                "nsaf_observable_length"
            ],
            true,
            true,
            true
        ),
        contract!(
            "peptide_q_covariate",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[
                "none",
                "hyperscore",
                "delta_next",
                "delta_best",
                "matched_peaks",
                "longest_b",
                "longest_y",
                "longest_y_pct",
                "matched_intensity_pct",
                "scored_candidates",
                "ms2_intensity",
                "peptide_len",
                "charge",
                "missed_cleavages",
                "best_matched_peaks",
                "best_longest_y_pct",
                "best_delta_rt_model",
                "best_hyperscore",
                "psm_count",
                "peptide_observed_run_count",
                "observed_unique_peptides",
                "observed_peptide_support",
                "protein_length",
                "observable_protein_peptides",
                "nsaf_observable_length"
            ],
            true,
            true,
            true
        ),
        contract!(
            "protein_q_covariate",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[
                "none",
                "hyperscore",
                "delta_next",
                "delta_best",
                "matched_peaks",
                "longest_b",
                "longest_y",
                "longest_y_pct",
                "matched_intensity_pct",
                "scored_candidates",
                "ms2_intensity",
                "peptide_len",
                "charge",
                "missed_cleavages",
                "best_matched_peaks",
                "best_longest_y_pct",
                "best_delta_rt_model",
                "best_hyperscore",
                "psm_count",
                "peptide_observed_run_count",
                "observed_unique_peptides",
                "observed_peptide_support",
                "protein_length",
                "observable_protein_peptides",
                "nsaf_observable_length"
            ],
            true,
            true,
            true
        ),
        contract!(
            "psm_q_covariate_bins",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            Some(20.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "peptide_q_covariate_bins",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            Some(20.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "protein_q_covariate_bins",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            Some(20.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "psm_q_covariate_weight_strength",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(5.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "peptide_q_covariate_weight_strength",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(5.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "protein_q_covariate_weight_strength",
            "q_value",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(5.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "decoy_free_protein_grouping",
            "aggregation",
            PER_EXPERT_AND_FINAL,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "report_psms_by_peptide_q",
            "reporting",
            HIERARCHICAL,
            Boolean,
            ValidationReportingOrProvenanceOnly,
            None,
            None,
            &[],
            false,
            false,
            false
        ),
        contract!(
            "precursor_fdr",
            "reporting",
            DEFAULT_ONLY,
            Float,
            FixedReportingThreshold,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            false,
            false
        ),
        contract!(
            "peptide_fdr",
            "reporting",
            DEFAULT_ONLY,
            Float,
            FixedReportingThreshold,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            false,
            false
        ),
        contract!(
            "protein_fdr",
            "reporting",
            DEFAULT_ONLY,
            Float,
            FixedReportingThreshold,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            false,
            false
        ),
        // Null-pool and Storey controls are independently resolvable for each expert.
        contract!(
            "min_null_size",
            "shared_null",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "min_rank_count",
            "shared_null",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "min_storey_n",
            "storey",
            PER_EXPERT_AND_FINAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "storey_pi0_clamp_min",
            "storey",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "storey_pi0_clamp_max",
            "storey",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "storey_lambda_min",
            "storey",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(0.99),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "storey_lambda_max",
            "storey",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.01),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "storey_lambda_step",
            "storey",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(1e-6),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "storey_lambda_min_for_agg",
            "storey",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "storey_pi0_agg",
            "storey",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["median", "trimmed_mean"],
            true,
            true,
            false
        ),
        contract!(
            "storey_degen_same_as_median_frac",
            "storey",
            PER_EXPERT_AND_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "storey_degen_eps",
            "storey",
            NUMERICAL,
            Float,
            NumericalConvergenceOrPrecision,
            Some(0.0),
            None,
            &[],
            true,
            false,
            false
        ),
        contract!(
            "storey_degen_pi0_eps",
            "storey",
            NUMERICAL,
            Float,
            NumericalConvergenceOrPrecision,
            Some(0.0),
            None,
            &[],
            true,
            false,
            false
        ),
        contract!(
            "storey_degen_fallback",
            "storey",
            PER_EXPERT_AND_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["bh", "none"],
            true,
            true,
            false
        ),
        // Expert-local model controls.
        contract!(
            "moments_min_null_rank",
            "moments",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "moments_max_null_rank",
            "moments",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "moments_purification_factor",
            "moments",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(0.9),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "moments_robust_fit",
            "moments",
            PER_EXPERT,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "moments_winsor_lower_q",
            "moments",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "moments_winsor_upper_q",
            "moments",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "mle_min_null_rank",
            "mle",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "mle_max_null_rank",
            "mle",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "mle_purification_factor",
            "mle",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(0.9),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "mle_robust_fit",
            "mle",
            PER_EXPERT,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "mle_winsor_lower_q",
            "mle",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "mle_winsor_upper_q",
            "mle",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "lower_order_min_null_rank",
            "lower_order",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "lower_order_max_null_rank",
            "lower_order",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "lower_order_purification_factor",
            "lower_order",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(0.9),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "lo_min_count_per_rank",
            "lower_order",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "lo_stratify",
            "lower_order",
            PER_EXPERT,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["global", "charge"],
            true,
            true,
            false
        ),
        contract!(
            "lo_evalue_candidate_count_power",
            "lower_order",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "lo_evalue_scale",
            "lower_order",
            PER_EXPERT,
            Float,
            UnsafeOrUnsupported,
            Some(1e-6),
            Some(1e6),
            &[],
            false,
            false,
            false
        ),
        contract!(
            "lo_tev_transform",
            "lower_order",
            PER_EXPERT,
            Enumeration,
            UnsafeOrUnsupported,
            None,
            None,
            &[
                "neg_log_e",
                "log1000_over_e",
                "scaled_log1000_over_e",
                "log_1000_over_e",
                "scaled_log_1000_over_e"
            ],
            false,
            false,
            false
        ),
        contract!(
            "lo_tnm_extrapolation_strength",
            "lower_order",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.25),
            Some(5.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr_min_null_rank",
            "msfdr_seeded",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr_max_null_rank",
            "msfdr_seeded",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr_seeded_purification_factor",
            "msfdr_seeded",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(0.9),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr_seeded_top_frac_init",
            "msfdr_seeded",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(1e-6),
            Some(0.999999),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr_multistart",
            "msfdr_seeded",
            PER_EXPERT,
            Integer,
            UnsafeOrUnsupported,
            Some(1.0),
            Some(25.0),
            &[],
            false,
            false,
            false
        ),
        contract!(
            "msfdr_pi_clamp_min",
            "msfdr_seeded",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr_pi_clamp_max",
            "msfdr_seeded",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "mix_em_max_iter",
            "msfdr_mixtures",
            NUMERICAL,
            Integer,
            NumericalConvergenceOrPrecision,
            Some(1.0),
            Some(10000.0),
            &[],
            true,
            false,
            false
        ),
        contract!(
            "mix_em_tol",
            "msfdr_mixtures",
            NUMERICAL,
            Float,
            NumericalConvergenceOrPrecision,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            false,
            false
        ),
        contract!(
            "msfdr1_bottom_frac_init",
            "msfdr_1smix",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(1e-6),
            Some(0.999999),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr1_top_frac_init",
            "msfdr_1smix",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(1e-6),
            Some(0.999999),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr1_pi_clamp_min",
            "msfdr_1smix",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr1_pi_clamp_max",
            "msfdr_1smix",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr2_smix_min_null_rank",
            "msfdr_2smix",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr2_smix_max_null_rank",
            "msfdr_2smix",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr2_bottom_frac_init",
            "msfdr_2smix",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(1e-6),
            Some(0.999999),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr2_top_frac_init",
            "msfdr_2smix",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(1e-6),
            Some(0.999999),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr2_pi_clamp_min",
            "msfdr_2smix",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "msfdr2_pi_clamp_max",
            "msfdr_2smix",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "nokoi_min_null_rank",
            "nokoi",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "nokoi_max_null_rank",
            "nokoi",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "nokoi_null_purification_factor",
            "nokoi",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(0.9),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "nokoi_positive_top_fraction",
            "nokoi",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(0.9),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "nokoi_k_folds",
            "nokoi",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(2.0),
            Some(20.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "nokoi_l1_lambda_min",
            "nokoi",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "nokoi_l1_lambda_max",
            "nokoi",
            PER_EXPERT,
            Float,
            ScientificOptimizationCandidate,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "nokoi_l1_lambda_steps",
            "nokoi",
            PER_EXPERT,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            false
        ),
        // Final Ensemble-only continuous combination. Participation is never optimized here.
        contract!(
            "ensemble_p_combiner",
            "ensemble",
            ENSEMBLE_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["fisher", "cauchy", "sidak_min_p", "best", "second_best"],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_cauchy_penalty",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(1.0),
            Some(100.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_pep_combiner",
            "ensemble",
            ENSEMBLE_FINAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[
                "median",
                "trimmed_mean",
                "max",
                "mean",
                "weighted_mean",
                "weighted_median",
                "winsorized_mean",
                "quantile",
                "top_k_mean",
                "geometric_mean",
                "logit_mean"
            ],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_pep_trim_frac",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(0.49),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_pep_quantile",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_pep_top_k",
            "ensemble",
            ENSEMBLE_FINAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_pep_logit_eps",
            "ensemble",
            NUMERICAL,
            Float,
            NumericalConvergenceOrPrecision,
            Some(1e-12),
            Some(1e-2),
            &[],
            true,
            false,
            false
        ),
        contract!(
            "ensemble_weight_moments",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_weight_mle",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_weight_lower_order",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_weight_msfdr_seeded",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_weight_msfdr_1smix",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_weight_msfdr_2smix",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "ensemble_weight_nokoi",
            "ensemble",
            ENSEMBLE_FINAL,
            Float,
            ScientificOptimizationCandidate,
            Some(f64::MIN_POSITIVE),
            None,
            &[],
            true,
            true,
            false
        ),
        // Later conditional blocks. Nested values use dotted canonical names;
        // they are validated by this optimizer but materialized by the workflow adapter.
        contract!(
            "enable_rt_confidence_adjustment",
            "physical",
            PHYSICAL,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "enable_ims_confidence_adjustment",
            "physical",
            PHYSICAL,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "physical_rescue.rt_mode",
            "physical",
            PHYSICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["off", "dart_bayes", "bounded_aux"],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.ims_mode",
            "physical",
            PHYSICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["off", "dart_bayes", "bounded_aux"],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.anchor_mode",
            "physical",
            PHYSICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["strict", "default", "relaxed", "evidence_only"],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.anchor_max_pep",
            "physical",
            PHYSICAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.anchor_max_q",
            "physical",
            PHYSICAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.min_anchor_count_per_run",
            "physical",
            PHYSICAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.min_anchor_count_per_charge",
            "physical",
            PHYSICAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.joint_mode",
            "physical",
            PHYSICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["min", "product", "independent"],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.reliability_floor",
            "physical",
            PHYSICAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.missing_penalty",
            "physical",
            PHYSICAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.rt_region_bins",
            "physical",
            PHYSICAL,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.use_local_rt_scale",
            "physical",
            PHYSICAL,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.cov_shrinkage",
            "physical",
            PHYSICAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.dart_cfg.dart_use_bootstrap",
            "physical",
            PHYSICAL,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "physical_rescue.dart_cfg.dart_bootstrap_method",
            "physical",
            PHYSICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["none", "parametric", "parametric_mixture", "non_parametric"],
            true,
            true,
            true
        ),
        contract!(
            "physical_rescue.dart_cfg.dart_mu_estimation",
            "physical",
            PHYSICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["mean", "median", "weighted_mean"],
            true,
            true,
            true
        ),
        contract!(
            "physical_rescue.dart_cfg.dart_bootstrap_iters",
            "physical",
            NUMERICAL,
            Integer,
            NumericalConvergenceOrPrecision,
            Some(1.0),
            None,
            &[],
            true,
            false,
            false
        ),
        contract!(
            "physical_rescue.dart_cfg.dart_leave_one_run_out",
            "physical",
            PHYSICAL,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "physical_rescue.dart_cfg.dart_null_rt_model",
            "physical",
            PHYSICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["normal", "uniform"],
            true,
            true,
            true
        ),
        contract!(
            "physical_rescue.dart_cfg.dart_true_rt_model",
            "physical",
            PHYSICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["normal", "laplace"],
            true,
            true,
            true
        ),
        contract!(
            "physical_rescue.dart_cfg.dart_recalc_q_from_posterior",
            "physical",
            PHYSICAL,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "physical_rescue.bounded_cfg.update_space",
            "physical",
            PHYSICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["logit_confidence"],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.bounded_cfg.max_rescue_shift",
            "physical",
            PHYSICAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "physical_rescue.bounded_cfg.max_penalty_shift",
            "physical",
            PHYSICAL,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "reproducibility.enabled",
            "reproducibility",
            REPRODUCIBILITY,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "enable_peptide_reproducibility_rescue",
            "reproducibility",
            REPRODUCIBILITY,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "enable_protein_reproducibility_rescue",
            "reproducibility",
            REPRODUCIBILITY,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.max_total_shift",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "reproducibility.max_agreement_shift",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "reproducibility.max_recurrence_shift",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            None,
            &[],
            true,
            true,
            false
        ),
        contract!(
            "reproducibility.use_expert_agreement",
            "reproducibility",
            REPRODUCIBILITY,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.use_cross_run_recurrence",
            "reproducibility",
            REPRODUCIBILITY,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.redundancy_discount",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            false
        ),
        contract!(
            "reproducibility.protein_eligibility.enabled",
            "reproducibility",
            REPRODUCIBILITY,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.protein_eligibility.q_threshold_physical",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.protein_eligibility.min_unique_passing_peptides",
            "reproducibility",
            REPRODUCIBILITY,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.protein_eligibility.min_unique_passing_fraction",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.peptide_eligibility.min_run_fraction",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.peptide_eligibility.min_run_count",
            "reproducibility",
            REPRODUCIBILITY,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.peptide_eligibility.strong_reference_q_threshold_physical",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.peptide_eligibility.strong_reference_pep_threshold_physical",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.peptide_eligibility.min_strong_run_fraction",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.peptide_eligibility.min_strong_run_count",
            "reproducibility",
            REPRODUCIBILITY,
            Integer,
            ScientificOptimizationCandidate,
            Some(1.0),
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.anchor.mode",
            "reproducibility",
            REPRODUCIBILITY,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["best", "second_best", "mean", "median", "trimmed_mean"],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.anchor.trim_fraction",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(0.49),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.rescue_band.strong_cutoff_pep",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.rescue_band.weak_cutoff_pep",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.rescue_band.max_rescue_fraction",
            "reproducibility",
            REPRODUCIBILITY,
            Float,
            ScientificOptimizationCandidate,
            Some(0.0),
            Some(1.0),
            &[],
            true,
            true,
            true
        ),
        contract!(
            "reproducibility.rescue_band.rescue_mode",
            "reproducibility",
            REPRODUCIBILITY,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["replace", "bounded_shrinkage"],
            true,
            true,
            true
        ),
        contract!(
            "hierarchical_inference.enabled",
            "hierarchical",
            HIERARCHICAL,
            Boolean,
            StructuralMethodFamilyChoice,
            None,
            None,
            &[],
            true,
            true,
            true
        ),
        contract!(
            "hierarchical_inference.mode",
            "hierarchical",
            HIERARCHICAL,
            Enumeration,
            StructuralMethodFamilyChoice,
            None,
            None,
            &["off", "protein_anchored"],
            true,
            true,
            true
        ),
        contract!(
            "hierarchical_inference.entrapment_validation",
            "validation",
            HIERARCHICAL,
            Boolean,
            ValidationReportingOrProvenanceOnly,
            None,
            None,
            &[],
            false,
            false,
            false
        ),
    ]
}

fn dependency_predicate(name: &str, class: ParameterClass, validity_required: bool) -> String {
    let predicate = match name {
        "moments_winsor_lower_q" | "moments_winsor_upper_q" => {
            "moments_robust_fit=true and lower_q<=upper_q"
        }
        "mle_winsor_lower_q" | "mle_winsor_upper_q" => {
            "mle_robust_fit=true and lower_q<=upper_q"
        }
        "p_combine_calibration_min_k" | "p_combine_calibration_max_k" => {
            "p_combine_calibration_mode=rank_null and min_k<=max_k"
        }
        "p_combine_tfisher_tau" => "selected peptide/protein combiner uses truncated Fisher",
        "bky_alpha" => "applicable q-value method=bky",
        "sfdr_gamma" => "applicable q-value method=sfdr",
        "psm_q_covariate_bins"
        | "peptide_q_covariate_bins"
        | "protein_q_covariate_bins"
        | "psm_q_covariate_weight_strength"
        | "peptide_q_covariate_weight_strength"
        | "protein_q_covariate_weight_strength" => {
            "applicable q-value method=covariate_weighted_bh and a validated covariate contract is present"
        }
        "ensemble_p_combiner" => "final_evidence_space=p_value",
        "ensemble_cauchy_penalty" => {
            "final_evidence_space=p_value and ensemble_p_combiner=cauchy"
        }
        "ensemble_pep_combiner" => "final_evidence_space=pep",
        "ensemble_pep_trim_frac" => {
            "final_evidence_space=pep and ensemble_pep_combiner is trimmed_mean or winsorized_mean"
        }
        "ensemble_pep_quantile" => {
            "final_evidence_space=pep and ensemble_pep_combiner=quantile"
        }
        "ensemble_pep_top_k" => {
            "final_evidence_space=pep, ensemble_pep_combiner=top_k_mean, and k<=selected voter count"
        }
        name if name.starts_with("ensemble_weight_") => {
            "final_evidence_space=pep, ensemble_pep_combiner is weighted_mean or weighted_median, named expert is JSON-selected, and effective weight is finite and >0"
        }
        name if name.starts_with("physical_rescue.") => {
            "deferred pending instrument/run metadata and a separately reviewed physical-evidence block"
        }
        name if name.starts_with("reproducibility.") => {
            "deferred pending run/group/sample-role boundaries and a separately reviewed recurrence block"
        }
        name if name.starts_with("hierarchical_inference.") => {
            "deferred pending a separately reviewed hierarchical-inference block"
        }
        _ if validity_required => "named statistical-validity contract is present and validated",
        _ if class == ParameterClass::StructuralMethodFamilyChoice => {
            "block declares structural_comparison=true"
        }
        _ => "runtime value satisfies its declared type, domain, and cross-parameter dependencies",
    };
    predicate.into()
}

/// One machine-verifiable production binding for every runtime catalog entry.
/// Direct executable settings are materialized into FdrOptions and resolved
/// by the normal FdrSettings::from production path. Conditional later-stage
/// families remain explicit but fail closed when placed in an enabled block.
pub fn parameter_production_bindings() -> Vec<ParameterProductionBinding> {
    let mut bindings = parameter_contracts()
        .into_iter()
        .map(|contract| {
            let deferred_scope = contract.scopes.iter().any(|scope| {
                matches!(
                    scope,
                    ParameterScope::Physical
                        | ParameterScope::Reproducibility
                        | ParameterScope::HierarchicalOrReporting
                        | ParameterScope::NumericalOnly
                        | ParameterScope::Default
                )
            });
            let deferred_class = matches!(
                contract.class,
                ParameterClass::NumericalConvergenceOrPrecision
                    | ParameterClass::FixedReportingThreshold
                    | ParameterClass::ValidationReportingOrProvenanceOnly
                    | ParameterClass::UnsafeOrUnsupported
            );
            let predicate = dependency_predicate(
                contract.name,
                contract.class,
                contract.validity_contract_required,
            );
            let status = if !contract.eligible || deferred_scope || deferred_class {
                ProductionBindingStatus::DeliberatelyDeferred
            } else if contract.validity_contract_required
                || contract.class == ParameterClass::StructuralMethodFamilyChoice
                || predicate
                    != "runtime value satisfies its declared type, domain, and cross-parameter dependencies"
            {
                ProductionBindingStatus::ConditionallyExecutable
            } else {
                ProductionBindingStatus::Executable
            };
            let direct = status != ProductionBindingStatus::DeliberatelyDeferred;
            let reason = match status {
                ProductionBindingStatus::Executable => {
                    "bound directly to the production Decoy-Free FdrOptions/FdrSettings path"
                }
                ProductionBindingStatus::ConditionallyExecutable => {
                    "bound to production and enabled only when the declared dependency/validity predicate holds"
                }
                ProductionBindingStatus::DeliberatelyDeferred => {
                    "not eligible in an active yield-optimization block under the current Step-3 contract"
                }
            };
            ParameterProductionBinding {
                canonical_name: contract.name.into(),
                supported_scopes: contract.scopes.to_vec(),
                setter_path: if direct {
                    "apply_fdr_overrides -> run_search_stage(parameter_optimizer_trial)".into()
                } else {
                    "none_active".into()
                },
                production_structure: if direct {
                    format!("FdrOptions.{} -> FdrSettings.{}", contract.name, contract.name)
                } else {
                    format!("deferred:{}", contract.name)
                },
                dependency_predicate: predicate,
                status,
                currently_executable: direct,
                conditionally_executable: status
                    == ProductionBindingStatus::ConditionallyExecutable,
                deliberately_deferred: status == ProductionBindingStatus::DeliberatelyDeferred,
                reason: reason.into(),
            }
        })
        .collect::<Vec<_>>();
    for (name, fields, predicate) in [
        (
            "moments_null_window",
            "FdrOptions.moments_min_null_rank/moments_max_null_rank",
            "2<=min<=max<=retained rank",
        ),
        (
            "mle_null_window",
            "FdrOptions.mle_min_null_rank/mle_max_null_rank",
            "2<=min<=max<=retained rank",
        ),
        (
            "lower_order_null_window",
            "FdrOptions.lower_order_min_null_rank/lower_order_max_null_rank",
            "at least two supported lower-order ranks",
        ),
        (
            "msfdr_seeded_null_window",
            "FdrOptions.msfdr_min_null_rank/msfdr_max_null_rank",
            "2<=min<=max<=retained rank and valid mixture fit",
        ),
        (
            "msfdr2_smix_null_window",
            "FdrOptions.msfdr2_smix_min_null_rank/msfdr2_smix_max_null_rank",
            "2<=min<=max<=retained rank and valid identifiable mixture fit",
        ),
        (
            "nokoi_null_window",
            "FdrOptions.nokoi_min_null_rank/nokoi_max_null_rank",
            "2<=min<=max<=retained rank with sufficient OOF fold support",
        ),
    ] {
        bindings.push(ParameterProductionBinding {
            canonical_name: name.into(),
            supported_scopes: vec![ParameterScope::PerExpert],
            setter_path:
                "OptimizerBlock.window_search -> apply_optimizer_window -> null_window_optimizer"
                    .into(),
            production_structure: fields.into(),
            dependency_predicate: predicate.into(),
            status: ProductionBindingStatus::ConditionallyExecutable,
            currently_executable: true,
            conditionally_executable: true,
            deliberately_deferred: false,
            reason: "bound to the existing model-local production null-window machinery".into(),
        });
    }
    bindings
}

fn production_binding(name: &str) -> Result<ParameterProductionBinding> {
    parameter_production_bindings()
        .into_iter()
        .find(|binding| binding.canonical_name == name)
        .with_context(|| format!("parameter {name} has no production binding coverage record"))
}

/// Stable fingerprint of the embedded runtime catalog. Only portable contract
/// fields are hashed; repository paths and build-host data never participate.
pub fn parameter_catalog_fingerprint() -> Result<String> {
    let portable = parameter_contracts()
        .into_iter()
        .map(|contract| {
            serde_json::json!({
                "name": contract.name,
                "owner": contract.owner,
                "scopes": contract.scopes,
                "kind": contract.kind,
                "class": contract.class,
                "json_exposed": contract.json_exposed,
                "minimum": contract.minimum,
                "maximum": contract.maximum,
                "enum_values": contract.enum_values,
                "affects_identifications": contract.affects_identifications,
                "eligible": contract.eligible,
                "validity_contract_required": contract.validity_contract_required,
                "production_binding": production_binding(contract.name)
                    .expect("every runtime contract has binding coverage"),
            })
        })
        .collect::<Vec<_>>();
    let mut hasher = Sha256::new();
    hasher.update(b"sage-parameter-catalog-v2\0");
    hasher.update(serde_json::to_vec(&portable)?);
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ParameterValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

impl ParameterValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(value) => Some(*value as f64),
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Bool(value) => (*value).into(),
            Self::Integer(value) => (*value).into(),
            Self::Float(value) => (*value).into(),
            Self::String(value) => value.clone().into(),
        }
    }

    fn canonical_key(&self) -> String {
        serde_json::to_string(&self.to_json()).unwrap_or_default()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerStrategy {
    ExhaustiveGrid,
    StagedCoordinate,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationClassification {
    DevelopmentOnly,
    Holdout,
    Release,
    StatisticalDefault,
    ProductionDefault,
}

/// Controls whether an underpowered empirical entrapment estimate blocks a
/// technically valid trial from development-only ranking. The default is the
/// historical fail-closed behavior.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnderpoweredTrialPolicy {
    #[default]
    NotEvaluable,
    DevelopmentEligible,
}

/// Controls whether a completed production optimization exits with its
/// +entrapment winners or continues into the historical post-selection and
/// target-only workflow. The default preserves schema-v1 manifest behavior.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerExecutionMode {
    OptimizationOnly,
    #[default]
    OptimizationAndPostSelection,
}

/// Determines whether entrapment labels are used as one development
/// population or prospectively separated into optimizer-selection and
/// post-freeze audit populations. This is independent of the cross-dataset
/// validation role.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntrapmentValidationMode {
    #[default]
    FullPopulationDevelopment,
    SelectionAudit,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EntrapmentValidationConfig {
    #[serde(default)]
    pub mode: EntrapmentValidationMode,
    #[serde(default = "default_entrapment_partition_schema_version")]
    pub partition_schema_version: u32,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub salt: String,
    #[serde(default = "default_selection_fraction")]
    pub selection_fraction: f64,
    #[serde(default = "default_audit_fraction")]
    pub audit_fraction: f64,
    #[serde(default)]
    pub require_existing_partition: bool,
}

impl Default for EntrapmentValidationConfig {
    fn default() -> Self {
        Self {
            mode: EntrapmentValidationMode::FullPopulationDevelopment,
            partition_schema_version: default_entrapment_partition_schema_version(),
            seed: 0,
            salt: String::new(),
            selection_fraction: default_selection_fraction(),
            audit_fraction: default_audit_fraction(),
            require_existing_partition: false,
        }
    }
}

fn default_entrapment_partition_schema_version() -> u32 {
    1
}

fn default_selection_fraction() -> f64 {
    0.5
}

fn default_audit_fraction() -> f64 {
    0.5
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveMetric {
    Level4Proteins,
    Level4CanonicalPeptides,
    Level4Peptidoforms,
    Level4Psms,
    AdjustedEntrapmentFdp,
    ModelComplexity,
    DeterministicParameterOrder,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObjectiveTerm {
    pub metric: ObjectiveMetric,
    pub direction: ObjectiveDirection,
}

pub fn default_objective() -> Vec<ObjectiveTerm> {
    use ObjectiveDirection::{Maximize, Minimize};
    use ObjectiveMetric::*;
    vec![
        ObjectiveTerm {
            metric: Level4Proteins,
            direction: Maximize,
        },
        ObjectiveTerm {
            metric: Level4CanonicalPeptides,
            direction: Maximize,
        },
        ObjectiveTerm {
            metric: Level4Peptidoforms,
            direction: Maximize,
        },
        ObjectiveTerm {
            metric: Level4Psms,
            direction: Maximize,
        },
        ObjectiveTerm {
            metric: AdjustedEntrapmentFdp,
            direction: Minimize,
        },
        ObjectiveTerm {
            metric: ModelComplexity,
            direction: Minimize,
        },
        ObjectiveTerm {
            metric: DeterministicParameterOrder,
            direction: Minimize,
        },
    ]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmpiricalEntrapmentConstraint {
    pub level: String,
    pub maximum_adjusted_fdp: f64,
    /// Schema-v3 name. Schema-v1/v2 manifests using
    /// `minimum_entrapment_count` remain loadable through the alias.
    #[serde(alias = "minimum_entrapment_count")]
    pub minimum_entrapment_observations_for_power: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerWindowSearch {
    pub strategy: String,
    pub min_rank_range: [u32; 2],
    pub max_rank_range: [u32; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerBlock {
    pub id: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub scope: ParameterScope,
    #[serde(default)]
    pub expert: Option<OptimizerExpert>,
    pub strategy: OptimizerStrategy,
    #[serde(default)]
    pub structural_comparison: bool,
    #[serde(default)]
    pub fixed: BTreeMap<String, ParameterValue>,
    #[serde(default)]
    pub space: BTreeMap<String, Vec<ParameterValue>>,
    #[serde(default)]
    pub window_search: Option<OptimizerWindowSearch>,
    #[serde(default)]
    pub use_external_features: bool,
    #[serde(default)]
    pub max_trials: Option<usize>,
    #[serde(default)]
    pub max_passes: Option<usize>,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterOptimizerConfig {
    pub schema_version: u32,
    pub enabled: bool,
    pub classification: OptimizationClassification,
    #[serde(default)]
    pub selected_experts: Vec<OptimizerExpert>,
    /// Prospective model-to-canonical-configuration identities for a frozen
    /// expert roster. Final-Ensemble-only optimization sets
    /// `require_expected_expert_configurations` so preflight cannot silently
    /// substitute a different expert configuration or artifact.
    #[serde(default)]
    pub expected_expert_configuration_sha256: BTreeMap<String, String>,
    #[serde(default)]
    pub require_expected_expert_configurations: bool,
    #[serde(default)]
    pub compiled_defaults: BTreeMap<String, ParameterValue>,
    #[serde(default)]
    pub workflow_defaults: BTreeMap<String, ParameterValue>,
    #[serde(default)]
    pub fixed_baseline_values: BTreeMap<String, ParameterValue>,
    pub seed: u64,
    pub maximum_trial_budget: usize,
    pub maximum_optimization_passes: usize,
    pub objective: Vec<ObjectiveTerm>,
    pub fixed_evaluation_threshold: f64,
    #[serde(default)]
    pub empirical_entrapment_constraints: Vec<EmpiricalEntrapmentConstraint>,
    /// Schema-v3 policy separating development selection from the power of
    /// empirical calibration evidence. Missing preserves schema-v1/v2
    /// behavior.
    #[serde(default)]
    pub underpowered_trial_policy: UnderpoweredTrialPolicy,
    /// Dataset-local label-isolation contract. Missing preserves the
    /// historical full-population development behavior.
    #[serde(default)]
    pub entrapment_validation: EntrapmentValidationConfig,
    #[serde(default)]
    pub statistical_validity_contracts: BTreeMap<String, String>,
    pub resume: bool,
    pub materialize_winner: bool,
    /// Versioned production execution boundary. Missing in schema-v1
    /// manifests and therefore defaults to the historical full workflow.
    #[serde(default)]
    pub execution_mode: OptimizerExecutionMode,
    /// Bounded infrastructure verification: run declared non-Ensemble
    /// optimizer blocks, write their artifacts, and skip ordinary/target-only
    /// workflow stages. Never use this for scientific optimization.
    #[serde(default)]
    pub implementation_smoke_only: bool,
    /// Bounded production-backed integration audit. Trials use the ordinary
    /// Decoy-Free evaluator and real +entrapment metrics, but the workflow
    /// stops after winner materialization and never enters target-only stages.
    #[serde(default)]
    pub production_smoke_only: bool,
    pub require_existing_candidate_pool: bool,
    pub require_existing_raw_annotation_cache: bool,
    pub target_only_outcomes_excluded: bool,
    #[serde(default)]
    pub block_order: Vec<String>,
    pub blocks: Vec<OptimizerBlock>,
}

impl ParameterOptimizerConfig {
    pub fn optimization_only(&self) -> bool {
        self.enabled && self.execution_mode == OptimizerExecutionMode::OptimizationOnly
    }

    pub fn stops_after_optimization(&self) -> bool {
        self.enabled
            && (self.optimization_only()
                || self.implementation_smoke_only
                || self.production_smoke_only)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            (1..=PARAMETER_OPTIMIZER_SCHEMA_VERSION).contains(&self.schema_version),
            "unsupported parameter_optimizer schema {}",
            self.schema_version
        );
        if !self.enabled {
            return Ok(());
        }
        anyhow::ensure!(
            self.classification == OptimizationClassification::DevelopmentOnly,
            "parameter optimizer must be development_only"
        );
        if self.underpowered_trial_policy == UnderpoweredTrialPolicy::DevelopmentEligible {
            anyhow::ensure!(
                self.schema_version >= 3,
                "underpowered_trial_policy=development_eligible requires parameter_optimizer schema_version 3"
            );
            anyhow::ensure!(
                self.classification == OptimizationClassification::DevelopmentOnly,
                "underpowered development eligibility is prohibited for holdout, release, statistical-default, and production-default claims"
            );
        }
        if self.entrapment_validation.mode == EntrapmentValidationMode::SelectionAudit {
            anyhow::ensure!(
                self.schema_version >= 4,
                "entrapment selection/audit partitioning requires parameter_optimizer schema_version 4"
            );
            anyhow::ensure!(
                self.entrapment_validation.partition_schema_version == 1,
                "unsupported entrapment partition schema {}",
                self.entrapment_validation.partition_schema_version
            );
            anyhow::ensure!(
                !self.entrapment_validation.salt.trim().is_empty(),
                "selection/audit entrapment partition requires a nonempty deterministic salt"
            );
            anyhow::ensure!(
                self.entrapment_validation.selection_fraction.is_finite()
                    && self.entrapment_validation.audit_fraction.is_finite()
                    && self.entrapment_validation.selection_fraction > 0.0
                    && self.entrapment_validation.audit_fraction > 0.0
                    && (self.entrapment_validation.selection_fraction
                        + self.entrapment_validation.audit_fraction
                        - 1.0)
                        .abs()
                        <= 1e-12,
                "selection/audit entrapment fractions must be finite, positive, and sum to one"
            );
            anyhow::ensure!(
                self.optimization_only(),
                "selection/audit entrapment validation requires execution_mode=optimization_only so audit occurs only after every winner is frozen"
            );
        }
        anyhow::ensure!(
            self.maximum_trial_budget > 0,
            "maximum_trial_budget must be positive"
        );
        anyhow::ensure!(
            self.maximum_optimization_passes > 0,
            "maximum_optimization_passes must be positive"
        );
        anyhow::ensure!(
            self.fixed_evaluation_threshold.is_finite()
                && (0.0..=1.0).contains(&self.fixed_evaluation_threshold),
            "fixed_evaluation_threshold must be finite and in [0,1]"
        );
        anyhow::ensure!(
            self.require_existing_candidate_pool,
            "parameter optimizer requires strict existing candidate-pool reuse"
        );
        anyhow::ensure!(
            self.require_existing_raw_annotation_cache,
            "parameter optimizer requires strict existing raw annotation-cache reuse"
        );
        anyhow::ensure!(
            self.target_only_outcomes_excluded,
            "target-only outcomes must be excluded from optimization"
        );
        anyhow::ensure!(
            !(self.implementation_smoke_only && self.production_smoke_only),
            "implementation_smoke_only and production_smoke_only are mutually exclusive"
        );
        if self.implementation_smoke_only {
            anyhow::ensure!(
                !self.selected_experts.contains(&OptimizerExpert::Ensemble),
                "implementation_smoke_only cannot assemble or optimize Ensemble"
            );
            anyhow::ensure!(
                self.maximum_trial_budget <= 16,
                "implementation_smoke_only is limited to 16 trials"
            );
            anyhow::ensure!(
                self.empirical_entrapment_constraints.is_empty(),
                "implementation_smoke_only uses no biological metrics or empirical constraints"
            );
            anyhow::ensure!(
                self.blocks
                    .iter()
                    .filter(|block| block.enabled)
                    .any(|block| block.use_external_features),
                "implementation_smoke_only must verify at least one raw annotation-cache consumer"
            );
            for expert in &self.selected_experts {
                anyhow::ensure!(
                    self.blocks
                        .iter()
                        .any(|block| block.enabled && block.expert == Some(*expert)),
                    "implementation_smoke_only selected expert {:?} has no optimizer block",
                    expert
                );
            }
        }
        if self.production_smoke_only {
            anyhow::ensure!(
                !self.selected_experts.contains(&OptimizerExpert::Ensemble),
                "production_smoke_only cannot assemble or optimize Ensemble"
            );
            anyhow::ensure!(
                self.maximum_trial_budget <= 16,
                "production_smoke_only is limited to 16 trials"
            );
            anyhow::ensure!(
                self.blocks
                    .iter()
                    .filter(|block| block.enabled)
                    .any(|block| block.use_external_features),
                "production_smoke_only must exercise the production raw annotation-cache consumer"
            );
            for expert in &self.selected_experts {
                anyhow::ensure!(
                    self.blocks
                        .iter()
                        .any(|block| block.enabled && block.expert == Some(*expert)),
                    "production_smoke_only selected expert {:?} has no optimizer block",
                    expert
                );
            }
        }
        anyhow::ensure!(
            !self.blocks.is_empty(),
            "enabled parameter optimizer requires at least one block"
        );
        anyhow::ensure!(
            !self.selected_experts.is_empty(),
            "enabled parameter optimizer requires selected_experts"
        );
        let selected = self
            .selected_experts
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        anyhow::ensure!(
            selected.len() == self.selected_experts.len(),
            "selected_experts contains duplicates"
        );
        let selected_models = self
            .selected_experts
            .iter()
            .filter(|expert| **expert != OptimizerExpert::Ensemble)
            .map(|expert| expert.slug().to_owned())
            .collect::<BTreeSet<_>>();
        anyhow::ensure!(
            self.expected_expert_configuration_sha256
                .keys()
                .all(|model| selected_models.contains(model)),
            "expected_expert_configuration_sha256 contains an unselected or non-expert model"
        );
        anyhow::ensure!(
            self.expected_expert_configuration_sha256
                .values()
                .all(|hash| {
                    hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                }),
            "expected expert configuration hashes must be 64 hexadecimal characters"
        );
        if self.require_expected_expert_configurations
            || !self.expected_expert_configuration_sha256.is_empty()
        {
            anyhow::ensure!(
                self.selected_experts.contains(&OptimizerExpert::Ensemble),
                "expected expert configurations are meaningful only for an Ensemble optimization"
            );
            let expected = self
                .expected_expert_configuration_sha256
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>();
            anyhow::ensure!(
                expected == selected_models,
                "expected expert configuration map is incomplete or disagrees with the selected frozen expert roster"
            );
        }
        anyhow::ensure!(
            !self.objective.is_empty(),
            "optimizer objective order is required"
        );
        anyhow::ensure!(
            self.objective
                .last()
                .is_some_and(|term| term.metric == ObjectiveMetric::DeterministicParameterOrder),
            "objective must end with deterministic_parameter_order"
        );
        for constraint in &self.empirical_entrapment_constraints {
            anyhow::ensure!(
                constraint.maximum_adjusted_fdp.is_finite()
                    && constraint.maximum_adjusted_fdp >= 0.0,
                "invalid empirical entrapment constraint for {}",
                constraint.level
            );
        }

        let registry = parameter_contracts()
            .into_iter()
            .map(|contract| (contract.name, contract))
            .collect::<BTreeMap<_, _>>();
        for (source, values) in [
            ("compiled_defaults", &self.compiled_defaults),
            ("workflow_defaults", &self.workflow_defaults),
            ("fixed_baseline_values", &self.fixed_baseline_values),
        ] {
            for (name, value) in values {
                let contract = registry
                    .get(name.as_str())
                    .with_context(|| format!("unknown {source} parameter {name}"))?;
                validate_value(contract, value)?;
            }
        }
        let mut ids = BTreeSet::new();
        for block in &self.blocks {
            anyhow::ensure!(
                !block.id.trim().is_empty(),
                "optimizer block id is required"
            );
            anyhow::ensure!(
                ids.insert(block.id.clone()),
                "duplicate optimizer block {}",
                block.id
            );
            if let Some(expert) = block.expert {
                anyhow::ensure!(
                    selected.contains(&expert),
                    "optimizer block {} belongs to unselected expert {:?}",
                    block.id,
                    expert
                );
            }
            validate_block(self, block, &registry)?;
        }
        let declared = self
            .blocks
            .iter()
            .filter(|block| block.enabled)
            .map(|block| block.id.clone())
            .collect::<BTreeSet<_>>();
        let ordered = self.block_order.iter().cloned().collect::<BTreeSet<_>>();
        anyhow::ensure!(
            declared == ordered && self.block_order.len() == declared.len(),
            "block_order must name every enabled block exactly once and no disabled block"
        );
        Ok(())
    }
}

fn validate_block(
    config: &ParameterOptimizerConfig,
    block: &OptimizerBlock,
    registry: &BTreeMap<&str, ParameterContract>,
) -> Result<()> {
    match block.scope {
        ParameterScope::PerExpert => anyhow::ensure!(
            block.expert.is_some(),
            "per_expert block {} requires expert",
            block.id
        ),
        ParameterScope::EnsembleFinal => anyhow::ensure!(
            block.expert == Some(OptimizerExpert::Ensemble),
            "ensemble_final block {} requires expert=ensemble",
            block.id
        ),
        ParameterScope::Physical
        | ParameterScope::Reproducibility
        | ParameterScope::HierarchicalOrReporting => anyhow::ensure!(
            block.expert.is_some(),
            "conditional block {} must declare the expert/final stream it modifies",
            block.id
        ),
        _ => {}
    }
    if block.expert == Some(OptimizerExpert::Msfdr1Smix) {
        anyhow::ensure!(
            block.window_search.is_none(),
            "MSFDR1-SMIX is fixed at rank 1-1 and has no optimizable null window"
        );
    }
    if let Some(window) = &block.window_search {
        anyhow::ensure!(
            window.strategy == "explicit_grid" || window.strategy == "landscape_adaptive",
            "unsupported model-local window strategy {}",
            window.strategy
        );
        anyhow::ensure!(
            window.min_rank_range[0] >= 2 && window.min_rank_range[0] <= window.min_rank_range[1],
            "invalid window min-rank range"
        );
        anyhow::ensure!(
            window.max_rank_range[0] >= 2 && window.max_rank_range[0] <= window.max_rank_range[1],
            "invalid window max-rank range"
        );
    }
    let validate_one = |name: &str, value: &ParameterValue, optimized: bool| -> Result<()> {
        let contract = registry
            .get(name)
            .with_context(|| format!("unknown optimizer parameter {name}"))?;
        anyhow::ensure!(
            contract.scopes.contains(&block.scope),
            "parameter {name} has wrong scope {:?}",
            block.scope
        );
        validate_owner(contract, block.expert)?;
        validate_value(contract, value)?;
        if optimized && block.enabled {
            let binding = production_binding(name)?;
            anyhow::ensure!(
                binding.status != ProductionBindingStatus::DeliberatelyDeferred,
                "parameter {name} is deliberately deferred and has no active production binding: {}",
                binding.reason
            );
            anyhow::ensure!(
                contract.eligible,
                "parameter {name} is not eligible for internal optimization"
            );
            anyhow::ensure!(
                !matches!(
                    contract.class,
                    ParameterClass::NumericalConvergenceOrPrecision
                        | ParameterClass::FixedReportingThreshold
                        | ParameterClass::ValidationReportingOrProvenanceOnly
                ),
                "parameter {name} cannot be optimized for identification yield"
            );
            if contract.class == ParameterClass::StructuralMethodFamilyChoice {
                anyhow::ensure!(
                    block.structural_comparison,
                    "structural parameter {name} requires structural_comparison=true"
                );
            }
            if contract.validity_contract_required {
                let key = format!("{name}:statistical_validity");
                anyhow::ensure!(
                    config
                        .statistical_validity_contracts
                        .get(&key)
                        .is_some_and(|text| !text.trim().is_empty()),
                    "parameter {name} requires explicit statistical-validity contract {key}"
                );
            }
        }
        Ok(())
    };
    for (name, value) in &block.fixed {
        validate_one(name, value, false)?;
    }
    for (name, values) in &block.space {
        anyhow::ensure!(!values.is_empty(), "parameter space {name} is empty");
        for value in values {
            validate_one(name, value, true)?;
        }
    }
    if block.enabled && block.strategy == OptimizerStrategy::ExhaustiveGrid {
        let trials = block.space.values().try_fold(1usize, |total, values| {
            total
                .checked_mul(values.len())
                .context("exhaustive grid overflows trial count")
        })?;
        let limit = block
            .max_trials
            .unwrap_or(config.maximum_trial_budget)
            .min(config.maximum_trial_budget);
        anyhow::ensure!(
            trials <= limit,
            "exhaustive grid declares {trials} trials above limit {limit}"
        );
    }
    validate_assignment(&block.fixed)?;
    validate_selected_weights(config, block)?;
    Ok(())
}

fn validate_owner(contract: &ParameterContract, expert: Option<OptimizerExpert>) -> Result<()> {
    let Some(expert) = expert else {
        return Ok(());
    };
    let owner = expert.slug();
    let compatible = matches!(
        contract.owner,
        "evidence" | "aggregation" | "q_value" | "shared_null" | "storey"
    ) || matches!(
        contract.owner,
        "physical" | "reproducibility" | "hierarchical" | "validation"
    ) || contract.owner == owner
        || (contract.owner == "ensemble" && expert == OptimizerExpert::Ensemble)
        || (contract.owner == "msfdr_mixtures"
            && matches!(
                expert,
                OptimizerExpert::Msfdr1Smix | OptimizerExpert::Msfdr2Smix
            ));
    anyhow::ensure!(
        compatible,
        "parameter {} belongs to {}, not {}",
        contract.name,
        contract.owner,
        owner
    );
    Ok(())
}

fn validate_value(contract: &ParameterContract, value: &ParameterValue) -> Result<()> {
    let kind_ok = matches!(
        (contract.kind, value),
        (ParameterKind::Boolean, ParameterValue::Bool(_))
            | (ParameterKind::Integer, ParameterValue::Integer(_))
            | (
                ParameterKind::Float,
                ParameterValue::Float(_) | ParameterValue::Integer(_)
            )
            | (ParameterKind::Enumeration, ParameterValue::String(_))
    );
    anyhow::ensure!(kind_ok, "parameter {} has wrong value type", contract.name);
    if let Some(numeric) = value.as_f64() {
        anyhow::ensure!(
            numeric.is_finite(),
            "parameter {} must be finite",
            contract.name
        );
        if let Some(minimum) = contract.minimum {
            anyhow::ensure!(
                numeric >= minimum,
                "parameter {} is below minimum {minimum}",
                contract.name
            );
        }
        if let Some(maximum) = contract.maximum {
            anyhow::ensure!(
                numeric <= maximum,
                "parameter {} is above maximum {maximum}",
                contract.name
            );
        }
    }
    if let ParameterValue::String(value) = value {
        anyhow::ensure!(
            contract.enum_values.contains(&value.as_str()),
            "parameter {} has unsupported value {value}",
            contract.name
        );
    }
    Ok(())
}

fn validate_selected_weights(
    config: &ParameterOptimizerConfig,
    block: &OptimizerBlock,
) -> Result<()> {
    if block.scope != ParameterScope::EnsembleFinal {
        return Ok(());
    }
    for expert in &config.selected_experts {
        if *expert == OptimizerExpert::Ensemble {
            continue;
        }
        let name = format!("ensemble_weight_{}", expert.slug());
        for value in block
            .space
            .get(&name)
            .into_iter()
            .flatten()
            .chain(block.fixed.get(&name))
            .chain(config.compiled_defaults.get(&name))
            .chain(config.workflow_defaults.get(&name))
            .chain(config.fixed_baseline_values.get(&name))
        {
            anyhow::ensure!(
                value.as_f64().is_some_and(|weight| weight > 0.0),
                "JSON-selected voter {expert:?} cannot receive zero effective weight"
            );
        }
    }
    Ok(())
}

pub fn validate_assignment(values: &BTreeMap<String, ParameterValue>) -> Result<()> {
    let pair = |low: &str, high: &str| -> Result<()> {
        if let (Some(low), Some(high)) = (
            values.get(low).and_then(ParameterValue::as_f64),
            values.get(high).and_then(ParameterValue::as_f64),
        ) {
            anyhow::ensure!(
                low <= high,
                "dependency violated: lower value exceeds upper value"
            );
        }
        Ok(())
    };
    for (low, high) in [
        ("storey_pi0_clamp_min", "storey_pi0_clamp_max"),
        ("storey_lambda_min", "storey_lambda_max"),
        ("moments_min_null_rank", "moments_max_null_rank"),
        ("moments_winsor_lower_q", "moments_winsor_upper_q"),
        ("mle_min_null_rank", "mle_max_null_rank"),
        ("mle_winsor_lower_q", "mle_winsor_upper_q"),
        ("lower_order_min_null_rank", "lower_order_max_null_rank"),
        ("msfdr_min_null_rank", "msfdr_max_null_rank"),
        ("msfdr_pi_clamp_min", "msfdr_pi_clamp_max"),
        ("msfdr1_pi_clamp_min", "msfdr1_pi_clamp_max"),
        ("msfdr2_smix_min_null_rank", "msfdr2_smix_max_null_rank"),
        ("msfdr2_pi_clamp_min", "msfdr2_pi_clamp_max"),
        ("nokoi_min_null_rank", "nokoi_max_null_rank"),
        ("nokoi_l1_lambda_min", "nokoi_l1_lambda_max"),
        ("p_combine_calibration_min_k", "p_combine_calibration_max_k"),
    ] {
        pair(low, high)?;
    }
    if let (Some(minimum), Some(aggregation)) = (
        values
            .get("storey_lambda_min")
            .and_then(ParameterValue::as_f64),
        values
            .get("storey_lambda_min_for_agg")
            .and_then(ParameterValue::as_f64),
    ) {
        anyhow::ensure!(
            aggregation >= minimum,
            "storey_lambda_min_for_agg is below storey_lambda_min"
        );
    }
    if let (Some(maximum), Some(aggregation)) = (
        values
            .get("storey_lambda_max")
            .and_then(ParameterValue::as_f64),
        values
            .get("storey_lambda_min_for_agg")
            .and_then(ParameterValue::as_f64),
    ) {
        anyhow::ensure!(
            aggregation <= maximum,
            "storey_lambda_min_for_agg exceeds storey_lambda_max"
        );
    }
    Ok(())
}

fn validate_active_dependencies(
    values: &BTreeMap<String, ParameterValue>,
    active: &BTreeSet<String>,
) -> Result<()> {
    let string_value = |name: &str| match values.get(name) {
        Some(ParameterValue::String(value)) => Some(value.as_str()),
        _ => None,
    };
    let bool_value = |name: &str| match values.get(name) {
        Some(ParameterValue::Bool(value)) => Some(*value),
        _ => None,
    };
    let any_active = |names: &[&str]| names.iter().any(|name| active.contains(*name));

    if any_active(&["moments_winsor_lower_q", "moments_winsor_upper_q"]) {
        anyhow::ensure!(
            bool_value("moments_robust_fit") == Some(true),
            "dependency violated: Moments winsor quantiles require moments_robust_fit=true"
        );
    }
    if any_active(&["mle_winsor_lower_q", "mle_winsor_upper_q"]) {
        anyhow::ensure!(
            bool_value("mle_robust_fit") == Some(true),
            "dependency violated: MLE winsor quantiles require mle_robust_fit=true"
        );
    }
    if active.contains("ensemble_p_combiner") {
        anyhow::ensure!(
            string_value("final_evidence_space") == Some("p_value"),
            "dependency violated: ensemble_p_combiner affects the selected decision stream only when final_evidence_space=p_value"
        );
    }
    if active.contains("ensemble_pep_combiner") {
        anyhow::ensure!(
            string_value("final_evidence_space") == Some("pep"),
            "dependency violated: ensemble_pep_combiner affects the selected decision stream only when final_evidence_space=pep"
        );
    }
    if active.contains("ensemble_cauchy_penalty") {
        anyhow::ensure!(
            string_value("final_evidence_space") == Some("p_value")
                && string_value("ensemble_p_combiner") == Some("cauchy"),
            "dependency violated: ensemble_cauchy_penalty requires final_evidence_space=p_value and ensemble_p_combiner=cauchy"
        );
    }
    if active.contains("ensemble_pep_trim_frac") {
        anyhow::ensure!(
            string_value("final_evidence_space") == Some("pep")
                && matches!(
                    string_value("ensemble_pep_combiner"),
                    Some("trimmed_mean" | "winsorized_mean")
                ),
            "dependency violated: ensemble_pep_trim_frac requires final_evidence_space=pep and a trimmed or winsorized PEP combiner"
        );
    }
    if active.contains("ensemble_pep_quantile") {
        anyhow::ensure!(
            string_value("final_evidence_space") == Some("pep")
                && string_value("ensemble_pep_combiner") == Some("quantile"),
            "dependency violated: ensemble_pep_quantile requires final_evidence_space=pep and ensemble_pep_combiner=quantile"
        );
    }
    if active.contains("ensemble_pep_top_k") {
        anyhow::ensure!(
            string_value("final_evidence_space") == Some("pep")
                && string_value("ensemble_pep_combiner") == Some("top_k_mean"),
            "dependency violated: ensemble_pep_top_k requires final_evidence_space=pep and ensemble_pep_combiner=top_k_mean"
        );
    }
    if active
        .iter()
        .any(|name| name.starts_with("ensemble_weight_"))
    {
        anyhow::ensure!(
            string_value("final_evidence_space") == Some("pep")
                && matches!(
                    string_value("ensemble_pep_combiner"),
                    Some("weighted_mean" | "weighted_median")
                ),
            "dependency violated: expert weights affect the selected decision stream only when final_evidence_space=pep and ensemble_pep_combiner is weighted_mean or weighted_median"
        );
    }
    if any_active(&["p_combine_calibration_min_k", "p_combine_calibration_max_k"]) {
        anyhow::ensure!(
            string_value("p_combine_calibration_mode") == Some("rank_null"),
            "dependency violated: p-combination calibration bounds require rank_null mode"
        );
    }
    if active.contains("p_combine_tfisher_tau") {
        anyhow::ensure!(
            matches!(string_value("peptide_p_combine"), Some("t_fisher"))
                || matches!(string_value("protein_p_combine"), Some("t_fisher")),
            "dependency violated: p_combine_tfisher_tau requires a selected t_fisher combiner"
        );
    }
    if active.contains("bky_alpha") {
        anyhow::ensure!(
            ["psm_q_method", "peptide_q_method", "protein_q_method"]
                .iter()
                .any(|name| string_value(name) == Some("bky")),
            "dependency violated: bky_alpha requires an applicable q-value method=bky"
        );
    }
    if active.contains("sfdr_gamma") {
        anyhow::ensure!(
            ["psm_q_method", "peptide_q_method", "protein_q_method"]
                .iter()
                .any(|name| string_value(name) == Some("sfdr")),
            "dependency violated: sfdr_gamma requires an applicable q-value method=sfdr"
        );
    }
    for level in ["psm", "peptide", "protein"] {
        if any_active(&[
            &format!("{level}_q_covariate"),
            &format!("{level}_q_covariate_bins"),
            &format!("{level}_q_covariate_weight_strength"),
        ]) {
            anyhow::ensure!(
                string_value(&format!("{level}_q_method")) == Some("covariate_weighted_bh"),
                "dependency violated: {level} q-value covariate settings require covariate_weighted_bh"
            );
            anyhow::ensure!(
                string_value(&format!("{level}_q_covariate")).is_some_and(|value| value != "none"),
                "dependency violated: {level} q-value covariate settings require a non-none validated covariate"
            );
        }
    }
    Ok(())
}

/// Apply a validated flat parameter map to JSON-exposed FDR options. Dotted
/// paths address nested structures. This preserves serde's enum spelling and
/// ensures the normal FdrSettings resolver remains authoritative.
pub fn apply_fdr_overrides(
    options: &mut sage_core::input::FdrOptions,
    values: &BTreeMap<String, ParameterValue>,
) -> Result<()> {
    let registry = parameter_contracts()
        .into_iter()
        .map(|entry| (entry.name, entry))
        .collect::<BTreeMap<_, _>>();
    let mut root = serde_json::to_value(&*options)?;
    for (name, value) in values {
        let contract = registry
            .get(name.as_str())
            .with_context(|| format!("unknown optimizer parameter {name}"))?;
        anyhow::ensure!(
            contract.json_exposed,
            "parameter {name} is internal-only and cannot be materialized"
        );
        set_json_path(&mut root, name, value.to_json())?;
    }
    *options = serde_json::from_value(root)
        .context("optimizer values do not deserialize as FdrOptions")?;
    Ok(())
}

fn set_json_path(root: &mut serde_json::Value, path: &str, value: serde_json::Value) -> Result<()> {
    let mut parts = path.split('.').peekable();
    let mut cursor = root;
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            cursor
                .as_object_mut()
                .context("optimizer JSON path parent is not an object")?
                .insert(part.into(), value);
            return Ok(());
        }
        let object = cursor
            .as_object_mut()
            .context("optimizer JSON path parent is not an object")?;
        cursor = object.entry(part).or_insert_with(|| serde_json::json!({}));
    }
    anyhow::bail!("empty optimizer JSON path")
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrialMetrics {
    pub level4_proteins: usize,
    pub level4_canonical_peptides: usize,
    pub level4_peptidoforms: usize,
    pub level4_psms: usize,
    pub adjusted_entrapment_fdp: Option<f64>,
    pub entrapment_count: usize,
    #[serde(default)]
    pub adjusted_entrapment_fdp_by_level: BTreeMap<String, Option<f64>>,
    #[serde(default)]
    pub entrapment_count_by_level: BTreeMap<String, usize>,
    pub model_complexity: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    Feasible,
    TechnicalFailure,
    EmpiricallyInfeasible,
    NotEvaluable,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EmpiricalCalibrationPower {
    #[default]
    NotAssessed,
    Underpowered,
    AdequatelyPowered,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatisticalValidationStatus {
    #[default]
    NotEvaluated,
    NotEvaluableUnderpowered,
    EmpiricallyEvaluable,
    EmpiricallyInfeasible,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatisticalDefaultEligibility {
    #[default]
    NotEvaluated,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrialEvaluation {
    pub status: TrialStatus,
    #[serde(default)]
    pub technical_reason: Option<String>,
    #[serde(default)]
    pub empirical_reason: Option<String>,
    #[serde(default)]
    pub metrics: Option<TrialMetrics>,
    #[serde(default)]
    pub development_selection_eligible: bool,
    #[serde(default)]
    pub empirical_point_estimate_within_limit: Option<bool>,
    #[serde(default)]
    pub empirical_calibration_power: EmpiricalCalibrationPower,
    #[serde(default)]
    pub statistical_validation_status: StatisticalValidationStatus,
    #[serde(default)]
    pub statistical_default_eligibility: StatisticalDefaultEligibility,
    #[serde(default)]
    pub compact_diagnostics: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrialRequest {
    pub trial_id: String,
    pub block_id: String,
    pub pass: usize,
    pub ordinal: usize,
    pub scope: ParameterScope,
    pub expert: Option<OptimizerExpert>,
    pub parameters: BTreeMap<String, ParameterValue>,
    pub use_external_features: bool,
    pub target_only_outcomes_allowed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrialRecord {
    pub request: TrialRequest,
    pub evaluation: TrialEvaluation,
    pub reused_from_checkpoint: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptedTransition {
    pub block_id: String,
    pub pass: usize,
    pub from_trial: Option<String>,
    pub to_trial: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerOutcome {
    ExhaustiveBoundedOptimum,
    CompletedHeuristicLocal,
    CompletedDevelopmentOptimization,
    UnderpoweredDevelopmentWinner,
    TrialBudgetExhausted,
    NoTechnicallyValidSolution,
    NoEmpiricallyFeasibleSolution,
    InterruptedResumable,
    NotEvaluable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerCheckpointPayload {
    pub schema_version: u32,
    pub optimizer_fingerprint: String,
    pub completed_trials: BTreeMap<String, TrialRecord>,
    pub accepted_transitions: Vec<AcceptedTransition>,
    pub current_parameters: BTreeMap<String, ParameterValue>,
    #[serde(default)]
    pub resolved_parameter_sets: BTreeMap<String, BTreeMap<String, ParameterValue>>,
    #[serde(default)]
    pub block_winners: BTreeMap<String, String>,
    pub winner_trial_id: Option<String>,
    pub outcome: OptimizerOutcome,
    pub resume_history: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerCheckpoint {
    pub payload: OptimizerCheckpointPayload,
    pub payload_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerIdentity {
    pub schema_version: u32,
    pub execution_mode: OptimizerExecutionMode,
    pub dataset_identity: String,
    pub candidate_pool_identity: String,
    pub raw_annotation_cache_identity: String,
    pub calibrated_annotation_identity: Option<String>,
    pub model_artifact_schema: u32,
    pub optimizer_schema: u32,
    pub optimizer_source_sha256: String,
    pub source_configuration_sha256: String,
    pub catalog_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entrapment_partition_identity: Option<String>,
}

pub fn optimizer_fingerprint(
    identity: &OptimizerIdentity,
    config: &ParameterOptimizerConfig,
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-decoy-free-parameter-optimizer-v1\0");
    hasher.update(serde_json::to_vec(identity)?);
    hasher.update(serde_json::to_vec(config)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn payload_sha256(payload: &OptimizerCheckpointPayload) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-parameter-optimizer-checkpoint-v1\0");
    hasher.update(serde_json::to_vec(payload)?);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn write_checkpoint(path: &Path, payload: &OptimizerCheckpointPayload) -> Result<()> {
    let checkpoint = OptimizerCheckpoint {
        payload: payload.clone(),
        payload_sha256: payload_sha256(payload)?,
    };
    write_json_atomic(path, &checkpoint)
}

pub fn load_checkpoint(
    path: &Path,
    expected_fingerprint: &str,
) -> Result<OptimizerCheckpointPayload> {
    let checkpoint: OptimizerCheckpoint = serde_json::from_slice(
        &std::fs::read(path)
            .with_context(|| format!("failed to read optimizer checkpoint {}", path.display()))?,
    )?;
    anyhow::ensure!(
        checkpoint.payload.schema_version == PARAMETER_OPTIMIZER_SCHEMA_VERSION,
        "incompatible optimizer checkpoint schema"
    );
    anyhow::ensure!(
        checkpoint.payload.optimizer_fingerprint == expected_fingerprint,
        "optimizer checkpoint fingerprint mismatch"
    );
    anyhow::ensure!(
        checkpoint.payload_sha256 == payload_sha256(&checkpoint.payload)?,
        "optimizer checkpoint payload integrity failure"
    );
    Ok(checkpoint.payload)
}

pub trait TrialEvaluator {
    fn evaluate(&mut self, request: &TrialRequest) -> Result<TrialEvaluation>;
    fn materialize_winner(&mut self, _record: &TrialRecord) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerRunResult {
    pub schema_version: u32,
    pub optimizer_fingerprint: String,
    /// Stable scientific result identity. Operational resume annotations are
    /// deliberately excluded so exact checkpoint replay has the same hash.
    pub scientific_result_sha256: String,
    pub parameter_binding_coverage: Vec<ParameterProductionBinding>,
    pub classification: OptimizationClassification,
    pub execution_mode: OptimizerExecutionMode,
    pub outcome: OptimizerOutcome,
    pub strategy_classification: String,
    pub requested_parameter_space: Vec<OptimizerBlock>,
    pub block_order: Vec<String>,
    pub resolved_parameters: BTreeMap<String, ParameterValue>,
    pub resolved_parameter_sets: BTreeMap<String, BTreeMap<String, ParameterValue>>,
    pub parameter_precedence: Vec<String>,
    pub objective: Vec<ObjectiveTerm>,
    pub empirical_constraints: Vec<EmpiricalEntrapmentConstraint>,
    pub underpowered_trial_policy: UnderpoweredTrialPolicy,
    pub powered_trial_count: usize,
    pub underpowered_trial_count: usize,
    pub empirical_power_not_assessed_trial_count: usize,
    pub trials: Vec<TrialRecord>,
    pub accepted_transitions: Vec<AcceptedTransition>,
    pub winner_trial_id: Option<String>,
    pub block_winners: BTreeMap<String, String>,
    pub winner_artifacts: BTreeMap<String, serde_json::Value>,
    pub target_only_non_leakage: String,
    pub development_only: bool,
    pub independent_evaluation_status: String,
    pub statistical_default_status: String,
    /// Computed from the already-frozen winner after optimization has
    /// terminated. This field is never present in optimizer checkpoints and
    /// never participates in winner selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frozen_audit: Option<FrozenWinnerAuditEvaluation>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuditLevelMetrics {
    pub targets: usize,
    pub audit_entrapments: usize,
    pub measured_audit_ratio: f64,
    pub adjusted_fdp: Option<f64>,
    pub adjusted_fdp_interval_95: Option<[f64; 2]>,
    pub minimum_observations_for_power: usize,
    pub empirical_calibration_power: EmpiricalCalibrationPower,
    pub maximum_adjusted_fdp: Option<f64>,
    pub empirical_point_estimate_within_limit: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrozenWinnerAuditEvaluation {
    pub schema_version: u32,
    pub partition_identity: String,
    pub expert: OptimizerExpert,
    pub winner_trial_id: String,
    pub winner_results_sha256: String,
    pub evaluated_after_winner_freeze: bool,
    pub psm: AuditLevelMetrics,
    pub canonical_peptide: AuditLevelMetrics,
    pub peptidoform: AuditLevelMetrics,
    pub protein: AuditLevelMetrics,
    pub empirical_calibration_power: EmpiricalCalibrationPower,
    pub statistical_validation_status: StatisticalValidationStatus,
    pub statistical_default_eligibility: StatisticalDefaultEligibility,
    pub voter_participation_effect: String,
    pub target_only_outcomes_used: bool,
    pub payload_sha256: String,
}

pub fn run_optimizer<E: TrialEvaluator>(
    config: &ParameterOptimizerConfig,
    identity: &OptimizerIdentity,
    checkpoint_path: &Path,
    evaluator: &mut E,
) -> Result<OptimizerRunResult> {
    config.validate()?;
    let fingerprint = optimizer_fingerprint(identity, config)?;
    let mut payload = if config.resume && checkpoint_path.is_file() {
        let mut payload = load_checkpoint(checkpoint_path, &fingerprint)?;
        payload
            .resume_history
            .push(format!("resume-{}", payload.resume_history.len() + 1));
        payload
    } else {
        OptimizerCheckpointPayload {
            schema_version: PARAMETER_OPTIMIZER_SCHEMA_VERSION,
            optimizer_fingerprint: fingerprint.clone(),
            completed_trials: BTreeMap::new(),
            accepted_transitions: Vec::new(),
            current_parameters: resolve_baseline(config),
            resolved_parameter_sets: BTreeMap::new(),
            block_winners: BTreeMap::new(),
            winner_trial_id: None,
            outcome: OptimizerOutcome::InterruptedResumable,
            resume_history: vec!["initial".into()],
        }
    };
    let mut total_trials = payload.completed_trials.len();
    let mut any_technical = false;
    let mut any_empirical = false;
    let mut any_exhaustive = false;
    let mut any_heuristic = false;

    for block_id in &config.block_order {
        let block = config
            .blocks
            .iter()
            .find(|block| &block.id == block_id)
            .context("validated optimizer block disappeared")?;
        let parameter_set_key = block_parameter_set_key(block);
        let mut base = resolve_baseline_for_block(config, block);
        for earlier_id in config.block_order.iter().take_while(|id| *id != block_id) {
            let Some(earlier) = config
                .blocks
                .iter()
                .find(|candidate| &candidate.id == earlier_id)
            else {
                continue;
            };
            if block_parameter_set_key(earlier) == parameter_set_key {
                if let Some(record) = payload
                    .block_winners
                    .get(earlier_id)
                    .and_then(|winner| payload.completed_trials.get(winner))
                {
                    base.extend(record.request.parameters.clone());
                }
            }
        }
        base.extend(block.fixed.clone());
        validate_assignment(&base)?;
        let block_limit = block
            .max_trials
            .unwrap_or(config.maximum_trial_budget)
            .min(config.maximum_trial_budget);
        let mut block_trials = payload
            .completed_trials
            .values()
            .filter(|record| record.request.block_id == block.id)
            .count();
        match block.strategy {
            OptimizerStrategy::ExhaustiveGrid => {
                any_exhaustive = true;
                let combinations = exhaustive_combinations(&block.space)?;
                let active_parameters = block.space.keys().cloned().collect::<BTreeSet<_>>();
                let mut best: Option<TrialRecord> = None;
                for (ordinal, trial_values) in combinations.into_iter().enumerate() {
                    let mut values = base.clone();
                    values.extend(trial_values);
                    let expected_id =
                        trial_id(&payload.optimizer_fingerprint, block, 1, ordinal, &values)?;
                    if (total_trials >= config.maximum_trial_budget || block_trials >= block_limit)
                        && !payload.completed_trials.contains_key(&expected_id)
                    {
                        payload.outcome = OptimizerOutcome::TrialBudgetExhausted;
                        write_checkpoint(checkpoint_path, &payload)?;
                        return finish_result(
                            config,
                            fingerprint,
                            payload,
                            evaluator,
                            any_exhaustive,
                            any_heuristic,
                        );
                    }
                    let record = evaluate_or_resume(
                        config,
                        block,
                        1,
                        ordinal,
                        values,
                        &active_parameters,
                        &mut payload,
                        evaluator,
                    )?;
                    total_trials = payload.completed_trials.len();
                    block_trials = payload
                        .completed_trials
                        .values()
                        .filter(|record| record.request.block_id == block.id)
                        .count();
                    any_technical |= record.evaluation.status != TrialStatus::TechnicalFailure;
                    any_empirical |= record.evaluation.status == TrialStatus::Feasible;
                    if record.evaluation.status == TrialStatus::Feasible
                        && best
                            .as_ref()
                            .is_none_or(|current| better(config, &record, current))
                    {
                        best = Some(record);
                    }
                    write_checkpoint(checkpoint_path, &payload)?;
                }
                if let Some(best) = best {
                    accept_transition(block, 1, &best, &mut payload);
                    payload.current_parameters = best.request.parameters.clone();
                    payload.winner_trial_id = Some(best.request.trial_id.clone());
                    payload
                        .resolved_parameter_sets
                        .insert(parameter_set_key.clone(), best.request.parameters.clone());
                    payload
                        .block_winners
                        .insert(block.id.clone(), best.request.trial_id.clone());
                }
            }
            OptimizerStrategy::StagedCoordinate => {
                any_heuristic = true;
                let passes = block
                    .max_passes
                    .unwrap_or(config.maximum_optimization_passes)
                    .min(config.maximum_optimization_passes);
                // The declared starting point is itself a recorded trial. A
                // coordinate result is otherwise impossible to audit because
                // accepted transitions would have an implicit baseline.
                let starting_id = trial_id(&payload.optimizer_fingerprint, block, 0, 0, &base)?;
                if (total_trials >= config.maximum_trial_budget || block_trials >= block_limit)
                    && !payload.completed_trials.contains_key(&starting_id)
                {
                    payload.outcome = OptimizerOutcome::TrialBudgetExhausted;
                    write_checkpoint(checkpoint_path, &payload)?;
                    return finish_result(
                        config,
                        fingerprint,
                        payload,
                        evaluator,
                        any_exhaustive,
                        any_heuristic,
                    );
                }
                let starting = evaluate_or_resume(
                    config,
                    block,
                    0,
                    0,
                    base.clone(),
                    &BTreeSet::new(),
                    &mut payload,
                    evaluator,
                )?;
                total_trials = payload.completed_trials.len();
                block_trials = payload
                    .completed_trials
                    .values()
                    .filter(|record| record.request.block_id == block.id)
                    .count();
                any_technical |= starting.evaluation.status != TrialStatus::TechnicalFailure;
                any_empirical |= starting.evaluation.status == TrialStatus::Feasible;
                let mut current = (starting.evaluation.status == TrialStatus::Feasible)
                    .then_some(starting.clone());
                if current.is_some() {
                    accept_transition(block, 0, &starting, &mut payload);
                }
                write_checkpoint(checkpoint_path, &payload)?;
                for pass in 1..=passes {
                    let mut improved = false;
                    for (parameter_index, (name, candidates)) in block.space.iter().enumerate() {
                        let mut local_best = current.clone();
                        for (candidate_index, candidate) in
                            sorted_values(candidates).into_iter().enumerate()
                        {
                            let mut values = base.clone();
                            if let Some(selected) = current.as_ref() {
                                values = selected.request.parameters.clone();
                            }
                            values.insert(name.clone(), candidate);
                            // Trial identity depends only on the declared block order,
                            // pass, parameter, and candidate. It must not depend on how
                            // many records happened to be loaded from a checkpoint.
                            let ordinal =
                                pass * 1_000_000 + parameter_index * 10_000 + candidate_index;
                            let expected_id = trial_id(
                                &payload.optimizer_fingerprint,
                                block,
                                pass,
                                ordinal,
                                &values,
                            )?;
                            if (total_trials >= config.maximum_trial_budget
                                || block_trials >= block_limit)
                                && !payload.completed_trials.contains_key(&expected_id)
                            {
                                payload.outcome = OptimizerOutcome::TrialBudgetExhausted;
                                write_checkpoint(checkpoint_path, &payload)?;
                                return finish_result(
                                    config,
                                    fingerprint,
                                    payload,
                                    evaluator,
                                    any_exhaustive,
                                    any_heuristic,
                                );
                            }
                            let record = evaluate_or_resume(
                                config,
                                block,
                                pass,
                                ordinal,
                                values,
                                &BTreeSet::from([name.clone()]),
                                &mut payload,
                                evaluator,
                            )?;
                            total_trials = payload.completed_trials.len();
                            block_trials = payload
                                .completed_trials
                                .values()
                                .filter(|record| record.request.block_id == block.id)
                                .count();
                            any_technical |=
                                record.evaluation.status != TrialStatus::TechnicalFailure;
                            any_empirical |= record.evaluation.status == TrialStatus::Feasible;
                            if record.evaluation.status == TrialStatus::Feasible
                                && local_best
                                    .as_ref()
                                    .is_none_or(|best| better(config, &record, best))
                            {
                                local_best = Some(record);
                            }
                            write_checkpoint(checkpoint_path, &payload)?;
                        }
                        if let Some(best) = local_best {
                            if current
                                .as_ref()
                                .is_none_or(|old| best.request.trial_id != old.request.trial_id)
                            {
                                accept_transition(block, pass, &best, &mut payload);
                                current = Some(best);
                                improved = true;
                            }
                        }
                    }
                    if !improved {
                        break;
                    }
                }
                if let Some(best) = current {
                    payload.current_parameters = best.request.parameters.clone();
                    payload.winner_trial_id = Some(best.request.trial_id.clone());
                    payload
                        .resolved_parameter_sets
                        .insert(parameter_set_key.clone(), best.request.parameters.clone());
                    payload
                        .block_winners
                        .insert(block.id.clone(), best.request.trial_id.clone());
                }
            }
        }
        if !payload.block_winners.contains_key(&block.id) {
            let records = payload
                .completed_trials
                .values()
                .filter(|record| record.request.block_id == block.id)
                .collect::<Vec<_>>();
            let nontechnical = records
                .iter()
                .filter(|record| record.evaluation.status != TrialStatus::TechnicalFailure)
                .collect::<Vec<_>>();
            payload.outcome = if nontechnical.is_empty() {
                OptimizerOutcome::NoTechnicallyValidSolution
            } else if nontechnical
                .iter()
                .all(|record| record.evaluation.status == TrialStatus::EmpiricallyInfeasible)
            {
                OptimizerOutcome::NoEmpiricallyFeasibleSolution
            } else {
                OptimizerOutcome::NotEvaluable
            };
            write_checkpoint(checkpoint_path, &payload)?;
            return finish_result(
                config,
                fingerprint,
                payload,
                evaluator,
                any_exhaustive,
                any_heuristic,
            );
        }
    }

    // Sequential blocks deliberately avoid the global Cartesian product. Even
    // when each block is exhaustive, the multi-block result is a deterministic
    // staged/local result rather than a global optimum.
    any_heuristic |= config.blocks.len() > 1;
    payload.outcome = if !any_technical {
        OptimizerOutcome::NoTechnicallyValidSolution
    } else if !any_empirical {
        OptimizerOutcome::NoEmpiricallyFeasibleSolution
    } else if config.underpowered_trial_policy == UnderpoweredTrialPolicy::DevelopmentEligible {
        let winner_is_underpowered = payload
            .winner_trial_id
            .as_ref()
            .and_then(|winner| payload.completed_trials.get(winner))
            .is_some_and(|record| {
                record.evaluation.empirical_calibration_power
                    == EmpiricalCalibrationPower::Underpowered
            });
        if winner_is_underpowered {
            OptimizerOutcome::UnderpoweredDevelopmentWinner
        } else {
            OptimizerOutcome::CompletedDevelopmentOptimization
        }
    } else if any_heuristic {
        OptimizerOutcome::CompletedHeuristicLocal
    } else {
        OptimizerOutcome::ExhaustiveBoundedOptimum
    };
    write_checkpoint(checkpoint_path, &payload)?;
    finish_result(
        config,
        fingerprint,
        payload,
        evaluator,
        any_exhaustive,
        any_heuristic,
    )
}

fn resolve_baseline(config: &ParameterOptimizerConfig) -> BTreeMap<String, ParameterValue> {
    let mut values = config.compiled_defaults.clone();
    values.extend(config.workflow_defaults.clone());
    values.extend(config.fixed_baseline_values.clone());
    values
}

fn resolve_baseline_for_block(
    config: &ParameterOptimizerConfig,
    block: &OptimizerBlock,
) -> BTreeMap<String, ParameterValue> {
    let registry = parameter_contracts()
        .into_iter()
        .map(|contract| (contract.name, contract))
        .collect::<BTreeMap<_, _>>();
    resolve_baseline(config)
        .into_iter()
        .filter(|(name, _)| {
            registry.get(name.as_str()).is_some_and(|contract| {
                contract.scopes.contains(&block.scope)
                    && validate_owner(contract, block.expert).is_ok()
            })
        })
        .collect()
}

fn block_parameter_set_key(block: &OptimizerBlock) -> String {
    match block.expert {
        Some(expert) => format!("{:?}:{}", block.scope, expert.slug()).to_ascii_lowercase(),
        None => format!("{:?}:shared", block.scope).to_ascii_lowercase(),
    }
}

fn sorted_values(values: &[ParameterValue]) -> Vec<ParameterValue> {
    let mut values = values.to_vec();
    values.sort_by_key(ParameterValue::canonical_key);
    values.dedup_by(|left, right| left.canonical_key() == right.canonical_key());
    values
}

fn exhaustive_combinations(
    space: &BTreeMap<String, Vec<ParameterValue>>,
) -> Result<Vec<BTreeMap<String, ParameterValue>>> {
    let mut combinations = vec![BTreeMap::new()];
    for (name, values) in space {
        let mut next = Vec::new();
        for combination in &combinations {
            for value in sorted_values(values) {
                let mut candidate = combination.clone();
                candidate.insert(name.clone(), value);
                next.push(candidate);
            }
        }
        combinations = next;
    }
    Ok(combinations)
}

fn trial_id(
    fingerprint: &str,
    block: &OptimizerBlock,
    pass: usize,
    ordinal: usize,
    values: &BTreeMap<String, ParameterValue>,
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-parameter-trial-v1\0");
    hasher.update(fingerprint.as_bytes());
    hasher.update(block.id.as_bytes());
    hasher.update(pass.to_le_bytes());
    hasher.update(ordinal.to_le_bytes());
    hasher.update(serde_json::to_vec(values)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn evaluate_or_resume<E: TrialEvaluator>(
    config: &ParameterOptimizerConfig,
    block: &OptimizerBlock,
    pass: usize,
    ordinal: usize,
    values: BTreeMap<String, ParameterValue>,
    active_parameters: &BTreeSet<String>,
    payload: &mut OptimizerCheckpointPayload,
    evaluator: &mut E,
) -> Result<TrialRecord> {
    let id = trial_id(
        &payload.optimizer_fingerprint,
        block,
        pass,
        ordinal,
        &values,
    )?;
    if let Some(existing) = payload.completed_trials.get(&id) {
        let mut reused = existing.clone();
        reused.reused_from_checkpoint = true;
        payload.completed_trials.insert(id, reused.clone());
        return Ok(reused);
    }
    let request = TrialRequest {
        trial_id: id.clone(),
        block_id: block.id.clone(),
        pass,
        ordinal,
        scope: block.scope,
        expert: block.expert,
        parameters: values,
        use_external_features: block.use_external_features,
        target_only_outcomes_allowed: false,
    };
    anyhow::ensure!(
        config.target_only_outcomes_excluded && !request.target_only_outcomes_allowed,
        "target-only optimization leakage"
    );
    // A combination can be individually well-typed yet violate a relational
    // dependency (for example lambda_min > lambda_max). Preserve that declared
    // point in the immutable trial sequence, but prune it before production
    // evaluation as an explicit technical-infeasible trial. This avoids both a
    // silent no-op and aborting an otherwise valid bounded search.
    let mut evaluation = match validate_assignment(&request.parameters)
        .and_then(|()| validate_active_dependencies(&request.parameters, active_parameters))
    {
        Ok(()) => evaluator.evaluate(&request)?,
        Err(error) => TrialEvaluation {
            status: TrialStatus::TechnicalFailure,
            technical_reason: Some(format!(
                "parameter_dependency_invalid_before_production: {error:#}"
            )),
            empirical_reason: None,
            metrics: None,
            development_selection_eligible: false,
            empirical_point_estimate_within_limit: None,
            empirical_calibration_power: EmpiricalCalibrationPower::NotAssessed,
            statistical_validation_status: StatisticalValidationStatus::NotEvaluated,
            statistical_default_eligibility: StatisticalDefaultEligibility::NotEvaluated,
            compact_diagnostics: BTreeMap::from([
                (
                    "production_evaluation_started".into(),
                    serde_json::json!(false),
                ),
                ("fallback_used".into(), serde_json::json!(false)),
                ("model_substitution".into(), serde_json::json!(false)),
                ("target_only_outcomes_used".into(), serde_json::json!(false)),
            ]),
        },
    };
    apply_empirical_constraints(config, &mut evaluation);
    let record = TrialRecord {
        request,
        evaluation,
        reused_from_checkpoint: false,
    };
    payload.completed_trials.insert(id, record.clone());
    Ok(record)
}

fn apply_empirical_constraints(
    config: &ParameterOptimizerConfig,
    evaluation: &mut TrialEvaluation,
) {
    evaluation.development_selection_eligible = false;
    evaluation.empirical_point_estimate_within_limit = None;
    evaluation.empirical_calibration_power = EmpiricalCalibrationPower::NotAssessed;
    evaluation.statistical_validation_status = StatisticalValidationStatus::NotEvaluated;
    evaluation.statistical_default_eligibility = StatisticalDefaultEligibility::NotEvaluated;
    if evaluation.status != TrialStatus::Feasible {
        return;
    }
    if evaluation
        .compact_diagnostics
        .get("fallback_used")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
        || evaluation
            .compact_diagnostics
            .get("model_substitution")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    {
        evaluation.status = TrialStatus::TechnicalFailure;
        evaluation.technical_reason =
            Some("prohibited fallback or model substitution in optimizer trial".into());
        return;
    }
    let Some(metrics) = evaluation.metrics.as_ref() else {
        evaluation.status = TrialStatus::NotEvaluable;
        evaluation.empirical_reason = Some("feasible trial did not provide metrics".into());
        return;
    };
    if !config.implementation_smoke_only
        && config
            .objective
            .iter()
            .any(|term| term.metric == ObjectiveMetric::AdjustedEntrapmentFdp)
        && metrics
            .adjusted_entrapment_fdp
            .is_none_or(|fdp| !fdp.is_finite())
    {
        evaluation.status = TrialStatus::NotEvaluable;
        evaluation.empirical_reason =
            Some("adjusted entrapment FDP objective is missing or nonfinite".into());
        return;
    }
    evaluation.development_selection_eligible = true;
    let mut underpowered = false;
    let mut adequately_powered = false;
    for constraint in &config.empirical_entrapment_constraints {
        let count = metrics
            .entrapment_count_by_level
            .get(&constraint.level)
            .copied()
            .unwrap_or(metrics.entrapment_count);
        let fdp = metrics
            .adjusted_entrapment_fdp_by_level
            .get(&constraint.level)
            .copied()
            .unwrap_or(metrics.adjusted_entrapment_fdp);
        let target_count = match constraint.level.as_str() {
            "psm" => metrics.level4_psms,
            "peptide" => metrics.level4_canonical_peptides,
            "peptidoform" => metrics.level4_peptidoforms,
            "protein" => metrics.level4_proteins,
            _ => 0,
        };
        if target_count == 0 {
            evaluation.status = TrialStatus::NotEvaluable;
            evaluation.development_selection_eligible = false;
            evaluation.empirical_reason = Some(format!(
                "{} has no accepted target discoveries",
                constraint.level
            ));
            return;
        }
        let Some(fdp) = fdp.filter(|fdp| fdp.is_finite()) else {
            evaluation.status = TrialStatus::NotEvaluable;
            evaluation.development_selection_eligible = false;
            evaluation.empirical_reason = Some(format!(
                "{} adjusted entrapment FDP is missing or nonfinite",
                constraint.level
            ));
            return;
        };
        let constraint_underpowered = count < constraint.minimum_entrapment_observations_for_power;
        if constraint_underpowered {
            underpowered = true;
        } else {
            adequately_powered = true;
        }
        if fdp > constraint.maximum_adjusted_fdp {
            evaluation.status = TrialStatus::EmpiricallyInfeasible;
            evaluation.development_selection_eligible = false;
            evaluation.empirical_point_estimate_within_limit = Some(false);
            evaluation.empirical_calibration_power = if underpowered {
                EmpiricalCalibrationPower::Underpowered
            } else {
                EmpiricalCalibrationPower::AdequatelyPowered
            };
            evaluation.statistical_validation_status =
                StatisticalValidationStatus::EmpiricallyInfeasible;
            evaluation.empirical_reason = Some(format!(
                "{} adjusted entrapment FDP violates declared maximum",
                constraint.level
            ));
            return;
        }
        evaluation.empirical_point_estimate_within_limit = Some(true);
    }
    if underpowered {
        evaluation.empirical_calibration_power = EmpiricalCalibrationPower::Underpowered;
        evaluation.statistical_validation_status =
            StatisticalValidationStatus::NotEvaluableUnderpowered;
        evaluation.empirical_reason =
            Some("empirical entrapment evidence is below the declared power threshold".into());
        if config.underpowered_trial_policy == UnderpoweredTrialPolicy::NotEvaluable {
            evaluation.status = TrialStatus::NotEvaluable;
            evaluation.development_selection_eligible = false;
        }
    } else if adequately_powered {
        evaluation.empirical_calibration_power = EmpiricalCalibrationPower::AdequatelyPowered;
        evaluation.statistical_validation_status =
            StatisticalValidationStatus::EmpiricallyEvaluable;
    }
}

fn better(config: &ParameterOptimizerConfig, left: &TrialRecord, right: &TrialRecord) -> bool {
    let (Some(left_metrics), Some(right_metrics)) = (
        left.evaluation.metrics.as_ref(),
        right.evaluation.metrics.as_ref(),
    ) else {
        return left.evaluation.metrics.is_some();
    };
    for term in &config.objective {
        use ObjectiveMetric::*;
        let ordering = match term.metric {
            Level4Proteins => left_metrics
                .level4_proteins
                .cmp(&right_metrics.level4_proteins),
            Level4CanonicalPeptides => left_metrics
                .level4_canonical_peptides
                .cmp(&right_metrics.level4_canonical_peptides),
            Level4Peptidoforms => left_metrics
                .level4_peptidoforms
                .cmp(&right_metrics.level4_peptidoforms),
            Level4Psms => left_metrics.level4_psms.cmp(&right_metrics.level4_psms),
            AdjustedEntrapmentFdp => left_metrics
                .adjusted_entrapment_fdp
                .unwrap_or(f64::INFINITY)
                .total_cmp(
                    &right_metrics
                        .adjusted_entrapment_fdp
                        .unwrap_or(f64::INFINITY),
                ),
            ModelComplexity => left_metrics
                .model_complexity
                .cmp(&right_metrics.model_complexity),
            DeterministicParameterOrder => serde_json::to_string(&left.request.parameters)
                .unwrap_or_default()
                .cmp(&serde_json::to_string(&right.request.parameters).unwrap_or_default()),
        };
        if !ordering.is_eq() {
            return match term.direction {
                ObjectiveDirection::Maximize => ordering.is_gt(),
                ObjectiveDirection::Minimize => ordering.is_lt(),
            };
        }
    }
    false
}

fn accept_transition(
    block: &OptimizerBlock,
    pass: usize,
    best: &TrialRecord,
    payload: &mut OptimizerCheckpointPayload,
) {
    let transition = AcceptedTransition {
        block_id: block.id.clone(),
        pass,
        from_trial: payload.winner_trial_id.clone(),
        to_trial: best.request.trial_id.clone(),
    };
    if !payload.accepted_transitions.iter().any(|existing| {
        existing.block_id == transition.block_id
            && existing.pass == transition.pass
            && existing.to_trial == transition.to_trial
    }) {
        payload.accepted_transitions.push(transition);
    }
}

fn finish_result<E: TrialEvaluator>(
    config: &ParameterOptimizerConfig,
    fingerprint: String,
    payload: OptimizerCheckpointPayload,
    evaluator: &mut E,
    any_exhaustive: bool,
    any_heuristic: bool,
) -> Result<OptimizerRunResult> {
    let mut winner_artifacts = BTreeMap::new();
    if config.materialize_winner {
        for (block, id) in &payload.block_winners {
            if let Some(record) = payload.completed_trials.get(id) {
                if let Some(artifact) = evaluator.materialize_winner(record)? {
                    winner_artifacts.insert(block.clone(), artifact);
                }
            }
        }
    }
    let mut trials = payload
        .completed_trials
        .values()
        .cloned()
        .collect::<Vec<_>>();
    trials.sort_by_key(|record| {
        (
            record.request.block_id.clone(),
            record.request.pass,
            record.request.ordinal,
            record.request.trial_id.clone(),
        )
    });
    let mut scientific_trials = trials.clone();
    for trial in &mut scientific_trials {
        trial.reused_from_checkpoint = false;
    }
    let mut result_hasher = Sha256::new();
    result_hasher.update(b"sage-parameter-optimizer-scientific-result-v1\0");
    result_hasher.update(serde_json::to_vec(&serde_json::json!({
        "optimizer_fingerprint": &fingerprint,
        "outcome": &payload.outcome,
        "requested_parameter_space": &config.blocks,
        "block_order": &config.block_order,
        "objective": &config.objective,
        "empirical_constraints": &config.empirical_entrapment_constraints,
        "trials": &scientific_trials,
        "accepted_transitions": &payload.accepted_transitions,
        "winner_trial_id": &payload.winner_trial_id,
        "block_winners": &payload.block_winners,
        "winner_artifacts": &winner_artifacts,
    }))?);
    let scientific_result_sha256 = format!("{:x}", result_hasher.finalize());
    let powered_trial_count = trials
        .iter()
        .filter(|trial| {
            trial.evaluation.empirical_calibration_power
                == EmpiricalCalibrationPower::AdequatelyPowered
        })
        .count();
    let underpowered_trial_count = trials
        .iter()
        .filter(|trial| {
            trial.evaluation.empirical_calibration_power == EmpiricalCalibrationPower::Underpowered
        })
        .count();
    let empirical_power_not_assessed_trial_count = trials
        .len()
        .saturating_sub(powered_trial_count + underpowered_trial_count);
    Ok(OptimizerRunResult {
        schema_version: PARAMETER_OPTIMIZER_SCHEMA_VERSION,
        optimizer_fingerprint: fingerprint,
        scientific_result_sha256,
        parameter_binding_coverage: parameter_production_bindings(),
        classification: config.classification,
        execution_mode: config.execution_mode,
        outcome: payload.outcome,
        strategy_classification: if any_heuristic { "deterministic_staged_or_coordinate_local_not_global" } else if any_exhaustive { "bounded_grid_global_within_single_declared_block" } else { "not_evaluable" }.into(),
        requested_parameter_space: config.blocks.clone(),
        block_order: config.block_order.clone(),
        resolved_parameters: payload.current_parameters,
        resolved_parameter_sets: payload.resolved_parameter_sets,
        parameter_precedence: vec!["compiled_default".into(), "workflow_default".into(), "per_expert_fixed_override".into(), "per_expert_optimizer_trial".into(), "ensemble_final".into()],
        objective: config.objective.clone(),
        empirical_constraints: config.empirical_entrapment_constraints.clone(),
        underpowered_trial_policy: config.underpowered_trial_policy,
        powered_trial_count,
        underpowered_trial_count,
        empirical_power_not_assessed_trial_count,
        trials,
        accepted_transitions: payload.accepted_transitions,
        winner_trial_id: payload.winner_trial_id,
        block_winners: payload.block_winners,
        winner_artifacts,
        target_only_non_leakage: "target-only outcomes excluded from feasibility, objective, ranking, early stopping, and selection".into(),
        development_only: true,
        independent_evaluation_status: "not_run_reserved_for_step4".into(),
        statistical_default_status: if config.underpowered_trial_policy
            == UnderpoweredTrialPolicy::DevelopmentEligible
        {
            "not_evaluated"
        } else {
            "development_candidate_not_a_default"
        }
        .into(),
        frozen_audit: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn config() -> ParameterOptimizerConfig {
        ParameterOptimizerConfig {
            schema_version: 1,
            enabled: true,
            classification: OptimizationClassification::DevelopmentOnly,
            selected_experts: vec![OptimizerExpert::Moments, OptimizerExpert::Mle],
            expected_expert_configuration_sha256: BTreeMap::new(),
            require_expected_expert_configurations: false,
            compiled_defaults: BTreeMap::from([
                (
                    "moments_purification_factor".into(),
                    ParameterValue::Float(0.1),
                ),
                ("min_null_size".into(), ParameterValue::Integer(100)),
            ]),
            workflow_defaults: BTreeMap::new(),
            fixed_baseline_values: BTreeMap::new(),
            seed: 7,
            maximum_trial_budget: 20,
            maximum_optimization_passes: 2,
            objective: default_objective(),
            fixed_evaluation_threshold: 0.01,
            empirical_entrapment_constraints: vec![],
            underpowered_trial_policy: UnderpoweredTrialPolicy::NotEvaluable,
            entrapment_validation: EntrapmentValidationConfig::default(),
            statistical_validity_contracts: BTreeMap::new(),
            resume: true,
            materialize_winner: true,
            execution_mode: OptimizerExecutionMode::OptimizationAndPostSelection,
            implementation_smoke_only: false,
            production_smoke_only: false,
            require_existing_candidate_pool: true,
            require_existing_raw_annotation_cache: true,
            target_only_outcomes_excluded: true,
            block_order: vec!["moments".into()],
            blocks: vec![OptimizerBlock {
                id: "moments".into(),
                enabled: true,
                scope: ParameterScope::PerExpert,
                expert: Some(OptimizerExpert::Moments),
                strategy: OptimizerStrategy::ExhaustiveGrid,
                structural_comparison: false,
                fixed: BTreeMap::new(),
                space: BTreeMap::from([
                    (
                        "moments_purification_factor".into(),
                        vec![ParameterValue::Float(0.1), ParameterValue::Float(0.2)],
                    ),
                    (
                        "min_null_size".into(),
                        vec![ParameterValue::Integer(100), ParameterValue::Integer(300)],
                    ),
                ]),
                window_search: None,
                use_external_features: false,
                max_trials: Some(4),
                max_passes: None,
            }],
        }
    }

    fn identity() -> OptimizerIdentity {
        OptimizerIdentity {
            schema_version: 1,
            execution_mode: OptimizerExecutionMode::OptimizationAndPostSelection,
            dataset_identity: "dataset".into(),
            candidate_pool_identity: "candidate".into(),
            raw_annotation_cache_identity: "raw".into(),
            calibrated_annotation_identity: None,
            model_artifact_schema: 2,
            optimizer_schema: PARAMETER_OPTIMIZER_SCHEMA_VERSION,
            optimizer_source_sha256: PARAMETER_OPTIMIZER_IMPLEMENTATION_SOURCE_SHA256.into(),
            source_configuration_sha256: "config".into(),
            catalog_sha256: "catalog".into(),
            entrapment_partition_identity: None,
        }
    }

    #[derive(Default)]
    struct Evaluator {
        calls: Vec<String>,
    }
    impl TrialEvaluator for Evaluator {
        fn evaluate(&mut self, request: &TrialRequest) -> Result<TrialEvaluation> {
            assert!(!request.target_only_outcomes_allowed);
            self.calls.push(request.trial_id.clone());
            let purification = request
                .parameters
                .get("moments_purification_factor")
                .and_then(ParameterValue::as_f64)
                .unwrap_or(0.1);
            Ok(TrialEvaluation {
                status: TrialStatus::Feasible,
                technical_reason: None,
                empirical_reason: None,
                metrics: Some(TrialMetrics {
                    level4_proteins: (purification * 100.0) as usize,
                    level4_canonical_peptides: 10,
                    level4_peptidoforms: 10,
                    level4_psms: 10,
                    adjusted_entrapment_fdp: Some(0.005),
                    entrapment_count: 5,
                    adjusted_entrapment_fdp_by_level: BTreeMap::new(),
                    entrapment_count_by_level: BTreeMap::new(),
                    model_complexity: 1,
                }),
                development_selection_eligible: false,
                empirical_point_estimate_within_limit: None,
                empirical_calibration_power: EmpiricalCalibrationPower::NotAssessed,
                statistical_validation_status: StatisticalValidationStatus::NotEvaluated,
                statistical_default_eligibility: StatisticalDefaultEligibility::NotEvaluated,
                compact_diagnostics: BTreeMap::new(),
            })
        }
        fn materialize_winner(
            &mut self,
            record: &TrialRecord,
        ) -> Result<Option<serde_json::Value>> {
            Ok(Some(serde_json::json!({"winner": record.request.trial_id})))
        }
    }

    struct AuditBlindEvaluator {
        hidden_audit_labels: Vec<bool>,
        inner: Evaluator,
    }

    impl TrialEvaluator for AuditBlindEvaluator {
        fn evaluate(&mut self, request: &TrialRequest) -> Result<TrialEvaluation> {
            // The optimizer evaluator interface intentionally exposes no
            // audit-label argument. Keep the synthetic labels alive solely to
            // prove that changing them cannot affect selection metrics.
            let _hidden_label_count = self.hidden_audit_labels.len();
            self.inner.evaluate(request)
        }

        fn materialize_winner(
            &mut self,
            record: &TrialRecord,
        ) -> Result<Option<serde_json::Value>> {
            self.inner.materialize_winner(record)
        }
    }

    #[test]
    fn execution_mode_is_backward_compatible_and_changes_identity() {
        let mut value = serde_json::to_value(config()).unwrap();
        value.as_object_mut().unwrap().remove("execution_mode");
        value["schema_version"] = serde_json::json!(1);
        let legacy: ParameterOptimizerConfig = serde_json::from_value(value).unwrap();
        assert_eq!(
            legacy.execution_mode,
            OptimizerExecutionMode::OptimizationAndPostSelection
        );
        legacy.validate().unwrap();

        let normal = config();
        let mut only = normal.clone();
        only.schema_version = PARAMETER_OPTIMIZER_SCHEMA_VERSION;
        only.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        let mut only_identity = identity();
        only_identity.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        assert_ne!(
            optimizer_fingerprint(&identity(), &normal).unwrap(),
            optimizer_fingerprint(&only_identity, &only).unwrap()
        );

        let directory = temp("execution-mode-checkpoint");
        std::fs::create_dir_all(&directory).unwrap();
        let checkpoint = directory.join("checkpoint.json");
        run_optimizer(&normal, &identity(), &checkpoint, &mut Evaluator::default()).unwrap();
        let error = run_optimizer(
            &only,
            &only_identity,
            &checkpoint,
            &mut Evaluator::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("fingerprint mismatch"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn entrapment_partition_mode_is_explicit_backward_compatible_and_checkpoint_bound() {
        let mut legacy_value = serde_json::to_value(config()).unwrap();
        legacy_value
            .as_object_mut()
            .unwrap()
            .remove("entrapment_validation");
        let legacy: ParameterOptimizerConfig = serde_json::from_value(legacy_value).unwrap();
        assert_eq!(
            legacy.entrapment_validation.mode,
            EntrapmentValidationMode::FullPopulationDevelopment
        );

        let mut selection = config();
        selection.schema_version = PARAMETER_OPTIMIZER_SCHEMA_VERSION;
        selection.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        selection.entrapment_validation = EntrapmentValidationConfig {
            mode: EntrapmentValidationMode::SelectionAudit,
            partition_schema_version: 1,
            seed: 19,
            salt: "dataset-local-selection-audit-v1".into(),
            selection_fraction: 0.7,
            audit_fraction: 0.3,
            require_existing_partition: true,
        };
        selection.validate().unwrap();

        let mut bad = selection.clone();
        bad.schema_version = 3;
        assert!(bad
            .validate()
            .unwrap_err()
            .to_string()
            .contains("schema_version 4"));
        let mut bad = selection.clone();
        bad.entrapment_validation.salt.clear();
        assert!(bad.validate().unwrap_err().to_string().contains("nonempty"));
        let mut bad = selection.clone();
        bad.entrapment_validation.audit_fraction = 0.4;
        assert!(bad
            .validate()
            .unwrap_err()
            .to_string()
            .contains("sum to one"));

        let mut first_identity = identity();
        first_identity.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        first_identity.entrapment_partition_identity = Some("partition-a".into());
        let mut second_identity = first_identity.clone();
        second_identity.entrapment_partition_identity = Some("partition-b".into());
        assert_eq!(
            first_identity.candidate_pool_identity,
            second_identity.candidate_pool_identity
        );
        assert_eq!(
            first_identity.raw_annotation_cache_identity,
            second_identity.raw_annotation_cache_identity
        );
        assert_ne!(
            optimizer_fingerprint(&first_identity, &selection).unwrap(),
            optimizer_fingerprint(&second_identity, &selection).unwrap()
        );

        let directory = temp("partition-checkpoint-identity");
        std::fs::create_dir_all(&directory).unwrap();
        let checkpoint = directory.join("checkpoint.json");
        let result = run_optimizer(
            &selection,
            &first_identity,
            &checkpoint,
            &mut Evaluator::default(),
        )
        .unwrap();
        assert!(result.frozen_audit.is_none());
        let checkpoint_text = std::fs::read_to_string(&checkpoint).unwrap();
        assert!(!checkpoint_text.contains("frozen_audit"));
        assert!(!checkpoint_text.contains("audit_entrapment"));
        let error = run_optimizer(
            &selection,
            &second_identity,
            &checkpoint,
            &mut Evaluator::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("fingerprint mismatch"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn hidden_audit_labels_cannot_change_trial_ranking_or_winner() {
        let mut cfg = config();
        cfg.schema_version = PARAMETER_OPTIMIZER_SCHEMA_VERSION;
        cfg.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        cfg.entrapment_validation = EntrapmentValidationConfig {
            mode: EntrapmentValidationMode::SelectionAudit,
            partition_schema_version: 1,
            seed: 19,
            salt: "audit-blindness".into(),
            selection_fraction: 0.5,
            audit_fraction: 0.5,
            require_existing_partition: false,
        };
        let mut partition_identity = identity();
        partition_identity.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        partition_identity.entrapment_partition_identity = Some("frozen-partition".into());
        let first_dir = temp("audit-blind-a");
        let second_dir = temp("audit-blind-b");
        std::fs::create_dir_all(&first_dir).unwrap();
        std::fs::create_dir_all(&second_dir).unwrap();
        let first = run_optimizer(
            &cfg,
            &partition_identity,
            &first_dir.join("checkpoint.json"),
            &mut AuditBlindEvaluator {
                hidden_audit_labels: vec![false, false, true],
                inner: Evaluator::default(),
            },
        )
        .unwrap();
        let second = run_optimizer(
            &cfg,
            &partition_identity,
            &second_dir.join("checkpoint.json"),
            &mut AuditBlindEvaluator {
                hidden_audit_labels: vec![true, true, false, true],
                inner: Evaluator::default(),
            },
        )
        .unwrap();
        assert_eq!(first.winner_trial_id, second.winner_trial_id);
        assert_eq!(
            first.scientific_result_sha256,
            second.scientific_result_sha256
        );
        std::fs::remove_dir_all(first_dir).unwrap();
        std::fs::remove_dir_all(second_dir).unwrap();
    }

    #[test]
    fn optimization_only_accepts_ensemble_and_more_than_sixteen_trials() {
        let mut cfg = config();
        cfg.schema_version = PARAMETER_OPTIMIZER_SCHEMA_VERSION;
        cfg.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        cfg.maximum_trial_budget = 32;
        cfg.selected_experts = vec![OptimizerExpert::Moments, OptimizerExpert::Ensemble];
        cfg.block_order = vec!["ensemble".into()];
        cfg.blocks = vec![OptimizerBlock {
            id: "ensemble".into(),
            enabled: true,
            scope: ParameterScope::EnsembleFinal,
            expert: Some(OptimizerExpert::Ensemble),
            strategy: OptimizerStrategy::ExhaustiveGrid,
            structural_comparison: false,
            fixed: BTreeMap::from([
                (
                    "final_evidence_space".into(),
                    ParameterValue::String("pep".into()),
                ),
                (
                    "ensemble_pep_combiner".into(),
                    ParameterValue::String("weighted_mean".into()),
                ),
            ]),
            space: BTreeMap::from([(
                "ensemble_weight_moments".into(),
                (1..=17)
                    .map(|value| ParameterValue::Float(value as f64 / 10.0))
                    .collect(),
            )]),
            window_search: None,
            use_external_features: false,
            max_trials: Some(17),
            max_passes: None,
        }];
        cfg.validate().unwrap();

        let directory = temp("optimization-only-over-sixteen");
        std::fs::create_dir_all(&directory).unwrap();
        let mut evaluator = Evaluator::default();
        let mut optimizer_identity = identity();
        optimizer_identity.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        let result = run_optimizer(
            &cfg,
            &optimizer_identity,
            &directory.join("checkpoint.json"),
            &mut evaluator,
        )
        .unwrap();
        assert_eq!(
            result.execution_mode,
            OptimizerExecutionMode::OptimizationOnly
        );
        assert_eq!(evaluator.calls.len(), 17);
        assert!(evaluator.calls.iter().all(|_| true));
        assert_eq!(result.trials.len(), 17);

        cfg.production_smoke_only = true;
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("cannot assemble or optimize Ensemble"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn ensemble_dependencies_prune_parameters_dormant_for_the_decision_stream() {
        let p_value_settings = BTreeMap::from([
            (
                "final_evidence_space".into(),
                ParameterValue::String("p_value".into()),
            ),
            (
                "ensemble_p_combiner".into(),
                ParameterValue::String("second_best".into()),
            ),
            (
                "ensemble_pep_combiner".into(),
                ParameterValue::String("median".into()),
            ),
            (
                "ensemble_cauchy_penalty".into(),
                ParameterValue::Float(1.0224),
            ),
            ("ensemble_pep_trim_frac".into(), ParameterValue::Float(0.1)),
            ("ensemble_weight_moments".into(), ParameterValue::Float(0.5)),
        ]);

        assert!(validate_active_dependencies(
            &p_value_settings,
            &BTreeSet::from(["ensemble_p_combiner".into()])
        )
        .is_ok());
        for dormant in [
            "ensemble_pep_combiner",
            "ensemble_cauchy_penalty",
            "ensemble_pep_trim_frac",
            "ensemble_weight_moments",
        ] {
            let error =
                validate_active_dependencies(&p_value_settings, &BTreeSet::from([dormant.into()]))
                    .unwrap_err()
                    .to_string();
            assert!(error.contains("dependency violated"), "{dormant}: {error}");
        }

        let mut pep_settings = p_value_settings;
        pep_settings.insert(
            "final_evidence_space".into(),
            ParameterValue::String("pep".into()),
        );
        pep_settings.insert(
            "ensemble_pep_combiner".into(),
            ParameterValue::String("weighted_median".into()),
        );
        assert!(validate_active_dependencies(
            &pep_settings,
            &BTreeSet::from([
                "ensemble_pep_combiner".into(),
                "ensemble_weight_moments".into(),
            ])
        )
        .is_ok());
        assert!(validate_active_dependencies(
            &pep_settings,
            &BTreeSet::from(["ensemble_p_combiner".into()])
        )
        .is_err());
    }

    #[test]
    fn dormant_ensemble_weight_is_pruned_before_production_evaluation() {
        let path = temp("dormant-ensemble-weight");
        std::fs::create_dir_all(&path).unwrap();
        let mut cfg = config();
        cfg.maximum_trial_budget = 1;
        cfg.selected_experts = vec![OptimizerExpert::Moments, OptimizerExpert::Ensemble];
        cfg.block_order = vec!["ensemble".into()];
        cfg.fixed_baseline_values.extend([
            (
                "final_evidence_space".into(),
                ParameterValue::String("p_value".into()),
            ),
            (
                "ensemble_p_combiner".into(),
                ParameterValue::String("second_best".into()),
            ),
            (
                "ensemble_pep_combiner".into(),
                ParameterValue::String("median".into()),
            ),
        ]);
        cfg.blocks = vec![OptimizerBlock {
            id: "ensemble".into(),
            enabled: true,
            scope: ParameterScope::EnsembleFinal,
            expert: Some(OptimizerExpert::Ensemble),
            strategy: OptimizerStrategy::ExhaustiveGrid,
            structural_comparison: false,
            fixed: BTreeMap::new(),
            space: BTreeMap::from([(
                "ensemble_weight_moments".into(),
                vec![ParameterValue::Float(0.5)],
            )]),
            window_search: None,
            use_external_features: false,
            max_trials: Some(1),
            max_passes: None,
        }];
        cfg.validate().unwrap();

        let mut evaluator = Evaluator::default();
        let result = run_optimizer(
            &cfg,
            &identity(),
            &path.join("checkpoint.json"),
            &mut evaluator,
        )
        .unwrap();
        assert!(evaluator.calls.is_empty());
        assert_eq!(result.trials.len(), 1);
        assert_eq!(
            result.trials[0].evaluation.status,
            TrialStatus::TechnicalFailure
        );
        assert!(result.trials[0]
            .evaluation
            .technical_reason
            .as_deref()
            .is_some_and(
                |reason| reason.contains("parameter_dependency_invalid_before_production")
            ));
        assert_eq!(
            result.trials[0]
                .evaluation
                .compact_diagnostics
                .get("production_evaluation_started"),
            Some(&serde_json::json!(false))
        );
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn optimization_only_represents_all_experts_without_admission_gates() {
        let mut cfg = config();
        cfg.schema_version = PARAMETER_OPTIMIZER_SCHEMA_VERSION;
        cfg.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        cfg.maximum_trial_budget = 64;
        cfg.selected_experts = vec![
            OptimizerExpert::Moments,
            OptimizerExpert::Mle,
            OptimizerExpert::LowerOrder,
            OptimizerExpert::MsfdrSeeded,
            OptimizerExpert::Msfdr1Smix,
            OptimizerExpert::Msfdr2Smix,
            OptimizerExpert::Nokoi,
            OptimizerExpert::Ensemble,
        ];
        let declarations = [
            (
                OptimizerExpert::Moments,
                ParameterScope::PerExpert,
                "moments_purification_factor",
                ParameterValue::Float(0.2),
            ),
            (
                OptimizerExpert::Mle,
                ParameterScope::PerExpert,
                "mle_purification_factor",
                ParameterValue::Float(0.2),
            ),
            (
                OptimizerExpert::LowerOrder,
                ParameterScope::PerExpert,
                "lower_order_purification_factor",
                ParameterValue::Float(0.25),
            ),
            (
                OptimizerExpert::MsfdrSeeded,
                ParameterScope::PerExpert,
                "msfdr_seeded_top_frac_init",
                ParameterValue::Float(0.2),
            ),
            (
                OptimizerExpert::Msfdr1Smix,
                ParameterScope::PerExpert,
                "msfdr1_top_frac_init",
                ParameterValue::Float(0.2),
            ),
            (
                OptimizerExpert::Msfdr2Smix,
                ParameterScope::PerExpert,
                "msfdr2_top_frac_init",
                ParameterValue::Float(0.2),
            ),
            (
                OptimizerExpert::Nokoi,
                ParameterScope::PerExpert,
                "nokoi_k_folds",
                ParameterValue::Integer(5),
            ),
            (
                OptimizerExpert::Ensemble,
                ParameterScope::EnsembleFinal,
                "ensemble_p_combiner",
                ParameterValue::String("second_best".into()),
            ),
        ];
        cfg.blocks = declarations
            .into_iter()
            .map(|(expert, scope, parameter, value)| OptimizerBlock {
                id: expert.slug().into(),
                enabled: true,
                scope,
                expert: Some(expert),
                strategy: OptimizerStrategy::ExhaustiveGrid,
                structural_comparison: parameter == "ensemble_p_combiner",
                fixed: BTreeMap::new(),
                space: BTreeMap::from([(parameter.into(), vec![value])]),
                window_search: None,
                use_external_features: true,
                max_trials: Some(1),
                max_passes: None,
            })
            .collect();
        cfg.block_order = cfg.blocks.iter().map(|block| block.id.clone()).collect();
        cfg.validate().unwrap();
        assert!(cfg
            .blocks
            .iter()
            .find(|block| block.expert == Some(OptimizerExpert::Msfdr1Smix))
            .unwrap()
            .window_search
            .is_none());
        assert!(cfg.blocks.iter().all(|block| !block
            .space
            .keys()
            .any(|name| name.contains("admission") || name.contains("exclude"))));
    }

    fn temp(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock precedes Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "sage-optimizer-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn optimizer_contract_rejects_unknown_wrong_owner_scope_and_nonoptimizable_fields() {
        let mut cfg = config();
        cfg.blocks[0]
            .space
            .insert("misspelled".into(), vec![ParameterValue::Float(1.0)]);
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("unknown optimizer parameter"));
        let mut cfg = config();
        cfg.blocks[0].space =
            BTreeMap::from([("nokoi_k_folds".into(), vec![ParameterValue::Integer(5)])]);
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("belongs to nokoi"));
        let mut cfg = config();
        cfg.blocks[0].space = BTreeMap::from([(
            "ensemble_p_combiner".into(),
            vec![ParameterValue::String("cauchy".into())],
        )]);
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("wrong scope"));
        let mut cfg = config();
        cfg.blocks[0].scope = ParameterScope::NumericalOnly;
        cfg.blocks[0].expert = Some(OptimizerExpert::Msfdr1Smix);
        cfg.selected_experts.push(OptimizerExpert::Msfdr1Smix);
        cfg.blocks[0].space =
            BTreeMap::from([("mix_em_tol".into(), vec![ParameterValue::Float(1e-6)])]);
        let error = cfg.validate().unwrap_err().to_string();
        assert!(
            error.contains("not eligible") || error.contains("deliberately deferred"),
            "{error}"
        );
    }

    #[test]
    fn covariates_require_a_statistical_validity_contract() {
        let mut cfg = config();
        cfg.blocks[0].structural_comparison = true;
        cfg.blocks[0].space = BTreeMap::from([(
            "psm_q_covariate".into(),
            vec![ParameterValue::String("matched_peaks".into())],
        )]);
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("statistical-validity contract"));
        cfg.statistical_validity_contracts.insert(
            "psm_q_covariate:statistical_validity".into(),
            "cross-fitted null-independence simulation v1".into(),
        );
        cfg.validate().unwrap();
    }

    #[test]
    fn msfdr1_window_and_zero_selected_voter_weight_fail_closed() {
        let mut cfg = config();
        cfg.blocks[0].expert = Some(OptimizerExpert::Msfdr1Smix);
        cfg.selected_experts.push(OptimizerExpert::Msfdr1Smix);
        cfg.blocks[0].space = BTreeMap::from([(
            "msfdr1_top_frac_init".into(),
            vec![ParameterValue::Float(0.2)],
        )]);
        cfg.blocks[0].window_search = Some(OptimizerWindowSearch {
            strategy: "explicit_grid".into(),
            min_rank_range: [2, 2],
            max_rank_range: [2, 2],
        });
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("fixed at rank 1-1"));
        let mut cfg = config();
        cfg.selected_experts.push(OptimizerExpert::Ensemble);
        cfg.block_order = vec!["ensemble".into()];
        cfg.blocks = vec![OptimizerBlock {
            id: "ensemble".into(),
            enabled: true,
            scope: ParameterScope::EnsembleFinal,
            expert: Some(OptimizerExpert::Ensemble),
            strategy: OptimizerStrategy::ExhaustiveGrid,
            structural_comparison: false,
            fixed: BTreeMap::new(),
            space: BTreeMap::from([(
                "ensemble_weight_moments".into(),
                vec![ParameterValue::Float(0.0)],
            )]),
            window_search: None,
            use_external_features: false,
            max_trials: Some(1),
            max_passes: None,
        }];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn precedence_is_compiled_then_workflow_then_fixed_then_trial() {
        let mut cfg = config();
        cfg.compiled_defaults
            .insert("min_null_size".into(), ParameterValue::Integer(1));
        cfg.workflow_defaults
            .insert("min_null_size".into(), ParameterValue::Integer(2));
        cfg.fixed_baseline_values
            .insert("min_null_size".into(), ParameterValue::Integer(3));
        let baseline = resolve_baseline(&cfg);
        assert_eq!(
            baseline.get("min_null_size"),
            Some(&ParameterValue::Integer(3))
        );
        let mut resolved = baseline;
        resolved.extend(cfg.blocks[0].fixed.clone());
        resolved.insert("min_null_size".into(), ParameterValue::Integer(100));
        assert_eq!(
            resolved.get("min_null_size"),
            Some(&ParameterValue::Integer(100))
        );
    }

    #[test]
    fn exhaustive_order_winner_and_exact_resume_are_deterministic() {
        let path = temp("resume");
        std::fs::create_dir_all(&path).unwrap();
        let checkpoint = path.join("checkpoint.json");
        let mut first = Evaluator::default();
        let result1 = run_optimizer(&config(), &identity(), &checkpoint, &mut first).unwrap();
        assert_eq!(first.calls.len(), 4);
        assert_eq!(result1.outcome, OptimizerOutcome::ExhaustiveBoundedOptimum);
        assert!(!result1.winner_artifacts.is_empty());
        let order1 = result1
            .trials
            .iter()
            .map(|trial| trial.request.trial_id.clone())
            .collect::<Vec<_>>();
        let mut second = Evaluator::default();
        let result2 = run_optimizer(&config(), &identity(), &checkpoint, &mut second).unwrap();
        assert!(
            second.calls.is_empty(),
            "completed trials must be reused exactly"
        );
        let order2 = result2
            .trials
            .iter()
            .map(|trial| trial.request.trial_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(order1, order2);
        assert_eq!(result1.winner_trial_id, result2.winner_trial_id);
        assert_eq!(
            result1.scientific_result_sha256,
            result2.scientific_result_sha256
        );
        std::fs::remove_dir_all(path).unwrap();
    }

    struct InterruptAfter {
        successful_before_interrupt: usize,
        calls: usize,
    }

    impl TrialEvaluator for InterruptAfter {
        fn evaluate(&mut self, request: &TrialRequest) -> Result<TrialEvaluation> {
            if self.calls == self.successful_before_interrupt {
                anyhow::bail!("controlled test interruption");
            }
            self.calls += 1;
            Evaluator::default().evaluate(request)
        }
    }

    #[test]
    fn controlled_interruption_resumes_only_missing_trials() {
        let path = temp("controlled-interruption");
        std::fs::create_dir_all(&path).unwrap();
        let checkpoint = path.join("checkpoint.json");
        let mut interrupted = InterruptAfter {
            successful_before_interrupt: 2,
            calls: 0,
        };
        assert!(run_optimizer(&config(), &identity(), &checkpoint, &mut interrupted).is_err());
        assert_eq!(interrupted.calls, 2);
        let mut resumed = Evaluator::default();
        let result = run_optimizer(&config(), &identity(), &checkpoint, &mut resumed).unwrap();
        assert_eq!(resumed.calls.len(), 2);
        assert_eq!(
            result
                .trials
                .iter()
                .filter(|trial| trial.reused_from_checkpoint)
                .count(),
            2
        );
        assert_eq!(result.outcome, OptimizerOutcome::ExhaustiveBoundedOptimum);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn corrupt_or_incompatible_checkpoint_is_rejected() {
        let path = temp("corrupt");
        std::fs::create_dir_all(&path).unwrap();
        let checkpoint = path.join("checkpoint.json");
        let mut evaluator = Evaluator::default();
        run_optimizer(&config(), &identity(), &checkpoint, &mut evaluator).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&checkpoint).unwrap()).unwrap();
        value["payload"]["optimizer_fingerprint"] = "wrong".into();
        write_json_atomic(&checkpoint, &value).unwrap();
        assert!(load_checkpoint(&checkpoint, "expected").is_err());
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn staged_coordinate_is_local_and_deterministic_without_expert_leakage() {
        let mut cfg = config();
        cfg.blocks[0].strategy = OptimizerStrategy::StagedCoordinate;
        cfg.blocks[0].max_trials = None;
        let path = temp("coordinate");
        std::fs::create_dir_all(&path).unwrap();
        let mut evaluator = Evaluator::default();
        let result = run_optimizer(
            &cfg,
            &identity(),
            &path.join("checkpoint.json"),
            &mut evaluator,
        )
        .unwrap();
        assert_eq!(result.outcome, OptimizerOutcome::CompletedHeuristicLocal);
        assert_eq!(
            result.strategy_classification,
            "deterministic_staged_or_coordinate_local_not_global"
        );
        assert!(result
            .trials
            .iter()
            .all(|trial| trial.request.expert == Some(OptimizerExpert::Moments)));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn fdr_override_materialization_keeps_expert_and_ensemble_values_separate() {
        let mut expert = sage_core::input::FdrOptions::default();
        apply_fdr_overrides(
            &mut expert,
            &BTreeMap::from([(
                "psm_q_method".into(),
                ParameterValue::String("storey".into()),
            )]),
        )
        .unwrap();
        let mut ensemble = sage_core::input::FdrOptions::default();
        apply_fdr_overrides(
            &mut ensemble,
            &BTreeMap::from([("psm_q_method".into(), ParameterValue::String("bh".into()))]),
        )
        .unwrap();
        assert_ne!(
            serde_json::to_value(expert).unwrap()["psm_q_method"],
            serde_json::to_value(ensemble).unwrap()["psm_q_method"]
        );
    }

    #[test]
    fn catalog_is_valid_json_and_covers_every_runtime_contract() {
        let value: serde_json::Value = serde_json::from_str(include_str!(
            "../../../validation/statistical_conformance/parameter_catalog.json"
        ))
        .unwrap();
        assert_eq!(value["schema_version"], 2);
        let serialized = serde_json::to_string(&value).unwrap();
        for contract in parameter_contracts() {
            assert!(
                serialized.contains(&format!("\"{}\"", contract.name)),
                "catalog is missing {}",
                contract.name
            );
        }
    }

    fn representative_value(contract: &ParameterContract) -> ParameterValue {
        match contract.kind {
            ParameterKind::Boolean => ParameterValue::Bool(true),
            ParameterKind::Integer => {
                ParameterValue::Integer(contract.minimum.unwrap_or(2.0).ceil() as i64)
            }
            ParameterKind::Float => {
                let low = contract.minimum.unwrap_or(0.25);
                let value = match contract.maximum {
                    Some(high) if high > low => low + (high - low) * 0.5,
                    _ if low == 0.0 => 0.25,
                    _ => low,
                };
                ParameterValue::Float(value)
            }
            ParameterKind::Enumeration => ParameterValue::String(contract.enum_values[0].into()),
        }
    }

    fn json_path<'a>(root: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
        path.split('.')
            .try_fold(root, |cursor, part| cursor.get(part))
    }

    #[test]
    fn production_binding_coverage_is_complete_and_materializable() {
        let catalog: serde_json::Value = serde_json::from_str(include_str!(
            "../../../validation/statistical_conformance/parameter_catalog.json"
        ))
        .unwrap();
        let templates = catalog["record_templates"].as_object().unwrap();
        let bindings = parameter_production_bindings();
        let contracts = parameter_contracts();
        for record in catalog["parameters"].as_array().unwrap() {
            let template = templates[record["template"].as_str().unwrap()]
                .as_object()
                .unwrap();
            let classification = record
                .get("classification")
                .or_else(|| template.get("classification"))
                .and_then(serde_json::Value::as_str)
                .unwrap();
            let eligible = record
                .get("eligible_for_internal_optimization")
                .or_else(|| template.get("eligible_for_internal_optimization"))
                .and_then(serde_json::Value::as_bool)
                .unwrap();
            if eligible && matches!(classification, "A" | "B") {
                let name = record["canonical_name"].as_str().unwrap();
                let binding = bindings
                    .iter()
                    .find(|binding| binding.canonical_name == name)
                    .unwrap_or_else(|| panic!("catalog candidate {name} has no binding record"));
                assert!(
                    binding.currently_executable && !binding.deliberately_deferred,
                    "catalog candidate {name} is advertised executable without production binding"
                );
            }
        }
        for binding in bindings
            .iter()
            .filter(|binding| binding.currently_executable)
        {
            if let Some(contract) = contracts
                .iter()
                .find(|contract| contract.name == binding.canonical_name)
            {
                let value = representative_value(contract);
                let mut options = sage_core::input::FdrOptions::default();
                apply_fdr_overrides(
                    &mut options,
                    &BTreeMap::from([(binding.canonical_name.clone(), value.clone())]),
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "production setter failed for {}: {error:#}",
                        binding.canonical_name
                    )
                });
                let materialized = serde_json::to_value(options).unwrap();
                assert_eq!(
                    json_path(&materialized, &binding.canonical_name),
                    Some(&value.to_json()),
                    "production setter did not materialize {}",
                    binding.canonical_name
                );
            } else {
                assert!(
                    binding.canonical_name.ends_with("_null_window"),
                    "non-FdrOptions binding {} is not a model-local window",
                    binding.canonical_name
                );
            }
        }
    }

    #[test]
    fn deliberately_deferred_binding_fails_closed_in_active_block() {
        let mut cfg = config();
        cfg.blocks[0].scope = ParameterScope::Physical;
        cfg.blocks[0].space = BTreeMap::from([(
            "physical_rescue.anchor_max_pep".into(),
            vec![ParameterValue::Float(0.05)],
        )]);
        let error = cfg.validate().unwrap_err().to_string();
        assert!(error.contains("deliberately deferred"), "{error}");
    }

    #[test]
    fn configuration_values_reach_resolved_fdr_settings() {
        let values = BTreeMap::from([
            ("moments_robust_fit".into(), ParameterValue::Bool(true)),
            ("moments_winsor_upper_q".into(), ParameterValue::Float(0.90)),
            (
                "lower_order_purification_factor".into(),
                ParameterValue::Float(0.25),
            ),
            (
                "lo_stratify".into(),
                ParameterValue::String("global".into()),
            ),
            (
                "lo_evalue_candidate_count_power".into(),
                ParameterValue::Float(1.0),
            ),
            ("lo_evalue_scale".into(), ParameterValue::Float(1.0)),
            (
                "lo_tev_transform".into(),
                ParameterValue::String("neg_log_e".into()),
            ),
            (
                "lo_tnm_extrapolation_strength".into(),
                ParameterValue::Float(1.85),
            ),
            ("msfdr_multistart".into(), ParameterValue::Integer(2)),
            ("nokoi_k_folds".into(), ParameterValue::Integer(5)),
            ("nokoi_l1_lambda_min".into(), ParameterValue::Float(0.2)),
            ("nokoi_l1_lambda_max".into(), ParameterValue::Float(1.0)),
            ("nokoi_l1_lambda_steps".into(), ParameterValue::Integer(10)),
            (
                "ensemble_p_combiner".into(),
                ParameterValue::String("second_best".into()),
            ),
            (
                "ensemble_cauchy_penalty".into(),
                ParameterValue::Float(1.0224),
            ),
            (
                "ensemble_pep_combiner".into(),
                ParameterValue::String("median".into()),
            ),
            (
                "psm_q_method".into(),
                ParameterValue::String("storey".into()),
            ),
            (
                "peptide_q_method".into(),
                ParameterValue::String("storey".into()),
            ),
            (
                "protein_q_method".into(),
                ParameterValue::String("storey".into()),
            ),
            ("psm_q_covariate_bins".into(), ParameterValue::Integer(5)),
            (
                "peptide_q_covariate_bins".into(),
                ParameterValue::Integer(5),
            ),
            (
                "protein_q_covariate_bins".into(),
                ParameterValue::Integer(5),
            ),
            (
                "psm_q_covariate_weight_strength".into(),
                ParameterValue::Float(1.0),
            ),
            (
                "peptide_q_covariate_weight_strength".into(),
                ParameterValue::Float(0.75),
            ),
            (
                "protein_q_covariate_weight_strength".into(),
                ParameterValue::Float(0.75),
            ),
        ]);
        let mut options = sage_core::input::FdrOptions::default();
        apply_fdr_overrides(&mut options, &values).unwrap();
        let settings = sage_core::input::FdrSettings::from(options);
        assert!(settings.moments_robust_fit);
        assert_eq!(settings.moments_winsor_upper_q, 0.90);
        assert_eq!(settings.lower_order_purification_factor, 0.25);
        assert_eq!(settings.lo_evalue_candidate_count_power, 1.0);
        assert_eq!(settings.lo_evalue_scale, 1.0);
        assert_eq!(settings.lo_tnm_extrapolation_strength, 1.85);
        assert_eq!(settings.msfdr_multistart, 2);
        assert_eq!(settings.nokoi_k_folds, 5);
        assert_eq!(settings.nokoi_l1_lambda_min, 0.2);
        assert_eq!(settings.nokoi_l1_lambda_max, 1.0);
        assert_eq!(settings.nokoi_l1_lambda_steps, 10);
        assert_eq!(settings.ensemble_cauchy_penalty, 1.0224);
        assert_eq!(settings.psm_q_covariate_bins, 5);
        assert_eq!(settings.peptide_q_covariate_bins, 5);
        assert_eq!(settings.protein_q_covariate_bins, 5);
        assert_eq!(settings.psm_q_covariate_weight_strength, 1.0);
        assert_eq!(settings.peptide_q_covariate_weight_strength, 0.75);
        assert_eq!(settings.protein_q_covariate_weight_strength, 0.75);
        assert_eq!(format!("{:?}", settings.lo_stratify), "Global");
        assert_eq!(format!("{:?}", settings.lo_tev_transform), "NegLogE");
        assert_eq!(format!("{:?}", settings.ensemble_p_combiner), "SecondBest");
        assert_eq!(format!("{:?}", settings.ensemble_pep_combiner), "Median");
        assert_eq!(format!("{:?}", settings.psm_q_method), "Storey");
        assert_eq!(format!("{:?}", settings.peptide_q_method), "Storey");
        assert_eq!(format!("{:?}", settings.protein_q_method), "Storey");
    }

    #[test]
    fn lower_order_tev_aliases_load_but_serialize_canonically() {
        use sage_core::input::LoTevTransform;

        let shifted: LoTevTransform = serde_json::from_str("\"log_1000_over_e\"").unwrap();
        let scaled: LoTevTransform = serde_json::from_str("\"scaled_log_1000_over_e\"").unwrap();
        assert_eq!(shifted, LoTevTransform::Log1000OverE);
        assert_eq!(scaled, LoTevTransform::ScaledLog1000OverE);
        assert_eq!(
            serde_json::to_string(&shifted).unwrap(),
            "\"log1000_over_e\""
        );
        assert_eq!(
            serde_json::to_string(&scaled).unwrap(),
            "\"scaled_log1000_over_e\""
        );
    }

    #[test]
    fn reparameterizations_and_unused_multistart_are_not_yield_variables() {
        for (expert, structural, name, value) in [
            (
                OptimizerExpert::LowerOrder,
                false,
                "lo_evalue_scale",
                ParameterValue::Float(0.5),
            ),
            (
                OptimizerExpert::LowerOrder,
                true,
                "lo_tev_transform",
                ParameterValue::String("log1000_over_e".into()),
            ),
            (
                OptimizerExpert::MsfdrSeeded,
                false,
                "msfdr_multistart",
                ParameterValue::Integer(2),
            ),
        ] {
            let mut cfg = config();
            cfg.selected_experts = vec![expert];
            cfg.block_order = vec!["audit".into()];
            cfg.blocks = vec![OptimizerBlock {
                id: "audit".into(),
                enabled: true,
                scope: ParameterScope::PerExpert,
                expert: Some(expert),
                strategy: OptimizerStrategy::ExhaustiveGrid,
                structural_comparison: structural,
                fixed: BTreeMap::new(),
                space: BTreeMap::from([(name.into(), vec![value])]),
                window_search: None,
                use_external_features: false,
                max_trials: Some(1),
                max_passes: None,
            }];
            let error = cfg.validate().unwrap_err().to_string();
            assert!(
                error.contains("deliberately deferred")
                    || error.contains("not eligible for internal optimization"),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn invalid_bounds_dependencies_and_grid_budget_fail_closed() {
        let mut cfg = config();
        cfg.blocks[0].max_trials = Some(3);
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("above limit"));

        let mut cfg = config();
        cfg.blocks[0].fixed = BTreeMap::from([
            ("moments_winsor_lower_q".into(), ParameterValue::Float(0.9)),
            ("moments_winsor_upper_q".into(), ParameterValue::Float(0.1)),
        ]);
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("dependency violated"));

        let mut cfg = config();
        cfg.blocks[0].space.insert(
            "moments_purification_factor".into(),
            vec![ParameterValue::Float(1.1)],
        );
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("above maximum"));
    }

    #[test]
    fn invalid_dependency_combination_is_recorded_without_production_evaluation() {
        let path = temp("dependency-pruning");
        std::fs::create_dir_all(&path).unwrap();
        let mut cfg = config();
        cfg.maximum_trial_budget = 2;
        cfg.blocks[0].max_trials = Some(2);
        cfg.compiled_defaults
            .insert("moments_robust_fit".into(), ParameterValue::Bool(true));
        cfg.blocks[0].space = BTreeMap::from([
            (
                "moments_winsor_lower_q".into(),
                vec![ParameterValue::Float(0.1), ParameterValue::Float(0.95)],
            ),
            (
                "moments_winsor_upper_q".into(),
                vec![ParameterValue::Float(0.9)],
            ),
        ]);
        let mut evaluator = Evaluator::default();
        let result = run_optimizer(
            &cfg,
            &identity(),
            &path.join("checkpoint.json"),
            &mut evaluator,
        )
        .unwrap();
        assert_eq!(result.trials.len(), 2);
        assert_eq!(evaluator.calls.len(), 1);
        let invalid = result
            .trials
            .iter()
            .find(|record| record.evaluation.status == TrialStatus::TechnicalFailure)
            .expect("invalid dependency trial was not recorded");
        assert!(invalid.evaluation.technical_reason.as_deref().is_some_and(
            |reason| reason.contains("parameter_dependency_invalid_before_production")
        ));
        assert_eq!(
            invalid
                .evaluation
                .compact_diagnostics
                .get("production_evaluation_started"),
            Some(&serde_json::json!(false))
        );
        assert!(result.winner_trial_id.is_some());
    }

    struct StatusEvaluator {
        status: TrialStatus,
        calls: usize,
    }

    impl TrialEvaluator for StatusEvaluator {
        fn evaluate(&mut self, request: &TrialRequest) -> Result<TrialEvaluation> {
            self.calls += 1;
            assert!(!request.target_only_outcomes_allowed);
            Ok(TrialEvaluation {
                status: self.status.clone(),
                technical_reason: (self.status == TrialStatus::TechnicalFailure)
                    .then(|| "msfdr_mixture_nonidentifiable: coincident components".into()),
                empirical_reason: (self.status == TrialStatus::EmpiricallyInfeasible)
                    .then(|| "declared protein FDP constraint".into()),
                metrics: None,
                development_selection_eligible: false,
                empirical_point_estimate_within_limit: None,
                empirical_calibration_power: EmpiricalCalibrationPower::NotAssessed,
                statistical_validation_status: StatisticalValidationStatus::NotEvaluated,
                statistical_default_eligibility: StatisticalDefaultEligibility::NotEvaluated,
                compact_diagnostics: BTreeMap::from([
                    ("fallback_used".into(), serde_json::json!(false)),
                    ("model_substitution".into(), serde_json::json!(false)),
                    ("p_values_produced".into(), serde_json::json!(false)),
                ]),
            })
        }
    }

    #[test]
    fn invalid_msfdr_trial_is_infeasible_without_fallback_or_substitution() {
        let path = temp("technical-msfdr");
        std::fs::create_dir_all(&path).unwrap();
        let mut evaluator = StatusEvaluator {
            status: TrialStatus::TechnicalFailure,
            calls: 0,
        };
        let result = run_optimizer(
            &config(),
            &identity(),
            &path.join("checkpoint.json"),
            &mut evaluator,
        )
        .unwrap();
        assert_eq!(result.outcome, OptimizerOutcome::NoTechnicallyValidSolution);
        assert!(result.winner_trial_id.is_none());
        assert!(result.trials.iter().all(|trial| {
            trial
                .evaluation
                .technical_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("nonidentifiable"))
                && trial.evaluation.compact_diagnostics["fallback_used"] == false
                && trial.evaluation.compact_diagnostics["model_substitution"] == false
                && trial.evaluation.compact_diagnostics["p_values_produced"] == false
        }));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn empirical_failure_is_distinct_from_technical_failure() {
        let path = temp("empirical");
        std::fs::create_dir_all(&path).unwrap();
        let mut evaluator = StatusEvaluator {
            status: TrialStatus::EmpiricallyInfeasible,
            calls: 0,
        };
        let result = run_optimizer(
            &config(),
            &identity(),
            &path.join("checkpoint.json"),
            &mut evaluator,
        )
        .unwrap();
        assert_eq!(
            result.outcome,
            OptimizerOutcome::NoEmpiricallyFeasibleSolution
        );
        assert!(result.trials.iter().all(|trial| {
            trial.evaluation.status == TrialStatus::EmpiricallyInfeasible
                && trial.evaluation.technical_reason.is_none()
        }));
        std::fs::remove_dir_all(path).unwrap();
    }

    fn empirical_constraint_config(policy: UnderpoweredTrialPolicy) -> ParameterOptimizerConfig {
        let mut cfg = config();
        cfg.schema_version = if policy == UnderpoweredTrialPolicy::DevelopmentEligible {
            3
        } else {
            2
        };
        cfg.underpowered_trial_policy = policy;
        cfg.empirical_entrapment_constraints = vec![EmpiricalEntrapmentConstraint {
            level: "protein".into(),
            maximum_adjusted_fdp: 0.01,
            minimum_entrapment_observations_for_power: 3,
        }];
        cfg
    }

    fn empirical_evaluation(entrapments: usize, fdp: f64) -> TrialEvaluation {
        TrialEvaluation {
            status: TrialStatus::Feasible,
            technical_reason: None,
            empirical_reason: None,
            metrics: Some(TrialMetrics {
                level4_proteins: 17,
                level4_canonical_peptides: 343,
                level4_peptidoforms: 421,
                level4_psms: 10_141,
                adjusted_entrapment_fdp: Some(fdp),
                entrapment_count: entrapments,
                adjusted_entrapment_fdp_by_level: BTreeMap::from([("protein".into(), Some(fdp))]),
                entrapment_count_by_level: BTreeMap::from([("protein".into(), entrapments)]),
                model_complexity: 1,
            }),
            development_selection_eligible: false,
            empirical_point_estimate_within_limit: None,
            empirical_calibration_power: EmpiricalCalibrationPower::NotAssessed,
            statistical_validation_status: StatisticalValidationStatus::NotEvaluated,
            statistical_default_eligibility: StatisticalDefaultEligibility::NotEvaluated,
            compact_diagnostics: BTreeMap::from([
                ("fallback_used".into(), serde_json::json!(false)),
                ("model_substitution".into(), serde_json::json!(false)),
                ("target_only_outcomes_used".into(), serde_json::json!(false)),
            ]),
        }
    }

    #[test]
    fn zero_entrapments_are_development_eligible_only_under_explicit_policy() {
        let mut development = empirical_evaluation(0, 0.0);
        apply_empirical_constraints(
            &empirical_constraint_config(UnderpoweredTrialPolicy::DevelopmentEligible),
            &mut development,
        );
        assert_eq!(development.status, TrialStatus::Feasible);
        assert!(development.development_selection_eligible);
        assert_eq!(
            development.empirical_point_estimate_within_limit,
            Some(true)
        );
        assert_eq!(
            development.empirical_calibration_power,
            EmpiricalCalibrationPower::Underpowered
        );
        assert_eq!(
            development.statistical_validation_status,
            StatisticalValidationStatus::NotEvaluableUnderpowered
        );
        assert_eq!(
            development.statistical_default_eligibility,
            StatisticalDefaultEligibility::NotEvaluated
        );

        let mut legacy = empirical_evaluation(0, 0.0);
        apply_empirical_constraints(
            &empirical_constraint_config(UnderpoweredTrialPolicy::NotEvaluable),
            &mut legacy,
        );
        assert_eq!(legacy.status, TrialStatus::NotEvaluable);
        assert!(!legacy.development_selection_eligible);
        assert_eq!(
            legacy.empirical_calibration_power,
            EmpiricalCalibrationPower::Underpowered
        );
    }

    #[test]
    fn empirical_point_limit_and_power_are_independent() {
        let config = empirical_constraint_config(UnderpoweredTrialPolicy::DevelopmentEligible);

        let mut underpowered_within = empirical_evaluation(1, 0.005);
        apply_empirical_constraints(&config, &mut underpowered_within);
        assert_eq!(underpowered_within.status, TrialStatus::Feasible);
        assert!(underpowered_within.development_selection_eligible);
        assert_eq!(
            underpowered_within.empirical_calibration_power,
            EmpiricalCalibrationPower::Underpowered
        );

        let mut underpowered_above = empirical_evaluation(1, 0.02);
        apply_empirical_constraints(&config, &mut underpowered_above);
        assert_eq!(
            underpowered_above.status,
            TrialStatus::EmpiricallyInfeasible
        );
        assert!(!underpowered_above.development_selection_eligible);
        assert_eq!(
            underpowered_above.empirical_calibration_power,
            EmpiricalCalibrationPower::Underpowered
        );
        assert_eq!(
            underpowered_above.empirical_point_estimate_within_limit,
            Some(false)
        );

        let mut powered_within = empirical_evaluation(3, 0.005);
        apply_empirical_constraints(&config, &mut powered_within);
        assert_eq!(powered_within.status, TrialStatus::Feasible);
        assert!(powered_within.development_selection_eligible);
        assert_eq!(
            powered_within.empirical_calibration_power,
            EmpiricalCalibrationPower::AdequatelyPowered
        );
        assert_eq!(
            powered_within.statistical_validation_status,
            StatisticalValidationStatus::EmpiricallyEvaluable
        );

        let mut powered_above = empirical_evaluation(3, 0.02);
        apply_empirical_constraints(&config, &mut powered_above);
        assert_eq!(powered_above.status, TrialStatus::EmpiricallyInfeasible);
        assert!(!powered_above.development_selection_eligible);
        assert_eq!(
            powered_above.empirical_calibration_power,
            EmpiricalCalibrationPower::AdequatelyPowered
        );
    }

    #[test]
    fn development_eligible_policy_is_versioned_and_development_only() {
        let mut legacy_json = serde_json::to_value(config()).unwrap();
        legacy_json
            .as_object_mut()
            .unwrap()
            .remove("underpowered_trial_policy");
        let legacy: ParameterOptimizerConfig = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(
            legacy.underpowered_trial_policy,
            UnderpoweredTrialPolicy::NotEvaluable
        );

        let mut old_schema =
            empirical_constraint_config(UnderpoweredTrialPolicy::DevelopmentEligible);
        old_schema.schema_version = 2;
        assert!(old_schema
            .validate()
            .unwrap_err()
            .to_string()
            .contains("requires parameter_optimizer schema_version 3"));

        for classification in [
            OptimizationClassification::Holdout,
            OptimizationClassification::Release,
            OptimizationClassification::StatisticalDefault,
            OptimizationClassification::ProductionDefault,
        ] {
            let mut nondevelopment =
                empirical_constraint_config(UnderpoweredTrialPolicy::DevelopmentEligible);
            nondevelopment.classification = classification;
            assert!(nondevelopment
                .validate()
                .unwrap_err()
                .to_string()
                .contains("must be development_only"));
        }

        let serialized = serde_json::to_value(empirical_constraint_config(
            UnderpoweredTrialPolicy::DevelopmentEligible,
        ))
        .unwrap();
        assert_eq!(
            serialized["empirical_entrapment_constraints"][0]
                ["minimum_entrapment_observations_for_power"],
            serde_json::json!(3)
        );
        assert!(serialized["empirical_entrapment_constraints"][0]
            .get("minimum_entrapment_count")
            .is_none());
        let old_constraint: EmpiricalEntrapmentConstraint =
            serde_json::from_value(serde_json::json!({
                "level": "protein",
                "maximum_adjusted_fdp": 0.01,
                "minimum_entrapment_count": 3
            }))
            .unwrap();
        assert_eq!(old_constraint.minimum_entrapment_observations_for_power, 3);
    }

    struct Step4MomentsEvaluator {
        calls: usize,
    }

    impl TrialEvaluator for Step4MomentsEvaluator {
        fn evaluate(&mut self, request: &TrialRequest) -> Result<TrialEvaluation> {
            self.calls += 1;
            assert!(!request.target_only_outcomes_allowed);
            let purification = request
                .parameters
                .get("moments_purification_factor")
                .and_then(ParameterValue::as_f64)
                .unwrap();
            let mut evaluation = empirical_evaluation(0, 0.0);
            if (purification - 0.2).abs() < f64::EPSILON {
                let metrics = evaluation.metrics.as_mut().unwrap();
                metrics.level4_canonical_peptides = 344;
                metrics.level4_peptidoforms = 422;
            }
            Ok(evaluation)
        }
    }

    #[test]
    fn step4_underpowered_moments_example_selects_by_existing_objective() {
        let path = temp("underpowered-step4-example");
        std::fs::create_dir_all(&path).unwrap();
        let mut cfg = empirical_constraint_config(UnderpoweredTrialPolicy::DevelopmentEligible);
        cfg.maximum_trial_budget = 2;
        cfg.maximum_optimization_passes = 1;
        cfg.compiled_defaults.insert(
            "moments_purification_factor".into(),
            ParameterValue::Float(0.25),
        );
        cfg.block_order = vec!["moments".into()];
        cfg.blocks = vec![OptimizerBlock {
            id: "moments".into(),
            enabled: true,
            scope: ParameterScope::PerExpert,
            expert: Some(OptimizerExpert::Moments),
            strategy: OptimizerStrategy::ExhaustiveGrid,
            structural_comparison: false,
            fixed: BTreeMap::new(),
            space: BTreeMap::from([(
                "moments_purification_factor".into(),
                vec![ParameterValue::Float(0.25), ParameterValue::Float(0.2)],
            )]),
            window_search: None,
            use_external_features: false,
            max_trials: Some(2),
            max_passes: None,
        }];
        let mut evaluator = Step4MomentsEvaluator { calls: 0 };
        let checkpoint = path.join("checkpoint.json");
        let result = run_optimizer(&cfg, &identity(), &checkpoint, &mut evaluator).unwrap();
        assert_eq!(evaluator.calls, 2);
        assert_eq!(
            result.outcome,
            OptimizerOutcome::UnderpoweredDevelopmentWinner
        );
        assert_eq!(result.underpowered_trial_count, 2);
        assert_eq!(result.powered_trial_count, 0);
        let winner = result
            .trials
            .iter()
            .find(|trial| Some(&trial.request.trial_id) == result.winner_trial_id.as_ref())
            .unwrap();
        assert_eq!(
            winner.request.parameters["moments_purification_factor"],
            ParameterValue::Float(0.2)
        );
        assert!(winner.evaluation.development_selection_eligible);
        assert_eq!(
            winner.evaluation.statistical_validation_status,
            StatisticalValidationStatus::NotEvaluableUnderpowered
        );
        assert_eq!(result.statistical_default_status, "not_evaluated");

        let first_fingerprint = result.optimizer_fingerprint.clone();
        let first_scientific_hash = result.scientific_result_sha256.clone();
        let mut replay = Step4MomentsEvaluator { calls: 0 };
        let resumed = run_optimizer(&cfg, &identity(), &checkpoint, &mut replay).unwrap();
        assert_eq!(replay.calls, 0);
        assert_eq!(resumed.optimizer_fingerprint, first_fingerprint);
        assert_eq!(resumed.scientific_result_sha256, first_scientific_hash);
        assert!(resumed
            .trials
            .iter()
            .all(|trial| trial.reused_from_checkpoint));

        let mut blocking = cfg.clone();
        blocking.underpowered_trial_policy = UnderpoweredTrialPolicy::NotEvaluable;
        assert_ne!(
            optimizer_fingerprint(&identity(), &cfg).unwrap(),
            optimizer_fingerprint(&identity(), &blocking).unwrap()
        );
        let error = load_checkpoint(
            &checkpoint,
            &optimizer_fingerprint(&identity(), &blocking).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("fingerprint mismatch"));
        let mut changed_power_threshold = cfg.clone();
        changed_power_threshold.empirical_entrapment_constraints[0]
            .minimum_entrapment_observations_for_power = 4;
        assert_ne!(
            optimizer_fingerprint(&identity(), &cfg).unwrap(),
            optimizer_fingerprint(&identity(), &changed_power_threshold).unwrap()
        );
        assert_eq!(
            cfg.selected_experts,
            vec![OptimizerExpert::Moments, OptimizerExpert::Mle]
        );
        std::fs::remove_dir_all(path).unwrap();
    }

    struct FixedEmpiricalEvaluator {
        entrapments: usize,
        fdp: f64,
    }

    impl TrialEvaluator for FixedEmpiricalEvaluator {
        fn evaluate(&mut self, request: &TrialRequest) -> Result<TrialEvaluation> {
            assert!(!request.target_only_outcomes_allowed);
            Ok(empirical_evaluation(self.entrapments, self.fdp))
        }
    }

    #[test]
    fn development_terminal_outcomes_distinguish_power_from_fdp_failure() {
        let powered_path = temp("powered-development-outcome");
        std::fs::create_dir_all(&powered_path).unwrap();
        let powered_config =
            empirical_constraint_config(UnderpoweredTrialPolicy::DevelopmentEligible);
        let powered = run_optimizer(
            &powered_config,
            &identity(),
            &powered_path.join("checkpoint.json"),
            &mut FixedEmpiricalEvaluator {
                entrapments: 3,
                fdp: 0.005,
            },
        )
        .unwrap();
        assert_eq!(
            powered.outcome,
            OptimizerOutcome::CompletedDevelopmentOptimization
        );
        assert_eq!(powered.powered_trial_count, powered.trials.len());
        assert_eq!(powered.underpowered_trial_count, 0);

        let infeasible_path = temp("fdp-infeasible-development-outcome");
        std::fs::create_dir_all(&infeasible_path).unwrap();
        let infeasible = run_optimizer(
            &powered_config,
            &identity(),
            &infeasible_path.join("checkpoint.json"),
            &mut FixedEmpiricalEvaluator {
                entrapments: 1,
                fdp: 0.02,
            },
        )
        .unwrap();
        assert_eq!(
            infeasible.outcome,
            OptimizerOutcome::NoEmpiricallyFeasibleSolution
        );
        assert!(infeasible.winner_trial_id.is_none());
        assert!(infeasible.trials.iter().all(|trial| {
            trial.evaluation.status == TrialStatus::EmpiricallyInfeasible
                && !trial.evaluation.development_selection_eligible
                && trial.evaluation.empirical_calibration_power
                    == EmpiricalCalibrationPower::Underpowered
        }));
        std::fs::remove_dir_all(powered_path).unwrap();
        std::fs::remove_dir_all(infeasible_path).unwrap();
    }

    #[test]
    fn resume_at_exact_trial_budget_reuses_all_trials() {
        let mut cfg = config();
        cfg.maximum_trial_budget = 4;
        let path = temp("exact-budget-resume");
        std::fs::create_dir_all(&path).unwrap();
        let checkpoint = path.join("checkpoint.json");
        let mut first = Evaluator::default();
        run_optimizer(&cfg, &identity(), &checkpoint, &mut first).unwrap();
        assert_eq!(first.calls.len(), 4);
        let mut second = Evaluator::default();
        let resumed = run_optimizer(&cfg, &identity(), &checkpoint, &mut second).unwrap();
        assert_eq!(resumed.outcome, OptimizerOutcome::ExhaustiveBoundedOptimum);
        assert!(second.calls.is_empty());
        assert!(resumed
            .trials
            .iter()
            .all(|trial| trial.reused_from_checkpoint));
        assert_eq!(resumed.scientific_result_sha256.len(), 64);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn portable_fingerprint_changes_with_science_not_process_context() {
        let cfg = config();
        let first = optimizer_fingerprint(&identity(), &cfg).unwrap();
        let second = optimizer_fingerprint(&identity(), &cfg).unwrap();
        assert_eq!(first, second);
        let mut changed = identity();
        changed.raw_annotation_cache_identity = "different-raw-cache".into();
        assert_ne!(first, optimizer_fingerprint(&changed, &cfg).unwrap());
        let mut changed_source = identity();
        changed_source.optimizer_source_sha256 = "different-optimizer-source".into();
        assert_ne!(first, optimizer_fingerprint(&changed_source, &cfg).unwrap());
        let mut frozen_experts = cfg.clone();
        frozen_experts
            .expected_expert_configuration_sha256
            .insert("moments".into(), "a".repeat(64));
        assert_ne!(
            first,
            optimizer_fingerprint(&identity(), &frozen_experts).unwrap(),
            "prospectively declared frozen expert hashes must bind checkpoints"
        );
        let serialized = serde_json::to_string(&identity()).unwrap();
        assert!(!serialized.contains(std::env::temp_dir().to_string_lossy().as_ref()));
    }

    #[test]
    fn selected_weight_in_defaults_cannot_silently_remove_voter() {
        let mut cfg = config();
        cfg.selected_experts.push(OptimizerExpert::Ensemble);
        cfg.workflow_defaults
            .insert("ensemble_weight_moments".into(), ParameterValue::Float(0.0));
        cfg.block_order = vec!["ensemble".into()];
        cfg.blocks = vec![OptimizerBlock {
            id: "ensemble".into(),
            enabled: true,
            scope: ParameterScope::EnsembleFinal,
            expert: Some(OptimizerExpert::Ensemble),
            strategy: OptimizerStrategy::ExhaustiveGrid,
            structural_comparison: false,
            fixed: BTreeMap::new(),
            space: BTreeMap::from([(
                "ensemble_weight_mle".into(),
                vec![ParameterValue::Float(1.0)],
            )]),
            window_search: None,
            use_external_features: false,
            max_trials: Some(1),
            max_passes: None,
        }];
        let error = cfg.validate().unwrap_err().to_string();
        assert!(
            error.contains("zero effective weight") || error.contains("below minimum"),
            "{error}"
        );
    }
}
