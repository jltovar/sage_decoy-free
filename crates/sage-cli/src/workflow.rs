use crate::candidate_pool::{
    content_sha256, search_fingerprint, stable_candidate_id,
    verify_usage as verify_candidate_pool_usage, CandidatePoolManifest, CandidatePoolRequest,
    CandidatePoolUsage, CANDIDATE_ID_SCHEMA,
};
use crate::entrapment::{
    compare_generated_to_legacy, entrapment_generation_input_sha256, generate_foreign_entrapment,
    inspect_frozen_entrapment, EntrapmentDatabaseMode, EntrapmentDatabaseReport,
    EntrapmentFastaParityReport, EntrapmentGenerationReport, ForeignSourceMode,
    LegacyEntrapmentReference, SharedPeptideExclusionMode,
};
use crate::external_feature_cache::{
    preflight_existing_cache_root, raw_generator_settings_sha256_with_existing_probe_root,
    raw_generator_settings_sha256_with_probe_root, verify_usage as verify_annotation_cache_usage,
    ExternalAnnotationCacheRequest, ExternalAnnotationCacheUsage,
    RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION,
};
use crate::input::Input;
use crate::parameter_optimizer::{
    apply_fdr_overrides, parameter_catalog_fingerprint, run_optimizer, OptimizerBlock,
    OptimizerExecutionMode, OptimizerExpert, OptimizerIdentity, OptimizerOutcome,
    OptimizerRunResult, OptimizerWindowSearch, ParameterOptimizerConfig, ParameterValue,
    TrialEvaluation, TrialEvaluator, TrialMetrics, TrialRecord, TrialRequest, TrialStatus,
};
use crate::provenance::{freeze_baseline, sha256_file, write_json_atomic, BaselineManifest};
use crate::runner::Runner;
use crate::validation::{
    accepted_target_peptides, ensemble_interaction_calibration, expert_quality_gates,
    is_target_only_stage, missing_parity_evidence, parity_comparisons, stage_comparisons,
    summarize_run, target_only_policy_capability, tdc_benchmark_comparisons, transfer_stability,
    EffectiveRatios, EnsembleInteractionCalibration, ExpertQualityGate, InvalidValidationRun,
    ParityComparison, ParityPair, RunValidationSummary, StageComparison,
    TargetOnlyCalibrationPolicy, TargetOnlyPolicyCapability, TdcBenchmarkComparison,
    ValidationMode, ValidationRun,
};
use anyhow::{Context, Result};
use sage_core::decoy_free_fdr::{DfRunArtifacts, FittedArtifactProvenance};
use sage_core::input::{
    AdaptiveNullWindowSearchOptions, EnsemblePCombiner, EnsemblePepCombiner, FdrMode, FdrOptions,
    FdrSettings, ModelFit, NullWindowCandidate, NullWindowOptimizerOptions, NullWindowSearchBounds,
    NullWindowSearchStrategy, NullWindowValidationScope,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NullWindow {
    pub min_rank: u32,
    pub max_rank: u32,
}

/// Compact, dataset-first window search declaration. Historical validation
/// manifests may continue to use `candidate_windows` for an exact ordered
/// replay; new datasets should normally use `strategy=landscape_adaptive` with
/// bounds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowOptimizerWorkflow {
    pub strategy: NullWindowSearchStrategy,
    pub min_rank_range: [u32; 2],
    pub max_rank_range: [u32; 2],
    #[serde(default)]
    pub adaptive: AdaptiveNullWindowSearchOptions,
}

impl WindowOptimizerWorkflow {
    fn bounds(&self) -> NullWindowSearchBounds {
        NullWindowSearchBounds {
            min_rank_min: self.min_rank_range[0],
            min_rank_max: self.min_rank_range[1],
            max_rank_min: self.max_rank_range[0],
            max_rank_max: self.max_rank_range[1],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ms2RescorePolicy {
    Never,
    Measure,
    Always,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleParticipation {
    /// Request one Ensemble vote when this independently fitted model passes
    /// the technical artifact and provenance checks.
    #[default]
    Auto,
    /// Keep the individual expert results, but never admit this expert to an
    /// Ensemble lock. A concrete reason is mandatory in the workflow manifest.
    Excluded,
}

impl Default for Ms2RescorePolicy {
    fn default() -> Self {
        Self::Measure
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelWorkflow {
    pub model: ModelFit,
    #[serde(default)]
    pub window: Option<NullWindow>,
    #[serde(default)]
    pub candidate_windows: Vec<NullWindow>,
    #[serde(default)]
    pub window_optimizer: Option<WindowOptimizerWorkflow>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub ms2rescore: Ms2RescorePolicy,
    #[serde(default)]
    pub maximum_raw_fdp_increase: Option<f64>,
    #[serde(default)]
    pub minimum_level4_peptide_gain: Option<usize>,
    /// Optional model-specific exception to the workflow-wide target-only
    /// calibration policy. Lower Order resolves this to refit-only semantics.
    #[serde(default)]
    pub target_only_calibration_policy: Option<TargetOnlyCalibrationPolicy>,
    /// Explicit JSON roster control. `auto` requests one vote from this model;
    /// `excluded` keeps the individual result but does not request a vote.
    #[serde(default, skip_serializing_if = "is_auto_ensemble_participation")]
    pub ensemble_participation: EnsembleParticipation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble_exclusion_reason: Option<String>,
    /// Membership in the established Ensemble used only as the counterfactual
    /// for post-assembly interaction reporting. It does not bypass or alter
    /// any participation gate. Legacy manifests default all experts to the
    /// baseline, producing an identity comparison.
    #[serde(default = "default_true")]
    pub ensemble_interaction_baseline: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntrapmentWorkflow {
    #[serde(default)]
    pub database_mode: EntrapmentDatabaseMode,
    #[serde(default)]
    pub foreign_fastas: Vec<PathBuf>,
    pub output_fasta: PathBuf,
    #[serde(default)]
    pub frozen_legacy_fasta: Option<PathBuf>,
    #[serde(default)]
    pub foreign_source_mode: ForeignSourceMode,
    #[serde(default)]
    pub shared_peptide_exclusion_mode: SharedPeptideExclusionMode,
    #[serde(default)]
    pub selected_foreign_fasta: Option<PathBuf>,
    #[serde(default)]
    pub legacy_parity_reference: Option<LegacyEntrapmentReference>,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_protein_fold")]
    pub protein_fold: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaselineWorkflow {
    #[serde(default)]
    pub paths: Vec<PathBuf>,
    pub output_manifest: PathBuf,
    #[serde(default = "default_baseline_status")]
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationWorkflow {
    #[serde(default)]
    pub effective_ratios: EffectiveRatios,
    #[serde(default)]
    pub null_window_validation_scope: NullWindowValidationScope,
    #[serde(default = "default_true")]
    pub use_generated_entrapment_ratios: bool,
    #[serde(default = "default_fdr")]
    pub fdr_threshold: f64,
    #[serde(default = "default_transfer_loss")]
    /// Deprecated for runtime admission. Retained as a nonblocking validation
    /// diagnostic and for backward-compatible workflow JSON loading.
    pub maximum_transfer_fraction_loss: f64,
    #[serde(default)]
    pub additional_runs: Vec<ValidationRun>,
    #[serde(default = "default_minimum_incremental_peptides")]
    /// Deprecated for runtime admission. Unique/incremental discoveries are
    /// reported but do not select Ensemble voters.
    pub minimum_incremental_ensemble_peptides: usize,
    #[serde(default = "default_minimum_ensemble_experts")]
    pub minimum_ensemble_experts: usize,
    #[serde(default = "default_minimum_entrapment_peptides_for_stable_estimate")]
    pub minimum_entrapment_peptides_for_stable_estimate: usize,
    #[serde(default)]
    pub parity_pairs: Vec<ParityPair>,
    /// Optional parity evidence produced by a separate engineering dataset.
    /// This lets an independent holdout consume a passed ISB parity gate
    /// without tuning or demanding equality to a holdout-tuned legacy run.
    #[serde(default)]
    pub external_parity_evidence: Option<PathBuf>,
    #[serde(default = "default_parity_tolerance")]
    pub maximum_parity_fraction_difference: f64,
    #[serde(default)]
    pub tdc_reference_method: Option<String>,
    #[serde(default)]
    pub dataset_role: ValidationDatasetRole,
    /// Cross-dataset artifact experiments are never release evidence. This
    /// flag is required when `artifact_reuse_policy` explicitly opts into the
    /// diagnostic-only escape hatch.
    #[serde(default)]
    pub diagnostic_only: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationDatasetRole {
    #[default]
    Development,
    Holdout,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactReusePolicy {
    /// Normal operation: every dataset fits and optimizes its own models. A
    /// fitted artifact may move only between stages of this same dataset and
    /// search configuration.
    #[default]
    DatasetLocalOnly,
    /// Explicit escape hatch retained only for diagnostic experiments such as
    /// the historical ISB-artifact-on-PXD run. It can never pass release gates.
    CrossDatasetDiagnostic,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetIdentity {
    pub schema_version: u32,
    pub dataset_id: String,
    pub fingerprint: String,
    pub target_fasta_sha256: String,
    pub spectra_sha256: Vec<String>,
    pub search_config_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleExpertLock {
    pub model: ModelFit,
    pub window: Option<NullWindow>,
    pub optimized_fitted_artifacts: PathBuf,
    pub optimized_fitted_artifacts_sha256: String,
    #[serde(default)]
    pub ms2rescore_fitted_artifacts: Option<PathBuf>,
    #[serde(default)]
    pub ms2rescore_fitted_artifacts_sha256: Option<String>,
    pub calibration_stage: String,
    pub calibration_results: PathBuf,
    #[serde(default, skip_serializing_if = "path_buf_is_empty")]
    pub target_only_results: PathBuf,
    #[serde(default)]
    pub target_only_calibration_policy: TargetOnlyCalibrationPolicy,
    pub enabled: bool,
    pub target_peptides: usize,
    pub incremental_target_peptides: usize,
    pub gate_reasons: Vec<String>,
    pub gate_warnings: Vec<String>,
    #[serde(default)]
    pub fit_search_fingerprint: String,
    #[serde(default)]
    pub candidate_id_schema: String,
    #[serde(default = "default_true")]
    pub interaction_baseline: bool,
    #[serde(default)]
    pub participation_decision: String,
    #[serde(default)]
    pub fallback_used: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_only_policy_capability: Option<TargetOnlyPolicyCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Content identity of this expert's independently fitted external profile.
    /// This is expert provenance, not the shared Ensemble-profile identity.
    pub fitted_external_profile_identity_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitted_external_profile_calibration: Option<sage_core::input::ExternalProfileCalibration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotation_cache_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotation_cache_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotation_cache_payload_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleLock {
    pub schema_version: u32,
    /// False for optimizer-only locks, whose provenance and diagnostics are
    /// intentionally restricted to the +entrapment development population.
    #[serde(default = "default_true")]
    pub post_selection_in_scope: bool,
    pub source_manifest_hash: String,
    #[serde(default)]
    pub dataset_fingerprint: String,
    pub experts: Vec<EnsembleExpertLock>,
    /// Canonical JSON-requested voter roster, before technical validation.
    #[serde(default)]
    pub requested_roster: Vec<String>,
    /// Canonical roster that passed technical validation and will vote.
    #[serde(default)]
    pub actual_roster: Vec<String>,
    /// Explicit JSON exclusions, separate from technical failures.
    #[serde(default)]
    pub explicit_exclusions: BTreeMap<String, String>,
    /// Per-model technical failures that prevented a requested vote.
    #[serde(default)]
    pub technical_failures: BTreeMap<String, Vec<String>>,
    #[serde(default = "default_ensemble_roster_contract")]
    pub roster_contract: String,
    pub minimum_required_experts: usize,
    /// A non-evaluable Ensemble is a recorded outcome, not a workflow error.
    /// Individual expert stages remain valid and the core release can proceed.
    #[serde(default = "default_true")]
    pub evaluable: bool,
    #[serde(default)]
    pub not_evaluable_reasons: Vec<String>,
    /// External empirical calibration is one shared dataset-local auxiliary
    /// profile, never a per-expert last-writer-wins artifact.
    #[serde(default = "default_ensemble_external_profile_contract")]
    pub external_profile_contract: String,
    #[serde(default)]
    /// Canonical identity of the shared Ensemble-profile fitting contract.
    /// It binds the common dataset/search/candidate identity and calibration,
    /// but intentionally excludes expert-specific artifacts and annotation
    /// caches. Fitted profile content is stage/search-space specific and is
    /// recorded in the Ensemble stage artifact and checkpoint.
    pub shared_external_profile_contract_sha256: Option<String>,
    #[serde(default)]
    pub shared_external_profile_calibration: Option<sage_core::input::ExternalProfileCalibration>,
    #[serde(default)]
    pub source_configuration_sha256: String,
    #[serde(default)]
    /// Complete lock/provenance identity. Unlike the shared-profile contract,
    /// this binds the roster and every expert's artifact and annotation cache.
    pub analysis_fingerprint: String,
    #[serde(default = "default_raw_q_interaction_warning_threshold")]
    pub raw_q_interaction_warning_threshold: f64,
    pub ensemble_p_combiner: EnsemblePCombiner,
    pub ensemble_pep_combiner: EnsemblePepCombiner,
}

fn default_ensemble_external_profile_contract() -> String {
    "shared_dataset_local".into()
}

fn path_buf_is_empty(path: &PathBuf) -> bool {
    path.as_os_str().is_empty()
}

fn default_ensemble_roster_contract() -> String {
    "json_requested_technical_validation_only".into()
}

fn resolved_ensemble_combiners(
    search_config: &Path,
) -> Result<(EnsemblePCombiner, EnsemblePepCombiner)> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(search_config)?)
        .with_context(|| format!("invalid search configuration {}", search_config.display()))?;
    let options: FdrOptions = serde_json::from_value(
        value
            .get("fdr")
            .filter(|value| !value.is_null())
            .cloned()
            .unwrap_or_else(|| serde_json::json!({})),
    )
    .context("invalid fdr settings in search configuration")?;
    let settings = FdrSettings::from(options);
    Ok((settings.ensemble_p_combiner, settings.ensemble_pep_combiner))
}

fn resolved_external_profile_calibration(
    search_config: &Path,
) -> Result<sage_core::input::ExternalProfileCalibration> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(search_config)?)
        .with_context(|| format!("invalid search configuration {}", search_config.display()))?;
    let options: FdrOptions = serde_json::from_value(
        value
            .get("fdr")
            .filter(|value| !value.is_null())
            .cloned()
            .unwrap_or_else(|| serde_json::json!({})),
    )
    .context("invalid fdr settings in search configuration")?;
    Ok(FdrSettings::from(options).external_profile_calibration)
}

#[derive(Clone)]
struct CompletedExpert {
    model: ModelFit,
    window: Option<NullWindow>,
    optimized_artifacts: PathBuf,
    optimized_results: PathBuf,
    ms2rescore_artifacts: Option<PathBuf>,
    ms2rescore_results: Option<PathBuf>,
    calibration_stage: String,
    calibration_results: PathBuf,
    target_only_results: PathBuf,
    target_only_calibration_policy: TargetOnlyCalibrationPolicy,
    calibration_search_fingerprint: String,
    fitted_external_profile_identity_sha256: Option<String>,
    fitted_external_profile_calibration: Option<sage_core::input::ExternalProfileCalibration>,
    annotation_cache_fingerprint: Option<String>,
    annotation_cache_manifest_sha256: Option<String>,
    annotation_cache_payload_sha256: Option<String>,
}

struct SharedDatabase {
    database: Arc<sage_core::database::IndexedDatabase>,
    resolved_database_parameters: sage_core::database::Parameters,
}

#[derive(Default)]
struct WorkflowRuntime {
    databases: HashMap<String, SharedDatabase>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkflowManifest {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    pub name: String,
    /// Human-readable dataset identifier. The content fingerprint remains the
    /// authoritative identity used for artifact safety.
    #[serde(default)]
    pub dataset_id: Option<String>,
    pub search_config: PathBuf,
    pub target_fasta: PathBuf,
    pub spectra: Vec<String>,
    pub output_root: PathBuf,
    /// Optional immutable candidate-pool root. When omitted, workflows retain
    /// the historical output-root-local cache layout.
    #[serde(default)]
    pub candidate_pool_root: Option<PathBuf>,
    /// Optional immutable MS2Rescore annotation-cache root. When omitted,
    /// workflows retain the historical output-root-local cache layout.
    #[serde(default)]
    pub annotation_cache_root: Option<PathBuf>,
    /// Optional target-only cache root when immutable +entrapment and
    /// target-only caches are stored separately.
    #[serde(default)]
    pub target_only_annotation_cache_root: Option<PathBuf>,
    pub entrapment: EntrapmentWorkflow,
    pub models: Vec<ModelWorkflow>,
    #[serde(default)]
    pub baseline: Option<BaselineWorkflow>,
    pub validation: ValidationWorkflow,
    #[serde(default = "default_true")]
    pub resume: bool,
    /// Fail closed unless an exact immutable candidate pool already exists.
    /// This is intended for annotation-only and downstream replay workflows.
    #[serde(default)]
    pub require_existing_candidate_pool: bool,
    /// Fail closed unless every required annotation cache is already present,
    /// compatible, and integrity-valid. Defaults false for from-scratch work.
    #[serde(default)]
    pub require_existing_annotation_cache: bool,
    /// Migrate an exact schema-v2 cache to the layered raw schema, but never
    /// invoke an annotation generator. Defaults false.
    #[serde(default)]
    pub migrate_schema_v2_annotation_cache_only: bool,
    #[serde(default)]
    pub annotate_target_matches: bool,
    #[serde(default)]
    pub ensemble_lock: Option<PathBuf>,
    #[serde(default)]
    pub locked_expert_artifacts: BTreeMap<String, PathBuf>,
    #[serde(default)]
    pub artifact_reuse_policy: ArtifactReusePolicy,
    /// Dataset-local target-only calibration semantics. The parity-oriented
    /// default locks the selected window but refits nuisance state.
    #[serde(default)]
    pub target_only_calibration_policy: TargetOnlyCalibrationPolicy,
    /// Versioned, development-only analysis-parameter optimization. The
    /// declaration is embedded in the portable workflow; the repository
    /// catalog is used for authoring and tests, never as a runtime file.
    #[serde(default)]
    pub parameter_optimizer: Option<ParameterOptimizerConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowProvenance {
    pub schema_version: u32,
    pub source_stage: String,
    pub source_model: String,
    pub source_dataset_id: String,
    pub source_dataset_fingerprint: String,
    #[serde(default)]
    pub min_rank: Option<u32>,
    #[serde(default)]
    pub max_rank: Option<u32>,
    #[serde(default)]
    pub source_fitted_artifact: Option<PathBuf>,
    #[serde(default)]
    pub source_fitted_artifact_sha256: Option<String>,
    #[serde(default)]
    pub source_search_fingerprint: Option<String>,
    pub candidate_id_schema: String,
}

#[derive(Clone, Debug, Serialize)]
struct TargetOnlyStageContext {
    policy: TargetOnlyCalibrationPolicy,
    release_candidate: bool,
    window_provenance: WindowProvenance,
    allow_candidate_pool_reuse: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageRecord {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    pub stage: String,
    pub model: String,
    pub input_hash: String,
    pub status: String,
    pub results: PathBuf,
    pub config_snapshot: PathBuf,
    #[serde(default)]
    pub results_sha256: String,
    #[serde(default)]
    pub config_snapshot_sha256: String,
    pub external_features_enabled: bool,
    pub calibration_mode: String,
    #[serde(default)]
    pub dataset_id: String,
    #[serde(default)]
    pub dataset_fingerprint: String,
    #[serde(default)]
    pub artifact_fit_dataset_fingerprint: Option<String>,
    #[serde(default)]
    pub candidate_pool: Option<CandidatePoolUsage>,
    #[serde(default)]
    pub require_existing_candidate_pool: bool,
    #[serde(default)]
    pub require_existing_annotation_cache: bool,
    #[serde(default)]
    pub ms2rescore_annotation_cache: Option<ExternalAnnotationCacheUsage>,
    #[serde(default)]
    pub target_only_calibration_policy: Option<TargetOnlyCalibrationPolicy>,
    #[serde(default = "default_true")]
    pub release_candidate: bool,
    #[serde(default)]
    pub window_provenance: Option<WindowProvenance>,
    #[serde(default)]
    pub external_profile_calibration: Option<sage_core::input::ExternalProfileCalibration>,
    /// Canonical shared-profile contract from the Ensemble lock, when this is
    /// an external Ensemble stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble_shared_profile_contract_sha256: Option<String>,
    /// Content identity of the one profile fitted by this stage/search space.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitted_external_profile_identity_sha256: Option<String>,
    /// Schema-v3 target-only evaluability. Older completed checkpoints default
    /// to evaluable and remain loadable.
    #[serde(default = "default_true")]
    pub evaluable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_evaluable_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_only_policy_capability: Option<TargetOnlyPolicyCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nuisance_state_provenance: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_only_window_tuning: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub complete_dataset_artifact_reused: Option<bool>,
    #[serde(default)]
    pub fallback_used: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_artifact_schema: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble_interaction_calibration: Option<EnsembleInteractionCalibration>,
    /// Resolved analysis-only optimizer values applied after workflow defaults.
    /// Empty for non-optimizer stages and legacy checkpoints.
    #[serde(default)]
    pub parameter_overrides: BTreeMap<String, ParameterValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkflowState {
    pub schema_version: u32,
    pub manifest_hash: String,
    pub dataset: DatasetIdentity,
    pub entrapment: Option<EntrapmentDatabaseReport>,
    #[serde(default)]
    pub entrapment_fasta_parity: Option<EntrapmentFastaParityReport>,
    pub baseline: Option<BaselineManifest>,
    pub stages: Vec<StageRecord>,
    #[serde(default)]
    pub candidate_pools: Vec<CandidatePoolUsage>,
    #[serde(default)]
    pub ms2rescore_annotation_caches: Vec<ExternalAnnotationCacheUsage>,
    pub validation: Vec<RunValidationSummary>,
    pub missing_runs: Vec<ValidationRun>,
    #[serde(default)]
    pub invalid_runs: Vec<InvalidValidationRun>,
    pub stage_comparisons: Vec<StageComparison>,
    /// Statistical validation diagnostics only; never controls the roster.
    pub ensemble_expert_gates: Vec<ExpertQualityGate>,
    #[serde(default = "nonblocking_ensemble_gate_effect")]
    pub ensemble_expert_gates_participation_effect: String,
    pub parity_comparisons: Vec<ParityComparison>,
    pub tdc_benchmarks: Vec<TdcBenchmarkComparison>,
    pub release_gate: ReleaseGate,
    pub transfer_stability: Vec<crate::validation::TransferStability>,
    pub pending_validation_gates: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble_interaction_calibration: Option<EnsembleInteractionCalibration>,
    #[serde(default)]
    pub resource_preflight: Vec<ResourcePreflightReport>,
    #[serde(default)]
    pub planned_models: Vec<PlannedModelReport>,
    /// Development-only parameter-optimization provenance. Target-only
    /// stages never contribute to these records.
    #[serde(default)]
    pub parameter_optimization: Vec<OptimizerRunResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_optimizer_execution: Option<OptimizerExecutionReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerWinnerSummary {
    pub winner_trial_id: String,
    pub scientific_result_sha256: String,
    pub outcome: OptimizerOutcome,
    pub winner_artifacts: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerExecutionReport {
    pub schema_version: u32,
    pub execution_mode: OptimizerExecutionMode,
    pub optimization_status: String,
    pub selected_entrapment_winners: BTreeMap<String, OptimizerWinnerSummary>,
    pub post_selection_stages: String,
    pub target_only_evaluation: String,
    pub matched_tdc_evaluation: String,
    pub independent_validation: String,
    pub statistical_default_eligibility: String,
}

fn optimizer_execution_report(
    config: &ParameterOptimizerConfig,
    results: &[OptimizerRunResult],
    optimization_status: impl Into<String>,
) -> OptimizerExecutionReport {
    let selected_entrapment_winners = results
        .iter()
        .filter_map(|result| {
            let expert = result
                .requested_parameter_space
                .iter()
                .find_map(|block| block.expert)?;
            Some((
                expert.slug().to_owned(),
                OptimizerWinnerSummary {
                    winner_trial_id: result.winner_trial_id.clone()?,
                    scientific_result_sha256: result.scientific_result_sha256.clone(),
                    outcome: result.outcome.clone(),
                    winner_artifacts: result.winner_artifacts.clone(),
                },
            ))
        })
        .collect();
    let optimization_only = config.optimization_only();
    OptimizerExecutionReport {
        schema_version: 1,
        execution_mode: config.execution_mode,
        optimization_status: optimization_status.into(),
        selected_entrapment_winners,
        post_selection_stages: if optimization_only {
            "not_run_by_execution_scope"
        } else {
            "in_scope"
        }
        .into(),
        target_only_evaluation: if optimization_only {
            "not_run_by_execution_scope"
        } else {
            "in_scope"
        }
        .into(),
        matched_tdc_evaluation: "not_run".into(),
        independent_validation: "not_run".into(),
        statistical_default_eligibility: "not_evaluated".into(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlannedModelReport {
    pub order: usize,
    pub model: String,
    pub window_mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_window: Option<[u32; 2]>,
    pub ms2rescore_policy: String,
    pub requested_for_ensemble: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourcePreflightReport {
    pub resource_type: String,
    pub search_space: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default = "legacy_resource_preflight_status")]
    pub status: String,
    pub requested_path: PathBuf,
    pub expected_fingerprint: String,
    pub actual_fingerprint: String,
    pub schema_version: u32,
    pub candidate_or_annotation_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_rank_depth: Option<usize>,
    pub manifest_sha256: String,
    pub payload_sha256: String,
    pub valid: bool,
    pub reused: bool,
    pub generation_allowed: bool,
    #[serde(default)]
    pub catalog_fingerprints: Vec<String>,
    #[serde(default)]
    pub original_source_uris: Vec<String>,
    #[serde(default)]
    pub current_source_uris: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub portable_identity_valid: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relocation_detected: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

fn legacy_resource_preflight_status() -> String {
    "validated_exact".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseGate {
    #[serde(default = "default_release_gate_status")]
    pub status: ReleaseGateStatus,
    pub eligible_for_statistical_default_change: bool,
    pub reasons: Vec<String>,
    #[serde(default)]
    pub not_evaluable_reasons: Vec<String>,
    #[serde(default)]
    pub not_eligible_reasons: Vec<String>,
    pub calibrated_tdc_improvements: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseGateStatus {
    Eligible,
    NotEligible,
    NotEvaluable,
}

fn default_release_gate_status() -> ReleaseGateStatus {
    ReleaseGateStatus::NotEvaluable
}

fn nonblocking_ensemble_gate_effect() -> String {
    "none_nonblocking_diagnostic".into()
}

fn default_true() -> bool {
    true
}
fn is_auto_ensemble_participation(value: &EnsembleParticipation) -> bool {
    *value == EnsembleParticipation::Auto
}
fn default_schema() -> u32 {
    1
}
fn default_protein_fold() -> usize {
    1
}
fn default_fdr() -> f64 {
    0.01
}
fn default_transfer_loss() -> f64 {
    0.20
}
fn default_baseline_status() -> String {
    "complete".into()
}
fn default_minimum_incremental_peptides() -> usize {
    1
}
fn default_minimum_ensemble_experts() -> usize {
    2
}
fn default_minimum_entrapment_peptides_for_stable_estimate() -> usize {
    3
}
fn default_parity_tolerance() -> f64 {
    0.001
}
fn default_raw_q_interaction_warning_threshold() -> f64 {
    0.01
}

fn concrete_target_only_policies(
    policy: TargetOnlyCalibrationPolicy,
) -> Vec<(TargetOnlyCalibrationPolicy, bool)> {
    match policy {
        TargetOnlyCalibrationPolicy::RefitWithLockedWindow => {
            vec![(TargetOnlyCalibrationPolicy::RefitWithLockedWindow, true)]
        }
        TargetOnlyCalibrationPolicy::ReuseDatasetArtifact => {
            vec![(TargetOnlyCalibrationPolicy::ReuseDatasetArtifact, true)]
        }
        TargetOnlyCalibrationPolicy::CompareBoth => vec![
            (TargetOnlyCalibrationPolicy::RefitWithLockedWindow, true),
            (TargetOnlyCalibrationPolicy::ReuseDatasetArtifact, false),
        ],
    }
}

fn allow_target_candidate_pool_reuse(annotate_target_matches: bool, policy_index: usize) -> bool {
    !annotate_target_matches || policy_index > 0
}

impl WorkflowManifest {
    fn resolved_candidate_pool_root(&self) -> PathBuf {
        self.candidate_pool_root
            .clone()
            .unwrap_or_else(|| self.output_root.join("candidate_pools"))
    }

    fn resolved_annotation_cache_root(&self, target_only: bool) -> PathBuf {
        if target_only {
            self.target_only_annotation_cache_root
                .clone()
                .or_else(|| self.annotation_cache_root.clone())
                .unwrap_or_else(|| self.output_root.join("ms2rescore_annotations"))
        } else {
            self.annotation_cache_root
                .clone()
                .unwrap_or_else(|| self.output_root.join("ms2rescore_annotations"))
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read workflow manifest {}", path.display()))?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid workflow manifest {}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.schema_version == 1, "unsupported workflow schema");
        anyhow::ensure!(
            !(self.require_existing_annotation_cache
                && self.migrate_schema_v2_annotation_cache_only),
            "require_existing_annotation_cache and migrate_schema_v2_annotation_cache_only are mutually exclusive: strict reuse is read-only"
        );
        anyhow::ensure!(
            !self.migrate_schema_v2_annotation_cache_only || self.require_existing_candidate_pool,
            "schema-v2 annotation-cache migration requires require_existing_candidate_pool=true"
        );
        anyhow::ensure!(!self.name.trim().is_empty(), "workflow name is required");
        anyhow::ensure!(self.search_config.is_file(), "search_config does not exist");
        anyhow::ensure!(self.target_fasta.is_file(), "target_fasta does not exist");
        anyhow::ensure!(
            !self.spectra.is_empty(),
            "at least one spectrum file is required"
        );
        anyhow::ensure!(!self.models.is_empty(), "at least one model is required");
        if let Some(optimizer) = self
            .parameter_optimizer
            .as_ref()
            .filter(|optimizer| optimizer.enabled)
        {
            optimizer.validate()?;
            anyhow::ensure!(
                self.require_existing_candidate_pool
                    && self.require_existing_annotation_cache
                    && !self.migrate_schema_v2_annotation_cache_only,
                "parameter optimization requires read-only existing candidate pools and raw annotation caches"
            );
            anyhow::ensure!(
                self.validation.dataset_role == ValidationDatasetRole::Development,
                "parameter optimization is development-only and requires a development dataset"
            );
            anyhow::ensure!(
                (self.validation.fdr_threshold - optimizer.fixed_evaluation_threshold).abs()
                    <= f64::EPSILON,
                "optimizer fixed_evaluation_threshold must equal the workflow validation threshold"
            );
            let selected = optimizer
                .selected_experts
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let requested = self
                .models
                .iter()
                .filter(|model| {
                    model.enabled
                        && (model.model == ModelFit::Ensemble
                            || model.ensemble_participation == EnsembleParticipation::Auto)
                })
                .map(|model| optimizer_expert(&model.model))
                .collect::<BTreeSet<_>>();
            anyhow::ensure!(
                selected == requested,
                "parameter_optimizer selected_experts must exactly match the JSON-selected technically eligible roster (including Ensemble when enabled)"
            );
        }
        let mut canonical_models = BTreeSet::new();
        for model in &self.models {
            anyhow::ensure!(
                canonical_models.insert(model_slug(&model.model)),
                "workflow contains duplicate canonical model {}",
                model_slug(&model.model)
            );
        }
        anyhow::ensure!(
            self.validation.use_generated_entrapment_ratios,
            "workflow ratios must be measured by Sage from this dataset's active entrapment FASTA; static cross-dataset ratios are prohibited"
        );
        match self.entrapment.database_mode {
            EntrapmentDatabaseMode::NativeGenerated => {
                anyhow::ensure!(
                    !self.entrapment.foreign_fastas.is_empty(),
                    "native entrapment generation requires at least one foreign FASTA"
                );
                for fasta in &self.entrapment.foreign_fastas {
                    anyhow::ensure!(
                        fasta.is_file(),
                        "foreign FASTA does not exist: {}",
                        fasta.display()
                    );
                }
                anyhow::ensure!(
                    self.entrapment.frozen_legacy_fasta.is_none(),
                    "native generation must not declare frozen_legacy_fasta"
                );
                match self.entrapment.foreign_source_mode {
                    ForeignSourceMode::Automatic => anyhow::ensure!(
                        self.entrapment.selected_foreign_fasta.is_none(),
                        "automatic source selection must not declare selected_foreign_fasta"
                    ),
                    ForeignSourceMode::Explicit | ForeignSourceMode::AutomaticWithOverride => {
                        anyhow::ensure!(
                            self.entrapment
                                .selected_foreign_fasta
                                .as_ref()
                                .is_some_and(|path| path.is_file()),
                            "{:?} source selection requires an existing selected_foreign_fasta",
                            self.entrapment.foreign_source_mode
                        );
                    }
                }
                if let Some(reference) = &self.entrapment.legacy_parity_reference {
                    anyhow::ensure!(
                        reference.fasta.is_file(),
                        "legacy parity FASTA does not exist: {}",
                        reference.fasta.display()
                    );
                    if let Some(path) = reference.foreign_fasta.as_ref() {
                        anyhow::ensure!(
                            path.is_file(),
                            "legacy foreign source does not exist: {}",
                            path.display()
                        );
                    }
                    if let Some(path) = reference.generation_log.as_ref() {
                        anyhow::ensure!(
                            path.is_file(),
                            "legacy generation log does not exist: {}",
                            path.display()
                        );
                    }
                }
            }
            EntrapmentDatabaseMode::FrozenLegacy => {
                anyhow::ensure!(
                    self.entrapment
                        .frozen_legacy_fasta
                        .as_ref()
                        .is_some_and(|path| path.is_file()),
                    "frozen_legacy mode requires an existing frozen_legacy_fasta"
                );
                anyhow::ensure!(
                    self.entrapment.legacy_parity_reference.is_none(),
                    "FASTA-generation parity is separate from frozen optimizer-input parity"
                );
            }
        }
        for model in &self.models {
            let window_sources = usize::from(model.window.is_some())
                + usize::from(!model.candidate_windows.is_empty())
                + usize::from(model.window_optimizer.is_some());
            anyhow::ensure!(
                window_sources <= 1,
                "specify only one of window, candidate_windows, or window_optimizer"
            );
            if let Some(window) = &model.window {
                anyhow::ensure!(window.min_rank > 1, "rank 1 cannot be a null window");
                anyhow::ensure!(window.max_rank >= window.min_rank, "invalid null window");
            }
            let mut explicit_windows = BTreeSet::new();
            for window in &model.candidate_windows {
                anyhow::ensure!(
                    window.min_rank > 1 && window.max_rank >= window.min_rank,
                    "invalid candidate null window"
                );
                anyhow::ensure!(
                    explicit_windows.insert((window.min_rank, window.max_rank)),
                    "candidate_windows contains a duplicate null window"
                );
            }
            if let Some(search) = &model.window_optimizer {
                anyhow::ensure!(
                    search.strategy != NullWindowSearchStrategy::Explicit,
                    "window_optimizer supports landscape_adaptive, adaptive, or exhaustive; use candidate_windows for explicit replay"
                );
                let bounds = search.bounds();
                anyhow::ensure!(
                    bounds.min_rank_min > 1
                        && bounds.max_rank_min > 1
                        && bounds.min_rank_min <= bounds.min_rank_max
                        && bounds.max_rank_min <= bounds.max_rank_max
                        && bounds.min_rank_min <= bounds.max_rank_max,
                    "invalid window_optimizer rank ranges"
                );
                if matches!(
                    search.strategy,
                    NullWindowSearchStrategy::Adaptive
                        | NullWindowSearchStrategy::LandscapeAdaptive
                ) {
                    anyhow::ensure!(
                        search.adaptive.sparse_row_step > 0
                            && search.adaptive.x_stride > 0
                            && !search.adaptive.sparse_offsets.is_empty()
                            && search.adaptive.boundary_dead_row_limit > 0
                            && search.adaptive.hill_max_steps > 0
                            && search
                                .adaptive
                                .sparse_eligible_fraction_for_hill
                                .is_finite()
                            && (0.0..=1.0)
                                .contains(&search.adaptive.sparse_eligible_fraction_for_hill),
                        "invalid adaptive window_optimizer settings"
                    );
                }
                if search.strategy == NullWindowSearchStrategy::LandscapeAdaptive {
                    anyhow::ensure!(
                        search.adaptive.landscape_coarse_row_count > 0
                            && !search.adaptive.landscape_coarse_offsets.is_empty()
                            && search.adaptive.landscape_seed_count > 0
                            && search
                                .adaptive
                                .landscape_min_feasible_row_fraction
                                .is_finite()
                            && (0.0..=1.0)
                                .contains(&search.adaptive.landscape_min_feasible_row_fraction)
                            && search.adaptive.landscape_frontier_edge_fraction.is_finite()
                            && (0.0..=1.0)
                                .contains(&search.adaptive.landscape_frontier_edge_fraction),
                        "invalid landscape_adaptive window_optimizer settings"
                    );
                }
            }
            if model.model == ModelFit::Msfdr1Smix {
                anyhow::ensure!(
                    model.window.is_none()
                        && model.candidate_windows.is_empty()
                        && model.window_optimizer.is_none(),
                    "MSFDR1-SMIX is rank-1-only and must not define a null window"
                );
            }
            match model.ensemble_participation {
                EnsembleParticipation::Auto => anyhow::ensure!(
                    model.ensemble_exclusion_reason.is_none(),
                    "ensemble_exclusion_reason requires ensemble_participation=excluded"
                ),
                EnsembleParticipation::Excluded => {
                    anyhow::ensure!(
                        model.model != ModelFit::Ensemble,
                        "exclude individual experts, not the Ensemble model itself"
                    );
                    anyhow::ensure!(
                        model
                            .ensemble_exclusion_reason
                            .as_deref()
                            .is_some_and(|reason| !reason.trim().is_empty()),
                        "ensemble_participation=excluded requires ensemble_exclusion_reason"
                    );
                }
            }
            if !self
                .parameter_optimizer
                .as_ref()
                .is_some_and(ParameterOptimizerConfig::optimization_only)
            {
                let requested_policy = model
                    .target_only_calibration_policy
                    .unwrap_or(self.target_only_calibration_policy);
                if requested_policy != TargetOnlyCalibrationPolicy::CompareBoth {
                    let capability = target_only_policy_capability(&model.model, requested_policy);
                    anyhow::ensure!(
                        capability.supported,
                        "{} does not support target-only policy {}: {}",
                        model_slug(&model.model),
                        requested_policy.stage_name(),
                        capability.reason.as_deref().unwrap_or("unsupported policy")
                    );
                }
            }
        }
        if self.validation.dataset_role == ValidationDatasetRole::Holdout {
            if let Some(path) = self.validation.external_parity_evidence.as_ref() {
                anyhow::ensure!(
                    path.is_file(),
                    "external_parity_evidence does not exist: {}",
                    path.display()
                );
            }
        }
        match self.artifact_reuse_policy {
            ArtifactReusePolicy::DatasetLocalOnly => {
                anyhow::ensure!(
                    self.locked_expert_artifacts.is_empty(),
                    "dataset-local workflows must optimize and fit this dataset; locked_expert_artifacts is reserved for explicit cross-dataset diagnostics"
                );
            }
            ArtifactReusePolicy::CrossDatasetDiagnostic => {
                anyhow::ensure!(
                    self.validation.diagnostic_only,
                    "cross_dataset_diagnostic artifact reuse requires validation.diagnostic_only=true"
                );
                for model in self
                    .models
                    .iter()
                    .filter(|model| model.enabled && model.model != ModelFit::Ensemble)
                {
                    anyhow::ensure!(
                        model.candidate_windows.is_empty() && model.window_optimizer.is_none(),
                        "a cross-dataset diagnostic cannot both import an artifact and optimize a null window"
                    );
                    anyhow::ensure!(
                        self.locked_expert_artifacts
                            .get(model_slug(&model.model))
                            .is_some_and(|path| path.is_file()),
                        "cross-dataset diagnostic model {:?} requires locked_expert_artifacts entry",
                        model.model
                    );
                }
            }
        }
        Ok(())
    }
}

fn compute_dataset_identity(manifest: &WorkflowManifest) -> Result<DatasetIdentity> {
    let target_fasta_sha256 = content_sha256(&manifest.target_fasta)?;
    let search_config_sha256 = sha256_file(&manifest.search_config)?;
    let mut spectra_sha256 = manifest
        .spectra
        .iter()
        .map(|source| {
            let path = Path::new(source);
            if path.is_file() {
                content_sha256(path)
            } else {
                // Remote/cloud sources cannot be content-hashed here. Their
                // stable source string is still incorporated fail-closed into
                // the identity rather than being silently ignored.
                let mut hasher = Sha256::new();
                hasher.update(b"unresolved-spectrum-source:");
                hasher.update(source.as_bytes());
                Ok(format!("{:x}", hasher.finalize()))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    // Dataset identity is independent of input ordering and host-specific
    // paths when files are locally available.
    spectra_sha256.sort();
    let mut hasher = Sha256::new();
    hasher.update(b"sage-decoy-free-dataset-v1\0");
    hasher.update(target_fasta_sha256.as_bytes());
    for digest in &spectra_sha256 {
        hasher.update(b"\0");
        hasher.update(digest.as_bytes());
    }
    Ok(DatasetIdentity {
        schema_version: 1,
        dataset_id: manifest
            .dataset_id
            .clone()
            .unwrap_or_else(|| manifest.name.clone()),
        fingerprint: format!("{:x}", hasher.finalize()),
        target_fasta_sha256,
        spectra_sha256,
        search_config_sha256,
    })
}

fn model_slug(model: &ModelFit) -> &'static str {
    match model {
        ModelFit::Moments => "moments",
        ModelFit::Mle => "mle",
        ModelFit::LowerOrder => "lower_order",
        ModelFit::Msfdr => "msfdr",
        ModelFit::Msfdr1Smix => "msfdr1_smix",
        ModelFit::Msfdr2Smix => "msfdr2_smix",
        ModelFit::Nokoi => "nokoi",
        ModelFit::Ensemble => "ensemble",
    }
}

fn planned_model_reports(manifest: &WorkflowManifest) -> Vec<PlannedModelReport> {
    manifest
        .models
        .iter()
        .filter(|model| model.enabled)
        .enumerate()
        .map(|(order, model)| {
            let (window_mode, fixed_window) = match model.model {
                ModelFit::Msfdr1Smix => ("fixed_rank_1_1".into(), Some([1, 1])),
                ModelFit::Ensemble => ("independent_constituent_artifacts".into(), None),
                _ => {
                    if model.window_optimizer.is_some() {
                        ("dataset_local_optimizer".into(), None)
                    } else if !model.candidate_windows.is_empty() {
                        ("dataset_local_explicit_grid".into(), None)
                    } else if let Some(window) = model.window.as_ref() {
                        (
                            "fixed_manifest_window".into(),
                            Some([window.min_rank, window.max_rank]),
                        )
                    } else {
                        ("search_configuration".into(), None)
                    }
                }
            };
            PlannedModelReport {
                order,
                model: model_slug(&model.model).into(),
                window_mode,
                fixed_window,
                ms2rescore_policy: match model.ms2rescore {
                    Ms2RescorePolicy::Never => "never",
                    Ms2RescorePolicy::Measure => "measure",
                    Ms2RescorePolicy::Always => "always",
                }
                .into(),
                requested_for_ensemble: model.model != ModelFit::Ensemble
                    && model.ensemble_participation == EnsembleParticipation::Auto,
            }
        })
        .collect()
}

fn strict_preflight_fasta(manifest: &WorkflowManifest) -> Result<PathBuf> {
    match manifest.entrapment.database_mode {
        EntrapmentDatabaseMode::FrozenLegacy => manifest
            .entrapment
            .frozen_legacy_fasta
            .clone()
            .context("frozen legacy entrapment FASTA is required for strict preflight"),
        EntrapmentDatabaseMode::NativeGenerated => {
            anyhow::ensure!(
                manifest.entrapment.output_fasta.is_file(),
                "strict read-only preflight cannot generate the entrapment FASTA; expected existing {}",
                manifest.entrapment.output_fasta.display()
            );
            Ok(manifest.entrapment.output_fasta.clone())
        }
    }
}

fn requested_rank_depth(manifest: &WorkflowManifest, runner: &Runner) -> usize {
    let model_rank = manifest
        .models
        .iter()
        .flat_map(|model| {
            model
                .candidate_windows
                .iter()
                .map(|window| window.max_rank as usize)
                .chain(
                    model
                        .window_optimizer
                        .iter()
                        .map(|search| search.max_rank_range[1] as usize),
                )
                .chain(model.window.iter().map(|window| window.max_rank as usize))
        })
        .max()
        .unwrap_or(1);
    runner
        .parameters
        .external_features
        .max_rank
        .map(|rank| rank as usize)
        .unwrap_or(runner.parameters.report_psms)
        .max(model_rank)
}

fn candidate_preflight_report(
    usage: &CandidatePoolUsage,
    search_space: &str,
    requested_path: &Path,
    generation_allowed: bool,
) -> Result<ResourcePreflightReport> {
    let manifest: CandidatePoolManifest = serde_json::from_slice(&std::fs::read(&usage.manifest)?)
        .with_context(|| {
            format!(
                "invalid candidate-pool manifest {}",
                usage.manifest.display()
            )
        })?;
    Ok(ResourcePreflightReport {
        resource_type: "candidate_pool".into(),
        search_space: search_space.into(),
        stage: None,
        status: "validated_exact".into(),
        requested_path: requested_path.to_path_buf(),
        expected_fingerprint: usage.search_fingerprint.clone(),
        actual_fingerprint: manifest.search_fingerprint.digest,
        schema_version: manifest.schema_version,
        candidate_or_annotation_count: manifest.candidate_count,
        retained_rank_depth: Some(manifest.capabilities.retained_rank_depth),
        manifest_sha256: sha256_file(&usage.manifest)?,
        payload_sha256: manifest.payload_sha256,
        valid: true,
        reused: true,
        generation_allowed,
        catalog_fingerprints: Vec::new(),
        original_source_uris: usage.original_source_uris.clone(),
        current_source_uris: usage.current_source_uris.clone(),
        portable_identity_valid: Some(usage.portable_identity_valid),
        relocation_detected: Some(usage.relocation_detected),
        failure_reason: None,
    })
}

fn deferred_annotation_preflight_report(
    model: &ModelWorkflow,
    search_space: &str,
    requested_path: &Path,
    candidate_count: usize,
    requested_max_rank: usize,
    generation_allowed: bool,
    catalog_fingerprints: &[String],
) -> ResourcePreflightReport {
    ResourcePreflightReport {
        resource_type: "stage_external_calibration".into(),
        search_space: search_space.into(),
        stage: Some(format!("{}:ms2rescore", model_slug(&model.model))),
        status: "deferred_until_calibration".into(),
        requested_path: requested_path.to_path_buf(),
        expected_fingerprint: String::new(),
        actual_fingerprint: String::new(),
        schema_version: RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION,
        candidate_or_annotation_count: candidate_count,
        retained_rank_depth: Some(requested_max_rank),
        manifest_sha256: String::new(),
        payload_sha256: String::new(),
        valid: false,
        reused: false,
        generation_allowed,
        catalog_fingerprints: catalog_fingerprints.to_vec(),
        original_source_uris: Vec::new(),
        current_source_uris: Vec::new(),
        portable_identity_valid: None,
        relocation_detected: None,
        failure_reason: Some(
            "raw external predictions were validated independently; this compact stage calibration identity depends on the stage's preliminary calibration_input_sha256 and is derived after native fitting"
                .into(),
        ),
    }
}

fn raw_prediction_preflight_report(
    usage: &ExternalAnnotationCacheUsage,
    search_space: &str,
    requested_path: &Path,
    generation_allowed: bool,
) -> Result<ResourcePreflightReport> {
    anyhow::ensure!(
        !usage.raw_prediction_cache_fingerprint.is_empty(),
        "strict raw-prediction preflight returned a legacy annotation cache"
    );
    Ok(ResourcePreflightReport {
        resource_type: "raw_external_prediction_cache".into(),
        search_space: search_space.into(),
        stage: None,
        status: "validated_exact".into(),
        requested_path: requested_path.to_path_buf(),
        expected_fingerprint: usage.raw_prediction_cache_fingerprint.clone(),
        actual_fingerprint: usage.raw_prediction_cache_fingerprint.clone(),
        schema_version: usage.raw_prediction_cache_schema_version,
        candidate_or_annotation_count: usage.annotation_count,
        retained_rank_depth: Some(usage.requested_max_rank as usize),
        manifest_sha256: sha256_file(&usage.manifest)?,
        payload_sha256: sha256_file(&usage.payload)?,
        valid: true,
        reused: true,
        generation_allowed,
        catalog_fingerprints: Vec::new(),
        original_source_uris: Vec::new(),
        current_source_uris: Vec::new(),
        portable_identity_valid: Some(true),
        relocation_detected: None,
        failure_reason: None,
    })
}

/// Resolve strict immutable resources without writing workflow state or
/// starting a search/annotation child process.
fn strict_resource_preflight(
    manifest: &WorkflowManifest,
    parallel: usize,
) -> Result<Vec<ResourcePreflightReport>> {
    let annotation_required = manifest
        .models
        .iter()
        .any(|model| model.enabled && !matches!(model.ms2rescore, Ms2RescorePolicy::Never));
    let entrapment_fasta = strict_preflight_fasta(manifest)?;
    let mut spaces = vec![("+entrapment", entrapment_fasta, false)];
    if !manifest
        .parameter_optimizer
        .as_ref()
        .is_some_and(|optimizer| {
            optimizer.enabled && (optimizer.production_smoke_only || optimizer.optimization_only())
        })
    {
        spaces.push(("target_only", manifest.target_fasta.clone(), true));
    }
    let mut reports = Vec::new();
    for (search_space, fasta, target_only) in spaces {
        let mut input = Input::load(manifest.search_config.to_string_lossy().as_ref())?;
        input.database.fasta = Some(fasta.display().to_string());
        input.mzml_paths = Some(manifest.spectra.clone());
        // `Input::build` creates a configured local output directory. Leave it
        // unset so strict plan/preflight remains read-only.
        input.output_directory = None;
        input.annotate_matches = Some(false);
        let fdr = input.fdr.get_or_insert_with(FdrOptions::default);
        fdr.mode = Some(FdrMode::DecoyFree);
        fdr.model_fit = Some(ModelFit::Moments);
        let parameters = input.build()?;
        let runner = Runner::new(parameters, parallel)?;
        let candidate_root = manifest.resolved_candidate_pool_root();
        let candidate_request = CandidatePoolRequest {
            root: candidate_root.clone(),
            required_rank_depth: requested_rank_depth(manifest, &runner),
            allow_reuse: true,
            require_existing: true,
        };
        let (candidate_usage, candidates) = runner
            .preflight_existing_candidate_pool(&candidate_request)
            .with_context(|| {
                format!(
                    "strict resource preflight failed for {search_space} candidate pool; generation_prohibited=true"
                )
            })?;
        reports.push(candidate_preflight_report(
            &candidate_usage,
            search_space,
            &candidate_root,
            !manifest.require_existing_candidate_pool,
        )?);

        if annotation_required {
            let cache_root = manifest.resolved_annotation_cache_root(target_only);
            let max_rank = runner
                .parameters
                .external_features
                .max_rank
                .unwrap_or(runner.parameters.report_psms as u32);
            let catalog_fingerprints = if manifest.require_existing_annotation_cache {
                let cache_request = ExternalAnnotationCacheRequest {
                    root: cache_root.clone(),
                    require_existing: true,
                    search_space: search_space.into(),
                    stage: "static_preflight".into(),
                    analysis_fingerprint: candidate_usage.search_fingerprint.clone(),
                    migration_only: false,
                };
                let database = runner.shared_database();
                let candidate_ids = candidates
                    .iter()
                    .map(|candidate| {
                        stable_candidate_id(
                            &candidate_usage.search_fingerprint,
                            candidate,
                            &database[candidate.peptide_idx].to_string(),
                        )
                    })
                    .collect::<HashSet<_>>();
                anyhow::ensure!(
                    candidate_ids.len() == candidates.len(),
                    "strict annotation-cache preflight found duplicate stable candidate IDs in {search_space} candidate pool"
                );
                let usages = preflight_existing_cache_root(
                    &cache_request,
                    &runner.parameters.external_features,
                    &candidate_usage.search_fingerprint,
                    &candidate_ids,
                    max_rank,
                )?;
                anyhow::ensure!(
                    usages.len() == 1,
                    "strict raw-prediction preflight returned {} cache records; expected exactly one",
                    usages.len()
                );
                reports.push(raw_prediction_preflight_report(
                    &usages[0],
                    search_space,
                    &cache_root,
                    false,
                )?);
                vec![usages[0].raw_prediction_cache_fingerprint.clone()]
            } else {
                Vec::new()
            };
            for model in manifest.models.iter().filter(|model| {
                model.enabled && !matches!(model.ms2rescore, Ms2RescorePolicy::Never)
            }) {
                reports.push(deferred_annotation_preflight_report(
                    model,
                    search_space,
                    &cache_root,
                    candidate_usage.candidate_count,
                    max_rank as usize,
                    !manifest.require_existing_annotation_cache,
                    &catalog_fingerprints,
                ));
            }
        }
    }
    Ok(reports)
}

fn apply_window(options: &mut FdrOptions, model: &ModelFit, window: &Option<NullWindow>) {
    let Some(window) = window else {
        return;
    };
    match model {
        ModelFit::Moments => {
            options.moments_min_null_rank = Some(window.min_rank);
            options.moments_max_null_rank = Some(window.max_rank);
        }
        ModelFit::Mle => {
            options.mle_min_null_rank = Some(window.min_rank);
            options.mle_max_null_rank = Some(window.max_rank);
        }
        ModelFit::LowerOrder => {
            options.lower_order_min_null_rank = Some(window.min_rank);
            options.lower_order_max_null_rank = Some(window.max_rank);
        }
        ModelFit::Msfdr => {
            options.msfdr_min_null_rank = Some(window.min_rank);
            options.msfdr_max_null_rank = Some(window.max_rank);
        }
        ModelFit::Msfdr2Smix => {
            options.msfdr2_smix_min_null_rank = Some(window.min_rank);
            options.msfdr2_smix_max_null_rank = Some(window.max_rank);
        }
        ModelFit::Nokoi => {
            options.nokoi_min_null_rank = Some(window.min_rank);
            options.nokoi_max_null_rank = Some(window.max_rank);
        }
        ModelFit::Msfdr1Smix | ModelFit::Ensemble => {}
    }
}

fn resolved_expert_window(model: &ModelFit, window: &Option<NullWindow>) -> Option<NullWindow> {
    if *model == ModelFit::Msfdr1Smix {
        Some(NullWindow {
            min_rank: 1,
            max_rank: 1,
        })
    } else {
        window.clone()
    }
}

fn artifact_contains_model(
    artifacts: &sage_core::decoy_free_fdr::DfRunArtifacts,
    model: &ModelFit,
) -> bool {
    let valid_skew_normal = |model: &sage_core::ml::skew_normal::SkewNormal| {
        model.location.is_finite()
            && model.scale.is_finite()
            && model.scale > 0.0
            && model.shape.is_finite()
    };
    match model {
        ModelFit::Moments => artifacts.moments.as_ref().is_some_and(|artifact| {
            artifact.schema_version == 1
                && artifact.model_version == "sage-moments-gumbel-v1"
                && artifact.min_rank > 1
                && artifact.max_rank >= artifact.min_rank
                && artifact.mu.is_finite()
                && artifact.beta.is_finite()
                && artifact.beta > 0.0
        }),
        ModelFit::Mle => artifacts.mle.as_ref().is_some_and(|artifact| {
            artifact.schema_version == 1
                && artifact.model_version == "sage-mle-gumbel-v1"
                && artifact.min_rank > 1
                && artifact.max_rank >= artifact.min_rank
                && artifact.mu.is_finite()
                && artifact.beta.is_finite()
                && artifact.beta > 0.0
        }),
        ModelFit::LowerOrder => artifacts.lower_order.as_ref().is_some_and(|artifact| {
            sage_core::ml::lower_order::LowerOrderModel::from_artifact(artifact).is_ok()
        }),
        ModelFit::Msfdr => {
            artifacts.msfdr_seeded.as_ref().is_some_and(|model| {
                model.null_loc.is_finite()
                    && model.null_scale.is_finite()
                    && model.null_scale > 0.0
                    && model.target_mean.is_finite()
                    && model.target_std.is_finite()
                    && model.target_std > 0.0
                    && model.target_alpha.is_finite()
                    && model.pi.is_finite()
                    && (0.0..=1.0).contains(&model.pi)
            }) && artifacts
                .msfdr_seeded_metadata
                .as_ref()
                .is_some_and(|metadata| {
                    metadata.schema_version == 1
                        && metadata.model_version == "sage-msfdr-seeded-v1"
                        && !metadata.rank1_only
                        && metadata.min_null_rank.is_some_and(|rank| rank > 1)
                        && metadata
                            .max_null_rank
                            .zip(metadata.min_null_rank)
                            .is_some_and(|(max, min)| max >= min)
                })
        }
        ModelFit::Msfdr1Smix => {
            artifacts.msfdr_1smix.as_ref().is_some_and(|model| {
                valid_skew_normal(&model.correct)
                    && valid_skew_normal(&model.incorrect1)
                    && model.a.is_finite()
                    && (0.0..=1.0).contains(&model.a)
            }) && artifacts
                .msfdr_1smix_metadata
                .as_ref()
                .is_some_and(|metadata| {
                    metadata.schema_version == 1
                        && metadata.model_version == "sage-msfdr-1smix-v1"
                        && metadata.rank1_only
                        && metadata.min_null_rank.is_none()
                        && metadata.max_null_rank.is_none()
                })
        }
        ModelFit::Msfdr2Smix => {
            artifacts.msfdr_2smix.as_ref().is_some_and(|model| {
                valid_skew_normal(&model.correct)
                    && valid_skew_normal(&model.incorrect1)
                    && valid_skew_normal(&model.incorrect2)
                    && model.a.is_finite()
                    && model.b.is_finite()
                    && (0.0..=1.0).contains(&model.a)
                    && (0.0..=1.0).contains(&model.b)
                    && model.a + model.b <= 1.0 + f64::EPSILON
            }) && artifacts
                .msfdr_2smix_metadata
                .as_ref()
                .is_some_and(|metadata| {
                    metadata.schema_version == 1
                        && metadata.model_version == "sage-msfdr-2smix-v1"
                        && !metadata.rank1_only
                        && metadata.min_null_rank.is_some_and(|rank| rank > 1)
                        && metadata
                            .max_null_rank
                            .zip(metadata.min_null_rank)
                            .is_some_and(|(max, min)| max >= min)
                })
        }
        ModelFit::Nokoi => artifacts
            .nokoi
            .as_ref()
            .is_some_and(|artifact| artifact.validate_portable().is_ok()),
        ModelFit::Ensemble => false,
    }
}

fn fitted_external_profile_identity(
    artifacts: &DfRunArtifacts,
) -> Result<Option<(String, sage_core::input::ExternalProfileCalibration)>> {
    artifacts
        .external_ms2rescore
        .as_ref()
        .map(|profiles| {
            let mut hasher = Sha256::new();
            hasher.update(b"sage-fitted-external-profile-content-identity-v1\0");
            hasher.update(serde_json::to_vec(profiles)?);
            Ok((
                format!("{:x}", hasher.finalize()),
                profiles.calibration.clone(),
            ))
        })
        .transpose()
}

fn shared_ensemble_profile_contract_identity(
    dataset: &DatasetIdentity,
    calibration: &sage_core::input::ExternalProfileCalibration,
    experts: &[EnsembleExpertLock],
) -> Result<String> {
    let fit_search_fingerprints = experts
        .iter()
        .filter(|expert| expert.enabled)
        .map(|expert| expert.fit_search_fingerprint.as_str())
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        fit_search_fingerprints.len() == 1,
        "shared Ensemble external profile requires exactly one fit search fingerprint"
    );
    let fit_search_fingerprint = fit_search_fingerprints
        .into_iter()
        .next()
        .context("shared Ensemble external profile has no fit search fingerprint")?;
    let value = serde_json::json!({
        "schema": "sage-ensemble-shared-external-profile-contract-v2",
        "dataset_fingerprint": dataset.fingerprint,
        "source_configuration_sha256": dataset.search_config_sha256,
        "fit_search_fingerprint": fit_search_fingerprint,
        "candidate_id_schema": CANDIDATE_ID_SCHEMA,
        "calibration": calibration,
    });
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(&value)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn apply_fitted_artifacts(
    fdr: &mut FdrOptions,
    model: &ModelFit,
    artifacts: sage_core::decoy_free_fdr::DfRunArtifacts,
    apply_external_profile: bool,
    target_only_policy: Option<TargetOnlyCalibrationPolicy>,
) -> Result<()> {
    if let Some(policy) = target_only_policy {
        anyhow::ensure!(
            policy != TargetOnlyCalibrationPolicy::CompareBoth,
            "compare_both must be resolved before artifact application"
        );
        let capability = target_only_policy_capability(model, policy);
        anyhow::ensure!(
            capability.supported,
            "{} does not support target-only policy {}: {}",
            model_slug(model),
            policy.stage_name(),
            capability.reason.as_deref().unwrap_or("unsupported policy")
        );
    }
    anyhow::ensure!(
        artifact_contains_model(&artifacts, model),
        "fitted artifact does not contain {:?}",
        model
    );
    if let Some(profiles) = artifacts.external_ms2rescore.as_ref() {
        anyhow::ensure!(
            profiles.schema_version == 2
                && profiles.model_version == "sage-external-ms2rescore-profiles-v2-explicit-window",
            "external MS2Rescore fitted artifact is not portable or has an unsupported version"
        );
    }
    if apply_external_profile {
        if let Some(profiles) = artifacts.external_ms2rescore.as_ref() {
            if let (Some(min_rank), Some(max_rank)) = (
                fdr.external_profile_min_null_rank,
                fdr.external_profile_max_null_rank,
            ) {
                anyhow::ensure!(
                    min_rank == profiles.calibration.min_null_rank
                        && max_rank == profiles.calibration.max_null_rank,
                    "configured external-profile window {}..={} disagrees with fitted artifact {}..={}",
                    min_rank,
                    max_rank,
                    profiles.calibration.min_null_rank,
                    profiles.calibration.max_null_rank
                );
            }
            fdr.external_profile_min_null_rank = Some(profiles.calibration.min_null_rank);
            fdr.external_profile_max_null_rank = Some(profiles.calibration.max_null_rank);
        }
        fdr.external_ms2rescore_frozen_profiles = artifacts.external_ms2rescore.clone();
    }
    match model {
        ModelFit::Moments => {
            let artifact = artifacts.moments.context("Moments artifact is missing")?;
            fdr.moments_min_null_rank = Some(artifact.min_rank);
            fdr.moments_max_null_rank = Some(artifact.max_rank);
            fdr.moments_frozen_parameters = Some(artifact);
        }
        ModelFit::Mle => {
            let artifact = artifacts.mle.context("MLE artifact is missing")?;
            fdr.mle_min_null_rank = Some(artifact.min_rank);
            fdr.mle_max_null_rank = Some(artifact.max_rank);
            fdr.mle_frozen_parameters = Some(artifact);
        }
        ModelFit::LowerOrder => {
            let artifact = artifacts
                .lower_order
                .context("Lower Order artifact is missing")?;
            fdr.lower_order_min_null_rank = Some(artifact.null_rank_min);
            fdr.lower_order_max_null_rank = Some(artifact.null_rank_max);
            fdr.lower_order_frozen_artifact = Some(artifact);
        }
        ModelFit::Msfdr => {
            let metadata = artifacts
                .msfdr_seeded_metadata
                .context("MSFDR artifact metadata is missing")?;
            fdr.msfdr_min_null_rank = metadata.min_null_rank;
            fdr.msfdr_max_null_rank = metadata.max_null_rank;
            fdr.msfdr_seeded_frozen_model = artifacts.msfdr_seeded;
        }
        ModelFit::Msfdr1Smix => fdr.msfdr_1smix_frozen_model = artifacts.msfdr_1smix,
        ModelFit::Msfdr2Smix => {
            let metadata = artifacts
                .msfdr_2smix_metadata
                .context("MSFDR2-SMIX artifact metadata is missing")?;
            fdr.msfdr2_smix_min_null_rank = metadata.min_null_rank;
            fdr.msfdr2_smix_max_null_rank = metadata.max_null_rank;
            fdr.msfdr_2smix_frozen_model = artifacts.msfdr_2smix;
        }
        ModelFit::Nokoi => {
            let artifact = artifacts.nokoi.context("Nokoi artifact is missing")?;
            fdr.nokoi_min_null_rank = Some(artifact.min_null_rank);
            fdr.nokoi_max_null_rank = Some(artifact.max_null_rank);
            fdr.nokoi_artifact_application_mode = Some(match target_only_policy {
                Some(TargetOnlyCalibrationPolicy::ReuseDatasetArtifact) => {
                    sage_core::ml::nokoi::NokoiArtifactApplicationMode::SameDatasetTargetOnly
                }
                _ => sage_core::ml::nokoi::NokoiArtifactApplicationMode::ExactFitPopulation,
            });
            fdr.nokoi_frozen_artifact = Some(artifact);
        }
        ModelFit::Ensemble => anyhow::bail!("use an Ensemble lock for Ensemble artifacts"),
    }
    Ok(())
}

fn apply_ensemble_lock(
    fdr: &mut FdrOptions,
    lock: &EnsembleLock,
    external: bool,
    dataset: &DatasetIdentity,
    policy: &ArtifactReusePolicy,
    reuse_fitted_artifacts: bool,
    target_only_policy: Option<TargetOnlyCalibrationPolicy>,
) -> Result<()> {
    fdr.nokoi_application_dataset_fingerprint = Some(dataset.fingerprint.clone());
    anyhow::ensure!(
        lock.schema_version == 8,
        "unsupported Ensemble lock schema {}; schema 8 is required for independent shared-profile and complete input-provenance identities",
        lock.schema_version
    );
    anyhow::ensure!(
        lock.evaluable,
        "Ensemble lock is not evaluable: {}",
        lock.not_evaluable_reasons.join("; ")
    );
    anyhow::ensure!(
        lock.external_profile_contract == "shared_dataset_local",
        "unsupported Ensemble external-profile contract {:?}",
        lock.external_profile_contract
    );
    anyhow::ensure!(
        matches!(
            lock.roster_contract.as_str(),
            "json_requested_technical_validation_only" | "interaction_diagnostic_baseline"
        ),
        "unsupported Ensemble roster contract {:?}",
        lock.roster_contract
    );
    anyhow::ensure!(
        !lock.analysis_fingerprint.is_empty()
            && lock.analysis_fingerprint == ensemble_lock_analysis_fingerprint(lock)?,
        "Ensemble lock analysis fingerprint is missing or does not match its payload"
    );
    anyhow::ensure!(
        lock.source_configuration_sha256 == dataset.search_config_sha256,
        "Ensemble lock configuration fingerprint does not match this dataset"
    );
    let resolved_settings = FdrSettings::from(fdr.clone());
    anyhow::ensure!(
        resolved_settings.ensemble_p_combiner == lock.ensemble_p_combiner
            && resolved_settings.ensemble_pep_combiner == lock.ensemble_pep_combiner,
        "Ensemble lock combiner settings do not match the resolved search configuration"
    );
    if external {
        anyhow::ensure!(
            lock.shared_external_profile_contract_sha256.is_some()
                && lock.shared_external_profile_calibration.is_some(),
            "external Ensemble lock has no shared external-profile identity"
        );
        anyhow::ensure!(
            lock.shared_external_profile_calibration
                .as_ref()
                .is_some_and(|calibration| {
                    calibration.min_null_rank == 9
                        && calibration.max_null_rank == 18
                        && calibration.provenance
                            == sage_core::input::ExternalProfileWindowProvenance::ExplicitConfiguration
                }),
            "external Ensemble lock does not use the explicit shared 9-18 calibration contract"
        );
        anyhow::ensure!(
            lock.shared_external_profile_contract_sha256.as_deref()
                == Some(
                    shared_ensemble_profile_contract_identity(
                        dataset,
                        lock.shared_external_profile_calibration.as_ref().unwrap(),
                        &lock.experts,
                    )?
                    .as_str(),
                ),
            "Ensemble lock shared external-profile contract identity is stale or inconsistent"
        );
    }
    // The external empirical calibration is dataset-local auxiliary evidence
    // shared by the assembled Ensemble. Expert artifacts contribute only their
    // base-model nuisance state; no expert may overwrite this shared profile.
    fdr.external_ms2rescore_frozen_profiles = None;
    fdr.enable_moments = Some(false);
    fdr.enable_mle = Some(false);
    fdr.enable_lower_order = Some(false);
    fdr.enable_msfdr_seeded = Some(false);
    fdr.enable_msfdr_1smix = Some(false);
    fdr.enable_msfdr_2smix = Some(false);
    fdr.enable_nokoi = Some(false);
    let enabled = lock.experts.iter().filter(|expert| expert.enabled).count();
    let all_models = lock
        .experts
        .iter()
        .map(|expert| model_slug(&expert.model))
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        all_models.len() == lock.experts.len(),
        "Ensemble lock contains duplicate canonical model entries"
    );
    let unique_enabled = lock
        .experts
        .iter()
        .filter(|expert| expert.enabled)
        .map(|expert| model_slug(&expert.model))
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        unique_enabled.len() == enabled,
        "Ensemble lock contains duplicate enabled expert models; refusing last-expert-wins artifact application"
    );
    let mut requested = lock.requested_roster.clone();
    requested.sort();
    let requested_unique = requested.iter().collect::<BTreeSet<_>>();
    anyhow::ensure!(
        requested == lock.requested_roster && requested_unique.len() == requested.len(),
        "Ensemble lock requested roster is not canonical and unique"
    );
    let actual = unique_enabled
        .iter()
        .map(|model| (*model).to_owned())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        actual == lock.actual_roster,
        "Ensemble lock actual roster disagrees with enabled experts"
    );
    anyhow::ensure!(
        lock.actual_roster
            .iter()
            .all(|model| requested_unique.contains(model)),
        "Ensemble lock actual roster contains an unrequested expert"
    );
    for (model, failures) in &lock.technical_failures {
        anyhow::ensure!(
            requested_unique.contains(model)
                && !lock.actual_roster.contains(model)
                && !failures.is_empty(),
            "Ensemble lock has inconsistent technical-failure provenance for {model}"
        );
    }
    for model in &lock.requested_roster {
        anyhow::ensure!(
            lock.actual_roster.contains(model) ^ lock.technical_failures.contains_key(model),
            "requested Ensemble expert {model} must appear exactly once in the actual roster or technical failures"
        );
    }
    anyhow::ensure!(
        lock.explicit_exclusions
            .keys()
            .all(|model| !requested_unique.contains(model)),
        "Ensemble lock explicit exclusions overlap the requested roster"
    );
    for expert in &lock.experts {
        let model = model_slug(&expert.model);
        let was_requested = requested_unique.contains(&model.to_owned());
        if expert.enabled {
            anyhow::ensure!(
                was_requested
                    && expert.participation_decision == "included_technical_validation_passed"
                    && expert.gate_reasons.is_empty(),
                "enabled Ensemble expert {model} has inconsistent roster provenance"
            );
        } else if was_requested
            && lock.roster_contract == "json_requested_technical_validation_only"
        {
            anyhow::ensure!(
                expert.participation_decision == "excluded_technical_failure"
                    && lock
                        .technical_failures
                        .get(model)
                        .is_some_and(|failures| failures == &expert.gate_reasons),
                "requested disabled Ensemble expert {model} lacks matching technical-failure provenance"
            );
        } else if !was_requested {
            anyhow::ensure!(
                lock.explicit_exclusions.contains_key(model),
                "unrequested Ensemble expert {model} lacks explicit-exclusion provenance"
            );
        }
    }
    let mut optimized_hashes = BTreeSet::new();
    let mut ms2_hashes = BTreeSet::new();
    for expert in lock.experts.iter().filter(|expert| expert.enabled) {
        anyhow::ensure!(
            !expert.optimized_fitted_artifacts_sha256.is_empty()
                && optimized_hashes.insert(expert.optimized_fitted_artifacts_sha256.as_str()),
            "Ensemble lock contains a duplicate optimized artifact vote"
        );
        if let Some(hash) = expert.ms2rescore_fitted_artifacts_sha256.as_deref() {
            anyhow::ensure!(
                ms2_hashes.insert(hash),
                "Ensemble lock contains a duplicate MS2Rescore artifact vote"
            );
        }
    }
    anyhow::ensure!(
        enabled >= lock.minimum_required_experts,
        "Ensemble lock has only {enabled} eligible experts"
    );
    for expert in lock.experts.iter().filter(|expert| expert.enabled) {
        anyhow::ensure!(
            expert.participation_decision == "included_technical_validation_passed"
                && !expert.fallback_used
                && expert.fallback_reason.is_none(),
            "Ensemble expert {:?} has inconsistent participation or fallback provenance",
            expert.model
        );
        anyhow::ensure!(
            expert.optimized_fitted_artifacts.is_file()
                && sha256_file(&expert.optimized_fitted_artifacts)?
                    == expert.optimized_fitted_artifacts_sha256,
            "Ensemble expert {:?} optimized artifact is missing or its hash changed",
            expert.model
        );
        let optimized_artifacts: DfRunArtifacts =
            serde_json::from_slice(&std::fs::read(&expert.optimized_fitted_artifacts)?)
                .context("Ensemble optimized artifact payload is unreadable")?;
        anyhow::ensure!(
            artifact_contains_model(&optimized_artifacts, &expert.model),
            "Ensemble expert optimized artifact does not contain a valid {:?} model",
            expert.model
        );
        if let Some(path) = expert.ms2rescore_fitted_artifacts.as_ref() {
            let expected = expert
                .ms2rescore_fitted_artifacts_sha256
                .as_deref()
                .context("MS2Rescore artifact hash is missing")?;
            anyhow::ensure!(
                path.is_file() && sha256_file(path)? == expected,
                "Ensemble expert {:?} MS2Rescore artifact is missing or its hash changed",
                expert.model
            );
            let artifacts: DfRunArtifacts = serde_json::from_slice(&std::fs::read(path)?)
                .context("Ensemble MS2Rescore artifact payload is unreadable")?;
            anyhow::ensure!(
                artifact_contains_model(&artifacts, &expert.model),
                "Ensemble expert MS2Rescore artifact does not contain a valid {:?} model",
                expert.model
            );
            if external {
                let (identity, calibration) = fitted_external_profile_identity(&artifacts)?
                    .context(
                        "Ensemble expert MS2Rescore artifact has no fitted external profile",
                    )?;
                anyhow::ensure!(
                    expert.fitted_external_profile_identity_sha256.as_deref()
                        == Some(identity.as_str())
                        && expert.fitted_external_profile_calibration.as_ref()
                            == Some(&calibration),
                    "Ensemble expert {:?} fitted external-profile provenance disagrees with its artifact",
                    expert.model
                );
            }
        } else if external {
            anyhow::bail!(
                "Ensemble expert {:?} has no MS2Rescore fitted artifact",
                expert.model
            );
        }
        if external {
            anyhow::ensure!(
                expert.fitted_external_profile_identity_sha256.is_some()
                    && expert.fitted_external_profile_calibration.as_ref().is_some_and(|calibration| {
                        calibration.min_null_rank == 9
                            && calibration.max_null_rank == 18
                            && calibration.provenance
                                == sage_core::input::ExternalProfileWindowProvenance::ExplicitConfiguration
                    }),
                "Ensemble expert {:?} has invalid fitted external-profile provenance",
                expert.model
            );
            anyhow::ensure!(
                expert.annotation_cache_fingerprint.is_some()
                    && expert.annotation_cache_manifest_sha256.is_some()
                    && expert.annotation_cache_payload_sha256.is_some(),
                "Ensemble expert {:?} has incomplete annotation-cache provenance",
                expert.model
            );
        }
        if lock.post_selection_in_scope {
            anyhow::ensure!(
                expert.target_only_policy_capability.as_ref()
                    == Some(&target_only_policy_capability(
                        &expert.model,
                        expert.target_only_calibration_policy,
                    )),
                "Ensemble expert {:?} target-only capability provenance is missing or stale",
                expert.model
            );
        } else {
            anyhow::ensure!(
                expert.target_only_results.as_os_str().is_empty()
                    && expert.target_only_policy_capability.is_none(),
                "optimizer-only Ensemble lock contains target-only provenance"
            );
        }
        if let Some(target_policy) = target_only_policy {
            let capability = target_only_policy_capability(&expert.model, target_policy);
            anyhow::ensure!(
                capability.supported,
                "{} does not support target-only policy {}: {}",
                model_slug(&expert.model),
                target_policy.stage_name(),
                capability.reason.as_deref().unwrap_or("unsupported policy")
            );
        }
        apply_window(fdr, &expert.model, &expert.window);
        match expert.model {
            ModelFit::Moments => fdr.enable_moments = Some(true),
            ModelFit::Mle => fdr.enable_mle = Some(true),
            ModelFit::LowerOrder => fdr.enable_lower_order = Some(true),
            ModelFit::Msfdr => fdr.enable_msfdr_seeded = Some(true),
            ModelFit::Msfdr1Smix => fdr.enable_msfdr_1smix = Some(true),
            ModelFit::Msfdr2Smix => fdr.enable_msfdr_2smix = Some(true),
            ModelFit::Nokoi => fdr.enable_nokoi = Some(true),
            ModelFit::Ensemble => anyhow::bail!("nested Ensemble expert is invalid"),
        }
        if !reuse_fitted_artifacts {
            continue;
        }
        let (artifact_path, expected_sha256) = if external {
            (
                expert
                    .ms2rescore_fitted_artifacts
                    .as_ref()
                    .with_context(|| {
                        format!(
                            "Ensemble expert {:?} has no MS2Rescore-fitted artifact",
                            expert.model
                        )
                    })?,
                expert
                    .ms2rescore_fitted_artifacts_sha256
                    .as_deref()
                    .context("MS2Rescore artifact hash is missing")?,
            )
        } else {
            (
                &expert.optimized_fitted_artifacts,
                expert.optimized_fitted_artifacts_sha256.as_str(),
            )
        };
        let artifacts: sage_core::decoy_free_fdr::DfRunArtifacts =
            serde_json::from_slice(&std::fs::read(artifact_path).with_context(|| {
                format!("missing fitted artifact {}", artifact_path.display())
            })?)?;
        anyhow::ensure!(
            sha256_file(artifact_path)? == expected_sha256,
            "Ensemble expert {:?} {} artifact hash changed",
            expert.model,
            if external { "MS2Rescore" } else { "optimized" }
        );
        anyhow::ensure!(
            artifact_contains_model(&artifacts, &expert.model),
            "Ensemble expert artifact does not contain {:?}",
            expert.model
        );
        match policy {
            ArtifactReusePolicy::DatasetLocalOnly => anyhow::ensure!(
                expert.candidate_id_schema == CANDIDATE_ID_SCHEMA
                    && !expert.fit_search_fingerprint.is_empty(),
                "Ensemble expert {:?} search provenance is missing or incompatible",
                expert.model
            ),
            ArtifactReusePolicy::CrossDatasetDiagnostic
                if expert.candidate_id_schema != CANDIDATE_ID_SCHEMA
                    || expert.fit_search_fingerprint.is_empty() =>
            {
                log::warn!(
                    "cross-dataset diagnostic Ensemble expert {:?} lacks Phase 5 search provenance",
                    expert.model
                );
            }
            ArtifactReusePolicy::CrossDatasetDiagnostic => {}
        }
        validate_artifact_reuse(
            &artifacts,
            dataset,
            policy,
            &expert.model,
            Some(&expert.fit_search_fingerprint),
        )?;
        apply_fitted_artifacts(fdr, &expert.model, artifacts, false, target_only_policy)?;
    }
    Ok(())
}

fn canonicalize_ensemble_lock(mut lock: EnsembleLock) -> EnsembleLock {
    lock.experts
        .sort_by(|left, right| model_slug(&left.model).cmp(model_slug(&right.model)));
    for expert in &mut lock.experts {
        expert.gate_reasons.sort();
        expert.gate_reasons.dedup();
        expert.gate_warnings.sort();
        expert.gate_warnings.dedup();
    }
    lock.requested_roster.sort();
    lock.actual_roster.sort();
    for failures in lock.technical_failures.values_mut() {
        failures.sort();
        failures.dedup();
    }
    lock.not_evaluable_reasons.sort();
    lock.not_evaluable_reasons.dedup();
    lock
}

fn ensemble_lock_analysis_fingerprint(lock: &EnsembleLock) -> Result<String> {
    let experts = lock
        .experts
        .iter()
        .map(|expert| {
            serde_json::json!({
                "model": expert.model,
                "window": expert.window,
                "optimized_fitted_artifacts_sha256": expert.optimized_fitted_artifacts_sha256,
                "ms2rescore_fitted_artifacts_sha256": expert.ms2rescore_fitted_artifacts_sha256,
                "target_only_calibration_policy": lock.post_selection_in_scope.then_some(expert.target_only_calibration_policy),
                "enabled": expert.enabled,
                "interaction_baseline": expert.interaction_baseline,
                "participation_decision": expert.participation_decision,
                "fallback_used": expert.fallback_used,
                "fallback_reason": expert.fallback_reason,
                "target_only_policy_capability": expert.target_only_policy_capability,
                "gate_reasons": expert.gate_reasons,
                "gate_warnings": expert.gate_warnings,
                "fit_search_fingerprint": expert.fit_search_fingerprint,
                "candidate_id_schema": expert.candidate_id_schema,
                "fitted_external_profile_identity_sha256": expert.fitted_external_profile_identity_sha256,
                "fitted_external_profile_calibration": expert.fitted_external_profile_calibration,
                "annotation_cache_fingerprint": expert.annotation_cache_fingerprint,
                "annotation_cache_manifest_sha256": expert.annotation_cache_manifest_sha256,
                "annotation_cache_payload_sha256": expert.annotation_cache_payload_sha256,
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "schema": "sage-ensemble-analysis-v4-independent-shared-profile-contract",
        "post_selection_in_scope": lock.post_selection_in_scope,
        "dataset_fingerprint": lock.dataset_fingerprint,
        "source_configuration_sha256": lock.source_configuration_sha256,
        "experts": experts,
        "requested_roster": lock.requested_roster,
        "actual_roster": lock.actual_roster,
        "explicit_exclusions": lock.explicit_exclusions,
        "technical_failures": lock.technical_failures,
        "roster_contract": lock.roster_contract,
        "minimum_required_experts": lock.minimum_required_experts,
        "evaluable": lock.evaluable,
        "not_evaluable_reasons": lock.not_evaluable_reasons,
        "external_profile_contract": lock.external_profile_contract,
        "shared_external_profile_contract_sha256": lock.shared_external_profile_contract_sha256,
        "shared_external_profile_calibration": lock.shared_external_profile_calibration,
        "raw_q_interaction_warning_threshold": lock.raw_q_interaction_warning_threshold,
        "ensemble_p_combiner": lock.ensemble_p_combiner,
        "ensemble_pep_combiner": lock.ensemble_pep_combiner,
    });
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(&value)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn stamp_ensemble_lock_analysis_fingerprint(mut lock: EnsembleLock) -> Result<EnsembleLock> {
    lock = canonicalize_ensemble_lock(lock);
    lock.analysis_fingerprint = ensemble_lock_analysis_fingerprint(&lock)?;
    Ok(lock)
}

fn interaction_baseline_lock(lock: &EnsembleLock) -> Result<EnsembleLock> {
    let mut baseline = lock.clone();
    baseline.roster_contract = "interaction_diagnostic_baseline".into();
    for expert in &mut baseline.experts {
        if expert.enabled && !expert.interaction_baseline {
            expert.enabled = false;
            expert.participation_decision = "interaction_baseline_only_exclusion".into();
            expert
                .gate_reasons
                .push("not a member of the preregistered Ensemble interaction baseline".into());
            baseline.explicit_exclusions.insert(
                model_slug(&expert.model).into(),
                "nonblocking interaction-diagnostic baseline exclusion".into(),
            );
        }
    }
    baseline.actual_roster = baseline
        .experts
        .iter()
        .filter(|expert| expert.enabled)
        .map(|expert| model_slug(&expert.model).into())
        .collect();
    baseline.actual_roster.sort();
    let enabled = baseline
        .experts
        .iter()
        .filter(|expert| expert.enabled)
        .count();
    baseline.evaluable = enabled >= baseline.minimum_required_experts;
    baseline.not_evaluable_reasons = (!baseline.evaluable)
        .then(|| {
            vec![format!(
                "interaction baseline has only {enabled} eligible experts; {} required",
                baseline.minimum_required_experts
            )]
        })
        .unwrap_or_default();
    stamp_ensemble_lock_analysis_fingerprint(baseline)
}

fn unavailable_interaction_report(
    baseline_experts: Vec<String>,
    final_experts: Vec<String>,
    reason: impl Into<String>,
) -> EnsembleInteractionCalibration {
    let baseline = baseline_experts.iter().cloned().collect::<BTreeSet<_>>();
    EnsembleInteractionCalibration {
        schema_version: 2,
        baseline_lock_analysis_fingerprint: None,
        final_lock_analysis_fingerprint: None,
        newly_participating_experts: final_experts
            .iter()
            .filter(|expert| !baseline.contains(*expert))
            .cloned()
            .collect(),
        baseline_experts,
        final_experts,
        raw_q: None,
        level4: None,
        raw_q_warning: None,
        final_level4_calibration_pass: false,
        evaluable: false,
        participation_effect: "none_nonblocking_diagnostic".into(),
        not_evaluable_reason: Some(reason.into()),
    }
}

#[cfg(test)]
fn build_ensemble_lock(
    manifest: &WorkflowManifest,
    manifest_hash: &str,
    dataset: &DatasetIdentity,
    experts: &[CompletedExpert],
) -> Result<EnsembleLock> {
    build_ensemble_lock_with_failures(manifest, manifest_hash, dataset, experts, &BTreeMap::new())
}

fn build_ensemble_lock_with_failures(
    manifest: &WorkflowManifest,
    manifest_hash: &str,
    dataset: &DatasetIdentity,
    experts: &[CompletedExpert],
    runtime_failures: &BTreeMap<String, Vec<String>>,
) -> Result<EnsembleLock> {
    let optimization_only = manifest
        .parameter_optimizer
        .as_ref()
        .is_some_and(ParameterOptimizerConfig::optimization_only);
    let (ensemble_p_combiner, ensemble_pep_combiner) =
        resolved_ensemble_combiners(&manifest.search_config)?;
    struct LockCandidate<'a> {
        expert: &'a CompletedExpert,
        requested: bool,
        technical_failures: Vec<String>,
        diagnostic_warnings: Vec<String>,
        peptides: BTreeSet<String>,
        optimized_hash: String,
        ms2_hash: Option<String>,
        fallback_used: bool,
        fallback_reason: Option<String>,
    }

    let mut requested_roster = manifest
        .models
        .iter()
        .filter(|model| {
            model.enabled
                && model.model != ModelFit::Ensemble
                && model.ensemble_participation == EnsembleParticipation::Auto
        })
        .map(|model| model_slug(&model.model).to_owned())
        .collect::<Vec<_>>();
    requested_roster.sort();
    anyhow::ensure!(
        requested_roster.iter().collect::<BTreeSet<_>>().len() == requested_roster.len(),
        "requested Ensemble roster contains duplicate canonical models"
    );
    let explicit_exclusions = manifest
        .models
        .iter()
        .filter(|model| {
            model.enabled
                && model.model != ModelFit::Ensemble
                && model.ensemble_participation == EnsembleParticipation::Excluded
        })
        .map(|model| {
            (
                model_slug(&model.model).to_owned(),
                model
                    .ensemble_exclusion_reason
                    .clone()
                    .unwrap_or_else(|| "unspecified explicit JSON exclusion".into()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut candidates = Vec::new();
    let ensemble_uses_ms2 = manifest.models.iter().any(|model| {
        model.enabled
            && model.model == ModelFit::Ensemble
            && !matches!(model.ms2rescore, Ms2RescorePolicy::Never)
    });
    let shared_external_profile_calibration = ensemble_uses_ms2
        .then(|| resolved_external_profile_calibration(&manifest.search_config))
        .transpose()?;
    if let Some(calibration) = shared_external_profile_calibration.as_ref() {
        anyhow::ensure!(
            calibration.min_null_rank == 9
                && calibration.max_null_rank == 18
                && calibration.provenance
                    == sage_core::input::ExternalProfileWindowProvenance::ExplicitConfiguration,
            "Ensemble shared external profile must use the explicit dataset-local 9-18 calibration contract"
        );
    }
    for expert in experts {
        let mut failures = Vec::new();
        let mut warnings = Vec::new();
        let configuration = manifest
            .models
            .iter()
            .find(|configuration| configuration.model == expert.model);
        let requested = configuration.is_some_and(|configuration| {
            configuration.enabled
                && configuration.ensemble_participation == EnsembleParticipation::Auto
        });
        if configuration.is_none() {
            failures.push("model is absent from the workflow configuration".into());
        }

        let (optimized_hash, artifacts) = match std::fs::read(&expert.optimized_artifacts) {
            Ok(bytes) => {
                let hash = format!("{:x}", Sha256::digest(&bytes));
                match serde_json::from_slice::<DfRunArtifacts>(&bytes) {
                    Ok(artifact) => (hash, Some(artifact)),
                    Err(error) => {
                        failures.push(format!("optimized fitted artifact is unreadable: {error}"));
                        (String::new(), None)
                    }
                }
            }
            Err(error) => {
                failures.push(format!("optimized fitted artifact is missing: {error}"));
                (String::new(), None)
            }
        };
        let optimized_fallback = !artifacts
            .as_ref()
            .is_some_and(|artifact| artifact_contains_model(artifact, &expert.model));
        if optimized_fallback {
            failures.push("optimized model artifact is absent (possible fit fallback)".into());
        }
        if let Some(artifact) = artifacts.as_ref() {
            if let Err(error) = validate_artifact_reuse(
                artifact,
                dataset,
                &ArtifactReusePolicy::DatasetLocalOnly,
                &expert.model,
                Some(&expert.calibration_search_fingerprint),
            ) {
                failures.push(format!("optimized artifact provenance is invalid: {error}"));
            }
        }

        let mut ms2_hash = None;
        let mut ms2_fallback = false;
        if ensemble_uses_ms2 {
            match expert.ms2rescore_artifacts.as_ref() {
                Some(path) => match std::fs::read(path) {
                    Ok(bytes) => {
                        let hash = format!("{:x}", Sha256::digest(&bytes));
                        match serde_json::from_slice::<DfRunArtifacts>(&bytes) {
                            Ok(artifact) => {
                                ms2_fallback = !artifact_contains_model(&artifact, &expert.model);
                                if ms2_fallback {
                                    failures.push(
                                        "MS2Rescore model artifact is absent (possible fit fallback)"
                                            .into(),
                                    );
                                }
                                if let Err(error) = validate_artifact_reuse(
                                    &artifact,
                                    dataset,
                                    &ArtifactReusePolicy::DatasetLocalOnly,
                                    &expert.model,
                                    Some(&expert.calibration_search_fingerprint),
                                ) {
                                    failures.push(format!(
                                        "MS2Rescore artifact provenance is invalid: {error}"
                                    ));
                                }
                                match fitted_external_profile_identity(&artifact) {
                                    Ok(Some((identity, calibration))) => {
                                        if expert.fitted_external_profile_identity_sha256.as_deref()
                                            != Some(identity.as_str())
                                            || expert.fitted_external_profile_calibration.as_ref()
                                                != Some(&calibration)
                                        {
                                            failures.push(
                                                "MS2Rescore artifact external-profile identity disagrees with stage provenance"
                                                    .into(),
                                            );
                                        }
                                    }
                                    Ok(None) => failures.push(
                                        "MS2Rescore artifact has no fitted external profile".into(),
                                    ),
                                    Err(error) => failures.push(format!(
                                        "MS2Rescore external-profile identity is invalid: {error}"
                                    )),
                                }
                                ms2_hash = Some(hash);
                            }
                            Err(error) => {
                                ms2_fallback = true;
                                failures.push(format!(
                                    "MS2Rescore fitted artifact is unreadable: {error}"
                                ));
                            }
                        }
                    }
                    Err(error) => {
                        ms2_fallback = true;
                        failures.push(format!("MS2Rescore fitted artifact is missing: {error}"));
                    }
                },
                None => {
                    ms2_fallback = true;
                    failures.push("MS2Rescore fitted artifact is missing".into());
                }
            }
            if expert.fitted_external_profile_identity_sha256.is_none()
                || expert.fitted_external_profile_calibration.is_none()
            {
                failures.push("fitted external-profile provenance is missing".into());
            } else if !expert
                .fitted_external_profile_calibration
                .as_ref()
                .is_some_and(|calibration| {
                    calibration.min_null_rank == 9
                    && calibration.max_null_rank == 18
                    && calibration.provenance
                        == sage_core::input::ExternalProfileWindowProvenance::ExplicitConfiguration
                })
            {
                failures.push(
                    "fitted external-profile calibration must use the explicit 9-18 contract"
                        .into(),
                );
            }
            if expert.annotation_cache_fingerprint.is_none()
                || expert.annotation_cache_manifest_sha256.is_none()
                || expert.annotation_cache_payload_sha256.is_none()
            {
                failures.push("MS2Rescore annotation-cache provenance is missing".into());
            }
        }
        if !optimization_only {
            let capability =
                target_only_policy_capability(&expert.model, expert.target_only_calibration_policy);
            if !capability.supported {
                failures.push(format!(
                    "unsupported target-only policy {}: {}",
                    expert.target_only_calibration_policy.stage_name(),
                    capability.reason.as_deref().unwrap_or("unsupported policy")
                ));
            }
        }

        // Statistical validation remains available as explicitly nonblocking
        // diagnostics. It never changes `requested` or technical eligibility.
        let run = ValidationRun {
            method: model_slug(&expert.model).into(),
            stage: expert.calibration_stage.clone(),
            results: expert.calibration_results.clone(),
            mode: ValidationMode::DecoyFree,
            expected_search_space: Some("+Ent".into()),
            calibration_stage: None,
            target_only_calibration_policy: None,
            release_candidate: true,
        };
        let summaries = match summarize_run(
            &run,
            &manifest.validation.effective_ratios,
            manifest.validation.fdr_threshold,
        ) {
            Ok(rows) => rows,
            Err(error) => {
                warnings.push(format!(
                    "nonblocking diagnostic: calibration result is unavailable: {error}"
                ));
                Vec::new()
            }
        };
        for layer in ["raw_q", "level4"] {
            match summaries.iter().find(|row| row.layer == layer) {
                None => warnings.push(format!(
                    "nonblocking diagnostic: missing {layer} calibration summary"
                )),
                Some(row) => {
                    if row.peptide.entrapment
                        < manifest
                            .validation
                            .minimum_entrapment_peptides_for_stable_estimate
                    {
                        warnings.push(format!(
                            "nonblocking diagnostic: {layer} peptide entrapment count {} is below the reporting reference {}",
                            row.peptide.entrapment,
                            manifest
                                .validation
                                .minimum_entrapment_peptides_for_stable_estimate
                        ));
                    }
                    if !row
                        .peptide
                        .combined_entrapment_fdp
                        .is_some_and(|fdp| fdp <= manifest.validation.fdr_threshold)
                    {
                        warnings.push(format!(
                            "nonblocking diagnostic: {layer} peptide entrapment FDP exceeds the reporting threshold"
                        ));
                    }
                }
            }
        }
        let peptides = match accepted_target_peptides(
            &expert.calibration_results,
            &ValidationMode::DecoyFree,
            manifest.validation.fdr_threshold,
            manifest.validation.null_window_validation_scope == NullWindowValidationScope::Level4,
        ) {
            Ok(peptides) => peptides,
            Err(error) => {
                warnings.push(format!(
                    "nonblocking diagnostic: calibration peptide counts are unavailable: {error}"
                ));
                BTreeSet::new()
            }
        };
        if !optimization_only {
            let target_peptides = match accepted_target_peptides(
                &expert.target_only_results,
                &ValidationMode::DecoyFree,
                manifest.validation.fdr_threshold,
                manifest.validation.null_window_validation_scope
                    == NullWindowValidationScope::Level4,
            ) {
                Ok(peptides) => peptides,
                Err(error) => {
                    warnings.push(format!(
                        "nonblocking diagnostic: target-only peptide counts are unavailable: {error}"
                    ));
                    BTreeSet::new()
                }
            };
            if !peptides.is_empty() {
                let change =
                    (target_peptides.len() as f64 - peptides.len() as f64) / peptides.len() as f64;
                if change < -manifest.validation.maximum_transfer_fraction_loss {
                    warnings.push(format!(
                        "nonblocking diagnostic: target-only peptide transfer loss is {:.1}%",
                        -100.0 * change
                    ));
                }
            }
        }

        let relevant_pairs = manifest
            .validation
            .parity_pairs
            .iter()
            .filter(|pair| pair.native_method == model_slug(&expert.model))
            .cloned()
            .collect::<Vec<_>>();
        if !relevant_pairs.is_empty() {
            let mut parity_summaries = Vec::new();
            let native_runs = [
                Some(ValidationRun {
                    method: model_slug(&expert.model).into(),
                    stage: "optimized".into(),
                    results: expert.optimized_results.clone(),
                    mode: ValidationMode::DecoyFree,
                    expected_search_space: Some("+Ent".into()),
                    calibration_stage: None,
                    target_only_calibration_policy: None,
                    release_candidate: true,
                }),
                expert
                    .ms2rescore_results
                    .as_ref()
                    .map(|results| ValidationRun {
                        method: model_slug(&expert.model).into(),
                        stage: "ms2rescore".into(),
                        results: results.clone(),
                        mode: ValidationMode::DecoyFree,
                        expected_search_space: Some("+Ent".into()),
                        calibration_stage: None,
                        target_only_calibration_policy: None,
                        release_candidate: true,
                    }),
                (!optimization_only).then(|| ValidationRun {
                    method: model_slug(&expert.model).into(),
                    stage: expert.target_only_calibration_policy.stage_name().into(),
                    results: expert.target_only_results.clone(),
                    mode: ValidationMode::DecoyFree,
                    expected_search_space: Some("No Ent".into()),
                    calibration_stage: Some(expert.calibration_stage.clone()),
                    target_only_calibration_policy: Some(expert.target_only_calibration_policy),
                    release_candidate: true,
                }),
            ];
            for parity_run in native_runs.into_iter().flatten().chain(
                manifest
                    .validation
                    .additional_runs
                    .iter()
                    .filter(|candidate| {
                        relevant_pairs
                            .iter()
                            .any(|pair| pair.baseline_method == candidate.method)
                    })
                    .cloned(),
            ) {
                match summarize_run(
                    &parity_run,
                    &manifest.validation.effective_ratios,
                    manifest.validation.fdr_threshold,
                ) {
                    Ok(rows) => parity_summaries.extend(rows),
                    Err(error) => warnings.push(format!(
                        "nonblocking diagnostic: parity evidence is unavailable for {} / {}: {error}",
                        parity_run.method, parity_run.stage
                    )),
                }
            }
            let comparisons = parity_comparisons(
                &parity_summaries,
                &relevant_pairs,
                manifest.validation.maximum_parity_fraction_difference,
            );
            for pair in &relevant_pairs {
                let layers = if pair.layers.is_empty() {
                    vec![match manifest.validation.null_window_validation_scope {
                        NullWindowValidationScope::RawQ => "raw_q",
                        NullWindowValidationScope::Level4 => "level4",
                    }]
                } else {
                    pair.layers.iter().map(String::as_str).collect()
                };
                for layer in layers {
                    let expected_stages = if pair.stages.is_empty() {
                        comparisons
                            .iter()
                            .filter(|comparison| {
                                comparison.baseline_method == pair.baseline_method
                                    && comparison.native_method == pair.native_method
                                    && comparison.layer == layer
                            })
                            .map(|comparison| comparison.stage.as_str())
                            .collect::<Vec<_>>()
                    } else {
                        pair.stages.iter().map(String::as_str).collect()
                    };
                    if expected_stages.is_empty() {
                        warnings.push(format!(
                            "nonblocking diagnostic: declared {layer} parity evidence is missing for {}",
                            pair.baseline_method
                        ));
                    }
                    for stage in expected_stages {
                        match comparisons.iter().find(|comparison| {
                            comparison.baseline_method == pair.baseline_method
                                && comparison.native_method == pair.native_method
                                && comparison.stage == stage
                                && comparison.layer == layer
                        }) {
                            Some(comparison) if comparison.within_tolerance => {}
                            Some(_) => warnings.push(format!(
                                "nonblocking diagnostic: declared {layer} parity exceeds tolerance at {stage}"
                            )),
                            None => warnings.push(format!(
                                "nonblocking diagnostic: declared {layer} parity evidence is missing at {stage}"
                            )),
                        }
                    }
                }
            }
        }

        let fallback_used = optimized_fallback || ms2_fallback;
        let fallback_reason = fallback_used.then(|| {
            match (optimized_fallback, ms2_fallback) {
                (true, true) => {
                    "optimized and MS2Rescore fitted artifacts do not contain the requested model"
                }
                (true, false) => "optimized fitted artifact does not contain the requested model",
                (false, true) => "MS2Rescore fitted artifact does not contain the requested model",
                (false, false) => unreachable!(),
            }
            .into()
        });
        candidates.push(LockCandidate {
            expert,
            requested,
            technical_failures: failures,
            diagnostic_warnings: warnings,
            peptides,
            optimized_hash,
            ms2_hash,
            fallback_used,
            fallback_reason,
        });
    }

    candidates.sort_by_key(|candidate| model_slug(&candidate.expert.model));
    let completed_models = candidates
        .iter()
        .map(|candidate| model_slug(&candidate.expert.model))
        .collect::<BTreeSet<_>>();
    let requested_set = requested_roster.iter().cloned().collect::<BTreeSet<_>>();
    let mut technical_failures = runtime_failures
        .iter()
        .filter(|(model, _)| requested_set.contains(*model))
        .map(|(model, failures)| (model.clone(), failures.clone()))
        .collect::<BTreeMap<_, _>>();
    for requested in &requested_roster {
        if !completed_models.contains(requested.as_str()) {
            technical_failures
                .entry(requested.clone())
                .or_insert_with(|| {
                    vec!["requested expert did not produce a completed fitted artifact".into()]
                });
        }
    }
    if ensemble_uses_ms2 {
        let fit_search_fingerprints = candidates
            .iter()
            .filter(|candidate| candidate.requested && candidate.technical_failures.is_empty())
            .map(|candidate| candidate.expert.calibration_search_fingerprint.as_str())
            .collect::<BTreeSet<_>>();
        if fit_search_fingerprints.len() > 1 {
            for candidate in candidates
                .iter_mut()
                .filter(|candidate| candidate.requested)
            {
                candidate.technical_failures.push(
                    "requested experts do not share one dataset-local fit search fingerprint"
                        .into(),
                );
            }
        }
    }

    let requested_valid_peptide_sets = candidates
        .iter()
        .filter(|candidate| candidate.requested && candidate.technical_failures.is_empty())
        .map(|candidate| {
            (
                model_slug(&candidate.expert.model),
                candidate.peptides.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut locked = Vec::new();
    let mut enabled_optimized_hashes = BTreeMap::<String, String>::new();
    let mut enabled_ms2_hashes = BTreeMap::<String, String>::new();
    for candidate in candidates {
        let expert = candidate.expert;
        let model = model_slug(&expert.model).to_owned();
        let mut failures = candidate.technical_failures;
        let unique_peptides = candidate
            .peptides
            .iter()
            .filter(|peptide| {
                !requested_valid_peptide_sets
                    .iter()
                    .any(|(other, peptides)| {
                        *other != model.as_str() && peptides.contains(*peptide)
                    })
            })
            .count();
        let enabled = candidate.requested && failures.is_empty();
        if enabled {
            if let Some(previous) =
                enabled_optimized_hashes.insert(candidate.optimized_hash.clone(), model.clone())
            {
                anyhow::bail!("duplicate optimized artifact vote for {previous} and {model}");
            }
            if let Some(hash) = candidate.ms2_hash.as_ref() {
                if let Some(previous) = enabled_ms2_hashes.insert(hash.clone(), model.clone()) {
                    anyhow::bail!("duplicate MS2Rescore artifact vote for {previous} and {model}");
                }
            }
        }
        if candidate.requested && !failures.is_empty() {
            failures.sort();
            failures.dedup();
            technical_failures.insert(model.clone(), failures.clone());
        }
        let participation_decision = if enabled {
            "included_technical_validation_passed"
        } else if candidate.requested {
            "excluded_technical_failure"
        } else {
            "excluded_by_json"
        };
        locked.push(EnsembleExpertLock {
            model: expert.model.clone(),
            window: expert.window.clone(),
            optimized_fitted_artifacts: expert.optimized_artifacts.clone(),
            optimized_fitted_artifacts_sha256: candidate.optimized_hash,
            ms2rescore_fitted_artifacts: expert.ms2rescore_artifacts.clone(),
            ms2rescore_fitted_artifacts_sha256: candidate.ms2_hash,
            calibration_stage: expert.calibration_stage.clone(),
            calibration_results: expert.calibration_results.clone(),
            target_only_results: if optimization_only {
                PathBuf::new()
            } else {
                expert.target_only_results.clone()
            },
            target_only_calibration_policy: expert.target_only_calibration_policy,
            enabled,
            target_peptides: candidate.peptides.len(),
            incremental_target_peptides: unique_peptides,
            gate_reasons: if candidate.requested {
                failures
            } else {
                vec![format!(
                    "explicit JSON exclusion: {}",
                    explicit_exclusions
                        .get(&model)
                        .map(String::as_str)
                        .unwrap_or("unspecified")
                )]
            },
            gate_warnings: candidate.diagnostic_warnings,
            fit_search_fingerprint: expert.calibration_search_fingerprint.clone(),
            candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
            interaction_baseline: configuration_for_model(manifest, &expert.model)
                .is_none_or(|configuration| configuration.ensemble_interaction_baseline),
            participation_decision: participation_decision.into(),
            fallback_used: candidate.fallback_used,
            fallback_reason: candidate.fallback_reason,
            target_only_policy_capability: (!optimization_only).then(|| {
                target_only_policy_capability(&expert.model, expert.target_only_calibration_policy)
            }),
            fitted_external_profile_identity_sha256: expert
                .fitted_external_profile_identity_sha256
                .clone(),
            fitted_external_profile_calibration: expert.fitted_external_profile_calibration.clone(),
            annotation_cache_fingerprint: expert.annotation_cache_fingerprint.clone(),
            annotation_cache_manifest_sha256: expert.annotation_cache_manifest_sha256.clone(),
            annotation_cache_payload_sha256: expert.annotation_cache_payload_sha256.clone(),
        });
    }
    let mut actual_roster = locked
        .iter()
        .filter(|expert| expert.enabled)
        .map(|expert| model_slug(&expert.model).to_owned())
        .collect::<Vec<_>>();
    actual_roster.sort();
    let enabled = actual_roster.len();
    let evaluable = enabled >= manifest.validation.minimum_ensemble_experts;
    let not_evaluable_reasons = (!evaluable)
        .then(|| {
            vec![format!(
                "only {enabled} technically valid requested experts are available; {} structurally required",
                manifest.validation.minimum_ensemble_experts
            )]
        })
        .unwrap_or_default();
    let shared_external_profile_contract_sha256 = shared_external_profile_calibration
        .as_ref()
        .filter(|_| enabled > 0)
        .map(|calibration| shared_ensemble_profile_contract_identity(dataset, calibration, &locked))
        .transpose()?;
    stamp_ensemble_lock_analysis_fingerprint(EnsembleLock {
        schema_version: 8,
        post_selection_in_scope: !optimization_only,
        source_manifest_hash: manifest_hash.into(),
        dataset_fingerprint: dataset.fingerprint.clone(),
        experts: locked,
        requested_roster,
        actual_roster,
        explicit_exclusions,
        technical_failures,
        roster_contract: default_ensemble_roster_contract(),
        minimum_required_experts: manifest.validation.minimum_ensemble_experts,
        evaluable,
        not_evaluable_reasons,
        external_profile_contract: default_ensemble_external_profile_contract(),
        shared_external_profile_contract_sha256,
        shared_external_profile_calibration,
        source_configuration_sha256: dataset.search_config_sha256.clone(),
        analysis_fingerprint: String::new(),
        raw_q_interaction_warning_threshold: default_raw_q_interaction_warning_threshold(),
        ensemble_p_combiner,
        ensemble_pep_combiner,
    })
}

fn configuration_for_model<'a>(
    manifest: &'a WorkflowManifest,
    model: &ModelFit,
) -> Option<&'a ModelWorkflow> {
    manifest
        .models
        .iter()
        .find(|configuration| configuration.model == *model)
}

fn fitted_artifact_provenance(
    dataset: &DatasetIdentity,
    stage: &str,
    model: &ModelFit,
    fit_search_fingerprint: &str,
) -> FittedArtifactProvenance {
    FittedArtifactProvenance {
        schema_version: 2,
        dataset_id: dataset.dataset_id.clone(),
        dataset_fingerprint: dataset.fingerprint.clone(),
        search_config_sha256: dataset.search_config_sha256.clone(),
        fit_search_fingerprint: fit_search_fingerprint.into(),
        candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
        fit_stage: stage.into(),
        model: model_slug(model).into(),
        external_profile_calibration: None,
    }
}

fn validate_artifact_reuse(
    artifacts: &DfRunArtifacts,
    dataset: &DatasetIdentity,
    policy: &ArtifactReusePolicy,
    expected_model: &ModelFit,
    expected_fit_search_fingerprint: Option<&str>,
) -> Result<()> {
    let Some(provenance) = artifacts.provenance.as_ref() else {
        return match policy {
            ArtifactReusePolicy::DatasetLocalOnly => anyhow::bail!(
                "fitted artifact has no dataset provenance; dataset-local reuse fails closed"
            ),
            ArtifactReusePolicy::CrossDatasetDiagnostic => {
                log::warn!(
                    "cross-dataset diagnostic is using a legacy artifact without dataset provenance"
                );
                Ok(())
            }
        };
    };
    let dataset_matches = provenance.dataset_fingerprint == dataset.fingerprint;
    let config_matches = provenance.search_config_sha256 == dataset.search_config_sha256;
    let model_matches = provenance.model == model_slug(expected_model);
    if expected_model == &ModelFit::Nokoi {
        let artifact = artifacts
            .nokoi
            .as_ref()
            .context("Nokoi fitted artifact is missing")?;
        artifact.validate_portable().map_err(|error| {
            anyhow::anyhow!("Nokoi portable artifact validation failed: {error}")
        })?;
        anyhow::ensure!(
            artifact.identity.dataset_id == provenance.dataset_id
                && artifact.identity.dataset_fingerprint == provenance.dataset_fingerprint
                && artifact.identity.fit_search_fingerprint == provenance.fit_search_fingerprint
                && artifact.identity.candidate_id_schema == provenance.candidate_id_schema,
            "Nokoi internal identity disagrees with fitted-artifact provenance"
        );
    }
    match policy {
        ArtifactReusePolicy::DatasetLocalOnly => {
            anyhow::ensure!(
                provenance.schema_version == 2,
                "fitted artifact provenance schema {} lacks Phase 5 search-assumption safeguards",
                provenance.schema_version
            );
            anyhow::ensure!(
                dataset_matches,
                "artifact was fitted on dataset '{}' ({}) but current dataset is '{}' ({})",
                provenance.dataset_id,
                provenance.dataset_fingerprint,
                dataset.dataset_id,
                dataset.fingerprint
            );
            anyhow::ensure!(
                config_matches,
                "artifact search configuration does not match the current workflow"
            );
            anyhow::ensure!(
                model_matches,
                "artifact provenance identifies model '{}' but {:?} was requested",
                provenance.model,
                expected_model
            );
            anyhow::ensure!(
                provenance.candidate_id_schema == CANDIDATE_ID_SCHEMA,
                "artifact candidate identity schema '{}' does not match '{}'",
                provenance.candidate_id_schema,
                CANDIDATE_ID_SCHEMA
            );
            anyhow::ensure!(
                !provenance.fit_search_fingerprint.is_empty(),
                "artifact fit search fingerprint is missing"
            );
            if let Some(expected) = expected_fit_search_fingerprint {
                anyhow::ensure!(
                    provenance.fit_search_fingerprint == expected,
                    "artifact fit search fingerprint does not match the selected calibration candidate pool"
                );
            }
        }
        ArtifactReusePolicy::CrossDatasetDiagnostic => {
            if provenance.schema_version != 2
                || provenance.candidate_id_schema != CANDIDATE_ID_SCHEMA
                || provenance.fit_search_fingerprint.is_empty()
            {
                log::warn!(
                    "cross-dataset diagnostic artifact lacks complete Phase 5 search provenance"
                );
            }
            if !dataset_matches || !config_matches || !model_matches {
                log::warn!(
                    "explicit diagnostic-only artifact reuse: fit_dataset={} current_dataset={} fit_config={} current_config={} fit_model={} current_model={}",
                    provenance.dataset_id,
                    dataset.dataset_id,
                    provenance.search_config_sha256,
                    dataset.search_config_sha256,
                    provenance.model,
                    model_slug(expected_model)
                );
            }
        }
    }
    Ok(())
}

fn stamp_fitted_artifacts(
    output_directory: &Path,
    dataset: &DatasetIdentity,
    stage: &str,
    model: &ModelFit,
    inherited: Option<FittedArtifactProvenance>,
    fit_search_fingerprint: &str,
    fit_analysis_fingerprint: &str,
) -> Result<Option<FittedArtifactProvenance>> {
    let path = output_directory.join("fitted_model_artifacts.json");
    if !path.is_file() {
        return Ok(None);
    }
    let mut artifacts: DfRunArtifacts = serde_json::from_slice(&std::fs::read(&path)?)
        .with_context(|| format!("invalid fitted artifacts {}", path.display()))?;
    let inherited_fit = inherited.is_some();
    let mut provenance = inherited.unwrap_or_else(|| {
        fitted_artifact_provenance(dataset, stage, model, fit_search_fingerprint)
    });
    provenance.external_profile_calibration = artifacts
        .external_ms2rescore
        .as_ref()
        .map(|profiles| profiles.calibration.clone());
    if let Some(artifact) = artifacts.nokoi.as_mut() {
        if inherited_fit {
            artifact.validate_portable().map_err(|error| {
                anyhow::anyhow!("inherited Nokoi portable artifact is invalid: {error}")
            })?;
            anyhow::ensure!(
                artifact.identity.dataset_id == provenance.dataset_id
                    && artifact.identity.dataset_fingerprint == provenance.dataset_fingerprint
                    && artifact.identity.fit_search_fingerprint
                        == provenance.fit_search_fingerprint
                    && artifact.identity.candidate_id_schema == provenance.candidate_id_schema,
                "inherited Nokoi internal fit identity disagrees with inherited provenance"
            );
        } else {
            artifact
                .stamp_workflow_identity(
                    &dataset.dataset_id,
                    &dataset.fingerprint,
                    fit_search_fingerprint,
                    fit_analysis_fingerprint,
                )
                .map_err(|error| {
                    anyhow::anyhow!("could not stamp portable Nokoi artifact identity: {error}")
                })?;
        }
    }
    artifacts.provenance = Some(provenance.clone());
    write_json_atomic(&path, &artifacts)?;
    Ok(Some(provenance))
}

fn fitted_model_artifact_schema(output_directory: &Path, model: &ModelFit) -> Result<Option<u32>> {
    let path = output_directory.join("fitted_model_artifacts.json");
    if !path.is_file() {
        return Ok(None);
    }
    let artifacts: DfRunArtifacts = serde_json::from_slice(&std::fs::read(&path)?)
        .with_context(|| format!("invalid fitted artifacts {}", path.display()))?;
    Ok(match model {
        ModelFit::Moments => artifacts.moments.map(|artifact| artifact.schema_version),
        ModelFit::Mle => artifacts.mle.map(|artifact| artifact.schema_version),
        ModelFit::LowerOrder => artifacts
            .lower_order
            .map(|artifact| artifact.schema_version),
        ModelFit::Msfdr => artifacts
            .msfdr_seeded_metadata
            .map(|metadata| metadata.schema_version),
        ModelFit::Msfdr1Smix => artifacts
            .msfdr_1smix_metadata
            .map(|metadata| metadata.schema_version),
        ModelFit::Msfdr2Smix => artifacts
            .msfdr_2smix_metadata
            .map(|metadata| metadata.schema_version),
        ModelFit::Nokoi => artifacts.nokoi.map(|artifact| artifact.schema_version),
        ModelFit::Ensemble => None,
    })
}

fn hash_stage(
    manifest: &WorkflowManifest,
    dataset: &DatasetIdentity,
    model: &ModelWorkflow,
    stage: &str,
    fasta: &Path,
    external: bool,
    frozen_artifact: Option<&Path>,
    ensemble_lock: Option<&EnsembleLock>,
    target_only: Option<&TargetOnlyStageContext>,
    parameter_overrides: Option<&BTreeMap<String, ParameterValue>>,
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-workflow-stage-v6-external-profile-and-required-pool\0");
    hasher.update(serde_json::to_vec(manifest)?);
    hasher.update(serde_json::to_vec(model)?);
    hasher.update(dataset.fingerprint.as_bytes());
    hasher.update(dataset.search_config_sha256.as_bytes());
    hasher.update(stage.as_bytes());
    hasher.update([external as u8]);
    hasher.update(sha256_file(&manifest.search_config)?.as_bytes());
    if external {
        let input = Input::load(manifest.search_config.to_string_lossy().as_ref())?;
        let settings =
            crate::input::ExternalFeatureGenerationSettings::from(input.external_features);
        let annotation_root = manifest.resolved_annotation_cache_root(target_only.is_some());
        let settings_sha256 = if manifest.require_existing_annotation_cache {
            raw_generator_settings_sha256_with_existing_probe_root(&settings, &annotation_root)?
        } else {
            raw_generator_settings_sha256_with_probe_root(&settings, &annotation_root)?
        };
        hasher.update(settings_sha256.as_bytes());
    }
    if fasta.is_file() {
        hasher.update(sha256_file(fasta)?.as_bytes());
    } else {
        hasher.update(b"missing:");
        hasher.update(fasta.display().to_string().as_bytes());
    }
    if let Some(path) = frozen_artifact {
        anyhow::ensure!(path.is_file(), "frozen model artifact does not exist");
        hasher.update(sha256_file(path)?.as_bytes());
    }
    if let Some(lock) = ensemble_lock {
        hasher.update(serde_json::to_vec(lock)?);
    }
    if let Some(target_only) = target_only {
        hasher.update(serde_json::to_vec(target_only)?);
    }
    if let Some(parameter_overrides) = parameter_overrides {
        hasher.update(b"parameter-optimizer-overrides-v1\0");
        hasher.update(serde_json::to_vec(parameter_overrides)?);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn stage_checkpoint_identity_matches(
    record: &StageRecord,
    input_hash: &str,
    stage: &str,
    model: &ModelFit,
    dataset: &DatasetIdentity,
    results: &Path,
    config_snapshot: &Path,
    target_only_policy: Option<TargetOnlyCalibrationPolicy>,
) -> bool {
    record.status == "complete"
        && record.input_hash == input_hash
        && record.stage == stage
        && record.model == model_slug(model)
        && record.dataset_id == dataset.dataset_id
        && record.dataset_fingerprint == dataset.fingerprint
        && record.results == results
        && record.config_snapshot == config_snapshot
        && results.is_file()
        && config_snapshot.is_file()
        && record.target_only_calibration_policy == target_only_policy
}

fn stage_output_hashes_match(record: &StageRecord) -> Result<bool> {
    if record.results_sha256.is_empty()
        || record.config_snapshot_sha256.is_empty()
        || !record.results.is_file()
        || !record.config_snapshot.is_file()
    {
        return Ok(false);
    }
    Ok(sha256_file(&record.results)? == record.results_sha256
        && sha256_file(&record.config_snapshot)? == record.config_snapshot_sha256)
}

fn run_search_stage(
    manifest: &WorkflowManifest,
    dataset: &DatasetIdentity,
    model: &ModelWorkflow,
    stage: &str,
    fasta: &Path,
    output_directory: &Path,
    external: bool,
    annotate_matches: bool,
    parallel: usize,
    plan_only: bool,
    frozen_model_artifacts: Option<&Path>,
    ensemble_lock: Option<&EnsembleLock>,
    target_only: Option<&TargetOnlyStageContext>,
    parameter_overrides: Option<&BTreeMap<String, ParameterValue>>,
    runtime: &mut WorkflowRuntime,
) -> Result<StageRecord> {
    if let Some(context) = target_only {
        anyhow::ensure!(
            context.policy != TargetOnlyCalibrationPolicy::CompareBoth,
            "compare_both must be expanded into concrete target-only stages"
        );
        anyhow::ensure!(
            stage == context.policy.stage_name(),
            "target-only stage name does not match its calibration policy"
        );
        match context.policy {
            TargetOnlyCalibrationPolicy::RefitWithLockedWindow => anyhow::ensure!(
                frozen_model_artifacts.is_none(),
                "refit_with_locked_window must not receive fitted nuisance state"
            ),
            TargetOnlyCalibrationPolicy::ReuseDatasetArtifact => {
                if !plan_only && model.model != ModelFit::Ensemble {
                    anyhow::ensure!(
                        frozen_model_artifacts.is_some(),
                        "reuse_dataset_artifact requires a fitted dataset-local artifact"
                    );
                }
            }
            TargetOnlyCalibrationPolicy::CompareBoth => unreachable!(),
        }
    }
    let input_hash = hash_stage(
        manifest,
        dataset,
        model,
        stage,
        fasta,
        external,
        frozen_model_artifacts,
        ensemble_lock,
        target_only,
        parameter_overrides,
    )?;
    let results = output_directory.join("results.sage.tsv");
    let config_snapshot = output_directory.join("workflow.search.resolved.json");
    let checkpoint = output_directory.join("workflow.stage.json");

    if manifest.resume && results.is_file() && checkpoint.is_file() {
        let mut old: StageRecord = serde_json::from_slice(&std::fs::read(&checkpoint)?)?;
        anyhow::ensure!(
            matches!(old.schema_version, 1 | 2 | 3 | 4),
            "unsupported workflow stage checkpoint schema {}",
            old.schema_version
        );
        if old.input_hash == input_hash {
            let identity_ready = stage_checkpoint_identity_matches(
                &old,
                &input_hash,
                stage,
                &model.model,
                dataset,
                &results,
                &config_snapshot,
                target_only.map(|context| context.policy),
            );
            if !identity_ready {
                log::warn!(
                    "workflow: completed stage identity changed for {} / {}; rebuilding",
                    model_slug(&model.model),
                    stage
                );
            }
            let mut upgraded_legacy_checkpoint = false;
            if identity_ready && old.schema_version == 1 {
                // Phase 1-8 checkpoints predate durable output hashes. Migrate
                // an otherwise fully matching checkpoint once; all later
                // resumes verify the recorded result/configuration bytes.
                if old.results_sha256.is_empty() && old.config_snapshot_sha256.is_empty() {
                    old.results_sha256 = sha256_file(&results)?;
                    old.config_snapshot_sha256 = sha256_file(&config_snapshot)?;
                }
                if !old.results_sha256.is_empty() && !old.config_snapshot_sha256.is_empty() {
                    old.schema_version = 2;
                    upgraded_legacy_checkpoint = true;
                }
            }
            if identity_ready && old.schema_version < 3 {
                old.schema_version = 3;
                old.evaluable = true;
                old.not_evaluable_reason = None;
                old.target_only_policy_capability = target_only
                    .map(|context| target_only_policy_capability(&model.model, context.policy));
                old.nuisance_state_provenance = target_only.map(|context| match context.policy {
                    TargetOnlyCalibrationPolicy::RefitWithLockedWindow => {
                        "refitted_in_target_only_candidate_space".into()
                    }
                    TargetOnlyCalibrationPolicy::ReuseDatasetArtifact => {
                        "reused_complete_dataset_artifact".into()
                    }
                    TargetOnlyCalibrationPolicy::CompareBoth => unreachable!(),
                });
                old.target_only_window_tuning = target_only.map(|_| false);
                old.complete_dataset_artifact_reused = target_only.map(|context| {
                    context.policy == TargetOnlyCalibrationPolicy::ReuseDatasetArtifact
                });
                old.fallback_used = false;
                old.fallback_reason = None;
                old.model_artifact_schema =
                    fitted_model_artifact_schema(output_directory, &model.model)?;
                upgraded_legacy_checkpoint = true;
            }
            let outputs_ready = identity_ready && stage_output_hashes_match(&old)?;
            if identity_ready && !outputs_ready {
                log::warn!(
                    "workflow: completed stage output hash changed for {} / {}; rebuilding",
                    model_slug(&model.model),
                    stage
                );
            }
            let pool_ready = if outputs_ready {
                if let Some(usage) = old.candidate_pool.as_ref() {
                    verify_candidate_pool_usage(usage)?;
                    true
                } else {
                    log::info!(
                        "workflow: completed legacy stage {} / {} has no candidate-pool provenance; rebuilding",
                        model_slug(&model.model),
                        stage
                    );
                    false
                }
            } else {
                false
            };
            let cache_ready = if external {
                if pool_ready {
                    if let Some(usage) = old.ms2rescore_annotation_cache.as_ref() {
                        verify_annotation_cache_usage(usage)?;
                        true
                    } else {
                        log::info!(
                        "workflow: completed legacy stage {} / {} has no Phase 4 annotation cache; rebuilding",
                        model_slug(&model.model),
                        stage
                    );
                        false
                    }
                } else {
                    false
                }
            } else {
                pool_ready
            };
            if cache_ready {
                if upgraded_legacy_checkpoint {
                    write_json_atomic(&checkpoint, &old)?;
                }
                log::info!(
                    "workflow: resuming completed stage {} / {}",
                    model_slug(&model.model),
                    stage
                );
                return Ok(old);
            }
        }
    }

    let mut input = Input::load(manifest.search_config.to_string_lossy().as_ref())?;
    input.database.fasta = Some(fasta.display().to_string());
    input.mzml_paths = Some(manifest.spectra.clone());
    input.output_directory = Some(output_directory.display().to_string());
    input.annotate_matches = Some(annotate_matches);
    let fdr = input.fdr.get_or_insert_with(FdrOptions::default);
    fdr.mode = Some(FdrMode::DecoyFree);
    fdr.model_fit = Some(model.model.clone());
    fdr.nokoi_application_dataset_fingerprint = Some(dataset.fingerprint.clone());
    if let Some(parameter_overrides) = parameter_overrides {
        apply_fdr_overrides(fdr, parameter_overrides)?;
    }
    let mut optimizer_ensemble_lock = ensemble_lock.cloned();
    if model.model == ModelFit::Ensemble && parameter_overrides.is_some() {
        if let Some(lock) = optimizer_ensemble_lock.as_mut() {
            let settings = FdrSettings::from(fdr.clone());
            lock.ensemble_p_combiner = settings.ensemble_p_combiner;
            lock.ensemble_pep_combiner = settings.ensemble_pep_combiner;
            lock.analysis_fingerprint = ensemble_lock_analysis_fingerprint(lock)?;
        }
    }
    if let Some(lock) = optimizer_ensemble_lock.as_ref() {
        anyhow::ensure!(
            model.model == ModelFit::Ensemble,
            "Ensemble lock on non-Ensemble stage"
        );
        apply_ensemble_lock(
            fdr,
            lock,
            external,
            dataset,
            &manifest.artifact_reuse_policy,
            target_only.is_none_or(|context| {
                context.policy == TargetOnlyCalibrationPolicy::ReuseDatasetArtifact
            }),
            target_only.map(|context| context.policy),
        )?;
    }
    apply_window(fdr, &model.model, &model.window);
    if matches!(stage, "optimized" | "parameter_optimizer_trial")
        && (!model.candidate_windows.is_empty() || model.window_optimizer.is_some())
    {
        let (strategy, bounds, adaptive) = model
            .window_optimizer
            .as_ref()
            .map(|search| {
                (
                    search.strategy,
                    Some(search.bounds()),
                    search.adaptive.clone(),
                )
            })
            .unwrap_or((
                NullWindowSearchStrategy::Explicit,
                None,
                AdaptiveNullWindowSearchOptions::default(),
            ));
        fdr.null_window_optimizer = Some(NullWindowOptimizerOptions {
            candidates: model
                .candidate_windows
                .iter()
                .map(|window| NullWindowCandidate {
                    min_rank: window.min_rank,
                    max_rank: window.max_rank,
                })
                .collect(),
            strategy,
            bounds,
            adaptive,
            validation_scope: manifest.validation.null_window_validation_scope,
            fdr_threshold: manifest.validation.fdr_threshold,
            psm_entrapment_ratio: manifest.validation.effective_ratios.psm,
            peptide_entrapment_ratio: manifest.validation.effective_ratios.peptide,
            protein_entrapment_ratio: manifest.validation.effective_ratios.protein,
            maximum_entrapment_fdp: manifest.validation.fdr_threshold,
            minimum_entrapment_count_for_stable_estimate: 3,
            verbose_diagnostics: false,
        });
    }
    let inherited_artifact_provenance = if let Some(path) = frozen_model_artifacts {
        let artifacts: DfRunArtifacts = serde_json::from_slice(&std::fs::read(path)?)
            .with_context(|| format!("invalid fitted artifacts {}", path.display()))?;
        validate_artifact_reuse(
            &artifacts,
            dataset,
            &manifest.artifact_reuse_policy,
            &model.model,
            target_only.and_then(|context| {
                context
                    .window_provenance
                    .source_search_fingerprint
                    .as_deref()
            }),
        )?;
        let provenance = artifacts.provenance.clone().or_else(|| {
            (manifest.artifact_reuse_policy == ArtifactReusePolicy::CrossDatasetDiagnostic).then(
                || FittedArtifactProvenance {
                    schema_version: 1,
                    dataset_id: "unknown-legacy-cross-dataset-source".into(),
                    dataset_fingerprint: format!(
                        "unprovenanced-artifact:{}",
                        sha256_file(path).unwrap_or_else(|_| "unavailable".into())
                    ),
                    search_config_sha256: "unknown".into(),
                    fit_search_fingerprint: String::new(),
                    candidate_id_schema: String::new(),
                    fit_stage: "unknown".into(),
                    model: model_slug(&model.model).into(),
                    external_profile_calibration: None,
                },
            )
        });
        if external {
            anyhow::ensure!(
                artifacts.external_ms2rescore.is_some(),
                "locked external-feature stage requires portable MS2Rescore empirical profiles in {}",
                path.display()
            );
        }
        apply_fitted_artifacts(
            fdr,
            &model.model,
            artifacts,
            true,
            target_only.map(|context| context.policy),
        )?;
        provenance
    } else {
        None
    };
    if let Some(external_options) = input.external_features.as_mut() {
        external_options.enabled = Some(external);
    } else if external {
        anyhow::bail!(
            "{} requests MS2Rescore, but search_config has no external_features configuration",
            model_slug(&model.model)
        );
    }
    let parameters = input.build()?;
    write_json_atomic(&config_snapshot, &parameters)?;

    let mut record = StageRecord {
        schema_version: 4,
        stage: stage.into(),
        model: model_slug(&model.model).into(),
        input_hash,
        status: if plan_only { "planned" } else { "running" }.into(),
        results,
        config_snapshot: config_snapshot.clone(),
        results_sha256: String::new(),
        config_snapshot_sha256: String::new(),
        external_features_enabled: external,
        calibration_mode: match target_only.map(|context| context.policy) {
            Some(TargetOnlyCalibrationPolicy::RefitWithLockedWindow) => {
                "refit_with_locked_window".into()
            }
            Some(TargetOnlyCalibrationPolicy::ReuseDatasetArtifact) => {
                "reuse_dataset_artifact".into()
            }
            Some(TargetOnlyCalibrationPolicy::CompareBoth) => unreachable!(),
            None if frozen_model_artifacts.is_some() => "reuse_dataset_artifact".into(),
            None => "fit_current_search_space".into(),
        },
        dataset_id: dataset.dataset_id.clone(),
        dataset_fingerprint: dataset.fingerprint.clone(),
        artifact_fit_dataset_fingerprint: inherited_artifact_provenance
            .as_ref()
            .map(|provenance| provenance.dataset_fingerprint.clone()),
        candidate_pool: None,
        require_existing_candidate_pool: manifest.require_existing_candidate_pool,
        require_existing_annotation_cache: manifest.require_existing_annotation_cache,
        ms2rescore_annotation_cache: None,
        target_only_calibration_policy: target_only.map(|context| context.policy),
        release_candidate: target_only.is_none_or(|context| context.release_candidate),
        window_provenance: target_only.map(|context| context.window_provenance.clone()),
        external_profile_calibration: None,
        ensemble_shared_profile_contract_sha256: ensemble_lock
            .and_then(|lock| lock.shared_external_profile_contract_sha256.clone()),
        fitted_external_profile_identity_sha256: None,
        evaluable: true,
        not_evaluable_reason: None,
        target_only_policy_capability: target_only
            .map(|context| target_only_policy_capability(&model.model, context.policy)),
        nuisance_state_provenance: target_only.map(|context| match context.policy {
            TargetOnlyCalibrationPolicy::RefitWithLockedWindow => {
                "refitted_in_target_only_candidate_space".into()
            }
            TargetOnlyCalibrationPolicy::ReuseDatasetArtifact => {
                "reused_complete_dataset_artifact".into()
            }
            TargetOnlyCalibrationPolicy::CompareBoth => unreachable!(),
        }),
        target_only_window_tuning: target_only.map(|_| false),
        complete_dataset_artifact_reused: target_only
            .map(|context| context.policy == TargetOnlyCalibrationPolicy::ReuseDatasetArtifact),
        fallback_used: false,
        fallback_reason: None,
        model_artifact_schema: None,
        ensemble_interaction_calibration: None,
        parameter_overrides: parameter_overrides.cloned().unwrap_or_default(),
    };
    std::fs::create_dir_all(output_directory)?;
    write_json_atomic(&checkpoint, &record)?;
    if plan_only {
        return Ok(record);
    }

    let unresolved_search_fingerprint = search_fingerprint(&parameters)?.digest;
    let runner = if let Some(shared) = runtime.databases.get(&unresolved_search_fingerprint) {
        let mut shared_parameters = parameters;
        shared_parameters.database = shared.resolved_database_parameters.clone();
        log::info!(
            "workflow: reusing database index for search fingerprint {}",
            unresolved_search_fingerprint
        );
        Runner::with_shared_database(shared_parameters, Arc::clone(&shared.database))?
    } else {
        let runner = Runner::new(parameters, parallel)?;
        runtime.databases.insert(
            unresolved_search_fingerprint.clone(),
            SharedDatabase {
                database: runner.shared_database(),
                resolved_database_parameters: runner.parameters.database.clone(),
            },
        );
        runner
    };
    // Runner::new may resolve an automatic prefilter chunk size. Record the
    // actual search parameters that produced or consumed the pool.
    write_json_atomic(&config_snapshot, &runner.parameters)?;

    let candidate_pool = (matches!(
        stage,
        "optimized" | "ms2rescore" | "parameter_optimizer_trial"
    ) || model.model == ModelFit::Ensemble
        || target_only.is_some())
    .then(|| {
        let requested_by_model = model
            .candidate_windows
            .iter()
            .map(|window| window.max_rank as usize)
            .chain(
                model
                    .window_optimizer
                    .iter()
                    .map(|search| search.max_rank_range[1] as usize),
            )
            .chain(model.window.iter().map(|window| window.max_rank as usize))
            .max()
            .unwrap_or(1);
        // The optimized stage creates the immutable +entrapment pool first.
        // Retain enough depth for its later MS2Rescore stage now so enabling
        // annotations cannot force a second native spectrum search.
        let will_run_external_stage =
            external || !matches!(model.ms2rescore, Ms2RescorePolicy::Never);
        let requested_by_external = will_run_external_stage.then_some(
            runner
                .parameters
                .external_features
                .max_rank
                .map(|rank| rank as usize)
                .unwrap_or(runner.parameters.report_psms),
        );
        CandidatePoolRequest {
            root: manifest.resolved_candidate_pool_root(),
            required_rank_depth: requested_by_external
                .map(|rank| rank.max(requested_by_model))
                .unwrap_or(requested_by_model),
            allow_reuse: manifest.require_existing_candidate_pool
                || target_only
                    .map(|context| context.allow_candidate_pool_reuse)
                    .unwrap_or(true),
            require_existing: manifest.require_existing_candidate_pool,
        }
    });
    let annotation_cache = external.then(|| ExternalAnnotationCacheRequest {
        root: manifest.resolved_annotation_cache_root(target_only.is_some()),
        require_existing: manifest.require_existing_annotation_cache,
        search_space: if target_only.is_some() {
            "target_only".into()
        } else {
            "+entrapment".into()
        },
        stage: format!("{}:{stage}", model_slug(&model.model)),
        analysis_fingerprint: record.input_hash.clone(),
        migration_only: manifest.migrate_schema_v2_annotation_cache_only,
    });
    let (_, candidate_usage, annotation_usage) =
        runner.run_with_workflow_caches(parallel, false, candidate_pool, annotation_cache)?;
    record.candidate_pool = candidate_usage;
    record.ms2rescore_annotation_cache = annotation_usage;
    if external {
        let usage = record
            .ms2rescore_annotation_cache
            .as_ref()
            .context("MS2Rescore stage completed without an annotation cache record")?;
        anyhow::ensure!(
            usage.annotation_count > 0 && usage.joined_annotation_count > 0,
            "MS2Rescore stage produced no usable external annotations"
        );
    }
    anyhow::ensure!(
        record.results.is_file(),
        "stage completed without results.sage.tsv"
    );
    let stamped = stamp_fitted_artifacts(
        output_directory,
        dataset,
        stage,
        &model.model,
        inherited_artifact_provenance,
        record
            .candidate_pool
            .as_ref()
            .context("fitted workflow stage has no candidate-pool provenance")?
            .search_fingerprint
            .as_str(),
        record
            .candidate_pool
            .as_ref()
            .context("fitted workflow stage has no candidate-pool provenance")?
            .analysis_fingerprint
            .as_str(),
    )?;
    record.artifact_fit_dataset_fingerprint = stamped
        .as_ref()
        .map(|provenance| provenance.dataset_fingerprint.clone());
    record.external_profile_calibration = stamped
        .as_ref()
        .and_then(|provenance| provenance.external_profile_calibration.clone());
    if external {
        let artifact_path = output_directory.join("fitted_model_artifacts.json");
        let artifacts: DfRunArtifacts = serde_json::from_slice(&std::fs::read(&artifact_path)?)
            .with_context(|| format!("invalid fitted artifacts {}", artifact_path.display()))?;
        record.fitted_external_profile_identity_sha256 =
            fitted_external_profile_identity(&artifacts)?.map(|(identity, _)| identity);
        anyhow::ensure!(
            record.fitted_external_profile_identity_sha256.is_some(),
            "external stage completed without a fitted external-profile identity"
        );
    }
    record.model_artifact_schema = fitted_model_artifact_schema(output_directory, &model.model)?;
    record.status = "complete".into();
    record.results_sha256 = sha256_file(&record.results)?;
    record.config_snapshot_sha256 = sha256_file(&record.config_snapshot)?;
    write_json_atomic(&checkpoint, &record)?;
    Ok(record)
}

fn optimizer_expert(model: &ModelFit) -> OptimizerExpert {
    match model {
        ModelFit::Moments => OptimizerExpert::Moments,
        ModelFit::Mle => OptimizerExpert::Mle,
        ModelFit::LowerOrder => OptimizerExpert::LowerOrder,
        ModelFit::Msfdr => OptimizerExpert::MsfdrSeeded,
        ModelFit::Msfdr1Smix => OptimizerExpert::Msfdr1Smix,
        ModelFit::Msfdr2Smix => OptimizerExpert::Msfdr2Smix,
        ModelFit::Nokoi => OptimizerExpert::Nokoi,
        ModelFit::Ensemble => OptimizerExpert::Ensemble,
    }
}

fn optimizer_config_for_expert(
    config: &ParameterOptimizerConfig,
    expert: OptimizerExpert,
) -> Option<ParameterOptimizerConfig> {
    if !config.enabled {
        return None;
    }
    let blocks = config
        .blocks
        .iter()
        .filter(|block| block.enabled && block.expert == Some(expert))
        .cloned()
        .collect::<Vec<_>>();
    if blocks.is_empty() {
        return None;
    }
    let ids = blocks
        .iter()
        .map(|block| block.id.clone())
        .collect::<BTreeSet<_>>();
    let mut selected = config.clone();
    selected.blocks = blocks;
    if expert != OptimizerExpert::Ensemble {
        selected.selected_experts = vec![expert];
    }
    selected.block_order = config
        .block_order
        .iter()
        .filter(|id| ids.contains(*id))
        .cloned()
        .collect();
    Some(selected)
}

fn apply_optimizer_window(model: &mut ModelWorkflow, search: &OptimizerWindowSearch) -> Result<()> {
    model.window = None;
    model.candidate_windows.clear();
    model.window_optimizer = None;
    match search.strategy.as_str() {
        "landscape_adaptive" => {
            model.window_optimizer = Some(WindowOptimizerWorkflow {
                strategy: NullWindowSearchStrategy::LandscapeAdaptive,
                min_rank_range: search.min_rank_range,
                max_rank_range: search.max_rank_range,
                adaptive: AdaptiveNullWindowSearchOptions::default(),
            });
        }
        "explicit_grid" => {
            for min_rank in search.min_rank_range[0]..=search.min_rank_range[1] {
                for max_rank in search.max_rank_range[0]..=search.max_rank_range[1] {
                    if max_rank >= min_rank {
                        model
                            .candidate_windows
                            .push(NullWindow { min_rank, max_rank });
                    }
                }
            }
            anyhow::ensure!(
                !model.candidate_windows.is_empty() && model.candidate_windows.len() <= 10_000,
                "explicit model-local null-window grid must contain 1..=10000 valid windows"
            );
        }
        strategy => anyhow::bail!("unsupported optimizer null-window strategy {strategy}"),
    }
    Ok(())
}

struct WorkflowTrialEvaluator<'a> {
    manifest: &'a WorkflowManifest,
    dataset: &'a DatasetIdentity,
    base_model: &'a ModelWorkflow,
    blocks: &'a [OptimizerBlock],
    fasta: &'a Path,
    root: &'a Path,
    parallel: usize,
    ensemble_lock: Option<&'a EnsembleLock>,
    runtime: &'a mut WorkflowRuntime,
}

struct InfrastructureSmokeEvaluator<'a> {
    root: &'a Path,
    candidate_pool_identity: &'a str,
    raw_annotation_cache_identity: &'a str,
}

impl TrialEvaluator for InfrastructureSmokeEvaluator<'_> {
    fn evaluate(&mut self, request: &TrialRequest) -> Result<TrialEvaluation> {
        anyhow::ensure!(
            !request.target_only_outcomes_allowed,
            "target-only outcomes are prohibited in optimizer smoke trials"
        );
        // Exercise the same JSON materialization contract without fitting or
        // reading biological result rows. Strict resource reuse was already
        // integrity-checked by workflow preflight.
        let mut options = FdrOptions::default();
        apply_fdr_overrides(&mut options, &request.parameters)?;
        Ok(TrialEvaluation {
            status: TrialStatus::Feasible,
            technical_reason: None,
            empirical_reason: None,
            metrics: Some(TrialMetrics {
                level4_proteins: 0,
                level4_canonical_peptides: 0,
                level4_peptidoforms: 0,
                level4_psms: 0,
                adjusted_entrapment_fdp: None,
                entrapment_count: 0,
                adjusted_entrapment_fdp_by_level: BTreeMap::new(),
                entrapment_count_by_level: BTreeMap::new(),
                model_complexity: request.parameters.len(),
            }),
            compact_diagnostics: BTreeMap::from([
                ("implementation_smoke_only".into(), serde_json::json!(true)),
                ("biological_metrics_used".into(), serde_json::json!(false)),
                ("candidate_pool_reused".into(), serde_json::json!(true)),
                (
                    "candidate_pool_identity".into(),
                    serde_json::json!(self.candidate_pool_identity),
                ),
                (
                    "raw_annotation_cache_reused".into(),
                    serde_json::json!(true),
                ),
                (
                    "raw_annotation_cache_identity".into(),
                    serde_json::json!(self.raw_annotation_cache_identity),
                ),
                ("spectrum_search_allowed".into(), serde_json::json!(false)),
                (
                    "raw_annotation_generation_allowed".into(),
                    serde_json::json!(false),
                ),
                ("target_only_outcomes_used".into(), serde_json::json!(false)),
            ]),
        })
    }

    fn materialize_winner(&mut self, record: &TrialRecord) -> Result<Option<serde_json::Value>> {
        let directory = self.root.join("winners").join(&record.request.block_id);
        let path = directory.join("resolved_parameters.json");
        write_json_atomic(&path, &record.request.parameters)?;
        Ok(Some(serde_json::json!({
            "artifact": format!("winners/{}/resolved_parameters.json", record.request.block_id),
            "sha256": sha256_file(&path)?,
            "scientific_fit": false,
            "classification": "configuration_only_implementation_smoke"
        })))
    }
}

impl WorkflowTrialEvaluator<'_> {
    fn trial_directory(&self, trial_id: &str) -> PathBuf {
        self.root.join("trials").join(trial_id)
    }

    fn model_for(&self, request: &TrialRequest) -> Result<ModelWorkflow> {
        let mut model = self.base_model.clone();
        if let Some(search) = self
            .blocks
            .iter()
            .find(|block| block.id == request.block_id)
            .and_then(|block| block.window_search.as_ref())
        {
            apply_optimizer_window(&mut model, search)?;
        }
        Ok(model)
    }
}

impl TrialEvaluator for WorkflowTrialEvaluator<'_> {
    fn evaluate(&mut self, request: &TrialRequest) -> Result<TrialEvaluation> {
        anyhow::ensure!(
            !request.target_only_outcomes_allowed,
            "target-only outcomes are prohibited in optimizer trials"
        );
        let model = self.model_for(request)?;
        let output = self.trial_directory(&request.trial_id);
        let stage = match run_search_stage(
            self.manifest,
            self.dataset,
            &model,
            "parameter_optimizer_trial",
            self.fasta,
            &output,
            request.use_external_features,
            false,
            self.parallel,
            false,
            None,
            self.ensemble_lock,
            None,
            Some(&request.parameters),
            self.runtime,
        ) {
            Ok(stage) => stage,
            Err(error) => {
                return Ok(TrialEvaluation {
                    status: TrialStatus::TechnicalFailure,
                    technical_reason: Some(format!("{error:#}")),
                    empirical_reason: None,
                    metrics: None,
                    compact_diagnostics: BTreeMap::from([
                        ("fallback_used".into(), serde_json::json!(false)),
                        ("model_substitution".into(), serde_json::json!(false)),
                    ]),
                });
            }
        };
        if stage.fallback_used {
            return Ok(TrialEvaluation {
                status: TrialStatus::TechnicalFailure,
                technical_reason: Some(
                    stage
                        .fallback_reason
                        .clone()
                        .unwrap_or_else(|| "production trial used an undocumented fallback".into()),
                ),
                empirical_reason: None,
                metrics: None,
                compact_diagnostics: BTreeMap::from([
                    ("fallback_used".into(), serde_json::json!(true)),
                    ("model_substitution".into(), serde_json::json!(false)),
                    (
                        "resolved_production_parameters".into(),
                        serde_json::to_value(&request.parameters)?,
                    ),
                    ("target_only_outcomes_used".into(), serde_json::json!(false)),
                ]),
            });
        }
        let fitted_artifacts = output.join("fitted_model_artifacts.json");
        if !fitted_artifacts.is_file() {
            return Ok(TrialEvaluation {
                status: TrialStatus::TechnicalFailure,
                technical_reason: Some(
                    "production trial completed without a fitted_model_artifacts.json payload"
                        .into(),
                ),
                empirical_reason: None,
                metrics: None,
                compact_diagnostics: BTreeMap::from([
                    ("fallback_used".into(), serde_json::json!(false)),
                    ("model_substitution".into(), serde_json::json!(false)),
                    (
                        "resolved_production_parameters".into(),
                        serde_json::to_value(&request.parameters)?,
                    ),
                    ("target_only_outcomes_used".into(), serde_json::json!(false)),
                ]),
            });
        }
        let summaries = summarize_run(
            &ValidationRun {
                method: model_slug(&model.model).into(),
                stage: "parameter_optimizer_trial".into(),
                results: stage.results.clone(),
                mode: ValidationMode::DecoyFree,
                expected_search_space: Some("+Ent".into()),
                calibration_stage: None,
                target_only_calibration_policy: None,
                release_candidate: false,
            },
            &self.manifest.validation.effective_ratios,
            self.manifest.validation.fdr_threshold,
        )?;
        let level4 = summaries
            .iter()
            .find(|row| row.layer == "level4")
            .context("optimizer trial has no Level-4 summary")?;
        let adjusted_entrapment_fdp_by_level = BTreeMap::from([
            ("psm".into(), level4.psm.combined_entrapment_fdp),
            ("peptide".into(), level4.peptide.combined_entrapment_fdp),
            (
                "peptidoform".into(),
                level4.peptidoform.combined_entrapment_fdp,
            ),
            ("protein".into(), level4.protein.combined_entrapment_fdp),
        ]);
        let entrapment_count_by_level = BTreeMap::from([
            ("psm".into(), level4.psm.entrapment),
            ("peptide".into(), level4.peptide.entrapment),
            ("peptidoform".into(), level4.peptidoform.entrapment),
            ("protein".into(), level4.protein.entrapment),
        ]);
        let mut diagnostics = BTreeMap::from([
            (
                "candidate_pool_identity".into(),
                serde_json::json!(stage
                    .candidate_pool
                    .as_ref()
                    .map(|usage| usage.search_fingerprint.clone())),
            ),
            (
                "raw_annotation_cache_identity".into(),
                serde_json::json!(stage
                    .ms2rescore_annotation_cache
                    .as_ref()
                    .map(|usage| usage.raw_prediction_cache_fingerprint.clone())),
            ),
            (
                "fallback_used".into(),
                serde_json::json!(stage.fallback_used),
            ),
            ("model_substitution".into(), serde_json::json!(false)),
            (
                "production_model".into(),
                serde_json::json!(model_slug(&model.model)),
            ),
            (
                "resolved_production_parameters".into(),
                serde_json::to_value(&request.parameters)?,
            ),
            (
                "production_config_snapshot_sha256".into(),
                serde_json::json!(stage.config_snapshot_sha256),
            ),
            (
                "trial_analysis_identity".into(),
                serde_json::json!(stage.input_hash),
            ),
            (
                "results_sha256".into(),
                serde_json::json!(stage.results_sha256),
            ),
            (
                "fitted_artifact_sha256".into(),
                serde_json::json!(sha256_file(&fitted_artifacts)?),
            ),
            (
                "candidate_pool_reused".into(),
                serde_json::json!(stage
                    .candidate_pool
                    .as_ref()
                    .is_some_and(|usage| usage.reused)),
            ),
            (
                "raw_annotation_cache_reused".into(),
                serde_json::json!(stage
                    .ms2rescore_annotation_cache
                    .as_ref()
                    .is_some_and(|usage| usage.reused)),
            ),
            ("target_only_outcomes_used".into(), serde_json::json!(false)),
        ]);
        let evaluations_path = output.join("null_window_evaluations.json");
        if evaluations_path.is_file() {
            let evaluations: Vec<sage_core::decoy_free_fdr::NullWindowEvaluation> =
                serde_json::from_slice(&std::fs::read(&evaluations_path)?)?;
            if let Some(selected) = evaluations.iter().find(|evaluation| evaluation.selected) {
                diagnostics.insert(
                    "selected_null_window".into(),
                    serde_json::json!([selected.min_rank, selected.max_rank]),
                );
            }
        } else if let Some(window) = resolved_expert_window(&model.model, &model.window) {
            // Explicitly configured windows do not produce a landscape file,
            // but they remain part of the fitted scientific state and winner
            // provenance. MSFDR1-SMIX resolves here to its fixed rank 1--1
            // contract.
            diagnostics.insert(
                "selected_null_window".into(),
                serde_json::json!([window.min_rank, window.max_rank]),
            );
        }
        Ok(TrialEvaluation {
            status: TrialStatus::Feasible,
            technical_reason: None,
            empirical_reason: None,
            metrics: Some(TrialMetrics {
                level4_proteins: level4.protein.target,
                level4_canonical_peptides: level4.peptide.target,
                level4_peptidoforms: level4.peptidoform.target,
                level4_psms: level4.psm.target,
                adjusted_entrapment_fdp: level4.protein.combined_entrapment_fdp,
                entrapment_count: level4.protein.entrapment,
                adjusted_entrapment_fdp_by_level,
                entrapment_count_by_level,
                model_complexity: request.parameters.len(),
            }),
            compact_diagnostics: diagnostics,
        })
    }

    fn materialize_winner(&mut self, record: &TrialRecord) -> Result<Option<serde_json::Value>> {
        let root = self.trial_directory(&record.request.trial_id);
        let results = root.join("results.sage.tsv");
        let artifacts = root.join("fitted_model_artifacts.json");
        anyhow::ensure!(results.is_file(), "winner results are missing");
        let selected_null_window = record
            .evaluation
            .compact_diagnostics
            .get("selected_null_window")
            .cloned()
            .or_else(|| {
                self.model_for(&record.request)
                    .ok()
                    .and_then(|model| resolved_expert_window(&model.model, &model.window))
                    .map(|window| serde_json::json!([window.min_rank, window.max_rank]))
            });
        Ok(Some(serde_json::json!({
            "trial_directory": format!("trials/{}", record.request.trial_id),
            "results_sha256": sha256_file(&results)?,
            "fitted_artifact_sha256": artifacts.is_file().then(|| sha256_file(&artifacts)).transpose()?,
            "selected_null_window": selected_null_window,
        })))
    }
}

fn optimizer_identity_from_preflight(
    manifest: &WorkflowManifest,
    dataset: &DatasetIdentity,
    reports: &[ResourcePreflightReport],
) -> Result<OptimizerIdentity> {
    anyhow::ensure!(
        manifest.artifact_reuse_policy == ArtifactReusePolicy::DatasetLocalOnly,
        "parameter optimization cannot use cross-dataset fitted artifacts"
    );
    let resource = |kind: &str| -> Result<&ResourcePreflightReport> {
        reports
            .iter()
            .find(|report| report.resource_type == kind && report.search_space == "+entrapment")
            .with_context(|| format!("strict preflight did not resolve +entrapment {kind}"))
    };
    let candidate = resource("candidate_pool")?;
    let raw = resource("raw_external_prediction_cache")?;
    anyhow::ensure!(
        candidate.valid && candidate.reused && !candidate.generation_allowed,
        "optimizer candidate pool is not strict immutable reuse"
    );
    anyhow::ensure!(
        raw.valid && raw.reused && !raw.generation_allowed,
        "optimizer raw annotation cache is not strict immutable reuse"
    );
    Ok(OptimizerIdentity {
        schema_version: 1,
        execution_mode: manifest
            .parameter_optimizer
            .as_ref()
            .map(|config| config.execution_mode)
            .unwrap_or_default(),
        dataset_identity: dataset.fingerprint.clone(),
        candidate_pool_identity: candidate.actual_fingerprint.clone(),
        raw_annotation_cache_identity: raw.actual_fingerprint.clone(),
        calibrated_annotation_identity: None,
        model_artifact_schema: 1,
        optimizer_schema: crate::parameter_optimizer::PARAMETER_OPTIMIZER_SCHEMA_VERSION,
        optimizer_source_sha256:
            crate::parameter_optimizer::PARAMETER_OPTIMIZER_IMPLEMENTATION_SOURCE_SHA256.into(),
        source_configuration_sha256: dataset.search_config_sha256.clone(),
        catalog_sha256: parameter_catalog_fingerprint()?,
    })
}

fn selected_optimizer_window(result: &OptimizerRunResult) -> Option<NullWindow> {
    result
        .block_order
        .iter()
        .rev()
        .filter_map(|block| result.block_winners.get(block))
        .filter_map(|winner| {
            result
                .trials
                .iter()
                .find(|record| &record.request.trial_id == winner)
        })
        .next()
        .and_then(|record| {
            record
                .evaluation
                .compact_diagnostics
                .get("selected_null_window")
        })
        .and_then(serde_json::Value::as_array)
        .filter(|ranks| ranks.len() == 2)
        .and_then(|ranks| {
            Some(NullWindow {
                min_rank: u32::try_from(ranks[0].as_u64()?).ok()?,
                max_rank: u32::try_from(ranks[1].as_u64()?).ok()?,
            })
        })
}

fn selected_optimizer_parameters(
    result: &OptimizerRunResult,
) -> Result<BTreeMap<String, ParameterValue>> {
    let winner = result
        .block_order
        .iter()
        .rev()
        .filter_map(|block| result.block_winners.get(block))
        .next()
        .context("completed optimizer has no block winner")?;
    Ok(result
        .trials
        .iter()
        .find(|record| &record.request.trial_id == winner)
        .context("optimizer block winner has no trial record")?
        .request
        .parameters
        .clone())
}

fn final_optimizer_winner_record(result: &OptimizerRunResult) -> Result<&TrialRecord> {
    let winner = result
        .block_order
        .iter()
        .rev()
        .filter_map(|block| result.block_winners.get(block))
        .next()
        .context("completed optimizer has no final block winner")?;
    result
        .trials
        .iter()
        .find(|record| &record.request.trial_id == winner)
        .context("optimizer final block winner has no trial record")
}

fn completed_optimizer_expert(
    manifest: &WorkflowManifest,
    model: &ModelWorkflow,
    result: &OptimizerRunResult,
    window: Option<NullWindow>,
) -> Result<CompletedExpert> {
    let winner = final_optimizer_winner_record(result)?;
    let root = manifest
        .output_root
        .join("parameter_optimizer")
        .join(optimizer_expert(&model.model).slug())
        .join("trials")
        .join(&winner.request.trial_id);
    let stage_path = root.join("workflow.stage.json");
    let stage: StageRecord =
        serde_json::from_slice(&std::fs::read(&stage_path).with_context(|| {
            format!(
                "optimizer winner stage is missing: {}",
                stage_path.display()
            )
        })?)?;
    anyhow::ensure!(
        stage.status == "complete"
            && stage.stage == "parameter_optimizer_trial"
            && stage.target_only_calibration_policy.is_none(),
        "optimizer-only expert winner has invalid or target-only stage provenance"
    );
    let artifacts = root.join("fitted_model_artifacts.json");
    anyhow::ensure!(
        stage.results.is_file() && artifacts.is_file(),
        "optimizer-only expert winner is incomplete"
    );
    let candidate = stage
        .candidate_pool
        .as_ref()
        .context("optimizer-only expert winner has no candidate-pool provenance")?;
    let annotation = stage.ms2rescore_annotation_cache.as_ref();
    Ok(CompletedExpert {
        model: model.model.clone(),
        window,
        optimized_artifacts: artifacts.clone(),
        optimized_results: stage.results.clone(),
        ms2rescore_artifacts: stage.external_features_enabled.then_some(artifacts),
        ms2rescore_results: stage
            .external_features_enabled
            .then_some(stage.results.clone()),
        calibration_stage: stage.stage,
        calibration_results: stage.results,
        target_only_results: PathBuf::new(),
        target_only_calibration_policy: manifest.target_only_calibration_policy,
        calibration_search_fingerprint: candidate.search_fingerprint.clone(),
        fitted_external_profile_identity_sha256: stage.fitted_external_profile_identity_sha256,
        fitted_external_profile_calibration: stage.external_profile_calibration,
        annotation_cache_fingerprint: annotation.map(|usage| usage.annotation_fingerprint.clone()),
        annotation_cache_manifest_sha256: annotation
            .map(|usage| sha256_file(&usage.manifest))
            .transpose()?,
        annotation_cache_payload_sha256: annotation
            .map(|usage| sha256_file(&usage.payload))
            .transpose()?,
    })
}

fn prune_nonwinner_trial_payloads(root: &Path, result: &OptimizerRunResult) -> Result<()> {
    let winners = result
        .block_winners
        .values()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let trials = root.join("trials");
    if !trials.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(&trials)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && !winners.contains(entry.file_name().to_string_lossy().as_ref())
        {
            std::fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn optimize_model_parameters(
    manifest: &WorkflowManifest,
    dataset: &DatasetIdentity,
    resource_preflight: &[ResourcePreflightReport],
    model: &ModelWorkflow,
    fasta: &Path,
    parallel: usize,
    ensemble_lock: Option<&EnsembleLock>,
    runtime: &mut WorkflowRuntime,
) -> Result<
    Option<(
        OptimizerRunResult,
        BTreeMap<String, ParameterValue>,
        Option<NullWindow>,
    )>,
> {
    let Some(config) = manifest
        .parameter_optimizer
        .as_ref()
        .and_then(|config| optimizer_config_for_expert(config, optimizer_expert(&model.model)))
    else {
        return Ok(None);
    };
    config.validate()?;
    let root = manifest
        .output_root
        .join("parameter_optimizer")
        .join(optimizer_expert(&model.model).slug());
    std::fs::create_dir_all(&root)?;
    let identity = optimizer_identity_from_preflight(manifest, dataset, resource_preflight)?;
    write_json_atomic(&root.join("identity.json"), &identity)?;
    let result = if config.implementation_smoke_only {
        let mut evaluator = InfrastructureSmokeEvaluator {
            root: &root,
            candidate_pool_identity: &identity.candidate_pool_identity,
            raw_annotation_cache_identity: &identity.raw_annotation_cache_identity,
        };
        run_optimizer(
            &config,
            &identity,
            &root.join("optimizer.checkpoint.json"),
            &mut evaluator,
        )?
    } else {
        let mut evaluator = WorkflowTrialEvaluator {
            manifest,
            dataset,
            base_model: model,
            blocks: &config.blocks,
            fasta,
            root: &root,
            parallel,
            ensemble_lock,
            runtime,
        };
        run_optimizer(
            &config,
            &identity,
            &root.join("optimizer.checkpoint.json"),
            &mut evaluator,
        )?
    };
    write_json_atomic(&root.join("optimizer.result.json"), &result)?;
    prune_nonwinner_trial_payloads(&root, &result)?;
    anyhow::ensure!(
        matches!(
            result.outcome,
            OptimizerOutcome::ExhaustiveBoundedOptimum | OptimizerOutcome::CompletedHeuristicLocal
        ),
        "parameter optimizer for {} ended as {:?}; no baseline substitution is permitted",
        model_slug(&model.model),
        result.outcome
    );
    let parameters = selected_optimizer_parameters(&result)?;
    let window = selected_optimizer_window(&result);
    Ok(Some((result, parameters, window)))
}

pub fn execute_workflow(
    manifest_path: &Path,
    source_repo: &Path,
    parallel: usize,
    plan_only: bool,
) -> Result<WorkflowState> {
    let mut manifest = WorkflowManifest::load(manifest_path)?;
    let manifest_hash = sha256_file(manifest_path)?;
    let dataset = compute_dataset_identity(&manifest)?;
    let strict_preflight_enabled =
        manifest.require_existing_candidate_pool || manifest.require_existing_annotation_cache;
    let resource_preflight = if strict_preflight_enabled {
        strict_resource_preflight(&manifest, parallel)?
    } else {
        Vec::new()
    };
    if plan_only && strict_preflight_enabled {
        let planned_models = planned_model_reports(&manifest);
        return Ok(WorkflowState {
            schema_version: 2,
            manifest_hash,
            dataset,
            entrapment: None,
            entrapment_fasta_parity: None,
            baseline: None,
            stages: Vec::new(),
            candidate_pools: Vec::new(),
            ms2rescore_annotation_caches: Vec::new(),
            validation: Vec::new(),
            missing_runs: Vec::new(),
            invalid_runs: Vec::new(),
            stage_comparisons: Vec::new(),
            ensemble_expert_gates: Vec::new(),
            ensemble_expert_gates_participation_effect: nonblocking_ensemble_gate_effect(),
            parity_comparisons: Vec::new(),
            tdc_benchmarks: Vec::new(),
            release_gate: ReleaseGate {
                status: ReleaseGateStatus::NotEvaluable,
                eligible_for_statistical_default_change: false,
                reasons: vec!["plan-only resource preflight; no scientific execution".into()],
                not_evaluable_reasons: vec![
                    "plan-only resource preflight; no scientific execution".into(),
                ],
                not_eligible_reasons: Vec::new(),
                calibrated_tdc_improvements: 0,
            },
            transfer_stability: Vec::new(),
            pending_validation_gates: Vec::new(),
            ensemble_interaction_calibration: None,
            resource_preflight,
            planned_models,
            parameter_optimization: Vec::new(),
            parameter_optimizer_execution: manifest.parameter_optimizer.as_ref().map(|config| {
                optimizer_execution_report(config, &[], "plan_only_preflight_complete")
            }),
        });
    }
    std::fs::create_dir_all(&manifest.output_root)?;
    write_json_atomic(
        &manifest.output_root.join("workflow.dataset.json"),
        &dataset,
    )?;
    write_json_atomic(
        &manifest.output_root.join("workflow.manifest.resolved.json"),
        &manifest,
    )?;

    let baseline = if let Some(specification) = &manifest.baseline {
        let frozen = freeze_baseline(
            &specification.paths,
            source_repo,
            specification.status.clone(),
        )?;
        write_json_atomic(&specification.output_manifest, &frozen)?;
        Some(frozen)
    } else {
        None
    };

    let parameter_input = Input::load(manifest.search_config.to_string_lossy().as_ref())?;
    let parameters = parameter_input.database.make_parameters();
    let entrapment_report_path = manifest.output_root.join("entrapment.generation.json");
    let (active_entrapment_fasta, entrapment) = match manifest.entrapment.database_mode {
        EntrapmentDatabaseMode::NativeGenerated => {
            let expected_input_sha256 = entrapment_generation_input_sha256(
                &parameters,
                &manifest.target_fasta,
                &manifest.entrapment.foreign_fastas,
                manifest.entrapment.seed,
                manifest.entrapment.protein_fold,
                &manifest.entrapment.foreign_source_mode,
                &manifest.entrapment.shared_peptide_exclusion_mode,
                manifest.entrapment.selected_foreign_fasta.as_deref(),
            )?;
            let report = if plan_only && !manifest.entrapment.output_fasta.is_file() {
                None
            } else if manifest.entrapment.output_fasta.is_file() && manifest.resume {
                let report: EntrapmentGenerationReport = serde_json::from_slice(
                    &std::fs::read(&entrapment_report_path).with_context(|| {
                        format!(
                            "resume requested but {} is missing",
                            entrapment_report_path.display()
                        )
                    })?,
                )?;
                anyhow::ensure!(
                    report.schema_version == 2,
                    "existing entrapment report predates Phase 2 provenance; regenerate it"
                );
                anyhow::ensure!(
                    report.generation_input_sha256 == expected_input_sha256,
                    "existing entrapment FASTA was generated from different inputs or digestion settings"
                );
                anyhow::ensure!(
                    report.output_sha256 == sha256_file(&manifest.entrapment.output_fasta)?,
                    "existing entrapment FASTA hash does not match its generation report"
                );
                Some(report)
            } else {
                let report = generate_foreign_entrapment(
                    &parameters,
                    &manifest.target_fasta,
                    &manifest.entrapment.foreign_fastas,
                    &manifest.entrapment.output_fasta,
                    manifest.entrapment.seed,
                    manifest.entrapment.protein_fold,
                    manifest.entrapment.foreign_source_mode.clone(),
                    manifest.entrapment.shared_peptide_exclusion_mode.clone(),
                    manifest.entrapment.selected_foreign_fasta.as_deref(),
                )?;
                write_json_atomic(&entrapment_report_path, &report)?;
                Some(report)
            };
            (
                manifest.entrapment.output_fasta.clone(),
                report.map(|generation| EntrapmentDatabaseReport::NativeGenerated { generation }),
            )
        }
        EntrapmentDatabaseMode::FrozenLegacy => {
            let path = manifest
                .entrapment
                .frozen_legacy_fasta
                .as_ref()
                .context("validated frozen legacy FASTA is missing")?;
            let frozen = inspect_frozen_entrapment(&parameters, &manifest.target_fasta, path)?;
            write_json_atomic(
                &manifest.output_root.join("entrapment.frozen.json"),
                &frozen,
            )?;
            (
                path.clone(),
                Some(EntrapmentDatabaseReport::FrozenLegacy { frozen }),
            )
        }
    };
    let entrapment_fasta_parity = match (&entrapment, &manifest.entrapment.legacy_parity_reference)
    {
        (Some(EntrapmentDatabaseReport::NativeGenerated { generation }), Some(reference)) => {
            let report = compare_generated_to_legacy(&parameters, generation, reference)?;
            write_json_atomic(
                &manifest.output_root.join("entrapment.fasta_parity.json"),
                &report,
            )?;
            Some(report)
        }
        _ => None,
    };
    if let Some(report) = entrapment.as_ref() {
        let measured = report.measured();
        manifest.validation.effective_ratios = EffectiveRatios {
            psm: measured.peptidoform_ratio,
            peptide: measured.peptide_ratio,
            protein: measured.protein_ratio,
        };
        write_json_atomic(&manifest.output_root.join("entrapment.input.json"), report)?;
        write_json_atomic(
            &manifest.output_root.join("workflow.manifest.resolved.json"),
            &manifest,
        )?;
    }

    let mut stages = Vec::new();
    let mut completed_experts = Vec::new();
    let mut runtime_expert_failures = BTreeMap::<String, Vec<String>>::new();
    let mut ensemble_interaction_report = None;
    let mut runtime = WorkflowRuntime::default();
    let mut optimization_results = Vec::new();
    let mut ordered_models = manifest
        .models
        .iter()
        .filter(|model| model.enabled)
        .collect::<Vec<_>>();
    ordered_models.sort_by_key(|model| (model.model == ModelFit::Ensemble) as u8);
    'models: for model in ordered_models {
        let model_root = manifest.output_root.join(model_slug(&model.model));
        let imported_diagnostic_artifact = if manifest.artifact_reuse_policy
            == ArtifactReusePolicy::CrossDatasetDiagnostic
            && model.model != ModelFit::Ensemble
        {
            manifest
                .locked_expert_artifacts
                .get(model_slug(&model.model))
                .map(PathBuf::as_path)
        } else {
            None
        };
        let ensemble_lock = if model.model == ModelFit::Ensemble && !plan_only {
            let lock = if let Some(path) = manifest.ensemble_lock.as_ref() {
                let lock = canonicalize_ensemble_lock(serde_json::from_slice::<EnsembleLock>(
                    &std::fs::read(path).with_context(|| {
                        format!("failed to read Ensemble lock {}", path.display())
                    })?,
                )?);
                match manifest.artifact_reuse_policy {
                    ArtifactReusePolicy::DatasetLocalOnly => anyhow::ensure!(
                        lock.dataset_fingerprint == dataset.fingerprint,
                        "Ensemble lock belongs to a different dataset"
                    ),
                    ArtifactReusePolicy::CrossDatasetDiagnostic => {
                        log::warn!("explicit diagnostic-only reuse of an Ensemble lock")
                    }
                }
                lock
            } else {
                build_ensemble_lock_with_failures(
                    &manifest,
                    &manifest_hash,
                    &dataset,
                    &completed_experts,
                    &runtime_expert_failures,
                )?
            };
            write_json_atomic(&manifest.output_root.join("ensemble.lock.json"), &lock)?;
            Some(lock)
        } else {
            None
        };
        if model.model == ModelFit::Ensemble
            && ensemble_lock.as_ref().is_some_and(|lock| !lock.evaluable)
        {
            let lock = ensemble_lock.as_ref().unwrap();
            log::warn!(
                "Ensemble is not evaluable and will be skipped without invalidating individual experts: {}",
                lock.not_evaluable_reasons.join("; ")
            );
            continue;
        }
        let mut resolved_model = (*model).clone();
        let mut parameter_overrides = None;
        if !plan_only {
            if let Some((result, parameters, window)) = optimize_model_parameters(
                &manifest,
                &dataset,
                &resource_preflight,
                &resolved_model,
                &active_entrapment_fasta,
                parallel,
                ensemble_lock.as_ref(),
                &mut runtime,
            )? {
                if let Some(window) = window {
                    resolved_model.window = Some(window);
                    resolved_model.candidate_windows.clear();
                    resolved_model.window_optimizer = None;
                }
                parameter_overrides = Some(parameters);
                if manifest
                    .parameter_optimizer
                    .as_ref()
                    .is_some_and(ParameterOptimizerConfig::optimization_only)
                    && model.model != ModelFit::Ensemble
                {
                    completed_experts.push(completed_optimizer_expert(
                        &manifest,
                        &resolved_model,
                        &result,
                        resolved_expert_window(&resolved_model.model, &resolved_model.window),
                    )?);
                }
                optimization_results.push(result);
            }
        }
        if manifest
            .parameter_optimizer
            .as_ref()
            .is_some_and(ParameterOptimizerConfig::stops_after_optimization)
        {
            continue;
        }
        let model = &resolved_model;
        let optimized = match run_search_stage(
            &manifest,
            &dataset,
            model,
            "optimized",
            &active_entrapment_fasta,
            &model_root.join("optimized"),
            false,
            false,
            parallel,
            plan_only,
            imported_diagnostic_artifact,
            ensemble_lock.as_ref(),
            None,
            parameter_overrides.as_ref(),
            &mut runtime,
        ) {
            Ok(record) => record,
            Err(error) if model.model != ModelFit::Ensemble => {
                let reason = format!("optimized expert stage failed technically: {error:#}");
                log::warn!("workflow: excluding {}: {reason}", model_slug(&model.model));
                runtime_expert_failures
                    .entry(model_slug(&model.model).into())
                    .or_default()
                    .push(reason);
                continue;
            }
            Err(error) => return Err(error),
        };
        stages.push(optimized.clone());

        let mut locked_model = model.clone();
        if !plan_only && (!model.candidate_windows.is_empty() || model.window_optimizer.is_some()) {
            let selected_window = (|| -> Result<NullWindow> {
                let path = model_root.join("optimized/null_window_evaluations.json");
                let evaluations: Vec<sage_core::decoy_free_fdr::NullWindowEvaluation> =
                    serde_json::from_slice(&std::fs::read(&path).with_context(|| {
                        format!("optimizer did not produce {}", path.display())
                    })?)?;
                let selected = evaluations
                    .iter()
                    .find(|evaluation| evaluation.selected)
                    .context("optimizer report has no selected window")?;
                Ok(NullWindow {
                    min_rank: selected.min_rank,
                    max_rank: selected.max_rank,
                })
            })();
            let selected_window = match selected_window {
                Ok(window) => window,
                Err(error) => {
                    let reason = format!("optimized expert window is invalid: {error:#}");
                    log::warn!("workflow: excluding {}: {reason}", model_slug(&model.model));
                    runtime_expert_failures
                        .entry(model_slug(&model.model).into())
                        .or_default()
                        .push(reason);
                    continue;
                }
            };
            locked_model.window = Some(selected_window);
            locked_model.candidate_windows.clear();
            locked_model.window_optimizer = None;
        }

        let mut ms2_record = None;
        if !matches!(locked_model.ms2rescore, Ms2RescorePolicy::Never) {
            let record = match run_search_stage(
                &manifest,
                &dataset,
                &locked_model,
                "ms2rescore",
                &active_entrapment_fasta,
                &model_root.join("ms2rescore"),
                true,
                false,
                parallel,
                plan_only,
                imported_diagnostic_artifact,
                ensemble_lock.as_ref(),
                None,
                parameter_overrides.as_ref(),
                &mut runtime,
            ) {
                Ok(record) => record,
                Err(error) if model.model != ModelFit::Ensemble => {
                    let reason = format!("MS2Rescore expert stage failed technically: {error:#}");
                    log::warn!("workflow: excluding {}: {reason}", model_slug(&model.model));
                    runtime_expert_failures
                        .entry(model_slug(&model.model).into())
                        .or_default()
                        .push(reason);
                    continue;
                }
                Err(error) => return Err(error),
            };
            stages.push(record.clone());
            ms2_record = Some(record);
        }
        let use_ms2_for_final = match locked_model.ms2rescore {
            Ms2RescorePolicy::Never => false,
            Ms2RescorePolicy::Always => true,
            Ms2RescorePolicy::Measure if plan_only => false,
            Ms2RescorePolicy::Measure => {
                let decision = (|| -> Result<bool> {
                    let ms2 = ms2_record
                        .as_ref()
                        .context("MS2Rescore measurement stage was not recorded")?;
                    let optimized_summary = summarize_run(
                        &ValidationRun {
                            method: model_slug(&locked_model.model).into(),
                            stage: "optimized".into(),
                            results: optimized.results.clone(),
                            mode: ValidationMode::DecoyFree,
                            expected_search_space: Some("+Ent".into()),
                            calibration_stage: None,
                            target_only_calibration_policy: None,
                            release_candidate: true,
                        },
                        &manifest.validation.effective_ratios,
                        manifest.validation.fdr_threshold,
                    )?;
                    let ms2_summary = summarize_run(
                        &ValidationRun {
                            method: model_slug(&locked_model.model).into(),
                            stage: "ms2rescore".into(),
                            results: ms2.results.clone(),
                            mode: ValidationMode::DecoyFree,
                            expected_search_space: Some("+Ent".into()),
                            calibration_stage: None,
                            target_only_calibration_policy: None,
                            release_candidate: true,
                        },
                        &manifest.validation.effective_ratios,
                        manifest.validation.fdr_threshold,
                    )?;
                    Ok(
                        match (
                            optimized_summary.iter().find(|row| row.layer == "level4"),
                            ms2_summary.iter().find(|row| row.layer == "level4"),
                        ) {
                            (Some(before), Some(after)) => {
                                let gain =
                                    after.peptide.target.saturating_sub(before.peptide.target);
                                let raw_before = optimized_summary
                                    .iter()
                                    .find(|row| row.layer == "raw_q")
                                    .and_then(|row| row.peptide.combined_entrapment_fdp);
                                let raw_after = ms2_summary
                                    .iter()
                                    .find(|row| row.layer == "raw_q")
                                    .and_then(|row| row.peptide.combined_entrapment_fdp);
                                let fdp_increase = match (raw_before, raw_after) {
                                    (Some(before), Some(after)) => after - before,
                                    _ => f64::INFINITY,
                                };
                                gain >= locked_model.minimum_level4_peptide_gain.unwrap_or(1)
                                    && fdp_increase
                                        <= locked_model.maximum_raw_fdp_increase.unwrap_or(0.0)
                            }
                            _ => false,
                        },
                    )
                })();
                match decision {
                    Ok(selected) => selected,
                    Err(error) => {
                        log::warn!(
                            "nonblocking MS2Rescore measurement is unavailable for {}: {error:#}; retaining optimized fitted evidence",
                            model_slug(&locked_model.model)
                        );
                        false
                    }
                }
            }
        };
        if model.model == ModelFit::Ensemble && !plan_only {
            let final_lock = ensemble_lock
                .as_ref()
                .context("Ensemble stage has no assembled lock")?;
            let baseline_lock = match interaction_baseline_lock(final_lock) {
                Ok(lock) => {
                    if let Err(error) = write_json_atomic(
                        &manifest
                            .output_root
                            .join("ensemble.interaction_baseline.lock.json"),
                        &lock,
                    ) {
                        log::warn!(
                            "nonblocking Ensemble interaction diagnostic lock was not written: {error:#}"
                        );
                    }
                    Some(lock)
                }
                Err(error) => {
                    log::warn!(
                        "nonblocking Ensemble interaction diagnostic lock is unavailable: {error:#}"
                    );
                    None
                }
            };
            let final_experts = final_lock
                .experts
                .iter()
                .filter(|expert| expert.enabled)
                .map(|expert| model_slug(&expert.model).into())
                .collect::<Vec<_>>();
            let baseline_experts = baseline_lock
                .as_ref()
                .map(|lock| {
                    lock.experts
                        .iter()
                        .filter(|expert| expert.enabled)
                        .map(|expert| model_slug(&expert.model).into())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let final_record = if use_ms2_for_final {
                ms2_record
                    .as_ref()
                    .context("selected Ensemble MS2Rescore stage is missing")?
            } else {
                &optimized
            };
            let report_attempt = (|| -> Result<EnsembleInteractionCalibration> {
                let baseline_lock = baseline_lock
                    .as_ref()
                    .context("interaction baseline lock is unavailable")?;
                let final_summaries = summarize_run(
                    &ValidationRun {
                        method: "ensemble".into(),
                        stage: final_record.stage.clone(),
                        results: final_record.results.clone(),
                        mode: ValidationMode::DecoyFree,
                        expected_search_space: Some("+Ent".into()),
                        calibration_stage: None,
                        target_only_calibration_policy: None,
                        release_candidate: true,
                    },
                    &manifest.validation.effective_ratios,
                    manifest.validation.fdr_threshold,
                )?;
                if baseline_experts == final_experts {
                    return ensemble_interaction_calibration(
                        &final_summaries,
                        &final_summaries,
                        &manifest.validation.effective_ratios,
                        manifest.validation.fdr_threshold,
                        final_lock.raw_q_interaction_warning_threshold,
                        baseline_experts.clone(),
                        final_experts.clone(),
                    );
                }
                if !baseline_lock.evaluable {
                    return Ok(unavailable_interaction_report(
                        baseline_experts.clone(),
                        final_experts.clone(),
                        baseline_lock.not_evaluable_reasons.join("; "),
                    ));
                }
                let mut baseline_record = run_search_stage(
                    &manifest,
                    &dataset,
                    &locked_model,
                    "ensemble_interaction_baseline",
                    &active_entrapment_fasta,
                    &model_root
                        .join("interaction_baseline")
                        .join(if use_ms2_for_final {
                            "ms2rescore"
                        } else {
                            "optimized"
                        }),
                    use_ms2_for_final,
                    false,
                    parallel,
                    false,
                    None,
                    Some(baseline_lock),
                    None,
                    parameter_overrides.as_ref(),
                    &mut runtime,
                )?;
                baseline_record.release_candidate = false;
                write_json_atomic(
                    &baseline_record
                        .config_snapshot
                        .parent()
                        .context("interaction baseline configuration has no parent directory")?
                        .join("workflow.stage.json"),
                    &baseline_record,
                )?;
                let baseline_summaries = summarize_run(
                    &ValidationRun {
                        method: "ensemble_interaction_baseline".into(),
                        stage: "ensemble_interaction_baseline".into(),
                        results: baseline_record.results,
                        mode: ValidationMode::DecoyFree,
                        expected_search_space: Some("+Ent".into()),
                        calibration_stage: None,
                        target_only_calibration_policy: None,
                        release_candidate: false,
                    },
                    &manifest.validation.effective_ratios,
                    manifest.validation.fdr_threshold,
                )?;
                ensemble_interaction_calibration(
                    &baseline_summaries,
                    &final_summaries,
                    &manifest.validation.effective_ratios,
                    manifest.validation.fdr_threshold,
                    final_lock.raw_q_interaction_warning_threshold,
                    baseline_experts.clone(),
                    final_experts.clone(),
                )
            })();
            let mut report = report_attempt.unwrap_or_else(|error| {
                log::warn!("nonblocking Ensemble interaction diagnostic is unavailable: {error:#}");
                unavailable_interaction_report(
                    baseline_experts.clone(),
                    final_experts.clone(),
                    format!("nonblocking interaction diagnostic unavailable: {error:#}"),
                )
            });
            report.baseline_lock_analysis_fingerprint = baseline_lock
                .as_ref()
                .map(|lock| lock.analysis_fingerprint.clone());
            report.final_lock_analysis_fingerprint = Some(final_lock.analysis_fingerprint.clone());
            if let Err(error) = write_json_atomic(
                &manifest
                    .output_root
                    .join("validation.ensemble_interaction.json"),
                &report,
            ) {
                log::warn!(
                    "nonblocking Ensemble interaction diagnostic was not written: {error:#}"
                );
            }
            if let Some(record) = stages
                .iter_mut()
                .find(|record| record.results == final_record.results)
            {
                record.schema_version = 4;
                record.ensemble_interaction_calibration = Some(report.clone());
                if let Some(parent) = record.config_snapshot.parent() {
                    if let Err(error) =
                        write_json_atomic(&parent.join("workflow.stage.json"), record)
                    {
                        log::warn!(
                            "nonblocking Ensemble interaction checkpoint update failed: {error:#}"
                        );
                    }
                } else {
                    log::warn!(
                        "nonblocking Ensemble interaction checkpoint has no parent directory"
                    );
                }
            }
            ensemble_interaction_report = Some(report);
        }
        let optimized_artifact = model_root.join("optimized/fitted_model_artifacts.json");
        let ms2_artifact = model_root.join("ms2rescore/fitted_model_artifacts.json");
        let frozen_model_artifacts = if imported_diagnostic_artifact.is_some() {
            imported_diagnostic_artifact
        } else if plan_only || model.model == ModelFit::Ensemble {
            None
        } else if ms2_artifact.is_file() && use_ms2_for_final {
            Some(ms2_artifact.as_path())
        } else if optimized_artifact.is_file() {
            Some(optimized_artifact.as_path())
        } else {
            let reason =
                "fitted model artifacts were not produced by the entrapment search".to_owned();
            log::warn!("workflow: excluding {}: {reason}", model_slug(&model.model));
            runtime_expert_failures
                .entry(model_slug(&model.model).into())
                .or_default()
                .push(reason);
            continue;
        };
        let calibration_record = if use_ms2_for_final {
            ms2_record
                .as_ref()
                .context("selected MS2Rescore stage is missing")?
        } else {
            &optimized
        };
        let selected_artifact = frozen_model_artifacts.map(Path::to_path_buf);
        let window_provenance = WindowProvenance {
            schema_version: 1,
            source_stage: calibration_record.stage.clone(),
            source_model: model_slug(&locked_model.model).into(),
            source_dataset_id: dataset.dataset_id.clone(),
            source_dataset_fingerprint: dataset.fingerprint.clone(),
            min_rank: locked_model.window.as_ref().map(|window| window.min_rank),
            max_rank: locked_model.window.as_ref().map(|window| window.max_rank),
            source_fitted_artifact: selected_artifact.clone(),
            source_fitted_artifact_sha256: selected_artifact
                .as_ref()
                .filter(|path| path.is_file())
                .map(|path| sha256_file(path))
                .transpose()?,
            source_search_fingerprint: calibration_record
                .candidate_pool
                .as_ref()
                .map(|usage| usage.search_fingerprint.clone()),
            candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
        };
        let requested_policy = locked_model
            .target_only_calibration_policy
            .unwrap_or(manifest.target_only_calibration_policy);
        let target_policies = concrete_target_only_policies(requested_policy);
        let mut release_target_only = None;
        let mut release_target_only_policy = None;
        for (index, (policy, release_candidate)) in target_policies.iter().copied().enumerate() {
            let context = TargetOnlyStageContext {
                policy,
                release_candidate,
                window_provenance: window_provenance.clone(),
                // The target FASTA creates a strict fingerprint distinct from
                // +entrapment. Reuse that exact target population across
                // models/policies unless matched-fragment output requires a
                // fresh search payload that the immutable pool omits.
                allow_candidate_pool_reuse: allow_target_candidate_pool_reuse(
                    manifest.annotate_target_matches,
                    index,
                ),
            };
            let target_output_directory = model_root.join("target_only").join(match policy {
                TargetOnlyCalibrationPolicy::RefitWithLockedWindow => "refit_with_locked_window",
                TargetOnlyCalibrationPolicy::ReuseDatasetArtifact => "reuse_dataset_artifact",
                TargetOnlyCalibrationPolicy::CompareBoth => unreachable!(),
            });
            let capability = target_only_policy_capability(&locked_model.model, policy);
            if !capability.supported {
                anyhow::ensure!(
                    !release_candidate,
                    "unsupported target-only policy cannot be a release candidate"
                );
                let reason = capability
                    .reason
                    .clone()
                    .context("unsupported target-only policy has no reason")?;
                let input_hash = hash_stage(
                    &manifest,
                    &dataset,
                    &locked_model,
                    policy.stage_name(),
                    &manifest.target_fasta,
                    use_ms2_for_final,
                    frozen_model_artifacts,
                    ensemble_lock.as_ref(),
                    Some(&context),
                    parameter_overrides.as_ref(),
                )?;
                let record = StageRecord {
                    schema_version: 4,
                    stage: policy.stage_name().into(),
                    model: model_slug(&locked_model.model).into(),
                    input_hash,
                    status: "not_evaluable".into(),
                    results: target_output_directory.join("results.sage.tsv"),
                    config_snapshot: target_output_directory.join("workflow.search.resolved.json"),
                    results_sha256: String::new(),
                    config_snapshot_sha256: String::new(),
                    external_features_enabled: use_ms2_for_final,
                    calibration_mode: "reuse_dataset_artifact".into(),
                    dataset_id: dataset.dataset_id.clone(),
                    dataset_fingerprint: dataset.fingerprint.clone(),
                    artifact_fit_dataset_fingerprint: optimized
                        .artifact_fit_dataset_fingerprint
                        .clone(),
                    candidate_pool: release_target_only
                        .as_ref()
                        .and_then(|stage: &StageRecord| stage.candidate_pool.clone()),
                    require_existing_candidate_pool: manifest.require_existing_candidate_pool,
                    require_existing_annotation_cache: manifest.require_existing_annotation_cache,
                    ms2rescore_annotation_cache: release_target_only
                        .as_ref()
                        .and_then(|stage: &StageRecord| stage.ms2rescore_annotation_cache.clone()),
                    target_only_calibration_policy: Some(policy),
                    release_candidate,
                    window_provenance: Some(window_provenance.clone()),
                    external_profile_calibration: release_target_only
                        .as_ref()
                        .and_then(|stage: &StageRecord| stage.external_profile_calibration.clone()),
                    ensemble_shared_profile_contract_sha256: ensemble_lock
                        .as_ref()
                        .and_then(|lock| lock.shared_external_profile_contract_sha256.clone()),
                    fitted_external_profile_identity_sha256: None,
                    evaluable: false,
                    not_evaluable_reason: Some(reason),
                    target_only_policy_capability: Some(capability),
                    nuisance_state_provenance: Some(
                        "not_applied_unsupported_cross_search_space_reuse".into(),
                    ),
                    target_only_window_tuning: Some(false),
                    complete_dataset_artifact_reused: Some(false),
                    fallback_used: false,
                    fallback_reason: None,
                    model_artifact_schema: optimized.model_artifact_schema,
                    ensemble_interaction_calibration: None,
                    parameter_overrides: parameter_overrides.clone().unwrap_or_default(),
                };
                std::fs::create_dir_all(&target_output_directory)?;
                write_json_atomic(
                    &target_output_directory.join("workflow.stage.json"),
                    &record,
                )?;
                stages.push(record);
                continue;
            }
            let target_only = match run_search_stage(
                &manifest,
                &dataset,
                &locked_model,
                policy.stage_name(),
                &manifest.target_fasta,
                &target_output_directory,
                use_ms2_for_final,
                manifest.annotate_target_matches && index == 0,
                parallel,
                plan_only,
                (policy == TargetOnlyCalibrationPolicy::ReuseDatasetArtifact)
                    .then_some(frozen_model_artifacts)
                    .flatten(),
                ensemble_lock.as_ref(),
                Some(&context),
                parameter_overrides.as_ref(),
                &mut runtime,
            ) {
                Ok(record) => record,
                Err(error) if model.model != ModelFit::Ensemble => {
                    let reason = format!("target-only expert stage failed technically: {error:#}");
                    log::warn!("workflow: excluding {}: {reason}", model_slug(&model.model));
                    runtime_expert_failures
                        .entry(model_slug(&model.model).into())
                        .or_default()
                        .push(reason);
                    continue 'models;
                }
                Err(error) => return Err(error),
            };
            if release_candidate {
                release_target_only = Some(target_only.clone());
                release_target_only_policy = Some(policy);
            }
            stages.push(target_only);
        }
        let target_only =
            release_target_only.context("target-only policy has no release result")?;
        let target_only_policy = release_target_only_policy
            .context("target-only policy has no release interpretation")?;
        if model.model != ModelFit::Ensemble && !plan_only {
            let completed = (|| -> Result<CompletedExpert> {
                anyhow::ensure!(
                    frozen_model_artifacts.is_some(),
                    "individual expert has no selected fitted artifact"
                );
                let (fitted_external_profile_identity_sha256, fitted_external_profile_calibration) =
                    if ms2_artifact.is_file() {
                        let artifacts: DfRunArtifacts =
                            serde_json::from_slice(&std::fs::read(&ms2_artifact)?).with_context(
                                || format!("invalid fitted artifacts {}", ms2_artifact.display()),
                            )?;
                        fitted_external_profile_identity(&artifacts)?
                            .map(|(identity, calibration)| (Some(identity), Some(calibration)))
                            .unwrap_or((None, None))
                    } else {
                        (None, None)
                    };
                let annotation_cache = ms2_record
                    .as_ref()
                    .and_then(|record| record.ms2rescore_annotation_cache.as_ref());
                Ok(CompletedExpert {
                    model: locked_model.model.clone(),
                    window: resolved_expert_window(&locked_model.model, &locked_model.window),
                    optimized_artifacts: optimized_artifact.clone(),
                    optimized_results: optimized.results.clone(),
                    ms2rescore_artifacts: ms2_artifact.is_file().then_some(ms2_artifact.clone()),
                    ms2rescore_results: ms2_record.as_ref().map(|record| record.results.clone()),
                    calibration_stage: if use_ms2_for_final {
                        "ms2rescore".into()
                    } else {
                        "optimized".into()
                    },
                    calibration_results: if use_ms2_for_final {
                        ms2_record
                            .as_ref()
                            .context("selected MS2Rescore stage is missing")?
                            .results
                            .clone()
                    } else {
                        optimized.results.clone()
                    },
                    target_only_results: target_only.results,
                    target_only_calibration_policy: target_only_policy,
                    calibration_search_fingerprint: calibration_record
                        .candidate_pool
                        .as_ref()
                        .context("calibration stage has no candidate-pool provenance")?
                        .search_fingerprint
                        .clone(),
                    fitted_external_profile_identity_sha256,
                    fitted_external_profile_calibration,
                    annotation_cache_fingerprint: annotation_cache
                        .map(|usage| usage.annotation_fingerprint.clone()),
                    annotation_cache_manifest_sha256: annotation_cache
                        .map(|usage| sha256_file(&usage.manifest))
                        .transpose()?,
                    annotation_cache_payload_sha256: annotation_cache
                        .map(|usage| sha256_file(&usage.payload))
                        .transpose()?,
                })
            })();
            match completed {
                Ok(expert) => completed_experts.push(expert),
                Err(error) => {
                    let reason = format!("expert completion provenance is invalid: {error:#}");
                    log::warn!("workflow: excluding {}: {reason}", model_slug(&model.model));
                    runtime_expert_failures
                        .entry(model_slug(&model.model).into())
                        .or_default()
                        .push(reason);
                    continue;
                }
            }
        }
    }

    if let Some(config) = manifest
        .parameter_optimizer
        .as_ref()
        .filter(|config| config.optimization_only())
    {
        let expected = config
            .selected_experts
            .iter()
            .filter(|expert| {
                config
                    .blocks
                    .iter()
                    .any(|block| block.enabled && block.expert == Some(**expert))
            })
            .count();
        anyhow::ensure!(
            optimization_results.len() == expected,
            "optimization_only completed {} optimizer result(s), expected {expected}",
            optimization_results.len()
        );
        let execution = optimizer_execution_report(
            config,
            &optimization_results,
            "completed_entrapment_optimization",
        );
        anyhow::ensure!(
            execution.selected_entrapment_winners.len() == expected,
            "optimization_only did not materialize every selected +entrapment winner"
        );
        write_json_atomic(
            &manifest
                .output_root
                .join("workflow.parameter_optimizer.execution.json"),
            &execution,
        )?;
        let scope_reason =
            "optimization_only completed; post-selection and target-only stages were not run by execution scope";
        let state = WorkflowState {
            schema_version: 2,
            manifest_hash,
            dataset,
            entrapment,
            entrapment_fasta_parity,
            baseline,
            stages: Vec::new(),
            candidate_pools: Vec::new(),
            ms2rescore_annotation_caches: Vec::new(),
            validation: Vec::new(),
            missing_runs: Vec::new(),
            invalid_runs: Vec::new(),
            stage_comparisons: Vec::new(),
            ensemble_expert_gates: Vec::new(),
            ensemble_expert_gates_participation_effect: nonblocking_ensemble_gate_effect(),
            parity_comparisons: Vec::new(),
            tdc_benchmarks: Vec::new(),
            release_gate: ReleaseGate {
                status: ReleaseGateStatus::NotEvaluable,
                eligible_for_statistical_default_change: false,
                reasons: vec![scope_reason.into()],
                not_evaluable_reasons: vec![scope_reason.into()],
                not_eligible_reasons: Vec::new(),
                calibrated_tdc_improvements: 0,
            },
            transfer_stability: Vec::new(),
            pending_validation_gates: vec![scope_reason.into()],
            ensemble_interaction_calibration: None,
            resource_preflight,
            planned_models: planned_model_reports(&manifest),
            parameter_optimization: optimization_results,
            parameter_optimizer_execution: Some(execution),
        };
        write_json_atomic(&manifest.output_root.join("workflow.state.json"), &state)?;
        return Ok(state);
    }

    let selected_calibration_stages = completed_experts
        .iter()
        .map(|expert| {
            (
                model_slug(&expert.model).to_owned(),
                expert.calibration_stage.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut runs = manifest.validation.additional_runs.clone();
    runs.extend(stages.iter().filter(|stage| stage.evaluable).map(|stage| {
        ValidationRun {
            method: stage.model.clone(),
            stage: stage.stage.clone(),
            results: stage.results.clone(),
            mode: ValidationMode::DecoyFree,
            expected_search_space: Some(if is_target_only_stage(&stage.stage) {
                "No Ent".into()
            } else {
                "+Ent".into()
            }),
            calibration_stage: is_target_only_stage(&stage.stage)
                .then(|| {
                    stage
                        .window_provenance
                        .as_ref()
                        .map(|provenance| provenance.source_stage.clone())
                        .or_else(|| selected_calibration_stages.get(&stage.model).cloned())
                })
                .flatten(),
            target_only_calibration_policy: stage.target_only_calibration_policy,
            release_candidate: stage.release_candidate,
        }
    }));
    let mut validation = Vec::new();
    let mut invalid_runs = Vec::new();
    let missing_runs = runs
        .iter()
        .filter(|run| !run.results.is_file())
        .cloned()
        .collect::<Vec<_>>();
    for run in &runs {
        if !run.results.is_file() {
            continue;
        }
        match summarize_run(
            run,
            &manifest.validation.effective_ratios,
            manifest.validation.fdr_threshold,
        ) {
            Ok(rows) => validation.extend(rows),
            Err(error) => invalid_runs.push(InvalidValidationRun {
                run: run.clone(),
                error: format!("{error:#}"),
            }),
        }
    }
    let stability = transfer_stability(
        &validation,
        manifest.validation.maximum_transfer_fraction_loss,
    );
    let comparisons = stage_comparisons(&validation);
    let expert_gate_layer = match manifest.validation.null_window_validation_scope {
        NullWindowValidationScope::RawQ => "raw_q",
        NullWindowValidationScope::Level4 => "level4",
    };
    let ensemble_gates = expert_quality_gates(
        &validation,
        &stability,
        manifest.validation.fdr_threshold,
        manifest
            .validation
            .minimum_entrapment_peptides_for_stable_estimate,
        expert_gate_layer,
    );
    let local_parity = parity_comparisons(
        &validation,
        &manifest.validation.parity_pairs,
        manifest.validation.maximum_parity_fraction_difference,
    );
    let mut parity = local_parity.clone();
    if let Some(path) = manifest.validation.external_parity_evidence.as_ref() {
        let external: Vec<ParityComparison> = serde_json::from_slice(
            &std::fs::read(path)
                .with_context(|| format!("failed to read parity evidence {}", path.display()))?,
        )
        .with_context(|| format!("invalid parity evidence {}", path.display()))?;
        anyhow::ensure!(
            !external.is_empty(),
            "external parity evidence is empty: {}",
            path.display()
        );
        parity.extend(external);
    }
    let tdc_benchmarks = tdc_benchmark_comparisons(
        &validation,
        manifest.validation.tdc_reference_method.as_deref(),
        manifest.validation.fdr_threshold,
    );
    let mut not_evaluable_reasons = Vec::new();
    let mut not_eligible_reasons = Vec::new();
    if manifest.validation.dataset_role != ValidationDatasetRole::Holdout {
        not_eligible_reasons.push("dataset is development, not holdout".into());
    }
    if manifest.validation.diagnostic_only
        || manifest.artifact_reuse_policy == ArtifactReusePolicy::CrossDatasetDiagnostic
    {
        not_eligible_reasons
            .push("workflow is diagnostic-only and cannot be release evidence".into());
    }
    if manifest.validation.parity_pairs.is_empty() {
        not_evaluable_reasons
            .push("dataset-local baseline/native parity comparison is missing".into());
    } else {
        not_evaluable_reasons.extend(missing_parity_evidence(
            &manifest.validation.parity_pairs,
            &local_parity,
        ));
        if local_parity
            .iter()
            .any(|comparison| !comparison.within_tolerance)
        {
            not_eligible_reasons
                .push("one or more dataset-local parity comparisons exceed tolerance".into());
        }
    }
    if !missing_runs.is_empty() {
        not_evaluable_reasons.push("required validation runs are missing".into());
    }
    if !invalid_runs.is_empty() {
        not_evaluable_reasons.push("one or more validation runs are invalid or unreadable".into());
    }
    if baseline
        .as_ref()
        .is_some_and(|baseline| baseline.status != "complete" || baseline.files.is_empty())
    {
        not_evaluable_reasons.push("the frozen baseline is incomplete or empty".into());
    }
    let native_methods = manifest
        .models
        .iter()
        .filter(|model| model.enabled && model.model != ModelFit::Ensemble)
        .map(|model| model_slug(&model.model))
        .collect::<BTreeSet<_>>();
    if stability.iter().any(|comparison| {
        native_methods.contains(comparison.method.as_str())
            && comparison.release_candidate
            && !comparison.stable
    }) {
        not_eligible_reasons.push("one or more search-space transfers are unstable".into());
    }
    let release_target_rows = validation
        .iter()
        .filter(|row| {
            matches!(row.mode, ValidationMode::DecoyFree)
                && row.release_candidate
                && is_target_only_stage(&row.stage)
                && row.layer == expert_gate_layer
                && native_methods.contains(row.method.as_str())
        })
        .collect::<Vec<_>>();
    for target in &release_target_rows {
        if target.calibration_stage.is_none() {
            not_evaluable_reasons.push(format!(
                "{} / {} has no target-only calibration provenance",
                target.method, target.stage
            ));
        }
        if !stability.iter().any(|comparison| {
            comparison.method == target.method
                && comparison.to_stage == target.stage
                && comparison.release_candidate
        }) {
            not_evaluable_reasons.push(format!(
                "{} / {} has no evaluable target-only transfer comparison",
                target.method, target.stage
            ));
        }
        match ensemble_gates
            .iter()
            .find(|gate| gate.model == target.method)
        {
            None => not_evaluable_reasons.push(format!(
                "{} has no calibration quality-gate result",
                target.method
            )),
            Some(gate) => {
                for reason in &gate.reasons {
                    if reason.contains("missing") {
                        not_evaluable_reasons.push(format!("{}: {reason}", target.method));
                    } else {
                        not_eligible_reasons.push(format!("{}: {reason}", target.method));
                    }
                }
            }
        }
    }
    if manifest.validation.tdc_reference_method.is_none() || tdc_benchmarks.is_empty() {
        not_evaluable_reasons.push("a matched TDC benchmark is missing".into());
    } else {
        for target in &release_target_rows {
            let comparison = tdc_benchmarks.iter().find(|comparison| {
                comparison.decoy_free_method == target.method
                    && comparison.stage == target.stage
                    && comparison.layer == target.layer
                    && comparison.release_candidate
            });
            match comparison {
                None => not_evaluable_reasons.push(format!(
                    "{} / {} has no matched TDC comparison",
                    target.method, target.stage
                )),
                Some(comparison) if !comparison.calibration_constrained => {
                    if comparison.peptide_entrapment_fdp.is_none() {
                        not_evaluable_reasons.push(format!(
                            "{} / {} has no evaluable entrapment calibration for its TDC comparison",
                            target.method, target.stage
                        ));
                    } else {
                        not_eligible_reasons.push(format!(
                            "{} / {} exceeds the entrapment calibration threshold",
                            target.method, target.stage
                        ));
                    }
                }
                Some(_) => {}
            }
        }
    }
    if manifest.validation.tdc_reference_method.is_some()
        && !tdc_benchmarks.is_empty()
        && !tdc_benchmarks.iter().any(|comparison| {
            native_methods.contains(comparison.decoy_free_method.as_str())
                && comparison.release_candidate
                && comparison.improves_peptide_yield
        })
    {
        not_eligible_reasons.push(
            "no calibrated Decoy-Free result improves peptide yield over the matched TDC".into(),
        );
    }
    // Expert-quality and post-assembly interaction results are validation
    // reports only. They must not retroactively change a JSON-requested,
    // technically valid Ensemble roster or suppress its target-only stage.
    let calibrated_tdc_improvements = tdc_benchmarks
        .iter()
        .filter(|comparison| {
            native_methods.contains(comparison.decoy_free_method.as_str())
                && comparison.release_candidate
                && comparison.improves_peptide_yield
        })
        .count();
    not_evaluable_reasons.sort();
    not_evaluable_reasons.dedup();
    not_eligible_reasons.sort();
    not_eligible_reasons.dedup();
    let status = if !not_evaluable_reasons.is_empty() {
        ReleaseGateStatus::NotEvaluable
    } else if !not_eligible_reasons.is_empty() {
        ReleaseGateStatus::NotEligible
    } else {
        ReleaseGateStatus::Eligible
    };
    let reasons = not_evaluable_reasons
        .iter()
        .chain(&not_eligible_reasons)
        .cloned()
        .collect();
    let release_gate = ReleaseGate {
        status,
        eligible_for_statistical_default_change: status == ReleaseGateStatus::Eligible,
        reasons,
        not_evaluable_reasons,
        not_eligible_reasons,
        calibrated_tdc_improvements,
    };
    write_json_atomic(
        &manifest.output_root.join("validation.summary.json"),
        &validation,
    )?;
    write_json_atomic(
        &manifest
            .output_root
            .join("validation.transfer_stability.json"),
        &stability,
    )?;
    write_json_atomic(
        &manifest
            .output_root
            .join("validation.stage_comparisons.json"),
        &comparisons,
    )?;
    write_json_atomic(
        &manifest.output_root.join("validation.missing_runs.json"),
        &missing_runs,
    )?;
    write_json_atomic(
        &manifest.output_root.join("validation.invalid_runs.json"),
        &invalid_runs,
    )?;
    write_json_atomic(
        &manifest
            .output_root
            .join("validation.ensemble_expert_gates.json"),
        &serde_json::json!({
            "schema_version": 2,
            "participation_effect": nonblocking_ensemble_gate_effect(),
            "diagnostics": ensemble_gates,
        }),
    )?;
    write_json_atomic(
        &manifest.output_root.join("validation.parity.json"),
        &parity,
    )?;
    write_json_atomic(
        &manifest.output_root.join("validation.tdc_benchmarks.json"),
        &tdc_benchmarks,
    )?;
    write_json_atomic(
        &manifest.output_root.join("validation.release_gate.json"),
        &release_gate,
    )?;
    write_json_atomic(
        &manifest
            .output_root
            .join("validation.ensemble_diagnostic_contract.json"),
        &serde_json::json!({
            "schema_version": 1,
            "participation_effect": "none_nonblocking_diagnostic",
            "reported_only": [
                "entrapment_fdp",
                "minimum_entrapment_observations",
                "target_only_transfer_loss",
                "parity",
                "unique_or_incremental_identifications",
                "interaction_calibration",
                "holdout_outcome",
                "release_or_statistical_default_eligibility"
            ]
        }),
    )?;

    let pending_validation_gates = if release_gate.eligible_for_statistical_default_change {
        Vec::new()
    } else {
        release_gate.reasons.clone()
    };
    let candidate_pools = stages
        .iter()
        .filter_map(|stage| stage.candidate_pool.clone())
        .collect::<Vec<_>>();
    write_json_atomic(
        &manifest.output_root.join("workflow.candidate_pools.json"),
        &candidate_pools,
    )?;
    let ms2rescore_annotation_caches = stages
        .iter()
        .filter_map(|stage| stage.ms2rescore_annotation_cache.clone())
        .collect::<Vec<_>>();
    write_json_atomic(
        &manifest
            .output_root
            .join("workflow.ms2rescore_annotations.json"),
        &ms2rescore_annotation_caches,
    )?;
    let parameter_optimizer_execution = manifest.parameter_optimizer.as_ref().map(|config| {
        optimizer_execution_report(
            config,
            &optimization_results,
            "post_selection_workflow_complete",
        )
    });
    let state = WorkflowState {
        schema_version: 2,
        manifest_hash,
        dataset,
        entrapment,
        entrapment_fasta_parity,
        baseline,
        stages,
        candidate_pools,
        ms2rescore_annotation_caches,
        validation,
        missing_runs,
        invalid_runs,
        stage_comparisons: comparisons,
        ensemble_expert_gates: ensemble_gates,
        ensemble_expert_gates_participation_effect: nonblocking_ensemble_gate_effect(),
        parity_comparisons: parity,
        tdc_benchmarks,
        release_gate,
        transfer_stability: stability,
        pending_validation_gates,
        ensemble_interaction_calibration: ensemble_interaction_report,
        resource_preflight,
        planned_models: planned_model_reports(&manifest),
        parameter_optimization: optimization_results,
        parameter_optimizer_execution,
    };
    write_json_atomic(&manifest.output_root.join("workflow.state.json"), &state)?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sage_core::decoy_free_fdr::{DfRunArtifacts, FrozenModelMetadata};
    use sage_core::input::{
        ExternalEmpiricalFeatureProfile, ExternalMs2RescoreProfiles, ExternalProfileCalibration,
        ExternalProfileWindowProvenance, FrozenGumbelParameters,
    };
    use sage_core::ml::msfdr::{Msfdr2SmixModel, MsfdrSeededModel};
    use sage_core::ml::skew_normal::SkewNormal;

    fn lower_order_artifacts(mu: f64) -> DfRunArtifacts {
        DfRunArtifacts {
            lower_order: Some(sage_core::ml::lower_order::LowerOrderArtifact {
                schema_version: 1,
                model_version: "sage-lower-order-local-lom-extrapolated-v1".into(),
                params_by_charge: vec![sage_core::ml::lower_order::LowerOrderChargeParameters {
                    charge: 2,
                    mu,
                    beta: 0.5,
                }],
                charge_fill_mode: sage_core::ml::lower_order::ChargeFillMode::MinimalDelta,
                fitted_charges_sorted: vec![2],
                max_fitted_charge: 2,
                null_rank_min: 6,
                null_rank_max: 9,
                evalue_candidate_count_power: 1.0,
                evalue_scale: 1.0,
                tev_transform: "NegLogE".into(),
                extrapolation_strength: 0.0,
                reference_candidate_counts: vec![10, 20],
            }),
            ..DfRunArtifacts::default()
        }
    }

    fn test_optimizer_config() -> ParameterOptimizerConfig {
        ParameterOptimizerConfig {
            schema_version: 1,
            enabled: true,
            classification: crate::parameter_optimizer::OptimizationClassification::DevelopmentOnly,
            selected_experts: vec![OptimizerExpert::Moments],
            compiled_defaults: BTreeMap::new(),
            workflow_defaults: BTreeMap::new(),
            fixed_baseline_values: BTreeMap::new(),
            seed: 42,
            maximum_trial_budget: 2,
            maximum_optimization_passes: 1,
            objective: crate::parameter_optimizer::default_objective(),
            fixed_evaluation_threshold: 0.01,
            empirical_entrapment_constraints: Vec::new(),
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
                scope: crate::parameter_optimizer::ParameterScope::PerExpert,
                expert: Some(OptimizerExpert::Moments),
                strategy: crate::parameter_optimizer::OptimizerStrategy::ExhaustiveGrid,
                structural_comparison: false,
                fixed: BTreeMap::new(),
                space: BTreeMap::from([(
                    "moments_purification_factor".into(),
                    vec![ParameterValue::Float(0.1), ParameterValue::Float(0.2)],
                )]),
                window_search: Some(OptimizerWindowSearch {
                    strategy: "explicit_grid".into(),
                    min_rank_range: [2, 3],
                    max_rank_range: [3, 4],
                }),
                use_external_features: true,
                max_trials: Some(2),
                max_passes: None,
            }],
        }
    }

    #[test]
    fn workflow_optimizer_requires_strict_reuse_and_exact_json_roster() {
        let directory = test_directory("workflow-parameter-optimizer-contract");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.parameter_optimizer = Some(test_optimizer_config());
        assert!(manifest
            .validate()
            .unwrap_err()
            .to_string()
            .contains("read-only existing candidate pools"));
        manifest.require_existing_candidate_pool = true;
        manifest.require_existing_annotation_cache = true;
        manifest.validate().unwrap();
        manifest
            .parameter_optimizer
            .as_mut()
            .unwrap()
            .selected_experts
            .push(OptimizerExpert::Mle);
        assert!(manifest
            .validate()
            .unwrap_err()
            .to_string()
            .contains("exactly match"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn model_local_optimizer_windows_are_bounded_and_deterministic() {
        let mut model = ModelWorkflow {
            model: ModelFit::Moments,
            window: Some(NullWindow {
                min_rank: 9,
                max_rank: 18,
            }),
            candidate_windows: Vec::new(),
            window_optimizer: None,
            enabled: true,
            ms2rescore: Ms2RescorePolicy::Never,
            maximum_raw_fdp_increase: None,
            minimum_level4_peptide_gain: None,
            target_only_calibration_policy: None,
            ensemble_participation: EnsembleParticipation::Auto,
            ensemble_exclusion_reason: None,
            ensemble_interaction_baseline: true,
        };
        let search = OptimizerWindowSearch {
            strategy: "explicit_grid".into(),
            min_rank_range: [2, 3],
            max_rank_range: [3, 4],
        };
        apply_optimizer_window(&mut model, &search).unwrap();
        let windows = model
            .candidate_windows
            .iter()
            .map(|window| (window.min_rank, window.max_rank))
            .collect::<Vec<_>>();
        assert_eq!(windows, vec![(2, 3), (2, 4), (3, 3), (3, 4)]);
        assert!(model.window.is_none());
    }
    use sage_core::ml::nokoi::{
        fit_nokoi_artifact_with_metadata, LogisticRegression, NokoiArtifact,
        NokoiArtifactApplicationMode, NokoiCalibrationPoint, NokoiConfig, NokoiFitMetadata,
        NokoiNormalization, NOKOI_FEATURE_SCHEMA,
    };
    use sage_core::scoring::FeatureCore;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_directory(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "sage-workflow-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn portable_nokoi_artifacts() -> DfRunArtifacts {
        let mut features = Vec::new();
        let mut stable_ids = Vec::new();
        for index in 0..300 {
            let rank = if index < 75 {
                1
            } else {
                2 + (index % 3) as u32
            };
            features.push(
                FeatureCore {
                    spec_id: format!("scan={index}"),
                    file_id: index % 9,
                    rank,
                    label: 1,
                    expmass: 500.0 + index as f32 / 100.0,
                    charge: 2 + (index % 3) as u8,
                    peptide_len: 8 + index % 10,
                    hyperscore: if rank == 1 {
                        80.0
                    } else {
                        8.0 + (index % 37) as f64
                    },
                    delta_next: (index % 29) as f64 / 10.0,
                    matched_peaks: 5 + (index % 20) as u32,
                    matched_intensity_pct: 0.1 + (index % 80) as f32 / 100.0,
                    longest_y_pct: 0.1 + (index % 70) as f32 / 100.0,
                    ms2_intensity: 1000.0 + index as f32,
                    lo_spectrum_candidate_count: 100 + (index % 20) as u32,
                    ..FeatureCore::default()
                }
                .to_df(),
            );
            stable_ids.push(format!("workflow-nokoi-{index:04}"));
        }
        let null_indices = features
            .iter()
            .enumerate()
            .filter_map(|(index, feature)| (feature.core.rank > 1).then_some(index))
            .collect::<Vec<_>>();
        let config = NokoiConfig {
            enabled: true,
            epochs: 30,
            patience: 4,
            l1_lambda_steps: 3,
            ..NokoiConfig::default()
        };
        let fitted = fit_nokoi_artifact_with_metadata(
            &features,
            &config,
            2,
            4,
            5,
            |feature| feature.core.rank == 1,
            &null_indices,
            NokoiFitMetadata {
                stable_ids: &stable_ids,
                positive_class_rule: "synthetic rank-1 positive",
                positive_top_fraction: 0.25,
                positive_threshold: 50.0,
                null_purification_rule: "synthetic ranks 2-4",
                null_purification_factor: 0.25,
            },
        )
        .unwrap();
        DfRunArtifacts {
            nokoi: Some(fitted.artifact),
            ..DfRunArtifacts::default()
        }
    }

    fn peptide(mut index: usize) -> String {
        const ALPHABET: &[u8] = b"ACDEFGHKMNPQRSTVWY";
        let mut value = String::from("PEP");
        for _ in 0..5 {
            value.push(ALPHABET[index % ALPHABET.len()] as char);
            index /= ALPHABET.len();
        }
        value
    }

    fn external_profile(name: &str) -> ExternalEmpiricalFeatureProfile {
        ExternalEmpiricalFeatureProfile {
            name: name.into(),
            enabled: false,
            higher_is_better: true,
            good_median: Some(1.0),
            null_median: Some(0.0),
            separation: Some(1.0),
            auc: Some(0.75),
            good_n: 100,
            null_n: 1000,
        }
    }

    fn external_profiles() -> ExternalMs2RescoreProfiles {
        ExternalMs2RescoreProfiles {
            schema_version: 2,
            model_version: "sage-external-ms2rescore-profiles-v2-explicit-window".into(),
            calibration: ExternalProfileCalibration {
                min_null_rank: 9,
                max_null_rank: 18,
                provenance: ExternalProfileWindowProvenance::ExplicitConfiguration,
            },
            ms2pip_pcc: external_profile("ms2pip_pcc"),
            spectral_angle: external_profile("spectral_angle"),
            fragment_intensity_agreement: external_profile("fragment_intensity_agreement"),
            deeplc_abs_rt_error: external_profile("deeplc_abs_rt_error"),
            ccs_pct_error: external_profile("ccs_pct_error"),
            ccs_abs_error: external_profile("ccs_abs_error"),
        }
    }

    fn minimal_manifest(directory: &Path, role: ValidationDatasetRole) -> WorkflowManifest {
        let search_config = directory.join("search.json");
        let target_fasta = directory.join("target.fasta");
        std::fs::write(&search_config, b"{}\n").unwrap();
        std::fs::write(&target_fasta, b">target\nPEPTIDER\n").unwrap();
        std::fs::write(directory.join("foreign.fasta"), b">foreign\nDIFFERENTK\n").unwrap();
        WorkflowManifest {
            schema_version: 1,
            name: "minimal".into(),
            dataset_id: Some("minimal-dataset".into()),
            search_config,
            target_fasta,
            spectra: vec!["unresolved-test.mzML".into()],
            output_root: directory.join("output"),
            candidate_pool_root: None,
            annotation_cache_root: None,
            target_only_annotation_cache_root: None,
            entrapment: EntrapmentWorkflow {
                database_mode: EntrapmentDatabaseMode::NativeGenerated,
                foreign_fastas: vec![directory.join("foreign.fasta")],
                output_fasta: directory.join("entrapment.fasta"),
                frozen_legacy_fasta: None,
                foreign_source_mode: ForeignSourceMode::Automatic,
                shared_peptide_exclusion_mode: SharedPeptideExclusionMode::SageSearchSpace,
                selected_foreign_fasta: None,
                legacy_parity_reference: None,
                seed: 1,
                protein_fold: 1,
            },
            models: vec![ModelWorkflow {
                model: ModelFit::Moments,
                window: None,
                candidate_windows: vec![NullWindow {
                    min_rank: 2,
                    max_rank: 8,
                }],
                window_optimizer: None,
                enabled: true,
                ms2rescore: Ms2RescorePolicy::Measure,
                maximum_raw_fdp_increase: None,
                minimum_level4_peptide_gain: None,
                target_only_calibration_policy: None,
                ensemble_participation: EnsembleParticipation::Auto,
                ensemble_exclusion_reason: None,
                ensemble_interaction_baseline: true,
            }],
            baseline: None,
            validation: ValidationWorkflow {
                effective_ratios: EffectiveRatios::default(),
                null_window_validation_scope: NullWindowValidationScope::Level4,
                use_generated_entrapment_ratios: true,
                fdr_threshold: 0.01,
                maximum_transfer_fraction_loss: 0.20,
                additional_runs: Vec::new(),
                minimum_incremental_ensemble_peptides: 1,
                minimum_ensemble_experts: 2,
                minimum_entrapment_peptides_for_stable_estimate: 3,
                parity_pairs: Vec::new(),
                external_parity_evidence: None,
                maximum_parity_fraction_difference: 0.001,
                tdc_reference_method: None,
                dataset_role: role,
                diagnostic_only: false,
            },
            resume: true,
            require_existing_candidate_pool: false,
            require_existing_annotation_cache: false,
            migrate_schema_v2_annotation_cache_only: false,
            annotate_target_matches: false,
            ensemble_lock: None,
            locked_expert_artifacts: BTreeMap::new(),
            artifact_reuse_policy: ArtifactReusePolicy::DatasetLocalOnly,
            target_only_calibration_policy: TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
            parameter_optimizer: None,
        }
    }

    #[test]
    fn holdout_runs_its_own_declared_optimizer() {
        let directory = test_directory("holdout-local-optimizer");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Holdout);
        manifest.models.push(ModelWorkflow {
            model: ModelFit::Ensemble,
            window: None,
            candidate_windows: Vec::new(),
            window_optimizer: None,
            enabled: true,
            ms2rescore: Ms2RescorePolicy::Measure,
            maximum_raw_fdp_increase: None,
            minimum_level4_peptide_gain: None,
            target_only_calibration_policy: None,
            ensemble_participation: EnsembleParticipation::Auto,
            ensemble_exclusion_reason: None,
            ensemble_interaction_baseline: true,
        });
        manifest.validate().unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn duplicate_canonical_models_fail_manifest_validation() {
        let directory = test_directory("duplicate-canonical-workflow-model");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.models.push(manifest.models[0].clone());
        let error = manifest.validate().unwrap_err().to_string();
        assert!(error.contains("duplicate canonical model moments"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn omitted_annotation_cache_requirement_defaults_false() {
        let directory = test_directory("annotation-cache-requirement-default");
        let manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        let mut value = serde_json::to_value(&manifest).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("require_existing_annotation_cache");
        let restored: WorkflowManifest = serde_json::from_value(value).unwrap();
        assert!(!restored.require_existing_annotation_cache);
        assert!(!restored.migrate_schema_v2_annotation_cache_only);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn annotation_cache_migration_is_explicit_and_incompatible_with_strict_mode() {
        let directory = test_directory("annotation-cache-migration-contract");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.migrate_schema_v2_annotation_cache_only = true;
        let error = manifest.validate().unwrap_err().to_string();
        assert!(error.contains("require_existing_candidate_pool=true"));

        manifest.require_existing_candidate_pool = true;
        manifest.require_existing_annotation_cache = true;
        let error = manifest.validate().unwrap_err().to_string();
        assert!(error.contains("mutually exclusive"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn strict_plan_preflights_both_spaces_without_creating_output() {
        let directory = test_directory("strict-resource-plan");
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source_search_config = workspace.join("tests/config.json");
        let search_config = directory.join("search.config.json");
        let mut search_value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&source_search_config).unwrap()).unwrap();
        search_value["external_features"] = serde_json::json!({"deeplc_calibration_set_size": 10});
        write_json_atomic(&search_config, &search_value).unwrap();
        let spectrum = workspace.join("tests/LQSRPAAPPAPGPGQLTLR.mzML");
        let fasta = workspace.join("tests/Q99536.fasta");
        let candidate_root = directory.join("immutable-candidates");
        let annotation_root = directory.join("immutable-entrapment-annotations");
        let target_annotation_root = directory.join("immutable-target-annotations");

        let mut input = Input::load(search_config.to_string_lossy().as_ref()).unwrap();
        input.database.fasta = Some(fasta.display().to_string());
        input.mzml_paths = Some(vec![spectrum.display().to_string()]);
        input.output_directory = Some(directory.join("unused").display().to_string());
        let fdr = input.fdr.get_or_insert_with(FdrOptions::default);
        fdr.mode = Some(FdrMode::DecoyFree);
        fdr.model_fit = Some(ModelFit::Moments);
        let runner = Runner::new(input.build().unwrap(), 1).unwrap();
        let search = search_fingerprint(&runner.parameters).unwrap();
        let mut candidate = sage_core::scoring::FeatureCore::default();
        candidate.peptide_idx = sage_core::database::PeptideIx(0);
        candidate.spec_id = "synthetic-scan".into();
        candidate.rank = 1;
        candidate.charge = 2;
        let database = runner.shared_database();
        let pool_directory = crate::candidate_pool::pool_directory(&candidate_root, &search);
        crate::candidate_pool::write_pool(
            &pool_directory,
            &search,
            &[candidate.clone()],
            &database,
        )
        .unwrap();
        let stable_id = stable_candidate_id(
            &search.digest,
            &candidate,
            &database[candidate.peptide_idx].to_string(),
        );
        let settings = runner.parameters.external_features.clone();
        let annotation_input = crate::external_feature_cache::ExternalAnnotationInput {
            stable_id: stable_id.clone(),
            score: candidate.hyperscore,
            q_value: Some(0.01),
            pep: Some(0.02),
            retention_time: candidate.rt,
            ion_mobility: candidate.ims,
            precursor_mass: candidate.expmass,
            charge: candidate.charge,
            rank: candidate.rank,
        };
        for root in [&annotation_root, &target_annotation_root] {
            let identity = crate::external_feature_cache::raw_prediction_identity_with_probe_root(
                &search.digest,
                &settings,
                std::slice::from_ref(&annotation_input),
                runner.parameters.report_psms as u32,
                root,
                false,
            )
            .unwrap();
            let cache_directory =
                crate::external_feature_cache::raw_cache_directory(root, &identity);
            let features = sage_core::scoring::ExternalPsmFeatures {
                ms2rescore_ms2pip_pcc: 0.8,
                ms2rescore_spectral_angle: 0.7,
                ms2rescore_fragment_intensity_agreement: 0.6,
                ms2rescore_deeplc_predicted_rt: 12.5,
                ms2rescore_deeplc_calibrated_rt: 12.0,
                ms2rescore_deeplc_rt_error: 0.5,
                ms2rescore_deeplc_abs_rt_error: 0.5,
                tims2rescore_observed_ion_mobility: 0.0,
                ms2rescore_feature_joined: true,
                ..sage_core::scoring::ExternalPsmFeatures::default()
            };
            crate::external_feature_cache::write_raw_cache(
                &cache_directory,
                &identity,
                vec![crate::external_feature_cache::ExternalAnnotationRecord {
                    stable_id: stable_id.clone(),
                    features,
                }],
                None,
            )
            .unwrap();
        }

        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.search_config = search_config;
        manifest.target_fasta = fasta.clone();
        manifest.spectra = vec![spectrum.display().to_string()];
        manifest.output_root = directory.join("strict-output-must-not-exist");
        manifest.candidate_pool_root = Some(candidate_root);
        manifest.annotation_cache_root = Some(annotation_root.clone());
        manifest.target_only_annotation_cache_root = Some(target_annotation_root.clone());
        manifest.entrapment.database_mode = EntrapmentDatabaseMode::FrozenLegacy;
        manifest.entrapment.foreign_fastas.clear();
        manifest.entrapment.frozen_legacy_fasta = Some(fasta);
        manifest.models[0].ms2rescore = Ms2RescorePolicy::Always;
        manifest.require_existing_candidate_pool = true;
        manifest.require_existing_annotation_cache = true;
        let manifest_path = directory.join("strict-workflow.json");
        write_json_atomic(&manifest_path, &manifest).unwrap();

        let state = execute_workflow(&manifest_path, &directory, 1, true).unwrap();
        assert_eq!(state.planned_models.len(), 1);
        assert_eq!(state.planned_models[0].model, "moments");
        assert_eq!(
            state.planned_models[0].window_mode,
            "dataset_local_explicit_grid"
        );
        assert_eq!(state.resource_preflight.len(), 6);
        let pools = state
            .resource_preflight
            .iter()
            .filter(|resource| resource.resource_type == "candidate_pool")
            .collect::<Vec<_>>();
        assert_eq!(pools.len(), 2);
        assert!(pools.iter().all(|resource| {
            resource.status == "validated_exact"
                && resource.valid
                && resource.reused
                && !resource.generation_allowed
        }));
        let raw_predictions = state
            .resource_preflight
            .iter()
            .filter(|resource| resource.resource_type == "raw_external_prediction_cache")
            .collect::<Vec<_>>();
        assert_eq!(raw_predictions.len(), 2);
        assert!(raw_predictions.iter().all(|resource| {
            resource.status == "validated_exact"
                && !resource.expected_fingerprint.is_empty()
                && resource.expected_fingerprint == resource.actual_fingerprint
                && resource.valid
                && resource.reused
                && !resource.generation_allowed
        }));
        let calibrations = state
            .resource_preflight
            .iter()
            .filter(|resource| resource.resource_type == "stage_external_calibration")
            .collect::<Vec<_>>();
        assert_eq!(calibrations.len(), 2);
        assert!(calibrations.iter().all(|resource| {
            resource.status == "deferred_until_calibration"
                && resource.expected_fingerprint.is_empty()
                && resource.actual_fingerprint.is_empty()
                && !resource.valid
                && !resource.reused
                && !resource.generation_allowed
                && resource.catalog_fingerprints.len() == 1
        }));
        assert!(!manifest.output_root.exists());

        std::fs::remove_dir_all(&target_annotation_root).unwrap();
        let error = execute_workflow(&manifest_path, &directory, 1, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("target_only"));
        assert!(error.contains("generation_prohibited=true"));
        assert!(!manifest.output_root.exists());

        let poison_target_root = directory.join("poison-target-only-cache");
        std::fs::create_dir_all(&poison_target_root).unwrap();
        std::fs::write(
            poison_target_root.join("manifest.json"),
            b"corrupt target-only cache that must never be opened\n",
        )
        .unwrap();
        manifest.target_only_annotation_cache_root = Some(poison_target_root.clone());
        let mut optimizer = test_optimizer_config();
        optimizer.schema_version = crate::parameter_optimizer::PARAMETER_OPTIMIZER_SCHEMA_VERSION;
        optimizer.execution_mode = OptimizerExecutionMode::OptimizationOnly;
        manifest.parameter_optimizer = Some(optimizer);
        write_json_atomic(&manifest_path, &manifest).unwrap();
        let state = execute_workflow(&manifest_path, &directory, 1, true).unwrap();
        assert_eq!(state.resource_preflight.len(), 3);
        assert!(state
            .resource_preflight
            .iter()
            .all(|resource| resource.search_space == "+entrapment"));
        assert!(state
            .resource_preflight
            .iter()
            .all(|resource| !resource.requested_path.starts_with(&poison_target_root)));
        assert_eq!(
            state
                .parameter_optimizer_execution
                .as_ref()
                .unwrap()
                .execution_mode,
            OptimizerExecutionMode::OptimizationOnly
        );
        assert!(!manifest.output_root.exists());

        std::fs::remove_dir_all(&annotation_root).unwrap();
        let error = execute_workflow(&manifest_path, &directory, 1, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("+entrapment"));
        assert!(error.contains("generation_prohibited=true"));
        assert!(!manifest.output_root.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn unavailable_interaction_diagnostic_has_no_participation_effect() {
        let report = unavailable_interaction_report(
            vec!["moments".into()],
            vec!["moments".into(), "mle".into()],
            "missing validation table",
        );
        assert!(!report.evaluable);
        assert!(!report.final_level4_calibration_pass);
        assert_eq!(report.participation_effect, "none_nonblocking_diagnostic");
        assert_eq!(report.final_experts, vec!["moments", "mle"]);
    }

    #[test]
    fn compact_adaptive_window_optimizer_is_valid_and_serializable() {
        let directory = test_directory("compact-adaptive-window-optimizer");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.models[0].candidate_windows.clear();
        manifest.models[0].window_optimizer = Some(WindowOptimizerWorkflow {
            strategy: NullWindowSearchStrategy::Adaptive,
            min_rank_range: [2, 10],
            max_rank_range: [2, 25],
            adaptive: AdaptiveNullWindowSearchOptions::default(),
        });
        manifest.validate().unwrap();
        let value = serde_json::to_value(&manifest.models[0]).unwrap();
        assert_eq!(value["window_optimizer"]["strategy"], "adaptive");
        assert_eq!(
            value["window_optimizer"]["min_rank_range"],
            serde_json::json!([2, 10])
        );
        assert_eq!(
            value["window_optimizer"]["max_rank_range"],
            serde_json::json!([2, 25])
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compact_landscape_adaptive_window_optimizer_is_valid_and_serializable() {
        let directory = test_directory("compact-landscape-adaptive-window-optimizer");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.models[0].candidate_windows.clear();
        manifest.models[0].window_optimizer = Some(WindowOptimizerWorkflow {
            strategy: NullWindowSearchStrategy::LandscapeAdaptive,
            min_rank_range: [2, 10],
            max_rank_range: [2, 25],
            adaptive: AdaptiveNullWindowSearchOptions::default(),
        });
        manifest.validate().unwrap();
        let value = serde_json::to_value(&manifest.models[0]).unwrap();
        assert_eq!(value["window_optimizer"]["strategy"], "landscape_adaptive");
        assert_eq!(
            value["window_optimizer"]["adaptive"]["landscape_coarse_row_count"],
            5
        );
        assert_eq!(
            value["window_optimizer"]["adaptive"]["landscape_coarse_offsets"],
            serde_json::json!([0, 8])
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn window_optimizer_is_mutually_exclusive_and_rejects_rank_one() {
        let directory = test_directory("window-optimizer-validation");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.models[0].window_optimizer = Some(WindowOptimizerWorkflow {
            strategy: NullWindowSearchStrategy::Exhaustive,
            min_rank_range: [2, 10],
            max_rank_range: [2, 25],
            adaptive: AdaptiveNullWindowSearchOptions::default(),
        });
        assert!(manifest.validate().is_err());
        manifest.models[0].candidate_windows.clear();
        manifest.models[0]
            .window_optimizer
            .as_mut()
            .unwrap()
            .min_rank_range = [1, 10];
        assert!(manifest.validate().is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn explicit_ensemble_exclusion_requires_a_reason() {
        let directory = test_directory("ensemble-exclusion-reason");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        let legacy_compatible = serde_json::to_value(&manifest.models[0]).unwrap();
        assert!(legacy_compatible.get("ensemble_participation").is_none());
        assert!(legacy_compatible.get("ensemble_exclusion_reason").is_none());
        manifest.models[0].ensemble_participation = EnsembleParticipation::Excluded;
        assert!(manifest.validate().is_err());
        manifest.models[0].ensemble_exclusion_reason = Some("parity evidence is incomplete".into());
        manifest.validate().unwrap();
        let excluded = serde_json::to_value(&manifest.models[0]).unwrap();
        assert_eq!(excluded["ensemble_participation"], "excluded");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn interrupted_and_modified_stage_checkpoints_are_not_resumed() {
        let directory = test_directory("stage-resume-integrity");
        let results = directory.join("results.sage.tsv");
        let config = directory.join("workflow.search.resolved.json");
        std::fs::write(&results, b"results\n").unwrap();
        std::fs::write(&config, b"{}\n").unwrap();
        let dataset = DatasetIdentity {
            schema_version: 1,
            dataset_id: "dataset".into(),
            fingerprint: "dataset-fingerprint".into(),
            target_fasta_sha256: "fasta".into(),
            spectra_sha256: vec!["spectra".into()],
            search_config_sha256: "config".into(),
        };
        let mut record = StageRecord {
            schema_version: 3,
            stage: "optimized".into(),
            model: "moments".into(),
            input_hash: "input".into(),
            status: "running".into(),
            results: results.clone(),
            config_snapshot: config.clone(),
            results_sha256: sha256_file(&results).unwrap(),
            config_snapshot_sha256: sha256_file(&config).unwrap(),
            external_features_enabled: false,
            calibration_mode: "fit_current_search_space".into(),
            dataset_id: dataset.dataset_id.clone(),
            dataset_fingerprint: dataset.fingerprint.clone(),
            artifact_fit_dataset_fingerprint: None,
            candidate_pool: None,
            require_existing_candidate_pool: false,
            require_existing_annotation_cache: false,
            ms2rescore_annotation_cache: None,
            target_only_calibration_policy: None,
            release_candidate: true,
            window_provenance: None,
            external_profile_calibration: None,
            ensemble_shared_profile_contract_sha256: None,
            fitted_external_profile_identity_sha256: None,
            evaluable: true,
            not_evaluable_reason: None,
            target_only_policy_capability: None,
            nuisance_state_provenance: None,
            target_only_window_tuning: None,
            complete_dataset_artifact_reused: None,
            fallback_used: false,
            fallback_reason: None,
            model_artifact_schema: None,
            ensemble_interaction_calibration: None,
            parameter_overrides: BTreeMap::new(),
        };
        let mut legacy_value = serde_json::to_value(&record).unwrap();
        legacy_value["schema_version"] = serde_json::json!(2);
        for field in [
            "evaluable",
            "not_evaluable_reason",
            "target_only_policy_capability",
            "nuisance_state_provenance",
            "target_only_window_tuning",
            "complete_dataset_artifact_reused",
            "fallback_used",
            "fallback_reason",
            "model_artifact_schema",
        ] {
            legacy_value.as_object_mut().unwrap().remove(field);
        }
        let legacy_record: StageRecord = serde_json::from_value(legacy_value).unwrap();
        assert!(legacy_record.evaluable);
        assert!(!legacy_record.fallback_used);
        assert!(!stage_checkpoint_identity_matches(
            &record,
            "input",
            "optimized",
            &ModelFit::Moments,
            &dataset,
            &results,
            &config,
            None,
        ));
        record.status = "complete".into();
        assert!(stage_checkpoint_identity_matches(
            &record,
            "input",
            "optimized",
            &ModelFit::Moments,
            &dataset,
            &results,
            &config,
            None,
        ));
        assert!(stage_output_hashes_match(&record).unwrap());
        std::fs::write(&results, b"modified\n").unwrap();
        assert!(!stage_output_hashes_match(&record).unwrap());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn target_only_policy_defaults_to_refit_and_compare_both_marks_reuse_diagnostic() {
        let directory = test_directory("target-only-policy-default");
        let manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        let mut value = serde_json::to_value(&manifest).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("target_only_calibration_policy");
        value["models"][0]
            .as_object_mut()
            .unwrap()
            .remove("target_only_calibration_policy");
        value["models"][0]
            .as_object_mut()
            .unwrap()
            .remove("ensemble_interaction_baseline");
        let restored: WorkflowManifest = serde_json::from_value(value).unwrap();
        assert_eq!(
            restored.target_only_calibration_policy,
            TargetOnlyCalibrationPolicy::RefitWithLockedWindow
        );
        assert!(restored.models[0].ensemble_interaction_baseline);
        assert_eq!(
            concrete_target_only_policies(TargetOnlyCalibrationPolicy::CompareBoth),
            vec![
                (TargetOnlyCalibrationPolicy::RefitWithLockedWindow, true),
                (TargetOnlyCalibrationPolicy::ReuseDatasetArtifact, false),
            ]
        );
        assert!(allow_target_candidate_pool_reuse(false, 0));
        assert!(allow_target_candidate_pool_reuse(false, 1));
        assert!(!allow_target_candidate_pool_reuse(true, 0));
        assert!(allow_target_candidate_pool_reuse(true, 1));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn lower_order_target_only_capability_allows_refit_and_rejects_reuse() {
        let directory = test_directory("lower-order-target-only-capability");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.models[0].model = ModelFit::LowerOrder;
        manifest.models[0].target_only_calibration_policy =
            Some(TargetOnlyCalibrationPolicy::RefitWithLockedWindow);
        manifest.validate().unwrap();

        manifest.models[0].target_only_calibration_policy =
            Some(TargetOnlyCalibrationPolicy::ReuseDatasetArtifact);
        let error = manifest.validate().unwrap_err().to_string();
        assert!(error.contains("nuisance parameters and candidate-count normalization"));

        // A reusable artifact from another model is unaffected by the Lower
        // Order-specific capability restriction.
        manifest.models[0].model = ModelFit::Moments;
        manifest.validate().unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn lower_order_artifact_application_rejects_target_reuse_but_allows_same_pool_replay() {
        let artifacts = lower_order_artifacts(-1.8742677346525838);
        let mut same_pool = FdrOptions::default();
        apply_fitted_artifacts(
            &mut same_pool,
            &ModelFit::LowerOrder,
            artifacts.clone(),
            false,
            None,
        )
        .unwrap();
        assert_eq!(same_pool.lower_order_min_null_rank, Some(6));
        assert_eq!(same_pool.lower_order_max_null_rank, Some(9));
        assert!(same_pool.lower_order_frozen_artifact.is_some());

        let error = apply_fitted_artifacts(
            &mut FdrOptions::default(),
            &ModelFit::LowerOrder,
            artifacts,
            false,
            Some(TargetOnlyCalibrationPolicy::ReuseDatasetArtifact),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("nuisance parameters and candidate-count normalization"));
    }

    #[test]
    fn production_artifact_stamping_preserves_lower_order_f64_bits() {
        let directory = test_directory("lower-order-artifact-f64-roundtrip");
        let expected = -1.8742677346525838_f64;
        write_json_atomic(
            &directory.join("fitted_model_artifacts.json"),
            &lower_order_artifacts(expected),
        )
        .unwrap();
        let dataset = DatasetIdentity {
            schema_version: 1,
            dataset_id: "dataset".into(),
            fingerprint: "dataset-fingerprint".into(),
            target_fasta_sha256: "target".into(),
            spectra_sha256: vec!["spectra".into()],
            search_config_sha256: "config".into(),
        };
        stamp_fitted_artifacts(
            &directory,
            &dataset,
            "optimized",
            &ModelFit::LowerOrder,
            None,
            "search-fingerprint",
            "analysis-fingerprint",
        )
        .unwrap();
        let restored: DfRunArtifacts = serde_json::from_slice(
            &std::fs::read(directory.join("fitted_model_artifacts.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            restored.lower_order.unwrap().params_by_charge[0]
                .mu
                .to_bits(),
            expected.to_bits()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compare_both_plan_materializes_distinct_target_only_interpretations() {
        let directory = test_directory("target-only-compare-plan");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        manifest.search_config = workspace.join("tests/config.json");
        manifest.spectra = vec![workspace
            .join("tests/LQSRPAAPPAPGPGQLTLR.mzML")
            .display()
            .to_string()];
        manifest.target_only_calibration_policy = TargetOnlyCalibrationPolicy::CompareBoth;
        manifest.models[0].ms2rescore = Ms2RescorePolicy::Never;
        let manifest_path = directory.join("workflow.json");
        write_json_atomic(&manifest_path, &manifest).unwrap();

        let state = execute_workflow(&manifest_path, &directory, 1, true).unwrap();
        let refit = state
            .stages
            .iter()
            .find(|stage| {
                stage.stage == TargetOnlyCalibrationPolicy::RefitWithLockedWindow.stage_name()
            })
            .unwrap();
        let reuse = state
            .stages
            .iter()
            .find(|stage| {
                stage.stage == TargetOnlyCalibrationPolicy::ReuseDatasetArtifact.stage_name()
            })
            .unwrap();
        assert!(refit.release_candidate);
        assert!(!reuse.release_candidate);
        assert_eq!(refit.calibration_mode, "refit_with_locked_window");
        assert_eq!(reuse.calibration_mode, "reuse_dataset_artifact");
        assert_ne!(refit.results, reuse.results);
        assert_eq!(
            refit
                .window_provenance
                .as_ref()
                .unwrap()
                .source_dataset_fingerprint,
            state.dataset.fingerprint
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn lower_order_compare_both_marks_reuse_not_evaluable_with_provenance() {
        let directory = test_directory("lower-order-compare-both-plan");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        manifest.search_config = workspace.join("tests/config.json");
        manifest.spectra = vec![workspace
            .join("tests/LQSRPAAPPAPGPGQLTLR.mzML")
            .display()
            .to_string()];
        manifest.models[0].model = ModelFit::LowerOrder;
        manifest.models[0].ms2rescore = Ms2RescorePolicy::Never;
        manifest.models[0].candidate_windows.clear();
        manifest.models[0].window = Some(NullWindow {
            min_rank: 6,
            max_rank: 9,
        });
        manifest.target_only_calibration_policy = TargetOnlyCalibrationPolicy::CompareBoth;
        let manifest_path = directory.join("workflow.json");
        write_json_atomic(&manifest_path, &manifest).unwrap();

        let state = execute_workflow(&manifest_path, &directory, 1, true).unwrap();
        let refit = state
            .stages
            .iter()
            .find(|stage| {
                stage.stage == TargetOnlyCalibrationPolicy::RefitWithLockedWindow.stage_name()
            })
            .unwrap();
        assert!(refit.evaluable);
        assert_eq!(refit.target_only_window_tuning, Some(false));
        assert_eq!(
            refit.nuisance_state_provenance.as_deref(),
            Some("refitted_in_target_only_candidate_space")
        );
        assert_eq!(refit.complete_dataset_artifact_reused, Some(false));
        assert_eq!(refit.window_provenance.as_ref().unwrap().min_rank, Some(6));
        assert_eq!(refit.window_provenance.as_ref().unwrap().max_rank, Some(9));

        let reuse = state
            .stages
            .iter()
            .find(|stage| {
                stage.stage == TargetOnlyCalibrationPolicy::ReuseDatasetArtifact.stage_name()
            })
            .unwrap();
        assert!(!reuse.evaluable);
        assert_eq!(reuse.status, "not_evaluable");
        assert!(!reuse.release_candidate);
        assert!(reuse
            .not_evaluable_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("search-space dependent")));
        assert_eq!(reuse.complete_dataset_artifact_reused, Some(false));
        assert!(reuse
            .target_only_policy_capability
            .as_ref()
            .is_some_and(|capability| !capability.supported));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cross_dataset_artifacts_are_rejected_by_default() {
        let current = DatasetIdentity {
            schema_version: 1,
            dataset_id: "pxd".into(),
            fingerprint: "pxd-fingerprint".into(),
            target_fasta_sha256: "pxd-target".into(),
            spectra_sha256: vec!["pxd-spectra".into()],
            search_config_sha256: "same-config".into(),
        };
        let artifacts = DfRunArtifacts {
            provenance: Some(FittedArtifactProvenance {
                schema_version: 2,
                dataset_id: "isb".into(),
                dataset_fingerprint: "isb-fingerprint".into(),
                search_config_sha256: "same-config".into(),
                fit_search_fingerprint: "isb-search".into(),
                candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
                fit_stage: "optimized".into(),
                model: "moments".into(),
                external_profile_calibration: None,
            }),
            ..DfRunArtifacts::default()
        };
        assert!(validate_artifact_reuse(
            &artifacts,
            &current,
            &ArtifactReusePolicy::DatasetLocalOnly,
            &ModelFit::Moments,
            None,
        )
        .is_err());
        assert!(validate_artifact_reuse(
            &artifacts,
            &current,
            &ArtifactReusePolicy::CrossDatasetDiagnostic,
            &ModelFit::Moments,
            None,
        )
        .is_ok());
    }

    #[test]
    fn dataset_artifact_reuse_fails_closed_on_candidate_population_mismatch() {
        let current = DatasetIdentity {
            schema_version: 1,
            dataset_id: "isb".into(),
            fingerprint: "isb-fingerprint".into(),
            target_fasta_sha256: "isb-target".into(),
            spectra_sha256: vec!["isb-spectra".into()],
            search_config_sha256: "same-config".into(),
        };
        let artifacts = DfRunArtifacts {
            provenance: Some(fitted_artifact_provenance(
                &current,
                "optimized",
                &ModelFit::Moments,
                "source-search",
            )),
            ..DfRunArtifacts::default()
        };
        assert!(validate_artifact_reuse(
            &artifacts,
            &current,
            &ArtifactReusePolicy::DatasetLocalOnly,
            &ModelFit::Moments,
            Some("different-search"),
        )
        .is_err());
        assert!(validate_artifact_reuse(
            &artifacts,
            &current,
            &ArtifactReusePolicy::DatasetLocalOnly,
            &ModelFit::Moments,
            Some("source-search"),
        )
        .is_ok());
    }

    #[test]
    fn cross_dataset_escape_hatch_requires_diagnostic_only() {
        let directory = test_directory("cross-dataset-diagnostic");
        let artifact = directory.join("artifact.json");
        std::fs::write(&artifact, b"{}\n").unwrap();
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Holdout);
        manifest.models[0].candidate_windows.clear();
        manifest.models[0].ms2rescore = Ms2RescorePolicy::Always;
        manifest
            .locked_expert_artifacts
            .insert("moments".into(), artifact);
        manifest.artifact_reuse_policy = ArtifactReusePolicy::CrossDatasetDiagnostic;
        assert!(manifest.validate().is_err());
        manifest.validation.diagnostic_only = true;
        manifest.validate().unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn write_validation_tsv(path: &Path, start: usize, entrapments: usize) {
        write_validation_tsv_counts(path, start, 700, entrapments);
    }

    fn write_validation_tsv_counts(path: &Path, start: usize, targets: usize, entrapments: usize) {
        let mut text = String::from(
            "psm_id\trank\tlabel\tproteins\tpeptide\tdecoy_free_q_value\tdecoy_free_peptide_q\tdecoy_free_protein_q\tdecoy_free_protein_supported_peptide\tdecoy_free_peptide_supported_psm\n",
        );
        for index in start..start + targets {
            text.push_str(&format!(
                "t{index}\t1\t1\tTarget_{index}\t{}\t0.001\t0.001\t0.001\ttrue\ttrue\n",
                peptide(index)
            ));
        }
        for index in 0..entrapments {
            text.push_str(&format!(
                "e{index}\t1\t1\tEnt_{index}\t{}\t0.001\t0.001\t0.001\ttrue\ttrue\n",
                peptide(100_000 + index)
            ));
        }
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn frozen_moments_artifact_restores_its_window() {
        let mut artifacts = DfRunArtifacts {
            moments: Some(FrozenGumbelParameters {
                schema_version: 1,
                model_version: "sage-moments-gumbel-v1".into(),
                min_rank: 9,
                max_rank: 18,
                mu: 1.0,
                beta: 2.0,
            }),
            external_ms2rescore: Some(external_profiles()),
            ..DfRunArtifacts::default()
        };
        artifacts
            .external_ms2rescore
            .as_mut()
            .unwrap()
            .ccs_abs_error
            .good_median = None;
        let encoded = serde_json::to_vec(&artifacts).unwrap();
        let artifacts: DfRunArtifacts = serde_json::from_slice(&encoded).unwrap();
        let mut fdr = FdrOptions::default();
        apply_fitted_artifacts(&mut fdr, &ModelFit::Moments, artifacts, true, None).unwrap();
        assert_eq!(fdr.moments_min_null_rank, Some(9));
        assert_eq!(fdr.moments_max_null_rank, Some(18));
        assert!(fdr.moments_frozen_parameters.is_some());
        assert!(fdr.external_ms2rescore_frozen_profiles.is_some());
    }

    #[test]
    fn msfdr1_smix_lock_window_is_explicitly_fixed_at_rank_one() {
        let window = resolved_expert_window(&ModelFit::Msfdr1Smix, &None).unwrap();
        assert_eq!(window.min_rank, 1);
        assert_eq!(window.max_rank, 1);
    }

    #[test]
    fn ensemble_experts_cannot_overwrite_shared_external_profile() {
        let shared = external_profiles();
        let mut fdr = FdrOptions {
            external_ms2rescore_frozen_profiles: Some(shared.clone()),
            ..Default::default()
        };
        let mut first_profile = external_profiles();
        first_profile.ms2pip_pcc.good_median = Some(2.0);
        let first = DfRunArtifacts {
            moments: Some(FrozenGumbelParameters {
                schema_version: 1,
                model_version: "sage-moments-gumbel-v1".into(),
                min_rank: 9,
                max_rank: 18,
                mu: 1.0,
                beta: 2.0,
            }),
            external_ms2rescore: Some(first_profile),
            ..Default::default()
        };
        let mut second_profile = external_profiles();
        second_profile.ms2pip_pcc.good_median = Some(3.0);
        let second = DfRunArtifacts {
            mle: Some(FrozenGumbelParameters {
                schema_version: 1,
                model_version: "sage-mle-gumbel-v1".into(),
                min_rank: 9,
                max_rank: 18,
                mu: 1.0,
                beta: 2.0,
            }),
            external_ms2rescore: Some(second_profile),
            ..Default::default()
        };
        let shared_identity = fitted_external_profile_identity(&DfRunArtifacts {
            external_ms2rescore: Some(shared.clone()),
            ..Default::default()
        })
        .unwrap()
        .unwrap();
        let first_identity = fitted_external_profile_identity(&first).unwrap().unwrap();
        let second_identity = fitted_external_profile_identity(&second).unwrap().unwrap();
        assert_ne!(shared_identity.0, first_identity.0);
        assert_ne!(first_identity.0, second_identity.0);
        assert_eq!(shared_identity.1.min_null_rank, 9);
        assert_eq!(shared_identity.1.max_null_rank, 18);
        apply_fitted_artifacts(&mut fdr, &ModelFit::Moments, first, false, None).unwrap();
        apply_fitted_artifacts(&mut fdr, &ModelFit::Mle, second, false, None).unwrap();
        assert_eq!(
            fdr.external_ms2rescore_frozen_profiles
                .as_ref()
                .unwrap()
                .ms2pip_pcc
                .good_median,
            shared.ms2pip_pcc.good_median
        );
    }

    #[test]
    fn shared_ensemble_profile_contract_is_independent_of_expert_specific_provenance() {
        let dataset = DatasetIdentity {
            schema_version: 1,
            dataset_id: "dataset".into(),
            fingerprint: "dataset-fingerprint".into(),
            target_fasta_sha256: "target".into(),
            spectra_sha256: vec!["spectrum".into()],
            search_config_sha256: "configuration".into(),
        };
        let calibration = external_profiles().calibration;
        let expert = |model, min_rank, max_rank, fitted_profile: &str| EnsembleExpertLock {
            model,
            window: Some(NullWindow { min_rank, max_rank }),
            optimized_fitted_artifacts: PathBuf::from("artifact.json"),
            optimized_fitted_artifacts_sha256: format!("optimized-{fitted_profile}"),
            ms2rescore_fitted_artifacts: Some(PathBuf::from("ms2-artifact.json")),
            ms2rescore_fitted_artifacts_sha256: Some(format!("ms2-{fitted_profile}")),
            calibration_stage: "ms2rescore".into(),
            calibration_results: PathBuf::from("calibration.tsv"),
            target_only_results: PathBuf::from("target.tsv"),
            target_only_calibration_policy: TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
            enabled: true,
            target_peptides: 0,
            incremental_target_peptides: 0,
            gate_reasons: Vec::new(),
            gate_warnings: Vec::new(),
            fit_search_fingerprint: "shared-search-fingerprint".into(),
            candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
            interaction_baseline: true,
            participation_decision: "included_technical_validation_passed".into(),
            fallback_used: false,
            fallback_reason: None,
            target_only_policy_capability: None,
            fitted_external_profile_identity_sha256: Some(fitted_profile.into()),
            fitted_external_profile_calibration: Some(calibration.clone()),
            annotation_cache_fingerprint: Some(format!("cache-{fitted_profile}")),
            annotation_cache_manifest_sha256: Some(format!("manifest-{fitted_profile}")),
            annotation_cache_payload_sha256: Some(format!("payload-{fitted_profile}")),
        };
        let experts = vec![
            expert(ModelFit::Moments, 9, 18, "moments-profile"),
            expert(ModelFit::Mle, 8, 25, "mle-profile"),
            expert(ModelFit::LowerOrder, 6, 9, "lower-order-profile"),
            expert(ModelFit::Msfdr, 9, 13, "msfdr-profile"),
            expert(ModelFit::Msfdr1Smix, 1, 1, "msfdr1-profile"),
            expert(ModelFit::Msfdr2Smix, 9, 17, "msfdr2-profile"),
            expert(ModelFit::Nokoi, 4, 5, "nokoi-profile"),
        ];
        let identity =
            shared_ensemble_profile_contract_identity(&dataset, &calibration, &experts).unwrap();
        let mut reversed = experts.clone();
        reversed.reverse();
        assert_eq!(
            identity,
            shared_ensemble_profile_contract_identity(&dataset, &calibration, &reversed).unwrap()
        );
        let mut different_expert_provenance = experts.clone();
        for expert in &mut different_expert_provenance {
            expert.optimized_fitted_artifacts_sha256 = format!("other-{:?}", expert.model);
            expert.ms2rescore_fitted_artifacts_sha256 =
                Some(format!("other-ms2-{:?}", expert.model));
            expert.annotation_cache_fingerprint = Some(format!("other-cache-{:?}", expert.model));
            expert.annotation_cache_manifest_sha256 =
                Some(format!("other-manifest-{:?}", expert.model));
            expert.annotation_cache_payload_sha256 =
                Some(format!("other-payload-{:?}", expert.model));
        }
        assert_eq!(
            identity,
            shared_ensemble_profile_contract_identity(
                &dataset,
                &calibration,
                &different_expert_provenance,
            )
            .unwrap()
        );
        let requested_roster = experts
            .iter()
            .map(|expert| model_slug(&expert.model).to_owned())
            .collect::<Vec<_>>();
        let lock = stamp_ensemble_lock_analysis_fingerprint(EnsembleLock {
            schema_version: 8,
            post_selection_in_scope: true,
            source_manifest_hash: "manifest".into(),
            dataset_fingerprint: dataset.fingerprint.clone(),
            experts: experts.clone(),
            requested_roster: requested_roster.clone(),
            actual_roster: requested_roster,
            explicit_exclusions: BTreeMap::new(),
            technical_failures: BTreeMap::new(),
            roster_contract: default_ensemble_roster_contract(),
            minimum_required_experts: 2,
            evaluable: true,
            not_evaluable_reasons: Vec::new(),
            external_profile_contract: default_ensemble_external_profile_contract(),
            shared_external_profile_contract_sha256: Some(identity.clone()),
            shared_external_profile_calibration: Some(calibration.clone()),
            source_configuration_sha256: dataset.search_config_sha256.clone(),
            analysis_fingerprint: String::new(),
            raw_q_interaction_warning_threshold: default_raw_q_interaction_warning_threshold(),
            ensemble_p_combiner: EnsemblePCombiner::SecondBest,
            ensemble_pep_combiner: EnsemblePepCombiner::Median,
        })
        .unwrap();
        let mut changed_lock_provenance = lock.clone();
        changed_lock_provenance.experts[0].annotation_cache_fingerprint =
            Some("changed-cache".into());
        let changed_lock_provenance =
            stamp_ensemble_lock_analysis_fingerprint(changed_lock_provenance).unwrap();
        assert_eq!(
            changed_lock_provenance.shared_external_profile_contract_sha256,
            lock.shared_external_profile_contract_sha256
        );
        assert_ne!(
            changed_lock_provenance.analysis_fingerprint,
            lock.analysis_fingerprint
        );
        assert_ne!(
            experts[0].fitted_external_profile_identity_sha256,
            experts[1].fitted_external_profile_identity_sha256
        );
        let mut other_dataset = dataset.clone();
        other_dataset.fingerprint = "other-dataset".into();
        assert_ne!(
            shared_ensemble_profile_contract_identity(&dataset, &calibration, &experts).unwrap(),
            shared_ensemble_profile_contract_identity(&other_dataset, &calibration, &experts)
                .unwrap()
        );
        let mut mismatched_search = experts.clone();
        mismatched_search[0].fit_search_fingerprint = "other-search".into();
        assert!(shared_ensemble_profile_contract_identity(
            &dataset,
            &calibration,
            &mismatched_search,
        )
        .is_err());
    }

    #[test]
    fn msfdr_artifact_requires_portable_window_metadata() {
        let model = MsfdrSeededModel {
            null_loc: 1.0,
            null_scale: 1.0,
            target_mean: 2.0,
            target_std: 1.0,
            target_alpha: 0.0,
            pi: 0.5,
        };
        let legacy = DfRunArtifacts {
            msfdr_seeded: Some(model.clone()),
            ..DfRunArtifacts::default()
        };
        assert!(!artifact_contains_model(&legacy, &ModelFit::Msfdr));

        let portable = DfRunArtifacts {
            msfdr_seeded: Some(model),
            msfdr_seeded_metadata: Some(FrozenModelMetadata {
                schema_version: 1,
                model_version: "sage-msfdr-seeded-v1".into(),
                min_null_rank: Some(9),
                max_null_rank: Some(13),
                rank1_only: false,
            }),
            ..DfRunArtifacts::default()
        };
        assert!(artifact_contains_model(&portable, &ModelFit::Msfdr));
    }

    fn assert_secondary_model_votes(
        label: &str,
        model: ModelFit,
        window: NullWindow,
        mut artifacts: DfRunArtifacts,
    ) {
        let directory = test_directory(label);
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.models[0].model = model.clone();
        manifest.models[0].candidate_windows.clear();
        manifest.models[0].window = Some(window.clone());
        manifest.models[0].ms2rescore = Ms2RescorePolicy::Never;
        manifest.validation.minimum_ensemble_experts = 1;
        let dataset = DatasetIdentity {
            schema_version: 1,
            dataset_id: "test-dataset".into(),
            fingerprint: "dataset-fingerprint".into(),
            target_fasta_sha256: "target-sha256".into(),
            spectra_sha256: vec!["spectra-sha256".into()],
            search_config_sha256: sha256_file(&manifest.search_config).unwrap(),
        };
        artifacts.provenance = Some(fitted_artifact_provenance(
            &dataset,
            "optimized",
            &model,
            "test-search-fingerprint",
        ));
        if let Some(artifact) = artifacts.nokoi.as_mut() {
            artifact
                .stamp_workflow_identity(
                    &dataset.dataset_id,
                    &dataset.fingerprint,
                    "test-search-fingerprint",
                    "test-analysis-fingerprint",
                )
                .unwrap();
        }
        let artifact_path = directory.join("artifacts.json");
        write_json_atomic(&artifact_path, &artifacts).unwrap();
        let calibration = directory.join("calibration.tsv");
        let target = directory.join("target.tsv");
        write_validation_tsv(&calibration, 0, 10);
        write_validation_tsv_counts(&target, 0, 20, 0);
        let expert = CompletedExpert {
            model: model.clone(),
            window: Some(window),
            optimized_artifacts: artifact_path,
            optimized_results: calibration.clone(),
            ms2rescore_artifacts: None,
            ms2rescore_results: None,
            calibration_stage: "optimized".into(),
            calibration_results: calibration,
            target_only_results: target,
            target_only_calibration_policy: TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
            calibration_search_fingerprint: "test-search-fingerprint".into(),
            fitted_external_profile_identity_sha256: None,
            fitted_external_profile_calibration: None,
            annotation_cache_fingerprint: None,
            annotation_cache_manifest_sha256: None,
            annotation_cache_payload_sha256: None,
        };
        let lock = build_ensemble_lock(
            &manifest,
            "manifest-hash",
            &dataset,
            std::slice::from_ref(&expert),
        )
        .unwrap();
        assert_eq!(lock.requested_roster, vec![model_slug(&model)]);
        assert_eq!(lock.actual_roster, lock.requested_roster);
        assert!(lock.experts[0].enabled);
        assert!(lock.technical_failures.is_empty());
        let mut fdr = FdrOptions::default();
        apply_ensemble_lock(
            &mut fdr,
            &lock,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            false,
            None,
        )
        .unwrap();
        match model {
            ModelFit::Msfdr => assert_eq!(fdr.enable_msfdr_seeded, Some(true)),
            ModelFit::Msfdr2Smix => assert_eq!(fdr.enable_msfdr_2smix, Some(true)),
            ModelFit::Nokoi => assert_eq!(fdr.enable_nokoi, Some(true)),
            _ => unreachable!(),
        }
        manifest.models[0].ensemble_participation = EnsembleParticipation::Excluded;
        manifest.models[0].ensemble_exclusion_reason = Some("explicit test exclusion".into());
        let excluded = build_ensemble_lock(
            &manifest,
            "manifest-hash",
            &dataset,
            std::slice::from_ref(&expert),
        )
        .unwrap();
        assert!(excluded.requested_roster.is_empty());
        assert!(excluded.actual_roster.is_empty());
        assert_eq!(
            excluded
                .explicit_exclusions
                .get(model_slug(&model))
                .map(String::as_str),
            Some("explicit test exclusion")
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn repaired_secondary_experts_vote_when_json_selected_and_technically_valid() {
        assert_secondary_model_votes(
            "msfdr-configured-voter",
            ModelFit::Msfdr,
            NullWindow {
                min_rank: 9,
                max_rank: 13,
            },
            DfRunArtifacts {
                msfdr_seeded: Some(MsfdrSeededModel {
                    null_loc: 1.0,
                    null_scale: 1.0,
                    target_mean: 2.0,
                    target_std: 1.0,
                    target_alpha: 0.0,
                    pi: 0.5,
                }),
                msfdr_seeded_metadata: Some(FrozenModelMetadata {
                    schema_version: 1,
                    model_version: "sage-msfdr-seeded-v1".into(),
                    min_null_rank: Some(9),
                    max_null_rank: Some(13),
                    rank1_only: false,
                }),
                ..DfRunArtifacts::default()
            },
        );
        assert_secondary_model_votes(
            "msfdr2-configured-voter",
            ModelFit::Msfdr2Smix,
            NullWindow {
                min_rank: 9,
                max_rank: 17,
            },
            DfRunArtifacts {
                msfdr_2smix: Some(Msfdr2SmixModel {
                    correct: SkewNormal::new(3.0, 1.0, 0.0),
                    incorrect1: SkewNormal::new(1.0, 1.0, 0.0),
                    incorrect2: SkewNormal::new(0.0, 1.0, 0.0),
                    a: 0.5,
                    b: 0.5,
                }),
                msfdr_2smix_metadata: Some(FrozenModelMetadata {
                    schema_version: 1,
                    model_version: "sage-msfdr-2smix-v1".into(),
                    min_null_rank: Some(9),
                    max_null_rank: Some(17),
                    rank1_only: false,
                }),
                ..DfRunArtifacts::default()
            },
        );
        assert_secondary_model_votes(
            "nokoi-v2-configured-voter",
            ModelFit::Nokoi,
            NullWindow {
                min_rank: 2,
                max_rank: 12,
            },
            portable_nokoi_artifacts(),
        );
    }

    #[test]
    fn nokoi_v1_artifact_is_diagnostic_not_portable() {
        let artifact = NokoiArtifact {
            schema_version: 1,
            model_version: "sage-nokoi-crossfit-portable-v1".into(),
            identity: Default::default(),
            feature_contract: Default::default(),
            feature_schema: NOKOI_FEATURE_SCHEMA
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            min_null_rank: 2,
            max_null_rank: 12,
            crossfit_seed: 0x5EED_5EED_5EED_5EED,
            k_folds: 5,
            fold_sizes: vec![10; 5],
            config: NokoiConfig::default(),
            lambda_grid: Vec::new(),
            fold_models: Vec::new(),
            selected_l1_lambda: 0.1,
            final_model: LogisticRegression {
                weights: vec![0.0; NOKOI_FEATURE_SCHEMA.len()],
                bias: 0.0,
            },
            final_optimization: Default::default(),
            normalization: NokoiNormalization {
                medians: vec![0.0; NOKOI_FEATURE_SCHEMA.len()],
                means: vec![0.0; NOKOI_FEATURE_SCHEMA.len()],
                stds: vec![1.0; NOKOI_FEATURE_SCHEMA.len()],
            },
            null_scores_oof: vec![0.5; 50],
            development_pi0: 1.0,
            calibration_contract: Default::default(),
            grenander_blocks: Vec::new(),
            pep_calibration: vec![
                NokoiCalibrationPoint {
                    p_value: 0.0,
                    pep: 0.0,
                },
                NokoiCalibrationPoint {
                    p_value: 1.0,
                    pep: 1.0,
                },
            ],
            positive_training_count: 100,
            negative_training_count: 100,
            training_contract: Default::default(),
            training_completed: false,
            training_fallback_used: false,
            feature_selection_state: Vec::new(),
            reference_candidate_counts: vec![100; 50],
            integrity: Default::default(),
        };
        let artifacts = DfRunArtifacts {
            nokoi: Some(artifact),
            ..DfRunArtifacts::default()
        };
        assert!(!artifact_contains_model(&artifacts, &ModelFit::Nokoi));
        let mut fdr = FdrOptions::default();
        assert!(apply_fitted_artifacts(&mut fdr, &ModelFit::Nokoi, artifacts, true, None).is_err());
    }

    #[test]
    fn nokoi_v2_is_portable_and_target_policy_is_explicit() {
        let artifacts = portable_nokoi_artifacts();
        assert!(artifact_contains_model(&artifacts, &ModelFit::Nokoi));

        let mut same_pool = FdrOptions::default();
        apply_fitted_artifacts(
            &mut same_pool,
            &ModelFit::Nokoi,
            artifacts.clone(),
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            same_pool.nokoi_artifact_application_mode,
            Some(NokoiArtifactApplicationMode::ExactFitPopulation)
        );

        let mut target_reuse = FdrOptions::default();
        apply_fitted_artifacts(
            &mut target_reuse,
            &ModelFit::Nokoi,
            artifacts,
            false,
            Some(TargetOnlyCalibrationPolicy::ReuseDatasetArtifact),
        )
        .unwrap();
        assert_eq!(
            target_reuse.nokoi_artifact_application_mode,
            Some(NokoiArtifactApplicationMode::SameDatasetTargetOnly)
        );
    }

    #[test]
    fn workflow_stamps_nokoi_v2_internal_and_outer_identity_together() {
        let directory = test_directory("nokoi-v2-stamping");
        write_json_atomic(
            &directory.join("fitted_model_artifacts.json"),
            &portable_nokoi_artifacts(),
        )
        .unwrap();
        let dataset = DatasetIdentity {
            schema_version: 1,
            dataset_id: "dataset".into(),
            fingerprint: "dataset-fingerprint".into(),
            target_fasta_sha256: "target".into(),
            spectra_sha256: vec!["spectra".into()],
            search_config_sha256: "config".into(),
        };
        stamp_fitted_artifacts(
            &directory,
            &dataset,
            "optimized",
            &ModelFit::Nokoi,
            None,
            "search-fingerprint",
            "analysis-fingerprint",
        )
        .unwrap();
        let restored: DfRunArtifacts = serde_json::from_slice(
            &std::fs::read(directory.join("fitted_model_artifacts.json")).unwrap(),
        )
        .unwrap();
        validate_artifact_reuse(
            &restored,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            &ModelFit::Nokoi,
            Some("search-fingerprint"),
        )
        .unwrap();
        let artifact = restored.nokoi.as_ref().unwrap();
        artifact.validate_portable().unwrap();
        assert_eq!(
            artifact.identity.fit_analysis_fingerprint,
            "analysis-fingerprint"
        );
        let other_dataset = DatasetIdentity {
            schema_version: 1,
            dataset_id: "other-dataset".into(),
            fingerprint: "other-dataset-fingerprint".into(),
            target_fasta_sha256: "other-target".into(),
            spectra_sha256: vec!["other-spectra".into()],
            search_config_sha256: dataset.search_config_sha256.clone(),
        };
        assert!(validate_artifact_reuse(
            &restored,
            &other_dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            &ModelFit::Nokoi,
            Some("search-fingerprint"),
        )
        .is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn ensemble_lock_copies_independent_windows_without_manual_entry() {
        let directory = test_directory("ensemble-lock");
        let moments_artifact = directory.join("moments.artifacts.json");
        let mle_artifact = directory.join("mle.artifacts.json");
        write_json_atomic(
            &moments_artifact,
            &DfRunArtifacts {
                moments: Some(FrozenGumbelParameters {
                    schema_version: 1,
                    model_version: "sage-moments-gumbel-v1".into(),
                    min_rank: 9,
                    max_rank: 18,
                    mu: 1.0,
                    beta: 2.0,
                }),
                ..DfRunArtifacts::default()
            },
        )
        .unwrap();
        write_json_atomic(
            &mle_artifact,
            &DfRunArtifacts {
                mle: Some(FrozenGumbelParameters {
                    schema_version: 1,
                    model_version: "sage-mle-gumbel-v1".into(),
                    min_rank: 8,
                    max_rank: 25,
                    mu: 1.0,
                    beta: 2.0,
                }),
                ..DfRunArtifacts::default()
            },
        )
        .unwrap();
        let moments_calibration = directory.join("moments.calibration.tsv");
        let moments_target = directory.join("moments.target.tsv");
        let mle_calibration = directory.join("mle.calibration.tsv");
        let mle_target = directory.join("mle.target.tsv");
        let search_config = directory.join("search.json");
        std::fs::write(&search_config, b"{}\n").unwrap();
        write_validation_tsv(&moments_calibration, 0, 3);
        write_validation_tsv(&moments_target, 0, 0);
        write_validation_tsv(&mle_calibration, 100, 3);
        write_validation_tsv(&mle_target, 100, 0);

        let model = |model, window| ModelWorkflow {
            model,
            window,
            candidate_windows: Vec::new(),
            window_optimizer: None,
            enabled: true,
            ms2rescore: Ms2RescorePolicy::Never,
            maximum_raw_fdp_increase: None,
            minimum_level4_peptide_gain: None,
            target_only_calibration_policy: None,
            ensemble_participation: EnsembleParticipation::Auto,
            ensemble_exclusion_reason: None,
            ensemble_interaction_baseline: true,
        };
        let mut manifest = WorkflowManifest {
            schema_version: 1,
            name: "test".into(),
            dataset_id: Some("test-dataset".into()),
            search_config: search_config.clone(),
            target_fasta: PathBuf::new(),
            spectra: vec!["test.mzML".into()],
            output_root: directory.clone(),
            candidate_pool_root: None,
            annotation_cache_root: None,
            target_only_annotation_cache_root: None,
            entrapment: EntrapmentWorkflow {
                database_mode: EntrapmentDatabaseMode::NativeGenerated,
                foreign_fastas: Vec::new(),
                output_fasta: PathBuf::new(),
                frozen_legacy_fasta: None,
                foreign_source_mode: ForeignSourceMode::Automatic,
                shared_peptide_exclusion_mode: SharedPeptideExclusionMode::SageSearchSpace,
                selected_foreign_fasta: None,
                legacy_parity_reference: None,
                seed: 0,
                protein_fold: 1,
            },
            models: vec![
                model(
                    ModelFit::Moments,
                    Some(NullWindow {
                        min_rank: 9,
                        max_rank: 18,
                    }),
                ),
                model(
                    ModelFit::Mle,
                    Some(NullWindow {
                        min_rank: 8,
                        max_rank: 25,
                    }),
                ),
                model(ModelFit::Ensemble, None),
            ],
            baseline: None,
            validation: ValidationWorkflow {
                effective_ratios: EffectiveRatios::default(),
                null_window_validation_scope: NullWindowValidationScope::Level4,
                use_generated_entrapment_ratios: false,
                fdr_threshold: 0.01,
                maximum_transfer_fraction_loss: 0.20,
                additional_runs: Vec::new(),
                minimum_incremental_ensemble_peptides: 1,
                minimum_ensemble_experts: 2,
                minimum_entrapment_peptides_for_stable_estimate: 3,
                parity_pairs: Vec::new(),
                external_parity_evidence: None,
                maximum_parity_fraction_difference: 0.001,
                tdc_reference_method: None,
                dataset_role: ValidationDatasetRole::Development,
                diagnostic_only: false,
            },
            resume: false,
            require_existing_candidate_pool: false,
            require_existing_annotation_cache: false,
            migrate_schema_v2_annotation_cache_only: false,
            annotate_target_matches: false,
            ensemble_lock: None,
            locked_expert_artifacts: BTreeMap::new(),
            artifact_reuse_policy: ArtifactReusePolicy::DatasetLocalOnly,
            target_only_calibration_policy: TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
            parameter_optimizer: None,
        };
        let dataset = DatasetIdentity {
            schema_version: 1,
            dataset_id: "test-dataset".into(),
            fingerprint: "dataset-fingerprint".into(),
            target_fasta_sha256: "target-sha256".into(),
            spectra_sha256: vec!["spectra-sha256".into()],
            search_config_sha256: sha256_file(&search_config).unwrap(),
        };
        for path in [&moments_artifact, &mle_artifact] {
            let mut artifacts: DfRunArtifacts =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            artifacts.provenance = Some(fitted_artifact_provenance(
                &dataset,
                "optimized",
                if path == &moments_artifact {
                    &ModelFit::Moments
                } else {
                    &ModelFit::Mle
                },
                "test-search-fingerprint",
            ));
            write_json_atomic(path, &artifacts).unwrap();
        }
        let experts = vec![
            CompletedExpert {
                model: ModelFit::Moments,
                window: manifest.models[0].window.clone(),
                optimized_artifacts: moments_artifact,
                optimized_results: moments_calibration.clone(),
                ms2rescore_artifacts: None,
                ms2rescore_results: None,
                calibration_stage: "optimized".into(),
                calibration_results: moments_calibration,
                target_only_results: moments_target,
                target_only_calibration_policy: TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
                calibration_search_fingerprint: "test-search-fingerprint".into(),
                fitted_external_profile_identity_sha256: None,
                fitted_external_profile_calibration: None,
                annotation_cache_fingerprint: None,
                annotation_cache_manifest_sha256: None,
                annotation_cache_payload_sha256: None,
            },
            CompletedExpert {
                model: ModelFit::Mle,
                window: manifest.models[1].window.clone(),
                optimized_artifacts: mle_artifact,
                optimized_results: mle_calibration.clone(),
                ms2rescore_artifacts: None,
                ms2rescore_results: None,
                calibration_stage: "optimized".into(),
                calibration_results: mle_calibration,
                target_only_results: mle_target,
                target_only_calibration_policy: TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
                calibration_search_fingerprint: "test-search-fingerprint".into(),
                fitted_external_profile_identity_sha256: None,
                fitted_external_profile_calibration: None,
                annotation_cache_fingerprint: None,
                annotation_cache_manifest_sha256: None,
                annotation_cache_payload_sha256: None,
            },
        ];
        let lock = build_ensemble_lock(&manifest, "manifest-hash", &dataset, &experts).unwrap();
        let canonical_bytes = serde_json::to_vec(&lock).unwrap();
        let mut reversed_experts = experts.clone();
        reversed_experts.reverse();
        assert_eq!(
            serde_json::to_vec(
                &build_ensemble_lock(&manifest, "manifest-hash", &dataset, &reversed_experts,)
                    .unwrap()
            )
            .unwrap(),
            canonical_bytes
        );
        let mut permuted_lock = lock.clone();
        permuted_lock.experts.reverse();
        assert_eq!(
            serde_json::to_vec(&canonicalize_ensemble_lock(permuted_lock)).unwrap(),
            canonical_bytes
        );
        assert_eq!(lock.dataset_fingerprint, dataset.fingerprint);
        assert_eq!(lock.schema_version, 8);
        assert_eq!(lock.requested_roster, vec!["mle", "moments"]);
        assert_eq!(lock.actual_roster, lock.requested_roster);
        assert!(lock.technical_failures.is_empty());
        let runtime_failure = BTreeMap::from([(
            "mle".into(),
            vec!["target-only expert stage failed technically: corrupt payload".into()],
        )]);
        let reduced = build_ensemble_lock_with_failures(
            &manifest,
            "manifest-hash",
            &dataset,
            &experts[..1],
            &runtime_failure,
        )
        .unwrap();
        assert_eq!(reduced.requested_roster, vec!["mle", "moments"]);
        assert_eq!(reduced.actual_roster, vec!["moments"]);
        assert_eq!(reduced.technical_failures, runtime_failure);
        assert!(!reduced.evaluable);
        assert!(lock.evaluable);
        assert!(lock.not_evaluable_reasons.is_empty());
        assert_eq!(
            lock.experts.iter().filter(|expert| expert.enabled).count(),
            2
        );
        assert!(lock.experts.iter().any(|expert| {
            expert.model == ModelFit::Moments
                && expert
                    .window
                    .as_ref()
                    .is_some_and(|window| window.min_rank == 9 && window.max_rank == 18)
        }));
        assert!(lock.experts.iter().any(|expert| {
            expert.model == ModelFit::Mle
                && expert
                    .window
                    .as_ref()
                    .is_some_and(|window| window.min_rank == 8 && window.max_rank == 25)
        }));
        let mut refit = FdrOptions::default();
        apply_ensemble_lock(
            &mut refit,
            &lock,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            false,
            None,
        )
        .unwrap();
        assert_eq!(refit.moments_min_null_rank, Some(9));
        assert_eq!(refit.moments_max_null_rank, Some(18));
        assert!(refit.moments_frozen_parameters.is_none());
        assert!(refit.mle_frozen_parameters.is_none());

        let mut legacy_lock = lock.clone();
        legacy_lock.schema_version = 7;
        assert!(apply_ensemble_lock(
            &mut FdrOptions::default(),
            &legacy_lock,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            false,
            None,
        )
        .is_err());

        let mut duplicate_lock = lock.clone();
        duplicate_lock.experts.push(lock.experts[0].clone());
        assert!(apply_ensemble_lock(
            &mut FdrOptions::default(),
            &duplicate_lock,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            false,
            None,
        )
        .is_err());

        let mut duplicate_artifact_vote = lock.clone();
        duplicate_artifact_vote.experts[1].optimized_fitted_artifacts_sha256 =
            duplicate_artifact_vote.experts[0]
                .optimized_fitted_artifacts_sha256
                .clone();
        let duplicate_artifact_vote =
            stamp_ensemble_lock_analysis_fingerprint(duplicate_artifact_vote).unwrap();
        let error = apply_ensemble_lock(
            &mut FdrOptions::default(),
            &duplicate_artifact_vote,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            false,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("duplicate optimized artifact vote"));

        let mut missing_failure_provenance = reduced.clone();
        missing_failure_provenance.technical_failures.clear();
        missing_failure_provenance.minimum_required_experts = 1;
        missing_failure_provenance.evaluable = true;
        missing_failure_provenance.not_evaluable_reasons.clear();
        let missing_failure_provenance =
            stamp_ensemble_lock_analysis_fingerprint(missing_failure_provenance).unwrap();
        let error = apply_ensemble_lock(
            &mut FdrOptions::default(),
            &missing_failure_provenance,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            false,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("actual roster or technical failures"));

        let mut reuse = FdrOptions::default();
        apply_ensemble_lock(
            &mut reuse,
            &lock,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            true,
            None,
        )
        .unwrap();
        assert!(reuse.moments_frozen_parameters.is_some());
        assert!(reuse.mle_frozen_parameters.is_some());

        // Statistical diagnostics, overlap, thresholds, and dataset role do
        // not change the configured technically valid voter roster.
        write_validation_tsv_counts(&experts[0].calibration_results, 0, 700, 10);
        std::fs::copy(
            &experts[0].calibration_results,
            &experts[1].calibration_results,
        )
        .unwrap();
        write_validation_tsv_counts(&experts[0].target_only_results, 0, 10, 0);
        std::fs::copy(
            &experts[0].target_only_results,
            &experts[1].target_only_results,
        )
        .unwrap();
        manifest.validation.maximum_transfer_fraction_loss = 0.0;
        manifest
            .validation
            .minimum_entrapment_peptides_for_stable_estimate = 100;
        manifest.validation.minimum_incremental_ensemble_peptides = usize::MAX;
        manifest.validation.dataset_role = ValidationDatasetRole::Holdout;
        let diagnostic_failures =
            build_ensemble_lock(&manifest, "manifest-hash", &dataset, &experts).unwrap();
        assert_eq!(diagnostic_failures.actual_roster, lock.actual_roster);
        assert!(diagnostic_failures
            .experts
            .iter()
            .all(|expert| expert.enabled && expert.incremental_target_peptides == 0));
        assert!(diagnostic_failures.experts.iter().all(|expert| {
            expert
                .gate_warnings
                .iter()
                .any(|warning| warning.contains("nonblocking diagnostic"))
        }));
        assert_ne!(
            diagnostic_failures.analysis_fingerprint,
            lock.analysis_fingerprint
        );
        assert_eq!(
            diagnostic_failures.source_configuration_sha256,
            lock.source_configuration_sha256
        );
        assert!(diagnostic_failures.experts.iter().zip(&lock.experts).all(
            |(diagnostic, original)| diagnostic.fit_search_fingerprint
                == original.fit_search_fingerprint
                && diagnostic.optimized_fitted_artifacts_sha256
                    == original.optimized_fitted_artifacts_sha256
        ));
        manifest.validation.fdr_threshold = 0.0001;
        let changed_reporting_threshold =
            build_ensemble_lock(&manifest, "manifest-hash", &dataset, &experts).unwrap();
        assert_eq!(
            changed_reporting_threshold.actual_roster,
            lock.actual_roster
        );
        manifest.validation.fdr_threshold = 0.01;

        manifest.models[1].ensemble_participation = EnsembleParticipation::Excluded;
        manifest.models[1].ensemble_exclusion_reason =
            Some("annotated parity exceeds the platform tolerance".into());
        let deferred = build_ensemble_lock(&manifest, "manifest-hash", &dataset, &experts).unwrap();
        assert!(!deferred.evaluable);
        assert_eq!(
            deferred
                .experts
                .iter()
                .filter(|expert| expert.enabled)
                .count(),
            1
        );
        assert!(deferred
            .not_evaluable_reasons
            .iter()
            .any(|reason| reason.contains("only 1 technically valid")));
        assert!(deferred.experts.iter().any(|expert| {
            expert.model == ModelFit::Mle
                && expert
                    .gate_reasons
                    .iter()
                    .any(|reason| reason.contains("explicit JSON exclusion"))
        }));
        assert_ne!(deferred.analysis_fingerprint, lock.analysis_fingerprint);
        assert_eq!(
            deferred.source_configuration_sha256,
            lock.source_configuration_sha256
        );
        let deferred_moments = deferred
            .experts
            .iter()
            .find(|expert| expert.model == ModelFit::Moments)
            .unwrap();
        let original_moments = lock
            .experts
            .iter()
            .find(|expert| expert.model == ModelFit::Moments)
            .unwrap();
        assert_eq!(
            deferred_moments.fit_search_fingerprint,
            original_moments.fit_search_fingerprint
        );
        assert_eq!(
            deferred_moments.optimized_fitted_artifacts_sha256,
            original_moments.optimized_fitted_artifacts_sha256
        );
        let mut rejected = FdrOptions::default();
        assert!(apply_ensemble_lock(
            &mut rejected,
            &deferred,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            false,
            None,
        )
        .is_err());

        manifest.models[1].ensemble_participation = EnsembleParticipation::Auto;
        manifest.models[1].ensemble_exclusion_reason = None;
        let mle_baseline = directory.join("mle.legacy.optimized.tsv");
        write_validation_tsv_counts(&mle_baseline, 1_000, 350, 3);
        manifest.validation.additional_runs.push(ValidationRun {
            method: "legacy_mle".into(),
            stage: "optimized".into(),
            results: mle_baseline,
            mode: ValidationMode::DecoyFree,
            expected_search_space: Some("+Ent".into()),
            calibration_stage: None,
            target_only_calibration_policy: None,
            release_candidate: true,
        });
        manifest.validation.parity_pairs.push(ParityPair {
            baseline_method: "legacy_mle".into(),
            native_method: "mle".into(),
            stages: vec!["optimized".into()],
            layers: vec!["level4".into()],
            maximum_fraction_difference: Some(0.001),
        });
        let parity_rejected =
            build_ensemble_lock(&manifest, "manifest-hash", &dataset, &experts).unwrap();
        assert!(parity_rejected.evaluable);
        assert!(parity_rejected.experts.iter().any(|expert| {
            expert.model == ModelFit::Mle
                && expert.enabled
                && expert.gate_warnings.iter().any(|reason| {
                    reason.contains("nonblocking diagnostic")
                        && reason.contains("parity exceeds tolerance")
                })
        }));

        std::fs::write(&experts[1].calibration_results, b"not a result table\n").unwrap();
        let unreadable_expert =
            build_ensemble_lock(&manifest, "manifest-hash", &dataset, &experts).unwrap();
        assert!(unreadable_expert.evaluable);
        assert!(unreadable_expert.experts.iter().any(|expert| {
            expert.model == ModelFit::Mle
                && expert.enabled
                && expert
                    .gate_warnings
                    .iter()
                    .any(|reason| reason.contains("nonblocking diagnostic"))
        }));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn lower_order_is_configured_voter_and_records_refit_only_capability() {
        let directory = test_directory("lower-order-auto-ensemble");
        let mut manifest = minimal_manifest(&directory, ValidationDatasetRole::Development);
        manifest.models[0].model = ModelFit::LowerOrder;
        manifest.models[0].candidate_windows.clear();
        manifest.models[0].window = Some(NullWindow {
            min_rank: 6,
            max_rank: 9,
        });
        manifest.models[0].ms2rescore = Ms2RescorePolicy::Never;
        manifest.models[0].ensemble_participation = EnsembleParticipation::Auto;
        manifest.models[0].ensemble_interaction_baseline = false;
        manifest.validation.minimum_ensemble_experts = 1;
        let dataset = DatasetIdentity {
            schema_version: 1,
            dataset_id: "test-dataset".into(),
            fingerprint: "dataset-fingerprint".into(),
            target_fasta_sha256: "target-sha256".into(),
            spectra_sha256: vec!["spectra-sha256".into()],
            search_config_sha256: sha256_file(&manifest.search_config).unwrap(),
        };
        let artifact_path = directory.join("lower-order.artifacts.json");
        let mut artifacts = lower_order_artifacts(-1.5);
        artifacts.provenance = Some(fitted_artifact_provenance(
            &dataset,
            "optimized",
            &ModelFit::LowerOrder,
            "test-search-fingerprint",
        ));
        write_json_atomic(&artifact_path, &artifacts).unwrap();
        let calibration = directory.join("lower-order.calibration.tsv");
        let target = directory.join("lower-order.target.tsv");
        write_validation_tsv_counts(&calibration, 500, 200, 1);
        write_validation_tsv_counts(&target, 500, 200, 0);
        let expert = CompletedExpert {
            model: ModelFit::LowerOrder,
            window: manifest.models[0].window.clone(),
            optimized_artifacts: artifact_path.clone(),
            optimized_results: calibration.clone(),
            ms2rescore_artifacts: None,
            ms2rescore_results: None,
            calibration_stage: "optimized".into(),
            calibration_results: calibration,
            target_only_results: target,
            target_only_calibration_policy: TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
            calibration_search_fingerprint: "test-search-fingerprint".into(),
            fitted_external_profile_identity_sha256: None,
            fitted_external_profile_calibration: None,
            annotation_cache_fingerprint: None,
            annotation_cache_manifest_sha256: None,
            annotation_cache_payload_sha256: None,
        };
        let lock =
            build_ensemble_lock(&manifest, "manifest-hash", &dataset, &[expert.clone()]).unwrap();
        let lower_order = &lock.experts[0];
        assert!(lower_order.enabled);
        assert_eq!(
            lower_order.participation_decision,
            "included_technical_validation_passed"
        );
        assert!(!lower_order.interaction_baseline);
        assert!(!lower_order.fallback_used);
        assert_eq!(
            lower_order.target_only_policy_capability,
            Some(target_only_policy_capability(
                &ModelFit::LowerOrder,
                TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
            ))
        );
        let mut target_refit = FdrOptions::default();
        apply_ensemble_lock(
            &mut target_refit,
            &lock,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            false,
            Some(TargetOnlyCalibrationPolicy::RefitWithLockedWindow),
        )
        .unwrap();
        assert_eq!(target_refit.enable_lower_order, Some(true));
        let error = apply_ensemble_lock(
            &mut FdrOptions::default(),
            &lock,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            true,
            Some(TargetOnlyCalibrationPolicy::ReuseDatasetArtifact),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("nuisance parameters and candidate-count normalization"));
        let baseline = interaction_baseline_lock(&lock).unwrap();
        assert!(!baseline.evaluable);
        assert_ne!(baseline.analysis_fingerprint, lock.analysis_fingerprint);
        assert_eq!(
            baseline.experts[0].fit_search_fingerprint,
            lock.experts[0].fit_search_fingerprint
        );

        std::fs::write(&artifact_path, b"{}\n").unwrap();
        let rejected =
            build_ensemble_lock(&manifest, "manifest-hash", &dataset, &[expert]).unwrap();
        assert!(!rejected.experts[0].enabled);
        assert!(rejected.experts[0].fallback_used);
        assert!(rejected.experts[0]
            .gate_reasons
            .iter()
            .any(|reason| reason.contains("artifact")));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
