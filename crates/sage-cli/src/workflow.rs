use crate::candidate_pool::{
    content_sha256, search_fingerprint, CandidatePoolRequest, CandidatePoolUsage,
    CANDIDATE_ID_SCHEMA,
};
use crate::entrapment::{
    compare_generated_to_legacy, entrapment_generation_input_sha256, generate_foreign_entrapment,
    inspect_frozen_entrapment, EntrapmentDatabaseMode, EntrapmentDatabaseReport,
    EntrapmentFastaParityReport, EntrapmentGenerationReport, ForeignSourceMode,
    LegacyEntrapmentReference, SharedPeptideExclusionMode,
};
use crate::external_feature_cache::{
    generator_settings_sha256, verify_usage as verify_annotation_cache_usage,
    ExternalAnnotationCacheRequest, ExternalAnnotationCacheUsage,
};
use crate::input::Input;
use crate::provenance::{freeze_baseline, sha256_file, write_json_atomic, BaselineManifest};
use crate::runner::Runner;
use crate::validation::{
    accepted_target_peptides, expert_quality_gates, is_target_only_stage, parity_comparisons,
    stage_comparisons, summarize_run, tdc_benchmark_comparisons, transfer_stability,
    EffectiveRatios, ExpertQualityGate, ParityComparison, ParityPair, RunValidationSummary,
    StageComparison, TargetOnlyCalibrationPolicy, TdcBenchmarkComparison, ValidationMode,
    ValidationRun,
};
use anyhow::{Context, Result};
use sage_core::decoy_free_fdr::{DfRunArtifacts, FittedArtifactProvenance};
use sage_core::input::{
    FdrMode, FdrOptions, ModelFit, NullWindowCandidate, NullWindowOptimizerOptions,
    NullWindowValidationScope,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NullWindow {
    pub min_rank: u32,
    pub max_rank: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ms2RescorePolicy {
    Never,
    Measure,
    Always,
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
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub ms2rescore: Ms2RescorePolicy,
    #[serde(default)]
    pub maximum_raw_fdp_increase: Option<f64>,
    #[serde(default)]
    pub minimum_level4_peptide_gain: Option<usize>,
    /// Optional model-specific exception to the workflow-wide target-only
    /// calibration policy (for example, while Lower Order is evaluated).
    #[serde(default)]
    pub target_only_calibration_policy: Option<TargetOnlyCalibrationPolicy>,
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
    pub maximum_transfer_fraction_loss: f64,
    #[serde(default)]
    pub additional_runs: Vec<ValidationRun>,
    #[serde(default = "default_minimum_incremental_peptides")]
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
    pub target_only_results: PathBuf,
    pub enabled: bool,
    pub target_peptides: usize,
    pub incremental_target_peptides: usize,
    pub gate_reasons: Vec<String>,
    pub gate_warnings: Vec<String>,
    #[serde(default)]
    pub fit_search_fingerprint: String,
    #[serde(default)]
    pub candidate_id_schema: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleLock {
    pub schema_version: u32,
    pub source_manifest_hash: String,
    #[serde(default)]
    pub dataset_fingerprint: String,
    pub experts: Vec<EnsembleExpertLock>,
    pub minimum_required_experts: usize,
}

#[derive(Clone)]
struct CompletedExpert {
    model: ModelFit,
    window: Option<NullWindow>,
    optimized_artifacts: PathBuf,
    ms2rescore_artifacts: Option<PathBuf>,
    calibration_stage: String,
    calibration_results: PathBuf,
    target_only_results: PathBuf,
    calibration_search_fingerprint: String,
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
    pub entrapment: EntrapmentWorkflow,
    pub models: Vec<ModelWorkflow>,
    #[serde(default)]
    pub baseline: Option<BaselineWorkflow>,
    pub validation: ValidationWorkflow,
    #[serde(default = "default_true")]
    pub resume: bool,
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
    pub stage: String,
    pub model: String,
    pub input_hash: String,
    pub status: String,
    pub results: PathBuf,
    pub config_snapshot: PathBuf,
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
    pub ms2rescore_annotation_cache: Option<ExternalAnnotationCacheUsage>,
    #[serde(default)]
    pub target_only_calibration_policy: Option<TargetOnlyCalibrationPolicy>,
    #[serde(default = "default_true")]
    pub release_candidate: bool,
    #[serde(default)]
    pub window_provenance: Option<WindowProvenance>,
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
    pub stage_comparisons: Vec<StageComparison>,
    pub ensemble_expert_gates: Vec<ExpertQualityGate>,
    pub parity_comparisons: Vec<ParityComparison>,
    pub tdc_benchmarks: Vec<TdcBenchmarkComparison>,
    pub release_gate: ReleaseGate,
    pub transfer_stability: Vec<crate::validation::TransferStability>,
    pub pending_validation_gates: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseGate {
    pub eligible_for_statistical_default_change: bool,
    pub reasons: Vec<String>,
    pub calibrated_tdc_improvements: usize,
}

fn default_true() -> bool {
    true
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

impl WorkflowManifest {
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
        anyhow::ensure!(!self.name.trim().is_empty(), "workflow name is required");
        anyhow::ensure!(self.search_config.is_file(), "search_config does not exist");
        anyhow::ensure!(self.target_fasta.is_file(), "target_fasta does not exist");
        anyhow::ensure!(
            !self.spectra.is_empty(),
            "at least one spectrum file is required"
        );
        anyhow::ensure!(!self.models.is_empty(), "at least one model is required");
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
            anyhow::ensure!(
                model.window.is_none() || model.candidate_windows.is_empty(),
                "specify either a locked window or candidate_windows, not both"
            );
            if let Some(window) = &model.window {
                anyhow::ensure!(window.min_rank > 1, "rank 1 cannot be a null window");
                anyhow::ensure!(window.max_rank >= window.min_rank, "invalid null window");
            }
            if model.model == ModelFit::Msfdr1Smix {
                anyhow::ensure!(
                    model.window.is_none() && model.candidate_windows.is_empty(),
                    "MSFDR1-SMIX is rank-1-only and must not define a null window"
                );
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
                        model.candidate_windows.is_empty(),
                        "a cross-dataset diagnostic cannot both import an artifact and optimize candidate windows"
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

fn artifact_contains_model(
    artifacts: &sage_core::decoy_free_fdr::DfRunArtifacts,
    model: &ModelFit,
) -> bool {
    match model {
        ModelFit::Moments => artifacts.moments.is_some(),
        ModelFit::Mle => artifacts.mle.is_some(),
        ModelFit::LowerOrder => artifacts.lower_order.is_some(),
        ModelFit::Msfdr => {
            artifacts.msfdr_seeded.is_some()
                && artifacts
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
            artifacts.msfdr_1smix.is_some()
                && artifacts
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
            artifacts.msfdr_2smix.is_some()
                && artifacts
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
        ModelFit::Nokoi => artifacts.nokoi.is_some(),
        ModelFit::Ensemble => false,
    }
}

fn apply_fitted_artifacts(
    fdr: &mut FdrOptions,
    model: &ModelFit,
    artifacts: sage_core::decoy_free_fdr::DfRunArtifacts,
) -> Result<()> {
    anyhow::ensure!(
        artifact_contains_model(&artifacts, model),
        "fitted artifact does not contain {:?}",
        model
    );
    if let Some(profiles) = artifacts.external_ms2rescore.as_ref() {
        anyhow::ensure!(
            profiles.schema_version == 1
                && profiles.model_version == "sage-external-ms2rescore-profiles-v1",
            "external MS2Rescore fitted artifact is not portable or has an unsupported version"
        );
    }
    fdr.external_ms2rescore_frozen_profiles = artifacts.external_ms2rescore.clone();
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
) -> Result<()> {
    fdr.enable_moments = Some(false);
    fdr.enable_mle = Some(false);
    fdr.enable_lower_order = Some(false);
    fdr.enable_msfdr_seeded = Some(false);
    fdr.enable_msfdr_1smix = Some(false);
    fdr.enable_msfdr_2smix = Some(false);
    fdr.enable_nokoi = Some(false);
    let enabled = lock.experts.iter().filter(|expert| expert.enabled).count();
    anyhow::ensure!(
        enabled >= lock.minimum_required_experts,
        "Ensemble lock has only {enabled} eligible experts"
    );
    for expert in lock.experts.iter().filter(|expert| expert.enabled) {
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
        apply_fitted_artifacts(fdr, &expert.model, artifacts)?;
    }
    Ok(())
}

fn build_ensemble_lock(
    manifest: &WorkflowManifest,
    manifest_hash: &str,
    dataset: &DatasetIdentity,
    experts: &[CompletedExpert],
) -> Result<EnsembleLock> {
    let mut candidates = Vec::new();
    let ensemble_uses_ms2 = manifest.models.iter().any(|model| {
        model.enabled
            && model.model == ModelFit::Ensemble
            && !matches!(model.ms2rescore, Ms2RescorePolicy::Never)
    });
    for expert in experts {
        let mut reasons = Vec::new();
        let mut warnings = Vec::new();
        if !expert.optimized_artifacts.is_file() {
            reasons.push("optimized fitted artifact is missing".into());
        }
        let artifacts = if expert.optimized_artifacts.is_file() {
            serde_json::from_slice::<sage_core::decoy_free_fdr::DfRunArtifacts>(&std::fs::read(
                &expert.optimized_artifacts,
            )?)
            .ok()
        } else {
            None
        };
        if !artifacts
            .as_ref()
            .is_some_and(|artifact| artifact_contains_model(artifact, &expert.model))
        {
            reasons.push("optimized model artifact is absent (possible fit fallback)".into());
        }
        if let Some(artifact) = artifacts.as_ref() {
            if let Err(error) = validate_artifact_reuse(
                artifact,
                dataset,
                &ArtifactReusePolicy::DatasetLocalOnly,
                &expert.model,
                Some(&expert.calibration_search_fingerprint),
            ) {
                reasons.push(format!("optimized artifact provenance is invalid: {error}"));
            }
        }
        if ensemble_uses_ms2 {
            let ms2_artifact_valid = expert
                .ms2rescore_artifacts
                .as_ref()
                .filter(|path| path.is_file())
                .and_then(|path| {
                    serde_json::from_slice::<sage_core::decoy_free_fdr::DfRunArtifacts>(
                        &std::fs::read(path).ok()?,
                    )
                    .ok()
                })
                .is_some_and(|artifact| {
                    artifact_contains_model(&artifact, &expert.model)
                        && validate_artifact_reuse(
                            &artifact,
                            dataset,
                            &ArtifactReusePolicy::DatasetLocalOnly,
                            &expert.model,
                            Some(&expert.calibration_search_fingerprint),
                        )
                        .is_ok()
                });
            if !ms2_artifact_valid {
                reasons.push("MS2Rescore model artifact is missing or fell back".into());
            }
        }
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
        let summaries = summarize_run(
            &run,
            &manifest.validation.effective_ratios,
            manifest.validation.fdr_threshold,
        )?;
        let gate_layer = match manifest.validation.null_window_validation_scope {
            NullWindowValidationScope::RawQ => "raw_q",
            NullWindowValidationScope::Level4 => "level4",
        };
        for layer in ["raw_q", "level4"] {
            let row = summaries.iter().find(|row| row.layer == layer);
            if row.is_none() {
                if layer == gate_layer {
                    reasons.push(format!("missing {layer} calibration summary"));
                } else {
                    warnings.push(format!("missing secondary {layer} calibration summary"));
                }
            } else if let Some(row) = row {
                if row.peptide.entrapment
                    < manifest
                        .validation
                        .minimum_entrapment_peptides_for_stable_estimate
                {
                    warnings.push(format!(
                        "{layer} peptide entrapment count {} is below the stability minimum {}",
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
                    if layer == gate_layer {
                        reasons.push(format!("{layer} peptide entrapment FDP exceeds threshold"));
                    } else {
                        warnings.push(format!(
                            "secondary {layer} peptide entrapment FDP exceeds threshold"
                        ));
                    }
                }
            }
        }
        let calibration_peptides = if expert.calibration_results.is_file() {
            accepted_target_peptides(
                &expert.calibration_results,
                &ValidationMode::DecoyFree,
                manifest.validation.fdr_threshold,
                manifest.validation.null_window_validation_scope
                    == NullWindowValidationScope::Level4,
            )?
        } else {
            BTreeSet::new()
        };
        let target_peptides = if expert.target_only_results.is_file() {
            accepted_target_peptides(
                &expert.target_only_results,
                &ValidationMode::DecoyFree,
                manifest.validation.fdr_threshold,
                manifest.validation.null_window_validation_scope
                    == NullWindowValidationScope::Level4,
            )?
        } else {
            reasons.push("target-only result is missing".into());
            BTreeSet::new()
        };
        if !calibration_peptides.is_empty() {
            let change = (target_peptides.len() as f64 - calibration_peptides.len() as f64)
                / calibration_peptides.len() as f64;
            if change < -manifest.validation.maximum_transfer_fraction_loss {
                reasons.push(format!(
                    "target-only peptide transfer loss is {:.1}%",
                    -100.0 * change
                ));
            }
        }
        candidates.push((expert, reasons, warnings, calibration_peptides));
    }
    candidates.sort_by(|left, right| {
        right
            .3
            .len()
            .cmp(&left.3.len())
            .then_with(|| model_slug(&left.0.model).cmp(model_slug(&right.0.model)))
    });
    let mut union = BTreeSet::new();
    let mut locked = Vec::new();
    for (expert, mut reasons, warnings, peptides) in candidates {
        let incremental = peptides.difference(&union).count();
        if reasons.is_empty()
            && incremental < manifest.validation.minimum_incremental_ensemble_peptides
        {
            reasons.push(format!(
                "adds only {incremental} new Level-4 target peptides"
            ));
        }
        let enabled = reasons.is_empty();
        if enabled {
            union.extend(peptides.iter().cloned());
        }
        locked.push(EnsembleExpertLock {
            model: expert.model.clone(),
            window: expert.window.clone(),
            optimized_fitted_artifacts: expert.optimized_artifacts.clone(),
            optimized_fitted_artifacts_sha256: if expert.optimized_artifacts.is_file() {
                sha256_file(&expert.optimized_artifacts)?
            } else {
                String::new()
            },
            ms2rescore_fitted_artifacts: expert.ms2rescore_artifacts.clone(),
            ms2rescore_fitted_artifacts_sha256: expert
                .ms2rescore_artifacts
                .as_ref()
                .filter(|path| path.is_file())
                .map(|path| sha256_file(path))
                .transpose()?,
            calibration_stage: expert.calibration_stage.clone(),
            calibration_results: expert.calibration_results.clone(),
            target_only_results: expert.target_only_results.clone(),
            enabled,
            target_peptides: peptides.len(),
            incremental_target_peptides: incremental,
            gate_reasons: reasons,
            gate_warnings: warnings,
            fit_search_fingerprint: expert.calibration_search_fingerprint.clone(),
            candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
        });
    }
    let enabled = locked.iter().filter(|expert| expert.enabled).count();
    anyhow::ensure!(
        enabled >= manifest.validation.minimum_ensemble_experts,
        "only {enabled} experts passed Ensemble gates; {} required",
        manifest.validation.minimum_ensemble_experts
    );
    Ok(EnsembleLock {
        schema_version: 3,
        source_manifest_hash: manifest_hash.into(),
        dataset_fingerprint: dataset.fingerprint.clone(),
        experts: locked,
        minimum_required_experts: manifest.validation.minimum_ensemble_experts,
    })
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
) -> Result<Option<FittedArtifactProvenance>> {
    let path = output_directory.join("fitted_model_artifacts.json");
    if !path.is_file() {
        return Ok(None);
    }
    let mut artifacts: DfRunArtifacts = serde_json::from_slice(&std::fs::read(&path)?)
        .with_context(|| format!("invalid fitted artifacts {}", path.display()))?;
    let provenance = inherited.unwrap_or_else(|| {
        fitted_artifact_provenance(dataset, stage, model, fit_search_fingerprint)
    });
    artifacts.provenance = Some(provenance.clone());
    write_json_atomic(&path, &artifacts)?;
    Ok(Some(provenance))
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
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-workflow-stage-v5-explicit-target-calibration\0");
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
        hasher.update(generator_settings_sha256(&settings)?.as_bytes());
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
    Ok(format!("{:x}", hasher.finalize()))
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
    )?;
    let results = output_directory.join("results.sage.tsv");
    let config_snapshot = output_directory.join("workflow.search.resolved.json");
    let checkpoint = output_directory.join("workflow.stage.json");

    if manifest.resume && results.is_file() && checkpoint.is_file() {
        let old: StageRecord = serde_json::from_slice(&std::fs::read(&checkpoint)?)?;
        if old.input_hash == input_hash && old.status == "complete" {
            let cache_ready = if external {
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
                true
            };
            if cache_ready {
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
    if let Some(lock) = ensemble_lock {
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
        )?;
    }
    apply_window(fdr, &model.model, &model.window);
    if stage == "optimized" && !model.candidate_windows.is_empty() {
        fdr.null_window_optimizer = Some(NullWindowOptimizerOptions {
            candidates: model
                .candidate_windows
                .iter()
                .map(|window| NullWindowCandidate {
                    min_rank: window.min_rank,
                    max_rank: window.max_rank,
                })
                .collect(),
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
        apply_fitted_artifacts(fdr, &model.model, artifacts)?;
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
        stage: stage.into(),
        model: model_slug(&model.model).into(),
        input_hash,
        status: if plan_only { "planned" } else { "running" }.into(),
        results,
        config_snapshot: config_snapshot.clone(),
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
        ms2rescore_annotation_cache: None,
        target_only_calibration_policy: target_only.map(|context| context.policy),
        release_candidate: target_only.is_none_or(|context| context.release_candidate),
        window_provenance: target_only.map(|context| context.window_provenance.clone()),
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

    let candidate_pool = (matches!(stage, "optimized" | "ms2rescore") || target_only.is_some())
        .then(|| {
            let requested_by_model = model
                .candidate_windows
                .iter()
                .map(|window| window.max_rank as usize)
                .chain(model.window.iter().map(|window| window.max_rank as usize))
                .max()
                .unwrap_or(1);
            let requested_by_external = external.then_some(
                runner
                    .parameters
                    .external_features
                    .max_rank
                    .map(|rank| rank as usize)
                    .unwrap_or(runner.parameters.report_psms),
            );
            CandidatePoolRequest {
                root: manifest.output_root.join("candidate_pools"),
                required_rank_depth: requested_by_external
                    .map(|rank| rank.max(requested_by_model))
                    .unwrap_or(requested_by_model),
                allow_reuse: target_only
                    .map(|context| context.allow_candidate_pool_reuse)
                    .unwrap_or(true),
            }
        });
    let annotation_cache = external.then(|| ExternalAnnotationCacheRequest {
        root: manifest.output_root.join("ms2rescore_annotations"),
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
    )?;
    record.artifact_fit_dataset_fingerprint = stamped
        .as_ref()
        .map(|provenance| provenance.dataset_fingerprint.clone());
    record.status = "complete".into();
    write_json_atomic(&checkpoint, &record)?;
    Ok(record)
}

pub fn execute_workflow(
    manifest_path: &Path,
    source_repo: &Path,
    parallel: usize,
    plan_only: bool,
) -> Result<WorkflowState> {
    let mut manifest = WorkflowManifest::load(manifest_path)?;
    std::fs::create_dir_all(&manifest.output_root)?;
    let manifest_hash = sha256_file(manifest_path)?;
    let dataset = compute_dataset_identity(&manifest)?;
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
    let mut runtime = WorkflowRuntime::default();
    let mut ordered_models = manifest
        .models
        .iter()
        .filter(|model| model.enabled)
        .collect::<Vec<_>>();
    ordered_models.sort_by_key(|model| (model.model == ModelFit::Ensemble) as u8);
    for model in ordered_models {
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
                let lock =
                    serde_json::from_slice::<EnsembleLock>(&std::fs::read(path).with_context(
                        || format!("failed to read Ensemble lock {}", path.display()),
                    )?)?;
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
                build_ensemble_lock(&manifest, &manifest_hash, &dataset, &completed_experts)?
            };
            write_json_atomic(&manifest.output_root.join("ensemble.lock.json"), &lock)?;
            Some(lock)
        } else {
            None
        };
        let optimized = run_search_stage(
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
            &mut runtime,
        )?;
        stages.push(optimized.clone());

        let mut locked_model = model.clone();
        if !plan_only && !model.candidate_windows.is_empty() {
            let path = model_root.join("optimized/null_window_evaluations.json");
            let evaluations: Vec<sage_core::decoy_free_fdr::NullWindowEvaluation> =
                serde_json::from_slice(
                    &std::fs::read(&path)
                        .with_context(|| format!("optimizer did not produce {}", path.display()))?,
                )?;
            let selected = evaluations
                .iter()
                .find(|evaluation| evaluation.selected)
                .context("optimizer report has no selected window")?;
            locked_model.window = Some(NullWindow {
                min_rank: selected.min_rank,
                max_rank: selected.max_rank,
            });
            locked_model.candidate_windows.clear();
        }

        let mut ms2_record = None;
        if !matches!(locked_model.ms2rescore, Ms2RescorePolicy::Never) {
            let record = run_search_stage(
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
                &mut runtime,
            )?;
            stages.push(record.clone());
            ms2_record = Some(record);
        }
        let use_ms2_for_final = match locked_model.ms2rescore {
            Ms2RescorePolicy::Never => false,
            Ms2RescorePolicy::Always => true,
            Ms2RescorePolicy::Measure if plan_only => false,
            Ms2RescorePolicy::Measure => {
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
                match (
                    optimized_summary.iter().find(|row| row.layer == "level4"),
                    ms2_summary.iter().find(|row| row.layer == "level4"),
                ) {
                    (Some(before), Some(after)) => {
                        let gain = after.peptide.target.saturating_sub(before.peptide.target);
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
                            && fdp_increase <= locked_model.maximum_raw_fdp_increase.unwrap_or(0.0)
                    }
                    _ => false,
                }
            }
        };
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
            anyhow::bail!("fitted model artifacts were not produced by the entrapment search")
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
        for (index, (policy, release_candidate)) in target_policies.iter().copied().enumerate() {
            let context = TargetOnlyStageContext {
                policy,
                release_candidate,
                window_provenance: window_provenance.clone(),
                allow_candidate_pool_reuse: index > 0,
            };
            let target_only = run_search_stage(
                &manifest,
                &dataset,
                &locked_model,
                policy.stage_name(),
                &manifest.target_fasta,
                &model_root.join("target_only").join(match policy {
                    TargetOnlyCalibrationPolicy::RefitWithLockedWindow => {
                        "refit_with_locked_window"
                    }
                    TargetOnlyCalibrationPolicy::ReuseDatasetArtifact => "reuse_dataset_artifact",
                    TargetOnlyCalibrationPolicy::CompareBoth => unreachable!(),
                }),
                use_ms2_for_final,
                manifest.annotate_target_matches && index == 0,
                parallel,
                plan_only,
                (policy == TargetOnlyCalibrationPolicy::ReuseDatasetArtifact)
                    .then_some(frozen_model_artifacts)
                    .flatten(),
                ensemble_lock.as_ref(),
                Some(&context),
                &mut runtime,
            )?;
            if release_candidate {
                release_target_only = Some(target_only.clone());
            }
            stages.push(target_only);
        }
        let target_only =
            release_target_only.context("target-only policy has no release result")?;
        if model.model != ModelFit::Ensemble && !plan_only {
            anyhow::ensure!(
                frozen_model_artifacts.is_some(),
                "individual expert has no selected fitted artifact"
            );
            completed_experts.push(CompletedExpert {
                model: locked_model.model.clone(),
                window: locked_model.window.clone(),
                optimized_artifacts: optimized_artifact,
                ms2rescore_artifacts: ms2_artifact.is_file().then_some(ms2_artifact),
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
                calibration_search_fingerprint: calibration_record
                    .candidate_pool
                    .as_ref()
                    .context("calibration stage has no candidate-pool provenance")?
                    .search_fingerprint
                    .clone(),
            });
        }
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
    runs.extend(stages.iter().map(|stage| {
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
    let missing_runs = runs
        .iter()
        .filter(|run| !run.results.is_file())
        .cloned()
        .collect::<Vec<_>>();
    for run in &runs {
        validation.extend(summarize_run(
            run,
            &manifest.validation.effective_ratios,
            manifest.validation.fdr_threshold,
        )?);
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
    let mut parity = parity_comparisons(
        &validation,
        &manifest.validation.parity_pairs,
        manifest.validation.maximum_parity_fraction_difference,
    );
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
    let mut release_reasons = Vec::new();
    if manifest.validation.dataset_role != ValidationDatasetRole::Holdout {
        release_reasons.push("dataset is development, not holdout".into());
    }
    if manifest.validation.diagnostic_only
        || manifest.artifact_reuse_policy == ArtifactReusePolicy::CrossDatasetDiagnostic
    {
        release_reasons.push("workflow is diagnostic-only and cannot be release evidence".into());
    }
    if manifest.validation.parity_pairs.is_empty() {
        release_reasons.push("dataset-local baseline/native parity comparison is missing".into());
    } else if parity.is_empty() {
        release_reasons.push("baseline/native parity comparison is missing".into());
    } else if manifest.validation.parity_pairs.iter().any(|pair| {
        !parity.iter().any(|comparison| {
            comparison.baseline_method == pair.baseline_method
                && comparison.native_method == pair.native_method
        })
    }) {
        release_reasons.push("one or more declared parity pairs have no matched results".into());
    } else if parity.iter().any(|comparison| !comparison.within_tolerance) {
        release_reasons.push("one or more parity comparisons exceed tolerance".into());
    }
    if !missing_runs.is_empty() {
        release_reasons.push("required validation runs are missing".into());
    }
    if stability
        .iter()
        .any(|comparison| comparison.release_candidate && !comparison.stable)
    {
        release_reasons.push("one or more search-space transfers are unstable".into());
    }
    if manifest.validation.tdc_reference_method.is_none() || tdc_benchmarks.is_empty() {
        release_reasons.push("a matched TDC benchmark is missing".into());
    } else if !tdc_benchmarks
        .iter()
        .any(|comparison| comparison.release_candidate && comparison.improves_peptide_yield)
    {
        release_reasons.push(
            "no calibrated Decoy-Free result improves peptide yield over the matched TDC".into(),
        );
    }
    let ensemble_requested = manifest
        .models
        .iter()
        .any(|model| model.enabled && model.model == ModelFit::Ensemble);
    if ensemble_requested
        && ensemble_gates.iter().filter(|gate| gate.eligible).count()
            < manifest.validation.minimum_ensemble_experts
    {
        release_reasons.push("too few experts passed the Ensemble quality gates".into());
    }
    let calibrated_tdc_improvements = tdc_benchmarks
        .iter()
        .filter(|comparison| comparison.release_candidate && comparison.improves_peptide_yield)
        .count();
    let release_gate = ReleaseGate {
        eligible_for_statistical_default_change: release_reasons.is_empty(),
        reasons: release_reasons,
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
        &manifest
            .output_root
            .join("validation.ensemble_expert_gates.json"),
        &ensemble_gates,
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
    let state = WorkflowState {
        schema_version: 1,
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
        stage_comparisons: comparisons,
        ensemble_expert_gates: ensemble_gates,
        parity_comparisons: parity,
        tdc_benchmarks,
        release_gate,
        transfer_stability: stability,
        pending_validation_gates,
    };
    write_json_atomic(&manifest.output_root.join("workflow.state.json"), &state)?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sage_core::decoy_free_fdr::{DfRunArtifacts, FrozenModelMetadata};
    use sage_core::input::{
        ExternalEmpiricalFeatureProfile, ExternalMs2RescoreProfiles, FrozenGumbelParameters,
    };
    use sage_core::ml::msfdr::MsfdrSeededModel;
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
            schema_version: 1,
            model_version: "sage-external-ms2rescore-profiles-v1".into(),
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
                enabled: true,
                ms2rescore: Ms2RescorePolicy::Measure,
                maximum_raw_fdp_increase: None,
                minimum_level4_peptide_gain: None,
                target_only_calibration_policy: None,
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
            annotate_target_matches: false,
            ensemble_lock: None,
            locked_expert_artifacts: BTreeMap::new(),
            artifact_reuse_policy: ArtifactReusePolicy::DatasetLocalOnly,
            target_only_calibration_policy: TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
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
            enabled: true,
            ms2rescore: Ms2RescorePolicy::Measure,
            maximum_raw_fdp_increase: None,
            minimum_level4_peptide_gain: None,
            target_only_calibration_policy: None,
        });
        manifest.validate().unwrap();
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
        let restored: WorkflowManifest = serde_json::from_value(value).unwrap();
        assert_eq!(
            restored.target_only_calibration_policy,
            TargetOnlyCalibrationPolicy::RefitWithLockedWindow
        );
        assert_eq!(
            concrete_target_only_policies(TargetOnlyCalibrationPolicy::CompareBoth),
            vec![
                (TargetOnlyCalibrationPolicy::RefitWithLockedWindow, true),
                (TargetOnlyCalibrationPolicy::ReuseDatasetArtifact, false),
            ]
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
        let mut text = String::from(
            "psm_id\trank\tlabel\tproteins\tpeptide\tdecoy_free_q_value\tdecoy_free_peptide_q\tdecoy_free_protein_q\tdecoy_free_protein_supported_peptide\tdecoy_free_peptide_supported_psm\n",
        );
        for index in start..start + 700 {
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
        apply_fitted_artifacts(&mut fdr, &ModelFit::Moments, artifacts).unwrap();
        assert_eq!(fdr.moments_min_null_rank, Some(9));
        assert_eq!(fdr.moments_max_null_rank, Some(18));
        assert!(fdr.moments_frozen_parameters.is_some());
        assert!(fdr.external_ms2rescore_frozen_profiles.is_some());
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
        write_validation_tsv(&moments_calibration, 0, 3);
        write_validation_tsv(&moments_target, 0, 0);
        write_validation_tsv(&mle_calibration, 100, 3);
        write_validation_tsv(&mle_target, 100, 0);

        let model = |model, window| ModelWorkflow {
            model,
            window,
            candidate_windows: Vec::new(),
            enabled: true,
            ms2rescore: Ms2RescorePolicy::Never,
            maximum_raw_fdp_increase: None,
            minimum_level4_peptide_gain: None,
            target_only_calibration_policy: None,
        };
        let manifest = WorkflowManifest {
            schema_version: 1,
            name: "test".into(),
            dataset_id: Some("test-dataset".into()),
            search_config: PathBuf::new(),
            target_fasta: PathBuf::new(),
            spectra: vec!["test.mzML".into()],
            output_root: directory.clone(),
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
            annotate_target_matches: false,
            ensemble_lock: None,
            locked_expert_artifacts: BTreeMap::new(),
            artifact_reuse_policy: ArtifactReusePolicy::DatasetLocalOnly,
            target_only_calibration_policy: TargetOnlyCalibrationPolicy::RefitWithLockedWindow,
        };
        let dataset = DatasetIdentity {
            schema_version: 1,
            dataset_id: "test-dataset".into(),
            fingerprint: "dataset-fingerprint".into(),
            target_fasta_sha256: "target-sha256".into(),
            spectra_sha256: vec!["spectra-sha256".into()],
            search_config_sha256: "config-sha256".into(),
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
                ms2rescore_artifacts: None,
                calibration_stage: "optimized".into(),
                calibration_results: moments_calibration,
                target_only_results: moments_target,
                calibration_search_fingerprint: "test-search-fingerprint".into(),
            },
            CompletedExpert {
                model: ModelFit::Mle,
                window: manifest.models[1].window.clone(),
                optimized_artifacts: mle_artifact,
                ms2rescore_artifacts: None,
                calibration_stage: "optimized".into(),
                calibration_results: mle_calibration,
                target_only_results: mle_target,
                calibration_search_fingerprint: "test-search-fingerprint".into(),
            },
        ];
        let lock = build_ensemble_lock(&manifest, "manifest-hash", &dataset, &experts).unwrap();
        assert_eq!(lock.dataset_fingerprint, dataset.fingerprint);
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
        )
        .unwrap();
        assert_eq!(refit.moments_min_null_rank, Some(9));
        assert_eq!(refit.moments_max_null_rank, Some(18));
        assert!(refit.moments_frozen_parameters.is_none());
        assert!(refit.mle_frozen_parameters.is_none());

        let mut reuse = FdrOptions::default();
        apply_ensemble_lock(
            &mut reuse,
            &lock,
            false,
            &dataset,
            &ArtifactReusePolicy::DatasetLocalOnly,
            true,
        )
        .unwrap();
        assert!(reuse.moments_frozen_parameters.is_some());
        assert!(reuse.mle_frozen_parameters.is_some());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
