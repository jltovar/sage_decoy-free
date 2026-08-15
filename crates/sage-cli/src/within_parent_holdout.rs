//! Development-only, feature-gated within-parent run-level holdout runner.
//!
//! This module intentionally has no dependency on [`crate::runner::Runner`] or
//! the external-feature generation layer. It can only verify and read complete
//! immutable parent candidate pools and annotation caches, derive in-memory
//! views that preserve parent identifiers, and run statistical analysis.

use crate::candidate_pool::{
    load_verified_parent_entries, search_fingerprint, stable_candidate_id, CandidatePoolEntry,
    CANDIDATE_ID_SCHEMA,
};
use crate::external_feature_cache::{
    load_verified_parent_annotations, EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION,
    EXTERNAL_ANNOTATION_FEATURE_SCHEMA,
};
use crate::input::{ExternalFeatureUseMode, Input, Search};
use crate::provenance::{sha256_file, write_json_atomic};
use crate::workflow::NullWindow;
use anyhow::{Context, Result};
use sage_core::database::IndexedDatabase;
use sage_core::decoy_free_fdr::{
    apply_external_ms2rescore_bounded_experts, apply_hierarchical_reporting_df,
    apply_peptide_q_to_psm_reporting_df, calculate_peptide_q_df, calculate_protein_q_df,
    optimize_null_window, DfRunArtifacts, NullWindowEvaluation,
};
use sage_core::input::{
    AdaptiveNullWindowSearchOptions, FdrSettings, ModelFit, NullWindowCandidate,
    NullWindowOptimizerOptions, NullWindowSearchStrategy, NullWindowValidationScope,
};
use sage_core::scoring::DfFeature;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SUBSET_SCHEMA: &str = "within-parent-subset-v1";
pub const ARTIFACT_SCHEMA: &str = "within-parent-holdout-artifact-v1";
pub const LOCK_SCHEMA: &str = "within-parent-holdout-lock-v2";
pub const PREREGISTRATION_SCHEMA: &str = "within-parent-holdout-preregistration-v2";
pub const PREFLIGHT_SCHEMA: &str = "within-parent-holdout-preflight-v2";
pub const RESULT_SCHEMA: &str = "within-parent-holdout-result-v2";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchSpace {
    PlusEntrapment,
    TargetOnly,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SubsetRole {
    Training,
    HeldOut,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceBuildIdentity {
    pub source_commit: String,
    pub source_tree_sha256: String,
    pub cargo_lock_sha256: String,
    pub release_binary_sha256: String,
    pub crate_version: String,
    pub rustc_version: String,
    pub cargo_version: String,
    pub target_triple: String,
    pub build_profile: String,
    pub enabled_features: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpectrumFileIdentity {
    pub ordinal: usize,
    pub filename: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParentSpaceIdentity {
    pub search_space: SearchSpace,
    pub fasta: PathBuf,
    pub fasta_sha256: String,
    pub search_config: PathBuf,
    pub search_config_sha256: String,
    pub pool_manifest: PathBuf,
    pub pool_manifest_sha256: String,
    pub search_fingerprint: String,
    pub pool_payload_sha256: String,
    pub candidate_count: usize,
    pub retained_rank_depth: usize,
    pub candidate_id_schema: String,
    pub annotation_manifest: PathBuf,
    pub annotation_manifest_sha256: String,
    pub annotation_fingerprint: String,
    pub annotation_payload_sha256: String,
    pub annotation_count: usize,
    pub annotation_schema_version: u32,
    pub annotation_feature_schema: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HoldoutFold {
    pub fold: usize,
    pub training_file_ids: Vec<usize>,
    pub held_out_file_ids: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelGrid {
    pub model: ModelFit,
    pub candidates: Vec<NullWindow>,
    #[serde(default)]
    pub fixed_window: Option<NullWindow>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HoldoutComposition {
    pub id: String,
    pub requested_experts: Vec<ModelFit>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HoldoutRuntimeGates {
    pub validation_scope: NullWindowValidationScope,
    pub fdr_threshold: f64,
    pub maximum_target_only_peptide_fraction_loss: f64,
    pub minimum_incremental_level4_target_peptides: usize,
    pub minimum_entrapment_peptides_for_stable_estimate: usize,
    pub minimum_ensemble_experts: usize,
    pub raw_q_interaction_warning_threshold: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HoldoutAcceptanceCriteria {
    pub require_all_folds_valid: bool,
    pub require_no_fallback: bool,
    pub require_positive_aggregate_target_evidence: bool,
    pub require_no_material_calibration_deterioration: bool,
    pub require_stable_target_only_transfer: bool,
    pub require_not_single_fold_driven: bool,
    pub fdr_threshold: f64,
    pub confidence_level: f64,
    pub maximum_absolute_ratio_adjusted_peptide_fdp_increase: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HoldoutPreregistration {
    pub schema: String,
    pub study: String,
    pub assignment_basis: String,
    pub parent_dataset_id: String,
    pub parent_dataset_fingerprint: String,
    pub spectra: Vec<SpectrumFileIdentity>,
    pub plus_entrapment: ParentSpaceIdentity,
    pub target_only: ParentSpaceIdentity,
    pub folds: Vec<HoldoutFold>,
    pub grid_source_description: String,
    pub grid_source_manifest_sha256: String,
    pub model_grids: Vec<ModelGrid>,
    pub baseline_experts: Vec<ModelFit>,
    pub comparison_matrix: Vec<HoldoutComposition>,
    pub target_only_policies: BTreeMap<String, String>,
    pub external_profile_window: NullWindow,
    pub effective_ratios: EffectiveRatios,
    pub optimizer_validation_scope: NullWindowValidationScope,
    pub optimizer_seed: u64,
    pub runtime_gates: HoldoutRuntimeGates,
    pub disclosures: Vec<String>,
    pub source_build: SourceBuildIdentity,
    pub aggregation_definition: String,
    pub uncertainty_method: String,
    pub acceptance: HoldoutAcceptanceCriteria,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EffectiveRatios {
    pub psm: f64,
    pub peptide: f64,
    pub protein: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WithinParentSubsetIdentity {
    pub schema: String,
    pub digest: String,
    pub parent_dataset_fingerprint: String,
    pub parent_search_fingerprint: String,
    pub parent_pool_payload_sha256: String,
    pub parent_annotation_manifest_sha256: String,
    pub parent_annotation_payload_sha256: String,
    pub fold_manifest_sha256: String,
    pub fold: usize,
    pub role: SubsetRole,
    pub search_space: SearchSpace,
    pub spectrum_files: Vec<PortableSpectrumIdentity>,
    pub selected_file_ids: Vec<usize>,
    pub ordered_stable_candidate_ids_sha256: String,
    pub selected_spectrum_count: usize,
    pub selected_candidate_count: usize,
    pub candidate_id_schema: String,
    pub annotation_schema_version: u32,
    pub annotation_feature_schema: String,
    pub source_build: SourceBuildIdentity,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct PortableSpectrumIdentity {
    pub ordinal: usize,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HoldoutPreflight {
    pub schema: String,
    pub manifest_sha256: String,
    pub parent_dataset_fingerprint: String,
    pub plus_entrapment_subsets: Vec<WithinParentSubsetIdentity>,
    pub target_only_subsets: Vec<WithinParentSubsetIdentity>,
    pub folds_are_disjoint_and_complete: bool,
    pub candidate_and_annotation_joins_complete: bool,
    pub capabilities: ValidationRunnerCapabilities,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnotationAuditEntry {
    pub label: String,
    pub manifest: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnotationAuditGroup {
    pub label: String,
    pub reference: AnnotationAuditEntry,
    pub comparisons: Vec<AnnotationAuditEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnotationAuditRequest {
    pub schema: String,
    pub groups: Vec<AnnotationAuditGroup>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnnotationFieldDifference {
    pub field: String,
    pub differing_values: usize,
    pub maximum_finite_absolute_difference: f64,
    pub nonfinite_mismatches: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnotationCacheInventory {
    pub label: String,
    pub manifest: PathBuf,
    pub manifest_sha256: String,
    pub annotation_fingerprint: String,
    pub search_fingerprint: String,
    pub generator_settings_sha256: String,
    pub calibration_input_sha256: String,
    pub payload_sha256: String,
    pub annotation_count: usize,
    pub requested_max_rank: u32,
    pub model_components: Vec<crate::external_feature_cache::ModelComponentIdentity>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnotationCacheComparison {
    pub reference: String,
    pub comparison: String,
    pub joined_stable_ids: usize,
    pub reference_only_stable_ids: usize,
    pub comparison_only_stable_ids: usize,
    pub duplicate_stable_ids: usize,
    pub exact_raw_feature_payload: bool,
    pub field_differences: Vec<AnnotationFieldDifference>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnnotationAuditResult {
    pub schema: String,
    pub capabilities: ValidationRunnerCapabilities,
    pub inventories: Vec<AnnotationCacheInventory>,
    pub comparisons: Vec<AnnotationCacheComparison>,
    pub conclusion: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationRunnerCapabilities {
    pub spectrum_search: bool,
    pub annotation_generation: bool,
    pub python_execution: bool,
    pub ms2pip_execution: bool,
    pub deeplc_execution: bool,
    pub wrapper_execution: bool,
}

pub const CAPABILITIES: ValidationRunnerCapabilities = ValidationRunnerCapabilities {
    spectrum_search: false,
    annotation_generation: false,
    python_execution: false,
    ms2pip_execution: false,
    deeplc_execution: false,
    wrapper_execution: false,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WithinParentHoldoutArtifact {
    pub schema: String,
    pub digest: String,
    pub manifest_sha256: String,
    pub training_subset_digest: String,
    pub parent_dataset_fingerprint: String,
    pub model: ModelFit,
    pub selected_window: Option<NullWindow>,
    pub fitted_payload_sha256: String,
    pub nuisance_state_provenance: String,
    pub complete_artifact_transfer_allowed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HoldoutLockExpert {
    pub model: ModelFit,
    pub selected_window: Option<NullWindow>,
    pub training_artifact_digest: String,
    pub enabled: bool,
    pub participation_reason: String,
    pub gate_reasons: Vec<String>,
    pub gate_warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WithinParentHoldoutLock {
    pub schema: String,
    pub digest: String,
    pub manifest_sha256: String,
    pub parent_dataset_fingerprint: String,
    pub training_subset_digest: String,
    pub fold: usize,
    pub composition: String,
    pub experts: Vec<HoldoutLockExpert>,
    pub external_profile_window: NullWindow,
    pub target_only_policy: String,
    pub production_evidence: bool,
}

#[derive(Clone)]
struct ParentContext {
    identity: ParentSpaceIdentity,
    search: Search,
    database: Arc<IndexedDatabase>,
}

#[derive(Clone, Debug)]
struct ViewRecord {
    stable_id: String,
    file_id: usize,
    spec_id: String,
    core: sage_core::scoring::FeatureCore,
}

#[derive(Clone, Debug)]
struct VerifiedSubset {
    identity: WithinParentSubsetIdentity,
    features: Vec<DfFeature>,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn hash_serialized<T: Serialize>(value: &T) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(value)?))
}

fn parent_dataset_fingerprint(
    target_fasta_sha256: &str,
    spectra: &[SpectrumFileIdentity],
) -> String {
    let mut hashes = spectra
        .iter()
        .map(|run| run.sha256.as_str())
        .collect::<Vec<_>>();
    hashes.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(b"sage-decoy-free-dataset-v1\0");
    hasher.update(target_fasta_sha256.as_bytes());
    for digest in hashes {
        hasher.update(b"\0");
        hasher.update(digest.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn validate_fold_assignments(folds: &[HoldoutFold], run_count: usize) -> Result<()> {
    anyhow::ensure!(folds.len() == 3, "ISB holdout requires exactly three folds");
    let all = (0..run_count).collect::<BTreeSet<_>>();
    let mut held_union = BTreeSet::new();
    for (expected_fold, fold) in (1..=3).zip(folds) {
        anyhow::ensure!(
            fold.fold == expected_fold,
            "folds must be canonically ordered"
        );
        let training = fold
            .training_file_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let held = fold
            .held_out_file_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        anyhow::ensure!(
            training.len() + held.len() == run_count && held.len() == run_count / 3,
            "fold has an invalid training/held-out size"
        );
        anyhow::ensure!(
            training.len() == fold.training_file_ids.len(),
            "duplicate training run"
        );
        anyhow::ensure!(
            held.len() == fold.held_out_file_ids.len(),
            "duplicate held-out run"
        );
        anyhow::ensure!(
            training.is_disjoint(&held),
            "training and held-out runs overlap"
        );
        anyhow::ensure!(
            training.union(&held).copied().collect::<BTreeSet<_>>() == all,
            "fold is not a complete parent partition"
        );
        for file_id in held {
            anyhow::ensure!(held_union.insert(file_id), "held-out folds overlap");
        }
    }
    anyhow::ensure!(
        held_union == all,
        "held-out folds do not cover all parent runs exactly once"
    );
    Ok(())
}

fn validate_preregistration(manifest: &HoldoutPreregistration) -> Result<()> {
    anyhow::ensure!(
        manifest.schema == PREREGISTRATION_SCHEMA,
        "unsupported within-parent preregistration schema"
    );
    anyhow::ensure!(
        manifest.spectra.len() == 9,
        "ISB holdout requires exactly nine runs"
    );
    anyhow::ensure!(
        manifest.folds.len() == 3,
        "ISB holdout requires exactly three folds"
    );
    anyhow::ensure!(
        manifest.assignment_basis
            == "stable acquisition order round-robin; no identification, score, entrapment, or model outcome used",
        "fold assignment basis is not the preregistered acquisition-only rule"
    );
    anyhow::ensure!(
        manifest.plus_entrapment.search_space == SearchSpace::PlusEntrapment
            && manifest.target_only.search_space == SearchSpace::TargetOnly,
        "parent search-space roles are invalid"
    );
    anyhow::ensure!(
        manifest.external_profile_window.min_rank == 9
            && manifest.external_profile_window.max_rank == 18,
        "within-parent external profile must be explicitly locked to 9..=18"
    );
    anyhow::ensure!(
        manifest.baseline_experts
            == vec![
                ModelFit::Moments,
                ModelFit::Mle,
                ModelFit::Msfdr1Smix,
                ModelFit::LowerOrder,
            ],
        "baseline expert set must be the current production candidates: Moments + MLE + MSFDR1-SMIX + Lower Order"
    );
    let expected_matrix = vec![
        HoldoutComposition {
            id: "A".into(),
            requested_experts: manifest.baseline_experts.clone(),
        },
        HoldoutComposition {
            id: "B".into(),
            requested_experts: manifest
                .baseline_experts
                .iter()
                .cloned()
                .chain([ModelFit::Msfdr])
                .collect(),
        },
        HoldoutComposition {
            id: "C".into(),
            requested_experts: manifest
                .baseline_experts
                .iter()
                .cloned()
                .chain([ModelFit::Msfdr2Smix])
                .collect(),
        },
        HoldoutComposition {
            id: "D".into(),
            requested_experts: manifest
                .baseline_experts
                .iter()
                .cloned()
                .chain([ModelFit::Msfdr, ModelFit::Msfdr2Smix])
                .collect(),
        },
    ];
    anyhow::ensure!(
        manifest.comparison_matrix == expected_matrix,
        "comparison matrix must be the preregistered A-D current-baseline/MSFDR/MSFDR2 design"
    );
    anyhow::ensure!(
        !manifest
            .model_grids
            .iter()
            .any(|grid| matches!(grid.model, ModelFit::Nokoi | ModelFit::Ensemble)),
        "Nokoi and nested Ensemble models are prohibited from this holdout"
    );
    anyhow::ensure!(
        manifest
            .target_only_policies
            .values()
            .all(|policy| policy == "refit_with_locked_window"),
        "every holdout target-only expert must use refit_with_locked_window"
    );
    anyhow::ensure!(
        manifest.acceptance.fdr_threshold == 0.01
            && manifest.acceptance.confidence_level == 0.95
            && manifest
                .acceptance
                .maximum_absolute_ratio_adjusted_peptide_fdp_increase
                == 0.01,
        "holdout acceptance thresholds differ from the preregistered contract"
    );
    anyhow::ensure!(
        manifest.runtime_gates.validation_scope == NullWindowValidationScope::Level4
            && manifest.runtime_gates.fdr_threshold == 0.01
            && manifest
                .runtime_gates
                .maximum_target_only_peptide_fraction_loss
                == 0.20
            && manifest
                .runtime_gates
                .minimum_incremental_level4_target_peptides
                == 1
            && manifest
                .runtime_gates
                .minimum_entrapment_peptides_for_stable_estimate
                == 3
            && manifest.runtime_gates.minimum_ensemble_experts == 2
            && manifest.runtime_gates.raw_q_interaction_warning_threshold == 0.01,
        "training-side runtime gates differ from current production policy"
    );
    anyhow::ensure!(
        manifest
            .disclosures
            .iter()
            .any(|item| item.contains("reused from the Lower Order"))
            && manifest
                .disclosures
                .iter()
                .any(|item| item.contains("Lower Order behavior has previously been observed"))
            && manifest.disclosures.iter().any(|item| {
                item.contains("MSFDR/MSFDR2 fold-level Ensemble outcomes have not been evaluated")
            })
            && manifest
                .disclosures
                .iter()
                .any(|item| item.contains("within-dataset run-level validation")),
        "required holdout disclosures are incomplete"
    );

    let expected_models = [
        ModelFit::Moments,
        ModelFit::Mle,
        ModelFit::LowerOrder,
        ModelFit::Msfdr,
        ModelFit::Msfdr1Smix,
        ModelFit::Msfdr2Smix,
    ]
    .into_iter()
    .map(|model| model_slug(&model))
    .collect::<BTreeSet<_>>();
    let observed_models = manifest
        .model_grids
        .iter()
        .map(|grid| model_slug(&grid.model))
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        observed_models == expected_models && manifest.model_grids.len() == expected_models.len(),
        "holdout model grids are incomplete or duplicated"
    );
    anyhow::ensure!(
        manifest
            .target_only_policies
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == expected_models,
        "target-only policy map does not cover the preregistered expert set exactly"
    );
    let mut maximum_rank = 1_u32;
    for grid in &manifest.model_grids {
        if grid.model == ModelFit::Msfdr1Smix {
            let fixed = grid
                .fixed_window
                .as_ref()
                .context("MSFDR1-SMIX fixed window is missing")?;
            anyhow::ensure!(
                grid.candidates.is_empty() && fixed.min_rank == 1 && fixed.max_rank == 1,
                "MSFDR1-SMIX must remain fixed at rank 1"
            );
        } else {
            anyhow::ensure!(
                grid.fixed_window.is_none() && !grid.candidates.is_empty(),
                "optimized expert grid is missing"
            );
            let unique = grid
                .candidates
                .iter()
                .map(|window| (window.min_rank, window.max_rank))
                .collect::<BTreeSet<_>>();
            anyhow::ensure!(
                unique.len() == grid.candidates.len(),
                "expert grid contains duplicate windows"
            );
            for window in &grid.candidates {
                anyhow::ensure!(
                    window.min_rank >= 1 && window.max_rank >= window.min_rank,
                    "invalid expert window"
                );
                maximum_rank = maximum_rank.max(window.max_rank);
            }
        }
    }
    maximum_rank = maximum_rank.max(manifest.external_profile_window.max_rank);
    anyhow::ensure!(
        manifest.plus_entrapment.retained_rank_depth >= maximum_rank as usize
            && manifest.target_only.retained_rank_depth >= maximum_rank as usize,
        "parent pool rank depth cannot satisfy the preregistered grid"
    );

    let mut ordinals = BTreeSet::new();
    for run in &manifest.spectra {
        anyhow::ensure!(ordinals.insert(run.ordinal), "duplicate parent run ordinal");
        anyhow::ensure!(
            run.path
                .file_name()
                .is_some_and(|name| name == run.filename.as_str()),
            "portable run filename does not match its source path"
        );
        anyhow::ensure!(
            run.path.is_file(),
            "spectrum is missing: {}",
            run.path.display()
        );
        anyhow::ensure!(
            sha256_file(&run.path)? == run.sha256,
            "spectrum content hash mismatch: {}",
            run.path.display()
        );
        anyhow::ensure!(
            run.path.metadata()?.len() == run.size_bytes,
            "spectrum size mismatch: {}",
            run.path.display()
        );
    }
    anyhow::ensure!(
        ordinals == (0..9).collect::<BTreeSet<_>>(),
        "parent run ordinals must be exactly 0..=8"
    );
    anyhow::ensure!(
        parent_dataset_fingerprint(&manifest.target_only.fasta_sha256, &manifest.spectra)
            == manifest.parent_dataset_fingerprint,
        "parent dataset fingerprint mismatch"
    );

    validate_fold_assignments(&manifest.folds, 9)?;
    Ok(())
}

fn verify_running_build(identity: &SourceBuildIdentity) -> Result<()> {
    anyhow::ensure!(
        identity.crate_version == env!("CARGO_PKG_VERSION"),
        "validation-runner crate version mismatch"
    );
    anyhow::ensure!(
        identity.build_profile == "release"
            && identity.enabled_features == vec!["within-parent-holdout"],
        "validation runner was not preregistered as the feature-gated release build"
    );
    let executable = std::env::current_exe().context("resolving validation-runner executable")?;
    anyhow::ensure!(
        sha256_file(&executable)? == identity.release_binary_sha256,
        "validation-runner binary hash mismatch"
    );
    let cargo_lock = std::env::current_dir()?.join("Cargo.lock");
    anyhow::ensure!(
        cargo_lock.is_file(),
        "Cargo.lock is not available from the validation working directory"
    );
    anyhow::ensure!(
        sha256_file(&cargo_lock)? == identity.cargo_lock_sha256,
        "validation-runner Cargo.lock hash mismatch"
    );
    Ok(())
}

fn verify_payload_hash(path: &Path, expected: &str, label: &str) -> Result<()> {
    anyhow::ensure!(path.is_file(), "{label} is missing: {}", path.display());
    anyhow::ensure!(sha256_file(path)? == expected, "{label} hash mismatch");
    Ok(())
}

fn verify_parent_identity(identity: &ParentSpaceIdentity) -> Result<()> {
    anyhow::ensure!(identity.fasta.is_file(), "parent FASTA is missing");
    anyhow::ensure!(
        sha256_file(&identity.fasta)? == identity.fasta_sha256,
        "parent FASTA hash mismatch"
    );
    anyhow::ensure!(
        identity.search_config.is_file(),
        "parent search configuration is missing"
    );
    anyhow::ensure!(
        sha256_file(&identity.search_config)? == identity.search_config_sha256,
        "parent search configuration hash mismatch"
    );
    anyhow::ensure!(
        identity.pool_manifest.is_file(),
        "parent pool manifest is missing"
    );
    anyhow::ensure!(
        sha256_file(&identity.pool_manifest)? == identity.pool_manifest_sha256,
        "parent pool manifest hash mismatch"
    );
    anyhow::ensure!(
        identity.annotation_manifest.is_file(),
        "parent annotation manifest is missing"
    );
    anyhow::ensure!(
        sha256_file(&identity.annotation_manifest)? == identity.annotation_manifest_sha256,
        "parent annotation manifest hash mismatch"
    );
    let pool_manifest: crate::candidate_pool::CandidatePoolManifest =
        serde_json::from_slice(&std::fs::read(&identity.pool_manifest)?)?;
    let pool_payload = identity
        .pool_manifest
        .parent()
        .context("pool manifest has no directory")?
        .join(&pool_manifest.payload_file);
    verify_payload_hash(
        &pool_payload,
        &identity.pool_payload_sha256,
        "parent pool payload",
    )?;
    let annotation_manifest: crate::external_feature_cache::ExternalAnnotationCacheManifest =
        serde_json::from_slice(&std::fs::read(&identity.annotation_manifest)?)?;
    let annotation_payload = identity
        .annotation_manifest
        .parent()
        .context("annotation manifest has no directory")?
        .join(&annotation_manifest.payload_file);
    verify_payload_hash(
        &annotation_payload,
        &identity.annotation_payload_sha256,
        "parent annotation payload",
    )?;
    anyhow::ensure!(
        identity.candidate_id_schema == CANDIDATE_ID_SCHEMA,
        "candidate-ID schema mismatch"
    );
    anyhow::ensure!(
        identity.annotation_schema_version == EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION,
        "annotation schema mismatch"
    );
    anyhow::ensure!(
        identity.annotation_feature_schema == EXTERNAL_ANNOTATION_FEATURE_SCHEMA,
        "annotation feature schema mismatch"
    );
    anyhow::ensure!(
        identity.annotation_count == identity.candidate_count,
        "parent candidate/annotation counts differ"
    );
    Ok(())
}

fn build_parent_context(identity: &ParentSpaceIdentity, scratch: &Path) -> Result<ParentContext> {
    verify_parent_identity(identity)?;
    let mut input = Input::load(identity.search_config.to_string_lossy().as_ref())?;
    input.database.fasta = Some(identity.fasta.display().to_string());
    input.output_directory = Some(scratch.display().to_string());
    let search = input.build()?;
    anyhow::ensure!(
        !search.database.prefilter,
        "within-parent validation refuses prefiltering because it could require spectra"
    );
    let fingerprint = search_fingerprint(&search)?;
    anyhow::ensure!(
        fingerprint.digest == identity.search_fingerprint,
        "resolved parent search fingerprint mismatch"
    );
    let fasta_url = sage_cloudpath::to_url(&search.database.fasta)?;
    let fasta = sage_cloudpath::util::read_fasta(
        &fasta_url,
        &search.database.decoy_tag,
        search.database.generate_decoys,
    )?;
    let database = Arc::new(search.database.clone().build(fasta));
    Ok(ParentContext {
        identity: identity.clone(),
        search,
        database,
    })
}

fn portable_spectra(
    manifest: &HoldoutPreregistration,
    selected_file_ids: &[usize],
) -> Result<Vec<PortableSpectrumIdentity>> {
    selected_file_ids
        .iter()
        .map(|file_id| {
            let run = manifest
                .spectra
                .iter()
                .find(|run| run.ordinal == *file_id)
                .with_context(|| format!("unknown selected file_id {file_id}"))?;
            Ok(PortableSpectrumIdentity {
                ordinal: run.ordinal,
                filename: run.filename.clone(),
                size_bytes: run.size_bytes,
                sha256: run.sha256.clone(),
            })
        })
        .collect()
}

fn subset_identity(
    manifest: &HoldoutPreregistration,
    parent: &ParentSpaceIdentity,
    manifest_sha256: &str,
    fold: usize,
    role: SubsetRole,
    selected_file_ids: &[usize],
    records: &[ViewRecord],
) -> Result<WithinParentSubsetIdentity> {
    let selected = selected_file_ids.iter().copied().collect::<BTreeSet<_>>();
    let mut ids_hasher = Sha256::new();
    ids_hasher.update(b"within-parent-ordered-candidate-ids-v1\0");
    let mut spectra = BTreeSet::new();
    for record in records
        .iter()
        .filter(|record| selected.contains(&record.file_id))
    {
        ids_hasher.update(record.stable_id.as_bytes());
        ids_hasher.update(b"\0");
        spectra.insert((record.file_id, record.spec_id.as_str()));
    }
    let selected_candidate_count = records
        .iter()
        .filter(|record| selected.contains(&record.file_id))
        .count();
    anyhow::ensure!(
        selected_candidate_count > 0,
        "within-parent subset is empty"
    );
    let mut identity = WithinParentSubsetIdentity {
        schema: SUBSET_SCHEMA.into(),
        digest: String::new(),
        parent_dataset_fingerprint: manifest.parent_dataset_fingerprint.clone(),
        parent_search_fingerprint: parent.search_fingerprint.clone(),
        parent_pool_payload_sha256: parent.pool_payload_sha256.clone(),
        parent_annotation_manifest_sha256: parent.annotation_manifest_sha256.clone(),
        parent_annotation_payload_sha256: parent.annotation_payload_sha256.clone(),
        fold_manifest_sha256: manifest_sha256.into(),
        fold,
        role,
        search_space: parent.search_space,
        spectrum_files: portable_spectra(manifest, selected_file_ids)?,
        selected_file_ids: selected_file_ids.to_vec(),
        ordered_stable_candidate_ids_sha256: format!("{:x}", ids_hasher.finalize()),
        selected_spectrum_count: spectra.len(),
        selected_candidate_count,
        candidate_id_schema: parent.candidate_id_schema.clone(),
        annotation_schema_version: parent.annotation_schema_version,
        annotation_feature_schema: parent.annotation_feature_schema.clone(),
        source_build: manifest.source_build.clone(),
    };
    identity.digest = hash_serialized(&identity)?;
    Ok(identity)
}

fn verify_join_and_records(
    entries: Vec<CandidatePoolEntry>,
    annotations: Vec<crate::external_feature_cache::ExternalAnnotationRecord>,
) -> Result<Vec<ViewRecord>> {
    anyhow::ensure!(
        entries.len() == annotations.len(),
        "parent candidate/annotation payload counts differ"
    );
    let mut annotation_map = HashMap::with_capacity(annotations.len());
    for annotation in annotations {
        anyhow::ensure!(
            annotation.features.ms2rescore_feature_joined,
            "parent annotation is not joined"
        );
        anyhow::ensure!(
            annotation_map
                .insert(annotation.stable_id, annotation.features)
                .is_none(),
            "duplicate stable candidate ID in parent annotations"
        );
    }
    let mut candidate_ids = HashSet::with_capacity(entries.len());
    let mut records = Vec::with_capacity(entries.len());
    for mut entry in entries {
        anyhow::ensure!(
            candidate_ids.insert(entry.stable_id.clone()),
            "duplicate stable candidate ID in parent candidates"
        );
        let features = annotation_map.remove(&entry.stable_id).with_context(|| {
            format!(
                "missing annotation for parent candidate {}",
                entry.stable_id
            )
        })?;
        entry.core.external_features = features;
        records.push(ViewRecord {
            stable_id: entry.stable_id,
            file_id: entry.core.file_id,
            spec_id: entry.core.spec_id.clone(),
            core: entry.core,
        });
    }
    anyhow::ensure!(
        annotation_map.is_empty(),
        "annotation cache contains IDs absent from parent candidates"
    );
    Ok(records)
}

fn load_complete_parent_records(context: &ParentContext) -> Result<Vec<ViewRecord>> {
    let pool_dir = context
        .identity
        .pool_manifest
        .parent()
        .context("pool manifest has no directory")?;
    let (_, entries) = load_verified_parent_entries(
        pool_dir,
        &context.identity.search_fingerprint,
        &context.identity.pool_payload_sha256,
        context.identity.candidate_count,
        context.identity.retained_rank_depth,
        &context.database,
    )?;
    let annotation_dir = context
        .identity
        .annotation_manifest
        .parent()
        .context("annotation manifest has no directory")?;
    let (_, annotations) = load_verified_parent_annotations(
        annotation_dir,
        &context.identity.annotation_fingerprint,
        &context.identity.search_fingerprint,
        &context.identity.annotation_payload_sha256,
        context.identity.annotation_count,
    )?;
    verify_join_and_records(entries, annotations)
}

fn derive_subset(
    manifest: &HoldoutPreregistration,
    context: &ParentContext,
    manifest_sha256: &str,
    fold: usize,
    role: SubsetRole,
    selected_file_ids: &[usize],
) -> Result<VerifiedSubset> {
    let records = load_complete_parent_records(context)?;
    let identity = subset_identity(
        manifest,
        &context.identity,
        manifest_sha256,
        fold,
        role,
        selected_file_ids,
        &records,
    )?;
    let features = materialize_subset_features(records, selected_file_ids);
    let selected = selected_file_ids.iter().copied().collect::<BTreeSet<_>>();
    anyhow::ensure!(
        features.len() == identity.selected_candidate_count,
        "subset materialization count mismatch"
    );
    anyhow::ensure!(
        features
            .iter()
            .all(|feature| selected.contains(&feature.core.file_id)),
        "full-parent candidate leaked into subset"
    );
    Ok(VerifiedSubset { identity, features })
}

fn materialize_subset_features(
    records: Vec<ViewRecord>,
    selected_file_ids: &[usize],
) -> Vec<DfFeature> {
    let selected = selected_file_ids.iter().copied().collect::<BTreeSet<_>>();
    records
        .into_iter()
        .filter(|record| selected.contains(&record.file_id))
        .map(|record| record.core.to_df())
        .collect()
}

fn verify_parent_run_coverage(records: &[ViewRecord], run_count: usize) -> Result<()> {
    let observed = records
        .iter()
        .map(|record| record.file_id)
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        observed == (0..run_count).collect::<BTreeSet<_>>(),
        "parent candidates do not cover exactly the preregistered run ordinals"
    );
    Ok(())
}

fn preflight_for_manifest(
    manifest: &HoldoutPreregistration,
    manifest_sha256: &str,
    scratch: &Path,
) -> Result<HoldoutPreflight> {
    validate_preregistration(manifest)?;
    verify_parent_identity(&manifest.plus_entrapment)?;
    verify_parent_identity(&manifest.target_only)?;
    let plus = build_parent_context(&manifest.plus_entrapment, &scratch.join("plus"))?;
    let target = build_parent_context(&manifest.target_only, &scratch.join("target"))?;
    let plus_records = load_complete_parent_records(&plus)?;
    let target_records = load_complete_parent_records(&target)?;
    verify_parent_run_coverage(&plus_records, manifest.spectra.len())?;
    verify_parent_run_coverage(&target_records, manifest.spectra.len())?;
    let mut plus_subsets = Vec::new();
    let mut target_subsets = Vec::new();
    for fold in &manifest.folds {
        plus_subsets.push(subset_identity(
            manifest,
            &manifest.plus_entrapment,
            manifest_sha256,
            fold.fold,
            SubsetRole::Training,
            &fold.training_file_ids,
            &plus_records,
        )?);
        plus_subsets.push(subset_identity(
            manifest,
            &manifest.plus_entrapment,
            manifest_sha256,
            fold.fold,
            SubsetRole::HeldOut,
            &fold.held_out_file_ids,
            &plus_records,
        )?);
        target_subsets.push(subset_identity(
            manifest,
            &manifest.target_only,
            manifest_sha256,
            fold.fold,
            SubsetRole::Training,
            &fold.training_file_ids,
            &target_records,
        )?);
        target_subsets.push(subset_identity(
            manifest,
            &manifest.target_only,
            manifest_sha256,
            fold.fold,
            SubsetRole::HeldOut,
            &fold.held_out_file_ids,
            &target_records,
        )?);
    }
    Ok(HoldoutPreflight {
        schema: PREFLIGHT_SCHEMA.into(),
        manifest_sha256: manifest_sha256.into(),
        parent_dataset_fingerprint: manifest.parent_dataset_fingerprint.clone(),
        plus_entrapment_subsets: plus_subsets,
        target_only_subsets: target_subsets,
        folds_are_disjoint_and_complete: true,
        candidate_and_annotation_joins_complete: true,
        capabilities: CAPABILITIES,
    })
}

pub fn preflight_holdout(manifest_path: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<()> {
    let manifest_path = manifest_path.as_ref();
    let bytes = std::fs::read(manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&bytes);
    let manifest: HoldoutPreregistration =
        serde_json::from_slice(&bytes).context("invalid holdout preregistration")?;
    verify_running_build(&manifest.source_build)?;
    let output = output.as_ref();
    std::fs::create_dir_all(output)?;
    let preflight = preflight_for_manifest(&manifest, &manifest_sha256, &output.join("scratch"))?;
    write_json_atomic(
        &output.join("within_parent_holdout.preflight.json"),
        &preflight,
    )
}

fn load_annotation_audit_entry(
    entry: &AnnotationAuditEntry,
) -> Result<(
    AnnotationCacheInventory,
    Vec<crate::external_feature_cache::ExternalAnnotationRecord>,
)> {
    let manifest: crate::external_feature_cache::ExternalAnnotationCacheManifest =
        serde_json::from_slice(&std::fs::read(&entry.manifest).with_context(|| {
            format!("reading annotation manifest {}", entry.manifest.display())
        })?)?;
    let directory = entry
        .manifest
        .parent()
        .context("annotation audit manifest has no parent directory")?;
    let (_, records) = load_verified_parent_annotations(
        directory,
        &manifest.identity.digest,
        &manifest.identity.search_fingerprint,
        &manifest.payload_sha256,
        manifest.annotation_count,
    )?;
    Ok((
        AnnotationCacheInventory {
            label: entry.label.clone(),
            manifest: entry.manifest.clone(),
            manifest_sha256: sha256_file(&entry.manifest)?,
            annotation_fingerprint: manifest.identity.digest,
            search_fingerprint: manifest.identity.search_fingerprint,
            generator_settings_sha256: manifest.identity.generator_settings_sha256,
            calibration_input_sha256: manifest.identity.calibration_input_sha256,
            payload_sha256: manifest.payload_sha256,
            annotation_count: manifest.annotation_count,
            requested_max_rank: manifest.identity.requested_max_rank,
            model_components: manifest.identity.model_components,
        },
        records,
    ))
}

const ANNOTATION_FIELD_NAMES: [&str; 15] = [
    "ms2pip_pcc",
    "spectral_angle",
    "fragment_intensity_agreement",
    "deeplc_predicted_rt",
    "deeplc_calibrated_rt",
    "deeplc_rt_error",
    "deeplc_abs_rt_error",
    "im2deep_predicted_ccs",
    "observed_ccs",
    "abs_ccs_error",
    "pct_ccs_error",
    "predicted_ion_mobility",
    "observed_ion_mobility",
    "abs_ion_mobility_error",
    "pct_ion_mobility_error",
];

fn annotation_values(features: &sage_core::scoring::ExternalPsmFeatures) -> [f32; 15] {
    [
        features.ms2rescore_ms2pip_pcc,
        features.ms2rescore_spectral_angle,
        features.ms2rescore_fragment_intensity_agreement,
        features.ms2rescore_deeplc_predicted_rt,
        features.ms2rescore_deeplc_calibrated_rt,
        features.ms2rescore_deeplc_rt_error,
        features.ms2rescore_deeplc_abs_rt_error,
        features.tims2rescore_im2deep_predicted_ccs,
        features.tims2rescore_observed_ccs,
        features.tims2rescore_abs_ccs_error,
        features.tims2rescore_pct_ccs_error,
        features.tims2rescore_predicted_ion_mobility,
        features.tims2rescore_observed_ion_mobility,
        features.tims2rescore_abs_ion_mobility_error,
        features.tims2rescore_pct_ion_mobility_error,
    ]
}

pub fn audit_annotation_caches(
    request_path: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<()> {
    let request: AnnotationAuditRequest =
        serde_json::from_slice(&std::fs::read(request_path.as_ref())?)?;
    anyhow::ensure!(
        request.schema == "within-parent-annotation-audit-v1",
        "unsupported annotation-audit request schema"
    );
    anyhow::ensure!(!request.groups.is_empty(), "annotation audit has no groups");
    let mut inventories = Vec::new();
    let mut comparisons = Vec::new();
    for group in request.groups {
        let (reference_inventory, reference_records) =
            load_annotation_audit_entry(&group.reference)?;
        let reference_search = reference_inventory.search_fingerprint.clone();
        let mut reference = HashMap::with_capacity(reference_records.len());
        for record in reference_records {
            anyhow::ensure!(
                reference
                    .insert(record.stable_id, record.features)
                    .is_none(),
                "reference annotation cache contains duplicate stable IDs"
            );
        }
        inventories.push(reference_inventory);
        for entry in group.comparisons {
            let (inventory, records) = load_annotation_audit_entry(&entry)?;
            anyhow::ensure!(
                inventory.search_fingerprint == reference_search,
                "annotation audit group {} mixes parent search fingerprints",
                group.label
            );
            let mut comparison = HashMap::with_capacity(records.len());
            for record in records {
                anyhow::ensure!(
                    comparison
                        .insert(record.stable_id, record.features)
                        .is_none(),
                    "comparison annotation cache contains duplicate stable IDs"
                );
            }
            let mut differing = [0_usize; 15];
            let mut maximum = [0.0_f64; 15];
            let mut nonfinite = [0_usize; 15];
            let mut joined = 0_usize;
            for (stable_id, left) in &reference {
                let Some(right) = comparison.get(stable_id) else {
                    continue;
                };
                joined += 1;
                for (index, (x, y)) in annotation_values(left)
                    .into_iter()
                    .zip(annotation_values(right))
                    .enumerate()
                {
                    if x.to_bits() != y.to_bits() {
                        differing[index] += 1;
                        let difference = if x.is_finite() && y.is_finite() {
                            (x as f64 - y as f64).abs()
                        } else {
                            nonfinite[index] += 1;
                            0.0
                        };
                        maximum[index] = maximum[index].max(difference);
                    }
                }
                anyhow::ensure!(
                    left.ms2rescore_feature_joined == right.ms2rescore_feature_joined,
                    "joined-feature flag differs for stable candidate {stable_id}"
                );
            }
            let field_differences = ANNOTATION_FIELD_NAMES
                .iter()
                .enumerate()
                .map(|(index, field)| AnnotationFieldDifference {
                    field: (*field).into(),
                    differing_values: differing[index],
                    maximum_finite_absolute_difference: maximum[index],
                    nonfinite_mismatches: nonfinite[index],
                })
                .collect::<Vec<_>>();
            let reference_only = reference.len() - joined;
            let comparison_only = comparison.len() - joined;
            comparisons.push(AnnotationCacheComparison {
                reference: group.reference.label.clone(),
                comparison: entry.label,
                joined_stable_ids: joined,
                reference_only_stable_ids: reference_only,
                comparison_only_stable_ids: comparison_only,
                duplicate_stable_ids: 0,
                exact_raw_feature_payload: reference_only == 0
                    && comparison_only == 0
                    && field_differences
                        .iter()
                        .all(|field| field.differing_values == 0),
                field_differences,
            });
            inventories.push(inventory);
        }
    }
    let exact = comparisons
        .iter()
        .all(|comparison| comparison.exact_raw_feature_payload);
    let result = AnnotationAuditResult {
        schema: "within-parent-annotation-audit-result-v1".into(),
        capabilities: CAPABILITIES,
        inventories,
        comparisons,
        conclusion: if exact {
            "all compared composition-specific caches contain byte-identical raw annotation fields by stable candidate ID; identity differences are preliminary-calibration-input metadata only"
        } else {
            "one or more composition-specific caches contain raw annotation differences"
        }
        .into(),
    };
    write_json_atomic(output.as_ref(), &result)
}

#[derive(Clone, Debug, Serialize)]
struct AuditPsmMetric {
    stable_id: String,
    spectrum_key: String,
    peptide: String,
    canonical_peptide: String,
    peptidoform: String,
    protein: Option<String>,
    entrapment: Option<bool>,
    score: Option<f64>,
    p_value: Option<f64>,
    pep: Option<f64>,
    psm_q: Option<f64>,
    peptide_q: Option<f64>,
    protein_q: Option<f64>,
    psm_level4_supported: Option<bool>,
    peptide_level4_supported: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
struct AuditExpertEntityStatus {
    expert: String,
    state: String,
    valid_for_support: bool,
    invalid_reason: Option<String>,
    score: Option<f64>,
    p_value: Option<f64>,
    pep: Option<f64>,
    q_value: Option<f64>,
    distance_from_acceptance_boundary: Option<f64>,
    hierarchical_support: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
struct AuditEntitySupport {
    level: String,
    entity: String,
    entrapment: Option<bool>,
    evidence_class: String,
    accepted_by: Vec<String>,
    valid_nonaccepting_experts: Vec<String>,
    expert_status: Vec<AuditExpertEntityStatus>,
    cross_level_support: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
struct AuditPairwise {
    left: String,
    right: String,
    exact_score_stream_duplicate: bool,
    joined_psms: usize,
    score_pearson: Option<f64>,
    pep_pearson: Option<f64>,
    q_value_pearson: Option<f64>,
    level4_target_psm_intersection: usize,
    level4_target_psm_union: usize,
    level4_target_psm_jaccard: Option<f64>,
    level4_target_peptide_intersection: usize,
    level4_target_peptide_union: usize,
    level4_target_peptide_jaccard: Option<f64>,
    accepted_by_left_only_psms: usize,
    accepted_by_right_only_psms: usize,
    both_narrowly_not_accepted_psms: usize,
}

#[derive(Clone, Debug, Serialize)]
struct AuditCoalition {
    experts: Vec<String>,
    valid_for_admission: bool,
    invalid_reasons: Vec<String>,
    plus_entrapment: Vec<LayerSummary>,
    target_only: Vec<LayerSummary>,
    plus_stream_sha256: String,
    target_stream_sha256: String,
}

fn psm_metric_map(
    stage: &CompletedStage,
    database: &IndexedDatabase,
    search_fingerprint: &str,
) -> BTreeMap<String, AuditPsmMetric> {
    stage
        .rank1_features
        .iter()
        .map(|feature| {
            let peptide = database[feature.core.peptide_idx].to_string();
            let stable_id = stable_candidate_id(search_fingerprint, &feature.core, &peptide);
            let proteins = database[feature.core.peptide_idx]
                .proteins(&database.decoy_tag, database.generate_decoys);
            let metric = AuditPsmMetric {
                stable_id: stable_id.clone(),
                spectrum_key: format!("{}\u{1f}{}", feature.core.file_id, feature.core.spec_id),
                canonical_peptide: canonical_peptide(&peptide),
                peptidoform: canonical_peptidoform(&peptide),
                protein: inferred_protein(feature, database),
                entrapment: classify_proteins(&proteins),
                peptide,
                score: feature.decoy_free_score,
                p_value: feature.decoy_free_p_value,
                pep: feature.decoy_free_pep,
                psm_q: feature.decoy_free_q_value,
                peptide_q: feature.decoy_free_peptide_q,
                protein_q: feature.decoy_free_protein_q,
                psm_level4_supported: feature.decoy_free_peptide_supported_psm,
                peptide_level4_supported: feature.decoy_free_protein_supported_peptide,
            };
            (stable_id, metric)
        })
        .collect()
}

fn pearson_pairs(values: impl Iterator<Item = (Option<f64>, Option<f64>)>) -> Option<f64> {
    let pairs = values
        .filter_map(|(x, y)| x.zip(y))
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let mx = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
    let my = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;
    let covariance = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum::<f64>();
    let vx = pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum::<f64>();
    let vy = pairs.iter().map(|(_, y)| (y - my).powi(2)).sum::<f64>();
    (vx > 0.0 && vy > 0.0).then(|| covariance / (vx * vy).sqrt())
}

fn jaccard<T: Ord>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> Option<f64> {
    let union = left.union(right).count();
    (union > 0).then(|| left.intersection(right).count() as f64 / union as f64)
}

fn stream_hash(stage: &CompletedStage) -> Result<String> {
    let stream = stage
        .rank1_features
        .iter()
        .map(|feature| {
            (
                feature.core.file_id,
                feature.core.spec_id.as_str(),
                feature.decoy_free_score.map(f64::to_bits),
                feature.decoy_free_p_value.map(f64::to_bits),
                feature.decoy_free_pep.map(f64::to_bits),
                feature.decoy_free_q_value.map(f64::to_bits),
            )
        })
        .collect::<Vec<_>>();
    hash_serialized(&stream)
}

fn next_q_plateau(
    metrics: &BTreeMap<String, AuditPsmMetric>,
    level: &str,
    threshold: f64,
) -> Option<f64> {
    metrics
        .values()
        .filter_map(|metric| match level {
            "psm" => metric.psm_q,
            "peptide" | "peptidoform" => metric.peptide_q,
            "protein" => metric.protein_q,
            _ => None,
        })
        .filter(|q| q.is_finite() && *q > threshold)
        .min_by(f64::total_cmp)
}

fn metric_q(metric: &AuditPsmMetric, level: &str) -> Option<f64> {
    match level {
        "psm" => metric.psm_q,
        "peptide" | "peptidoform" => metric.peptide_q,
        "protein" => metric.protein_q,
        _ => None,
    }
}

fn entity_key(metric: &AuditPsmMetric, level: &str) -> Option<String> {
    match level {
        "psm" => Some(metric.stable_id.clone()),
        "peptide" => Some(metric.canonical_peptide.clone()),
        "peptidoform" => Some(metric.peptidoform.clone()),
        "protein" => metric.protein.clone(),
        _ => None,
    }
}

fn entity_is_accepted(sets: &EvidenceSets, level: &str, entity: &str, entrapment: bool) -> bool {
    match (level, entrapment) {
        ("psm", false) => sets.target_psms.contains(entity),
        ("psm", true) => sets.entrapment_psms.contains(entity),
        ("peptide", false) => sets.target_peptides.contains(entity),
        ("peptide", true) => sets.entrapment_peptides.contains(entity),
        ("peptidoform", false) => sets.target_peptidoforms.contains(entity),
        ("peptidoform", true) => sets.entrapment_peptidoforms.contains(entity),
        ("protein", false) => sets.target_proteins.contains(entity),
        ("protein", true) => sets.entrapment_proteins.contains(entity),
        _ => false,
    }
}

fn best_entity_metric<'a>(
    metrics: &'a BTreeMap<String, AuditPsmMetric>,
    level: &str,
    entity: &str,
) -> Option<&'a AuditPsmMetric> {
    metrics
        .values()
        .filter(|metric| entity_key(metric, level).as_deref() == Some(entity))
        .min_by(|left, right| {
            metric_q(left, level)
                .unwrap_or(f64::INFINITY)
                .total_cmp(&metric_q(right, level).unwrap_or(f64::INFINITY))
        })
}

fn exact_stream_duplicate(
    left: &BTreeMap<String, AuditPsmMetric>,
    right: &BTreeMap<String, AuditPsmMetric>,
) -> bool {
    left.len() == right.len()
        && left.iter().all(|(id, x)| {
            right.get(id).is_some_and(|y| {
                x.score.map(f64::to_bits) == y.score.map(f64::to_bits)
                    && x.p_value.map(f64::to_bits) == y.p_value.map(f64::to_bits)
                    && x.pep.map(f64::to_bits) == y.pep.map(f64::to_bits)
                    && x.psm_q.map(f64::to_bits) == y.psm_q.map(f64::to_bits)
            })
        })
}

fn build_support_matrix(
    experts: &[String],
    valid: &BTreeMap<String, Option<String>>,
    stages: &BTreeMap<String, CompletedStage>,
    metrics: &BTreeMap<String, BTreeMap<String, AuditPsmMetric>>,
    threshold: f64,
) -> Vec<AuditEntitySupport> {
    let exact_duplicates = experts
        .iter()
        .flat_map(|left| {
            experts.iter().filter_map(move |right| {
                (left < right
                    && exact_stream_duplicate(
                        metrics.get(left).expect("expert metrics"),
                        metrics.get(right).expect("expert metrics"),
                    ))
                .then(|| (left.clone(), right.clone()))
            })
        })
        .collect::<BTreeSet<_>>();
    let mut output = Vec::new();
    for level in ["psm", "peptidoform", "peptide", "protein"] {
        let mut universe = BTreeSet::new();
        for expert in experts {
            for metric in metrics.get(expert).expect("expert metrics").values() {
                if let Some(entity) = entity_key(metric, level) {
                    universe.insert(entity);
                }
            }
        }
        let next_plateaus = experts
            .iter()
            .map(|expert| {
                (
                    expert.clone(),
                    next_q_plateau(
                        metrics.get(expert).expect("expert metrics"),
                        level,
                        threshold,
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for entity in universe {
            let entrapment = experts.iter().find_map(|expert| {
                best_entity_metric(metrics.get(expert).expect("expert metrics"), level, &entity)
                    .and_then(|metric| metric.entrapment)
            });
            let mut statuses = Vec::new();
            let mut accepted_by = Vec::new();
            let mut nonaccepting = Vec::new();
            for expert in experts {
                let invalid_reason = valid.get(expert).cloned().flatten();
                let metric = best_entity_metric(
                    metrics.get(expert).expect("expert metrics"),
                    level,
                    &entity,
                );
                let accepted = metric.is_some_and(|metric| {
                    metric.entrapment.is_some_and(|is_entrapment| {
                        entity_is_accepted(
                            &stages.get(expert).expect("expert stage").level4,
                            level,
                            &entity,
                            is_entrapment,
                        )
                    })
                });
                let valid_for_support = invalid_reason.is_none();
                let q = metric.and_then(|metric| metric_q(metric, level));
                let hierarchical = metric.and_then(|metric| match level {
                    "psm" => metric.psm_level4_supported,
                    "peptide" | "peptidoform" => metric.peptide_level4_supported,
                    "protein" => Some(true),
                    _ => None,
                });
                let state = if !valid_for_support {
                    "expert_invalid_or_excluded"
                } else if metric.is_none() {
                    "candidate_or_entity_absent"
                } else if q.is_none() {
                    "not_evaluable"
                } else if accepted {
                    "accepted"
                } else if q.is_some_and(|value| {
                    next_plateaus
                        .get(expert)
                        .and_then(|plateau| *plateau)
                        .is_some_and(|boundary| value.to_bits() == boundary.to_bits())
                }) {
                    "scored_but_narrowly_not_accepted"
                } else {
                    "scored_and_clearly_not_accepted"
                };
                if accepted && valid_for_support {
                    accepted_by.push(expert.clone());
                } else if valid_for_support && metric.is_some() {
                    nonaccepting.push(expert.clone());
                }
                statuses.push(AuditExpertEntityStatus {
                    expert: expert.clone(),
                    state: state.into(),
                    valid_for_support,
                    invalid_reason,
                    score: metric.and_then(|metric| metric.score),
                    p_value: metric.and_then(|metric| metric.p_value),
                    pep: metric.and_then(|metric| metric.pep),
                    q_value: q,
                    distance_from_acceptance_boundary: q.map(|value| value - threshold),
                    hierarchical_support: hierarchical,
                });
            }
            let accepted_classes = accepted_by
                .iter()
                .map(|expert| {
                    accepted_by
                        .iter()
                        .filter(|other| {
                            *other < expert
                                && exact_duplicates.contains(&(other.to_string(), expert.clone()))
                        })
                        .min()
                        .cloned()
                        .unwrap_or_else(|| expert.clone())
                })
                .collect::<BTreeSet<_>>()
                .len();
            let has_clear_disagreement = statuses.iter().any(|status| {
                status.valid_for_support && status.state == "scored_and_clearly_not_accepted"
            });
            let evaluable_count = statuses
                .iter()
                .filter(|status| status.valid_for_support && status.state != "not_evaluable")
                .count();
            let evidence_class = classify_support_evidence(
                accepted_by.len(),
                accepted_classes,
                evaluable_count,
                has_clear_disagreement,
            );
            let mut cross_level_support = BTreeMap::new();
            if let Some(metric) = experts.iter().find_map(|expert| {
                best_entity_metric(metrics.get(expert).expect("expert metrics"), level, &entity)
            }) {
                cross_level_support.insert(
                    "accepted_psms_for_same_peptide".into(),
                    experts
                        .iter()
                        .map(|expert| {
                            stages
                                .get(expert)
                                .expect("expert stage")
                                .level4
                                .target_psms
                                .iter()
                                .filter(|psm| {
                                    metrics
                                        .get(expert)
                                        .and_then(|map| map.get(*psm))
                                        .is_some_and(|candidate| {
                                            candidate.canonical_peptide == metric.canonical_peptide
                                        })
                                })
                                .count()
                        })
                        .sum(),
                );
            }
            output.push(AuditEntitySupport {
                level: level.into(),
                entity,
                entrapment,
                evidence_class: evidence_class.into(),
                accepted_by,
                valid_nonaccepting_experts: nonaccepting,
                expert_status: statuses,
                cross_level_support,
            });
        }
    }
    output
}

fn audit_pairwise(
    experts: &[String],
    stages: &BTreeMap<String, CompletedStage>,
    metrics: &BTreeMap<String, BTreeMap<String, AuditPsmMetric>>,
    threshold: f64,
) -> Vec<AuditPairwise> {
    let mut output = Vec::new();
    for (index, left_name) in experts.iter().enumerate() {
        for right_name in experts.iter().skip(index + 1) {
            let left = metrics.get(left_name).expect("left metrics");
            let right = metrics.get(right_name).expect("right metrics");
            let left_stage = stages.get(left_name).expect("left stage");
            let right_stage = stages.get(right_name).expect("right stage");
            let joined = left
                .iter()
                .filter_map(|(id, metric)| right.get(id).map(|other| (metric, other)))
                .collect::<Vec<_>>();
            let left_boundary = next_q_plateau(left, "psm", threshold);
            let right_boundary = next_q_plateau(right, "psm", threshold);
            let both_narrow = joined
                .iter()
                .filter(|(x, y)| {
                    x.psm_q
                        .zip(left_boundary)
                        .is_some_and(|(q, boundary)| q.to_bits() == boundary.to_bits())
                        && y.psm_q
                            .zip(right_boundary)
                            .is_some_and(|(q, boundary)| q.to_bits() == boundary.to_bits())
                })
                .count();
            let left_psm = &left_stage.level4.target_psms;
            let right_psm = &right_stage.level4.target_psms;
            let left_peptide = &left_stage.level4.target_peptides;
            let right_peptide = &right_stage.level4.target_peptides;
            output.push(AuditPairwise {
                left: left_name.clone(),
                right: right_name.clone(),
                exact_score_stream_duplicate: exact_stream_duplicate(left, right),
                joined_psms: joined.len(),
                score_pearson: pearson_pairs(joined.iter().map(|(x, y)| (x.score, y.score))),
                pep_pearson: pearson_pairs(joined.iter().map(|(x, y)| (x.pep, y.pep))),
                q_value_pearson: pearson_pairs(joined.iter().map(|(x, y)| (x.psm_q, y.psm_q))),
                level4_target_psm_intersection: left_psm.intersection(right_psm).count(),
                level4_target_psm_union: left_psm.union(right_psm).count(),
                level4_target_psm_jaccard: jaccard(left_psm, right_psm),
                level4_target_peptide_intersection: left_peptide
                    .intersection(right_peptide)
                    .count(),
                level4_target_peptide_union: left_peptide.union(right_peptide).count(),
                level4_target_peptide_jaccard: jaccard(left_peptide, right_peptide),
                accepted_by_left_only_psms: left_psm.difference(right_psm).count(),
                accepted_by_right_only_psms: right_psm.difference(left_psm).count(),
                both_narrowly_not_accepted_psms: both_narrow,
            });
        }
    }
    output
}

fn factorial(value: usize) -> f64 {
    (1..=value).product::<usize>() as f64
}

fn evidence_count(sets: &EvidenceSets, level: &str, entrapment: bool) -> usize {
    match (level, entrapment) {
        ("psm", false) => sets.target_psms.len(),
        ("psm", true) => sets.entrapment_psms.len(),
        ("peptide", false) => sets.target_peptides.len(),
        ("peptide", true) => sets.entrapment_peptides.len(),
        ("peptidoform", false) => sets.target_peptidoforms.len(),
        ("peptidoform", true) => sets.entrapment_peptidoforms.len(),
        ("protein", false) => sets.target_proteins.len(),
        ("protein", true) => sets.entrapment_proteins.len(),
        _ => 0,
    }
}

fn evidence_set<'a>(sets: &'a EvidenceSets, level: &str, entrapment: bool) -> &'a BTreeSet<String> {
    match (level, entrapment) {
        ("psm", false) => &sets.target_psms,
        ("psm", true) => &sets.entrapment_psms,
        ("peptide", false) => &sets.target_peptides,
        ("peptide", true) => &sets.entrapment_peptides,
        ("peptidoform", false) => &sets.target_peptidoforms,
        ("peptidoform", true) => &sets.entrapment_peptidoforms,
        ("protein", false) => &sets.target_proteins,
        ("protein", true) => &sets.entrapment_proteins,
        _ => unreachable!("validated evidence level"),
    }
}

fn classify_support_evidence(
    accepted_count: usize,
    accepted_nonredundant_classes: usize,
    evaluable_count: usize,
    has_clear_disagreement: bool,
) -> &'static str {
    if accepted_count == 0 {
        if evaluable_count == 0 {
            "not_evaluable"
        } else {
            "not_accepted"
        }
    } else if accepted_nonredundant_classes >= 2 {
        "corroborated"
    } else if evaluable_count == accepted_count {
        "uniquely_evaluable"
    } else if has_clear_disagreement {
        "disputed"
    } else {
        "singly_supported_but_consistent"
    }
}

fn classify_low_input_expert(
    invalid_reason: Option<&str>,
    exact_duplicate: bool,
    unique_discoveries: usize,
    corroborated: usize,
    shapley_target_peptides: f64,
) -> &'static str {
    if invalid_reason.is_some() {
        "invalid"
    } else if shapley_target_peptides < 0.0 {
        "harmful"
    } else if unique_discoveries > 0 {
        "novel_contributor"
    } else if corroborated > 0 && !exact_duplicate {
        "supporting_corroborating_contributor"
    } else if exact_duplicate || corroborated == 0 {
        "redundant"
    } else {
        "not_evaluable"
    }
}

fn sequential_unique_credit(
    order: &[&str],
    accepted: &BTreeMap<String, BTreeSet<String>>,
    minimum_unique: usize,
) -> Vec<(String, usize, bool)> {
    let mut union = BTreeSet::new();
    order
        .iter()
        .map(|expert| {
            let evidence = &accepted[*expert];
            let incremental = evidence.difference(&union).count();
            let included = incremental >= minimum_unique;
            if included {
                union.extend(evidence.iter().cloned());
            }
            ((*expert).to_string(), incremental, included)
        })
        .collect()
}

fn shapley_rows(
    valid_experts: &[String],
    bit_for_expert: &BTreeMap<String, usize>,
    coalition_evidence: &BTreeMap<usize, (EvidenceSets, EvidenceSets)>,
) -> Vec<serde_json::Value> {
    let n = valid_experts.len();
    let mut rows = Vec::new();
    for layer in ["raw_q", "level4"] {
        for level in ["psm", "peptidoform", "peptide", "protein"] {
            for entrapment in [false, true] {
                for expert in valid_experts {
                    let bit = bit_for_expert[expert];
                    let mut contribution = 0.0;
                    for subset_index in 0..(1_usize << n) {
                        let mut mask = 0_usize;
                        let mut size = 0_usize;
                        for (local_index, member) in valid_experts.iter().enumerate() {
                            if subset_index & (1 << local_index) != 0 {
                                mask |= bit_for_expert[member];
                                size += 1;
                            }
                        }
                        if mask & bit != 0 {
                            continue;
                        }
                        let with = mask | bit;
                        let value = |coalition: usize| {
                            if coalition == 0 {
                                0
                            } else {
                                let sets = &coalition_evidence[&coalition];
                                evidence_count(
                                    if layer == "raw_q" { &sets.0 } else { &sets.1 },
                                    level,
                                    entrapment,
                                )
                            }
                        };
                        let weight = factorial(size) * factorial(n - size - 1) / factorial(n);
                        contribution += weight * (value(with) as f64 - value(mask) as f64);
                    }
                    rows.push(serde_json::json!({
                        "expert": expert,
                        "layer": layer,
                        "level": level,
                        "evidence": if entrapment { "entrapment" } else { "target" },
                        "shapley_count_contribution": contribution
                    }));
                }
            }
        }
    }
    rows
}

/// Post-failure, training-only audit. This path deliberately never derives a
/// held-out subset and cannot search spectra or generate annotations.
pub fn audit_training_usefulness(
    manifest_path: impl AsRef<Path>,
    failed_run_root: impl AsRef<Path>,
    fold_number: usize,
    output: impl AsRef<Path>,
) -> Result<()> {
    let manifest_path = manifest_path.as_ref();
    let manifest_bytes = std::fs::read(manifest_path)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest: HoldoutPreregistration = serde_json::from_slice(&manifest_bytes)?;
    validate_preregistration(&manifest)?;
    verify_parent_identity(&manifest.plus_entrapment)?;
    verify_parent_identity(&manifest.target_only)?;
    let fold = manifest
        .folds
        .iter()
        .find(|fold| fold.fold == fold_number)
        .context("requested training fold is absent from preregistration")?;
    let output = output.as_ref();
    std::fs::create_dir_all(output)?;
    let plus_context =
        build_parent_context(&manifest.plus_entrapment, &output.join("scratch/plus"))?;
    let target_context =
        build_parent_context(&manifest.target_only, &output.join("scratch/target"))?;
    let training = derive_subset(
        &manifest,
        &plus_context,
        &manifest_sha256,
        fold.fold,
        SubsetRole::Training,
        &fold.training_file_ids,
    )?;
    let training_target = derive_subset(
        &manifest,
        &target_context,
        &manifest_sha256,
        fold.fold,
        SubsetRole::Training,
        &fold.training_file_ids,
    )?;

    let failed_fold_root = failed_run_root.as_ref().join(format!("fold_{}", fold.fold));
    let preserved_gates_path = failed_fold_root.join("A.training_gates.json");
    let preserved_gates: Vec<ExpertTrainingGate> =
        serde_json::from_slice(&std::fs::read(&preserved_gates_path).with_context(|| {
            format!("reading preserved gates {}", preserved_gates_path.display())
        })?)?;
    let mut windows = BTreeMap::<String, Option<NullWindow>>::new();
    let mut selected_artifacts = Vec::new();
    for grid in &manifest.model_grids {
        let slug = model_slug(&grid.model);
        let artifact_path = failed_fold_root
            .join("training_artifacts")
            .join(slug)
            .join("within_parent_holdout_artifact.json");
        let artifact: WithinParentHoldoutArtifact =
            serde_json::from_slice(&std::fs::read(&artifact_path)?)?;
        anyhow::ensure!(
            artifact.schema == ARTIFACT_SCHEMA,
            "invalid preserved artifact schema"
        );
        anyhow::ensure!(
            artifact.manifest_sha256 == manifest_sha256,
            "preserved artifact manifest mismatch"
        );
        anyhow::ensure!(
            artifact.training_subset_digest == training.identity.digest,
            "preserved artifact training subset mismatch"
        );
        anyhow::ensure!(
            artifact.model == grid.model,
            "preserved artifact model mismatch"
        );
        let mut unhashed = artifact.clone();
        unhashed.digest.clear();
        anyhow::ensure!(
            hash_serialized(&unhashed)? == artifact.digest,
            "preserved artifact digest mismatch"
        );
        windows.insert(slug.into(), artifact.selected_window.clone());
        selected_artifacts.push(artifact);
    }

    let audited_models = [
        ModelFit::Moments,
        ModelFit::Mle,
        ModelFit::Msfdr1Smix,
        ModelFit::LowerOrder,
    ];
    let expert_names = audited_models
        .iter()
        .map(|model| model_slug(model).to_string())
        .collect::<Vec<_>>();
    let mut plus_stages = BTreeMap::<String, CompletedStage>::new();
    let mut target_stages = BTreeMap::<String, CompletedStage>::new();
    for model in &audited_models {
        let slug = model_slug(model);
        let window = windows.get(slug).cloned().flatten();
        let plus_settings =
            settings_for_model(&plus_context.search.fdr, model.clone(), window.clone());
        let target_settings = settings_for_model(&target_context.search.fdr, model.clone(), window);
        plus_stages.insert(
            slug.into(),
            run_scored_stage(
                &training,
                &plus_settings,
                &plus_context.database,
                slug,
                manifest.effective_ratios,
                SearchSpace::PlusEntrapment,
            )?,
        );
        target_stages.insert(
            slug.into(),
            run_scored_stage(
                &training_target,
                &target_settings,
                &target_context.database,
                slug,
                manifest.effective_ratios,
                SearchSpace::TargetOnly,
            )?,
        );
    }
    let validity = preserved_gates
        .iter()
        .map(|gate| {
            let non_usefulness = gate
                .reasons
                .iter()
                .filter(|reason| !reason.starts_with("adds only "))
                .cloned()
                .collect::<Vec<_>>();
            (
                model_slug(&gate.model).to_string(),
                (!non_usefulness.is_empty()).then(|| non_usefulness.join("; ")),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let metrics = expert_names
        .iter()
        .map(|expert| {
            (
                expert.clone(),
                psm_metric_map(
                    plus_stages.get(expert).expect("plus stage"),
                    &plus_context.database,
                    &manifest.plus_entrapment.search_fingerprint,
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let support = build_support_matrix(
        &expert_names,
        &validity,
        &plus_stages,
        &metrics,
        manifest.runtime_gates.fdr_threshold,
    );
    write_json_atomic(&output.join("training_support_matrix.json"), &support)?;
    let support_sha256 = sha256_file(&output.join("training_support_matrix.json"))?;
    let pairwise = audit_pairwise(
        &expert_names,
        &plus_stages,
        &metrics,
        manifest.runtime_gates.fdr_threshold,
    );

    let baseline_names = ["moments", "mle", "msfdr1_smix"];
    let baseline_accepted = baseline_names
        .iter()
        .map(|name| {
            (
                name.to_string(),
                plus_stages[*name].level4.target_peptides.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    .into_iter()
    .map(|order| {
        let ordered_names = order.map(|index| baseline_names[index]);
        let assignments = sequential_unique_credit(
            &ordered_names,
            &baseline_accepted,
            manifest
                .runtime_gates
                .minimum_incremental_level4_target_peptides,
        )
        .into_iter()
        .map(|(expert, incremental, included)| {
            serde_json::json!({
                "expert": expert,
                "incremental_level4_target_peptides": incremental,
                "included": included && validity[&expert].is_none()
            })
        })
        .collect::<Vec<_>>();
        serde_json::json!({"order": ordered_names, "assignments": assignments})
    })
    .collect::<Vec<_>>();

    let coalition_models = audited_models.to_vec();
    let bit_for_expert = coalition_models
        .iter()
        .enumerate()
        .map(|(index, model)| (model_slug(model).to_string(), 1_usize << index))
        .collect::<BTreeMap<_, _>>();
    let mut coalition_rows = Vec::new();
    let mut coalition_evidence = BTreeMap::new();
    for mask in 1_usize..(1_usize << coalition_models.len()) {
        let models = coalition_models
            .iter()
            .enumerate()
            .filter(|(index, _)| mask & (1 << index) != 0)
            .map(|(_, model)| model.clone())
            .collect::<Vec<_>>();
        let names = models
            .iter()
            .map(|model| model_slug(model).to_string())
            .collect::<Vec<_>>();
        let coalition_windows = names
            .iter()
            .map(|name| (name.clone(), windows.get(name).cloned().flatten()))
            .collect::<BTreeMap<_, _>>();
        let (plus_settings, target_settings) = if models.len() == 1 {
            let model = models[0].clone();
            let window = coalition_windows.get(model_slug(&model)).cloned().flatten();
            (
                settings_for_model(&plus_context.search.fdr, model.clone(), window.clone()),
                settings_for_model(&target_context.search.fdr, model, window),
            )
        } else {
            (
                settings_for_ensemble(&plus_context.search.fdr, &coalition_windows, &models)?,
                settings_for_ensemble(&target_context.search.fdr, &coalition_windows, &models)?,
            )
        };
        let plus = run_scored_stage(
            &training,
            &plus_settings,
            &plus_context.database,
            &format!("coalition_{mask}"),
            manifest.effective_ratios,
            SearchSpace::PlusEntrapment,
        )?;
        let target = run_scored_stage(
            &training_target,
            &target_settings,
            &target_context.database,
            &format!("coalition_{mask}"),
            manifest.effective_ratios,
            SearchSpace::TargetOnly,
        )?;
        let invalid_reasons = names
            .iter()
            .filter_map(|name| {
                validity
                    .get(name)
                    .cloned()
                    .flatten()
                    .map(|reason| format!("{name}: {reason}"))
            })
            .collect::<Vec<_>>();
        coalition_rows.push(AuditCoalition {
            experts: names,
            valid_for_admission: invalid_reasons.is_empty(),
            invalid_reasons,
            plus_entrapment: plus.summary.layers.clone(),
            target_only: target.summary.layers.clone(),
            plus_stream_sha256: stream_hash(&plus)?,
            target_stream_sha256: stream_hash(&target)?,
        });
        coalition_evidence.insert(mask, (plus.raw, plus.level4));
    }
    let shapley = shapley_rows(
        &baseline_names
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>(),
        &bit_for_expert,
        &coalition_evidence,
    );
    let mut coalition_transitions = Vec::new();
    let valid_baseline = baseline_names
        .iter()
        .filter(|name| validity.get(**name).is_some_and(Option::is_none))
        .map(|name| name.to_string())
        .collect::<Vec<_>>();
    for expert in &valid_baseline {
        let bit = bit_for_expert[expert];
        for subset in 0_usize..(1_usize << valid_baseline.len()) {
            let mut mask = 0_usize;
            for (index, member) in valid_baseline.iter().enumerate() {
                if subset & (1 << index) != 0 {
                    mask |= bit_for_expert[member];
                }
            }
            if mask & bit != 0 {
                continue;
            }
            let with = mask | bit;
            for layer in ["raw_q", "level4"] {
                for level in ["psm", "peptidoform", "peptide", "protein"] {
                    for entrapment in [false, true] {
                        let empty = EvidenceSets::default();
                        let before_pair = coalition_evidence.get(&mask);
                        let before = before_pair
                            .map(|pair| if layer == "raw_q" { &pair.0 } else { &pair.1 })
                            .unwrap_or(&empty);
                        let after_pair = &coalition_evidence[&with];
                        let after = if layer == "raw_q" {
                            &after_pair.0
                        } else {
                            &after_pair.1
                        };
                        let before_set = evidence_set(before, level, entrapment);
                        let after_set = evidence_set(after, level, entrapment);
                        coalition_transitions.push(serde_json::json!({
                            "expert_added": expert,
                            "coalition_before_mask": mask,
                            "coalition_after_mask": with,
                            "layer": layer,
                            "level": level,
                            "evidence": if entrapment { "entrapment" } else { "target" },
                            "gained": after_set.difference(before_set).count(),
                            "lost": before_set.difference(after_set).count()
                        }));
                    }
                }
            }
        }
    }

    let mut proposed_classifications = Vec::new();
    for expert in &expert_names {
        let unique = support
            .iter()
            .filter(|row| {
                row.level == "peptide"
                    && row.entrapment == Some(false)
                    && row.accepted_by == vec![expert.clone()]
            })
            .count();
        let corroborated = support
            .iter()
            .filter(|row| {
                row.level == "peptide"
                    && row.entrapment == Some(false)
                    && row.evidence_class == "corroborated"
                    && row.accepted_by.contains(expert)
            })
            .count();
        let disputed = support
            .iter()
            .filter(|row| {
                row.level == "peptide"
                    && row.entrapment == Some(false)
                    && row.evidence_class == "disputed"
                    && row.accepted_by.contains(expert)
            })
            .count();
        let duplicate = pairwise.iter().any(|pair| {
            pair.exact_score_stream_duplicate && (&pair.left == expert || &pair.right == expert)
        });
        let shapley_peptides = shapley
            .iter()
            .find(|row| {
                row.get("expert").and_then(serde_json::Value::as_str) == Some(expert)
                    && row.get("layer").and_then(serde_json::Value::as_str) == Some("level4")
                    && row.get("level").and_then(serde_json::Value::as_str) == Some("peptide")
                    && row.get("evidence").and_then(serde_json::Value::as_str) == Some("target")
            })
            .and_then(|row| row.get("shapley_count_contribution"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        proposed_classifications.push(serde_json::json!({
            "expert": expert,
            "classification": classify_low_input_expert(
                validity[expert].as_deref(),
                duplicate,
                unique,
                corroborated,
                shapley_peptides,
            ),
            "invalid_reason": validity[expert],
            "unique_level4_target_peptides": unique,
            "corroborated_level4_target_peptides": corroborated,
            "disputed_level4_target_peptides": disputed,
            "exact_score_stream_duplicate": duplicate,
            "shapley_level4_target_peptide_contribution": shapley_peptides
        }));
    }
    let proposed_valid_nonredundant = proposed_classifications
        .iter()
        .filter(|row| {
            matches!(
                row.get("classification")
                    .and_then(serde_json::Value::as_str),
                Some("novel_contributor" | "supporting_corroborating_contributor")
            )
        })
        .count();
    let proposed_baseline = baseline_names
        .iter()
        .map(|name| name.to_string())
        .collect::<BTreeSet<_>>();
    let proposed_baseline_final_level4_peptide_fdp = coalition_rows
        .iter()
        .find(|row| row.experts.iter().cloned().collect::<BTreeSet<_>>() == proposed_baseline)
        .and_then(|row| {
            row.plus_entrapment
                .iter()
                .find(|layer| layer.layer == "level4")
        })
        .and_then(|layer| layer.peptide.ratio_adjusted_fdp);
    let proposed_baseline_final_level4_calibration_pass =
        proposed_baseline_final_level4_peptide_fdp
            .is_some_and(|fdp| fdp <= manifest.runtime_gates.fdr_threshold);

    let plus_lo = plus_stages
        .get("lower_order")
        .expect("lower order plus stage");
    let target_lo = target_stages
        .get("lower_order")
        .expect("lower order target stage");
    let plus_lo_metrics = metrics
        .get("lower_order")
        .expect("lower order plus metrics");
    let target_lo_metrics = psm_metric_map(
        target_lo,
        &target_context.database,
        &manifest.target_only.search_fingerprint,
    );
    let target_by_spectrum = target_lo_metrics
        .values()
        .map(|metric| (metric.spectrum_key.clone(), metric))
        .collect::<BTreeMap<_, _>>();
    let mut matched_spectra = 0_usize;
    let mut same_rank1_candidate = 0_usize;
    let mut plus_accepted_with_same_target_candidate = 0_usize;
    let mut plus_accepted_same_candidate_also_accepted = 0_usize;
    for metric in plus_lo_metrics.values() {
        if let Some(target_metric) = target_by_spectrum.get(&metric.spectrum_key) {
            matched_spectra += 1;
            if metric.peptidoform == target_metric.peptidoform {
                same_rank1_candidate += 1;
                if plus_lo.level4.target_psms.contains(&metric.stable_id) {
                    plus_accepted_with_same_target_candidate += 1;
                    if target_lo
                        .level4
                        .target_psms
                        .contains(&target_metric.stable_id)
                    {
                        plus_accepted_same_candidate_also_accepted += 1;
                    }
                }
            }
        }
    }
    let lo_peptide_intersection = plus_lo
        .level4
        .target_peptides
        .intersection(&target_lo.level4.target_peptides)
        .count();
    let lower_order_transfer = serde_json::json!({
        "aggregate_plus_level4_target_peptides": plus_lo.level4.target_peptides.len(),
        "aggregate_target_level4_target_peptides": target_lo.level4.target_peptides.len(),
        "aggregate_reported_loss_fraction": 1.0 - target_lo.level4.target_peptides.len() as f64 / plus_lo.level4.target_peptides.len() as f64,
        "matched_spectrum_keys": matched_spectra,
        "same_rank1_peptidoform": same_rank1_candidate,
        "plus_accepted_psms_with_same_target_rank1_peptidoform": plus_accepted_with_same_target_candidate,
        "same_candidate_accepted_in_both_spaces": plus_accepted_same_candidate_also_accepted,
        "level4_target_peptide_intersection": lo_peptide_intersection,
        "plus_only_level4_target_peptides": plus_lo.level4.target_peptides.len() - lo_peptide_intersection,
        "target_only_level4_target_peptides": target_lo.level4.target_peptides.len() - lo_peptide_intersection,
        "matched_peptide_jaccard": jaccard(&plus_lo.level4.target_peptides, &target_lo.level4.target_peptides)
    });

    let mut support_class_counts = BTreeMap::<String, usize>::new();
    for row in &support {
        *support_class_counts
            .entry(format!("{}:{}", row.level, row.evidence_class))
            .or_default() += 1;
    }
    let individual = expert_names
        .iter()
        .map(|name| {
            serde_json::json!({
                "expert": name,
                "validity_without_usefulness": validity[name],
                "plus_entrapment": plus_stages[name].summary.layers,
                "target_only": target_stages[name].summary.layers,
                "plus_stream_sha256": stream_hash(&plus_stages[name]).expect("stream hash"),
                "target_stream_sha256": stream_hash(&target_stages[name]).expect("stream hash")
            })
        })
        .collect::<Vec<_>>();
    let executable = std::env::current_exe()?;
    let result = serde_json::json!({
        "schema": "within-parent-training-usefulness-audit-v1",
        "capabilities": CAPABILITIES,
        "manifest_sha256": manifest_sha256,
        "failed_gate_audit_preserved": preserved_gates_path,
        "audit_binary_sha256": sha256_file(&executable)?,
        "fold": fold.fold,
        "training_file_ids": fold.training_file_ids,
        "held_out_data_accessed": false,
        "selected_artifacts": selected_artifacts,
        "preserved_sequential_gates": preserved_gates,
        "production_semantics": {
            "sequential": true,
            "deterministic_sort": "descending accepted Level-4 target peptide count, then model slug",
            "first_expert_receives_shared_credit": true,
            "later_expert_requires_unique_peptide": true,
            "minimum_expert_count_uses_post_usefulness_eligibility": true,
            "manifest_or_hash_map_order_affects_result": false,
            "scientific_credit_is_order_asymmetric": true
        },
        "individual_experts": individual,
        "sequential_permutations": permutations,
        "pairwise": pairwise,
        "support_matrix_path": output.join("training_support_matrix.json"),
        "support_matrix_sha256": support_sha256,
        "support_class_counts": support_class_counts,
        "coalitions": coalition_rows,
        "coalition_transitions": coalition_transitions,
        "shapley_count_contributions": shapley,
        "proposed_low_input_expert_classifications": proposed_classifications,
        "proposed_fold1_baseline_could_form": proposed_valid_nonredundant >= manifest.runtime_gates.minimum_ensemble_experts
            && proposed_baseline_final_level4_calibration_pass,
        "proposed_valid_nonredundant_expert_count": proposed_valid_nonredundant,
        "proposed_baseline_final_level4_peptide_fdp": proposed_baseline_final_level4_peptide_fdp,
        "proposed_baseline_final_level4_calibration_pass": proposed_baseline_final_level4_calibration_pass,
        "lower_order_transfer": lower_order_transfer,
        "ensemble_combination_audit": {
            "p_combiner": "second_best",
            "pep_combiner": "median",
            "p_values_are_not_explicitly_modeled_as_independent_by_second_best_but_duplicate streams can alter order statistics once three or more experts participate": true,
            "median_pep_can_double_weight_duplicate_or highly correlated streams": true,
            "final_level4_entrapment_calibration_remains_the blocking empirical safeguard": true
        }
    });
    write_json_atomic(&output.join("training_usefulness_audit.json"), &result)
}

#[derive(Clone, Debug, Default)]
struct EvidenceSets {
    target_psms: BTreeSet<String>,
    entrapment_psms: BTreeSet<String>,
    target_peptides: BTreeSet<String>,
    entrapment_peptides: BTreeSet<String>,
    target_peptidoforms: BTreeSet<String>,
    entrapment_peptidoforms: BTreeSet<String>,
    target_proteins: BTreeSet<String>,
    entrapment_proteins: BTreeSet<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CountWithEntrapment {
    pub target: usize,
    pub entrapment: usize,
    pub ratio_adjusted_fdp: Option<f64>,
    pub ratio_adjusted_fdp_wilson_95: Option<[f64; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LayerSummary {
    pub layer: String,
    pub psm: CountWithEntrapment,
    pub peptide: CountWithEntrapment,
    pub peptidoform: CountWithEntrapment,
    pub protein: CountWithEntrapment,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageSummary {
    pub search_space: SearchSpace,
    pub composition: String,
    pub subset_digest: String,
    pub layers: Vec<LayerSummary>,
    pub fallback_used: bool,
    pub unexplained_na_count: usize,
    pub external_profile_window: NullWindow,
    pub external_profile_sha256: String,
    pub nuisance_state_provenance: String,
    pub complete_artifact_reused: bool,
    pub window_retuned: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelWindowSelection {
    pub model: ModelFit,
    pub window: Option<NullWindow>,
    pub evaluated_windows: usize,
    pub feasible_windows: usize,
    pub fallback_used: bool,
    pub evaluation_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IncrementalEvidence {
    pub layer: String,
    pub added_target_psms: usize,
    pub lost_target_psms: usize,
    pub added_target_peptides: usize,
    pub lost_target_peptides: usize,
    pub added_target_peptidoforms: usize,
    pub lost_target_peptidoforms: usize,
    pub added_target_proteins: usize,
    pub lost_target_proteins: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertTrainingGate {
    pub model: ModelFit,
    pub selected_window: Option<NullWindow>,
    pub eligible: bool,
    pub participation_reason: String,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
    pub calibration_level4_target_peptides: usize,
    pub target_only_level4_target_peptides: usize,
    pub incremental_level4_target_peptides: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InteractionDelta {
    pub layer: String,
    pub level: String,
    pub baseline_fdp: Option<f64>,
    pub final_fdp: Option<f64>,
    pub absolute_change: Option<f64>,
    pub raw_q_warning: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CompositionComparison {
    pub baseline: String,
    pub comparison: String,
    pub plus_entrapment_incremental: Vec<IncrementalEvidence>,
    pub target_only_incremental: Vec<IncrementalEvidence>,
    pub interaction_deltas: Vec<InteractionDelta>,
    pub final_level4_calibration_decision: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldCompositionSummary {
    pub composition: String,
    pub lock: WithinParentHoldoutLock,
    pub plus_entrapment: StageSummary,
    pub target_only: StageSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldSummary {
    pub fold: usize,
    pub training_subset_digest: String,
    pub held_out_plus_entrapment_subset_digest: String,
    pub held_out_target_only_subset_digest: String,
    pub selected_windows: Vec<ModelWindowSelection>,
    pub training_gates: BTreeMap<String, Vec<ExpertTrainingGate>>,
    pub compositions: Vec<FoldCompositionSummary>,
    pub comparisons_to_baseline: Vec<CompositionComparison>,
    pub technically_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AggregateCompositionSummary {
    pub composition: String,
    pub plus_entrapment: Vec<LayerSummary>,
    pub target_only: Vec<LayerSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AggregateSummary {
    pub compositions: Vec<AggregateCompositionSummary>,
    pub comparisons_to_baseline: Vec<CompositionComparison>,
    pub construction: String,
    pub every_run_once: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HoldoutClassification {
    pub validation_runner_integrity: String,
    pub fold_technical_validity: String,
    pub window_stability: String,
    pub calibration_behavior: String,
    pub incremental_useful_evidence: String,
    pub target_only_stability: String,
    pub production_admission_readiness: String,
    pub production_policy_changed: bool,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HoldoutResult {
    pub schema: String,
    pub manifest_sha256: String,
    pub preflight_sha256: String,
    pub source_build: SourceBuildIdentity,
    pub capabilities: ValidationRunnerCapabilities,
    pub folds: Vec<FoldSummary>,
    pub aggregate: AggregateSummary,
    pub classifications: BTreeMap<String, HoldoutClassification>,
}

#[derive(Clone)]
struct CompletedStage {
    summary: StageSummary,
    raw: EvidenceSets,
    level4: EvidenceSets,
    artifacts: DfRunArtifacts,
    rank1_features: Vec<DfFeature>,
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

fn clear_frozen_state(settings: &mut FdrSettings) {
    settings.null_window_optimizer = None;
    settings.moments_frozen_parameters = None;
    settings.mle_frozen_parameters = None;
    settings.lower_order_frozen_artifact = None;
    settings.msfdr_seeded_frozen_model = None;
    settings.msfdr_1smix_frozen_model = None;
    settings.msfdr_2smix_frozen_model = None;
    settings.nokoi_frozen_artifact = None;
    settings.external_ms2rescore_frozen_profiles = None;
}

fn apply_model_window(settings: &mut FdrSettings, model: &ModelFit, window: &Option<NullWindow>) {
    let Some(window) = window else { return };
    match model {
        ModelFit::Moments => {
            settings.moments_min_null_rank = window.min_rank;
            settings.moments_max_null_rank = window.max_rank;
        }
        ModelFit::Mle => {
            settings.mle_min_null_rank = window.min_rank;
            settings.mle_max_null_rank = window.max_rank;
        }
        ModelFit::LowerOrder => {
            settings.lower_order_min_null_rank = window.min_rank;
            settings.lower_order_max_null_rank = window.max_rank;
        }
        ModelFit::Msfdr => {
            settings.msfdr_min_null_rank = window.min_rank;
            settings.msfdr_max_null_rank = window.max_rank;
        }
        ModelFit::Msfdr2Smix => {
            settings.msfdr2_smix_min_null_rank = window.min_rank;
            settings.msfdr2_smix_max_null_rank = window.max_rank;
        }
        ModelFit::Nokoi => {
            settings.nokoi_min_null_rank = window.min_rank;
            settings.nokoi_max_null_rank = window.max_rank;
        }
        ModelFit::Msfdr1Smix | ModelFit::Ensemble => {}
    }
}

fn settings_for_model(
    base: &FdrSettings,
    model: ModelFit,
    window: Option<NullWindow>,
) -> FdrSettings {
    let mut settings = base.clone();
    clear_frozen_state(&mut settings);
    settings.model_fit = model.clone();
    settings.enable_moments = false;
    settings.enable_mle = false;
    settings.enable_lower_order = false;
    settings.enable_msfdr_seeded = false;
    settings.enable_msfdr_1smix = false;
    settings.enable_msfdr_2smix = false;
    settings.enable_nokoi = false;
    apply_model_window(&mut settings, &model, &window);
    settings
}

fn settings_for_ensemble(
    base: &FdrSettings,
    windows: &BTreeMap<String, Option<NullWindow>>,
    experts: &[ModelFit],
) -> Result<FdrSettings> {
    let mut settings = settings_for_model(base, ModelFit::Ensemble, None);
    let unique = experts.iter().map(model_slug).collect::<BTreeSet<_>>();
    anyhow::ensure!(
        unique.len() == experts.len(),
        "Ensemble composition contains duplicate experts"
    );
    settings.enable_moments = experts.contains(&ModelFit::Moments);
    settings.enable_mle = experts.contains(&ModelFit::Mle);
    settings.enable_lower_order = experts.contains(&ModelFit::LowerOrder);
    settings.enable_msfdr_seeded = experts.contains(&ModelFit::Msfdr);
    settings.enable_msfdr_1smix = experts.contains(&ModelFit::Msfdr1Smix);
    settings.enable_msfdr_2smix = experts.contains(&ModelFit::Msfdr2Smix);
    settings.enable_nokoi = experts.contains(&ModelFit::Nokoi);
    for model in experts {
        anyhow::ensure!(
            !matches!(model, ModelFit::Nokoi | ModelFit::Ensemble),
            "unsupported holdout Ensemble expert {}",
            model_slug(model)
        );
        if *model == ModelFit::Msfdr1Smix {
            continue;
        }
        let window = windows
            .get(model_slug(model))
            .with_context(|| format!("missing locked window for {}", model_slug(model)))?;
        apply_model_window(&mut settings, model, window);
    }
    Ok(settings)
}

fn optimizer_settings(
    base: &FdrSettings,
    grid: &ModelGrid,
    manifest: &HoldoutPreregistration,
) -> Result<FdrSettings> {
    anyhow::ensure!(
        grid.fixed_window.is_none(),
        "fixed models must not enter the optimizer"
    );
    anyhow::ensure!(!grid.candidates.is_empty(), "optimizer grid is empty");
    let mut settings = settings_for_model(base, grid.model.clone(), None);
    settings.null_window_optimizer = Some(NullWindowOptimizerOptions {
        candidates: grid
            .candidates
            .iter()
            .map(|window| NullWindowCandidate {
                min_rank: window.min_rank,
                max_rank: window.max_rank,
            })
            .collect(),
        strategy: NullWindowSearchStrategy::Explicit,
        bounds: None,
        adaptive: AdaptiveNullWindowSearchOptions::default(),
        validation_scope: manifest.optimizer_validation_scope,
        fdr_threshold: manifest.acceptance.fdr_threshold,
        psm_entrapment_ratio: manifest.effective_ratios.psm,
        peptide_entrapment_ratio: manifest.effective_ratios.peptide,
        protein_entrapment_ratio: manifest.effective_ratios.protein,
        maximum_entrapment_fdp: manifest.acceptance.fdr_threshold,
        minimum_entrapment_count_for_stable_estimate: 3,
        verbose_diagnostics: false,
    });
    Ok(settings)
}

fn artifact_contains_model(artifacts: &DfRunArtifacts, model: &ModelFit) -> bool {
    match model {
        ModelFit::Moments => artifacts.moments.is_some(),
        ModelFit::Mle => artifacts.mle.is_some(),
        ModelFit::LowerOrder => artifacts.lower_order.is_some(),
        ModelFit::Msfdr => artifacts.msfdr_seeded.is_some(),
        ModelFit::Msfdr1Smix => artifacts.msfdr_1smix.is_some(),
        ModelFit::Msfdr2Smix => artifacts.msfdr_2smix.is_some(),
        ModelFit::Nokoi => artifacts.nokoi.is_some(),
        ModelFit::Ensemble => false,
    }
}

fn select_training_windows(
    manifest: &HoldoutPreregistration,
    features: &[DfFeature],
    base: &FdrSettings,
    database: &IndexedDatabase,
) -> Result<(
    Vec<ModelWindowSelection>,
    BTreeMap<String, Option<NullWindow>>,
    BTreeMap<String, Vec<NullWindowEvaluation>>,
)> {
    let mut selections = Vec::new();
    let mut windows = BTreeMap::new();
    let mut all_evaluations = BTreeMap::new();
    for grid in &manifest.model_grids {
        if let Some(window) = grid.fixed_window.clone() {
            anyhow::ensure!(
                grid.model == ModelFit::Msfdr1Smix && window.min_rank == 1 && window.max_rank == 1,
                "only MSFDR1-SMIX may use fixed 1..=1"
            );
            windows.insert(model_slug(&grid.model).into(), None);
            selections.push(ModelWindowSelection {
                model: grid.model.clone(),
                window: Some(window),
                evaluated_windows: 0,
                feasible_windows: 1,
                fallback_used: false,
                evaluation_sha256: sha256_bytes(b"fixed-msfdr1-smix-rank1-v1"),
            });
            continue;
        }
        let settings = optimizer_settings(base, grid, manifest)?;
        let optimized =
            optimize_null_window(features, &settings, database).map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            optimized.evaluations.len() == grid.candidates.len(),
            "optimizer did not evaluate the preregistered grid exactly"
        );
        anyhow::ensure!(
            artifact_contains_model(&optimized.artifacts, &grid.model),
            "optimizer fit fallback for {}",
            model_slug(&grid.model)
        );
        let selected = NullWindow {
            min_rank: optimized.report.selected_window.min_rank,
            max_rank: optimized.report.selected_window.max_rank,
        };
        anyhow::ensure!(
            grid.candidates
                .iter()
                .any(|candidate| candidate.min_rank == selected.min_rank
                    && candidate.max_rank == selected.max_rank),
            "selected window is outside preregistered grid"
        );
        let mut deterministic_evaluations = optimized.evaluations;
        for row in &mut deterministic_evaluations {
            row.elapsed_milliseconds = 0;
        }
        let feasible_windows = deterministic_evaluations
            .iter()
            .filter(|row| row.feasible)
            .count();
        let evaluation_sha256 = hash_serialized(&deterministic_evaluations)?;
        windows.insert(model_slug(&grid.model).into(), Some(selected.clone()));
        all_evaluations.insert(
            model_slug(&grid.model).into(),
            deterministic_evaluations.clone(),
        );
        selections.push(ModelWindowSelection {
            model: grid.model.clone(),
            window: Some(selected),
            evaluated_windows: deterministic_evaluations.len(),
            feasible_windows,
            fallback_used: false,
            evaluation_sha256,
        });
    }
    selections.sort_by(|left, right| model_slug(&left.model).cmp(model_slug(&right.model)));
    Ok((selections, windows, all_evaluations))
}

fn run_scored_stage(
    subset: &VerifiedSubset,
    settings: &FdrSettings,
    database: &IndexedDatabase,
    composition: &str,
    ratios: EffectiveRatios,
    search_space: SearchSpace,
) -> Result<CompletedStage> {
    anyhow::ensure!(
        subset
            .features
            .iter()
            .all(|feature| feature.core.external_features.ms2rescore_feature_joined),
        "subset contains a candidate without a joined parent annotation"
    );
    anyhow::ensure!(
        settings.null_window_optimizer.is_none(),
        "held-out stage must not contain an optimizer"
    );
    anyhow::ensure!(
        settings.external_profile_calibration.min_null_rank == 9
            && settings.external_profile_calibration.max_null_rank == 18,
        "held-out external profile window is not 9..=18"
    );
    let (mut features, mut artifacts) = sage_core::decoy_free_fdr::run_df_layers_with_artifacts(
        &subset.features,
        settings,
        database,
    );
    let profiles = apply_external_ms2rescore_bounded_experts(&mut features, settings)
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        profiles.calibration.min_null_rank == 9 && profiles.calibration.max_null_rank == 18,
        "resolved external profile changed"
    );
    artifacts.external_ms2rescore = Some(profiles.clone());
    features.retain(|feature| feature.core.rank == 1);
    let _ = calculate_peptide_q_df(&mut features, database, settings, settings.peptide_fdr);
    apply_peptide_q_to_psm_reporting_df(&mut features, settings);
    let _ = calculate_protein_q_df(&mut features, database, settings);
    let _ = apply_hierarchical_reporting_df(&mut features, database, settings);
    let unexplained_na_count = features
        .iter()
        .filter(|feature| {
            feature.decoy_free_p_value.is_none()
                || feature.decoy_free_pep.is_none()
                || feature.decoy_free_q_value.is_none()
                || feature.decoy_free_peptide_q.is_none()
                || feature.decoy_free_protein_q.is_none()
        })
        .count();
    anyhow::ensure!(
        unexplained_na_count == 0,
        "held-out stage produced unexplained NA values"
    );
    let threshold = settings.peptide_fdr as f64;
    let raw = collect_evidence(
        &features,
        database,
        &subset.identity.parent_search_fingerprint,
        threshold,
        false,
    )?;
    let level4 = collect_evidence(
        &features,
        database,
        &subset.identity.parent_search_fingerprint,
        threshold,
        true,
    )?;
    let summary = StageSummary {
        search_space,
        composition: composition.into(),
        subset_digest: subset.identity.digest.clone(),
        layers: vec![
            summarize_sets("raw_q", &raw, ratios, search_space),
            summarize_sets("level4", &level4, ratios, search_space),
        ],
        fallback_used: false,
        unexplained_na_count,
        external_profile_window: NullWindow {
            min_rank: 9,
            max_rank: 18,
        },
        external_profile_sha256: hash_serialized(&profiles)?,
        nuisance_state_provenance: match search_space {
            SearchSpace::PlusEntrapment => "refitted_in_held_out_plus_entrapment_subset".into(),
            SearchSpace::TargetOnly => "refitted_in_held_out_target_only_subset".into(),
        },
        complete_artifact_reused: false,
        window_retuned: false,
    };
    Ok(CompletedStage {
        summary,
        raw,
        level4,
        artifacts,
        rank1_features: features,
    })
}

fn canonical_peptide(peptide: &str) -> String {
    let mut output = String::new();
    let mut bracket_depth = 0_u32;
    for character in peptide.chars() {
        match character {
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            _ if bracket_depth == 0 && character.is_ascii_alphabetic() => {
                let upper = character.to_ascii_uppercase();
                output.push(if upper == 'I' { 'L' } else { upper });
            }
            _ => {}
        }
    }
    output
}

fn canonical_peptidoform(peptide: &str) -> String {
    let mut output = String::new();
    let mut bracket_depth = 0_u32;
    for character in peptide.trim().chars() {
        match character {
            '[' => {
                bracket_depth += 1;
                output.push(character);
            }
            ']' => {
                bracket_depth = bracket_depth.saturating_sub(1);
                output.push(character);
            }
            _ if bracket_depth == 0 && character.is_ascii_alphabetic() => {
                let upper = character.to_ascii_uppercase();
                output.push(if upper == 'I' { 'L' } else { upper });
            }
            _ => output.push(character),
        }
    }
    output
}

fn classify_proteins(proteins: &str) -> Option<bool> {
    if proteins.contains("Cont_") {
        return None;
    }
    let mut target = false;
    let mut entrapment = false;
    for protein in proteins
        .split(';')
        .map(str::trim)
        .filter(|protein| !protein.is_empty())
    {
        if protein.contains("Ent_") {
            entrapment = true;
        } else {
            target = true;
        }
    }
    match (target, entrapment) {
        (true, false) => Some(false),
        (false, true) => Some(true),
        _ => None,
    }
}

fn inferred_protein(feature: &DfFeature, database: &IndexedDatabase) -> Option<String> {
    if let (Some(group), 1) = (
        feature.protein_groups.as_deref(),
        feature.num_protein_groups,
    ) {
        return (!group.is_empty() && !group.contains(';')).then(|| group.to_owned());
    }
    let peptide = &database[feature.core.peptide_idx];
    (peptide.proteins.len() == 1)
        .then(|| peptide.proteins(&database.decoy_tag, database.generate_decoys))
}

fn collect_evidence(
    features: &[DfFeature],
    database: &IndexedDatabase,
    search_fingerprint: &str,
    threshold: f64,
    level4: bool,
) -> Result<EvidenceSets> {
    let mut sets = EvidenceSets::default();
    for feature in features
        .iter()
        .filter(|feature| feature.core.rank == 1 && feature.core.label == 1)
    {
        let peptide = database[feature.core.peptide_idx].to_string();
        let proteins = database[feature.core.peptide_idx]
            .proteins(&database.decoy_tag, database.generate_decoys);
        let Some(entrapment) = classify_proteins(&proteins) else {
            continue;
        };
        let psm_key = stable_candidate_id(search_fingerprint, &feature.core, &peptide);
        let peptide_key = canonical_peptide(&peptide);
        let peptidoform_key = canonical_peptidoform(&peptide);
        let psm_ok = feature.decoy_free_q_value.is_some_and(|q| q <= threshold)
            && (!level4 || feature.decoy_free_peptide_supported_psm == Some(true));
        let peptide_ok = feature.decoy_free_peptide_q.is_some_and(|q| q <= threshold)
            && (!level4 || feature.decoy_free_protein_supported_peptide == Some(true));
        let protein_key = inferred_protein(feature, database);
        let protein_ok = feature.decoy_free_protein_q.is_some_and(|q| q <= threshold);
        let (psms, peptides, peptidoforms, proteins_set) = if entrapment {
            (
                &mut sets.entrapment_psms,
                &mut sets.entrapment_peptides,
                &mut sets.entrapment_peptidoforms,
                &mut sets.entrapment_proteins,
            )
        } else {
            (
                &mut sets.target_psms,
                &mut sets.target_peptides,
                &mut sets.target_peptidoforms,
                &mut sets.target_proteins,
            )
        };
        if psm_ok {
            psms.insert(psm_key);
        }
        if peptide_ok {
            peptides.insert(peptide_key);
            peptidoforms.insert(peptidoform_key);
        }
        if protein_ok {
            if let Some(protein) = protein_key {
                if classify_proteins(&protein) == Some(entrapment) {
                    proteins_set.insert(protein);
                }
            }
        }
    }
    Ok(sets)
}

fn wilson_adjusted(
    target: usize,
    entrapment: usize,
    ratio: f64,
    has_entrapment: bool,
) -> (Option<f64>, Option<[f64; 2]>) {
    let n = target + entrapment;
    if !has_entrapment || n == 0 || !ratio.is_finite() || ratio <= 0.0 {
        return (None, None);
    }
    let scale = 1.0 + 1.0 / ratio;
    let p = entrapment as f64 / n as f64;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n as f64;
    let center = (p + z2 / (2.0 * n as f64)) / denominator;
    let half =
        z * ((p * (1.0 - p) / n as f64 + z2 / (4.0 * (n as f64).powi(2))).sqrt()) / denominator;
    (
        Some((p * scale).clamp(0.0, 1.0)),
        Some([
            ((center - half) * scale).clamp(0.0, 1.0),
            ((center + half) * scale).clamp(0.0, 1.0),
        ]),
    )
}

fn count_set(
    target: usize,
    entrapment: usize,
    ratio: f64,
    search_space: SearchSpace,
) -> CountWithEntrapment {
    let (ratio_adjusted_fdp, ratio_adjusted_fdp_wilson_95) = wilson_adjusted(
        target,
        entrapment,
        ratio,
        search_space == SearchSpace::PlusEntrapment,
    );
    CountWithEntrapment {
        target,
        entrapment,
        ratio_adjusted_fdp,
        ratio_adjusted_fdp_wilson_95,
    }
}

fn summarize_sets(
    layer: &str,
    sets: &EvidenceSets,
    ratios: EffectiveRatios,
    search_space: SearchSpace,
) -> LayerSummary {
    LayerSummary {
        layer: layer.into(),
        psm: count_set(
            sets.target_psms.len(),
            sets.entrapment_psms.len(),
            ratios.psm,
            search_space,
        ),
        peptide: count_set(
            sets.target_peptides.len(),
            sets.entrapment_peptides.len(),
            ratios.peptide,
            search_space,
        ),
        peptidoform: count_set(
            sets.target_peptidoforms.len(),
            sets.entrapment_peptidoforms.len(),
            ratios.psm,
            search_space,
        ),
        protein: count_set(
            sets.target_proteins.len(),
            sets.entrapment_proteins.len(),
            ratios.protein,
            search_space,
        ),
    }
}

fn incremental(
    layer: &str,
    baseline: &EvidenceSets,
    comparison: &EvidenceSets,
) -> IncrementalEvidence {
    IncrementalEvidence {
        layer: layer.into(),
        added_target_psms: comparison
            .target_psms
            .difference(&baseline.target_psms)
            .count(),
        lost_target_psms: baseline
            .target_psms
            .difference(&comparison.target_psms)
            .count(),
        added_target_peptides: comparison
            .target_peptides
            .difference(&baseline.target_peptides)
            .count(),
        lost_target_peptides: baseline
            .target_peptides
            .difference(&comparison.target_peptides)
            .count(),
        added_target_peptidoforms: comparison
            .target_peptidoforms
            .difference(&baseline.target_peptidoforms)
            .count(),
        lost_target_peptidoforms: baseline
            .target_peptidoforms
            .difference(&comparison.target_peptidoforms)
            .count(),
        added_target_proteins: comparison
            .target_proteins
            .difference(&baseline.target_proteins)
            .count(),
        lost_target_proteins: baseline
            .target_proteins
            .difference(&comparison.target_proteins)
            .count(),
    }
}

fn summary_layer<'a>(summary: &'a StageSummary, layer: &str) -> Result<&'a LayerSummary> {
    summary
        .layers
        .iter()
        .find(|row| row.layer == layer)
        .with_context(|| format!("{} has no {layer} summary", summary.composition))
}

fn composition_gate_models(
    manifest: &HoldoutPreregistration,
    requested_experts: &[ModelFit],
) -> Result<Vec<ModelFit>> {
    let requested_models = manifest
        .model_grids
        .iter()
        .filter(|grid| requested_experts.contains(&grid.model))
        .map(|grid| grid.model.clone())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        requested_models.len() == requested_experts.len(),
        "composition training-gate expert set is incomplete"
    );
    Ok(requested_models)
}

fn build_training_gates(
    manifest: &HoldoutPreregistration,
    windows: &BTreeMap<String, Option<NullWindow>>,
    plus: &BTreeMap<String, CompletedStage>,
    target: &BTreeMap<String, CompletedStage>,
    requested_experts: &[ModelFit],
) -> Result<Vec<ExpertTrainingGate>> {
    let requested_models = composition_gate_models(manifest, requested_experts)?;
    let mut order = requested_models
        .iter()
        .map(|grid| {
            let slug = model_slug(grid);
            let stage = plus
                .get(slug)
                .with_context(|| format!("missing training +entrapment stage for {slug}"))?;
            Ok((grid.clone(), stage.level4.target_peptides.len()))
        })
        .collect::<Result<Vec<_>>>()?;
    order.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| model_slug(&left.0).cmp(model_slug(&right.0)))
    });
    let mut union = BTreeSet::new();
    let mut gates = Vec::new();
    for (model, _) in order {
        let slug = model_slug(&model);
        let plus_stage = plus.get(slug).context("missing +entrapment gate stage")?;
        let target_stage = target.get(slug).context("missing target-only gate stage")?;
        let plus_level4 = summary_layer(&plus_stage.summary, "level4")?;
        let mut reasons = Vec::new();
        let mut warnings = Vec::new();
        if plus_stage.summary.fallback_used || target_stage.summary.fallback_used {
            reasons.push("fit fallback occurred".into());
        }
        if plus_stage.summary.unexplained_na_count > 0
            || target_stage.summary.unexplained_na_count > 0
        {
            reasons.push("unexplained NA values occurred".into());
        }
        if !plus_level4
            .peptide
            .ratio_adjusted_fdp
            .is_some_and(|fdp| fdp <= manifest.runtime_gates.fdr_threshold)
        {
            reasons.push("Level-4 peptide entrapment FDP is missing or exceeds threshold".into());
        }
        if plus_level4.peptide.entrapment
            < manifest
                .runtime_gates
                .minimum_entrapment_peptides_for_stable_estimate
        {
            warnings.push(format!(
                "Level-4 peptide entrapment count {} is below stability minimum {}",
                plus_level4.peptide.entrapment,
                manifest
                    .runtime_gates
                    .minimum_entrapment_peptides_for_stable_estimate
            ));
        }
        let calibration_peptides = plus_stage.level4.target_peptides.len();
        let target_peptides = target_stage.level4.target_peptides.len();
        if calibration_peptides > 0 {
            let loss = (calibration_peptides as f64 - target_peptides as f64)
                / calibration_peptides as f64;
            if loss
                > manifest
                    .runtime_gates
                    .maximum_target_only_peptide_fraction_loss
            {
                reasons.push(format!(
                    "target-only peptide transfer loss is {:.1}%",
                    100.0 * loss
                ));
            }
        }
        let incremental_peptides = plus_stage.level4.target_peptides.difference(&union).count();
        if reasons.is_empty()
            && incremental_peptides
                < manifest
                    .runtime_gates
                    .minimum_incremental_level4_target_peptides
        {
            reasons.push(format!(
                "adds only {incremental_peptides} new Level-4 target peptides"
            ));
        }
        let eligible = reasons.is_empty();
        if eligible {
            union.extend(plus_stage.level4.target_peptides.iter().cloned());
        }
        gates.push(ExpertTrainingGate {
            model: model.clone(),
            selected_window: windows.get(slug).cloned().flatten(),
            eligible,
            participation_reason: if eligible {
                "included"
            } else {
                "excluded_by_training_runtime_gates"
            }
            .into(),
            reasons,
            warnings,
            calibration_level4_target_peptides: calibration_peptides,
            target_only_level4_target_peptides: target_peptides,
            incremental_level4_target_peptides: incremental_peptides,
        });
    }
    gates.sort_by(|left, right| model_slug(&left.model).cmp(model_slug(&right.model)));
    Ok(gates)
}

fn interaction_comparison(
    manifest: &HoldoutPreregistration,
    baseline_id: &str,
    comparison_id: &str,
    baseline_plus: &CompletedStage,
    comparison_plus: &CompletedStage,
    baseline_target: &CompletedStage,
    comparison_target: &CompletedStage,
) -> Result<CompositionComparison> {
    let plus_incremental = vec![
        incremental("raw_q", &baseline_plus.raw, &comparison_plus.raw),
        incremental("level4", &baseline_plus.level4, &comparison_plus.level4),
    ];
    let target_incremental = vec![
        incremental("raw_q", &baseline_target.raw, &comparison_target.raw),
        incremental("level4", &baseline_target.level4, &comparison_target.level4),
    ];
    let mut interaction_deltas = Vec::new();
    for layer in ["raw_q", "level4"] {
        let baseline = summary_layer(&baseline_plus.summary, layer)?;
        let final_result = summary_layer(&comparison_plus.summary, layer)?;
        for (level, before, after) in [
            ("psm", &baseline.psm, &final_result.psm),
            ("peptide", &baseline.peptide, &final_result.peptide),
            (
                "peptidoform",
                &baseline.peptidoform,
                &final_result.peptidoform,
            ),
        ] {
            let absolute_change = before
                .ratio_adjusted_fdp
                .zip(after.ratio_adjusted_fdp)
                .map(|(x, y)| y - x);
            interaction_deltas.push(InteractionDelta {
                layer: layer.into(),
                level: level.into(),
                baseline_fdp: before.ratio_adjusted_fdp,
                final_fdp: after.ratio_adjusted_fdp,
                absolute_change,
                raw_q_warning: layer == "raw_q"
                    && absolute_change.is_some_and(|change| {
                        change > manifest.runtime_gates.raw_q_interaction_warning_threshold
                    }),
            });
        }
    }
    let final_peptide_fdp = summary_layer(&comparison_plus.summary, "level4")?
        .peptide
        .ratio_adjusted_fdp;
    let final_level4_calibration_decision = match final_peptide_fdp {
        None => "not_evaluable_missing_final_level4_peptide_fdp",
        Some(fdp) if fdp > manifest.runtime_gates.fdr_threshold => {
            "not_eligible_final_level4_peptide_fdp_exceeds_threshold"
        }
        Some(_) => "pass",
    }
    .into();
    Ok(CompositionComparison {
        baseline: baseline_id.into(),
        comparison: comparison_id.into(),
        plus_entrapment_incremental: plus_incremental,
        target_only_incremental: target_incremental,
        interaction_deltas,
        final_level4_calibration_decision,
    })
}

fn aggregate_comparison(
    manifest: &HoldoutPreregistration,
    baseline: &AggregateCompositionSummary,
    comparison: &AggregateCompositionSummary,
    baseline_plus: (&EvidenceSets, &EvidenceSets),
    comparison_plus: (&EvidenceSets, &EvidenceSets),
    baseline_target: (&EvidenceSets, &EvidenceSets),
    comparison_target: (&EvidenceSets, &EvidenceSets),
) -> Result<CompositionComparison> {
    let plus_entrapment_incremental = vec![
        incremental("raw_q", baseline_plus.0, comparison_plus.0),
        incremental("level4", baseline_plus.1, comparison_plus.1),
    ];
    let target_only_incremental = vec![
        incremental("raw_q", baseline_target.0, comparison_target.0),
        incremental("level4", baseline_target.1, comparison_target.1),
    ];
    let mut interaction_deltas = Vec::new();
    for layer in ["raw_q", "level4"] {
        let before = baseline
            .plus_entrapment
            .iter()
            .find(|row| row.layer == layer)
            .context("aggregate baseline layer is missing")?;
        let after = comparison
            .plus_entrapment
            .iter()
            .find(|row| row.layer == layer)
            .context("aggregate comparison layer is missing")?;
        for (level, left, right) in [
            ("psm", &before.psm, &after.psm),
            ("peptide", &before.peptide, &after.peptide),
            ("peptidoform", &before.peptidoform, &after.peptidoform),
        ] {
            let absolute_change = left
                .ratio_adjusted_fdp
                .zip(right.ratio_adjusted_fdp)
                .map(|(x, y)| y - x);
            interaction_deltas.push(InteractionDelta {
                layer: layer.into(),
                level: level.into(),
                baseline_fdp: left.ratio_adjusted_fdp,
                final_fdp: right.ratio_adjusted_fdp,
                absolute_change,
                raw_q_warning: layer == "raw_q"
                    && absolute_change.is_some_and(|change| {
                        change > manifest.runtime_gates.raw_q_interaction_warning_threshold
                    }),
            });
        }
    }
    let final_level4_calibration_decision = match comparison
        .plus_entrapment
        .iter()
        .find(|row| row.layer == "level4")
        .and_then(|row| row.peptide.ratio_adjusted_fdp)
    {
        None => "not_evaluable_missing_final_level4_peptide_fdp",
        Some(fdp) if fdp > manifest.runtime_gates.fdr_threshold => {
            "not_eligible_final_level4_peptide_fdp_exceeds_threshold"
        }
        Some(_) => "pass",
    }
    .into();
    Ok(CompositionComparison {
        baseline: baseline.composition.clone(),
        comparison: comparison.composition.clone(),
        plus_entrapment_incremental,
        target_only_incremental,
        interaction_deltas,
        final_level4_calibration_decision,
    })
}

fn write_training_artifact(
    root: &Path,
    manifest: &HoldoutPreregistration,
    manifest_sha256: &str,
    training_subset: &WithinParentSubsetIdentity,
    model: ModelFit,
    window: Option<NullWindow>,
    artifacts: &DfRunArtifacts,
) -> Result<WithinParentHoldoutArtifact> {
    anyhow::ensure!(
        artifact_contains_model(artifacts, &model),
        "training artifact is missing {}",
        model_slug(&model)
    );
    let model_root = root.join("training_artifacts").join(model_slug(&model));
    std::fs::create_dir_all(&model_root)?;
    let payload_path = model_root.join("fitted_model_artifacts.json");
    write_json_atomic(&payload_path, artifacts)?;
    let mut artifact = WithinParentHoldoutArtifact {
        schema: ARTIFACT_SCHEMA.into(),
        digest: String::new(),
        manifest_sha256: manifest_sha256.into(),
        training_subset_digest: training_subset.digest.clone(),
        parent_dataset_fingerprint: manifest.parent_dataset_fingerprint.clone(),
        model,
        selected_window: window,
        fitted_payload_sha256: sha256_file(&payload_path)?,
        nuisance_state_provenance: "fitted_from_training_subset_only_not_transferred_to_held_out"
            .into(),
        complete_artifact_transfer_allowed: false,
    };
    artifact.digest = hash_serialized(&artifact)?;
    write_json_atomic(
        &model_root.join("within_parent_holdout_artifact.json"),
        &artifact,
    )?;
    Ok(artifact)
}

fn build_holdout_lock(
    manifest: &HoldoutPreregistration,
    manifest_sha256: &str,
    fold: usize,
    training_subset: &WithinParentSubsetIdentity,
    artifacts: &[WithinParentHoldoutArtifact],
    composition: &HoldoutComposition,
    gates: &[ExpertTrainingGate],
) -> Result<WithinParentHoldoutLock> {
    let mut experts = artifacts
        .iter()
        .filter(|artifact| composition.requested_experts.contains(&artifact.model))
        .map(|artifact| {
            let gate = gates
                .iter()
                .find(|gate| gate.model == artifact.model)
                .with_context(|| {
                    format!("missing training gate for {}", model_slug(&artifact.model))
                })?;
            Ok(HoldoutLockExpert {
                model: artifact.model.clone(),
                selected_window: artifact.selected_window.clone(),
                training_artifact_digest: artifact.digest.clone(),
                enabled: gate.eligible,
                participation_reason: gate.participation_reason.clone(),
                gate_reasons: gate.reasons.clone(),
                gate_warnings: gate.warnings.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    experts.sort_by(|left, right| model_slug(&left.model).cmp(model_slug(&right.model)));
    anyhow::ensure!(
        experts.len() == composition.requested_experts.len(),
        "holdout lock expert count mismatch"
    );
    let enabled = experts.iter().filter(|expert| expert.enabled).count();
    anyhow::ensure!(
        enabled >= manifest.runtime_gates.minimum_ensemble_experts,
        "composition {} has only {enabled} runtime-eligible experts; {} required",
        composition.id,
        manifest.runtime_gates.minimum_ensemble_experts
    );
    let mut lock = WithinParentHoldoutLock {
        schema: LOCK_SCHEMA.into(),
        digest: String::new(),
        manifest_sha256: manifest_sha256.into(),
        parent_dataset_fingerprint: manifest.parent_dataset_fingerprint.clone(),
        training_subset_digest: training_subset.digest.clone(),
        fold,
        composition: composition.id.clone(),
        experts,
        external_profile_window: manifest.external_profile_window.clone(),
        target_only_policy: "refit_with_locked_window".into(),
        production_evidence: false,
    };
    lock.digest = hash_serialized(&lock)?;
    Ok(lock)
}

fn validate_holdout_lock(
    lock: &WithinParentHoldoutLock,
    manifest: &HoldoutPreregistration,
    manifest_sha256: &str,
    training_subset: &WithinParentSubsetIdentity,
) -> Result<()> {
    anyhow::ensure!(lock.schema == LOCK_SCHEMA, "invalid holdout lock schema");
    anyhow::ensure!(
        !lock.production_evidence,
        "validation-only lock cannot be production evidence"
    );
    anyhow::ensure!(
        lock.manifest_sha256 == manifest_sha256,
        "holdout lock manifest mismatch"
    );
    anyhow::ensure!(
        lock.parent_dataset_fingerprint == manifest.parent_dataset_fingerprint,
        "cross-parent holdout lock application is prohibited"
    );
    anyhow::ensure!(
        lock.training_subset_digest == training_subset.digest,
        "holdout lock training subset mismatch"
    );
    anyhow::ensure!(
        lock.external_profile_window.min_rank == 9 && lock.external_profile_window.max_rank == 18,
        "holdout lock external profile mismatch"
    );
    let unique = lock
        .experts
        .iter()
        .map(|expert| model_slug(&expert.model))
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        unique.len() == lock.experts.len(),
        "holdout lock contains duplicate experts"
    );
    for expert in &lock.experts {
        anyhow::ensure!(
            (expert.enabled
                && expert.participation_reason == "included"
                && expert.gate_reasons.is_empty())
                || (!expert.enabled
                    && expert.participation_reason == "excluded_by_training_runtime_gates"
                    && !expert.gate_reasons.is_empty()),
            "holdout expert {} has inconsistent runtime-gate provenance",
            model_slug(&expert.model)
        );
    }
    let mut unhashed = lock.clone();
    unhashed.digest.clear();
    anyhow::ensure!(
        hash_serialized(&unhashed)? == lock.digest,
        "holdout lock digest mismatch"
    );
    Ok(())
}

fn windows_from_lock(lock: &WithinParentHoldoutLock) -> BTreeMap<String, Option<NullWindow>> {
    lock.experts
        .iter()
        .filter(|expert| expert.enabled)
        .map(|expert| {
            let window = if expert.model == ModelFit::Msfdr1Smix {
                None
            } else {
                expert.selected_window.clone()
            };
            (model_slug(&expert.model).into(), window)
        })
        .collect()
}

fn experts_from_lock(lock: &WithinParentHoldoutLock) -> Vec<ModelFit> {
    lock.experts
        .iter()
        .filter(|expert| expert.enabled)
        .map(|expert| expert.model.clone())
        .collect()
}

fn aggregate_layers(
    raw: &EvidenceSets,
    level4: &EvidenceSets,
    ratios: EffectiveRatios,
    space: SearchSpace,
) -> Vec<LayerSummary> {
    vec![
        summarize_sets("raw_q", raw, ratios, space),
        summarize_sets("level4", level4, ratios, space),
    ]
}

fn recompute_out_of_fold_evidence(
    mut rank1_features: Vec<DfFeature>,
    database: &IndexedDatabase,
    settings: &FdrSettings,
    search_fingerprint: &str,
    ratios: EffectiveRatios,
    search_space: SearchSpace,
) -> Result<(EvidenceSets, EvidenceSets)> {
    let mut stable_ids = HashSet::with_capacity(rank1_features.len());
    for feature in &mut rank1_features {
        anyhow::ensure!(
            feature.core.rank == 1,
            "out-of-fold aggregate contains a non-rank-1 candidate"
        );
        let peptide = database[feature.core.peptide_idx].to_string();
        let stable_id = stable_candidate_id(search_fingerprint, &feature.core, &peptide);
        anyhow::ensure!(
            stable_ids.insert(stable_id),
            "a rank-1 PSM appears in more than one held-out fold"
        );
        feature.decoy_free_peptide_q = None;
        feature.decoy_free_protein_q = None;
        feature.decoy_free_protein_supported_peptide = None;
        feature.decoy_free_peptide_supported_psm = None;
        feature.protein_groups = None;
        feature.num_protein_groups = 0;
    }
    let _ = calculate_peptide_q_df(
        &mut rank1_features,
        database,
        settings,
        settings.peptide_fdr,
    );
    apply_peptide_q_to_psm_reporting_df(&mut rank1_features, settings);
    let _ = calculate_protein_q_df(&mut rank1_features, database, settings);
    let _ = apply_hierarchical_reporting_df(&mut rank1_features, database, settings);
    let threshold = settings.peptide_fdr as f64;
    let raw = collect_evidence(
        &rank1_features,
        database,
        search_fingerprint,
        threshold,
        false,
    )?;
    let level4 = collect_evidence(
        &rank1_features,
        database,
        search_fingerprint,
        threshold,
        true,
    )?;
    anyhow::ensure!(
        aggregate_layers(&raw, &level4, ratios, search_space).len() == 2,
        "out-of-fold aggregate layer construction failed"
    );
    Ok((raw, level4))
}

fn classify_result(
    manifest: &HoldoutPreregistration,
    folds: &[FoldSummary],
    comparison_id: &str,
    aggregate: &CompositionComparison,
) -> HoldoutClassification {
    let all_valid = folds.iter().all(|fold| fold.technically_valid);
    let level4 = aggregate
        .plus_entrapment_incremental
        .iter()
        .find(|row| row.layer == "level4");
    let positive = level4.is_some_and(|row| {
        row.added_target_peptides > row.lost_target_peptides
            || row.added_target_peptidoforms > row.lost_target_peptidoforms
            || row.added_target_proteins > row.lost_target_proteins
    });
    let contributing_folds = folds
        .iter()
        .filter(|fold| {
            fold.comparisons_to_baseline
                .iter()
                .find(|comparison| comparison.comparison == comparison_id)
                .and_then(|comparison| {
                    comparison
                        .plus_entrapment_incremental
                        .iter()
                        .find(|row| row.layer == "level4")
                })
                .is_some_and(|incremental| {
                    incremental.added_target_peptides > 0
                        || incremental.added_target_peptidoforms > 0
                        || incremental.added_target_proteins > 0
                })
        })
        .count();
    let not_single_fold = contributing_folds >= 2;
    let calibration_ok = aggregate.final_level4_calibration_decision == "pass"
        && folds.iter().all(|fold| {
            fold.comparisons_to_baseline
                .iter()
                .find(|comparison| comparison.comparison == comparison_id)
                .is_some_and(|comparison| comparison.final_level4_calibration_decision == "pass")
        });
    let target_stable = aggregate
        .target_only_incremental
        .iter()
        .find(|row| row.layer == "level4")
        .is_some_and(|row| {
            row.lost_target_peptides == 0
                && row.lost_target_peptidoforms == 0
                && row.lost_target_proteins == 0
        });
    let requested = manifest
        .comparison_matrix
        .iter()
        .find(|composition| composition.id == comparison_id)
        .map(|composition| {
            composition
                .requested_experts
                .iter()
                .filter(|expert| !manifest.baseline_experts.contains(expert))
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let requested_usable = !requested.is_empty()
        && folds.iter().all(|fold| {
            fold.compositions
                .iter()
                .find(|composition| composition.composition == comparison_id)
                .is_some_and(|composition| {
                    requested.iter().all(|model| {
                        composition
                            .lock
                            .experts
                            .iter()
                            .any(|expert| expert.model == *model && expert.enabled)
                    })
                })
        });
    let ready = all_valid
        && requested_usable
        && positive
        && not_single_fold
        && calibration_ok
        && target_stable;
    let mut reasons = Vec::new();
    if !all_valid {
        reasons.push("one or more folds failed technical validity".into());
    }
    if !requested_usable {
        reasons.push(
            "one or more requested diagnostic experts failed a training-side runtime gate".into(),
        );
    }
    if !positive {
        reasons.push(
            "aggregate held-out Level-4 higher-level target evidence did not increase".into(),
        );
    }
    if !not_single_fold {
        reasons.push(
            "incremental higher-level evidence was absent or driven by fewer than two folds".into(),
        );
    }
    if !calibration_ok {
        reasons.push("final Level-4 peptide calibration is missing or fails the existing release requirement".into());
    }
    if !target_stable {
        reasons.push("target-only Level-4 higher-level evidence was lost".into());
    }
    HoldoutClassification {
        validation_runner_integrity: if all_valid { "pass" } else { "fail" }.into(),
        fold_technical_validity: if all_valid { "pass" } else { "fail" }.into(),
        window_stability: "reported_without_requiring_identical_fold_windows".into(),
        calibration_behavior: if calibration_ok { "pass" } else { "fail" }.into(),
        incremental_useful_evidence: if positive && not_single_fold {
            "pass"
        } else {
            "fail"
        }
        .into(),
        target_only_stability: if target_stable { "pass" } else { "fail" }.into(),
        production_admission_readiness: if ready {
            "ready_for_controlled_policy_review"
        } else {
            "not_ready"
        }
        .into(),
        production_policy_changed: false,
        reasons,
    }
}

pub fn execute_holdout(manifest_path: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<()> {
    let manifest_path = manifest_path.as_ref();
    let manifest_bytes = std::fs::read(manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest: HoldoutPreregistration =
        serde_json::from_slice(&manifest_bytes).context("invalid holdout preregistration")?;
    validate_preregistration(&manifest)?;
    verify_running_build(&manifest.source_build)?;

    let output = output.as_ref();
    std::fs::create_dir_all(output)?;
    let preflight = preflight_for_manifest(&manifest, &manifest_sha256, &output.join("scratch"))?;
    let preflight_path = output.join("within_parent_holdout.preflight.json");
    write_json_atomic(&preflight_path, &preflight)?;
    let preflight_sha256 = sha256_file(&preflight_path)?;

    let plus_context =
        build_parent_context(&manifest.plus_entrapment, &output.join("scratch/plus-run"))?;
    let target_context =
        build_parent_context(&manifest.target_only, &output.join("scratch/target-run"))?;
    anyhow::ensure!(
        matches!(
            plus_context.search.external_features.use_mode,
            ExternalFeatureUseMode::BoundedDfExperts
        ),
        "holdout requires bounded external experts"
    );
    anyhow::ensure!(
        matches!(
            target_context.search.external_features.use_mode,
            ExternalFeatureUseMode::BoundedDfExperts
        ),
        "target holdout requires bounded external experts"
    );

    let mut fold_summaries = Vec::new();
    let mut aggregate_plus_features = BTreeMap::<String, Vec<DfFeature>>::new();
    let mut aggregate_target_features = BTreeMap::<String, Vec<DfFeature>>::new();
    for composition in &manifest.comparison_matrix {
        aggregate_plus_features.insert(composition.id.clone(), Vec::new());
        aggregate_target_features.insert(composition.id.clone(), Vec::new());
    }

    for fold in &manifest.folds {
        let fold_root = output.join(format!("fold_{}", fold.fold));
        std::fs::create_dir_all(&fold_root)?;
        let training = derive_subset(
            &manifest,
            &plus_context,
            &manifest_sha256,
            fold.fold,
            SubsetRole::Training,
            &fold.training_file_ids,
        )?;
        let training_target = derive_subset(
            &manifest,
            &target_context,
            &manifest_sha256,
            fold.fold,
            SubsetRole::Training,
            &fold.training_file_ids,
        )?;
        let (selected_windows, windows, mut evaluations) = select_training_windows(
            &manifest,
            &training.features,
            &plus_context.search.fdr,
            &plus_context.database,
        )?;
        let evaluation_root = fold_root.join("training_evaluations");
        std::fs::create_dir_all(&evaluation_root)?;
        for (model, rows) in evaluations.iter_mut() {
            for row in rows.iter_mut() {
                row.elapsed_milliseconds = 0;
            }
            write_json_atomic(&evaluation_root.join(format!("{model}.json")), rows)?;
        }

        let mut training_artifacts = Vec::new();
        let mut training_plus_stages = BTreeMap::new();
        let mut training_target_stages = BTreeMap::new();
        for grid in &manifest.model_grids {
            let model = grid.model.clone();
            let window = if model == ModelFit::Msfdr1Smix {
                None
            } else {
                windows.get(model_slug(&model)).cloned().flatten()
            };
            let plus_settings =
                settings_for_model(&plus_context.search.fdr, model.clone(), window.clone());
            let plus_stage = run_scored_stage(
                &training,
                &plus_settings,
                &plus_context.database,
                model_slug(&model),
                manifest.effective_ratios,
                SearchSpace::PlusEntrapment,
            )?;
            training_artifacts.push(write_training_artifact(
                &fold_root,
                &manifest,
                &manifest_sha256,
                &training.identity,
                model.clone(),
                window.clone(),
                &plus_stage.artifacts,
            )?);
            let target_settings =
                settings_for_model(&target_context.search.fdr, model.clone(), window);
            let target_stage = run_scored_stage(
                &training_target,
                &target_settings,
                &target_context.database,
                model_slug(&model),
                manifest.effective_ratios,
                SearchSpace::TargetOnly,
            )?;
            anyhow::ensure!(
                target_stage.summary.window_retuned == false
                    && target_stage.summary.complete_artifact_reused == false,
                "training target-only stage violated locked-window refit semantics"
            );
            training_plus_stages.insert(model_slug(&model).into(), plus_stage);
            training_target_stages.insert(model_slug(&model).into(), target_stage);
        }
        let mut training_gates = BTreeMap::new();
        let mut locks = BTreeMap::new();
        for composition in &manifest.comparison_matrix {
            let gates = build_training_gates(
                &manifest,
                &windows,
                &training_plus_stages,
                &training_target_stages,
                &composition.requested_experts,
            )?;
            write_json_atomic(
                &fold_root.join(format!("{}.training_gates.json", composition.id)),
                &gates,
            )?;
            let lock = build_holdout_lock(
                &manifest,
                &manifest_sha256,
                fold.fold,
                &training.identity,
                &training_artifacts,
                composition,
                &gates,
            )?;
            validate_holdout_lock(&lock, &manifest, &manifest_sha256, &training.identity)?;
            write_json_atomic(
                &fold_root.join(format!("{}.holdout.lock.json", composition.id)),
                &lock,
            )?;
            training_gates.insert(composition.id.clone(), gates);
            locks.insert(composition.id.clone(), lock);
        }
        drop(training);
        drop(training_target);
        drop(training_plus_stages);
        drop(training_target_stages);

        let held_plus = derive_subset(
            &manifest,
            &plus_context,
            &manifest_sha256,
            fold.fold,
            SubsetRole::HeldOut,
            &fold.held_out_file_ids,
        )?;
        let held_plus_digest = held_plus.identity.digest.clone();
        let held_target = derive_subset(
            &manifest,
            &target_context,
            &manifest_sha256,
            fold.fold,
            SubsetRole::HeldOut,
            &fold.held_out_file_ids,
        )?;
        let held_target_digest = held_target.identity.digest.clone();
        let mut fold_plus = BTreeMap::new();
        let mut fold_target = BTreeMap::new();
        let mut composition_summaries = Vec::new();
        for composition in &manifest.comparison_matrix {
            let lock = locks
                .get(&composition.id)
                .context("composition lock is missing")?;
            let experts = experts_from_lock(lock);
            let plus_settings = settings_for_ensemble(
                &plus_context.search.fdr,
                &windows_from_lock(lock),
                &experts,
            )?;
            let target_settings = settings_for_ensemble(
                &target_context.search.fdr,
                &windows_from_lock(lock),
                &experts,
            )?;
            anyhow::ensure!(
                target_settings.lower_order_frozen_artifact.is_none()
                    && target_settings.msfdr_seeded_frozen_model.is_none()
                    && target_settings.msfdr_2smix_frozen_model.is_none(),
                "complete +entrapment nuisance artifact leaked into target-only refit"
            );
            let plus_stage = run_scored_stage(
                &held_plus,
                &plus_settings,
                &plus_context.database,
                &composition.id,
                manifest.effective_ratios,
                SearchSpace::PlusEntrapment,
            )?;
            let target_stage = run_scored_stage(
                &held_target,
                &target_settings,
                &target_context.database,
                &composition.id,
                manifest.effective_ratios,
                SearchSpace::TargetOnly,
            )?;
            aggregate_plus_features
                .get_mut(&composition.id)
                .context("aggregate +entrapment composition is missing")?
                .extend(plus_stage.rank1_features.iter().cloned());
            aggregate_target_features
                .get_mut(&composition.id)
                .context("aggregate target-only composition is missing")?
                .extend(target_stage.rank1_features.iter().cloned());
            composition_summaries.push(FoldCompositionSummary {
                composition: composition.id.clone(),
                lock: lock.clone(),
                plus_entrapment: plus_stage.summary.clone(),
                target_only: target_stage.summary.clone(),
            });
            fold_plus.insert(composition.id.clone(), plus_stage);
            fold_target.insert(composition.id.clone(), target_stage);
        }
        let plus_profiles = composition_summaries
            .iter()
            .map(|summary| summary.plus_entrapment.external_profile_sha256.as_str())
            .collect::<BTreeSet<_>>();
        let target_profiles = composition_summaries
            .iter()
            .map(|summary| summary.target_only.external_profile_sha256.as_str())
            .collect::<BTreeSet<_>>();
        anyhow::ensure!(
            plus_profiles.len() == 1 && target_profiles.len() == 1,
            "composition-specific external profiles differ within one subset"
        );
        let baseline_plus = fold_plus.get("A").context("baseline A is missing")?;
        let baseline_target = fold_target
            .get("A")
            .context("baseline A target is missing")?;
        let mut comparisons = Vec::new();
        for composition in manifest
            .comparison_matrix
            .iter()
            .filter(|composition| composition.id != "A")
        {
            comparisons.push(interaction_comparison(
                &manifest,
                "A",
                &composition.id,
                baseline_plus,
                fold_plus
                    .get(&composition.id)
                    .context("comparison +entrapment stage is missing")?,
                baseline_target,
                fold_target
                    .get(&composition.id)
                    .context("comparison target-only stage is missing")?,
            )?);
        }
        let technically_valid = composition_summaries.iter().all(|summary| {
            !summary.plus_entrapment.fallback_used
                && !summary.target_only.fallback_used
                && summary.plus_entrapment.unexplained_na_count == 0
                && summary.target_only.unexplained_na_count == 0
                && !summary.plus_entrapment.window_retuned
                && !summary.target_only.window_retuned
                && !summary.plus_entrapment.complete_artifact_reused
                && !summary.target_only.complete_artifact_reused
        });
        let fold_summary = FoldSummary {
            fold: fold.fold,
            training_subset_digest: locks
                .get("A")
                .context("baseline lock is missing")?
                .training_subset_digest
                .clone(),
            held_out_plus_entrapment_subset_digest: held_plus_digest,
            held_out_target_only_subset_digest: held_target_digest,
            selected_windows,
            training_gates,
            compositions: composition_summaries,
            comparisons_to_baseline: comparisons,
            technically_valid,
        };
        write_json_atomic(&fold_root.join("fold_summary.json"), &fold_summary)?;
        fold_summaries.push(fold_summary);
    }

    let first_fold = fold_summaries
        .first()
        .context("holdout produced no folds")?;
    let mut aggregate_compositions = Vec::new();
    let mut aggregate_evidence =
        BTreeMap::<String, (EvidenceSets, EvidenceSets, EvidenceSets, EvidenceSets)>::new();
    for composition in &manifest.comparison_matrix {
        let fold_composition = first_fold
            .compositions
            .iter()
            .find(|row| row.composition == composition.id)
            .context("first-fold composition is missing")?;
        let experts = experts_from_lock(&fold_composition.lock);
        let plus_settings = settings_for_ensemble(
            &plus_context.search.fdr,
            &windows_from_lock(&fold_composition.lock),
            &experts,
        )?;
        let target_settings = settings_for_ensemble(
            &target_context.search.fdr,
            &windows_from_lock(&fold_composition.lock),
            &experts,
        )?;
        let (plus_raw, plus_level4) = recompute_out_of_fold_evidence(
            aggregate_plus_features
                .remove(&composition.id)
                .context("aggregate +entrapment features are missing")?,
            &plus_context.database,
            &plus_settings,
            &manifest.plus_entrapment.search_fingerprint,
            manifest.effective_ratios,
            SearchSpace::PlusEntrapment,
        )?;
        let (target_raw, target_level4) = recompute_out_of_fold_evidence(
            aggregate_target_features
                .remove(&composition.id)
                .context("aggregate target-only features are missing")?,
            &target_context.database,
            &target_settings,
            &manifest.target_only.search_fingerprint,
            manifest.effective_ratios,
            SearchSpace::TargetOnly,
        )?;
        aggregate_compositions.push(AggregateCompositionSummary {
            composition: composition.id.clone(),
            plus_entrapment: aggregate_layers(
                &plus_raw,
                &plus_level4,
                manifest.effective_ratios,
                SearchSpace::PlusEntrapment,
            ),
            target_only: aggregate_layers(
                &target_raw,
                &target_level4,
                manifest.effective_ratios,
                SearchSpace::TargetOnly,
            ),
        });
        aggregate_evidence.insert(
            composition.id.clone(),
            (plus_raw, plus_level4, target_raw, target_level4),
        );
    }
    let baseline_summary = aggregate_compositions
        .iter()
        .find(|row| row.composition == "A")
        .context("aggregate baseline A is missing")?;
    let baseline_evidence = aggregate_evidence
        .get("A")
        .context("aggregate baseline evidence is missing")?;
    let mut aggregate_comparisons = Vec::new();
    for composition in aggregate_compositions
        .iter()
        .filter(|row| row.composition != "A")
    {
        let evidence = aggregate_evidence
            .get(&composition.composition)
            .context("aggregate comparison evidence is missing")?;
        aggregate_comparisons.push(aggregate_comparison(
            &manifest,
            baseline_summary,
            composition,
            (&baseline_evidence.0, &baseline_evidence.1),
            (&evidence.0, &evidence.1),
            (&baseline_evidence.2, &baseline_evidence.3),
            (&evidence.2, &evidence.3),
        )?);
    }
    let aggregate = AggregateSummary {
        compositions: aggregate_compositions,
        comparisons_to_baseline: aggregate_comparisons.clone(),
        construction: "union of held-out rank-1 PSM identities exactly once, followed by aggregate peptide q-value calculation, protein inference/q-value calculation, hierarchical reporting, and canonical evidence-set counting; no fold-level peptide or protein counts are summed".into(),
        every_run_once: true,
    };
    let classifications = aggregate_comparisons
        .iter()
        .map(|comparison| {
            (
                comparison.comparison.clone(),
                classify_result(
                    &manifest,
                    &fold_summaries,
                    &comparison.comparison,
                    comparison,
                ),
            )
        })
        .collect();
    let result = HoldoutResult {
        schema: RESULT_SCHEMA.into(),
        manifest_sha256,
        preflight_sha256,
        source_build: manifest.source_build,
        capabilities: CAPABILITIES,
        folds: fold_summaries,
        aggregate,
        classifications,
    };
    write_json_atomic(&output.join("within_parent_holdout.result.json"), &result)
}

/// Resolve the exact frozen expert grids into a self-contained preregistration
/// before any fold outcome is evaluated.
pub fn create_preregistration(
    draft_path: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<()> {
    let mut draft: serde_json::Value =
        serde_json::from_slice(&std::fs::read(draft_path.as_ref())?)?;
    let object = draft
        .as_object_mut()
        .context("holdout draft must be a JSON object")?;
    let grid_source = object
        .remove("grid_source_manifest")
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .context("holdout draft is missing grid_source_manifest")?;
    let expected_grid_source_sha256 = object
        .get("grid_source_manifest_sha256")
        .and_then(serde_json::Value::as_str)
        .context("holdout draft is missing grid_source_manifest_sha256")?;
    anyhow::ensure!(
        sha256_file(Path::new(&grid_source))? == expected_grid_source_sha256,
        "candidate-grid source manifest hash mismatch"
    );
    let source: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&grid_source)
            .with_context(|| format!("reading grid source {grid_source}"))?,
    )?;
    let source_models = source
        .get("models")
        .and_then(serde_json::Value::as_array)
        .context("grid source has no models")?;
    let mut grids = Vec::new();
    for name in ["moments", "mle", "lower_order", "msfdr", "msfdr2_smix"] {
        let model = source_models
            .iter()
            .find(|model| model.get("model").and_then(serde_json::Value::as_str) == Some(name))
            .with_context(|| format!("grid source is missing {name}"))?;
        let candidates = model
            .get("candidate_windows")
            .and_then(serde_json::Value::as_array)
            .context("model grid is missing candidate_windows")?;
        anyhow::ensure!(!candidates.is_empty(), "model grid is empty");
        grids.push(serde_json::json!({"model": name, "candidates": candidates}));
    }
    grids.push(serde_json::json!({
        "model": "msfdr1_smix",
        "candidates": [],
        "fixed_window": {"min_rank": 1, "max_rank": 1}
    }));
    object.insert("model_grids".into(), serde_json::Value::Array(grids));
    let manifest: HoldoutPreregistration = serde_json::from_value(draft)?;
    validate_preregistration(&manifest)?;
    write_json_atomic(output.as_ref(), &manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::external_feature_cache::ExternalAnnotationRecord;
    use sage_core::ml::lower_order::{
        ChargeFillMode, LowerOrderArtifact, LowerOrderChargeParameters,
    };
    use sage_core::scoring::{ExternalPsmFeatures, FeatureCore};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "sage-within-parent-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn source_build() -> SourceBuildIdentity {
        SourceBuildIdentity {
            source_commit: "commit".into(),
            source_tree_sha256: "tree".into(),
            cargo_lock_sha256: "lock".into(),
            release_binary_sha256: "binary".into(),
            crate_version: "test".into(),
            rustc_version: "test".into(),
            cargo_version: "test".into(),
            target_triple: "test".into(),
            build_profile: "release".into(),
            enabled_features: vec!["within-parent-holdout".into()],
        }
    }

    fn parent(search_space: SearchSpace, root: &str) -> ParentSpaceIdentity {
        ParentSpaceIdentity {
            search_space,
            fasta: PathBuf::from(root).join("db.fasta"),
            fasta_sha256: "fasta".into(),
            search_config: PathBuf::from(root).join("search.json"),
            search_config_sha256: "config".into(),
            pool_manifest: PathBuf::from(root).join("pool/manifest.json"),
            pool_manifest_sha256: "pool-manifest".into(),
            search_fingerprint: format!("search-{search_space:?}"),
            pool_payload_sha256: "pool-payload".into(),
            candidate_count: 4,
            retained_rank_depth: 50,
            candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
            annotation_manifest: PathBuf::from(root).join("annotations/manifest.json"),
            annotation_manifest_sha256: "annotation-manifest".into(),
            annotation_fingerprint: "annotation".into(),
            annotation_payload_sha256: "annotation-payload".into(),
            annotation_count: 4,
            annotation_schema_version: EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION,
            annotation_feature_schema: EXTERNAL_ANNOTATION_FEATURE_SCHEMA.into(),
        }
    }

    fn manifest(root: &str) -> HoldoutPreregistration {
        let spectra = (0..9)
            .map(|ordinal| SpectrumFileIdentity {
                ordinal,
                filename: format!("run-{ordinal}.mzML"),
                path: PathBuf::from(root).join(format!("run-{ordinal}.mzML")),
                size_bytes: 100 + ordinal as u64,
                sha256: format!("spectrum-{ordinal}"),
            })
            .collect::<Vec<_>>();
        let folds = vec![
            HoldoutFold {
                fold: 1,
                training_file_ids: vec![1, 2, 4, 5, 7, 8],
                held_out_file_ids: vec![0, 3, 6],
            },
            HoldoutFold {
                fold: 2,
                training_file_ids: vec![0, 2, 3, 5, 6, 8],
                held_out_file_ids: vec![1, 4, 7],
            },
            HoldoutFold {
                fold: 3,
                training_file_ids: vec![0, 1, 3, 4, 6, 7],
                held_out_file_ids: vec![2, 5, 8],
            },
        ];
        HoldoutPreregistration {
            schema: PREREGISTRATION_SCHEMA.into(),
            study: "synthetic".into(),
            assignment_basis: "stable acquisition order round-robin; no identification, score, entrapment, or model outcome used".into(),
            parent_dataset_id: "parent".into(),
            parent_dataset_fingerprint: "dataset".into(),
            spectra,
            plus_entrapment: parent(SearchSpace::PlusEntrapment, root),
            target_only: parent(SearchSpace::TargetOnly, root),
            folds,
            grid_source_description: "fixture".into(),
            grid_source_manifest_sha256: "grid-source".into(),
            model_grids: Vec::new(),
            baseline_experts: vec![
                ModelFit::Moments,
                ModelFit::Mle,
                ModelFit::Msfdr1Smix,
                ModelFit::LowerOrder,
            ],
            comparison_matrix: vec![
                HoldoutComposition {
                    id: "A".into(),
                    requested_experts: vec![
                        ModelFit::Moments,
                        ModelFit::Mle,
                        ModelFit::Msfdr1Smix,
                        ModelFit::LowerOrder,
                    ],
                },
                HoldoutComposition {
                    id: "B".into(),
                    requested_experts: vec![
                        ModelFit::Moments,
                        ModelFit::Mle,
                        ModelFit::Msfdr1Smix,
                        ModelFit::LowerOrder,
                        ModelFit::Msfdr,
                    ],
                },
                HoldoutComposition {
                    id: "C".into(),
                    requested_experts: vec![
                        ModelFit::Moments,
                        ModelFit::Mle,
                        ModelFit::Msfdr1Smix,
                        ModelFit::LowerOrder,
                        ModelFit::Msfdr2Smix,
                    ],
                },
                HoldoutComposition {
                    id: "D".into(),
                    requested_experts: vec![
                        ModelFit::Moments,
                        ModelFit::Mle,
                        ModelFit::Msfdr1Smix,
                        ModelFit::LowerOrder,
                        ModelFit::Msfdr,
                        ModelFit::Msfdr2Smix,
                    ],
                },
            ],
            target_only_policies: [
                ("moments".into(), "refit_with_locked_window".into()),
                ("mle".into(), "refit_with_locked_window".into()),
                ("msfdr1_smix".into(), "refit_with_locked_window".into()),
                ("lower_order".into(), "refit_with_locked_window".into()),
                ("msfdr".into(), "refit_with_locked_window".into()),
                ("msfdr2_smix".into(), "refit_with_locked_window".into()),
            ]
            .into_iter()
            .collect(),
            external_profile_window: NullWindow {
                min_rank: 9,
                max_rank: 18,
            },
            effective_ratios: EffectiveRatios {
                psm: 0.75,
                peptide: 0.75,
                protein: 1.0,
            },
            optimizer_validation_scope: NullWindowValidationScope::Level4,
            optimizer_seed: 0,
            runtime_gates: HoldoutRuntimeGates {
                validation_scope: NullWindowValidationScope::Level4,
                fdr_threshold: 0.01,
                maximum_target_only_peptide_fraction_loss: 0.20,
                minimum_incremental_level4_target_peptides: 1,
                minimum_entrapment_peptides_for_stable_estimate: 3,
                minimum_ensemble_experts: 2,
                raw_q_interaction_warning_threshold: 0.01,
            },
            disclosures: vec![
                "fold assignments reused from the Lower Order validation".into(),
                "baseline Lower Order behavior has previously been observed".into(),
                "MSFDR/MSFDR2 fold-level Ensemble outcomes have not been evaluated previously".into(),
                "within-dataset run-level validation, not an external-dataset claim".into(),
            ],
            source_build: source_build(),
            aggregation_definition: "union".into(),
            uncertainty_method: "Wilson".into(),
            acceptance: HoldoutAcceptanceCriteria {
                require_all_folds_valid: true,
                require_no_fallback: true,
                require_positive_aggregate_target_evidence: true,
                require_no_material_calibration_deterioration: true,
                require_stable_target_only_transfer: true,
                require_not_single_fold_driven: true,
                fdr_threshold: 0.01,
                confidence_level: 0.95,
                maximum_absolute_ratio_adjusted_peptide_fdp_increase: 0.01,
            },
        }
    }

    fn record(id: &str, file_id: usize, spec_id: &str) -> ViewRecord {
        let core = FeatureCore {
            file_id,
            spec_id: spec_id.into(),
            ..FeatureCore::default()
        };
        ViewRecord {
            stable_id: id.into(),
            file_id,
            spec_id: spec_id.into(),
            core,
        }
    }

    fn candidate(id: &str, file_id: usize) -> CandidatePoolEntry {
        CandidatePoolEntry {
            stable_id: id.into(),
            peptide: "PEPTIDE".into(),
            core: FeatureCore {
                file_id,
                spec_id: format!("scan={file_id}"),
                ..FeatureCore::default()
            },
        }
    }

    fn annotation(id: &str) -> ExternalAnnotationRecord {
        let mut features = ExternalPsmFeatures::default();
        features.ms2rescore_feature_joined = true;
        ExternalAnnotationRecord {
            stable_id: id.into(),
            features,
        }
    }

    #[test]
    fn subset_identity_is_deterministic_and_path_portable() {
        let records = vec![
            record("a", 0, "scan=1"),
            record("b", 0, "scan=1"),
            record("c", 1, "scan=2"),
        ];
        let first_manifest = manifest("/linux/absolute/location");
        let second_manifest = manifest("/different/platform/location");
        let first = subset_identity(
            &first_manifest,
            &first_manifest.plus_entrapment,
            "manifest",
            1,
            SubsetRole::HeldOut,
            &[0],
            &records,
        )
        .unwrap();
        let second = subset_identity(
            &second_manifest,
            &second_manifest.plus_entrapment,
            "manifest",
            1,
            SubsetRole::HeldOut,
            &[0],
            &records,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.selected_candidate_count, 2);
        assert_eq!(first.selected_spectrum_count, 1);
    }

    #[test]
    fn fold_partitions_are_disjoint_and_exact() {
        let valid = manifest("/unused").folds;
        validate_fold_assignments(&valid, 9).unwrap();
        let mut duplicate = valid.clone();
        duplicate[0].held_out_file_ids[1] = duplicate[0].held_out_file_ids[0];
        assert!(validate_fold_assignments(&duplicate, 9).is_err());
        let mut overlap = valid.clone();
        overlap[1].held_out_file_ids[0] = 0;
        assert!(validate_fold_assignments(&overlap, 9).is_err());
        let mut missing = valid;
        missing[2].held_out_file_ids.pop();
        assert!(validate_fold_assignments(&missing, 9).is_err());
    }

    #[test]
    fn diagnostic_experts_cannot_consume_baseline_usefulness_union() {
        let mut manifest = manifest("/unused");
        manifest.model_grids = [
            ModelFit::Moments,
            ModelFit::Mle,
            ModelFit::LowerOrder,
            ModelFit::Msfdr,
            ModelFit::Msfdr1Smix,
            ModelFit::Msfdr2Smix,
        ]
        .into_iter()
        .map(|model| ModelGrid {
            model,
            candidates: Vec::new(),
            fixed_window: None,
        })
        .collect();
        let baseline = composition_gate_models(&manifest, &manifest.baseline_experts).unwrap();
        assert_eq!(baseline.len(), 4);
        assert!(!baseline.contains(&ModelFit::Msfdr));
        assert!(!baseline.contains(&ModelFit::Msfdr2Smix));
        let combined =
            composition_gate_models(&manifest, &manifest.comparison_matrix[3].requested_experts)
                .unwrap();
        assert_eq!(combined.len(), 6);
    }

    #[test]
    fn candidate_annotation_join_fails_closed() {
        assert!(verify_join_and_records(vec![candidate("a", 0)], Vec::new()).is_err());
        assert!(verify_join_and_records(
            vec![candidate("a", 0), candidate("a", 0)],
            vec![annotation("a"), annotation("b")]
        )
        .is_err());
        assert!(verify_join_and_records(
            vec![candidate("a", 0), candidate("b", 1)],
            vec![annotation("a"), annotation("a")]
        )
        .is_err());
        assert!(verify_join_and_records(vec![candidate("a", 0)], vec![annotation("b")]).is_err());
        assert_eq!(
            verify_join_and_records(vec![candidate("a", 0)], vec![annotation("a")])
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn corrupt_parent_payload_is_rejected() {
        let path = temp_path("payload");
        std::fs::write(&path, b"original").unwrap();
        let expected = sha256_file(&path).unwrap();
        verify_payload_hash(&path, &expected, "fixture").unwrap();
        std::fs::write(&path, b"changed").unwrap();
        assert!(verify_payload_hash(&path, &expected, "fixture").is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn subset_materialization_cannot_leak_parent_candidates() {
        let features = materialize_subset_features(
            vec![
                record("a", 0, "scan=1"),
                record("b", 1, "scan=2"),
                record("c", 2, "scan=3"),
            ],
            &[0, 2],
        );
        assert_eq!(features.len(), 2);
        assert!(features
            .iter()
            .all(|feature| feature.core.file_id == 0 || feature.core.file_id == 2));
        let training_before = materialize_subset_features(
            vec![record("a", 0, "scan=1"), record("b", 1, "scan=2")],
            &[0],
        );
        let mut changed_held_out = record("b", 1, "scan=2");
        changed_held_out.core.label = -1;
        let training_after =
            materialize_subset_features(vec![record("a", 0, "scan=1"), changed_held_out], &[0]);
        assert_eq!(
            serde_json::to_vec(&training_before).unwrap(),
            serde_json::to_vec(&training_after).unwrap()
        );
        assert!(verify_parent_run_coverage(
            &(0..9)
                .map(|file_id| record(&format!("id-{file_id}"), file_id, "scan"))
                .collect::<Vec<_>>(),
            9
        )
        .is_ok());
        assert!(verify_parent_run_coverage(&[record("a", 0, "scan")], 9).is_err());
    }

    #[test]
    fn locked_window_settings_refit_and_never_retune() {
        let options: sage_core::input::FdrOptions =
            serde_json::from_value(serde_json::json!({})).unwrap();
        let mut base: FdrSettings = options.into();
        base.lower_order_frozen_artifact = Some(LowerOrderArtifact {
            schema_version: 1,
            model_version: "fixture".into(),
            params_by_charge: vec![LowerOrderChargeParameters {
                charge: 2,
                mu: -1.0,
                beta: 0.5,
            }],
            charge_fill_mode: ChargeFillMode::MinimalDelta,
            fitted_charges_sorted: vec![2],
            max_fitted_charge: 2,
            null_rank_min: 6,
            null_rank_max: 9,
            evalue_candidate_count_power: 1.0,
            evalue_scale: 1.0,
            tev_transform: "NegLogE".into(),
            extrapolation_strength: 0.0,
            reference_candidate_counts: vec![10],
        });
        base.null_window_optimizer = Some(NullWindowOptimizerOptions {
            candidates: vec![NullWindowCandidate {
                min_rank: 1,
                max_rank: 2,
            }],
            strategy: NullWindowSearchStrategy::Explicit,
            bounds: None,
            adaptive: AdaptiveNullWindowSearchOptions::default(),
            validation_scope: NullWindowValidationScope::Level4,
            fdr_threshold: 0.01,
            psm_entrapment_ratio: 1.0,
            peptide_entrapment_ratio: 1.0,
            protein_entrapment_ratio: 1.0,
            maximum_entrapment_fdp: 0.01,
            minimum_entrapment_count_for_stable_estimate: 3,
            verbose_diagnostics: false,
        });
        let windows = [
            (
                "moments".into(),
                Some(NullWindow {
                    min_rank: 9,
                    max_rank: 18,
                }),
            ),
            (
                "mle".into(),
                Some(NullWindow {
                    min_rank: 8,
                    max_rank: 25,
                }),
            ),
            (
                "lower_order".into(),
                Some(NullWindow {
                    min_rank: 6,
                    max_rank: 9,
                }),
            ),
            (
                "msfdr".into(),
                Some(NullWindow {
                    min_rank: 9,
                    max_rank: 13,
                }),
            ),
            ("msfdr1_smix".into(), None),
            (
                "msfdr2_smix".into(),
                Some(NullWindow {
                    min_rank: 9,
                    max_rank: 17,
                }),
            ),
        ]
        .into_iter()
        .collect();
        let settings = settings_for_ensemble(
            &base,
            &windows,
            &[
                ModelFit::Moments,
                ModelFit::Mle,
                ModelFit::Msfdr1Smix,
                ModelFit::LowerOrder,
                ModelFit::Msfdr,
                ModelFit::Msfdr2Smix,
            ],
        )
        .unwrap();
        assert!(settings.null_window_optimizer.is_none());
        assert!(settings.lower_order_frozen_artifact.is_none());
        assert!(settings.msfdr_seeded_frozen_model.is_none());
        assert!(settings.msfdr_2smix_frozen_model.is_none());
        assert!(settings.enable_msfdr_seeded);
        assert!(settings.enable_msfdr_2smix);
        assert_eq!(
            (
                settings.lower_order_min_null_rank,
                settings.lower_order_max_null_rank
            ),
            (6, 9)
        );
    }

    #[test]
    fn validation_and_production_locks_cannot_masquerade() {
        let manifest = manifest("/unused");
        let subset = WithinParentSubsetIdentity {
            schema: SUBSET_SCHEMA.into(),
            digest: "subset".into(),
            parent_dataset_fingerprint: "dataset".into(),
            parent_search_fingerprint: "search".into(),
            parent_pool_payload_sha256: "pool".into(),
            parent_annotation_manifest_sha256: "manifest".into(),
            parent_annotation_payload_sha256: "annotations".into(),
            fold_manifest_sha256: "fold-manifest".into(),
            fold: 1,
            role: SubsetRole::Training,
            search_space: SearchSpace::PlusEntrapment,
            spectrum_files: Vec::new(),
            selected_file_ids: vec![0],
            ordered_stable_candidate_ids_sha256: "ids".into(),
            selected_spectrum_count: 1,
            selected_candidate_count: 1,
            candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
            annotation_schema_version: EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION,
            annotation_feature_schema: EXTERNAL_ANNOTATION_FEATURE_SCHEMA.into(),
            source_build: source_build(),
        };
        let artifacts = [
            ModelFit::Moments,
            ModelFit::Mle,
            ModelFit::Msfdr1Smix,
            ModelFit::LowerOrder,
        ]
        .into_iter()
        .map(|model| WithinParentHoldoutArtifact {
            schema: ARTIFACT_SCHEMA.into(),
            digest: model_slug(&model).into(),
            manifest_sha256: "manifest".into(),
            training_subset_digest: "subset".into(),
            parent_dataset_fingerprint: "dataset".into(),
            model,
            selected_window: None,
            fitted_payload_sha256: "payload".into(),
            nuisance_state_provenance: "training".into(),
            complete_artifact_transfer_allowed: false,
        })
        .collect::<Vec<_>>();
        let gates = artifacts
            .iter()
            .map(|artifact| ExpertTrainingGate {
                model: artifact.model.clone(),
                selected_window: artifact.selected_window.clone(),
                eligible: true,
                participation_reason: "included".into(),
                reasons: Vec::new(),
                warnings: Vec::new(),
                calibration_level4_target_peptides: 1,
                target_only_level4_target_peptides: 1,
                incremental_level4_target_peptides: 1,
            })
            .collect::<Vec<_>>();
        let composition = &manifest.comparison_matrix[0];
        let lock = build_holdout_lock(
            &manifest,
            "manifest",
            1,
            &subset,
            &artifacts,
            composition,
            &gates,
        )
        .unwrap();
        let mut reversed = artifacts.clone();
        reversed.reverse();
        let reversed_lock = build_holdout_lock(
            &manifest,
            "manifest",
            1,
            &subset,
            &reversed,
            composition,
            &gates,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&lock).unwrap(),
            serde_json::to_vec(&reversed_lock).unwrap()
        );
        let bytes = serde_json::to_vec(&lock).unwrap();
        assert!(serde_json::from_slice::<crate::workflow::EnsembleLock>(&bytes).is_err());
        let production = serde_json::json!({
            "schema_version": 5, "source_manifest_hash": "x", "experts": [], "minimum_required_experts": 0
        });
        assert!(serde_json::from_value::<WithinParentHoldoutLock>(production).is_err());
    }

    #[test]
    fn runner_capabilities_and_output_are_deterministic() {
        assert_eq!(
            CAPABILITIES,
            ValidationRunnerCapabilities {
                spectrum_search: false,
                annotation_generation: false,
                python_execution: false,
                ms2pip_execution: false,
                deeplc_execution: false,
                wrapper_execution: false,
            }
        );
        let records = vec![record("a", 0, "scan=1"), record("b", 0, "scan=2")];
        let manifest = manifest("/one");
        let first = subset_identity(
            &manifest,
            &manifest.plus_entrapment,
            "manifest",
            1,
            SubsetRole::Training,
            &[0],
            &records,
        )
        .unwrap();
        let second = subset_identity(
            &manifest,
            &manifest.plus_entrapment,
            "manifest",
            1,
            SubsetRole::Training,
            &[0],
            &records,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
    }

    #[test]
    fn generic_interaction_warning_is_informational_and_level4_remains_blocking() {
        fn count(fdp: Option<f64>) -> CountWithEntrapment {
            CountWithEntrapment {
                target: 100,
                entrapment: usize::from(fdp.is_some_and(|value| value > 0.0)),
                ratio_adjusted_fdp: fdp,
                ratio_adjusted_fdp_wilson_95: None,
            }
        }
        fn layer(name: &str, fdp: Option<f64>) -> LayerSummary {
            LayerSummary {
                layer: name.into(),
                psm: count(fdp),
                peptide: count(fdp),
                peptidoform: count(fdp),
                protein: count(fdp),
            }
        }
        let baseline = AggregateCompositionSummary {
            composition: "A".into(),
            plus_entrapment: vec![layer("raw_q", Some(0.001)), layer("level4", Some(0.008))],
            target_only: vec![layer("raw_q", None), layer("level4", None)],
        };
        let comparison = AggregateCompositionSummary {
            composition: "B".into(),
            plus_entrapment: vec![layer("raw_q", Some(0.02)), layer("level4", Some(0.009))],
            target_only: vec![layer("raw_q", None), layer("level4", None)],
        };
        let evidence = EvidenceSets::default();
        let result = aggregate_comparison(
            &manifest("/unused"),
            &baseline,
            &comparison,
            (&evidence, &evidence),
            (&evidence, &evidence),
            (&evidence, &evidence),
            (&evidence, &evidence),
        )
        .unwrap();
        assert_eq!(result.final_level4_calibration_decision, "pass");
        assert!(result
            .interaction_deltas
            .iter()
            .any(|delta| delta.layer == "raw_q" && delta.raw_q_warning));

        let mut failed = comparison;
        failed.plus_entrapment[1] = layer("level4", Some(0.011));
        let failed = aggregate_comparison(
            &manifest("/unused"),
            &baseline,
            &failed,
            (&evidence, &evidence),
            (&evidence, &evidence),
            (&evidence, &evidence),
            (&evidence, &evidence),
        )
        .unwrap();
        assert!(failed
            .final_level4_calibration_decision
            .starts_with("not_eligible"));
    }

    #[test]
    fn support_classification_distinguishes_corroboration_disagreement_and_nonevaluable() {
        assert_eq!(classify_support_evidence(2, 2, 3, false), "corroborated");
        assert_eq!(
            classify_support_evidence(1, 1, 3, false),
            "singly_supported_but_consistent"
        );
        assert_eq!(classify_support_evidence(1, 1, 3, true), "disputed");
        assert_eq!(
            classify_support_evidence(1, 1, 1, false),
            "uniquely_evaluable"
        );
        assert_eq!(classify_support_evidence(0, 0, 0, false), "not_evaluable");
        assert_eq!(classify_support_evidence(0, 0, 3, false), "not_accepted");
    }

    #[test]
    fn low_input_contract_distinguishes_novel_supporting_redundant_harmful_and_invalid() {
        assert_eq!(
            classify_low_input_expert(None, false, 2, 10, 3.0),
            "novel_contributor"
        );
        assert_eq!(
            classify_low_input_expert(None, false, 0, 10, 3.0),
            "supporting_corroborating_contributor"
        );
        assert_eq!(
            classify_low_input_expert(None, true, 0, 10, 3.0),
            "redundant"
        );
        assert_eq!(
            classify_low_input_expert(None, false, 0, 10, -0.5),
            "harmful"
        );
        assert_eq!(
            classify_low_input_expert(Some("poor calibration"), false, 4, 10, 5.0),
            "invalid"
        );
    }

    #[test]
    fn sequential_unique_credit_is_order_dependent_for_shared_evidence() {
        let accepted = [
            (
                "a".into(),
                ["shared", "a-only"].into_iter().map(String::from).collect(),
            ),
            (
                "b".into(),
                ["shared"].into_iter().map(String::from).collect(),
            ),
        ]
        .into_iter()
        .collect::<BTreeMap<String, BTreeSet<String>>>();
        let ab = sequential_unique_credit(&["a", "b"], &accepted, 1);
        let ba = sequential_unique_credit(&["b", "a"], &accepted, 1);
        assert_eq!(ab[1], ("b".into(), 0, false));
        assert_eq!(ba[0], ("b".into(), 1, true));
        assert_eq!(ba[1], ("a".into(), 1, true));
    }

    #[test]
    fn exact_redundancy_requires_identical_score_and_calibration_streams() {
        let metric = AuditPsmMetric {
            stable_id: "id".into(),
            spectrum_key: "run\u{1f}scan".into(),
            peptide: "PEPTIDE".into(),
            canonical_peptide: "PEPTLDE".into(),
            peptidoform: "PEPTLDE".into(),
            protein: Some("P1".into()),
            entrapment: Some(false),
            score: Some(2.0),
            p_value: Some(0.01),
            pep: Some(0.02),
            psm_q: Some(0.03),
            peptide_q: Some(0.03),
            protein_q: Some(0.03),
            psm_level4_supported: Some(true),
            peptide_level4_supported: Some(true),
        };
        let left = [("id".into(), metric.clone())].into_iter().collect();
        let mut changed = metric.clone();
        changed.score = Some(f64::from_bits(2.0_f64.to_bits() + 1));
        let identical = [("id".into(), metric)].into_iter().collect();
        let distinct = [("id".into(), changed)].into_iter().collect();
        assert!(exact_stream_duplicate(&left, &identical));
        assert!(!exact_stream_duplicate(&left, &distinct));
    }

    #[test]
    fn distinct_score_streams_can_corroborate_without_unique_peptides() {
        assert_eq!(
            classify_low_input_expert(None, false, 0, 12, 4.0),
            "supporting_corroborating_contributor"
        );
        assert_eq!(
            classify_low_input_expert(None, true, 0, 12, 4.0),
            "redundant"
        );
    }

    #[test]
    fn low_input_classification_is_invariant_to_expert_permutation() {
        let inputs = [
            ("novel", None, false, 2, 8, 3.0),
            ("supporting", None, false, 0, 8, 2.0),
            ("redundant", None, true, 0, 8, 2.0),
            ("invalid", Some("poor calibration"), false, 4, 8, 5.0),
        ];
        let classify = |order: &[usize]| {
            order
                .iter()
                .map(|index| {
                    let (name, invalid, duplicate, unique, corroborated, shapley) = inputs[*index];
                    (
                        name,
                        classify_low_input_expert(
                            invalid,
                            duplicate,
                            unique,
                            corroborated,
                            shapley,
                        ),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(classify(&[0, 1, 2, 3]), classify(&[3, 1, 0, 2]));
    }

    #[test]
    fn psm_disagreement_can_coexist_with_peptide_level_corroboration() {
        assert_eq!(classify_support_evidence(1, 1, 3, true), "disputed");
        assert_eq!(classify_support_evidence(2, 2, 3, true), "corroborated");
    }

    #[test]
    fn singleton_and_multi_peptide_protein_support_remain_distinct_counts() {
        let singleton_supporting_psms = 1_usize;
        let multi_peptide_supporting_psms = 3_usize;
        assert!(multi_peptide_supporting_psms > singleton_supporting_psms);
        assert_eq!(
            classify_support_evidence(1, 1, 3, false),
            "singly_supported_but_consistent"
        );
    }

    #[test]
    fn coalition_loss_is_preserved_as_informative_not_silently_dropped() {
        let before = ["psm-a".into(), "psm-b".into()]
            .into_iter()
            .collect::<BTreeSet<String>>();
        let after = ["psm-b".into(), "psm-c".into()]
            .into_iter()
            .collect::<BTreeSet<String>>();
        assert_eq!(after.difference(&before).count(), 1);
        assert_eq!(before.difference(&after).count(), 1);
    }

    #[test]
    fn minimum_expert_contract_counts_only_valid_nonredundant_information_sources() {
        let classifications = [
            classify_low_input_expert(None, false, 1, 4, 1.0),
            classify_low_input_expert(None, false, 0, 4, 1.0),
            classify_low_input_expert(None, true, 0, 4, 1.0),
            classify_low_input_expert(Some("fallback"), false, 3, 4, 1.0),
        ];
        let valid = classifications
            .iter()
            .filter(|classification| {
                **classification == "novel_contributor"
                    || **classification == "supporting_corroborating_contributor"
            })
            .count();
        assert_eq!(valid, 2);
    }
}
