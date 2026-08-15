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
pub const LOCK_SCHEMA: &str = "within-parent-holdout-lock-v1";
pub const PREREGISTRATION_SCHEMA: &str = "within-parent-holdout-preregistration-v1";
pub const PREFLIGHT_SCHEMA: &str = "within-parent-holdout-preflight-v1";
pub const RESULT_SCHEMA: &str = "within-parent-holdout-result-v1";

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
    pub comparison_additional_expert: ModelFit,
    pub target_only_policies: BTreeMap<String, String>,
    pub external_profile_window: NullWindow,
    pub effective_ratios: EffectiveRatios,
    pub optimizer_validation_scope: NullWindowValidationScope,
    pub optimizer_seed: u64,
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WithinParentHoldoutLock {
    pub schema: String,
    pub digest: String,
    pub manifest_sha256: String,
    pub parent_dataset_fingerprint: String,
    pub training_subset_digest: String,
    pub fold: usize,
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
        manifest.baseline_experts == vec![ModelFit::Moments, ModelFit::Mle, ModelFit::Msfdr1Smix],
        "baseline expert set must be Moments + MLE + MSFDR1-SMIX"
    );
    anyhow::ensure!(
        manifest.comparison_additional_expert == ModelFit::LowerOrder,
        "the only comparison expert may be Lower Order"
    );
    anyhow::ensure!(
        !manifest.model_grids.iter().any(|grid| matches!(
            grid.model,
            ModelFit::Msfdr | ModelFit::Msfdr2Smix | ModelFit::Nokoi
        )),
        "deferred experts are prohibited from this holdout"
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

    let expected_models = [
        ModelFit::Moments,
        ModelFit::Mle,
        ModelFit::LowerOrder,
        ModelFit::Msfdr1Smix,
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
pub struct FoldSummary {
    pub fold: usize,
    pub training_subset_digest: String,
    pub held_out_plus_entrapment_subset_digest: String,
    pub held_out_target_only_subset_digest: String,
    pub selected_windows: Vec<ModelWindowSelection>,
    pub baseline_lock: WithinParentHoldoutLock,
    pub comparison_lock: WithinParentHoldoutLock,
    pub plus_entrapment_baseline: StageSummary,
    pub plus_entrapment_comparison: StageSummary,
    pub plus_entrapment_incremental: Vec<IncrementalEvidence>,
    pub target_only_baseline: StageSummary,
    pub target_only_comparison: StageSummary,
    pub target_only_incremental: Vec<IncrementalEvidence>,
    pub technically_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AggregateSummary {
    pub baseline_plus_entrapment: Vec<LayerSummary>,
    pub comparison_plus_entrapment: Vec<LayerSummary>,
    pub plus_entrapment_incremental: Vec<IncrementalEvidence>,
    pub baseline_target_only: Vec<LayerSummary>,
    pub comparison_target_only: Vec<LayerSummary>,
    pub target_only_incremental: Vec<IncrementalEvidence>,
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
    pub classification: HoldoutClassification,
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
    include_lower_order: bool,
) -> Result<FdrSettings> {
    let mut settings = settings_for_model(base, ModelFit::Ensemble, None);
    settings.enable_moments = true;
    settings.enable_mle = true;
    settings.enable_msfdr_1smix = true;
    settings.enable_lower_order = include_lower_order;
    for model in [ModelFit::Moments, ModelFit::Mle, ModelFit::LowerOrder] {
        if model == ModelFit::LowerOrder && !include_lower_order {
            continue;
        }
        let window = windows
            .get(model_slug(&model))
            .with_context(|| format!("missing locked window for {}", model_slug(&model)))?;
        apply_model_window(&mut settings, &model, window);
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
    include_lower_order: bool,
) -> Result<WithinParentHoldoutLock> {
    let mut experts = artifacts
        .iter()
        .filter(|artifact| include_lower_order || artifact.model != ModelFit::LowerOrder)
        .map(|artifact| HoldoutLockExpert {
            model: artifact.model.clone(),
            selected_window: artifact.selected_window.clone(),
            training_artifact_digest: artifact.digest.clone(),
        })
        .collect::<Vec<_>>();
    experts.sort_by(|left, right| model_slug(&left.model).cmp(model_slug(&right.model)));
    let expected = if include_lower_order { 4 } else { 3 };
    anyhow::ensure!(
        experts.len() == expected,
        "holdout lock expert count mismatch"
    );
    let mut lock = WithinParentHoldoutLock {
        schema: LOCK_SCHEMA.into(),
        digest: String::new(),
        manifest_sha256: manifest_sha256.into(),
        parent_dataset_fingerprint: manifest.parent_dataset_fingerprint.clone(),
        training_subset_digest: training_subset.digest.clone(),
        fold,
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
    aggregate_incremental: &[IncrementalEvidence],
    target_incremental: &[IncrementalEvidence],
) -> HoldoutClassification {
    let all_valid = folds.iter().all(|fold| fold.technically_valid);
    let level4 = aggregate_incremental
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
            fold.plus_entrapment_incremental
                .iter()
                .find(|row| row.layer == "level4")
                .is_some_and(|row| {
                    row.added_target_peptides > 0
                        || row.added_target_peptidoforms > 0
                        || row.added_target_proteins > 0
                })
        })
        .count();
    let not_single_fold = contributing_folds >= 2;
    let calibration_ok = folds.iter().all(|fold| {
        ["raw_q", "level4"].into_iter().all(|layer| {
            let baseline = fold
                .plus_entrapment_baseline
                .layers
                .iter()
                .find(|row| row.layer == layer);
            let comparison = fold
                .plus_entrapment_comparison
                .layers
                .iter()
                .find(|row| row.layer == layer);
            match (baseline, comparison) {
                (Some(a), Some(b)) => {
                    match (a.peptide.ratio_adjusted_fdp, b.peptide.ratio_adjusted_fdp) {
                        (Some(x), Some(y)) => {
                            y <= x + manifest
                                .acceptance
                                .maximum_absolute_ratio_adjusted_peptide_fdp_increase
                        }
                        (None, None) => true,
                        _ => false,
                    }
                }
                _ => false,
            }
        })
    });
    let target_stable = target_incremental
        .iter()
        .find(|row| row.layer == "level4")
        .is_some_and(|row| {
            row.lost_target_peptides == 0
                && row.lost_target_peptidoforms == 0
                && row.lost_target_proteins == 0
        });
    let ready = all_valid && positive && not_single_fold && calibration_ok && target_stable;
    let mut reasons = Vec::new();
    if !all_valid {
        reasons.push("one or more folds failed technical validity".into());
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
        reasons.push("material fold-level peptide calibration deterioration was observed".into());
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
    let mut aggregate_plus_a_features = Vec::new();
    let mut aggregate_plus_b_features = Vec::new();
    let mut aggregate_target_a_features = Vec::new();
    let mut aggregate_target_b_features = Vec::new();

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
        for model in [
            ModelFit::Moments,
            ModelFit::Mle,
            ModelFit::Msfdr1Smix,
            ModelFit::LowerOrder,
        ] {
            let window = if model == ModelFit::Msfdr1Smix {
                None
            } else {
                windows.get(model_slug(&model)).cloned().flatten()
            };
            let settings =
                settings_for_model(&plus_context.search.fdr, model.clone(), window.clone());
            let stage = run_scored_stage(
                &training,
                &settings,
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
                model,
                window,
                &stage.artifacts,
            )?);
        }
        let baseline_lock = build_holdout_lock(
            &manifest,
            &manifest_sha256,
            fold.fold,
            &training.identity,
            &training_artifacts,
            false,
        )?;
        let comparison_lock = build_holdout_lock(
            &manifest,
            &manifest_sha256,
            fold.fold,
            &training.identity,
            &training_artifacts,
            true,
        )?;
        validate_holdout_lock(
            &baseline_lock,
            &manifest,
            &manifest_sha256,
            &training.identity,
        )?;
        validate_holdout_lock(
            &comparison_lock,
            &manifest,
            &manifest_sha256,
            &training.identity,
        )?;
        write_json_atomic(
            &fold_root.join("baseline.holdout.lock.json"),
            &baseline_lock,
        )?;
        write_json_atomic(
            &fold_root.join("comparison.holdout.lock.json"),
            &comparison_lock,
        )?;
        drop(training);

        let held_plus = derive_subset(
            &manifest,
            &plus_context,
            &manifest_sha256,
            fold.fold,
            SubsetRole::HeldOut,
            &fold.held_out_file_ids,
        )?;
        let settings_a = settings_for_ensemble(
            &plus_context.search.fdr,
            &windows_from_lock(&baseline_lock),
            false,
        )?;
        let settings_b = settings_for_ensemble(
            &plus_context.search.fdr,
            &windows_from_lock(&comparison_lock),
            true,
        )?;
        let plus_a = run_scored_stage(
            &held_plus,
            &settings_a,
            &plus_context.database,
            "baseline",
            manifest.effective_ratios,
            SearchSpace::PlusEntrapment,
        )?;
        let plus_b = run_scored_stage(
            &held_plus,
            &settings_b,
            &plus_context.database,
            "baseline_plus_lower_order",
            manifest.effective_ratios,
            SearchSpace::PlusEntrapment,
        )?;
        aggregate_plus_a_features.extend(plus_a.rank1_features.iter().cloned());
        aggregate_plus_b_features.extend(plus_b.rank1_features.iter().cloned());
        let plus_incremental = vec![
            incremental("raw_q", &plus_a.raw, &plus_b.raw),
            incremental("level4", &plus_a.level4, &plus_b.level4),
        ];
        let held_plus_digest = held_plus.identity.digest.clone();
        drop(held_plus);

        let held_target = derive_subset(
            &manifest,
            &target_context,
            &manifest_sha256,
            fold.fold,
            SubsetRole::HeldOut,
            &fold.held_out_file_ids,
        )?;
        let target_settings_a = settings_for_ensemble(
            &target_context.search.fdr,
            &windows_from_lock(&baseline_lock),
            false,
        )?;
        let target_settings_b = settings_for_ensemble(
            &target_context.search.fdr,
            &windows_from_lock(&comparison_lock),
            true,
        )?;
        anyhow::ensure!(
            target_settings_b.lower_order_frozen_artifact.is_none(),
            "Lower Order nuisance artifact leaked into target-only refit"
        );
        let target_a = run_scored_stage(
            &held_target,
            &target_settings_a,
            &target_context.database,
            "baseline",
            manifest.effective_ratios,
            SearchSpace::TargetOnly,
        )?;
        let target_b = run_scored_stage(
            &held_target,
            &target_settings_b,
            &target_context.database,
            "baseline_plus_lower_order",
            manifest.effective_ratios,
            SearchSpace::TargetOnly,
        )?;
        aggregate_target_a_features.extend(target_a.rank1_features.iter().cloned());
        aggregate_target_b_features.extend(target_b.rank1_features.iter().cloned());
        let target_incremental = vec![
            incremental("raw_q", &target_a.raw, &target_b.raw),
            incremental("level4", &target_a.level4, &target_b.level4),
        ];

        let technically_valid = !plus_a.summary.fallback_used
            && !plus_b.summary.fallback_used
            && !target_a.summary.fallback_used
            && !target_b.summary.fallback_used
            && plus_a.summary.unexplained_na_count == 0
            && plus_b.summary.unexplained_na_count == 0
            && target_a.summary.unexplained_na_count == 0
            && target_b.summary.unexplained_na_count == 0;
        let fold_summary = FoldSummary {
            fold: fold.fold,
            training_subset_digest: baseline_lock.training_subset_digest.clone(),
            held_out_plus_entrapment_subset_digest: held_plus_digest,
            held_out_target_only_subset_digest: held_target.identity.digest.clone(),
            selected_windows,
            baseline_lock,
            comparison_lock,
            plus_entrapment_baseline: plus_a.summary,
            plus_entrapment_comparison: plus_b.summary,
            plus_entrapment_incremental: plus_incremental,
            target_only_baseline: target_a.summary,
            target_only_comparison: target_b.summary,
            target_only_incremental: target_incremental,
            technically_valid,
        };
        write_json_atomic(&fold_root.join("fold_summary.json"), &fold_summary)?;
        fold_summaries.push(fold_summary);
    }

    let first_fold = fold_summaries
        .first()
        .context("holdout produced no folds")?;
    let aggregate_plus_settings_a = settings_for_ensemble(
        &plus_context.search.fdr,
        &windows_from_lock(&first_fold.baseline_lock),
        false,
    )?;
    let aggregate_plus_settings_b = settings_for_ensemble(
        &plus_context.search.fdr,
        &windows_from_lock(&first_fold.comparison_lock),
        true,
    )?;
    let aggregate_target_settings_a = settings_for_ensemble(
        &target_context.search.fdr,
        &windows_from_lock(&first_fold.baseline_lock),
        false,
    )?;
    let aggregate_target_settings_b = settings_for_ensemble(
        &target_context.search.fdr,
        &windows_from_lock(&first_fold.comparison_lock),
        true,
    )?;
    let (aggregate_plus_a_raw, aggregate_plus_a_l4) = recompute_out_of_fold_evidence(
        aggregate_plus_a_features,
        &plus_context.database,
        &aggregate_plus_settings_a,
        &manifest.plus_entrapment.search_fingerprint,
        manifest.effective_ratios,
        SearchSpace::PlusEntrapment,
    )?;
    let (aggregate_plus_b_raw, aggregate_plus_b_l4) = recompute_out_of_fold_evidence(
        aggregate_plus_b_features,
        &plus_context.database,
        &aggregate_plus_settings_b,
        &manifest.plus_entrapment.search_fingerprint,
        manifest.effective_ratios,
        SearchSpace::PlusEntrapment,
    )?;
    let (aggregate_target_a_raw, aggregate_target_a_l4) = recompute_out_of_fold_evidence(
        aggregate_target_a_features,
        &target_context.database,
        &aggregate_target_settings_a,
        &manifest.target_only.search_fingerprint,
        manifest.effective_ratios,
        SearchSpace::TargetOnly,
    )?;
    let (aggregate_target_b_raw, aggregate_target_b_l4) = recompute_out_of_fold_evidence(
        aggregate_target_b_features,
        &target_context.database,
        &aggregate_target_settings_b,
        &manifest.target_only.search_fingerprint,
        manifest.effective_ratios,
        SearchSpace::TargetOnly,
    )?;

    let plus_incremental = vec![
        incremental("raw_q", &aggregate_plus_a_raw, &aggregate_plus_b_raw),
        incremental("level4", &aggregate_plus_a_l4, &aggregate_plus_b_l4),
    ];
    let target_incremental = vec![
        incremental("raw_q", &aggregate_target_a_raw, &aggregate_target_b_raw),
        incremental("level4", &aggregate_target_a_l4, &aggregate_target_b_l4),
    ];
    let aggregate = AggregateSummary {
        baseline_plus_entrapment: aggregate_layers(&aggregate_plus_a_raw, &aggregate_plus_a_l4, manifest.effective_ratios, SearchSpace::PlusEntrapment),
        comparison_plus_entrapment: aggregate_layers(&aggregate_plus_b_raw, &aggregate_plus_b_l4, manifest.effective_ratios, SearchSpace::PlusEntrapment),
        plus_entrapment_incremental: plus_incremental.clone(),
        baseline_target_only: aggregate_layers(&aggregate_target_a_raw, &aggregate_target_a_l4, manifest.effective_ratios, SearchSpace::TargetOnly),
        comparison_target_only: aggregate_layers(&aggregate_target_b_raw, &aggregate_target_b_l4, manifest.effective_ratios, SearchSpace::TargetOnly),
        target_only_incremental: target_incremental.clone(),
        construction: "union of held-out rank-1 PSM identities exactly once, followed by aggregate peptide q-value calculation, protein inference/q-value calculation, hierarchical reporting, and canonical evidence-set counting; no fold-level peptide or protein counts are summed".into(),
        every_run_once: true,
    };
    let classification = classify_result(
        &manifest,
        &fold_summaries,
        &plus_incremental,
        &target_incremental,
    );
    let result = HoldoutResult {
        schema: RESULT_SCHEMA.into(),
        manifest_sha256,
        preflight_sha256,
        source_build: manifest.source_build,
        capabilities: CAPABILITIES,
        folds: fold_summaries,
        aggregate,
        classification,
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
    for name in ["moments", "mle", "lower_order"] {
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
            assignment_basis: "acquisition order only".into(),
            parent_dataset_id: "parent".into(),
            parent_dataset_fingerprint: "dataset".into(),
            spectra,
            plus_entrapment: parent(SearchSpace::PlusEntrapment, root),
            target_only: parent(SearchSpace::TargetOnly, root),
            folds,
            grid_source_description: "fixture".into(),
            grid_source_manifest_sha256: "grid-source".into(),
            model_grids: Vec::new(),
            baseline_experts: vec![ModelFit::Moments, ModelFit::Mle, ModelFit::Msfdr1Smix],
            comparison_additional_expert: ModelFit::LowerOrder,
            target_only_policies: [
                ("moments".into(), "refit_with_locked_window".into()),
                ("mle".into(), "refit_with_locked_window".into()),
                ("msfdr1_smix".into(), "refit_with_locked_window".into()),
                ("lower_order".into(), "refit_with_locked_window".into()),
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
            ("msfdr1_smix".into(), None),
        ]
        .into_iter()
        .collect();
        let settings = settings_for_ensemble(&base, &windows, true).unwrap();
        assert!(settings.null_window_optimizer.is_none());
        assert!(settings.lower_order_frozen_artifact.is_none());
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
        let artifacts = [ModelFit::Moments, ModelFit::Mle, ModelFit::Msfdr1Smix]
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
        let lock =
            build_holdout_lock(&manifest, "manifest", 1, &subset, &artifacts, false).unwrap();
        let mut reversed = artifacts.clone();
        reversed.reverse();
        let reversed_lock =
            build_holdout_lock(&manifest, "manifest", 1, &subset, &reversed, false).unwrap();
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
}
