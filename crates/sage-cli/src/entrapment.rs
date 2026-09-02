use crate::input::Input;
use crate::parameter_optimizer::{
    EntrapmentValidationConfig, EntrapmentValidationMode,
    PARAMETER_OPTIMIZER_IMPLEMENTATION_SOURCE_SHA256,
};
use crate::provenance::{sha256_file, write_json_atomic};
use anyhow::{Context, Result};
use sage_core::database::Parameters;
use sage_core::fasta::Fasta;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const ENTRAPMENT_PARTITION_ASSIGNMENT_ALGORITHM_SCHEMA: &str =
    "sage-entrapment-component-partition-assignment-v1";
pub const ENTRAPMENT_PARTITION_SCIENTIFIC_CONTENT_SCHEMA_VERSION: u32 = 1;
pub const ENTRAPMENT_PARTITION_VERIFICATION_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug)]
struct FastaRecord {
    header: String,
    accession: String,
    sequence: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EntrapmentRatios {
    pub target_proteins: usize,
    pub entrapment_proteins: usize,
    pub protein_ratio: f64,
    pub target_peptides: usize,
    pub entrapment_peptides: usize,
    pub peptide_ratio: f64,
    pub target_peptidoforms: usize,
    pub entrapment_peptidoforms: usize,
    pub peptidoform_ratio: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntrapmentPartitionAssignment {
    Selection,
    Audit,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntrapmentComponentAssignment {
    pub component_identity: String,
    pub partition: EntrapmentPartitionAssignment,
    pub proteins: Vec<String>,
    pub canonical_peptide_count: usize,
    pub peptidoform_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EntrapmentPartitionArtifact {
    pub schema_version: u32,
    pub partition_identity: String,
    pub dataset_identity: String,
    pub target_fasta_sha256: String,
    pub active_entrapment_fasta_sha256: String,
    pub digestion_search_space_identity: String,
    pub entrapment_construction_identity: String,
    pub seed: u64,
    pub salt: String,
    pub requested_selection_fraction: f64,
    pub requested_audit_fraction: f64,
    pub realized_selection_fraction: f64,
    pub realized_audit_fraction: f64,
    pub component_assignments: Vec<EntrapmentComponentAssignment>,
    pub selection_proteins: Vec<String>,
    pub audit_proteins: Vec<String>,
    pub selection_canonical_peptides: Vec<String>,
    pub audit_canonical_peptides: Vec<String>,
    pub selection_peptidoforms: Vec<String>,
    pub audit_peptidoforms: Vec<String>,
    pub selection_ratios: EntrapmentRatios,
    pub audit_ratios: EntrapmentRatios,
    pub source_implementation_identity: String,
    pub payload_sha256: String,
}

/// Complete portable scientific content of a partition. Historical generator
/// and current verifier implementation identities intentionally live outside
/// this projection.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EntrapmentPartitionScientificContentV1 {
    pub schema_version: u32,
    pub partition_schema_version: u32,
    pub assignment_algorithm_schema: String,
    /// Current portable content-derived dataset identity. The historical
    /// artifact's path-derived dataset and partition identities are retained
    /// in `HistoricalEntrapmentPartitionGeneratorIdentityV1` instead.
    pub dataset_identity: String,
    pub target_fasta_sha256: String,
    pub active_entrapment_fasta_sha256: String,
    pub digestion_search_space_identity: String,
    pub entrapment_construction_identity: String,
    pub seed: u64,
    pub salt: String,
    pub requested_selection_fraction: f64,
    pub requested_audit_fraction: f64,
    pub realized_selection_fraction: f64,
    pub realized_audit_fraction: f64,
    pub component_assignments: Vec<EntrapmentComponentAssignment>,
    pub selection_proteins: Vec<String>,
    pub audit_proteins: Vec<String>,
    pub selection_canonical_peptides: Vec<String>,
    pub audit_canonical_peptides: Vec<String>,
    pub selection_peptidoforms: Vec<String>,
    pub audit_peptidoforms: Vec<String>,
    pub selection_ratios: EntrapmentRatios,
    pub audit_ratios: EntrapmentRatios,
    pub component_overlap_count: usize,
    pub protein_overlap_count: usize,
    pub canonical_peptide_overlap_count: usize,
    pub peptidoform_overlap_count: usize,
    /// Connected components are the partition's protein-group units.
    pub protein_group_overlap_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HistoricalEntrapmentPartitionGeneratorIdentityV1 {
    pub schema_version: u32,
    pub generator_schema: String,
    pub assignment_algorithm_schema: String,
    pub source_implementation_identity: String,
    pub historical_dataset_identity_schema: String,
    pub historical_dataset_identity: String,
    pub original_partition_identity: String,
    pub original_artifact_sha256: String,
    pub original_payload_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CurrentEntrapmentPartitionVerifierIdentityV1 {
    pub schema_version: u32,
    pub verification_algorithm_schema: String,
    pub source_implementation_identity: String,
    pub executable_sha256: String,
    pub current_dataset_identity_schema: String,
    pub current_dataset_identity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct HistoricalDatasetIdentityAlias {
    pub schema: String,
    pub identity: String,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PartitionDatasetIdentityContext<'a> {
    pub current: &'a str,
    /// Compatibility aliases are derived by trusted workflow code from the
    /// active manifest; they are not accepted as user-supplied hashes.
    pub historical_aliases: &'a [HistoricalDatasetIdentityAlias],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntrapmentPartitionVerifiedUseV1 {
    pub schema_version: u32,
    pub scientific_content_sha256: String,
    pub exact_artifact_sha256: String,
    pub historical_generator: HistoricalEntrapmentPartitionGeneratorIdentityV1,
    pub current_verifier: CurrentEntrapmentPartitionVerifierIdentityV1,
    pub verified_use_sha256: String,
    /// Machine-local audit metadata; excluded from portable identities.
    pub artifact_path: PathBuf,
    /// Machine-local audit metadata; excluded from portable identities.
    pub verified_at_unix_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VerifiedEntrapmentPartition {
    pub artifact: EntrapmentPartitionArtifact,
    pub verification: EntrapmentPartitionVerifiedUseV1,
}

/// The only partition data available to model fitting and optimizer trial
/// evaluation. Audit identities and ratios deliberately have no field here.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EntrapmentSelectionView {
    pub partition_identity: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scientific_content_sha256: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub exact_artifact_sha256: String,
    pub selection_proteins: Vec<String>,
    pub selection_ratios: EntrapmentRatios,
}

impl EntrapmentPartitionArtifact {
    pub fn selection_protein_set(&self) -> BTreeSet<String> {
        self.selection_proteins.iter().cloned().collect()
    }

    pub fn audit_protein_set(&self) -> BTreeSet<String> {
        self.audit_proteins.iter().cloned().collect()
    }

    pub fn selection_view(&self) -> EntrapmentSelectionView {
        EntrapmentSelectionView {
            partition_identity: self.partition_identity.clone(),
            scientific_content_sha256: String::new(),
            exact_artifact_sha256: String::new(),
            selection_proteins: self.selection_proteins.clone(),
            selection_ratios: self.selection_ratios.clone(),
        }
    }

    pub fn selection_view_verified(
        &self,
        verification: &EntrapmentPartitionVerifiedUseV1,
    ) -> EntrapmentSelectionView {
        EntrapmentSelectionView {
            partition_identity: self.partition_identity.clone(),
            scientific_content_sha256: verification.scientific_content_sha256.clone(),
            exact_artifact_sha256: verification.exact_artifact_sha256.clone(),
            selection_proteins: self.selection_proteins.clone(),
            selection_ratios: self.selection_ratios.clone(),
        }
    }
}

impl EntrapmentSelectionView {
    pub fn selection_protein_set(&self) -> BTreeSet<String> {
        self.selection_proteins.iter().cloned().collect()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForeignSourceMode {
    /// Evaluate every candidate source and use the one whose measured peptide
    /// and peptidoform ratios are jointly closest to the protein fold.
    #[default]
    Automatic,
    /// Use the declared source without treating the other candidates as part
    /// of the selection experiment.
    Explicit,
    /// Evaluate and report the automatic recommendation, but use the declared
    /// override source for generation.
    AutomaticWithOverride,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SharedPeptideExclusionMode {
    /// Exclude overlaps only when the peptide exists in Sage's configured
    /// searchable mass/modification space.
    #[default]
    SageSearchSpace,
    /// Reproduce FDRBench 0.0.4's length-only digest before its seeded Java
    /// HashMap/Collections.shuffle selection. Ratios are still measured by Sage.
    Fdrbench004Compatibility,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForeignCandidateReport {
    pub fasta: PathBuf,
    pub sha256: String,
    pub total_proteins: usize,
    pub eligible_proteins: usize,
    pub excluded_shared_target_peptide: usize,
    pub selected_proteins: usize,
    pub measured: EntrapmentRatios,
    pub selection_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntrapmentGenerationReport {
    pub schema_version: u32,
    /// Legacy phase-local whole-`Parameters` digest. Retained for historical
    /// provenance only; cross-phase reuse must use `scientific_input_sha256`.
    pub generation_input_sha256: String,
    #[serde(default)]
    pub scientific_inputs: Option<EntrapmentGenerationScientificInputsV1>,
    #[serde(default)]
    pub scientific_input_sha256: Option<String>,
    #[serde(default = "default_generator_version")]
    pub generator_version: String,
    #[serde(default = "default_selection_algorithm")]
    pub selection_algorithm: String,
    pub target_fasta: PathBuf,
    pub target_sha256: String,
    pub selected_foreign_fasta: PathBuf,
    pub selected_foreign_sha256: String,
    pub output_fasta: PathBuf,
    pub output_sha256: String,
    pub seed: u64,
    pub protein_fold: usize,
    #[serde(default)]
    pub shared_peptide_exclusion_mode: SharedPeptideExclusionMode,
    #[serde(default)]
    pub foreign_source_mode: ForeignSourceMode,
    #[serde(default)]
    pub automatically_recommended_foreign_fasta: Option<PathBuf>,
    #[serde(default)]
    pub automatically_recommended_foreign_sha256: Option<String>,
    #[serde(default)]
    pub override_applied: bool,
    pub candidates: Vec<ForeignCandidateReport>,
    pub selected_accessions: Vec<String>,
    pub excluded_shared_target_peptide: Vec<String>,
    pub source_accession_mapping: BTreeMap<String, String>,
    pub measured: EntrapmentRatios,
    pub target_header_order_sha256: String,
    pub entrapment_header_order_sha256: String,
    pub selected_accession_order_sha256: String,
    pub excluded_accessions_sha256: String,
    pub source_accession_mapping_sha256: String,
    pub deterministic_selection_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenEntrapmentReport {
    pub schema_version: u32,
    pub target_fasta: PathBuf,
    pub target_sha256: String,
    pub frozen_entrapment_fasta: PathBuf,
    pub frozen_entrapment_sha256: String,
    pub target_headers: usize,
    pub entrapment_headers: usize,
    pub target_header_order_sha256: String,
    pub entrapment_header_order_sha256: String,
    pub measured: EntrapmentRatios,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LegacyEntrapmentReference {
    pub fasta: PathBuf,
    #[serde(default)]
    pub foreign_fasta: Option<PathBuf>,
    #[serde(default)]
    pub generation_log: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CountRatioComparison {
    pub native: EntrapmentRatios,
    pub legacy: EntrapmentRatios,
    pub protein_ratio_delta: f64,
    pub peptide_ratio_delta: f64,
    pub peptidoform_ratio_delta: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntrapmentFastaParityReport {
    pub schema_version: u32,
    pub native_fasta: PathBuf,
    pub native_sha256: String,
    pub legacy_fasta: PathBuf,
    pub legacy_sha256: String,
    pub exact_fasta_match: bool,
    pub native_selected_foreign_fasta: PathBuf,
    pub legacy_selected_foreign_fasta: Option<PathBuf>,
    pub selected_foreign_source_match: Option<bool>,
    pub selected_accession_set_match: bool,
    pub selected_accession_order_match: bool,
    pub native_only_selected_accessions: Vec<String>,
    pub legacy_only_selected_accessions: Vec<String>,
    pub excluded_shared_peptide_set_match: Option<bool>,
    pub native_only_excluded_accessions: Vec<String>,
    pub legacy_only_excluded_accessions: Vec<String>,
    pub target_header_order_match: bool,
    pub entrapment_header_order_match: bool,
    pub exact_header_order_match: bool,
    pub native_mapping_sha256: String,
    pub legacy_mapping_sha256: String,
    pub mapping_match: bool,
    pub seed: u64,
    pub deterministic_selection_sha256: String,
    pub ratios: CountRatioComparison,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntrapmentDatabaseMode {
    #[default]
    NativeGenerated,
    FrozenLegacy,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EntrapmentGenerationMode {
    #[default]
    WorkflowLocal,
    RequireExisting,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExistingEntrapmentResourceReference {
    pub schema_version: u32,
    pub artifact_sha256: String,
    pub combined_fasta_sha256: String,
    pub construction_identity: String,
    pub scientific_input_sha256: String,
    pub resource_identity: String,
    pub legacy_generation_input_sha256: String,
    pub reused: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalModificationV1 {
    pub specificity: String,
    pub mass_bits: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalVariableModificationV1 {
    pub specificity: String,
    pub mass_bits: Vec<u32>,
}

/// Canonical, resolved inputs that can change native entrapment construction.
/// Paths, search scoring, reporting, external features, and runtime controls
/// are deliberately absent.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntrapmentGenerationScientificInputsV1 {
    pub schema_version: u32,
    pub generator_version: String,
    pub selection_algorithm: String,
    pub canonical_peptide_semantics: String,
    pub canonical_peptidoform_semantics: String,
    pub header_generation_semantics: String,
    pub target_fasta_sha256: String,
    pub foreign_fasta_sha256: Vec<String>,
    pub selected_foreign_fasta_sha256: Option<String>,
    pub foreign_source_mode: ForeignSourceMode,
    pub shared_peptide_exclusion_mode: SharedPeptideExclusionMode,
    pub seed: u64,
    pub protein_fold: usize,
    pub missed_cleavages: u8,
    pub peptide_min_length: usize,
    pub peptide_max_length: usize,
    pub cleave_at: String,
    pub restrict: String,
    pub c_terminal: bool,
    pub semi_enzymatic: bool,
    pub peptide_min_mass_bits: u32,
    pub peptide_max_mass_bits: u32,
    pub static_modifications: Vec<CanonicalModificationV1>,
    pub variable_modifications: Vec<CanonicalVariableModificationV1>,
    pub maximum_variable_modifications: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExistingEntrapmentResourceLock {
    pub schema_version: u32,
    pub scientific_inputs: EntrapmentGenerationScientificInputsV1,
    pub scientific_input_sha256: String,
    pub generation_implementation_identity: String,
    pub target_fasta_sha256: String,
    pub foreign_fasta_sha256: Vec<String>,
    pub selected_foreign_fasta_sha256: String,
    pub generated_combined_fasta_sha256: String,
    pub legacy_audit_artifact_sha256: String,
    pub legacy_audit_schema_version: u32,
    pub legacy_generation_schema_version: u32,
    pub legacy_generation_input_sha256: String,
    pub historical_search_config_sha256: String,
    pub database_report: EntrapmentDatabaseReport,
    pub construction_identity: String,
    pub resource_identity: String,
    /// Operational evidence only. Excluded from portable scientific identity.
    pub audit_artifact_path: PathBuf,
    /// Operational evidence only. Excluded from portable scientific identity.
    pub historical_search_config_path: PathBuf,
    pub payload_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "database_mode", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum EntrapmentDatabaseReport {
    NativeGenerated {
        generation: EntrapmentGenerationReport,
    },
    FrozenLegacy {
        frozen: FrozenEntrapmentReport,
    },
}

impl EntrapmentDatabaseReport {
    pub fn measured(&self) -> &EntrapmentRatios {
        match self {
            Self::NativeGenerated { generation } => &generation.measured,
            Self::FrozenLegacy { frozen } => &frozen.measured,
        }
    }
}

fn validate_measured_ratios(ratios: &EntrapmentRatios) -> Result<()> {
    anyhow::ensure!(
        ratios.target_proteins > 0
            && ratios.entrapment_proteins > 0
            && ratios.target_peptides > 0
            && ratios.entrapment_peptides > 0
            && ratios.target_peptidoforms > 0
            && ratios.entrapment_peptidoforms > 0,
        "existing entrapment resource contains an empty measured population"
    );
    let expected = [
        ratios.entrapment_proteins as f64 / ratios.target_proteins as f64,
        ratios.entrapment_peptides as f64 / ratios.target_peptides as f64,
        ratios.entrapment_peptidoforms as f64 / ratios.target_peptidoforms as f64,
    ];
    let recorded = [
        ratios.protein_ratio,
        ratios.peptide_ratio,
        ratios.peptidoform_ratio,
    ];
    anyhow::ensure!(
        recorded
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            && recorded
                .iter()
                .zip(expected)
                .all(|(recorded, expected)| (*recorded - expected).abs() <= 1e-12),
        "existing entrapment resource contains inconsistent measured ratios"
    );
    Ok(())
}

fn sorted_static_modifications(parameters: &Parameters) -> Vec<CanonicalModificationV1> {
    let mut modifications = parameters
        .static_mods
        .iter()
        .map(|(specificity, mass)| CanonicalModificationV1 {
            specificity: specificity.to_string(),
            mass_bits: mass.to_bits(),
        })
        .collect::<Vec<_>>();
    modifications.sort_by(|left, right| {
        left.specificity
            .cmp(&right.specificity)
            .then_with(|| left.mass_bits.cmp(&right.mass_bits))
    });
    modifications
}

fn sorted_variable_modifications(parameters: &Parameters) -> Vec<CanonicalVariableModificationV1> {
    let mut modifications = parameters
        .variable_mods
        .iter()
        .map(|(specificity, masses)| {
            let mut mass_bits = masses.iter().map(|mass| mass.to_bits()).collect::<Vec<_>>();
            mass_bits.sort_unstable();
            CanonicalVariableModificationV1 {
                specificity: specificity.to_string(),
                mass_bits,
            }
        })
        .collect::<Vec<_>>();
    modifications.sort_by(|left, right| {
        left.specificity
            .cmp(&right.specificity)
            .then_with(|| left.mass_bits.cmp(&right.mass_bits))
    });
    modifications
}

#[allow(clippy::too_many_arguments)]
pub fn entrapment_generation_scientific_inputs(
    parameters: &Parameters,
    target_fasta: &Path,
    foreign_fastas: &[PathBuf],
    seed: u64,
    protein_fold: usize,
    source_mode: &ForeignSourceMode,
    exclusion_mode: &SharedPeptideExclusionMode,
    selected_foreign_fasta: Option<&Path>,
) -> Result<EntrapmentGenerationScientificInputsV1> {
    anyhow::ensure!(protein_fold > 0, "protein_fold must be positive");
    anyhow::ensure!(target_fasta.is_file(), "target FASTA does not exist");
    anyhow::ensure!(
        !foreign_fastas.is_empty() && foreign_fastas.iter().all(|path| path.is_file()),
        "all foreign FASTA inputs must exist"
    );
    match source_mode {
        ForeignSourceMode::Automatic => anyhow::ensure!(
            selected_foreign_fasta.is_none(),
            "automatic foreign-source selection must not declare a selected source"
        ),
        ForeignSourceMode::Explicit | ForeignSourceMode::AutomaticWithOverride => {
            anyhow::ensure!(
                selected_foreign_fasta.is_some_and(Path::is_file),
                "explicit or override source selection requires an existing selected source"
            );
        }
    }
    let mut foreign_fasta_sha256 = foreign_fastas
        .iter()
        .map(|path| sha256_file(path))
        .collect::<Result<Vec<_>>>()?;
    foreign_fasta_sha256.sort();
    foreign_fasta_sha256.dedup();
    anyhow::ensure!(
        foreign_fasta_sha256.len() == foreign_fastas.len(),
        "duplicate foreign FASTA content identities are not allowed"
    );
    let selected_foreign_fasta_sha256 = selected_foreign_fasta.map(sha256_file).transpose()?;
    if let Some(selected) = &selected_foreign_fasta_sha256 {
        anyhow::ensure!(
            foreign_fasta_sha256.binary_search(selected).is_ok(),
            "selected foreign source is not one of the declared foreign inputs"
        );
    }
    let enzyme = &parameters.enzyme;
    Ok(EntrapmentGenerationScientificInputsV1 {
        schema_version: 1,
        generator_version: default_generator_version(),
        selection_algorithm: default_selection_algorithm(),
        canonical_peptide_semantics: "sage-entrapment-canonical-peptide-il-v1".into(),
        canonical_peptidoform_semantics: "sage-entrapment-peptidoform-mass-bits-v1".into(),
        header_generation_semantics: "sage-entrapment-header-ent-serial-source-v1".into(),
        target_fasta_sha256: sha256_file(target_fasta)?,
        foreign_fasta_sha256,
        selected_foreign_fasta_sha256,
        foreign_source_mode: source_mode.clone(),
        shared_peptide_exclusion_mode: exclusion_mode.clone(),
        seed,
        protein_fold,
        missed_cleavages: enzyme.missed_cleavages.unwrap_or(1),
        peptide_min_length: enzyme.min_len.unwrap_or(5),
        peptide_max_length: enzyme.max_len.unwrap_or(50),
        cleave_at: enzyme.cleave_at.clone().unwrap_or_else(|| "KR".into()),
        restrict: enzyme.restrict.clone().unwrap_or_default(),
        c_terminal: enzyme.c_terminal.unwrap_or(true),
        semi_enzymatic: enzyme.semi_enzymatic.unwrap_or(false),
        peptide_min_mass_bits: parameters.peptide_min_mass.to_bits(),
        peptide_max_mass_bits: parameters.peptide_max_mass.to_bits(),
        static_modifications: sorted_static_modifications(parameters),
        variable_modifications: sorted_variable_modifications(parameters),
        maximum_variable_modifications: parameters.max_variable_mods,
    })
}

pub fn entrapment_generation_scientific_input_sha256(
    inputs: &EntrapmentGenerationScientificInputsV1,
) -> Result<String> {
    anyhow::ensure!(
        inputs.schema_version == 1,
        "unsupported scientific-input schema"
    );
    let mut hasher = Sha256::new();
    hasher.update(b"sage-entrapment-generation-scientific-inputs-v1\0");
    hasher.update(serde_json::to_vec(inputs)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn resource_lock_payload_sha256(lock: &ExistingEntrapmentResourceLock) -> Result<String> {
    let mut payload = lock.clone();
    payload.payload_sha256.clear();
    let mut hasher = Sha256::new();
    hasher.update(b"sage-existing-entrapment-resource-lock-payload-v1\0");
    hasher.update(serde_json::to_vec(&payload)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn entrapment_resource_identity(
    scientific_input_sha256: &str,
    generated_combined_fasta_sha256: &str,
    construction_identity: &str,
    measured: &EntrapmentRatios,
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-existing-entrapment-scientific-resource-v1\0");
    hasher.update(scientific_input_sha256.as_bytes());
    hasher.update(generated_combined_fasta_sha256.as_bytes());
    hasher.update(construction_identity.as_bytes());
    hasher.update(serde_json::to_vec(measured)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_existing_entrapment_resource_lock(lock: &ExistingEntrapmentResourceLock) -> Result<()> {
    anyhow::ensure!(
        lock.schema_version == 1,
        "unsupported existing entrapment resource-lock schema"
    );
    anyhow::ensure!(
        resource_lock_payload_sha256(lock)? == lock.payload_sha256,
        "existing entrapment resource-lock payload hash mismatch"
    );
    anyhow::ensure!(
        entrapment_generation_scientific_input_sha256(&lock.scientific_inputs)?
            == lock.scientific_input_sha256,
        "existing entrapment resource-lock scientific-input hash mismatch"
    );
    anyhow::ensure!(
        lock.generation_implementation_identity == generation_implementation_identity()
            && lock.scientific_inputs.generator_version == default_generator_version()
            && lock.scientific_inputs.selection_algorithm == default_selection_algorithm(),
        "unsupported existing entrapment resource-lock construction implementation"
    );
    anyhow::ensure!(
        lock.target_fasta_sha256 == lock.scientific_inputs.target_fasta_sha256
            && lock.foreign_fasta_sha256 == lock.scientific_inputs.foreign_fasta_sha256,
        "existing entrapment resource-lock input identities are inconsistent"
    );
    let generation = match &lock.database_report {
        EntrapmentDatabaseReport::NativeGenerated { generation } => generation,
        EntrapmentDatabaseReport::FrozenLegacy { .. } => {
            anyhow::bail!("existing entrapment resource lock is not native-generated")
        }
    };
    anyhow::ensure!(
        generation.target_sha256 == lock.target_fasta_sha256
            && generation.selected_foreign_sha256 == lock.selected_foreign_fasta_sha256
            && generation.output_sha256 == lock.generated_combined_fasta_sha256
            && generation.generation_input_sha256 == lock.legacy_generation_input_sha256,
        "existing entrapment resource-lock generation identities are inconsistent"
    );
    validate_measured_ratios(&generation.measured)?;
    anyhow::ensure!(
        entrapment_construction_identity(&lock.database_report)? == lock.construction_identity,
        "existing entrapment resource-lock construction identity mismatch"
    );
    anyhow::ensure!(
        entrapment_resource_identity(
            &lock.scientific_input_sha256,
            &lock.generated_combined_fasta_sha256,
            &lock.construction_identity,
            &generation.measured,
        )? == lock.resource_identity,
        "existing entrapment resource-lock scientific resource identity mismatch"
    );
    Ok(())
}

fn json_difference_paths(
    recorded: &serde_json::Value,
    expected: &serde_json::Value,
    path: &str,
    differences: &mut Vec<String>,
) {
    match (recorded, expected) {
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            let keys = left
                .keys()
                .chain(right.keys())
                .cloned()
                .collect::<BTreeSet<_>>();
            for key in keys {
                let child = format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
                match (left.get(&key), right.get(&key)) {
                    (Some(left), Some(right)) => {
                        json_difference_paths(left, right, &child, differences)
                    }
                    _ => differences.push(child),
                }
            }
        }
        (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
            if left.len() != right.len() {
                differences.push(format!("{path}/length"));
            }
            for (index, (left, right)) in left.iter().zip(right).enumerate() {
                json_difference_paths(left, right, &format!("{path}/{index}"), differences);
            }
        }
        _ if recorded != expected => differences.push(path.to_owned()),
        _ => {}
    }
}

fn scientific_input_difference_paths(
    recorded: &EntrapmentGenerationScientificInputsV1,
    expected: &EntrapmentGenerationScientificInputsV1,
) -> Result<Vec<String>> {
    let mut differences = Vec::new();
    json_difference_paths(
        &serde_json::to_value(recorded)?,
        &serde_json::to_value(expected)?,
        "",
        &mut differences,
    );
    Ok(differences)
}

#[allow(clippy::too_many_arguments)]
pub fn load_existing_entrapment_resource(
    artifact_path: &Path,
    expected_artifact_sha256: &str,
    expected_combined_fasta_sha256: &str,
    parameters: &Parameters,
    target_fasta: &Path,
    foreign_fastas: &[PathBuf],
    active_entrapment_fasta: &Path,
    seed: u64,
    protein_fold: usize,
    source_mode: &ForeignSourceMode,
    exclusion_mode: &SharedPeptideExclusionMode,
    selected_foreign_fasta: Option<&Path>,
) -> Result<(
    EntrapmentDatabaseReport,
    ExistingEntrapmentResourceReference,
)> {
    anyhow::ensure!(
        artifact_path.is_file(),
        "required existing entrapment artifact does not exist"
    );
    let artifact_sha256 = sha256_file(artifact_path)?;
    anyhow::ensure!(
        artifact_sha256 == expected_artifact_sha256,
        "existing entrapment artifact content hash mismatch"
    );
    let lock: ExistingEntrapmentResourceLock =
        serde_json::from_slice(&std::fs::read(artifact_path)?)
            .context("invalid existing Sage entrapment resource lock")?;
    validate_existing_entrapment_resource_lock(&lock)?;
    let generation = match &lock.database_report {
        EntrapmentDatabaseReport::NativeGenerated { generation } => generation,
        EntrapmentDatabaseReport::FrozenLegacy { .. } => unreachable!(),
    };
    anyhow::ensure!(
        (generation.schema_version == 2 || generation.schema_version == 3)
            && generation.generator_version == default_generator_version()
            && generation.selection_algorithm == default_selection_algorithm(),
        "unsupported existing entrapment generation schema or implementation"
    );
    let target_sha256 = sha256_file(target_fasta)?;
    let combined_sha256 = sha256_file(active_entrapment_fasta)?;
    anyhow::ensure!(
        target_sha256 == lock.target_fasta_sha256,
        "existing entrapment target FASTA mismatch"
    );
    anyhow::ensure!(
        combined_sha256 == expected_combined_fasta_sha256
            && lock.generated_combined_fasta_sha256 == expected_combined_fasta_sha256,
        "existing entrapment combined FASTA mismatch"
    );
    let active_database = Path::new(&parameters.fasta);
    anyhow::ensure!(
        active_database.is_file()
            && sha256_file(active_database)? == lock.generated_combined_fasta_sha256,
        "active optimization database is not the generated combined entrapment FASTA"
    );
    let expected_scientific_inputs = entrapment_generation_scientific_inputs(
        parameters,
        target_fasta,
        foreign_fastas,
        seed,
        protein_fold,
        source_mode,
        exclusion_mode,
        selected_foreign_fasta,
    )?;
    let expected_scientific_input_sha256 =
        entrapment_generation_scientific_input_sha256(&expected_scientific_inputs)?;
    let scientific_differences =
        scientific_input_difference_paths(&lock.scientific_inputs, &expected_scientific_inputs)?;
    anyhow::ensure!(
        scientific_differences.is_empty()
            && lock.scientific_input_sha256 == expected_scientific_input_sha256,
        "existing entrapment phase-scoped scientific-input mismatch; differing components: {}",
        scientific_differences.join(", ")
    );
    if let Some(selected) = selected_foreign_fasta {
        anyhow::ensure!(
            sha256_file(selected)? == generation.selected_foreign_sha256,
            "existing entrapment selected foreign-source mismatch"
        );
    }
    validate_measured_ratios(&generation.measured)?;
    let report = lock.database_report.clone();
    let reference = ExistingEntrapmentResourceReference {
        schema_version: lock.schema_version,
        artifact_sha256,
        combined_fasta_sha256: combined_sha256,
        construction_identity: lock.construction_identity,
        scientific_input_sha256: lock.scientific_input_sha256,
        resource_identity: lock.resource_identity,
        legacy_generation_input_sha256: lock.legacy_generation_input_sha256,
        reused: true,
    };
    Ok((report, reference))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntrapmentAuditManifest {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub search_config: PathBuf,
    pub target_fasta: PathBuf,
    pub output_directory: PathBuf,
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
pub struct EntrapmentAuditReport {
    pub schema_version: u32,
    pub database: EntrapmentDatabaseReport,
    #[serde(default)]
    pub fasta_parity: Option<EntrapmentFastaParityReport>,
}

fn default_schema_version() -> u32 {
    1
}

fn default_selection_algorithm() -> String {
    "fdrbench_0_0_4_java_hashmap_collections_shuffle".into()
}

fn default_generator_version() -> String {
    "sage-foreign-entrapment-generation-v3".into()
}

fn default_protein_fold() -> usize {
    1
}

pub fn execute_entrapment_audit(manifest_path: &Path) -> Result<EntrapmentAuditReport> {
    let manifest: EntrapmentAuditManifest = serde_json::from_slice(
        &std::fs::read(manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?,
    )
    .with_context(|| {
        format!(
            "invalid entrapment audit manifest {}",
            manifest_path.display()
        )
    })?;
    anyhow::ensure!(manifest.schema_version == 1, "unsupported audit schema");
    anyhow::ensure!(
        manifest.search_config.is_file(),
        "search_config does not exist"
    );
    anyhow::ensure!(
        manifest.target_fasta.is_file(),
        "target_fasta does not exist"
    );
    anyhow::ensure!(manifest.protein_fold > 0, "protein_fold must be positive");
    std::fs::create_dir_all(&manifest.output_directory)?;
    let input = Input::load(manifest.search_config.to_string_lossy().as_ref())?;
    let parameters = input.database.make_parameters();

    let (database, fasta_parity) = match manifest.database_mode {
        EntrapmentDatabaseMode::NativeGenerated => {
            let generation = generate_foreign_entrapment(
                &parameters,
                &manifest.target_fasta,
                &manifest.foreign_fastas,
                &manifest.output_fasta,
                manifest.seed,
                manifest.protein_fold,
                manifest.foreign_source_mode,
                manifest.shared_peptide_exclusion_mode,
                manifest.selected_foreign_fasta.as_deref(),
            )?;
            write_json_atomic(
                &manifest.output_directory.join("entrapment.generation.json"),
                &generation,
            )?;
            let fasta_parity = manifest
                .legacy_parity_reference
                .as_ref()
                .map(|reference| compare_generated_to_legacy(&parameters, &generation, reference))
                .transpose()?;
            if let Some(parity) = fasta_parity.as_ref() {
                write_json_atomic(
                    &manifest
                        .output_directory
                        .join("entrapment.fasta_parity.json"),
                    parity,
                )?;
            }
            (
                EntrapmentDatabaseReport::NativeGenerated { generation },
                fasta_parity,
            )
        }
        EntrapmentDatabaseMode::FrozenLegacy => {
            anyhow::ensure!(
                manifest.legacy_parity_reference.is_none(),
                "frozen optimizer-input audit and FASTA-generation parity must be separate"
            );
            let frozen_path = manifest
                .frozen_legacy_fasta
                .as_ref()
                .context("frozen_legacy mode requires frozen_legacy_fasta")?;
            let frozen =
                inspect_frozen_entrapment(&parameters, &manifest.target_fasta, frozen_path)?;
            write_json_atomic(
                &manifest.output_directory.join("entrapment.frozen.json"),
                &frozen,
            )?;
            (EntrapmentDatabaseReport::FrozenLegacy { frozen }, None)
        }
    };
    let report = EntrapmentAuditReport {
        schema_version: 1,
        database,
        fasta_parity,
    };
    write_json_atomic(
        &manifest.output_directory.join("entrapment.audit.json"),
        &report,
    )?;
    Ok(report)
}

fn generation_implementation_identity() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-entrapment-generation-implementation-v1\0");
    hasher.update(default_generator_version().as_bytes());
    hasher.update([0]);
    hasher.update(default_selection_algorithm().as_bytes());
    hasher.update([0]);
    hasher.update(b"scientific-input-schema-v1");
    format!("{:x}", hasher.finalize())
}

/// Derive an immutable, phase-scoped resource lock from a historical Sage
/// audit. This verifies but never regenerates the combined FASTA.
pub fn lock_existing_entrapment_resource(
    audit_manifest_path: &Path,
    audit_report_path: &Path,
    output: &Path,
) -> Result<ExistingEntrapmentResourceLock> {
    anyhow::ensure!(
        !output.exists(),
        "existing entrapment resource-lock output already exists"
    );
    let manifest: EntrapmentAuditManifest = serde_json::from_slice(
        &std::fs::read(audit_manifest_path)
            .with_context(|| format!("failed to read {}", audit_manifest_path.display()))?,
    )
    .context("invalid historical entrapment audit manifest")?;
    anyhow::ensure!(
        manifest.schema_version == 1
            && manifest.database_mode == EntrapmentDatabaseMode::NativeGenerated,
        "resource locking requires a supported native-generated audit manifest"
    );
    anyhow::ensure!(
        manifest.search_config.is_file()
            && manifest.target_fasta.is_file()
            && manifest.output_fasta.is_file()
            && manifest.foreign_fastas.iter().all(|path| path.is_file()),
        "historical entrapment manifest references a missing input or output"
    );
    let audit_bytes = std::fs::read(audit_report_path)
        .with_context(|| format!("failed to read {}", audit_report_path.display()))?;
    let audit: EntrapmentAuditReport = serde_json::from_slice(&audit_bytes)
        .context("invalid historical entrapment audit report")?;
    anyhow::ensure!(
        audit.schema_version == 1,
        "unsupported historical audit schema"
    );
    let generation = match &audit.database {
        EntrapmentDatabaseReport::NativeGenerated { generation } => generation,
        EntrapmentDatabaseReport::FrozenLegacy { .. } => {
            anyhow::bail!("historical audit is not Sage native-generated")
        }
    };
    anyhow::ensure!(
        generation.schema_version == 2 || generation.schema_version == 3,
        "unsupported historical entrapment generation schema"
    );
    anyhow::ensure!(
        generation.generator_version == default_generator_version()
            && generation.selection_algorithm == default_selection_algorithm(),
        "unsupported historical entrapment generation implementation"
    );
    let input = Input::load(manifest.search_config.to_string_lossy().as_ref())?;
    let parameters = input.database.make_parameters();
    let legacy_input_sha256 = entrapment_generation_input_sha256(
        &parameters,
        &manifest.target_fasta,
        &manifest.foreign_fastas,
        manifest.seed,
        manifest.protein_fold,
        &manifest.foreign_source_mode,
        &manifest.shared_peptide_exclusion_mode,
        manifest.selected_foreign_fasta.as_deref(),
    )?;
    anyhow::ensure!(
        generation.generation_input_sha256 == legacy_input_sha256,
        "historical audit's legacy full-input hash does not match its frozen manifest and search configuration"
    );
    let scientific_inputs = entrapment_generation_scientific_inputs(
        &parameters,
        &manifest.target_fasta,
        &manifest.foreign_fastas,
        manifest.seed,
        manifest.protein_fold,
        &manifest.foreign_source_mode,
        &manifest.shared_peptide_exclusion_mode,
        manifest.selected_foreign_fasta.as_deref(),
    )?;
    let scientific_input_sha256 =
        entrapment_generation_scientific_input_sha256(&scientific_inputs)?;
    let target_fasta_sha256 = sha256_file(&manifest.target_fasta)?;
    let generated_combined_fasta_sha256 = sha256_file(&manifest.output_fasta)?;
    anyhow::ensure!(
        generation.target_sha256 == target_fasta_sha256
            && generation.output_sha256 == generated_combined_fasta_sha256
            && generation.seed == manifest.seed
            && generation.protein_fold == manifest.protein_fold
            && generation.foreign_source_mode == manifest.foreign_source_mode
            && generation.shared_peptide_exclusion_mode == manifest.shared_peptide_exclusion_mode,
        "historical audit generation fields do not match its frozen manifest or FASTA content"
    );
    let selected_path = manifest
        .selected_foreign_fasta
        .as_deref()
        .unwrap_or(&generation.selected_foreign_fasta);
    let selected_foreign_fasta_sha256 = sha256_file(selected_path)?;
    anyhow::ensure!(
        selected_foreign_fasta_sha256 == generation.selected_foreign_sha256
            && scientific_inputs
                .foreign_fasta_sha256
                .binary_search(&selected_foreign_fasta_sha256)
                .is_ok(),
        "historical audit selected foreign-source identity mismatch"
    );
    let mut recorded_candidate_hashes = generation
        .candidates
        .iter()
        .map(|candidate| candidate.sha256.clone())
        .collect::<Vec<_>>();
    recorded_candidate_hashes.sort();
    recorded_candidate_hashes.dedup();
    let expected_candidate_hashes = if manifest.foreign_source_mode == ForeignSourceMode::Explicit {
        vec![selected_foreign_fasta_sha256.clone()]
    } else {
        scientific_inputs.foreign_fasta_sha256.clone()
    };
    anyhow::ensure!(
        recorded_candidate_hashes == expected_candidate_hashes,
        "historical audit candidate source identity mismatch"
    );
    let frozen =
        inspect_frozen_entrapment(&parameters, &manifest.target_fasta, &manifest.output_fasta)?;
    validate_measured_ratios(&generation.measured)?;
    anyhow::ensure!(
        frozen.measured == generation.measured,
        "historical audit counts or measured ratios do not match the generated FASTA"
    );
    let construction_identity = entrapment_construction_identity(&audit.database)?;
    let resource_identity = entrapment_resource_identity(
        &scientific_input_sha256,
        &generated_combined_fasta_sha256,
        &construction_identity,
        &generation.measured,
    )?;
    let mut lock = ExistingEntrapmentResourceLock {
        schema_version: 1,
        scientific_inputs: scientific_inputs.clone(),
        scientific_input_sha256,
        generation_implementation_identity: generation_implementation_identity(),
        target_fasta_sha256,
        foreign_fasta_sha256: scientific_inputs.foreign_fasta_sha256.clone(),
        selected_foreign_fasta_sha256,
        generated_combined_fasta_sha256,
        legacy_audit_artifact_sha256: sha256_file(audit_report_path)?,
        legacy_audit_schema_version: audit.schema_version,
        legacy_generation_schema_version: generation.schema_version,
        legacy_generation_input_sha256: generation.generation_input_sha256.clone(),
        historical_search_config_sha256: sha256_file(&manifest.search_config)?,
        database_report: audit.database,
        construction_identity,
        resource_identity,
        audit_artifact_path: audit_report_path.to_path_buf(),
        historical_search_config_path: manifest.search_config,
        payload_sha256: String::new(),
    };
    lock.payload_sha256 = resource_lock_payload_sha256(&lock)?;
    validate_existing_entrapment_resource_lock(&lock)?;
    write_json_atomic(output, &lock)?;
    let reopened: ExistingEntrapmentResourceLock = serde_json::from_slice(&std::fs::read(output)?)?;
    validate_existing_entrapment_resource_lock(&reopened)?;
    anyhow::ensure!(
        reopened.payload_sha256 == lock.payload_sha256
            && reopened.resource_identity == lock.resource_identity,
        "existing entrapment resource lock failed atomic reopen verification"
    );
    Ok(reopened)
}

fn parse_fasta(path: &Path) -> Result<Vec<FastaRecord>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read FASTA {}", path.display()))?;
    let mut records = Vec::new();
    let mut header: Option<String> = None;
    let mut sequence = String::new();

    let flush = |header: Option<String>, sequence: &mut String, records: &mut Vec<FastaRecord>| {
        if let Some(header) = header {
            if !sequence.is_empty() {
                let accession = header
                    .split_ascii_whitespace()
                    .next()
                    .unwrap_or(&header)
                    .to_owned();
                records.push(FastaRecord {
                    header,
                    accession,
                    sequence: std::mem::take(sequence),
                });
            }
        }
    };

    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(next_header) = line.strip_prefix('>') {
            flush(header.take(), &mut sequence, &mut records);
            header = Some(next_header.to_owned());
        } else {
            anyhow::ensure!(
                header.is_some(),
                "FASTA sequence appears before first header"
            );
            sequence.push_str(line);
        }
    }
    flush(header, &mut sequence, &mut records);
    anyhow::ensure!(
        !records.is_empty(),
        "FASTA contains no records: {}",
        path.display()
    );
    Ok(records)
}

fn hash_strings<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.as_bytes());
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

fn hash_mapping(mapping: &BTreeMap<String, String>) -> String {
    let mut hasher = Sha256::new();
    for (key, value) in mapping {
        hasher.update(key.as_bytes());
        hasher.update([0]);
        hasher.update(value.as_bytes());
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

fn is_entrapment_record(record: &FastaRecord) -> bool {
    record.accession.contains("Ent_") || record.header.contains("_p_target")
}

fn canonical_source_accession(accession: &str) -> String {
    let accession = accession.trim_end_matches("_p_target");
    if let Some(rest) = accession.strip_prefix("Ent_") {
        if let Some((serial, original)) = rest.split_once('_') {
            if serial.chars().all(|character| character.is_ascii_digit()) {
                return original.to_owned();
            }
        }
        return rest.to_owned();
    }
    let mut fields = accession.split('|').map(str::to_owned).collect::<Vec<_>>();
    if fields.len() > 1 {
        if let Some(stripped) = fields[1].strip_prefix("Ent_") {
            fields[1] = stripped.to_owned();
        }
        return fields.join("|");
    }
    accession.replace("Ent_", "")
}

fn split_combined_records(records: &[FastaRecord]) -> (Vec<FastaRecord>, Vec<FastaRecord>) {
    records
        .iter()
        .cloned()
        .partition(|record| !is_entrapment_record(record))
}

#[derive(Clone, Debug)]
struct ProteinSearchEvidence {
    record: FastaRecord,
    peptides: BTreeSet<String>,
    peptidoforms: BTreeSet<String>,
}

fn canonical_digested_keys(peptide: &sage_core::peptide::Peptide) -> (String, String) {
    let sequence = String::from_utf8_lossy(&peptide.sequence)
        .chars()
        .map(|residue| {
            let residue = residue.to_ascii_uppercase();
            if residue == 'I' {
                'L'
            } else {
                residue
            }
        })
        .collect::<String>();
    let modifications = peptide
        .modifications
        .iter()
        .map(|mass| mass.to_bits().to_string())
        .collect::<Vec<_>>()
        .join(",");
    let peptidoform = format!(
        "{}|n={:?}|c={:?}|m={}",
        sequence,
        peptide.nterm.map(f32::to_bits),
        peptide.cterm.map(f32::to_bits),
        modifications
    );
    (sequence, peptidoform)
}

fn protein_search_evidence(
    parameters: &Parameters,
    records: &[FastaRecord],
) -> Result<Vec<ProteinSearchEvidence>> {
    let mut by_accession = records
        .iter()
        .cloned()
        .map(|record| {
            (
                record.accession.clone(),
                ProteinSearchEvidence {
                    record,
                    peptides: BTreeSet::new(),
                    peptidoforms: BTreeSet::new(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    anyhow::ensure!(
        by_accession.len() == records.len(),
        "entrapment FASTA contains duplicate accessions"
    );
    for peptide in parameters
        .digest(&fasta_from_records(records))
        .into_iter()
        .filter(|peptide| !peptide.decoy)
    {
        let (canonical, peptidoform) = canonical_digested_keys(&peptide);
        for protein in &peptide.proteins {
            let accession = protein.to_string();
            let evidence = by_accession.get_mut(&accession).with_context(|| {
                format!("digested peptide refers to unknown FASTA protein {accession}")
            })?;
            evidence.peptides.insert(canonical.clone());
            evidence.peptidoforms.insert(peptidoform.clone());
        }
    }
    Ok(by_accession.into_values().collect())
}

fn stable_component_identity(proteins: &[&ProteinSearchEvidence]) -> String {
    let mut members = proteins
        .iter()
        .map(|protein| {
            let sequence_sha256 =
                format!("{:x}", Sha256::digest(protein.record.sequence.as_bytes()));
            format!(
                "{}\0{}",
                canonical_source_accession(&protein.record.accession),
                sequence_sha256
            )
        })
        .collect::<Vec<_>>();
    members.sort();
    let mut hasher = Sha256::new();
    hasher.update(b"sage-entrapment-component-v1\0");
    for member in members {
        hasher.update(member.as_bytes());
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

pub fn digestion_search_space_identity(parameters: &Parameters) -> Result<String> {
    let mut static_mods = parameters
        .static_mods
        .iter()
        .map(|(specificity, mass)| Ok((serde_json::to_string(specificity)?, mass.to_bits())))
        .collect::<Result<Vec<_>>>()?;
    static_mods.sort();
    let mut variable_mods = parameters
        .variable_mods
        .iter()
        .map(|(specificity, masses)| {
            let mut masses = masses.iter().map(|mass| mass.to_bits()).collect::<Vec<_>>();
            masses.sort_unstable();
            Ok((serde_json::to_string(specificity)?, masses))
        })
        .collect::<Result<Vec<_>>>()?;
    variable_mods.sort();
    let portable = serde_json::json!({
        "schema": "sage-entrapment-digestion-search-space-v1",
        "enzyme": parameters.enzyme,
        "peptide_min_mass_bits": parameters.peptide_min_mass.to_bits(),
        "peptide_max_mass_bits": parameters.peptide_max_mass.to_bits(),
        "static_mods": static_mods,
        "variable_mods": variable_mods,
        "max_variable_mods": parameters.max_variable_mods,
    });
    let mut hasher = Sha256::new();
    hasher.update(b"sage-entrapment-digestion-search-space-v1\0");
    hasher.update(serde_json::to_vec(&portable)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn artifact_payload_sha256(artifact: &EntrapmentPartitionArtifact) -> Result<String> {
    let mut payload = artifact.clone();
    payload.payload_sha256.clear();
    let mut hasher = Sha256::new();
    hasher.update(b"sage-entrapment-partition-payload-v1\0");
    hasher.update(serde_json::to_vec(&payload)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn artifact_partition_identity(artifact: &EntrapmentPartitionArtifact) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-entrapment-partition-identity-v1\0");
    hasher.update(artifact.dataset_identity.as_bytes());
    hasher.update(artifact.target_fasta_sha256.as_bytes());
    hasher.update(artifact.active_entrapment_fasta_sha256.as_bytes());
    hasher.update(artifact.digestion_search_space_identity.as_bytes());
    hasher.update(artifact.entrapment_construction_identity.as_bytes());
    hasher.update(artifact.seed.to_le_bytes());
    hasher.update(artifact.salt.as_bytes());
    hasher.update(
        artifact
            .requested_selection_fraction
            .to_bits()
            .to_le_bytes(),
    );
    hasher.update(artifact.requested_audit_fraction.to_bits().to_le_bytes());
    hasher.update(serde_json::to_vec(&artifact.component_assignments)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn partition_scientific_content(
    artifact: &EntrapmentPartitionArtifact,
    portable_dataset_identity: &str,
) -> EntrapmentPartitionScientificContentV1 {
    let selection_components = artifact
        .component_assignments
        .iter()
        .filter(|assignment| assignment.partition == EntrapmentPartitionAssignment::Selection)
        .map(|assignment| assignment.component_identity.as_str())
        .collect::<BTreeSet<_>>();
    let audit_components = artifact
        .component_assignments
        .iter()
        .filter(|assignment| assignment.partition == EntrapmentPartitionAssignment::Audit)
        .map(|assignment| assignment.component_identity.as_str())
        .collect::<BTreeSet<_>>();
    let selection_proteins = artifact
        .selection_proteins
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let audit_proteins = artifact
        .audit_proteins
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let selection_peptides = artifact
        .selection_canonical_peptides
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let audit_peptides = artifact
        .audit_canonical_peptides
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let selection_peptidoforms = artifact
        .selection_peptidoforms
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let audit_peptidoforms = artifact
        .audit_peptidoforms
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    EntrapmentPartitionScientificContentV1 {
        schema_version: ENTRAPMENT_PARTITION_SCIENTIFIC_CONTENT_SCHEMA_VERSION,
        partition_schema_version: artifact.schema_version,
        assignment_algorithm_schema: ENTRAPMENT_PARTITION_ASSIGNMENT_ALGORITHM_SCHEMA.into(),
        dataset_identity: portable_dataset_identity.into(),
        target_fasta_sha256: artifact.target_fasta_sha256.clone(),
        active_entrapment_fasta_sha256: artifact.active_entrapment_fasta_sha256.clone(),
        digestion_search_space_identity: artifact.digestion_search_space_identity.clone(),
        entrapment_construction_identity: artifact.entrapment_construction_identity.clone(),
        seed: artifact.seed,
        salt: artifact.salt.clone(),
        requested_selection_fraction: artifact.requested_selection_fraction,
        requested_audit_fraction: artifact.requested_audit_fraction,
        realized_selection_fraction: artifact.realized_selection_fraction,
        realized_audit_fraction: artifact.realized_audit_fraction,
        component_assignments: artifact.component_assignments.clone(),
        selection_proteins: artifact.selection_proteins.clone(),
        audit_proteins: artifact.audit_proteins.clone(),
        selection_canonical_peptides: artifact.selection_canonical_peptides.clone(),
        audit_canonical_peptides: artifact.audit_canonical_peptides.clone(),
        selection_peptidoforms: artifact.selection_peptidoforms.clone(),
        audit_peptidoforms: artifact.audit_peptidoforms.clone(),
        selection_ratios: artifact.selection_ratios.clone(),
        audit_ratios: artifact.audit_ratios.clone(),
        component_overlap_count: selection_components.intersection(&audit_components).count(),
        protein_overlap_count: selection_proteins.intersection(&audit_proteins).count(),
        canonical_peptide_overlap_count: selection_peptides.intersection(&audit_peptides).count(),
        peptidoform_overlap_count: selection_peptidoforms
            .intersection(&audit_peptidoforms)
            .count(),
        protein_group_overlap_count: selection_components.intersection(&audit_components).count(),
    }
}

pub fn entrapment_partition_scientific_content_sha256(
    artifact: &EntrapmentPartitionArtifact,
) -> Result<String> {
    entrapment_partition_scientific_content_sha256_for_dataset(artifact, &artifact.dataset_identity)
}

fn entrapment_partition_scientific_content_sha256_for_dataset(
    artifact: &EntrapmentPartitionArtifact,
    portable_dataset_identity: &str,
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-entrapment-partition-scientific-content-v1\0");
    hasher.update(serde_json::to_vec(&partition_scientific_content(
        artifact,
        portable_dataset_identity,
    ))?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_partition_structure(artifact: &EntrapmentPartitionArtifact) -> Result<()> {
    anyhow::ensure!(
        artifact.schema_version == 1,
        "unsupported entrapment partition schema"
    );
    anyhow::ensure!(
        artifact.partition_identity == artifact_partition_identity(artifact)?,
        "entrapment partition identity integrity failure"
    );
    anyhow::ensure!(
        artifact.payload_sha256 == artifact_payload_sha256(artifact)?,
        "entrapment partition payload integrity failure"
    );
    anyhow::ensure!(
        artifact.source_implementation_identity.len() == 64
            && artifact
                .source_implementation_identity
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "entrapment partition historical generator identity is invalid"
    );

    let content = partition_scientific_content(artifact, &artifact.dataset_identity);
    anyhow::ensure!(
        content.component_overlap_count == 0
            && content.protein_group_overlap_count == 0
            && content.protein_overlap_count == 0
            && content.canonical_peptide_overlap_count == 0
            && content.peptidoform_overlap_count == 0,
        "entrapment partition contains cross-partition overlap"
    );
    let mut component_ids = BTreeSet::new();
    let mut assigned_proteins = BTreeMap::<&str, EntrapmentPartitionAssignment>::new();
    for assignment in &artifact.component_assignments {
        anyhow::ensure!(
            component_ids.insert(assignment.component_identity.as_str()),
            "entrapment partition contains a duplicate component identity"
        );
        anyhow::ensure!(
            !assignment.proteins.is_empty(),
            "entrapment partition contains an empty component"
        );
        for protein in &assignment.proteins {
            anyhow::ensure!(
                assigned_proteins
                    .insert(protein.as_str(), assignment.partition)
                    .is_none(),
                "entrapment partition assigns a protein more than once"
            );
        }
    }
    for (proteins, role) in [
        (
            &artifact.selection_proteins,
            EntrapmentPartitionAssignment::Selection,
        ),
        (
            &artifact.audit_proteins,
            EntrapmentPartitionAssignment::Audit,
        ),
    ] {
        anyhow::ensure!(
            proteins.iter().collect::<BTreeSet<_>>().len() == proteins.len(),
            "entrapment partition contains duplicate protein membership"
        );
        for protein in proteins {
            anyhow::ensure!(
                assigned_proteins.get(protein.as_str()) == Some(&role),
                "entrapment partition component and protein membership disagree"
            );
        }
    }
    anyhow::ensure!(
        assigned_proteins.len()
            == artifact.selection_proteins.len() + artifact.audit_proteins.len(),
        "entrapment partition component membership is incomplete"
    );
    for (members, label) in [
        (&artifact.selection_canonical_peptides, "selection peptide"),
        (&artifact.audit_canonical_peptides, "audit peptide"),
        (&artifact.selection_peptidoforms, "selection peptidoform"),
        (&artifact.audit_peptidoforms, "audit peptidoform"),
    ] {
        anyhow::ensure!(
            members.iter().collect::<BTreeSet<_>>().len() == members.len(),
            "entrapment partition contains duplicate {label} membership"
        );
    }
    for (ratios, proteins, peptides, peptidoforms, label) in [
        (
            &artifact.selection_ratios,
            artifact.selection_proteins.len(),
            artifact.selection_canonical_peptides.len(),
            artifact.selection_peptidoforms.len(),
            "selection",
        ),
        (
            &artifact.audit_ratios,
            artifact.audit_proteins.len(),
            artifact.audit_canonical_peptides.len(),
            artifact.audit_peptidoforms.len(),
            "audit",
        ),
    ] {
        anyhow::ensure!(
            ratios.entrapment_proteins == proteins
                && ratios.entrapment_peptides == peptides
                && ratios.entrapment_peptidoforms == peptidoforms
                && ratios.protein_ratio.is_finite()
                && ratios.peptide_ratio.is_finite()
                && ratios.peptidoform_ratio.is_finite(),
            "entrapment partition {label} ratios disagree with membership"
        );
    }
    Ok(())
}

fn scientific_mismatch_paths(
    existing: &serde_json::Value,
    expected: &serde_json::Value,
    path: &str,
    output: &mut Vec<String>,
) {
    match (existing, expected) {
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            let keys = left.keys().chain(right.keys()).collect::<BTreeSet<_>>();
            for key in keys {
                let child = format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
                match (left.get(key), right.get(key)) {
                    (Some(left), Some(right)) => {
                        scientific_mismatch_paths(left, right, &child, output)
                    }
                    _ => output.push(child),
                }
            }
        }
        (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
            let length = left.len().max(right.len());
            for index in 0..length {
                let child = format!("{path}/{index}");
                match (left.get(index), right.get(index)) {
                    (Some(left), Some(right)) => {
                        scientific_mismatch_paths(left, right, &child, output)
                    }
                    _ => output.push(child),
                }
            }
        }
        _ if existing != expected => output.push(path.to_owned()),
        _ => {}
    }
}

fn verified_use_identity(
    scientific_content_sha256: &str,
    exact_artifact_sha256: &str,
    historical_generator: &HistoricalEntrapmentPartitionGeneratorIdentityV1,
    current_verifier: &CurrentEntrapmentPartitionVerifierIdentityV1,
) -> Result<String> {
    let portable = serde_json::json!({
        "schema_version": ENTRAPMENT_PARTITION_VERIFICATION_SCHEMA_VERSION,
        "scientific_content_sha256": scientific_content_sha256,
        "exact_artifact_sha256": exact_artifact_sha256,
        "historical_generator": historical_generator,
        "current_verifier": current_verifier,
    });
    let mut hasher = Sha256::new();
    hasher.update(b"sage-entrapment-partition-verified-use-v1\0");
    hasher.update(serde_json::to_vec(&portable)?);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn entrapment_construction_identity(report: &EntrapmentDatabaseReport) -> Result<String> {
    let value = match report {
        EntrapmentDatabaseReport::NativeGenerated { generation } => serde_json::json!({
            "mode": "native_generated",
            "generator_version": generation.generator_version,
            "selection_algorithm": generation.selection_algorithm,
            "target_sha256": generation.target_sha256,
            "selected_foreign_sha256": generation.selected_foreign_sha256,
            "output_sha256": generation.output_sha256,
            "seed": generation.seed,
            "protein_fold": generation.protein_fold,
            "shared_peptide_exclusion_mode": generation.shared_peptide_exclusion_mode,
            "foreign_source_mode": generation.foreign_source_mode,
            "automatically_recommended_foreign_sha256": generation.automatically_recommended_foreign_sha256,
            "override_applied": generation.override_applied,
            "selected_accessions": generation.selected_accessions,
            "excluded_shared_target_peptide": generation.excluded_shared_target_peptide,
            "source_accession_mapping": generation.source_accession_mapping,
            "deterministic_selection_sha256": generation.deterministic_selection_sha256,
        }),
        EntrapmentDatabaseReport::FrozenLegacy { frozen } => serde_json::json!({
            "mode": "frozen_legacy",
            "frozen_entrapment_sha256": frozen.frozen_entrapment_sha256,
            "target_sha256": frozen.target_sha256,
        }),
    };
    let mut hasher = Sha256::new();
    hasher.update(b"sage-entrapment-construction-identity-v1\0");
    hasher.update(serde_json::to_vec(&value)?);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn build_entrapment_partition(
    parameters: &Parameters,
    dataset_identity: &str,
    target_fasta: &Path,
    active_entrapment_fasta: &Path,
    entrapment_construction_identity: &str,
    config: &EntrapmentValidationConfig,
) -> Result<EntrapmentPartitionArtifact> {
    anyhow::ensure!(
        config.mode == EntrapmentValidationMode::SelectionAudit,
        "partition construction requires selection_audit mode"
    );
    anyhow::ensure!(
        config.partition_schema_version == 1,
        "unsupported entrapment partition schema"
    );
    anyhow::ensure!(
        config.selection_fraction > 0.0
            && config.audit_fraction > 0.0
            && (config.selection_fraction + config.audit_fraction - 1.0).abs() <= 1e-12,
        "selection/audit fractions must be positive and sum to one"
    );

    let target_records = parse_fasta(target_fasta)?;
    let combined_records = parse_fasta(active_entrapment_fasta)?;
    let (_, entrapment_records) = split_combined_records(&combined_records);
    anyhow::ensure!(
        entrapment_records.len() >= 2,
        "selection/audit partition requires at least two entrapment proteins"
    );
    let target_evidence = protein_search_evidence(parameters, &target_records)?;
    let entrapment_evidence = protein_search_evidence(parameters, &entrapment_records)?;
    let target_peptides = target_evidence
        .iter()
        .flat_map(|protein| protein.peptides.iter().cloned())
        .collect::<BTreeSet<_>>();
    let target_peptidoforms = target_evidence
        .iter()
        .flat_map(|protein| protein.peptidoforms.iter().cloned())
        .collect::<BTreeSet<_>>();
    for protein in &entrapment_evidence {
        anyhow::ensure!(
            protein.peptides.is_disjoint(&target_peptides),
            "entrapment protein {} shares a searchable canonical peptide with the target FASTA",
            protein.record.accession
        );
    }

    let mut parent = (0..entrapment_evidence.len()).collect::<Vec<_>>();
    fn find(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        index
    }
    fn union(parent: &mut [usize], left: usize, right: usize) {
        let left = find(parent, left);
        let right = find(parent, right);
        if left != right {
            let (keep, merge) = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            parent[merge] = keep;
        }
    }
    let mut peptide_owner = BTreeMap::<String, usize>::new();
    for (index, protein) in entrapment_evidence.iter().enumerate() {
        for peptide in &protein.peptides {
            if let Some(previous) = peptide_owner.insert(peptide.clone(), index) {
                union(&mut parent, previous, index);
            }
        }
    }
    let mut component_indices = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..entrapment_evidence.len() {
        let root = find(&mut parent, index);
        component_indices.entry(root).or_default().push(index);
    }
    anyhow::ensure!(
        component_indices.len() >= 2,
        "entrapment searchable-peptide graph has only one component; nonempty selection and audit partitions are impossible"
    );

    #[derive(Clone)]
    struct Component {
        identity: String,
        assignment_hash: String,
        proteins: Vec<String>,
        peptides: BTreeSet<String>,
        peptidoforms: BTreeSet<String>,
    }
    let mut components = component_indices
        .into_values()
        .map(|indices| {
            let members = indices
                .iter()
                .map(|index| &entrapment_evidence[*index])
                .collect::<Vec<_>>();
            let identity = stable_component_identity(&members);
            let mut assignment_hasher = Sha256::new();
            assignment_hasher.update(b"sage-entrapment-partition-assignment-v1\0");
            assignment_hasher.update(config.seed.to_le_bytes());
            assignment_hasher.update(config.salt.as_bytes());
            assignment_hasher.update(identity.as_bytes());
            let assignment_hash = format!("{:x}", assignment_hasher.finalize());
            let mut proteins = members
                .iter()
                .map(|protein| protein.record.accession.clone())
                .collect::<Vec<_>>();
            proteins.sort();
            let peptides = members
                .iter()
                .flat_map(|protein| protein.peptides.iter().cloned())
                .collect();
            let peptidoforms = members
                .iter()
                .flat_map(|protein| protein.peptidoforms.iter().cloned())
                .collect();
            Component {
                identity,
                assignment_hash,
                proteins,
                peptides,
                peptidoforms,
            }
        })
        .collect::<Vec<_>>();
    let mut component_identities = BTreeSet::new();
    let mut assignment_hashes = BTreeSet::new();
    for component in &components {
        anyhow::ensure!(
            component_identities.insert(component.identity.clone()),
            "entrapment partition contains indistinguishable component payloads"
        );
        anyhow::ensure!(
            assignment_hashes.insert(component.assignment_hash.clone()),
            "entrapment partition assignment hash collision"
        );
    }
    components.sort_by(|left, right| {
        left.assignment_hash
            .cmp(&right.assignment_hash)
            .then_with(|| left.identity.cmp(&right.identity))
    });

    let requested_audit_proteins = config.audit_fraction * entrapment_evidence.len() as f64;
    let mut cumulative = 0usize;
    let mut best_audit_components = 1usize;
    let mut best_distance = f64::INFINITY;
    for count in 1..components.len() {
        cumulative += components[count - 1].proteins.len();
        let distance = (cumulative as f64 - requested_audit_proteins).abs();
        if distance < best_distance {
            best_distance = distance;
            best_audit_components = count;
        }
    }

    let mut assignments = Vec::with_capacity(components.len());
    let mut selection_proteins = BTreeSet::new();
    let mut audit_proteins = BTreeSet::new();
    let mut selection_peptides = BTreeSet::new();
    let mut audit_peptides = BTreeSet::new();
    let mut selection_peptidoforms = BTreeSet::new();
    let mut audit_peptidoforms = BTreeSet::new();
    for (index, component) in components.into_iter().enumerate() {
        let partition = if index < best_audit_components {
            audit_proteins.extend(component.proteins.iter().cloned());
            audit_peptides.extend(component.peptides.iter().cloned());
            audit_peptidoforms.extend(component.peptidoforms.iter().cloned());
            EntrapmentPartitionAssignment::Audit
        } else {
            selection_proteins.extend(component.proteins.iter().cloned());
            selection_peptides.extend(component.peptides.iter().cloned());
            selection_peptidoforms.extend(component.peptidoforms.iter().cloned());
            EntrapmentPartitionAssignment::Selection
        };
        assignments.push(EntrapmentComponentAssignment {
            component_identity: component.identity,
            partition,
            proteins: component.proteins,
            canonical_peptide_count: component.peptides.len(),
            peptidoform_count: component.peptidoforms.len(),
        });
    }
    assignments.sort_by(|left, right| left.component_identity.cmp(&right.component_identity));
    anyhow::ensure!(
        !selection_proteins.is_empty() && !audit_proteins.is_empty(),
        "selection/audit partition produced an empty population"
    );
    anyhow::ensure!(
        selection_proteins.is_disjoint(&audit_proteins)
            && selection_peptides.is_disjoint(&audit_peptides)
            && selection_peptidoforms.is_disjoint(&audit_peptidoforms),
        "selection/audit partition overlaps at protein, peptide, or peptidoform level"
    );

    let ratios_for = |proteins: &BTreeSet<String>,
                      peptides: &BTreeSet<String>,
                      peptidoforms: &BTreeSet<String>| EntrapmentRatios {
        target_proteins: target_records.len(),
        entrapment_proteins: proteins.len(),
        protein_ratio: ratio(proteins.len(), target_records.len()),
        target_peptides: target_peptides.len(),
        entrapment_peptides: peptides.len(),
        peptide_ratio: ratio(peptides.len(), target_peptides.len()),
        target_peptidoforms: target_peptidoforms.len(),
        entrapment_peptidoforms: peptidoforms.len(),
        peptidoform_ratio: ratio(peptidoforms.len(), target_peptidoforms.len()),
    };
    let total_entrapment = entrapment_evidence.len() as f64;
    let selection_ratios = ratios_for(
        &selection_proteins,
        &selection_peptides,
        &selection_peptidoforms,
    );
    let audit_ratios = ratios_for(&audit_proteins, &audit_peptides, &audit_peptidoforms);
    for (population, ratios) in [("selection", &selection_ratios), ("audit", &audit_ratios)] {
        anyhow::ensure!(
            ratios.target_proteins > 0
                && ratios.target_peptides > 0
                && ratios.target_peptidoforms > 0
                && ratios.entrapment_proteins > 0
                && ratios.entrapment_peptides > 0
                && ratios.entrapment_peptidoforms > 0
                && ratios.protein_ratio.is_finite()
                && ratios.protein_ratio > 0.0
                && ratios.peptide_ratio.is_finite()
                && ratios.peptide_ratio > 0.0
                && ratios.peptidoform_ratio.is_finite()
                && ratios.peptidoform_ratio > 0.0,
            "{population} entrapment partition has an empty or invalid observable protein/peptide/peptidoform ratio"
        );
    }
    let mut artifact = EntrapmentPartitionArtifact {
        schema_version: 1,
        partition_identity: String::new(),
        dataset_identity: dataset_identity.into(),
        target_fasta_sha256: sha256_file(target_fasta)?,
        active_entrapment_fasta_sha256: sha256_file(active_entrapment_fasta)?,
        digestion_search_space_identity: digestion_search_space_identity(parameters)?,
        entrapment_construction_identity: entrapment_construction_identity.into(),
        seed: config.seed,
        salt: config.salt.clone(),
        requested_selection_fraction: config.selection_fraction,
        requested_audit_fraction: config.audit_fraction,
        realized_selection_fraction: selection_proteins.len() as f64 / total_entrapment,
        realized_audit_fraction: audit_proteins.len() as f64 / total_entrapment,
        component_assignments: assignments,
        selection_proteins: selection_proteins.into_iter().collect(),
        audit_proteins: audit_proteins.into_iter().collect(),
        selection_canonical_peptides: selection_peptides.iter().cloned().collect(),
        audit_canonical_peptides: audit_peptides.iter().cloned().collect(),
        selection_peptidoforms: selection_peptidoforms.iter().cloned().collect(),
        audit_peptidoforms: audit_peptidoforms.iter().cloned().collect(),
        selection_ratios,
        audit_ratios,
        source_implementation_identity: PARAMETER_OPTIMIZER_IMPLEMENTATION_SOURCE_SHA256.into(),
        payload_sha256: String::new(),
    };
    artifact.partition_identity = artifact_partition_identity(&artifact)?;
    artifact.payload_sha256 = artifact_payload_sha256(&artifact)?;
    Ok(artifact)
}

pub(crate) fn resolve_entrapment_partition_with_provenance_and_dataset_aliases(
    parameters: &Parameters,
    dataset_identities: PartitionDatasetIdentityContext<'_>,
    target_fasta: &Path,
    active_entrapment_fasta: &Path,
    entrapment_construction_identity: &str,
    config: &EntrapmentValidationConfig,
    artifact_path: &Path,
) -> Result<VerifiedEntrapmentPartition> {
    let expected = build_entrapment_partition(
        parameters,
        dataset_identities.current,
        target_fasta,
        active_entrapment_fasta,
        entrapment_construction_identity,
        config,
    )?;
    let existing = if artifact_path.is_file() {
        let existing: EntrapmentPartitionArtifact =
            serde_json::from_slice(&std::fs::read(artifact_path).with_context(|| {
                format!(
                    "failed to read entrapment partition {}",
                    artifact_path.display()
                )
            })?)
            .context("invalid entrapment partition artifact")?;
        existing
    } else {
        anyhow::ensure!(
            !config.require_existing_partition,
            "required existing entrapment partition artifact is missing: {}",
            artifact_path.display()
        );
        write_json_atomic(artifact_path, &expected)?;
        expected.clone()
    };

    // Authenticate the immutable historical artifact using its own original
    // fields before comparing it with anything reconstructed by this build.
    validate_partition_structure(&existing)?;
    let existing_scientific = partition_scientific_content(&existing, &expected.dataset_identity);
    let expected_scientific = partition_scientific_content(&expected, &expected.dataset_identity);
    let historical_dataset_identity_schema =
        if existing.dataset_identity == expected.dataset_identity {
            "sage-decoy-free-dataset-current-content-v1".to_owned()
        } else {
            dataset_identities
                .historical_aliases
                .iter()
                .find(|alias| alias.identity == existing.dataset_identity)
                .map(|alias| alias.schema.clone())
                .with_context(|| {
                    format!(
                        "entrapment partition dataset identity mismatch: historical={} current={}",
                        existing.dataset_identity, expected.dataset_identity
                    )
                })?
        };
    // Dataset identity algorithms are provenance layers, not assignment
    // algorithms. Once the current manifest independently authenticates the
    // historical identity through a supported alias, compare every scientific
    // field under the current portable content-derived dataset identity.
    if existing_scientific != expected_scientific {
        let existing_value = serde_json::to_value(&existing_scientific)?;
        let expected_value = serde_json::to_value(&expected_scientific)?;
        let mut mismatches = Vec::new();
        scientific_mismatch_paths(&existing_value, &expected_value, "", &mut mismatches);
        anyhow::bail!(
            "entrapment partition scientific content mismatch at: {}",
            mismatches.join(", ")
        );
    }

    let exact_artifact_sha256 = sha256_file(artifact_path)?;
    let scientific_content_sha256 = entrapment_partition_scientific_content_sha256_for_dataset(
        &existing,
        &expected.dataset_identity,
    )?;
    let historical_generator = HistoricalEntrapmentPartitionGeneratorIdentityV1 {
        schema_version: 1,
        generator_schema: "sage-entrapment-partition-generation-v1".into(),
        assignment_algorithm_schema: ENTRAPMENT_PARTITION_ASSIGNMENT_ALGORITHM_SCHEMA.into(),
        source_implementation_identity: existing.source_implementation_identity.clone(),
        historical_dataset_identity_schema,
        historical_dataset_identity: existing.dataset_identity.clone(),
        original_partition_identity: existing.partition_identity.clone(),
        original_artifact_sha256: exact_artifact_sha256.clone(),
        original_payload_sha256: existing.payload_sha256.clone(),
    };
    let executable =
        std::env::current_exe().context("failed to resolve partition verifier executable")?;
    let current_verifier = CurrentEntrapmentPartitionVerifierIdentityV1 {
        schema_version: 1,
        verification_algorithm_schema: "sage-entrapment-partition-verification-v1".into(),
        source_implementation_identity: PARAMETER_OPTIMIZER_IMPLEMENTATION_SOURCE_SHA256.into(),
        executable_sha256: sha256_file(&executable)?,
        current_dataset_identity_schema: "sage-decoy-free-dataset-v1-input-path-content-identity"
            .into(),
        current_dataset_identity: expected.dataset_identity.clone(),
    };
    let verified_use_sha256 = verified_use_identity(
        &scientific_content_sha256,
        &exact_artifact_sha256,
        &historical_generator,
        &current_verifier,
    )?;
    let verified_at_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_secs();
    Ok(VerifiedEntrapmentPartition {
        artifact: existing,
        verification: EntrapmentPartitionVerifiedUseV1 {
            schema_version: ENTRAPMENT_PARTITION_VERIFICATION_SCHEMA_VERSION,
            scientific_content_sha256,
            exact_artifact_sha256,
            historical_generator,
            current_verifier,
            verified_use_sha256,
            artifact_path: artifact_path.to_path_buf(),
            verified_at_unix_seconds,
        },
    })
}

pub fn resolve_entrapment_partition_with_provenance(
    parameters: &Parameters,
    dataset_identity: &str,
    target_fasta: &Path,
    active_entrapment_fasta: &Path,
    entrapment_construction_identity: &str,
    config: &EntrapmentValidationConfig,
    artifact_path: &Path,
) -> Result<VerifiedEntrapmentPartition> {
    resolve_entrapment_partition_with_provenance_and_dataset_aliases(
        parameters,
        PartitionDatasetIdentityContext {
            current: dataset_identity,
            historical_aliases: &[],
        },
        target_fasta,
        active_entrapment_fasta,
        entrapment_construction_identity,
        config,
        artifact_path,
    )
}

pub fn resolve_entrapment_partition(
    parameters: &Parameters,
    dataset_identity: &str,
    target_fasta: &Path,
    active_entrapment_fasta: &Path,
    entrapment_construction_identity: &str,
    config: &EntrapmentValidationConfig,
    artifact_path: &Path,
) -> Result<EntrapmentPartitionArtifact> {
    Ok(resolve_entrapment_partition_with_provenance(
        parameters,
        dataset_identity,
        target_fasta,
        active_entrapment_fasta,
        entrapment_construction_identity,
        config,
        artifact_path,
    )?
    .artifact)
}

fn fdrbench_canonical_sequence(sequence: &str) -> String {
    sequence
        .trim_end_matches('*')
        .chars()
        .map(|residue| match residue.to_ascii_uppercase() {
            'I' => 'L',
            'B' => 'N',
            'Z' => 'Q',
            other => other,
        })
        .collect()
}

fn parse_legacy_exclusions(path: &Path) -> Result<Vec<String>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read generation log {}", path.display()))?;
    let mut accessions = contents
        .lines()
        .filter_map(|line| {
            let (_, rest) = line.split_once("Ignore entrapment protein ")?;
            let (accession, _) = rest.split_once(" due to shared peptide")?;
            Some(canonical_source_accession(accession.trim()))
        })
        .collect::<Vec<_>>();
    accessions.sort();
    accessions.dedup();
    Ok(accessions)
}

fn fasta_from_records(records: &[FastaRecord]) -> Fasta {
    let text = records
        .iter()
        .map(|record| format!(">{}\n{}\n", record.accession, record.sequence))
        .collect::<String>();
    Fasta::parse(text, "rev_", false)
}

fn peptide_keys(
    parameters: &Parameters,
    records: &[FastaRecord],
) -> (BTreeSet<String>, BTreeSet<String>) {
    let fasta = fasta_from_records(records);
    let peptides = parameters.digest(&fasta);
    let mut unmodified = BTreeSet::new();
    let mut peptidoforms = BTreeSet::new();
    for peptide in peptides.into_iter().filter(|peptide| !peptide.decoy) {
        let sequence = String::from_utf8_lossy(&peptide.sequence)
            .chars()
            .map(|residue| {
                let residue = residue.to_ascii_uppercase();
                if residue == 'I' {
                    'L'
                } else {
                    residue
                }
            })
            .collect::<String>();
        unmodified.insert(sequence.clone());
        let modifications = peptide
            .modifications
            .iter()
            .map(|mass| mass.to_bits().to_string())
            .collect::<Vec<_>>()
            .join(",");
        peptidoforms.insert(format!(
            "{}|n={:?}|c={:?}|m={}",
            sequence,
            peptide.nterm.map(f32::to_bits),
            peptide.cterm.map(f32::to_bits),
            modifications
        ));
    }
    (unmodified, peptidoforms)
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        numerator as f64 / denominator as f64
    }
}

fn measure_records(
    parameters: &Parameters,
    target: &[FastaRecord],
    entrapment: &[FastaRecord],
) -> EntrapmentRatios {
    let (target_peptides, target_peptidoforms) = peptide_keys(parameters, target);
    let (entrapment_peptides, entrapment_peptidoforms) = peptide_keys(parameters, entrapment);
    EntrapmentRatios {
        target_proteins: target.len(),
        entrapment_proteins: entrapment.len(),
        protein_ratio: ratio(entrapment.len(), target.len()),
        target_peptides: target_peptides.len(),
        entrapment_peptides: entrapment_peptides.len(),
        peptide_ratio: ratio(entrapment_peptides.len(), target_peptides.len()),
        target_peptidoforms: target_peptidoforms.len(),
        entrapment_peptidoforms: entrapment_peptidoforms.len(),
        peptidoform_ratio: ratio(entrapment_peptidoforms.len(), target_peptidoforms.len()),
    }
}

pub fn measure_entrapment_ratios(
    parameters: &Parameters,
    target_fasta: &Path,
    entrapment_fasta: &Path,
) -> Result<EntrapmentRatios> {
    Ok(measure_records(
        parameters,
        &parse_fasta(target_fasta)?,
        &parse_fasta(entrapment_fasta)?,
    ))
}

pub fn inspect_frozen_entrapment(
    parameters: &Parameters,
    target_fasta: &Path,
    combined_fasta: &Path,
) -> Result<FrozenEntrapmentReport> {
    let target = parse_fasta(target_fasta)?;
    let combined = parse_fasta(combined_fasta)?;
    let (combined_targets, entrapment) = split_combined_records(&combined);
    anyhow::ensure!(
        !entrapment.is_empty(),
        "frozen entrapment FASTA contains no records labeled Ent_ or _p_target"
    );
    anyhow::ensure!(
        combined_targets.len() == target.len(),
        "frozen entrapment FASTA has {} target records; target FASTA has {}",
        combined_targets.len(),
        target.len()
    );
    anyhow::ensure!(
        combined_targets
            .iter()
            .zip(&target)
            .all(|(combined, source)| combined.accession == source.accession
                && fdrbench_canonical_sequence(&combined.sequence)
                    == fdrbench_canonical_sequence(&source.sequence)),
        "frozen entrapment FASTA does not preserve the target FASTA records and order"
    );
    Ok(FrozenEntrapmentReport {
        schema_version: 1,
        target_fasta: target_fasta.to_path_buf(),
        target_sha256: sha256_file(target_fasta)?,
        frozen_entrapment_fasta: combined_fasta.to_path_buf(),
        frozen_entrapment_sha256: sha256_file(combined_fasta)?,
        target_headers: target.len(),
        entrapment_headers: entrapment.len(),
        target_header_order_sha256: hash_strings(
            target.iter().map(|record| record.header.as_str()),
        ),
        entrapment_header_order_sha256: hash_strings(
            entrapment.iter().map(|record| record.header.as_str()),
        ),
        measured: measure_records(parameters, &target, &entrapment),
    })
}

fn java_string_hash(value: &str) -> u32 {
    value.encode_utf16().fold(0_u32, |hash, unit| {
        hash.wrapping_mul(31).wrapping_add(unit as u32)
    })
}

fn fdrbench_hashmap_order(records: &[FastaRecord]) -> Vec<FastaRecord> {
    // FDRBench 0.0.4 inserts `protein_id + "_p_target"` into a default Java 8+
    // HashMap, then constructs an ArrayList from keySet(). HashMap iteration
    // scans the final table by bucket and preserves insertion order within a
    // bucket. Reproducing that order is necessary before Collections.shuffle.
    let mut capacity = 16_usize;
    while records.len() > capacity * 3 / 4 {
        capacity *= 2;
    }
    let mut buckets = vec![Vec::new(); capacity];
    for record in records {
        let key = format!("{}_p_target", record.accession);
        let hash = java_string_hash(&key);
        let spread = hash ^ (hash >> 16);
        buckets[spread as usize & (capacity - 1)].push(record.clone());
    }
    buckets.into_iter().flatten().collect()
}

struct JavaRandom {
    state: u64,
}

impl JavaRandom {
    const MULTIPLIER: u64 = 0x5DEECE66D;
    const ADDEND: u64 = 0xB;
    const MASK: u64 = (1_u64 << 48) - 1;

    fn new(seed: u64) -> Self {
        Self {
            state: (seed ^ Self::MULTIPLIER) & Self::MASK,
        }
    }

    fn next(&mut self, bits: u32) -> u32 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::ADDEND)
            & Self::MASK;
        (self.state >> (48 - bits)) as u32
    }

    fn next_int(&mut self, bound: usize) -> usize {
        assert!(bound > 0 && bound <= i32::MAX as usize);
        if bound.is_power_of_two() {
            return ((bound as u64 * self.next(31) as u64) >> 31) as usize;
        }
        loop {
            let bits = self.next(31) as usize;
            let value = bits % bound;
            if bits - value + (bound - 1) <= i32::MAX as usize {
                return value;
            }
        }
    }
}

fn shared_exclusion_parameters(
    parameters: &Parameters,
    exclusion_mode: &SharedPeptideExclusionMode,
) -> Parameters {
    let mut exclusion_parameters = parameters.clone();
    if *exclusion_mode == SharedPeptideExclusionMode::Fdrbench004Compatibility {
        exclusion_parameters.peptide_min_mass = 0.0;
        exclusion_parameters.peptide_max_mass = f32::MAX;
        exclusion_parameters.static_mods.clear();
        exclusion_parameters.variable_mods.clear();
        exclusion_parameters.max_variable_mods = 0;
    }
    exclusion_parameters
}

fn shared_exclusion_records(
    records: &[FastaRecord],
    exclusion_mode: &SharedPeptideExclusionMode,
) -> Vec<FastaRecord> {
    if *exclusion_mode == SharedPeptideExclusionMode::Fdrbench004Compatibility {
        records
            .iter()
            .cloned()
            .map(|mut record| {
                record.sequence = fdrbench_canonical_sequence(&record.sequence);
                record
            })
            .collect()
    } else {
        records.to_vec()
    }
}

fn eligible_foreign_records(
    parameters: &Parameters,
    target_peptides: &BTreeSet<String>,
    foreign: &[FastaRecord],
    exclusion_mode: &SharedPeptideExclusionMode,
) -> (Vec<FastaRecord>, Vec<String>) {
    // Digest the candidate source once and use Sage's grouped protein mapping
    // to mark every protein contributing a shared target peptide. This is
    // equivalent to per-protein digestion but remains practical for PXD-sized
    // proteomes.
    let exclusion_parameters = shared_exclusion_parameters(parameters, exclusion_mode);
    let exclusion_records = shared_exclusion_records(foreign, exclusion_mode);
    let peptides = exclusion_parameters.digest(&fasta_from_records(&exclusion_records));
    let mut excluded_set = BTreeSet::new();
    for peptide in peptides.into_iter().filter(|peptide| !peptide.decoy) {
        let sequence = String::from_utf8_lossy(&peptide.sequence)
            .chars()
            .map(|residue| {
                let residue = residue.to_ascii_uppercase();
                if residue == 'I' {
                    'L'
                } else {
                    residue
                }
            })
            .collect::<String>();
        if target_peptides.contains(&sequence) {
            excluded_set.extend(peptide.proteins.iter().map(|protein| protein.to_string()));
        }
    }
    let mut eligible = Vec::new();
    let mut excluded = Vec::new();
    for record in foreign {
        if excluded_set.contains(&record.accession) {
            excluded.push(record.accession.clone());
        } else {
            eligible.push(record.clone());
        }
    }
    (eligible, excluded)
}

fn select_records(records: &[FastaRecord], count: usize, seed: u64) -> Result<Vec<FastaRecord>> {
    anyhow::ensure!(
        records.len() >= count,
        "foreign FASTA has only {} eligible proteins; {} are required",
        records.len(),
        count
    );
    let mut ranked = fdrbench_hashmap_order(records);
    let mut random = JavaRandom::new(seed);
    for index in (1..ranked.len()).rev() {
        let other = random.next_int(index + 1);
        ranked.swap(index, other);
    }
    ranked.truncate(count);
    Ok(ranked)
}

fn selection_score(measured: &EntrapmentRatios) -> f64 {
    let expected = measured.protein_ratio;
    let log_distance = |value: f64| {
        if value.is_finite() && value > 0.0 && expected.is_finite() && expected > 0.0 {
            (value / expected).ln().abs()
        } else {
            f64::INFINITY
        }
    };
    log_distance(measured.peptide_ratio) + log_distance(measured.peptidoform_ratio)
}

#[derive(Clone)]
struct CandidateMaterial {
    score: f64,
    fasta: PathBuf,
    sha256: String,
    selected: Vec<FastaRecord>,
    excluded: Vec<String>,
    measured: EntrapmentRatios,
}

fn same_source(left: &Path, left_sha256: &str, right: &Path) -> Result<bool> {
    Ok(left == right || (right.is_file() && sha256_file(right)? == left_sha256))
}

pub fn entrapment_generation_input_sha256(
    parameters: &Parameters,
    target_fasta: &Path,
    foreign_fastas: &[PathBuf],
    seed: u64,
    protein_fold: usize,
    source_mode: &ForeignSourceMode,
    exclusion_mode: &SharedPeptideExclusionMode,
    selected_foreign_fasta: Option<&Path>,
) -> Result<String> {
    let mut input_hasher = Sha256::new();
    input_hasher.update(default_generator_version().as_bytes());
    input_hasher.update([0]);
    input_hasher.update(default_selection_algorithm().as_bytes());
    input_hasher.update([0]);
    input_hasher.update(sha256_file(target_fasta)?.as_bytes());
    input_hasher.update(serde_json::to_vec(parameters)?);
    input_hasher.update(seed.to_le_bytes());
    input_hasher.update(protein_fold.to_le_bytes());
    input_hasher.update(format!("{:?}", source_mode).as_bytes());
    input_hasher.update(format!("{:?}", exclusion_mode).as_bytes());
    let mut foreign_hashes = foreign_fastas
        .iter()
        .map(|path| sha256_file(path))
        .collect::<Result<Vec<_>>>()?;
    foreign_hashes.sort();
    for sha256 in &foreign_hashes {
        input_hasher.update([0]);
        input_hasher.update(sha256.as_bytes());
    }
    if let Some(path) = selected_foreign_fasta {
        input_hasher.update([0]);
        input_hasher.update(sha256_file(path)?.as_bytes());
    }
    Ok(format!("{:x}", input_hasher.finalize()))
}

#[allow(clippy::too_many_arguments)]
pub fn validate_entrapment_generation_report_inputs(
    report: &EntrapmentGenerationReport,
    parameters: &Parameters,
    target_fasta: &Path,
    foreign_fastas: &[PathBuf],
    seed: u64,
    protein_fold: usize,
    source_mode: &ForeignSourceMode,
    exclusion_mode: &SharedPeptideExclusionMode,
    selected_foreign_fasta: Option<&Path>,
) -> Result<()> {
    anyhow::ensure!(
        report.generator_version == default_generator_version()
            && report.selection_algorithm == default_selection_algorithm(),
        "unsupported entrapment generation implementation"
    );
    match report.schema_version {
        2 => {
            let expected = entrapment_generation_input_sha256(
                parameters,
                target_fasta,
                foreign_fastas,
                seed,
                protein_fold,
                source_mode,
                exclusion_mode,
                selected_foreign_fasta,
            )?;
            anyhow::ensure!(
                report.generation_input_sha256 == expected,
                "legacy entrapment generation full-input identity mismatch"
            );
        }
        3 => {
            let expected = entrapment_generation_scientific_inputs(
                parameters,
                target_fasta,
                foreign_fastas,
                seed,
                protein_fold,
                source_mode,
                exclusion_mode,
                selected_foreign_fasta,
            )?;
            let expected_sha256 = entrapment_generation_scientific_input_sha256(&expected)?;
            anyhow::ensure!(
                report.scientific_inputs.as_ref() == Some(&expected)
                    && report.scientific_input_sha256.as_deref() == Some(expected_sha256.as_str()),
                "entrapment generation phase-scoped scientific-input identity mismatch"
            );
        }
        _ => anyhow::bail!("unsupported entrapment generation report schema"),
    }
    anyhow::ensure!(
        report.target_sha256 == sha256_file(target_fasta)?
            && report.seed == seed
            && report.protein_fold == protein_fold
            && &report.foreign_source_mode == source_mode
            && &report.shared_peptide_exclusion_mode == exclusion_mode,
        "entrapment generation report settings mismatch"
    );
    Ok(())
}

pub fn generate_foreign_entrapment(
    parameters: &Parameters,
    target_fasta: &Path,
    foreign_fastas: &[PathBuf],
    output_fasta: &Path,
    seed: u64,
    protein_fold: usize,
    source_mode: ForeignSourceMode,
    exclusion_mode: SharedPeptideExclusionMode,
    selected_foreign_fasta: Option<&Path>,
) -> Result<EntrapmentGenerationReport> {
    anyhow::ensure!(protein_fold > 0, "protein_fold must be positive");
    anyhow::ensure!(
        !foreign_fastas.is_empty(),
        "at least one foreign FASTA is required"
    );
    match source_mode {
        ForeignSourceMode::Automatic => anyhow::ensure!(
            selected_foreign_fasta.is_none(),
            "automatic foreign-source selection must not declare selected_foreign_fasta"
        ),
        ForeignSourceMode::Explicit | ForeignSourceMode::AutomaticWithOverride => {
            anyhow::ensure!(
                selected_foreign_fasta.is_some_and(Path::is_file),
                "{:?} foreign-source selection requires an existing selected_foreign_fasta",
                source_mode
            );
        }
    }

    let target = parse_fasta(target_fasta)?;
    let generation_input_sha256 = entrapment_generation_input_sha256(
        parameters,
        target_fasta,
        foreign_fastas,
        seed,
        protein_fold,
        &source_mode,
        &exclusion_mode,
        selected_foreign_fasta,
    )?;
    let scientific_inputs = entrapment_generation_scientific_inputs(
        parameters,
        target_fasta,
        foreign_fastas,
        seed,
        protein_fold,
        &source_mode,
        &exclusion_mode,
        selected_foreign_fasta,
    )?;
    let scientific_input_sha256 =
        entrapment_generation_scientific_input_sha256(&scientific_inputs)?;
    let exclusion_parameters = shared_exclusion_parameters(parameters, &exclusion_mode);
    let exclusion_target = shared_exclusion_records(&target, &exclusion_mode);
    let (target_peptides, _) = peptide_keys(&exclusion_parameters, &exclusion_target);
    let required = target.len().saturating_mul(protein_fold);
    let mut evaluated = Vec::new();
    let mut candidate_material = Vec::new();

    for foreign_fasta in foreign_fastas {
        let foreign_sha256 = sha256_file(foreign_fasta)?;
        if source_mode == ForeignSourceMode::Explicit
            && !same_source(
                foreign_fasta,
                &foreign_sha256,
                selected_foreign_fasta.context("explicit source is missing")?,
            )?
        {
            continue;
        }
        let foreign = parse_fasta(foreign_fasta)?;
        let (eligible, excluded) =
            eligible_foreign_records(parameters, &target_peptides, &foreign, &exclusion_mode);
        if eligible.len() < required {
            evaluated.push(ForeignCandidateReport {
                fasta: foreign_fasta.clone(),
                sha256: foreign_sha256,
                total_proteins: foreign.len(),
                eligible_proteins: eligible.len(),
                excluded_shared_target_peptide: excluded.len(),
                selected_proteins: 0,
                measured: EntrapmentRatios {
                    target_proteins: target.len(),
                    entrapment_proteins: 0,
                    protein_ratio: 0.0,
                    target_peptides: target_peptides.len(),
                    entrapment_peptides: 0,
                    peptide_ratio: 0.0,
                    target_peptidoforms: 0,
                    entrapment_peptidoforms: 0,
                    peptidoform_ratio: 0.0,
                },
                selection_score: f64::INFINITY,
            });
            continue;
        }
        let selected = select_records(&eligible, required, seed)?;
        let measured = measure_records(parameters, &target, &selected);
        let score = selection_score(&measured);
        evaluated.push(ForeignCandidateReport {
            fasta: foreign_fasta.clone(),
            sha256: foreign_sha256.clone(),
            total_proteins: foreign.len(),
            eligible_proteins: eligible.len(),
            excluded_shared_target_peptide: excluded.len(),
            selected_proteins: selected.len(),
            measured: measured.clone(),
            selection_score: score,
        });
        candidate_material.push(CandidateMaterial {
            score,
            fasta: foreign_fasta.clone(),
            sha256: foreign_sha256,
            selected,
            excluded,
            measured,
        });
    }

    candidate_material.sort_by(|left, right| {
        left.score
            .partial_cmp(&right.score)
            .unwrap_or(Ordering::Greater)
            .then_with(|| left.sha256.cmp(&right.sha256))
            .then_with(|| left.fasta.cmp(&right.fasta))
    });
    let Some(automatic) = candidate_material.first().cloned() else {
        anyhow::bail!("no foreign FASTA contains enough eligible proteins");
    };
    let selected_material = match source_mode {
        ForeignSourceMode::Automatic | ForeignSourceMode::Explicit => automatic.clone(),
        ForeignSourceMode::AutomaticWithOverride => {
            let requested = selected_foreign_fasta.context("override source is missing")?;
            let requested_sha256 = sha256_file(requested)?;
            candidate_material
                .iter()
                .find(|candidate| {
                    candidate.fasta == requested || candidate.sha256 == requested_sha256
                })
                .cloned()
                .with_context(|| {
                    format!(
                        "override source {} is not an eligible foreign candidate",
                        requested.display()
                    )
                })?
        }
    };
    let selected_foreign_sha256 = selected_material.sha256.clone();
    let selected_fasta = selected_material.fasta;
    let selected = selected_material.selected;
    let excluded = selected_material.excluded;
    let measured = selected_material.measured;

    let parent = output_fasta.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut output = String::new();
    for record in &target {
        output.push('>');
        output.push_str(&record.header);
        output.push('\n');
        output.push_str(&record.sequence);
        output.push('\n');
    }
    let mut mapping = BTreeMap::new();
    let mut selected_accessions = Vec::new();
    let mut entrapment_headers = Vec::new();
    for (index, record) in selected.iter().enumerate() {
        let entrapment_accession = format!("Ent_{:06}_{}", index + 1, record.accession);
        mapping.insert(entrapment_accession.clone(), record.accession.clone());
        selected_accessions.push(record.accession.clone());
        output.push('>');
        output.push_str(&entrapment_accession);
        output.push_str(" source=");
        output.push_str(&selected_fasta.display().to_string());
        output.push('\n');
        entrapment_headers.push(format!(
            "{} source={}",
            entrapment_accession,
            selected_fasta.display()
        ));
        output.push_str(&record.sequence);
        output.push('\n');
    }
    std::fs::write(output_fasta, output)
        .with_context(|| format!("failed to write {}", output_fasta.display()))?;

    let mut deterministic_hasher = Sha256::new();
    deterministic_hasher.update(b"sage-foreign-entrapment-selection-v1\0");
    deterministic_hasher.update(seed.to_le_bytes());
    deterministic_hasher.update(protein_fold.to_le_bytes());
    deterministic_hasher.update(sha256_file(&selected_fasta)?.as_bytes());
    for accession in &selected_accessions {
        deterministic_hasher.update([0]);
        deterministic_hasher.update(accession.as_bytes());
    }
    let deterministic_selection_sha256 = format!("{:x}", deterministic_hasher.finalize());
    let override_applied = source_mode == ForeignSourceMode::AutomaticWithOverride
        && selected_foreign_sha256 != automatic.sha256;

    Ok(EntrapmentGenerationReport {
        schema_version: 3,
        generation_input_sha256,
        scientific_inputs: Some(scientific_inputs),
        scientific_input_sha256: Some(scientific_input_sha256),
        generator_version: default_generator_version(),
        selection_algorithm: default_selection_algorithm(),
        target_fasta: target_fasta.to_path_buf(),
        target_sha256: sha256_file(target_fasta)?,
        selected_foreign_fasta: selected_fasta.clone(),
        selected_foreign_sha256: sha256_file(&selected_fasta)?,
        output_fasta: output_fasta.to_path_buf(),
        output_sha256: sha256_file(output_fasta)?,
        seed,
        protein_fold,
        shared_peptide_exclusion_mode: exclusion_mode,
        foreign_source_mode: source_mode.clone(),
        automatically_recommended_foreign_fasta: (source_mode != ForeignSourceMode::Explicit)
            .then(|| automatic.fasta.clone()),
        automatically_recommended_foreign_sha256: (source_mode != ForeignSourceMode::Explicit)
            .then(|| automatic.sha256.clone()),
        override_applied,
        candidates: evaluated,
        selected_accession_order_sha256: hash_strings(
            selected_accessions.iter().map(String::as_str),
        ),
        excluded_accessions_sha256: hash_strings(excluded.iter().map(String::as_str)),
        target_header_order_sha256: hash_strings(
            target.iter().map(|record| record.header.as_str()),
        ),
        entrapment_header_order_sha256: hash_strings(entrapment_headers.iter().map(String::as_str)),
        source_accession_mapping_sha256: hash_mapping(&mapping),
        deterministic_selection_sha256,
        selected_accessions,
        excluded_shared_target_peptide: excluded,
        source_accession_mapping: mapping,
        measured,
    })
}

fn canonical_sequence_mapping(records: &[FastaRecord]) -> BTreeMap<String, String> {
    records
        .iter()
        .map(|record| {
            let mut hasher = Sha256::new();
            hasher.update(fdrbench_canonical_sequence(&record.sequence).as_bytes());
            (
                canonical_source_accession(&record.accession),
                format!("{:x}", hasher.finalize()),
            )
        })
        .collect()
}

pub fn compare_generated_to_legacy(
    parameters: &Parameters,
    native: &EntrapmentGenerationReport,
    reference: &LegacyEntrapmentReference,
) -> Result<EntrapmentFastaParityReport> {
    let native_records = parse_fasta(&native.output_fasta)?;
    let legacy_records = parse_fasta(&reference.fasta)?;
    let (native_targets, native_entrapment) = split_combined_records(&native_records);
    let (legacy_targets, legacy_entrapment) = split_combined_records(&legacy_records);
    anyhow::ensure!(
        native_entrapment.len() == native.selected_accessions.len(),
        "native report and generated FASTA disagree about selected protein count"
    );
    anyhow::ensure!(
        !legacy_entrapment.is_empty(),
        "legacy FASTA contains no records labeled Ent_ or _p_target"
    );

    let native_accessions = native
        .selected_accessions
        .iter()
        .map(|accession| canonical_source_accession(accession))
        .collect::<Vec<_>>();
    let legacy_accessions = legacy_entrapment
        .iter()
        .map(|record| canonical_source_accession(&record.accession))
        .collect::<Vec<_>>();
    let native_accession_set = native_accessions.iter().cloned().collect::<BTreeSet<_>>();
    let legacy_accession_set = legacy_accessions.iter().cloned().collect::<BTreeSet<_>>();
    let native_mapping = canonical_sequence_mapping(&native_entrapment);
    let legacy_mapping = canonical_sequence_mapping(&legacy_entrapment);

    let native_excluded = native
        .excluded_shared_target_peptide
        .iter()
        .map(|accession| canonical_source_accession(accession))
        .collect::<BTreeSet<_>>();
    let legacy_excluded = reference
        .generation_log
        .as_ref()
        .map(|path| parse_legacy_exclusions(path))
        .transpose()?
        .map(|accessions| accessions.into_iter().collect::<BTreeSet<_>>());

    let legacy_report =
        inspect_frozen_entrapment(parameters, &native.target_fasta, &reference.fasta)?;
    let legacy_source_sha256 = reference
        .foreign_fasta
        .as_ref()
        .map(|path| sha256_file(path))
        .transpose()?;
    let selected_foreign_source_match = legacy_source_sha256
        .as_ref()
        .map(|sha256| sha256 == &native.selected_foreign_sha256);
    let legacy_only_excluded_accessions = legacy_excluded
        .as_ref()
        .map(|legacy| legacy.difference(&native_excluded).cloned().collect())
        .unwrap_or_default();
    let native_only_excluded_accessions = legacy_excluded
        .as_ref()
        .map(|legacy| native_excluded.difference(legacy).cloned().collect())
        .unwrap_or_default();

    Ok(EntrapmentFastaParityReport {
        schema_version: 1,
        native_fasta: native.output_fasta.clone(),
        native_sha256: native.output_sha256.clone(),
        legacy_fasta: reference.fasta.clone(),
        legacy_sha256: legacy_report.frozen_entrapment_sha256,
        exact_fasta_match: sha256_file(&native.output_fasta)? == sha256_file(&reference.fasta)?,
        native_selected_foreign_fasta: native.selected_foreign_fasta.clone(),
        legacy_selected_foreign_fasta: reference.foreign_fasta.clone(),
        selected_foreign_source_match,
        selected_accession_set_match: native_accession_set == legacy_accession_set,
        selected_accession_order_match: native_accessions == legacy_accessions,
        native_only_selected_accessions: native_accession_set
            .difference(&legacy_accession_set)
            .cloned()
            .collect(),
        legacy_only_selected_accessions: legacy_accession_set
            .difference(&native_accession_set)
            .cloned()
            .collect(),
        excluded_shared_peptide_set_match: legacy_excluded
            .as_ref()
            .map(|legacy| legacy == &native_excluded),
        native_only_excluded_accessions,
        legacy_only_excluded_accessions,
        target_header_order_match: hash_strings(
            native_targets.iter().map(|record| record.header.as_str()),
        ) == hash_strings(
            legacy_targets.iter().map(|record| record.header.as_str()),
        ),
        entrapment_header_order_match: native_accessions == legacy_accessions,
        exact_header_order_match: hash_strings(
            native_records.iter().map(|record| record.header.as_str()),
        ) == hash_strings(
            legacy_records.iter().map(|record| record.header.as_str()),
        ),
        native_mapping_sha256: hash_mapping(&native_mapping),
        legacy_mapping_sha256: hash_mapping(&legacy_mapping),
        mapping_match: native_mapping == legacy_mapping,
        seed: native.seed,
        deterministic_selection_sha256: native.deterministic_selection_sha256.clone(),
        ratios: CountRatioComparison {
            protein_ratio_delta: native.measured.protein_ratio
                - legacy_report.measured.protein_ratio,
            peptide_ratio_delta: native.measured.peptide_ratio
                - legacy_report.measured.peptide_ratio,
            peptidoform_ratio_delta: native.measured.peptidoform_ratio
                - legacy_report.measured.peptidoform_ratio,
            native: native.measured.clone(),
            legacy: legacy_report.measured,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sage_core::database::{EnzymeBuilder, Parameters};
    use sage_core::ion_series::Kind;
    use sage_core::modification::ModificationSpecificity;
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_directory(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "sage-entrapment-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn parameters() -> Parameters {
        Parameters {
            bucket_size: 8192,
            enzyme: EnzymeBuilder {
                missed_cleavages: Some(0),
                min_len: Some(3),
                max_len: Some(50),
                cleave_at: Some("KR".into()),
                restrict: Some("P".into()),
                c_terminal: Some(true),
                semi_enzymatic: Some(false),
            },
            peptide_min_mass: 0.0,
            peptide_max_mass: 10000.0,
            ion_kinds: vec![Kind::B, Kind::Y],
            min_ion_index: 2,
            static_mods: HashMap::new(),
            variable_mods: HashMap::new(),
            max_variable_mods: 2,
            decoy_tag: "rev_".into(),
            generate_decoys: false,
            fasta: String::new(),
            prefilter_chunk_size: 0,
            prefilter: false,
            prefilter_low_memory: true,
        }
    }

    fn partition_config() -> EntrapmentValidationConfig {
        EntrapmentValidationConfig {
            mode: EntrapmentValidationMode::SelectionAudit,
            partition_schema_version: 1,
            seed: 73,
            salt: "synthetic-dataset-local-holdout-v1".into(),
            selection_fraction: 0.5,
            audit_fraction: 0.5,
            require_existing_partition: false,
        }
    }

    fn partition_fastas(directory: &Path, reverse: bool) -> (PathBuf, PathBuf) {
        let target = directory.join(if reverse {
            "target-reversed.fasta"
        } else {
            "target.fasta"
        });
        let combined = directory.join(if reverse {
            "combined-reversed.fasta"
        } else {
            "combined.fasta"
        });
        std::fs::write(&target, b">Target_A\nTTTKQQQQK\n>Target_B\nLLLLKMMMMK\n").unwrap();
        let entrapments = if reverse {
            ">Ent_D\nNNNNKPPPPK\n>Ent_C\nDDDKEEEEK\n>Ent_B\nGGGKCCCCK\n>Ent_A\nAAAKCCCCK\n"
        } else {
            ">Ent_A\nAAAKCCCCK\n>Ent_B\nGGGKCCCCK\n>Ent_C\nDDDKEEEEK\n>Ent_D\nNNNNKPPPPK\n"
        };
        std::fs::write(
            &combined,
            format!(">Target_A\nTTTKQQQQK\n>Target_B\nLLLLKMMMMK\n{entrapments}"),
        )
        .unwrap();
        (target, combined)
    }

    #[test]
    fn selection_audit_partition_is_order_independent_and_component_safe() {
        let directory = test_directory("partition-order");
        let (target, combined) = partition_fastas(&directory, false);
        let (target_reversed, combined_reversed) = partition_fastas(&directory, true);
        let mut first_parameters = parameters();
        first_parameters.fasta = "/unrelated/machine/a/target.fasta".into();
        let first = build_entrapment_partition(
            &first_parameters,
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
        )
        .unwrap();
        let mut reordered_parameters = parameters();
        reordered_parameters.fasta = "D:\\unrelated\\machine\\b\\target.fasta".into();
        let reordered = build_entrapment_partition(
            &reordered_parameters,
            "dataset",
            &target_reversed,
            &combined_reversed,
            "construction",
            &partition_config(),
        )
        .unwrap();
        assert_eq!(first.component_assignments, reordered.component_assignments);
        assert_eq!(
            first.digestion_search_space_identity,
            reordered.digestion_search_space_identity
        );
        assert_eq!(first.selection_proteins, reordered.selection_proteins);
        assert_eq!(first.audit_proteins, reordered.audit_proteins);
        assert_eq!(first.selection_ratios, reordered.selection_ratios);
        assert_eq!(first.audit_ratios, reordered.audit_ratios);
        let concurrent = (0..4)
            .map(|_| {
                let target = target.clone();
                let combined = combined.clone();
                std::thread::spawn(move || {
                    build_entrapment_partition(
                        &parameters(),
                        "dataset",
                        &target,
                        &combined,
                        "construction",
                        &partition_config(),
                    )
                    .unwrap()
                    .component_assignments
                })
            })
            .collect::<Vec<_>>();
        for assignment in concurrent {
            assert_eq!(first.component_assignments, assignment.join().unwrap());
        }
        assert!(!first.selection_proteins.is_empty());
        assert!(!first.audit_proteins.is_empty());
        assert!(first
            .selection_protein_set()
            .is_disjoint(&first.audit_protein_set()));
        assert!(first
            .selection_canonical_peptides
            .iter()
            .all(|peptide| !first.audit_canonical_peptides.contains(peptide)));
        assert!(first
            .selection_peptidoforms
            .iter()
            .all(|form| !first.audit_peptidoforms.contains(form)));

        let assignment = |protein: &str| {
            first
                .component_assignments
                .iter()
                .find(|component| component.proteins.iter().any(|item| item == protein))
                .map(|component| component.partition)
                .unwrap()
        };
        assert_eq!(assignment("Ent_A"), assignment("Ent_B"));
        let selection_view = serde_json::to_string(&first.selection_view()).unwrap();
        for audit_protein in &first.audit_proteins {
            assert!(!selection_view.contains(audit_protein));
        }
        assert!(!selection_view.contains("audit_ratios"));
        assert_eq!(
            first.selection_ratios.entrapment_proteins + first.audit_ratios.entrapment_proteins,
            4
        );
        assert_eq!(first.selection_ratios.target_proteins, 2);
        assert_eq!(first.audit_ratios.target_proteins, 2);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn target_shared_peptides_and_malformed_partition_fail_closed() {
        let directory = test_directory("partition-fail-closed");
        let target = directory.join("target.fasta");
        let combined = directory.join("combined.fasta");
        std::fs::write(&target, b">Target\nAAAKCCCCK\n").unwrap();
        std::fs::write(
            &combined,
            b">Target\nAAAKCCCCK\n>Ent_shared\nGGGKCCCCK\n>Ent_other\nDDDKEEEEK\n",
        )
        .unwrap();
        assert!(build_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
        )
        .unwrap_err()
        .to_string()
        .contains("shares a searchable canonical peptide"));

        std::fs::write(&target, b">Target\nTARGETPEPK\n").unwrap();
        std::fs::write(
            &combined,
            b">Target\nTARGETPEPK\n>Ent_one\nAAAKSHAREDK\n>Ent_two\nGGGKSHAREDK\n",
        )
        .unwrap();
        assert!(build_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
        )
        .unwrap_err()
        .to_string()
        .contains("only one component"));

        // Distinct FASTA accessions that canonicalize to the same source and
        // have the same non-searchable sequence would otherwise produce two
        // indistinguishable singleton component payloads. Fail closed instead
        // of allowing record order to resolve the tie.
        std::fs::write(&target, b">Target\nTARGETPEPK\n").unwrap();
        std::fs::write(
            &combined,
            b">Target\nTARGETPEPK\n>Ent_000001_same\nAA\n>Ent_000002_same\nAA\n>Ent_other\nDDDKEEEEK\n",
        )
        .unwrap();
        assert!(build_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
        )
        .unwrap_err()
        .to_string()
        .contains("indistinguishable component payloads"));

        let (target, combined) = partition_fastas(&directory, false);
        let artifact_path = directory.join("partition.json");
        let artifact = resolve_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
            &artifact_path,
        )
        .unwrap();
        let mut corrupt = artifact;
        corrupt.selection_proteins.push("Ent_missing".into());
        write_json_atomic(&artifact_path, &corrupt).unwrap();
        assert!(resolve_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
            &artifact_path,
        )
        .unwrap_err()
        .to_string()
        .contains("payload integrity failure"));

        let mut inconsistent = resolve_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
            &directory.join("fresh-partition.json"),
        )
        .unwrap();
        inconsistent.component_assignments.pop();
        inconsistent.selection_ratios.protein_ratio = 0.123;
        inconsistent.payload_sha256 = artifact_payload_sha256(&inconsistent).unwrap();
        write_json_atomic(&artifact_path, &inconsistent).unwrap();
        assert!(resolve_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
            &artifact_path,
        )
        .unwrap_err()
        .to_string()
        .contains("partition identity integrity failure"));

        let mut require_existing = partition_config();
        require_existing.require_existing_partition = true;
        assert!(resolve_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &require_existing,
            &directory.join("missing.json"),
        )
        .unwrap_err()
        .to_string()
        .contains("required existing"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn write_self_consistent_partition(path: &Path, artifact: &mut EntrapmentPartitionArtifact) {
        artifact.partition_identity = artifact_partition_identity(artifact).unwrap();
        artifact.payload_sha256 = artifact_payload_sha256(artifact).unwrap();
        write_json_atomic(path, artifact).unwrap();
    }

    #[test]
    fn layered_partition_provenance_accepts_historical_generator_only() {
        let directory = test_directory("partition-layered-provenance");
        let (target, combined) = partition_fastas(&directory, false);
        let artifact_path = directory.join("partition.json");
        let mut historical = build_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
        )
        .unwrap();
        historical.source_implementation_identity = "a".repeat(64);
        historical.dataset_identity = "legacy-dataset-identity".into();
        write_self_consistent_partition(&artifact_path, &mut historical);
        let original_bytes = std::fs::read(&artifact_path).unwrap();
        let original_payload = historical.payload_sha256.clone();

        let error = resolve_entrapment_partition_with_provenance(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
            &artifact_path,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("dataset identity mismatch"), "{error}");
        let aliases = [HistoricalDatasetIdentityAlias {
            schema: "historical-directory-unaware-v1".into(),
            identity: "legacy-dataset-identity".into(),
        }];
        let verified = resolve_entrapment_partition_with_provenance_and_dataset_aliases(
            &parameters(),
            PartitionDatasetIdentityContext {
                current: "dataset",
                historical_aliases: &aliases,
            },
            &target,
            &combined,
            "construction",
            &partition_config(),
            &artifact_path,
        )
        .unwrap();
        assert_eq!(
            verified
                .verification
                .historical_generator
                .source_implementation_identity,
            "a".repeat(64)
        );
        assert_eq!(
            verified
                .verification
                .current_verifier
                .source_implementation_identity,
            PARAMETER_OPTIMIZER_IMPLEMENTATION_SOURCE_SHA256
        );
        assert_eq!(
            verified
                .verification
                .historical_generator
                .original_payload_sha256,
            original_payload
        );
        assert_eq!(
            verified
                .verification
                .historical_generator
                .historical_dataset_identity,
            "legacy-dataset-identity"
        );
        assert_eq!(
            verified
                .verification
                .current_verifier
                .current_dataset_identity,
            "dataset"
        );
        assert_eq!(std::fs::read(&artifact_path).unwrap(), original_bytes);
        assert_eq!(
            verified.verification.scientific_content_sha256,
            entrapment_partition_scientific_content_sha256_for_dataset(&historical, "dataset")
                .unwrap()
        );
        assert_ne!(
            verified.verification.scientific_content_sha256,
            entrapment_partition_scientific_content_sha256(&historical).unwrap(),
            "legacy path-derived dataset identity must not define portable scientific content"
        );
        let mut another_verifier = verified.verification.current_verifier.clone();
        another_verifier.source_implementation_identity = "b".repeat(64);
        assert_ne!(
            verified.verification.verified_use_sha256,
            verified_use_identity(
                &verified.verification.scientific_content_sha256,
                &verified.verification.exact_artifact_sha256,
                &verified.verification.historical_generator,
                &another_verifier,
            )
            .unwrap(),
            "current verifier changes belong to verified-use provenance, not partition science"
        );

        let relocated = directory.join("relocated.json");
        std::fs::copy(&artifact_path, &relocated).unwrap();
        let relocated_verified = resolve_entrapment_partition_with_provenance_and_dataset_aliases(
            &parameters(),
            PartitionDatasetIdentityContext {
                current: "dataset",
                historical_aliases: &aliases,
            },
            &target,
            &combined,
            "construction",
            &partition_config(),
            &relocated,
        )
        .unwrap();
        assert_eq!(
            verified.verification.scientific_content_sha256,
            relocated_verified.verification.scientific_content_sha256
        );
        assert_eq!(
            verified.verification.exact_artifact_sha256,
            relocated_verified.verification.exact_artifact_sha256
        );
        assert_eq!(
            verified.verification.verified_use_sha256,
            relocated_verified.verification.verified_use_sha256
        );
        assert_ne!(
            verified.verification.artifact_path,
            relocated_verified.verification.artifact_path
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn layered_partition_provenance_rejects_every_scientific_change() {
        let directory = test_directory("partition-layered-mismatch");
        let (target, combined) = partition_fastas(&directory, false);
        let baseline = build_entrapment_partition(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
        )
        .unwrap();

        let verify_artifact = |name: &str, mut artifact: EntrapmentPartitionArtifact| {
            let path = directory.join(name);
            write_self_consistent_partition(&path, &mut artifact);
            resolve_entrapment_partition_with_provenance(
                &parameters(),
                "dataset",
                &target,
                &combined,
                "construction",
                &partition_config(),
                &path,
            )
            .unwrap_err()
            .to_string()
        };

        let mut assignment = baseline.clone();
        assignment.component_assignments[0].partition =
            match assignment.component_assignments[0].partition {
                EntrapmentPartitionAssignment::Selection => EntrapmentPartitionAssignment::Audit,
                EntrapmentPartitionAssignment::Audit => EntrapmentPartitionAssignment::Selection,
            };
        assert!(verify_artifact("assignment.json", assignment)
            .contains("component and protein membership disagree"));

        let mut component = baseline.clone();
        component.component_assignments[0]
            .proteins
            .push("Ent_extra".into());
        assert!(verify_artifact("component.json", component)
            .contains("component membership is incomplete"));

        let mut ratio = baseline.clone();
        ratio.selection_ratios.protein_ratio += 0.01;
        assert!(verify_artifact("ratio.json", ratio).contains("scientific content mismatch"));

        let mut overlap = baseline.clone();
        overlap
            .selection_proteins
            .push(overlap.audit_proteins[0].clone());
        assert!(verify_artifact("overlap.json", overlap).contains("cross-partition overlap"));

        let mut unsupported = baseline.clone();
        unsupported.schema_version = 2;
        assert!(verify_artifact("unsupported.json", unsupported)
            .contains("unsupported entrapment partition schema"));

        let path = directory.join("baseline.json");
        let mut self_consistent = baseline.clone();
        write_self_consistent_partition(&path, &mut self_consistent);
        for (config, expected_path) in [
            (
                {
                    let mut c = partition_config();
                    c.seed += 1;
                    c
                },
                "/seed",
            ),
            (
                {
                    let mut c = partition_config();
                    c.salt.push_str("-changed");
                    c
                },
                "/salt",
            ),
            (
                {
                    let mut c = partition_config();
                    c.selection_fraction = 0.4;
                    c.audit_fraction = 0.6;
                    c
                },
                "/requested_",
            ),
        ] {
            let error = resolve_entrapment_partition_with_provenance(
                &parameters(),
                "dataset",
                &target,
                &combined,
                "construction",
                &config,
                &path,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(expected_path), "{error}");
        }
        let error = resolve_entrapment_partition_with_provenance(
            &parameters(),
            "dataset",
            &target,
            &combined,
            "different-construction",
            &partition_config(),
            &path,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("/entrapment_construction_identity"),
            "{error}"
        );
        let mut changed_parameters = parameters();
        changed_parameters.enzyme.missed_cleavages = Some(1);
        let error = resolve_entrapment_partition_with_provenance(
            &changed_parameters,
            "dataset",
            &target,
            &combined,
            "construction",
            &partition_config(),
            &path,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("/digestion_search_space_identity"),
            "{error}"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn ratios_use_native_digest() {
        let target = vec![FastaRecord {
            header: "target".into(),
            accession: "target".into(),
            sequence: "AAAKCCCCK".into(),
        }];
        let entrapment = vec![FastaRecord {
            header: "foreign".into(),
            accession: "foreign".into(),
            sequence: "DDDKEEEEK".into(),
        }];
        let measured = measure_records(&parameters(), &target, &entrapment);
        assert_eq!(measured.protein_ratio, 1.0);
        assert_eq!(measured.peptide_ratio, 1.0);
    }

    #[test]
    fn frozen_combined_fasta_does_not_count_targets_as_entrapment() {
        let directory = test_directory("frozen-ratios");
        let target = directory.join("target.fasta");
        let combined = directory.join("combined.fasta");
        std::fs::write(&target, b">target\nAAAKCCCCK\n").unwrap();
        std::fs::write(&combined, b">target\nAAAKCCCCK\n>Ent_foreign\nDDDKEEEEK\n").unwrap();
        let report = inspect_frozen_entrapment(&parameters(), &target, &combined).unwrap();
        assert_eq!(report.target_headers, 1);
        assert_eq!(report.entrapment_headers, 1);
        assert_eq!(report.measured.protein_ratio, 1.0);
        assert_eq!(report.measured.peptide_ratio, 1.0);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn source_modes_are_distinct_and_automatic_is_order_independent() {
        let directory = test_directory("source-modes");
        let target = directory.join("target.fasta");
        let good = directory.join("good.fasta");
        let sparse = directory.join("sparse.fasta");
        std::fs::write(&target, b">target\nAAAKCCCCK\n").unwrap();
        std::fs::write(&good, b">tr|A|A_SPEC\nDDDKEEEEK\n").unwrap();
        std::fs::write(&sparse, b">tr|B|B_SPEC\nLLLLLLLLK\n").unwrap();
        let candidates = vec![sparse.clone(), good.clone()];

        let automatic = generate_foreign_entrapment(
            &parameters(),
            &target,
            &candidates,
            &directory.join("automatic.fasta"),
            42,
            1,
            ForeignSourceMode::Automatic,
            SharedPeptideExclusionMode::SageSearchSpace,
            None,
        )
        .unwrap();
        assert_eq!(automatic.selected_foreign_fasta, good);
        assert!(!automatic.override_applied);

        let reversed = generate_foreign_entrapment(
            &parameters(),
            &target,
            &[good.clone(), sparse.clone()],
            &directory.join("automatic-reversed.fasta"),
            42,
            1,
            ForeignSourceMode::Automatic,
            SharedPeptideExclusionMode::SageSearchSpace,
            None,
        )
        .unwrap();
        assert_eq!(
            automatic.deterministic_selection_sha256,
            reversed.deterministic_selection_sha256
        );
        assert_eq!(automatic.selected_accessions, reversed.selected_accessions);

        let explicit = generate_foreign_entrapment(
            &parameters(),
            &target,
            &candidates,
            &directory.join("explicit.fasta"),
            42,
            1,
            ForeignSourceMode::Explicit,
            SharedPeptideExclusionMode::SageSearchSpace,
            Some(&sparse),
        )
        .unwrap();
        assert_eq!(explicit.selected_foreign_fasta, sparse);
        assert!(explicit.automatically_recommended_foreign_fasta.is_none());

        let overridden = generate_foreign_entrapment(
            &parameters(),
            &target,
            &candidates,
            &directory.join("overridden.fasta"),
            42,
            1,
            ForeignSourceMode::AutomaticWithOverride,
            SharedPeptideExclusionMode::SageSearchSpace,
            Some(&sparse),
        )
        .unwrap();
        assert_eq!(
            overridden.automatically_recommended_foreign_fasta,
            Some(good)
        );
        assert_eq!(overridden.selected_foreign_fasta, sparse);
        assert!(overridden.override_applied);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn fdrbench_compatibility_includes_below_search_mass_shared_peptides() {
        let directory = test_directory("fdrbench-exclusion-space");
        let target = directory.join("target.fasta");
        let foreign = directory.join("foreign.fasta");
        std::fs::write(&target, b">target\nGGGGGGK\n").unwrap();
        std::fs::write(
            &foreign,
            b">tr|SHARED|SHARED_SPEC\nGGGGGGK\n>tr|UNIQUE|UNIQUE_SPEC\nVVVVVVK\n",
        )
        .unwrap();
        let mut search_parameters = parameters();
        search_parameters.peptide_min_mass = 500.0;

        let sage = generate_foreign_entrapment(
            &search_parameters,
            &target,
            std::slice::from_ref(&foreign),
            &directory.join("sage-space.fasta"),
            7,
            1,
            ForeignSourceMode::Explicit,
            SharedPeptideExclusionMode::SageSearchSpace,
            Some(&foreign),
        )
        .unwrap();
        assert!(sage.excluded_shared_target_peptide.is_empty());

        let compatibility = generate_foreign_entrapment(
            &search_parameters,
            &target,
            std::slice::from_ref(&foreign),
            &directory.join("fdrbench-space.fasta"),
            7,
            1,
            ForeignSourceMode::Explicit,
            SharedPeptideExclusionMode::Fdrbench004Compatibility,
            Some(&foreign),
        )
        .unwrap();
        assert_eq!(
            compatibility.excluded_shared_target_peptide,
            vec!["tr|SHARED|SHARED_SPEC"]
        );
        assert_eq!(
            compatibility.selected_accessions,
            vec!["tr|UNIQUE|UNIQUE_SPEC"]
        );
        assert_eq!(compatibility.measured.target_peptides, 0);
        assert_eq!(compatibility.measured.entrapment_peptides, 1);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn parity_report_normalizes_fdrbench_headers_and_mappings() {
        let directory = test_directory("fdrbench-parity");
        let target = directory.join("target.fasta");
        let foreign = directory.join("foreign.fasta");
        let native_path = directory.join("native.fasta");
        let legacy_path = directory.join("legacy.fasta");
        let legacy_log = directory.join("legacy.log");
        std::fs::write(&target, b">target description\nAAAKCCCCK\n").unwrap();
        std::fs::write(&foreign, b">tr|A|A_SPEC source\nDDDKEEEEK\n").unwrap();
        std::fs::write(
            &legacy_path,
            b">target description\nAAAKCCCCK\n>tr|Ent_A|A_SPEC_p_target source\nDDDKEEEEK\n",
        )
        .unwrap();
        std::fs::write(&legacy_log, b"no exclusions\n").unwrap();
        let native = generate_foreign_entrapment(
            &parameters(),
            &target,
            std::slice::from_ref(&foreign),
            &native_path,
            7,
            1,
            ForeignSourceMode::Explicit,
            SharedPeptideExclusionMode::SageSearchSpace,
            Some(&foreign),
        )
        .unwrap();
        let parity = compare_generated_to_legacy(
            &parameters(),
            &native,
            &LegacyEntrapmentReference {
                fasta: legacy_path,
                foreign_fasta: Some(foreign),
                generation_log: Some(legacy_log),
            },
        )
        .unwrap();
        assert_eq!(parity.selected_foreign_source_match, Some(true));
        assert!(parity.selected_accession_set_match);
        assert!(parity.selected_accession_order_match);
        assert!(parity.mapping_match);
        assert_eq!(parity.excluded_shared_peptide_set_match, Some(true));
        assert!(!parity.exact_fasta_match);
        assert!(!parity.exact_header_order_match);
        assert_eq!(parity.ratios.peptide_ratio_delta, 0.0);
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn scientific_identity(
        parameters: &Parameters,
        target: &Path,
        foreign: &Path,
        seed: u64,
        protein_fold: usize,
        exclusion: SharedPeptideExclusionMode,
    ) -> String {
        let inputs = entrapment_generation_scientific_inputs(
            parameters,
            target,
            &[foreign.to_path_buf()],
            seed,
            protein_fold,
            &ForeignSourceMode::Explicit,
            &exclusion,
            Some(foreign),
        )
        .unwrap();
        entrapment_generation_scientific_input_sha256(&inputs).unwrap()
    }

    #[test]
    fn phase_scoped_identity_tracks_consumed_inputs_only() {
        let directory = test_directory("scientific-identity");
        let target = directory.join("target.fasta");
        let foreign = directory.join("foreign.fasta");
        std::fs::write(&target, b">target\nAAAKCCCCK\n").unwrap();
        std::fs::write(&foreign, b">foreign\nDDDKEEEEK\n").unwrap();
        let base = parameters();
        let identity = scientific_identity(
            &base,
            &target,
            &foreign,
            42,
            1,
            SharedPeptideExclusionMode::SageSearchSpace,
        );

        let mut operational = base.clone();
        operational.fasta = "/another/phase/combined.fasta".into();
        operational.bucket_size = 32768;
        operational.ion_kinds = vec![Kind::A, Kind::C];
        operational.min_ion_index = 7;
        operational.decoy_tag = "different_".into();
        operational.generate_decoys = true;
        operational.prefilter = true;
        operational.prefilter_chunk_size = 999;
        operational.prefilter_low_memory = false;
        assert_eq!(
            identity,
            scientific_identity(
                &operational,
                &target,
                &foreign,
                42,
                1,
                SharedPeptideExclusionMode::SageSearchSpace,
            )
        );

        let mut relevant_variants = Vec::new();
        let mut changed = base.clone();
        changed.enzyme.missed_cleavages = Some(1);
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed.enzyme.min_len = Some(4);
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed.enzyme.max_len = Some(49);
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed.enzyme.cleave_at = Some("K".into());
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed.enzyme.restrict = Some("".into());
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed.enzyme.c_terminal = Some(false);
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed.enzyme.semi_enzymatic = Some(true);
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed.peptide_min_mass = 1.0;
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed.peptide_max_mass = 9999.0;
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed
            .static_mods
            .insert(ModificationSpecificity::Residue(b'C'), 57.0);
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed
            .variable_mods
            .insert(ModificationSpecificity::Residue(b'M'), vec![16.0]);
        relevant_variants.push(changed);
        let mut changed = base.clone();
        changed.max_variable_mods = 3;
        relevant_variants.push(changed);
        for changed in relevant_variants {
            assert_ne!(
                identity,
                scientific_identity(
                    &changed,
                    &target,
                    &foreign,
                    42,
                    1,
                    SharedPeptideExclusionMode::SageSearchSpace,
                )
            );
        }
        assert_ne!(
            identity,
            scientific_identity(
                &base,
                &target,
                &foreign,
                43,
                1,
                SharedPeptideExclusionMode::SageSearchSpace,
            )
        );
        assert_ne!(
            identity,
            scientific_identity(
                &base,
                &target,
                &foreign,
                42,
                2,
                SharedPeptideExclusionMode::SageSearchSpace,
            )
        );
        assert_ne!(
            identity,
            scientific_identity(
                &base,
                &target,
                &foreign,
                42,
                1,
                SharedPeptideExclusionMode::Fdrbench004Compatibility,
            )
        );

        let moved = directory.join("moved");
        std::fs::create_dir(&moved).unwrap();
        let moved_target = moved.join("renamed-target.fasta");
        let moved_foreign = moved.join("renamed-foreign.fasta");
        std::fs::copy(&target, &moved_target).unwrap();
        std::fs::copy(&foreign, &moved_foreign).unwrap();
        assert_eq!(
            identity,
            scientific_identity(
                &base,
                &moved_target,
                &moved_foreign,
                42,
                1,
                SharedPeptideExclusionMode::SageSearchSpace,
            )
        );
        std::fs::write(&moved_foreign, b">foreign\nDDDKEEEER\n").unwrap();
        assert_ne!(
            identity,
            scientific_identity(
                &base,
                &moved_target,
                &moved_foreign,
                42,
                1,
                SharedPeptideExclusionMode::SageSearchSpace,
            )
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn existing_resource_fixture(
        name: &str,
    ) -> (PathBuf, PathBuf, PathBuf, PathBuf, PathBuf, String, String) {
        let directory = test_directory(name);
        let target = directory.join("target.fasta");
        let foreign = directory.join("foreign.fasta");
        let combined = directory.join("combined.fasta");
        let audit_artifact = directory.join("entrapment.audit.json");
        let audit_manifest = directory.join("entrapment.audit.manifest.json");
        let search_config = directory.join("generation.search.json");
        let artifact = directory.join("entrapment.resource.lock.json");
        std::fs::write(&target, b">Target_A\nAAAKCCCCK\n>Target_B\nGGGKLLLLK\n").unwrap();
        std::fs::write(
            &foreign,
            b">Foreign_A\nDDDKEEEEK\n>Foreign_B\nNNNNKQQQQK\n>Foreign_C\nSSSSKTTTTK\n",
        )
        .unwrap();
        let mut generation_parameters = parameters();
        generation_parameters.fasta = target.to_string_lossy().into_owned();
        let generation = generate_foreign_entrapment(
            &generation_parameters,
            &target,
            std::slice::from_ref(&foreign),
            &combined,
            42,
            1,
            ForeignSourceMode::Explicit,
            SharedPeptideExclusionMode::SageSearchSpace,
            Some(&foreign),
        )
        .unwrap();
        write_json_atomic(
            &audit_artifact,
            &EntrapmentAuditReport {
                schema_version: 1,
                database: EntrapmentDatabaseReport::NativeGenerated { generation },
                fasta_parity: None,
            },
        )
        .unwrap();
        write_json_atomic(
            &search_config,
            &serde_json::json!({
                "database": {
                    "bucket_size": 8192,
                    "enzyme": {
                        "missed_cleavages": 0,
                        "min_len": 3,
                        "max_len": 50,
                        "cleave_at": "KR",
                        "restrict": "P",
                        "c_terminal": true,
                        "semi_enzymatic": false
                    },
                    "peptide_min_mass": 0.0,
                    "peptide_max_mass": 10000.0,
                    "ion_kinds": ["b", "y"],
                    "min_ion_index": 2,
                    "static_mods": {},
                    "variable_mods": {},
                    "max_variable_mods": 2,
                    "decoy_tag": "rev_",
                    "generate_decoys": false,
                    "fasta": target,
                    "prefilter_chunk_size": 0,
                    "prefilter": false,
                    "prefilter_low_memory": true
                },
                "precursor_tol": {"ppm": [-20.0, 20.0]},
                "fragment_tol": {"ppm": [-20.0, 20.0]},
                "mzml_paths": []
            }),
        )
        .unwrap();
        write_json_atomic(
            &audit_manifest,
            &EntrapmentAuditManifest {
                schema_version: 1,
                search_config,
                target_fasta: target.clone(),
                output_directory: directory.clone(),
                database_mode: EntrapmentDatabaseMode::NativeGenerated,
                foreign_fastas: vec![foreign.clone()],
                output_fasta: combined.clone(),
                frozen_legacy_fasta: None,
                foreign_source_mode: ForeignSourceMode::Explicit,
                shared_peptide_exclusion_mode: SharedPeptideExclusionMode::SageSearchSpace,
                selected_foreign_fasta: Some(foreign.clone()),
                legacy_parity_reference: None,
                seed: 42,
                protein_fold: 1,
            },
        )
        .unwrap();
        lock_existing_entrapment_resource(&audit_manifest, &audit_artifact, &artifact).unwrap();
        let artifact_sha = sha256_file(&artifact).unwrap();
        let combined_sha = sha256_file(&combined).unwrap();
        (
            directory,
            target,
            foreign,
            combined,
            artifact,
            artifact_sha,
            combined_sha,
        )
    }

    #[test]
    fn existing_sage_entrapment_resource_is_verified_and_relocation_invariant() {
        let (directory, target, foreign, combined, artifact, artifact_sha, combined_sha) =
            existing_resource_fixture("existing-valid");
        let mut active_parameters = parameters();
        active_parameters.fasta = combined.to_string_lossy().into_owned();
        let (report, reference) = load_existing_entrapment_resource(
            &artifact,
            &artifact_sha,
            &combined_sha,
            &active_parameters,
            &target,
            std::slice::from_ref(&foreign),
            &combined,
            42,
            1,
            &ForeignSourceMode::Explicit,
            &SharedPeptideExclusionMode::SageSearchSpace,
            Some(&foreign),
        )
        .unwrap();
        assert!(reference.reused);
        assert_eq!(
            reference.construction_identity,
            entrapment_construction_identity(&report).unwrap()
        );
        let relocated = directory.join("relocated");
        std::fs::create_dir(&relocated).unwrap();
        let relocated_artifact = relocated.join("audit.json");
        let relocated_fasta = relocated.join("combined.fasta");
        std::fs::copy(&artifact, &relocated_artifact).unwrap();
        std::fs::copy(&combined, &relocated_fasta).unwrap();
        let mut relocated_parameters = parameters();
        relocated_parameters.fasta = relocated_fasta.to_string_lossy().into_owned();
        let (_, relocated_reference) = load_existing_entrapment_resource(
            &relocated_artifact,
            &artifact_sha,
            &combined_sha,
            &relocated_parameters,
            &target,
            std::slice::from_ref(&foreign),
            &relocated_fasta,
            42,
            1,
            &ForeignSourceMode::Explicit,
            &SharedPeptideExclusionMode::SageSearchSpace,
            Some(&foreign),
        )
        .unwrap();
        assert_eq!(
            reference.construction_identity,
            relocated_reference.construction_identity
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn existing_sage_entrapment_resource_mismatches_fail_closed() {
        let (directory, target, foreign, combined, artifact, artifact_sha, combined_sha) =
            existing_resource_fixture("existing-mismatch");
        let attempt = |artifact_hash: &str, combined_hash: &str, seed: u64| {
            let mut active_parameters = parameters();
            active_parameters.fasta = combined.to_string_lossy().into_owned();
            load_existing_entrapment_resource(
                &artifact,
                artifact_hash,
                combined_hash,
                &active_parameters,
                &target,
                std::slice::from_ref(&foreign),
                &combined,
                seed,
                1,
                &ForeignSourceMode::Explicit,
                &SharedPeptideExclusionMode::SageSearchSpace,
                Some(&foreign),
            )
        };
        assert!(attempt(&"0".repeat(64), &combined_sha, 42).is_err());
        assert!(attempt(&artifact_sha, &"1".repeat(64), 42).is_err());
        assert!(attempt(&artifact_sha, &combined_sha, 43)
            .unwrap_err()
            .to_string()
            .contains("/seed"));
        let other_target = directory.join("other-target.fasta");
        std::fs::write(&other_target, b">Target_A\nAAAKCCCCR\n").unwrap();
        let mut active_parameters = parameters();
        active_parameters.fasta = combined.to_string_lossy().into_owned();
        assert!(load_existing_entrapment_resource(
            &artifact,
            &artifact_sha,
            &combined_sha,
            &active_parameters,
            &other_target,
            std::slice::from_ref(&foreign),
            &combined,
            42,
            1,
            &ForeignSourceMode::Explicit,
            &SharedPeptideExclusionMode::SageSearchSpace,
            Some(&foreign),
        )
        .is_err());
        let other_foreign = directory.join("other-foreign.fasta");
        std::fs::write(&other_foreign, b">Foreign_A\nDDDKEEEER\n").unwrap();
        assert!(load_existing_entrapment_resource(
            &artifact,
            &artifact_sha,
            &combined_sha,
            &active_parameters,
            &target,
            std::slice::from_ref(&other_foreign),
            &combined,
            42,
            1,
            &ForeignSourceMode::Explicit,
            &SharedPeptideExclusionMode::SageSearchSpace,
            Some(&other_foreign),
        )
        .is_err());
        let other_combined = directory.join("other-combined.fasta");
        std::fs::write(&other_combined, b">Target_A\nAAAKCCCCK\n").unwrap();
        let other_combined_sha = sha256_file(&other_combined).unwrap();
        let mut other_database = parameters();
        other_database.fasta = other_combined.to_string_lossy().into_owned();
        assert!(load_existing_entrapment_resource(
            &artifact,
            &artifact_sha,
            &other_combined_sha,
            &other_database,
            &target,
            std::slice::from_ref(&foreign),
            &other_combined,
            42,
            1,
            &ForeignSourceMode::Explicit,
            &SharedPeptideExclusionMode::SageSearchSpace,
            Some(&foreign),
        )
        .is_err());
        let mut audit: ExistingEntrapmentResourceLock =
            serde_json::from_slice(&std::fs::read(&artifact).unwrap()).unwrap();
        audit.schema_version = 99;
        write_json_atomic(&artifact, &audit).unwrap();
        let changed_sha = sha256_file(&artifact).unwrap();
        assert!(attempt(&changed_sha, &combined_sha, 42).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
