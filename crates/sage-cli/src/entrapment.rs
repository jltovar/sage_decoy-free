use crate::input::Input;
use crate::provenance::{sha256_file, write_json_atomic};
use anyhow::{Context, Result};
use sage_core::database::Parameters;
use sage_core::fasta::Fasta;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct FastaRecord {
    header: String,
    accession: String,
    sequence: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    pub generation_input_sha256: String,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "database_mode", rename_all = "snake_case")]
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
        schema_version: 2,
        generation_input_sha256,
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
}
