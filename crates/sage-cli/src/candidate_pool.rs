use crate::input::Search;
use crate::input_path_identity::{input_path_identity, InputPathIdentity, InputPathKind};
use crate::provenance::{sha256_file, write_json_atomic};
use anyhow::{ensure, Context, Result};
use sage_cloudpath::Url;
use sage_core::database::{IndexedDatabase, PeptideIx};
use sage_core::scoring::{ExternalPsmFeatures, FeatureCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

pub const CANDIDATE_POOL_SCHEMA_VERSION: u32 = 1;
pub const CANDIDATE_ID_SCHEMA: &str = "sage-candidate-id-v1";
const SEARCH_FINGERPRINT_SCHEMA: &str = "sage-search-fingerprint-v1";
const ANALYSIS_FINGERPRINT_SCHEMA: &str = "sage-analysis-fingerprint-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpectrumFingerprint {
    pub ordinal: usize,
    pub source: String,
    pub sha256: String,
    #[serde(default)]
    pub input_kind: InputPathKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory_identity_schema: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regular_file_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchFingerprint {
    pub schema_version: u32,
    pub digest: String,
    pub fasta_sha256: String,
    pub spectra: Vec<SpectrumFingerprint>,
    pub normalized_search_sha256: String,
    pub retained_rank_depth: usize,
    pub candidate_schema: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnalysisFingerprint {
    pub schema_version: u32,
    pub digest: String,
    pub search_fingerprint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidatePoolCapabilities {
    pub retained_rank_depth: usize,
    pub native_rt_predictions: bool,
    pub native_ims_predictions: bool,
    pub matched_fragments: bool,
    pub external_annotations: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidatePoolManifest {
    pub schema_version: u32,
    pub search_fingerprint: SearchFingerprint,
    pub stable_candidate_id_schema: String,
    pub capabilities: CandidatePoolCapabilities,
    pub candidate_count: usize,
    pub spectrum_count: usize,
    pub observed_max_rank: u32,
    pub payload_file: String,
    pub payload_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidatePoolRecord {
    stable_id: String,
    peptide_index: u32,
    peptide: String,
    core: FeatureCore,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidatePoolPayload {
    schema_version: u32,
    search_fingerprint: String,
    records: Vec<CandidatePoolRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidatePoolUsage {
    pub search_fingerprint: String,
    pub analysis_fingerprint: String,
    pub manifest: PathBuf,
    pub payload: PathBuf,
    pub reused: bool,
    pub candidate_count: usize,
    pub retained_rank_depth: usize,
    /// Source locations recorded when the pool was created. These are
    /// provenance only and are deliberately excluded from portable identity.
    #[serde(default)]
    pub original_source_uris: Vec<String>,
    /// Source locations resolved by the current workflow invocation.
    #[serde(default)]
    pub current_source_uris: Vec<String>,
    /// Whether every equality-defining portable search component matched.
    #[serde(default)]
    pub portable_identity_valid: bool,
    /// True when portable identity matched but one or more source URIs moved.
    #[serde(default)]
    pub relocation_detected: bool,
}

/// Verify the durable identity and payload integrity recorded in a completed
/// workflow stage before resuming it without rebuilding the database/search.
pub fn verify_usage(usage: &CandidatePoolUsage) -> Result<()> {
    anyhow::ensure!(
        usage.manifest.is_file(),
        "candidate-pool manifest is missing: {}",
        usage.manifest.display()
    );
    let manifest: CandidatePoolManifest = serde_json::from_slice(&std::fs::read(&usage.manifest)?)
        .with_context(|| {
            format!(
                "invalid candidate-pool manifest {}",
                usage.manifest.display()
            )
        })?;
    anyhow::ensure!(
        manifest.schema_version == CANDIDATE_POOL_SCHEMA_VERSION
            && manifest.stable_candidate_id_schema == CANDIDATE_ID_SCHEMA,
        "candidate-pool schema is incompatible"
    );
    anyhow::ensure!(
        manifest.search_fingerprint.digest == usage.search_fingerprint
            && !usage.analysis_fingerprint.is_empty(),
        "candidate-pool fingerprint does not match workflow usage"
    );
    anyhow::ensure!(
        manifest.candidate_count == usage.candidate_count
            && manifest.capabilities.retained_rank_depth >= usage.retained_rank_depth,
        "candidate-pool capability or count does not match workflow usage"
    );
    let expected_payload = usage
        .manifest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&manifest.payload_file);
    anyhow::ensure!(
        expected_payload == usage.payload && usage.payload.is_file(),
        "candidate-pool payload path is missing or inconsistent"
    );
    anyhow::ensure!(
        sha256_file(&usage.payload)? == manifest.payload_sha256,
        "candidate-pool payload hash mismatch"
    );
    Ok(())
}

/// One immutable search candidate plus the stable key that Phase 4 can use to
/// attach cached MS2Rescore annotations without relying on process-local PSM
/// numbering.
#[derive(Clone, Debug)]
pub struct CandidatePoolEntry {
    pub stable_id: String,
    pub peptide: String,
    pub core: FeatureCore,
}

#[derive(Clone, Debug)]
pub struct CandidatePoolRequest {
    pub root: PathBuf,
    pub required_rank_depth: usize,
    /// Whether an existing compatible pool may satisfy this request. Workflow
    /// stages set this to false when a fresh search is methodologically
    /// required, while still writing the resulting immutable pool for a later
    /// analysis of the same candidate population.
    pub allow_reuse: bool,
    /// Require an exact, already-materialized pool. When true the runner must
    /// never invoke Sage's spectrum-search path to satisfy this request.
    pub require_existing: bool,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub(crate) fn content_sha256(path: &Path) -> Result<String> {
    sha256_file(path)
}

fn source_identity(url: &Url) -> Result<InputPathIdentity> {
    if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("invalid local spectrum URL: {url}"))?;
        return input_path_identity(&path);
    }
    let mut hasher = Sha256::new();
    hasher.update(b"unresolved-spectrum-source\0");
    hasher.update(url.as_str().as_bytes());
    Ok(InputPathIdentity {
        kind: InputPathKind::RemoteSource,
        sha256: format!("{:x}", hasher.finalize()),
        directory_schema: None,
        regular_file_count: None,
        total_bytes: 0,
    })
}

fn normalized_search_value(search: &Search) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(search)?;
    let object = value
        .as_object_mut()
        .context("resolved Sage search did not serialize as an object")?;

    // These settings are downstream analysis/annotation settings. They must
    // invalidate analysis artifacts, but they must not force another spectrum
    // search. Paths are replaced by content hashes below.
    object.remove("fdr");
    object.remove("external_features");
    object.remove("output_paths");
    object.remove("mzml_paths");
    object.remove("protein_grouping");
    object.remove("protein_grouping_peptide_fdr");
    if let Some(database) = object
        .get_mut("database")
        .and_then(serde_json::Value::as_object_mut)
    {
        database.remove("fasta");
    }
    Ok(value)
}

pub fn search_fingerprint(search: &Search) -> Result<SearchFingerprint> {
    let fasta_url = sage_cloudpath::to_url(&search.database.fasta)
        .with_context(|| format!("resolving candidate-pool FASTA {}", search.database.fasta))?;
    let fasta_sha256 = if fasta_url.scheme() == "file" {
        let path = fasta_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("invalid local FASTA URL: {fasta_url}"))?;
        content_sha256(&path)
    } else {
        Ok(source_identity(&fasta_url)?.sha256)
    }
    .with_context(|| format!("hashing candidate-pool FASTA {}", search.database.fasta))?;
    let spectra = search
        .mzml_paths
        .iter()
        .enumerate()
        .map(|(ordinal, source)| {
            let identity = source_identity(source)?;
            Ok(SpectrumFingerprint {
                ordinal,
                source: source.to_string(),
                sha256: identity.sha256,
                input_kind: identity.kind,
                directory_identity_schema: identity.directory_schema,
                regular_file_count: identity.regular_file_count,
                total_bytes: (identity.kind == InputPathKind::Directory)
                    .then_some(identity.total_bytes),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let normalized = serde_json::to_vec(&normalized_search_value(search)?)?;
    let normalized_search_sha256 = sha256_bytes(&normalized);

    let mut hasher = Sha256::new();
    hasher.update(SEARCH_FINGERPRINT_SCHEMA.as_bytes());
    hasher.update(b"\0");
    hasher.update(fasta_sha256.as_bytes());
    for source in &spectra {
        hasher.update(b"\0");
        hasher.update(source.ordinal.to_le_bytes());
        hasher.update(source.sha256.as_bytes());
    }
    hasher.update(b"\0");
    hasher.update(normalized_search_sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(search.report_psms.to_le_bytes());
    hasher.update(b"\0");
    hasher.update(CANDIDATE_ID_SCHEMA.as_bytes());

    Ok(SearchFingerprint {
        schema_version: CANDIDATE_POOL_SCHEMA_VERSION,
        digest: format!("{:x}", hasher.finalize()),
        fasta_sha256,
        spectra,
        normalized_search_sha256,
        retained_rank_depth: search.report_psms,
        candidate_schema: CANDIDATE_ID_SCHEMA.into(),
    })
}

pub fn analysis_fingerprint(
    search: &Search,
    search_fingerprint: &SearchFingerprint,
) -> Result<AnalysisFingerprint> {
    let value = serde_json::json!({
        "fdr": &search.fdr,
        "external_features": &search.external_features,
        "protein_grouping": search.protein_grouping,
        "protein_grouping_peptide_fdr": search.protein_grouping_peptide_fdr,
    });
    let mut hasher = Sha256::new();
    hasher.update(ANALYSIS_FINGERPRINT_SCHEMA.as_bytes());
    hasher.update(b"\0");
    hasher.update(search_fingerprint.digest.as_bytes());
    hasher.update(b"\0");
    hasher.update(serde_json::to_vec(&value)?);
    Ok(AnalysisFingerprint {
        schema_version: CANDIDATE_POOL_SCHEMA_VERSION,
        digest: format!("{:x}", hasher.finalize()),
        search_fingerprint: search_fingerprint.digest.clone(),
    })
}

/// Resolve and validate the complete scientific identity required before the
/// candidate-pool-only boundary is allowed to enter native spectrum search.
pub fn candidate_pool_identity_preflight(
    search: &Search,
    required_rank_depth: usize,
) -> Result<(SearchFingerprint, AnalysisFingerprint)> {
    ensure!(required_rank_depth > 0, "rank depth must be positive");
    let search_fingerprint = search_fingerprint(search)?;
    ensure!(
        required_rank_depth == search_fingerprint.retained_rank_depth,
        "candidate-pool-only --rank-depth={} must exactly equal the frozen search retained depth {}",
        required_rank_depth,
        search_fingerprint.retained_rank_depth
    );
    let analysis_fingerprint = analysis_fingerprint(search, &search_fingerprint)?;
    Ok((search_fingerprint, analysis_fingerprint))
}

pub fn stable_candidate_id(search_fingerprint: &str, core: &FeatureCore, peptide: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CANDIDATE_ID_SCHEMA.as_bytes());
    hasher.update(b"\0");
    hasher.update(search_fingerprint.as_bytes());
    hasher.update(b"\0");
    hasher.update(core.file_id.to_le_bytes());
    hasher.update(b"\0");
    hasher.update(core.spec_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(peptide.as_bytes());
    hasher.update(b"\0");
    hasher.update([core.charge]);
    hasher.update(core.rank.to_le_bytes());
    hasher.update(core.label.to_le_bytes());
    hasher.update(core.expmass.to_bits().to_le_bytes());
    hasher.update(core.isotope_error.to_bits().to_le_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn pool_directory(root: &Path, fingerprint: &SearchFingerprint) -> PathBuf {
    root.join(&fingerprint.digest)
}

pub fn manifest_path(directory: &Path) -> PathBuf {
    directory.join("candidate_pool.json")
}

fn payload_path(directory: &Path) -> PathBuf {
    directory.join("candidate_pool.bin.zst")
}

pub fn inspect_compatible_pool(
    directory: &Path,
    expected: &SearchFingerprint,
    required_rank_depth: usize,
) -> Result<Option<CandidatePoolManifest>> {
    let manifest_path = manifest_path(directory);
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest: CandidatePoolManifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)
        .with_context(|| {
        format!(
            "invalid candidate-pool manifest {}",
            manifest_path.display()
        )
    })?;
    if manifest.schema_version != CANDIDATE_POOL_SCHEMA_VERSION
        || !portable_search_fingerprint_matches(&manifest.search_fingerprint, expected)
        || manifest.capabilities.retained_rank_depth < required_rank_depth
        || manifest.stable_candidate_id_schema != CANDIDATE_ID_SCHEMA
    {
        return Ok(None);
    }
    let payload = directory.join(&manifest.payload_file);
    if !payload.is_file() || sha256_file(&payload)? != manifest.payload_sha256 {
        return Ok(None);
    }
    Ok(Some(manifest))
}

/// Compare the equality-defining, portable search identity. Source URIs are
/// retained in both records for auditability, but content hashes and ordinals
/// establish spectrum identity across filesystem relocation.
pub fn portable_search_fingerprint_matches(
    stored: &SearchFingerprint,
    current: &SearchFingerprint,
) -> bool {
    let required_scalars_present = !stored.digest.is_empty()
        && !stored.fasta_sha256.is_empty()
        && !stored.normalized_search_sha256.is_empty()
        && !stored.candidate_schema.is_empty()
        && !current.digest.is_empty()
        && !current.fasta_sha256.is_empty()
        && !current.normalized_search_sha256.is_empty()
        && !current.candidate_schema.is_empty();
    required_scalars_present
        && stored.schema_version == current.schema_version
        && stored.digest == current.digest
        && stored.fasta_sha256 == current.fasta_sha256
        && stored.normalized_search_sha256 == current.normalized_search_sha256
        && stored.retained_rank_depth == current.retained_rank_depth
        && stored.candidate_schema == current.candidate_schema
        && stored.spectra.len() == current.spectra.len()
        && stored
            .spectra
            .iter()
            .zip(&current.spectra)
            .all(|(left, right)| {
                let directory_metadata_matches = match (left.input_kind, right.input_kind) {
                    (InputPathKind::Directory, InputPathKind::Directory) => {
                        left.directory_identity_schema.is_some()
                            && left.directory_identity_schema == right.directory_identity_schema
                            && left.regular_file_count == right.regular_file_count
                            && left.total_bytes == right.total_bytes
                    }
                    (InputPathKind::RegularFile, InputPathKind::RegularFile) => true,
                    (InputPathKind::RemoteSource, InputPathKind::RemoteSource) => true,
                    _ => false,
                };
                directory_metadata_matches
                    && !left.sha256.is_empty()
                    && !right.sha256.is_empty()
                    && left.ordinal == right.ordinal
                    && left.sha256 == right.sha256
            })
}

pub fn relocation_provenance(
    stored: &SearchFingerprint,
    current: &SearchFingerprint,
) -> (Vec<String>, Vec<String>, bool, bool) {
    let original_source_uris = stored
        .spectra
        .iter()
        .map(|spectrum| spectrum.source.clone())
        .collect::<Vec<_>>();
    let current_source_uris = current
        .spectra
        .iter()
        .map(|spectrum| spectrum.source.clone())
        .collect::<Vec<_>>();
    let portable_identity_valid = portable_search_fingerprint_matches(stored, current);
    let relocation_detected =
        portable_identity_valid && original_source_uris != current_source_uris;
    (
        original_source_uris,
        current_source_uris,
        portable_identity_valid,
        relocation_detected,
    )
}

pub fn write_pool(
    directory: &Path,
    fingerprint: &SearchFingerprint,
    features: &[FeatureCore],
    db: &IndexedDatabase,
) -> Result<CandidatePoolManifest> {
    std::fs::create_dir_all(directory)?;
    let payload = payload_path(directory);
    let temporary = directory.join("candidate_pool.bin.zst.tmp");
    let mut ids = HashSet::with_capacity(features.len());
    let records = features
        .iter()
        .map(|feature| {
            let peptide = db[feature.peptide_idx].to_string();
            let stable_id = stable_candidate_id(&fingerprint.digest, feature, &peptide);
            anyhow::ensure!(
                ids.insert(stable_id.clone()),
                "duplicate stable candidate identifier for {} rank {}",
                feature.spec_id,
                feature.rank
            );
            let mut core = feature.clone();
            core.fragments = None;
            core.external_features = ExternalPsmFeatures::default();
            Ok(CandidatePoolRecord {
                stable_id,
                peptide_index: feature.peptide_idx.0,
                peptide,
                core,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let payload_value = CandidatePoolPayload {
        schema_version: CANDIDATE_POOL_SCHEMA_VERSION,
        search_fingerprint: fingerprint.digest.clone(),
        records,
    };
    {
        let file = File::create(&temporary)?;
        let writer = BufWriter::new(file);
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        bincode::serialize_into(&mut encoder, &payload_value)?;
        encoder.finish()?;
    }
    std::fs::rename(&temporary, &payload)?;

    let observed_max_rank = features
        .iter()
        .map(|feature| feature.rank)
        .max()
        .unwrap_or(0);
    let spectrum_count = features
        .iter()
        .map(|feature| (feature.file_id, feature.spec_id.as_str()))
        .collect::<HashSet<_>>()
        .len();
    let manifest = CandidatePoolManifest {
        schema_version: CANDIDATE_POOL_SCHEMA_VERSION,
        search_fingerprint: fingerprint.clone(),
        stable_candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
        capabilities: CandidatePoolCapabilities {
            retained_rank_depth: fingerprint.retained_rank_depth,
            native_rt_predictions: features.iter().any(|feature| feature.predicted_rt != 0.0),
            native_ims_predictions: features.iter().any(|feature| feature.predicted_ims != 0.0),
            matched_fragments: false,
            external_annotations: false,
        },
        candidate_count: features.len(),
        spectrum_count,
        observed_max_rank,
        payload_file: payload
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("candidate_pool.bin.zst")
            .into(),
        payload_sha256: sha256_file(&payload)?,
    };
    write_json_atomic(&manifest_path(directory), &manifest)?;
    Ok(manifest)
}

/// Publish a complete candidate pool as one directory-level transaction.
///
/// An exact existing pool is verified and reused. An incomplete or
/// incompatible final directory is never overwritten and never triggers a
/// fallback search. New payloads are written and fully reopened under a
/// uniquely named sibling staging directory before the final atomic rename.
pub fn publish_pool_atomic(
    directory: &Path,
    fingerprint: &SearchFingerprint,
    features: &[FeatureCore],
    db: &IndexedDatabase,
) -> Result<(CandidatePoolManifest, bool)> {
    if directory.exists() {
        let (manifest, _) =
            load_pool_entries(directory, fingerprint, fingerprint.retained_rank_depth, db)
                .with_context(|| {
                    format!(
                        "existing final candidate-pool directory is incomplete or incompatible: {}",
                        directory.display()
                    )
                })?;
        return Ok((manifest, true));
    }

    let parent = directory
        .parent()
        .context("candidate-pool directory has no parent")?;
    std::fs::create_dir_all(parent)?;
    let stem = directory
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("candidate-pool");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging = parent.join(format!(".{stem}.partial.{}.{}", std::process::id(), nonce));
    anyhow::ensure!(
        !staging.exists(),
        "candidate-pool staging directory already exists: {}",
        staging.display()
    );

    let result = (|| -> Result<CandidatePoolManifest> {
        let written = write_pool(&staging, fingerprint, features, db)?;
        let (verified, reopened) =
            load_pool_entries(&staging, fingerprint, fingerprint.retained_rank_depth, db)?;
        anyhow::ensure!(
            reopened.len() == features.len()
                && verified.candidate_count == written.candidate_count
                && verified.payload_sha256 == written.payload_sha256,
            "candidate-pool staging verification disagrees with the written population"
        );
        anyhow::ensure!(
            !directory.exists(),
            "candidate-pool final directory appeared during construction; refusing to overwrite {}",
            directory.display()
        );
        std::fs::rename(&staging, directory).with_context(|| {
            format!(
                "atomically publishing candidate pool {} -> {}",
                staging.display(),
                directory.display()
            )
        })?;
        #[cfg(unix)]
        std::fs::File::open(parent)?.sync_all()?;
        let (published, reopened) =
            load_pool_entries(directory, fingerprint, fingerprint.retained_rank_depth, db)?;
        anyhow::ensure!(
            reopened.len() == features.len()
                && published.candidate_count == written.candidate_count
                && published.payload_sha256 == written.payload_sha256,
            "published candidate pool failed immutable reopen verification"
        );
        Ok(published)
    })();

    if result.is_err() && staging.exists() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result.map(|manifest| (manifest, false))
}

pub fn load_pool_entries(
    directory: &Path,
    expected: &SearchFingerprint,
    required_rank_depth: usize,
    db: &IndexedDatabase,
) -> Result<(CandidatePoolManifest, Vec<CandidatePoolEntry>)> {
    let manifest = inspect_compatible_pool(directory, expected, required_rank_depth)?
        .with_context(|| format!("no compatible candidate pool in {}", directory.display()))?;
    let payload = directory.join(&manifest.payload_file);
    let file = File::open(&payload)?;
    let reader = BufReader::new(file);
    let mut decoder = zstd::stream::read::Decoder::new(reader)?;
    let decoded: CandidatePoolPayload = bincode::deserialize_from(&mut decoder)?;
    anyhow::ensure!(
        decoded.schema_version == CANDIDATE_POOL_SCHEMA_VERSION
            && decoded.search_fingerprint == expected.digest,
        "candidate-pool payload identity mismatch"
    );
    anyhow::ensure!(
        decoded.records.len() == manifest.candidate_count,
        "candidate-pool count mismatch: payload={} manifest={}",
        decoded.records.len(),
        manifest.candidate_count
    );
    let decoded_spectrum_count = decoded
        .records
        .iter()
        .map(|record| (record.core.file_id, record.core.spec_id.as_str()))
        .collect::<HashSet<_>>()
        .len();
    anyhow::ensure!(
        decoded_spectrum_count == manifest.spectrum_count,
        "candidate-pool spectrum count mismatch: payload={} manifest={}",
        decoded_spectrum_count,
        manifest.spectrum_count
    );

    let mut ids = HashSet::with_capacity(decoded.records.len());
    let mut entries = Vec::with_capacity(decoded.records.len());
    for mut record in decoded.records {
        anyhow::ensure!(
            expected.spectra.is_empty() || (record.core.file_id as usize) < expected.spectra.len(),
            "candidate-pool file ordinal {} is outside the portable spectrum identity",
            record.core.file_id
        );
        let peptide_idx = PeptideIx(record.peptide_index);
        anyhow::ensure!(
            (record.peptide_index as usize) < db.peptides.len(),
            "candidate-pool peptide index {} is outside rebuilt database",
            record.peptide_index
        );
        anyhow::ensure!(
            db[peptide_idx].to_string() == record.peptide,
            "candidate-pool peptide identity mismatch at index {}",
            record.peptide_index
        );
        record.core.peptide_idx = peptide_idx;
        anyhow::ensure!(
            record.core.rank as usize <= manifest.capabilities.retained_rank_depth,
            "candidate rank {} exceeds pool capability {}",
            record.core.rank,
            manifest.capabilities.retained_rank_depth
        );
        let expected_id = stable_candidate_id(&expected.digest, &record.core, &record.peptide);
        anyhow::ensure!(
            record.stable_id == expected_id,
            "candidate stable-ID mismatch"
        );
        anyhow::ensure!(
            ids.insert(record.stable_id.clone()),
            "duplicate candidate stable-ID"
        );
        // A pool may retain a deeper population for a later capability (for
        // example rank-50 MS2Rescore annotations) than the current analysis
        // needs. Validate every durable record, but materialize only the
        // requested ranks so a rank-25 optimizer does not repeatedly process
        // the unused rank-26..50 population.
        if record.core.rank as usize <= required_rank_depth {
            entries.push(CandidatePoolEntry {
                stable_id: record.stable_id,
                peptide: record.peptide,
                core: record.core,
            });
        }
    }
    Ok((manifest, entries))
}

pub fn load_pool(
    directory: &Path,
    expected: &SearchFingerprint,
    required_rank_depth: usize,
    db: &IndexedDatabase,
) -> Result<(CandidatePoolManifest, Vec<FeatureCore>)> {
    let (manifest, entries) = load_pool_entries(directory, expected, required_rank_depth, db)?;
    Ok((
        manifest,
        entries.into_iter().map(|entry| entry.core).collect(),
    ))
}

/// Strict replay entry point: every incompatibility is an error and callers
/// must not fall back to a search after this function fails.
pub fn load_required_pool(
    directory: &Path,
    expected: &SearchFingerprint,
    required_rank_depth: usize,
    db: &IndexedDatabase,
) -> Result<(CandidatePoolManifest, Vec<FeatureCore>)> {
    inspect_compatible_pool(directory, expected, required_rank_depth)?.with_context(|| {
        format!(
            "required existing candidate pool is missing or incompatible in {} (fingerprint={}, schema={}, retained rank depth >= {})",
            directory.display(), expected.digest, CANDIDATE_ID_SCHEMA, required_rank_depth
        )
    })?;
    load_pool(directory, expected, required_rank_depth, db)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::Input;
    use crate::runner::Runner;
    use sage_core::input::{FdrMode, ModelFit};

    #[test]
    fn stable_id_changes_with_rank_and_analysis_is_not_part_of_identity() {
        let core = FeatureCore {
            file_id: 2,
            spec_id: "controllerType=0 scan=10".into(),
            charge: 3,
            rank: 1,
            label: 1,
            expmass: 1200.0,
            isotope_error: 0.0,
            ..FeatureCore::default()
        };
        let first = stable_candidate_id("search", &core, "PEPTIDE");
        let mut second_core = core.clone();
        second_core.rank = 2;
        let second = stable_candidate_id("search", &second_core, "PEPTIDE");
        assert_ne!(first, second);
        assert_eq!(first, stable_candidate_id("search", &core, "PEPTIDE"));
    }

    #[test]
    fn statistical_settings_change_analysis_but_not_search_identity() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut input = Input::load(
            workspace
                .join("tests/config.json")
                .to_string_lossy()
                .as_ref(),
        )
        .unwrap();
        input.database.fasta = Some(workspace.join("tests/Q99536.fasta").display().to_string());
        input.mzml_paths = Some(vec![workspace
            .join("tests/LQSRPAAPPAPGPGQLTLR.mzML")
            .display()
            .to_string()]);
        let search = input.build().unwrap();
        let search_id = search_fingerprint(&search).unwrap();
        let analysis_id = analysis_fingerprint(&search, &search_id).unwrap();

        let mut changed = search.clone();
        changed.fdr.peptide_fdr = 0.02;
        changed.protein_grouping = !changed.protein_grouping;
        changed.protein_grouping_peptide_fdr = 0.02;
        let changed_search_id = search_fingerprint(&changed).unwrap();
        let changed_analysis_id = analysis_fingerprint(&changed, &changed_search_id).unwrap();
        assert_eq!(search_id, changed_search_id);
        assert_ne!(analysis_id.digest, changed_analysis_id.digest);

        changed.report_psms += 1;
        assert_ne!(
            search_id.digest,
            search_fingerprint(&changed).unwrap().digest
        );
    }

    #[test]
    fn regular_file_strict_search_fingerprint_is_backward_compatible() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut input = Input::load(
            workspace
                .join("tests/config.json")
                .to_string_lossy()
                .as_ref(),
        )
        .unwrap();
        input.database.fasta = Some(workspace.join("tests/Q99536.fasta").display().to_string());
        input.mzml_paths = Some(vec![workspace
            .join("tests/LQSRPAAPPAPGPGQLTLR.mzML")
            .display()
            .to_string()]);
        let search = input.build().unwrap();
        let (fingerprint, analysis) =
            candidate_pool_identity_preflight(&search, search.report_psms).unwrap();
        assert_eq!(
            fingerprint.digest,
            "ccfbc7b99c167bc4f1c13e7f6b7608fd577f6006cd01576652b5c429e6e21e7b"
        );
        assert_eq!(
            fingerprint.spectra[0].input_kind,
            InputPathKind::RegularFile
        );
        assert_eq!(
            fingerprint.spectra[0].sha256,
            "b22b4253ab566878b74c0ade3afc4abd1986edf6b62fafb18545377c4327adb2"
        );
        assert_eq!(analysis.search_fingerprint, fingerprint.digest);
    }

    #[test]
    fn candidate_pool_identity_preflight_accepts_directory_backed_spectrum() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = std::env::temp_dir().join(format!(
            "sage-candidate-directory-preflight-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::write(root.join("analysis.tdf"), b"synthetic vendor data").unwrap();
        std::fs::write(root.join("nested/metadata.bin"), b"metadata").unwrap();
        let mut input = Input::load(
            workspace
                .join("tests/config.json")
                .to_string_lossy()
                .as_ref(),
        )
        .unwrap();
        input.database.fasta = Some(workspace.join("tests/Q99536.fasta").display().to_string());
        input.mzml_paths = Some(vec![root.display().to_string()]);
        let search = input.build().unwrap();
        let (fingerprint, analysis) =
            candidate_pool_identity_preflight(&search, search.report_psms).unwrap();
        let spectrum = &fingerprint.spectra[0];
        assert_eq!(spectrum.input_kind, InputPathKind::Directory);
        assert_eq!(
            spectrum.directory_identity_schema.as_deref(),
            Some(crate::input_path_identity::DIRECTORY_IDENTITY_SCHEMA)
        );
        assert_eq!(spectrum.regular_file_count, Some(2));
        assert_eq!(spectrum.total_bytes, Some(29));
        assert!(!fingerprint.digest.is_empty());
        assert_eq!(analysis.search_fingerprint, fingerprint.digest);
        let core = FeatureCore::default();
        let stable_before = stable_candidate_id(&fingerprint.digest, &core, "PEPTIDE");
        std::fs::write(root.join("analysis.tdf"), b"changed synthetic vendor data").unwrap();
        let changed = search_fingerprint(&search).unwrap();
        assert_ne!(changed.digest, fingerprint.digest);
        assert_ne!(
            stable_candidate_id(&changed.digest, &core, "PEPTIDE"),
            stable_before
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_directory_backed_spectrum_fails_before_search() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let missing = std::env::temp_dir().join(format!(
            "sage-candidate-directory-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut input = Input::load(
            workspace
                .join("tests/config.json")
                .to_string_lossy()
                .as_ref(),
        )
        .unwrap();
        input.database.fasta = Some(workspace.join("tests/Q99536.fasta").display().to_string());
        input.mzml_paths = Some(vec![missing.display().to_string()]);
        let error = input
            .build()
            .and_then(|search| search_fingerprint(&search))
            .unwrap_err();
        assert!(
            error.to_string().contains("failed to stat input path")
                || error.to_string().contains("No such file or directory")
        );
        assert!(!missing.exists());
    }

    #[test]
    fn pool_is_capability_checked_and_payload_verified() {
        let root = std::env::temp_dir().join(format!(
            "sage-candidate-pool-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fingerprint = SearchFingerprint {
            schema_version: CANDIDATE_POOL_SCHEMA_VERSION,
            digest: "search-digest".into(),
            fasta_sha256: "fasta".into(),
            spectra: Vec::new(),
            normalized_search_sha256: "config".into(),
            retained_rank_depth: 50,
            candidate_schema: CANDIDATE_ID_SCHEMA.into(),
        };
        let directory = pool_directory(&root, &fingerprint);
        let manifest =
            write_pool(&directory, &fingerprint, &[], &IndexedDatabase::default()).unwrap();
        let usage = CandidatePoolUsage {
            search_fingerprint: fingerprint.digest.clone(),
            analysis_fingerprint: "analysis-digest".into(),
            manifest: manifest_path(&directory),
            payload: directory.join(&manifest.payload_file),
            reused: true,
            candidate_count: manifest.candidate_count,
            retained_rank_depth: manifest.capabilities.retained_rank_depth,
            original_source_uris: Vec::new(),
            current_source_uris: Vec::new(),
            portable_identity_valid: true,
            relocation_detected: false,
        };
        verify_usage(&usage).unwrap();
        assert!(inspect_compatible_pool(&directory, &fingerprint, 50)
            .unwrap()
            .is_some());
        assert!(inspect_compatible_pool(&directory, &fingerprint, 51)
            .unwrap()
            .is_none());
        let (_, features) =
            load_pool(&directory, &fingerprint, 18, &IndexedDatabase::default()).unwrap();
        assert!(features.is_empty());

        std::fs::write(directory.join(manifest.payload_file), b"corrupt").unwrap();
        assert!(verify_usage(&usage).is_err());
        assert!(inspect_compatible_pool(&directory, &fingerprint, 18)
            .unwrap()
            .is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    fn portable_test_fingerprint(root: &str) -> SearchFingerprint {
        SearchFingerprint {
            schema_version: CANDIDATE_POOL_SCHEMA_VERSION,
            digest: "portable-search".into(),
            fasta_sha256: "fasta-content".into(),
            spectra: vec![
                SpectrumFingerprint {
                    ordinal: 0,
                    source: format!("file://{root}/first.mzML"),
                    sha256: "first-content".into(),
                    input_kind: InputPathKind::RegularFile,
                    directory_identity_schema: None,
                    regular_file_count: None,
                    total_bytes: None,
                },
                SpectrumFingerprint {
                    ordinal: 1,
                    source: format!("file://{root}/second.mzML"),
                    sha256: "second-content".into(),
                    input_kind: InputPathKind::RegularFile,
                    directory_identity_schema: None,
                    regular_file_count: None,
                    total_bytes: None,
                },
            ],
            normalized_search_sha256: "search-configuration".into(),
            retained_rank_depth: 50,
            candidate_schema: CANDIDATE_ID_SCHEMA.into(),
        }
    }

    #[test]
    fn portable_pool_identity_ignores_only_source_relocation() {
        let macos = portable_test_fingerprint("/Users/example/data");
        let mut wsl = portable_test_fingerprint("/mnt/d/data");
        assert!(portable_search_fingerprint_matches(&macos, &wsl));
        let (original, current, valid, relocated) = relocation_provenance(&macos, &wsl);
        assert!(valid);
        assert!(relocated);
        assert_ne!(original, current);

        // Filenames are provenance as well: identical ordered content is the
        // portable identity, not path spelling or the final path component.
        wsl.spectra[0].source = "file:///mnt/e/renamed-one.raw".into();
        wsl.spectra[1].source = "file:///mnt/e/renamed-two.raw".into();
        assert!(portable_search_fingerprint_matches(&macos, &wsl));

        let mut changed = wsl.clone();
        changed.spectra[0].sha256 = "changed-content".into();
        assert!(!portable_search_fingerprint_matches(&macos, &changed));

        let mut reordered = wsl.clone();
        reordered.spectra.swap(0, 1);
        assert!(!portable_search_fingerprint_matches(&macos, &reordered));

        let mut removed = wsl.clone();
        removed.spectra.pop();
        assert!(!portable_search_fingerprint_matches(&macos, &removed));

        let mut added = wsl.clone();
        added.spectra.push(SpectrumFingerprint {
            ordinal: 2,
            source: "file:///mnt/e/third.mzML".into(),
            sha256: "third-content".into(),
            input_kind: InputPathKind::RegularFile,
            directory_identity_schema: None,
            regular_file_count: None,
            total_bytes: None,
        });
        assert!(!portable_search_fingerprint_matches(&macos, &added));

        let mut fasta = wsl.clone();
        fasta.fasta_sha256 = "different-fasta".into();
        assert!(!portable_search_fingerprint_matches(&macos, &fasta));

        let mut search = wsl.clone();
        search.normalized_search_sha256 = "different-search".into();
        assert!(!portable_search_fingerprint_matches(&macos, &search));

        let mut rank = wsl.clone();
        rank.retained_rank_depth = 49;
        assert!(!portable_search_fingerprint_matches(&macos, &rank));

        let mut schema = wsl.clone();
        schema.candidate_schema = "different-candidate-schema".into();
        assert!(!portable_search_fingerprint_matches(&macos, &schema));

        let mut legacy_directory = wsl.clone();
        legacy_directory.spectra[0].input_kind = InputPathKind::Directory;
        legacy_directory.spectra[0].directory_identity_schema = None;
        legacy_directory.spectra[0].regular_file_count = None;
        legacy_directory.spectra[0].total_bytes = None;
        assert!(!portable_search_fingerprint_matches(
            &legacy_directory,
            &legacy_directory
        ));

        let mut missing = wsl;
        missing.spectra[0].sha256.clear();
        assert!(!portable_search_fingerprint_matches(&macos, &missing));
    }

    #[test]
    fn relocated_pool_loads_and_records_provenance_without_changing_identity() {
        let root = std::env::temp_dir().join(format!(
            "sage-candidate-pool-relocation-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let stored = portable_test_fingerprint("/Users/example/data");
        let current = portable_test_fingerprint("/mnt/d/data");
        let directory = pool_directory(&root, &stored);
        let manifest = write_pool(&directory, &stored, &[], &IndexedDatabase::default()).unwrap();
        assert_eq!(stored.digest, current.digest);
        assert!(inspect_compatible_pool(&directory, &current, 50)
            .unwrap()
            .is_some());
        load_pool(&directory, &current, 50, &IndexedDatabase::default()).unwrap();
        let (_, _, valid, relocated) =
            relocation_provenance(&manifest.search_fingerprint, &current);
        assert!(valid);
        assert!(relocated);

        let mut changed_manifest = manifest.clone();
        changed_manifest.candidate_count = 1;
        write_json_atomic(&manifest_path(&directory), &changed_manifest).unwrap();
        assert!(load_pool(&directory, &current, 50, &IndexedDatabase::default()).is_err());

        write_json_atomic(&manifest_path(&directory), &manifest).unwrap();
        std::fs::write(directory.join(&manifest.payload_file), b"corrupt").unwrap();
        assert!(inspect_compatible_pool(&directory, &current, 50)
            .unwrap()
            .is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pool_load_materializes_only_the_requested_rank_depth() {
        use sage_core::peptide::Peptide;

        let root = std::env::temp_dir().join(format!(
            "sage-candidate-pool-rank-depth-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fingerprint = SearchFingerprint {
            schema_version: CANDIDATE_POOL_SCHEMA_VERSION,
            digest: "rank-depth-search".into(),
            fasta_sha256: "fasta".into(),
            spectra: Vec::new(),
            normalized_search_sha256: "config".into(),
            retained_rank_depth: 50,
            candidate_schema: CANDIDATE_ID_SCHEMA.into(),
        };
        let directory = pool_directory(&root, &fingerprint);
        let database = IndexedDatabase {
            peptides: vec![Peptide {
                sequence: std::sync::Arc::from(&b"PEPTIDE"[..]),
                ..Peptide::default()
            }],
            ..IndexedDatabase::default()
        };
        let features = vec![
            FeatureCore {
                spec_id: "scan=1".into(),
                rank: 1,
                peptide_idx: PeptideIx(0),
                ..FeatureCore::default()
            },
            FeatureCore {
                spec_id: "scan=1".into(),
                rank: 20,
                peptide_idx: PeptideIx(0),
                ..FeatureCore::default()
            },
            FeatureCore {
                spec_id: "scan=1".into(),
                rank: 50,
                peptide_idx: PeptideIx(0),
                ..FeatureCore::default()
            },
        ];
        write_pool(&directory, &fingerprint, &features, &database).unwrap();

        let (manifest, through_twenty) =
            load_pool(&directory, &fingerprint, 20, &database).unwrap();
        assert_eq!(manifest.candidate_count, 3);
        assert_eq!(
            through_twenty
                .iter()
                .map(|feature| feature.rank)
                .collect::<Vec<_>>(),
            vec![1, 20]
        );
        let (_, through_fifty) = load_pool(&directory, &fingerprint, 50, &database).unwrap();
        assert_eq!(through_fifty.len(), 3);

        let (_, strict) = load_required_pool(&directory, &fingerprint, 50, &database).unwrap();
        assert_eq!(strict.len(), 3);
        assert!(load_required_pool(&root.join("missing"), &fingerprint, 1, &database).is_err());
        let mut wrong_fingerprint = fingerprint.clone();
        wrong_fingerprint.digest = "different-search".into();
        assert!(load_required_pool(&directory, &wrong_fingerprint, 1, &database).is_err());
        assert!(load_required_pool(&directory, &fingerprint, 51, &database).is_err());

        let payload = directory.join("candidate_pool.bin.zst");
        let original_payload = std::fs::read(&payload).unwrap();
        std::fs::write(&payload, b"corrupt").unwrap();
        assert!(load_required_pool(&directory, &fingerprint, 1, &database).is_err());
        std::fs::write(&payload, original_payload).unwrap();

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_pool_publication_never_accepts_partial_or_incompatible_state() {
        use sage_core::peptide::Peptide;

        let root = std::env::temp_dir().join(format!(
            "sage-candidate-pool-atomic-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let fingerprint = SearchFingerprint {
            schema_version: CANDIDATE_POOL_SCHEMA_VERSION,
            digest: "atomic-search".into(),
            fasta_sha256: "fasta".into(),
            spectra: Vec::new(),
            normalized_search_sha256: "config".into(),
            retained_rank_depth: 1,
            candidate_schema: CANDIDATE_ID_SCHEMA.into(),
        };
        let database = IndexedDatabase {
            peptides: vec![Peptide {
                sequence: std::sync::Arc::from(&b"PEPTIDE"[..]),
                ..Peptide::default()
            }],
            ..IndexedDatabase::default()
        };
        let feature = FeatureCore {
            spec_id: "scan=1".into(),
            rank: 1,
            peptide_idx: PeptideIx(0),
            ..FeatureCore::default()
        };
        let final_directory = pool_directory(&root, &fingerprint);

        let duplicate_error = publish_pool_atomic(
            &final_directory,
            &fingerprint,
            &[feature.clone(), feature.clone()],
            &database,
        )
        .unwrap_err();
        assert!(duplicate_error.to_string().contains("duplicate stable"));
        assert!(!final_directory.exists());

        let (manifest, reused) = publish_pool_atomic(
            &final_directory,
            &fingerprint,
            std::slice::from_ref(&feature),
            &database,
        )
        .unwrap();
        assert!(!reused);
        assert_eq!(manifest.candidate_count, 1);
        let (_, reused) = publish_pool_atomic(
            &final_directory,
            &fingerprint,
            std::slice::from_ref(&feature),
            &database,
        )
        .unwrap();
        assert!(reused);

        std::fs::write(final_directory.join(&manifest.payload_file), b"corrupt").unwrap();
        let error = publish_pool_atomic(
            &final_directory,
            &fingerprint,
            std::slice::from_ref(&feature),
            &database,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("existing final candidate-pool directory"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn candidate_pool_only_stops_before_external_and_statistical_stages() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = std::env::temp_dir().join(format!(
            "sage-candidate-pool-only-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let output = root.join("ordinary-output-must-remain-empty");
        let marker = root.join("external-generator-must-not-run");
        let mut input = Input::load(
            workspace
                .join("tests/config.json")
                .to_string_lossy()
                .as_ref(),
        )
        .unwrap();
        input.database.fasta = Some(workspace.join("tests/Q99536.fasta").display().to_string());
        input.database.generate_decoys = Some(false);
        input.mzml_paths = Some(vec![workspace
            .join("tests/LQSRPAAPPAPGPGQLTLR.mzML")
            .display()
            .to_string()]);
        input.output_directory = Some(output.display().to_string());
        let mut search = input.build().unwrap();
        search.fdr.mode = FdrMode::DecoyFree;
        search.fdr.model_fit = ModelFit::Moments;
        search.external_features.enabled = true;
        search.external_features.command_path = Some(marker.display().to_string());
        let report = Runner::new(search, 1)
            .unwrap()
            .construct_candidate_pool_only(1, root.join("pools"), 10)
            .unwrap();
        assert_eq!(report.execution_scope, "candidate_pool_only");
        assert_eq!(report.status, "verified_complete");
        assert!(report.native_search_performed);
        assert!(report.downstream_stages_entered.is_empty());
        assert!(report.manifest.is_file());
        assert!(report.payload.is_file());
        assert!(!marker.exists());
        assert!(!output.join("results.sage.tsv").exists());
        assert!(!output.join("fitted_model_artifacts.json").exists());
        assert!(!output
            .join("null_window_optimizer.checkpoint.json")
            .exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn raw_cache_only_missing_pool_fails_before_search_or_wrapper_fallback() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = std::env::temp_dir().join(format!(
            "sage-raw-cache-only-missing-pool-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let output = root.join("ordinary-output-must-remain-empty");
        let marker = root.join("external-generator-must-not-run");
        let mut input = Input::load(
            workspace
                .join("tests/config.json")
                .to_string_lossy()
                .as_ref(),
        )
        .unwrap();
        input.database.fasta = Some(workspace.join("tests/Q99536.fasta").display().to_string());
        input.database.generate_decoys = Some(false);
        input.database.prefilter = Some(false);
        input.mzml_paths = Some(vec![workspace
            .join("tests/LQSRPAAPPAPGPGQLTLR.mzML")
            .display()
            .to_string()]);
        input.output_directory = Some(output.display().to_string());
        let mut search = input.build().unwrap();
        search.fdr.mode = FdrMode::DecoyFree;
        search.external_features.enabled = true;
        search.external_features.max_rank = Some(10);
        search.external_features.command_path = Some(marker.display().to_string());
        let error = Runner::new(search, 1)
            .unwrap()
            .construct_raw_annotation_cache_only(
                root.join("missing-pools"),
                root.join("annotations"),
                10,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("search fallback is prohibited"));
        assert!(!marker.exists());
        assert!(!root.join("annotations").exists());
        assert!(!output.join("results.sage.tsv").exists());
        if root.exists() {
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn runner_reuses_search_candidates_across_statistical_analyses() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = std::env::temp_dir().join(format!(
            "sage-candidate-runner-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let build_search = |output: &Path, peptide_fdr: f32| {
            std::fs::create_dir_all(output).unwrap();
            let mut input = Input::load(
                workspace
                    .join("tests/config.json")
                    .to_string_lossy()
                    .as_ref(),
            )
            .unwrap();
            input.database.fasta = Some(workspace.join("tests/Q99536.fasta").display().to_string());
            input.database.generate_decoys = Some(false);
            input.mzml_paths = Some(vec![workspace
                .join("tests/LQSRPAAPPAPGPGQLTLR.mzML")
                .display()
                .to_string()]);
            input.output_directory = Some(output.display().to_string());
            let mut search = input.build().unwrap();
            search.fdr.mode = FdrMode::DecoyFree;
            search.fdr.model_fit = ModelFit::Moments;
            search.fdr.peptide_fdr = peptide_fdr;
            search.fdr.min_null_size = 1;
            search.fdr.moments_min_null_rank = 2;
            search.fdr.moments_max_null_rank = 10;
            search
        };

        let pool_root = root.join("pools");
        let first = Runner::new(build_search(&root.join("first"), 0.01), 1).unwrap();
        let (_, first_usage) = first
            .run_with_candidate_pool(
                1,
                false,
                Some(CandidatePoolRequest {
                    root: pool_root.clone(),
                    required_rank_depth: 1,
                    allow_reuse: true,
                    require_existing: false,
                }),
            )
            .unwrap();
        let first_usage = first_usage.unwrap();
        assert!(!first_usage.reused);

        let second = Runner::new(build_search(&root.join("second"), 0.02), 1).unwrap();
        let (_, second_usage) = second
            .run_with_candidate_pool(
                1,
                false,
                Some(CandidatePoolRequest {
                    root: pool_root,
                    required_rank_depth: 1,
                    allow_reuse: true,
                    require_existing: false,
                }),
            )
            .unwrap();
        let second_usage = second_usage.unwrap();
        assert!(second_usage.reused);
        assert_eq!(
            first_usage.search_fingerprint,
            second_usage.search_fingerprint
        );
        assert_ne!(
            first_usage.analysis_fingerprint,
            second_usage.analysis_fingerprint
        );

        let required = Runner::new(build_search(&root.join("required"), 0.025), 1).unwrap();
        let (_, required_usage) = required
            .run_with_candidate_pool(
                1,
                false,
                Some(CandidatePoolRequest {
                    root: root.join("pools"),
                    required_rank_depth: 1,
                    allow_reuse: true,
                    require_existing: true,
                }),
            )
            .unwrap();
        assert!(required_usage.unwrap().reused);

        let missing = Runner::new(build_search(&root.join("required-missing"), 0.026), 1).unwrap();
        let error = missing
            .run_with_candidate_pool(
                1,
                false,
                Some(CandidatePoolRequest {
                    root: root.join("missing-pools"),
                    required_rank_depth: 1,
                    allow_reuse: true,
                    require_existing: true,
                }),
            )
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("spectrum search fallback is disabled"));

        let forced = Runner::new(build_search(&root.join("forced"), 0.03), 1).unwrap();
        let (_, forced_usage) = forced
            .run_with_candidate_pool(
                1,
                false,
                Some(CandidatePoolRequest {
                    root: root.join("pools"),
                    required_rank_depth: 1,
                    allow_reuse: false,
                    require_existing: false,
                }),
            )
            .unwrap();
        assert!(!forced_usage.unwrap().reused);
        std::fs::remove_dir_all(root).unwrap();
    }
}
