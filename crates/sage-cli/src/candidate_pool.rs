use crate::input::Search;
use crate::provenance::{sha256_file, write_json_atomic};
use anyhow::{Context, Result};
use sage_cloudpath::Url;
use sage_core::database::{IndexedDatabase, PeptideIx};
use sage_core::scoring::{ExternalPsmFeatures, FeatureCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub const CANDIDATE_POOL_SCHEMA_VERSION: u32 = 1;
pub const CANDIDATE_ID_SCHEMA: &str = "sage-candidate-id-v1";
const SEARCH_FINGERPRINT_SCHEMA: &str = "sage-search-fingerprint-v1";
const ANALYSIS_FINGERPRINT_SCHEMA: &str = "sage-analysis-fingerprint-v1";

#[derive(Clone)]
struct CachedFileDigest {
    size: u64,
    modified_nanos: Option<u128>,
    sha256: String,
}

static FILE_DIGEST_CACHE: OnceLock<Mutex<HashMap<PathBuf, CachedFileDigest>>> = OnceLock::new();

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpectrumFingerprint {
    pub ordinal: usize,
    pub source: String,
    pub sha256: String,
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
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub(crate) fn content_sha256(path: &Path) -> Result<String> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("resolving fingerprint input {}", path.display()))?;
    let metadata = canonical
        .metadata()
        .with_context(|| format!("reading fingerprint metadata {}", canonical.display()))?;
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos());
    let cache = FILE_DIGEST_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = cache
        .lock()
        .expect("candidate fingerprint cache mutex poisoned")
        .get(&canonical)
        .filter(|cached| cached.size == metadata.len() && cached.modified_nanos == modified_nanos)
        .cloned()
    {
        return Ok(cached.sha256);
    }
    let sha256 = sha256_file(&canonical)?;
    cache
        .lock()
        .expect("candidate fingerprint cache mutex poisoned")
        .insert(
            canonical,
            CachedFileDigest {
                size: metadata.len(),
                modified_nanos,
                sha256: sha256.clone(),
            },
        );
    Ok(sha256)
}

fn source_sha256(url: &Url) -> Result<String> {
    if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("invalid local spectrum URL: {url}"))?;
        return content_sha256(&path);
    }
    let mut hasher = Sha256::new();
    hasher.update(b"unresolved-spectrum-source\0");
    hasher.update(url.as_str().as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
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
    let fasta_sha256 = source_sha256(&fasta_url)
        .with_context(|| format!("hashing candidate-pool FASTA {}", search.database.fasta))?;
    let spectra = search
        .mzml_paths
        .iter()
        .enumerate()
        .map(|(ordinal, source)| {
            Ok(SpectrumFingerprint {
                ordinal,
                source: source.to_string(),
                sha256: source_sha256(source)?,
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
        || manifest.search_fingerprint.digest != expected.digest
        || manifest.search_fingerprint != *expected
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

    let mut ids = HashSet::with_capacity(decoded.records.len());
    let mut entries = Vec::with_capacity(decoded.records.len());
    for mut record in decoded.records {
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
        entries.push(CandidatePoolEntry {
            stable_id: record.stable_id,
            peptide: record.peptide,
            core: record.core,
        });
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
        assert!(inspect_compatible_pool(&directory, &fingerprint, 18)
            .unwrap()
            .is_none());
        std::fs::remove_dir_all(root).unwrap();
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
        std::fs::remove_dir_all(root).unwrap();
    }
}
