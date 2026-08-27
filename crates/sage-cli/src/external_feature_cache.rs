use crate::candidate_pool::CANDIDATE_ID_SCHEMA;
use crate::input::ExternalFeatureGenerationSettings;
use crate::provenance::{sha256_file, write_json_atomic};
use anyhow::{Context, Result};
use sage_core::scoring::ExternalPsmFeatures;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION: u32 = 2;
pub const EXTERNAL_ANNOTATION_FEATURE_SCHEMA: &str = "sage-external-psm-features-v1";
const EXTERNAL_ANNOTATION_FINGERPRINT_SCHEMA: &str =
    "sage-external-annotation-fingerprint-v2-model-content";
pub const RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION: u32 = 1;
pub const RAW_EXTERNAL_PREDICTION_FEATURE_SCHEMA: &str = "sage-raw-external-prediction-features-v1";
const RAW_EXTERNAL_PREDICTION_FINGERPRINT_SCHEMA: &str =
    "sage-raw-external-prediction-fingerprint-v1";
const STAGE_EXTERNAL_CALIBRATION_FINGERPRINT_SCHEMA: &str =
    "sage-stage-external-calibration-fingerprint-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelComponentIdentity {
    pub generator: String,
    pub logical_model_name: String,
    pub relative_filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct ExternalAnnotationCacheRequest {
    pub root: PathBuf,
    /// Require a complete compatible cache and prohibit every annotation
    /// generation path. This execution control is excluded from cache identity.
    pub require_existing: bool,
    /// Durable search-space label used only in fail-closed provenance.
    pub search_space: String,
    /// Model/stage label used only by the inexpensive derived-calibration
    /// identity. It is deliberately excluded from raw prediction identity.
    pub stage: String,
    /// Stage analysis identity used only by the derived-calibration contract.
    /// It is deliberately excluded from raw prediction identity.
    pub analysis_fingerprint: String,
    /// Permit a write only when it is an exact schema-v2-to-raw migration.
    /// External generation remains prohibited.
    pub migration_only: bool,
}

#[derive(Clone, Debug)]
pub struct ExternalAnnotationInput {
    pub stable_id: String,
    pub score: f64,
    pub q_value: Option<f64>,
    pub pep: Option<f64>,
    pub retention_time: f32,
    pub ion_mobility: f32,
    pub precursor_mass: f32,
    pub charge: u8,
    pub rank: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalAnnotationIdentity {
    pub schema_version: u32,
    pub digest: String,
    pub search_fingerprint: String,
    pub generator_settings_sha256: String,
    pub calibration_input_sha256: String,
    pub stable_candidate_id_schema: String,
    pub feature_schema: String,
    pub requested_candidate_count: usize,
    pub requested_max_rank: u32,
    pub model_components: Vec<ModelComponentIdentity>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalAnnotationCacheManifest {
    pub schema_version: u32,
    pub identity: ExternalAnnotationIdentity,
    pub payload_file: String,
    pub payload_sha256: String,
    pub annotation_count: usize,
    pub joined_annotation_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalAnnotationRecord {
    pub stable_id: String,
    pub features: ExternalPsmFeatures,
}

/// Portable identity of expensive external prediction outputs. Statistical
/// scores, q-values, PEPs, selected windows, and fitted artifacts are absent by
/// construction.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RawExternalPredictionIdentity {
    pub schema_version: u32,
    pub digest: String,
    pub search_fingerprint: String,
    pub generator_settings_sha256: String,
    pub raw_input_sha256: String,
    pub stable_candidate_id_schema: String,
    pub feature_schema: String,
    pub requested_candidate_count: usize,
    pub requested_max_rank: u32,
    pub model_components: Vec<ModelComponentIdentity>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageExternalCalibrationIdentity {
    pub schema_version: u32,
    pub digest: String,
    pub raw_prediction_fingerprint: String,
    pub calibration_input_sha256: String,
    pub stage: String,
    pub analysis_fingerprint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawExternalPredictionCacheManifest {
    pub schema_version: u32,
    pub identity: RawExternalPredictionIdentity,
    pub payload_file: String,
    pub payload_sha256: String,
    pub prediction_count: usize,
    pub joined_prediction_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migrated_from_schema_v2_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migrated_from_schema_v2_payload_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RawExternalPredictionPayload {
    schema_version: u32,
    raw_prediction_identity: RawExternalPredictionIdentity,
    records: Vec<ExternalAnnotationRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ExternalAnnotationPayload {
    schema_version: u32,
    annotation_identity: ExternalAnnotationIdentity,
    records: Vec<ExternalAnnotationRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalAnnotationCacheUsage {
    pub annotation_fingerprint: String,
    pub search_fingerprint: String,
    pub manifest: PathBuf,
    pub payload: PathBuf,
    pub reused: bool,
    pub annotation_count: usize,
    pub joined_annotation_count: usize,
    pub requested_max_rank: u32,
    #[serde(default)]
    pub requested_root: PathBuf,
    #[serde(default)]
    pub generation_allowed: bool,
    #[serde(default)]
    pub preflight_result: String,
    /// Schema-v1 raw prediction cache identity shared across statistical
    /// models and null windows for this candidate population.
    #[serde(default)]
    pub raw_prediction_cache_fingerprint: String,
    #[serde(default)]
    pub raw_prediction_cache_schema_version: u32,
    /// Inexpensive stage-specific calibration identity. This preserves the
    /// separation between raw inference and model/window calibration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_calibration_identity: Option<StageExternalCalibrationIdentity>,
    #[serde(default)]
    pub migrated_from_schema_v2: bool,
}

#[derive(Serialize)]
struct GeneratorSettingsIdentity<'a> {
    schema: &'static str,
    sage_version: &'a str,
    engine: &'a crate::input::ExternalFeatureEngine,
    command: Option<SourceIdentity>,
    python: Option<SourceIdentity>,
    python_environment: Option<String>,
    spectrum_file_mapping: BTreeMap<String, SourceIdentity>,
    feature_only: bool,
    max_rank: Option<u32>,
    feature_generators: Option<serde_json::Value>,
    processes: Option<usize>,
    ms2pip_model: Option<&'a str>,
    ms2pip_ms2_tolerance_bits: Option<u64>,
    deeplc_retrain: Option<bool>,
    deeplc_n_epochs: Option<usize>,
    deeplc_calibration_set_size: Option<usize>,
    modification_mapping: Option<serde_json::Value>,
    fixed_modifications: Option<serde_json::Value>,
    model_components: &'a [ModelComponentIdentity],
}

#[derive(Serialize)]
struct RawGeneratorSettingsIdentity<'a> {
    schema: &'static str,
    sage_version: &'a str,
    engine: &'a crate::input::ExternalFeatureEngine,
    command: Option<PortableSourceIdentity>,
    python: Option<PortableSourceIdentity>,
    python_environment: Option<String>,
    spectrum_file_mapping: BTreeMap<String, PortableSourceIdentity>,
    feature_only: bool,
    max_rank: Option<u32>,
    feature_generators: Option<serde_json::Value>,
    processes: Option<usize>,
    ms2pip_model: Option<&'a str>,
    ms2pip_ms2_tolerance_bits: Option<u64>,
    deeplc_retrain: Option<bool>,
    deeplc_n_epochs: Option<usize>,
    deeplc_calibration_set_size: Option<usize>,
    modification_mapping: Option<serde_json::Value>,
    fixed_modifications: Option<serde_json::Value>,
    model_components: &'a [ModelComponentIdentity],
    raw_candidate_export_contract: &'static str,
    rust_source_sha256: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct SourceIdentity {
    source: String,
    kind: String,
    sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct PortableSourceIdentity {
    kind: String,
    sha256: String,
}

impl From<SourceIdentity> for PortableSourceIdentity {
    fn from(value: SourceIdentity) -> Self {
        Self {
            kind: value.kind,
            sha256: value.sha256,
        }
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.iter().map(canonical_json).collect())
        }
        serde_json::Value::Object(values) => {
            let sorted = values
                .iter()
                .map(|(key, value)| (key.clone(), canonical_json(value)))
                .collect::<BTreeMap<_, _>>();
            serde_json::to_value(sorted).expect("canonical JSON object must serialize")
        }
        other => other.clone(),
    }
}

fn portable_feature_generators(value: &serde_json::Value) -> serde_json::Value {
    let mut value = canonical_json(value);
    if let Some(generators) = value.as_object_mut() {
        if let Some(ms2pip) = generators
            .get_mut("ms2pip")
            .and_then(serde_json::Value::as_object_mut)
        {
            ms2pip.remove("model_dir");
        }
        if let Some(deeplc) = generators
            .get_mut("deeplc")
            .and_then(serde_json::Value::as_object_mut)
        {
            deeplc.remove("path_model");
        }
    }
    value
}

fn directory_sha256(path: &Path) -> Result<String> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("resolving annotation source directory {}", path.display()))?;
    let mut pending = vec![canonical.clone()];
    let mut visited_directories = HashSet::new();
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        let directory = directory.canonicalize().with_context(|| {
            format!(
                "resolving annotation source directory {}",
                directory.display()
            )
        })?;
        if !visited_directories.insert(directory.clone()) {
            continue;
        }
        let mut entries = std::fs::read_dir(&directory)
            .with_context(|| {
                format!(
                    "reading annotation source directory {}",
                    directory.display()
                )
            })?
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file() {
                files.push(entry.path());
            } else if file_type.is_symlink() {
                let target = entry.path().canonicalize()?;
                if target.is_dir() {
                    pending.push(target);
                } else if target.is_file() {
                    files.push(target);
                }
            }
        }
    }
    files.sort();
    let mut hasher = Sha256::new();
    hasher.update(b"sage-annotation-directory-v1\0");
    for file in files {
        let relative = file.strip_prefix(&canonical).unwrap_or(&file);
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        hasher.update(sha256_file(&file)?.as_bytes());
        hasher.update(b"\0");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn source_identity(source: &str) -> Result<SourceIdentity> {
    let path = Path::new(source);
    if path.is_file() {
        return Ok(SourceIdentity {
            source: source.into(),
            kind: "file".into(),
            sha256: sha256_file(path)?,
        });
    }
    if path.is_dir() {
        return Ok(SourceIdentity {
            source: source.into(),
            kind: "directory".into(),
            sha256: directory_sha256(path)?,
        });
    }
    Ok(SourceIdentity {
        source: source.into(),
        kind: "unresolved".into(),
        sha256: sha256_bytes(source.as_bytes()),
    })
}

#[derive(Deserialize)]
struct PythonEnvironmentProbe {
    identity: String,
    metadata_paths: Vec<String>,
}

fn python_environment_identity(
    python: Option<&str>,
) -> Result<(Option<String>, Vec<OperationalFileIdentity>)> {
    let Some(python) = python else {
        return Ok((None, Vec::new()));
    };
    let script = r#"
import importlib.metadata as m
import json, platform
names = ['ms2rescore', 'ms2pip', 'deeplc', 'im2deep', 'psm-utils', 'numpy', 'pandas']
versions = {}
metadata_paths = []
for name in names:
    try:
        dist = m.distribution(name)
        versions[name] = dist.version
        path = getattr(dist, '_path', None)
        if path is not None:
            metadata_paths.append(str(path / 'METADATA'))
    except m.PackageNotFoundError:
        versions[name] = None
identity = {'python': platform.python_version(), 'packages': versions}
print(json.dumps({'identity': json.dumps(identity, sort_keys=True),
                  'metadata_paths': metadata_paths}, sort_keys=True))
"#;
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .output()
        .with_context(|| format!("probing relevant annotation packages with {python}"))?;
    anyhow::ensure!(
        output.status.success(),
        "annotation package-version probe failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let probe: PythonEnvironmentProbe = serde_json::from_slice(&output.stdout)
        .context("annotation package-version probe returned invalid JSON")?;
    let metadata = probe
        .metadata_paths
        .iter()
        .map(|path| operational_file_identity(Path::new(path)))
        .collect::<Result<Vec<_>>>()?;
    Ok((Some(probe.identity), metadata))
}

#[derive(Deserialize)]
struct ModelComponentProbe {
    generator: String,
    logical_model_name: String,
    relative_filename: String,
    path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OperationalFileIdentity {
    path: PathBuf,
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawGeneratorSourceIdentity {
    pub source: String,
    pub kind: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawGeneratorFileIdentity {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawGeneratorProvenance {
    pub schema_version: u32,
    pub generator_settings_sha256: String,
    pub command: Option<RawGeneratorSourceIdentity>,
    pub python: Option<RawGeneratorSourceIdentity>,
    pub python_environment: Option<String>,
    pub package_metadata: Vec<RawGeneratorFileIdentity>,
    pub model_components: Vec<ModelComponentIdentity>,
    pub model_files: Vec<RawGeneratorFileIdentity>,
    pub probe_path: Option<PathBuf>,
    pub probe_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GeneratorProbeCache {
    schema_version: u32,
    probe_key: String,
    python_environment: Option<String>,
    package_metadata: Vec<OperationalFileIdentity>,
    model_components: Vec<ModelComponentIdentity>,
    model_files: Vec<OperationalFileIdentity>,
}

fn operational_file_identity(path: &Path) -> Result<OperationalFileIdentity> {
    let metadata = path
        .metadata()
        .with_context(|| format!("reading annotation identity source {}", path.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "annotation identity source is not a file"
    );
    Ok(OperationalFileIdentity {
        path: path.to_path_buf(),
        size_bytes: metadata.len(),
        sha256: sha256_file(path)?,
    })
}

fn operational_file_is_current(file: &OperationalFileIdentity) -> bool {
    file.path
        .metadata()
        .ok()
        .filter(|metadata| metadata.is_file() && metadata.len() == file.size_bytes)
        .and_then(|_| sha256_file(&file.path).ok())
        .is_some_and(|sha256| sha256 == file.sha256)
}

fn selected_model_components(
    settings: &ExternalFeatureGenerationSettings,
) -> Result<(Vec<ModelComponentIdentity>, Vec<OperationalFileIdentity>)> {
    let Some(python) = settings.python_executable.as_deref() else {
        return Ok((Vec::new(), Vec::new()));
    };
    let generators = settings
        .feature_generators
        .as_ref()
        .and_then(serde_json::Value::as_object);
    if !generators.is_some_and(|g| g.contains_key("ms2pip") || g.contains_key("deeplc")) {
        return Ok((Vec::new(), Vec::new()));
    }
    // Resolve files without loading TensorFlow, running a predictor, downloading
    // weights, or modifying the Python environment.
    let script = r#"
import importlib.metadata as md
import json, pathlib, sys
cfg = json.loads(sys.argv[1])
gens = cfg.get('feature_generators') or {}
out = []
if 'ms2pip' in gens:
    from ms2pip.constants import MODELS
    g = gens.get('ms2pip') or {}
    name = g.get('model') or cfg.get('ms2pip_model') or 'HCD'
    if name not in MODELS:
        raise RuntimeError(f'unknown MS2PIP model {name!r}')
    model_dir = pathlib.Path(g.get('model_dir') or pathlib.Path.home() / '.ms2pip').expanduser()
    for ion, filename in sorted((MODELS[name].get('xgboost_model_files') or {}).items()):
        out.append({'generator':'ms2pip','logical_model_name':f'{name}:{ion}',
                    'relative_filename':filename,'path':str(model_dir / filename)})
if 'deeplc' in gens:
    g = gens.get('deeplc') or {}
    configured = g.get('path_model')
    if configured:
        paths = configured if isinstance(configured, list) else [configured]
        for i, value in enumerate(paths):
            path = pathlib.Path(value).expanduser()
            out.append({'generator':'deeplc','logical_model_name':f'configured:{i}',
                        'relative_filename':path.name,'path':str(path)})
    else:
        dist = md.distribution('deeplc')
        files = sorted(str(p) for p in (dist.files or [])
                       if str(p).replace('\\', '/').startswith('deeplc/mods/')
                       and str(p).endswith('.keras'))
        if g.get('single_model_mode', True):
            files = files[:1]
        for i, relative in enumerate(files):
            path = pathlib.Path(dist.locate_file(relative))
            out.append({'generator':'deeplc','logical_model_name':f'default_candidate:{i}',
                        'relative_filename':relative.replace('\\', '/'), 'path':str(path)})
print(json.dumps(out, sort_keys=True))
"#;
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(serde_json::to_string(settings)?)
        .output()
        .with_context(|| format!("probing selected annotation model files with {python}"))?;
    anyhow::ensure!(
        output.status.success(),
        "annotation model-file probe failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let probes: Vec<ModelComponentProbe> = serde_json::from_slice(&output.stdout)
        .context("annotation model-file probe returned invalid JSON")?;
    let mut components = Vec::with_capacity(probes.len());
    let mut model_files = Vec::with_capacity(probes.len());
    for probe in probes {
        let path = Path::new(&probe.path);
        let file = operational_file_identity(path)
            .with_context(|| format!("selected {} model file is missing", probe.generator))?;
        components.push(ModelComponentIdentity {
            generator: probe.generator,
            logical_model_name: probe.logical_model_name,
            relative_filename: probe.relative_filename,
            size_bytes: file.size_bytes,
            sha256: file.sha256.clone(),
        });
        model_files.push(file);
    }
    components.sort_by(|a, b| {
        (&a.generator, &a.logical_model_name, &a.relative_filename).cmp(&(
            &b.generator,
            &b.logical_model_name,
            &b.relative_filename,
        ))
    });
    Ok((components, model_files))
}

pub fn generator_settings_sha256(settings: &ExternalFeatureGenerationSettings) -> Result<String> {
    Ok(generator_identity(settings, None, false)?.0)
}

/// Resolve the portable generator identity through the same durable probe
/// cache used by annotation generation. Workflow stage hashing calls this on
/// resume so a verified, unchanged cache does not need to launch Python merely
/// to reconstruct provenance that is already durable.
pub fn generator_settings_sha256_with_probe_root(
    settings: &ExternalFeatureGenerationSettings,
    probe_root: &Path,
) -> Result<String> {
    Ok(generator_identity(settings, Some(probe_root), false)?.0)
}

/// Resolve the durable generator identity without probing Python or writing a
/// probe record. Strict cache-only workflows use this before stage execution.
pub fn generator_settings_sha256_with_existing_probe_root(
    settings: &ExternalFeatureGenerationSettings,
    probe_root: &Path,
) -> Result<String> {
    Ok(generator_identity(settings, Some(probe_root), true)?.0)
}

pub fn raw_generator_settings_sha256_with_probe_root(
    settings: &ExternalFeatureGenerationSettings,
    probe_root: &Path,
) -> Result<String> {
    Ok(raw_generator_identity(settings, Some(probe_root), false)?.0)
}

pub fn raw_generator_settings_sha256_with_existing_probe_root(
    settings: &ExternalFeatureGenerationSettings,
    probe_root: &Path,
) -> Result<String> {
    Ok(raw_generator_identity(settings, Some(probe_root), true)?.0)
}

fn generator_identity(
    settings: &ExternalFeatureGenerationSettings,
    probe_root: Option<&Path>,
    require_existing_probe: bool,
) -> Result<(String, Vec<ModelComponentIdentity>)> {
    let probe = resolve_generator_probe(settings, probe_root, require_existing_probe)?;
    let digest = generator_identity_digest(
        settings,
        probe.python_environment.clone(),
        &probe.model_components,
    )?;
    Ok((digest, probe.model_components))
}

fn resolve_generator_probe(
    settings: &ExternalFeatureGenerationSettings,
    probe_root: Option<&Path>,
    require_existing_probe: bool,
) -> Result<GeneratorProbeCache> {
    let probe_key = generator_probe_key(settings)?;
    let probe_path = probe_root.map(|root| {
        root.join("generator_identity_probes")
            .join(format!("{probe_key}.json"))
    });
    let cached = probe_path
        .as_ref()
        .filter(|path| path.is_file())
        .and_then(|path| std::fs::read(path).ok())
        .and_then(|bytes| serde_json::from_slice::<GeneratorProbeCache>(&bytes).ok())
        .filter(|probe| {
            probe.schema_version == 2
                && probe.probe_key == probe_key
                && probe
                    .package_metadata
                    .iter()
                    .all(operational_file_is_current)
                && probe.model_files.iter().all(operational_file_is_current)
                && probe.model_files.len() == probe.model_components.len()
        });
    let probe = if let Some(probe) = cached {
        probe
    } else if require_existing_probe {
        anyhow::bail!(
            "required annotation cache has no valid durable package/model identity probe under {}; Python/model resolution and annotation generation are prohibited",
            probe_root
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<no probe root>".into())
        );
    } else {
        let (python_environment, package_metadata) =
            python_environment_identity(settings.python_executable.as_deref())?;
        let (model_components, model_files) = selected_model_components(settings)?;
        let probe = GeneratorProbeCache {
            schema_version: 2,
            probe_key,
            python_environment,
            package_metadata,
            model_components,
            model_files,
        };
        if let Some(path) = probe_path.as_ref() {
            std::fs::create_dir_all(path.parent().expect("probe path has a parent"))?;
            write_json_atomic(path, &probe)?;
        }
        probe
    };
    Ok(probe)
}

fn raw_generator_identity(
    settings: &ExternalFeatureGenerationSettings,
    probe_root: Option<&Path>,
    require_existing_probe: bool,
) -> Result<(String, Vec<ModelComponentIdentity>)> {
    validate_model_independent_generator_contract(settings)?;
    let probe = resolve_generator_probe(settings, probe_root, require_existing_probe)?;
    let command = settings
        .command_path
        .as_deref()
        .map(source_identity)
        .transpose()?
        .map(PortableSourceIdentity::from);
    let python = settings
        .python_executable
        .as_deref()
        .map(source_identity)
        .transpose()?
        .map(PortableSourceIdentity::from);
    let spectrum_file_mapping = settings
        .spectrum_file_mapping
        .iter()
        .map(|(name, source)| {
            Ok((
                name.clone(),
                PortableSourceIdentity::from(source_identity(source)?),
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let value = RawGeneratorSettingsIdentity {
        schema: RAW_EXTERNAL_PREDICTION_FINGERPRINT_SCHEMA,
        sage_version: env!("CARGO_PKG_VERSION"),
        engine: &settings.engine,
        command,
        python,
        python_environment: probe.python_environment.clone(),
        spectrum_file_mapping,
        feature_only: settings.feature_only,
        max_rank: settings.max_rank,
        feature_generators: settings
            .feature_generators
            .as_ref()
            .map(portable_feature_generators),
        processes: settings.processes,
        ms2pip_model: settings.ms2pip_model.as_deref(),
        ms2pip_ms2_tolerance_bits: settings.ms2pip_ms2_tolerance.map(f64::to_bits),
        deeplc_retrain: settings.deeplc_retrain,
        deeplc_n_epochs: settings.deeplc_n_epochs,
        deeplc_calibration_set_size: settings.deeplc_calibration_set_size,
        modification_mapping: settings.modification_mapping.as_ref().map(canonical_json),
        fixed_modifications: settings.fixed_modifications.as_ref().map(canonical_json),
        model_components: &probe.model_components,
        raw_candidate_export_contract: "sage-external-raw-candidate-export-v1-neutral-statistics",
        rust_source_sha256: env!("SAGE_EXTERNAL_CACHE_SOURCE_SHA256"),
    };
    Ok((
        sha256_bytes(&serde_json::to_vec(&value)?),
        probe.model_components,
    ))
}

/// Resolve the complete durable provenance consumed by raw-cache-only
/// construction. The returned record makes wrapper, Python environment,
/// package metadata, and selected model-file identities independently
/// inspectable instead of leaving them only inside a composite digest.
pub fn raw_generator_provenance(
    settings: &ExternalFeatureGenerationSettings,
    probe_root: &Path,
    require_existing_probe: bool,
) -> Result<RawGeneratorProvenance> {
    let (generator_settings_sha256, model_components) =
        raw_generator_identity(settings, Some(probe_root), require_existing_probe)?;
    let probe = resolve_generator_probe(settings, Some(probe_root), true)?;
    anyhow::ensure!(
        probe.model_components == model_components,
        "raw generator model identities changed while resolving durable provenance"
    );
    let command = settings
        .command_path
        .as_deref()
        .map(source_identity)
        .transpose()?
        .map(|identity| RawGeneratorSourceIdentity {
            source: identity.source,
            kind: identity.kind,
            sha256: identity.sha256,
        });
    let python = settings
        .python_executable
        .as_deref()
        .map(source_identity)
        .transpose()?
        .map(|identity| RawGeneratorSourceIdentity {
            source: identity.source,
            kind: identity.kind,
            sha256: identity.sha256,
        });
    let probe_path = Some(
        probe_root
            .join("generator_identity_probes")
            .join(format!("{}.json", probe.probe_key)),
    );
    let probe_sha256 = probe_path
        .as_ref()
        .filter(|path| path.is_file())
        .map(|path| sha256_file(path))
        .transpose()?;
    let map_file = |file: &OperationalFileIdentity| RawGeneratorFileIdentity {
        path: file.path.clone(),
        size_bytes: file.size_bytes,
        sha256: file.sha256.clone(),
    };
    Ok(RawGeneratorProvenance {
        schema_version: 1,
        generator_settings_sha256,
        command,
        python,
        python_environment: probe.python_environment,
        package_metadata: probe.package_metadata.iter().map(map_file).collect(),
        model_components,
        model_files: probe.model_files.iter().map(map_file).collect(),
        probe_path,
        probe_sha256,
    })
}

fn validate_model_independent_generator_contract(
    settings: &ExternalFeatureGenerationSettings,
) -> Result<()> {
    let deeplc_calibration_set_size = match settings.feature_generators.as_ref() {
        Some(generators) => generators.get("deeplc").map(|deeplc| {
            deeplc
                .get("calibration_set_size")
                .and_then(serde_json::Value::as_u64)
        }),
        None => Some(settings.deeplc_calibration_set_size.map(|size| size as u64)),
    };

    if let Some(calibration_set_size) = deeplc_calibration_set_size {
        anyhow::ensure!(
            calibration_set_size.is_some_and(|size| size > 0),
            "layered raw prediction caching requires DeepLC calibration_set_size to be an explicit positive integer; implicit q-value-based calibration selection is stage dependent"
        );
    }
    Ok(())
}

fn generator_probe_key(settings: &ExternalFeatureGenerationSettings) -> Result<String> {
    let mut operational = canonical_json(&serde_json::to_value(settings)?);
    if let Some(object) = operational.as_object_mut() {
        for ignored in [
            "enabled",
            "temp_directory",
            "output_directory",
            "fail_policy",
            "use_mode",
            "log_level",
        ] {
            object.remove(ignored);
        }
        object.remove("command_path");
        object.remove("python_executable");
        object.remove("spectrum_file_mapping");
        if let Some(generators) = object.get_mut("feature_generators") {
            *generators = portable_feature_generators(generators);
        }
    }
    let spectrum_file_mapping = settings
        .spectrum_file_mapping
        .iter()
        .map(|(name, source)| {
            Ok((
                name.clone(),
                PortableSourceIdentity::from(source_identity(source)?),
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let value = serde_json::json!({
        "schema": "sage-generator-probe-key-v2-portable",
        "settings": operational,
        "command": settings.command_path.as_deref().map(source_identity).transpose()?.map(PortableSourceIdentity::from),
        "python": settings.python_executable.as_deref().map(source_identity).transpose()?.map(PortableSourceIdentity::from),
        "spectrum_file_mapping": spectrum_file_mapping,
    });
    Ok(sha256_bytes(&serde_json::to_vec(&canonical_json(&value))?))
}

fn generator_identity_digest(
    settings: &ExternalFeatureGenerationSettings,
    python_environment: Option<String>,
    model_components: &[ModelComponentIdentity],
) -> Result<String> {
    let command = settings
        .command_path
        .as_deref()
        .map(source_identity)
        .transpose()?;
    let python = settings
        .python_executable
        .as_deref()
        .map(source_identity)
        .transpose()?;
    let spectrum_file_mapping = settings
        .spectrum_file_mapping
        .iter()
        .map(|(name, source)| Ok((name.clone(), source_identity(source)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let value = GeneratorSettingsIdentity {
        schema: EXTERNAL_ANNOTATION_FINGERPRINT_SCHEMA,
        sage_version: env!("CARGO_PKG_VERSION"),
        engine: &settings.engine,
        command,
        python,
        python_environment,
        spectrum_file_mapping,
        feature_only: settings.feature_only,
        max_rank: settings.max_rank,
        feature_generators: settings
            .feature_generators
            .as_ref()
            .map(portable_feature_generators),
        processes: settings.processes,
        ms2pip_model: settings.ms2pip_model.as_deref(),
        ms2pip_ms2_tolerance_bits: settings.ms2pip_ms2_tolerance.map(f64::to_bits),
        deeplc_retrain: settings.deeplc_retrain,
        deeplc_n_epochs: settings.deeplc_n_epochs,
        deeplc_calibration_set_size: settings.deeplc_calibration_set_size,
        modification_mapping: settings.modification_mapping.as_ref().map(canonical_json),
        fixed_modifications: settings.fixed_modifications.as_ref().map(canonical_json),
        model_components,
    };
    Ok(sha256_bytes(&serde_json::to_vec(&value)?))
}

fn calibration_input_sha256(inputs: &[ExternalAnnotationInput]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-external-annotation-input-v1\0");
    for input in inputs {
        hasher.update(input.stable_id.as_bytes());
        hasher.update(b"\0");
        hasher.update(input.score.to_bits().to_le_bytes());
        hasher.update(
            input
                .q_value
                .map(f64::to_bits)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        hasher.update(
            input
                .pep
                .map(f64::to_bits)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        hasher.update(input.retention_time.to_bits().to_le_bytes());
        hasher.update(input.ion_mobility.to_bits().to_le_bytes());
        hasher.update(input.precursor_mass.to_bits().to_le_bytes());
        hasher.update([input.charge]);
        hasher.update(input.rank.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn sorted_inputs(inputs: &[ExternalAnnotationInput]) -> Result<Vec<&ExternalAnnotationInput>> {
    let mut sorted = inputs.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.stable_id.cmp(&right.stable_id));
    anyhow::ensure!(
        sorted
            .windows(2)
            .all(|pair| pair[0].stable_id != pair[1].stable_id),
        "duplicate stable candidate ID in external prediction inputs"
    );
    Ok(sorted)
}

fn raw_candidate_id_sha256<'a>(ids: impl Iterator<Item = &'a str>) -> Result<String> {
    let mut ids = ids.collect::<Vec<_>>();
    ids.sort_unstable();
    anyhow::ensure!(
        ids.windows(2).all(|pair| pair[0] != pair[1]),
        "duplicate stable candidate ID in external prediction inputs"
    );
    let mut hasher = Sha256::new();
    hasher.update(b"sage-external-raw-input-v1\0");
    for stable_id in ids {
        hasher.update(stable_id.as_bytes());
        hasher.update(b"\0");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn ordered_calibration_input_sha256(inputs: &[ExternalAnnotationInput]) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-external-stage-calibration-input-v1\0");
    for input in sorted_inputs(inputs)? {
        hasher.update(input.stable_id.as_bytes());
        hasher.update(b"\0");
        hasher.update(input.score.to_bits().to_le_bytes());
        hasher.update(
            input
                .q_value
                .map(f64::to_bits)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        hasher.update(
            input
                .pep
                .map(f64::to_bits)
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        hasher.update(input.retention_time.to_bits().to_le_bytes());
        hasher.update(input.ion_mobility.to_bits().to_le_bytes());
        hasher.update(input.precursor_mass.to_bits().to_le_bytes());
        hasher.update([input.charge]);
        hasher.update(input.rank.to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn raw_prediction_identity_with_probe_root(
    search_fingerprint: &str,
    settings: &ExternalFeatureGenerationSettings,
    inputs: &[ExternalAnnotationInput],
    requested_max_rank: u32,
    probe_root: &Path,
    require_existing_probe: bool,
) -> Result<RawExternalPredictionIdentity> {
    build_raw_prediction_identity(
        search_fingerprint,
        settings,
        inputs.iter().map(|input| input.stable_id.as_str()),
        inputs.len(),
        requested_max_rank,
        probe_root,
        require_existing_probe,
    )
}

pub fn raw_prediction_identity_from_candidate_ids_with_probe_root(
    search_fingerprint: &str,
    settings: &ExternalFeatureGenerationSettings,
    candidate_ids: &HashSet<String>,
    requested_max_rank: u32,
    probe_root: &Path,
    require_existing_probe: bool,
) -> Result<RawExternalPredictionIdentity> {
    build_raw_prediction_identity(
        search_fingerprint,
        settings,
        candidate_ids.iter().map(String::as_str),
        candidate_ids.len(),
        requested_max_rank,
        probe_root,
        require_existing_probe,
    )
}

fn build_raw_prediction_identity<'a>(
    search_fingerprint: &str,
    settings: &ExternalFeatureGenerationSettings,
    candidate_ids: impl Iterator<Item = &'a str>,
    candidate_count: usize,
    requested_max_rank: u32,
    probe_root: &Path,
    require_existing_probe: bool,
) -> Result<RawExternalPredictionIdentity> {
    let (generator_settings_sha256, model_components) =
        raw_generator_identity(settings, Some(probe_root), require_existing_probe)?;
    let raw_input_sha256 = raw_candidate_id_sha256(candidate_ids)?;
    let mut hasher = Sha256::new();
    for part in [
        RAW_EXTERNAL_PREDICTION_FINGERPRINT_SCHEMA,
        search_fingerprint,
        generator_settings_sha256.as_str(),
        raw_input_sha256.as_str(),
        CANDIDATE_ID_SCHEMA,
        RAW_EXTERNAL_PREDICTION_FEATURE_SCHEMA,
    ] {
        hasher.update(part.as_bytes());
        hasher.update(b"\0");
    }
    hasher.update(requested_max_rank.to_le_bytes());
    hasher.update((candidate_count as u64).to_le_bytes());
    Ok(RawExternalPredictionIdentity {
        schema_version: RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION,
        digest: format!("{:x}", hasher.finalize()),
        search_fingerprint: search_fingerprint.into(),
        generator_settings_sha256,
        raw_input_sha256,
        stable_candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
        feature_schema: RAW_EXTERNAL_PREDICTION_FEATURE_SCHEMA.into(),
        requested_candidate_count: candidate_count,
        requested_max_rank,
        model_components,
    })
}

pub fn stage_calibration_identity(
    raw: &RawExternalPredictionIdentity,
    inputs: &[ExternalAnnotationInput],
    request: &ExternalAnnotationCacheRequest,
) -> Result<StageExternalCalibrationIdentity> {
    anyhow::ensure!(
        !request.stage.is_empty() && !request.analysis_fingerprint.is_empty(),
        "external stage calibration requires explicit stage and analysis provenance"
    );
    let calibration_input_sha256 = ordered_calibration_input_sha256(inputs)?;
    let mut hasher = Sha256::new();
    for part in [
        STAGE_EXTERNAL_CALIBRATION_FINGERPRINT_SCHEMA,
        raw.digest.as_str(),
        calibration_input_sha256.as_str(),
        request.stage.as_str(),
        request.analysis_fingerprint.as_str(),
    ] {
        hasher.update(part.as_bytes());
        hasher.update(b"\0");
    }
    Ok(StageExternalCalibrationIdentity {
        schema_version: 1,
        digest: format!("{:x}", hasher.finalize()),
        raw_prediction_fingerprint: raw.digest.clone(),
        calibration_input_sha256,
        stage: request.stage.clone(),
        analysis_fingerprint: request.analysis_fingerprint.clone(),
    })
}

pub fn annotation_identity(
    search_fingerprint: &str,
    settings: &ExternalFeatureGenerationSettings,
    inputs: &[ExternalAnnotationInput],
    requested_max_rank: u32,
) -> Result<ExternalAnnotationIdentity> {
    annotation_identity_with_probe_root(
        search_fingerprint,
        settings,
        inputs,
        requested_max_rank,
        None,
    )
}

pub fn annotation_identity_with_probe_root(
    search_fingerprint: &str,
    settings: &ExternalFeatureGenerationSettings,
    inputs: &[ExternalAnnotationInput],
    requested_max_rank: u32,
    probe_root: Option<&Path>,
) -> Result<ExternalAnnotationIdentity> {
    let (generator_settings_sha256, model_components) =
        generator_identity(settings, probe_root, false)?;
    build_annotation_identity(
        search_fingerprint,
        inputs,
        requested_max_rank,
        generator_settings_sha256,
        model_components,
    )
}

/// Resolve an annotation identity only from an existing durable generator
/// probe. This path performs no Python or model-resolution invocation and no
/// writes, so strict cache replay can fail before generation is possible.
pub fn annotation_identity_with_existing_probe_root(
    search_fingerprint: &str,
    settings: &ExternalFeatureGenerationSettings,
    inputs: &[ExternalAnnotationInput],
    requested_max_rank: u32,
    probe_root: &Path,
) -> Result<ExternalAnnotationIdentity> {
    let (generator_settings_sha256, model_components) =
        generator_identity(settings, Some(probe_root), true)?;
    build_annotation_identity(
        search_fingerprint,
        inputs,
        requested_max_rank,
        generator_settings_sha256,
        model_components,
    )
}

fn build_annotation_identity(
    search_fingerprint: &str,
    inputs: &[ExternalAnnotationInput],
    requested_max_rank: u32,
    generator_settings_sha256: String,
    model_components: Vec<ModelComponentIdentity>,
) -> Result<ExternalAnnotationIdentity> {
    let calibration_input_sha256 = calibration_input_sha256(inputs);
    let mut hasher = Sha256::new();
    hasher.update(EXTERNAL_ANNOTATION_FINGERPRINT_SCHEMA.as_bytes());
    hasher.update(b"\0");
    hasher.update(search_fingerprint.as_bytes());
    hasher.update(b"\0");
    hasher.update(generator_settings_sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(calibration_input_sha256.as_bytes());
    hasher.update(b"\0");
    hasher.update(CANDIDATE_ID_SCHEMA.as_bytes());
    hasher.update(b"\0");
    hasher.update(EXTERNAL_ANNOTATION_FEATURE_SCHEMA.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    Ok(ExternalAnnotationIdentity {
        schema_version: EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION,
        digest,
        search_fingerprint: search_fingerprint.into(),
        generator_settings_sha256,
        calibration_input_sha256,
        stable_candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
        feature_schema: EXTERNAL_ANNOTATION_FEATURE_SCHEMA.into(),
        requested_candidate_count: inputs.len(),
        requested_max_rank,
        model_components,
    })
}

pub fn cache_directory(root: &Path, identity: &ExternalAnnotationIdentity) -> PathBuf {
    root.join(&identity.digest)
}

pub fn cache_manifest_path(directory: &Path) -> PathBuf {
    directory.join("ms2rescore_annotations.json")
}

fn cache_payload_path(directory: &Path) -> PathBuf {
    directory.join("ms2rescore_annotations.bin.zst")
}

pub fn load_cache(
    directory: &Path,
    expected: &ExternalAnnotationIdentity,
) -> Result<
    Option<(
        ExternalAnnotationCacheManifest,
        Vec<ExternalAnnotationRecord>,
    )>,
> {
    let manifest_path = cache_manifest_path(directory);
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest: ExternalAnnotationCacheManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path)?).with_context(|| {
            format!(
                "invalid annotation cache manifest {}",
                manifest_path.display()
            )
        })?;
    anyhow::ensure!(
        manifest.schema_version == EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION
            && manifest.identity == *expected,
        "MS2Rescore annotation cache identity mismatch in {}",
        manifest_path.display()
    );
    let payload_path = directory.join(&manifest.payload_file);
    anyhow::ensure!(
        payload_path.is_file(),
        "MS2Rescore annotation cache payload is missing"
    );
    anyhow::ensure!(
        sha256_file(&payload_path)? == manifest.payload_sha256,
        "MS2Rescore annotation cache payload hash mismatch"
    );
    let file = File::open(&payload_path)?;
    let reader = BufReader::new(file);
    let mut decoder = zstd::stream::read::Decoder::new(reader)?;
    let payload: ExternalAnnotationPayload = bincode::deserialize_from(&mut decoder)?;
    anyhow::ensure!(
        payload.schema_version == EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION
            && payload.annotation_identity == *expected
            && payload.annotation_identity == manifest.identity,
        "MS2Rescore annotation cache payload identity mismatch"
    );
    anyhow::ensure!(
        payload.records.len() == manifest.annotation_count
            && manifest.annotation_count == expected.requested_candidate_count,
        "MS2Rescore annotation cache count mismatch"
    );
    let mut ids = HashSet::with_capacity(payload.records.len());
    for record in &payload.records {
        anyhow::ensure!(
            ids.insert(record.stable_id.as_str()),
            "duplicate stable candidate ID in MS2Rescore annotation cache"
        );
    }
    let joined = payload
        .records
        .iter()
        .filter(|record| record.features.ms2rescore_feature_joined)
        .count();
    anyhow::ensure!(
        joined == manifest.joined_annotation_count,
        "MS2Rescore annotation cache joined-count mismatch"
    );
    Ok(Some((manifest, payload.records)))
}

pub fn write_cache(
    directory: &Path,
    identity: &ExternalAnnotationIdentity,
    records: Vec<ExternalAnnotationRecord>,
) -> Result<ExternalAnnotationCacheManifest> {
    anyhow::ensure!(
        records.len() == identity.requested_candidate_count,
        "refusing to cache a partial MS2Rescore annotation set"
    );
    let mut ids = HashSet::with_capacity(records.len());
    for record in &records {
        anyhow::ensure!(
            ids.insert(record.stable_id.as_str()),
            "duplicate stable candidate ID while writing MS2Rescore annotation cache"
        );
    }
    std::fs::create_dir_all(directory)?;
    let payload_path = cache_payload_path(directory);
    let temporary = directory.join("ms2rescore_annotations.bin.zst.tmp");
    let payload = ExternalAnnotationPayload {
        schema_version: EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION,
        annotation_identity: identity.clone(),
        records,
    };
    {
        let file = File::create(&temporary)?;
        let writer = BufWriter::new(file);
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        bincode::serialize_into(&mut encoder, &payload)?;
        encoder.finish()?;
    }
    std::fs::rename(&temporary, &payload_path)?;
    let joined_annotation_count = payload
        .records
        .iter()
        .filter(|record| record.features.ms2rescore_feature_joined)
        .count();
    let manifest = ExternalAnnotationCacheManifest {
        schema_version: EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION,
        identity: identity.clone(),
        payload_file: payload_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("ms2rescore_annotations.bin.zst")
            .into(),
        payload_sha256: sha256_file(&payload_path)?,
        annotation_count: payload.records.len(),
        joined_annotation_count,
    };
    write_json_atomic(&cache_manifest_path(directory), &manifest)?;
    Ok(manifest)
}

pub fn raw_cache_directory(root: &Path, identity: &RawExternalPredictionIdentity) -> PathBuf {
    root.join("raw_predictions").join(&identity.digest)
}

pub fn raw_cache_manifest_path(directory: &Path) -> PathBuf {
    directory.join("raw_external_predictions.json")
}

fn raw_cache_payload_path(directory: &Path) -> PathBuf {
    directory.join("raw_external_predictions.bin.zst")
}

fn required_prediction_fields_are_finite(features: &ExternalPsmFeatures) -> bool {
    [
        features.ms2rescore_ms2pip_pcc,
        features.ms2rescore_spectral_angle,
        features.ms2rescore_fragment_intensity_agreement,
        features.ms2rescore_deeplc_predicted_rt,
        features.ms2rescore_deeplc_calibrated_rt,
        features.ms2rescore_deeplc_rt_error,
        features.ms2rescore_deeplc_abs_rt_error,
        features.tims2rescore_observed_ion_mobility,
    ]
    .into_iter()
    .all(f32::is_finite)
}

fn validate_raw_records(
    records: &[ExternalAnnotationRecord],
    identity: &RawExternalPredictionIdentity,
) -> Result<()> {
    anyhow::ensure!(
        records.len() == identity.requested_candidate_count,
        "raw external prediction cache candidate-count mismatch"
    );
    let mut ids = HashSet::with_capacity(records.len());
    for record in records {
        anyhow::ensure!(
            ids.insert(record.stable_id.as_str()),
            "duplicate stable candidate ID in raw external prediction cache"
        );
        anyhow::ensure!(
            record.features.ms2rescore_feature_joined,
            "raw external prediction cache contains an incomplete candidate"
        );
        anyhow::ensure!(
            required_prediction_fields_are_finite(&record.features),
            "raw external prediction cache contains nonfinite required MS2PIP/DeepLC fields for {}",
            record.stable_id
        );
    }
    let payload_input_sha256 =
        raw_candidate_id_sha256(records.iter().map(|record| record.stable_id.as_str()))?;
    anyhow::ensure!(
        payload_input_sha256 == identity.raw_input_sha256,
        "raw external prediction cache stable-ID population mismatch"
    );
    Ok(())
}

pub fn load_raw_cache(
    directory: &Path,
    expected: &RawExternalPredictionIdentity,
) -> Result<
    Option<(
        RawExternalPredictionCacheManifest,
        Vec<ExternalAnnotationRecord>,
    )>,
> {
    let manifest_path = raw_cache_manifest_path(directory);
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest: RawExternalPredictionCacheManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path)?).with_context(|| {
            format!(
                "invalid raw prediction cache manifest {}",
                manifest_path.display()
            )
        })?;
    anyhow::ensure!(
        manifest.schema_version == RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION
            && manifest.identity == *expected,
        "raw external prediction cache identity mismatch in {}",
        manifest_path.display()
    );
    let payload_path = directory.join(&manifest.payload_file);
    anyhow::ensure!(
        payload_path.is_file(),
        "raw external prediction cache payload is missing"
    );
    anyhow::ensure!(
        sha256_file(&payload_path)? == manifest.payload_sha256,
        "raw external prediction cache payload hash mismatch"
    );
    let decoder = zstd::stream::read::Decoder::new(BufReader::new(File::open(&payload_path)?))?;
    let payload: RawExternalPredictionPayload = bincode::deserialize_from(decoder)?;
    anyhow::ensure!(
        payload.schema_version == RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION
            && payload.raw_prediction_identity == *expected
            && payload.raw_prediction_identity == manifest.identity,
        "raw external prediction cache payload identity mismatch"
    );
    validate_raw_records(&payload.records, expected)?;
    let joined = payload
        .records
        .iter()
        .filter(|record| record.features.ms2rescore_feature_joined)
        .count();
    anyhow::ensure!(
        manifest.prediction_count == payload.records.len()
            && manifest.joined_prediction_count == joined,
        "raw external prediction cache manifest/payload count mismatch"
    );
    Ok(Some((manifest, payload.records)))
}

pub fn write_raw_cache(
    directory: &Path,
    identity: &RawExternalPredictionIdentity,
    records: Vec<ExternalAnnotationRecord>,
    migrated_from: Option<&ExternalAnnotationCacheManifest>,
) -> Result<RawExternalPredictionCacheManifest> {
    validate_raw_records(&records, identity)?;
    if directory.exists() {
        let (manifest, existing) = load_raw_cache(directory, identity)?
            .context("refusing to overwrite an incompatible raw external prediction cache")?;
        anyhow::ensure!(
            bincode::serialize(&existing)? == bincode::serialize(&records)?,
            "raw external predictions changed under an identical portable identity"
        );
        return Ok(manifest);
    }
    std::fs::create_dir_all(directory)?;
    let payload_path = raw_cache_payload_path(directory);
    let temporary = directory.join("raw_external_predictions.bin.zst.tmp");
    let payload = RawExternalPredictionPayload {
        schema_version: RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION,
        raw_prediction_identity: identity.clone(),
        records,
    };
    {
        let mut encoder =
            zstd::stream::write::Encoder::new(BufWriter::new(File::create(&temporary)?), 3)?;
        bincode::serialize_into(&mut encoder, &payload)?;
        encoder.finish()?;
    }
    std::fs::rename(&temporary, &payload_path)?;
    let manifest = RawExternalPredictionCacheManifest {
        schema_version: RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION,
        identity: identity.clone(),
        payload_file: payload_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("raw_external_predictions.bin.zst")
            .into(),
        payload_sha256: sha256_file(&payload_path)?,
        prediction_count: payload.records.len(),
        joined_prediction_count: payload.records.len(),
        migrated_from_schema_v2_fingerprint: migrated_from
            .map(|manifest| manifest.identity.digest.clone()),
        migrated_from_schema_v2_payload_sha256: migrated_from
            .map(|manifest| manifest.payload_sha256.clone()),
    };
    write_json_atomic(&raw_cache_manifest_path(directory), &manifest)?;
    Ok(manifest)
}

/// Publish a raw prediction cache as one verified directory transaction.
/// Exact existing caches are reopened and reused; an incompatible final path
/// fails closed and is never overwritten or regenerated in place.
pub fn publish_raw_cache_atomic(
    directory: &Path,
    identity: &RawExternalPredictionIdentity,
    records: Vec<ExternalAnnotationRecord>,
) -> Result<(RawExternalPredictionCacheManifest, bool)> {
    if directory.exists() {
        let (manifest, existing) = load_raw_cache(directory, identity)?
            .context("existing final raw prediction cache is incomplete or incompatible")?;
        anyhow::ensure!(
            bincode::serialize(&existing)? == bincode::serialize(&records)?,
            "raw predictions changed under an existing portable identity"
        );
        return Ok((manifest, true));
    }

    let parent = directory
        .parent()
        .context("raw prediction cache directory has no parent")?;
    std::fs::create_dir_all(parent)?;
    let stem = directory
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("raw-predictions");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging = parent.join(format!(".{stem}.partial.{}.{}", std::process::id(), nonce));
    anyhow::ensure!(
        !staging.exists(),
        "raw-cache staging directory already exists: {}",
        staging.display()
    );

    let result = (|| -> Result<RawExternalPredictionCacheManifest> {
        let written = write_raw_cache(&staging, identity, records, None)?;
        let (verified, _) = load_raw_cache(&staging, identity)?
            .context("raw-cache staging verification did not find a complete cache")?;
        anyhow::ensure!(
            verified.payload_sha256 == written.payload_sha256
                && verified.prediction_count == written.prediction_count
                && verified.joined_prediction_count == written.joined_prediction_count,
            "raw-cache staging verification disagrees with the written resource"
        );
        anyhow::ensure!(
            !directory.exists(),
            "raw-cache final directory appeared during generation; refusing to overwrite {}",
            directory.display()
        );
        std::fs::rename(&staging, directory).with_context(|| {
            format!(
                "atomically publishing raw cache {} -> {}",
                staging.display(),
                directory.display()
            )
        })?;
        #[cfg(unix)]
        std::fs::File::open(parent)?.sync_all()?;
        let (published, _) = load_raw_cache(directory, identity)?
            .context("published raw cache failed immutable reopen verification")?;
        anyhow::ensure!(
            published.payload_sha256 == written.payload_sha256
                && published.prediction_count == written.prediction_count
                && published.joined_prediction_count == written.joined_prediction_count,
            "published raw cache differs from the verified staging resource"
        );
        Ok(published)
    })();

    if result.is_err() && staging.exists() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result.map(|manifest| (manifest, false))
}

pub fn raw_cache_usage(
    directory: &Path,
    manifest: &RawExternalPredictionCacheManifest,
    calibration: Option<StageExternalCalibrationIdentity>,
    reused: bool,
    request: &ExternalAnnotationCacheRequest,
) -> ExternalAnnotationCacheUsage {
    ExternalAnnotationCacheUsage {
        annotation_fingerprint: calibration
            .as_ref()
            .map(|identity| identity.digest.clone())
            .unwrap_or_else(|| manifest.identity.digest.clone()),
        search_fingerprint: manifest.identity.search_fingerprint.clone(),
        manifest: raw_cache_manifest_path(directory),
        payload: directory.join(&manifest.payload_file),
        reused,
        annotation_count: manifest.prediction_count,
        joined_annotation_count: manifest.joined_prediction_count,
        requested_max_rank: manifest.identity.requested_max_rank,
        requested_root: request.root.clone(),
        generation_allowed: !request.require_existing && !request.migration_only,
        preflight_result: if reused {
            "validated_raw_prediction_cache".into()
        } else if manifest.migrated_from_schema_v2_fingerprint.is_some() {
            "migrated_schema_v2_raw_prediction_cache".into()
        } else {
            "generated_raw_prediction_cache".into()
        },
        raw_prediction_cache_fingerprint: manifest.identity.digest.clone(),
        raw_prediction_cache_schema_version: manifest.schema_version,
        stage_calibration_identity: calibration,
        migrated_from_schema_v2: manifest.migrated_from_schema_v2_fingerprint.is_some(),
    }
}

pub fn usage(
    directory: &Path,
    manifest: &ExternalAnnotationCacheManifest,
    reused: bool,
    request: &ExternalAnnotationCacheRequest,
) -> ExternalAnnotationCacheUsage {
    ExternalAnnotationCacheUsage {
        annotation_fingerprint: manifest.identity.digest.clone(),
        search_fingerprint: manifest.identity.search_fingerprint.clone(),
        manifest: cache_manifest_path(directory),
        payload: directory.join(&manifest.payload_file),
        reused,
        annotation_count: manifest.annotation_count,
        joined_annotation_count: manifest.joined_annotation_count,
        requested_max_rank: manifest.identity.requested_max_rank,
        requested_root: request.root.clone(),
        generation_allowed: !request.require_existing && !request.migration_only,
        preflight_result: if reused {
            "validated_exact".into()
        } else {
            "generated_cache".into()
        },
        raw_prediction_cache_fingerprint: String::new(),
        raw_prediction_cache_schema_version: 0,
        stage_calibration_identity: None,
        migrated_from_schema_v2: false,
    }
}

pub fn verify_usage(usage: &ExternalAnnotationCacheUsage) -> Result<()> {
    if !usage.raw_prediction_cache_fingerprint.is_empty() {
        let manifest: RawExternalPredictionCacheManifest =
            serde_json::from_slice(&std::fs::read(&usage.manifest)?).with_context(|| {
                format!(
                    "invalid raw prediction cache manifest {}",
                    usage.manifest.display()
                )
            })?;
        let directory = usage
            .manifest
            .parent()
            .context("raw prediction cache manifest has no parent directory")?;
        anyhow::ensure!(
            raw_cache_manifest_path(directory) == usage.manifest
                && directory.join(&manifest.payload_file) == usage.payload,
            "raw prediction cache usage paths do not match its manifest"
        );
        anyhow::ensure!(
            manifest.schema_version == RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION
                && manifest.identity.digest == usage.raw_prediction_cache_fingerprint
                && manifest.identity.search_fingerprint == usage.search_fingerprint
                && manifest.prediction_count == usage.annotation_count
                && manifest.joined_prediction_count == usage.joined_annotation_count
                && manifest.identity.requested_max_rank == usage.requested_max_rank,
            "raw prediction cache usage record does not match its verified manifest"
        );
        if let Some(calibration) = usage.stage_calibration_identity.as_ref() {
            anyhow::ensure!(
                calibration.digest == usage.annotation_fingerprint
                    && calibration.raw_prediction_fingerprint
                        == usage.raw_prediction_cache_fingerprint,
                "stage calibration identity is inconsistent with raw prediction provenance"
            );
        } else {
            anyhow::ensure!(
                usage.annotation_fingerprint == usage.raw_prediction_cache_fingerprint,
                "preflight raw prediction usage has an unexpected derived identity"
            );
        }
        anyhow::ensure!(
            usage.requested_root
                == directory
                    .parent()
                    .and_then(Path::parent)
                    .unwrap_or(directory),
            "raw prediction cache requested root is inconsistent"
        );
        anyhow::ensure!(
            usage.payload.is_file() && sha256_file(&usage.payload)? == manifest.payload_sha256,
            "raw prediction cache payload hash mismatch"
        );
        anyhow::ensure!(
            load_raw_cache(directory, &manifest.identity)?.is_some(),
            "raw prediction cache disappeared during verification"
        );
        return Ok(());
    }
    let manifest: ExternalAnnotationCacheManifest =
        serde_json::from_slice(&std::fs::read(&usage.manifest)?).with_context(|| {
            format!(
                "invalid annotation cache manifest {}",
                usage.manifest.display()
            )
        })?;
    let directory = usage
        .manifest
        .parent()
        .context("annotation cache manifest has no parent directory")?;
    anyhow::ensure!(
        cache_manifest_path(directory) == usage.manifest
            && directory.join(&manifest.payload_file) == usage.payload,
        "MS2Rescore annotation cache usage paths do not match its manifest"
    );
    anyhow::ensure!(
        manifest.schema_version == EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION
            && manifest.identity.schema_version == EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION
            && manifest.identity.stable_candidate_id_schema == CANDIDATE_ID_SCHEMA
            && manifest.identity.feature_schema == EXTERNAL_ANNOTATION_FEATURE_SCHEMA
            && manifest.identity.digest == usage.annotation_fingerprint
            && manifest.identity.search_fingerprint == usage.search_fingerprint
            && manifest.annotation_count == usage.annotation_count
            && manifest.joined_annotation_count == usage.joined_annotation_count
            && manifest.identity.requested_max_rank == usage.requested_max_rank,
        "MS2Rescore annotation cache usage record does not match its verified manifest"
    );
    let legacy_execution_provenance = usage.requested_root.as_os_str().is_empty()
        && usage.preflight_result.is_empty()
        && !usage.generation_allowed;
    anyhow::ensure!(
        legacy_execution_provenance
            || (usage.requested_root == directory.parent().unwrap_or(directory)
                && (!usage.reused
                    || usage.preflight_result == "valid_existing_cache"
                    || usage.preflight_result == "validated_exact")
                && (usage.generation_allowed || usage.reused)),
        "MS2Rescore annotation cache execution provenance is inconsistent"
    );
    anyhow::ensure!(
        usage.payload.is_file() && sha256_file(&usage.payload)? == manifest.payload_sha256,
        "MS2Rescore annotation cache payload hash mismatch"
    );
    let (verified_manifest, _) = load_cache(directory, &manifest.identity)?
        .context("MS2Rescore annotation cache disappeared during verification")?;
    anyhow::ensure!(
        verified_manifest.payload_sha256 == manifest.payload_sha256,
        "MS2Rescore annotation cache manifest and payload disagree"
    );
    Ok(())
}

/// Read-only strict preflight for the model-independent raw prediction layer.
/// Unlike schema-v2 annotation identity, this identity is completely derivable
/// before model/window calibration and never launches Python or writes state.
pub fn preflight_existing_cache_root(
    request: &ExternalAnnotationCacheRequest,
    settings: &ExternalFeatureGenerationSettings,
    search_fingerprint: &str,
    candidate_ids: &HashSet<String>,
    requested_max_rank: u32,
) -> Result<Vec<ExternalAnnotationCacheUsage>> {
    anyhow::ensure!(
        request.require_existing,
        "strict annotation preflight requires require_existing=true"
    );
    let identity = raw_prediction_identity_from_candidate_ids_with_probe_root(
        search_fingerprint,
        settings,
        candidate_ids,
        requested_max_rank,
        &request.root,
        true,
    )
    .with_context(|| {
        format!(
            "strict raw-prediction-cache preflight failed: classification=generator_provenance_unavailable root={} search_space={} candidate_population={} generation_prohibited=true",
            request.root.display(), request.search_space, search_fingerprint
        )
    })?;
    let directory = raw_cache_directory(&request.root, &identity);
    let (manifest, records) = load_raw_cache(&directory, &identity)?
        .with_context(|| {
            format!(
                "strict raw-prediction-cache preflight failed: classification=missing_exact root={} search_space={} candidate_population={} expected_raw_fingerprint={} expected_schema={} generation_prohibited=true",
                request.root.display(),
                request.search_space,
                search_fingerprint,
                identity.digest,
                RAW_EXTERNAL_PREDICTION_CACHE_SCHEMA_VERSION
            )
        })?;
    let joined_ids = records
        .iter()
        .map(|record| record.stable_id.clone())
        .collect::<HashSet<_>>();
    anyhow::ensure!(
        joined_ids == *candidate_ids,
        "strict raw-prediction-cache preflight failed: classification=candidate_population_mismatch root={} search_space={} expected={} actual={} generation_prohibited=true",
        request.root.display(), request.search_space, candidate_ids.len(), joined_ids.len()
    );
    Ok(vec![raw_cache_usage(
        &directory, &manifest, None, true, request,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layered_settings() -> ExternalFeatureGenerationSettings {
        ExternalFeatureGenerationSettings {
            deeplc_calibration_set_size: Some(10),
            ..ExternalFeatureGenerationSettings::default()
        }
    }

    fn write_empty_probe(root: &Path, settings: &ExternalFeatureGenerationSettings) {
        let probe_key = generator_probe_key(settings).unwrap();
        let path = root
            .join("generator_identity_probes")
            .join(format!("{probe_key}.json"));
        write_json_atomic(
            &path,
            &GeneratorProbeCache {
                schema_version: 2,
                probe_key,
                python_environment: None,
                package_metadata: Vec::new(),
                model_components: Vec::new(),
                model_files: Vec::new(),
            },
        )
        .unwrap();
    }

    fn identity() -> ExternalAnnotationIdentity {
        ExternalAnnotationIdentity {
            schema_version: EXTERNAL_ANNOTATION_CACHE_SCHEMA_VERSION,
            digest: "annotation-digest".into(),
            search_fingerprint: "search-digest".into(),
            generator_settings_sha256: "settings".into(),
            calibration_input_sha256: "input".into(),
            stable_candidate_id_schema: CANDIDATE_ID_SCHEMA.into(),
            feature_schema: EXTERNAL_ANNOTATION_FEATURE_SCHEMA.into(),
            requested_candidate_count: 1,
            requested_max_rank: 10,
            model_components: Vec::new(),
        }
    }

    fn complete_features() -> ExternalPsmFeatures {
        ExternalPsmFeatures {
            ms2rescore_ms2pip_pcc: 0.8,
            ms2rescore_spectral_angle: 0.7,
            ms2rescore_fragment_intensity_agreement: 0.6,
            ms2rescore_deeplc_predicted_rt: 12.5,
            ms2rescore_deeplc_calibrated_rt: 12.0,
            ms2rescore_deeplc_rt_error: 0.5,
            ms2rescore_deeplc_abs_rt_error: 0.5,
            tims2rescore_observed_ion_mobility: 1.1,
            ms2rescore_feature_joined: true,
            ..ExternalPsmFeatures::default()
        }
    }

    #[test]
    fn cache_round_trip_is_hash_and_identity_checked() {
        let root = std::env::temp_dir().join(format!(
            "sage-external-annotation-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let identity = identity();
        let directory = cache_directory(&root, &identity);
        let mut features = ExternalPsmFeatures::default();
        features.ms2rescore_feature_joined = true;
        let written = write_cache(
            &directory,
            &identity,
            vec![ExternalAnnotationRecord {
                stable_id: "candidate".into(),
                features,
            }],
        )
        .unwrap();
        assert_eq!(written.joined_annotation_count, 1);
        let (loaded, records) = load_cache(&directory, &identity).unwrap().unwrap();
        assert_eq!(loaded.payload_sha256, written.payload_sha256);
        assert_eq!(records[0].stable_id, "candidate");
        let request = ExternalAnnotationCacheRequest {
            root: root.clone(),
            require_existing: false,
            search_space: "+entrapment".into(),
            stage: "legacy".into(),
            analysis_fingerprint: "legacy".into(),
            migration_only: false,
        };
        let recorded_usage = usage(&directory, &written, false, &request);
        verify_usage(&recorded_usage).unwrap();
        let mut legacy_value = serde_json::to_value(&recorded_usage).unwrap();
        let legacy_object = legacy_value.as_object_mut().unwrap();
        legacy_object.remove("requested_root");
        legacy_object.remove("generation_allowed");
        legacy_object.remove("preflight_result");
        let legacy_usage: ExternalAnnotationCacheUsage =
            serde_json::from_value(legacy_value).unwrap();
        verify_usage(&legacy_usage).unwrap();

        std::fs::write(directory.join(&written.payload_file), b"corrupt").unwrap();
        assert!(load_cache(&directory, &identity).is_err());
        assert!(verify_usage(&recorded_usage).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn calibration_values_are_part_of_annotation_identity() {
        let settings = layered_settings();
        let mut input = ExternalAnnotationInput {
            stable_id: "candidate".into(),
            score: 10.0,
            q_value: Some(0.01),
            pep: Some(0.02),
            retention_time: 12.0,
            ion_mobility: 1.1,
            precursor_mass: 900.0,
            charge: 2,
            rank: 1,
        };
        let first = annotation_identity("search", &settings, &[input.clone()], 10).unwrap();
        input.q_value = Some(0.05);
        let second = annotation_identity("search", &settings, &[input], 10).unwrap();
        assert_ne!(first.digest, second.digest);
    }

    #[test]
    fn raw_identity_is_calibration_independent_and_order_invariant() {
        let root = std::env::temp_dir().join(format!(
            "sage-layered-identity-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let settings = layered_settings();
        let mut inputs = vec![
            ExternalAnnotationInput {
                stable_id: "b".into(),
                score: 20.0,
                q_value: Some(0.01),
                pep: Some(0.02),
                retention_time: 12.0,
                ion_mobility: 1.1,
                precursor_mass: 900.0,
                charge: 2,
                rank: 1,
            },
            ExternalAnnotationInput {
                stable_id: "a".into(),
                score: 10.0,
                q_value: Some(0.03),
                pep: Some(0.04),
                retention_time: 13.0,
                ion_mobility: 1.2,
                precursor_mass: 800.0,
                charge: 3,
                rank: 2,
            },
        ];
        let first =
            raw_prediction_identity_with_probe_root("search", &settings, &inputs, 50, &root, false)
                .unwrap();
        inputs.reverse();
        inputs[0].q_value = Some(0.9);
        inputs[0].pep = Some(0.8);
        let second =
            raw_prediction_identity_with_probe_root("search", &settings, &inputs, 50, &root, false)
                .unwrap();
        assert_eq!(first, second);
        let target_only = raw_prediction_identity_with_probe_root(
            "target-only-search",
            &settings,
            &inputs,
            50,
            &root,
            false,
        )
        .unwrap();
        assert_ne!(first.digest, target_only.digest);

        let request_a = ExternalAnnotationCacheRequest {
            root: root.clone(),
            require_existing: false,
            search_space: "+entrapment".into(),
            stage: "moments:ms2rescore".into(),
            analysis_fingerprint: "analysis-a".into(),
            migration_only: false,
        };
        let request_b = ExternalAnnotationCacheRequest {
            stage: "mle:ms2rescore".into(),
            analysis_fingerprint: "analysis-b".into(),
            ..request_a.clone()
        };
        let calibration_a = stage_calibration_identity(&first, &inputs, &request_a).unwrap();
        let calibration_b = stage_calibration_identity(&first, &inputs, &request_b).unwrap();
        assert_ne!(calibration_a.digest, calibration_b.digest);
        assert_eq!(
            calibration_a.raw_prediction_fingerprint,
            calibration_b.raw_prediction_fingerprint
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn raw_cache_rejects_partial_duplicate_nonfinite_and_corrupt_state() {
        let root = std::env::temp_dir().join(format!(
            "sage-layered-integrity-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let settings = layered_settings();
        let inputs = [ExternalAnnotationInput {
            stable_id: "candidate".into(),
            score: 10.0,
            q_value: None,
            pep: None,
            retention_time: 12.0,
            ion_mobility: 1.1,
            precursor_mass: 900.0,
            charge: 2,
            rank: 1,
        }];
        let identity =
            raw_prediction_identity_with_probe_root("search", &settings, &inputs, 1, &root, false)
                .unwrap();
        let directory = raw_cache_directory(&root, &identity);
        assert!(write_raw_cache(&directory, &identity, Vec::new(), None).is_err());
        assert!(write_raw_cache(
            &directory,
            &identity,
            vec![ExternalAnnotationRecord {
                stable_id: "different-candidate".into(),
                features: complete_features(),
            }],
            None,
        )
        .unwrap_err()
        .to_string()
        .contains("stable-ID population mismatch"));
        let duplicate_identity = RawExternalPredictionIdentity {
            requested_candidate_count: 2,
            ..identity.clone()
        };
        assert!(write_raw_cache(
            &raw_cache_directory(&root, &duplicate_identity),
            &duplicate_identity,
            vec![
                ExternalAnnotationRecord {
                    stable_id: "candidate".into(),
                    features: complete_features(),
                },
                ExternalAnnotationRecord {
                    stable_id: "candidate".into(),
                    features: complete_features(),
                },
            ],
            None,
        )
        .is_err());
        let mut nonfinite = complete_features();
        nonfinite.ms2rescore_deeplc_predicted_rt = f32::NAN;
        assert!(write_raw_cache(
            &directory,
            &identity,
            vec![ExternalAnnotationRecord {
                stable_id: "candidate".into(),
                features: nonfinite,
            }],
            None,
        )
        .is_err());
        let manifest = write_raw_cache(
            &directory,
            &identity,
            vec![ExternalAnnotationRecord {
                stable_id: "candidate".into(),
                features: complete_features(),
            }],
            None,
        )
        .unwrap();
        std::fs::write(directory.join(manifest.payload_file), b"corrupt").unwrap();
        assert!(load_raw_cache(&directory, &identity).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_raw_cache_publication_never_accepts_partial_or_incompatible_state() {
        let root = std::env::temp_dir().join(format!(
            "sage-atomic-raw-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let settings = layered_settings();
        let inputs = [ExternalAnnotationInput {
            stable_id: "candidate".into(),
            score: 10.0,
            q_value: None,
            pep: None,
            retention_time: 12.0,
            ion_mobility: 1.1,
            precursor_mass: 900.0,
            charge: 2,
            rank: 1,
        }];
        let identity =
            raw_prediction_identity_with_probe_root("search", &settings, &inputs, 1, &root, false)
                .unwrap();
        let directory = raw_cache_directory(&root, &identity);

        assert!(publish_raw_cache_atomic(&directory, &identity, Vec::new()).is_err());
        assert!(!directory.exists());

        let record = ExternalAnnotationRecord {
            stable_id: "candidate".into(),
            features: complete_features(),
        };
        let (manifest, reused) =
            publish_raw_cache_atomic(&directory, &identity, vec![record.clone()]).unwrap();
        assert!(!reused);
        assert_eq!(manifest.prediction_count, 1);
        let (_, reused) =
            publish_raw_cache_atomic(&directory, &identity, vec![record.clone()]).unwrap();
        assert!(reused);

        let mut changed = record;
        changed.features.ms2rescore_deeplc_rt_error = 2.0;
        let changed_error = publish_raw_cache_atomic(&directory, &identity, vec![changed])
            .unwrap_err()
            .to_string();
        assert!(changed_error.contains("raw predictions changed"));

        std::fs::write(directory.join(&manifest.payload_file), b"corrupt").unwrap();
        assert!(publish_raw_cache_atomic(
            &directory,
            &identity,
            vec![ExternalAnnotationRecord {
                stable_id: "candidate".into(),
                features: complete_features(),
            }],
        )
        .is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn raw_generator_identity_is_portable_across_source_relocation() {
        let root = std::env::temp_dir().join(format!(
            "sage-layered-relocation-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let first_spectrum = root.join("mac/run.mzML");
        let moved_spectrum = root.join("wsl/different-name.mzML");
        let first_wrapper = root.join("mac/wrapper.py");
        let moved_wrapper = root.join("wsl/wrapper-renamed.py");
        std::fs::create_dir_all(first_spectrum.parent().unwrap()).unwrap();
        std::fs::create_dir_all(moved_spectrum.parent().unwrap()).unwrap();
        std::fs::write(&first_spectrum, b"identical spectrum bytes").unwrap();
        std::fs::write(&moved_spectrum, b"identical spectrum bytes").unwrap();
        std::fs::write(&first_wrapper, b"identical wrapper bytes").unwrap();
        std::fs::write(&moved_wrapper, b"identical wrapper bytes").unwrap();
        let mut first = layered_settings();
        first.command_path = Some(first_wrapper.display().to_string());
        first.spectrum_file_mapping.insert(
            "stable-run-ordinal-0".into(),
            first_spectrum.display().to_string(),
        );
        let mut moved = first.clone();
        moved.command_path = Some(moved_wrapper.display().to_string());
        moved.spectrum_file_mapping.insert(
            "stable-run-ordinal-0".into(),
            moved_spectrum.display().to_string(),
        );
        let probe_root = root.join("probe");
        let first_digest = raw_generator_identity(&first, Some(&probe_root), false)
            .unwrap()
            .0;
        // Relocating content-identical sources reuses the durable probe in
        // strict mode and therefore cannot invoke Python or another resolver.
        let moved_digest = raw_generator_identity(&moved, Some(&probe_root), true)
            .unwrap()
            .0;
        assert_eq!(first_digest, moved_digest);
        std::fs::write(&moved_spectrum, b"changed spectrum bytes").unwrap();
        assert_ne!(
            first_digest,
            raw_generator_identity(&moved, Some(&probe_root), false)
                .unwrap()
                .0
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn strict_preflight_reuses_complete_cache_without_identity_probe() {
        let root = std::env::temp_dir().join(format!(
            "sage-strict-annotation-preflight-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let settings = layered_settings();
        write_empty_probe(&root, &settings);
        let inputs = vec![ExternalAnnotationInput {
            stable_id: "candidate".into(),
            score: 10.0,
            q_value: Some(0.01),
            pep: Some(0.02),
            retention_time: 12.0,
            ion_mobility: 1.1,
            precursor_mass: 900.0,
            charge: 2,
            rank: 1,
        }];
        let identity = raw_prediction_identity_with_probe_root(
            "search-digest",
            &settings,
            &inputs,
            10,
            &root,
            true,
        )
        .unwrap();
        let directory = raw_cache_directory(&root, &identity);
        let features = complete_features();
        write_raw_cache(
            &directory,
            &identity,
            vec![ExternalAnnotationRecord {
                stable_id: "candidate".into(),
                features,
            }],
            None,
        )
        .unwrap();
        let request = ExternalAnnotationCacheRequest {
            root: root.clone(),
            require_existing: true,
            search_space: "+entrapment".into(),
            stage: "static_preflight".into(),
            analysis_fingerprint: "search-digest".into(),
            migration_only: false,
        };
        let candidate_ids = ["candidate".to_string()].into_iter().collect();
        let usages =
            preflight_existing_cache_root(&request, &settings, "search-digest", &candidate_ids, 10)
                .unwrap();
        assert_eq!(usages.len(), 1);
        assert!(usages[0].reused);
        assert!(!usages[0].generation_allowed);
        assert_eq!(usages[0].preflight_result, "validated_raw_prediction_cache");

        let different_ids = ["different-candidate".to_string()].into_iter().collect();
        let error =
            preflight_existing_cache_root(&request, &settings, "search-digest", &different_ids, 10)
                .unwrap_err()
                .to_string();
        assert!(error.contains("missing_exact"));

        std::fs::write(&usages[0].payload, b"corrupt").unwrap();
        let error =
            preflight_existing_cache_root(&request, &settings, "search-digest", &candidate_ids, 10)
                .unwrap_err()
                .to_string();
        assert!(error.contains("payload hash mismatch"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn strict_preflight_missing_cache_fails_without_creating_root() {
        let root = std::env::temp_dir().join(format!(
            "sage-strict-annotation-missing-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let request = ExternalAnnotationCacheRequest {
            root: root.clone(),
            require_existing: true,
            search_space: "target_only".into(),
            stage: "static_preflight".into(),
            analysis_fingerprint: "search".into(),
            migration_only: false,
        };
        let error = preflight_existing_cache_root(
            &request,
            &layered_settings(),
            "search",
            &["candidate".to_string()].into_iter().collect(),
            10,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("generator_provenance_unavailable"));
        assert!(error.contains("generation_prohibited=true"));
        assert!(!root.exists());
    }

    #[test]
    fn strict_execution_control_does_not_change_annotation_identity() {
        let identity = identity();
        let root = PathBuf::from("portable-cache-root");
        let relaxed = ExternalAnnotationCacheRequest {
            root: root.clone(),
            require_existing: false,
            search_space: "+entrapment".into(),
            stage: "moments:ms2rescore".into(),
            analysis_fingerprint: "analysis".into(),
            migration_only: false,
        };
        let strict = ExternalAnnotationCacheRequest {
            root,
            require_existing: true,
            search_space: "+entrapment".into(),
            stage: "moments:ms2rescore".into(),
            analysis_fingerprint: "analysis".into(),
            migration_only: false,
        };
        assert_eq!(
            cache_directory(&relaxed.root, &identity),
            cache_directory(&strict.root, &identity)
        );
    }

    #[test]
    fn analysis_policy_and_cache_location_do_not_change_generator_identity() {
        let settings = layered_settings();
        let expected = generator_settings_sha256(&settings).unwrap();

        let mut changed = settings.clone();
        changed.fail_policy = crate::input::ExternalFeatureFailPolicy::WarnAndContinue;
        changed.use_mode = crate::input::ExternalFeatureUseMode::BoundedDfExperts;
        changed.temp_directory = Some("/different/temp".into());
        changed.output_directory = Some("/different/output".into());
        assert_eq!(expected, generator_settings_sha256(&changed).unwrap());

        changed.ms2pip_model = Some("different-model".into());
        assert_ne!(expected, generator_settings_sha256(&changed).unwrap());
    }

    #[test]
    fn model_content_identity_is_path_independent_and_byte_sensitive() {
        let root = std::env::temp_dir().join(format!(
            "sage-model-identity-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let first = root.join("first/model.keras");
        let second = root.join("moved/model.keras");
        std::fs::create_dir_all(first.parent().unwrap()).unwrap();
        std::fs::create_dir_all(second.parent().unwrap()).unwrap();
        std::fs::write(&first, b"same-model-bytes").unwrap();
        std::fs::write(&second, b"same-model-bytes").unwrap();
        let identify = |path: &Path| ModelComponentIdentity {
            generator: "deeplc".into(),
            logical_model_name: "configured:0".into(),
            relative_filename: "model.keras".into(),
            size_bytes: path.metadata().unwrap().len(),
            sha256: sha256_file(path).unwrap(),
        };
        let first_component = identify(&first);
        let second_component = identify(&second);
        assert_eq!(first_component, second_component);
        let mut first_settings = ExternalFeatureGenerationSettings::default();
        first_settings.feature_generators = Some(serde_json::json!({
            "deeplc": {"path_model": first}
        }));
        let mut moved_settings = first_settings.clone();
        moved_settings.feature_generators = Some(serde_json::json!({
            "deeplc": {"path_model": second}
        }));
        let first_digest =
            generator_identity_digest(&first_settings, None, &[first_component.clone()]).unwrap();
        let moved_digest =
            generator_identity_digest(&moved_settings, None, &[second_component]).unwrap();
        assert_eq!(first_digest, moved_digest);
        std::fs::write(&second, b"one-byte-change!").unwrap();
        let changed_component = identify(&second);
        assert_ne!(first_component, changed_component);
        assert_ne!(
            first_digest,
            generator_identity_digest(&moved_settings, None, &[changed_component]).unwrap()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn wrapper_and_relevant_environment_changes_invalidate_identity() {
        let root = std::env::temp_dir().join(format!(
            "sage-generator-identity-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let wrapper = root.join("wrapper.py");
        std::fs::write(&wrapper, b"print('v1')\n").unwrap();
        let mut settings = ExternalFeatureGenerationSettings::default();
        settings.command_path = Some(wrapper.display().to_string());
        let env_v1 = Some(r#"{"python":"3.12.0","packages":{"ms2pip":"4.1.2"}}"#.into());
        let env_v2 = Some(r#"{"python":"3.12.0","packages":{"ms2pip":"4.1.3"}}"#.into());
        let first = generator_identity_digest(&settings, env_v1.clone(), &[]).unwrap();
        let package_changed = generator_identity_digest(&settings, env_v2, &[]).unwrap();
        assert_ne!(first, package_changed);
        std::fs::write(&wrapper, b"print('v2')\n").unwrap();
        let wrapper_changed = generator_identity_digest(&settings, env_v1, &[]).unwrap();
        assert_ne!(first, wrapper_changed);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn cached_generator_probe_avoids_a_second_python_invocation() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "sage-generator-probe-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let marker = root.join("invocations");
        let python = root.join("python");
        std::fs::write(
            &python,
            format!(
                "#!/bin/sh\nprintf x >> '{}'\nprintf '%s\\n' '{{\"identity\":\"test-environment\",\"metadata_paths\":[]}}'\n",
                marker.display()
            ),
        )
        .unwrap();
        let mut permissions = python.metadata().unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&python, permissions).unwrap();

        let mut settings = ExternalFeatureGenerationSettings::default();
        settings.python_executable = Some(python.display().to_string());
        let first = generator_settings_sha256_with_probe_root(&settings, &root).unwrap();
        let second = generator_settings_sha256_with_probe_root(&settings, &root).unwrap();
        assert_eq!(first, second);
        assert_eq!(std::fs::read(&marker).unwrap(), b"x");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn raw_cache_rejects_implicit_q_value_based_deeplc_calibration() {
        let settings = ExternalFeatureGenerationSettings::default();
        let error = raw_generator_identity(&settings, None, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("calibration_set_size"));
        assert!(error.contains("stage dependent"));

        let no_deeplc = ExternalFeatureGenerationSettings {
            feature_generators: Some(serde_json::json!({"ms2pip": {}})),
            ..ExternalFeatureGenerationSettings::default()
        };
        raw_generator_identity(&no_deeplc, None, false).unwrap();
    }
}
