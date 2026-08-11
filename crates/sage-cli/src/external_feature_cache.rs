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

#[derive(Clone, Debug, Serialize)]
struct SourceIdentity {
    source: String,
    kind: String,
    sha256: String,
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
    Ok(generator_identity(settings, None)?.0)
}

fn generator_identity(
    settings: &ExternalFeatureGenerationSettings,
    probe_root: Option<&Path>,
) -> Result<(String, Vec<ModelComponentIdentity>)> {
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
    let digest = generator_identity_digest(
        settings,
        probe.python_environment.clone(),
        &probe.model_components,
    )?;
    Ok((digest, probe.model_components))
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
    }
    let value = serde_json::json!({
        "schema": "sage-generator-probe-key-v1",
        "settings": operational,
        "operational_home": std::env::var_os("HOME").map(|value| value.to_string_lossy().into_owned()),
        "command": settings.command_path.as_deref().map(source_identity).transpose()?,
        "python": settings.python_executable.as_deref().map(source_identity).transpose()?,
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
    let (generator_settings_sha256, model_components) = generator_identity(settings, probe_root)?;
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

pub fn usage(
    directory: &Path,
    manifest: &ExternalAnnotationCacheManifest,
    reused: bool,
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
    }
}

pub fn verify_usage(usage: &ExternalAnnotationCacheUsage) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let recorded_usage = usage(&directory, &written, false);
        verify_usage(&recorded_usage).unwrap();

        std::fs::write(directory.join(&written.payload_file), b"corrupt").unwrap();
        assert!(load_cache(&directory, &identity).is_err());
        assert!(verify_usage(&recorded_usage).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn calibration_values_are_part_of_annotation_identity() {
        let settings = ExternalFeatureGenerationSettings::default();
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
    fn analysis_policy_and_cache_location_do_not_change_generator_identity() {
        let settings = ExternalFeatureGenerationSettings::default();
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
        let first = generator_identity(&settings, Some(&root)).unwrap();
        let second = generator_identity(&settings, Some(&root)).unwrap();
        assert_eq!(first, second);
        assert_eq!(std::fs::read(&marker).unwrap(), b"x");
        std::fs::remove_dir_all(root).unwrap();
    }
}
