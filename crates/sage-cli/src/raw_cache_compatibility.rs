//! Read-only, explicitly pinned reuse of a reviewed historical raw contract.
//!
//! Historical producer identities remain untouched. This adapter verifies the
//! historical self hashes, all current generator/population components, payload
//! integrity and lane semantics, and records the current reader separately.
use crate::external_feature_cache::*;
use crate::input::ExternalFeatureGenerationSettings;
use crate::provenance::sha256_file;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const CONTRACT: &str = "raw-v3-whole-lane-v1-to-scoped-v2";
const HISTORICAL_SOURCE: &str = "430cdd14b0a35f0d473bf0c05f07652c44832e23e13eec3d956cffc7c704470d";
// A source change to production/parsing requires a new compatibility review.
// This module is reader provenance, excluded from the producer source digest.
const REVIEWED_CURRENT_SOURCE: &str =
    "33447ee50d5d9ac5e9aa628eabe89498d8572ced91e6b5577028b2828bd386ba";
const HISTORICAL_VERSION: &str = "0.15.0-beta.1.decoyfree-prealpha.1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExistingRawCacheReference {
    pub compatibility_contract: String,
    pub fingerprint: String,
    pub manifest_sha256: String,
    pub payload_sha256: String,
}

impl ExistingRawCacheReference {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.compatibility_contract == CONTRACT,
            "unsupported raw-cache compatibility_contract: {}",
            self.compatibility_contract
        );
        for (field, value) in [
            ("fingerprint", &self.fingerprint),
            ("manifest_sha256", &self.manifest_sha256),
            ("payload_sha256", &self.payload_sha256),
        ] {
            anyhow::ensure!(
                value.len() == 64
                    && value
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "invalid raw-cache reference {field}"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RawCacheVerifiedUse {
    pub schema_version: u32,
    pub reference: ExistingRawCacheReference,
    pub historical_finalizer: RawCacheFinalizerIdentity,
    pub current_finalizer: RawCacheFinalizerIdentity,
    pub current_reader_source_sha256: String,
    pub current_binary_sha256: String,
    pub generator_execution_sha256: String,
    pub content_fingerprint: String,
    pub complete_population_and_missingness_verified: bool,
    pub digest: String,
}

impl RawCacheVerifiedUse {
    fn payload_digest(&self) -> Result<String> {
        let mut value = serde_json::to_value(self)?;
        value.as_object_mut().unwrap().remove("digest");
        let mut h = Sha256::new();
        h.update(b"sage-raw-cache-verified-use-v1\0");
        h.update(serde_json::to_vec(&value)?);
        Ok(format!("{:x}", h.finalize()))
    }

    pub fn verify_record(&self, manifest: &Path, payload: &Path) -> Result<()> {
        self.reference.validate()?;
        anyhow::ensure!(
            self.schema_version == 1
                && self.complete_population_and_missingness_verified
                && self.payload_digest()? == self.digest,
            "invalid raw-cache verified-use record"
        );
        anyhow::ensure!(
            sha256_file(manifest)? == self.reference.manifest_sha256
                && sha256_file(payload)? == self.reference.payload_sha256,
            "verified-use artifact hashes changed"
        );
        let stored: RawExternalPredictionCacheManifest =
            serde_json::from_slice(&std::fs::read(manifest)?)?;
        anyhow::ensure!(stored.identity.digest == self.reference.fingerprint
            && stored.identity.finalizer == self.historical_finalizer
            && stored.generator_execution.digest == self.generator_execution_sha256
            && stored.content_fingerprint == self.content_fingerprint
            && self.historical_finalizer.schema_version == 1
            && self.current_finalizer.schema_version == 2,
            "verified-use historical and current responsibilities disagree with the preserved artifact");
        Ok(())
    }
}

#[derive(Serialize)]
struct Difference {
    field: String,
    recorded: Value,
    expected: Value,
}

fn differences(prefix: &str, recorded: &Value, expected: &Value, out: &mut Vec<Difference>) {
    if let (Some(a), Some(b)) = (recorded.as_object(), expected.as_object()) {
        let keys = a
            .keys()
            .chain(b.keys())
            .collect::<std::collections::BTreeSet<_>>();
        for key in keys {
            differences(
                &format!("{prefix}/{key}"),
                a.get(key).unwrap_or(&Value::Null),
                b.get(key).unwrap_or(&Value::Null),
                out,
            );
        }
    } else if recorded != expected {
        out.push(Difference {
            field: prefix.into(),
            recorded: recorded.clone(),
            expected: expected.clone(),
        });
    }
}

fn finalizer_digest(identity: &RawCacheFinalizerIdentity) -> String {
    let schema = if identity.schema_version == 1 {
        "sage-raw-cache-finalizer-identity-v1"
    } else {
        RAW_CACHE_FINALIZER_IDENTITY_SCHEMA
    };
    let mut h = Sha256::new();
    for part in [
        schema,
        &identity.sage_version,
        &identity.rust_source_sha256,
        &identity.parser_schema,
        &identity.missingness_schema,
        &identity.generator_provenance_schema,
    ] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part.as_bytes());
    }
    h.update(identity.cache_schema_version.to_le_bytes());
    format!("{:x}", h.finalize())
}

fn raw_digest(identity: &RawExternalPredictionIdentity) -> String {
    let mut h = Sha256::new();
    for part in [
        "sage-raw-external-prediction-fingerprint-v3-layered-provenance",
        &identity.search_fingerprint,
        &identity.generator_execution_settings_sha256,
        &identity.raw_input_sha256,
        &identity.stable_candidate_id_schema,
        &identity.feature_schema,
        &identity.finalizer.digest,
    ] {
        h.update(part.as_bytes());
        h.update([0]);
    }
    h.update(identity.requested_max_rank.to_le_bytes());
    h.update((identity.requested_candidate_count as u64).to_le_bytes());
    format!("{:x}", h.finalize())
}

fn portable_generator(components: &RawGeneratorProvenance) -> Value {
    let source = |s: &Option<RawGeneratorSourceIdentity>| {
        s.as_ref()
            .map(|s| json!({"kind": s.kind, "sha256": s.sha256}))
    };
    let files = |files: &[RawGeneratorFileIdentity]| {
        let mut values = files
            .iter()
            .map(|f| (f.sha256.clone(), f.size_bytes))
            .collect::<Vec<_>>();
        values.sort();
        values
    };
    json!({"schema_version": components.schema_version,
        "generator_settings_sha256": components.generator_settings_sha256,
        "command": source(&components.command), "python": source(&components.python),
        "python_environment": components.python_environment,
        "package_metadata": files(&components.package_metadata),
        "model_components": components.model_components, "model_files": files(&components.model_files)})
}

fn verify_components(
    manifest: &RawExternalPredictionCacheManifest,
    current: &RawExternalPredictionIdentity,
    generator: &RawGeneratorProvenance,
) -> Result<()> {
    let mut out = Vec::new();
    let old = &manifest.identity;
    let mut expected = serde_json::to_value(current)?;
    let mut recorded = serde_json::to_value(old)?;
    for key in ["digest", "finalizer"] {
        expected.as_object_mut().unwrap().remove(key);
        recorded.as_object_mut().unwrap().remove(key);
    }
    differences("/raw_identity", &recorded, &expected, &mut out);
    differences(
        "/manifest/schema_version",
        &json!(manifest.schema_version),
        &json!(3),
        &mut out,
    );
    let mut historical = current.finalizer.clone();
    historical.schema_version = 1;
    historical.sage_version = HISTORICAL_VERSION.into();
    historical.rust_source_sha256 = HISTORICAL_SOURCE.into();
    historical.digest = finalizer_digest(&historical);
    differences(
        "/historical_finalizer",
        &serde_json::to_value(&old.finalizer)?,
        &serde_json::to_value(&historical)?,
        &mut out,
    );
    differences(
        "/current_finalizer/rust_source_sha256",
        &json!(current.finalizer.rust_source_sha256),
        &json!(REVIEWED_CURRENT_SOURCE),
        &mut out,
    );
    differences(
        "/current_finalizer/schema_version",
        &json!(current.finalizer.schema_version),
        &json!(2),
        &mut out,
    );
    differences(
        "/current_finalizer/digest",
        &json!(current.finalizer.digest),
        &json!(finalizer_digest(&current.finalizer)),
        &mut out,
    );
    differences(
        "/raw_identity/digest",
        &json!(old.digest),
        &json!(raw_digest(old)),
        &mut out,
    );
    differences(
        "/generator_components",
        &portable_generator(&manifest.generator_execution.generator_components),
        &portable_generator(generator),
        &mut out,
    );
    differences(
        "/generator_output",
        &serde_json::to_value(&manifest.generator_output)?,
        &serde_json::to_value(&manifest.generator_execution.generator_output)?,
        &mut out,
    );
    differences(
        "/generator_execution/schema_version",
        &json!(manifest.generator_execution.schema_version),
        &json!(1),
        &mut out,
    );
    let execution = &manifest.generator_execution;
    if ![
        RAW_GENERATOR_RUN_PROVENANCE_SCHEMA_V1,
        RAW_GENERATOR_RUN_PROVENANCE_SCHEMA_V2,
    ]
    .contains(&execution.recorded_provenance_schema.as_str())
    {
        differences(
            "/generator_execution/recorded_provenance_schema",
            &json!(execution.recorded_provenance_schema),
            &json!([
                RAW_GENERATOR_RUN_PROVENANCE_SCHEMA_V1,
                RAW_GENERATOR_RUN_PROVENANCE_SCHEMA_V2
            ]),
            &mut out,
        );
    }
    // Reconstruct with historical fields, never current producer defaults.
    let recomputed = verified_raw_generator_execution(
        old,
        execution.generator_components.clone(),
        VerifiedRawGeneratorExecutionArtifacts {
            recorded_provenance_schema: execution.recorded_provenance_schema.clone(),
            candidate_export: execution.candidate_export.clone(),
            generator_configuration: execution.generator_configuration.clone(),
            generator_output: execution.generator_output.clone(),
            provenance_sha256: execution.provenance_sha256.clone(),
            legacy_raw_prediction_fingerprint: execution.legacy_raw_prediction_fingerprint.clone(),
            legacy_generator_settings_sha256: execution.legacy_generator_settings_sha256.clone(),
            legacy_rust_source_sha256: execution.legacy_rust_source_sha256.clone(),
        },
    );
    match recomputed {
        Ok(value) => differences(
            "/generator_execution",
            &serde_json::to_value(execution)?,
            &serde_json::to_value(value)?,
            &mut out,
        ),
        Err(_) => out.push(Difference {
            field: "/generator_execution".into(),
            recorded: json!("invalid component binding"),
            expected: json!("valid historical generator binding"),
        }),
    }
    anyhow::ensure!(
        out.is_empty(),
        "raw-cache compatibility mismatch: {}",
        serde_json::to_string(&out)?
    );
    Ok(())
}

fn require_regular(path: &Path) -> Result<()> {
    anyhow::ensure!(
        std::fs::symlink_metadata(path)?.file_type().is_file(),
        "raw-cache path is not a regular file: {}",
        path.display()
    );
    Ok(())
}

#[derive(Debug)]
pub struct VerifiedExistingRawCache {
    pub directory: PathBuf,
    pub manifest: RawExternalPredictionCacheManifest,
    pub records: Vec<RawExternalPredictionRecord>,
    pub verified_use: RawCacheVerifiedUse,
}

pub fn load_pinned_cache(
    request: &ExternalAnnotationCacheRequest,
    settings: &ExternalFeatureGenerationSettings,
    current: &RawExternalPredictionIdentity,
) -> Result<VerifiedExistingRawCache> {
    anyhow::ensure!(
        request.require_existing && !request.migration_only,
        "pinned raw-cache reuse prohibits generation and migration"
    );
    let reference = request
        .existing_raw_cache
        .as_ref()
        .context("missing explicit raw-cache reference")?;
    reference.validate()?;
    let parent = request.root.join("raw_predictions");
    let directory = parent.join(&reference.fingerprint);
    for path in [&parent, &directory] {
        anyhow::ensure!(
            std::fs::symlink_metadata(path)?.file_type().is_dir(),
            "raw-cache directory is missing, special, or a symlink: {}",
            path.display()
        );
    }
    let manifest_path = raw_cache_manifest_path(&directory);
    let payload_path = directory.join("raw_external_predictions.bin.zst");
    require_regular(&manifest_path)?;
    require_regular(&payload_path)?;
    anyhow::ensure!(
        sha256_file(&manifest_path)? == reference.manifest_sha256,
        "raw-cache /manifest_sha256 differs from frozen reference"
    );
    let manifest: RawExternalPredictionCacheManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    anyhow::ensure!(
        manifest.payload_file == "raw_external_predictions.bin.zst"
            && manifest.payload_sha256 == reference.payload_sha256
            && manifest.identity.digest == reference.fingerprint,
        "raw-cache /payload_file, /payload_sha256 or /fingerprint differs from frozen reference"
    );
    let generator = raw_generator_provenance(settings, &request.root, true)?;
    verify_components(&manifest, current, &generator)?;
    let (verified_manifest, records) =
        load_raw_cache(&directory, &manifest.identity)?.context("incomplete pinned raw cache")?;
    // Full durable loader verifies compressed bytes, decoded payload, candidate
    // ID uniqueness/population, finite observed lanes and missingness state.
    anyhow::ensure!(
        sha256_file(&manifest_path)? == reference.manifest_sha256
            && sha256_file(&payload_path)? == reference.payload_sha256,
        "raw-cache artifacts changed during verification"
    );
    let mut verified_use = RawCacheVerifiedUse {
        schema_version: 1,
        reference: reference.clone(),
        historical_finalizer: manifest.identity.finalizer.clone(),
        current_finalizer: current.finalizer.clone(),
        current_reader_source_sha256: env!("SAGE_RAW_CACHE_READER_SHA256").into(),
        current_binary_sha256: sha256_file(&std::env::current_exe()?)?,
        generator_execution_sha256: manifest.generator_execution.digest.clone(),
        content_fingerprint: manifest.content_fingerprint.clone(),
        complete_population_and_missingness_verified: true,
        digest: String::new(),
    };
    verified_use.digest = verified_use.payload_digest()?;
    Ok(VerifiedExistingRawCache {
        directory,
        manifest: verified_manifest,
        records,
        verified_use,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::external_feature_cache::tests::{complete_features, write_empty_probe};
    use std::collections::HashSet;

    pub(crate) fn install_historical(
        root: &Path,
        settings: &ExternalFeatureGenerationSettings,
        inputs: &[ExternalAnnotationInput],
        records: Vec<RawExternalPredictionRecord>,
    ) -> (
        ExternalAnnotationCacheRequest,
        RawExternalPredictionIdentity,
    ) {
        write_empty_probe(root, settings);
        let current =
            raw_prediction_identity_with_probe_root("search", settings, inputs, 1, root, true)
                .unwrap();
        let mut historical = current.clone();
        historical.finalizer.schema_version = 1;
        historical.finalizer.sage_version = HISTORICAL_VERSION.into();
        historical.finalizer.rust_source_sha256 = HISTORICAL_SOURCE.into();
        historical.finalizer.digest = finalizer_digest(&historical.finalizer);
        historical.digest = raw_digest(&historical);
        let components = raw_generator_provenance(settings, root, true).unwrap();
        let file = GeneratorOutputIdentity {
            sha256: "a".repeat(64),
            size_bytes: 100,
        };
        let execution = verified_raw_generator_execution(
            &historical,
            components,
            VerifiedRawGeneratorExecutionArtifacts {
                recorded_provenance_schema: RAW_GENERATOR_RUN_PROVENANCE_SCHEMA_V2.into(),
                candidate_export: file.clone(),
                generator_configuration: file.clone(),
                generator_output: file,
                provenance_sha256: "b".repeat(64),
                legacy_raw_prediction_fingerprint: None,
                legacy_generator_settings_sha256: None,
                legacy_rust_source_sha256: None,
            },
        )
        .unwrap();
        let directory = raw_cache_directory(root, &historical);
        let (manifest, reused) =
            publish_raw_cache_atomic_with_output(&directory, &historical, records, execution)
                .unwrap();
        assert!(!reused);
        let request = ExternalAnnotationCacheRequest {
            root: root.into(),
            require_existing: true,
            migration_only: false,
            search_space: "+entrapment".into(),
            stage: "moments:annotated".into(),
            analysis_fingerprint: "analysis".into(),
            existing_raw_cache: Some(ExistingRawCacheReference {
                compatibility_contract: CONTRACT.into(),
                fingerprint: historical.digest,
                manifest_sha256: sha256_file(&raw_cache_manifest_path(&directory)).unwrap(),
                payload_sha256: manifest.payload_sha256,
            }),
        };
        (request, current)
    }

    fn fixture() -> (
        PathBuf,
        ExternalFeatureGenerationSettings,
        ExternalAnnotationCacheRequest,
        RawExternalPredictionIdentity,
    ) {
        let root = std::env::temp_dir().join(format!(
            "sage-historical-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let settings = ExternalFeatureGenerationSettings {
            deeplc_calibration_set_size: Some(10),
            ..Default::default()
        };
        let inputs = [ExternalAnnotationInput {
            stable_id: "candidate".into(),
            score: 5.0,
            q_value: Some(0.01),
            pep: Some(0.01),
            retention_time: 12.0,
            ion_mobility: 1.0,
            precursor_mass: 900.0,
            charge: 2,
            rank: 1,
        }];
        let mut unavailable = complete_features();
        unavailable.ms2rescore_ms2pip_pcc = f32::NAN;
        unavailable.ms2rescore_spectral_angle = f32::NAN;
        unavailable.ms2rescore_fragment_intensity_agreement = f32::NAN;
        let (request, current) = install_historical(
            &root,
            &settings,
            &inputs,
            vec![raw_record("candidate".into(), unavailable).unwrap()],
        );
        (root, settings, request, current)
    }

    #[test]
    fn historical_strict_preflight_preserves_missingness_and_exact_replay() {
        let (root, settings, request, current) = fixture();
        let before = request.existing_raw_cache.clone().unwrap();
        let loaded = load_pinned_cache(&request, &settings, &current).unwrap();
        assert_eq!(loaded.records.len(), 1);
        assert!(!loaded.records[0].availability.ms2pip_available);
        assert!(loaded.records[0].availability.deeplc_available);
        assert!(loaded.records[0].features.ms2rescore_ms2pip_pcc.is_nan());
        assert_eq!(
            loaded.verified_use.historical_finalizer.rust_source_sha256,
            HISTORICAL_SOURCE
        );
        assert_ne!(
            loaded.verified_use.current_finalizer,
            loaded.verified_use.historical_finalizer
        );
        let ids = HashSet::from(["candidate".into()]);
        let usage = preflight_existing_cache_root(&request, &settings, "search", &ids, 1).unwrap();
        assert!(usage[0].reused && !usage[0].generation_allowed);
        verify_usage(&usage[0]).unwrap();
        let replay = load_pinned_cache(&request, &settings, &current).unwrap();
        assert_eq!(loaded.verified_use, replay.verified_use);
        assert_eq!(
            before.manifest_sha256,
            sha256_file(&raw_cache_manifest_path(&replay.directory)).unwrap()
        );
        assert_eq!(
            before.payload_sha256,
            sha256_file(&replay.directory.join(&replay.manifest.payload_file)).unwrap()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn historical_component_diagnostics_reject_all_substitutions() {
        let (root, settings, request, current) = fixture();
        let loaded = load_pinned_cache(&request, &settings, &current).unwrap();
        let generator = raw_generator_provenance(&settings, &root, true).unwrap();
        let cases: Vec<(&str, Value)> = vec![
            ("/identity/search_fingerprint", json!("other")),
            ("/identity/raw_input_sha256", json!("other")),
            ("/identity/requested_candidate_count", json!(2)),
            ("/identity/requested_max_rank", json!(2)),
            (
                "/identity/generator_execution_settings_sha256",
                json!("other"),
            ),
            ("/identity/feature_schema", json!("unknown")),
            ("/identity/finalizer/parser_schema", json!("incompatible")),
            ("/identity/finalizer/rust_source_sha256", json!("unknown")),
            ("/schema_version", json!(99)),
            ("/generator_output/sha256", json!("c".repeat(64))),
            (
                "/generator_execution/candidate_export/sha256",
                json!("c".repeat(64)),
            ),
            (
                "/generator_execution/generator_configuration/size_bytes",
                json!(999),
            ),
            (
                "/generator_execution/recorded_provenance_schema",
                json!("unknown"),
            ),
            (
                "/generator_execution/generator_components/python_environment",
                json!("other"),
            ),
            (
                "/generator_execution/generator_components/command",
                json!({"source":"wrapper", "kind":"file", "sha256":"c".repeat(64)}),
            ),
            (
                "/generator_execution/generator_components/model_components",
                json!([{"generator":"ms2pip", "logical_model_name":"other", "relative_filename":"model", "size_bytes":1, "sha256":"c".repeat(64)}]),
            ),
        ];
        for (field, value) in cases {
            let mut changed = serde_json::to_value(&loaded.manifest).unwrap();
            *changed.pointer_mut(field).unwrap() = value;
            let changed = serde_json::from_value(changed).unwrap();
            let error = verify_components(&changed, &current, &generator)
                .unwrap_err()
                .to_string();
            assert!(error.contains("mismatch"), "{field}: {error}");
        }
        let mut changed = current.clone();
        changed.finalizer.rust_source_sha256 = "unreviewed parser".into();
        changed.finalizer.digest = finalizer_digest(&changed.finalizer);
        let error = verify_components(&loaded.manifest, &changed, &generator)
            .unwrap_err()
            .to_string();
        assert!(error.contains("/current_finalizer/rust_source_sha256"));
        let mut changed = loaded.manifest.clone();
        changed.identity.search_fingerprint = "wrong dataset".into();
        changed.identity.raw_input_sha256 = "wrong population".into();
        let error = verify_components(&changed, &current, &generator)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("/raw_identity/search_fingerprint")
                && error.contains("/raw_identity/raw_input_sha256")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pinned_resource_is_explicit_complete_and_immutable() {
        let (root, settings, request, current) = fixture();
        let loaded = load_pinned_cache(&request, &settings, &current).unwrap();
        let mut relaxed = request.clone();
        relaxed.require_existing = false;
        assert!(load_pinned_cache(&relaxed, &settings, &current).is_err());
        let mut unknown = request.clone();
        unknown
            .existing_raw_cache
            .as_mut()
            .unwrap()
            .compatibility_contract = "unknown".into();
        assert!(load_pinned_cache(&unknown, &settings, &current).is_err());
        let mut missing = request.clone();
        missing.existing_raw_cache.as_mut().unwrap().fingerprint = "0".repeat(64);
        assert!(load_pinned_cache(&missing, &settings, &current).is_err());
        let manifest_path = raw_cache_manifest_path(&loaded.directory);
        let original = std::fs::read(&manifest_path).unwrap();
        std::fs::write(&manifest_path, b"{}").unwrap();
        assert!(load_pinned_cache(&request, &settings, &current)
            .unwrap_err()
            .to_string()
            .contains("manifest_sha256"));
        std::fs::write(&manifest_path, original).unwrap();
        let payload = loaded.directory.join(&loaded.manifest.payload_file);
        std::fs::write(&payload, b"truncated").unwrap();
        assert!(load_pinned_cache(&request, &settings, &current).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn diagnostic_source_is_analysis_not_raw_production_identity() {
        let build = include_str!("../build.rs");
        let producer = build.split("let mut reader").next().unwrap();
        assert!(producer.contains("src/external_features.rs"));
        assert!(producer.contains("../sage/src/scoring.rs"));
        assert!(producer.contains("../../Cargo.lock"));
        assert!(!producer.contains("external_feature_diagnostics.rs"));
        assert!(!producer.contains("ml/external_auc.rs"));
        let analysis = build.split("let optimizer_sources").nth(1).unwrap();
        assert!(
            analysis.contains("external_feature_diagnostics.rs")
                && analysis.contains("ml/external_auc.rs")
        );
    }

    #[test]
    fn relocation_is_content_preserving_but_symlink_and_missing_payload_fail_closed() {
        let (root, settings, mut request, current) = fixture();
        let before = load_pinned_cache(&request, &settings, &current).unwrap();
        let moved = root.with_extension("moved");
        std::fs::rename(&root, &moved).unwrap();
        request.root = moved.clone();
        let after = load_pinned_cache(&request, &settings, &current).unwrap();
        assert_eq!(before.verified_use, after.verified_use);
        let payload = after.directory.join(&after.manifest.payload_file);
        let saved = after.directory.join("preserved.payload");
        std::fs::rename(&payload, &saved).unwrap();
        assert!(load_pinned_cache(&request, &settings, &current).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&saved, &payload).unwrap();
            assert!(load_pinned_cache(&request, &settings, &current).is_err());
        }
        std::fs::remove_dir_all(moved).unwrap();
    }
}
