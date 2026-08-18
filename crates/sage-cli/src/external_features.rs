use anyhow::{bail, Context, Result};
use csv::{ReaderBuilder, WriterBuilder};
use sage_cloudpath::Url;
use sage_core::database::IndexedDatabase;
use sage_core::scoring::{DfFeature, ExternalPsmFeatures};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::candidate_pool::stable_candidate_id;
use crate::external_feature_cache::{
    annotation_identity_with_probe_root, cache_directory, load_cache, load_raw_cache,
    raw_cache_directory, raw_cache_usage, raw_prediction_identity_with_probe_root,
    stage_calibration_identity, write_raw_cache, ExternalAnnotationCacheRequest,
    ExternalAnnotationCacheUsage, ExternalAnnotationInput, ExternalAnnotationRecord,
};
use crate::input::{
    ExternalFeatureEngine, ExternalFeatureFailPolicy, ExternalFeatureGenerationSettings,
    ExternalFeatureUseMode,
};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ExternalFeatureJoinKey {
    pub raw_file: String,
    pub spectrum_id: String,
    pub modified_peptide: String,
    pub charge: u8,
    pub rank: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ParsedExternalPsmFeatures {
    pub psm_id: Option<u64>,
    pub features: ExternalPsmFeatures,
}

#[derive(Clone, Debug, Default)]
pub struct ParsedExternalFeatureTable {
    pub by_psm_id: HashMap<u64, ParsedExternalPsmFeatures>,
    pub by_key: HashMap<ExternalFeatureJoinKey, ParsedExternalPsmFeatures>,
}

pub fn maybe_add_external_features(
    features: &mut [DfFeature],
    settings: &ExternalFeatureGenerationSettings,
    mzml_paths: &[Url],
    db: &IndexedDatabase,
    search_fingerprint: Option<&str>,
    cache_request: Option<&ExternalAnnotationCacheRequest>,
) -> Result<Option<ExternalAnnotationCacheUsage>> {
    if !settings.enabled {
        return Ok(None);
    }

    if !settings.feature_only {
        bail!("external feature generation must run in feature-only mode");
    }

    anyhow::ensure!(
        search_fingerprint.is_some() == cache_request.is_some(),
        "MS2Rescore cache requires both a search fingerprint and a cache root"
    );

    let result = add_external_features_inner(
        features,
        settings,
        mzml_paths,
        db,
        search_fingerprint,
        cache_request,
    );

    if cache_request.is_some_and(|request| request.require_existing || request.migration_only) {
        return result;
    }

    match (&settings.fail_policy, result) {
        (_, Ok(usage)) => Ok(usage),

        (ExternalFeatureFailPolicy::Error, Err(e)) => Err(e),

        (ExternalFeatureFailPolicy::WarnAndContinue, Err(e)) => {
            log::warn!(
                "external feature generation failed; continuing without imported features: {e:#}"
            );
            Ok(None)
        }

        (ExternalFeatureFailPolicy::Disable, Err(e)) => {
            log::info!("external feature generation disabled after failure: {e:#}");
            Ok(None)
        }
    }
}

fn add_external_features_inner(
    features: &mut [DfFeature],
    settings: &ExternalFeatureGenerationSettings,
    mzml_paths: &[Url],
    db: &IndexedDatabase,
    search_fingerprint: Option<&str>,
    cache_request: Option<&ExternalAnnotationCacheRequest>,
) -> Result<Option<ExternalAnnotationCacheUsage>> {
    let requested_max_rank = settings.max_rank.unwrap_or_else(|| {
        features
            .iter()
            .map(|feature| feature.core.rank)
            .max()
            .unwrap_or(0)
    });
    let prepared_cache = match (search_fingerprint, cache_request) {
        (Some(search_fingerprint), Some(request)) => {
            let (inputs, candidate_indices) =
                annotation_inputs(features, db, search_fingerprint, requested_max_rank);
            let raw_identity = raw_prediction_identity_with_probe_root(
                search_fingerprint,
                settings,
                &inputs,
                requested_max_rank,
                &request.root,
                request.require_existing,
            )
            .with_context(|| {
                format!(
                    "strict raw-prediction-cache preflight failed: classification=generator_provenance_unavailable root={} search_space={} candidate_population={} generation_prohibited={}",
                    request.root.display(), request.search_space, search_fingerprint, request.require_existing
                )
            })?;
            let calibration_identity = stage_calibration_identity(&raw_identity, &inputs, request)?;
            let directory = raw_cache_directory(&request.root, &raw_identity);
            match load_raw_cache(&directory, &raw_identity) {
                Ok(Some((manifest, records))) => {
                    apply_cached_annotations(features, &candidate_indices, records)?;
                    log::info!(
                        "raw MS2Rescore prediction cache: reused {}/{} predictions from {} (raw_fingerprint={}, calibration_fingerprint={})",
                        manifest.joined_prediction_count,
                        manifest.prediction_count,
                        directory.display(),
                        raw_identity.digest,
                        calibration_identity.digest
                    );
                    log_external_feature_local_separation(features, db);
                    return Ok(Some(raw_cache_usage(
                        &directory,
                        &manifest,
                        Some(calibration_identity),
                        true,
                        request,
                    )));
                }
                Ok(None) if request.require_existing => {
                    anyhow::bail!(
                        "strict raw-prediction-cache preflight failed: classification=missing_exact root={} search_space={} candidate_population={} expected_raw_fingerprint={} expected_schema={} generation_prohibited=true",
                        request.root.display(),
                        request.search_space,
                        search_fingerprint,
                        raw_identity.digest,
                        raw_identity.schema_version
                    );
                }
                Err(error) if request.require_existing => {
                    anyhow::bail!(
                        "strict raw-prediction-cache preflight failed: classification=invalid_or_incompatible root={} search_space={} candidate_population={} expected_raw_fingerprint={} expected_schema={} generation_prohibited=true: {error:#}",
                        request.root.display(),
                        request.search_space,
                        search_fingerprint,
                        raw_identity.digest,
                        raw_identity.schema_version
                    );
                }
                Err(error) => return Err(error),
                Ok(None) => {}
            }

            // Explicit one-time compatibility path: only migration mode may
            // extract an exact, integrity-valid schema-v2 stage cache into the
            // model-independent raw layer. Normal cache misses keep the
            // established generation behavior; strict mode never writes.
            if request.migration_only {
                let legacy_identity = annotation_identity_with_probe_root(
                    search_fingerprint,
                    settings,
                    &inputs,
                    requested_max_rank,
                    Some(&request.root),
                )?;
                let legacy_directory = cache_directory(&request.root, &legacy_identity);
                if let Some((legacy_manifest, records)) =
                    load_cache(&legacy_directory, &legacy_identity)?
                {
                    apply_cached_annotations(features, &candidate_indices, records.clone())?;
                    let raw_manifest = write_raw_cache(
                        &directory,
                        &raw_identity,
                        records,
                        Some(&legacy_manifest),
                    )?;
                    log::info!(
                        "raw MS2Rescore prediction cache: migrated schema-v2 cache {} to {} (raw_fingerprint={}, calibration_fingerprint={})",
                        legacy_identity.digest,
                        directory.display(),
                        raw_identity.digest,
                        calibration_identity.digest
                    );
                    log_external_feature_local_separation(features, db);
                    return Ok(Some(raw_cache_usage(
                        &directory,
                        &raw_manifest,
                        Some(calibration_identity),
                        false,
                        request,
                    )));
                }
            }

            Some((
                raw_identity,
                calibration_identity,
                directory,
                candidate_indices,
                inputs,
            ))
        }
        (None, None) => None,
        _ => unreachable!("cache arguments were validated by caller"),
    };

    if cache_request.is_some_and(|request| request.migration_only) {
        let request = cache_request.expect("migration-only request exists");
        anyhow::bail!(
            "schema-v2 raw-cache migration failed: classification=missing_exact_legacy_cache root={} search_space={} generation_prohibited=true",
            request.root.display(), request.search_space
        );
    }

    let tmp_dir = settings
        .temp_directory
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sage_external_features"));

    std::fs::create_dir_all(&tmp_dir)
        .with_context(|| format!("creating external feature temp dir {}", tmp_dir.display()))?;

    let psm_path = tmp_dir.join("sage_decoy_free_external_features.input.tsv");
    let config_path = tmp_dir.join("sage_decoy_free_external_features.config.json");
    let output_root = tmp_dir.join("sage_decoy_free_external_features.output");
    let output_tsv = PathBuf::from(format!("{}.psms.tsv", output_root.display()));

    export_candidate_table(
        features,
        &psm_path,
        mzml_paths,
        db,
        settings.max_rank,
        prepared_cache.is_some(),
    )?;

    write_feature_config(settings, &psm_path, &output_root, &config_path, mzml_paths)?;

    // A configured temp directory may contain output from an earlier run. Never
    // allow a successful-but-empty wrapper invocation to make stale annotations
    // appear current.
    if output_tsv.is_file() {
        std::fs::remove_file(&output_tsv).with_context(|| {
            format!(
                "removing stale external feature output {}",
                output_tsv.display()
            )
        })?;
    }

    if let (Some((expected, _, _, _, inputs)), Some(search_fingerprint), Some(cache_request)) =
        (&prepared_cache, search_fingerprint, cache_request)
    {
        let current = raw_prediction_identity_with_probe_root(
            search_fingerprint,
            settings,
            inputs,
            requested_max_rank,
            &cache_request.root,
            false,
        )?;
        anyhow::ensure!(
            current == *expected,
            "annotation generator identity changed after candidate export; refusing to run with stale model/package/wrapper identity"
        );
    }

    run_external_process(settings, &config_path)?;

    let parsed = parse_feature_output(&output_tsv)
        .with_context(|| format!("parsing external feature output {}", output_tsv.display()))?;

    join_features(features, parsed, mzml_paths, db)?;

    let joined = features
        .iter()
        .filter(|feature| feature.core.external_features.ms2rescore_feature_joined)
        .count();
    anyhow::ensure!(
        joined > 0 || features.is_empty(),
        "external feature generator returned no joinable MS2Rescore annotations"
    );

    log_external_feature_local_separation(features, db);

    match settings.use_mode {
        ExternalFeatureUseMode::DiagnosticsOnly => {
            log::info!("external features imported for diagnostics/output only");
        }
        ExternalFeatureUseMode::ScoringCovariates => {
            log::warn!("external feature q-covariates requested; wire this only after diagnostic validation");
        }
        ExternalFeatureUseMode::BoundedDfExperts => {
            log::warn!("external bounded DF expert features requested; not default-enabled");
        }
    }

    if let Some((identity, calibration_identity, directory, candidate_indices, inputs)) =
        prepared_cache
    {
        let current = raw_prediction_identity_with_probe_root(
            search_fingerprint.expect("prepared cache has a search fingerprint"),
            settings,
            &inputs,
            requested_max_rank,
            &cache_request
                .expect("prepared cache has a cache request")
                .root,
            false,
        )?;
        anyhow::ensure!(
            current == identity,
            "annotation generator identity changed during annotation; refusing to cache results under stale model/package/wrapper identity"
        );
        let records = candidate_indices
            .into_iter()
            .map(|(index, stable_id)| ExternalAnnotationRecord {
                stable_id,
                features: features[index].core.external_features,
            })
            .collect();
        let manifest = write_raw_cache(&directory, &identity, records, None)?;
        log::info!(
            "raw MS2Rescore prediction cache: wrote {}/{} joined predictions to {} (raw_fingerprint={}, calibration_fingerprint={})",
            manifest.joined_prediction_count,
            manifest.prediction_count,
            directory.display(),
            identity.digest,
            calibration_identity.digest
        );
        if settings.output_directory.is_none() {
            for intermediate in [&psm_path, &config_path, &output_tsv] {
                if intermediate.is_file() {
                    if let Err(error) = std::fs::remove_file(intermediate) {
                        log::warn!(
                            "could not remove cached MS2Rescore intermediate {}: {error}",
                            intermediate.display()
                        );
                    }
                }
            }
        }
        return Ok(Some(raw_cache_usage(
            &directory,
            &manifest,
            Some(calibration_identity),
            false,
            cache_request.expect("prepared cache has a request"),
        )));
    }

    Ok(None)
}

fn annotation_inputs(
    features: &[DfFeature],
    db: &IndexedDatabase,
    search_fingerprint: &str,
    requested_max_rank: u32,
) -> (Vec<ExternalAnnotationInput>, Vec<(usize, String)>) {
    let mut inputs = Vec::new();
    let mut candidate_indices = Vec::new();
    for (index, feature) in features
        .iter()
        .enumerate()
        .filter(|(_, feature)| feature.core.rank <= requested_max_rank)
    {
        let peptide = db[feature.core.peptide_idx].to_string();
        let stable_id = stable_candidate_id(search_fingerprint, &feature.core, &peptide);
        inputs.push(ExternalAnnotationInput {
            stable_id: stable_id.clone(),
            score: feature.core.hyperscore,
            q_value: feature.decoy_free_q_value,
            pep: feature.decoy_free_pep,
            retention_time: feature.core.rt,
            ion_mobility: feature.core.ims,
            precursor_mass: feature.core.expmass,
            charge: feature.core.charge,
            rank: feature.core.rank,
        });
        candidate_indices.push((index, stable_id));
    }
    (inputs, candidate_indices)
}

fn apply_cached_annotations(
    features: &mut [DfFeature],
    candidate_indices: &[(usize, String)],
    records: Vec<ExternalAnnotationRecord>,
) -> Result<()> {
    anyhow::ensure!(
        records.len() == candidate_indices.len(),
        "MS2Rescore annotation cache candidate-count mismatch"
    );
    let mut by_id = records
        .into_iter()
        .map(|record| (record.stable_id, record.features))
        .collect::<HashMap<_, _>>();
    for (index, stable_id) in candidate_indices {
        let annotation = by_id
            .remove(stable_id)
            .with_context(|| format!("MS2Rescore annotation cache is missing {stable_id}"))?;
        features[*index].core.external_features = annotation;
    }
    anyhow::ensure!(
        by_id.is_empty(),
        "MS2Rescore annotation cache contains candidates not present in the current analysis"
    );
    Ok(())
}

fn export_candidate_table(
    features: &[DfFeature],
    path: &Path,
    mzml_paths: &[Url],
    db: &IndexedDatabase,
    max_rank: Option<u32>,
    model_independent_raw: bool,
) -> Result<()> {
    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(path)?;

    writer.write_record([
        "psm_id",
        "spectrum_id",
        "raw_file",
        "peptidoform",
        "sequence",
        "charge",
        "rank",
        "score",
        "qvalue",
        "pep",
        "retention_time",
        "ion_mobility",
        "precursor_mz",
        "is_decoy",
    ])?;

    let max_rank = max_rank.unwrap_or(u32::MAX);

    for f in features.iter().filter(|f| f.core.rank <= max_rank) {
        let peptide = db[f.core.peptide_idx].to_string();
        let raw_file = raw_file_name(mzml_paths, f.core.file_id);

        // Layered raw inference is deliberately independent of the preliminary
        // statistical model/window. Preserve the historical standalone
        // (non-cache) export contract for callers outside the workflow.
        let export_qvalue = if model_independent_raw {
            1.0
        } else if f.core.rank == 1 {
            f.decoy_free_q_value.unwrap_or(1.0)
        } else {
            1.0
        };
        let export_pep = if model_independent_raw {
            1.0
        } else if f.core.rank == 1 {
            f.decoy_free_pep.unwrap_or(1.0)
        } else {
            1.0
        };

        const PROTON_MASS: f64 = 1.007_276_466_621;
        let charge = f.core.charge as f64;
        let precursor_mz = if charge > 0.0 {
            (f.core.expmass as f64 + charge * PROTON_MASS) / charge
        } else {
            f64::NAN
        };

        writer.write_record([
            f.core.psm_id.to_string(),
            f.core.spec_id.clone(),
            raw_file,
            peptide.clone(),
            peptide,
            f.core.charge.to_string(),
            f.core.rank.to_string(),
            f.core.hyperscore.to_string(),
            export_qvalue.to_string(),
            export_pep.to_string(),
            f.core.rt.to_string(),
            f.core.ims.to_string(),
            precursor_mz.to_string(),
            // No fake decoys. These are all target-only Decoy-Free candidates.
            "false".to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

fn write_feature_config(
    settings: &ExternalFeatureGenerationSettings,
    psm_path: &Path,
    output_root: &Path,
    config_path: &Path,
    mzml_paths: &[Url],
) -> Result<()> {
    let spectrum_paths: Vec<String> = mzml_paths
        .iter()
        .map(|url| {
            let key = raw_file_name_for_url(url);
            if let Some(mapped) = settings.spectrum_file_mapping.get(&key) {
                external_process_path_from_str(mapped)
            } else {
                external_process_path(url)
            }
        })
        .collect::<Result<_>>()?;

    let engine_name = match settings.engine {
        ExternalFeatureEngine::Tims2rescore => "tims2rescore",
        ExternalFeatureEngine::Ms2rescore => "ms2rescore",
    };

    let feature_generators = settings
        .feature_generators
        .clone()
        .unwrap_or_else(|| default_feature_generators(settings));

    let modification_mapping = settings
        .modification_mapping
        .clone()
        .unwrap_or_else(|| serde_json::json!({}));

    let fixed_modifications = settings
        .fixed_modifications
        .clone()
        .unwrap_or_else(|| serde_json::json!({}));

    let config = serde_json::json!({
        "engine": engine_name,
        "feature_only_no_decoys": true,
        "psm_file": psm_path,
        "psm_file_type": "sage_decoy_free_external",
        "spectrum_path": spectrum_paths,
        "output_path": output_root,

        "log_level": settings.log_level.as_deref().unwrap_or("info"),
        "processes": settings.processes.unwrap_or(1),

        "feature_generators": feature_generators,

        // Used by the wrapper to normalize Sage mass-delta peptidoforms before psm-utils/MS2PIP.
        "modification_mapping": modification_mapping,
        "fixed_modifications": fixed_modifications,

        // Explicitly empty. We do not allow Percolator/Mokapot to become statistical authority.
        "rescoring_engine": {}
    });

    std::fs::write(config_path, serde_json::to_vec_pretty(&config)?)?;
    Ok(())
}

fn default_feature_generators(settings: &ExternalFeatureGenerationSettings) -> serde_json::Value {
    let ms2pip_model = settings
        .ms2pip_model
        .as_deref()
        .unwrap_or(match settings.engine {
            ExternalFeatureEngine::Tims2rescore => "timsTOF2024",
            ExternalFeatureEngine::Ms2rescore => "timsTOF2024",
        });

    let ms2pip_ms2_tolerance = settings.ms2pip_ms2_tolerance.unwrap_or(0.02);
    let processes = settings.processes.unwrap_or(1);

    let mut deeplc = serde_json::Map::new();
    deeplc.insert(
        "deeplc_retrain".to_string(),
        serde_json::json!(settings.deeplc_retrain.unwrap_or(false)),
    );

    if let Some(n_epochs) = settings.deeplc_n_epochs {
        deeplc.insert("n_epochs".to_string(), serde_json::json!(n_epochs));
    }

    if let Some(calibration_set_size) = settings.deeplc_calibration_set_size {
        deeplc.insert(
            "calibration_set_size".to_string(),
            serde_json::json!(calibration_set_size),
        );
    }

    serde_json::json!({
        "basic": {},
        "ms2pip": {
            "model": ms2pip_model,
            "ms2_tolerance": ms2pip_ms2_tolerance,
            "processes": processes
        },
        "deeplc": serde_json::Value::Object(deeplc),
        "im2deep": {}
    })
}

fn run_external_process(
    settings: &ExternalFeatureGenerationSettings,
    config_path: &Path,
) -> Result<()> {
    let command_path = settings
        .command_path
        .as_ref()
        .context("external_features.command_path is required")?;

    let mut cmd = if let Some(py) = &settings.python_executable {
        let mut c = Command::new(py);
        c.arg(command_path);
        c
    } else {
        Command::new(command_path)
    };

    let status = cmd
        .arg("--config")
        .arg(config_path)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .with_context(|| format!("running external feature command {}", command_path))?;

    if !status.success() {
        bail!(
            "external feature command failed with status {:?}",
            status.code()
        );
    }

    Ok(())
}

fn parse_feature_output(path: &Path) -> Result<ParsedExternalFeatureTable> {
    let mut reader = ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    let headers = reader.headers()?.clone();

    log::info!(
        "external feature output has {} columns: {:?}",
        headers.len(),
        headers.iter().collect::<Vec<_>>()
    );

    let mut out = ParsedExternalFeatureTable::default();
    let mut row_count = 0usize;

    for row in reader.records() {
        let row = row?;
        row_count += 1;

        let psm_id =
            get_any(&row, &headers, &["psm_id", "sage_psm_id"]).and_then(|x| x.parse::<u64>().ok());

        let raw_file =
            get_any(&row, &headers, &["raw_file", "run", "filename", "source"]).unwrap_or_default();

        let spectrum_id =
            get_any(&row, &headers, &["spectrum_id", "spec_id", "PSMId"]).unwrap_or_default();

        let modified_peptide = get_any(
            &row,
            &headers,
            &["peptidoform", "modified_peptide", "peptide"],
        )
        .unwrap_or_default();

        let charge = get_any(&row, &headers, &["charge"])
            .and_then(|x| x.parse::<u8>().ok())
            .unwrap_or(0);

        let rank = get_any(&row, &headers, &["rank", "sage_rank"])
            .and_then(|x| x.parse::<u32>().ok())
            .unwrap_or(1);

        let mut f = ExternalPsmFeatures::default();

        f.ms2rescore_ms2pip_pcc = get_f32(
            &row,
            &headers,
            &[
                "spec_pearson",
                "spec_pearson_norm",
                "Ms2pip:Correlation",
                "ms2pip_correlation",
                "ms2pip_corr",
                "pcc",
                "correlation",
            ],
        );

        f.ms2rescore_spectral_angle = get_f32(
            &row,
            &headers,
            &[
                "cos",
                "cos_norm",
                "dotprod",
                "dotprod_norm",
                "spectral_angle",
                "Ms2pip:SpectralAngle",
                "ms2pip_spectral_angle",
                "spectral_angle_similarity",
            ],
        );

        f.ms2rescore_fragment_intensity_agreement = get_f32(
            &row,
            &headers,
            &[
                "dotprod",
                "dotprod_norm",
                "cos",
                "cos_norm",
                "spec_pearson",
                "spec_pearson_norm",
                "fragment_intensity_agreement",
                "ms2pip_fragment_intensity_agreement",
            ],
        );

        f.ms2rescore_deeplc_predicted_rt = get_f32(
            &row,
            &headers,
            &[
                "predicted_retention_time",
                "predicted_retention_time_best",
                "DeepLC:PredictedRetentionTime",
                "deeplc_predicted_rt",
                "predicted_rt",
                "rt_pred",
            ],
        );

        f.ms2rescore_deeplc_calibrated_rt = get_f32(
            &row,
            &headers,
            &[
                "observed_retention_time",
                "observed_retention_time_best",
                "DeepLC:CalibratedRetentionTime",
                "deeplc_calibrated_rt",
                "calibrated_rt",
            ],
        );

        f.ms2rescore_deeplc_rt_error = get_f32(
            &row,
            &headers,
            &[
                "rt_diff",
                "rt_diff_best",
                "DeepLC:RetentionTimeError",
                "deeplc_rt_error",
                "rt_error",
                "delta_rt",
            ],
        );

        let rt_err = get_f32(
            &row,
            &headers,
            &[
                "rt_diff",
                "rt_diff_best",
                "DeepLC:AbsRetentionTimeError",
                "deeplc_abs_rt_error",
                "abs_rt_error",
                "abs_delta_rt",
            ],
        );

        f.ms2rescore_deeplc_abs_rt_error = if rt_err.is_finite() {
            rt_err.abs()
        } else {
            f32::NAN
        };

        f.tims2rescore_im2deep_predicted_ccs = get_f32(
            &row,
            &headers,
            &[
                "ccs_predicted_im2deep",
                "IM2Deep:PredictedCCS",
                "im2deep_predicted_ccs",
                "predicted_ccs",
                "ccs_predicted",
            ],
        );

        f.tims2rescore_observed_ccs = get_f32(
            &row,
            &headers,
            &[
                "ccs_observed_im2deep",
                "IM2Deep:ObservedCCS",
                "im2deep_observed_ccs",
                "observed_ccs",
                "ccs_observed",
            ],
        );

        f.tims2rescore_abs_ccs_error = get_f32(
            &row,
            &headers,
            &[
                "abs_ccs_error_im2deep",
                "IM2Deep:AbsCCSError",
                "im2deep_abs_ccs_error",
                "abs_ccs_error",
                "ccs_error_abs",
            ],
        );

        f.tims2rescore_pct_ccs_error = get_f32(
            &row,
            &headers,
            &[
                "perc_ccs_error_im2deep",
                "IM2Deep:PercentualCCSError",
                "im2deep_pct_ccs_error",
                "percent_ccs_error",
                "pct_ccs_error",
                "ccs_error_percent",
            ],
        );

        f.tims2rescore_predicted_ion_mobility = get_f32(
            &row,
            &headers,
            &[
                "predicted_ion_mobility",
                "ion_mobility_predicted",
                "im_predicted",
            ],
        );

        f.tims2rescore_observed_ion_mobility = get_f32(
            &row,
            &headers,
            &[
                "ion_mobility",
                "observed_ion_mobility",
                "ion_mobility_observed",
                "im_observed",
            ],
        );

        f.tims2rescore_abs_ion_mobility_error = get_f32(
            &row,
            &headers,
            &[
                "abs_ion_mobility_error",
                "ion_mobility_error_abs",
                "abs_im_error",
            ],
        );

        f.tims2rescore_pct_ion_mobility_error = get_f32(
            &row,
            &headers,
            &[
                "pct_ion_mobility_error",
                "ion_mobility_error_percent",
                "pct_im_error",
            ],
        );

        f.ms2rescore_feature_joined = true;

        let parsed = ParsedExternalPsmFeatures {
            psm_id,
            features: f,
        };

        if let Some(id) = psm_id {
            out.by_psm_id.insert(id, parsed.clone());
        }

        out.by_key.insert(
            ExternalFeatureJoinKey {
                raw_file,
                spectrum_id,
                modified_peptide,
                charge,
                rank,
            },
            parsed,
        );
    }

    log::info!(
        "parsed {} external feature rows: {} keyed by psm_id, {} keyed by compound key",
        row_count,
        out.by_psm_id.len(),
        out.by_key.len()
    );

    Ok(out)
}

fn join_features(
    features: &mut [DfFeature],
    parsed: ParsedExternalFeatureTable,
    mzml_paths: &[Url],
    db: &IndexedDatabase,
) -> Result<()> {
    let mut joined = 0usize;
    let mut joined_by_psm_id = 0usize;
    let mut joined_by_key = 0usize;
    let mut missed_examples = Vec::new();

    for f in features.iter_mut() {
        let psm_id = f.core.psm_id as u64;

        if let Some(parsed_features) = parsed.by_psm_id.get(&psm_id) {
            f.core.external_features = parsed_features.features;
            joined += 1;
            joined_by_psm_id += 1;
            continue;
        }

        let key = ExternalFeatureJoinKey {
            raw_file: raw_file_name(mzml_paths, f.core.file_id),
            spectrum_id: f.core.spec_id.clone(),
            modified_peptide: db[f.core.peptide_idx].to_string(),
            charge: f.core.charge,
            rank: f.core.rank,
        };

        if let Some(parsed_features) = parsed.by_key.get(&key) {
            f.core.external_features = parsed_features.features;
            joined += 1;
            joined_by_key += 1;
        } else if missed_examples.len() < 5 {
            missed_examples.push(format!("{:?}", key));
        }
    }

    log::info!(
        "joined external TIMS2/MS2Rescore features onto {}/{} candidate PSMs (by_psm_id={}, by_compound_key={})",
        joined,
        features.len(),
        joined_by_psm_id,
        joined_by_key
    );

    if joined == 0 {
        log::warn!(
            "external feature join matched zero rows; first missed keys: {:?}",
            missed_examples
        );
    }

    Ok(())
}

fn log_external_feature_local_separation(features: &[DfFeature], db: &IndexedDatabase) {
    let joined = features
        .iter()
        .filter(|f| f.core.external_features.ms2rescore_feature_joined)
        .count();

    if joined == 0 {
        log::warn!("external feature local diagnostics skipped: no joined features");
        return;
    }

    log::info!(
        "external feature local diagnostics: joined_features={}/{}",
        joined,
        features.len()
    );

    log_external_feature_one_global(features, db, "ms2rescore_ms2pip_pcc", true, |f| {
        f.core.external_features.ms2rescore_ms2pip_pcc as f64
    });

    log_external_feature_one_global(features, db, "ms2rescore_spectral_angle", true, |f| {
        f.core.external_features.ms2rescore_spectral_angle as f64
    });

    log_external_feature_one_global(features, db, "ms2rescore_deeplc_abs_rt_error", false, |f| {
        f.core.external_features.ms2rescore_deeplc_abs_rt_error as f64
    });

    log_external_feature_one_global(features, db, "tims2rescore_abs_ccs_error", false, |f| {
        f.core.external_features.tims2rescore_abs_ccs_error as f64
    });

    log_external_feature_one_global(features, db, "tims2rescore_pct_ccs_error", false, |f| {
        f.core.external_features.tims2rescore_pct_ccs_error as f64
    });

    log_external_feature_by_matched_peaks_bin(
        features,
        db,
        "ms2rescore_deeplc_abs_rt_error",
        false,
        |f| f.core.external_features.ms2rescore_deeplc_abs_rt_error as f64,
    );

    log_external_feature_by_matched_peaks_bin(
        features,
        db,
        "tims2rescore_pct_ccs_error",
        false,
        |f| f.core.external_features.tims2rescore_pct_ccs_error as f64,
    );

    log_external_feature_by_matched_peaks_bin(features, db, "ms2rescore_ms2pip_pcc", true, |f| {
        f.core.external_features.ms2rescore_ms2pip_pcc as f64
    });
}

fn log_external_feature_one_global<F>(
    features: &[DfFeature],
    db: &IndexedDatabase,
    name: &str,
    higher_is_better: bool,
    getter: F,
) where
    F: Fn(&DfFeature) -> f64,
{
    let mut reference = Vec::new();
    let mut entrapment = Vec::new();

    for f in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.external_features.ms2rescore_feature_joined)
    {
        let x = getter(f);
        if !x.is_finite() {
            continue;
        }

        let proteins = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        if external_feature_is_entrapment(&proteins) {
            entrapment.push(x);
        } else {
            reference.push(x);
        }
    }

    if reference.len() < 10 || entrapment.len() < 10 {
        log::info!(
            "external feature diagnostic {name}: insufficient rank1 reference/entrapment values reference={} entrapment={}",
            reference.len(),
            entrapment.len()
        );
        return;
    }

    let auc = external_feature_auc(&reference, &entrapment, higher_is_better);

    log::info!(
        "external feature diagnostic {name}: reference_n={} entrapment_n={} reference_median={:.6} entrapment_median={:.6} auc_ref_vs_ent={:.4} higher_is_better={}",
        reference.len(),
        entrapment.len(),
        external_feature_median(reference),
        external_feature_median(entrapment),
        auc,
        higher_is_better
    );
}

fn log_external_feature_by_matched_peaks_bin<F>(
    features: &[DfFeature],
    db: &IndexedDatabase,
    name: &str,
    higher_is_better: bool,
    getter: F,
) where
    F: Fn(&DfFeature) -> f64,
{
    let bins: &[(u32, u32, &str)] = &[
        (0, 4, "matched_peaks_0_4"),
        (5, 7, "matched_peaks_5_7"),
        (8, 12, "matched_peaks_8_12"),
        (13, u32::MAX, "matched_peaks_13_plus"),
    ];

    for &(lo, hi, label) in bins {
        let mut reference = Vec::new();
        let mut entrapment = Vec::new();

        for f in features
            .iter()
            .filter(|f| f.core.rank == 1 && f.core.external_features.ms2rescore_feature_joined)
        {
            let mp = f.core.matched_peaks;
            if mp < lo || mp > hi {
                continue;
            }

            let x = getter(f);
            if !x.is_finite() {
                continue;
            }

            let proteins = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            if external_feature_is_entrapment(&proteins) {
                entrapment.push(x);
            } else {
                reference.push(x);
            }
        }

        if reference.len() < 10 || entrapment.len() < 10 {
            log::info!(
                "external feature diagnostic {name} bin={label}: insufficient values reference={} entrapment={}",
                reference.len(),
                entrapment.len()
            );
            continue;
        }

        let auc = external_feature_auc(&reference, &entrapment, higher_is_better);

        log::info!(
            "external feature diagnostic {name} bin={label}: reference_n={} entrapment_n={} reference_median={:.6} entrapment_median={:.6} auc_ref_vs_ent={:.4} higher_is_better={}",
            reference.len(),
            entrapment.len(),
            external_feature_median(reference),
            external_feature_median(entrapment),
            auc,
            higher_is_better
        );
    }
}

fn external_feature_is_entrapment(proteins: &str) -> bool {
    let u = proteins.to_ascii_uppercase();

    u.contains("ENTRAP")
        || u.contains("FOREIGN")
        || u.contains("ARATH")
        || u.contains("YEAST")
        || u.contains("CAEEL")
        || u.contains("DROME")
        || u.contains("ECOLI")
        || u.contains("HUMAN")
        || u.contains("RAT")
}

fn external_feature_median(mut xs: Vec<f64>) -> f64 {
    xs.retain(|x| x.is_finite());
    if xs.is_empty() {
        return f64::NAN;
    }

    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn external_feature_auc(reference: &[f64], entrapment: &[f64], higher_is_better: bool) -> f64 {
    if reference.is_empty() || entrapment.is_empty() {
        return f64::NAN;
    }

    let mut wins = 0.0f64;
    let mut total = 0.0f64;

    for &r in reference {
        if !r.is_finite() {
            continue;
        }

        for &e in entrapment {
            if !e.is_finite() {
                continue;
            }

            total += 1.0;

            if higher_is_better {
                if r > e {
                    wins += 1.0;
                } else if r == e {
                    wins += 0.5;
                }
            } else if r < e {
                wins += 1.0;
            } else if r == e {
                wins += 0.5;
            }
        }
    }

    if total <= 0.0 {
        f64::NAN
    } else {
        wins / total
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::external_feature_cache::{write_cache, write_raw_cache};
    use sage_core::database::PeptideIx;
    use sage_core::peptide::Peptide;
    use sage_core::scoring::FeatureCore;

    fn complete_annotation() -> ExternalPsmFeatures {
        ExternalPsmFeatures {
            ms2rescore_ms2pip_pcc: 0.75,
            ms2rescore_spectral_angle: 0.7,
            ms2rescore_fragment_intensity_agreement: 0.6,
            ms2rescore_deeplc_predicted_rt: 12.5,
            ms2rescore_deeplc_calibrated_rt: 12.0,
            ms2rescore_deeplc_rt_error: 0.5,
            ms2rescore_deeplc_abs_rt_error: 0.5,
            tims2rescore_observed_ion_mobility: 0.0,
            ms2rescore_feature_joined: true,
            ..ExternalPsmFeatures::default()
        }
    }

    #[test]
    fn cached_annotations_join_only_by_stable_candidate_id() {
        let mut features = vec![FeatureCore::default().to_df()];
        let mut annotation = ExternalPsmFeatures::default();
        annotation.ms2rescore_feature_joined = true;
        annotation.ms2rescore_ms2pip_pcc = 0.75;
        apply_cached_annotations(
            &mut features,
            &[(0, "stable-candidate".into())],
            vec![ExternalAnnotationRecord {
                stable_id: "stable-candidate".into(),
                features: annotation,
            }],
        )
        .unwrap();
        assert!(features[0].core.external_features.ms2rescore_feature_joined);
        assert_eq!(
            features[0].core.external_features.ms2rescore_ms2pip_pcc,
            0.75
        );

        assert!(apply_cached_annotations(
            &mut features,
            &[(0, "different-candidate".into())],
            vec![ExternalAnnotationRecord {
                stable_id: "stable-candidate".into(),
                features: annotation,
            }],
        )
        .is_err());
    }

    #[test]
    fn layered_export_is_neutral_while_standalone_export_remains_compatible() {
        let root = std::env::temp_dir().join(format!(
            "sage-layered-export-contract-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let database = IndexedDatabase {
            peptides: vec![Peptide {
                sequence: std::sync::Arc::from(&b"PEPTIDE"[..]),
                ..Peptide::default()
            }],
            ..IndexedDatabase::default()
        };
        let mut core = FeatureCore::default();
        core.peptide_idx = PeptideIx(0);
        core.spec_id = "scan=1".into();
        core.rank = 1;
        core.charge = 2;
        core.expmass = 900.0;
        let mut feature = core.to_df();
        feature.decoy_free_q_value = Some(0.01);
        feature.decoy_free_pep = Some(0.02);

        let standalone = root.join("standalone.tsv");
        let layered = root.join("layered.tsv");
        export_candidate_table(
            std::slice::from_ref(&feature),
            &standalone,
            &[],
            &database,
            Some(1),
            false,
        )
        .unwrap();
        export_candidate_table(&[feature], &layered, &[], &database, Some(1), true).unwrap();

        let read_q_pep = |path: &Path| {
            let mut reader = ReaderBuilder::new()
                .delimiter(b'\t')
                .from_path(path)
                .unwrap();
            let headers = reader.headers().unwrap().clone();
            let q = headers.iter().position(|value| value == "qvalue").unwrap();
            let pep = headers.iter().position(|value| value == "pep").unwrap();
            let row = reader.records().next().unwrap().unwrap();
            (row[q].to_string(), row[pep].to_string())
        };
        assert_eq!(read_q_pep(&standalone), ("0.01".into(), "0.02".into()));
        assert_eq!(read_q_pep(&layered), ("1".into(), "1".into()));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn strict_stage_rejects_another_calibration_cache_before_export() {
        let root = std::env::temp_dir().join(format!(
            "sage-stage-local-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let temp_output = root.join("must-not-be-created");
        let database = IndexedDatabase {
            peptides: vec![Peptide {
                sequence: std::sync::Arc::from(&b"PEPTIDE"[..]),
                ..Peptide::default()
            }],
            ..IndexedDatabase::default()
        };
        let mut core = FeatureCore::default();
        core.peptide_idx = PeptideIx(0);
        core.spec_id = "scan=1".into();
        core.rank = 1;
        core.charge = 2;
        let mut features = vec![core.to_df()];
        features[0].decoy_free_q_value = Some(0.01);
        features[0].decoy_free_pep = Some(0.02);

        let mut settings = ExternalFeatureGenerationSettings {
            enabled: true,
            max_rank: Some(1),
            deeplc_calibration_set_size: Some(10),
            temp_directory: Some(temp_output.display().to_string()),
            ..ExternalFeatureGenerationSettings::default()
        };
        let (mut wrong_inputs, _) = annotation_inputs(&features, &database, "search", 1);
        wrong_inputs[0].q_value = Some(0.5);
        let wrong_identity =
            annotation_identity_with_probe_root("search", &settings, &wrong_inputs, 1, Some(&root))
                .unwrap();
        let wrong_directory = cache_directory(&root, &wrong_identity);
        let mut annotation = ExternalPsmFeatures::default();
        annotation.ms2rescore_feature_joined = true;
        write_cache(
            &wrong_directory,
            &wrong_identity,
            vec![ExternalAnnotationRecord {
                stable_id: wrong_inputs[0].stable_id.clone(),
                features: annotation,
            }],
        )
        .unwrap();

        // Changing execution-only paths cannot alter identity. Keep the
        // generator probe durable while ensuring no export directory exists.
        settings.output_directory = Some(root.join("unused-output").display().to_string());
        let request = ExternalAnnotationCacheRequest {
            root: root.clone(),
            require_existing: true,
            search_space: "+entrapment".into(),
            stage: "moments:ms2rescore".into(),
            analysis_fingerprint: "analysis".into(),
            migration_only: false,
        };
        let error = add_external_features_inner(
            &mut features,
            &settings,
            &[],
            &database,
            Some("search"),
            Some(&request),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("missing_exact"));
        assert!(error.contains("generation_prohibited=true"));
        assert!(!temp_output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn strict_stage_loads_shared_raw_cache_and_derives_calibration_identity() {
        let root = std::env::temp_dir().join(format!(
            "sage-stage-local-exact-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let temp_output = root.join("must-not-be-created");
        let database = IndexedDatabase {
            peptides: vec![Peptide {
                sequence: std::sync::Arc::from(&b"PEPTIDE"[..]),
                ..Peptide::default()
            }],
            ..IndexedDatabase::default()
        };
        let mut core = FeatureCore::default();
        core.peptide_idx = PeptideIx(0);
        core.spec_id = "scan=1".into();
        core.rank = 1;
        core.charge = 2;
        let mut features = vec![core.to_df()];
        features[0].decoy_free_q_value = Some(0.01);
        features[0].decoy_free_pep = Some(0.02);
        let settings = ExternalFeatureGenerationSettings {
            enabled: true,
            max_rank: Some(1),
            deeplc_calibration_set_size: Some(10),
            temp_directory: Some(temp_output.display().to_string()),
            ..ExternalFeatureGenerationSettings::default()
        };
        let (inputs, _) = annotation_inputs(&features, &database, "search", 1);
        let identity =
            raw_prediction_identity_with_probe_root("search", &settings, &inputs, 1, &root, false)
                .unwrap();
        let directory = raw_cache_directory(&root, &identity);
        let annotation = complete_annotation();
        write_raw_cache(
            &directory,
            &identity,
            vec![ExternalAnnotationRecord {
                stable_id: inputs[0].stable_id.clone(),
                features: annotation,
            }],
            None,
        )
        .unwrap();
        let request = ExternalAnnotationCacheRequest {
            root: root.clone(),
            require_existing: true,
            search_space: "+entrapment".into(),
            stage: "moments:ms2rescore".into(),
            analysis_fingerprint: "analysis".into(),
            migration_only: false,
        };
        let usage = add_external_features_inner(
            &mut features,
            &settings,
            &[],
            &database,
            Some("search"),
            Some(&request),
        )
        .unwrap()
        .unwrap();
        assert_eq!(usage.raw_prediction_cache_fingerprint, identity.digest);
        assert_eq!(usage.preflight_result, "validated_raw_prediction_cache");
        assert_ne!(usage.annotation_fingerprint, identity.digest);
        assert!(usage.stage_calibration_identity.is_some());
        assert!(usage.reused);
        assert_eq!(
            features[0].core.external_features.ms2rescore_ms2pip_pcc,
            0.75
        );
        assert!(!temp_output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn schema_v2_migration_seeds_one_raw_cache_for_multiple_models() {
        let root = std::env::temp_dir().join(format!(
            "sage-layered-migration-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let temp_output = root.join("must-not-be-created");
        let database = IndexedDatabase {
            peptides: vec![Peptide {
                sequence: std::sync::Arc::from(&b"PEPTIDE"[..]),
                ..Peptide::default()
            }],
            ..IndexedDatabase::default()
        };
        let mut core = FeatureCore::default();
        core.peptide_idx = PeptideIx(0);
        core.spec_id = "scan=1".into();
        core.rank = 1;
        core.charge = 2;
        let mut features = vec![core.to_df()];
        features[0].decoy_free_q_value = Some(0.01);
        features[0].decoy_free_pep = Some(0.02);
        let settings = ExternalFeatureGenerationSettings {
            enabled: true,
            max_rank: Some(1),
            deeplc_calibration_set_size: Some(10),
            temp_directory: Some(temp_output.display().to_string()),
            ..ExternalFeatureGenerationSettings::default()
        };
        let (inputs, _) = annotation_inputs(&features, &database, "search", 1);
        let legacy =
            annotation_identity_with_probe_root("search", &settings, &inputs, 1, Some(&root))
                .unwrap();
        write_cache(
            &cache_directory(&root, &legacy),
            &legacy,
            vec![ExternalAnnotationRecord {
                stable_id: inputs[0].stable_id.clone(),
                features: complete_annotation(),
            }],
        )
        .unwrap();
        let first_request = ExternalAnnotationCacheRequest {
            root: root.clone(),
            require_existing: false,
            search_space: "+entrapment".into(),
            stage: "moments:ms2rescore".into(),
            analysis_fingerprint: "moments-analysis".into(),
            migration_only: true,
        };
        let first = add_external_features_inner(
            &mut features,
            &settings,
            &[],
            &database,
            Some("search"),
            Some(&first_request),
        )
        .unwrap()
        .unwrap();
        assert!(first.migrated_from_schema_v2);
        assert!(!first.reused);
        assert!(!temp_output.exists());

        features[0].decoy_free_q_value = Some(0.9);
        features[0].decoy_free_pep = Some(0.8);
        let second_request = ExternalAnnotationCacheRequest {
            require_existing: true,
            stage: "mle:ms2rescore".into(),
            analysis_fingerprint: "mle-analysis".into(),
            ..first_request
        };
        let second = add_external_features_inner(
            &mut features,
            &settings,
            &[],
            &database,
            Some("search"),
            Some(&second_request),
        )
        .unwrap()
        .unwrap();
        assert!(second.reused);
        assert_eq!(
            first.raw_prediction_cache_fingerprint,
            second.raw_prediction_cache_fingerprint
        );
        assert_ne!(first.annotation_fingerprint, second.annotation_fingerprint);
        assert!(!temp_output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}

fn raw_file_name(mzml_paths: &[Url], file_id: usize) -> String {
    mzml_paths
        .get(file_id)
        .map(raw_file_name_for_url)
        .unwrap_or_default()
}

/// Return the same basename that the pre-URL integration used for join keys.
/// Local file URLs are converted back to paths first so percent-encoded names
/// (for example, spaces) do not leak into the exported `raw_file` column.
fn raw_file_name_for_url(url: &Url) -> String {
    if url.scheme() == "file" {
        if let Ok(path) = url.to_file_path() {
            if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                return name.to_string();
            }
        }
    }

    url.path_segments()
        .and_then(|mut segments| segments.next_back())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| url.as_str())
        .to_string()
}

/// Convert an internal URL to the spelling expected by an external process.
/// Local `file://` URLs must cross the process boundary as native filesystem
/// paths; remote object-store URLs remain URLs for tools that support them.
fn external_process_path(url: &Url) -> Result<String> {
    if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("invalid local file URL: {url}"))?;
        return Ok(path.to_string_lossy().into_owned());
    }

    Ok(url.as_str().to_string())
}

/// Apply the same boundary conversion to an explicitly configured spectrum
/// mapping while preserving ordinary local paths and Windows drive paths.
fn external_process_path_from_str(path: &str) -> Result<String> {
    match sage_cloudpath::try_parse_url(path) {
        Some(url) => external_process_path(&url),
        None => Ok(path.to_string()),
    }
}

fn get_any(row: &csv::StringRecord, headers: &csv::StringRecord, names: &[&str]) -> Option<String> {
    for name in names {
        if let Some(i) = headers.iter().position(|h| h == *name) {
            return row.get(i).map(|x| x.to_string());
        }
    }
    None
}

fn get_f32(row: &csv::StringRecord, headers: &csv::StringRecord, names: &[&str]) -> f32 {
    get_any(row, headers, names)
        .and_then(|x| x.parse::<f32>().ok())
        .unwrap_or(f32::NAN)
}

#[cfg(test)]
mod path_tests {
    use super::*;

    #[test]
    fn local_url_round_trips_at_external_process_boundary() {
        let path = std::env::current_dir().unwrap().join("sample name.mzML");
        let url = Url::from_file_path(&path).unwrap();

        assert_eq!(
            external_process_path(&url).unwrap(),
            path.to_string_lossy().into_owned()
        );
        assert_eq!(raw_file_name_for_url(&url), "sample name.mzML");
    }

    #[test]
    fn mapped_file_url_round_trips_but_plain_paths_are_unchanged() {
        let path = std::env::current_dir().unwrap().join("mapped sample.mzML");
        let url = Url::from_file_path(&path).unwrap();

        assert_eq!(
            external_process_path_from_str(url.as_str()).unwrap(),
            path.to_string_lossy().into_owned()
        );
        assert_eq!(
            external_process_path_from_str("relative/sample.mzML").unwrap(),
            "relative/sample.mzML"
        );
    }

    #[test]
    fn remote_urls_keep_url_and_basename_spelling() {
        let url = Url::parse("s3://bucket/prefix/sample.mzML").unwrap();

        assert_eq!(external_process_path(&url).unwrap(), url.as_str());
        assert_eq!(raw_file_name_for_url(&url), "sample.mzML");
    }
}
