use anyhow::{bail, Context, Result};
use csv::{ReaderBuilder, WriterBuilder};
use sage_core::database::IndexedDatabase;
use sage_core::scoring::{DfFeature, ExternalPsmFeatures};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    mzml_paths: &[String],
    db: &IndexedDatabase,
) -> Result<()> {
    if !settings.enabled {
        return Ok(());
    }

    if !settings.feature_only {
        bail!("external feature generation must run in feature-only mode");
    }

    let result = add_external_features_inner(features, settings, mzml_paths, db);

    match (&settings.fail_policy, result) {
        (_, Ok(())) => Ok(()),

        (ExternalFeatureFailPolicy::Error, Err(e)) => Err(e),

        (ExternalFeatureFailPolicy::WarnAndContinue, Err(e)) => {
            log::warn!(
                "external feature generation failed; continuing without imported features: {e:#}"
            );
            Ok(())
        }

        (ExternalFeatureFailPolicy::Disable, Err(e)) => {
            log::info!("external feature generation disabled after failure: {e:#}");
            Ok(())
        }
    }
}

fn add_external_features_inner(
    features: &mut [DfFeature],
    settings: &ExternalFeatureGenerationSettings,
    mzml_paths: &[String],
    db: &IndexedDatabase,
) -> Result<()> {
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

    export_candidate_table(features, &psm_path, mzml_paths, db, settings.max_rank)?;

    write_feature_config(settings, &psm_path, &output_root, &config_path, mzml_paths)?;

    run_external_process(settings, &config_path)?;

    let parsed = parse_feature_output(&output_tsv)
        .with_context(|| format!("parsing external feature output {}", output_tsv.display()))?;

    join_features(features, parsed, mzml_paths, db)?;

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

    Ok(())
}

fn export_candidate_table(
    features: &[DfFeature],
    path: &Path,
    mzml_paths: &[String],
    db: &IndexedDatabase,
    max_rank: Option<u32>,
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

        // DeepLC/IM2Deep need a confident calibration subset.
        // Use Decoy-Free q/PEP for rank-1 candidates only.
        // Lower-rank null candidates remain exported for feature generation,
        // but they must not become calibration anchors.
        let export_qvalue = if f.core.rank == 1 {
            f.decoy_free_q_value.unwrap_or(1.0)
        } else {
            1.0
        };

        let export_pep = if f.core.rank == 1 {
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
    mzml_paths: &[String],
) -> Result<()> {
    let spectrum_paths: Vec<String> = mzml_paths
        .iter()
        .map(|p| {
            let key = Path::new(p)
                .file_name()
                .and_then(|x| x.to_str())
                .unwrap_or(p);
            settings
                .spectrum_file_mapping
                .get(key)
                .cloned()
                .unwrap_or_else(|| p.clone())
        })
        .collect();

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
    mzml_paths: &[String],
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

fn raw_file_name(mzml_paths: &[String], file_id: usize) -> String {
    mzml_paths
        .get(file_id)
        .and_then(|p| Path::new(p).file_name())
        .and_then(|x| x.to_str())
        .unwrap_or_else(|| mzml_paths.get(file_id).map(String::as_str).unwrap_or(""))
        .to_string()
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
