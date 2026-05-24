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
    pub features: ExternalPsmFeatures,
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

    let config = serde_json::json!({
        "engine": engine_name,
        "feature_only_no_decoys": true,
        "psm_file": psm_path,
        "psm_file_type": "sage_decoy_free_external",
        "spectrum_path": spectrum_paths,
        "output_path": output_root,
        "feature_generators": {
            "basic": {},
            "ms2pip": {
                "model": "timsTOF",
                "ms2_tolerance": 0.02
            },
            "deeplc": {
                "deeplc_retrain": false
            },
            "im2deep": {},
            "ionmob": {}
        },
        "rescoring_engine": {}
    });

    std::fs::write(config_path, serde_json::to_vec_pretty(&config)?)?;
    Ok(())
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

    let output = cmd
        .arg("--config")
        .arg(config_path)
        .output()
        .with_context(|| format!("running external feature command {}", command_path))?;

    if !output.stdout.is_empty() {
        log::info!(
            "external feature stdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    if !output.stderr.is_empty() {
        log::warn!(
            "external feature stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    if !output.status.success() {
        bail!(
            "external feature command failed with status {:?}",
            output.status.code()
        );
    }

    Ok(())
}

fn parse_feature_output(
    path: &Path,
) -> Result<HashMap<ExternalFeatureJoinKey, ParsedExternalPsmFeatures>> {
    let mut reader = ReaderBuilder::new().delimiter(b'\t').from_path(path)?;
    let headers = reader.headers()?.clone();

    let mut out = HashMap::new();

    for row in reader.records() {
        let row = row?;

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

        let rank = get_any(&row, &headers, &["rank"])
            .and_then(|x| x.parse::<u32>().ok())
            .unwrap_or(1);

        let mut f = ExternalPsmFeatures::default();

        f.ms2rescore_ms2pip_pcc = get_f32(
            &row,
            &headers,
            &["Ms2pip:Correlation", "ms2pip_correlation", "pcc"],
        );
        f.ms2rescore_spectral_angle =
            get_f32(&row, &headers, &["spectral_angle", "Ms2pip:SpectralAngle"]);
        f.ms2rescore_fragment_intensity_agreement =
            get_f32(&row, &headers, &["fragment_intensity_agreement"]);

        f.ms2rescore_deeplc_predicted_rt = get_f32(
            &row,
            &headers,
            &["DeepLC:PredictedRetentionTime", "predicted_rt"],
        );
        f.ms2rescore_deeplc_calibrated_rt = get_f32(
            &row,
            &headers,
            &["DeepLC:CalibratedRetentionTime", "calibrated_rt"],
        );
        f.ms2rescore_deeplc_rt_error =
            get_f32(&row, &headers, &["DeepLC:RetentionTimeError", "rt_error"]);
        f.ms2rescore_deeplc_abs_rt_error = get_f32(
            &row,
            &headers,
            &["DeepLC:AbsRetentionTimeError", "abs_rt_error"],
        );

        f.tims2rescore_im2deep_predicted_ccs =
            get_f32(&row, &headers, &["IM2Deep:PredictedCCS", "predicted_ccs"]);
        f.tims2rescore_observed_ccs =
            get_f32(&row, &headers, &["IM2Deep:ObservedCCS", "observed_ccs"]);
        f.tims2rescore_abs_ccs_error =
            get_f32(&row, &headers, &["IM2Deep:AbsCCSError", "abs_ccs_error"]);
        f.tims2rescore_pct_ccs_error = get_f32(
            &row,
            &headers,
            &["IM2Deep:PercentualCCSError", "percent_ccs_error"],
        );

        f.tims2rescore_predicted_ion_mobility =
            get_f32(&row, &headers, &["predicted_ion_mobility"]);
        f.tims2rescore_observed_ion_mobility =
            get_f32(&row, &headers, &["ion_mobility", "observed_ion_mobility"]);
        f.tims2rescore_abs_ion_mobility_error =
            get_f32(&row, &headers, &["abs_ion_mobility_error"]);
        f.tims2rescore_pct_ion_mobility_error =
            get_f32(&row, &headers, &["pct_ion_mobility_error"]);

        f.ms2rescore_feature_joined = true;

        out.insert(
            ExternalFeatureJoinKey {
                raw_file,
                spectrum_id,
                modified_peptide,
                charge,
                rank,
            },
            ParsedExternalPsmFeatures { features: f },
        );
    }

    Ok(out)
}

fn join_features(
    features: &mut [DfFeature],
    parsed: HashMap<ExternalFeatureJoinKey, ParsedExternalPsmFeatures>,
    mzml_paths: &[String],
    db: &IndexedDatabase,
) -> Result<()> {
    let mut joined = 0usize;

    for f in features.iter_mut() {
        let key = ExternalFeatureJoinKey {
            raw_file: raw_file_name(mzml_paths, f.core.file_id),
            spectrum_id: f.core.spec_id.clone(),
            modified_peptide: db[f.core.peptide_idx].to_string(),
            charge: f.core.charge,
            rank: f.core.rank,
        };

        if let Some(parsed_features) = parsed.get(&key) {
            f.core.external_features = parsed_features.features;
            joined += 1;
        }
    }

    log::info!(
        "joined external TIMS2/MS2Rescore features onto {}/{} candidate PSMs",
        joined,
        features.len()
    );

    Ok(())
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
