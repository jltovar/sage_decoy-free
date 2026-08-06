use crate::provenance::write_json_atomic;
use crate::validation::{
    expert_quality_gates, parity_comparisons, stage_comparisons, summarize_run,
    tdc_benchmark_comparisons, transfer_stability, EffectiveRatios, ExpertQualityGate,
    ParityComparison, ParityPair, RunValidationSummary, StageComparison, TdcBenchmarkComparison,
    TransferStability, ValidationRun,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationAuditManifest {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    pub name: String,
    pub output_root: PathBuf,
    pub effective_ratios: EffectiveRatios,
    #[serde(default = "default_fdr")]
    pub fdr_threshold: f64,
    #[serde(default = "default_transfer_loss")]
    pub maximum_transfer_fraction_loss: f64,
    #[serde(default = "default_minimum_entrapment_peptides")]
    pub minimum_entrapment_peptides_for_stable_estimate: usize,
    #[serde(default = "default_validation_layer")]
    pub validation_layer: String,
    #[serde(default)]
    pub parity_pairs: Vec<ParityPair>,
    #[serde(default = "default_parity_tolerance")]
    pub maximum_parity_fraction_difference: f64,
    #[serde(default)]
    pub tdc_reference_method: Option<String>,
    pub runs: Vec<ValidationRun>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationAuditReport {
    pub schema_version: u32,
    pub name: String,
    pub summaries: Vec<RunValidationSummary>,
    pub missing_runs: Vec<ValidationRun>,
    pub transfer_stability: Vec<TransferStability>,
    pub stage_comparisons: Vec<StageComparison>,
    pub expert_quality_gates: Vec<ExpertQualityGate>,
    pub parity_comparisons: Vec<ParityComparison>,
    pub tdc_benchmarks: Vec<TdcBenchmarkComparison>,
}

fn default_schema() -> u32 {
    1
}

fn default_fdr() -> f64 {
    0.01
}

fn default_transfer_loss() -> f64 {
    0.20
}

fn default_minimum_entrapment_peptides() -> usize {
    3
}

fn default_validation_layer() -> String {
    "level4".into()
}

fn default_parity_tolerance() -> f64 {
    0.001
}

pub fn execute_validation_audit(manifest_path: &Path) -> Result<ValidationAuditReport> {
    let bytes = std::fs::read(manifest_path).with_context(|| {
        format!(
            "failed to read validation audit {}",
            manifest_path.display()
        )
    })?;
    let manifest: ValidationAuditManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid validation audit {}", manifest_path.display()))?;
    anyhow::ensure!(manifest.schema_version == 1, "unsupported audit schema");
    anyhow::ensure!(!manifest.name.trim().is_empty(), "audit name is required");
    anyhow::ensure!(
        !manifest.runs.is_empty(),
        "at least one validation run is required"
    );
    anyhow::ensure!(
        matches!(
            manifest.validation_layer.as_str(),
            "raw_q" | "level4" | "reportable_q"
        ),
        "validation_layer must be raw_q, level4, or reportable_q"
    );

    std::fs::create_dir_all(&manifest.output_root)?;
    let missing_runs = manifest
        .runs
        .iter()
        .filter(|run| !run.results.is_file())
        .cloned()
        .collect::<Vec<_>>();
    let mut summaries = Vec::new();
    for run in &manifest.runs {
        summaries.extend(summarize_run(
            run,
            &manifest.effective_ratios,
            manifest.fdr_threshold,
        )?);
    }
    let transfer = transfer_stability(&summaries, manifest.maximum_transfer_fraction_loss);
    let comparisons = stage_comparisons(&summaries);
    let expert_gates = expert_quality_gates(
        &summaries,
        &transfer,
        manifest.fdr_threshold,
        manifest.minimum_entrapment_peptides_for_stable_estimate,
        &manifest.validation_layer,
    );
    let parity = parity_comparisons(
        &summaries,
        &manifest.parity_pairs,
        manifest.maximum_parity_fraction_difference,
    );
    let tdc = tdc_benchmark_comparisons(
        &summaries,
        manifest.tdc_reference_method.as_deref(),
        manifest.fdr_threshold,
    );
    let report = ValidationAuditReport {
        schema_version: 1,
        name: manifest.name,
        summaries,
        missing_runs,
        transfer_stability: transfer,
        stage_comparisons: comparisons,
        expert_quality_gates: expert_gates,
        parity_comparisons: parity,
        tdc_benchmarks: tdc,
    };

    write_json_atomic(
        &manifest.output_root.join("validation.summary.json"),
        &report.summaries,
    )?;
    write_json_atomic(
        &manifest.output_root.join("validation.missing_runs.json"),
        &report.missing_runs,
    )?;
    write_json_atomic(
        &manifest
            .output_root
            .join("validation.transfer_stability.json"),
        &report.transfer_stability,
    )?;
    write_json_atomic(
        &manifest
            .output_root
            .join("validation.stage_comparisons.json"),
        &report.stage_comparisons,
    )?;
    write_json_atomic(
        &manifest.output_root.join("validation.expert_gates.json"),
        &report.expert_quality_gates,
    )?;
    write_json_atomic(
        &manifest.output_root.join("validation.parity.json"),
        &report.parity_comparisons,
    )?;
    write_json_atomic(
        &manifest.output_root.join("validation.tdc_benchmarks.json"),
        &report.tdc_benchmarks,
    )?;
    write_json_atomic(&manifest.output_root.join("validation.audit.json"), &report)?;
    Ok(report)
}
