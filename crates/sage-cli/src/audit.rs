use crate::provenance::write_json_atomic;
use crate::validation::{
    expert_quality_gates, is_target_only_stage, missing_parity_evidence, parity_comparisons,
    stage_comparisons, summarize_run, tdc_benchmark_comparisons, transfer_stability,
    EffectiveRatios, ExpertQualityGate, InvalidValidationRun, ParityComparison, ParityPair,
    RunValidationSummary, StageComparison, TdcBenchmarkComparison, TransferStability,
    ValidationMode, ValidationRun,
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
    #[serde(default)]
    pub invalid_runs: Vec<InvalidValidationRun>,
    #[serde(default)]
    pub evaluable: bool,
    #[serde(default)]
    pub not_evaluable_reasons: Vec<String>,
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
    let mut invalid_runs = Vec::new();
    for run in &manifest.runs {
        if !run.results.is_file() {
            continue;
        }
        match summarize_run(run, &manifest.effective_ratios, manifest.fdr_threshold) {
            Ok(rows) => summaries.extend(rows),
            Err(error) => invalid_runs.push(InvalidValidationRun {
                run: run.clone(),
                error: format!("{error:#}"),
            }),
        }
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
    let mut not_evaluable_reasons = Vec::new();
    if !missing_runs.is_empty() {
        not_evaluable_reasons.push("required validation runs are missing".into());
    }
    if !invalid_runs.is_empty() {
        not_evaluable_reasons.push("one or more validation runs are invalid or unreadable".into());
    }
    not_evaluable_reasons.extend(missing_parity_evidence(&manifest.parity_pairs, &parity));
    if manifest.tdc_reference_method.is_some() && tdc.is_empty() {
        not_evaluable_reasons.push("the declared TDC benchmark has no matched results".into());
    }
    for target in summaries.iter().filter(|row| {
        matches!(row.mode, ValidationMode::DecoyFree)
            && row.release_candidate
            && is_target_only_stage(&row.stage)
            && row.layer == manifest.validation_layer
    }) {
        if target.calibration_stage.is_none() {
            not_evaluable_reasons.push(format!(
                "{} / {} has no target-only calibration provenance",
                target.method, target.stage
            ));
        }
        if !transfer.iter().any(|comparison| {
            comparison.method == target.method
                && comparison.to_stage == target.stage
                && comparison.release_candidate
        }) {
            not_evaluable_reasons.push(format!(
                "{} / {} has no evaluable target-only transfer comparison",
                target.method, target.stage
            ));
        }
        if expert_gates
            .iter()
            .find(|gate| gate.model == target.method)
            .is_none_or(|gate| gate.reasons.iter().any(|reason| reason.contains("missing")))
        {
            not_evaluable_reasons.push(format!(
                "{} has missing calibration quality-gate evidence",
                target.method
            ));
        }
        if manifest.tdc_reference_method.is_some()
            && tdc
                .iter()
                .find(|comparison| {
                    comparison.decoy_free_method == target.method
                        && comparison.stage == target.stage
                        && comparison.layer == target.layer
                        && comparison.release_candidate
                })
                .is_none_or(|comparison| {
                    !comparison.calibration_constrained
                        && comparison.peptide_entrapment_fdp.is_none()
                })
        {
            not_evaluable_reasons.push(format!(
                "{} / {} has no evaluable calibrated TDC comparison",
                target.method, target.stage
            ));
        }
    }
    not_evaluable_reasons.sort();
    not_evaluable_reasons.dedup();
    let evaluable = not_evaluable_reasons.is_empty();
    let report = ValidationAuditReport {
        schema_version: 2,
        name: manifest.name,
        summaries,
        missing_runs,
        invalid_runs,
        evaluable,
        not_evaluable_reasons,
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
        &manifest.output_root.join("validation.invalid_runs.json"),
        &report.invalid_runs,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::{ValidationMode, ValidationRun};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn missing_and_invalid_evidence_is_reported_not_evaluable() {
        let root = std::env::temp_dir().join(format!(
            "sage-validation-audit-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let invalid = root.join("invalid.tsv");
        std::fs::write(&invalid, b"not a Sage result table\n").unwrap();
        let run = |method: &str, results: PathBuf| ValidationRun {
            method: method.into(),
            stage: "optimized".into(),
            results,
            mode: ValidationMode::DecoyFree,
            expected_search_space: Some("+Ent".into()),
            calibration_stage: None,
            target_only_calibration_policy: None,
            release_candidate: true,
        };
        let manifest = ValidationAuditManifest {
            schema_version: 1,
            name: "missing-evidence".into(),
            output_root: root.clone(),
            effective_ratios: EffectiveRatios::default(),
            fdr_threshold: 0.01,
            maximum_transfer_fraction_loss: 0.20,
            minimum_entrapment_peptides_for_stable_estimate: 3,
            validation_layer: "level4".into(),
            parity_pairs: vec![ParityPair {
                baseline_method: "baseline".into(),
                native_method: "native".into(),
                stages: vec!["optimized".into()],
                layers: vec!["level4".into()],
                maximum_fraction_difference: None,
            }],
            maximum_parity_fraction_difference: 0.001,
            tdc_reference_method: Some("tdc".into()),
            runs: vec![
                run("baseline", root.join("missing.tsv")),
                run("native", invalid),
            ],
        };
        let manifest_path = root.join("audit.json");
        write_json_atomic(&manifest_path, &manifest).unwrap();

        let report = execute_validation_audit(&manifest_path).unwrap();
        assert!(!report.evaluable);
        assert_eq!(report.missing_runs.len(), 1);
        assert_eq!(report.invalid_runs.len(), 1);
        assert!(report
            .not_evaluable_reasons
            .iter()
            .any(|reason| reason.contains("parity evidence")));
        assert!(root.join("validation.missing_runs.json").is_file());
        assert!(root.join("validation.invalid_runs.json").is_file());
        std::fs::remove_dir_all(root).unwrap();
    }
}
