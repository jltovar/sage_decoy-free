use anyhow::{Context, Result};
use csv::StringRecord;
use sage_core::input::ModelFit;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationMode {
    DecoyFree,
    Tdc,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetOnlyCalibrationPolicy {
    /// Lock only the dataset-local null window and re-estimate nuisance state
    /// in the target-only candidate space. This is the legacy-parity default.
    #[default]
    RefitWithLockedWindow,
    /// Apply the complete fitted +entrapment artifact without refitting it.
    ReuseDatasetArtifact,
    /// Materialize both interpretations as independent, provenance-bearing
    /// target-only stages. This value is orchestration-only.
    CompareBoth,
}

impl TargetOnlyCalibrationPolicy {
    pub fn stage_name(self) -> &'static str {
        match self {
            Self::RefitWithLockedWindow => "target_only_refit_with_locked_window",
            Self::ReuseDatasetArtifact => "target_only_reuse_dataset_artifact",
            Self::CompareBoth => "target_only_compare_both",
        }
    }
}

pub const LOWER_ORDER_TARGET_ONLY_REUSE_UNSUPPORTED_REASON: &str =
    "Lower Order nuisance parameters and candidate-count normalization are search-space dependent and must be refitted after the FASTA/candidate space changes.";

/// Resolved model capability for one concrete target-only interpretation.
///
/// Keep this separate from workflow parsing so artifact application and other
/// lower-level callers enforce the same scientific contract.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetOnlyPolicyCapability {
    pub model: String,
    pub policy: TargetOnlyCalibrationPolicy,
    pub supported: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

pub fn target_only_policy_capability(
    model: &ModelFit,
    policy: TargetOnlyCalibrationPolicy,
) -> TargetOnlyPolicyCapability {
    let unsupported_lower_order_reuse = *model == ModelFit::LowerOrder
        && policy == TargetOnlyCalibrationPolicy::ReuseDatasetArtifact;
    TargetOnlyPolicyCapability {
        model: match model {
            ModelFit::Moments => "moments",
            ModelFit::Mle => "mle",
            ModelFit::LowerOrder => "lower_order",
            ModelFit::Msfdr => "msfdr",
            ModelFit::Msfdr1Smix => "msfdr1_smix",
            ModelFit::Msfdr2Smix => "msfdr2_smix",
            ModelFit::Nokoi => "nokoi",
            ModelFit::Ensemble => "ensemble",
        }
        .into(),
        policy,
        supported: !unsupported_lower_order_reuse,
        reason: unsupported_lower_order_reuse
            .then(|| LOWER_ORDER_TARGET_ONLY_REUSE_UNSUPPORTED_REASON.into()),
    }
}

pub fn is_target_only_stage(stage: &str) -> bool {
    stage == "target_only" || stage.starts_with("target_only_")
}

fn default_release_candidate() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationRun {
    pub method: String,
    pub stage: String,
    pub results: PathBuf,
    pub mode: ValidationMode,
    #[serde(default)]
    pub expected_search_space: Option<String>,
    /// For a target-only result, identify the +entrapment stage whose frozen
    /// calibration artifact was applied. This prevents a measured but rejected
    /// MS2Rescore stage from being credited as the calibration source.
    #[serde(default)]
    pub calibration_stage: Option<String>,
    /// Explicit interpretation used for target-only calibration. Legacy
    /// `target_only` validation rows may omit this field.
    #[serde(default)]
    pub target_only_calibration_policy: Option<TargetOnlyCalibrationPolicy>,
    /// `compare_both` retains one diagnostic interpretation without allowing
    /// it to veto the release candidate selected by the manifest.
    #[serde(default = "default_release_candidate")]
    pub release_candidate: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InvalidValidationRun {
    pub run: ValidationRun,
    pub error: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectiveRatios {
    pub psm: f64,
    pub peptide: f64,
    pub protein: f64,
}

impl Default for EffectiveRatios {
    fn default() -> Self {
        Self {
            psm: 1.0,
            peptide: 1.0,
            protein: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct IdentificationCount {
    pub target: usize,
    pub entrapment: usize,
    pub combined_entrapment_fdp: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RunValidationSummary {
    pub method: String,
    pub stage: String,
    pub results: PathBuf,
    pub mode: ValidationMode,
    #[serde(default)]
    pub calibration_stage: Option<String>,
    #[serde(default)]
    pub target_only_calibration_policy: Option<TargetOnlyCalibrationPolicy>,
    #[serde(default = "default_release_candidate")]
    pub release_candidate: bool,
    pub complete: bool,
    pub search_space: String,
    pub layer: String,
    pub psm: IdentificationCount,
    pub peptide: IdentificationCount,
    #[serde(default)]
    pub peptidoform: IdentificationCount,
    pub protein: IdentificationCount,
    pub counting_definition: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EnsembleInteractionLevel {
    pub measured_entrapment_ratio: f64,
    pub baseline_target: usize,
    pub baseline_entrapment: usize,
    pub baseline_fdp_numerator: Option<f64>,
    pub baseline_fdp_denominator: usize,
    pub baseline_fdp: Option<f64>,
    pub final_target: usize,
    pub final_entrapment: usize,
    pub final_fdp_numerator: Option<f64>,
    pub final_fdp_denominator: usize,
    pub final_fdp: Option<f64>,
    pub absolute_fdp_change: Option<f64>,
    pub relative_fdp_change: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EnsembleInteractionLayer {
    pub psm: EnsembleInteractionLevel,
    pub peptide: EnsembleInteractionLevel,
    pub peptidoform: EnsembleInteractionLevel,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EnsembleInteractionWarning {
    pub code: String,
    pub threshold: f64,
    pub affected_levels: Vec<String>,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EnsembleInteractionCalibration {
    pub schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_lock_analysis_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_lock_analysis_fingerprint: Option<String>,
    pub baseline_experts: Vec<String>,
    pub final_experts: Vec<String>,
    pub newly_participating_experts: Vec<String>,
    pub raw_q: Option<EnsembleInteractionLayer>,
    pub level4: Option<EnsembleInteractionLayer>,
    pub raw_q_warning: Option<EnsembleInteractionWarning>,
    pub final_level4_calibration_pass: bool,
    pub evaluable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_evaluable_reason: Option<String>,
}

fn interaction_level(
    baseline: &IdentificationCount,
    final_count: &IdentificationCount,
    measured_entrapment_ratio: f64,
) -> EnsembleInteractionLevel {
    let absolute_fdp_change = baseline
        .combined_entrapment_fdp
        .zip(final_count.combined_entrapment_fdp)
        .map(|(before, after)| after - before);
    let relative_fdp_change = baseline
        .combined_entrapment_fdp
        .zip(final_count.combined_entrapment_fdp)
        .and_then(|(before, after)| {
            if before > 0.0 {
                Some((after - before) / before)
            } else if after == 0.0 {
                Some(0.0)
            } else {
                None
            }
        });
    EnsembleInteractionLevel {
        measured_entrapment_ratio,
        baseline_target: baseline.target,
        baseline_entrapment: baseline.entrapment,
        baseline_fdp_numerator: (measured_entrapment_ratio.is_finite()
            && measured_entrapment_ratio > 0.0)
            .then(|| baseline.entrapment as f64 * (1.0 + 1.0 / measured_entrapment_ratio)),
        baseline_fdp_denominator: baseline.target + baseline.entrapment,
        baseline_fdp: baseline.combined_entrapment_fdp,
        final_target: final_count.target,
        final_entrapment: final_count.entrapment,
        final_fdp_numerator: (measured_entrapment_ratio.is_finite()
            && measured_entrapment_ratio > 0.0)
            .then(|| final_count.entrapment as f64 * (1.0 + 1.0 / measured_entrapment_ratio)),
        final_fdp_denominator: final_count.target + final_count.entrapment,
        final_fdp: final_count.combined_entrapment_fdp,
        absolute_fdp_change,
        relative_fdp_change,
    }
}

fn interaction_layer(
    baseline: &RunValidationSummary,
    final_result: &RunValidationSummary,
    ratios: &EffectiveRatios,
) -> EnsembleInteractionLayer {
    EnsembleInteractionLayer {
        psm: interaction_level(&baseline.psm, &final_result.psm, ratios.psm),
        peptide: interaction_level(&baseline.peptide, &final_result.peptide, ratios.peptide),
        // Peptidoforms use the PSM-space measured ratio in summarize_run.
        peptidoform: interaction_level(
            &baseline.peptidoform,
            &final_result.peptidoform,
            ratios.psm,
        ),
    }
}

fn interaction_summary<'a>(
    rows: &'a [RunValidationSummary],
    layer: &str,
) -> Option<&'a RunValidationSummary> {
    rows.iter()
        .find(|row| row.layer == layer && row.search_space == "+Ent")
}

pub fn ensemble_interaction_calibration(
    baseline: &[RunValidationSummary],
    final_result: &[RunValidationSummary],
    ratios: &EffectiveRatios,
    fdr_threshold: f64,
    raw_q_warning_threshold: f64,
    mut baseline_experts: Vec<String>,
    mut final_experts: Vec<String>,
) -> Result<EnsembleInteractionCalibration> {
    anyhow::ensure!(
        fdr_threshold.is_finite() && fdr_threshold >= 0.0,
        "invalid Ensemble interaction FDR threshold"
    );
    anyhow::ensure!(
        raw_q_warning_threshold.is_finite() && raw_q_warning_threshold >= 0.0,
        "invalid Ensemble interaction warning threshold"
    );
    baseline_experts.sort();
    baseline_experts.dedup();
    final_experts.sort();
    final_experts.dedup();
    anyhow::ensure!(
        baseline_experts
            .iter()
            .all(|expert| final_experts.contains(expert)),
        "Ensemble interaction baseline contains an expert absent from the final Ensemble"
    );
    let newly_participating_experts = final_experts
        .iter()
        .filter(|expert| !baseline_experts.contains(expert))
        .cloned()
        .collect::<Vec<_>>();
    let baseline_raw = interaction_summary(baseline, "raw_q")
        .context("baseline Ensemble has no evaluable +entrapment raw-q calibration summary")?;
    let final_raw = interaction_summary(final_result, "raw_q")
        .context("final Ensemble has no evaluable +entrapment raw-q calibration summary")?;
    let baseline_level4 = interaction_summary(baseline, "level4")
        .context("baseline Ensemble has no evaluable +entrapment Level-4 calibration summary")?;
    let final_level4 = interaction_summary(final_result, "level4")
        .context("final Ensemble has no evaluable +entrapment Level-4 calibration summary")?;
    let raw_q = interaction_layer(baseline_raw, final_raw, ratios);
    let level4 = interaction_layer(baseline_level4, final_level4, ratios);
    let affected_levels = [
        ("psm", raw_q.psm.absolute_fdp_change),
        ("peptide", raw_q.peptide.absolute_fdp_change),
        ("peptidoform", raw_q.peptidoform.absolute_fdp_change),
    ]
    .into_iter()
    .filter(|(_, change)| change.is_some_and(|change| change > raw_q_warning_threshold))
    .map(|(level, _)| level.into())
    .collect::<Vec<_>>();
    let raw_q_warning = (!affected_levels.is_empty()).then(|| EnsembleInteractionWarning {
        code: "raw_q_ensemble_interaction_deterioration".into(),
        threshold: raw_q_warning_threshold,
        affected_levels,
        message: "post-assembly raw-q entrapment FDP deterioration exceeds the informational validation reference; this warning is not a passing calibration gate".into(),
    });
    // The production release policy gates peptide calibration. PSM and
    // peptidoform FDP remain fully reported without creating a new,
    // post-observation admission threshold.
    let final_level4_calibration_pass = level4
        .peptide
        .final_fdp
        .is_some_and(|fdp| fdp <= fdr_threshold);
    let evaluable = level4.peptide.final_fdp.is_some();
    Ok(EnsembleInteractionCalibration {
        schema_version: 1,
        baseline_lock_analysis_fingerprint: None,
        final_lock_analysis_fingerprint: None,
        baseline_experts,
        final_experts,
        newly_participating_experts,
        raw_q: Some(raw_q),
        level4: Some(level4),
        raw_q_warning,
        final_level4_calibration_pass,
        evaluable,
        not_evaluable_reason: (!evaluable)
            .then(|| "final Ensemble Level-4 peptide entrapment FDP is missing".into()),
    })
}

fn column(headers: &StringRecord, names: &[&str]) -> Option<usize> {
    names
        .iter()
        .find_map(|name| headers.iter().position(|header| header == *name))
}

fn parse_bool(value: Option<&str>) -> bool {
    matches!(
        value.unwrap_or("").trim().to_ascii_lowercase().as_str(),
        "true" | "t" | "1" | "yes" | "y"
    )
}

fn parse_f64(value: Option<&str>) -> Option<f64> {
    value?.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn parse_i64(value: Option<&str>) -> Option<i64> {
    value?.parse::<i64>().ok()
}

fn canonical_peptide(peptide: &str) -> String {
    let mut output = String::new();
    let mut bracket_depth = 0_u32;
    for character in peptide.chars() {
        match character {
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            _ if bracket_depth == 0 && character.is_ascii_alphabetic() => {
                let upper = character.to_ascii_uppercase();
                output.push(if upper == 'I' { 'L' } else { upper });
            }
            _ => {}
        }
    }
    output
}

fn canonical_peptidoform(peptide: &str) -> String {
    let mut output = String::new();
    let mut bracket_depth = 0_u32;
    for character in peptide.trim().chars() {
        match character {
            '[' => {
                bracket_depth += 1;
                output.push(character);
            }
            ']' => {
                bracket_depth = bracket_depth.saturating_sub(1);
                output.push(character);
            }
            _ if bracket_depth == 0 && character.is_ascii_alphabetic() => {
                let upper = character.to_ascii_uppercase();
                output.push(if upper == 'I' { 'L' } else { upper });
            }
            _ => output.push(character),
        }
    }
    output
}

fn protein_key(proteins: &str) -> Option<String> {
    if proteins.split(';').count() != 1 {
        return None;
    }
    let key = proteins.trim();
    (!key.is_empty()).then(|| key.to_owned())
}

fn is_contaminant(proteins: &str) -> bool {
    proteins.contains("Cont_")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProteinClass {
    Target,
    Entrapment,
    Ambiguous,
}

fn protein_class(proteins: &str) -> Option<ProteinClass> {
    if is_contaminant(proteins) {
        return None;
    }
    let mut target = false;
    let mut entrapment = false;
    for protein in proteins.split(';').map(str::trim).filter(|x| !x.is_empty()) {
        if protein.contains("Ent_") {
            entrapment = true;
        } else {
            target = true;
        }
    }
    match (target, entrapment) {
        (true, false) => Some(ProteinClass::Target),
        (false, true) => Some(ProteinClass::Entrapment),
        (true, true) | (false, false) => Some(ProteinClass::Ambiguous),
    }
}

pub fn accepted_target_peptides(
    results: &PathBuf,
    mode: &ValidationMode,
    fdr_threshold: f64,
    require_level4: bool,
) -> Result<BTreeSet<String>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(results)
        .with_context(|| format!("failed to open {}", results.display()))?;
    let headers = reader.headers()?.clone();
    let rank = column(&headers, &["rank"]).context("results missing rank")?;
    let label = column(&headers, &["label"]).context("results missing label")?;
    let proteins = column(&headers, &["proteins"]).context("results missing proteins")?;
    let peptide = column(&headers, &["peptide", "sequence"]).context("results missing peptide")?;
    let peptide_q = match mode {
        ValidationMode::DecoyFree => column(&headers, &["decoy_free_peptide_q"]),
        ValidationMode::Tdc => column(&headers, &["peptide_q"]),
    }
    .context("results missing peptide q-value")?;
    let supported = column(&headers, &["decoy_free_protein_supported_peptide"]);
    if require_level4 && matches!(mode, ValidationMode::DecoyFree) && supported.is_none() {
        anyhow::bail!("Level-4 peptide support column is missing");
    }
    let mut accepted = BTreeSet::new();
    for row in reader.records() {
        let row = row?;
        if parse_i64(row.get(rank)) != Some(1)
            || parse_i64(row.get(label)) != Some(1)
            || !parse_f64(row.get(peptide_q)).is_some_and(|q| q <= fdr_threshold)
        {
            continue;
        }
        let protein = row.get(proteins).unwrap_or("");
        if protein_class(protein) != Some(ProteinClass::Target) {
            continue;
        }
        if require_level4
            && matches!(mode, ValidationMode::DecoyFree)
            && !parse_bool(supported.and_then(|index| row.get(index)))
        {
            continue;
        }
        let key = canonical_peptide(row.get(peptide).unwrap_or(""));
        if !key.is_empty() {
            accepted.insert(key);
        }
    }
    Ok(accepted)
}

fn combined_fdp(
    count: &IdentificationCount,
    effective_r: f64,
    has_entrapment: bool,
) -> Option<f64> {
    let total = count.target + count.entrapment;
    if !has_entrapment || total == 0 || !effective_r.is_finite() || effective_r <= 0.0 {
        return None;
    }
    Some(((count.entrapment as f64) * (1.0 + 1.0 / effective_r) / total as f64).clamp(0.0, 1.0))
}

pub fn summarize_run(
    run: &ValidationRun,
    ratios: &EffectiveRatios,
    fdr_threshold: f64,
) -> Result<Vec<RunValidationSummary>> {
    if !run.results.is_file() {
        return Ok(Vec::new());
    }
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(&run.results)
        .with_context(|| format!("failed to open {}", run.results.display()))?;
    let headers = reader.headers()?.clone();
    let rank = column(&headers, &["rank"]).context("results missing rank")?;
    let label = column(&headers, &["label"]).context("results missing label")?;
    let proteins = column(&headers, &["proteins"]).context("results missing proteins")?;
    let peptide = column(&headers, &["peptide", "sequence"]).context("results missing peptide")?;
    let psm_id = column(&headers, &["psm_id"]);
    let filename = column(&headers, &["filename"]);
    let scan = column(&headers, &["scannr"]);

    let decoy_free = matches!(run.mode, ValidationMode::DecoyFree);
    let psm_q = if decoy_free {
        column(&headers, &["decoy_free_q_value"])
    } else {
        column(
            &headers,
            &["spectrum_q", "sage_discriminant_q_value", "q_value"],
        )
    }
    .context("results missing PSM q-value")?;
    let peptide_q = if decoy_free {
        column(&headers, &["decoy_free_peptide_q"])
    } else {
        column(&headers, &["peptide_q"])
    }
    .context("results missing peptide q-value")?;
    let protein_q = if decoy_free {
        column(&headers, &["decoy_free_protein_q"])
    } else {
        column(&headers, &["protein_group_q", "protein_q"])
    }
    .context("results missing protein q-value")?;
    let peptide_supported = column(&headers, &["decoy_free_protein_supported_peptide"]);
    let psm_supported = column(&headers, &["decoy_free_peptide_supported_psm"]);
    let has_level4 = decoy_free && peptide_supported.is_some() && psm_supported.is_some();

    #[derive(Default)]
    struct Sets {
        target_psm: BTreeSet<String>,
        ent_psm: BTreeSet<String>,
        target_peptide: BTreeSet<String>,
        ent_peptide: BTreeSet<String>,
        target_peptidoform: BTreeSet<String>,
        ent_peptidoform: BTreeSet<String>,
        target_protein: BTreeSet<String>,
        ent_protein: BTreeSet<String>,
    }

    let mut raw = Sets::default();
    let mut reportable = Sets::default();
    let mut has_entrapment = false;
    for (row_index, row) in reader.records().enumerate() {
        let row = row?;
        if parse_i64(row.get(rank)) != Some(1) || parse_i64(row.get(label)) != Some(1) {
            continue;
        }
        let protein_text = row.get(proteins).unwrap_or("");
        if is_contaminant(protein_text) {
            continue;
        }
        let entrapment = match protein_class(protein_text) {
            Some(ProteinClass::Target) => false,
            Some(ProteinClass::Entrapment) => true,
            Some(ProteinClass::Ambiguous) | None => continue,
        };
        let protein = protein_key(protein_text);
        has_entrapment |= entrapment;
        let peptide_text = row.get(peptide).unwrap_or("");
        let peptide = canonical_peptide(peptide_text);
        let peptidoform = canonical_peptidoform(peptide_text);
        if peptide.is_empty() {
            continue;
        }
        let psm = psm_id
            .and_then(|index| row.get(index).map(ToOwned::to_owned))
            .or_else(|| {
                Some(format!(
                    "{}:{}:{}",
                    filename
                        .and_then(|index| row.get(index))
                        .unwrap_or("unknown"),
                    scan.and_then(|index| row.get(index)).unwrap_or("unknown"),
                    row_index
                ))
            })
            .unwrap();

        let insert = |sets: &mut Sets, psm_ok: bool, peptide_ok: bool, protein_ok: bool| {
            if psm_ok {
                if entrapment {
                    sets.ent_psm.insert(psm.clone());
                } else {
                    sets.target_psm.insert(psm.clone());
                }
            }
            if peptide_ok {
                if entrapment {
                    sets.ent_peptide.insert(peptide.clone());
                } else {
                    sets.target_peptide.insert(peptide.clone());
                }
                if entrapment {
                    sets.ent_peptidoform.insert(peptidoform.clone());
                } else {
                    sets.target_peptidoform.insert(peptidoform.clone());
                }
            }
            if protein_ok {
                if let Some(protein) = protein.as_ref() {
                    if entrapment {
                        sets.ent_protein.insert(protein.clone());
                    } else {
                        sets.target_protein.insert(protein.clone());
                    }
                }
            }
        };

        let raw_psm = parse_f64(row.get(psm_q)).is_some_and(|q| q <= fdr_threshold);
        let raw_peptide = parse_f64(row.get(peptide_q)).is_some_and(|q| q <= fdr_threshold);
        let raw_protein = parse_f64(row.get(protein_q)).is_some_and(|q| q <= fdr_threshold);
        insert(&mut raw, raw_psm, raw_peptide, raw_protein);

        if has_level4 {
            insert(
                &mut reportable,
                raw_psm && parse_bool(psm_supported.and_then(|index| row.get(index))),
                raw_peptide && parse_bool(peptide_supported.and_then(|index| row.get(index))),
                raw_protein,
            );
        } else {
            insert(&mut reportable, raw_psm, raw_peptide, raw_protein);
        }
    }

    let observed_search_space = if has_entrapment { "+Ent" } else { "No Ent" };
    if let Some(expected) = run.expected_search_space.as_deref() {
        anyhow::ensure!(
            expected == observed_search_space,
            "{} expected search space {expected}, but results contain {observed_search_space}",
            run.results.display()
        );
    }

    let make = |layer: &str, sets: &Sets| {
        let mut psm = IdentificationCount {
            target: sets.target_psm.len(),
            entrapment: sets.ent_psm.len(),
            combined_entrapment_fdp: None,
        };
        let mut peptide = IdentificationCount {
            target: sets.target_peptide.len(),
            entrapment: sets.ent_peptide.len(),
            combined_entrapment_fdp: None,
        };
        let mut peptidoform = IdentificationCount {
            target: sets.target_peptidoform.len(),
            entrapment: sets.ent_peptidoform.len(),
            combined_entrapment_fdp: None,
        };
        let mut protein = IdentificationCount {
            target: sets.target_protein.len(),
            entrapment: sets.ent_protein.len(),
            combined_entrapment_fdp: None,
        };
        psm.combined_entrapment_fdp = combined_fdp(&psm, ratios.psm, has_entrapment);
        peptide.combined_entrapment_fdp = combined_fdp(&peptide, ratios.peptide, has_entrapment);
        peptidoform.combined_entrapment_fdp =
            combined_fdp(&peptidoform, ratios.psm, has_entrapment);
        protein.combined_entrapment_fdp = combined_fdp(&protein, ratios.protein, has_entrapment);
        RunValidationSummary {
            method: run.method.clone(),
            stage: run.stage.clone(),
            results: run.results.clone(),
            mode: run.mode.clone(),
            calibration_stage: run.calibration_stage.clone(),
            target_only_calibration_policy: run.target_only_calibration_policy,
            release_candidate: run.release_candidate,
            complete: true,
            search_space: observed_search_space.into(),
            layer: layer.into(),
            psm,
            peptide,
            peptidoform,
            protein,
            counting_definition: "rank=1,label=1,non-contaminant,unambiguous target/entrapment mapping; PSM is a distinct result-table PSM identity; peptide removes bracketed modifications and canonicalizes I/L; peptidoform retains bracketed modification annotations while canonicalizing unmodified I/L; protein requires one inferred protein key"
                .into(),
        }
    };
    Ok(vec![
        make("raw_q", &raw),
        make(
            if has_level4 { "level4" } else { "reportable_q" },
            &reportable,
        ),
    ])
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferStability {
    pub method: String,
    pub from_stage: String,
    pub to_stage: String,
    pub psm_fraction_change: Option<f64>,
    pub peptide_fraction_change: Option<f64>,
    #[serde(default)]
    pub peptidoform_fraction_change: Option<f64>,
    pub protein_fraction_change: Option<f64>,
    pub stable: bool,
    #[serde(default)]
    pub target_only_calibration_policy: Option<TargetOnlyCalibrationPolicy>,
    #[serde(default = "default_release_candidate")]
    pub release_candidate: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageComparison {
    pub method: String,
    pub layer: String,
    pub from_stage: String,
    pub to_stage: String,
    pub target_psm_change: i64,
    pub target_peptide_change: i64,
    #[serde(default)]
    pub target_peptidoform_change: i64,
    pub target_protein_change: i64,
    pub peptide_fdp_change: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParityPair {
    pub baseline_method: String,
    pub native_method: String,
    /// Restrict parity to stages whose semantics are intentionally unchanged.
    /// An empty list preserves the historical behavior of comparing every
    /// common stage.
    #[serde(default)]
    pub stages: Vec<String>,
    /// Restrict parity to selected reporting layers. An empty list compares
    /// every common layer.
    #[serde(default)]
    pub layers: Vec<String>,
    /// Optional pair-specific tolerance for known cross-platform numerical
    /// variation. The comparison still reports every metric's fraction change.
    #[serde(default)]
    pub maximum_fraction_difference: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParityComparison {
    pub baseline_method: String,
    pub native_method: String,
    pub stage: String,
    pub layer: String,
    pub psm_fraction_change: Option<f64>,
    pub peptide_fraction_change: Option<f64>,
    #[serde(default)]
    pub peptidoform_fraction_change: Option<f64>,
    pub protein_fraction_change: Option<f64>,
    pub within_tolerance: bool,
}

pub fn parity_comparisons(
    summaries: &[RunValidationSummary],
    pairs: &[ParityPair],
    maximum_fraction_difference: f64,
) -> Vec<ParityComparison> {
    let fraction = |baseline: usize, native: usize| {
        (baseline > 0).then_some((native as f64 - baseline as f64) / baseline as f64)
    };
    let mut output = Vec::new();
    for pair in pairs {
        let pair_tolerance = pair
            .maximum_fraction_difference
            .unwrap_or(maximum_fraction_difference);
        for baseline in summaries
            .iter()
            .filter(|row| row.method == pair.baseline_method)
            .filter(|row| pair.stages.is_empty() || pair.stages.contains(&row.stage))
            .filter(|row| pair.layers.is_empty() || pair.layers.contains(&row.layer))
        {
            if let Some(native) = summaries.iter().find(|row| {
                row.method == pair.native_method
                    && (row.stage == baseline.stage
                        || (baseline.stage == "target_only"
                            && row.stage
                                == TargetOnlyCalibrationPolicy::RefitWithLockedWindow.stage_name())
                        || (row.stage == "target_only"
                            && baseline.stage
                                == TargetOnlyCalibrationPolicy::RefitWithLockedWindow.stage_name()))
                    && row.layer == baseline.layer
            }) {
                let psm = fraction(baseline.psm.target, native.psm.target);
                let peptide = fraction(baseline.peptide.target, native.peptide.target);
                let peptidoform = fraction(baseline.peptidoform.target, native.peptidoform.target);
                let protein = fraction(baseline.protein.target, native.protein.target);
                let within_tolerance = [
                    (baseline.psm.target, native.psm.target, psm),
                    (baseline.peptide.target, native.peptide.target, peptide),
                    (
                        baseline.peptidoform.target,
                        native.peptidoform.target,
                        peptidoform,
                    ),
                    (baseline.protein.target, native.protein.target, protein),
                ]
                .into_iter()
                .all(|(baseline, native, change)| {
                    if baseline == 0 {
                        native == 0
                    } else {
                        change.is_some_and(|value| value.abs() <= pair_tolerance)
                    }
                });
                output.push(ParityComparison {
                    baseline_method: pair.baseline_method.clone(),
                    native_method: pair.native_method.clone(),
                    stage: baseline.stage.clone(),
                    layer: baseline.layer.clone(),
                    psm_fraction_change: psm,
                    peptide_fraction_change: peptide,
                    peptidoform_fraction_change: peptidoform,
                    protein_fraction_change: protein,
                    within_tolerance,
                });
            }
        }
    }
    output
}

pub fn missing_parity_evidence(
    pairs: &[ParityPair],
    comparisons: &[ParityComparison],
) -> Vec<String> {
    let mut reasons = Vec::new();
    for pair in pairs {
        let pair_comparisons = comparisons
            .iter()
            .filter(|comparison| {
                comparison.baseline_method == pair.baseline_method
                    && comparison.native_method == pair.native_method
            })
            .collect::<Vec<_>>();
        let required_stages = if pair.stages.is_empty() {
            vec![None]
        } else {
            pair.stages.iter().map(Some).collect()
        };
        let required_layers = if pair.layers.is_empty() {
            vec![None]
        } else {
            pair.layers.iter().map(Some).collect()
        };
        for stage in &required_stages {
            for layer in &required_layers {
                if !pair_comparisons.iter().any(|comparison| {
                    stage.is_none_or(|stage| comparison.stage == *stage)
                        && layer.is_none_or(|layer| comparison.layer == *layer)
                }) {
                    reasons.push(format!(
                        "declared dataset-local parity evidence is missing for {} -> {}{}{}",
                        pair.baseline_method,
                        pair.native_method,
                        stage
                            .map(|stage| format!(" stage={stage}"))
                            .unwrap_or_default(),
                        layer
                            .map(|layer| format!(" layer={layer}"))
                            .unwrap_or_default()
                    ));
                }
            }
        }
    }
    reasons.sort();
    reasons.dedup();
    reasons
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TdcBenchmarkComparison {
    pub decoy_free_method: String,
    pub tdc_method: String,
    pub stage: String,
    pub layer: String,
    pub tdc_layer: String,
    pub psm_yield_difference: i64,
    pub peptide_yield_difference: i64,
    #[serde(default)]
    pub peptidoform_yield_difference: i64,
    pub protein_yield_difference: i64,
    pub peptide_entrapment_fdp: Option<f64>,
    pub calibration_stage: Option<String>,
    pub calibration_constrained: bool,
    pub improves_peptide_yield: bool,
    #[serde(default)]
    pub target_only_calibration_policy: Option<TargetOnlyCalibrationPolicy>,
    #[serde(default = "default_release_candidate")]
    pub release_candidate: bool,
}

pub fn tdc_benchmark_comparisons(
    summaries: &[RunValidationSummary],
    tdc_method: Option<&str>,
    maximum_fdp: f64,
) -> Vec<TdcBenchmarkComparison> {
    let Some(tdc_method) = tdc_method else {
        return Vec::new();
    };
    let mut output = Vec::new();
    for decoy_free in summaries
        .iter()
        .filter(|row| matches!(row.mode, ValidationMode::DecoyFree))
    {
        let comparable_tdc_layer = if decoy_free.layer == "level4" {
            "reportable_q"
        } else {
            decoy_free.layer.as_str()
        };
        let exact_tdc = summaries.iter().find(|row| {
            row.method == tdc_method
                && matches!(row.mode, ValidationMode::Tdc)
                && row.stage == decoy_free.stage
                && row.layer == comparable_tdc_layer
        });
        let compatible_target_only_tdc = is_target_only_stage(&decoy_free.stage).then(|| {
            summaries.iter().find(|row| {
                row.method == tdc_method
                    && matches!(row.mode, ValidationMode::Tdc)
                    && is_target_only_stage(&row.stage)
                    && row.layer == comparable_tdc_layer
            })
        });
        if let Some(tdc) = exact_tdc.or_else(|| compatible_target_only_tdc.flatten()) {
            let peptide_difference = decoy_free.peptide.target as i64 - tdc.peptide.target as i64;
            let calibration = if is_target_only_stage(&decoy_free.stage) {
                let requested = decoy_free.calibration_stage.as_deref();
                match requested {
                    Some(stage) => summaries.iter().find(|row| {
                        row.method == decoy_free.method
                            && row.stage == stage
                            && row.layer == decoy_free.layer
                            && matches!(row.mode, ValidationMode::DecoyFree)
                    }),
                    None => ["ms2rescore", "optimized"].into_iter().find_map(|stage| {
                        summaries.iter().find(|row| {
                            row.method == decoy_free.method
                                && row.stage == stage
                                && row.layer == decoy_free.layer
                                && matches!(row.mode, ValidationMode::DecoyFree)
                        })
                    }),
                }
            } else {
                Some(decoy_free)
            };
            let calibration_fdp = calibration.and_then(|row| row.peptide.combined_entrapment_fdp);
            let calibrated = calibration_fdp.is_some_and(|fdp| fdp <= maximum_fdp);
            output.push(TdcBenchmarkComparison {
                decoy_free_method: decoy_free.method.clone(),
                tdc_method: tdc_method.into(),
                stage: decoy_free.stage.clone(),
                layer: decoy_free.layer.clone(),
                tdc_layer: tdc.layer.clone(),
                psm_yield_difference: decoy_free.psm.target as i64 - tdc.psm.target as i64,
                peptide_yield_difference: peptide_difference,
                peptidoform_yield_difference: decoy_free.peptidoform.target as i64
                    - tdc.peptidoform.target as i64,
                protein_yield_difference: decoy_free.protein.target as i64
                    - tdc.protein.target as i64,
                peptide_entrapment_fdp: calibration_fdp,
                calibration_stage: calibration.map(|row| row.stage.clone()),
                calibration_constrained: calibrated,
                improves_peptide_yield: calibrated && peptide_difference > 0,
                target_only_calibration_policy: decoy_free.target_only_calibration_policy,
                release_candidate: decoy_free.release_candidate,
            });
        }
    }
    output
}

pub fn stage_comparisons(summaries: &[RunValidationSummary]) -> Vec<StageComparison> {
    let mut output = Vec::new();
    let methods = summaries
        .iter()
        .map(|row| row.method.as_str())
        .collect::<BTreeSet<_>>();
    for method in methods {
        for layer in ["raw_q", "level4", "reportable_q"] {
            let mut pairs = vec![("optimized", "ms2rescore")];
            let target_stages = summaries
                .iter()
                .filter(|row| {
                    row.method == method && row.layer == layer && is_target_only_stage(&row.stage)
                })
                .map(|row| row.stage.as_str())
                .collect::<BTreeSet<_>>();
            for target_stage in &target_stages {
                pairs.push(("ms2rescore", target_stage));
                pairs.push(("optimized", target_stage));
            }
            let refit_stage = TargetOnlyCalibrationPolicy::RefitWithLockedWindow.stage_name();
            let reuse_stage = TargetOnlyCalibrationPolicy::ReuseDatasetArtifact.stage_name();
            if target_stages.contains(refit_stage) && target_stages.contains(reuse_stage) {
                pairs.push((refit_stage, reuse_stage));
            }
            for (from_stage, to_stage) in pairs {
                let from = summaries.iter().find(|row| {
                    row.method == method && row.layer == layer && row.stage == from_stage
                });
                let to = summaries.iter().find(|row| {
                    row.method == method && row.layer == layer && row.stage == to_stage
                });
                if let (Some(from), Some(to)) = (from, to) {
                    output.push(StageComparison {
                        method: method.into(),
                        layer: layer.into(),
                        from_stage: from_stage.into(),
                        to_stage: to_stage.into(),
                        target_psm_change: to.psm.target as i64 - from.psm.target as i64,
                        target_peptide_change: to.peptide.target as i64
                            - from.peptide.target as i64,
                        target_peptidoform_change: to.peptidoform.target as i64
                            - from.peptidoform.target as i64,
                        target_protein_change: to.protein.target as i64
                            - from.protein.target as i64,
                        peptide_fdp_change: match (
                            from.peptide.combined_entrapment_fdp,
                            to.peptide.combined_entrapment_fdp,
                        ) {
                            (Some(a), Some(b)) => Some(b - a),
                            _ => None,
                        },
                    });
                }
            }
        }
    }
    output
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertQualityGate {
    pub model: String,
    pub eligible: bool,
    pub reasons: Vec<String>,
    pub validation_stage: Option<String>,
    pub validation_layer: String,
    pub target_peptides: usize,
    pub entrapment_peptides: usize,
    pub peptide_fdp: Option<f64>,
    pub underpowered: bool,
    pub warnings: Vec<String>,
}

pub fn expert_quality_gates(
    summaries: &[RunValidationSummary],
    stability: &[TransferStability],
    maximum_fdp: f64,
    minimum_entrapment_peptides: usize,
    validation_layer: &str,
) -> Vec<ExpertQualityGate> {
    let methods = summaries
        .iter()
        .filter(|row| row.method != "ensemble" && matches!(row.mode, ValidationMode::DecoyFree))
        .map(|row| row.method.as_str())
        .collect::<BTreeSet<_>>();
    methods
        .into_iter()
        .map(|method| {
            let release_target = summaries
                .iter()
                .find(|row| {
                    row.method == method
                        && is_target_only_stage(&row.stage)
                        && row.layer == validation_layer
                        && row.release_candidate
                });
            let selected_stage = release_target.and_then(|row| row.calibration_stage.as_deref());
            let best = if release_target.is_some() {
                selected_stage.and_then(|stage| {
                    summaries.iter().find(|row| {
                        row.method == method
                            && row.stage == stage
                            && row.layer == validation_layer
                    })
                })
            } else {
                ["ms2rescore", "optimized"].into_iter().find_map(|stage| {
                    summaries.iter().find(|row| {
                        row.method == method
                            && row.stage == stage
                            && row.layer == validation_layer
                    })
                })
            };
            let mut reasons = Vec::new();
            let mut warnings = Vec::new();
            if release_target.is_some() && selected_stage.is_none() {
                reasons.push("target-only calibration provenance is missing".into());
            }
            if best.is_none() {
                reasons.push(format!("missing {validation_layer} entrapment validation"));
            }
            let target_peptides = best.map(|row| row.peptide.target).unwrap_or(0);
            let entrapment_peptides = best.map(|row| row.peptide.entrapment).unwrap_or(0);
            let fdp = best.and_then(|row| row.peptide.combined_entrapment_fdp);
            if target_peptides == 0 {
                reasons.push("no accepted target peptides".into());
            }
            match fdp {
                None => reasons.push("peptide entrapment FDP is missing".into()),
                Some(value) if value > maximum_fdp => {
                    reasons.push("peptide entrapment FDP is above threshold".into())
                }
                Some(_) => {}
            }
            if entrapment_peptides < minimum_entrapment_peptides {
                warnings.push(format!(
                    "accepted entrapment peptide count {entrapment_peptides} is below the stability minimum {minimum_entrapment_peptides}"
                ));
            }
            if stability
                .iter()
                .find(|row| row.method == method && row.release_candidate)
                .is_some_and(|row| !row.stable)
            {
                reasons.push("target-only search-space transfer is unstable".into());
            }
            ExpertQualityGate {
                model: method.into(),
                eligible: reasons.is_empty(),
                reasons,
                validation_stage: best.map(|row| row.stage.clone()),
                validation_layer: validation_layer.into(),
                target_peptides,
                entrapment_peptides,
                peptide_fdp: fdp,
                underpowered: entrapment_peptides < minimum_entrapment_peptides,
                warnings,
            }
        })
        .collect()
}

pub fn transfer_stability(
    summaries: &[RunValidationSummary],
    maximum_fraction_loss: f64,
) -> Vec<TransferStability> {
    let reportable = summaries
        .iter()
        .filter(|summary| summary.layer == "level4" || summary.layer == "reportable_q")
        .collect::<Vec<_>>();
    let mut by_method: BTreeMap<&str, Vec<&RunValidationSummary>> = BTreeMap::new();
    for summary in reportable {
        by_method.entry(&summary.method).or_default().push(summary);
    }
    let fraction =
        |from: usize, to: usize| (from > 0).then_some((to as f64 - from as f64) / from as f64);
    let mut output = Vec::new();
    for (method, rows) in by_method {
        let targets = rows
            .iter()
            .filter(|row| is_target_only_stage(&row.stage))
            .copied()
            .collect::<Vec<_>>();
        for to in targets {
            let from = match to.calibration_stage.as_deref() {
                Some(stage) => rows.iter().find(|row| row.stage == stage),
                None => rows
                    .iter()
                    .find(|row| row.stage == "ms2rescore")
                    .or_else(|| rows.iter().find(|row| row.stage == "optimized")),
            };
            let Some(from) = from else {
                continue;
            };
            let psm = fraction(from.psm.target, to.psm.target);
            let peptide = fraction(from.peptide.target, to.peptide.target);
            let peptidoform = fraction(from.peptidoform.target, to.peptidoform.target);
            let protein = fraction(from.protein.target, to.protein.target);
            let stable = [psm, peptide, peptidoform, protein]
                .into_iter()
                .flatten()
                .all(|change| change >= -maximum_fraction_loss);
            output.push(TransferStability {
                method: method.into(),
                from_stage: from.stage.clone(),
                to_stage: to.stage.clone(),
                psm_fraction_change: psm,
                peptide_fraction_change: peptide,
                peptidoform_fraction_change: peptidoform,
                protein_fraction_change: protein,
                stable,
                target_only_calibration_policy: to.target_only_calibration_policy,
                release_candidate: to.release_candidate,
            });
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(
        method: &str,
        stage: &str,
        layer: &str,
        targets: usize,
        entrapments: usize,
    ) -> RunValidationSummary {
        let count = IdentificationCount {
            target: targets,
            entrapment: entrapments,
            combined_entrapment_fdp: Some(
                (2.0 * entrapments as f64 / (targets + entrapments).max(1) as f64).min(1.0),
            ),
        };
        RunValidationSummary {
            method: method.into(),
            stage: stage.into(),
            results: PathBuf::new(),
            mode: ValidationMode::DecoyFree,
            calibration_stage: None,
            target_only_calibration_policy: None,
            release_candidate: true,
            complete: true,
            search_space: "+Ent".into(),
            layer: layer.into(),
            psm: count.clone(),
            peptide: count.clone(),
            peptidoform: count.clone(),
            protein: count,
            counting_definition: "test".into(),
        }
    }

    #[test]
    fn parity_rejects_nonzero_native_when_baseline_is_zero() {
        let summaries = vec![
            summary("baseline", "optimized", "level4", 0, 0),
            summary("native", "optimized", "level4", 1, 0),
        ];
        let result = parity_comparisons(
            &summaries,
            &[ParityPair {
                baseline_method: "baseline".into(),
                native_method: "native".into(),
                stages: Vec::new(),
                layers: Vec::new(),
                maximum_fraction_difference: None,
            }],
            0.001,
        );
        assert_eq!(result.len(), 1);
        assert!(!result[0].within_tolerance);
    }

    #[test]
    fn parity_can_exclude_a_stage_with_intentionally_changed_semantics() {
        let summaries = vec![
            summary("baseline", "optimized", "level4", 100, 0),
            summary("native", "optimized", "level4", 100, 0),
            summary("baseline", "target_only", "level4", 200, 0),
            summary("native", "target_only", "level4", 100, 0),
        ];
        let result = parity_comparisons(
            &summaries,
            &[ParityPair {
                baseline_method: "baseline".into(),
                native_method: "native".into(),
                stages: vec!["optimized".into()],
                layers: Vec::new(),
                maximum_fraction_difference: None,
            }],
            0.001,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].stage, "optimized");
        assert!(result[0].within_tolerance);
    }

    #[test]
    fn every_declared_parity_stage_and_layer_requires_local_evidence() {
        let pair = ParityPair {
            baseline_method: "baseline".into(),
            native_method: "native".into(),
            stages: vec!["optimized".into(), "ms2rescore".into()],
            layers: vec!["level4".into()],
            maximum_fraction_difference: None,
        };
        let comparisons = parity_comparisons(
            &[
                summary("baseline", "optimized", "level4", 100, 0),
                summary("native", "optimized", "level4", 100, 0),
            ],
            std::slice::from_ref(&pair),
            0.001,
        );
        let missing = missing_parity_evidence(&[pair], &comparisons);
        assert_eq!(missing.len(), 1);
        assert!(missing[0].contains("stage=ms2rescore"));
        assert!(missing[0].contains("layer=level4"));
    }

    #[test]
    fn transfer_uses_optimized_when_ms2rescore_is_not_run() {
        let summaries = vec![
            summary("moments", "optimized", "level4", 100, 1),
            summary("moments", "target_only", "level4", 95, 0),
        ];
        let result = transfer_stability(&summaries, 0.10);
        assert_eq!(result.len(), 1);
        assert!(result[0].stable);
        assert_eq!(result[0].from_stage, "optimized");
    }

    #[test]
    fn compare_both_keeps_separate_transfer_evidence_and_release_status() {
        let mut refit = summary(
            "moments",
            TargetOnlyCalibrationPolicy::RefitWithLockedWindow.stage_name(),
            "level4",
            95,
            0,
        );
        refit.target_only_calibration_policy =
            Some(TargetOnlyCalibrationPolicy::RefitWithLockedWindow);
        let mut reuse = summary(
            "moments",
            TargetOnlyCalibrationPolicy::ReuseDatasetArtifact.stage_name(),
            "level4",
            60,
            0,
        );
        reuse.target_only_calibration_policy =
            Some(TargetOnlyCalibrationPolicy::ReuseDatasetArtifact);
        reuse.release_candidate = false;
        let result = transfer_stability(
            &[
                summary("moments", "optimized", "level4", 100, 1),
                refit,
                reuse,
            ],
            0.10,
        );
        assert_eq!(result.len(), 2);
        assert!(result.iter().any(|row| row.release_candidate && row.stable));
        assert!(result
            .iter()
            .any(|row| !row.release_candidate && !row.stable));
    }

    #[test]
    fn legacy_target_only_parity_maps_only_to_refit_semantics() {
        let baseline = summary("legacy", "target_only", "level4", 100, 0);
        let refit = summary(
            "native",
            TargetOnlyCalibrationPolicy::RefitWithLockedWindow.stage_name(),
            "level4",
            100,
            0,
        );
        let mut reuse = summary(
            "native",
            TargetOnlyCalibrationPolicy::ReuseDatasetArtifact.stage_name(),
            "level4",
            50,
            0,
        );
        reuse.release_candidate = false;
        let result = parity_comparisons(
            &[baseline, refit, reuse],
            &[ParityPair {
                baseline_method: "legacy".into(),
                native_method: "native".into(),
                stages: vec!["target_only".into()],
                layers: vec!["level4".into()],
                maximum_fraction_difference: None,
            }],
            0.0,
        );
        assert_eq!(result.len(), 1);
        assert!(result[0].within_tolerance);
    }

    #[test]
    fn expert_gate_warns_about_underpowered_zero_entrapment_result() {
        let summaries = vec![summary("moments", "optimized", "level4", 500, 0)];
        let gates = expert_quality_gates(&summaries, &[], 0.01, 3, "level4");
        assert_eq!(gates.len(), 1);
        assert!(gates[0].eligible);
        assert!(gates[0].underpowered);
        assert!(gates[0]
            .warnings
            .iter()
            .any(|reason| reason.contains("stability minimum")));
    }

    #[test]
    fn expert_gate_uses_the_artifact_selected_for_target_only() {
        let optimized = summary("moments", "optimized", "level4", 500, 0);
        let rejected_ms2 = summary("moments", "ms2rescore", "level4", 510, 10);
        let mut target_only = summary("moments", "target_only", "level4", 505, 0);
        target_only.calibration_stage = Some("optimized".into());
        let gates = expert_quality_gates(
            &[optimized, rejected_ms2, target_only],
            &[],
            0.01,
            3,
            "level4",
        );
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].validation_stage.as_deref(), Some("optimized"));
        assert!(gates[0].eligible);
    }

    #[test]
    fn stage_report_includes_direct_optimized_to_target_only_change() {
        let summaries = vec![
            summary("moments", "optimized", "level4", 100, 1),
            summary("moments", "ms2rescore", "level4", 105, 1),
            summary("moments", "target_only", "level4", 103, 0),
        ];
        let comparisons = stage_comparisons(&summaries);
        assert!(comparisons.iter().any(|comparison| {
            comparison.from_stage == "optimized"
                && comparison.to_stage == "target_only"
                && comparison.target_peptide_change == 3
        }));
    }

    #[test]
    fn target_only_tdc_comparison_uses_declared_entrapment_calibration() {
        let calibration = summary("moments", "optimized", "level4", 100, 0);
        let mut target_only = summary("moments", "target_only", "level4", 110, 0);
        target_only.calibration_stage = Some("optimized".into());
        target_only.peptide.combined_entrapment_fdp = None;
        let mut tdc = summary("tdc_primary", "target_only", "reportable_q", 105, 0);
        tdc.mode = ValidationMode::Tdc;
        tdc.peptide.combined_entrapment_fdp = None;
        let mut secondary_tdc = summary("tdc_secondary", "target_only", "level4", 50, 0);
        secondary_tdc.mode = ValidationMode::Tdc;

        let comparisons = tdc_benchmark_comparisons(
            &[calibration, target_only, tdc, secondary_tdc],
            Some("tdc_primary"),
            0.01,
        );
        assert_eq!(comparisons.len(), 1);
        assert_eq!(comparisons[0].stage, "target_only");
        assert_eq!(
            comparisons[0].calibration_stage.as_deref(),
            Some("optimized")
        );
        assert!(comparisons[0].calibration_constrained);
        assert!(comparisons[0].improves_peptide_yield);
        assert_eq!(comparisons[0].peptide_yield_difference, 5);
    }

    #[test]
    fn missing_declared_calibration_never_falls_back_to_another_stage() {
        let optimized = summary("moments", "optimized", "level4", 100, 0);
        let mut target_only = summary("moments", "target_only", "level4", 110, 0);
        target_only.calibration_stage = Some("missing_calibration".into());
        let mut tdc = summary("tdc", "target_only", "reportable_q", 105, 0);
        tdc.mode = ValidationMode::Tdc;

        let summaries = [optimized, target_only, tdc];
        let gates = expert_quality_gates(&summaries, &[], 0.01, 3, "level4");
        assert!(!gates[0].eligible);
        assert!(gates[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("missing level4 entrapment validation")));
        assert!(transfer_stability(&summaries, 0.20).is_empty());

        let comparisons = tdc_benchmark_comparisons(&summaries, Some("tdc"), 0.01);
        assert_eq!(comparisons.len(), 1);
        assert!(!comparisons[0].calibration_constrained);
        assert!(comparisons[0].calibration_stage.is_none());
    }

    #[test]
    fn summaries_count_unmodified_peptides_and_peptidoforms_separately() {
        let path = std::env::temp_dir().join(format!(
            "sage-peptidoform-summary-{}-{}.tsv",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(
            &path,
            "psm_id\trank\tlabel\tproteins\tpeptide\tdecoy_free_q_value\tdecoy_free_peptide_q\tdecoy_free_protein_q\tdecoy_free_protein_supported_peptide\tdecoy_free_peptide_supported_psm\n\
             p1\t1\t1\tTarget_A\tPEPTIDE\t0.001\t0.001\t0.001\ttrue\ttrue\n\
             p2\t1\t1\tTarget_A\tPEPTI[+15.99]DE\t0.001\t0.001\t0.001\ttrue\ttrue\n\
             e1\t1\t1\tEnt_A\tPEPTIDE\t0.001\t0.001\t0.001\ttrue\ttrue\n",
        )
        .unwrap();
        let rows = summarize_run(
            &ValidationRun {
                method: "moments".into(),
                stage: "optimized".into(),
                results: path.clone(),
                mode: ValidationMode::DecoyFree,
                expected_search_space: Some("+Ent".into()),
                calibration_stage: None,
                target_only_calibration_policy: None,
                release_candidate: true,
            },
            &EffectiveRatios::default(),
            0.01,
        )
        .unwrap();
        let level4 = rows.iter().find(|row| row.layer == "level4").unwrap();
        assert_eq!(level4.peptide.target, 1);
        assert_eq!(level4.peptidoform.target, 2);
        assert_eq!(level4.peptidoform.entrapment, 1);
        assert!(level4.counting_definition.contains("peptidoform retains"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn ensemble_interaction_reports_raw_warning_separately_from_level4_gate() {
        let mut baseline_raw = summary("baseline", "optimized", "raw_q", 1_000, 5);
        let mut final_raw = summary("ensemble", "optimized", "raw_q", 1_010, 12);
        baseline_raw.psm.combined_entrapment_fdp = Some(0.005);
        baseline_raw.peptide.combined_entrapment_fdp = Some(0.004);
        baseline_raw.peptidoform.combined_entrapment_fdp = Some(0.006);
        final_raw.psm.combined_entrapment_fdp = Some(0.016);
        final_raw.peptide.combined_entrapment_fdp = Some(0.015);
        final_raw.peptidoform.combined_entrapment_fdp = Some(0.018);
        let mut baseline_level4 = summary("baseline", "optimized", "level4", 900, 3);
        let mut final_level4 = summary("ensemble", "optimized", "level4", 910, 3);
        baseline_level4.peptide.combined_entrapment_fdp = Some(0.007);
        final_level4.peptide.combined_entrapment_fdp = Some(0.006);

        let baseline = vec![baseline_raw, baseline_level4];
        let final_result = vec![final_raw, final_level4];
        let report = ensemble_interaction_calibration(
            &baseline,
            &final_result,
            &EffectiveRatios {
                psm: 2.0,
                peptide: 3.0,
                protein: 4.0,
            },
            0.01,
            0.01,
            vec!["mle".into(), "moments".into()],
            vec!["lower_order".into(), "moments".into(), "mle".into()],
        )
        .unwrap();
        let repeated = ensemble_interaction_calibration(
            &baseline,
            &final_result,
            &EffectiveRatios {
                psm: 2.0,
                peptide: 3.0,
                protein: 4.0,
            },
            0.01,
            0.01,
            vec!["moments".into(), "mle".into()],
            vec!["mle".into(), "moments".into(), "lower_order".into()],
        )
        .unwrap();

        assert_eq!(report.baseline_experts, vec!["mle", "moments"]);
        assert_eq!(report.newly_participating_experts, vec!["lower_order"]);
        assert!(report.final_level4_calibration_pass);
        assert_eq!(
            serde_json::to_vec(&report).unwrap(),
            serde_json::to_vec(&repeated).unwrap()
        );
        let warning = report.raw_q_warning.as_ref().unwrap();
        assert_eq!(warning.code, "raw_q_ensemble_interaction_deterioration");
        assert_eq!(
            warning.affected_levels,
            vec!["psm", "peptide", "peptidoform"]
        );
        assert!(warning.message.contains("not a passing calibration gate"));
        let raw_q = report.raw_q.as_ref().unwrap();
        assert_eq!(raw_q.peptide.measured_entrapment_ratio, 3.0);
        assert_eq!(raw_q.peptide.final_fdp_numerator, Some(16.0));
        assert_eq!(raw_q.peptide.final_fdp_denominator, 1_022);
        assert_eq!(raw_q.peptide.absolute_fdp_change, Some(0.011));
    }

    #[test]
    fn invalid_final_level4_peptide_calibration_does_not_pass() {
        let baseline = vec![
            summary("baseline", "optimized", "raw_q", 1_000, 1),
            summary("baseline", "optimized", "level4", 900, 1),
        ];
        let mut final_result = vec![
            summary("ensemble", "optimized", "raw_q", 1_001, 1),
            summary("ensemble", "optimized", "level4", 901, 1),
        ];
        final_result[1].peptide.combined_entrapment_fdp = None;
        let report = ensemble_interaction_calibration(
            &baseline,
            &final_result,
            &EffectiveRatios::default(),
            0.01,
            0.01,
            vec!["moments".into()],
            vec!["moments".into(), "lower_order".into()],
        )
        .unwrap();
        assert!(!report.evaluable);
        assert!(!report.final_level4_calibration_pass);
        assert!(report
            .not_evaluable_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("missing")));
    }
}
