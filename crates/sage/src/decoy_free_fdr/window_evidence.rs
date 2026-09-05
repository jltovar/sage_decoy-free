//! Diagnostics are observations of the existing computation, not extra
//! feasibility gates. In particular a defined zero-entrapment estimate is not
//! the same thing as an undefined estimate from an empty accepted population.
use super::*;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
pub fn artifact_contains_model(
    artifacts: &crate::decoy_free_fdr::DfRunArtifacts,
    model: &ModelFit,
) -> bool {
    let valid_skew_normal = |model: &crate::ml::skew_normal::SkewNormal| {
        model.location.is_finite()
            && model.scale.is_finite()
            && model.scale > 0.0
            && model.shape.is_finite()
    };
    match model {
        ModelFit::Moments => artifacts.moments.as_ref().is_some_and(|artifact| {
            artifact.schema_version == 1
                && artifact.model_version == "sage-moments-gumbel-v1"
                && artifact.min_rank > 1
                && artifact.max_rank >= artifact.min_rank
                && artifact.mu.is_finite()
                && artifact.beta.is_finite()
                && artifact.beta > 0.0
        }),
        ModelFit::Mle => artifacts.mle.as_ref().is_some_and(|artifact| {
            artifact.schema_version == 1
                && artifact.model_version == "sage-mle-gumbel-v1"
                && artifact.min_rank > 1
                && artifact.max_rank >= artifact.min_rank
                && artifact.mu.is_finite()
                && artifact.beta.is_finite()
                && artifact.beta > 0.0
        }),
        ModelFit::LowerOrder => artifacts.lower_order.as_ref().is_some_and(|artifact| {
            crate::ml::lower_order::LowerOrderModel::from_artifact(artifact).is_ok()
        }),
        ModelFit::Msfdr => {
            artifacts.msfdr_seeded.as_ref().is_some_and(|model| {
                model.null_loc.is_finite()
                    && model.null_scale.is_finite()
                    && model.null_scale > 0.0
                    && model.target_mean.is_finite()
                    && model.target_std.is_finite()
                    && model.target_std > 0.0
                    && model.target_alpha.is_finite()
                    && model.pi.is_finite()
                    && (0.0..=1.0).contains(&model.pi)
            }) && artifacts
                .msfdr_seeded_metadata
                .as_ref()
                .is_some_and(|metadata| {
                    metadata.schema_version == 1
                        && metadata.model_version == "sage-msfdr-seeded-v1"
                        && !metadata.rank1_only
                        && metadata.min_null_rank.is_some_and(|rank| rank > 1)
                        && metadata
                            .max_null_rank
                            .zip(metadata.min_null_rank)
                            .is_some_and(|(max, min)| max >= min)
                })
        }
        ModelFit::Msfdr1Smix => {
            artifacts.msfdr_1smix.as_ref().is_some_and(|model| {
                valid_skew_normal(&model.correct)
                    && valid_skew_normal(&model.incorrect1)
                    && model.a.is_finite()
                    && (0.0..=1.0).contains(&model.a)
            }) && artifacts
                .msfdr_1smix_metadata
                .as_ref()
                .is_some_and(|metadata| {
                    metadata.schema_version == 1
                        && metadata.model_version == "sage-msfdr-1smix-v1"
                        && metadata.rank1_only
                        && metadata.min_null_rank.is_none()
                        && metadata.max_null_rank.is_none()
                })
        }
        ModelFit::Msfdr2Smix => {
            artifacts.msfdr_2smix.as_ref().is_some_and(|model| {
                valid_skew_normal(&model.correct)
                    && valid_skew_normal(&model.incorrect1)
                    && valid_skew_normal(&model.incorrect2)
                    && model.a.is_finite()
                    && model.b.is_finite()
                    && (0.0..=1.0).contains(&model.a)
                    && (0.0..=1.0).contains(&model.b)
                    && model.a + model.b <= 1.0 + f64::EPSILON
            }) && artifacts
                .msfdr_2smix_metadata
                .as_ref()
                .is_some_and(|metadata| {
                    metadata.schema_version == 1
                        && metadata.model_version == "sage-msfdr-2smix-v1"
                        && !metadata.rank1_only
                        && metadata.min_null_rank.is_some_and(|rank| rank > 1)
                        && metadata
                            .max_null_rank
                            .zip(metadata.min_null_rank)
                            .is_some_and(|(max, min)| max >= min)
                })
        }
        ModelFit::Nokoi => artifacts
            .nokoi
            .as_ref()
            .is_some_and(|artifact| artifact.validate_portable().is_ok()),
        ModelFit::Ensemble => false,
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ComputationObservations {
    pub fallback_events: Vec<String>,
    pub fit_events: Vec<serde_json::Value>,
    pub q_methods: Vec<serde_json::Value>,
}

thread_local! {
    // The native window evaluator and its fit/q dispatch run synchronously on
    // the calling thread. No global logger parsing or cross-trial state.
    static OBSERVATIONS: RefCell<Option<ComputationObservations>> = const { RefCell::new(None) };
}

pub(super) struct ObservationGuard;
impl ObservationGuard {
    pub(super) fn start() -> Self {
        OBSERVATIONS.with(|cell| {
            assert!(
                cell.borrow().is_none(),
                "nested window diagnostic collection"
            );
            *cell.borrow_mut() = Some(ComputationObservations::default());
        });
        Self
    }
    pub(super) fn snapshot(&self) -> ComputationObservations {
        OBSERVATIONS.with(|cell| cell.borrow().clone().unwrap_or_default())
    }
}
impl Drop for ObservationGuard {
    fn drop(&mut self) {
        OBSERVATIONS.with(|cell| *cell.borrow_mut() = None);
    }
}

pub(super) fn observe_fallback(reason: String) {
    OBSERVATIONS.with(|cell| {
        if let Some(observations) = cell.borrow_mut().as_mut() {
            observations.fallback_events.push(reason);
        }
    });
}

pub(super) fn observe_fit(event: serde_json::Value) {
    OBSERVATIONS.with(|cell| {
        if let Some(observations) = cell.borrow_mut().as_mut() {
            observations.fit_events.push(event);
        }
    });
}

pub(super) fn observe_q(level: &str, report: &QValueComputation) {
    OBSERVATIONS.with(|cell| {
        if let Some(observations) = cell.borrow_mut().as_mut() {
            observations.q_methods.push(serde_json::json!({
                "level": level,
                "requested": report.requested_method,
                "effective": report.effective_method,
                "actual": report.actual_method,
                "fallback_reason": report.fallback_reason,
                "pi0": report.pi0,
                "values": report.q_values.len(),
                "invalid_values": report.q_values.iter().filter(|v| !v.is_finite() || **v < 0.0 || **v > 1.0).count(),
            }));
        }
    });
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FdpPredicateEvidence {
    pub level: String,
    pub targets: usize,
    pub selection_entrapments: usize,
    pub measured_ratio: f64,
    pub numerator: Option<f64>,
    pub denominator: usize,
    pub value: Option<f64>,
    pub limit: f64,
    pub predicate: String,
    pub underpowered: bool,
}
impl FdpPredicateEvidence {
    pub fn new(
        level: &str,
        targets: usize,
        entrapments: usize,
        ratio: f64,
        limit: f64,
        minimum: usize,
    ) -> Self {
        let value = entrapment_fdp(targets, entrapments, ratio);
        Self {
            level: level.into(),
            targets,
            selection_entrapments: entrapments,
            measured_ratio: ratio,
            numerator: (ratio.is_finite() && ratio > 0.0)
                .then(|| entrapments as f64 * (1.0 + 1.0 / ratio)),
            denominator: targets + entrapments,
            value,
            limit,
            predicate: match value {
                None => "unavailable_empirical_metric",
                Some(v) if v > limit => "empirical_fdp_above_limit",
                Some(_) => "passed",
            }
            .into(),
            underpowered: entrapments < minimum,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NullWindowEvidence {
    pub schema_version: u32,
    pub effective_settings_sha256: String,
    pub model: ModelFit,
    pub hierarchical_reporting: HierarchicalReportingMode,
    pub hierarchical_entrapment_validation: bool,
    pub numerical_fit_valid: bool,
    pub fitted_artifact: serde_json::Value,
    pub observed_invalid_probability_values: usize,
    pub missing_required_probability_values: usize,
    pub numerical_evaluation_valid: bool,
    pub rank1_rows: usize,
    pub rank1_with_psm_q: usize,
    pub rank1_with_peptide_q: usize,
    pub rank1_with_protein_q: usize,
    pub annotation_state: serde_json::Value,
    pub observations: ComputationObservations,
    pub predicates: Vec<FdpPredicateEvidence>,
    pub outcome: String,
}

fn compact_fitted_state(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(entries) if entries.len() > 64 => {
            let bytes = serde_json::to_vec(&entries).expect("JSON value serialization");
            serde_json::json!({"compact_array_entries": entries.len(), "sha256": format!("{:x}", Sha256::digest(bytes))})
        }
        serde_json::Value::Array(entries) => {
            entries.into_iter().map(compact_fitted_state).collect()
        }
        serde_json::Value::Object(entries) => serde_json::Value::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key, compact_fitted_state(value)))
                .collect(),
        ),
        scalar => scalar,
    }
}

impl NullWindowEvidence {
    pub(super) fn collect(
        settings: &FdrSettings,
        artifacts: &DfRunArtifacts,
        features: &[DfFeature],
        observations: ComputationObservations,
        predicates: Vec<FdpPredicateEvidence>,
    ) -> Result<Self, String> {
        let numerical_fit_valid = artifact_contains_model(artifacts, &settings.model_fit);
        let rank1: Vec<_> = features
            .iter()
            .filter(|f| f.core.rank == 1 && f.core.label == 1)
            .collect();
        let invalid = rank1
            .iter()
            .flat_map(|f| {
                [
                    f.decoy_free_p_value,
                    f.decoy_free_pep,
                    f.decoy_free_q_value,
                    f.decoy_free_peptide_q,
                    f.decoy_free_protein_q,
                ]
            })
            .flatten()
            .filter(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
            .count();
        let invalid = invalid
            + observations
                .q_methods
                .iter()
                .filter_map(|q| q["invalid_values"].as_u64())
                .sum::<u64>() as usize;
        let missing_required = rank1
            .iter()
            .map(|f| {
                let active = match active_evidence_space(settings) {
                    ActiveEvidenceSpace::PValue => f.decoy_free_p_value,
                    ActiveEvidenceSpace::Pep => f.decoy_free_pep,
                };
                [
                    active,
                    f.decoy_free_q_value,
                    f.decoy_free_peptide_q,
                    f.decoy_free_protein_q,
                ]
                .iter()
                .filter(|v| v.is_none())
                .count()
            })
            .sum::<usize>();
        let numerical_evaluation_valid =
            numerical_fit_valid && invalid == 0 && missing_required == 0;
        let outcome = if !numerical_evaluation_valid {
            "technical_failure"
        } else if predicates.iter().any(|p| p.value.is_none()) {
            "unavailable_empirical_metrics"
        } else if predicates
            .iter()
            .any(|p| p.predicate == "empirical_fdp_above_limit")
        {
            "empirically_infeasible"
        } else {
            "feasible"
        };
        let settings_bytes = serde_json::to_vec(settings).map_err(|e| e.to_string())?;
        // Store the selected expert artifact, not all candidates or prediction
        // values. This records the actual fit rather than inferring validity
        // from having reached the evaluator.
        let artifact_bytes = serde_json::to_vec(artifacts).map_err(|e| e.to_string())?;
        let fitted_artifact = serde_json::json!({
            "serialized_sha256": format!("{:x}", Sha256::digest(&artifact_bytes)),
            "serialized_bytes": artifact_bytes.len(),
            "state": compact_fitted_state(serde_json::to_value(artifacts).map_err(|e| e.to_string())?),
        });
        Ok(Self {
            schema_version: 1,
            effective_settings_sha256: format!("{:x}", Sha256::digest(settings_bytes)),
            model: settings.model_fit.clone(),
            hierarchical_reporting: settings.hierarchical_reporting,
            hierarchical_entrapment_validation: settings.hierarchical_entrapment_validation,
            numerical_fit_valid,
            fitted_artifact,
            observed_invalid_probability_values: invalid,
            missing_required_probability_values: missing_required,
            numerical_evaluation_valid,
            rank1_rows: rank1.len(),
            rank1_with_psm_q: rank1
                .iter()
                .filter(|f| f.decoy_free_q_value.is_some())
                .count(),
            rank1_with_peptide_q: rank1
                .iter()
                .filter(|f| f.decoy_free_peptide_q.is_some())
                .count(),
            rank1_with_protein_q: rank1
                .iter()
                .filter(|f| f.decoy_free_protein_q.is_some())
                .count(),
            annotation_state: serde_json::json!({
                "phase": "native_before_external_feature_join",
                "rows": features.len(),
                "external_feature_joined_rows": features.iter().filter(|f| f.core.external_features.ms2rescore_feature_joined).count(),
            }),
            observations,
            predicates,
            outcome: outcome.into(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NullWindowFailure {
    pub schema_version: u32,
    pub message: String,
    pub classification: String,
    pub search_completed: bool,
    pub candidate_universe_size: usize,
    pub adaptive_mode: Option<String>,
    pub adaptive_mode_reason: Option<String>,
    pub global_optimum_guaranteed: bool,
    pub evaluations: Vec<NullWindowEvaluation>,
}
impl NullWindowFailure {
    pub fn no_feasible(
        evaluations: Vec<NullWindowEvaluation>,
        candidate_universe_size: usize,
        adaptive_mode: Option<String>,
        adaptive_mode_reason: Option<String>,
        global_optimum_guaranteed: bool,
    ) -> Self {
        let classes: std::collections::BTreeSet<_> = evaluations
            .iter()
            .map(|e| {
                e.evidence
                    .as_ref()
                    .map(|d| d.outcome.as_str())
                    .unwrap_or("technical_failure")
            })
            .collect();
        let classification = if classes.len() == 1 {
            *classes.first().unwrap()
        } else {
            "mixed_window_outcomes"
        }
        .to_string();
        Self {
            schema_version: 1,
            message: "no feasible evaluated window".into(),
            classification,
            search_completed: true,
            candidate_universe_size,
            adaptive_mode,
            adaptive_mode_reason,
            global_optimum_guaranteed,
            evaluations,
        }
    }
}
impl From<String> for NullWindowFailure {
    fn from(message: String) -> Self {
        Self {
            schema_version: 1,
            message,
            classification: "technical_failure".into(),
            search_completed: false,
            candidate_universe_size: 0,
            adaptive_mode: None,
            adaptive_mode_reason: None,
            global_optimum_guaranteed: false,
            evaluations: Vec::new(),
        }
    }
}
impl From<String> for Box<NullWindowFailure> {
    fn from(message: String) -> Self {
        Box::new(NullWindowFailure::from(message))
    }
}
impl std::fmt::Display for NullWindowFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}, {} evaluated windows)",
            self.message,
            self.classification,
            self.evaluations.len()
        )
    }
}
impl std::error::Error for NullWindowFailure {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_moments_fits_above_all_level4_limits_are_empirical_not_technical() {
        use crate::input::{FdrOptions, HierarchicalInferenceConfig, HierarchicalInferenceMode};
        let peptides = [
            "PEPTIDEA", "PEPTIDEC", "PEPTIDED", "PEPTIDEE", "PEPTIDEF", "PEPTIDEG",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, sequence)| crate::peptide::Peptide {
            sequence: sequence.as_bytes().to_vec().into(),
            modifications: vec![0.0; sequence.len()],
            proteins: vec![Arc::from(["target", "Ent_selection", "Ent_audit"][i / 2])],
            ..Default::default()
        })
        .collect();
        let db = IndexedDatabase {
            peptides,
            decoy_tag: "rev_".into(),
            generate_decoys: false,
            ..Default::default()
        };
        let mut features = Vec::new();
        for rank in 1..=3 {
            for i in 0..300 {
                features.push(
                    crate::scoring::FeatureCore {
                        rank,
                        label: 1,
                        psm_id: features.len(),
                        spec_id: format!("synthetic-{i}"),
                        peptide_idx: crate::database::PeptideIx((i % 6) as u32),
                        hyperscore: if rank == 1 {
                            100.0 + i as f64 / 300.0
                        } else {
                            10.0 + i as f64 / 30.0
                        },
                        ..Default::default()
                    }
                    .to_df(),
                );
            }
        }
        let mut settings = FdrSettings::from(FdrOptions {
            model_fit: Some(ModelFit::Moments),
            min_null_size: Some(10),
            min_rank_count: Some(1),
            moments_purification_factor: Some(0.0),
            psm_q_method: Some(QMethod::Bh),
            peptide_q_method: Some(QMethod::Bh),
            protein_q_method: Some(QMethod::Bh),
            hierarchical_inference: Some(HierarchicalInferenceConfig {
                enabled: true,
                mode: HierarchicalInferenceMode::ProteinAnchored,
                entrapment_validation: true,
            }),
            selection_entrapment_proteins: Some(vec!["Ent_selection".into()]),
            ..Default::default()
        });
        settings.null_window_optimizer = Some(NullWindowOptimizerOptions {
            candidates: (2..=3)
                .map(|rank| NullWindowCandidate {
                    min_rank: rank,
                    max_rank: rank,
                })
                .collect(),
            strategy: NullWindowSearchStrategy::Explicit,
            bounds: None,
            adaptive: AdaptiveNullWindowSearchOptions::default(),
            validation_scope: NullWindowValidationScope::Level4,
            fdr_threshold: 0.01,
            psm_entrapment_ratio: 0.5,
            peptide_entrapment_ratio: 0.5,
            protein_entrapment_ratio: 0.5,
            maximum_entrapment_fdp: 0.01,
            minimum_entrapment_count_for_stable_estimate: 3,
            selection_entrapment_proteins: Some(vec!["Ent_selection".into()]),
            verbose_diagnostics: false,
        });
        let failure =
            optimize_null_window_resumable_detailed(&features, &settings, &db, Vec::new(), |_| {
                Ok(())
            })
            .unwrap_err();
        assert_eq!(
            failure.classification, "empirically_infeasible",
            "{failure:?}"
        );
        assert_eq!(failure.evaluations.len(), 2);
        for row in &failure.evaluations {
            let evidence = row.evidence.as_ref().unwrap();
            assert!(evidence.numerical_fit_valid);
            assert_eq!(
                evidence.hierarchical_reporting,
                HierarchicalReportingMode::Strict
            );
            assert_eq!(evidence.observed_invalid_probability_values, 0);
            assert!(evidence
                .predicates
                .iter()
                .all(|p| p.predicate == "empirical_fdp_above_limit"));
            assert_eq!((row.target_psms, row.entrapment_psms), (100, 100));
            assert_eq!((row.target_proteins, row.entrapment_proteins), (1, 1));
        }
    }

    fn rejected_window(
        min_rank: u32,
        targets: usize,
        entrapments: usize,
        valid: bool,
    ) -> NullWindowEvaluation {
        let settings = FdrSettings::from(crate::input::FdrOptions {
            model_fit: Some(ModelFit::Moments),
            ..Default::default()
        });
        let mut artifacts = DfRunArtifacts::default();
        if valid {
            artifacts.moments = Some(FrozenGumbelParameters {
                schema_version: 1,
                model_version: "sage-moments-gumbel-v1".into(),
                min_rank,
                max_rank: min_rank,
                mu: 3.0,
                beta: 1.0,
            });
        }
        let predicates = ["psm", "peptide", "protein"]
            .into_iter()
            .map(|level| FdpPredicateEvidence::new(level, targets, entrapments, 0.5, 0.01, 3))
            .collect();
        let evidence = NullWindowEvidence::collect(
            &settings,
            &artifacts,
            &[],
            ComputationObservations::default(),
            predicates,
        )
        .unwrap();
        NullWindowEvaluation {
            evidence: Some(evidence),
            min_rank,
            max_rank: min_rank,
            validation_scope: NullWindowValidationScope::RawQ,
            target_psms: targets,
            entrapment_psms: entrapments,
            target_peptides: targets,
            entrapment_peptides: entrapments,
            target_proteins: targets,
            entrapment_proteins: entrapments,
            psm_fdp: entrapment_fdp(targets, entrapments, 0.5),
            peptide_fdp: entrapment_fdp(targets, entrapments, 0.5),
            protein_fdp: entrapment_fdp(targets, entrapments, 0.5),
            feasible: false,
            low_count_warning: entrapments < 3,
            selected: false,
            elapsed_milliseconds: 1,
        }
    }

    #[test]
    fn complete_rejected_window_replay_never_reevaluates_and_retains_mixed_reasons() {
        let mut settings = FdrSettings::from(crate::input::FdrOptions::default());
        settings.null_window_optimizer = Some(NullWindowOptimizerOptions {
            candidates: (2..=4)
                .map(|rank| NullWindowCandidate {
                    min_rank: rank,
                    max_rank: rank,
                })
                .collect(),
            strategy: NullWindowSearchStrategy::Explicit,
            bounds: None,
            adaptive: AdaptiveNullWindowSearchOptions::default(),
            validation_scope: NullWindowValidationScope::RawQ,
            fdr_threshold: 0.01,
            psm_entrapment_ratio: 0.5,
            peptide_entrapment_ratio: 0.5,
            protein_entrapment_ratio: 0.5,
            maximum_entrapment_fdp: 0.01,
            minimum_entrapment_count_for_stable_estimate: 3,
            selection_entrapment_proteins: None,
            verbose_diagnostics: false,
        });
        let rows = vec![
            rejected_window(2, 90, 10, true),
            rejected_window(3, 0, 0, true),
            rejected_window(4, 0, 0, false),
        ];
        let bytes = serde_json::to_vec(&rows).unwrap();
        let reopened = serde_json::from_slice(&bytes).unwrap();
        let failure = optimize_null_window_resumable_detailed(
            &[],
            &settings,
            &IndexedDatabase::default(),
            reopened,
            |_| panic!("completed window reevaluated"),
        )
        .unwrap_err();
        assert_eq!(failure.classification, "mixed_window_outcomes");
        assert!(failure.search_completed && failure.global_optimum_guaranteed);
        assert_eq!(failure.evaluations.len(), 3);
        assert_eq!(
            failure.evaluations[0].evidence.as_ref().unwrap().outcome,
            "empirically_infeasible"
        );
        assert_eq!(
            failure.evaluations[1].evidence.as_ref().unwrap().outcome,
            "unavailable_empirical_metrics"
        );
        assert_eq!(
            failure.evaluations[2].evidence.as_ref().unwrap().outcome,
            "technical_failure"
        );
        let all_excess = NullWindowFailure::no_feasible(
            vec![rejected_window(2, 90, 10, true)],
            500,
            Some("boundary".into()),
            None,
            false,
        );
        assert_eq!(all_excess.classification, "empirically_infeasible");
        assert_eq!(all_excess.message, "no feasible evaluated window");
        assert!(!all_excess.global_optimum_guaranteed);
    }

    #[test]
    fn defined_zero_entrapment_is_underpowered_not_undefined_or_rejected() {
        let zero = FdpPredicateEvidence::new("protein", 9, 0, 0.5, 0.01, 3);
        assert_eq!(zero.value, Some(0.0));
        assert_eq!(zero.predicate, "passed");
        assert!(zero.underpowered);
        let empty = FdpPredicateEvidence::new("protein", 0, 0, 0.5, 0.01, 3);
        assert_eq!(empty.value, None);
        assert_eq!(empty.predicate, "unavailable_empirical_metric");
        let excess = FdpPredicateEvidence::new("protein", 90, 10, 0.5, 0.01, 3);
        assert_eq!(excess.numerator, Some(30.0));
        assert_eq!(excess.denominator, 100);
        assert_eq!(excess.value, Some(0.3));
        assert_eq!(excess.predicate, "empirical_fdp_above_limit");
    }

    #[test]
    fn actual_fit_validity_and_fallback_are_independent_observations() {
        let settings = FdrSettings::from(crate::input::FdrOptions {
            model_fit: Some(ModelFit::Moments),
            ..Default::default()
        });
        let mut artifacts = DfRunArtifacts::default();
        let evidence = NullWindowEvidence::collect(
            &settings,
            &artifacts,
            &[],
            ComputationObservations::default(),
            vec![],
        )
        .unwrap();
        assert!(!evidence.numerical_fit_valid);
        assert_eq!(evidence.outcome, "technical_failure");
        artifacts.moments = Some(FrozenGumbelParameters {
            schema_version: 1,
            model_version: "sage-moments-gumbel-v1".into(),
            min_rank: 2,
            max_rank: 2,
            mu: 3.0,
            beta: 1.0,
        });
        let guard = ObservationGuard::start();
        observe_fallback("synthetic unpurified pool fallback".into());
        let evidence = NullWindowEvidence::collect(
            &settings,
            &artifacts,
            &[],
            guard.snapshot(),
            vec![FdpPredicateEvidence::new("protein", 90, 10, 0.5, 0.01, 3)],
        )
        .unwrap();
        assert!(evidence.numerical_fit_valid);
        assert_eq!(evidence.outcome, "empirically_infeasible");
        assert_eq!(evidence.observations.fallback_events.len(), 1);
        artifacts.moments.as_mut().unwrap().beta = -1.0;
        assert!(!artifact_contains_model(&artifacts, &ModelFit::Moments));
    }
}
