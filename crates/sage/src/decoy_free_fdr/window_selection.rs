//! Pure selection over preserved measurements. No fitting, reporting mutation,
//! label lookup or resource access is performed here.
use super::*;
use crate::input::{NullWindowFdpReferences, NullWindowSelectionPolicy};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowRankingKey {
    pub zero_selection_entrapment_proteins: bool,
    /// Protein, canonical peptide, PSM, in lexicographic priority order.
    pub excess: [f64; 3],
    pub target_yields: [usize; 3],
    pub adjusted_fdps: [f64; 3],
    pub width: u32,
    pub min_rank: u32,
    pub max_rank: u32,
}

impl WindowRankingKey {
    fn compare(&self, other: &Self) -> Ordering {
        self.zero_selection_entrapment_proteins
            .cmp(&other.zero_selection_entrapment_proteins)
            .then_with(|| compare_smaller(&self.excess, &other.excess))
            .then_with(|| self.target_yields.cmp(&other.target_yields))
            .then_with(|| compare_smaller(&self.adjusted_fdps, &other.adjusted_fdps))
            .then_with(|| other.width.cmp(&self.width))
            .then_with(|| other.min_rank.cmp(&self.min_rank))
            .then_with(|| other.max_rank.cmp(&self.max_rank))
    }
}

fn compare_smaller(left: &[f64; 3], right: &[f64; 3]) -> Ordering {
    (0..3)
        .map(|i| right[i].total_cmp(&left[i]))
        .find(|x| *x != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowSelectionDecision {
    pub min_rank: u32,
    pub max_rank: u32,
    pub technical_valid: bool,
    pub empirically_rankable: bool,
    pub selection_eligible: bool,
    pub reference_met: [Option<bool>; 3],
    pub key: Option<WindowRankingKey>,
    pub reason: String,
}

pub fn window_selection_decision(
    v: &NullWindowEvaluation,
    policy: NullWindowSelectionPolicy,
    refs: NullWindowFdpReferences,
) -> WindowSelectionDecision {
    let fdps = [v.protein_fdp, v.peptide_fdp, v.psm_fdp];
    let limits = [refs.protein, refs.peptide, refs.psm];
    let technical_valid = v
        .evidence
        .as_ref()
        .is_some_and(|e| e.numerical_fit_valid && e.numerical_evaluation_valid);
    let valid_values = fdps
        .iter()
        .all(|v| v.is_some_and(|x| x.is_finite() && (0.0..=1.0).contains(&x)));
    let nonempty = v.target_proteins > 0 && v.target_peptides > 0 && v.target_psms > 0;
    let rankable = valid_values && nonempty && v.min_rank > 1 && v.max_rank >= v.min_rank;
    let met = std::array::from_fn(|i| fdps[i].filter(|x| x.is_finite()).map(|x| x <= limits[i]));
    let eligible = if policy.is_strict() {
        v.feasible
    } else {
        technical_valid && rankable
    };
    let key = (technical_valid && rankable).then(|| {
        let values = fdps.map(|x| x.expect("validated values"));
        WindowRankingKey {
            zero_selection_entrapment_proteins: v.entrapment_proteins == 0,
            excess: std::array::from_fn(|i| (values[i] - limits[i]).max(0.0)),
            target_yields: [v.target_proteins, v.target_peptides, v.target_psms],
            adjusted_fdps: values,
            width: v.max_rank - v.min_rank + 1,
            min_rank: v.min_rank,
            max_rank: v.max_rank,
        }
    });
    WindowSelectionDecision {
        min_rank: v.min_rank,
        max_rank: v.max_rank,
        technical_valid,
        empirically_rankable: rankable,
        selection_eligible: eligible,
        reference_met: met,
        key,
        reason: if !technical_valid {
            "technical_failure"
        } else if !rankable {
            "empty_targets_or_unavailable_empirical_metrics"
        } else if eligible {
            "selection_eligible_not_a_calibration_claim"
        } else {
            "strict_empirical_constraint_rejection"
        }
        .into(),
    }
}

pub(super) fn compare_for_policy(
    left: &NullWindowEvaluation,
    right: &NullWindowEvaluation,
    options: &NullWindowOptimizerOptions,
) -> Ordering {
    if options.selection_policy.is_strict() {
        return compare_visited_null_window_evaluations(left, right);
    }
    let l = window_selection_decision(
        left,
        options.selection_policy,
        options.resolved_references(),
    );
    let r = window_selection_decision(
        right,
        options.selection_policy,
        options.resolved_references(),
    );
    l.selection_eligible
        .cmp(&r.selection_eligible)
        .then_with(|| match (l.key, r.key) {
            (Some(l), Some(r)) => l.compare(&r),
            _ => Ordering::Equal,
        })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowSelectionReport {
    pub schema_version: u32,
    pub policy: NullWindowSelectionPolicy,
    pub references: NullWindowFdpReferences,
    pub ordered_eligible_windows: Vec<WindowSelectionDecision>,
    pub excluded_windows: Vec<WindowSelectionDecision>,
    pub winner_vs_alternatives: Vec<serde_json::Value>,
    pub zero_protein_entrapment_windows_observed: usize,
    pub interpretation: String,
}

pub fn rank_null_window_evidence(
    values: &[NullWindowEvaluation],
    options: &NullWindowOptimizerOptions,
) -> Result<WindowSelectionReport, String> {
    let refs = options.resolved_references();
    if [refs.psm, refs.peptide, refs.protein]
        .iter()
        .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
    {
        return Err("invalid empirical FDP reference".into());
    }
    let mut seen = FnvHashSet::default();
    if values
        .iter()
        .any(|v| !seen.insert((v.min_rank, v.max_rank)))
    {
        return Err("duplicate window evidence".into());
    }
    let mut eligible: Vec<_> = values
        .iter()
        .filter(|v| window_selection_decision(v, options.selection_policy, refs).selection_eligible)
        .collect();
    eligible.sort_by(|l, r| compare_for_policy(r, l, options));
    let ordered: Vec<_> = eligible
        .iter()
        .map(|v| window_selection_decision(v, options.selection_policy, refs))
        .collect();
    let zero = ordered
        .iter()
        .filter(|v| {
            v.key
                .as_ref()
                .is_some_and(|k| k.zero_selection_entrapment_proteins)
        })
        .count();
    let comparisons = eligible.first().map(|winner| eligible.iter().skip(1).map(|other| {
        let a = window_selection_decision(winner, options.selection_policy, refs).key;
        let b = window_selection_decision(other, options.selection_policy, refs).key;
        let reason = match (a,b) {
            (Some(a),Some(b)) if !options.selection_policy.is_strict() => {
                if a.zero_selection_entrapment_proteins != b.zero_selection_entrapment_proteins {"zero_selection_entrapment_proteins"}
                else if a.excess[0] != b.excess[0] {"protein_excess"}
                else if a.excess[1] != b.excess[1] {"peptide_excess"}
                else if a.excess[2] != b.excess[2] {"psm_excess"}
                else if a.target_yields != b.target_yields {"target_yields_protein_peptide_psm"}
                else if a.adjusted_fdps != b.adjusted_fdps {"lower_fdps_protein_peptide_psm"}
                else {"narrower_lower_window"}
            }, _ => "legacy_strict_order",
        };
        serde_json::json!({"alternative": [other.min_rank,other.max_rank], "decisive_priority": reason,
            "target_yield_delta_winner_minus_alternative": [winner.target_proteins as i128-other.target_proteins as i128,winner.target_peptides as i128-other.target_peptides as i128,winner.target_psms as i128-other.target_psms as i128]})
    }).collect()).unwrap_or_default();
    Ok(WindowSelectionReport { schema_version: 1, policy: options.selection_policy, references: refs,
        ordered_eligible_windows: ordered, excluded_windows: values.iter().map(|v| window_selection_decision(v,options.selection_policy,refs)).filter(|v| !v.selection_eligible).collect(),
        winner_vs_alternatives: comparisons, zero_protein_entrapment_windows_observed: zero,
        interpretation: "Best among supplied evaluated windows only. No inference about unvisited windows, no audit validation, no fitted artifact or production winner lock created.".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> NullWindowOptimizerOptions {
        serde_json::from_value(serde_json::json!({"selection_policy":"reporting_guided_v1", "fdp_references":{"protein":0.01,"peptide":0.01,"psm":0.01}})).unwrap()
    }
    fn evaluation(
        rank: u32,
        entrapments: usize,
        fdps: [f64; 3],
        yields: [usize; 3],
    ) -> NullWindowEvaluation {
        let evidence = NullWindowEvidence {
            schema_version: 1,
            effective_settings_sha256: "synthetic".into(),
            model: ModelFit::Moments,
            hierarchical_reporting: HierarchicalReportingMode::Strict,
            hierarchical_entrapment_validation: true,
            numerical_fit_valid: true,
            fitted_artifact: serde_json::json!({}),
            observed_invalid_probability_values: 0,
            missing_required_probability_values: 0,
            numerical_evaluation_valid: true,
            rank1_rows: 100,
            rank1_with_psm_q: 100,
            rank1_with_peptide_q: 100,
            rank1_with_protein_q: 100,
            annotation_state: serde_json::json!({"raw_q_diagnostic_fdp":0.9}),
            observations: Default::default(),
            predicates: vec![],
            outcome: "empirically_infeasible".into(),
        };
        NullWindowEvaluation {
            evidence: Some(evidence),
            min_rank: rank,
            max_rank: rank,
            validation_scope: NullWindowValidationScope::Level4,
            target_psms: yields[2],
            entrapment_psms: 10,
            target_peptides: yields[1],
            entrapment_peptides: 2,
            target_proteins: yields[0],
            entrapment_proteins: entrapments,
            psm_fdp: Some(fdps[2]),
            peptide_fdp: Some(fdps[1]),
            protein_fdp: Some(fdps[0]),
            feasible: fdps.iter().all(|x| *x <= 0.01),
            low_count_warning: entrapments < 3,
            selected: false,
            elapsed_milliseconds: 0,
        }
    }
    fn selected(values: &[NullWindowEvaluation]) -> u32 {
        rank_null_window_evidence(values, &options())
            .unwrap()
            .ordered_eligible_windows[0]
            .min_rank
    }
    #[test]
    fn reporting_guided_zero_preference_reports_sacrificed_yield() {
        let a = evaluation(2, 0, [0.0, 0.4, 0.5], [10, 20, 30]);
        let b = evaluation(3, 1, [0.001, 0.001, 0.001], [100, 200, 300]);
        let report = rank_null_window_evidence(&[b, a], &options()).unwrap();
        assert_eq!(report.ordered_eligible_windows[0].min_rank, 2);
        assert_eq!(
            report.winner_vs_alternatives[0]["decisive_priority"],
            "zero_selection_entrapment_proteins"
        );
        assert_eq!(
            report.winner_vs_alternatives[0]["target_yield_delta_winner_minus_alternative"],
            serde_json::json!([-90, -180, -270])
        );
        assert_eq!(
            report.ordered_eligible_windows[0].reference_met,
            [Some(true), Some(false), Some(false)]
        );
    }
    #[test]
    fn reporting_guided_excess_is_lexicographic_and_one_sided() {
        for (a, b) in [
            ([0.02, 0.9, 0.9], [0.03, 0.0, 0.0]),
            ([0.02, 0.02, 0.9], [0.02, 0.03, 0.0]),
            ([0.02, 0.02, 0.02], [0.02, 0.02, 0.03]),
        ] {
            assert_eq!(
                selected(&[
                    evaluation(3, 1, b, [100, 200, 300]),
                    evaluation(2, 1, a, [1, 2, 3])
                ]),
                2
            );
        }
        // Equal excess and yield: lower FDP wins, not the value closest to 1%.
        assert_eq!(
            selected(&[
                evaluation(3, 1, [0.0099; 3], [10; 3]),
                evaluation(2, 1, [0.001; 3], [10; 3])
            ]),
            2
        );
        // Once excess ties, yield precedes residual sub-reference FDP.
        assert_eq!(
            selected(&[
                evaluation(3, 1, [0.0099; 3], [11; 3]),
                evaluation(2, 1, [0.001; 3], [10; 3])
            ]),
            3
        );
    }
    #[test]
    fn reporting_guided_no_zero_window_still_selects_and_ties_are_deterministic() {
        let a = evaluation(2, 1, [0.1; 3], [10; 3]);
        let b = evaluation(3, 2, [0.1; 3], [10; 3]);
        assert_eq!(selected(&[b.clone(), a.clone()]), 2);
        assert_eq!(selected(&[a, b]), 2);
        let a = evaluation(2, 0, [0.0, 0.02, 0.03], [10; 3]);
        let b = evaluation(3, 0, [0.0, 0.02, 0.03], [10; 3]);
        assert_eq!(selected(&[b, a]), 2);
    }
    #[test]
    fn reporting_guided_empty_undefined_and_invalid_are_not_perfect_zero() {
        let mut a = evaluation(2, 0, [0.0; 3], [0; 3]);
        let refs = options().resolved_references();
        let decision = window_selection_decision(&a, options().selection_policy, refs);
        assert!(decision.technical_valid);
        assert!(!decision.selection_eligible);
        a.target_psms = 10;
        a.target_peptides = 10;
        a.target_proteins = 10;
        a.psm_fdp = None;
        assert!(
            !window_selection_decision(&a, options().selection_policy, refs).selection_eligible
        );
        a.psm_fdp = Some(0.0);
        a.evidence.as_mut().unwrap().numerical_fit_valid = false;
        assert!(
            !window_selection_decision(&a, options().selection_policy, refs).selection_eligible
        );
        a.evidence.as_mut().unwrap().numerical_fit_valid = true;
        a.evidence.as_mut().unwrap().numerical_evaluation_valid = false;
        assert!(
            !window_selection_decision(&a, options().selection_policy, refs).selection_eligible
        );
    }
    #[test]
    fn reporting_guided_diagnostics_and_audit_metadata_cannot_rank_or_mutate_level4() {
        let mut a = evaluation(2, 0, [0.0, 0.03, 0.1], [10; 3]);
        let before = serde_json::to_vec(&a).unwrap();
        let report = rank_null_window_evidence(&[a.clone()], &options()).unwrap();
        assert_eq!(serde_json::to_vec(&a).unwrap(), before);
        a.evidence.as_mut().unwrap().annotation_state =
            serde_json::json!({"audit_entrapments":999999,"raw_q_diagnostic_fdp":1.0});
        assert_eq!(
            serde_json::to_value(report).unwrap(),
            serde_json::to_value(rank_null_window_evidence(&[a], &options()).unwrap()).unwrap()
        );
    }
    #[test]
    fn reporting_guided_legacy_strict_still_rejects_above_limit() {
        let a = evaluation(2, 1, [0.1; 3], [10; 3]);
        let mut opts = options();
        opts.selection_policy = NullWindowSelectionPolicy::StrictV1;
        assert!(rank_null_window_evidence(std::slice::from_ref(&a), &opts)
            .unwrap()
            .ordered_eligible_windows
            .is_empty());
        opts.selection_policy = NullWindowSelectionPolicy::ReportingGuidedV1;
        assert_eq!(
            rank_null_window_evidence(std::slice::from_ref(&a), &opts)
                .unwrap()
                .ordered_eligible_windows
                .len(),
            1
        );
        assert!(rank_null_window_evidence(&[a.clone(), a], &opts).is_err());
        let legacy: NullWindowOptimizerOptions = serde_json::from_str("{}").unwrap();
        assert!(legacy.selection_policy.is_strict());
        assert!(serde_json::to_value(legacy)
            .unwrap()
            .get("selection_policy")
            .is_none());
    }

    #[test]
    fn reporting_guided_reporting_thresholds_are_not_empirical_references() {
        let settings = FdrSettings::from(crate::input::FdrOptions {
            precursor_fdr: Some(0.02),
            peptide_fdr: Some(0.03),
            protein_fdr: Some(0.04),
            ..Default::default()
        });
        let opts = options();
        let resolved = settings_for_null_window(
            &settings,
            &opts,
            NullWindowCandidate {
                min_rank: 2,
                max_rank: 3,
            },
        )
        .unwrap();
        assert_eq!(
            [
                resolved.precursor_fdr,
                resolved.peptide_fdr,
                resolved.protein_fdr
            ],
            [0.02, 0.03, 0.04]
        );
        let default = FdrSettings::from(crate::input::FdrOptions::default());
        assert_eq!(null_window_reporting_thresholds(&default, &opts), [0.01; 3]);
        let mut legacy = opts;
        legacy.selection_policy = Default::default();
        let old = settings_for_null_window(
            &settings,
            &legacy,
            NullWindowCandidate {
                min_rank: 2,
                max_rank: 3,
            },
        )
        .unwrap();
        assert_eq!(
            [old.precursor_fdr, old.peptide_fdr, old.protein_fdr],
            [0.01; 3]
        );
    }

    #[test]
    fn reporting_guided_adaptive_navigation_accepts_above_reference_without_fits() {
        let mut opts = options();
        let bounds = NullWindowSearchBounds {
            min_rank_min: 2,
            min_rank_max: 3,
            max_rank_min: 2,
            max_rank_max: 3,
        };
        opts.bounds = Some(bounds);
        let mut a = evaluation(2, 1, [0.1; 3], [10; 3]);
        a.max_rank = 3;
        let prior = vec![
            evaluation(2, 1, [0.2; 3], [20; 3]),
            a,
            evaluation(3, 1, [0.3; 3], [30; 3]),
        ];
        let settings = FdrSettings::from(crate::input::FdrOptions::default());
        let db = IndexedDatabase::default();
        let mut callback = |_: &[NullWindowEvaluation]| -> Result<(), String> {
            panic!("must use saved synthetic windows")
        };
        let mut evaluator = NullWindowEvaluator::new(
            &[],
            &settings,
            &opts,
            &db,
            prior.len(),
            prior,
            &mut callback,
        )
        .unwrap();
        assert!(evaluator.is_navigation_eligible(0));
        run_adaptive_null_window_search(&mut evaluator, bounds, &opts.adaptive).unwrap();
        assert_eq!(evaluator.best_index(), Some(1));
    }
}
