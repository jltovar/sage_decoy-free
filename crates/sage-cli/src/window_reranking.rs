//! Evidence-only policy analysis. This module deliberately has no Runner,
//! dataset reader, process launcher, fitting or winner-materialization path.
use anyhow::{Context, Result};
use sage_core::decoy_free_fdr::{
    artifact_contains_model, rank_null_window_evidence, DfRunArtifacts, FdpPredicateEvidence,
    NullWindowEvaluation,
};
use sage_core::input::{
    NullWindowFdpReferences, NullWindowOptimizerOptions, NullWindowSelectionPolicy,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RerankingPolicy {
    pub schema_version: u32,
    pub selection_policy: NullWindowSelectionPolicy,
    pub fdp_references: NullWindowFdpReferences,
}

fn verified_json(path: &Path, expected: &str) -> Result<serde_json::Value> {
    anyhow::ensure!(
        crate::provenance::sha256_file(path)? == expected,
        "evidence content hash mismatch: {}",
        path.display()
    );
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn verify_evaluation(v: &NullWindowEvaluation) -> Result<()> {
    let e = v
        .evidence
        .as_ref()
        .context("missing numerical/window evidence")?;
    anyhow::ensure!(
        e.schema_version == 1 && e.effective_settings_sha256.len() == 64,
        "unsupported or unbound window evidence"
    );
    if e.numerical_fit_valid {
        let artifacts: DfRunArtifacts = serde_json::from_value(
            e.fitted_artifact
                .get("state")
                .context("missing fitted state")?
                .clone(),
        )?;
        anyhow::ensure!(
            artifact_contains_model(&artifacts, &e.model),
            "recorded fit validity disagrees with fitted state"
        );
    }
    anyhow::ensure!(
        !e.numerical_evaluation_valid
            || (e.numerical_fit_valid
                && e.observed_invalid_probability_values == 0
                && e.missing_required_probability_values == 0
                && e.observations.fallback_events.is_empty()),
        "inconsistent numerical validity"
    );
    anyhow::ensure!(
        e.predicates.len() == 3,
        "expected all three empirical predicates"
    );
    for (level, t, n, value) in [
        ("psm", v.target_psms, v.entrapment_psms, v.psm_fdp),
        (
            "peptide",
            v.target_peptides,
            v.entrapment_peptides,
            v.peptide_fdp,
        ),
        (
            "protein",
            v.target_proteins,
            v.entrapment_proteins,
            v.protein_fdp,
        ),
    ] {
        let matches: Vec<_> = e.predicates.iter().filter(|p| p.level == level).collect();
        anyhow::ensure!(matches.len() == 1, "duplicate or missing empirical level");
        let p = matches[0];
        let expected = FdpPredicateEvidence::new(level, t, n, p.measured_ratio, p.limit, 0);
        // Comparisons allow JSON floating point round-trip noise only.
        let equal = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(a), Some(b)) => {
                a.is_finite()
                    && b.is_finite()
                    && (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
            }
            (None, None) => true,
            _ => false,
        };
        anyhow::ensure!(
            p.targets == t
                && p.selection_entrapments == n
                && p.denominator == expected.denominator
                && equal(p.numerator, expected.numerator)
                && equal(p.value, expected.value)
                && equal(value, expected.value)
                && p.predicate == expected.predicate,
            "inconsistent {level} counts/ratio/FDP evidence"
        );
    }
    Ok(())
}

pub fn rerank(
    checkpoint: &Path,
    checkpoint_sha256: &str,
    policy_path: &Path,
    policy_sha256: &str,
    output: &Path,
) -> Result<serde_json::Value> {
    anyhow::ensure!(
        !output.exists(),
        "reranking output already exists; never overwrite evidence"
    );
    let original = verified_json(checkpoint, checkpoint_sha256)?;
    anyhow::ensure!(
        original["schema"] == "sage-null-window-checkpoint-v2",
        "unsupported checkpoint schema"
    );
    let policy: RerankingPolicy =
        serde_json::from_value(verified_json(policy_path, policy_sha256)?)?;
    anyhow::ensure!(
        policy.schema_version == 1,
        "unsupported reranking policy schema"
    );
    let values: Vec<NullWindowEvaluation> = serde_json::from_value(
        original
            .get("evaluations")
            .context("missing evaluations")?
            .clone(),
    )?;
    anyhow::ensure!(!values.is_empty(), "no saved window evaluations");
    for v in &values {
        verify_evaluation(v)?;
    }
    let options: NullWindowOptimizerOptions = serde_json::from_value(
        serde_json::json!({"selection_policy":policy.selection_policy,"fdp_references":policy.fdp_references}),
    )?;
    let ranking = rank_null_window_evidence(&values, &options).map_err(anyhow::Error::msg)?;
    let ranked_measurements: Vec<_> = ranking
        .ordered_eligible_windows
        .iter()
        .map(|d| {
            values
                .iter()
                .find(|v| v.min_rank == d.min_rank && v.max_rank == d.max_rank)
                .expect("ranked measurement")
        })
        .collect();
    let report = serde_json::json!({"schema_version":1,"classification":"saved_evidence_reranking_only", "checkpoint_sha256":checkpoint_sha256,"policy_sha256":policy_sha256,
        "historical_checkpoint_fingerprint":original["fingerprint"],"historical_search_fingerprint":original["search_fingerprint"],
        "current_implementation_sha256":crate::parameter_optimizer::PARAMETER_OPTIMIZER_IMPLEMENTATION_SOURCE_SHA256,
        "current_binary_sha256":crate::provenance::sha256_file(&std::env::current_exe()?)?,
        "evaluated_windows":values.len(),"ranking":ranking,"ranked_measurements":ranked_measurements,
        "no_new_evaluation":true,"production_winner_lock_created":false,"original_checkpoint_modified":false});
    crate::provenance::write_json_atomic(output, &report)?;
    std::fs::File::open(output)?.sync_all()?;
    let reopened: serde_json::Value = serde_json::from_slice(&std::fs::read(output)?)?;
    anyhow::ensure!(
        reopened == report,
        "reranking publication verification failed"
    );
    anyhow::ensure!(
        crate::provenance::sha256_file(checkpoint)? == checkpoint_sha256
            && crate::provenance::sha256_file(policy_path)? == policy_sha256,
        "input changed during reranking"
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn evidence_only_reranking_verifies_inputs_and_never_overwrites() {
        let root = std::env::temp_dir().join(format!(
            "sage-rerank-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let artifacts = DfRunArtifacts {
            moments: Some(sage_core::input::FrozenGumbelParameters {
                schema_version: 1,
                model_version: "sage-moments-gumbel-v1".into(),
                min_rank: 2,
                max_rank: 2,
                mu: 3.0,
                beta: 1.0,
            }),
            ..Default::default()
        };
        let predicates: Vec<_> = ["psm", "peptide", "protein"]
            .into_iter()
            .map(|level| FdpPredicateEvidence::new(level, 100, 1, 1.0, 0.01, 3))
            .collect();
        let row = serde_json::json!({"min_rank":2,"max_rank":2,"validation_scope":"level4","target_psms":100,"entrapment_psms":1,"target_peptides":100,"entrapment_peptides":1,"target_proteins":100,"entrapment_proteins":1,"psm_fdp":2.0/101.0,"peptide_fdp":2.0/101.0,"protein_fdp":2.0/101.0,"feasible":false,"selected":false,"low_count_warning":true,
            "evidence":{"schema_version":1,"effective_settings_sha256":"a".repeat(64),"model":"moments","hierarchical_reporting":"strict","hierarchical_entrapment_validation":true,"numerical_fit_valid":true,"fitted_artifact":{"state":artifacts},"observed_invalid_probability_values":0,"missing_required_probability_values":0,"numerical_evaluation_valid":true,"rank1_rows":101,"rank1_with_psm_q":101,"rank1_with_peptide_q":101,"rank1_with_protein_q":101,"annotation_state":{},"observations":{"fallback_events":[],"fit_events":[],"q_methods":[]},"predicates":predicates,"outcome":"empirically_infeasible"}});
        let cp = root.join("checkpoint.json");
        let pol = root.join("policy.json");
        let out = root.join("reranking.json");
        crate::provenance::write_json_atomic(&cp,&serde_json::json!({"schema":"sage-null-window-checkpoint-v2","fingerprint":"historical","search_fingerprint":"search","evaluations":[row.clone()]})).unwrap();
        crate::provenance::write_json_atomic(&pol,&serde_json::json!({"schema_version":1,"selection_policy":"reporting_guided_v1","fdp_references":{"psm":0.01,"peptide":0.01,"protein":0.01}})).unwrap();
        let hash = crate::provenance::sha256_file(&cp).unwrap();
        let ph = crate::provenance::sha256_file(&pol).unwrap();
        assert!(rerank(&cp, "wrong", &pol, &ph, &out).is_err());
        assert!(!out.exists());
        let report = rerank(&cp, &hash, &pol, &ph, &out).unwrap();
        assert_eq!(report["evaluated_windows"], 1);
        assert_eq!(
            report["ranking"]["ordered_eligible_windows"][0]["min_rank"],
            2
        );
        assert_eq!(crate::provenance::sha256_file(&cp).unwrap(), hash);
        assert!(rerank(&cp, &hash, &pol, &ph, &out).is_err());
        let mut corrupt: NullWindowEvaluation = serde_json::from_value(row).unwrap();
        corrupt.entrapment_proteins = 0;
        assert!(verify_evaluation(&corrupt).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
