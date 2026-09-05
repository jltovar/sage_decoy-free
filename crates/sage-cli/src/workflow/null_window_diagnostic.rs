//! Single historical trial's native null-window search. This deliberately
//! does not call execute_workflow or run_optimizer, and cannot publish a
//! production winner. Historical checkpoints are read, never resumed/written.
use super::*;

fn validate_native_diagnostic_resources(resources: &[ResourcePreflightReport]) -> Result<()> {
    for required in [
        "entrapment_partition",
        "candidate_pool",
        "raw_external_prediction_cache",
    ] {
        anyhow::ensure!(
            resources.iter().any(|r| r.resource_type == required
                && r.valid
                && r.reused
                && !r.generation_allowed),
            "diagnostic missing verified existing {required}"
        );
    }
    anyhow::ensure!(
        resources.iter().all(|r| !r.generation_allowed
            && if r.resource_type == "stage_external_calibration" {
                r.status == "deferred_until_calibration"
            } else {
                r.valid && r.reused
            }),
        "diagnostic resource preflight did not verify exact reuse"
    );
    Ok(())
}

pub fn diagnose_null_window_trial(
    manifest_path: &Path,
    checkpoint_path: &Path,
    expected_checkpoint_sha256: &str,
    trial_id: &str,
    output: &Path,
    parallel: usize,
) -> Result<serde_json::Value> {
    anyhow::ensure!(
        !output.exists(),
        "diagnostic output must be new; never resume or overwrite a historical run"
    );
    anyhow::ensure!(
        sha256_file(checkpoint_path)? == expected_checkpoint_sha256,
        "historical checkpoint file identity mismatch"
    );
    let checkpoint: crate::parameter_optimizer::OptimizerCheckpoint =
        serde_json::from_slice(&std::fs::read(checkpoint_path)?)?;
    let payload = crate::parameter_optimizer::load_checkpoint(
        checkpoint_path,
        &checkpoint.payload.optimizer_fingerprint,
    )?;
    let trial = payload
        .completed_trials
        .get(trial_id)
        .context("requested historical trial is missing")?;
    anyhow::ensure!(
        trial.request.trial_id == trial_id && !trial.request.target_only_outcomes_allowed,
        "invalid diagnostic trial scope"
    );
    let mut manifest = WorkflowManifest::load_before_resource_access(manifest_path)?;
    manifest.validate()?;
    anyhow::ensure!(
        manifest.require_existing_candidate_pool && manifest.require_existing_annotation_cache,
        "diagnostic requires strict existing pool and raw cache"
    );
    let config = manifest
        .parameter_optimizer
        .as_ref()
        .context("diagnostic requires historical optimizer configuration")?;
    anyhow::ensure!(
        config.optimization_only()
            && config.entrapment_validation.require_existing_partition
            && config.entrapment_validation.mode == EntrapmentValidationMode::SelectionAudit,
        "diagnostic requires frozen optimization-only selection/audit provenance"
    );
    let historical: OptimizerProposalSpaceResolution = serde_json::from_slice(&std::fs::read(
        config
            .proposal_space_artifact
            .as_ref()
            .context("historical proposal artifact missing")?,
    )?)?;
    anyhow::ensure!(
        historical.schema_version == 1
            && Some(&historical.proposal_space_sha256)
                == trial.request.root_proposal_space_sha256.as_ref()
            && config.expected_proposal_space_sha256.as_ref()
                == Some(&historical.proposal_space_sha256)
            && historical.proposal_space_sha256 == proposal_space_identity(&historical)?
            && historical.payload_sha256 == proposal_space_payload_sha256(&historical)?,
        "historical proposal-space provenance mismatch"
    );
    let current = resolve_optimizer_proposal_space_from_manifest(&manifest)?;
    anyhow::ensure!(
        current.workflow_definition_sha256 == historical.workflow_definition_sha256
            && current.search_configuration_sha256 == historical.search_configuration_sha256
            && current.parameter_catalog_sha256 == historical.parameter_catalog_sha256
            && serde_json::to_value(&current.canonical_optimizer)?
                == serde_json::to_value(&historical.canonical_optimizer)?,
        "diagnostic manifest or scientific proposal space differs from the historical root"
    );
    let expert = trial
        .request
        .expert
        .as_ref()
        .context("individual expert required")?;
    anyhow::ensure!(
        *expert != OptimizerExpert::Ensemble,
        "single-expert native window diagnostic cannot execute Ensemble"
    );
    let mut model = manifest
        .models
        .iter()
        .find(|model| optimizer_expert(&model.model) == *expert)
        .cloned()
        .context("historical expert missing from manifest")?;
    let block = config
        .blocks
        .iter()
        .find(|block| block.id == trial.request.block_id)
        .context("historical block missing from manifest")?;
    apply_optimizer_window(
        &mut model,
        block
            .window_search
            .as_ref()
            .context("historical trial has no declared window search")?,
    )?;
    let diagnostics = &trial.evaluation.compact_diagnostics;
    let recorded_policy = diagnostics
        .get("resolved_model_window_policy")
        .context("historical window policy not preserved")?;
    anyhow::ensure!(
        *recorded_policy
            == serde_json::json!({"window": model.window, "candidate_windows": model.candidate_windows, "window_optimizer": model.window_optimizer}),
        "historical window policy differs from manifest"
    );
    let options: FdrOptions = serde_json::from_value(
        diagnostics
            .get("resolved_effective_fdr_options")
            .context("historical complete options missing")?
            .clone(),
    )?;
    let resolved = build_resolved_expert_configuration(&model.model, options.clone())?;
    anyhow::ensure!(
        Some(&resolved.resolved_fdr_settings) == diagnostics.get("resolved_effective_fdr_settings"),
        "current effective settings differ from historical scientific settings"
    );

    // Same strict resource verifier as workflow preflight, but no current-root
    // optimizer/checkpoint rebinding: this is historical diagnostic execution.
    let resources = strict_resource_preflight(&manifest, parallel)?;
    validate_native_diagnostic_resources(&resources)?;
    let partition_path = manifest
        .entrapment
        .partition_artifact
        .as_ref()
        .context("partition path missing")?;
    let partition: EntrapmentPartitionArtifact =
        serde_json::from_slice(&std::fs::read(partition_path)?)?;
    let selection = partition.selection_view();
    let mut proteins = selection.selection_proteins.clone();
    proteins.sort();
    manifest.validation.effective_ratios = EffectiveRatios {
        psm: selection.selection_ratios.peptidoform_ratio,
        peptide: selection.selection_ratios.peptide_ratio,
        protein: selection.selection_ratios.protein_ratio,
    };
    let mut input = Input::load(manifest.search_config.to_string_lossy().as_ref())?;
    // Canonical scientific configurations intentionally omit runtime labels,
    // dataset context and adaptive-policy carriers. Reconstruct those from
    // the same immutable inputs and production functions as run_search_stage;
    // never treat the canonical option carrier as a complete execution input.
    let execution_options = input.fdr.get_or_insert_with(FdrOptions::default);
    execution_options.mode = Some(FdrMode::DecoyFree);
    execution_options.model_fit = Some(model.model.clone());
    execution_options.selection_entrapment_proteins = Some(proteins);
    execution_options.nokoi_application_dataset_fingerprint =
        Some(compute_dataset_identity(&manifest)?.fingerprint);
    apply_fdr_overrides(execution_options, &trial.request.parameters)?;
    apply_window(execution_options, &model.model, &model.window);
    install_null_window_policy(execution_options, &model, &manifest, Some(&selection));
    let execution_resolved =
        build_resolved_expert_configuration(&model.model, execution_options.clone())?;
    anyhow::ensure!(
        execution_resolved.resolved_fdr_settings == resolved.resolved_fdr_settings,
        "reconstructed execution settings differ from historical scientific configuration"
    );
    input.mzml_paths = Some(manifest.spectra.clone());
    input.database.fasta = Some(strict_preflight_fasta(&manifest)?.display().to_string());
    input.output_directory = None;
    input.annotate_matches = Some(false);
    if let Some(external) = input.external_features.as_mut() {
        external.enabled = Some(trial.request.use_external_features);
    } else {
        anyhow::ensure!(
            !trial.request.use_external_features,
            "historical external-feature configuration missing"
        );
    }
    let runner = Runner::new(input.build()?, parallel)?;
    let settings = runner.parameters.fdr.clone();
    let request = CandidatePoolRequest {
        root: manifest.resolved_candidate_pool_root(),
        required_rank_depth: requested_rank_depth(&manifest, &runner),
        require_existing: true,
        allow_reuse: true,
    };
    std::fs::create_dir(output)?;
    write_json_atomic(
        &output.join("diagnostic.input.json"),
        &serde_json::json!({
            "schema_version": 1, "scope": "single_historical_trial_native_window_diagnostic",
            "historical_checkpoint_sha256": expected_checkpoint_sha256,
            "historical_optimizer_fingerprint": payload.optimizer_fingerprint,
            "historical_trial": trial,
            "historical_manifest_sha256": sha256_file(manifest_path)?,
        "current_resolved_configuration": resolved,
        "current_proposal_space_execution_identity": current.proposal_space_sha256,
        "historical_proposal_space_execution_identity": historical.proposal_space_sha256,
            "binary_sha256": sha256_file(&std::env::current_exe()?)?,
            "scientific_settings_equal": true, "resources": resources,
            "raw_cache_verified_not_joined": true, "production_winner_allowed": false,
        }),
    )?;
    runner.diagnose_native_null_windows(&request, &settings, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_resources_are_required_but_post_native_calibration_remains_deferred() {
        let resource = |kind: &str| -> ResourcePreflightReport {
            serde_json::from_value(serde_json::json!({
                "resource_type": kind, "search_space": "+entrapment", "stage": null,
                "status": "validated_exact", "requested_path": "synthetic", "expected_fingerprint": "f", "actual_fingerprint": "f",
                "schema_version": 1, "candidate_or_annotation_count": 2, "retained_rank_depth": 3,
                "manifest_sha256": "m", "payload_sha256": "p", "valid": true, "reused": true,
                "generation_allowed": false, "catalog_fingerprints": [], "original_source_uris": [], "current_source_uris": [],
                "portable_identity_valid": true, "relocation_detected": false, "failure_reason": null
            })).unwrap()
        };
        let mut resources = [
            "entrapment_partition",
            "candidate_pool",
            "raw_external_prediction_cache",
            "stage_external_calibration",
        ]
        .into_iter()
        .map(resource)
        .collect::<Vec<_>>();
        resources[3].status = "deferred_until_calibration".into();
        resources[3].valid = false;
        resources[3].reused = false;
        validate_native_diagnostic_resources(&resources).unwrap();
        resources[2].valid = false;
        assert!(validate_native_diagnostic_resources(&resources).is_err());
        resources[2].valid = true;
        resources[3].generation_allowed = true;
        assert!(validate_native_diagnostic_resources(&resources).is_err());
        assert!(validate_native_diagnostic_resources(&resources[..2]).is_err());
    }

    #[test]
    fn diagnostic_never_overwrites_or_resumes_an_existing_output() {
        let root = std::env::temp_dir().join(format!(
            "sage-native-window-boundary-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let marker = root.join("immutable.checkpoint");
        std::fs::write(&marker, b"preserved").unwrap();
        let error = diagnose_null_window_trial(
            Path::new("missing-workflow"),
            Path::new("missing-checkpoint"),
            "hash",
            "trial",
            &root,
            1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("diagnostic output must be new"));
        assert_eq!(std::fs::read(&marker).unwrap(), b"preserved");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn canonical_configuration_is_not_a_complete_runtime_label_carrier() {
        let runtime = FdrOptions {
            model_fit: Some(ModelFit::Moments),
            selection_entrapment_proteins: Some(vec!["Ent_selection".into()]),
            moments_purification_factor: Some(0.25),
            ..Default::default()
        };
        let canonical =
            build_resolved_expert_configuration(&ModelFit::Moments, runtime.clone()).unwrap();
        assert!(canonical
            .effective_fdr_options
            .selection_entrapment_proteins
            .is_none());
        assert_eq!(
            FdrSettings::from(runtime).selection_entrapment_proteins,
            Some(vec!["Ent_selection".into()])
        );
        // Diagnostic reconstruction must restore the verified runtime context,
        // then compare canonical scientific content, not invent new settings.
        let mut restored = canonical.effective_fdr_options.clone();
        restored.selection_entrapment_proteins = Some(vec!["Ent_selection".into()]);
        assert_eq!(
            build_resolved_expert_configuration(&ModelFit::Moments, restored)
                .unwrap()
                .resolved_fdr_settings,
            canonical.resolved_fdr_settings
        );
    }
}
