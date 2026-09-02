use clap::{value_parser, Arg, Command, ValueHint};
use rayon::ThreadPoolBuilder;
use sage_cli::audit::execute_validation_audit;
use sage_cli::entrapment::execute_entrapment_audit;
use sage_cli::input::Input;
use sage_cli::provenance::{freeze_baseline, write_json_atomic};
use sage_cli::runner::Runner;
use sage_cli::workflow::{
    execute_workflow, materialize_workflow_entrapment_partition,
    resolve_frozen_expert_configurations, resolve_optimizer_proposal_space,
};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::default()
        .filter_level(log::LevelFilter::Error)
        .parse_env(env_logger::Env::default().filter_or("SAGE_LOG", "error,sage=info"))
        .init();

    let matches = Command::new("sage")
		.version(clap::crate_version!())
		.author("Sage by Michael Lazear <michaellazear92@gmail.com>\nSage Decoy-free version by JLTovar")
		.about("\u{1F52E} Sage \u{1F9D9} - Proteomics searching so fast—and now decoy-free—it feels like magic!")
		.arg(
			Arg::new("parameters")
				.required(false)
				.value_parser(clap::builder::NonEmptyStringValueParser::new())
				.help("Path to configuration parameters (JSON file)")
				.value_hint(ValueHint::FilePath),
		)
        .arg(
            Arg::new("mzml_paths")
                .num_args(1..)
                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                .help(
                    "Paths to mzML files to process. Overrides mzML files listed in the \
                     configuration file.",
                )
                .value_hint(ValueHint::FilePath),
        )
        .subcommand(
            Command::new("workflow")
                .about("Run or plan a resumable Sage Decoy-Free validation workflow")
                .arg(
                    Arg::new("manifest")
                        .required(true)
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Path to a Sage Decoy-Free workflow manifest"),
                )
                .arg(
                    Arg::new("plan-only")
                        .long("plan-only")
                        .action(clap::ArgAction::SetTrue)
                        .help("Validate and materialize the workflow plan without running searches"),
                ),
        )
        .subcommand(
            Command::new("materialize-entrapment-partition")
                .about(
                    "Materialize or validate a workflow's deterministic entrapment partition without running searches or optimizer trials",
                )
                .arg(
                    Arg::new("manifest")
                        .required(true)
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Path to a Sage Decoy-Free workflow manifest"),
                )
                .arg(
                    Arg::new("inputs-only")
                        .long("inputs-only")
                        .action(clap::ArgAction::SetTrue)
                        .help("Print prospective partition input identities without assigning components or writing an artifact"),
                ),
        )
        .subcommand(
            Command::new("resolve-optimizer-proposal-space")
                .about(
                    "Resolve and freeze an unresolved optimizer proposal space without data access or winner selection",
                )
                .arg(
                    Arg::new("manifest")
                        .required(true)
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Path to a prospective optimization workflow manifest"),
                )
                .arg(
                    Arg::new("output")
                        .required(true)
                        .long("output")
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("New immutable canonical proposal-space artifact"),
                ),
        )
        .subcommand(
            Command::new("resolve-frozen-expert-configurations")
                .about(
                    "Resolve and freeze canonical expert configurations without spectra, caches, fitting, or optimizer trials",
                )
                .arg(
                    Arg::new("manifest")
                        .required(true)
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Path to a single-valued frozen-expert workflow manifest"),
                )
                .arg(
                    Arg::new("output")
                        .required(true)
                        .long("output")
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("New immutable canonical resolution artifact"),
                ),
        )
        .subcommand(
            Command::new("candidate-pool-only")
                .about("Construct and verify one immutable candidate pool, then stop before every statistical or annotation stage")
                .arg(
                    Arg::new("parameters")
                        .required(true)
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Frozen Sage search-parameter JSON"),
                )
                .arg(
                    Arg::new("candidate-pool-root")
                        .required(true)
                        .long("candidate-pool-root")
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::DirPath)
                        .help("Root for content-addressed immutable candidate pools"),
                )
                .arg(
                    Arg::new("rank-depth")
                        .required(true)
                        .long("rank-depth")
                        .value_parser(value_parser!(u32).range(1..))
                        .help("Required retained candidate rank depth"),
                )
                .arg(
                    Arg::new("report")
                        .required(true)
                        .long("report")
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Atomic construction-boundary report JSON"),
                ),
        )
        .subcommand(
            Command::new("raw-cache-only")
                .about("Verify an exact candidate pool, construct and verify one raw MS2PIP/DeepLC cache, then stop")
                .arg(
                    Arg::new("parameters")
                        .required(true)
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Frozen Sage search-parameter JSON with external_features enabled"),
                )
                .arg(
                    Arg::new("candidate-pool-root")
                        .required(true)
                        .long("candidate-pool-root")
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::DirPath)
                        .help("Root containing the exact immutable candidate pool"),
                )
                .arg(
                    Arg::new("annotation-cache-root")
                        .required(true)
                        .long("annotation-cache-root")
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::DirPath)
                        .help("Root for content-addressed raw external predictions"),
                )
                .arg(
                    Arg::new("rank-depth")
                        .required(true)
                        .long("rank-depth")
                        .value_parser(value_parser!(u32).range(1..))
                        .help("Rank depth; must equal frozen external_features.max_rank"),
                )
                .arg(
                    Arg::new("report")
                        .required(true)
                        .long("report")
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Atomic raw-cache construction catalog/report JSON"),
                ),
        )
        .subcommand(
            Command::new("finalize-raw-cache-from-existing-output")
                .about("Verify and atomically finalize a raw cache from a preserved generator output without launching external processes")
                .arg(Arg::new("parameters").required(true).value_hint(ValueHint::FilePath))
                .arg(
                    Arg::new("candidate-pool-root")
                        .required(true)
                        .long("candidate-pool-root")
                        .value_hint(ValueHint::DirPath),
                )
                .arg(
                    Arg::new("annotation-cache-root")
                        .required(true)
                        .long("annotation-cache-root")
                        .value_hint(ValueHint::DirPath),
                )
                .arg(
                    Arg::new("generator-output-tsv")
                        .required(true)
                        .long("generator-output-tsv")
                        .value_hint(ValueHint::FilePath),
                )
                .arg(
                    Arg::new("legacy-generator-source-root")
                        .long("legacy-generator-source-root")
                        .value_hint(ValueHint::DirPath)
                        .help("Exact historical repository source root required to verify legacy v1 generator provenance"),
                )
                .arg(
                    Arg::new("rank-depth")
                        .required(true)
                        .long("rank-depth")
                        .value_parser(value_parser!(u32).range(1..)),
                )
                .arg(
                    Arg::new("report")
                        .required(true)
                        .long("report")
                        .value_hint(ValueHint::FilePath),
                ),
        )
        .subcommand(
            Command::new("audit-entrapment")
                .about("Generate or inspect an entrapment FASTA without running spectral searches")
                .arg(
                    Arg::new("manifest")
                        .required(true)
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Path to an entrapment audit manifest"),
                ),
        )
        .subcommand(
            Command::new("freeze-baseline")
                .about("Freeze hashes and source provenance for corrected validation results")
                .arg(
                    Arg::new("output")
                        .required(true)
                        .long("output")
                        .value_hint(ValueHint::FilePath),
                )
                .arg(
                    Arg::new("paths")
                        .required(true)
                        .num_args(1..)
                        .value_hint(ValueHint::AnyPath),
                )
                .arg(
                    Arg::new("status")
                        .long("status")
                        .default_value("complete")
                        .help("Scientific status recorded in the frozen manifest"),
                ),
        )
        .subcommand(
            Command::new("validate-results")
                .about("Audit completed Sage result tables without rerunning searches")
                .arg(
                    Arg::new("manifest")
                        .required(true)
                        .value_parser(clap::builder::NonEmptyStringValueParser::new())
                        .value_hint(ValueHint::FilePath)
                        .help("Path to a validation audit manifest"),
                ),
        )
        .arg(
            Arg::new("fasta")
                .short('f')
                .long("fasta")
                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                .help(
                    "Path to FASTA database. Overrides the FASTA file \
                     specified in the configuration file.",
                )
                .value_hint(ValueHint::FilePath),
        )
        .arg(
            Arg::new("output_directory")
                .short('o')
                .long("output_directory")
                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                .help(
                    "Path where search and quant results will be written. \
                     Overrides the directory specified in the configuration file.",
                )
                .value_hint(ValueHint::DirPath),
        )
        .arg(
            Arg::new("batch-size")
                .long("batch-size")
                .value_parser(value_parser!(u16).range(1..))
                .help("Number of files to load and search in parallel (default = # of CPUs/2)")
                .value_hint(ValueHint::Other),
        )
        .arg(
            Arg::new("parquet")
                .long("parquet")
                .action(clap::ArgAction::SetTrue)
                .help("Write search output in parquet format instead of tsv"),
        )
        .arg(
            Arg::new("annotate-matches")
                .long("annotate-matches")
                .action(clap::ArgAction::SetTrue)
                .help("Write matched fragments output file."),
        )
        .arg(
            Arg::new("write-pin")
                .long("write-pin")
                .action(clap::ArgAction::SetTrue)
                .help("Write percolator-compatible `.pin` output files"),
        )
        .arg(
            Arg::new("disable-telemetry")
                .long("disable-telemetry-i-dont-want-to-improve-sage")
                .action(clap::ArgAction::SetFalse)
                .help("Disable sending telemetry data"),
        )
        .arg(
            Arg::new("stack-size")
                .long("stack-size")
                .value_parser(value_parser!(u32).range(1..))
                .help("Set Rayon worker thread stack size in MiB (default: 64 MiB)")
                .value_hint(ValueHint::Other),
        )
        .help_template(
            "{usage-heading} {usage}\n\n\
             {about-with-newline}\n\
             Written by {author-with-newline}Version {version}\n\n\
             {all-args}{after-help}",
        )
        .get_matches();

    // Decoy-free Lower Order and ensemble workflows can require substantially
    // more stack than vanilla Sage. Preserve the fork's proven-safe default,
    // while exposing upstream's override for constrained environments.
    let stack_size_mib = matches.get_one::<u32>("stack-size").copied().unwrap_or(64);
    let stack_size_bytes = stack_size_mib as usize * 1024 * 1024;
    log::trace!(
        "setting Rayon worker thread stack size to {} MiB",
        stack_size_mib
    );
    ThreadPoolBuilder::new()
        .stack_size(stack_size_bytes)
        .build_global()
        .expect("configure Rayon pool");

    let parallel = matches
        .get_one::<u16>("batch-size")
        .copied()
        .unwrap_or_else(|| (num_cpus::get() as u16 / 2).max(1)) as usize;

    if let Some(("audit-entrapment", audit_matches)) = matches.subcommand() {
        let manifest = audit_matches
            .get_one::<String>("manifest")
            .expect("required entrapment audit manifest");
        let report = execute_entrapment_audit(std::path::Path::new(manifest))?;
        log::info!(
            "entrapment audit complete: protein_ratio={} peptide_ratio={} peptidoform_ratio={}",
            report.database.measured().protein_ratio,
            report.database.measured().peptide_ratio,
            report.database.measured().peptidoform_ratio
        );
        return Ok(());
    }
    if let Some(("candidate-pool-only", boundary_matches)) = matches.subcommand() {
        let parameters = boundary_matches
            .get_one::<String>("parameters")
            .expect("required search parameters");
        let root = boundary_matches
            .get_one::<String>("candidate-pool-root")
            .expect("required candidate-pool root");
        let rank_depth = boundary_matches
            .get_one::<u32>("rank-depth")
            .copied()
            .expect("required rank depth") as usize;
        let report_path = boundary_matches
            .get_one::<String>("report")
            .expect("required report path");
        let input = Input::load(parameters)?;
        let runner = input
            .build()
            .and_then(|parameters| Runner::new(parameters, parallel))?;
        let report = runner.construct_candidate_pool_only(
            parallel,
            std::path::PathBuf::from(root),
            rank_depth,
        )?;
        write_json_atomic(std::path::Path::new(report_path), &report)?;
        let reopened: sage_cli::runner::CandidatePoolConstructionReport =
            serde_json::from_slice(&std::fs::read(report_path)?)?;
        anyhow::ensure!(
            reopened.status == "verified_complete"
                && reopened.search_fingerprint == report.search_fingerprint
                && reopened.payload_sha256 == report.payload_sha256,
            "candidate-pool-only report failed atomic reopen verification"
        );
        println!("{}", serde_json::to_string_pretty(&reopened)?);
        return Ok(());
    }
    if let Some(("raw-cache-only", boundary_matches)) = matches.subcommand() {
        let parameters = boundary_matches
            .get_one::<String>("parameters")
            .expect("required search parameters");
        let candidate_root = boundary_matches
            .get_one::<String>("candidate-pool-root")
            .expect("required candidate-pool root");
        let annotation_root = boundary_matches
            .get_one::<String>("annotation-cache-root")
            .expect("required annotation-cache root");
        let rank_depth = boundary_matches
            .get_one::<u32>("rank-depth")
            .copied()
            .expect("required rank depth") as usize;
        let report_path = boundary_matches
            .get_one::<String>("report")
            .expect("required report path");
        let input = Input::load(parameters)?;
        let parameters = input.build()?;
        anyhow::ensure!(
            !parameters.database.prefilter,
            "raw-cache-only prohibits database.prefilter before Runner construction because it can launch a native spectrum search"
        );
        let runner = Runner::new(parameters, parallel)?;
        let report = runner.construct_raw_annotation_cache_only(
            std::path::PathBuf::from(candidate_root),
            std::path::PathBuf::from(annotation_root),
            rank_depth,
        )?;
        write_json_atomic(std::path::Path::new(report_path), &report)?;
        let reopened: sage_cli::runner::RawCacheConstructionReport =
            serde_json::from_slice(&std::fs::read(report_path)?)?;
        anyhow::ensure!(
            reopened.status == "verified_complete"
                && reopened.raw_cache.identity == report.raw_cache.identity
                && reopened.raw_cache.payload_sha256 == report.raw_cache.payload_sha256,
            "raw-cache-only report failed atomic reopen verification"
        );
        println!("{}", serde_json::to_string_pretty(&reopened)?);
        return Ok(());
    }
    if let Some(("finalize-raw-cache-from-existing-output", boundary_matches)) =
        matches.subcommand()
    {
        let parameters = boundary_matches
            .get_one::<String>("parameters")
            .expect("required search parameters");
        let candidate_root = boundary_matches
            .get_one::<String>("candidate-pool-root")
            .expect("required candidate-pool root");
        let annotation_root = boundary_matches
            .get_one::<String>("annotation-cache-root")
            .expect("required annotation-cache root");
        let generator_output = boundary_matches
            .get_one::<String>("generator-output-tsv")
            .expect("required generator output");
        let legacy_generator_source_root = boundary_matches
            .get_one::<String>("legacy-generator-source-root")
            .map(std::path::PathBuf::from);
        let rank_depth = boundary_matches
            .get_one::<u32>("rank-depth")
            .copied()
            .expect("required rank depth") as usize;
        let report_path = boundary_matches
            .get_one::<String>("report")
            .expect("required report path");
        let input = Input::load(parameters)?;
        let parameters = input.build()?;
        anyhow::ensure!(
            !parameters.database.prefilter,
            "raw-cache recovery prohibits database.prefilter before Runner construction"
        );
        let runner = Runner::new(parameters, parallel)?;
        let report = runner.finalize_raw_annotation_cache_from_existing_output(
            std::path::PathBuf::from(candidate_root),
            std::path::PathBuf::from(annotation_root),
            rank_depth,
            std::path::PathBuf::from(generator_output),
            legacy_generator_source_root,
        )?;
        write_json_atomic(std::path::Path::new(report_path), &report)?;
        let reopened: sage_cli::runner::RawCacheConstructionReport =
            serde_json::from_slice(&std::fs::read(report_path)?)?;
        anyhow::ensure!(
            reopened.status == "verified_complete"
                && reopened.raw_cache.manifest.content_fingerprint
                    == report.raw_cache.manifest.content_fingerprint
                && reopened.raw_cache.payload_sha256 == report.raw_cache.payload_sha256,
            "raw-cache recovery report failed atomic reopen verification"
        );
        println!("{}", serde_json::to_string_pretty(&reopened)?);
        return Ok(());
    }
    if let Some(("materialize-entrapment-partition", partition_matches)) = matches.subcommand() {
        let manifest = partition_matches
            .get_one::<String>("manifest")
            .expect("required workflow manifest");
        if partition_matches
            .get_one::<bool>("inputs-only")
            .copied()
            .unwrap_or(false)
        {
            let inputs = sage_cli::workflow::inspect_workflow_entrapment_partition_inputs(
                std::path::Path::new(manifest),
            )?;
            println!("{}", serde_json::to_string_pretty(&inputs)?);
        } else {
            let artifact =
                materialize_workflow_entrapment_partition(std::path::Path::new(manifest))?;
            println!("{}", serde_json::to_string_pretty(&artifact)?);
        }
        return Ok(());
    }
    if let Some(("resolve-frozen-expert-configurations", resolver_matches)) = matches.subcommand() {
        let manifest = resolver_matches
            .get_one::<String>("manifest")
            .expect("required workflow manifest");
        let output = resolver_matches
            .get_one::<String>("output")
            .expect("required resolution output");
        let artifact = resolve_frozen_expert_configurations(
            std::path::Path::new(manifest),
            std::path::Path::new(output),
        )?;
        println!("{}", serde_json::to_string_pretty(&artifact)?);
        return Ok(());
    }
    if let Some(("resolve-optimizer-proposal-space", resolver_matches)) = matches.subcommand() {
        let manifest = resolver_matches
            .get_one::<String>("manifest")
            .expect("required workflow manifest");
        let output = resolver_matches
            .get_one::<String>("output")
            .expect("required resolution output");
        let artifact = resolve_optimizer_proposal_space(
            std::path::Path::new(manifest),
            std::path::Path::new(output),
        )?;
        println!("{}", serde_json::to_string_pretty(&artifact)?);
        return Ok(());
    }

    if let Some(("workflow", workflow_matches)) = matches.subcommand() {
        let manifest = workflow_matches
            .get_one::<String>("manifest")
            .expect("required workflow manifest");
        let plan_only = workflow_matches
            .get_one::<bool>("plan-only")
            .copied()
            .unwrap_or(false);
        let source_repo = std::env::current_dir()?;
        let state = execute_workflow(
            std::path::Path::new(manifest),
            &source_repo,
            parallel,
            plan_only,
        )?;
        if plan_only {
            println!("{}", serde_json::to_string_pretty(&state)?);
        }
        log::info!(
            "workflow complete: stages={} validation_summaries={} pending_gates={}",
            state.stages.len(),
            state.validation.len(),
            state.pending_validation_gates.len()
        );
        return Ok(());
    }
    if let Some(("freeze-baseline", baseline_matches)) = matches.subcommand() {
        let output = baseline_matches
            .get_one::<String>("output")
            .expect("required baseline output");
        let paths = baseline_matches
            .get_many::<String>("paths")
            .expect("required baseline paths")
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>();
        let status = baseline_matches
            .get_one::<String>("status")
            .expect("defaulted baseline status");
        let source_repo = std::env::current_dir()?;
        let frozen = freeze_baseline(&paths, &source_repo, status)?;
        write_json_atomic(std::path::Path::new(output), &frozen)?;
        log::info!("frozen baseline files={}", frozen.files.len());
        return Ok(());
    }
    if let Some(("validate-results", audit_matches)) = matches.subcommand() {
        let manifest = audit_matches
            .get_one::<String>("manifest")
            .expect("required validation audit manifest");
        let report = execute_validation_audit(std::path::Path::new(manifest))?;
        log::info!(
            "validation audit complete: summaries={} missing={} tdc_comparisons={}",
            report.summaries.len(),
            report.missing_runs.len(),
            report.tdc_benchmarks.len()
        );
        return Ok(());
    }

    anyhow::ensure!(
        matches.get_one::<String>("parameters").is_some(),
        "a search parameter JSON or the `workflow` subcommand is required"
    );

    let parquet = matches.get_one::<bool>("parquet").copied().unwrap_or(false);
    let send_telemetry = matches
        .get_one::<bool>("disable-telemetry")
        .copied()
        .unwrap_or(true);

    let input = Input::from_arguments(matches)?;

    let runner = input
        .build()
        .and_then(|parameters| Runner::new(parameters, parallel))?;

    let tel = runner.run(parallel, parquet)?;

    if send_telemetry {
        tel.send();
    }

    Ok(())
}
