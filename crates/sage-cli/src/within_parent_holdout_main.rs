use clap::{Arg, Command};
use sage_cli::within_parent_holdout::{
    audit_annotation_caches, audit_training_usefulness, create_preregistration, execute_holdout,
    preflight_holdout,
};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::default()
        .filter_level(log::LevelFilter::Error)
        .parse_env(env_logger::Env::default().filter_or("SAGE_LOG", "error,sage=info"))
        .init();

    let matches = Command::new("sage-within-parent-holdout")
        .about("Development-only verified within-parent run-level holdout runner")
        .subcommand_required(true)
        .subcommand(
            Command::new("preregister")
                .arg(Arg::new("draft").required(true))
                .arg(Arg::new("output").required(true)),
        )
        .subcommand(
            Command::new("preflight")
                .arg(Arg::new("manifest").required(true))
                .arg(Arg::new("output").required(true)),
        )
        .subcommand(
            Command::new("run")
                .arg(Arg::new("manifest").required(true))
                .arg(Arg::new("output").required(true)),
        )
        .subcommand(
            Command::new("audit-annotations")
                .arg(Arg::new("request").required(true))
                .arg(Arg::new("output").required(true)),
        )
        .subcommand(
            Command::new("audit-training-usefulness")
                .arg(Arg::new("manifest").required(true))
                .arg(Arg::new("failed-run-root").required(true))
                .arg(Arg::new("fold").required(true))
                .arg(Arg::new("output").required(true)),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("preregister", args)) => create_preregistration(
            args.get_one::<String>("draft").expect("required"),
            args.get_one::<String>("output").expect("required"),
        ),
        Some(("preflight", args)) => preflight_holdout(
            args.get_one::<String>("manifest").expect("required"),
            args.get_one::<String>("output").expect("required"),
        ),
        Some(("run", args)) => execute_holdout(
            args.get_one::<String>("manifest").expect("required"),
            args.get_one::<String>("output").expect("required"),
        ),
        Some(("audit-annotations", args)) => audit_annotation_caches(
            args.get_one::<String>("request").expect("required"),
            args.get_one::<String>("output").expect("required"),
        ),
        Some(("audit-training-usefulness", args)) => audit_training_usefulness(
            args.get_one::<String>("manifest").expect("required"),
            args.get_one::<String>("failed-run-root").expect("required"),
            args.get_one::<String>("fold").expect("required").parse()?,
            args.get_one::<String>("output").expect("required"),
        ),
        _ => unreachable!(),
    }
}
