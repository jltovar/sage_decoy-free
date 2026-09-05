use sha2::{Digest, Sha256};
use std::fs;

fn main() {
    let sources = ["src/external_feature_cache.rs", "src/external_features.rs"];
    let mut hasher = Sha256::new();
    hasher.update(b"sage-layered-external-cache-source-v1\0");
    for source in sources {
        println!("cargo:rerun-if-changed={source}");
        hasher.update(source.as_bytes());
        hasher.update(b"\0");
        // Hash the complete source file. Test modules can be interleaved with
        // production helpers, so truncating at the first `#[cfg(test)]` would
        // leave later production behavior outside the durable cache identity.
        hasher.update(fs::read(source).unwrap_or_else(|error| panic!("reading {source}: {error}")));
        hasher.update(b"\0");
    }
    println!(
        "cargo:rustc-env=SAGE_EXTERNAL_CACHE_SOURCE_SHA256={:x}",
        hasher.finalize()
    );

    let optimizer_sources = [
        "src/parameter_optimizer.rs",
        "src/workflow.rs",
        "src/workflow/null_window_diagnostic.rs",
        "src/runner.rs",
        "src/entrapment.rs",
        "src/validation.rs",
        "../sage/src/input.rs",
        "../sage/src/decoy_free_fdr.rs",
        "../sage/src/decoy_free_fdr/window_evidence.rs",
    ];
    let mut optimizer_hasher = Sha256::new();
    optimizer_hasher.update(b"sage-parameter-optimizer-source-v1\0");
    for source in optimizer_sources {
        println!("cargo:rerun-if-changed={source}");
        let content = fs::read(source).unwrap_or_else(|error| panic!("reading {source}: {error}"));
        optimizer_hasher.update((content.len() as u64).to_le_bytes());
        optimizer_hasher.update(content);
    }
    println!(
        "cargo:rustc-env=SAGE_PARAMETER_OPTIMIZER_SOURCE_SHA256={:x}",
        optimizer_hasher.finalize()
    );
}
