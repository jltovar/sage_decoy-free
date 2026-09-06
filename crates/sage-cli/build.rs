use sha2::{Digest, Sha256};
use std::fs;

fn main() {
    // Raw production/parsing/serialization dependencies. Diagnostics and the
    // compatibility reader are separately identified below, not generators.
    // Whole files (including interleaved helpers) are deliberately retained.
    let sources = [
        "src/external_feature_cache.rs",
        "src/external_features.rs",
        "src/candidate_pool.rs",
        "src/input.rs",
        "src/input_path_identity.rs",
        "src/provenance.rs",
        "../sage/src/scoring.rs",
        "../sage/src/peptide.rs",
        "../sage/src/database.rs",
        "../sage/src/mass.rs",
        "../sage/src/modification.rs",
        "../sage/src/fasta.rs",
        "../sage/src/enzyme.rs",
        "../sage/src/spectrum.rs",
        "../sage-cloudpath/src/lib.rs",
        "../../Cargo.lock",
        "Cargo.toml",
        "../sage/Cargo.toml",
        "../sage-cloudpath/Cargo.toml",
    ];
    let mut hasher = Sha256::new();
    hasher.update(b"sage-raw-cache-production-source-v2\0");
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
    let mut reader = Sha256::new();
    reader.update(b"sage-raw-cache-reader-source-v1\0");
    for source in ["src/raw_cache_compatibility.rs", "build.rs"] {
        println!("cargo:rerun-if-changed={source}");
        let bytes = fs::read(source).unwrap_or_else(|error| panic!("reading {source}: {error}"));
        reader.update((bytes.len() as u64).to_le_bytes());
        reader.update(bytes);
    }
    println!(
        "cargo:rustc-env=SAGE_RAW_CACHE_READER_SHA256={:x}",
        reader.finalize()
    );
    let mut analysis = Sha256::new();
    analysis.update(b"sage-external-analysis-source-v1\0");
    for source in [
        "src/external_feature_diagnostics.rs",
        "../sage/src/ml/external_auc.rs",
        "../sage/src/decoy_free_fdr.rs",
    ] {
        println!("cargo:rerun-if-changed={source}");
        let bytes = fs::read(source).unwrap_or_else(|error| panic!("reading {source}: {error}"));
        analysis.update((bytes.len() as u64).to_le_bytes());
        analysis.update(bytes);
    }
    println!(
        "cargo:rustc-env=SAGE_EXTERNAL_ANALYSIS_SOURCE_SHA256={:x}",
        analysis.finalize()
    );

    let optimizer_sources = [
        "src/parameter_optimizer.rs",
        "src/workflow.rs",
        "src/workflow/null_window_diagnostic.rs",
        "src/runner.rs",
        "src/external_feature_diagnostics.rs",
        "src/raw_cache_compatibility.rs",
        "src/entrapment.rs",
        "src/validation.rs",
        "../sage/src/input.rs",
        "../sage/src/decoy_free_fdr.rs",
        "../sage/src/decoy_free_fdr/window_evidence.rs",
        "../sage/src/decoy_free_fdr/window_selection.rs",
        "../sage/src/ml/external_auc.rs",
        "src/window_reranking.rs",
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
