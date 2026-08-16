use sha2::{Digest, Sha256};
use std::fs;

fn main() {
    let source = "src/ml/nokoi.rs";
    println!("cargo:rerun-if-changed={source}");
    let payload = fs::read(source).expect("reading Nokoi implementation source");
    // Test-only additions must not invalidate otherwise identical scientific
    // artifacts. Hash the complete production implementation, ending at the
    // explicitly cfg(test)-gated portable test module.
    let marker = b"#[cfg(test)]\nmod portable_tests";
    let production_len = payload
        .windows(marker.len())
        .position(|window| window == marker)
        .expect("locating Nokoi production/test source boundary");
    let digest = Sha256::digest(&payload[..production_len]);
    println!("cargo:rustc-env=SAGE_NOKOI_SOURCE_SHA256={digest:x}");
}
