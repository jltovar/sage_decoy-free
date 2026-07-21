//! Ensure that we exhaustively visit all fragment ions matching tolerances

use quickcheck_macros::quickcheck;
use sage_core::database::{Builder, IndexedDatabase, PeptideIx};
use sage_core::fasta::Fasta;
use sage_core::mass::Tolerance;

const FASTA: &'static str = r#"
>sp|Q99536|VAT1_HUMAN Synaptic vesicle membrane protein VAT-1 homolog OS=Homo sapiens OX=9606 GN=VAT1 PE=1 SV=2
MSDEREVAEAATGEDASSPPPKTEAASDPQHPAASEGAAAAAASPPLLRCLVLTGFGGYD
KVKLQSRPAAPPAPGPGQLTLRLRACGLNFADLMARQGLYDRLPPLPVTPGMEGAGVVIA
VGEGVSDRKAGDRVMVLNRSGMWQEEVTVPSVQTFLIPEAMTFEEAAALLVNYITAYMVL
FDFGNLQPGHSVLVHMAAGGVGMAAVQLCRTVENVTVFGTASASKHEALKENGVTHPIDY
HTTDYVDEIKKISPKGVDIVMDPLGGSDTAKGYNLLKPMGKVVTYGMANLLTGPKRNLMA
LARTWWNQFSVTALQLLQANRAVCGFHLGYLDGEVELVSGVVARLLALYNQGHIKPHIDS
VWPFEKVADAMKQMQEKKNVGKVLLVPGPEKEN
"#;

fn mk_database(bucket_size: usize) -> IndexedDatabase {
    let builder = Builder {
        bucket_size: Some(bucket_size),
        fasta: Some("static".into()),
        ..Default::default()
    };
    let fasta = Fasta::parse(FASTA.into(), "rev_", false);

    builder.make_parameters().build(fasta)
}

#[quickcheck]
fn check_all_ions_visited(target_fragment_mz: f32, bucket_size: usize) {
    let database = mk_database(bucket_size.clamp(1, 8192));

    // Map PeptideIx -> number of fragments between 500 & 700 m/z
    // We want to make sure that IndexedDatabase::query hits *all* of them
    let mut expected = vec![0usize; database.peptides.len()];

    let fragment_tol = Tolerance::Da(-100.0, 100.0);
    let (frag_lo, frag_hi) = fragment_tol.bounds(target_fragment_mz);

    for (chunk_idx, chunk) in database.fragments.chunks(database.bucket_size).enumerate() {
        // Check for total ordering by PeptideIx within a chunk
        let mut last = PeptideIx(0);
        for frag in chunk {
            assert!(frag.peptide_index >= last);
            assert!(frag.fragment_mz >= database.buckets()[chunk_idx]);
            if chunk_idx + 1 < database.buckets().len() {
                assert!(frag.fragment_mz <= database.buckets()[chunk_idx + 1]);
            }

            if frag.fragment_mz >= frag_lo && frag.fragment_mz <= frag_hi {
                expected[frag.peptide_index.0 as usize] += 1;
            }
            last = frag.peptide_index;
        }
    }

    let mut visited = vec![0usize; database.peptides.len()];

    // Hit all peptides in database, track how many of the 500-700 fragment m/z's
    // are returned to us by searching the database.
    let query = database.query(1000.0, Tolerance::Da(-5000.0, 5000.0), fragment_tol);

    for fragment in query.page_search(target_fragment_mz) {
        visited[fragment.peptide_index.0 as usize] += 1;
    }

    // Make sure we visited every possible fragment
    assert_eq!(expected, visited);
}

#[test]
fn ppm_fragment_window_is_charge_invariant() {
    let database = mk_database(128);
    let target = database.fragments[database.fragments.len() / 2];
    let query = database.query(
        database.peptides[target.peptide_index.0 as usize].monoisotopic,
        Tolerance::Da(-0.01, 0.01),
        Tolerance::Ppm(-10.0, 10.0),
    );

    // An 8-ppm observed error remains 8 ppm after converting m/z to neutral
    // fragment mass, regardless of charge. Dividing the configured PPM window
    // by charge would incorrectly reject the z=2 and z=3 observations.
    for charge in 1..=3 {
        let observed_mz = target.fragment_mz * (1.0 + 8.0e-6) / charge as f32;
        let neutral_mass = observed_mz * charge as f32;
        assert!(query.page_search(neutral_mass).any(|fragment| {
            fragment.peptide_index == target.peptide_index
                && fragment.fragment_mz == target.fragment_mz
        }));
    }
}
