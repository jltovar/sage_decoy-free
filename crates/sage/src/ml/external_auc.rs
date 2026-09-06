//! Exact empirical external-feature AUC, without enumerating good/null pairs.
//!
//! Only finite observations participate. Numerical equality (including signed
//! zero) earns half credit; sorting's total order never defines equality.
//! Integer pair counts are checked u128 values. Up to 2^52 valid pairs, every
//! legacy half-integer accumulation is exactly representable in f64, so the
//! final division is bit-identical to the pairwise implementation. Above that
//! range we still count exactly, then round counts to f64 once; unlike repeated
//! floating addition this does not lose small increments or saturate at 2^53.
//! No epsilon, sampling, caps, or parallel floating-point reductions are used.

use rayon::prelude::*;

/// O(G log G + N log N) time and O(G + N) temporary numeric storage.
/// Uses the caller's Rayon pool, never a private/nested pool. Workers only sort
/// owned f64 vectors: no candidate rows, labels, settings, observation contexts,
/// logging, or mutable scientific/provenance state cross the worker boundary.
pub fn exact_external_auc(good: &[f64], null: &[f64], higher_is_better: bool) -> f64 {
    calculate(good, null, higher_is_better, true)
}

fn calculate(good: &[f64], null: &[f64], higher: bool, parallel: bool) -> f64 {
    let mut good: Vec<_> = good.iter().copied().filter(|v| v.is_finite()).collect();
    let mut null: Vec<_> = null.iter().copied().filter(|v| v.is_finite()).collect();
    if good.is_empty() || null.is_empty() {
        return f64::NAN;
    }
    if parallel {
        rayon::join(
            || good.par_sort_unstable_by(f64::total_cmp),
            || null.par_sort_unstable_by(f64::total_cmp),
        );
    } else {
        good.sort_unstable_by(f64::total_cmp);
        null.sort_unstable_by(f64::total_cmp);
    }
    sorted_counts(&good, &null, higher)
        .and_then(|(wins, ties, total)| ratio(wins, ties, total))
        .unwrap_or(f64::NAN)
}

fn sorted_counts(good: &[f64], null: &[f64], higher: bool) -> Option<(u128, u128, u128)> {
    let total = (good.len() as u128).checked_mul(null.len() as u128)?;
    let (mut wins, mut ties) = (0_u128, 0_u128);
    let (mut i, mut lower, mut upper) = (0, 0, 0);
    while i < good.len() {
        let g = good[i];
        let mut end = i + 1;
        while end < good.len() && good[end] == g {
            end += 1;
        }
        while lower < null.len() && null[lower] < g {
            lower += 1;
        }
        upper = upper.max(lower);
        while upper < null.len() && null[upper] == g {
            upper += 1;
        }
        let multiplicity = (end - i) as u128;
        let strict = if higher { lower } else { null.len() - upper };
        wins = wins.checked_add(multiplicity.checked_mul(strict as u128)?)?;
        ties = ties.checked_add(multiplicity.checked_mul((upper - lower) as u128)?)?;
        i = end;
    }
    Some((wins, ties, total))
}

fn ratio(wins: u128, ties: u128, total: u128) -> Option<f64> {
    if total == 0 || wins.checked_add(ties)? > total {
        return None;
    }
    Some((wins as f64 + 0.5 * ties as f64) / total as f64)
}

/// Serial sorting control for bounded performance comparisons only.
#[cfg(feature = "bench")]
pub fn benchmark_serial_auc(good: &[f64], null: &[f64], higher: bool) -> f64 {
    calculate(good, null, higher, false)
}

/// Original algorithm, restricted to bounded synthetic test/benchmark inputs.
#[cfg(any(test, feature = "bench"))]
pub fn bounded_pairwise_reference(good: &[f64], null: &[f64], higher: bool) -> f64 {
    assert!(good.len().saturating_mul(null.len()) <= 10_000_000);
    let (mut wins, mut total) = (0.0, 0.0);
    for &g in good.iter().filter(|g| g.is_finite()) {
        for &n in null.iter().filter(|n| n.is_finite()) {
            total += 1.0;
            if (higher && g > n) || (!higher && g < n) {
                wins += 1.0;
            } else if g == n {
                wins += 0.5;
            }
        }
    }
    if total == 0.0 {
        f64::NAN
    } else {
        wins / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn same(a: f64, b: f64) {
        assert!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "{a:?} != {b:?}"
        );
    }
    #[test]
    fn exact_auc_examples_directions_ties_signed_zero_and_filtering() {
        same(exact_external_auc(&[1.0, 2.0], &[0.0, 1.0], true), 0.875);
        same(exact_external_auc(&[1.0, 2.0], &[0.0, 1.0], false), 0.125);
        for (g, n) in [
            (vec![-0.0, 0.0], vec![0.0, -0.0, 0.0]),
            (vec![1.0, f64::from_bits(1.0_f64.to_bits() + 1)], vec![1.0]),
            (
                vec![f64::NAN, 3.0, f64::INFINITY],
                vec![3.0, f64::NEG_INFINITY],
            ),
            (vec![], vec![1.0]),
            (vec![1.0], vec![]),
            (vec![f64::NAN, f64::INFINITY], vec![f64::NEG_INFINITY]),
            (vec![3.0; 9], vec![3.0; 17]),
        ] {
            for higher in [false, true] {
                same(
                    exact_external_auc(&g, &n, higher),
                    bounded_pairwise_reference(&g, &n, higher),
                );
            }
        }
    }
    #[test]
    fn exact_auc_randomized_permutations_and_worker_counts() {
        use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let pools: Vec<_> = [1, 2, 4, 8]
            .into_iter()
            .map(|n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap()
            })
            .collect();
        for _ in 0..100 {
            let mut g: Vec<_> = (0..rng.gen_range(0..80))
                .map(|_| rng.gen_range(-10..=10) as f64)
                .collect();
            let mut n: Vec<_> = (0..rng.gen_range(0..130))
                .map(|_| rng.gen_range(-10..=10) as f64)
                .collect();
            for higher in [false, true] {
                let expected = bounded_pairwise_reference(&g, &n, higher);
                for pool in &pools {
                    g.shuffle(&mut rng);
                    n.shuffle(&mut rng);
                    same(
                        pool.install(|| exact_external_auc(&g, &n, higher)),
                        expected,
                    );
                }
            }
        }
    }
    #[test]
    fn exact_auc_large_counts_and_precision_boundary() {
        assert_eq!(ratio(0, 1_u128 << 53, 1_u128 << 53), Some(0.5));
        assert_eq!(ratio(1_u128 << 80, 0, 1_u128 << 81), Some(0.5));
        assert_eq!(ratio(u128::MAX, 1, u128::MAX), None);
        assert_eq!(ratio(2, 0, 1), None);
        assert_eq!(ratio(0, 0, 0), None);
        let pairs = 1_u128 << 52;
        assert_eq!(
            ratio(pairs - 1, 1, pairs),
            Some((pairs as f64 - 0.5) / pairs as f64)
        );
        let g = vec![0.0; 100_000];
        let n = vec![-0.0; 90_000];
        assert_eq!(
            sorted_counts(&g, &n, true),
            Some((0, 9_000_000_000, 9_000_000_000))
        );
        same(exact_external_auc(&g, &n, true), 0.5);
    }
}
