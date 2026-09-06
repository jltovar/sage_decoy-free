//! Downstream diagnostics only; never part of raw prediction production or parsing.
use sage_core::database::IndexedDatabase;
use sage_core::scoring::DfFeature;

pub(crate) fn log_external_feature_local_separation(features: &[DfFeature], db: &IndexedDatabase) {
    let joined = features
        .iter()
        .filter(|f| f.core.external_features.ms2rescore_feature_joined)
        .count();

    if joined == 0 {
        log::warn!("external feature local diagnostics skipped: no joined features");
        return;
    }

    log::info!(
        "external feature local diagnostics: joined_features={}/{}",
        joined,
        features.len()
    );

    log_external_feature_one_global(features, db, "ms2rescore_ms2pip_pcc", true, |f| {
        f.core.external_features.ms2rescore_ms2pip_pcc as f64
    });

    log_external_feature_one_global(features, db, "ms2rescore_spectral_angle", true, |f| {
        f.core.external_features.ms2rescore_spectral_angle as f64
    });

    log_external_feature_one_global(features, db, "ms2rescore_deeplc_abs_rt_error", false, |f| {
        f.core.external_features.ms2rescore_deeplc_abs_rt_error as f64
    });

    log_external_feature_one_global(features, db, "tims2rescore_abs_ccs_error", false, |f| {
        f.core.external_features.tims2rescore_abs_ccs_error as f64
    });

    log_external_feature_one_global(features, db, "tims2rescore_pct_ccs_error", false, |f| {
        f.core.external_features.tims2rescore_pct_ccs_error as f64
    });

    log_external_feature_by_matched_peaks_bin(
        features,
        db,
        "ms2rescore_deeplc_abs_rt_error",
        false,
        |f| f.core.external_features.ms2rescore_deeplc_abs_rt_error as f64,
    );

    log_external_feature_by_matched_peaks_bin(
        features,
        db,
        "tims2rescore_pct_ccs_error",
        false,
        |f| f.core.external_features.tims2rescore_pct_ccs_error as f64,
    );

    log_external_feature_by_matched_peaks_bin(features, db, "ms2rescore_ms2pip_pcc", true, |f| {
        f.core.external_features.ms2rescore_ms2pip_pcc as f64
    });
}

fn log_external_feature_one_global<F>(
    features: &[DfFeature],
    db: &IndexedDatabase,
    name: &str,
    higher_is_better: bool,
    getter: F,
) where
    F: Fn(&DfFeature) -> f64,
{
    let mut reference = Vec::new();
    let mut entrapment = Vec::new();

    for f in features
        .iter()
        .filter(|f| f.core.rank == 1 && f.core.external_features.ms2rescore_feature_joined)
    {
        let x = getter(f);
        if !x.is_finite() {
            continue;
        }

        let proteins = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
        if external_feature_is_entrapment(&proteins) {
            entrapment.push(x);
        } else {
            reference.push(x);
        }
    }

    if reference.len() < 10 || entrapment.len() < 10 {
        log::info!(
            "external feature diagnostic {name}: insufficient rank1 reference/entrapment values reference={} entrapment={}",
            reference.len(),
            entrapment.len()
        );
        return;
    }

    let auc = external_feature_auc(&reference, &entrapment, higher_is_better);

    log::info!(
        "external feature diagnostic {name}: reference_n={} entrapment_n={} reference_median={:.6} entrapment_median={:.6} auc_ref_vs_ent={:.4} higher_is_better={}",
        reference.len(),
        entrapment.len(),
        external_feature_median(reference),
        external_feature_median(entrapment),
        auc,
        higher_is_better
    );
}

fn log_external_feature_by_matched_peaks_bin<F>(
    features: &[DfFeature],
    db: &IndexedDatabase,
    name: &str,
    higher_is_better: bool,
    getter: F,
) where
    F: Fn(&DfFeature) -> f64,
{
    let bins: &[(u32, u32, &str)] = &[
        (0, 4, "matched_peaks_0_4"),
        (5, 7, "matched_peaks_5_7"),
        (8, 12, "matched_peaks_8_12"),
        (13, u32::MAX, "matched_peaks_13_plus"),
    ];

    for &(lo, hi, label) in bins {
        let mut reference = Vec::new();
        let mut entrapment = Vec::new();

        for f in features
            .iter()
            .filter(|f| f.core.rank == 1 && f.core.external_features.ms2rescore_feature_joined)
        {
            let mp = f.core.matched_peaks;
            if mp < lo || mp > hi {
                continue;
            }

            let x = getter(f);
            if !x.is_finite() {
                continue;
            }

            let proteins = db[f.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            if external_feature_is_entrapment(&proteins) {
                entrapment.push(x);
            } else {
                reference.push(x);
            }
        }

        if reference.len() < 10 || entrapment.len() < 10 {
            log::info!(
                "external feature diagnostic {name} bin={label}: insufficient values reference={} entrapment={}",
                reference.len(),
                entrapment.len()
            );
            continue;
        }

        let auc = external_feature_auc(&reference, &entrapment, higher_is_better);

        log::info!(
            "external feature diagnostic {name} bin={label}: reference_n={} entrapment_n={} reference_median={:.6} entrapment_median={:.6} auc_ref_vs_ent={:.4} higher_is_better={}",
            reference.len(),
            entrapment.len(),
            external_feature_median(reference),
            external_feature_median(entrapment),
            auc,
            higher_is_better
        );
    }
}

fn external_feature_is_entrapment(proteins: &str) -> bool {
    let u = proteins.to_ascii_uppercase();

    u.contains("ENTRAP")
        || u.contains("FOREIGN")
        || u.contains("ARATH")
        || u.contains("YEAST")
        || u.contains("CAEEL")
        || u.contains("DROME")
        || u.contains("ECOLI")
        || u.contains("HUMAN")
        || u.contains("RAT")
}

fn external_feature_median(mut xs: Vec<f64>) -> f64 {
    xs.retain(|x| x.is_finite());
    if xs.is_empty() {
        return f64::NAN;
    }

    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn external_feature_auc(good: &[f64], null: &[f64], higher_is_better: bool) -> f64 {
    sage_core::ml::external_auc::exact_external_auc(good, null, higher_is_better)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_auc_matches_bounded_pairwise_in_both_directions() {
        let good = [
            -0.0,
            0.0,
            1.0,
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::from_bits(1.0f64.to_bits() + 1),
        ];
        let null = [0.0, -0.0, 1.0, 2.0, f64::NEG_INFINITY];
        for higher in [false, true] {
            let mut credit = 0.0f64;
            let mut pairs = 0.0f64;
            for a in good.into_iter().filter(|v| v.is_finite()) {
                for b in null.into_iter().filter(|v| v.is_finite()) {
                    pairs += 1.0;
                    credit += if a == b {
                        0.5
                    } else if (higher && a > b) || (!higher && a < b) {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
            assert_eq!(
                external_feature_auc(&good, &null, higher).to_bits(),
                (credit / pairs).to_bits()
            );
            assert!(external_feature_auc(&[], &null, higher).is_nan());
        }
    }
}
