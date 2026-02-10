//! Linear Discriminant Analysis for FDR refinement

use super::gauss::Gauss;
use super::matrix::Matrix;
use rayon::prelude::*;

use crate::mass::Tolerance;
use crate::scoring::{FeatureCore, TdcFeature};

const FEATURES: usize = 20;

#[allow(dead_code)]
const FEATURE_NAMES: [&str; FEATURES] = [
    "rank",
    "charge",
    "ln1p(hyperscore)",
    "ln1p(delta_next)",
    "ln1p(delta_best)",
    "delta_mass_model",
    "isotope_error",
    "average_ppm",
    "ln1p(-poisson)",
    "ln1p(matched_intensity_pct)",
    "ln1p(matched_peaks)",
    "ln1p(longest_b)",
    "ln1p(longest_y)",
    "longest_y_pct",
    "ln1p(peptide_len)",
    "missed_cleavages",
    "rt",
    "ims",
    "sqrt(delta_rt_model)",
    "sqrt(delta_ims_model)",
];

#[allow(dead_code)]
struct Features<'a>(&'a [f64]);

impl std::fmt::Debug for Features<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(FEATURE_NAMES.iter().zip(self.0))
            .finish()
    }
}

pub struct LinearDiscriminantAnalysis {
    eigenvector: Vec<f64>,
}

impl LinearDiscriminantAnalysis {
    pub fn train(features: &Matrix, labels: &[bool]) -> Option<Self> {
        assert_eq!(
            features.rows,
            labels.len(),
            "Features and labels must have the same number of rows"
        );

        let mut target_means = vec![0.0; features.cols];
        let mut decoy_means = vec![0.0; features.cols];
        let mut target_count = 0;
        let mut decoy_count = 0;

        for i in 0..features.rows {
            // With the updated matrix.rs, row() yields &f64
            let row = features.row(i);
            if labels[i] {
                target_count += 1;
                for (j, val) in row.enumerate() {
                    target_means[j] += *val;
                }
            } else {
                decoy_count += 1;
                for (j, val) in row.enumerate() {
                    decoy_means[j] += *val;
                }
            }
        }

        if target_count < 2 || decoy_count < 2 {
            log::warn!(
                "LDA: Too few targets ({}) or decoys ({}) to train model",
                target_count,
                decoy_count
            );
            return None;
        }

        for i in 0..features.cols {
            target_means[i] /= target_count as f64;
            decoy_means[i] /= decoy_count as f64;
        }

        let mut scatter_matrix = Matrix::zeros(features.cols, features.cols);

        for i in 0..features.rows {
            // row(i) yields &f64, so .cloned() -> f64 is correct for creating Vec<f64>
            let row: Vec<f64> = features.row(i).cloned().collect();
            let means = if labels[i] {
                &target_means
            } else {
                &decoy_means
            };

            for j in 0..features.cols {
                for k in 0..features.cols {
                    let d1 = row[j] - means[j];
                    let d2 = row[k] - means[k];
                    scatter_matrix.data[j * features.cols + k] += d1 * d2;
                }
            }
        }

        let mean_diff: Vec<f64> = target_means
            .iter()
            .zip(decoy_means.iter())
            .map(|(t, d)| t - d)
            .collect();

        // Convert mean_diff to column matrix for solve
        let mean_diff_mat = Matrix::col_vector(mean_diff);
        let eigenvector = Gauss::solve(scatter_matrix, mean_diff_mat)?;

        // Unwrap matrix back to vec
        Some(Self {
            eigenvector: eigenvector.take(),
        })
    }

    pub fn score(&self, features: &Matrix) -> Vec<f64> {
        let eigen_mat = Matrix::col_vector(self.eigenvector.clone());
        features.dot(&eigen_mat).take()
    }
}

fn standardize(features: &mut [f64], n_features: usize) {
    for i in 0..n_features {
        let mut mean = 0.0;
        let mut variance = 0.0;
        let mut count = 0.0;

        for j in (i..features.len()).step_by(n_features) {
            let x = features[j];
            mean += x;
            variance += x * x;
            count += 1.0;
        }

        mean /= count;
        variance = (variance / count) - (mean * mean);
        let std_dev = variance.sqrt();

        for j in (i..features.len()).step_by(n_features) {
            features[j] = (features[j] - mean) / std_dev;
        }
    }
}

#[rustfmt::skip]
fn embed(feature: &FeatureCore, precursor_tol: Tolerance) -> [f64; FEATURES] {
    let delta_mass_model = match precursor_tol {
        Tolerance::Ppm(_, _) => feature.delta_mass,
        Tolerance::Da(_, _) => feature.expmass - feature.calcmass - feature.isotope_error,
        Tolerance::Pct(_, _) => unreachable!("Pct tolerance should never be used on mz"),
    };

    [
        (feature.rank as f64).ln_1p(),
        feature.charge as f64,
        feature.hyperscore.ln_1p(),
        feature.delta_next.ln_1p(),
        feature.delta_best.ln_1p(),
        delta_mass_model as f64,
        feature.isotope_error as f64,
        feature.average_ppm as f64,
        (-feature.poisson).ln_1p(),
        feature.matched_intensity_pct.ln_1p() as f64,
        (feature.matched_peaks as f64).ln_1p(),
        (feature.longest_b as f64).ln_1p(),
        (feature.longest_y as f64).ln_1p(),
        feature.longest_y_pct as f64,
        (feature.peptide_len as f64).ln_1p(),
        feature.missed_cleavages as f64,
        feature.rt as f64,
        feature.ims as f64,
        (feature.delta_rt_model as f64).clamp(0.001, 1.0).sqrt(),
        (feature.delta_ims_model as f64).clamp(0.0, 1.0).sqrt(),
    ]
}

pub fn score_psms(
    features: &mut [TdcFeature],
    precursor_tol: Tolerance,
    decoy_free: bool,
) -> Option<()> {
    if features.is_empty() {
        return None;
    }

    if decoy_free {
        return Some(());
    }

    log::info!("- fitting linear discriminant model...");

    let embedding: Vec<f64> = features
        .par_iter()
        .flat_map(|feat| embed(&feat.core, precursor_tol))
        .collect();

    let mut matrix = Matrix::new(embedding, features.len(), FEATURES);
    standardize(&mut matrix.data, FEATURES);

    let labels: Vec<bool> = features.iter().map(|feat| feat.core.label == 1).collect();

    let lda = LinearDiscriminantAnalysis::train(&matrix, &labels)?;
    let scores = lda.score(&matrix);

    log::info!("- calculating posterior error probabilities...");
    let kde = super::kde::Builder::default().build(&scores, &labels);

    features
        .par_iter_mut()
        .zip(scores.into_par_iter())
        .for_each(|(feat, score)| {
            feat.discriminant_score = score as f32;
            feat.posterior_error = kde.posterior_error(score).log10() as f32;
            if feat.posterior_error.is_infinite() {
                feat.posterior_error = -324.0;
            }
        });

    Some(())
}
