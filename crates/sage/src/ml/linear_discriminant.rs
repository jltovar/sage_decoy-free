//! Linear Discriminant Analysis for FDR refinement

use super::matrix::Matrix;
use rayon::prelude::*;

use crate::mass::Tolerance;
use crate::scoring::{FeatureCore, TdcFeature};

// Declare, so that we have compile time checking of matrix dimensions
const FEATURES: usize = 20;
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

struct Features<'a>(&'a [f64]);

impl std::fmt::Debug for Features<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (i, name) in FEATURE_NAMES.iter().enumerate() {
            map.entry(name, &self.0[i]);
        }
        map.finish()
    }
}

// Vanilla-compatible Gauss-Jordan solver used for the TDC/LDA path.
// This local implementation preserves the legacy solved-state semantics used by
// vanilla LDA and is intentionally kept separate from the decoy-free solver.
#[derive(Debug)]
struct GaussVanilla {
    left: Matrix,
    right: Matrix,
}

#[inline]
fn swap_rows(m: &mut Matrix, i: usize, j: usize) {
    for k in 0..m.cols {
        let tmp = m[(i, k)];
        m[(i, k)] = m[(j, k)];
        m[(j, k)] = tmp;
    }
}

impl GaussVanilla {
    #[inline]
    fn approx_zero(x: f64, tol: f64) -> bool {
        x.abs() <= tol
    }

    #[inline]
    fn approx_one(x: f64, tol: f64) -> bool {
        (x - 1.0).abs() <= tol
    }

    fn fill_zero(&mut self, eps: f64) {
        for i in 0..self.left.cols {
            self.left[(i, i)] += eps;
        }
    }

    // Vanilla-compatible solved-state check preserving legacy off-diagonal semantics.
    fn left_solved(&self) -> bool {
        let n = self.left.cols;
        let diag_eps = 1e-8;

        for i in 0..n {
            for j in 0..n {
                let x = self.left[(i, j)];
                if i == j {
                    if !Self::approx_one(x, diag_eps) && !Self::approx_zero(x, diag_eps) {
                        log::debug!(
                            "Finding solution to linear system failed: left side of matrix [{},{}] = {}",
                            i, j, x
                        );
                        return false;
                    }
                } else if x > 1E-8 {
                    log::debug!(
                        "Finding solution to linear system failed: left side of matrix [{},{}] = {}",
                        i, j, x
                    );
                    return false;
                }
            }
        }
        true
    }

    fn echelon(&mut self) {
        let (m, n) = self.left.shape();
        let mut h = 0;
        let mut k = 0;

        while h < m && k < n {
            let mut max = (h, self.left[(h, k)].abs());
            for i in h..m {
                let candidate = self.left[(i, k)].abs();
                if candidate > max.1 {
                    max = (i, candidate);
                }
            }

            let i = max.0;
            if Self::approx_zero(self.left[(i, k)], 1e-12) {
                k += 1;
                continue;
            }

            if h != i {
                swap_rows(&mut self.left, h, i);
                swap_rows(&mut self.right, h, i);
            }

            for i in h + 1..m {
                let factor = self.left[(i, k)] / self.left[(h, k)];
                self.left[(i, k)] = 0.0;
                for j in k + 1..n {
                    self.left[(i, j)] -= self.left[(h, j)] * factor;
                }
                for j in 0..self.right.cols {
                    self.right[(i, j)] -= self.right[(h, j)] * factor;
                }
            }
            h += 1;
            k += 1;
        }
    }

    fn reduce(&mut self) {
        for i in (0..self.left.rows).rev() {
            for j in 0..self.left.cols {
                let x = self.left[(i, j)];
                if Self::approx_zero(x, 1e-12) {
                    continue;
                }
                for k in j..self.left.cols {
                    self.left[(i, k)] /= x;
                }
                for k in 0..self.right.cols {
                    self.right[(i, k)] /= x;
                }
                break;
            }
        }
    }

    fn backfill(&mut self) {
        for i in (0..self.left.rows).rev() {
            for j in 0..self.left.cols {
                if Self::approx_zero(self.left[(i, j)], 1e-12) {
                    continue;
                }
                for k in 0..i {
                    let factor = self.left[(k, j)] / self.left[(i, j)];
                    for h in 0..self.left.cols {
                        self.left[(k, h)] -= self.left[(i, h)] * factor;
                    }
                    for h in 0..self.right.cols {
                        self.right[(k, h)] -= self.right[(i, h)] * factor;
                    }
                }
                break;
            }
        }
    }

    fn solve_inner(left: Matrix, right: Matrix, eps: f64) -> Option<Matrix> {
        let mut g = GaussVanilla { left, right };
        g.fill_zero(eps);
        g.echelon();
        g.reduce();
        g.backfill();
        if g.left_solved() {
            Some(g.right)
        } else {
            None
        }
    }

    fn solve(left: Matrix, right: Matrix) -> Option<Matrix> {
        let mut eps = 1E-8;
        while eps <= 1.0 {
            if let Some(mat) = GaussVanilla::solve_inner(left.clone(), right.clone(), eps) {
                return Some(mat);
            }
            eps *= 10.0;
        }
        None
    }
}

struct LinearDiscriminantAnalysis {
    eigenvector: Vec<f64>,
}

impl LinearDiscriminantAnalysis {
    pub fn train(features: &Matrix, decoy: &[bool]) -> Option<LinearDiscriminantAnalysis> {
        if features.rows != decoy.len() {
            log::warn!(
        "linear discriminant training received mismatched feature/label lengths: features.rows={}, labels={}; using heuristic fallback",
        features.rows,
        decoy.len()
    );
            return None;
        }

        let n_decoy = decoy.iter().filter(|&&label| label).count();
        let n_target = decoy.len().saturating_sub(n_decoy);
        if n_decoy == 0 || n_target == 0 {
            log::warn!("linear discriminant training requires at least one target and one decoy");
            return None;
        }

        // Calculate class means, and overall mean
        let x_bar = features.mean();
        let mut scatter_within = Matrix::zeros(features.cols, features.cols);
        let mut scatter_between = Matrix::zeros(features.cols, features.cols);

        let mut class_means = Vec::new();

        for class in [true, false] {
            let count = decoy.iter().filter(|&label| *label == class).count();

            let class_data = (0..features.rows)
                .zip(decoy)
                .filter(|&(_, label)| *label == class)
                .flat_map(|(row, _)| features.row(row).copied())
                .collect::<Vec<f64>>();

            let mut class_data = Matrix::new(class_data, count, features.cols);
            let class_mean = class_data.mean();

            for row in 0..class_data.rows {
                for col in 0..class_data.cols {
                    class_data[(row, col)] -= class_mean[col];
                }
            }

            let cov = class_data.transpose().dot(&class_data) / class_data.rows as f64;
            scatter_within += cov;

            let diff = Matrix::col_vector(
                class_mean
                    .iter()
                    .zip(x_bar.iter())
                    .map(|(x, y)| x - y)
                    .collect::<Vec<_>>(),
            );

            let weighted_diff = Matrix::col_vector(
                diff.data
                    .iter()
                    .map(|x| x * count as f64)
                    .collect::<Vec<_>>(),
            );
            scatter_between += weighted_diff.dot(&diff.transpose());
            class_means.extend(class_mean);
        }

        // Use overall mean as the initial vector for power method
        let mut evec = GaussVanilla::solve(scatter_within, scatter_between)
            .map(|mat| mat.power_method(&x_bar))?;

        // Ensure Target class scores are higher than Decoy for consistent ranking
        let class_means = Matrix::new(class_means, 2, features.cols);
        let coef = class_means.dotv(&evec);
        if coef[1] < coef[0] {
            evec.iter_mut().for_each(|c| *c *= -1.0);
        }

        log::trace!("- linear model fit with {:?}", Features(&evec));
        Some(LinearDiscriminantAnalysis { eigenvector: evec })
    }

    pub fn score(&self, features: &Matrix) -> Vec<f64> {
        features.dotv(&self.eigenvector)
    }
}

pub fn score_psms(
    scores: &mut [TdcFeature],
    precursor_tol: Tolerance,
    decoy_free: bool,
) -> Option<()> {
    if scores.is_empty() {
        return None;
    }

    // Decoy-free mode bypasses the vanilla TDC/LDA rescoring path.
    if decoy_free {
        return Some(());
    }

    log::trace!("fitting linear discriminant model...");

    // Vanilla TDC labels decoys as -1.
    let decoys = scores
        .par_iter()
        .map(|sc| sc.core.label == -1)
        .collect::<Vec<_>>();

    // Use the vanilla mass-error definition for the active precursor tolerance.
    let mass_error = match precursor_tol {
        Tolerance::Ppm(_, _) => |feat: &FeatureCore| feat.delta_mass as f64,
        Tolerance::Pct(_, _) => unreachable!("Pct tolerance should never be used on mz"),
        Tolerance::Da(_, _) => |feat: &FeatureCore| (feat.expmass - feat.calcmass) as f64,
    };

    let (bw_adjust, bin_size) = match precursor_tol {
        Tolerance::Ppm(lo, hi) => (2.0f64, (hi - lo).max(100.0)),
        Tolerance::Pct(_, _) => unreachable!("Pct tolerance should never be used on mz"),
        Tolerance::Da(lo, hi) => (0.1f64, (hi - lo).max(1000.0)),
    };

    let delta_mass = scores
        .par_iter()
        .map(|s| mass_error(&s.core))
        .collect::<Vec<_>>();

    // Fit the vanilla non-parametric mass-error model.
    let mass_model = super::kde::Builder::default()
        .monotonic(false)
        .bw_adjust(move |x| x * bw_adjust)
        .bins(bin_size.ceil().abs() as usize)
        .build(&delta_mass, &decoys);

    // Construct the vanilla LDA feature embedding, including aligned RT.
    let features = scores
        .par_iter()
        .flat_map_iter(|s| {
            let perc = &s.core;

            let p = perc.spectrum_p_value as f64;
            let p = if p.is_finite() && p > 0.0 {
                p.min(1.0)
            } else {
                f64::MIN_POSITIVE
            };

            let poisson = (-p.log10()).ln_1p();

            let x: [f64; FEATURES] = [
                (perc.rank as f64),
                (perc.charge as f64),
                (perc.hyperscore).ln_1p(),
                (perc.delta_next).ln_1p(),
                (perc.delta_best).ln_1p(),
                mass_model.posterior_error(mass_error(perc)),
                (perc.isotope_error as f64),
                (perc.average_ppm as f64),
                (poisson),
                (perc.matched_intensity_pct as f64).ln_1p(),
                (perc.matched_peaks as f64),
                (perc.longest_b as f64).ln_1p(),
                (perc.longest_y as f64).ln_1p(),
                (perc.longest_y as f64 / perc.peptide_len as f64),
                (perc.peptide_len as f64).ln_1p(),
                (perc.missed_cleavages as f64),
                (perc.aligned_rt as f64),
                (perc.ims as f64),
                (perc.delta_rt_model as f64).clamp(0.001, 0.999).sqrt(),
                (perc.delta_ims_model as f64).clamp(0.001, 0.999).sqrt(),
            ];
            Some(x)
        })
        .collect::<Vec<_>>();

    let nrows = features.len();

    if nrows != decoys.len() {
        log::warn!(
        "linear discriminant feature/label length mismatch: features={}, labels={}; using heuristic fallback",
        nrows,
        decoys.len()
    );
        return None;
    }

    let features = features
        .into_iter()
        .flat_map(|row| row.into_iter())
        .collect::<Vec<f64>>();

    let features = Matrix::new(features, nrows, FEATURES);

    if features.rows != decoys.len() {
        log::warn!(
        "linear discriminant matrix/label length mismatch: matrix_rows={}, labels={}; using heuristic fallback",
        features.rows,
        decoys.len()
    );
        return None;
    }

    let lda = LinearDiscriminantAnalysis::train(&features, &decoys)?;

    if !lda.eigenvector.iter().all(|f| f.is_finite()) {
        log::error!(
            "linear model eigenvector includes NaN: this likely indicates a bug, please report!"
        );
        for row in 0..features.rows {
            if features.row(row).any(|f| !f.is_finite()) {
                let row = features.row(row).collect::<Vec<_>>();
                log::error!("example feature vector with NaN: {:?}", row);
                break;
            }
        }
        return None;
    }

    let discriminants = lda.score(&features);

    log::trace!("- fitting non-parametric model for posterior error probabilities");
    let kde = super::kde::Builder::default().build(&discriminants, &decoys);

    scores
        .par_iter_mut()
        .zip(&discriminants)
        .for_each(|(perc, score)| {
            perc.discriminant_score = *score as f32;
            perc.posterior_error = kde.posterior_error(*score).log10() as f32;
            if perc.posterior_error.is_infinite() {
                perc.posterior_error = -324.0;
            }
        });

    Some(())
}
