//! Decoy-free Nokoi-like model fitting utility.
//!
//! The methods in this module are based on the work of Giulia Gonnelli et al. published here:
//!
//! A Decoy-Free Approach to the Identification of Peptides
//! Giulia Gonnelli, Michiel Stock, Jan Verwaeren, Davy Maddelein, Bernard De Baets, Lennart Martens, and Sven Degroeve
//! Journal of Proteome Research 2015 14 (4), 1792-1798
//! DOI: 10.1021/pr501164r
//! https://pubs.acs.org/doi/10.1021/pr501164r

use crate::scoring::{DfFeature, FeatureCore};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

const NOKOI_CROSSFIT_SEED: u64 = 0x5EED_5EED_5EED_5EED;

#[derive(Clone, Debug)]
pub struct NokoiEvidence {
    pub p_values: Vec<f64>,
    pub peps: Vec<f64>,
}

/// Configuration for Nokoi Rescoring
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NokoiConfig {
    pub enabled: bool,
    pub train_fdr: f32,

    // Optimizer hyperparameters
    pub learning_rate: f64,
    pub epochs: usize,
    pub patience: usize,

    // Initial/fallback single-lambda value used by direct training calls
    pub l1_lambda: f64,

    // CV-tuned lambda grid
    pub l1_lambda_min: f64,
    pub l1_lambda_max: f64,
    pub l1_lambda_steps: usize,
}

impl Default for NokoiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            train_fdr: 0.01,
            learning_rate: 0.1,
            epochs: 250,
            patience: 10,
            l1_lambda: 0.001,
            l1_lambda_min: 1e-4,
            l1_lambda_max: 1e-1,
            l1_lambda_steps: 10,
        }
    }
}

/// A standardized feature vector for a PSM
#[derive(Debug, Clone)]
pub struct PsmData {
    pub features: Vec<f64>,
    pub label: f64,
    pub original_idx: usize,
}

/// Extract features from a Sage Feature struct
pub fn extract_features(feat: &FeatureCore) -> Vec<f64> {
    // Safety check for log transform
    let log_intensity = if feat.ms2_intensity > 0.0 {
        (feat.ms2_intensity as f64).log10()
    } else {
        0.0
    };

    vec![
        feat.hyperscore as f64,              // 1. Hyperscore
        feat.delta_next as f64,              // 2. Delta Next
        (feat.average_ppm as f64).abs(),     // 3. Precursor PPM (Abs)
        (feat.delta_rt_model as f64).abs(),  // 4. Delta RT (Abs)
        (feat.delta_ims_model as f64).abs(), // 5. Delta IM (Abs)
        feat.charge as f64,                  // 6. Charge
        feat.peptide_len as f64,             // 7. Length
        feat.matched_peaks as f64,           // 8. Matched Peaks
        feat.matched_intensity_pct as f64,   // 9. Intensity Pct
        feat.isotope_error as f64,           // 10. Isotope Error
        feat.longest_y_pct as f64,           // 11. Longest Y Pct
        log_intensity,                       // 12. Log MS2 Intensity
    ]
}

/// L1 Proximal Operator (Soft Thresholding)
/// Shrinks weights towards zero. If |w| < lambda, w becomes 0.
#[inline(always)]
fn soft_threshold(z: f64, threshold: f64) -> f64 {
    if z > threshold {
        z - threshold
    } else if z < -threshold {
        z + threshold
    } else {
        0.0
    }
}

/// Logistic Regression with L1 Lasso & FISTA Optimization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogisticRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LogisticRegression {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        let dot: f64 = self.weights.iter().zip(features).map(|(w, x)| w * x).sum();
        Self::sigmoid(dot + self.bias)
    }

    /// Calculate Log Loss (Cross Entropy) on a dataset
    fn evaluate(&self, data: &[PsmData]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut loss = 0.0;
        for sample in data {
            let p = self.predict(&sample.features).clamp(1e-9, 1.0 - 1e-9);
            let y = sample.label;
            loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
        }
        loss / (data.len() as f64)
    }

    /// Train using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
    /// Supports L1 Regularization and Early Stopping.
    ///
    /// Returns the final validation loss.
    fn train_with_early_stopping(
        &mut self,
        train_data: &[PsmData],
        val_data: &[PsmData],
        config: &NokoiConfig,
        lambda: f64,
    ) -> f64 {
        let n_train = train_data.len() as f64;
        if n_train == 0.0 {
            return f64::INFINITY;
        }

        // --- FISTA State Initialization ---
        // x_k: current weights
        let mut w = self.weights.clone();
        let mut b = self.bias;

        // y_k: momentum point (where we calculate gradient)
        let mut w_y = w.clone();
        let mut b_y = b;

        // t_k: step size parameter for momentum
        let mut t = 1.0;

        // Best model tracking
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let mut best_w = w.clone();
        let mut best_b = b;

        for _epoch in 0..config.epochs {
            // 1. Calculate Gradient at Momentum Point (y_k)
            let mut grad_w = vec![0.0; w.len()];
            let mut grad_b = 0.0;

            for sample in train_data {
                // Predict using momentum point weights
                let dot: f64 = w_y.iter().zip(&sample.features).map(|(w, x)| w * x).sum();
                let pred = Self::sigmoid(dot + b_y);
                let error = pred - sample.label;

                for (i, x) in sample.features.iter().enumerate() {
                    grad_w[i] += error * x;
                }
                grad_b += error;
            }

            // 2. Gradient Descent Step (Unregularized)
            let mut w_new = vec![0.0; w.len()];
            for i in 0..w.len() {
                // Standard GD step
                let z = w_y[i] - config.learning_rate * (grad_w[i] / n_train);
                // 3. Proximal Operator (L1 Soft Thresholding)
                // Threshold is learning_rate * lambda
                w_new[i] = soft_threshold(z, config.learning_rate * lambda);
            }
            let b_new = b_y - config.learning_rate * (grad_b / n_train);

            // 4. Validation & Early Stopping Check
            // Check loss every epoch (or every few epochs for speed, but every epoch is safer for small data)
            // We use the *actual* weights (w_new) for validation, not the momentum weights
            self.weights = w_new.clone();
            self.bias = b_new;
            let val_loss = self.evaluate(val_data);

            if val_loss < best_loss {
                best_loss = val_loss;
                best_w = w_new.clone();
                best_b = b_new;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            if patience_counter >= config.patience {
                break;
            }

            // 5. FISTA Momentum Update
            // t_{k+1}
            let t_new = (1.0 + (1.0 + 4.0_f64 * t * t).sqrt()) / 2.0;
            // momentum factor
            let mom = (t - 1.0) / t_new;

            for i in 0..w.len() {
                w_y[i] = w_new[i] + mom * (w_new[i] - w[i]);
            }
            b_y = b_new + mom * (b_new - b);

            // Update state
            w = w_new;
            b = b_new;
            t = t_new;
        }

        // Restore best weights
        self.weights = best_w;
        self.bias = best_b;

        best_loss
    }

    /// Train using K-fold Cross Validation to tune L1 Lambda
    pub fn train_cv(&mut self, data: &[PsmData], config: &NokoiConfig) -> f64 {
        let n_samples = data.len();
        if n_samples == 0 {
            return config.l1_lambda;
        }

        // Build log-spaced lambda grid from config
        let steps = config.l1_lambda_steps.max(1);
        let lambda_min = config.l1_lambda_min.max(1e-12);
        let lambda_max = config.l1_lambda_max.max(lambda_min);

        let mut lambdas = Vec::with_capacity(steps);
        let min_log = lambda_min.log10();
        let max_log = lambda_max.log10();
        let step = if steps > 1 {
            (max_log - min_log) / (steps as f64 - 1.0)
        } else {
            0.0
        };

        for i in 0..steps {
            lambdas.push(10.0_f64.powf(min_log + step * (i as f64)));
        }

        let k_folds = if n_samples >= 100 { 5 } else { 3 };

        let mut best_lambda = lambda_min;
        let mut best_loss = f64::INFINITY;

        log::info!(
            "Nokoi: Tuning L1 Lambda via {}-fold CV (range {:.6e} .. {:.6e}, steps={})...",
            k_folds,
            lambda_min,
            lambda_max,
            steps
        );

        for &lambda in &lambdas {
            let mut total_loss = 0.0;
            let mut fold_collapsed = false;

            for fold in 0..k_folds {
                let start = fold * n_samples / k_folds;
                let end = (fold + 1) * n_samples / k_folds;

                let mut train_set = Vec::with_capacity(n_samples - (end - start));
                let mut val_set = Vec::with_capacity(end - start);

                for (i, d) in data.iter().enumerate() {
                    if i >= start && i < end {
                        val_set.push(d.clone());
                    } else {
                        train_set.push(d.clone());
                    }
                }

                let mut fold_model = LogisticRegression::new(self.weights.len());
                let final_loss =
                    fold_model.train_with_early_stopping(&train_set, &val_set, config, lambda);

                let nonzero_features = fold_model
                    .weights
                    .iter()
                    .filter(|&&w| w.abs() > 1e-6)
                    .count();

                if nonzero_features == 0 {
                    fold_collapsed = true;
                    break;
                }

                total_loss += final_loss;
            }

            let avg_loss = if fold_collapsed {
                f64::INFINITY
            } else {
                total_loss / k_folds as f64
            };

            if avg_loss < best_loss {
                best_loss = avg_loss;
                best_lambda = lambda;
            }
        }

        if !best_loss.is_finite() {
            log::warn!(
                "Nokoi: all CV lambdas collapsed the model; defaulting to lowest lambda {:.6e}",
                lambda_min
            );
            best_lambda = lambda_min;
        } else {
            log::info!(
                "Nokoi: Best L1 Lambda = {:.6e} (Loss: {:.6})",
                best_lambda,
                best_loss
            );
        }

        let split_idx = (n_samples as f64 * 0.8) as usize;
        let (train, val) = data.split_at(split_idx);
        self.train_with_early_stopping(train, val, config, best_lambda);

        let nonzero_features = self.weights.iter().filter(|&&w| w.abs() > 1e-6).count();
        log::info!(
            "Nokoi: Final model selected {}/{} features.",
            nonzero_features,
            self.weights.len()
        );
        best_lambda
    }
}

pub const NOKOI_FEATURE_SCHEMA: [&str; 12] = [
    "hyperscore",
    "delta_next",
    "abs_average_ppm",
    "abs_delta_rt_model",
    "abs_delta_ims_model",
    "charge",
    "peptide_len",
    "matched_peaks",
    "matched_intensity_pct",
    "isotope_error",
    "longest_y_pct",
    "log10_ms2_intensity",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NokoiNormalization {
    pub medians: Vec<f64>,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl NokoiNormalization {
    fn apply(&self, values: &mut [f64]) -> bool {
        if values.len() != self.medians.len()
            || values.len() != self.means.len()
            || values.len() != self.stds.len()
        {
            return false;
        }
        for (index, value) in values.iter_mut().enumerate() {
            if !value.is_finite() {
                *value = self.medians[index];
            }
            let std = self.stds[index];
            *value = if std.is_finite() && std > 1e-9 {
                (*value - self.means[index]) / std
            } else {
                0.0
            };
        }
        values.iter().all(|value| value.is_finite())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NokoiCalibrationPoint {
    pub p_value: f64,
    pub pep: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NokoiArtifact {
    pub schema_version: u32,
    pub model_version: String,
    pub feature_schema: Vec<String>,
    pub min_null_rank: u32,
    pub max_null_rank: u32,
    pub crossfit_seed: u64,
    pub k_folds: usize,
    pub fold_sizes: Vec<usize>,
    pub config: NokoiConfig,
    pub selected_l1_lambda: f64,
    pub final_model: LogisticRegression,
    pub normalization: NokoiNormalization,
    pub null_scores_oof: Vec<f64>,
    pub development_pi0: f64,
    pub pep_calibration: Vec<NokoiCalibrationPoint>,
    pub positive_training_count: usize,
    pub negative_training_count: usize,
    #[serde(default)]
    pub reference_candidate_counts: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct NokoiFitResult {
    pub evidence: NokoiEvidence,
    pub artifact: NokoiArtifact,
}

fn normalize_features_portable(data: &mut [PsmData]) -> NokoiNormalization {
    let n_samples = data.len();
    if n_samples == 0 {
        return NokoiNormalization {
            medians: Vec::new(),
            means: Vec::new(),
            stds: Vec::new(),
        };
    }
    let n_features = data[0].features.len();
    let mut medians = vec![0.0; n_features];
    for index in 0..n_features {
        let mut values = data
            .iter()
            .map(|sample| sample.features[index])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        values.sort_by(|left, right| left.total_cmp(right));
        medians[index] = match values.len() {
            0 => 0.0,
            n if n % 2 == 1 => values[n / 2],
            n => (values[n / 2 - 1] + values[n / 2]) / 2.0,
        };
    }
    for sample in data.iter_mut() {
        for (index, value) in sample.features.iter_mut().enumerate() {
            if !value.is_finite() {
                *value = medians[index];
            }
        }
    }
    let n = n_samples as f64;
    let mut means = vec![0.0; n_features];
    for sample in data.iter() {
        for (index, value) in sample.features.iter().enumerate() {
            means[index] += *value;
        }
    }
    for mean in &mut means {
        *mean /= n;
    }
    let mut stds = vec![0.0; n_features];
    for sample in data.iter() {
        for (index, value) in sample.features.iter().enumerate() {
            stds[index] += (*value - means[index]).powi(2);
        }
    }
    for std in &mut stds {
        *std = (*std / n).sqrt();
        if !std.is_finite() || *std < 1e-9 {
            *std = 1.0;
        }
    }
    let normalization = NokoiNormalization {
        medians,
        means,
        stds,
    };
    for sample in data {
        debug_assert!(normalization.apply(&mut sample.features));
    }
    normalization
}

/// Feature preprocessing: median imputation followed by mean/std z-score standardization.
pub fn normalize_features(data: &mut [PsmData]) -> (Vec<f64>, Vec<f64>) {
    let normalization = normalize_features_portable(data);
    (normalization.means, normalization.stds)
}

/// Main entry point: Train model and return probabilities for all features
///
/// `is_positive` determines if a feature is considered a high-confidence target for training.
pub fn rescore(
    features: &[FeatureCore],
    train_fdr: f32,
    is_positive: impl Fn(&FeatureCore) -> bool,
) -> Option<Vec<f64>> {
    // Nokoi Config
    let config = NokoiConfig {
        enabled: true,
        train_fdr,
        learning_rate: 0.1,
        epochs: 500,
        patience: 15,
        l1_lambda: 0.005,
        l1_lambda_min: 1e-4,
        l1_lambda_max: 1e-1,
        l1_lambda_steps: 10,
    };

    // 1. Separate Positives and Negatives
    let mut positives = Vec::new();
    let mut negatives = Vec::new();
    let mut all_data = Vec::with_capacity(features.len());

    for (i, f) in features.iter().enumerate() {
        let feat_vec = extract_features(f);
        let psm = PsmData {
            features: feat_vec.clone(),
            label: 0.0, // Default, will set to 1.0 for positives in separate list
            original_idx: i,
        };

        // Check if positive using the provided closure
        if is_positive(f) {
            let mut pos_psm = psm.clone();
            pos_psm.label = 1.0;
            positives.push(pos_psm);
        }
        // Negative class: rank > 1
        else if f.rank > 1 {
            negatives.push(psm.clone());
        }

        all_data.push(psm); // We predict on everything later
    }

    // --- Fail-closed for Low Data (<50) to prevent unstable ML training ---
    let confident_count = positives.len();
    if confident_count < 50 {
        log::warn!(
            "Nokoi DF: Too few positives ({} < 50) - failing closed",
            confident_count
        );
        return None;
    } else if confident_count < 100 {
        // Between 50 and 100, we proceed but warn
        log::warn!(
            "Nokoi: Low training data ({}), model may vary.",
            confident_count
        );
    }

    // 1. Feature Extraction
    // --- Balanced Sampling ---
    // Create a balanced training set (cap at 10k per class) to prevent
    // model bias towards the majority negative class.

    let mut rng = thread_rng();
    positives.shuffle(&mut rng);
    negatives.shuffle(&mut rng);

    // Determine sample size: min of (pos count, neg count, 10,000 cap)
    let n_pos = positives.len();
    let n_neg = negatives.len();
    let sample_size = n_pos.min(n_neg).min(10_000);

    let mut training_data = Vec::with_capacity(sample_size * 2);
    training_data.extend_from_slice(&positives[0..sample_size]);
    training_data.extend_from_slice(&negatives[0..sample_size]);

    // Shuffle the training set so positives/negatives aren't clustered
    training_data.shuffle(&mut rng);

    log::info!(
        "Nokoi: Constructed balanced training set: {} positives, {} negatives (drawn from {} pos, {} neg)",
        sample_size, sample_size, n_pos, n_neg
    );

    // 2. Normalize
    // IMPORTANT: Learn normalization parameters (mean/std) ONLY from the balanced set.
    // If we learned from 'all_data', the massive number of low-quality spectra would skew the means.
    let (means, stds) = normalize_features(&mut training_data);

    // Apply that same normalization to the full dataset so predictions are valid
    for sample in all_data.iter_mut() {
        for (i, x) in sample.features.iter_mut().enumerate() {
            // Apply Z-score: (x - mean) / std
            if stds[i] > 1e-9 {
                *x = (*x - means[i]) / stds[i];
            } else {
                *x = 0.0;
            }
        }
    }

    // 3. Train with CV + Early Stopping + L1
    let n_features = training_data[0].features.len();
    let mut model = LogisticRegression::new(n_features);

    log::info!(
        "Nokoi: Training on {} pos, {} neg samples...",
        training_data.iter().filter(|d| d.label == 1.0).count(),
        training_data.iter().filter(|d| d.label == 0.0).count()
    );

    model.train_cv(&training_data, &config);

    // 4. Predict
    // We predict probabilities for *every* PSM in the file
    let probabilities: Vec<f64> = all_data
        .iter()
        .map(|sample| model.predict(&sample.features))
        .collect();

    if probabilities.len() != features.len() {
        log::warn!("Nokoi DF: output length mismatch. Failing closed.");
        return None;
    }
    if probabilities.iter().any(|p| !p.is_finite()) {
        log::warn!("Nokoi DF: output contains non-finite probabilities. Failing closed.");
        return None;
    }

    Some(probabilities)
}

/// Decoy-free entry point:
/// - Positives: decided by caller via `is_positive`
/// - Negatives: rank in [min_null_rank, max_null_rank]
///
/// Returns P(target) for every PSM (aligned 1:1 with `features`).
pub fn rescore_df(
    features: &[DfFeature],
    train_fdr: f32,
    min_null_rank: u32,
    max_null_rank: u32,
    is_positive: impl Fn(&DfFeature) -> bool,
) -> Option<Vec<f64>> {
    // Nokoi Config (same defaults as `rescore`)
    let config = NokoiConfig {
        enabled: true,
        train_fdr,
        learning_rate: 0.1,
        epochs: 500,
        patience: 15,
        l1_lambda: 0.005,
        l1_lambda_min: 1e-4,
        l1_lambda_max: 1e-1,
        l1_lambda_steps: 10,
    };

    // 1) Build positives/negatives + all_data (predict on everything)
    let mut positives: Vec<PsmData> = Vec::new();
    let mut negatives: Vec<PsmData> = Vec::new();
    let mut all_data: Vec<PsmData> = Vec::with_capacity(features.len());

    for (i, f) in features.iter().enumerate() {
        let feat_vec = extract_features(&f.core);
        let psm = PsmData {
            features: feat_vec.clone(),
            label: 0.0,
            original_idx: i,
        };

        // Positives: caller-defined (should include rank==1 gate)
        if is_positive(f) {
            let mut pos_psm = psm.clone();
            pos_psm.label = 1.0;
            positives.push(pos_psm);
        } else {
            // Negatives: rank-window only
            let r = f.core.rank as u32;
            if r >= min_null_rank && r <= max_null_rank {
                negatives.push(psm.clone());
            }
        }

        all_data.push(psm);
    }

    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "Nokoi DEBUG rescore_df: crossfit=OFF (current). min_null_rank={} max_null_rank={} positives={} negatives={} all={}",
            min_null_rank,
            max_null_rank,
            positives.len(),
            negatives.len(),
            all_data.len()
        );
    }

    // --- Fail-closed for low data to prevent unstable ML training ---
    let confident_count = positives.len();
    if confident_count < 50 {
        log::warn!(
            "Nokoi: Too few positives ({} < 50) - failing closed",
            confident_count
        );
        return None;
    } else if confident_count < 100 {
        log::warn!(
            "Nokoi DF: Low training data ({}), model may vary.",
            confident_count
        );
    }

    // 2) Balanced sampling (same as existing)
    let mut rng = thread_rng();
    positives.shuffle(&mut rng);
    negatives.shuffle(&mut rng);

    let n_pos = positives.len();
    let n_neg = negatives.len();

    // If negatives are too few, fail closed.
    if n_neg < 50 {
        log::warn!(
            "Nokoi DF: Too few negatives from rank window ({} < 50) - failing closed",
            n_neg
        );
        return None;
    }

    let sample_size = n_pos.min(n_neg).min(10_000);

    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "Nokoi DEBUG training sample_size={} (n_pos={} n_neg={})",
            sample_size,
            n_pos,
            n_neg
        );
    }

    let mut training_data = Vec::with_capacity(sample_size * 2);
    training_data.extend_from_slice(&positives[0..sample_size]);
    training_data.extend_from_slice(&negatives[0..sample_size]);

    training_data.shuffle(&mut rng);

    log::info!(
        "Nokoi DF: Constructed balanced training set: {} positives, {} negatives (drawn from {} pos, {} neg)",
        sample_size, sample_size, n_pos, n_neg
    );

    // 3) Normalize (same as existing)
    let (means, stds) = normalize_features(&mut training_data);

    for sample in all_data.iter_mut() {
        for (i, x) in sample.features.iter_mut().enumerate() {
            if stds[i] > 1e-9 {
                *x = (*x - means[i]) / stds[i];
            } else {
                *x = 0.0;
            }
        }
    }

    // 4) Train (same as existing)
    let n_features = training_data[0].features.len();
    let mut model = LogisticRegression::new(n_features);

    log::info!(
        "Nokoi DF: Training on {} pos, {} neg samples...",
        training_data.iter().filter(|d| d.label == 1.0).count(),
        training_data.iter().filter(|d| d.label == 0.0).count()
    );

    model.train_cv(&training_data, &config);

    // 5) Predict on all
    let probabilities: Vec<f64> = all_data
        .iter()
        .map(|sample| model.predict(&sample.features))
        .collect();

    if probabilities.len() != features.len() {
        log::warn!(
            "Nokoi: output length mismatch ({} vs {}). Failing closed.",
            probabilities.len(),
            features.len()
        );
        return None;
    }
    if probabilities.iter().any(|p| !p.is_finite()) {
        log::warn!("Nokoi: output contains non-finite probabilities. Failing closed.");
        return None;
    }

    Some(probabilities)
}

struct NokoiCrossfitTraining {
    prob_target_all: Vec<f64>,
    null_scores_oof: Vec<f64>,
    final_model: LogisticRegression,
    normalization: NokoiNormalization,
    selected_l1_lambda: f64,
    fold_sizes: Vec<usize>,
    positive_training_count: usize,
    negative_training_count: usize,
}

fn train_nokoi_crossfit(
    features: &[DfFeature],
    config: &NokoiConfig,
    min_null_rank: u32,
    max_null_rank: u32,
    k_folds: usize,
    is_positive: impl Fn(&DfFeature) -> bool,
    null_indices: &[usize],
) -> Option<NokoiCrossfitTraining> {
    // ---- 0) Fast fallbacks for empty/low data ----
    if features.is_empty() {
        return None;
    }

    // Build positives + rank-window negatives (indices)
    let mut positives_idx: Vec<usize> = Vec::new();
    let mut negatives_idx: Vec<usize> = Vec::new();
    for (i, f) in features.iter().enumerate() {
        if is_positive(f) {
            positives_idx.push(i);
        } else {
            let r = f.core.rank as u32;
            if r >= min_null_rank && r <= max_null_rank {
                negatives_idx.push(i);
            }
        }
    }

    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "Nokoi DEBUG rescore_df_crossfit: min_null_rank={} max_null_rank={} k_folds={} positives_idx={} negatives_idx={} null_indices={}",
            min_null_rank,
            max_null_rank,
            k_folds,
            positives_idx.len(),
            negatives_idx.len(),
            null_indices.len()
        );
    }

    if positives_idx.len() < 50 || negatives_idx.len() < 50 {
        log::warn!(
            "Nokoi DF crossfit: insufficient data (pos={} neg={}) - disabling Nokoi (fail-closed)",
            positives_idx.len(),
            negatives_idx.len()
        );
        return None;
    }

    // ---- 1) Cross-fit null candidate set: intersection of provided null_indices and rank-window negatives ----
    let mut neg_mask = vec![false; features.len()];
    for &i in &negatives_idx {
        neg_mask[i] = true;
    }

    let mut null_cand: Vec<usize> = Vec::new();
    for &j in null_indices {
        if j < neg_mask.len() && neg_mask[j] {
            null_cand.push(j);
        }
    }

    if null_cand.len() < 50 {
        log::warn!(
            "Nokoi DF crossfit: too few null candidates after intersect ({} < 50) - disabling Nokoi (fail-closed)",
            null_cand.len()
        );
        return None;
    }

    // Deterministic shuffle for folds
    let k = k_folds.max(2).min(null_cand.len()); // cannot have more folds than items
    let mut rng = StdRng::seed_from_u64(NOKOI_CROSSFIT_SEED);
    null_cand.shuffle(&mut rng);

    // Fold assignment by contiguous blocks
    let fold_size = (null_cand.len() + k - 1) / k; // ceil
    let mut folds: Vec<&[usize]> = Vec::with_capacity(k);
    for i in 0..k {
        let start = i * fold_size;
        if start >= null_cand.len() {
            break;
        }
        let end = ((i + 1) * fold_size).min(null_cand.len());
        folds.push(&null_cand[start..end]);
    }

    // ---- 2) Helper: train a logistic model given pos indices + neg indices; return (model, means, stds) ----
    let train_one = |pos_idx: &[usize],
                     neg_idx: &[usize],
                     seed: u64|
     -> Option<(LogisticRegression, NokoiNormalization, f64)> {
        // Build labeled PsmData pools
        let mut pos: Vec<PsmData> = Vec::with_capacity(pos_idx.len());
        let mut neg: Vec<PsmData> = Vec::with_capacity(neg_idx.len());

        for &i in pos_idx {
            let feat_vec = extract_features(&features[i].core);
            pos.push(PsmData {
                features: feat_vec,
                label: 1.0,
                original_idx: i,
            });
        }
        for &i in neg_idx {
            let feat_vec = extract_features(&features[i].core);
            neg.push(PsmData {
                features: feat_vec,
                label: 0.0,
                original_idx: i,
            });
        }

        if pos.len() < 50 || neg.len() < 50 {
            return None;
        }

        // Deterministic balanced sampling
        let mut rng = StdRng::seed_from_u64(seed);
        pos.shuffle(&mut rng);
        neg.shuffle(&mut rng);

        let sample_size = pos.len().min(neg.len()).min(10_000);
        if sample_size == 0 {
            return None;
        }

        let mut training_data = Vec::with_capacity(sample_size * 2);
        training_data.extend_from_slice(&pos[0..sample_size]);
        training_data.extend_from_slice(&neg[0..sample_size]);
        training_data.shuffle(&mut rng);

        // Normalize (in-place) and capture means/stds
        let normalization = normalize_features_portable(&mut training_data);

        let n_features = training_data[0].features.len();
        let mut model = LogisticRegression::new(n_features);
        let selected_lambda = model.train_cv(&training_data, config);

        Some((model, normalization, selected_lambda))
    };

    // ---- 3) Cross-fit: out-of-fold predictions for null candidates ----
    let mut null_scores_oof: Vec<f64> = Vec::with_capacity(null_cand.len());

    for (fold_i, heldout) in folds.iter().enumerate() {
        // Training negatives = null_cand excluding heldout
        let mut train_neg: Vec<usize> =
            Vec::with_capacity(null_cand.len().saturating_sub(heldout.len()));
        // mark heldout for fast exclusion
        let mut heldout_flag = vec![false; features.len()];
        for &j in *heldout {
            heldout_flag[j] = true;
        }
        for &j in &null_cand {
            if !heldout_flag[j] {
                train_neg.push(j);
            }
        }

        if log::log_enabled!(log::Level::Debug) {
            let overlap = train_neg.iter().filter(|&&x| heldout_flag[x]).count();
            log::debug!(
                "Nokoi DEBUG crossfit fold {}/{}: heldout={} train_neg={} overlap_check={}",
                fold_i + 1,
                folds.len(),
                heldout.len(),
                train_neg.len(),
                overlap
            );
        }

        let seed = NOKOI_CROSSFIT_SEED ^ (fold_i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (model, normalization, _) = match train_one(&positives_idx, &train_neg, seed) {
            Some(x) => x,
            None => {
                // If a fold cannot train (rare), skip OOF and fail closed later.
                log::warn!(
                    "Nokoi DF crossfit: fold {} could not train; skipping heldout",
                    fold_i
                );
                continue;
            }
        };

        // Predict on heldout (OOF) only
        for &j in *heldout {
            let mut x = extract_features(&features[j].core);
            if !normalization.apply(&mut x) {
                return None;
            }
            let p = model.predict(&x);
            null_scores_oof.push(p.clamp(0.0, 1.0));
        }
    }

    // If cross-fit produced too few null scores, fail closed.
    if null_scores_oof.len() < 50 {
        log::warn!(
            "Nokoi DF crossfit: insufficient OOF null scores ({} < 50) - failing closed",
            null_scores_oof.len()
        );
        return None;
    }

    // ---- 4) Final model for prob_target_all: train on all positives + ALL rank-window negatives ----
    let seed_final = NOKOI_CROSSFIT_SEED ^ 0xF11A_1EED_1234_5678u64;
    let (model, normalization, selected_l1_lambda) =
        match train_one(&positives_idx, &negatives_idx, seed_final) {
            Some(x) => x,
            None => {
                log::warn!("Nokoi DF crossfit: final model could not train - failing closed");
                return None;
            }
        };

    let mut prob_target_all: Vec<f64> = Vec::with_capacity(features.len());
    for f in features {
        let mut x = extract_features(&f.core);
        if !normalization.apply(&mut x) {
            return None;
        }
        prob_target_all.push(model.predict(&x).clamp(0.0, 1.0));
    }

    if prob_target_all.len() != features.len() {
        log::warn!("Nokoi DF crossfit: output length mismatch. Failing closed.");
        return None;
    }
    if prob_target_all.iter().any(|p| !p.is_finite()) {
        log::warn!("Nokoi DF crossfit: output contains non-finite probabilities. Failing closed.");
        return None;
    }
    if null_scores_oof.is_empty() {
        log::warn!("Nokoi DF crossfit: OOF null scores array is empty. Failing closed.");
        return None;
    }
    if null_scores_oof.iter().any(|p| !p.is_finite()) {
        log::warn!("Nokoi DF crossfit: OOF null scores contain non-finite values. Failing closed.");
        return None;
    }

    Some(NokoiCrossfitTraining {
        prob_target_all,
        null_scores_oof,
        final_model: model,
        normalization,
        selected_l1_lambda,
        fold_sizes: folds.iter().map(|fold| fold.len()).collect(),
        positive_training_count: positives_idx.len(),
        negative_training_count: negatives_idx.len(),
    })
}

/// Backward-compatible cross-fit scores without exposing the portable model.
pub fn rescore_df_crossfit(
    features: &[DfFeature],
    config: &NokoiConfig,
    min_null_rank: u32,
    max_null_rank: u32,
    k_folds: usize,
    is_positive: impl Fn(&DfFeature) -> bool,
    null_indices: &[usize],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let fitted = train_nokoi_crossfit(
        features,
        config,
        min_null_rank,
        max_null_rank,
        k_folds,
        is_positive,
        null_indices,
    )?;
    Some((fitted.prob_target_all, fitted.null_scores_oof))
}

/// Convert model scores to upper-tail empirical p-values using the null-score distribution
/// from rank > 1 PSMs, with a +1 smoothing correction.
pub fn calc_empirical_p_values(features: &[FeatureCore], probs: &[f64]) -> Vec<f64> {
    let mut neg_probs: Vec<f64> = features
        .iter()
        .zip(probs)
        .filter(|(f, _)| f.rank > 1)
        .map(|(_, &p)| p)
        .collect();

    neg_probs.sort_by(|a, b| a.total_cmp(b));
    let n_neg = neg_probs.len() as f64;

    if n_neg < 10.0 {
        return vec![1.0; probs.len()];
    }

    probs
        .iter()
        .map(|&p| {
            // Find how many negatives have a probability score >= p
            let idx = neg_probs.partition_point(|&x| x < p);
            let count_ge = (neg_probs.len() - idx) as f64;
            // Conservative p-value: (count_ge + 1) / (N + 1)
            (count_ge + 1.0) / (n_neg + 1.0)
        })
        .collect()
}

/// Convert model scores to upper-tail empirical p-values using an externally
/// supplied null-score distribution.
///
/// For decoy-free Nokoi, this should be the out-of-fold null distribution
/// produced by `rescore_df_crossfit()`. Using the cross-fitted null avoids
/// circular in-sample null calibration and respects the caller's configured
/// Nokoi null-rank window.
pub fn calc_empirical_p_values_from_null_scores(
    prob_target_all: &[f64],
    null_scores_oof: &[f64],
) -> Vec<f64> {
    let mut null_scores: Vec<f64> = null_scores_oof
        .iter()
        .copied()
        .filter(|p| p.is_finite())
        .map(|p| p.clamp(0.0, 1.0))
        .collect();

    null_scores.sort_by(|a, b| a.total_cmp(b));

    let n_null = null_scores.len();

    if n_null < 10 {
        log::warn!(
            "Nokoi DF: too few cross-fitted null scores for empirical calibration ({} < 10); returning p=1.",
            n_null
        );
        return vec![1.0; prob_target_all.len()];
    }

    prob_target_all
        .iter()
        .copied()
        .map(|p| {
            let p = p.clamp(0.0, 1.0);

            // Upper-tail null survival: P_null(score >= observed score).
            let idx = null_scores.partition_point(|&x| x < p);
            let count_ge = n_null - idx;

            // Conservative +1 smoothing.
            ((count_ge as f64) + 1.0) / ((n_null as f64) + 1.0)
        })
        .map(|p| p.clamp(1e-300, 1.0))
        .collect()
}

fn estimate_pi0_for_nokoi_p_values(p_values: &[f64]) -> f64 {
    let vals: Vec<f64> = p_values
        .iter()
        .copied()
        .filter(|p| p.is_finite())
        .map(|p| p.clamp(0.0, 1.0))
        .collect();

    let n = vals.len();
    if n < 20 {
        return 1.0;
    }

    let lambdas = [0.50, 0.60, 0.70, 0.80, 0.90];

    let mut estimates: Vec<f64> = lambdas
        .iter()
        .filter_map(|&lambda| {
            let denom = (n as f64) * (1.0 - lambda);
            if denom <= 0.0 {
                return None;
            }

            let count = vals.iter().filter(|&&p| p >= lambda).count() as f64;
            Some((count / denom).clamp(0.0, 1.0))
        })
        .collect();

    if estimates.is_empty() {
        return 1.0;
    }

    estimates.sort_by(|a, b| a.total_cmp(b));
    estimates[estimates.len() / 2].clamp(0.0, 1.0)
}

fn nokoi_grenander_pep_from_p_values(p_values: &[f64], pi0: f64) -> Vec<f64> {
    const EPS: f64 = 1e-300;

    let n = p_values.len();
    if n == 0 {
        return Vec::new();
    }

    let mut pairs: Vec<(f64, usize)> = p_values
        .iter()
        .copied()
        .enumerate()
        .map(|(i, p)| (p.clamp(EPS, 1.0), i))
        .collect();

    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    #[derive(Clone, Debug)]
    struct Block {
        start_p: f64,
        end_p: f64,
        count: usize,
    }

    impl Block {
        fn width(&self) -> f64 {
            (self.end_p - self.start_p).max(EPS)
        }

        fn density(&self, n: usize) -> f64 {
            (self.count as f64) / ((n as f64) * self.width())
        }
    }

    let mut blocks: Vec<Block> = Vec::new();
    let mut prev = 0.0f64;

    for &(p, _) in pairs.iter() {
        let end = p.max(prev + EPS).min(1.0);

        blocks.push(Block {
            start_p: prev,
            end_p: end,
            count: 1,
        });

        prev = end;

        while blocks.len() >= 2 {
            let m = blocks.len();

            let d_prev = blocks[m - 2].density(n);
            let d_last = blocks[m - 1].density(n);

            if d_prev >= d_last {
                break;
            }

            let last = blocks.pop().unwrap();
            let prev_block = blocks.pop().unwrap();

            blocks.push(Block {
                start_p: prev_block.start_p,
                end_p: last.end_p,
                count: prev_block.count + last.count,
            });
        }
    }

    if prev < 1.0 {
        blocks.push(Block {
            start_p: prev,
            end_p: 1.0,
            count: 0,
        });

        while blocks.len() >= 2 {
            let m = blocks.len();

            let d_prev = blocks[m - 2].density(n);
            let d_last = blocks[m - 1].density(n);

            if d_prev >= d_last {
                break;
            }

            let last = blocks.pop().unwrap();
            let prev_block = blocks.pop().unwrap();

            blocks.push(Block {
                start_p: prev_block.start_p,
                end_p: last.end_p,
                count: prev_block.count + last.count,
            });
        }
    }

    let mut out = vec![1.0f64; n];

    let mut block_idx = 0usize;
    for &(p, original_idx) in pairs.iter() {
        while block_idx + 1 < blocks.len() && p > blocks[block_idx].end_p {
            block_idx += 1;
        }

        let density = blocks[block_idx].density(n).max(EPS);
        let pep = (pi0 / density).clamp(EPS, 1.0);

        out[original_idx] = pep;
    }

    out
}

pub fn build_nokoi_evidence_from_crossfit_null(
    prob_target_all: &[f64],
    null_scores_oof: &[f64],
) -> NokoiEvidence {
    let p_values: Vec<f64> =
        calc_empirical_p_values_from_null_scores(prob_target_all, null_scores_oof)
            .into_iter()
            .map(|p| p.clamp(0.0, 1.0).max(1e-300))
            .collect();

    let pi0 = estimate_pi0_for_nokoi_p_values(&p_values);

    let peps: Vec<f64> = nokoi_grenander_pep_from_p_values(&p_values, pi0)
        .into_iter()
        .map(|pep| pep.clamp(0.0, 1.0).max(1e-300))
        .collect();

    let n = p_values.len();
    if n > 0 {
        let n_p_one = p_values.iter().filter(|&&p| p >= 0.999999).count();
        let n_pep_one = peps.iter().filter(|&&p| p >= 0.999999).count();

        let mut ps = p_values.clone();
        let mut es = peps.clone();
        ps.sort_by(|a, b| a.total_cmp(b));
        es.sort_by(|a, b| a.total_cmp(b));

        let q = |xs: &[f64], frac: f64| -> f64 {
            let idx = (frac.clamp(0.0, 1.0) * ((xs.len() - 1) as f64)).round() as usize;
            xs[idx.min(xs.len() - 1)]
        };

        log::info!(
            "Nokoi DF evidence diagnostics: n={} null_oof={} pi0={:.4} p_one={} pep_one={} p_q=[{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}] pep_q=[{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}]",
            n,
            null_scores_oof.len(),
            pi0,
            n_p_one,
            n_pep_one,
            q(&ps, 0.00),
            q(&ps, 0.10),
            q(&ps, 0.50),
            q(&ps, 0.90),
            q(&ps, 1.00),
            q(&es, 0.00),
            q(&es, 0.10),
            q(&es, 0.50),
            q(&es, 0.90),
            q(&es, 1.00)
        );
    }

    NokoiEvidence { p_values, peps }
}

fn freeze_pep_calibration(p_values: &[f64], peps: &[f64]) -> Vec<NokoiCalibrationPoint> {
    let mut pairs = p_values
        .iter()
        .copied()
        .zip(peps.iter().copied())
        .filter(|(p, pep)| p.is_finite() && pep.is_finite())
        .map(|(p, pep)| (p.clamp(1e-300, 1.0), pep.clamp(1e-300, 1.0)))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut points: Vec<NokoiCalibrationPoint> = Vec::new();
    for (p_value, pep) in pairs {
        if let Some(last) = points.last_mut().filter(|last| last.p_value == p_value) {
            last.pep = last.pep.max(pep);
        } else {
            points.push(NokoiCalibrationPoint { p_value, pep });
        }
    }
    // Local FDR should not improve as the p-value worsens. Enforce that
    // monotonicity conservatively before persisting the calibration curve.
    let mut running = 1e-300_f64;
    for point in &mut points {
        running = running.max(point.pep);
        point.pep = running;
    }
    points
}

fn apply_frozen_pep_calibration(
    p_values: &[f64],
    points: &[NokoiCalibrationPoint],
) -> Option<Vec<f64>> {
    if points.len() < 2
        || points.iter().any(|point| {
            !point.p_value.is_finite()
                || !point.pep.is_finite()
                || point.p_value <= 0.0
                || point.p_value > 1.0
                || point.pep <= 0.0
                || point.pep > 1.0
        })
    {
        return None;
    }
    let mut output = Vec::with_capacity(p_values.len());
    for &p_value in p_values {
        let p = p_value.clamp(1e-300, 1.0);
        let upper = points.partition_point(|point| point.p_value < p);
        let pep = if upper == 0 {
            points[0].pep
        } else if upper >= points.len() {
            points.last()?.pep
        } else {
            let left = &points[upper - 1];
            let right = &points[upper];
            let width = right.p_value - left.p_value;
            if width <= 1e-300 {
                left.pep.max(right.pep)
            } else {
                let weight = ((p - left.p_value) / width).clamp(0.0, 1.0);
                left.pep + weight * (right.pep - left.pep)
            }
        };
        output.push(pep.clamp(1e-300, 1.0));
    }
    Some(output)
}

pub fn fit_nokoi_artifact(
    features: &[DfFeature],
    config: &NokoiConfig,
    min_null_rank: u32,
    max_null_rank: u32,
    k_folds: usize,
    is_positive: impl Fn(&DfFeature) -> bool,
    null_indices: &[usize],
) -> Option<NokoiFitResult> {
    let fitted = train_nokoi_crossfit(
        features,
        config,
        min_null_rank,
        max_null_rank,
        k_folds,
        is_positive,
        null_indices,
    )?;
    let evidence =
        build_nokoi_evidence_from_crossfit_null(&fitted.prob_target_all, &fitted.null_scores_oof);
    let development_pi0 = estimate_pi0_for_nokoi_p_values(&evidence.p_values);
    let pep_calibration = freeze_pep_calibration(&evidence.p_values, &evidence.peps);
    if pep_calibration.len() < 2 {
        return None;
    }
    let mut reference_candidate_counts = features
        .iter()
        .filter(|feature| feature.core.rank == 1)
        .map(|feature| feature.core.lo_spectrum_candidate_count)
        .filter(|&count| count > 0)
        .collect::<Vec<_>>();
    reference_candidate_counts.sort_unstable();
    let artifact = NokoiArtifact {
        schema_version: 1,
        model_version: "sage-nokoi-crossfit-portable-v1".into(),
        feature_schema: NOKOI_FEATURE_SCHEMA
            .iter()
            .map(|name| (*name).to_string())
            .collect(),
        min_null_rank,
        max_null_rank,
        crossfit_seed: NOKOI_CROSSFIT_SEED,
        k_folds: fitted.fold_sizes.len(),
        fold_sizes: fitted.fold_sizes,
        config: config.clone(),
        selected_l1_lambda: fitted.selected_l1_lambda,
        final_model: fitted.final_model,
        normalization: fitted.normalization,
        null_scores_oof: fitted.null_scores_oof,
        development_pi0,
        pep_calibration,
        positive_training_count: fitted.positive_training_count,
        negative_training_count: fitted.negative_training_count,
        reference_candidate_counts,
    };
    Some(NokoiFitResult { evidence, artifact })
}

pub fn apply_nokoi_artifact(
    features: &[DfFeature],
    artifact: &NokoiArtifact,
) -> Option<NokoiEvidence> {
    if artifact.schema_version != 1
        || artifact.feature_schema
            != NOKOI_FEATURE_SCHEMA
                .iter()
                .map(|name| (*name).to_string())
                .collect::<Vec<_>>()
        || artifact.final_model.weights.len() != NOKOI_FEATURE_SCHEMA.len()
        || artifact.normalization.medians.len() != NOKOI_FEATURE_SCHEMA.len()
        || artifact.normalization.means.len() != NOKOI_FEATURE_SCHEMA.len()
        || artifact.normalization.stds.len() != NOKOI_FEATURE_SCHEMA.len()
        || artifact.null_scores_oof.len() < 50
        || artifact.pep_calibration.len() < 2
        || artifact.k_folds < 2
        || artifact.fold_sizes.len() != artifact.k_folds
    {
        log::error!("Nokoi frozen artifact is incomplete or incompatible");
        return None;
    }
    let mut probabilities = Vec::with_capacity(features.len());
    for feature in features {
        let mut values = extract_features(&feature.core);
        if !artifact.normalization.apply(&mut values) {
            return None;
        }
        let probability = artifact.final_model.predict(&values);
        if !probability.is_finite() {
            return None;
        }
        probabilities.push(probability.clamp(0.0, 1.0));
    }
    let p_values =
        calc_empirical_p_values_from_null_scores(&probabilities, &artifact.null_scores_oof);
    let peps = apply_frozen_pep_calibration(&p_values, &artifact.pep_calibration)?;
    Some(NokoiEvidence { p_values, peps })
}

#[cfg(test)]
mod portable_tests {
    use super::*;

    #[test]
    fn portable_normalization_imputes_nonfinite_values() {
        let mut data = vec![
            PsmData {
                features: vec![1.0, f64::NAN],
                label: 0.0,
                original_idx: 0,
            },
            PsmData {
                features: vec![3.0, 4.0],
                label: 1.0,
                original_idx: 1,
            },
            PsmData {
                features: vec![5.0, 6.0],
                label: 1.0,
                original_idx: 2,
            },
        ];
        let normalization = normalize_features_portable(&mut data);
        assert_eq!(normalization.medians, vec![3.0, 5.0]);
        assert!(data
            .iter()
            .flat_map(|row| &row.features)
            .all(|value| value.is_finite()));
        let mut new_values = vec![f64::NAN, 5.0];
        assert!(normalization.apply(&mut new_values));
        assert!(new_values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn frozen_pep_curve_is_monotone_and_reusable() {
        let points = freeze_pep_calibration(&[0.01, 0.1, 0.5, 0.9], &[0.02, 0.08, 0.07, 0.8]);
        assert!(points.windows(2).all(|pair| pair[0].pep <= pair[1].pep));
        let applied = apply_frozen_pep_calibration(&[0.05, 0.2, 0.8], &points).unwrap();
        assert!(applied.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(applied.iter().all(|value| (0.0..=1.0).contains(value)));
    }
}
