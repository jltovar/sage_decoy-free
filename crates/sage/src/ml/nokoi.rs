use crate::scoring::{DfFeature, FeatureCore};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Configuration for Nokoi Rescoring 2.0
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NokoiConfig {
    pub enabled: bool,
    pub train_fdr: f32,
    // Hyperparameters
    pub learning_rate: f64,
    pub epochs: usize,
    pub l1_lambda: f64,  // L1 (Lasso) Regularization
    pub patience: usize, // Early stopping patience
}

impl Default for NokoiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            train_fdr: 0.01,
            learning_rate: 0.1,
            epochs: 250,      // Reduced default because FISTA converges faster
            l1_lambda: 0.001, // L1 usually requires smaller lambda than L2
            patience: 10,     // Stop if no improvement for 10 epochs
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
/// Implements the "High-Performance Feature Set" (~12 features)
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
#[derive(Clone, Debug)]
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
    pub fn train_cv(&mut self, data: &[PsmData], config: &NokoiConfig) {
        let n_samples = data.len();
        if n_samples == 0 {
            return;
        }

        // --- Log-Spaced Lambda Grid ---
        // Search orders of magnitude: [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        // This is much better for finding the "sparsity sweet spot"
        // let lambdas: Vec<f64> = (0..5).map(|i| 1e-4 * 10f64.powi(i)).collect();

        // Paper uses 2^-10 to 2^5.
        // 2^-10 = 0.00097, 2^5 = 32.0.
        let lambdas: Vec<f64> = (-10..=5).map(|i| 2.0_f64.powi(i)).collect();

        // Use 3-fold CV for speed, or 5 if data allows
        let k_folds = if n_samples >= 100 { 5 } else { 3 };

        let mut best_lambda = 0.001;
        let mut best_loss = f64::INFINITY;

        log::info!(
            "Nokoi: Tuning L1 Lambda via {}-fold CV with Early Stopping...",
            k_folds
        );

        for &lambda in &lambdas {
            let mut total_loss = 0.0;

            for fold in 0..k_folds {
                // Split Data for this Fold
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

                total_loss += final_loss;
            }

            let avg_loss = total_loss / k_folds as f64;

            if avg_loss < best_loss {
                best_loss = avg_loss;
                best_lambda = lambda;
            }
        }

        log::info!(
            "Nokoi: Best L1 Lambda = {:.5} (Loss: {:.4})",
            best_lambda,
            best_loss
        );

        // 3. Final Fit on All Data
        let split_idx = (n_samples as f64 * 0.8) as usize;
        let (train, val) = data.split_at(split_idx);
        self.train_with_early_stopping(train, val, config, best_lambda);

        // Inspect weights to see what was zeroed out
        let nonzero_features = self.weights.iter().filter(|&&w| w.abs() > 1e-6).count();
        log::info!(
            "Nokoi: Final model selected {}/{} features.",
            nonzero_features,
            self.weights.len()
        );
    }
}

/// Robust Preprocessing: Median Imputation + Z-Score Standardization
pub fn normalize_features(data: &mut [PsmData]) -> (Vec<f64>, Vec<f64>) {
    let n_samples = data.len();
    if n_samples == 0 {
        return (Vec::new(), Vec::new());
    }

    let n_features = data[0].features.len();
    let n = n_samples as f64;

    let mut means = vec![0.0; n_features];
    let mut stds = vec![0.0; n_features];

    // 1. Median Imputation
    let mut medians = vec![0.0; n_features];
    for j in 0..n_features {
        let mut vals: Vec<f64> = data
            .iter()
            .map(|s| s.features[j])
            .filter(|v| v.is_finite())
            .collect();

        if vals.is_empty() {
            medians[j] = 0.0;
        } else {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let mid = vals.len() / 2;
            medians[j] = if vals.len() % 2 == 1 {
                vals[mid]
            } else {
                (vals[mid - 1] + vals[mid]) / 2.0
            };
        }
    }

    for sample in data.iter_mut() {
        for j in 0..n_features {
            if !sample.features[j].is_finite() {
                sample.features[j] = medians[j];
            }
        }
    }

    // 2. Calculate Mean
    for sample in data.iter() {
        for (i, x) in sample.features.iter().enumerate() {
            means[i] += *x;
        }
    }
    for m in means.iter_mut() {
        *m /= n;
    }

    // 3. Calculate Std Dev
    for sample in data.iter() {
        for (i, x) in sample.features.iter().enumerate() {
            stds[i] += (x - means[i]).powi(2);
        }
    }
    for (_i, s) in stds.iter_mut().enumerate() {
        *s = (*s / n).sqrt();
        if !s.is_finite() || *s < 1e-9 {
            *s = 1.0;
        }
    }

    // 4. Apply Z-Score
    for sample in data.iter_mut() {
        for (i, x) in sample.features.iter_mut().enumerate() {
            *x = (*x - means[i]) / stds[i];
        }
    }

    (means, stds)
}

/// Main entry point: Train model and return probabilities for all features
///
/// `is_positive` determines if a feature is considered a high-confidence target for training.
pub fn rescore(
    features: &[FeatureCore],
    train_fdr: f32,
    is_positive: impl Fn(&FeatureCore) -> bool,
) -> Option<Vec<f64>> {
    // Nokoi 2.0 Config
    let config = NokoiConfig {
        enabled: true,
        train_fdr,
        learning_rate: 0.1,
        epochs: 500,
        l1_lambda: 0.005, // Initial guess, but train_cv will explore better grid
        patience: 15,
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
        // Paper criteria for negative class: "rank2 PSMs" (or lower ranks)
        else if f.rank > 1 {
            negatives.push(psm.clone());
        }

        all_data.push(psm); // We predict on everything later
    }

    // --- Graceful Fallback for Low Data (<50) ---
    // If we have fewer than 50 confident PSMs, ML training is unstable.
    // Instead of failing, we return "probabilities" based on min-max scaled Hyperscore.
    let confident_count = positives.len();
    if confident_count < 50 {
        log::warn!(
            "Nokoi: Too few positives ({} < 50) - falling back to normalized hyperscore",
            confident_count
        );

        if features.is_empty() {
            return None;
        }

        let min_hs = features
            .iter()
            .map(|f| f.hyperscore as f64)
            .fold(f64::INFINITY, f64::min);
        let max_hs = features
            .iter()
            .map(|f| f.hyperscore as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_hs - min_hs;

        if range == 0.0 {
            // Edge case: all scores identical
            return Some(vec![0.5; features.len()]);
        }

        let probabilities: Vec<f64> = features
            .iter()
            .map(|f| (f.hyperscore as f64 - min_hs) / range)
            .collect();

        return Some(probabilities);
    } else if confident_count < 100 {
        // Between 50 and 100, we proceed but warn
        log::warn!(
            "Nokoi: Low training data ({}), model may vary.",
            confident_count
        );
    }

    // 1. Feature Extraction
    // --- Balanced Sampling  ---
    // The paper explicitly creates a "balanced training data set" of 10k pos / 10k neg.
    // Without this, the model will just predict "0.0" for everything because negatives
    // vastly outnumber positives.

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
    // Nokoi 2.0 Config (same defaults as `rescore`)
    let config = NokoiConfig {
        enabled: true,
        train_fdr,
        learning_rate: 0.1,
        epochs: 500,
        l1_lambda: 0.005,
        patience: 15,
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

    // --- Graceful fallback for low data (match your existing behavior) ---
    let confident_count = positives.len();
    if confident_count < 50 {
        log::warn!(
            "Nokoi DF: Too few positives ({} < 50) - falling back to normalized hyperscore",
            confident_count
        );

        if features.is_empty() {
            return None;
        }

        let min_hs = features
            .iter()
            .map(|f| f.core.hyperscore as f64)
            .fold(f64::INFINITY, f64::min);
        let max_hs = features
            .iter()
            .map(|f| f.core.hyperscore as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_hs - min_hs;

        if range == 0.0 {
            return Some(vec![0.5; features.len()]);
        }

        let probabilities: Vec<f64> = features
            .iter()
            .map(|f| (f.core.hyperscore as f64 - min_hs) / range)
            .collect();

        return Some(probabilities);
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

    // If negatives are too few, fail closed to fallback hyperscore behavior (consistent + safe)
    if n_neg < 50 {
        log::warn!(
            "Nokoi DF: Too few negatives from rank window ({} < 50) - falling back to normalized hyperscore",
            n_neg
        );
        if features.is_empty() {
            return None;
        }
        let min_hs = features
            .iter()
            .map(|f| f.core.hyperscore as f64)
            .fold(f64::INFINITY, f64::min);
        let max_hs = features
            .iter()
            .map(|f| f.core.hyperscore as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_hs - min_hs;
        if range == 0.0 {
            return Some(vec![0.5; features.len()]);
        }
        let probabilities: Vec<f64> = features
            .iter()
            .map(|f| (f.core.hyperscore as f64 - min_hs) / range)
            .collect();
        return Some(probabilities);
    }

    let sample_size = n_pos.min(n_neg).min(10_000);

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

    Some(probabilities)
}

/// Convert Probabilities to P-values using ECDF of Negatives (Rank > 1)
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
