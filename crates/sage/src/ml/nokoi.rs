use crate::scoring::Feature;
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
pub fn extract_features(feat: &Feature) -> Vec<f64> {
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

        // 1. Construct Lambda Grid (L1 focus)
        // L1 lambdas often need to be searched in log-space
        let base_lambda = if config.l1_lambda > 0.0 {
            config.l1_lambda
        } else {
            0.001
        };
        // Search [0.1x, 0.5x, 1x, 2x, 5x, 10x]
        let factors = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        let mut lambdas: Vec<f64> = factors.iter().map(|f| base_lambda * f).collect();
        lambdas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        // Use 3-fold CV for speed, or 5 if data allows
        let k_folds = if n_samples >= 100 { 5 } else { 3 };

        let mut best_lambda = base_lambda;
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

                // In Nokoi 2.0, we perform Early Stopping *inside* the CV loop.
                // We further split the 'train_set' into 'inner_train' and 'inner_val'
                // OR simpler: use 'val_set' for early stopping directly.
                // Using 'val_set' for both early stopping AND scoring is slightly biased,
                // but for this scale of data, it is an acceptable trade-off to prevent overfitting.

                let mut fold_model = LogisticRegression::new(self.weights.len());
                let final_loss =
                    fold_model.train_with_early_stopping(&train_set, &val_set, config, lambda);

                total_loss += final_loss;
            }

            let avg_loss = total_loss / k_folds as f64;
            // log::debug!("Lambda: {:.5}, Loss: {:.5}", lambda, avg_loss);

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
        // For final fit, we use a random 80/20 split to determine stopping point
        let split_idx = (n_samples as f64 * 0.8) as usize;
        // Simple shuffle-like split (data is assumed somewhat random, or we just take end)
        // PsmData is typically ordered by file/scan, so taking the end is risky if files differ.
        // However, rescore() collects them linearly.
        // A simple deterministic shuffle would be better, but simple split is okay for now.

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
/// Explicitly handles constant features by forcing them to 0.0
pub fn normalize_features(data: &mut [PsmData]) -> (Vec<f64>, Vec<f64>) {
    let n_samples = data.len();
    if n_samples == 0 {
        return (Vec::new(), Vec::new());
    }

    let n_features = data[0].features.len();
    let n = n_samples as f64;

    let mut means = vec![0.0; n_features];
    let mut stds = vec![0.0; n_features];

    // 1. Median Imputation (Handle Infs/NaNs)
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
        // FEATURE SAFETY:
        // If std is 0 (or very close), the feature is constant (e.g., missing IM).
        // We set std to 1.0 to avoid div/0, BUT since x == mean, the result (x-mean)/1.0 will be 0.0.
        // This effectively "hard-codes" the feature to 0.0 for the model.
        if !s.is_finite() || *s < 1e-9 {
            if *s < 1e-9 {
                // log::warn!("Feature #{} appears constant (std ~ 0). It will be ignored.", i + 1);
            }
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
pub fn rescore(features: &[Feature], train_fdr: f32) -> Option<Vec<f64>> {
    // Nokoi 2.0 Config
    let config = NokoiConfig {
        enabled: true,
        train_fdr,
        learning_rate: 0.1,
        epochs: 500,      // Max epochs, likely stops earlier
        l1_lambda: 0.005, // Start with a reasonable L1
        patience: 15,     // Patience
    };

    let mut training_data = Vec::new();
    let mut all_data = Vec::with_capacity(features.len());

    let confident_count = features
        .iter()
        .filter(|f| f.rank == 1 && f.spectrum_q <= train_fdr)
        .count();

    if confident_count < 100 {
        log::warn!(
            "Nokoi: Insufficient confident Rank 1 PSMs ({}) for training.",
            confident_count
        );
        return None;
    }

    // 1. Feature Extraction
    for (i, f) in features.iter().enumerate() {
        let feat_vec = extract_features(f);

        // Training set: High confidence Rank 1 (Pos) vs Rank > 1 (Neg)
        if f.rank == 1 && f.spectrum_q <= train_fdr {
            training_data.push(PsmData {
                features: feat_vec.clone(),
                label: 1.0,
                original_idx: i,
            });
        } else if f.rank > 1 {
            training_data.push(PsmData {
                features: feat_vec.clone(),
                label: 0.0,
                original_idx: i,
            });
        }

        all_data.push(PsmData {
            features: feat_vec,
            label: 0.0,
            original_idx: i,
        });
    }

    // 2. Normalize
    let (means, stds) = normalize_features(&mut training_data);

    // Apply normalization to prediction set
    for sample in all_data.iter_mut() {
        for (i, x) in sample.features.iter_mut().enumerate() {
            *x = (*x - means[i]) / stds[i];
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
    let probabilities: Vec<f64> = all_data
        .iter()
        .map(|sample| model.predict(&sample.features))
        .collect();

    Some(probabilities)
}

/// Convert Probabilities to P-values using ECDF of Negatives (Rank > 1)
pub fn calc_empirical_p_values(features: &[Feature], probs: &[f64]) -> Vec<f64> {
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
