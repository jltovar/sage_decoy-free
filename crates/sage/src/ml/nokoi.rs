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
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

mod exact_f64 {
    use serde::de::{Error, Visitor};
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{:016x}", value.to_bits()))
    }

    struct ExactVisitor;
    impl<'de> Visitor<'de> for ExactVisitor {
        type Value = f64;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a hexadecimal f64 bit string or legacy JSON number")
        }
        fn visit_str<E: Error>(self, value: &str) -> Result<f64, E> {
            u64::from_str_radix(value, 16)
                .map(f64::from_bits)
                .map_err(E::custom)
        }
        fn visit_f64<E: Error>(self, value: f64) -> Result<f64, E> {
            Ok(value)
        }
        fn visit_i64<E: Error>(self, value: i64) -> Result<f64, E> {
            Ok(value as f64)
        }
        fn visit_u64<E: Error>(self, value: u64) -> Result<f64, E> {
            Ok(value as f64)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        deserializer.deserialize_any(ExactVisitor)
    }
}

mod exact_f32 {
    use serde::de::{Error, Visitor};
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{:08x}", value.to_bits()))
    }

    struct ExactVisitor;
    impl<'de> Visitor<'de> for ExactVisitor {
        type Value = f32;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a hexadecimal f32 bit string or legacy JSON number")
        }
        fn visit_str<E: Error>(self, value: &str) -> Result<f32, E> {
            u32::from_str_radix(value, 16)
                .map(f32::from_bits)
                .map_err(E::custom)
        }
        fn visit_f64<E: Error>(self, value: f64) -> Result<f32, E> {
            Ok(value as f32)
        }
        fn visit_i64<E: Error>(self, value: i64) -> Result<f32, E> {
            Ok(value as f32)
        }
        fn visit_u64<E: Error>(self, value: u64) -> Result<f32, E> {
            Ok(value as f32)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
        deserializer.deserialize_any(ExactVisitor)
    }
}

mod exact_vec_f64 {
    use serde::de::{Error, SeqAccess, Visitor};
    use serde::ser::SerializeSeq;
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(values: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(values.len()))?;
        for value in values {
            sequence.serialize_element(&format!("{:016x}", value.to_bits()))?;
        }
        sequence.end()
    }

    struct ExactValueVisitor;
    impl<'de> Visitor<'de> for ExactValueVisitor {
        type Value = f64;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a hexadecimal f64 bit string or legacy JSON number")
        }
        fn visit_str<E: Error>(self, value: &str) -> Result<f64, E> {
            u64::from_str_radix(value, 16)
                .map(f64::from_bits)
                .map_err(E::custom)
        }
        fn visit_f64<E: Error>(self, value: f64) -> Result<f64, E> {
            Ok(value)
        }
        fn visit_i64<E: Error>(self, value: i64) -> Result<f64, E> {
            Ok(value as f64)
        }
        fn visit_u64<E: Error>(self, value: u64) -> Result<f64, E> {
            Ok(value as f64)
        }
    }

    struct ExactVecVisitor;
    impl<'de> Visitor<'de> for ExactVecVisitor {
        type Value = Vec<f64>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a sequence of exact f64 values")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Vec<f64>, A::Error> {
            let mut output = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
            while let Some(value) = sequence.next_element_seed(ExactValueSeed)? {
                output.push(value);
            }
            Ok(output)
        }
    }

    struct ExactValueSeed;
    impl<'de> serde::de::DeserializeSeed<'de> for ExactValueSeed {
        type Value = f64;
        fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<f64, D::Error> {
            deserializer.deserialize_any(ExactValueVisitor)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
        deserializer.deserialize_seq(ExactVecVisitor)
    }
}

pub const NOKOI_CROSSFIT_SEED: u64 = 0x5EED_5EED_5EED_5EED;
pub const NOKOI_ARTIFACT_SCHEMA_VERSION: u32 = 2;
pub const NOKOI_MODEL_VERSION: &str = "sage-nokoi-crossfit-portable-v2";
pub const NOKOI_FEATURE_SCHEMA_VERSION: &str = "sage-nokoi-feature-schema-v1";
pub const NOKOI_CANDIDATE_ID_SCHEMA: &str = "sage-candidate-id-v1";
pub const NOKOI_STABLE_CANDIDATE_SCHEMA: &str = "sage-nokoi-stable-candidate-v1";
pub const NOKOI_IMPLEMENTATION_IDENTITY: &str = "sage-nokoi-rust-fista-crossfit-v2";
pub const NOKOI_IMPLEMENTATION_SOURCE_SHA256: &str = env!("SAGE_NOKOI_SOURCE_SHA256");

#[derive(Clone, Debug)]
pub struct NokoiEvidence {
    pub p_values: Vec<f64>,
    pub peps: Vec<f64>,
}

/// Configuration for Nokoi Rescoring
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct NokoiConfig {
    pub enabled: bool,
    #[serde(with = "exact_f32")]
    pub train_fdr: f32,

    // Optimizer hyperparameters
    #[serde(with = "exact_f64")]
    pub learning_rate: f64,
    pub epochs: usize,
    pub patience: usize,

    // Initial/fallback single-lambda value used by direct training calls
    #[serde(with = "exact_f64")]
    pub l1_lambda: f64,

    // CV-tuned lambda grid
    #[serde(with = "exact_f64")]
    pub l1_lambda_min: f64,
    #[serde(with = "exact_f64")]
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
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct LogisticRegression {
    #[serde(with = "exact_vec_f64")]
    pub weights: Vec<f64>,
    #[serde(with = "exact_f64")]
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
    ) -> NokoiTrainingRun {
        let n_train = train_data.len() as f64;
        if n_train == 0.0 {
            return NokoiTrainingRun {
                validation_loss: f64::INFINITY,
                epochs_completed: 0,
                early_stopped: false,
            };
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
        let mut epochs_completed = 0;
        let mut early_stopped = false;

        for epoch in 0..config.epochs {
            epochs_completed = epoch + 1;
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
                early_stopped = true;
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

        NokoiTrainingRun {
            validation_loss: best_loss,
            epochs_completed,
            early_stopped,
        }
    }

    /// Train using K-fold Cross Validation to tune L1 Lambda
    pub fn train_cv(&mut self, data: &[PsmData], config: &NokoiConfig) -> f64 {
        self.train_cv_report(data, config).selected_l1_lambda
    }

    fn train_cv_report(
        &mut self,
        data: &[PsmData],
        config: &NokoiConfig,
    ) -> NokoiOptimizationState {
        let n_samples = data.len();
        if n_samples == 0 {
            return NokoiOptimizationState {
                selected_l1_lambda: config.l1_lambda,
                lambda_selection_fallback_used: true,
                convergence_rule: "empty training population".into(),
                ..NokoiOptimizationState::default()
            };
        }

        // Build the canonical log-spaced lambda grid. Strict `<` below makes
        // ties choose the first (smallest) lambda deterministically.
        let lambdas = lambda_grid(config);
        let steps = lambdas.len();
        let lambda_min = lambdas[0];
        let lambda_max = *lambdas.last().expect("nonempty lambda grid");

        let k_folds = if n_samples >= 100 { 5 } else { 3 };

        let mut lambda_evaluations = Vec::with_capacity(lambdas.len());

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
                let training =
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

                total_loss += training.validation_loss;
            }

            let avg_loss = if fold_collapsed {
                f64::INFINITY
            } else {
                total_loss / k_folds as f64
            };

            lambda_evaluations.push(NokoiLambdaEvaluation {
                l1_lambda: lambda,
                mean_validation_loss: if avg_loss.is_finite() { avg_loss } else { 0.0 },
                valid: avg_loss.is_finite(),
            });
        }

        let selected_lambda_index = selected_lambda_evaluation(&lambda_evaluations);
        let lambda_selection_fallback_used = selected_lambda_index.is_none();
        let (best_lambda, best_loss) = selected_lambda_index
            .map(|index| {
                (
                    lambda_evaluations[index].l1_lambda,
                    lambda_evaluations[index].mean_validation_loss,
                )
            })
            .unwrap_or((lambda_min, f64::INFINITY));
        if lambda_selection_fallback_used {
            log::warn!(
                "Nokoi: all CV lambdas collapsed the model; defaulting to lowest lambda {:.6e}",
                lambda_min
            );
        } else {
            log::info!(
                "Nokoi: Best L1 Lambda = {:.6e} (Loss: {:.6})",
                best_lambda,
                best_loss
            );
        }

        let split_idx = (n_samples as f64 * 0.8) as usize;
        let (train, val) = data.split_at(split_idx);
        let final_training = self.train_with_early_stopping(train, val, config, best_lambda);

        let nonzero_features = self.weights.iter().filter(|&&w| w.abs() > 1e-6).count();
        log::info!(
            "Nokoi: Final model selected {}/{} features.",
            nonzero_features,
            self.weights.len()
        );
        NokoiOptimizationState {
            lambda_evaluations,
            selected_lambda_index: selected_lambda_index.unwrap_or(0),
            selected_l1_lambda: best_lambda,
            selected_mean_validation_loss: if best_loss.is_finite() {
                best_loss
            } else {
                0.0
            },
            final_validation_loss: final_training.validation_loss,
            final_epochs_completed: final_training.epochs_completed,
            final_early_stopped: final_training.early_stopped,
            lambda_selection_fallback_used,
            convergence_rule: "deterministic FISTA; retain lowest validation-loss iterate; stop at configured patience or maximum epochs".into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct NokoiTrainingRun {
    validation_loss: f64,
    epochs_completed: usize,
    early_stopped: bool,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NokoiNormalization {
    #[serde(with = "exact_vec_f64")]
    pub medians: Vec<f64>,
    #[serde(with = "exact_vec_f64")]
    pub means: Vec<f64>,
    #[serde(with = "exact_vec_f64")]
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NokoiCalibrationPoint {
    #[serde(with = "exact_f64")]
    pub p_value: f64,
    #[serde(with = "exact_f64")]
    pub pep: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NokoiArtifactIdentity {
    pub implementation_source_identity: String,
    pub implementation_source_sha256: String,
    pub dataset_id: String,
    pub dataset_fingerprint: String,
    pub fit_search_fingerprint: String,
    pub fit_analysis_fingerprint: String,
    pub candidate_id_schema: String,
    pub stable_identity_schema: String,
    pub configuration_sha256: String,
    pub input_feature_schema_sha256: String,
    pub artifact_creation_mode: String,
    pub fit_population_sha256: String,
    pub fit_population_count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NokoiFeatureContract {
    pub schema_version: String,
    pub names: Vec<String>,
    pub dimensionality: usize,
    pub extraction_semantics: Vec<String>,
    pub missing_value_rule: String,
    pub normalization_rule: String,
    pub finite_range_expectation: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NokoiTrainingContract {
    pub positive_class_rule: String,
    #[serde(with = "exact_f64")]
    pub positive_top_fraction: f64,
    #[serde(with = "exact_f64")]
    pub positive_threshold: f64,
    pub negative_class_rule: String,
    pub null_purification_rule: String,
    #[serde(with = "exact_f64")]
    pub null_purification_factor: f64,
    pub min_null_rank: u32,
    pub max_null_rank: u32,
    pub class_balancing_rule: String,
    pub maximum_samples_per_class: usize,
    pub deterministic_seed: u64,
    pub deterministic_ordering_rule: String,
    pub fold_assignment_rule: String,
    pub positive_training_count: usize,
    pub negative_training_count: usize,
    pub positive_population_sha256: String,
    pub negative_population_sha256: String,
    pub null_candidate_population_sha256: String,
    pub candidate_count_population_sha256: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NokoiLambdaEvaluation {
    #[serde(with = "exact_f64")]
    pub l1_lambda: f64,
    #[serde(with = "exact_f64")]
    pub mean_validation_loss: f64,
    pub valid: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NokoiOptimizationState {
    pub lambda_evaluations: Vec<NokoiLambdaEvaluation>,
    pub selected_lambda_index: usize,
    #[serde(with = "exact_f64")]
    pub selected_l1_lambda: f64,
    #[serde(with = "exact_f64")]
    pub selected_mean_validation_loss: f64,
    #[serde(with = "exact_f64")]
    pub final_validation_loss: f64,
    pub final_epochs_completed: usize,
    pub final_early_stopped: bool,
    pub lambda_selection_fallback_used: bool,
    pub convergence_rule: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NokoiFoldModel {
    pub fold_index: usize,
    pub heldout_count: usize,
    pub heldout_stable_ids_sha256: String,
    pub training_negative_count: usize,
    #[serde(with = "exact_f64")]
    pub selected_l1_lambda: f64,
    pub optimization: NokoiOptimizationState,
    pub model: LogisticRegression,
    pub normalization: NokoiNormalization,
    pub fit_completed: bool,
    pub fallback_used: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NokoiGrenanderBlock {
    #[serde(with = "exact_f64")]
    pub start_p: f64,
    #[serde(with = "exact_f64")]
    pub end_p: f64,
    pub count: usize,
    #[serde(with = "exact_f64")]
    pub density: f64,
    #[serde(with = "exact_f64")]
    pub pep: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NokoiCalibrationContract {
    pub score_direction: String,
    pub empirical_p_value_rule: String,
    pub tie_handling_rule: String,
    pub interpolation_rule: String,
    pub boundary_rule: String,
    pub pi0_rule: String,
    pub pep_calibration_rule: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NokoiArtifactIntegrity {
    pub block_sha256: BTreeMap<String, String>,
    pub canonical_payload_sha256: String,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NokoiArtifactApplicationMode {
    #[default]
    ExactFitPopulation,
    SameDatasetTargetOnly,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NokoiArtifact {
    pub schema_version: u32,
    pub model_version: String,
    #[serde(default)]
    pub identity: NokoiArtifactIdentity,
    #[serde(default)]
    pub feature_contract: NokoiFeatureContract,
    pub feature_schema: Vec<String>,
    pub min_null_rank: u32,
    pub max_null_rank: u32,
    pub crossfit_seed: u64,
    pub k_folds: usize,
    pub fold_sizes: Vec<usize>,
    pub config: NokoiConfig,
    #[serde(default)]
    #[serde(with = "exact_vec_f64")]
    pub lambda_grid: Vec<f64>,
    #[serde(default)]
    pub fold_models: Vec<NokoiFoldModel>,
    #[serde(with = "exact_f64")]
    pub selected_l1_lambda: f64,
    pub final_model: LogisticRegression,
    #[serde(default)]
    pub final_optimization: NokoiOptimizationState,
    pub normalization: NokoiNormalization,
    #[serde(with = "exact_vec_f64")]
    pub null_scores_oof: Vec<f64>,
    #[serde(with = "exact_f64")]
    pub development_pi0: f64,
    #[serde(default)]
    pub calibration_contract: NokoiCalibrationContract,
    #[serde(default)]
    pub grenander_blocks: Vec<NokoiGrenanderBlock>,
    pub pep_calibration: Vec<NokoiCalibrationPoint>,
    pub positive_training_count: usize,
    pub negative_training_count: usize,
    #[serde(default)]
    pub training_contract: NokoiTrainingContract,
    #[serde(default)]
    pub training_completed: bool,
    #[serde(default)]
    pub training_fallback_used: bool,
    #[serde(default)]
    pub feature_selection_state: Vec<bool>,
    #[serde(default)]
    pub reference_candidate_counts: Vec<u32>,
    #[serde(default)]
    pub integrity: NokoiArtifactIntegrity,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_serialized(value: &impl Serialize) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|error| format!("serializing Nokoi identity block failed: {error}"))
}

fn canonical_feature_names() -> Vec<String> {
    NOKOI_FEATURE_SCHEMA
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

fn canonical_feature_contract() -> NokoiFeatureContract {
    NokoiFeatureContract {
        schema_version: NOKOI_FEATURE_SCHEMA_VERSION.into(),
        names: canonical_feature_names(),
        dimensionality: NOKOI_FEATURE_SCHEMA.len(),
        extraction_semantics: vec![
            "FeatureCore.hyperscore as f64".into(),
            "FeatureCore.delta_next as f64".into(),
            "abs(FeatureCore.average_ppm as f64)".into(),
            "abs(FeatureCore.delta_rt_model as f64)".into(),
            "abs(FeatureCore.delta_ims_model as f64)".into(),
            "FeatureCore.charge as f64".into(),
            "FeatureCore.peptide_len as f64".into(),
            "FeatureCore.matched_peaks as f64".into(),
            "FeatureCore.matched_intensity_pct as f64".into(),
            "FeatureCore.isotope_error as f64".into(),
            "FeatureCore.longest_y_pct as f64".into(),
            "log10(FeatureCore.ms2_intensity) when positive, else 0".into(),
        ],
        missing_value_rule: "replace each nonfinite feature with its frozen training median".into(),
        normalization_rule:
            "subtract frozen balanced-training mean and divide by frozen population standard deviation; frozen std<=1e-9 maps to zero"
                .into(),
        finite_range_expectation:
            "all post-imputation normalized values, weights, scores, p-values, and PEPs must be finite; probabilities are in [0,1]"
                .into(),
    }
}

fn canonical_calibration_contract() -> NokoiCalibrationContract {
    NokoiCalibrationContract {
        score_direction: "larger logistic P(target) is stronger target evidence".into(),
        empirical_p_value_rule:
            "upper-tail survival in the frozen sorted OOF null-score distribution with +1 smoothing"
                .into(),
        tie_handling_rule: "all null scores equal to the observed score count in the upper tail"
            .into(),
        interpolation_rule: "piecewise-linear interpolation between frozen p-to-PEP knots".into(),
        boundary_rule: "use the nearest frozen endpoint outside the calibration-knot range".into(),
        pi0_rule: "median Storey estimate over lambda=[0.50,0.60,0.70,0.80,0.90] clamped to [0,1]"
            .into(),
        pep_calibration_rule:
            "Grenander decreasing-density blocks frozen as a monotone p-to-PEP mapping".into(),
    }
}

pub fn stable_candidate_identity(core: &FeatureCore, peptide: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(NOKOI_STABLE_CANDIDATE_SCHEMA.as_bytes());
    hasher.update(b"\0");
    hasher.update(core.file_id.to_le_bytes());
    hasher.update(b"\0");
    hasher.update(core.spec_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(peptide.as_bytes());
    hasher.update(b"\0");
    hasher.update([core.charge]);
    hasher.update(core.rank.to_le_bytes());
    hasher.update(core.label.to_le_bytes());
    hasher.update(core.expmass.to_bits().to_le_bytes());
    hasher.update(core.isotope_error.to_bits().to_le_bytes());
    format!("{:x}", hasher.finalize())
}

fn sorted_identity_digest(ids: impl IntoIterator<Item = String>) -> Result<String, String> {
    let mut ids = ids.into_iter().collect::<Vec<_>>();
    ids.sort();
    if ids.iter().any(String::is_empty) {
        return Err("Nokoi stable candidate identity is empty".into());
    }
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("Nokoi stable candidate identities contain duplicates".into());
    }
    sha256_serialized(&ids)
}

pub fn stable_population_digest(ids: &[String]) -> Result<String, String> {
    sorted_identity_digest(ids.iter().cloned())
}

fn deterministic_key(seed: u64, class: u8, stable_id: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-nokoi-deterministic-order-v1\0");
    hasher.update(seed.to_le_bytes());
    hasher.update([class]);
    hasher.update(stable_id.as_bytes());
    hasher.finalize().into()
}

fn deterministic_feature_row_key(seed: u64, class: u8, row: &PsmData) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-nokoi-deterministic-feature-row-v1\0");
    hasher.update(seed.to_le_bytes());
    hasher.update([class]);
    hasher.update(row.label.to_bits().to_le_bytes());
    for value in &row.features {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.finalize().into()
}

pub fn stable_fold_index(stable_id: &str, folds: usize, seed: u64) -> Result<usize, String> {
    if stable_id.is_empty() || folds < 2 {
        return Err(
            "Nokoi fold assignment requires a stable identity and at least two folds".into(),
        );
    }
    let digest = deterministic_key(seed, 0x46, stable_id);
    let value = u64::from_le_bytes(digest[..8].try_into().expect("fixed digest width"));
    Ok((value % folds as u64) as usize)
}

fn lambda_grid(config: &NokoiConfig) -> Vec<f64> {
    let steps = config.l1_lambda_steps.max(1);
    let lambda_min = config.l1_lambda_min.max(1e-12);
    let lambda_max = config.l1_lambda_max.max(lambda_min);
    let min_log = lambda_min.log10();
    let max_log = lambda_max.log10();
    let step = if steps > 1 {
        (max_log - min_log) / (steps as f64 - 1.0)
    } else {
        0.0
    };
    (0..steps)
        .map(|index| 10.0_f64.powf(min_log + step * index as f64))
        .collect()
}

fn selected_lambda_evaluation(evaluations: &[NokoiLambdaEvaluation]) -> Option<usize> {
    evaluations
        .iter()
        .enumerate()
        .filter(|(_, evaluation)| evaluation.valid)
        .min_by(|(left_index, left), (right_index, right)| {
            left.mean_validation_loss
                .total_cmp(&right.mean_validation_loss)
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
}

fn finite_model(model: &LogisticRegression, dimensions: usize) -> bool {
    model.weights.len() == dimensions
        && model.weights.iter().all(|value| value.is_finite())
        && model.bias.is_finite()
}

fn finite_normalization(normalization: &NokoiNormalization, dimensions: usize) -> bool {
    normalization.medians.len() == dimensions
        && normalization.means.len() == dimensions
        && normalization.stds.len() == dimensions
        && normalization
            .medians
            .iter()
            .chain(&normalization.means)
            .chain(&normalization.stds)
            .all(|value| value.is_finite())
        && normalization.stds.iter().all(|value| *value > 0.0)
}

fn valid_optimization_state(state: &NokoiOptimizationState, lambda_grid: &[f64]) -> bool {
    if state.lambda_evaluations.len() != lambda_grid.len()
        || state.selected_lambda_index >= state.lambda_evaluations.len()
        || state.lambda_selection_fallback_used
        || state.convergence_rule.is_empty()
        || state.final_epochs_completed == 0
        || !state.final_validation_loss.is_finite()
        || !state.selected_mean_validation_loss.is_finite()
    {
        return false;
    }
    if state
        .lambda_evaluations
        .iter()
        .zip(lambda_grid)
        .any(|(evaluation, lambda)| {
            evaluation.l1_lambda.to_bits() != lambda.to_bits()
                || !evaluation.mean_validation_loss.is_finite()
                || (!evaluation.valid && evaluation.mean_validation_loss != 0.0)
        })
    {
        return false;
    }
    let selected = &state.lambda_evaluations[state.selected_lambda_index];
    selected.valid
        && selected.l1_lambda.to_bits() == state.selected_l1_lambda.to_bits()
        && selected.mean_validation_loss.to_bits() == state.selected_mean_validation_loss.to_bits()
}

impl NokoiArtifact {
    fn block_hashes(&self) -> Result<BTreeMap<String, String>, String> {
        let mut hashes = BTreeMap::new();
        macro_rules! block {
            ($name:literal, $value:expr) => {
                hashes.insert($name.into(), sha256_serialized($value)?);
            };
        }
        block!("identity", &self.identity);
        block!("feature_contract", &self.feature_contract);
        block!("feature_schema", &self.feature_schema);
        block!("config", &self.config);
        block!(
            "model_contract",
            &(
                self.schema_version,
                &self.model_version,
                self.min_null_rank,
                self.max_null_rank,
                self.crossfit_seed,
                self.k_folds,
                self.selected_l1_lambda,
            )
        );
        block!("lambda_grid", &self.lambda_grid);
        block!("fold_sizes", &self.fold_sizes);
        block!("fold_models", &self.fold_models);
        block!("final_model", &self.final_model);
        block!("final_optimization", &self.final_optimization);
        block!("normalization_medians", &self.normalization.medians);
        block!("normalization_means", &self.normalization.means);
        block!("normalization_stds", &self.normalization.stds);
        block!("null_scores_oof", &self.null_scores_oof);
        block!("development_pi0", &self.development_pi0);
        block!("calibration_contract", &self.calibration_contract);
        block!("grenander_blocks", &self.grenander_blocks);
        block!("pep_calibration", &self.pep_calibration);
        block!("training_contract", &self.training_contract);
        block!(
            "training_status",
            &(
                self.positive_training_count,
                self.negative_training_count,
                self.training_completed,
                self.training_fallback_used,
            )
        );
        block!("feature_selection_state", &self.feature_selection_state);
        block!(
            "reference_candidate_counts",
            &self.reference_candidate_counts
        );
        Ok(hashes)
    }

    fn canonical_payload_sha256(&self) -> Result<String, String> {
        let mut canonical = self.clone();
        canonical.integrity.canonical_payload_sha256.clear();
        sha256_serialized(&canonical)
    }

    pub fn finalize_integrity(&mut self) -> Result<(), String> {
        self.integrity.block_sha256 = self.block_hashes()?;
        self.integrity.canonical_payload_sha256.clear();
        self.integrity.canonical_payload_sha256 = self.canonical_payload_sha256()?;
        Ok(())
    }

    pub fn stamp_workflow_identity(
        &mut self,
        dataset_id: &str,
        dataset_fingerprint: &str,
        fit_search_fingerprint: &str,
        fit_analysis_fingerprint: &str,
    ) -> Result<(), String> {
        self.identity.dataset_id = dataset_id.into();
        self.identity.dataset_fingerprint = dataset_fingerprint.into();
        self.identity.fit_search_fingerprint = fit_search_fingerprint.into();
        self.identity.fit_analysis_fingerprint = fit_analysis_fingerprint.into();
        self.identity.artifact_creation_mode = "workflow_dataset_local_fit".into();
        self.finalize_integrity()
    }

    pub fn validate_portable(&self) -> Result<(), String> {
        if self.schema_version != NOKOI_ARTIFACT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported Nokoi artifact schema {}; expected {}",
                self.schema_version, NOKOI_ARTIFACT_SCHEMA_VERSION
            ));
        }
        if self.model_version != NOKOI_MODEL_VERSION {
            return Err(format!(
                "unsupported Nokoi model version {:?}",
                self.model_version
            ));
        }
        if self.identity.implementation_source_identity != NOKOI_IMPLEMENTATION_IDENTITY
            || self.identity.implementation_source_sha256 != NOKOI_IMPLEMENTATION_SOURCE_SHA256
            || self.identity.dataset_id.is_empty()
            || self.identity.dataset_fingerprint.is_empty()
            || self.identity.fit_search_fingerprint.is_empty()
            || self.identity.fit_analysis_fingerprint.is_empty()
            || self.identity.candidate_id_schema != NOKOI_CANDIDATE_ID_SCHEMA
            || self.identity.stable_identity_schema != NOKOI_STABLE_CANDIDATE_SCHEMA
            || self.identity.configuration_sha256.is_empty()
            || self.identity.input_feature_schema_sha256.is_empty()
            || self.identity.artifact_creation_mode.is_empty()
            || self.identity.fit_population_sha256.is_empty()
            || self.identity.fit_population_count == 0
        {
            return Err("Nokoi artifact identity is incomplete".into());
        }
        let expected_features = canonical_feature_contract();
        if self.feature_contract != expected_features
            || self.feature_schema != canonical_feature_names()
            || self.identity.input_feature_schema_sha256 != sha256_serialized(&expected_features)?
        {
            return Err(
                "Nokoi feature schema, order, or extraction contract is incompatible".into(),
            );
        }
        let dimensions = NOKOI_FEATURE_SCHEMA.len();
        if self.min_null_rank < 2 || self.max_null_rank < self.min_null_rank {
            return Err("Nokoi artifact null window is invalid".into());
        }
        if self.crossfit_seed != self.training_contract.deterministic_seed
            || self.k_folds < 2
            || self.fold_sizes.len() != self.k_folds
            || self.fold_models.len() != self.k_folds
            || self.fold_models.iter().enumerate().any(|(index, fold)| {
                fold.fold_index != index
                    || fold.heldout_count != self.fold_sizes[index]
                    || fold.heldout_stable_ids_sha256.is_empty()
                    || !fold.fit_completed
                    || fold.fallback_used
                    || !fold.selected_l1_lambda.is_finite()
                    || fold.selected_l1_lambda <= 0.0
                    || fold.selected_l1_lambda.to_bits()
                        != fold.optimization.selected_l1_lambda.to_bits()
                    || !valid_optimization_state(&fold.optimization, &self.lambda_grid)
                    || !finite_model(&fold.model, dimensions)
                    || !finite_normalization(&fold.normalization, dimensions)
            })
        {
            return Err("Nokoi fold-specific training state is incomplete or invalid".into());
        }
        if self.lambda_grid.is_empty()
            || self
                .lambda_grid
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || !self.selected_l1_lambda.is_finite()
            || self.selected_l1_lambda <= 0.0
            || !finite_model(&self.final_model, dimensions)
            || self.selected_l1_lambda.to_bits()
                != self.final_optimization.selected_l1_lambda.to_bits()
            || !valid_optimization_state(&self.final_optimization, &self.lambda_grid)
            || !finite_normalization(&self.normalization, dimensions)
            || self.feature_selection_state.len() != dimensions
            || !self.training_completed
            || self.training_fallback_used
        {
            return Err(
                "Nokoi final fitted state is incomplete, nonfinite, or dimensionally invalid"
                    .into(),
            );
        }
        if self.training_contract.min_null_rank != self.min_null_rank
            || self.training_contract.max_null_rank != self.max_null_rank
            || self.training_contract.positive_training_count != self.positive_training_count
            || self.training_contract.negative_training_count != self.negative_training_count
            || self.training_contract.positive_population_sha256.is_empty()
            || self.training_contract.negative_population_sha256.is_empty()
            || self
                .training_contract
                .null_candidate_population_sha256
                .is_empty()
            || self
                .training_contract
                .candidate_count_population_sha256
                .is_empty()
            || !self.training_contract.positive_threshold.is_finite()
            || !self.training_contract.positive_top_fraction.is_finite()
            || !self.training_contract.null_purification_factor.is_finite()
        {
            return Err("Nokoi training contract is incomplete or inconsistent".into());
        }
        let expected_configuration_sha256 = sha256_serialized(&(
            &self.config,
            self.min_null_rank,
            self.max_null_rank,
            self.k_folds,
            self.training_contract.positive_class_rule.as_str(),
            self.training_contract.positive_top_fraction,
            self.training_contract.positive_threshold,
            self.training_contract.null_purification_rule.as_str(),
            self.training_contract.null_purification_factor,
        ))?;
        if self.identity.configuration_sha256 != expected_configuration_sha256
            || self.training_contract.candidate_count_population_sha256
                != sha256_serialized(&self.reference_candidate_counts)?
            || self.reference_candidate_counts.is_empty()
            || self
                .reference_candidate_counts
                .windows(2)
                .any(|pair| pair[0] > pair[1])
            || self.fold_sizes.iter().sum::<usize>() != self.null_scores_oof.len()
            || self.fold_models.iter().any(|fold| {
                fold.training_negative_count + fold.heldout_count != self.null_scores_oof.len()
                    || !self.lambda_grid.contains(&fold.selected_l1_lambda)
            })
            || !self.lambda_grid.contains(&self.selected_l1_lambda)
        {
            return Err(
                "Nokoi configuration, candidate-count, fold, or lambda identity is inconsistent"
                    .into(),
            );
        }
        if self.null_scores_oof.len() < 50
            || self
                .null_scores_oof
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            || self
                .null_scores_oof
                .windows(2)
                .any(|pair| pair[0] > pair[1])
            || !self.development_pi0.is_finite()
            || !(0.0..=1.0).contains(&self.development_pi0)
        {
            return Err("Nokoi frozen null-score or pi0 calibration state is invalid".into());
        }
        if self.calibration_contract != canonical_calibration_contract()
            || self.grenander_blocks.is_empty()
            || self
                .grenander_blocks
                .first()
                .map_or(true, |block| block.start_p != 0.0)
            || self
                .grenander_blocks
                .last()
                .map_or(true, |block| block.end_p != 1.0)
            || self
                .grenander_blocks
                .iter()
                .map(|block| block.count)
                .sum::<usize>()
                != self.identity.fit_population_count
            || self.grenander_blocks.iter().any(|block| {
                !block.start_p.is_finite()
                    || !block.end_p.is_finite()
                    || block.start_p < 0.0
                    || block.end_p <= block.start_p
                    || block.end_p > 1.0
                    || !block.density.is_finite()
                    || block.density < 0.0
                    || !block.pep.is_finite()
                    || !(0.0..=1.0).contains(&block.pep)
            })
            || self
                .grenander_blocks
                .windows(2)
                .any(|pair| pair[0].end_p != pair[1].start_p || pair[0].density < pair[1].density)
        {
            return Err("Nokoi Grenander calibration state is invalid or nonmonotone".into());
        }
        if self.pep_calibration.len() < 2
            || self.pep_calibration.iter().any(|point| {
                !point.p_value.is_finite()
                    || point.p_value <= 0.0
                    || point.p_value > 1.0
                    || !point.pep.is_finite()
                    || point.pep <= 0.0
                    || point.pep > 1.0
            })
            || self
                .pep_calibration
                .windows(2)
                .any(|pair| pair[0].p_value >= pair[1].p_value || pair[0].pep > pair[1].pep)
        {
            return Err("Nokoi frozen p-to-PEP mapping is invalid or nonmonotone".into());
        }
        let expected_blocks = self.block_hashes()?;
        if self.integrity.block_sha256 != expected_blocks
            || self.integrity.canonical_payload_sha256.is_empty()
            || self.integrity.canonical_payload_sha256 != self.canonical_payload_sha256()?
        {
            return Err("Nokoi artifact integrity hash mismatch".into());
        }
        Ok(())
    }
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

    let seed = NOKOI_CROSSFIT_SEED ^ 0x4C45_4741_4359_0001;
    positives.sort_by_key(|row| deterministic_feature_row_key(seed, 1, row));
    negatives.sort_by_key(|row| deterministic_feature_row_key(seed, 0, row));

    // Determine sample size: min of (pos count, neg count, 10,000 cap)
    let n_pos = positives.len();
    let n_neg = negatives.len();
    let sample_size = n_pos.min(n_neg).min(10_000);

    let mut training_data = Vec::with_capacity(sample_size * 2);
    training_data.extend_from_slice(&positives[0..sample_size]);
    training_data.extend_from_slice(&negatives[0..sample_size]);

    // Shuffle the training set so positives/negatives aren't clustered
    training_data.sort_by_key(|row| deterministic_feature_row_key(seed, row.label as u8, row));

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
    let seed = NOKOI_CROSSFIT_SEED ^ 0x4C45_4741_4359_0002;
    positives.sort_by_key(|row| deterministic_feature_row_key(seed, 1, row));
    negatives.sort_by_key(|row| deterministic_feature_row_key(seed, 0, row));

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

    training_data.sort_by_key(|row| deterministic_feature_row_key(seed, row.label as u8, row));

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
    final_optimization: NokoiOptimizationState,
    fold_sizes: Vec<usize>,
    fold_models: Vec<NokoiFoldModel>,
    positive_training_count: usize,
    negative_training_count: usize,
    positive_population_sha256: String,
    negative_population_sha256: String,
    null_candidate_population_sha256: String,
}

fn train_nokoi_crossfit(
    features: &[DfFeature],
    stable_ids: &[String],
    config: &NokoiConfig,
    min_null_rank: u32,
    max_null_rank: u32,
    k_folds: usize,
    is_positive: impl Fn(&DfFeature) -> bool,
    null_indices: &[usize],
) -> Result<NokoiCrossfitTraining, String> {
    // ---- 0) Fast fallbacks for empty/low data ----
    if features.is_empty() {
        return Err("Nokoi fit population is empty".into());
    }
    if features.len() != stable_ids.len() {
        return Err(format!(
            "Nokoi stable identity count {} does not match feature count {}",
            stable_ids.len(),
            features.len()
        ));
    }
    stable_population_digest(stable_ids)?;

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
    positives_idx.sort_by(|left, right| stable_ids[*left].cmp(&stable_ids[*right]));
    negatives_idx.sort_by(|left, right| stable_ids[*left].cmp(&stable_ids[*right]));

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
        return Err(format!(
            "Nokoi crossfit has insufficient training data: positives={} negatives={}",
            positives_idx.len(),
            negatives_idx.len()
        ));
    }

    // ---- 1) Cross-fit null candidate set: intersection of provided null_indices and rank-window negatives ----
    let mut neg_mask = vec![false; features.len()];
    for &i in &negatives_idx {
        neg_mask[i] = true;
    }

    let mut null_cand: Vec<usize> = Vec::new();
    let mut seen_null_indices = BTreeSet::new();
    for &j in null_indices {
        if j >= neg_mask.len() {
            return Err(format!("Nokoi null candidate index {j} is out of bounds"));
        }
        if !seen_null_indices.insert(j) {
            return Err(format!("Nokoi null candidate index {j} is duplicated"));
        }
        if neg_mask[j] {
            null_cand.push(j);
        }
    }
    null_cand.sort_by(|left, right| stable_ids[*left].cmp(&stable_ids[*right]));

    if null_cand.len() < 50 {
        return Err(format!(
            "Nokoi crossfit has too few purified null candidates: {} < 50",
            null_cand.len()
        ));
    }

    // Stable-ID hash assignment makes folds independent of input row order,
    // process-local PSM numbering, and worker count.
    let k = k_folds.max(2).min(null_cand.len());
    let mut folds = vec![Vec::<usize>::new(); k];
    for &index in &null_cand {
        let fold = stable_fold_index(&stable_ids[index], k, NOKOI_CROSSFIT_SEED)?;
        folds[fold].push(index);
    }
    if folds.iter().any(Vec::is_empty) {
        return Err("Nokoi stable-ID fold assignment produced an empty fold".into());
    }

    // ---- 2) Helper: train a logistic model given pos indices + neg indices; return (model, means, stds) ----
    let train_one = |pos_idx: &[usize],
                     neg_idx: &[usize],
                     seed: u64|
     -> Option<(
        LogisticRegression,
        NokoiNormalization,
        NokoiOptimizationState,
    )> {
        if pos_idx.len() < 50 || neg_idx.len() < 50 {
            return None;
        }

        // Deterministic balanced sampling: take the smallest seeded SHA-256
        // keys within each class, then order the merged training rows by a
        // separate seeded key. No input position participates.
        let mut pos = pos_idx.to_vec();
        let mut neg = neg_idx.to_vec();
        pos.sort_by_key(|index| deterministic_key(seed, 1, &stable_ids[*index]));
        neg.sort_by_key(|index| deterministic_key(seed, 0, &stable_ids[*index]));
        let sample_size = pos.len().min(neg.len()).min(10_000);
        if sample_size == 0 {
            return None;
        }
        let mut selected = pos[..sample_size]
            .iter()
            .copied()
            .map(|index| (index, 1.0, deterministic_key(seed, 3, &stable_ids[index])))
            .chain(
                neg[..sample_size]
                    .iter()
                    .copied()
                    .map(|index| (index, 0.0, deterministic_key(seed, 2, &stable_ids[index]))),
            )
            .collect::<Vec<_>>();
        selected.sort_by(|left, right| {
            left.2
                .cmp(&right.2)
                .then_with(|| stable_ids[left.0].cmp(&stable_ids[right.0]))
        });
        let mut training_data = selected
            .into_iter()
            .map(|(index, label, _)| PsmData {
                features: extract_features(&features[index].core),
                label,
                original_idx: index,
            })
            .collect::<Vec<_>>();

        // Normalize (in-place) and capture means/stds
        let normalization = normalize_features_portable(&mut training_data);

        let n_features = training_data[0].features.len();
        let mut model = LogisticRegression::new(n_features);
        let optimization = model.train_cv_report(&training_data, config);
        if optimization.lambda_selection_fallback_used
            || !optimization.final_validation_loss.is_finite()
            || optimization.final_epochs_completed == 0
        {
            return None;
        }

        Some((model, normalization, optimization))
    };

    // ---- 3) Cross-fit: out-of-fold predictions for null candidates ----
    let mut null_scores_oof: Vec<f64> = Vec::with_capacity(null_cand.len());
    let mut fold_models = Vec::with_capacity(folds.len());

    for (fold_i, heldout) in folds.iter().enumerate() {
        // Training negatives = null_cand excluding heldout
        let mut train_neg: Vec<usize> =
            Vec::with_capacity(null_cand.len().saturating_sub(heldout.len()));
        // mark heldout for fast exclusion
        let mut heldout_flag = vec![false; features.len()];
        for &j in heldout {
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
        let (model, normalization, optimization) = match train_one(&positives_idx, &train_neg, seed)
        {
            Some(x) => x,
            None => {
                return Err(format!("Nokoi crossfit fold {fold_i} could not train"));
            }
        };

        let heldout_stable_ids_sha256 =
            sorted_identity_digest(heldout.iter().map(|index| stable_ids[*index].clone()))?;
        fold_models.push(NokoiFoldModel {
            fold_index: fold_i,
            heldout_count: heldout.len(),
            heldout_stable_ids_sha256,
            training_negative_count: train_neg.len(),
            selected_l1_lambda: optimization.selected_l1_lambda,
            optimization,
            model: model.clone(),
            normalization: normalization.clone(),
            fit_completed: true,
            fallback_used: false,
        });

        // Predict on heldout (OOF) only
        for &j in heldout {
            let mut x = extract_features(&features[j].core);
            if !normalization.apply(&mut x) {
                return Err(format!("Nokoi fold {fold_i} normalization failed"));
            }
            let p = model.predict(&x);
            if !p.is_finite() {
                return Err(format!("Nokoi fold {fold_i} produced a nonfinite score"));
            }
            null_scores_oof.push(p.clamp(0.0, 1.0));
        }
    }

    // If cross-fit produced too few null scores, fail closed.
    if null_scores_oof.len() < 50 {
        return Err(format!(
            "Nokoi crossfit produced too few OOF null scores: {} < 50",
            null_scores_oof.len()
        ));
    }
    null_scores_oof.sort_by(|left, right| left.total_cmp(right));

    // ---- 4) Final model for prob_target_all: train on all positives + ALL rank-window negatives ----
    let seed_final = NOKOI_CROSSFIT_SEED ^ 0xF11A_1EED_1234_5678u64;
    let (model, normalization, final_optimization) =
        match train_one(&positives_idx, &negatives_idx, seed_final) {
            Some(x) => x,
            None => {
                return Err("Nokoi final model could not train".into());
            }
        };

    let mut prob_target_all: Vec<f64> = Vec::with_capacity(features.len());
    for f in features {
        let mut x = extract_features(&f.core);
        if !normalization.apply(&mut x) {
            return Err("Nokoi final normalization failed".into());
        }
        prob_target_all.push(model.predict(&x).clamp(0.0, 1.0));
    }

    if prob_target_all.len() != features.len() {
        return Err("Nokoi final score length mismatch".into());
    }
    if prob_target_all.iter().any(|p| !p.is_finite()) {
        return Err("Nokoi final score stream contains nonfinite probabilities".into());
    }
    if null_scores_oof.is_empty() {
        return Err("Nokoi OOF null-score distribution is empty".into());
    }
    if null_scores_oof.iter().any(|p| !p.is_finite()) {
        return Err("Nokoi OOF null-score distribution contains nonfinite values".into());
    }

    Ok(NokoiCrossfitTraining {
        prob_target_all,
        null_scores_oof,
        final_model: model,
        normalization,
        selected_l1_lambda: final_optimization.selected_l1_lambda,
        final_optimization,
        fold_sizes: folds.iter().map(|fold| fold.len()).collect(),
        fold_models,
        positive_training_count: positives_idx.len(),
        negative_training_count: negatives_idx.len(),
        positive_population_sha256: sorted_identity_digest(
            positives_idx.iter().map(|index| stable_ids[*index].clone()),
        )?,
        negative_population_sha256: sorted_identity_digest(
            negatives_idx.iter().map(|index| stable_ids[*index].clone()),
        )?,
        null_candidate_population_sha256: sorted_identity_digest(
            null_cand.iter().map(|index| stable_ids[*index].clone()),
        )?,
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
    let stable_ids = features
        .iter()
        .map(|feature| {
            stable_candidate_identity(
                &feature.core,
                &format!("legacy-peptide-index-{}", feature.core.peptide_idx.0),
            )
        })
        .collect::<Vec<_>>();
    let fitted = train_nokoi_crossfit(
        features,
        &stable_ids,
        config,
        min_null_rank,
        max_null_rank,
        k_folds,
        is_positive,
        null_indices,
    )
    .ok()?;
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

fn nokoi_grenander_pep_state_from_p_values(
    p_values: &[f64],
    pi0: f64,
) -> (Vec<f64>, Vec<NokoiGrenanderBlock>) {
    const EPS: f64 = 1e-300;

    let n = p_values.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
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

    let frozen = blocks
        .into_iter()
        .map(|block| {
            let density = block.density(n).max(0.0);
            NokoiGrenanderBlock {
                start_p: block.start_p,
                end_p: block.end_p,
                count: block.count,
                density,
                pep: if density > 0.0 {
                    (pi0 / density).clamp(EPS, 1.0)
                } else {
                    1.0
                },
            }
        })
        .collect();
    (out, frozen)
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

    let peps: Vec<f64> = nokoi_grenander_pep_state_from_p_values(&p_values, pi0)
        .0
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

pub struct NokoiFitMetadata<'a> {
    pub stable_ids: &'a [String],
    pub positive_class_rule: &'a str,
    pub positive_top_fraction: f64,
    pub positive_threshold: f64,
    pub null_purification_rule: &'a str,
    pub null_purification_factor: f64,
}

pub fn fit_nokoi_artifact(
    features: &[DfFeature],
    config: &NokoiConfig,
    min_null_rank: u32,
    max_null_rank: u32,
    k_folds: usize,
    is_positive: impl Fn(&DfFeature) -> bool,
    null_indices: &[usize],
) -> Result<NokoiFitResult, String> {
    let stable_ids = features
        .iter()
        .map(|feature| {
            stable_candidate_identity(
                &feature.core,
                &format!("legacy-peptide-index-{}", feature.core.peptide_idx.0),
            )
        })
        .collect::<Vec<_>>();
    fit_nokoi_artifact_with_metadata(
        features,
        config,
        min_null_rank,
        max_null_rank,
        k_folds,
        is_positive,
        null_indices,
        NokoiFitMetadata {
            stable_ids: &stable_ids,
            positive_class_rule: "caller-supplied rank-1 predicate",
            positive_top_fraction: 0.0,
            positive_threshold: 0.0,
            null_purification_rule: "caller-supplied null-index intersection",
            null_purification_factor: 0.0,
        },
    )
}

pub fn fit_nokoi_artifact_with_metadata(
    features: &[DfFeature],
    config: &NokoiConfig,
    min_null_rank: u32,
    max_null_rank: u32,
    k_folds: usize,
    is_positive: impl Fn(&DfFeature) -> bool,
    null_indices: &[usize],
    metadata: NokoiFitMetadata<'_>,
) -> Result<NokoiFitResult, String> {
    if min_null_rank < 2 || max_null_rank < min_null_rank {
        return Err(format!(
            "Nokoi v2 requires a lower-rank null window with 2 <= min <= max; got {min_null_rank}-{max_null_rank}"
        ));
    }
    let fitted = train_nokoi_crossfit(
        features,
        metadata.stable_ids,
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
    let (_, grenander_blocks) =
        nokoi_grenander_pep_state_from_p_values(&evidence.p_values, development_pi0);
    let pep_calibration = freeze_pep_calibration(&evidence.p_values, &evidence.peps);
    if pep_calibration.len() < 2 {
        return Err("Nokoi fit produced an incomplete PEP calibration".into());
    }
    let mut reference_candidate_counts = features
        .iter()
        .filter(|feature| feature.core.rank == 1)
        .map(|feature| feature.core.lo_spectrum_candidate_count)
        .filter(|&count| count > 0)
        .collect::<Vec<_>>();
    reference_candidate_counts.sort_unstable();
    let feature_contract = canonical_feature_contract();
    let input_feature_schema_sha256 = sha256_serialized(&feature_contract)?;
    let configuration_sha256 = sha256_serialized(&(
        config,
        min_null_rank,
        max_null_rank,
        k_folds,
        metadata.positive_class_rule,
        metadata.positive_top_fraction,
        metadata.positive_threshold,
        metadata.null_purification_rule,
        metadata.null_purification_factor,
    ))?;
    let fit_population_sha256 = stable_population_digest(metadata.stable_ids)?;
    let candidate_count_population_sha256 = sha256_serialized(&reference_candidate_counts)?;
    let feature_selection_state = fitted
        .final_model
        .weights
        .iter()
        .map(|weight| weight.abs() > 1e-6)
        .collect();
    let mut artifact = NokoiArtifact {
        schema_version: NOKOI_ARTIFACT_SCHEMA_VERSION,
        model_version: NOKOI_MODEL_VERSION.into(),
        identity: NokoiArtifactIdentity {
            implementation_source_identity: NOKOI_IMPLEMENTATION_IDENTITY.into(),
            implementation_source_sha256: NOKOI_IMPLEMENTATION_SOURCE_SHA256.into(),
            dataset_id: "direct-core-fit".into(),
            dataset_fingerprint: fit_population_sha256.clone(),
            fit_search_fingerprint: format!("direct-population:{fit_population_sha256}"),
            fit_analysis_fingerprint: configuration_sha256.clone(),
            candidate_id_schema: NOKOI_CANDIDATE_ID_SCHEMA.into(),
            stable_identity_schema: NOKOI_STABLE_CANDIDATE_SCHEMA.into(),
            configuration_sha256,
            input_feature_schema_sha256,
            artifact_creation_mode: "core_fit_pending_workflow_provenance".into(),
            fit_population_sha256,
            fit_population_count: metadata.stable_ids.len(),
        },
        feature_contract,
        feature_schema: canonical_feature_names(),
        min_null_rank,
        max_null_rank,
        crossfit_seed: NOKOI_CROSSFIT_SEED,
        k_folds: fitted.fold_sizes.len(),
        fold_sizes: fitted.fold_sizes,
        config: config.clone(),
        lambda_grid: lambda_grid(config),
        fold_models: fitted.fold_models,
        selected_l1_lambda: fitted.selected_l1_lambda,
        final_model: fitted.final_model,
        final_optimization: fitted.final_optimization,
        normalization: fitted.normalization,
        null_scores_oof: fitted.null_scores_oof,
        development_pi0,
        calibration_contract: canonical_calibration_contract(),
        grenander_blocks,
        pep_calibration,
        positive_training_count: fitted.positive_training_count,
        negative_training_count: fitted.negative_training_count,
        training_contract: NokoiTrainingContract {
            positive_class_rule: metadata.positive_class_rule.into(),
            positive_top_fraction: metadata.positive_top_fraction,
            positive_threshold: metadata.positive_threshold,
            negative_class_rule:
                "non-positive candidates whose original rank is within the selected null window"
                    .into(),
            null_purification_rule: metadata.null_purification_rule.into(),
            null_purification_factor: metadata.null_purification_factor,
            min_null_rank,
            max_null_rank,
            class_balancing_rule:
                "equal positive/negative counts selected by seeded stable-ID SHA-256 order".into(),
            maximum_samples_per_class: 10_000,
            deterministic_seed: NOKOI_CROSSFIT_SEED,
            deterministic_ordering_rule:
                "seeded SHA-256 of canonical stable candidate identity; no process-local row index"
                    .into(),
            fold_assignment_rule: "u64_le(SHA-256(seed || stable_candidate_id)[0..8]) modulo k"
                .into(),
            positive_training_count: fitted.positive_training_count,
            negative_training_count: fitted.negative_training_count,
            positive_population_sha256: fitted.positive_population_sha256,
            negative_population_sha256: fitted.negative_population_sha256,
            null_candidate_population_sha256: fitted.null_candidate_population_sha256,
            candidate_count_population_sha256,
        },
        training_completed: true,
        training_fallback_used: false,
        feature_selection_state,
        reference_candidate_counts,
        integrity: NokoiArtifactIntegrity::default(),
    };
    artifact.finalize_integrity()?;
    artifact.validate_portable()?;
    Ok(NokoiFitResult { evidence, artifact })
}

pub fn apply_nokoi_artifact(
    features: &[DfFeature],
    stable_ids: &[String],
    artifact: &NokoiArtifact,
) -> Result<NokoiEvidence, String> {
    apply_nokoi_artifact_with_mode(
        features,
        stable_ids,
        artifact,
        NokoiArtifactApplicationMode::ExactFitPopulation,
        None,
    )
}

pub fn apply_nokoi_artifact_with_mode(
    features: &[DfFeature],
    stable_ids: &[String],
    artifact: &NokoiArtifact,
    mode: NokoiArtifactApplicationMode,
    application_dataset_fingerprint: Option<&str>,
) -> Result<NokoiEvidence, String> {
    artifact.validate_portable()?;
    if mode == NokoiArtifactApplicationMode::SameDatasetTargetOnly
        && application_dataset_fingerprint != Some(artifact.identity.dataset_fingerprint.as_str())
    {
        return Err(
            "Nokoi target-only artifact reuse requires the exact parent dataset fingerprint".into(),
        );
    }
    if features.len() != stable_ids.len() {
        return Err(format!(
            "Nokoi application stable identity count {} does not match feature count {}",
            stable_ids.len(),
            features.len()
        ));
    }
    let application_population_sha256 = stable_population_digest(stable_ids)?;
    if mode == NokoiArtifactApplicationMode::ExactFitPopulation
        && (artifact.identity.fit_population_count != features.len()
            || artifact.identity.fit_population_sha256 != application_population_sha256)
    {
        return Err("Nokoi exact-population artifact application does not match the fitted stable candidate population".into());
    }
    let mut probabilities = Vec::with_capacity(features.len());
    for feature in features {
        let mut values = extract_features(&feature.core);
        if !artifact.normalization.apply(&mut values) {
            return Err("Nokoi frozen normalization failed".into());
        }
        let probability = artifact.final_model.predict(&values);
        if !probability.is_finite() {
            return Err("Nokoi frozen model produced a nonfinite score".into());
        }
        probabilities.push(probability.clamp(0.0, 1.0));
    }
    let p_values =
        calc_empirical_p_values_from_null_scores(&probabilities, &artifact.null_scores_oof);
    let peps = apply_frozen_pep_calibration(&p_values, &artifact.pep_calibration)
        .ok_or_else(|| "Nokoi frozen p-to-PEP calibration application failed".to_string())?;
    if p_values.len() != features.len()
        || peps.len() != features.len()
        || p_values
            .iter()
            .chain(&peps)
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err("Nokoi frozen evidence is incomplete or nonfinite".into());
    }
    Ok(NokoiEvidence { p_values, peps })
}

#[cfg(test)]
mod portable_tests {
    use super::*;

    fn synthetic_population() -> (Vec<DfFeature>, Vec<String>, Vec<usize>, NokoiConfig) {
        let mut features = Vec::new();
        let mut stable_ids = Vec::new();
        for index in 0..300 {
            let rank = if index < 75 {
                1
            } else {
                2 + ((index - 75) % 3) as u32
            };
            let signal = if rank == 1 {
                80.0
            } else {
                8.0 + (index % 37) as f64
            };
            let core = FeatureCore {
                spec_id: format!("scan={}", 10_000 + index),
                file_id: index % 9,
                rank,
                label: 1,
                expmass: 500.0 + index as f32 / 100.0,
                charge: 2 + (index % 3) as u8,
                peptide_len: 8 + index % 12,
                hyperscore: signal + (index % 11) as f64 / 10.0,
                delta_next: signal / 10.0,
                average_ppm: (index as f32 % 7.0) - 3.0,
                delta_rt_model: (index as f32 % 13.0) / 10.0,
                delta_ims_model: (index as f32 % 5.0) / 20.0,
                matched_peaks: 5 + (index % 25) as u32,
                matched_intensity_pct: 0.1 + (index % 80) as f32 / 100.0,
                isotope_error: (index % 2) as f32,
                longest_y_pct: 0.1 + (index % 70) as f32 / 100.0,
                ms2_intensity: 1_000.0 + index as f32 * 17.0,
                lo_spectrum_candidate_count: 100 + (index % 31) as u32,
                ..FeatureCore::default()
            };
            stable_ids.push(format!("synthetic-candidate-{index:04}"));
            features.push(core.to_df());
        }
        let null_indices = features
            .iter()
            .enumerate()
            .filter_map(|(index, feature)| (feature.core.rank > 1).then_some(index))
            .collect();
        let config = NokoiConfig {
            enabled: true,
            train_fdr: 0.01,
            learning_rate: 0.05,
            epochs: 40,
            patience: 5,
            l1_lambda: 0.001,
            l1_lambda_min: 0.0001,
            l1_lambda_max: 0.01,
            l1_lambda_steps: 3,
        };
        (features, stable_ids, null_indices, config)
    }

    fn synthetic_fit() -> (Vec<DfFeature>, Vec<String>, NokoiFitResult) {
        let (features, stable_ids, null_indices, config) = synthetic_population();
        let fitted = fit_nokoi_artifact_with_metadata(
            &features,
            &config,
            2,
            4,
            5,
            |feature| feature.core.rank == 1,
            &null_indices,
            NokoiFitMetadata {
                stable_ids: &stable_ids,
                positive_class_rule: "synthetic rank-1 positive",
                positive_top_fraction: 0.25,
                positive_threshold: 50.0,
                null_purification_rule: "synthetic ranks 2-4",
                null_purification_factor: 0.25,
            },
        )
        .unwrap();
        (features, stable_ids, fitted)
    }

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

    #[test]
    fn portable_v2_roundtrip_replays_without_refitting() {
        let (features, stable_ids, fitted) = synthetic_fit();
        fitted.artifact.validate_portable().unwrap();
        let payload = serde_json::to_vec(&fitted.artifact).unwrap();
        let restored: NokoiArtifact = serde_json::from_slice(&payload).unwrap();
        assert_eq!(restored, fitted.artifact);
        let replay = apply_nokoi_artifact(&features, &stable_ids, &restored).unwrap();
        assert_eq!(
            replay
                .p_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fitted
                .evidence
                .p_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            replay
                .peps
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fitted
                .evidence
                .peps
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn stable_ids_make_fit_and_application_permutation_invariant() {
        let (features, stable_ids, null_indices, config) = synthetic_population();
        let fit = |features: &[DfFeature], stable_ids: &[String], null_indices: &[usize]| {
            fit_nokoi_artifact_with_metadata(
                features,
                &config,
                2,
                4,
                5,
                |feature| feature.core.rank == 1,
                null_indices,
                NokoiFitMetadata {
                    stable_ids,
                    positive_class_rule: "synthetic rank-1 positive",
                    positive_top_fraction: 0.25,
                    positive_threshold: 50.0,
                    null_purification_rule: "synthetic ranks 2-4",
                    null_purification_factor: 0.25,
                },
            )
            .unwrap()
        };
        let first = fit(&features, &stable_ids, &null_indices);
        let order = (0..features.len()).rev().collect::<Vec<_>>();
        let permuted_features = order
            .iter()
            .map(|&index| features[index].clone())
            .collect::<Vec<_>>();
        let permuted_ids = order
            .iter()
            .map(|&index| stable_ids[index].clone())
            .collect::<Vec<_>>();
        let permuted_null = permuted_features
            .iter()
            .enumerate()
            .filter_map(|(index, feature)| (feature.core.rank > 1).then_some(index))
            .collect::<Vec<_>>();
        let second = fit(&permuted_features, &permuted_ids, &permuted_null);
        assert_eq!(first.artifact, second.artifact);
        let second_by_id = permuted_ids
            .iter()
            .zip(&second.evidence.p_values)
            .map(|(id, value)| (id.as_str(), value.to_bits()))
            .collect::<BTreeMap<_, _>>();
        for (id, value) in stable_ids.iter().zip(&first.evidence.p_values) {
            assert_eq!(second_by_id[id.as_str()], value.to_bits());
        }
    }

    #[test]
    fn portable_v2_integrity_and_contract_fail_closed() {
        let (_, _, fitted) = synthetic_fit();
        let mut cases = Vec::new();
        let mut schema = fitted.artifact.clone();
        schema.schema_version = 1;
        cases.push(schema);
        let mut model = fitted.artifact.clone();
        model.model_version = "wrong-model".into();
        cases.push(model);
        let mut feature_order = fitted.artifact.clone();
        feature_order.feature_schema.swap(0, 1);
        cases.push(feature_order);
        let mut dimensions = fitted.artifact.clone();
        dimensions.final_model.weights.pop();
        cases.push(dimensions);
        let mut nonfinite = fitted.artifact.clone();
        nonfinite.final_model.bias = f64::NAN;
        cases.push(nonfinite);
        let mut nonmonotone = fitted.artifact.clone();
        nonmonotone.pep_calibration[1].pep = 0.0;
        cases.push(nonmonotone);
        let mut corrupted = fitted.artifact.clone();
        corrupted.null_scores_oof[0] = 0.123456789;
        cases.push(corrupted);
        let mut source = fitted.artifact.clone();
        source.identity.implementation_source_sha256 = "wrong-source".into();
        cases.push(source);
        let mut missing_fold = fitted.artifact.clone();
        missing_fold.fold_models.clear();
        cases.push(missing_fold);
        let mut missing_calibration = fitted.artifact.clone();
        missing_calibration.pep_calibration.clear();
        cases.push(missing_calibration);
        let mut missing_training_identity = fitted.artifact.clone();
        missing_training_identity
            .training_contract
            .positive_population_sha256
            .clear();
        cases.push(missing_training_identity);
        let mut contradictory_config = fitted.artifact.clone();
        contradictory_config.config.epochs += 1;
        contradictory_config.finalize_integrity().unwrap();
        cases.push(contradictory_config);
        let mut contradictory_calibration = fitted.artifact.clone();
        contradictory_calibration
            .calibration_contract
            .tie_handling_rule = "silently changed tie rule".into();
        contradictory_calibration.finalize_integrity().unwrap();
        cases.push(contradictory_calibration);
        let mut incomplete_optimization = fitted.artifact.clone();
        incomplete_optimization
            .final_optimization
            .final_epochs_completed = 0;
        incomplete_optimization.finalize_integrity().unwrap();
        cases.push(incomplete_optimization);
        for artifact in cases {
            assert!(artifact.validate_portable().is_err());
        }
    }

    #[test]
    fn portable_identity_and_scoring_survive_artifact_relocation() {
        let (features, stable_ids, fitted) = synthetic_fit();
        let payload = serde_json::to_vec(&fitted.artifact).unwrap();
        let root =
            std::env::temp_dir().join(format!("sage-nokoi-relocation-{}", std::process::id()));
        let first = root.join("first/location/artifact.json");
        let second = root.join("unrelated/absolute/location/artifact.json");
        std::fs::create_dir_all(first.parent().unwrap()).unwrap();
        std::fs::create_dir_all(second.parent().unwrap()).unwrap();
        std::fs::write(&first, &payload).unwrap();
        std::fs::write(&second, &payload).unwrap();

        let first_artifact: NokoiArtifact =
            serde_json::from_slice(&std::fs::read(&first).unwrap()).unwrap();
        let second_artifact: NokoiArtifact =
            serde_json::from_slice(&std::fs::read(&second).unwrap()).unwrap();
        assert_eq!(first_artifact, second_artifact);
        assert_eq!(
            first_artifact.integrity.canonical_payload_sha256,
            second_artifact.integrity.canonical_payload_sha256
        );
        let first_evidence = apply_nokoi_artifact(&features, &stable_ids, &first_artifact).unwrap();
        let second_evidence =
            apply_nokoi_artifact(&features, &stable_ids, &second_artifact).unwrap();
        assert_eq!(first_evidence.p_values, second_evidence.p_values);
        assert_eq!(first_evidence.peps, second_evidence.peps);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stable_fold_assignment_depends_only_on_identity_seed_and_fold_count() {
        let (_, stable_ids, _, _) = synthetic_population();
        let assignments = stable_ids
            .iter()
            .map(|id| stable_fold_index(id, 5, NOKOI_CROSSFIT_SEED).unwrap())
            .collect::<Vec<_>>();
        let reversed = stable_ids
            .iter()
            .rev()
            .map(|id| (id, stable_fold_index(id, 5, NOKOI_CROSSFIT_SEED).unwrap()))
            .collect::<BTreeMap<_, _>>();
        for (id, assignment) in stable_ids.iter().zip(assignments) {
            assert_eq!(reversed[id], assignment);
        }
    }

    #[test]
    fn lambda_grid_and_equal_loss_tie_breaking_are_deterministic() {
        let config = NokoiConfig {
            l1_lambda_min: 0.001,
            l1_lambda_max: 0.1,
            l1_lambda_steps: 3,
            ..NokoiConfig::default()
        };
        let grid = lambda_grid(&config);
        assert_eq!(grid.len(), 3);
        assert_eq!(grid[0].to_bits(), 0.001_f64.to_bits());
        assert_eq!(grid[2].to_bits(), 0.1_f64.to_bits());
        let evaluations = grid
            .iter()
            .copied()
            .map(|l1_lambda| NokoiLambdaEvaluation {
                l1_lambda,
                mean_validation_loss: 0.25,
                valid: true,
            })
            .collect::<Vec<_>>();
        assert_eq!(selected_lambda_evaluation(&evaluations), Some(0));
    }

    #[test]
    fn legacy_direct_df_entry_point_is_aligned_permutation_invariant() {
        let (features, stable_ids, _, _) = synthetic_population();
        let first = rescore_df(&features, 0.01, 2, 4, |feature| feature.core.rank == 1).unwrap();
        let reversed_features = features.iter().cloned().rev().collect::<Vec<_>>();
        let reversed_ids = stable_ids.iter().cloned().rev().collect::<Vec<_>>();
        let second = rescore_df(&reversed_features, 0.01, 2, 4, |feature| {
            feature.core.rank == 1
        })
        .unwrap();
        let second_by_id = reversed_ids
            .iter()
            .zip(second)
            .map(|(id, value)| (id.as_str(), value.to_bits()))
            .collect::<BTreeMap<_, _>>();
        for (id, value) in stable_ids.iter().zip(first) {
            assert_eq!(second_by_id[id.as_str()], value.to_bits());
        }
    }

    #[test]
    fn exact_and_same_dataset_target_application_are_explicit() {
        let (features, stable_ids, fitted) = synthetic_fit();
        let target_features = features[..100].to_vec();
        let target_ids = stable_ids[..100].to_vec();
        assert!(apply_nokoi_artifact(&target_features, &target_ids, &fitted.artifact).is_err());
        let evidence = apply_nokoi_artifact_with_mode(
            &target_features,
            &target_ids,
            &fitted.artifact,
            NokoiArtifactApplicationMode::SameDatasetTargetOnly,
            Some(&fitted.artifact.identity.dataset_fingerprint),
        )
        .unwrap();
        assert_eq!(evidence.p_values.len(), target_features.len());
        let duplicate_ids = vec![stable_ids[0].clone(); target_features.len()];
        assert!(apply_nokoi_artifact_with_mode(
            &target_features,
            &duplicate_ids,
            &fitted.artifact,
            NokoiArtifactApplicationMode::SameDatasetTargetOnly,
            Some(&fitted.artifact.identity.dataset_fingerprint),
        )
        .is_err());
        assert!(apply_nokoi_artifact_with_mode(
            &target_features,
            &target_ids,
            &fitted.artifact,
            NokoiArtifactApplicationMode::SameDatasetTargetOnly,
            Some("different-parent-dataset"),
        )
        .is_err());
    }
}
