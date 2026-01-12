use crate::ml::skew_normal::SkewNormal;
use crate::ml::stats;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, Gumbel};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MsfdrModel {
    /// Proportion of true targets (pi)
    pub target_weight: f64,
    /// Distribution of True Targets (Skew-Normal)
    pub target_dist: SkewNormal,
    /// Distribution of Null/Decoys (Gumbel)
    pub null_location: f64,
    pub null_scale: f64,
}

impl MsfdrModel {
    pub fn new(
        target_weight: f64,
        target_dist: SkewNormal,
        null_location: f64,
        null_scale: f64,
    ) -> Self {
        Self {
            target_weight,
            target_dist,
            null_location,
            null_scale,
        }
    }

    /// Calculate the Posterior Probability that a score belongs to the Target distribution
    /// P(Target | x) = (pi * f_target(x)) / (pi * f_target(x) + (1-pi) * f_null(x))
    pub fn posterior_probability(&self, score: f64) -> f64 {
        // 1. Calculate Densities
        let f_target = self.target_dist.pdf(score);

        // Construct Gumbel on the fly to get PDF
        let gumbel = Gumbel::new(self.null_location, self.null_scale).unwrap();
        let f_null = gumbel.pdf(score);

        // 2. Weight them
        let prob_target = self.target_weight * f_target;
        let prob_null = (1.0 - self.target_weight) * f_null;

        // 3. Normalize
        let total_prob = prob_target + prob_null;

        if total_prob == 0.0 {
            0.0
        } else {
            prob_target / total_prob
        }
    }

    /// Calculate PEP (Posterior Error Probability)
    /// PEP = P(Null | x) = 1 - P(Target | x)
    pub fn calculate_pep(&self, score: f64) -> f64 {
        1.0 - self.posterior_probability(score)
    }

    /// Fit the MSFDR mixture model using Expectation-Maximization (EM).
    ///
    /// * `scores`: The list of all Rank 1 scores (targets + decoys mixed).
    /// * `null_mu`: Fixed Gumbel location (from Decoys/Lower-Order).
    /// * `null_beta`: Fixed Gumbel scale (from Decoys/Lower-Order).
    pub fn fit(scores: &[f64], null_mu: f64, null_beta: f64) -> Option<Self> {
        let n = scores.len() as f64;
        if n < 10.0 {
            return None;
        }

        // --- 1. Initialization ---
        // Assume start with 80% targets (optimistic)
        let mut target_weight = 0.8;

        // Initialize Target Dist as a simple Normal Distribution first
        // Take the top 50% of scores to estimate mean/var for initialization
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.total_cmp(b));
        let top_half = &sorted_scores[(n as usize / 2)..];

        let init_mean = stats::mean(top_half);
        let init_std = stats::std_dev(top_half);

        // Initial Target: Normal (alpha=0)
        let mut target_dist = SkewNormal::new(init_mean, init_std, 0.0);

        // Fixed Null Dist
        let null_dist = Gumbel::new(null_mu, null_beta).unwrap();

        // --- 2. EM Loop ---
        for _iter in 0..15 {
            // 15 iterations usually sufficient

            // --- E-STEP: Calculate Responsibilities ---
            // resp[i] = P(Target | score[i])
            let mut responsibilities = Vec::with_capacity(scores.len());
            let mut sum_resp = 0.0;

            for &x in scores {
                let f_target = target_dist.pdf(x);
                let f_null = null_dist.pdf(x);

                let num = target_weight * f_target;
                let den = num + (1.0 - target_weight) * f_null;

                let r = if den > 0.0 { num / den } else { 0.0 };
                responsibilities.push(r);
                sum_resp += r;
            }

            // --- M-STEP: Update Parameters ---

            // 1. Update Weight
            let new_weight = sum_resp / n;
            // Clamp weight to avoid collapse (e.g. 1% to 99%)
            target_weight = new_weight.clamp(0.01, 0.99);

            // 2. Update Target Parameters (Weighted Moments)
            // Calculate Weighted Mean
            let w_mean: f64 = scores
                .iter()
                .zip(&responsibilities)
                .map(|(&x, &r)| x * r)
                .sum::<f64>()
                / sum_resp;

            // Calculate Weighted Variance
            let w_var: f64 = scores
                .iter()
                .zip(&responsibilities)
                .map(|(&x, &r)| r * (x - w_mean).powi(2))
                .sum::<f64>()
                / sum_resp;

            // Calculate Weighted Skewness
            // skew = (sum r * (x - mu)^3) / (sum_resp * sigma^3)
            let w_std = w_var.sqrt();
            let w_skew: f64 = scores
                .iter()
                .zip(&responsibilities)
                .map(|(&x, &r)| r * ((x - w_mean) / w_std).powi(3))
                .sum::<f64>()
                / sum_resp;

            // Fit Skew-Normal from these moments
            // If fit fails (e.g. skew too high), keep previous dist
            if let Some(new_dist) = SkewNormal::from_moments(w_mean, w_var, w_skew) {
                target_dist = new_dist;
            }
        }

        Some(Self {
            target_weight,
            target_dist,
            null_location: null_mu,
            null_scale: null_beta,
        })
    }
}
