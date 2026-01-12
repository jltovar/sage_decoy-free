use crate::ml::skew_normal::SkewNormal;
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

        // --- 1. Data-Driven Initialization ---
        // Estimate initial target_weight as fraction of rank-1 scores above null mean + 2*std
        // This acts as a proxy for "target-like" scores using the known lower-rank null params.
        
        // Variance of Gumbel = (pi^2 * beta^2) / 6  =>  StdDev = (pi * beta) / sqrt(6)
        let null_std = null_beta * (std::f64::consts::PI / 6.0f64.sqrt()); 
        let threshold = null_mu + 2.0 * null_std;
        
        let target_frac = scores.iter().filter(|&&s| s > threshold).count() as f64 / n;
        
        // Clamp initial weight between 0.2 and 0.8 for stability.
        let mut target_weight = target_frac.clamp(0.2, 0.8);

        // Initialize Target Dist: SkewNormal starting as standard Normal
        let mut target_dist = SkewNormal::new(0.0, 1.0, 0.0); 
        
        // Fixed Null Dist (Pre-calculated from lower ranks)
        let null_dist = Gumbel::new(null_mu, null_beta).unwrap();

        // --- 2. EM Loop ---
        let mut prev_ll = f64::NEG_INFINITY;
        
        for _iter in 0..15 {
            let mut current_ll = 0.0;
            
            // --- E-STEP: Calculate Responsibilities ---
            // resp[i] = P(Target | score[i])
            let mut responsibilities = Vec::with_capacity(scores.len());
            let mut sum_resp = 0.0;

            for &x in scores {
                let f_target = target_dist.pdf(x);
                let f_null = null_dist.pdf(x);

                let num = target_weight * f_target;
                let den = num + (1.0 - target_weight) * f_null;

                // Log-likelihood contribution: sum(ln(density))
                if den > 0.0 {
                    current_ll += den.ln();
                }

                let r = if den > 0.0 { num / den } else { 0.0 };
                responsibilities.push(r);
                sum_resp += r;
            }

            // --- Convergence Check ---
            // If log-likelihood hasn't improved significantly, stop early.
            if (current_ll - prev_ll).abs() < 1e-5 {
                break;
            }
            prev_ll = current_ll;

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