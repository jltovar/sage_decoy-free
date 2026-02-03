use serde::{Deserialize, Serialize};
use statrs::consts::SQRT_2PI;
use statrs::function::erf::erf;
use std::f64::consts::PI;
use std::f64::consts::SQRT_2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkewNormal {
    pub location: f64, // xi
    pub scale: f64,    // omega
    pub shape: f64,    // alpha
}

impl SkewNormal {
    pub fn new(location: f64, scale: f64, shape: f64) -> Self {
        Self {
            location,
            scale,
            shape,
        }
    }

    /// Probability Density Function (PDF)
    pub fn pdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 {
            return 0.0;
        }
        let z = (x - self.location) / self.scale;

        // Standard Normal PDF: phi(z)
        let phi = (-0.5 * z * z).exp() / SQRT_2PI;

        // Standard Normal CDF: Phi(alpha * z)
        // Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        let input_cdf = self.shape * z;
        let capital_phi = 0.5 * (1.0 + erf(input_cdf / SQRT_2));

        // Result: (2 / scale) * phi(z) * Phi(alpha * z)
        (2.0 / self.scale) * phi * capital_phi
    }

    /// Estimate Skew-Normal parameters from sample Mean, Variance, and Skewness
    /// using Method of Moments.
    pub fn from_moments(mean: f64, variance: f64, skewness: f64) -> Option<Self> {
        // Limit skewness to theoretical max/min for Skew-Normal (~ +/- 0.995)
        let max_skew = 0.99;
        let clamped_skew = skewness.clamp(-max_skew, max_skew);

        // Solve for Delta (correlation coefficient) using algebraic inversion
        let const_term = ((4.0 - PI) / 2.0).powf(2.0 / 3.0);
        let skew_term = clamped_skew.abs().powf(2.0 / 3.0);
        let delta_sq = (PI / 2.0) * (skew_term / (skew_term + const_term));
        let delta = delta_sq.sqrt().copysign(skewness);

        // Solve for Alpha (Shape)
        let delta_sq_clamped = delta_sq.min(0.9999);
        let alpha = delta / (1.0 - delta_sq_clamped).sqrt();

        // Solve for Omega (Scale)
        let scale_factor = 1.0 - (2.0 * delta_sq_clamped) / PI;
        let omega = (variance / scale_factor).sqrt();

        // Solve for Xi (Location)
        let xi = mean - omega * delta * (2.0 / PI).sqrt();

        if xi.is_nan() || omega.is_nan() || alpha.is_nan() {
            None
        } else {
            Some(Self {
                location: xi,
                scale: omega,
                shape: alpha,
            })
        }
    }
}
