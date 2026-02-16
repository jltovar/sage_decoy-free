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
        let input_cdf = self.shape * z;
        let capital_phi = 0.5 * (1.0 + erf(input_cdf / SQRT_2));

        // Result: (2 / scale) * phi(z) * Phi(alpha * z)
        (2.0 / self.scale) * phi * capital_phi
    }

    /// Estimate Skew-Normal parameters from sample Mean, Variance, and Skewness
    /// using Method of Moments.
    pub fn from_moments(mean: f64, variance: f64, skewness: f64) -> Option<Self> {
        // Clamp skewness for numerical stability (theoretical max is ~0.995)
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

    /// Cumulative Distribution Function (CDF)
    ///
    /// Uses the standard skew-normal CDF identity:
    ///   F(x) = Φ(z) - 2 * T(z, α),   where z = (x - ξ) / ω
    pub fn cdf(&self, x: f64) -> f64 {
        if self.scale <= 0.0 || !x.is_finite() {
            return 0.0;
        }
        let z = (x - self.location) / self.scale;
        if !z.is_finite() {
            return if z.is_sign_positive() { 1.0 } else { 0.0 };
        }

        // Standard Normal CDF Φ(z)
        let phi_cdf = 0.5 * (1.0 + erf(z / SQRT_2));

        // Owen’s T term (tail-stable approximations + Simpson)
        let t = owen_t(z, self.shape);

        // Clamp defensively to [0,1] for stability in EM responsibilities
        (phi_cdf - 2.0 * t).clamp(0.0, 1.0)
    }

    /// Survival Function (SF) = 1 - CDF
    pub fn sf(&self, x: f64) -> f64 {
        (1.0 - self.cdf(x)).clamp(0.0, 1.0)
    }
}

// =============================================================================
// Owen’s T(h, a) implementation for skew-normal CDF
//
// Definition:
//   T(h,a) = (1 / (2π)) ∫_0^a exp(-0.5 * h^2 * (1 + t^2)) / (1 + t^2) dt
// =============================================================================

#[inline]
fn std_norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

#[inline]
fn owen_t_integrand(h: f64, t: f64) -> f64 {
    // exp(-0.5*h^2*(1+t^2)) / (1+t^2)
    let den = 1.0 + t * t;
    (-0.5 * h * h * den).exp() / den
}

/// Owen’s T via fixed high-N Simpson rule (N=128) with regime shortcuts.
fn owen_t(h: f64, a: f64) -> f64 {
    if !h.is_finite() || !a.is_finite() {
        return 0.0;
    }
    if a == 0.0 {
        return 0.0;
    }

    // Symmetry: T(h, -a) = -T(h, a)
    let sign = if a < 0.0 { -1.0 } else { 1.0 };
    let a = a.abs();

    // Very large |h| => integrand ~ exp(-0.5*h^2*(...)) ~ 0
    // This is safe and stabilizes extreme tails.
    if h.abs() > 12.0 {
        return 0.0;
    }

    // Small-|a| linear approximation (good enough and avoids tiny-step noise)
    if a < 1e-6 {
        let approx = a * std_norm_pdf(h) / (2.0 * PI);
        return sign * approx;
    }

    // Simpson integration on [0, a]
    // N must be even.
    const N: usize = 128;
    let n = N;
    let h_step = a / (n as f64);

    let mut sum = owen_t_integrand(h, 0.0) + owen_t_integrand(h, a);
    for i in 1..n {
        let t = (i as f64) * h_step;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * owen_t_integrand(h, t);
    }

    let integral = (h_step / 3.0) * sum;
    let tval = integral / (2.0 * PI);

    sign * tval
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{ContinuousCDF, Normal};

    #[test]
    fn cdf_is_monotone_increasing() {
        let sn = SkewNormal::new(0.0, 1.0, 3.0);
        let xs = [-6.0, -3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0];

        let mut prev = 0.0;
        for &x in &xs {
            let c = sn.cdf(x);
            assert!(c.is_finite());
            assert!(
                c >= prev - 1e-12,
                "cdf not monotone: prev={} cdf({})={}",
                prev,
                x,
                c
            );
            prev = c;
        }
    }

    #[test]
    fn sf_is_one_minus_cdf() {
        let sn = SkewNormal::new(0.0, 1.0, -2.0);
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            let c = sn.cdf(x);
            let s = sn.sf(x);
            let diff = (s - (1.0 - c)).abs();
            assert!(diff <= 1e-12, "sf != 1-cdf at x={}: sf={} cdf={}", x, s, c);
        }
    }

    #[test]
    fn shape_zero_reduces_to_normal_cdf_approximately() {
        let sn = SkewNormal::new(1.25, 2.0, 0.0);
        let n = Normal::new(1.25, 2.0).unwrap();

        for &x in &[-5.0, -1.0, 0.0, 1.25, 3.0, 6.0, 10.0] {
            let c_sn = sn.cdf(x);
            let c_n = n.cdf(x);
            // Owen's T should be ~0 when shape=0 so these should match closely.
            assert!(
                (c_sn - c_n).abs() < 5e-6,
                "shape=0 mismatch at x={}: sn={} normal={}",
                x,
                c_sn,
                c_n
            );
        }
    }

    #[test]
    fn cdf_sf_clamp_to_unit_interval() {
        let sn = SkewNormal::new(0.0, 1.0, 50.0);
        for &x in &[-100.0, -20.0, -10.0, 0.0, 10.0, 20.0, 100.0] {
            let c = sn.cdf(x);
            let s = sn.sf(x);
            assert!(
                (0.0..=1.0).contains(&c),
                "cdf out of range at x={}: {}",
                x,
                c
            );
            assert!(
                (0.0..=1.0).contains(&s),
                "sf out of range at x={}: {}",
                x,
                s
            );
        }
    }
}
