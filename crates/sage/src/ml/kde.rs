//! Calculate posterior error probabilities for PSMs.
//! We use Kernel Density Estimation to fit a non-parametric model to the
//! discriminant score distribution. Linear interpolation and binning is used to
//! dramatically speed up the PEP calculation
//!
//! Käll, 2008 [https://pubmed.ncbi.nlm.nih.gov/18052118/]
//! Ma, 2012 [https://pubmed.ncbi.nlm.nih.gov/23176103/]

use std::convert::identity;

use super::*;
use rayon::prelude::*;

pub struct Kde<'a> {
    sample: &'a [f64],
    pub bandwidth: f64,
    constant: f64,
}

impl<'a> Kde<'a> {
    pub fn new(sample: &'a [f64], bw_adjust: impl Fn(f64) -> f64) -> Self {
        let n = sample.len();
        assert!(n > 0, "Kde::new requires a non-empty sample");

        let factor = 4.0 / 3.0;
        let exponent = 1.0 / 5.0;
        let sigma = std(sample);

        let raw_bandwidth = sigma * (factor / n as f64).powf(exponent);
        let bandwidth = bw_adjust(raw_bandwidth);

        assert!(
            bandwidth.is_finite() && bandwidth > 0.0,
            "Kde::new requires a positive finite bandwidth"
        );

        let constant = (2.0 * std::f64::consts::PI).sqrt() * bandwidth * n as f64;

        Self {
            sample,
            bandwidth,
            constant,
        }
    }

    fn kernel(&self, x: f64) -> f64 {
        (-0.5 * x.powi(2)).exp()
    }

    pub fn pdf(&self, x: f64) -> f64 {
        let h = self.bandwidth;

        let sum = self
            .sample
            .par_iter()
            .fold(|| 0.0, |acc, xi| acc + self.kernel((x - xi) / h))
            .sum::<f64>();

        sum / self.constant
    }
}

pub struct Builder {
    monotonic: bool,
    bins: usize,
    bw_adjust: Box<dyn Fn(f64) -> f64>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            monotonic: true,
            bins: 1000,
            bw_adjust: Box::new(identity),
        }
    }
}

impl Builder {
    pub fn monotonic(mut self, monotonic: bool) -> Self {
        self.monotonic = monotonic;
        self
    }

    pub fn bw_adjust<F: 'static + Fn(f64) -> f64>(mut self, bw_adjust: F) -> Self {
        self.bw_adjust = Box::new(bw_adjust);
        self
    }

    pub fn bins(mut self, bins: usize) -> Self {
        self.bins = bins;
        self
    }

    pub fn build(self, scores: &[f64], decoys: &[bool]) -> Estimator {
        assert_eq!(
            scores.len(),
            decoys.len(),
            "scores and decoys must have the same length"
        );
        assert!(
            !scores.is_empty(),
            "KDE builder requires at least one score"
        );
        assert!(self.bins >= 2, "KDE builder requires at least two bins");

        let d = scores
            .par_iter()
            .zip(decoys)
            .filter(|&(_, d)| *d)
            .map(|(s, _)| *s)
            .collect::<Vec<_>>();

        let t = scores
            .par_iter()
            .zip(decoys)
            .filter(|&(_, d)| !*d)
            .map(|(s, _)| *s)
            .collect::<Vec<_>>();

        assert!(
            !d.is_empty(),
            "KDE builder requires at least one decoy score"
        );
        assert!(
            !t.is_empty(),
            "KDE builder requires at least one target score"
        );

        // P(decoy)
        let pi = d.len() as f64 / scores.len() as f64;
        let decoy = Kde::new(&d, &self.bw_adjust);
        let target = Kde::new(&t, &self.bw_adjust);

        let mut min_score = f64::INFINITY;
        let mut max_score = f64::NEG_INFINITY;
        for s in scores {
            min_score = min_score.min(*s);
            max_score = max_score.max(*s);
        }

        if !min_score.is_finite() || !max_score.is_finite() {
            panic!("KDE builder requires finite scores");
        }

        if min_score == max_score {
            let pep = pi.clamp(0.0, 1.0);
            return Estimator {
                bins: vec![pep; self.bins],
                min_score,
                score_step: 1.0,
            };
        }

        let score_step = (max_score - min_score) / (self.bins - 1) as f64;

        let mut bins = (0..self.bins)
            .map(|bin| {
                let score = (bin as f64 * score_step) + min_score;
                let decoy_density = decoy.pdf(score) * pi;
                let target_density = target.pdf(score) * (1.0 - pi);
                let denom = target_density + decoy_density;

                if denom > 0.0 && denom.is_finite() {
                    (decoy_density / denom).clamp(0.0, 1.0)
                } else {
                    pi.clamp(0.0, 1.0)
                }
            })
            .collect::<Vec<_>>();

        if self.monotonic {
            // Enforce non-increasing PEP as score increases.
            let init = *bins.last().unwrap();
            bins.iter_mut().rev().fold(init, |acc, x| {
                *x = acc.max(*x);
                *x
            });
        }

        Estimator {
            bins,
            min_score,
            score_step,
        }
    }
}

pub struct Estimator {
    bins: Vec<f64>,
    min_score: f64,
    score_step: f64,
}

impl Estimator {
    /// Calculate the posterior error probability for a given score, under the
    /// pre-fit non-parametric probability model.
    pub fn posterior_error(&self, score: f64) -> f64 {
        if !self.score_step.is_finite() || self.score_step <= 0.0 {
            return self.bins.first().copied().unwrap_or(1.0).clamp(0.0, 1.0);
        }

        let bin_lo = self
            .bins
            .len()
            .saturating_sub(1)
            .min(((score - self.min_score) / self.score_step).floor() as usize);
        let bin_hi = self.bins.len().saturating_sub(1).min(bin_lo + 1);

        let lower = self.bins[bin_lo];
        let upper = self.bins[bin_hi];

        let bin_lo_score = bin_lo as f64 * self.score_step + self.min_score;
        let linear = ((score - bin_lo_score) / self.score_step).clamp(0.0, 1.0);

        let delta = upper - lower;
        (lower + (delta * linear)).clamp(0.0, 1.0)
    }
}
