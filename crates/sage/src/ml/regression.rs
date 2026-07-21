//! Streaming OLS linear regression.
//!
//! Fits `beta = (X^T X)^-1 X^T y` without materializing the `n x D` design
//! matrix: one parallel fold/reduce pass accumulates `X^T X`, `X^T y`,
//! `sum(y)` and `n` from per-row outer products. A second pass
//! evaluates `sum((X beta - y)^2)` to report r^2 that matches the
//! materialized formula numerically.
//!
//! Per-worker scratch is `O(D^2)` (the cov accumulator), independent of `n`.

use super::{gauss::Gauss, matrix::Matrix};
use rayon::prelude::*;

pub struct LinearRegression {
    pub beta: Vec<f64>,
    pub r2: f64,
    pub mse: f64,
    pub training_n: usize,
    pub target_variance: f64,
}

struct Acc {
    cov: Vec<f64>, // D*D row-major
    b: Vec<f64>,   // D
    sum_y: f64,
    n: usize,
}

impl Acc {
    fn zero(d: usize) -> Self {
        Self {
            cov: vec![0.0; d * d],
            b: vec![0.0; d],
            sum_y: 0.0,
            n: 0,
        }
    }

    fn add_row(&mut self, row: &[f64], y: f64) {
        let d = row.len();
        for j in 0..d {
            let rj = row[j];
            self.b[j] += rj * y;
            let off = j * d;
            for k in 0..d {
                self.cov[off + k] += rj * row[k];
            }
        }
        self.sum_y += y;
        self.n += 1;
    }

    fn merge(mut self, other: Acc) -> Acc {
        for i in 0..self.cov.len() {
            self.cov[i] += other.cov[i];
        }
        for i in 0..self.b.len() {
            self.b[i] += other.b[i];
        }
        self.sum_y += other.sum_y;
        self.n += other.n;
        self
    }
}

impl LinearRegression {
    /// Fit OLS over `items` with predicate `filter`. `embed(item)` produces a
    /// design row of length `D`; `target(item)` produces the response.
    ///
    /// Returns `None` if no items pass the filter or `X^T X` is singular.
    pub fn fit<T: Sync, const D: usize>(
        items: &[T],
        filter: impl Fn(&T) -> bool + Sync,
        embed: impl Fn(&T) -> [f64; D] + Sync,
        target: impl Fn(&T) -> f64 + Sync,
    ) -> Option<Self> {
        Self::fit_inner(items, filter, embed, target, false)
    }

    /// Fit with the legacy Sage Gauss-Jordan solved-state semantics. This is
    /// kept for the vanilla TDC compatibility path; decoy-free fits use the
    /// stricter solver above.
    pub fn fit_vanilla_compat<T: Sync, const D: usize>(
        items: &[T],
        filter: impl Fn(&T) -> bool + Sync,
        embed: impl Fn(&T) -> [f64; D] + Sync,
        target: impl Fn(&T) -> f64 + Sync,
    ) -> Option<Self> {
        Self::fit_inner(items, filter, embed, target, true)
    }

    fn fit_inner<T: Sync, const D: usize>(
        items: &[T],
        filter: impl Fn(&T) -> bool + Sync,
        embed: impl Fn(&T) -> [f64; D] + Sync,
        target: impl Fn(&T) -> f64 + Sync,
        vanilla_compat: bool,
    ) -> Option<Self> {
        let acc = items
            .par_iter()
            .filter(|x| filter(x))
            .fold(
                || Acc::zero(D),
                |mut acc, x| {
                    let row = embed(x);
                    acc.add_row(&row, target(x));
                    acc
                },
            )
            .reduce(|| Acc::zero(D), Acc::merge);

        if acc.n == 0 {
            return None;
        }

        let nf = acc.n as f64;
        let y_mean = acc.sum_y / nf;

        let cov = Matrix::new(acc.cov, D, D);
        let b_mat = Matrix::col_vector(acc.b);
        let beta = if vanilla_compat {
            Gauss::solve_vanilla_compat(cov, b_mat)?.take()
        } else {
            Gauss::solve(cov, b_mat)?.take()
        };

        if !vanilla_compat && !beta.iter().all(|x| x.is_finite()) {
            log::warn!("linear regression produced non-finite coefficients");
            return None;
        }

        // Streaming pass for SSE and the centered target sum of squares. The
        // direct centered sum is more stable than sum(y^2) - sum(y)^2 / n.
        let (sse, y_var): (f64, f64) = items
            .par_iter()
            .filter(|x| filter(x))
            .map(|x| {
                let row = embed(x);
                let pred: f64 = row.iter().zip(&beta).map(|(v, w)| v * w).sum();
                let act = target(x);
                ((pred - act).powi(2), (act - y_mean).powi(2))
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        if !vanilla_compat && (!sse.is_finite() || !y_var.is_finite() || y_var <= 0.0) {
            log::warn!(
                "linear regression has invalid residual/target variance (sse={sse}, variance={y_var})"
            );
            return None;
        }

        let r2 = 1.0 - sse / y_var;
        let mse = sse / nf;
        Some(Self {
            beta,
            r2,
            mse,
            training_n: acc.n,
            target_variance: y_var,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fit_perfect_line() {
        // y = 2 x + 1, with intercept embedded as the last column.
        let items: Vec<(f64, f64)> = (0..50).map(|i| (i as f64, 2.0 * i as f64 + 1.0)).collect();
        let lr = LinearRegression::fit::<_, 2>(&items, |_| true, |&(x, _)| [x, 1.0], |&(_, y)| y)
            .unwrap();
        assert!((lr.beta[0] - 2.0).abs() < 1e-9, "slope: {}", lr.beta[0]);
        assert!((lr.beta[1] - 1.0).abs() < 1e-9, "intercept: {}", lr.beta[1]);
        assert!((lr.r2 - 1.0).abs() < 1e-9, "r2: {}", lr.r2);
    }

    #[test]
    fn fit_with_noise() {
        // y ~= 3 x + 2 with a deterministic perturbation; r^2 should be high.
        let items: Vec<(f64, f64)> = (0..200)
            .map(|i| {
                let x = i as f64 / 10.0;
                let noise = ((i as f64) * 0.7).sin() * 0.1;
                (x, 3.0 * x + 2.0 + noise)
            })
            .collect();
        let lr = LinearRegression::fit::<_, 2>(&items, |_| true, |&(x, _)| [x, 1.0], |&(_, y)| y)
            .unwrap();
        assert!((lr.beta[0] - 3.0).abs() < 0.05, "slope: {}", lr.beta[0]);
        assert!((lr.beta[1] - 2.0).abs() < 0.1, "intercept: {}", lr.beta[1]);
        assert!(lr.r2 > 0.99, "r2: {}", lr.r2);
    }

    #[test]
    fn empty_filter_returns_none() {
        let items: Vec<f64> = vec![1.0, 2.0, 3.0];
        let lr = LinearRegression::fit::<_, 1>(&items, |_| false, |_| [1.0], |&y| y);
        assert!(lr.is_none());
    }

    #[test]
    fn strict_fit_rejects_constant_response_but_vanilla_preserves_it() {
        let items = vec![2.0; 20];
        let strict = LinearRegression::fit::<_, 1>(&items, |_| true, |_| [1.0], |&y| y);
        assert!(strict.is_none());

        let vanilla =
            LinearRegression::fit_vanilla_compat::<_, 1>(&items, |_| true, |_| [1.0], |&y| y)
                .expect("the compatibility path should retain the historical degenerate fit");
        assert_eq!(vanilla.training_n, items.len());
        assert_eq!(vanilla.target_variance, 0.0);
        assert!(!vanilla.r2.is_finite());
    }
}
