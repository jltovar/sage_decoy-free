//! Gauss-Jordan elimination for solving systems of linear equations.
//!
//! LDA requires solving the generalized eigenvalue problem for scatter matrices
//! Sb and Sw. This can be written as the standard eigenvalue problem for
//! inv(Sw).dot(Sb), or equivalently as the linear system Sw.dot(x) = Sb,
//! followed by eigenvalue evaluation on x. This module uses the latter form.

use super::matrix::Matrix;

#[derive(Debug)]
pub struct Gauss {
    pub left: Matrix,
    pub right: Matrix,
}

impl Matrix {
    fn swap_rows(&mut self, i: usize, j: usize) {
        for k in 0..self.cols {
            let tmp = self[(i, k)];
            self[(i, k)] = self[(j, k)];
            self[(j, k)] = tmp;
        }
    }
}

impl Gauss {
	fn approx_zero(x: f64, tol: f64) -> bool {
		x.abs() <= tol
	}
	
	fn approx_one(x: f64, tol: f64) -> bool {
		(x - 1.0).abs() <= tol
	}
	
    pub fn solve_inner(left: Matrix, right: Matrix, eps: f64) -> Option<Matrix> {
        let mut g = Gauss { left, right };
        g.fill_zero(eps);
        g.echelon();
        g.reduce();
        g.backfill();

        // If `left` is the identity matrix, then `right` contains
        // the solution to the system of equations
        match g.left_solved_strict() {
            true => Some(g.right),
            false => None,
        }
    }

    pub fn solve_inner_vanilla(left: Matrix, right: Matrix, eps: f64) -> Option<Matrix> {
        let mut g = Gauss { left, right };
        g.fill_zero(eps);
        g.echelon();
        g.reduce();
        g.backfill();

        match g.left_solved_vanilla() {
            true => Some(g.right),
            false => None,
        }
    }

    /// Vanilla-compatible solve (for TDC parity).
    pub fn solve_vanilla_compat(left: Matrix, right: Matrix) -> Option<Matrix> {
        let mut eps = 1E-8;
        while eps <= 1.0 {
            if let Some(mat) = Gauss::solve_inner_vanilla(left.clone(), right.clone(), eps) {
                return Some(mat);
            }
            eps *= 10.0;
        }
        None
    }

    pub fn solve(left: Matrix, right: Matrix) -> Option<Matrix> {
        let mut eps = 1E-8;
        while eps <= 1.0 {
            if let Some(mat) = Gauss::solve_inner(left.clone(), right.clone(), eps) {
                return Some(mat);
            }
            eps *= 10.0;
        }
        None
    }
    /// Add a small diagonal regularization term eps * I before elimination to
    /// reduce singularity and near-singularity in covariance-like systems.
    fn fill_zero(&mut self, eps: f64) {
        for i in 0..self.left.cols {
            self.left[(i, i)] += eps;
        }
    }

    // Check whether the left matrix is numerically consistent with an identity
    // block and any remaining rows are numerically zero.
    fn left_solved_strict(&self) -> bool {
		let n = self.left.cols;
		let diag_eps = 1e-8;
		let off_diag_eps = 1e-8;
	
		for i in 0..n {
			for j in 0..n {
				let x = self.left[(i, j)];
	
				if i == j {
					if !Self::approx_one(x, diag_eps) && !Self::approx_zero(x, diag_eps) {
						log::debug!(
							"Finding solution to linear system failed: left side of matrix [{},{}] = {}",
							i,
							j,
							x
						);
						return false;
					}
				} else if x.abs() > off_diag_eps {
					log::debug!(
						"Finding solution to linear system failed: left side of matrix [{},{}] = {}",
						i,
						j,
						x
					);
					return false;
				}
			}
		}
		true
	}

    // Vanilla-compatible solved-state check preserving legacy off-diagonal semantics.
    fn left_solved_vanilla(&self) -> bool {
		let n = self.left.cols;
		let diag_eps = 1e-8;
	
		for i in 0..n {
			for j in 0..n {
				let x = self.left[(i, j)];
				if i == j {
					if !Self::approx_one(x, diag_eps) && !Self::approx_zero(x, diag_eps) {
						log::debug!(
							"Finding solution to linear system failed: left side of matrix [{},{}] = {}",
							i,
							j,
							x
						);
						return false;
					}
				} else if x > 1E-8 {
					log::debug!(
						"Finding solution to linear system failed: left side of matrix [{},{}] = {}",
						i,
						j,
						x
					);
					return false;
				}
			}
		}
		true
	}

    fn echelon(&mut self) {
        let (m, n) = self.left.shape();
        let mut h = 0;
        let mut k = 0;

        while h < m && k < n {
			// Find the row with the largest-magnitude pivot in the current column.
			let mut max = (h, self.left[(h, k)].abs());
			for i in h..m {
				let candidate = self.left[(i, k)].abs();
				if candidate > max.1 {
					max = (i, candidate);
				}
			}
		
			let i = max.0;
			if Self::approx_zero(self.left[(i, k)], 1e-12) {
				k += 1;
				continue;
			}
		
			// Swap rows (partial pivoting)
			if h != i {
				self.left.swap_rows(h, i);
				self.right.swap_rows(h, i);
			}

            // Clear rows below pivot row
            for i in h + 1..m {
                let factor = self.left[(i, k)] / self.left[(h, k)];
                self.left[(i, k)] = 0.0;
                for j in k + 1..n {
                    self.left[(i, j)] -= self.left[(h, j)] * factor;
                }
                for j in 0..self.right.cols {
                    self.right[(i, j)] -= self.right[(h, j)] * factor;
                }
            }
            h += 1;
            k += 1;
        }
    }

    // Normalize each pivot row so that leading entries are one.
    fn reduce(&mut self) {
        for i in (0..self.left.rows).rev() {
            for j in 0..self.left.cols {
                let x = self.left[(i, j)];
                if x == 0.0 {
                    continue;
                }
                for k in j..self.left.cols {
                    self.left[(i, k)] /= x;
                }
                for k in 0..self.right.cols {
                    self.right[(i, k)] /= x;
                }
                break;
            }
        }
    }

    // Solve the upper triangular matrix
    fn backfill(&mut self) {
        for i in (0..self.left.rows).rev() {
            for j in 0..self.left.cols {
                if Self::approx_zero(self.left[(i, j)], 1e-12) {
                    continue;
                }
                for k in 0..i {
                    let factor = self.left[(k, j)] / self.left[(i, j)];
                    for h in 0..self.left.cols {
                        self.left[(k, h)] -= self.left[(i, h)] * factor;
                    }
                    for h in 0..self.right.cols {
                        self.right[(k, h)] -= self.right[(i, h)] * factor;
                    }
                }
                break;
            }
        }
    }
}
