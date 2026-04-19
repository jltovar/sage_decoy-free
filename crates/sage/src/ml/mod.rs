//! Linear Algebra, Machine Learning & FDR refinement

pub mod gauss;
pub mod kde;
pub mod linear_discriminant;
pub mod lower_order;
pub mod matrix;
pub mod mobility_model;
pub mod msfdr;
pub mod nokoi;
pub mod qvalue;
pub mod retention_alignment;
pub mod retention_model;
pub mod skew_normal;
pub mod stats;

#[allow(dead_code)]
fn all_close(lhs: &[f64], rhs: &[f64], eps: f64) -> bool {
    lhs.iter()
        .zip(rhs.iter())
        .all(|(l, r)| (l - r).abs() <= eps)
}

pub fn norm(slice: &[f64]) -> f64 {
    slice.iter().fold(0.0, |acc, x| acc + x.powi(2)).sqrt()
}

pub fn mean(slice: &[f64]) -> f64 {
    assert!(!slice.is_empty(), "mean requires a non-empty slice");
    slice.iter().sum::<f64>() / slice.len() as f64
}

pub fn std(slice: &[f64]) -> f64 {
    assert!(!slice.is_empty(), "std requires a non-empty slice");
    let mean = mean(slice);
    let x = slice.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2));
    (x / slice.len() as f64).sqrt()
}
