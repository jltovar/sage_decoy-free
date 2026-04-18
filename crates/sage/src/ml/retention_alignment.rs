//! Perform global retention time alignment

use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::atomic::AtomicU32;

use super::matrix::Matrix;
use crate::database::PeptideIx;
use crate::scoring::FeatureCore;
use dashmap::DashMap;
use fnv::FnvHasher;
use rayon::prelude::*;

type FnvDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FnvHasher>>;

fn max_rt_by_file(features: &[FeatureCore], n_files: usize) -> Vec<f64> {
    let max_rt = (0..n_files)
        .map(|_| AtomicU32::new(0))
        .map(|_| AtomicU32::new(0))
        .collect::<Vec<_>>();

    features.par_iter().for_each(|feat| {
        max_rt[feat.file_id].fetch_max(feat.rt.ceil() as u32, std::sync::atomic::Ordering::SeqCst);
    });

    max_rt
        .into_iter()
        .map(|v| v.load(std::sync::atomic::Ordering::Acquire) as f64)
        .collect()
}

fn mean_rt_by_file(
    features: &[FeatureCore],
    filter: impl Fn(&FeatureCore) -> bool + Sync + Send,
) -> FnvDashMap<PeptideIx, HashMap<usize, f64>> {
    let rts: FnvDashMap<PeptideIx, HashMap<usize, f64>> = DashMap::default();
    features
        .par_iter()
        .filter(|feat| filter(feat))
        .for_each(|feat| {
            rts.entry(feat.peptide_idx)
                .or_default()
                .entry(feat.file_id)
                .and_modify(|f| *f = f.min(feat.rt as f64))
                .or_insert(feat.rt as f64);
        });
    rts
}

fn rt_matrix(
    features: &[FeatureCore],
    max_rt: &[f64],
    filter: impl Fn(&FeatureCore) -> bool + Sync + Send,
) -> (HashMap<PeptideIx, f64>, Matrix) {
    let mean_rt = mean_rt_by_file(features, filter);

    let (means, mat): (HashMap<PeptideIx, f64>, Vec<_>) = mean_rt
        .par_iter()
        .map(|entry| {
            let mut v = vec![f64::NAN; max_rt.len()];

            let mut sum = 0.0;
            let mut len = 0.0;
            for (&file_id, &rt) in entry.value() {
                let rt = rt / max_rt[file_id];
                v[file_id] = rt;
                sum += rt;
                len += 1.0;
            }

            ((*entry.key(), sum / len), v)
        })
        .filter(|((_, mean), _)| mean.is_normal())
        .unzip();
    let n = mat.len();
    let mat: Vec<f64> = mat.into_par_iter().flatten().collect();

    (means, Matrix::new(mat, n, max_rt.len()))
}

#[derive(Copy, Clone, Debug)]
pub struct Alignment {
    pub file_id: usize,
    pub max_rt: f32,
    pub slope: f32,
    pub intercept: f32,
    pub support_count: usize,
    pub residual_spread: f32,
    pub is_normalized: bool,
    pub coordinate_system: &'static str,
}

use std::collections::HashSet;

pub fn global_alignment_vanilla_compat(
    features: &mut [FeatureCore],
    n_files: usize,
    selected_psm_ids: &HashSet<usize>,
) -> Vec<Alignment> {
    global_alignment(features, n_files, |f: &FeatureCore| {
        // Vanilla uses: label == 1 && spectrum_q <= 0.01
        // In the fork, spectrum_q isn’t on FeatureCore, so we use the runner’s
        // vanilla-style q-gate (selected_psm_ids) as the equivalent.
        f.label == 1 && selected_psm_ids.contains(&f.psm_id)
    })
}

pub fn global_alignment(
    features: &mut [FeatureCore],
    n_files: usize,
    filter: impl Fn(&FeatureCore) -> bool + Sync + Send,
) -> Vec<Alignment> {
    let max_rt = max_rt_by_file(features, n_files);
    let (_, rt) = rt_matrix(features, &max_rt, filter);

    let mean_rts: Vec<f64> = (0..rt.rows)
        .into_par_iter()
        .map(|row| {
            let (len, sum) = rt
                .row(row)
                .filter(|rt| rt.is_finite())
                .fold((0, 0.0f64), |(len, sum), x| (len + 1, sum + x));
            sum / len as f64
        })
        .collect();

    let alignments = (0..n_files)
        .into_par_iter()
        .map(|file_id| {
            let (len, dot, sum_x, sum_y) = rt
                .col(file_id)
                .zip(mean_rts.iter())
                .filter(|(x, _)| x.is_finite())
                .fold(
                    (0, 0.0f64, 0.0f64, 0.0f64),
                    |(len, dot, sum_x, sum_y), (x, y)| (len + 1, dot + x * y, sum_x + x, sum_y + y),
                );

            let x_mean = sum_x / len as f64;
            let y_mean = sum_y / len as f64;
            let ssxy = dot - len as f64 * x_mean * y_mean;

            let sx2 = rt
                .col(file_id)
                .filter(|rt| rt.is_finite())
                .fold(1E-8f64, |sum, x| sum + (x - x_mean).powi(2));

            let mut slope = ssxy / sx2;
            let mut intercept = y_mean - slope * x_mean;

            if !slope.is_finite() {
                slope = 1.0;
            }

            if !intercept.is_finite() {
                intercept = 0.0;
            }

            // Phase 3: compute residual spread
            let residual_ss = rt
                .col(file_id)
                .zip(mean_rts.iter())
                .filter(|(x, _)| x.is_finite())
                .fold(0.0f64, |acc, (x, y)| {
                    let pred = x * slope as f64 + intercept as f64;
                    acc + (y - pred).powi(2)
                });
            let residual_spread = if len > 0 {
                (residual_ss / len as f64).sqrt() as f32
            } else {
                0.0
            };

            log::info!(
                "aligning file #{file}: y = {m:.4}x + {b:.4}",
                file = file_id,
                m = slope,
                b = intercept
            );

            Alignment {
                file_id,
                max_rt: max_rt[file_id] as f32,
                slope: slope as f32,
                intercept: intercept as f32,
                support_count: len,
                residual_spread,
                is_normalized: true,
                coordinate_system: "normalized_unit_interval",
            }
        })
        .collect::<Vec<Alignment>>();

    log::info!("aligned retention times across {} files", n_files);

    features.par_iter_mut().for_each(|feature| {
        let a = alignments[feature.file_id];
        feature.aligned_rt = (feature.rt / a.max_rt) * a.slope + a.intercept;
    });

    alignments
}
