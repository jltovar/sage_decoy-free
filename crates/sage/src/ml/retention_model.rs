//! Retention time prediction using linear regression
//!
//! See Klammer et al., Anal. Chem. 2007, 79, 16, 6111–6118
//! https://doi.org/10.1021/ac070262k

use super::regression::LinearRegression;
use crate::database::IndexedDatabase;
use crate::mass::VALID_AA;
use crate::peptide::Peptide;
use crate::scoring::FeatureCore;
use rayon::prelude::*;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct RtDiagnostics {
    pub training_n: usize,
    pub r2: f64,
    pub residual_spread: f64,
    pub is_normalized: bool,
    pub unit_label: &'static str,
}

impl Default for RtDiagnostics {
    fn default() -> Self {
        Self {
            training_n: 0,
            r2: 0.0,
            residual_spread: 0.0,
            is_normalized: true,
            unit_label: "unit_interval",
        }
    }
}

/// Try to fit a retention time prediction model
pub fn predict(
    db: &IndexedDatabase,
    features: &mut [FeatureCore],
    filter: impl Fn(&FeatureCore) -> bool + Sync + Send,
) -> Option<RtDiagnostics> {
    let lr = RetentionModel::fit(db, features, filter)?;
    features.par_iter_mut().for_each(|feat| {
        let rt = lr.predict_peptide(db, feat);
        let bounded = rt.clamp(0.0, 1.0) as f32;
        feat.predicted_rt = bounded;
        feat.delta_rt_model = (feat.aligned_rt - bounded).abs();
    });
    Some(RtDiagnostics {
        training_n: lr.training_n,
        r2: lr.r2,
        residual_spread: lr.residual_spread,
        is_normalized: true,
        unit_label: "unit_interval",
    })
}

/// Vanilla-compatible retention prediction:
/// - no min-N training gate
/// - vanilla-style Gauss solve behavior
/// - training set is label==1 and in `selected_psm_ids` (proxy for spectrum_q<=0.01)
pub fn predict_vanilla_compat(
    db: &IndexedDatabase,
    features: &mut [FeatureCore],
    selected_psm_ids: &HashSet<usize>,
) -> Option<RtDiagnostics> {
    let lr = RetentionModel::fit_vanilla_compat(db, features, selected_psm_ids)?;
    features.par_iter_mut().for_each(|feat| {
        let rt = lr.predict_peptide(db, feat);
        let bounded = rt.clamp(0.0, 1.0) as f32;
        feat.predicted_rt = bounded;
        feat.delta_rt_model = (feat.aligned_rt - bounded).abs();
    });
    Some(RtDiagnostics {
        training_n: lr.training_n,
        r2: lr.r2,
        residual_spread: lr.residual_spread,
        is_normalized: true,
        unit_label: "unit_interval",
    })
}

pub struct RetentionModel {
    beta: Vec<f64>,
    map: [usize; 26],
    pub r2: f64,
    pub training_n: usize,
    pub residual_spread: f64,
}

const FEATURES: usize = VALID_AA.len() * 3 + 3;
const N_TERMINAL: usize = VALID_AA.len();
const C_TERMINAL: usize = VALID_AA.len() * 2;
const PEPTIDE_LEN: usize = FEATURES - 3;
const PEPTIDE_MASS: usize = FEATURES - 2;
const INTERCEPT: usize = FEATURES - 1;

impl RetentionModel {
    /// Encode a peptide into a linear feature vector for retention time regression.
    fn embed(peptide: &Peptide, map: &[usize; 26]) -> [f64; FEATURES] {
        let mut embedding = [0.0; FEATURES];
        let cterm = peptide.sequence.len().saturating_sub(3);

        for (aa_idx, residue) in peptide.sequence.iter().enumerate() {
            let idx = map[(residue - b'A') as usize];
            embedding[idx] += 1.0;

            // Embed the first two and last two residues as terminal-position features.
            match aa_idx {
                0 | 1 => embedding[N_TERMINAL + idx] += 1.0,
                x if x == cterm || x == cterm + 1 => {
                    embedding[C_TERMINAL + idx] += 1.0;
                }
                _ => {}
            }
        }

        embedding[PEPTIDE_LEN] = peptide.sequence.len() as f64;
        embedding[PEPTIDE_MASS] = (peptide.monoisotopic as f64).ln_1p();
        embedding[INTERCEPT] = 1.0;
        embedding
    }

    /// Attempt to fit a linear regression model: retention time ~ peptide features
    pub fn fit(
        db: &IndexedDatabase,
        training_set: &[FeatureCore],
        filter: impl Fn(&FeatureCore) -> bool + Sync + Send,
    ) -> Option<Self> {
        // Create AA -> index map
        let mut map = [0; 26];
        for (idx, aa) in VALID_AA.iter().enumerate() {
            map[(aa - b'A') as usize] = idx;
        }

        let training_n = training_set.par_iter().filter(|feat| filter(feat)).count();
        if training_n < 10 {
            log::warn!(
                "Not enough high-quality PSMs ({}) to train the retention time model.",
                training_n
            );
            return None;
        }

        let lr = LinearRegression::fit::<_, FEATURES>(
            training_set,
            |feat| filter(feat),
            |psm| Self::embed(&db[psm.peptide_idx], &map),
            |psm| psm.aligned_rt as f64,
        )
        .or_else(|| {
            log::warn!("Retention model training aborted: singular fit or invalid RT variance");
            None
        })?;

        log::info!("- fit retention time model, rsq = {}", lr.r2);
        Some(Self {
            beta: lr.beta,
            map,
            r2: lr.r2,
            training_n: lr.training_n,
            residual_spread: lr.mse.sqrt(),
        })
    }

    pub fn fit_vanilla_compat(
        db: &IndexedDatabase,
        training_set: &[FeatureCore],
        selected_psm_ids: &HashSet<usize>,
    ) -> Option<Self> {
        let mut map = [0; 26];
        for (idx, aa) in VALID_AA.iter().enumerate() {
            map[(aa - b'A') as usize] = idx;
        }

        let lr = LinearRegression::fit_vanilla_compat::<_, FEATURES>(
            training_set,
            |feat| feat.label == 1 && selected_psm_ids.contains(&feat.psm_id),
            |psm| Self::embed(&db[psm.peptide_idx], &map),
            |psm| psm.aligned_rt as f64,
        )?;

        // The historical RT path explicitly reported zero r-squared for a
        // constant response, unlike the IMS compatibility path.
        let r2 = if lr.target_variance > 0.0 { lr.r2 } else { 0.0 };
        log::info!("- fit retention time model, rsq = {}", r2);

        Some(Self {
            beta: lr.beta,
            map,
            r2,
            training_n: lr.training_n,
            residual_spread: lr.mse.sqrt(),
        })
    }

    /// Predict retention times for a collection of PSMs
    pub fn predict_peptide(&self, db: &IndexedDatabase, psm: &FeatureCore) -> f64 {
        let v = Self::embed(&db[psm.peptide_idx], &self.map);
        v.into_iter()
            .zip(&self.beta)
            .fold(0.0f64, |sum, (x, y)| sum + x * y)
    }
}
