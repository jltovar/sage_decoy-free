//! Ion mobility prediction using linear regression.
//!
//! The model uses a linear feature embedding and solves the associated
//! normal equations to obtain regression coefficients.

use super::regression::LinearRegression;
use crate::database::IndexedDatabase;
use crate::mass::VALID_AA;
use crate::peptide::Peptide;
use crate::scoring::FeatureCore;
use rayon::prelude::*;

#[derive(Clone, Debug, Default)]
pub struct ImsDiagnostics {
    pub training_n: usize,
    pub r2: f64,
    pub mse: f64,
}

/// Try to fit an ion mobility prediction model
pub fn predict(
    db: &IndexedDatabase,
    feats: &mut [FeatureCore],
    filter: impl Fn(&FeatureCore) -> bool + Sync + Send,
) -> Option<ImsDiagnostics> {
    let lr = match MobilityModel::fit(db, feats, filter) {
        Some(lr) => lr,
        None => {
            log::warn!("Mobility model failed to train");
            return None;
        }
    };

    feats.par_iter_mut().for_each(|feat| {
        let ims = lr.predict_peptide(db, feat);
        // Constrain predictions to a bounded physical range used by this model path.
        let bounded = ims.clamp(0.0, 2.0) as f32;
        feat.predicted_ims = bounded;
        feat.delta_ims_model = (feat.ims - bounded).abs();
    });

    Some(ImsDiagnostics {
        training_n: lr.training_n,
        r2: lr.r2,
        mse: lr.mse,
    })
}

/// Vanilla-compatible mobility prediction:
/// - does NOT require ims > 0
/// - does NOT require a minimum count
/// - does NOT abort on zero variance
/// This matches upstream Sage behavior, including degenerate r2/mse.
pub fn predict_vanilla_compat(
    db: &IndexedDatabase,
    feats: &mut [FeatureCore],
    filter: impl Fn(&FeatureCore) -> bool + Sync + Send,
) -> Option<ImsDiagnostics> {
    let lr = match MobilityModel::fit_vanilla_compat(db, feats, filter) {
        Some(lr) => lr,
        None => {
            log::warn!("Mobility model failed to train");
            return None;
        }
    };

    feats.par_iter_mut().for_each(|feat| {
        let ims = lr.predict_peptide(db, feat);
        // Constrain predictions to a bounded physical range used by this model path.
        let bounded = ims.clamp(0.0, 2.0) as f32;
        feat.predicted_ims = bounded;
        feat.delta_ims_model = (feat.ims - bounded).abs();
    });

    Some(ImsDiagnostics {
        training_n: lr.training_n,
        r2: lr.r2,
        mse: lr.mse,
    })
}

pub struct MobilityModel {
    beta: Vec<f64>,
    map: [usize; 26],
    pub r2: f64,
    pub mse: f64,
    pub training_n: usize,
}

const BULKY_AA_IDXS: [usize; 6] = [
    b'L' as usize - b'A' as usize,
    b'V' as usize - b'A' as usize,
    b'I' as usize - b'A' as usize,
    b'F' as usize - b'A' as usize,
    b'W' as usize - b'A' as usize,
    b'Y' as usize - b'A' as usize,
];

const UNCHARGED_POLAR_AA_IDXS: [usize; 4] = [
    b'S' as usize - b'A' as usize,
    b'T' as usize - b'A' as usize,
    b'N' as usize - b'A' as usize,
    b'Q' as usize - b'A' as usize,
];

const POSITIVE_AA_IDXS: [usize; 3] = [
    b'R' as usize - b'A' as usize,
    b'K' as usize - b'A' as usize,
    b'H' as usize - b'A' as usize,
];

const NEGATIVE_AA_IDXS: [usize; 2] = [b'D' as usize - b'A' as usize, b'E' as usize - b'A' as usize];

const TINY_AA_IDXS: [usize; 3] = [
    b'G' as usize - b'A' as usize,
    0,
    b'S' as usize - b'A' as usize,
];

const BRANCHED_AA_IDXS: [usize; 3] = [
    b'L' as usize - b'A' as usize,
    b'I' as usize - b'A' as usize,
    b'V' as usize - b'A' as usize,
];

const FEATURES: usize = VALID_AA.len() * 4 + 12;
const PCT_FEATURES_START: usize = VALID_AA.len();
const N_TERMINAL: usize = VALID_AA.len() * 2;
const C_TERMINAL: usize = VALID_AA.len() * 3;
const NUM_BRANCHED: usize = FEATURES - 12;
const NUM_TINY: usize = FEATURES - 11;
const NUM_UC_POLAR: usize = FEATURES - 10;
const NUM_BULKY: usize = FEATURES - 9;
const NUM_POSITIVE: usize = FEATURES - 8;
const NUM_NEGATIVE: usize = FEATURES - 7;
const INV_PEPTIDE_CHARGE: usize = FEATURES - 6;
const PEPTIDE_CHARGE: usize = FEATURES - 5;
const PEPTIDE_MZ: usize = FEATURES - 4;
const PEPTIDE_LEN: usize = FEATURES - 3;
const PEPTIDE_MASS: usize = FEATURES - 2;
const INTERCEPT: usize = FEATURES - 1;

impl MobilityModel {
    /// One-hot encoding of peptide sequences into feature vector
    fn embed(peptide: &Peptide, charge: &u8, map: &[usize; 26]) -> [f64; FEATURES] {
        let mut embedding = [0.0; FEATURES];
        let cterm = peptide.sequence.len().saturating_sub(3);
        let pep_length = peptide.sequence.len() as f64;

        for (aa_idx, residue) in peptide.sequence.iter().enumerate() {
            let idx = map[(residue - b'A') as usize];
            embedding[idx] += 1.0;
            // Embed N- and C-terminal AA's
            match aa_idx {
                0 | 1 => embedding[N_TERMINAL + idx] += 1.0,
                x if x > cterm => embedding[C_TERMINAL + idx] += 1.0,
                _ => {}
            }
            let x = idx;

            if BULKY_AA_IDXS.contains(&x) {
                embedding[NUM_BULKY] += 1.0;
            };
            if UNCHARGED_POLAR_AA_IDXS.contains(&x) {
                embedding[NUM_UC_POLAR] += 1.0;
            };
            if POSITIVE_AA_IDXS.contains(&x) {
                embedding[NUM_POSITIVE] += 1.0
            };
            if NEGATIVE_AA_IDXS.contains(&x) {
                embedding[NUM_NEGATIVE] += 1.0
            };
            if TINY_AA_IDXS.contains(&x) {
                embedding[NUM_TINY] += 1.0
            };
            if BRANCHED_AA_IDXS.contains(&x) {
                embedding[NUM_BRANCHED] += 1.0
            };
        }

        for idx in 0..VALID_AA.len() {
            let pct_val = embedding[idx] / pep_length;
            embedding[PCT_FEATURES_START + idx] = pct_val;
        }

        let charge_feature: f64 = *charge as f64;
        embedding[PEPTIDE_CHARGE] = charge_feature;
        embedding[INV_PEPTIDE_CHARGE] = 1. / charge_feature;
        embedding[PEPTIDE_LEN] = peptide.sequence.len() as f64;
        embedding[PEPTIDE_MASS] = (peptide.monoisotopic as f64) / 1000.0;
        embedding[PEPTIDE_MZ] = ((peptide.monoisotopic as f64) / charge_feature) / 1000.0;
        embedding[INTERCEPT] = 1.0;
        embedding
    }

    /// Fit a linear ion-mobility regression model from peptide-derived features.
    pub fn fit(
        db: &IndexedDatabase,
        training_set: &[FeatureCore],
        filter: impl Fn(&FeatureCore) -> bool + Sync + Send,
    ) -> Option<Self> {
        let mut map = [0; 26];
        for (idx, aa) in VALID_AA.iter().enumerate() {
            map[(aa - b'A') as usize] = idx;
        }

        let training_n = training_set
            .par_iter()
            .filter(|feat| filter(feat) && feat.ims > 0.0)
            .count();
        if training_n < 10 {
            log::warn!(
                "Not enough high-quality IMS PSMs ({}) to train the ion mobility model.",
                training_n
            );
            return None;
        }

        let lr = LinearRegression::fit::<_, FEATURES>(
            training_set,
            |feat| filter(feat) && feat.ims > 0.0,
            |psm| Self::embed(&db[psm.peptide_idx], &psm.charge, &map),
            |psm| psm.ims as f64,
        )
        .or_else(|| {
            log::warn!("Mobility model training aborted: singular fit or invalid IMS variance");
            None
        })?;

        log::info!("- fit mobility model, rsq = {}, mse = {}", lr.r2, lr.mse);
        Some(Self {
            beta: lr.beta,
            map,
            r2: lr.r2,
            mse: lr.mse,
            training_n: lr.training_n,
        })
    }

    /// Vanilla-compatible fit preserving the upstream training behavior:
    /// no ims > 0 filter, no minimum-count check, and no zero-variance abort.
    pub fn fit_vanilla_compat(
        db: &IndexedDatabase,
        training_set: &[FeatureCore],
        filter: impl Fn(&FeatureCore) -> bool + Sync + Send,
    ) -> Option<Self> {
        let mut map = [0; 26];
        for (idx, aa) in VALID_AA.iter().enumerate() {
            map[(aa - b'A') as usize] = idx;
        }

        let lr = LinearRegression::fit_vanilla_compat::<_, FEATURES>(
            training_set,
            |feat| filter(feat),
            |psm| Self::embed(&db[psm.peptide_idx], &psm.charge, &map),
            |psm| psm.ims as f64,
        )?;

        log::info!("- fit mobility model, rsq = {}, mse = {}", lr.r2, lr.mse);
        Some(Self {
            beta: lr.beta,
            map,
            r2: lr.r2,
            mse: lr.mse,
            training_n: lr.training_n,
        })
    }

    pub fn predict_peptide(&self, db: &IndexedDatabase, psm: &FeatureCore) -> f64 {
        let v = Self::embed(&db[psm.peptide_idx], &psm.charge, &self.map);
        v.into_iter()
            .zip(&self.beta)
            .fold(0.0f64, |sum, (x, y)| sum + x * y)
    }
}
