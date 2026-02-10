//! TMT quantification
#![allow(clippy::excessive_precision)]
#![allow(unused_imports)]
use crate::database::binary_search_slice;
use crate::ion_series::{IonSeries, Kind};
use crate::mass::{Tolerance, H2O, NH3, PROTON};
use crate::peptide::Peptide;
use crate::scoring::Scorer;
use crate::spectrum::{self, Peak, Precursor, ProcessedSpectrum};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Isobaric {
    Tmt6,
    Tmt10,
    Tmt11,
    Tmt16,
    Tmt18,
    User(Vec<f32>),
}

impl Isobaric {
    /// Return the monoisotopic mass of reporter ions
    pub fn reporter_masses(&self) -> &[f32] {
        match self {
            Isobaric::Tmt6 => &TMT6PLEX,
            Isobaric::Tmt10 => &TMT11PLEX[0..10],
            Isobaric::Tmt11 => &TMT11PLEX,
            Isobaric::Tmt16 => &TMT18PLEX[0..16],
            Isobaric::Tmt18 => &TMT18PLEX,
            Isobaric::User(labels) => labels,
        }
    }

    /// Return the monoisotopic mass of tag
    pub fn modification_mass(&self) -> Option<f32> {
        match self {
            Isobaric::Tmt6 | Isobaric::Tmt10 | Isobaric::Tmt11 => Some(229.162932),
            Isobaric::Tmt16 => Some(304.2071),
            Isobaric::Tmt18 => Some(304.2135),
            Isobaric::User(_) => None,
        }
    }

    /// Return a column name for each tag
    pub fn headers(&self) -> Vec<String> {
        match self {
            Isobaric::User(v) => v
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("user_{}", idx + 1))
                .collect(),
            _ => self
                .reporter_masses()
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("tmt_{}", idx + 1))
                .collect(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Purity {
    pub ratio: f32,
    pub correct_precursors: usize,
    pub incorrect_precursors: usize,
}

#[derive(Debug)]
pub struct Quant<'ms3> {
    pub hit_purity: Purity,
    pub chimera_purity: Option<Purity>,
    pub intensities: Vec<Option<&'ms3 Peak>>,
    pub spectrum: &'ms3 ProcessedSpectrum<Peak>,
}

pub fn find_reporter_ions<'a>(
    peaks: &'a [Peak],
    labels: &[f32],
    label_tolerance: Tolerance,
) -> Vec<Option<&'a Peak>> {
    labels
        .iter()
        .map(|&label| {
            spectrum::select_most_intense_peak(peaks, label, label_tolerance, Some(-PROTON))
        })
        .collect()
}

const TMT6PLEX: [f32; 6] = [
    126.127726, 127.124761, 128.134436, 129.131471, 130.141145, 131.138180,
];

const TMT11PLEX: [f32; 11] = [
    126.127726, 127.124761, 127.131081, 128.128116, 128.134436, 129.131471, 129.137790, 130.134825,
    130.141145, 131.138180, 131.144499,
];

const TMT18PLEX: [f32; 18] = [
    126.127726, 127.124761, 127.131081, 128.128116, 128.134436, 129.131471, 129.137790, 130.134825,
    130.141145, 131.138180, 131.144500, 132.141535, 132.147855, 133.144890, 133.151210, 134.148245,
    134.154565, 135.15160,
];

#[derive(Clone)]
pub struct TmtQuant {
    pub spec_id: String,
    pub file_id: usize,
    pub ion_injection_time: f32,
    pub peaks: Vec<f32>,
}

pub fn quantify(
    spectra: &[ProcessedSpectrum<Peak>],
    isobaric_labels: &Isobaric,
    isobaric_tolerance: Tolerance,
    level: u8,
) -> Vec<TmtQuant> {
    spectra
        .par_iter()
        .filter(|spectrum| spectrum.level == level)
        .filter_map(|spectrum| {
            let spec_id = match level {
                1 => return None,
                2 => spectrum.id.clone(),
                _ => spectrum
                    .precursors
                    .first()
                    .and_then(|precursor| precursor.spectrum_ref.clone())
                    .unwrap_or_default(),
            };

            let peaks = find_reporter_ions(
                &spectrum.peaks,
                isobaric_labels.reporter_masses(),
                isobaric_tolerance,
            )
            .into_iter()
            .map(|peak| peak.map(|p| p.intensity).unwrap_or_default())
            .collect();

            Some(TmtQuant {
                spec_id,
                file_id: spectrum.file_id,
                ion_injection_time: spectrum.ion_injection_time,
                peaks,
            })
        })
        .collect()
}
