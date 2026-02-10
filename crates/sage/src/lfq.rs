use crate::database::{binary_search_slice, IndexedDatabase, PeptideIx};
use crate::mass::{composition, Composition, Tolerance, NEUTRON};
use crate::ml::{matrix::Matrix, retention_alignment::Alignment};
use crate::scoring::{DfFeature, FeatureCore, TdcFeature};
use crate::spectrum::MS1Spectra;
use dashmap::DashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimum normalized spectral angle required to integrate a peak
// const MIN_SPECTRAL_ANGLE: f64 = 0.70;
/// Retention time tolerance, in fraction of total run length, to search for
/// precursor ions
const RT_TOL: f32 = 0.0050;
/// Width of gaussian kernel used for smoothing intensities
const K_WIDTH: usize = 10;
/// Mass tolerance, in ppm, to seach for precursor ions
// const PPM_TOL: f32 = 5.0;
/// Number of equally spaced bins that will be used to integrate ions in (-RT_TOL, +RT_TOL)
const GRID_SIZE: usize = 100;
/// Number of isotopes to search for
const N_ISOTOPES: usize = 3;

// --- Decoy-free LFQ shadow-null parameters (target-derived off-target sampling) ---
// We generate a SINGLE shadow trace per (peptide, charge, isotope) but the shift is
// *deterministic and jittered* so it spreads across m/z space and RT, avoiding the
// pathological fixed-shift collisions of a constant offset.
const SHADOW_SHIFT_MIN_DA: f32 = 4.5; // Da (before /charge)
const SHADOW_SHIFT_MAX_DA: f32 = 25.0; // Da (before /charge)
const SHADOW_RT_JITTER_FRAC: f32 = 1.25; // * RT_TOL, jitter in [-1.25*RT_TOL, +1.25*RT_TOL]

#[inline]
fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Deterministic pseudo-random in [0,1).
#[inline]
fn u01_from_u64(seed: u64) -> f32 {
    // take top 24 bits -> [0, 2^24) then normalize
    let v = (seed >> 40) as u32;
    (v as f32) / (u32::MAX as f32)
}

/// Deterministic shadow mass shift (Da, before /charge) for a given precursor identity.
/// Returned value is in [SHADOW_SHIFT_MIN_DA, SHADOW_SHIFT_MAX_DA].
#[inline]
fn shadow_shift_da(peptide: PeptideIx, charge: u8, isotope: usize) -> f32 {
    let seed = ((peptide.0 as u64) << 32)
        ^ ((charge as u64) << 16)
        ^ (isotope as u64)
        ^ 0x9E37_79B9_7F4A_7C15u64;
    let r = u01_from_u64(xorshift64(seed));
    SHADOW_SHIFT_MIN_DA + (SHADOW_SHIFT_MAX_DA - SHADOW_SHIFT_MIN_DA) * r
}

/// Deterministic RT jitter in [-SHADOW_RT_JITTER_FRAC*RT_TOL, +SHADOW_RT_JITTER_FRAC*RT_TOL]
#[inline]
fn shadow_rt_jitter(rt: f32, peptide: PeptideIx, charge: u8, isotope: usize) -> f32 {
    let seed = ((peptide.0 as u64) << 32)
        ^ ((charge as u64) << 16)
        ^ (isotope as u64)
        ^ 0xD1B5_4A32_D192_ED03u64;
    let r = u01_from_u64(xorshift64(seed)); // [0,1)
    let jitter = (2.0 * r - 1.0) * (SHADOW_RT_JITTER_FRAC * RT_TOL);
    (rt + jitter).max(0.0)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeakScoringStrategy {
    RetentionTime,
    SpectralAngle,
    Intensity,
    Hybrid,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum IntegrationStrategy {
    Apex,
    Sum,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrecursorId {
    Combined(PeptideIx),
    Charged((PeptideIx, u8)),
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct LfqSettings {
    pub peak_scoring: PeakScoringStrategy,
    pub integration: IntegrationStrategy,
    pub spectral_angle: f64,
    pub ppm_tolerance: f32,
    pub mobility_pct_tolerance: f32,
    pub combine_charge_states: bool,
    pub peptide_q_value: f32,
}

impl Default for LfqSettings {
    fn default() -> Self {
        Self {
            peak_scoring: PeakScoringStrategy::Hybrid,
            integration: IntegrationStrategy::Sum,
            spectral_angle: 0.70,
            ppm_tolerance: 5.0,
            mobility_pct_tolerance: 1.0,
            combine_charge_states: true,
            peptide_q_value: 0.01,
        }
    }
}

// --- Trait for features that can be quantified ---
pub trait Quantifiable: Sync + Send {
    fn core(&self) -> &FeatureCore;
    fn passes_filter(&self, settings: &LfqSettings) -> bool;
}

impl Quantifiable for TdcFeature {
    fn core(&self) -> &FeatureCore {
        &self.core
    }
    fn passes_filter(&self, settings: &LfqSettings) -> bool {
        self.peptide_q <= settings.peptide_q_value && self.core.label == 1
    }
}

impl Quantifiable for DfFeature {
    fn core(&self) -> &FeatureCore {
        &self.core
    }
    fn passes_filter(&self, _settings: &LfqSettings) -> bool {
        self.core.rank == 1
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PrecursorRange {
    pub rt: f32,
    pub mass_lo: f32,
    pub mass_hi: f32,
    pub mobility_lo: f32,
    pub mobility_hi: f32,
    pub charge: u8,
    pub isotope: usize,
    pub peptide: PeptideIx,
    pub file_id: usize,
    pub decoy: bool,
}

pub struct FeatureMap {
    pub ranges: Vec<PrecursorRange>,
    pub min_rts: Vec<f32>,
    pub bin_size: usize,
    pub settings: LfqSettings,
}

pub fn build_feature_map<F: Quantifiable>(
    settings: LfqSettings,
    precursor_charge: (u8, u8),
    features: &[F],
    decoy_free: bool,
) -> FeatureMap {
    let map: DashMap<PeptideIx, PrecursorRange, fnv::FnvBuildHasher> = DashMap::default();

    // 1. Filter and insert Targets
    features
        .iter()
        .filter(|feat| feat.passes_filter(&settings))
        .for_each(|feat| {
            if !map.contains_key(&feat.core().peptide_idx) {
                let core = feat.core();
                let (mobility_lo, mobility_hi) = Tolerance::Pct(
                    -settings.mobility_pct_tolerance,
                    settings.mobility_pct_tolerance,
                )
                .bounds(core.ims);
                map.insert(
                    core.peptide_idx,
                    PrecursorRange {
                        rt: core.aligned_rt,
                        mass_lo: core.calcmass,
                        mass_hi: 0.0, // This is ignored during lookup generation
                        peptide: core.peptide_idx,
                        charge: core.charge,
                        isotope: 0,
                        file_id: core.file_id,
                        mobility_lo,
                        mobility_hi,
                        decoy: false,
                    },
                );
            }
        });

    // 2. Expand into (Target) and optionally (Shadow/Decoy) ranges
    let mut ranges = map
        .into_par_iter()
        .flat_map_iter(|(_, range)| {
            (precursor_charge.0..=precursor_charge.1).flat_map(move |charge| {
                (0..N_ISOTOPES).flat_map(move |isotope| {
                    let mass = (range.mass_lo + isotope as f32 * NEUTRON) / charge as f32;
                    let (mass_lo, mass_hi) =
                        Tolerance::Ppm(-settings.ppm_tolerance, settings.ppm_tolerance)
                            .bounds(mass);

                    // A. The Real Target
                    let fwd = PrecursorRange {
                        mass_lo,
                        mass_hi,
                        charge,
                        isotope,
                        decoy: false,
                        ..range
                    };

                    // B. The Decoy / Shadow
                    let rev = if decoy_free {
                        // --- TARGET-DERIVED NULL (off-target / shadow sampling) ---
                        // Deterministic per-precursor (peptide, charge, isotope) jitter:
                        // - mass is shifted by a per-precursor Da offset (then /charge already applied via `mass`)
                        // - RT is jittered within ~±RT_TOL to decorrelate from true apex
                        let shift_da = shadow_shift_da(range.peptide, charge, isotope);
                        let shift_mz = shift_da / charge as f32;

                        let (shadow_lo, shadow_hi) =
                            Tolerance::Ppm(-settings.ppm_tolerance, settings.ppm_tolerance)
                                .bounds(mass + shift_mz);

                        PrecursorRange {
                            rt: shadow_rt_jitter(fwd.rt, range.peptide, charge, isotope),
                            mass_lo: shadow_lo,
                            mass_hi: shadow_hi,
                            decoy: true,
                            ..fwd
                        }
                    } else {
                        // --- STANDARD TARGET-DECOY LOGIC ---
                        let (mass_lo, mass_hi) =
                            Tolerance::Ppm(-settings.ppm_tolerance, settings.ppm_tolerance)
                                .bounds(mass + 11.06);

                        PrecursorRange {
                            rt: (fwd.rt - RT_TOL * 2.0).max(0.0),
                            mass_lo,
                            mass_hi,
                            decoy: true,
                            ..fwd
                        }
                    };

                    [fwd, rev]
                })
            })
        })
        .collect::<Vec<_>>();

    ranges.par_sort_unstable_by(|a, b| a.rt.total_cmp(&b.rt));
    let min_rts = ranges
        .par_chunks_mut(16 * 1024)
        .map(|chunk| {
            let min = chunk[0].rt;
            chunk.par_sort_unstable_by(|a, b| a.mass_lo.total_cmp(&b.mass_lo));
            min
        })
        .collect::<Vec<_>>();

    log::trace!("building feature map");
    FeatureMap {
        ranges,
        min_rts,
        bin_size: 16 * 1024,
        settings,
    }
}

struct Query<'a> {
    ranges: &'a [PrecursorRange],
    page_lo: usize,
    page_hi: usize,
    bin_size: usize,
    min_rt: f32,
    max_rt: f32,
}

impl FeatureMap {
    fn rt_slice(&self, rt: f32, rt_tol: f32) -> Query<'_> {
        let (page_lo, page_hi) = binary_search_slice(
            &self.min_rts,
            |rt, x| rt.total_cmp(x),
            rt - rt_tol,
            rt + rt_tol,
        );

        Query {
            ranges: &self.ranges,
            page_lo,
            page_hi,
            bin_size: self.bin_size,
            max_rt: rt + rt_tol,
            min_rt: rt - rt_tol,
        }
    }
}

impl FeatureMap {
    pub fn quantify(
        &self,
        db: &IndexedDatabase,
        spectra: &MS1Spectra,
        alignments: &[Alignment],
    ) -> HashMap<(PrecursorId, bool), (Peak, Vec<f64>), fnv::FnvBuildHasher> {
        let scores: DashMap<(PrecursorId, bool), Grid, fnv::FnvBuildHasher> = DashMap::default();

        log::info!("tracing MS1 features");

        match spectra {
            MS1Spectra::NoMobility(spectra) => spectra.par_iter().for_each(|spectrum| {
                let a = alignments[spectrum.file_id];
                let rt = (spectrum.scan_start_time / a.max_rt) * a.slope + a.intercept;
                let query = self.rt_slice(rt, RT_TOL);

                for peak in &spectrum.peaks {
                    for entry in query.mass_lookup(peak.mass) {
                        let id = match self.settings.combine_charge_states {
                            true => PrecursorId::Combined(entry.peptide),
                            false => PrecursorId::Charged((entry.peptide, entry.charge)),
                        };

                        let mut grid = scores.entry((id, entry.decoy)).or_insert_with(|| {
                            let p = &db[entry.peptide];
                            let composition = p
                                .sequence
                                .iter()
                                .map(|r| composition(*r))
                                .sum::<Composition>();
                            let dist = crate::isotopes::peptide_isotopes(
                                composition.carbon,
                                composition.sulfur,
                            );
                            Grid::new(entry, RT_TOL, dist, alignments.len(), GRID_SIZE)
                        });

                        grid.add_entry(rt, entry.isotope, spectrum.file_id, peak.intensity);
                    }
                }
            }),
            MS1Spectra::WithMobility(spectra) => spectra.par_iter().for_each(|spectrum| {
                let a = alignments[spectrum.file_id];
                let rt = (spectrum.scan_start_time / a.max_rt) * a.slope + a.intercept;
                let query = self.rt_slice(rt, RT_TOL);

                for peak in &spectrum.peaks {
                    for entry in query.mass_mobility_lookup(peak.mass, peak.mobility) {
                        let id = match self.settings.combine_charge_states {
                            true => PrecursorId::Combined(entry.peptide),
                            false => PrecursorId::Charged((entry.peptide, entry.charge)),
                        };

                        let mut grid = scores.entry((id, entry.decoy)).or_insert_with(|| {
                            let p = &db[entry.peptide];
                            let composition = p
                                .sequence
                                .iter()
                                .map(|r| composition(*r))
                                .sum::<Composition>();
                            let dist = crate::isotopes::peptide_isotopes(
                                composition.carbon,
                                composition.sulfur,
                            );
                            Grid::new(entry, RT_TOL, dist, alignments.len(), GRID_SIZE)
                        });

                        grid.add_entry(rt, entry.isotope, spectrum.file_id, peak.intensity);
                    }
                }
            }),
            MS1Spectra::Empty => {
                log::warn!("no MS1 spectra found for quantification");
            }
        };

        log::info!("integrating MS1 features");

        scores
            .into_par_iter()
            .filter_map(|(peptide_ix, mut grid)| {
                let mut traces = grid.summarize_traces();
                let (peak, data) = traces.integrate(&self.settings)?;
                Some((peptide_ix, (peak, data)))
            })
            .collect::<HashMap<_, _, _>>()
    }
}

pub struct Grid {
    rt_min: f32,
    rt_step: f32,
    files: usize,
    reference_file_id: usize,
    pub distribution: [f32; N_ISOTOPES],
    pub matrix: Matrix,
}

pub struct Traces {
    pub dot_product: Matrix,
    pub spectral_angle: Matrix,
    reference_file_id: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Peak {
    pub rt: usize,
    pub spectral_angle: f64,
    pub score: f64,
    pub q_value: f32,
}

impl Traces {
    fn warp(&mut self) {
        let time_warps = self.find_time_warps(&self.dot_product, 75);
        Self::apply_time_warps(&mut self.spectral_angle, &time_warps);
        Self::apply_time_warps(&mut self.dot_product, &time_warps);
    }

    pub fn find_time_warps(&self, matrix: &Matrix, slack: isize) -> Vec<isize> {
        let reference = matrix.row_slice(self.reference_file_id);
        let mut offsets = vec![0; matrix.rows];

        for (row, offset) in offsets.iter_mut().enumerate() {
            let run = matrix.row_slice(row);
            let mut best_offset = (0, 0.0);
            for offset in -slack..=slack {
                let mut dot = 0.0;
                for (i, ref_int) in reference.iter().enumerate() {
                    let j = i as isize + offset;
                    if j >= 0 && j < run.len() as isize {
                        dot += ref_int * run[j as usize];
                    }
                }
                if dot >= best_offset.1 {
                    best_offset = (offset, dot);
                }
            }
            *offset = best_offset.0;
        }
        offsets
    }

    fn apply_time_warps(matrix: &mut Matrix, time_warps: &[isize]) {
        for (row, warp) in time_warps.iter().enumerate() {
            let run = matrix.row_slice_mut(row);
            let mut shifted = vec![0.0; run.len()];
            for (i, val) in shifted.iter_mut().enumerate() {
                let j = i as isize + warp;
                if j >= 0 && j < run.len() as isize {
                    *val = run[j as usize];
                }
            }
            run.copy_from_slice(&shifted);
        }
    }

    pub fn scores(&self, strategy: PeakScoringStrategy) -> (Vec<f64>, Vec<f64>) {
        let mut spectral = Vec::with_capacity(self.spectral_angle.cols);
        let mut intensity = Vec::with_capacity(self.spectral_angle.cols);
        let mut max = 0.0f64;
        for col in 0..self.spectral_angle.cols {
            let mut summed_int = 1.0;
            let mut weighted = 0.0;
            for (sa, dotp) in self.spectral_angle.col(col).zip(self.dot_product.col(col)) {
                weighted += sa * dotp;
                summed_int += dotp;
            }
            spectral.push(weighted / summed_int);
            intensity.push(summed_int);
            max = max.max(summed_int);
        }

        let center = self.spectral_angle.cols as isize / 2;
        let scores = spectral
            .iter()
            .zip(intensity.iter())
            .enumerate()
            .map(|(rt, (s, i))| match strategy {
                PeakScoringStrategy::RetentionTime => {
                    (1.0 - ((rt as isize - center).abs() as f64 / center as f64)).powf(0.33)
                }
                PeakScoringStrategy::SpectralAngle => *s,
                PeakScoringStrategy::Intensity => (*i / max).sqrt(),
                PeakScoringStrategy::Hybrid => {
                    let rt = 1.0 - ((rt as isize - center).abs() as f64 / center as f64);
                    s.powi(3) * rt.powf(0.33) * (*i / max).sqrt()
                }
            })
            .collect();
        (scores, spectral)
    }

    pub fn integrate(&mut self, settings: &LfqSettings) -> Option<(Peak, Vec<f64>)> {
        self.warp();

        let (scores, spectral) = self.scores(settings.peak_scoring);
        let mut best = Peak::default();
        for (rt, s) in scores.iter().enumerate() {
            if *s > best.score && spectral[rt] >= settings.spectral_angle {
                best.score = *s;
                best.rt = rt;
            }
        }

        if best.score == 0.0 {
            return None;
        }

        let mut left = best.rt.saturating_sub(1);
        let mut right = best.rt.saturating_add(1);
        let threshold = best.score * 0.50;

        while left > best.rt.saturating_sub(scores.len() / 5)
            && scores[left] >= threshold
            && spectral[left] >= settings.spectral_angle
        {
            left -= 1;
        }

        while right < scores.len().saturating_sub(1).min(best.rt + 20)
            && scores[right] >= threshold
            && spectral[right] >= settings.spectral_angle
        {
            right += 1;
        }

        let mut areas = Vec::with_capacity(self.dot_product.rows);
        for file in 0..self.dot_product.rows {
            let area = match settings.integration {
                IntegrationStrategy::Sum => self.dot_product.row_slice(file)[left..right]
                    .iter()
                    .sum::<f64>(),
                IntegrationStrategy::Apex => self.dot_product.row_slice(file)[best.rt],
            };
            areas.push(area);
        }

        let mut summed_int = 1.0;
        let mut weighted = 0.0;
        for (sa, dotp) in self
            .spectral_angle
            .col(best.rt)
            .zip(self.dot_product.col(best.rt))
        {
            weighted += sa * dotp;
            summed_int += dotp;
        }
        best.spectral_angle = weighted / summed_int;
        Some((best, areas))
    }
}

impl Grid {
    pub fn new(
        entry: &PrecursorRange,
        rt_tol: f32,
        distribution: [f32; N_ISOTOPES],
        files: usize,
        grid_size: usize,
    ) -> Grid {
        let matrix = Matrix::new(
            vec![0.0; grid_size * files * N_ISOTOPES],
            files * N_ISOTOPES,
            grid_size,
        );
        let rt_step = (rt_tol * 2.0) / (grid_size) as f32;

        Grid {
            rt_min: entry.rt - rt_tol,
            rt_step,
            distribution,
            matrix,
            files,
            reference_file_id: entry.file_id,
        }
    }

    pub fn add_entry(&mut self, spectrum_rt: f32, isotope: usize, file_id: usize, intensity: f32) {
        let bin_lo = ((spectrum_rt - self.rt_min) / self.rt_step).floor() as usize;
        let bin_lo = bin_lo.min(self.matrix.cols - 1);
        let bin_hi = (bin_lo + 1).min(self.matrix.cols - 1);

        let bin_lo_rt = bin_lo as f32 * self.rt_step + self.rt_min;
        let interp = (spectrum_rt - bin_lo_rt) / self.rt_step;

        self.matrix[(file_id * N_ISOTOPES + isotope, bin_lo)] +=
            ((1.0 - interp) * intensity) as f64;
        self.matrix[(file_id * N_ISOTOPES + isotope, bin_hi)] += (interp * intensity) as f64;
    }

    pub fn summarize_traces(&mut self) -> Traces {
        let k = gaussian_kernel(0.5, K_WIDTH);

        let mut spectral_angle = Matrix::new(
            vec![0.0; self.files * self.matrix.cols],
            self.files,
            self.matrix.cols,
        );

        let mut dot_product = spectral_angle.clone();
        let ss_dist = self
            .distribution
            .iter()
            .map(|x| x.powi(2))
            .sum::<f32>()
            .sqrt() as f64;

        for file in 0..self.files {
            let mut summed_squared_intensities = vec![0.0; self.matrix.cols];
            for isotope in 0..N_ISOTOPES {
                let convolved = convolve(self.matrix.row_slice(file * N_ISOTOPES + isotope), &k);
                for (col, intensity) in convolved.iter().enumerate() {
                    spectral_angle[(file, col)] += intensity * self.distribution[isotope] as f64;
                    summed_squared_intensities[col] += intensity.powi(2);
                }
                self.matrix
                    .row_slice_mut(file * N_ISOTOPES + isotope)
                    .copy_from_slice(&convolved);
            }

            for (col, ss) in summed_squared_intensities.iter().enumerate() {
                let dot = spectral_angle[(file, col)];
                let similarity = if *ss > 0.0 {
                    dot / (ss.sqrt() * ss_dist)
                } else {
                    0.0
                };
                spectral_angle[(file, col)] = 1.0 - 2.0 * similarity.acos() / std::f64::consts::PI;
                dot_product[(file, col)] = dot;
            }
        }

        Traces {
            dot_product,
            spectral_angle,
            reference_file_id: self.reference_file_id,
        }
    }
}

fn gaussian_kernel(sigma: f64, len: usize) -> Vec<f64> {
    let step = 2.0 / (len - 1) as f64;
    let constant = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());

    let mut kernel = (0..len)
        .map(|i| {
            let x = i as f64 * step - 1.0;
            constant * (-0.5 * (x / sigma).powi(2)).exp()
        })
        .collect::<Vec<_>>();

    let sum = kernel.iter().sum::<f64>();
    kernel.iter_mut().for_each(|x| *x /= sum);
    kernel
}

fn convolve(slice: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = kernel.len() - (kernel.len() / 2);
    (0..slice.len())
        .map(|idx| {
            let k = &kernel[kernel.len().saturating_sub(n + idx)..];
            let w = &slice[idx.saturating_sub(n - 1)..];
            w.iter().zip(k).fold(0.0, |acc, (x, y)| acc + x * y)
        })
        .collect()
}

impl Query<'_> {
    pub fn mass_lookup(&self, mass: f32) -> impl Iterator<Item = &PrecursorRange> {
        (self.page_lo..self.page_hi).flat_map(move |page| {
            let left_idx = page * self.bin_size;
            let right_idx = (left_idx + self.bin_size).min(self.ranges.len());
            let slice = &self.ranges[left_idx..right_idx];

            let (inner_left, inner_right) = binary_search_slice(
                slice,
                |frag, bounds| frag.mass_lo.total_cmp(bounds),
                mass - 0.1,
                mass + 0.1,
            );

            slice[inner_left..inner_right].iter().filter(move |frag| {
                frag.rt <= self.max_rt
                    && frag.rt >= self.min_rt
                    && mass >= frag.mass_lo
                    && mass <= frag.mass_hi
            })
        })
    }

    pub fn mass_mobility_lookup(
        &self,
        mass: f32,
        mobility: f32,
    ) -> impl Iterator<Item = &PrecursorRange> {
        self.mass_lookup(mass).filter(move |precursor| {
            (precursor.mobility_hi >= mobility) && (precursor.mobility_lo <= mobility)
        })
    }
}
