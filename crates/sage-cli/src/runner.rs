use super::input::Search;
use super::output::SageResults;
use super::telemetry;
use anyhow::Context;
use csv::ByteRecord;
use log::info;
use rayon::prelude::*;
use sage_cloudpath::{CloudPath, FileFormat};
use sage_core::database::{IndexedDatabase, Parameters, PeptideIx};
use sage_core::fasta::Fasta;
use sage_core::input::FdrMode;
use sage_core::ion_series::Kind;
use sage_core::lfq::{Peak, PrecursorId};
use sage_core::mass::Tolerance;
use sage_core::ml::linear_discriminant::score_psms;
use sage_core::peptide::Peptide;
use sage_core::scoring::Fragments;
use sage_core::scoring::{DfFeature, FeatureCore, Scorer, TdcFeature};
use sage_core::spectrum::{MS1Spectra, ProcessedSpectrum, RawSpectrum, SpectrumProcessor};
use sage_core::tmt::TmtQuant;
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Instant;

pub struct Runner {
    pub database: IndexedDatabase,
    pub parameters: Search,
    pub start: Instant,
    pub decoy_free_mode: bool,
}

#[derive(Default)]
struct RawSpectrumAccumulator {
    pub ms1: Vec<RawSpectrum>,
    pub msn: Vec<RawSpectrum>,
}

impl RawSpectrumAccumulator {
    pub fn fold_op(mut self, rhs: RawSpectrum) -> Self {
        if rhs.ms_level == 1 {
            self.ms1.push(rhs);
        } else {
            self.msn.push(rhs);
        }
        self
    }

    pub fn reduce(mut self, other: Self) -> Self {
        self.ms1.extend(other.ms1);
        self.msn.extend(other.msn);
        self
    }
}

impl FromParallelIterator<RawSpectrum> for RawSpectrumAccumulator {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = RawSpectrum>,
    {
        par_iter
            .into_par_iter()
            .fold(
                RawSpectrumAccumulator::default,
                RawSpectrumAccumulator::fold_op,
            )
            .reduce(
                RawSpectrumAccumulator::default,
                RawSpectrumAccumulator::reduce,
            )
    }
}

impl FromIterator<RawSpectrum> for RawSpectrumAccumulator {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = RawSpectrum>,
    {
        iter.into_iter().fold(
            RawSpectrumAccumulator::default(),
            RawSpectrumAccumulator::fold_op,
        )
    }
}

impl Runner {
    pub fn new(parameters: Search, parallel: usize) -> anyhow::Result<Self> {
        let mut parameters = parameters.clone();
        let start = Instant::now();

        let decoy_free_mode = matches!(parameters.fdr.mode, FdrMode::DecoyFree);
        log::info!(
            "FDR mode at runtime: {:?} (decoy_free_mode = {})",
            parameters.fdr.mode,
            decoy_free_mode
        );

        let fasta = sage_cloudpath::util::read_fasta(
            &parameters.database.fasta,
            &parameters.database.decoy_tag,
            false,
        )
        .with_context(|| {
            format!(
                "Failed to build database from `{}`",
                parameters.database.fasta
            )
        })?;

        if decoy_free_mode && parameters.report_psms < 10 {
            log::warn!("decoy_free mode requires report_psms >= 10; overriding to 10");
            parameters.report_psms = 10;
        }

        let generate_decoys = parameters.database.generate_decoys;
        if decoy_free_mode && generate_decoys {
            log::warn!(
                "fdr.mode=decoy_free but database.generate_decoys=true; decoys will be searched but ignored \
                 by decoy-free FDR. For speed, set database.generate_decoys=false."
            );
        }

        let database = match parameters.database.prefilter {
            false => {
                let fasta_for_build = sage_cloudpath::util::read_fasta(
                    &parameters.database.fasta,
                    &parameters.database.decoy_tag,
                    generate_decoys,
                )?;
                parameters.database.clone().build(fasta_for_build)
            }
            true => {
                parameters
                    .database
                    .auto_calculate_prefilter_chunk_size(&fasta);
                if parameters.database.prefilter_chunk_size >= fasta.targets.len() {
                    parameters.database.clone().build(fasta)
                } else {
                    info!(
                        "using {} db chunks of size {}",
                        (fasta.targets.len() + parameters.database.prefilter_chunk_size - 1)
                            / parameters.database.prefilter_chunk_size,
                        parameters.database.prefilter_chunk_size,
                    );
                    let mini_runner = Self {
                        database: IndexedDatabase::default(),
                        parameters: parameters.clone(),
                        start,
                        decoy_free_mode,
                    };
                    let peptides = mini_runner.prefilter_peptides(parallel, fasta);
                    parameters.database.clone().build_from_peptides(peptides)
                }
            }
        };

        info!(
            "generated {} fragments, {} peptides in {:#?}",
            database.fragments.len(),
            database.peptides.len(),
            (start.elapsed())
        );

        Ok(Self {
            database,
            parameters,
            start,
            decoy_free_mode,
        })
    }

    pub fn prefilter_peptides(self, parallel: usize, fasta: Fasta) -> Vec<Peptide> {
        let spectra: Option<(
            MS1Spectra,
            Vec<ProcessedSpectrum<sage_core::spectrum::Peak>>,
        )> = match parallel >= self.parameters.mzml_paths.len() {
            true => Some(self.read_processed_spectra(&self.parameters.mzml_paths, 0, 0)),
            false => None,
        };
        let mut all_peptides: Vec<Peptide> = fasta
            .iter_chunks(self.parameters.database.prefilter_chunk_size)
            .enumerate()
            .flat_map(|(chunk_id, fasta_chunk)| {
                let start = Instant::now();
                info!("pre-filtering fasta chunk {}", chunk_id,);
                let db = &self.parameters.database.clone().build(fasta_chunk);
                info!(
                    "generated {} fragments, {} peptides in {}ms",
                    db.fragments.len(),
                    db.peptides.len(),
                    (Instant::now() - start).as_millis()
                );
                let scorer = Scorer {
                    db,
                    precursor_tol: self.parameters.precursor_tol,
                    fragment_tol: self.parameters.fragment_tol,
                    min_matched_peaks: self.parameters.min_matched_peaks,
                    min_isotope_err: self.parameters.isotope_errors.0,
                    max_isotope_err: self.parameters.isotope_errors.1,
                    min_precursor_charge: self.parameters.precursor_charge.0,
                    max_precursor_charge: self.parameters.precursor_charge.1,
                    override_precursor_charge: self.parameters.override_precursor_charge,
                    max_fragment_charge: self.parameters.max_fragment_charge,
                    chimera: self.parameters.chimera,
                    report_psms: self.parameters.report_psms + 1,
                    wide_window: self.parameters.wide_window,
                    annotate_matches: self.parameters.annotate_matches,
                    score_type: self.parameters.score_type,
                };
                let peptide_idxs: HashSet<PeptideIx> = match &spectra {
                    Some(spectra) => self.peptide_filter_processed_spectra(&scorer, &spectra.1),
                    None => self
                        .parameters
                        .mzml_paths
                        .chunks(parallel)
                        .enumerate()
                        .flat_map(|(chunk_idx, chunk)| {
                            let spectra_chunk =
                                self.read_processed_spectra(chunk, chunk_idx, parallel);
                            self.peptide_filter_processed_spectra(&scorer, &spectra_chunk.1)
                        })
                        .collect(),
                }
                .into_iter()
                .collect();
                let peptides: Vec<Peptide> = peptide_idxs
                    .into_iter()
                    .map(|idx| db[idx].clone())
                    .collect();
                info!(
                    "found {} pre-filtered peptides for fasta chunk {}",
                    peptides.len(),
                    chunk_id,
                );
                peptides
            })
            .collect();
        Parameters::reorder_peptides(&mut all_peptides);
        all_peptides
    }

    fn peptide_filter_processed_spectra(
        &self,
        scorer: &Scorer,
        spectra: &Vec<ProcessedSpectrum<sage_core::spectrum::Peak>>,
    ) -> Vec<PeptideIx> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = AtomicUsize::new(0);
        let start = Instant::now();

        let peptide_idxs: Vec<_> = spectra
            .par_iter()
            .filter(|spec| spec.peaks.len() >= self.parameters.min_peaks && spec.level == 2)
            .map(|x| {
                let prev = counter.fetch_add(1, Ordering::Relaxed);
                if prev > 0 && prev % 10_000 == 0 {
                    let duration = Instant::now().duration_since(start).as_millis() as usize;

                    let rate = prev * 1000 / (duration + 1);
                    log::trace!("- searched {} spectra ({} spectra/s)", prev, rate);
                }
                x
            })
            .flat_map(|spec| {
                scorer.quick_score(spec, self.parameters.database.prefilter_low_memory)
            })
            .collect();

        let duration = Instant::now().duration_since(start).as_millis() as usize;
        let prev = counter.load(Ordering::Relaxed);
        let rate = prev * 1000 / (duration + 1);
        log::info!("- search:  {:8} ms ({} spectra/s)", duration, rate);
        peptide_idxs
    }

    /// Unified Spectrum FDR function (TDC Mode only)
    /// Runs LDA (score_psms) and calculates spectrum q-values.
    fn spectrum_fdr(&self, features: &mut [TdcFeature]) -> usize {
        // 1. Run Linear Discriminant Analysis (LDA)
        // score_psms must be updated to accept &mut [TdcFeature] in linear_discriminant.rs
        let score_res = score_psms(
            features,
            self.parameters.precursor_tol,
            self.decoy_free_mode,
        );

        if score_res.is_none() {
            log::warn!("linear model fitting failed, using heuristic score");
            features.par_iter_mut().for_each(|feat| {
                feat.discriminant_score =
                    (-feat.core.poisson as f32).ln_1p() + feat.core.longest_y_pct / 3.0
            });
            features
                .par_sort_unstable_by(|a, b| b.discriminant_score.total_cmp(&a.discriminant_score));
            return sage_core::ml::qvalue::spectrum_q_value(features);
        }

        // 2. Calculate Q-Values (TDC)
        features.par_sort_unstable_by(|a, b| b.discriminant_score.total_cmp(&a.discriminant_score));

        // spectrum_q_value must be updated to accept &mut [TdcFeature]
        return sage_core::ml::qvalue::spectrum_q_value(features);
    }

    fn make_path<S: AsRef<str>>(&self, file_name: S) -> CloudPath {
        let mut path = self.parameters.output_directory.clone();
        path.push(file_name);
        path
    }

    fn search_processed_spectra(
        &self,
        scorer: &Scorer,
        msn_spectra: &Vec<ProcessedSpectrum<sage_core::spectrum::Peak>>,
    ) -> Vec<FeatureCore> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = AtomicUsize::new(0);
        let start = Instant::now();

        let features: Vec<_> = msn_spectra
            .par_iter()
            .filter(|spec| spec.peaks.len() >= self.parameters.min_peaks && spec.level == 2)
            .map(|x| {
                let prev = counter.fetch_add(1, Ordering::Relaxed);
                if prev > 0 && prev % 10_000 == 0 {
                    let duration = Instant::now().duration_since(start).as_millis() as usize;

                    let rate = prev * 1000 / (duration + 1);
                    log::trace!("- searched {} spectra ({} spectra/s)", prev, rate);
                }
                x
            })
            .flat_map(|spec| scorer.score(spec))
            .collect();

        let duration = Instant::now().duration_since(start).as_millis() as usize;
        let prev = counter.load(Ordering::Relaxed);
        let rate = prev * 1000 / (duration + 1);
        log::info!("- search:  {:8} ms ({} spectra/s)", duration, rate);
        features
    }

    fn complete_features(
        &self,
        msn_spectra: Vec<ProcessedSpectrum<sage_core::spectrum::Peak>>,
        ms1_spectra: MS1Spectra,
        features: Vec<FeatureCore>,
    ) -> SageResults {
        let quant = self
            .parameters
            .quant
            .tmt
            .as_ref()
            .map(|isobaric| {
                let level = self.parameters.quant.tmt_settings.level;
                if level != 2 && level != 3 {
                    log::warn!("TMT quant level set at {}, is this correct?", level);
                }
                sage_core::tmt::quantify(&msn_spectra, isobaric, Tolerance::Ppm(-20.0, 20.0), level)
            })
            .unwrap_or_default();

        SageResults {
            features,
            quant,
            ms1: ms1_spectra,
        }
    }

    fn requires_ms1(&self) -> bool {
        self.parameters.quant.lfq
    }

    fn process_chunk(
        &self,
        scorer: &Scorer,
        chunk: &[String],
        chunk_idx: usize,
        batch_size: usize,
    ) -> SageResults {
        let spectra = self.read_processed_spectra(chunk, chunk_idx, batch_size);
        let features = self.search_processed_spectra(scorer, &spectra.1);
        self.complete_features(spectra.1, spectra.0, features)
    }

    fn read_processed_spectra(
        &self,
        chunk: &[String],
        chunk_idx: usize,
        batch_size: usize,
    ) -> (
        MS1Spectra,
        Vec<ProcessedSpectrum<sage_core::spectrum::Peak>>,
    ) {
        info!(
            "processing files {} .. {} ",
            batch_size * chunk_idx,
            batch_size * chunk_idx + chunk.len()
        );
        let start = Instant::now();

        let sn = self
            .parameters
            .quant
            .tmt_settings
            .sn
            .then_some(self.parameters.quant.tmt_settings.level);

        let min_deisotope_mz = match &self.parameters.quant.tmt {
            Some(i) => match self.parameters.quant.tmt_settings.level {
                2 => i.reporter_masses().last().map(|x| x * (1.0 + 20E-6)),
                _ => None,
            },
            None => None,
        };

        let sp = SpectrumProcessor::new(
            self.parameters.max_peaks,
            self.parameters.deisotope,
            min_deisotope_mz.unwrap_or(0.0),
        );

        let file_serial_read = chunk
            .iter()
            .all(|path| FileFormat::from(path.as_ref()).within_file_parallel());
        log::trace!("file serial read: {}", file_serial_read);
        let inner_closure = |(idx, path)| {
            let file_id = chunk_idx * batch_size + idx;
            let res = sage_cloudpath::util::read_spectra(
                path,
                file_id,
                sn,
                self.parameters.bruker_config.clone(),
                self.requires_ms1(),
            );

            match res {
                Ok(s) => {
                    log::trace!("- {}: read {} spectra", path, s.len());
                    Ok(s)
                }
                Err(e) => {
                    log::error!("- {}: {}", path, e);
                    Err(e)
                }
            }
        };

        let spectra: RawSpectrumAccumulator = if file_serial_read {
            chunk
                .iter()
                .enumerate()
                .flat_map(inner_closure)
                .flatten()
                .collect()
        } else {
            chunk
                .par_iter()
                .enumerate()
                .flat_map(inner_closure)
                .flatten()
                .collect()
        };

        let msn_spectra = spectra
            .msn
            .into_par_iter()
            .map(|s| sp.process(s))
            .collect::<Vec<_>>();

        // If all the MS1 spectra contain IMS, then we can process them
        // we use the IMS! otherwise we dont.
        // Note: Empty iterators return true.
        let all_contain_ims = spectra.ms1.iter().all(|x| x.mobility.is_some());
        let ms1_empty = spectra.ms1.is_empty();
        let ms1_spectra = if ms1_empty {
            log::trace!("no MS1 spectra found");
            MS1Spectra::Empty
        } else if all_contain_ims {
            log::trace!("Processing MS1 spectra with IMS");
            let spectra = spectra
                .ms1
                .into_iter()
                .map(|x| sp.process_with_mobility(x))
                .collect();
            MS1Spectra::WithMobility(spectra)
        } else {
            log::trace!("Processing MS1 spectra without IMS");
            let spectra = spectra.ms1.into_iter().map(|s| sp.process(s)).collect();
            MS1Spectra::NoMobility(spectra)
        };

        let io_time = Instant::now() - start;
        info!("- file IO: {:8} ms", io_time.as_millis());

        (ms1_spectra, msn_spectra)
    }

    pub fn batch_files(&self, scorer: &Scorer, batch_size: usize) -> SageResults {
        self.parameters
            .mzml_paths
            .chunks(batch_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| self.process_chunk(scorer, chunk, chunk_idx, batch_size))
            .collect::<SageResults>()
    }

    pub fn run(mut self, parallel: usize, parquet: bool) -> anyhow::Result<telemetry::Telemetry> {
        let scorer = Scorer {
            db: &self.database,
            precursor_tol: self.parameters.precursor_tol,
            fragment_tol: self.parameters.fragment_tol,
            min_matched_peaks: self.parameters.min_matched_peaks,
            min_isotope_err: self.parameters.isotope_errors.0,
            max_isotope_err: self.parameters.isotope_errors.1,
            min_precursor_charge: self.parameters.precursor_charge.0,
            max_precursor_charge: self.parameters.precursor_charge.1,
            override_precursor_charge: self.parameters.override_precursor_charge,
            max_fragment_charge: self.parameters.max_fragment_charge,
            chimera: self.parameters.chimera,
            report_psms: self.parameters.report_psms,
            wide_window: self.parameters.wide_window,
            annotate_matches: self.parameters.annotate_matches,
            score_type: self.parameters.score_type,
        };

        // Collect all results into a single container
        let mut outputs = self.batch_files(&scorer, parallel);

        let filenames = self
            .parameters
            .mzml_paths
            .iter()
            .map(|s| {
                s.parse::<CloudPath>()
                    .ok()
                    .and_then(|c| c.filename().map(|s| s.to_string()))
                    .unwrap_or_else(|| s.clone())
            })
            .collect::<Vec<_>>();

        log::trace!("processing outputs");

        // The pipeline branches HERE.
        // We perform distinct operations for Decoy-Free vs TDC to ensure type safety.

        if self.decoy_free_mode {
            // ======================== DECOY-FREE WORKFLOW ========================
            debug_assert!(
                !self.parameters.database.decoy_tag.is_empty(),
                "decoy_free mode requires non-empty database.decoy_tag"
            );

            // 1. FILTER DECOY-LABELED PSMS (on FeatureCore)
            // Decoy-Free mode does not use explicit decoys for FDR.
            let n_before = outputs.features.len();
            outputs.features.retain(|feat| feat.label != -1);
            let n_dropped = n_before.saturating_sub(outputs.features.len());
            if n_dropped > 0 {
                log::info!("decoy_free mode: dropped {} decoy-labeled PSMs", n_dropped);
            }

            // 2. ML / RT PREDICTION (on FeatureCore)
            // For DF, we train on high-confidence targets: rank 1 & label 1.
            let alignments = if self.parameters.predict_rt {
                let selector = |f: &FeatureCore| f.rank == 1 && f.label == 1;

                let local = sage_core::ml::retention_alignment::global_alignment(
                    &mut outputs.features,
                    self.parameters.mzml_paths.len(),
                    selector,
                );

                let _ = sage_core::ml::retention_model::predict(
                    &self.database,
                    &mut outputs.features,
                    selector,
                );

                let _ = sage_core::ml::mobility_model::predict(
                    &self.database,
                    &mut outputs.features,
                    selector,
                );

                Some(local)
            } else {
                None
            };

            // 3. CONVERT TO DF FEATURE (Strict Type Boundary)
            let mut features: Vec<DfFeature> = outputs
                .features
                .into_par_iter()
                .map(|f| f.to_df())
                .collect();

            // 4. RUN DECOY-FREE FDR
            let fdr_settings = self.parameters.fdr.clone();

            // Calculate Q-values (DF)
            features = sage_core::decoy_free_fdr::calculate_q_values(
                &features,
                &fdr_settings,
                &self.database,
            );

            // Logging
            let q_spectrum = features
                .iter()
                .filter(|f| {
                    f.core.rank == 1
                        && f.core.label == 1
                        && f.decoy_free_q_value.unwrap_or(1.0) <= self.parameters.fdr.peptide_fdr
                })
                .count();

            log::info!(
                "discovered {} target peptide-spectrum matches at {}% FDR (Decoy-Free)",
                q_spectrum,
                self.parameters.fdr.peptide_fdr * 100.0
            );

            // 5. PEPTIDE / PROTEIN AGGREGATION (DF)
            let q_peptide = sage_core::decoy_free_fdr::calculate_peptide_q_df(
                &mut features,
                &self.database,
                fdr_settings.peptide_fdr,
            );

            let q_protein = sage_core::decoy_free_fdr::calculate_protein_q_df(
                &mut features,
                &self.database,
                &fdr_settings,
            );

            log::info!(
                "discovered {} target peptides at {}% FDR (Decoy-Free)",
                q_peptide,
                fdr_settings.peptide_fdr * 100.0
            );
            log::info!("discovered {} target proteins (Decoy-Free)", q_protein);

            // 6. LFQ (DF)
            let areas = alignments.as_ref().and_then(|alignments_ref| {
                if self.parameters.quant.lfq {
                    log::info!("Performing Decoy-Free LFQ...");
                    let mut areas_map = sage_core::lfq::build_feature_map(
                        self.parameters.quant.lfq_settings,
                        self.parameters.precursor_charge,
                        &features, // Pass DfFeature slice
                        true,      // decoy_free_mode = true
                    )
                    .quantify(&self.database, &outputs.ms1, alignments_ref);

                    // Use shadow-trace based precursor FDR
                    let q_precursor = sage_core::decoy_free_fdr::decoy_free_precursor(
                        &mut areas_map,
                        fdr_settings.precursor_fdr,
                    );
                    log::info!(
                        "discovered {} target MS1 peaks at {}% FDR",
                        q_precursor,
                        fdr_settings.precursor_fdr * 100.0
                    );
                    Some(areas_map)
                } else {
                    None
                }
            });

            // 7. WRITE OUTPUTS (DF)
            if !parquet {
                self.parameters
                    .output_paths
                    .push(self.write_features_df(&features, &filenames)?);

                if self.parameters.annotate_matches {
                    // Cast to FeatureCore for fragments
                    let cores: Vec<&FeatureCore> = features.iter().map(|f| &f.core).collect();
                    self.parameters
                        .output_paths
                        .push(self.write_fragments(&cores)?);
                }

                if !outputs.quant.is_empty() {
                    self.parameters
                        .output_paths
                        .push(self.write_tmt(&outputs.quant, &filenames)?);
                }
                if let Some(areas) = areas {
                    self.parameters
                        .output_paths
                        .push(self.write_lfq(areas, &filenames)?);
                }
            } else {
                log::warn!("Parquet not supported for Decoy-Free mode yet.");
            }
        } else {
            // ======================== TDC WORKFLOW (VANILLA) ========================

            // 1. ML / RT PREDICTION (TDC) (on FeatureCore)
            // Vanilla RT/IMS training gate:
            // 1) Sort by poisson (as vanilla does for ML q-value computation).
            // 2) Compute a *temporary ML q-value gate* (via ml::qvalue / spectrum_q_value) on a TdcFeature view.
            // 3) Train RT/IMS on FeatureCore using ONLY those PSMs that pass: label==1 && (temporary ML q <= 0.01).
            // NOTE: This is NOT the post-FDR spectrum_q; it is the ML-qvalue gate used only for RT/IMS training selection.
            let alignments = if self.parameters.predict_rt {
                // Keep parity with vanilla ordering for ML q-value computation.
                outputs
                    .features
                    .par_sort_unstable_by(|a, b| a.poisson.total_cmp(&b.poisson));

                // Stage A: compute the vanilla gate using a temporary TdcFeature view.
                let mut tmp_tdc: Vec<TdcFeature> = outputs
                    .features
                    .iter()
                    .cloned() // clones FeatureCore
                    .map(FeatureCore::to_tdc) // consumes the cloned FeatureCore
                    .collect();

                // ML qvalue is order-dependent; tmp_tdc is already in poisson-sorted order.
                let _ = self.spectrum_fdr(&mut tmp_tdc);

                // Build a membership set of PSM ids that pass the vanilla training gate.
                let selected_psm_ids: HashSet<usize> = tmp_tdc
                    .iter()
                    .filter(|f| f.core.label == 1 && f.spectrum_q <= 0.01)
                    .map(|f| f.core.psm_id)
                    .collect();

                // Stage B: train RT/IMS on FeatureCore using the vanilla-equivalent selector.
                let selector =
                    |f: &FeatureCore| f.label == 1 && selected_psm_ids.contains(&f.psm_id);

                let local = sage_core::ml::retention_alignment::global_alignment(
                    &mut outputs.features,
                    self.parameters.mzml_paths.len(),
                    &selector,
                );

                let _ = sage_core::ml::retention_model::predict(
                    &self.database,
                    &mut outputs.features,
                    &selector,
                );

                let _ = sage_core::ml::mobility_model::predict(
                    &self.database,
                    &mut outputs.features,
                    &selector,
                );

                Some(local)
            } else {
                None
            };

            // 2. CONVERT TO TDC FEATURE (Strict Type Boundary)
            let mut features: Vec<TdcFeature> = outputs
                .features
                .into_par_iter()
                .map(|f| f.to_tdc())
                .collect();

            // 3. RESTORE ORDER & RUN SPECTRUM FDR
            // Restore canonical order by PSM ID before processing
            features.par_sort_unstable_by_key(|f| f.core.psm_id);

            // LDA + Spectrum Q-Value
            let q_spectrum = self.spectrum_fdr(&mut features);

            // Picked Peptide/Protein
            let q_peptide = sage_core::fdr::picked_peptide(&self.database, &mut features);
            let q_protein = sage_core::fdr::picked_protein(&self.database, &mut features);

            // Logging
            log::info!("discovered {} target PSMs at 1% FDR", q_spectrum);
            log::info!("discovered {} target peptides at 1% FDR", q_peptide);
            log::info!("discovered {} target proteins at 1% FDR", q_protein);

            // 4. LFQ (TDC)
            let areas = alignments.as_ref().and_then(|alignments_ref| {
                if self.parameters.quant.lfq {
                    log::trace!("performing LFQ");
                    let mut areas_map = sage_core::lfq::build_feature_map(
                        self.parameters.quant.lfq_settings,
                        self.parameters.precursor_charge,
                        &features, // Pass TdcFeature slice
                        false,     // decoy_free_mode = false
                    )
                    .quantify(&self.database, &outputs.ms1, alignments_ref);

                    let q_precursor = sage_core::fdr::picked_precursor(&mut areas_map);
                    log::info!("discovered {} target MS1 peaks at 5% FDR", q_precursor);
                    Some(areas_map)
                } else {
                    None
                }
            });

            // 5. WRITE OUTPUTS (TDC)
            if !parquet {
                self.parameters
                    .output_paths
                    .push(self.write_features_tdc(&features, &filenames)?);

                if self.parameters.annotate_matches {
                    let cores: Vec<&FeatureCore> = features.iter().map(|f| &f.core).collect();
                    self.parameters
                        .output_paths
                        .push(self.write_fragments(&cores)?);
                }
                // PIN file only for TDC
                if self.parameters.write_pin {
                    self.parameters
                        .output_paths
                        .push(self.write_pin(&features, &filenames)?);
                }
                if !outputs.quant.is_empty() {
                    self.parameters
                        .output_paths
                        .push(self.write_tmt(&outputs.quant, &filenames)?);
                }
                if let Some(areas) = areas {
                    self.parameters
                        .output_paths
                        .push(self.write_lfq(areas, &filenames)?);
                }
            } else {
                // ======================== TDC PARQUET (VANILLA) ========================
                // Match vanilla runner.rs parquet behavior:
                //  - results.sage.parquet (features + embedded metadata)
                //  - matched_fragments.sage.parquet (optional)
                //  - tmt.parquet (if any)
                //  - lfq.parquet (if any)

                // 1) features parquet
                {
                    let path = self.make_path("results.sage.parquet");
                    let bytes = sage_cloudpath::parquet::serialize_features(
                        &features,
                        &outputs.quant,
                        &filenames,
                        &self.database,
                    )?;
                    path.write_bytes_sync(bytes)?;
                    self.parameters.output_paths.push(path.to_string());
                }

                // 2) fragments parquet (optional)
                if self.parameters.annotate_matches {
                    let path = self.make_path("matched_fragments.sage.parquet");
                    let cores: Vec<&FeatureCore> = features.iter().map(|f| &f.core).collect();
                    let bytes = sage_cloudpath::parquet::serialize_fragments(&cores, &filenames)?;
                    path.write_bytes_sync(bytes)?;
                    self.parameters.output_paths.push(path.to_string());
                }

                // 3) tmt parquet (if any)
                if !outputs.quant.is_empty() {
                    let path = self.make_path("tmt.parquet");
                    let bytes = sage_cloudpath::parquet::serialize_tmt(&outputs.quant, &filenames)?;
                    path.write_bytes_sync(bytes)?;
                    self.parameters.output_paths.push(path.to_string());
                }

                // 4) lfq parquet (if any)
                if let Some(areas) = areas {
                    let path = self.make_path("lfq.parquet");
                    let bytes =
                        sage_cloudpath::parquet::serialize_lfq(&areas, &filenames, &self.database)?;
                    path.write_bytes_sync(bytes)?;
                    self.parameters.output_paths.push(path.to_string());
                }
            }
        }

        // Final Metadata Write
        let path = self.make_path("results.json");
        self.parameters.output_paths.push(path.to_string());
        println!("{}", serde_json::to_string_pretty(&self.parameters)?);

        let bytes = serde_json::to_vec_pretty(&self.parameters)?;
        path.write_bytes_sync(bytes)?;

        let run_time = (Instant::now() - self.start).as_secs();
        info!("finished in {}s", run_time);

        let telemetry = telemetry::Telemetry::new(
            self.parameters,
            self.database.peptides.len(),
            self.database.fragments.len(),
            parquet,
            run_time,
        );

        Ok(telemetry)
    }

    // --- TDC WRITERS (Vanilla) ---

    pub fn serialize_tdc_feature(
        &self,
        feature: &TdcFeature,
        filenames: &[String],
    ) -> csv::ByteRecord {
        let mut record = csv::ByteRecord::new();
        let core = &feature.core;

        record.push_field(itoa::Buffer::new().format(core.psm_id).as_bytes());
        let peptide = &self.database[core.peptide_idx];
        record.push_field(peptide.to_string().as_bytes());
        record.push_field(
            peptide
                .proteins(&self.database.decoy_tag, self.database.generate_decoys)
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format(peptide.proteins.len())
                .as_bytes(),
        );
        record.push_field(filenames[core.file_id].as_bytes());
        record.push_field(core.spec_id.as_bytes());
        record.push_field(itoa::Buffer::new().format(core.rank).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.label).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.expmass).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.calcmass).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.charge).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.peptide_len).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.missed_cleavages).as_bytes());
        record.push_field(
            itoa::Buffer::new()
                .format(peptide.semi_enzymatic as u8)
                .as_bytes(),
        );
        record.push_field(ryu::Buffer::new().format(core.isotope_error).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_mass).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.average_ppm).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.hyperscore).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_next).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_best).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.rt).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.aligned_rt).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.predicted_rt).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_rt_model).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.ims).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.predicted_ims).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_ims_model).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.matched_peaks).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.longest_b).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.longest_y).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.longest_y_pct).as_bytes());
        record.push_field(
            ryu::Buffer::new()
                .format(core.matched_intensity_pct)
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format(core.scored_candidates)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format((-core.poisson).ln_1p())
                .as_bytes(),
        );

        // Vanilla specific columns
        record.push_field(
            ryu::Buffer::new()
                .format(feature.discriminant_score)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(feature.posterior_error)
                .as_bytes(),
        );
        record.push_field(ryu::Buffer::new().format(feature.spectrum_q).as_bytes());
        record.push_field(ryu::Buffer::new().format(feature.peptide_q).as_bytes());
        record.push_field(ryu::Buffer::new().format(feature.protein_q).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.ms2_intensity).as_bytes());

        record
    }

    pub fn write_features_tdc(
        &self,
        features: &[TdcFeature],
        filenames: &[String],
    ) -> anyhow::Result<String> {
        let path = self.make_path("results.sage.tsv");
        let mut wtr = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(vec![]);

        let headers = vec![
            "psm_id",
            "peptide",
            "proteins",
            "num_proteins",
            "filename",
            "scannr",
            "rank",
            "label",
            "expmass",
            "calcmass",
            "charge",
            "peptide_len",
            "missed_cleavages",
            "semi_enzymatic",
            "isotope_error",
            "precursor_ppm",
            "fragment_ppm",
            "hyperscore",
            "delta_next",
            "delta_best",
            "rt",
            "aligned_rt",
            "predicted_rt",
            "delta_rt_model",
            "ion_mobility",
            "predicted_mobility",
            "delta_mobility",
            "matched_peaks",
            "longest_b",
            "longest_y",
            "longest_y_pct",
            "matched_intensity_pct",
            "scored_candidates",
            "poisson",
            "sage_discriminant_score",
            "posterior_error",
            "spectrum_q",
            "peptide_q",
            "protein_q",
            "ms2_intensity",
        ];

        wtr.write_byte_record(&csv::ByteRecord::from(headers))?;

        let records: Vec<csv::ByteRecord> = features
            .par_iter()
            .map(|feat| self.serialize_tdc_feature(feat, filenames))
            .collect();

        for record in records {
            wtr.write_byte_record(&record)?;
        }

        wtr.flush()?;
        let bytes = wtr.into_inner()?;
        let cp = CloudPath::from(path);
        cp.write_bytes_sync(bytes)?;
        Ok(cp.to_string())
    }

    // --- DF WRITERS (Decoy-Free) ---

    pub fn serialize_df_feature(
        &self,
        feature: &DfFeature,
        filenames: &[String],
    ) -> csv::ByteRecord {
        let mut record = csv::ByteRecord::new();
        let core = &feature.core;

        // Core Columns
        record.push_field(itoa::Buffer::new().format(core.psm_id).as_bytes());
        let peptide = &self.database[core.peptide_idx];
        record.push_field(peptide.to_string().as_bytes());
        record.push_field(
            peptide
                .proteins(&self.database.decoy_tag, self.database.generate_decoys)
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format(peptide.proteins.len())
                .as_bytes(),
        );
        record.push_field(filenames[core.file_id].as_bytes());
        record.push_field(core.spec_id.as_bytes());
        record.push_field(itoa::Buffer::new().format(core.rank).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.label).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.expmass).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.calcmass).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.charge).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.peptide_len).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.missed_cleavages).as_bytes());
        record.push_field(
            itoa::Buffer::new()
                .format(peptide.semi_enzymatic as u8)
                .as_bytes(),
        );
        record.push_field(ryu::Buffer::new().format(core.isotope_error).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_mass).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.average_ppm).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.hyperscore).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_next).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_best).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.rt).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.aligned_rt).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.predicted_rt).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_rt_model).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.ims).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.predicted_ims).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.delta_ims_model).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.matched_peaks).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.longest_b).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.longest_y).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.longest_y_pct).as_bytes());
        record.push_field(
            ryu::Buffer::new()
                .format(core.matched_intensity_pct)
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format(core.scored_candidates)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format((-core.poisson).ln_1p())
                .as_bytes(),
        );

        // Write MS2 Intensity
        record.push_field(ryu::Buffer::new().format(core.ms2_intensity).as_bytes());

        // Decoy-Free specific columns
        let fmt_f32 = |val: Option<f32>| {
            val.map(|v| v.to_string())
                .unwrap_or_else(|| "NaN".to_string())
        };

        record.push_field(fmt_f32(feature.decoy_free_p_value).as_bytes());
        record.push_field(fmt_f32(feature.decoy_free_pep).as_bytes());
        record.push_field(fmt_f32(feature.decoy_free_score).as_bytes());
        record.push_field(fmt_f32(feature.decoy_free_q_value).as_bytes());
        record.push_field(fmt_f32(feature.decoy_free_peptide_q).as_bytes());
        record.push_field(fmt_f32(feature.decoy_free_protein_q).as_bytes());

        // Per-method diagnostics
        record.push_field(fmt_f32(feature.p_mom).as_bytes());
        record.push_field(fmt_f32(feature.p_mle).as_bytes());
        record.push_field(fmt_f32(feature.p_lo).as_bytes());
        record.push_field(fmt_f32(feature.p_msfdr).as_bytes());
        record.push_field(fmt_f32(feature.p_nokoi).as_bytes());

        record.push_field(fmt_f32(feature.q_mom).as_bytes());
        record.push_field(fmt_f32(feature.q_mle).as_bytes());
        record.push_field(fmt_f32(feature.q_lo).as_bytes());
        record.push_field(fmt_f32(feature.q_msfdr).as_bytes());
        record.push_field(fmt_f32(feature.q_nokoi).as_bytes());

        record.push_field(fmt_f32(feature.pep_mom).as_bytes());
        record.push_field(fmt_f32(feature.pep_mle).as_bytes());
        record.push_field(fmt_f32(feature.pep_lo).as_bytes());
        record.push_field(fmt_f32(feature.pep_msfdr).as_bytes());
        record.push_field(fmt_f32(feature.pep_nokoi).as_bytes());

        record
    }

    pub fn write_features_df(
        &self,
        features: &[DfFeature],
        filenames: &[String],
    ) -> anyhow::Result<String> {
        let path = self.make_path("results.sage.tsv");
        let mut wtr = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(vec![]);

        let headers = vec![
            "psm_id",
            "peptide",
            "proteins",
            "num_proteins",
            "filename",
            "scannr",
            "rank",
            "label",
            "expmass",
            "calcmass",
            "charge",
            "peptide_len",
            "missed_cleavages",
            "semi_enzymatic",
            "isotope_error",
            "precursor_ppm",
            "fragment_ppm",
            "hyperscore",
            "delta_next",
            "delta_best",
            "rt",
            "aligned_rt",
            "predicted_rt",
            "delta_rt_model",
            "ion_mobility",
            "predicted_mobility",
            "delta_mobility",
            "matched_peaks",
            "longest_b",
            "longest_y",
            "longest_y_pct",
            "matched_intensity_pct",
            "scored_candidates",
            "poisson",
            "ms2_intensity",
            // DF Columns
            "decoy_free_p_value",
            "decoy_free_pep",
            "decoy_free_score",
            "decoy_free_q_value",
            "decoy_free_peptide_q",
            "decoy_free_protein_q",
            "p_mom",
            "p_mle",
            "p_lo",
            "p_msfdr",
            "p_nokoi",
            "q_mom",
            "q_mle",
            "q_lo",
            "q_msfdr",
            "q_nokoi",
            "pep_mom",
            "pep_mle",
            "pep_lo",
            "pep_msfdr",
            "pep_nokoi",
        ];

        wtr.write_byte_record(&csv::ByteRecord::from(headers))?;

        let records: Vec<csv::ByteRecord> = features
            .par_iter()
            .map(|feat| self.serialize_df_feature(feat, filenames))
            .collect();

        for record in records {
            wtr.write_byte_record(&record)?;
        }

        wtr.flush()?;
        let bytes = wtr.into_inner()?;
        let cp = CloudPath::from(path);
        cp.write_bytes_sync(bytes)?;
        Ok(cp.to_string())
    }

    // --- SHARED WRITERS ---

    pub fn write_fragments(&self, features: &[&FeatureCore]) -> anyhow::Result<String> {
        let path = self.make_path("matched_fragments.sage.tsv");

        let mut wtr = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(vec![]);

        let headers = csv::ByteRecord::from(vec![
            "psm_id",
            "fragment_type",
            "fragment_ordinals",
            "fragment_charge",
            "fragment_mz_calculated",
            "fragment_mz_experimental",
            "fragment_intensity",
        ]);

        wtr.write_byte_record(&headers)?;

        for record in features
            .into_par_iter()
            .map(|feat| self.serialize_fragments(feat.psm_id, &feat.fragments))
            .flatten()
            .collect::<Vec<_>>()
        {
            wtr.write_byte_record(&record)?;
        }

        wtr.flush()?;
        let bytes = wtr.into_inner()?;
        let cp = CloudPath::from(path);
        cp.write_bytes_sync(bytes)?;
        Ok(cp.to_string())
    }

    pub fn serialize_fragments(
        &self,
        psm_id: usize,
        fragments_: &Option<Fragments>,
    ) -> Vec<ByteRecord> {
        let mut frag_records = vec![];

        if let Some(fragments) = fragments_ {
            for id in 0..fragments.fragment_ordinals.len() {
                let mut record = ByteRecord::new();
                record.push_field(itoa::Buffer::new().format(psm_id).as_bytes());
                let ion_type = match fragments.kinds[id] {
                    Kind::A => "a",
                    Kind::B => "b",
                    Kind::C => "c",
                    Kind::X => "x",
                    Kind::Y => "y",
                    Kind::Z => "z",
                };
                record.push_field(ion_type.as_bytes());
                record.push_field(
                    itoa::Buffer::new()
                        .format(fragments.fragment_ordinals[id])
                        .as_bytes(),
                );
                record.push_field(itoa::Buffer::new().format(fragments.charges[id]).as_bytes());
                record.push_field(
                    ryu::Buffer::new()
                        .format(fragments.mz_calculated[id])
                        .as_bytes(),
                );
                record.push_field(
                    ryu::Buffer::new()
                        .format(fragments.mz_experimental[id])
                        .as_bytes(),
                );
                record.push_field(
                    ryu::Buffer::new()
                        .format(fragments.intensities[id])
                        .as_bytes(),
                );
                frag_records.push(record);
            }
        }

        frag_records
    }

    pub fn write_pin(
        &self,
        features: &[TdcFeature],
        filenames: &[String],
    ) -> anyhow::Result<String> {
        let path = self.make_path("results.sage.pin");

        let mut wtr = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(vec![]);

        let headers = csv::ByteRecord::from(vec![
            "SpecId",
            "Label",
            "ScanNr",
            "ExpMass",
            "CalcMass",
            "FileName",
            "retentiontime",
            "ion_mobility",
            "rank",
            "z=2",
            "z=3",
            "z=4",
            "z=5",
            "z=6",
            "z=other",
            "peptide_len",
            "missed_cleavages",
            "semi_enzymatic",
            "isotope_error",
            "ln(precursor_ppm)",
            "fragment_ppm",
            "ln(hyperscore)",
            "ln(delta_next)",
            "ln(delta_best)",
            "aligned_rt",
            "predicted_rt",
            "sqrt(delta_rt_model)",
            "predicted_mobility",
            "sqrt(delta_mobility)",
            "matched_peaks",
            "longest_b",
            "longest_y",
            "longest_y_pct",
            "ln(matched_intensity_pct)",
            "scored_candidates",
            "ln(-poisson)",
            "posterior_error",
            "Peptide",
            "Proteins",
        ]);

        let re = regex::Regex::new(r"scan=(\d+)").expect("This is valid regex");

        wtr.write_byte_record(&headers)?;
        for record in features
            .into_par_iter()
            .map(|feat| self.serialize_pin(&re, feat, filenames))
            .collect::<Vec<_>>()
        {
            wtr.write_byte_record(&record)?;
        }

        wtr.flush()?;
        let bytes = wtr.into_inner()?;
        let cp = CloudPath::from(path);
        cp.write_bytes_sync(bytes)?;
        Ok(cp.to_string())
    }

    fn serialize_pin(
        &self,
        re: &regex::Regex,
        feature: &TdcFeature,
        filenames: &[String],
    ) -> csv::ByteRecord {
        let core = &feature.core;

        let scannr = re
            .captures_iter(&core.spec_id)
            .last()
            .and_then(|cap| cap.get(1).map(|cap| cap.as_str()))
            .unwrap_or(&core.spec_id);

        let mut record = csv::ByteRecord::new();
        let peptide = &self.database[core.peptide_idx];
        record.push_field(itoa::Buffer::new().format(core.psm_id).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.label).as_bytes());
        record.push_field(scannr.as_bytes());
        record.push_field(ryu::Buffer::new().format(core.expmass).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.calcmass).as_bytes());
        record.push_field(filenames[core.file_id].as_bytes());
        record.push_field(ryu::Buffer::new().format(core.rt).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.ims).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.rank).as_bytes());
        record.push_field(
            itoa::Buffer::new()
                .format((core.charge == 2) as i32)
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format((core.charge == 3) as i32)
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format((core.charge == 4) as i32)
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format((core.charge == 5) as i32)
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format((core.charge == 6) as i32)
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format(if core.charge < 2 || core.charge > 6 {
                    core.charge
                } else {
                    0
                })
                .as_bytes(),
        );
        record.push_field(itoa::Buffer::new().format(core.peptide_len).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.missed_cleavages).as_bytes());
        record.push_field(
            itoa::Buffer::new()
                .format(peptide.semi_enzymatic as u8)
                .as_bytes(),
        );
        record.push_field(ryu::Buffer::new().format(core.isotope_error).as_bytes());
        record.push_field(
            ryu::Buffer::new()
                .format(core.delta_mass.abs().ln_1p())
                .as_bytes(),
        );
        record.push_field(ryu::Buffer::new().format(core.average_ppm).as_bytes());
        record.push_field(
            ryu::Buffer::new()
                .format(core.hyperscore.ln_1p())
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(core.delta_next.ln_1p())
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(core.delta_best.ln_1p())
                .as_bytes(),
        );
        record.push_field(ryu::Buffer::new().format(core.aligned_rt).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.predicted_rt).as_bytes());
        record.push_field(
            ryu::Buffer::new()
                .format(core.delta_rt_model.clamp(0.001, 1.0).sqrt())
                .as_bytes(),
        );
        record.push_field(ryu::Buffer::new().format(core.predicted_ims).as_bytes());
        record.push_field(
            ryu::Buffer::new()
                .format(core.delta_ims_model.clamp(0.0, 1.0).sqrt())
                .as_bytes(),
        );
        record.push_field(itoa::Buffer::new().format(core.matched_peaks).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.longest_b).as_bytes());
        record.push_field(itoa::Buffer::new().format(core.longest_y).as_bytes());
        record.push_field(ryu::Buffer::new().format(core.longest_y_pct).as_bytes());
        record.push_field(
            ryu::Buffer::new()
                .format(core.matched_intensity_pct.ln_1p())
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format(core.scored_candidates)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format((-core.poisson).ln_1p())
                .as_bytes(),
        );
        // Only TdcFeature has posterior_error
        record.push_field(
            ryu::Buffer::new()
                .format(feature.posterior_error)
                .as_bytes(),
        );
        record.push_field(peptide.to_string().as_bytes());
        record.push_field(
            peptide
                .proteins(&self.database.decoy_tag, self.database.generate_decoys)
                .as_bytes(),
        );
        record
    }

    pub fn write_lfq(
        &self,
        areas: HashMap<(PrecursorId, bool), (Peak, Vec<f64>), fnv::FnvBuildHasher>,
        filenames: &[String],
    ) -> anyhow::Result<String> {
        let path = self.make_path("lfq.tsv");

        let mut wtr = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(vec![]);
        let mut headers = csv::ByteRecord::from(vec![
            "peptide",
            "charge",
            "proteins",
            "q_value",
            "score",
            "spectral_angle",
        ]);
        headers.extend(filenames);

        wtr.write_byte_record(&headers)?;

        let records = areas
            .into_par_iter()
            .filter_map(|((id, decoy), (peak, data))| {
                if decoy {
                    return None;
                };
                let mut record = csv::ByteRecord::new();
                let (peptide_ix, charge) = match id {
                    PrecursorId::Combined(x) => (x, None),
                    PrecursorId::Charged((x, charge)) => (x, Some(charge as i32)),
                };
                record.push_field(self.database[peptide_ix].to_string().as_bytes());
                record.push_field(itoa::Buffer::new().format(charge.unwrap_or(-1)).as_bytes());
                record.push_field(
                    self.database[peptide_ix]
                        .proteins(&self.database.decoy_tag, self.database.generate_decoys)
                        .as_bytes(),
                );
                record.push_field(ryu::Buffer::new().format(peak.q_value).as_bytes());
                record.push_field(ryu::Buffer::new().format(peak.score).as_bytes());
                record.push_field(ryu::Buffer::new().format(peak.spectral_angle).as_bytes());
                for x in data {
                    record.push_field(ryu::Buffer::new().format(x).as_bytes());
                }
                Some(record)
            })
            .collect::<Vec<csv::ByteRecord>>();

        for record in records {
            wtr.write_record(&record)?;
        }
        wtr.flush()?;

        let bytes = wtr.into_inner()?;
        let cp = CloudPath::from(path); // Use CloudPath from path (already created)
        cp.write_bytes_sync(bytes)?;
        Ok(cp.to_string())
    }

    pub fn write_tmt(&self, quant: &[TmtQuant], filenames: &[String]) -> anyhow::Result<String> {
        let path = self.make_path("tmt.tsv");

        let mut wtr = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(vec![]);
        let mut headers = csv::ByteRecord::from(vec!["filename", "scannr", "ion_injection_time"]);
        headers.extend(
            self.parameters
                .quant
                .tmt
                .as_ref()
                .map(|tmt| tmt.headers())
                .expect("TMT quant cannot be performed without setting this parameter"),
        );

        wtr.write_byte_record(&headers)?;

        let records = quant
            .into_par_iter()
            .map(|q| {
                let mut record = csv::ByteRecord::new();
                record.push_field(filenames[q.file_id].as_bytes());
                record.push_field(q.spec_id.as_bytes());
                record.push_field(ryu::Buffer::new().format(q.ion_injection_time).as_bytes());
                for peak in &q.peaks {
                    record.push_field(ryu::Buffer::new().format(*peak).as_bytes());
                }
                record
            })
            .collect::<Vec<csv::ByteRecord>>();

        for record in records {
            wtr.write_record(&record)?;
        }
        wtr.flush()?;

        let bytes = wtr.into_inner()?;
        let cp = CloudPath::from(path);
        cp.write_bytes_sync(bytes)?;
        Ok(cp.to_string())
    }
}
