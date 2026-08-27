use super::input::{ExternalFeatureUseMode, Search};
use super::output::SageResults;
use super::telemetry;
use crate::candidate_pool::{
    analysis_fingerprint, candidate_pool_identity_preflight, inspect_compatible_pool, load_pool,
    load_required_pool, manifest_path, pool_directory, publish_pool_atomic, relocation_provenance,
    search_fingerprint, write_pool, CandidatePoolRequest, CandidatePoolUsage,
};
use crate::external_feature_cache::{ExternalAnnotationCacheRequest, ExternalAnnotationCacheUsage};
use crate::external_features::{
    construct_raw_cache_only, maybe_add_external_features, RawCacheOnlyConstructionResult,
};
use crate::provenance::sha256_file;
use anyhow::Context;
use csv::ByteRecord;
use log::info;
use rayon::prelude::*;
use sage_cloudpath::{FileFormat, Url};
use sage_core::database::{IndexedDatabase, Parameters, PeptideIx};
use sage_core::fasta::Fasta;
use sage_core::input::{EntrapmentReportMode, FdrMode, HierarchicalReportingMode, ModelFit};
use sage_core::ion_series::Kind;
use sage_core::lfq::{Peak, PrecursorId};
use sage_core::mass::Tolerance;
use sage_core::ml::linear_discriminant::score_psms;
use sage_core::peptide::Peptide;
use sage_core::scoring::Fragments;
use sage_core::scoring::{DfFeature, FeatureCore, Scorer, TdcFeature};
use sage_core::spectrum::{ProcessedSpectrum, RawSpectrum, SpectrumProcessor};
use sage_core::tmt::TmtQuant;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

#[derive(serde::Serialize, serde::Deserialize)]
struct NullWindowOptimizerCheckpoint {
    schema: String,
    fingerprint: String,
    evaluations: Vec<sage_core::decoy_free_fdr::NullWindowEvaluation>,
}

fn null_window_checkpoint_fingerprint(
    features: &[DfFeature],
    settings: &sage_core::input::FdrSettings,
) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"sage-null-window-checkpoint-v1\0");
    hasher.update(serde_json::to_vec(settings)?);
    hasher.update((features.len() as u64).to_le_bytes());
    for feature in features {
        hasher.update((feature.core.spec_id.len() as u64).to_le_bytes());
        hasher.update(feature.core.spec_id.as_bytes());
        hasher.update((feature.core.file_id as u64).to_le_bytes());
        hasher.update(feature.core.rank.to_le_bytes());
        hasher.update(feature.core.peptide_idx.0.to_le_bytes());
        hasher.update(feature.core.hyperscore.to_bits().to_le_bytes());
        hasher.update(feature.core.poisson_log10_p_value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub struct Runner {
    pub database: Arc<IndexedDatabase>,
    pub parameters: Search,
    pub start: Instant,
    pub decoy_free_mode: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CandidatePoolConstructionReport {
    pub schema_version: u32,
    pub execution_scope: String,
    pub status: String,
    pub search_fingerprint: String,
    pub analysis_fingerprint: String,
    pub requested_root: std::path::PathBuf,
    pub pool_directory: std::path::PathBuf,
    pub manifest: std::path::PathBuf,
    pub payload: std::path::PathBuf,
    pub manifest_sha256: String,
    pub payload_sha256: String,
    pub candidate_count: usize,
    pub spectrum_count: usize,
    pub retained_rank_depth: usize,
    pub observed_max_rank: u32,
    pub reused_existing_exact: bool,
    pub native_search_performed: bool,
    pub native_rt_ims_prediction_performed: bool,
    pub downstream_stages_entered: Vec<String>,
    pub stop_guarantee: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RawCacheConstructionReport {
    pub schema_version: u32,
    pub execution_scope: String,
    pub status: String,
    pub candidate_pool: CandidatePoolUsage,
    pub candidate_pool_manifest_sha256: String,
    pub candidate_pool_payload_sha256: String,
    pub raw_cache: RawCacheOnlyConstructionResult,
    pub downstream_stages_entered: Vec<String>,
    pub stop_guarantee: Vec<String>,
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

#[derive(Clone, Debug)]
enum DfDynamicColumn {
    Final(&'static str),
    ReportingFlag(&'static str),
    BaseModel(&'static str),
    RtModel(&'static str),
    ImsModel(&'static str),
    PeptideRescueModel(&'static str),
    ProteinRescueModel(&'static str),
}

impl Runner {
    /// Construct or exactly reopen one immutable candidate pool and return
    /// before any Decoy-Free statistical or external-annotation stage.
    pub fn construct_candidate_pool_only(
        self,
        parallel: usize,
        root: std::path::PathBuf,
        required_rank_depth: usize,
    ) -> anyhow::Result<CandidatePoolConstructionReport> {
        anyhow::ensure!(
            self.decoy_free_mode,
            "candidate-pool-only requires fdr.mode=decoy_free"
        );
        anyhow::ensure!(
            !self.parameters.database.generate_decoys,
            "candidate-pool-only requires database.generate_decoys=false"
        );
        anyhow::ensure!(
            !self.parameters.quant.lfq
                && self.parameters.quant.tmt.is_none()
                && !self.parameters.annotate_matches,
            "candidate-pool-only prohibits LFQ, TMT, and matched-fragment annotation"
        );
        let (search, analysis) =
            candidate_pool_identity_preflight(&self.parameters, required_rank_depth)?;
        let directory = pool_directory(&root, &search);

        let (manifest, reused_existing_exact, native_search_performed) = if directory.exists() {
            let (manifest, _) = load_required_pool(
                &directory,
                &search,
                search.retained_rank_depth,
                &self.database,
            )
            .with_context(|| {
                format!(
                    "existing final candidate-pool directory is incomplete or incompatible; fallback search is prohibited: {}",
                    directory.display()
                )
            })?;
            (manifest, true, false)
        } else {
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
                annotate_matches: false,
                score_type: self.parameters.score_type,
            };
            let mut outputs = self.batch_files(&scorer, parallel);
            outputs.features.retain(|feature| feature.label != -1);

            if self.parameters.predict_rt {
                let selector = |feature: &FeatureCore| feature.rank == 1 && feature.label == 1;
                let _ = sage_core::ml::retention_alignment::global_alignment(
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
            }

            let (manifest, reused) =
                publish_pool_atomic(&directory, &search, &outputs.features, &self.database)?;
            anyhow::ensure!(
                !reused,
                "candidate-pool final directory appeared during native search; refusing ambiguous publication"
            );
            (manifest, false, true)
        };

        let (
            original_source_uris,
            current_source_uris,
            portable_identity_valid,
            relocation_detected,
        ) = relocation_provenance(&manifest.search_fingerprint, &search);
        let usage = CandidatePoolUsage {
            search_fingerprint: search.digest.clone(),
            analysis_fingerprint: analysis.digest.clone(),
            manifest: manifest_path(&directory),
            payload: directory.join(&manifest.payload_file),
            reused: reused_existing_exact,
            candidate_count: manifest.candidate_count,
            retained_rank_depth: manifest.capabilities.retained_rank_depth,
            original_source_uris,
            current_source_uris,
            portable_identity_valid,
            relocation_detected,
        };
        crate::candidate_pool::verify_usage(&usage)?;
        let manifest_sha256 = sha256_file(&usage.manifest)?;
        let payload_sha256 = sha256_file(&usage.payload)?;
        anyhow::ensure!(
            payload_sha256 == manifest.payload_sha256,
            "candidate-pool payload changed after publication"
        );

        Ok(CandidatePoolConstructionReport {
            schema_version: 1,
            execution_scope: "candidate_pool_only".into(),
            status: "verified_complete".into(),
            search_fingerprint: search.digest,
            analysis_fingerprint: analysis.digest,
            requested_root: root,
            pool_directory: directory,
            manifest: usage.manifest,
            payload: usage.payload,
            manifest_sha256,
            payload_sha256,
            candidate_count: manifest.candidate_count,
            spectrum_count: manifest.spectrum_count,
            retained_rank_depth: manifest.capabilities.retained_rank_depth,
            observed_max_rank: manifest.observed_max_rank,
            reused_existing_exact,
            native_search_performed,
            native_rt_ims_prediction_performed: native_search_performed
                && self.parameters.predict_rt,
            downstream_stages_entered: Vec::new(),
            stop_guarantee: vec![
                "no Decoy-Free fitting or q-value calculation".into(),
                "no parameter or null-window optimization".into(),
                "no external annotation process".into(),
                "no audit, target-only, or TDC stage".into(),
                "no ordinary results table or fitted artifact".into(),
            ],
        })
    }

    /// Construct or exactly reopen only the model-independent raw external
    /// prediction cache from a required immutable candidate pool. The method
    /// has no search fallback and never enters ordinary workflow stages.
    pub fn construct_raw_annotation_cache_only(
        self,
        candidate_pool_root: std::path::PathBuf,
        annotation_cache_root: std::path::PathBuf,
        required_rank_depth: usize,
    ) -> anyhow::Result<RawCacheConstructionReport> {
        anyhow::ensure!(
            self.decoy_free_mode,
            "raw-cache-only requires fdr.mode=decoy_free"
        );
        anyhow::ensure!(
            !self.parameters.database.prefilter,
            "raw-cache-only prohibits database.prefilter because rebuilding a prefiltered index can launch a native spectrum search"
        );
        anyhow::ensure!(required_rank_depth > 0, "rank depth must be positive");
        anyhow::ensure!(
            self.parameters.external_features.enabled,
            "raw-cache-only requires external_features.enabled=true"
        );
        anyhow::ensure!(
            self.parameters.external_features.max_rank == Some(required_rank_depth as u32),
            "raw-cache-only --rank-depth must exactly equal frozen external_features.max_rank"
        );
        let request = CandidatePoolRequest {
            root: candidate_pool_root,
            required_rank_depth,
            allow_reuse: true,
            require_existing: true,
        };
        let (candidate_usage, candidates) = self
            .preflight_existing_candidate_pool(&request)
            .context("raw-cache-only exact candidate-pool verification failed; search fallback is prohibited")?;
        crate::candidate_pool::verify_usage(&candidate_usage)?;
        let candidate_pool_manifest_sha256 = sha256_file(&candidate_usage.manifest)?;
        let candidate_pool_payload_sha256 = sha256_file(&candidate_usage.payload)?;
        let mut features = candidates
            .into_iter()
            .map(FeatureCore::to_df)
            .collect::<Vec<_>>();
        let raw_cache = construct_raw_cache_only(
            &mut features,
            &self.parameters.external_features,
            &self.parameters.mzml_paths,
            &self.database,
            &candidate_usage.search_fingerprint,
            &annotation_cache_root,
        )?;
        anyhow::ensure!(
            raw_cache.requested_candidate_count == features.len()
                && raw_cache.joined_candidate_count == features.len(),
            "raw-cache-only coverage differs from the verified candidate-pool population"
        );

        Ok(RawCacheConstructionReport {
            schema_version: 1,
            execution_scope: "raw_annotation_cache_only".into(),
            status: "verified_complete".into(),
            candidate_pool: candidate_usage,
            candidate_pool_manifest_sha256,
            candidate_pool_payload_sha256,
            raw_cache,
            downstream_stages_entered: Vec::new(),
            stop_guarantee: vec![
                "required candidate pool fully verified before generator provenance resolution"
                    .into(),
                "no spectrum-search fallback".into(),
                "no stage calibration or statistical fitting".into(),
                "no optimizer, audit, winner, target-only, or TDC stage".into(),
            ],
        })
    }

    /// Read-only strict candidate-pool preflight. It validates the complete
    /// manifest, compressed payload, stable IDs, count, schema, and rank depth
    /// without searching spectra or writing workflow state.
    pub(crate) fn preflight_existing_candidate_pool(
        &self,
        request: &CandidatePoolRequest,
    ) -> anyhow::Result<(CandidatePoolUsage, Vec<FeatureCore>)> {
        let search = search_fingerprint(&self.parameters)?;
        anyhow::ensure!(
            request.required_rank_depth <= search.retained_rank_depth,
            "candidate pool retains ranks 1..={} but preflight requires rank {}",
            search.retained_rank_depth,
            request.required_rank_depth
        );
        let analysis = analysis_fingerprint(&self.parameters, &search)?;
        let directory = pool_directory(&request.root, &search);
        let (manifest, features) = load_required_pool(
            &directory,
            &search,
            request.required_rank_depth,
            &self.database,
        )?;
        let (
            original_source_uris,
            current_source_uris,
            portable_identity_valid,
            relocation_detected,
        ) = relocation_provenance(&manifest.search_fingerprint, &search);
        Ok((
            CandidatePoolUsage {
                search_fingerprint: search.digest,
                analysis_fingerprint: analysis.digest,
                manifest: manifest_path(&directory),
                payload: directory.join(&manifest.payload_file),
                reused: true,
                candidate_count: manifest.candidate_count,
                retained_rank_depth: manifest.capabilities.retained_rank_depth,
                original_source_uris,
                current_source_uris,
                portable_identity_valid,
                relocation_detected,
            },
            features,
        ))
    }

    pub fn new(parameters: Search, parallel: usize) -> anyhow::Result<Self> {
        let mut parameters = parameters.clone();
        let start = Instant::now();

        let decoy_free_mode = matches!(parameters.fdr.mode, FdrMode::DecoyFree);
        log::info!(
            "FDR mode at runtime: {:?} (decoy_free_mode = {})",
            parameters.fdr.mode,
            decoy_free_mode
        );

        let fasta_url = sage_cloudpath::to_url(&parameters.database.fasta)?;
        let fasta =
            sage_cloudpath::util::read_fasta(&fasta_url, &parameters.database.decoy_tag, false)
                .with_context(|| {
                    format!(
                        "Failed to build database from `{}`",
                        parameters.database.fasta
                    )
                })?;

        if decoy_free_mode && parameters.report_psms < 10 {
            log::warn!(
				"decoy_free mode requires report_psms >= 10 to retain sufficient candidate depth for stable downstream Decoy-Free modeling and diagnostics; overriding to 10"
			);
            parameters.report_psms = 10;
        }

        if let Some(max_rank) = parameters.external_features.max_rank {
            if parameters.report_psms < max_rank as usize {
                anyhow::bail!(
					"external_features.max_rank={} exceeds report_psms={}; lower-rank candidates would be unavailable",
					max_rank,
					parameters.report_psms
				);
            }
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
                    &fasta_url,
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
                        database: Arc::new(IndexedDatabase::default()),
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
            database: Arc::new(database),
            parameters,
            start,
            decoy_free_mode,
        })
    }

    /// Construct another analysis runner over an already-built database index.
    /// The caller must guard reuse with the same strict search fingerprint used
    /// for the candidate pool.
    pub fn with_shared_database(
        mut parameters: Search,
        database: Arc<IndexedDatabase>,
    ) -> anyhow::Result<Self> {
        let decoy_free_mode = matches!(parameters.fdr.mode, FdrMode::DecoyFree);
        if decoy_free_mode && parameters.report_psms < 10 {
            parameters.report_psms = 10;
        }
        if let Some(max_rank) = parameters.external_features.max_rank {
            anyhow::ensure!(
                parameters.report_psms >= max_rank as usize,
                "external_features.max_rank={} exceeds report_psms={}",
                max_rank,
                parameters.report_psms
            );
        }
        Ok(Self {
            database,
            parameters,
            start: Instant::now(),
            decoy_free_mode,
        })
    }

    pub fn shared_database(&self) -> Arc<IndexedDatabase> {
        Arc::clone(&self.database)
    }

    pub fn prefilter_peptides(self, parallel: usize, fasta: Fasta) -> Vec<Peptide> {
        let spectra: Option<(Vec<ProcessedSpectrum>, Vec<ProcessedSpectrum>)> =
            match parallel >= self.parameters.mzml_paths.len() {
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
        spectra: &[ProcessedSpectrum],
    ) -> Vec<PeptideIx> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = AtomicUsize::new(0);
        let start = Instant::now();

        let peptide_idxs: Vec<_> = spectra
            .par_iter()
            .filter(|spec| spec.masses.len() >= self.parameters.min_peaks && spec.level == 2)
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

    /// Compute target-decoy spectrum-level q-values.
    ///
    /// This routine fits the linear discriminant model when possible and falls
    /// back to the heuristic score otherwise, then computes spectrum q-values
    /// on the resulting discriminant ordering.
    fn spectrum_fdr(&self, features: &mut [TdcFeature]) -> usize {
        // Fit the linear discriminant model used for TDC spectrum ranking.
        //
        // TDC scoring should fail closed to the deterministic heuristic score if
        // LDA cannot be fit. This preserves search completion while making the
        // fallback visible in the logs.
        let score_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            score_psms(
                features,
                self.parameters.precursor_tol,
                self.decoy_free_mode,
            )
        }));

        let lda_ok = match score_res {
            Ok(Some(())) => true,
            Ok(None) => {
                log::warn!("linear model fitting failed, using heuristic score");
                false
            }
            Err(_) => {
                log::warn!(
                    "linear model fitting panicked, using heuristic score; \
                 this indicates an internal LDA edge case that should be audited"
                );
                false
            }
        };

        if !lda_ok {
            features.par_iter_mut().for_each(|feat| {
                let poisson = if feat.core.poisson_log10_p_value.is_finite() {
                    (-feat.core.poisson_log10_p_value).max(0.0).ln_1p() as f32
                } else {
                    0.0
                };

                feat.discriminant_score = poisson + feat.core.longest_y_pct / 3.0;
            });
        }

        features.par_sort_unstable_by(|a, b| {
            b.discriminant_score
                .total_cmp(&a.discriminant_score)
                .then_with(|| b.core.hyperscore.total_cmp(&a.core.hyperscore))
                .then_with(|| {
                    a.core
                        .poisson_log10_p_value
                        .total_cmp(&b.core.poisson_log10_p_value)
                })
                .then_with(|| a.core.psm_id.cmp(&b.core.psm_id))
        });

        sage_core::ml::qvalue::spectrum_q_value(features)
    }

    fn make_path<S: AsRef<str>>(&self, file_name: S) -> Url {
        self.parameters
            .output_directory
            .join(file_name.as_ref())
            .expect("valid output path segment")
    }

    fn search_processed_spectra(
        &self,
        scorer: &Scorer,
        msn_spectra: &[ProcessedSpectrum],
    ) -> Vec<FeatureCore> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = AtomicUsize::new(0);
        let start = Instant::now();

        let features: Vec<_> = msn_spectra
            .par_iter()
            .filter(|spec| spec.masses.len() >= self.parameters.min_peaks && spec.level == 2)
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
        msn_spectra: Vec<ProcessedSpectrum>,
        ms1_spectra: Vec<ProcessedSpectrum>,
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
        chunk: &[Url],
        chunk_idx: usize,
        batch_size: usize,
    ) -> SageResults {
        let spectra = self.read_processed_spectra(chunk, chunk_idx, batch_size);
        let features = self.search_processed_spectra(scorer, &spectra.1);
        self.complete_features(spectra.1, spectra.0, features)
    }

    fn read_processed_spectra(
        &self,
        chunk: &[Url],
        chunk_idx: usize,
        batch_size: usize,
    ) -> (Vec<ProcessedSpectrum>, Vec<ProcessedSpectrum>) {
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

        // SoA spectra preserve mobility per spectrum. This supports no-IMS,
        // full-IMS, and mixed-IMS inputs without dropping mobility globally.
        let ms1_spectra = spectra
            .ms1
            .into_iter()
            .map(|spectrum| sp.process(spectrum))
            .collect();

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

    pub fn run(self, parallel: usize, parquet: bool) -> anyhow::Result<telemetry::Telemetry> {
        self.run_with_candidate_pool(parallel, parquet, None)
            .map(|(telemetry, _)| telemetry)
    }

    pub fn run_with_candidate_pool(
        self,
        parallel: usize,
        parquet: bool,
        candidate_pool: Option<CandidatePoolRequest>,
    ) -> anyhow::Result<(telemetry::Telemetry, Option<CandidatePoolUsage>)> {
        let (telemetry, candidate_pool, _) =
            self.run_with_workflow_caches(parallel, parquet, candidate_pool, None)?;
        Ok((telemetry, candidate_pool))
    }

    pub fn run_with_workflow_caches(
        mut self,
        parallel: usize,
        parquet: bool,
        candidate_pool: Option<CandidatePoolRequest>,
        annotation_cache: Option<ExternalAnnotationCacheRequest>,
    ) -> anyhow::Result<(
        telemetry::Telemetry,
        Option<CandidatePoolUsage>,
        Option<ExternalAnnotationCacheUsage>,
    )> {
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

        let pool_identity = candidate_pool
            .as_ref()
            .map(|request| {
                let search = search_fingerprint(&self.parameters)?;
                anyhow::ensure!(
                    request.required_rank_depth <= search.retained_rank_depth,
                    "candidate pool retains ranks 1..={} but analysis requires rank {}",
                    search.retained_rank_depth,
                    request.required_rank_depth
                );
                let analysis = analysis_fingerprint(&self.parameters, &search)?;
                let directory = pool_directory(&request.root, &search);
                Ok::<_, anyhow::Error>((search, analysis, directory))
            })
            .transpose()?;

        let mut pool_was_loaded = false;
        let mut pool_usage = None;
        let annotation_search_fingerprint = if annotation_cache.is_some() {
            Some(match pool_identity.as_ref() {
                Some((search, _, _)) => search.clone(),
                None => search_fingerprint(&self.parameters)?,
            })
        } else {
            None
        };
        let mut annotation_usage = None;

        // Collect all results into a single container, or restore the immutable
        // pre-FDR candidates when an exact compatible pool already exists.
        let mut outputs = if let (Some(request), Some((search, analysis, directory))) =
            (candidate_pool.as_ref(), pool_identity.as_ref())
        {
            let compatible = if request.allow_reuse || request.require_existing {
                inspect_compatible_pool(directory, search, request.required_rank_depth)?
            } else {
                None
            };
            if request.require_existing && compatible.is_none() {
                anyhow::bail!(
                    "required existing candidate pool is missing or incompatible at {} (expected fingerprint={}, schema={}, retained rank depth >= {}); spectrum search fallback is disabled",
                    directory.display(),
                    search.digest,
                    crate::candidate_pool::CANDIDATE_ID_SCHEMA,
                    request.required_rank_depth
                );
            }
            if compatible.is_some() {
                anyhow::ensure!(
                    !self.parameters.quant.lfq
                        && self.parameters.quant.tmt.is_none()
                        && !self.parameters.annotate_matches,
                    "candidate-pool reuse currently supports identification/statistical stages only; LFQ, TMT, and matched-fragment output require a fresh search"
                );
                let (manifest, features) = load_pool(
                    directory,
                    search,
                    request.required_rank_depth,
                    &self.database,
                )?;
                let (
                    original_source_uris,
                    current_source_uris,
                    portable_identity_valid,
                    relocation_detected,
                ) = relocation_provenance(&manifest.search_fingerprint, search);
                pool_was_loaded = true;
                log::info!(
                    "candidate pool: reused {} candidates from {} (fingerprint={}, ranks=1..={})",
                    manifest.candidate_count,
                    directory.display(),
                    search.digest,
                    manifest.capabilities.retained_rank_depth
                );
                pool_usage = Some(CandidatePoolUsage {
                    search_fingerprint: search.digest.clone(),
                    analysis_fingerprint: analysis.digest.clone(),
                    manifest: manifest_path(directory),
                    payload: directory.join(&manifest.payload_file),
                    reused: true,
                    candidate_count: manifest.candidate_count,
                    retained_rank_depth: manifest.capabilities.retained_rank_depth,
                    original_source_uris,
                    current_source_uris,
                    portable_identity_valid,
                    relocation_detected,
                });
                SageResults {
                    features,
                    ..SageResults::default()
                }
            } else {
                self.batch_files(&scorer, parallel)
            }
        } else {
            self.batch_files(&scorer, parallel)
        };

        let filenames = self
            .parameters
            .mzml_paths
            .iter()
            .map(|url| {
                sage_cloudpath::filename(url)
                    .map(str::to_owned)
                    .unwrap_or_else(|| url.to_string())
            })
            .collect::<Vec<_>>();

        log::trace!("processing outputs");

        if self.decoy_free_mode {
            debug_assert!(
                !self.parameters.database.decoy_tag.is_empty(),
                "decoy_free mode requires non-empty database.decoy_tag"
            );

            // Decoy-Free inference excludes explicit decoy-labeled PSMs before model fitting.
            let n_before = outputs.features.len();
            outputs.features.retain(|feat| feat.label != -1);
            let n_dropped = n_before.saturating_sub(outputs.features.len());
            if n_dropped > 0 {
                log::info!("decoy_free mode: dropped {} decoy-labeled PSMs", n_dropped);
            }

            // Train RT and IMS models on target rank-1 PSMs only.
            let alignments = if self.parameters.predict_rt && !pool_was_loaded {
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

            if !pool_was_loaded {
                if let (Some(request), Some((search, analysis, directory))) =
                    (candidate_pool.as_ref(), pool_identity.as_ref())
                {
                    let manifest =
                        write_pool(directory, search, &outputs.features, &self.database)?;
                    log::info!(
                        "candidate pool: wrote {} candidates to {} (fingerprint={}, ranks=1..={})",
                        manifest.candidate_count,
                        directory.display(),
                        search.digest,
                        manifest.capabilities.retained_rank_depth
                    );
                    pool_usage = Some(CandidatePoolUsage {
                        search_fingerprint: search.digest.clone(),
                        analysis_fingerprint: analysis.digest.clone(),
                        manifest: manifest_path(directory),
                        payload: directory.join(&manifest.payload_file),
                        reused: false,
                        candidate_count: manifest.candidate_count,
                        retained_rank_depth: manifest.capabilities.retained_rank_depth,
                        original_source_uris: search
                            .spectra
                            .iter()
                            .map(|spectrum| spectrum.source.clone())
                            .collect(),
                        current_source_uris: search
                            .spectra
                            .iter()
                            .map(|spectrum| spectrum.source.clone())
                            .collect(),
                        portable_identity_valid: true,
                        relocation_detected: false,
                    });
                    anyhow::ensure!(
                        request.required_rank_depth <= manifest.capabilities.retained_rank_depth,
                        "new candidate pool does not satisfy required rank depth"
                    );
                }
            }

            // Convert generic PSM features to the Decoy-Free feature representation.
            let mut features: Vec<DfFeature> = outputs
                .features
                .into_par_iter()
                .map(|f| f.to_df())
                .collect();

            let mut fdr_settings = self.parameters.fdr.clone();
            let mut optimizer_evaluations = None;
            let mut optimizer_report = None;
            let mut optimizer_checkpoint_path = None;

            if self.parameters.write_pin {
                log::warn!(
					"write_pin=true was requested, but PIN output is not supported in decoy-free mode; \
					 PIN is a target-decoy rescoring format and will be skipped."
				);
            }

            // Pass 1: native Sage Decoy-Free.
            //
            // This first pass deliberately uses only Sage-native features. Its purpose is to
            // generate preliminary Decoy-Free p/PEP/q values while ranks 1..N are still present.
            // Those preliminary values are then used only to calibrate external feature
            // generators such as DeepLC/IM2Deep. MS2Rescore/TIMS2Rescore remains feature-only.
            let mut fitted_artifacts;
            if fdr_settings.null_window_optimizer.is_some() {
                let checkpoint_path = self.make_path("null_window_optimizer.checkpoint.json");
                optimizer_checkpoint_path = Some(checkpoint_path.clone());
                let checkpoint_fingerprint =
                    null_window_checkpoint_fingerprint(&features, &fdr_settings)?;
                let prior_evaluations = checkpoint_path
                    .to_file_path()
                    .ok()
                    .filter(|path| path.is_file())
                    .map(|path| -> anyhow::Result<_> {
                        let checkpoint: NullWindowOptimizerCheckpoint =
                            serde_json::from_slice(&std::fs::read(&path)?)
                                .with_context(|| format!("invalid optimizer checkpoint {}", path.display()))?;
                        if checkpoint.schema != "sage-null-window-checkpoint-v1" {
                            anyhow::bail!(
                                "unsupported optimizer checkpoint schema in {}",
                                path.display()
                            );
                        }
                        if checkpoint.fingerprint == checkpoint_fingerprint {
                            log::info!(
                                "DF null-window optimizer: resuming {} cached evaluations",
                                checkpoint.evaluations.len()
                            );
                            Ok(checkpoint.evaluations)
                        } else {
                            log::info!(
                                "DF null-window optimizer: checkpoint fingerprint changed; starting a new optimization"
                            );
                            Ok(Vec::new())
                        }
                    })
                    .transpose()?
                    .unwrap_or_default();
                let checkpoint_url = checkpoint_path.clone();
                let checkpoint_fingerprint_for_write = checkpoint_fingerprint.clone();
                let optimized = sage_core::decoy_free_fdr::optimize_null_window_resumable(
                    &features,
                    &fdr_settings,
                    &self.database,
                    prior_evaluations,
                    move |evaluations| {
                        let checkpoint = NullWindowOptimizerCheckpoint {
                            schema: "sage-null-window-checkpoint-v1".to_string(),
                            fingerprint: checkpoint_fingerprint_for_write.clone(),
                            evaluations: evaluations.to_vec(),
                        };
                        if let Ok(local_path) = checkpoint_url.to_file_path() {
                            return crate::provenance::write_json_atomic(&local_path, &checkpoint)
                                .map_err(|error| error.to_string());
                        }
                        let bytes = serde_json::to_vec_pretty(&checkpoint)
                            .map_err(|error| error.to_string())?;
                        sage_cloudpath::write_bytes_sync(&checkpoint_url, bytes)
                            .map_err(|error| error.to_string())
                    },
                )
                .map_err(anyhow::Error::msg)?;
                features = optimized.features;
                fdr_settings = optimized.settings;
                fitted_artifacts = optimized.artifacts;
                optimizer_evaluations = Some(optimized.evaluations);
                optimizer_report = Some(optimized.report);
            } else {
                let (first_pass_features, artifacts) =
                    sage_core::decoy_free_fdr::run_df_layers_with_artifacts(
                        &features,
                        &fdr_settings,
                        &self.database,
                    );
                features = first_pass_features;
                fitted_artifacts = artifacts;
            }

            if self.parameters.external_features.enabled
                && matches!(
                    self.parameters.external_features.use_mode,
                    ExternalFeatureUseMode::BoundedDfExperts
                )
            {
                let calibration = &fdr_settings.external_profile_calibration;
                anyhow::ensure!(
                    calibration.min_null_rank > 1
                        && calibration.max_null_rank >= calibration.min_null_rank,
                    "external MS2Rescore scoring requires both external_profile_min_null_rank and external_profile_max_null_rank with a valid window above rank 1; resolved {}..={} ({:?})",
                    calibration.min_null_rank,
                    calibration.max_null_rank,
                    calibration.provenance
                );
                anyhow::ensure!(
                    features.iter().any(|feature| {
                        feature.core.rank >= calibration.min_null_rank
                            && feature.core.rank <= calibration.max_null_rank
                    }),
                    "external MS2Rescore calibration window {}..={} has no retained candidates; increase external_features.max_rank/candidate-pool rank depth",
                    calibration.min_null_rank,
                    calibration.max_null_rank
                );
            }

            annotation_usage = maybe_add_external_features(
                &mut features,
                &self.parameters.external_features,
                &self.parameters.mzml_paths,
                &self.database,
                annotation_search_fingerprint
                    .as_ref()
                    .map(|fingerprint| fingerprint.digest.as_str()),
                annotation_cache.as_ref(),
            )
            .context("TIMS2/MS2Rescore external feature generation failed")?;

            // Pass 2: final Sage Decoy-Free after external features have been joined.
            //
            // At this stage the imported features are available for diagnostics and, later,
            // explicitly bounded auxiliary evidence. In diagnostics_only mode this second pass
            // should reproduce the native Sage statistical stream while logging the imported
            // feature separation.
            if self.parameters.external_features.enabled {
                log::info!(
                    "running second Decoy-Free pass after external TIMS2/MS2Rescore feature join"
                );

                let (rescored_features, rescored_artifacts) =
                    sage_core::decoy_free_fdr::run_df_layers_with_artifacts(
                        &features,
                        &fdr_settings,
                        &self.database,
                    );
                features = rescored_features;
                fitted_artifacts = rescored_artifacts;

                match self.parameters.external_features.use_mode {
                    ExternalFeatureUseMode::DiagnosticsOnly => {
                        log::info!(
                            "external TIMS2/MS2Rescore features are diagnostics/output only; no DF score update applied"
                        );
                    }

                    ExternalFeatureUseMode::ScoringCovariates => {
                        log::warn!(
                            "external_features.use_mode=scoring_covariates is not implemented yet; no DF score update applied"
                        );
                    }

                    ExternalFeatureUseMode::BoundedDfExperts => {
                        log::info!(
                            "applying external TIMS2/MS2Rescore features as bounded Decoy-Free expert evidence"
                        );

                        fitted_artifacts.external_ms2rescore = Some(
                            sage_core::decoy_free_fdr::apply_external_ms2rescore_bounded_experts(
                                &mut features,
                                &fdr_settings,
                            )
                            .map_err(anyhow::Error::msg)?,
                        );
                    }
                }
            }

            // Enforce the rank-1 contract for downstream reporting, aggregation,
            // quantification, and output.
            features.retain(|f| f.core.rank == 1);

            // Aggregate rank-1 Decoy-Free PSMs to peptide- and protein-level q-values.
            let (q_peptide, _ent_peptide) = sage_core::decoy_free_fdr::calculate_peptide_q_df(
                &mut features,
                &self.database,
                &fdr_settings,
                fdr_settings.peptide_fdr,
            );

            sage_core::decoy_free_fdr::apply_peptide_q_to_psm_reporting_df(
                &mut features,
                &fdr_settings,
            );

            let q_psm = features
                .iter()
                .filter(|f| {
                    f.core.label == 1
                        && f.decoy_free_q_value.unwrap_or(1.0) <= fdr_settings.peptide_fdr as f64
                })
                .count();

            log::info!(
                "discovered {} target peptide-spectrum matches at {}% FDR (Decoy-Free; using peptide_fdr as the primary DF reporting threshold)",
                q_psm,
                fdr_settings.peptide_fdr * 100.0
            );

            let q_protein = sage_core::decoy_free_fdr::calculate_protein_q_df(
                &mut features,
                &self.database,
                &fdr_settings,
            );

            let (level4_peptides, level4_psms) =
                sage_core::decoy_free_fdr::apply_hierarchical_reporting_df(
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

            if fdr_settings.hierarchical_reporting
                != sage_core::input::HierarchicalReportingMode::Off
            {
                log::info!(
                    "DF Level 4 reporting: protein_supported_peptides={} peptide_supported_psms={}",
                    level4_peptides,
                    level4_psms
                );
            }

            let emit_entrapment_counts = match fdr_settings.entrapment_report {
                sage_core::input::EntrapmentReportMode::Off => false,
                sage_core::input::EntrapmentReportMode::On => true,
                sage_core::input::EntrapmentReportMode::Auto => {
                    sage_core::decoy_free_fdr::has_entrapment_proteins(&self.database)
                }
            };

            if emit_entrapment_counts {
                let ent = sage_core::decoy_free_fdr::calculate_entrapment_counts_df(
                    &features,
                    &self.database,
                    &fdr_settings,
                    fdr_settings.peptide_fdr,
                    fdr_settings.protein_fdr,
                );

                log::info!(
                    "discovered {} entrapment peptide-spectrum matches at {}% FDR (Decoy-Free)",
                    ent.psms,
                    fdr_settings.peptide_fdr * 100.0
                );
                log::info!(
                    "discovered {} entrapment peptides at {}% FDR (Decoy-Free)",
                    ent.peptides,
                    fdr_settings.peptide_fdr * 100.0
                );
                log::info!(
                    "discovered {} entrapment proteins (Decoy-Free)",
                    ent.proteins
                );
            }

            let areas = alignments.as_ref().and_then(|alignments_ref| {
                if self.parameters.quant.lfq {
                    log::info!("Performing Decoy-Free LFQ...");
                    let mut areas_map = sage_core::lfq::build_feature_map(
                        self.parameters.quant.lfq_settings,
                        self.parameters.precursor_charge,
                        &features,
                        true,
                    )
                    .quantify(&self.database, &outputs.ms1, alignments_ref);

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

            if !parquet {
                features.par_sort_unstable_by(|a, b| {
                    a.decoy_free_q_value
                        .unwrap_or(f64::INFINITY)
                        .total_cmp(&b.decoy_free_q_value.unwrap_or(f64::INFINITY))
                        .then_with(|| b.core.hyperscore.total_cmp(&a.core.hyperscore))
                        .then_with(|| {
                            a.core
                                .poisson_log10_p_value
                                .total_cmp(&b.core.poisson_log10_p_value)
                        })
                });

                if let Some(artifact) = fitted_artifacts.lower_order.as_ref() {
                    let path = self.make_path("lower_order_model_artifact.json");
                    let bytes = serde_json::to_vec_pretty(artifact)
                        .context("serialize Lower Order model artifact")?;
                    sage_cloudpath::write_bytes_sync(&path, bytes)?;
                    self.parameters.output_paths.push(path);
                }
                let path = self.make_path("fitted_model_artifacts.json");
                let bytes = serde_json::to_vec_pretty(&fitted_artifacts)
                    .context("serialize fitted model artifacts")?;
                sage_cloudpath::write_bytes_sync(&path, bytes)?;
                self.parameters.output_paths.push(path);
                if let Some(evaluations) = optimizer_evaluations.as_ref() {
                    let path = self.make_path("null_window_evaluations.json");
                    let bytes = serde_json::to_vec_pretty(evaluations)
                        .context("serialize null-window evaluations")?;
                    sage_cloudpath::write_bytes_sync(&path, bytes)?;
                    self.parameters.output_paths.push(path);
                }
                if let Some(report) = optimizer_report.as_ref() {
                    let path = self.make_path("null_window_optimization.json");
                    let bytes = serde_json::to_vec_pretty(report)
                        .context("serialize null-window optimization report")?;
                    sage_cloudpath::write_bytes_sync(&path, bytes)?;
                    self.parameters.output_paths.push(path);
                }
                if let Some(path) = optimizer_checkpoint_path.as_ref() {
                    self.parameters.output_paths.push(path.clone());
                }

                self.parameters
                    .output_paths
                    .push(self.write_features_df(&features, &filenames)?);

                if self.parameters.annotate_matches {
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
                anyhow::bail!(
                    "Parquet output is not supported for decoy-free mode yet. \
					 Re-run without --parquet to write decoy-free results.sage.tsv, matched_fragments.sage.tsv, \
					 tmt.tsv, and lfq.tsv."
                );
            }
        } else {
            // In TDC mode, keep vanilla Sage RT/IMS training behavior.
            // Do not run the full LDA/TDC spectrum_fdr() path here. That path belongs to
            // final TDC FDR scoring, not the preliminary RT-training q-value prepass.
            let alignments = if self.parameters.predict_rt {
                // Vanilla-style preliminary q-value gate:
                // sort by raw spectrum p-value / Poisson-like score, convert to a temporary
                // TDC view only because qvalue::spectrum_q_value currently operates on
                // TdcFeature, then select the admitted PSM ids for RT/IMS training.
                outputs.features.par_sort_unstable_by(|a, b| {
                    a.poisson_log10_p_value
                        .total_cmp(&b.poisson_log10_p_value)
                        .then_with(|| b.hyperscore.total_cmp(&a.hyperscore))
                        .then_with(|| a.psm_id.cmp(&b.psm_id))
                });

                let mut tmp_tdc: Vec<TdcFeature> = outputs
                    .features
                    .iter()
                    .cloned()
                    .map(FeatureCore::to_tdc)
                    .collect();

                sage_core::ml::qvalue::spectrum_q_value(&mut tmp_tdc);

                let selected_psm_ids: HashSet<usize> = tmp_tdc
                    .iter()
                    .filter(|f| f.core.label == 1 && f.spectrum_q <= 0.01)
                    .map(|f| f.core.psm_id)
                    .collect();

                let selector =
                    |f: &FeatureCore| f.label == 1 && selected_psm_ids.contains(&f.psm_id);

                let local = sage_core::ml::retention_alignment::global_alignment_vanilla_compat(
                    &mut outputs.features,
                    self.parameters.mzml_paths.len(),
                    &selected_psm_ids,
                );

                let _ = sage_core::ml::retention_model::predict_vanilla_compat(
                    &self.database,
                    &mut outputs.features,
                    &selected_psm_ids,
                );

                let _ = sage_core::ml::mobility_model::predict_vanilla_compat(
                    &self.database,
                    &mut outputs.features,
                    selector,
                );

                Some(local)
            } else {
                None
            };

            // Convert generic PSM features to the TDC feature representation after
            // RT/IMS prediction has been applied to FeatureCore.
            let mut features: Vec<TdcFeature> = outputs
                .features
                .into_par_iter()
                .map(FeatureCore::to_tdc)
                .collect();

            // Compute final TDC spectrum-level q-values using the vanilla LDA/fallback path.
            let q_spectrum = self.spectrum_fdr(&mut features);

            // TDC-only picked peptide/protein-group inference. Decoy-free uses
            // its separate protein evidence and hierarchical reporting path.
            let q_peptide = sage_core::fdr::picked_peptide(&self.database, &mut features);
            sage_core::protein_grouping::generate_protein_groups(
                &self.database,
                &mut features,
                self.parameters.protein_grouping,
                Some(self.parameters.protein_grouping_peptide_fdr),
            );
            let q_protein = sage_core::fdr::picked_protein(&self.database, &mut features);
            let q_protein_group =
                sage_core::fdr::picked_protein_group(&self.database, &mut features);

            // Logging
            log::info!("discovered {} target PSMs at 1% FDR", q_spectrum);
            log::info!("discovered {} target peptides at 1% FDR", q_peptide);
            log::info!("discovered {} target proteins at 1% FDR", q_protein);
            log::info!(
                "discovered {} target protein groups at 1% FDR",
                q_protein_group
            );

            let emit_entrapment_counts = match self.parameters.fdr.entrapment_report {
                sage_core::input::EntrapmentReportMode::Off => false,
                sage_core::input::EntrapmentReportMode::On => true,
                sage_core::input::EntrapmentReportMode::Auto => {
                    sage_core::decoy_free_fdr::has_entrapment_proteins(&self.database)
                }
            };

            if emit_entrapment_counts {
                let ent = sage_core::decoy_free_fdr::calculate_entrapment_counts_tdc(
                    &features,
                    &self.database,
                    0.01,
                    0.01,
                    0.01,
                );

                log::info!("discovered {} entrapment PSMs at 1% FDR", ent.psms);
                log::info!("discovered {} entrapment peptides at 1% FDR", ent.peptides);
                log::info!("discovered {} entrapment proteins at 1% FDR", ent.proteins);
            }

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
                    sage_cloudpath::write_bytes_sync(&path, bytes)?;
                    self.parameters.output_paths.push(path);
                }

                // 2) fragments parquet (optional)
                if self.parameters.annotate_matches {
                    let path = self.make_path("matched_fragments.sage.parquet");
                    let cores: Vec<&FeatureCore> = features.iter().map(|f| &f.core).collect();
                    let bytes = sage_cloudpath::parquet::serialize_fragments(&cores, &filenames)?;
                    sage_cloudpath::write_bytes_sync(&path, bytes)?;
                    self.parameters.output_paths.push(path);
                }

                // 3) tmt parquet (if any)
                if !outputs.quant.is_empty() {
                    let path = self.make_path("tmt.parquet");
                    let bytes = sage_cloudpath::parquet::serialize_tmt(&outputs.quant, &filenames)?;
                    sage_cloudpath::write_bytes_sync(&path, bytes)?;
                    self.parameters.output_paths.push(path);
                }

                // 4) lfq parquet (if any)
                if let Some(areas) = areas {
                    let path = self.make_path("lfq.parquet");
                    let bytes =
                        sage_cloudpath::parquet::serialize_lfq(&areas, &filenames, &self.database)?;
                    sage_cloudpath::write_bytes_sync(&path, bytes)?;
                    self.parameters.output_paths.push(path);
                }
            }
        }

        // Final Metadata Write
        let path = self.make_path("results.json");
        self.parameters.output_paths.push(path.clone());
        println!("{}", serde_json::to_string_pretty(&self.parameters)?);

        let bytes = serde_json::to_vec_pretty(&self.parameters)?;
        sage_cloudpath::write_bytes_sync(&path, bytes)?;

        let run_time = (Instant::now() - self.start).as_secs();
        info!("finished in {}s", run_time);

        let telemetry = telemetry::Telemetry::new(
            self.parameters,
            self.database.peptides.len(),
            self.database.fragments.len(),
            parquet,
            run_time,
        );

        Ok((telemetry, pool_usage, annotation_usage))
    }

    // TDC output writers.

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
        record.push_field(feature.protein_groups.as_deref().unwrap_or("").as_bytes());
        record.push_field(
            itoa::Buffer::new()
                .format(peptide.proteins.len())
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format(feature.num_protein_groups)
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
        let ims_active = self.parameters.fdr.enable_ims_confidence_adjustment
            && core.ims.is_finite()
            && core.predicted_ims.is_finite()
            && core.delta_ims_model.is_finite()
            && !(core.ims == 0.0
                && core.predicted_ims == 0.0
                && (core.delta_ims_model - 0.999).abs() < 1e-6);

        let fmt_ims_or_nan = |v: f32| {
            if ims_active {
                v.to_string()
            } else {
                "NaN".to_string()
            }
        };

        record.push_field(fmt_ims_or_nan(core.ims).as_bytes());
        record.push_field(fmt_ims_or_nan(core.predicted_ims).as_bytes());
        record.push_field(fmt_ims_or_nan(core.delta_ims_model).as_bytes());
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
                .format((-core.poisson_log10_p_value).max(0.0).ln_1p())
                .as_bytes(),
        );

        // TDC-specific output columns.
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
        record.push_field(
            ryu::Buffer::new()
                .format(feature.protein_group_q)
                .as_bytes(),
        );
        record.push_field(ryu::Buffer::new().format(core.ms2_intensity).as_bytes());

        record
    }

    pub fn write_features_tdc(
        &self,
        features: &[TdcFeature],
        filenames: &[String],
    ) -> anyhow::Result<Url> {
        let path = self.make_path("results.sage.tsv");
        let mut wtr = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(vec![]);

        let headers = vec![
            "psm_id",
            "peptide",
            "proteins",
            "protein_groups",
            "num_proteins",
            "num_protein_groups",
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
            "protein_group_q",
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
        sage_cloudpath::write_bytes_sync(&path, bytes)?;
        Ok(path)
    }

    #[inline]
    fn df_is_entrapment_protein_key(protein_key: &str) -> bool {
        // Keep this in sync with decoy_free_fdr.rs::is_entrapment_str().
        protein_key.contains("|Ent_") || protein_key.contains("Ent_")
    }

    fn active_df_model_suffixes(&self) -> Vec<&'static str> {
        let fdr = &self.parameters.fdr;

        match fdr.model_fit {
            ModelFit::Moments => vec!["mom"],
            ModelFit::Mle => vec!["mle"],
            ModelFit::LowerOrder => vec!["lo"],
            ModelFit::Msfdr => vec!["msfdr"],
            ModelFit::Msfdr1Smix => vec!["1smix"],
            ModelFit::Msfdr2Smix => vec!["2smix"],
            ModelFit::Nokoi => vec!["nokoi"],

            ModelFit::Ensemble => {
                let mut v = Vec::new();

                if fdr.enable_moments {
                    v.push("mom");
                }
                if fdr.enable_mle {
                    v.push("mle");
                }
                if fdr.enable_lower_order {
                    v.push("lo");
                }
                if fdr.enable_msfdr_seeded {
                    v.push("msfdr");
                }
                if fdr.enable_msfdr_1smix {
                    v.push("1smix");
                }
                if fdr.enable_msfdr_2smix {
                    v.push("2smix");
                }
                if fdr.enable_nokoi {
                    v.push("nokoi");
                }

                // Ensemble is itself the active model fit.
                v.push("ensemble");

                v
            }
        }
    }

    fn df_dynamic_columns(&self) -> Vec<DfDynamicColumn> {
        let fdr = &self.parameters.fdr;
        let mut cols = Vec::new();

        cols.extend([
            DfDynamicColumn::Final("decoy_free_p_value"),
            DfDynamicColumn::Final("decoy_free_pep"),
            DfDynamicColumn::Final("decoy_free_score"),
            DfDynamicColumn::Final("decoy_free_q_value"),
            DfDynamicColumn::Final("decoy_free_peptide_q"),
            DfDynamicColumn::Final("decoy_free_protein_q"),
        ]);

        let entrapment_reporting_active = fdr.entrapment_report != EntrapmentReportMode::Off
            || fdr.hierarchical_entrapment_validation;

        let level4_reporting_active = fdr.hierarchical_reporting != HierarchicalReportingMode::Off;

        if entrapment_reporting_active {
            cols.push(DfDynamicColumn::ReportingFlag("decoy_free_is_entrapment"));
        }

        if level4_reporting_active {
            cols.extend([
                DfDynamicColumn::ReportingFlag("decoy_free_protein_supported_peptide"),
                DfDynamicColumn::ReportingFlag("decoy_free_peptide_supported_psm"),
            ]);
        }

        for suffix in self.active_df_model_suffixes() {
            cols.push(DfDynamicColumn::BaseModel(suffix));

            if fdr.enable_rt_confidence_adjustment {
                cols.push(DfDynamicColumn::RtModel(suffix));
            }

            if fdr.enable_ims_confidence_adjustment {
                cols.push(DfDynamicColumn::ImsModel(suffix));
            }

            if fdr.enable_peptide_reproducibility_rescue {
                cols.push(DfDynamicColumn::PeptideRescueModel(suffix));
            }

            if fdr.enable_protein_reproducibility_rescue {
                cols.push(DfDynamicColumn::ProteinRescueModel(suffix));
            }
        }

        cols
    }

    fn push_df_dynamic_headers(headers: &mut Vec<String>, cols: &[DfDynamicColumn]) {
        for col in cols {
            match col {
                DfDynamicColumn::Final(name) => {
                    headers.push((*name).to_string());
                }

                DfDynamicColumn::ReportingFlag(name) => {
                    headers.push((*name).to_string());
                }

                DfDynamicColumn::BaseModel(suffix) => {
                    headers.push(format!("p_{suffix}"));
                    headers.push(format!("q_{suffix}"));
                    headers.push(format!("pep_{suffix}"));
                }

                DfDynamicColumn::RtModel(suffix) => {
                    headers.push(format!("rt_adjust_p_{suffix}"));
                    headers.push(format!("rt_adjust_q_{suffix}"));
                    headers.push(format!("rt_adjust_pep_{suffix}"));
                }

                DfDynamicColumn::ImsModel(suffix) => {
                    headers.push(format!("ims_adjust_p_{suffix}"));
                    headers.push(format!("ims_adjust_q_{suffix}"));
                    headers.push(format!("ims_adjust_pep_{suffix}"));
                }

                DfDynamicColumn::PeptideRescueModel(suffix) => {
                    headers.push(format!("peptide_rescue_p_{suffix}"));
                    headers.push(format!("peptide_rescue_q_{suffix}"));
                    headers.push(format!("peptide_rescue_pep_{suffix}"));
                }

                DfDynamicColumn::ProteinRescueModel(suffix) => {
                    headers.push(format!("protein_rescue_p_{suffix}"));
                    headers.push(format!("protein_rescue_q_{suffix}"));
                    headers.push(format!("protein_rescue_pep_{suffix}"));
                }
            }
        }
    }

    fn df_base_model_values(
        feature: &DfFeature,
        suffix: &str,
    ) -> (Option<f64>, Option<f64>, Option<f64>) {
        match suffix {
            "mom" => (feature.p_mom, feature.q_mom, feature.pep_mom),
            "mle" => (feature.p_mle, feature.q_mle, feature.pep_mle),
            "lo" => (feature.p_lo, feature.q_lo, feature.pep_lo),
            "msfdr" => (feature.p_msfdr, feature.q_msfdr, feature.pep_msfdr),
            "1smix" => (feature.p_1smix, feature.q_1smix, feature.pep_1smix),
            "2smix" => (feature.p_2smix, feature.q_2smix, feature.pep_2smix),
            "nokoi" => (feature.p_nokoi, feature.q_nokoi, feature.pep_nokoi),
            "ensemble" => (feature.p_ensemble, feature.q_ensemble, feature.pep_ensemble),
            _ => (None, None, None),
        }
    }

    fn df_rt_model_values(
        feature: &DfFeature,
        suffix: &str,
    ) -> (Option<f64>, Option<f64>, Option<f64>) {
        match suffix {
            "mom" => (
                feature.rt_adjust_p_mom,
                feature.rt_adjust_q_mom,
                feature.rt_adjust_pep_mom,
            ),
            "mle" => (
                feature.rt_adjust_p_mle,
                feature.rt_adjust_q_mle,
                feature.rt_adjust_pep_mle,
            ),
            "lo" => (
                feature.rt_adjust_p_lo,
                feature.rt_adjust_q_lo,
                feature.rt_adjust_pep_lo,
            ),
            "msfdr" => (
                feature.rt_adjust_p_msfdr,
                feature.rt_adjust_q_msfdr,
                feature.rt_adjust_pep_msfdr,
            ),
            "1smix" => (
                feature.rt_adjust_p_1smix,
                feature.rt_adjust_q_1smix,
                feature.rt_adjust_pep_1smix,
            ),
            "2smix" => (
                feature.rt_adjust_p_2smix,
                feature.rt_adjust_q_2smix,
                feature.rt_adjust_pep_2smix,
            ),
            "nokoi" => (
                feature.rt_adjust_p_nokoi,
                feature.rt_adjust_q_nokoi,
                feature.rt_adjust_pep_nokoi,
            ),
            "ensemble" => (
                feature.rt_adjust_p_ensemble,
                feature.rt_adjust_q_ensemble,
                feature.rt_adjust_pep_ensemble,
            ),
            _ => (None, None, None),
        }
    }

    fn df_ims_model_values(
        feature: &DfFeature,
        suffix: &str,
    ) -> (Option<f64>, Option<f64>, Option<f64>) {
        match suffix {
            "mom" => (
                feature.ims_adjust_p_mom,
                feature.ims_adjust_q_mom,
                feature.ims_adjust_pep_mom,
            ),
            "mle" => (
                feature.ims_adjust_p_mle,
                feature.ims_adjust_q_mle,
                feature.ims_adjust_pep_mle,
            ),
            "lo" => (
                feature.ims_adjust_p_lo,
                feature.ims_adjust_q_lo,
                feature.ims_adjust_pep_lo,
            ),
            "msfdr" => (
                feature.ims_adjust_p_msfdr,
                feature.ims_adjust_q_msfdr,
                feature.ims_adjust_pep_msfdr,
            ),
            "1smix" => (
                feature.ims_adjust_p_1smix,
                feature.ims_adjust_q_1smix,
                feature.ims_adjust_pep_1smix,
            ),
            "2smix" => (
                feature.ims_adjust_p_2smix,
                feature.ims_adjust_q_2smix,
                feature.ims_adjust_pep_2smix,
            ),
            "nokoi" => (
                feature.ims_adjust_p_nokoi,
                feature.ims_adjust_q_nokoi,
                feature.ims_adjust_pep_nokoi,
            ),
            "ensemble" => (
                feature.ims_adjust_p_ensemble,
                feature.ims_adjust_q_ensemble,
                feature.ims_adjust_pep_ensemble,
            ),
            _ => (None, None, None),
        }
    }

    fn df_peptide_rescue_model_values(
        feature: &DfFeature,
        suffix: &str,
    ) -> (Option<f64>, Option<f64>, Option<f64>) {
        match suffix {
            "mom" => (
                feature.peptide_rescue_p_mom,
                feature.peptide_rescue_q_mom,
                feature.peptide_rescue_pep_mom,
            ),
            "mle" => (
                feature.peptide_rescue_p_mle,
                feature.peptide_rescue_q_mle,
                feature.peptide_rescue_pep_mle,
            ),
            "lo" => (
                feature.peptide_rescue_p_lo,
                feature.peptide_rescue_q_lo,
                feature.peptide_rescue_pep_lo,
            ),
            "msfdr" => (
                feature.peptide_rescue_p_msfdr,
                feature.peptide_rescue_q_msfdr,
                feature.peptide_rescue_pep_msfdr,
            ),
            "1smix" => (
                feature.peptide_rescue_p_1smix,
                feature.peptide_rescue_q_1smix,
                feature.peptide_rescue_pep_1smix,
            ),
            "2smix" => (
                feature.peptide_rescue_p_2smix,
                feature.peptide_rescue_q_2smix,
                feature.peptide_rescue_pep_2smix,
            ),
            "nokoi" => (
                feature.peptide_rescue_p_nokoi,
                feature.peptide_rescue_q_nokoi,
                feature.peptide_rescue_pep_nokoi,
            ),
            "ensemble" => (
                feature.peptide_rescue_p_ensemble,
                feature.peptide_rescue_q_ensemble,
                feature.peptide_rescue_pep_ensemble,
            ),
            _ => (None, None, None),
        }
    }

    fn df_protein_rescue_model_values(
        feature: &DfFeature,
        suffix: &str,
    ) -> (Option<f64>, Option<f64>, Option<f64>) {
        match suffix {
            "mom" => (
                feature.protein_rescue_p_mom,
                feature.protein_rescue_q_mom,
                feature.protein_rescue_pep_mom,
            ),
            "mle" => (
                feature.protein_rescue_p_mle,
                feature.protein_rescue_q_mle,
                feature.protein_rescue_pep_mle,
            ),
            "lo" => (
                feature.protein_rescue_p_lo,
                feature.protein_rescue_q_lo,
                feature.protein_rescue_pep_lo,
            ),
            "msfdr" => (
                feature.protein_rescue_p_msfdr,
                feature.protein_rescue_q_msfdr,
                feature.protein_rescue_pep_msfdr,
            ),
            "1smix" => (
                feature.protein_rescue_p_1smix,
                feature.protein_rescue_q_1smix,
                feature.protein_rescue_pep_1smix,
            ),
            "2smix" => (
                feature.protein_rescue_p_2smix,
                feature.protein_rescue_q_2smix,
                feature.protein_rescue_pep_2smix,
            ),
            "nokoi" => (
                feature.protein_rescue_p_nokoi,
                feature.protein_rescue_q_nokoi,
                feature.protein_rescue_pep_nokoi,
            ),
            "ensemble" => (
                feature.protein_rescue_p_ensemble,
                feature.protein_rescue_q_ensemble,
                feature.protein_rescue_pep_ensemble,
            ),
            _ => (None, None, None),
        }
    }

    fn push_opt_f64(record: &mut csv::ByteRecord, val: Option<f64>) {
        record.push_field(
            val.map(|v| v.to_string())
                .unwrap_or_else(|| "NaN".to_string())
                .as_bytes(),
        );
    }

    fn push_opt_bool(record: &mut csv::ByteRecord, val: Option<bool>) {
        record.push_field(
            val.map(|v| v.to_string())
                .unwrap_or_else(|| "NaN".to_string())
                .as_bytes(),
        );
    }

    fn push_df_triplet(
        record: &mut csv::ByteRecord,
        vals: (Option<f64>, Option<f64>, Option<f64>),
    ) {
        Self::push_opt_f64(record, vals.0);
        Self::push_opt_f64(record, vals.1);
        Self::push_opt_f64(record, vals.2);
    }

    fn push_external_feature_headers(headers: &mut Vec<String>) {
        headers.extend([
            "ms2rescore_ms2pip_pcc".to_string(),
            "ms2rescore_spectral_angle".to_string(),
            "ms2rescore_fragment_intensity_agreement".to_string(),
            "ms2rescore_deeplc_predicted_rt".to_string(),
            "ms2rescore_deeplc_calibrated_rt".to_string(),
            "ms2rescore_deeplc_rt_error".to_string(),
            "ms2rescore_deeplc_abs_rt_error".to_string(),
            "tims2rescore_im2deep_predicted_ccs".to_string(),
            "tims2rescore_observed_ccs".to_string(),
            "tims2rescore_abs_ccs_error".to_string(),
            "tims2rescore_pct_ccs_error".to_string(),
            "tims2rescore_predicted_ion_mobility".to_string(),
            "tims2rescore_observed_ion_mobility".to_string(),
            "tims2rescore_abs_ion_mobility_error".to_string(),
            "tims2rescore_pct_ion_mobility_error".to_string(),
            "ms2rescore_feature_joined".to_string(),
        ]);
    }

    fn push_external_feature_values(record: &mut csv::ByteRecord, core: &FeatureCore) {
        let ext = core.external_features;

        record.push_field(
            ryu::Buffer::new()
                .format(ext.ms2rescore_ms2pip_pcc)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.ms2rescore_spectral_angle)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.ms2rescore_fragment_intensity_agreement)
                .as_bytes(),
        );

        record.push_field(
            ryu::Buffer::new()
                .format(ext.ms2rescore_deeplc_predicted_rt)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.ms2rescore_deeplc_calibrated_rt)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.ms2rescore_deeplc_rt_error)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.ms2rescore_deeplc_abs_rt_error)
                .as_bytes(),
        );

        record.push_field(
            ryu::Buffer::new()
                .format(ext.tims2rescore_im2deep_predicted_ccs)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.tims2rescore_observed_ccs)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.tims2rescore_abs_ccs_error)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.tims2rescore_pct_ccs_error)
                .as_bytes(),
        );

        record.push_field(
            ryu::Buffer::new()
                .format(ext.tims2rescore_predicted_ion_mobility)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.tims2rescore_observed_ion_mobility)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.tims2rescore_abs_ion_mobility_error)
                .as_bytes(),
        );
        record.push_field(
            ryu::Buffer::new()
                .format(ext.tims2rescore_pct_ion_mobility_error)
                .as_bytes(),
        );

        record.push_field(ext.ms2rescore_feature_joined.to_string().as_bytes());
    }

    // --- DF WRITERS (Decoy-Free) ---
    fn serialize_df_feature(
        &self,
        feature: &DfFeature,
        filenames: &[String],
        df_cols: &[DfDynamicColumn],
    ) -> csv::ByteRecord {
        let mut record = csv::ByteRecord::new();
        let core = &feature.core;

        // Core Columns
        record.push_field(itoa::Buffer::new().format(core.psm_id).as_bytes());
        let peptide = &self.database[core.peptide_idx];
        let protein_key = peptide.proteins(&self.database.decoy_tag, self.database.generate_decoys);

        record.push_field(peptide.to_string().as_bytes());
        record.push_field(protein_key.as_bytes());
        record.push_field(feature.protein_groups.as_deref().unwrap_or("").as_bytes());
        record.push_field(
            itoa::Buffer::new()
                .format(peptide.proteins.len())
                .as_bytes(),
        );
        record.push_field(
            itoa::Buffer::new()
                .format(feature.num_protein_groups)
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
        let ims_active = self.parameters.fdr.enable_ims_confidence_adjustment
            && core.ims.is_finite()
            && core.predicted_ims.is_finite()
            && core.delta_ims_model.is_finite()
            && !(core.ims == 0.0
                && core.predicted_ims == 0.0
                && (core.delta_ims_model - 0.999).abs() < 1e-6);

        let fmt_ims_or_nan = |v: f32| {
            if ims_active {
                v.to_string()
            } else {
                "NaN".to_string()
            }
        };

        record.push_field(fmt_ims_or_nan(core.ims).as_bytes());
        record.push_field(fmt_ims_or_nan(core.predicted_ims).as_bytes());
        record.push_field(fmt_ims_or_nan(core.delta_ims_model).as_bytes());
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
                .format((-core.poisson_log10_p_value).max(0.0).ln_1p())
                .as_bytes(),
        );

        // Write MS2 Intensity
        record.push_field(ryu::Buffer::new().format(core.ms2_intensity).as_bytes());

        if self.parameters.external_features.enabled {
            Self::push_external_feature_values(&mut record, core);
        }

        // Decoy-Free output columns.
        for col in df_cols {
            match col {
                DfDynamicColumn::Final(name) => match *name {
                    "decoy_free_p_value" => {
                        Self::push_opt_f64(&mut record, feature.decoy_free_p_value)
                    }
                    "decoy_free_pep" => Self::push_opt_f64(&mut record, feature.decoy_free_pep),
                    "decoy_free_score" => Self::push_opt_f64(&mut record, feature.decoy_free_score),
                    "decoy_free_q_value" => {
                        Self::push_opt_f64(&mut record, feature.decoy_free_q_value)
                    }
                    "decoy_free_peptide_q" => {
                        Self::push_opt_f64(&mut record, feature.decoy_free_peptide_q)
                    }
                    "decoy_free_protein_q" => {
                        Self::push_opt_f64(&mut record, feature.decoy_free_protein_q)
                    }
                    _ => Self::push_opt_f64(&mut record, None),
                },

                DfDynamicColumn::ReportingFlag(name) => match *name {
                    "decoy_free_is_entrapment" => {
                        record.push_field(
                            Self::df_is_entrapment_protein_key(&protein_key)
                                .to_string()
                                .as_bytes(),
                        );
                    }

                    "decoy_free_protein_supported_peptide" => Self::push_opt_bool(
                        &mut record,
                        feature.decoy_free_protein_supported_peptide,
                    ),

                    "decoy_free_peptide_supported_psm" => {
                        Self::push_opt_bool(&mut record, feature.decoy_free_peptide_supported_psm)
                    }

                    _ => Self::push_opt_bool(&mut record, None),
                },

                DfDynamicColumn::BaseModel(suffix) => {
                    Self::push_df_triplet(&mut record, Self::df_base_model_values(feature, suffix));
                }

                DfDynamicColumn::RtModel(suffix) => {
                    Self::push_df_triplet(&mut record, Self::df_rt_model_values(feature, suffix));
                }

                DfDynamicColumn::ImsModel(suffix) => {
                    Self::push_df_triplet(&mut record, Self::df_ims_model_values(feature, suffix));
                }

                DfDynamicColumn::PeptideRescueModel(suffix) => {
                    Self::push_df_triplet(
                        &mut record,
                        Self::df_peptide_rescue_model_values(feature, suffix),
                    );
                }

                DfDynamicColumn::ProteinRescueModel(suffix) => {
                    Self::push_df_triplet(
                        &mut record,
                        Self::df_protein_rescue_model_values(feature, suffix),
                    );
                }
            }
        }

        record
    }

    pub fn write_features_df(
        &self,
        features: &[DfFeature],
        filenames: &[String],
    ) -> anyhow::Result<Url> {
        let path = self.make_path("results.sage.tsv");
        let mut wtr = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(vec![]);

        let mut headers: Vec<String> = vec![
            "psm_id",
            "peptide",
            "proteins",
            "protein_groups",
            "num_proteins",
            "num_protein_groups",
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
        ]
        .into_iter()
        .map(String::from)
        .collect();

        if self.parameters.external_features.enabled {
            Self::push_external_feature_headers(&mut headers);
        }

        let df_cols = self.df_dynamic_columns();
        Self::push_df_dynamic_headers(&mut headers, &df_cols);

        wtr.write_byte_record(&csv::ByteRecord::from(
            headers.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))?;

        let records: Vec<csv::ByteRecord> = features
            .par_iter()
            .map(|feat| self.serialize_df_feature(feat, filenames, &df_cols))
            .collect();

        for record in records {
            wtr.write_byte_record(&record)?;
        }

        wtr.flush()?;
        let bytes = wtr.into_inner()?;
        sage_cloudpath::write_bytes_sync(&path, bytes)?;
        Ok(path)
    }

    // --- SHARED WRITERS ---

    pub fn write_fragments(&self, features: &[&FeatureCore]) -> anyhow::Result<Url> {
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
        sage_cloudpath::write_bytes_sync(&path, bytes)?;
        Ok(path)
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

    pub fn write_pin(&self, features: &[TdcFeature], filenames: &[String]) -> anyhow::Result<Url> {
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
        sage_cloudpath::write_bytes_sync(&path, bytes)?;
        Ok(path)
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
        let ims_active = self.parameters.fdr.enable_ims_confidence_adjustment
            && core.ims.is_finite()
            && core.predicted_ims.is_finite()
            && core.delta_ims_model.is_finite()
            && !(core.ims == 0.0
                && core.predicted_ims == 0.0
                && (core.delta_ims_model - 0.999).abs() < 1e-6);

        let predicted_ims_out = if ims_active {
            core.predicted_ims.to_string()
        } else {
            "NaN".to_string()
        };

        let delta_ims_out = if ims_active {
            core.delta_ims_model.clamp(0.0, 1.0).sqrt().to_string()
        } else {
            "NaN".to_string()
        };

        record.push_field(predicted_ims_out.as_bytes());
        record.push_field(delta_ims_out.as_bytes());
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
                .format((-core.poisson_log10_p_value).max(0.0).ln_1p())
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
    ) -> anyhow::Result<Url> {
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
        sage_cloudpath::write_bytes_sync(&path, bytes)?;
        Ok(path)
    }

    pub fn write_tmt(&self, quant: &[TmtQuant], filenames: &[String]) -> anyhow::Result<Url> {
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
        sage_cloudpath::write_bytes_sync(&path, bytes)?;
        Ok(path)
    }
}
