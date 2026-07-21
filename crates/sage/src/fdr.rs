//! False discovery rate control using double-competition (picked-peptide &
//! picked-protein) approaches
//!
//! Lin et al., https://pubmed.ncbi.nlm.nih.gov/36166314/
//! Savitski et al., https://pubmed.ncbi.nlm.nih.gov/25987413/

use crate::database::{IndexedDatabase, PeptideIx};
use crate::lfq::PrecursorId;
use crate::ml::kde::Estimator;
use crate::scoring::TdcFeature;
use fnv::FnvHashMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::BuildHasher;

#[derive(Copy, Clone, Debug)]
pub struct Competition<Ix> {
    pub forward: f32,
    pub foward_ix: Option<Ix>,
    pub reverse: f32,
    pub reverse_ix: Option<Ix>,
}

struct Row<Ix> {
    ix: Ix,
    decoy: bool,
    score: f32,
    q: f32,
}

impl<Ix: Default + Send> Default for Competition<Ix> {
    fn default() -> Self {
        Self {
            forward: f32::MIN,
            reverse: f32::MIN,
            foward_ix: None,
            reverse_ix: None,
        }
    }
}

impl<Ix: Default + Send> Competition<Ix> {
    fn score(&self) -> f32 {
        self.forward.max(self.reverse)
    }

    fn is_decoy(&self) -> bool {
        self.reverse >= self.forward
    }

    fn fit_kde<K, B>(scores: &HashMap<K, Self, B>) -> Estimator {
        let (scores, decoys): (Vec<f64>, Vec<bool>) = scores
            .values()
            .map(|score| (score.score() as f64, score.is_decoy()))
            .unzip();
        crate::ml::kde::Builder::default().build(&scores, &decoys)
    }

    fn assign_q_value<K, B>(
        scores: HashMap<K, Self, B>,
        threshold: f32,
    ) -> (HashMap<Ix, f32, B>, usize)
    where
        K: Eq + std::hash::Hash + Send,
        Ix: Eq + std::hash::Hash,
        B: BuildHasher + Default + Send,
    {
        let estimator = Self::fit_kde(&scores);
        let mut scores = scores
            .into_par_iter()
            .flat_map(|(_, comp)| {
                [
                    (comp.foward_ix, false, comp.forward),
                    (comp.reverse_ix, true, comp.reverse),
                ]
            })
            .filter_map(|(ix, decoy, score)| {
                ix.map(|ix| Row {
                    ix,
                    decoy,
                    score,
                    q: 1.0,
                })
            })
            .collect::<Vec<Row<Ix>>>();

        scores.par_sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut decoy = 1.0;
        let mut target = 0.0;
        for score in scores.iter_mut() {
            let pep = estimator.posterior_error(score.score as f64) as f32;

            // Cumulative sum of PEP ~ # of decoys
            decoy += pep;
            if !score.decoy {
                target += 1.0;
            }
            score.q = decoy / target;
        }
        // Q-value is the minimum q-value at any given score threshold
        // `q = q[::-1].cummin()[::-1] in python`
        let mut q_min = 1.0f32;
        let mut passing = 0;
        for score in scores.iter_mut().rev() {
            q_min = q_min.min(score.q);
            score.q = q_min;
            if q_min <= threshold && !score.decoy {
                passing += 1;
            }
        }

        (
            scores
                .into_par_iter()
                .map(|score| (score.ix, score.q))
                .collect(),
            passing,
        )
    }
}

pub fn picked_peptide(db: &IndexedDatabase, features: &mut [TdcFeature]) -> usize {
    let mut map: FnvHashMap<String, Competition<PeptideIx>> = FnvHashMap::default();
    for feat in features.iter() {
        let peptide = &db[feat.core.peptide_idx];
        // Only reverse the peptide sequence if we generated decoys ourselves
        let key = match db.generate_decoys && peptide.decoy {
            true => peptide.reverse().to_string(),
            false => peptide.to_string(),
        };

        let entry = map.entry(key).or_default();
        match peptide.decoy {
            true => {
                entry.reverse = entry.reverse.max(feat.discriminant_score);
                entry.reverse_ix = Some(feat.core.peptide_idx);
            }
            false => {
                entry.forward = entry.forward.max(feat.discriminant_score);
                entry.foward_ix = Some(feat.core.peptide_idx);
            }
        }
    }

    let (scores, passing) = Competition::assign_q_value(map, 0.01);

    features.par_iter_mut().for_each(|feat| {
        feat.peptide_q = scores[&feat.core.peptide_idx];
    });

    passing
}

pub fn picked_protein(db: &IndexedDatabase, features: &mut [TdcFeature]) -> usize {
    // Critical: All non-proteotypic, non-unique, or shared peptides are discarded
    // else the assumptions of picked protein FDR are invalid. Shared peptides are
    // still reported, albeit with protein FDR = 1.0
    let mut map: FnvHashMap<_, Competition<String>> = FnvHashMap::default();
    for feat in features
        .iter()
        .filter(|x| db[x.core.peptide_idx].proteins.len() == 1)
    {
        let peptide = &db[feat.core.peptide_idx];
        let decoy = peptide.decoy;
        let entry = map.entry(&peptide.proteins).or_default();
        let proteins = peptide.proteins(&db.decoy_tag, db.generate_decoys);
        match decoy {
            true => {
                entry.reverse = entry.reverse.max(feat.discriminant_score);
                entry.reverse_ix = Some(proteins);
            }
            false => {
                entry.forward = entry.forward.max(feat.discriminant_score);
                entry.foward_ix = Some(proteins);
            }
        }
    }

    if map.is_empty() {
        return 0;
    }
    let (scores, passing) = Competition::assign_q_value(map, 0.01);

    features
        .par_iter_mut()
        .filter(|x| db[x.core.peptide_idx].proteins.len() == 1)
        .for_each(|feat| {
            let proteins = db[feat.core.peptide_idx].proteins(&db.decoy_tag, db.generate_decoys);
            if let Some(q) = scores.get(&proteins) {
                feat.protein_q = *q;
            }
        });

    passing
}

pub fn picked_protein_group(db: &IndexedDatabase, features: &mut [TdcFeature]) -> usize {
    // Critical: All non-proteotypic, non-unique, or shared peptides are discarded
    // else the assumptions of picked group FDR are invalid. Shared peptides are
    // still reported, albeit with protein group FDR = 1.0
    let mut map: FnvHashMap<_, Competition<String>> = FnvHashMap::default();
    for feat in features
        .iter()
        .filter(|x| x.num_protein_groups == 1 && x.protein_groups.is_some())
    {
        let decoy = db[feat.core.peptide_idx].decoy;
        let entry = map.entry(feat.protein_groups.clone()).or_default();
        match decoy {
            true => {
                entry.reverse = entry.reverse.max(feat.discriminant_score);
                entry.reverse_ix = feat.protein_groups.clone();
            }
            false => {
                entry.forward = entry.forward.max(feat.discriminant_score);
                entry.foward_ix = feat.protein_groups.clone();
            }
        }
    }

    if map.is_empty() {
        return 0;
    }
    let (scores, passing) = Competition::assign_q_value(map, 0.01);

    features
        .par_iter_mut()
        .filter(|x| x.num_protein_groups == 1 && x.protein_groups.is_some())
        .for_each(|feat| {
            if let Some(protein_groups) = feat.protein_groups.as_deref() {
                if let Some(q) = scores.get(protein_groups) {
                    feat.protein_group_q = *q;
                }
            }
        });

    passing
}

pub fn picked_precursor<H: BuildHasher>(
    peaks: &mut HashMap<(PrecursorId, bool), (crate::lfq::Peak, Vec<f64>), H>,
) -> usize {
    let mut scores = peaks
        .par_iter()
        .map(|(key, (peak, _))| Row {
            ix: key.0,
            decoy: key.1,
            score: peak.score as f32,
            q: 1.0,
        })
        .collect::<Vec<_>>();

    scores.par_sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut decoy = 1.0;
    let mut target = 0.0;
    for score in scores.iter_mut() {
        match score.decoy {
            true => decoy += 1.0,
            false => target += 1.0,
        };
        score.q = decoy / target;
    }

    let mut q_min: f32 = 1.0;
    let mut passing = 0;
    let mut precursor_q = HashMap::new();

    for score in scores.iter_mut().rev() {
        q_min = q_min.min(score.q);
        score.q = q_min;
        if q_min <= 0.05 && !score.decoy {
            passing += 1;
        }
        precursor_q.insert((score.ix, score.decoy), score.q);
    }

    for ((id, decoy), (peak, _)) in peaks.iter_mut() {
        if let Some(q) = precursor_q.get(&(*id, *decoy)) {
            peak.q_value = *q;
        }
    }

    passing
}
