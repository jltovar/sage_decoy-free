use crate::scoring::TdcFeature;

#[inline(always)]
fn df_qvalue_trace_enabled() -> bool {
    cfg!(feature = "decoy_free_debug")
}

/// Assign q_values in place to a set of TDC PSMs, returning the number of PSMs
/// q <= 0.01
///
/// # Invariants
/// * `scores` must be sorted in descending order (e.g. best PSM is first)
pub fn spectrum_q_value(scores: &mut [TdcFeature]) -> usize {
    let mut decoy = 0;
    let mut target = 0;

        for score in scores.iter_mut() {
        if score.core.label == -1 {
            decoy += 1;
        } else {
            target += 1;
        }

        score.spectrum_q = if target > 0 {
            decoy as f32 / target as f32
        } else {
            1.0
        };
    }

    // Reverse slice, and calculate the cumulative minimum
    let mut q_min = 1.0f32;
    let mut passing = 0;

    for score in scores.iter_mut().rev() {
        q_min = q_min.min(score.spectrum_q);

        // DIAGNOSTIC: shows when/where the cumulative-min step overwrites spectrum_q
        if df_qvalue_trace_enabled() {
            log::trace!(
				"ml::qvalue: forcing spectrum_q=q_min (psm_id={} label={} rank={} old_q={} q_min={})",
				score.core.psm_id,
				score.core.label,
				score.core.rank,
				score.spectrum_q,
				q_min
			);
        }

        score.spectrum_q = q_min;
        if q_min <= 0.01 {
            passing += 1;
        }
    }

    passing
}
