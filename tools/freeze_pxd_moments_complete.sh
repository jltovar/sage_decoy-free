#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAGE_BIN="${SAGE_BIN:-${REPO_ROOT}/target/release/sage}"
VALIDATION_ROOT="${VALIDATION_ROOT:-/mnt/d/OneDrive/Search_Data_Files/SAGE_DECOYFREE/VALIDATION}"
FASTA_ROOT="${FASTA_ROOT:-/mnt/d/OneDrive/FASTA_Files/FASTA}"
PXD="${VALIDATION_ROOT}/PXD001468"
MOMENTS="${PXD}/DECOYFREE/MOMENTS"
OUTPUT="${1:-${REPO_ROOT}/validation/baselines/pxd001468_moments_corrected_complete_2026-08-05.json}"

[[ -x "$SAGE_BIN" ]] || { echo "Missing Sage executable: $SAGE_BIN" >&2; exit 1; }

"$SAGE_BIN" freeze-baseline \
  --output "$OUTPUT" \
  --status corrected_selected_outputs_complete \
  "$MOMENTS/validation_PXD001468_ent_decoyfree.json" \
  "$MOMENTS/best_result.txt" \
  "$MOMENTS/cumulative_results_long.csv" \
  "$MOMENTS/cumulative_results_wide.csv" \
  "$MOMENTS/search_trace.txt" \
  "$MOMENTS/json/run_026.json" \
  "$MOMENTS/logs/run_026.log" \
  "$MOMENTS/results.json" \
  "$MOMENTS/results.sage.tsv" \
  "$MOMENTS/2nd_ms2rescore_validation_PXD001468_ent_decoyfree.json" \
  "$MOMENTS/2nd_ms2rescore/results.json" \
  "$MOMENTS/2nd_ms2rescore/results.sage.tsv" \
  "$MOMENTS/validation_PXD001468_no_ent_decoyfree.json" \
  "$MOMENTS/3rd_no_ent/results.json" \
  "$MOMENTS/3rd_no_ent/results.sage.tsv" \
  "$PXD/DECOY/validation_PXD001468_ent_decoy.json" \
  "$PXD/DECOY/results.json" \
  "$PXD/DECOY/results.sage.tsv" \
  "$PXD/DECOY/validation_PXD001468_no_ent_decoy.json" \
  "$PXD/DECOY/3rd_no_ent/results.json" \
  "$PXD/DECOY/3rd_no_ent/results.sage.tsv" \
  "$PXD/DECOY/VANILLA/vanilla_validation_PXD001468_ent_decoy.json" \
  "$PXD/DECOY/VANILLA/results.json" \
  "$PXD/DECOY/VANILLA/results.sage.tsv" \
  "$PXD/DECOY/VANILLA/vanilla_validation_PXD001468_no_ent_decoy.json" \
  "$PXD/DECOY/VANILLA/3rd_no_ent/results.json" \
  "$PXD/DECOY/VANILLA/3rd_no_ent/results.sage.tsv" \
  "$FASTA_ROOT/validation/PXD001468.fasta" \
  "$FASTA_ROOT/target_entrapment/PXD001468_foreign_species_entrapment_protein.fasta" \
  "$FASTA_ROOT/target_entrapment/PXD001468_foreign_species_entrapment_protein.log"

echo "Frozen corrected PXD Moments selected outputs: $OUTPUT"
