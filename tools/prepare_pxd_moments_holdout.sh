#!/usr/bin/env bash
set -euo pipefail

if [[ "${SAGE_ALLOW_CROSS_DATASET_DIAGNOSTIC:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
This legacy helper imports an ISB-fitted artifact into PXD and is diagnostic-only.
It is not part of the dataset-local Decoy-Free refactor and cannot produce release evidence.

For historical reproduction only, set:
  SAGE_ALLOW_CROSS_DATASET_DIAGNOSTIC=1
EOF
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISB_ROOT="${1:-/mnt/d/OneDrive/Search_Data_Files/SAGE_DECOYFREE/VALIDATION/NATIVE_WORKFLOW/ISB18/DEVELOPMENT_PARITY_2026-08-05}"
TEMPLATE="${2:-${REPO_ROOT}/validation/manifests/pxd001468_moments_isb_locked_holdout_wsl.template.json}"
OUTPUT="${3:-${REPO_ROOT}/validation/manifests/pxd001468_moments_isb_locked_diagnostic_wsl_2026-08-05.json}"

STATE="${ISB_ROOT}/workflow.state.json"
PARITY="${ISB_ROOT}/validation.parity.json"

[[ -f "$STATE" ]] || { echo "Missing ISB workflow state: $STATE" >&2; exit 1; }
[[ -f "$PARITY" ]] || { echo "Missing ISB parity evidence: $PARITY" >&2; exit 1; }

if ! jq -e '
  [
    .[]
    | select(
        .native_method == "moments"
        and (.stage == "optimized" or .stage == "ms2rescore")
        and (.layer == "raw_q" or .layer == "level4")
      )
  ] as $moments
  | ($moments | length) == 4
    and all($moments[]; .within_tolerance == true)
' "$PARITY" >/dev/null; then
  echo "ISB Moments parity has not passed for optimized/MS2Rescore raw and Level-4 evidence; refusing to prepare the PXD Moments holdout." >&2
  exit 1
fi

CALIBRATION_STAGE="$({
  jq -r '
    .validation[]
    | select(.method == "moments" and .stage == "target_only" and .layer == "level4")
    | .calibration_stage
  ' "$STATE"
} | head -n 1)"

case "$CALIBRATION_STAGE" in
  optimized)
    MS2_POLICY="never"
    ;;
  ms2rescore)
    MS2_POLICY="always"
    ;;
  *)
    echo "Unexpected or missing Moments calibration stage: $CALIBRATION_STAGE" >&2
    exit 1
    ;;
esac

ARTIFACT="${ISB_ROOT}/moments/${CALIBRATION_STAGE}/fitted_model_artifacts.json"
[[ -f "$ARTIFACT" ]] || { echo "Missing locked Moments artifact: $ARTIFACT" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"
TEMPORARY="${OUTPUT}.tmp"
jq \
  --arg artifact "$ARTIFACT" \
  --arg parity "$PARITY" \
  --arg policy "$MS2_POLICY" \
  '.locked_expert_artifacts.moments = $artifact
   | .validation.external_parity_evidence = $parity
   | .validation.diagnostic_only = true
   | .artifact_reuse_policy = "cross_dataset_diagnostic"
   | .models[0].ms2rescore = $policy' \
  "$TEMPLATE" > "$TEMPORARY"
mv "$TEMPORARY" "$OUTPUT"

echo "Prepared diagnostic-only cross-dataset PXD Moments manifest: $OUTPUT"
echo "Locked calibration stage: $CALIBRATION_STAGE"
echo "Locked artifact: $ARTIFACT"
