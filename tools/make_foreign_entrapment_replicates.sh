#!/usr/bin/env bash
# Build independently seeded one-fold foreign-species entrapment FASTAs.
set -euo pipefail

TARGET_DB=""
FOREIGN_DB=""
OUTPUT_DIR=""
OUTPUT_PREFIX="foreign_entrapment"
JAR="${HOME}/fdrbench/fdrbench-0.0.4.jar"
SEEDS="2001 2002 2003"
ENT_LABEL="Ent_"

usage() {
  cat <<'EOF'
Usage:
  make_foreign_entrapment_replicates.sh \
    --target-db FILE --foreign-db FILE --output-dir DIR [options]

Options:
  --output-prefix NAME   default: foreign_entrapment
  --jar FILE             default: ~/fdrbench/fdrbench-0.0.4.jar
  --seeds "N N N"        default: "2001 2002 2003"
  --ent-label PREFIX     default: Ent_

The digestion settings match the current PXD001468 validation configuration:
trypsin, two missed cleavages, peptide length 7-50, I-to-L conversion,
protein-level one-fold foreign entrapment, and no shared peptides.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-db) TARGET_DB="$2"; shift 2 ;;
    --foreign-db) FOREIGN_DB="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
    --jar) JAR="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --ent-label) ENT_LABEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$TARGET_DB" ]] || { echo "Missing --target-db" >&2; exit 2; }
[[ -n "$FOREIGN_DB" ]] || { echo "Missing --foreign-db" >&2; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { echo "Missing --output-dir" >&2; exit 2; }
[[ -f "$TARGET_DB" ]] || { echo "Target FASTA not found: $TARGET_DB" >&2; exit 2; }
[[ -f "$FOREIGN_DB" ]] || { echo "Foreign FASTA not found: $FOREIGN_DB" >&2; exit 2; }
[[ -f "$JAR" ]] || { echo "FDRBench jar not found: $JAR" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/${OUTPUT_PREFIX}_manifest.tsv"
printf 'seed\tfasta\tsha256\n' > "$MANIFEST"

for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
  stem="${OUTPUT_PREFIX}_seed_${seed}_protein"
  raw="$OUTPUT_DIR/${stem}_raw.fasta"
  final="$OUTPUT_DIR/${stem}.fasta"
  log="$OUTPUT_DIR/${stem}.log"

  cmd=(java -jar "$JAR"
    -db "$TARGET_DB"
    -o "$raw"
    -ms "$FOREIGN_DB"
    -enzyme 1
    -miss_c 2
    -minLength 7
    -maxLength 50
    -level protein
    -fold 1
    -seed "$seed"
    -I2L
    -ns
    -debug
  )

  printf 'Running FDRBench seed=%s\n' "$seed" | tee "$log"
  printf 'Command: ' | tee -a "$log"
  printf '%q ' "${cmd[@]}" | tee -a "$log"
  printf '\n' | tee -a "$log"
  "${cmd[@]}" 2>&1 | tee -a "$log"

  perl -pe 's/^(>[^|]+\|)(?=.*_p_target)/$1'"${ENT_LABEL}"'/' "$raw" > "$final"
  checksum=$(sha256sum "$final" | awk '{print $1}')
  printf '%s\t%s\t%s\n' "$seed" "$final" "$checksum" >> "$MANIFEST"
done

unique_count=$(tail -n +2 "$MANIFEST" | cut -f3 | sort -u | wc -l)
seed_count=$(wc -w <<< "$SEEDS")
if [[ "$unique_count" -ne "$seed_count" ]]; then
  echo "Seeded FASTAs are not all distinct; inspect $MANIFEST and FDRBench logs." >&2
  exit 1
fi

echo "Wrote $seed_count distinct entrapment replicates and $MANIFEST"
