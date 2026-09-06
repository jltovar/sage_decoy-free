# Exact external AUC and bounded within-evaluation parallelism

All six external calibration profiles and CLI reference/entrapment diagnostics
use `ml::external_auc::exact_external_auc`.
Their scientific values and diagnostics are preserved. The statistic is unchanged: count strict
wins plus half of exact ties, divided by all finite good/null pairs. Lower-is-
better reverses strict wins, not the definition of ties. NaN and infinities are
excluded, and an empty finite population returns NaN. Signed zeros tie; close
distinct values do not. There is no epsilon, sampling, cap, or imputation.

## Counting and numerical representation

Filtered numeric vectors are sorted, then equal-value runs are counted with a
linear merge. Sorting uses total ordering, but tie groups use numerical `==`,
so -0 and +0 remain one tie group. Counts and products use checked u128; zero
denominators, inconsistent counts, or overflow return unavailable (NaN), not a
fabricated AUC. Complexity is O(G log G + N log N), with O(G + N) temporary f64
storage, instead of O(G*N) comparisons.

For at most 2^52 finite pairs the legacy half-integer numerator and integer
denominator are exactly representable in f64. Conversion and the final division
are bit-identical to the legacy loop. Above that range, integer counts remain
exact and are rounded to f64 only once: the implementation does not reproduce
the legacy loop's loss of half increments or eventual floating-counter
saturation. Tests exercise this representability boundary using counts without
allocating enormous populations. The historical pairwise function remains
test/benchmark-only, with an enforced ten-million-pair bound; never benchmark it
on production-size populations.

## Runtime scope and isolation

Only sorting owned numeric vectors runs on Rayon workers. The two sorts share
the existing pool through `rayon::join`; no private pool is created by production
code. Candidate rows are never copied per worker or reordered. Profiles and
their diagnostic messages remain in fixed order. Floating reductions, bounded
updates, q-value calculations, and adaptive transitions are not parallelized by
this change. Selection/audit labels, explicit FdrSettings, and thread-local
window observations stay on their existing calling thread; sorting receives no
labels or observation/provenance state. Tests compare every serialized PSM,
all profiles, q/fit observations and Level4 output across worker counts.

Set `RAYON_NUM_THREADS` before launching the executable; Sage logs its actual
worker count. The setting is runtime evidence, not a scientific parameter. No
universal machine-specific worker count is imposed. Benchmark progressively
from one worker and stop at a plateau; more logical CPUs need not improve small
sorts or memory-bandwidth-limited operations. Do not introduce nested OpenMP or
BLAS pools. Linux process 100% CPU generally means one busy logical CPU, unlike
whole-machine percentages. Cgroup quota, cpuset, affinity, WSL limits and memory
availability must be checked before choosing a runtime budget.

## Reproducible bounded synthetic benchmark

```
cargo build --release -p sage-core --example external_auc_benchmark --features bench
/usr/bin/time -v target/release/examples/external_auc_benchmark 1 600000 2100000 3
/usr/bin/time -v target/release/examples/external_auc_benchmark 2 600000 2100000 3
```

Arguments are workers, finite good count, finite null count, repetitions. Each
repetition computes six AUCs with both directions. The report separates bounded
old/new algorithm timings from full-population serial/parallel sorting timings,
asserts exact results, and reports actual Rayon workers. Capture GNU time CPU,
elapsed and peak RSS separately. Data are deterministic synthetic values, not
loaded spectra, predictions, candidates, or fitted profiles.

This does not redesign sequential optimization. Whole-trial concurrency needs a
separate dependency, checkpoint, observation-isolation and memory-budget design;
candidate decoding, hashing, serial fitting, and filesystem waits remain possible
costs. Do not infer a filesystem bottleneck solely from a pathname or relocate
authoritative resources without measurements and authorization.

Implementation/proposal identities change normally and include the new AUC
source. Search and candidate-pool identities are unchanged. New raw finalizers
use a versioned, scoped production-source identity; old raw caches retain their
exact historical fingerprint and finalizer provenance, never relabeled as newly
generated. No scientific settings, activation rules, thresholds, calibration
windows, Level4 flags or Ensemble participation rules are changed.

## Raw production, diagnostics, and historical verification

`external_features.rs` retains export, neutral generator configuration, process
invocation, TSV parsing, whole-lane missingness interpretation and candidate-ID
joining. `external_feature_cache.rs` retains generator identity resolution,
record validation, serialization, hashing and atomic publication. Their entire
sources remain in raw-finalizer identity (including helpers after test modules).
The v2 source list additionally binds candidate IDs, input normalization, path
hashing, atomic/hash utilities, core feature/peptide/database representations,
their digestion/mass/modification dependencies, cloud-path handling, package
manifests and the dependency lock. This conservative dependency coverage means
even some unrelated changes in those shared files require compatibility review.

`external_feature_diagnostics.rs` contains only post-join diagnostics and calls
the same exact AUC routine as calibration. It and the AUC/calibration sources
are bound to downstream analysis v2, stage v7, and optimizer implementation
provenance, not raw prediction production. No observation context or candidate
row moves into sorting workers.

Historical reuse is explicit and read-only, for example in a workflow:

```json
{
  "require_existing_annotation_cache": true,
  "existing_raw_cache": {
    "compatibility_contract": "raw-v3-whole-lane-v1-to-scoped-v2",
    "fingerprint": "<frozen 64-character lowercase SHA-256>",
    "manifest_sha256": "<frozen manifest SHA-256>",
    "payload_sha256": "<frozen compressed payload SHA-256>"
  }
}
```

Replace the explanatory hash strings with frozen hashes before execution; they
are deliberately invalid placeholders. Target-only reuse has a distinct
`target_only_existing_raw_cache` reference and never inherits the combined
reference. The cache remains at its original fingerprint under the configured
cache root. No directory discovery, copying, migration or generation fallback
is performed. The reference requires strict existing mode, without migration.

The named contract supports the reviewed v3 payload / v2 whole-empty-lane parser
and historical v1 finalizer source recorded in `raw_cache_compatibility.rs`.
It also pins the reviewed current producer source: an unreviewed parser or
dependency change fails closed, even if a developer forgets to bump a schema.
This is a source-contract allowlist, not a dataset or artifact allowlist. Unknown
contracts, sources, schemas and fields in the reference are rejected.

Both preflight and runtime verify historical self hashes, frozen artifact hashes,
current generator settings, wrapper/interpreter/package/model content identities,
exact candidate population, compressed payload, content/missingness identity,
and every durable record. Whole unavailable lanes remain explicitly unavailable;
no row is removed or prediction imputed. Component mismatches are reported by
field. Operational source locations do not replace content verification.

The verified-use v1 record binds historical artifact/finalizer/generator identities
and the current producer, reader, binary and successful content verification.
Historical artifacts are never rewritten. Reader source is separately versioned
and hashed, and its record is included in stage evidence. Future compatibility
changes require an explicit review of raw interpretation, not replacement of an
old artifact's source hash. Exact existing-cache reuse is not regeneration.
