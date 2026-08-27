# Controlled resource-construction boundaries

Sage exposes two narrow Decoy-Free execution scopes for prospective validation work. They are
resource constructors, not shortened forms of `sage workflow`: each has a dedicated call graph,
returns immediately after reopening and verifying its resource, and emits an atomic JSON boundary
report. Neither command runs target-only processing, audit evaluation, or target-decoy competition.

## Traced persistence boundaries

The general `Runner::run_with_workflow_caches` path constructs a database and scorer, searches the
configured spectra with `batch_files`, excludes explicit decoy-labelled PSMs in Decoy-Free mode,
and, when enabled, performs native RT alignment plus RT and mobility prediction. It then persists
the candidate pool. After that persistence point the general path converts candidates to
`DfFeature`, fits Decoy-Free models or null-window trials, invokes external annotation, performs a
second Decoy-Free pass, calculates and aggregates q-values, writes fitted artifacts and results,
and may return to workflow-level optimization, audit, or target-only orchestration.

The general external-feature path first derives stable candidate IDs and a raw identity, resolves
the wrapper, Python/package environment, and selected model-file identities, exports a neutral
candidate table, invokes the configured feature-only wrapper, parses and joins its output, and
persists the model-independent raw prediction cache. After raw persistence, the general workflow
derives a model/window-specific calibration identity and external empirical profile, refits the
active Decoy-Free model, and continues to aggregation and workflow orchestration.

The dedicated scopes cut those call graphs at the persistence points:

- `Runner::construct_candidate_pool_only` contains search and candidate publication but has no
  call to Decoy-Free fitting, q-value, optimizer, or external-feature code.
- `Runner::construct_raw_annotation_cache_only` requires
  `preflight_existing_candidate_pool`, converts that verified payload to annotation inputs, and
  calls `external_features::construct_raw_cache_only`. That function has no stage-calibration or
  statistical-fit request and returns immediately after complete cache reopen verification.

## Candidate-pool-only command

```bash
sage candidate-pool-only frozen-search.json \
  --candidate-pool-root /external/resources/candidate-pools \
  --rank-depth 50 \
  --report /external/evidence/candidate-pool-only.report.json
```

The parameter JSON must use `fdr.mode: "decoy_free"`, must disable generated decoys, and must not
request LFQ, TMT, or matched-fragment output. `--rank-depth` must exactly equal the retained depth
in the frozen search identity. The command builds the configured database and performs only the
native spectral work needed for the immutable pre-FDR pool, including native RT/IMS prediction
when that frozen search setting is enabled. It then atomically publishes and fully reopens the
manifest and compressed payload before writing its report.

Identity preflight occurs before `batch_files` enters native spectrum search. Every local spectrum
source must resolve through the canonical input-path identity implementation. Ordinary files keep
their legacy full-file SHA-256 behavior. Directory-backed sources require the versioned recursive
content identity described in [`INPUT_PATH_IDENTITY.md`](INPUT_PATH_IDENTITY.md); no path-only,
directory-metadata-only, or legacy directory identity is accepted. A missing, empty, unreadable,
mutating, symlink-containing, or special-entry directory therefore fails before search.

If the content-addressed final directory already exists, Sage fully verifies and reuses it without
searching. An incomplete, corrupt, or identity-incompatible final directory is an error; it is not
overwritten and does not trigger a fallback search.

The successful report records the search and analysis fingerprints, pool paths and SHA-256 values,
candidate and spectrum counts, retained/observed ranks, whether exact reuse occurred, whether the
native search and native RT/IMS prediction ran, an empty `downstream_stages_entered` list, and the
enforced stopping guarantees.

## Raw-cache-only command

```bash
sage raw-cache-only frozen-search.json \
  --candidate-pool-root /external/resources/candidate-pools \
  --annotation-cache-root /external/resources/raw-annotations \
  --rank-depth 50 \
  --report /external/evidence/raw-cache-only.report.json
```

The parameter JSON must enable feature-only external generation with `fail_policy: "error"`.
`--rank-depth` must exactly equal `external_features.max_rank`. Before resolving or invoking the
generator, Sage requires the exact content-addressed candidate pool and verifies its schema,
search identity, FASTA/spectrum/configuration identity, rank capability, stable candidate IDs,
population, manifest, and payload. There is no native-search fallback.
`database.prefilter` is prohibited for this command because rebuilding a prefiltered database index
can itself launch a native spectrum search before pool loading.

The raw constructor requires durable wrapper and Python identities as applicable. For configured
MS2PIP, DeepLC, or IM2Deep generators it verifies the relevant installed package versions plus the
wrapper package and `psm-utils`; for MS2PIP and DeepLC it also requires complete selected model-file
content identities. Generation exports the frozen candidate population with neutral q/PEP fields,
invokes the wrapper once, requires a one-to-one stable-ID join with finite raw features, and
atomically publishes and reopens the raw manifest and payload. Duplicate, missing, surplus, or
nonfinite records fail before publication.

If the exact raw cache already exists, Sage fully verifies and reuses it without invoking Python or
the wrapper. An incomplete, corrupt, or incompatible final cache fails closed and never triggers
regeneration. The successful catalog/report records both candidate-resource hashes; raw manifest,
payload, and candidate-ID coverage hashes; model, wrapper, Python, package, and probe provenance;
population counts; execution flags; an empty `downstream_stages_entered` list; and the stopping
guarantees. It performs no model/window-specific calibration.

## Atomicity and resumption

New resources are written into a unique sibling `.partial.<pid>.<nonce>` directory, fully reopened
and validated there, then published with one directory rename. The parent directory is synced on
Unix. A partial directory is never accepted as a final resource. Failed staging is removed; a
completed final directory is immutable. Concurrent appearance of the final directory is treated as
ambiguous and fails rather than overwriting either producer. Boundary reports use the existing
atomic JSON writer and are reopened and identity-checked by the CLI before success is printed.

Safe resumption therefore means rerunning the same boundary with byte-identical scientific inputs:
a complete exact resource is verified and reused, while any mismatch stops. There is no `--force`
or repair path in either command.

Directory-backed spectra add a second resumption condition: the stored input kind, directory
schema, root-relative entry count, total bytes, and content digest must match the newly resolved
identity. Manifests lacking this directory identity cannot reopen a directory-backed pool.

## Scientific and operational audit scope

Candidate construction changes no reporting thresholds and performs no statistical selection.
Raw annotation construction consumes the exact pre-FDR population and excludes preliminary q/PEP
values from its export and identity, so it cannot select model windows or winners. The reports make
the stopping claim machine-inspectable, but downstream workflow authorization remains a separate
operator gate.
