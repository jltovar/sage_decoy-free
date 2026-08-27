# Input-path identity for spectral evidence

Sage uses one canonical implementation, `input_path_identity`, whenever a local spectral source
must contribute scientific identity. It distinguishes regular files, directory-backed datasets,
and unsupported path types. It never substitutes an absolute path, directory timestamp, aggregate
size, or other metadata-only value for content identity.

## Root cause and production call graph

Before directory support, the candidate-pool path resolved a local spectrum URL through
`candidate_pool::source_sha256`, then `candidate_pool::content_sha256`, and finally
`provenance::sha256_file`. Applying that regular-file reader to a native vendor directory failed
at `File::open` with `Is a directory (os error 21)`. Separately,
`workflow::compute_dataset_identity` content-hashed only paths for which `Path::is_file` was true;
it treated a directory as an unresolved source string. The two identities therefore disagreed.
The raw annotation provenance code also had a separate recursive directory implementation with
different framing and symlink behavior.

The corrected paths are:

- Direct candidate/search identity: `candidate_pool::search_fingerprint` resolves every spectrum
  through `input_path_identity` before candidate search.
- Candidate-pool-only: `Runner::construct_candidate_pool_only` calls
  `candidate_pool_identity_preflight`, which resolves the strict search and analysis identities
  before `batch_files` can be called.
- Candidate pool reuse and raw-cache-only preflight: `Runner::preflight_existing_candidate_pool`
  recomputes the same strict identity before loading a pool. Raw-cache identity then binds that
  verified search fingerprint and uses the same helper for local source provenance.
- Workflow dataset identity and preflight: `workflow::compute_dataset_identity` records the same
  detailed spectrum identities; shared database creation separately keys on
  `candidate_pool::search_fingerprint`.
- Target-only reconstruction: the reconstructed search enters the same candidate-pool and runner
  fingerprint paths, so a changed target database remains distinct while spectrum identity stays
  canonical.
- Checkpoint, resume, fitted-artifact, and comparison/report provenance: these bind the workflow
  dataset fingerprint and/or strict candidate search fingerprint produced above. Candidate stable
  IDs bind the strict fingerprint through `stable_candidate_id`.

Ordinary direct Sage TDC execution does not publish or consume an immutable candidate-pool strict
fingerprint. Its spectra are read by the normal search path. Whenever that search participates in
a Decoy-Free pool, workflow identity, cache, resume, or comparison provenance, the relevant
persisted identity is supplied by the canonical paths above.

## Regular files

Regular-file sources retain the historical complete-file SHA-256. The strict-search digest still
binds the same hexadecimal file digest in the same order and framing, so existing mzML, MGF, RAW,
and other ordinary-file fingerprints and stable candidate IDs do not change. Existing compatible
regular-file pools remain reopenable.

## Directory schema

Directory-backed sources use `sage-input-directory-content-v1`, domain-separated by
`sage-input-path-identity`, `directory-content`, and version `v1`. The digest length-frames:

1. the domain and schema;
2. total regular-file count and total bytes;
3. for each deterministically sorted entry, the `regular_file` type, normalized root-relative
   path, file size, and raw 32-byte complete-file SHA-256.

The absolute root, root directory name, username, mount point, permissions, ownership, and
timestamps do not enter the digest. Traversal order also does not enter it. Relocating an unchanged
directory therefore preserves identity, while adding, removing, renaming, truncating, or changing
any included file changes identity.

Before content hashing, Sage freezes the sorted root-relative inventory and file stability
metadata. Each path and open file descriptor must still name a regular file with the same size and
stability metadata before and after content hashing. Sage then rescans the complete directory and
requires the final inventory to equal the frozen inventory. Symlinks are never followed inside a
directory; symlinked directory roots, sockets, devices, FIFOs, other special entries, invalid
relative names, duplicate normalized paths, unreadable files, empty directories, and any observed
concurrent mutation are errors.

Candidate-pool manifests and workflow provenance record the input kind, directory schema, entry
count, total bytes, and content digest. Existing artifacts without a matching directory-kind
identity fail closed. This changes identity and resume semantics only for directory-backed
spectra; it does not change parsing, scoring, ranking, retained depth, annotations, FDR,
optimization, or Ensemble behavior.
