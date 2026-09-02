# Layered selection/audit partition provenance

Selection/audit partition artifacts are immutable historical computation records. Schema-v1
artifacts retain their original `source_implementation_identity` and `payload_sha256`; verification
never rewrites either value.

Verification separates four identities:

1. **Frozen scientific content** (`sage-entrapment-partition-scientific-content-v1`) binds the
   partition and assignment-algorithm schemas, the current portable content-derived dataset and
   entrapment-resource identities, target
   and combined FASTA hashes, digestion identity, seed, salt, requested and realized fractions,
   connected-component assignments, protein/canonical-peptide/peptidoform membership, component
   (protein-group) membership, measured ratios, and explicit zero-overlap results.
2. **Historical generator** preserves the source identity, dataset-identity algorithm and value,
   legacy partition identity, original artifact hash, original payload hash, generator schema, and
   assignment schema recorded by the artifact.
3. **Current verifier** records the current source identity, executable hash, and verification
   schema. It does not claim to have generated the historical bytes.
4. **Verified use** binds the exact historical artifact, its frozen scientific content, historical
   generator, and current verifier. Verification time and local path are audit metadata excluded
   from the portable identity.

Existing-partition verification authenticates the original payload using its stored historical
fields, reconstructs the partition deterministically from the declared FASTAs and digestion/search
space, and compares the complete scientific projection. For artifacts made before directory-backed
spectra gained a portable content identity, the verifier also reproduces the exact historical
directory-unaware path-string algorithm to authenticate the stored legacy dataset identity. It then
binds the verified assignments to the current directory-content identity. The path-derived legacy
dataset and partition identities remain historical metadata and never replace the portable
scientific identity. A current source change is compatible only when that complete projection is
byte-for-byte equivalent. Any changed input, component,
assignment, membership, ratio, overlap, or unsupported schema fails closed with component paths.

Optimizer and stage fingerprints bind both the scientific-content hash and exact artifact hash.
Thus checkpoint resume and winner materialization reject another partition even when a legacy
display identity happens to match. Audit labels remain absent from the optimizer-facing selection
view.

The legacy `partition_identity`, historical `source_implementation_identity`, and original payload
hash remain available for audit and self-authentication, but none is compared to a value synthesized
from the current repository source. Workflow preflight, runtime selection views, optimizer roots,
trial diagnostics, checkpoints, fitted stages, frozen audit records, and winner replay all carry the
layered scientific-content and exact-artifact hashes. No remaining partition compatibility predicate
uses an absolute artifact path, timestamp, workflow-local output path, or whole current-source
equality.

Absolute artifact locations and verification timestamps are operational evidence only. Moving
unchanged bytes does not change the artifact hash, scientific content, or verified-use identity.
