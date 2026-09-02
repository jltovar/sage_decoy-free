# Phase-scoped entrapment resource identity

Native entrapment construction parses the target and foreign FASTAs, digests their records with
Sage, excludes foreign proteins contributing a target-shared canonical peptide, deterministically
samples eligible proteins, writes the combined FASTA, and measures protein, canonical-peptide, and
peptidoform ratios. Selection/audit connected components are constructed later and are not an
entrapment-generation input.

## Consumption-based field audit

The portable `EntrapmentGenerationScientificInputsV1` projection classifies inputs as follows.

| Class | Inputs | Reason |
|---|---|---|
| A — scientific construction | Target and all foreign FASTA content hashes; selected-source content hash where declared; resolved enzyme cleavage residues, restriction, terminus, semi-enzymatic behavior, missed cleavages, peptide length limits; peptide mass limits; fixed and variable modification mass bits; maximum variable modifications; source mode; shared-peptide exclusion mode; seed; protein fold | These values can alter eligibility, sampling, generated content, canonical peptide/peptidoform populations, or measured ratios. |
| B — algorithm/provenance | Scientific-input schema, generator version, selection algorithm, canonical I/L peptide semantics, peptidoform mass-bit semantics, deterministic header and sampling semantics | These define how class-A inputs are interpreted. |
| C — generated output | Combined FASTA content hash, selected/excluded accessions and mappings, construction identity, measured counts and ratios, original audit hash | These verify the realized immutable resource; they are not substituted for generation inputs. |
| D — irrelevant search/optimization | Precursor/fragment tolerances, isotope errors, charge and ion-mobility handling, score type, FDR thresholds and models, optimizer grids/objectives, fragment-ion kinds and minimum ion index, database bucket/prefilter controls, decoy prefix and decoy-generation flag | Construction does not consume these as evidence. Although Sage's digest implementation can emit decoys, construction immediately excludes decoy peptides; changing decoy controls cannot change the retained target peptide keys. |
| E — operational/reporting | Active database path, historical/active output paths, Python executable, external-feature wrapper/models/cache/temp paths, spectrum paths, report paths, thread count | These control another phase or runtime placement and cannot change construction. They may be retained as nonportable audit evidence. |

FASTA roles are separate: construction binds the target-only FASTA as an input; the resource binds
the generated target-plus-entrapment FASTA as output; optimization must use that exact output as its
active database. Absolute paths never define these portable identities. Protein-header content is
already bound by complete FASTA content hashing. Entrapment prefix/header formation is fixed by the
versioned implementation rather than a configurable field.

## Historical locks and compatibility

`lock-existing-entrapment-resource` verifies an immutable Sage audit manifest and report, their
historical search configuration, every referenced FASTA, the legacy full-parameter digest, selected
source, counts, measured ratios, and generated output. It writes and reopens an atomic schema-v1
lock. The legacy digest remains historical provenance only.

Existing-resource preflight recomputes the phase-scoped projection from the active workflow's
declared generation inputs, verifies the lock payload and generated combined FASTA, then separately
requires the active optimization database to hash to that combined FASTA. Partition validation
continues to require the exact construction identity. Candidate-pool, raw-cache, proposal-space,
checkpoint, trial, winner, and target-policy identities retain their existing stricter scopes.

The only complete `Parameters` digest retained in the entrapment module is the legacy schema-v2
generation digest. It is used to authenticate historical evidence while building a lock and to
resume an old workflow-local generation within the same phase. It is never used as the
cross-phase existing-resource compatibility predicate. The partition digestion identity is already
a path-free digestion/modification projection. Other search, candidate, cache, and optimizer hashes
remain intentionally scoped to their own phases.
