#!/usr/bin/env python3
"""Measure effective entrapment search-space ratios from a Sage FASTA/config.

The peptide ratio counts unique fully enzymatic peptide sequences that survive
the Sage length and mass filters.  The PSM ratio counts the corresponding
variable-modification peptidoforms.  This reproduces the effective ratios used
by the FDRBench combined estimator for the current PXD001468 configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


WATER_MASS = 18.010564684
RESIDUE_MASS = {
    "A": 71.037113805,
    "R": 156.101111050,
    "N": 114.042927470,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.058577540,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084063975,
    "L": 113.084063975,
    "K": 128.094963015,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.079312980,
    "Y": 163.063328575,
    "V": 99.068413945,
}


@dataclass(frozen=True)
class RatioResult:
    target_proteins: int
    entrapment_proteins: int
    target_peptides: int
    entrapment_peptides: int
    target_peptidoforms: int
    entrapment_peptidoforms: int
    shared_peptides: int
    psm_r: float
    peptide_r: float
    protein_r: float


def read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    sequence: list[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence).upper()
                header = line[1:]
                sequence = []
            elif header is None:
                raise ValueError(f"{path}: sequence occurs before first FASTA header")
            else:
                sequence.append(line)
    if header is not None:
        yield header, "".join(sequence).upper()


def cleavage_sites(
    sequence: str, cleave_at: str, restrict: str | None, c_terminal: bool
) -> list[int]:
    restricted = set(restrict or "")
    sites = [0]
    if c_terminal:
        for index, residue in enumerate(sequence):
            next_residue = sequence[index + 1] if index + 1 < len(sequence) else None
            if residue in cleave_at and next_residue not in restricted:
                sites.append(index + 1)
    else:
        for index, residue in enumerate(sequence):
            previous = sequence[index - 1] if index > 0 else None
            if residue in cleave_at and previous not in restricted:
                sites.append(index)
    if sites[-1] != len(sequence):
        sites.append(len(sequence))
    return sorted(set(sites))


def digest(
    sequence: str,
    *,
    cleave_at: str,
    restrict: str | None,
    c_terminal: bool,
    missed_cleavages: int,
    min_len: int,
    max_len: int,
    min_mass: float,
    max_mass: float,
    static_mods: dict[str, float],
) -> Iterable[str]:
    sites = cleavage_sites(sequence, cleave_at, restrict, c_terminal)
    residue_masses = {
        residue: mass + float(static_mods.get(residue, 0.0))
        for residue, mass in RESIDUE_MASS.items()
    }
    for start_index in range(len(sites) - 1):
        for missed in range(missed_cleavages + 1):
            end_index = start_index + missed + 1
            if end_index >= len(sites):
                break
            peptide = sequence[sites[start_index] : sites[end_index]]
            if not min_len <= len(peptide) <= max_len:
                continue
            if any(residue not in residue_masses for residue in peptide):
                continue
            mass = WATER_MASS + sum(residue_masses[residue] for residue in peptide)
            if min_mass <= mass <= max_mass:
                yield peptide


def peptidoform_count(
    peptides: Iterable[str], variable_mods: dict[str, Sequence[float]], max_mods: int
) -> int:
    total = 0
    for peptide in peptides:
        # coefficient[k] is the number of assignments with exactly k modified sites.
        coefficients = [1] + [0] * max_mods
        for residue in peptide:
            alternatives = len(variable_mods.get(residue, ()))
            if alternatives == 0:
                continue
            for modified_sites in range(max_mods, 0, -1):
                coefficients[modified_sites] += (
                    coefficients[modified_sites - 1] * alternatives
                )
        total += sum(coefficients)
    return total


def measure(config: dict, fasta: Path, entrapment_marker: str) -> RatioResult:
    database = config["database"]
    enzyme = database["enzyme"]
    if enzyme.get("semi_enzymatic", False):
        raise ValueError("semi-enzymatic digestion is not supported by this calculator")

    target_peptides: set[str] = set()
    entrapment_peptides: set[str] = set()
    target_proteins = 0
    entrapment_proteins = 0
    digest_options = {
        "cleave_at": enzyme["cleave_at"],
        "restrict": enzyme.get("restrict"),
        "c_terminal": enzyme.get("c_terminal", True),
        "missed_cleavages": int(enzyme.get("missed_cleavages", 0)),
        "min_len": int(enzyme.get("min_len", 5)),
        "max_len": int(enzyme.get("max_len", 50)),
        "min_mass": float(database.get("peptide_min_mass", 500.0)),
        "max_mass": float(database.get("peptide_max_mass", 5000.0)),
        "static_mods": {
            key: float(value) for key, value in database.get("static_mods", {}).items()
        },
    }

    for header, sequence in read_fasta(fasta):
        if entrapment_marker in header:
            entrapment_proteins += 1
            entrapment_peptides.update(digest(sequence, **digest_options))
        else:
            target_proteins += 1
            target_peptides.update(digest(sequence, **digest_options))

    if target_proteins == 0 or entrapment_proteins == 0:
        raise ValueError(
            "FASTA must contain both target and entrapment proteins; "
            f"marker={entrapment_marker!r}"
        )
    shared = target_peptides.intersection(entrapment_peptides)
    if shared:
        raise ValueError(
            f"FASTA has {len(shared)} peptide sequences shared between target and "
            "entrapment; rebuild with FDRBench no-shared filtering"
        )

    variable_mods = {
        key: tuple(float(value) for value in values)
        for key, values in database.get("variable_mods", {}).items()
    }
    max_mods = int(database.get("max_variable_mods", 0))
    target_peptidoforms = peptidoform_count(target_peptides, variable_mods, max_mods)
    entrapment_peptidoforms = peptidoform_count(
        entrapment_peptides, variable_mods, max_mods
    )

    result = RatioResult(
        target_proteins=target_proteins,
        entrapment_proteins=entrapment_proteins,
        target_peptides=len(target_peptides),
        entrapment_peptides=len(entrapment_peptides),
        target_peptidoforms=target_peptidoforms,
        entrapment_peptidoforms=entrapment_peptidoforms,
        shared_peptides=len(shared),
        psm_r=entrapment_peptidoforms / target_peptidoforms,
        peptide_r=len(entrapment_peptides) / len(target_peptides),
        protein_r=entrapment_proteins / target_proteins,
    )
    for value in (result.psm_r, result.peptide_r, result.protein_r):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("effective ratios must be finite and positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Sage JSON config")
    parser.add_argument(
        "--fasta",
        type=Path,
        help="target-plus-entrapment FASTA; defaults to database.fasta in the config",
    )
    parser.add_argument("--entrapment-marker", default="Ent_")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with args.config.open() as handle:
            config = json.load(handle)
        fasta = args.fasta or Path(config["database"]["fasta"])
        result = measure(config, fasta, args.entrapment_marker)
        if args.format == "json":
            print(json.dumps(asdict(result), indent=2))
        else:
            print(f"psm_r={result.psm_r:.12g}")
            print(f"peptide_r={result.peptide_r:.12g}")
            print(f"protein_r={result.protein_r:.12g}")
            print(
                "counts: "
                f"proteins={result.entrapment_proteins}/{result.target_proteins}, "
                f"peptides={result.entrapment_peptides}/{result.target_peptides}, "
                f"peptidoforms={result.entrapment_peptidoforms}/"
                f"{result.target_peptidoforms}"
            )
        return 0
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
