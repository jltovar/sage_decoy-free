#!/usr/bin/env python3
"""Compare a native in-memory null-window grid with a frozen legacy CSV.

Legacy FDRBench searches include several rank-1 setup rows that are not valid
null windows.  Phase 2 established the parity convention of comparing the
visited rows whose minimum rank is greater than one; this tool preserves that
convention and reports every matched point, not only the selected window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


COUNT_FIELDS = (
    "target_psms",
    "entrapment_psms",
    "target_peptides",
    "entrapment_peptides",
    "target_proteins",
    "entrapment_proteins",
)
FDP_FIELDS = ("psm_fdp", "peptide_fdp", "protein_fdp")


def combined_fdp(entrapment: int, target: int, ratio: float) -> float:
    total = entrapment + target
    if total == 0:
        return 1.0
    return entrapment * (1.0 + 1.0 / ratio) / total


def legacy_metrics(row: dict[str, str], scope: str, ratios: dict[str, float]) -> dict[str, Any]:
    prefix = "level4_" if scope == "level4" else ""
    counts = {
        "target_psms": int(row[f"{prefix}target_psm"]),
        "entrapment_psms": int(row[f"{prefix}ent_psm"]),
        "target_peptides": int(row[f"{prefix}target_peptide"]),
        "entrapment_peptides": int(row[f"{prefix}ent_peptide"]),
        # Level 4 does not change the protein definition.
        "target_proteins": int(row["target_protein"]),
        "entrapment_proteins": int(row["ent_protein"]),
    }
    counts["psm_fdp"] = combined_fdp(
        counts["entrapment_psms"], counts["target_psms"], ratios["psm"]
    )
    counts["peptide_fdp"] = combined_fdp(
        counts["entrapment_peptides"], counts["target_peptides"], ratios["peptide"]
    )
    counts["protein_fdp"] = combined_fdp(
        counts["entrapment_proteins"], counts["target_proteins"], ratios["protein"]
    )
    return counts


def parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "t", "1", "yes", "y"}


def parse_float(value: str | None) -> float | None:
    try:
        parsed = float(value or "")
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def canonical_peptide(peptide: str) -> str:
    output: list[str] = []
    bracket_depth = 0
    for character in peptide:
        if character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif bracket_depth == 0 and character.isascii() and character.isalpha():
            upper = character.upper()
            output.append("L" if upper == "I" else upper)
    return "".join(output)


def protein_class(proteins: str) -> str | None:
    if "Cont_" in proteins:
        return None
    target = False
    entrapment = False
    for protein in (item.strip() for item in proteins.split(";")):
        if not protein:
            continue
        if "Ent_" in protein:
            entrapment = True
        else:
            target = True
    if target == entrapment:
        return None
    return "entrapment" if entrapment else "target"


def summarize_legacy_results(
    results: Path,
    scope: str,
    ratios: dict[str, float],
    fdr_threshold: float,
) -> dict[str, Any]:
    sets = {
        field: set()
        for field in (
            "target_psms",
            "entrapment_psms",
            "target_peptides",
            "entrapment_peptides",
            "target_proteins",
            "entrapment_proteins",
        )
    }
    with results.open(newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            if row.get("rank") != "1" or row.get("label") != "1":
                continue
            classification = protein_class(row.get("proteins", ""))
            peptide = canonical_peptide(row.get("peptide", ""))
            if classification is None or not peptide:
                continue
            psm_q = parse_float(row.get("decoy_free_q_value"))
            peptide_q = parse_float(row.get("decoy_free_peptide_q"))
            protein_q = parse_float(row.get("decoy_free_protein_q"))
            psm_ok = psm_q is not None and psm_q <= fdr_threshold
            peptide_ok = peptide_q is not None and peptide_q <= fdr_threshold
            if scope == "level4":
                psm_ok &= parse_bool(row.get("decoy_free_peptide_supported_psm"))
                peptide_ok &= parse_bool(
                    row.get("decoy_free_protein_supported_peptide")
                )
            prefix = f"{classification}_"
            if psm_ok:
                psm = row.get("psm_id") or (
                    f"{row.get('filename', 'unknown')}:"
                    f"{row.get('scannr', 'unknown')}:{row_index}"
                )
                sets[f"{prefix}psms"].add(psm)
            if peptide_ok:
                sets[f"{prefix}peptides"].add(peptide)
            if protein_q is not None and protein_q <= fdr_threshold:
                proteins = row.get("proteins", "")
                if len(proteins.split(";")) == 1 and proteins.strip():
                    sets[f"{prefix}proteins"].add(proteins.strip())
    counts = {field: len(values) for field, values in sets.items()}
    counts["psm_fdp"] = combined_fdp(
        counts["entrapment_psms"], counts["target_psms"], ratios["psm"]
    )
    counts["peptide_fdp"] = combined_fdp(
        counts["entrapment_peptides"], counts["target_peptides"], ratios["peptide"]
    )
    counts["protein_fdp"] = combined_fdp(
        counts["entrapment_proteins"], counts["target_proteins"], ratios["protein"]
    )
    return counts


def fraction_difference(baseline: int, native: int) -> float | None:
    return None if baseline == 0 else (native - baseline) / baseline


def numeric_difference(baseline: Any, native: Any) -> float | None:
    """Return a finite numeric difference, or None for undefined metrics."""
    try:
        baseline_value = float(baseline)
        native_value = float(native)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(baseline_value) or not math.isfinite(native_value):
        return None
    return native_value - baseline_value


def read_legacy_rows(legacy_csv: Path) -> list[dict[str, str]]:
    with legacy_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"legacy CSV has no header: {legacy_csv}")
        if "min_null_rank" in reader.fieldnames:
            return list(reader)
        if reader.fieldnames[0] != "metric":
            raise ValueError(
                "legacy CSV must be cumulative_results_long.csv or "
                "cumulative_results_wide.csv"
            )
        run_ids = reader.fieldnames[1:]
        transposed = {run_id: {"run_id": run_id} for run_id in run_ids}
        for metric_row in reader:
            metric = metric_row["metric"]
            for run_id in run_ids:
                transposed[run_id][metric] = metric_row[run_id]
        return [transposed[run_id] for run_id in run_ids]


def compare(
    legacy_csv: Path,
    native_json: Path,
    legacy_results_root: Path | None,
    ratios: dict[str, float],
    maximum_fdp: float,
    expected_window: tuple[int, int] | None,
    maximum_count_fraction_difference: float,
    maximum_fdp_difference: float,
) -> dict[str, Any]:
    legacy_rows = [
        row for row in read_legacy_rows(legacy_csv) if int(row["min_null_rank"]) > 1
    ]
    native_rows = json.loads(native_json.read_text())
    native_by_window = {
        (int(row["min_rank"]), int(row["max_rank"])): row for row in native_rows
    }
    legacy_windows = {
        (int(row["min_null_rank"]), int(row["max_null_rank"])) for row in legacy_rows
    }
    native_windows = set(native_by_window)
    scopes = {row.get("validation_scope", "level4") for row in native_rows}
    if len(scopes) != 1:
        raise ValueError(f"native evaluation table has mixed validation scopes: {scopes}")
    scope = scopes.pop()

    point_rows: list[dict[str, Any]] = []
    for legacy in legacy_rows:
        window = (int(legacy["min_null_rank"]), int(legacy["max_null_rank"]))
        native = native_by_window.get(window)
        if native is None:
            continue
        if legacy_results_root is None:
            baseline = legacy_metrics(legacy, scope, ratios)
        else:
            baseline = summarize_legacy_results(
                legacy_results_root / legacy["run_id"] / "results.sage.tsv",
                scope,
                ratios,
                maximum_fdp,
            )
        count_differences = {
            field: int(native[field]) - int(baseline[field]) for field in COUNT_FIELDS
        }
        count_fraction_differences = {
            field: fraction_difference(int(baseline[field]), int(native[field]))
            for field in COUNT_FIELDS
        }
        corrected_feasible = all(baseline[field] <= maximum_fdp for field in FDP_FIELDS)
        native_feasible = bool(native["feasible"])
        fdp_differences = {
            field: numeric_difference(baseline[field], native[field])
            for field in FDP_FIELDS
        }
        fdp_comparison_applicable = corrected_feasible or native_feasible
        fdp_values_available = all(
            difference is not None for difference in fdp_differences.values()
        )
        point_rows.append(
            {
                "run_id": legacy["run_id"],
                "min_rank": window[0],
                "max_rank": window[1],
                "legacy": baseline,
                "native": {field: native[field] for field in (*COUNT_FIELDS, *FDP_FIELDS)},
                "count_differences": count_differences,
                "count_fraction_differences": count_fraction_differences,
                "fdp_differences": fdp_differences,
                "legacy_recorded_feasible": legacy.get("feasible") == "1",
                "legacy_corrected_feasible": corrected_feasible,
                "native_feasible": native_feasible,
                "feasibility_matches_corrected_legacy": native_feasible
                == corrected_feasible,
                "fdp_comparison_applicable": fdp_comparison_applicable,
                "fdp_values_available": fdp_values_available,
                "fdp_comparison_satisfied": (
                    not fdp_comparison_applicable or fdp_values_available
                ),
                "native_selected": bool(native["selected"]),
            }
        )

    selected = [
        (int(row["min_rank"]), int(row["max_rank"]))
        for row in native_rows
        if row.get("selected")
    ]
    finite_fraction_differences = [
        abs(value)
        for row in point_rows
        for value in row["count_fraction_differences"].values()
        if value is not None and math.isfinite(value)
    ]
    finite_fdp_differences = [
        abs(value)
        for row in point_rows
        for value in row["fdp_differences"].values()
        if value is not None and math.isfinite(value)
    ]
    max_count_fraction = max(finite_fraction_differences, default=0.0)
    max_fdp_delta = max(finite_fdp_differences, default=0.0)
    selected_matches = expected_window is None or selected == [expected_window]
    grid_matches = legacy_windows == native_windows
    feasibility_matches = all(
        row["feasibility_matches_corrected_legacy"] for row in point_rows
    )
    fdp_availability_satisfied = all(
        row["fdp_comparison_satisfied"] for row in point_rows
    )
    return {
        "schema_version": 1,
        "legacy_csv": str(legacy_csv),
        "legacy_results_root": (
            str(legacy_results_root) if legacy_results_root is not None else None
        ),
        "native_evaluations": str(native_json),
        "validation_scope": scope,
        "counting_definition": (
            "rank=1,label=1,non-contaminant,unambiguous target/entrapment mapping; "
            "peptide removes bracketed modifications and canonicalizes I/L; protein "
            "requires one inferred protein key"
            if legacy_results_root is not None
            else "frozen legacy cumulative CSV counters"
        ),
        "effective_ratios": ratios,
        "maximum_fdp": maximum_fdp,
        "expected_selected_window": (
            {"min_rank": expected_window[0], "max_rank": expected_window[1]}
            if expected_window
            else None
        ),
        "native_selected_windows": [
            {"min_rank": window[0], "max_rank": window[1]} for window in selected
        ],
        "summary": {
            "legacy_visited_windows": len(legacy_windows),
            "native_evaluated_windows": len(native_windows),
            "matched_windows": len(point_rows),
            "missing_native_windows": [
                {"min_rank": window[0], "max_rank": window[1]}
                for window in sorted(legacy_windows - native_windows)
            ],
            "extra_native_windows": [
                {"min_rank": window[0], "max_rank": window[1]}
                for window in sorted(native_windows - legacy_windows)
            ],
            "exact_count_rows": sum(
                all(delta == 0 for delta in row["count_differences"].values())
                for row in point_rows
            ),
            "feasibility_match_rows": sum(
                row["feasibility_matches_corrected_legacy"] for row in point_rows
            ),
            "fdp_comparable_rows": sum(
                row["fdp_comparison_applicable"] for row in point_rows
            ),
            "fdp_available_rows": sum(
                row["fdp_comparison_applicable"] and row["fdp_values_available"]
                for row in point_rows
            ),
            "maximum_absolute_count_fraction_difference": max_count_fraction,
            "maximum_absolute_fdp_difference": max_fdp_delta,
            "grid_matches": grid_matches,
            "selected_window_matches": selected_matches,
            "corrected_feasibility_matches": feasibility_matches,
            "required_fdp_values_available": fdp_availability_satisfied,
            "within_count_tolerance": max_count_fraction
            <= maximum_count_fraction_difference,
            "within_fdp_tolerance": max_fdp_delta <= maximum_fdp_difference,
        },
        "passed": grid_matches
        and selected_matches
        and feasibility_matches
        and fdp_availability_satisfied
        and max_count_fraction <= maximum_count_fraction_difference
        and max_fdp_delta <= maximum_fdp_difference,
        "points": point_rows,
    }


def parse_window(value: str) -> tuple[int, int]:
    minimum, maximum = value.split("-", 1)
    return int(minimum), int(maximum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_csv", type=Path)
    parser.add_argument("native_json", type=Path)
    parser.add_argument(
        "--legacy-results-root",
        type=Path,
        help="legacy outdirs root containing RUN_ID/results.sage.tsv; enables shared canonical counting",
    )
    parser.add_argument("--psm-ratio", type=float, required=True)
    parser.add_argument("--peptide-ratio", type=float, required=True)
    parser.add_argument("--protein-ratio", type=float, required=True)
    parser.add_argument("--maximum-fdp", type=float, default=0.01)
    parser.add_argument("--expected-window", type=parse_window)
    parser.add_argument("--maximum-count-fraction-difference", type=float, default=0.0)
    parser.add_argument("--maximum-fdp-difference", type=float, default=1e-12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(
        args.legacy_csv,
        args.native_json,
        args.legacy_results_root,
        {"psm": args.psm_ratio, "peptide": args.peptide_ratio, "protein": args.protein_ratio},
        args.maximum_fdp,
        args.expected_window,
        args.maximum_count_fraction_difference,
        args.maximum_fdp_difference,
    )
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
