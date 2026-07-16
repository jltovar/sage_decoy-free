#!/usr/bin/env python3
"""Select null-rank windows using FDRBench entrapment bounds.

The optimizer's cumulative CSV records original-target and entrapment discovery
counts.  This tool converts those counts into the FDRBench lower and combined
estimates and selects the highest-yield window that satisfies every configured
level in every replicate.

The combined estimate is

    E * (1 + 1 / r) / (T + E)

where ``r`` is the effective entrapment-to-original-target search-space ratio.
For foreign-species entrapment, ``r`` should be measured separately at the PSM,
peptide, and protein levels.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_COLUMNS = {
    "run_id",
    "min_null_rank",
    "max_null_rank",
    "num_ranks_used",
    "target_psm",
    "target_peptide",
    "target_protein",
    "ent_psm",
    "ent_peptide",
    "ent_protein",
    "level4_target_peptide",
    "level4_target_psm",
    "level4_ent_peptide",
    "level4_ent_psm",
}


@dataclass(frozen=True)
class EffectiveRatios:
    psm: float
    peptide: float
    protein: float

    def validate(self) -> None:
        for name, value in (
            ("psm", self.psm),
            ("peptide", self.peptide),
            ("protein", self.protein),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"effective {name} r must be finite and > 0; got {value}")


@dataclass(frozen=True)
class Limits:
    psm: float
    peptide: float
    protein: float

    def validate(self) -> None:
        for name, value in (
            ("psm", self.psm),
            ("peptide", self.peptide),
            ("protein", self.protein),
        ):
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"maximum {name} FDP must be between 0 and 1; got {value}")


@dataclass(frozen=True)
class Estimate:
    lower: float
    combined: float


@dataclass(frozen=True)
class EvaluatedRow:
    source: Path
    row: dict[str, str]
    raw: dict[str, Estimate]
    level4: dict[str, Estimate]

    @property
    def window(self) -> tuple[int, int]:
        return (int(self.row["min_null_rank"]), int(self.row["max_null_rank"]))


def fdrbench_estimate(entrapment: int, target: int, effective_r: float) -> Estimate:
    """Return the lower and combined FDRBench point estimates."""

    if entrapment < 0 or target < 0:
        raise ValueError("discovery counts must be non-negative")
    if not math.isfinite(effective_r) or effective_r <= 0.0:
        raise ValueError("effective r must be finite and > 0")
    total = entrapment + target
    if total == 0:
        return Estimate(lower=1.0, combined=1.0)
    lower = entrapment / total
    combined = entrapment * (1.0 + 1.0 / effective_r) / total
    return Estimate(lower=lower, combined=combined)


def _counts(row: dict[str, str], scope: str) -> dict[str, tuple[int, int]]:
    if scope == "raw":
        return {
            "psm": (int(row["ent_psm"]), int(row["target_psm"])),
            "peptide": (int(row["ent_peptide"]), int(row["target_peptide"])),
            "protein": (int(row["ent_protein"]), int(row["target_protein"])),
        }
    if scope == "level4":
        return {
            "psm": (int(row["level4_ent_psm"]), int(row["level4_target_psm"])),
            "peptide": (
                int(row["level4_ent_peptide"]),
                int(row["level4_target_peptide"]),
            ),
            "protein": (int(row["ent_protein"]), int(row["target_protein"])),
        }
    raise ValueError(f"unknown scope: {scope}")


def evaluate_row(
    source: Path, row: dict[str, str], ratios: EffectiveRatios
) -> EvaluatedRow:
    ratio_by_level = {
        "psm": ratios.psm,
        "peptide": ratios.peptide,
        "protein": ratios.protein,
    }

    def evaluate_scope(scope: str) -> dict[str, Estimate]:
        return {
            level: fdrbench_estimate(ent, target, ratio_by_level[level])
            for level, (ent, target) in _counts(row, scope).items()
        }

    return EvaluatedRow(
        source=source,
        row=row,
        raw=evaluate_scope("raw"),
        level4=evaluate_scope("level4"),
    )


def read_results(path: Path, ratios: EffectiveRatios) -> list[EvaluatedRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = [evaluate_row(path, row, ratios) for row in reader]
    if not rows:
        raise ValueError(f"{path}: no result rows")
    return rows


def _scope_metrics(row: EvaluatedRow, scope: str) -> dict[str, Estimate]:
    return row.level4 if scope == "level4" else row.raw


def is_feasible(row: EvaluatedRow, scope: str, limits: Limits) -> bool:
    metrics = _scope_metrics(row, scope)
    return (
        metrics["psm"].combined <= limits.psm
        and metrics["peptide"].combined <= limits.peptide
        and metrics["protein"].combined <= limits.protein
    )


def group_common_windows(
    replicates: Sequence[Sequence[EvaluatedRow]],
) -> dict[tuple[int, int], list[EvaluatedRow]]:
    by_replicate = [{row.window: row for row in rows} for rows in replicates]
    common = set(by_replicate[0])
    for rows in by_replicate[1:]:
        common.intersection_update(rows)
    return {window: [rows[window] for rows in by_replicate] for window in sorted(common)}


def _mean(values: Iterable[int]) -> float:
    return statistics.fmean(values)


def selection_key(rows: Sequence[EvaluatedRow], scope: str) -> tuple[float, ...]:
    """Prefer reproducible yield, then smaller worst-case FDP and windows."""

    target_proteins = [int(row.row["target_protein"]) for row in rows]
    target_peptides = [int(row.row["level4_target_peptide"]) for row in rows]
    target_psms = [int(row.row["level4_target_psm"]) for row in rows]

    worst = {
        level: max(_scope_metrics(row, scope)[level].combined for row in rows)
        for level in ("protein", "peptide", "psm")
    }
    first = rows[0]
    return (
        -min(target_proteins),
        -_mean(target_proteins),
        -min(target_peptides),
        -_mean(target_peptides),
        -min(target_psms),
        -_mean(target_psms),
        worst["protein"],
        worst["peptide"],
        worst["psm"],
        int(first.row["num_ranks_used"]),
        first.window[0],
        first.window[1],
    )


def select_best(
    grouped: dict[tuple[int, int], list[EvaluatedRow]],
    scope: str,
    limits: Limits,
) -> tuple[tuple[int, int], list[EvaluatedRow]] | None:
    feasible = [
        (window, rows)
        for window, rows in grouped.items()
        if all(is_feasible(row, scope, limits) for row in rows)
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda item: selection_key(item[1], scope))


AUDIT_FIELDS = [
    "min_null_rank",
    "max_null_rank",
    "replicates",
    "feasible",
    "min_target_protein",
    "mean_target_protein",
    "min_level4_target_peptide",
    "mean_level4_target_peptide",
    "min_level4_target_psm",
    "mean_level4_target_psm",
    "worst_raw_psm_lower",
    "worst_raw_psm_combined",
    "worst_raw_peptide_lower",
    "worst_raw_peptide_combined",
    "worst_raw_protein_lower",
    "worst_raw_protein_combined",
    "worst_level4_psm_lower",
    "worst_level4_psm_combined",
    "worst_level4_peptide_lower",
    "worst_level4_peptide_combined",
    "worst_level4_protein_lower",
    "worst_level4_protein_combined",
    "run_ids",
    "sources",
]


def audit_record(
    window: tuple[int, int], rows: Sequence[EvaluatedRow], scope: str, limits: Limits
) -> dict[str, object]:
    result: dict[str, object] = {
        "min_null_rank": window[0],
        "max_null_rank": window[1],
        "replicates": len(rows),
        "feasible": int(all(is_feasible(row, scope, limits) for row in rows)),
        "min_target_protein": min(int(row.row["target_protein"]) for row in rows),
        "mean_target_protein": _mean(int(row.row["target_protein"]) for row in rows),
        "min_level4_target_peptide": min(
            int(row.row["level4_target_peptide"]) for row in rows
        ),
        "mean_level4_target_peptide": _mean(
            int(row.row["level4_target_peptide"]) for row in rows
        ),
        "min_level4_target_psm": min(int(row.row["level4_target_psm"]) for row in rows),
        "mean_level4_target_psm": _mean(
            int(row.row["level4_target_psm"]) for row in rows
        ),
        "run_ids": ";".join(row.row["run_id"] for row in rows),
        "sources": ";".join(str(row.source) for row in rows),
    }
    for metric_scope in ("raw", "level4"):
        for level in ("psm", "peptide", "protein"):
            estimates = [_scope_metrics(row, metric_scope)[level] for row in rows]
            result[f"worst_{metric_scope}_{level}_lower"] = max(
                value.lower for value in estimates
            )
            result[f"worst_{metric_scope}_{level}_combined"] = max(
                value.combined for value in estimates
            )
    return result


def write_audit(
    path: Path,
    grouped: dict[tuple[int, int], list[EvaluatedRow]],
    scope: str,
    limits: Limits,
) -> None:
    records = [audit_record(window, rows, scope, limits) for window, rows in grouped.items()]
    records.sort(
        key=lambda row: (
            -int(row["feasible"]),
            -int(row["min_target_protein"]),
            -float(row["mean_target_protein"]),
            int(row["min_null_rank"]),
            int(row["max_null_rank"]),
        )
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.6f}%"


def format_best_summary(
    window: tuple[int, int],
    rows: Sequence[EvaluatedRow],
    scope: str,
    ratios: EffectiveRatios,
    limits: Limits,
) -> str:
    lines = [
        "Best run selected by FDRBench combined-bound ranking",
        (
            "Eligibility: every replicate must satisfy the configured PSM, peptide, "
            f"and protein combined estimates in scope={scope}"
        ),
        (
            "Ranking: largest minimum target-protein yield, then mean protein yield, "
            "then Level-4 peptide and PSM yield, then smallest worst-case combined estimates"
        ),
        "",
        f"min_null_rank: {window[0]}",
        f"max_null_rank: {window[1]}",
        f"replicates: {len(rows)}",
        f"scope: {scope}",
        f"effective_r_psm: {ratios.psm:.12g}",
        f"effective_r_peptide: {ratios.peptide:.12g}",
        f"effective_r_protein: {ratios.protein:.12g}",
        f"max_psm_combined_fdp: {limits.psm:.12g}",
        f"max_peptide_combined_fdp: {limits.peptide:.12g}",
        f"max_protein_combined_fdp: {limits.protein:.12g}",
    ]
    for index, row in enumerate(rows, start=1):
        metrics = _scope_metrics(row, scope)
        lines.extend(
            [
                "",
                f"replicate_{index}_source: {row.source}",
                f"replicate_{index}_run_id: {row.row['run_id']}",
                f"replicate_{index}_target_protein: {row.row['target_protein']}",
                f"replicate_{index}_ent_protein: {row.row['ent_protein']}",
                f"replicate_{index}_level4_target_peptide: {row.row['level4_target_peptide']}",
                f"replicate_{index}_level4_ent_peptide: {row.row['level4_ent_peptide']}",
                f"replicate_{index}_level4_target_psm: {row.row['level4_target_psm']}",
                f"replicate_{index}_level4_ent_psm: {row.row['level4_ent_psm']}",
                f"replicate_{index}_combined_psm_fdp: {_format_pct(metrics['psm'].combined)}",
                f"replicate_{index}_combined_peptide_fdp: {_format_pct(metrics['peptide'].combined)}",
                f"replicate_{index}_combined_protein_fdp: {_format_pct(metrics['protein'].combined)}",
                f"replicate_{index}_raw_combined_psm_fdp: {_format_pct(row.raw['psm'].combined)}",
                f"replicate_{index}_raw_combined_peptide_fdp: {_format_pct(row.raw['peptide'].combined)}",
                f"replicate_{index}_raw_combined_protein_fdp: {_format_pct(row.raw['protein'].combined)}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="cumulative_results_long.csv; repeat for independent entrapment replicates",
    )
    parser.add_argument("--scope", choices=("raw", "level4"), default="level4")
    parser.add_argument("--psm-r", type=float, default=1.0)
    parser.add_argument("--peptide-r", type=float, default=1.0)
    parser.add_argument("--protein-r", type=float, default=1.0)
    parser.add_argument("--max-psm-fdp", type=float, default=0.01)
    parser.add_argument("--max-peptide-fdp", type=float, default=0.01)
    parser.add_argument("--max-protein-fdp", type=float, default=0.01)
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--best-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ratios = EffectiveRatios(args.psm_r, args.peptide_r, args.protein_r)
    limits = Limits(args.max_psm_fdp, args.max_peptide_fdp, args.max_protein_fdp)
    try:
        ratios.validate()
        limits.validate()
        replicates = [read_results(path, ratios) for path in args.input]
        grouped = group_common_windows(replicates)
        if not grouped:
            raise ValueError("replicate CSVs have no common rank windows")
        if args.audit_csv:
            write_audit(args.audit_csv, grouped, args.scope, limits)
        best = select_best(grouped, args.scope, limits)
        if best is None:
            print("No rank window satisfies all FDRBench combined bounds.", file=sys.stderr)
            return 2
        summary = format_best_summary(*best, args.scope, ratios, limits)
        if args.best_output:
            args.best_output.write_text(summary)
        sys.stdout.write(summary)
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
