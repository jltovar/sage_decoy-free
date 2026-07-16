#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.fdrbench_window_selector import (
    EffectiveRatios,
    Limits,
    fdrbench_estimate,
    group_common_windows,
    read_results,
    select_best,
)


FIELDS = [
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
]


def row(run_id: str, minimum: int, maximum: int, *, protein: int, ent_protein: int):
    return {
        "run_id": run_id,
        "min_null_rank": minimum,
        "max_null_rank": maximum,
        "num_ranks_used": maximum - minimum + 1,
        "target_psm": 1000,
        "target_peptide": 500,
        "target_protein": protein,
        "ent_psm": 20,
        "ent_peptide": 5,
        "ent_protein": ent_protein,
        "level4_target_peptide": 450,
        "level4_target_psm": 900,
        "level4_ent_peptide": 1,
        "level4_ent_psm": 1,
    }


class SelectorTests(unittest.TestCase):
    def write_csv(self, directory: Path, name: str, rows: list[dict[str, object]]) -> Path:
        path = directory / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_fdrbench_formula(self):
        estimate = fdrbench_estimate(entrapment=10, target=990, effective_r=0.5)
        self.assertAlmostEqual(estimate.lower, 0.01)
        self.assertAlmostEqual(estimate.combined, 0.03)

    def test_level4_can_select_higher_yield_than_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.write_csv(
                root,
                "results.csv",
                [
                    row("conservative", 1, 2, protein=100, ent_protein=0),
                    row("power", 3, 4, protein=120, ent_protein=1),
                ],
            )
            ratios = EffectiveRatios(1.0, 1.0, 1.0)
            grouped = group_common_windows([read_results(path, ratios)])
            level4 = select_best(grouped, "level4", Limits(0.01, 0.01, 0.02))
            raw = select_best(grouped, "raw", Limits(0.01, 0.01, 0.02))
            self.assertEqual(level4[0], (3, 4))
            self.assertIsNone(raw)

    def test_every_replicate_must_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.write_csv(
                root,
                "first.csv",
                [
                    row("a-safe", 1, 2, protein=100, ent_protein=0),
                    row("a-high", 3, 4, protein=130, ent_protein=0),
                ],
            )
            bad_high = row("b-high", 3, 4, protein=130, ent_protein=3)
            second = self.write_csv(
                root,
                "second.csv",
                [row("b-safe", 1, 2, protein=100, ent_protein=0), bad_high],
            )
            ratios = EffectiveRatios(1.0, 1.0, 1.0)
            grouped = group_common_windows(
                [read_results(first, ratios), read_results(second, ratios)]
            )
            best = select_best(grouped, "level4", Limits(0.01, 0.01, 0.01))
            self.assertEqual(best[0], (1, 2))


if __name__ == "__main__":
    unittest.main()
