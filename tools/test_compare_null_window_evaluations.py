#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.compare_null_window_evaluations import compare, read_legacy_rows


class CompareNullWindowEvaluationsTests(unittest.TestCase):
    def test_wide_legacy_csv_is_transposed(self):
        with tempfile.TemporaryDirectory() as temporary:
            legacy = Path(temporary) / "legacy_wide.csv"
            legacy.write_text(
                "metric,run_001,run_002\n"
                "min_null_rank,1,3\n"
                "max_null_rank,1,5\n"
            )
            rows = read_legacy_rows(legacy)
            self.assertEqual(rows[1]["run_id"], "run_002")
            self.assertEqual(rows[1]["min_null_rank"], "3")
            self.assertEqual(rows[1]["max_null_rank"], "5")

    def test_rank1_setup_rows_are_excluded_and_exact_grid_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy.csv"
            fields = [
                "run_id", "min_null_rank", "max_null_rank", "target_psm", "ent_psm",
                "target_peptide", "ent_peptide", "target_protein", "ent_protein",
                "level4_target_psm", "level4_ent_psm", "level4_target_peptide",
                "level4_ent_peptide", "feasible",
            ]
            with legacy.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {
                        "run_id": "setup", "min_null_rank": 1, "max_null_rank": 1,
                        "target_psm": 0, "ent_psm": 0, "target_peptide": 0,
                        "ent_peptide": 0, "target_protein": 0, "ent_protein": 0,
                        "level4_target_psm": 0, "level4_ent_psm": 0,
                        "level4_target_peptide": 0, "level4_ent_peptide": 0,
                        "feasible": 0,
                    },
                    {
                        "run_id": "run", "min_null_rank": 3, "max_null_rank": 5,
                        "target_psm": 100, "ent_psm": 0, "target_peptide": 20,
                        "ent_peptide": 0, "target_protein": 5, "ent_protein": 0,
                        "level4_target_psm": 90, "level4_ent_psm": 0,
                        "level4_target_peptide": 18, "level4_ent_peptide": 0,
                        "feasible": 1,
                    },
                ])
            native = root / "native.json"
            native.write_text(json.dumps([{
                "min_rank": 3, "max_rank": 5, "validation_scope": "level4",
                "target_psms": 90, "entrapment_psms": 0,
                "target_peptides": 18, "entrapment_peptides": 0,
                "target_proteins": 5, "entrapment_proteins": 0,
                "psm_fdp": 0.0, "peptide_fdp": 0.0, "protein_fdp": 0.0,
                "feasible": True, "selected": True,
            }]))
            report = compare(
                legacy, native, None,
                {"psm": 1.0, "peptide": 1.0, "protein": 1.0},
                0.01, (3, 5), 0.0, 0.0,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["legacy_visited_windows"], 1)
            self.assertEqual(report["summary"]["exact_count_rows"], 1)

    def test_legacy_results_use_shared_canonical_counting_definition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy.csv"
            with legacy.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "run_id", "min_null_rank", "max_null_rank", "target_psm",
                    "ent_psm", "target_peptide", "ent_peptide", "target_protein",
                    "ent_protein", "level4_target_psm", "level4_ent_psm",
                    "level4_target_peptide", "level4_ent_peptide", "feasible",
                ])
                writer.writeheader()
                writer.writerow({
                    "run_id": "run_001", "min_null_rank": 3, "max_null_rank": 5,
                    "target_psm": 2, "ent_psm": 0, "target_peptide": 2,
                    "ent_peptide": 0, "target_protein": 1, "ent_protein": 0,
                    "level4_target_psm": 2, "level4_ent_psm": 0,
                    "level4_target_peptide": 2, "level4_ent_peptide": 0,
                    "feasible": 1,
                })
            results_root = root / "outdirs"
            results = results_root / "run_001" / "results.sage.tsv"
            results.parent.mkdir(parents=True)
            with results.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, delimiter="\t", fieldnames=[
                    "psm_id", "rank", "label", "proteins", "peptide",
                    "decoy_free_q_value", "decoy_free_peptide_q",
                    "decoy_free_protein_q", "decoy_free_protein_supported_peptide",
                    "decoy_free_peptide_supported_psm",
                ])
                writer.writeheader()
                writer.writerows([
                    {
                        "psm_id": "a", "rank": 1, "label": 1, "proteins": "P1",
                        "peptide": "PEPTI[+1]DE", "decoy_free_q_value": 0.001,
                        "decoy_free_peptide_q": 0.001, "decoy_free_protein_q": 0.001,
                        "decoy_free_protein_supported_peptide": True,
                        "decoy_free_peptide_supported_psm": True,
                    },
                    {
                        "psm_id": "b", "rank": 1, "label": 1, "proteins": "P1",
                        "peptide": "PEPTLDE", "decoy_free_q_value": 0.001,
                        "decoy_free_peptide_q": 0.001, "decoy_free_protein_q": 0.001,
                        "decoy_free_protein_supported_peptide": True,
                        "decoy_free_peptide_supported_psm": True,
                    },
                ])
            native = root / "native.json"
            native.write_text(json.dumps([{
                "min_rank": 3, "max_rank": 5, "validation_scope": "level4",
                "target_psms": 2, "entrapment_psms": 0,
                "target_peptides": 1, "entrapment_peptides": 0,
                "target_proteins": 1, "entrapment_proteins": 0,
                "psm_fdp": 0.0, "peptide_fdp": 0.0, "protein_fdp": 0.0,
                "feasible": True, "selected": True,
            }]))
            report = compare(
                legacy, native, results_root,
                {"psm": 1.0, "peptide": 1.0, "protein": 1.0},
                0.01, (3, 5), 0.0, 0.0,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["exact_count_rows"], 1)

    def test_undefined_fdp_is_allowed_only_for_mutually_infeasible_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "legacy.csv"
            fields = [
                "run_id", "min_null_rank", "max_null_rank", "target_psm", "ent_psm",
                "target_peptide", "ent_peptide", "target_protein", "ent_protein",
                "level4_target_psm", "level4_ent_psm", "level4_target_peptide",
                "level4_ent_peptide", "feasible",
            ]
            with legacy.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "run_id": "run", "min_null_rank": 3, "max_null_rank": 3,
                    "target_psm": 0, "ent_psm": 0, "target_peptide": 0,
                    "ent_peptide": 0, "target_protein": 0, "ent_protein": 0,
                    "level4_target_psm": 0, "level4_ent_psm": 0,
                    "level4_target_peptide": 0, "level4_ent_peptide": 0,
                    "feasible": 0,
                })
            native = root / "native.json"
            native.write_text(json.dumps([{
                "min_rank": 3, "max_rank": 3, "validation_scope": "level4",
                "target_psms": 0, "entrapment_psms": 0,
                "target_peptides": 0, "entrapment_peptides": 0,
                "target_proteins": 0, "entrapment_proteins": 0,
                "psm_fdp": None, "peptide_fdp": None, "protein_fdp": None,
                "feasible": False, "selected": False,
            }]))
            report = compare(
                legacy, native, None,
                {"psm": 1.0, "peptide": 1.0, "protein": 1.0},
                0.01, None, 0.0, 0.0,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["fdp_comparable_rows"], 0)
            self.assertTrue(report["summary"]["required_fdp_values_available"])


if __name__ == "__main__":
    unittest.main()
