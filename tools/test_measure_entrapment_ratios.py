#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.measure_entrapment_ratios import measure, peptidoform_count


class EntrapmentRatioTests(unittest.TestCase):
    def test_counts_variable_modification_assignments(self):
        self.assertEqual(peptidoform_count(["AMM"], {"M": [15.994915]}, 2), 4)

    def test_measures_unique_digest_spaces(self):
        config = {
            "database": {
                "enzyme": {
                    "missed_cleavages": 0,
                    "min_len": 2,
                    "max_len": 20,
                    "cleave_at": "K",
                    "restrict": "P",
                    "c_terminal": True,
                    "semi_enzymatic": False,
                },
                "peptide_min_mass": 0,
                "peptide_max_mass": 5000,
                "static_mods": {},
                "variable_mods": {"M": [15.994915]},
                "max_variable_mods": 2,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "test.fasta"
            fasta.write_text(
                ">target_one\nAMMKAAAAK\n"
                ">target_two\nAMMK\n"
                ">Ent_one\nMMMMKGGGGK\n"
            )
            result = measure(config, fasta, "Ent_")
        self.assertEqual(result.target_proteins, 2)
        self.assertEqual(result.entrapment_proteins, 1)
        self.assertEqual(result.target_peptides, 2)
        self.assertEqual(result.entrapment_peptides, 2)
        self.assertEqual(result.target_peptidoforms, 5)
        self.assertEqual(result.entrapment_peptidoforms, 12)
        self.assertAlmostEqual(result.peptide_r, 1.0)
        self.assertAlmostEqual(result.psm_r, 12 / 5)
        self.assertAlmostEqual(result.protein_r, 0.5)


if __name__ == "__main__":
    unittest.main()
