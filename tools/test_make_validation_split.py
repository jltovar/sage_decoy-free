#!/usr/bin/env python3

import unittest

from tools.make_validation_split import split_config


class ValidationSplitTests(unittest.TestCase):
    def test_locks_same_window_in_disjoint_splits(self):
        source = {
            "mzml_paths": ["sample_01A_x.mzML", "sample_01B_x.mzML"],
            "output_directory": "old",
            "fdr": {"model_fit": "moments", "moments_min_null_rank": 1, "moments_max_null_rank": 2},
        }
        development, holdout = split_config(
            source,
            r"\d\dA_",
            r"\d\dB_",
            "moments",
            3,
            9,
            "development",
            "holdout",
        )
        self.assertEqual(development["mzml_paths"], ["sample_01A_x.mzML"])
        self.assertEqual(holdout["mzml_paths"], ["sample_01B_x.mzML"])
        for config in (development, holdout):
            self.assertEqual(config["fdr"]["moments_min_null_rank"], 3)
            self.assertEqual(config["fdr"]["moments_max_null_rank"], 9)

    def test_rejects_unmatched_paths(self):
        source = {"mzml_paths": ["sample_unknown.mzML"], "fdr": {}}
        with self.assertRaises(ValueError):
            split_config(source, "A", "B", "moments", 1, 2, "development", "holdout")


if __name__ == "__main__":
    unittest.main()
