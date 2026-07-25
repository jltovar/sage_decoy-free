#!/usr/bin/env python3

import unittest

from tools.make_fdr_ablation_configs import make_variants


class FdrAblationConfigTests(unittest.TestCase):
    def test_generates_factorial_without_changing_locked_window(self):
        source = {
            "output_directory": "old",
            "fdr": {
                "moments_min_null_rank": 5,
                "moments_max_null_rank": 9,
                "enable_rt_confidence_adjustment": True,
                "reproducibility": {"use_cross_run_recurrence": True},
            },
        }
        variants = make_variants(source, "/results")
        expected = {
            "full": (True, True),
            "no_rt": (False, True),
            "no_recurrence": (True, False),
            "no_rt_no_recurrence": (False, False),
        }
        for name, (rt_enabled, recurrence_enabled) in expected.items():
            fdr = variants[name]["fdr"]
            self.assertEqual(fdr["moments_min_null_rank"], 5)
            self.assertEqual(fdr["moments_max_null_rank"], 9)
            self.assertEqual(fdr["enable_rt_confidence_adjustment"], rt_enabled)
            self.assertEqual(
                fdr["reproducibility"]["use_cross_run_recurrence"],
                recurrence_enabled,
            )
        self.assertTrue(source["fdr"]["enable_rt_confidence_adjustment"])


if __name__ == "__main__":
    unittest.main()
