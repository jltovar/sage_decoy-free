#!/usr/bin/env python3

import ast
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("sage_ms2rescore_feature_only.py")


class SpawnSafeImportsTests(unittest.TestCase):
    def test_ms2rescore_imports_are_deferred_to_main(self):
        tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))

        module_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertFalse(
            any(
                (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").startswith("ms2rescore")
                )
                or (
                    isinstance(node, ast.Import)
                    and any(alias.name.startswith("ms2rescore") for alias in node.names)
                )
                for node in module_imports
            ),
            "MS2Rescore imports at module scope initialize TensorFlow in spawned MS2PIP workers",
        )

        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        deferred_modules = {
            node.module
            for node in ast.walk(main)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertIn("ms2rescore.feature_generators", deferred_modules)
        self.assertIn("ms2rescore.parse_spectra", deferred_modules)


if __name__ == "__main__":
    unittest.main()
