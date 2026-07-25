#!/usr/bin/env python3
"""Create a locked-window 2x2 RT/recurrence ablation configuration set."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Sequence


VARIANTS = {
    "full": (True, True),
    "no_rt": (False, True),
    "no_recurrence": (True, False),
    "no_rt_no_recurrence": (False, False),
}


def make_variants(config: dict, output_root: str) -> dict[str, dict]:
    fdr = config.get("fdr")
    if not isinstance(fdr, dict):
        raise ValueError("input config has no fdr object")
    reproducibility = fdr.get("reproducibility")
    if not isinstance(reproducibility, dict):
        raise ValueError("input config has no fdr.reproducibility object")

    variants: dict[str, dict] = {}
    for name, (rt_enabled, recurrence_enabled) in VARIANTS.items():
        variant = copy.deepcopy(config)
        variant["output_directory"] = str(Path(output_root) / name)
        variant["fdr"]["enable_rt_confidence_adjustment"] = rt_enabled
        variant["fdr"]["reproducibility"][
            "use_cross_run_recurrence"
        ] = recurrence_enabled
        variants[name] = variant
    return variants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="locked Sage JSON")
    parser.add_argument(
        "--output-root",
        required=True,
        help="root for variant result directories and the configs/ subdirectory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with args.input.open() as handle:
            config = json.load(handle)
        variants = make_variants(config, args.output_root)
        config_directory = Path(args.output_root) / "configs"
        config_directory.mkdir(parents=True, exist_ok=True)
        for name, variant in variants.items():
            output = config_directory / f"{name}.json"
            output.write_text(json.dumps(variant, indent=2) + "\n")
            print(output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
