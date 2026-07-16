#!/usr/bin/env python3
"""Create development and held-out Sage validation configurations.

The selected null window is written identically to both outputs.  Optimize only
the development configuration, then evaluate the already locked window on the
held-out configuration without running the optimizer on the held-out results.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Sequence


MODEL_KEYS = {
    "moments": ("moments_min_null_rank", "moments_max_null_rank"),
    "mle": ("mle_min_null_rank", "mle_max_null_rank"),
    "lower_order": ("lower_order_min_null_rank", "lower_order_max_null_rank"),
    "msfdr": ("msfdr_min_null_rank", "msfdr_max_null_rank"),
    "msfdr1_smix": ("msfdr1_smix_min_null_rank", "msfdr1_smix_max_null_rank"),
    "msfdr2_smix": ("msfdr2_smix_min_null_rank", "msfdr2_smix_max_null_rank"),
    "nokoi": ("nokoi_min_null_rank", "nokoi_max_null_rank"),
}


def split_config(
    config: dict,
    development_pattern: str,
    holdout_pattern: str,
    model_fit: str,
    min_null_rank: int,
    max_null_rank: int,
    development_directory: str,
    holdout_directory: str,
) -> tuple[dict, dict]:
    if min_null_rank < 1 or max_null_rank < min_null_rank:
        raise ValueError("null ranks must satisfy 1 <= min <= max")
    if model_fit not in MODEL_KEYS:
        raise ValueError(f"unsupported model fit: {model_fit}")

    development_regex = re.compile(development_pattern)
    holdout_regex = re.compile(holdout_pattern)
    development_paths: list[str] = []
    holdout_paths: list[str] = []
    unmatched: list[str] = []
    overlapping: list[str] = []

    for path in config.get("mzml_paths", []):
        in_development = development_regex.search(path) is not None
        in_holdout = holdout_regex.search(path) is not None
        if in_development and in_holdout:
            overlapping.append(path)
        elif in_development:
            development_paths.append(path)
        elif in_holdout:
            holdout_paths.append(path)
        else:
            unmatched.append(path)

    if overlapping:
        raise ValueError(f"paths match both split patterns: {overlapping}")
    if unmatched:
        raise ValueError(f"paths match neither split pattern: {unmatched}")
    if not development_paths or not holdout_paths:
        raise ValueError("both development and holdout splits must be non-empty")

    minimum_key, maximum_key = MODEL_KEYS[model_fit]

    def make(paths: list[str], output_directory: str) -> dict:
        result = copy.deepcopy(config)
        result["mzml_paths"] = paths
        result["output_directory"] = output_directory
        result.setdefault("fdr", {})["model_fit"] = model_fit
        result["fdr"][minimum_key] = min_null_rank
        result["fdr"][maximum_key] = max_null_rank
        return result

    return (
        make(development_paths, development_directory),
        make(holdout_paths, holdout_directory),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--development-regex", required=True)
    parser.add_argument("--holdout-regex", required=True)
    parser.add_argument("--development-directory", required=True)
    parser.add_argument("--holdout-directory", required=True)
    parser.add_argument("--model-fit", choices=sorted(MODEL_KEYS), default="moments")
    parser.add_argument("--min-null-rank", type=int, required=True)
    parser.add_argument("--max-null-rank", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with args.input.open() as handle:
            config = json.load(handle)
        development, holdout = split_config(
            config,
            args.development_regex,
            args.holdout_regex,
            args.model_fit,
            args.min_null_rank,
            args.max_null_rank,
            args.development_directory,
            args.holdout_directory,
        )
        args.development_output.write_text(json.dumps(development, indent=2) + "\n")
        args.holdout_output.write_text(json.dumps(holdout, indent=2) + "\n")
        print(
            f"wrote development={len(development['mzml_paths'])} files and "
            f"holdout={len(holdout['mzml_paths'])} files with locked "
            f"{args.model_fit} window [{args.min_null_rank}, {args.max_null_rank}]"
        )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
