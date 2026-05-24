#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import psm_utils.io
from psm_utils import PSM, PSMList
from ms2rescore.feature_generators import FEATURE_GENERATORS
from ms2rescore.parse_spectra import add_precursor_values


def spectrum_path_for_raw_file(raw_file, configured_paths):
    raw_file = str(raw_file)

    if isinstance(configured_paths, str):
        return configured_paths

    for path in configured_paths:
        path = str(path)
        if os.path.basename(path) == raw_file:
            return path

    for path in configured_paths:
        path = str(path)
        if raw_file in path:
            return path

    raise RuntimeError(
        f"Could not map raw_file={raw_file!r} to any configured spectrum path: {configured_paths}"
    )


def make_psm(row):
    return PSM(
		peptidoform=str(row["peptidoform"]),
		spectrum_id=str(row["spectrum_id"]),
		run=str(row["raw_file"]),
		collection=None,
		score=float(row["score"]),
		qvalue=float(row.get("qvalue", 1.0)),
		pep=float(row.get("pep", 1.0)),
		is_decoy=False,
		rank=int(row["rank"]),
		source=str(row["raw_file"]),
        provenance_data={
            "raw_file": str(row["raw_file"]),
            "sage_rank": int(row["rank"]),
            "sage_psm_id": int(row["psm_id"]),
        },
        metadata={
            "charge": int(row["charge"]),
            "retention_time": float(row["retention_time"]),
            "ion_mobility": float(row["ion_mobility"]),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = json.load(handle)

    df = pd.read_csv(cfg["psm_file"], sep="\t")

    all_psms = []

    for raw_file, sub_df in df.groupby("raw_file", sort=False):
        spectrum_path = spectrum_path_for_raw_file(raw_file, cfg["spectrum_path"])

        print(
            f"Processing raw_file={raw_file} with spectrum_path={spectrum_path} "
            f"and {len(sub_df)} candidate PSMs",
            flush=True,
        )

        psm_list = PSMList(psm_list=[make_psm(row) for _, row in sub_df.iterrows()])

        required_ms_data = {
            ms_data
            for fgen_name in cfg["feature_generators"].keys()
            for ms_data in FEATURE_GENERATORS[fgen_name].required_ms_data
        }

        available_ms_data = add_precursor_values(
            psm_list,
            required_ms_data,
            spectrum_path=spectrum_path,
            spectrum_id_pattern=cfg.get("spectrum_id_pattern"),
        )

        for fgen_name, fgen_cfg in cfg["feature_generators"].items():
            conf = dict(cfg)
            conf.update(fgen_cfg)

            # Important: per-run single path, not the original list.
            conf["spectrum_path"] = spectrum_path

            fgen = FEATURE_GENERATORS[fgen_name](**conf)

            missing = fgen.required_ms_data - available_ms_data
            if missing:
                print(f"Skipping {fgen_name}; missing {missing}", flush=True)
                continue

            print(f"Adding features from {fgen_name}", flush=True)
            fgen.add_features(psm_list)

        all_psms.extend(psm_list.psm_list)

    out = cfg["output_path"] + ".psms.tsv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    psm_utils.io.write_file(PSMList(psm_list=all_psms), out, filetype="tsv")

    print(f"Wrote feature-enriched PSMs to {out}", flush=True)


if __name__ == "__main__":
    main()