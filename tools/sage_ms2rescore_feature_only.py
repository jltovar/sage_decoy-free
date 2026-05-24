#!/usr/bin/env python

import argparse
import json
import pandas as pd
import psm_utils.io
from psm_utils import PSM, PSMList
from ms2rescore.feature_generators import FEATURE_GENERATORS
from ms2rescore.parse_spectra import add_precursor_values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)

    df = pd.read_csv(cfg["psm_file"], sep="\t")

    psms = []
    for _, r in df.iterrows():
        psms.append(
            PSM(
                peptidoform=str(r["peptidoform"]),
                spectrum_id=str(r["spectrum_id"]),
                run=str(r["raw_file"]),
                collection=None,
                score=float(r["score"]),
                qvalue=1.0,
                pep=1.0,
                is_decoy=False,
                rank=int(r["rank"]),
                source=str(r["raw_file"]),
                provenance_data={
                    "raw_file": str(r["raw_file"]),
                    "sage_rank": int(r["rank"]),
                    "sage_psm_id": int(r["psm_id"]),
                },
                metadata={
                    "charge": int(r["charge"]),
                    "retention_time": float(r["retention_time"]),
                    "ion_mobility": float(r["ion_mobility"]),
                },
            )
        )

    psm_list = PSMList(psm_list=psms)

    required_ms_data = {
        ms_data
        for fgen_name in cfg["feature_generators"].keys()
        for ms_data in FEATURE_GENERATORS[fgen_name].required_ms_data
    }

    available_ms_data = add_precursor_values(
        psm_list,
        required_ms_data,
        spectrum_path=cfg["spectrum_path"],
        spectrum_id_pattern=cfg.get("spectrum_id_pattern"),
    )

    feature_names = set()

    for fgen_name, fgen_cfg in cfg["feature_generators"].items():
        conf = dict(cfg)
        conf.update(fgen_cfg)
        fgen = FEATURE_GENERATORS[fgen_name](**conf)

        missing = fgen.required_ms_data - available_ms_data
        if missing:
            print(f"Skipping {fgen_name}; missing {missing}")
            continue

        fgen.add_features(psm_list)
        feature_names.update(fgen.feature_names)

    out = cfg["output_path"] + ".psms.tsv"
    psm_utils.io.write_file(psm_list, out, filetype="tsv")

if __name__ == "__main__":
    main()