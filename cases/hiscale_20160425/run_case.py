#!/usr/bin/env python3
"""Run the HISCALE April 25, 2016 LD-Chem case."""

from __future__ import annotations

import argparse
from pathlib import Path

from helpers.preprocess_inputs import preprocess_inputs
from helpers.run_ensemble import run_ensemble
from helpers.postprocess_figures import postprocess_figures


CASE_ID = "hiscale_20160425"
CASE_DESCRIPTION = "HISCALE April 25, 2016 LD-Chem trajectory ensemble case"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CASE_DESCRIPTION)
    parser.add_argument(
        "--preprocess-inputs",
        action="store_true",
        help="Regenerate LD-Chem inputs from WRF-FLEXPART and ARM data.",
    )
    parser.add_argument(
        "--run-ensemble",
        action="store_true",
        help="Run the LD-Chem ensemble from preprocessed inputs.",
    )
    parser.add_argument(
        "--postprocess-figures",
        action="store_true",
        help="Run optional figure/post-processing workflow.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("model_inputs"),
        help="Directory containing committed LD-Chem-ready input files.",
    )
    parser.add_argument(
        "--generated-input-dir",
        type=Path,
        default=Path("generated_model_inputs"),
        help="Directory where optional generated LD-Chem-ready inputs are written.",
    )
    parser.add_argument(
        "--obs-data-dir",
        type=Path,
        default=Path("data/obs"),
        help="Directory containing local observational input files for optional preprocessing.",
    )
    parser.add_argument(
        "--flexpart-file",
        type=Path,
        default=Path("data/flexpart/FLEXPART_output_traj_0001.txt"),
        help="Path to the local FLEXPART text trajectory file for optional preprocessing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_outputs"),
        help="Directory for generated LD-Chem model outputs.",
    )
    parser.add_argument(
        "--max-simulations",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default behavior: run the ensemble from preprocessed inputs.
    if not (args.preprocess_inputs or args.run_ensemble or args.postprocess_figures):
        args.run_ensemble = True

    run_input_dir = args.input_dir

    if args.preprocess_inputs:
        preprocess_inputs(
            obs_data_dir=args.obs_data_dir,
            flexpart_file=args.flexpart_file,
            output_dir=args.generated_input_dir,
        )
        run_input_dir = args.generated_input_dir

    if args.run_ensemble:
        run_ensemble(
            input_dir=run_input_dir,
            output_dir=args.output_dir,
            max_simulations=args.max_simulations,
        )

    if args.postprocess_figures:
        postprocess_figures(input_dir=args.output_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
