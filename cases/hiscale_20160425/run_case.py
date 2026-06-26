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
        help="Directory containing preprocessed LD-Chem inputs.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing archived WRF-FLEXPART and ARM data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated LD-Chem outputs.",
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

    if args.preprocess_inputs:
        preprocess_inputs(raw_data_dir=args.raw_data_dir, output_dir=args.input_dir)

    if args.run_ensemble:
        run_ensemble(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_simulations=args.max_simulations,
        )

    if args.postprocess_figures:
        postprocess_figures(input_dir=args.output_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
