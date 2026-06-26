"""Preprocess WRF-FLEXPART and ARM inputs for the HISCALE case.

This module is a scaffold for the full preprocessing workflow. The workflow
will convert archived WRF-FLEXPART output and ARM observational data into
preprocessed LD-Chem input files in ``model_inputs/``.

The main user workflow does not require this preprocessing step if the
preprocessed LD-Chem inputs are already available.
"""

from __future__ import annotations

from pathlib import Path


def preprocess_inputs(raw_data_dir: Path, output_dir: Path) -> None:
    """Regenerate LD-Chem inputs from archived WRF-FLEXPART and ARM data.

    Parameters
    ----------
    raw_data_dir
        Directory containing archived WRF-FLEXPART and ARM input files.
    output_dir
        Directory where preprocessed LD-Chem input files should be written.

    Notes
    -----
    This function is intentionally a placeholder until the full preprocessing
    workflow is added. The legacy HISCALE example contains source material for
    the aerosol-population construction step.
    """
    raise NotImplementedError(
        "The HISCALE preprocessing workflow has not been implemented yet. "
        "Use the preprocessed inputs in model_inputs/ to run the ensemble."
    )
