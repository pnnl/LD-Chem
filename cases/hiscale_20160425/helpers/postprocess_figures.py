"""Postprocess HISCALE LD-Chem outputs.

This module is a placeholder for optional paper-figure post-processing.
"""

from __future__ import annotations

from pathlib import Path


def postprocess_figures(input_dir: Path, output_dir: Path) -> None:
    """Generate postprocessed analysis products or figures.

    Parameters
    ----------
    input_dir
        Directory containing LD-Chem output files.
    output_dir
        Directory where postprocessed files should be written.
    """
    raise NotImplementedError(
        "Figure post-processing has not been implemented for this release."
    )
