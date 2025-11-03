"""
Utility functions that provide *one-stop* reproducibility.
"""

import os
import random
from typing import Optional

import numpy as np
import torch
import pyro


# ---------------------------------------------------------------------
def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, PyTorch (CPU & CUDA) and Pyro.

    Parameters
    ----------
    seed : int
        Global seed to be used.
    deterministic : bool
        If ``True`` switches CuDNN / Torch into deterministic mode
        (slower but fully reproducible).
    """
    # 0)  Make Python’s own hashing deterministic (important for sets / dicts)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 1)  Python RNG
    random.seed(seed)

    # 2)  NumPy RNG
    np.random.seed(seed)

    # 3)  PyTorch RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4)  Pyro RNG (internally relies on torch + python)
    pyro.set_rng_seed(seed)

    # 5)  Deterministic Torch / CuDNN (optional)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[seed_utils]  global seed set to {seed}")


# ---------------------------------------------------------------------
# Extra helpers for DataLoader reproducibility
# ---------------------------------------------------------------------
def seed_worker(worker_id: int) -> None:
    """
    To be passed as ``worker_init_fn`` to DataLoader so that each worker
    is deterministically seeded, too.

    Example
    -------
    >>> loader = DataLoader(
    ...     dataset, batch_size=64, shuffle=True,
    ...     num_workers=4,
    ...     worker_init_fn=seed_worker,
    ...     generator=make_generator(1234))
    """
    # Derive each worker’s seed in a reproducible but unique way.
    # torch.initial_seed() already incorporates the base Generator seed.
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """
    Returns a torch.Generator initialised with the provided seed.
    Pass the result to ``DataLoader(..., generator=gen)`` for
    deterministic shuffling.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
