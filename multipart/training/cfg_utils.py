from __future__ import annotations
import types
from pathlib import Path
from pprint import pformat
import training.config as cfg
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def show_run_plot(run_dir: str | Path,
                  filename: str = "pred_plot.png",
                  figsize=(6, 4)):
    """
    Display the prediction plot that training saved in <run_dir>/<filename>
    directly inside a Jupyter notebook.

    Parameters
    ----------
    run_dir : str | Path
        Timestamped artefact directory, e.g.  artifacts/20250708_113045
    filename : str
        Image file to load (default: 'pred_plot.png')
    figsize : tuple(float,float)
        Size for matplotlib.figure
    """

    # Append "artifacts" to run_dir - if not already included
    if "artifacts" not in str(run_dir):
        run_dir = Path(cfg.ARTIFACT_DIR) / run_dir


    img_path = Path(run_dir) / filename
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    img = Image.open(img_path)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.title(img_path.name)
    plt.show()

def show_loss_plot(run_dir: str | Path,
                  filename: str = "loss_curve.png",
                  figsize=(6, 4)):
    """
    Display the loss plot that training saved in <run_dir>/<filename>
    directly inside a Jupyter notebook.

    Parameters
    ----------
    run_dir : str | Path
        Timestamped artefact directory, e.g.  artifacts/20250708_113045
    filename : str
        Image file to load (default: 'pred_plot.png')
    figsize : tuple(float,float)
        Size for matplotlib.figure
    """

    # Append "artifacts" to run_dir - if not already included
    if "artifacts" not in str(run_dir):
        run_dir = Path(cfg.ARTIFACT_DIR) / run_dir


    img_path = Path(run_dir) / filename
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    img = Image.open(img_path)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.title(img_path.name)
    plt.show()

def _public_cfg_keys():
    return [k for k in dir(cfg)
            if not k.startswith("_")
            and not isinstance(getattr(cfg, k), types.ModuleType)]


def update_cfg(**kwargs):
    """
    Override arbitrary attributes in training.config.

    Example
    -------
    update_cfg(LR=1e-4, EPOCHS=200)
    """
    for k, v in kwargs.items():
        if k not in _public_cfg_keys():
            raise AttributeError(f"'{k}' is not a recognised config key")
        setattr(cfg, k, v)


def show_cfg(keys: list[str] | None = None):
    """
    Nicely print current configuration.

    Parameters
    ----------
    keys : list[str] | None
        If None → print all public keys.
    """
    keys = keys or _public_cfg_keys()
    print("\nCurrent configuration\n" + "-" * 70)
    for k in keys:
        v = getattr(cfg, k)
        if isinstance(v, (list, dict, tuple, Path)):
            pretty = pformat(v, compact=True, width=80)
            pretty = pretty.replace("\n", "\n" + " " * (len(k) + 3))
            print(f"{k:<18} = {pretty}")
        else:
            print(f"{k:<18} = {v}")
    print("-" * 70 + "\n")