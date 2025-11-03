# training/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import numpy as np


def plot_results(
        y_true: np.ndarray,
        mu: np.ndarray,
        std_ale: Optional[np.ndarray],
        std_epi: Optional[np.ndarray],
        low_res: np.ndarray,
        targets: List[str],
        save_path: Path,
        cap_frac: float = 0.02):
    """
    Creates a figure with one subplot per target (stacked vertically).
    The legend is placed to the right of the panels and its entries are
    stacked vertically.

    Parameters
    ----------
    y_true   : (N,D) ground-truth (high-res) outputs
    mu       : (N,D) predictive means
    std_ale  : (N,D) aleatoric std  (None if mode='none')
    std_epi  : (N,D) epistemic std  (None if deterministic head)
    low_res  : (N,D) low-res (Mie) baseline
    targets  : list with D target names
    save_path: output file
    cap_frac : half-width of aleatoric cap w.r.t. x-span (fraction)
    """
    assert mu.shape == y_true.shape
    n_t = len(targets)

    # -------------- layout: n_t rows, 1 column -----------------
    fig, axes = plt.subplots(
        n_t, 1,
        figsize=(8, 5 * n_t),       # height scales with number of rows
        squeeze=False
    )
    axes = axes[:, 0]              # flatten

    # pre-compute cap widths
    x_span = y_true.max(axis=0) - y_true.min(axis=0)
    cap_widths = cap_frac * x_span

    for j, t in enumerate(targets):
        ax = axes[j]

        # scatter points
        ax.scatter(y_true[:, j], low_res[:, j],
                   color="red", label="Low-res (Mie)",
                   alpha=0.6, s=20)
        ax.scatter(y_true[:, j], mu[:, j],
                   color="green", label="Predicted μ",
                   alpha=0.8, s=20)

        # --------- uncertainty bars ----------
        if std_ale is not None or std_epi is not None:
            if std_ale is None:            # only epistemic
                std_tot = std_epi[:, j]
                std_ale_j = None
            elif std_epi is None:          # deterministic head
                std_tot = std_ale[:, j]
                std_ale_j = std_tot
            else:                          # both
                std_tot = np.sqrt(std_ale[:, j]**2 + std_epi[:, j]**2)
                std_ale_j = std_ale[:, j]

            # total ±2σ vertical bars
            for x_gt, m, s in zip(y_true[:, j], mu[:, j], std_tot):
                ax.vlines(x_gt, m - 2*s, m + 2*s,
                          color="green", alpha=0.25, linewidth=0.8)

            # aleatoric caps
            if std_ale_j is not None:
                half_cap = cap_widths[j] / 2
                for x_gt, m, s_a in zip(y_true[:, j], mu[:, j], std_ale_j):
                    ax.hlines(m - 2*s_a, x_gt - half_cap, x_gt + half_cap,
                               color="green", alpha=0.9, linewidth=0.8)
                    ax.hlines(m + 2*s_a, x_gt - half_cap, x_gt + half_cap,
                               color="green", alpha=0.9, linewidth=0.8)

        # y = x reference
        lo = min(y_true[:, j].min(), low_res[:, j].min(), mu[:, j].min())
        hi = max(y_true[:, j].max(), low_res[:, j].max(), mu[:, j].max())
        ax.plot([lo, hi], [lo, hi], "b--", linewidth=1)

        ax.set_title(t, size=14)
        ax.set_xlabel("High-res (DDA)", size=12)
        ax.set_ylabel("Value", size=12)

    # ------------- legend outside, stacked vertically -------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='center left',
               bbox_to_anchor=(0.88, 0.5),   # x=0.88 ⇒ just right of axes
               ncol=1,                       # stacked
               frameon=False,
               fontsize=12)

    # Leave space on the right for the legend (up to 85 % of width)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(Path(save_path), bbox_inches='tight')
    print(f"plot saved to {save_path}")
    plt.close(fig)

def plot_loss(train_hist, val_hist, out_dir, fname="loss_curve.png"):
    epochs = np.arange(1, len(train_hist) + 1)
    train = np.asarray(train_hist)
    val   = np.asarray(val_hist)

    # decide plotting scale
    use_log = (train > 0).all() and (val > 0).all()

    plt.figure()
    if use_log:
        plt.semilogy(epochs, train, label="Train")
        plt.semilogy(epochs, val,   label="Val")
        plt.ylabel("Loss")
    else:
        plt.plot(epochs, train, label="Train")
        plt.plot(epochs, val,   label="Val")
        plt.ylabel("Loss")

    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / fname, dpi=300)
    plt.close()