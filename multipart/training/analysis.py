"""
analysis.py – Helper functions to evaluate and visualise model output.

Example
-------
import pandas as pd
import analysis as ana

df = pd.read_parquet("predictions.parquet")   # whatever your file is
run_dir = "figures"                           # where to store PNGs

metrics = ana.run_all(df, run_dir)            # one-liner
print(metrics)
"""

from pathlib import Path
from typing   import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ------------------------------------------------------------------ #
# -----------------------  basic metrics ---------------------------- #
# ------------------------------------------------------------------ #
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae (a, b): return float(np.mean(np.abs(a - b)))

def nll_gauss(y, mu, sigma):
    var = sigma ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * var) +
                         0.5 * ( (y - mu) ** 2 ) / var))

def coverage(y, mu, sigma, z):
    """Empirical P(|y-mu| <= z * sigma)."""
    return float((np.abs(y - mu) <= z * sigma).mean())

# ------------------------------------------------------------------ #
# -------------------  1. Metrics table ----------------------------- #
# ------------------------------------------------------------------ #
def make_metrics_table(df: pd.DataFrame,
                       targets: List[str],
                       tf_eps: float = 1e-12) -> pd.DataFrame:
    """
    Compute point-estimate and (if available) probabilistic metrics.

    Returns
    -------
    pandas.DataFrame  indexed by target with columns
        RMSE_HS, RMSE_pred, MAE_HS, MAE_pred,
        [NLL, sharpness, z_var, cov_68, cov_90, cov_95]
    """
    rows = []
    for t in targets:
        y_true = df[f"true_{t}"].values
        y_hs   = df[f"{t}_HS"].values
        mu     = df[f"pred_{t}"].values

        row = dict(
            target     = t,
            RMSE_HS    = rmse(y_true, y_hs),
            RMSE_pred  = rmse(y_true, mu),
            MAE_HS     = mae(y_true, y_hs),
            MAE_pred   = mae(y_true, mu)
        )

        std_col = f"std_ale_{t}"
        if std_col in df.columns:
            sigma = df[std_col].values + tf_eps
            row.update(
                NLL       = nll_gauss(y_true, mu, sigma),
                sharpness = float(np.mean(sigma)),
                z_var     = float(np.mean(((y_true - mu)/sigma)**2)),
                cov_68    = coverage(y_true, mu, sigma, 1.0),
                cov_90    = coverage(y_true, mu, sigma, norm.ppf(0.95)),
                cov_95    = coverage(y_true, mu, sigma, norm.ppf(0.975)),
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index("target")


# ------------------------------------------------------------------ #
# --------------- 2. Histogram of normalised residuals -------------- #
# ------------------------------------------------------------------ #
def plot_z_histograms(df: pd.DataFrame,
                      targets: List[str],
                      save_dir: Optional[Path] = None) -> None:
    """
    For every target plot histogram of z = (y_true - mu)/sigma.

    If `save_dir` is given the PNGs are stored there, otherwise shown inline.
    """
    save_dir = Path(save_dir) if save_dir is not None else None
    for t in targets:
        std_col = f"std_ale_{t}"
        if std_col not in df.columns:
            continue

        z = (df[f"true_{t}"] - df[f"pred_{t}"]) / df[std_col]
        plt.figure(figsize=(4, 3))
        plt.hist(z, bins=40, density=True, alpha=0.6, label="z histogram")
        xs = np.linspace(-4, 4, 200)
        plt.plot(xs, norm.pdf(xs), 'r', lw=2, label="N(0,1)")
        plt.title(f"{t}:\n normalised residuals")
        plt.xlabel("z"); plt.ylabel("density")
        plt.legend(); plt.grid(True); plt.tight_layout()

        if save_dir is None:
            plt.show()
        else:
            out = save_dir / f"z_hist_{t}.png"
            plt.savefig(out, bbox_inches="tight")
            print(f"saved {out}")
            plt.close()


# ------------------------------------------------------------------ #
# ------------------- 3. Calibration curves ------------------------- #
# ------------------------------------------------------------------ #
def plot_calibration_curves(df: pd.DataFrame,
                            targets: List[str],
                            save_dir: Optional[Path] = None) -> None:
    """
    Plot nominal vs empirical coverage curves.
    """
    qs = np.linspace(0.05, 0.95, 19)

    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], 'k--', label="ideal")

    for t in targets:
        std_col = f"std_ale_{t}"
        if std_col not in df.columns:
            continue
        y  = df[f"true_{t}"].values
        mu = df[f"pred_{t}"].values
        sig = df[std_col].values

        cov = _empirical_coverage(y, mu, sig, qs)
        plt.plot(qs, cov, marker='o', label=t)

    plt.xlabel("Nominal coverage probability")
    plt.ylabel("Empirical coverage")
    plt.title("Calibration curve")
    plt.legend(); plt.grid(True); plt.tight_layout()

    if save_dir is None:
        plt.show()
    else:
        out = Path(save_dir) / "calibration_curve.png"
        plt.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
        plt.close()


def _empirical_coverage(y, mu, sigma, qs):
    cov = []
    for q in qs:
        z = norm.ppf(0.5 + q / 2)
        cov.append(((np.abs(y - mu) <= z * sigma)).mean())
    return np.array(cov)


# ------------------------------------------------------------------ #
# ---------------- 4. Sharpness vs squared error -------------------- #
# ------------------------------------------------------------------ #
def plot_sharpness_vs_error(df,
                            targets: List[str],
                            save_dir: Optional[Path] = None,
                            quant: float = 0.99,
                            log: bool = True):
    """
    Scatter of predicted variance vs squared error.

    Parameters
    ----------
    df         : DataFrame with columns true_<t>, pred_<t>, std_ale_<t>
    targets    : list of target names
    save_dir   : directory to save PNGs; if None figures are shown inline
    quant      : upper quantile for axis clipping (e.g. 0.99 keeps 1 % tails)
    log        : if True use symlog scale on both axes
    """
    for t in targets:
        std_col = f"std_ale_{t}"
        if std_col not in df.columns:
            continue

        err2 = (df[f"true_{t}"] - df[f"pred_{t}"]) ** 2
        sig2 = df[std_col] ** 2

        # -------- axis limits (percentile clipping) ----------------
        x_max = np.quantile(sig2, quant)
        y_max = np.quantile(err2, quant)

        plt.figure(figsize=(4, 3))
        plt.scatter(sig2, err2, alpha=0.5)

        # reference y = x line (ideal calibration)
        xs = np.linspace(0, x_max, 200)
        plt.plot(xs, xs, 'k--', lw=1)

        if log:
            plt.xscale('symlog', linthresh=1e-6)   # avoids log(0)
            plt.yscale('symlog', linthresh=1e-6)

        plt.xlim(0, x_max)
        plt.ylim(0, y_max)

        plt.xlabel("Predicted variance σ²")
        plt.ylabel("Squared error")
        plt.title(f"{t}: sharpness vs error")
        plt.grid(True)
        plt.tight_layout()

        if save_dir is None:
            plt.show()
        else:
            out = Path(save_dir) / f"sharp_vs_err_{t}.png"
            plt.savefig(out, bbox_inches="tight")
            print(f"saved {out}")
            plt.close()

# ------------------------------------------------------------------ #
# ------------------- 5. Convenience wrapper ------------------------ #
# ------------------------------------------------------------------ #
def run_all(df: pd.DataFrame,
            save_dir: Optional[str | Path] = None,
            targets: Optional[List[str]] = None,
            show_metrics: bool = True):
    """
    Run all analyses (metrics table + 3 plot groups).

    Parameters
    ----------
    df        : DataFrame with columns
                  true_<T>, <T>_HS, pred_<T>, [std_ale_<T>] for every T in targets
    save_dir  : str or Path or None.  If given, figures are saved there.
                If None, they are shown inline (Jupyter).
    targets   : list of target names.  Default ['Qext','SSA','g']
    show_metrics : if True, display DataFrame in notebook.

    Returns
    -------
    pandas.DataFrame with metrics.
    """
    if targets is None:
        targets = ["Qext", "SSA", "g"]

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1) metrics table
    metrics = make_metrics_table(df, targets)
    if show_metrics:
        try:
            from IPython.display import display
            display(metrics.style.format("{:.3g}"))
        except Exception:  # outside notebook
            print(metrics.to_string(float_format="%.3g"))

    # 2) histograms
    plot_z_histograms(df, targets, save_dir)

    # 3) calibration curve
    plot_calibration_curves(df, targets, save_dir)

    # 4) sharpness vs error
    plot_sharpness_vs_error(df, targets, save_dir)

    return metrics