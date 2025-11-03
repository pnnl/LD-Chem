# training/train.py
"""
End-to-end training / evaluation script for the hybrid deterministic /
Bayesian network defined in `training.model.HybridNet`.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
import pyro
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import Adam as PyroAdam
from pyro.optim import ClippedAdam

from training.data_utils import load_dataset
from training.model import (HybridNet, mse_loss,
                            gaussian_nll_diag, gaussian_nll_full, build_cov_from_params)
from training.plotting import plot_results, plot_loss

from scipy.special import expit

# Import Dict for type hinting
from typing import Dict

import training.config as cfg
import yaml


# ---------- inverse scientific transform --------------------------------
def inverse_phys_tf(arr: np.ndarray, tf_info: Dict[str, str]):
    """
    Undo per-target transforms (log, logit, none).
    """
    out = arr.copy()
    idx = {c: i for i, c in enumerate(cfg.TARGETS)}
    for col, tf in tf_info.items():
        j = idx[col]
        if tf == "log":
            out[:, j] = np.exp(out[:, j]) - cfg.TF_EPS
        elif tf == "logit":
            out[:, j] = expit(out[:, j])
    return out


# ---------- delta-method std propagation --------------------------------
def propagate_std(std_in: np.ndarray,
                  mu_in: np.ndarray,
                  tf_info):
    """
    Propagate std through inverse transforms using first-order Taylor
    expansion (delta method).
    """
    out = std_in.copy()
    idx = {c: i for i, c in enumerate(cfg.TARGETS)}
    for col, tf in tf_info.items():
        j = idx[col]
        if tf == "log":
            out[:, j] = std_in[:, j] * np.exp(mu_in[:, j])
        elif tf == "logit":
            s = expit(mu_in[:, j])
            out[:, j] = std_in[:, j] * s * (1 - s)
    return out


# --- shared smooth maps --------------------------------------------
def smooth_logvar(raw):
    return cfg.LOG_VAR_MIN + (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN) * torch.sigmoid(raw)


def smooth_offdiag(raw):
    return cfg.OFF_DIAG_MAX * torch.tanh(raw)


# ----------------------------------------------------------------------
# SVI helper (KL warm-up + per-parameter LR)
# ----------------------------------------------------------------------
def make_svi(model, epoch, warmup=cfg.WARMUP):
    beta = min(1.0, epoch / max(1, warmup))

    def scaled_guide(*a, **kw):
        with pyro.poutine.scale(scale=beta):
            return model.guide(*a, **kw)

    def lr_cfg(name, _):
        """Per-parameter optimiser settings."""
        lr = cfg.LR * cfg.LR_SCALE_FACTOR if "scale" in name else cfg.LR
        cfg_ = {"lr": lr}  # FIXME - Terrible naming clash
        if cfg.GRAD_CLIP_NORM and cfg.GRAD_CLIP_NORM > 0:
            cfg_["clip_norm"] = cfg.GRAD_CLIP_NORM
        return cfg_

    return SVI(
        model,
        scaled_guide if beta < 1.0 else model.guide,
        ClippedAdam(lr_cfg),  # <- use the clipped optimiser
        loss=TraceMeanField_ELBO(),
    )


# ----------------------------------------------------------------------
# Loss for deterministic training
# ----------------------------------------------------------------------
def make_det_loss():
    d = len(cfg.TARGETS)
    if cfg.UNCERTAINTY_MODE == "none":
        return mse_loss
    if cfg.UNCERTAINTY_MODE == "diag":
        return lambda o, t: gaussian_nll_diag(o, t, d)
    if cfg.UNCERTAINTY_MODE == "full":
        return lambda o, t: gaussian_nll_full(o, t, d)
    raise ValueError("bad UNCERTAINTY_MODE")


# ----------------------------------------------------------------------
# Split raw network output into μ and σ
# ----------------------------------------------------------------------
def split_output(out: torch.Tensor):
    d = len(cfg.TARGETS)
    mode = cfg.UNCERTAINTY_MODE.lower()

    if mode == "none":
        return out, None

    if mode == "diag":
        mu, raw = out[:, :d], out[:, d:]
        log_var = smooth_logvar(raw)  # <- no clamp!
        std = torch.exp(0.5 * log_var)
        return mu, std

    # -------- full ---------------------------------------------------
    mu, params = out[:, :d], out[:, d:]
    raw_log_var = params[:, :d]  # d components
    rho_raw = params[:, d:]  # remaining

    sigma = torch.exp(0.5 * smooth_logvar(raw_log_var))
    cov = build_cov_from_params(sigma, rho_raw)

    std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    return mu, std  # keep interface unchanged (mu, std)


# ----------------------------------------------------------------------
# Training / evaluation helpers
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimiser=None, svi=None, loss_fn=None):
    model.train()
    total = 0.0
    if svi is not None:  # Bayesian
        for xb, yb in loader:
            total += svi.step(xb.to(cfg.DEVICE), yb.to(cfg.DEVICE))
        return total / len(loader.dataset)

    # deterministic
    for xb, yb in loader:
        xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
        loss = loss_fn(model(xb), yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.GRAD_CLIP_NORM and cfg.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        optimiser.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, meta, svi=None, run_dir=cfg.ARTIFACT_DIR):
    y_scaler = meta["y_scaler"]
    tf_info = meta["tf_info"]  # << NEW
    model.eval()
    bayes = svi is not None

    mu_ls, ale_ls, epi_ls = [], [], []
    for xb, _ in loader:
        xb = xb.to(cfg.DEVICE)
        # ------- forward / MC sampling (unchanged) ------------------
        if bayes:
            pred = pyro.infer.Predictive(model, guide=model.guide,
                                         num_samples=cfg.BAYES_NUM_SAMPLES,
                                         return_sites=["_RETURN"])(xb)
            outs = pred["_RETURN"].transpose(0, 1)  # (B,S,H)
            mus, vars_ = [], []
            for s in range(outs.shape[1]):
                mu_s, std_s = split_output(outs[:, s, :])
                mus.append(mu_s)
                if std_s is not None:
                    vars_.append(std_s.pow(2))
            mus = torch.stack(mus, 0)  # (S,B,D)
            mu_m = mus.mean(0)
            std_e = mus.var(0, unbiased=False).sqrt()
            std_a = torch.stack(vars_, 0).mean(0).sqrt() if vars_ else None
        else:
            out = model(xb)
            mu_m, std_a = split_output(out)
            std_e = None

        # ------- undo z-scaling -------------------------------------
        mu_scaled = (y_scaler.inverse_transform(mu_m.cpu().numpy())
                     if cfg.USE_ZSCORE_SCALING else mu_m.cpu().numpy())

        # ------- inverse scientific transform -----------------------
        mu_phys = inverse_phys_tf(mu_scaled, tf_info)
        mu_ls.append(mu_phys)

        if std_a is not None:
            std_a_np = (y_scaler.scale_ * std_a.cpu().numpy()
                        if cfg.USE_ZSCORE_SCALING else std_a.cpu().numpy())
            std_a_np = propagate_std(std_a_np, mu_scaled, tf_info)
            ale_ls.append(std_a_np)

        if std_e is not None:
            std_e_np = (y_scaler.scale_ * std_e.cpu().numpy()
                        if cfg.USE_ZSCORE_SCALING else std_e.cpu().numpy())
            std_e_np = propagate_std(std_e_np, mu_scaled, tf_info)
            epi_ls.append(std_e_np)

    mu = np.concatenate(mu_ls, 0)
    std_ale = np.concatenate(ale_ls, 0) if ale_ls else None
    std_epi = np.concatenate(epi_ls, 0) if epi_ls else None

    # Dump the outputs to a csv file (along with input features and true targets)
    X_all = meta["x_test_raw"]  # shape (N, 7)

    in_cols = list(cfg.FEATURES) + [f"{t}_HS" for t in cfg.TARGETS]
    true_cols = [f"true_{t}" for t in cfg.TARGETS]
    pred_cols = [f"pred_{t}" for t in cfg.TARGETS]

    df = pd.DataFrame(
        np.hstack([X_all, meta["y_test_raw"], mu]),
        columns=in_cols + true_cols + pred_cols
    )

    # Optional uncertainties
    if std_ale is not None:
        df = pd.concat(
            [df,
             pd.DataFrame(std_ale, columns=[f"std_ale_{t}" for t in cfg.TARGETS])],
            axis=1)

    if std_epi is not None:
        df = pd.concat(
            [df,
             pd.DataFrame(std_epi, columns=[f"std_epi_{t}" for t in cfg.TARGETS])],
            axis=1)

    preds_csv = run_dir / "predictions.csv"
    df.to_csv(preds_csv, index=False)
    print(f"[INFO] saved predictions → {preds_csv}")

    return mu, std_ale, std_epi


def loss_on_loader(model, loader, loss_fn=None, svi=None):
    model.eval()
    total = 0.0
    if svi is not None:
        for xb, yb in loader:
            total += svi.evaluate_loss(xb.to(cfg.DEVICE), yb.to(cfg.DEVICE))
        return total / len(loader.dataset)

    with torch.no_grad():
        for xb, yb in loader:
            total += loss_fn(model(xb.to(cfg.DEVICE)), yb.to(cfg.DEVICE)).item() * len(xb)
    return total / len(loader.dataset)


def _is_better(new, best, min_delta):
    """Return True if `new` is better than `best` by at least `min_delta` (relative)."""
    return (best - new) / max(abs(best), 1e-12) > min_delta


# --- helper: turn non-serialisable objects into strings -------------
def _serialisable(x):
    from pathlib import Path
    import torch
    import numpy as np

    if isinstance(x, (Path, torch.device)):
        return str(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()          # convert numpy scalar → Python scalar
    return x

# ----------------------------------------------------------------------
def main():
    # 0)  run dir -----------------------------------------------------
    run_dir = cfg.ARTIFACT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    print(f"[INFO] artifacts → {run_dir}")

    # Save config used for this run
    cfg_dict = {k: _serialisable(getattr(cfg, k))
                for k in dir(cfg) if k.isupper()}
    with (run_dir / "config_used.yaml").open("w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    print(f"[INFO] saved config   → {run_dir / 'config_used.yaml'}")


    # 1)  data --------------------------------------------------------
    train_loader, val_loader, test_loader, meta = load_dataset(
        cfg.DATA_FILE, cfg.FEATURES, cfg.TARGETS,
        cfg.TRAIN_FRACTION, cfg.VAL_FRACTION, cfg.DEVICE)

    model = HybridNet(next(iter(train_loader))[0].shape[1]).to(cfg.DEVICE)
    is_bnn = getattr(model, "has_bayes", False)

    det_loss = make_det_loss()
    optimiser = optim.Adam(model.parameters(), lr=cfg.LR) if not is_bnn else None
    svi = None

    # ------ early-stopping bookkeeping ------------------------------
    best_val = 99999999
    best_state = None
    best_epoch = 0

    tr_hist, va_hist = [], []

    # 2)  training ----------------------------------------------------
    for epoch in range(1, cfg.EPOCHS + 1):
        # ---- training step -----------------------------------------
        if is_bnn:
            svi = make_svi(model, epoch)
            tr_loss = train_one_epoch(model, train_loader, svi=svi)
        else:
            tr_loss = train_one_epoch(model, train_loader,
                                      optimiser=optimiser, loss_fn=det_loss)

        # ---- validation -------------------------------------------
        va_loss = loss_on_loader(model, val_loader, det_loss, svi)

        tr_hist.append(tr_loss)
        va_hist.append(va_loss)

        print(f"epoch {epoch:3d}/{cfg.EPOCHS} | train {tr_loss:.4e} | val {va_loss:.4e}")

        # ---- early–stopping logic ---------------------------------
        if cfg.EARLY_STOP_PATIENCE > 0:
            if _is_better(va_loss, best_val, cfg.EARLY_STOP_MIN_DELTA):
                best_val = va_loss
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
            elif epoch - best_epoch >= cfg.EARLY_STOP_PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch} "
                      f"(no val improvement for {cfg.EARLY_STOP_PATIENCE} epochs).")
                break

    # restore best weights (if early stopping used)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Loaded best model from epoch {best_epoch} (val loss {best_val:.4e})")

    # 3)  evaluation --------------------------------------------------
    mu, std_ale, std_epi = evaluate(model, test_loader, meta, svi, run_dir)

    # 4)  store & plot -----------------------------------------------
    epochs_done = len(tr_hist)
    pd.DataFrame({"epoch": np.arange(1, epochs_done + 1),
                  "train": tr_hist,
                  "val": va_hist}).to_csv(run_dir / "loss.csv", index=False)

    plot_loss(tr_hist, va_hist, run_dir)
    plot_results(meta["y_test_raw"], mu, std_ale, std_epi,
                 meta["low_res_test_raw"], cfg.TARGETS,
                 run_dir / "pred_plot.png")

    torch.save(model.state_dict(), run_dir / "model.pth")
    torch.save(meta, run_dir / "data_meta.pt")
    if svi is not None:
        pyro.get_param_store().save(run_dir / "pyro_params.pt")

    print("[INFO] training complete")
    return run_dir


if __name__ == "__main__":
    main()
