# training/data_utils.py
import os, random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from scipy.special import logit, expit

import training.config as cfg

# ──────────────────────────────────────────────────────────────────────
# 0)  Global seeding helper
# ──────────────────────────────────────────────────────────────────────
def _set_global_seed(seed: int):
    """Seed Python, NumPy, Torch (CPU & CUDA) and CuDNN for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python ≥3.3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # deterministic CuDNN (optional but often helpful)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if cfg.SEED is not None:
    _set_global_seed(cfg.SEED)


def _apply_target_forward(df_targets):
    """Return transformed ndarray and per-target metadata."""
    Y = df_targets.copy()
    info = {}
    for col in df_targets.columns:
        tf = cfg.TARGET_TF.get(col, "none")
        if tf == "log":
            Y[col] = np.log(df_targets[col] + cfg.TF_EPS)
        elif tf == "logit":
            Y[col] = logit(np.clip(df_targets[col], cfg.TF_EPS, 1 - cfg.TF_EPS))
        info[col] = tf
    return Y.values, info


# ──────────────────────────────────────────────────────────────────────
# helper: dummy scaler when Z-score is disabled
# ──────────────────────────────────────────────────────────────────────
class IdentityScaler:
    def fit(self, x):               return self

    def transform(self, x):         return x

    def fit_transform(self, x):     return x

    def inverse_transform(self, x): return x

    @property
    def scale_(self):               return np.ones(self._dim)

    @scale_.setter
    def scale_(self, v):            self._dim = len(v)


# ──────────────────────────────────────────────────────────────────────
# feature helpers
# ──────────────────────────────────────────────────────────────────────
def _extract_imaginary(col: pd.Series) -> np.ndarray:
    def one(x):
        if pd.isna(x):             return np.nan
        if isinstance(x, complex): return x.imag
        try:
            return complex(x).imag
        except Exception:
            return np.nan

    return col.apply(one).to_numpy(dtype=float)


def _scale_and_zero_missing(arr: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    mask = np.isnan(arr)
    filled = np.where(mask, np.nanmean(arr, axis=0), arr)
    scaler = StandardScaler().fit(filled)
    scaled = scaler.transform(filled)
    scaled[mask] = 0.0
    return scaled, scaler


# ──────────────────────────────────────────────────────────────────────
# scientific-transform helpers
# ──────────────────────────────────────────────────────────────────────
BOUNDED_TARGETS = {"SSA", "g"}
LOG_TARGETS_POS = {"Qext"}


def _forward_tf(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = df.copy()
    eps = 1e-6
    for col in y.columns:
        if col in BOUNDED_TARGETS:
            y[col] = logit(np.clip(y[col], eps, 1 - eps))
        elif col in LOG_TARGETS_POS:
            y[col] = np.log(y[col] + eps)
    return y.values, {"eps": eps}


# ──────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ──────────────────────────────────────────────────────────────────────
def _seed_worker(worker_id: int):
    """
    Ensures each DataLoader worker has a deterministic and unique seed.
    """
    worker_seed = (cfg.SEED + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# shared generator for all DataLoaders
_dl_generator = torch.Generator()
_dl_generator.manual_seed(cfg.SEED if cfg.SEED is not None else 0)


def _make_loader(x, y, batch_size=cfg.BATCH_SIZE, shuffle=False):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=0,  # set >0 if you need workers
                      worker_init_fn=_seed_worker if shuffle else None,
                      generator=_dl_generator)


def load_dataset(path: Path,
                 features: List[str],
                 targets: List[str],
                 train_fraction: float,
                 val_fraction: float = 0.1,
                 device=cfg.DEVICE):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # engineered feature ------------------------------------------------
    if {"Dve", "wavelength"}.issubset(df.columns):
        df["Xve"] = np.pi * df["Dve"] / df["wavelength"]

    # ------------------ build X ---------------------------------------
    cols = []
    for f in features:
        if f == "coating_RI_imag":
            cols.append(_extract_imaginary(df["coating_RI"]))
        else:
            cols.append(df[f].to_numpy(dtype=float))

    low_res_cols = [df[f"{t}_HS"].to_numpy(dtype=float) for t in targets]
    cols.extend(low_res_cols)

    X_raw = np.column_stack(cols)
    low_res_raw = X_raw[:, -len(targets):]

    # ------------------ targets ---------------------------------------
    y_raw_np, tf_info = _apply_target_forward(df[targets])

    # ------------------ scaling ---------------------------------------
    if cfg.USE_ZSCORE_SCALING:
        X_scaled, x_scaler = _scale_and_zero_missing(X_raw)
        y_scaled, y_scaler = _scale_and_zero_missing(y_raw_np)
    else:
        x_scaler = IdentityScaler();
        x_scaler.scale_ = np.ones(X_raw.shape[1])
        y_scaler = IdentityScaler();
        y_scaler.scale_ = np.ones(y_raw_np.shape[1])
        X_scaled, y_scaled = X_raw.copy(), y_raw_np.copy()

    # ------------------ split -----------------------------------------
    n = len(X_scaled)
    n_train = int(train_fraction * n)
    n_val = int(val_fraction * n)

    idx = np.random.permutation(n)
    idx_tr = idx[:n_train]
    idx_va = idx[n_train: n_train + n_val]
    idx_te = idx[n_train + n_val:]

    X_tr, y_tr = X_scaled[idx_tr], y_scaled[idx_tr]
    X_va, y_va = X_scaled[idx_va], y_scaled[idx_va]
    X_te, y_te = X_scaled[idx_te], y_scaled[idx_te]

    # -----------------------------------------------------------------
    # Balanced sampler (optional, can handle multiple targets)
    # -----------------------------------------------------------------
    sampler = None
    shuffle_flag = True

    if cfg.BALANCE_TARGETS:
        weights = np.ones(len(y_tr), dtype=np.float32)

        for tgt in cfg.BALANCE_TARGETS:
            if tgt not in targets:
                raise ValueError(f"{tgt} not among TARGETS {targets}")

            col_idx = targets.index(tgt)
            vals = y_tr[:, col_idx]  # transformed space
            bins = np.linspace(vals.min(), vals.max(), cfg.N_BINS + 1)
            bin_idx = np.digitize(vals, bins) - 1
            freq = np.bincount(bin_idx, minlength=cfg.N_BINS) + 1e-8
            inv_f = 1.0 / freq[bin_idx]

            if cfg.BALANCE_STRATEGY == "prod":
                weights *= inv_f
            elif cfg.BALANCE_STRATEGY == "mean":
                weights += inv_f
            elif cfg.BALANCE_STRATEGY == "max":
                weights = np.maximum(weights, inv_f)
            else:
                raise ValueError("cfg.BALANCE_STRATEGY must be 'prod', 'mean', or 'max'")

        # clip extreme weights
        weights = np.clip(weights, 0, 10 * weights.mean())

        if cfg.BALANCE_STRATEGY == "mean":
            weights /= len(cfg.BALANCE_TARGETS)

        weights /= weights.mean()  # normalise
        sampler = WeightedRandomSampler(weights,
                                        num_samples=len(weights),
                                        replacement=True)
        shuffle_flag = False

    # ------------------ DataLoaders -----------------------------------
    def _make_loader(x, y, shuffle=False, sampler=None):
        ds = TensorDataset(torch.tensor(x, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.float32))
        return DataLoader(
            ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=0,
            generator=_dl_generator,
            worker_init_fn=_seed_worker if sampler is None else None,
        )

    train_loader = _make_loader(X_tr, y_tr, shuffle=shuffle_flag, sampler=sampler)
    val_loader = _make_loader(X_va, y_va)
    test_loader = _make_loader(X_te, y_te)

    # --- move tensors to device --------------------------------------
    if device != torch.device("cpu"):
        for dl in (train_loader, val_loader, test_loader):
            dl.dataset.tensors = tuple(t.to(device) for t in dl.dataset.tensors)

    meta = {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "tf_info": tf_info,
        "y_test_raw": df[cfg.TARGETS].values[idx_te],
        "low_res_test_raw": low_res_raw[idx_te],
        "x_test_raw": X_raw[idx_te],

    }

    return train_loader, val_loader, test_loader, meta
