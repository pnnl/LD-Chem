"""
training.inference_api  –  run-time predictions with a trained HybridNet
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Union

import numpy as np
import yaml
import torch
import pyro

import training.config as cfg
from training.model import HybridNet
from training.train import split_output, inverse_phys_tf, propagate_std

# ------------------------------------------------------------------ #
# Run-directory resolution                                           #
# ------------------------------------------------------------------ #
def _resolve_run_dir(model_dir: Union[str, Path]) -> Path:
    """
    Accepts one of
      • absolute / relative path to a run dir
      • bare timestamp (e.g. 20250711_141533)
      • path relative to repo root
      • path to artifacts root      → selects most recent run

    Returns Path whose sub-file 'model.pth' exists.
    """
    p_in = Path(model_dir).expanduser()
    root = cfg.ROOT_DIR

    # Candidate paths to test (in order)
    candidates = [
        p_in,
        root / p_in,                      # relative to repo root
        cfg.ARTIFACT_DIR / p_in.name      # bare timestamp
    ]

    for cand in candidates:
        cand = cand.resolve()
        if (cand / "model.pth").exists():
            return cand

    # treat argument as artifacts root and pick newest run
    if p_in.is_dir():
        runs = [d for d in p_in.iterdir() if (d / "model.pth").exists()]
        if runs:
            runs.sort()
            return runs[-1]

    raise FileNotFoundError(f"Could not locate a run directory from {model_dir}")


# ------------------------------------------------------------------ #
# Inference                                                          #
# ------------------------------------------------------------------ #
def run_inference(model_dir: str | Path,
                  x_raw: np.ndarray | torch.Tensor,
                  device: torch.device = cfg.DEVICE,
                  num_mc: int | None = None):
    """
    Returns
    -------
    mu_phys : (N,D) predictive means in physical space
    std_ale : (N,D) aleatoric std (or None)
    std_epi : (N,D) epistemic  std (or None)
    """
    # ------------------------------------------------------------------
    # reproducibility
    # ------------------------------------------------------------------
    if cfg.SEED is not None:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        pyro.set_rng_seed(cfg.SEED)

    run_dir = _resolve_run_dir(model_dir)
    ckpt_path  = run_dir / "model.pth"
    meta_path  = run_dir / "data_meta.pt"
    pyro_path  = run_dir / "pyro_params.pt"
    cfg_path   = run_dir / "config_used.yaml"

    # ------------------------------------------------------------------
    # load *saved* configuration so architecture matches
    # ------------------------------------------------------------------
    if cfg_path.exists():
        with cfg_path.open("r") as f:
            saved_cfg = yaml.safe_load(f) or {}
        for k, v in saved_cfg.items():
            if k.isupper():
                setattr(cfg, k, v)

    num_mc = int(num_mc or cfg.BAYES_NUM_SAMPLES)

    # ------------------------------------------------------------------
    # meta & scalers
    # ------------------------------------------------------------------
    meta     = torch.load(meta_path, map_location="cpu", weights_only=False)
    x_scaler = meta["x_scaler"]
    y_scaler = meta["y_scaler"]
    tf_info  = meta["tf_info"]

    n_expected = int(getattr(x_scaler, "n_features_in_", None)
                     or x_scaler.mean_.shape[0])

    # ------------------------------------------------------------------
    # prepare input
    # ------------------------------------------------------------------
    if isinstance(x_raw, torch.Tensor):
        x_np = x_raw.detach().cpu().numpy().astype(np.float32)
    else:
        x_np = np.asarray(x_raw, dtype=np.float32)

    if x_np.ndim != 2 or x_np.shape[1] != n_expected:
        raise ValueError(f"Input must have shape (N,{n_expected}), got {x_np.shape}")

    if cfg.USE_ZSCORE_SCALING:
        x_np = x_scaler.transform(x_np)

    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # rebuild model / load weights
    # ------------------------------------------------------------------
    model = HybridNet(x.shape[1]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False),
                          strict=False)
    model.eval()

    is_bayes = getattr(model, "has_bayes", False)
    if is_bayes and pyro_path.exists():
        pyro_state = torch.load(pyro_path, map_location=device, weights_only=False)
        pyro.get_param_store().set_state(pyro_state)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    with torch.no_grad():
        if is_bayes:
            pred = pyro.infer.Predictive(
                model, guide=model.guide,
                num_samples=num_mc, return_sites=["_RETURN"])(x)
            outs = pred["_RETURN"].transpose(0, 1)        # (N,S,H)

            mus, vars_ = [], []
            for s in range(outs.shape[1]):
                mu_s, std_s = split_output(outs[:, s, :])
                mus.append(mu_s)
                if std_s is not None:
                    vars_.append(std_s ** 2)

            mus = torch.stack(mus, 0)                     # (S,N,D)
            mu_t       = mus.mean(0)
            std_epi_t  = mus.var(0, unbiased=False).sqrt()
            std_ale_t  = torch.stack(vars_, 0).mean(0).sqrt() if vars_ else None
        else:
            out        = model(x)
            mu_t, std_ale_t = split_output(out)
            std_epi_t  = None

    # ------------------------------------------------------------------
    # inverse transforms
    # ------------------------------------------------------------------
    mu_scaled = (y_scaler.inverse_transform(mu_t.cpu().numpy())
                 if cfg.USE_ZSCORE_SCALING else mu_t.cpu().numpy())

    def _post(std_t):
        if std_t is None:
            return None
        std_np = (y_scaler.scale_ * std_t.cpu().numpy()
                  if cfg.USE_ZSCORE_SCALING else std_t.cpu().numpy())
        return propagate_std(std_np, mu_scaled, tf_info)

    std_ale_np = _post(std_ale_t)
    std_epi_np = _post(std_epi_t)

    mu_phys = inverse_phys_tf(mu_scaled, tf_info)

    return mu_phys, std_ale_np, std_epi_np


# ---------------------------------------------------------------------------#
# Example usage
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    dummy = torch.rand(128, len(cfg.FEATURES + cfg.TARGETS))
    mu, s_ale, s_epi = run_inference("artifacts", dummy, num_mc=50)
    print(mu.shape, None if s_ale is None else s_ale.shape)