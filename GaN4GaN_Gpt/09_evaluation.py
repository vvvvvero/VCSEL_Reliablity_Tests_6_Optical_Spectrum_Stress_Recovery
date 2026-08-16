"""
09_evaluation.py
================
Evaluation, ablation studies, and visualisation for the PI-TimeGAN model.

Implements the 6-layer physics validation protocol described in the design doc:
  1. Structural validation    (bounds, monotonicity, temperature ordering)
  2. Observation anchoring    (decoder sensitivity analysis)
  3. Counterfactual intervention
  4. Temperature response verification
  5. Predictive ablation study
  6. Identifiability & stability (multi-seed statistics)

Additionally provides:
  - Long-horizon RMSE metrics at specified prediction horizons
  - Failure-time prediction (time to cross a degradation threshold)
  - Temperature-group breakdown of errors
  - Visualisation: latent trajectories, feature reconstructions, heatmaps
"""

import os
import sys
import logging
import pickle
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: load model from checkpoint
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, model_class) -> nn.Module:
    """Load a model from a state-dict checkpoint file."""
    model = model_class()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        model_state = dict(state["model_state"])
        # Keep decoder sparsity from current config instead of checkpoint buffer.
        model_state.pop("decoder.mask", None)
        model.load_state_dict(model_state, strict=False)
    else:
        model_state = dict(state)
        model_state.pop("decoder.mask", None)
        model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_from_prefix(
    model,
    enc_input:  torch.Tensor,   # (B, T, 15)
    x_true:     torch.Tensor,   # (B, T, 6)
    mask:       torch.Tensor,   # (B, T)
    times_h:    torch.Tensor,   # (B, T)
    T_K:        torch.Tensor,   # (B,)
    x0:         torch.Tensor,   # (B, 6)
    prefix_len: int,
    device:     torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Encode the first `prefix_len` time steps, then integrate the ODE
    forward to all remaining steps and decode.

    Returns dict with:
      'z_enc'   : (B, prefix_len, 5)  encoder latent states for prefix
      'z_ode'   : (B, T, 5)           full ODE trajectory from t[prefix_len-1]
      'x_pred'  : (B, T, 6)           decoded predictions
      'alpha'   : (B,)
    """
    model.eval()
    with torch.no_grad():
        enc_input = enc_input.to(device)
        x_true    = x_true.to(device)
        mask      = mask.to(device)
        times_h   = times_h.to(device)
        T_K       = T_K.to(device)
        x0        = x0.to(device)

        # Encode prefix
        z_prefix, _ = model.encoder(enc_input[:, :prefix_len, :],
                                    mask[:, :prefix_len])      # (B, prefix_len, 5)
        alpha       = model.alpha_net(x0, T_K)                  # (B,)

        # Initial state for ODE: last valid encoder state in prefix
        z0_ode = z_prefix[:, -1, :]   # (B, 5)

        # Integrate ODE from t[prefix_len-1] to all future steps
        # Build time array starting from the prefix end
        t_start = times_h[:, prefix_len - 1]   # (B,)
        t_future = times_h[:, prefix_len - 1:]  # (B, T - prefix_len + 1)

        z_ode_future = model.ode.integrate_trajectory(
            z0_ode, T_K, t_future, alpha)   # (B, T-prefix+1, 5)

        # Assemble full trajectory: encoder prefix + ODE future
        if prefix_len > 1:
            z_ode_full = torch.cat([
                z_prefix[:, :-1, :],    # (B, prefix_len-1, 5)
                z_ode_future,            # (B, T-prefix+1, 5)
            ], dim=1)                   # (B, T, 5)
        else:
            z_ode_full = z_ode_future   # (B, T, 5)

        z_ref = z_prefix[:, 0, :]
        x_pred = model.decoder(z_ode_full, z_ref=z_ref)   # (B, T, 6)

    return {
        "z_enc":   z_prefix,
        "z_ode":   z_ode_full,
        "x_pred":  x_pred,
        "alpha":   alpha,
    }


# ---------------------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------------------

def compute_rmse(
    x_pred: np.ndarray,
    x_true: np.ndarray,
    mask:   np.ndarray,
    per_feature: bool = True,
) -> Dict:
    """
    Compute RMSE between predictions and ground truth.

    Args:
        x_pred : (N, T, 6)
        x_true : (N, T, 6)  NaN where missing
        mask   : (N, T)

    Returns:
        dict with 'overall' and per-feature RMSE values
    """
    valid_3d = mask[:, :, None] & ~np.isnan(x_true)  # (N, T, 6)
    diff = np.where(valid_3d, (x_pred - np.nan_to_num(x_true, nan=0.0)) ** 2, 0.0)

    results = {}
    results["overall"] = np.sqrt(diff.sum() / max(valid_3d.sum(), 1))

    if per_feature:
        for fi, fname in enumerate(cfg.FEATURES):
            d = diff[:, :, fi].sum()
            n = valid_3d[:, :, fi].sum()
            results[fname] = np.sqrt(d / max(n, 1))

    return results


def compute_mae(
    x_pred: np.ndarray,
    x_true: np.ndarray,
    mask:   np.ndarray,
    per_feature: bool = True,
) -> Dict:
    """Compute MAE between predictions and ground truth."""
    valid_3d = mask[:, :, None] & ~np.isnan(x_true)
    abs_err = np.where(valid_3d, np.abs(x_pred - np.nan_to_num(x_true, nan=0.0)), 0.0)

    results = {}
    results["overall"] = abs_err.sum() / max(valid_3d.sum(), 1)

    if per_feature:
        for fi, fname in enumerate(cfg.FEATURES):
            d = abs_err[:, :, fi].sum()
            n = valid_3d[:, :, fi].sum()
            results[fname] = d / max(n, 1)

    return results


def compute_valid_point_count(x_true: np.ndarray, mask: np.ndarray) -> Dict[str, int]:
    """Count valid evaluation points overall and per feature."""
    valid_3d = mask[:, :, None] & ~np.isnan(x_true)
    out = {"overall": int(valid_3d.sum())}
    for fi, fname in enumerate(cfg.FEATURES):
        out[fname] = int(valid_3d[:, :, fi].sum())
    return out


def compute_multiplicative_error_factor(rmse_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Approximate multiplicative error factor exp(RMSE) for log-ratio features.
    Not meaningful for Vth (x1), so returns NaN there.
    """
    out = {}
    for k, v in rmse_dict.items():
        if k == "Vth":
            out[k] = float("nan")
            continue
        out[k] = float(np.exp(v))
    return out


def make_future_mask(mask: np.ndarray, prefix_len: int) -> np.ndarray:
    """Keep only observations strictly after the encoder prefix."""
    future_mask = mask.copy().astype(bool)
    future_mask[:, :prefix_len] = False
    return future_mask


def make_prefix_mask(mask: np.ndarray, prefix_len: int) -> np.ndarray:
    """Keep only points inside the encoder prefix window."""
    prefix_mask = mask.copy().astype(bool)
    prefix_mask[:, prefix_len:] = False
    return prefix_mask


def compute_rmse_by_temperature(
    x_pred: np.ndarray,
    x_true: np.ndarray,
    mask: np.ndarray,
    T_K: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute RMSE grouped by storage temperature."""
    results = {}
    for tc in cfg.TEMPERATURES_C:
        tk = tc + cfg.CELSIUS_TO_KELVIN
        sel = np.abs(T_K - tk) < 1.0
        if sel.sum() < 1:
            continue
        results[f"{tc}C"] = compute_rmse(x_pred[sel], x_true[sel], mask[sel])
    return results


def compute_rmse_by_time_interval(
    x_pred: np.ndarray,
    x_true: np.ndarray,
    mask: np.ndarray,
    times_h: np.ndarray,
    intervals: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute RMSE grouped by time intervals (hours)."""
    if intervals is None:
        intervals = [(0.0, 100.0), (100.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0)]

    results = {}
    for lo, hi in intervals:
        tmask = (times_h >= lo) & (times_h < hi)
        m = mask & tmask
        key = f"[{int(lo)},{int(hi)})h"
        if m.sum() < 1:
            results[key] = {"overall": float("nan")}
            continue
        results[key] = compute_rmse(x_pred, x_true, m)
    return results


def compute_rmse_by_device(
    x_pred: np.ndarray,
    x_true: np.ndarray,
    mask: np.ndarray,
    device_ids: List[str],
) -> Dict[str, float]:
    """Compute overall RMSE per device."""
    out = {}
    for i, did in enumerate(device_ids):
        r = compute_rmse(x_pred[i:i+1], x_true[i:i+1], mask[i:i+1], per_feature=False)
        out[did] = float(r["overall"])
    return out


def denormalize_x(x_norm: np.ndarray, norm_stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Map normalized transformed features back to transformed feature scale."""
    x = x_norm.copy().astype(float)
    for fi, fname in enumerate(cfg.FEATURES):
        lo = norm_stats[fname]["min"]
        hi = norm_stats[fname]["max"]
        x[:, :, fi] = x[:, :, fi] * (hi - lo) + lo
    return x


def check_zero_time_features(x_deg: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Check whether transformed degradation features are near zero at t=0.
    Returns per-feature mean absolute value at t=0 on valid rows.
    """
    out = {}
    m0 = mask[:, 0]
    for fi, fname in enumerate(cfg.FEATURES):
        valid = m0 & ~np.isnan(x_deg[:, 0, fi])
        if valid.sum() == 0:
            out[fname] = float("nan")
            continue
        out[fname] = float(np.nanmean(np.abs(x_deg[valid, 0, fi])))
    return out


def compute_nrmse_r2(x_pred: np.ndarray, x_true: np.ndarray, mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute per-feature NRMSE and R^2 in transformed space."""
    metrics = {"nrmse": {}, "r2": {}}
    valid_3d = mask[:, :, None] & ~np.isnan(x_true)

    for fi, fname in enumerate(cfg.FEATURES):
        valid = valid_3d[:, :, fi]
        y = x_true[:, :, fi][valid]
        yhat = x_pred[:, :, fi][valid]
        if y.size < 2:
            metrics["nrmse"][fname] = float("nan")
            metrics["r2"][fname] = float("nan")
            continue
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        y_std = float(np.std(y))
        metrics["nrmse"][fname] = rmse / max(y_std, 1e-12)

        ss_res = float(np.sum((yhat - y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        metrics["r2"][fname] = 1.0 - ss_res / max(ss_tot, 1e-12)

    return metrics


def _fit_logtime_linear_baseline(x_train: np.ndarray, mask_train: np.ndarray, times_train: np.ndarray, T_train: np.ndarray) -> np.ndarray:
    """Fit per-feature linear baseline: x = a + b*log(1+t) + c/T."""
    coeffs = np.zeros((cfg.FEATURE_DIM, 3), dtype=float)
    for fi in range(cfg.FEATURE_DIM):
        y_all = []
        X_all = []
        for i in range(x_train.shape[0]):
            for j in range(x_train.shape[1]):
                if not mask_train[i, j]:
                    continue
                y = x_train[i, j, fi]
                if np.isnan(y):
                    continue
                X_all.append([1.0, np.log1p(times_train[i, j]), 1.0 / max(T_train[i], 1.0)])
                y_all.append(y)
        if len(y_all) < 3:
            coeffs[fi] = np.array([0.0, 0.0, 0.0])
            continue
        X = np.asarray(X_all, dtype=float)
        y = np.asarray(y_all, dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        coeffs[fi] = beta
    return coeffs


def _predict_logtime_linear(coeffs: np.ndarray, times: np.ndarray, T_K: np.ndarray) -> np.ndarray:
    """Predict transformed features with fitted log-time linear baseline."""
    N, T = times.shape
    pred = np.zeros((N, T, cfg.FEATURE_DIM), dtype=float)
    logt = np.log1p(times)
    invT = 1.0 / np.maximum(T_K[:, None], 1.0)
    for fi in range(cfg.FEATURE_DIM):
        a, b, c = coeffs[fi]
        pred[:, :, fi] = a + b * logt + c * invT
    return pred


def _build_prefix_persistence_baseline(
    x_true: np.ndarray,
    mask: np.ndarray,
    prefix_len: int,
) -> np.ndarray:
    """
    Prefix-persistence baseline for open-loop forecasting.

    Uses only the prefix segment to estimate one last value per feature, then
    holds it constant for all future steps.
    """
    pred = np.zeros_like(x_true, dtype=float)
    N, T, F = x_true.shape
    for i in range(N):
        for fi in range(F):
            last = 0.0
            for j in range(min(prefix_len, T)):
                if mask[i, j] and not np.isnan(x_true[i, j, fi]):
                    last = float(x_true[i, j, fi])
            pred[i, :, fi] = last
    return pred


def compute_skill_scores(x_model: np.ndarray, x_true: np.ndarray, mask: np.ndarray, baselines: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Skill = 1 - MSE_model / MSE_baseline (overall + per feature)."""
    valid_3d = mask[:, :, None] & ~np.isnan(x_true)
    mse_model_overall = float(np.mean((x_model[valid_3d] - x_true[valid_3d]) ** 2)) if valid_3d.any() else float("nan")

    out = {}
    for bname, x_base in baselines.items():
        m = {"overall": float("nan")}
        mse_b_overall = float(np.mean((x_base[valid_3d] - x_true[valid_3d]) ** 2)) if valid_3d.any() else float("nan")
        if np.isfinite(mse_b_overall) and mse_b_overall > 0:
            m["overall"] = 1.0 - mse_model_overall / mse_b_overall

        for fi, fname in enumerate(cfg.FEATURES):
            valid = valid_3d[:, :, fi]
            if valid.sum() < 1:
                m[fname] = float("nan")
                continue
            mse_m = float(np.mean((x_model[:, :, fi][valid] - x_true[:, :, fi][valid]) ** 2))
            mse_b = float(np.mean((x_base[:, :, fi][valid] - x_true[:, :, fi][valid]) ** 2))
            m[fname] = 1.0 - mse_m / mse_b if mse_b > 0 else float("nan")
        out[bname] = m
    return out


def compute_leakage_phase1_diagnostics(
    x_true_phys: np.ndarray,
    x_pred_phys: np.ndarray,
    x_base_phys: np.ndarray,
    mask: np.ndarray,
    leakage_floor_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Summarise leakage floor proximity and decreasing-transition behaviour."""
    out = {}
    for leak_name in ("IDLeak", "IGLeak"):
        fi = cfg.FEATURES.index(leak_name)
        floor = float((leakage_floor_map or {}).get(leak_name, cfg.LEAKAGE_FLOOR_DEFAULT)) if leakage_floor_map else cfg.LEAKAGE_FLOOR_DEFAULT
        valid = mask & ~np.isnan(x_true_phys[:, :, fi])
        true_vals = x_true_phys[:, :, fi][valid]
        pred_vals = x_pred_phys[:, :, fi][valid]
        base_vals = x_base_phys[:, :, fi][valid]

        def _frac_near_floor(vals: np.ndarray) -> float:
            if vals.size == 0:
                return float("nan")
            return float(np.mean(np.abs(vals) <= (abs(floor) + cfg.LEAKAGE_CONFIDENCE_MARGIN)))

        def _decreasing_fraction(vals: np.ndarray, m: np.ndarray) -> float:
            count = 0
            dec = 0
            for i in range(vals.shape[0]):
                for j in range(1, vals.shape[1]):
                    if not m[i, j] or not m[i, j - 1]:
                        continue
                    if np.isnan(vals[i, j]) or np.isnan(vals[i, j - 1]):
                        continue
                    count += 1
                    if vals[i, j] < vals[i, j - 1] - 1e-12:
                        dec += 1
            return float(dec / max(count, 1)) if count > 0 else float("nan")

        out[leak_name] = {
            "near_floor_fraction_true": _frac_near_floor(true_vals),
            "near_floor_fraction_pred": _frac_near_floor(pred_vals),
            "near_floor_fraction_base": _frac_near_floor(base_vals),
            "decreasing_fraction_true": _decreasing_fraction(x_true_phys[:, :, fi], mask),
            "decreasing_fraction_pred": _decreasing_fraction(x_pred_phys[:, :, fi], mask),
            "decreasing_fraction_base": _decreasing_fraction(x_base_phys[:, :, fi], mask),
        }
    return out


def summarize_decoder_leakage_routes(model) -> Dict[str, float]:
    """Report the relative strength of reversible leakage branches vs zL."""
    if not hasattr(model, "decoder"):
        return {}
    try:
        w = model.decoder.get_weight_matrix()
    except Exception:
        return {}
    if w.ndim != 2:
        return {}

    out = {}
    for leak_name, row_idx in (("IDLeak", cfg.IDLEAK_DECODER_ROW), ("IGLeak", cfg.IGLEAK_DECODER_ROW)):
        z_g = abs(float(w[row_idx, 0])) if w.shape[1] > 0 else 0.0
        z_b = abs(float(w[row_idx, 1])) if w.shape[1] > 1 else 0.0
        z_l = abs(float(w[row_idx, 3])) if w.shape[1] > 3 else 0.0
        denom = max(z_l, 1e-12)
        out[f"{leak_name}_zGzB_to_zL"] = (z_g + z_b) / denom
        out[f"{leak_name}_zG_to_zL"] = z_g / denom
        out[f"{leak_name}_zB_to_zL"] = z_b / denom
    return out


# ---------------------------------------------------------------------------
# Alpha identifiability (Step 3)
# ---------------------------------------------------------------------------

def check_alpha_identifiability(
    model,
    dataset: dict,
    device: torch.device,
    prefix_len: int = cfg.STAGE3_PREFIX_LEN,
    min_identifiable_corr: float = 0.20,
) -> Dict:
    """
    Checks whether alpha_net's output correlates with actual future degradation.

    For each training device:
      - Get alpha = alpha_net(x0, T_K)
      - Measure total future degradation = mean abs(x_future - x_prefix_end) over features

    If |rank_corr(alpha, future_degradation)| < min_identifiable_corr for all
    features → alpha is not identifiable → recommend fixing alpha=1.

    Returns:
        {
          'per_feature_corr': {feature: spearman_rho},
          'alpha_stats': {mean, std, min, max},
          'is_identifiable': bool,
          'recommendation': str,
        }
    """
    from scipy.stats import spearmanr
    from torch.utils.data import DataLoader

    train_idx = dataset["split"]["train"]

    class _DS(torch.utils.data.Dataset):
        def __init__(self, ds, idx):
            self.x   = torch.from_numpy(ds["x"]).float()
            self.feat = torch.from_numpy(ds["feature_mask"]).bool() if "feature_mask" in ds else None
            self.mask = torch.from_numpy(ds["mask"]).bool()
            self.t    = torch.from_numpy(ds["times_h"]).float()
            self.TK   = torch.from_numpy(ds["T_K"]).float()
            x0_key = "x0_normalized" if "x0_normalized" in ds else "x0_static"
            self.x0   = torch.from_numpy(ds[x0_key]).float()
            self.idx  = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            b = self.idx[i]
            x = self.x[b]; mask = self.mask[b]; t = self.t[b]; TK = self.TK[b]; x0 = self.x0[b]
            T_norm = torch.tensor((TK.item() - cfg.T_REF_K) / cfg.T_REF_K)
            T_f = T_norm.expand(x.shape[0], 1)
            lt = torch.log(t + 1.0); dl = torch.zeros_like(lt); dl[:-1] = lt[1:] - lt[:-1]
            enc = torch.cat([torch.nan_to_num(x, nan=0.0),
                             (x.isfinite()).float(), T_f, lt.unsqueeze(1), dl.unsqueeze(1)], dim=1)
            return {"enc_input": enc, "x": x, "mask": mask, "times_h": t, "T_K": TK, "x0": x0}

    ds   = _DS(dataset, train_idx)
    dl   = DataLoader(ds, batch_size=16, shuffle=False,
                      collate_fn=lambda b: {k: torch.stack([i[k] for i in b]) for k in b[0]})

    all_alpha, all_future_deg = [], []

    model.eval()
    with torch.no_grad():
        for batch in dl:
            x_true  = batch["x"].to(device)
            mask    = batch["mask"].to(device)
            times_h = batch["times_h"].to(device)
            T_K     = batch["T_K"].to(device)
            x0      = batch["x0"].to(device)
            enc_in  = batch["enc_input"].to(device)

            z_enc, _ = model.encoder(enc_in, mask)
            alpha_b  = model.alpha_net(x0, T_K).cpu().numpy()   # (B,)
            all_alpha.append(alpha_b)

            # Future degradation proxy: mean abs change per feature from prefix end
            B = x_true.shape[0]
            fx = x_true[:, prefix_len:, :].cpu().numpy()   # (B, T_fut, F)
            fm = mask[:, prefix_len:].cpu().numpy()
            x_at_pfx = x_true[:, prefix_len - 1:prefix_len, :].cpu().numpy()  # (B,1,F)
            abs_change = np.abs(fx - x_at_pfx)   # (B, T_fut, F)
            valid_3d = fm[:, :, None] & np.isfinite(abs_change)
            means = np.array([
                float(np.nanmean(abs_change[i][valid_3d[i]])) if valid_3d[i].any() else float("nan")
                for i in range(B)
            ])
            all_future_deg.append(means)

    alphas = np.concatenate(all_alpha, 0)
    future_deg = np.concatenate(all_future_deg, 0)

    # Overall Spearman correlation alpha ↔ mean future degradation
    valid = np.isfinite(alphas) & np.isfinite(future_deg)
    if valid.sum() < 5:
        corr_overall = float("nan")
    else:
        corr_overall, _ = spearmanr(alphas[valid], future_deg[valid])

    alpha_stats = {
        "mean": float(np.nanmean(alphas)),
        "std":  float(np.nanstd(alphas)),
        "min":  float(np.nanmin(alphas)),
        "max":  float(np.nanmax(alphas)),
    }

    is_identifiable = np.isfinite(corr_overall) and abs(corr_overall) >= min_identifiable_corr

    rec = ("identifiable — alpha correlates with future degradation"
           if is_identifiable
           else "NOT identifiable — recommend cfg.GENERATOR_FIXED_ALPHA=True (alpha=1)")

    log.info(
        "Alpha identifiability | Spearman(alpha, future_deg)=%.3f | "
        "stats: mean=%.4f std=%.5f | %s",
        corr_overall, alpha_stats["mean"], alpha_stats["std"], rec,
    )

    return {
        "spearman_corr_overall": corr_overall,
        "alpha_stats": alpha_stats,
        "is_identifiable": is_identifiable,
        "recommendation": rec,
    }


def inverse_transform_to_physical(x_deg: np.ndarray, x0_static: np.ndarray, s_vth: float) -> np.ndarray:
    """
    Convert transformed degradation features back to physical space.
    Output channels match cfg.FEATURES order.
    """
    eps = cfg.EPSILON
    x0 = x0_static[:, None, :]  # (N,1,6)
    out = np.zeros_like(x_deg, dtype=float)

    # Vth
    out[:, :, 0] = x0[:, :, 0] + s_vth * x_deg[:, :, 0]
    # IDSS
    out[:, :, 1] = (np.abs(x0[:, :, 1]) + eps) * np.exp(-x_deg[:, :, 1]) - eps
    # RON
    out[:, :, 2] = (np.abs(x0[:, :, 2]) + eps) * np.exp(x_deg[:, :, 2]) - eps
    # gmmax
    out[:, :, 3] = (np.abs(x0[:, :, 3]) + eps) * np.exp(-x_deg[:, :, 3]) - eps
    # IDLeak / IGLeak magnitudes
    out[:, :, 4] = (np.abs(x0[:, :, 4]) + eps) * np.exp(x_deg[:, :, 4]) - eps
    out[:, :, 5] = (np.abs(x0[:, :, 5]) + eps) * np.exp(x_deg[:, :, 5]) - eps
    return out


def compute_physical_metrics(y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute MAE/RMSE/relative-error in physical space."""
    valid_3d = mask[:, :, None] & ~np.isnan(y_true)
    eps = cfg.EPSILON
    out = {"rmse": {}, "mae": {}, "mre": {}}
    for fi, fname in enumerate(cfg.FEATURES):
        valid = valid_3d[:, :, fi]
        if valid.sum() < 1:
            out["rmse"][fname] = float("nan")
            out["mae"][fname] = float("nan")
            out["mre"][fname] = float("nan")
            continue
        yt = y_true[:, :, fi][valid]
        yp = y_pred[:, :, fi][valid]
        err = yp - yt
        out["rmse"][fname] = float(np.sqrt(np.mean(err ** 2)))
        out["mae"][fname] = float(np.mean(np.abs(err)))
        out["mre"][fname] = float(np.mean(np.abs(err) / (np.abs(yt) + eps)))
    return out


def compute_horizon_rmse(
    x_pred:   np.ndarray,
    x_true:   np.ndarray,
    mask:     np.ndarray,
    times_h:  np.ndarray,
    horizons: list = None,
) -> Dict:
    """Compute RMSE at specific time horizons using only valid observations."""
    if horizons is None:
        horizons = cfg.EVAL_HORIZONS_H

    canonical = cfg.TIME_POINTS_H
    results = {}
    for h in horizons:
        # Find the canonical time step closest to h
        diffs = [abs(t - h) for t in canonical]
        t_idx = int(np.argmin(diffs))

        # Only-valid aggregation at this horizon.
        # We keep devices that are present at this time step; feature-wise NaNs
        # are handled inside compute_rmse via valid_3d masking.
        mask_h = mask[:, t_idx]

        # If no device is valid at this exact step, fall back to nearest step
        # with at least one valid device to provide a stable numeric summary.
        if mask_h.sum() == 0:
            best_idx = None
            best_dist = None
            for j in range(len(canonical)):
                n_valid = int(mask[:, j].sum())
                if n_valid == 0:
                    continue
                dist = abs(canonical[j] - h)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = j
            if best_idx is None:
                # Entire split has no valid points; return stable zeros.
                results[h] = {"overall": 0.0}
                for fname in cfg.FEATURES:
                    results[h][fname] = 0.0
                continue
            t_idx = best_idx
            mask_h = mask[:, t_idx]

        r = compute_rmse(
            x_pred[mask_h, t_idx : t_idx + 1, :],
            x_true[mask_h, t_idx : t_idx + 1, :],
            mask[mask_h, t_idx : t_idx + 1],
        )
        results[h] = r
    return results


def compute_feature_distribution_stats(
    x_pred: np.ndarray,
    x_true: np.ndarray,
    mask: np.ndarray,
    x_base: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Summarize future-point prediction/target distribution mismatch."""
    out = {}
    valid_3d = mask[:, :, None] & ~np.isnan(x_true)
    for fi, feature_name in enumerate(cfg.FEATURES):
        valid = valid_3d[:, :, fi]
        if valid.sum() < 1:
            out[feature_name] = {
                "n": 0,
                "true_mean": float("nan"),
                "true_std": float("nan"),
                "true_p5": float("nan"),
                "true_p50": float("nan"),
                "true_p95": float("nan"),
                "pred_mean": float("nan"),
                "pred_std": float("nan"),
                "pred_p5": float("nan"),
                "pred_p50": float("nan"),
                "pred_p95": float("nan"),
            }
            if x_base is not None:
                out[feature_name].update({
                    "base_mean": float("nan"),
                    "base_std": float("nan"),
                    "base_p5": float("nan"),
                    "base_p50": float("nan"),
                    "base_p95": float("nan"),
                })
            continue
        y_true = x_true[:, :, fi][valid]
        y_pred = x_pred[:, :, fi][valid]
        stats = {
            "n": int(valid.sum()),
            "true_mean": float(np.mean(y_true)),
            "true_std": float(np.std(y_true)),
            "true_p5": float(np.percentile(y_true, 5)),
            "true_p50": float(np.percentile(y_true, 50)),
            "true_p95": float(np.percentile(y_true, 95)),
            "pred_mean": float(np.mean(y_pred)),
            "pred_std": float(np.std(y_pred)),
            "pred_p5": float(np.percentile(y_pred, 5)),
            "pred_p50": float(np.percentile(y_pred, 50)),
            "pred_p95": float(np.percentile(y_pred, 95)),
        }
        if x_base is not None:
            y_base = x_base[:, :, fi][valid]
            stats.update({
                "base_mean": float(np.mean(y_base)),
                "base_std": float(np.std(y_base)),
                "base_p5": float(np.percentile(y_base, 5)),
                "base_p50": float(np.percentile(y_base, 50)),
                "base_p95": float(np.percentile(y_base, 95)),
            })
        out[feature_name] = stats
    return out


def compute_decreasing_transition_fraction(
    x: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Fraction of valid adjacent transitions with negative delta."""
    out = {}
    for fi, feature_name in enumerate(cfg.FEATURES):
        valid = (
            mask[:, :-1]
            & mask[:, 1:]
            & ~np.isnan(x[:, :-1, fi])
            & ~np.isnan(x[:, 1:, fi])
        )
        if valid.sum() < 1:
            out[feature_name] = float("nan")
            continue
        delta = x[:, 1:, fi][valid] - x[:, :-1, fi][valid]
        out[feature_name] = float(np.mean(delta < 0.0))
    return out


def compute_latent_saturation(
    z_traj: np.ndarray,
    T_K: np.ndarray,
    times_h: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.95,
    early_horizon_h: float = 100.0,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Diagnose whether latent states approach their upper bound too early."""
    results = {}
    for temp_c in cfg.TEMPERATURES_C:
        temp_k = temp_c + cfg.CELSIUS_TO_KELVIN
        selected = np.abs(T_K - temp_k) < 1.0
        if selected.sum() == 0:
            continue

        results[temp_c] = {}
        selected_indices = np.where(selected)[0]
        for zi, latent_name in enumerate(cfg.LATENT_NAMES):
            z_sub = z_traj[selected, :, zi]
            mask_sub = mask[selected]
            valid_values = z_sub[mask_sub]
            if valid_values.size == 0:
                continue

            saturation_fraction = float(np.mean(valid_values >= threshold))
            early_mask = mask_sub & (times_h[selected] <= early_horizon_h)
            early_values = z_sub[early_mask]
            early_saturation_fraction = (
                float(np.mean(early_values >= threshold))
                if early_values.size > 0 else float("nan")
            )
            first_times = []
            for global_idx in selected_indices:
                valid_steps = np.where(mask[global_idx])[0]
                first_saturated = None
                for step in valid_steps:
                    if z_traj[global_idx, step, zi] >= threshold:
                        first_saturated = times_h[global_idx, step]
                        break
                if first_saturated is not None:
                    first_times.append(first_saturated)

            results[temp_c][latent_name] = {
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
                "max": float(np.max(valid_values)),
                "fraction_ge_095": saturation_fraction,
                "fraction_ge_095_pre100h": early_saturation_fraction,
                "median_first_saturation_h": (
                    float(np.median(first_times)) if first_times else float("nan")
                ),
            }
    return results


def compute_zc_time_profile(
    z_traj: np.ndarray,
    T_K: np.ndarray,
    times_h: np.ndarray,
    mask: np.ndarray,
    checkpoints_h=None,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """Summarize zC at selected physical-time checkpoints by temperature."""
    if checkpoints_h is None:
        checkpoints_h = [0, 1, 5, 20, 100, 500, 1000, 2000]

    out: Dict[int, Dict[int, Dict[str, float]]] = {}
    for temp_c in cfg.TEMPERATURES_C:
        temp_k = temp_c + cfg.CELSIUS_TO_KELVIN
        sel = np.abs(T_K - temp_k) < 1.0
        if sel.sum() == 0:
            continue

        out[temp_c] = {}
        z_sub = z_traj[sel, :, 4]
        t_sub = times_h[sel]
        m_sub = mask[sel]
        for h in checkpoints_h:
            at_h = np.isclose(t_sub, float(h), atol=1e-6)
            valid = at_h & m_sub
            vals = z_sub[valid]
            if vals.size == 0:
                out[temp_c][int(h)] = {
                    "n": 0,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "p5": float("nan"),
                    "p50": float("nan"),
                    "p95": float("nan"),
                }
                continue
            out[temp_c][int(h)] = {
                "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p5": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
            }
    return out


def freeze_zc_trajectory(z_traj: np.ndarray, prefix_len: int) -> np.ndarray:
    """Freeze zC after prefix to the last observed prefix state."""
    z_frozen = z_traj.copy()
    if prefix_len <= 0:
        return z_frozen
    zc_at_prefix = z_frozen[:, prefix_len - 1:prefix_len, 4]
    z_frozen[:, prefix_len:, 4] = np.repeat(zc_at_prefix, z_frozen.shape[1] - prefix_len, axis=1)
    return z_frozen


def _future_latent_mask(mask: np.ndarray, prefix_len: int) -> np.ndarray:
    """Boolean mask that keeps only future steps after the prefix."""
    m = mask.copy().astype(bool)
    m[:, :prefix_len] = False
    return m


def compute_future_latent_correlations(
    z_traj: np.ndarray,
    mask: np.ndarray,
    prefix_len: int,
) -> Dict[str, float]:
    """Correlations corr(zC, z*) in future region over valid points."""
    out = {}
    m = _future_latent_mask(mask, prefix_len)
    zc = z_traj[:, :, 4][m]
    if zc.size < 3:
        for name in ["zG", "zB", "zM", "zL"]:
            out[f"corr(zC,{name})"] = float("nan")
        return out

    zc_std = float(np.std(zc))
    for zi, name in enumerate(cfg.LATENT_NAMES[:4]):
        zv = z_traj[:, :, zi][m]
        if zv.size < 3 or zc_std < 1e-12 or float(np.std(zv)) < 1e-12:
            out[f"corr(zC,{name})"] = float("nan")
            continue
        out[f"corr(zC,{name})"] = float(np.corrcoef(zc, zv)[0, 1])
    return out


def compute_zc_logtime_r2(
    z_traj: np.ndarray,
    times_h: np.ndarray,
    T_K: np.ndarray,
    mask: np.ndarray,
    prefix_len: int,
) -> float:
    """Fit zC ~ a + b*log(1+t) + c*(1/T) on future valid points and return R^2."""
    m = _future_latent_mask(mask, prefix_len)
    valid = m
    y = z_traj[:, :, 4][valid]
    if y.size < 3:
        return float("nan")

    logt = np.log1p(times_h[valid])
    T_grid = np.broadcast_to(T_K[:, None], times_h.shape)
    invT = 1.0 / np.maximum(T_grid[valid], 1.0)
    X = np.stack([np.ones_like(logt), logt, invT], axis=1)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def compute_decoder_zc_contribution(
    model,
    z_traj: np.ndarray,
    z_ref: np.ndarray,
    mask: np.ndarray,
    prefix_len: int,
) -> Dict[str, Dict[str, float]]:
    """Estimate per-feature zC contribution by counterfactual decoder difference."""
    z_freeze = freeze_zc_trajectory(z_traj, prefix_len)
    with torch.no_grad():
        z_full_t = torch.from_numpy(z_traj).float()
        z_free_t = torch.from_numpy(z_freeze).float()
        z_ref_t = torch.from_numpy(z_ref).float()
        x_full = model.decoder(z_full_t, z_ref=z_ref_t).cpu().numpy()
        x_free = model.decoder(z_free_t, z_ref=z_ref_t).cpu().numpy()

    future_mask = _future_latent_mask(mask, prefix_len)
    out = {}
    eps = 1e-12
    for fi, fname in enumerate(cfg.FEATURES):
        valid = future_mask
        delta = np.abs((x_full[:, :, fi] - x_free[:, :, fi])[valid])
        base = np.abs(x_full[:, :, fi][valid])
        if delta.size < 1:
            out[fname] = {
                "abs_contribution": float("nan"),
                "relative_contribution": float("nan"),
            }
            continue
        out[fname] = {
            "abs_contribution": float(np.mean(delta)),
            "relative_contribution": float(np.mean(delta) / max(np.mean(base), eps)),
        }

    # Expose raw decoder zC column weights for direct structure checking.
    with torch.no_grad():
        W = model.decoder._effective_weight().cpu().numpy()
    out["_decoder_zc_weights"] = {
        fname: float(W[fi, 4]) for fi, fname in enumerate(cfg.FEATURES)
    }
    return out


def compute_freeze_zc_ablation(
    model,
    z_traj: np.ndarray,
    z_ref: np.ndarray,
    x_true_norm: np.ndarray,
    mask: np.ndarray,
    norm_stats: Dict[str, Dict[str, float]],
    prefix_len: int,
) -> Dict[str, Dict[str, float]]:
    """Compare full-model vs frozen-zC future RMSE and contribution skill."""
    z_freeze = freeze_zc_trajectory(z_traj, prefix_len)
    with torch.no_grad():
        z_full_t = torch.from_numpy(z_traj).float()
        z_free_t = torch.from_numpy(z_freeze).float()
        z_ref_t = torch.from_numpy(z_ref).float()
        x_full_norm = model.decoder(z_full_t, z_ref=z_ref_t).cpu().numpy()
        x_free_norm = model.decoder(z_free_t, z_ref=z_ref_t).cpu().numpy()

    x_true_deg = denormalize_x(x_true_norm, norm_stats)
    x_full_deg = denormalize_x(x_full_norm, norm_stats)
    x_free_deg = denormalize_x(x_free_norm, norm_stats)
    future_mask = _future_latent_mask(mask, prefix_len)

    rmse_full = compute_rmse(x_full_deg, x_true_deg, future_mask)
    rmse_frozen = compute_rmse(x_free_deg, x_true_deg, future_mask)

    valid_3d = future_mask[:, :, None] & ~np.isnan(x_true_deg)
    mse_full = float(np.mean((x_full_deg[valid_3d] - x_true_deg[valid_3d]) ** 2)) if valid_3d.any() else float("nan")
    mse_frozen = float(np.mean((x_free_deg[valid_3d] - x_true_deg[valid_3d]) ** 2)) if valid_3d.any() else float("nan")
    contribution_skill = 1.0 - mse_full / max(mse_frozen, 1e-12)

    per_feature_skill = {}
    for fi, fname in enumerate(cfg.FEATURES):
        valid = valid_3d[:, :, fi]
        if valid.sum() < 1:
            per_feature_skill[fname] = float("nan")
            continue
        mse_m = float(np.mean((x_full_deg[:, :, fi][valid] - x_true_deg[:, :, fi][valid]) ** 2))
        mse_f = float(np.mean((x_free_deg[:, :, fi][valid] - x_true_deg[:, :, fi][valid]) ** 2))
        per_feature_skill[fname] = 1.0 - mse_m / max(mse_f, 1e-12)

    return {
        "rmse_full": rmse_full,
        "rmse_frozen_zc": rmse_frozen,
        "zc_contribution_skill_overall": contribution_skill,
        "zc_contribution_skill_by_feature": per_feature_skill,
    }


# ---------------------------------------------------------------------------
# 2. Structural validation
# ---------------------------------------------------------------------------

def _monotone_violation_rate(z_traj: np.ndarray, mask: np.ndarray) -> float:
    """Compute monotone violation rate for zM/zL/zC on valid adjacent steps."""
    mono_idx = [2, 3, 4]
    violations = 0
    pairs = 0
    _, T, _ = z_traj.shape
    for step in range(1, T):
        valid = mask[:, step - 1] & mask[:, step]
        if valid.sum() < 1:
            continue
        delta = z_traj[valid][:, step, mono_idx] - z_traj[valid][:, step - 1, mono_idx]
        violations += int((delta < -1e-4).sum())
        pairs += delta.size
    return violations / max(pairs, 1)


def validate_structure(
    z_traj: np.ndarray,
    z_prefix: np.ndarray,
    T_K: np.ndarray,
    mask: np.ndarray,
    prefix_len: int,
) -> Dict:
    """
    Check physics structural constraints:
      - bounds: z ∈ [0,1]
      - monotonicity: zM, zL, zC non-decreasing
      - temperature ordering: higher T → more damage (zC)
    """
    results = {}

    # Bounds check
    in_bounds = ((z_traj >= 0.0) & (z_traj <= 1.0))
    results["bounds_fraction"] = float(in_bounds[mask[:, :, None].repeat(5, axis=2)].mean())

    # Monotonicity reported separately for full trajectory, encoder prefix,
    # and ODE-driven future portion.
    results["monotone_violation_rate_full"] = _monotone_violation_rate(z_traj, mask)

    prefix_mask = mask[:, :z_prefix.shape[1]]
    results["monotone_violation_rate_prefix_encoder"] = _monotone_violation_rate(
        z_prefix,
        prefix_mask,
    )

    ode_mask = mask.copy().astype(bool)
    ode_mask[:, :max(prefix_len - 1, 0)] = False
    results["monotone_violation_rate_ode_future"] = _monotone_violation_rate(
        z_traj,
        ode_mask,
    )

    N, _, _ = z_traj.shape

    # Temperature ordering (zC at last valid step)
    n_temps = len(cfg.TEMPERATURES_C)
    order_violations = 0
    order_pairs = 0
    temp_vals = [tc + cfg.CELSIUS_TO_KELVIN for tc in cfg.TEMPERATURES_C]
    last_zC = np.full(N, np.nan)
    for b in range(N):
        valid_steps = np.where(mask[b])[0]
        if len(valid_steps) > 0:
            last_zC[b] = z_traj[b, valid_steps[-1], 4]

    for i in range(N):
        for j in range(N):
            if T_K[i] < T_K[j] - 1.0 and not np.isnan(last_zC[i]) and not np.isnan(last_zC[j]):
                if last_zC[i] > last_zC[j]:
                    order_violations += 1
                order_pairs += 1
    results["temp_order_violation_rate"] = (
        order_violations / max(order_pairs, 1)
    )

    return results


def compute_initial_latent_stats(
    z_prefix: np.ndarray,
    T_K: np.ndarray,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Summarize initial latent distributions by temperature."""
    out = {}
    z0 = z_prefix[:, 0, :]
    for temp_c in cfg.TEMPERATURES_C:
        temp_k = temp_c + cfg.CELSIUS_TO_KELVIN
        sel = np.abs(T_K - temp_k) < 1.0
        if sel.sum() < 1:
            continue
        out[temp_c] = {}
        z0_t = z0[sel]
        for zi, lname in enumerate(cfg.LATENT_NAMES):
            vals = z0_t[:, zi]
            out[temp_c][lname] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "p5": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
                "fraction_ge_09": float(np.mean(vals >= 0.9)),
            }
    return out


def counterfactual_temperature_ordering(
    model,
    z0: np.ndarray,
    alpha: np.ndarray,
    times_h: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """
    Counterfactual temperature test with fixed z0/alpha/time grid.
    For each sample and time, check ordering 275C <= 300C <= 325C.
    """
    model.eval()
    with torch.no_grad():
        z0_t = torch.from_numpy(z0).float().to(device)
        alpha_t = torch.from_numpy(alpha).float().to(device)
        times_t = torch.from_numpy(times_h).float().to(device)

        t275 = torch.full((z0_t.shape[0],), 275.0 + cfg.CELSIUS_TO_KELVIN, device=device)
        t300 = torch.full((z0_t.shape[0],), 300.0 + cfg.CELSIUS_TO_KELVIN, device=device)
        t325 = torch.full((z0_t.shape[0],), 325.0 + cfg.CELSIUS_TO_KELVIN, device=device)

        z_275 = model.ode.integrate_trajectory(z0_t, t275, times_t, alpha_t).cpu().numpy()
        z_300 = model.ode.integrate_trajectory(z0_t, t300, times_t, alpha_t).cpu().numpy()
        z_325 = model.ode.integrate_trajectory(z0_t, t325, times_t, alpha_t).cpu().numpy()

    def _pair_violation(a: np.ndarray, b: np.ndarray) -> float:
        bad = np.sum(a > b + 1e-6)
        total = a.size
        return float(bad / max(total, 1))

    zL_viol = 0.5 * (_pair_violation(z_275[:, :, 3], z_300[:, :, 3]) + _pair_violation(z_300[:, :, 3], z_325[:, :, 3]))
    zC_viol = 0.5 * (_pair_violation(z_275[:, :, 4], z_300[:, :, 4]) + _pair_violation(z_300[:, :, 4], z_325[:, :, 4]))

    return {
        "zL_violation_rate": zL_viol,
        "zC_violation_rate": zC_viol,
    }


# ---------------------------------------------------------------------------
# 3. Counterfactual intervention
# ---------------------------------------------------------------------------

def counterfactual_intervention(
    model,
    z_base: torch.Tensor,     # (1, 5) base latent state
    T_K:    torch.Tensor,     # (1,) temperature
    decoder,
    delta:  float = 0.2,
) -> Dict[str, np.ndarray]:
    """
    Perturb each latent state by +delta and observe the change in decoded
    output. Returns the sensitivity matrix: dx[fi] per dz[li].

    Args:
        z_base  : (1, 5)
        T_K     : not used for linear decoder but kept for API consistency
        decoder : SparsePhysicsDecoder
        delta   : perturbation size

    Returns:
        dict { 'sensitivity': (6, 5) numpy array }
    """
    model.eval()
    with torch.no_grad():
        x_base = decoder(z_base)   # (1, 6)
        sens   = np.zeros((cfg.FEATURE_DIM, cfg.LATENT_DIM))
        for li in range(cfg.LATENT_DIM):
            z_pert = z_base.clone()
            z_pert[0, li] += delta
            z_pert = z_pert.clamp(0, 1)
            x_pert = decoder(z_pert)   # (1, 6)
            sens[:, li] = (x_pert - x_base).squeeze().cpu().numpy() / delta

    return {"sensitivity": sens}


# ---------------------------------------------------------------------------
# 4. Ablation study
# ---------------------------------------------------------------------------

def ablation_study(
    model,
    dataset: dict,
    test_dl,
    device:  torch.device,
    prefix_len: int = 4,
) -> Dict:
    """
    Ablate each latent state (set it to 0) and measure impact on prediction
    RMSE.

    Returns dict: state_name → per-feature RMSE change vs. full model.
    """
    state_names = cfg.LATENT_NAMES

    def _eval_with_ablation(ablated_dim: Optional[int]) -> Dict:
        all_pred, all_true, all_mask = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in test_dl:
                enc_input = batch["enc_input"].to(device)
                x_true    = batch["x"].to(device)
                mask      = batch["mask"].to(device)
                times_h   = batch["times_h"].to(device)
                T_K       = batch["T_K"].to(device)
                x0        = batch["x0"].to(device)

                out = predict_from_prefix(
                    model, enc_input, x_true, mask,
                    times_h, T_K, x0, prefix_len, device)
                z_ode = out["z_ode"]

                if ablated_dim is not None:
                    z_ode = z_ode.clone()
                    z_ode[:, :, ablated_dim] = 0.0

                x_pred = model.decoder(z_ode)
                all_pred.append(x_pred.cpu().numpy())
                all_true.append(x_true.cpu().numpy())
                all_mask.append(mask.cpu().numpy())

        return compute_rmse(
            np.concatenate(all_pred, 0),
            np.concatenate(all_true, 0),
            np.concatenate(all_mask, 0),
        )

    results = {}
    baseline = _eval_with_ablation(None)
    results["baseline"] = baseline

    for li, name in enumerate(state_names):
        abl = _eval_with_ablation(li)
        delta = {k: abl[k] - baseline[k] for k in baseline}
        results[f"ablate_{name}"] = delta
        log.info("  Ablate %s → overall ΔRMSE = %.5f", name, delta["overall"])

    return results


# ---------------------------------------------------------------------------
# 5. Visualisation utilities
# ---------------------------------------------------------------------------

def plot_latent_trajectories(
    z_traj:   np.ndarray,     # (N, T, 5)
    T_K:      np.ndarray,     # (N,)
    mask:     np.ndarray,     # (N, T)
    times_h:  np.ndarray,     # (N, T)
    save_path: str = None,
    n_devices: int = 6,
):
    """Plot latent state trajectories coloured by temperature."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    temp_vals = sorted(set(T_K.round().astype(int).tolist()))
    cmap = plt.get_cmap("plasma", len(temp_vals))
    color_map = {tv: cmap(i) for i, tv in enumerate(temp_vals)}

    state_names = cfg.LATENT_NAMES
    device_idx = np.random.choice(len(z_traj),
                                   min(n_devices, len(z_traj)),
                                   replace=False)
    for li, ax in enumerate(axes):
        for di in device_idx:
            valid = mask[di]
            t_plot = times_h[di, valid]
            z_plot = z_traj[di, valid, li]
            color  = color_map.get(int(T_K[di].round()), cmap(0))
            ax.plot(t_plot, z_plot, "-o", color=color, markersize=3, linewidth=1.2)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(state_names[li])
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Physics Latent State Trajectories", fontsize=12)
    plt.tight_layout()

    # Legend
    handles = [
        plt.Line2D([0], [0], color=color_map[tv], lw=2,
                   label=f"{tv - 273:.0f}°C")
        for tv in temp_vals if tv in color_map
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", save_path)
    plt.close()


def plot_reconstruction(
    x_pred:    np.ndarray,   # (T, 6)
    x_true:    np.ndarray,   # (T, 6)
    mask:      np.ndarray,   # (T,)
    times_h:   np.ndarray,   # (T,)
    title:     str = "",
    save_path: str = None,
):
    """Plot observed vs. reconstructed degradation features for one device."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for fi, (ax, fname) in enumerate(zip(axes, cfg.FEATURES)):
        t_valid = times_h[mask]
        y_true  = x_true[mask, fi]
        y_pred  = x_pred[mask, fi]
        ax.plot(t_valid, y_true, "o-", label="True",  linewidth=1.5, markersize=4)
        ax.plot(t_valid, y_pred, "s--", label="Pred", linewidth=1.5, markersize=4)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(f"x{fi+1} [{fname}]")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", save_path)
    plt.close()


def plot_sensitivity_heatmap(
    sensitivity: np.ndarray,   # (6, 5)
    save_path:   str = None,
):
    """Heatmap of decoder sensitivity ∂x_i/∂z_j."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(sensitivity, cmap="RdBu_r",
                   vmin=-np.abs(sensitivity).max(),
                   vmax= np.abs(sensitivity).max(),
                   aspect="auto")
    ax.set_xticks(range(cfg.LATENT_DIM))
    ax.set_xticklabels(cfg.LATENT_NAMES)
    ax.set_yticks(range(cfg.FEATURE_DIM))
    ax.set_yticklabels([f"x{i+1}[{n}]" for i, n in enumerate(cfg.FEATURES)])
    ax.set_xlabel("Latent state")
    ax.set_ylabel("Feature")
    plt.colorbar(im, ax=ax, label="∂x_i / ∂z_j")
    plt.title("Decoder Sensitivity (Physics Structure Verification)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", save_path)
    plt.close()


def plot_temperature_response(
    z_traj_by_temp: Dict[float, np.ndarray],   # T_K → (T, 5)
    times_h: np.ndarray,
    save_path: str = None,
):
    """Compare latent state zC across three temperatures for a fixed device."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["blue", "orange", "red"]
    for (tk, z_traj), color in zip(sorted(z_traj_by_temp.items()), colors):
        tc = tk - cfg.CELSIUS_TO_KELVIN
        ax.plot(times_h, z_traj[:, 4], "-o", color=color,
                label=f"{tc:.0f}°C", linewidth=2, markersize=4)
    ax.set_xlabel("Storage time (h)")
    ax.set_ylabel("zC (cumulative damage)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Temperature-Ordered Cumulative Damage (zC)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(model, dataset: dict, prefix_len: int = 4, results_tag: Optional[str] = None):
    """
    Run the complete evaluation pipeline on the test set and save results.
    """
    device = torch.device("cpu")  # evaluation can run on CPU
    split  = dataset["split"]
    test_idx = split["test"]

    # Build test data
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __init__(self, ds, idx):
            self.x   = torch.from_numpy(ds["x"]).float()
            if "feature_mask" in ds:
                self.feature_mask = torch.from_numpy(ds["feature_mask"]).bool()
            else:
                self.feature_mask = torch.from_numpy(np.isfinite(ds["x"]).astype(bool))
            self.mask = torch.from_numpy(ds["mask"]).bool()
            self.t    = torch.from_numpy(ds["times_h"]).float()
            self.TK   = torch.from_numpy(ds["T_K"]).float()
            x0_key = "x0_normalized" if "x0_normalized" in ds else "x0_static"
            self.x0   = torch.from_numpy(ds[x0_key]).float()
            self.x0_static = torch.from_numpy(ds["x0_static"]).float()
            self.idx  = idx

        def __len__(self): return len(self.idx)

        def __getitem__(self, i):
            b = self.idx[i]
            x    = self.x[b]
            feature_mask = self.feature_mask[b]
            mask = self.mask[b]
            t    = self.t[b]
            TK   = self.TK[b]
            x0   = self.x0[b]
            x0s  = self.x0_static[b]
            T_norm = torch.tensor((TK.item() - cfg.T_REF_K) / cfg.T_REF_K)
            T_f  = T_norm.expand(x.shape[0], 1)
            lt   = torch.log(t + 1.0)
            dl   = torch.zeros_like(lt)
            dl[:-1] = lt[1:] - lt[:-1]
            enc  = torch.cat([
                torch.nan_to_num(x, nan=0.0),
                feature_mask.float(),
                T_f,
                lt.unsqueeze(1),
                dl.unsqueeze(1),
            ], dim=1)
            return {"enc_input": enc, "x": x, "mask": mask,
                    "times_h": t, "T_K": TK, "x0": x0, "x0_static": x0s}

    test_ds = _DS(dataset, test_idx)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False,
                         collate_fn=lambda b: {k: torch.stack([i[k] for i in b]) for k in b[0]})

    all_pred, all_true, all_mask, all_TK, all_t, all_z, all_z_prefix, all_alpha, all_x0_static = [], [], [], [], [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            enc_in  = batch["enc_input"]
            x_true  = batch["x"]
            mask    = batch["mask"]
            times_h = batch["times_h"]
            T_K     = batch["T_K"]
            x0      = batch["x0"]
            x0_static = batch["x0_static"]

            out = predict_from_prefix(
                model, enc_in, x_true, mask, times_h, T_K, x0, prefix_len, device)

            all_pred.append(out["x_pred"].numpy())
            all_true.append(x_true.numpy())
            all_mask.append(mask.numpy())
            all_TK.append(T_K.numpy())
            all_t.append(times_h.numpy())
            all_z.append(out["z_ode"].numpy())
            all_z_prefix.append(out["z_enc"].numpy())
            all_alpha.append(out["alpha"].numpy())
            all_x0_static.append(x0_static.numpy())

    x_pred_all = np.concatenate(all_pred, 0)
    x_true_all = np.concatenate(all_true, 0)
    mask_all   = np.concatenate(all_mask, 0)
    TK_all     = np.concatenate(all_TK,  0)
    t_all      = np.concatenate(all_t,   0)
    z_all      = np.concatenate(all_z,   0)
    z_prefix_all = np.concatenate(all_z_prefix, 0)
    alpha_all  = np.concatenate(all_alpha, 0)
    x0_all     = np.concatenate(all_x0_static,  0)
    test_device_ids = [dataset["device_ids"][i] for i in test_idx]

    # Denormalized transformed feature space
    x_true_deg = denormalize_x(x_true_all, dataset["norm_stats"])
    x_pred_deg = denormalize_x(x_pred_all, dataset["norm_stats"])
    future_mask_all = make_future_mask(mask_all, prefix_len)
    prefix_mask_all = make_prefix_mask(mask_all, prefix_len)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    # ---- Metric tables (transformed space) ----
    overall_rmse = compute_rmse(x_pred_deg, x_true_deg, future_mask_all)
    overall_mae = compute_mae(x_pred_deg, x_true_deg, future_mask_all)
    valid_count = compute_valid_point_count(x_true_deg, future_mask_all)
    prefix_rmse = compute_rmse(x_pred_deg, x_true_deg, prefix_mask_all)
    prefix_rmse_norm = compute_rmse(x_pred_all, x_true_all, prefix_mask_all)
    prefix_macro_rmse_norm = float(np.mean([prefix_rmse_norm[f] for f in cfg.FEATURES]))
    log.info("RMSE transformed space: %s", {k: f"{v:.5f}" for k, v in overall_rmse.items()})
    log.info("MAE transformed space: %s", {k: f"{v:.5f}" for k, v in overall_mae.items()})
    log.info("Valid point count: %s", valid_count)

    log.info("Prefix RMSE transformed space: %s", {k: f"{v:.5f}" for k, v in prefix_rmse.items()})
    log.info("Prefix RMSE normalized space: %s", {k: f"{v:.5f}" for k, v in prefix_rmse_norm.items()})
    log.info("Prefix macro-RMSE normalized (feature average): %.5f", prefix_macro_rmse_norm)

    # Decoder reference sanity check: x_pred at t0 should be close to zero.
    t0_abs = np.abs(x_pred_deg[:, 0, :])
    log.info("Predicted x at t0 |mean(abs)| by feature: %s", {
        fname: f"{float(np.nanmean(t0_abs[:, fi])):.5e}"
        for fi, fname in enumerate(cfg.FEATURES)
    })

    horizon_rmse = compute_horizon_rmse(x_pred_deg, x_true_deg, future_mask_all, t_all)
    log.info("Horizon RMSE: %s", {
        h: {k: f"{v:.4f}" for k, v in d.items() if k == "overall"}
        for h, d in horizon_rmse.items()
    })

    # ---- Zero-time and scale diagnostics in transformed space ----
    zero_t = check_zero_time_features(x_true_deg, mask_all)
    log.info("t=0 |mean(abs(x_j(0)))|: %s", {k: f"{v:.4e}" for k, v in zero_t.items()})

    # s_Vth diagnostics by split (computed on full dataset)
    s_vth = float(dataset.get("x0_vth_sdev", 1.0))
    x_full_deg = denormalize_x(dataset["x"], dataset["norm_stats"])
    vth_delta_full = x_full_deg[:, :, 0] * s_vth
    split_stats = {}
    for split_name, idx in split.items():
        m = dataset["mask"][idx] & ~np.isnan(vth_delta_full[idx])
        vals = vth_delta_full[idx][m]
        split_stats[split_name] = float(np.std(vals)) if vals.size > 1 else float("nan")
    log.info("s_Vth fixed=%.6f | std(ΔVth) train/val/test = %.6f / %.6f / %.6f",
             s_vth,
             split_stats.get("train", float("nan")),
             split_stats.get("val", float("nan")),
             split_stats.get("test", float("nan")))

    train_idx = split["train"]
    x_train_full = denormalize_x(dataset["x"][train_idx], dataset["norm_stats"])
    mask_train_full = dataset["mask"][train_idx]
    times_train_full = dataset["times_h"][train_idx]
    T_train_full = dataset["T_K"][train_idx]

    for leak_name in ("IDLeak", "IGLeak"):
        st = dataset.get("norm_stats", {}).get(leak_name, {})
        fi = cfg.FEATURES.index(leak_name)
        train_vals = x_train_full[:, :, fi][mask_train_full]
        train_vals = train_vals[np.isfinite(train_vals)]
        if train_vals.size > 0:
            p1, p50, p99 = np.percentile(train_vals, [1, 50, 99])
            log.info(
                "%s normalization stats (train transformed): min=%.4f p1=%.4f p50=%.4f p99=%.4f max=%.4f",
                leak_name,
                float(st.get("min", float("nan"))),
                float(p1),
                float(p50),
                float(p99),
                float(st.get("max", float("nan"))),
            )

    # ---- Baselines and skill scores in transformed space ----
    baseline_zero = np.zeros_like(x_true_deg, dtype=float)
    baseline_persist = _build_prefix_persistence_baseline(
        x_true_deg, mask_all, prefix_len)
    coeffs_loglin = _fit_logtime_linear_baseline(
        x_train_full,
        mask_train_full,
        times_train_full,
        T_train_full,
    )
    baseline_loglin = _predict_logtime_linear(coeffs_loglin, t_all, TK_all)

    skill = compute_skill_scores(
        x_pred_deg,
        x_true_deg,
        future_mask_all,
        {
            "zero": baseline_zero,
            "persistence": baseline_persist,
            "logtime_linear": baseline_loglin,
        },
    )
    log.info("Skill(overall): zero=%.4f | persistence=%.4f | logtime_linear=%.4f",
             skill["zero"]["overall"],
             skill["persistence"]["overall"],
             skill["logtime_linear"]["overall"])
    for baseline_name, metrics in skill.items():
        feature_stats = {fname: f"{metrics[fname]:.4f}" for fname in cfg.FEATURES}
        log.info("Skill by feature vs %s: %s", baseline_name, feature_stats)

    nrmse_r2 = compute_nrmse_r2(x_pred_deg, x_true_deg, future_mask_all)
    log.info("NRMSE: %s", {k: f"{v:.4f}" for k, v in nrmse_r2["nrmse"].items()})
    log.info("R2: %s", {k: f"{v:.4f}" for k, v in nrmse_r2["r2"].items()})

    mult_factor = compute_multiplicative_error_factor(overall_rmse)
    log.info("Multiplicative error factor (~exp(RMSE)): %s", {
        k: ("nan" if not np.isfinite(v) else f"{v:.4f}")
        for k, v in mult_factor.items()
    })

    rmse_by_temp = compute_rmse_by_temperature(x_pred_deg, x_true_deg, future_mask_all, TK_all)
    log.info("RMSE by temperature (overall): %s", {
        k: f"{v['overall']:.5f}" for k, v in rmse_by_temp.items()
    })

    rmse_by_interval = compute_rmse_by_time_interval(x_pred_deg, x_true_deg, future_mask_all, t_all)
    log.info("RMSE by time interval (overall): %s", {
        k: ("nan" if not np.isfinite(v.get("overall", np.nan)) else f"{v['overall']:.5f}")
        for k, v in rmse_by_interval.items()
    })

    rmse_by_device = compute_rmse_by_device(x_pred_deg, x_true_deg, future_mask_all, test_device_ids)
    if len(rmse_by_device) > 0:
        vals = np.array(list(rmse_by_device.values()), dtype=float)
        log.info("RMSE by device summary: n=%d | mean=%.5f | median=%.5f | p90=%.5f",
                 len(vals), float(np.mean(vals)), float(np.median(vals)), float(np.percentile(vals, 90)))

    # ---- Physical-space metrics (inverse transformed) ----
    y_true_phys = inverse_transform_to_physical(x_true_deg, x0_all, s_vth)
    y_pred_phys = inverse_transform_to_physical(x_pred_deg, x0_all, s_vth)
    baseline_loglin_phys = inverse_transform_to_physical(baseline_loglin, x0_all, s_vth)
    physical_metrics = compute_physical_metrics(y_pred_phys, y_true_phys, future_mask_all)
    log.info("Physical-space RMSE: %s", {k: f"{v:.5e}" for k, v in physical_metrics["rmse"].items()})
    log.info("Physical-space MAE: %s", {k: f"{v:.5e}" for k, v in physical_metrics["mae"].items()})
    log.info("Physical-space MRE: %s", {k: f"{v:.5f}" for k, v in physical_metrics["mre"].items()})

    # ---- Structural validation ----
    struct = validate_structure(z_all, z_prefix_all, TK_all, mask_all, prefix_len)
    log.info("Structural validation: %s",
             {k: f"{v:.4f}" for k, v in struct.items()})

    init_latent_stats = compute_initial_latent_stats(z_prefix_all, TK_all)
    for temp_c, latent_map in init_latent_stats.items():
        zl = latent_map.get("zL", {})
        log.info(
            "zL(0) stats @ %dC | mean=%.4f std=%.4f median=%.4f p5=%.4f p95=%.4f frac>=0.9=%.4f",
            temp_c,
            zl.get("mean", float("nan")),
            zl.get("std", float("nan")),
            zl.get("median", float("nan")),
            zl.get("p5", float("nan")),
            zl.get("p95", float("nan")),
            zl.get("fraction_ge_09", float("nan")),
        )
        z0_summary = {name: f"{vals.get('mean', float('nan')):.4f}" for name, vals in latent_map.items()}
        log.info("z(0) means @ %dC: %s", temp_c, z0_summary)

    temp_cf = counterfactual_temperature_ordering(
        model,
        z_prefix_all[:, -1, :],
        alpha_all,
        t_all,
        device,
    )
    log.info("Counterfactual temperature ordering (fixed z0/alpha): %s",
             {k: f"{v:.4f}" for k, v in temp_cf.items()})

    for tc in cfg.TEMPERATURES_C:
        tk = tc + cfg.CELSIUS_TO_KELVIN
        sel = np.abs(TK_all - tk) < 1.0
        if sel.sum() == 0:
            continue
        vals = alpha_all[sel]
        log.info(
            "Alpha by temperature @ %dC | mean=%.4f std=%.4f min=%.4f max=%.4f n=%d",
            tc,
            float(np.mean(vals)),
            float(np.std(vals)),
            float(np.min(vals)),
            float(np.max(vals)),
            int(vals.size),
        )

    feature_dist = compute_feature_distribution_stats(
        x_pred_deg, x_true_deg, future_mask_all, x_base=baseline_loglin)
    for feature_name, values in feature_dist.items():
        log.info(
            "Feature distribution %s | n=%d | true mean/std/p5/p50/p95=% .4f/% .4f/% .4f/% .4f/% .4f | pred mean/std/p5/p50/p95=% .4f/% .4f/% .4f/% .4f/% .4f | base mean/std/p5/p50/p95=% .4f/% .4f/% .4f/% .4f/% .4f",
            feature_name,
            values["n"],
            values["true_mean"],
            values["true_std"],
            values["true_p5"],
            values["true_p50"],
            values["true_p95"],
            values["pred_mean"],
            values["pred_std"],
            values["pred_p5"],
            values["pred_p50"],
            values["pred_p95"],
            values.get("base_mean", float("nan")),
            values.get("base_std", float("nan")),
            values.get("base_p5", float("nan")),
            values.get("base_p50", float("nan")),
            values.get("base_p95", float("nan")),
        )

    leakage_floor_map = dataset.get("leakage_floor", {}) or {}
    future_transitions_true = compute_decreasing_transition_fraction(y_true_phys, future_mask_all)
    future_transitions_pred = compute_decreasing_transition_fraction(y_pred_phys, future_mask_all)
    future_transitions_base = compute_decreasing_transition_fraction(baseline_loglin_phys, future_mask_all)
    leakage_phase1 = compute_leakage_phase1_diagnostics(
        y_true_phys,
        y_pred_phys,
        baseline_loglin_phys,
        future_mask_all,
        leakage_floor_map=leakage_floor_map,
    )
    for leak_name in ("IDLeak", "IGLeak"):
        fi = cfg.FEATURES.index(leak_name)
        floor = float(leakage_floor_map.get(leak_name, float("nan")))
        valid = future_mask_all & ~np.isnan(y_true_phys[:, :, fi])
        true_vals = y_true_phys[:, :, fi][valid]
        pred_vals = y_pred_phys[:, :, fi][valid]
        base_vals = baseline_loglin_phys[:, :, fi][valid]
        tol = max(abs(floor) * 1e-6, 1e-15) if np.isfinite(floor) else 1e-15
        true_floor_hits = int(np.sum(np.isfinite(true_vals) & np.isclose(true_vals, floor, rtol=0.0, atol=tol))) if np.isfinite(floor) else 0
        pred_floor_hits = int(np.sum(np.isfinite(pred_vals) & np.isclose(pred_vals, floor, rtol=0.0, atol=tol))) if np.isfinite(floor) else 0
        base_floor_hits = int(np.sum(np.isfinite(base_vals) & np.isclose(base_vals, floor, rtol=0.0, atol=tol))) if np.isfinite(floor) else 0
        log.info(
            "Leakage diagnostics %s | floor=%.3e | true mean/std/p5/p50/p95=% .5e/% .5e/% .5e/% .5e/% .5e | pred mean/std/p5/p50/p95=% .5e/% .5e/% .5e/% .5e/% .5e | base mean/std/p5/p50/p95=% .5e/% .5e/% .5e/% .5e/% .5e | frac(true<0)=%.4f | decreasing frac true/pred/base=%.4f/%.4f/%.4f | near-floor true/pred/base=%.4f/%.4f/%.4f | floor hits true/pred/base=%d/%d/%d",
            leak_name,
            floor,
            float(np.mean(true_vals)) if true_vals.size else float("nan"),
            float(np.std(true_vals)) if true_vals.size else float("nan"),
            float(np.percentile(true_vals, 5)) if true_vals.size else float("nan"),
            float(np.percentile(true_vals, 50)) if true_vals.size else float("nan"),
            float(np.percentile(true_vals, 95)) if true_vals.size else float("nan"),
            float(np.mean(pred_vals)) if pred_vals.size else float("nan"),
            float(np.std(pred_vals)) if pred_vals.size else float("nan"),
            float(np.percentile(pred_vals, 5)) if pred_vals.size else float("nan"),
            float(np.percentile(pred_vals, 50)) if pred_vals.size else float("nan"),
            float(np.percentile(pred_vals, 95)) if pred_vals.size else float("nan"),
            float(np.mean(base_vals)) if base_vals.size else float("nan"),
            float(np.std(base_vals)) if base_vals.size else float("nan"),
            float(np.percentile(base_vals, 5)) if base_vals.size else float("nan"),
            float(np.percentile(base_vals, 50)) if base_vals.size else float("nan"),
            float(np.percentile(base_vals, 95)) if base_vals.size else float("nan"),
            float(np.mean(true_vals < 0.0)) if true_vals.size else float("nan"),
            future_transitions_true.get(leak_name, float("nan")),
            future_transitions_pred.get(leak_name, float("nan")),
            future_transitions_base.get(leak_name, float("nan")),
            leakage_phase1.get(leak_name, {}).get("near_floor_fraction_true", float("nan")),
            leakage_phase1.get(leak_name, {}).get("near_floor_fraction_pred", float("nan")),
            leakage_phase1.get(leak_name, {}).get("near_floor_fraction_base", float("nan")),
            true_floor_hits,
            pred_floor_hits,
            base_floor_hits,
        )

    for leak_name in ("IDLeak", "IGLeak"):
        fi = cfg.FEATURES.index(leak_name)
        leak_rmse_by_temp = {}
        leak_rmse_by_interval = {}
        for tc in cfg.TEMPERATURES_C:
            tk = tc + cfg.CELSIUS_TO_KELVIN
            sel = np.abs(TK_all - tk) < 1.0
            if sel.sum() < 1:
                continue
            leak_rmse_by_temp[f"{tc}C"] = compute_rmse(
                x_pred_deg[sel, :, fi:fi+1],
                x_true_deg[sel, :, fi:fi+1],
                future_mask_all[sel],
                per_feature=False,
            )["overall"]
        for lo, hi in [(0.0, 100.0), (100.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0)]:
            tmask = (t_all >= lo) & (t_all < hi)
            m = future_mask_all & tmask
            if m.sum() < 1:
                continue
            leak_rmse_by_interval[f"[{int(lo)},{int(hi)})h"] = compute_rmse(
                x_pred_deg[:, :, fi:fi+1],
                x_true_deg[:, :, fi:fi+1],
                m,
                per_feature=False,
            )["overall"]
        log.info("Leakage RMSE by temperature %s: %s", leak_name, {
            k: f"{v:.5f}" for k, v in leak_rmse_by_temp.items()
        })
        # Phase 1 item 7: leakage skill vs logtime-linear baseline by temperature
        leak_skill_by_temp = {}
        for tc in cfg.TEMPERATURES_C:
            tk = tc + cfg.CELSIUS_TO_KELVIN
            sel = np.abs(TK_all - tk) < 1.0
            if sel.sum() < 1:
                continue
            valid = future_mask_all[sel] & ~np.isnan(x_true_deg[sel, :, fi])
            if valid.sum() < 1:
                leak_skill_by_temp[f"{tc}C"] = float("nan")
                continue
            mse_m = float(np.mean(
                (x_pred_deg[sel, :, fi][valid] - x_true_deg[sel, :, fi][valid]) ** 2
            ))
            mse_b = float(np.mean(
                (baseline_loglin[sel, :, fi][valid] - x_true_deg[sel, :, fi][valid]) ** 2
            ))
            leak_skill_by_temp[f"{tc}C"] = 1.0 - mse_m / max(mse_b, 1e-12)
        log.info("Leakage skill vs logtime baseline by temperature %s: %s", leak_name, {
            k: f"{v:.4f}" for k, v in leak_skill_by_temp.items()
        })
        log.info("Leakage RMSE by time interval %s: %s", leak_name, {
            k: f"{v:.5f}" for k, v in leak_rmse_by_interval.items()
        })

    latent_saturation = compute_latent_saturation(z_all, TK_all, t_all, mask_all)
    for temp_c, latent_results in latent_saturation.items():
        log.info("Latent saturation diagnostics: %dC", temp_c)
        for latent_name, values in latent_results.items():
            sat_time = (
                f"{values['median_first_saturation_h']:.1f}"
                if np.isfinite(values["median_first_saturation_h"])
                else "none"
            )
            log.info(
                "  %-3s mean=%.3f max=%.3f fraction(z>=0.95)=%.3f pre100h=%.3f median first saturation=%s h",
                latent_name,
                values["mean"],
                values["max"],
                values["fraction_ge_095"],
                values.get("fraction_ge_095_pre100h", float("nan")),
                sat_time,
            )

    zc_profile = compute_zc_time_profile(z_all, TK_all, t_all, mask_all)
    for temp_c, profile in zc_profile.items():
        for h in [0, 1, 5, 20, 100, 500, 1000, 2000]:
            vals = profile.get(h, {})
            log.info(
                "zC stats @ %dC t=%4dh | n=%d mean=%.4f std=%.4f p5=%.4f p50=%.4f p95=%.4f",
                temp_c,
                h,
                int(vals.get("n", 0)),
                float(vals.get("mean", float("nan"))),
                float(vals.get("std", float("nan"))),
                float(vals.get("p5", float("nan"))),
                float(vals.get("p50", float("nan"))),
                float(vals.get("p95", float("nan"))),
            )

    # ---- Debug7 diagnostics: freeze-zC ablation and identifiability ----
    z_ref_all = z_prefix_all[:, 0, :]
    freeze_diag = compute_freeze_zc_ablation(
        model=model,
        z_traj=z_all,
        z_ref=z_ref_all,
        x_true_norm=x_true_all,
        mask=mask_all,
        norm_stats=dataset["norm_stats"],
        prefix_len=prefix_len,
    )
    log.info(
        "Freeze-zC ablation | overall RMSE full=%.5f frozen=%.5f | zC contribution skill=%.4f",
        freeze_diag["rmse_full"]["overall"],
        freeze_diag["rmse_frozen_zc"]["overall"],
        freeze_diag["zc_contribution_skill_overall"],
    )

    rmse_loglin = compute_rmse(baseline_loglin, x_true_deg, future_mask_all)
    for fname in cfg.FEATURES:
        log.info(
            "Feature diagnostic %-6s | model_future_rmse=%.5f | logtime_future_rmse=%.5f | skill_vs_logtime=%.4f | frozen_zc_rmse=%.5f | zc_contrib_skill=%.4f",
            fname,
            overall_rmse.get(fname, float("nan")),
            rmse_loglin.get(fname, float("nan")),
            skill["logtime_linear"].get(fname, float("nan")),
            freeze_diag["rmse_frozen_zc"].get(fname, float("nan")),
            freeze_diag["zc_contribution_skill_by_feature"].get(fname, float("nan")),
        )

    latent_corr = compute_future_latent_correlations(z_all, mask_all, prefix_len)
    log.info("Future latent correlations with zC: %s", {
        k: ("nan" if not np.isfinite(v) else f"{v:.4f}")
        for k, v in latent_corr.items()
    })

    zc_r2 = compute_zc_logtime_r2(z_all, t_all, TK_all, mask_all, prefix_len)
    log.info("zC ~ [1, log(1+t), 1/T] future R2: %.5f", zc_r2)

    # ---- Alpha identifiability (Step 3) ----
    alpha_id = check_alpha_identifiability(model, dataset, device, prefix_len=prefix_len)
    if not alpha_id["is_identifiable"]:
        log.warning(
            "Alpha may not be identifiable (Spearman=%.3f). "
            "Consider setting cfg.GENERATOR_FIXED_ALPHA=True before Stage 4/5.",
            alpha_id["spearman_corr_overall"],
        )

    zc_contrib = compute_decoder_zc_contribution(
        model=model,
        z_traj=z_all,
        z_ref=z_ref_all,
        mask=mask_all,
        prefix_len=prefix_len,
    )
    zc_weights = zc_contrib.pop("_decoder_zc_weights", {})
    leakage_routes = summarize_decoder_leakage_routes(model)
    log.info("Decoder zC column weights: %s", {
        k: f"{v:.5f}" for k, v in zc_weights.items()
    })
    if leakage_routes:
        log.info("Leakage branch ratios (reversible vs zL): %s", {
            k: f"{v:.4f}" for k, v in leakage_routes.items()
        })
    for fname in cfg.FEATURES:
        vals = zc_contrib.get(fname, {})
        log.info(
            "Decoder zC contribution %-6s | abs=%.6f rel=%.4f",
            fname,
            float(vals.get("abs_contribution", float("nan"))),
            float(vals.get("relative_contribution", float("nan"))),
        )

    # ---- Sensitivity heatmap ----
    sens_result = counterfactual_intervention(
        model,
        torch.full((1, cfg.LATENT_DIM), 0.3),
        T_K[:1] if len(all_TK) > 0 else torch.tensor([573.15]),
        model.decoder,
    )
    plot_sensitivity_heatmap(
        sens_result["sensitivity"],
        save_path=os.path.join(cfg.FIGURES_DIR, "decoder_sensitivity.png"))

    # ---- Latent trajectory plots ----
    plot_latent_trajectories(
        z_all, TK_all, mask_all, t_all,
        save_path=os.path.join(cfg.FIGURES_DIR, "latent_trajectories.png"))

    # ---- Representative device reconstruction ----
    idx0 = 0
    plot_reconstruction(
        x_pred_all[idx0], x_true_all[idx0], mask_all[idx0], t_all[idx0],
        title="Test device #0 reconstruction",
        save_path=os.path.join(cfg.FIGURES_DIR, "device0_reconstruction.png"))

    # ---- Phase 2: Stochastic residual evaluation ----
    stoch_metrics = None
    try:
        import importlib.util as _ilu
        _stoch_spec = _ilu.spec_from_file_location(
            "_pi_stoch",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "10_stochastic_residual.py"),
        )
        _stoch = _ilu.module_from_spec(_stoch_spec)
        _stoch_spec.loader.exec_module(_stoch)

        # Build train-set predictions using the same inline dataset class
        train_idx_s = split["train"]
        train_ds_s  = test_ds.__class__(dataset, train_idx_s)
        train_dl_s  = DataLoader(train_ds_s, batch_size=16, shuffle=False,
                                 collate_fn=lambda b: {k: torch.stack([i[k] for i in b])
                                                        for k in b[0]})
        tr_pred, tr_true, tr_mask, tr_TK, tr_t = [], [], [], [], []
        model.eval()
        with torch.no_grad():
            for _b in train_dl_s:
                _out = predict_from_prefix(
                    model, _b["enc_input"], _b["x"], _b["mask"],
                    _b["times_h"], _b["T_K"], _b["x0"], prefix_len, device)
                tr_pred.append(_out["x_pred"].numpy())
                tr_true.append(_b["x"].numpy())
                tr_mask.append(_b["mask"].numpy())
                tr_TK.append(_b["T_K"].numpy())
                tr_t.append(_b["times_h"].numpy())
        x_tr_pred_deg = denormalize_x(np.concatenate(tr_pred, 0), dataset["norm_stats"])
        x_tr_true_deg = denormalize_x(np.concatenate(tr_true, 0), dataset["norm_stats"])
        mk_tr = np.concatenate(tr_mask, 0)
        TK_tr = np.concatenate(tr_TK,  0)
        t_tr  = np.concatenate(tr_t,   0)

        log.info("Phase 2 stochastic residual evaluation …")
        n_stoch = min(50, int(getattr(cfg, "STOCH_RESIDUAL_N_SAMPLES", 100)))
        stoch_metrics = _stoch.evaluate_stochastic_residual(
            model=model,
            x_pred_deg=x_pred_deg,
            x_true_deg=x_true_deg,
            T_K=TK_all,
            times_h=t_all,
            mask=mask_all,
            train_idx=train_idx_s,
            x_train_pred=x_tr_pred_deg,
            x_train_true=x_tr_true_deg,
            T_K_train=TK_tr,
            times_train=t_tr,
            mask_train=mk_tr,
            prefix_len=prefix_len,
            n_samples=n_stoch,
            run_calibration=True,
        )
    except Exception as _stoch_err:
        log.warning("Phase 2 stochastic residual skipped: %s", _stoch_err)

    # ---- Save numeric results ----
    results = {
        "overall_rmse": overall_rmse,
        "prefix_rmse": prefix_rmse,
        "prefix_rmse_normalized": prefix_rmse_norm,
        "prefix_macro_rmse_normalized": prefix_macro_rmse_norm,
        "overall_mae": overall_mae,
        "valid_point_count": valid_count,
        "horizon_rmse": horizon_rmse,
        "structural":   struct,
        "counterfactual_temperature_ordering": temp_cf,
        "initial_latent_stats": init_latent_stats,
        "zero_time_abs_mean": zero_t,
        "svth_diagnostics": {
            "s_vth": s_vth,
            "std_delta_vth_train": split_stats.get("train", float("nan")),
            "std_delta_vth_val": split_stats.get("val", float("nan")),
            "std_delta_vth_test": split_stats.get("test", float("nan")),
        },
        "skill_scores": skill,
        "nrmse_r2": nrmse_r2,
        "multiplicative_error_factor": mult_factor,
        "rmse_by_temperature": rmse_by_temp,
        "rmse_by_time_interval": rmse_by_interval,
        "rmse_by_device": rmse_by_device,
        "physical_metrics": physical_metrics,
        "feature_distribution": feature_dist,
        "latent_saturation": latent_saturation,
        "zc_time_profile": zc_profile,
        "freeze_zc_ablation": freeze_diag,
        "zC_future_latent_correlations": latent_corr,
        "zC_logtime_r2_future": zc_r2,
        "decoder_zc_contribution": zc_contrib,
        "decoder_zc_weights": zc_weights,
        "alpha_identifiability": alpha_id,
        "stochastic_residual": stoch_metrics,
    }
    fname = "evaluation_results.pkl" if results_tag is None else f"evaluation_results_{results_tag}.pkl"
    result_path = os.path.join(cfg.RESULTS_DIR, fname)
    with open(result_path, "wb") as f:
        pickle.dump(results, f)
    log.info("Results saved → %s", result_path)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Dynamic imports
    from importlib.util import spec_from_file_location, module_from_spec
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def _load(alias, filename):
        spec = spec_from_file_location(alias, os.path.join(base_dir, filename))
        mod = module_from_spec(spec); sys.modules[alias] = mod
        spec.loader.exec_module(mod); return mod

    _load("_enc",  "03_model_encoder.py")
    _load("_dec",  "04_model_decoder.py")
    _load("_ode",  "02_physics_latent.py")
    _load("_gen",  "05_model_generator.py")
    _load("_disc", "06_model_discriminator.py")
    _load("_prep", "01_data_preprocessing.py")

    import _enc  as enc_mod
    import _dec  as dec_mod
    import _ode  as ode_mod
    import _gen  as gen_mod
    import _disc as disc_mod
    import _prep as prep_mod

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = enc_mod.PhysicsEncoder()
            self.decoder   = dec_mod.SparsePhysicsDecoder()
            self.ode       = ode_mod.PhysicsODE()
            self.alpha_net = ode_mod.DeviceAlphaNet()
            self.generator = gen_mod.PITimeGANGenerator()
            self.disc      = disc_mod.PITimeGANDiscriminator()

    # Load data
    dataset = prep_mod.load_dataset()

    # Load model
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "stage5_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "final_model.pt")
    model = load_model(ckpt_path, _Model)

    run_evaluation(model, dataset)
