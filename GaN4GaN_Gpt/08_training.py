"""
08_training.py
==============
Five-stage training loop for the PI-TimeGAN reliability prediction model.

Stage 1  (AE)      : Train encoder + decoder as a constrained autoencoder.
                     Losses: recon + bounds + monotone + temp_order.
Stage 2  (ODE)     : Freeze decoder, add ODE consistency loss.
                     Losses: recon + ode + bounds + mono + temp.
Stage 3  (MultiStep): Full physics training with multi-step prediction.
                     Losses: recon + ode + multistep + bounds + mono + temp.
Stage 4  (GenPre)  : Pre-train generator by distribution matching to real z.
                     No adversarial loss yet.
Stage 5  (GAN)     : Adversarial fine-tuning with small GAN weight.
                     All losses + small adversarial term.

Data handling
-------------
The dataset contains a ragged set of sequences: different devices have
observations at different time points (mask indicates validity).
Batching is done by device (all time steps of a device in one sample).
"""

import os
import sys
import time
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config as cfg

# Loss functions are injected at runtime by main.py (or the __main__ block
# below).  We declare module-level variables here; they are filled before any
# training function is called.
reconstruction_loss         = None
balanced_feature_huber_loss = None
ode_residual_loss           = None
bounds_loss                 = None
monotonicity_loss           = None
temperature_ordering_loss   = None
zc_prefix_separation_loss    = None
z_phys_rank_loss            = None
multistep_prediction_loss   = None
adversarial_generator_loss  = None
adversarial_discriminator_loss = None
distribution_matching_loss  = None
total_physics_loss          = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset  (wraps the preprocessed numpy arrays)
# ---------------------------------------------------------------------------

class DeviceDegradationDataset(Dataset):
    """
    Torch Dataset wrapping the preprocessed dataset dict.

        Each item: dict with keys:
            enc_input (T, 15), x (T, 6), feature_mask (T, 6), mask (T,),
            times_h (T,), T_K (), x0 (6,)
    """

    def __init__(self, dataset: dict, indices: np.ndarray):
        self.x         = torch.from_numpy(dataset["x"]).float()          # (N,T,6)
        self.feature_mask = torch.from_numpy(dataset["feature_mask"]).bool()  # (N,T,6)
        self.mask      = torch.from_numpy(dataset["mask"]).bool()        # (N,T)
        self.times_h   = torch.from_numpy(dataset["times_h"]).float()    # (N,T)
        self.T_K       = torch.from_numpy(dataset["T_K"]).float()        # (N,)
        x0_key = "x0_normalized" if "x0_normalized" in dataset else "x0_static"
        self.x0_static = torch.from_numpy(dataset[x0_key]).float()        # (N,6)
        leakage_floor = dataset.get("leakage_floor", {}) or {}
        self.leakage_floor = torch.tensor([
            float(leakage_floor.get("IDLeak", cfg.LEAKAGE_FLOOR_DEFAULT)),
            float(leakage_floor.get("IGLeak", cfg.LEAKAGE_FLOOR_DEFAULT)),
        ], dtype=torch.float32)
        self.indices   = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x       = self.x[idx]           # (T,6)
        feature_mask = self.feature_mask[idx]  # (T,6)
        mask    = self.mask[idx]         # (T,)
        times_h = self.times_h[idx]     # (T,)
        T_K     = self.T_K[idx]         # scalar
        x0      = self.x0_static[idx]   # (6,)

        # Build encoder input: [x(6), feature_mask(6), T_norm, log_t, delta_log_t]
        T_norm = torch.tensor((T_K.item() - cfg.T_REF_K) / cfg.T_REF_K)
        T_feat = T_norm.expand(x.shape[0], 1)                  # (T,1)
        log_t  = torch.log(times_h + 1.0)
        dlt    = torch.zeros_like(log_t)
        dlt[:-1] = log_t[1:] - log_t[:-1]
        log_t_feat = log_t.unsqueeze(1)                         # (T,1)
        dlt_feat = dlt.unsqueeze(1)                             # (T,1)
        enc_input = torch.cat([
            torch.nan_to_num(x, nan=0.0),
            feature_mask.float(),
            T_feat,
            log_t_feat,
            dlt_feat,
        ], dim=1)                                               # (T,15)

        return {
            "enc_input": enc_input,
            "x":         x,
            "feature_mask": feature_mask,
            "mask":      mask,
            "times_h":   times_h,
            "T_K":       T_K,
            "x0":        x0,
            "leakage_floor": self.leakage_floor,
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _get_device() -> torch.device:
    if cfg.TORCH_DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sanitize_nonfinite_gradients(params) -> int:
    """
    Replace NaN/Inf gradients with finite values in-place.

    Returns:
        Number of parameter tensors that contained any non-finite gradient.
    """
    fixed = 0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        if not torch.isfinite(g).all():
            fixed += 1
            p.grad = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return fixed


class EarlyStopping:
    def __init__(self, patience: int = cfg.EARLY_STOPPING_PATIENCE,
                 min_delta: float = 1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0
        self.stop      = False

    def step(self, val_loss: float):
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def _save_checkpoint(model, opt_dict: dict, stage: int, epoch: int,
                     val_loss: float, path: str,
                     selection_metric: str = "val_loss",
                     extra_metrics: Optional[Dict[str, float]] = None):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "stage": stage,
        "epoch": epoch,
        "val_loss": val_loss,
        "selection_metric": selection_metric,
        "selection_value": val_loss,
        "model_state": model.state_dict(),
        "opt_states": {k: o.state_dict() for k, o in opt_dict.items()},
    }
    if extra_metrics:
        ckpt.update(extra_metrics)
    fname = os.path.join(path, f"stage{stage}_best.pt")
    torch.save(ckpt, fname)
    log.info("  Checkpoint saved → %s  (%s=%.5f)",
             fname, selection_metric, val_loss)


def _load_checkpoint(model, path: str, stage: int) -> Optional[dict]:
    fname = os.path.join(path, f"stage{stage}_best.pt")
    if not os.path.exists(fname):
        return None
    ckpt = torch.load(fname, map_location="cpu")
    model_state = dict(ckpt["model_state"])
    # Keep decoder sparsity from current config instead of checkpoint buffer.
    model_state.pop("decoder.mask", None)
    model.load_state_dict(model_state, strict=False)
    metric = ckpt.get("selection_metric", "val_loss")
    value = ckpt.get("selection_value", ckpt.get("val_loss", float("nan")))
    log.info("  Loaded checkpoint: %s  (%s=%.5f)", fname, metric, value)
    return ckpt


# ---------------------------------------------------------------------------
# Forward pass helper
# ---------------------------------------------------------------------------

def _forward(model, batch: dict, device: torch.device):
    """
    Run a full forward pass (encoder → alpha → decoder) and return
    intermediate tensors.
    """
    enc_input    = batch["enc_input"].to(device)     # (B,T,15)
    x_true       = batch["x"].to(device)             # (B,T,6)
    feature_mask = batch["feature_mask"].to(device)  # (B,T,6)
    mask         = batch["mask"].to(device)          # (B,T)
    times_h      = batch["times_h"].to(device)       # (B,T)
    T_K          = batch["T_K"].to(device)           # (B,)
    x0           = batch["x0"].to(device)            # (B,6)

    # Encoder
    z_enc, _     = model.encoder(enc_input, mask)    # (B,T,5)

    # Per-device alpha
    alpha        = model.alpha_net(x0, T_K)          # (B,)

    # Decode relative to device-specific initial latent state so x(z_ref)=0.
    z_ref = z_enc[:, 0, :]
    x_hat = model.decoder(z_enc, z_ref=z_ref)        # (B,T,6)

    return z_enc, alpha, x_hat, x_true, feature_mask, mask, times_h, T_K, x0


def _log_stage3_physics_summary(model, alpha_stats: dict, alpha_by_temp: dict):
    """Log Arrhenius/rate diagnostics to verify temperature separation."""
    with torch.no_grad():
        ode = model.ode
        temps = torch.tensor([
            275.0 + cfg.CELSIUS_TO_KELVIN,
            300.0 + cfg.CELSIUS_TO_KELVIN,
            325.0 + cfg.CELSIUS_TO_KELVIN,
        ], dtype=torch.float32, device=next(ode.parameters()).device)
        frev = ode._arrhenius(ode.Ea_rev, temps).detach().cpu().numpy()
        firr = ode._arrhenius(ode.Ea_irrev, temps).detach().cpu().numpy()

        log.info(
            "  ODE params | Ea_rev=%.4f Ea_irrev=%.4f | kGc=%.4f kGe=%.4f kBc=%.4f kBe=%.4f kM=%.4f kL=%.4f kC=%.4f aLG=%.4f aLB=%.4f",
            float(ode.Ea_rev.detach().cpu()),
            float(ode.Ea_irrev.detach().cpu()),
            float(ode.kGc.detach().cpu()),
            float(ode.kGe.detach().cpu()),
            float(ode.kBc.detach().cpu()),
            float(ode.kBe.detach().cpu()),
            float(ode.kM.detach().cpu()),
            float(ode.kL.detach().cpu()),
            float(ode.kC.detach().cpu()),
            float(ode.aLG.detach().cpu()),
            float(ode.aLB.detach().cpu()),
        )
        log.info(
            "  Arrhenius frev 275/300/325C = %.4f / %.4f / %.4f | firrev 275/300/325C = %.4f / %.4f / %.4f",
            float(frev[0]), float(frev[1]), float(frev[2]),
            float(firr[0]), float(firr[1]), float(firr[2]),
        )
        log.info(
            "  Alpha stats | mean=%.4f std=%.4f min=%.4f max=%.4f",
            alpha_stats["mean"], alpha_stats["std"], alpha_stats["min"], alpha_stats["max"]
        )
        parts = []
        for tc in cfg.TEMPERATURES_C:
            entry = alpha_by_temp.get(tc)
            if entry is None:
                continue
            parts.append(f"{tc}C mean={entry['mean']:.4f} std={entry['std']:.4f} n={entry['n']}")
        if parts:
            log.info("  Alpha by temperature | %s", " | ".join(parts))


def validation_prefix_rollout_mse(model, val_dl, device,
                                  prefix_len: int = cfg.STAGE3_PREFIX_LEN) -> float:
    """
    Future-only open-loop rollout MSE for checkpoint selection.

    The model can only observe the prefix; scoring excludes prefix points.
    """
    model.eval()
    total_squared_error = 0.0
    total_count = 0.0

    with torch.no_grad():
        for batch in val_dl:
            batch = _move_batch(batch, device)
            enc_input = batch["enc_input"]
            x_true = batch["x"]
            feature_mask = batch["feature_mask"]
            mask = batch["mask"]
            times_h = batch["times_h"]
            T_K = batch["T_K"]
            x0 = batch["x0"]

            if enc_input.shape[1] <= prefix_len:
                continue

            z_prefix, _ = model.encoder(enc_input[:, :prefix_len, :],
                                        mask[:, :prefix_len])

            batch_size = enc_input.shape[0]
            latent_dim = z_prefix.shape[-1]

            prefix_valid = mask[:, :prefix_len].any(dim=1)
            if not prefix_valid.any():
                continue

            last_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            for b in range(batch_size):
                valid_steps = mask[b, :prefix_len].nonzero(as_tuple=False)
                if len(valid_steps) > 0:
                    last_indices[b] = valid_steps[-1, 0]

            gather_idx = last_indices.view(batch_size, 1, 1).expand(batch_size, 1, latent_dim)
            z_start = z_prefix.gather(1, gather_idx).squeeze(1)
            alpha = model.alpha_net(x0, T_K)

            t_future = times_h[:, prefix_len - 1:]
            z_future = model.ode.integrate_trajectory(z_start, T_K, t_future, alpha)
            z_ref = z_prefix[:, 0, :]
            x_future_pred = model.decoder(z_future, z_ref=z_ref)[:, 1:, :]
            x_future_true = x_true[:, prefix_len:, :]
            mask_future = mask[:, prefix_len:]

            feature_mask_future = feature_mask[:, prefix_len:, :]
            feature_valid = (
                feature_mask_future
                & mask_future[:, :, None]
                & prefix_valid[:, None, None]
                & torch.isfinite(x_future_true)
                & torch.isfinite(x_future_pred)
            )
            if feature_valid.any():
                error = (x_future_pred - torch.nan_to_num(x_future_true, nan=0.0)) ** 2
                total_squared_error += (error * feature_valid.float()).sum().item()
                total_count += feature_valid.float().sum().item()

    if total_count == 0:
        return float("inf")
    return total_squared_error / total_count


# ---------------------------------------------------------------------------
# Stage 1: Constrained Autoencoder
# ---------------------------------------------------------------------------

def train_stage1(model, train_dl, val_dl, device):
    """
    Train encoder + decoder + alpha_net as a constrained autoencoder.
    ODE is NOT included here; physics priors applied via bounds, mono, temp.
    """
    log.info("\n=== Stage 1: Constrained Autoencoder ===")
    params = (
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.alpha_net.parameters())
    )
    opt = torch.optim.Adam(params, lr=cfg.LR_STAGE1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=20, factor=0.5, min_lr=1e-5)
    es = EarlyStopping()

    best_val = float("inf")
    for epoch in range(1, cfg.EPOCHS_STAGE1 + 1):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            z_enc, alpha, x_hat, x_true, feature_mask, mask, times_h, T_K, x0 = \
                _forward(model, batch, device)
            loss_dict = total_physics_loss(
                x_hat, x_true, z_enc, T_K, times_h, alpha,
                mask, model.ode, feature_mask=feature_mask, include_temp_order=False,
                leakage_floor=batch.get("leakage_floor"),
            )
            # Debug5: Stage1 should prioritize reconstruction quality.
            loss = (
                cfg.LAMBDA_RECON * loss_dict["recon"]
                + cfg.STAGE1_LAMBDA_ANCHOR * loss_dict["anchor"]
                + cfg.STAGE1_LAMBDA_SMOOTH * loss_dict["mono"]
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP_NORM)
            opt.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                z_enc, alpha, x_hat, x_true, feature_mask, mask, times_h, T_K, x0 = \
                    _forward(model, batch, device)
                vl = balanced_feature_huber_loss(
                    x_hat,
                    x_true,
                    feature_mask,
                    beta=cfg.LOSS_HUBER_BETA,
                )
                val_loss += vl.item()

        train_loss /= max(len(train_dl), 1)
        val_loss   /= max(len(val_dl), 1)
        sched.step(val_loss)

        if epoch % 20 == 0:
            log.info("  Stage1 Epoch %3d | train=%.5f | val=%.5f",
                     epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(model, {"opt_s1": opt}, 1, epoch, val_loss,
                             cfg.CHECKPOINT_DIR)
        es.step(val_loss)
        if es.stop:
            log.info("  Early stopping at epoch %d", epoch)
            break

    _load_checkpoint(model, cfg.CHECKPOINT_DIR, 1)
    log.info("Stage 1 done. Best val loss = %.5f", best_val)


# ---------------------------------------------------------------------------
# Stage 2: Add ODE consistency
# ---------------------------------------------------------------------------

def train_stage2(model, train_dl, val_dl, device):
    log.info("\n=== Stage 2: ODE Consistency ===")
    params = (
        list(model.ode.parameters()) +
        list(model.alpha_net.parameters())
    )
    opt = torch.optim.Adam(params, lr=cfg.LR_STAGE2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=20, factor=0.5, min_lr=1e-5)
    es = EarlyStopping()

    best_val = float("inf")
    for epoch in range(1, cfg.EPOCHS_STAGE2 + 1):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            z_enc, alpha, x_hat, x_true, feature_mask, mask, times_h, T_K, x0 = \
                _forward(model, batch, device)
            loss_dict = total_physics_loss(
                x_hat, x_true, z_enc, T_K, times_h, alpha,
                mask, model.ode, feature_mask=feature_mask,
            )
            loss = loss_dict["total"]
            opt.zero_grad()
            loss.backward()
            n_fixed = _sanitize_nonfinite_gradients(params)
            if n_fixed > 0:
                log.warning("  Stage2: sanitized non-finite gradients in %d tensors", n_fixed)
            torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP_NORM)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                z_enc, alpha, x_hat, x_true, feature_mask, mask, times_h, T_K, x0 = \
                    _forward(model, batch, device)
                ld = total_physics_loss(
                    x_hat, x_true, z_enc, T_K, times_h, alpha,
                    mask, model.ode, feature_mask=feature_mask)
                val_loss += ld["total"].item()

        train_loss /= max(len(train_dl), 1)
        val_loss   /= max(len(val_dl), 1)
        sched.step(val_loss)

        if epoch % 20 == 0:
            log.info("  Stage2 Epoch %3d | train=%.5f | val=%.5f",
                     epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(model, {"opt_s2": opt}, 2, epoch, val_loss,
                             cfg.CHECKPOINT_DIR)
        es.step(val_loss)
        if es.stop:
            log.info("  Early stopping at epoch %d", epoch)
            break

    _load_checkpoint(model, cfg.CHECKPOINT_DIR, 2)
    log.info("Stage 2 done. Best val loss = %.5f", best_val)


# ---------------------------------------------------------------------------
# Stage 3: Multi-step prediction
# ---------------------------------------------------------------------------

def train_stage3(model, train_dl, val_dl, device):
    log.info("\n=== Stage 3: Multi-step Prediction ===")
    params = (
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.ode.parameters()) +
        list(model.alpha_net.parameters())
    )
    opt = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": cfg.LR_STAGE3},
        {"params": model.ode.parameters(), "lr": cfg.LR_STAGE3},
        {"params": model.alpha_net.parameters(), "lr": cfg.LR_STAGE3},
        {"params": model.decoder.parameters(), "lr": cfg.LR_STAGE3 * cfg.STAGE3_DECODER_LR_SCALE},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=15, factor=0.5, min_lr=1e-5)
    es = EarlyStopping()

    best_forecast_mse = float("inf")
    for epoch in range(1, cfg.EPOCHS_STAGE3 + 1):
        model.train()
        train_loss = 0.0
        train_ld_total = 0.0
        train_ms_total = 0.0
        train_zc_sep_total = 0.0
        train_leak_total = 0.0
        alpha_all = []
        temp_all = []
        for batch in train_dl:
            z_enc, alpha, x_hat, x_true, feature_mask, mask, times_h, T_K, x0 = \
                _forward(model, batch, device)
            alpha_all.append(alpha.detach())
            temp_all.append(T_K.detach())

            possible_prefix_lens = [p for p in [2, 3, 4, 5, 6] if p < z_enc.shape[1]]
            if len(possible_prefix_lens) == 0:
                start_step = max(0, min(cfg.STAGE3_PREFIX_LEN - 1, z_enc.shape[1] - 2))
            else:
                probs_map = {2: 0.10, 3: 0.15, 4: 0.45, 5: 0.15, 6: 0.15}
                probs = np.array([probs_map[p] for p in possible_prefix_lens], dtype=float)
                probs = probs / probs.sum()
                prefix_len = int(np.random.choice(possible_prefix_lens, p=probs))
                start_step = prefix_len - 1

            loss_dict = total_physics_loss(
                x_hat, x_true, z_enc, T_K, times_h, alpha,
                mask, model.ode, feature_mask=feature_mask,
            )
            # Future rollout term — up-weight 325°C devices to reduce the 2x RMSE gap.
            future_loss = multistep_prediction_loss(
                model.decoder, model.ode, z_enc, x_true,
                T_K, times_h, alpha, mask,
                feature_mask=feature_mask,
                start_step=start_step,
                decay=1.0,
                leakage_floor=batch.get("leakage_floor"),
            )
            _high_T_K = (cfg.STAGE3_HIGH_TEMP_C + cfg.CELSIUS_TO_KELVIN) if hasattr(cfg, "STAGE3_HIGH_TEMP_C") else 1e9
            _high_w   = float(getattr(cfg, "STAGE3_HIGH_TEMP_LOSS_WEIGHT", 1.0))
            if _high_w > 1.0 and T_K.max().item() >= _high_T_K - 0.5:
                # Scale up the loss contribution for 325°C batches / mixed batches
                _t_frac = (T_K >= _high_T_K - 0.5).float().mean()
                future_loss = future_loss * (1.0 + (_high_w - 1.0) * _t_frac)

            # Prefix reconstruction preservation term.
            prefix_loss = balanced_feature_huber_loss(
                model.decoder(z_enc[:, :prefix_len, :], z_ref=z_enc[:, 0, :]),
                x_true[:, :prefix_len, :],
                feature_mask[:, :prefix_len, :],
                beta=cfg.LOSS_HUBER_BETA,
            )

            zc_sep_loss = zc_prefix_separation_loss(
                z_enc,
                mask,
                prefix_len=prefix_len,
                margin=cfg.STAGE3_ZC_PREFIX_MARGIN,
            )

            # z_phys residual-rank loss (07_losses.py::z_phys_rank_loss): off
            # by default (LAMBDA_ZPHYS_RANK=0). Uses a light independent
            # ODE+decoder rollout (not multistep_prediction_loss's internal
            # one) purely to score per-device future-residual magnitude for
            # ranking — detached, so it does not add a second gradient path
            # into the ODE/decoder, only into the encoder via z_pfx and the
            # small resid_rank_proj head.
            _lambda_zrank = float(getattr(cfg, "LAMBDA_ZPHYS_RANK", 0.0))
            zrank_loss = torch.zeros(1, device=z_enc.device).squeeze()
            if _lambda_zrank > 0:
                with torch.no_grad():
                    z_pfx_nograd = z_enc[:, start_step, :].detach()
                    z_future_nograd = model.ode.integrate_trajectory(
                        z_pfx_nograd, T_K, times_h[:, start_step:], alpha,
                    )
                    x_future_hat = model.decoder(z_future_nograd, z_ref=z_enc[:, 0, :].detach())
                    resid = x_true[:, start_step:, :] - x_future_hat
                    fut_mask = mask[:, start_step:].unsqueeze(-1).float()
                    resid = torch.nan_to_num(resid, nan=0.0) * fut_mask
                    n_valid = fut_mask.sum(dim=(1, 2)).clamp(min=1.0)
                    resid_std = (resid.pow(2).sum(dim=(1, 2)) / n_valid).sqrt()
                    resid_magnitude = torch.log1p(resid_std)
                zrank_loss = z_phys_rank_loss(
                    z_enc[:, start_step, :], resid_magnitude,
                    model.encoder.resid_rank_proj,
                    margin=float(getattr(cfg, "ZPHYS_RANK_MARGIN", 0.1)),
                )

            # Soft upper-bound penalty for kM and kL rate constants (debug13)
            # Penalise only when the rate constants exceed KM_MAX/KL_MAX (soft wall).
            _km_max = float(getattr(cfg, "KM_MAX", 1e9))
            _kl_max = float(getattr(cfg, "KL_MAX", 1e9))
            _kM = model.ode.kM
            _kL = model.ode.kL
            km_penalty = torch.relu(_kM - _km_max) ** 2
            kl_penalty = torch.relu(_kL - _kl_max) ** 2
            rate_bound_loss = km_penalty + kl_penalty

            loss = (
                cfg.STAGE3_LAMBDA_FUTURE * future_loss
                + cfg.STAGE3_LAMBDA_PREFIX * prefix_loss
                + cfg.STAGE3_LAMBDA_ODE * loss_dict["ode"]
                + cfg.STAGE3_LAMBDA_ANCHOR * loss_dict["anchor"]
                + cfg.STAGE3_LAMBDA_ZC_SEPARATION * zc_sep_loss
                + cfg.STAGE3_LAMBDA_LEAKAGE * loss_dict["leakage"]
                + 10.0 * rate_bound_loss
                + _lambda_zrank * zrank_loss
            )

            opt.zero_grad()
            loss.backward()
            n_fixed = _sanitize_nonfinite_gradients(params)
            if n_fixed > 0:
                log.warning("  Stage3: sanitized non-finite gradients in %d tensors", n_fixed)
            torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP_NORM)
            opt.step()
            train_loss += loss.item()
            train_ld_total += loss_dict["total"].item()
            train_ms_total += future_loss.item()
            train_zc_sep_total += zc_sep_loss.item()
            train_leak_total += loss_dict["leakage"].item()

        model.eval()
        val_objective = 0.0
        val_ld_total = 0.0
        val_ms_total = 0.0
        val_zc_sep_total = 0.0
        val_leak_total = 0.0
        with torch.no_grad():
            for batch in val_dl:
                z_enc, alpha, x_hat, x_true, feature_mask, mask, times_h, T_K, x0 = \
                    _forward(model, batch, device)
                ld = total_physics_loss(
                    x_hat, x_true, z_enc, T_K, times_h, alpha,
                    mask, model.ode, feature_mask=feature_mask)
                val_prefix_len = max(2, min(cfg.STAGE3_PREFIX_LEN, z_enc.shape[1] - 1))
                fixed_start = max(0, min(val_prefix_len - 1, z_enc.shape[1] - 2))
                ms = multistep_prediction_loss(
                    model.decoder, model.ode, z_enc, x_true,
                    T_K, times_h, alpha, mask,
                    feature_mask=feature_mask,
                    start_step=fixed_start,
                    decay=1.0)
                val_prefix = balanced_feature_huber_loss(
                    model.decoder(z_enc[:, :val_prefix_len, :], z_ref=z_enc[:, 0, :]),
                    x_true[:, :val_prefix_len, :],
                    feature_mask[:, :val_prefix_len, :],
                    beta=cfg.LOSS_HUBER_BETA,
                )
                val_zc_sep = zc_prefix_separation_loss(
                    z_enc,
                    mask,
                    prefix_len=val_prefix_len,
                    margin=cfg.STAGE3_ZC_PREFIX_MARGIN,
                )
                val_objective += (
                    cfg.STAGE3_LAMBDA_FUTURE * ms
                    + cfg.STAGE3_LAMBDA_PREFIX * val_prefix
                    + cfg.STAGE3_LAMBDA_ODE * ld["ode"]
                    + cfg.STAGE3_LAMBDA_ANCHOR * ld["anchor"]
                    + cfg.STAGE3_LAMBDA_ZC_SEPARATION * val_zc_sep
                    + cfg.STAGE3_LAMBDA_LEAKAGE * ld["leakage"]
                ).item()
                val_ld_total += ld["total"].item()
                val_ms_total += ms.item()
                val_zc_sep_total += val_zc_sep.item()
                val_leak_total += ld["leakage"].item()

        val_forecast_mse = validation_prefix_rollout_mse(
            model, val_dl, device, prefix_len=cfg.STAGE3_PREFIX_LEN)

        train_loss /= max(len(train_dl), 1)
        train_ld_total /= max(len(train_dl), 1)
        train_ms_total /= max(len(train_dl), 1)
        train_zc_sep_total /= max(len(train_dl), 1)
        train_leak_total /= max(len(train_dl), 1)
        val_objective /= max(len(val_dl), 1)
        val_ld_total /= max(len(val_dl), 1)
        val_ms_total /= max(len(val_dl), 1)
        val_zc_sep_total /= max(len(val_dl), 1)
        val_leak_total /= max(len(val_dl), 1)
        sched.step(val_forecast_mse)

        if alpha_all:
            alpha_cat = torch.cat(alpha_all, dim=0).float()
            temp_cat = torch.cat(temp_all, dim=0).float()
            alpha_stats = {
                "mean": float(alpha_cat.mean().item()),
                "std": float(alpha_cat.std(unbiased=False).item()),
                "min": float(alpha_cat.min().item()),
                "max": float(alpha_cat.max().item()),
            }
            alpha_by_temp = {}
            for tc in cfg.TEMPERATURES_C:
                tk = tc + cfg.CELSIUS_TO_KELVIN
                sel = torch.abs(temp_cat - tk) < 1.0
                if sel.any():
                    vals = alpha_cat[sel]
                    alpha_by_temp[tc] = {
                        "mean": float(vals.mean().item()),
                        "std": float(vals.std(unbiased=False).item()),
                        "n": int(vals.numel()),
                    }
        else:
            alpha_stats = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
            alpha_by_temp = {}

        if epoch % 20 == 0:
            log.info(
                "  Stage3 Epoch %3d | train=%.5f (ld=%.5f, ms=%.5f, zc=%.5f, leak=%.5f) | val=%.5f (ld=%.5f, ms=%.5f, zc=%.5f, leak=%.5f)",
                epoch,
                train_loss,
                train_ld_total,
                train_ms_total,
                train_zc_sep_total,
                train_leak_total,
                val_objective,
                val_ld_total,
                val_ms_total,
                val_zc_sep_total,
                val_leak_total,
            )
            log.info("  Stage3 Epoch %3d | val future-rollout MSE=%.5f",
                     epoch, val_forecast_mse)

        if epoch % 10 == 0:
            _log_stage3_physics_summary(model, alpha_stats, alpha_by_temp)

        if val_forecast_mse < best_forecast_mse:
            best_forecast_mse = val_forecast_mse
            _save_checkpoint(
                model,
                {"opt_s3": opt},
                3,
                epoch,
                val_forecast_mse,
                cfg.CHECKPOINT_DIR,
                selection_metric="future_rollout_mse",
                extra_metrics={
                    "val_training_objective": float(val_objective),
                    "val_loss": float(val_objective),
                },
            )
        es.step(val_forecast_mse)
        if es.stop:
            log.info("  Early stopping at epoch %d", epoch)
            break

    _load_checkpoint(model, cfg.CHECKPOINT_DIR, 3)
    log.info("Stage 3 done. Best future rollout MSE = %.5f", best_forecast_mse)


# ---------------------------------------------------------------------------
# Phase 4: Generative checkpoint-selection scoring for Stage 5
# ---------------------------------------------------------------------------

def _generative_val_score(
    model,
    val_dl,
    device,
    n_samples: int = None,
    prefix_len: int = None,
) -> float:
    """
    Compute the Stage 5 generative checkpoint selection score on the val set.

    Score = CRPS  +  λ_w1 * W1(Δx)  +  λ_cov * |Coverage90 - 0.9|  +  λ_phys * phys_violation
    Lower is better.

    CRPS uses the energy-score approximation:  E|Y-X| - 0.5*E|X-X'|
    W1(Δx) is approximated as |median_diff| + |std_diff| on increments.
    """
    import numpy as np

    if n_samples is None:
        n_samples = getattr(cfg, "STAGE5_N_GEN_SAMPLES", 20)
    if prefix_len is None:
        prefix_len = cfg.STAGE3_PREFIX_LEN

    model.eval()

    all_xt, all_mask, all_TK, all_ts = [], [], [], []
    all_xs: list = []          # each entry: list-of-S arrays (B, T, F)
    all_zf: list = []          # each entry: list-of-S arrays (B, T, 5)

    with torch.no_grad():
        for batch in val_dl:
            enc_input = batch["enc_input"].to(device)
            x_true    = batch["x"].to(device)
            mask      = batch["mask"].to(device)
            times_h   = batch["times_h"].to(device)
            T_K       = batch["T_K"].to(device)
            x0        = batch["x0"].to(device)

            # Prefix context for Phase 3 conditioning
            z_prefix, _ = model.encoder(enc_input[:, :prefix_len, :],
                                         mask[:, :prefix_len])
            z_prefix_last   = z_prefix[:, -1, :]
            log_prefix_time = torch.log1p(times_h[:, prefix_len - 1]).unsqueeze(1)
            z_ref = z_prefix[:, 0, :]

            xs_batch, zf_batch = [], []
            for _ in range(n_samples):
                z0_f, al_f = model.generator(
                    x0, T_K,
                    z_prefix_last=z_prefix_last,
                    log_prefix_time=log_prefix_time,
                )
                z_fake = model.ode.integrate_trajectory(z0_f, T_K, times_h, al_f)
                x_fake = model.decoder(z_fake, z_ref=z_ref)
                xs_batch.append(x_fake.cpu().numpy())
                zf_batch.append(z_fake.cpu().numpy())

            all_xt.append(x_true.cpu().numpy())
            all_mask.append(mask.cpu().numpy())
            all_TK.append(T_K.cpu().numpy())
            all_ts.append(times_h.cpu().numpy())
            # Stack (S, B, T, F) and append
            all_xs.append(np.stack(xs_batch, axis=0))
            all_zf.append(np.stack(zf_batch, axis=0))

    x_true_np = np.concatenate(all_xt,   axis=0)   # (N, T, F)
    mask_np   = np.concatenate(all_mask, axis=0)    # (N, T)
    # samples: concatenate along batch axis (axis=1)
    samples   = np.concatenate(all_xs, axis=1)      # (S, N, T, F)
    z_fakes   = np.concatenate(all_zf, axis=1)      # (S, N, T, 5)

    S, N, T, F = samples.shape
    future_mask = mask_np.copy().astype(bool)
    future_mask[:, :prefix_len] = False

    rng = np.random.default_rng(42)
    n_pairs = min(20, S)
    idx1 = rng.integers(0, S, n_pairs)
    idx2 = rng.integers(0, S, n_pairs)

    # ---- CRPS ----
    crps_list = []
    for fi in range(F):
        valid = future_mask & ~np.isnan(x_true_np[:, :, fi])
        if valid.sum() < 1:
            continue
        y   = x_true_np[:, :, fi][valid]
        xs_ = samples[:, :, :, fi][:, valid]          # (S, M)
        mae_term = float(np.mean(np.abs(xs_ - y[None, :]).mean(axis=1)))
        spread   = float(np.mean(np.abs(xs_[idx1] - xs_[idx2]).mean(axis=1)))
        crps_list.append(mae_term - 0.5 * spread)
    crps = float(np.mean(crps_list)) if crps_list else 0.0

    # ---- W1(Δx) ----
    w1_list = []
    for fi in range(F):
        td, pd = [], []
        for n in range(N):
            for t in range(1, T):
                if not future_mask[n, t] or not future_mask[n, t - 1]:
                    continue
                if np.isnan(x_true_np[n, t, fi]) or np.isnan(x_true_np[n, t - 1, fi]):
                    continue
                td.append(float(x_true_np[n, t, fi] - x_true_np[n, t - 1, fi]))
                for s in range(S):
                    pd.append(float(samples[s, n, t, fi] - samples[s, n, t - 1, fi]))
        if not td:
            continue
        td_a, pd_a = np.array(td), np.array(pd)
        w1_list.append(
            abs(float(np.median(pd_a)) - float(np.median(td_a))) +
            abs(float(np.std(pd_a))    - float(np.std(td_a)))
        )
    w1 = float(np.mean(w1_list)) if w1_list else 0.0

    # ---- Coverage 90 % ----
    p5  = np.percentile(samples,  5, axis=0)
    p95 = np.percentile(samples, 95, axis=0)
    cov_list = []
    for fi in range(F):
        valid = future_mask & ~np.isnan(x_true_np[:, :, fi])
        if valid.sum() < 1:
            continue
        in_int = (x_true_np[:, :, fi] >= p5[:, :, fi]) & (x_true_np[:, :, fi] <= p95[:, :, fi])
        cov_list.append(float(np.mean(in_int[valid])))
    coverage = float(np.mean(cov_list)) if cov_list else 0.5
    cov_penalty = abs(coverage - 0.9)

    # ---- Physics violations ----
    mono_v_list = []
    bounds_v_list = []
    for s in range(S):
        z = z_fakes[s]                    # (N, T, 5)
        for mi in cfg.MONOTONE_IDX:       # zM, zL, zC (see cfg.LATENT_NAMES)
            d = z[:, 1:, mi] - z[:, :-1, mi]
            mono_v_list.append(float(np.mean(d < -1e-4)))
        bounds_v_list.append(float(np.mean((z < 0.0) | (z > 1.0))))
    phys_viol = float(np.mean(mono_v_list) + np.mean(bounds_v_list))

    w1_w    = float(getattr(cfg, "STAGE5_W1_WEIGHT",       0.5))
    cov_w   = float(getattr(cfg, "STAGE5_COVERAGE_WEIGHT", 2.0))
    phys_w  = float(getattr(cfg, "STAGE5_PHYSICS_WEIGHT",  1.0))
    score   = crps + w1_w * w1 + cov_w * cov_penalty + phys_w * phys_viol

    log.info(
        "  GenScore=%.5f | CRPS=%.5f W1=%.5f CovPen=%.5f[cov=%.3f] PhysViol=%.5f",
        score, crps, w1, cov_penalty, coverage, phys_viol,
    )
    return score


# ---------------------------------------------------------------------------
# Stage 4: Generator pre-training (distribution matching)
# ---------------------------------------------------------------------------

def train_stage4(model, train_dl, val_dl, device):
    log.info("\n=== Stage 4: Generator Pre-training ===")
    # Freeze encoder, decoder, ODE
    for p in list(model.encoder.parameters()) + \
             list(model.decoder.parameters()) + \
             list(model.ode.parameters()):
        p.requires_grad_(False)

    params_gen = list(model.generator.parameters()) + \
                 list(model.alpha_net.parameters())
    opt_g = torch.optim.Adam(params_gen, lr=cfg.LR_STAGE4)
    es = EarlyStopping(patience=30)

    best_val = float("inf")
    for epoch in range(1, cfg.EPOCHS_STAGE4 + 1):
        model.train()
        model.generator.train()
        train_loss = 0.0
        train_loss_z = 0.0
        train_loss_x = 0.0

        for batch in train_dl:
            enc_input = batch["enc_input"].to(device)
            x_true    = batch["x"].to(device)
            feature_mask = batch["feature_mask"].to(device)
            mask      = batch["mask"].to(device)
            times_h   = batch["times_h"].to(device)
            T_K       = batch["T_K"].to(device)
            x0        = batch["x0"].to(device)

            # Real latent trajectories from encoder (no grad)
            with torch.no_grad():
                z_real, _  = model.encoder(enc_input, mask)    # (B,T,5)

            # Phase 3: extract prefix context for conditional generation
            prefix_len = cfg.STAGE3_PREFIX_LEN
            z_prefix_last   = z_real[:, prefix_len - 1, :].detach()
            log_prefix_time = torch.log1p(times_h[:, prefix_len - 1]).unsqueeze(1).detach()

            # Fake latent trajectories from generator + ODE
            z0_fake, alpha_fake = model.generator(
                x0, T_K,
                z_prefix_last=z_prefix_last,
                log_prefix_time=log_prefix_time,
            )
            z_traj_fake = model.ode.integrate_trajectory(
                z0_fake, T_K, times_h, alpha_fake)             # (B,T,5)
            x_hat_fake = model.decoder(z_traj_fake, z_ref=z_traj_fake[:, 0, :])  # (B,T,6)

            loss_terms = distribution_matching_loss(
                z_traj_fake, z_real.detach(),
                x_hat_fake, x_true, mask,
                return_components=True,
            )
            loss = loss_terms["total"]
            # NOTE: alpha diversity regularisation removed (Step 3 — verify
            # identifiability first; if alpha is not identifiable, forcing
            # diversity only hurts training.  Use GENERATOR_FIXED_ALPHA=True
            # in config.py if you want to fix alpha=1 instead.)

            opt_g.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_gen, cfg.GRAD_CLIP_NORM)
            opt_g.step()
            train_loss += loss.item()
            train_loss_z += loss_terms["z"].item()
            train_loss_x += loss_terms["x"].item()

        train_loss /= max(len(train_dl), 1)
        train_loss_z /= max(len(train_dl), 1)
        train_loss_x /= max(len(train_dl), 1)
        if epoch % 20 == 0:
            log.info(
                "  Stage4 Epoch %3d | distrib_match=%.5f (z=%.5f, x=%.5f)",
                epoch, train_loss, train_loss_z, train_loss_x,
            )
            # Per-20-epoch: report alpha stats + clipping rate for z0 perturbation
            with torch.no_grad():
                log.info(
                    "    alpha stats | mean=%.4f std=%.5f min=%.4f max=%.4f",
                    alpha_fake.mean().item(), alpha_fake.std().item(),
                    alpha_fake.min().item(), alpha_fake.max().item(),
                )
                gen = model.generator
                if hasattr(gen, "z0_pert_log"):
                    scales = torch.exp(gen.z0_pert_log).detach().cpu().numpy()
                    log.info(
                        "    z0_pert_scales (zG,zB,zM,zL,zC) = %s",
                        [f"{s:.4f}" for s in scales],
                    )
                if hasattr(gen, "_total_count") and gen._total_count.item() > 0:
                    clip_rate = float(gen._clip_count.item() / gen._total_count.item())
                    log.info("    z0 clipping rate = %.4f", clip_rate)

        if train_loss < best_val:
            best_val = train_loss
            _save_checkpoint(model, {"opt_g": opt_g}, 4, epoch, train_loss,
                             cfg.CHECKPOINT_DIR)
        es.step(train_loss)
        if es.stop:
            log.info("  Early stopping at epoch %d", epoch)
            break

    # Unfreeze
    for p in list(model.encoder.parameters()) + \
             list(model.decoder.parameters()) + \
             list(model.ode.parameters()):
        p.requires_grad_(True)

    _load_checkpoint(model, cfg.CHECKPOINT_DIR, 4)
    log.info("Stage 4 done. Best loss = %.5f", best_val)


# ---------------------------------------------------------------------------
# Stage 5: Adversarial fine-tuning
# ---------------------------------------------------------------------------

def train_stage5(model, train_dl, val_dl, device):
    log.info("\n=== Stage 5: Adversarial Fine-tuning ===")
    params_g = (
        list(model.encoder.parameters()) +
        list(model.generator.parameters()) +
        list(model.alpha_net.parameters()) +
        list(model.decoder.parameters()) +
        list(model.ode.parameters())
    )
    params_d = list(model.disc.parameters())

    opt_g = torch.optim.Adam(params_g, lr=cfg.LR_STAGE5)
    opt_d = torch.optim.Adam(params_d, lr=cfg.LR_STAGE5)
    es = EarlyStopping(patience=40)

    best_val = float("inf")
    for epoch in range(1, cfg.EPOCHS_STAGE5 + 1):
        model.train()
        g_loss_sum = d_loss_sum = 0.0
        g_phy_sum = 0.0
        g_adv_sum = 0.0
        g_adv_weighted_sum = 0.0
        g_count = d_count = 0
        skipped_d = skipped_g = 0
        sanitized_d = sanitized_g = 0

        for batch in train_dl:
            enc_input    = batch["enc_input"].to(device)
            x_true       = batch["x"].to(device)
            feature_mask = batch["feature_mask"].to(device)
            mask         = batch["mask"].to(device)
            times_h      = batch["times_h"].to(device)
            T_K          = batch["T_K"].to(device)
            x0           = batch["x0"].to(device)

            # Phase 3: prefix context for conditional generation
            prefix_len_s5 = cfg.STAGE3_PREFIX_LEN
            with torch.no_grad():
                z_pre_s5, _ = model.encoder(enc_input[:, :prefix_len_s5, :],
                                             mask[:, :prefix_len_s5])
            z_prefix_last_s5   = z_pre_s5[:, -1, :].detach()
            log_prefix_time_s5 = torch.log1p(times_h[:, prefix_len_s5 - 1]).unsqueeze(1).detach()

            # ---- Discriminator step ----
            z_real, _   = model.encoder(enc_input, mask)
            alpha_real  = model.alpha_net(x0, T_K)
            x_hat_real  = model.decoder(z_real, z_ref=z_real[:, 0, :])

            z0_fake, alpha_fake = model.generator(
                x0, T_K,
                z_prefix_last=z_prefix_last_s5,
                log_prefix_time=log_prefix_time_s5,
            )
            z_traj_fake = model.ode.integrate_trajectory(
                z0_fake, T_K, times_h, alpha_fake)
            x_hat_fake  = model.decoder(
                z_traj_fake.detach(),
                z_ref=z_traj_fake.detach()[:, 0, :],
            )

            logit_z_real = model.disc.latent(z_real.detach(), T_K, mask)
            logit_z_fake = model.disc.latent(z_traj_fake.detach(), T_K, mask)
            logit_x_real = model.disc.observation(x_hat_real.detach(), T_K, mask)
            logit_x_fake = model.disc.observation(x_hat_fake, T_K, mask)

            d_loss = (
                adversarial_discriminator_loss(logit_z_real, logit_z_fake) +
                adversarial_discriminator_loss(logit_x_real, logit_x_fake)
            )
            if not torch.isfinite(d_loss):
                skipped_d += 1
                continue
            opt_d.zero_grad()
            d_loss.backward()
            sanitized_d += _sanitize_nonfinite_gradients(params_d)
            torch.nn.utils.clip_grad_norm_(params_d, cfg.GRAD_CLIP_NORM)
            opt_d.step()
            d_loss_sum += d_loss.item()
            d_count += 1

            # ---- Generator / Encoder step ----
            z_real, _  = model.encoder(enc_input, mask)
            alpha_real = model.alpha_net(x0, T_K)
            x_hat      = model.decoder(z_real, z_ref=z_real[:, 0, :])

            z0_fake, alpha_fake = model.generator(
                x0, T_K,
                z_prefix_last=z_prefix_last_s5,
                log_prefix_time=log_prefix_time_s5,
            )
            z_traj_fake = model.ode.integrate_trajectory(
                z0_fake, T_K, times_h, alpha_fake)
            x_hat_fake2 = model.decoder(z_traj_fake, z_ref=z_traj_fake[:, 0, :])

            logit_z_f2 = model.disc.latent(z_traj_fake, T_K, mask)
            logit_x_f2 = model.disc.observation(x_hat_fake2, T_K, mask)

            loss_dict = total_physics_loss(
                x_hat, x_true, z_real, T_K, times_h, alpha_real,
                mask, model.ode, feature_mask=feature_mask)
            adv_g = adversarial_generator_loss(logit_z_f2, logit_x_f2)

            g_loss = loss_dict["total"] + cfg.LAMBDA_ADV_G * adv_g
            if not torch.isfinite(g_loss):
                skipped_g += 1
                continue
            opt_g.zero_grad()
            g_loss.backward()
            sanitized_g += _sanitize_nonfinite_gradients(params_g)
            torch.nn.utils.clip_grad_norm_(params_g, cfg.GRAD_CLIP_NORM)
            opt_g.step()
            g_loss_sum += g_loss.item()
            g_phy_sum += loss_dict["total"].item()
            g_adv_sum += adv_g.item()
            g_adv_weighted_sum += (cfg.LAMBDA_ADV_G * adv_g).item()
            g_count += 1

        g_loss_sum /= max(g_count, 1)
        g_phy_sum /= max(g_count, 1)
        g_adv_sum /= max(g_count, 1)
        g_adv_weighted_sum /= max(g_count, 1)
        d_loss_sum /= max(d_count, 1)

        # Validation: Phase 4 generative score replaces reconstruction RMSE
        # Score = CRPS + λ_w1*W1(Δx) + λ_cov*|Cov90-0.9| + λ_phys*phys_viol
        gen_score = _generative_val_score(model, val_dl, device)
        # Also compute val recon loss for logging purposes
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_dl:
                enc_input    = batch["enc_input"].to(device)
                x_true       = batch["x"].to(device)
                feature_mask = batch["feature_mask"].to(device)
                mask         = batch["mask"].to(device)
                times_h      = batch["times_h"].to(device)
                T_K          = batch["T_K"].to(device)
                x0           = batch["x0"].to(device)
                z_enc, _  = model.encoder(enc_input, mask)
                x_hat     = model.decoder(z_enc, z_ref=z_enc[:, 0, :])
                vl        = reconstruction_loss(x_hat, x_true, mask, feature_mask=feature_mask)
                if torch.isfinite(vl):
                    val_loss += vl.item()
                    val_count += 1
        val_loss = val_loss / max(val_count, 1) if val_count > 0 else float("inf")

        if epoch % 20 == 0:
            log.info(
                "  Stage5 Epoch %3d | G=%.5f (phy=%.5f, adv=%.5f, w*adv=%.5f) | D=%.5f | val_recon=%.5f | val_genscore=%.5f | skip(D/G)=%d/%d | sanitize(D/G)=%d/%d",
                epoch,
                g_loss_sum,
                g_phy_sum,
                g_adv_sum,
                g_adv_weighted_sum,
                d_loss_sum,
                val_loss,
                gen_score,
                skipped_d,
                skipped_g,
                sanitized_d,
                sanitized_g,
            )

        # Checkpoint selection driven by generative quality score (lower = better)
        if gen_score < best_val:
            best_val = gen_score
            _save_checkpoint(
                model, {"opt_g": opt_g, "opt_d": opt_d},
                5, epoch, gen_score, cfg.CHECKPOINT_DIR,
                selection_metric="generative_score",
                extra_metrics={"val_recon": val_loss},
            )
        es.step(gen_score)
        if es.stop:
            log.info("  Early stopping at epoch %d", epoch)
            break

    _load_checkpoint(model, cfg.CHECKPOINT_DIR, 5)
    log.info("Stage 5 done. Best generative score = %.5f", best_val)


# ---------------------------------------------------------------------------
# Entry point (run standalone via: python 08_training.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # When run directly, delegate to main.py for full module setup
    import subprocess
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
         "--mode", "train"],
        check=True,
    )
