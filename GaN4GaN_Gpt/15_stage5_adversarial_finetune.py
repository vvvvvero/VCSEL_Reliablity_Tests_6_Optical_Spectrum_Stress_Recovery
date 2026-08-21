#!/usr/bin/env python
"""
15_stage5_adversarial_finetune.py
==================================
Stage 5: Adversarial fine-tuning of Stage 4B generator.

Design choices:
  - Physics backbone (encoder, decoder, ODE) is FROZEN.
  - Only the AR1GuidedResidualGenerator (from Stage 4B) and a fresh
    ObservationDiscriminator are trainable.
  - Discriminator operates on FUTURE residuals (x_true - x_pred)
    in normalised space to stay in the space the generator was trained in.
  - Generator loss: lambda_crps * L_CRPS + lambda_adv * L_adv
                     + lambda_phys_preserve * L_phys_preserve
  - lambda_adv = 0.005 (intentionally very small, a light regulariser).
  - L_phys_preserve = || mean_m(x_fake^(m)) - mu_phys ||^2, where mu_phys is the
    frozen backbone's deterministic mean trajectory (x_hat).  Since
    x_fake^(m) = x_hat + delta^(m), this reduces to || mean_m(delta^(m)) ||^2:
    the ensemble-mean *residual* must stay at zero, so the adversarial term
    cannot drag the whole predictive distribution off the physics mean.
  - Collapse guard (all conditions must pass every epoch):
      1. val_CRPS must not degrade > CRPS_TOL (5%) vs Stage 4B baseline.
      2. Generated-sample diversity (std across S) >= DIVERSITY_FLOOR * baseline.
      3. Discriminator balanced-accuracy <= DISC_ACC_CEIL (max 85%).
  - Saves Stage 5 checkpoint ONLY if all collapse guards pass AND
    val_CRPS is the best seen so far.
"""

import argparse
import importlib.util
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ─── Hyperparameters ────────────────────────────────────────────────────────
STAGE5_EPOCHS      = 15
STAGE5_LR_G        = 2e-4    # generator LR (lower than Stage 4B to stay conservative)
STAGE5_LR_D        = 1e-4    # discriminator LR
LAMBDA_CRPS        = 1.0
LAMBDA_ADV         = 0.005   # very small adversarial regulariser
LAMBDA_VAR         = 0.30    # keep variance loss from Stage 4B
LAMBDA_AR1         = 0.50    # keep AR1 loss from Stage 4B
LAMBDA_PHYS_PRESERVE = cfg.STAGE5_PHYSICS_WEIGHT   # penalise ensemble-mean drift from ODE mean
N_TRAIN_SAMPLES    = 4
N_VAL_SAMPLES      = 20
PATIENCE           = 6

# Collapse guard thresholds
CRPS_TOL           = 0.05    # max allowed fractional CRPS degradation vs Stage 4B
DIVERSITY_FLOOR    = 0.70    # min generated diversity as fraction of Stage 4B baseline
DISC_ACC_CEIL      = 0.85    # max discriminator balanced-accuracy before "mode collapse" flag


# ─── Module loader ────────────────────────────────────────────────────────────
def _load_module(alias, filename):
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Residual discriminator (fresh, Stage-5 only) ────────────────────────────
class ResidualDiscriminator(nn.Module):
    """Discriminate real from generated residuals, using residuals + increments.

    Input per time step: [residuals(F) || increments(F) || T_norm(1)] = 2F+1 dims.

    Including increments makes the discriminator sensitive to temporal correlation
    patterns: low ACF generates large increments (noisy), high ACF generates small
    increments (smooth).  This is key for detecting when generated trajectories
    have the wrong temporal structure even if their marginal distribution is correct.
    """

    def __init__(
        self,
        feature_dim: int = cfg.FEATURE_DIM,
        hidden_dim:  int = 64,
        num_layers:  int = 2,
    ):
        super().__init__()
        # 2 * feature_dim for [residual || increment] + 1 for T_norm
        self.gru = nn.GRU(
            input_size=2 * feature_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, residual: torch.Tensor, T_K: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        residual : (B, T_future, F)  normalised residuals (x_true - x_hat)
        T_K      : (B,)
        mask     : (B, T_future)     future validity mask
        """
        B, T, F = residual.shape
        res = torch.nan_to_num(residual, nan=0.0)

        # Compute residual increments: Δd_t = d_t - d_{t-1};  Δd_0 = d_0
        incr = torch.zeros_like(res)
        incr[:, 0, :]  = res[:, 0, :]
        incr[:, 1:, :] = res[:, 1:, :] - res[:, :-1, :]

        T_norm = ((T_K - cfg.T_REF_K) / cfg.T_REF_K).view(B, 1, 1).expand(B, T, 1)
        inp = torch.cat([res, incr, T_norm], dim=2)   # (B, T, 2F+1)

        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(inp, lengths, batch_first=True, enforce_sorted=False)
        _, h_n  = self.gru(packed)
        h_last  = h_n[-1]
        return self.head(h_last).squeeze(1)   # (B,)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _diversity(deltas: torch.Tensor) -> float:
    """Mean std of generated deltas across sample axis."""
    return float(deltas.std(dim=0).mean().item())


def physics_preserve_loss(deltas: torch.Tensor, mask: "torch.Tensor" = None) -> torch.Tensor:
    """L_phys-preserve = || (1/M) sum_m x_fake^(m) - mu_phys ||^2.

    x_fake^(m) = x_hat + deltas[m], mu_phys = x_hat (frozen backbone mean), so
    this reduces to the squared norm of the ensemble-mean residual: the
    generator's samples must average back to the physics-backbone trajectory,
    not just individually track it.

    deltas : (S, B, T_future, F)  generated residual samples
    mask   : (B, T_future) optional validity mask; if given, restricts the
             mean to valid time steps only.
    """
    ensemble_mean = deltas.mean(dim=0)   # (B, T_future, F)
    if mask is not None:
        w = mask.float().unsqueeze(-1)   # (B, T_future, 1)
        denom = w.sum().clamp(min=1.0)
        sq = (ensemble_mean ** 2) * w
        return sq.sum() / denom
    return (ensemble_mean ** 2).mean()


def _disc_balanced_accuracy(logit_real: torch.Tensor, logit_fake: torch.Tensor) -> float:
    """Balanced accuracy of binary discriminator on a batch."""
    with torch.no_grad():
        acc_real = float((logit_real > 0).float().mean().item())
        acc_fake = float((logit_fake < 0).float().mean().item())
    return 0.5 * (acc_real + acc_fake)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_stage5(
    model3,
    generator,          # AR1GuidedResidualGenerator from Stage 4B
    train_cache,
    val_cache,
    device,
    stage4b_val_crps: float,
    stage4b_diversity: float,
    output_dir: str,
    epochs:          int   = STAGE5_EPOCHS,
    lr_g:            float = STAGE5_LR_G,
    lr_d:            float = STAGE5_LR_D,
    lambda_crps:     float = LAMBDA_CRPS,
    lambda_adv:      float = LAMBDA_ADV,
    lambda_var:      float = LAMBDA_VAR,
    lambda_ar1:      float = LAMBDA_AR1,
    lambda_phys_preserve: float = LAMBDA_PHYS_PRESERVE,
    n_train_samples: int   = N_TRAIN_SAMPLES,
    n_val_samples:   int   = N_VAL_SAMPLES,
    patience:        int   = PATIENCE,
    crps_tol:        float = CRPS_TOL,
    diversity_floor: float = DIVERSITY_FLOOR,
    disc_acc_ceil:   float = DISC_ACC_CEIL,
):
    # Import Stage 4B loss helpers
    s4b_mod = sys.modules.get("_s5_stage4b")
    crps_mc_loss = s4b_mod.crps_mc_loss
    pinball_loss  = s4b_mod.pinball_loss
    _fit_ar1_targets = s4b_mod._fit_ar1_targets
    _temp_equalized_var_loss = s4b_mod._temp_equalized_var_loss
    STABLE_IDX = s4b_mod.STABLE_FEAT_INDICES  # [0, 1, 2, 3]

    # Detect whether generator is Stage4C stable-only
    is_stable_gen = hasattr(generator, 'STABLE_INDICES')

    disc = ResidualDiscriminator(feature_dim=len(STABLE_IDX) if is_stable_gen else cfg.FEATURE_DIM).to(device)
    disc.train()

    # Physics modules are FROZEN — only generator + discriminator trained
    model3.eval()
    for p in model3.parameters():
        p.requires_grad_(False)

    generator.train()
    for p in generator.parameters():
        p.requires_grad_(True)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=lr_g, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(),      lr=lr_d, weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, patience=max(1, patience // 2), factor=0.5)

    prefix_len = cfg.STAGE3_PREFIX_LEN
    best_val_crps = float("inf")
    best_epoch    = 0
    no_improve    = 0
    collapse_count = 0
    history = []

    ckpt_path = os.path.join(output_dir, "stage5_best.pt")

    # ─── Target gradient ratio (actual measured, not proxy) ───────────────────
    TARGET_R_GRAD = float(lambda_adv) if lambda_adv > 0 else 0.001  # interpret lambda_adv as target
    _lambda_adv   = 0.0   # start at 0; calibrate after D pre-training
    GRAD_MEASURE_INTERVAL = 5   # measure actual R_grad every N batches

    log.info("=" * 70)
    log.info("=== Stage 5: Adversarial fine-tuning (calibrated) ===")
    log.info("  target_R_grad=%.4f  lambda_crps=%.3f  lambda_var=%.3f  lambda_ar1=%.3f  lambda_phys_preserve=%.3f",
             TARGET_R_GRAD, lambda_crps, lambda_var, lambda_ar1, lambda_phys_preserve)
    log.info("  CRPS tol=%.2f%%  diversity_floor=%.2f  disc_acc_ceil=%.2f",
             crps_tol * 100, diversity_floor, disc_acc_ceil)
    log.info("  Stage4B baseline: val_CRPS=%.4f  diversity=%.4f",
             stage4b_val_crps, stage4b_diversity)

    # ─── Phase 0: Discriminator pre-training (50 steps on first batch) ────────
    # Gives D meaningful gradients before G starts adversarial updates.
    log.info("  Phase 0: D pre-training (50 steps)...")
    disc.train()
    N_DISC_PRETRAIN = 50
    if train_cache:
        rec0 = train_cache[0]
        plen0 = rec0["plen"]
        T_future0 = rec0["T_len"] - plen0
        if T_future0 > 0:
            sfx_t = torch.tensor(STABLE_IDX, device=device) if is_stable_gen else None
            with torch.no_grad():
                real_res_pt = (rec0["x_true"].to(device)[:, plen0:, sfx_t if sfx_t is not None else slice(None)]
                               - rec0["x_hat"].to(device)[:, plen0:, sfx_t if sfx_t is not None else slice(None)])
                t_f_pt = rec0["times"][:, plen0:].to(device) if "times" in rec0 else None
                fake_res_pt = generator.sample_n(
                    rec0["z_pfx"].to(device), rec0["T_K"].to(device),
                    rec0["x0"].to(device), rec0["log_t"].to(device),
                    n_samples=1, T_future=T_future0,
                    **({'times_future': t_f_pt} if t_f_pt is not None and hasattr(generator, 'LOG10_T_REF') else {}),
                )[0].detach()
            fm_pt = rec0["mask"].to(device)[:, plen0:].bool()
            T_K_pt = rec0["T_K"].to(device)
            for _ in range(N_DISC_PRETRAIN):
                opt_d.zero_grad()
                lr = disc(real_res_pt, T_K_pt, fm_pt)
                lf = disc(fake_res_pt, T_K_pt, fm_pt)
                dl = (F.binary_cross_entropy_with_logits(lr, torch.ones_like(lr)) +
                      F.binary_cross_entropy_with_logits(lf, torch.zeros_like(lf)))
                dl.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                opt_d.step()
            disc_acc_pt = 0.5 * (float((disc(real_res_pt, T_K_pt, fm_pt).detach() > 0).float().mean())
                                 + float((disc(fake_res_pt, T_K_pt, fm_pt).detach() < 0).float().mean()))
            log.info("  D pre-training done. disc_acc=%.4f", disc_acc_pt)

    # ─── Calibrate initial lambda_adv via actual gradient measurement ─────────
    log.info("  Calibrating initial lambda_adv via actual gradient measurement...")
    if train_cache and TARGET_R_GRAD > 0:
        rec0 = train_cache[0]
        plen0 = rec0["plen"]
        T_future0 = rec0["T_len"] - plen0
        if T_future0 > 0:
            sfx_t = torch.tensor(STABLE_IDX, device=device) if is_stable_gen else None
            t_f0 = rec0["times"][:, plen0:].to(device) if "times" in rec0 else None
            deltas0 = generator.sample_n(
                rec0["z_pfx"].to(device), rec0["T_K"].to(device),
                rec0["x0"].to(device), rec0["log_t"].to(device),
                n_samples=4, T_future=T_future0,
                **({'times_future': t_f0} if t_f0 is not None and hasattr(generator, 'LOG10_T_REF') else {}),
            )
            fm0 = rec0["mask"].to(device)[:, plen0:].bool()
            x_true0 = rec0["x_true"].to(device)
            x_hat0  = rec0["x_hat"].to(device)
            T_K0    = rec0["T_K"].to(device)
            mask0   = rec0["mask"].to(device)
            rho_p0, sigma_p0 = generator._context_params(rec0["z_pfx"].to(device), T_K0, rec0["x0"].to(device), rec0["log_t"].to(device))
            rho_t0, sigma_t0 = _fit_ar1_targets(x_true0, x_hat0, plen0,
                                                  feat_indices=STABLE_IDX if is_stable_gen else None,
                                                  device_center=True, times_future=t_f0)

            # Measure ||grad(L_4C)|| via single backward
            opt_g.zero_grad()
            if is_stable_gen:
                xp0 = x_hat0[:, plen0:, sfx_t].unsqueeze(0) + deltas0
                xpfx0 = x_hat0[:, :plen0, sfx_t].unsqueeze(0).expand(4, -1, -1, -1)
                xpv0 = torch.cat([xpfx0, xp0], dim=2)
                loss_4c = (lambda_crps * crps_mc_loss(xpv0, x_true0[:,:,sfx_t], mask0, prefix_len=plen0)
                           + lambda_ar1 * ((rho_p0 - rho_t0)**2).mean()
                           + lambda_var * _temp_equalized_var_loss(sigma_p0, sigma_t0, T_K0))
            else:
                xp0 = x_hat0[:, plen0:].unsqueeze(0) + deltas0
                xpfx0 = x_hat0[:, :plen0].unsqueeze(0).expand(4,-1,-1,-1)
                xpv0 = torch.cat([xpfx0, xp0], dim=2)
                loss_4c = (lambda_crps * crps_mc_loss(xpv0, x_true0, mask0, prefix_len=plen0)
                           + lambda_ar1 * ((rho_p0 - rho_t0)**2).mean()
                           + lambda_var * _temp_equalized_var_loss(sigma_p0, sigma_t0, T_K0))
            loss_4c.backward(retain_graph=True)
            norm_4c_cal = sum(p.grad.norm().item()**2 for p in generator.parameters() if p.grad is not None)**0.5

            opt_g.zero_grad()
            logit_cal = disc(deltas0[0], T_K0, fm0)
            loss_adv_cal = F.binary_cross_entropy_with_logits(logit_cal, torch.ones_like(logit_cal))
            loss_adv_cal.backward()
            norm_adv_cal = sum(p.grad.norm().item()**2 for p in generator.parameters() if p.grad is not None)**0.5
            opt_g.zero_grad()

            r_cal = norm_adv_cal / (norm_4c_cal + 1e-10)
            _lambda_adv = float(np.clip(TARGET_R_GRAD / max(r_cal, 1e-12), 1e-4, 50.0))
            log.info("  Calibration: ||grad_4C||=%.4f  ||grad_adv||=%.6f  R_grad=%.4f%%  => lambda_adv=%.4f",
                     norm_4c_cal, norm_adv_cal, r_cal*100, _lambda_adv)

    log.info("=" * 70)

    for epoch in range(1, epochs + 1):
        generator.train()
        disc.train()
        t0 = time.time()

        g_crps_sum = g_adv_sum = g_var_sum = g_ar1_sum = g_phys_sum = 0.0
        d_loss_sum = disc_acc_sum = diversity_sum = 0.0
        r_grad_sum = 0.0
        n_batches = 0

        for rec in train_cache:
            z_pfx   = rec["z_pfx"].to(device)
            x_hat   = rec["x_hat"].to(device)
            x_true  = rec["x_true"].to(device)
            mask    = rec["mask"].to(device)
            T_K     = rec["T_K"].to(device)
            log_t   = rec["log_t"].to(device)
            x0      = rec["x0"].to(device)
            plen    = rec["plen"]
            T_future = rec["T_len"] - plen
            if T_future <= 0:
                continue

            # Generate samples
            deltas = generator.sample_n(z_pfx, T_K, x0, log_t, n_train_samples, T_future=T_future)
            # deltas: (S, B, T_future, F_gen)  where F_gen=4 for stable, 6 for full

            # ── Discriminator step ──────────────────────────────────────────
            future_mask = mask[:, plen:].bool()
            real_resid_full = (x_true[:, plen:, :] - x_hat[:, plen:, :]).detach()
            fake_resid_full = deltas[0].detach()

            # If generator is stable-only, discriminator only sees stable features
            if is_stable_gen:
                # deltas already (S, B, T_f, 4); build 6-feature real resid slice
                sfx = torch.tensor(STABLE_IDX, device=device)
                real_resid = real_resid_full[:, :, sfx]          # (B, T_f, 4)
                fake_resid = fake_resid_full                      # (B, T_f, 4) already
            else:
                real_resid = real_resid_full
                fake_resid = fake_resid_full

            logit_r = disc(real_resid, T_K, future_mask)
            logit_f = disc(fake_resid, T_K, future_mask)

            d_loss = (
                F.binary_cross_entropy_with_logits(logit_r, torch.ones_like(logit_r))
                + F.binary_cross_entropy_with_logits(logit_f, torch.zeros_like(logit_f))
            )
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()

            disc_acc = _disc_balanced_accuracy(logit_r, logit_f)

            # ── Generator step ──────────────────────────────────────────────
            rho_p, sigma_p  = generator._context_params(z_pfx, T_K, x0, log_t)
            rho_t, sigma_t  = _fit_ar1_targets(x_true, x_hat, plen,
                                               feat_indices=STABLE_IDX if is_stable_gen else None)
            rho_t    = rho_t.to(device)
            sigma_t  = sigma_t.to(device)

            if is_stable_gen:
                sfx = torch.tensor(STABLE_IDX, device=device)
                x_hat_fut_s = x_hat[:, plen:, sfx].unsqueeze(0)
                x_pred_fut_s = x_hat_fut_s + deltas              # (S, B, T_f, 4)
                x_pfx_s = x_hat[:, :plen, sfx].unsqueeze(0).expand(n_train_samples, -1, -1, -1)
                x_pv_s  = torch.cat([x_pfx_s, x_pred_fut_s], dim=2)
                x_true_s = x_true[:, :, sfx]
                crps_l   = crps_mc_loss(x_pv_s, x_true_s, mask, prefix_len=plen)
                adv_fake = deltas[0]                              # (B, T_f, 4)
            else:
                x_pf   = x_hat[:, plen:, :].unsqueeze(0) + deltas
                xpfx   = x_hat[:, :plen, :].unsqueeze(0).expand(n_train_samples, -1, -1, -1)
                x_pv   = torch.cat([xpfx, x_pf], dim=2)
                crps_l = crps_mc_loss(x_pv, x_true, mask, prefix_len=plen)
                adv_fake = deltas[0]

            ar1_l  = ((rho_p - rho_t) ** 2).mean()
            var_l  = _temp_equalized_var_loss(sigma_p, sigma_t, T_K)

            # Adversarial loss for generator: fool discriminator
            logit_f_g    = disc(adv_fake, T_K, future_mask)
            adv_l = F.binary_cross_entropy_with_logits(logit_f_g, torch.ones_like(logit_f_g))

            # Physics preservation: ensemble-mean residual must not drift from
            # the frozen ODE/decoder mean (deltas already zero-centred target).
            phys_l = physics_preserve_loss(deltas, future_mask)

            g_loss = (
                lambda_crps * crps_l + lambda_ar1 * ar1_l + lambda_var * var_l
                + _lambda_adv * adv_l + lambda_phys_preserve * phys_l
            )
            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()

            g_crps_sum  += crps_l.item()
            g_adv_sum   += adv_l.item()
            g_var_sum   += var_l.item()
            g_ar1_sum   += ar1_l.item()
            g_phys_sum  += phys_l.item()
            d_loss_sum  += d_loss.item()
            disc_acc_sum+= disc_acc
            diversity_sum += _diversity(deltas)
            n_batches   += 1

        if n_batches == 0:
            continue

        # ── Validation ─────────────────────────────────────────────────────
        generator.eval()
        disc.eval()
        val_crps_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for rec in val_cache:
                z_pfx   = rec["z_pfx"].to(device)
                x_hat   = rec["x_hat"].to(device)
                x_true  = rec["x_true"].to(device)
                mask    = rec["mask"].to(device)
                T_K     = rec["T_K"].to(device)
                log_t   = rec["log_t"].to(device)
                x0      = rec["x0"].to(device)
                plen    = rec["plen"]
                T_future= rec["T_len"] - plen
                if T_future <= 0:
                    continue
                deltas_v = generator.sample_n(z_pfx, T_K, x0, log_t, n_val_samples, T_future=T_future)
                if is_stable_gen:
                    sfx_v = torch.tensor(STABLE_IDX, device=device)
                    x_pf  = x_hat[:, plen:, :][:, :, sfx_v].unsqueeze(0) + deltas_v
                    xpfx  = x_hat[:, :plen, :][:, :, sfx_v].unsqueeze(0).expand(n_val_samples, -1, -1, -1)
                    x_pv  = torch.cat([xpfx, x_pf], dim=2)
                    val_crps_sum += crps_mc_loss(x_pv, x_true[:, :, sfx_v], mask, prefix_len=plen).item()
                else:
                    x_pf = x_hat[:, plen:, :].unsqueeze(0) + deltas_v
                    xpfx = x_hat[:, :plen, :].unsqueeze(0).expand(n_val_samples, -1, -1, -1)
                    x_pv = torch.cat([xpfx, x_pf], dim=2)
                    val_crps_sum += crps_mc_loss(x_pv, x_true, mask, prefix_len=plen).item()
                n_val += 1

        val_crps = val_crps_sum / max(n_val, 1)
        sched_g.step(val_crps)
        elapsed = time.time() - t0

        # ── Adaptive lambda_adv: measure actual R_grad every epoch on one batch ──
        # Use two separate backward passes for accuracy (not proxy via loss ratio).
        r_meas_epoch = float("nan")
        if TARGET_R_GRAD > 0 and train_cache:
            rec_cal = train_cache[0]
            plen_c  = rec_cal["plen"]
            T_f_c   = rec_cal["T_len"] - plen_c
            if T_f_c > 0:
                sfx_c = torch.tensor(STABLE_IDX, device=device) if is_stable_gen else None
                t_fc  = rec_cal["times"][:, plen_c:].to(device) if "times" in rec_cal else None
                kw_tf = {'times_future': t_fc} if (t_fc is not None and hasattr(generator, 'LOG10_T_REF')) else {}
                generator.eval()
                d_c = generator.sample_n(
                    rec_cal["z_pfx"].to(device), rec_cal["T_K"].to(device),
                    rec_cal["x0"].to(device), rec_cal["log_t"].to(device),
                    n_samples=2, T_future=T_f_c, **kw_tf,
                )
                generator.train()
                xt_c  = rec_cal["x_true"].to(device)
                xh_c  = rec_cal["x_hat"].to(device)
                T_K_c = rec_cal["T_K"].to(device)
                m_c   = rec_cal["mask"].to(device)
                fm_c  = m_c[:, plen_c:].bool()
                rp_c, sp_c = generator._context_params(rec_cal["z_pfx"].to(device), T_K_c, rec_cal["x0"].to(device), rec_cal["log_t"].to(device))
                rt_c, st_c = _fit_ar1_targets(xt_c, xh_c, plen_c,
                                               feat_indices=STABLE_IDX if is_stable_gen else None,
                                               device_center=True, times_future=t_fc)
                # Backward 1: non-adversarial only
                opt_g.zero_grad()
                if is_stable_gen:
                    xp_c = xh_c[:, plen_c:, sfx_c].unsqueeze(0) + d_c
                    xpfx_c = xh_c[:, :plen_c, sfx_c].unsqueeze(0).expand(2,-1,-1,-1)
                    xpv_c  = torch.cat([xpfx_c, xp_c], dim=2)
                    l4c    = (lambda_crps * crps_mc_loss(xpv_c, xt_c[:,:,sfx_c], m_c, prefix_len=plen_c)
                              + lambda_ar1 * ((rp_c - rt_c)**2).mean()
                              + lambda_var * _temp_equalized_var_loss(sp_c, st_c, T_K_c))
                else:
                    xp_c = xh_c[:, plen_c:].unsqueeze(0) + d_c
                    xpfx_c = xh_c[:, :plen_c].unsqueeze(0).expand(2,-1,-1,-1)
                    xpv_c  = torch.cat([xpfx_c, xp_c], dim=2)
                    l4c    = lambda_crps * crps_mc_loss(xpv_c, xt_c, m_c, prefix_len=plen_c)
                l4c.backward(retain_graph=True)
                n4c = sum(p.grad.norm().item()**2 for p in generator.parameters() if p.grad is not None)**0.5
                # Backward 2: adversarial only
                opt_g.zero_grad()
                logit_adv = disc(d_c[0], T_K_c, fm_c)
                ladv_only = F.binary_cross_entropy_with_logits(logit_adv, torch.ones_like(logit_adv))
                ladv_only.backward()
                nadv = sum(p.grad.norm().item()**2 for p in generator.parameters() if p.grad is not None)**0.5
                opt_g.zero_grad()

                r_meas = nadv / (n4c + 1e-10)
                r_meas_epoch = float(r_meas)
                if r_meas > 0 and n4c > 0:
                    _lambda_adv = float(np.clip(
                        _lambda_adv * (TARGET_R_GRAD / r_meas) ** 0.5,
                        1e-4, 100.0,
                    ))

        # ── Collapse guards ────────────────────────────────────────────────
        avg_diversity = diversity_sum / n_batches
        avg_disc_acc  = disc_acc_sum  / n_batches
        crps_degraded = (val_crps - stage4b_val_crps) / max(abs(stage4b_val_crps), 1e-8)
        div_ratio     = avg_diversity / max(stage4b_diversity, 1e-8)

        guard_pass = (
            crps_degraded <= crps_tol and
            div_ratio      >= diversity_floor and
            avg_disc_acc   <= disc_acc_ceil
        )
        if not guard_pass:
            collapse_count += 1
        else:
            collapse_count = 0

        log.info(
            "Epoch %3d/%d | val_CRPS=%.4f (S4B=%.4f) | G_CRPS=%.4f G_adv=%.4f G_phys=%.5f "
            "| disc_acc=%.3f div_ratio=%.2f | r_meas=%.4g lam_adv=%.6f | guards=%s | %.0fs",
            epoch, epochs, val_crps, stage4b_val_crps,
            g_crps_sum / n_batches, g_adv_sum / n_batches, g_phys_sum / n_batches,
            avg_disc_acc, div_ratio, r_meas_epoch, _lambda_adv,
            "PASS" if guard_pass else f"FAIL({collapse_count})",
            elapsed,
        )

        if collapse_count >= 3:
            log.warning("Collapse guard failed 3 consecutive epochs — early stopping Stage 5.")
            break

        history.append({
            "epoch": epoch, "val_crps": val_crps,
            "disc_acc": avg_disc_acc, "div_ratio": div_ratio,
            "crps_degraded": crps_degraded,
            "r_meas": r_meas_epoch,
            "g_phys_preserve": g_phys_sum / n_batches,
        })

        if guard_pass and val_crps < best_val_crps:
            best_val_crps = val_crps
            best_epoch    = epoch
            no_improve    = 0
            torch.save({
                "epoch":       epoch,
                "val_crps":    best_val_crps,
                "state_dict":  generator.state_dict(),
                "disc_state":  disc.state_dict(),
                "history":     history,
            }, ckpt_path)
            log.info("  ✓ Stage 5 checkpoint saved (val_CRPS=%.4f) -> %s", best_val_crps, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  Early stopping at epoch %d", epoch)
                break

    log.info("Stage 5 done. Best epoch=%d val_CRPS=%.4f", best_epoch, best_val_crps)
    return {"best_val_crps": best_val_crps, "best_epoch": best_epoch, "history": history}


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Stage 5: adversarial fine-tuning of Stage 4B")
    ap.add_argument("--checkpoint-stage3",  required=True)
    ap.add_argument("--checkpoint-stage4b", required=True)
    ap.add_argument("--output-dir",         required=True)
    ap.add_argument("--epochs",        type=int,   default=STAGE5_EPOCHS)
    ap.add_argument("--lr-g",          type=float, default=STAGE5_LR_G)
    ap.add_argument("--lr-d",          type=float, default=STAGE5_LR_D)
    ap.add_argument("--lambda-adv",    type=float, default=LAMBDA_ADV)
    ap.add_argument("--lambda-crps",   type=float, default=LAMBDA_CRPS)
    ap.add_argument("--lambda-var",    type=float, default=LAMBDA_VAR)
    ap.add_argument("--lambda-ar1",    type=float, default=LAMBDA_AR1)
    ap.add_argument("--lambda-phys-preserve", type=float, default=LAMBDA_PHYS_PRESERVE)
    ap.add_argument("--crps-tol",      type=float, default=CRPS_TOL)
    ap.add_argument("--diversity-floor",type=float,default=DIVERSITY_FLOOR)
    ap.add_argument("--disc-acc-ceil", type=float, default=DISC_ACC_CEIL)
    ap.add_argument("--device",        default="cpu")
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    import random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load modules
    mods = {}
    for alias, fname in [
        ("_s5_prep",    "01_data_preprocessing.py"),
        ("_s5_ode",     "02_physics_latent.py"),
        ("_s5_enc",     "03_model_encoder.py"),
        ("_s5_dec",     "04_model_decoder.py"),
        ("_s5_gen",     "05_model_generator.py"),
        ("_s5_disc",    "06_model_discriminator.py"),
        ("_s5_train",   "08_training.py"),
        ("_s5_stage4b", "14_stage4b_ar1_guided_generator.py"),
    ]:
        mods[alias] = _load_module(alias, fname)

    # Build backbone model
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = mods["_s5_enc"].PhysicsEncoder()
            self.decoder   = mods["_s5_dec"].SparsePhysicsDecoder()
            self.ode       = mods["_s5_ode"].PhysicsODE()
            self.alpha_net = mods["_s5_ode"].DeviceAlphaNet()
            self.generator = mods["_s5_gen"].PITimeGANGenerator()
            self.disc      = mods["_s5_disc"].PITimeGANDiscriminator()

    model3 = mods["_s5_train"].load_model_for_training(  # or eval.load_model
        args.checkpoint_stage3, _Model
    ).to(device) if hasattr(mods["_s5_train"], "load_model_for_training") else \
        torch.load(args.checkpoint_stage3, map_location=device)
    # Use eval loader
    eval_mod = _load_module("_s5_eval", "09_evaluation.py")
    model3   = eval_mod.load_model(args.checkpoint_stage3, _Model).to(device)
    model3.eval()
    for p in model3.parameters():
        p.requires_grad_(False)

    # Load Stage 4B / Stage 4C generator
    ckpt4b  = torch.load(args.checkpoint_stage4b, map_location=device)
    sd      = ckpt4b["state_dict"]
    STABLE_IDX = mods["_s5_stage4b"].STABLE_FEAT_INDICES
    # Detect phys-gated / Arrhenius-sigma checkpoints first (each has a
    # submodule/parameter not present in the other variants) — checked before
    # the plain-stable output-layer-size heuristic, since that shape check
    # alone can't distinguish AR1GuidedResidualGeneratorStable from the newer
    # variants.
    is_phys_gated_ckpt  = any(k.startswith("z_phys_encoder.") for k in sd)
    is_arrhenius_ckpt   = "log_Ea_sigma" in sd
    output_size = sd.get("net.5.bias", sd.get("net.6.bias", None))
    is_stable_ckpt = (output_size is not None and output_size.numel() == 2 * mods["_s5_stage4b"].N_STABLE_FEATURES)
    if is_arrhenius_ckpt:
        log.info("Detected Arrhenius-sigma Stage 4C checkpoint (physically-constrained sigma(T))")
        gen4b = mods["_s5_stage4b"].AR1GuidedResidualGeneratorArrhenius().to(device)
    elif is_phys_gated_ckpt:
        log.info("Detected phys-gated Stage 4C checkpoint (separate z_phys/context encoders)")
        gen4b = mods["_s5_stage4b"].AR1GuidedResidualGeneratorPhysGated().to(device)
    elif is_stable_ckpt:
        log.info("Detected Stage 4C checkpoint (stable-only, %d output features)",
                 mods["_s5_stage4b"].N_STABLE_FEATURES)
        gen4b = mods["_s5_stage4b"].AR1GuidedResidualGeneratorStable().to(device)
    else:
        gen4b = mods["_s5_stage4b"].AR1GuidedResidualGenerator().to(device)
    missing, _ = gen4b.load_state_dict(sd, strict=False)
    if missing:
        log.warning("Missing keys: %s", missing)

    # Cache trajectories
    _cache_trajectories = mods["_s5_stage4b"]._cache_trajectories
    _forward = mods["_s5_train"]._forward

    dataset   = mods["_s5_prep"].load_dataset()
    from torch.utils.data import DataLoader
    train_ds = mods["_s5_train"].DeviceDegradationDataset(dataset, dataset["split"]["train"])
    val_ds   = mods["_s5_train"].DeviceDegradationDataset(dataset, dataset["split"]["val"])
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          collate_fn=mods["_s5_train"].collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                          collate_fn=mods["_s5_train"].collate_fn)

    log.info("Caching training trajectories...")
    train_cache = _cache_trajectories(model3, train_dl, device, _forward, cfg.STAGE3_PREFIX_LEN)
    log.info("Caching validation trajectories...")
    val_cache   = _cache_trajectories(model3, val_dl,   device, _forward, cfg.STAGE3_PREFIX_LEN)

    # Compute Stage 4B baseline val_CRPS and diversity for collapse guard
    log.info("Computing Stage 4B baseline metrics for collapse guard...")
    crps_mc_loss_fn = mods["_s5_stage4b"].crps_mc_loss
    s4b_crps_sum = 0.0
    s4b_div_sum  = 0.0
    n_val = 0
    gen4b.eval()
    with torch.no_grad():
        for rec in val_cache:
            z_pfx   = rec["z_pfx"].to(device)
            x_hat   = rec["x_hat"].to(device)
            x_true  = rec["x_true"].to(device)
            mask    = rec["mask"].to(device)
            T_K     = rec["T_K"].to(device)
            log_t   = rec["log_t"].to(device)
            x0      = rec["x0"].to(device)
            plen    = rec["plen"]
            T_future= rec["T_len"] - plen
            if T_future <= 0:
                continue
            deltas  = gen4b.sample_n(z_pfx, T_K, x0, log_t, N_VAL_SAMPLES, T_future=T_future)
            if is_stable_ckpt:
                _sfx = torch.tensor(STABLE_IDX, device=device)
                x_f  = x_hat[:, plen:, :][:, :, _sfx].unsqueeze(0) + deltas
                xpfx = x_hat[:, :plen, :][:, :, _sfx].unsqueeze(0).expand(N_VAL_SAMPLES, -1, -1, -1)
                xpv  = torch.cat([xpfx, x_f], dim=2)
                s4b_crps_sum += crps_mc_loss_fn(xpv, x_true[:, :, _sfx], mask, prefix_len=plen).item()
            else:
                x_f  = x_hat[:, plen:, :].unsqueeze(0) + deltas
                xpfx = x_hat[:, :plen, :].unsqueeze(0).expand(N_VAL_SAMPLES, -1, -1, -1)
                xpv  = torch.cat([xpfx, x_f], dim=2)
                s4b_crps_sum += crps_mc_loss_fn(xpv, x_true, mask, prefix_len=plen).item()
            s4b_div_sum  += _diversity(deltas)
            n_val += 1

    stage4b_val_crps  = s4b_crps_sum  / max(n_val, 1)
    stage4b_diversity = s4b_div_sum   / max(n_val, 1)
    log.info("Stage 4B baseline: val_CRPS=%.4f  diversity=%.4f",
             stage4b_val_crps, stage4b_diversity)

    # Train Stage 5
    result = train_stage5(
        model3=model3,
        generator=gen4b,
        train_cache=train_cache,
        val_cache=val_cache,
        device=device,
        stage4b_val_crps=stage4b_val_crps,
        stage4b_diversity=stage4b_diversity,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_crps=args.lambda_crps,
        lambda_adv=args.lambda_adv,
        lambda_var=args.lambda_var,
        lambda_ar1=args.lambda_ar1,
        lambda_phys_preserve=args.lambda_phys_preserve,
        crps_tol=args.crps_tol,
        diversity_floor=args.diversity_floor,
        disc_acc_ceil=args.disc_acc_ceil,
    )

    log.info("Stage 5 complete: %s", result)


if __name__ == "__main__":
    main()
