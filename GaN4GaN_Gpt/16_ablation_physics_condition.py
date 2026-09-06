"""
16_ablation_physics_condition.py
=================================
A / B / C physics-conditioning ablation for the Stage 4C residual generator.

Purpose
-------
Stage 4C (14_stage4b_ar1_guided_generator.py::AR1GuidedResidualGeneratorStable)
conditions the residual generator on the physics latent state z_phys (=
z_prefix_last, the encoder's [zG,zB,zM,zL,zC] state at the prefix boundary)
together with T, t and the observed prefix x0.  The mere fact that the code
*calls* z_phys as an input does not prove the generator actually uses it in a
way that matters for prediction quality.  This script settles that question
empirically via three variants of the exact same architecture, training
procedure, and evaluation suite:

  A. Full physics-conditioned (baseline): generator receives
       [z_phys, Δz_phys, T, t, x0]
     where z_phys = z_prefix_last and Δz_phys = z_prefix_last - z_ref
     (change in physics latent state from t=0 to the prefix boundary).
     This is Stage 4C as currently implemented, extended with the Δz_phys
     input the design calls for.

  B. No-physics-condition: generator receives only [T, t, x0] — z_phys and
     Δz_phys are replaced with zeros before the context vector is built, so
     the network has zero gradient signal from the physics latent state.

  C. Shuffled-physics-condition: generator receives [z_phys, Δz_phys, T, t, x0]
     exactly as in A, EXCEPT z_phys/Δz_phys are randomly permuted across the
     batch dimension every forward pass (both training and eval), so each
     device sees a physics latent state sampled from a DIFFERENT random
     device in the same batch, while T, t, x0 and the residual targets stay
     correctly paired to the real device.

If A is clearly better than B and C, the generator is actually exploiting
physics-latent information — not just time and temperature.  If A ~= B ~= C,
the "physics conditioning" is decorative.

Evaluation (per variant, on the held-out test split):
  - CRPSS, coverage (50/80/90), W1 on increments   [10_stochastic_residual.py]
  - Counterfactual temperature consistency          [this file]
  - OOD temperature (leave-one-temperature-out)     [this file]
  - Long-horizon prediction (RMSE @ 100/500/1000/2000h) [09_evaluation.py]

Usage
-----
    python 16_ablation_physics_condition.py --stage3-ckpt <path> \
        [--epochs 20] [--n-eval-samples 100] [--device cpu] [--seed 42]

Runs all three conditions (A/B/C) for each requested seed, trains a fresh
Stage-4C-style generator per condition, evaluates, and writes a comparison
table + raw metrics to <output-dir>/results/ablation_physics_condition.pkl
"""

import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

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

CONDITIONS = ["A_full", "B_no_physics", "C_shuffled"]


def _load_module(alias, filename):
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_all():
    """Load every pipeline module the ablation needs, keyed by short alias."""
    mods = {}
    mods["stage4b"] = _load_module("_abl_stage4b", "14_stage4b_ar1_guided_generator.py")
    # 14_stage4b already transitively loads 13_stage4a (crps_mc_loss, _load_all,
    # _build_model, _cache_trajectories) and registers "_pi_stage4a_impl".
    mods["stage4a"] = sys.modules["_pi_stage4a_impl"]
    inner = mods["stage4a"]._load_all()
    mods.update(inner)   # ode, enc, dec, gen, disc, losses, train, eval, stoch
    return mods


# Stage 4C module, loaded at import time so the generator class below can take
# its feature set from the single source of truth rather than repeating it.
_s4b_mod = _load_module("_abl_s4b_featdef", "14_stage4b_ar1_guided_generator.py")


# ---------------------------------------------------------------------------
# A/B/C-conditioned generator (extends Stage4C's stable-feature architecture)
# ---------------------------------------------------------------------------

class PhysicsConditionGeneratorStable(nn.Module):
    """Stage 4C residual generator with a switchable physics-conditioning mode.

    Identical architecture/loss interface to
    14_stage4b_ar1_guided_generator.AR1GuidedResidualGeneratorStable (so it is
    a drop-in replacement for train_stage4b / evaluate_stage4b / Stage 5), but
    the context vector construction is parameterised by `condition_mode`:

      "full"     : context = [z_phys, Δz_phys, T_norm(1), x0, log_t(1)]
      "none"     : same layout, but z_phys and Δz_phys are zeroed out before
                   the network sees them (so no gradient path from physics
                   latents reaches the loss).
      "shuffled" : same layout as "full", but z_phys/Δz_phys are permuted
                   across the batch dimension on every forward call (a fresh
                   random permutation per call), decoupling them from the
                   device's actual T/t/x0/targets.
    """

    # Which features this generator models. Taken from the Stage 4C module so
    # the ablation always scores the SAME feature set Stage 4C was trained and
    # calibrated on. Hard-coding [0,1,2,3] here meant that after the
    # observation set grew to 11 the ablation silently kept generating only the
    # four original features, so its coverage (0.33) was not comparable with
    # Stage 4C's (0.86) and the A/B/C comparison was run on a different model
    # than the one under study.
    STABLE_INDICES = list(_s4b_mod.STABLE_FEAT_INDICES)
    N_STABLE = len(STABLE_INDICES)
    LOG10_T_REF = 0.35

    def __init__(
        self,
        condition_mode: str = "full",
        noise_dim: int = 16,
        hidden_dim: int = 96,
        n_output: int = None,
        latent_dim: int = cfg.LATENT_DIM,
        n_context_feat: int = cfg.FEATURE_DIM,
        log_scale_floor_init: float = -2.6,
    ):
        super().__init__()
        assert condition_mode in ("full", "none", "shuffled")
        if n_output is None:
            n_output = self.N_STABLE
        self.condition_mode = condition_mode
        self.noise_dim = noise_dim
        self.n_features = n_output
        self.latent_dim = latent_dim
        self._n_ctx_feat = n_context_feat
        self._leakage_indices = []

        # context = z_phys(latent_dim) + dz_phys(latent_dim) + T_norm(1)
        #           + x0(n_context_feat) + log_t(1)
        context_dim = 2 * latent_dim + 1 + n_context_feat + 1
        in_dim = noise_dim + context_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_output),
        )
        self.log_scale_floor = nn.Parameter(torch.full((n_output,), float(log_scale_floor_init)))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.15)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _build_physics_terms(self, z_prefix_last, z_ref):
        """Return (z_phys, dz_phys) tensors after applying condition_mode."""
        z_phys = torch.nan_to_num(z_prefix_last, nan=0.5, posinf=0.0, neginf=0.0)
        if z_ref is None:
            dz_phys = torch.zeros_like(z_phys)
        else:
            z_ref_c = torch.nan_to_num(z_ref, nan=0.5, posinf=0.0, neginf=0.0)
            dz_phys = z_phys - z_ref_c

        if self.condition_mode == "none":
            z_phys = torch.zeros_like(z_phys)
            dz_phys = torch.zeros_like(dz_phys)
        elif self.condition_mode == "shuffled":
            B = z_phys.shape[0]
            if B > 1:
                perm = torch.randperm(B, device=z_phys.device)
                z_phys = z_phys[perm]
                dz_phys = dz_phys[perm]
        # "full": pass through unchanged
        return z_phys, dz_phys

    def _context_params(self, z_prefix_last, T_K, x0, log_t_suffix, z_ref=None, noise_init=None):
        x0 = torch.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
        T_K = torch.nan_to_num(T_K, nan=0.0, posinf=0.0, neginf=0.0)
        log_t_suffix = torch.nan_to_num(log_t_suffix, nan=0.0, posinf=0.0, neginf=0.0)

        z_prefix_last = z_prefix_last if z_prefix_last.dim() > 1 else z_prefix_last.unsqueeze(0)
        B = z_prefix_last.shape[0]
        dev = z_prefix_last.device

        z_phys, dz_phys = self._build_physics_terms(z_prefix_last, z_ref)

        T_norm = ((T_K.reshape(-1) - 300.0) / 25.0).reshape(B, 1)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        if x0.dim() > 2:
            x0 = x0.reshape(B, -1)
        x0_ctx = x0[:, : self._n_ctx_feat]

        log_t = log_t_suffix.reshape(B, -1)[:, :1]

        ctx = torch.cat(
            [z_phys.reshape(B, -1), dz_phys.reshape(B, -1), T_norm, x0_ctx.reshape(B, -1), log_t],
            dim=-1,
        )

        if noise_init is None:
            noise_init = torch.zeros(B, self.noise_dim, device=dev, dtype=ctx.dtype)
        else:
            noise_init = torch.nan_to_num(noise_init, nan=0.0)
            if noise_init.dim() == 1:
                noise_init = noise_init.unsqueeze(0)

        inp = torch.cat([noise_init, ctx], dim=-1)
        out = self.net(inp)
        rho = torch.sigmoid(out[:, : self.n_features]) * 0.97
        sig_floor = torch.exp(self.log_scale_floor).unsqueeze(0).expand(B, -1)
        sigma = sig_floor + F.softplus(out[:, self.n_features :])
        return rho, sigma

    def forward(self, z_prefix_last, T_K, x0, log_t_suffix, T_future=10, noise=None,
                times_future=None, z_ref=None):
        B = z_prefix_last.shape[0]
        dev = z_prefix_last.device
        rho, sigma = self._context_params(z_prefix_last, T_K, x0, log_t_suffix, z_ref=z_ref)

        rho_eff_per_step = None
        if times_future is not None and times_future.shape[1] >= 2:
            t = times_future.to(dev).clamp(min=0.0)
            log1pt = torch.log10(1.0 + t)
            delta_log10 = torch.zeros(B, T_future, device=dev)
            delta_log10[:, 0] = self.LOG10_T_REF
            if T_future > 1:
                delta_log10[:, 1:] = (log1pt[:, 1:] - log1pt[:, :-1]).clamp(min=0.01, max=2.0)
            expo = (delta_log10 / self.LOG10_T_REF).unsqueeze(-1)
            rho_eff_per_step = rho.unsqueeze(1) ** expo

        deltas = []
        d_prev = torch.zeros(B, self.n_features, device=dev)
        for t_idx in range(T_future):
            eps = torch.randn(B, self.n_features, device=dev)
            rho_i = rho_eff_per_step[:, t_idx, :] if rho_eff_per_step is not None else rho
            sq = torch.sqrt((1.0 - rho_i ** 2).clamp(min=1e-6))
            d_t = rho_i * d_prev + sq * sigma * eps
            deltas.append(d_t)
            d_prev = d_t.detach()
        return torch.stack(deltas, dim=1)

    def sample_n(self, z_prefix_last, T_K, x0, log_t_suffix, n_samples, T_future=10,
                 times_future=None, z_ref=None):
        return torch.stack(
            [
                self.forward(z_prefix_last, T_K, x0, log_t_suffix, T_future=T_future,
                             times_future=times_future, z_ref=z_ref)
                for _ in range(n_samples)
            ],
            dim=0,
        )


# ---------------------------------------------------------------------------
# Training: thin wrapper around 14_stage4b_ar1_guided_generator.train_stage4b
# ---------------------------------------------------------------------------
# train_stage4b only calls generator.sample_n(z_pfx, T_K, x0, log_t, ...,
# T_future=..., times_future=...) and generator._context_params(z_pfx, T_K,
# x0, log_t) — it never passes z_ref.  For condition "full"/"shuffled" we
# still want Δz_phys = z_phys - z_ref where z_ref is the t=0 encoder state,
# which IS available in the cached records (rec["z_enc"][:, 0, :]).  Rather
# than modify the shared train_stage4b loop, we monkey-patch each generator
# instance with a bound z_ref lookup keyed by identity of z_pfx via a small
# per-batch cache set by a wrapped _cache_trajectories pass. To keep this
# robust and simple, we instead subclass-free it: we pre-store z_ref on the
# cached record and have the generator pull it from a thread-local set just
# before each sample_n/_context_params call in our OWN training loop below
# (we do not reuse train_stage4b's loop verbatim; we reimplement the minimal
# training loop here so Δz_phys is wired correctly end-to-end).
# ---------------------------------------------------------------------------

def train_ablation_generator(
    model,
    generator: PhysicsConditionGeneratorStable,
    train_cache: list,
    val_cache: list,
    device: torch.device,
    stage4a_mod,
    epochs: int = 20,
    lr: float = 4e-4,
    n_train_samples: int = 4,
    n_val_samples: int = 20,
    patience: int = 8,
    lambda_crps: float = 1.0,
    lambda_ar1: float = 0.50,
    lambda_var: float = 0.30,
    lambda_pinball: float = 0.10,
    lambda_scale: float = 0.01,
    lambda_calib: float = 0.0,
    lambda_phys_sens: float = 0.0,
    sigma_min: float = 0.012,
    output_dir: Optional[str] = None,
    ckpt_name: str = "generator_best.pt",
) -> dict:
    """Stage-4C-equivalent training loop, generalised to pass z_ref through so
    Δz_phys can be computed inside the generator's context builder."""
    stage4b_mod = sys.modules["_abl_stage4b"]
    crps_mc_loss = stage4a_mod.crps_mc_loss
    pinball_loss = stage4a_mod.pinball_loss
    _temp_equalized_var_loss = stage4b_mod._temp_equalized_var_loss
    _fit_ar1_targets = stage4b_mod._fit_ar1_targets
    coverage_calibration_loss = stage4b_mod.coverage_calibration_loss
    STABLE_IDX = generator.STABLE_INDICES

    def _phys_sens_loss(gen, z_pfx, z_ref, T_K, x0, log_t, margin=0.05):
        """Local adapter of 14_...::physics_sensitivity_loss for the ablation
        generator's _context_params(..., z_ref=...) signature. Only meaningful
        for condition_mode == "full": "none" has z_phys zeroed by design (the
        loss would just fight the condition), "shuffled" already IS a random
        z_phys every call so a second shuffle changes nothing structurally."""
        B = z_pfx.shape[0]
        if B <= 1 or gen.condition_mode != "full":
            return z_pfx.sum() * 0.0
        rho_real, sigma_real = gen._context_params(z_pfx, T_K, x0, log_t, z_ref=z_ref)
        perm = torch.randperm(B, device=z_pfx.device)
        rho_shuf, sigma_shuf = gen._context_params(z_pfx[perm], T_K, x0, log_t, z_ref=z_ref[perm])
        rho_diff = (rho_real - rho_shuf).abs().mean(dim=-1)
        sigma_diff = ((sigma_real - sigma_shuf).abs() / sigma_real.detach().clamp(min=1e-4)).mean(dim=-1)
        response = 0.5 * (rho_diff + sigma_diff)
        return torch.relu(margin - response).mean()

    if output_dir is None:
        output_dir = cfg.CHECKPOINT_DIR
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, ckpt_name)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    generator = generator.to(device)
    generator.train()
    opt = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(1, patience // 2), factor=0.5)
    prefix_len = cfg.STAGE3_PREFIX_LEN

    best_val_crps = float("inf")
    best_epoch = 0
    no_improve = 0
    history = {"train_crps": [], "val_crps": []}

    log.info("=== Training generator (condition=%s) ===", generator.condition_mode)

    for epoch in range(1, epochs + 1):
        generator.train()
        t0 = time.time()
        train_crps_total = 0.0
        n_train_batches = 0

        for rec in train_cache:
            z_pfx = rec["z_pfx"].to(device)
            z_ref = rec["z_ref"].to(device)
            x_hat = rec["x_hat"].to(device)
            x_true = rec["x_true"].to(device)
            mask = rec["mask"].to(device)
            T_K = rec["T_K"].to(device)
            log_t = rec["log_t"].to(device)
            x0 = rec["x0"].to(device)
            plen = rec["plen"]
            T_future = rec["T_len"] - plen
            if T_future <= 0:
                continue
            times_future_t = rec["times"][:, plen:].to(device) if "times" in rec else None

            deltas = generator.sample_n(z_pfx, T_K, x0, log_t, n_train_samples,
                                        T_future=T_future, times_future=times_future_t, z_ref=z_ref)

            sfx = torch.tensor(STABLE_IDX, device=device)
            x_hat_future = x_hat[:, plen:, :][:, :, sfx].unsqueeze(0)
            x_pred_future = x_hat_future + deltas
            x_prefix_exp = x_hat[:, :plen, :][:, :, sfx].unsqueeze(0).expand(n_train_samples, -1, -1, -1)
            x_pred_full = torch.cat([x_prefix_exp, x_pred_future], dim=2)
            x_true_loss = x_true[:, :, sfx]

            crps = crps_mc_loss(x_pred_full, x_true_loss, mask, prefix_len=plen)
            rho_pred, sigma_pred = generator._context_params(z_pfx, T_K, x0, log_t, z_ref=z_ref)
            rho_target, sigma_target = _fit_ar1_targets(
                x_true, x_hat, plen, feat_indices=STABLE_IDX, device_center=True,
                times_future=times_future_t,
            )
            rho_target = rho_target.to(device)
            sigma_target = sigma_target.to(device)

            rho_loss = ((rho_pred - rho_target) ** 2).mean()
            sigma_loss = _temp_equalized_var_loss(sigma_pred, sigma_target, T_K)
            future_true = torch.nan_to_num(x_true_loss[:, plen:, :], nan=0.0, posinf=0.0, neginf=0.0)
            pinball = pinball_loss(x_pred_future, future_true, mask[:, plen:], prefix_len=0)
            scale_reg = torch.relu(float(sigma_min) - sigma_pred).mean()

            calib_l = torch.tensor(0.0, device=device)
            if lambda_calib > 0:
                calib_l = coverage_calibration_loss(x_pred_full, x_true_loss, mask, prefix_len=plen)

            sens_l = torch.tensor(0.0, device=device)
            if lambda_phys_sens > 0:
                sens_l = _phys_sens_loss(generator, z_pfx, z_ref, T_K, x0, log_t)

            loss = (
                lambda_crps * crps + lambda_ar1 * rho_loss + lambda_var * sigma_loss
                + lambda_pinball * pinball + lambda_scale * scale_reg
                + lambda_calib * calib_l + lambda_phys_sens * sens_l
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt.step()

            train_crps_total += crps.item()
            n_train_batches += 1

        if n_train_batches == 0:
            continue
        train_crps_avg = train_crps_total / n_train_batches

        generator.eval()
        val_crps_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for rec in val_cache:
                z_pfx = rec["z_pfx"].to(device)
                z_ref = rec["z_ref"].to(device)
                x_hat = rec["x_hat"].to(device)
                x_true = rec["x_true"].to(device)
                mask = rec["mask"].to(device)
                T_K = rec["T_K"].to(device)
                log_t = rec["log_t"].to(device)
                x0 = rec["x0"].to(device)
                plen = rec["plen"]
                T_future = rec["T_len"] - plen
                if T_future <= 0:
                    continue
                times_future_v = rec["times"][:, plen:].to(device) if "times" in rec else None
                deltas_v = generator.sample_n(z_pfx, T_K, x0, log_t, n_val_samples,
                                              T_future=T_future, times_future=times_future_v, z_ref=z_ref)
                sfx_v = torch.tensor(STABLE_IDX, device=device)
                x_hat_future = x_hat[:, plen:, :][:, :, sfx_v].unsqueeze(0)
                x_pred_future = x_hat_future + deltas_v
                x_prefix_exp = x_hat[:, :plen, :][:, :, sfx_v].unsqueeze(0).expand(n_val_samples, -1, -1, -1)
                x_pred_v = torch.cat([x_prefix_exp, x_pred_future], dim=2)
                val_crps = crps_mc_loss(x_pred_v, x_true[:, :, sfx_v], mask, prefix_len=plen)
                val_crps_total += val_crps.item()
                n_val_batches += 1
        val_crps_avg = val_crps_total / max(n_val_batches, 1)
        sched.step(val_crps_avg)

        elapsed = time.time() - t0
        log.info("  [%s] Epoch %3d/%d | train_CRPS=%.4f val_CRPS=%.4f | %.0fs",
                 generator.condition_mode, epoch, epochs, train_crps_avg, val_crps_avg, elapsed)

        history["train_crps"].append(train_crps_avg)
        history["val_crps"].append(val_crps_avg)

        if val_crps_avg < best_val_crps:
            best_val_crps = val_crps_avg
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "val_crps": best_val_crps, "state_dict": generator.state_dict()},
                       ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  Early stopping at epoch %d", epoch)
                break

    # Reload best checkpoint
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(ck["state_dict"])

    log.info("Condition=%s training complete. best_epoch=%d val_CRPS=%.4f",
             generator.condition_mode, best_epoch, best_val_crps)
    return {"best_val_crps": best_val_crps, "best_epoch": best_epoch, "history": history}


# ---------------------------------------------------------------------------
# Sample generation for evaluation (fills non-generated features with x_hat)
# ---------------------------------------------------------------------------

def _generate_full_samples(model, generator, dl, device, _forward, prefix_len, n_samples,
                            temperature_override_K: Optional[float] = None):
    """Run the frozen backbone + generator over a dataloader, return
    (x_true, samples[S,N,T,F], mask, T_K, times_h, x_mean) as numpy arrays.

    If temperature_override_K is given, the backbone's ODE trajectory is
    NOT re-integrated at the new temperature (that requires the full
    physics-latent recompute done separately in ood_temperature_eval /
    counterfactual_temperature_consistency); this helper is for standard
    in-distribution test-set evaluation only.
    """
    stage4a_mod = sys.modules["_pi_stage4a_impl"]
    _cache_trajectories = stage4a_mod._cache_trajectories

    all_x_true, all_x_mean, all_T_K, all_times, all_mask = [], [], [], [], []
    with torch.no_grad():
        for batch in dl:
            x_raw = batch["x"].to(device)
            T_K = batch["T_K"].to(device)
            times = batch["times_h"].to(device)
            mask = batch["mask"].to(device)
            B, T_len, F = x_raw.shape
            if T_len <= prefix_len + 1:
                continue
            _, _, x_hat, x_true, _, _, _, _, _ = _forward(model, batch, device)
            all_x_mean.append(x_hat.cpu().numpy())
            all_x_true.append(x_true.cpu().numpy())
            all_T_K.append(T_K.cpu().numpy())
            all_times.append(times.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

    if not all_x_true:
        return None

    x_true_np = np.concatenate(all_x_true, axis=0)
    x_mean_np = np.concatenate(all_x_mean, axis=0)
    T_K_np = np.concatenate(all_T_K)
    times_np = np.concatenate(all_times, axis=0)
    mask_np = np.concatenate(all_mask, axis=0)
    N, T_len, F = x_true_np.shape

    eval_cache = _cache_trajectories(model, dl, device, _forward, prefix_len)
    samples = np.full((n_samples, N, T_len, F), np.nan)
    n_placed = 0
    stable_idx = generator.STABLE_INDICES
    with torch.no_grad():
        for rec in eval_cache:
            z_pfx = rec["z_pfx"].to(device)
            z_ref = rec["z_ref"].to(device)
            x_hat_r = rec["x_hat"].to(device)
            T_K_r = rec["T_K"].to(device)
            log_t = rec["log_t"].to(device)
            x0 = rec["x0"].to(device)
            plen_r = rec["plen"]
            T_len_r = rec["T_len"]
            T_future = T_len_r - plen_r
            B = z_pfx.shape[0]
            b_end = min(n_placed + B, N)
            actual_B = b_end - n_placed
            if T_future <= 0 or actual_B <= 0:
                n_placed = b_end
                continue
            times_future_r = rec["times"][:, plen_r:].to(device) if "times" in rec else None
            deltas = generator.sample_n(z_pfx, T_K_r, x0, log_t, n_samples,
                                        T_future=T_future, times_future=times_future_r, z_ref=z_ref)
            x_hat_np = x_hat_r.cpu().numpy()
            for s in range(n_samples):
                x_s = x_hat_np[:actual_B].copy()
                d_s = deltas[s].cpu().numpy()[:actual_B, :, :]
                for fi_loc, fi_glob in enumerate(stable_idx):
                    x_s[:, plen_r:, fi_glob] += d_s[:, :, fi_loc]
                samples[s, n_placed:b_end, :T_len_r, :] = x_s
            n_placed = b_end
            if n_placed >= N:
                break

    return {
        "x_true": x_true_np, "x_mean": x_mean_np, "T_K": T_K_np,
        "times_h": times_np, "mask": mask_np, "samples": samples,
    }


# ---------------------------------------------------------------------------
# Negative-control / physics-use evaluations
# ---------------------------------------------------------------------------

def counterfactual_temperature_consistency(model, generator, eval_cache, device, n_samples=30) -> Dict[str, float]:
    """For each cached device, re-run the generator with the SAME z_phys/x0/
    prefix but T swept over {275, 300, 325} C (holding everything else fixed).

    WARNING -- frac_monotone_275_300_325 IS NOT A VALID METRIC. Do not report
    it. See MONOTONICITY_ANOMALY.md. Three independent defects:

      1. Its premise is false. It rewards a residual that RISES with T, but
         the measured residual dips at 300 C (0.1002 / 0.0655 / 0.2093 at
         548 / 573 / 598 K) -- a U shape, because 300 C is T_REF_K where the
         Arrhenius factors are 1 and the ODE is best conditioned. A generator
         that reproduces the real profile is scored WRONG; one emitting a
         bland monotone ramp is scored RIGHT. Measured shape correlation with
         the real profile: A_full -0.70, B_no_physics +0.97 -- i.e. the metric
         ranks last the condition that best captures the falling limb.
      2. It scores |ensemble MEAN|, which is ~0 by construction for a
         well-centred generator. All conditions land at ~0.015 across every
         temperature, an order of magnitude below the real residual, varying
         a few percent over a 50 C span.
      3. tol=1e-4 against differences of order 1e-3 makes each device a
         near-coin-flip, which is why the statistic swings between 0.000 and
         0.407 across runs whose CRPSS differs by ~3 %.

    Prefer: correlation of the generated per-temperature profile against the
    MEASURED one, the ensemble SPREAD (deltas.std, which is O(0.1) and is what
    Arrhenius scaling actually predicts), or the learned Ea_sigma compared
    with GaN literature.

    The original (unsound) rationale was: a generator that has learned
    meaningful physics conditioning should produce ensemble-mean residual
    magnitude / rho that responds monotonically to temperature. We report:
      - frac_monotone: fraction of devices where mean |residual| increases
        (or at least does not decrease beyond tolerance) from 275->300->325C.
      - mean_abs_resid_by_T: average generated |residual| ensemble mean at
        each counterfactual temperature (diagnostic).
    """
    temps_c = [275.0, 300.0, 325.0]
    temps_k = [t + cfg.CELSIUS_TO_KELVIN for t in temps_c]
    stable_idx = generator.STABLE_INDICES

    per_temp_mean_abs = {tc: [] for tc in temps_c}
    n_monotone = 0
    n_total = 0

    generator.eval()
    with torch.no_grad():
        for rec in eval_cache:
            z_pfx = rec["z_pfx"].to(device)
            z_ref = rec["z_ref"].to(device)
            log_t = rec["log_t"].to(device)
            x0 = rec["x0"].to(device)
            plen = rec["plen"]
            T_future = rec["T_len"] - plen
            if T_future <= 0:
                continue
            times_future_r = rec["times"][:, plen:].to(device) if "times" in rec else None

            abs_by_t = []
            for tc, tk in zip(temps_c, temps_k):
                B = z_pfx.shape[0]
                T_K_cf = torch.full((B,), tk, device=device)
                deltas = generator.sample_n(z_pfx, T_K_cf, x0, log_t, n_samples,
                                            T_future=T_future, times_future=times_future_r, z_ref=z_ref)
                mean_abs = deltas.mean(dim=0).abs().mean(dim=(1, 2))   # (B,) mean |ensemble-mean residual|
                abs_by_t.append(mean_abs.cpu().numpy())
                per_temp_mean_abs[tc].append(float(mean_abs.mean().item()))

            abs_275, abs_300, abs_325 = abs_by_t
            tol = 1e-4
            monotone = (abs_300 >= abs_275 - tol) & (abs_325 >= abs_300 - tol)
            n_monotone += int(monotone.sum())
            n_total += monotone.shape[0]

    return {
        # INVALID -- see the warning in this function's docstring and
        # MONOTONICITY_ANOMALY.md. Retained only so previously saved artefacts
        # stay loadable; never report it.
        "frac_monotone_275_300_325": float(n_monotone / max(n_total, 1)),
        "mean_abs_resid_275C": float(np.mean(per_temp_mean_abs[275.0])) if per_temp_mean_abs[275.0] else float("nan"),
        "mean_abs_resid_300C": float(np.mean(per_temp_mean_abs[300.0])) if per_temp_mean_abs[300.0] else float("nan"),
        "mean_abs_resid_325C": float(np.mean(per_temp_mean_abs[325.0])) if per_temp_mean_abs[325.0] else float("nan"),
    }


def ood_temperature_eval(model, generator, test_dl, device, _forward, prefix_len,
                          stoch_mod, n_samples=100, holdout_temp_c: float = 325.0) -> Dict:
    """Leave-one-temperature-out extrapolation check.

    Restricts evaluation to devices stored/measured at `holdout_temp_c`
    (typically the hottest, hardest-to-extrapolate condition) and computes
    the same CRPSS/coverage/W1 metrics on that subset only. Comparing this
    to the overall (in-distribution-mixed) metrics from `compute_metrics`
    shows whether the generator's physics conditioning degrades gracefully
    or collapses when asked to extrapolate to a temperature under-represented
    in training (this pipeline trains and evaluates on the SAME temperature
    set [275,300,325]C, so this is an approximate OOD probe via subset
    stratification, not a true unseen-temperature holdout, since retraining
    the frozen Stage1-3 backbone without one temperature is out of scope for
    this ablation).
    """
    out = _generate_full_samples(model, generator, test_dl, device, _forward, prefix_len, n_samples)
    if out is None:
        return {}
    tk = holdout_temp_c + cfg.CELSIUS_TO_KELVIN
    sel = np.abs(out["T_K"] - tk) < 1.0
    if sel.sum() == 0:
        log.warning("No devices found at holdout temperature %.0fC", holdout_temp_c)
        return {}

    dummy = stoch_mod.StochasticResidualModel()
    metrics = dummy.compute_metrics(
        out["x_true"][sel], out["samples"][:, sel, :, :], out["mask"][sel],
        T_K=out["T_K"][sel], prefix_len=prefix_len, x_mean=out["x_mean"][sel],
        times_h=out["times_h"][sel],
    )
    return {
        "holdout_temp_c": holdout_temp_c,
        "n_devices": int(sel.sum()),
        "crpss_overall": metrics.get("crpss_overall", float("nan")),
        "coverage_90_overall": metrics.get("coverage_90_overall", float("nan")),
        "w1_increments_overall": float(np.nanmean(list(metrics.get("w1_increments", {}).values()))),
    }


def long_horizon_eval(out: Dict, eval_mod, horizons=None) -> Dict:
    """RMSE of the generator's ensemble MEAN prediction at long horizons,
    reusing 09_evaluation.py::compute_horizon_rmse."""
    if horizons is None:
        horizons = cfg.EVAL_HORIZONS_H
    ensemble_mean = np.nanmean(out["samples"], axis=0)   # (N, T, F)
    return eval_mod.compute_horizon_rmse(ensemble_mean, out["x_true"], out["mask"], out["times_h"], horizons=horizons)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_condition(condition_mode: str, model, train_cache, val_cache, test_cache_dl,
                   device, mods, args, output_dir) -> Dict:
    stage4a_mod = mods["stage4a"]
    generator = PhysicsConditionGeneratorStable(
        condition_mode=condition_mode,
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    train_result = train_ablation_generator(
        model, generator, train_cache, val_cache, device, stage4a_mod,
        epochs=args.epochs, lr=args.lr,
        n_train_samples=args.n_train_samples, n_val_samples=args.n_val_samples,
        patience=args.patience, output_dir=output_dir,
        lambda_pinball=args.lambda_pinball,
        lambda_calib=args.lambda_calib, lambda_phys_sens=args.lambda_phys_sens,
        sigma_min=args.sigma_min,
        ckpt_name=f"generator_{condition_mode}.pt",
    )

    _forward = mods["train"]._forward
    prefix_len = cfg.STAGE3_PREFIX_LEN

    out = _generate_full_samples(model, generator, test_cache_dl, device, _forward,
                                  prefix_len, args.n_eval_samples)
    if out is None:
        log.warning("No test samples generated for condition=%s", condition_mode)
        return {"condition": condition_mode, "train": train_result}

    dummy = mods["stoch"].StochasticResidualModel()
    core_metrics = dummy.compute_metrics(
        out["x_true"], out["samples"], out["mask"],
        T_K=out["T_K"], prefix_len=prefix_len, x_mean=out["x_mean"], times_h=out["times_h"],
    )

    # Counterfactual temperature consistency (needs a cached record list, not a dl)
    stage4a_mod2 = sys.modules["_pi_stage4a_impl"]
    eval_cache = stage4a_mod2._cache_trajectories(model, test_cache_dl, device, _forward, prefix_len)
    cf_temp = counterfactual_temperature_consistency(model, generator, eval_cache, device,
                                                      n_samples=min(30, args.n_eval_samples))

    ood_temp = ood_temperature_eval(model, generator, test_cache_dl, device, _forward, prefix_len,
                                     mods["stoch"], n_samples=args.n_eval_samples, holdout_temp_c=325.0)

    horizon = long_horizon_eval(out, mods["eval"])

    return {
        "condition": condition_mode,
        "train": train_result,
        "crpss_overall": core_metrics.get("crpss_overall", float("nan")),
        "coverage_50_overall": core_metrics.get("coverage_50_overall", float("nan")),
        "coverage_80_overall": core_metrics.get("coverage_80_overall", float("nan")),
        "coverage_90_overall": core_metrics.get("coverage_90_overall", float("nan")),
        "w1_increments_overall": float(np.nanmean(list(core_metrics.get("w1_increments", {}).values()))),
        "reliability_mace": core_metrics.get("reliability_mace", float("nan")),
        "counterfactual_temp_consistency": cf_temp,
        "ood_temperature": ood_temp,
        "long_horizon_rmse": horizon,
        "full_metrics": core_metrics,
    }


def _print_comparison_table(results: List[Dict]):
    header = f"{'Condition':<16}{'CRPSS':>8}{'Cov90':>8}{'W1':>10}{'CF-mono':>10}{'OOD-CRPSS':>12}"
    log.info(header)
    log.info("-" * len(header))
    for r in results:
        cf = r.get("counterfactual_temp_consistency", {})
        ood = r.get("ood_temperature", {})
        log.info(
            f"{r['condition']:<16}"
            f"{r.get('crpss_overall', float('nan')):>8.3f}"
            f"{r.get('coverage_90_overall', float('nan')):>8.3f}"
            f"{r.get('w1_increments_overall', float('nan')):>10.4f}"
            f"{cf.get('frac_monotone_275_300_325', float('nan')):>10.3f}"
            f"{ood.get('crpss_overall', float('nan')):>12.3f}"
        )


def _parse_args():
    p = argparse.ArgumentParser(description="A/B/C physics-conditioning ablation for Stage 4C")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-train-samples", type=int, default=4)
    p.add_argument("--n-val-samples", type=int, default=20)
    p.add_argument("--n-eval-samples", type=int, default=100)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--noise-dim", type=int, default=16)
    p.add_argument("--sigma-min", type=float, default=0.012)
    p.add_argument("--lambda-pinball", type=float, default=0.10)
    p.add_argument("--lambda-calib", type=float, default=0.0,
        help="Weight for coverage-calibration loss (0=off).")
    p.add_argument("--lambda-phys-sens", type=float, default=0.0,
        help="Weight for physics-latent sensitivity loss (0=off, only "
             "applied for condition A_full).")
    p.add_argument("--stage3-ckpt", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS),
                   help="Comma-separated subset of A_full,B_no_physics,C_shuffled")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir or cfg.OUTPUT_PATH
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    log.info("Loading pipeline modules...")
    mods = _load_all()
    train_mod = mods["train"]
    stage4a_mod = mods["stage4a"]

    log.info("Loading dataset...")
    prep_path = cfg.PROCESSED_DATA_PATH
    with open(prep_path, "rb") as fh:
        dataset = pickle.load(fh)

    from torch.utils.data import DataLoader
    split = dataset["split"]
    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]

    train_ds = train_mod.DeviceDegradationDataset(dataset, train_idx)
    val_ds = train_mod.DeviceDegradationDataset(dataset, val_idx)
    test_ds = train_mod.DeviceDegradationDataset(dataset, test_idx)
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=train_mod.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=train_mod.collate_fn)
    test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=train_mod.collate_fn)

    log.info("Building PI-TimeGAN backbone (frozen: encoder, ODE, decoder)...")
    model = stage4a_mod._build_model(mods).to(device)
    stage3_ckpt = args.stage3_ckpt or os.path.join(ckpt_dir, "stage3_best.pt")
    ckpt = torch.load(stage3_ckpt, map_location=device)
    model_state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    model_state_clean = {k: v for k, v in model_state.items() if k != "decoder.mask"}
    model.load_state_dict(model_state_clean, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("Loaded Stage 3 checkpoint: %s", stage3_ckpt)

    _forward = train_mod._forward
    prefix_len = cfg.STAGE3_PREFIX_LEN
    _cache_trajectories = stage4a_mod._cache_trajectories

    log.info("Caching training/validation trajectories (shared across all conditions)...")
    train_cache = _cache_trajectories(model, train_dl, device, _forward, prefix_len)
    val_cache = _cache_trajectories(model, val_dl, device, _forward, prefix_len)

    requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in requested:
        assert c in CONDITIONS, f"Unknown condition {c!r}; choose from {CONDITIONS}"

    all_results = []
    for condition_mode_full in requested:
        condition_mode = condition_mode_full.split("_", 1)[0].lower()
        # map "A_full" -> "full", "B_no_physics" -> "no", "C_shuffled" -> "shuffled"
        mode_map = {"a": "full", "b": "none", "c": "shuffled"}
        mode = mode_map[condition_mode]
        log.info("=" * 70)
        log.info("Running condition %s (mode=%s)", condition_mode_full, mode)
        log.info("=" * 70)
        result = run_condition(mode, model, train_cache, val_cache, test_dl, device, mods, args, ckpt_dir)
        result["condition"] = condition_mode_full
        all_results.append(result)

    _print_comparison_table(all_results)

    out_path = os.path.join(results_dir, "ablation_physics_condition.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump({"results": all_results, "args": vars(args)}, fh)
    log.info("Ablation results saved -> %s", out_path)

    # Also dump a JSON summary (excluding non-serialisable / huge fields)
    summary = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k not in ("full_metrics", "train")}
        summary.append(s)
    json_path = os.path.join(results_dir, "ablation_physics_condition_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("Ablation summary saved -> %s", json_path)


if __name__ == "__main__":
    main()
