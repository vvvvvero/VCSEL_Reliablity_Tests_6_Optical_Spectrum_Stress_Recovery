"""
13_stage4a_residual_generator.py
=================================
Stage 4A: Residual Latent Innovation Generator for PI-TimeGAN.
Non-adversarial pretraining with CRPS loss (debug13 design).

Architecture
-------------
The physics ODE provides a deterministic mean trajectory.
This generator learns LATENT-SPACE PERTURBATIONS δ_z that, when added to the
prefix-boundary latent state and propagated through the frozen ODE, produce
diverse stochastic trajectories in observation space.

Key difference from old Stage 4:
  Old Stage 4: G generates z0 from scratch (full trajectory) → collapses
  Stage 4A:    G generates δ_z = SMALL correction to z_prefix_last → stable

Architecture:
  Input:  [ε(noise_dim), z_prefix_last(5), T_norm(1), x0(n_features), log_t(1)]
  Output: δ_z(5) — per-state additive innovations
  Per-state scales (log-space, learnable):
    zG/zB: very small (reversible, already well-modeled)
    zM:    tiny (channel transport, monotone, well-anchored)
    zL:    moderate (leakage-path, main stochastic driver)
    zC:    small (cumulative damage, slow dynamics)

Training (Stage 4A - non-adversarial CRPS):
  Loss = λ_crps * CRPS(decoded samples, x_true_future)
       + λ_ar1  * AR1_consistency(δ_z across prefix steps)
       + λ_mono * monotone_physics_penalty(z_pred + δ_z)

Entry to Stage 4B (adversarial) ONLY if:
  CRPSS_stage4A > AR1_CRPSS  (e.g. > 0.15) AND Cov90 > 0.80

Usage
------
  python 13_stage4a_residual_generator.py [--epochs 40] [--n-train-samples 8]
                                           [--output-dir <path>]
"""

import argparse
import logging
import math
import os
import sys
import time
from importlib.util import spec_from_file_location, module_from_spec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# Dynamic module loading (same pattern as 11_grouped_cv_stability.py)
# ---------------------------------------------------------------------------

def _load_module(alias: str, filename: str):
    path = os.path.join(BASE_DIR, filename)
    spec = spec_from_file_location(alias, path)
    mod = module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_all():
    mods = {
        "ode":    _load_module("_pi_ode",    "02_physics_latent.py"),
        "enc":    _load_module("_pi_enc",    "03_model_encoder.py"),
        "dec":    _load_module("_pi_dec",    "04_model_decoder.py"),
        "gen":    _load_module("_pi_gen",    "05_model_generator.py"),
        "disc":   _load_module("_pi_disc",   "06_model_discriminator.py"),
        "losses": _load_module("_pi_losses", "07_losses.py"),
        "train":  _load_module("_pi_train",  "08_training.py"),
        "eval":   _load_module("_pi_eval",   "09_evaluation.py"),
        "stoch":  _load_module("_pi_stoch",  "10_stochastic_residual.py"),
    }
    sys.modules["_07_losses_import"] = mods["losses"]
    return mods


def _build_model(mods):
    """Build the same PITimeGAN model structure used across all scripts."""

    class _PIModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = mods["enc"].PhysicsEncoder()
            self.decoder   = mods["dec"].SparsePhysicsDecoder()
            self.ode       = mods["ode"].PhysicsODE()
            self.alpha_net = mods["ode"].DeviceAlphaNet()
            self.generator = mods["gen"].PITimeGANGenerator()
            self.disc      = mods["disc"].PITimeGANDiscriminator()

    return _PIModel()

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------

STAGE4A_EPOCHS          = 40
STAGE4A_N_TRAIN_SAMPLES = 8      # noise samples per device per batch
STAGE4A_N_VAL_SAMPLES   = 30     # samples for validation CRPS
STAGE4A_LR              = 5e-4
STAGE4A_PATIENCE        = 12
STAGE4A_GRAD_CLIP       = 1.0
STAGE4A_HIDDEN_DIM      = 64
STAGE4A_NOISE_DIM       = 16
LAMBDA_CRPS             = 1.0
LAMBDA_AR1              = 0.10    # AR1-autocorrelation regularizer weight
LAMBDA_VAR_MATCH        = 0.20    # match residual variance to empirical residuals
LAMBDA_PINBALL          = 0.10    # quantile/pinball coverage proxy
LAMBDA_SCALE_REG        = 0.01    # L2 on log_scale to prevent collapse
AR1_TARGET_RHO          = 0.65   # target AR(1) autocorrelation for innovations

# ---------------------------------------------------------------------------
# Observation-Space Residual Generator
# (Replaces latent-space generator: avoids ODE rollouts entirely)
# ---------------------------------------------------------------------------

class ResidualLatentInnovationGenerator(nn.Module):
    """
    Learned observation-space residual generator.

    Generates per-feature RESIDUAL SCALES conditioned on context, then
    produces AR(1)-structured residual sequences from them.

    Why observation space instead of latent space:
      - Avoids expensive ODE rollouts during training (huge speedup on CPU)
      - Directly comparable to AR(1) empirical baseline
      - Gradient path: CRPS → x_pred (= x_mean + δ_x) → δ_x → generator
      - Physics consistency maintained via monotone penalty on leakage features

    Architecture:
      Context: [z_pfx(5), T_norm(1), x0(n_feat), log_t(1)] → hidden → [μ(F), log_σ(F)]
      Residual generation: AR(1) with learned rho and learned σ per feature
        δ_f(t) = rho_f * δ_f(t-1) + sqrt(1-rho_f²) * σ_f * ε_t
      σ_f is conditioned on context; rho_f is a global learnable parameter.
    """

    def __init__(
        self,
        noise_dim:   int = STAGE4A_NOISE_DIM,
        hidden_dim:  int = STAGE4A_HIDDEN_DIM,
        n_features:  int = cfg.FEATURE_DIM,     # 6
        latent_dim:  int = cfg.LATENT_DIM,      # 5
    ):
        super().__init__()
        self.noise_dim  = noise_dim
        self.n_features = n_features

        # Context: z_pfx(5) + T_norm(1) + x0(n_feat) + log_t(1)
        context_dim = latent_dim + 1 + n_features + 1
        in_dim      = noise_dim + context_dim

        # Network outputs: log_sigma per feature (context-conditioned scale)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features),   # log_sigma (F,)
        )

        # Learnable global AR(1) autocorrelation per feature
        # Init at logit(0.65) ≈ 0.619 so sigmoid → 0.65
        _rho_init = torch.full((n_features,), 0.619)
        self.rho_logit = nn.Parameter(_rho_init)   # sigmoid → rho ∈ (0, 1)

        # Global scale floor (prevents collapse to zero)
        # Init: log(0.05) ≈ -3.0 — moderate residual
        self.log_scale_floor = nn.Parameter(
            torch.full((n_features,), -3.0)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        # Last layer: small init so context-conditioning starts near zero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def rho(self) -> torch.Tensor:
        """AR(1) autocorrelation per feature, ∈ (0, 1)."""
        return torch.sigmoid(self.rho_logit) * 0.95   # cap at 0.95

    def forward(
        self,
        z_prefix_last:  torch.Tensor,   # (B, 5)
        T_K:            torch.Tensor,   # (B,)
        x0:             torch.Tensor,   # (B, n_features)
        log_t_suffix:   torch.Tensor,   # (B,)
        T_future:       int = 10,       # number of future steps to generate
        noise:          torch.Tensor = None,  # (B, T_future, n_features) or None
    ) -> torch.Tensor:
        """
        Generate residual trajectory.

        Returns δ_x: (B, T_future, n_features) — additive residuals to x_mean.
        """
        B   = z_prefix_last.shape[0]
        dev = z_prefix_last.device

        # Sanitize NaN inputs
        z_prefix_last = torch.nan_to_num(z_prefix_last, nan=0.5, posinf=0.0, neginf=0.0)
        x0            = torch.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
        T_K           = torch.nan_to_num(T_K, nan=0.0, posinf=0.0, neginf=0.0)
        log_t_suffix  = torch.nan_to_num(log_t_suffix, nan=0.0, posinf=0.0, neginf=0.0)

        if noise is None:
            noise_init = torch.randn(B, self.noise_dim, device=dev)
        else:
            noise_init = noise[:, 0, :] if noise.dim() == 3 else noise

        T_norm = ((T_K - 300.0) / 25.0).unsqueeze(-1)
        log_t  = log_t_suffix.unsqueeze(-1)
        ctx    = torch.cat([z_prefix_last, T_norm, x0, log_t], dim=-1)
        inp    = torch.cat([noise_init, ctx], dim=-1)

        # Context-conditioned log-scale correction + global floor
        log_sigma_ctx  = self.net(inp)                              # (B, F)
        sigma_floor    = torch.exp(self.log_scale_floor)            # (F,)
        sigma          = sigma_floor + F.softplus(log_sigma_ctx)    # (B, F) > 0

        # Generate AR(1) residual sequence
        rho  = self.rho                                             # (F,)
        sq   = torch.sqrt((1.0 - rho ** 2).clamp(min=1e-6))        # (F,) innovation scale

        deltas = []
        d_prev = torch.zeros(B, self.n_features, device=dev)

        for t in range(T_future):
            eps   = torch.randn(B, self.n_features, device=dev)
            d_t   = rho * d_prev + sq * eps * sigma                 # (B, F)
            deltas.append(d_t)
            d_prev = d_t.detach()  # detach to prevent gradient accumulation

        return torch.stack(deltas, dim=1)                           # (B, T_future, F)

    def sample_n(
        self,
        z_prefix_last:  torch.Tensor,
        T_K:            torch.Tensor,
        x0:             torch.Tensor,
        log_t_suffix:   torch.Tensor,
        n_samples:      int,
        T_future:       int = 10,
    ) -> torch.Tensor:
        """
        Sample n_samples independent residual trajectories per device.
        Returns: (n_samples, B, T_future, n_features)
        """
        deltas = [self.forward(z_prefix_last, T_K, x0, log_t_suffix, T_future=T_future)
                  for _ in range(n_samples)]
        return torch.stack(deltas, dim=0)


# ---------------------------------------------------------------------------
# CRPS loss (differentiable, Monte Carlo approximation)
# ---------------------------------------------------------------------------

def crps_mc_loss(
    samples:  torch.Tensor,   # (S, B, T, F)  S=n_samples prediction trajectories
    x_true:   torch.Tensor,   # (B, T, F)
    mask:     torch.Tensor,   # (B, T)  1=valid, 0=missing
    prefix_len: int,
) -> torch.Tensor:
    """
    Differentiable CRPS via Monte Carlo energy score.

    CRPS(F, y) ≈ E_s[|s - y|] - 0.5 * E_{s,s'}[|s - s'|]

    Only computed over future (post-prefix) time steps.
    Handles NaN in x_true via masking.
    """
    S, B, T, F = samples.shape

    # Future-only binary mask
    future_mask = mask.clone()
    future_mask[:, :prefix_len] = 0.0
    valid = (future_mask > 0).unsqueeze(0).unsqueeze(-1)  # (1, B, T, 1)

    x_true_exp = x_true.unsqueeze(0).expand(S, -1, -1, -1)  # (S, B, T, F)
    nan_mask   = ~torch.isnan(x_true_exp)
    final_mask = valid & nan_mask                             # (S, B, T, F)

    if final_mask.sum() == 0:
        return samples.mean() * 0.0  # differentiable zero

    # Replace NaN with 0 to avoid gradient issues
    samples = torch.nan_to_num(samples, nan=0.0)  # handle NaN decoder outputs
    s_clean = torch.where(nan_mask, samples, torch.zeros_like(samples))
    y_clean = torch.where(nan_mask, x_true_exp, torch.zeros_like(x_true_exp))

    # E[|s - y|]
    term1 = (torch.abs(s_clean - y_clean) * final_mask.float()).sum() / final_mask.float().sum().clamp(min=1)

    # E[|s - s'|] via random pairs (avoid O(S^2) by sampling n_pairs)
    n_pairs = min(S, 8)
    idx1 = torch.randperm(S, device=samples.device)[:n_pairs]
    idx2 = torch.randperm(S, device=samples.device)[:n_pairs]
    # Use future mask (collapse S dim)
    pair_mask = (future_mask > 0).unsqueeze(-1) & ~torch.isnan(x_true)  # (B, T, F)
    pair_mask_exp = pair_mask.unsqueeze(0).expand(n_pairs, -1, -1, -1)  # (n_pairs, B, T, F)
    s1 = s_clean[idx1]   # (n_pairs, B, T, F)
    s2 = s_clean[idx2]
    term2_vals = torch.abs(s1 - s2) * pair_mask_exp.float()
    term2 = term2_vals.sum() / pair_mask_exp.float().sum().clamp(min=1)

    return term1 - 0.5 * term2


# ---------------------------------------------------------------------------
# AR(1) consistency regularizer
# ---------------------------------------------------------------------------

def ar1_consistency_loss(
    deltas: torch.Tensor,   # (n_samples, B, 5) single-step innovations
    target_rho: float = AR1_TARGET_RHO,
) -> torch.Tensor:
    """
    Encourages the generated innovations to have AR(1)-like autocorrelation.

    Since we generate single-step innovations, we estimate the variance
    structure across the sample dimension and encourage it to match the
    empirical AR(1) statistics of the training residuals.

    Loss: ||Var(δ) - target_var||^2
    The main goal is to prevent δ → 0 collapse (scale regularization).
    """
    S, B, _ = deltas.shape
    # Variance across samples (per device, per state) → should be non-trivial
    var_est = deltas.var(dim=0)        # (B, 5)
    # Only penalise if variance collapses (< 1e-6)
    # This acts as a diversity regularizer, not strict AR(1) matching
    collapse_penalty = torch.relu(1e-4 - var_est).mean()
    return collapse_penalty


# ---------------------------------------------------------------------------
# Monotone physics penalty
# ---------------------------------------------------------------------------

def monotone_latent_penalty(
    z_perturbed: torch.Tensor,   # (B, 5) = z_prefix_last + delta
    z_prefix_last: torch.Tensor, # (B, 5) reference state
) -> torch.Tensor:
    """
    Penalise perturbations that push monotone states (zM=2, zL=3, zC=4) below
    their prefix-boundary values. These states are strictly non-decreasing.

    Loss: mean(relu(z_prefix_last[mono] - z_perturbed[mono]))^2
    """
    mono_idx = [2, 3, 4]  # zM, zL, zC
    z_ref  = z_prefix_last[:, mono_idx]
    z_pert = z_perturbed[:, mono_idx]
    decrease = torch.relu(z_ref - z_pert)  # positive when pert decreases mono state
    return (decrease ** 2).mean()


def variance_matching_loss(
    samples: torch.Tensor,   # (S, B, T, F)
    x_true: torch.Tensor,    # (B, T, F)
    mask: torch.Tensor,      # (B, T)
    prefix_len: int,
) -> torch.Tensor:
    """
    Match the generated residual variance to the empirical residual variance
    of the deterministic mean forecast on the current batch.
    """
    S, B, T, F = samples.shape
    future_mask = mask.clone()
    future_mask[:, :prefix_len] = 0.0
    valid = (future_mask > 0).unsqueeze(0).unsqueeze(-1).float()  # (1, B, T, 1)

    samples_clean = torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    x_true_clean = torch.nan_to_num(x_true, nan=0.0, posinf=0.0, neginf=0.0)

    x_true_exp = x_true_clean.unsqueeze(0).expand(S, -1, -1, -1)
    residuals = (samples_clean - samples_clean.mean(dim=0, keepdim=True)).detach()
    generated_std = residuals.std(dim=(0, 1, 2)).clamp(min=1e-6)  # (F,)

    emp_resid = (x_true_exp - x_true_exp.mean(dim=0, keepdim=True)).detach()
    target_std = emp_resid.std(dim=(0, 1, 2)).clamp(min=1e-6)  # (F,)
    return ((generated_std - target_std) ** 2).mean()


def pinball_loss(
    samples: torch.Tensor,   # (S, B, T, F)
    x_true: torch.Tensor,    # (B, T, F)
    mask: torch.Tensor,      # (B, T)
    prefix_len: int,
    taus: tuple = (0.05, 0.5, 0.95),
) -> torch.Tensor:
    """
    Quantile/pinball loss on the future horizon so the generator learns
    a wider and better-calibrated predictive interval, not just a narrow mean.
    """
    S, B, T, F = samples.shape
    future_mask = mask.clone()
    future_mask[:, :prefix_len] = 0.0
    valid = (future_mask > 0).unsqueeze(0).unsqueeze(-1).float()  # (1, B, T, 1)

    if valid.sum() <= 0:
        return samples.mean() * 0.0

    samples_clean = torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    x_true_clean = torch.nan_to_num(x_true, nan=0.0, posinf=0.0, neginf=0.0)
    x_true_exp = x_true_clean.unsqueeze(0).expand(S, -1, -1, -1)
    losses = []
    for tau in taus:
        q = torch.quantile(samples_clean, tau, dim=0)  # (B, T, F)
        diff = q - x_true_exp
        pinball = torch.maximum(tau * diff, (tau - 1.0) * diff)
        losses.append((pinball * valid).sum() / valid.sum().clamp(min=1))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# ODE integration helper for Stage 4A
# ---------------------------------------------------------------------------

@torch.no_grad()
def _integrate_from_z0_nograd(
    z0:        torch.Tensor,   # (B, 5)
    T_K:       torch.Tensor,   # (B,)
    times_h:   torch.Tensor,   # (B, T)
    alpha:     torch.Tensor,   # (B,)
    ode,
    start_step: int,
) -> torch.Tensor:
    """
    Integrate ODE from z0 (at time index start_step) forward.
    Returns (B, T-start_step, 5) — future latent trajectory.
    """
    B, T = times_h.shape
    steps = T - start_step
    z_traj = []
    z = z0.clone()
    for i in range(steps):
        t_idx = start_step + i
        if t_idx + 1 >= T:
            z_traj.append(z)
            break
        dt = (times_h[:, t_idx + 1] - times_h[:, t_idx]).clamp(min=1e-6)
        z = ode.integrate(z, T_K, dt, alpha)
        z_traj.append(z)
    if not z_traj:
        return z0.unsqueeze(1)
    return torch.stack(z_traj, dim=1)   # (B, steps, 5)


def _integrate_from_z0_grad(
    z0:        torch.Tensor,   # (B, 5)
    T_K:       torch.Tensor,   # (B,)
    times_h:   torch.Tensor,   # (B, T)
    alpha:     torch.Tensor,   # (B,)
    ode,
    start_step: int,
) -> torch.Tensor:
    """
    Same as above but with gradients (for loss backprop through δ_z → z0 → trajectory).
    """
    B, T = times_h.shape
    steps = T - start_step
    z_traj = []
    z = z0
    for i in range(steps):
        t_idx = start_step + i
        if t_idx + 1 >= T:
            z_traj.append(z)
            break
        dt = (times_h[:, t_idx + 1] - times_h[:, t_idx]).clamp(min=1e-6)
        z = ode.integrate(z, T_K, dt, alpha)
        z_traj.append(z)
    if not z_traj:
        return z0.unsqueeze(1)
    return torch.stack(z_traj, dim=1)   # (B, steps, 5)


# ---------------------------------------------------------------------------
# Stage 4A training loop
# ---------------------------------------------------------------------------
# Cache helper: pre-compute ODE trajectories for the full dataset
# ---------------------------------------------------------------------------

def _cache_trajectories(model, dataloader, device, _forward_fn, prefix_len: int) -> list:
    """
    Pre-compute all (z_enc, z_pfx, alpha, x_true, mask, T_K, times, log_t, x0)
    for the full dataloader. Since the backbone is frozen, these are reusable
    across all generator training epochs. Returns a list of record dicts.
    """
    log.info("  Pre-caching ODE trajectories (frozen backbone) ...")
    records = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            z_enc, alpha, x_hat, x_true, feature_mask, mask, times, T_K, x0 = \
                _forward_fn(model, batch, device)

            B, T_len, _ = x_true.shape
            if T_len <= prefix_len + 1:
                continue

            plen  = min(prefix_len, z_enc.shape[1] - 1)
            z_pfx = z_enc[:, plen - 1, :].cpu()
            log_t = torch.log1p(times[:, plen - 1].clamp(min=0)).cpu()
            z_ref = z_enc[:, 0, :].cpu()

            records.append({
                "z_enc":   z_enc.cpu(),    # (B, T, 5) — encoder latent trajectory
                "z_pfx":   z_pfx,          # (B, 5)   — encoder state at prefix end
                "z_ref":   z_ref,          # (B, 5)   — encoder state at t=0
                "x_hat":   x_hat.cpu(),    # (B, T, F) — deterministic mean prediction
                "alpha":   alpha.cpu(),    # (B,)
                "x_true":  x_true.cpu(),   # (B, T, F)
                "mask":    mask.cpu(),     # (B, T)
                "T_K":     T_K.cpu(),      # (B,)
                "times":   times.cpu(),    # (B, T)
                "log_t":   log_t,          # (B,)
                "x0":      batch["x"][:, 0, :].cpu(),  # (B, F)
                "plen":    plen,
                "T_len":   T_len,
            })
    log.info("  Cached %d batches (%d devices total)",
             len(records), sum(r["z_pfx"].shape[0] for r in records))
    return records


# ---------------------------------------------------------------------------

def train_stage4a(
    model,
    generator: ResidualLatentInnovationGenerator,
    train_dl,
    val_dl,
    device: torch.device,
    mods: dict = None,
    n_train_samples: int = STAGE4A_N_TRAIN_SAMPLES,
    epochs: int          = STAGE4A_EPOCHS,
    lr: float            = STAGE4A_LR,
    patience: int        = STAGE4A_PATIENCE,
    output_dir: str      = None,
) -> dict:
    """
    Train Stage 4A generator with CRPS loss.
    Backbone is FROZEN — only generator parameters are updated.

    Speed optimisation: pre-cache all ODE trajectories once before training.
    Each epoch only calls generator + decoder (no per-epoch ODE forward pass).
    Gradient path: CRPS → x_pred → decoder → z_boundary → delta → generator.
    """
    _forward = mods["train"]._forward if mods else None
    assert _forward is not None, "mods dict with 'train' module is required"
    if output_dir is None:
        output_dir = cfg.CHECKPOINT_DIR
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "stage4a_best.pt")

    # Freeze the deterministic backbone
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    generator = generator.to(device)
    generator.train()

    opt   = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=patience // 2, factor=0.5)

    prefix_len = cfg.STAGE3_PREFIX_LEN

    # ---- Pre-cache trajectories (one-time cost) ----
    log.info("Pre-caching training trajectories (frozen backbone) ...")
    train_cache = _cache_trajectories(model, train_dl, device, _forward, prefix_len)
    log.info("Pre-caching validation trajectories ...")
    val_cache   = _cache_trajectories(model, val_dl,   device, _forward, prefix_len)

    best_val_crps = float("inf")
    best_epoch    = 0
    no_improve    = 0
    history       = {"train_crps": [], "val_crps": [], "log_scale": []}

    log.info("=" * 60)
    log.info("=== Stage 4A: Residual Latent Innovation Generator ===")
    log.info("  Epochs=%d  n_train_samples=%d  lr=%.2e", epochs, n_train_samples, lr)
    log.info(
        "  Loss weights: CRPS(λ=%.2f) AR1(λ=%.3f) Var(λ=%.3f) Pin(λ=%.3f) ScaleReg(λ=%.3f)",
        LAMBDA_CRPS, LAMBDA_AR1, LAMBDA_VAR_MATCH, LAMBDA_PINBALL, LAMBDA_SCALE_REG,
    )
    log.info("  Cache: %d train batches, %d val batches", len(train_cache), len(val_cache))
    log.info("=" * 60)

    for epoch in range(1, epochs + 1):
        generator.train()
        t0 = time.time()
        train_crps_total = 0.0
        train_ar1_total  = 0.0
        train_var_total  = 0.0
        train_pin_total  = 0.0
        train_mono_total = 0.0
        n_train_batches  = 0

        # ---- Training: iterate over pre-cached records (no ODE calls) ----
        for rec in train_cache:
            z_pfx  = rec["z_pfx"].to(device)    # (B, 5)
            x_hat  = rec["x_hat"].to(device)    # (B, T, F) — deterministic mean
            x_true = rec["x_true"].to(device)   # (B, T, F)
            mask   = rec["mask"].to(device)     # (B, T)
            T_K    = rec["T_K"].to(device)      # (B,)
            log_t  = rec["log_t"].to(device)    # (B,)
            x0     = rec["x0"].to(device)       # (B, F)
            plen   = rec["plen"]
            T_len  = rec["T_len"]
            T_future = T_len - plen

            if T_future <= 0:
                continue

            # ---- Generate n_train_samples observation-space residual trajectories ----
            # δ_x: (n_train_samples, B, T_future, F)
            deltas = generator.sample_n(z_pfx, T_K, x0, log_t,
                                        n_train_samples, T_future=T_future)

            # ---- Stochastic predictions: x_mean + δ_x ----
            # x_hat_future: (B, T_future, F) — deterministic forecast from prefix end
            x_hat_future  = x_hat[:, plen:, :]                         # (B, T_future, F)
            x_hat_future  = x_hat_future.detach()
            x_hat_exp     = x_hat_future.unsqueeze(0)                  # (1, B, T_future, F)

            x_pred_future = x_hat_exp + deltas                        # (S, B, T_future, F)

            # Pad prefix with mean predictions (no stochasticity in prefix)
            x_prefix_exp  = x_hat[:, :plen, :].detach().unsqueeze(0).expand(
                                n_train_samples, -1, -1, -1)           # (S, B, plen, F)
            x_pred_full   = torch.cat([x_prefix_exp, x_pred_future], dim=2)  # (S, B, T, F)

            crps = crps_mc_loss(x_pred_full, x_true, mask, prefix_len=plen)

            # AR(1) diversity: penalise if residuals collapse or over-correlate
            if T_future > 2:
                d_t   = torch.nan_to_num(deltas[:, :, 1:, :], nan=0.0, posinf=0.0, neginf=0.0)
                d_tm1 = torch.nan_to_num(deltas[:, :, :-1, :], nan=0.0, posinf=0.0, neginf=0.0)
                cov  = (d_t * d_tm1).mean(dim=(0, 1, 2))   # (F,)
                var  = (d_tm1 ** 2).mean(dim=(0, 1, 2)).clamp(min=1e-8)
                rho_emp = (cov / var).clamp(-1, 1)          # (F,)
                rho_target = generator.rho.detach()          # (F,)
                ar1_loss = ((rho_emp - rho_target) ** 2).mean()
            else:
                ar1_loss = deltas.mean() * 0.0  # differentiable zero

            # Match generated residual variance to empirical residual variance
            future_true = torch.nan_to_num(x_true[:, plen:, :], nan=0.0, posinf=0.0, neginf=0.0)
            future_hat  = torch.nan_to_num(x_hat[:, plen:, :], nan=0.0, posinf=0.0, neginf=0.0)
            emp_resid   = (future_true - future_hat).detach()
            target_std  = emp_resid.std(dim=(0, 1)).clamp(min=1e-4)
            gen_std     = deltas.std(dim=(0, 1, 2)).clamp(min=1e-4)
            var_loss    = ((gen_std - target_std) ** 2).mean()

            # Coverage proxy via pinball loss on future steps
            pinball = pinball_loss(x_pred_future, future_true, mask[:, plen:], prefix_len=0)

            # Scale regularizer: prevent collapse (std should be > 0.01)
            sigma_emp   = deltas.std(dim=(0, 1, 2))          # (F,)
            scale_reg   = torch.relu(0.01 - sigma_emp).mean()

            loss = (
                LAMBDA_CRPS * crps
                + LAMBDA_AR1 * ar1_loss
                + LAMBDA_VAR_MATCH * var_loss
                + LAMBDA_PINBALL * pinball
                + LAMBDA_SCALE_REG * scale_reg
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), STAGE4A_GRAD_CLIP)
            opt.step()

            train_crps_total += crps.item()
            train_ar1_total  += ar1_loss.item() if isinstance(ar1_loss, torch.Tensor) else 0.0
            train_var_total  += var_loss.item() if isinstance(var_loss, torch.Tensor) else 0.0
            train_pin_total  += pinball.item() if isinstance(pinball, torch.Tensor) else 0.0
            train_mono_total += scale_reg.item()
            n_train_batches  += 1

        if n_train_batches == 0:
            log.warning("Epoch %d: no valid training batches!", epoch)
            continue

        train_crps_avg = train_crps_total / n_train_batches

        # ---- Validation: use pre-cached records ----
        generator.eval()
        val_crps_total = 0.0
        n_val_batches  = 0
        with torch.no_grad():
            for rec in val_cache:
                z_pfx  = rec["z_pfx"].to(device)
                x_hat  = rec["x_hat"].to(device)
                x_true = rec["x_true"].to(device)
                mask   = rec["mask"].to(device)
                T_K    = rec["T_K"].to(device)
                log_t  = rec["log_t"].to(device)
                x0     = rec["x0"].to(device)
                plen   = rec["plen"]
                T_len  = rec["T_len"]
                T_future = T_len - plen

                if T_future <= 0:
                    continue

                deltas_v   = generator.sample_n(z_pfx, T_K, x0, log_t,
                                                STAGE4A_N_VAL_SAMPLES, T_future=T_future)
                x_hat_fut  = x_hat[:, plen:, :].unsqueeze(0)
                x_pred_fut = x_hat_fut + deltas_v
                x_pre_exp  = x_hat[:, :plen, :].unsqueeze(0).expand(
                                 STAGE4A_N_VAL_SAMPLES, -1, -1, -1)
                x_pred_v   = torch.cat([x_pre_exp, x_pred_fut], dim=2)
                val_crps   = crps_mc_loss(x_pred_v, x_true, mask, prefix_len=plen)
                val_crps_total += val_crps.item()
                n_val_batches  += 1

        val_crps_avg = val_crps_total / max(n_val_batches, 1)
        sched.step(val_crps_avg)

        # Log rho and sigma floor values for monitoring
        rho_vals   = generator.rho.detach().cpu().numpy().tolist()
        sigma_vals = torch.exp(generator.log_scale_floor).detach().cpu().numpy().tolist()
        rho_str    = " ".join(f"{r:.3f}" for r in rho_vals)
        sigma_str  = " ".join(f"{s:.4f}" for s in sigma_vals)

        elapsed = time.time() - t0
        log.info(
            "Epoch %3d/%d | train_CRPS=%.4f  val_CRPS=%.4f | "
            "AR1=%.4f  Var=%.4f  Pin=%.4f  ScaleReg=%.4f | rho=[%s] | sigma_floor=[%s] | %.0fs",
            epoch, epochs,
            train_crps_avg, val_crps_avg,
            train_ar1_total / n_train_batches,
            train_var_total / n_train_batches,
            train_pin_total / n_train_batches,
            train_mono_total / n_train_batches,
            rho_str, sigma_str, elapsed,
        )

        history["train_crps"].append(train_crps_avg)
        history["val_crps"].append(val_crps_avg)
        history["log_scale"].append(sigma_vals)

        if val_crps_avg < best_val_crps:
            best_val_crps = val_crps_avg
            best_epoch    = epoch
            no_improve    = 0
            rho_list   = generator.rho.detach().cpu().numpy().tolist()
            sigma_list = torch.exp(generator.log_scale_floor).detach().cpu().numpy().tolist()
            torch.save({
                "epoch":       epoch,
                "val_crps":    best_val_crps,
                "state_dict":  generator.state_dict(),
                "rho":         rho_list,
                "sigma_floor": sigma_list,
            }, ckpt_path)
            log.info(
                "  ✓ Saved best Stage 4A checkpoint (val_CRPS=%.4f) -> %s",
                best_val_crps,
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    log.info("Stage 4A training complete. Best epoch=%d  val_CRPS=%.4f", best_epoch, best_val_crps)
    return {"best_val_crps": best_val_crps, "best_epoch": best_epoch, "history": history}


# ---------------------------------------------------------------------------
# Stage 4A evaluation: compare vs AR(1) baseline
# ---------------------------------------------------------------------------

def evaluate_stage4a(
    model,
    generator: ResidualLatentInnovationGenerator,
    test_dl,
    device: torch.device,
    mods: dict = None,
    n_eval_samples: int = 100,
    prefix_len: int     = None,
    output_dir: str     = None,
) -> dict:
    """
    Full probabilistic evaluation of Stage 4A generator.
    Reuses the metric computation from 10_stochastic_residual.py.
    Also runs the AR(1) baseline for comparison (CRPSS relative to AR(1)).
    """
    import pickle
    _forward = mods["train"]._forward if mods else None
    StochasticResidualModel = mods["stoch"].StochasticResidualModel if mods else None
    assert _forward is not None and StochasticResidualModel is not None, "mods required"

    if prefix_len is None:
        prefix_len = cfg.STAGE3_PREFIX_LEN
    if output_dir is None:
        output_dir = cfg.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    generator.eval()

    all_x_true, all_x_mean, all_T_K, all_times, all_mask = [], [], [], [], []

    with torch.no_grad():
        for batch in test_dl:
            x_raw = batch["x"].to(device)
            T_K   = batch["T_K"].to(device)
            times = batch["times_h"].to(device)
            mask  = batch["mask"].to(device)
            x0    = x_raw[:, 0, :]

            B, T_len, F = x_raw.shape
            if T_len <= prefix_len + 1:
                continue

            z_enc, alpha, x_hat, x_true, feature_mask, _, _, _, _ = \
                _forward(model, batch, device)

            # Deterministic mean (ODE prediction)
            all_x_mean.append(x_hat.cpu().numpy())
            all_x_true.append(x_true.cpu().numpy())
            all_T_K.append(T_K.cpu().numpy())
            all_times.append(times.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

    if not all_x_true:
        log.warning("No valid test batches for Stage 4A evaluation.")
        return {}

    x_true_np = np.concatenate(all_x_true, axis=0)    # (N, T, F)
    x_mean_np = np.concatenate(all_x_mean, axis=0)    # (N, T, F)
    T_K_np    = np.concatenate(all_T_K)               # (N,)
    times_np  = np.concatenate(all_times, axis=0)     # (N, T)
    mask_np   = np.concatenate(all_mask, axis=0)      # (N, T)

    N, T_len, F = x_true_np.shape

    # ---- Stage 4A samples (observation-space: x_mean + δ_x) ----
    log.info("Generating Stage 4A samples (n=%d)...", n_eval_samples)
    # Use cached trajectories for efficiency
    eval_cache = _cache_trajectories(model, test_dl, device, _forward, prefix_len)

    # Reconstruct full arrays in dataset order
    samples_4a = np.full((n_eval_samples, N, T_len, F), np.nan)
    n_placed   = 0

    with torch.no_grad():
        for rec in eval_cache:
            z_pfx    = rec["z_pfx"].to(device)
            x_hat_r  = rec["x_hat"].to(device)    # (B, T, F) — mean prediction
            T_K      = rec["T_K"].to(device)
            log_t    = rec["log_t"].to(device)
            x0       = rec["x0"].to(device)
            plen_r   = rec["plen"]
            T_len_r  = rec["T_len"]
            T_future = T_len_r - plen_r
            B        = z_pfx.shape[0]

            if T_future <= 0:
                n_placed += B
                continue

            deltas_v = generator.sample_n(
                z_pfx, T_K, x0, log_t, n_eval_samples, T_future=T_future
            )  # (n_eval, B, T_future, F)

            x_hat_np    = x_hat_r.cpu().numpy()   # (B, T, F)
            deltas_np   = deltas_v.cpu().numpy()  # (n_eval, B, T_future, F)
            b_end = min(n_placed + B, N)
            actual_B = b_end - n_placed

            for s in range(n_eval_samples):
                x_s = x_hat_np[:actual_B].copy()                  # (actual_B, T, F)
                x_s[:, plen_r:, :] += deltas_np[s, :actual_B, :, :]
                samples_4a[s, n_placed:b_end, :T_len_r, :] = x_s
            n_placed = b_end
            if n_placed >= N:
                break

    # ---- Compute metrics using the same framework as AR(1) ----
    # Use a dummy StochasticResidualModel just for metrics computation
    dummy = StochasticResidualModel()

    metrics_4a = dummy.compute_metrics(
        x_true_np, samples_4a, mask_np,
        T_K=T_K_np, prefix_len=prefix_len,
        x_mean=x_mean_np, times_h=times_np,
    )

    log.info(
        "Stage 4A Results: CRPS=%.4f  CRPSS=%.3f  Cov90=%.2f  MACE=%.4f",
        metrics_4a.get("crps_overall", float("nan")),
        metrics_4a.get("crpss_overall", float("nan")),
        metrics_4a.get("coverage_90_overall", float("nan")),
        metrics_4a.get("reliability_mace", float("nan")),
    )

    # Save results
    results = {
        "stage4a_metrics": metrics_4a,
        "n_eval_samples": n_eval_samples,
    }
    out_path = os.path.join(output_dir, "evaluation_results_stage4a.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(results, fh)
    log.info("Stage 4A evaluation saved → %s", out_path)

    # ---- Compare with AR(1) benchmark ----
    ar1_path = os.path.join(output_dir, "evaluation_results_stage3.pkl")
    if os.path.exists(ar1_path):
        with open(ar1_path, "rb") as fh:
            ar1_data = pickle.load(fh)
        # AR(1) metrics are stored under 'stochastic_residual' key
        ar1_sr   = ar1_data.get("stochastic_residual", ar1_data.get("stoch_metrics", {}))
        ar1_crpss = ar1_sr.get("crpss_overall", float("nan"))
        ar1_cov90 = ar1_sr.get("coverage_90_overall", float("nan"))
        s4a_crpss = metrics_4a.get("crpss_overall", float("nan"))
        s4a_cov90 = metrics_4a.get("coverage_90_overall", float("nan"))
        log.info("=" * 55)
        log.info("  Comparison: Stage4A vs AR(1) baseline")
        log.info("  AR(1):    CRPSS=%.3f  Cov90=%.2f", ar1_crpss, ar1_cov90)
        log.info("  Stage4A:  CRPSS=%.3f  Cov90=%.2f", s4a_crpss, s4a_cov90)
        go_to_4b = (s4a_crpss > ar1_crpss + 0.05 and s4a_cov90 > 0.80)
        log.info("  Proceed to Stage 4B: %s", "YES ✓" if go_to_4b else "NO — stay with AR(1)")
        log.info("=" * 55)
        results["ar1_crpss"]     = ar1_crpss
        results["go_to_stage4b"] = go_to_4b
    else:
        log.warning("AR(1) baseline results not found at %s", ar1_path)

    return results


# ---------------------------------------------------------------------------
# Module import aliases (for use from main.py or other scripts)
# ---------------------------------------------------------------------------

# Allow importing like: from _pi_stage4a import ResidualLatentInnovationGenerator
def _ensure_module_alias():
    """Register this module under _pi_stage4a alias for convenient imports."""
    import importlib
    mod_name = "_pi_stage4a"
    if mod_name not in sys.modules:
        sys.modules[mod_name] = sys.modules[__name__]

_ensure_module_alias()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Stage 4A Residual Latent Innovation Generator")
    p.add_argument("--epochs",           type=int,   default=STAGE4A_EPOCHS)
    p.add_argument("--n-train-samples",  type=int,   default=STAGE4A_N_TRAIN_SAMPLES)
    p.add_argument("--n-eval-samples",   type=int,   default=100)
    p.add_argument("--lr",               type=float, default=STAGE4A_LR)
    p.add_argument("--hidden-dim",       type=int,   default=STAGE4A_HIDDEN_DIM)
    p.add_argument("--noise-dim",        type=int,   default=STAGE4A_NOISE_DIM)
    p.add_argument("--stage3-ckpt",      type=str,   default=None,
                   help="Path to Stage 3 checkpoint; defaults to cfg.CHECKPOINT_DIR/stage3_best.pt")
    p.add_argument("--output-dir",       type=str,   default=None)
    p.add_argument("--skip-train",       action="store_true",
                   help="Skip training; load existing Stage 4A checkpoint for evaluation only")
    p.add_argument("--device",           type=str,   default="cpu")
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)

    output_dir  = args.output_dir or cfg.OUTPUT_PATH
    ckpt_dir    = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")

    # ---- Load all pipeline modules ----
    log.info("Loading pipeline modules ...")
    mods = _load_all()
    train_mod = mods["train"]

    # ---- Load dataset and build dataloaders ----
    log.info("Loading dataset ...")
    import pickle as _pkl
    prep_path = cfg.PROCESSED_DATA_PATH
    if not os.path.exists(prep_path):
        log.error("Processed dataset not found: %s", prep_path)
        log.error("Run main.py --mode preprocess first.")
        sys.exit(1)
    with open(prep_path, "rb") as fh:
        dataset = _pkl.load(fh)

    from torch.utils.data import DataLoader
    split    = dataset["split"]
    all_idx  = list(range(len(dataset["device_ids"])))
    train_idx = split.get("train", all_idx[:int(0.7 * len(all_idx))])
    val_idx   = split.get("val",   all_idx[int(0.7 * len(all_idx)):])
    test_idx  = split.get("test",  val_idx)

    train_ds = train_mod.DeviceDegradationDataset(dataset, train_idx)
    val_ds   = train_mod.DeviceDegradationDataset(dataset, val_idx)
    test_ds  = train_mod.DeviceDegradationDataset(dataset, test_idx)
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                          shuffle=True,  collate_fn=train_mod.collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                          shuffle=False, collate_fn=train_mod.collate_fn)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                          shuffle=False, collate_fn=train_mod.collate_fn)

    # ---- Build backbone model ----
    log.info("Building PI-TimeGAN model ...")
    model = _build_model(mods).to(device)

    stage3_ckpt = args.stage3_ckpt or os.path.join(ckpt_dir, "stage3_best.pt")
    if not os.path.exists(stage3_ckpt):
        log.error("Stage 3 checkpoint not found: %s", stage3_ckpt)
        sys.exit(1)
    ckpt = torch.load(stage3_ckpt, map_location=device)
    # Stage checkpoints use key "model_state" (not "model_state_dict")
    model_state = ckpt.get("model_state", ckpt.get("model_state_dict", None))
    if model_state is None:
        log.error("Checkpoint %s has no 'model_state' key (keys: %s)", stage3_ckpt, list(ckpt.keys()))
        sys.exit(1)
    model_state_clean = {k: v for k, v in model_state.items() if k != "decoder.mask"}
    model.load_state_dict(model_state_clean, strict=False)
    sel_val = ckpt.get("selection_value", ckpt.get("val_mse", ckpt.get("val_loss", float("nan"))))
    log.info("Loaded Stage 3 checkpoint: %s  (metric=%.5f)", stage3_ckpt, sel_val)

    # ---- Build generator ----
    generator = ResidualLatentInnovationGenerator(
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # ---- Train (or load) ----
    stage4a_ckpt = os.path.join(ckpt_dir, "stage4a_best.pt")
    if args.skip_train and os.path.exists(stage4a_ckpt):
        ckpt4a = torch.load(stage4a_ckpt, map_location=device)
        generator.load_state_dict(ckpt4a["state_dict"])
        log.info("Loaded existing Stage 4A checkpoint (val_CRPS=%.4f)", ckpt4a.get("val_crps", float("nan")))
    else:
        train_stage4a(
            model, generator, train_dl, val_dl, device,
            mods=mods,
            n_train_samples=args.n_train_samples,
            epochs=args.epochs,
            lr=args.lr,
            output_dir=ckpt_dir,
        )
        # Reload best checkpoint
        if os.path.exists(stage4a_ckpt):
            ckpt4a = torch.load(stage4a_ckpt, map_location=device)
            generator.load_state_dict(ckpt4a["state_dict"])
            log.info("Reloaded best Stage 4A checkpoint.")

    # ---- Evaluate ----
    log.info("Running Stage 4A evaluation ...")
    results = evaluate_stage4a(
        model, generator, test_dl, device,
        mods=mods,
        n_eval_samples=args.n_eval_samples,
        output_dir=results_dir,
    )

    log.info("Stage 4A complete.")
    return results


if __name__ == "__main__":
    main()
