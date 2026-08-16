"""
07_losses.py
============
All loss functions for the PI-TimeGAN pipeline.

Losses implemented
------------------
1. reconstruction_loss      – MSE between decoded x_hat and observed x (masked)
2. ode_residual_loss        – MSE between ODE-predicted z and encoder z
3. bounds_loss              – penalty if latent states leave [0,1]
4. monotonicity_loss        – penalty if zM, zL, zC decrease over time
5. zc_prefix_separation_loss – penalty if zC grows too fast inside prefix
6. temperature_ordering_loss – penalty if higher-T devices degrade slower
6. adversarial_loss_generator   – generator adversarial loss (non-saturating)
7. adversarial_loss_discriminator – discriminator GAN loss

Total training objective:
  L = λ_recon  * L_recon
    + λ_ode    * L_ode
    + λ_bounds * L_bounds
    + λ_mono   * L_mono
    + λ_temp   * L_temp
    + λ_adv    * L_adv

All losses return scalar tensors.
"""

import torch
import torch.nn.functional as F
from typing import Optional

import config as cfg

EPS = cfg.EPSILON


def _resolve_leakage_floors(floors, device, dtype):
    if floors is None:
        return torch.full((2,), cfg.LEAKAGE_FLOOR_DEFAULT, device=device, dtype=dtype)
    if isinstance(floors, dict):
        vals = [floors.get(name, cfg.LEAKAGE_FLOOR_DEFAULT) for name in ("IDLeak", "IGLeak")]
        return torch.tensor(vals, device=device, dtype=dtype)
    if isinstance(floors, torch.Tensor):
        if floors.ndim == 0:
            return floors.to(device=device, dtype=dtype).expand(2)
        if floors.ndim == 2:
            # Collated batch of shape (B, 2): all devices share the same floor
            # (computed from training set statistics), so take mean across batch.
            return floors.float().mean(dim=0).to(device=device, dtype=dtype)
        return floors.to(device=device, dtype=dtype)
    if isinstance(floors, (list, tuple)):
        return torch.tensor(list(floors), device=device, dtype=dtype)
    return torch.tensor([floors, floors], device=device, dtype=dtype)


def leakage_confidence_weights(
    x_true: torch.Tensor,
    feature_mask: torch.Tensor,
    floors=None,
    floor_margin: float = None,
) -> torch.Tensor:
    """Down-weight leakage values that stay close to the estimated floor."""
    if floor_margin is None:
        floor_margin = cfg.LEAKAGE_CONFIDENCE_MARGIN
    if x_true.dim() != 3 or x_true.shape[-1] != cfg.FEATURE_DIM:
        raise ValueError("x_true must have shape (B, T, 6)")

    weights = torch.ones_like(x_true, dtype=torch.float32)
    leakage_idx = [cfg.FEATURES.index(name) for name in ("IDLeak", "IGLeak")]
    floors_t = _resolve_leakage_floors(floors, x_true.device, x_true.dtype)
    for local_idx, global_idx in enumerate(leakage_idx):
        valid = feature_mask[..., global_idx] & torch.isfinite(x_true[..., global_idx])
        if not valid.any():
            continue
        abs_vals = torch.abs(x_true[..., global_idx][valid])
        floor_abs = abs(float(floors_t[local_idx].detach().cpu()))
        floor_abs = floor_abs if torch.isfinite(torch.tensor(floor_abs)) else cfg.LEAKAGE_FLOOR_DEFAULT
        margin = max(floor_abs, 1e-12) + floor_margin
        near_floor = abs_vals <= margin
        conf = torch.where(
            near_floor,
            torch.full_like(abs_vals, cfg.LEAKAGE_CONFIDENCE_MIN),
            torch.full_like(abs_vals, cfg.LEAKAGE_CONFIDENCE_MAX),
        )
        conf = conf.clamp(cfg.LEAKAGE_CONFIDENCE_MIN, cfg.LEAKAGE_CONFIDENCE_MAX)
        w = torch.zeros_like(x_true[..., global_idx], dtype=torch.float32)
        w[valid] = conf
        weights[..., global_idx] = w
    return weights


def weighted_feature_huber_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    feature_mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    beta: float = 0.2,
) -> torch.Tensor:
    """Feature-wise Huber loss with optional per-point weights."""
    feature_losses = []
    for fi in range(x_true.shape[-1]):
        valid = (
            feature_mask[:, :, fi]
            & torch.isfinite(x_true[:, :, fi])
            & torch.isfinite(x_pred[:, :, fi])
        )
        if not valid.any():
            continue
        pred_valid = x_pred[:, :, fi][valid]
        true_valid = x_true[:, :, fi][valid]
        if weights is None:
            weight_valid = torch.ones_like(pred_valid)
        else:
            weight_valid = weights[:, :, fi][valid].reshape_as(pred_valid).float()
        loss_f = F.smooth_l1_loss(pred_valid, true_valid, reduction="none", beta=beta)
        if weight_valid.sum().clamp_min(EPS) > 0:
            feature_losses.append((loss_f * weight_valid).sum() / weight_valid.sum().clamp_min(EPS))

    if not feature_losses:
        return torch.zeros(1, device=x_pred.device).squeeze()
    return torch.stack(feature_losses).mean()


def leakage_mean_loss(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    feature_mask: torch.Tensor,
    floors=None,
    floor_margin: float = None,
) -> torch.Tensor:
    """Bias the leakage-channel reconstruction loss toward well-resolved points."""
    weights = leakage_confidence_weights(x_true, feature_mask, floors=floors, floor_margin=floor_margin)
    leakage_idx = [cfg.FEATURES.index(name) for name in ("IDLeak", "IGLeak")]
    losses = []
    for fi in leakage_idx:
        valid = (
            feature_mask[:, :, fi]
            & torch.isfinite(x_true[:, :, fi])
            & torch.isfinite(x_hat[:, :, fi])
        )
        if not valid.any():
            continue
        pred_valid = x_hat[:, :, fi][valid]
        true_valid = x_true[:, :, fi][valid]
        w_valid = weights[:, :, fi][valid].reshape_as(pred_valid).float()
        loss_f = F.smooth_l1_loss(pred_valid, true_valid, reduction="none")
        losses.append((loss_f * w_valid).sum() / w_valid.sum().clamp_min(EPS))
    if not losses:
        return torch.zeros(1, device=x_hat.device).squeeze()
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 1. Reconstruction loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    x_hat:  torch.Tensor,
    x_true: torch.Tensor,
    mask:   torch.Tensor,
    feature_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Masked MSE between decoded features and observations.
    Leakage features (x5, x6) are already in log-domain in x_true,
    so a simple MSE is appropriate for all features.

    Args:
        x_hat  : (B, T, 6)  decoder output
        x_true : (B, T, 6)  ground truth (NaN where missing)
        mask   : (B, T)     bool, True = valid time step

    Returns:
        loss : scalar
    """
    if feature_mask is None:
        feature_mask = mask.unsqueeze(2).expand_as(x_hat)
    return balanced_feature_huber_loss(
        x_hat,
        x_true,
        feature_mask,
        beta=cfg.LOSS_HUBER_BETA,
    )


def balanced_feature_huber_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    feature_mask: torch.Tensor,
    beta: float = 0.2,
) -> torch.Tensor:
    """
    Equal-weight reconstruction loss across features regardless of valid count.
    """
    return weighted_feature_huber_loss(
        x_pred,
        x_true,
        feature_mask,
        weights=None,
        beta=beta,
    )


# ---------------------------------------------------------------------------
# 2. ODE residual loss  (delegates to PhysicsODE.ode_residual)
# ---------------------------------------------------------------------------

def ode_residual_loss(
    ode_module,
    z_enc:       torch.Tensor,
    T_K:         torch.Tensor,
    times_h:     torch.Tensor,
    device_alpha: torch.Tensor,
    mask:        torch.Tensor,
) -> torch.Tensor:
    """
    Wrapper around PhysicsODE.ode_residual for consistent API.

    Args:
        ode_module   : PhysicsODE instance
        z_enc        : (B, T, 5)  encoder latent states
        T_K          : (B,)
        times_h      : (B, T)
        device_alpha : (B,)
        mask         : (B, T)

    Returns:
        loss : scalar
    """
    return ode_module.ode_residual(z_enc, T_K, times_h, device_alpha, mask)


# ---------------------------------------------------------------------------
# 3. Bounds loss
# ---------------------------------------------------------------------------

def bounds_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Soft penalty for latent states outside [0, 1].
    Penalises z < 0 and z > 1 quadratically.

    Args:
        z : (B, T, 5) or (B, 5)

    Returns:
        loss : scalar
    """
    below = F.relu(-z)           # > 0 where z < 0
    above = F.relu(z - 1.0)     # > 0 where z > 1
    return (below ** 2 + above ** 2).mean()


# ---------------------------------------------------------------------------
# 4. Monotonicity loss  (zM, zL, zC must not decrease)
# ---------------------------------------------------------------------------

# Indices of monotone states in the latent vector [zG, zB, zM, zL, zC]
MONOTONE_IDX = [2, 3, 4]   # zM, zL, zC


def monotonicity_loss(
    z: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Penalises negative increments in monotone latent states (zM, zL, zC)
    between consecutive valid time steps.

    Args:
        z    : (B, T, 5)
        mask : (B, T)  bool

    Returns:
        loss : scalar
    """
    B, T, D = z.shape
    total = torch.zeros(1, device=z.device)
    count = 0

    z_mono = z[:, :, MONOTONE_IDX]   # (B, T, 3)

    for step in range(1, T):
        valid = mask[:, step - 1] & mask[:, step]
        if not valid.any():
            continue
        delta = z_mono[valid, step, :] - z_mono[valid, step - 1, :]  # (Bv, 3)
        # Penalise decreases
        total = total + F.relu(-delta).mean()
        count += 1

    if count == 0:
        return torch.zeros(1, device=z.device).squeeze()
    return (total / count).squeeze()


def latent_initial_anchor_loss(z_enc: torch.Tensor) -> torch.Tensor:
    """
    Anchor initial monotone latent states near zero.

    zM/zL/zC represent cumulative degradation channels: their t=0 values
    should be close to zero (no damage before stress begins).
    zM receives LAMBDA_ZM_ANCHOR_BOOST for stronger anchoring (was 0.90 before fix).
    zC receives LAMBDA_ZC_ANCHOR_BOOST (moderate) to prevent info migration from zM.
    (debug13: soft-target approach — prevent zC becoming initial-state storage slot)
    """
    if z_enc.dim() != 3 or z_enc.shape[1] < 1:
        return torch.zeros(1, device=z_enc.device).squeeze()

    z0 = z_enc[:, 0, :]
    zM0 = z0[:, 2]
    zL0 = z0[:, 3]
    zC0 = z0[:, 4]

    zm_boost = float(getattr(cfg, "LAMBDA_ZM_ANCHOR_BOOST", 1.0))
    zc_boost = float(getattr(cfg, "LAMBDA_ZC_ANCHOR_BOOST", 1.0))
    return (
        zm_boost * 0.5 * torch.mean(zM0 ** 2) +
        1.0 * torch.mean(zL0 ** 2) +
        zc_boost * 1.0 * torch.mean(zC0 ** 2)
    )


def zc_prefix_separation_loss(
    z_traj: torch.Tensor,
    mask: torch.Tensor,
    prefix_len: int,
    margin: float = None,
) -> torch.Tensor:
    """
    Penalise early zC growth inside the observed prefix.

    This is a prefix-only regulariser: it keeps the cumulative-damage driver
    from absorbing the prefix reconstruction constraint too early, while
    leaving the ODE future rollout free to grow afterwards.
    """
    if margin is None:
        margin = cfg.STAGE3_ZC_PREFIX_MARGIN

    if z_traj.dim() != 3 or z_traj.shape[1] < 2 or prefix_len < 2:
        return torch.zeros(1, device=z_traj.device).squeeze()

    prefix_len = min(prefix_len, z_traj.shape[1])
    zC = z_traj[:, :prefix_len, 4]   # (B, prefix)
    valid = mask[:, :prefix_len].float()
    if valid.sum() <= 0:
        return torch.zeros(1, device=z_traj.device).squeeze()

    # zC should stay near its prefix-start level; later prefix steps are weighted
    # more strongly so early saturation gets penalised even if it is monotone.
    zC0 = zC[:, :1]
    excess = F.relu(zC - zC0 - margin)
    step_weights = torch.linspace(0.0, 1.0, steps=prefix_len, device=z_traj.device)
    step_weights = step_weights.pow(2).view(1, -1)
    weighted = (excess ** 2) * step_weights * valid
    denom = (step_weights * valid).sum().clamp_min(EPS)
    return weighted.sum() / denom


# ---------------------------------------------------------------------------
# 6. Temperature ordering loss
# ---------------------------------------------------------------------------

def temperature_ordering_loss(
    z_traj: torch.Tensor,
    T_K:    torch.Tensor,
    mask:   torch.Tensor,
    final_step: bool = True,
) -> torch.Tensor:
    """
    For pairs of devices with the same initial state but different temperatures,
    the higher temperature device should have greater cumulative damage (zC)
    at any given time.

    Practical implementation: within each batch, form pairs (i, j) where
    T_K[i] < T_K[j], and penalise if zC_i > zC_j at the last valid time step.

    Args:
        z_traj     : (B, T, 5)
        T_K        : (B,)
        mask       : (B, T)
        final_step : if True, compare only at the last valid step

    Returns:
        loss : scalar
    """
    B = z_traj.shape[0]
    if B < 2:
        return torch.zeros(1, device=z_traj.device).squeeze()

    # Get zC at last valid step for each device
    last_idx = torch.zeros(B, dtype=torch.long, device=z_traj.device)
    for b in range(B):
        valid = mask[b].nonzero(as_tuple=False)
        if len(valid) > 0:
            last_idx[b] = valid[-1, 0]

    zC = z_traj[:, :, 4]   # (B, T)  – cumulative damage state
    idx_exp = last_idx.view(B, 1)
    zC_last = zC.gather(1, idx_exp).squeeze(1)   # (B,)

    # Form all pairs (i, j) with T_K[i] < T_K[j]
    total = torch.zeros(1, device=z_traj.device)
    count = 0
    for i in range(B):
        for j in range(B):
            if T_K[i] < T_K[j] - 1.0:   # meaningful temperature difference
                # Higher T (j) should have larger zC
                # Penalise if zC_j <= zC_i
                total = total + F.relu(zC_last[i] - zC_last[j])
                count += 1

    if count == 0:
        return torch.zeros(1, device=z_traj.device).squeeze()
    return (total / count).squeeze()


# ---------------------------------------------------------------------------
# 7. Multi-step prediction loss
# ---------------------------------------------------------------------------

def multistep_prediction_loss(
    decoder,
    ode_module,
    z_enc:        torch.Tensor,
    x_true:       torch.Tensor,
    T_K:          torch.Tensor,
    times_h:      torch.Tensor,
    device_alpha: torch.Tensor,
    mask:         torch.Tensor,
    feature_mask: Optional[torch.Tensor] = None,
    z_ref:        Optional[torch.Tensor] = None,
    start_step:   int = 0,
    decay:        float = None,
    detach_every: int = 4,
    leakage_floor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    From encoder state at `start_step`, integrate the ODE forward to all
    subsequent time steps, decode, and compare to observations.

    Closer steps weight more (via exponential decay in the weight).

    Args:
        decoder      : SparsePhysicsDecoder
        ode_module   : PhysicsODE
        z_enc        : (B, T, 5)   encoder states (used only up to start_step)
        x_true       : (B, T, 6)
        T_K          : (B,)
        times_h      : (B, T)
        device_alpha : (B,)
        mask         : (B, T)
        start_step   : step index from which to begin ODE integration
        decay        : weight decay per step (default: cfg.LAMBDA_MULTISTEP_DECAY)

    Returns:
        loss : scalar
    """
    if decay is None:
        decay = cfg.LAMBDA_MULTISTEP_DECAY

    if z_ref is None:
        z_ref = z_enc[:, 0, :]

    B, T, _ = z_enc.shape
    z_curr = z_enc[:, start_step, :]   # (B, 5)

    total = torch.zeros(1, device=z_enc.device)
    weight_sum = 0.0
    weight = 1.0

    for step in range(start_step + 1, T):
        dt = (times_h[:, step] - times_h[:, step - 1]).clamp(min=cfg.ODE_MIN_DT_H)
        z_next = ode_module.integrate(z_curr, T_K, dt, device_alpha)

        valid  = mask[:, step]             # (B,)
        if valid.any():
            x_tgt = x_true[valid, step, :]
            x_pred = decoder(z_next[valid], z_ref=z_ref[valid])
            if feature_mask is not None:
                fm = feature_mask[valid, step:step + 1, :]
            else:
                fm = torch.ones(
                    (x_pred.shape[0], 1, x_pred.shape[1]),
                    dtype=torch.bool,
                    device=x_pred.device,
                )
            err = balanced_feature_huber_loss(
                x_pred.unsqueeze(1),
                x_tgt.unsqueeze(1),
                fm,
                beta=cfg.LOSS_HUBER_BETA,
            )
            if leakage_floor is not None:
                leak_err = leakage_mean_loss(
                    x_pred.unsqueeze(1),
                    x_tgt.unsqueeze(1),
                    fm,
                    floors=leakage_floor,
                )
                err = err + cfg.STAGE3_LAMBDA_LEAKAGE * leak_err
            if torch.isfinite(err):
                total = total + weight * err
                weight_sum += weight

        if detach_every > 0 and ((step - start_step) % detach_every == 0):
            z_curr = z_next.detach()
        else:
            z_curr = z_next
        weight *= decay

    if weight_sum < 1e-8:
        return torch.zeros(1, device=z_enc.device).squeeze()
    return (total / weight_sum).squeeze()


# ---------------------------------------------------------------------------
# 8. Adversarial losses (non-saturating GAN)
# ---------------------------------------------------------------------------

def adversarial_generator_loss(
    logit_fake_lat: Optional[torch.Tensor],
    logit_fake_obs: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Non-saturating generator loss: - E[log σ(D(fake))]

    Args:
        logit_fake_lat : (B,) or None
        logit_fake_obs : (B,) or None

    Returns:
        loss : scalar
    """
    loss = torch.zeros(1)
    if logit_fake_lat is not None:
        loss = loss.to(logit_fake_lat.device)
        loss = loss + F.softplus(-logit_fake_lat).mean()
    if logit_fake_obs is not None:
        loss = loss.to(logit_fake_obs.device)
        loss = loss + F.softplus(-logit_fake_obs).mean()
    return loss.squeeze()


def adversarial_discriminator_loss(
    logit_real: torch.Tensor,
    logit_fake: torch.Tensor,
) -> torch.Tensor:
    """
    Standard GAN discriminator loss:
      L_D = -E[log σ(D(real))] - E[log σ(1 - D(fake))]
          = E[softplus(-D(real))] + E[softplus(D(fake))]

    Args:
        logit_real : (B,)
        logit_fake : (B,)

    Returns:
        loss : scalar
    """
    loss_real = F.softplus(-logit_real).mean()
    loss_fake = F.softplus(logit_fake).mean()
    return (loss_real + loss_fake).squeeze()


# ---------------------------------------------------------------------------
# 9. Distribution matching loss (Stage 4 generator pre-training)
# ---------------------------------------------------------------------------

def distribution_matching_loss(
    z_fake:   torch.Tensor,
    z_real:   torch.Tensor,
    x_fake:   torch.Tensor,
    x_real:   torch.Tensor,
    mask_real: torch.Tensor,
    return_components: bool = False,
) -> torch.Tensor | dict:
    """
    Pre-trains the generator by matching mean and variance of the
    generated latent and observation distributions to real ones.
    Avoids adversarial instability in early generator training.

    Args:
        z_fake    : (B_fake, T, 5) or (B_fake, 5)
        z_real    : (B_real, T, 5) or (B_real, 5)
        x_fake    : (B_fake, T, 6) or (B_fake, 6)
        x_real    : (B_real, T, 6) or (B_real, 6)
        mask_real : (B_real, T) or None

    Returns:
        loss : scalar
        or dict with component terms when return_components=True
    """
    def _moments(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = t.reshape(-1, t.shape[-1])
        flat = flat[~torch.isnan(flat).any(1)]
        if flat.shape[0] < 2:
            return flat.mean(0), flat.var(0) + EPS
        return flat.mean(0), flat.var(0) + EPS

    mZ_f, vZ_f = _moments(z_fake)
    mZ_r, vZ_r = _moments(z_real)
    mX_f, vX_f = _moments(x_fake)
    mX_r, vX_r = _moments(torch.nan_to_num(x_real, nan=0.0))

    L_z_mean = F.mse_loss(mZ_f, mZ_r.detach())
    L_z_var  = F.mse_loss(vZ_f, vZ_r.detach())
    L_x_mean = F.mse_loss(mX_f, mX_r.detach())
    L_x_var  = F.mse_loss(vX_f, vX_r.detach())

    L_z = L_z_mean + L_z_var
    L_x = L_x_mean + L_x_var
    L_total = L_z + L_x

    if return_components:
        return {
            "z_mean": L_z_mean,
            "z_var": L_z_var,
            "x_mean": L_x_mean,
            "x_var": L_x_var,
            "z": L_z,
            "x": L_x,
            "total": L_total,
        }
    return L_total


# ---------------------------------------------------------------------------
# Combined loss aggregator
# ---------------------------------------------------------------------------

def total_physics_loss(
    x_hat:         torch.Tensor,
    x_true:        torch.Tensor,
    z_enc:         torch.Tensor,
    T_K:           torch.Tensor,
    times_h:       torch.Tensor,
    device_alpha:  torch.Tensor,
    mask:          torch.Tensor,
    ode_module,
    feature_mask:  Optional[torch.Tensor] = None,
    include_temp_order: bool = True,
    leakage_floor: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute all physics-informed losses and return a dict.

    Returns:
        {
          'recon':   reconstruction loss,
          'ode':     ODE residual loss,
          'bounds':  bounds penalty,
          'mono':    monotonicity penalty,
          'anchor':  initial-state anchor penalty,
          'temp':    temperature ordering loss,
          'total':   weighted sum,
        }
    """
    L_recon  = reconstruction_loss(x_hat, x_true, mask, feature_mask=feature_mask)
    L_leak   = leakage_mean_loss(
        x_hat,
        x_true,
        feature_mask if feature_mask is not None else mask.unsqueeze(-1).expand(-1, -1, cfg.FEATURE_DIM),
        floors=leakage_floor,
    ) if leakage_floor is not None else torch.zeros(1, device=x_hat.device).squeeze()
    L_ode    = ode_residual_loss(ode_module, z_enc, T_K, times_h, device_alpha, mask)
    L_bounds = bounds_loss(z_enc)
    L_mono   = monotonicity_loss(z_enc, mask)
    L_anchor = latent_initial_anchor_loss(z_enc)
    L_temp   = (
        temperature_ordering_loss(z_enc, T_K, mask)
        if include_temp_order
        else torch.zeros(1, device=x_hat.device).squeeze()
    )

    L_total = (
        cfg.LAMBDA_RECON   * L_recon  +
        cfg.LAMBDA_ODE     * L_ode    +
        cfg.LAMBDA_BOUNDS  * L_bounds +
        cfg.LAMBDA_MONOTONE * L_mono  +
        cfg.LAMBDA_INITIAL_ANCHOR * L_anchor +
        cfg.LAMBDA_TEMP_ORDER * L_temp +
        cfg.STAGE3_LAMBDA_LEAKAGE * L_leak
    )

    return {
        "recon":  L_recon,
        "ode":    L_ode,
        "bounds": L_bounds,
        "mono":   L_mono,
        "anchor": L_anchor,
        "temp":   L_temp,
        "leakage": L_leak,
        "total":  L_total,
    }
