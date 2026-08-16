"""
05_model_generator.py
=====================
Generator for the PI-TimeGAN framework.

Instead of generating every time step's latent state freely (which would
bypass the physics ODE and risk mode collapse), the generator creates:
  1. An initial latent state z0  (B, 5) ∈ [0,1]^5
  2. A per-device rate multiplier α_device  (B,) ≥ 0

The full latent trajectory is then produced by the physics ODE from z0.
This ensures all generated trajectories are physically consistent.

Architecture
------------
Noise z_noise ~ N(0, I^GENERATOR_NOISE_DIM)
Context  c   = [x0_static (6), T_norm (1)]   →  dim = 7

Generator MLP:
  [z_noise, c]  →  [z0_raw (5),  α_raw (1)]
  z0    = sigmoid(z0_raw)        ∈ [0,1]^5
  alpha = softplus(alpha_raw)    ∈ (0,∞)

The generator is conditioned on initial device parameters and temperature so
that generated trajectories match the input-space statistics of each
temperature group.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class PITimeGANGenerator(nn.Module):
    """
    Conditional generator that produces initial latent state and per-device
    rate multiplier, which are then propagated through the physics ODE.

    Args:
        noise_dim  : dimension of the input Gaussian noise
        hidden_dim : MLP hidden layer size
        context_dim: dimension of the conditioning context (x0_static + T_norm)
    """

    def __init__(
        self,
        noise_dim:   int = cfg.GENERATOR_NOISE_DIM,
        hidden_dim:  int = cfg.GENERATOR_HIDDEN_DIM,
        context_dim: int = cfg.FEATURE_DIM + 1,     # 7
        latent_dim:  int = cfg.LATENT_DIM,           # 5
    ):
        super().__init__()
        self.noise_dim  = noise_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        in_dim = noise_dim + context_dim   # 8 + 7 = 15

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim + 1),  # z0_raw (5) + alpha_raw (1)
        )

        # Phase 3: optional additive prefix-context conditioning
        # Both projections start at zero so they have no effect at initialisation.
        self.prefix_proj = nn.Linear(latent_dim, context_dim, bias=True)
        self.time_proj   = nn.Linear(1,           context_dim, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.prefix_proj.weight)
            nn.init.zeros_(self.prefix_proj.bias)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)

        # Phase 3: learnable z0-perturbation log-scale — one value per latent state.
        # zG/zB (reversible): larger perturbation  → exp(-1) ≈ 0.37
        # zM/zL (monotone):   medium               → exp(-2) ≈ 0.14
        # zC (cumulative):    tiny                 → exp(-4) ≈ 0.018
        _pert_init = torch.tensor([-1.0, -1.0, -2.0, -2.0, -4.0])
        self.z0_pert_log = nn.Parameter(_pert_init)   # (latent_dim,)
        # clipping rate tracker (not a parameter — updated during forward)
        self.register_buffer("_clip_count", torch.zeros(1))
        self.register_buffer("_total_count", torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # Initialise last bias so z0_raw ≈ 0  →  sigmoid ≈ 0.5
        # and alpha_raw ≈ 0.54  →  softplus ≈ 1.0
        with torch.no_grad():
            self.net[-1].bias[-1] = 0.5413

    def forward(
        self,
        x0_static: torch.Tensor,
        T_K:       torch.Tensor,
        noise:     torch.Tensor = None,
        z_prefix_last:   torch.Tensor = None,  # (B, 5)  Phase 3: last prefix encoder state
        log_prefix_time: torch.Tensor = None,  # (B, 1)  Phase 3: log(1+t_prefix_end)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x0_static       : (B, 6)   initial device parameters (normalised)
            T_K             : (B,)     temperature in Kelvin
            noise           : (B, noise_dim) or None (sampled internally)
            z_prefix_last   : (B, 5)   encoder latent state at end of prefix (optional)
            log_prefix_time : (B, 1)   log(1 + last_prefix_time_h)           (optional)

        Returns:
            z0    : (B, 5)   initial latent state ∈ [0,1]^5
            alpha : (B,)     per-device cumulative-damage rate multiplier > 0
        """
        B = x0_static.shape[0]
        device = x0_static.device

        if noise is None:
            noise = torch.randn(B, self.noise_dim, device=device)

        T_norm  = ((T_K - cfg.T_REF_K) / cfg.T_REF_K).unsqueeze(1)  # (B,1)
        context = torch.cat([x0_static, T_norm], dim=1)               # (B, context_dim)

        # Phase 3: additive prefix conditioning (zero at init → backward-compat)
        if z_prefix_last is not None and getattr(cfg, "GENERATOR_USE_PREFIX_CONTEXT", True):
            context = context + self.prefix_proj(z_prefix_last)
        if log_prefix_time is not None and getattr(cfg, "GENERATOR_USE_PREFIX_CONTEXT", True):
            t_feat = log_prefix_time if log_prefix_time.dim() == 2 else log_prefix_time.unsqueeze(1)
            context = context + self.time_proj(t_feat)

        inp = torch.cat([noise, context], dim=1)                       # (B, noise + context_dim)

        out      = self.net(inp)                        # (B, 6)
        z0_raw   = out[:, :self.latent_dim]             # (B, 5)
        alpha_raw = out[:, self.latent_dim]              # (B,)

        if z_prefix_last is not None and getattr(cfg, "GENERATOR_USE_PREFIX_CONTEXT", True):
            # Logit-space per-state perturbation (eliminates hard clipping).
            # z0 = sigmoid(logit(z_prefix_last) + scale * tanh(delta))
            _eps = 1e-6
            pert_scale = torch.exp(self.z0_pert_log)              # (latent_dim,)
            logit_pfx  = torch.log(
                (z_prefix_last.clamp(_eps, 1 - _eps)) /
                (1.0 - z_prefix_last.clamp(_eps, 1 - _eps))
            )   # (B, latent_dim)
            logit_pert = logit_pfx + pert_scale * torch.tanh(z0_raw)
            z0 = torch.sigmoid(logit_pert)                         # always in (0,1)
            # Track how far perturbation moves logit (informational)
            with torch.no_grad():
                self._clip_count += torch.zeros(1, device=z0.device)  # no clipping needed
                self._total_count += float(z0.numel())
        else:
            z0 = torch.sigmoid(z0_raw)                           # (B, 5) ∈ [0,1]

        alpha = F.softplus(alpha_raw)                            # (B,)   > 0
        # Clamp alpha to a physically plausible range to prevent runaway diversity
        _alpha_min = float(getattr(cfg, "GENERATOR_ALPHA_MIN", 0.5))
        _alpha_max = float(getattr(cfg, "GENERATOR_ALPHA_MAX", 2.0))
        alpha = alpha.clamp(_alpha_min, _alpha_max)
        if getattr(cfg, "GENERATOR_FIXED_ALPHA", False):
            alpha = torch.ones_like(alpha)   # fix alpha=1 when not identifiable

        return z0, alpha

    def sample_trajectory(
        self,
        x0_static:  torch.Tensor,
        T_K:        torch.Tensor,
        times_h:    torch.Tensor,
        ode_module,                           # PhysicsODE instance
        n_samples:  int = 1,
        noise:      torch.Tensor = None,
        z_prefix_last:   torch.Tensor = None,  # (B, 5)  optional prefix context
        log_prefix_time: torch.Tensor = None,  # (B, 1)  optional log-time context
    ) -> dict:
        """
        Generate a full latent trajectory (and the corresponding device alpha).

        Args:
            x0_static        : (B, 6)
            T_K              : (B,)
            times_h          : (B, T)  observation time grid
            ode_module       : PhysicsODE
            n_samples        : number of trajectory samples per device
            noise            : (B*n_samples, noise_dim) optional fixed noise
            z_prefix_last    : (B, 5)   encoder state at end of prefix (optional)
            log_prefix_time  : (B, 1)   log(1 + t_prefix_end)          (optional)

        Returns:
            dict with keys:
              'z_traj'     : (B*n_samples, T, 5)
              'z0'         : (B*n_samples, 5)
              'alpha'      : (B*n_samples,)
        """
        B = x0_static.shape[0]
        Bn = B * n_samples

        x0_rep  = x0_static.repeat_interleave(n_samples, dim=0)
        T_K_rep = T_K.repeat_interleave(n_samples)
        t_rep   = times_h.repeat_interleave(n_samples, dim=0)

        z_pre_rep = None
        t_pre_rep = None
        if z_prefix_last is not None:
            z_pre_rep = z_prefix_last.repeat_interleave(n_samples, dim=0)
        if log_prefix_time is not None:
            t_pre_rep = log_prefix_time.repeat_interleave(n_samples, dim=0)

        z0, alpha = self.forward(
            x0_rep, T_K_rep, noise,
            z_prefix_last=z_pre_rep,
            log_prefix_time=t_pre_rep,
        )

        with torch.no_grad():
            z_traj = ode_module.integrate_trajectory(z0, T_K_rep, t_rep, alpha)

        return {
            "z_traj": z_traj,
            "z0":     z0,
            "alpha":  alpha,
        }
