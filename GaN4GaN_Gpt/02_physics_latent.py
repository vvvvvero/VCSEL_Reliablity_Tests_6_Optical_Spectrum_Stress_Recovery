"""
02_physics_latent.py
====================
Physics-informed Latent ODE system for GaN HEMT thermal-storage degradation.

5 Effective Latent States
--------------------------
  zG  –  Gate / interface / barrier effective charged-defect occupancy   [0,1]
  zB  –  Buffer / access-region effective charged-defect occupancy        [0,1]
  zM  –  Channel transport degradation                                    [0,1]  (monotone)
  zL  –  Leakage-path degradation                                         [0,1]  (monotone)
  zC  –  Cumulative irreversible structural damage                        [0,1]  (monotone)

ODE (simplified V1 for stable first-pass training)
---------------------------------------------------
  dzG/dt = kGc·frev(T)·(1-zG)  -  kGe·frev(T)·zG
  dzB/dt = kBc·frev(T)·(1-zB)  -  kBe·frev(T)·zB
  dzM/dt = kM ·firrev(T)·(wMG·zG + wMB·zB + zC)
  dzL/dt = kL ·firrev(T)·(1-zL)·(aLG·zG + aLB·zB + zC)
    dzC/dt = kC ·device_alpha·firrev(T)·(1-zC)^2

Arrhenius temperature factors (relative to T_ref):
  frev  (T) = exp(-Ea_rev  / kB · (1/T - 1/T_ref))
  firrev(T) = exp(-Ea_irrev / kB · (1/T - 1/T_ref))

All rate constants are parameterised via softplus to guarantee positivity.
Monotonicity of zM, zL, zC is enforced by the ODE structure (non-negative RHS).

Differentiable RK4 integrator operates on true physical time t [hours].

Usage
-----
    from 02_physics_latent import PhysicsODE
    ode = PhysicsODE().to(device)
    z_next = ode.integrate(z_curr, T_K_batch, dt_h)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg

ARRHENIUS_EXP_CLAMP = 15.0
RHS_CLAMP = 50.0

# ---------------------------------------------------------------------------
# Helper: softplus parameter factory (guarantees positivity)
# ---------------------------------------------------------------------------

def _sp(raw: torch.Tensor) -> torch.Tensor:
    """Softplus: maps ℝ → (0, ∞)."""
    return F.softplus(raw)


# ---------------------------------------------------------------------------
# Physics ODE Module
# ---------------------------------------------------------------------------

class PhysicsODE(nn.Module):
    """
    Learnable physics ODE for the 5 latent states.

    Learnable parameters
    --------------------
    log_Ea_rev, log_Ea_irrev   : shared activation energies [eV]
    kGc_raw, kGe_raw           : gate capture / emission rate constants
    kBc_raw, kBe_raw           : buffer capture / emission rate constants
    kM_raw                     : channel transport damage rate
    wMG_raw, wMB_raw           : mixing weights (zG, zB → zM)
    kL_raw                     : leakage path growth rate
    aLG_raw, aLB_raw           : mixing weights (zG, zB → zL)
    kC_raw                     : cumulative damage rate

    All _raw parameters are unconstrained; positivity enforced via softplus.
    Ea values are stored as log(Ea) to ensure > 0.
    """

    def __init__(self):
        super().__init__()

        # Shared activation energies (eV)
        # Reasonable initialisation: Ea_rev ≈ 0.3 eV, Ea_irrev ≈ 0.7 eV
        self.log_Ea_rev   = nn.Parameter(torch.tensor(math.log(0.30)))
        self.log_Ea_irrev = nn.Parameter(torch.tensor(math.log(0.70)))

        # Gate / interface trap kinetics
        self.kGc_raw = nn.Parameter(torch.tensor(-2.0))   # capture rate
        self.kGe_raw = nn.Parameter(torch.tensor(-3.0))   # emission rate

        # Buffer / access-region trap kinetics
        self.kBc_raw = nn.Parameter(torch.tensor(-2.5))
        self.kBe_raw = nn.Parameter(torch.tensor(-3.5))

        # Channel transport degradation
        self.kM_raw  = nn.Parameter(torch.tensor(-4.0))
        self.wMG_raw = nn.Parameter(torch.tensor(0.0))    # mixing weight
        self.wMB_raw = nn.Parameter(torch.tensor(0.0))

        # Leakage path degradation
        self.kL_raw  = nn.Parameter(torch.tensor(-4.5))
        self.aLG_raw = nn.Parameter(torch.tensor(0.0))
        self.aLB_raw = nn.Parameter(torch.tensor(0.0))

        # Cumulative irreversible damage
        self.kC_raw  = nn.Parameter(torch.tensor(-5.0))

    # ------------------------------------------------------------------
    # Property accessors (positive values)
    # ------------------------------------------------------------------

    @property
    def Ea_rev(self) -> torch.Tensor:
        return torch.exp(self.log_Ea_rev)

    @property
    def Ea_irrev(self) -> torch.Tensor:
        return torch.exp(self.log_Ea_irrev)

    @property
    def kGc(self) -> torch.Tensor:
        return _sp(self.kGc_raw)

    @property
    def kGe(self) -> torch.Tensor:
        return _sp(self.kGe_raw)

    @property
    def kBc(self) -> torch.Tensor:
        return _sp(self.kBc_raw)

    @property
    def kBe(self) -> torch.Tensor:
        return _sp(self.kBe_raw)

    @property
    def kM(self) -> torch.Tensor:
        return _sp(self.kM_raw)

    @property
    def wMG(self) -> torch.Tensor:
        return _sp(self.wMG_raw)

    @property
    def wMB(self) -> torch.Tensor:
        return _sp(self.wMB_raw)

    @property
    def kL(self) -> torch.Tensor:
        return _sp(self.kL_raw)

    @property
    def aLG(self) -> torch.Tensor:
        return _sp(self.aLG_raw)

    @property
    def aLB(self) -> torch.Tensor:
        return _sp(self.aLB_raw)

    @property
    def kC(self) -> torch.Tensor:
        return _sp(self.kC_raw)

    # ------------------------------------------------------------------
    # Arrhenius factors
    # ------------------------------------------------------------------

    def _arrhenius(self, Ea: torch.Tensor, T_K: torch.Tensor) -> torch.Tensor:
        """
        f(T) = exp(-Ea/kB · (1/T - 1/T_ref))

        Args:
            Ea  : scalar [eV]
            T_K : (B,) or (B,1) temperature in Kelvin

        Returns:
            f   : same shape as T_K
        """
        kb = cfg.KB_EV
        T_ref = cfg.T_REF_K
        inv_T_diff = 1.0 / T_K - 1.0 / T_ref
        exponent = -Ea / kb * inv_T_diff
        exponent = torch.clamp(exponent, -ARRHENIUS_EXP_CLAMP, ARRHENIUS_EXP_CLAMP)
        return torch.exp(exponent)

    def _sanitize_rhs(self, dzdt: torch.Tensor) -> torch.Tensor:
        """Keep RK4 derivatives finite and within a numerically safe range."""
        dzdt = torch.nan_to_num(dzdt, nan=0.0, posinf=RHS_CLAMP, neginf=-RHS_CLAMP)
        return torch.clamp(dzdt, -RHS_CLAMP, RHS_CLAMP)

    def _sanitize_state(self, z: torch.Tensor) -> torch.Tensor:
        """Project states back to a finite bounded latent box after each substep."""
        z = torch.nan_to_num(z, nan=0.5, posinf=1.0, neginf=0.0)
        return torch.clamp(z, 0.0, 1.0)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def rhs(self,
            z: torch.Tensor,
            T_K: torch.Tensor,
            device_alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute dz/dt for the 5-state system.

        Args:
            z            : (B, 5)  current latent state [zG, zB, zM, zL, zC]
            T_K          : (B,)    temperature [K]
            device_alpha : (B,)    per-device rate multiplier for zC (≥ 0)

        Returns:
            dzdt         : (B, 5)
        """
        T_K = T_K.view(-1, 1)                       # (B,1)
        device_alpha = device_alpha.view(-1, 1)

        frev   = self._arrhenius(self.Ea_rev,   T_K)  # (B,1)
        firrev = self._arrhenius(self.Ea_irrev, T_K)  # (B,1)

        zG = z[:, 0:1]
        zB = z[:, 1:2]
        zM = z[:, 2:3]
        zL = z[:, 3:4]
        zC = z[:, 4:5]

        # dzG/dt  (capture-emission; can decrease on detrapping)
        dzG = self.kGc * frev * (1.0 - zG) - self.kGe * frev * zG

        # dzB/dt
        dzB = self.kBc * frev * (1.0 - zB) - self.kBe * frev * zB

        # dzM/dt  (monotone increasing: driven by trap occupancy + cumulative)
        driving_M = self.wMG * zG + self.wMB * zB + zC
        dzM = self.kM * firrev * driving_M * (1.0 - zM)

        # dzL/dt  (monotone increasing: nucleation-saturation form)
        driving_L = self.aLG * zG + self.aLB * zB + zC
        dzL = self.kL * firrev * (1.0 - zL) * driving_L

        # dzC/dt  (monotone increasing: slower approach to saturation)
        dzC = self.kC * device_alpha * firrev * (1.0 - zC) ** 2

        dzdt = torch.cat([dzG, dzB, dzM, dzL, dzC], dim=1)  # (B,5)
        return dzdt

    # ------------------------------------------------------------------
    # Differentiable RK4 integrator
    # ------------------------------------------------------------------

    def integrate(self,
                  z0: torch.Tensor,
                  T_K: torch.Tensor,
                  dt_h: torch.Tensor,
                  device_alpha: torch.Tensor,
                  n_substeps: int = None) -> torch.Tensor:
        """
        Integrate the ODE from t to t + dt_h using fixed-step RK4.

        Args:
            z0           : (B, 5)  initial state
            T_K          : (B,)    temperature [K]
            dt_h         : (B,)    integration interval [hours]
            device_alpha : (B,)    per-device rate multiplier
            n_substeps   : number of RK4 substeps; auto-computed if None

        Returns:
            z_end        : (B, 5)
        """
        if n_substeps is None:
            # Heuristic: 10 substeps per log-decade of dt
            dt_min = dt_h.min().item()
            dt_max = dt_h.max().item()
            if dt_max < 1e-6:
                return z0.clone()
            log_span = max(1.0, math.log10(max(dt_max, 1e-6) + 1))
            n_substeps = max(5, int(cfg.ODE_SUBSTEPS_PER_LOG_DECADE * log_span))

        z = z0
        # True physical-time RK4 on the actual dt_h interval.
        sub_dt = dt_h / n_substeps   # (B,)

        for _ in range(n_substeps):
            h = sub_dt.unsqueeze(1)
            k1 = self._sanitize_rhs(self.rhs(z, T_K, device_alpha))
            z2 = self._sanitize_state(z + 0.5 * h * k1)
            k2 = self._sanitize_rhs(self.rhs(z2, T_K, device_alpha))
            z3 = self._sanitize_state(z + 0.5 * h * k2)
            k3 = self._sanitize_rhs(self.rhs(z3, T_K, device_alpha))
            z4 = self._sanitize_state(z + h * k3)
            k4 = self._sanitize_rhs(self.rhs(z4, T_K, device_alpha))
            dz = (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            z = self._sanitize_state(z + dz)

        return z

    # ------------------------------------------------------------------
    # Integrate over a full trajectory
    # ------------------------------------------------------------------

    def integrate_trajectory(self,
                              z0: torch.Tensor,
                              T_K: torch.Tensor,
                              times_h: torch.Tensor,
                              device_alpha: torch.Tensor) -> torch.Tensor:
        """
        Produce the full latent trajectory from z0 over a sequence of
        observation times.

        Args:
            z0         : (B, 5)   initial latent state
            T_K        : (B,)     temperature [K]
            times_h    : (B, T)   observation times [h]; first column = t0
            device_alpha: (B,)    per-device rate multiplier

        Returns:
            z_traj     : (B, T, 5)   latent states at each time point
        """
        B, T = times_h.shape
        z_traj = [z0.unsqueeze(1)]         # (B,1,5)
        z_curr = z0

        for step in range(1, T):
            dt = times_h[:, step] - times_h[:, step - 1]  # (B,)
            dt = dt.clamp(min=cfg.ODE_MIN_DT_H)
            z_next = self.integrate(z_curr, T_K, dt, device_alpha)
            z_traj.append(z_next.unsqueeze(1))
            z_curr = z_next

        return torch.cat(z_traj, dim=1)   # (B, T, 5)

    # ------------------------------------------------------------------
    # Utility: ODE residual  (used in loss computation)
    # ------------------------------------------------------------------

    def ode_residual(self,
                     z_enc: torch.Tensor,
                     T_K: torch.Tensor,
                     times_h: torch.Tensor,
                     device_alpha: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """
                Compute a stable one-step residual between consecutive encoder latent
                states and the ODE-implied latent transition.

                This loss is intentionally teacher-forced:
                - z(t_i) and z(t_{i+1}) are detached targets from the encoder.
                - The residual only trains the ODE parameters and alpha network.
                - A bounded one-step update is used instead of backpropagating through
                    a long-horizon solver inside the residual term.

                Longer-horizon behavior is still handled elsewhere through rollout.

        Args:
            z_enc      : (B, T, 5)  encoder latent outputs (after sigmoid)
            T_K        : (B,)
            times_h    : (B, T)
            device_alpha: (B,)
            mask       : (B, T)  bool, True where observation is valid

        Returns:
            residual   : scalar mean squared error
        """
        B, T, _ = z_enc.shape
        total_loss = torch.zeros(1, device=z_enc.device)
        count = 0

        for step in range(1, T):
            # Only compute for pairs where both t-1 and t are valid
            valid = mask[:, step - 1] & mask[:, step]  # (B,)
            if not valid.any():
                continue

            z_prev  = z_enc[valid, step - 1, :].detach()    # (Bv, 5)
            z_curr  = z_enc[valid, step, :].detach()        # (Bv, 5)
            T_K_v   = T_K[valid]
            dt_v    = (times_h[valid, step] - times_h[valid, step - 1]).clamp(min=cfg.ODE_MIN_DT_H)
            alpha_v = device_alpha[valid]

            z_pred = self.integrate(z_prev, T_K_v, dt_v, alpha_v)
            total_loss = total_loss + F.mse_loss(z_pred, z_curr)
            count += 1

        if count == 0:
            return torch.zeros(1, device=z_enc.device)
        return total_loss / count


# ---------------------------------------------------------------------------
# Device-specific rate multiplier module
# ---------------------------------------------------------------------------

class DeviceAlphaNet(nn.Module):
    """
    Estimates per-device cumulative-damage rate multiplier α_i ≥ 0
    from the initial electrical parameters x0 and temperature.

    Architecture: small MLP  [x0_static (6) + T_norm (1)] → α (1)
    Output is bounded to [1/ALPHA_MAX, ALPHA_MAX] to avoid extreme
    zC growth rates from outlier devices.
    """

    def __init__(self, input_dim: int = cfg.FEATURE_DIM + 1,
                 hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize around alpha=1.0 (tanh(0)=0 -> exp(0)=1).
        with torch.no_grad():
            self.net[-1].bias.fill_(0.0)

    def forward(self,
                x0_static: torch.Tensor,
                T_K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0_static : (B, 6)  initial parameter values (can be normalised)
            T_K       : (B,)    temperature in Kelvin

        Returns:
            alpha     : (B,)    per-device rate multiplier > 0
        """
        # Compress wide-ranging absolute parameters before feeding the MLP.
        # This avoids tanh saturation on raw magnitudes and makes α depend on
        # relative differences rather than feature scale.
        x0_feat = torch.log1p(torch.clamp(x0_static, min=0.0))
        T_norm = ((T_K - cfg.T_REF_K) / cfg.T_REF_K).unsqueeze(1)  # (B,1)
        inp = torch.cat([x0_feat, T_norm], dim=1)                    # (B,7)
        alpha_raw = self.net(inp).squeeze(1)
        scale = math.log(float(cfg.ALPHA_MAX))
        alpha = torch.exp(scale * torch.tanh(alpha_raw))
        return alpha
