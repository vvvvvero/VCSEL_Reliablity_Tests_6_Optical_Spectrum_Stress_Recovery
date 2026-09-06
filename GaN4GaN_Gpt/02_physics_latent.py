"""
02_physics_latent.py
====================
Physics-informed Latent ODE system for GaN HEMT thermal-storage degradation.

5 Effective Latent States
--------------------------
  zG  –  Gate / interface effective trap occupancy fraction              [0,1]
  zB  –  Buffer / access-region effective trap occupancy fraction        [0,1]
  zM  –  Channel transport degradation                                    [0,1]  (monotone)
  zL  –  Leakage-path degradation                                         [0,1]  (monotone)
  zC  –  Cumulative irreversible structural damage                        [0,1]  (monotone)

ODE (V2: SRH-form trap kinetics + explicit boundary-continuity constraints)
-----------------------------------------------------------------------------
zG and zB follow simplified Shockley-Read-Hall (SRH) trap-occupancy kinetics.
Full SRH statistics for a single trap level exchanging with both carrier
bands gives (using f_t = trap occupancy fraction):

    df_t/dt = c_n·n·(1-f_t) - e_n·f_t - c_p·p·f_t + e_p·(1-f_t)

Detailed balance ties each capture/emission pair to the SAME activation
energy (they differ only by a temperature-independent trap-level prefactor,
not by a separate thermal barrier), so emission and capture for one carrier
type share one Arrhenius factor:

    dzG/dt = kGc·frev(T)·(1-zG) - kGe·frev(T)·zG                 [majority-carrier term, as V1]
             + gammaG·[ kGc·frev(T)·(1-zG) - kGe·frev(T)·zG ]      [minority-carrier correction]
           = (1+gammaG)·[ kGc·frev(T)·(1-zG) - kGe·frev(T)·zG ]

  i.e. gammaG >= 0 is a single extra scalar per fast state that scales the
  net SRH exchange rate to account for the (smaller, same-sign) opposite-
  carrier contribution, WITHOUT introducing 4 independent rate constants
  per state (which the dataset — 203 devices, low single digits per
  device-type x temperature cell — cannot identify; see debug notes on
  alpha_identifiability). This keeps the same functional form as V1's
  (1-zG)/zG relaxation (so it stays a well-posed, bounded-drift ODE) while
  giving kGe/kGc a falsifiable SRH interpretation: at fixed T, the
  equilibrium occupancy z_G,eq = kGc/(kGc+kGe) is the SRH-implied trap
  Fermi-level occupancy, and gammaG is now a REPORTED, testable quantity
  (see minority_carrier_weight_gG/gB) rather than folded silently into kGc.

zM, zL, zC keep their V1 saturating-drift structure (driven by trap
occupancy + cumulative damage, monotone via a (1-z) saturation factor),
which is structurally identical to a diffusion-limited drift term: the
"driving force" (zG, zB, zC mixture) plays the role of a concentration
gradient forcing the state toward its saturation boundary, and the (1-z)
factor is the SRH-style saturation cutoff (occupancy cannot exceed 1).

  dzM/dt = kM ·firrev(T)·(wMG·zG + wMB·zB + zC)·(1-zM)
  dzL/dt = kL ·firrev(T)·(1-zL)·(aLG·zG + aLB·zB + zC)
  dzC/dt = kC ·device_alpha·firrev(T)·(1-zC)^2

Arrhenius temperature factors (relative to T_ref):
  frev  (T) = exp(-Ea_rev  / kB · (1/T - 1/T_ref))
  firrev(T) = exp(-Ea_irrev / kB · (1/T - 1/T_ref))

Boundary-continuity conditions
-------------------------------
Two continuity requirements are enforced, one structurally (built into the
RHS so it holds at every integration step, not just approximately) and one
as a soft training penalty (checked, not assumed):

  1. Saturation-boundary flux continuity (structural): every monotone
     state's RHS carries an explicit (1-z) or (1-z)^2 saturation factor, so
     dz/dt -> 0 smoothly (no kink, first-derivative-continuous in z) as
     z -> 1. This was already true in V1; V2 keeps it and extends it to the
     new SRH terms (the minority-carrier correction reuses the same
     (1-zG)/(1-zB) factors, so it cannot introduce a discontinuity).

  2. Trap-to-damage handoff continuity (soft, checked via
     handoff_continuity_residual): the driving force feeding zM/zL
     (wMG*zG + wMB*zB + zC and aLG*zG + aLB*zB + zC) must be continuous
     across the observed trajectory — i.e. no instantaneous jump in the
     effective driving term at any observed time step. This is checked
     numerically (max jump in driving force between consecutive observed
     points, relative to the local RK4 step) and penalised if it exceeds a
     tolerance, catching integrator/parameter pathologies (e.g. a badly
     scaled gammaG causing a near-discontinuous jump) that (1) alone does
     not rule out.

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
SOFT_CLAMP_BETA = 100.0
# Half-width of the "do nothing" band around [0,1] for _soft_clamp01.
#
# The ODE cannot mathematically leave [0,1]: every monotone state carries an
# explicit (1-z) (or (1-z)^2) factor that drives dz/dt -> 0 at the boundary,
# and the fast modes relax to k_c/(k_c+k_e) in (0,1). Measured with bare RK4
# and NO sanitisation, a 1000 h step lands at zM = 0.99981 with 8e-7 error at
# ANY substep count — the raw dynamics are exact and well behaved.
#
# The old always-on soft clamp therefore was not protecting against real
# excursions, it was distorting legitimate states: at beta=100 it pulls
# z=0.999 down to 0.9926 (-0.0064), ALWAYS downward, on every one of the 6+
# calls per RK4 substep. That accumulated into ~0.03 trajectory error and,
# because the sign is systematic, into exactly the downward Stage-3
# prediction bias (+0.11..+0.14 at 2000 h) this investigation started from.
# It also destroyed RK4's convergence order (measured 0.1-0.6 instead of 4).
#
# With this margin the clamp is the exact identity throughout the physically
# reachable range and only engages on genuine numerical excursions (NaN/inf
# or overshoot beyond the margin), where its smooth non-zero gradient still
# prevents the NaN-gradient failure it was originally introduced to fix.
#
# 0.20 rather than 0.05: zM/zL legitimately reach 0.999+, and at margin=0.05
# the clamp still biased those by -6.6e-5 per call. Applied ~6x per RK4
# substep that accumulated LINEARLY in the substep count (measured: zM error
# 5.4e-4 at n=30 growing to 5.5e-3 at n=480 — more substeps made the answer
# WORSE, a negative convergence order). At margin=0.20 the deviation at
# z=0.9998 is 2.6e-8 (2500x smaller) and exactly 0 at z=1.0, while the
# out-of-range gradient (9.4e-14) and the saturation behaviour are unchanged.
SOFT_CLAMP_MARGIN = 0.20

# --- DeviceAlphaNet conditioning (see the DeviceAlphaNet docstring) --------
# Divisor on the tanh argument. 1.0 (the original) saturated 100 % of devices
# and pinned alpha at its lower bound with zero gradient. 3.0 widens the
# responsive input band ~3x; the reachable alpha range is unchanged.
ALPHA_TANH_SLOPE = 3.0
# Std of the final-layer weight init. The PyTorch default (U(-1/sqrt(16),
# 1/sqrt(16)) ~ +-0.25) over 16 hidden units can put |alpha_raw| in the
# saturated region before training starts; 0.01 begins near alpha=1.0.
ALPHA_HEAD_INIT_STD = 0.01

# ---------------------------------------------------------------------------
# Helper: softplus parameter factory (guarantees positivity)
# ---------------------------------------------------------------------------

def _sp(raw: torch.Tensor) -> torch.Tensor:
    """Softplus: maps ℝ → (0, ∞)."""
    return F.softplus(raw)


def _soft_clamp01(z: torch.Tensor, beta: float = SOFT_CLAMP_BETA,
                   margin: float = SOFT_CLAMP_MARGIN) -> torch.Tensor:
    """Smooth saturation onto [-margin, 1+margin], identity strictly inside.

    Implemented as a softplus-based smooth clamp onto the WIDENED interval
    [-margin, 1+margin] rather than onto [0,1] directly. Softplus is
    asymptotically exact, so a state anywhere in the physically reachable
    range (which the ODE structure confines to [0,1] — see SOFT_CLAMP_MARGIN)
    passes through unchanged to float precision, while genuine numerical
    excursions beyond the margin are still smoothly saturated.

    Unlike torch.clamp, whose gradient is EXACTLY zero outside its range (the
    failure mode that caused the RK4 backward pass to accumulate NaN once
    zM/zL saturate near 1 — see 2026-08-16 investigation notes), this keeps a
    strictly positive sigmoid(beta*x)-shaped gradient everywhere finite.

    Callers still get hard [0,1] containment where they need it: the raw
    dynamics never leave [0,1], and _sanitize_state() additionally replaces
    non-finite values.
    """
    lo, hi = -margin, 1.0 + margin
    # smooth max(z, lo) then smooth min(., hi), both on the widened interval
    soft_lo = lo + F.softplus(z - lo, beta=beta)
    return hi - F.softplus(hi - soft_lo, beta=beta)


# ---------------------------------------------------------------------------
# Physics ODE Module
# ---------------------------------------------------------------------------

class PhysicsODE(nn.Module):
    """
    Learnable physics ODE for the 5 latent states (SRH-form trap kinetics).

    Learnable parameters
    --------------------
    log_Ea_rev, log_Ea_irrev   : shared activation energies [eV]
    kGc_raw, kGe_raw           : gate SRH majority-carrier capture / emission rates
    gammaG_raw                 : gate SRH minority-carrier correction weight (>=0)
    kBc_raw, kBe_raw           : buffer SRH majority-carrier capture / emission rates
    gammaB_raw                 : buffer SRH minority-carrier correction weight (>=0)
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

        # Gate / interface trap kinetics (SRH majority-carrier term)
        self.kGc_raw = nn.Parameter(torch.tensor(-2.0))   # capture rate
        self.kGe_raw = nn.Parameter(torch.tensor(-3.0))   # emission rate
        # SRH minority-carrier correction: init small (gamma≈0.05) so V2
        # starts close to V1's pure two-state relaxation and only grows the
        # correction if the data support it.
        self.gammaG_raw = nn.Parameter(torch.tensor(-3.0))

        # Buffer / access-region trap kinetics (SRH majority-carrier term)
        self.kBc_raw = nn.Parameter(torch.tensor(-2.5))
        self.kBe_raw = nn.Parameter(torch.tensor(-3.5))
        self.gammaB_raw = nn.Parameter(torch.tensor(-3.0))

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
    def gammaG(self) -> torch.Tensor:
        """SRH minority-carrier correction weight for zG, >= 0."""
        return _sp(self.gammaG_raw)

    @property
    def kBc(self) -> torch.Tensor:
        return _sp(self.kBc_raw)

    @property
    def kBe(self) -> torch.Tensor:
        return _sp(self.kBe_raw)

    @property
    def gammaB(self) -> torch.Tensor:
        """SRH minority-carrier correction weight for zB, >= 0."""
        return _sp(self.gammaB_raw)

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

    @property
    def zG_equilibrium(self) -> torch.Tensor:
        """SRH-implied equilibrium (steady-state) occupancy of zG at T_ref:
        z_G,eq = kGc / (kGc + kGe). Exposed for physical-plausibility checks
        (should stay in (0,1); a value near 0 or 1 signals the trap level is
        essentially always empty/full, which is a testable SRH prediction)."""
        return self.kGc / (self.kGc + self.kGe + 1e-12)

    @property
    def zB_equilibrium(self) -> torch.Tensor:
        """SRH-implied equilibrium occupancy of zB at T_ref (see zG_equilibrium)."""
        return self.kBc / (self.kBc + self.kBe + 1e-12)

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
        """Project states back to a finite bounded latent box after each substep.

        Uses a smooth soft-clamp (see _soft_clamp01) rather than a hard
        torch.clamp: RK4 is called after every substep and every intermediate
        RK4 stage (k1..k4), so this function runs dozens of times per
        integration interval. A hard clamp's exactly-zero gradient outside
        [0,1] compounds across that many calls once a monotone state (zM/zL)
        saturates near 1, and was the root cause of a real NaN-gradient
        failure during Stage 3 training (391 sanitized-gradient events in a
        full retrain — see 2026-08-16/17 investigation). The soft-clamp is
        the exact identity deep in the interior, so this changes nothing
        about the model's normal-operating-range numerics; it only smooths
        the corner right at 0/1.
        """
        z = torch.nan_to_num(z, nan=0.5, posinf=1.0, neginf=0.0)
        return _soft_clamp01(z)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def driving_forces(self, z: torch.Tensor) -> torch.Tensor:
        """Return (driving_M, driving_L), the trap-to-damage handoff terms
        that feed zM and zL. Exposed separately (not just inlined in rhs())
        so the boundary-continuity check can evaluate them directly on
        observed/encoder trajectories without re-deriving the RHS.

        Args:
            z : (B, 5)  [zG, zB, zM, zL, zC]
        Returns:
            (driving_M, driving_L) : each (B, 1)
        """
        zG = z[:, 0:1]
        zB = z[:, 1:2]
        zC = z[:, 4:5]
        driving_M = self.wMG * zG + self.wMB * zB + zC
        driving_L = self.aLG * zG + self.aLB * zB + zC
        return driving_M, driving_L

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

        # dzG/dt  — SRH occupancy kinetics: majority-carrier capture/emission
        # scaled by (1 + gammaG) to fold in the minority-carrier contribution
        # (same functional form, same activation energy — see module docstring).
        #
        # Numerical note: the deep RK4-through-multi-interval rollout used by
        # Stage 3 (integrate_trajectory over up to ~60 chained substeps per
        # interval, 11 intervals) already sits within ~1e10x of float32
        # overflow at initialization for the plain V1 majority-carrier term
        # once zM/zL saturate near 1 (a pre-existing _sanitize_state clamp-
        # gradient fragility, tracked separately — not fixed here). The
        # primary term (coefficient 1, below) reproduces V1 exactly and
        # carries the same borderline-but-finite gradient V1 always had.
        # gammaG is a SMALL scalar correction weight — training it does not
        # require differentiating through srh_G's own deep RK4 history a
        # second time (that would double-count an already near-overflowing
        # gradient path and reliably tip it into NaN, confirmed empirically).
        # detach() on the minority branch's srh_G reference removes that
        # second gradient path while leaving gammaG's own gradient (via the
        # multiplication) and the primary term's gradient both intact.
        srh_G = self.kGc * frev * (1.0 - zG) - self.kGe * frev * zG
        dzG = srh_G + self.gammaG * srh_G.detach()

        # dzB/dt  — same SRH structure for the buffer/access-region trap.
        srh_B = self.kBc * frev * (1.0 - zB) - self.kBe * frev * zB
        dzB = srh_B + self.gammaB * srh_B.detach()

        # dzM/dt  (monotone increasing: driven by trap occupancy + cumulative)
        driving_M, driving_L = self.driving_forces(z)
        dzM = self.kM * firrev * driving_M * (1.0 - zM)

        # dzL/dt  (monotone increasing: nucleation-saturation form)
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

        if getattr(cfg, "ODE_USE_IMEX", True):
            return self._integrate_imex(z0, T_K, dt_h, device_alpha, n_substeps)

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
    # IMEX (exponential) integrator — see module docstring "Stiffness"
    # ------------------------------------------------------------------

    def _fast_mode_coeffs(self, T_K: torch.Tensor):
        """Return (a_G, b_G, a_B, b_B), the linear capture/emission rates of
        the fast trap-occupancy modes at temperature T_K, each shaped (B,1).

        dzG/dt = a_G*(1 - zG) - b_G*zG   (and analogously for zB)

        The (1 + gamma) SRH minority-carrier correction is a plain scalar
        multiplier on the whole expression (see rhs()), so it folds into both
        rates identically and does not change the linear structure.
        """
        T_K = T_K.view(-1, 1)
        frev = self._arrhenius(self.Ea_rev, T_K)          # (B,1)
        gG = 1.0 + self.gammaG
        gB = 1.0 + self.gammaB
        a_G = self.kGc * frev * gG
        b_G = self.kGe * frev * gG
        a_B = self.kBc * frev * gB
        b_B = self.kBe * frev * gB
        return a_G, b_G, a_B, b_B

    def _exact_fast_update(self, z_fast, a, b, dt):
        """Closed-form solution of dz/dt = a*(1-z) - b*z over an interval dt.

            z(t+dt) = z_eq + (z(t) - z_eq) * exp(-(a+b)*dt),  z_eq = a/(a+b)

        Exact for ANY dt, so the fast modes impose no stability limit on the
        step size at all.
        """
        rate = (a + b).clamp(min=1e-12)
        z_eq = a / rate
        decay = torch.exp(-(rate * dt).clamp(max=ARRHENIUS_EXP_CLAMP))
        return z_eq + (z_fast - z_eq) * decay

    def _integrate_imex(self,
                        z0: torch.Tensor,
                        T_K: torch.Tensor,
                        dt_h: torch.Tensor,
                        device_alpha: torch.Tensor,
                        n_substeps: int) -> torch.Tensor:
        """Split (IMEX / exponential-integrator) scheme for this stiff system.

        The 5-state system mixes two very different time scales (measured on
        the trained model at 325 C):
          - FAST, linear   : zG, zB   relaxation tau ~ 5-8 h
          - SLOW, nonlinear: zM, zL, zC  evolving over 1e3-1e4 h

        The observation grid steps out to dt = 1000 h, i.e. up to ~200x the
        fast relaxation time. Explicit RK4 is only stable for dt <~ tau, so
        the plain RK4 path silently produced large errors on the long steps
        (measured: 0.11 absolute error in zG on the 1000->2000 h step, landing
        on 0.699 instead of the correct 0.587). That error grows with horizon
        and matched, in both sign and magnitude, the systematic
        under-prediction bias seen in Stage 3 forecasts (+0.11..+0.14 at
        2000 h). Reformulating on a log-time axis does NOT help — it rescales
        the step distribution but leaves the stiffness ratio untouched, and
        the (1+t)*ln10 Jacobian actually makes the worst step worse
        (0.41 vs 0.11, measured).

        Fix: integrate the fast modes with their exact exponential solution
        (unconditionally stable for any dt) and keep RK4 only for the slow
        nonlinear modes, whose time constants are far longer than any step.
        The fast states are held at their sub-step-endpoint values while the
        slow RK4 stages are evaluated, which is the standard Lie-Trotter
        split; with tau_fast << dt the fast modes sit at equilibrium during
        the step, so the splitting error is negligible exactly where the old
        scheme was worst.
        """
        z = z0
        sub_dt = (dt_h / n_substeps).unsqueeze(1)   # (B,1)
        half_dt = 0.5 * sub_dt
        a_G, b_G, a_B, b_B = self._fast_mode_coeffs(T_K)

        def slow_rhs(z_in):
            full = self._sanitize_rhs(self.rhs(z_in, T_K, device_alpha))
            # zero out the fast components: they are handled exactly, below
            return torch.cat([torch.zeros_like(full[:, :2]), full[:, 2:]], dim=1)

        def advance_fast(z_in, dt):
            zG = self._exact_fast_update(z_in[:, 0:1], a_G, b_G, dt)
            zB = self._exact_fast_update(z_in[:, 1:2], a_B, b_B, dt)
            return self._sanitize_state(torch.cat([zG, zB, z_in[:, 2:]], dim=1))

        for _ in range(n_substeps):
            # Strang splitting: half fast -> full slow -> half fast.
            #
            # A naive Lie-Trotter split (full fast, then full slow) is only
            # first order, and measured even worse here (observed convergence
            # order 0.2-0.5): it makes the slow RK4 stages at t+h/2 see fast
            # modes already advanced to t+h. Strang evaluates the slow stages
            # against fast values centred on the interval, restoring second
            # order overall while keeping the fast modes exact.
            z_half = advance_fast(z, half_dt)

            h = sub_dt
            k1 = slow_rhs(z_half)
            k2 = slow_rhs(self._sanitize_state(z_half + 0.5 * h * k1))
            k3 = slow_rhs(self._sanitize_state(z_half + 0.5 * h * k2))
            k4 = slow_rhs(self._sanitize_state(z_half + h * k3))
            dz = (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            z_slow = self._sanitize_state(z_half + dz)

            z = advance_fast(z_slow, half_dt)

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

    # ------------------------------------------------------------------
    # Boundary-continuity check #2: trap-to-damage handoff continuity
    # ------------------------------------------------------------------

    def handoff_continuity_residual(
        self,
        z_enc: torch.Tensor,
        mask: torch.Tensor,
        rel_tol: float = 0.5,
    ) -> torch.Tensor:
        """
        Soft penalty on discontinuous jumps in the driving forces that feed
        zM/zL (see module docstring, boundary condition #2).

        For each pair of consecutive VALID observed time steps, computes the
        jump in driving_M and driving_L implied by the encoder's own latent
        trajectory, and penalises jumps that exceed `rel_tol` times the
        step's own driving-force magnitude — i.e. this does not forbid the
        driving force from changing (it must, that is the whole point of the
        dynamics), it forbids RELATIVE jumps larger than rel_tol, which would
        indicate a non-physical discontinuity (e.g. from an under-resolved
        substep count or a badly conditioned parameter) rather than a smooth
        physical transition.

        Args:
            z_enc   : (B, T, 5)  encoder latent trajectory
            mask    : (B, T)     bool, True where observation is valid
            rel_tol : float      max allowed relative jump before penalising

        Returns:
            residual : scalar
        """
        B, T, _ = z_enc.shape
        if T < 2:
            return torch.zeros(1, device=z_enc.device).squeeze()

        total = torch.zeros(1, device=z_enc.device)
        count = 0
        for step in range(1, T):
            valid = mask[:, step - 1] & mask[:, step]
            if not valid.any():
                continue
            z_prev = z_enc[valid, step - 1, :]
            z_curr = z_enc[valid, step, :]
            dM_prev, dL_prev = self.driving_forces(z_prev)
            dM_curr, dL_curr = self.driving_forces(z_curr)

            jump_M = (dM_curr - dM_prev).abs()
            jump_L = (dL_curr - dL_prev).abs()
            scale_M = 0.5 * (dM_curr.abs() + dM_prev.abs()) + 1e-4
            scale_L = 0.5 * (dL_curr.abs() + dL_prev.abs()) + 1e-4

            excess_M = F.relu(jump_M / scale_M - rel_tol)
            excess_L = F.relu(jump_L / scale_L - rel_tol)
            total = total + (excess_M ** 2).mean() + (excess_L ** 2).mean()
            count += 1

        if count == 0:
            return torch.zeros(1, device=z_enc.device).squeeze()
        return (total / count).squeeze()


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

    Saturation history (why the code looks like this)
    -------------------------------------------------
    An earlier version fed ``log1p(clamp(x0, min=0))`` and squashed with a
    unit-slope ``tanh``. Measured on the trained checkpoint, that combination
    killed alpha completely: alpha_raw drifted to -5.09, tanh saturated to
    -0.9999 for 100 % of devices, d(alpha)/d(raw) vanished, and every device
    sat pinned at the lower bound (alpha std = 3.3e-05 across 170 devices).

    Two compounding causes, both fixed below:

    1. ``clamp(x0, min=0)`` — x0 here is the *normalised absolute* initial
       parameter value (std ~ 1, symmetric about 0), NOT a degradation ratio,
       so the clamp flattened the entire negative half onto zero: 99.5 % of
       RON, 99.0 % of IGLeak, 94.6 % of IDLeak values were destroyed. 52.7 %
       of devices ended up with an all-zero input vector and the number of
       distinguishable devices collapsed from 202/203 to 80/203. asinh is the
       right transform: sign-preserving, ~linear near 0, log-like in the tails.

    2. Unit-slope tanh with a default-initialised output layer put the model
       in the saturated region from the start, so the gradient could never
       recover once it drifted. A gentler slope plus a small final-layer
       initialisation keeps alpha in the responsive region.

    alpha is NOT structurally unidentifiable — forcing it from 0.5 to 2.0
    moves x_pred by up to 0.235, which is the same magnitude as the entire
    test RMSE (0.240). It simply never received gradient.
    """

    def __init__(self, input_dim: int = None,
                 hidden_dim: int = 16):
        super().__init__()
        # x0 keeps its six base columns even when the observation set is
        # extended, so size from X0_STATIC_DIM rather than FEATURE_DIM.
        if input_dim is None:
            input_dim = getattr(cfg, "X0_STATIC_DIM", cfg.FEATURE_DIM) + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Start near alpha=1.0 (tanh(0)=0 -> exp(0)=1) AND near the linear
        # part of the tanh: a small final weight keeps |alpha_raw| << the
        # saturation scale for the first steps, so gradient actually flows.
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight, std=ALPHA_HEAD_INIT_STD)
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
        # Compress the wide dynamic range WITHOUT discarding the sign: x0 is
        # normalised and symmetric about 0, so clamping at 0 would erase most
        # of the per-device information (see the class docstring). asinh is
        # ~identity near 0 and ~log in the tails, so outlier devices are tamed
        # while the negative half survives intact.
        x0_feat = torch.asinh(x0_static)
        T_norm = ((T_K - cfg.T_REF_K) / cfg.T_REF_K).unsqueeze(1)  # (B,1)
        inp = torch.cat([x0_feat, T_norm], dim=1)                    # (B,7)
        alpha_raw = self.net(inp).squeeze(1)
        scale = math.log(float(cfg.ALPHA_MAX))
        # Gentler slope keeps the map in the responsive part of tanh. The
        # reachable range is unchanged -- still exactly [1/ALPHA_MAX,
        # ALPHA_MAX] -- but saturation now needs |alpha_raw| ~ 3x larger,
        # which is what lets gradient survive.
        alpha = torch.exp(scale * torch.tanh(alpha_raw / ALPHA_TANH_SLOPE))
        return alpha
