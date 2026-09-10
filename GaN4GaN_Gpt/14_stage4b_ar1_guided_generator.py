"""
14_stage4b_ar1_guided_generator.py
==================================
Stage 4B: AR(1)-guided observation-space residual generator.

This version is explicitly guided by AR(1) dynamics rather than a generic
residual head. The generator predicts per-feature autoregressive parameters
from the prefix context and uses them to generate residual trajectories.

Key idea:
  - predict rho and sigma from context
  - generate residuals via delta_t = rho * delta_{t-1} + sqrt(1-rho^2) * sigma * eps
  - fit rho and sigma to empirical residual autocorrelation / variance from
    the deterministic backbone forecast
"""

import argparse
import logging
import os
import sys
import time
import importlib.util
import pickle
import random
from typing import Dict, Optional

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


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_stage4a_module():
    path = os.path.join(BASE_DIR, "13_stage4a_residual_generator.py")
    spec = importlib.util.spec_from_file_location("_pi_stage4a_impl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stage4a_mod = _load_stage4a_module()


# Reuse helper logic from Stage 4A
_load_all = stage4a_mod._load_all
_build_model = stage4a_mod._build_model
_cache_trajectories = stage4a_mod._cache_trajectories
crps_mc_loss = stage4a_mod.crps_mc_loss
pinball_loss = stage4a_mod.pinball_loss


# Validated setting (2026-08-11): stable across 3 seeds and better than AR(1) baseline.
STAGE4B_EPOCHS = 20
STAGE4B_N_TRAIN_SAMPLES = 4
STAGE4B_N_VAL_SAMPLES = 20
STAGE4B_LR = 4e-4
STAGE4B_PATIENCE = 8
STAGE4B_HIDDEN_DIM = 96
STAGE4B_NOISE_DIM = 16

LAMBDA_CRPS = 1.0
LAMBDA_AR1 = 0.50
LAMBDA_VAR = 0.30
LAMBDA_PINBALL = 0.10
LAMBDA_SCALE = 0.01
DEFAULT_SIGMA_MIN = 0.012
DEFAULT_LOG_SCALE_FLOOR_INIT = -2.6

# Indices for leakage features whose mean residual may be non-zero
_LEAKAGE_FEAT_INDICES = [cfg.IDLEAK_DECODER_ROW, cfg.IGLEAK_DECODER_ROW]  # [4, 5]
_LEAKAGE_BIAS_BOUND = 0.30  # tanh-bounded: max |bias| in normalised space

# Features the Stage 4C generator / Stage 5 discriminator model.
#
# In the 6-feature model IDLeak (4) and IGLeak (5) had essentially zero
# residual variance -- the decoder could not move them -- so generating them
# was meaningless and they were excluded, with Stage 3's deterministic
# prediction used as an auxiliary output instead.
#
# On the 11-feature / 6-latent model the zero-variance argument no longer
# holds -- measured residual spreads are Vth 0.268, IDSS 0.927, RON 0.337,
# gmmax 0.409, IDLeak 0.327, IGLeak 0.397, SS_lin 0.107, SS_sat 0.122,
# gm_fwhm_sat 0.073, DIBL 0.104, V_gmpeak_sat 0.126 -- so the five curve
# features are generated. The leakage pair still is NOT, for a different
# reason, established by measurement rather than assumed:
#
# generating all 11 gave IDLeak Cov50/80/90 = 0.024 / 0.060 / 0.107 and
# IGLeak 0.000 / 0.021 / 0.043. An interval that covers NOTHING is not a
# calibration shortfall, it is a failed fit. Leakage residuals are dominated
# by detector-floor noise and are heavy-tailed (which is why the preprocessor
# applies LEAKAGE_LOG_CLIP and a floor at all), so a Gaussian sigma fitted to
# their typical scale cannot bracket them. Excluding the pair moves
# Cov90 0.596 -> 0.711 and CRPS 0.121 -> 0.080 across the remaining nine.
#
# Modelling them would need a heavy-tailed head (Student-t, or a much larger
# sigma floor on those channels), not simply a place in this list.
# As before, Stage 3's deterministic prediction is used for them instead.
#
# NOTE this changes what CRPSS / coverage / W1 are averaged over, so these
# metrics are NOT comparable with the 6-feature runs. Set EXTENDED_FEATURES
# to False in config.py to reproduce the old definition.
# INCLUDE_LEAKAGE re-admits IDLeak/IGLeak to the generated set. They were
# excluded because their intervals covered almost nothing (Cov50/80/90 of
# 0.024/0.060/0.107 and 0.000/0.021/0.043), which was attributed to heavy
# tails. That attribution was wrong: measured leakage kurtosis is 2.53 and
# 2.48, LIGHTER than Gaussian's 3.0, and a Student-t fit returns df ~ 1e8.
# The residuals are simply off-centre (+0.31 and +0.23 sigma), which the
# zero-mean AR(1) process could not represent until the per-feature offset
# was added. With that in place the exclusion may no longer be needed --
# this switch is what tests it.
#
# Tested twice, and it still does not work -- but each attempt moved the
# diagnosis forward.
#
# Attempt 1 (offset only): Cov90 0.0595 / 0.0213. Cause was width, not centre:
# sigma_ref sat at ~0.080 for EVERY feature while leakage needs 0.327/0.397,
# a 4-5x shortfall. That motivated the per-feature sigma_ref initialisation.
#
# Attempt 2 (offset + per-feature sigma_ref): Cov90 0.4048 / 0.2447. A large
# gain, still far short. sigma_ref is now correct (ratio 0.98 of the needed
# spread for every feature) so the remaining gap is elsewhere. Measured:
#
#   * The bounded context correction SHRINKS sigma below its reference. For
#     IDLeak the effective sigma is 0.219 against a sigma_ref of 0.322 -- a
#     32 % cut, close to the -50 % correction bound. The generator uses that
#     freedom to narrow leakage specifically.
#   * The AR(1) recursion starts from d=0, so the first forecast step has sd
#     sigma*sqrt(1-rho^2) rather than sigma. Measured at t0: IDLeak 0.188 vs
#     0.322. Leakage has the fewest observations and they sit early, so it is
#     hit hardest.
#
# Together those put the realised interval at roughly half the width the
# residuals need. For reference, a symmetric interval built directly from the
# empirical residual sd covers 0.869 (IDLeak) and 0.862 (IGLeak), so the data
# are coverable -- the generator's parameterisation is what falls short.
#
# Admitting leakage also still costs the other nine (CRPSS 0.2325 -> 0.2190),
# so it stays excluded. The fix is not another sigma tweak: it is to stop the
# context correction from narrowing, and to initialise the AR(1) state from
# its stationary distribution instead of zero.
INCLUDE_LEAKAGE = False

_LEAKAGE_SET = set(_LEAKAGE_FEAT_INDICES)
if getattr(cfg, "EXTENDED_FEATURES", False):
    STABLE_FEAT_INDICES = [i for i in range(cfg.FEATURE_DIM)
                           if INCLUDE_LEAKAGE or i not in _LEAKAGE_SET]
else:
    STABLE_FEAT_INDICES = [0, 1, 2, 3]   # Vth, IDSS, RON, gmmax
N_STABLE_FEATURES   = len(STABLE_FEAT_INDICES)


def _temp_equalized_var_loss(
    sigma_pred: "torch.Tensor",
    sigma_target: "torch.Tensor",
    T_K: "torch.Tensor",
) -> "torch.Tensor":
    """Variance MSE with equal weight per temperature bucket.

    Each unique temperature group (275/300/325 °C) contributes equally to the
    total loss regardless of how many devices fall in that group.  This prevents
    the high-sample-count temperature from dominating and ensures 325 °C (often
    under-represented and hardest to fit) receives equal attention.
    """
    import torch
    temps_c = torch.round(T_K - 273.15).long()
    unique_temps = torch.unique(temps_c)
    per_temp_losses = []
    for tc in unique_temps:
        idx = (temps_c == tc).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        per_temp_losses.append(((sigma_pred[idx] - sigma_target[idx]) ** 2).mean())
    if not per_temp_losses:
        return torch.tensor(0.0, device=sigma_pred.device, dtype=sigma_pred.dtype)
    return torch.stack(per_temp_losses).mean()


LAMBDA_CALIB = 0.0   # default off; set > 0 to activate coverage-calibration loss
CALIB_LEVELS = (0.50, 0.80, 0.90)   # nominal central-interval coverage targets


def coverage_calibration_loss(
    samples: "torch.Tensor",   # (S, B, T, F) predicted trajectories (prefix+future)
    x_true:  "torch.Tensor",   # (B, T, F)
    mask:    "torch.Tensor",   # (B, T)
    prefix_len: int,
    levels: tuple = CALIB_LEVELS,
) -> "torch.Tensor":
    """Direct calibration loss: penalise the squared gap between empirical
    coverage of the sample-based central interval and its nominal level.

    For each level p in `levels`, forms the [ (1-p)/2, (1+p)/2 ] empirical
    quantile interval from `samples` (differentiable via soft comparison of
    each sample against the true value, i.e. a smoothed indicator), and
    penalises (empirical_coverage - p)^2.

    Unlike CRPS (which only rewards sharpness+accuracy jointly), this term
    directly targets interval calibration, closing the loop between what is
    optimised during training and what coverage_XX_overall measures at eval
    time — CRPS alone can be minimised by a systematically under-dispersed
    (over-confident) generator, which is exactly the Cov90 << 0.90 failure
    mode observed empirically.
    """
    S, B, T, F = samples.shape
    future_mask = mask.clone()
    future_mask[:, :prefix_len] = 0.0
    valid = (future_mask > 0).unsqueeze(-1) & ~torch.isnan(x_true)   # (B, T, F)
    if valid.sum() == 0:
        return samples.mean() * 0.0

    x_true_c = torch.nan_to_num(x_true, nan=0.0)
    samples_c = torch.nan_to_num(samples, nan=0.0)

    total = torch.zeros((), device=samples.device, dtype=samples.dtype)
    for p in levels:
        lo_q, hi_q = (1.0 - p) / 2.0, (1.0 + p) / 2.0
        lo = torch.quantile(samples_c, lo_q, dim=0)   # (B, T, F)
        hi = torch.quantile(samples_c, hi_q, dim=0)   # (B, T, F)
        # Smooth (differentiable) inside-interval indicator via sigmoid soft-step,
        # scaled by the interval half-width so the softness adapts to local spread.
        width = (hi - lo).clamp(min=1e-4)
        sharpness = 8.0 / width
        inside = torch.sigmoid(sharpness * (x_true_c - lo)) * torch.sigmoid(sharpness * (hi - x_true_c))
        empirical_cov = (inside * valid.float()).sum() / valid.float().sum().clamp(min=1)
        total = total + (empirical_cov - p) ** 2
    return total / len(levels)


LAMBDA_PHYS_SENS = 0.0   # default off; set > 0 to activate physics-latent sensitivity loss
PHYS_SENS_MARGIN = 0.05  # minimum required (rho,sigma) response to a shuffled z_phys, normalised


def physics_sensitivity_loss(
    generator,
    z_pfx: "torch.Tensor",   # (B, latent_dim) real physics latent state
    T_K: "torch.Tensor",
    x0: "torch.Tensor",
    log_t: "torch.Tensor",
    margin: float = PHYS_SENS_MARGIN,
) -> "torch.Tensor":
    """Penalise the generator for being insensitive to z_phys.

    Ablation A/B/C showed the Stage 4C generator's CRPSS/coverage are
    statistically indistinguishable whether z_phys is real, zeroed, or
    shuffled across the batch — i.e. the network learned to ignore the
    physics latent state and rely on T/t/x0 alone.  Nonzero gradients
    through z_phys don't guarantee the *prediction* actually depends on it
    in a way that matters, so this loss targets that directly: it shuffles
    z_phys across the batch (same device's T/t/x0 kept fixed) and requires
    the predicted (rho, sigma) to differ from the real-z_phys prediction by
    at least `margin` (normalised, hinge-penalised below the margin).

    This does not by itself guarantee the *correct* use of z_phys — a
    network could satisfy this by reacting to z_phys noise arbitrarily. It
    is a necessary-but-not-sufficient condition, meant to be combined with
    re-running the A/B/C ablation to confirm CRPSS/coverage/OOD-temperature
    performance improves with real z_phys relative to shuffled — that is
    the actual test of whether the sensitivity is meaningful, not just present.
    """
    import torch
    B = z_pfx.shape[0]
    if B <= 1:
        return z_pfx.sum() * 0.0

    rho_real, sigma_real = generator._context_params(z_pfx, T_K, x0, log_t)

    perm = torch.randperm(B, device=z_pfx.device)
    z_shuffled = z_pfx[perm]
    rho_shuf, sigma_shuf = generator._context_params(z_shuffled, T_K, x0, log_t)

    rho_diff = (rho_real - rho_shuf).abs().mean(dim=-1)      # (B,)
    sigma_diff = ((sigma_real - sigma_shuf).abs() / sigma_real.detach().clamp(min=1e-4)).mean(dim=-1)  # (B,)
    response = 0.5 * (rho_diff + sigma_diff)                  # (B,)

    return torch.relu(margin - response).mean()


LAMBDA_ZPHYS_CONTRAST = 0.0   # default off; set > 0 to activate z_phys contrastive loss
ZPHYS_CONTRAST_MARGIN = 0.02  # min required CRPS degradation (shuffled - real), in normalised units


def _per_device_crps(
    samples: "torch.Tensor",   # (S, B, T_future, F)
    x_true_future: "torch.Tensor",   # (B, T_future, F)
    future_mask: "torch.Tensor",     # (B, T_future) bool
) -> "torch.Tensor":
    """Per-device CRPS (energy-score MC estimator), NOT averaged over the
    batch — returns (B,). Same formula as crps_mc_loss (13_stage4a...), but
    reduced over (T_future, F) independently per device so the contrastive
    loss below can compare real-z_phys vs shuffled-z_phys CRPS device-by-
    device rather than only as a batch aggregate (which would wash out the
    per-device signal the ablation is actually about)."""
    S, B, T, F = samples.shape
    valid = future_mask.unsqueeze(0).unsqueeze(-1).expand(S, B, T, F)   # (S,B,T,F)
    x_true_exp = x_true_future.unsqueeze(0).expand(S, -1, -1, -1)
    nan_mask = ~torch.isnan(x_true_exp)
    final_mask = valid & nan_mask

    samples_c = torch.nan_to_num(samples, nan=0.0)
    s_clean = torch.where(nan_mask, samples_c, torch.zeros_like(samples_c))
    y_clean = torch.where(nan_mask, x_true_exp, torch.zeros_like(x_true_exp))

    denom = final_mask.float().sum(dim=(0, 2, 3)).clamp(min=1)   # (B,)
    term1 = (torch.abs(s_clean - y_clean) * final_mask.float()).sum(dim=(0, 2, 3)) / denom   # (B,)

    n_pairs = min(S, 8)
    idx1 = torch.randperm(S, device=samples.device)[:n_pairs]
    idx2 = torch.randperm(S, device=samples.device)[:n_pairs]
    pair_mask = (future_mask.unsqueeze(-1) & ~torch.isnan(x_true_future)).unsqueeze(0).expand(n_pairs, -1, -1, -1)
    s1 = s_clean[idx1]
    s2 = s_clean[idx2]
    pdenom = pair_mask.float().sum(dim=(0, 2, 3)).clamp(min=1)   # (B,)
    term2 = (torch.abs(s1 - s2) * pair_mask.float()).sum(dim=(0, 2, 3)) / pdenom   # (B,)

    return term1 - 0.5 * term2   # (B,)


def z_phys_contrastive_loss(
    generator,
    z_pfx: "torch.Tensor",        # (B, latent_dim)
    T_K: "torch.Tensor",
    x0: "torch.Tensor",
    log_t: "torch.Tensor",
    x_true_future: "torch.Tensor",   # (B, T_future, F_stable)
    future_mask: "torch.Tensor",     # (B, T_future) bool
    x_hat_future: "torch.Tensor",    # (B, T_future, F_stable) deterministic backbone mean
    n_samples: int = 4,
    times_future: "torch.Tensor" = None,
    margin: float = ZPHYS_CONTRAST_MARGIN,
) -> "torch.Tensor":
    """Directly optimises what the A/B/C ablation measures: real z_phys
    should give a BETTER (lower) CRPS than a wrong device's shuffled z_phys.

    physics_sensitivity_loss (above) only required the predicted (rho,sigma)
    to CHANGE under a shuffled z_phys — satisfiable by reacting to z_phys
    noise in a way that does not help or could even hurt prediction quality.
    This loss closes that gap by scoring the actual downstream quantity
    (per-device CRPS) under both conditions and penalising the model unless
    real z_phys wins by at least `margin`:

        L = mean_i  relu(margin - (CRPS_shuffled_i - CRPS_real_i))

    Gradients flow through both the real-z_phys and shuffled-z_phys forward
    passes into the SAME generator parameters, so satisfying this loss
    requires the network to have actually learned a z_phys-dependent
    function whose correctness (not just its existence) matters for CRPS —
    the necessary-and-sufficient condition the ablation checks for.
    """
    B = z_pfx.shape[0]
    if B <= 1:
        return z_pfx.sum() * 0.0

    deltas_real = generator.sample_n(z_pfx, T_K, x0, log_t, n_samples,
                                      T_future=x_true_future.shape[1], times_future=times_future)
    x_pred_real = x_hat_future.unsqueeze(0) + deltas_real
    crps_real = _per_device_crps(x_pred_real, x_true_future, future_mask)   # (B,)

    perm = torch.randperm(B, device=z_pfx.device)
    z_shuffled = z_pfx[perm]
    deltas_shuf = generator.sample_n(z_shuffled, T_K, x0, log_t, n_samples,
                                      T_future=x_true_future.shape[1], times_future=times_future)
    x_pred_shuf = x_hat_future.unsqueeze(0) + deltas_shuf
    crps_shuf = _per_device_crps(x_pred_shuf, x_true_future, future_mask)   # (B,)

    gap = crps_shuf - crps_real   # positive => real z_phys is better, as desired
    return torch.relu(margin - gap).mean()


LAMBDA_ARRHENIUS_TREND = 0.0   # default off; set > 0 to activate Arrhenius sigma-trend loss

# Reference activation energies for residual-variance temperature scaling,
# per stable feature [Vth, IDSS, RON, gmmax], fitted from a robustified
# (MAD-based, outlier-resistant) per-temperature-bucket analysis of decoder
# residuals on this dataset (2026-08-21 investigation). This is a FIXED
# constant, not a learned parameter — AR1GuidedResidualGeneratorArrhenius
# (a hard-constraint variant tried first) showed that when Ea_sigma is left
# learnable inside the sigma-prediction path itself, the optimizer has no
# pressure to move it off a sane init and the model instead collapses the
# context-dependent correction term to near-zero. Keeping the reference
# fixed and only constraining the population-level TREND (not individual
# sigma values) avoids that failure mode: the MLP stays fully free to fit
# per-device sigma, only the batch-averaged cross-temperature-group ratio
# is nudged towards physical plausibility.
# feature index -> reference Ea [eV]. Only the four original features have a
# value; features absent from this dict are simply skipped by the loss.
#
# Deliberately NOT extended to the curve features. Fitting Ea from the
# residual spread across the three temperatures gives numbers, but the spread
# is not monotone in T for several features (RON: 0.541, 0.098, 0.174 at
# 548/573/598 K), so a two-parameter Arrhenius fit through three
# non-monotone points mostly encodes the 598 K outlier rather than an
# activation energy. Constraining sigma(T) towards such a value would inject
# that artefact as a prior. The four existing values were validated
# separately and are kept.
ARRHENIUS_TREND_EA_REF = {0: 0.67, 1: 0.35, 2: 0.20, 3: 0.35}

# Starting Ea for a generated feature with no validated reference value. 0.35 eV
# is the middle of the four measured ones; it is only an INITIALISATION -- the
# Arrhenius-sigma generator learns Ea_sigma per feature, and the trend loss
# constrains only the features listed in ARRHENIUS_TREND_EA_REF above.
EA_SIGMA_DEFAULT = 0.35

# Bound on the learnable per-feature residual offset, in normalised units.
#
# The AR(1) residual process is zero-mean by construction, so the generator can
# only place its interval symmetrically around the Stage-3 prediction. Measured
# on the 11-feature backbone, several residuals are systematically off-centre:
#
#   feature        mean/std of residual      fraction positive
#   SS_sat              +0.61                     0.91
#   gm_fwhm_sat         -0.48                     0.14
#   DIBL                +0.32                     0.82
#   IDLeak              +0.31                     0.61
#   SS_lin              +0.30                     0.64
#   IGLeak              +0.23                     0.59
#
# A 90 % interval of the right WIDTH but the wrong CENTRE under-covers. Direct
# check on IDLeak: a symmetric interval about zero covers 0.892, the same
# interval recentred on the residual mean covers 0.931.
#
# 0.6 in normalised units is ~2x the largest measured offset, so the bound
# never binds in practice; it exists to stop the offset absorbing signal that
# belongs in the ODE.
OFFSET_BOUND = 0.6

# Per-feature starting value for the Arrhenius sigma reference, keyed by
# feature index, in normalised residual units.
#
# Previously every feature started at a single scalar (0.08) and barely moved:
# after training, sigma_ref spanned 0.0793-0.0818, a 3.2 % spread across
# features whose residual spreads differ 13-fold (0.073 for gm_fwhm_sat to
# 0.927 for IDSS). The bounded context correction (+-50 %) cannot bridge that:
# 6 of 9 features could not reach their required width even at the cap, and
# leakage was short by 4-5x. Coverage was therefore capped from below by an
# interval that was simply too narrow -- Vth needed 3.3x more.
#
# These are the MEASURED per-device residual standard deviations of the
# Stage-3 forecast on the 11-feature backbone, so the generator starts at the
# right order of magnitude and the optimiser refines rather than rediscovers
# it. Features absent from the map fall back to SIGMA_REF_DEFAULT.
SIGMA_REF_BY_FEATURE = {
    "Vth": 0.268, "IDSS": 0.927, "RON": 0.337, "gmmax": 0.409,
    "IDLeak": 0.327, "IGLeak": 0.397,
    "SS_lin": 0.107, "SS_sat": 0.122, "gm_fwhm_sat": 0.073,
    "DIBL": 0.104, "V_gmpeak_sat": 0.126,
}
SIGMA_REF_DEFAULT = 0.08

# Whether sigma_ref should be re-derived from the CURRENT dataset with a robust
# spread estimator rather than read from the table above.
#
# The table holds plain standard deviations, and those are inflated by a few
# devices: measured sd/MAD ratios are IDSS 30.1x, DIBL 12.4x, RON 11.5x,
# V_gmpeak 8.5x, gmmax 8.4x, and only IDLeak 1.3x / IGLeak 1.1x. IDSS at 30x
# is why it over-covers (Cov90 0.9565) with an interval far wider than the
# typical device needs. Dropping the three faulty devices only takes IDSS from
# 0.903 to 0.655 against a MAD of 0.030, so device filtering alone does not
# fix it.
#
# Left OFF until measured, and note that "robust" is NOT simply narrower --
# computed on the filtered dataset it moves features in OPPOSITE directions
# (table value / 1.5*MAD):
#
#   IDSS 20.8x   RON 6.0x   gmmax 3.9x    <- table too WIDE, robust narrows
#   DIBL 0.8x    V_gmpeak 0.6x            <- about right
#   Vth 0.4x     SS_lin 0.5x  SS_sat 0.5x <- table too NARROW, robust widens
#   IDLeak 0.2x  IGLeak 0.2x              <- table far too narrow: robust
#                                            widens these ~5x, which is close
#                                            to the 4-5x shortfall measured
#                                            when leakage under-covered
#
# So this is a re-scaling per feature, not a global shrink, and it may
# incidentally address the leakage width problem. It still needs the same
# before/after comparison every other width change got.
SIGMA_REF_ROBUST = False
# 1.4826 * MAD estimates the sd of a Gaussian; scale it up so the interval is
# not set by the median device alone.
SIGMA_ROBUST_SCALE = 1.5

_ROBUST_CACHE = {}


def _robust_sigma_table():
    """Per-feature sigma from a MAD spread of the CURRENT dataset.

    Computed from the data rather than the hard-coded table, so it follows
    whichever dataset is loaded (filtered or not). Cached per path since the
    generator is constructed many times per run.
    """
    path = cfg.PROCESSED_DATA_PATH
    if path in _ROBUST_CACHE:
        return _ROBUST_CACHE[path]
    import pickle as _pk
    try:
        with open(path, "rb") as fh:
            _ds = _pk.load(fh)
    except Exception:                                  # noqa: BLE001
        return SIGMA_REF_BY_FEATURE
    X = np.asarray(_ds["x_raw_deg"])
    FMk = np.asarray(_ds["feature_mask"])
    MKk = np.asarray(_ds["mask"])
    out = {}
    for _f, _n in enumerate(cfg.FEATURES):
        v = X[:, :, _f][FMk[:, :, _f] & MKk]
        v = v[np.isfinite(v)]
        if v.size < 10:
            out[_n] = SIGMA_REF_BY_FEATURE.get(_n, SIGMA_REF_DEFAULT)
            continue
        mad = 1.4826 * float(np.median(np.abs(v - np.median(v))))
        out[_n] = max(mad * SIGMA_ROBUST_SCALE, 1e-3)
    _ROBUST_CACHE[path] = out
    return out

# --- interval-width fixes (both default OFF; see the leakage notes above) ---
#
# Measured on the sigma-initialised generator, two mechanisms leave the
# realised interval at roughly half the width the residuals need:
#
#   1. The bounded context correction can shrink sigma as well as grow it.
#      IDLeak's effective sigma came out 0.219 against a sigma_ref of 0.322,
#      a 32 % cut sitting near the -50 % bound.
#   2. The AR(1) recursion starts at d=0, so the FIRST forecast step has
#      sd = sigma*sqrt(1-rho^2) instead of sigma. Measured at t0: IDLeak
#      0.188 vs 0.322, Vth 0.209 vs 0.262. Early steps are systematically
#      under-dispersed for every feature.
#
# Both are opt-in so each can be attributed separately.
#
# TESTED, AND BOTH ARE LEFT OFF. Each raises Cov90 but costs more sharpness
# than it buys, on the nine features currently generated:
#
#   variant            CRPS    CRPSS   Cov50   Cov80   Cov90    MACE
#   baseline         0.0754   0.2325  0.6445  0.8584  0.9079  0.0287
#   one-sided only   0.0805   0.1880  0.7147  0.8966  0.9258  0.0486
#   stat-init only   0.0808   0.1918  0.6531  0.8579  0.9053  0.0334
#   both             0.1003   0.0269  0.7719  0.9155  0.9442  0.0677
#
# The baseline already meets the 0.90 target, so extra width is pure loss:
# with both on, Vth reaches 0.9815 and IDSS 0.9938 -- far past nominal -- and
# CRPSS collapses from 0.2325 to 0.0269. The diagnosis that leakage is
# under-covered because its realised interval is too narrow still stands, but
# widening EVERY feature is the wrong instrument, because only leakage is
# short. A leakage-specific remedy would have to widen those two channels
# without touching the nine that are already calibrated.
CORRECTION_ONE_SIDED = False   # if True, context may only widen sigma, never narrow
AR1_STATIONARY_INIT  = False   # if True, seed the AR(1) state from N(0, sigma)


def arrhenius_trend_loss(
    sigma_pred: "torch.Tensor",   # (B, n_features)
    T_K: "torch.Tensor",          # (B,)
    ea_ref: dict = ARRHENIUS_TREND_EA_REF,
) -> "torch.Tensor":
    """Soft population-level constraint: the BATCH-AVERAGE predicted sigma,
    grouped by temperature, should scale across temperature groups
    consistently with a physically-plausible Arrhenius activation energy —
    without constraining any individual device's sigma.

    For every pair of temperature groups (t_i, t_j) present in the batch,
    compares the empirical log-ratio of mean sigma to the log-ratio implied
    by the fixed reference Ea (see ARRHENIUS_TREND_EA_REF), and penalises
    the squared difference:

        log(sigma_bar(t_i)/sigma_bar(t_j)) should ~= -Ea/kB * (1/t_i - 1/t_j)

    This targets exactly the quantity independently validated on this
    dataset (population-level temperature scaling of residual variance,
    confirmed via a robustified per-temperature-bucket fit — see
    AR1GuidedResidualGeneratorArrhenius docstring), while leaving individual
    predictions fully free — unlike the earlier hard-constraint variant,
    there is no per-device sigma being pinned to a physical formula, so the
    MLP cannot "give up" on device-level fitting to satisfy this term; it
    only has to keep the handful of per-temperature-group AVERAGES on trend.
    """
    kb = cfg.KB_EV
    B, F = sigma_pred.shape
    temps_c = torch.round(T_K - 273.15).long()
    unique_temps = torch.unique(temps_c)
    if unique_temps.numel() < 2:
        return torch.zeros((), device=sigma_pred.device, dtype=sigma_pred.dtype)

    group_mean_sigma = {}   # tc -> (F,) mean sigma across devices in that group
    group_T_K = {}
    for tc in unique_temps.tolist():
        idx = (temps_c == tc).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        group_mean_sigma[tc] = sigma_pred[idx].mean(dim=0)   # (F,)
        group_T_K[tc] = float(tc) + cfg.CELSIUS_TO_KELVIN

    temps_list = sorted(group_mean_sigma.keys())
    if len(temps_list) < 2:
        return torch.zeros((), device=sigma_pred.device, dtype=sigma_pred.dtype)

    ea_t = torch.tensor([ea_ref.get(fi, 0.3) for fi in range(F)],
                         device=sigma_pred.device, dtype=sigma_pred.dtype)   # (F,)

    total = torch.zeros((), device=sigma_pred.device, dtype=sigma_pred.dtype)
    n_pairs = 0
    for i in range(len(temps_list)):
        for j in range(i + 1, len(temps_list)):
            ti, tj = temps_list[i], temps_list[j]
            log_ratio_pred = torch.log(group_mean_sigma[ti].clamp(min=1e-6)) - \
                              torch.log(group_mean_sigma[tj].clamp(min=1e-6))   # (F,)
            inv_T_diff = 1.0 / group_T_K[ti] - 1.0 / group_T_K[tj]
            log_ratio_target = -ea_t / kb * inv_T_diff   # (F,)
            total = total + ((log_ratio_pred - log_ratio_target) ** 2).mean()
            n_pairs += 1

    return total / max(n_pairs, 1)


LAMBDA_ACF = 0.0   # default off; set e.g. 0.10 for Stage 4C-ACF variant


def _batch_acf_at_lag(
    residuals: "torch.Tensor",
    valid_mask: "torch.Tensor",
    lag: int,
) -> "torch.Tensor":
    """Estimate index-based autocorrelation at given lag.

    residuals  : (B, T, F)  zero-mean residuals
    valid_mask : (B, T)     True for valid time steps
    Returns    : (F,)       ACF estimate per feature
    """
    import torch
    B, T, F = residuals.shape
    if T <= lag:
        return torch.zeros(F, device=residuals.device, dtype=residuals.dtype)

    res = torch.nan_to_num(residuals, nan=0.0)
    # Paired mask: both t and t+lag must be valid
    pair_mask = (valid_mask[:, :T - lag].float() * valid_mask[:, lag:].float())  # (B, T-lag)
    n_pairs = pair_mask.sum().clamp(min=2.0)

    r_t    = res[:, :T - lag, :]   # (B, T-lag, F)
    r_tlag = res[:, lag:,     :]   # (B, T-lag, F)
    w      = pair_mask.unsqueeze(-1)  # (B, T-lag, 1)

    mu_t   = (r_t   * w).sum(dim=[0, 1]) / n_pairs  # (F,)
    mu_lag = (r_tlag * w).sum(dim=[0, 1]) / n_pairs  # (F,)

    dt   = (r_t   - mu_t)   * w   # (B, T-lag, F)
    dlag = (r_tlag - mu_lag) * w   # (B, T-lag, F)

    cov    = (dt * dlag).sum(dim=[0, 1])          # (F,)
    var_t  = (dt   ** 2).sum(dim=[0, 1]).clamp(min=1e-8)   # (F,)
    var_l  = (dlag ** 2).sum(dim=[0, 1]).clamp(min=1e-8)   # (F,)

    return (cov / torch.sqrt(var_t * var_l)).clamp(-1.0, 1.0)


def acf_matching_loss(
    deltas: "torch.Tensor",         # (S, B, T_future, F_s)  generated residuals
    x_true: "torch.Tensor",         # (B, T, F_all)
    x_hat:  "torch.Tensor",         # (B, T, F_all)
    mask:   "torch.Tensor",         # (B, T)
    prefix_len: int,
    stable_feat_indices: list,
    max_lag: int = 2,
    weights = None,
) -> "torch.Tensor":
    """Index-based ACF matching loss.

    L_ACF = (1/F) sum_f sum_{l=1}^{max_lag}  w_l * (rho_gen_f(l) - rho_real_f(l))^2

    Designed for Stage 4C training to improve temporal correlation of generated
    residuals without increasing the adversarial weight.
    """
    import torch
    if weights is None:
        weights = [1.0, 0.5]   # lag-1 weight=1.0, lag-2 weight=0.5

    sfx = torch.tensor(stable_feat_indices, device=deltas.device)
    future_mask = mask[:, prefix_len:].bool()           # (B, T_future)

    # True residuals in future window (stable features)
    r_real = (x_true[:, prefix_len:, sfx] - x_hat[:, prefix_len:, sfx]).detach()  # (B, T_f, F_s)

    S = deltas.shape[0]
    n_lags = min(max_lag, len(weights))
    total = torch.zeros(1, device=deltas.device, dtype=deltas.dtype)

    for li in range(n_lags):
        lag = li + 1
        w   = weights[li]
        rho_real = _batch_acf_at_lag(r_real, future_mask, lag)          # (F_s,)
        rho_gen  = torch.stack([
            _batch_acf_at_lag(deltas[s], future_mask, lag) for s in range(S)
        ]).mean(dim=0)                                                   # (F_s,)
        total = total + w * ((rho_gen - rho_real) ** 2).mean()

    return total / max(n_lags, 1)


class AR1GuidedResidualGenerator(nn.Module):
    """AR(1)-guided observation-space residual generator."""

    def __init__(
        self,
        noise_dim: int = STAGE4B_NOISE_DIM,
        hidden_dim: int = STAGE4B_HIDDEN_DIM,
        n_features: int = cfg.FEATURE_DIM,
        latent_dim: int = cfg.LATENT_DIM,
        log_scale_floor_init: float = DEFAULT_LOG_SCALE_FLOOR_INIT,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_features = n_features
        self.latent_dim = latent_dim

        context_dim = latent_dim + 1 + n_features + 1
        in_dim = noise_dim + context_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_features),
        )

        self.log_scale_floor = nn.Parameter(torch.full((n_features,), float(log_scale_floor_init)))

        # Independent bias head for leakage features (IDLeak, IGLeak).
        # Learns a deterministic, context-dependent mean correction for each
        # leakage feature so the stochastic AR(1) residuals can remain zero-mean.
        self._leakage_indices = _LEAKAGE_FEAT_INDICES
        n_leakage = len(self._leakage_indices)
        self.leakage_bias_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_leakage),
        )
        # Initialise to zero so the bias starts at no-correction
        nn.init.zeros_(self.leakage_bias_head[-1].weight)
        nn.init.zeros_(self.leakage_bias_head[-1].bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.15)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _compute_leakage_bias(
        self,
        z_prefix_last: "torch.Tensor",
        T_K: "torch.Tensor",
        x0: "torch.Tensor",
        log_t_suffix: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return a bounded (B, n_leakage) bias for IDLeak / IGLeak.

        The bias is deterministic (context-only, no noise) and bounded to
        ±LEAKAGE_BIAS_BOUND in normalised space via tanh.  The AR(1) stochastic
        residuals remain zero-mean and are added on top of this correction.
        """
        import torch
        z_prefix_last = torch.nan_to_num(z_prefix_last, nan=0.5)
        x0 = torch.nan_to_num(x0, nan=0.0)
        T_K = torch.nan_to_num(T_K, nan=0.0)
        log_t_suffix = torch.nan_to_num(log_t_suffix, nan=0.0)

        B = z_prefix_last.shape[0]
        dev = z_prefix_last.device

        if T_K.dim() <= 1:
            T_norm = ((T_K.reshape(-1) - 300.0) / 25.0).reshape(B, 1)
        else:
            T_norm = T_K.reshape(B, -1)[:, :1]

        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        if x0.dim() > 2:
            x0 = x0.reshape(B, -1)
        if x0.shape[0] != B:
            x0 = x0.expand(B, -1)

        if log_t_suffix.dim() == 1:
            log_t = log_t_suffix.reshape(B, 1)
        else:
            log_t = log_t_suffix.reshape(B, -1)[:, :1]

        ctx = torch.cat([z_prefix_last.reshape(B, -1), T_norm, x0.reshape(B, -1), log_t], dim=-1)
        raw = self.leakage_bias_head(ctx)          # (B, n_leakage)
        return _LEAKAGE_BIAS_BOUND * torch.tanh(raw)

    def _context_params(self, z_prefix_last, T_K, x0, log_t_suffix, noise_init=None):
        z_prefix_last = torch.nan_to_num(z_prefix_last, nan=0.5, posinf=0.0, neginf=0.0)
        x0 = torch.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)
        T_K = torch.nan_to_num(T_K, nan=0.0, posinf=0.0, neginf=0.0)
        log_t_suffix = torch.nan_to_num(log_t_suffix, nan=0.0, posinf=0.0, neginf=0.0)

        if z_prefix_last.dim() == 1:
            z_prefix_last = z_prefix_last.unsqueeze(0)
        B = z_prefix_last.shape[0]
        dev = z_prefix_last.device

        if T_K.dim() == 0:
            T_K = T_K.unsqueeze(0)
        if T_K.dim() == 1:
            T_norm = ((T_K - 300.0) / 25.0).reshape(B, -1)[:, :1]
        else:
            T_norm = T_K.reshape(B, -1)[:, :1]

        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        if x0.dim() > 2:
            x0 = x0.reshape(B, -1)
        if x0.shape[0] != B:
            x0 = x0.expand(B, -1)

        if log_t_suffix.dim() == 1:
            log_t_suffix = log_t_suffix.reshape(B, -1)
        elif log_t_suffix.dim() > 2:
            log_t_suffix = log_t_suffix.reshape(B, -1)
        log_t = log_t_suffix.reshape(B, -1)[:, :1]

        ctx = torch.cat([z_prefix_last.reshape(B, -1), T_norm, x0.reshape(B, -1), log_t], dim=-1)
        if len(ctx.shape) == 1:
            ctx = ctx.unsqueeze(0)
        if ctx.shape[-1] != self.latent_dim + 1 + self.n_features + 1:
            target_dim = self.latent_dim + 1 + self.n_features + 1
            if ctx.shape[-1] > target_dim:
                ctx = ctx[:, :target_dim]
            else:
                pad = torch.zeros(B, target_dim - ctx.shape[-1], device=dev, dtype=ctx.dtype)
                ctx = torch.cat([ctx, pad], dim=-1)

        # Keep parameter prediction deterministic by default.
        # Stochasticity for trajectories is already injected in the AR(1) innovation eps.
        if noise_init is None:
            noise_init = torch.zeros(B, self.noise_dim, device=dev, dtype=ctx.dtype)
        else:
            noise_init = torch.nan_to_num(noise_init, nan=0.0, posinf=0.0, neginf=0.0)
            if noise_init.dim() == 1:
                noise_init = noise_init.unsqueeze(0)
            if noise_init.shape[0] != B:
                noise_init = noise_init.expand(B, -1)
        inp = torch.cat([noise_init, ctx], dim=-1)
        out = self.net(inp)
        rho_raw = out[:, : self.n_features]
        log_sigma_raw = out[:, self.n_features :]

        rho = torch.tanh(rho_raw).clamp(-0.95, 0.95)
        sigma_floor = torch.exp(self.log_scale_floor).unsqueeze(0).expand(B, -1)
        sigma = sigma_floor + F.softplus(log_sigma_raw)
        return rho, sigma

    def forward(self, z_prefix_last, T_K, x0, log_t_suffix, T_future=10, noise=None):
        B = z_prefix_last.shape[0]
        dev = z_prefix_last.device
        rho, sigma = self._context_params(z_prefix_last, T_K, x0, log_t_suffix)

        # --- Zero-mean AR(1) stochastic residuals ---
        deltas = []
        d_prev = torch.zeros(B, self.n_features, device=dev)
        for _ in range(T_future):
            eps = torch.randn(B, self.n_features, device=dev)
            sq = torch.sqrt((1.0 - rho ** 2).clamp(min=1e-6))
            d_t = rho * d_prev + sq * sigma * eps
            deltas.append(d_t)
            d_prev = d_t.detach()

        stoch = torch.stack(deltas, dim=1)   # (B, T_future, F) — zero-mean in expectation

        # --- Deterministic leakage bias correction ---
        # Bounded ±LEAKAGE_BIAS_BOUND for IDLeak / IGLeak; zero for all other features.
        bias = torch.zeros(B, self.n_features, device=dev)
        leakage_bias = self._compute_leakage_bias(z_prefix_last, T_K, x0, log_t_suffix)  # (B, n_leakage)
        for local_i, feat_i in enumerate(self._leakage_indices):
            bias[:, feat_i] = leakage_bias[:, local_i]

        return stoch + bias.unsqueeze(1)     # (B, T_future, F)

    def sample_n(self, z_prefix_last, T_K, x0, log_t_suffix, n_samples, T_future=10):
        deltas = [
            self.forward(z_prefix_last, T_K, x0, log_t_suffix, T_future=T_future)
            for _ in range(n_samples)
        ]
        return torch.stack(deltas, dim=0)


class AR1GuidedResidualGeneratorStable(nn.Module):
    """Stage 4C: AR(1) generator for stable features ONLY (Vth, IDSS, RON, gmmax).

    IDLeak and IGLeak are EXCLUDED from generation and adversarial training.
    Stage 3 deterministic predictions serve as their auxiliary outputs.

    Architecture:
    - Context uses the full 6-dim x0 vector (richer conditioning).
    - Network outputs 2 × 4 = 8 values → rho + sigma for 4 stable features.
    - No leakage bias head.
    - forward / sample_n output shape: (B, T_future, 4) / (S, B, T_future, 4).
    """
    STABLE_INDICES = STABLE_FEAT_INDICES   # [0, 1, 2, 3]
    N_STABLE       = N_STABLE_FEATURES     # 4

    def __init__(
        self,
        noise_dim:           int   = STAGE4B_NOISE_DIM,
        hidden_dim:          int   = STAGE4B_HIDDEN_DIM,
        n_output:            int   = N_STABLE_FEATURES,
        latent_dim:          int   = cfg.LATENT_DIM,
        n_context_feat:      int   = cfg.FEATURE_DIM,   # full 6-dim x0 context
        log_scale_floor_init:float = DEFAULT_LOG_SCALE_FLOOR_INIT,
    ):
        super().__init__()
        self.noise_dim      = noise_dim
        self.n_features     = n_output          # 4
        self.latent_dim     = latent_dim
        self._n_ctx_feat    = n_context_feat    # 6
        self._leakage_indices = []              # no leakage bias

        context_dim = latent_dim + 1 + n_context_feat + 1
        in_dim      = noise_dim + context_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_output),
        )
        self.log_scale_floor = nn.Parameter(
            torch.full((n_output,), float(log_scale_floor_init))
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.15)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Per-feature residual offset; see OFFSET_BOUND. Zero at init, so an
        # untrained generator reproduces the pre-offset behaviour exactly.
        self.offset_raw = nn.Parameter(torch.zeros(n_output))

    @property
    def offset(self) -> torch.Tensor:
        """Learned per-feature residual offset, bounded to +-OFFSET_BOUND."""
        return OFFSET_BOUND * torch.tanh(self.offset_raw)

    def _context_params(self, z_prefix_last, T_K, x0, log_t_suffix, noise_init=None):
        """Return (rho, sigma) tensors shaped (B, n_output), one per generated feature."""
        z_prefix_last = torch.nan_to_num(z_prefix_last, nan=0.5, posinf=0.0, neginf=0.0)
        x0            = torch.nan_to_num(x0,            nan=0.0, posinf=0.0, neginf=0.0)
        T_K           = torch.nan_to_num(T_K,           nan=0.0, posinf=0.0, neginf=0.0)
        log_t_suffix  = torch.nan_to_num(log_t_suffix,  nan=0.0, posinf=0.0, neginf=0.0)

        if z_prefix_last.dim() == 1:
            z_prefix_last = z_prefix_last.unsqueeze(0)
        B   = z_prefix_last.shape[0]
        dev = z_prefix_last.device

        T_norm = ((T_K.reshape(-1) - 300.0) / 25.0).reshape(B, 1)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        if x0.dim() > 2:
            x0 = x0.reshape(B, -1)
        x0_ctx = x0[:, :self._n_ctx_feat]   # first 6 dims (or however many exist)

        log_t = log_t_suffix.reshape(B, -1)[:, :1]

        ctx = torch.cat([z_prefix_last.reshape(B, -1), T_norm, x0_ctx.reshape(B, -1), log_t], dim=-1)

        if noise_init is None:
            noise_init = torch.zeros(B, self.noise_dim, device=dev, dtype=ctx.dtype)
        else:
            noise_init = torch.nan_to_num(noise_init, nan=0.0)
            if noise_init.dim() == 1:
                noise_init = noise_init.unsqueeze(0)

        inp      = torch.cat([noise_init, ctx], dim=-1)
        out      = self.net(inp)
        # Positive-only rho: sigmoid * 0.97  so rho in (0, 0.97)
        # Removes the sign×abs^expo NaN hazard and restricts to decay-only OU.
        rho      = torch.sigmoid(out[:, :self.n_features]) * 0.97
        sig_floor= torch.exp(self.log_scale_floor).unsqueeze(0).expand(B, -1)
        sigma    = sig_floor + F.softplus(out[:, self.n_features:])
        return rho, sigma

    # Reference log10-time step (≈ median Δlog10t on the future window of this dataset)
    # Future times: 10, 20, 50, 100, 200, 500, 1000, 2000 h → Δlog10 ≈ 0.30–0.40
    LOG10_T_REF = 0.35   # reference Deltalog10(1+t) step for the OU parameterisation

    def forward(self, z_prefix_last, T_K, x0, log_t_suffix, T_future=10, noise=None,
                times_future=None):
        """Continuous-time OU on the log10(1+t) axis.

        rho_eff(i) = rho_ref ^ (Deltalog10(1+t_i) / LOG10_T_REF)

        log10(1+t) handles t=0 gracefully.
        rho_ref is in (0, 0.97) so rho_eff is always a valid decay coefficient;
        no NaN from non-integer exponentiation of negative numbers.
        sqrt(1-rho_i^2) is included as innovation scale at every step.
        """
        B   = z_prefix_last.shape[0]
        dev = z_prefix_last.device
        rho, sigma = self._context_params(z_prefix_last, T_K, x0, log_t_suffix)

        rho_eff_per_step = None
        if times_future is not None and times_future.shape[1] >= 2:
            t = times_future.to(dev).clamp(min=0.0)      # (B, T_future)
            log1pt = torch.log10(1.0 + t)                 # log10(1+t)
            delta_log10 = torch.zeros(B, T_future, device=dev)
            delta_log10[:, 0] = self.LOG10_T_REF           # first step: reference interval
            if T_future > 1:
                delta_log10[:, 1:] = (
                    log1pt[:, 1:] - log1pt[:, :-1]
                ).clamp(min=0.01, max=2.0)
            expo = (delta_log10 / self.LOG10_T_REF).unsqueeze(-1)  # (B, T_future, 1)
            # rho in (0, 0.97), expo > 0  =>  rho^expo in (0, 0.97): always real
            rho_eff_per_step = rho.unsqueeze(1) ** expo             # (B, T_future, F)

        deltas = []
        d_prev = torch.zeros(B, self.n_features, device=dev)
        for t_idx in range(T_future):
            eps   = torch.randn(B, self.n_features, device=dev)
            rho_i = rho_eff_per_step[:, t_idx, :] if rho_eff_per_step is not None else rho
            sq    = torch.sqrt((1.0 - rho_i ** 2).clamp(min=1e-6))  # innovation scale
            d_t   = rho_i * d_prev + sq * sigma * eps
            deltas.append(d_t)
            d_prev = d_t.detach()
        out = torch.stack(deltas, dim=1)          # (B, T_future, n_features)
        # Shift the whole zero-mean AR(1) path onto the residual's true centre.
        # Added AFTER the recursion so it does not feed back through rho and
        # inflate later steps.
        return out + self.offset.view(1, 1, -1)

    def sample_n(self, z_prefix_last, T_K, x0, log_t_suffix, n_samples, T_future=10,
                 times_future=None):
        return torch.stack([
            self.forward(z_prefix_last, T_K, x0, log_t_suffix, T_future=T_future,
                         times_future=times_future)
            for _ in range(n_samples)
        ], dim=0)   # (S, B, T_future, 4)


class AR1GuidedResidualGeneratorPhysGated(nn.Module):
    """Stage 4C generator variant with a structural physics bottleneck.

    Motivation (2026-08-16..21 investigation): AR1GuidedResidualGeneratorStable
    concatenates z_phys(5) directly into a 13-dim context vector alongside
    T/t/x0(6) and feeds the whole thing into one shared MLP trunk. An A/B/C
    physics-conditioning ablation found this generator's CRPSS/coverage were
    statistically indistinguishable whether z_phys was real, zeroed, or
    shuffled to a different device — the network learned to route around
    z_phys and rely on x0 alone, because x0 is a strictly easier, lower-noise
    signal for minimising CRPS and nothing in the architecture forced use of
    the other branch. Root-caused via three ruled-out alternative
    explanations (RK4 gradient instability, outlier-dominated regression
    targets, encoder not being trained to encode a useful z_phys) — none of
    which fixed the ablation result — before concluding the generator's own
    architecture was the bottleneck.

    Fix, structural half: z_phys and (T, t, x0) are each first mapped through
    their OWN small encoder into an embedding of comparable width, and only
    THEN concatenated and passed to the shared trunk. This does not
    guarantee the trunk uses the z_phys embedding, but it removes the
    "cheapest path" of just learning near-zero input weights on 5 of 13
    raw-concatenated dimensions — the z_phys embedding now has to be
    actively suppressed via the whole z_phys_encoder subnetwork, not simply
    ignored by a few zeroed first-layer weights. Combined with
    z_phys_contrastive_loss (07_losses.py) at training time, which
    DIRECTLY penalises the model for producing an equally-good CRPS with a
    shuffled z_phys — i.e. optimises the exact quantity the A/B/C ablation
    measures, instead of leaving it as an unsupervised side effect.
    """
    STABLE_INDICES = STABLE_FEAT_INDICES   # [0, 1, 2, 3]
    N_STABLE       = N_STABLE_FEATURES     # 4
    LOG10_T_REF    = 0.35

    def __init__(
        self,
        noise_dim:           int   = STAGE4B_NOISE_DIM,
        hidden_dim:          int   = STAGE4B_HIDDEN_DIM,
        n_output:            int   = N_STABLE_FEATURES,
        latent_dim:          int   = cfg.LATENT_DIM,
        n_context_feat:      int   = cfg.FEATURE_DIM,
        z_embed_dim:         int   = 16,
        ctx_embed_dim:       int   = 16,
        log_scale_floor_init:float = DEFAULT_LOG_SCALE_FLOOR_INIT,
    ):
        super().__init__()
        self.noise_dim      = noise_dim
        self.n_features     = n_output
        self.latent_dim     = latent_dim
        self._n_ctx_feat    = n_context_feat
        self._leakage_indices = []

        # Separate encoders: z_phys cannot be shortcut around by a few
        # near-zero first-layer weights, since it now has a dedicated
        # nonlinear path with its own capacity that must be actively
        # suppressed (not just ignored) for the model to end up ignoring it.
        self.z_phys_encoder = nn.Sequential(
            nn.Linear(latent_dim, z_embed_dim),
            nn.GELU(),
            nn.LayerNorm(z_embed_dim),
        )
        # T_norm(1) + x0(n_context_feat) + log_t(1)
        raw_ctx_dim = 1 + n_context_feat + 1
        self.ctx_encoder = nn.Sequential(
            nn.Linear(raw_ctx_dim, ctx_embed_dim),
            nn.GELU(),
            nn.LayerNorm(ctx_embed_dim),
        )

        trunk_in_dim = noise_dim + z_embed_dim + ctx_embed_dim
        self.net = nn.Sequential(
            nn.Linear(trunk_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_output),
        )
        self.log_scale_floor = nn.Parameter(
            torch.full((n_output,), float(log_scale_floor_init))
        )
        for sub in (self.z_phys_encoder, self.ctx_encoder, self.net):
            for m in sub:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.15)
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Per-feature residual offset; see OFFSET_BOUND. Zero at init, so an
        # untrained generator reproduces the pre-offset behaviour exactly.
        self.offset_raw = nn.Parameter(torch.zeros(n_output))

    @property
    def offset(self) -> torch.Tensor:
        """Learned per-feature residual offset, bounded to +-OFFSET_BOUND."""
        return OFFSET_BOUND * torch.tanh(self.offset_raw)

    def _context_params(self, z_prefix_last, T_K, x0, log_t_suffix, noise_init=None):
        """Return (rho, sigma) tensors shaped (B, n_output), one per generated feature."""
        z_prefix_last = torch.nan_to_num(z_prefix_last, nan=0.5, posinf=0.0, neginf=0.0)
        x0            = torch.nan_to_num(x0,            nan=0.0, posinf=0.0, neginf=0.0)
        T_K           = torch.nan_to_num(T_K,           nan=0.0, posinf=0.0, neginf=0.0)
        log_t_suffix  = torch.nan_to_num(log_t_suffix,  nan=0.0, posinf=0.0, neginf=0.0)

        if z_prefix_last.dim() == 1:
            z_prefix_last = z_prefix_last.unsqueeze(0)
        B   = z_prefix_last.shape[0]
        dev = z_prefix_last.device

        T_norm = ((T_K.reshape(-1) - 300.0) / 25.0).reshape(B, 1)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        if x0.dim() > 2:
            x0 = x0.reshape(B, -1)
        x0_ctx = x0[:, :self._n_ctx_feat]

        log_t = log_t_suffix.reshape(B, -1)[:, :1]

        z_embed   = self.z_phys_encoder(z_prefix_last.reshape(B, -1))
        raw_ctx   = torch.cat([T_norm, x0_ctx.reshape(B, -1), log_t], dim=-1)
        ctx_embed = self.ctx_encoder(raw_ctx)

        if noise_init is None:
            noise_init = torch.zeros(B, self.noise_dim, device=dev, dtype=z_embed.dtype)
        else:
            noise_init = torch.nan_to_num(noise_init, nan=0.0)
            if noise_init.dim() == 1:
                noise_init = noise_init.unsqueeze(0)

        inp      = torch.cat([noise_init, z_embed, ctx_embed], dim=-1)
        out      = self.net(inp)
        rho      = torch.sigmoid(out[:, :self.n_features]) * 0.97
        sig_floor= torch.exp(self.log_scale_floor).unsqueeze(0).expand(B, -1)
        sigma    = sig_floor + F.softplus(out[:, self.n_features:])
        return rho, sigma

    def forward(self, z_prefix_last, T_K, x0, log_t_suffix, T_future=10, noise=None,
                times_future=None):
        """Continuous-time OU on the log10(1+t) axis (see AR1GuidedResidualGeneratorStable)."""
        B   = z_prefix_last.shape[0]
        dev = z_prefix_last.device
        rho, sigma = self._context_params(z_prefix_last, T_K, x0, log_t_suffix)

        rho_eff_per_step = None
        if times_future is not None and times_future.shape[1] >= 2:
            t = times_future.to(dev).clamp(min=0.0)
            log1pt = torch.log10(1.0 + t)
            delta_log10 = torch.zeros(B, T_future, device=dev)
            delta_log10[:, 0] = self.LOG10_T_REF
            if T_future > 1:
                delta_log10[:, 1:] = (
                    log1pt[:, 1:] - log1pt[:, :-1]
                ).clamp(min=0.01, max=2.0)
            expo = (delta_log10 / self.LOG10_T_REF).unsqueeze(-1)
            rho_eff_per_step = rho.unsqueeze(1) ** expo

        deltas = []
        if AR1_STATIONARY_INIT:
            # Seed from the process's stationary distribution N(0, sigma) so
            # step 0 has sd sigma rather than sigma*sqrt(1-rho^2). Starting at
            # exactly zero makes every early step too narrow, which matters
            # most for features whose observations sit early.
            d_prev = sigma * torch.randn(B, self.n_features, device=dev)
        else:
            d_prev = torch.zeros(B, self.n_features, device=dev)
        for t_idx in range(T_future):
            eps   = torch.randn(B, self.n_features, device=dev)
            rho_i = rho_eff_per_step[:, t_idx, :] if rho_eff_per_step is not None else rho
            sq    = torch.sqrt((1.0 - rho_i ** 2).clamp(min=1e-6))
            d_t   = rho_i * d_prev + sq * sigma * eps
            deltas.append(d_t)
            d_prev = d_t.detach()
        out = torch.stack(deltas, dim=1)          # (B, T_future, n_features)
        # Shift the whole zero-mean AR(1) path onto the residual's true centre.
        # Added AFTER the recursion so it does not feed back through rho and
        # inflate later steps.
        return out + self.offset.view(1, 1, -1)

    def sample_n(self, z_prefix_last, T_K, x0, log_t_suffix, n_samples, T_future=10,
                 times_future=None):
        return torch.stack([
            self.forward(z_prefix_last, T_K, x0, log_t_suffix, T_future=T_future,
                         times_future=times_future)
            for _ in range(n_samples)
        ], dim=0)   # (S, B, T_future, 4)


class AR1GuidedResidualGeneratorArrhenius(nn.Module):
    """Stage 4C generator variant with an Arrhenius-parameterised sigma.

    Motivation (2026-08-21): three attempts to make the network LEARN to use
    per-device z_phys (sensitivity loss, ranking loss, contrastive loss) all
    failed — even the contrastive loss, trained directly against the exact
    quantity later checked, only reached a 56% real-vs-shuffled win rate on
    its own training set (chance is 50%) and inverted on held-out test
    devices. The common failure mode: all three tried to recover a PER-DEVICE
    signal from z_phys, and per-device residual variance turns out to be
    dominated by outlier noise at this sample size (CV~5 raw, ~2 after log1p;
    203 devices split across 4 device types x 3 temperatures) — there simply
    isn't enough independent information per (type, temperature) cell to
    learn an individual multiplier reliably.

    This variant sidesteps that entirely by targeting a POPULATION-level
    physical relationship instead: residual variance is a stochastic process
    driven by the same thermally-activated defect kinetics as the
    deterministic degradation (trap occupancy fluctuations, SRH recombination
    noise), so it should itself follow an Arrhenius temperature dependence,

        sigma_f(T) = sigma_ref,f * exp(-Ea_sigma,f / kB * (1/T - 1/T_ref))

    A robustified per-temperature-bucket analysis of this dataset (MAD-based
    std of decoder residuals, computed independently per stable feature)
    confirmed this is physically supported: fitted Ea_sigma values are
    0.17-0.83 eV across the 4 stable features (Vth: 0.67 eV, cleanly
    monotone 275->300->325C) — the same order of magnitude as the trained
    ODE's own Ea_rev (~0.28-0.30 eV) / Ea_irrev (~0.43-0.77 eV). Unlike
    per-device z_phys signal, this only requires enough samples PER
    TEMPERATURE BUCKET (41-52 devices each here), which this dataset has.

    sigma_ref and Ea_sigma are hard-coded into the architecture as learnable
    parameters (8 total: one ref-scale + one activation-energy per stable
    feature) rather than left for an MLP to discover from noisy per-device
    residuals — the physical form is now structurally guaranteed rather than
    hoped-for via a loss term, closing the gap that made the previous three
    loss-based attempts fail. Context (z_phys, x0) still contributes a small,
    BOUNDED multiplicative correction around the Arrhenius baseline (not
    replacing it), so per-device information can still help within a
    physically-anchored envelope rather than being asked to explain
    everything on its own.

    rho keeps the same context-conditioned MLP form as
    AR1GuidedResidualGeneratorStable — only the temperature-dependence of
    sigma is being architecturally constrained here, since that was the
    dimension with an independently verified physical signal.
    """
    STABLE_INDICES = STABLE_FEAT_INDICES   # [0, 1, 2, 3]
    N_STABLE       = N_STABLE_FEATURES     # 4
    LOG10_T_REF    = 0.35

    def __init__(
        self,
        noise_dim:            int   = STAGE4B_NOISE_DIM,
        hidden_dim:            int   = STAGE4B_HIDDEN_DIM,
        n_output:              int   = N_STABLE_FEATURES,
        latent_dim:            int   = cfg.LATENT_DIM,
        n_context_feat:        int   = cfg.FEATURE_DIM,
        sigma_ref_init:        float = None,
        # Per-feature Ea_sigma initialisation. None (the default) builds a
        # length-n_output list from ARRHENIUS_TREND_EA_REF, falling back to
        # EA_SIGMA_DEFAULT for features with no validated reference value.
        # The four original entries were seeded from the robustified
        # per-temperature-bucket fit on this dataset (see docstring); training
        # can move them, this is only a starting point in the physically
        # plausible 0.1-1 eV range.
        ea_sigma_init:         tuple = None,
        correction_bound:      float = 0.5,   # max +-50% multiplicative deviation from Arrhenius baseline
    ):
        super().__init__()
        self.noise_dim      = noise_dim
        self.n_features     = n_output
        self.latent_dim     = latent_dim
        self._n_ctx_feat    = n_context_feat
        self._leakage_indices = []
        self.correction_bound = correction_bound

        context_dim = latent_dim + 1 + n_context_feat + 1
        in_dim      = noise_dim + context_dim

        # rho + a small bounded log-correction on sigma (not sigma itself)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_output),   # [rho_raw, sigma_correction_raw]
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.15)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Arrhenius sigma parameters — the physical constraint, hard-coded
        # into the architecture. log_sigma_ref ensures sigma_ref > 0 via exp;
        # Ea_sigma similarly via exp(log_Ea_sigma) (same convention as
        # PhysicsODE.Ea_rev/Ea_irrev in 02_physics_latent.py).
        # Per-feature init from the measured residual spreads. sigma_ref_init
        # is honoured when explicitly passed (tests, ablations); otherwise the
        # per-feature table is used, since one scalar across features whose
        # spreads differ 13-fold left every interval the same width.
        if sigma_ref_init is None:
            _names = [cfg.FEATURES[i] for i in STABLE_FEAT_INDICES[:n_output]]                      if len(STABLE_FEAT_INDICES) >= n_output else []
            _tbl = _robust_sigma_table() if SIGMA_REF_ROBUST else SIGMA_REF_BY_FEATURE
            _init = [_tbl.get(nm, SIGMA_REF_DEFAULT) for nm in _names]                     if _names else [SIGMA_REF_DEFAULT] * n_output
            sigma_ref_t = torch.tensor(_init[:n_output], dtype=torch.float32)
        else:
            sigma_ref_t = torch.full((n_output,), float(sigma_ref_init))
        assert sigma_ref_t.numel() == n_output, (sigma_ref_t.numel(), n_output)
        self.log_sigma_ref = nn.Parameter(torch.log(sigma_ref_t.clamp(min=1e-4)))
        # One Ea per generated feature. Previously a fixed 4-tuple sliced by
        # [:n_output], which SILENTLY produced a length-4 parameter when
        # n_output was larger -- log_sigma_ref (11) and log_Ea_sigma (4) then
        # broadcast-clashed only later, inside _arrhenius_sigma. Build it to
        # length n_output explicitly, taking any known reference Ea per
        # feature index and a neutral default for the rest.
        if ea_sigma_init is None:
            ea_list = [float(ARRHENIUS_TREND_EA_REF.get(i, EA_SIGMA_DEFAULT))
                       for i in range(n_output)]
        else:
            ea_list = [float(v) for v in ea_sigma_init]
            if len(ea_list) < n_output:
                ea_list += [EA_SIGMA_DEFAULT] * (n_output - len(ea_list))
            ea_list = ea_list[:n_output]
        ea_init_t = torch.tensor(ea_list, dtype=torch.float32)
        assert ea_init_t.numel() == n_output, (ea_init_t.numel(), n_output)
        self.log_Ea_sigma = nn.Parameter(torch.log(ea_init_t.clamp(min=0.05)))

        # Per-feature residual offset, tanh-bounded to +-OFFSET_BOUND. Starts
        # at zero so an untrained generator reproduces the previous behaviour
        # exactly, and the CRPS loss decides whether to move it.
        self.offset_raw = nn.Parameter(torch.zeros(n_output))

    @property
    def sigma_ref(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_ref)

    @property
    def Ea_sigma(self) -> torch.Tensor:
        return torch.exp(self.log_Ea_sigma)

    @property
    def offset(self) -> torch.Tensor:
        """Learned per-feature residual offset, bounded to +-OFFSET_BOUND."""
        return OFFSET_BOUND * torch.tanh(self.offset_raw)

    def _arrhenius_sigma(self, T_K: torch.Tensor) -> torch.Tensor:
        """sigma_ref * exp(-Ea_sigma/kB * (1/T - 1/T_ref)), shape (B, n_features)."""
        kb = cfg.KB_EV
        T_ref = cfg.T_REF_K
        T_K = T_K.reshape(-1, 1)
        inv_T_diff = 1.0 / T_K.clamp(min=1.0) - 1.0 / T_ref
        exponent = -self.Ea_sigma.unsqueeze(0) / kb * inv_T_diff
        exponent = torch.clamp(exponent, -15.0, 15.0)
        return self.sigma_ref.unsqueeze(0) * torch.exp(exponent)

    def _context_params(self, z_prefix_last, T_K, x0, log_t_suffix, noise_init=None):
        """Return (rho, sigma) tensors shaped (B, n_output), one per generated feature."""
        z_prefix_last = torch.nan_to_num(z_prefix_last, nan=0.5, posinf=0.0, neginf=0.0)
        x0            = torch.nan_to_num(x0,            nan=0.0, posinf=0.0, neginf=0.0)
        T_K           = torch.nan_to_num(T_K,           nan=0.0, posinf=0.0, neginf=0.0)
        log_t_suffix  = torch.nan_to_num(log_t_suffix,  nan=0.0, posinf=0.0, neginf=0.0)

        if z_prefix_last.dim() == 1:
            z_prefix_last = z_prefix_last.unsqueeze(0)
        B   = z_prefix_last.shape[0]
        dev = z_prefix_last.device

        T_norm = ((T_K.reshape(-1) - 300.0) / 25.0).reshape(B, 1)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        if x0.dim() > 2:
            x0 = x0.reshape(B, -1)
        x0_ctx = x0[:, :self._n_ctx_feat]

        log_t = log_t_suffix.reshape(B, -1)[:, :1]

        ctx = torch.cat([z_prefix_last.reshape(B, -1), T_norm, x0_ctx.reshape(B, -1), log_t], dim=-1)

        if noise_init is None:
            noise_init = torch.zeros(B, self.noise_dim, device=dev, dtype=ctx.dtype)
        else:
            noise_init = torch.nan_to_num(noise_init, nan=0.0)
            if noise_init.dim() == 1:
                noise_init = noise_init.unsqueeze(0)

        inp = torch.cat([noise_init, ctx], dim=-1)
        out = self.net(inp)
        rho = torch.sigmoid(out[:, :self.n_features]) * 0.97

        # Physical baseline (structurally guaranteed Arrhenius form) times a
        # small bounded multiplicative correction from context. tanh keeps
        # the correction in [-correction_bound, +correction_bound] so context
        # can nudge sigma but cannot override the physical temperature
        # scaling — at init (net[-1]=0) correction=0 and sigma is EXACTLY
        # the Arrhenius baseline.
        sigma_base = self._arrhenius_sigma(T_K.reshape(-1))   # (B, n_features)
        correction = self.correction_bound * torch.tanh(out[:, self.n_features:])
        if CORRECTION_ONE_SIDED:
            # Context may widen the interval but not narrow it. sigma_ref is
            # initialised from the measured residual spread, so narrowing it
            # can only under-cover; the freedom was being spent that way.
            correction = correction.clamp(min=0.0)
        sigma = sigma_base * (1.0 + correction)
        sigma = sigma.clamp(min=1e-4)
        return rho, sigma

    # Reference log10-time step (see AR1GuidedResidualGeneratorStable)
    def forward(self, z_prefix_last, T_K, x0, log_t_suffix, T_future=10, noise=None,
                times_future=None):
        B   = z_prefix_last.shape[0]
        dev = z_prefix_last.device
        rho, sigma = self._context_params(z_prefix_last, T_K, x0, log_t_suffix)

        rho_eff_per_step = None
        if times_future is not None and times_future.shape[1] >= 2:
            t = times_future.to(dev).clamp(min=0.0)
            log1pt = torch.log10(1.0 + t)
            delta_log10 = torch.zeros(B, T_future, device=dev)
            delta_log10[:, 0] = self.LOG10_T_REF
            if T_future > 1:
                delta_log10[:, 1:] = (
                    log1pt[:, 1:] - log1pt[:, :-1]
                ).clamp(min=0.01, max=2.0)
            expo = (delta_log10 / self.LOG10_T_REF).unsqueeze(-1)
            rho_eff_per_step = rho.unsqueeze(1) ** expo

        deltas = []
        if AR1_STATIONARY_INIT:
            # Seed from the process's stationary distribution N(0, sigma) so
            # step 0 has sd sigma rather than sigma*sqrt(1-rho^2). Starting at
            # exactly zero makes every early step too narrow, which matters
            # most for features whose observations sit early.
            d_prev = sigma * torch.randn(B, self.n_features, device=dev)
        else:
            d_prev = torch.zeros(B, self.n_features, device=dev)
        for t_idx in range(T_future):
            eps   = torch.randn(B, self.n_features, device=dev)
            rho_i = rho_eff_per_step[:, t_idx, :] if rho_eff_per_step is not None else rho
            sq    = torch.sqrt((1.0 - rho_i ** 2).clamp(min=1e-6))
            d_t   = rho_i * d_prev + sq * sigma * eps
            deltas.append(d_t)
            d_prev = d_t.detach()
        out = torch.stack(deltas, dim=1)          # (B, T_future, n_features)
        # Shift the whole zero-mean AR(1) path onto the residual's true centre.
        # Added AFTER the recursion so it does not feed back through rho and
        # inflate later steps.
        return out + self.offset.view(1, 1, -1)

    def sample_n(self, z_prefix_last, T_K, x0, log_t_suffix, n_samples, T_future=10,
                 times_future=None):
        return torch.stack([
            self.forward(z_prefix_last, T_K, x0, log_t_suffix, T_future=T_future,
                         times_future=times_future)
            for _ in range(n_samples)
        ], dim=0)   # (S, B, T_future, 4)


def _fit_ar1_targets(
    x_true,
    x_hat,
    prefix_len: int,
    feat_indices=None,
    device_center: bool = True,
    times_future=None,
    log10_t_ref: float = 0.35,
):
    """Compute per-device AR(1) rho and sigma targets.

    Args:
        device_center: Remove per-device mean from residuals before computing
            autocorrelation.  This isolates the within-device temporal dynamics
            from slow systematic drift, which is already captured by the physics
            backbone.  Strongly recommended for log-time generators.
        times_future: (B, T_future) actual times [h] for future steps.  When
            provided, computes log-time consistent rho_ref by fitting:
                log(rho_ref) = mean_i [log(corr_i) * log10_t_ref / delta_log10t_i]
            If None, uses the standard index-based estimate.
        log10_t_ref: reference log10(1+t) step for the OU parameterisation.
    """
    x_true = torch.nan_to_num(x_true, nan=0.0, posinf=0.0, neginf=0.0)
    x_hat  = torch.nan_to_num(x_hat,  nan=0.0, posinf=0.0, neginf=0.0)

    resid = x_true[:, prefix_len:, :] - x_hat[:, prefix_len:, :]
    if feat_indices is not None:
        resid = resid[:, :, feat_indices]
    if resid.shape[1] <= 1:
        B, _, Fdim = resid.shape
        return (torch.zeros(B, Fdim, device=resid.device),
                torch.ones(B, Fdim,  device=resid.device) * 1e-3)

    # ── Device-centering: remove per-device mean ─────────────────────────────
    # Eliminates slow systematic drift (already captured by physics backbone)
    # so ACF estimates reflect the within-device stochastic dynamics only.
    if device_center:
        resid = resid - resid.mean(dim=1, keepdim=True)

    # ── Standard per-device autocorrelation estimate ─────────────────────────
    r_prev = resid[:, :-1, :]   # (B, T-1, F)
    r_curr = resid[:, 1:,  :]   # (B, T-1, F)
    mu_prev = r_prev.mean(dim=1, keepdim=True)
    mu_curr = r_curr.mean(dim=1, keepdim=True)
    cov      = ((r_prev - mu_prev) * (r_curr - mu_curr)).mean(dim=1)  # (B, F)
    var_prev = ((r_prev - mu_prev) ** 2).mean(dim=1).clamp(min=1e-6)  # (B, F)
    rho_idx  = (cov / var_prev).clamp(1e-6, 0.97)   # positive; clamp to valid range

    # ── Log-time consistent rho_ref (if times provided) ─────────────────────
    if times_future is not None:
        # For each step i, get delta_log10(1+t_i)
        t = times_future.clamp(min=0.0).to(resid.device)  # (B, T_future)
        T_f = t.shape[1]
        if T_f > 1:
            log1pt = torch.log10(1.0 + t)              # (B, T_future)
            d_log  = (log1pt[:, 1:] - log1pt[:, :-1]).clamp(min=0.01, max=2.0)  # (B, T_f-1)

            # Per-step correlation estimate
            corr_step = (cov / var_prev).clamp(1e-6, 0.97)   # same as rho_idx (B, F)

            # Log-time rho_ref: average of log(rho_idx) / d_log * log10_t_ref
            # (using the batch-average d_log since rho_idx is already batch-aggregated)
            mean_d_log = d_log.mean()   # scalar; batch + time average
            log_rho_ref = torch.log(corr_step.clamp(min=1e-6)) * (log10_t_ref / mean_d_log)
            rho_logtime = torch.exp(log_rho_ref).clamp(1e-6, 0.97)  # (B, F)
            rho_target = rho_logtime
        else:
            rho_target = rho_idx
    else:
        rho_target = rho_idx

    sigma_target = resid.std(dim=1).clamp(min=1e-4)   # (B, F)
    return rho_target, sigma_target


def _whitened_innovation_diagnostics(deltas_s, rho_pred, sigma_pred, future_mask):
    """Compute whitened innovation statistics for one sample trajectory.

    innovation = (d_t - rho * d_{t-1}) / (sigma * sqrt(1 - rho^2))
    Should be ~ N(0,1) if the AR(1) model is correct.

    Returns dict with mean, std, |kurtosis| across all valid innovations.
    """
    S, B, T_f, F = deltas_s.shape
    with torch.no_grad():
        sq    = torch.sqrt((1.0 - rho_pred ** 2).clamp(min=1e-6))  # (B, F)
        scale = (sigma_pred * sq).unsqueeze(0).unsqueeze(2).clamp(min=1e-8)  # (1, B, 1, F)
        innov = (deltas_s[:, :, 1:, :] - rho_pred.unsqueeze(0).unsqueeze(2) * deltas_s[:, :, :-1, :])
        innov = innov / scale                                        # (S, B, T-1, F)
        fm    = future_mask[:, 1:].float().unsqueeze(0).unsqueeze(-1).expand_as(innov).bool()
        vals  = innov[fm]                                            # (N_valid,)
    if vals.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "abs_kurt": float("nan")}
    mean = float(vals.mean().item())
    std  = float(vals.std().item())
    kurt = float(((vals - vals.mean()) ** 4).mean().item() / max(std ** 4, 1e-8)) - 3
    return {"mean": mean, "std": std, "abs_kurt": abs(kurt)}


def train_stage4b(
    model,
    generator: AR1GuidedResidualGenerator,
    train_dl,
    val_dl,
    device: torch.device,
    mods: Optional[dict] = None,
    n_train_samples: int = STAGE4B_N_TRAIN_SAMPLES,
    n_val_samples: int = STAGE4B_N_VAL_SAMPLES,
    epochs: int = STAGE4B_EPOCHS,
    lr: float = STAGE4B_LR,
    stable_feat_indices: Optional[list] = None,  # None = all 6 features; [0,1,2,3] = Stage 4C
    lambda_acf: float = LAMBDA_ACF,              # ACF matching weight (0 = off)
    patience: int = STAGE4B_PATIENCE,
    lambda_crps: float = LAMBDA_CRPS,
    lambda_ar1: float = LAMBDA_AR1,
    lambda_var: float = LAMBDA_VAR,
    lambda_pinball: float = LAMBDA_PINBALL,
    lambda_scale: float = LAMBDA_SCALE,
    lambda_calib: float = LAMBDA_CALIB,
    lambda_phys_sens: float = LAMBDA_PHYS_SENS,
    lambda_zphys_contrast: float = LAMBDA_ZPHYS_CONTRAST,
    lambda_arrhenius_trend: float = LAMBDA_ARRHENIUS_TREND,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    output_dir: Optional[str] = None,
):
    _forward = mods["train"]._forward if mods else None
    assert _forward is not None, "mods dict with 'train' module is required"
    if output_dir is None:
        output_dir = cfg.CHECKPOINT_DIR
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "stage4b_best.pt")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    generator = generator.to(device)
    generator.train()

    opt = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(1, patience // 2), factor=0.5)
    prefix_len = cfg.STAGE3_PREFIX_LEN

    log.info("Pre-caching training trajectories for Stage 4B...")
    train_cache = _cache_trajectories(model, train_dl, device, _forward, prefix_len)
    log.info("Pre-caching validation trajectories for Stage 4B...")
    val_cache = _cache_trajectories(model, val_dl, device, _forward, prefix_len)

    best_val_crps = float("inf")
    best_epoch = 0
    no_improve = 0
    history = {"train_crps": [], "val_crps": []}

    log.info("=" * 60)
    log.info("=== Stage 4B: AR(1)-guided residual generator ===")
    log.info("  Epochs=%d  n_train_samples=%d  lr=%.2e", epochs, n_train_samples, lr)
    log.info("  n_val_samples=%d  sigma_min=%.4f", n_val_samples, sigma_min)
    log.info("  Loss weights: CRPS(λ=%.2f) AR1(λ=%.2f) Var(λ=%.2f) Pin(λ=%.2f) Scale(λ=%.2f) Calib(λ=%.2f) PhysSens(λ=%.2f) ZContrast(λ=%.2f) ArrTrend(λ=%.2f)",
             lambda_crps, lambda_ar1, lambda_var, lambda_pinball, lambda_scale, lambda_calib, lambda_phys_sens, lambda_zphys_contrast, lambda_arrhenius_trend)
    log.info("=" * 60)

    for epoch in range(1, epochs + 1):
        generator.train()
        t0 = time.time()
        train_crps_total  = 0.0
        train_ar1_total   = 0.0
        train_var_total   = 0.0
        train_pin_total   = 0.0
        train_scale_total = 0.0
        train_acf_total   = 0.0
        train_calib_total = 0.0
        train_sens_total  = 0.0
        train_contrast_total = 0.0
        train_arrtrend_total = 0.0
        n_train_batches   = 0

        for rec in train_cache:
            z_pfx = rec["z_pfx"].to(device)
            x_hat = rec["x_hat"].to(device)
            x_true = rec["x_true"].to(device)
            mask = rec["mask"].to(device)
            T_K = rec["T_K"].to(device)
            log_t = rec["log_t"].to(device)
            x0 = rec["x0"].to(device)
            plen = rec["plen"]
            T_len = rec["T_len"]
            T_future = T_len - plen
            if T_future <= 0:
                continue
            # Log-time future schedule (if cached)
            times_future_t = None
            if "times" in rec:
                times_future_t = rec["times"][:, plen:].to(device)   # (B, T_future)

            deltas = generator.sample_n(z_pfx, T_K, x0, log_t, n_train_samples, T_future=T_future,
                                        times_future=times_future_t)

            # ── Build predictions for loss computation ───────────────────────
            # If stable_feat_indices is set (Stage 4C), deltas covers only those
            # features; we build predictions in that reduced feature space.
            if stable_feat_indices is not None:
                sfx = torch.tensor(stable_feat_indices, device=device)
                x_hat_future = x_hat[:, plen:, :][:, :, sfx].unsqueeze(0)
                x_pred_future = x_hat_future + deltas          # (S, B, T_f, |sfx|)
                x_prefix_exp  = x_hat[:, :plen, :][:, :, sfx].unsqueeze(0).expand(n_train_samples, -1, -1, -1)
                x_pred_full   = torch.cat([x_prefix_exp, x_pred_future], dim=2)
                x_true_loss   = x_true[:, :, sfx]
            else:
                x_hat_future = x_hat[:, plen:, :].detach().unsqueeze(0)
                x_pred_future = x_hat_future + deltas
                x_prefix_exp = x_hat[:, :plen, :].detach().unsqueeze(0).expand(n_train_samples, -1, -1, -1)
                x_pred_full = torch.cat([x_prefix_exp, x_pred_future], dim=2)
                x_true_loss = x_true

            crps = crps_mc_loss(x_pred_full, x_true_loss, mask, prefix_len=plen)
            rho_pred, sigma_pred = generator._context_params(z_pfx, T_K, x0, log_t)
            # Use device-centered, log-time consistent rho targets
            rho_target, sigma_target = _fit_ar1_targets(
                x_true, x_hat, plen,
                feat_indices=stable_feat_indices,
                device_center=True,
                times_future=times_future_t,
            )
            rho_target = rho_target.to(device)
            sigma_target = sigma_target.to(device)

            rho_loss = ((rho_pred - rho_target) ** 2).mean()
            sigma_loss = _temp_equalized_var_loss(sigma_pred, sigma_target, T_K)
            future_true = torch.nan_to_num(x_true_loss[:, plen:, :], nan=0.0, posinf=0.0, neginf=0.0)
            pinball = pinball_loss(x_pred_future, future_true, mask[:, plen:], prefix_len=0)
            scale_reg = torch.relu(float(sigma_min) - sigma_pred).mean()

            # ACF matching loss (optional, activated when lambda_acf > 0)
            acf_l = torch.tensor(0.0, device=device)
            if lambda_acf > 0:
                _sfx = stable_feat_indices if stable_feat_indices is not None else list(range(x_true.shape[-1]))
                acf_l = acf_matching_loss(
                    deltas, x_true, x_hat, mask, plen,
                    stable_feat_indices=_sfx,
                )

            # Coverage-calibration loss (optional, activated when lambda_calib > 0)
            calib_l = torch.tensor(0.0, device=device)
            if lambda_calib > 0:
                calib_l = coverage_calibration_loss(x_pred_full, x_true_loss, mask, prefix_len=plen)

            # Physics-latent sensitivity loss (optional, activated when lambda_phys_sens > 0)
            sens_l = torch.tensor(0.0, device=device)
            if lambda_phys_sens > 0:
                sens_l = physics_sensitivity_loss(generator, z_pfx, T_K, x0, log_t)

            # z_phys contrastive loss (optional, activated when lambda_zphys_contrast > 0):
            # real z_phys must give a strictly better CRPS than a shuffled one.
            contrast_l = torch.tensor(0.0, device=device)
            if lambda_zphys_contrast > 0:
                future_mask_bool = mask[:, plen:].bool()
                x_hat_future_raw = (
                    x_hat[:, plen:, :][:, :, sfx] if stable_feat_indices is not None
                    else x_hat[:, plen:, :]
                ).detach()
                contrast_l = z_phys_contrastive_loss(
                    generator, z_pfx, T_K, x0, log_t,
                    x_true_future=future_true, future_mask=future_mask_bool,
                    x_hat_future=x_hat_future_raw, n_samples=n_train_samples,
                    times_future=times_future_t,
                )

            # Arrhenius sigma-trend loss (optional, activated when lambda_arrhenius_trend > 0):
            # batch-averaged sigma per temperature group should follow a
            # physically-plausible Arrhenius scaling — soft, population-level,
            # does not constrain any individual device's sigma.
            arrtrend_l = torch.tensor(0.0, device=device)
            if lambda_arrhenius_trend > 0:
                arrtrend_l = arrhenius_trend_loss(sigma_pred, T_K)

            loss = (
                lambda_crps * crps
                + lambda_ar1 * rho_loss
                + lambda_var * sigma_loss
                + lambda_pinball * pinball
                + lambda_scale * scale_reg
                + lambda_acf  * acf_l
                + lambda_calib * calib_l
                + lambda_phys_sens * sens_l
                + lambda_zphys_contrast * contrast_l
                + lambda_arrhenius_trend * arrtrend_l
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt.step()

            train_crps_total  += crps.item()
            train_ar1_total   += rho_loss.item()
            train_var_total   += sigma_loss.item()
            train_pin_total   += pinball.item()
            train_scale_total += scale_reg.item()
            train_acf_total   += acf_l.item()
            train_calib_total += calib_l.item()
            train_sens_total  += sens_l.item()
            train_contrast_total += contrast_l.item()
            train_arrtrend_total += arrtrend_l.item()
            n_train_batches += 1

        if n_train_batches == 0:
            log.warning("Epoch %d: no valid batches", epoch)
            continue

        train_crps_avg = train_crps_total / n_train_batches

        generator.eval()
        val_crps_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for rec in val_cache:
                z_pfx = rec["z_pfx"].to(device)
                x_hat = rec["x_hat"].to(device)
                x_true = rec["x_true"].to(device)
                mask = rec["mask"].to(device)
                T_K = rec["T_K"].to(device)
                log_t = rec["log_t"].to(device)
                x0 = rec["x0"].to(device)
                plen = rec["plen"]
                T_len = rec["T_len"]
                T_future = T_len - plen
                if T_future <= 0:
                    continue
                times_future_v = rec["times"][:, plen:].to(device) if "times" in rec else None
                deltas_v = generator.sample_n(z_pfx, T_K, x0, log_t, n_val_samples, T_future=T_future,
                                              times_future=times_future_v)
                if stable_feat_indices is not None:
                    sfx_v = torch.tensor(stable_feat_indices, device=device)
                    x_hat_future = x_hat[:, plen:, :][:, :, sfx_v].unsqueeze(0)
                    x_pred_future = x_hat_future + deltas_v
                    x_prefix_exp = x_hat[:, :plen, :][:, :, sfx_v].unsqueeze(0).expand(n_val_samples, -1, -1, -1)
                    x_pred_v = torch.cat([x_prefix_exp, x_pred_future], dim=2)
                    val_crps = crps_mc_loss(x_pred_v, x_true[:, :, sfx_v], mask, prefix_len=plen)
                else:
                    x_hat_future = x_hat[:, plen:, :].unsqueeze(0)
                    x_pred_future = x_hat_future + deltas_v
                    x_prefix_exp = x_hat[:, :plen, :].unsqueeze(0).expand(n_val_samples, -1, -1, -1)
                    x_pred_v = torch.cat([x_prefix_exp, x_pred_future], dim=2)
                    val_crps = crps_mc_loss(x_pred_v, x_true, mask, prefix_len=plen)
                val_crps_total += val_crps.item()
                n_val_batches += 1

        val_crps_avg = val_crps_total / max(n_val_batches, 1)
        sched.step(val_crps_avg)

        # Whitened innovation diagnostics (once per epoch on first val batch)
        innov_diag = {"mean": float("nan"), "std": float("nan"), "abs_kurt": float("nan")}
        if val_cache:
            _rec0 = val_cache[0]
            with torch.no_grad():
                _rp, _sp = generator._context_params(
                    _rec0["z_pfx"].to(device), _rec0["T_K"].to(device),
                    _rec0["x0"].to(device), _rec0["log_t"].to(device)
                )
                _tfut = _rec0["times"][:, _rec0["plen"]:].to(device) if "times" in _rec0 else None
                _dl   = generator.sample_n(
                    _rec0["z_pfx"].to(device), _rec0["T_K"].to(device),
                    _rec0["x0"].to(device), _rec0["log_t"].to(device),
                    n_samples=4, T_future=_rec0["T_len"] - _rec0["plen"],
                    times_future=_tfut,
                )
                innov_diag = _whitened_innovation_diagnostics(
                    _dl, _rp, _sp,
                    _rec0["mask"][:, _rec0["plen"]:].to(device).bool(),
                )

        elapsed = time.time() - t0
        log.info(
            "Epoch %3d/%d | train_CRPS=%.4f  val_CRPS=%.4f | AR1=%.4f  Var=%.4f  ACF=%.4f  Calib=%.4f  Sens=%.4f  ZContrast=%.4f  ArrTrend=%.4f"
            " | innov(mu=%.3f,std=%.3f,|k|=%.2f) | %.0fs",
            epoch, epochs, train_crps_avg, val_crps_avg,
            train_ar1_total   / n_train_batches,
            train_var_total   / n_train_batches,
            train_acf_total   / n_train_batches,
            train_calib_total / n_train_batches,
            train_sens_total  / n_train_batches,
            train_contrast_total / n_train_batches,
            train_arrtrend_total / n_train_batches,
            innov_diag["mean"], innov_diag["std"], innov_diag.get("abs_kurt", float("nan")),
            elapsed,
        )

        history["train_crps"].append(train_crps_avg)
        history["val_crps"].append(val_crps_avg)

        if val_crps_avg < best_val_crps:
            best_val_crps = val_crps_avg
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "val_crps": best_val_crps,
                "state_dict": generator.state_dict(),
            }, ckpt_path)
            log.info("  ✓ Saved best Stage 4B checkpoint (val_CRPS=%.4f) -> %s", best_val_crps, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  Early stopping at epoch %d", epoch)
                break

    log.info("Stage 4B training complete. Best epoch=%d val_CRPS=%.4f", best_epoch, best_val_crps)
    return {"best_val_crps": best_val_crps, "best_epoch": best_epoch, "history": history}


def evaluate_stage4b(
    model,
    generator: AR1GuidedResidualGenerator,
    test_dl,
    device: torch.device,
    mods: Optional[dict] = None,
    n_eval_samples: int = 100,
    prefix_len: Optional[int] = None,
    output_dir: Optional[str] = None,
):
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
            T_K = batch["T_K"].to(device)
            times = batch["times_h"].to(device)
            mask = batch["mask"].to(device)
            x0 = x_raw[:, 0, :]
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
        log.warning("No valid test batches for Stage 4B evaluation")
        return {}

    x_true_np = np.concatenate(all_x_true, axis=0)
    x_mean_np = np.concatenate(all_x_mean, axis=0)
    T_K_np = np.concatenate(all_T_K)
    times_np = np.concatenate(all_times, axis=0)
    mask_np = np.concatenate(all_mask, axis=0)
    N, T_len, F = x_true_np.shape

    eval_cache = _cache_trajectories(model, test_dl, device, _forward, prefix_len)
    samples_4b = np.full((n_eval_samples, N, T_len, F), np.nan)
    n_placed = 0
    with torch.no_grad():
        for rec in eval_cache:
            z_pfx = rec["z_pfx"].to(device)
            x_hat_r = rec["x_hat"].to(device)
            T_K = rec["T_K"].to(device)
            log_t = rec["log_t"].to(device)
            x0 = rec["x0"].to(device)
            plen_r = rec["plen"]
            b_end = n_placed + x_hat_r.shape[0]
            if b_end > N:
                b_end = N
            deltas = generator.sample_n(z_pfx, T_K, x0, log_t, n_eval_samples, T_future=max(0, T_len - plen_r))
            x_hat_np = x_hat_r.cpu().numpy()
            _stable = getattr(generator, 'STABLE_INDICES', None)
            for s in range(n_eval_samples):
                x_s = x_hat_np[: b_end - n_placed].copy()
                d_s = deltas[s].cpu().numpy()[: b_end - n_placed, :, :]
                if _stable is not None:
                    # Stable-only generator: apply residuals to stable columns only
                    for fi_loc, fi_glob in enumerate(_stable):
                        x_s[:, plen_r:, fi_glob] += d_s[:, :, fi_loc]
                else:
                    x_s[:, plen_r:, :] += d_s
                samples_4b[s, n_placed:b_end, :T_len, :] = x_s
            n_placed = b_end
            if n_placed >= N:
                break

    dummy = StochasticResidualModel()
    metrics_4b = dummy.compute_metrics(
        x_true_np, samples_4b, mask_np,
        T_K=T_K_np, prefix_len=prefix_len, x_mean=x_mean_np, times_h=times_np,
    )

    # "_overall" fields average across all 6 features, but IDLeak/IGLeak are
    # NOT generated by this model (zero-variance point copies of x_hat, see
    # the `_stable is not None` branch above) — their coverage is 0 by
    # construction and drags the blended average down in a way that does not
    # reflect this generator's actual calibration. Report a stable-features
    # -only summary alongside the raw (misleading-if-read-alone) "_overall".
    _stable = getattr(generator, 'STABLE_INDICES', None)
    if _stable is not None:
        stable_names = [cfg.FEATURES[i] for i in _stable]
        for key in ("coverage_50", "coverage_80", "coverage_90", "crps_by_feature", "crpss"):
            per_feat = metrics_4b.get(key, {})
            vals = [per_feat[n] for n in stable_names if n in per_feat and np.isfinite(per_feat[n])]
            metrics_4b[f"{key}_stable_overall"] = float(np.mean(vals)) if vals else float("nan")

    log.info(
        "Stage 4B Results: CRPS=%.4f  CRPSS=%.3f  Cov90(6-feat,incl.leakage)=%.2f  "
        "Cov90(4-feat,generated only)=%.2f  MACE=%.4f",
        metrics_4b.get("crps_overall", float("nan")),
        metrics_4b.get("crpss_overall", float("nan")),
        metrics_4b.get("coverage_90_overall", float("nan")),
        metrics_4b.get("coverage_90_stable_overall", float("nan")),
        metrics_4b.get("reliability_mace", float("nan")),
    )

    out_path = os.path.join(output_dir, "evaluation_results_stage4b.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump({"stage4b_metrics": metrics_4b, "n_eval_samples": n_eval_samples}, fh)
    log.info("Stage 4B evaluation saved → %s", out_path)
    return {"stage4b_metrics": metrics_4b}


def build_smoke_test_dataset(n_devices: int = 2, seq_len: int = 8):
    x = np.linspace(0.0, 1.0, seq_len, endpoint=True).astype(np.float32)
    x = np.stack([x, 0.5 + 0.1 * np.arange(seq_len), 0.2 + 0.05 * np.arange(seq_len), np.linspace(0.8, 0.4, seq_len), np.linspace(0.1, 0.2, seq_len), np.linspace(0.05, 0.1, seq_len)], axis=1)
    x = np.repeat(x[None, :, :], n_devices, axis=0)
    x = np.tile(x, (1, 1, 1))
    x = x + np.array([0.0, 0.01, -0.01, 0.0, 0.0, 0.0])[None, None, :]

    feature_mask = np.ones((n_devices, seq_len, cfg.FEATURE_DIM), dtype=bool)
    mask = np.ones((n_devices, seq_len), dtype=bool)
    times_h = np.array([np.array([0, 1, 2, 5, 10, 20, 50, 100][:seq_len], dtype=np.float32) for _ in range(n_devices)], dtype=np.float32)
    T_K = np.full((n_devices,), 573.15 + 25.0, dtype=np.float32)
    x0_static = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.05, 0.02], dtype=np.float32), (n_devices, 1))

    dataset = {
        "x": x.astype(np.float32),
        "feature_mask": feature_mask.astype(bool),
        "mask": mask.astype(bool),
        "times_h": times_h.astype(np.float32),
        "T_K": T_K.astype(np.float32),
        "x0_normalized": x0_static.astype(np.float32),
        "device_ids": [f"smoke_{i}" for i in range(n_devices)],
        "split": {"train": [0], "val": [1], "test": [1]},
        "leakage_floor": {"IDLeak": 1e-9, "IGLeak": 1e-9},
    }
    return dataset


def _parse_args():
    p = argparse.ArgumentParser(description="Stage 4B AR(1)-guided residual generator")
    p.add_argument("--epochs", type=int, default=STAGE4B_EPOCHS)
    p.add_argument("--n-train-samples", type=int, default=STAGE4B_N_TRAIN_SAMPLES)
    p.add_argument("--n-val-samples", type=int, default=STAGE4B_N_VAL_SAMPLES)
    p.add_argument("--n-eval-samples", type=int, default=100)
    p.add_argument("--lr", type=float, default=STAGE4B_LR)
    p.add_argument("--hidden-dim", type=int, default=STAGE4B_HIDDEN_DIM)
    p.add_argument("--noise-dim", type=int, default=STAGE4B_NOISE_DIM)
    p.add_argument("--lambda-crps", type=float, default=LAMBDA_CRPS)
    p.add_argument("--lambda-ar1", type=float, default=LAMBDA_AR1)
    p.add_argument("--lambda-var", type=float, default=LAMBDA_VAR)
    p.add_argument("--lambda-pinball", type=float, default=LAMBDA_PINBALL)
    p.add_argument("--lambda-scale", type=float, default=LAMBDA_SCALE)
    p.add_argument("--lambda-calib", type=float, default=LAMBDA_CALIB,
        help="Weight for direct coverage-calibration loss (0=off). Targets "
             "50/80/90%% empirical coverage matching nominal levels.")
    p.add_argument("--lambda-phys-sens", type=float, default=LAMBDA_PHYS_SENS,
        help="Weight for physics-latent sensitivity loss (0=off). Penalises "
             "the generator for predicting (rho,sigma) insensitive to z_phys.")
    p.add_argument("--lambda-zphys-contrast", type=float, default=LAMBDA_ZPHYS_CONTRAST,
        help="Weight for z_phys contrastive loss (0=off). Penalises the "
             "generator unless real z_phys gives strictly better CRPS than "
             "a shuffled (wrong-device) z_phys.")
    p.add_argument("--lambda-arrhenius-trend", type=float, default=LAMBDA_ARRHENIUS_TREND,
        help="Weight for the Arrhenius sigma-trend loss (0=off). Soft, "
             "population-level constraint: batch-averaged sigma per "
             "temperature group should follow a physically-plausible "
             "Arrhenius scaling, without pinning any individual device's "
             "sigma (use with AR1GuidedResidualGeneratorStable / "
             "--stable-only, not --arrhenius-sigma).")
    p.add_argument("--sigma-min", type=float, default=DEFAULT_SIGMA_MIN)
    p.add_argument("--log-scale-floor-init", type=float, default=DEFAULT_LOG_SCALE_FLOOR_INIT)
    p.add_argument("--stage3-ckpt", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true", help="Run a lightweight smoke test without the full dataset")
    p.add_argument("--stable-only", action="store_true",
        help="Stage 4C mode: train AR1GuidedResidualGeneratorStable on stable features only.")
    p.add_argument("--phys-gated", action="store_true",
        help="Use AR1GuidedResidualGeneratorPhysGated (separate z_phys/context "
             "encoders) instead of AR1GuidedResidualGeneratorStable. Implies "
             "--stable-only. Recommended together with --lambda-zphys-contrast.")
    p.add_argument("--arrhenius-sigma", action="store_true",
        help="Use AR1GuidedResidualGeneratorArrhenius: sigma(T) is "
             "structurally constrained to an Arrhenius temperature "
             "dependence (learnable Ea_sigma/sigma_ref per feature) instead "
             "of being freely predicted by the MLP. Implies --stable-only. "
             "Population-level physical constraint (validated on this "
             "dataset), unlike the per-device z_phys approaches which did "
             "not generalise to held-out devices.")
    p.add_argument("--lambda-acf", type=float, default=LAMBDA_ACF,
        help="Weight for ACF matching loss (0=off). Use e.g. 0.10 for Stage 4C-ACF variant.")
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)
    set_global_seed(args.seed)
    log.info("Using random seed: %d", args.seed)

    if args.smoke_test:
        dataset = build_smoke_test_dataset()
        out_path = os.path.join(args.output_dir or cfg.OUTPUT_PATH, "smoke_test_dataset.pkl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as fh:
            pickle.dump(dataset, fh)
        log.info("Smoke-test dataset written to %s", out_path)

        generator = AR1GuidedResidualGenerator().to(device)
        dummy = torch.randn(2, cfg.STAGE3_PREFIX_LEN, cfg.FEATURE_DIM, device=device)
        with torch.no_grad():
            sample = generator.sample_n(
                torch.randn(2, cfg.LATENT_DIM, device=device),
                torch.tensor([573.15 + 25.0, 573.15 + 25.0], device=device),
                dummy,
                torch.ones(2, cfg.STAGE3_PREFIX_LEN, device=device),
                n_samples=2,
                T_future=4,
            )
        metrics = {"smoke_sample_shape": list(sample.shape), "smoke_ok": True}
        metrics_path = os.path.join(os.path.dirname(out_path), "stage4b_smoke_metrics.pkl")
        with open(metrics_path, "wb") as fh:
            pickle.dump(metrics, fh)
        log.info("Smoke-test metrics written to %s", metrics_path)
        return

    output_dir = args.output_dir or cfg.OUTPUT_PATH
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")

    log.info("Loading pipeline modules for Stage 4B...")
    mods = _load_all()
    train_mod = mods["train"]

    log.info("Loading dataset...")
    prep_path = cfg.PROCESSED_DATA_PATH
    if not os.path.exists(prep_path):
        log.warning("Processed dataset not found at %s; falling back to synthetic smoke-test dataset", prep_path)
        dataset = build_smoke_test_dataset()
        out_path = os.path.join(output_dir, "smoke_test_dataset.pkl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as fh:
            pickle.dump(dataset, fh)
        log.info("Wrote fallback dataset to %s", out_path)
    else:
        with open(prep_path, "rb") as fh:
            dataset = pickle.load(fh)

    from torch.utils.data import DataLoader
    split = dataset["split"]
    all_idx = list(range(len(dataset["device_ids"])))
    train_idx = split.get("train", all_idx[: int(0.7 * len(all_idx))])
    val_idx = split.get("val", all_idx[int(0.7 * len(all_idx)) :])
    test_idx = split.get("test", val_idx)

    train_ds = train_mod.DeviceDegradationDataset(dataset, train_idx)
    val_ds = train_mod.DeviceDegradationDataset(dataset, val_idx)
    test_ds = train_mod.DeviceDegradationDataset(dataset, test_idx)
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=train_mod.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=train_mod.collate_fn)
    test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=train_mod.collate_fn)

    log.info("Building PI-TimeGAN backbone...")
    model = _build_model(mods).to(device)

    stage3_ckpt = args.stage3_ckpt or os.path.join(ckpt_dir, "stage3_best.pt")
    if not os.path.exists(stage3_ckpt):
        log.warning("Stage 3 checkpoint not found at %s; using a randomly initialized backbone for smoke execution", stage3_ckpt)
        model = model.eval()
    else:
        ckpt = torch.load(stage3_ckpt, map_location=device)
        model_state = ckpt.get("model_state", ckpt.get("model_state_dict", None))
        if model_state is None:
            log.error("Checkpoint %s has no model_state", stage3_ckpt)
            sys.exit(1)
        model_state_clean = {k: v for k, v in model_state.items() if k != "decoder.mask"}
        model.load_state_dict(model_state_clean, strict=False)
        log.info("Loaded Stage 3 checkpoint: %s", stage3_ckpt)

    is_stable = getattr(args, "stable_only", False)
    is_phys_gated = getattr(args, "phys_gated", False)
    is_arrhenius = getattr(args, "arrhenius_sigma", False)
    if is_arrhenius:
        log.info("Stage 4C mode: Arrhenius-sigma generator (physically-constrained sigma(T)), stable features %s",
                 STABLE_FEAT_INDICES)
        generator = AR1GuidedResidualGeneratorArrhenius(
            noise_dim=args.noise_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        stable_fi  = STABLE_FEAT_INDICES
    elif is_phys_gated:
        log.info("Stage 4C mode: phys-gated generator (separate z_phys/context encoders), stable features %s",
                 STABLE_FEAT_INDICES)
        generator = AR1GuidedResidualGeneratorPhysGated(
            noise_dim=args.noise_dim,
            hidden_dim=args.hidden_dim,
            log_scale_floor_init=args.log_scale_floor_init,
        ).to(device)
        stable_fi  = STABLE_FEAT_INDICES
    elif is_stable:
        log.info("Stage 4C mode: generating only stable features %s", STABLE_FEAT_INDICES)
        generator = AR1GuidedResidualGeneratorStable(
            noise_dim=args.noise_dim,
            hidden_dim=args.hidden_dim,
            log_scale_floor_init=args.log_scale_floor_init,
        ).to(device)
        stable_fi  = STABLE_FEAT_INDICES
    else:
        generator = AR1GuidedResidualGenerator(
            noise_dim=args.noise_dim,
            hidden_dim=args.hidden_dim,
            log_scale_floor_init=args.log_scale_floor_init,
        ).to(device)
        stable_fi  = None
    # train_stage4b always saves to "stage4b_best.pt"; use that as the canonical name
    ckpt_label = "stage4b_best.pt"

    stage4b_ckpt = os.path.join(ckpt_dir, ckpt_label)
    if args.skip_train and os.path.exists(stage4b_ckpt):
        ckpt4b = torch.load(stage4b_ckpt, map_location=device)
        generator.load_state_dict(ckpt4b["state_dict"])
        log.info("Loaded existing checkpoint: %s", stage4b_ckpt)
    else:
        train_stage4b(
            model, generator, train_dl, val_dl, device,
            mods=mods,
            n_train_samples=args.n_train_samples,
            n_val_samples=args.n_val_samples,
            epochs=args.epochs,
            lr=args.lr,
            lambda_crps=args.lambda_crps,
            lambda_ar1=args.lambda_ar1,
            lambda_var=args.lambda_var,
            lambda_pinball=args.lambda_pinball,
            lambda_scale=args.lambda_scale,
            lambda_acf=getattr(args, 'lambda_acf', LAMBDA_ACF),
            lambda_calib=getattr(args, 'lambda_calib', LAMBDA_CALIB),
            lambda_phys_sens=getattr(args, 'lambda_phys_sens', LAMBDA_PHYS_SENS),
            lambda_zphys_contrast=getattr(args, 'lambda_zphys_contrast', LAMBDA_ZPHYS_CONTRAST),
            lambda_arrhenius_trend=getattr(args, 'lambda_arrhenius_trend', LAMBDA_ARRHENIUS_TREND),
            sigma_min=args.sigma_min,
            output_dir=ckpt_dir,
            stable_feat_indices=stable_fi,
        )
        ckpt4b = torch.load(stage4b_ckpt, map_location=device)
        generator.load_state_dict(ckpt4b["state_dict"])

    evaluate_stage4b(
        model, generator, test_dl, device,
        mods=mods,
        n_eval_samples=args.n_eval_samples,
        output_dir=results_dir,
    )


if __name__ == "__main__":
    main()
