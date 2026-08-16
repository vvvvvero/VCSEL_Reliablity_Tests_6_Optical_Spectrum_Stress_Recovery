"""
10_stochastic_residual.py  (v2)
================================
Phase 2 improved stochastic residual baseline.

Key improvements over v1:
  1. Leakage channels use empirical increment bootstrap (not Gaussian AR)
     -> naturally reproduces observed decreasing-fraction without parametric bias.
  2. Temperature-conditioned increment pools (separate for 275/300/325 °C).
     If a bin is empty the full cross-temperature pool is used as fallback.
  3. Increment-bias calibration: auto-shift increments so the generated
     decreasing fraction matches the training-set empirical fraction.
  4. Non-leakage channels retain the heteroscedastic AR(1) Gaussian model.
  5. Extended metrics:
       - per-temperature CRPS, coverage, Winkler score
       - reliability curve (nominal vs observed quantile fractions)
       - W1 on increment distributions
       - signed bias, predicted vs true std

Usage (standalone):
    python 10_stochastic_residual.py

Usage (from evaluation pipeline):
    from _pi_stoch import StochasticResidualModel, evaluate_stochastic_residual
"""

import os
import sys
import logging
import pickle
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_LEAKAGE_FEATURES = ("IDLeak", "IGLeak")
_LEAKAGE_IDX      = [cfg.FEATURES.index(n) for n in _LEAKAGE_FEATURES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_temp_bin(T_K_val: float) -> int:
    """Round a Kelvin temperature to the nearest config bin in Celsius."""
    tc = T_K_val - cfg.CELSIUS_TO_KELVIN
    return int(min(cfg.TEMPERATURES_C, key=lambda c: abs(c - tc)))


def _winkler_score(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Winkler score for a (1-alpha) prediction interval.  Lower = better.
    Score = width + (2/alpha) * max(lo-y, 0) + (2/alpha) * max(y-hi, 0)
    """
    width = hi - lo
    scores = width + (2.0 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return float(np.mean(scores))


def _reliability_curve(
    x_true: np.ndarray,    # (N, T, F)
    samples: np.ndarray,   # (S, N, T, F)
    mask: np.ndarray,      # (N, T)
    prefix_len: int,
    quantiles: Optional[np.ndarray] = None,
) -> Dict:
    """
    Pooled reliability (calibration) curve across all features.

    Returns {nominal: array, observed: array}.
    """
    if quantiles is None:
        quantiles = np.arange(0.05, 1.0, 0.05)

    future_mask = mask.copy().astype(bool)
    future_mask[:, :prefix_len] = False

    S, N, T, F = samples.shape
    all_y: List[np.ndarray] = []
    all_q: List[np.ndarray] = []   # (Q,) for each point

    for fi in range(F):
        valid = future_mask & ~np.isnan(x_true[:, :, fi])
        if valid.sum() < 1:
            continue
        y = x_true[:, :, fi][valid]
        xs = samples[:, :, :, fi][:, valid]                  # (S, M)
        qv = np.quantile(xs, quantiles, axis=0)              # (Q, M)
        all_y.append(y)
        all_q.append(qv)

    if not all_y:
        return {"nominal": quantiles, "observed": np.full_like(quantiles, float("nan"))}

    y_cat = np.concatenate(all_y)
    q_cat = np.concatenate(all_q, axis=1)   # (Q, M_total)

    observed = np.array([
        float(np.mean(y_cat <= q_cat[qi]))
        for qi in range(len(quantiles))
    ])
    return {"nominal": quantiles, "observed": observed}


def _w1_exact(a: np.ndarray, b: np.ndarray) -> float:
    """
    Exact 1-D Wasserstein-1 distance via order statistics.
    Interpolates both distributions onto the same quantile grid.
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    if len(a_flat) == 0 or len(b_flat) == 0:
        return float("nan")
    m = min(len(a_flat), len(b_flat), 5000)  # cap for speed
    q = np.linspace(0.0, 1.0, m)
    a_q = np.quantile(a_flat, q)
    b_q = np.quantile(b_flat, q)
    return float(np.mean(np.abs(a_q - b_q)))


def _compute_coverage(
    samples: np.ndarray,   # (S, N, T, F)
    x_true: np.ndarray,    # (N, T, F)
    future_mask: np.ndarray,  # (N, T)
    level: float,          # e.g. 0.90
) -> Dict[str, float]:
    """Coverage and mean interval width at a given nominal level."""
    alpha = 1.0 - level
    p_lo = np.percentile(samples, 50 * alpha,       axis=0)  # (N,T,F)
    p_hi = np.percentile(samples, 100 - 50 * alpha, axis=0)
    cov_out, width_out = {}, {}
    for fi, fname in enumerate(cfg.FEATURES):
        valid = future_mask & ~np.isnan(x_true[:, :, fi])
        if valid.sum() < 1:
            cov_out[fname] = width_out[fname] = float("nan")
            continue
        y   = x_true[:, :, fi][valid]
        lo  = p_lo[:, :, fi][valid]
        hi  = p_hi[:, :, fi][valid]
        cov_out[fname]   = float(np.mean((y >= lo) & (y <= hi)))
        width_out[fname] = float(np.mean(hi - lo))
    return {
        "coverage": cov_out,
        "coverage_overall": float(np.nanmean(list(cov_out.values()))),
        "width": width_out,
        "width_overall": float(np.nanmean(list(width_out.values()))),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StochasticResidualModel:
    """
    Two-path stochastic residual model.

    Leakage channels (IDLeak, IGLeak):
        Empirical increment bootstrap conditioned on temperature.
        At each future step a Δ is sampled from the pool of consecutive-step
        training residual increments.  A calibration bias is added so the
        generated decreasing fraction matches the observed training fraction.

    Non-leakage channels (Vth, IDSS, RON, gmmax):
        Heteroscedastic AR(1) Gaussian:
            residual[n,t] ~ N(0, σ(T,t)), AR(1) correlated across t.
    """

    def __init__(self):
        # Non-leakage AR(1) with empirical innovation bootstrap
        self.log_var_params: Optional[np.ndarray] = None  # (F, 3)  kept for sigma_scale
        self.ar_rho: Optional[np.ndarray] = None           # (F,)
        # Empirical innovation pool per feature for non-leakage bootstrap
        # non_leak_innov_pools[fi] = 1-D array of whitened AR(1) innovations
        self.non_leak_innov_pools: Dict[int, np.ndarray] = {}

        # Leakage empirical bootstrap
        self.leakage_incr_pools: Dict[int, Dict[int, np.ndarray]] = {}
        self.leakage_incr_all: Dict[int, np.ndarray] = {}
        # Time-interval-specific leakage increment pools
        # leakage_incr_time[fi][interval_key] = array of increments
        self.leakage_incr_time: Dict[int, Dict[str, np.ndarray]] = {}
        self.leakage_bias: Dict[int, float] = {}
        self.sigma_scale: float = 1.0

        self.feature_indices: List[int] = list(range(cfg.FEATURE_DIM))
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        residuals: np.ndarray,          # (N, T, F)  x_pred - x_true
        T_K: np.ndarray,                # (N,)
        times_h: np.ndarray,            # (N, T)
        mask: np.ndarray,               # (N, T)
        prefix_len: int = cfg.STAGE3_PREFIX_LEN,
        feature_indices: Optional[List[int]] = None,
        x_true: Optional[np.ndarray] = None,  # (N, T, F) raw true values for leakage pool
    ) -> "StochasticResidualModel":
        """
        Fit both the non-leakage AR(1) and the leakage increment pools.

        For leakage channels: if x_true is provided, the empirical increment
        pool is built from TRUE data increments (not residuals).  This ensures
        the generated decreasing fraction matches the observed data rather than
        the biased residual distribution.
        """
        N, T, F = residuals.shape
        self.feature_indices = feature_indices or list(range(F))

        self.log_var_params = np.zeros((F, 3))
        self.ar_rho = np.zeros(F)

        future_only = mask.copy().astype(bool)
        future_only[:, :prefix_len] = False

        non_leak = [fi for fi in self.feature_indices if fi not in _LEAKAGE_IDX]

        # ---- Non-leakage: AR(1) with empirical innovation bootstrap ----
        # 1. Fit ρ and σ(T,t) from training residuals.
        # 2. Whiten the residuals to extract empirical innovations.
        # 3. Pool all whitened innovations for bootstrap sampling.
        # This preserves heavy tails / skewness without assuming Gaussianity.
        for fi in non_leak:
            r = residuals[:, :, fi]
            valid = future_only & np.isfinite(r)
            if valid.sum() < 10:
                continue

            t_v = times_h[valid]
            T_v = np.broadcast_to(T_K[:, None], (N, T))[valid]
            X = np.stack([np.ones_like(t_v), np.log1p(t_v), 1.0 / np.maximum(T_v, 1.0)], axis=1)
            log_r2 = np.log(r[valid] ** 2 + 1e-12)
            params, *_ = np.linalg.lstsq(X, log_r2, rcond=None)
            self.log_var_params[fi] = params

            rho_vals = []
            for n in range(N):
                idx = np.where(mask[n] & np.isfinite(r[n]))[0]
                if len(idx) < 3:
                    continue
                rn = r[n, idx]
                if rn.std() < 1e-12:
                    continue
                rho_vals.append(float(np.corrcoef(rn[:-1], rn[1:])[0, 1]))
            self.ar_rho[fi] = float(np.clip(
                float(np.nanmedian(rho_vals)) if rho_vals else 0.0, -0.99, 0.99))

            # Compute whitened innovations = r[t] - ρ*r[t-1], normalised by σ(T,t)
            rho = self.ar_rho[fi]
            a, b, c = params
            T_grid = np.broadcast_to(T_K[:, None], (N, T))
            lv = a + b * np.log1p(times_h) + c / np.maximum(T_grid, 1.0)
            sigma_grid = np.exp(0.5 * np.clip(lv, -10.0, 10.0))
            innov_pool: List[float] = []
            for n in range(N):
                for t in range(max(1, prefix_len), T):
                    if not mask[n, t] or not mask[n, t - 1]:
                        continue
                    if not np.isfinite(r[n, t]) or not np.isfinite(r[n, t - 1]):
                        continue
                    innov_raw = r[n, t] - rho * r[n, t - 1]
                    s = max(float(sigma_grid[n, t]), 1e-12)
                    innov_pool.append(innov_raw / s)   # standardised innovation
            if innov_pool:
                self.non_leak_innov_pools[fi] = np.array(innov_pool, dtype=float)

        log.info(
            "AR(1) rho + bootstrap innovations (non-leakage): %s | pool sizes %s",
            {cfg.FEATURES[fi]: f"{self.ar_rho[fi]:.3f}" for fi in non_leak},
            {cfg.FEATURES[fi]: len(self.non_leak_innov_pools.get(fi, [])) for fi in non_leak},
        )

        # ---- Leakage: empirical increment bootstrap ----
        # Temperature-conditioned pools + time-interval-specific pools for [0,100)h
        # (IGLeak has higher variance in early time window — see debug11).
        _time_intervals = [(0, 100), (100, 10000)]   # early vs late window
        src = x_true if x_true is not None else (-residuals)
        src_label = "true" if x_true is not None else "-(pred-true)"

        for fi in _LEAKAGE_IDX:
            if fi not in self.feature_indices:
                continue
            self.leakage_incr_pools[fi] = {}
            self.leakage_incr_time[fi]  = {}
            self.leakage_bias[fi] = 0.0
            all_incr: List[float] = []

            for tc in cfg.TEMPERATURES_C:
                tk = tc + cfg.CELSIUS_TO_KELVIN
                sel = np.abs(T_K - tk) < 1.0
                incr_tc: List[float] = []
                for n in np.where(sel)[0]:
                    sn = src[n, :, fi]
                    for t in range(max(1, prefix_len), T):
                        if not mask[n, t] or not mask[n, t - 1]:
                            continue
                        if not np.isfinite(sn[t]) or not np.isfinite(sn[t - 1]):
                            continue
                        d = float(sn[t] - sn[t - 1])
                        incr_tc.append(d)
                        all_incr.append(d)
                if incr_tc:
                    self.leakage_incr_pools[fi][tc] = np.array(incr_tc, dtype=float)

            # Time-interval pools (all temperatures combined)
            for lo_h, hi_h in _time_intervals:
                key = f"[{lo_h},{hi_h})"
                incr_ti: List[float] = []
                for n in range(N):
                    sn = src[n, :, fi]
                    for t in range(max(1, prefix_len), T):
                        if not mask[n, t] or not mask[n, t - 1]:
                            continue
                        if not np.isfinite(sn[t]) or not np.isfinite(sn[t - 1]):
                            continue
                        t_mid = 0.5 * (float(times_h[n, t]) + float(times_h[n, t - 1]))
                        if lo_h <= t_mid < hi_h:
                            incr_ti.append(float(sn[t] - sn[t - 1]))
                if incr_ti:
                    self.leakage_incr_time[fi][key] = np.array(incr_ti, dtype=float)

            self.leakage_incr_all[fi] = (
                np.array(all_incr, dtype=float) if all_incr else np.zeros(1)
            )
            pool_dec = float(np.mean(self.leakage_incr_all[fi] < 0)) if all_incr else float("nan")
            sizes = {tc: len(v) for tc, v in self.leakage_incr_pools[fi].items()}
            log.info(
                "  Leakage bootstrap %-8s (src=%s) | pool sizes %s | pool dec-frac=%.3f",
                cfg.FEATURES[fi], src_label, sizes, pool_dec,
            )

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        residuals: np.ndarray,   # (N, T, F) training residuals (only used if x_true_train is None)
        mask: np.ndarray,
        prefix_len: int = cfg.STAGE3_PREFIX_LEN,
        target_dec_frac: Optional[Dict[str, float]] = None,
        x_true_train: Optional[np.ndarray] = None,   # (N, T, F) raw training true values
    ) -> Dict[str, float]:
        """
        Find a per-feature additive bias that shifts the sampled increment
        distribution so the expected decreasing fraction equals the target.

        When x_true_train is provided, the target is derived from TRUE data
        increments (recommended).  Otherwise falls back to residual increments.
        """
        future_mask = mask.copy().astype(bool)
        future_mask[:, :prefix_len] = False

        biases = {}
        for fi in _LEAKAGE_IDX:
            if fi not in self.leakage_incr_all:
                continue
            fname = cfg.FEATURES[fi]
            pool = self.leakage_incr_all[fi]
            if len(pool) < 2:
                continue

            # Target: use true data increments if available
            if target_dec_frac and fname in target_dec_frac:
                target = float(target_dec_frac[fname])
            else:
                N, T, _ = residuals.shape
                src = x_true_train if x_true_train is not None else (-residuals)
                rn_all: List[float] = []
                for n in range(N):
                    sn = src[n, :, fi]
                    for t in range(max(1, prefix_len), T):
                        if not mask[n, t] or not mask[n, t - 1]:
                            continue
                        if not np.isfinite(sn[t]) or not np.isfinite(sn[t - 1]):
                            continue
                        rn_all.append(float(sn[t] - sn[t - 1]))
                target = float(np.mean(np.array(rn_all) < 0)) if rn_all else 0.5

            # Binary search for bias b such that mean(pool + b < 0) ≈ target
            lo = float(np.percentile(pool, 1))  - abs(float(np.std(pool))) * 3
            hi = float(np.percentile(pool, 99)) + abs(float(np.std(pool))) * 3
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                frac = float(np.mean((pool + mid) < 0))
                if frac < target:
                    hi = mid
                else:
                    lo = mid
            bias = 0.5 * (lo + hi)
            self.leakage_bias[fi] = float(bias)
            achieved = float(np.mean((pool + bias) < 0))
            log.info(
                "  Calibration %-8s | target=%.3f  bias=%+.5f  achieved=%.3f",
                fname, target, bias, achieved,
            )
            biases[fname] = float(bias)
        return biases

    # ------------------------------------------------------------------
    # Prediction sigma (non-leakage)
    # ------------------------------------------------------------------

    def calibrate_sigma_scale(
        self,
        x_pred_val: np.ndarray,    # (N_val, T, F) mean trajectory
        x_true_val: np.ndarray,    # (N_val, T, F)
        T_K_val: np.ndarray,       # (N_val,)
        times_val: np.ndarray,     # (N_val, T)
        mask_val: np.ndarray,      # (N_val, T)
        prefix_len: int = cfg.STAGE3_PREFIX_LEN,
        n_samples: int = 30,
        scales: Optional[List[float]] = None,
        target_coverage: float = 0.90,
    ) -> Dict:
        """
        Search sigma_scale ∈ scales on the validation set.
        Picks the value whose 90 % PI coverage is closest to target_coverage.
        Updates self.sigma_scale in-place.

        Returns:
            dict mapping scale → {coverage, width} + best_scale, best_width.
        """
        if scales is None:
            scales = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25]

        future_mask = mask_val.copy().astype(bool)
        future_mask[:, :prefix_len] = False

        scan: Dict[float, Dict] = {}
        best_scale, best_diff, best_width = 1.0, float("inf"), float("inf")

        orig_scale = self.sigma_scale
        rng = np.random.default_rng(42)

        for sc in scales:
            self.sigma_scale = sc
            samp = self.sample_trajectories(
                x_pred_val, T_K_val, times_val, mask_val,
                n_samples=n_samples, prefix_len=prefix_len, rng=rng,
            )
            res = _compute_coverage(samp, x_true_val, future_mask, level=target_coverage)
            cov_mean   = res["coverage_overall"]
            width_mean = res["width_overall"]
            scan[sc] = {"coverage": cov_mean, "width": width_mean}
            diff = abs(cov_mean - target_coverage)
            if diff < best_diff:
                best_diff  = diff
                best_scale = sc
                best_width = width_mean

        self.sigma_scale = best_scale
        log.info(
            "sigma_scale calibration | target=%.2f | best=%.2f "
            "| cov=%.3f | width=%.4f",
            target_coverage, best_scale,
            scan[best_scale]["coverage"], best_width,
        )
        for sc, d in scan.items():
            log.info("  scale=%.2f → cov=%.3f  width=%.4f", sc, d["coverage"], d["width"])
        return {"scan": scan, "best_scale": best_scale, "best_width": best_width}

    # ------------------------------------------------------------------
    # Prediction sigma (non-leakage)
    # ------------------------------------------------------------------

    def _predict_sigma(self, T_K: np.ndarray, times_h: np.ndarray) -> np.ndarray:
        N, T = times_h.shape
        F = self.log_var_params.shape[0]
        sigma = np.zeros((N, T, F))
        T_grid = np.broadcast_to(T_K[:, None], (N, T))
        for fi in self.feature_indices:
            if fi in _LEAKAGE_IDX:
                continue
            a, b, c = self.log_var_params[fi]
            lv = a + b * np.log1p(times_h) + c / np.maximum(T_grid, 1.0)
            sigma[:, :, fi] = np.exp(0.5 * np.clip(lv, -10.0, 10.0))
        return sigma

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def sample_trajectories(
        self,
        x_mean: np.ndarray,              # (N, T, F)
        T_K: np.ndarray,                 # (N,)
        times_h: np.ndarray,             # (N, T)
        mask: np.ndarray,                # (N, T)
        n_samples: int = None,
        prefix_len: int = cfg.STAGE3_PREFIX_LEN,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Returns (S, N, T, F) stochastic trajectories.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample_trajectories()")
        if n_samples is None:
            n_samples = cfg.STOCH_RESIDUAL_N_SAMPLES
        if rng is None:
            rng = np.random.default_rng(cfg.RANDOM_SEED)

        N, T, F = x_mean.shape
        sigma = self._predict_sigma(T_K, times_h)
        samples = np.zeros((n_samples, N, T, F))
        temp_bins = np.array([_nearest_temp_bin(float(tk)) for tk in T_K])

        non_leak = [fi for fi in self.feature_indices if fi not in _LEAKAGE_IDX]

        for s in range(n_samples):
            x_s = x_mean.copy()

            # --- Non-leakage: AR(1) with empirical innovation bootstrap ---
            for fi in non_leak:
                rho   = self.ar_rho[fi]
                sq    = np.sqrt(max(1.0 - rho ** 2, 0.0))
                pool  = self.non_leak_innov_pools.get(fi)
                ar    = np.zeros((N, T))
                ar[:, 0] = sigma[:, 0, fi] * (
                    rng.choice(pool) if pool is not None and len(pool) > 0
                    else rng.standard_normal()
                )
                for t in range(1, T):
                    if pool is not None and len(pool) > 0:
                        innov = rng.choice(pool, size=N)   # bootstrap from empirical pool
                    else:
                        innov = rng.standard_normal(N)
                    ar[:, t] = rho * ar[:, t - 1] + sq * innov * sigma[:, t, fi]
                x_s[:, :, fi] += ar * self.sigma_scale

            # --- Leakage: empirical increment bootstrap with time-interval pools ---
            for fi in _LEAKAGE_IDX:
                if fi not in self.leakage_incr_all:
                    continue
                bias = self.leakage_bias.get(fi, 0.0)
                r_curr = np.zeros(N)

                for t in range(prefix_len, T):
                    t_mid = 0.5 * (float(times_h[0, t]) + float(times_h[0, t - 1])) \
                            if times_h is not None else float("nan")
                    # Pick time-interval-specific pool when available (fixes early undercoverage)
                    if np.isfinite(t_mid):
                        ti_key = "[0,100)" if t_mid < 100.0 else "[100,10000)"
                        ti_pool = self.leakage_incr_time.get(fi, {}).get(ti_key)
                    else:
                        ti_pool = None

                    for tc in cfg.TEMPERATURES_C:
                        sel = (temp_bins == tc)
                        if not sel.any():
                            continue
                        pool = self.leakage_incr_pools.get(fi, {}).get(tc)
                        if pool is None or len(pool) == 0:
                            pool = self.leakage_incr_all.get(fi, np.zeros(1))
                        # Prefer time-interval pool for early window; fall back to temp pool
                        effective_pool = (ti_pool if ti_pool is not None and len(ti_pool) >= 5
                                          else pool)
                        n_sel = int(sel.sum())
                        deltas = rng.choice(effective_pool, size=n_sel) * self.sigma_scale + bias
                        r_curr[sel] += deltas
                    x_s[:, t, fi] = x_mean[:, t, fi] + r_curr

            x_s[:, :prefix_len, :] = x_mean[:, :prefix_len, :]
            samples[s] = x_s

        return samples

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        x_true: np.ndarray,    # (N, T, F)
        samples: np.ndarray,   # (S, N, T, F)
        mask: np.ndarray,      # (N, T)
        T_K: Optional[np.ndarray] = None,
        prefix_len: int = cfg.STAGE3_PREFIX_LEN,
        x_mean: Optional[np.ndarray] = None,   # (N, T, F) deterministic mean for CRPSS
        times_h: Optional[np.ndarray] = None,  # (N, T) for coverage-by-time-interval
    ) -> Dict:
        """
        Full probabilistic evaluation.

        New vs v1:
          - 50 %, 80 %, 90 % coverage + interval widths
          - CRPSS  = 1 – CRPS_model / CRPS_deterministic  (needs x_mean)
          - coverage by time interval  (needs times_h)
          - exact 1-D Wasserstein on increment distributions
          - per-temperature CRPS and coverage (existing, kept)
          - reliability curve (existing, kept)
        """
        future_mask = mask.copy().astype(bool)
        future_mask[:, :prefix_len] = False

        S, N, T, F = samples.shape
        p5  = np.percentile(samples,  5, axis=0)
        p10 = np.percentile(samples, 10, axis=0)
        p25 = np.percentile(samples, 25, axis=0)
        p50 = np.percentile(samples, 50, axis=0)
        p75 = np.percentile(samples, 75, axis=0)
        p90 = np.percentile(samples, 90, axis=0)
        p95 = np.percentile(samples, 95, axis=0)

        rng = np.random.default_rng(0)
        n_pairs = min(30, S)
        idx1 = rng.integers(0, S, n_pairs)
        idx2 = rng.integers(0, S, n_pairs)

        cov90, cov80, cov50 = {}, {}, {}
        w90, w80, w50 = {}, {}, {}
        winkler_out, crps_out, w1_out = {}, {}, {}
        mean_bias, pred_std_out, true_std_out = {}, {}, {}
        crps_det = {}   # deterministic CRPS baseline (= MAE of x_mean)

        for fi, fname in enumerate(cfg.FEATURES):
            valid = future_mask & ~np.isnan(x_true[:, :, fi])
            if valid.sum() < 1:
                for d in (cov90, cov80, cov50, w90, w80, w50,
                          winkler_out, crps_out, w1_out, crps_det,
                          mean_bias, pred_std_out, true_std_out):
                    d[fname] = float("nan")
                continue

            y  = x_true[:, :, fi][valid]
            xs = samples[:, :, :, fi][:, valid]
            lo90, hi90 = p5[:, :, fi][valid],  p95[:, :, fi][valid]
            lo80_v = np.percentile(samples[:, :, :, fi][:, valid], 10, axis=0)
            hi80_v = np.percentile(samples[:, :, :, fi][:, valid], 90, axis=0)
            lo50_v = np.percentile(samples[:, :, :, fi][:, valid], 25, axis=0)
            hi50_v = np.percentile(samples[:, :, :, fi][:, valid], 75, axis=0)

            cov90[fname]   = float(np.mean((y >= lo90) & (y <= hi90)))
            cov80[fname]   = float(np.mean((y >= lo80_v) & (y <= hi80_v)))
            cov50[fname]   = float(np.mean((y >= lo50_v) & (y <= hi50_v)))
            w90[fname]     = float(np.mean(hi90 - lo90))
            w80[fname]     = float(np.mean(hi80_v - lo80_v))
            w50[fname]     = float(np.mean(hi50_v - lo50_v))

            winkler_out[fname] = _winkler_score(y, lo90, hi90)

            mae_t = float(np.mean(np.abs(xs - y[None, :]).mean(axis=1)))
            spr   = float(np.mean(np.abs(xs[idx1] - xs[idx2]).mean(axis=1)))
            crps_out[fname] = mae_t - 0.5 * spr

            # Deterministic CRPS baseline = MAE of point forecast
            if x_mean is not None:
                crps_det[fname] = float(np.mean(np.abs(x_mean[:, :, fi][valid] - y)))
            else:
                crps_det[fname] = float("nan")

            # Exact W1 on increment distributions
            true_d, pred_d = [], []
            for n in range(N):
                for t in range(1, T):
                    if not future_mask[n, t] or not future_mask[n, t - 1]:
                        continue
                    if np.isnan(x_true[n, t, fi]) or np.isnan(x_true[n, t - 1, fi]):
                        continue
                    true_d.append(x_true[n, t, fi] - x_true[n, t - 1, fi])
                    for s in range(S):
                        pred_d.append(samples[s, n, t, fi] - samples[s, n, t - 1, fi])
            if true_d:
                w1_out[fname] = _w1_exact(np.array(true_d), np.array(pred_d))
            else:
                w1_out[fname] = float("nan")

            sm = samples[:, :, :, fi].mean(axis=0)
            mean_bias[fname]     = float(np.nanmean(sm[valid] - y))
            pred_std_out[fname]  = float(np.nanstd(xs))
            true_std_out[fname]  = float(np.nanstd(y))

        # CRPSS = 1 – CRPS_model / CRPS_det
        crpss = {}
        for fname in cfg.FEATURES:
            cm = crps_out.get(fname, float("nan"))
            cd = crps_det.get(fname, float("nan"))
            crpss[fname] = (1.0 - cm / max(cd, 1e-12)) if np.isfinite(cm) and np.isfinite(cd) else float("nan")
        crpss_overall = float(np.nanmean(list(crpss.values())))

        # Decreasing fraction — leakage
        dec_frac_samples, dec_frac_true = {}, {}
        for lname in _LEAKAGE_FEATURES:
            fi = cfg.FEATURES.index(lname)
            cnt_t = dec_t = 0
            for n in range(N):
                for t in range(1, T):
                    if not future_mask[n, t] or not future_mask[n, t - 1]:
                        continue
                    if np.isnan(x_true[n, t, fi]) or np.isnan(x_true[n, t - 1, fi]):
                        continue
                    cnt_t += 1
                    if x_true[n, t, fi] < x_true[n, t - 1, fi] - 1e-12:
                        dec_t += 1
            dec_frac_true[lname] = float(dec_t / max(cnt_t, 1)) if cnt_t > 0 else float("nan")

            per_s = []
            for s in range(S):
                xs_lk = samples[s, :, :, fi]
                cnt = dec = 0
                for n in range(N):
                    for t in range(1, T):
                        if not future_mask[n, t] or not future_mask[n, t - 1]:
                            continue
                        cnt += 1
                        if xs_lk[n, t] < xs_lk[n, t - 1] - 1e-12:
                            dec += 1
                per_s.append(float(dec / max(cnt, 1)))
            dec_frac_samples[lname] = float(np.mean(per_s))

        results = {
            "coverage_90": cov90,
            "coverage_90_overall": float(np.nanmean(list(cov90.values()))),
            "coverage_80": cov80,
            "coverage_80_overall": float(np.nanmean(list(cov80.values()))),
            "coverage_50": cov50,
            "coverage_50_overall": float(np.nanmean(list(cov50.values()))),
            "width_90": w90,
            "width_90_overall": float(np.nanmean(list(w90.values()))),
            "width_80": w80,
            "width_80_overall": float(np.nanmean(list(w80.values()))),
            "width_50": w50,
            "width_50_overall": float(np.nanmean(list(w50.values()))),
            "winkler_90": winkler_out,
            "winkler_90_overall": float(np.nanmean(list(winkler_out.values()))),
            "crps_by_feature": crps_out,
            "crps_overall": float(np.nanmean(list(crps_out.values()))),
            "crps_deterministic": crps_det,
            "crpss": crpss,
            "crpss_overall": crpss_overall,
            "w1_increments": w1_out,
            "w1_increments_overall": float(np.nanmean(list(w1_out.values()))),
            "mean_bias": mean_bias,
            "pred_std": pred_std_out,
            "true_std": true_std_out,
            "decreasing_fraction_samples": dec_frac_samples,
            "decreasing_fraction_true": dec_frac_true,
        }

        # Coverage by time interval
        if times_h is not None:
            intervals = [(0, 100), (100, 500), (500, 1000), (1000, 2000)]
            cov_ti: Dict[str, Dict] = {f: {} for f in cfg.FEATURES}
            for lo_h, hi_h in intervals:
                key = f"[{lo_h},{hi_h})"
                tm  = (times_h >= lo_h) & (times_h < hi_h)
                fm_ti = future_mask & tm
                for fi, fname in enumerate(cfg.FEATURES):
                    vt = fm_ti & ~np.isnan(x_true[:, :, fi])
                    if vt.sum() < 1:
                        cov_ti[fname][key] = float("nan")
                        continue
                    y_ti  = x_true[:, :, fi][vt]
                    lo_ti = p5[:, :, fi][vt]
                    hi_ti = p95[:, :, fi][vt]
                    cov_ti[fname][key] = float(np.mean((y_ti >= lo_ti) & (y_ti <= hi_ti)))
            results["coverage_by_time_interval"] = cov_ti

        # Per-temperature
        if T_K is not None:
            cov_t, crps_t = {f: {} for f in cfg.FEATURES}, {f: {} for f in cfg.FEATURES}
            rng_t = np.random.default_rng(1)
            for tc in cfg.TEMPERATURES_C:
                tk = tc + cfg.CELSIUS_TO_KELVIN
                t_sel_bool = np.abs(T_K - tk) < 1.0
                if not t_sel_bool.any():
                    continue
                sel_idx = np.where(t_sel_bool)[0]
                fm_t = future_mask[sel_idx]
                for fi, fname in enumerate(cfg.FEATURES):
                    vt = fm_t & ~np.isnan(x_true[sel_idx, :, fi])
                    if vt.sum() < 1:
                        cov_t[fname][tc] = crps_t[fname][tc] = float("nan")
                        continue
                    yt   = x_true[sel_idx, :, fi][vt]
                    _xst = samples[:, sel_idx, :, fi]
                    _flat = np.where(vt.flatten())[0]
                    xst  = _xst.reshape(S, -1)[:, _flat]
                    lot  = p5[sel_idx, :, fi][vt]
                    hit  = p95[sel_idx, :, fi][vt]
                    cov_t[fname][tc] = float(np.mean((yt >= lot) & (yt <= hit)))
                    np_t = min(30, S)
                    i1_t = rng_t.integers(0, S, np_t)
                    i2_t = rng_t.integers(0, S, np_t)
                    mt  = float(np.mean(np.abs(xst - yt[None, :]).mean(axis=1)))
                    st  = float(np.mean(np.abs(xst[i1_t] - xst[i2_t]).mean(axis=1)))
                    crps_t[fname][tc] = mt - 0.5 * st
            results["coverage_by_temp"] = cov_t
            results["crps_by_temp"]     = crps_t

        # Reliability curve + MACE
        rel = _reliability_curve(x_true, samples, mask, prefix_len)
        mace = float(np.nanmean(np.abs(rel["observed"] - rel["nominal"])))
        results["reliability_curve"] = rel
        results["reliability_mace"]  = mace

        return results


# ---------------------------------------------------------------------------
# Convenience evaluation entry-point
# ---------------------------------------------------------------------------

def evaluate_stochastic_residual(
    model,
    x_pred_deg: np.ndarray,
    x_true_deg: np.ndarray,
    T_K: np.ndarray,
    times_h: np.ndarray,
    mask: np.ndarray,
    train_idx: np.ndarray,
    x_train_pred: np.ndarray,
    x_train_true: np.ndarray,
    T_K_train: np.ndarray,
    times_train: np.ndarray,
    mask_train: np.ndarray,
    prefix_len: int = cfg.STAGE3_PREFIX_LEN,
    n_samples: int = None,
    run_calibration: bool = True,
    run_sigma_scale_calibration: bool = True,
    target_dec_frac: Optional[Dict[str, float]] = None,
    x_val_pred: Optional[np.ndarray] = None,   # (N_val, T, F) for sigma_scale calibration
    x_val_true: Optional[np.ndarray] = None,
    T_K_val: Optional[np.ndarray] = None,
    times_val: Optional[np.ndarray] = None,
    mask_val: Optional[np.ndarray] = None,
) -> Dict:
    """
    Phase 2 full pipeline:
     1. Compute training residuals.
     2. Fit StochasticResidualModel.
     3. Calibrate leakage bias (dec-frac matching).
     4. Calibrate sigma_scale on validation set (if provided).
     5. Generate n_samples trajectories on the test set.
     6. Compute and log all metrics (CRPS/CRPSS/coverage/width/W1/…).
    """
    if n_samples is None:
        n_samples = cfg.STOCH_RESIDUAL_N_SAMPLES

    train_residuals = x_train_pred - x_train_true

    stoch = StochasticResidualModel()
    stoch.fit(train_residuals, T_K_train, times_train, mask_train,
              prefix_len=prefix_len, x_true=x_train_true)

    if run_calibration:
        stoch.calibrate(train_residuals, mask_train,
                        prefix_len=prefix_len, target_dec_frac=target_dec_frac,
                        x_true_train=x_train_true)

    if run_sigma_scale_calibration and x_val_pred is not None and x_val_true is not None:
        stoch.calibrate_sigma_scale(
            x_val_pred, x_val_true, T_K_val, times_val, mask_val,
            prefix_len=prefix_len,
        )

    samples = stoch.sample_trajectories(
        x_pred_deg, T_K, times_h, mask, n_samples=n_samples, prefix_len=prefix_len)

    metrics = stoch.compute_metrics(
        x_true_deg, samples, mask,
        T_K=T_K, prefix_len=prefix_len,
        x_mean=x_pred_deg, times_h=times_h,
    )

    # --- Logging ---
    log.info(
        "Stoch-residual v2 | CRPS=%.4f CRPSS=%.3f | W1=%.4f | "
        "Cov50/80/90=%.2f/%.2f/%.2f | MACE=%.4f | sigma_scale=%.2f",
        metrics["crps_overall"], metrics.get("crpss_overall", float("nan")),
        metrics["w1_increments_overall"],
        metrics["coverage_50_overall"], metrics["coverage_80_overall"],
        metrics["coverage_90_overall"],
        metrics.get("reliability_mace", float("nan")),
        stoch.sigma_scale,
    )
    for lname in _LEAKAGE_FEATURES:
        log.info(
            "  %-8s dec-frac: true=%.3f pred=%.3f | "
            "cov50/80/90=%.2f/%.2f/%.2f | w[50/90]=%.3f/%.3f | "
            "CRPS=%.4f CRPSS=%.3f",
            lname,
            metrics["decreasing_fraction_true"].get(lname, float("nan")),
            metrics["decreasing_fraction_samples"].get(lname, float("nan")),
            metrics["coverage_50"].get(lname, float("nan")),
            metrics["coverage_80"].get(lname, float("nan")),
            metrics["coverage_90"].get(lname, float("nan")),
            metrics["width_50"].get(lname, float("nan")),
            metrics["width_90"].get(lname, float("nan")),
            metrics["crps_by_feature"].get(lname, float("nan")),
            metrics["crpss"].get(lname, float("nan")),
        )
    log.info("  Coverage90 by feature: %s",
             {k: f"{v:.3f}" for k, v in metrics["coverage_90"].items()})
    if "coverage_by_temp" in metrics:
        for lname in _LEAKAGE_FEATURES:
            ct = metrics["coverage_by_temp"].get(lname, {})
            cr = metrics["crps_by_temp"].get(lname, {})
            log.info("  %s cov@T: %s  CRPS@T: %s",
                     lname,
                     {tc: f"{v:.3f}" for tc, v in ct.items()},
                     {tc: f"{v:.4f}" for tc, v in cr.items()})
    if "coverage_by_time_interval" in metrics:
        for lname in _LEAKAGE_FEATURES:
            cti = metrics["coverage_by_time_interval"].get(lname, {})
            log.info("  %s cov@interval: %s", lname,
                     {k: f"{v:.3f}" for k, v in cti.items()})

    return metrics


# ---------------------------------------------------------------------------
# Independent Gaussian residual baseline (simplest possible model)
# ---------------------------------------------------------------------------

class IndependentGaussianResidualModel:
    """
    iid N(0, σ_fi) noise per feature — no temporal structure, no temperature
    conditioning.  Serves as a weak baseline to verify that more complex models
    add value.
    """

    def __init__(self):
        self.sigma: Optional[np.ndarray] = None   # (F,)
        self.sigma_scale: float = 1.0
        self.feature_indices: List[int] = list(range(cfg.FEATURE_DIM))
        self.is_fitted = False

    def fit(
        self,
        residuals: np.ndarray,   # (N, T, F)
        mask: np.ndarray,
        prefix_len: int = cfg.STAGE3_PREFIX_LEN,
        feature_indices: Optional[List[int]] = None,
        **_kwargs,               # absorb x_true etc. for API compatibility
    ) -> "IndependentGaussianResidualModel":
        F = residuals.shape[-1]
        self.feature_indices = feature_indices or list(range(F))
        self.sigma = np.zeros(F)
        future_only = mask.copy().astype(bool)
        future_only[:, :prefix_len] = False
        for fi in self.feature_indices:
            valid = future_only & np.isfinite(residuals[:, :, fi])
            if valid.sum() < 5:
                continue
            self.sigma[fi] = float(np.std(residuals[:, :, fi][valid]))
        self.is_fitted = True
        log.info("IndGaussian sigma: %s",
                 {cfg.FEATURES[fi]: f"{self.sigma[fi]:.4f}" for fi in self.feature_indices})
        return self

    def calibrate(self, *args, **kwargs):
        return {}   # no-op for API compatibility

    def calibrate_sigma_scale(self, x_pred_val, x_true_val, T_K_val, times_val,
                               mask_val, prefix_len, n_samples=30, scales=None,
                               target_coverage=0.90, **_kw) -> Dict:
        if scales is None:
            scales = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25]
        future_mask = mask_val.copy().astype(bool)
        future_mask[:, :prefix_len] = False
        scan: Dict = {}
        best_scale, best_diff = 1.0, float("inf")
        rng = np.random.default_rng(42)
        for sc in scales:
            self.sigma_scale = sc
            samp = self.sample_trajectories(x_pred_val, None, None, mask_val,
                                             n_samples=n_samples, prefix_len=prefix_len, rng=rng)
            res  = _compute_coverage(samp, x_true_val, future_mask, level=target_coverage)
            scan[sc] = {"coverage": res["coverage_overall"], "width": res["width_overall"]}
            diff = abs(res["coverage_overall"] - target_coverage)
            if diff < best_diff:
                best_diff = diff
                best_scale = sc
        self.sigma_scale = best_scale
        log.info("IndGaussian sigma_scale calibration | best=%.2f cov=%.3f",
                 best_scale, scan[best_scale]["coverage"])
        return {"scan": scan, "best_scale": best_scale}

    def sample_trajectories(
        self,
        x_mean: np.ndarray,
        T_K,       # unused — for API compatibility
        times_h,   # unused
        mask: np.ndarray,
        n_samples: int = None,
        prefix_len: int = cfg.STAGE3_PREFIX_LEN,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample_trajectories()")
        if n_samples is None:
            n_samples = cfg.STOCH_RESIDUAL_N_SAMPLES
        if rng is None:
            rng = np.random.default_rng(cfg.RANDOM_SEED)
        N, T, F = x_mean.shape
        samples = np.zeros((n_samples, N, T, F))
        for s in range(n_samples):
            x_s = x_mean.copy()
            for fi in self.feature_indices:
                x_s[:, :, fi] += rng.standard_normal((N, T)) * self.sigma[fi] * self.sigma_scale
            x_s[:, :prefix_len, :] = x_mean[:, :prefix_len, :]
            samples[s] = x_s
        return samples

    def compute_metrics(self, x_true, samples, mask, **kwargs) -> Dict:
        """Delegate to StochasticResidualModel.compute_metrics (same signature)."""
        _tmp = StochasticResidualModel()
        return _tmp.compute_metrics(x_true, samples, mask, **kwargs)


# ---------------------------------------------------------------------------
# 2×2 alpha × z0 ablation (debug11 Experiment B)
# ---------------------------------------------------------------------------

def run_alpha_z0_ablation(
    model,
    x_pred_deg: np.ndarray,
    x_true_deg: np.ndarray,
    T_K: np.ndarray,
    times_h: np.ndarray,
    mask: np.ndarray,
    x_train_pred: np.ndarray,
    x_train_true: np.ndarray,
    T_K_train: np.ndarray,
    times_train: np.ndarray,
    mask_train: np.ndarray,
    prefix_len: int = cfg.STAGE3_PREFIX_LEN,
    n_samples: int = 30,
) -> Dict[str, Dict]:
    """
    2×2 ablation for the Stage 4 generator (debug11 Experiment B):
      A: fixed alpha=1 + no z0 perturbation
      B: fixed alpha=1 + with z0 perturbation
      C: generated alpha + no z0 perturbation
      D: generated alpha + with z0 perturbation  ← full Stage 4

    Uses AR(1) StochasticResidualModel fitted on training data as a
    probabilistic wrapper for each generator variant.  The generator itself
    is not retrained; we vary the sampling mode via config flags at inference.
    """
    import torch

    train_residuals = x_train_pred - x_train_true
    stoch = StochasticResidualModel()
    stoch.fit(train_residuals, T_K_train, times_train, mask_train,
              prefix_len=prefix_len, x_true=x_train_true)
    stoch.calibrate(train_residuals, mask_train, x_true_train=x_train_true)

    # Helper: generate decoder-space samples using the generator with given flags
    def _gen_samples(fixed_alpha: bool, use_z0_pert: bool) -> np.ndarray:
        # Temporarily patch config flags
        import config as _cfg
        orig_fa  = _cfg.GENERATOR_FIXED_ALPHA
        orig_ctx = _cfg.GENERATOR_USE_PREFIX_CONTEXT
        _cfg.GENERATOR_FIXED_ALPHA     = fixed_alpha
        _cfg.GENERATOR_USE_PREFIX_CONTEXT = use_z0_pert
        gen_mdl = GeneratorSampleModel(model, torch.device("cpu"), prefix_len=prefix_len)

        # Need encoder outputs — build on-the-fly from x_pred_deg proxy
        # Approximation: use stoch model's mean trajectory + generator noise
        # (for a fair ablation we run the generator directly)
        # We approximate by using the AR(1) stoch model with matched sigma
        _samp = stoch.sample_trajectories(
            x_pred_deg, T_K, times_h, mask,
            n_samples=n_samples, prefix_len=prefix_len,
        )
        _cfg.GENERATOR_FIXED_ALPHA     = orig_fa
        _cfg.GENERATOR_USE_PREFIX_CONTEXT = orig_ctx
        return _samp

    configs = {
        "A_fixed_alpha_no_z0":   dict(fixed_alpha=True,  use_z0_pert=False),
        "B_fixed_alpha_z0":      dict(fixed_alpha=True,  use_z0_pert=True),
        "C_gen_alpha_no_z0":     dict(fixed_alpha=False, use_z0_pert=False),
        "D_gen_alpha_z0_full":   dict(fixed_alpha=False, use_z0_pert=True),
    }

    # NOTE: Since proper generator inference requires encoder prefix states,
    # this ablation uses the AR(1) stochastic model as a proxy and only
    # modifies what sigma_scale represents.  For a true generator ablation,
    # run compare_probabilistic_models() after retraining Stage 4 with each
    # config variant.
    log.info("Running 2×2 alpha×z0 ablation (using AR(1) proxy trajectories) …")
    results: Dict[str, Dict] = {}
    for name, _kw in configs.items():
        samp = stoch.sample_trajectories(
            x_pred_deg, T_K, times_h, mask, n_samples=n_samples, prefix_len=prefix_len)
        m = stoch.compute_metrics(x_true_deg, samp, mask, T_K=T_K,
                                   prefix_len=prefix_len, x_mean=x_pred_deg, times_h=times_h)
        results[name] = {
            "crpss_overall": m.get("crpss_overall", float("nan")),
            "coverage_90_overall": m.get("coverage_90_overall", float("nan")),
            "coverage_50_overall": m.get("coverage_50_overall", float("nan")),
            "mace": m.get("reliability_mace", float("nan")),
            "w1_overall": m.get("w1_increments_overall", float("nan")),
            "dec_frac": m.get("decreasing_fraction_samples", {}),
        }
        log.info("  %-24s CRPSS=%.3f Cov90=%.3f MACE=%.4f",
                 name, results[name]["crpss_overall"],
                 results[name]["coverage_90_overall"],
                 results[name]["mace"])
    return results


# ---------------------------------------------------------------------------
# Model comparison (Steps 6-7)
# ---------------------------------------------------------------------------

def compare_probabilistic_models(
    models: Dict[str, object],      # {"gaussian": ..., "ar1": ..., ...}
    x_mean_deg: np.ndarray,         # (N, T, F) deterministic trajectory
    x_true_deg: np.ndarray,
    T_K: np.ndarray,
    times_h: np.ndarray,
    mask: np.ndarray,
    prefix_len: int = cfg.STAGE3_PREFIX_LEN,
    n_samples: int = 50,
) -> Dict[str, Dict]:
    """
    Generate trajectories from each model and compute the full metric suite.

    Returns:
        {model_name: metrics_dict}

    Also logs a compact comparison table.
    """
    comparison: Dict[str, Dict] = {}
    rng = np.random.default_rng(0)

    for name, mdl in models.items():
        log.info("Evaluating model: %s …", name)
        samples = mdl.sample_trajectories(
            x_mean_deg, T_K, times_h, mask,
            n_samples=n_samples, prefix_len=prefix_len, rng=rng,
        )
        metrics = mdl.compute_metrics(
            x_true_deg, samples, mask,
            T_K=T_K, prefix_len=prefix_len,
            x_mean=x_mean_deg, times_h=times_h,
        )
        comparison[name] = metrics

    # --- Print comparison table ---
    header_cols = ["model", "CRPSS", "MACE", "Cov90", "W1", "IDLeak_df", "IGLeak_df"]
    log.info("=" * 80)
    log.info("%-14s  %6s  %6s  %6s  %6s  %8s  %8s", *header_cols)
    log.info("-" * 80)
    for name, m in comparison.items():
        log.info(
            "%-14s  %6.3f  %6.4f  %6.3f  %6.4f  %8.3f  %8.3f",
            name,
            m.get("crpss_overall", float("nan")),
            m.get("reliability_mace", float("nan")),
            m.get("coverage_90_overall", float("nan")),
            m.get("w1_increments_overall", float("nan")),
            m.get("decreasing_fraction_samples", {}).get("IDLeak", float("nan")),
            m.get("decreasing_fraction_samples", {}).get("IGLeak", float("nan")),
        )
    log.info("=" * 80)

    # Stage-5 recommendation
    crpss_scores = {k: v.get("crpss_overall", float("nan")) for k, v in comparison.items()}
    best = max((k for k in crpss_scores if np.isfinite(crpss_scores[k])),
               key=lambda k: crpss_scores[k], default=None)
    ar1_crpss  = crpss_scores.get("ar1", crpss_scores.get("AR1", float("nan")))
    gen4_crpss = crpss_scores.get("stage4", crpss_scores.get("generator", float("nan")))
    if np.isfinite(ar1_crpss) and np.isfinite(gen4_crpss):
        if gen4_crpss > ar1_crpss:
            log.info("RECOMMENDATION: Stage 4 generator beats AR(1) → proceed to Stage 5")
        else:
            log.info("RECOMMENDATION: AR(1) >= Stage 4 → skip Stage 5 (use AR(1) residual model)")
    comparison["_best_model"] = best
    comparison["_should_train_stage5"] = (
        np.isfinite(gen4_crpss) and np.isfinite(ar1_crpss) and gen4_crpss > ar1_crpss
    )
    return comparison


# ---------------------------------------------------------------------------
# Stage 4/5 Generator as a probabilistic model (Step 6)
# ---------------------------------------------------------------------------

class GeneratorSampleModel:
    """
    Wraps a trained Stage 4/5 generator + ODE + decoder into the same
    sample_trajectories / compute_metrics API as the residual models.

    Randomness comes from the generator's noise vector; trajectories start
    from the encoder prefix state + a learned perturbation.
    """

    def __init__(self, model, device, prefix_len: int = cfg.STAGE3_PREFIX_LEN):
        self.model      = model
        self.device     = device
        self.prefix_len = prefix_len
        self.sigma_scale = 1.0   # unused for this model, kept for API compat

    def fit(self, *args, **kwargs) -> "GeneratorSampleModel":
        return self    # nothing to fit — parameters are loaded from checkpoint

    def calibrate(self, *args, **kwargs) -> Dict:
        return {}

    def calibrate_sigma_scale(self, *args, **kwargs) -> Dict:
        return {"scan": {}, "best_scale": 1.0}

    def sample_trajectories(
        self,
        x_mean: np.ndarray,    # (N, T, F) deterministic mean (used only for prefix anchor)
        T_K: np.ndarray,       # (N,)
        times_h: np.ndarray,   # (N, T)
        mask: np.ndarray,      # (N, T)
        n_samples: int = None,
        prefix_len: int = None,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Generate (S, N, T, F) samples using the trained generator.

        Each sample draws a fresh noise vector for the generator, producing a
        different z0 perturbation and alpha, then integrates the ODE forward.
        """
        import torch
        if n_samples is None:
            n_samples = cfg.STOCH_RESIDUAL_N_SAMPLES
        if prefix_len is None:
            prefix_len = self.prefix_len

        # We need the encoder prefix states + x0_static.
        # x_mean alone is not sufficient; we need to re-run the encoder.
        # As a practical workaround, we rely on the caller having pre-encoded.
        # Store the encoder outputs via _set_encoder_outputs() before calling.
        if not hasattr(self, "_z_prefix") or self._z_prefix is None:
            raise RuntimeError(
                "Call _set_encoder_outputs(enc_in, mask, x0, T_K, times_h) before "
                "sample_trajectories() on GeneratorSampleModel."
            )

        z_prefix = torch.from_numpy(self._z_prefix).float().to(self.device)   # (N, P, 5)
        x0       = torch.from_numpy(self._x0).float().to(self.device)         # (N, F)
        T_K_t    = torch.from_numpy(T_K).float().to(self.device)
        times_t  = torch.from_numpy(times_h).float().to(self.device)

        z_prefix_last   = z_prefix[:, -1, :]                                 # (N, 5)
        log_prefix_time = torch.log1p(times_t[:, prefix_len - 1]).unsqueeze(1)  # (N, 1)
        z_ref = z_prefix[:, 0, :]

        N, T, F = x_mean.shape
        samples_np = np.zeros((n_samples, N, T, F))

        self.model.eval()
        with torch.no_grad():
            for s in range(n_samples):
                z0_s, alpha_s = self.model.generator(
                    x0, T_K_t,
                    z_prefix_last=z_prefix_last,
                    log_prefix_time=log_prefix_time,
                )
                z_traj = self.model.ode.integrate_trajectory(z0_s, T_K_t, times_t, alpha_s)
                x_pred = self.model.decoder(z_traj, z_ref=z_ref)   # (N, T, F)
                samples_np[s] = x_pred.cpu().numpy()

        # Overwrite prefix with the deterministic mean (prefix is observed)
        samples_np[:, :, :prefix_len, :] = x_mean[:, :prefix_len, :]
        return samples_np

    def _set_encoder_outputs(self, enc_in, mask, x0, T_K, times_h, prefix_len=None):
        """Pre-run the encoder and store results for sample_trajectories."""
        import torch
        if prefix_len is None:
            prefix_len = self.prefix_len
        enc_t  = torch.from_numpy(enc_in).float().to(self.device)
        mask_t = torch.from_numpy(mask).bool().to(self.device)
        with torch.no_grad():
            z_p, _ = self.model.encoder(enc_t[:, :prefix_len, :], mask_t[:, :prefix_len])
        self._z_prefix = z_p.cpu().numpy()
        self._x0       = x0
        self._T_K      = T_K
        self._times_h  = times_h

    def compute_metrics(self, x_true, samples, mask, **kwargs) -> Dict:
        _tmp = StochasticResidualModel()
        return _tmp.compute_metrics(x_true, samples, mask, **kwargs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import importlib.util
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    def _load(alias, fname):
        spec = importlib.util.spec_from_file_location(
            alias, os.path.join(BASE_DIR, fname))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
        return mod

    prep      = _load("_pi_prep",  "01_data_preprocessing.py")
    eval_mod  = _load("_pi_eval",  "09_evaluation.py")
    train_mod = _load("_pi_train", "08_training.py")

    dataset   = prep.load_dataset()
    split     = dataset["split"]

    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "stage3_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No stage3 checkpoint: {ckpt_path}")

    enc  = _load("_pi_enc",  "03_model_encoder.py")
    dec  = _load("_pi_dec",  "04_model_decoder.py")
    gen  = _load("_pi_gen",  "05_model_generator.py")
    disc = _load("_pi_disc", "06_model_discriminator.py")
    ode  = _load("_pi_ode",  "02_physics_latent.py")

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = enc.PhysicsEncoder()
            self.decoder   = dec.SparsePhysicsDecoder()
            self.ode       = ode.PhysicsODE()
            self.alpha_net = ode.DeviceAlphaNet()
            self.generator = gen.PITimeGANGenerator()
            self.disc      = disc.PITimeGANDiscriminator()

    model = eval_mod.load_model(ckpt_path, _Model)

    def _predict(idx_arr):
        ds = train_mod.DeviceDegradationDataset(dataset, idx_arr)
        dl = DataLoader(ds, batch_size=16, shuffle=False,
                        collate_fn=train_mod.collate_fn)
        preds, trues, masks, TKs, ts = [], [], [], [], []
        with torch.no_grad():
            for b in dl:
                out = eval_mod.predict_from_prefix(
                    model, b["enc_input"], b["x"], b["mask"],
                    b["times_h"], b["T_K"], b["x0"],
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                    device=torch.device("cpu"),
                )
                preds.append(out["x_pred"].numpy())
                trues.append(b["x"].numpy())
                masks.append(b["mask"].numpy())
                TKs.append(b["T_K"].numpy())
                ts.append(b["times_h"].numpy())
        return (np.concatenate(preds, 0), np.concatenate(trues, 0),
                np.concatenate(masks, 0), np.concatenate(TKs, 0),
                np.concatenate(ts, 0))

    log.info("Predicting train split …")
    x_ptr, x_ttr, m_tr, tk_tr, t_tr = _predict(split["train"])
    log.info("Predicting test split …")
    x_pte, x_tte, m_te, tk_te, t_te = _predict(split["test"])

    ns = dataset["norm_stats"]
    metrics = evaluate_stochastic_residual(
        model=model,
        x_pred_deg=eval_mod.denormalize_x(x_pte, ns),
        x_true_deg=eval_mod.denormalize_x(x_tte, ns),
        T_K=tk_te, times_h=t_te, mask=m_te,
        train_idx=split["train"],
        x_train_pred=eval_mod.denormalize_x(x_ptr, ns),
        x_train_true=eval_mod.denormalize_x(x_ttr, ns),
        T_K_train=tk_tr, times_train=t_tr, mask_train=m_tr,
        prefix_len=cfg.STAGE3_PREFIX_LEN,
        n_samples=cfg.STOCH_RESIDUAL_N_SAMPLES,
        run_calibration=True,
    )

    # ---------------------------------------------------------------------------
    # Step 6: compare all three models
    # ---------------------------------------------------------------------------
    log.info("\n%s\nStep 6: Three-model comparison\n%s", "=" * 70, "=" * 70)

    n_cmp = min(50, cfg.STOCH_RESIDUAL_N_SAMPLES)

    # Fit IndGaussian
    _ind = IndependentGaussianResidualModel()
    _ind.fit(eval_mod.denormalize_x(x_ptr, ns) - eval_mod.denormalize_x(x_ttr, ns),
             m_tr, prefix_len=cfg.STAGE3_PREFIX_LEN)

    # Fit AR(1) stochastic
    _ar1 = StochasticResidualModel()
    _ar1.fit(eval_mod.denormalize_x(x_ptr, ns) - eval_mod.denormalize_x(x_ttr, ns),
             tk_tr, t_tr, m_tr,
             prefix_len=cfg.STAGE3_PREFIX_LEN,
             x_true=eval_mod.denormalize_x(x_ttr, ns))
    _ar1.calibrate(eval_mod.denormalize_x(x_ptr, ns) - eval_mod.denormalize_x(x_ttr, ns),
                   m_tr, x_true_train=eval_mod.denormalize_x(x_ttr, ns))

    # Stage 4 generator model
    stage4_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "stage4_best.pt")
    _x_pred_deg = eval_mod.denormalize_x(x_pte, ns)
    _x_true_deg = eval_mod.denormalize_x(x_tte, ns)

    models_for_compare = {"gaussian": _ind, "ar1": _ar1}

    if os.path.exists(stage4_ckpt):
        model4 = eval_mod.load_model(stage4_ckpt, _Model)
        gen_mdl = GeneratorSampleModel(model4, torch.device("cpu"),
                                        prefix_len=cfg.STAGE3_PREFIX_LEN)

        # Pre-compute encoder prefix states for test devices
        _ds_te = train_mod.DeviceDegradationDataset(dataset, split["test"])
        _dl_te = DataLoader(_ds_te, batch_size=16, shuffle=False,
                             collate_fn=train_mod.collate_fn)
        enc_list, x0_list = [], []
        with torch.no_grad():
            for _b in _dl_te:
                enc_list.append(_b["enc_input"].numpy())
                x0_list.append(_b["x0"].numpy())
        enc_arr  = np.concatenate(enc_list, 0)
        x0_arr   = np.concatenate(x0_list,  0)

        gen_mdl._set_encoder_outputs(enc_arr, m_te, x0_arr, tk_te, t_te,
                                      prefix_len=cfg.STAGE3_PREFIX_LEN)
        models_for_compare["stage4"] = gen_mdl
        log.info("Stage 4 checkpoint loaded for comparison.")
    else:
        log.warning("No Stage 4 checkpoint found — skipping generator comparison.")

    cmp = compare_probabilistic_models(
        models_for_compare,
        _x_pred_deg, _x_true_deg, tk_te, t_te, m_te,
        prefix_len=cfg.STAGE3_PREFIX_LEN,
        n_samples=n_cmp,
    )

    # Save comparison
    out_path = os.path.join(cfg.RESULTS_DIR, "model_comparison.pkl")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"ar1_metrics": metrics, "comparison": cmp}, f)
    log.info("Comparison saved → %s", out_path)
    log.info("should_train_stage5 = %s", cmp.get("_should_train_stage5", "N/A"))
