"""
11_grouped_cv_stability.py
==========================
Grouped 5-fold x multi-seed stability validation for probabilistic models.

Goal:
- Quantify whether AR(1) stochastic residual keeps its advantage across
  grouped folds and random seeds.

Outputs (under --output-dir):
- grouped_cv_fold_metrics.csv
- grouped_cv_model_summary.csv
- grouped_cv_seed_summary.csv
- grouped_cv_stability.pkl
"""

import argparse
import csv
import os
import pickle
import random
import sys
from importlib.util import module_from_spec, spec_from_file_location
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg


def _load_module(alias: str, filename: str):
    path = os.path.join(BASE_DIR, filename)
    spec = spec_from_file_location(alias, path)
    mod = module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_model(mods: Dict):
    import torch.nn as nn

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = mods["enc"].PhysicsEncoder()
            self.decoder = mods["dec"].SparsePhysicsDecoder()
            self.ode = mods["ode"].PhysicsODE()
            self.alpha_net = mods["ode"].DeviceAlphaNet()
            self.generator = mods["gen"].PITimeGANGenerator()
            self.disc = mods["disc"].PITimeGANDiscriminator()

    return _Model


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _parse_seeds(seed_arg: Optional[str]) -> List[int]:
    if seed_arg and seed_arg.strip():
        return [int(x.strip()) for x in seed_arg.split(",") if x.strip()]
    base = int(cfg.RANDOM_SEED)
    n = int(getattr(cfg, "N_RANDOM_SEEDS", 5))
    return [base + i for i in range(n)]


def _select_eval_indices(dataset: Dict, split_scope: str) -> np.ndarray:
    split = dataset["split"]
    if split_scope == "all":
        n = len(dataset["device_ids"])
        return np.arange(n, dtype=int)
    if split_scope == "test":
        return np.asarray(split["test"], dtype=int)
    if split_scope == "trainval":
        return np.concatenate([
            np.asarray(split["train"], dtype=int),
            np.asarray(split["val"], dtype=int),
        ])
    raise ValueError(f"Unknown split_scope: {split_scope}")


def _grouped_folds(
    indices: np.ndarray,
    device_types: List[str],
    t_k: np.ndarray,
    n_folds: int,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    strata: Dict[Tuple[str, int], List[int]] = {}
    for idx in indices.tolist():
        dt = str(device_types[int(idx)])
        tc = int(round(float(t_k[int(idx)] - cfg.CELSIUS_TO_KELVIN)))
        strata.setdefault((dt, tc), []).append(int(idx))

    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for key in sorted(strata.keys()):
        items = strata[key]
        rng.shuffle(items)
        for i, item in enumerate(items):
            folds[i % n_folds].append(item)

    out: List[np.ndarray] = []
    for f in folds:
        arr = np.asarray(sorted(f), dtype=int)
        out.append(arr)
    return out


def _split_fit_cal(
    train_idx: np.ndarray,
    device_types: List[str],
    t_k: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    inner = _grouped_folds(train_idx, device_types, t_k, n_folds=5, seed=seed)
    cal_idx = inner[0]
    fit_idx = np.concatenate([inner[i] for i in range(1, len(inner))])

    if len(cal_idx) == 0:
        cal_idx = fit_idx[: max(1, len(fit_idx) // 5)]
        fit_idx = fit_idx[len(cal_idx) :]
    if len(fit_idx) == 0:
        fit_idx = cal_idx[: max(1, len(cal_idx) // 2)]
        cal_idx = cal_idx[len(fit_idx) :]
    return fit_idx, cal_idx


def _predict_indices(model, dataset: Dict, indices: np.ndarray, train_mod, eval_mod, device):
    import torch
    from torch.utils.data import DataLoader

    ds = train_mod.DeviceDegradationDataset(dataset, indices)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=train_mod.collate_fn)

    preds, trues, masks, tks, times = [], [], [], [], []
    enc_inputs, x0_list = [], []
    with torch.no_grad():
        for b in dl:
            out = eval_mod.predict_from_prefix(
                model,
                b["enc_input"],
                b["x"],
                b["mask"],
                b["times_h"],
                b["T_K"],
                b["x0"],
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                device=device,
            )
            preds.append(out["x_pred"].cpu().numpy())
            trues.append(b["x"].cpu().numpy())
            masks.append(b["mask"].cpu().numpy())
            tks.append(b["T_K"].cpu().numpy())
            times.append(b["times_h"].cpu().numpy())
            enc_inputs.append(b["enc_input"].cpu().numpy())
            x0_list.append(b["x0"].cpu().numpy())

    return {
        "x_pred_norm": np.concatenate(preds, axis=0),
        "x_true_norm": np.concatenate(trues, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "T_K": np.concatenate(tks, axis=0),
        "times_h": np.concatenate(times, axis=0),
        "enc_input": np.concatenate(enc_inputs, axis=0),
        "x0": np.concatenate(x0_list, axis=0),
    }


def _subset(arr: np.ndarray, src_indices: np.ndarray, wanted_indices: np.ndarray) -> np.ndarray:
    pos = {int(idx): i for i, idx in enumerate(src_indices.tolist())}
    sel = [pos[int(idx)] for idx in wanted_indices.tolist()]
    return arr[np.asarray(sel, dtype=int)]


class _Stage4AResidualSampleModel:
    """
    Wrap Stage 4A observation-space residual generator under the same API used
    by grouped CV for stochastic residual models.
    """

    def __init__(self, mean_model, generator, stoch_mod, device, prefix_len: int):
        self.mean_model = mean_model
        self.generator = generator
        self.device = device
        self.prefix_len = prefix_len
        self._metric_model = stoch_mod.StochasticResidualModel()
        self._z_prefix_last = None
        self._x0 = None
        self._T_K = None
        self._log_t = None
        self.sigma_scale = 1.0
        self.sigma_scale_by_temp: Dict[int, float] = {}
        self.sigma_scale_by_feat_temp: Dict[int, np.ndarray] = {}  # temp_c -> (F,) scales

    @staticmethod
    def _temp_bucket_from_kelvin(T_K: np.ndarray) -> np.ndarray:
        t_c = np.round(np.asarray(T_K, dtype=float) - cfg.CELSIUS_TO_KELVIN).astype(int)
        return t_c

    def _set_encoder_outputs(
        self,
        enc_in: np.ndarray,
        mask: np.ndarray,
        x0: np.ndarray,
        T_K: np.ndarray,
        times_h: np.ndarray,
        prefix_len: Optional[int] = None,
    ):
        import torch

        if prefix_len is None:
            prefix_len = self.prefix_len

        enc_t = torch.from_numpy(enc_in).float().to(self.device)
        mask_t = torch.from_numpy(mask).bool().to(self.device)
        x0_t = torch.from_numpy(x0).float().to(self.device)
        T_K_t = torch.from_numpy(T_K).float().to(self.device)
        times_t = torch.from_numpy(times_h).float().to(self.device)

        with torch.no_grad():
            z_prefix, _ = self.mean_model.encoder(enc_t[:, :prefix_len, :], mask_t[:, :prefix_len])

        self._z_prefix_last = z_prefix[:, -1, :]
        self._x0 = x0_t
        self._T_K = T_K_t
        self._log_t = torch.log1p(times_t[:, prefix_len - 1].clamp(min=0.0))

    def sample_trajectories(
        self,
        x_mean: np.ndarray,
        T_K: np.ndarray,
        times_h: np.ndarray,
        mask: np.ndarray,
        n_samples: int = None,
        prefix_len: int = None,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        import torch

        del mask, rng  # Retained for API compatibility.

        if self._z_prefix_last is None:
            raise RuntimeError("Call _set_encoder_outputs(...) before sample_trajectories().")

        if n_samples is None:
            n_samples = cfg.STOCH_RESIDUAL_N_SAMPLES
        if prefix_len is None:
            prefix_len = self.prefix_len

        N, T, F = x_mean.shape
        T_future = max(0, T - prefix_len)

        samples = np.repeat(x_mean[None, :, :, :], n_samples, axis=0)
        if T_future <= 0:
            return samples

        # Pass future time schedule to log-time AR(1) generators if available
        times_future_t = None
        if times_h is not None and hasattr(self.generator, 'LOG10_T_REF'):
            times_future_t = torch.from_numpy(
                np.asarray(times_h, dtype=np.float32)[:, prefix_len:]
            ).to(self.device)

        self.generator.eval()
        with torch.no_grad():
            deltas = self.generator.sample_n(
                self._z_prefix_last,
                self._T_K,
                self._x0,
                self._log_t,
                n_samples,
                T_future=T_future,
                **({'times_future': times_future_t} if times_future_t is not None else {}),
            )
        deltas_np = deltas.cpu().numpy()

        # Detect Stage 4C stable-only generator: deltas cover only STABLE_FEAT_INDICES
        _stable_idx = getattr(self.generator, 'STABLE_INDICES', None)

        if _stable_idx is not None:
            # Stage 4C: apply residuals only to stable feature columns.
            # Leakage columns (IDLeak, IGLeak) remain at x_mean (deterministic).
            col = np.array(_stable_idx)          # [0, 1, 2, 3]
            t_bucket_arr = self._temp_bucket_from_kelvin(T_K)
            F_s = deltas_np.shape[-1]            # 4

            if self.sigma_scale_by_feat_temp:
                for fi_loc, fi_glob in enumerate(col):
                    per_dev_fi = np.array([
                        float(self.sigma_scale_by_feat_temp.get(int(t), np.ones(F_s) * self.sigma_scale)[fi_loc])
                        for t in t_bucket_arr
                    ])  # (N,)
                    samples[:, :, prefix_len:, fi_glob] += deltas_np[:, :, :, fi_loc] * per_dev_fi[None, :, None]
            elif self.sigma_scale_by_temp:
                per_dev = np.array([float(self.sigma_scale_by_temp.get(int(t), self.sigma_scale)) for t in t_bucket_arr])
                for fi_loc, fi_glob in enumerate(col):
                    samples[:, :, prefix_len:, fi_glob] += deltas_np[:, :, :, fi_loc] * per_dev[None, :, None]
            else:
                for fi_loc, fi_glob in enumerate(col):
                    samples[:, :, prefix_len:, fi_glob] += float(self.sigma_scale) * deltas_np[:, :, :, fi_loc]
            # Leakage columns stay at x_mean (already set in samples)
        elif self.sigma_scale_by_feat_temp:
            # Feature × temperature scales: multiply deltas by an (N, F) matrix.
            t_bucket = self._temp_bucket_from_kelvin(T_K)
            F_samp = deltas_np.shape[-1]
            feat_temp_scales = np.stack(
                [self.sigma_scale_by_feat_temp.get(int(t), np.ones(F_samp, dtype=float) * self.sigma_scale)
                 for t in t_bucket],
                axis=0,
            )  # (N, F)
            samples[:, :, prefix_len:, :] += deltas_np * feat_temp_scales[None, :, None, :]
        elif self.sigma_scale_by_temp:
            t_bucket = self._temp_bucket_from_kelvin(T_K)
            per_dev = np.array([float(self.sigma_scale_by_temp.get(int(t), self.sigma_scale)) for t in t_bucket], dtype=float)
            samples[:, :, prefix_len:, :] += deltas_np * per_dev[None, :, None, None]
        else:
            samples[:, :, prefix_len:, :] += float(self.sigma_scale) * deltas_np
        samples[:, :, :prefix_len, :] = x_mean[:, :prefix_len, :]
        return samples

    def calibrate_sigma_scale(
        self,
        x_cal_mean: np.ndarray,
        x_cal_true: np.ndarray,
        mask_cal: np.ndarray,
        T_K_cal: np.ndarray,
        times_h_cal: np.ndarray,
        n_samples: int,
        prefix_len: int,
        target_coverage: float,
        scale_grid: List[float],
        crpss_drop_tol: float = 0.01,
    ) -> Dict[str, float]:
        if self._z_prefix_last is None:
            raise RuntimeError("Call _set_encoder_outputs(...) before calibrate_sigma_scale().")

        # Global calibration mode uses one shared scale.
        self.sigma_scale_by_temp = {}

        chosen_scale = 1.0
        best_obj = float("inf")
        best_width = float("inf")
        best_crpss = -float("inf")
        base_crpss = None
        coverage_at_best = float("nan")

        for s in scale_grid:
            self.sigma_scale = float(s)
            samples = self.sample_trajectories(
                x_cal_mean,
                T_K_cal,
                times_h_cal,
                mask_cal,
                n_samples=n_samples,
                prefix_len=prefix_len,
            )
            metrics = self.compute_metrics(
                x_cal_true,
                samples,
                mask_cal,
                T_K=T_K_cal,
                prefix_len=prefix_len,
                x_mean=x_cal_mean,
                times_h=times_h_cal,
            )
            cov = float(metrics.get("coverage_90_overall", np.nan))
            width = float(metrics.get("width_90_overall", np.nan))
            crpss = float(metrics.get("crpss_overall", np.nan))

            if abs(float(s) - 1.0) < 1e-9:
                base_crpss = crpss

            if np.isnan(cov) or np.isnan(width) or np.isnan(crpss):
                continue

            crpss_floor = -float("inf") if base_crpss is None else (base_crpss - float(crpss_drop_tol))
            if crpss < crpss_floor:
                continue

            obj = abs(cov - float(target_coverage))
            if (obj < best_obj) or (abs(obj - best_obj) <= 1e-9 and width < best_width):
                best_obj = obj
                best_width = width
                chosen_scale = float(s)
                best_crpss = crpss
                coverage_at_best = cov

        if not np.isfinite(best_obj):
            for s in scale_grid:
                self.sigma_scale = float(s)
                samples = self.sample_trajectories(
                    x_cal_mean,
                    T_K_cal,
                    times_h_cal,
                    mask_cal,
                    n_samples=n_samples,
                    prefix_len=prefix_len,
                )
                metrics = self.compute_metrics(
                    x_cal_true,
                    samples,
                    mask_cal,
                    T_K=T_K_cal,
                    prefix_len=prefix_len,
                    x_mean=x_cal_mean,
                    times_h=times_h_cal,
                )
                crpss = float(metrics.get("crpss_overall", np.nan))
                cov = float(metrics.get("coverage_90_overall", np.nan))
                if np.isnan(crpss):
                    continue
                if crpss > best_crpss:
                    best_crpss = crpss
                    chosen_scale = float(s)
                    coverage_at_best = cov

        self.sigma_scale = chosen_scale
        return {
            "sigma_scale": float(chosen_scale),
            "base_crpss": float(base_crpss) if base_crpss is not None else float("nan"),
            "best_crpss": float(best_crpss),
            "coverage_90": float(coverage_at_best),
        }

    def calibrate_sigma_scale_by_feat_temp(
        self,
        x_cal_mean: np.ndarray,
        x_cal_true: np.ndarray,
        mask_cal: np.ndarray,
        T_K_cal: np.ndarray,
        times_h_cal: np.ndarray,
        n_samples: int,
        prefix_len: int,
        target_coverage: float,
        scale_grid: List[float],
        crpss_drop_tol: float = 0.01,
        leakage_scale_grid: Optional[List[float]] = None,
    ) -> Dict:
        """Calibrate a separate sigma scale for each (feature, temperature) pair.

        For each temperature bucket, generate raw deltas once (scale=1.0), then
        analytically find the best per-feature scale from the grid by checking
        per-feature coverage on the calibration set.
        Stores results in ``self.sigma_scale_by_feat_temp`` (Dict[temp_c, np.ndarray(F,)]).

        ``leakage_scale_grid`` (optional) overrides ``scale_grid`` for IDLeak and IGLeak
        features, allowing a wider search range (e.g. up to 2.5).
        """
        F = x_cal_mean.shape[-1]
        t_bucket = self._temp_bucket_from_kelvin(T_K_cal)

        # ── Generate raw deltas with scale = 1.0 (no temperature weighting) ──
        saved_scale = self.sigma_scale
        saved_by_temp = self.sigma_scale_by_temp.copy()
        saved_by_feat_temp = {k: v.copy() for k, v in self.sigma_scale_by_feat_temp.items()}
        self.sigma_scale = 1.0
        self.sigma_scale_by_temp = {}
        self.sigma_scale_by_feat_temp = {}

        smp_raw = self.sample_trajectories(
            x_cal_mean, T_K_cal, times_h_cal, mask_cal,
            n_samples=n_samples, prefix_len=prefix_len,
        )  # (S, N, T, F)
        raw_deltas = smp_raw - x_cal_mean[np.newaxis, :, :, :]  # (S, N, T, F)

        # Restore previous state while we search
        self.sigma_scale = saved_scale
        self.sigma_scale_by_temp = saved_by_temp
        self.sigma_scale_by_feat_temp = saved_by_feat_temp

        future_mask = mask_cal.copy().astype(bool)
        future_mask[:, :prefix_len] = False

        feat_temp_result: Dict[int, np.ndarray] = {}
        for temp_c in sorted(set(t_bucket.tolist())):
            idx_t = np.where(t_bucket == int(temp_c))[0]
            if len(idx_t) == 0:
                continue
            scales_ft = np.ones(F, dtype=float)

            for fi in range(F):
                valid_ft = future_mask[idx_t] & ~np.isnan(x_cal_true[idx_t, :, fi])
                if valid_ft.sum() < 5:
                    continue

                # Gather all valid (time) points for (feature fi, temperature temp_c)
                y_lst, xm_lst, d_lst = [], [], []
                for dev_n in idx_t:
                    t_ok = future_mask[dev_n] & ~np.isnan(x_cal_true[dev_n, :, fi])
                    if t_ok.sum() == 0:
                        continue
                    y_lst.append(x_cal_true[dev_n, t_ok, fi])       # (m_n,)
                    xm_lst.append(x_cal_mean[dev_n, t_ok, fi])       # (m_n,)
                    d_lst.append(raw_deltas[:, dev_n, t_ok, fi])     # (S, m_n)
                if not y_lst:
                    continue
                y = np.concatenate(y_lst)                            # (M,)
                x_m = np.concatenate(xm_lst)                         # (M,)
                d = np.concatenate(d_lst, axis=1)                    # (S, M)

                best_s, best_obj = 1.0, float("inf")
                # Use leakage_scale_grid for IDLeak / IGLeak if provided
                _leakage_idxs = [cfg.IDLEAK_DECODER_ROW, cfg.IGLEAK_DECODER_ROW]
                fi_grid = (
                    leakage_scale_grid
                    if (leakage_scale_grid is not None and fi in _leakage_idxs)
                    else scale_grid
                )
                for s in fi_grid:
                    q05 = np.percentile(x_m[None, :] + s * d, 5, axis=0)
                    q95 = np.percentile(x_m[None, :] + s * d, 95, axis=0)
                    cov = float(np.mean((y >= q05) & (y <= q95)))
                    obj = abs(cov - target_coverage)
                    if obj < best_obj:
                        best_obj = obj
                        best_s = float(s)
                scales_ft[fi] = best_s

            feat_temp_result[int(temp_c)] = scales_ft
            import config as _cfg
            import logging as _log
            _log.getLogger(__name__).info(
                "  sigma_scale_by_feat_temp[%d°C] = {%s}",
                temp_c,
                ", ".join(f"{_cfg.FEATURES[fi]}:{scales_ft[fi]:.2f}" for fi in range(F)),
            )

        # Commit and clear lower-priority scales
        self.sigma_scale_by_feat_temp = feat_temp_result
        self.sigma_scale_by_temp = {}
        self.sigma_scale = 1.0  # fallback for unknown temps
        return {"sigma_scale_by_feat_temp": {tc: list(sc) for tc, sc in feat_temp_result.items()}}

    def calibrate_sigma_scale_by_temp(
        self,
        x_cal_mean: np.ndarray,
        x_cal_true: np.ndarray,
        mask_cal: np.ndarray,
        T_K_cal: np.ndarray,
        times_h_cal: np.ndarray,
        n_samples: int,
        prefix_len: int,
        target_coverage: float,
        scale_grid: List[float],
        crpss_drop_tol: float = 0.01,
    ) -> Dict[str, float]:
        t_bucket = self._temp_bucket_from_kelvin(T_K_cal)
        per_temp: Dict[int, float] = {}
        per_temp_cov: Dict[int, float] = {}
        per_temp_crpss: Dict[int, float] = {}

        # Calibrate one scalar per temperature bucket using only that subgroup.
        for temp_c in sorted(set(t_bucket.tolist())):
            idx = np.where(t_bucket == int(temp_c))[0]
            if idx.size == 0:
                continue
            info = self.calibrate_sigma_scale(
                x_cal_mean=x_cal_mean[idx],
                x_cal_true=x_cal_true[idx],
                mask_cal=mask_cal[idx],
                T_K_cal=T_K_cal[idx],
                times_h_cal=times_h_cal[idx],
                n_samples=n_samples,
                prefix_len=prefix_len,
                target_coverage=target_coverage,
                scale_grid=scale_grid,
                crpss_drop_tol=crpss_drop_tol,
            )
            per_temp[int(temp_c)] = float(info["sigma_scale"])
            per_temp_cov[int(temp_c)] = float(info.get("coverage_90", np.nan))
            per_temp_crpss[int(temp_c)] = float(info.get("best_crpss", np.nan))

        self.sigma_scale_by_temp = per_temp
        self.sigma_scale = float(np.mean(list(per_temp.values()))) if per_temp else 1.0
        return {
            "sigma_scale": float(self.sigma_scale),
            "base_crpss": float("nan"),
            "best_crpss": float(np.nanmean(list(per_temp_crpss.values()))) if per_temp_crpss else float("nan"),
            "coverage_90": float(np.nanmean(list(per_temp_cov.values()))) if per_temp_cov else float("nan"),
        }

    def compute_metrics(self, x_true: np.ndarray, samples: np.ndarray, mask: np.ndarray, **kwargs) -> Dict:
        """Delegate to stochastic metric model.
        For Stage 4C (stable-only generator), temporarily restrict cfg.FEATURES to
        the stable subset so compute_metrics only iterates over those 4 features.
        Also patches _LEAKAGE_FEATURES to empty so no leakage-specific logic runs.
        """
        _stable_idx = getattr(self.generator, 'STABLE_INDICES', None)
        if _stable_idx is not None:
            import config as _cfg
            import sys as _sys
            orig_features  = _cfg.FEATURES[:]
            _cfg.FEATURES  = [orig_features[i] for i in _stable_idx]
            # Also patch the stochastic residual module's _LEAKAGE_FEATURES constant
            _stoch_mod = _sys.modules.get("_pi_stoch_cv") or self._metric_model.__class__.__module__
            _stoch = _sys.modules.get("_pi_stoch_cv")
            orig_leak = None
            if _stoch is not None and hasattr(_stoch, '_LEAKAGE_FEATURES'):
                orig_leak = _stoch._LEAKAGE_FEATURES
                _stoch._LEAKAGE_FEATURES = ()
            kw = dict(kwargs)
            if 'x_mean' in kw and kw['x_mean'] is not None:
                kw['x_mean'] = kw['x_mean'][:, :, _stable_idx]
            try:
                result = self._metric_model.compute_metrics(
                    x_true[:, :, _stable_idx],
                    samples[:, :, :, _stable_idx],
                    mask,
                    **kw,
                )
            finally:
                _cfg.FEATURES = orig_features
                if _stoch is not None and orig_leak is not None:
                    _stoch._LEAKAGE_FEATURES = orig_leak
            return result
        return self._metric_model.compute_metrics(x_true, samples, mask, **kwargs)

    @staticmethod
    def compute_coverage_variants(
        x_true: np.ndarray,
        samples: np.ndarray,
        mask: np.ndarray,
        percentile: float = 90.0,
        prefix_len: int = 0,
        eval_feat_indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compute three distinct Cov90 definitions using a unified `covered` array.

        ``eval_feat_indices`` (optional): evaluate coverage only on these feature
        columns. Use this for Stage 4C to skip leakage features.
        All three metrics share the same valid-point set:
          - future only  (time >= prefix_len)
          - mask == True
          - x_true is not NaN
        """
        # Restrict to requested features before any computation
        if eval_feat_indices is not None:
            x_true  = x_true[:, :, eval_feat_indices]
            samples = samples[:, :, :, eval_feat_indices]
        S, N, T, F = samples.shape
        q_low  = (100.0 - percentile) / 2.0
        q_high = 100.0 - q_low

        # PI bounds
        pi_low  = np.percentile(samples, q_low,  axis=0)  # (N, T, F)
        pi_high = np.percentile(samples, q_high, axis=0)  # (N, T, F)

        # Future mask: exclude prefix region
        future_mask = mask.copy().astype(bool)
        if prefix_len > 0:
            future_mask[:, :prefix_len] = False

        # Valid triplet mask: future + not-NaN per feature
        valid_3d = (future_mask[:, :, None] & ~np.isnan(x_true))  # (N, T, F)

        # In-PI array (False for invalid positions by construction since valid_3d gates)
        in_pi = (
            (x_true >= pi_low) &
            (x_true <= pi_high) &
            valid_3d
        )  # (N, T, F)

        n_valid = int(np.sum(valid_3d))

        # 1. Pointwise
        pointwise_cov = float(np.sum(in_pi)) / max(n_valid, 1) if n_valid > 0 else float("nan")

        # 2. Device-average
        device_coverages = []
        for n in range(N):
            n_v = int(np.sum(valid_3d[n]))
            if n_v > 0:
                device_coverages.append(float(np.sum(in_pi[n])) / n_v)
        device_avg_cov = float(np.nanmean(device_coverages)) if device_coverages else float("nan")

        # 3. Simultaneous trajectory: all valid (time, feature) of device must be in PI
        traj_covered = []
        for n in range(N):
            if np.any(valid_3d[n]):
                traj_covered.append(float(np.all(in_pi[n][valid_3d[n]])))
        trajectory_cov = float(np.nanmean(traj_covered)) if traj_covered else float("nan")

        return {
            f"coverage_{int(percentile)}_pointwise":    pointwise_cov,
            f"coverage_{int(percentile)}_device_avg":   device_avg_cov,
            f"coverage_{int(percentile)}_simultaneous": trajectory_cov,
        }


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser(description="Grouped 5-fold x multi-seed stability validation")
    p.add_argument("--checkpoint-stage3", default=os.path.join(cfg.CHECKPOINT_DIR, "stage3_best.pt"))
    p.add_argument("--checkpoint-stage4", default=os.path.join(cfg.CHECKPOINT_DIR, "stage4_best.pt"))
    p.add_argument("--checkpoint-stage4b", type=str, default="")
    p.add_argument("--split-scope", choices=["all", "test", "trainval"], default="all")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--n-samples", type=int, default=40)
    p.add_argument("--stage4b-post-calibration", action="store_true")
    p.add_argument("--stage4b-tempwise-calibration", action="store_true")
    p.add_argument("--stage4b-target-coverage", type=float, default=0.90)
    p.add_argument("--stage4b-crpss-drop-tol", type=float, default=0.01)
    p.add_argument("--stage4b-scale-grid", type=str, default="1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.35")
    p.add_argument(
        "--stage4b-leakage-scale-grid",
        type=str,
        default="1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,2.00,2.25,2.50",
        help="Scale grid for IDLeak/IGLeak in feat-temp calibration (wider than default).",
    )
    p.add_argument(
        "--stage4b-feat-temp-calibration",
        action="store_true",
        help="Calibrate a separate sigma scale for each (feature, temperature) pair "
             "instead of the global or tempwise scalar.",
    )
    p.add_argument("--output-dir", type=str, default=os.path.join(cfg.RESULTS_DIR, "grouped_cv_stability"))
    p.add_argument(
        "--save-diag",
        action="store_true",
        help="Save raw tensors (x_true_norm, x_pred_norm, samples, mask, T_K) from the "
             "first Stage4B fold evaluation to diag_state.pkl for diagnostic analysis.",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = _parse_seeds(args.seeds)

    mods = {
        "prep": _load_module("_pi_prep_cv", "01_data_preprocessing.py"),
        "ode": _load_module("_pi_ode_cv", "02_physics_latent.py"),
        "enc": _load_module("_pi_enc_cv", "03_model_encoder.py"),
        "dec": _load_module("_pi_dec_cv", "04_model_decoder.py"),
        "gen": _load_module("_pi_gen_cv", "05_model_generator.py"),
        "disc": _load_module("_pi_disc_cv", "06_model_discriminator.py"),
        "train": _load_module("_pi_train_cv", "08_training.py"),
        "eval": _load_module("_pi_eval_cv", "09_evaluation.py"),
        "stoch": _load_module("_pi_stoch_cv", "10_stochastic_residual.py"),
        "stage4a": _load_module("_pi_stage4a_cv", "13_stage4a_residual_generator.py"),
        "stage4b": _load_module("_pi_stage4b_cv", "14_stage4b_ar1_guided_generator.py"),
    }

    dataset = mods["prep"].load_dataset()
    eval_indices = _select_eval_indices(dataset, args.split_scope)

    if not os.path.exists(args.checkpoint_stage3):
        raise FileNotFoundError(f"Missing stage3 checkpoint: {args.checkpoint_stage3}")

    import torch

    device = torch.device("cpu")
    model_cls = _build_model(mods)
    model3 = mods["eval"].load_model(args.checkpoint_stage3, model_cls).to(device)
    model3.eval()

    # Predict once for all eval indices, then slice per-fold.
    all_pred = _predict_indices(model3, dataset, eval_indices, mods["train"], mods["eval"], device)
    ns = dataset["norm_stats"]
    x_pred_deg_all = mods["eval"].denormalize_x(all_pred["x_pred_norm"], ns)
    x_true_deg_all = mods["eval"].denormalize_x(all_pred["x_true_norm"], ns)

    # Optional stage4 model.
    stage4_mode = "none"
    stage4_audit: Dict = {
        "checkpoint_path": args.checkpoint_stage4,
        "checkpoint_exists": bool(os.path.exists(args.checkpoint_stage4)),
    }
    model4 = None
    stage4a_generator = None
    stage4a_sampler = None
    stage4b_generator = None
    stage4b_sampler = None

    if args.checkpoint_stage4b:
        stage4_audit["checkpoint_stage4b_path"] = args.checkpoint_stage4b
        stage4_audit["checkpoint_stage4b_exists"] = bool(os.path.exists(args.checkpoint_stage4b))

    if args.checkpoint_stage4b and os.path.exists(args.checkpoint_stage4b):
        ckpt4b = torch.load(args.checkpoint_stage4b, map_location=device)
        # Auto-detect Stage 4C (stable-only) by output layer size
        _sd4b = ckpt4b["state_dict"]
        _out_sz = _sd4b.get("net.5.bias", _sd4b.get("net.6.bias", None))
        _is_stable4c = (
            _out_sz is not None
            and _out_sz.numel() == 2 * mods["stage4b"].N_STABLE_FEATURES
        )
        if _is_stable4c:
            print("[grouped-cv] Detected Stage 4C checkpoint (stable features only)")
            stage4b_generator = mods["stage4b"].AR1GuidedResidualGeneratorStable().to(device)
        else:
            stage4b_generator = mods["stage4b"].AR1GuidedResidualGenerator().to(device)
        missing, unexpected = stage4b_generator.load_state_dict(_sd4b, strict=False)
        if missing:
            print(f"[stage4b] Missing keys in checkpoint (new params): {missing}")
        if unexpected:
            print(f"[stage4b] Unexpected keys in checkpoint: {unexpected}")
        stage4b_generator.eval()
        stage4b_sampler = _Stage4AResidualSampleModel(
            mean_model=model3,
            generator=stage4b_generator,
            stoch_mod=mods["stoch"],
            device=device,
            prefix_len=cfg.STAGE3_PREFIX_LEN,
        )
        stage4_mode = "stage4b_residual"
        stage4_audit.update({
            "checkpoint_stage_or_type": "stage4b_residual",
            "model_class": type(stage4b_generator).__name__,
            "generator_output_dimension": int(getattr(stage4b_generator, "n_features", -1)),
            "uses_z0_perturbation": False,
            "uses_generated_alpha": False,
            "uses_observation_residual_generator": True,
            "best_stage4b_epoch": int(ckpt4b.get("epoch", -1)),
            "best_stage4b_val_crps": float(ckpt4b.get("val_crps", np.nan)),
        })

    if stage4_mode == "none" and os.path.exists(args.checkpoint_stage4):
        raw_ckpt = torch.load(args.checkpoint_stage4, map_location=device)
        stage4_audit["checkpoint_keys"] = sorted(raw_ckpt.keys()) if isinstance(raw_ckpt, dict) else []

        is_stage4a = (
            isinstance(raw_ckpt, dict)
            and "state_dict" in raw_ckpt
            and "model_state" not in raw_ckpt
            and "model_state_dict" not in raw_ckpt
        )

        if is_stage4a:
            stage4_mode = "stage4a_residual"
            stage4a_generator = mods["stage4a"].ResidualLatentInnovationGenerator().to(device)
            stage4a_generator.load_state_dict(raw_ckpt["state_dict"], strict=True)
            stage4a_generator.eval()
            stage4a_sampler = _Stage4AResidualSampleModel(
                mean_model=model3,
                generator=stage4a_generator,
                stoch_mod=mods["stoch"],
                device=device,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
            )
            stage4_audit.update({
                "checkpoint_stage_or_type": "stage4a_residual",
                "model_class": type(stage4a_generator).__name__,
                "generator_output_dimension": int(getattr(stage4a_generator, "n_features", -1)),
                "uses_z0_perturbation": False,
                "uses_generated_alpha": False,
                "uses_observation_residual_generator": True,
                "best_stage4a_epoch": int(raw_ckpt.get("epoch", -1)),
            })
        else:
            stage4_mode = "legacy_stage4"
            model4 = mods["eval"].load_model(args.checkpoint_stage4, model_cls).to(device)
            model4.eval()
            stage4_audit.update({
                "checkpoint_stage_or_type": "legacy_stage4_generator",
                "model_class": type(model4.generator).__name__,
                "generator_output_dimension": int(getattr(model4.generator, "z_dim", -1)),
                "uses_z0_perturbation": True,
                "uses_generated_alpha": True,
                "uses_observation_residual_generator": False,
                "best_stage4a_epoch": -1,
            })

    print("Stage4 checkpoint audit:")
    print(f"  checkpoint path: {stage4_audit['checkpoint_path']}")
    print(f"  checkpoint exists: {stage4_audit['checkpoint_exists']}")
    print(f"  detected mode: {stage4_mode}")
    if stage4_audit.get("checkpoint_keys"):
        print(f"  checkpoint keys: {stage4_audit['checkpoint_keys']}")
    if stage4_mode != "none":
        print(f"  model class: {stage4_audit.get('model_class')}")
        print(f"  stage/type: {stage4_audit.get('checkpoint_stage_or_type')}")
        print(f"  generator output dimension: {stage4_audit.get('generator_output_dimension')}")
        print(f"  uses z0 perturbation: {stage4_audit.get('uses_z0_perturbation')}")
        print(f"  uses generated alpha: {stage4_audit.get('uses_generated_alpha')}")
        print(f"  uses observation residual generator: {stage4_audit.get('uses_observation_residual_generator')}")
        print(f"  best Stage4A epoch: {stage4_audit.get('best_stage4a_epoch')}")

    device_types = dataset["device_types"]
    t_k_full = dataset["T_K"]

    fold_rows: List[Dict] = []

    for seed in seeds:
        _set_seed(seed)
        folds = _grouped_folds(eval_indices, device_types, t_k_full, n_folds=args.folds, seed=seed)

        for fold_id in range(args.folds):
            val_idx = folds[fold_id]
            tr_idx = np.concatenate([folds[j] for j in range(args.folds) if j != fold_id])
            fit_idx, cal_idx = _split_fit_cal(tr_idx, device_types, t_k_full, seed + fold_id)

            x_fit_pred = _subset(x_pred_deg_all, eval_indices, fit_idx)
            x_fit_true = _subset(x_true_deg_all, eval_indices, fit_idx)
            m_fit = _subset(all_pred["mask"], eval_indices, fit_idx)
            tk_fit = _subset(all_pred["T_K"], eval_indices, fit_idx)
            t_fit = _subset(all_pred["times_h"], eval_indices, fit_idx)

            x_cal_pred = _subset(x_pred_deg_all, eval_indices, cal_idx)
            x_cal_true = _subset(x_true_deg_all, eval_indices, cal_idx)
            m_cal = _subset(all_pred["mask"], eval_indices, cal_idx)
            tk_cal = _subset(all_pred["T_K"], eval_indices, cal_idx)
            t_cal = _subset(all_pred["times_h"], eval_indices, cal_idx)
            # Normalised versions (for Stage4B which operates in normalised space)
            x_cal_pred_norm = _subset(all_pred["x_pred_norm"], eval_indices, cal_idx)
            x_cal_true_norm = _subset(all_pred["x_true_norm"], eval_indices, cal_idx)

            x_val_pred = _subset(x_pred_deg_all, eval_indices, val_idx)
            x_val_true = _subset(x_true_deg_all, eval_indices, val_idx)
            m_val = _subset(all_pred["mask"], eval_indices, val_idx)
            tk_val = _subset(all_pred["T_K"], eval_indices, val_idx)
            t_val = _subset(all_pred["times_h"], eval_indices, val_idx)
            # Normalised versions (for Stage4B which operates in normalised space)
            x_val_pred_norm = _subset(all_pred["x_pred_norm"], eval_indices, val_idx)
            x_val_true_norm = _subset(all_pred["x_true_norm"], eval_indices, val_idx)

            # Baseline 1: Independent Gaussian
            mdl_gauss = mods["stoch"].IndependentGaussianResidualModel()
            mdl_gauss.fit(x_fit_pred - x_fit_true, m_fit, prefix_len=cfg.STAGE3_PREFIX_LEN)
            mdl_gauss.calibrate_sigma_scale(
                x_cal_pred, x_cal_true, tk_cal, t_cal, m_cal,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                n_samples=min(30, args.n_samples),
            )
            smp_g = mdl_gauss.sample_trajectories(
                x_val_pred, tk_val, t_val, m_val,
                n_samples=args.n_samples,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                rng=np.random.default_rng(seed + 1000 + fold_id),
            )
            met_g = mdl_gauss.compute_metrics(
                x_val_true, smp_g, m_val,
                T_K=tk_val,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                x_mean=x_val_pred,
                times_h=t_val,
            )

            # Baseline 2: AR(1) stochastic residual
            mdl_ar1 = mods["stoch"].StochasticResidualModel()
            mdl_ar1.fit(
                x_fit_pred - x_fit_true,
                tk_fit,
                t_fit,
                m_fit,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                x_true=x_fit_true,
            )
            mdl_ar1.calibrate(
                x_fit_pred - x_fit_true,
                m_fit,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                x_true_train=x_fit_true,
            )
            mdl_ar1.calibrate_sigma_scale(
                x_cal_pred, x_cal_true, tk_cal, t_cal, m_cal,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                n_samples=min(30, args.n_samples),
            )
            smp_a = mdl_ar1.sample_trajectories(
                x_val_pred, tk_val, t_val, m_val,
                n_samples=args.n_samples,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                rng=np.random.default_rng(seed + 2000 + fold_id),
            )
            met_a = mdl_ar1.compute_metrics(
                x_val_true, smp_a, m_val,
                T_K=tk_val,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                x_mean=x_val_pred,
                times_h=t_val,
            )

            def _append_row(
                model_name: str,
                metrics: Dict,
                stage4_sigma_scale: float = np.nan,
                stage4_sigma_calibrated: bool = False,
                stage4_sigma_mode: str = "none",
                stage4_sigma_map: str = "",
                cov90_variants: Dict[str, float] = None,
            ):
                if cov90_variants is None:
                    cov90_variants = {}
                fold_rows.append({
                    "seed": seed,
                    "fold": fold_id,
                    "model": model_name,
                    "n_val_devices": int(len(val_idx)),
                    "crpss_overall": float(metrics.get("crpss_overall", np.nan)),
                    "mace": float(metrics.get("reliability_mace", np.nan)),
                    "coverage_50_overall": float(metrics.get("coverage_50_overall", np.nan)),
                    "coverage_80_overall": float(metrics.get("coverage_80_overall", np.nan)),
                    "coverage_90_overall": float(metrics.get("coverage_90_overall", np.nan)),
                    "coverage_90_pointwise": float(cov90_variants.get("coverage_90_pointwise", np.nan)),
                    "coverage_90_device_avg": float(cov90_variants.get("coverage_90_device_avg", np.nan)),
                    "coverage_90_simultaneous": float(cov90_variants.get("coverage_90_simultaneous", np.nan)),
                    "width_50_overall": float(metrics.get("width_50_overall", np.nan)),
                    "width_80_overall": float(metrics.get("width_80_overall", np.nan)),
                    "width_90_overall": float(metrics.get("width_90_overall", np.nan)),
                    "w1_increments_overall": float(metrics.get("w1_increments_overall", np.nan)),
                    "stage4_sigma_scale": float(stage4_sigma_scale),
                    "stage4_sigma_calibrated": bool(stage4_sigma_calibrated),
                    "stage4_sigma_mode": str(stage4_sigma_mode),
                    "stage4_sigma_map": str(stage4_sigma_map),
                })

            _append_row("gaussian", met_g)
            _append_row("ar1", met_a)

            # Optional model 3: legacy stage4 generator
            if stage4_mode == "legacy_stage4" and model4 is not None:
                gen_m = mods["stoch"].GeneratorSampleModel(model4, device, prefix_len=cfg.STAGE3_PREFIX_LEN)
                enc_val = _subset(all_pred["enc_input"], eval_indices, val_idx)
                x0_val = _subset(all_pred["x0"], eval_indices, val_idx)
                gen_m._set_encoder_outputs(enc_val, m_val, x0_val, tk_val, t_val, prefix_len=cfg.STAGE3_PREFIX_LEN)
                smp_s4 = gen_m.sample_trajectories(
                    x_val_pred, tk_val, t_val, m_val,
                    n_samples=args.n_samples,
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                    rng=np.random.default_rng(seed + 3000 + fold_id),
                )
                met_s4 = gen_m.compute_metrics(
                    x_val_true, smp_s4, m_val,
                    T_K=tk_val,
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                    x_mean=x_val_pred,
                    times_h=t_val,
                )
                _append_row("stage4_legacy", met_s4)

            # Optional model 4: Stage 4A residual generator
            if stage4_mode == "stage4a_residual" and stage4a_sampler is not None:
                enc_val = _subset(all_pred["enc_input"], eval_indices, val_idx)
                x0_val = _subset(all_pred["x0"], eval_indices, val_idx)
                stage4a_sampler._set_encoder_outputs(
                    enc_val,
                    m_val,
                    x0_val,
                    tk_val,
                    t_val,
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                )
                smp_s4a = stage4a_sampler.sample_trajectories(
                    x_val_pred,
                    tk_val,
                    t_val,
                    m_val,
                    n_samples=args.n_samples,
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                    rng=np.random.default_rng(seed + 3000 + fold_id),
                )
                met_s4a = stage4a_sampler.compute_metrics(
                    x_val_true,
                    smp_s4a,
                    m_val,
                    T_K=tk_val,
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                    x_mean=x_val_pred,
                    times_h=t_val,
                )
                _append_row("stage4a", met_s4a)

            # Optional model 5: Stage 4B AR(1)-guided residual generator
            if stage4_mode == "stage4b_residual" and stage4b_sampler is not None:
                stage4b_sampler.sigma_scale = 1.0
                stage4b_sampler.sigma_scale_by_temp = {}
                stage4b_sampler.sigma_scale_by_feat_temp = {}
                cal_info = None
                sigma_mode = "none"

                if args.stage4b_post_calibration:
                    grid = [float(x.strip()) for x in str(args.stage4b_scale_grid).split(",") if x.strip()]
                    if not grid:
                        grid = [1.0]
                    enc_cal = _subset(all_pred["enc_input"], eval_indices, cal_idx)
                    x0_cal = _subset(all_pred["x0"], eval_indices, cal_idx)
                    # Set encoder context on the full cal set for global / feat-temp modes
                    stage4b_sampler._set_encoder_outputs(
                        enc_cal, m_cal, x0_cal, tk_cal, t_cal,
                        prefix_len=cfg.STAGE3_PREFIX_LEN,
                    )

                    if getattr(args, "stage4b_feat_temp_calibration", False):
                        sigma_mode = "feat_temp"
                        _lk_grid = [float(x.strip()) for x in str(args.stage4b_leakage_scale_grid).split(",") if x.strip()] if getattr(args, "stage4b_leakage_scale_grid", None) else None
                        cal_info = stage4b_sampler.calibrate_sigma_scale_by_feat_temp(
                            x_cal_mean=x_cal_pred_norm,
                            x_cal_true=x_cal_true_norm,
                            mask_cal=m_cal,
                            T_K_cal=tk_cal,
                            times_h_cal=t_cal,
                            n_samples=min(30, args.n_samples),
                            prefix_len=cfg.STAGE3_PREFIX_LEN,
                            target_coverage=float(args.stage4b_target_coverage),
                            scale_grid=grid,
                            crpss_drop_tol=float(args.stage4b_crpss_drop_tol),
                            leakage_scale_grid=_lk_grid,
                        )
                    elif args.stage4b_tempwise_calibration:
                        sigma_mode = "tempwise"
                        t_bucket_cal = np.round(tk_cal - cfg.CELSIUS_TO_KELVIN).astype(int)
                        scale_map = {}
                        for temp_c in sorted(set(t_bucket_cal.tolist())):
                            idx_t = np.where(t_bucket_cal == int(temp_c))[0]
                            if idx_t.size == 0:
                                continue
                            enc_cal_t = enc_cal[idx_t]
                            x0_cal_t = x0_cal[idx_t]
                            m_cal_t = m_cal[idx_t]
                            tk_cal_t = tk_cal[idx_t]
                            t_cal_t = t_cal[idx_t]
                            x_cal_pred_norm_t = x_cal_pred_norm[idx_t]
                            x_cal_true_norm_t = x_cal_true_norm[idx_t]

                            stage4b_sampler._set_encoder_outputs(
                                enc_cal_t, m_cal_t, x0_cal_t, tk_cal_t, t_cal_t,
                                prefix_len=cfg.STAGE3_PREFIX_LEN,
                            )
                            info_t = stage4b_sampler.calibrate_sigma_scale(
                                x_cal_mean=x_cal_pred_norm_t,
                                x_cal_true=x_cal_true_norm_t,
                                mask_cal=m_cal_t,
                                T_K_cal=tk_cal_t,
                                times_h_cal=t_cal_t,
                                n_samples=min(30, args.n_samples),
                                prefix_len=cfg.STAGE3_PREFIX_LEN,
                                target_coverage=float(args.stage4b_target_coverage),
                                scale_grid=grid,
                                crpss_drop_tol=float(args.stage4b_crpss_drop_tol),
                            )
                            scale_map[int(temp_c)] = float(info_t.get("sigma_scale", 1.0))

                        stage4b_sampler.sigma_scale_by_temp = scale_map
                        stage4b_sampler.sigma_scale = float(np.mean(list(scale_map.values()))) if scale_map else 1.0
                        cal_info = {
                            "sigma_scale": float(stage4b_sampler.sigma_scale),
                            "base_crpss": float("nan"),
                            "best_crpss": float("nan"),
                            "coverage_90": float("nan"),
                        }
                    else:
                        sigma_mode = "global"
                        cal_info = stage4b_sampler.calibrate_sigma_scale(
                            x_cal_mean=x_cal_pred_norm,
                            x_cal_true=x_cal_true_norm,
                            mask_cal=m_cal,
                            T_K_cal=tk_cal,
                            times_h_cal=t_cal,
                            n_samples=min(30, args.n_samples),
                            prefix_len=cfg.STAGE3_PREFIX_LEN,
                            target_coverage=float(args.stage4b_target_coverage),
                            scale_grid=grid,
                            crpss_drop_tol=float(args.stage4b_crpss_drop_tol),
                        )

                enc_val = _subset(all_pred["enc_input"], eval_indices, val_idx)
                x0_val = _subset(all_pred["x0"], eval_indices, val_idx)
                stage4b_sampler._set_encoder_outputs(
                    enc_val,
                    m_val,
                    x0_val,
                    tk_val,
                    t_val,
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                )
                # Stage4B works in normalised space — use normalised mean and truths
                smp_s4b = stage4b_sampler.sample_trajectories(
                    x_val_pred_norm,
                    tk_val,
                    t_val,
                    m_val,
                    n_samples=args.n_samples,
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                    rng=np.random.default_rng(seed + 4000 + fold_id),
                )
                met_s4b = stage4b_sampler.compute_metrics(
                    x_val_true_norm,
                    smp_s4b,
                    m_val,
                    T_K=tk_val,
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                    x_mean=x_val_pred_norm,
                    times_h=t_val,
                )
                # Compute three Cov90 variants for Stage4B / Stage4C
                _eval_fi = getattr(stage4b_sampler.generator, 'STABLE_INDICES', None)
                cov90_var = _Stage4AResidualSampleModel.compute_coverage_variants(
                    x_val_true_norm,
                    smp_s4b,
                    m_val,
                    percentile=90.0,
                    prefix_len=cfg.STAGE3_PREFIX_LEN,
                    eval_feat_indices=_eval_fi,   # None for Stage4B, [0,1,2,3] for Stage4C
                )
                # Save raw tensors for diagnostic analysis (first fold only)
                if args.save_diag and seed == seeds[0] and fold_id == 0:
                    diag_path = os.path.join(os.path.dirname(__file__), "diag_state.pkl")
                    with open(diag_path, "wb") as _f:
                        pickle.dump({
                            "x_true_norm": x_val_true_norm,
                            "x_pred_norm": x_val_pred_norm,
                            "samples":     smp_s4b,
                            "mask":        m_val,
                            "T_K":         tk_val,
                        }, _f)
                    print(f"[diag] Saved raw tensors to {diag_path}")
                _append_row(
                    "stage4b",
                    met_s4b,
                    stage4_sigma_scale=float(stage4b_sampler.sigma_scale),
                    stage4_sigma_calibrated=bool(cal_info is not None),
                    stage4_sigma_mode=sigma_mode,
                    stage4_sigma_map=(
                        stage4b_sampler.sigma_scale_by_feat_temp
                        if stage4b_sampler.sigma_scale_by_feat_temp
                        else stage4b_sampler.sigma_scale_by_temp
                    ),
                    cov90_variants=cov90_var,
                )

    # Aggregate summaries.
    metric_keys = [
        "crpss_overall",
        "mace",
        "coverage_50_overall",
        "coverage_80_overall",
        "coverage_90_overall",
        "coverage_90_pointwise",
        "coverage_90_device_avg",
        "coverage_90_simultaneous",
        "width_50_overall",
        "width_80_overall",
        "width_90_overall",
        "w1_increments_overall",
    ]

    def _group_stats(rows: List[Dict], group_fields: List[str]) -> List[Dict]:
        groups: Dict[Tuple, List[Dict]] = {}
        for r in rows:
            key = tuple(r[g] for g in group_fields)
            groups.setdefault(key, []).append(r)

        out = []
        for key, items in sorted(groups.items()):
            row = {g: key[i] for i, g in enumerate(group_fields)}
            for mk in metric_keys:
                vals = np.asarray([float(it[mk]) for it in items], dtype=float)
                row[f"{mk}_mean"] = float(np.nanmean(vals))
                row[f"{mk}_std"] = float(np.nanstd(vals))
            row["n_runs"] = len(items)
            out.append(row)
        return out

    model_summary = _group_stats(fold_rows, ["model"])
    seed_summary = _group_stats(fold_rows, ["seed", "model"])

    # AR1 win-rate diagnostics.
    runs_by_key: Dict[Tuple[int, int], Dict[str, Dict]] = {}
    for r in fold_rows:
        key = (int(r["seed"]), int(r["fold"]))
        runs_by_key.setdefault(key, {})[str(r["model"])] = r

    wins_vs_gauss, tot_vs_gauss = 0, 0
    wins_vs_stage4, tot_vs_stage4 = 0, 0
    present_models = {str(r["model"]) for r in fold_rows}
    stage4_label = (
        "stage4b"
        if "stage4b" in present_models
        else ("stage4a" if "stage4a" in present_models else ("stage4_legacy" if "stage4_legacy" in present_models else None))
    )
    for _, bundle in runs_by_key.items():
        if "ar1" in bundle and "gaussian" in bundle:
            tot_vs_gauss += 1
            if float(bundle["ar1"]["crpss_overall"]) > float(bundle["gaussian"]["crpss_overall"]):
                wins_vs_gauss += 1
        if stage4_label is not None and "ar1" in bundle and stage4_label in bundle:
            tot_vs_stage4 += 1
            if float(bundle["ar1"]["crpss_overall"]) > float(bundle[stage4_label]["crpss_overall"]):
                wins_vs_stage4 += 1

    stability = {
        "fold_rows": fold_rows,
        "model_summary": model_summary,
        "seed_summary": seed_summary,
        "seeds": seeds,
        "split_scope": args.split_scope,
        "n_folds": args.folds,
        "stage4_mode": stage4_mode,
        "stage4_audit": stage4_audit,
        "stage4_model_label": stage4_label,
        "ar1_win_rate_vs_gaussian": (wins_vs_gauss / tot_vs_gauss) if tot_vs_gauss else np.nan,
        "ar1_win_rate_vs_stage4": (wins_vs_stage4 / tot_vs_stage4) if tot_vs_stage4 else np.nan,
        "n_comparisons_vs_gaussian": tot_vs_gauss,
        "n_comparisons_vs_stage4": tot_vs_stage4,
    }

    fold_csv = os.path.join(args.output_dir, "grouped_cv_fold_metrics.csv")
    model_csv = os.path.join(args.output_dir, "grouped_cv_model_summary.csv")
    seed_csv = os.path.join(args.output_dir, "grouped_cv_seed_summary.csv")
    pkl_path = os.path.join(args.output_dir, "grouped_cv_stability.pkl")

    _write_csv(fold_csv, fold_rows, fieldnames=list(fold_rows[0].keys()) if fold_rows else [])
    _write_csv(model_csv, model_summary, fieldnames=list(model_summary[0].keys()) if model_summary else [])
    _write_csv(seed_csv, seed_summary, fieldnames=list(seed_summary[0].keys()) if seed_summary else [])

    with open(pkl_path, "wb") as f:
        pickle.dump(stability, f)

    print("Saved:")
    print(f"  {fold_csv}")
    print(f"  {model_csv}")
    print(f"  {seed_csv}")
    print(f"  {pkl_path}")
    print("AR1 win rates:")
    print(f"  vs gaussian: {stability['ar1_win_rate_vs_gaussian']}")
    if stage4_label is not None:
        print(f"  vs {stage4_label}:   {stability['ar1_win_rate_vs_stage4']}")
    else:
        print("  vs stage4:   nan")


if __name__ == "__main__":
    main()
