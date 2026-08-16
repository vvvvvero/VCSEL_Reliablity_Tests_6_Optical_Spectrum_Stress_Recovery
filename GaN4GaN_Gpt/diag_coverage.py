#!/usr/bin/env python
"""
Phase 1 – metric validation diagnostic for Stage4B Cov90.

Checks:
  1. sample-axis orientation and shape
  2. prefix vs future split (prefix has zero-width PI → never covered)
  3. NaN distribution per feature
  4. manual spot-check of quantile arithmetic
  5. per-feature pointwise vs original coverage_90_overall alignment
  6. IDLeak / IGLeak separate pointwise coverage
  7. consistent `covered` array derivation for pointwise + device-average
  8. residual = samples − x_mean diagnostic (σ ratio)
"""
import os
import sys
import pickle
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────
PKL   = r"D:\2026\article\GaN4GaN\output\pi_timegan_groupedcv_cfgC_stage4b_cov90variants\grouped_cv_stability.pkl"
DATA  = r"D:\2026\article\GaN4GaN\output\pi_timegan\processed_data.pkl"

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

PREFIX_LEN = cfg.STAGE3_PREFIX_LEN      # 4
FEATURES   = cfg.FEATURES               # 6 names

# ─── Load per-fold samples saved by diagnostic_state (if present) ────────
# We regenerate samples on-the-fly from the first fold's data stored in PKL.
with open(PKL, "rb") as f:
    stability = pickle.load(f)

fold_rows = stability.get("fold_rows", [])
stage4b_rows = [r for r in fold_rows if r["model"] == "stage4b"]

print("=" * 90)
print("PHASE 1 — METRIC VALIDATION DIAGNOSTIC")
print("=" * 90)

# ─── Check 1: shapes ────────────────────────────────────────────────────
print("\n[1] fold_rows['stage4b'] count :", len(stage4b_rows))
print("    First row keys:", list(stage4b_rows[0].keys()) if stage4b_rows else "empty")

# ─── Load state snapshot from diag_state.py if available ────────────────
DIAG_STATE = os.path.join(os.path.dirname(__file__), "diag_state.pkl")
if not os.path.exists(DIAG_STATE):
    print("\n[!] diag_state.pkl not found — run with --save-diag flag to capture tensors.")
    print("    Proceeding with fold_rows data only.")
    state = None
else:
    with open(DIAG_STATE, "rb") as f:
        state = pickle.load(f)
    print(f"\n[2] diag_state loaded — keys: {list(state.keys())}")

    x_true_norm = state["x_true_norm"]    # (N, T, F)
    x_pred_norm = state["x_pred_norm"]    # (N, T, F)
    samples     = state["samples"]        # (S, N, T, F)
    mask        = state["mask"]           # (N, T)
    T_K         = state["T_K"]            # (N,)

    S, N, T, F = samples.shape
    print(f"\n[2a] Tensor shapes: x_true={x_true_norm.shape}, x_pred={x_pred_norm.shape}, "
          f"samples={samples.shape}, mask={mask.shape}")

    # Future mask (same as compute_metrics)
    future_mask = mask.copy().astype(bool)
    future_mask[:, :PREFIX_LEN] = False

    # ── Check 2: prefix PI width ─────────────────────────────────────────
    print("\n[3] PI width at PREFIX steps (should be ~0 because samples=x_mean):")
    p5_full  = np.percentile(samples,  5, axis=0)
    p95_full = np.percentile(samples, 95, axis=0)
    width_prefix = (p95_full - p5_full)[:, :PREFIX_LEN, :]   # (N, P, F)
    print(f"    max width at prefix: {np.max(np.abs(width_prefix)):.6f}")
    print(f"    mean abs width at prefix: {np.mean(np.abs(width_prefix)):.6f}")

    width_future = (p95_full - p5_full)[:, PREFIX_LEN:, :]
    print(f"    mean abs width at FUTURE: {np.mean(np.abs(width_future)):.6f}")

    # ── Check 3: NaN distribution per feature ───────────────────────────
    print("\n[4] NaN count in x_true_norm per feature (future steps only):")
    for fi, fname in enumerate(FEATURES):
        fut_vals = x_true_norm[:, PREFIX_LEN:, fi]   # future only
        n_nan  = np.sum(np.isnan(fut_vals))
        n_tot  = np.sum(future_mask)
        print(f"    {fname:10s}: NaN={n_nan:6d} / {n_tot:6d} = {n_nan/max(n_tot,1)*100:.1f}%")

    # ── Check 4: manual spot-check for 5 random (n, t, f) triples ───────
    print("\n[5] Manual spot-check: 5 random future valid points")
    rng = np.random.default_rng(42)
    valid_idxs = np.argwhere(future_mask)
    chosen = valid_idxs[rng.choice(len(valid_idxs), size=min(5, len(valid_idxs)), replace=False)]
    for n, t in chosen:
        for fi in [0, 4, 5]:    # Vth, IDLeak, IGLeak
            fname = FEATURES[fi]
            samps_ntf = samples[:, n, t, fi]
            q05 = np.percentile(samps_ntf, 5)
            q50 = np.percentile(samps_ntf, 50)
            q95 = np.percentile(samps_ntf, 95)
            y   = x_true_norm[n, t, fi]
            covered = q05 <= y <= q95 if not np.isnan(y) else None
            print(f"    n={n:3d} t={t:3d} {fname:8s}: q05={q05:.4f} q50={q50:.4f} q95={q95:.4f} "
                  f"true={y:.4f}  covered={covered}")

    # ── Check 5: Per-feature coverage — both methods ─────────────────────
    print("\n[6] Per-feature Cov90 comparison")
    print(f"    {'Feature':10s}  {'Original':>10s}  {'Ptwise(all T)':>14s}  {'Ptwise(future)':>14s}")
    for fi, fname in enumerate(FEATURES):
        # Original method: future only, NaN excluded
        valid_f = future_mask & ~np.isnan(x_true_norm[:, :, fi])
        y_f     = x_true_norm[:, :, fi][valid_f]
        lo_f    = p5_full[:, :, fi][valid_f]
        hi_f    = p95_full[:, :, fi][valid_f]
        cov_orig = float(np.mean((y_f >= lo_f) & (y_f <= hi_f))) if len(y_f) > 0 else np.nan

        # Pointwise with full mask (bug – includes prefix, includes NaN)
        valid_allT = mask.astype(bool)
        in_f_allT  = ((x_true_norm[:, :, fi] >= p5_full[:, :, fi]) &
                      (x_true_norm[:, :, fi] <= p95_full[:, :, fi]) &
                      valid_allT)
        n_trip_allT = int(np.sum(valid_allT))
        cov_allT    = float(np.sum(in_f_allT)) / max(n_trip_allT, 1)

        # Pointwise with future mask + NaN excluded (fixed)
        valid_fut_nonan = valid_f
        in_f_fut  = ((x_true_norm[:, :, fi] >= p5_full[:, :, fi]) &
                     (x_true_norm[:, :, fi] <= p95_full[:, :, fi]) &
                     valid_fut_nonan)
        cov_fut   = float(np.sum(in_f_fut)) / max(int(np.sum(valid_fut_nonan)), 1)

        print(f"    {fname:10s}  {cov_orig:10.4f}  {cov_allT:14.4f}  {cov_fut:14.4f}")

    # ── Check 6: IDLeak + IGLeak deep dive ───────────────────────────────
    print("\n[7] IDLeak / IGLeak separate pointwise coverage")
    for fi in [4, 5]:
        fname   = FEATURES[fi]
        valid_f = future_mask & ~np.isnan(x_true_norm[:, :, fi])
        y_f     = x_true_norm[:, :, fi][valid_f]
        lo_f    = p5_full[:, :, fi][valid_f]
        hi_f    = p95_full[:, :, fi][valid_f]
        if len(y_f) == 0:
            print(f"    {fname}: no valid future points")
            continue
        cov     = float(np.mean((y_f >= lo_f) & (y_f <= hi_f)))
        width   = float(np.mean(hi_f - lo_f))
        # Median of samples at valid future positions
        med_full = np.percentile(samples, 50, axis=0)  # (N, T, F)
        med_f    = med_full[:, :, fi][valid_f]
        bias    = float(np.mean(med_f - y_f))
        below   = float(np.mean(y_f < lo_f))
        above   = float(np.mean(y_f > hi_f))
        print(f"    {fname}: Cov90={cov:.4f}, width90={width:.4f}, median_bias={bias:.4f}, "
              f"frac_below_PI={below:.3f}, frac_above_PI={above:.3f}")

    # ── Check 7: Unified covered array for pointwise + device-avg ────────
    print("\n[8] Unified covered array (future + NaN-excluded)")
    # Build (N, T, F) valid mask
    valid_3d = (future_mask[:, :, None] &
                ~np.isnan(x_true_norm))   # (N, T, F)
    # (N, T, F) in-PI array
    in_pi = ((x_true_norm >= p5_full) &
             (x_true_norm <= p95_full) &
             valid_3d)                     # (N, T, F)

    # Pointwise
    n_valid  = int(np.sum(valid_3d))
    ptwise   = float(np.sum(in_pi)) / max(n_valid, 1)

    # Device-avg
    dev_covs = []
    for n in range(N):
        n_v = int(np.sum(valid_3d[n]))
        if n_v > 0:
            dev_covs.append(float(np.sum(in_pi[n])) / n_v)
    dev_avg = float(np.nanmean(dev_covs))

    # Simultaneous (all future valid points of device in PI)
    dev_simul = []
    for n in range(N):
        if np.any(valid_3d[n]):
            dev_simul.append(float(np.all(in_pi[n][valid_3d[n]])))
    simul = float(np.nanmean(dev_simul))

    print(f"    Pointwise   (future, no-NaN): {ptwise:.4f}")
    print(f"    Device-avg  (future, no-NaN): {dev_avg:.4f}")
    print(f"    Simultaneous(future, no-NaN): {simul:.4f}")
    print(f"    Expected coverage_90_overall: ~{np.mean([np.mean((x_true_norm[:,:,fi][future_mask & ~np.isnan(x_true_norm[:,:,fi])] >= p5_full[:,:,fi][future_mask & ~np.isnan(x_true_norm[:,:,fi])]) & (x_true_norm[:,:,fi][future_mask & ~np.isnan(x_true_norm[:,:,fi])] <= p95_full[:,:,fi][future_mask & ~np.isnan(x_true_norm[:,:,fi])])) for fi in range(F)]):.4f}")

    # ── Check 9: residual sigma ratio (global blend) ──────────────────────
    print("\n[9a] Residual sigma (global blend: all devices/times/temps mixed)")
    print(f"    {'Feature':10s}  {'sg_gen':>8s}  {'sg_real':>8s}  {'ratio':>8s}  {'median_bias':>12s}")
    for fi, fname in enumerate(FEATURES):
        valid_f = future_mask & ~np.isnan(x_true_norm[:, :, fi])
        if valid_f.sum() == 0:
            print(f"    {fname}: no data")
            continue
        # generated residual = samples - x_pred (=x_mean at future)
        gen_residuals = (samples[:, :, :, fi] - x_pred_norm[np.newaxis, :, :, fi])  # (S, N, T)
        gen_res_future = gen_residuals[:, valid_f]         # (S, M)
        sigma_gen  = float(np.std(gen_res_future))

        # real residual = x_true - x_pred
        real_res_future = x_true_norm[:, :, fi][valid_f] - x_pred_norm[:, :, fi][valid_f]
        sigma_real = float(np.std(real_res_future))

        # median bias of samples vs true
        median_pred = np.percentile(samples[:, :, :, fi], 50, axis=0)[valid_f]
        median_bias = float(np.mean(median_pred - x_true_norm[:, :, fi][valid_f]))

        ratio = sigma_gen / max(sigma_real, 1e-12)
        print(f"    {fname:10s}  {sigma_gen:8.4f}  {sigma_real:8.4f}  {ratio:8.3f}  {median_bias:12.4f}")

    # ── Check 10: per-device sigma ratio (conditional) ─────────────────────
    print("\n[9b] Residual sigma per-device (conditional: device-specific sigma)")
    print(f"    {'Feature':10s}  {'sg_gen|d':>11s}  {'sg_real|d':>11s}  {'ratio|d':>10s}  {'n_devices':>10s}")
    for fi, fname in enumerate(FEATURES):
        gen_per_dev = []
        real_per_dev = []
        n_devs_ok = 0
        for n in range(N):
            valid_n = future_mask[n] & ~np.isnan(x_true_norm[n, :, fi])
            if np.sum(valid_n) < 2:  # need at least 2 points
                continue
            n_devs_ok += 1
            # sigma within this device
            gen_res_n = samples[:, n, valid_n, fi] - x_pred_norm[n, valid_n, fi][np.newaxis, :]  # (S, M_n)
            real_res_n = x_true_norm[n, valid_n, fi] - x_pred_norm[n, valid_n, fi]
            gen_per_dev.append(float(np.std(gen_res_n)))
            real_per_dev.append(float(np.std(real_res_n)))
        
        if gen_per_dev:
            sigma_gen_cond  = float(np.mean(gen_per_dev))
            sigma_real_cond = float(np.mean(real_per_dev))
            ratio_cond      = sigma_gen_cond / max(sigma_real_cond, 1e-12)
            print(f"    {fname:10s}  {sigma_gen_cond:11.4f}  {sigma_real_cond:11.4f}  {ratio_cond:10.3f}  {n_devs_ok:10d}")
        else:
            print(f"    {fname:10s}  no data")

    # ── Check 11: per-temperature sigma ratio (conditional) ────────────────
    print("\n[9c] Residual sigma per-temperature (conditional)")
    print(f"    {'Feature':10s}  {'Temp(C)':>8s}  {'sg_gen|T':>9s}  {'sg_real|T':>9s}  {'ratio|T':>9s}  {'n_points':>9s}")
    temps_c = np.round(T_K - 273.15).astype(int)
    for fi, fname in enumerate(FEATURES):
        for temp_bucket in sorted(set(temps_c)):
            # collect residuals at this temperature
            gen_list = []
            real_list = []
            for n in range(N):
                if temps_c[n] != temp_bucket:
                    continue
                valid_n = future_mask[n] & ~np.isnan(x_true_norm[n, :, fi])
                if np.sum(valid_n) < 1:
                    continue
                # residuals for this device at this temp
                gen_res_n = samples[:, n, valid_n, fi] - x_pred_norm[n, valid_n, fi][np.newaxis, :]  # (S, M_n)
                real_res_n = x_true_norm[n, valid_n, fi] - x_pred_norm[n, valid_n, fi]  # (M_n,)
                gen_list.append(gen_res_n)
                real_list.append(real_res_n)
            
            if len(gen_list) == 0:
                continue
            gen_all = np.concatenate(gen_list, axis=1)  # (S, total_points_at_T)
            real_all = np.concatenate(real_list, axis=0)  # (total_points_at_T,)
            
            sigma_gen_t = float(np.std(gen_all))
            sigma_real_t = float(np.std(real_all))
            ratio_t = sigma_gen_t / max(sigma_real_t, 1e-12)
            n_pts = len(real_all)
            print(f"    {fname:10s}  {temp_bucket:8d}  {sigma_gen_t:9.4f}  {sigma_real_t:9.4f}  {ratio_t:9.3f}  {n_pts:9d}")

print("\n" + "=" * 90)
print("NOTE: Run 11_grouped_cv_stability.py with --save-diag to populate diag_state.pkl")
print("=" * 90)
