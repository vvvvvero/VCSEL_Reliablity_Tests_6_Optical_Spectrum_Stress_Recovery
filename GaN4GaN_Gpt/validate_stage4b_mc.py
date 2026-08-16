#!/usr/bin/env python
"""
validate_stage4b_mc.py
======================
Monte Carlo stability validation for Stage 4B.

Pipeline:
  1. Calibrate feat x temp sigma scales on the VALIDATION split
     (IDLeak / IGLeak use a wider scale grid up to 2.5).
  2. For each of N_MC_SEEDS sampling seeds, draw N_TRAJ trajectories
     on the TEST split.
  3. Report mean +- SD over seeds for:
        Pointwise Cov90, Device-avg Cov90, CRPSS, MACE
  4. Report IGLeak median bias, fraction below / above PI.
  5. Explicitly check ensemble mean of stochastic residuals per feature.
  6. Decision: if Pointwise Cov90 mean >= PASS_THRESHOLD, save baseline flag.

Usage:
  python validate_stage4b_mc.py \\
      --checkpoint-stage3 <path> \\
      --checkpoint-stage4b <path> \\
      [--n-mc-seeds 5] \\
      [--n-traj 500] \\
      [--scale-grid "1.00,...,1.50"] \\
      [--leakage-scale-grid "1.00,...,2.50"] \\
      [--pass-threshold 0.75] \\
      [--output-json <path>]
"""

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg


# ─── Module loader ───────────────────────────────────────────────────────────
def _load_module(alias, filename):
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Subset helper ───────────────────────────────────────────────────────────
def _subset(arr, global_indices, local_indices):
    """Select rows by local_indices from arr indexed at global_indices."""
    positions = np.where(np.isin(global_indices, local_indices))[0]
    return arr[positions]


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-stage3", required=True)
    ap.add_argument("--checkpoint-stage4b", required=True)
    ap.add_argument("--n-mc-seeds",  type=int, default=5)
    ap.add_argument("--n-traj",      type=int, default=500)
    ap.add_argument("--scale-grid",  default="1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.50")
    ap.add_argument("--leakage-scale-grid",
                    default="1.00,1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,2.00,2.25,2.50")
    ap.add_argument("--pass-threshold", type=float, default=0.75,
                    help="Pointwise Cov90 mean needed to declare Stage 4B validated.")
    ap.add_argument("--output-json", default="stage4b_mc_validation.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device)

    # ── Load modules ──────────────────────────────────────────────────────────
    mods = {
        "prep":   _load_module("_vmc_prep",   "01_data_preprocessing.py"),
        "ode":    _load_module("_vmc_ode",    "02_physics_latent.py"),
        "enc":    _load_module("_vmc_enc",    "03_model_encoder.py"),
        "dec":    _load_module("_vmc_dec",    "04_model_decoder.py"),
        "gen":    _load_module("_vmc_gen",    "05_model_generator.py"),
        "disc":   _load_module("_vmc_disc",   "06_model_discriminator.py"),
        "train":  _load_module("_vmc_train",  "08_training.py"),
        "eval":   _load_module("_vmc_eval",   "09_evaluation.py"),
        "stoch":  _load_module("_vmc_stoch",  "10_stochastic_residual.py"),
        "stage4b":_load_module("_vmc_s4b",    "14_stage4b_ar1_guided_generator.py"),
        "gcv":    _load_module("_vmc_gcv",    "11_grouped_cv_stability.py"),
    }
    _Stage4AResidualSampleModel = mods["gcv"]._Stage4AResidualSampleModel
    _predict_indices            = mods["gcv"]._predict_indices

    scale_grid   = [float(x.strip()) for x in args.scale_grid.split(",")   if x.strip()]
    lk_grid      = [float(x.strip()) for x in args.leakage_scale_grid.split(",") if x.strip()]

    # ── Build backbone model ──────────────────────────────────────────────────
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = mods["enc"].PhysicsEncoder()
            self.decoder   = mods["dec"].SparsePhysicsDecoder()
            self.ode       = mods["ode"].PhysicsODE()
            self.alpha_net = mods["ode"].DeviceAlphaNet()
            self.generator = mods["gen"].PITimeGANGenerator()
            self.disc      = mods["disc"].PITimeGANDiscriminator()

    model3 = mods["eval"].load_model(args.checkpoint_stage3, _Model).to(device)
    model3.eval()

    # ── Load Stage 4B generator ───────────────────────────────────────────────
    ckpt4b = torch.load(args.checkpoint_stage4b, map_location=device)
    gen4b  = mods["stage4b"].AR1GuidedResidualGenerator().to(device)
    missing, _ = gen4b.load_state_dict(ckpt4b["state_dict"], strict=False)
    if missing:
        print(f"[Stage4B load] Missing keys: {missing}")
    gen4b.eval()

    sampler = _Stage4AResidualSampleModel(
        mean_model=model3,
        generator=gen4b,
        stoch_mod=mods["stoch"],
        device=device,
        prefix_len=cfg.STAGE3_PREFIX_LEN,
    )

    # ── Load dataset, get splits ──────────────────────────────────────────────
    dataset     = mods["prep"].load_dataset()
    split       = dataset["split"]
    val_indices = np.asarray(split["val"],  dtype=int)
    test_indices= np.asarray(split["test"], dtype=int)
    all_indices = np.arange(len(dataset["device_ids"]), dtype=int)

    print(f"Dataset: {len(dataset['device_ids'])} devices | "
          f"val={len(val_indices)} | test={len(test_indices)}")

    # ── Predict on val + test ─────────────────────────────────────────────────
    ns = dataset["norm_stats"]
    all_pred = _predict_indices(model3, dataset, all_indices, mods["train"], mods["eval"], device)

    def _get_subset(key, idx_subset):
        return _subset(all_pred[key], all_indices, idx_subset)

    # ── Calibrate feat x temp on VALIDATION split ─────────────────────────────
    print("\n--- Calibration on VALIDATION set ---")
    enc_val = _get_subset("enc_input", val_indices)
    x0_val  = _get_subset("x0",        val_indices)
    m_val   = _get_subset("mask",       val_indices)
    tk_val  = _get_subset("T_K",        val_indices)
    t_val   = _get_subset("times_h",    val_indices)
    xp_val  = _get_subset("x_pred_norm",val_indices)
    xt_val  = _get_subset("x_true_norm",val_indices)

    sampler._set_encoder_outputs(
        enc_val, m_val, x0_val, tk_val, t_val,
        prefix_len=cfg.STAGE3_PREFIX_LEN,
    )
    cal_info = sampler.calibrate_sigma_scale_by_feat_temp(
        x_cal_mean=xp_val,
        x_cal_true=xt_val,
        mask_cal=m_val,
        T_K_cal=tk_val,
        times_h_cal=t_val,
        n_samples=50,
        prefix_len=cfg.STAGE3_PREFIX_LEN,
        target_coverage=0.90,
        scale_grid=scale_grid,
        leakage_scale_grid=lk_grid,
    )
    print("Calibration complete.")
    for tc, scales in sorted(sampler.sigma_scale_by_feat_temp.items()):
        print(f"  {tc}C: " + " ".join(f"{cfg.FEATURES[fi]}={scales[fi]:.2f}" for fi in range(len(cfg.FEATURES))))

    # ── Prepare test prediction data ──────────────────────────────────────────
    enc_test = _get_subset("enc_input",  test_indices)
    x0_test  = _get_subset("x0",         test_indices)
    m_test   = _get_subset("mask",        test_indices)
    tk_test  = _get_subset("T_K",         test_indices)
    t_test   = _get_subset("times_h",     test_indices)
    xp_test  = _get_subset("x_pred_norm", test_indices)
    xt_test  = _get_subset("x_true_norm", test_indices)

    future_mask_test = m_test.copy().astype(bool)
    future_mask_test[:, :cfg.STAGE3_PREFIX_LEN] = False

    # ── Set encoder outputs for test split ────────────────────────────────────
    sampler._set_encoder_outputs(
        enc_test, m_test, x0_test, tk_test, t_test,
        prefix_len=cfg.STAGE3_PREFIX_LEN,
    )

    # ── Monte Carlo evaluation (5 seeds x 500 trajectories) ──────────────────
    MC_SEEDS = list(range(9001, 9001 + args.n_mc_seeds))
    print(f"\n--- MC Evaluation: {args.n_mc_seeds} seeds x {args.n_traj} trajectories ---")

    seed_results = []
    for mc_seed in MC_SEEDS:
        rng = np.random.default_rng(mc_seed)
        smp = sampler.sample_trajectories(
            xp_test, tk_test, t_test, m_test,
            n_samples=args.n_traj,
            prefix_len=cfg.STAGE3_PREFIX_LEN,
            rng=rng,
        )  # (S, N, T, F)

        # Standard metrics via compute_metrics
        met = sampler.compute_metrics(
            xt_test, smp, m_test,
            T_K=tk_test,
            prefix_len=cfg.STAGE3_PREFIX_LEN,
            x_mean=xp_test,
            times_h=t_test,
        )

        # Corrected coverage variants (future + NaN-excluded)
        cov_vars = _Stage4AResidualSampleModel.compute_coverage_variants(
            xt_test, smp, m_test,
            percentile=90.0,
            prefix_len=cfg.STAGE3_PREFIX_LEN,
        )

        seed_results.append({
            "seed":           mc_seed,
            "crpss":          float(met.get("crpss_overall",           np.nan)),
            "mace":           float(met.get("reliability_mace",        np.nan)),
            "cov90_overall":  float(met.get("coverage_90_overall",     np.nan)),
            "cov90_pointwise":float(cov_vars["coverage_90_pointwise"]),
            "cov90_device_avg":float(cov_vars["coverage_90_device_avg"]),
        })

        print(f"  seed={mc_seed}: ptwise={cov_vars['coverage_90_pointwise']:.4f}  "
              f"dev_avg={cov_vars['coverage_90_device_avg']:.4f}  "
              f"CRPSS={met.get('crpss_overall',np.nan):.4f}  "
              f"MACE={met.get('reliability_mace',np.nan):.4f}")

    # ── Aggregate statistics ──────────────────────────────────────────────────
    def _agg(key):
        vals = [r[key] for r in seed_results if not np.isnan(r[key])]
        return float(np.mean(vals)), float(np.std(vals))

    pw_mean, pw_std       = _agg("cov90_pointwise")
    da_mean, da_std       = _agg("cov90_device_avg")
    crpss_mean, crpss_std = _agg("crpss")
    mace_mean, mace_std   = _agg("mace")

    print("\n--- MC SUMMARY ---")
    print(f"  Pointwise Cov90:    {pw_mean:.4f} +- {pw_std:.4f}")
    print(f"  Device-avg Cov90:   {da_mean:.4f} +- {da_std:.4f}")
    print(f"  CRPSS:              {crpss_mean:.4f} +- {crpss_std:.4f}")
    print(f"  MACE:               {mace_mean:.4f} +- {mace_std:.4f}")

    # ── IGLeak / IDLeak diagnostics on last MC run ────────────────────────────
    print("\n--- LEAKAGE DIAGNOSTICS (last MC seed) ---")
    S, N, T, F = smp.shape
    p50 = np.percentile(smp, 50, axis=0)  # (N, T, F)
    p05 = np.percentile(smp,  5, axis=0)
    p95 = np.percentile(smp, 95, axis=0)

    leakage_diag = {}
    for fi, fname in enumerate(cfg.FEATURES):
        if fi not in (cfg.IDLEAK_DECODER_ROW, cfg.IGLEAK_DECODER_ROW):
            continue
        valid = future_mask_test & ~np.isnan(xt_test[:, :, fi])
        if valid.sum() == 0:
            continue
        y    = xt_test[:, :, fi][valid]
        med  = p50[:, :, fi][valid]
        lo   = p05[:, :, fi][valid]
        hi   = p95[:, :, fi][valid]
        bias = float(np.mean(med - y))
        frac_below = float(np.mean(y < lo))
        frac_above = float(np.mean(y > hi))
        cov = float(np.mean((y >= lo) & (y <= hi)))
        print(f"  {fname}: bias={bias:+.4f}  below={frac_below:.3f}  "
              f"above={frac_above:.3f}  cov90={cov:.4f}")
        leakage_diag[fname] = dict(bias=bias, frac_below=frac_below,
                                   frac_above=frac_above, cov90=cov)

    # ── Ensemble mean of stochastic residuals ─────────────────────────────────
    print("\n--- ENSEMBLE MEAN OF RESIDUALS per feature (future, non-NaN) ---")
    print("    (should be ~0 for non-leakage features; equals bias_head for leakage)")
    ensemble_mean_dev = smp.mean(axis=0)  # (N, T, F)
    residual_ensemble = ensemble_mean_dev - xp_test  # = bias + stoch_mean ≈ bias

    ensemble_stats = {}
    for fi, fname in enumerate(cfg.FEATURES):
        valid = future_mask_test & ~np.isnan(xt_test[:, :, fi])
        if valid.sum() == 0:
            continue
        res_vals = residual_ensemble[:, :, fi][valid]
        mean_res = float(np.mean(res_vals))
        std_res  = float(np.std(res_vals))
        print(f"  {fname}: ensemble_mean_residual = {mean_res:+.5f}  (SD={std_res:.5f})")
        ensemble_stats[fname] = dict(mean=mean_res, std=std_res)

    # ── Decision ─────────────────────────────────────────────────────────────
    validated = (pw_mean >= args.pass_threshold and pw_std <= 0.02)
    print(f"\n--- DECISION ---")
    print(f"  Pointwise Cov90 = {pw_mean:.4f} +- {pw_std:.4f}  "
          f"(threshold >= {args.pass_threshold})")
    print(f"  Stage 4B {'VALIDATED as baseline' if validated else 'NOT YET validated'}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "stage4b_checkpoint": args.checkpoint_stage4b,
        "n_mc_seeds":  args.n_mc_seeds,
        "n_traj":      args.n_traj,
        "summary": {
            "cov90_pointwise_mean":  pw_mean,
            "cov90_pointwise_std":   pw_std,
            "cov90_device_avg_mean": da_mean,
            "cov90_device_avg_std":  da_std,
            "crpss_mean":            crpss_mean,
            "crpss_std":             crpss_std,
            "mace_mean":             mace_mean,
            "mace_std":              mace_std,
        },
        "leakage_diagnostics":      leakage_diag,
        "ensemble_residual_mean":   ensemble_stats,
        "validated":                validated,
        "seed_results":             seed_results,
        "sigma_scale_by_feat_temp": {
            tc: list(v) for tc, v in sampler.sigma_scale_by_feat_temp.items()
        },
    }

    out_path = args.output_json
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved -> {out_path}")
    return result


if __name__ == "__main__":
    main()
