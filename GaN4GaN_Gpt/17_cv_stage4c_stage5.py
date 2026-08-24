"""
17_cv_stage4c_stage5.py
=======================
Multi-seed x grouped-K-fold statistical validation of Stage 4C vs Stage 5.

WHY THIS EXISTS
---------------
All Stage 4C / Stage 5 numbers reported during development came from a
single fixed train/val/test split with n_test = 27 devices, single seed.
A paired bootstrap on that single split put the Stage 5 CRPS improvement at
mean +0.00049 with a 95% CI of [-0.00002, +0.00097] (Wilcoxon p = 0.052) —
i.e. the headline "CRPSS 0.171 -> 0.186" is NOT established at that sample
size, and neither are the differences between the various Stage 4C
architecture variants tried during development (they differ by ~0.01, well
inside this noise band).

11_grouped_cv_stability.py already provides grouped (device_type x
temperature stratified) folds, but it loads ONE pre-trained checkpoint and
only re-splits the EVALUATION set. That measures evaluation variance while
holding training fixed, so it cannot answer "does Stage 5 beat Stage 4C
robustly across retrainings?".

This script instead RETRAINS per fold:
  for each seed:
    for each grouped fold:
      - hold out that fold's devices as the test set
      - train Stage 4C from the frozen Stage 3 backbone on the rest
      - train Stage 5 from that Stage 4C checkpoint
      - evaluate BOTH on the held-out fold, per-device
  -> paired per-device CRPS/coverage across all (seed, fold) combinations
  -> paired bootstrap CI + Wilcoxon on the pooled paired differences

The Stage 3 physics backbone is NOT retrained per fold (it is frozen in
Stage 4C/5 by design, and retraining it per fold would cost ~7h/fold). This
is a deliberate, documented limitation: conclusions are about the
Stage 4C/Stage 5 *generator* stages conditional on a fixed physics backbone,
which is exactly the claim the paper makes about those stages.

Usage
-----
    python 17_cv_stage4c_stage5.py --seeds 0,1,2 --folds 5 \
        --stage3-ckpt <path> --output-dir <dir>
"""

import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


def _load(alias: str, fname: str):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _grouped_folds(indices: np.ndarray, device_types, t_k: np.ndarray,
                    n_folds: int, seed: int) -> List[np.ndarray]:
    """Stratified grouped folds by (device_type, temperature).

    Same strategy as 11_grouped_cv_stability.py::_grouped_folds — every fold
    gets a proportional share of each (type, temperature) cell, so no fold
    ends up missing a temperature entirely (which would make the
    Arrhenius-trend loss and the per-temperature metrics degenerate).
    """
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
    return [np.asarray(sorted(f), dtype=int) for f in folds]


def _per_device_metrics(s4b_mod, generator, model, dataset, indices, train_mod,
                         device, n_samples: int):
    """Evaluate one generator on a set of device indices, returning per-device
    CRPS and per-device Cov90 (both over the 4 stable features only, which is
    what Stage 4C actually generates)."""
    from torch.utils.data import DataLoader
    stage4a = s4b_mod.stage4a_mod
    ds = train_mod.DeviceDegradationDataset(dataset, list(indices))
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=train_mod.collate_fn)
    cache = stage4a._cache_trajectories(model, dl, device, train_mod._forward,
                                        cfg.STAGE3_PREFIX_LEN)
    SI = s4b_mod.STABLE_FEAT_INDICES
    sfx = torch.tensor(SI, device=device)

    crps_out, cov_hits, cov_tot = [], [], []
    generator.eval()
    with torch.no_grad():
        for rec in cache:
            plen = rec["plen"]
            T_future = rec["T_len"] - plen
            if T_future <= 0:
                continue
            tfut = rec["times"][:, plen:].to(device) if "times" in rec else None
            x_hat_f = rec["x_hat"][:, plen:, :][:, :, sfx].to(device)
            x_true_f = rec["x_true"][:, plen:, :][:, :, sfx].to(device)
            fmask = rec["mask"][:, plen:].bool().to(device)
            deltas = generator.sample_n(
                rec["z_pfx"].to(device), rec["T_K"].to(device),
                rec["x0"].to(device), rec["log_t"].to(device),
                n_samples, T_future=T_future, times_future=tfut,
            )
            x_pred = x_hat_f.unsqueeze(0) + deltas
            crps_out.append(s4b_mod._per_device_crps(x_pred, x_true_f, fmask).cpu().numpy())

            # per-device Cov90 over the same (future timestep, feature) cells
            lo = torch.quantile(x_pred, 0.05, dim=0)
            hi = torch.quantile(x_pred, 0.95, dim=0)
            valid = fmask.unsqueeze(-1) & ~torch.isnan(x_true_f)
            xt = torch.nan_to_num(x_true_f, nan=0.0)
            inside = (xt >= lo) & (xt <= hi) & valid
            cov_hits.append(inside.sum(dim=(1, 2)).cpu().numpy())
            cov_tot.append(valid.sum(dim=(1, 2)).cpu().numpy())

    if not crps_out:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(crps_out),
            np.concatenate(cov_hits).astype(float),
            np.concatenate(cov_tot).astype(float))


def _paired_stats(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> Dict:
    """Paired bootstrap CI + Wilcoxon on a vector of per-device differences."""
    out = {"n": int(len(diff)), "mean": float(np.mean(diff)), "std": float(np.std(diff)),
           "median": float(np.median(diff)),
           "frac_positive": float(np.mean(diff > 0))}
    if len(diff) < 2:
        out.update({"ci_low": float("nan"), "ci_high": float("nan"),
                    "wilcoxon_p": float("nan"), "significant": False})
        return out
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(diff, len(diff), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["ci_low"], out["ci_high"] = float(lo), float(hi)
    try:
        from scipy.stats import wilcoxon
        out["wilcoxon_p"] = float(wilcoxon(diff).pvalue)
    except Exception:
        out["wilcoxon_p"] = float("nan")
    # "significant improvement" = CI entirely above 0 (diff defined as S4C - S5,
    # so positive means Stage 5 reduced CRPS)
    out["significant"] = bool(lo > 0)
    return out


def main():
    ap = argparse.ArgumentParser(description="Multi-seed grouped-CV validation of Stage 4C vs Stage 5")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stage3-ckpt", type=str,
                    default=os.path.join(cfg.CHECKPOINT_DIR, "stage3_best.pt"))
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "cv_stage4c_stage5"))
    ap.add_argument("--epochs-4c", type=int, default=20)
    ap.add_argument("--epochs-5", type=int, default=15)
    ap.add_argument("--n-eval-samples", type=int, default=100)
    ap.add_argument("--lambda-calib", type=float, default=2.0)
    ap.add_argument("--lambda-pinball", type=float, default=0.20)
    ap.add_argument("--lambda-arrhenius-trend", type=float, default=1.0)
    ap.add_argument("--sigma-min", type=float, default=0.02)
    ap.add_argument("--skip-stage5", action="store_true",
                    help="Only evaluate Stage 4C (faster; no adversarial stage).")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    log.info("Loading pipeline modules...")
    s4b = _load("_cv_s4b", "14_stage4b_ar1_guided_generator.py")
    s5  = _load("_cv_s5", "15_stage5_adversarial_finetune.py")
    # 15_stage5's train_stage5() resolves its Stage-4B helpers via
    # sys.modules.get("_s5_stage4b"), an alias normally set up only by that
    # file's own main(). Register it so train_stage5 is callable as a library
    # function from here.
    sys.modules["_s5_stage4b"] = s4b
    stage4a = s4b.stage4a_mod
    mods = stage4a._load_all()
    train_mod = mods["train"]

    with open(cfg.PROCESSED_DATA_PATH, "rb") as f:
        dataset = pickle.load(f)
    device_types = dataset.get("device_types")
    if device_types is None:
        device_types = [d.split("_")[0] for d in dataset["device_ids"]]
    t_k = np.asarray(dataset["T_K"])
    all_idx = np.arange(len(dataset["device_ids"]))

    log.info("Loading frozen Stage 3 backbone: %s", args.stage3_ckpt)
    model = stage4a._build_model(mods).to(device)
    ck = torch.load(args.stage3_ckpt, map_location=device)
    ms = ck.get("model_state", ck.get("model_state_dict"))
    ms = {k: v for k, v in ms.items() if k != "decoder.mask"}
    model.load_state_dict(ms, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    from torch.utils.data import DataLoader
    rows: List[Dict] = []
    paired_crps: List[np.ndarray] = []
    s4c_cov_hits, s4c_cov_tot = [], []
    s5_cov_hits, s5_cov_tot = [], []

    tmp_dir = os.path.join(args.output_dir, "_ckpt_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    t_start = time.time()
    for seed in seeds:
        folds = _grouped_folds(all_idx, device_types, t_k, args.folds, seed)
        for fi, test_idx in enumerate(folds):
            _set_seed(seed * 1000 + fi)
            train_idx = np.setdiff1d(all_idx, test_idx)
            # inner val split for early stopping, stratified the same way
            inner = _grouped_folds(train_idx, device_types, t_k, 5, seed + 77)
            val_idx = inner[0]
            fit_idx = np.concatenate([inner[i] for i in range(1, len(inner))])

            log.info("=" * 70)
            log.info("seed=%d fold=%d/%d | fit=%d val=%d test=%d",
                     seed, fi + 1, args.folds, len(fit_idx), len(val_idx), len(test_idx))

            fit_dl = DataLoader(train_mod.DeviceDegradationDataset(dataset, list(fit_idx)),
                                batch_size=cfg.BATCH_SIZE, shuffle=True,
                                collate_fn=train_mod.collate_fn)
            val_dl = DataLoader(train_mod.DeviceDegradationDataset(dataset, list(val_idx)),
                                batch_size=cfg.BATCH_SIZE, shuffle=False,
                                collate_fn=train_mod.collate_fn)

            # ── Stage 4C ────────────────────────────────────────────────────
            gen4c = s4b.AR1GuidedResidualGeneratorStable().to(device)
            fold_ckpt_dir = os.path.join(tmp_dir, f"s{seed}_f{fi}")
            os.makedirs(fold_ckpt_dir, exist_ok=True)
            s4b.train_stage4b(
                model, gen4c, fit_dl, val_dl, device, mods=mods,
                epochs=args.epochs_4c,
                stable_feat_indices=s4b.STABLE_FEAT_INDICES,
                lambda_calib=args.lambda_calib,
                lambda_pinball=args.lambda_pinball,
                lambda_arrhenius_trend=args.lambda_arrhenius_trend,
                sigma_min=args.sigma_min,
                output_dir=fold_ckpt_dir,
            )
            s4c_path = os.path.join(fold_ckpt_dir, "stage4b_best.pt")
            gen4c.load_state_dict(torch.load(s4c_path, map_location=device)["state_dict"])

            c4, h4, t4 = _per_device_metrics(s4b, gen4c, model, dataset, test_idx,
                                              train_mod, device, args.n_eval_samples)

            row = {"seed": seed, "fold": fi, "n_test": int(len(c4)),
                   "s4c_crps": float(np.mean(c4)) if len(c4) else float("nan"),
                   "s4c_cov90": float(h4.sum() / max(t4.sum(), 1)) if len(c4) else float("nan")}

            # ── Stage 5 ─────────────────────────────────────────────────────
            if not args.skip_stage5:
                fit_cache = stage4a._cache_trajectories(model, fit_dl, device,
                                                        train_mod._forward, cfg.STAGE3_PREFIX_LEN)
                val_cache = stage4a._cache_trajectories(model, val_dl, device,
                                                        train_mod._forward, cfg.STAGE3_PREFIX_LEN)
                # Stage 4C baselines for the collapse guard, computed the same
                # way 15_stage5's own main() does.
                b_crps, b_div, bh, bt = 0.0, 0.0, 0, 0
                nb = 0
                gen4c.eval()
                with torch.no_grad():
                    for rec in val_cache:
                        plen = rec["plen"]; T_future = rec["T_len"] - plen
                        if T_future <= 0:
                            continue
                        d = gen4c.sample_n(rec["z_pfx"].to(device), rec["T_K"].to(device),
                                           rec["x0"].to(device), rec["log_t"].to(device),
                                           s5.N_VAL_SAMPLES, T_future=T_future)
                        _sfx = torch.tensor(s4b.STABLE_FEAT_INDICES, device=device)
                        xf = rec["x_hat"][:, plen:, :][:, :, _sfx].to(device).unsqueeze(0) + d
                        xp = rec["x_hat"][:, :plen, :][:, :, _sfx].to(device).unsqueeze(0).expand(s5.N_VAL_SAMPLES, -1, -1, -1)
                        xpv = torch.cat([xp, xf], dim=2)
                        xt = rec["x_true"][:, :, _sfx].to(device)
                        b_crps += s4b.crps_mc_loss(xpv, xt, rec["mask"].to(device), prefix_len=plen).item()
                        hh, tt = s5._coverage90_counts(xpv, xt, rec["mask"].to(device), plen)
                        bh += hh; bt += tt
                        b_div += s5._diversity(d)
                        nb += 1
                b_crps /= max(nb, 1); b_div /= max(nb, 1)
                b_cov = bh / max(bt, 1)

                gen5 = s4b.AR1GuidedResidualGeneratorStable().to(device)
                gen5.load_state_dict(torch.load(s4c_path, map_location=device)["state_dict"])
                s5.train_stage5(
                    model3=model, generator=gen5,
                    train_cache=fit_cache, val_cache=val_cache, device=device,
                    stage4b_val_crps=b_crps, stage4b_diversity=b_div,
                    stage4b_cov90=b_cov,
                    output_dir=fold_ckpt_dir, epochs=args.epochs_5,
                )
                s5_path = os.path.join(fold_ckpt_dir, "stage5_best.pt")
                if os.path.exists(s5_path):
                    gen5.load_state_dict(torch.load(s5_path, map_location=device)["state_dict"])
                    row["s5_saved"] = True
                else:
                    # Guard refused to save (no epoch passed) — Stage 5 is then
                    # a no-op and the honest comparison is S4C vs itself.
                    row["s5_saved"] = False
                    log.warning("  seed=%d fold=%d: Stage 5 saved no checkpoint (all guards failed)", seed, fi)

                c5, h5, t5 = _per_device_metrics(s4b, gen5, model, dataset, test_idx,
                                                  train_mod, device, args.n_eval_samples)
                row["s5_crps"] = float(np.mean(c5)) if len(c5) else float("nan")
                row["s5_cov90"] = float(h5.sum() / max(t5.sum(), 1)) if len(c5) else float("nan")
                if len(c4) and len(c5) and len(c4) == len(c5):
                    paired_crps.append(c4 - c5)   # positive => Stage 5 better
                    s5_cov_hits.append(h5); s5_cov_tot.append(t5)

            s4c_cov_hits.append(h4); s4c_cov_tot.append(t4)
            rows.append(row)
            log.info("  -> S4C crps=%.5f cov90=%.4f | S5 crps=%s cov90=%s",
                     row["s4c_crps"], row["s4c_cov90"],
                     f"{row.get('s5_crps', float('nan')):.5f}", f"{row.get('s5_cov90', float('nan')):.4f}")

    # ── Aggregate ───────────────────────────────────────────────────────────
    summary: Dict = {"seeds": seeds, "folds": args.folds, "n_runs": len(rows),
                     "elapsed_s": time.time() - t_start}
    s4c_crps_all = np.array([r["s4c_crps"] for r in rows if np.isfinite(r["s4c_crps"])])
    summary["s4c_crps_mean"] = float(s4c_crps_all.mean())
    summary["s4c_crps_std"]  = float(s4c_crps_all.std())
    s4c_cov_all = np.array([r["s4c_cov90"] for r in rows if np.isfinite(r["s4c_cov90"])])
    summary["s4c_cov90_mean"] = float(s4c_cov_all.mean())
    summary["s4c_cov90_std"]  = float(s4c_cov_all.std())

    if paired_crps:
        pooled = np.concatenate(paired_crps)
        summary["paired_crps_s4c_minus_s5"] = _paired_stats(pooled)
        s5_crps_all = np.array([r["s5_crps"] for r in rows if np.isfinite(r.get("s5_crps", float("nan")))])
        s5_cov_all  = np.array([r["s5_cov90"] for r in rows if np.isfinite(r.get("s5_cov90", float("nan")))])
        summary["s5_crps_mean"] = float(s5_crps_all.mean())
        summary["s5_crps_std"]  = float(s5_crps_all.std())
        summary["s5_cov90_mean"] = float(s5_cov_all.mean())
        summary["s5_cov90_std"]  = float(s5_cov_all.std())
        summary["n_folds_stage5_saved"] = int(sum(1 for r in rows if r.get("s5_saved")))

    with open(os.path.join(args.output_dir, "cv_rows.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    with open(os.path.join(args.output_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("=" * 70)
    log.info("RESULTS over %d runs (%d seeds x %d folds)", len(rows), len(seeds), args.folds)
    log.info("  Stage 4C: CRPS = %.5f +- %.5f | Cov90 = %.4f +- %.4f",
             summary["s4c_crps_mean"], summary["s4c_crps_std"],
             summary["s4c_cov90_mean"], summary["s4c_cov90_std"])
    if "s5_crps_mean" in summary:
        log.info("  Stage 5 : CRPS = %.5f +- %.5f | Cov90 = %.4f +- %.4f  (checkpoint saved in %d/%d folds)",
                 summary["s5_crps_mean"], summary["s5_crps_std"],
                 summary["s5_cov90_mean"], summary["s5_cov90_std"],
                 summary["n_folds_stage5_saved"], len(rows))
        ps = summary["paired_crps_s4c_minus_s5"]
        log.info("  Paired per-device CRPS improvement (S4C - S5), n=%d devices:", ps["n"])
        log.info("    mean=%+.6f  median=%+.6f  frac_devices_improved=%.3f",
                 ps["mean"], ps["median"], ps["frac_positive"])
        log.info("    bootstrap 95%% CI = [%+.6f, %+.6f]  wilcoxon p=%.4f  -> %s",
                 ps["ci_low"], ps["ci_high"], ps["wilcoxon_p"],
                 "SIGNIFICANT" if ps["significant"] else "NOT SIGNIFICANT")
    log.info("Saved -> %s", args.output_dir)


if __name__ == "__main__":
    main()
