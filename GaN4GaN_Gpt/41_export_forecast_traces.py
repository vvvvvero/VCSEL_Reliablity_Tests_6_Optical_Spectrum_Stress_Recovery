"""
41_export_forecast_traces.py
============================
Export the per-device trajectories behind an open-loop forecast figure:
observed prefix, true future, physics mean forecast, and the probabilistic
ensemble.

Why this script exists
----------------------
Nothing in the pipeline stores trajectories. `evaluation_results_stage4b.pkl`
holds aggregate metrics only -- coverage, width, CRPS -- and the sample paths
are computed inside the evaluation loop and discarded. A forecast figure needs
the paths themselves, so they are regenerated and written out here.

How "representative" is chosen, and why it matters
--------------------------------------------------
A figure captioned "representative" invites the reader to treat it as typical.
Choosing the device that happens to look best would make that caption false,
and it is the easiest kind of cherry-picking to do without noticing.

Devices are therefore ranked by their own CRPS and selected by QUANTILE, with
the quantile printed and stored alongside each trace:

    median    the 50th percentile device -- the honest "representative"
    good      the 10th percentile
    poor      the 90th percentile

Exporting all three lets the figure show a median case while the caption can
state, truthfully, where it sits in the distribution. If only one panel is
used, use the median one.

Selection runs on the TEST split only, so nothing shown was fitted on.

What is written
---------------
For each selected device, one CSV per feature set plus a tidy long-form file:

    time_h            the 12-point grid
    segment           "prefix" (observed, t <= prefix boundary) or "future"
    observed          measured value; NaN where masked or censored
    mean_forecast     physics backbone + Stage 4C residual mean
    p05 p25 p50 p75 p95   ensemble quantiles
    sample_00..NN     individual ensemble trajectories, for spaghetti plots

The prefix rows carry the observed series only: the model sees them and does
not forecast them, so a "forecast" column there would be a reconstruction and
should not be drawn as if it were a prediction.

Usage
-----
    python 41_export_forecast_traces.py
    python 41_export_forecast_traces.py --features Vth,IDSS,RON --n-samples 200
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

QUANTILES = {"good": 0.10, "median": 0.50, "poor": 0.90}


def _load(alias, fn):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, os.path.join(BASE_DIR, fn))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description="Export forecast trajectories for plotting")
    ap.add_argument("--dataset", default=os.path.join(
        cfg.OUTPUT_PATH, "processed_data_ext_filtered.pkl"))
    ap.add_argument("--backbone", default=os.path.join(cfg.OUTPUT_PATH, "ext11_filtered"))
    ap.add_argument("--run", default="stage4c",
                    help="Stage 4C run directory under the backbone")
    ap.add_argument("--features", default="Vth,IDSS,RON",
                    help="comma-separated; must be features the generator produces")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--n-traces", type=int, default=20,
                    help="individual sample paths written per device")
    ap.add_argument("--output-dir", default=os.path.join(cfg.RESULTS_DIR, "forecast_traces"))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg.PROCESSED_DATA_PATH = args.dataset

    s4b = _load("_ft_s4b", "14_stage4b_ar1_guided_generator.py")
    s4a = s4b.stage4a_mod
    mods = s4a._load_all()
    tm = mods["train"]
    device = torch.device("cpu")

    model = s4a._build_model(mods)
    ck = torch.load(os.path.join(args.backbone, "checkpoints", "stage3_best.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    gpath = os.path.join(args.backbone, args.run, "checkpoints", "stage4b_best.pt")
    gck = torch.load(gpath, map_location="cpu", weights_only=False)
    gen = s4b.AR1GuidedResidualGeneratorStable()
    gen.load_state_dict(gck["state_dict"], strict=False)
    gen.eval()
    log.info("backbone %s | generator %s", os.path.basename(args.backbone), args.run)

    with open(args.dataset, "rb") as f:
        ds = pickle.load(f)
    idx = (list(range(len(ds["device_ids"]))) if args.split == "all"
           else list(ds["split"][args.split]))
    log.info("split %s -> %d devices", args.split, len(idx))

    from torch.utils.data import DataLoader
    dl = DataLoader(tm.DeviceDegradationDataset(ds, idx), batch_size=cfg.BATCH_SIZE,
                    shuffle=False, collate_fn=tm.collate_fn)
    P = cfg.STAGE3_PREFIX_LEN
    cache = s4a._cache_trajectories(model, dl, device, tm._forward, P)

    SI = list(s4b.STABLE_FEAT_INDICES)
    gen_names = [cfg.FEATURES[i] for i in SI]
    want = [f.strip() for f in args.features.split(",") if f.strip()]
    missing = [f for f in want if f not in gen_names]
    if missing:
        log.error("not generated by this model: %s", missing)
        log.error("available: %s", gen_names)
        return
    sfx = torch.tensor(SI, device=device)

    # ---- one pass: per-device CRPS (for ranking) and the sample paths -----
    recs = []
    pos = 0
    with torch.no_grad():
        for rec in cache:
            plen, T_len = rec["plen"], rec["T_len"]
            T_future = T_len - plen
            if T_future <= 0:
                pos += rec["z_pfx"].shape[0]
                continue
            tf = rec["times"][:, plen:].to(device)
            x_hat_f = rec["x_hat"][:, plen:, :][:, :, sfx].to(device)
            x_true_f = rec["x_true"][:, plen:, :][:, :, sfx].to(device)
            fmask = rec["mask"][:, plen:].bool().to(device)
            deltas = gen.sample_n(rec["z_pfx"].to(device), rec["T_K"].to(device),
                                  rec["x0"].to(device), rec["log_t"].to(device),
                                  args.n_samples, T_future=T_future, times_future=tf)
            x_pred = x_hat_f.unsqueeze(0) + deltas          # (S,B,Tf,F)
            crps = s4b._per_device_crps(x_pred, x_true_f, fmask).cpu().numpy()
            B = x_pred.shape[1]
            for b in range(B):
                recs.append({
                    "device_index": int(idx[pos + b]),
                    "device_id": str(ds["device_ids"][idx[pos + b]]),
                    "T_K": float(rec["T_K"][b]),
                    "crps": float(crps[b]),
                    "plen": int(plen),
                    "times": rec["times"][b].cpu().numpy(),
                    "x_true": rec["x_true"][b][:, sfx].cpu().numpy(),
                    "mask": rec["mask"][b].cpu().numpy().astype(bool),
                    "x_pred": x_pred[:, b].cpu().numpy(),    # (S,Tf,F)
                })
            pos += B

    crps_all = np.array([r["crps"] for r in recs])
    order = np.argsort(crps_all)
    log.info("")
    log.info("per-device CRPS on the %s split: median %.5f, IQR %.5f-%.5f",
             args.split, np.median(crps_all),
             np.percentile(crps_all, 25), np.percentile(crps_all, 75))

    chosen = {}
    for label, q in QUANTILES.items():
        k = order[min(int(round(q * (len(order) - 1))), len(order) - 1)]
        chosen[label] = recs[k]
        log.info("  %-7s q=%.2f  %-16s T=%.0f C  CRPS=%.5f", label, q,
                 recs[k]["device_id"], recs[k]["T_K"] - 273.15, recs[k]["crps"])

    # ---- write ------------------------------------------------------------
    manifest = {"dataset": os.path.basename(args.dataset),
                "backbone": os.path.basename(args.backbone),
                "generator_run": args.run, "split": args.split,
                "prefix_len": P, "n_samples": args.n_samples,
                "features": want, "devices": {}}

    qs = [5, 25, 50, 75, 95]
    for label, r in chosen.items():
        plen = r["plen"]
        times = r["times"]
        manifest["devices"][label] = {
            "device_id": r["device_id"], "device_index": r["device_index"],
            "temperature_C": r["T_K"] - 273.15, "crps": r["crps"],
            "crps_quantile": QUANTILES[label],
            "crps_rank": int(np.searchsorted(np.sort(crps_all), r["crps"])) + 1,
            "n_devices_in_split": len(crps_all),
            "prefix_boundary_h": float(times[plen - 1]),
        }
        for fname in want:
            j = gen_names.index(fname)
            path = os.path.join(args.output_dir, f"{label}_{fname}.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                head = (["time_h", "segment", "observed", "mean_forecast"]
                        + [f"p{q:02d}" for q in qs]
                        + [f"sample_{i:02d}" for i in range(args.n_traces)])
                w.writerow(head)
                for t in range(len(times)):
                    obs = r["x_true"][t, j]
                    if not r["mask"][t] or not np.isfinite(obs):
                        obs = ""
                    else:
                        obs = f"{obs:.6f}"
                    if t < plen:
                        # Observed prefix: the model conditions on these points
                        # rather than forecasting them, so no forecast columns.
                        w.writerow([f"{times[t]:.4g}", "prefix", obs, ""]
                                   + [""] * (len(qs) + args.n_traces))
                    else:
                        k = t - plen
                        col = r["x_pred"][:, k, j]           # (S,)
                        qq = np.percentile(col, qs)
                        w.writerow([f"{times[t]:.4g}", "future", obs,
                                    f"{col.mean():.6f}"]
                                   + [f"{v:.6f}" for v in qq]
                                   + [f"{v:.6f}" for v in col[:args.n_traces]])
            log.info("  wrote %s", os.path.basename(path))

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    log.info("")
    log.info("Saved -> %s", args.output_dir)
    log.info("")
    log.info("For a figure captioned 'representative', use the MEDIAN device and")
    log.info("state its quantile in the caption. manifest.json carries the rank so")
    log.info("the claim can be checked. The 'good' and 'poor' traces are exported")
    log.info("so the spread can be shown honestly rather than implied.")


if __name__ == "__main__":
    main()
