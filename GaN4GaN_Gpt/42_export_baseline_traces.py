"""
42_export_baseline_traces.py
============================
Export open-loop forecast trajectories for the physics model and the two
physics-free baselines, so a figure can compare them on the same devices.

What these traces are, and are not
----------------------------------
All three models here come from `23_baseline_vanilla.py`: same training script,
same budget, same seed, differing only in the latent dynamics. That is the
comparison the paper's headline number rests on (-8.93 % against the Neural
ODE, -23.01 % against the GRU), so a figure drawn from these traces matches the
claim it illustrates.

All three are DETERMINISTIC. They are trained on prefix-conditioned rollout
MSE and emit one trajectory, not a distribution. The Neural ODE and GRU have no
probabilistic component anywhere in this repository -- no Stage 4C residual
generator was ever trained on either backbone -- so there is no ensemble or
interval to export for them, and drawing one would invent a result.

The uncertainty band belongs to a different object: the full pipeline's physics
backbone plus its Stage 4C residual generator, exported by
`41_export_forecast_traces.py`. If a figure shows both, label them as what they
are. The honest reading of the pair is that all three models give a point
forecast and only the pipeline gives a calibrated interval -- which is a
statement about the pipeline, not about the baselines being worse at something
they were never built to do.

Device selection
----------------
The same three devices `41_` selected, read from its manifest, so the panels
line up. Those were chosen by quantile of the PIPELINE's per-device CRPS, which
makes them representative of pipeline accuracy; a baseline may rank differently
on the same device, and its own rank is reported alongside so the figure is not
read as a like-for-like ranking.

Usage
-----
    python 42_export_baseline_traces.py
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

MODES = ["physics", "node", "gru"]


def _load(alias, fn):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, os.path.join(BASE_DIR, fn))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description="Export baseline forecast trajectories")
    ap.add_argument("--dataset", default=os.path.join(
        cfg.OUTPUT_PATH, "processed_data_ext_filtered.pkl"))
    ap.add_argument("--ckpt-dir", default=os.path.join(
        cfg.RESULTS_DIR, "baseline_traces"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", default="Vth,IDSS,RON")
    ap.add_argument("--reference-manifest", default=os.path.join(
        cfg.RESULTS_DIR, "forecast_traces", "manifest.json"))
    ap.add_argument("--output-dir", default=os.path.join(
        cfg.RESULTS_DIR, "forecast_traces"))
    args = ap.parse_args()

    cfg.PROCESSED_DATA_PATH = args.dataset
    bl = _load("_bt_bl", "23_baseline_vanilla.py")
    mods = bl._load_all() if hasattr(bl, "_load_all") else None
    if mods is None:
        s4b = _load("_bt_s4b", "14_stage4b_ar1_guided_generator.py")
        mods = s4b.stage4a_mod._load_all()
    tm, em = mods["train"], mods["eval"]
    device = torch.device("cpu")

    with open(args.reference_manifest, encoding="utf-8") as f:
        ref = json.load(f)
    want_dev = {k: v["device_id"] for k, v in ref["devices"].items()}
    log.info("reference devices: %s", want_dev)

    with open(args.dataset, "rb") as f:
        ds = pickle.load(f)
    ids = list(ds["device_ids"])
    test_idx = list(ds["split"]["test"])
    feats = [f.strip() for f in args.features.split(",") if f.strip()]

    from torch.utils.data import DataLoader

    results = {}     # (mode, case) -> {feature: {time: pred}}
    rmse_rank = {}   # (mode, case) -> rank among test devices by own RMSE
    for mode in MODES:
        tag = f"{mode}_frac1.00_seed{args.seed}"
        ckp = os.path.join(args.ckpt_dir, f"{tag}.pt")
        if not os.path.exists(ckp):
            log.warning("missing checkpoint for %s -- run 23_ with --save-checkpoint", mode)
            continue
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        model = bl.build_model(mods, mode, ck.get("hidden_dim", 64)).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        log.info("%-8s test RMSE %.4f (from its own run)", mode, ck.get("test_rmse", float("nan")))

        dl = DataLoader(tm.DeviceDegradationDataset(ds, test_idx),
                        batch_size=cfg.BATCH_SIZE, shuffle=False,
                        collate_fn=tm.collate_fn)
        P = cfg.STAGE3_PREFIX_LEN
        per_dev_sq, pos = {}, 0
        store = {}
        with torch.no_grad():
            for b in dl:
                out = em.predict_from_prefix(model, b["enc_input"], b["x"], b["mask"],
                                             b["times_h"], b["T_K"], b["x0"], P, device)
                xp = out["x_pred"].cpu().numpy()
                xt = b["x"].numpy(); mk = b["mask"].numpy().astype(bool)
                fm = b["feature_mask"].numpy().astype(bool)
                th = b["times_h"].numpy()
                for i in range(xt.shape[0]):
                    gi = test_idx[pos + i]
                    did = str(ids[gi])
                    sq = []
                    for j in range(P, xt.shape[1]):
                        if not mk[i, j]:
                            continue
                        for fn in feats:
                            fi = cfg.FEATURES.index(fn)
                            if not fm[i, j, fi]:
                                continue
                            d = xt[i, j, fi] - xp[i, j, fi]
                            if np.isfinite(d):
                                sq.append(d * d)
                    per_dev_sq[did] = float(np.sqrt(np.mean(sq))) if sq else float("nan")
                    if did in want_dev.values():
                        store[did] = {"times": th[i].copy(),
                                      "pred": {fn: xp[i, :, cfg.FEATURES.index(fn)].copy()
                                               for fn in feats},
                                      "obs": {fn: np.where(
                                          mk[i] & fm[i, :, cfg.FEATURES.index(fn)],
                                          xt[i, :, cfg.FEATURES.index(fn)], np.nan)
                                          for fn in feats}}
                pos += xt.shape[0]

        srt = sorted(d for d in per_dev_sq.values() if np.isfinite(d))
        for case, did in want_dev.items():
            if did not in store:
                continue
            results[(mode, case)] = store[did]
            r = per_dev_sq.get(did, float("nan"))
            rmse_rank[(mode, case)] = {
                "rmse": r,
                "rank": int(np.searchsorted(srt, r)) + 1 if np.isfinite(r) else None,
                "n": len(srt)}

    if not results:
        log.error("no checkpoints found in %s", args.ckpt_dir)
        return

    # ---- write ------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    P = cfg.STAGE3_PREFIX_LEN
    path = os.path.join(args.output_dir, "baselines_tidy.csv")
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "device_id", "model", "feature", "time_h", "segment",
                    "series", "value"])
        for (mode, case), rec in sorted(results.items()):
            did = want_dev[case]
            for fn in feats:
                for k, t in enumerate(rec["times"]):
                    seg = "prefix" if k < P else "future"
                    o = rec["obs"][fn][k]
                    if np.isfinite(o):
                        w.writerow([case, did, "observed", fn, f"{t:.4g}", seg,
                                    "observed", f"{o:.6f}"]); n += 1
                    if k >= P:
                        w.writerow([case, did, mode, fn, f"{t:.4g}", seg,
                                    "point_forecast", f"{rec['pred'][fn][k]:.6f}"]); n += 1
    log.info("")
    log.info("wrote %s  (%d rows)", os.path.basename(path), n)

    meta = {"source": "23_baseline_vanilla.py", "seed": args.seed,
            "dataset": os.path.basename(args.dataset),
            "prefix_len": P, "deterministic": True,
            "note": ("all three models emit a single trajectory; the Neural ODE "
                     "and GRU have no probabilistic component in this repository, "
                     "so no interval exists for them"),
            "per_device_rmse": {f"{m}|{c}": v for (m, c), v in rmse_rank.items()}}
    with open(os.path.join(args.output_dir, "baselines_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    log.info("")
    log.info("PER-DEVICE RMSE on the three plotted devices")
    log.info("  %-8s %-10s %10s %10s", "model", "case", "RMSE", "rank")
    for (m, c), v in sorted(rmse_rank.items()):
        log.info("  %-8s %-10s %10.4f %7s/%d", m, c, v["rmse"],
                 v["rank"], v["n"])
    log.info("")
    log.info("The devices were picked on the PIPELINE's CRPS ranking, so a")
    log.info("baseline's own rank on the same device will differ. Both are")
    log.info("reported; do not present one model's quantile as the other's.")
    log.info("")
    log.info("Saved -> %s", args.output_dir)


if __name__ == "__main__":
    main()
