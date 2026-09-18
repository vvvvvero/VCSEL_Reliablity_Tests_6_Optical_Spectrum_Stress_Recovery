"""
19_inference_window_curve.py
============================
Inference-time observation-window curve: take ONE trained backbone and feed
it progressively longer observation windows, all starting at t=0.

    observe [0,1,2,5]          -> predict 10 h .. 2000 h
    observe [0,1,2,5,10]       -> predict 20 h .. 2000 h
    observe [0,1,2,5,10,20]    -> predict 50 h .. 2000 h
    ...

This answers a different question from 18_prefix_len_sweep.py:

  18_ asks  "does TRAINING for a given window length help?"  (retrain per
            setting, ~14 h each)
  19_ asks  "given a model I already have, does giving it MORE data at
            inference time improve its forecast?"  (no retraining, minutes)

The second question is the one that matters operationally: during an
accelerated-ageing run you keep measuring, and you want to know whether
re-forecasting with the extra points actually buys you anything, or whether
the forecast is already as good as it will get after 5 h.

t=0 is always retained: the degradation features are defined relative to the
t=0 baseline (x2 = -log(IDS/IDS_0) etc.), so dropping it would remove the
reference the features are built on.

As in 18_, every window is scored on a COMMON horizon (default t >= 50 h) so
that a longer window is not flattered by having fewer / later points left to
predict.

Usage
-----
    python 19_inference_window_curve.py --checkpoint <stage3_best.pt>
    python 19_inference_window_curve.py --windows 4,5,6,7,8 --common-start-h 100
"""

import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
from typing import Dict, List

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


def main():
    ap = argparse.ArgumentParser(description="Inference-time observation-window curve")
    ap.add_argument("--dataset", type=str, default=None,
                    help="explicit dataset path; the default followed "
                         "cfg.PROCESSED_DATA_PATH and silently paired the "
                         "6-feature file with an 11-feature checkpoint")
    ap.add_argument("--checkpoint", type=str,
                    default=os.path.join(cfg.CHECKPOINT_DIR, "stage3_best.pt"))
    ap.add_argument("--windows", type=str, default="2,3,4,5,6,7,8",
                    help="Comma-separated prefix lengths to feed at inference.")
    ap.add_argument("--common-start-h", type=float, default=None,
                    help="Score only t >= this. Default: the first predicted "
                         "point of the LONGEST window, so all windows are "
                         "compared on an identical horizon.")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "inference_window_curve"))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    grid = cfg.TIME_POINTS_H
    windows = [w for w in windows if 2 <= w < len(grid)]

    # Default common horizon = first point the LONGEST window must predict.
    common_start = args.common_start_h
    if common_start is None:
        common_start = float(grid[max(windows)])
    log.info("time grid: %s", grid)
    for w in windows:
        log.info("  window=%d -> observe to %gh, predict %s", w, grid[w - 1], grid[w:])
    log.info("common scoring horizon: t >= %g h", common_start)

    s4b = _load("_iw_s4b", "14_stage4b_ar1_guided_generator.py")
    s4a = s4b.stage4a_mod
    mods = s4a._load_all()
    train_mod = mods["train"]
    eval_mod = mods["eval"]

    model = s4a._build_model(mods)
    ck = torch.load(args.checkpoint, map_location="cpu")
    ms = ck.get("model_state", ck.get("model_state_dict"))
    ms = {k: v for k, v in ms.items() if k != "decoder.mask"}
    model.load_state_dict(ms, strict=False)
    model.eval()
    log.info("loaded backbone: %s", args.checkpoint)

    ds_path = args.dataset or cfg.PROCESSED_DATA_PATH
    log.info("dataset %s", os.path.basename(ds_path))
    with open(ds_path, "rb") as f:
        dataset = pickle.load(f)
    if args.split == "all":
        idx = list(range(len(dataset["device_ids"])))
    else:
        idx = dataset["split"][args.split]

    from torch.utils.data import DataLoader
    dl = DataLoader(train_mod.DeviceDegradationDataset(dataset, idx),
                    batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=train_mod.collate_fn)

    SI = s4b.STABLE_FEAT_INDICES
    FN = [cfg.FEATURES[i] for i in SI]
    sfx = torch.tensor(SI)

    results: List[Dict] = []
    for w in windows:
        # IMPORTANT: use predict_from_prefix (09_evaluation.py), which encodes
        # ONLY the first w steps and then ODE-extrapolates. This is a genuine
        # forecast.
        #
        # Do NOT use _cache_trajectories(..., prefix_len=w) here: that calls
        # 08_training.py::_forward, which encodes the FULL sequence and decodes
        # x_hat straight from the encoder states. Its prefix_len argument only
        # selects which latent gets labelled z_pfx; x_hat itself is a
        # reconstruction that has already seen the "future" points, so it is
        # identical for every w (verified: byte-identical RMSE across w=2..8).
        sq = {fn: [] for fn in FN}
        bias = {fn: [] for fn in FN}
        with torch.no_grad():
            for batch in dl:
                out = eval_mod.predict_from_prefix(
                    model, batch["enc_input"], batch["x"], batch["mask"],
                    batch["times_h"], batch["T_K"], batch["x0"],
                    w, torch.device("cpu"))
                x_pred = out["x_pred"].cpu().numpy()[:, :, SI]
                x_true = batch["x"].cpu().numpy()[:, :, SI]
                msk = batch["mask"].cpu().numpy().astype(bool)
                times = batch["times_h"].cpu().numpy()
                for b in range(x_true.shape[0]):
                    for t in range(w, x_true.shape[1]):   # forecast region only
                        if not msk[b, t] or times[b, t] < common_start - 1e-6:
                            continue
                        for fi, fn in enumerate(FN):
                            v = x_true[b, t, fi] - x_pred[b, t, fi]
                            if np.isfinite(v):
                                sq[fn].append(v * v)
                                bias[fn].append(v)
        allsq = [v for fn in FN for v in sq[fn]]
        r = {
            "window": w,
            "observe_to_h": grid[w - 1],
            "rmse": {fn: (float(np.sqrt(np.mean(sq[fn]))) if sq[fn] else float("nan")) for fn in FN},
            "median_bias": {fn: (float(np.median(bias[fn])) if bias[fn] else float("nan")) for fn in FN},
            "n_points": {fn: len(sq[fn]) for fn in FN},
            "rmse_overall": float(np.sqrt(np.mean(allsq))) if allsq else float("nan"),
        }
        results.append(r)
        log.info("  window=%d (obs to %gh): RMSE=%.4f  n=%d",
                 w, grid[w - 1], r["rmse_overall"], sum(r["n_points"].values()))

    with open(os.path.join(args.output_dir, "window_curve.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "split": args.split,
                   "common_start_h": common_start, "results": results}, f,
                  indent=2, default=str)

    log.info("=" * 74)
    log.info("INFERENCE-TIME WINDOW CURVE  (split=%s, scored on t >= %g h)",
             args.split, common_start)
    log.info("%-8s %-10s " % ("window", "obs_to_h")
             + " ".join(f"{fn:>9}" for fn in FN) + f" {'overall':>9} {'n':>6}")
    for r in results:
        log.info("%-8d %-10g " % (r["window"], r["observe_to_h"])
                 + " ".join(f"{r['rmse'][fn]:>9.4f}" for fn in FN)
                 + f" {r['rmse_overall']:>9.4f} {sum(r['n_points'].values()):>6}")
    log.info("")
    log.info("median bias:")
    log.info("%-8s %-10s " % ("window", "obs_to_h") + " ".join(f"{fn:>9}" for fn in FN))
    for r in results:
        log.info("%-8d %-10g " % (r["window"], r["observe_to_h"])
                 + " ".join(f"{r['median_bias'][fn]:>+9.4f}" for fn in FN))

    base = results[0]["rmse_overall"] if results else float("nan")
    best = min((r["rmse_overall"] for r in results), default=float("nan"))
    log.info("")
    log.info("shortest window RMSE = %.4f | best = %.4f | relative gain = %.1f%%",
             base, best, 100.0 * (base - best) / base if base else float("nan"))
    log.info("Saved -> %s", args.output_dir)


if __name__ == "__main__":
    main()
