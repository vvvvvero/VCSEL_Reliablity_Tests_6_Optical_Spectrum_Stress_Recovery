"""
18_prefix_len_sweep.py
======================
How long must an accelerated-ageing experiment run before the model can
extrapolate reliably to 2000 h?

Trains a full Stage 1-3 backbone at several observation-window lengths
(STAGE3_PREFIX_LEN) and compares them on a COMMON evaluation horizon.

Why the common horizon matters
------------------------------
The canonical grid is [0,1,2,5,10,20,50,100,200,500,1000,2000] h, so:
    prefix=4 -> observe to    5 h, extrapolate 8 points (10 h .. 2000 h)
    prefix=5 -> observe to   10 h, extrapolate 7 points (20 h .. 2000 h)
    prefix=6 -> observe to   20 h, extrapolate 6 points (50 h .. 2000 h)
A longer prefix therefore has BOTH more information AND an easier task
(fewer, later points to predict). Comparing overall RMSE across prefixes
would confound the two. Every configuration is scored only on the points
that ALL configurations must predict — by default 50 h .. 2000 h — so the
comparison isolates the effect of the observation window itself.

Physical motivation
-------------------
The fast trap modes (zG, zB) relax with tau ~ 5-8 h at 325 C, so the three
settings correspond to observing for roughly 1, 2 and 4 time constants:
    prefix=4 -> ~63 % of the fast transient complete
    prefix=5 -> ~86 %
    prefix=6 -> ~98 % (essentially equilibrated)
The sweep therefore tests directly whether the encoder needs to see the
fast transient settle before the ODE can be trusted to extrapolate.

Usage
-----
    python 18_prefix_len_sweep.py --prefixes 4,5,6 --output-dir <dir>
    python 18_prefix_len_sweep.py --prefixes 5,6 --skip-train   # eval only
"""

import argparse
import importlib.util
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

PY = sys.executable


def _load(alias: str, fname: str):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _train_one(prefix_len: int, ckpt_dir: str, log_path: str) -> bool:
    """Run a full Stage 1-3 training with STAGE3_PREFIX_LEN=prefix_len.

    Done in a SUBPROCESS with the override injected via sitecustomize-style
    env var handling rather than in-process, because 08_training.py reads
    cfg.STAGE3_PREFIX_LEN in several places and the modules cache config
    state; a fresh interpreter per prefix is the reliable way to keep the
    runs independent.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    driver = os.path.join(ckpt_dir, "_train_driver.py")
    # NOTE: the embedded BASE_DIR contains non-ASCII characters (the repo path
    # includes "Łukasiewicz"), so the generated file must be written AND
    # declared as UTF-8 or Python refuses to parse it.
    with open(driver, "w", encoding="utf-8") as fh:
        fh.write(
            "# -*- coding: utf-8 -*-\n"
            "import sys, os\n"
            f"sys.path.insert(0, r'{BASE_DIR}')\n"
            "import config as cfg\n"
            f"cfg.STAGE3_PREFIX_LEN = {prefix_len}\n"
            f"cfg.CHECKPOINT_DIR = r'{ckpt_dir}'\n"
            "import runpy\n"
            "sys.argv = ['main.py','--mode','train','--start-stage','1','--end-stage','3','--no-preprocess']\n"
            f"os.chdir(r'{BASE_DIR}')\n"
            "runpy.run_path(os.path.join(r'"
            + BASE_DIR
            + "', 'main.py'), run_name='__main__')\n"
        )
    with open(log_path, "w") as lf:
        rc = subprocess.call([PY, driver], stdout=lf, stderr=subprocess.STDOUT)
    return rc == 0


def _evaluate(prefix_len: int, ckpt_path: str, common_start_h: float) -> Dict:
    """Evaluate one trained backbone on the COMMON horizon only.

    Returns per-feature RMSE and per-feature median bias, restricted to
    observation times >= common_start_h so every prefix setting is scored on
    an identical set of time points.
    """
    s4b = _load(f"_sw_s4b_{prefix_len}", "14_stage4b_ar1_guided_generator.py")
    s4a = s4b.stage4a_mod
    mods = s4a._load_all()
    train_mod = mods["train"]

    model = s4a._build_model(mods)
    ck = torch.load(ckpt_path, map_location="cpu")
    ms = ck.get("model_state", ck.get("model_state_dict"))
    ms = {k: v for k, v in ms.items() if k != "decoder.mask"}
    model.load_state_dict(ms, strict=False)
    model.eval()

    with open(cfg.PROCESSED_DATA_PATH, "rb") as f:
        dataset = pickle.load(f)
    from torch.utils.data import DataLoader

    SI = s4b.STABLE_FEAT_INDICES
    FN = [cfg.FEATURES[i] for i in SI]
    sfx = torch.tensor(SI)

    out: Dict = {"prefix_len": prefix_len, "common_start_h": common_start_h,
                 "val_rollout_mse": float(ck.get("selection_value",
                                                 ck.get("future_rollout_mse", float("nan"))))}
    for split in ("train", "test"):
        dl = DataLoader(train_mod.DeviceDegradationDataset(dataset, dataset["split"][split]),
                        batch_size=cfg.BATCH_SIZE, shuffle=False,
                        collate_fn=train_mod.collate_fn)
        cache = s4a._cache_trajectories(model, dl, torch.device("cpu"),
                                        train_mod._forward, prefix_len)
        sq = {fn: [] for fn in FN}
        bias = {fn: [] for fn in FN}
        for rec in cache:
            plen = rec["plen"]
            if rec["T_len"] - plen <= 0:
                continue
            times = rec["times"][:, plen:].numpy()
            resid = (rec["x_true"][:, plen:, :][:, :, sfx]
                     - rec["x_hat"][:, plen:, :][:, :, sfx]).numpy()
            fm = rec["mask"][:, plen:].bool().numpy()
            for b in range(resid.shape[0]):
                for t in range(resid.shape[1]):
                    if not fm[b, t]:
                        continue
                    if times[b, t] < common_start_h - 1e-6:
                        continue   # outside the common horizon
                    for fi, fn in enumerate(FN):
                        v = resid[b, t, fi]
                        if np.isfinite(v):
                            sq[fn].append(v * v)
                            bias[fn].append(v)
        out[f"{split}_rmse"] = {fn: (float(np.sqrt(np.mean(sq[fn]))) if sq[fn] else float("nan"))
                                for fn in FN}
        out[f"{split}_median_bias"] = {fn: (float(np.median(bias[fn])) if bias[fn] else float("nan"))
                                       for fn in FN}
        out[f"{split}_n_points"] = {fn: len(sq[fn]) for fn in FN}
        allsq = [v for fn in FN for v in sq[fn]]
        out[f"{split}_rmse_overall"] = float(np.sqrt(np.mean(allsq))) if allsq else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description="Observation-window (prefix_len) sweep")
    ap.add_argument("--prefixes", type=str, default="4,5,6")
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "prefix_len_sweep"))
    ap.add_argument("--common-start-h", type=float, default=50.0,
                    help="Score only observation times >= this, so all prefixes "
                         "are compared on an identical horizon (default 50 h, "
                         "the first point every prefix in {4,5,6} must predict).")
    ap.add_argument("--skip-train", action="store_true",
                    help="Reuse existing per-prefix checkpoints; evaluate only.")
    args = ap.parse_args()

    prefixes = [int(p) for p in args.prefixes.split(",") if p.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    grid = cfg.TIME_POINTS_H
    log.info("time grid: %s", grid)
    for p in prefixes:
        log.info("  prefix=%d -> observe to %gh, extrapolate %s",
                 p, grid[p - 1], grid[p:])
    log.info("common evaluation horizon: t >= %g h", args.common_start_h)

    results: List[Dict] = []
    for p in prefixes:
        ck_dir = os.path.join(args.output_dir, f"prefix{p}", "checkpoints")
        ck_path = os.path.join(ck_dir, "stage3_best.pt")
        if not args.skip_train:
            log.info("=" * 70)
            log.info("TRAINING prefix_len=%d  (this takes several hours)", p)
            t0 = time.time()
            ok = _train_one(p, ck_dir, os.path.join(args.output_dir, f"train_prefix{p}.log"))
            log.info("  prefix=%d training %s in %.1f min",
                     p, "OK" if ok else "FAILED", (time.time() - t0) / 60)
            if not ok:
                log.error("  see %s", os.path.join(args.output_dir, f"train_prefix{p}.log"))
                continue
        if not os.path.exists(ck_path):
            log.warning("  no checkpoint for prefix=%d at %s, skipping", p, ck_path)
            continue
        r = _evaluate(p, ck_path, args.common_start_h)
        results.append(r)
        log.info("  prefix=%d  test RMSE(common horizon)=%.4f",
                 p, r.get("test_rmse_overall", float("nan")))

    with open(os.path.join(args.output_dir, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    if results:
        FN = list(results[0]["test_rmse"].keys())
        log.info("=" * 70)
        log.info("RESULTS on the common horizon (t >= %g h)", args.common_start_h)
        log.info("%-8s %-10s " % ("prefix", "obs_to_h") + " ".join(f"{fn:>9}" for fn in FN)
                 + f" {'overall':>9}")
        for r in results:
            p = r["prefix_len"]
            log.info("%-8d %-10g " % (p, grid[p - 1])
                     + " ".join(f"{r['test_rmse'][fn]:>9.4f}" for fn in FN)
                     + f" {r['test_rmse_overall']:>9.4f}")
        log.info("")
        log.info("median bias on the common horizon (test):")
        log.info("%-8s " % "prefix" + " ".join(f"{fn:>9}" for fn in FN))
        for r in results:
            log.info("%-8d " % r["prefix_len"]
                     + " ".join(f"{r['test_median_bias'][fn]:>+9.4f}" for fn in FN))
    log.info("Saved -> %s", args.output_dir)


if __name__ == "__main__":
    main()
