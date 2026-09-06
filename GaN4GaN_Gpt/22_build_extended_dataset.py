"""
22_build_extended_dataset.py
============================
Merge the five retained curve-shape features from 21_iv_curve_features.py
into the existing processed dataset, producing an 11-feature observation
tensor.

Why a separate script instead of editing 01_data_preprocessing.py
------------------------------------------------------------------
01_ reads 142 spreadsheets and rebuilds splits, normalisation statistics and
leakage floors from scratch. Re-running it is how processed_data.pkl was
destroyed once already. This script instead LOADS the existing processed
dataset, appends new feature columns aligned to its device and time axes, and
writes a NEW file. The original is opened read-only.

Keeping the same device order, the same train/val/test split and the same
first six feature columns means the extended dataset is directly comparable
with every earlier result.

Degradation transforms
----------------------
The existing convention is x = -log(P/P0) for quantities that DECREASE with
damage (IDSS, gmmax), +log(P/P0) for those that increase (RON), a scaled
difference for Vth, and a floored log-ratio for leakage. That convention
assumes a strictly positive quantity, so it cannot be reused blindly:

  SS_lin, SS_sat   positive, increase with damage   -> +log(SS/SS0)
  gm_fwhm_sat      positive, increases              -> +log(W/W0)
  DIBL             SIGNED voltage, crosses zero     -> (D - D0) / scale
  V_gmpeak_sat     SIGNED voltage, crosses zero     -> (V - V0) / scale

Using a log ratio on the two signed features would produce NaN wherever the
baseline or the value is negative, silently dropping those devices. They are
therefore scaled differences, following the Vth precedent: divide by the
training-split standard deviation of the baseline so the column is O(1).

Missing data
------------
A device/timepoint present in the scalar dataset but absent from the IV
sweeps gets NaN in the new columns and False in feature_mask, exactly as the
existing pipeline marks missing scalar measurements. 199 of 203 devices have
a t=0 baseline; the 4 without one cannot have a relative feature computed and
are all-NaN in the new columns (their original six columns are untouched).

Usage
-----
    python 22_build_extended_dataset.py
    python 22_build_extended_dataset.py --report-only
"""

import argparse
import collections
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# New feature columns, appended after the original six. Must stay in sync
# with cfg.CURVE_FEATURES.
# "log"  -> +log(v / v0)          positive quantity that grows with damage
# "diff" -> (v - v0) / scale      signed quantity; scale from the train split
# V_knee is deliberately absent: the IDVD sweep steps VD in 0.1 V, so the knee
# takes only 83 distinct values, 22 % of devices move by exactly zero, and the
# median drift is +0.000 from 10 h to 1000 h. It is quantised by the
# measurement grid, not resolving degradation.
NEW_FEATURES = [
    ("SS_lin",       "log"),
    ("SS_sat",       "log"),
    ("gm_fwhm_sat",  "log"),
    ("DIBL",         "diff"),
    ("V_gmpeak_sat", "diff"),
]

# Guards against the heavy tails seen in the raw extraction (SS up to 7e6 on
# devices that failed outright): clip the transformed value, matching how the
# existing pipeline clips leakage log-ratios and Vth.
LOG_CLIP = 3.0      # +-3 in natural log = a factor of ~20 either way
DIFF_CLIP = 6.0     # in units of the baseline standard deviation

# A baseline this small means the extraction failed rather than the device
# genuinely having a near-zero swing/width.
MIN_POSITIVE_BASELINE = 1e-9


def normalise_id(dev_id: str) -> str:
    """Match 21_'s convention: TYPE_TEMP_CELL with an unpadded cell number."""
    import re
    m = re.match(r"^(.+)_(\d+)_([A-Za-z]+)0*(\d+)$", dev_id)
    return (f"{m.group(1)}_{m.group(2)}_{m.group(3).upper()}{int(m.group(4))}"
            if m else dev_id)


def build_lookup(records: List[Dict]) -> Dict[str, Dict[float, Dict]]:
    """{normalised_device_id: {hours: record}}."""
    out: Dict[str, Dict[float, Dict]] = collections.defaultdict(dict)
    for r in records:
        out[normalise_id(r["device_id"])][float(r["hours"])] = r
    return out


def main():
    ap = argparse.ArgumentParser(description="Build the extended-feature dataset")
    ap.add_argument("--iv-features", default=os.path.join(cfg.OUTPUT_PATH,
                                                          "iv_curve_features.pkl"))
    ap.add_argument("--source", default=cfg.PROCESSED_DATA_PATH,
                    help="Existing processed dataset (read-only)")
    ap.add_argument("--output", default=os.path.join(cfg.OUTPUT_PATH,
                                                     "processed_data_ext.pkl"))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if os.path.abspath(args.output) == os.path.abspath(args.source):
        log.error("refusing to overwrite the source dataset")
        return

    with open(args.source, "rb") as f:
        ds = pickle.load(f)
    with open(args.iv_features, "rb") as f:
        ivp = pickle.load(f)
    lookup = build_lookup(ivp["records"])

    dev_ids = list(ds["device_ids"])
    times = np.asarray(ds["times_h"])            # (N, T)
    mask = np.asarray(ds["mask"])                # (N, T)
    x_raw = np.asarray(ds["x_raw_deg"])          # (N, T, 6)
    x_norm = np.asarray(ds["x"])
    fmask = np.asarray(ds["feature_mask"])       # (N, T, 6)
    N, T, F_old = x_raw.shape
    F_new = len(NEW_FEATURES)
    log.info("source: %d devices x %d timepoints x %d features", N, T, F_old)
    log.info("adding %d features: %s", F_new, [n for n, _ in NEW_FEATURES])

    # ---- gather raw values on the (device, time) grid ---------------------
    raw = np.full((N, T, F_new), np.nan, dtype=float)
    base = np.full((N, F_new), np.nan, dtype=float)
    hit = collections.Counter()
    for i, dev in enumerate(dev_ids):
        tp = lookup.get(normalise_id(dev))
        if not tp:
            hit["device_not_in_iv"] += 1
            continue
        b = tp.get(0.0)
        for k, (name, _) in enumerate(NEW_FEATURES):
            if b is not None:
                v = b.get(name)
                if v is not None and np.isfinite(v):
                    base[i, k] = float(v)
        for j in range(T):
            if not mask[i, j]:
                continue
            rec = tp.get(float(times[i, j]))
            if rec is None:
                hit["timepoint_missing"] += 1
                continue
            hit["matched"] += 1
            for k, (name, _) in enumerate(NEW_FEATURES):
                v = rec.get(name)
                if v is not None and np.isfinite(v):
                    raw[i, j, k] = float(v)
    log.info("grid fill: %s", dict(hit))
    log.info("devices with a usable t=0 baseline per feature: %s",
             {n: int(np.isfinite(base[:, k]).sum())
              for k, (n, _) in enumerate(NEW_FEATURES)})

    # ---- transform to degradation variables -------------------------------
    train_idx = ds["split"]["train"]
    deg = np.full((N, T, F_new), np.nan, dtype=float)
    scales: Dict[str, float] = {}
    for k, (name, kind) in enumerate(NEW_FEATURES):
        b = base[:, k]
        if kind == "log":
            ok_b = np.isfinite(b) & (b > MIN_POSITIVE_BASELINE)
            with np.errstate(divide="ignore", invalid="ignore"):
                for j in range(T):
                    v = raw[:, j, k]
                    good = ok_b & np.isfinite(v) & (v > MIN_POSITIVE_BASELINE)
                    deg[good, j, k] = np.log(v[good] / b[good])
            deg[:, :, k] = np.clip(deg[:, :, k], -LOG_CLIP, LOG_CLIP)
            scales[name] = 1.0
        else:
            # Signed quantity: scale the difference by the spread of the
            # baseline over the TRAINING split only, so val/test leak nothing.
            tb = b[train_idx]
            tb = tb[np.isfinite(tb)]
            s = float(np.std(tb)) if len(tb) > 1 else 1.0
            if not np.isfinite(s) or s < 1e-6:
                s = 1.0
            scales[name] = s
            ok_b = np.isfinite(b)
            for j in range(T):
                v = raw[:, j, k]
                good = ok_b & np.isfinite(v)
                deg[good, j, k] = (v[good] - b[good]) / s
            deg[:, :, k] = np.clip(deg[:, :, k], -DIFF_CLIP, DIFF_CLIP)
    log.info("scales for signed features: %s",
             {n: round(scales[n], 5) for n, kd in NEW_FEATURES if kd == "diff"})

    # ---- normalise, using TRAIN-split statistics only ---------------------
    norm_stats = dict(ds["norm_stats"])
    deg_norm = np.full_like(deg, np.nan)
    for k, (name, _) in enumerate(NEW_FEATURES):
        tv = deg[train_idx, :, k]
        tv = tv[np.isfinite(tv)]
        if len(tv) < 10:
            lo, hi = 0.0, 1.0
            log.warning("feature %s has only %d finite training values", name, len(tv))
        else:
            lo, hi = float(np.min(tv)), float(np.max(tv))
            if hi - lo < 1e-9:
                hi = lo + 1.0
        norm_stats[name] = {"min": lo, "max": hi}
        deg_norm[:, :, k] = (deg[:, :, k] - lo) / (hi - lo)

    # ---- assemble ---------------------------------------------------------
    new_fmask = np.isfinite(deg)
    x_raw_ext = np.concatenate([x_raw, deg], axis=2)
    x_ext = np.concatenate([x_norm, np.nan_to_num(deg_norm, nan=0.0)], axis=2)
    fmask_ext = np.concatenate([fmask, new_fmask], axis=2)

    log.info("=" * 74)
    log.info("EXTENDED FEATURE COVERAGE (over observed device/timepoint cells)")
    obs = mask.sum()
    for k, (name, kind) in enumerate(NEW_FEATURES):
        c = int((new_fmask[:, :, k] & mask).sum())
        vals = deg[:, :, k][new_fmask[:, :, k]]
        log.info("  %-14s %-5s %6.1f%%  median=%+8.4f  p5=%+7.3f  p95=%+7.3f",
                 name, kind, 100.0 * c / obs, float(np.median(vals)),
                 float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))
    per_dev = (new_fmask & mask[:, :, None]).any(axis=1).all(axis=1)
    log.info("  devices with ALL new features somewhere: %d / %d",
             int(per_dev.sum()), N)

    if args.report_only:
        log.info("--report-only: not writing")
        return

    out = dict(ds)
    out["x"] = x_ext.astype(np.float32)
    out["x_raw_deg"] = x_raw_ext.astype(np.float32)
    out["feature_mask"] = fmask_ext.astype(bool)
    out["norm_stats"] = norm_stats
    out["extended_features"] = [n for n, _ in NEW_FEATURES]
    out["extended_transforms"] = {n: k for n, k in NEW_FEATURES}
    out["extended_scales"] = scales
    out["source_dataset"] = args.source
    out["note"] = ("Extended from the 6-feature dataset by appending curve-shape "
                   "features from iv_curve_features.pkl. Device order, splits "
                   "and the first six feature columns are unchanged.")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    log.info("Saved -> %s   x shape %s", args.output, out["x"].shape)
    log.info("")
    log.info("config.py still declares %d features. Set FEATURES / "
             "DECODER_SPARSITY before training on this file.", cfg.FEATURE_DIM)


if __name__ == "__main__":
    main()
