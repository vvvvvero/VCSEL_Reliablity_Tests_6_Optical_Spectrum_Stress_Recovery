"""
28_filter_extreme_devices.py
============================
Pipeline step: drop devices whose degradation is so extreme that they dominate
every aggregate statistic, and write a filtered copy of the dataset.

Runs after 22_build_extended_dataset.py and before any Stage 1-3 training that
should not be steered by a handful of catastrophic parts.

The problem they cause
----------------------
On the 11-feature backbone the six worst training devices carry 98.2 % of the
total squared forecast error, and P10C_325_E13 alone has RMSE 11.30 against a
median device RMSE of 0.085 -- 130x the typical case.

That made train RMSE (0.434) look far worse than test (0.178), which had been
mis-read as underfitting. It is not: on MEDIAN device RMSE the splits behave
normally (train 0.085, test 0.133). RMSE squares the residual, so three
devices set the number.

It also corrupts anything built from a residual spread. SIGMA_REF_BY_FEATURE
in 14_ is initialised from per-feature residual standard deviations, and
IDSS 0.927 is inflated by these devices -- the typical IDSS residual is about
0.26. Hence IDSS over-covers at Cov90 0.9565 with an interval ~3.5x wider than
most devices need.

What is removed, and the caveat that must travel with it
--------------------------------------------------------
These are almost certainly REAL catastrophic failures, not measurement noise:

  device            T        largest |degradation|     (95th pct, all devices)
  P10C_325_E13    325 C      Vth 37.17, IDSS 20.14      Vth 1.55, IDSS 0.37
  P10C_325_B09    325 C      IDSS 14.85, gmmax 12.86    gmmax 0.47
  A8A_275_G2      275 C      IDSS 9.67, RON 7.28        RON 0.38

All sit at the highest stress conditions, which is where real failures belong.
Excluding them is a deliberate modelling choice with a cost that has to be
stated wherever the filtered model is used: **it has never seen catastrophic
failure and will under-predict that tail.** For a reliability model that is
the optimistic direction, so the filtered model must not be used to bound
worst-case failure rates without saying so.

--report-only prints the robust (MAD) spreads without writing, which is what a
keep-the-devices alternative would use for sigma_ref.

Usage
-----
    python 28_filter_extreme_devices.py --report-only
    python 28_filter_extreme_devices.py
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# Listed explicitly rather than by a threshold, so the exclusion is auditable
# and cannot silently grow as the model changes.
EXCLUDE_DEVICES = [
    "P10C_325_E13",   # Vth 37.17 vs p95 1.55 -- catastrophic
    "P10C_325_B09",   # IDSS 14.85 vs p95 0.37
    "A8A_275_G2",     # IDSS 9.67, RON 7.28 vs p95 0.38
]


def robust_spread(x):
    """MAD-based sd estimate: 1.4826 * median(|x - median(x)|)."""
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def main():
    ap = argparse.ArgumentParser(description="Filter extreme-degradation devices")
    ap.add_argument("--source", default=os.path.join(cfg.OUTPUT_PATH,
                                                     "processed_data_ext.pkl"))
    ap.add_argument("--output", default=os.path.join(cfg.OUTPUT_PATH,
                                                     "processed_data_ext_filtered.pkl"))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if os.path.abspath(args.source) == os.path.abspath(args.output):
        log.error("refusing to overwrite the source dataset")
        return

    with open(args.source, "rb") as f:
        ds = pickle.load(f)
    ids = list(ds["device_ids"])
    drop = [i for i, d in enumerate(ids) if d in set(EXCLUDE_DEVICES)]
    missing = set(EXCLUDE_DEVICES) - {ids[i] for i in drop}
    if missing:
        log.warning("not found in dataset: %s", sorted(missing))
    keep = [i for i in range(len(ids)) if i not in set(drop)]
    log.info("dropping %d of %d devices: %s", len(drop), len(ids),
             [ids[i] for i in drop])

    X = np.asarray(ds["x_raw_deg"])
    FM = np.asarray(ds["feature_mask"])
    MK = np.asarray(ds["mask"])

    log.info("")
    log.info("PER-FEATURE RESIDUAL SPREAD -- what sigma_ref is built from")
    log.info("%-14s %10s %10s %10s %10s", "feature", "sd (all)", "sd (kept)",
             "MAD (all)", "sd/MAD")
    Xi, FMi, MKi = X[keep], FM[keep], MK[keep]
    for f, name in enumerate(cfg.FEATURES):
        v_all = X[:, :, f][FM[:, :, f] & MK]
        v_all = v_all[np.isfinite(v_all)]
        v_keep = Xi[:, :, f][FMi[:, :, f] & MKi]
        v_keep = v_keep[np.isfinite(v_keep)]
        sd_all = float(np.std(v_all)) if v_all.size else float("nan")
        sd_keep = float(np.std(v_keep)) if v_keep.size else float("nan")
        mad = robust_spread(v_all)
        ratio = sd_all / mad if mad and np.isfinite(mad) and mad > 0 else float("nan")
        log.info("%-14s %10.4f %10.4f %10.4f %9.1fx", name, sd_all, sd_keep,
                 mad, ratio)
    log.info("")
    log.info("A large sd/MAD ratio means that feature's spread is set by a few")
    log.info("devices, so a sigma_ref built from sd over-widens the typical case.")

    if args.report_only:
        log.info("")
        log.info("--report-only: nothing written")
        return

    out = dict(ds)
    for key in ("x", "x_raw_deg", "feature_mask", "mask", "times_h", "T_K",
                "x0_static", "x0_normalized"):
        if key in ds:
            out[key] = np.asarray(ds[key])[keep]
    for key in ("device_ids", "device_types"):
        if key in ds:
            out[key] = [ds[key][i] for i in keep]

    # Splits store GLOBAL indices, so they must be REMAPPED, not just filtered.
    remap = {old: new for new, old in enumerate(keep)}
    out["split"] = {k: [remap[i] for i in v if i in remap]
                    for k, v in ds["split"].items()}
    out["excluded_devices"] = [ids[i] for i in drop]
    out["excluded_reason"] = ("catastrophic degradation dominating aggregate "
                              "statistics; see 28_filter_extreme_devices.py")
    out["source_dataset"] = args.source

    n_before, n_after = len(ids), len(out["device_ids"])
    log.info("")
    log.info("split sizes: %s -> %s",
             {k: len(v) for k, v in ds["split"].items()},
             {k: len(v) for k, v in out["split"].items()})
    log.info("devices %d -> %d", n_before, n_after)
    assert n_after == n_before - len(drop), (n_after, n_before, len(drop))
    assert sum(len(v) for v in out["split"].values()) == n_after, "split/device mismatch"
    assert max(max(v) for v in out["split"].values() if v) < n_after, "stale split index"

    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    log.info("Saved -> %s", args.output)
    log.info("")
    log.info("NOTE the filtered model has never seen catastrophic failure and")
    log.info("will under-predict that tail. Do not use it to bound worst-case")
    log.info("failure rates without stating this.")


if __name__ == "__main__":
    main()
