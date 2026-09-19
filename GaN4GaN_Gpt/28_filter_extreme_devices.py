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

What is removed: measurement faults, not catastrophic failures
--------------------------------------------------------------
An earlier version of this file called these real failures. Inspecting the
trajectories shows otherwise -- they carry signatures that physical
degradation cannot produce:

  P10C_325_E13   Vth   0h +0.00  1h +37.17  2h +37.17  5h +37.17  50h -37.17
                 The value SIGN-REVERSES from +37 to -37, and sits at exactly
                 37.17 for three consecutive points. Degradation does not
                 reverse, and a physical quantity does not repeat to 5 decimal
                 places across a decade of stress time.

  A8A_275_G2     IGLeak 1h -6.00  2h -6.00  10h -0.00  1000h -6.00
                 Drops to the clip bound, RECOVERS to zero, drops again.
                 Irreversible damage cannot recover.

  P10C_325_B09   IDSS   normal (0.12 -> 0.37) through 500 h, then 14.85 at
                 1000 h, with gmmax jumping to 12.86 at the SAME timepoint.
                 Multiple channels failing together in one measurement is the
                 signature of a probe or instrument fault, not of the device.

Recurring values of exactly +-3.00 and +-6.00 are the preprocessing clip
bounds (LOG_CLIP and LEAKAGE_LOG_CLIP), i.e. those readings had already left
the representable range before the model ever saw them.

Scope of this filter
--------------------
These three are removed because their error magnitude dominates every
aggregate. They are NOT the only affected devices: scoring all 203 for the
same signatures (sign reversal, clip pinning, recovery, outsized jumps) gives
a CONTINUOUS distribution -- 147 of 203 show at least one, the sorted scores
step down 30, 25, 23, 23, 22, 20 with no gap, and these three rank 10th and
below. There is no data-driven cut, so any broader exclusion would be a
threshold chosen rather than discovered.

The deliberate decision is therefore to remove only these three, by name.
Every other device is RETAINED, and its bad readings are handled at
point level instead (below).

Point-level masking
-------------------
A reading sitting exactly on a preprocessing clip bound (+-3.00 for the log
features, +-6.00 for leakage) is not a measurement: it is a censoring marker
saying the true value was outside the representable range. Training a
regression on it teaches the model a value that was never observed.

Those points are masked individually via feature_mask -- the device stays, the
rest of its trajectory stays, only the censored readings are withheld. Scope,
measured on the 200 retained devices:

  observations masked   136 of 18862   (0.72 %)
  devices affected      52 of 200
  worst-affected device P10C_275_H3 at 25 %; no device loses more than that,
                        and none loses over half

  by feature   SS_sat 63, SS_lin 49, gm_fwhm_sat 11, IDLeak 11, IGLeak 2

This is the conservative half of the trade: it removes readings that are
provably uninformative while keeping every device and every valid point. Use
--no-mask-clipped to disable it and compare.

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
# Preprocessing clip bounds, from 01_/22_. A reading equal to one of these is
# censored -- the true value left the representable range -- so it is masked
# rather than trained on.
CLIP_BOUNDS = {"IDLeak": 6.0, "IGLeak": 6.0}
DEFAULT_CLIP = 3.0
CLIP_TOL = 1e-6

EXCLUDE_DEVICES = [
    "P10C_325_E13",   # Vth sign-reverses +37.17 -> -37.17; pinned 3 points
    "P10C_325_B09",   # IDSS and gmmax both jump at the SAME 1000 h point
    "A8A_275_G2",     # IGLeak drops to the clip bound, recovers, drops again
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
    ap.add_argument("--no-mask-clipped", action="store_true",
                    help="keep readings pinned at a clip bound (default: mask them)")
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

    # ---- point-level masking of censored readings -------------------------
    # A value sitting exactly on a clip bound is a censoring marker, not a
    # measurement. Mask those points; keep the device and the rest of its
    # trajectory. Done on the FULL arrays before the device filter so the
    # reported counts match the retained set.
    FM_new = FM.copy()
    # Censoring is DIRECTIONAL and that direction is information: a reading
    # pinned at +c says the true value was at least c, not that it is unknown.
    # Dropping the point discards that. These arrays record which cells were
    # censored and on which side, so a likelihood-based treatment (see 38_)
    # can use P(Y >= c) or P(Y <= c) instead of removing the observation.
    cens_upper = np.zeros_like(FM, dtype=bool)
    cens_lower = np.zeros_like(FM, dtype=bool)
    n_masked = 0
    if not args.no_mask_clipped:
        keep_set = set(keep)
        for f, name in enumerate(cfg.FEATURES):
            bound = CLIP_BOUNDS.get(name, DEFAULT_CLIP)
            hit_hi = (np.abs(X[:, :, f] - bound) < CLIP_TOL) & FM[:, :, f] & MK
            hit_lo = (np.abs(X[:, :, f] + bound) < CLIP_TOL) & FM[:, :, f] & MK
            cens_upper[:, :, f] = hit_hi
            cens_lower[:, :, f] = hit_lo
            hit = hit_hi | hit_lo
            for i in range(X.shape[0]):
                if i not in keep_set:
                    continue
                n_masked += int(hit[i].sum())
            FM_new[:, :, f] &= ~hit
        log.info("")
        log.info("masked %d censored readings (pinned at a clip bound) across "
                 "the retained devices", n_masked)

    out = dict(ds)
    out["feature_mask"] = FM_new
    for key in ("x", "x_raw_deg", "feature_mask", "mask", "times_h", "T_K",
                "x0_static", "x0_normalized"):
        if key in out:
            out[key] = np.asarray(out[key])[keep]
    for key in ("device_ids", "device_types"):
        if key in ds:
            out[key] = [ds[key][i] for i in keep]

    # Splits store GLOBAL indices, so they must be REMAPPED, not just filtered.
    remap = {old: new for new, old in enumerate(keep)}
    out["split"] = {k: [remap[i] for i in v if i in remap]
                    for k, v in ds["split"].items()}
    out["excluded_devices"] = [ids[i] for i in drop]
    out["n_points_masked_censored"] = int(n_masked)
    out["censored_upper"] = cens_upper[keep]
    out["censored_lower"] = cens_lower[keep]
    out["censoring_bounds"] = {n: CLIP_BOUNDS.get(n, DEFAULT_CLIP)
                               for n in cfg.FEATURES}
    log.info("censored cells retained as directional flags: %d upper, %d lower",
             int(cens_upper[keep].sum()), int(cens_lower[keep].sum()))
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
    log.info("NOTE only these three DEVICES are removed. 147 of 203 show at")
    log.info("least one fault signature, on a continuous scale with no natural")
    log.info("cut, so the rest are retained and only their censored readings")
    log.info("are masked point-by-point.")


if __name__ == "__main__":
    main()
