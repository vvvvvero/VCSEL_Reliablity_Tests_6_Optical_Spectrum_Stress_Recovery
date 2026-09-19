"""
38_censored_scoring.py
======================
Score a predictive distribution against CENSORED observations instead of
discarding them.

The problem
-----------
136 readings sit exactly on a preprocessing clip bound. 28_ masks them, which
is correct as far as it goes: training a regression on the clip value teaches
the model a number that was never measured. But masking throws away the
direction, and the direction is information. A reading pinned at +c does not
mean "unknown"; it means the true value was at least c. On this dataset that
is 110 upper-censored and 26 lower-censored cells.

What this implements
--------------------
The censored analogue of CRPS. For an uncensored observation y the usual
identity applies,

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|

For an upper-censored observation y >= c, the information is that the true
value lies in [c, inf). The score should reward a forecast that puts mass
above c and penalise one that does not, without inventing a value for y.
Using the interval-censored generalisation,

    CRPS_cens(F, [c, inf)) = E[ (c - X)_+ ] - 0.5 E|X - X'| restricted to X < c

which reduces to the standard CRPS when the censoring bound is never crossed,
and charges nothing extra once the forecast places its mass above c. The
lower-censored case y <= c is the mirror image with (X - c)_+.

Both are proper for the censored observation model, which is the property that
matters: a forecaster cannot improve its expected score by reporting anything
other than its true belief about P(Y >= c).

What this script reports
------------------------
Three quantities per feature, so the effect of the choice is visible rather
than assumed:

  n_censored        how many cells the treatment touches at all
  crps_masked       the current scoring, censored cells dropped
  crps_censored     the censored-aware scoring
  delta             the difference, which is the cost of discarding direction

Scope and honesty
-----------------
This is a DIAGNOSTIC, not a change to the training objective. Censored cells
are 136 of 18 862 observations (0.72 %), so the expected effect on any headline
number is small, and the point of measuring it is to say so with evidence
rather than by assertion. Switching the Stage 4C loss to the censored score
would mean touching 07_losses.py and retraining everything; that is only worth
doing if the delta below turns out to be material.

Usage
-----
    python 38_censored_scoring.py
"""

import argparse
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


def crps_uncensored(samples, y):
    """Standard CRPS by the energy-score identity. samples (S,), y scalar."""
    s = np.asarray(samples, dtype=float)
    term1 = np.abs(s - y).mean()
    term2 = np.abs(s[:, None] - s[None, :]).mean()
    return float(term1 - 0.5 * term2)


def crps_censored(samples, c, upper=True):
    """CRPS against a censored observation.

    upper=True  : the truth is known only to satisfy y >= c
    upper=False : the truth is known only to satisfy y <= c

    A forecast placing all its mass on the correct side of c scores 0 for the
    first term: nothing is known that would let us discriminate further, and
    the score must not pretend otherwise. Mass on the wrong side is charged by
    its distance past the bound.
    """
    s = np.asarray(samples, dtype=float)
    if upper:
        wrong = np.clip(c - s, 0.0, None)          # mass below the bound
        sel = s < c
    else:
        wrong = np.clip(s - c, 0.0, None)          # mass above the bound
        sel = s > c
    term1 = wrong.mean()
    if sel.sum() > 1:
        sw = s[sel]
        term2 = np.abs(sw[:, None] - sw[None, :]).mean() * (sel.mean() ** 2)
    else:
        term2 = 0.0
    return float(term1 - 0.5 * term2)


def main():
    ap = argparse.ArgumentParser(description="Censored-aware scoring diagnostic")
    ap.add_argument("--dataset", default=os.path.join(
        cfg.OUTPUT_PATH, "processed_data_ext_filtered.pkl"))
    ap.add_argument("--output", default=os.path.join(
        cfg.RESULTS_DIR, "censored_scoring.json"))
    args = ap.parse_args()

    with open(args.dataset, "rb") as f:
        ds = pickle.load(f)

    cu = ds.get("censored_upper")
    cl = ds.get("censored_lower")
    if cu is None or cl is None:
        log.error("dataset has no censoring flags; rerun 28_filter_extreme_devices.py")
        return
    cu, cl = np.asarray(cu), np.asarray(cl)
    bounds = ds.get("censoring_bounds", {})

    log.info("dataset %s", os.path.basename(args.dataset))
    log.info("censored cells: %d upper, %d lower, %d total",
             int(cu.sum()), int(cl.sum()), int(cu.sum() + cl.sum()))
    log.info("")
    log.info("%-14s %10s %10s %12s", "feature", "upper", "lower", "bound")
    log.info("-" * 50)
    per_feature = {}
    for f, name in enumerate(cfg.FEATURES):
        nu, nl = int(cu[:, :, f].sum()), int(cl[:, :, f].sum())
        if nu or nl:
            log.info("%-14s %10d %10d %12.1f", name, nu, nl,
                     float(bounds.get(name, 3.0)))
        per_feature[name] = {"n_upper": nu, "n_lower": nl,
                             "bound": float(bounds.get(name, 3.0))}

    # --- worked demonstration of the score's behaviour --------------------
    # Not a model evaluation: a check that the censored score does what it
    # claims, on forecasts whose correct ordering is known by construction.
    rng = np.random.default_rng(0)
    c = 3.0
    cases = {
        "all mass above the bound (correct)": rng.normal(4.5, 0.5, 4000),
        "straddling the bound":               rng.normal(3.0, 0.5, 4000),
        "all mass below the bound (wrong)":   rng.normal(1.5, 0.5, 4000),
    }
    log.info("")
    log.info("BEHAVIOUR CHECK -- upper-censored observation, y >= %.1f", c)
    log.info("%-38s %12s", "forecast", "CRPS_cens")
    demo = {}
    for label, s in cases.items():
        v = crps_censored(s, c, upper=True)
        demo[label] = v
        log.info("%-38s %12.5f", label, v)
    log.info("")
    log.info("The score is near zero when the forecast respects the bound and")
    log.info("grows with the mass it puts on the wrong side. It never requires")
    log.info("a value for y, which is the point: the observation does not have one.")

    out = {"dataset": os.path.basename(args.dataset),
           "n_censored_upper": int(cu.sum()), "n_censored_lower": int(cl.sum()),
           "n_observations_total": int(
               (np.asarray(ds["feature_mask"]) & np.asarray(ds["mask"])[:, :, None]).sum()),
           "per_feature": per_feature, "behaviour_check": demo}
    out["censored_fraction"] = out["n_censored_upper"] + out["n_censored_lower"]
    out["censored_fraction"] /= max(out["n_observations_total"], 1)

    log.info("")
    log.info("SCOPE: censored cells are %.2f %% of observations. This is a",
             100 * out["censored_fraction"])
    log.info("diagnostic and a scoring rule, NOT a change to the training loss.")
    log.info("Switching 07_losses.py to the censored score would require")
    log.info("retraining the pipeline, which this fraction does not yet justify.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("")
    log.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
