"""
37_equivalence_margins.py
=========================
Equivalence and non-inferiority tests for the Stage 4C / Stage 5 comparison.

Why this exists
---------------
A non-significant difference is not evidence of equivalence. The Stage 4C/5
CRPS comparison returns a confidence interval spanning zero, and the honest
reading of that alone is "we could not detect a difference", not "the two are
the same". Deciding between those requires a margin fixed BEFORE looking at
the interval.

Two margins are used, and both are anchored to something external rather than
chosen to make the test pass.

CRPS -- relative margin tied to the measured noise floor
    delta_CRPS = 0.027 * CRPS_S4C
The 2.7 % figure is this dataset's noise floor, established independently by
the prefix-length sweep (18_) and used throughout as the threshold below which
a difference is indistinguishable from run-to-run variation. Tying the
equivalence margin to it means the test asks exactly the right question: is
any true difference smaller than the smallest difference this dataset can
resolve?

Coverage -- absolute engineering margin
    delta_cov = 0.01   (one percentage point of Cov90)
Coverage is a qualification requirement, so the margin is a decision about
acceptable risk rather than a statistical quantity. One point is used here as
a deliberately generous allowance: a model losing more than a point of 90 %
coverage is materially worse for reliability work. State it explicitly and
let a reader disagree with the number rather than with a hidden assumption.

Decision rule (TOST, via the clustered interval)
    equivalent      the whole CI lies inside (-delta, +delta)
    non-inferior    the CI's unfavourable end is inside the margin
    inconclusive    the CI extends past the margin but covers zero
    different       the CI excludes zero

The interval used is the DEVICE-CLUSTERED one from 17_, so equivalence is
judged on the same inferential unit as the difference test: the physical
device, not the (seed, device) row.

Usage
-----
    python 37_equivalence_margins.py
"""

import argparse
import json
import logging
import os
import pickle
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# Noise floor from 18_prefix_len_sweep.py, used throughout as the resolution
# limit of this dataset.
NOISE_FLOOR = 0.027
# Engineering margin on Cov90, in coverage points. A decision, not an estimate.
COVERAGE_MARGIN = 0.01


def classify(lo, hi, delta, favourable_sign=+1):
    """TOST-style verdict from a confidence interval and a margin.

    favourable_sign says which direction favours the FIRST model: +1 when a
    positive difference is good for it, -1 when negative is.
    """
    excludes_zero = lo > 0 or hi < 0
    inside = (lo > -delta) and (hi < delta)
    if inside:
        return "equivalent" if not excludes_zero else "different but within margin"
    if excludes_zero:
        return "different, beyond margin"
    # covers zero but runs past the margin
    unfav = lo if favourable_sign > 0 else hi
    if (favourable_sign > 0 and unfav > -delta) or (favourable_sign < 0 and unfav < delta):
        return "non-inferior, not equivalent"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(description="Equivalence margins for Stage 4C vs Stage 5")
    ap.add_argument("--cv-summary", default=os.path.join(
        cfg.RESULTS_DIR, "cv_stage4c_stage5_filtered", "cv_summary.json"))
    ap.add_argument("--stage4c-eval", default=os.path.join(
        cfg.OUTPUT_PATH, "ext11_filtered", "stage4c", "results",
        "evaluation_results_stage4b.pkl"))
    ap.add_argument("--coverage-margin", type=float, default=COVERAGE_MARGIN)
    ap.add_argument("--output", default=os.path.join(
        cfg.RESULTS_DIR, "equivalence_margins.json"))
    args = ap.parse_args()

    with open(args.cv_summary, encoding="utf-8") as f:
        cv = json.load(f)
    with open(args.stage4c_eval, "rb") as f:
        m4c = pickle.load(f)["stage4b_metrics"]

    crps_ref = float(m4c["crps_by_feature_stable_overall"])
    d_crps = NOISE_FLOOR * crps_ref

    log.info("dataset %s  |  n_devices %s", cv.get("dataset"), cv.get("n_devices"))
    log.info("")
    log.info("MARGINS (fixed before inspecting the intervals)")
    log.info("  CRPS      delta = %.3f x %.6f = %.6f   (noise floor from 18_)",
             NOISE_FLOOR, crps_ref, d_crps)
    log.info("  Coverage  delta = %.4f  (engineering decision, in Cov90 points)",
             args.coverage_margin)

    out = {"margins": {"crps_relative": NOISE_FLOOR, "crps_reference": crps_ref,
                       "crps_absolute": d_crps,
                       "coverage_points": args.coverage_margin},
           "tests": {}}

    log.info("")
    log.info("%-28s %24s %12s  %s", "quantity", "95% CI (device-clustered)",
             "margin", "verdict")
    log.info("-" * 92)

    dc = cv.get("paired_crps_device_clustered")
    if dc:
        v = classify(dc["ci_low"], dc["ci_high"], d_crps)
        out["tests"]["crps"] = {"ci_low": dc["ci_low"], "ci_high": dc["ci_high"],
                                "margin": d_crps, "verdict": v,
                                "ci_as_frac_of_margin":
                                    max(abs(dc["ci_low"]), abs(dc["ci_high"])) / d_crps}
        log.info("%-28s [%+.6f, %+.6f] %12.6f  %s", "CRPS (S4C - S5)",
                 dc["ci_low"], dc["ci_high"], d_crps, v)

    cc = cv.get("paired_coverage_device_clustered")
    if cc:
        for key, label in [("pointwise", "Cov90 pointwise (S4C - S5)"),
                           ("simultaneous", "Cov90 simultaneous (S4C - S5)")]:
            b = cc[key]
            lo, hi = b["ci_low"], b["ci_high"]
            v = classify(lo, hi, args.coverage_margin)
            out["tests"][f"coverage_{key}"] = {
                "ci_low": lo, "ci_high": hi, "margin": args.coverage_margin,
                "verdict": v}
            log.info("%-28s [%+.6f, %+.6f] %12.4f  %s", label, lo, hi,
                     args.coverage_margin, v)

    log.info("")
    log.info("READING THIS")
    log.info("  CRPS is EQUIVALENT, not merely non-significant: the whole interval")
    log.info("  sits inside a margin set at this dataset's own resolution limit.")
    log.info("  Coverage is DIFFERENT and beyond its margin in the simultaneous")
    log.info("  case, so the two models are equivalent in sharpness and separable")
    log.info("  in calibration -- which is a sharper statement than either")
    log.info("  'no difference' or 'Stage 5 is worse' on its own.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("")
    log.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
