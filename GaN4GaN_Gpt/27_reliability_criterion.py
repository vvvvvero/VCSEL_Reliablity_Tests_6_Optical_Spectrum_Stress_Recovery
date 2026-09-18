"""
27_reliability_criterion.py
===========================
A model-selection criterion built from reliability-engineering requirements,
used where a GAN pipeline would use its discriminator.

The naming, deliberately
------------------------
This is a CRITERION, not a discriminator. In the GAN literature a
discriminator is a network trained adversarially against the generator to
supply gradients. What follows trains nothing, has no parameters, and provides
no gradient -- it scores a finished predictive distribution against what a
qualification report needs. Calling it a "discriminator" would invite the
reasonable objection that it has no architecture and no minimax game, and
would obscure the actual claim, which is stronger:

    the adversarial term is not improved, it is REPLACED -- by criteria that
    are closed-form, need no training, cannot collapse, and are already
    standard in reliability work.

Why the discriminator cannot serve this role
--------------------------------------------
A per-sample discriminator asks "does this trajectory look real?". Interval
width and tail mass are properties of a DISTRIBUTION, not of any one sample,
so no per-sample judge can observe them. What it can observe is that genuine
tail trajectories are rare, so it learns to call them suspicious, and the
generator's cheapest reply is to stop producing them. Measured here: Stage 5
narrows intervals (sample diversity -16 % in three epochs) and its PIT tail
mass rises to 0.2325 against an expected 0.20 -- observations piling up
outside the interval. Those rare trajectories are the entire object of a
reliability study.

This is structural rather than a tuning failure. The repository already tried
the tuning route: the discriminator was cut from 64x2 to 16x1 and five guards
were added (sigma floor, diversity floor, accuracy ceiling, CRPS tolerance,
physics preservation). Shrinkage fell from ~40 % to ~16 %; it did not stop,
because the adversarial gradient keeps pointing that way.

The criterion
-------------
Four terms, each answering a question an engineer actually asks:

    Winkler_90        is the interval usable?   width + (2/a) * shortfall,
                      so at 90 % a miss costs 20x the width it saved
    |PIT_tail - 0.20| is the tail probability right?
    PIT KS            is the distribution the right SHAPE?
    (0.90 - Cov_90)+  does it meet the coverage requirement? ONE-SIDED --
                      over-covering is free, under-covering is penalised,
                      which is the engineering risk posture

    ROCC = W + lam_tail*|PIT_tail-0.20| + lam_ks*PIT_KS + lam_cov*(0.90-Cov90)+

Lower is better. Weights are not chosen by taste: each is set so that its
term's spread across the candidate models equals the Winkler term's spread,
so no term silently dominates the ranking. From the four models scored in
26_reliability_assessment.py that gives lam_tail = 0.62, lam_cov = 0.53, and
lam_ks = 0.87. Re-derive with --calibrate-weights when the candidate set
changes, and report the values used -- they are part of the criterion.

Robustness
----------
The top-two gap is small (0.016), so the ranking was checked against 125
weight combinations spanning lam in {0, 0.3, 0.62, 1.0, 2.0} for all three
terms. Adversarial fine-tuning ranks LAST in 107 of 125 (86 %), and it loses
to the best model on all four terms independently -- Winkler 0.8657 vs 0.8315,
tail deviation 0.0325 vs 0.0227, KS 0.1451 vs 0.1251, coverage shortfall
0.0748 vs 0.0359. The conclusion therefore does not rest on the weighting.

Which Stage 4C variant comes first is NOT robust (offset wins 66/125, sigma
46/125), and should not be reported as a ranking -- they are two good models
trading sharpness against coverage.

Usage
-----
    python 27_reliability_criterion.py
    python 27_reliability_criterion.py --calibrate-weights
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# Nominal coverage the criterion holds the model to.
NOMINAL = 0.90
# Expected PIT mass in the two extreme bins of a 10-bin histogram.
PIT_TAIL_EXPECTED = 0.20

# Derived by equalising each term's spread with Winkler's across the four
# models in reliability_assessment.json (see --calibrate-weights).
LAMBDA_TAIL = 0.62
LAMBDA_KS   = 0.87
LAMBDA_COV  = 0.53


def terms(o: Dict) -> Dict[str, float]:
    """The four criterion terms for one model's overall metrics."""
    return {
        "winkler_90": float(o["winkler_90"]),
        "pit_tail_dev": abs(float(o["pit_tail_mass"]) - PIT_TAIL_EXPECTED),
        "pit_ks": float(o["pit_ks"]),
        "cov_shortfall": max(0.0, NOMINAL - float(o["cov_90"])),
    }


def rocc(t: Dict[str, float],
         lam_tail: float = LAMBDA_TAIL,
         lam_ks: float = LAMBDA_KS,
         lam_cov: float = LAMBDA_COV) -> float:
    return (t["winkler_90"]
            + lam_tail * t["pit_tail_dev"]
            + lam_ks * t["pit_ks"]
            + lam_cov * t["cov_shortfall"])


def calibrate(all_terms: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Weights that equalise each term's spread with the Winkler term's.

    A criterion whose terms have different natural scales is really a
    criterion with one term. Equalising spreads makes the weighting explicit
    and reproducible rather than a matter of taste.
    """
    def spread(key):
        v = [t[key] for t in all_terms.values()]
        return max(v) - min(v)
    w = spread("winkler_90")
    out = {}
    for key, name in [("pit_tail_dev", "lambda_tail"),
                      ("pit_ks", "lambda_ks"),
                      ("cov_shortfall", "lambda_cov")]:
        s = spread(key)
        out[name] = float(w / s) if s > 1e-12 else float("nan")
    return out


# Grid used for the robustness sweep. The docstring above quotes "107 of 125"
# from this sweep; it is computed and written out here so the claim rests on a
# file rather than on a number typed into a comment.
WEIGHT_GRID = [0.0, 0.3, 0.62, 1.0, 2.0]


def weight_grid_sweep(all_terms, grid=None):
    """Rank every model under each (lam_tail, lam_ks, lam_cov) in the grid.

    Reports how often each model comes last and how often it comes first. A
    conclusion that survives the whole grid does not depend on the particular
    weights chosen, which is the only reason those weights are defensible.
    """
    import itertools
    grid = grid or WEIGHT_GRID
    models = list(all_terms)
    last = {m: 0 for m in models}
    first = {m: 0 for m in models}
    rows = []
    for lt, lk, lc in itertools.product(grid, repeat=3):
        order = sorted(models, key=lambda m: rocc(all_terms[m], lt, lk, lc))
        first[order[0]] += 1
        last[order[-1]] += 1
        rows.append({"lambda_tail": lt, "lambda_ks": lk, "lambda_cov": lc,
                     "ranking": order})
    n = len(rows)
    return {"grid": grid, "n_weightings": n, "combinations": rows,
            "n_first": first, "n_last": last,
            "frac_last": {m: last[m] / n for m in models},
            "frac_first": {m: first[m] / n for m in models}}


def main():
    ap = argparse.ArgumentParser(description="Reliability-oriented model selection")
    ap.add_argument("--input", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "reliability_assessment.json"))
    ap.add_argument("--output", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "reliability_criterion.json"))
    ap.add_argument("--weight-grid", action="store_true",
                    help="sweep the 5x5x5 weight grid and write the robustness file")
    ap.add_argument("--grid-output", type=str, default=None,
                    help="where to write the sweep (default: reliability_weight_grid.json)")
    ap.add_argument("--calibrate-weights", action="store_true",
                    help="Re-derive the weights from the current candidate set")
    ap.add_argument("--lam-tail", type=float, default=LAMBDA_TAIL)
    ap.add_argument("--lam-ks", type=float, default=LAMBDA_KS)
    ap.add_argument("--lam-cov", type=float, default=LAMBDA_COV)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        log.error("no assessment at %s -- run 26_reliability_assessment.py first",
                  args.input)
        return
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    all_terms = {run: terms(r["overall"]) for run, r in data.items()
                 if "overall" in r and "winkler_90" in r["overall"]}
    if not all_terms:
        log.error("no scored models in %s", args.input)
        return

    lam_tail, lam_ks, lam_cov = args.lam_tail, args.lam_ks, args.lam_cov
    if args.calibrate_weights:
        w = calibrate(all_terms)
        lam_tail = w.get("lambda_tail", lam_tail)
        lam_ks = w.get("lambda_ks", lam_ks)
        lam_cov = w.get("lambda_cov", lam_cov)
        log.info("calibrated weights: tail=%.2f ks=%.2f cov=%.2f",
                 lam_tail, lam_ks, lam_cov)
    else:
        log.info("weights: tail=%.2f ks=%.2f cov=%.2f  (--calibrate-weights to re-derive)",
                 lam_tail, lam_ks, lam_cov)

    scored = {run: {"terms": t, "rocc": rocc(t, lam_tail, lam_ks, lam_cov)}
              for run, t in all_terms.items()}
    order = sorted(scored, key=lambda r: scored[r]["rocc"])

    log.info("=" * 92)
    log.info("RELIABILITY-ORIENTED CALIBRATION CRITERION  (lower is better)")
    log.info("%-22s %10s %11s %9s %12s %11s", "model", "Winkler90",
             "|tail-.20|", "PIT KS", "cov shortfall", "ROCC")
    for run in order:
        t, v = scored[run]["terms"], scored[run]["rocc"]
        log.info("%-22s %10.4f %11.4f %9.4f %12.4f %11.4f",
                 run, t["winkler_90"], t["pit_tail_dev"], t["pit_ks"],
                 t["cov_shortfall"], v)

    best, worst = order[0], order[-1]
    log.info("")
    log.info("Best : %s  (ROCC %.4f)", best, scored[best]["rocc"])
    log.info("Worst: %s  (ROCC %.4f)", worst, scored[worst]["rocc"])

    adv = [r for r in order if "stage5" in r.lower()]
    if adv:
        a = adv[0]
        rank = order.index(a) + 1
        log.info("")
        log.info("Adversarial fine-tuning ranks %d of %d under this criterion.", rank, len(order))
        log.info("  Its coverage shortfall is %.4f against %.4f for the best model,",
                 scored[a]["terms"]["cov_shortfall"], scored[best]["terms"]["cov_shortfall"])
        log.info("  and its PIT tail deviation %.4f against %.4f -- observations",
                 scored[a]["terms"]["pit_tail_dev"], scored[best]["terms"]["pit_tail_dev"])
        log.info("  sitting outside the interval rather than inside it.")

    if args.weight_grid:
        sw = weight_grid_sweep(all_terms)
        log.info("")
        log.info("WEIGHT-GRID ROBUSTNESS  (%d weightings, lambda in %s)",
                 sw["n_weightings"], sw["grid"])
        log.info("%-22s %12s %12s", "model", "ranks first", "ranks last")
        for m in sorted(all_terms, key=lambda m: -sw["n_last"][m]):
            log.info("%-22s %7d (%3.0f%%) %7d (%3.0f%%)", m,
                     sw["n_first"][m], 100 * sw["frac_first"][m],
                     sw["n_last"][m], 100 * sw["frac_last"][m])
        gpath = args.grid_output or os.path.join(
            os.path.dirname(args.output), "reliability_weight_grid.json")
        with open(gpath, "w", encoding="utf-8") as f:
            json.dump(sw, f, indent=2)
        log.info("Saved grid -> %s", gpath)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"weights": {"lambda_tail": lam_tail, "lambda_ks": lam_ks,
                               "lambda_cov": lam_cov},
                   "nominal": NOMINAL, "pit_tail_expected": PIT_TAIL_EXPECTED,
                   "scored": scored, "ranking": order}, f, indent=2)
    log.info("")
    log.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
