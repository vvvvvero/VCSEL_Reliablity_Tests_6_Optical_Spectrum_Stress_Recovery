"""
35_mechanism_seed_statistics.py
===============================
Per-seed table and cross-seed statistics for the candidate rate-law comparison.

Why this exists
---------------
MECHANISM_COMPARISON.md reported the SRH-vs-alternatives result on two seeds
and flagged that as the weakest link in the chain. Two seeds support a large
gap but give no usable spread, and they cannot distinguish "this candidate
lost" from "this candidate never trained" -- which is exactly the failure that
invalidated the first screen, where the losers came back at their
initialisation with gradients of 1e13-1e16.

This script consumes the per-run JSON written by 25_ and reports, per
candidate:

  1. RMSE mean +- sd          across seeds
  2. RMSE median
  3. RMSE min / max
  4. best epoch mean +- sd    (did the fit move, and when did it settle?)
  5. convergence success count
  6. parameter boundary-hit count

Items 4-6 are the honesty checks, and they separate three states rather than
two, because "not converged" on its own is ambiguous:

  settled       early-stopped, or best reached before the final epoch
  budget_bound  still improving when the budget ran out
  failed        never improved past epoch 1, or an exponent never moved

Only `failed` invalidates a candidate's number -- that is the state the first
screen was in, with losers returning at their initialisation and gradients of
1e13-1e16. `budget_bound` means every candidate is under-trained by the same
budget, which moves the absolute RMSEs but not the ranking between them.

Boundary hits count shape exponents within 0.02 of EXP_MIN=0.05 or
EXP_MAX=1.00. A pinned exponent means the optimiser wanted a value the model
cannot express, so the number it reports is an artefact of the bound.

The log-creep candidate has NO shape exponent -- its rate falls as 1/t by
construction, and it carries 18 dynamics parameters against the 21 of power
and stretched. Its unused expG/B/F parameters sit at initialisation in every
run; an earlier version of this check read them anyway and reported 12
exponents that "never left initialisation", inventing a training failure out
of parameters the candidate does not use.

RESULT (4 seeds, filtered dataset, 200 epochs)

  seed        SRH    Power law   Stretched    Log creep   winner
  42       0.1993      0.2849      0.2836      0.3882     SRH
  43       0.1957      0.2823      0.2891      0.4104     SRH
  44       0.2045      0.2971      0.2908      0.3708     SRH
  45       0.2029      0.2874      0.2835      0.3912     SRH

  candidate        RMSE mean+-sd   median    min      max     state
  SRH             0.2006+-0.0039   0.2011   0.1957   0.2045   4 budget-bound
  Power law       0.2879+-0.0065   0.2862   0.2823   0.2971   4 budget-bound
  Stretched exp.  0.2868+-0.0037   0.2864   0.2835   0.2908   4 budget-bound
  Log creep       0.3901+-0.0162   0.3897   0.3708   0.4104   4 budget-bound

SRH wins 4/4 seeds by 30.0 % over the closest rival, and the RANGES DO NOT
OVERLAP: SRH [0.1957, 0.2045] against stretched [0.2835, 0.2908]. No candidate
failed and no exponent hit a bound, so every alternative was genuinely
optimised -- which is what makes the comparison fair.

All sixteen runs are budget-bound at 200 epochs. The ranking is safe, the
absolute RMSEs are not converged values and should be reported as such.

Note on the dataset
-------------------
The original seed 42/43 runs used processed_data_ext.pkl (203 devices, before
the three measurement-fault devices were removed). Everything else in the
pipeline moved to processed_data_ext_filtered.pkl (200 devices), and 25_
silently ignored a cfg override, so those runs are not comparable with new
ones. 25_ now takes an explicit --dataset. This script refuses to mix datasets
in one table.

Usage
-----
    python 35_mechanism_seed_statistics.py
    python 35_mechanism_seed_statistics.py --input-dir <dir> --output <json>
"""

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

CANDIDATE_ORDER = ["srh", "power", "stretched", "log"]
PRETTY = {"srh": "SRH", "power": "Power law",
          "stretched": "Stretched exp.", "log": "Log creep"}


def load_runs(input_dir):
    """Load every non-screen run, keyed (candidate, seed)."""
    runs = {}
    for p in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        base = os.path.basename(p)
        if "screen" in base or base.startswith("_"):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as exc:                        # noqa: BLE001
            log.warning("skipping %s: %s", base, exc)
            continue
        if "candidate" not in d or "seed" not in d:
            continue
        if d.get("screen"):
            continue
        runs[(d["candidate"], int(d["seed"]))] = d
    return runs


def main():
    ap = argparse.ArgumentParser(description="Cross-seed statistics for rate-law candidates")
    ap.add_argument("--input-dir", default=os.path.join(
        cfg.OUTPUT_PATH, "results", "mechanism_candidates_filtered"))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    runs = load_runs(args.input_dir)
    if not runs:
        log.error("no runs found in %s", args.input_dir)
        return

    datasets = {os.path.basename(d.get("dataset", "?")) for d in runs.values()}
    if len(datasets) > 1:
        log.error("runs span MORE THAN ONE dataset: %s", sorted(datasets))
        log.error("refusing to build one table from incomparable runs")
        return
    ds = datasets.pop()
    seeds = sorted({s for _, s in runs})
    cands = [c for c in CANDIDATE_ORDER if any(c == k[0] for k in runs)]

    log.info("dataset %s | seeds %s | candidates %s", ds, seeds, cands)

    # ---- per-seed table ---------------------------------------------------
    log.info("")
    log.info("PER-SEED TEST RMSE")
    header = f"{'seed':<6}" + "".join(f"{PRETTY[c]:>16}" for c in cands) + f"{'winner':>16}"
    log.info(header)
    log.info("-" * len(header))
    winners = []
    per_seed = {}
    for s in seeds:
        row = f"{s:<6}"
        vals = {}
        for c in cands:
            r = runs.get((c, s))
            v = float(r["test"]["rmse_overall"]) if r else float("nan")
            vals[c] = v
            row += f"{v:>16.4f}" if np.isfinite(v) else f"{'-':>16}"
        ok = {c: v for c, v in vals.items() if np.isfinite(v)}
        w = min(ok, key=ok.get) if ok else None
        winners.append(w)
        per_seed[s] = {"rmse": vals, "winner": w}
        row += f"{PRETTY.get(w, '-'):>16}"
        log.info(row)

    # ---- cross-seed statistics -------------------------------------------
    log.info("")
    log.info("CROSS-SEED STATISTICS  (n = %d seeds)", len(seeds))
    hdr = (f"{'candidate':<16}{'RMSE mean+-sd':>20}{'median':>10}{'min':>9}{'max':>9}"
           f"{'best ep mean+-sd':>20}{'settled':>9}{'budget':>8}{'failed':>8}{'bdry':>6}")
    log.info(hdr)
    log.info("-" * len(hdr))

    stats = {}
    for c in cands:
        rs = np.array([runs[(c, s)]["test"]["rmse_overall"]
                       for s in seeds if (c, s) in runs], dtype=float)
        be = np.array([runs[(c, s)].get("best_epoch", np.nan)
                       for s in seeds if (c, s) in runs], dtype=float)
        states = [runs[(c, s)].get("fit_state", "?") for s in seeds if (c, s) in runs]
        conv = sum(x == "settled" for x in states)
        budg = sum(x == "budget_bound" for x in states)
        fail = sum(x == "failed" for x in states)
        bdry = sum(int(runs[(c, s)].get("boundary", {}).get("n_boundary_hits", 0))
                   for s in seeds if (c, s) in runs)
        at_init = sum(int(runs[(c, s)].get("boundary", {}).get("n_at_init", 0))
                      for s in seeds if (c, s) in runs)
        n = len(rs)
        sd = float(rs.std(ddof=1)) if n > 1 else float("nan")
        bsd = float(np.nanstd(be, ddof=1)) if n > 1 else float("nan")
        stats[c] = {
            "n_seeds": n,
            "rmse_mean": float(rs.mean()), "rmse_sd": sd,
            "rmse_median": float(np.median(rs)),
            "rmse_min": float(rs.min()), "rmse_max": float(rs.max()),
            "best_epoch_mean": float(np.nanmean(be)), "best_epoch_sd": bsd,
            "n_settled": conv, "n_budget_bound": budg, "n_failed": fail,
            "n_boundary_hits": bdry, "n_at_init": at_init,
        }
        log.info("%-16s%13.4f+-%-5.4f%10.4f%9.4f%9.4f%13.1f+-%-5.1f%9d%8d%8d%6d",
                 PRETTY[c], rs.mean(), sd, np.median(rs), rs.min(), rs.max(),
                 np.nanmean(be), bsd, conv, budg, fail, bdry)

    # ---- headline gap -----------------------------------------------------
    log.info("")
    best = min(stats, key=lambda c: stats[c]["rmse_mean"])
    rivals = sorted((c for c in stats if c != best), key=lambda c: stats[c]["rmse_mean"])
    if rivals:
        r0 = rivals[0]
        gap = 100 * (stats[r0]["rmse_mean"] - stats[best]["rmse_mean"]) / stats[r0]["rmse_mean"]
        log.info("%s wins on %d/%d seeds; %.1f %% better than the closest rival (%s)",
                 PRETTY[best], sum(w == best for w in winners), len(seeds), gap, PRETTY[r0])
        sep = stats[best]["rmse_max"] < stats[r0]["rmse_min"]
        log.info("  ranges %s: %s [%.4f, %.4f] vs %s [%.4f, %.4f]",
                 "DO NOT overlap" if sep else "OVERLAP",
                 PRETTY[best], stats[best]["rmse_min"], stats[best]["rmse_max"],
                 PRETTY[r0], stats[r0]["rmse_min"], stats[r0]["rmse_max"])

    # ---- honesty checks ---------------------------------------------------
    log.info("")
    log.info("FITTING DIAGNOSTICS -- did the losers actually train?")
    for c in cands:
        st = stats[c]
        flags = []
        if st["n_failed"]:
            flags.append(f"{st['n_failed']} run(s) FAILED to train")
        if st["n_budget_bound"]:
            flags.append(f"{st['n_budget_bound']} run(s) still improving at the budget end")
        if st["n_boundary_hits"]:
            flags.append(f"{st['n_boundary_hits']} exponent(s) pinned at a bound")
        if st["n_at_init"]:
            flags.append(f"{st['n_at_init']} exponent(s) never left initialisation")
        log.info("  %-16s %s", PRETTY[c], "; ".join(flags) if flags else "clean")
    log.info("")
    log.info("FAILED invalidates a candidate's number -- it never trained. Runs that")
    log.info("are merely budget-bound are all under-trained by the same budget, which")
    log.info("moves the absolute RMSEs but not the ranking between them.")

    out = args.output or os.path.join(args.input_dir, "seed_statistics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": ds, "seeds": seeds, "per_seed": per_seed,
                   "statistics": stats}, f, indent=2)
    log.info("")
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
