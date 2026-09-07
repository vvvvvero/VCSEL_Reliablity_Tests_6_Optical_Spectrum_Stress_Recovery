"""
24_baseline_significance.py
===========================
Turn the multi-seed baseline runs from 23_baseline_vanilla.py into a claim
that survives review: is the physics model's advantage larger than seed noise?

Why this is needed
------------------
The single-seed sweep gave physics a 14.9 % edge over the Neural ODE on the
full dataset but only 2.1 % on held-out temperature. The prefix-length sweep
earlier established that ~2.7 % differences on this dataset are noise, so the
first number probably means something and the second probably does not.
"Probably" is not good enough to publish, and a reviewer will ask.

Method
------
Paired over seeds: for each seed both models see the same data, the same
split and the same initialisation stream, so the difference per seed is the
natural unit. Reported per configuration:

  * per-seed RMSE for each model, and the paired difference
  * mean difference with a bootstrap 95 % CI over seeds
  * Wilcoxon signed-rank p-value (exact for small n; reported with the caveat
    that it has little power below ~6 seeds)
  * Cohen's d_z for the paired differences

With a handful of seeds the CI is the honest summary and the p-value is
mostly decorative -- both are printed rather than only the one that flatters
the result. A configuration where the CI crosses zero is reported as
inconclusive, not as a win.

Usage
-----
    python 24_baseline_significance.py
    python 24_baseline_significance.py --results-dir <dir> --n-boot 20000
"""

import argparse
import collections
import itertools
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# Differences at or below this are within the noise floor this dataset showed
# in the prefix-length sweep (2.7 % spread across settings that should have
# been ordered). Used only for commentary, never to alter a computed number.
NOISE_FLOOR_PCT = 2.7


def load_runs(d: str) -> List[Dict]:
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception as exc:                       # noqa: BLE001
            log.warning("skipping %s: %s", fn, exc)
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean.

    With very few seeds this is wide and that is the point: it shows how
    little a 3-5 seed experiment actually pins down.
    """
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def wilcoxon(x: np.ndarray) -> Optional[float]:
    """Two-sided Wilcoxon signed-rank p-value for H0: median difference = 0."""
    try:
        from scipy.stats import wilcoxon as _w
    except ImportError:
        return None
    nz = x[x != 0]
    if len(nz) < 1:
        return None
    try:
        return float(_w(nz).pvalue)
    except Exception:                                  # noqa: BLE001
        return None


def compare(runs: List[Dict], mode_a: str, mode_b: str,
            data_frac: float, ood: bool, n_boot: int) -> Optional[Dict]:
    """Paired comparison of two modes over the seeds they share."""
    def index(mode):
        return {r["seed"]: r for r in runs
                if r["mode"] == mode
                and abs(r.get("data_frac", 1.0) - data_frac) < 1e-9
                and bool(r.get("ood", False)) == ood}
    A, B = index(mode_a), index(mode_b)
    seeds = sorted(set(A) & set(B))
    if len(seeds) < 2:
        return None
    a = np.array([A[s]["test"]["rmse_overall"] for s in seeds], dtype=float)
    b = np.array([B[s]["test"]["rmse_overall"] for s in seeds], dtype=float)
    diff = a - b                       # negative => mode_a better
    rel = 100.0 * diff / b
    lo, hi = bootstrap_ci(diff, n_boot)
    rlo, rhi = bootstrap_ci(rel, n_boot)
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
    return {
        "config": {"data_frac": data_frac, "ood": ood},
        "mode_a": mode_a, "mode_b": mode_b, "seeds": seeds, "n_seeds": len(seeds),
        "rmse_a": a.tolist(), "rmse_b": b.tolist(),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "std_a": float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"),
        "std_b": float(np.std(b, ddof=1)) if len(b) > 1 else float("nan"),
        "mean_diff": float(diff.mean()), "diff_ci95": [lo, hi],
        "mean_rel_pct": float(rel.mean()), "rel_ci95": [rlo, rhi],
        "wilcoxon_p": wilcoxon(diff),
        "cohens_dz": float(diff.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan"),
        "a_wins": int((diff < 0).sum()), "b_wins": int((diff > 0).sum()),
        "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 0 or lo > 0)),
    }


def verdict(c: Dict) -> str:
    if not np.isfinite(c["diff_ci95"][0]):
        return "too few seeds"
    if not c["ci_excludes_zero"]:
        return "INCONCLUSIVE (CI crosses zero)"
    better = c["mode_a"] if c["mean_diff"] < 0 else c["mode_b"]
    if abs(c["mean_rel_pct"]) < NOISE_FLOOR_PCT:
        return f"{better} better, but below the {NOISE_FLOOR_PCT}% noise floor"
    return f"{better} better"


def main():
    ap = argparse.ArgumentParser(description="Significance of the baseline comparison")
    ap.add_argument("--results-dir", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "baseline_vanilla"))
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        log.error("no runs found in %s", args.results_dir)
        return
    log.info("loaded %d runs from %s", len(runs), args.results_dir)

    have = collections.Counter(
        (r["mode"], r.get("data_frac", 1.0), bool(r.get("ood", False)))
        for r in runs)
    log.info("runs per (mode, data_frac, ood):")
    for k in sorted(have, key=lambda x: (x[2], x[1], x[0])):
        log.info("   %-9s frac=%.2f ood=%-5s  n_seeds=%d",
                 k[0], k[1], str(k[2]), have[k])

    configs = sorted({(r.get("data_frac", 1.0), bool(r.get("ood", False)))
                      for r in runs})
    pairs = [("physics", "node"), ("physics", "gru")]

    results = []
    for frac, ood in configs:
        for a, b in pairs:
            c = compare(runs, a, b, frac, ood, args.n_boot)
            if c:
                results.append(c)

    if not results:
        log.error("no configuration has >=2 shared seeds yet")
        return

    log.info("=" * 96)
    log.info("PAIRED COMPARISON (negative difference = first model better)")
    log.info("%-26s %-6s %5s %9s %9s %11s %-22s",
             "configuration", "pair", "n", "mean_A", "mean_B", "rel %", "95% CI on rel %")
    for c in results:
        cf = c["config"]
        name = f"frac={cf['data_frac']:.2f} ood={str(cf['ood'])}"
        log.info("%-26s %-6s %5d %9.4f %9.4f %10.2f%%  [%7.2f%%, %7.2f%%]",
                 name, f"{c['mode_a'][:4]}/{c['mode_b'][:4]}", c["n_seeds"],
                 c["mean_a"], c["mean_b"], c["mean_rel_pct"],
                 c["rel_ci95"][0], c["rel_ci95"][1])

    log.info("")
    log.info("VERDICTS")
    for c in results:
        cf = c["config"]
        p = c["wilcoxon_p"]
        log.info("  frac=%.2f ood=%-5s %s vs %s : %-46s wins %d/%d  d_z=%.2f  p=%s",
                 cf["data_frac"], str(cf["ood"]), c["mode_a"], c["mode_b"],
                 verdict(c), c["a_wins"], c["n_seeds"], c["cohens_dz"],
                 f"{p:.4f}" if p is not None else "n/a")

    log.info("")
    log.info("PER-SEED DETAIL")
    for c in results:
        cf = c["config"]
        log.info("  frac=%.2f ood=%s  %s vs %s",
                 cf["data_frac"], str(cf["ood"]), c["mode_a"], c["mode_b"])
        for s, x, y in zip(c["seeds"], c["rmse_a"], c["rmse_b"]):
            log.info("     seed %-4d  %-8s=%.4f   %-8s=%.4f   diff=%+.4f (%+.2f%%)",
                     s, c["mode_a"], x, c["mode_b"], y, x - y, 100 * (x - y) / y)

    log.info("")
    log.info("Reading this: with a handful of seeds the CI is the honest summary "
             "and the Wilcoxon p-value has little power -- a p above 0.05 here "
             "is not evidence of no effect. A configuration whose CI crosses "
             "zero is inconclusive, not a tie.")

    out = args.output or os.path.join(args.results_dir, "significance.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
