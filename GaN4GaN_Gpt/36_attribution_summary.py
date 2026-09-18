"""
36_attribution_summary.py
=========================
Aggregate the per-backbone mechanism attributions into one reportable table.

Why this exists
---------------
29_ writes one JSON per backbone, and its own output warns that a single fit's
split should not be quoted. The cross-backbone agreement that lifts that
warning was computed by hand in a conversation and never written to a file, so
the numbers in the write-up had no artefact behind them. This script produces
that artefact.

For every observable it reports, across the supplied backbones:

  * the dominant latent, and whether every backbone agrees on it
  * each share as mean +- sd, with min/max
  * the spread of shares across backbones, which is the quotable uncertainty

Rows whose sparsity mask admits a single latent are labelled `forced` and
excluded from the agreement count: their share is the prior restated, not a
finding, so counting them would inflate the agreement statistic.

What agreement does and does not establish
------------------------------------------
Agreement across independent fits rules out run-to-run optimisation noise.
It does NOT establish that the shares are physically correct -- fits of the
same model under the same sparsity mask and sign constraints can agree and be
wrong together. Validating the split against physics needs independent
evidence, such as comparing fitted activation energies with literature values
for the mechanism each latent represents.

Usage
-----
    python 36_attribution_summary.py
    python 36_attribution_summary.py --input <a.json> --input <b.json> ...
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

DEFAULT_BACKBONES = ["ext11_filtered", "seed101", "seed202", "seed303"]


def main():
    ap = argparse.ArgumentParser(description="Cross-backbone attribution summary")
    ap.add_argument("--input", action="append", default=None,
                    help="mechanism_attribution.json; repeat per backbone")
    ap.add_argument("--output", type=str, default=os.path.join(
        cfg.RESULTS_DIR, "mechanism_attribution_summary.json"))
    args = ap.parse_args()

    paths = args.input or [
        os.path.join(cfg.OUTPUT_PATH, b, "results", "mechanism_attribution.json")
        for b in DEFAULT_BACKBONES]
    paths = [p for p in paths if os.path.exists(p)]
    if len(paths) < 2:
        log.error("need at least two attribution files; found %d", len(paths))
        return

    J, names = [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            J.append(json.load(f))
        names.append(os.path.basename(os.path.dirname(os.path.dirname(p))))

    hours = {d.get("hours") for d in J}
    datasets = {os.path.basename(str(d.get("dataset", "?"))) for d in J}
    if len(hours) > 1 or len(datasets) > 1:
        log.error("inputs disagree on hours %s or dataset %s -- refusing to merge",
                  sorted(hours), sorted(datasets))
        return

    L, F, K = J[0]["latents"], J[0]["features"], J[0]["row_kind"]
    log.info("backbones %s | t = %s h | %s", names, hours.pop(), datasets.pop())

    summary, agree, total, all_sd = {}, 0, 0, []
    log.info("")
    log.info("%-14s %-8s %-7s %17s %18s", "feature", "kind", "dominant",
             "share mean+-sd", "range")
    log.info("-" * 72)
    for f in F:
        vals = {n: np.array([d["share_mean"][f][n] for d in J]) for n in L}
        doms = [max(L, key=lambda n: d["share_mean"][f][n]) for d in J]
        same = len(set(doms)) == 1
        rec = {"kind": K[f], "dominant": doms[0], "dominant_agrees": bool(same),
               "dominant_per_backbone": doms, "shares": {}}
        for n in L:
            v = vals[n]
            if v.max() <= 0:
                continue
            rec["shares"][n] = {"mean": float(v.mean()),
                                "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                "min": float(v.min()), "max": float(v.max()),
                                "per_backbone": [float(x) for x in v]}
            if K[f] == "free":
                all_sd.append(rec["shares"][n]["sd"])
        summary[f] = rec
        if K[f] == "free":
            total += 1
            agree += same
            dv = vals[doms[0]]
            log.info("%-14s %-8s %-7s %10.3f+-%-6.3f %8.3f-%-8.3f %s",
                     f, K[f], doms[0], dv.mean(),
                     dv.std(ddof=1) if len(dv) > 1 else 0.0,
                     dv.min(), dv.max(), "" if same else "<< DISAGREES")
        else:
            log.info("%-14s %-8s %-7s %17s", f, K[f], doms[0], "(prior, not a finding)")

    sd = np.array(all_sd)
    stats = {"n_backbones": len(J), "backbones": names,
             "dominant_agreement": f"{agree}/{total}",
             "n_free_rows": total, "n_agreeing": agree,
             "share_sd_mean": float(sd.mean()), "share_sd_median": float(np.median(sd)),
             "share_sd_max": float(sd.max()), "n_share_terms": int(sd.size)}
    log.info("")
    log.info("dominant mechanism agrees on %d/%d free rows", agree, total)
    log.info("share sd across backbones: mean %.4f  median %.4f  max %.4f  (n=%d)",
             sd.mean(), np.median(sd), sd.max(), sd.size)
    log.info("")
    log.info("Agreement rules out run-to-run optimisation noise. It does NOT show")
    log.info("the shares are physically correct -- fits of the same model under the")
    log.info("same mask and sign constraints can agree and be wrong together.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"statistics": stats, "per_feature": summary,
                   "sources": paths}, f, indent=2)
    log.info("")
    log.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
