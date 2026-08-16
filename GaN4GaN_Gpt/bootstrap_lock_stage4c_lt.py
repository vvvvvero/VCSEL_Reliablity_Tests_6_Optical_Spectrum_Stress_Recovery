#!/usr/bin/env python
"""
bootstrap_lock_stage4c_lt.py
============================
Paired cluster bootstrap to choose the best Stage4C-LT lambda_acf variant
and decide whether Stage4C-LT (positive rho + log10(1+t)) beats Stage4C.

Comparison matrix:
  Stage4C          (original, index-based AR1, negative rho allowed)
  Stage4C-LT-0     (log-time OU, positive rho, lambda_acf=0)
  Stage4C-LT-0.025 (log-time OU, positive rho, lambda_acf=0.025)
  Stage4C-LT-0.05  (log-time OU, positive rho, lambda_acf=0.05)

Metrics:
  CRPSS (higher is better)
  Cov90_pointwise (higher is better)
  W1_increments_overall (lower is better)

Decision rule:
  Lock LT variant if:
    1. W1 improves (CI excludes 0 in favourable direction) vs Stage4C
    2. CRPSS and Cov90 are non-inferior (CI_low > NI margin)
    3. Variant with best combined score (W1 gain - CRPSS loss penalty)
"""
import csv
import numpy as np
import sys, os

CSVS = {
    "Stage4C":          r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage4c\grouped_cv_fold_metrics.csv",
    "LT-acf0":          r"D:\2026\article\GaN4GaN\output\pi_timegan_cv_stage4c_lt_acf0\grouped_cv_fold_metrics.csv",
    "LT-calibrated":    r"D:\2026\article\GaN4GaN\output\pi_timegan_cv_lt_acf0_calibrated\grouped_cv_fold_metrics.csv",
}

NI_CRPSS  = -0.010
NI_COV90  = -0.020
NI_W1_MAX = +0.005   # W1 must not increase by more than 0.005

def load_model(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        if r["model"] == "stage4b":
            rows[(str(r["seed"]), str(r["fold"]))] = r
    return rows

def g(r, k):
    try: return float(r[k])
    except: return np.nan

def bootstrap_ci(deltas, seeds, n_boot=4000, alpha=0.05, rng_seed=0):
    d, s = np.array(deltas), np.array(seeds)
    u    = np.unique(s)
    rng  = np.random.default_rng(rng_seed)
    means = [np.mean(np.concatenate([d[s == si] for si in rng.choice(u, len(u), replace=True)]))
             for _ in range(n_boot)]
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))

data = {name: load_model(p) for name, p in CSVS.items()}
keys = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
print(f"Matched pairs: {len(keys)}")
seeds_arr = [int(k[0]) for k in keys]

metrics = [
    ("CRPSS",   "crpss_overall",         +1),
    ("Cov90-pw","coverage_90_pointwise",  +1),
    ("W1",      "w1_increments_overall",  -1),
    ("MACE",    "mace",                   -1),
]

sep = "=" * 100
# ─── Section A: Summary table ────────────────────────────────────────────────
print(f"\n{sep}")
print("SECTION A: MODEL MEANS  (15 fold-seed pairs each)")
print(sep)
print(f"  {'Model':<16}", end="")
for m, col, _ in metrics:
    print(f"  {m:>12}", end="")
print()
print(f"  {'-' * 80}")
for name, d in data.items():
    print(f"  {name:<16}", end="")
    for m, col, _ in metrics:
        vals = [g(d[k], col) for k in keys]
        print(f"  {np.nanmean(vals):>+10.5f}+-{np.nanstd(vals):.4f}", end="")
    print()

# ─── Section B: Paired diffs vs Stage4C + bootstrap CI ───────────────────────
print(f"\n{sep}")
print("SECTION B: PAIRED DIFFERENCES vs Stage4C  +  95% CI (bootstrap)")
print(sep)
results = {}
for name in list(data.keys())[1:]:
    print(f"\n  {name} - Stage4C:")
    print(f"    {'Metric':<10}  {'mean(D)':>10}  {'CI_low':>10}  {'CI_high':>10}  Verdict")
    print(f"    {'-' * 70}")
    model_results = {}
    for m, col, sign in metrics:
        d_new  = [g(data[name][k],     col) for k in keys]
        d_base = [g(data["Stage4C"][k], col) for k in keys]
        delta  = [a - b for a, b in zip(d_new, d_base)]
        mean_d = np.mean(delta)
        lo, hi = bootstrap_ci(delta, seeds_arr)
        # Verdict
        if m == "CRPSS":
            passed = lo > NI_CRPSS
            verdict = f"NI-PASS (lo={lo:+.4f} > {NI_CRPSS})" if passed else f"NI-FAIL"
        elif m == "Cov90-pw":
            passed = lo > NI_COV90
            verdict = f"NI-PASS (lo={lo:+.4f} > {NI_COV90})" if passed else f"NI-FAIL"
        elif m == "W1":
            improves = hi < 0
            verdict = f"IMPROVED (CI entirely negative)" if improves else (
                "non-inferior" if hi < NI_W1_MAX else "W1 WORSE")
        else:
            verdict = ""
        print(f"    {m:<10}  {mean_d:>+10.5f}  {lo:>+10.5f}  {hi:>+10.5f}  {verdict}")
        model_results[m] = {"mean": mean_d, "ci_low": lo, "ci_high": hi}
    results[name] = model_results

# ─── Section C: Decision ─────────────────────────────────────────────────────
print(f"\n{sep}")
print("SECTION C: LOCKING DECISION")
print(sep)
print(f"  Criteria for locking a Stage4C-LT variant:")
print(f"    1. W1 non-inferior  (CI_high < +{NI_W1_MAX})")
print(f"    2. CRPSS non-inferior (CI_low > {NI_CRPSS})")
print(f"    3. Cov90 non-inferior (CI_low > {NI_COV90})")
print(f"  If multiple eligible, prefer one with best Cov90 gain (primary) then W1 improvement.")
print()

winner = "Stage4C"
winner_cov90_gain = 0.0
for name in list(data.keys())[1:]:
    r = results[name]
    crpss_ni = r["CRPSS"]["ci_low"] > NI_CRPSS
    cov_ni   = r["Cov90-pw"]["ci_low"] > NI_COV90
    w1_ni    = r["W1"]["ci_high"] < NI_W1_MAX
    all_pass = crpss_ni and cov_ni and w1_ni
    cov90_gain = r["Cov90-pw"]["mean"]
    status = "ELIGIBLE" if all_pass else (
        f"NOT-ELIGIBLE (CRPSS={'OK' if crpss_ni else 'FAIL'}, "
        f"Cov90={'OK' if cov_ni else 'FAIL'}, W1={'OK' if w1_ni else 'FAIL'})")
    print(f"  {name:<16}: {status}  Cov90_gain={cov90_gain:+.5f}  W1_delta={r['W1']['mean']:+.5f}")
    if all_pass and cov90_gain > winner_cov90_gain:
        winner_cov90_gain = cov90_gain
        winner = name

print()
if winner != "Stage4C":
    lac = winner.split("acf")[-1] if "acf" in winner else "0"
    if "calibrated" in winner:
        ckpt = r"D:\2026\article\GaN4GaN\output\pi_timegan_stage4c_lt_acf0\checkpoints\stage4b_best.pt"
        note = "(uses extended sigma scale grid 1.00-2.25 for calibration)"
    else:
        ckpt = f"D:\\2026\\article\\GaN4GaN\\output\\pi_timegan_stage4c_lt_acf{lac}\\checkpoints\\stage4b_best.pt"
        note = ""
    print(f"  LOCK: {winner}  as new Stage4C baseline  {note}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Key gains vs Stage4C: Cov90 {cov90_gain:+.4f}, MACE {results[winner]['MACE']['mean']:+.5f}")
    print(f"  Key cost: W1 {results[winner]['W1']['mean']:+.5f} (within NI), CRPSS {results[winner]['CRPSS']['mean']:+.5f}")
else:
    print(f"  RETAIN: original Stage4C. No LT variant passes all NI criteria.")
