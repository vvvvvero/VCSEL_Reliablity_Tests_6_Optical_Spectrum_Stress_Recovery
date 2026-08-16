#!/usr/bin/env python
"""Quick screen: Stage 5 gradient ratio scan vs Stage 5 control (lambda_adv=0).

Compares lambda_adv = {0, 0.001, 0.005, 0.01} on the quick 5-fold screen (seed 11).
Requires adversarial variants to IMPROVE W1 vs control AND be NI on CRPSS/Cov90.
"""
import csv, numpy as np

NI_CRPSS = -0.010
NI_COV90 = -0.020
W1_IMPROVE_THRESHOLD = 0.0   # mean(D) < 0 = W1 improved

# Quick 5-fold screen CSVs (seed 11 only)
BASE = r"D:\2026\article\GaN4GaN\output"
CSVS = {
    "ctrl_lad0":   f"{BASE}\\pi_timegan_quickcv_stage5_lt_lad0\\grouped_cv_fold_metrics.csv",
    "lad0.001":    f"{BASE}\\pi_timegan_quickcv_stage5_lt_lad0.001\\grouped_cv_fold_metrics.csv",
    "lad0.005":    f"{BASE}\\pi_timegan_quickcv_stage5_lt_lad0.005\\grouped_cv_fold_metrics.csv",
    "lad0.01":     f"{BASE}\\pi_timegan_quickcv_stage5_lt_lad0.01\\grouped_cv_fold_metrics.csv",
}

def load_model(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        if r["model"] == "stage4b":
            rows[(str(r["seed"]), str(r["fold"]))] = r
    return rows

def g(r, k):
    try: return float(r[k])
    except: return np.nan

def bootstrap_ci(deltas, n_boot=3000, alpha=0.05, rng_seed=0):
    d = np.array(deltas)
    rng = np.random.default_rng(rng_seed)
    means = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))

data = {name: load_model(p) for name, p in CSVS.items()}
keys = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
print(f"Quick screen: {len(keys)} fold-seed pairs")

metrics = [
    ("CRPSS",   "crpss_overall",         +1),
    ("Cov90-pw","coverage_90_pointwise",  +1),
    ("W1",      "w1_increments_overall",  -1),
    ("MACE",    "mace",                   -1),
]

sep = "=" * 90
print(f"\n{sep}")
print("SECTION A: MEANS  (quick 5-fold screen, seed 11)")
print(sep)
print(f"  {'Variant':<14}", end="")
for m, col, _ in metrics:
    print(f"  {m:>12}", end="")
print()
print(f"  {'-' * 70}")
for name, d in data.items():
    print(f"  {name:<14}", end="")
    for m, col, _ in metrics:
        vals = [g(d[k], col) for k in keys]
        print(f"  {np.nanmean(vals):>+9.5f}+-{np.nanstd(vals):.4f}", end="")
    print()

print(f"\n{sep}")
print("SECTION B: PAIRED DIFFERENCES vs ctrl_lad0 (no adversarial)")
print(sep)
ctrl = data["ctrl_lad0"]
results = {}
best_lad = None
best_w1_gain = 0.0

for name in list(data.keys())[1:]:
    print(f"\n  {name} vs ctrl_lad0:")
    print(f"    {'Metric':<10}  {'mean':>10}  {'CI_low':>10}  {'CI_high':>10}  Verdict")
    print(f"    {'-' * 60}")
    model_res = {}
    for m, col, sign in metrics:
        d_new  = [g(data[name][k], col) for k in keys]
        d_base = [g(ctrl[k], col) for k in keys]
        delta  = [a - b for a, b in zip(d_new, d_base)]
        mn = np.mean(delta)
        lo, hi = bootstrap_ci(delta)
        if m == "CRPSS":
            v = "NI-PASS" if lo > NI_CRPSS else "NI-FAIL"
        elif m == "Cov90-pw":
            v = "NI-PASS" if lo > NI_COV90 else "NI-FAIL"
        elif m == "W1":
            v = "W1-IMPROVED" if mn < 0 else "W1-WORSE"
        else:
            v = ""
        model_res[m] = {"mean": mn, "ci_low": lo, "ci_high": hi}
        print(f"    {m:<10}  {mn:>+10.5f}  {lo:>+10.5f}  {hi:>+10.5f}  {v}")
    results[name] = model_res
    # Check if eligible and W1 improved
    crpss_ni = model_res["CRPSS"]["ci_low"] > NI_CRPSS
    cov_ni   = model_res["Cov90-pw"]["ci_low"] > NI_COV90
    w1_imp   = model_res["W1"]["mean"] < W1_IMPROVE_THRESHOLD
    if crpss_ni and cov_ni and w1_imp and model_res["W1"]["mean"] < best_w1_gain:
        best_w1_gain = model_res["W1"]["mean"]
        best_lad = name

print(f"\n{sep}")
print("SECTION C: QUICK-SCREEN DECISION")
print(sep)
for name in list(data.keys())[1:]:
    r = results[name]
    crpss_ni = r["CRPSS"]["ci_low"] > NI_CRPSS
    cov_ni   = r["Cov90-pw"]["ci_low"] > NI_COV90
    w1_imp   = r["W1"]["mean"] < 0
    status = "PASS" if (crpss_ni and cov_ni and w1_imp) else (
        f"FAIL (CRPSS={'OK' if crpss_ni else 'FAIL'}, Cov90={'OK' if cov_ni else 'FAIL'}, W1={'improved' if w1_imp else 'worse'})")
    print(f"  {name:<14}: {status}")

print()
if best_lad is not None:
    print(f"  PROCEED TO FULL CV with: {best_lad}  (best W1 gain: {best_w1_gain:+.5f})")
else:
    print(f"  NO VARIANT passes all criteria. Adversarial training does not improve W1 vs control.")
    print(f"  Stage 5 control (lambda_adv=0) provides only the extra-epoch benefit.")
    print(f"  Recommendation: Use Stage 5 control checkpoint if CRPSS/Cov90 stay NI vs LT-calibrated.")
