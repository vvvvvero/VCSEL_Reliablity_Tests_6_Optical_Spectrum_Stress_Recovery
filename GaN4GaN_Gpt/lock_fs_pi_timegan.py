#!/usr/bin/env python
"""Final FS-PI-TimeGAN locking decision.

Compares:
  Stage4C-LT-calibrated  (the new Stage 4C with positive rho + log-time OU + extended calibration)
  Stage5-ctrl            (LT-calibrated + 1 extra CRPS epoch, no effective adversarial signal)

If Stage5-ctrl is NI on all metrics vs LT-calibrated AND provides any improvement,
it becomes FS-PI-TimeGAN.  Otherwise, LT-calibrated is locked.

Note: All Stage 5 adversarial lambda_adv variants (0, 0.001, 0.005, 0.01) produced
identical checkpoints (best epoch = 1 for all).  Adversarial training does not
contribute measurably at these gradient ratios (< 0.1% of CRPS gradient).
"""
import csv, numpy as np

NI_CRPSS  = -0.010
NI_COV90  = -0.020
NI_W1_MAX = +0.005
NI_MACE   = +0.010

CSVS = {
    "LT-calibrated": r"D:\2026\article\GaN4GaN\output\pi_timegan_cv_lt_acf0_calibrated\grouped_cv_fold_metrics.csv",
    "Stage5-ctrl":   r"D:\2026\article\GaN4GaN\output\pi_timegan_fullcv_stage5_ctrl\grouped_cv_fold_metrics.csv",
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

def bootstrap_ci(deltas, seeds, n_boot=4000, alpha=0.05, rng_seed=0):
    d, s = np.array(deltas), np.array(seeds)
    u    = np.unique(s)
    rng  = np.random.default_rng(rng_seed)
    means = [np.mean(np.concatenate([d[s == si] for si in rng.choice(u, len(u), replace=True)]))
             for _ in range(n_boot)]
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))

data = {name: load_model(p) for name, p in CSVS.items()}
keys = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
seeds_arr = [int(k[0]) for k in keys]
print(f"Matched pairs: {len(keys)} (should be 15)")

metrics = [
    ("CRPSS",   "crpss_overall",         +1),
    ("Cov90-pw","coverage_90_pointwise",  +1),
    ("W1",      "w1_increments_overall",  -1),
    ("MACE",    "mace",                   -1),
]

sep = "=" * 90

print(f"\n{sep}")
print("SECTION A: FINAL COMPARISON MEANS  (15 fold-seed pairs)")
print(sep)
print(f"  {'Model':<18}", end="")
for m, col, _ in metrics:
    print(f"  {m:>12}", end="")
print()
print(f"  {'-' * 75}")
for name, d in data.items():
    print(f"  {name:<18}", end="")
    for m, col, _ in metrics:
        vals = [g(d[k], col) for k in keys]
        print(f"  {np.nanmean(vals):>+9.5f}+-{np.nanstd(vals):.4f}", end="")
    print()

print(f"\n{sep}")
print("SECTION B: Stage5-ctrl vs LT-calibrated  (paired cluster bootstrap 95% CI)")
print(sep)
print(f"  {'Metric':<10}  {'mean(D)':>10}  {'CI_low':>10}  {'CI_high':>10}  NI-criterion  Verdict")
print(f"  {'-' * 75}")

results = {}
ctrl = data["LT-calibrated"]
cand = data["Stage5-ctrl"]
all_ni = True
any_improvement = False

for m, col, sign in metrics:
    d_cand = [g(cand[k], col) for k in keys]
    d_base = [g(ctrl[k], col) for k in keys]
    delta  = [a - b for a, b in zip(d_cand, d_base)]
    mn = np.mean(delta)
    lo, hi = bootstrap_ci(delta, seeds_arr)

    if m == "CRPSS":
        ni_pass = lo > NI_CRPSS
        ni_str  = f"> {NI_CRPSS}"
    elif m == "Cov90-pw":
        ni_pass = lo > NI_COV90
        ni_str  = f"> {NI_COV90}"
    elif m == "W1":
        ni_pass = hi < NI_W1_MAX
        ni_str  = f"hi < +{NI_W1_MAX}"
    elif m == "MACE":
        ni_pass = hi < NI_MACE
        ni_str  = f"hi < +{NI_MACE}"
    else:
        ni_pass = True
        ni_str  = ""

    improved = (mn * sign) > 1e-6
    if improved:
        any_improvement = True
    verdict  = ("PASS" if ni_pass else "FAIL") + (" + IMPROVED" if improved else "")
    if not ni_pass:
        all_ni = False
    results[m] = {"mean": mn, "ci_low": lo, "ci_high": hi, "ni_pass": ni_pass}
    print(f"  {m:<10}  {mn:>+10.5f}  {lo:>+10.5f}  {hi:>+10.5f}  {ni_str:<14}  {verdict}")

print(f"\n{sep}")
print("SECTION C: FINAL LOCKING DECISION — FS-PI-TimeGAN")
print(sep)
print(f"  Stage5-ctrl NI on all metrics: {'YES' if all_ni else 'NO'}")
print(f"  Stage5-ctrl improves any metric vs LT-calibrated: {'YES' if any_improvement else 'NO'}")
print()

if all_ni and any_improvement:
    print(f"  *** FINAL MODEL: Stage5-ctrl ***")
    print(f"  Checkpoint: D:\\2026\\article\\GaN4GaN\\output\\pi_timegan_stage5_lt_lad0\\stage5_best.pt")
    print(f"  (= LT-calibrated + 1 extra CRPS-optimisation epoch)")
    print()
    print(f"  Interpretation: Stage 5 provides marginal further optimisation.")
    print(f"  Adversarial contribution: ZERO (all lambda_adv variants produced identical checkpoints).")
    print(f"  The benefit comes entirely from 1 additional training epoch.")
else:
    print(f"  *** FINAL MODEL: LT-calibrated (Stage4C) ***")
    print(f"  Checkpoint: D:\\2026\\article\\GaN4GaN\\output\\pi_timegan_stage4c_lt_acf0\\checkpoints\\stage4b_best.pt")
    print(f"  Calibration: extended scale_grid (stable 1.0-2.25, leakage 1.0-2.50)")
    print()
    if not all_ni:
        print(f"  Reason: Stage5-ctrl fails NI criterion — adversarial framework degrades performance.")
    else:
        print(f"  Reason: Stage5-ctrl is non-inferior but provides no additional benefit.")
        print(f"  The 1 extra training epoch does not change any metric meaningfully.")

print()
print(f"  Architecture of final FS-PI-TimeGAN:")
print(f"    Stage 3:  Physics backbone (encoder/decoder/ODE) — frozen")
print(f"    Stage 4C: AR1GuidedResidualGeneratorStable")
print(f"              - Stable features only: Vth, IDSS, RON, gmmax")
print(f"              - Positive rho in (0, 0.97) via sigmoid parameterisation")
print(f"              - Log-time OU: rho_eff(i) = rho_ref^(Delta_log10(1+t_i)/0.35)")
print(f"              - Device-centered residuals for rho target fitting")
print(f"              - Feature x temperature sigma calibration (extended scale grid)")
print(f"    IDLeak/IGLeak: Stage 3 deterministic auxiliary outputs (excluded from generator)")
