#!/usr/bin/env python
"""Paired Stage 4C vs Stage 5 (stable) grouped-CV comparison."""
import csv, numpy as np

def load_model_summary(path, model='stage4b'):
    rows = {r['model']: r for r in csv.DictReader(open(path))}
    return rows.get(model, {})

def load_fold_crpss(path, model='stage4b'):
    return [float(r['crpss_overall']) for r in csv.DictReader(open(path)) if r['model'] == model]

s4c_summary = load_model_summary(
    r'D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage4c\grouped_cv_model_summary.csv')
s5s_summary = load_model_summary(
    r'D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage5_stable\grouped_cv_model_summary.csv')

s4c_folds = load_fold_crpss(
    r'D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage4c\grouped_cv_fold_metrics.csv')
s5s_folds = load_fold_crpss(
    r'D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage5_stable\grouped_cv_fold_metrics.csv')

def fget(r, k): return float(r.get(k, 'nan'))

print("=" * 95)
print("PAIRED COMPARISON: Stage 4C (stable)  vs  Stage 5 (stable+adv)")
print("Metrics evaluated on STABLE features only: Vth, IDSS, RON, gmmax")
print("IDLeak / IGLeak: Stage 3 deterministic auxiliary output (excluded from gen/disc)")
print("=" * 95)
print(f"{'Metric':<30} {'Stage4C':>10} {'Stage5(adv)':>12} {'Delta':>10} {'Winner':>10}")
print("-" * 95)

metrics = [
    ("CRPSS",           "crpss_overall_mean"),
    ("Cov90 overall",   "coverage_90_overall_mean"),
    ("Cov90 pointwise", "coverage_90_pointwise_mean"),
    ("Cov90 dev-avg",   "coverage_90_device_avg_mean"),
    ("Cov90 simult.",   "coverage_90_simultaneous_mean"),
    ("MACE",            "mace_mean"),
    ("Width90",         "width_90_overall_mean"),
]

decisions = {}
for label, key in metrics:
    v4 = fget(s4c_summary, key)
    v5 = fget(s5s_summary,  key)
    delta = v5 - v4
    higher_better = "MACE" not in label and "Width" not in label
    win = "Stage5" if (delta > 0 and higher_better) or (delta < 0 and not higher_better) else "Stage4C"
    decisions[label] = win
    print(f"  {label:<28} {v4:>10.4f} {v5:>12.4f} {delta:>+10.4f} {win:>10}")

wins_s5 = sum(1 for a, b in zip(s5s_folds, s4c_folds) if a > b)
print(f"\n  CRPSS win-rate (Stage5 > Stage4C): {wins_s5}/{len(s4c_folds)} folds")

improved_crpss     = decisions.get("CRPSS", "Stage4C") == "Stage5"
improved_pointwise = decisions.get("Cov90 pointwise", "Stage4C") == "Stage5"

print("\n" + "=" * 95)
if improved_crpss and improved_pointwise:
    print("VERDICT: Stage 5 improves BOTH CRPSS and Pointwise Cov90 on stable features  ->  ADOPT Stage 5")
elif improved_crpss:
    print("VERDICT: Stage 5 improves CRPSS only  ->  RETAIN Stage 4C")
elif improved_pointwise:
    print("VERDICT: Stage 5 improves Cov90 only  ->  RETAIN Stage 4C")
else:
    print("VERDICT: Stage 5 does not improve either metric  ->  RETAIN Stage 4C as final model")
print("=" * 95)
