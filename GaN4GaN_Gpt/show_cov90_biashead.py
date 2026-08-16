#!/usr/bin/env python
import csv

BASE = r'D:\2026\article\GaN4GaN\output\pi_timegan_groupedcv_stage4b_biashead_feattemp'

rows = list(csv.DictReader(open(BASE + r'\grouped_cv_model_summary.csv')))
print("=== NEW MODEL (bias head + feat-temp calibration) ===")
print(f"{'Model':<12} {'Overall':>9} {'Pointwise':>10} {'DeviceAvg':>10} {'Simult':>8} {'CRPSS':>8} {'Width90':>8}")
for r in rows:
    if r['model'] in ['ar1', 'gaussian', 'stage4b']:
        o  = float(r.get('coverage_90_overall_mean','nan'))
        pw = float(r.get('coverage_90_pointwise_mean','nan'))
        da = float(r.get('coverage_90_device_avg_mean','nan'))
        si = float(r.get('coverage_90_simultaneous_mean','nan'))
        cr = float(r.get('crpss_overall_mean','nan'))
        w  = float(r.get('width_90_overall_mean','nan'))
        print(f"{r['model']:<12} {o:>9.4f} {pw:>10.4f} {da:>10.4f} {si:>8.4f} {cr:>8.4f} {w:>8.4f}")

rows2 = [r for r in csv.DictReader(open(BASE + r'\grouped_cv_seed_summary.csv')) if r['model'] == 'stage4b']
print("\n=== Stage4B per-seed (new) ===")
print(f"{'Seed':<6} {'Overall':>9} {'Pointwise':>10} {'DeviceAvg':>10} {'Simult':>8}")
for r in rows2:
    o  = float(r.get('coverage_90_overall_mean','nan'))
    pw = float(r.get('coverage_90_pointwise_mean','nan'))
    da = float(r.get('coverage_90_device_avg_mean','nan'))
    si = float(r.get('coverage_90_simultaneous_mean','nan'))
    print(f"{r['seed']:<6} {o:>9.4f} {pw:>10.4f} {da:>10.4f} {si:>8.4f}")

# Sample sigma maps
fold_rows = list(csv.DictReader(open(BASE + r'\grouped_cv_fold_metrics.csv')))
s4b_rows = [r for r in fold_rows if r['model'] == 'stage4b']
print("\n=== Sample sigma_scale_by_feat_temp maps (first 3 folds, seed 11) ===")
for r in s4b_rows[:3]:
    print(f"  fold={r['fold']} mode={r['stage4_sigma_mode']} map={r['stage4_sigma_map'][:120]}")
