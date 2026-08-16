#!/usr/bin/env python
import csv

BASE = r'D:\2026\article\GaN4GaN\output\pi_timegan_groupedcv_cfgC_stage4b_cov90fixed'

rows = list(csv.DictReader(open(BASE + r'\grouped_cv_model_summary.csv')))
print("=== Model Summary (corrected: future-only + NaN-excluded) ===")
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
print("\n=== Stage4B per-seed (corrected) ===")
print(f"{'Seed':<6} {'Overall':>9} {'Pointwise':>10} {'DeviceAvg':>10} {'Simult':>8}")
for r in rows2:
    o  = float(r.get('coverage_90_overall_mean','nan'))
    pw = float(r.get('coverage_90_pointwise_mean','nan'))
    da = float(r.get('coverage_90_device_avg_mean','nan'))
    si = float(r.get('coverage_90_simultaneous_mean','nan'))
    print(f"{r['seed']:<6} {o:>9.4f} {pw:>10.4f} {da:>10.4f} {si:>8.4f}")
