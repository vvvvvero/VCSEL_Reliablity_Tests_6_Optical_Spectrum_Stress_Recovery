#!/usr/bin/env python
import csv
import sys

def extract_model_summary():
    p = r'D:\2026\article\GaN4GaN\output\pi_timegan_groupedcv_cfgC_stage4b_cov90variants\grouped_cv_model_summary.csv'
    rows = list(csv.DictReader(open(p)))
    
    print("=" * 80)
    print("MODEL SUMMARY: Cov90 三种定义对比")
    print("=" * 80)
    print(f"{'Model':<12} {'Pointwise':<12} {'Device-Avg':<12} {'Simultaneous':<12} {'CRPSS':<10}")
    print("-" * 80)
    
    for r in rows:
        if r['model'] in ['ar1', 'gaussian', 'stage4b']:
            pointwise = r.get('coverage_90_pointwise_mean', 'N/A')
            device_avg = r.get('coverage_90_device_avg_mean', 'N/A')
            simul = r.get('coverage_90_simultaneous_mean', 'N/A')
            crpss = r.get('crpss_overall_mean', 'N/A')
            print(f"{r['model']:<12} {pointwise:<12} {device_avg:<12} {simul:<12} {crpss:<10}")

def extract_seed_summary():
    p = r'D:\2026\article\GaN4GaN\output\pi_timegan_groupedcv_cfgC_stage4b_cov90variants\grouped_cv_seed_summary.csv'
    rows = [r for r in csv.DictReader(open(p)) if r['model'] == 'stage4b']
    
    print("\n" + "=" * 80)
    print("SEED SUMMARY: Stage4B 三种 Cov90 跨 seed 稳定性")
    print("=" * 80)
    print(f"{'Seed':<6} {'Pointwise':<12} {'Device-Avg':<12} {'Simultaneous':<12}")
    print("-" * 80)
    
    for r in rows:
        seed = r['seed']
        pointwise = r.get('coverage_90_pointwise_mean', 'N/A')
        device_avg = r.get('coverage_90_device_avg_mean', 'N/A')
        simul = r.get('coverage_90_simultaneous_mean', 'N/A')
        print(f"{seed:<6} {pointwise:<12} {device_avg:<12} {simul:<12}")

if __name__ == '__main__':
    extract_model_summary()
    extract_seed_summary()
    
    print("\n" + "=" * 80)
    print("解释:")
    print("=" * 80)
    print("1. Pointwise: 所有(设备,时间,特征)点中有多少%落在PI内")
    print("2. Device-Avg: 先算每个设备的覆盖率,再在设备间平均")
    print("3. Simultaneous: 整条设备轨迹的所有时间点都在PI内的设备比例")
    print("=" * 80)
