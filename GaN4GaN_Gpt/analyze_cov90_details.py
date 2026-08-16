#!/usr/bin/env python
import csv
import numpy as np

def detailed_analysis():
    p = r'D:\2026\article\GaN4GaN\output\pi_timegan_groupedcv_cfgC_stage4b_cov90variants\grouped_cv_fold_metrics.csv'
    rows = [r for r in csv.DictReader(open(p)) if r['model'] == 'stage4b']
    
    print("=" * 100)
    print("STAGE4B FOLD-LEVEL Cov90 详细分析")
    print("=" * 100)
    print(f"{'Fold':<6} {'Seed':<6} {'Overall':<12} {'Pointwise':<12} {'Device-Avg':<12} {'Simultaneous':<12} {'CRPSS':<10}")
    print("-" * 100)
    
    overall_vals = []
    pointwise_vals = []
    device_avg_vals = []
    simul_vals = []
    crpss_vals = []
    
    for r in rows:
        fold = r['fold']
        seed = r['seed']
        overall = float(r.get('coverage_90_overall', 'nan'))
        pointwise = float(r.get('coverage_90_pointwise', 'nan'))
        device_avg = float(r.get('coverage_90_device_avg', 'nan'))
        simul = float(r.get('coverage_90_simultaneous', 'nan'))
        crpss = float(r.get('crpss_overall', 'nan'))
        
        print(f"{fold:<6} {seed:<6} {overall:<12.4f} {pointwise:<12.4f} {device_avg:<12.4f} {simul:<12.4f} {crpss:<10.4f}")
        
        if not np.isnan(overall):
            overall_vals.append(overall)
        if not np.isnan(pointwise):
            pointwise_vals.append(pointwise)
        if not np.isnan(device_avg):
            device_avg_vals.append(device_avg)
        if not np.isnan(simul):
            simul_vals.append(simul)
        if not np.isnan(crpss):
            crpss_vals.append(crpss)
    
    print("-" * 100)
    print(f"{'Mean':<6} {'':<6} {np.mean(overall_vals):<12.4f} {np.mean(pointwise_vals):<12.4f} {np.mean(device_avg_vals):<12.4f} {np.mean(simul_vals):<12.4f} {np.mean(crpss_vals):<10.4f}")
    print(f"{'Std':<6} {'':<6} {np.std(overall_vals):<12.4f} {np.std(pointwise_vals):<12.4f} {np.std(device_avg_vals):<12.4f} {np.std(simul_vals):<12.4f} {np.std(crpss_vals):<10.4f}")
    
    print("\n" + "=" * 100)
    print("三种定义的关系分析:")
    print("=" * 100)
    print(f"Coverage_90_overall (原始,按设备平均): {np.mean(overall_vals):.4f}")
    print(f"Pointwise (所有单点): {np.mean(pointwise_vals):.4f}")
    print(f"Device-Avg (先按设备再平均): {np.mean(device_avg_vals):.4f}")
    print(f"Simultaneous (完整轨迹): {np.mean(simul_vals):.4f}")
    print()
    print("🔍 为什么这三个值都远低于整体的70%?")
    print("   - coverage_90_overall是按设备算的覆盖率平均")
    print("   - pointwise/device_avg/simultaneous是按点和特征来算的")
    print("   - 需要检查calculate_metrics中如何定义coverage_90_overall")

if __name__ == '__main__':
    detailed_analysis()
