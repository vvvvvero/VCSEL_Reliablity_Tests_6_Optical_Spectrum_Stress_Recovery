#!/usr/bin/env python
"""
Cov90三种定义的对比与解释
===========================

发现关键差异：
- coverage_90_overall (0.7065) = 按特征逐个计算覆盖率，再按特征平均
  * 对于每个特征，统计该特征的所有有效点中有多少%在90% PI内
  * 然后对所有特征平均
  
- coverage_90_pointwise (0.3074) = 联合所有特征，计算所有点的覆盖率
  * 对所有(device, time, feature)三元组，计算有多少%落在各自特征的PI内
  * 这反映的是"任意给定一个随机点，它落在PI内的概率"
  
- coverage_90_device_avg (0.2927) = 设备级别覆盖率平均
  * 对于每个设备，计算其所有时间点和特征的覆盖率
  * 然后对所有设备平均
  
- coverage_90_simultaneous (0.0000) = 完整轨迹覆盖
  * 只有当设备的所有时间步的所有特征都同时在PI内时，才算覆盖
  * 这是最严格的定义，结果为0说明没有完全覆盖的轨迹
"""

import csv
import numpy as np

def explain_metrics():
    p = r'D:\2026\article\GaN4GaN\output\pi_timegan_groupedcv_cfgC_stage4b_cov90variants\grouped_cv_model_summary.csv'
    rows = list(csv.DictReader(open(p)))
    
    s4b = rows[2]  # stage4b row
    
    print(__doc__)
    print("\n" + "=" * 80)
    print("Stage4B 实测数值:")
    print("=" * 80)
    print(f"coverage_90_overall (按特征逐算):    {float(s4b['coverage_90_overall_mean']):.4f}  ± {float(s4b['coverage_90_overall_std']):.4f}")
    print(f"coverage_90_pointwise (全点联合):    {float(s4b['coverage_90_pointwise_mean']):.4f}  ± {float(s4b['coverage_90_pointwise_std']):.4f}")
    print(f"coverage_90_device_avg (设备平均):   {float(s4b['coverage_90_device_avg_mean']):.4f}  ± {float(s4b['coverage_90_device_avg_std']):.4f}")
    print(f"coverage_90_simultaneous (完整轨迹):  {float(s4b['coverage_90_simultaneous_mean']):.4f}  ± {float(s4b['coverage_90_simultaneous_std']):.4f}")
    
    print("\n" + "=" * 80)
    print("为什么会有这么大的差异?")
    print("=" * 80)
    print("""
1. coverage_90_overall (0.7065) 最高
   ✓ 原因：按特征逐个统计，每个特征独立判断是否在PI内
   ✓ 好处：反映单个特征维度的预测准确度
   ✓ 缺点：忽视了特征间的关联性

2. coverage_90_pointwise (0.3074) 明显低于overall
   ✓ 原因：要求每个点同时满足该特征的PI条件
   ✓ 表示：在所有被观测的(设备,时间,特征)三元组中，
          有30.74%的点同时落在它们各自的预测区间内
   ✓ 更严格：体现了多维观测的真实难度

3. coverage_90_device_avg (0.2927) ≈ pointwise
   ✓ 结果接近意味着特征间覆盖率一致
   ✓ 说明：各个设备的覆盖情况差异不大

4. coverage_90_simultaneous (0.0000) 最严格
   ✓ 原因：要求一整条轨迹的所有时间步都覆盖
   ✓ 含义：在202个设备中，没有一个的完整未来序列
          都落在预测区间内
   ✓ 启示：任何一个特征在任何一个时间步的误差，
          都会导致该设备不满足"完整轨迹覆盖"
""")
    
    print("\n" + "=" * 80)
    print("推荐用途:")
    print("=" * 80)
    print("""
✓ coverage_90_overall:     用于与其他论文对标，反映特征维度准确度
✓ coverage_90_pointwise:   更保守的评估，反映实际应用中多维预测的困难
✓ coverage_90_device_avg:  中间指标，平衡特征维度和设备多样性
✓ coverage_90_simultaneous: 最严格要求，评估长期可靠性
""")

if __name__ == '__main__':
    explain_metrics()
