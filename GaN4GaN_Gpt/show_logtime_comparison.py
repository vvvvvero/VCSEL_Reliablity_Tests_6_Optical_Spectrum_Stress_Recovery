import csv, numpy as np
models = {
    "Stage4C":     r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage4c\grouped_cv_model_summary.csv",
    "Stage4C-ACF": r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage4c_acf\grouped_cv_model_summary.csv",
    "Stage4C-LT":  r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage4c_logtime\grouped_cv_model_summary.csv",
    "S5-ctrl":     r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage5_ctrl\grouped_cv_model_summary.csv",
    "S5-adv":      r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage5_stable\grouped_cv_model_summary.csv",
}
print(f"{'Model':<14}  {'CRPSS':>8}  {'Cov90-pw':>9}  {'W1-incr':>8}")
print("-" * 50)
for name, p in models.items():
    rows = {r["model"]: r for r in csv.DictReader(open(p))}
    r = rows.get("stage4b", {})
    cr = float(r.get("crpss_overall_mean", "nan"))
    cp = float(r.get("coverage_90_pointwise_mean", "nan"))
    w1 = float(r.get("w1_increments_overall_mean", "nan"))
    print(f"{name:<14}  {cr:>+8.4f}  {cp:>9.4f}  {w1:>8.4f}")
