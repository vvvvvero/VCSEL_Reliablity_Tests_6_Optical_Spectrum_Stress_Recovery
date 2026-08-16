"""Diagnostic script for current model state — run with gan312 interpreter."""
import pickle, sys, torch
import numpy as np

# ── 1. Stage3 evaluation results ──────────────────────────────────────────
r = pickle.load(open(
    'D:/2026/article/GaN4GaN/output/pi_timegan/results/evaluation_results_stage3.pkl', 'rb'))

print("=== latent saturation ===")
ls = r.get('latent_saturation', {})
for tc, d in ls.items():
    zm = d['zM']
    zc = d['zC']
    print(f"  {tc}C  zM: mean={zm['mean']:.3f}  frac>=0.95={zm['fraction_ge_095']:.3f}  pre100h={zm.get('fraction_ge_095_pre100h', float('nan')):.3f}")
    print(f"        zC: mean={zc['mean']:.3f}  frac>=0.95={zc['fraction_ge_095']:.3f}")

print("\n=== initial latent z0 means ===")
ils = r.get('initial_latent_stats', {})
for tc, d in ils.items():
    means = {k: round(d[k]['mean'], 4) for k in d}
    print(f"  {tc}C:", means)

print("\n=== RMSE by temperature ===")
rbt = r.get('rmse_by_temperature', {})
for tc, d in rbt.items():
    ov = d.get('overall', float('nan'))
    il = d.get('IDLeak', float('nan'))
    ig = d.get('IGLeak', float('nan'))
    vt = d.get('Vth', float('nan'))
    print(f"  {tc}: overall={ov:.4f}  Vth={vt:.4f}  IDLeak={il:.4f}  IGLeak={ig:.4f}")

print("\n=== stochastic residual (current pkl) ===")
sr = r.get('stochastic_residual', {})
if sr:
    print("  CRPS:", sr.get('crps_overall'))
    print("  CRPSS:", sr.get('crpss_overall'))
    print("  Cov90:", round(sr.get('coverage_90_overall', float('nan')), 4))
    print("  MACE:", round(sr.get('reliability_mace', float('nan')), 4))
    dfs = sr.get('decreasing_fraction_samples', {})
    dft = sr.get('decreasing_fraction_true', {})
    print("  dec-frac IDLeak: pred={:.3f}  true={:.3f}".format(
        dfs.get('IDLeak', float('nan')), dft.get('IDLeak', float('nan'))))
    print("  dec-frac IGLeak: pred={:.3f}  true={:.3f}".format(
        dfs.get('IGLeak', float('nan')), dft.get('IGLeak', float('nan'))))
else:
    print("  (no stochastic_residual key)")

# ── 2. ODE learned parameters ─────────────────────────────────────────────
print("\n=== Stage3 ODE learned parameters ===")
ckpt = torch.load(
    'D:/2026/article/GaN4GaN/output/pi_timegan/checkpoints/stage3_best.pt',
    map_location='cpu')
for k, v in ckpt['model_state'].items():
    if any(sub in k for sub in ['Ea_', 'kGc', 'kGe', 'kBc', 'kBe', 'kM', 'kL', 'kC', 'aLG', 'aLB']):
        print(f"  {k}: {float(v):.6f}")

# ── 3. Grouped CV stability ────────────────────────────────────────────────
print("\n=== Grouped CV stability (25 runs each) ===")
stab = pickle.load(open(
    'D:/2026/article/GaN4GaN/output/pi_timegan/results/grouped_cv_stability/grouped_cv_stability.pkl',
    'rb'))
print("  AR1 win vs Gaussian:", round(stab['ar1_win_rate_vs_gaussian'], 3),
      f"  ({stab['n_comparisons_vs_gaussian']} comparisons)")
print("  AR1 win vs Stage4:",   round(stab['ar1_win_rate_vs_stage4'], 3),
      f"  ({stab['n_comparisons_vs_stage4']} comparisons)")
for row in stab['model_summary']:
    print(f"  {row['model']:10s}  CRPSS={row['crpss_overall_mean']:.4f}+/-{row['crpss_overall_std']:.4f}"
          f"  Cov90={row['coverage_90_overall_mean']:.3f}+/-{row['coverage_90_overall_std']:.3f}"
          f"  MACE={row['mace_mean']:.4f}  n={row['n_runs']}")

print("\nDone.")
