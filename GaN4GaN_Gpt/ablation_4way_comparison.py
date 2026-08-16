#!/usr/bin/env python
"""
ablation_4way_comparison.py
============================
4-way ablation: Stage4C  /  Stage4C-ACF  /  Stage5-control(lam_adv=0)  /  Stage5-adv
Plus: time-normalised ACF analysis for the irregular time grid.

Answers:
  Q1. Is Stage 5's CRPSS gain from adversarial learning or just extra training?
      Compare S5-ctrl vs S5-adv (same 1 epoch from Stage4C, only lambda_adv differs)
  Q2. Does ACF loss improve temporal correlation without hurting CRPSS/Cov90?
      Compare Stage4C vs Stage4C-ACF
  Q3. What do the true ACF patterns look like under the irregular time grid?
"""
import csv, os, sys, json, importlib.util
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─── CSV paths ──────────────────────────────────────────────────────────────
PATHS = {
    "Stage4C":     r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage4c\grouped_cv_fold_metrics.csv",
    "Stage4C-ACF": r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage4c_acf\grouped_cv_fold_metrics.csv",
    "S5-ctrl":     r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage5_ctrl\grouped_cv_fold_metrics.csv",
    "S5-adv":      r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage5_stable\grouped_cv_fold_metrics.csv",
}

# ─── Bootstrap CI helper ─────────────────────────────────────────────────────
def bootstrap_ci(deltas, seeds, n_boot=3000, alpha=0.05, rng_seed=0):
    d, s = np.array(deltas), np.array(seeds)
    u    = np.unique(s)
    rng  = np.random.default_rng(rng_seed)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(u, len(u), replace=True)
        means.append(np.mean(np.concatenate([d[s == si] for si in samp])))
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))

# ─── Load fold metrics ────────────────────────────────────────────────────────
def load_model(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        if r["model"] == "stage4b":
            rows[(str(r["seed"]), str(r["fold"]))] = r
    return rows

data = {name: load_model(p) for name, p in PATHS.items()}
keys = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
assert len(keys) == 15, f"Expected 15 matched pairs, got {len(keys)}"

seeds_arr = [int(k[0]) for k in keys]

def g(r, c):
    try: return float(r[c])
    except: return np.nan

metrics = [
    ("CRPSS",    "crpss_overall",          +1),
    ("Cov90-pw", "coverage_90_pointwise",   +1),
    ("MACE",     "mace",                   -1),
    ("W1-incr",  "w1_increments_overall",  -1),
]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: MODEL × METRIC SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 100)
print("SECTION A: 4-WAY MODEL SUMMARY  (mean +/- SD over 15 fold-seed pairs)")
print("=" * 100)
print(f"  {'Model':<14}", end="")
for m, col, _ in metrics:
    print(f"  {m:>12}", end="")
print()
print(f"  {'-'*90}")

for name, d in data.items():
    print(f"  {name:<14}", end="")
    for m, col, _ in metrics:
        vals = [g(d[k], col) for k in keys if not np.isnan(g(d[k], col))]
        print(f"  {np.mean(vals):>+6.4f}+-{np.std(vals):.4f}", end="")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: Q1 — Is Stage 5 gain adversarial or just extra training?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("SECTION B: Q1  —  Is Stage 5 improvement from adversarial training or just extra epochs?")
print("  Comparison: S5-adv  vs  S5-ctrl  (both 1 extra epoch from Stage4C; only lambda_adv differs)")
print(f"{'=' * 100}")
print(f"  {'Metric':<12}  {'mean(adv-ctrl)':>16}  {'95% CI':>20}  {'Verdict'}")
print(f"  {'-' * 80}")

for m, col, sign in metrics:
    d_adv  = [g(data["S5-adv"][k],  col) for k in keys]
    d_ctrl = [g(data["S5-ctrl"][k], col) for k in keys]
    delta  = [a - b for a, b in zip(d_adv, d_ctrl)]
    mean_d = np.mean(delta)
    lo, hi = bootstrap_ci(delta, seeds_arr)

    # Adversarial is beneficial if mean_d is in the favourable direction
    fav = (mean_d * sign) > 0
    ci_excludes_zero = lo > 0 or hi < 0
    if fav and ci_excludes_zero:
        verdict = "ADV wins (CI excludes 0)"
    elif abs(mean_d) < 1e-4:
        verdict = "identical (extra-training artefact)"
    else:
        verdict = "marginal / noise"
    print(f"  {m:<12}  {mean_d:>+16.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: Q2 — Does ACF loss improve temporal correlation?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("SECTION C: Q2  —  Does ACF matching loss improve metrics vs Stage4C baseline?")
print("  Comparison: Stage4C-ACF  vs  Stage4C")
print(f"{'=' * 100}")
print(f"  {'Metric':<12}  {'mean(ACF-base)':>16}  {'95% CI':>20}  {'Verdict'}")
print(f"  {'-' * 80}")

for m, col, sign in metrics:
    d_acf  = [g(data["Stage4C-ACF"][k], col) for k in keys]
    d_base = [g(data["Stage4C"][k],     col) for k in keys]
    delta  = [a - b for a, b in zip(d_acf, d_base)]
    mean_d = np.mean(delta)
    lo, hi = bootstrap_ci(delta, seeds_arr)
    fav = (mean_d * sign) > 0
    ci_fav = (lo * sign > 0 and hi * sign > 0)
    if ci_fav:
        verdict = "ACF-loss IMPROVES (CI on correct side)"
    elif (lo * sign > -0.001):
        verdict = "non-inferior (CI low > -0.001)"
    else:
        verdict = "ACF-loss HURTS"
    print(f"  {m:<12}  {mean_d:>+16.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: Time-normalised ACF (irregular grid analysis)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("SECTION D: TIME-NORMALISED ACF  (irregular time grid: 0,1,2,5,10,...,2000 h)")
print("  Standard index-based ACF mixes very different Dt windows.")
print("  Here we bin pairs by actual Dt and fit exp(-Dt/tau) to compare generators.")
print(f"{'=' * 100}")

import torch
import importlib.util

def _lm(alias, fname):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

mods = {
    "prep":  _lm("_ab_prep",  "01_data_preprocessing.py"),
    "ode":   _lm("_ab_ode",   "02_physics_latent.py"),
    "enc":   _lm("_ab_enc",   "03_model_encoder.py"),
    "dec":   _lm("_ab_dec",   "04_model_decoder.py"),
    "gen":   _lm("_ab_gen",   "05_model_generator.py"),
    "disc":  _lm("_ab_disc",  "06_model_discriminator.py"),
    "train": _lm("_ab_trn",   "08_training.py"),
    "eval9": _lm("_ab_ev9",   "09_evaluation.py"),
    "s4b":   _lm("_ab_s4b",   "14_stage4b_ar1_guided_generator.py"),
}

CKPTS = {
    "Stage4C":     r"D:\2026\article\GaN4GaN\output\pi_timegan_cfgC_stage4c\checkpoints\stage4b_best.pt",
    "Stage4C-ACF": r"D:\2026\article\GaN4GaN\output\pi_timegan_cfgC_stage4c_acf\checkpoints\stage4b_best.pt",
    "S5-ctrl":     r"D:\2026\article\GaN4GaN\output\pi_timegan_stage5_control\stage5_best.pt",
    "S5-adv":      r"D:\2026\article\GaN4GaN\output\pi_timegan_stage5_stable\stage5_best.pt",
}
S3 = r"D:\2026\article\GaN4GaN\output\pi_timegan\checkpoints\stage3_best.pt"

import config as cfg
STABLE_IDX = mods["s4b"].STABLE_FEAT_INDICES

class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = mods["enc"].PhysicsEncoder()
        self.decoder   = mods["dec"].SparsePhysicsDecoder()
        self.ode       = mods["ode"].PhysicsODE()
        self.alpha_net = mods["ode"].DeviceAlphaNet()
        self.generator = mods["gen"].PITimeGANGenerator()
        self.disc      = mods["disc"].PITimeGANDiscriminator()

model3 = mods["eval9"].load_model(S3, _Model).to("cpu")
model3.eval()
for p in model3.parameters():
    p.requires_grad_(False)

from torch.utils.data import DataLoader
dataset   = mods["prep"].load_dataset()
test_idx  = dataset["split"]["test"]
test_ds   = mods["train"].DeviceDegradationDataset(dataset, test_idx)
test_dl   = DataLoader(test_ds, batch_size=16, shuffle=False,
                       collate_fn=mods["train"].collate_fn)
test_cache = mods["s4b"]._cache_trajectories(model3, test_dl, "cpu",
                                              mods["train"]._forward,
                                              cfg.STAGE3_PREFIX_LEN)

# ── Helper: generate samples from a checkpoint ──────────────────────────────
def load_gen(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out_sz = sd.get("net.5.bias", sd.get("net.6.bias", None))
    if out_sz is not None and out_sz.numel() == 2 * mods["s4b"].N_STABLE_FEATURES:
        gen = mods["s4b"].AR1GuidedResidualGeneratorStable().to("cpu")
    else:
        gen = mods["s4b"].AR1GuidedResidualGenerator().to("cpu")
    gen.load_state_dict(sd, strict=False)
    gen.eval()
    return gen

def gen_residuals(gen, cache, n_samp=100):
    """Return true_resid (N,T,F) and gen_resid (N,T,F) mean across samples."""
    true_res_list, gen_res_list, times_list, masks_list = [], [], [], []
    sfx = torch.tensor(STABLE_IDX)
    plen = cfg.STAGE3_PREFIX_LEN

    with torch.no_grad():
        for rec in cache:
            z_pfx = rec["z_pfx"]; x_hat = rec["x_hat"]; x_true = rec["x_true"]
            T_K = rec["T_K"]; log_t = rec["log_t"]; x0 = rec["x0"]
            mask = rec["mask"]
            T_future = rec["T_len"] - plen
            if T_future <= 0:
                continue
            deltas = gen.sample_n(z_pfx, T_K, x0, log_t, n_samples=n_samp, T_future=T_future)
            # Mean generated residual: (B, T_future, F_s)
            gen_mean = deltas.mean(dim=0).numpy()

            # True residuals
            true_r = (x_true[:, plen:, sfx] - x_hat[:, plen:, sfx]).numpy()
            # Get times for future steps only
            times_np = rec.get("times_h", None)
            if times_np is not None:
                times_np = times_np[:, plen:].numpy()
            else:
                times_np = np.arange(T_future)[None, :] * np.ones((x_hat.shape[0], 1))

            true_res_list.append(true_r)
            gen_res_list.append(gen_mean)
            times_list.append(times_np)
            masks_list.append(mask[:, plen:].numpy())

    return (np.concatenate(true_res_list, axis=0),
            np.concatenate(gen_res_list,  axis=0),
            np.concatenate(times_list,    axis=0),
            np.concatenate(masks_list,    axis=0))


# ── Compute time-normalised ACF ───────────────────────────────────────────────
def time_normalised_acf(residuals, times_h, valid_mask, bins_dt=None):
    """
    For each consecutive valid pair (n, t), (n, t+1):
      compute actual dt = times_h[n,t+1] - times_h[n,t]
      and correlation value r[n,t] * r[n,t+1] / (sigma^2)
    Then average within each dt bin and fit exp(-dt/tau).

    residuals  : (N, T, F)
    times_h    : (N, T)
    valid_mask : (N, T) bool

    Returns per-feature dict:
      bin_mean_corr: {bin_label: float}
      tau_fit: float (from exp(-dt/tau) fit)
    """
    from scipy.optimize import curve_fit

    if bins_dt is None:
        bins_dt = [(0.5, 3), (3, 15), (15, 100), (100, 1000), (1000, np.inf)]
        bin_labels = ["1-2h", "3-14h", "15-100h", "100-1000h", ">1000h"]
    else:
        bin_labels = [f"{lo}-{hi}h" for lo, hi in bins_dt]

    N, T, F = residuals.shape
    results = {}

    for fi in range(F):
        fname = [cfg.FEATURES[i] for i in STABLE_IDX][fi]
        r = np.nan_to_num(residuals[:, :, fi], nan=0.0)   # (N, T)

        # Per-device variance for normalisation
        all_pairs_dt, all_pairs_prod = [], []
        bin_data = {i: [] for i in range(len(bins_dt))}

        for n in range(N):
            valid_t = np.where(valid_mask[n])[0]
            if len(valid_t) < 2:
                continue
            var_n = np.var(r[n, valid_t]) + 1e-8

            for i in range(len(valid_t) - 1):
                t1, t2 = valid_t[i], valid_t[i+1]
                if np.isnan(residuals[n, t1, fi]) or np.isnan(residuals[n, t2, fi]):
                    continue
                dt = times_h[n, t2] - times_h[n, t1]
                if dt <= 0:
                    continue
                corr_proxy = r[n, t1] * r[n, t2] / var_n   # normalised cross-product

                all_pairs_dt.append(dt)
                all_pairs_prod.append(corr_proxy)

                for bi, (lo, hi) in enumerate(bins_dt):
                    if lo < dt <= hi:
                        bin_data[bi].append(corr_proxy)
                        break

        bin_corr = {}
        for bi, bl in enumerate(bin_labels):
            v = bin_data[bi]
            bin_corr[bl] = float(np.mean(v)) if len(v) >= 3 else np.nan

        # Fit exp(-dt/tau)
        dt_arr   = np.array(all_pairs_dt)
        corr_arr = np.array(all_pairs_prod)
        tau = np.nan
        if len(dt_arr) >= 10:
            # Use log-linear regression on positive correlations only
            pos = corr_arr > 0
            if pos.sum() >= 5:
                log_corr = np.log(corr_arr[pos].clip(1e-8))
                dt_pos   = dt_arr[pos]
                try:
                    popt, _ = curve_fit(lambda dt, tau: -dt / tau, dt_pos, log_corr, p0=[200.0])
                    tau = float(abs(popt[0]))
                except Exception:
                    pass

        results[fname] = {"bin_corr": bin_corr, "tau_h": tau,
                          "n_pairs": len(dt_arr)}

    return results


# ── Run for true data and all four generators ─────────────────────────────────
print("\n  Loading generators and generating residuals on test set (S=100 samples)...")

models = {name: load_gen(ckpt) for name, ckpt in CKPTS.items()}

true_res, _, times, mask_fut = gen_residuals(models["Stage4C"], test_cache, n_samp=1)
# true_res is the same regardless of generator; use from Stage4C run

print(f"  Test set: {true_res.shape[0]} devices, {true_res.shape[1]} future steps, {true_res.shape[2]} stable features")

all_res = {}
for name, gen in models.items():
    print(f"  Generating [{name}]...", end=" ", flush=True)
    _, gr, _, _ = gen_residuals(gen, test_cache, n_samp=100)
    all_res[name] = gr
    print("done")

# Index-based ACF comparison
print(f"\n  Index-based ACF(lag=1) [generated vs true]")
print(f"  {'Feature':<8}", end="")
for name in CKPTS:
    print(f"  {name:>14}", end="")
print(f"  {'True':>8}")
print(f"  {'-' * 80}")

from scipy.stats import pearsonr

for fi, fname in enumerate([cfg.FEATURES[i] for i in STABLE_IDX]):
    print(f"  {fname:<8}", end="")
    for name in CKPTS:
        gr = all_res[name]
        pairs_x, pairs_y = [], []
        for n in range(gr.shape[0]):
            vt = np.where(mask_fut[n])[0]
            if len(vt) < 2:
                continue
            for i in range(len(vt) - 1):
                r1, r2 = gr[n, vt[i], fi], gr[n, vt[i+1], fi]
                if not (np.isnan(r1) or np.isnan(r2)):
                    pairs_x.append(r1); pairs_y.append(r2)
        if len(pairs_x) >= 5 and np.std(pairs_x) > 1e-8:
            acf = float(pearsonr(pairs_x, pairs_y)[0])
        else:
            acf = np.nan
        print(f"  {acf:>14.4f}", end="")

    # True ACF
    pairs_x, pairs_y = [], []
    for n in range(true_res.shape[0]):
        vt = np.where(mask_fut[n])[0]
        if len(vt) < 2:
            continue
        for i in range(len(vt) - 1):
            v = true_res[n, :, fi]
            r1 = np.nan_to_num(v[vt[i]], nan=float('nan'))
            r2 = np.nan_to_num(v[vt[i+1]], nan=float('nan'))
            if not (np.isnan(r1) or np.isnan(r2)):
                pairs_x.append(r1); pairs_y.append(r2)
    if len(pairs_x) >= 5 and np.std(pairs_x) > 1e-8:
        acf_true = float(pearsonr(pairs_x, pairs_y)[0])
    else:
        acf_true = np.nan
    print(f"  {acf_true:>8.4f}")

# Time-normalised ACF (exponential decay tau)
print(f"\n  Time-normalised ACF — correlation decay constant tau [hours]")
print(f"  (exp(-Dt/tau) fit to true data and generated residuals)")
bins_dt = [(0.5, 3), (3, 15), (15, 100), (100, 1000), (1000, 5000)]
print(f"\n  True data:")
true_acf_res = time_normalised_acf(true_res, times, mask_fut, bins_dt=bins_dt)
for fname, rr in true_acf_res.items():
    tau_str = f"tau={rr['tau_h']:.0f}h" if not np.isnan(rr['tau_h']) else "tau=N/A"
    bins_str = "  ".join(f"{bl}:{v:+.3f}" for bl, v in rr['bin_corr'].items())
    print(f"    {fname}: {tau_str}   [{bins_str}]  (n_pairs={rr['n_pairs']})")

print(f"\n  {'Model':<14}  ", end="")
for fname in [cfg.FEATURES[i] for i in STABLE_IDX]:
    print(f"{fname:>12}", end="")
print()
print(f"  {'-' * 70}")
for name, gen in models.items():
    print(f"  {name:<14}  ", end="")
    gen_acf = time_normalised_acf(all_res[name], times, mask_fut, bins_dt=bins_dt)
    for fname, rr in gen_acf.items():
        tau_str = f"{rr['tau_h']:.0f}" if not np.isnan(rr['tau_h']) else "N/A"
        print(f"{tau_str:>12}", end="")
    print()

print(f"\n  True tau (h): ", end="")
for fname, rr in true_acf_res.items():
    tau_str = f"{rr['tau_h']:.0f}" if not np.isnan(rr['tau_h']) else "N/A"
    print(f"{tau_str:>12}", end="")
print()

print(f"\n  NOTE: Index-based ACF(lag=1) mixes intervals spanning")
print(f"  1h, 3h, 8h, 10h, 30h, ... and 1000h - those are NOT comparable!")
print(f"  Time-normalised tau shows the TRUE decay timescale.")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("SUMMARY OF FINDINGS")
print(f"{'=' * 100}")
print("""
  Q1. Is Stage 5 improvement from adversarial or extra training?
      => Check Section B. If S5-adv vs S5-ctrl differences are tiny (~0),
         Stage 5's effect is purely from 1 extra training epoch (not adversarial).

  Q2. Does ACF loss help?
      => Check Section C. If Stage4C-ACF has better CRPSS or Cov90 without
         hurting other metrics, the ACF loss is worth keeping.
         If W1-incr improves, temporal structure is genuinely better.

  Q3. Time grid and ACF:
      => True ACF(lag=1) ≈ 0.5 is dominated by 1-3h interval pairs (tiny Dt).
         Generated residuals show near-zero index-based ACF because the AR(1)
         generator treats all time steps as equally spaced.
         Time-normalised tau shows the actual temporal structure.
""")
