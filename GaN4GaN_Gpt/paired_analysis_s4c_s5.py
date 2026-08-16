#!/usr/bin/env python
"""
paired_analysis_s4c_s5.py
==========================
Comprehensive statistical analysis: Stage 4C vs Stage 5 (stable, adversarial).

Sections:
  1. Paired-differences statistics (CRPSS, Cov90-ptwise, MACE) — 15 pairs
  2. Fold-cluster bootstrap 95% CI
  3. Non-inferiority assessment
  4. Adversarial signal verification
     a. Generator parameter change ||theta_S5 - theta_S4C||
     b. Training dynamics (disc_acc, div_ratio, crps_degraded from history)
     c. Post-hoc gradient norm ratio on one validation batch
  5. Generated residual distribution comparison
     a. Per-feature increment Wasserstein-1 (from w1_increments in CSV)
     b. Per-feature residual skewness, kurtosis, ACF-lag-1 (from generated samples)
     c. Discriminator distinguishability D(gen_S4C) vs D(gen_S5) vs D(real)
"""
import importlib.util, json, os, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

# ─── Paths ─────────────────────────────────────────────────────────────────
CSV_S4C  = r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage4c\grouped_cv_fold_metrics.csv"
CSV_S5   = r"D:\2026\article\GaN4GaN\output\pi_timegan_pairedcv_stage5_stable\grouped_cv_fold_metrics.csv"
CKPT_S3  = r"D:\2026\article\GaN4GaN\output\pi_timegan\checkpoints\stage3_best.pt"
CKPT_S4C = r"D:\2026\article\GaN4GaN\output\pi_timegan_cfgC_stage4c\checkpoints\stage4b_best.pt"
CKPT_S5  = r"D:\2026\article\GaN4GaN\output\pi_timegan_stage5_stable\stage5_best.pt"
OUT_JSON = r"D:\2026\article\GaN4GaN\output\paired_analysis_s4c_s5.json"

# Non-inferiority margins (user-specified)
NI_CRPSS   = -0.01
NI_COV90   = -0.02
NI_MACE    =  0.01   # upper bound (Stage5 MACE - Stage4C MACE < +0.01)

# ─── Load CSV ──────────────────────────────────────────────────────────────
import csv

def load_fold_csv(path, model_name="stage4b"):
    rows = {}
    for r in csv.DictReader(open(path)):
        if r["model"] == model_name:
            key = (str(r["seed"]), str(r["fold"]))
            rows[key] = r
    return rows

rows_4c = load_fold_csv(CSV_S4C)
rows_s5 = load_fold_csv(CSV_S5)
common  = sorted(set(rows_4c) & set(rows_s5))
assert len(common) == 15, f"Expected 15 paired rows, got {len(common)}"

def col(r, k, fallback=np.nan):
    try: return float(r.get(k, fallback))
    except: return np.nan

metrics = {
    "CRPSS":        ("crpss_overall",           +1),  # +1 = higher is better
    "Cov90_ptwise": ("coverage_90_pointwise",    +1),
    "Cov90_devavg": ("coverage_90_device_avg",   +1),
    "MACE":         ("mace",                     -1),  # -1 = lower is better
    "W1_incr":      ("w1_increments_overall",    -1),
}

paired = {m: [] for m in metrics}
seeds_list = []
folds_list = []
for key in common:
    s, f = key
    seeds_list.append(int(s))
    folds_list.append(int(f))
    for m, (col_name, sign) in metrics.items():
        v4 = col(rows_4c[key], col_name)
        v5 = col(rows_s5[key], col_name)
        paired[m].append(v5 - v4)   # raw difference (S5 - S4C)

# ─── Section 1: Summary statistics ─────────────────────────────────────────
def summarise(deltas, sign):
    """sign: +1 means higher delta is better, -1 means lower delta is better."""
    d = np.array([x for x in deltas if not np.isnan(x)])
    return {
        "mean":    float(np.mean(d)),
        "median":  float(np.median(d)),
        "std":     float(np.std(d, ddof=1)),
        "min":     float(np.min(d)),
        "max":     float(np.max(d)),
        "p5":      float(np.percentile(d, 5)),
        "p95":     float(np.percentile(d, 95)),
        "n_favorable": int(np.sum(np.array(d) * sign > 0)),  # favorable direction
        "n_pairs": len(d),
    }

sep = "=" * 90
print(sep)
print("SECTION 1: PAIRED DIFFERENCES  Stage 5 - Stage 4C  (n=15 fold-seed pairs)")
print(sep)
hdr = f"{'Metric':<14} {'Mean':>8} {'Median':>8} {'SD':>8} {'Min':>8} {'Max':>8} {'P5':>7} {'P95':>7} {'n>0':>5}"
print(hdr)
print("-" * 90)

stats_summary = {}
for m, (_, sign) in metrics.items():
    s = summarise(paired[m], sign)
    stats_summary[m] = s
    fav_lbl = "higher" if sign == +1 else "lower"
    print(f"  {m:<12} {s['mean']:>+8.5f} {s['median']:>+8.5f} {s['std']:>8.5f} "
          f"{s['min']:>+8.5f} {s['max']:>+8.5f} {s['p5']:>+7.5f} {s['p95']:>+7.5f} "
          f"{s['n_favorable']:>3d}/15  ({fav_lbl} is better)")

# ─── Section 2: Fold-cluster bootstrap 95% CI ──────────────────────────────
def fold_cluster_bootstrap_ci(deltas, seeds, n_bootstrap=4000, alpha=0.05, rng_seed=42):
    """Block-bootstrap by seed. Returns (ci_low, ci_high)."""
    rng = np.random.default_rng(rng_seed)
    d   = np.array(deltas)
    seeds_arr = np.array(seeds)
    unique_seeds = np.unique(seeds_arr)

    boot_means = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_seeds, size=len(unique_seeds), replace=True)
        boot_d  = np.concatenate([d[seeds_arr == s] for s in sampled])
        boot_means.append(np.mean(boot_d))

    boot_means = np.array(boot_means)
    return (float(np.percentile(boot_means, 100 * alpha / 2)),
            float(np.percentile(boot_means, 100 * (1 - alpha / 2))))

print(f"\n{'=' * 90}")
print("SECTION 2: FOLD-CLUSTER BOOTSTRAP 95% CI  (4000 resamples, block by seed)")
print(f"{'=' * 90}")
print(f"  {'Metric':<14} {'CI_low':>10} {'CI_high':>10}  Interpretation")
print(f"  {'-' * 80}")

ci_results = {}
for m, (_, sign) in metrics.items():
    d    = paired[m]
    lo, hi = fold_cluster_bootstrap_ci(d, seeds_list)
    ci_results[m] = (lo, hi)
    # Favourable direction note
    if sign == +1:
        interp = "S5 > S4C" if lo > 0 else ("ambiguous" if hi > 0 else "S5 < S4C")
    else:
        interp = "S5 < S4C (favourable)" if hi < 0 else ("ambiguous" if lo < 0 else "S5 > S4C")
    print(f"  {m:<14} {lo:>+10.5f} {hi:>+10.5f}  {interp}")

# ─── Section 3: Non-inferiority assessment ──────────────────────────────────
print(f"\n{'=' * 90}")
print("SECTION 3: NON-INFERIORITY ASSESSMENT")
print(f"  Margins:  dCRPSS > {NI_CRPSS}   dCov90_ptwise > {NI_COV90}   dMACE < +{NI_MACE}")
print(f"{'=' * 90}")

def ni_verdict(ci_low, ci_high, margin, direction):
    """direction: 'above' means we need CI_low > margin; 'below' means CI_high < margin."""
    if direction == "above":
        passed = ci_low > margin
        status = "PASS" if passed else "FAIL"
        detail = f"CI_low={ci_low:+.5f} {'>' if passed else '<='} margin={margin:+.4f}"
    else:
        passed = ci_high < margin
        status = "PASS" if passed else "FAIL"
        detail = f"CI_high={ci_high:+.5f} {'<' if passed else '>='} margin={margin:+.4f}"
    return status, detail

crpss_status,  crpss_detail  = ni_verdict(*ci_results["CRPSS"],       NI_CRPSS, "above")
cov90_status,  cov90_detail  = ni_verdict(*ci_results["Cov90_ptwise"], NI_COV90, "above")
mace_status,   mace_detail   = ni_verdict(*ci_results["MACE"],         NI_MACE,  "below")

ni_all_pass = (crpss_status == "PASS" and cov90_status == "PASS" and mace_status == "PASS")

print(f"  CRPSS:      [{crpss_status}]  {crpss_detail}")
print(f"  Cov90-ptw:  [{cov90_status}]  {cov90_detail}")
print(f"  MACE:       [{mace_status}]  {mace_detail}")
print()
if ni_all_pass:
    print("  VERDICT: Stage 5 satisfies ALL non-inferiority criteria.")
    print("           Stage 5 is non-inferior to Stage 4C while adding adversarial modeling.")
else:
    fails = [m for m, s in [("CRPSS", crpss_status), ("Cov90", cov90_status), ("MACE", mace_status)] if s == "FAIL"]
    print(f"  VERDICT: Non-inferiority FAILED for: {', '.join(fails)}")
    print("           Stage 4C is retained as the final model.")

# ─── Section 4a: Generator parameter change ────────────────────────────────
print(f"\n{'=' * 90}")
print("SECTION 4a: GENERATOR PARAMETER CHANGE  ||theta_S5 - theta_S4C||")
print(f"{'=' * 90}")
import torch

sd_4c = torch.load(CKPT_S4C, map_location="cpu")["state_dict"]
sd_s5 = torch.load(CKPT_S5,  map_location="cpu")["state_dict"]

total_sq_change = 0.0
total_sq_norm   = 0.0
layer_rows = []
for key in sd_4c:
    if key not in sd_s5:
        continue
    d = (sd_s5[key].float() - sd_4c[key].float())
    abs_ch = float(d.norm().item())
    base   = float(sd_4c[key].float().norm().item())
    rel_ch = abs_ch / max(base, 1e-8)
    total_sq_change += abs_ch ** 2
    total_sq_norm   += base   ** 2
    layer_rows.append((key, abs_ch, base, rel_ch))

total_abs = total_sq_change ** 0.5
total_rel = total_abs / max(total_sq_norm ** 0.5, 1e-8)

print(f"  Total ||Δtheta||_2: {total_abs:.6f}  (relative: {total_rel:.4%})")
print(f"\n  {'Layer':<45} {'||Δ||':>10} {'||theta||':>10} {'rel%':>8}")
print(f"  {'-' * 80}")
for key, ab, ba, re in sorted(layer_rows, key=lambda x: -x[1])[:8]:
    print(f"  {key[:44]:<45} {ab:>10.6f} {ba:>10.6f} {re*100:>7.3f}%")

# ─── Section 4b: Stage 5 training dynamics from history ─────────────────────
print(f"\n{'=' * 90}")
print("SECTION 4b: STAGE 5 TRAINING DYNAMICS (from saved history)")
print(f"{'=' * 90}")
ckpt_s5_data = torch.load(CKPT_S5, map_location="cpu")
history = ckpt_s5_data.get("history", [])
if history:
    print(f"  {'Epoch':>5} {'val_CRPS':>10} {'disc_acc':>10} {'div_ratio':>10} {'crps_degr':>10} {'guards':>7}")
    print(f"  {'-' * 65}")
    for h in history:
        ep = h.get("epoch", "?")
        vc = h.get("val_crps",       np.nan)
        da = h.get("disc_acc",       np.nan)
        dr = h.get("div_ratio",      np.nan)
        cd = h.get("crps_degraded",  np.nan)
        guard = "PASS" if (abs(cd) <= 0.05 and dr >= 0.70 and da <= 0.85) else "FAIL"
        print(f"  {ep:>5} {vc:>10.4f} {da:>10.4f} {dr:>10.4f} {cd:>+10.4f} {guard:>7}")

    disc_accs = [h.get("disc_acc", np.nan) for h in history]
    div_ratios= [h.get("div_ratio", np.nan) for h in history]
    print(f"\n  disc_acc: mean={np.nanmean(disc_accs):.4f}  "
          f"(range [{np.nanmin(disc_accs):.3f}, {np.nanmax(disc_accs):.3f}])")
    print(f"  div_ratio: mean={np.nanmean(div_ratios):.4f}  "
          f"(range [{np.nanmin(div_ratios):.3f}, {np.nanmax(div_ratios):.3f}])")
    near_chance = np.nanmean([abs(d - 0.5) for d in disc_accs]) < 0.08
    print(f"\n  Discriminator near chance (|acc - 0.5| < 0.08): {'YES - adversarial signal active' if near_chance else 'NO - discriminator dominates generator'}")

# ─── Section 4c: Post-hoc gradient norm ratio ──────────────────────────────
print(f"\n{'=' * 90}")
print("SECTION 4c: POST-HOC GRADIENT NORM RATIO  (1 validation batch, Stage 5 weights)")
print(f"  R_grad = ||grad(lambda_adv * L_adv)|| / (||grad(L_4C)|| + eps)")
print(f"{'=' * 90}")

def _load_module(alias, fname):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

try:
    mods = {
        "prep":   _load_module("_pa_prep",   "01_data_preprocessing.py"),
        "ode":    _load_module("_pa_ode",    "02_physics_latent.py"),
        "enc":    _load_module("_pa_enc",    "03_model_encoder.py"),
        "dec":    _load_module("_pa_dec",    "04_model_decoder.py"),
        "gen":    _load_module("_pa_gen",    "05_model_generator.py"),
        "disc6":  _load_module("_pa_disc6",  "06_model_discriminator.py"),
        "train":  _load_module("_pa_train",  "08_training.py"),
        "eval9":  _load_module("_pa_eval9",  "09_evaluation.py"),
        "stoch":  _load_module("_pa_stoch",  "10_stochastic_residual.py"),
        "s4b":    _load_module("_pa_s4b",    "14_stage4b_ar1_guided_generator.py"),
        "s15":    _load_module("_pa_s15",    "15_stage5_adversarial_finetune.py"),
    }
    _loaded_mods = True
except Exception as e:
    print(f"  [WARNING] Could not load all modules: {e}")
    _loaded_mods = False

LAMBDA_ADV  = 0.005
LAMBDA_CRPS = 1.0
LAMBDA_AR1  = 0.50
LAMBDA_VAR  = 0.30
STABLE_IDX  = mods["s4b"].STABLE_FEAT_INDICES if _loaded_mods else [0,1,2,3]

if _loaded_mods:
    from torch.utils.data import DataLoader

    # Build backbone
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = mods["enc"].PhysicsEncoder()
            self.decoder   = mods["dec"].SparsePhysicsDecoder()
            self.ode       = mods["ode"].PhysicsODE()
            self.alpha_net = mods["ode"].DeviceAlphaNet()
            self.generator = mods["gen"].PITimeGANGenerator()
            self.disc      = mods["disc6"].PITimeGANDiscriminator()

    model3 = mods["eval9"].load_model(CKPT_S3, _Model).to("cpu")
    model3.eval()
    for p in model3.parameters():
        p.requires_grad_(False)

    # Load Stage 5 generator
    gen_s5 = mods["s4b"].AR1GuidedResidualGeneratorStable().to("cpu")
    gen_s5.load_state_dict(sd_s5, strict=False)
    gen_s5.train()

    # Load Stage 4C generator
    gen_4c = mods["s4b"].AR1GuidedResidualGeneratorStable().to("cpu")
    gen_4c.load_state_dict(sd_4c, strict=False)
    gen_4c.eval()

    # Load discriminator from Stage 5 checkpoint
    disc_s5 = mods["s15"].ResidualDiscriminator(feature_dim=len(STABLE_IDX)).to("cpu")
    if "disc_state" in ckpt_s5_data:
        disc_s5.load_state_dict(ckpt_s5_data["disc_state"], strict=True)
    disc_s5.eval()

    # Cache one validation batch
    dataset   = mods["prep"].load_dataset()
    val_idx   = dataset["split"]["val"][:8]          # small subset for speed
    val_ds    = mods["train"].DeviceDegradationDataset(dataset, val_idx)
    val_dl    = DataLoader(val_ds, batch_size=8, shuffle=False,
                           collate_fn=mods["train"].collate_fn)
    _cache    = mods["s4b"]._cache_trajectories(model3, val_dl, "cpu",
                                                 mods["train"]._forward,
                                                 cfg.STAGE3_PREFIX_LEN)
    rec       = _cache[0]

    z_pfx = rec["z_pfx"]
    x_hat = rec["x_hat"]
    x_true= rec["x_true"]
    mask  = rec["mask"]
    T_K   = rec["T_K"]
    log_t = rec["log_t"]
    x0    = rec["x0"]
    plen  = rec["plen"]
    T_future = rec["T_len"] - plen
    sfx   = torch.tensor(STABLE_IDX)

    crps_fn = mods["s4b"].crps_mc_loss
    ar1_fn  = mods["s4b"]._fit_ar1_targets
    var_fn  = mods["s4b"]._temp_equalized_var_loss

    def _grad_norm(gen, loss_fn, retain=False):
        """Compute gradient norm of loss_fn w.r.t. gen.parameters()."""
        gen.zero_grad()
        loss = loss_fn()
        loss.backward(retain_graph=retain)
        total = sum(p.grad.norm().item() ** 2
                    for p in gen.parameters() if p.grad is not None) ** 0.5
        gen.zero_grad()
        return total, loss.item()

    # 1) Non-adversarial loss (CRPS + AR1 + Var)
    def l_nonadv():
        deltas  = gen_s5.sample_n(z_pfx, T_K, x0, log_t, n_samples=2, T_future=T_future)
        x_pf    = x_hat[:, plen:, sfx].unsqueeze(0) + deltas
        xpfx    = x_hat[:, :plen, sfx].unsqueeze(0).expand(2, -1, -1, -1)
        xpv     = torch.cat([xpfx, x_pf], dim=2)
        crps    = crps_fn(xpv, x_true[:, :, sfx], mask, prefix_len=plen)
        rho_p, sig_p = gen_s5._context_params(z_pfx, T_K, x0, log_t)
        rho_t, sig_t = ar1_fn(x_true, x_hat, plen, feat_indices=STABLE_IDX)
        ar1l    = ((rho_p - rho_t) ** 2).mean()
        varl    = var_fn(sig_p, sig_t, T_K)
        return LAMBDA_CRPS * crps + LAMBDA_AR1 * ar1l + LAMBDA_VAR * varl

    # 2) Adversarial loss only
    def l_adv():
        deltas  = gen_s5.sample_n(z_pfx, T_K, x0, log_t, n_samples=1, T_future=T_future)
        adv_res = deltas[0]                      # (B, T_f, 4)
        fm      = mask[:, plen:].bool()
        logit   = disc_s5(adv_res, T_K, fm)
        return LAMBDA_ADV * torch.nn.functional.binary_cross_entropy_with_logits(
            logit, torch.ones_like(logit))

    import torch.nn.functional as F

    norm_nonadv, val_nonadv = _grad_norm(gen_s5, l_nonadv, retain=True)
    norm_adv,    val_adv    = _grad_norm(gen_s5, l_adv)
    R_grad = norm_adv / (norm_nonadv + 1e-8)

    print(f"  ||grad(L_4C)||           = {norm_nonadv:.6f}  (L_4C value = {val_nonadv:.5f})")
    print(f"  ||grad(lambda_adv*L_adv)|| = {norm_adv:.6f}  (L_adv value = {val_adv:.5f})")
    print(f"  R_grad = {R_grad:.6f}  (adversarial share of gradient)")
    if R_grad < 0.01:
        print("  INTERPRETATION: Adversarial gradient is < 1% of non-adversarial gradient.")
        print("                  Adversarial signal is present but very small (as designed with lambda_adv=0.005).")
    elif R_grad < 0.05:
        print("  INTERPRETATION: Adversarial gradient is 1-5% of non-adversarial gradient (moderate regulariser).")
    else:
        print("  INTERPRETATION: Adversarial gradient is significant.")

# ─── Section 5: Distribution comparison ────────────────────────────────────
print(f"\n{'=' * 90}")
print("SECTION 5: RESIDUAL DISTRIBUTION COMPARISON (from w1_increments in fold CSV)")
print(f"{'=' * 90}")

# Use w1_increments_overall from existing CSVs
w1_4c = [col(rows_4c[k], "w1_increments_overall") for k in common]
w1_s5 = [col(rows_s5[k], "w1_increments_overall") for k in common]
delta_w1 = [a - b for a, b in zip(w1_s5, w1_4c)]  # negative = S5 is closer to true

print(f"\n  W1 increment distance  (lower is better  =  generated increments closer to real)")
dw1 = summarise(delta_w1, sign=-1)
print(f"  dW1 (S5-S4C): mean={dw1['mean']:+.5f}  median={dw1['median']:+.5f}  "
      f"SD={dw1['std']:.5f}  {dw1['n_favorable']}/15 pairs favourable")
lo_w1, hi_w1 = fold_cluster_bootstrap_ci(delta_w1, seeds_list)
print(f"  Bootstrap 95% CI: [{lo_w1:+.5f}, {hi_w1:+.5f}]")

if _loaded_mods:
    print(f"\n  Per-feature residual statistics: skewness, kurtosis, ACF(lag=1)")
    print(f"  Generating S=200 samples on test set from Stage 4C and Stage 5...")
    from scipy import stats as spstats

    test_idx = dataset["split"]["test"]
    test_ds  = mods["train"].DeviceDegradationDataset(dataset, test_idx)
    test_dl  = DataLoader(test_ds, batch_size=16, shuffle=False,
                          collate_fn=mods["train"].collate_fn)
    test_cache = mods["s4b"]._cache_trajectories(model3, test_dl, "cpu",
                                                   mods["train"]._forward,
                                                   cfg.STAGE3_PREFIX_LEN)

    def generate_samples(gen, cache, n_samples=200):
        """Generate (S, N, T, F_full=6) samples; stable features stochastic, leakage=x_mean."""
        all_pred = []
        all_true = []
        all_mask = []
        smp_list_stable = []

        for rec in cache:
            z_pfx = rec["z_pfx"]
            x_hat_ = rec["x_hat"]
            x_true_= rec["x_true"]
            mask_  = rec["mask"]
            T_K_   = rec["T_K"]
            log_t_ = rec["log_t"]
            x0_    = rec["x0"]
            plen_  = rec["plen"]
            T_f_   = rec["T_len"] - plen_
            if T_f_ <= 0:
                continue
            with torch.no_grad():
                deltas_ = gen.sample_n(z_pfx, T_K_, x0_, log_t_, n_samples=n_samples, T_future=T_f_)
            # Build full 6-feature sample (stable columns stochastic, leakage = x_hat)
            B_ = x_hat_.shape[0]
            T_ = x_hat_.shape[1]
            smp_ = np.repeat(x_hat_.numpy()[None,:,:,:], n_samples, axis=0)  # (S,B,T,6)
            col_ = np.array(STABLE_IDX)
            for fi_loc, fi_glob in enumerate(col_):
                smp_[:, :, plen_:, fi_glob] += deltas_[:,: ,: ,fi_loc].numpy()
            smp_list_stable.append(smp_)
            all_pred.append(x_hat_.numpy())
            all_true.append(x_true_.numpy())
            all_mask.append(mask_.numpy())

        x_pred_np = np.concatenate(all_pred, axis=0)
        x_true_np = np.concatenate(all_true, axis=0)
        mask_np   = np.concatenate(all_mask, axis=0)
        smp_np    = np.concatenate(smp_list_stable, axis=1)  # (S, N_total, T, 6)
        return x_pred_np, x_true_np, mask_np, smp_np

    print("  [Stage 4C]", end=" ", flush=True)
    xp_4c, xt_4c, m_4c, smp_4c = generate_samples(gen_4c, test_cache, n_samples=200)
    print("done")
    print("  [Stage 5] ", end=" ", flush=True)
    xp_s5, xt_s5, m_s5, smp_s5 = generate_samples(gen_s5, test_cache, n_samples=200)
    print("done")

    plen_  = cfg.STAGE3_PREFIX_LEN
    fut_m  = m_4c.copy().astype(bool)
    fut_m[:, :plen_] = False

    def residual_stats(smp, x_pred, x_true, mask_fut, feat_idx):
        """Compute skewness, kurtosis, ACF-lag1 for generated residuals of one feature."""
        fi = feat_idx
        # Generated residuals across all samples: (S, N, T) -> future positions
        gen_res = smp[:, :, :, fi] - x_pred[np.newaxis, :, :, fi]  # (S, N, T)
        # Flatten over S, N, T (future only)
        fut_vals = gen_res[:, mask_fut].ravel()
        # True residuals
        true_res = x_true[:, :, fi] - x_pred[:, :, fi]  # (N, T)
        true_res_fut = true_res[mask_fut]

        skew_gen  = float(spstats.skew(fut_vals))
        kurt_gen  = float(spstats.kurtosis(fut_vals))
        skew_true = float(spstats.skew(true_res_fut))
        kurt_true = float(spstats.kurtosis(true_res_fut))

        # ACF lag-1 for generated (average over devices × samples)
        acf_gen_list = []
        for n in range(gen_res.shape[1]):
            fut_steps = np.where(mask_fut[n])[0]
            if len(fut_steps) < 2:
                continue
            # Average ACF over all samples for this device
            r = gen_res[:, n, fut_steps]   # (S, T_fut)
            for s in range(r.shape[0]):
                ts = r[s]
                if len(ts) >= 2 and ts.std() > 1e-8:
                    c = np.corrcoef(ts[:-1], ts[1:])[0, 1]
                    if np.isfinite(c):
                        acf_gen_list.append(float(c))
        acf_gen  = float(np.mean(acf_gen_list)) if acf_gen_list else np.nan

        # True ACF
        acf_true_list = []
        for n in range(x_true.shape[0]):
            fut_steps = np.where(mask_fut[n])[0]
            if len(fut_steps) >= 2:
                ts = true_res[n, fut_steps]
                v  = x_true[n, fut_steps, fi]
                valid = ~np.isnan(v)
                ts2 = ts[valid]
                if len(ts2) >= 2 and np.std(ts2) > 1e-8:
                    acf_true_list.append(float(np.corrcoef(ts2[:-1], ts2[1:])[0,1]))
        acf_true = float(np.mean(acf_true_list)) if acf_true_list else np.nan

        return dict(
            skew_gen=skew_gen,   skew_true=skew_true,
            kurt_gen=kurt_gen,   kurt_true=kurt_true,
            acf1_gen=acf_gen,    acf1_true=acf_true,
        )

    print(f"\n  {'Feature':<8} {'Stat':<12} {'Stage4C':>9} {'Stage5':>9} {'True':>9}")
    print(f"  {'-' * 60}")

    distrib_results = {}
    for fi, fname in enumerate(cfg.FEATURES[:4]):  # stable features only
        r4c  = residual_stats(smp_4c, xp_4c, xt_4c, fut_m, fi)
        rs5  = residual_stats(smp_s5, xp_s5, xt_s5, fut_m, fi)
        distrib_results[fname] = {"stage4c": r4c, "stage5": rs5}
        for stat_key, label in [("skew_gen","Skewness"), ("kurt_gen","Kurtosis"), ("acf1_gen","ACF(lag1)")]:
            tv = r4c.get(stat_key.replace("gen","true"), np.nan)
            print(f"  {fname:<8} {label:<12} {r4c[stat_key]:>9.4f} {rs5[stat_key]:>9.4f} {tv:>9.4f}")
        print()

    # Discriminator distinguishability
    print(f"\n  Discriminator distinguishability (Stage 5 disc applied to real and generated):")
    print(f"  D(real) mean >> 0.5 means disc correctly identifies real trajectories")
    print(f"  D(gen_S5) > D(gen_S4C) means S5 generates MORE realistic residuals")

    scores = {"real": [], "gen_4c": [], "gen_s5": []}
    with torch.no_grad():
        for rec in test_cache:
            plen_ = rec["plen"]
            x_hat_r = rec["x_hat"]
            x_true_r= rec["x_true"]
            T_K_r   = rec["T_K"]
            mask_r  = rec["mask"]
            z_pfx_r = rec["z_pfx"]
            log_t_r = rec["log_t"]
            x0_r    = rec["x0"]
            T_future_r = rec["T_len"] - plen_
            if T_future_r <= 0:
                continue
            sfx_t   = torch.tensor(STABLE_IDX)
            fm_r    = mask_r[:, plen_:].bool()

            # Real residuals
            real_res = (x_true_r[:, plen_:, sfx_t] - x_hat_r[:, plen_:, sfx_t])
            lv_real = disc_s5(real_res, T_K_r, fm_r)
            scores["real"].extend(lv_real.sigmoid().tolist())

            # Generated S4C
            d4c = gen_4c.sample_n(z_pfx_r, T_K_r, x0_r, log_t_r, n_samples=5, T_future=T_future_r)
            for s in range(5):
                lv = disc_s5(d4c[s], T_K_r, fm_r)
                scores["gen_4c"].extend(lv.sigmoid().tolist())

            # Generated S5
            ds5 = gen_s5.sample_n(z_pfx_r, T_K_r, x0_r, log_t_r, n_samples=5, T_future=T_future_r)
            for s in range(5):
                lv = disc_s5(ds5[s], T_K_r, fm_r)
                scores["gen_s5"].extend(lv.sigmoid().tolist())

    for label, key in [("Real residuals  ", "real"),
                        ("Stage 4C gen    ", "gen_4c"),
                        ("Stage 5 gen     ", "gen_s5")]:
        v = np.array(scores[key])
        print(f"  {label}: mean={v.mean():.4f}  SD={v.std():.4f}  "
              f"(n={len(v)})")
    realism_improved = np.mean(scores["gen_s5"]) > np.mean(scores["gen_4c"])
    print(f"\n  D(gen_S5) {'>' if realism_improved else '<='} D(gen_S4C)  =>  "
          f"Stage 5 {'improves' if realism_improved else 'does not improve'} trajectory realism per discriminator.")

# ─── Save JSON ──────────────────────────────────────────────────────────────
results = {
    "paired_stats": {m: stats_summary[m] for m in metrics},
    "bootstrap_ci": {m: list(ci_results[m]) for m in metrics},
    "non_inferiority": {
        "CRPSS":        {"status": crpss_status,  "ci_low": ci_results["CRPSS"][0]},
        "Cov90_ptwise": {"status": cov90_status,  "ci_low": ci_results["Cov90_ptwise"][0]},
        "MACE":         {"status": mace_status,   "ci_high": ci_results["MACE"][1]},
        "overall_pass": ni_all_pass,
    },
    "param_change": {
        "total_abs": total_abs,
        "total_rel": total_rel,
    },
}
if _loaded_mods:
    results["gradient_norms"] = {
        "norm_nonadv": norm_nonadv,
        "norm_adv":    norm_adv,
        "R_grad":      R_grad,
    }

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved -> {OUT_JSON}")
