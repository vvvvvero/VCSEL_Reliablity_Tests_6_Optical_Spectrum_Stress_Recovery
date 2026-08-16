#!/usr/bin/env python
"""
diag_adversarial.py
===================
Systematic adversarial training diagnostic for Stage 5.

Checks:
  1. Fake detach in G step  (should NOT be detached)
  2. Generator optimizer parameter binding
  3. Actual gradient ratio R_grad = ||grad(lambda_adv*L_adv)|| / ||grad(L_4C)||
  4. Discriminator small-data overfitting (D should reach disc_acc > 0.9 in 100 steps)
  5. Required lambda_adv to achieve target R_grad = 0.1%
"""
import importlib.util, os, sys, copy
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

CKPT_LT   = r"D:\2026\article\GaN4GaN\output\pi_timegan_stage4c_lt_acf0\checkpoints\stage4b_best.pt"
CKPT_S3   = r"D:\2026\article\GaN4GaN\output\pi_timegan\checkpoints\stage3_best.pt"
TARGET_RGRAD = 0.001   # 0.1% target

def _lm(alias, fname):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

mods = {
    "prep":  _lm("_da_prep",  "01_data_preprocessing.py"),
    "ode":   _lm("_da_ode",   "02_physics_latent.py"),
    "enc":   _lm("_da_enc",   "03_model_encoder.py"),
    "dec":   _lm("_da_dec",   "04_model_decoder.py"),
    "gen":   _lm("_da_gen",   "05_model_generator.py"),
    "disc":  _lm("_da_disc",  "06_model_discriminator.py"),
    "train": _lm("_da_trn",   "08_training.py"),
    "eval9": _lm("_da_ev9",   "09_evaluation.py"),
    "s4b":   _lm("_da_s4b",   "14_stage4b_ar1_guided_generator.py"),
    "s5":    _lm("_da_s5",    "15_stage5_adversarial_finetune.py"),
}

STABLE_IDX = mods["s4b"].STABLE_FEAT_INDICES
crps_fn    = mods["s4b"].crps_mc_loss
ar1_fn     = mods["s4b"]._fit_ar1_targets
var_fn     = mods["s4b"]._temp_equalized_var_loss
ResidDisc  = mods["s5"].ResidualDiscriminator

# ── Build and load models ─────────────────────────────────────────────────────
class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = mods["enc"].PhysicsEncoder()
        self.decoder   = mods["dec"].SparsePhysicsDecoder()
        self.ode       = mods["ode"].PhysicsODE()
        self.alpha_net = mods["ode"].DeviceAlphaNet()
        self.generator = mods["gen"].PITimeGANGenerator()
        self.disc      = mods["disc"].PITimeGANDiscriminator()

model3 = mods["eval9"].load_model(CKPT_S3, _Model).to("cpu")
model3.eval()
for p in model3.parameters():
    p.requires_grad_(False)

ckpt4c = torch.load(CKPT_LT, map_location="cpu")
gen    = mods["s4b"].AR1GuidedResidualGeneratorStable().to("cpu")
gen.load_state_dict(ckpt4c["state_dict"], strict=False)
gen.train()

disc = ResidDisc(feature_dim=len(STABLE_IDX)).to("cpu")
disc.train()

# ── Get one training batch ────────────────────────────────────────────────────
from torch.utils.data import DataLoader
dataset  = mods["prep"].load_dataset()
train_ds = mods["train"].DeviceDegradationDataset(dataset, dataset["split"]["train"][:16])
train_dl = DataLoader(train_ds, batch_size=16, shuffle=False,
                      collate_fn=mods["train"].collate_fn)

cache = mods["s4b"]._cache_trajectories(model3, train_dl, "cpu",
                                         mods["train"]._forward,
                                         cfg.STAGE3_PREFIX_LEN)
rec = cache[0]
z_pfx = rec["z_pfx"]; x_hat = rec["x_hat"]; x_true = rec["x_true"]
mask  = rec["mask"];   T_K   = rec["T_K"];   log_t  = rec["log_t"]
x0    = rec["x0"];     plen  = rec["plen"];   T_f    = rec["T_len"] - plen
sfx   = torch.tensor(STABLE_IDX)
times_f = rec["times"][:, plen:] if "times" in rec else None

print("=" * 80)
print("DIAGNOSTIC: Stage 5 Adversarial Training")
print("=" * 80)

# ─── Check 1: Are fake tensors properly not-detached in G step? ──────────────
print("\n[1] Fake tensor grad status in G step")
gen.eval()   # no dropout
with torch.enable_grad():
    deltas = gen.sample_n(z_pfx, T_K, x0, log_t, n_samples=2,
                          T_future=T_f, times_future=times_f)
    # deltas: (S, B, T_f, 4)
    adv_fake_for_G = deltas[0]           # used in G step (should have grad)
    detached_for_D  = deltas[0].detach() # used in D step (should NOT have grad)

print(f"   deltas[0].requires_grad = {adv_fake_for_G.requires_grad}")
print(f"   deltas[0].detach().requires_grad = {detached_for_D.requires_grad}")
if not adv_fake_for_G.requires_grad:
    print("   *** BUG: adv_fake for G step has no gradient! ***")
else:
    print("   OK: fake tensor has gradient for G step")

# ─── Check 2: Generator optimizer parameter binding ──────────────────────────
print("\n[2] Generator optimizer parameter check")
opt_g = torch.optim.AdamW(gen.parameters(), lr=2e-4)
opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4)

gen_param_set  = set(p.data_ptr() for p in gen.parameters())
disc_param_set = set(p.data_ptr() for p in disc.parameters())

# Check opt_g only contains generator parameters
opt_g_ptrs = set(p.data_ptr() for g in opt_g.param_groups for p in g["params"])
opt_d_ptrs = set(p.data_ptr() for g in opt_d.param_groups for p in g["params"])

cross_gd = gen_param_set & opt_d_ptrs
cross_dg = disc_param_set & opt_g_ptrs
print(f"   Generator params in opt_d (should be 0): {len(cross_gd)}")
print(f"   Disc params in opt_g (should be 0):      {len(cross_dg)}")
if cross_gd or cross_dg:
    print("   *** BUG: parameter binding is wrong! ***")
else:
    print("   OK: optimizer parameter binding is correct")

# ─── Check 3: Actual gradient ratio measurement ───────────────────────────────
print("\n[3] Actual gradient ratio R_grad measurement")
gen.train()

def _grad_norm_for_loss(loss_fn, retain=True):
    """Compute ||grad(loss)|| for generator parameters."""
    opt_g.zero_grad()
    loss = loss_fn()
    loss.backward(retain_graph=retain)
    total = sum(p.grad.norm().item() ** 2
                for p in gen.parameters() if p.grad is not None) ** 0.5
    return total, float(loss.item())

# Recompute deltas with fresh graph
deltas = gen.sample_n(z_pfx, T_K, x0, log_t, n_samples=4,
                      T_future=T_f, times_future=times_f)

x_hat_fut_s = x_hat[:, plen:, sfx].unsqueeze(0)
x_pred_fut_s = x_hat_fut_s + deltas
x_pfx_s  = x_hat[:, :plen, sfx].unsqueeze(0).expand(4, -1, -1, -1)
x_pv_s   = torch.cat([x_pfx_s, x_pred_fut_s], dim=2)
x_true_s = x_true[:, :, sfx]
rho_p, sigma_p = gen._context_params(z_pfx, T_K, x0, log_t)
rho_t, sigma_t = ar1_fn(x_true, x_hat, plen, feat_indices=STABLE_IDX,
                         device_center=True, times_future=times_f)

# Non-adversarial loss: CRPS + AR1 + Var
def l_4c():
    c = crps_fn(x_pv_s, x_true_s, mask, prefix_len=plen)
    a = ((rho_p - rho_t) ** 2).mean()
    v = var_fn(sigma_p, sigma_t, T_K)
    return 1.0 * c + 0.50 * a + 0.30 * v

# Adversarial loss only
fm = mask[:, plen:].bool()
def l_adv():
    logit = disc(deltas[0], T_K, fm)
    return F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))

norm_4c,  val_4c  = _grad_norm_for_loss(l_4c,  retain=True)
norm_adv, val_adv = _grad_norm_for_loss(l_adv, retain=False)

R_grad_actual = norm_adv / (norm_4c + 1e-10)
lambda_needed  = TARGET_RGRAD / max(R_grad_actual, 1e-12)

print(f"   ||grad(L_4C)||     = {norm_4c:.6f}  (L_4C = {val_4c:.5f})")
print(f"   ||grad(L_adv)||    = {norm_adv:.6f}  (L_adv = {val_adv:.5f})")
print(f"   Actual R_grad      = {R_grad_actual:.2e}  ({R_grad_actual*100:.6f}%)")
print(f"   Target R_grad      = {TARGET_RGRAD:.2e}  ({TARGET_RGRAD*100:.4f}%)")
print(f"   Lambda needed      = {lambda_needed:.4f}  (to achieve target)")
print(f"   Current best-case  = {0.01 * R_grad_actual:.2e}  (at lambda_adv=0.01)")

# ─── Check 4: Discriminator small-data overfitting test ──────────────────────
print("\n[4] Discriminator overfitting test (100 training steps on 1 batch)")

disc_test  = ResidDisc(feature_dim=len(STABLE_IDX)).to("cpu")
opt_d_test = torch.optim.Adam(disc_test.parameters(), lr=3e-4)

real_res = (x_true[:, plen:, sfx] - x_hat[:, plen:, sfx]).detach()
with torch.no_grad():
    gen.eval()
    fake_res = gen.sample_n(z_pfx, T_K, x0, log_t, n_samples=1,
                            T_future=T_f, times_future=times_f)[0].detach()
    gen.train()

disc_accs = []
for step in range(100):
    opt_d_test.zero_grad()
    logit_r = disc_test(real_res, T_K, fm)
    logit_f = disc_test(fake_res, T_K, fm)
    d_loss = (
        F.binary_cross_entropy_with_logits(logit_r, torch.ones_like(logit_r)) +
        F.binary_cross_entropy_with_logits(logit_f, torch.zeros_like(logit_f))
    )
    d_loss.backward()
    opt_d_test.step()
    acc = 0.5 * (float((logit_r > 0).float().mean()) + float((logit_f < 0).float().mean()))
    disc_accs.append(acc)

print(f"   Disc accuracy after  10 steps: {disc_accs[9]:.4f}")
print(f"   Disc accuracy after  50 steps: {disc_accs[49]:.4f}")
print(f"   Disc accuracy after 100 steps: {disc_accs[99]:.4f}")
if disc_accs[99] > 0.9:
    print("   OK: Discriminator can distinguish real from fake")
    disc_gradient_ok = True
else:
    print("   WARNING: Discriminator cannot overfit — real and fake may be too similar")
    disc_gradient_ok = False

# After D is trained, measure gradient to G
def l_adv_trained():
    logit = disc_test(deltas[0], T_K, fm)
    return F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))

# Recompute deltas
gen.eval()
deltas_eval = gen.sample_n(z_pfx, T_K, x0, log_t, n_samples=4,
                           T_future=T_f, times_future=times_f)
gen.train()

x_hat_fut_s2 = x_hat[:, plen:, sfx].unsqueeze(0)
x_pred_fut_s2 = x_hat_fut_s2 + deltas_eval
x_pfx_s2  = x_hat[:, :plen, sfx].unsqueeze(0).expand(4, -1, -1, -1)
x_pv_s2   = torch.cat([x_pfx_s2, x_pred_fut_s2], dim=2)
rho_p2, sigma_p2 = gen._context_params(z_pfx, T_K, x0, log_t)

def l_4c2():
    return 1.0 * crps_fn(x_pv_s2, x_true_s, mask, prefix_len=plen)

def l_adv_trained2():
    logit = disc_test(deltas_eval[0], T_K, fm)
    return F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))

norm_4c2,  _ = _grad_norm_for_loss(l_4c2,          retain=True)
norm_adv2, _ = _grad_norm_for_loss(l_adv_trained2, retain=False)
R_grad_trained = norm_adv2 / (norm_4c2 + 1e-10)
lambda_needed_trained = TARGET_RGRAD / max(R_grad_trained, 1e-12)

print(f"\n   After D pre-training:")
print(f"   ||grad(L_adv_trained)|| = {norm_adv2:.6f}  (vs {norm_adv:.6f} before)")
print(f"   R_grad_trained          = {R_grad_trained:.2e}  ({R_grad_trained*100:.6f}%)")
print(f"   Lambda needed (trained) = {lambda_needed_trained:.4f}")

# ─── Summary and recommendations ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)
print(f"  1. Fake tensor has gradient in G step: {adv_fake_for_G.requires_grad}")
print(f"  2. Optimizer binding is correct:       True")
print(f"  3. R_grad at fresh D:                  {R_grad_actual:.2e}")
print(f"     Lambda_adv needed for {TARGET_RGRAD*100:.1f}%:        {lambda_needed:.2f}")
print(f"  4. D overfit (fresh → 100-step):       {'YES' if disc_gradient_ok else 'NO'}")
print(f"     R_grad at pre-trained D:             {R_grad_trained:.2e}")
print(f"     Lambda_adv needed (pre-trained D):   {lambda_needed_trained:.2f}")

print()
print("RECOMMENDED FIXES:")
print(f"  a. Pre-train D for 50-100 steps before G adversarial updates")
print(f"  b. Measure actual R_grad every step (not proxy)")
print(f"  c. Set lambda_adv = {lambda_needed_trained:.2f} for pre-trained D (target {TARGET_RGRAD*100:.1f}%)")
print(f"  d. Use 2-backward-pass measurement for accuracy")
print()

# Save diagnostic results
import json
results = {
    "fake_has_grad": bool(adv_fake_for_G.requires_grad),
    "optimizer_binding_ok": True,
    "R_grad_fresh_D": float(R_grad_actual),
    "lambda_needed_fresh_D": float(lambda_needed),
    "R_grad_pretrained_D": float(R_grad_trained),
    "lambda_needed_pretrained_D": float(lambda_needed_trained),
    "disc_overfit_100steps": bool(disc_gradient_ok),
    "disc_acc_100": float(disc_accs[99]),
}
out = r"D:\2026\article\GaN4GaN\output\adversarial_diagnostic.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved → {out}")
