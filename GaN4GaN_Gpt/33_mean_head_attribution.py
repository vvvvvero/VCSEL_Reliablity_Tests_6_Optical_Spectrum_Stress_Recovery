# -*- coding: utf-8 -*-
"""
33_mean_head_attribution.py
===========================
The conditioned mean head gains +18 % CRPSS and A/B/C still ties. Why?

The prediction being tested
---------------------------
32_ argued the A/B/C null was structural: the z_phys signal lives in the
residual MEAN (R^2 +0.190) while the generator could only express spread
(+0.074), because _context_params returns just (rho, sigma) driving a
zero-mean AR(1). The falsifiable prediction was that conditioning the mean on
z_phys would make A/B/C separate.

Result: HALF right, and the wrong half is the informative one.

  variant                    A_full   B_noph   C_shuf      A-B      A-C
  default (rho,sigma only)   0.1638   0.1619   0.1650    +1.2 %   -0.8 %
  + phys_sens 0.5            0.1667   0.1672   0.1653    -0.3 %   +0.9 %
  + contrastive 0.5          0.1629   0.1647   0.1651    -1.1 %   -1.3 %
  + CONDITIONED MEAN         0.1937   0.1966   0.1973    -1.5 %   -1.9 %

The mean head is a large real gain -- A_full CRPSS 0.1638 -> 0.1937, +18.3 %,
Cov90 0.655 -> 0.700 -- so the missing-capacity half of the diagnosis was
right. But all three conditions gain EQUALLY, so it is not being obtained
from z_phys.

What the head actually learned
------------------------------
Re-feeding the trained head with one input group neutralised at a time:

  input zeroed     mean |delta mu|    relative to |mu|
  z_phys               0.00022             0.8 %
  x0                   0.00010             0.3 %
  T                    0.00002             0.1 %
  log_t                0.00010             0.3 %

mu responds to essentially NOTHING. And per feature:

  feature        mean mu    sd across devices   sd/|mean|
  Vth            -0.0372         0.00004          0.1 %
  IDSS           -0.0419         0.00004          0.1 %
  RON            -0.0456         0.00005          0.1 %
  gmmax          -0.0453         0.00005          0.1 %
  ...            (every feature)                  0.1-0.2 %

The head collapsed to a PER-FEATURE CONSTANT -- functionally identical to
Stage 4C's offset_raw, just routed through an MLP. |mu| ~ 0.029 against a
bound of 0.60, so the bound is not what limits it.

The pattern this belongs to
---------------------------
This is the THIRD occurrence of one failure mode in this codebase. 14_ already
documents it for Ea_sigma: "when Ea_sigma is left learnable inside the
sigma-prediction path itself, the optimizer has no pressure to move it off a
sane init and the model instead collapses the context-dependent correction
term to near-zero."

Same shape here: a context-dependent correction, free to use z_phys, collapses
to its context-independent part. The CRPS objective is minimised just as well
by a constant offset, and nothing pushes the head to spend capacity on
per-device structure -- so it does not.

What this means for the physics claim
-------------------------------------
The A/B/C null is NOT explained by missing capacity alone, and the honest
statement is now:

  the physics latent is informative (31_: R^2 0.37 of the Stage 3 residual,
  vs 0.0001 for temperature), the information is concentrated in the residual
  mean (32_), the architecture can now represent that mean (+18 % CRPSS), and
  the optimiser still will not connect the two.

That points at the training signal rather than the architecture: a per-device
mean is only worth learning if the loss rewards per-device accuracy more than
a constant offset already does. The next thing to try is the contrastive loss
ON TOP of the mean head -- it was tested against a head that had no mean to
condition, so it never had a lever to pull.

Usage
-----
    python 33_mean_head_attribution.py
"""
import sys, os, glob, pickle, numpy as np, torch
BASE = os.getcwd(); sys.path.insert(0, BASE)
import config as cfg
cfg.PROCESSED_DATA_PATH = os.path.join(cfg.OUTPUT_PATH, "processed_data_ext_filtered.pkl")
EXT = os.path.join(cfg.OUTPUT_PATH, "ext11_filtered")
import importlib.util
def L(a, fn):
    s = importlib.util.spec_from_file_location(a, os.path.join(BASE, fn))
    m = importlib.util.module_from_spec(s); sys.modules[a] = m; s.loader.exec_module(m); return m
abl = L("_abl", "16_ablation_physics_condition.py"); mods = abl._load_all()
s4a = sys.modules["_pi_stage4a_impl"]
model = s4a._build_model(mods)
ck = torch.load(os.path.join(EXT, "checkpoints", "stage3_best.pt"), map_location="cpu", weights_only=False)
r0 = model.load_state_dict(ck["model_state"], strict=False)
assert not [k for k in r0.missing_keys if k.startswith("encoder")]
model.eval()
ds = pickle.load(open(cfg.PROCESSED_DATA_PATH, "rb"))
from torch.utils.data import DataLoader
tm = mods["train"]; P = cfg.STAGE3_PREFIX_LEN
dl = DataLoader(tm.DeviceDegradationDataset(ds, ds["split"]["test"]),
                batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=tm.collate_fn)
cache = s4a._cache_trajectories(model, dl, "cpu", tm._forward, P)

G = abl.PhysicsConditionGeneratorStable(condition_mode="full", cond_mean=True,
                                        noise_dim=16, hidden_dim=96)
import sys as _s
VAR = _s.argv[1] if len(_s.argv) > 1 else "ablation_abc_s42_cmean"
print("VARIANT:", VAR)
ckp = os.path.join(EXT, VAR, "checkpoints", "generator_full.pt")
G.load_state_dict(torch.load(ckp, map_location="cpu", weights_only=False)["state_dict"])
G.eval()

mus, parts = [], {k: [] for k in ["zero_zphys", "zero_x0", "zero_T", "zero_logt"]}
with torch.no_grad():
    for rec in cache:
        z, T, x0, lt = rec["z_pfx"], rec["T_K"], rec["x0"], rec["log_t"]
        zr = rec["z_ref"]
        _, _, mu = G._context_params(z, T, x0, lt, z_ref=zr)
        mus.append(mu.numpy())
        _, _, m1 = G._context_params(torch.zeros_like(z), T, x0, lt, z_ref=torch.zeros_like(zr))
        parts["zero_zphys"].append((m1 - mu).numpy())
        _, _, m2 = G._context_params(z, T, torch.zeros_like(x0), lt, z_ref=zr)
        parts["zero_x0"].append((m2 - mu).numpy())
        _, _, m3 = G._context_params(z, torch.full_like(T, 573.2), x0, lt, z_ref=zr)
        parts["zero_T"].append((m3 - mu).numpy())
        _, _, m4 = G._context_params(z, T, x0, torch.zeros_like(lt), z_ref=zr)
        parts["zero_logt"].append((m4 - mu).numpy())

MU = np.concatenate(mus)
print("")
print("trained mu: mean |mu| %.4f   sd %.4f   max |mu| %.4f  (bound %.2f)"
      % (np.abs(MU).mean(), MU.std(), np.abs(MU).max(), abl.MEAN_HEAD_BOUND))
print("")
print("how far mu moves when one input group is removed (bigger = more used):")
for k, v in parts.items():
    d = np.concatenate(v)
    print("  %-12s mean|delta mu| %.5f   relative to |mu| %.1f%%"
          % (k, np.abs(d).mean(), 100 * np.abs(d).mean() / max(np.abs(MU).mean(), 1e-12)))
print("")
# How much of mu is a per-feature CONSTANT vs device-dependent?
print("per-feature mu: is it a constant, or does it vary across devices?")
print("  %-14s %10s %10s %10s" % ("feature", "mean mu", "sd across", "sd/|mean|"))
names = [cfg.FEATURES[i] for i in G.STABLE_INDICES]
for f, nm in enumerate(names):
    col = MU[:, f]
    ratio = col.std() / max(abs(col.mean()), 1e-12)
    print("  %-14s %+10.4f %10.5f %9.1f%%" % (nm, col.mean(), col.std(), 100 * ratio))
print("")
print("A near-zero sd across devices means the head collapsed to Stage 4C's")
print("existing per-feature constant offset -- it gained the MEAN, not the")
print("CONDITIONING, which is exactly why all three A/B/C conditions gained")
print("equally.")
print("")
print("If zero_zphys moves mu far less than zero_x0, the head is rebuilding the")
print("mean from the observed prefix, not from the physics latent -- a real gain")
print("that A/B/C cannot distinguish, because it is available in all conditions.")
