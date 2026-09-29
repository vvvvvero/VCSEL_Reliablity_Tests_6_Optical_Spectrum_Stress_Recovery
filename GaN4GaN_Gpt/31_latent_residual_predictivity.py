# -*- coding: utf-8 -*-
"""
31_latent_residual_predictivity.py
==================================
Does z_phys predict the Stage 3 residual -- the quantity Stage 4C models?

Why this matters
----------------
16_ finds no difference between physics-conditioned, no-physics and
shuffled-physics generators. 30_ removes the explanation that was on record
(z_phys collapsing to a function of T). This script asks the next question:
maybe the information is there but says nothing about what Stage 3 got WRONG,
in which case the null would be a property of the data and no architecture
could fix it.

The data says the opposite, and this is the central diagnostic result:

    R^2  T only        0.0001
    R^2  T + z_phys    0.3749
    gain              +0.3748

Temperature -- the variable the whole accelerated test is organised around --
explains essentially NONE of the per-device residual. The physics latent
explains 37 % of it. Per-latent marginal gains: zF +0.317, zM +0.247,
zG +0.144, zB +0.077, zC +0.063, zL +0.023.

Two guards, because a 6-column fit on 192 points can manufacture R^2:

  permutation null (z rows shuffled, 500 draws)
      mean +0.0316, 95th pct +0.0645, max +0.1455, observed +0.3748, p < 0.002
  5-fold OUT-OF-SAMPLE R^2
      T only -0.0362   T + z_phys +0.2670

The gain survives both, so it is signal, not parameter counting.

What this implies for the A/B/C null
------------------------------------
The physics state is present (30_) and predictive (here), yet conditioning on
it changes nothing (16_). The failure is therefore in the CONDITIONING
PATHWAY of the residual generator -- how z_phys enters the context vector and
whether the training signal can reward using it -- not in the encoder, not in
the physics, and not in the data. That is an actionable target, and it is
where the physics-informed claim has to be won.

Usage
-----
    python 31_latent_residual_predictivity.py
"""
import sys, os, pickle, numpy as np, torch

BASE = r"c:\Users\veronica.gao.zhan\OneDrive - Centrum Lukasiewicz\Data\2026\VCSELs\programs\GaN4GaN_Gpt"
BASE = os.path.dirname(os.path.abspath(__file__)) if os.path.basename(os.getcwd()) != "GaN4GaN_Gpt" else os.getcwd()
sys.path.insert(0, BASE); os.chdir(BASE)
import config as cfg
cfg.PROCESSED_DATA_PATH = os.path.join(cfg.OUTPUT_PATH, "processed_data_ext_filtered.pkl")
EXT = os.path.join(cfg.OUTPUT_PATH, "ext11_filtered")

import importlib.util
def L(alias, fn):
    s = importlib.util.spec_from_file_location(alias, os.path.join(BASE, fn))
    m = importlib.util.module_from_spec(s); sys.modules[alias] = m; s.loader.exec_module(m); return m

abl = L("_abl", "16_ablation_physics_condition.py")
mods = abl._load_all()
s4a = sys.modules["_pi_stage4a_impl"]
model = s4a._build_model(mods)
ck = torch.load(os.path.join(EXT, "checkpoints", "stage3_best.pt"),
                map_location="cpu", weights_only=False)
res = model.load_state_dict(ck["model_state"], strict=False)
assert not [k for k in res.missing_keys if k.startswith("encoder")], "encoder not loaded"
model.eval()

ds = pickle.load(open(cfg.PROCESSED_DATA_PATH, "rb"))
from torch.utils.data import DataLoader
tm = mods["train"]
P = cfg.STAGE3_PREFIX_LEN
idx = list(range(len(ds["device_ids"])))
dl = DataLoader(tm.DeviceDegradationDataset(ds, idx), batch_size=cfg.BATCH_SIZE,
                shuffle=False, collate_fn=tm.collate_fn)
cache = s4a._cache_trajectories(model, dl, "cpu", tm._forward, P)

zs, rs, Ts = [], [], []
for rec in cache:
    xt, xh = rec["x_true"].numpy(), rec["x_hat"].numpy()
    # cache carries the per-timestep mask (B,T), not feature_mask; NaNs in
    # x_true already mark the point-level censoring 28_ applied
    mk = rec["mask"].numpy().astype(bool)[:, P:, None]
    d = np.where(mk, xt[:, P:] - xh[:, P:], np.nan)
    with np.errstate(invalid="ignore"):
        r = np.sqrt(np.nanmean(d ** 2, axis=(1, 2)))
    zs.append(rec["z_pfx"].numpy()); rs.append(r); Ts.append(rec["T_K"].numpy())

zl = np.concatenate(zs); r = np.concatenate(rs); tt = np.concatenate(Ts).ravel()
ok = np.isfinite(r)
r, zz, tt = r[ok], zl[ok], tt[ok]

def r2(Xd, y):
    Xd = np.column_stack([np.ones(len(y)), Xd])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return 1 - ((y - Xd @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()

a = r2(tt[:, None], r)
b = r2(np.column_stack([tt, zz]), r)
print("")
print("n devices %d   mean |resid| %.4f" % (len(r), r.mean()))
print("R^2  T only        %.4f" % a)
print("R^2  T + z_phys    %.4f" % b)
print("gain from z_phys   %+.4f" % (b - a))
print("")
print("per-latent marginal gain over T alone:")
for j, nm in enumerate(cfg.LATENT_NAMES):
    print("  %-5s %+.4f" % (nm, r2(np.column_stack([tt, zz[:, j]]), r) - a))

# --- is the gain real, or just 6 extra free parameters on 192 points? -------
rng = np.random.default_rng(0)
# (1) permutation null: shuffle z rows, refit. Destroys the pairing only.
null = np.array([r2(np.column_stack([tt, zz[rng.permutation(len(zz))]]), r) - a
                 for _ in range(500)])
print("")
print("permutation null (z rows shuffled, 500 draws):")
print("  mean %+.4f   95th pct %+.4f   max %+.4f" % (null.mean(), np.percentile(null, 95), null.max()))
print("  observed %+.4f  -> p = %.4f" % (b - a, (null >= (b - a)).mean()))

# (2) 5-fold out-of-sample R^2: in-sample R^2 always rises with more columns
def cv_r2(cols):
    o = rng.permutation(len(r)); f = np.array_split(o, 5); acc = []
    for k in range(5):
        te = f[k]; tr = np.concatenate([f[j] for j in range(5) if j != k])
        Xtr = np.column_stack([np.ones(len(tr))] + [c[tr] for c in cols])
        Xte = np.column_stack([np.ones(len(te))] + [c[te] for c in cols])
        bb, *_ = np.linalg.lstsq(Xtr, r[tr], rcond=None)
        acc.append(1 - ((r[te] - Xte @ bb) ** 2).sum() / ((r[te] - r[tr].mean()) ** 2).sum())
    return float(np.mean(acc))
cT = [tt]
cTZ = [tt] + [zz[:, j] for j in range(zz.shape[1])]
print("")
print("5-fold OUT-OF-SAMPLE R^2 (guards against parameter counting):")
print("  T only      %+.4f" % cv_r2(cT))
print("  T + z_phys  %+.4f" % cv_r2(cTZ))
