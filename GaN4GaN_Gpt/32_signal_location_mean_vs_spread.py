# -*- coding: utf-8 -*-
"""
32_signal_location_mean_vs_spread.py
====================================
Why conditioning on z_phys cannot help: the signal is in the residual MEAN,
and the generator can only express spread.

The chain
---------
16_  physics-conditioned / no-physics / shuffled-physics tie, on 3 seeds,
     after every fix. Three interventions were tried and all tied:
       lambda_phys_sens     = 0.5   A-B -0.3 %, A-C +0.9 %
       lambda_zphys_contrast= 0.5   A-B -1.1 %, A-C -1.3 %
     all under the 2.7 % noise floor.
30_  z_phys is NOT a function of (T,t): 94-96 % of its spread survives
     conditioning on temperature.
31_  z_phys explains 37 % of the per-device Stage 3 residual; temperature
     explains 0.01 %. Survives a permutation null and 5-fold CV.
32_  (this file) the predictable part is the SIGNED MEAN, which the
     generator's conditioning pathway structurally cannot represent.

The measurement
---------------
Per feature, R^2 gain of z_phys over temperature alone, for the signed mean
residual and for its spread:

  feature          SIGNED mean    spread
  Vth                  +0.0336   +0.0720
  IDSS                 +0.2236   +0.0771
  RON                  +0.3123   +0.0539
  gmmax                +0.2183   +0.0670
  IDLeak               +0.2276   +0.1006
  IGLeak               +0.1511   +0.1288
  SS_lin               +0.1600   +0.0895
  SS_sat               +0.1083   +0.0636
  gm_fwhm_sat          +0.3984   +0.0331
  DIBL                 +0.1459   +0.0595
  V_gmpeak_sat         +0.1144   +0.0636
  MEAN                 +0.1903   +0.0735

The mean carries 2.6x the signal of the spread.

Why that is fatal for the current architecture
----------------------------------------------
_context_params returns exactly two things: rho (AR(1) correlation) and sigma
(band width). The AR(1) process it drives is ZERO-MEAN. Stage 4C adds a
per-feature offset, but offset_raw is a bare nn.Parameter -- one constant per
feature, shared across all devices, NOT a function of z_phys.

So there is no path from the physics latent to the residual mean. z_phys can
only widen or narrow the band and change its autocorrelation. The strongest
part of the signal is unreachable, which is why the ablation ties no matter
how hard the loss pushes.

Direct confirmation: the contrastive loss failed at its OWN objective. On the
trained generator, CRPS(shuffled) - CRPS(real) is NEGATIVE both without it
(-0.00272) and with it (-0.00323), and real z_phys wins on only 26-30 % of
devices. The optimiser could not make the pathway prefer the correct z_phys,
because the pathway cannot use it.

What this implies
-----------------
The fix is architectural, not a loss weight: make the residual mean a
function of z_phys -- i.e. replace the per-feature constant offset with a
z_phys-conditioned mean head. That is a falsifiable prediction: if this
diagnosis is right, the ablation should separate once the mean is
conditioned, and this is the experiment to run next.

Usage
-----
    python 32_signal_location_mean_vs_spread.py
"""
import sys, os, pickle, numpy as np, torch
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
dl = DataLoader(tm.DeviceDegradationDataset(ds, list(range(len(ds["device_ids"])))),
                batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=tm.collate_fn)
cache = s4a._cache_trajectories(model, dl, "cpu", tm._forward, P)

zs, ms, sds, Ts = [], [], [], []
for rec in cache:
    xt, xh = rec["x_true"].numpy(), rec["x_hat"].numpy()
    mk = rec["mask"].numpy().astype(bool)[:, P:, None]
    d = np.where(mk, xt[:, P:] - xh[:, P:], np.nan)
    with np.errstate(invalid="ignore"):
        ms.append(np.nanmean(d, axis=1))   # (B,F) signed mean over time
        sds.append(np.nanstd(d, axis=1))   # (B,F) spread over time
    zs.append(rec["z_pfx"].numpy()); Ts.append(rec["T_K"].numpy())
zl = np.concatenate(zs); MU = np.concatenate(ms); SD = np.concatenate(sds)
tt = np.concatenate(Ts).ravel()

def r2(cols, y):
    ok = np.isfinite(y) & np.all(np.isfinite(np.column_stack(cols)), axis=1)
    if ok.sum() < 20: return float("nan")
    X = np.column_stack([np.ones(ok.sum())] + [c[ok] for c in cols]); yy = y[ok]
    b, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return 1 - ((yy - X @ b) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()

zc = [zl[:, j] for j in range(zl.shape[1])]
print("")
print("R^2 of z_phys (over temperature alone), per feature")
print("%-14s %14s %14s" % ("feature", "SIGNED mean", "spread (sd)"))
gm, gs = [], []
for f, nm in enumerate(cfg.FEATURES):
    a1 = r2([tt], MU[:, f]); b1 = r2([tt] + zc, MU[:, f])
    a2 = r2([tt], SD[:, f]); b2 = r2([tt] + zc, SD[:, f])
    gm.append(b1 - a1); gs.append(b2 - a2)
    print("%-14s %+13.4f %+13.4f" % (nm, b1 - a1, b2 - a2))
print("%-14s %+13.4f %+13.4f" % ("MEAN", np.nanmean(gm), np.nanmean(gs)))
print("")
print("The generator routes z_phys only into rho and sigma, so only the right")
print("column is reachable. Signal in the left column cannot be expressed.")
