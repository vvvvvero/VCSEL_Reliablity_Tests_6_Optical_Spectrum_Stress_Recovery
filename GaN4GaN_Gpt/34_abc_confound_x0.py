# -*- coding: utf-8 -*-
"""
34_abc_confound_x0.py
=====================
Why A/B/C cannot separate: condition B is not physics-free.

The problem with the experiment, not the model
----------------------------------------------
All three A/B/C conditions receive the observed prefix x0. Only z_phys and
dz_phys are zeroed (B) or permuted (C). But the encoder builds z_phys FROM
that prefix, so much of it is recoverable from what B still sees:

    latent   R^2 of z_phys on (T, x0)
    zG            0.2073
    zB            0.1299
    zF            0.4186
    zM            0.4319
    zL            0.1391
    zC            0.2735
    mean          0.2667

The two latents carrying the most residual signal in 31_ (zF, zM) are the two
most recoverable.

What that does to the comparison
--------------------------------
Predicting the per-device Stage 3 residual:

    T                     0.0001
    T + x0                0.3615     <- available in ALL of A/B/C
    T + z_phys            0.3749
    T + x0 + z_phys       0.5112
    unique to z_phys     +0.1497

    5-fold out-of-sample: T+x0 +0.2264, T+z_phys +0.2564, both +0.3284

x0 alone recovers nearly all of the 0.37 that 31_ credited to z_phys. So
condition B ("no physics") still has most of the physics signal, by a
different route. The ablation was never contrasting physics against
no-physics; it was contrasting physics-via-latent against
physics-via-prefix, which is a much smaller difference and one comparable to
the 2.7 % noise floor.

This is why four separate interventions all tied:

  variant                      A-B      A-C
  default                    +1.2 %   -0.8 %
  + phys_sens 0.5            -0.3 %   +0.9 %
  + contrastive 0.5          -1.1 %   -1.3 %
  + cond MEAN                -1.5 %   -1.9 %
  + cond MEAN + contrastive  -2.6 %   -0.0 %

and why the mean head collapsed to a constant even with a contrastive loss
pushing it (33_): a per-device mean built from z_phys is largely redundant
with one the network can already build from x0, so the optimiser has little
to gain by using the latent.

The honest conclusion
---------------------
The claim "physics conditioning does not help" is NOT supported by this
experiment, and neither is its opposite. The experiment cannot answer the
question as designed, because its control condition leaks the thing it
removes.

To answer it, the control has to withhold the prefix too -- compare
[z_phys, T, t] against [x0, T, t] against [z_phys, x0, T, t] -- so that the
physics latent is the only route to per-device information in its arm. That
ablation is implemented as D_z_only / E_x0_only / F_both in 16_.

RESULT (filtered backbone, cond-mean head, 70 epochs, early stopping DISABLED
so all three conditions get an identical budget -- verified: 70/70 epochs each,
zero early stops, 3 seeds):

  seed      D_z_only  E_x0_only     F_both      D-E
  42          0.1987     0.2002     0.1964    -0.7 %
  101         0.1929     0.1887     0.1843    +2.2 %
  202         0.1975     0.1996     0.1582    -1.1 %
  mean        0.1964     0.1962     0.1796    +0.1 %

D - E = +0.1 %, D wins 1/3 seeds -- far below the 2.7 % noise floor.

So with the leak removed and the budget equalised, the physics latent and the
raw prefix are INTERCHANGEABLE as conditioning information. The latent is not
worse, which matters: it reaches the same forecast quality from 6 numbers
instead of 11 raw observations, and those 6 carry mechanism labels. But it is
not better, and this experiment gives no support for a claim that it is.

An honest caveat on F_both. It is worst on test CRPSS (0.1796) while reaching
the BEST val_CRPS (0.0536 vs 0.0560 / 0.0565), and its seed spread is 6x D's
(sd 0.0195 vs 0.0031). This is not overfitting in the usual sense -- its final
train_CRPS (0.0853) is essentially the others' (0.0864 / 0.0865) -- and its
long-horizon RMSE is normal (0.3197 vs 0.3135 / 0.3098 at 2000 h), so the
POINT prediction is fine. The damage is in the interval: F has the lowest
Cov90 (0.6379 vs 0.6668 / 0.6747) and the worst MACE (0.0704 vs 0.0558 /
0.0512). Given both inputs the model narrows its bands and mis-calibrates,
unstably across seeds. F is therefore not a usable upper bound, and the D vs E
comparison -- which is the one the paper needs -- stands on its own.

Usage
-----
    python 34_abc_confound_x0.py
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
abl = L("_abl","16_ablation_physics_condition.py"); mods = abl._load_all()
s4a = sys.modules["_pi_stage4a_impl"]
model = s4a._build_model(mods)
ck = torch.load(os.path.join(EXT,"checkpoints","stage3_best.pt"),map_location="cpu",weights_only=False)
r0=model.load_state_dict(ck["model_state"],strict=False)
assert not [k for k in r0.missing_keys if k.startswith("encoder")]
model.eval()
ds = pickle.load(open(cfg.PROCESSED_DATA_PATH,"rb"))
from torch.utils.data import DataLoader
tm = mods["train"]; P=6
dl = DataLoader(tm.DeviceDegradationDataset(ds, list(range(len(ds["device_ids"])))),
                batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=tm.collate_fn)
cache = s4a._cache_trajectories(model, dl, "cpu", tm._forward, P)
Z,X0,T=[],[],[]
for rec in cache:
    Z.append(rec["z_pfx"].numpy()); X0.append(rec["x0"].numpy()); T.append(rec["T_K"].numpy())
Z=np.concatenate(Z); X0=np.concatenate(X0); T=np.concatenate(T).ravel()
X0=np.nan_to_num(X0)
def r2(Xd,y):
    Xd=np.column_stack([np.ones(len(y)),Xd])
    b,*_=np.linalg.lstsq(Xd,y,rcond=None)
    return 1-((y-Xd@b)**2).sum()/((y-y.mean())**2).sum()
print("Can the generator RECONSTRUCT z_phys from inputs it already has (T, x0)?")
print("%-6s %10s" % ("latent","R^2"))
vals=[]
for j,nm in enumerate(cfg.LATENT_NAMES):
    v=r2(np.column_stack([T,X0]),Z[:,j]); vals.append(v)
    print("%-6s %10.4f" % (nm,v))
print("mean R^2 %.4f" % np.mean(vals))
print("")
# The decisive number: how much of z_phys's predictive power for the residual
# is UNIQUE to it, i.e. survives already knowing T and x0?
MU=[]
for rec in cache:
    xt,xh=rec["x_true"].numpy(),rec["x_hat"].numpy()
    mk=rec["mask"].numpy().astype(bool)[:,P:,None]
    d=np.where(mk,xt[:,P:]-xh[:,P:],np.nan)
    with np.errstate(invalid="ignore"):
        MU.append(np.sqrt(np.nanmean(d**2,axis=(1,2))))
r=np.concatenate(MU); ok=np.isfinite(r)
rr,ZZ,XX,TT=r[ok],Z[ok],X0[ok],T[ok]
base=r2(np.column_stack([TT]),rr)
withx0=r2(np.column_stack([TT,XX]),rr)
withz=r2(np.column_stack([TT,ZZ]),rr)
both=r2(np.column_stack([TT,XX,ZZ]),rr)
print("")
print("Predicting the Stage 3 residual (in-sample R^2):")
print("  T                     %.4f" % base)
print("  T + x0                %.4f   <- available in ALL of A/B/C" % withx0)
print("  T + z_phys            %.4f" % withz)
print("  T + x0 + z_phys       %.4f" % both)
print("  UNIQUE to z_phys      %+.4f   (both minus T+x0)" % (both-withx0))
print("")
def cvr2(cols,y,seed=0):
    rng=np.random.default_rng(seed); o=rng.permutation(len(y)); f=np.array_split(o,5); acc=[]
    for k in range(5):
        te=f[k]; tr=np.concatenate([f[j] for j in range(5) if j!=k])
        A=np.column_stack([np.ones(len(tr))]+[c[tr] for c in cols])
        B=np.column_stack([np.ones(len(te))]+[c[te] for c in cols])
        b,*_=np.linalg.lstsq(A,y[tr],rcond=None)
        acc.append(1-((y[te]-B@b)**2).sum()/((y[te]-y[tr].mean())**2).sum())
    return float(np.mean(acc))
cx=[TT]+[XX[:,j] for j in range(XX.shape[1])]
cz=[TT]+[ZZ[:,j] for j in range(ZZ.shape[1])]
cb=cx+[ZZ[:,j] for j in range(ZZ.shape[1])]
print("5-fold OUT-OF-SAMPLE R^2 (x0 has %d columns, so in-sample favours it):" % XX.shape[1])
print("  T + x0                %+.4f" % cvr2(cx,rr))
print("  T + z_phys            %+.4f" % cvr2(cz,rr))
print("  T + x0 + z_phys       %+.4f" % cvr2(cb,rr))
print("")
print("If z_phys adds little OVER x0, then B ('no physics') still has most of")
print("the signal via x0, and A/B/C cannot separate however well mu is built.")
print("")
print("If this is high, condition B (z_phys zeroed) can REBUILD the physics")
print("state from x0, so B is not really 'no physics' -- and A/B/C would be")
print("expected to tie no matter how well the mean is conditioned.")
