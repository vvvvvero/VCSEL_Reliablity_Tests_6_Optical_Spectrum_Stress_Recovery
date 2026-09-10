# -*- coding: utf-8 -*-
"""
30_latent_information_content.py
================================
How much information does z_phys carry beyond temperature?

Why this exists
---------------
The A/B/C ablation (16_) compares physics-conditioned (A), no-physics (B) and
shuffled-physics (C) residual generators. It found no significant difference,
and the explanation on record was that z_phys had collapsed to a deterministic
function of (T, t) -- per-device std ~1e-8 -- so all three conditions were
informationally identical.

That explanation is now WRONG, and this script is what shows it. On the
current backbone the per-device spread is large (zF span 0.53, zL 0.42, zC 0.50
on a [0,1] latent) and 94-96 % of it survives conditioning on temperature.
z_phys is NOT a relabelling of T: it carries real per-device information.

So the null result needs a different explanation. The remaining candidates are
that the residual generator cannot exploit that information, or that the
information is not predictive of the residual -- not that it is absent.

A measurement bug worth remembering
-----------------------------------
The first two versions of this measurement reported sd ~2e-2 with ratios near
1.0, and were meaningless: the checkpoint stores weights under "model_state",
while the loader tried "model_state_dict" then fell back to the whole
checkpoint dict. With strict=False that silently matched NOTHING, so every
number came from a RANDOMLY INITIALISED encoder. The assertion on
missing_keys, and the weight hash, are here so that cannot recur -- a run that
loads no encoder now fails instead of printing plausible numbers.

Usage
-----
    python 30_latent_information_content.py
"""
import sys, os, pickle, numpy as np, torch
BASE = r"c:\Users\veronica.gao.zhan\OneDrive - Centrum Łukasiewicz\Data\2026\VCSELs\programs\GaN4GaN_Gpt"
sys.path.insert(0, BASE); os.chdir(BASE)
import config as cfg
cfg.PROCESSED_DATA_PATH = os.path.join(cfg.OUTPUT_PATH, "processed_data_ext_filtered.pkl")
EXT = r"D:\2026\article\GaN4GaN\output\pi_timegan\ext11_filtered"

import importlib.util
def L(alias, fn):
    s = importlib.util.spec_from_file_location(alias, os.path.join(BASE, fn))
    m = importlib.util.module_from_spec(s); sys.modules[alias] = m; s.loader.exec_module(m); return m

abl = L("_abl", "16_ablation_physics_condition.py")
mods = abl._load_all()
s4a = sys.modules["_pi_stage4a_impl"]
model = s4a._build_model(mods)
ck = torch.load(os.path.join(EXT, "checkpoints", "stage3_best.pt"), map_location="cpu", weights_only=False)
sd = ck["model_state"]   # NOT model_state_dict; the fallback silently
                         # matched nothing and strict=False hid it
res = model.load_state_dict(sd, strict=False)
print(f"[ckpt] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
enc_missing = [k for k in res.missing_keys if k.startswith("encoder")]
print(f"[ckpt] encoder keys NOT loaded: {enc_missing}")
assert not enc_missing, "encoder was left at random init -- reading meaningless"
import hashlib
wh = hashlib.md5(model.encoder.gru.weight_ih_l0.detach().numpy().tobytes()).hexdigest()[:8]
print(f"[ckpt] encoder weight hash {wh}  (must be identical across runs)")
torch.manual_seed(0)
model.eval()

ds = pickle.load(open(cfg.PROCESSED_DATA_PATH, "rb"))
X = torch.tensor(np.asarray(ds["x"]), dtype=torch.float32)
Tk = torch.tensor(np.asarray(ds["T_K"]), dtype=torch.float32)
th = torch.tensor(np.asarray(ds["times_h"]), dtype=torch.float32)
x0 = torch.tensor(np.asarray(ds["x0_static"]), dtype=torch.float32)
FM = torch.tensor(np.asarray(ds["feature_mask"]), dtype=torch.bool)
P = 6  # prefix boundary used by Stage 4C

# Encoder input exactly as 08_training.py builds it:
# [x(11), feature_mask(11), T_norm, log_t, delta_log_t] -> 25
T_norm = ((Tk - cfg.T_REF_K) / cfg.T_REF_K).unsqueeze(1).unsqueeze(2).expand(-1, X.shape[1], 1)
log_t = torch.log(th + 1.0)
dlt = torch.zeros_like(log_t); dlt[:, :-1] = log_t[:, 1:] - log_t[:, :-1]
enc_input = torch.cat([torch.nan_to_num(X, nan=0.0), FM.float(), T_norm,
                       log_t.unsqueeze(2), dlt.unsqueeze(2)], dim=2)
assert enc_input.shape[-1] == 25, enc_input.shape
MK = torch.tensor(np.asarray(ds["mask"]), dtype=torch.bool)
with torch.no_grad():
    z, _ = model.encoder(enc_input[:, :P], MK[:, :P])
zl = z[:, -1]

zl = zl.numpy()
Tv = Tk.numpy().round(1)
print(f"z_phys at prefix boundary: shape {zl.shape}")
print(f"{'latent':<8}{'overall sd':>12}{'within-T sd':>14}{'ratio':>9}")
names = cfg.LATENT_NAMES
for j in range(zl.shape[1]):
    ov = float(np.std(zl[:, j]))
    w = float(np.mean([np.std(zl[Tv == t, j]) for t in np.unique(Tv)]))
    print(f"{names[j]:<8}{ov:>12.3e}{w:>14.3e}{(w/ov if ov>0 else float('nan')):>9.3f}")
print()
print(f"range of each latent across devices:")
for j in range(zl.shape[1]):
    print(f"  {names[j]:<5} min {zl[:,j].min():.4f}  max {zl[:,j].max():.4f}  span {(zl[:,j].max()-zl[:,j].min()):.4f}")
print()
print("within-T sd is the information a per-device physics state carries BEYOND")
print("temperature. If it is ~0, z_phys is a relabelling of T and A/B/C must tie.")
