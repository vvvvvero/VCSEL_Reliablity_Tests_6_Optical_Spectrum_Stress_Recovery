"""
29_mechanism_attribution.py
===========================
How much of each observed degradation does each physical mechanism account
for? Decomposes every prediction into per-latent contributions.

This question could not be asked before. While the decoder had effective rank
1.06 and zG/zB/zM were mutually substitutable to within 1.7 %, any attribution
would have been an arbitrary split of one shared direction. With the observation
set extended and the mask giving each latent its own signature (rank 2.33,
substitutability 54-98 %), the split is now meaningful.

The decomposition is exact, not a heuristic
-------------------------------------------
The decoder is a sparse LINEAR map, so for every feature f

    x_f(t) - x_f(0) = SUM_k  W[f,k] * ( z_k(t) - z_k(0) )

is an identity. Each term is that mechanism's contribution in the feature's
own units; shares below are |term| / SUM_k |term|, so they answer "what
fraction of the movement does this mechanism explain".

Absolute (signed) contributions are reported alongside the shares, because
two mechanisms can push a feature in opposite directions and cancel -- a case
the shares alone would hide.

What is a finding and what is an assumption
-------------------------------------------
A share is only evidence where the mask left a CHOICE. DIBL reads 1.000 on zB
because zB is the only latent its row admits; that is the prior, restated.
Rows where several latents compete -- the four main-channel features, and the
curve features with two or more admitted latents -- are where the model
actually decided something.

Every row is therefore labelled `forced` (one latent admitted, share is
structural) or `free` (several admitted, share is fitted). Only `free` rows
support a claim about which mechanism dominates.

Robustness
----------
A single fit can split a contested row arbitrarily. With --checkpoints given
several times the same decomposition is run per checkpoint and the spread of
each share is reported, so a share quoted in a write-up can be shown to be
stable rather than a one-run artefact.

Usage
-----
    python 29_mechanism_attribution.py
    python 29_mechanism_attribution.py --checkpoint A.pt --checkpoint B.pt
    python 29_mechanism_attribution.py --by-temperature --at-hours 2000

Reproducibility across independent backbones
--------------------------------------------
The warning this script prints -- that a single checkpoint's split should be
repeated before being quoted -- has been acted on with FOUR independent Stage
1-3 fits on the same filtered dataset (ext11_filtered unseeded, plus seeds
101/202/303). The backbones agree closely: Stage 1 val loss 0.0725-0.0729,
Stage 2 0.1056-0.1067, future rollout MSE 0.02587 / 0.02661 / 0.02609 /
0.02785.

  feature       dominant   share (mean +- sd)   range
  Vth           zM           0.396 +- 0.019     0.374-0.413
  IDSS          zM           0.351 +- 0.008     0.339-0.358
  RON           zM           0.378 +- 0.011     0.369-0.391
  gmmax         zM           0.412 +- 0.029     0.376-0.438
  IDLeak        zL           0.787 +- 0.015     0.774-0.808
  IGLeak        zL           0.940 +- 0.015     0.930-0.962
  SS_lin        zF           0.707 +- 0.044     0.678-0.771
  SS_sat        zB           0.715 +- 0.032     0.683-0.759
  gm_fwhm_sat   zM           0.575 +- 0.035     0.538-0.621
  V_gmpeak_sat  zM           0.765 +- 0.034     0.735-0.808

  dominant mechanism agrees on 10/10 free rows across all four fits
  share sd across backbones: mean 0.0194, median 0.0159, max 0.0436 (n=28)

Full table, mean +- sd over the four backbones:

  feature              zG            zB            zF            zM            zL            zC
  Vth           0.123±0.016   0.345±0.018             -   0.396±0.019             -   0.137±0.007
  IDSS          0.101±0.007   0.288±0.006             -   0.351±0.008             -   0.260±0.005
  RON           0.121±0.013   0.350±0.009             -   0.378±0.011             -   0.151±0.005
  gmmax         0.124±0.018   0.339±0.020             -   0.412±0.029             -   0.125±0.005
  IDLeak                  -   0.213±0.015             -             -   0.787±0.015             -
  IGLeak        0.060±0.015             -             -             -   0.940±0.015             -
  SS_lin        0.293±0.044             -   0.707±0.044             -             -             -
  SS_sat        0.285±0.032   0.715±0.032             -             -             -             -
  gm_fwhm_sat             -             -   0.425±0.035   0.575±0.035             -             -
  V_gmpeak_sat  0.235±0.034             -             -   0.765±0.034             -             -

So the assignment is a property of the data and the physics prior, not of one
optimisation run. Shares are quotable at about +-0.02 (1 sd), and the dominant
mechanism per feature is stable across every fit.

Note what this does and does not establish. It rules out run-to-run
optimisation noise, which was the standing objection. It does NOT show the
shares are physically correct: four fits of the SAME model under the SAME
sparsity mask and sign constraints can agree and still be wrong together.
Validating the split against physics needs independent evidence -- e.g.
comparing the fitted activation energies with literature values for the
mechanisms each latent is supposed to represent.
"""

import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# Contributions below this share are treated as absent rather than printed as
# tiny non-zero noise from a masked-out weight.
SHARE_EPS = 5e-4


def _load(alias: str, fname: str):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def latent_trajectories(model, mods, dataset, idx, device, prefix_len):
    """Integrated latent paths and temperatures for the given devices."""
    from torch.utils.data import DataLoader
    train_mod, eval_mod = mods["train"], mods["eval"]
    dl = DataLoader(train_mod.DeviceDegradationDataset(dataset, idx),
                    batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=train_mod.collate_fn)
    Z, TK, TT = [], [], []
    with torch.no_grad():
        for b in dl:
            out = eval_mod.predict_from_prefix(
                model, b["enc_input"], b["x"], b["mask"], b["times_h"],
                b["T_K"], b["x0"], prefix_len, device)
            Z.append(out["z_ode"].cpu().numpy())
            TK.append(b["T_K"].cpu().numpy())
            TT.append(b["times_h"].cpu().numpy())
    return np.concatenate(Z), np.concatenate(TK), np.concatenate(TT)


def decompose(W: np.ndarray, Z: np.ndarray, t_index: int):
    """Per-latent contribution to each feature at one timepoint.

    Returns signed contributions (N, F, K). Referenced to z(0) so the result
    is the contribution to the CHANGE, which is what the degradation features
    measure.
    """
    dz = Z[:, t_index, :] - Z[:, 0, :]            # (N, K)
    return dz[:, None, :] * W[None, :, :]         # (N, F, K)


def summarise(contrib: np.ndarray) -> Dict[str, np.ndarray]:
    """Shares and signed means over devices."""
    absc = np.abs(contrib)
    tot = absc.sum(axis=2, keepdims=True)
    share = np.divide(absc, np.maximum(tot, 1e-12))
    return {"share_mean": share.mean(axis=0),        # (F, K)
            "share_std": share.std(axis=0),
            "signed_mean": contrib.mean(axis=0),
            "abs_total_mean": np.abs(contrib).sum(axis=2).mean(axis=0)}


def row_kind(mask_row: np.ndarray) -> str:
    """`forced` when the mask admits one latent, `free` when several compete."""
    return "forced" if int(mask_row.sum()) <= 1 else "free"


def main():
    ap = argparse.ArgumentParser(description="Per-mechanism attribution")
    ap.add_argument("--checkpoint", action="append", default=None,
                    help="Stage 3 checkpoint; repeat for a robustness check")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--at-hours", type=float, default=None,
                    help="timepoint to decompose at (default: the last)")
    ap.add_argument("--by-temperature", action="store_true")
    ap.add_argument("--split", type=str, default="all",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--output", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "mechanism_attribution.json"))
    args = ap.parse_args()

    ds_path = args.dataset or (
        os.path.join(cfg.OUTPUT_PATH, "processed_data_ext_filtered.pkl")
        if os.path.exists(os.path.join(cfg.OUTPUT_PATH,
                                       "processed_data_ext_filtered.pkl"))
        else os.path.join(cfg.OUTPUT_PATH, "processed_data_ext.pkl"))
    cfg.PROCESSED_DATA_PATH = ds_path

    ckpts = args.checkpoint or [os.path.join(cfg.OUTPUT_PATH, "ext11_filtered",
                                             "checkpoints", "stage3_best.pt")]
    ckpts = [c for c in ckpts if os.path.exists(c)]
    if not ckpts:
        log.error("no checkpoint found; pass --checkpoint")
        return

    s4b = _load("_at_s4b", "14_stage4b_ar1_guided_generator.py")
    s4a = s4b.stage4a_mod
    mods = s4a._load_all()
    device = torch.device("cpu")

    with open(ds_path, "rb") as f:
        dataset = pickle.load(f)
    idx = (list(range(len(dataset["device_ids"]))) if args.split == "all"
           else dataset["split"][args.split])
    log.info("dataset %s  (%d devices, split=%s -> %d)",
             os.path.basename(ds_path), len(dataset["device_ids"]),
             args.split, len(idx))

    LN, FN = cfg.LATENT_NAMES, cfg.FEATURES
    mask = np.array(cfg.DECODER_SPARSITY)
    kinds = [row_kind(mask[f]) for f in range(len(FN))]

    per_ckpt = []
    for c in ckpts:
        model = s4a._build_model(mods)
        ck = torch.load(c, map_location="cpu")
        ms = ck.get("model_state", ck.get("model_state_dict"))
        model.load_state_dict({k: v for k, v in ms.items() if k != "decoder.mask"},
                              strict=False)
        model.eval()
        W = model.decoder.get_weight_matrix()
        Z, TK, TT = latent_trajectories(model, mods, dataset, idx, device,
                                        cfg.STAGE3_PREFIX_LEN)
        if args.at_hours is None:
            ti = Z.shape[1] - 1
        else:
            ti = int(np.argmin(np.abs(np.asarray(cfg.TIME_POINTS_H) - args.at_hours)))
        contrib = decompose(W, Z, ti)
        per_ckpt.append({"checkpoint": c, "t_index": ti,
                         "hours": float(cfg.TIME_POINTS_H[ti]),
                         "summary": summarise(contrib),
                         "contrib": contrib, "TK": TK})
        log.info("decomposed %s at t=%g h", os.path.basename(c),
                 cfg.TIME_POINTS_H[ti])

    ref = per_ckpt[0]
    S = ref["summary"]["share_mean"]
    hours = ref["hours"]

    log.info("=" * 92)
    log.info("MECHANISM ATTRIBUTION AT t = %g h  (share of |contribution|)", hours)
    log.info("%-14s %-7s" % ("feature", "kind") + "".join(f"{n:>9}" for n in LN))
    for f, fn in enumerate(FN):
        cells = "".join(f"{S[f, k]:>9.3f}" if S[f, k] > SHARE_EPS else f"{'-':>9}"
                        for k in range(len(LN)))
        log.info("%-14s %-7s" % (fn, kinds[f]) + cells)
    log.info("")
    log.info("kind=forced: the mask admits ONE latent, so the share is the prior")
    log.info("restated, not a finding. kind=free: several latents compete and the")
    log.info("model chose the split -- only these support a claim.")

    log.info("")
    log.info("SIGNED CONTRIBUTIONS on the free rows -- do mechanisms cancel?")
    SM = ref["summary"]["signed_mean"]
    log.info("%-14s" % "feature" + "".join(f"{n:>10}" for n in LN) + f"{'net':>10}")
    for f, fn in enumerate(FN):
        if kinds[f] != "free":
            continue
        log.info("%-14s" % fn
                 + "".join(f"{SM[f, k]:>+10.4f}" if abs(SM[f, k]) > 1e-6 else f"{'-':>10}"
                           for k in range(len(LN)))
                 + f"{SM[f].sum():>+10.4f}")

    if len(per_ckpt) > 1:
        log.info("")
        log.info("ROBUSTNESS across %d checkpoints -- max share spread per free row",
                 len(per_ckpt))
        stack = np.stack([p["summary"]["share_mean"] for p in per_ckpt])  # (C,F,K)
        for f, fn in enumerate(FN):
            if kinds[f] != "free":
                continue
            sp = (stack[:, f, :].max(axis=0) - stack[:, f, :].min(axis=0))
            worst = int(np.argmax(sp))
            log.info("  %-14s max spread %.3f on %s", fn, sp[worst], LN[worst])
        log.info("  A share that moves little across checkpoints can be quoted;")
        log.info("  one that swings is a property of the fit, not of the physics.")
    else:
        log.info("")
        log.info("Single checkpoint: shares on free rows are one fit's split and")
        log.info("should be repeated with --checkpoint before being quoted.")

    if args.by_temperature:
        log.info("")
        log.info("BY TEMPERATURE -- does the mechanism mix shift with stress?")
        contrib, TK = ref["contrib"], ref["TK"]
        temps = sorted(set(np.round(TK)))
        for f, fn in enumerate(FN):
            if kinds[f] != "free":
                continue
            log.info("  %s", fn)
            for T in temps:
                m = np.round(TK) == T
                sub = np.abs(contrib[m][:, f, :])
                sh = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1e-12)
                log.info("     %3.0f C (n=%3d)  " % (T - 273.15, m.sum())
                         + "".join(f"{LN[k]}={sh[:, k].mean():.3f}  "
                                   for k in range(len(LN))
                                   if sh[:, k].mean() > SHARE_EPS))

    out = {
        "dataset": ds_path, "hours": hours, "split": args.split,
        "latents": LN, "features": FN,
        "row_kind": {FN[f]: kinds[f] for f in range(len(FN))},
        "share_mean": {FN[f]: {LN[k]: float(S[f, k]) for k in range(len(LN))}
                       for f in range(len(FN))},
        "share_std_across_devices": {
            FN[f]: {LN[k]: float(ref["summary"]["share_std"][f, k])
                    for k in range(len(LN))} for f in range(len(FN))},
        "signed_mean": {FN[f]: {LN[k]: float(SM[f, k]) for k in range(len(LN))}
                        for f in range(len(FN))},
        "checkpoints": [p["checkpoint"] for p in per_ckpt],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info("")
    log.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
