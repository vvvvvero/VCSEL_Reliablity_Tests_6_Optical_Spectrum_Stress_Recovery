"""
20_latent_degeneracy.py
=======================
Is the physics latent space actually identifiable, or are the latent states
interchangeable?

Motivation
----------
The A/B/C physics-conditioning ablation (16_) found no significant difference
between physics-conditioned, no-physics and shuffled-physics generators. The
usual reading of such a result is "the physics prior does not help". This
script tests a different explanation: that the physics channel carries no
per-device information in the first place, because the latent states are
structurally interchangeable.

The decoder is a sparse CONSTRAINED LINEAR map (04_model_decoder.py), not a
free MLP -- it already has a sparsity mask, sign constraints and relative
decoding. But the mask makes the four main-channel rows identical:

    Vth, IDSS, RON, gmmax  ->  [zG, zB, zM, --, zC]   (same pattern)
    IDLeak                 ->  [--, zB, --, zL, --]
    IGLeak                 ->  [zG, --, --, zL, --]

so nothing in the architecture distinguishes zG/zB/zM/zC from each other on
the four features that carry most of the signal. They differ only by learned
scalar weights, and the optimiser is free to trade one against another.

What this script measures
-------------------------
1. The effective decoder weight matrix W (6x5), after mask + sign constraints.
2. Pairwise cosine similarity of the four main-channel rows. Cosine -> 1 means
   the rows are proportional, i.e. the features respond to one shared latent
   direction rather than to distinct mechanisms.
3. SVD of the 4x4 main block: singular values, the fraction of Frobenius
   energy in sigma1, and the participation-ratio effective rank
   (1 = fully degenerate, 4 = four independent mechanisms).
4. Substitutability: for each latent j, how well W[:, j] is reproduced by a
   least-squares combination of the other columns. A small residual means
   that latent's effect on the observables can be mimicked by the others, so
   the optimiser has no reason to prefer any particular assignment.
5. Optionally (--z-spread, needs a dataset) the per-device spread of the
   integrated latent trajectories, split into a within-(T,t) part and a part
   explained by temperature alone. This connects the structural degeneracy to
   the observed collapse of z_phys onto a function of (T, t).

Interpreting the output
-----------------------
Degeneracy is a statement about the OBSERVATION MODEL, not about the ODE. If
the effective rank is ~1, then no matter how rich the ODE is, the six scalar
extracted parameters cannot tell the mechanisms apart, and a per-device
degree of freedom (alpha) has nothing to latch onto. The fix is to add
observables with mechanism-specific signatures (subthreshold swing, gm shape,
knee voltage), not to add more latent states.

Usage
-----
    python 20_latent_degeneracy.py
    python 20_latent_degeneracy.py --checkpoint <a.pt> --checkpoint <b.pt>
    python 20_latent_degeneracy.py --z-spread --split test
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

# Taken from config, not hard-coded: this file was written for the 5-latent
# model and crashed with an IndexError once zF was added, because the decoder
# had six columns and this list still had five names.
LATENT_NAMES = list(cfg.LATENT_NAMES)

# Rows carrying the main-channel signal. IDLeak/IGLeak are excluded: they have
# a different (leakage) mask, and in this dataset they are zero-variance and
# not generated, so including them would dilute the measurement.
MAIN_ROWS = [cfg.FEATURES.index(f) for f in ("Vth", "IDSS", "RON", "gmmax")
             if f in cfg.FEATURES]

# Substitutability residual, as a fraction of the column norm, below which a
# latent's effect is considered reproducible by the others.
SUBST_FULL = 0.10
SUBST_PARTIAL = 0.40


def _load(alias: str, fname: str):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def effective_weight(ckpt_path: str, dec_mod) -> Optional[np.ndarray]:
    """Rebuild the decoder from a checkpoint and return its effective W (6x5).

    Uses the decoder's own _effective_weight() so the mask and the sign
    constraints (softplus on the non-free rows) are applied exactly as they
    are during training -- reading W_raw directly would give the wrong matrix.
    """
    if not os.path.exists(ckpt_path):
        log.warning("checkpoint not found: %s", ckpt_path)
        return None
    ck = torch.load(ckpt_path, map_location="cpu")
    ms = ck.get("model_state", ck.get("model_state_dict"))
    if ms is None:
        log.warning("no model state in %s", ckpt_path)
        return None
    dec = dec_mod.SparsePhysicsDecoder()
    sd = {k.replace("decoder.", ""): v for k, v in ms.items() if k.startswith("decoder.")}
    if not sd:
        log.warning("no decoder.* keys in %s", ckpt_path)
        return None
    dec.load_state_dict(sd, strict=False)
    return dec.get_weight_matrix()


def row_similarity(W: np.ndarray, rows: List[int], cols: List[int]) -> Dict:
    """Cosine similarity between decoder rows, restricted to `cols`.

    Rows are normalised first, so this compares the DIRECTION each feature
    reads out of latent space, independent of that feature's overall scale.
    """
    R = W[rows][:, cols]
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    Rn = R / np.maximum(norms, 1e-12)
    C = Rn @ Rn.T
    off = [float(C[i, j]) for i in range(len(rows)) for j in range(len(rows)) if i < j]
    return {
        "cosine_matrix": C.tolist(),
        "row_norms": norms.ravel().tolist(),
        "normalised_rows": Rn.tolist(),
        "mean_offdiag_cosine": float(np.mean(off)) if off else float("nan"),
        "min_offdiag_cosine": float(np.min(off)) if off else float("nan"),
    }


def block_svd(W: np.ndarray, rows: List[int], cols: List[int]) -> Dict:
    """SVD of the sub-block, with two readouts of how degenerate it is.

    - energy_frac_sigma1: share of Frobenius energy in the leading direction.
    - effective_rank: participation ratio (sum s^2)^2 / sum s^4. This is a
      continuous rank measure: 1.0 means one direction does everything,
      len(cols) means all directions contribute equally. Preferred over
      counting non-zero singular values, which is all-or-nothing.
    """
    R = W[rows][:, cols]
    s = np.linalg.svd(R, compute_uv=False)
    U, sv, Vt = np.linalg.svd(R)
    s2 = s ** 2
    return {
        "singular_values": s.tolist(),
        "energy_frac_sigma1": float(s2[0] / np.sum(s2)) if np.sum(s2) > 0 else float("nan"),
        "effective_rank": float(np.sum(s2) ** 2 / np.sum(s2 ** 2)) if np.sum(s2) > 0 else float("nan"),
        "condition_number": float(s[0] / max(s[-1], 1e-12)),
        "v1": Vt[0].tolist(),
        "u1": U[:, 0].tolist(),
    }


def substitutability(W: np.ndarray, rows: List[int], cols: List[int]) -> Dict:
    """For each latent in `cols`, how well can the others reproduce its column?

    Solves  min_c || W[rows, j] - W[rows, others] c ||  and reports the
    residual as a fraction of the column norm. A near-zero residual means the
    optimiser can move effect freely between latents, which is exactly the
    condition under which a per-device multiplier on one latent (alpha) has no
    identifiable role.
    """
    out = {}
    for j in cols:
        others = [k for k in cols if k != j]
        A = W[rows][:, others]
        b = W[rows][:, j]
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = float(np.linalg.norm(b - A @ c))
        nrm = float(np.linalg.norm(b))
        frac = resid / nrm if nrm > 0 else float("nan")
        out[LATENT_NAMES[j]] = {
            "residual": resid,
            "column_norm": nrm,
            "residual_frac": frac,
            "verdict": ("fully substitutable" if frac < SUBST_FULL
                        else "partly substitutable" if frac < SUBST_PARTIAL
                        else "distinct"),
        }
    return out


def z_spread(ckpt_path: str, split: str) -> Optional[Dict]:
    """Per-device spread of the integrated latent trajectories.

    Splits the spread of each latent into the part that survives WITHIN a
    fixed (temperature, timestep) cell -- i.e. genuine device-to-device
    variation -- and the part explained by temperature alone. If the
    within-cell part is ~0, every device at a given temperature follows the
    same latent trajectory, so conditioning a generator on z_phys is
    equivalent to conditioning it on (T, t).
    """
    s4b = _load("_deg_s4b", "14_stage4b_ar1_guided_generator.py")
    s4a = s4b.stage4a_mod
    mods = s4a._load_all()
    train_mod, eval_mod = mods["train"], mods["eval"]

    model = s4a._build_model(mods)
    ck = torch.load(ckpt_path, map_location="cpu")
    ms = ck.get("model_state", ck.get("model_state_dict"))
    model.load_state_dict({k: v for k, v in ms.items() if k != "decoder.mask"}, strict=False)
    model.eval()

    with open(cfg.PROCESSED_DATA_PATH, "rb") as f:
        dataset = pickle.load(f)
    idx = (list(range(len(dataset["device_ids"]))) if split == "all"
           else dataset["split"][split])

    from torch.utils.data import DataLoader
    dl = DataLoader(train_mod.DeviceDegradationDataset(dataset, idx),
                    batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=train_mod.collate_fn)

    W = cfg.STAGE3_PREFIX_LEN
    Z, TT = [], []
    with torch.no_grad():
        for b in dl:
            out = eval_mod.predict_from_prefix(
                model, b["enc_input"], b["x"], b["mask"], b["times_h"],
                b["T_K"], b["x0"], W, torch.device("cpu"))
            Z.append(out["z_ode"].cpu().numpy())
            TT.append(b["T_K"].cpu().numpy())
    Z = np.concatenate(Z)     # (N, T, 5)
    TT = np.concatenate(TT)   # (N,)

    res = {"n_devices": int(Z.shape[0]), "split": split, "per_latent": {}}
    temps = np.unique(np.round(TT))
    for si, nm in enumerate(LATENT_NAMES):
        within = []
        for T in temps:
            m = np.round(TT) == T
            for t in range(Z.shape[1]):
                within.append(Z[m][:, t, si].std())
        overall = float(Z[:, :, si].std())
        wmean = float(np.mean(within))
        final = Z[:, -1, si]
        tot_var = float(final.var())
        within_var = float(np.mean([final[np.round(TT) == T].var() for T in temps]))
        res["per_latent"][nm] = {
            "overall_std": overall,
            "mean_within_T_t_std": wmean,
            "device_specific_frac": (wmean / overall) if overall > 0 else float("nan"),
            "final_std_across_devices": float(final.std()),
            "r2_temperature": (1.0 - within_var / tot_var) if tot_var > 0 else float("nan"),
        }
    return res


def report(tag: str, W: np.ndarray, cols_main: List[int]) -> Dict:
    FN = cfg.FEATURES
    log.info("=" * 78)
    log.info("%s -- effective decoder weight W (rows = features, cols = latents)", tag)
    log.info("%-9s" % "" + "".join(f"{c:>11}" for c in LATENT_NAMES))
    for i, f in enumerate(FN):
        log.info("%-9s" % f + "".join(f"{W[i, j]:>11.5f}" for j in range(W.shape[1])))

    sim = row_similarity(W, MAIN_ROWS, cols_main)
    log.info("")
    log.info("  Main-channel rows %s, over latents %s",
             [FN[r] for r in MAIN_ROWS], [LATENT_NAMES[c] for c in cols_main])
    log.info("  normalised rows (readout DIRECTION per feature):")
    for i, r in enumerate(MAIN_ROWS):
        log.info("    %-8s" % FN[r] + "".join(f"{v:>11.5f}" for v in sim["normalised_rows"][i]))
    log.info("  pairwise cosine similarity (1.000 => proportional => degenerate):")
    log.info("    %-8s" % "" + "".join(f"{FN[r]:>9}" for r in MAIN_ROWS))
    C = np.array(sim["cosine_matrix"])
    for i, r in enumerate(MAIN_ROWS):
        log.info("    %-8s" % FN[r] + "".join(f"{C[i, j]:>9.4f}" for j in range(len(MAIN_ROWS))))
    log.info("    mean off-diagonal = %.4f   min = %.4f",
             sim["mean_offdiag_cosine"], sim["min_offdiag_cosine"])

    svd = block_svd(W, MAIN_ROWS, cols_main)
    log.info("")
    log.info("  singular values: %s", np.round(np.array(svd["singular_values"]), 6).tolist())
    log.info("    sigma1 holds %.2f%% of Frobenius energy", 100 * svd["energy_frac_sigma1"])
    log.info("    effective rank = %.3f  (1 = degenerate, %d = independent)",
             svd["effective_rank"], len(cols_main))
    log.info("    condition number = %.3e", svd["condition_number"])
    log.info("    v1 over %s = %s", [LATENT_NAMES[c] for c in cols_main],
             np.round(np.array(svd["v1"]), 4).tolist())
    log.info("    u1 over %s = %s", [FN[r] for r in MAIN_ROWS],
             np.round(np.array(svd["u1"]), 4).tolist())

    sub_all = substitutability(W, list(range(W.shape[0])), list(range(W.shape[1])))
    sub_main = substitutability(W, MAIN_ROWS, cols_main)
    log.info("")
    log.info("  Substitutability -- can the OTHER latents reproduce this one?")
    log.info("    %-8s %12s %10s   %s", "latent", "residual", "frac", "verdict")
    log.info("    -- all 6 feature rows --")
    for nm, d in sub_all.items():
        log.info("    %-8s %12.5f %9.1f%%   %s", nm, d["residual"],
                 100 * d["residual_frac"], d["verdict"])
    log.info("    -- main-channel rows only --")
    for nm, d in sub_main.items():
        log.info("    %-8s %12.5f %9.1f%%   %s", nm, d["residual"],
                 100 * d["residual_frac"], d["verdict"])

    return {"tag": tag, "W": W.tolist(), "row_similarity": sim, "svd": svd,
            "substitutability_all_rows": sub_all,
            "substitutability_main_rows": sub_main}


def main():
    ap = argparse.ArgumentParser(description="Physics-latent degeneracy diagnostic")
    ap.add_argument("--checkpoint", action="append", default=None,
                    help="Checkpoint to analyse; repeat to compare several. "
                         "Default: <CHECKPOINT_DIR>/stage3_best.pt")
    ap.add_argument("--z-spread", action="store_true",
                    help="Also measure per-device spread of the integrated "
                         "latent trajectories (needs the processed dataset).")
    ap.add_argument("--split", type=str, default="all",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "latent_degeneracy"))
    args = ap.parse_args()

    ckpts = args.checkpoint or [os.path.join(cfg.CHECKPOINT_DIR, "stage3_best.pt")]
    os.makedirs(args.output_dir, exist_ok=True)

    dec_mod = _load("_deg_dec", "04_model_decoder.py")

    # Latents the main rows are allowed to see. Columns masked to zero for
    # every main row (zL) carry no direction there and would make the
    # normalisation and the SVD meaningless, so drop them.
    mask = np.array(cfg.DECODER_SPARSITY, dtype=float)
    cols_main = [j for j in range(mask.shape[1]) if mask[MAIN_ROWS, j].any()]
    log.info("main-channel latents: %s (zL excluded: masked out of every main row)",
             [LATENT_NAMES[c] for c in cols_main])

    results: List[Dict] = []
    for ck in ckpts:
        W = effective_weight(ck, dec_mod)
        if W is None:
            continue
        tag = os.path.basename(os.path.dirname(os.path.dirname(ck))) or os.path.basename(ck)
        r = report(f"{tag}  [{ck}]", W, cols_main)
        r["checkpoint"] = ck
        if args.z_spread:
            zs = z_spread(ck, args.split)
            if zs:
                r["z_spread"] = zs
                log.info("")
                log.info("  Per-device spread of z_phys (split=%s, n=%d)",
                         zs["split"], zs["n_devices"])
                log.info("    %-6s %12s %14s %12s %10s", "latent", "overall_std",
                         "within(T,t)", "dev_frac", "R2_temp")
                for nm, d in zs["per_latent"].items():
                    log.info("    %-6s %12.3e %14.3e %11.1f%% %10.4f",
                             nm, d["overall_std"], d["mean_within_T_t_std"],
                             100 * d["device_specific_frac"], d["r2_temperature"])
        results.append(r)

    if not results:
        log.error("no checkpoints could be analysed")
        return

    out = os.path.join(args.output_dir, "latent_degeneracy.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("=" * 78)
    log.info("SUMMARY")
    log.info("%-34s %14s %14s %12s", "checkpoint", "mean cosine", "eff. rank", "sigma1 %")
    for r in results:
        log.info("%-34s %14.4f %14.3f %11.2f%%",
                 os.path.basename(os.path.dirname(os.path.dirname(r["checkpoint"]))) or "?",
                 r["row_similarity"]["mean_offdiag_cosine"],
                 r["svd"]["effective_rank"],
                 100 * r["svd"]["energy_frac_sigma1"])
    log.info("")
    log.info("An effective rank near 1 means the main-channel features share a "
             "single readout direction: they measure one scalar amount of "
             "degradation rather than distinct mechanisms, so zG/zB/zM/zC are "
             "interchangeable and a per-device multiplier has nothing to bind to.")
    log.info("Saved -> %s", out)


if __name__ == "__main__":
    main()
