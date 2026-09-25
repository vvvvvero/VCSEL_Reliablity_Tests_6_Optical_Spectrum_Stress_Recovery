"""
40_identifiability_analysis.py
==============================
What would it take to establish UNIQUE identification of the degradation
mechanisms, and why the present design cannot?

The distinction this script is about
------------------------------------
Three separate claims keep being conflated:

  reproducible    independent fits agree            -- established (36_)
  constrained     the data determine the parameter  -- true for Ea_irrev (39_)
  identified      no other mechanism assignment
                  explains the data equally well    -- NOT established

The third is what "unique identification" means, and nothing in this
repository currently tests it. Reproducibility across seeds says the optimiser
lands in the same place; it says nothing about whether a different assignment
of latents to physical mechanisms would fit just as well.

Three structural obstacles, measured
------------------------------------
1. TEMPERATURE CANNOT SEPARATE THE REVERSIBLE CHANNELS.
   zG, zB and zF share a single Ea_rev. Their temperature dependence is
   therefore IDENTICAL by construction, and no number of stress temperatures
   can distinguish them -- only the rev/irrev split. Adding temperatures helps
   identify Ea, not the mechanism assignment.

2. RATE AND ACTIVATION ENERGY ARE NEAR-COLLINEAR ON THREE POINTS.
   The design matrix (1/T - 1/Tref) takes values {+7.96e-5, 0, -7.29e-5}: two
   non-zero points, enough for one slope. Doubling Ea from 0.3 to 0.6 eV can
   be compensated by scaling k between 1.32x and 0.78x across the three
   temperatures -- imperfectly, which is why Ea_irrev is weakly identified,
   but well enough that a weak channel's Ea (Ea_rev) is not.

3. THE TIME CONSTANTS OVERLAP.
   Fitted tau = 1/k_c across four backbones:
       zG  4.31 h  (CV 11 %)
       zB  6.32 h  (CV 52 %)
       zF  1.02 h  (CV  5 %)
   zG and zB differ by a factor of 1.5 and zB's own spread is 52 % -- they are
   not separated by the time grid. zF is distinct. So the observation schedule
   resolves fast-versus-slow but not gate-versus-buffer.

What identification would actually require
------------------------------------------
Each obstacle needs a different experiment, and none is a bigger version of
what was already run.

  A. MECHANISM-SPECIFIC ACTIVATION ENERGIES + more temperatures.
     Give zG, zB and zF their own Ea. This is a MODEL change, and it only
     becomes estimable with 5-6 stress temperatures; at three it would make
     the fit less identified, not more. Test first with a profile likelihood
     on synthetic data before committing furnace time.

  B. RECOVERY (STRESS-MEASURE-RECOVER) CYCLES.
     The single most informative addition. Reversible channels predict
     relaxation once stress is removed, with mechanism-specific time
     constants; irreversible ones predict none. A recovery experiment
     separates rev from irrev DIRECTLY rather than through a shared Arrhenius
     factor, and gives zF -- whose tau is 1 h -- a signal the 0-2000 h
     schedule barely samples.

  C. BIAS-DEPENDENT STRESS.
     Gate-bias stress should load zG; drain-bias stress should load zM. This
     is the only proposal here that tests the ASSIGNMENT of latents to named
     mechanisms rather than their parameters, which is what "identification of
     the mechanism" means. Predict the shift before running it, then check.

  D. A PERMUTATION / SWAP TEST, which needs no new data.
     Exchange two latents' decoder columns and see whether the loss notices.
     Free, and it bounds how much identification the present data could
     possibly support.

RESULT OF (D), and it is the sharpest finding here
--------------------------------------------------
Swapping the zG and zB columns naively costs +36.12 % -- which looks like
strong evidence that the two are distinguishable, and is not. Their sparsity
columns differ on 5 of 11 rows (IDLeak, IGLeak, SS_lin, DIBL, V_gmpeak_sat),
so the swap moves weight onto positions the mask forbids. That penalty
measures the structure we imposed, not the mechanism the model learned.

Restricting the swap to the 6 rows where both columns carry the SAME mask
pattern:

    masked swap  ->  -0.44 %

Below the 2.7 % noise floor, and negative. On every observable where the mask
lets gate traps and buffer traps act alike, THE DATA CANNOT TELL THEM APART.
What separates zG from zB in the fitted model is the sparsity pattern written
into 04_model_decoder.py by hand, not anything measured.

This bounds the whole attribution claim. The 10/10 reproducible dominant
mechanisms of 36_ are reproducible partly because the mask makes them so. The
shares are real coefficient ratios, but the ASSIGNMENT of a column to "gate
trap" versus "buffer trap" rests on the prior, and this dataset does not
test it.

This script performs (D) and reports the structural analysis behind (A)-(C).

Usage
-----
    python 40_identifiability_analysis.py
    python 40_identifiability_analysis.py --swap-test
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

KB = 8.617333262e-5
BACKBONES = ["ext11_filtered", "seed101", "seed202", "seed303"]


def structural_analysis(temps_c=(275.0, 300.0, 325.0), t_ref_c=300.0):
    """The three obstacles, computed rather than asserted."""
    T = np.array(temps_c) + 273.15
    x = 1.0 / T - 1.0 / (t_ref_c + 273.15)

    log.info("OBSTACLE 1 -- temperature cannot separate the reversible channels")
    log.info("  zG, zB, zF share one Ea_rev, so their temperature dependence is")
    log.info("  identical by construction. More temperatures identify Ea; they")
    log.info("  cannot distinguish which reversible mechanism is acting.")

    log.info("")
    log.info("OBSTACLE 2 -- rate and activation energy are near-collinear")
    log.info("  design points (1/T - 1/Tref) = %s", np.round(x, 7).tolist())
    log.info("  %d non-zero values -> one estimable slope", int((x != 0).sum()))
    for e1, e2 in [(0.3, 0.6), (0.6, 1.2)]:
        r = np.exp(-e2 / KB * x) / np.exp(-e1 / KB * x)
        log.info("  Ea %.1f -> %.1f eV compensated by k x %.3f..%.3f",
                 e1, e2, 1 / r.max(), 1 / r.min())
    return {"design_points": x.tolist(),
            "n_nonzero": int((x != 0).sum())}


def fitted_time_constants():
    """tau = 1/k_c per reversible mechanism, across backbones."""
    out = {}
    for m in ("G", "B", "F"):
        taus = []
        for b in BACKBONES:
            p = os.path.join(cfg.OUTPUT_PATH, b, "checkpoints", "stage3_best.pt")
            if not os.path.exists(p):
                continue
            sd = torch.load(p, map_location="cpu", weights_only=False)["model_state"]
            raw = sd.get(f"ode.k{m}c_raw")
            if raw is None:
                continue
            k = math.exp(float(raw))
            if k > 0:
                taus.append(1.0 / k)
        if taus:
            a = np.array(taus)
            out[f"z{m}"] = {"tau_h": a.tolist(), "mean": float(a.mean()),
                            "cv_pct": float(100 * a.std(ddof=1) / a.mean())
                            if len(a) > 1 else float("nan")}
    log.info("")
    log.info("OBSTACLE 3 -- the time constants overlap")
    log.info("  %-6s %10s %10s", "latent", "tau [h]", "CV")
    for k, v in out.items():
        log.info("  %-6s %10.2f %9.1f %%", k, v["mean"], v["cv_pct"])
    if "zG" in out and "zB" in out:
        ratio = max(out["zG"]["mean"], out["zB"]["mean"]) / \
                min(out["zG"]["mean"], out["zB"]["mean"])
        log.info("  zG and zB differ by a factor of %.2f, and zB's own spread is", ratio)
        log.info("  %.0f %% -- the observation grid does not separate them.",
                 out["zB"]["cv_pct"])
    return out


def swap_test(dataset_path, backbone, pairs=(("zG", "zB"),)):
    """Exchange two latents' decoder columns and re-measure the rollout loss.

    IMPORTANT -- a naive swap measures the MASK, not the mechanism. zG and zB
    have different sparsity columns (they differ on IDLeak, IGLeak, SS_lin,
    DIBL and V_gmpeak_sat), so exchanging them moves weight onto positions the
    mask forbids. That alone costs +36 % and says nothing about whether the
    two mechanisms are distinguishable.

    The test is therefore run BOTH ways:

      raw       swap every row -- dominated by the mask mismatch, reported
                only to show how large that artefact is
      masked    swap only the rows where the two columns have the SAME mask
                pattern, which is the comparison that isolates the fitted
                coefficients from the imposed structure

    Either way this is a necessary condition, not a sufficient one: a swap
    that costs something shows the columns are distinguishable, not that
    either matches the physics it is named after.
    """
    import importlib.util

    def L(alias, fn):
        spec = importlib.util.spec_from_file_location(alias, os.path.join(BASE_DIR, fn))
        m = importlib.util.module_from_spec(spec)
        sys.modules[alias] = m
        spec.loader.exec_module(m)
        return m

    cfg.PROCESSED_DATA_PATH = dataset_path
    abl = L("_id_abl", "16_ablation_physics_condition.py")
    mods = abl._load_all()
    s4a = sys.modules["_pi_stage4a_impl"]
    model = s4a._build_model(mods)
    ck = torch.load(os.path.join(backbone, "checkpoints", "stage3_best.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    import pickle
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)
    from torch.utils.data import DataLoader
    tm = mods["train"]
    dl = DataLoader(tm.DeviceDegradationDataset(ds, ds["split"]["val"]),
                    batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=tm.collate_fn)

    def loss():
        return float(tm.validation_prefix_rollout_mse(
            model, dl, "cpu", prefix_len=cfg.STAGE3_PREFIX_LEN))

    base = loss()
    log.info("")
    log.info("SWAP TEST -- are the latent labels interchangeable?")
    log.info("  baseline val rollout MSE = %.6f", base)
    names = list(cfg.LATENT_NAMES)
    W = model.decoder.W_raw
    M = model.decoder.mask
    M = M.detach().cpu().numpy() if hasattr(M, "detach") else np.asarray(M)
    results = {}
    for a, b in pairs:
        ia, ib = names.index(a), names.index(b)
        shared = np.where(M[:, ia] == M[:, ib])[0]
        n_diff = M.shape[0] - len(shared)
        log.info("  %s vs %s: mask columns differ on %d of %d rows",
                 a, b, n_diff, M.shape[0])

        # raw swap -- dominated by the mask mismatch
        with torch.no_grad():
            W[:, [ia, ib]] = W[:, [ib, ia]]
        raw = loss()
        with torch.no_grad():
            W[:, [ia, ib]] = W[:, [ib, ia]]

        # masked swap -- only rows where the mask treats both columns alike
        with torch.no_grad():
            tmp = W[shared, ia].clone()
            W[shared, ia] = W[shared, ib]
            W[shared, ib] = tmp
        masked = loss()
        with torch.no_grad():
            tmp = W[shared, ia].clone()
            W[shared, ia] = W[shared, ib]
            W[shared, ib] = tmp

        r_raw = 100 * (raw - base) / base
        r_msk = 100 * (masked - base) / base
        results[f"{a}<->{b}"] = {
            "baseline": base, "raw_swapped": raw, "raw_rel_pct": r_raw,
            "masked_swapped": masked, "masked_rel_pct": r_msk,
            "n_rows_shared_mask": int(len(shared)),
            "n_rows_mask_differs": int(n_diff)}
        log.info("    raw swap    MSE %.6f (%+.2f %%)  <- includes the mask artefact",
                 raw, r_raw)
        log.info("    masked swap MSE %.6f (%+.2f %%) on %d shared rows -> %s",
                 masked, r_msk, len(shared),
                 "DISTINGUISHABLE" if abs(r_msk) > 2.7 else
                 "INTERCHANGEABLE at the noise floor")
    log.info("")
    log.info("  Read the MASKED row. The raw swap moves weight onto positions the")
    log.info("  sparsity mask forbids, so it measures the imposed structure rather")
    log.info("  than the fitted mechanism. If the masked swap costs less than the")
    log.info("  2.7 %% noise floor, the two columns are exchangeable as far as this")
    log.info("  data can tell and their physical labels are a convention.")
    return results


def main():
    ap = argparse.ArgumentParser(description="Identifiability analysis")
    ap.add_argument("--dataset", default=os.path.join(
        cfg.OUTPUT_PATH, "processed_data_ext_filtered.pkl"))
    ap.add_argument("--backbone", default=os.path.join(cfg.OUTPUT_PATH, "ext11_filtered"))
    ap.add_argument("--swap-test", action="store_true",
                    help="run the decoder-column swap test (needs the backbone)")
    ap.add_argument("--output", default=os.path.join(
        cfg.RESULTS_DIR, "identifiability_analysis.json"))
    args = ap.parse_args()

    log.info("=" * 78)
    log.info("WHAT WOULD ESTABLISH UNIQUE MECHANISM IDENTIFICATION")
    log.info("=" * 78)
    log.info("reproducible : independent fits agree              -- established")
    log.info("constrained  : the data determine the parameter    -- Ea_irrev only")
    log.info("identified   : no other assignment fits as well    -- NOT established")
    log.info("")

    struct = structural_analysis()
    taus = fitted_time_constants()

    swaps = None
    if args.swap_test:
        swaps = swap_test(args.dataset, args.backbone)

    log.info("")
    log.info("=" * 78)
    log.info("WHAT EACH PROPOSED EXPERIMENT CAN AND CANNOT SETTLE")
    log.info("  more temperatures  : identifies Ea; does NOT separate zG/zB/zF")
    log.info("                       while they share one Ea_rev")
    log.info("  per-mechanism Ea   : a MODEL change; needs 5-6 temperatures to be")
    log.info("                       estimable, and would worsen the fit at three")
    log.info("  recovery cycles    : separates rev from irrev DIRECTLY, and gives")
    log.info("                       zF (tau ~ 1 h) a signal this schedule misses")
    log.info("  bias-dependent     : the only one testing the ASSIGNMENT of")
    log.info("                       latents to named mechanisms")
    log.info("  swap test          : free, bounds what the present data support")

    out = {"structural": struct, "time_constants": taus, "swap_test": swaps}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("")
    log.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
