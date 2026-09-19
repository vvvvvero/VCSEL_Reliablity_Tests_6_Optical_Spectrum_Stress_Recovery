"""
39_activation_energy_validation.py
==================================
External physical validation: do the fitted activation energies agree with
published values for the mechanisms the latents are supposed to represent?

Why this is the debt to settle
------------------------------
29_ and 36_ establish that the mechanism attribution is REPRODUCIBLE -- four
independent fits agree on the dominant latent for 10/10 free rows. That rules
out optimisation noise and nothing more. Four fits of the same model under the
same sparsity mask can agree and be wrong together, so reproducibility is not
identification. The only route from one to the other is agreement with
evidence the model never saw.

Activation energy is the natural candidate. The ODE carries two shared
Arrhenius energies -- Ea_rev for the reversible trap channels and Ea_irrev for
the irreversible ones -- and both are dimensional, physical quantities with
decades of independent measurement behind them.

What this script does
---------------------
1. Reads Ea_rev and Ea_irrev from every available backbone.
2. Profiles the Stage 3 rollout loss over each, freezing the parameter at a
   grid and leaving everything else at its fitted value. A parameter the data
   constrains shows a minimum; one it cannot see shows a flat profile.
3. Compares the constrained energies against published ranges.

A measurement bug worth recording
---------------------------------
The first profile used 08_training._forward, and returned a PERFECTLY flat
loss -- identical to six decimals from Ea = 0.05 to 2.20 eV. That path is
encoder -> decoder reconstruction and never invokes the ODE, so no Arrhenius
parameter can affect it. The profile has to use
validation_prefix_rollout_mse, which is the Stage 3 selection metric and does
integrate the ODE forward from the prefix. A flat profile is exactly what a
broken probe looks like, which is why the fix mattered.

RESULT
------
The two energies behave completely differently, and only one is identified.

Ea_irrev -- CONSTRAINED
    Clear parabolic minimum at 0.595 eV (vertex of a quadratic fit).
    MSE degrades monotonically on both sides; 0.05 eV costs 1.1 %, 2.20 eV
    costs 10.0 %. The four backbones independently fit 0.6444-0.6987 eV
    (mean 0.667, sd 0.023), which sits right at the profile minimum. The
    interval within 0.2 % of the best MSE is roughly [0.40, 0.80] eV.

Ea_rev -- NOT CONSTRAINED
    Monotonically decreasing across the whole grid; the best value on a range
    spanning 0.05 to 2.20 eV is the endpoint, 2.20, and the total spread is
    0.83 % of the MSE. There is no minimum. Correspondingly the four backbones
    scatter over 0.2778-0.6587 eV -- a coefficient of variation of 38 %
    against 3.4 % for Ea_irrev. The fitted Ea_rev is an artefact of
    initialisation and the optimiser's path, not a measurement.

Literature comparison, for Ea_irrev only
----------------------------------------
Reported thermal-activation energies for irreversible degradation in GaN HEMTs
cluster in a broad band, commonly quoted between about 0.5 and 1.3 eV
depending on the mechanism and the stress condition; values near 0.6-0.7 eV
are frequently associated with buffer- and interface-related trap processes
under thermal storage. The fitted 0.60-0.67 eV falls inside that band.

State the strength of that agreement honestly: a broad literature band and a
loosely constrained fit overlapping is CONSISTENCY, not confirmation. It would
have been informative if the fit had landed at 0.1 or 2 eV, and it did not.
That is the weakest useful form of external validation, and it should be
reported as such rather than as physical identification.

What would actually identify the mechanisms
-------------------------------------------
Three routes, none of which this dataset can support:

  * more temperatures. Three points give two independent ratios and the model
    carries two energies plus per-mechanism rate constants, so Ea and k are
    near-collinear. Five or six stress temperatures would separate them.
  * recovery experiments. zF and the reversible channels predict relaxation
    after stress is removed; measuring it tests the reversible/irreversible
    split directly, which is what Ea_rev's unidentifiability currently blocks.
  * bias-dependent stress. Gate versus drain stress should shift the
    attribution between zG and zM in a direction the physics predicts.

Usage
-----
    python 39_activation_energy_validation.py
"""

import argparse
import json
import logging
import math
import os
import pickle
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

BACKBONES = ["ext11_filtered", "seed101", "seed202", "seed303"]
GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00, 1.30, 1.60, 2.20]

# Published ranges are broad and mechanism-dependent; this records the band the
# comparison is made against so a reader can substitute their own.
LITERATURE = {
    "Ea_irrev": {"low": 0.5, "high": 1.3,
                 "note": "irreversible thermal degradation in GaN HEMTs; "
                         "0.6-0.7 eV commonly associated with buffer/interface "
                         "trap processes under thermal storage"},
    "Ea_rev": {"low": 0.2, "high": 0.9,
               "note": "reversible trapping/detrapping; wide spread across "
                       "reports and stress conditions"},
}


def _load(alias, fn):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, os.path.join(BASE_DIR, fn))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description="External validation of fitted Ea")
    ap.add_argument("--dataset", default=os.path.join(
        cfg.OUTPUT_PATH, "processed_data_ext_filtered.pkl"))
    ap.add_argument("--backbone", default=os.path.join(cfg.OUTPUT_PATH, "ext11_filtered"))
    ap.add_argument("--output", default=os.path.join(
        cfg.RESULTS_DIR, "activation_energy_validation.json"))
    args = ap.parse_args()

    cfg.PROCESSED_DATA_PATH = args.dataset

    # ---- fitted values across every available backbone -------------------
    log.info("FITTED ACTIVATION ENERGIES")
    log.info("%-18s %14s %14s", "backbone", "Ea_rev [eV]", "Ea_irrev [eV]")
    fitted = {"Ea_rev": [], "Ea_irrev": []}
    for b in BACKBONES:
        p = os.path.join(cfg.OUTPUT_PATH, b, "checkpoints", "stage3_best.pt")
        if not os.path.exists(p):
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)["model_state"]
        r = math.exp(float(sd["ode.log_Ea_rev"]))
        i = math.exp(float(sd["ode.log_Ea_irrev"]))
        fitted["Ea_rev"].append(r)
        fitted["Ea_irrev"].append(i)
        log.info("%-18s %14.4f %14.4f", b, r, i)
    for k, v in fitted.items():
        a = np.array(v)
        log.info("  %s: mean %.4f  sd %.4f  CV %.1f %%  range %.4f-%.4f",
                 k, a.mean(), a.std(ddof=1), 100 * a.std(ddof=1) / a.mean(),
                 a.min(), a.max())

    # ---- profile likelihood ----------------------------------------------
    abl = _load("_ea_abl", "16_ablation_physics_condition.py")
    mods = abl._load_all()
    s4a = sys.modules["_pi_stage4a_impl"]
    model = s4a._build_model(mods)
    ck = torch.load(os.path.join(args.backbone, "checkpoints", "stage3_best.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    with open(cfg.PROCESSED_DATA_PATH, "rb") as f:
        ds = pickle.load(f)
    from torch.utils.data import DataLoader
    tm = mods["train"]
    dl = DataLoader(tm.DeviceDegradationDataset(ds, ds["split"]["val"]),
                    batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=tm.collate_fn)

    def loss():
        # MUST be the ODE rollout: the encoder->decoder path in tm._forward
        # never touches the Arrhenius parameters and profiles perfectly flat.
        return float(tm.validation_prefix_rollout_mse(
            model, dl, "cpu", prefix_len=cfg.STAGE3_PREFIX_LEN))

    profiles = {}
    for name, par in [("Ea_rev", model.ode.log_Ea_rev),
                      ("Ea_irrev", model.ode.log_Ea_irrev)]:
        base = float(par.detach())
        log.info("")
        log.info("PROFILE over %s (all else frozen at the fitted solution)", name)
        log.info("%9s %14s", "Ea [eV]", "val MSE")
        vals = []
        for ea in GRID:
            with torch.no_grad():
                par.fill_(math.log(ea))
            v = loss()
            vals.append(v)
            log.info("%9.2f %14.6f", ea, v)
        with torch.no_grad():
            par.fill_(base)
        vals = np.array(vals)
        g = np.array(GRID)
        best = float(g[vals.argmin()])
        spread = 100 * (vals.max() - vals.min()) / vals.min()
        interior = 0 < vals.argmin() < len(g) - 1
        profiles[name] = {"grid": GRID, "mse": vals.tolist(),
                          "argmin_eV": best, "spread_pct": spread,
                          "has_interior_minimum": bool(interior)}
        log.info("  minimum at %.2f eV | spread %.2f %% | interior minimum: %s",
                 best, spread, interior)
        if interior:
            c = np.polyfit(g, vals, 2)
            vtx = float(-c[1] / (2 * c[0]))
            profiles[name]["parabola_vertex_eV"] = vtx
            log.info("  parabolic vertex at %.3f eV", vtx)
        else:
            log.info("  NO interior minimum: this energy is NOT identified by")
            log.info("  the data, and its fitted value reflects initialisation.")

    # ---- verdict ----------------------------------------------------------
    log.info("")
    log.info("=" * 78)
    log.info("VERDICT")
    for name in ("Ea_irrev", "Ea_rev"):
        pr = profiles[name]
        lit = LITERATURE[name]
        arr = np.array(fitted[name])
        if pr["has_interior_minimum"]:
            v = pr.get("parabola_vertex_eV", pr["argmin_eV"])
            inside = lit["low"] <= v <= lit["high"]
            log.info("  %s: CONSTRAINED, minimum %.3f eV, backbones %.3f-%.3f",
                     name, v, arr.min(), arr.max())
            log.info("     literature band %.1f-%.1f eV -> %s",
                     lit["low"], lit["high"],
                     "CONSISTENT" if inside else "OUTSIDE THE BAND")
        else:
            log.info("  %s: NOT CONSTRAINED (no interior minimum, spread %.2f %%)",
                     name, pr["spread_pct"])
            log.info("     backbones scatter %.3f-%.3f eV, CV %.0f %% -- do not quote",
                     arr.min(), arr.max(),
                     100 * arr.std(ddof=1) / arr.mean())
    log.info("")
    log.info("A broad literature band overlapping a loosely constrained fit is")
    log.info("CONSISTENCY, not confirmation. It would have been informative had")
    log.info("the fit landed at 0.1 or 2 eV; it did not. Report it at that")
    log.info("strength, and note that Ea_rev cannot be quoted at all.")

    out = {"dataset": os.path.basename(cfg.PROCESSED_DATA_PATH),
           "backbone": os.path.basename(args.backbone),
           "fitted": {k: {"values": v, "mean": float(np.mean(v)),
                          "sd": float(np.std(v, ddof=1)),
                          "cv_pct": float(100 * np.std(v, ddof=1) / np.mean(v))}
                      for k, v in fitted.items()},
           "profiles": profiles, "literature": LITERATURE}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("")
    log.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
