"""
23_baseline_vanilla.py
======================
Physics-free baselines, to answer the question the A/B/C ablation cannot:
**is the physics prior itself worth anything?**

Why A/B/C cannot answer it
--------------------------
In 16_ablation_physics_condition.py all three conditions share ONE trained
physics backbone; only the residual generator's *conditioning input* changes.
Condition B ("no physics") still gets its predictions from the SRH/Arrhenius
ODE -- it merely stops looking at z_phys when generating the correction. So
A vs B measures whether feeding physics latents to the residual head helps,
not whether physics helps. A reviewer asking "how do you know the physics
constraints do anything?" is asking for THIS comparison instead.

What is replaced
----------------
Only the latent dynamics. Encoder, decoder, alpha net, training schedule,
data, splits and losses are untouched, so any difference is attributable to
the ODE and not to capacity or tuning.

    PhysicsODE                          baseline
    ------------------------------      ----------------------------------
    SRH trap kinetics, learned k_c/k_e  free MLP vector field  (--mode node)
    Arrhenius exp(-Ea/kB (1/T-1/Tref))  T as a plain input feature
    (1-z) saturation -> monotone        unconstrained
    z bounded to [0,1] structurally     unconstrained
    6 interpretable states              6 opaque states

Two baselines are provided:

  --mode node   Neural ODE: dz/dt = MLP([z, T_norm, alpha]), integrated with
                the SAME RK4 driver over the same physical time grid. This is
                the fair "same architecture, no physics" control.
  --mode gru    GRUCell stepped over the observation grid with dt as an input.
                A sequence model with no ODE structure at all; the weaker,
                more conventional baseline.

Capacity is matched deliberately. PhysicsODE has ~20 scalar parameters, so an
MLP with default width has far MORE capacity, not less -- if the baseline
still loses, it cannot be blamed on being starved. --hidden-dim controls this
and is reported.

What is measured
----------------
1. Overall test RMSE on the generated features.
2. Long-horizon extrapolation at 500 / 1000 / 2000 h -- where a prior that
   encodes saturation and Arrhenius scaling should matter most.
3. OOD temperature: train on 275+300 C, test on the held-out 325 C. Physics
   should transfer; a free vector field has no reason to.
4. Small-sample learning curve at 25 / 50 / 100 % of the training devices.
   Inductive bias is worth most when data is scarce (n=203 here).

Usage
-----
    python 23_baseline_vanilla.py --mode node
    python 23_baseline_vanilla.py --mode node --ood
    python 23_baseline_vanilla.py --mode gru --data-frac 0.5
    python 23_baseline_vanilla.py --compare      # summarise saved runs
"""

import argparse
import copy
import importlib.util
import json
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

RK4_SUBSTEPS = 8      # fixed-step RK4 per observation interval, as in PhysicsODE
STATE_CLAMP = 50.0    # runaway guard only; NOT the [0,1] physical bound


def _load(alias: str, fname: str):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Physics-free latent dynamics
# ---------------------------------------------------------------------------

class NeuralODEDynamics(nn.Module):
    """dz/dt = MLP([z, T_norm, alpha]) -- a free vector field.

    Deliberately mirrors PhysicsODE's public surface (integrate_trajectory
    with the same signature) so it is a drop-in replacement and every call
    site, loss and evaluation path stays identical.

    Nothing here encodes SRH kinetics, Arrhenius temperature scaling,
    saturation or monotonicity. Temperature enters as one more input number.
    """

    def __init__(self, latent_dim: int = None, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim or cfg.LATENT_DIM
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        # Small final layer: start near dz/dt = 0 so the first steps do not
        # blow up before the field has learned a scale.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def rhs(self, z, T_K, device_alpha):
        T_norm = ((T_K.view(-1, 1) - cfg.T_REF_K) / cfg.T_REF_K)
        a = device_alpha.view(-1, 1)
        return self.net(torch.cat([z, T_norm, a], dim=1))

    def integrate(self, z0, T_K, dt_h, device_alpha, n_substeps=RK4_SUBSTEPS):
        """Classical RK4 -- the same integrator the physics model uses, so the
        comparison isolates the vector field rather than the solver."""
        z = z0
        h = (dt_h / n_substeps).unsqueeze(1)
        for _ in range(n_substeps):
            k1 = self.rhs(z, T_K, device_alpha)
            k2 = self.rhs(z + 0.5 * h * k1, T_K, device_alpha)
            k3 = self.rhs(z + 0.5 * h * k2, T_K, device_alpha)
            k4 = self.rhs(z + h * k3, T_K, device_alpha)
            z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            z = torch.nan_to_num(z, nan=0.0, posinf=STATE_CLAMP, neginf=-STATE_CLAMP)
            z = z.clamp(-STATE_CLAMP, STATE_CLAMP)
        return z

    def integrate_trajectory(self, z0, T_K, times_h, device_alpha):
        B, T = times_h.shape
        traj = [z0.unsqueeze(1)]
        z = z0
        for t in range(1, T):
            dt = (times_h[:, t] - times_h[:, t - 1]).clamp(min=0.0)
            z = self.integrate(z, T_K, dt, device_alpha)
            traj.append(z.unsqueeze(1))
        return torch.cat(traj, dim=1)


class GRUDynamics(nn.Module):
    """A GRUCell stepped over the observation grid, with dt as an input.

    No ODE at all: the weaker, more conventional sequence baseline. The latent
    IS the GRU hidden state, so latent_dim is the hidden size and the decoder
    reads it unchanged.
    """

    def __init__(self, latent_dim: int = None, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim or cfg.LATENT_DIM
        # input per step: [T_norm, alpha, log1p(dt)]
        self.cell = nn.GRUCell(3, self.latent_dim)
        self.hidden_dim = hidden_dim

    def integrate_trajectory(self, z0, T_K, times_h, device_alpha):
        B, T = times_h.shape
        T_norm = ((T_K.view(-1, 1) - cfg.T_REF_K) / cfg.T_REF_K)
        a = device_alpha.view(-1, 1)
        traj = [z0.unsqueeze(1)]
        z = z0
        for t in range(1, T):
            dt = (times_h[:, t] - times_h[:, t - 1]).clamp(min=0.0).unsqueeze(1)
            inp = torch.cat([T_norm, a, torch.log1p(dt)], dim=1)
            z = self.cell(inp, z)
            z = torch.nan_to_num(z, nan=0.0)
            traj.append(z.unsqueeze(1))
        return torch.cat(traj, dim=1)

    # Present so loss code that probes for it degrades gracefully.
    def integrate(self, z0, T_K, dt_h, device_alpha, n_substeps=None):
        T_norm = ((T_K.view(-1, 1) - cfg.T_REF_K) / cfg.T_REF_K)
        inp = torch.cat([T_norm, device_alpha.view(-1, 1),
                         torch.log1p(dt_h.unsqueeze(1).clamp(min=0.0))], dim=1)
        return self.cell(inp, z0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_model(mods, dynamics: str, hidden_dim: int):
    """The Stage 3 model with its ODE swapped out. Everything else is shared."""
    s4a = mods["_s4a"]
    model = s4a._build_model(mods)
    if dynamics == "physics":
        pass
    elif dynamics == "node":
        model.ode = NeuralODEDynamics(hidden_dim=hidden_dim)
    elif dynamics == "gru":
        model.ode = GRUDynamics(hidden_dim=hidden_dim)
    else:
        raise ValueError(dynamics)
    return model


def _forward_loss(model, batch, device, prefix_len):
    """Prefix-conditioned rollout MSE -- the Stage 3 training objective.

    Uses only the observed cells (feature_mask), matching the physics runs.
    """
    z_enc, _ = model.encoder(batch["enc_input"].to(device), batch["mask"].to(device))
    z0 = z_enc[:, 0, :]
    alpha = model.alpha_net(batch["x0"].to(device), batch["T_K"].to(device))
    z_traj = model.ode.integrate_trajectory(
        z0, batch["T_K"].to(device), batch["times_h"].to(device), alpha)
    x_hat = model.decoder(z_traj, z_ref=z0)
    x_true = batch["x"].to(device)
    fm = batch["feature_mask"].to(device) & batch["mask"].to(device).unsqueeze(-1)
    fm[:, :prefix_len, :] = False          # score the forecast region only
    diff = (x_hat - torch.nan_to_num(x_true)) ** 2
    n = fm.sum().clamp(min=1)
    return (diff * fm).sum() / n


def train_baseline(model, mods, dataset, train_idx, val_idx, device,
                   epochs: int, lr: float, prefix_len: int, patience: int = 12):
    from torch.utils.data import DataLoader
    train_mod = mods["train"]
    dl_tr = DataLoader(train_mod.DeviceDegradationDataset(dataset, train_idx),
                       batch_size=cfg.BATCH_SIZE, shuffle=True,
                       collate_fn=train_mod.collate_fn)
    dl_va = DataLoader(train_mod.DeviceDegradationDataset(dataset, val_idx),
                       batch_size=cfg.BATCH_SIZE, shuffle=False,
                       collate_fn=train_mod.collate_fn)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for b in dl_tr:
            opt.zero_grad()
            loss = _forward_loss(model, b, device, prefix_len)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
        model.eval()
        with torch.no_grad():
            vl = [float(_forward_loss(model, b, device, prefix_len)) for b in dl_va]
        v = float(np.mean([x for x in vl if np.isfinite(x)])) if vl else float("nan")
        if np.isfinite(v) and v < best - 1e-6:
            best, bad = v, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            log.info("    epoch %3d/%d  train=%.5f  val=%.5f  best=%.5f",
                     ep, epochs, tot / max(len(dl_tr), 1), v, best)
        if bad >= patience:
            log.info("    early stop at epoch %d (best val=%.5f)", ep, best)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, mods, dataset, idx, device, prefix_len, feat_idx) -> Dict:
    """RMSE overall, per horizon and per temperature, on observed cells only."""
    from torch.utils.data import DataLoader
    train_mod, eval_mod = mods["train"], mods["eval"]
    dl = DataLoader(train_mod.DeviceDegradationDataset(dataset, idx),
                    batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=train_mod.collate_fn)
    sq, by_h, by_T = [], {}, {}
    model.eval()
    with torch.no_grad():
        for b in dl:
            out = eval_mod.predict_from_prefix(
                model, b["enc_input"], b["x"], b["mask"], b["times_h"],
                b["T_K"], b["x0"], prefix_len, device)
            xp = out["x_pred"].cpu().numpy()
            xt = b["x"].numpy()
            mk = b["mask"].numpy().astype(bool)
            fm = b["feature_mask"].numpy().astype(bool)
            tm = b["times_h"].numpy()
            TK = b["T_K"].numpy()
            for i in range(xt.shape[0]):
                for j in range(prefix_len, xt.shape[1]):
                    if not mk[i, j]:
                        continue
                    for f in feat_idx:
                        if not fm[i, j, f]:
                            continue
                        d = xt[i, j, f] - xp[i, j, f]
                        if not np.isfinite(d):
                            continue
                        sq.append(d * d)
                        by_h.setdefault(float(tm[i, j]), []).append(d * d)
                        by_T.setdefault(round(float(TK[i])), []).append(d * d)
    def _r(a):
        return float(np.sqrt(np.mean(a))) if a else float("nan")
    return {
        "rmse_overall": _r(sq),
        "n_points": len(sq),
        "rmse_by_horizon": {str(int(h)): _r(v) for h, v in sorted(by_h.items())},
        "rmse_by_temperature": {str(t): _r(v) for t, v in sorted(by_T.items())},
    }


def main():
    ap = argparse.ArgumentParser(description="Physics-free baselines")
    ap.add_argument("--mode", choices=["node", "gru", "physics"], default="node",
                    help="physics re-runs the real ODE through this same "
                         "harness, as an apples-to-apples control")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS_STAGE3)
    ap.add_argument("--lr", type=float, default=cfg.LR_STAGE3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-frac", type=float, default=1.0,
                    help="fraction of TRAIN devices to use (learning curve)")
    ap.add_argument("--ood", action="store_true",
                    help="train on 275+300 C only, test on held-out 325 C")
    ap.add_argument("--dataset", type=str, default=None,
                    help="default: processed_data_ext.pkl when EXTENDED_FEATURES")
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "baseline_vanilla"))
    ap.add_argument("--compare", action="store_true",
                    help="summarise saved runs and exit")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.compare:
        rows = []
        for fn in sorted(os.listdir(args.output_dir)):
            if fn.endswith(".json"):
                rows.append(json.load(open(os.path.join(args.output_dir, fn))))
        if not rows:
            log.info("no saved runs in %s", args.output_dir)
            return
        log.info("%-10s %-6s %-6s %8s %9s %9s %9s %9s",
                 "mode", "frac", "ood", "val", "RMSE", "t=500h", "t=1000h", "t=2000h")
        for r in sorted(rows, key=lambda x: (x["mode"], x["data_frac"], x["ood"])):
            h = r["test"]["rmse_by_horizon"]
            log.info("%-10s %-6.2f %-6s %8.5f %9.4f %9.4f %9.4f %9.4f",
                     r["mode"], r["data_frac"], str(r["ood"]), r["val_loss"],
                     r["test"]["rmse_overall"], h.get("500", float("nan")),
                     h.get("1000", float("nan")), h.get("2000", float("nan")))
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    s4b = _load("_bl_s4b", "14_stage4b_ar1_guided_generator.py")
    s4a = s4b.stage4a_mod
    mods = s4a._load_all()
    mods["_s4a"] = s4a

    ds_path = args.dataset or (
        os.path.join(cfg.OUTPUT_PATH, "processed_data_ext.pkl")
        if getattr(cfg, "EXTENDED_FEATURES", False) else cfg.PROCESSED_DATA_PATH)
    with open(ds_path, "rb") as f:
        dataset = pickle.load(f)
    log.info("dataset: %s   x %s", ds_path, np.asarray(dataset["x"]).shape)

    feat_idx = list(s4b.STABLE_FEAT_INDICES)
    train_idx = list(dataset["split"]["train"])
    val_idx = list(dataset["split"]["val"])
    test_idx = list(dataset["split"]["test"])

    if args.ood:
        # Hold out the hottest temperature entirely. This is the comparison a
        # physics prior should win: Arrhenius scaling extrapolates in T, a free
        # vector field has never seen the regime.
        TK = np.asarray(dataset["T_K"])
        hot = round(float(np.max(np.round(TK))))
        is_hot = np.round(TK) == hot
        train_idx = [i for i in range(len(TK)) if not is_hot[i] and i not in val_idx]
        test_idx = [i for i in range(len(TK)) if is_hot[i]]
        log.info("OOD: holding out %d K -> %d train / %d test devices",
                 hot, len(train_idx), len(test_idx))

    if args.data_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        k = max(8, int(round(len(train_idx) * args.data_frac)))
        train_idx = sorted(rng.choice(train_idx, size=k, replace=False).tolist())
        log.info("learning curve: using %d/%d train devices (%.0f%%)",
                 len(train_idx), len(dataset["split"]["train"]), 100 * args.data_frac)

    device = torch.device("cpu")
    model = build_model(mods, args.mode, args.hidden_dim)
    n_dyn = sum(p.numel() for p in model.ode.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    log.info("mode=%s  dynamics params=%d  total=%d", args.mode, n_dyn, n_all)

    t0 = time.time()
    model, val_loss = train_baseline(model, mods, dataset, train_idx, val_idx,
                                     device, args.epochs, args.lr,
                                     cfg.STAGE3_PREFIX_LEN)
    mins = (time.time() - t0) / 60.0

    res_test = evaluate(model, mods, dataset, test_idx, device,
                        cfg.STAGE3_PREFIX_LEN, feat_idx)
    res_train = evaluate(model, mods, dataset, train_idx, device,
                         cfg.STAGE3_PREFIX_LEN, feat_idx)

    out = {
        "mode": args.mode, "hidden_dim": args.hidden_dim, "seed": args.seed,
        "data_frac": args.data_frac, "ood": bool(args.ood),
        "epochs": args.epochs, "minutes": mins,
        "n_dynamics_params": n_dyn, "n_total_params": n_all,
        "n_train_devices": len(train_idx), "n_test_devices": len(test_idx),
        "val_loss": val_loss, "test": res_test, "train": res_train,
        "dataset": ds_path, "features_scored": [cfg.FEATURES[i] for i in feat_idx],
    }
    tag = f"{args.mode}_frac{args.data_frac:.2f}{'_ood' if args.ood else ''}_seed{args.seed}"
    path = os.path.join(args.output_dir, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    log.info("=" * 74)
    log.info("mode=%s  val=%.5f  test RMSE=%.4f  (train RMSE=%.4f)  %.1f min",
             args.mode, val_loss, res_test["rmse_overall"],
             res_train["rmse_overall"], mins)
    log.info("  by horizon   : %s", {k: round(v, 4)
                                     for k, v in res_test["rmse_by_horizon"].items()})
    log.info("  by temperature: %s", {k: round(v, 4)
                                      for k, v in res_test["rmse_by_temperature"].items()})
    log.info("Saved -> %s", path)


if __name__ == "__main__":
    main()
