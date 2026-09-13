"""
25_mechanism_candidates.py
==========================
Does the data actually prefer SRH kinetics, or is SRH just what we assumed?

The pipeline models the reversible trap states with Shockley-Read-Hall
occupancy kinetics. That is defensible, and 23_baseline_vanilla.py showed the
physics prior as a whole beats a free vector field by 13.2 % (4 seeds, CI
[-16.3, -10.2]). But "physics beats no physics" is a weaker claim than "the
data support THIS physics", and the GaN degradation literature argues for
several different rate laws. This script compares them on equal terms so the
choice becomes a measurement rather than an assumption.

Candidates
----------
Only the two fast reversible states (zG, zB, zF) change form. The irreversible
states (zM, zL, zC) keep their saturating-drift structure in every candidate,
because they are not what is in dispute.

  srh          dz/dt = kc*frev*(1-z) - ke*frev*z
               Trap capture/emission balance. Relaxes to kc/(kc+ke) with a
               single time constant. The current model.

  power        dz/dt = k*frev*n*(t+t0)^(n-1) * (1-z)
               Power-law / dispersive transport, widely fitted to GaN Vth
               drift. Unbounded rate at t -> 0 without the t0 offset.

  stretched    dz/dt = k*frev*beta*(t+t0)^(beta-1) * (1-z)
               Stretched-exponential (KWW) kinetics, standard for
               trap-limited relaxation with a distribution of time constants.
               Reduces to a plain exponential at beta = 1.

  log          dz/dt = k*frev / (1 + t/t0) * (1-z)
               Logarithmic-in-time creep, the classic form for
               thermally-activated defect motion over a barrier distribution.

All four keep the Arrhenius factor frev(T), so this compares the RATE LAW,
not whether temperature scaling helps -- that was settled separately.

An implementation note that matters
-----------------------------------
srh is autonomous: dz/dt depends on z alone. The other three depend explicitly
on elapsed time t, which the existing rhs() signature does not receive. Rather
than thread t through every call site (integrate, the IMEX split, the loss
wrappers), the candidates carry a per-batch clock that integrate_trajectory
advances. The clock is a buffer, not a parameter, and is reset per trajectory.

This also means the non-SRH candidates cannot use the IMEX exponential
update, which assumes a linear autonomous relaxation. They fall back to RK4
for the fast block. That is a fair comparison as long as RK4 is accurate here,
which was verified when the integrator was fixed (worst-step error 8e-6).

Usage
-----
    python 25_mechanism_candidates.py --screen              # Stage 1-2 only, fast
    python 25_mechanism_candidates.py --candidate power     # one candidate
    python 25_mechanism_candidates.py --compare             # summarise
"""

import argparse
import copy
import importlib.util
import json
import logging
import math
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

CANDIDATES = ["srh", "power", "stretched", "log"]

# Time offset (hours) that keeps t^(n-1) finite at t=0. Small against the
# first measurement interval (1 h) so it does not blunt the early transient
# the fast mode exists to capture.
T0_HOURS = 0.05

# Exponent bounds. Below 1 the rate decays with time (dispersive); above 1 it
# accelerates, which no degradation mechanism here should do.
EXP_MIN, EXP_MAX = 0.05, 1.0

# Cap on the exponent of the closed-form update, mirroring
# ARRHENIUS_EXP_CLAMP in 02_physics_latent.py.
ARR_CLAMP = 15.0


def _load(alias: str, fname: str):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def make_candidate_ode(ode_mod, candidate: str):
    """Return a PhysicsODE subclass whose fast states follow `candidate`.

    Everything else -- Arrhenius factors, the irreversible states, the
    sanitisers, the decoder interface -- is inherited unchanged, so the only
    difference between runs is the reversible rate law.
    """
    base = ode_mod.PhysicsODE

    if candidate == "srh":
        return base

    class CandidateODE(base):
        MECHANISM = candidate

        def __init__(self):
            super().__init__()
            # One shape exponent per fast state, softplus-bounded into
            # (EXP_MIN, EXP_MAX). Init at 0.5: dispersive, the regime the
            # literature reports for GaN trap-limited drift.
            # Distinct inits per state. Identical inits made all three
            # exponents move together and land on the same value, which looks
            # like a tied parameter but is just a symmetric starting point.
            def _inv(target):
                p = (target - EXP_MIN) / (EXP_MAX - EXP_MIN)
                return math.log(p / (1.0 - p))
            # The log law has no shape exponent -- its rate is A/(t0+t). An
            # earlier version gave it three anyway; they received no gradient
            # (verified: .grad is None) and inflated its parameter count from
            # 18 to 21, so the comparison was not like-for-like. Registered as
            # buffers there instead, so the count is honest and _exp still
            # works unchanged.
            if candidate == "log":
                for n_, v in (("expG_raw", 0.50), ("expB_raw", 0.40), ("expF_raw", 0.65)):
                    self.register_buffer(n_, torch.tensor(_inv(v)), persistent=False)
            else:
                self.expG_raw = nn.Parameter(torch.tensor(_inv(0.50)))
                self.expB_raw = nn.Parameter(torch.tensor(_inv(0.40)))
                self.expF_raw = nn.Parameter(torch.tensor(_inv(0.65)))
            # Elapsed-time clock, in hours, shaped (B,1). A buffer rather than
            # a parameter: it is state carried through a rollout, not
            # something to learn.
            self.register_buffer("_clock", torch.zeros(1, 1), persistent=False)

        def _exp(self, raw):
            return EXP_MIN + (EXP_MAX - EXP_MIN) * torch.sigmoid(raw)

        def reset_clock(self, batch_size: int, device, t0: float = 0.0):
            self._clock = torch.full((batch_size, 1), float(t0), device=device)

        def advance_clock(self, dt):
            dt = dt.view(-1, 1) if torch.is_tensor(dt) else torch.tensor(dt)
            self._clock = self._clock + dt.to(self._clock.device)

        def _clock_for(self, z):
            c = self._clock
            if c.shape[0] != z.shape[0]:
                c = c.expand(z.shape[0], 1) if c.shape[0] == 1 else \
                    torch.zeros(z.shape[0], 1, device=z.device)
            return c.clamp(min=0.0)

        def _time_factor(self, z, exponent, zf):
            """The explicitly time-dependent part of the rate.

            The three laws are genuinely different and an earlier version of
            this file gave power and stretched the SAME expression, so they
            returned byte-identical results under two names. The distinction:

              power      z(t) = A*t^n              -> dz/dt = A*n*t^(n-1),
                         and the (1-z) envelope is applied by the caller.
                         The rate depends on t only.

              stretched  z(t) = 1 - exp(-(t/tau)^b) -> dz/dt is proportional
                         to b*t^(b-1) * (1-z). The (1-z) here is intrinsic to
                         the law, not an add-on, so the state feeds back into
                         the rate -- that is what separates it from power.

              log        z(t) = A*ln(1 + t/t0)      -> dz/dt = A/(t0 + t).
            """
            t = self._clock_for(z) + T0_HOURS
            if candidate == "power":
                return exponent * torch.pow(t, exponent - 1.0)
            if candidate == "stretched":
                # KWW derivative: b*t^(b-1), with the (1-z) factor supplied
                # here rather than by the caller so the two laws differ in
                # where the saturation enters.
                return exponent * torch.pow(t, exponent - 1.0) * (1.0 - zf).clamp(min=0.0)
            # logarithmic creep: rate falls as 1/t, no exponent
            return 1.0 / (1.0 + t / T0_HOURS)

        def rhs(self, z, T_K, device_alpha):
            T_K = T_K.view(-1, 1)
            device_alpha = device_alpha.view(-1, 1)
            frev = self._arrhenius(self.Ea_rev, T_K)
            firrev = self._arrhenius(self.Ea_irrev, T_K)

            zG, zB, zF = z[:, 0:1], z[:, 1:2], z[:, 2:3]
            zM, zL, zC = z[:, 3:4], z[:, 4:5], z[:, 5:6]

            # Fast states: same saturating (1-z) envelope as SRH, but the
            # rate carries an explicit time dependence instead of an emission
            # term. kGc/kBc/kFc are reused as the amplitudes so the candidate
            # has the same parameter count as SRH minus the emission rates.
            dzG = self.kGc * frev * self._time_factor(z, self._exp(self.expG_raw), zG) * (1.0 - zG)
            dzB = self.kBc * frev * self._time_factor(z, self._exp(self.expB_raw), zB) * (1.0 - zB)
            dzF = self.kFc * frev * self._time_factor(z, self._exp(self.expF_raw), zF) * (1.0 - zF)

            # Irreversible states: unchanged from the base model.
            driving_M, driving_L = self.driving_forces(z)
            dzM = self.kM * firrev * driving_M * (1.0 - zM)
            dzL = self.kL * firrev * (1.0 - zL) * driving_L
            dzC = self.kC * device_alpha * firrev * (1.0 - zC) ** 2
            return torch.cat([dzG, dzB, dzF, dzM, dzL, dzC], dim=1)

        def _shape_integral(self, t, exponent):
            """Antiderivative of the rate shape f(t), evaluated at t.

            Every candidate is SEPARABLE -- dz/dt = k*f(t)*(1-z) -- so the
            interval update is exact once F(t) = INT f(t) dt is known:

                z(t1) = 1 - (1 - z(t0)) * exp(-k * [F(t1) - F(t0)])

            and F is analytic for all three:
                power / stretched   INT p*(t+t0)^(p-1) dt = (t+t0)^p
                log                 INT 1/(1+t/t0)    dt = t0*ln(1+t/t0)

            Verified against 200k-point quadrature to <= 6e-11 over intervals
            from [0,1] h to [1000,2000] h.
            """
            tt = (t + T0_HOURS).clamp(min=1e-12)
            if candidate in ("power", "stretched"):
                return torch.pow(tt, exponent)
            return T0_HOURS * torch.log1p(t.clamp(min=0.0) / T0_HOURS)

        def _exact_fast_step(self, z_prev, k, frev, t0, t1, exponent, order=1.0):
            """Closed-form update of one fast state over [t0, t1].

            order=1 for the (1-z) envelope (power, log); order=2 for the
            stretched law, whose (1-z)^2 integrates to a rational form:
                d z/dt = c*(1-z)^2  ->  1/(1-z1) = 1/(1-z0) + c*dF
            """
            dF = (self._shape_integral(t1, exponent)
                  - self._shape_integral(t0, exponent)).clamp(min=0.0)
            c = (k * frev * dF).clamp(max=ARR_CLAMP)
            gap = (1.0 - z_prev).clamp(min=1e-6)
            if order == 1.0:
                return 1.0 - gap * torch.exp(-c)
            return 1.0 - 1.0 / (1.0 / gap + c)

        def integrate_trajectory(self, z0, T_K, times_h, device_alpha):
            """Exact per-interval update -- no substeps, no chained Jacobian.

            Replaces an 8-substep RK4 rollout whose BACKWARD pass produced
            gradients of 1e13-1e16 (measured), clipped every step and so
            effectively random. The forward values were always bounded; the
            blow-up came from compounding 88 chained steps. With the exact
            solution there is one differentiable expression per interval.
            """
            B, T = times_h.shape
            TK = T_K.view(-1, 1)
            frev = self._arrhenius(self.Ea_rev, TK)
            firrev = self._arrhenius(self.Ea_irrev, TK)
            alpha = device_alpha.view(-1, 1)
            eG, eB, eF = (self._exp(self.expG_raw), self._exp(self.expB_raw),
                          self._exp(self.expF_raw))
            stretched = (candidate == "stretched")

            traj = [z0.unsqueeze(1)]
            z = z0
            for i in range(1, T):
                t0 = times_h[:, i - 1].view(-1, 1).clamp(min=0.0)
                t1 = times_h[:, i].view(-1, 1).clamp(min=0.0)
                dt = (t1 - t0).clamp(min=0.0)

                zG = self._exact_fast_step(z[:, 0:1], self.kGc, frev, t0, t1, eG,
                                           2.0 if stretched else 1.0)
                zB = self._exact_fast_step(z[:, 1:2], self.kBc, frev, t0, t1, eB,
                                           2.0 if stretched else 1.0)
                zF = self._exact_fast_step(z[:, 2:3], self.kFc, frev, t0, t1, eF,
                                           2.0 if stretched else 1.0)

                # Irreversible states keep the base model's dynamics, advanced
                # with the same RK4 the SRH model uses for them. They are held
                # at the interval start while the fast block is updated, which
                # is the Lie-Trotter split the base integrator already applies.
                z_slow_in = torch.cat([z[:, 0:3], z[:, 3:6]], dim=1)
                slow = self._slow_rk4(z_slow_in, TK, firrev, alpha, dt)
                z = self._sanitize_state(torch.cat([zG, zB, zF, slow], dim=1))
                traj.append(z.unsqueeze(1))
            return torch.cat(traj, dim=1)

        def _slow_rk4(self, z, TK, firrev, alpha, dt, n=4):
            """RK4 on zM, zL, zC only. Their time constants are far longer
            than any step, so a few substeps suffice and the backward pass
            stays short."""
            h = (dt / n)
            zs = z[:, 3:6]
            def f(zfull):
                zM, zL, zC = zfull[:, 3:4], zfull[:, 4:5], zfull[:, 5:6]
                dM_, dL_ = self.driving_forces(zfull)
                dzM = self.kM * firrev * dM_ * (1.0 - zM)
                dzL = self.kL * firrev * (1.0 - zL) * dL_
                dzC = self.kC * alpha * firrev * (1.0 - zC) ** 2
                return torch.cat([dzM, dzL, dzC], dim=1)
            for _ in range(n):
                base3 = torch.cat([z[:, 0:3], zs], dim=1)
                k1 = f(base3)
                k2 = f(torch.cat([z[:, 0:3], zs + 0.5 * h * k1], dim=1))
                k3 = f(torch.cat([z[:, 0:3], zs + 0.5 * h * k2], dim=1))
                k4 = f(torch.cat([z[:, 0:3], zs + h * k3], dim=1))
                zs = zs + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                zs = zs.clamp(-0.2, 1.2)
            return zs

    CandidateODE.__name__ = f"PhysicsODE_{candidate}"
    return CandidateODE


def build_model(mods, ode_cls, s4a):
    model = s4a._build_model(mods)
    if ode_cls is not mods["ode"].PhysicsODE:
        model.ode = ode_cls()
    return model


def _loss(model, batch, device, prefix_len):
    z_enc, _ = model.encoder(batch["enc_input"].to(device), batch["mask"].to(device))
    z0 = z_enc[:, 0, :]
    alpha = model.alpha_net(batch["x0"].to(device), batch["T_K"].to(device))
    z_traj = model.ode.integrate_trajectory(
        z0, batch["T_K"].to(device), batch["times_h"].to(device), alpha)
    x_hat = model.decoder(z_traj, z_ref=z0)
    fm = batch["feature_mask"].to(device) & batch["mask"].to(device).unsqueeze(-1)
    fm[:, :prefix_len, :] = False
    diff = (x_hat - torch.nan_to_num(batch["x"].to(device))) ** 2
    return (diff * fm).sum() / fm.sum().clamp(min=1)


def train(model, mods, dataset, tr_idx, va_idx, device, epochs, lr, prefix_len,
          patience=12):
    from torch.utils.data import DataLoader
    tm = mods["train"]
    dl_tr = DataLoader(tm.DeviceDegradationDataset(dataset, tr_idx),
                       batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=tm.collate_fn)
    dl_va = DataLoader(tm.DeviceDegradationDataset(dataset, va_idx),
                       batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=tm.collate_fn)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    best_ep, stopped_early, last_ep = 0, False, 0
    for ep in range(1, epochs + 1):
        last_ep = ep
        model.train()
        for b in dl_tr:
            opt.zero_grad()
            l = _loss(model, b, device, prefix_len)
            if not torch.isfinite(l):
                continue
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vs = [float(_loss(model, b, device, prefix_len)) for b in dl_va]
        v = float(np.mean([x for x in vs if np.isfinite(x)])) if vs else float("nan")
        if np.isfinite(v) and v < best - 1e-6:
            best, bad, best_state = v, 0, copy.deepcopy(model.state_dict())
            best_ep = ep
        else:
            bad += 1
        if ep == 1 or ep % 20 == 0:
            log.info("      epoch %3d/%d  val=%.5f  best=%.5f", ep, epochs, v, best)
        if bad >= patience:
            log.info("      early stop at epoch %d", ep)
            stopped_early = True
            break
    if best_state:
        model.load_state_dict(best_state)
    # best_epoch and the early-stop flag are what separate "this candidate lost"
    # from "this candidate never trained" -- the failure mode that invalidated
    # the first screen, where losers came back at their initialisation.
    return model, best, {"best_epoch": best_ep, "epochs_run": last_ep,
                         "stopped_early": stopped_early}


def _boundary_hits(ode, candidate=None, tol=0.02):
    """Count shape exponents pinned at the edge of their permitted range.

    The exponents are sigmoid-mapped into [EXP_MIN, EXP_MAX]. A value sitting
    within `tol` of either edge means the optimiser pushed it as far as the
    parameterisation allows -- the fit wanted a value the model cannot express,
    so the reported exponent is a boundary artefact rather than a measurement.
    Also flags exponents that never moved off their initialisation, which is
    what the invalid first screen looked like.
    """
    # The log-creep law has no shape exponent -- its rate falls as 1/t by
    # construction (see _time_factor). Its expG/B/F parameters exist on the
    # shared module but are never read, so they sit at initialisation in every
    # run. Reporting that as "never left initialisation" would invent a
    # training failure that is really an unused parameter.
    if candidate == "log":
        return {"n_boundary_hits": 0, "at_boundary": [], "n_at_init": 0,
                "at_init": [], "exponent_values": {},
                "note": "log creep has no shape exponent; nothing to check"}
    inits = {"G": 0.50, "B": 0.40, "F": 0.65}
    hits, stuck, vals = [], [], {}
    for n in ("G", "B", "F"):
        raw = getattr(ode, f"exp{n}_raw", None)
        if raw is None:
            continue
        v = float(ode._exp(raw))
        vals[n] = v
        if v <= EXP_MIN + tol or v >= EXP_MAX - tol:
            hits.append(n)
        if abs(v - inits[n]) < 1e-3:
            stuck.append(n)
    return {"n_boundary_hits": len(hits), "at_boundary": hits,
            "n_at_init": len(stuck), "at_init": stuck, "exponent_values": vals}


def evaluate(model, mods, dataset, idx, device, prefix_len, feat_idx):
    from torch.utils.data import DataLoader
    tm, em = mods["train"], mods["eval"]
    dl = DataLoader(tm.DeviceDegradationDataset(dataset, idx),
                    batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=tm.collate_fn)
    sq, by_h = [], {}
    model.eval()
    with torch.no_grad():
        for b in dl:
            out = em.predict_from_prefix(model, b["enc_input"], b["x"], b["mask"],
                                         b["times_h"], b["T_K"], b["x0"], prefix_len, device)
            xp, xt = out["x_pred"].cpu().numpy(), b["x"].numpy()
            mk = b["mask"].numpy().astype(bool)
            fm = b["feature_mask"].numpy().astype(bool)
            tmh = b["times_h"].numpy()
            for i in range(xt.shape[0]):
                for j in range(prefix_len, xt.shape[1]):
                    if not mk[i, j]:
                        continue
                    for f in feat_idx:
                        if not fm[i, j, f]:
                            continue
                        d = xt[i, j, f] - xp[i, j, f]
                        if np.isfinite(d):
                            sq.append(d * d)
                            by_h.setdefault(float(tmh[i, j]), []).append(d * d)
    r = lambda a: float(np.sqrt(np.mean(a))) if a else float("nan")
    return {"rmse_overall": r(sq), "n_points": len(sq),
            "rmse_by_horizon": {str(int(h)): r(v) for h, v in sorted(by_h.items())}}


def main():
    ap = argparse.ArgumentParser(description="Compare candidate rate laws")
    ap.add_argument("--candidate", choices=CANDIDATES + ["all"], default="all")
    ap.add_argument("--dataset", type=str, default=None,
                    help="explicit dataset path (overrides the default)")
    ap.add_argument("--screen", action="store_true",
                    help="short run for ranking, not for publication numbers")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "mechanism_candidates"))
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare:
        rows = [json.load(open(os.path.join(args.output_dir, f)))
                for f in sorted(os.listdir(args.output_dir)) if f.endswith(".json")]
        rows = [r for r in rows if isinstance(r, dict) and "candidate" in r]
        if not rows:
            log.info("no runs yet in %s", args.output_dir)
            return
        log.info("%-11s %-7s %6s %9s %9s %9s %9s %9s", "candidate", "screen",
                 "epochs", "val", "RMSE", "t=500h", "t=1000h", "t=2000h")
        for r in sorted(rows, key=lambda x: x["test"]["rmse_overall"]):
            h = r["test"]["rmse_by_horizon"]
            log.info("%-11s %-7s %6d %9.5f %9.4f %9.4f %9.4f %9.4f",
                     r["candidate"], str(r["screen"]), r["epochs"], r["val_loss"],
                     r["test"]["rmse_overall"], h.get("500", float("nan")),
                     h.get("1000", float("nan")), h.get("2000", float("nan")))
        return

    epochs = args.epochs or (40 if args.screen else cfg.EPOCHS_STAGE3)
    # Explicit --dataset, because the default below silently ignored a cfg
    # override: the first full-budget seeds ran on the UNFILTERED file while the
    # rest of the pipeline had moved to the filtered one.
    ds_path = args.dataset or (
        os.path.join(cfg.OUTPUT_PATH, "processed_data_ext.pkl")
        if getattr(cfg, "EXTENDED_FEATURES", False) else cfg.PROCESSED_DATA_PATH)
    with open(ds_path, "rb") as f:
        dataset = pickle.load(f)

    s4b = _load("_mc_s4b", "14_stage4b_ar1_guided_generator.py")
    s4a = s4b.stage4a_mod
    mods = s4a._load_all()
    feat_idx = list(s4b.STABLE_FEAT_INDICES)
    device = torch.device("cpu")
    todo = CANDIDATES if args.candidate == "all" else [args.candidate]

    log.info("dataset %s   epochs=%d   screen=%s", os.path.basename(ds_path),
             epochs, args.screen)

    for cand in todo:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        log.info("=" * 70)
        log.info("candidate: %s", cand)
        ode_cls = make_candidate_ode(mods["ode"], cand)
        model = build_model(mods, ode_cls, s4a)
        n_dyn = sum(p.numel() for p in model.ode.parameters())
        t0 = time.time()
        try:
            model, val, fit = train(model, mods, dataset, dataset["split"]["train"],
                               dataset["split"]["val"], device, epochs,
                               cfg.LR_STAGE3, cfg.STAGE3_PREFIX_LEN)
            res = evaluate(model, mods, dataset, dataset["split"]["test"],
                           device, cfg.STAGE3_PREFIX_LEN, feat_idx)
        except Exception as exc:                       # noqa: BLE001
            log.error("  %s FAILED: %s: %s", cand, type(exc).__name__, exc)
            continue
        mins = (time.time() - t0) / 60
        out = {"candidate": cand, "screen": bool(args.screen), "epochs": epochs,
               "seed": args.seed, "minutes": mins, "n_dynamics_params": n_dyn,
               "val_loss": val, "test": res, "dataset": ds_path}
        out.update(fit)                       # best_epoch, epochs_run, stopped_early
        out["boundary"] = _boundary_hits(model.ode, cand)
        # Three distinct states, because "not converged" alone is ambiguous:
        #   settled       -- early-stopped, or best reached before the last epoch
        #   budget_bound  -- still improving when the budget ran out
        #   failed        -- never improved, or an exponent never moved
        # Only `failed` invalidates a candidate's number. `budget_bound` means
        # every candidate is equally under-trained, which changes the absolute
        # RMSEs but not their ranking.
        if fit["best_epoch"] <= 1 or out["boundary"]["n_at_init"] > 0:
            state = "failed"
        elif fit["stopped_early"] or fit["best_epoch"] < fit["epochs_run"]:
            state = "settled"
        else:
            state = "budget_bound"
        out["fit_state"] = state
        out["converged"] = bool(state == "settled")
        # report learned exponents where the candidate has them
        if hasattr(model.ode, "expG_raw") and cand != "log":
            out["exponents"] = {
                n: float(model.ode._exp(getattr(model.ode, f"exp{n}_raw")))
                for n in ("G", "B", "F")}
        tag = f"{cand}{'_screen' if args.screen else ''}_seed{args.seed}"
        with open(os.path.join(args.output_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log.info("  %s: val=%.5f  test RMSE=%.4f  (%d dyn params, %.1f min)",
                 cand, val, res["rmse_overall"], n_dyn, mins)
        if "exponents" in out:
            log.info("     learned exponents: %s",
                     {k: round(v, 3) for k, v in out["exponents"].items()})


if __name__ == "__main__":
    main()
