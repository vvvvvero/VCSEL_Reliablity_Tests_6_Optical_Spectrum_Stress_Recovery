"""
26_reliability_assessment.py
============================
Score the predictive distribution the way reliability engineering scores it,
rather than the way generative modelling does.

Why a separate script
---------------------
CRPS and mean coverage answer "is the distribution about right on average".
A reliability engineer asks narrower questions: what fraction of parts fail
before the warranty horizon, what derating keeps 90 % alive, is a 2000 h
extrapolation to 10 years trustworthy. Every one of those depends on the TAIL
and the WIDTH of the predictive distribution, and a model can look good on
CRPS while getting both wrong.

Two standard instruments are used here, neither of which is currently reported:

**Winkler interval score** (Winkler 1972; the interval score in Gneiting &
Raftery 2007). For a (1-a) interval,

    S = width + (2/a) * [ (lo - y)+ + (y - hi)+ ]

At a = 0.10 a miss costs 20x the width it saved. That asymmetry is the point:
an interval that is too narrow understates risk, which in a qualification
report is the dangerous direction, while an interval that is too wide merely
costs margin. The implementation is reused from 10_stochastic_residual.py.

**PIT — probability integral transform** (Dawid 1984; Gneiting et al. 2007).
Evaluate each observation in its own predictive CDF. If the model is
calibrated the resulting values are Uniform(0,1). The SHAPE of the departure
diagnoses the fault, which a single coverage number cannot:

    hump in the middle   intervals too wide  (over-dispersed)
    U-shape / bathtub    intervals too narrow (under-dispersed) -- the
                         dangerous case, mass piling up in both tails
    sloped               biased -- the whole distribution sits off-centre

Reported with a Kolmogorov-Smirnov distance from uniformity and, separately,
the two extreme bins, since tail behaviour is what reliability work turns on.

Usage
-----
    python 26_reliability_assessment.py                       # Stage 4C variants
    python 26_reliability_assessment.py --compare-stage5
    python 26_reliability_assessment.py --runs stage4c,stage4c_sigma
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

EXT_DIR = os.path.join(cfg.OUTPUT_PATH, "ext11")

# Nominal levels to score. 90 % is the headline; the others show whether a
# fault is confined to the tail or runs through the whole distribution.
ALPHAS = [0.50, 0.20, 0.10]

PIT_BINS = 10
N_SAMPLES = 200          # more than evaluation's 100: PIT resolves the tail


def _load(alias: str, fname: str):
    path = os.path.join(BASE_DIR, fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Interval score. Lower is better; a miss costs (2/alpha) x the shortfall."""
    width = hi - lo
    return float(np.mean(width + (2.0 / alpha)
                         * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))))


def pit_values(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """PIT via the empirical CDF of the sample ensemble.

    Ties are broken by averaging the open and closed ranks, which keeps the
    values uniform for discrete ensembles instead of biasing them downward.
    """
    below = (samples < y[None, :]).mean(axis=0)
    equal = (samples == y[None, :]).mean(axis=0)
    return below + 0.5 * equal


def ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between the PIT values and Uniform(0,1)."""
    if len(u) == 0:
        return float("nan")
    us = np.sort(u)
    n = len(us)
    cdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(cdf - us)))


def pit_shape(u: np.ndarray, bins: int = PIT_BINS) -> str:
    """Name the departure from uniformity, since the shape is the diagnosis."""
    if len(u) < 30:
        return "too few points"
    h, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    h = h / max(h.sum(), 1)
    exp = 1.0 / bins
    edges = h[0] + h[-1]
    middle = h[bins // 2 - 1] + h[bins // 2]
    if edges > 2.6 * exp:
        return "U-shaped -> intervals TOO NARROW (under-dispersed)"
    if middle > 2.6 * exp:
        return "central hump -> intervals TOO WIDE (over-dispersed)"
    first, last = h[: bins // 2].sum(), h[bins // 2:].sum()
    if abs(first - last) > 0.24:
        return f"sloped -> BIASED ({'low' if first > last else 'high'} side heavy)"
    return "approximately uniform"


def collect(model, generator, cache, s4b, device, n_samples: int,
            feat_idx: List[int]) -> Dict[str, Dict[str, np.ndarray]]:
    """Draw ensembles and return per-feature (observations, samples).

    Scored on the forecast region only, and only where the observation is
    actually present, matching every other evaluation in the repo.
    """
    out = {cfg.FEATURES[f]: {"y": [], "s": []} for f in feat_idx}
    generator.eval()
    with torch.no_grad():
        for rec in cache:
            plen = rec["plen"]
            T_future = rec["T_len"] - plen
            if T_future <= 0:
                continue
            # Signature matches evaluate_stage4b in 14_: no z_ref, and
            # times_future only where the generator accepts it.
            deltas = generator.sample_n(
                rec["z_pfx"].to(device), rec["T_K"].to(device), rec["x0"].to(device),
                rec["log_t"].to(device), n_samples, T_future=T_future)
            x_hat = rec["x_hat"][:, plen:, :].to(device)
            x_true = rec["x_true"][:, plen:, :].cpu().numpy()
            fmask = rec["mask"][:, plen:].bool().cpu().numpy()
            fm_feat = (rec["feature_mask"][:, plen:, :].bool().cpu().numpy()
                       if "feature_mask" in rec else None)
            for slot, f in enumerate(feat_idx):
                # generator slot order follows STABLE_FEAT_INDICES
                samp = (x_hat[..., f].unsqueeze(0) + deltas[..., slot]).cpu().numpy()
                for b in range(x_true.shape[0]):
                    for t in range(x_true.shape[1]):
                        if not fmask[b, t]:
                            continue
                        if fm_feat is not None and not fm_feat[b, t, f]:
                            continue
                        yv = x_true[b, t, f]
                        if not np.isfinite(yv):
                            continue
                        col = samp[:, b, t]
                        if not np.isfinite(col).all():
                            continue
                        out[cfg.FEATURES[f]]["y"].append(float(yv))
                        out[cfg.FEATURES[f]]["s"].append(col)
    return {k: {"y": np.array(v["y"]),
                "s": (np.array(v["s"]).T if v["s"] else np.zeros((0, 0)))}
            for k, v in out.items()}


def assess(data: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    """Winkler at each level, plus PIT, per feature and pooled."""
    res = {"per_feature": {}, "overall": {}}
    all_u: List[np.ndarray] = []
    wink_acc = {a: [] for a in ALPHAS}
    cov_acc = {a: [] for a in ALPHAS}
    for name, d in data.items():
        y, s = d["y"], d["s"]
        if y.size == 0 or s.size == 0:
            continue
        entry = {"n": int(y.size)}
        for a in ALPHAS:
            lo = np.quantile(s, a / 2.0, axis=0)
            hi = np.quantile(s, 1.0 - a / 2.0, axis=0)
            entry[f"winkler_{int((1-a)*100)}"] = winkler(y, lo, hi, a)
            entry[f"width_{int((1-a)*100)}"] = float(np.mean(hi - lo))
            cov = float(np.mean((y >= lo) & (y <= hi)))
            entry[f"cov_{int((1-a)*100)}"] = cov
            wink_acc[a].append(entry[f"winkler_{int((1-a)*100)}"])
            cov_acc[a].append(cov)
        u = pit_values(y, s)
        entry["pit_ks"] = ks_uniform(u)
        entry["pit_shape"] = pit_shape(u)
        h, _ = np.histogram(u, bins=PIT_BINS, range=(0, 1))
        entry["pit_hist"] = (h / max(h.sum(), 1)).tolist()
        entry["pit_tail_mass"] = float((h[0] + h[-1]) / max(h.sum(), 1))
        res["per_feature"][name] = entry
        all_u.append(u)
    if all_u:
        u = np.concatenate(all_u)
        res["overall"]["pit_ks"] = ks_uniform(u)
        res["overall"]["pit_shape"] = pit_shape(u)
        h, _ = np.histogram(u, bins=PIT_BINS, range=(0, 1))
        res["overall"]["pit_hist"] = (h / max(h.sum(), 1)).tolist()
        res["overall"]["pit_tail_mass"] = float((h[0] + h[-1]) / max(h.sum(), 1))
        res["overall"]["pit_tail_expected"] = 2.0 / PIT_BINS
        for a in ALPHAS:
            lvl = int((1 - a) * 100)
            res["overall"][f"winkler_{lvl}"] = float(np.mean(wink_acc[a]))
            res["overall"][f"cov_{lvl}"] = float(np.mean(cov_acc[a]))
    return res


def build(run: str, s4b, s4a, mods, model, device, stage5: bool = False):
    """Load one generator checkpoint by run directory name."""
    if stage5:
        path = os.path.join(EXT_DIR, "stage5", "stage5_best.pt")
    else:
        path = os.path.join(EXT_DIR, run, "checkpoints", "stage4b_best.pt")
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu")
    gen = s4b.AR1GuidedResidualGeneratorArrhenius().to(device)
    gen.load_state_dict(ck["state_dict"], strict=False)
    gen.eval()
    return gen


def main():
    ap = argparse.ArgumentParser(description="Reliability-engineering scoring")
    ap.add_argument("--runs", type=str,
                    default="stage4c,stage4c_offset,stage4c_sigma",
                    help="Stage 4C run directories under ext11/")
    ap.add_argument("--compare-stage5", action="store_true")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--output", type=str,
                    default=os.path.join(cfg.RESULTS_DIR, "reliability_assessment.json"))
    args = ap.parse_args()

    cfg.PROCESSED_DATA_PATH = os.path.join(cfg.OUTPUT_PATH, "processed_data_ext.pkl")
    cfg.CHECKPOINT_DIR = os.path.join(EXT_DIR, "checkpoints")

    s4b = _load("_ra_s4b", "14_stage4b_ar1_guided_generator.py")
    s4a = s4b.stage4a_mod
    mods = s4a._load_all()
    train_mod = mods["train"]
    device = torch.device("cpu")

    model = s4a._build_model(mods)
    ck = torch.load(os.path.join(EXT_DIR, "checkpoints", "stage3_best.pt"),
                    map_location="cpu")
    ms = ck.get("model_state", ck.get("model_state_dict"))
    model.load_state_dict({k: v for k, v in ms.items() if k != "decoder.mask"},
                          strict=False)
    model.eval()

    with open(cfg.PROCESSED_DATA_PATH, "rb") as f:
        ds = pickle.load(f)
    from torch.utils.data import DataLoader
    dl = DataLoader(train_mod.DeviceDegradationDataset(ds, ds["split"][args.split]),
                    batch_size=cfg.BATCH_SIZE, shuffle=False,
                    collate_fn=train_mod.collate_fn)
    cache = s4a._cache_trajectories(model, dl, device, train_mod._forward,
                                    cfg.STAGE3_PREFIX_LEN)
    feat_idx = list(s4b.STABLE_FEAT_INDICES)
    log.info("split=%s  %d cached batches  %d features  %d samples",
             args.split, len(cache), len(feat_idx), args.n_samples)

    jobs = [(r, False) for r in args.runs.split(",") if r.strip()]
    if args.compare_stage5:
        jobs.append(("stage5", True))

    results = {}
    for run, is_s5 in jobs:
        gen = build(run, s4b, s4a, mods, model, device, stage5=is_s5)
        if gen is None:
            log.warning("no checkpoint for %s, skipping", run)
            continue
        torch.manual_seed(0)
        data = collect(model, gen, cache, s4b, device, args.n_samples, feat_idx)
        results[run] = assess(data)
        log.info("scored %s", run)

    if not results:
        log.error("nothing scored")
        return

    lvl = 90
    log.info("=" * 78)
    log.info("WINKLER INTERVAL SCORE at %d%% (lower is better; a miss costs 20x)", lvl)
    log.info("%-22s" % "run" + "".join(f"{int((1-a)*100):>12}%" for a in ALPHAS))
    for run, r in results.items():
        o = r["overall"]
        log.info("%-22s" % run
                 + "".join(f"{o.get(f'winkler_{int((1-a)*100)}', float('nan')):>13.4f}"
                           for a in ALPHAS))
    log.info("")
    log.info("COVERAGE (nominal 50 / 80 / 90)")
    log.info("%-22s" % "run" + "".join(f"{int((1-a)*100):>12}%" for a in ALPHAS))
    for run, r in results.items():
        o = r["overall"]
        log.info("%-22s" % run
                 + "".join(f"{o.get(f'cov_{int((1-a)*100)}', float('nan')):>13.4f}"
                           for a in ALPHAS))
    log.info("")
    log.info("PIT — departure from uniformity (KS distance; lower is better)")
    log.info("%-22s %10s %12s   %s", "run", "KS", "tail mass", "diagnosis")
    for run, r in results.items():
        o = r["overall"]
        log.info("%-22s %10.4f %12.4f   %s", run, o.get("pit_ks", float("nan")),
                 o.get("pit_tail_mass", float("nan")), o.get("pit_shape", "-"))
    log.info("   (expected tail mass for a calibrated model: %.2f)",
             2.0 / PIT_BINS)
    log.info("")
    log.info("PIT HISTOGRAM, 10 bins — flat is calibrated")
    for run, r in results.items():
        h = r["overall"].get("pit_hist", [])
        bars = " ".join(f"{v:.3f}" for v in h)
        log.info("  %-20s %s", run, bars)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("")
    log.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
