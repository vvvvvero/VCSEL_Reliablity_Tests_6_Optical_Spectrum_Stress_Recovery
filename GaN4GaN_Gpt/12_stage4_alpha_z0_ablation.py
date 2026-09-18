"""
12_stage4_alpha_z0_ablation.py
==============================
True 2x2 alpha x z0 Stage4 ablation with independent short training per config.

Configs:
- A: fixed alpha=1, no prefix z0 perturbation
- B: fixed alpha=1, with prefix z0 perturbation
- C: learned alpha, no prefix z0 perturbation
- D: learned alpha, with prefix z0 perturbation

Each config is trained from the same Stage3 checkpoint, then evaluated using
GeneratorSampleModel on the same test split.

Outputs (under --output-dir):
- stage4_ablation_table.csv
- stage4_ablation_table.md
- stage4_ablation_full.pkl
"""

import argparse
import csv
import os
import pickle
import random
import shutil
import sys
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from typing import Dict, List

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg


def _load_module(alias: str, filename: str):
    path = os.path.join(BASE_DIR, filename)
    spec = spec_from_file_location(alias, path)
    mod = module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_model(mods: Dict):
    import torch.nn as nn

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = mods["enc"].PhysicsEncoder()
            self.decoder = mods["dec"].SparsePhysicsDecoder()
            self.ode = mods["ode"].PhysicsODE()
            self.alpha_net = mods["ode"].DeviceAlphaNet()
            self.generator = mods["gen"].PITimeGANGenerator()
            self.disc = mods["disc"].PITimeGANDiscriminator()

    return _Model


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@contextmanager
def _override_cfg(overrides: Dict):
    backup = {}
    for k, v in overrides.items():
        backup[k] = getattr(cfg, k)
        setattr(cfg, k, v)
    try:
        yield
    finally:
        for k, v in backup.items():
            setattr(cfg, k, v)


def _predict_indices(model, dataset: Dict, indices: np.ndarray, train_mod, eval_mod, device):
    import torch
    from torch.utils.data import DataLoader

    ds = train_mod.DeviceDegradationDataset(dataset, indices)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=train_mod.collate_fn)

    preds, trues, masks, tks, times = [], [], [], [], []
    enc_inputs, x0s = [], []
    with torch.no_grad():
        for b in dl:
            out = eval_mod.predict_from_prefix(
                model,
                b["enc_input"],
                b["x"],
                b["mask"],
                b["times_h"],
                b["T_K"],
                b["x0"],
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                device=device,
            )
            preds.append(out["x_pred"].cpu().numpy())
            trues.append(b["x"].cpu().numpy())
            masks.append(b["mask"].cpu().numpy())
            tks.append(b["T_K"].cpu().numpy())
            times.append(b["times_h"].cpu().numpy())
            enc_inputs.append(b["enc_input"].cpu().numpy())
            x0s.append(b["x0"].cpu().numpy())

    return {
        "x_pred_norm": np.concatenate(preds, axis=0),
        "x_true_norm": np.concatenate(trues, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "T_K": np.concatenate(tks, axis=0),
        "times_h": np.concatenate(times, axis=0),
        "enc_input": np.concatenate(enc_inputs, axis=0),
        "x0": np.concatenate(x0s, axis=0),
    }


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _to_md_table(rows: List[Dict], cols: List[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def main():
    p = argparse.ArgumentParser(description="True 2x2 Stage4 alpha x z0 ablation")
    p.add_argument("--dataset", type=str, default=None,
                   help="explicit dataset path; the default followed "
                        "cfg.PROCESSED_DATA_PATH and could pair the wrong "
                        "feature set with the checkpoint")
    p.add_argument("--checkpoint-stage3", default=os.path.join(cfg.CHECKPOINT_DIR, "stage3_best.pt"))
    p.add_argument("--output-dir", default=os.path.join(cfg.RESULTS_DIR, "stage4_alpha_z0_ablation"))
    p.add_argument("--short-epochs", type=int, default=40)
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=int(cfg.RANDOM_SEED))
    args = p.parse_args()

    if not os.path.exists(args.checkpoint_stage3):
        raise FileNotFoundError(f"Missing stage3 checkpoint: {args.checkpoint_stage3}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_root = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    mods = {
        "prep": _load_module("_pi_prep_ab", "01_data_preprocessing.py"),
        "ode": _load_module("_pi_ode_ab", "02_physics_latent.py"),
        "enc": _load_module("_pi_enc_ab", "03_model_encoder.py"),
        "dec": _load_module("_pi_dec_ab", "04_model_decoder.py"),
        "gen": _load_module("_pi_gen_ab", "05_model_generator.py"),
        "disc": _load_module("_pi_disc_ab", "06_model_discriminator.py"),
        "losses": _load_module("_pi_losses_ab", "07_losses.py"),
        "train": _load_module("_pi_train_ab", "08_training.py"),
        "eval": _load_module("_pi_eval_ab", "09_evaluation.py"),
        "stoch": _load_module("_pi_stoch_ab", "10_stochastic_residual.py"),
    }

    # Inject losses into training module exactly like main.py.
    for lname in [
        "reconstruction_loss",
        "balanced_feature_huber_loss",
        "ode_residual_loss",
        "bounds_loss",
        "monotonicity_loss",
        "temperature_ordering_loss",
        "zc_prefix_separation_loss",
        "multistep_prediction_loss",
        "adversarial_generator_loss",
        "adversarial_discriminator_loss",
        "distribution_matching_loss",
        "total_physics_loss",
    ]:
        if hasattr(mods["losses"], lname):
            setattr(mods["train"], lname, getattr(mods["losses"], lname))

    dataset = mods["prep"].load_dataset(args.dataset)
    split = dataset["split"]

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cpu")

    train_ds = mods["train"].DeviceDegradationDataset(dataset, split["train"])
    val_ds = mods["train"].DeviceDegradationDataset(dataset, split["val"])
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=mods["train"].collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=mods["train"].collate_fn)

    model_cls = _build_model(mods)

    configs = [
        {
            "name": "A_fixed_alpha_no_z0",
            "fixed_alpha": True,
            "use_prefix_context": False,
        },
        {
            "name": "B_fixed_alpha_z0",
            "fixed_alpha": True,
            "use_prefix_context": True,
        },
        {
            "name": "C_gen_alpha_no_z0",
            "fixed_alpha": False,
            "use_prefix_context": False,
        },
        {
            "name": "D_gen_alpha_z0_full",
            "fixed_alpha": False,
            "use_prefix_context": True,
        },
    ]

    rows = []
    raw = {}

    for i, conf in enumerate(configs):
        run_seed = int(args.seed + i)
        _set_seed(run_seed)

        run_ckpt_dir = os.path.join(ckpt_root, conf["name"])
        os.makedirs(run_ckpt_dir, exist_ok=True)

        # Train Stage4 from the same Stage3 checkpoint under config-specific settings.
        model = mods["eval"].load_model(args.checkpoint_stage3, model_cls).to(device)

        overrides = {
            "CHECKPOINT_DIR": run_ckpt_dir,
            "EPOCHS_STAGE4": int(args.short_epochs),
            "GENERATOR_FIXED_ALPHA": bool(conf["fixed_alpha"]),
            "GENERATOR_USE_PREFIX_CONTEXT": bool(conf["use_prefix_context"]),
        }

        with _override_cfg(overrides):
            mods["train"].train_stage4(model, train_dl, val_dl, device)

            # Unified test evaluation for this config.
            pred = _predict_indices(model, dataset, np.asarray(split["test"], dtype=int), mods["train"], mods["eval"], device)
            ns = dataset["norm_stats"]
            x_pred_deg = mods["eval"].denormalize_x(pred["x_pred_norm"], ns)
            x_true_deg = mods["eval"].denormalize_x(pred["x_true_norm"], ns)

            gen_m = mods["stoch"].GeneratorSampleModel(model, device, prefix_len=cfg.STAGE3_PREFIX_LEN)
            gen_m._set_encoder_outputs(
                pred["enc_input"],
                pred["mask"],
                pred["x0"],
                pred["T_K"],
                pred["times_h"],
                prefix_len=cfg.STAGE3_PREFIX_LEN,
            )
            samples = gen_m.sample_trajectories(
                x_pred_deg,
                pred["T_K"],
                pred["times_h"],
                pred["mask"],
                n_samples=args.n_samples,
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                rng=np.random.default_rng(run_seed + 100),
            )
            metrics = gen_m.compute_metrics(
                x_true_deg,
                samples,
                pred["mask"],
                T_K=pred["T_K"],
                prefix_len=cfg.STAGE3_PREFIX_LEN,
                x_mean=x_pred_deg,
                times_h=pred["times_h"],
            )

        # Keep a copy of stage4 checkpoint at a stable name for this config.
        src = os.path.join(run_ckpt_dir, "stage4_best.pt")
        dst = os.path.join(run_ckpt_dir, f"{conf['name']}_stage4_best.pt")
        if os.path.exists(src):
            shutil.copy2(src, dst)

        gen_obj = model.generator
        clip_rate = np.nan
        if hasattr(gen_obj, "_total_count") and float(gen_obj._total_count.item()) > 0.0:
            clip_rate = float(gen_obj._clip_count.item() / gen_obj._total_count.item())

        row = {
            "config": conf["name"],
            "fixed_alpha": conf["fixed_alpha"],
            "use_prefix_context": conf["use_prefix_context"],
            "seed": run_seed,
            "stage4_epochs": int(args.short_epochs),
            "crpss_overall": float(metrics.get("crpss_overall", np.nan)),
            "mace": float(metrics.get("reliability_mace", np.nan)),
            "coverage_50_overall": float(metrics.get("coverage_50_overall", np.nan)),
            "coverage_80_overall": float(metrics.get("coverage_80_overall", np.nan)),
            "coverage_90_overall": float(metrics.get("coverage_90_overall", np.nan)),
            "width_50_overall": float(metrics.get("width_50_overall", np.nan)),
            "width_80_overall": float(metrics.get("width_80_overall", np.nan)),
            "width_90_overall": float(metrics.get("width_90_overall", np.nan)),
            "w1_increments_overall": float(metrics.get("w1_increments_overall", np.nan)),
            "idleak_dec_frac": float(metrics.get("decreasing_fraction_samples", {}).get("IDLeak", np.nan)),
            "igleak_dec_frac": float(metrics.get("decreasing_fraction_samples", {}).get("IGLeak", np.nan)),
            "z0_clip_rate": clip_rate,
            "checkpoint_dir": run_ckpt_dir,
        }
        rows.append(row)
        raw[conf["name"]] = metrics

    rows_sorted = sorted(rows, key=lambda r: float(r["crpss_overall"]), reverse=True)

    csv_path = os.path.join(args.output_dir, "stage4_ablation_table.csv")
    md_path = os.path.join(args.output_dir, "stage4_ablation_table.md")
    pkl_path = os.path.join(args.output_dir, "stage4_ablation_full.pkl")

    fieldnames = list(rows_sorted[0].keys()) if rows_sorted else []
    _write_csv(csv_path, rows_sorted, fieldnames)

    md_cols = [
        "config",
        "fixed_alpha",
        "use_prefix_context",
        "crpss_overall",
        "mace",
        "coverage_90_overall",
        "width_90_overall",
        "w1_increments_overall",
        "idleak_dec_frac",
        "igleak_dec_frac",
        "z0_clip_rate",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_to_md_table(rows_sorted, md_cols))
        f.write("\n")

    with open(pkl_path, "wb") as f:
        pickle.dump({"rows": rows_sorted, "metrics": raw}, f)

    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {pkl_path}")
    if rows_sorted:
        print("Best config by CRPSS:", rows_sorted[0]["config"], rows_sorted[0]["crpss_overall"])


if __name__ == "__main__":
    main()
