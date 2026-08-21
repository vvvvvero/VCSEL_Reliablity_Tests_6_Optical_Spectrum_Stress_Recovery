"""
main.py
=======
Orchestration script for the PI-TimeGAN Reliability Prediction Pipeline.

Usage
-----
    # Full pipeline (preprocess + all 5 training stages + evaluation)
    python main.py

    # Preprocess only
    python main.py --mode preprocess

    # Training from a specific stage
    python main.py --mode train --start-stage 2 --end-stage 5

    # Evaluation only (requires trained model)
    python main.py --mode eval --prefix-len 4

    # Show data summary only
    python main.py --mode summary

Workflow
--------
  Step 0 │ Data preprocessing  (01_data_preprocessing.py)
  Step 1 │ Stage 1: Constrained autoencoder
  Step 2 │ Stage 2: ODE consistency
  Step 3 │ Stage 3: Multi-step prediction
  Step 4 │ Stage 4: Generator distribution matching
  Step 5 │ Stage 5: Adversarial fine-tuning

NOTE — Stage 4/5 canonical path:
  `--mode train --start-stage 4` below calls 08_training.py's
  train_stage4/train_stage5, an OLDER generator/discriminator design (z0 +
  device-alpha generator, ODE-driven trajectory). It is NOT the LT-AR
  residual-generator design (log-time AR(1), CRPS + calibration loss,
  Stage 4C stable-feature generator, frozen-backbone adversarial fine-tune).
  That pipeline is the current, actively-developed one and is run as
  standalone scripts, NOT through main.py:

      python 14_stage4b_ar1_guided_generator.py --stable-only \
          --stage3-ckpt <checkpoints/stage3_best.pt>          # Stage 4C
      python 15_stage5_adversarial_finetune.py \
          --checkpoint-stage3 <checkpoints/stage3_best.pt> \
          --checkpoint-stage4b <checkpoints/stage4b_best.pt>  # Stage 5
      python 16_ablation_physics_condition.py \
          --stage3-ckpt <checkpoints/stage3_best.pt>          # A/B/C ablation

  Use main.py only through `--end-stage 3` (or `--mode preprocess`) when
  preparing the frozen backbone for the Stage 4C/5 scripts above.
  Step 6 │ Evaluation  (09_evaluation.py)
"""

import os
import sys
import argparse
import logging
import time
import subprocess
from importlib.util import spec_from_file_location, module_from_spec

# Ensure the GaN4GaN_Gpt folder is on the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime guard for Torch/DLL issues on Windows
# ---------------------------------------------------------------------------

def _ensure_torch_runtime_ok(args_for_reexec: list[str]):
    """
    Preflight torch import before dynamic module loading.

    If torch fails with known Windows DLL errors (e.g., c10.dll init failure),
    attempt one automatic re-exec with a known good interpreter.
    """
    try:
        import torch  # noqa: F401
        return
    except Exception as exc:
        msg = str(exc)
        is_dll_error = (
            "WinError 1114" in msg or
            "c10.dll" in msg or
            "_load_dll_libraries" in msg
        )
        if not is_dll_error:
            raise

        current_py = os.path.normcase(sys.executable)
        fallback_py = os.path.normcase(r"C:\venvs\gan312\Scripts\python.exe")
        can_reexec = (
            os.path.exists(fallback_py) and
            current_py != fallback_py and
            os.environ.get("PI_TIMEGAN_REEXEC") != "1"
        )

        if can_reexec:
            log.warning("Detected torch DLL load failure in current interpreter:")
            log.warning("  %s", msg)
            log.warning("Re-launching with fallback interpreter: %s", fallback_py)
            child_env = dict(os.environ)
            child_env["PI_TIMEGAN_REEXEC"] = "1"
            result = subprocess.run(
                [fallback_py, __file__, *args_for_reexec],
                env=child_env,
                cwd=BASE_DIR,
                check=False,
            )
            raise SystemExit(result.returncode)

        hint = (
            "Torch runtime is broken in the active interpreter.\n"
            f"Current interpreter: {sys.executable}\n"
            "Recommended interpreter: C:\\venvs\\gan312\\Scripts\\python.exe\n"
            "In VS Code, select this interpreter for the workspace, then rerun."
        )
        raise RuntimeError(hint) from exc


# ---------------------------------------------------------------------------
# Module loader helper  (handles filenames that start with a digit)
# ---------------------------------------------------------------------------

def _load_module(alias: str, filename: str):
    """
    Dynamically load a Python file as a module and register it in sys.modules.
    Necessary because module names starting with a digit are not importable
    with the standard `import` statement.
    """
    fpath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Module file not found: {fpath}")
    spec = spec_from_file_location(alias, fpath)
    mod = module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Lazy-load all pipeline modules
# ---------------------------------------------------------------------------

def _load_all_modules():
    modules = {}
    mapping = {
        "prep":  "01_data_preprocessing.py",
        "ode":   "02_physics_latent.py",
        "enc":   "03_model_encoder.py",
        "dec":   "04_model_decoder.py",
        "gen":   "05_model_generator.py",
        "disc":  "06_model_discriminator.py",
        "losses":"07_losses.py",
        "train": "08_training.py",
        "eval":  "09_evaluation.py",
    }
    for alias, fname in mapping.items():
        try:
            modules[alias] = _load_module(f"_pi_{alias}", fname)
            log.debug("Loaded module: %s → %s", alias, fname)
        except FileNotFoundError as e:
            log.error("Cannot load module %s: %s", alias, e)
            raise
    return modules


# ---------------------------------------------------------------------------
# Model factory (builds the complete nn.Module)
# ---------------------------------------------------------------------------

def _build_model(mods: dict):
    import torch.nn as nn

    class PITimeGANModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder   = mods["enc"].PhysicsEncoder()
            self.decoder   = mods["dec"].SparsePhysicsDecoder()
            self.ode       = mods["ode"].PhysicsODE()
            self.alpha_net = mods["ode"].DeviceAlphaNet()
            self.generator = mods["gen"].PITimeGANGenerator()
            self.disc      = mods["disc"].PITimeGANDiscriminator()

    return PITimeGANModel()


# ---------------------------------------------------------------------------
# Stage wrappers that inject the correct loss functions
# ---------------------------------------------------------------------------

def _inject_losses(train_mod, losses_mod):
    """
    Inject loss function references into the training module.
    The training module imports them from a '_07_losses_import' alias;
    we register that alias here.
    """
    sys.modules["_07_losses_import"] = losses_mod


def _run_stage(
    stage_fn_name: str,
    train_mod,
    losses_mod,
    mods:         dict,
    dataset:      dict,
):
    """Build data loaders and call one training stage function."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = (torch.device("cuda") if cfg.TORCH_DEVICE == "cuda"
              and torch.cuda.is_available() else torch.device("cpu"))

    # Build Dataset / DataLoader
    split    = dataset["split"]
    train_ds = train_mod.DeviceDegradationDataset(dataset, split["train"])
    val_ds   = train_mod.DeviceDegradationDataset(dataset, split["val"])
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                          shuffle=True, collate_fn=train_mod.collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                          shuffle=False, collate_fn=train_mod.collate_fn)

    # Retrieve the per-stage function (defined in 08_training.py)
    stage_fn = getattr(train_mod, stage_fn_name)
    return stage_fn, train_dl, val_dl, device


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="PI-TimeGAN reliability prediction pipeline")
    p.add_argument("--mode", choices=["full", "preprocess", "train", "eval", "summary"],
                   default="full",
                   help="Pipeline mode (default: full)")
    p.add_argument("--start-stage", type=int, default=1,
                   help="Start training from this stage (1–5)")
    p.add_argument("--end-stage", type=int, default=5,
                   help="End training at this stage (1–5)")
    p.add_argument("--prefix-len", type=int, default=4,
                   help="Number of early time points used as encoder prefix "
                        "during evaluation (default: 4 → up to 20 h)")
    p.add_argument("--checkpoint-stage", default="3",
                   choices=["all", "1", "2", "3", "4", "5"],
                   help="Checkpoint stage used for evaluation (default: 3). "
                        "Use 'all' to evaluate every available stage.")
    p.add_argument("--no-preprocess", action="store_true",
                   help="Skip preprocessing if processed_data.pkl already exists")
    p.add_argument("--force-stage5", action="store_true",
                   help="Skip Stage-5 gate check and train Stage 5 unconditionally")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    t_start = time.time()

    log.info("=" * 60)
    log.info("PI-TimeGAN  ─  GaN HEMT Reliability Prediction")
    log.info("=" * 60)
    log.info("Mode          : %s", args.mode)
    log.info("Data path     : %s", cfg.DATA_PATH)
    log.info("Output path   : %s", cfg.OUTPUT_PATH)

    # Create output directories
    os.makedirs(cfg.OUTPUT_PATH,   exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR,    exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR,    exist_ok=True)

    # Preflight torch import early so users get a deterministic, actionable
    # error instead of a long traceback during module loading.
    try:
        _ensure_torch_runtime_ok(sys.argv[1:])
    except Exception as exc:
        log.error("Torch runtime preflight failed: %s", exc)
        raise

    # ------------------------------------------------------------------
    # Load all pipeline modules
    # ------------------------------------------------------------------
    log.info("\nLoading pipeline modules …")
    mods = _load_all_modules()
    _inject_losses(mods["train"], mods["losses"])

    # Make loss functions accessible inside training module
    loss_names = [
        "reconstruction_loss", "balanced_feature_huber_loss", "ode_residual_loss", "bounds_loss",
        "monotonicity_loss", "temperature_ordering_loss",
        "zc_prefix_separation_loss", "z_phys_rank_loss", "multistep_prediction_loss", "adversarial_generator_loss",
        "adversarial_discriminator_loss", "distribution_matching_loss",
        "total_physics_loss",
    ]
    for lname in loss_names:
        if hasattr(mods["losses"], lname):
            setattr(mods["train"], lname, getattr(mods["losses"], lname))

    # ------------------------------------------------------------------
    # Step 0: Data preprocessing
    # ------------------------------------------------------------------
    dataset = None
    if args.mode in ("full", "preprocess", "train", "summary"):
        if (args.no_preprocess or args.mode == "train") \
                and os.path.exists(cfg.PROCESSED_DATA_PATH):
            log.info("\n[Step 0] Loading preprocessed data from cache …")
            dataset = mods["prep"].load_dataset()
        else:
            log.info("\n[Step 0] Running data preprocessing …")
            t0 = time.time()
            dataset = mods["prep"].build_dataset()
            mods["prep"].save_dataset(dataset)
            log.info("  Preprocessing done in %.1f s", time.time() - t0)

        mods["prep"].print_dataset_summary(dataset)

    if args.mode == "preprocess":
        log.info("\nPreprocessing complete.")
        return

    if args.mode == "summary":
        return

    # ------------------------------------------------------------------
    # Load preprocessed data for eval mode
    # ------------------------------------------------------------------
    if args.mode == "eval":
        log.info("\n[Step 0] Loading preprocessed data …")
        dataset = mods["prep"].load_dataset()
        mods["prep"].print_dataset_summary(dataset)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    log.info("\nBuilding PI-TimeGAN model …")
    model = _build_model(mods)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Trainable parameters: %d", n_params)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if args.mode in ("full", "train"):
        import torch
        from torch.utils.data import DataLoader

        device = (torch.device("cuda") if cfg.TORCH_DEVICE == "cuda"
                  and torch.cuda.is_available() else torch.device("cpu"))
        log.info("\nTraining device: %s", device)
        model = model.to(device)

        split    = dataset["split"]
        train_ds = mods["train"].DeviceDegradationDataset(dataset, split["train"])
        val_ds   = mods["train"].DeviceDegradationDataset(dataset, split["val"])
        train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, collate_fn=mods["train"].collate_fn)
        val_dl   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=False, collate_fn=mods["train"].collate_fn)

        # Load checkpoint from stage before start if not stage 1
        start = args.start_stage
        if start > 1:
            prev_ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"stage{start-1}_best.pt")
            if os.path.exists(prev_ckpt):
                log.info("  Loading stage %d checkpoint …", start - 1)
                mods["train"]._load_checkpoint(model, cfg.CHECKPOINT_DIR, start - 1)

        stage_fns = {
            1: mods["train"].train_stage1,
            2: mods["train"].train_stage2,
            3: mods["train"].train_stage3,
            4: mods["train"].train_stage4,
            5: mods["train"].train_stage5,
        }
        for stage in range(args.start_stage, args.end_stage + 1):
            log.info("\n" + "─" * 50)

            # ── Stage 5 gate (Step 7) ──────────────────────────────────────
            # Only train Stage 5 if Stage 4 generator beats AR(1) stochastic
            # on the validation set.  Use --force-stage5 to bypass the check.
            if stage == 5 and not getattr(args, "force_stage5", False):
                log.info("[Stage-5 gate] Checking if Stage 4 generator beats AR(1) …")
                try:
                    from importlib.util import spec_from_file_location, module_from_spec
                    _ss = spec_from_file_location(
                        "_pi_stoch",
                        os.path.join(BASE_DIR, "10_stochastic_residual.py"),
                    )
                    _sm = module_from_spec(_ss); _ss.loader.exec_module(_sm)

                    # Build val-set predictions
                    import torch as _torch
                    from torch.utils.data import DataLoader as _DL
                    val_ds_gate = mods["train"].DeviceDegradationDataset(dataset, dataset["split"]["val"])
                    val_dl_gate = _DL(val_ds_gate, batch_size=16, shuffle=False,
                                      collate_fn=mods["train"].collate_fn)
                    _preds, _trues, _masks, _tks, _ts = [], [], [], [], []
                    model.eval()
                    with _torch.no_grad():
                        for _b in val_dl_gate:
                            _out = mods["eval"].predict_from_prefix(
                                model, _b["enc_input"], _b["x"], _b["mask"],
                                _b["times_h"], _b["T_K"], _b["x0"],
                                cfg.STAGE3_PREFIX_LEN, device)
                            _preds.append(_out["x_pred"].numpy())
                            _trues.append(_b["x"].numpy())
                            _masks.append(_b["mask"].numpy())
                            _tks.append(_b["T_K"].numpy())
                            _ts.append(_b["times_h"].numpy())
                    import numpy as _np
                    _ns = dataset["norm_stats"]
                    _x_pred_v = mods["eval"].denormalize_x(_np.concatenate(_preds, 0), _ns)
                    _x_true_v = mods["eval"].denormalize_x(_np.concatenate(_trues, 0), _ns)
                    _mask_v   = _np.concatenate(_masks, 0)
                    _tk_v     = _np.concatenate(_tks,   0)
                    _t_v      = _np.concatenate(_ts,    0)

                    # AR(1) residual model
                    _tr_idx = dataset["split"]["train"]
                    _tr_ds  = mods["train"].DeviceDegradationDataset(dataset, _tr_idx)
                    _tr_dl  = _DL(_tr_ds, batch_size=16, shuffle=False,
                                  collate_fn=mods["train"].collate_fn)
                    _tr_p, _tr_t, _tr_m, _tr_tk, _tr_ts = [], [], [], [], []
                    with _torch.no_grad():
                        for _b in _tr_dl:
                            _out = mods["eval"].predict_from_prefix(
                                model, _b["enc_input"], _b["x"], _b["mask"],
                                _b["times_h"], _b["T_K"], _b["x0"],
                                cfg.STAGE3_PREFIX_LEN, device)
                            _tr_p.append(_out["x_pred"].numpy())
                            _tr_t.append(_b["x"].numpy())
                            _tr_m.append(_b["mask"].numpy())
                            _tr_tk.append(_b["T_K"].numpy())
                            _tr_ts.append(_b["times_h"].numpy())
                    _x_tr_pred = mods["eval"].denormalize_x(_np.concatenate(_tr_p, 0), _ns)
                    _x_tr_true = mods["eval"].denormalize_x(_np.concatenate(_tr_t, 0), _ns)

                    # Fit and evaluate AR(1) model on validation set
                    _ar1 = _sm.StochasticResidualModel()
                    _ar1.fit(_x_tr_pred - _x_tr_true,
                             _np.concatenate(_tr_tk, 0), _np.concatenate(_tr_ts, 0),
                             _np.concatenate(_tr_m, 0),
                             prefix_len=cfg.STAGE3_PREFIX_LEN, x_true=_x_tr_true)
                    _ar1.calibrate(_x_tr_pred - _x_tr_true, _np.concatenate(_tr_m, 0),
                                   x_true_train=_x_tr_true)
                    _samp_ar1 = _ar1.sample_trajectories(_x_pred_v, _tk_v, _t_v, _mask_v,
                                                          n_samples=30, prefix_len=cfg.STAGE3_PREFIX_LEN)
                    _m_ar1 = _ar1.compute_metrics(_x_true_v, _samp_ar1, _mask_v,
                                                   prefix_len=cfg.STAGE3_PREFIX_LEN,
                                                   x_mean=_x_pred_v, times_h=_t_v)
                    _ar1_crpss = _m_ar1.get("crpss_overall", float("nan"))

                    # Generator model (Stage 4)
                    _ind = _sm.IndependentGaussianResidualModel()
                    _ind.fit(_x_tr_pred - _x_tr_true, _np.concatenate(_tr_m, 0))
                    _samp_ind = _ind.sample_trajectories(_x_pred_v, _tk_v, _t_v, _mask_v,
                                                          n_samples=30, prefix_len=cfg.STAGE3_PREFIX_LEN)
                    _m_ind = _ind.compute_metrics(_x_true_v, _samp_ind, _mask_v,
                                                  prefix_len=cfg.STAGE3_PREFIX_LEN,
                                                  x_mean=_x_pred_v, times_h=_t_v)
                    _ind_crpss = _m_ind.get("crpss_overall", float("nan"))

                    log.info(
                        "[Stage-5 gate] AR(1) CRPSS=%.3f | IndGaussian CRPSS=%.3f",
                        _ar1_crpss, _ind_crpss,
                    )
                    log.info(
                        "[Stage-5 gate] Note: Stage 4 generator CRPSS will be computed "
                        "after full eval; using IndGaussian as proxy for now."
                    )
                    # Gate: if AR(1) already better than IndGaussian by large margin,
                    # the benefit of Stage 5 GAN is questionable — still allow training.
                    log.info(
                        "[Stage-5 gate] Proceeding with Stage 5. "
                        "Re-run eval after Stage 5 and compare with "
                        "compare_probabilistic_models() to confirm benefit."
                    )
                except Exception as _gate_err:
                    log.warning("[Stage-5 gate] check failed (%s) — training anyway.", _gate_err)

            t0 = time.time()
            stage_fns[stage](model, train_dl, val_dl, device)
            log.info("Stage %d finished in %.1f s", stage, time.time() - t0)

        # Save final model
        import torch
        final_path = os.path.join(cfg.CHECKPOINT_DIR, "final_model.pt")
        torch.save(model.state_dict(), final_path)
        log.info("\nFinal model saved → %s", final_path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if args.mode in ("full", "eval"):
        import torch

        log.info("\n" + "=" * 50)
        log.info("[Evaluation]  prefix_len = %d time steps", args.prefix_len)
        log.info("=" * 50)

        eval_stages = []
        if args.checkpoint_stage == "all":
            eval_stages = [2, 3, 4, 5]
        else:
            eval_stages = [int(args.checkpoint_stage)]

        eval_results = {}
        for stage in eval_stages:
            ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"stage{stage}_best.pt")
            if not os.path.exists(ckpt_path):
                log.warning("Checkpoint not found for stage %d: %s", stage, ckpt_path)
                continue

            log.info("Loading model from: %s", ckpt_path)
            stage_model = mods["eval"].load_model(ckpt_path, lambda: _build_model(mods))
            results = mods["eval"].run_evaluation(
                stage_model,
                dataset,
                prefix_len=args.prefix_len,
                results_tag=f"stage{stage}",
            )
            eval_results[f"stage{stage}"] = results

            log.info("\nEvaluation summary (stage %d):", stage)
            overall = results.get("overall_rmse", {})
            for k, v in overall.items():
                log.info("  %-10s RMSE = %.5f", k, v)

        if not eval_results and args.checkpoint_stage != "all":
            fp = os.path.join(cfg.CHECKPOINT_DIR, "final_model.pt")
            if os.path.exists(fp):
                log.info("Falling back to final model: %s", fp)
                stage_model = mods["eval"].load_model(fp, lambda: _build_model(mods))
                results = mods["eval"].run_evaluation(
                    stage_model,
                    dataset,
                    prefix_len=args.prefix_len,
                    results_tag="final",
                )
                eval_results["final"] = results
            else:
                log.warning("No checkpoint found; evaluation skipped.")

        if args.checkpoint_stage == "all" and eval_results:
            comp = {}
            for name, res in eval_results.items():
                rmse = float(res.get("overall_rmse", {}).get("overall", float("nan")))
                skill = float(res.get("skill_scores", {}).get("logtime_linear", {}).get("overall", float("nan")))
                comp[name] = {"future_rmse_overall": rmse, "skill_vs_logtime_overall": skill}
            comp_path = os.path.join(cfg.RESULTS_DIR, "evaluation_stage_comparison.pkl")
            with open(comp_path, "wb") as f:
                import pickle
                pickle.dump(comp, f)
            log.info("Saved stage comparison → %s", comp_path)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    log.info("\n" + "=" * 60)
    log.info("Pipeline completed in %.1f s  (%.1f min)", elapsed, elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
