"""
post_stage3_pipeline.py
========================
Automated post-Stage3 pipeline:
  1. Waits until PID of Stage3 retrain exits (or detects new checkpoint is final)
  2. Runs diag_state.py → parses zC(0) values
  3. If zC(0) < ZC_THRESHOLD, launches Stage 4A training
  4. If zC(0) >= threshold, logs warning but proceeds anyway (soft gate)

Usage:
    python post_stage3_pipeline.py --stage3-pid 35356 --zc-threshold 0.25
"""

import argparse
import os
import subprocess
import sys
import time
import re
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable

ZC_THRESHOLD_SOFT = 0.25   # warn if above, but still proceed
ZC_THRESHOLD_HARD = 0.40   # abort if above (zC still too high, retrain needed)
POLL_INTERVAL_S   = 60     # seconds between polling for Stage3 PID


def _is_pid_alive(pid: int) -> bool:
    """Check if a PID is still running (Windows-compatible)."""
    try:
        import ctypes
        PROCESS_QUERY_INFORMATION = 0x0400
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong(0)
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return exit_code.value == STILL_ACTIVE
    except Exception:
        # Fallback: try os.kill (raises OSError if dead)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def wait_for_pid(pid: int, log_file: str = None):
    """Poll until the given PID exits."""
    print(f"[post_pipeline] Waiting for Stage3 PID {pid} to finish ...", flush=True)
    waited = 0
    while True:
        if not _is_pid_alive(pid):
            print(f"[post_pipeline] PID {pid} exited after ~{waited//60}m {waited%60}s", flush=True)
            return
        # Show latest checkpoint as heartbeat
        if log_file and os.path.exists(log_file) and waited % 300 == 0 and waited > 0:
            try:
                lines = open(log_file, encoding="utf-8", errors="replace").readlines()
                last = [l.rstrip() for l in lines if "mse" in l.lower() or "saved" in l.lower()]
                if last:
                    print(f"  [heartbeat] {last[-1]}", flush=True)
            except Exception:
                pass
        time.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S


def run_diag_state() -> dict:
    """
    Directly inspect Stage 3 checkpoint: load model, run encoder on all devices,
    compute z(0) statistics. Faster and more reliable than running diag_state.py.
    """
    import pickle
    from importlib.util import spec_from_file_location, module_from_spec

    def _lm(alias, filename):
        path = os.path.join(BASE_DIR, filename)
        spec = spec_from_file_location(alias, path)
        mod  = module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
        return mod

    print("[post_pipeline] Loading pipeline modules for z0 inspection ...", flush=True)
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        import config as cfg_m

        ode_m  = _lm("_pi_ode_d",  "02_physics_latent.py")
        enc_m  = _lm("_pi_enc_d",  "03_model_encoder.py")
        dec_m  = _lm("_pi_dec_d",  "04_model_decoder.py")
        gen_m  = _lm("_pi_gen_d",  "05_model_generator.py")
        disc_m = _lm("_pi_disc_d", "06_model_discriminator.py")
        train_m = _lm("_pi_train_d","08_training.py")
        sys.modules["_07_losses_import"] = _lm("_pi_losses_d", "07_losses.py")

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder   = enc_m.PhysicsEncoder()
                self.decoder   = dec_m.SparsePhysicsDecoder()
                self.ode       = ode_m.PhysicsODE()
                self.alpha_net = ode_m.DeviceAlphaNet()
                self.generator = gen_m.PITimeGANGenerator()
                self.disc      = disc_m.PITimeGANDiscriminator()

        model = _Model()
        ckpt_path = os.path.join(cfg_m.CHECKPOINT_DIR, "stage3_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[post_pipeline] ERROR: Stage3 checkpoint not found: {ckpt_path}", flush=True)
            return {}

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = {k: v for k, v in ckpt["model_state"].items() if k != "decoder.mask"}
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"[post_pipeline] Loaded Stage3 ckpt (metric={ckpt.get('selection_value', float('nan')):.5f})", flush=True)

        # Load preprocessed dataset
        prep_path = cfg_m.PROCESSED_DATA_PATH
        with open(prep_path, "rb") as fh:
            dataset = pickle.load(fh)

        all_idx  = list(range(len(dataset["device_ids"])))
        ds = train_m.DeviceDegradationDataset(dataset, all_idx)
        dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=train_m.collate_fn)

        z0_all = []
        with torch.no_grad():
            for batch in dl:
                z_enc, alpha, _, _, _, _, _, _, _ = train_m._forward(model, batch, torch.device("cpu"))
                z0_all.append(z_enc[:, 0, :].cpu().numpy())  # (B, 5)

        z0 = np.concatenate(z0_all, axis=0)   # (N, 5)
        names = ["zG", "zB", "zM", "zL", "zC"]
        print("\n[post_pipeline] === z(0) statistics (all devices) ===", flush=True)
        for i, name in enumerate(names):
            print(f"  {name}(0): mean={z0[:, i].mean():.4f}  std={z0[:, i].std():.4f}  max={z0[:, i].max():.4f}", flush=True)

        result = {
            "zM0": float(z0[:, 2].mean()),
            "zL0": float(z0[:, 3].mean()),
            "zC0": float(z0[:, 4].mean()),
            "stage3_mse": float(ckpt.get("selection_value", float("nan"))),
        }
        print(f"\n[post_pipeline] Key diagnostics: zM(0)={result['zM0']:.4f}  zC(0)={result['zC0']:.4f}", flush=True)
        return result

    except Exception as exc:
        print(f"[post_pipeline] Direct inspection failed: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return {}


def run_stage4a(n_eval_samples: int = 100):
    """Launch Stage 4A Residual Latent Innovation Generator training."""
    script = os.path.join(BASE_DIR, "13_stage4a_residual_generator.py")
    log    = os.path.join(os.path.dirname(BASE_DIR), "output", "pi_timegan", "results",
                          "stage4a_training.log")
    log_err = log.replace(".log", ".err")
    os.makedirs(os.path.dirname(log), exist_ok=True)

    print(f"[post_pipeline] Launching Stage 4A training ...", flush=True)
    print(f"  Script: {script}", flush=True)
    print(f"  Log:    {log}", flush=True)

    proc = subprocess.Popen(
        [PYTHON, script, "--n-eval-samples", str(n_eval_samples)],
        cwd=BASE_DIR,
        stdout=open(log, "w", encoding="utf-8"),
        stderr=open(log_err, "w", encoding="utf-8"),
    )
    print(f"[post_pipeline] Stage 4A started with PID {proc.pid}", flush=True)
    print(f"  Monitor: Get-Content '{log_err}' -Wait", flush=True)
    return proc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage3-pid",    type=int,   default=None,
                   help="PID of Stage3 training process to wait for. If omitted, runs immediately.")
    p.add_argument("--zc-threshold",  type=float, default=ZC_THRESHOLD_SOFT,
                   help="Soft zC(0) threshold; warn if above (default 0.25)")
    p.add_argument("--zc-hard-abort", type=float, default=ZC_THRESHOLD_HARD,
                   help="Hard zC(0) threshold; abort if above (default 0.40)")
    p.add_argument("--skip-diag",     action="store_true",
                   help="Skip diag_state.py and proceed directly to Stage 4A")
    p.add_argument("--stage3-log",    type=str,   default=None,
                   help="Path to Stage3 err log for heartbeat monitoring")
    p.add_argument("--n-eval-samples", type=int,  default=100)
    args = p.parse_args()

    stage3_log = args.stage3_log or os.path.join(
        os.path.dirname(BASE_DIR), "output", "pi_timegan", "results", "stage3_zc_retrain.err"
    )

    # Step 1: Wait for Stage 3
    if args.stage3_pid is not None:
        wait_for_pid(args.stage3_pid, log_file=stage3_log)
        # Give it 30s for final checkpoint flush
        print("[post_pipeline] Waiting 30s for checkpoint flush ...", flush=True)
        time.sleep(30)
    else:
        print("[post_pipeline] No Stage3 PID provided — proceeding immediately.", flush=True)

    # Step 2: Run diagnostics
    if not args.skip_diag:
        diag = run_diag_state()
        zC0  = diag.get("zC0", None)
        zM0  = diag.get("zM0", None)

        print("\n[post_pipeline] === Diagnostic Gate ===", flush=True)
        if zC0 is None:
            print("  WARNING: Could not parse zC(0) from diag_state output.", flush=True)
            print("  Proceeding to Stage 4A anyway.", flush=True)
        elif zC0 >= args.zc_hard_abort:
            print(f"  ABORT: zC(0) = {zC0:.3f} >= hard threshold {args.zc_hard_abort}", flush=True)
            print("  Stage 3 retrain did not sufficiently reduce zC(0).", flush=True)
            print("  Consider: increasing LAMBDA_ZC_ANCHOR_BOOST further (try 3.0).", flush=True)
            sys.exit(1)
        elif zC0 >= args.zc_threshold:
            print(f"  WARN: zC(0) = {zC0:.3f} (above soft threshold {args.zc_threshold})", flush=True)
            print("  Proceeding to Stage 4A (zC not ideal but acceptable).", flush=True)
        else:
            print(f"  OK: zC(0) = {zC0:.3f} < {args.zc_threshold} ✓", flush=True)

        if zM0 is not None:
            status = "✓" if zM0 < 0.10 else "WARN"
            print(f"  zM(0) = {zM0:.3f} {status}", flush=True)
    else:
        print("[post_pipeline] Skipping diag_state.py (--skip-diag)", flush=True)

    # Step 3: Launch Stage 4A
    proc = run_stage4a(n_eval_samples=args.n_eval_samples)
    print(f"\n[post_pipeline] Stage 4A running (PID={proc.pid}).", flush=True)
    print(f"  Progress: gc 'D:\\2026\\article\\GaN4GaN\\output\\pi_timegan\\results\\stage4a_training.err' | select -last 20", flush=True)


if __name__ == "__main__":
    main()
