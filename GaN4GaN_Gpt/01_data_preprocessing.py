"""
01_data_preprocessing.py
========================
Data loading, cleaning, feature engineering, normalisation, and sequence
construction for the PI-TimeGAN reliability pipeline.

Data format (thermalstorage_Data folder)
-----------------------------------------
Files:  {DeviceType}_{Temp}_{Parameter}[Suffix].xlsx
  * DeviceType : A2ACH4FP | A2ACH4 | A8A | P10C
  * Temp       : 275 | 300 | 325  (degrees Celsius)
  * Parameter  : IDSS | Vth | IDLeak | IGLeak | RON | gmmax
  * Suffix     : empty, or letter A/B/C/D, or '  (V)', etc.
  * Files whose name contains '%' are percentage-change files → skipped.

Inside each xlsx
  Row 0 : unit header (h, NaN …)
  Row 1+: time points in hours (0 or ~0.2, 1, 2, 5, 10, 20, 50, 100, … 2000)
  Col 0 : 'Storage Time'
  Col 1+: device identifiers (F2, F3, G2, …)
  Missing entries: NaN or the string '--'

Output
------
A dict saved as a pickle:
  data['x']          (N, T, 6)  float32  – degradation features, NaN where missing
    data['feature_mask'] (N, T, 6) bool    – True where feature value is valid
    data['mask']       (N, T)     bool     – True where at least one feature is valid
  data['times_h']    (N, T)     float32  – measurement times [h]
  data['T_K']        (N,)       float32  – storage temperature [K]
  data['device_ids'] list[str]           – unique device identifier
  data['device_types'] list[str]
  data['x0_static']  (N, 6)    float32  – initial absolute parameter values
  data['x0_vth_sdev'] float             – std of ΔVth across training set
  data['norm_stats']  dict               – min/max per feature for Min-Max norm
  data['split']       dict               – {'train':idx, 'val':idx, 'test':idx}

Feature definitions (x1…x6)
  x1 = ΔVth / s_Vth               (Vth shift; sign free)
  x2 = -log((IDS+ε)/(IDS0+ε))     (IDSS degradation; ≥ 0 expected)
  x3 =  log((RON+ε)/(RON0+ε))     (RON increase; ≥ 0 expected)
  x4 = -log((gm +ε)/(gm0 +ε))     (gm   decrease; ≥ 0 expected)
  x5 =  log((|IDSoff|+ε)/(|IDSoff0|+ε))  (leakage increase)
  x6 =  log((|IGoff| +ε)/(|IGoff0| +ε))  (gate leakage increase)

Usage
-----
    python 01_data_preprocessing.py          # full run, saves pickle
    from 01_data_preprocessing import load_dataset
"""

import os
import re
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _scan_files(data_path: str) -> pd.DataFrame:
    """
    Scan the data folder and parse filenames into a catalogue DataFrame.
    Columns: device_type, temp_c, param_key, filepath, has_percent
    """
    rows = []
    for fname in os.listdir(data_path):
        if not fname.lower().endswith(".xlsx"):
            continue
        fpath = os.path.join(data_path, fname)
        stem = fname[:-5]                    # remove .xlsx
        has_pct = "%" in stem

        # Match:  {DeviceType}_{Temp}_{rest}
        m = re.match(r"^([A-Za-z0-9]+?)_(\d{3})_(.+)$", stem)
        if m is None:
            continue
        dev_type, temp_str, rest = m.group(1), m.group(2), m.group(3)
        temp_c = int(temp_str)

        # Map rest to a canonical parameter key
        param_key = _classify_param(rest)
        if param_key is None:
            continue

        rows.append(dict(
            device_type=dev_type,
            temp_c=temp_c,
            param_key=param_key,
            has_percent=has_pct,
            rest=rest,
            filepath=fpath,
        ))

    return pd.DataFrame(rows)


def _classify_param(rest: str) -> Optional[str]:
    """Map the filename tail to a canonical parameter key."""
    rest_lower = rest.lower()
    if "idleak" in rest_lower or "idleak" in rest_lower:
        return "IDLeak"
    if "igleak" in rest_lower or "igleak" in rest_lower:
        return "IGLeak"
    if "idss" in rest_lower:
        return "IDSS"
    if "vth" in rest_lower:
        return "Vth"
    if "ron" in rest_lower:
        return "RON"
    if "gmmax" in rest_lower or "gm_max" in rest_lower:
        return "gmmax"
    return None


def _pick_best_file(group: pd.DataFrame) -> Optional[str]:
    """
    From a group of files all sharing (device_type, temp_c, param_key),
    pick the best non-percentage file.

    Priority:
      1. Non-% file with a unit indicator like '(V)' (preferred: explicit
         and unambiguous about what quantity is stored)
      2. Non-% plain file (shortest rest, no trailing single-letter series tag)
      3. Non-% file with a single trailing-letter series tag (…A, …B, …C, …D)
         — these are alternate/duplicate measurement runs, lowest priority
    """
    non_pct = group[~group["has_percent"]]
    if non_pct.empty:
        return None

    def _is_series_tag(row):
        # A true series tag is a SINGLE trailing uppercase letter directly
        # appended to the parameter name (VthB, VthC, IDSSD, …). Do NOT match
        # multi-letter trailing runs like "...A2AFP" or "...P10C" — those are
        # device-type suffixes baked into the filename, not a series marker,
        # and mistaking them for one caused a real unit-file to lose to a
        # duplicate absolute-value file (e.g. A2ACH4FP_325_Vth (V)_A2AFP.xlsx
        # losing to A2ACH4FP_325_VthB.xlsx).
        m = re.search(r"[A-Z]$", row["rest"])
        if m is None:
            return 0
        # Only count it if the character before the final letter is NOT
        # itself uppercase (i.e. exactly one trailing capital, not a run).
        rest = row["rest"]
        if len(rest) >= 2 and rest[-2].isupper():
            return 0
        return 1

    def _has_unit(row):
        # Files with '(V)' or similar unit labels are more reliable
        return int("(" in row["rest"])

    non_pct = non_pct.copy()
    non_pct["_is_series"] = non_pct.apply(_is_series_tag, axis=1)
    non_pct["_has_unit"]  = non_pct.apply(_has_unit,  axis=1)
    non_pct["_rest_len"]  = non_pct["rest"].str.len()

    # Sort: unit-labelled files first, then non-series-tag files, then
    # shorter rest (more basic name) as a tiebreaker.
    non_pct = non_pct.sort_values(
        ["_has_unit", "_is_series", "_rest_len"],
        ascending=[False, True, True],
    )
    return non_pct.iloc[0]["filepath"]


# ---------------------------------------------------------------------------
# Single-file reader
# ---------------------------------------------------------------------------

def _read_param_file(fpath: str) -> pd.DataFrame:
    """
    Read one parameter xlsx and return a tidy DataFrame:
      columns: time_h (float), device_id (str), value (float or NaN)
    """
    df_raw = pd.read_excel(fpath, header=0)

    # Row 0 is the unit row (h, NaN …) – drop it
    df_raw = df_raw.iloc[1:].reset_index(drop=True)

    time_col = df_raw.columns[0]
    device_cols = [c for c in df_raw.columns if c != time_col]

    # Parse time column (may contain strings like '--')
    def _to_float(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else np.nan
        except (ValueError, TypeError):
            return np.nan

    times = df_raw[time_col].apply(_to_float).values  # (T,)

    records = []
    for dev_col in device_cols:
        dev_id = str(dev_col).strip()
        for i, t in enumerate(times):
            if np.isnan(t):
                continue
            raw_val = df_raw.iloc[i][dev_col]
            val = _to_float(raw_val)
            records.append({"time_h": t, "device_id": dev_id, "value": val})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Build raw parameter table for one (device_type, temp_c)
# ---------------------------------------------------------------------------

def _build_raw_table(
        catalogue: pd.DataFrame,
        device_type: str,
        temp_c: int,
) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict  param_key → tidy DataFrame(time_h, device_id, value)
    """
    mask = (catalogue["device_type"] == device_type) & (catalogue["temp_c"] == temp_c)
    sub = catalogue[mask]

    result = {}
    for param_key in cfg.FEATURES:
        group = sub[sub["param_key"] == param_key]
        fpath = _pick_best_file(group)
        if fpath is None:
            log.warning("  No file found for %s %d°C %s", device_type, temp_c, param_key)
            result[param_key] = pd.DataFrame(columns=["time_h", "device_id", "value"])
        else:
            log.debug("  Loading %s", os.path.basename(fpath))
            result[param_key] = _read_param_file(fpath)
    return result


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def _remove_outliers(series: np.ndarray, ref_val: float) -> np.ndarray:
    """
    Replace values that deviate from ref_val by more than
    OUTLIER_RATIO_THRESHOLD (×ref_val) with NaN.
    Also removes negative IDSS / RON / gmmax values.
    """
    out = series.copy().astype(float)
    if np.isnan(ref_val) or ref_val == 0:
        return out
    ratio = np.abs(out / (ref_val + 1e-30))
    bad = ratio > (1 + cfg.OUTLIER_RATIO_THRESHOLD)
    out[bad] = np.nan
    return out


# ---------------------------------------------------------------------------
# Assemble per-device time series
# ---------------------------------------------------------------------------

def _get_reference_value(values: np.ndarray,
                          times: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Get the reference value at t=0 (or the first valid measurement ≤ 1 h).
    Returns (ref_value, values_after_outlier_removal).
    """
    # Find the first valid measurement at t ≤ 1 h
    ref_val = np.nan
    early_mask = times <= 1.0
    early_vals = values[early_mask]
    if len(early_vals) > 0:
        valid_early = early_vals[~np.isnan(early_vals)]
        if len(valid_early) > 0:
            ref_val = valid_early[0]
    return ref_val


def _compute_degradation(
        param: str,
        values: np.ndarray,
        ref_val: float,
        vth_sdev: float = 1.0,
    leakage_floor_map: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute degradation variable for one device, one parameter.
    Returns NaN where input is NaN.
    """
    eps = cfg.EPSILON
    x = np.full_like(values, np.nan, dtype=float)

    valid = ~np.isnan(values)
    v = values[valid]

    if param == "Vth":
        # Vth file may already store ΔVth (reference ≈ 0) or absolute Vth.
        # Threshold 0.5 V: if |ref| < 0.5 treat as already-relative.
        if np.isnan(ref_val) or abs(ref_val) < 0.5:
            delta_vth = v   # already relative
        else:
            delta_vth = v - ref_val
        # Physical range guard: clamp extreme outliers before normalisation
        delta_vth = np.clip(delta_vth, -10.0, 10.0)
        x[valid] = delta_vth / (vth_sdev + eps)

    elif param == "IDSS":
        if np.isnan(ref_val) or ref_val <= 0:
            pass
        else:
            x[valid] = -np.log((np.abs(v) + eps) / (np.abs(ref_val) + eps))

    elif param == "RON":
        if np.isnan(ref_val) or ref_val <= 0:
            pass
        else:
            x[valid] = np.log((np.abs(v) + eps) / (np.abs(ref_val) + eps))

    elif param == "gmmax":
        if np.isnan(ref_val) or ref_val <= 0:
            pass
        else:
            x[valid] = -np.log((np.abs(v) + eps) / (np.abs(ref_val) + eps))

    elif param in ("IDLeak", "IGLeak"):
        if np.isnan(ref_val):
            pass
        else:
            floor = cfg.LEAKAGE_FLOOR_DEFAULT
            if leakage_floor_map is not None and param in leakage_floor_map:
                floor = float(leakage_floor_map[param])
            x[valid] = np.log(
                (np.maximum(np.abs(v), floor) + eps)
                / (np.maximum(np.abs(ref_val), floor) + eps)
            )
            x[valid] = np.clip(x[valid], -cfg.LEAKAGE_LOG_CLIP, cfg.LEAKAGE_LOG_CLIP)

    return x


def _estimate_leakage_floors(
        all_records: List[dict],
        train_idx: np.ndarray,
) -> Dict[str, float]:
    """Estimate leakage detection floors from training split only."""
    floors: Dict[str, float] = {}
    for pname in ("IDLeak", "IGLeak"):
        vals = []
        for rec_i in train_idx:
            arr = all_records[int(rec_i)]["raw"][pname]
            arr = np.abs(arr[np.isfinite(arr)])
            if arr.size > 0:
                vals.append(arr)
        if vals:
            merged = np.concatenate(vals)
            p = float(np.percentile(merged, cfg.LEAKAGE_FLOOR_PERCENTILE))
            floor = max(cfg.LEAKAGE_FLOOR_MIN, p)
        else:
            floor = cfg.LEAKAGE_FLOOR_DEFAULT
        floors[pname] = float(floor)
    return floors


def stratified_device_split(
        device_types: np.ndarray,
        temperatures_k: np.ndarray,
        train_frac: float,
        val_frac: float,
        seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratify split by (device type, storage temperature)."""
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    strata: Dict[Tuple[str, int], List[int]] = {}
    for i, (device_type, temp_k) in enumerate(zip(device_types, temperatures_k)):
        temp_c = int(round(float(temp_k - cfg.CELSIUS_TO_KELVIN)))
        key = (str(device_type), temp_c)
        strata.setdefault(key, []).append(i)

    for _, group_indices in strata.items():
        group_indices = np.asarray(group_indices, dtype=int)
        rng.shuffle(group_indices)
        n = len(group_indices)

        if n == 1:
            train_indices.extend(group_indices.tolist())
            continue
        if n == 2:
            train_indices.append(int(group_indices[0]))
            test_indices.append(int(group_indices[1]))
            continue

        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_indices.extend(group_indices[:n_train].tolist())
        val_indices.extend(group_indices[n_train:n_train + n_val].tolist())
        test_indices.extend(group_indices[n_train + n_val:].tolist())

    train_idx = np.asarray(train_indices, dtype=int)
    val_idx = np.asarray(val_indices, dtype=int)
    test_idx = np.asarray(test_indices, dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def log_split_distribution(
        split_name: str,
        indices: np.ndarray,
        device_types: np.ndarray,
        temperatures_k: np.ndarray,
):
    """Log per-split composition by temperature and device type."""
    log.info("%s split: n=%d", split_name, len(indices))
    for temp_c in cfg.TEMPERATURES_C:
        temp_k = temp_c + cfg.CELSIUS_TO_KELVIN
        count = int(np.sum(np.abs(temperatures_k[indices] - temp_k) < 1.0))
        log.info("  %dC: %d devices", temp_c, count)
    for device_type in cfg.DEVICE_TYPES:
        count = sum(device_types[i] == device_type for i in indices)
        log.info("  %-10s: %d devices", device_type, count)

    # Distribution over joint strata: device_type x temperature.
    for device_type in cfg.DEVICE_TYPES:
        for temp_c in cfg.TEMPERATURES_C:
            temp_k = temp_c + cfg.CELSIUS_TO_KELVIN
            cnt = int(np.sum(
                (device_types[indices] == device_type) &
                (np.abs(temperatures_k[indices] - temp_k) < 1.0)
            ))
            log.info("  %-10s @ %3dC: %d devices", device_type, temp_c, cnt)


def log_split_feature_stats(
        split_name: str,
        indices: np.ndarray,
        x_raw_deg: np.ndarray,
        s_vth: float,
):
    """Log per-split degradation statistics requested for diagnostics."""
    log.info("%s split feature stats:", split_name)
    for fi, fname in enumerate(cfg.FEATURES):
        vals = x_raw_deg[indices, :, fi].reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            log.info("  %-8s mean=nan std=nan", fname)
            continue
        log.info("  %-8s mean=% .5f std=% .5f", fname, float(np.mean(vals)), float(np.std(vals)))

    vth_delta = (x_raw_deg[indices, :, 0] * s_vth).reshape(-1)
    vth_delta = vth_delta[np.isfinite(vth_delta)]
    if vth_delta.size == 0:
        log.info("  VthΔ   mean=nan std=nan p5=nan p95=nan")
    else:
        log.info(
            "  VthΔ   mean=% .5f std=% .5f p5=% .5f p95=% .5f",
            float(np.mean(vth_delta)),
            float(np.std(vth_delta)),
            float(np.percentile(vth_delta, 5)),
            float(np.percentile(vth_delta, 95)),
        )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset() -> dict:
    """
    Full pipeline: discover files → parse → compute degradation variables →
    outlier removal → split → normalization → sequence construction.

    Returns the dataset dict described at the top of this module.
    """
    if not os.path.exists(cfg.DATA_PATH):
        raise FileNotFoundError(
            f"Data path not found: {cfg.DATA_PATH}. "
            "Update config.DATA_PATH or point to a local copy of the raw GaN4GaN Excel files."
        )

    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)

    log.info("Scanning data folder: %s", cfg.DATA_PATH)
    catalogue = _scan_files(cfg.DATA_PATH)
    log.info("  Found %d xlsx files", len(catalogue))

    canonical_times = np.array(cfg.TIME_POINTS_H, dtype=float)  # (13,)
    T = len(canonical_times)

    all_records = []   # list of dicts, one per device

    for dev_type in cfg.DEVICE_TYPES:
        for temp_c in cfg.TEMPERATURES_C:
            log.info("Processing %s @ %d °C …", dev_type, temp_c)
            tables = _build_raw_table(catalogue, dev_type, temp_c)

            # Collect all device IDs that appear in any parameter file
            all_dev_ids = set()
            for pname in cfg.FEATURES:
                df = tables[pname]
                all_dev_ids.update(df["device_id"].unique())

            for dev_id in sorted(all_dev_ids):
                record = {
                    "device_uid": f"{dev_type}_{temp_c}_{dev_id}",
                    "device_type": dev_type,
                    "temp_c": temp_c,
                    "T_K": temp_c + cfg.CELSIUS_TO_KELVIN,
                    "raw": {},  # param → array aligned to canonical_times
                    "ref": {},  # param → reference value
                }

                has_enough = False
                for pname in cfg.FEATURES:
                    df_dev = tables[pname][tables[pname]["device_id"] == dev_id]

                    # Align to canonical times (nearest match within 0.6 h)
                    arr = np.full(T, np.nan, dtype=float)
                    for _, row_d in df_dev.iterrows():
                        diffs = np.abs(canonical_times - row_d["time_h"])
                        best = np.argmin(diffs)
                        if diffs[best] <= 0.6:
                            arr[best] = row_d["value"]

                    # Outlier removal
                    times = canonical_times
                    ref_v = _get_reference_value(arr, times)
                    arr = _remove_outliers(arr, ref_v)

                    record["raw"][pname] = arr
                    record["ref"][pname] = ref_v

                    if pname == "IDSS" and np.sum(~np.isnan(arr)) >= cfg.MIN_VALID_TIMEPOINTS:
                        has_enough = True

                if has_enough or any(
                        np.sum(~np.isnan(record["raw"][p])) >= cfg.MIN_VALID_TIMEPOINTS
                        for p in cfg.FEATURES
                ):
                    all_records.append(record)

    log.info("Total devices after filtering: %d", len(all_records))
    if len(all_records) == 0:
        raise RuntimeError("No valid device records found. Check DATA_PATH and file formats.")

    # -----------------------------------------------------------------------
    # Metadata first (used for stratified split and train-only scaling stats)
    # -----------------------------------------------------------------------
    N = len(all_records)
    device_types_arr = np.array([rec["device_type"] for rec in all_records], dtype=object)
    T_K_meta = np.array([rec["T_K"] for rec in all_records], dtype=np.float32)

    train_idx, val_idx, test_idx = stratified_device_split(
        device_types=device_types_arr,
        temperatures_k=T_K_meta,
        train_frac=cfg.TRAIN_FRAC,
        val_frac=cfg.VAL_FRAC,
        seed=cfg.RANDOM_SEED,
    )
    log_split_distribution("Train", train_idx, device_types_arr, T_K_meta)
    log_split_distribution("Val", val_idx, device_types_arr, T_K_meta)
    log_split_distribution("Test", test_idx, device_types_arr, T_K_meta)

    # -----------------------------------------------------------------------
    # Compute train-only preprocessing statistics (avoid split leakage)
    # -----------------------------------------------------------------------
    vth_deltas = []
    for rec_i in train_idx:
        rec = all_records[int(rec_i)]
        arr = rec["raw"]["Vth"]
        ref = rec["ref"]["Vth"]
        if np.isnan(ref):
            continue
        vals = arr[~np.isnan(arr)] if abs(ref) < 0.5 else (arr[~np.isnan(arr)] - ref)
        vals = vals[(np.abs(vals) < 5.0)]
        vth_deltas.extend(vals.tolist())

    if len(vth_deltas) > 10:
        vth_arr = np.array(vth_deltas)
        q25, q75 = np.percentile(vth_arr, [25, 75])
        iqr = q75 - q25
        s_vth = max(float(iqr * 1.4826), 0.01)
    else:
        s_vth = 0.1
    log.info("s_Vth (train-only robust IQR-based std) = %.4f V", s_vth)

    leakage_floor_map = _estimate_leakage_floors(all_records, train_idx)
    log.info(
        "Leakage floors (train-only): IDLeak=%.3e IGLeak=%.3e | clip=±%.2f",
        leakage_floor_map.get("IDLeak", float("nan")),
        leakage_floor_map.get("IGLeak", float("nan")),
        cfg.LEAKAGE_LOG_CLIP,
    )

    # -----------------------------------------------------------------------
    # Build degradation feature arrays  x: (N, T, 6)
    # -----------------------------------------------------------------------
    x_raw = np.full((N, T, cfg.FEATURE_DIM), np.nan, dtype=np.float32)
    x0_static = np.full((N, cfg.FEATURE_DIM), np.nan, dtype=np.float32)
    times_arr = np.tile(canonical_times, (N, 1)).astype(np.float32)
    T_K_arr = np.zeros(N, dtype=np.float32)
    device_ids = []
    device_types = []

    for i, rec in enumerate(all_records):
        T_K_arr[i] = rec["T_K"]
        device_ids.append(rec["device_uid"])
        device_types.append(rec["device_type"])

        for fi, pname in enumerate(cfg.FEATURES):
            arr = rec["raw"][pname]
            ref = rec["ref"][pname]
            x0_static[i, fi] = ref if not np.isnan(ref) else 0.0
            deg = _compute_degradation(
                pname,
                arr,
                ref,
                vth_sdev=s_vth,
                leakage_floor_map=leakage_floor_map,
            )
            x_raw[i, :, fi] = deg.astype(np.float32)

    feature_mask = np.isfinite(x_raw)     # (N, T, 6)
    mask = feature_mask.any(axis=2)       # (N, T) True where ≥1 feature valid

    # Requested split diagnostics in transformed degradation space.
    log_split_feature_stats("Train", train_idx, x_raw, s_vth)
    log_split_feature_stats("Val", val_idx, x_raw, s_vth)
    log_split_feature_stats("Test", test_idx, x_raw, s_vth)

    # -----------------------------------------------------------------------
    # Min-Max normalisation fitted on training set only
    # -----------------------------------------------------------------------
    norm_stats = {}
    x_norm = x_raw.copy()
    for fi in range(cfg.FEATURE_DIM):
        col_valid = x_raw[train_idx, :, fi].reshape(-1)
        col_valid = col_valid[np.isfinite(col_valid)]
        if len(col_valid) < 2:
            lo, hi = 0.0, 1.0
        else:
            lo = float(np.percentile(col_valid, 1))
            hi = float(np.percentile(col_valid, 99))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        norm_stats[cfg.FEATURES[fi]] = {"min": lo, "max": hi}
        x_norm[:, :, fi] = (x_raw[:, :, fi] - lo) / (hi - lo)

    for leak_name in ("IDLeak", "IGLeak"):
        fi = cfg.FEATURES.index(leak_name)
        train_vals = x_raw[train_idx, :, fi].reshape(-1)
        train_vals = train_vals[np.isfinite(train_vals)]
        if train_vals.size > 0:
            p1, p50, p99 = np.percentile(train_vals, [1, 50, 99])
            st = norm_stats[leak_name]
            log.info(
                "%s transformed stats (train): min=%.4f p1=%.4f p50=%.4f p99=%.4f max=%.4f",
                leak_name,
                st["min"],
                float(p1),
                float(p50),
                float(p99),
                st["max"],
            )

    # -----------------------------------------------------------------------
    # x0 normalization for alpha-net stability (train-set statistics only)
    # -----------------------------------------------------------------------
    x0_work = x0_static.astype(np.float32).copy()
    x0_work[:, 4] = np.log1p(np.abs(x0_work[:, 4]))  # IDLeak
    x0_work[:, 5] = np.log1p(np.abs(x0_work[:, 5]))  # IGLeak

    x0_norm_stats = {}
    x0_normalized = x0_work.copy()
    for fi, fname in enumerate(cfg.FEATURES):
        train_vals = x0_work[train_idx, fi]
        train_vals = train_vals[np.isfinite(train_vals)]
        if train_vals.size < 2:
            mu, sigma = 0.0, 1.0
        else:
            mu = float(np.mean(train_vals))
            sigma = float(np.std(train_vals))
            if sigma < 1e-6:
                sigma = 1.0
        x0_norm_stats[fname] = {"mean": mu, "std": sigma}
        x0_normalized[:, fi] = (x0_work[:, fi] - mu) / sigma

    # -----------------------------------------------------------------------
    # Assemble output dict
    # -----------------------------------------------------------------------
    dataset = {
        "x":            x_norm.astype(np.float32),    # normalised degradation
        "x_raw_deg":    x_raw.astype(np.float32),     # unnormalised degradation
        "feature_mask": feature_mask.astype(bool),
        "mask":         mask,                          # (N, T) bool
        "times_h":      times_arr,                    # (N, T)
        "T_K":          T_K_arr,                      # (N,)
        "device_ids":   device_ids,                   # list[str]
        "device_types": device_types,                 # list[str]
        "x0_static":    x0_static,                    # initial absolute values
        "x0_normalized": x0_normalized.astype(np.float32),
        "x0_norm_stats": x0_norm_stats,
        "x0_vth_sdev":  s_vth,
        "leakage_floor": leakage_floor_map,
        "norm_stats":   norm_stats,
        "split": {
            "train": train_idx,
            "val":   val_idx,
            "test":  test_idx,
        },
        "canonical_times_h": canonical_times,
    }

    return dataset


# ---------------------------------------------------------------------------
# Dataset I/O helpers
# ---------------------------------------------------------------------------

def save_dataset(dataset: dict, path: str = None):
    path = path or cfg.PROCESSED_DATA_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)
    log.info("Dataset saved → %s", path)


def load_dataset(path: str = None) -> dict:
    path = path or cfg.PROCESSED_DATA_PATH
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    log.info("Dataset loaded from %s  (%d devices)", path,
             len(dataset["device_ids"]))
    return dataset


# ---------------------------------------------------------------------------
# Sequence construction utilities (called by the training script)
# ---------------------------------------------------------------------------

def make_encoder_input(x: np.ndarray,
                       feature_mask: np.ndarray,
                       T_K: np.ndarray,
                       times_h: np.ndarray) -> np.ndarray:
    """
        Build the 15-dim encoder input at each time step:
            [x1..x6, feature_mask(6), T_norm, log_t, delta_log_t]

    Args:
        x       : (N, T, 6)   normalised degradation features
        T_K     : (N,)        temperature in Kelvin
        times_h : (N, T)      time points in hours

    Returns:
        enc_in  : (N, T, 15)
    """
    N, T, F = x.shape

    # Normalise temperature: (T_K - T_REF) / T_REF
    T_norm = ((T_K - cfg.T_REF_K) / cfg.T_REF_K).reshape(N, 1)  # (N,1)
    T_feat = np.repeat(T_norm, T, axis=1)[:, :, None]             # (N,T,1)

    # Log-time channels
    log_t = np.log(times_h + 1.0)                                 # (N,T)
    delta_log_t = np.zeros_like(log_t)
    delta_log_t[:, :-1] = log_t[:, 1:] - log_t[:, :-1]

    log_t = log_t[:, :, None]                                      # (N,T,1)
    delta_log_t = delta_log_t[:, :, None]                         # (N,T,1)

    x_filled = np.nan_to_num(x, nan=0.0)
    mask_feat = feature_mask.astype(np.float32)
    enc_in = np.concatenate([x_filled, mask_feat, T_feat, log_t, delta_log_t], axis=2)
    return enc_in.astype(np.float32)


def print_dataset_summary(dataset: dict):
    N = len(dataset["device_ids"])
    split = dataset["split"]
    log.info("=" * 55)
    log.info("Dataset summary")
    log.info("  Total devices   : %d", N)
    log.info("  Train / Val / Test: %d / %d / %d",
             len(split["train"]), len(split["val"]), len(split["test"]))
    log.info("  Time points     : %s", cfg.TIME_POINTS_H)
    log.info("  Features        : %s", cfg.FEATURES)

    # Per-temperature counts
    T_K = dataset["T_K"]
    for tc in cfg.TEMPERATURES_C:
        tk = tc + cfg.CELSIUS_TO_KELVIN
        cnt = int((np.abs(T_K - tk) < 1).sum())
        log.info("  T = %d °C        : %d devices", tc, cnt)

    # Completeness
    mask = dataset["mask"]
    obs_per_device = mask.sum(axis=1)
    log.info("  Valid obs/device: min=%d  median=%d  max=%d",
             obs_per_device.min(), int(np.median(obs_per_device)),
             obs_per_device.max())
    log.info("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = build_dataset()
    print_dataset_summary(dataset)
    save_dataset(dataset)
    print("Preprocessing complete.")
