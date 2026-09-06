"""
21_iv_curve_features.py
=======================
Extract mechanism-discriminating features from the RAW IDVG / IDVD / RON /
DIODES sweeps, which the current pipeline throws away.

Why
---
20_latent_degeneracy.py showed the trained decoder has an effective rank of
1.06 out of 4 on its main-channel block: Vth, IDSS, RON and gmmax respond to
one shared latent direction (pairwise cosine 0.91-0.996), so zG/zB/zM/zC are
interchangeable, alpha has nothing to bind to, z_phys collapses onto a
function of (T, t), and the A/B/C physics-conditioning ablation compares
three informationally identical models.

The cause is the OBSERVATION set, not the ODE. Six scalars that all measure
"how much has this device degraded" cannot separate five mechanisms. The raw
sweeps do contain separating information -- measured on 1989 complete sweeps,
the curve-shape features below are predicted by the four old scalars with
R^2 = 0.08-0.33, i.e. they are largely independent signal, while the old
scalars correlate with each other at 0.98+. Effective rank of the observable
set rises 1.06 -> 4.34.

Physical signatures (what separates the mechanisms)
---------------------------------------------------
  SS (subthreshold swing)   interface-state density D_it. Traps that merely
                            charge up shift Vth without changing SS, so SS
                            separates "trap filling" from "trap creation".
  gm peak / FWHM / position mobility degradation vs threshold shift: a pure
                            Vth shift moves the gm curve, mobility loss
                            lowers and broadens it.
  DIBL = Vth_lin - Vth_sat  short-channel / buffer depletion control.
  V_knee, R_on,lin          contact and access resistance vs channel.
  diode ideality n, I_G     gate-stack degradation path.

Outputs
-------
Writes a NEW pickle (default: <OUTPUT_PATH>/iv_curve_features.pkl). It never
touches processed_data.pkl -- that file was once destroyed by a script that
was assumed to respect an --output-dir flag, and is not backed up.

Data-layout notes (verified, not assumed)
-----------------------------------------
* Folder ``275Deg`` holds the 275 C data. It was originally named ``250Deg``;
  the trained dataset is 275/300/325 C and the device IDs inside carry _275_.
* ``1000 Hours_complete`` is a strict SUBSET of ``1000 Hours`` (53 vs 91
  files, identical bytes for shared cells, only the naming differs:
  ``A2A_No_fieldPlate`` vs ``A2A_CH4``). Despite the name it is LESS complete,
  so it is skipped.
* Device-type folders are spelled HEMT / HEMTs / Hemt, and the A2A variants
  appear as A2A_CH4_FP (275, 325) and A2A_FP_CH4 (300). ``A2A_CH4_FP``
  contains ``A2A_CH4`` as a substring, so field-plate variants MUST be tested
  first or every FP device is mislabelled.
* At 275 C the ``A2A`` preliminary folder also contains CH5 sweeps, a channel
  that is not in the trained set; those are skipped.
* Repeat measurements appear as ``_repeat``, ``_bis`` or a doubled underscore.
  The LAST measurement is kept (user decision: re-measurements supersede).
* Sweep shapes differ per device type and must be read from the file, never
  hard-coded: IDVG is 81 VG points x 10 VD biases, except P10C which
  sometimes has 9 VD columns; IDVD is 201 VD points x 7 VG biases, except
  P10C at 151.

Usage
-----
    python 21_iv_curve_features.py
    python 21_iv_curve_features.py --limit 50        # quick smoke test
    python 21_iv_curve_features.py --report-only     # coverage, no write
"""

import argparse
import collections
import glob
import io
import json
import logging
import os
import pickle
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config as cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

RAW_ROOT = os.path.join(os.path.dirname(cfg.DATA_PATH))

TEMP_DIRS = {"275Deg": 275, "300Deg": 300, "325Deg": 325}

# Skipped: strict subset of "1000 Hours" with different naming (see header).
SKIP_TIMEPOINT_DIRS = {"1000 Hours_complete"}

PRELIM_DIR_NAME = "preliminary"     # matched case-insensitively; this is t=0

# Bias points at which the linear- and saturation-region features are taken.
# Chosen from the measured VD array [0.1 0.5 0.9 1 3 5 7 10 15 20]: 0.5 V is
# safely inside the linear region, 20 V is the deepest saturation available.
VD_LIN, VD_SAT = 0.5, 20.0

# Current floor for log-slope work: below this the measurement is noise.
I_FLOOR = 1e-12

# Temperature of the ELECTRICAL MEASUREMENT. The 275/300/325 C in the folder
# names is the thermal-storage STRESS temperature: parts are soaked hot,
# cooled, then probed at room temperature. Any kT/q in an extraction formula
# must use this, not the stress temperature.
MEAS_TEMP_C = 25.0

# Device-type folder / filename spellings -> canonical type used in the
# trained dataset. ORDER MATTERS: A2A_CH4_FP contains A2A_CH4.
DEVTYPE_PATTERNS: List[Tuple[str, str]] = [
    (r"A2A[_ ]?CH4[_ ]?FP", "A2ACH4FP"),
    (r"A2A[_ ]?FP[_ ]?CH4", "A2ACH4FP"),
    (r"A2A[_ ]?CH4FP",      "A2ACH4FP"),
    (r"A2A[_ ]?NO[_ ]?FIELDPLATE", "A2ACH4"),
    (r"A2A[_ ]?CH5",        None),        # channel not in the trained set
    (r"A2A[_ ]?CH4",        "A2ACH4"),
    (r"A8A",                "A8A"),
    (r"P10C",               "P10C"),
]

# Ordering used to pick a winner when one device/timepoint has several files.
# Higher = later measurement = preferred.
REMEASURE_RANK = [
    (r"_repeat",  3),
    (r"_bis",     3),
    (r"__",       2),
    (r"_ter",     4),
]


# ---------------------------------------------------------------------------
# Raw file parsing
# ---------------------------------------------------------------------------

def parse_sections(path: str) -> Dict[str, List[str]]:
    """Parse the instrument's text format into {section_name: [lines]}.

    Layout is a named block whose closing tag repeats the name:

        VD ARRAY
        1.0E-1\t5.0E-1\t...
        VD ARRAY

    Matrices are tab-separated with one row per sweep point.
    """
    lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    sec: Dict[str, List[str]] = {}
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        if name and not re.match(r"^[-+0-9.]", name):
            j, buf = i + 1, []
            while j < len(lines) and lines[j].strip() != name:
                s = lines[j].strip()
                if s:
                    buf.append(s)
                j += 1
            sec[name] = buf
            i = j + 1
        else:
            i += 1
    return sec


def _matrix(sec: Dict[str, List[str]], key: str) -> Optional[np.ndarray]:
    """Tab-separated matrix -> (n_points, n_bias). Ragged rows are truncated."""
    if key not in sec:
        return None
    rows = []
    for ln in sec[key]:
        vals = [v for v in ln.split("\t") if v.strip()]
        if vals:
            rows.append(vals)
    if not rows:
        return None
    n = min(len(r) for r in rows)
    try:
        return np.array([[float(v) for v in r[:n]] for r in rows], dtype=float)
    except ValueError:
        return None


def _array(sec: Dict[str, List[str]], key: str) -> Optional[np.ndarray]:
    """Read a 1-D section, in either of the two layouts the tool emits.

    Bias arrays (VD ARRAY, VTH ARRAY, ...) are written as ONE tab-separated
    line, while the diode sweeps (VGS/IGS/VGD/IGD ARRAY) are written as one
    value PER LINE -- 427 of them. Reading only the first line, as an earlier
    version did, silently returned a length-1 array and made every ideality
    factor NaN. Flattening every line covers both.
    """
    if key not in sec or not sec[key]:
        return None
    vals: List[float] = []
    for ln in sec[key]:
        for tok in ln.split("\t"):
            tok = tok.strip()
            if not tok:
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                return None
    return np.array(vals, dtype=float) if vals else None


# ---------------------------------------------------------------------------
# Physics feature extraction
# ---------------------------------------------------------------------------

def subthreshold_swing(vg: np.ndarray, idrain: np.ndarray,
                       gm: Optional[np.ndarray]) -> float:
    """Minimum dVG/dlog10(ID) in mV/decade over the subthreshold region.

    The subthreshold region is taken as gate voltages below the peak-gm point;
    above it the device is in strong inversion and the slope no longer
    reflects D_it. The minimum (steepest) slope is the conventional estimate,
    and is robust to the tail flattening out at the noise floor.
    """
    idrain = np.abs(idrain)
    ok = idrain > I_FLOOR
    if ok.sum() < 5:
        return float("nan")
    v, lg = vg[ok], np.log10(idrain[ok])
    dv, dl = np.gradient(v), np.gradient(lg)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss = np.where(np.abs(dl) > 1e-9, dv / dl, np.nan) * 1000.0
    if gm is not None and np.isfinite(gm).any():
        v_pk = vg[int(np.nanargmax(gm))]
        cut = max(int(np.searchsorted(v, v_pk)), 5)
    else:
        cut = len(v)
    region = (np.arange(len(v)) < cut) & (ss > 0) & np.isfinite(ss)
    return float(np.nanmin(ss[region])) if region.any() else float("nan")


def gm_shape(vg: np.ndarray, gm: np.ndarray) -> Dict[str, float]:
    """Peak transconductance, the gate voltage at which it occurs, and FWHM.

    A rigid Vth shift moves V_gmpeak but leaves gm_peak and the width alone;
    mobility degradation lowers the peak and broadens the curve. Tracking all
    three is what lets the two mechanisms be told apart.
    """
    if gm is None or not np.isfinite(gm).any():
        return {"gm_peak": float("nan"), "V_gmpeak": float("nan"),
                "gm_fwhm": float("nan")}
    pk = float(np.nanmax(gm))
    ipk = int(np.nanargmax(gm))
    above = np.where(gm >= pk / 2.0)[0]
    fwhm = float(vg[above[-1]] - vg[above[0]]) if len(above) > 1 else float("nan")
    return {"gm_peak": pk, "V_gmpeak": float(vg[ipk]), "gm_fwhm": fwhm}


def diode_ideality(vgs: np.ndarray, igs: np.ndarray,
                   T_meas_C: float = MEAS_TEMP_C) -> float:
    """Ideality factor n of the forward gate diode: n = q/(kT) * dV/dln(I).

    T_meas_C is the temperature of the ELECTRICAL MEASUREMENT, which is room
    temperature -- NOT the 275/300/325 C storage stress. These are unpowered
    thermal-storage tests: the parts are soaked hot, then cooled and probed.
    Feeding the stress temperature in here (as an earlier version did) scales
    kT/q by ~2x and yields n ~ 0.66 for every device, which is impossible
    since n >= 1. Measured slope is 95.8 mV/decade, i.e. n = 1.62 at 25 C.

    Only the genuinely exponential part of the forward branch is fitted. Two
    traps are avoided here, both found by measurement:

    * Near zero bias the gate current is instrument noise -- values of order
      1e-11 A that flip sign from point to point (30 % negative). Fitting a
      sliding window and keeping the steepest one, as an earlier version did,
      locked onto that noise and returned n ~ 0.67, which is unphysical: n
      cannot be below 1. The fit therefore starts only once |I_G| has climbed
      well clear of the noise floor.
    * At the top of the sweep the series resistance bends the curve over,
      which biases n upward, so the top decade is dropped.

    Returns NaN rather than a number whenever the usable window is too short
    or the resulting n is outside the physically meaningful range.
    """
    kT_q = 8.617333e-5 * (T_meas_C + 273.15)
    i = np.abs(igs)
    fwd = (vgs > 0.0) & np.isfinite(i) & (i > 0)
    if fwd.sum() < 10:
        return float("nan")
    v, ii = vgs[fwd], i[fwd]

    # Noise floor: the near-zero-bias plateau. Anything within 10x of it is
    # not diode conduction.
    floor = max(float(np.median(ii[:max(3, len(ii) // 10)])), I_FLOOR)
    usable = ii > 10.0 * floor
    if usable.sum() < 8:
        return float("nan")
    v, ii = v[usable], ii[usable]

    # Drop the compliance plateau: the sweep clamps at 1e-3 A, and those
    # points carry no slope information.
    imax = ii.max()
    not_clamped = ii < 0.95 * imax
    if not_clamped.sum() >= 8:
        v, ii = v[not_clamped], ii[not_clamped]

    # Drop the top decade of what remains, where series resistance bends the
    # curve and biases n upward -- but only when enough range survives. These
    # sweeps span ~7 decades from noise floor to compliance, so trimming
    # unconditionally (as an earlier version did) threw away all but 1 % of
    # the devices.
    if len(v) >= 16 and ii.max() / ii.min() >= 100.0:
        keep = ii <= ii.max() / 10.0
        if keep.sum() >= 8:
            v, ii = v[keep], ii[keep]
    if len(v) < 8 or ii.max() / ii.min() < 10.0:
        return float("nan")

    slope = float(np.polyfit(v, np.log(ii), 1)[0])
    if slope <= 0:
        return float("nan")
    n = 1.0 / (kT_q * slope)
    return float(n) if 1.0 <= n < 20.0 else float("nan")


def knee_voltage(vd: np.ndarray, idrain: np.ndarray, frac: float = 0.9) -> float:
    """Drain voltage where ID first reaches `frac` of its saturation value.

    Buffer trapping and access-resistance growth both push the knee out, so
    this is the output-curve counterpart to RON. Uses the 95th percentile
    rather than the max as the saturation reference to resist single-point
    spikes.
    """
    i = np.abs(idrain)
    if not np.isfinite(i).any() or np.nanmax(i) <= 0:
        return float("nan")
    isat = float(np.nanpercentile(i, 95))
    if isat <= 0:
        return float("nan")
    hit = np.where(i >= frac * isat)[0]
    return float(vd[hit[0]]) if len(hit) else float("nan")


def features_from_idvg(path: str) -> Optional[Dict[str, float]]:
    """Linear- and saturation-region curve-shape features from one IDVG file."""
    sec = parse_sections(path)
    VD = _array(sec, "VD ARRAY")
    VG = _matrix(sec, "VG MATRIX")
    ID = _matrix(sec, "ID MATRIX")
    GM = _matrix(sec, "GM MATRIX")
    VT = _array(sec, "VTH ARRAY")
    if VD is None or VG is None or ID is None:
        return None
    # Shapes vary by device type (P10C sometimes has 9 VD columns), so always
    # trust the file over any expected layout.
    ncol = min(VG.shape[1], ID.shape[1], len(VD))
    if ncol < 2 or VG.shape[0] < 10:
        return None
    vg = VG[:, 0]
    out: Dict[str, float] = {}
    for tag, vd_target in (("lin", VD_LIN), ("sat", VD_SAT)):
        c = int(np.argmin(np.abs(VD[:ncol] - vd_target)))
        idc = ID[:, c]
        gmc = GM[:, c] if GM is not None and c < GM.shape[1] else None
        out[f"SS_{tag}"] = subthreshold_swing(vg, idc, gmc)
        for k, v in gm_shape(vg, gmc).items():
            out[f"{k}_{tag}"] = v
        out[f"Ion_{tag}"] = float(np.nanmax(np.abs(idc)))
        out[f"Ioff_{tag}"] = float(np.nanmean(np.abs(idc[:5])))
        out[f"Vth_{tag}"] = float(VT[c]) if VT is not None and c < len(VT) else float("nan")
        out[f"VD_{tag}"] = float(VD[c])
    out["DIBL"] = out["Vth_lin"] - out["Vth_sat"]
    out["SS_ratio"] = (out["SS_sat"] / out["SS_lin"]
                       if out.get("SS_lin") else float("nan"))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_IonIoff_sat"] = float(np.log10(max(out["Ion_sat"], I_FLOOR) /
                                                max(out["Ioff_sat"], I_FLOOR)))
    return out


def features_from_out(path: str) -> Dict[str, float]:
    """Knee voltage and output conductance from the IDVD (OUT) sweep."""
    sec = parse_sections(path)
    VG = _array(sec, "VG ARRAY")
    VD = _matrix(sec, "VD MATRIX")
    ID = _matrix(sec, "ID MATRIX")
    if VD is None or ID is None or VG is None:
        return {}
    ncol = min(VD.shape[1], ID.shape[1], len(VG))
    if ncol < 1:
        return {}
    c = int(np.argmax(VG[:ncol]))       # most positive gate bias = strongest on-state
    vd, idr = VD[:, c], ID[:, c]
    out = {"V_knee": knee_voltage(vd, idr), "VG_knee_bias": float(VG[c])}
    lin = (vd > 0) & (vd <= 2.0) & np.isfinite(idr)
    if lin.sum() >= 3 and np.ptp(idr[lin]) > 0:
        slope = np.polyfit(vd[lin], idr[lin], 1)[0]
        out["Ron_out"] = float(1.0 / slope) if slope > 0 else float("nan")
    return out


def features_from_diodes(path: str) -> Dict[str, float]:
    """Gate-diode ideality and the recorded leakage levels."""
    sec = parse_sections(path)
    out: Dict[str, float] = {}
    for key, name in (("IGS LEAK", "IGS_leak"), ("IDS LEAK", "IDS_leak")):
        a = _array(sec, key)
        if a is not None and len(a):
            out[name] = float(a[0])
    vgs, igs = _array(sec, "VGS ARRAY"), _array(sec, "IGS ARRAY")
    if vgs is not None and igs is not None and len(vgs) == len(igs):
        out["n_ideality"] = diode_ideality(vgs, igs)
    return out


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------

def canonical_devtype(text: str) -> Optional[str]:
    """Map a folder or filename fragment to the trained device-type name.

    Returns None for channels that are not part of the trained set (CH5) and
    for anything unrecognised, so the caller can skip them explicitly.
    """
    s = text.upper()
    for pat, canon in DEVTYPE_PATTERNS:
        if re.search(pat, s):
            return canon
    return None


def cell_from_stem(stem: str) -> Optional[str]:
    """Pull the die-cell code (F2, G10, ...) out of a filename stem.

    Filenames look like CE16_QB_F6_A2A_CH4_275C_1000H, but some are written
    CE16_G3P10C_275_10Hs, so both a standalone token and a letter+digit prefix
    glued to the device type are accepted. Zero padding is normalised (F09 and
    F9 are the same cell) because the trained IDs use both.
    """
    for tok in stem.split("_")[1:5]:
        m = re.fullmatch(r"([A-Za-z]{1,2})(\d{1,2})", tok)
        if m:
            return f"{m.group(1).upper()}{int(m.group(2))}"
        m = re.match(r"^([A-Za-z]{1,2})(\d{1,2})(P10C|A8A|A2A)", tok, re.I)
        if m:
            return f"{m.group(1).upper()}{int(m.group(2))}"
    return None


def normalise_id(dev_id: str) -> str:
    """TYPE_TEMP_CELL with the cell number unpadded (A2ACH4FP_300_F09 -> _F9)."""
    m = re.match(r"^(.+)_(\d+)_([A-Za-z]+)0*(\d+)$", dev_id)
    return (f"{m.group(1)}_{m.group(2)}_{m.group(3).upper()}{int(m.group(4))}"
            if m else dev_id)


def remeasure_rank(stem: str) -> int:
    """Preference score for duplicate sweeps of the same device/timepoint.

    Re-measurements supersede the original (user decision), so a file marked
    repeat/bis/ter wins over the plain one.
    """
    s = stem.lower()
    rank = 0
    for pat, r in REMEASURE_RANK:
        if re.search(pat, s):
            rank = max(rank, r)
    return rank


def parse_hours(dirname: str) -> Optional[float]:
    """'100 Hours', '100H', '1000 Hs' -> 100.0 / 1000.0 ; preliminary -> 0.0."""
    if PRELIM_DIR_NAME in dirname.lower():
        return 0.0
    m = re.match(r"^\s*(\d+)\s*(?:H|Hs|Hr|Hour|Hours)\b", dirname.strip(), re.I)
    return float(m.group(1)) if m else None


def find_hemt_dir(path: str) -> Optional[str]:
    """Locate the HEMT subfolder, spelled HEMT / HEMTs / Hemt across temps."""
    if not os.path.isdir(path):
        return None
    for d in os.listdir(path):
        if d.lower().startswith("hemt") and os.path.isdir(os.path.join(path, d)):
            return os.path.join(path, d)
    return None


# ---------------------------------------------------------------------------
# Sweep collection
# ---------------------------------------------------------------------------

def collect_sweeps(root: str) -> Dict[Tuple[str, float], Dict[str, str]]:
    """Walk the raw tree and index the best file per (device_id, hours, kind).

    Returns {(normalised_id, hours): {kind: path}} where kind is one of
    idvg / out / ron / diodes. Duplicate measurements are resolved by
    remeasure_rank, so the last measurement wins.
    """
    best: Dict[Tuple[str, float], Dict[str, Tuple[int, str]]] = collections.defaultdict(dict)
    stats = collections.Counter()

    for temp_dir, T_C in TEMP_DIRS.items():
        base = os.path.join(root, temp_dir)
        if not os.path.isdir(base):
            log.warning("missing temperature folder: %s", base)
            continue
        for entry in sorted(os.listdir(base)):
            if entry in SKIP_TIMEPOINT_DIRS:
                stats["skipped_duplicate_dir"] += 1
                continue
            hours = parse_hours(entry)
            if hours is None:
                continue
            hemt = find_hemt_dir(os.path.join(base, entry))
            if hemt is None:
                stats["no_hemt_dir"] += 1
                continue
            for sub in sorted(os.listdir(hemt)):
                subp = os.path.join(hemt, sub)
                if not os.path.isdir(subp):
                    continue
                folder_type = canonical_devtype(sub)
                for path in glob.glob(os.path.join(subp, "*.txt")):
                    fname = os.path.basename(path)
                    m = re.match(r"^(.*)_(IDVGGo_|OUT|RON|HV|DIODES[\w.]*)\.txt$",
                                 fname, re.I)
                    if not m:
                        stats["unrecognised_kind"] += 1
                        continue
                    stem, kind_raw = m.group(1), m.group(2).upper()
                    kind = {"IDVGGO_": "idvg", "OUT": "out",
                            "RON": "ron", "HV": "hv"}.get(kind_raw)
                    if kind is None:
                        kind = "diodes" if kind_raw.startswith("DIODES") else None
                    if kind is None or kind == "hv":
                        continue
                    # Filename wins over folder: the 275 C A2A folder mixes
                    # CH4, CH4_FP and CH5 together.
                    dtype = canonical_devtype(stem) or folder_type
                    if dtype is None:
                        stats["skipped_devtype"] += 1
                        continue
                    cell = cell_from_stem(stem)
                    if cell is None:
                        stats["no_cell"] += 1
                        continue
                    dev_id = normalise_id(f"{dtype}_{T_C}_{cell}")
                    rank = remeasure_rank(stem)
                    key = (dev_id, hours)
                    prev = best[key].get(kind)
                    if prev is None or rank > prev[0]:
                        best[key][kind] = (rank, path)
                        stats[f"kept_{kind}"] += 1

    log.info("scan stats: %s", dict(stats))
    return {k: {kind: p for kind, (_, p) in v.items()} for k, v in best.items()}


def main():
    ap = argparse.ArgumentParser(description="Extract IV-curve physics features")
    ap.add_argument("--raw-root", default=RAW_ROOT,
                    help="Folder containing 275Deg/300Deg/325Deg")
    ap.add_argument("--output", default=os.path.join(cfg.OUTPUT_PATH,
                                                     "iv_curve_features.pkl"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only N sweeps (smoke test)")
    ap.add_argument("--report-only", action="store_true",
                    help="Report coverage without writing the output file")
    args = ap.parse_args()

    if os.path.abspath(args.output) == os.path.abspath(cfg.PROCESSED_DATA_PATH):
        log.error("refusing to overwrite the processed dataset: %s", args.output)
        return

    log.info("raw root: %s", args.raw_root)
    sweeps = collect_sweeps(args.raw_root)
    log.info("indexed %d (device, timepoint) pairs", len(sweeps))

    with open(cfg.PROCESSED_DATA_PATH, "rb") as f:
        trained = pickle.load(f)
    trained_ids = {normalise_id(i): i for i in trained["device_ids"]}
    log.info("trained devices: %d", len(trained_ids))

    keys = sorted(sweeps)
    if args.limit:
        keys = keys[:args.limit]

    records: List[Dict] = []
    fail = collections.Counter()
    for dev_id, hours in keys:
        paths = sweeps[(dev_id, hours)]
        if "idvg" not in paths:
            fail["no_idvg"] += 1
            continue
        T_C = float(dev_id.split("_")[1])
        try:
            feats = features_from_idvg(paths["idvg"])
        except Exception as exc:                      # noqa: BLE001
            log.debug("idvg parse failed %s: %s", paths["idvg"], exc)
            feats = None
        if feats is None:
            fail["idvg_unparsable"] += 1
            continue
        for kind, fn in (("out", features_from_out),
                         ("diodes", features_from_diodes)):
            if kind in paths:
                try:
                    feats.update(fn(paths[kind]))
                except Exception:                     # noqa: BLE001
                    fail[f"{kind}_unparsable"] += 1
        feats.update({"device_id": dev_id, "hours": hours, "T_C": T_C,
                      "in_trained_set": dev_id in trained_ids,
                      "device_type": dev_id.split("_")[0]})
        records.append(feats)

    log.info("extracted %d records   failures: %s", len(records), dict(fail))
    if not records:
        log.error("nothing extracted")
        return

    feat_names = sorted({k for r in records for k in r
                         if k not in {"device_id", "hours", "T_C",
                                      "in_trained_set", "device_type"}})
    log.info("features (%d): %s", len(feat_names), feat_names)

    covered = {r["device_id"] for r in records if r["in_trained_set"]}
    log.info("=" * 74)
    log.info("COVERAGE")
    log.info("  trained devices covered : %d / %d (%.1f%%)",
             len(covered), len(trained_ids), 100 * len(covered) / len(trained_ids))
    missing = sorted(set(trained_ids) - covered)
    if missing:
        log.info("  missing (%d): %s%s", len(missing), missing[:10],
                 " ..." if len(missing) > 10 else "")
    per_dev = collections.Counter(r["device_id"] for r in records
                                  if r["in_trained_set"])
    if per_dev:
        counts = collections.Counter(per_dev.values())
        log.info("  timepoints per covered device: %s", dict(sorted(counts.items())))
    with_t0 = {r["device_id"] for r in records
               if r["in_trained_set"] and r["hours"] == 0.0}
    log.info("  covered devices WITH a t=0 baseline: %d", len(with_t0))

    log.info("")
    log.info("FEATURE COMPLETENESS (fraction finite, trained devices only)")
    sub = [r for r in records if r["in_trained_set"]]
    for name in feat_names:
        vals = np.array([r.get(name, np.nan) for r in sub], dtype=float)
        frac = float(np.isfinite(vals).mean()) if len(vals) else 0.0
        fin = vals[np.isfinite(vals)]
        rng = f"[{fin.min():.4g}, {fin.max():.4g}]" if len(fin) else "-"
        log.info("  %-18s %6.1f%%  median=%-11.4g range=%s",
                 name, 100 * frac,
                 float(np.median(fin)) if len(fin) else float("nan"), rng)

    if args.report_only:
        log.info("--report-only: not writing")
        return

    payload = {
        "records": records,
        "feature_names": feat_names,
        "vd_lin": VD_LIN, "vd_sat": VD_SAT,
        "raw_root": args.raw_root,
        "note": ("Curve-shape features extracted from raw IDVG/IDVD/DIODES "
                 "sweeps. Separate from processed_data.pkl, which is not "
                 "modified by this script."),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)
    log.info("Saved -> %s  (%d records)", args.output, len(records))


if __name__ == "__main__":
    main()
