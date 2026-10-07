#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B1500 + Avantes Spectrometer Step Stress Measurement

Primary flow:
    Measurement -> Stress(V1) -> Measurement -> Stress(V2) -> ...

Stress levels are defined by start/stop/step, aligned with
b1500_step_stress_measurement.py. This module also keeps a legacy constant-
stress cycle mode for backward compatibility with older scripts.

Author: Veronica GaoZhan
Date: June 2026
"""

import sys
import os
import csv
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ctypes

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
                             QDoubleSpinBox, QTextEdit, QFileDialog, QMessageBox, QProgressBar,
                             QGridLayout, QCheckBox, QSplitter, QStatusBar, QFrame,
                             QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    import pyvisa
    PYVISA_AVAILABLE = True
except ImportError:
    PYVISA_AVAILABLE = False
    print("Warning: pyvisa not installed.")

try:
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
except (AttributeError, OSError, TypeError):
    pass


TIME_DESIGN_LINEAR = "linear"
TIME_DESIGN_LOG = "log"
LOG_POINTS_PER_DECADE_OPTIONS = (1, 2, 3, 5, 10)
LOG_CYCLE_MANTISSAS = {
    1: (1.0,),
    2: (1.0, 3.0),
    3: (1.0, 2.0, 5.0),
    5: (1.0, 2.0, 3.0, 5.0, 8.0),
    10: (1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 8.0),
}


# =============================================================================
# Spectrometer SDK constants and structures
# =============================================================================

AVS_SERIAL_LEN = 10
USER_ID_LEN = 64
MAX_NR_PIXELS = 4096
INVALID_AVS_HANDLE_VALUE = 1000


class AvsIdentityType(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("SerialNumber", ctypes.c_char * AVS_SERIAL_LEN),
        ("UserFriendlyName", ctypes.c_char * USER_ID_LEN),
        ("Status", ctypes.c_char)
    ]


class MeasConfigType(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("m_StartPixel", ctypes.c_uint16),
        ("m_StopPixel", ctypes.c_uint16),
        ("m_IntegrationTime", ctypes.c_float),
        ("m_IntegrationDelay", ctypes.c_uint32),
        ("m_NrAverages", ctypes.c_uint32),
        ("m_CorDynDark_m_Enable", ctypes.c_uint8),
        ("m_CorDynDark_m_ForgetPercentage", ctypes.c_uint8),
        ("m_Smoothing_m_SmoothPix", ctypes.c_uint16),
        ("m_Smoothing_m_SmoothModel", ctypes.c_uint8),
        ("m_SaturationDetection", ctypes.c_uint8),
        ("m_Trigger_m_Mode", ctypes.c_uint8),
        ("m_Trigger_m_Source", ctypes.c_uint8),
        ("m_Trigger_m_SourceType", ctypes.c_uint8),
        ("m_Control_m_StrobeControl", ctypes.c_uint16),
        ("m_Control_m_LaserDelay", ctypes.c_uint32),
        ("m_Control_m_LaserWidth", ctypes.c_uint32),
        ("m_Control_m_LaserWaveLength", ctypes.c_float),
        ("m_Control_m_StoreToRam", ctypes.c_uint16)
    ]


# =============================================================================
# Spectrometer DLL loader  (identical logic to b1500_stress_spectroscopy.py)
# =============================================================================

_spec_lib = None
_spec_func = None
_spec_dll_handles = []


def _spec_search_dirs() -> List[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("AVASPEC_DLL_PATH"),
        script_dir,
        os.path.join(script_dir, "AVASpec DLL"),
        r"C:\AvaSpecX64-DLL_9.14.0.0",
        os.path.join(os.environ.get("ProgramFiles", r"C:"), "Avantes"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:"), "Avantes"),
    ]
    seen, result = set(), []
    for d in candidates:
        if not d:
            continue
        full = os.path.normcase(os.path.abspath(os.path.expandvars(d)))
        if full not in seen:
            seen.add(full)
            result.append(os.path.abspath(os.path.expandvars(d)))
    return result


def load_spectrometer_library():
    global _spec_lib, _spec_func, _spec_dll_handles
    if _spec_lib and _spec_func:
        return _spec_lib, _spec_func

    if 'linux' in sys.platform:
        _spec_lib = ctypes.CDLL("/usr/local/lib/libavs.so.0")
        _spec_func = ctypes.CFUNCTYPE
    elif 'darwin' in sys.platform:
        _spec_lib = ctypes.CDLL("/usr/local/lib/libavs.0.dylib")
        _spec_func = ctypes.CFUNCTYPE
    else:
        import ctypes.wintypes
        dll_name = "avaspecx64.dll" if ctypes.sizeof(ctypes.c_voidp) == 8 else "avaspec.dll"
        errors = []
        if hasattr(os, "add_dll_directory"):
            for d in _spec_search_dirs():
                if os.path.isdir(d):
                    try:
                        _spec_dll_handles.append(os.add_dll_directory(d))
                    except OSError:
                        pass
        for d in [None] + _spec_search_dirs():
            path = dll_name if d is None else os.path.join(d, dll_name)
            if d is not None and not os.path.exists(path):
                continue
            try:
                _spec_lib = ctypes.WinDLL(path)
                break
            except OSError as e:
                errors.append(str(e))
        if _spec_lib is None:
            raise FileNotFoundError(f"Cannot load {dll_name}. Errors: {'; '.join(errors[:3])}")
        _spec_func = ctypes.WINFUNCTYPE
    return _spec_lib, _spec_func


# =============================================================================
# Enums and data classes
# =============================================================================

class TestPhase(Enum):
    IDLE = "idle"
    MEASUREMENT = "measurement"
    STRESS = "stress"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class SweepConfig:
    smu: int = 1
    mode: str = "iv"          # "iv" (source V, meas I) or "vi" (source I, meas V)
    start: float = 0.0
    stop: float = 2.0
    steps: int = 21
    dwell_s: float = 0.1
    compliance: float = 0.1

    @property
    def setpoints(self) -> List[float]:
        if self.steps < 2:
            return [self.start]
        return [self.start + i * (self.stop - self.start) / (self.steps - 1)
                for i in range(self.steps)]


@dataclass
class StressConfig:
    mode: str = "voltage"      # "voltage" or "current"
    value: float = 2.0           # Legacy fixed stress level for cycle mode
    # Step-stress mode (aligned with b1500_step_stress_measurement.py)
    start_value: Optional[float] = 2.0
    stop_value: Optional[float] = 5.0
    step_value: Optional[float] = 0.5
    duration_s: float = 60.0
    # Legacy time-design fields kept for backward compatibility
    time_design: str = TIME_DESIGN_LINEAR
    log_points_per_decade: int = 5
    sample_interval_s: float = 1.0
    compliance: float = 0.1

    @property
    def stress_levels(self) -> List[float]:
        if (self.start_value is None or self.stop_value is None or
                self.step_value is None or abs(self.step_value) < 1e-12):
            return []

        levels = []
        current = float(self.start_value)
        stop = float(self.stop_value)
        step = float(self.step_value)

        if step > 0:
            while current <= stop + 1e-9:
                levels.append(current)
                current += step
        else:
            while current >= stop - 1e-9:
                levels.append(current)
                current += step
        return levels


@dataclass
class SpecConfig:
    """Spectrometer settings with SEPARATE integration times for stress vs measurement"""
    enabled: bool = True
    # Integration time during the IV characterisation phase (per spectrum)
    meas_integration_ms: float = 100.0
    # Integration time during the stress phase
    stress_integration_ms: float = 100.0
    num_averages: int = 1
    # How often to acquire a spectrum during stress (ms between acquisitions)
    stress_interval_ms: float = 1000.0


@dataclass
class CycleConfig:
    sweep: SweepConfig = field(default_factory=SweepConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
    num_cycles: int = 10  # Legacy cycle count when use_step_stress=False
    use_step_stress: bool = False
    initial_measurement: bool = True
    output_folder: str = "results"
    device_name: str = "Device_001"
    autosave: bool = True


@dataclass
class IVPoint:
    cycle: int
    stress_level: float
    point_index: int
    timestamp: float
    setpoint: float
    voltage: float
    current: float


@dataclass
class SpectrumPoint:
    cycle: int
    stress_level: float
    phase: str                 # "measurement" or "stress"
    index: int
    timestamp: float
    relative_time: float
    bias_voltage: float
    bias_current: float
    wavelength: List[float]
    intensity: List[float]
    peak_wavelength: float
    peak_intensity: float
    integrated_intensity: float


@dataclass
class StressMonitorPoint:
    cycle: int
    stress_level: float
    timestamp: float
    elapsed_s: float
    voltage: float
    current: float


@dataclass
class CycleSummary:
    cycle: int
    stress_level: float
    timestamp: str
    peak_current: float
    peak_spec_intensity: float   # Peak intensity from measurement spectrum


# =============================================================================
# Spectrometer controller
# =============================================================================

class SpectrometerController:
    def __init__(self):
        self.handle = None
        self.num_pixels = 0
        self.wavelength: Optional[List[float]] = None
        self.lib = None
        self.func = None
        self.lock = threading.Lock()
        self.connected = False

    def initialize(self) -> bool:
        try:
            self.lib, self.func = load_spectrometer_library()
            proto = self.func(ctypes.c_int, ctypes.c_int)
            AVS_Init = proto(("AVS_Init", self.lib), ((1, "port"),))
            n = AVS_Init(0)
            if n < 0:
                print(f"AVS_Init error: {n}")
                return False
            print(f"Spectrometer SDK: {n} device(s)")
            return True
        except Exception as e:
            print(f"Spectrometer init error: {e}")
            return False

    def connect_device(self, index: int = 0) -> bool:
        try:
            proto = self.func(ctypes.c_int)
            n = proto(("AVS_UpdateUSBDevices", self.lib),)()
            if n <= 0:
                print("No USB spectrometers found")
                return False

            proto = self.func(ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(AvsIdentityType * n))
            pf = (1, "listsize"), (2, "req"), (2, "IDlist")
            AVS_GetList = proto(("AVS_GetList", self.lib), pf)
            _, device_list = AVS_GetList(n * 75)

            if index >= n:
                print(f"Invalid device index {index}")
                return False

            dev = device_list[index]
            dt = ctypes.c_byte * 75
            temp = dt()
            for x in range(9):
                temp[x] = dev.SerialNumber[x]
            temp[9] = 0
            for x in range(10, 74):
                temp[x] = 0
            temp[74] = int.from_bytes(dev.Status, byteorder='big')

            proto2 = self.func(ctypes.c_int, ctypes.c_byte * 75)
            AVS_Activate = proto2(("AVS_Activate", self.lib), ((1, "deviceId"),))
            self.handle = AVS_Activate(temp)

            if self.handle == INVALID_AVS_HANDLE_VALUE or self.handle < 0:
                print(f"AVS_Activate failed: {self.handle}")
                return False

            self._enable_high_res_adc()
            self._get_device_info()
            self.connected = True
            print(f"Spectrometer connected (handle={self.handle}, 16-bit ADC)")
            return True
        except Exception as e:
            print(f"Spectrometer connect error: {e}")
            return False

    def _enable_high_res_adc(self):
        try:
            proto = self.func(ctypes.c_int, ctypes.c_int, ctypes.c_bool)
            AVS_UseHighResAdc = proto(("AVS_UseHighResAdc", self.lib),
                                     ((1, "handle"), (1, "enable")))
            ret = AVS_UseHighResAdc(self.handle, True)
            if ret == 0:
                print("16-bit ADC enabled (max 65535 counts)")
            else:
                print(f"Warning: 16-bit ADC not enabled (ret={ret})")
        except Exception as e:
            print(f"High-res ADC warning: {e}")

    def _get_device_info(self):
        proto = self.func(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_short))
        result = proto(("AVS_GetNumPixels", self.lib),
                       ((1, "handle"), (2, "numPixels")))(self.handle)
        self.num_pixels = result[1] if isinstance(result, tuple) else result

        proto2 = self.func(ctypes.c_int, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_double * MAX_NR_PIXELS))
        wd = proto2(("AVS_GetLambda", self.lib),
                    ((1, "handle"), (2, "wavelength")))(self.handle)
        raw = wd[0] if isinstance(wd, tuple) else wd
        self.wavelength = list(raw)[:self.num_pixels]
        print(f"Spectrometer: {self.num_pixels} px, "
              f"{self.wavelength[0]:.1f}–{self.wavelength[-1]:.1f} nm")

    def configure_measurement(self, integration_ms: float, num_averages: int = 1) -> bool:
        if not self.handle:
            return False
        with self.lock:
            cfg = MeasConfigType()
            cfg.m_StartPixel = 0
            cfg.m_StopPixel = self.num_pixels - 1
            cfg.m_IntegrationTime = float(integration_ms)
            cfg.m_IntegrationDelay = 0
            cfg.m_NrAverages = num_averages
            cfg.m_CorDynDark_m_Enable = 0
            cfg.m_CorDynDark_m_ForgetPercentage = 0
            cfg.m_Smoothing_m_SmoothPix = 0
            cfg.m_Smoothing_m_SmoothModel = 0
            cfg.m_SaturationDetection = 0
            cfg.m_Trigger_m_Mode = 0
            cfg.m_Trigger_m_Source = 0
            cfg.m_Trigger_m_SourceType = 0
            cfg.m_Control_m_StrobeControl = 0
            cfg.m_Control_m_LaserDelay = 0
            cfg.m_Control_m_LaserWidth = 0
            cfg.m_Control_m_LaserWaveLength = 0.0
            cfg.m_Control_m_StoreToRam = 0

            proto = self.func(ctypes.c_int, ctypes.c_int, ctypes.POINTER(MeasConfigType))
            AVS_PrepareMeasure = proto(("AVS_PrepareMeasure", self.lib),
                                      ((1, "handle"), (1, "measconf")))
            ret = AVS_PrepareMeasure(self.handle, ctypes.byref(cfg))
            if ret != 0:
                print(f"PrepareMeasure error: {ret}")
                return False
        return True

    def measure(self, timeout_ms: int = 10000) -> Tuple[Optional[List[float]], Optional[float]]:
        if not self.handle:
            return None, None
        with self.lock:
            if 'linux' not in sys.platform and 'darwin' not in sys.platform:
                import ctypes.wintypes
                proto = self.func(ctypes.c_int, ctypes.c_int,
                                  ctypes.wintypes.HWND, ctypes.c_uint16)
            else:
                proto = self.func(ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_uint16)
            AVS_Measure = proto(("AVS_Measure", self.lib),
                                ((1, "handle"), (1, "windowhandle"), (1, "nummeas")))
            if AVS_Measure(self.handle, 0, 1) != 0:
                return None, None

            proto2 = self.func(ctypes.c_bool, ctypes.c_int)
            AVS_PollScan = proto2(("AVS_PollScan", self.lib), ((1, "handle"),))
            t0, tlimit = time.time(), timeout_ms / 1000.0
            while not AVS_PollScan(self.handle):
                if time.time() - t0 > tlimit:
                    return None, None
                time.sleep(0.005)

            proto3 = self.func(ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_uint32),
                               ctypes.POINTER(ctypes.c_double * MAX_NR_PIXELS))
            _, spectrum_data = proto3(("AVS_GetScopeData", self.lib),
                                      ((1, "handle"), (2, "timelabel"),
                                       (2, "spectrum")))(self.handle)
            raw = spectrum_data[0] if isinstance(spectrum_data, tuple) else spectrum_data
            spectrum = list(raw)[:self.num_pixels]
        return spectrum, time.time()

    def disconnect(self):
        if self.handle:
            try:
                proto = self.func(ctypes.c_bool, ctypes.c_int)
                proto(("AVS_Deactivate", self.lib), ((1, "handle"),))(self.handle)
            except Exception:
                pass
            self.handle = None
            self.connected = False

    def cleanup(self):
        self.disconnect()
        if self.lib:
            try:
                proto = self.func(ctypes.c_int)
                proto(("AVS_Done", self.lib),)()
            except Exception:
                pass


# =============================================================================
# B1500 controller
# =============================================================================

class B1500Controller:
    def __init__(self):
        self.rm = None
        self.inst = None
        self.idn: str = ""
        self.lock = threading.Lock()
        self.connected = False

    def _rm(self):
        try:
            return pyvisa.ResourceManager()
        except Exception:
            return pyvisa.ResourceManager("@py")

    def list_gpib(self) -> List[str]:
        if not PYVISA_AVAILABLE:
            return []
        rm = self._rm()
        try:
            return sorted(r for r in rm.list_resources() if "GPIB" in r.upper())
        except Exception:
            return []
        finally:
            try:
                rm.close()
            except Exception:
                pass

    def connect(self, resource: str, timeout_ms: int = 15000) -> Tuple[bool, str]:
        self.disconnect()
        try:
            self.rm = self._rm()
            self.inst = self.rm.open_resource(resource)
            self.inst.timeout = timeout_ms
            self.inst.write_termination = "\n"
            self.inst.read_termination = "\n"
            with self.lock:
                self.idn = self.inst.query("*IDN?").strip()
                self.inst.write("FMT 21,0")
                time.sleep(0.1)
            self.connected = True
            return True, f"Connected: {self.idn}"
        except Exception as e:
            self.disconnect()
            return False, str(e)

    def disconnect(self):
        for obj in [self.inst, self.rm]:
            if obj:
                try:
                    obj.close()
                except Exception:
                    pass
        self.inst = self.rm = None
        self.connected = False

    def configure_smu(self, smu: int, mode: str, compliance: float):
        if not self.inst:
            return
        with self.lock:
            self.inst.write(f"CN {smu}")
            time.sleep(0.05)
            self.inst.write(f"AAD {smu},1")
            self.inst.write("AV 1,0")
            if mode == "iv":
                self.inst.write(f"RI {smu},0")
            else:
                self.inst.write(f"RV {smu},0")
            self.inst.write(f"MM 1,{smu}")
            time.sleep(0.05)

    def measure_spot(self, smu: int, mode: str, setpoint: float,
                     compliance: float, dwell_s: float = 0.05) -> Tuple[float, float]:
        """Returns (voltage, current)."""
        if not self.inst:
            return (setpoint, 0.0) if mode == "iv" else (0.0, setpoint)
        try:
            with self.lock:
                if mode == "iv":
                    self.inst.write(f"DV {smu},0,{setpoint},{compliance}")
                else:
                    self.inst.write(f"DI {smu},0,{setpoint},{compliance}")
                if dwell_s > 0:
                    time.sleep(dwell_s)
                self.inst.write("XE")
                time.sleep(0.015)
                old_t = self.inst.timeout
                self.inst.timeout = 3000
                try:
                    raw = self.inst.read_raw().decode('latin-1', errors='ignore').strip()
                finally:
                    self.inst.timeout = old_t

            values = []
            for p in raw.replace(";", ",").split(","):
                p = p.strip()
                if not p:
                    continue
                for i, c in enumerate(p):
                    if c in '+-' and i > 0 and i + 1 < len(p) and p[i+1].isdigit():
                        try:
                            values.append(float(p[i:]))
                        except ValueError:
                            pass
                        break

            if values:
                m = values[0]
                return (setpoint, m) if mode == "iv" else (m, setpoint)
        except Exception as e:
            print(f"measure_spot error: {e}")
        return (setpoint, 0.0) if mode == "iv" else (0.0, setpoint)

    def output_off(self, smu: int):
        if not self.inst:
            return
        with self.lock:
            try:
                self.inst.write(f"CL {smu}")
            except Exception:
                pass


# =============================================================================
# Cycling engine
# =============================================================================

class StressCycleEngine:
    """Step-stress spectroscopy engine with legacy cycle-mode fallback."""

    def __init__(self, b1500: B1500Controller,
                 spectrometer: Optional[SpectrometerController],
                 config: CycleConfig):
        self.b1500 = b1500
        self.spec = spectrometer
        self.config = config

        self.iv_data: List[IVPoint] = []
        self.spectra_data: List[SpectrumPoint] = []
        self.stress_monitor: List[StressMonitorPoint] = []
        self.cycle_summaries: List[CycleSummary] = []

        self.running = False
        self.stop_requested = False
        self.current_phase = TestPhase.IDLE
        self.current_cycle = 0
        self.current_stress_level = 0.0
        self._mode_label = "step"

        self.on_iv_point = None
        self.on_spectrum = None
        self.on_stress_point = None
        self.on_phase_change = None
        self.on_cycle_complete = None
        self.on_step_complete = None
        self.on_progress = None
        self.on_log = None

        self.session_folder: Optional[Path] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}] {msg}"
        print(full)
        if self.on_log:
            self.on_log(full)

    def _set_phase(self, phase: TestPhase):
        self.current_phase = phase
        if self.on_phase_change:
            self.on_phase_change(phase)

    def stop(self):
        self.stop_requested = True

    def _using_step_stress(self) -> bool:
        return bool(self.config.use_step_stress)

    def _resolve_stress_levels(self) -> List[float]:
        if self._using_step_stress():
            levels = self.config.stress.stress_levels
            if not levels:
                raise ValueError(
                    "Invalid step-stress settings: check start/stop/step values"
                )
            return levels

        # Legacy fallback: fixed stress value repeated for num_cycles
        total_cycles = max(1, int(self.config.num_cycles))
        return [float(self.config.stress.value) for _ in range(total_cycles)]

    @staticmethod
    def _build_cycle_timing_plan(total_cycles: int, base_duration_s: float,
                                 time_design: str,
                                 log_points_per_decade: int) -> Tuple[List[float], List[float]]:
        """Return per-cycle stress durations and cumulative elapsed times."""
        base = max(1e-9, float(base_duration_s))
        design = (time_design or TIME_DESIGN_LINEAR).strip().lower()

        if design == TIME_DESIGN_LOG:
            ppd = int(log_points_per_decade)
            if ppd not in LOG_POINTS_PER_DECADE_OPTIONS:
                ppd = min(LOG_POINTS_PER_DECADE_OPTIONS, key=lambda v: abs(v - ppd))
            mantissas = LOG_CYCLE_MANTISSAS[ppd]
            durations = [
                base * mantissas[idx % ppd] * (10 ** (idx // ppd))
                for idx in range(total_cycles)
            ]
        else:
            durations = [base for _ in range(total_cycles)]

        durations = [max(1e-9, float(v)) for v in durations]
        cumulative_elapsed: List[float] = []
        running = 0.0
        for dt in durations:
            running += dt
            cumulative_elapsed.append(running)
        return durations, cumulative_elapsed

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        self.running = True
        self.stop_requested = False
        self.iv_data.clear()
        self.spectra_data.clear()
        self.stress_monitor.clear()
        self.cycle_summaries.clear()

        self._create_session_folder()

        # Configure B1500
        if self.b1500.connected:
            self.b1500.configure_smu(self.config.sweep.smu,
                                     self.config.sweep.mode,
                                     self.config.sweep.compliance)
            self._log("B1500 configured")

        stress_cfg = self.config.stress
        stress_levels = self._resolve_stress_levels()
        total_steps = len(stress_levels)
        self._mode_label = "step" if self._using_step_stress() else "legacy_cycle"

        if self._using_step_stress():
            self._log(f"Starting step-stress test: {total_steps} levels")
            self._log(
                f"Stress range: {stress_cfg.start_value} -> {stress_cfg.stop_value} "
                f"(step: {stress_cfg.step_value})"
            )
            self._log(f"Stress duration per level: {stress_cfg.duration_s}s")
        else:
            self._log(f"Starting legacy cycling mode: {total_steps} cycles")
            self._log(
                f"Fixed stress: {stress_cfg.value}"
                f"{'V' if stress_cfg.mode == 'voltage' else 'A'} | "
                f"duration={stress_cfg.duration_s}s"
            )

        if self.spec and self.spec.connected:
            self._log(f"Spectrometer: meas={self.config.spec.meas_integration_ms}ms, "
                      f"stress={self.config.spec.stress_integration_ms}ms")

        try:
            # Baseline measurement (step 0)
            if self.config.initial_measurement:
                self.current_cycle = 0
                self.current_stress_level = 0.0
                self._run_measurement_phase()
                if self.stop_requested:
                    self._finish()
                    return

            for step_idx, stress_level in enumerate(stress_levels, start=1):
                if self.stop_requested:
                    break

                self.current_cycle = step_idx
                self.current_stress_level = stress_level
                self._log(f"\n{'='*50}")
                self._log(f"STEP {step_idx}/{total_steps}: stress level = {stress_level:.6g}"
                          f"{'V' if stress_cfg.mode == 'voltage' else 'A'}")
                self._log(f"{'='*50}")

                self._run_stress_phase(stress_level, stress_cfg.duration_s)
                if self.stop_requested:
                    break

                self._run_measurement_phase()

                if self.on_progress:
                    self.on_progress(step_idx, total_steps)
                if self.on_cycle_complete:
                    self.on_cycle_complete(step_idx)
                if self.on_step_complete:
                    self.on_step_complete(step_idx, stress_level)

            if self.b1500.connected:
                self.b1500.output_off(self.config.sweep.smu)

            self._save_summary()
            self._finish()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"Engine error: {e}")
        finally:
            self.running = False

    def _finish(self):
        phase = TestPhase.COMPLETED if not self.stop_requested else TestPhase.STOPPED
        self._set_phase(phase)
        self._log(f"Done. IV points={len(self.iv_data)}, "
                  f"spectra={len(self.spectra_data)}, "
                  f"stress pts={len(self.stress_monitor)}")

    # ------------------------------------------------------------------
    # Measurement phase  (IV sweep + one spectrum per setpoint at high power)
    # ------------------------------------------------------------------

    def _run_measurement_phase(self):
        self._set_phase(TestPhase.MEASUREMENT)
        self._log(
            f"Measurement phase (step {self.current_cycle}, "
            f"after stress={self.current_stress_level:.6g})"
        )

        # Configure spectrometer for measurement integration time
        if self.spec and self.spec.connected and self.config.spec.enabled:
            self.spec.configure_measurement(
                self.config.spec.meas_integration_ms,
                self.config.spec.num_averages
            )

        cfg = self.config.sweep
        setpoints = cfg.setpoints
        cycle_iv: List[IVPoint] = []
        spec_index = 0

        for idx, sp in enumerate(setpoints):
            if self.stop_requested:
                break

            ts = time.time()
            v, i = self.b1500.measure_spot(cfg.smu, cfg.mode, sp, cfg.compliance, cfg.dwell_s) \
                if self.b1500.connected else ((sp, 0.0) if cfg.mode == "iv" else (0.0, sp))

            pt = IVPoint(cycle=self.current_cycle, stress_level=self.current_stress_level,
                         point_index=idx,
                         timestamp=ts, setpoint=sp, voltage=v, current=i)
            self.iv_data.append(pt)
            cycle_iv.append(pt)
            if self.on_iv_point:
                self.on_iv_point(pt)

            # Acquire one spectrum at maximum setpoint of sweep
            # (i.e. last point of the forward sweep)
            if (self.spec and self.spec.connected and self.config.spec.enabled
                    and idx == len(setpoints) - 1):
                spectrum, spec_ts = self.spec.measure(timeout_ms=5000)
                if spectrum and self.spec.wavelength:
                    sp_pt = self._make_spectrum_point(
                        "measurement", spec_index, spec_ts or ts,
                        0.0, v, i, spectrum
                    )
                    self.spectra_data.append(sp_pt)
                    spec_index += 1
                    if self.on_spectrum:
                        self.on_spectrum(sp_pt)

        self._save_measurement_cycle(cycle_iv)
        self._save_meas_spectra_cycle()

        if cycle_iv:
            self.cycle_summaries.append(self._make_summary(cycle_iv))

        self._log(f"Measurement done: {len(cycle_iv)} IV points")

    # ------------------------------------------------------------------
    # Stress phase  (constant bias + spectra at stress integration time)
    # ------------------------------------------------------------------

    def _run_stress_phase(self, stress_level: float, duration_s: float):
        self._set_phase(TestPhase.STRESS)
        cfg = self.config.stress
        self._log(f"Stress: {stress_level}"
                  f"{'V' if cfg.mode == 'voltage' else 'A'} for {duration_s}s")

        # Configure spectrometer for stress integration time
        if self.spec and self.spec.connected and self.config.spec.enabled:
            self.spec.configure_measurement(
                self.config.spec.stress_integration_ms,
                self.config.spec.num_averages
            )

        # Apply bias
        if self.b1500.connected:
            with self.b1500.lock:
                smu = self.config.sweep.smu
                if cfg.mode == "voltage":
                    self.b1500.inst.write(f"DV {smu},0,{stress_level},{cfg.compliance}")
                else:
                    self.b1500.inst.write(f"DI {smu},0,{stress_level},{cfg.compliance}")

        t0 = time.time()
        last_spec_time = 0.0
        sample_count = 0
        spec_index = 0
        cycle_stress: List[StressMonitorPoint] = []
        spec_interval_s = self.config.spec.stress_interval_ms / 1000.0

        while True:
            now = time.time()
            elapsed = now - t0
            if elapsed >= duration_s or self.stop_requested:
                break

            # Current monitor sample
            if self.b1500.connected:
                v, i = self.b1500.measure_spot(
                    self.config.sweep.smu, cfg.mode,
                    stress_level, cfg.compliance, dwell_s=0.01
                )
            else:
                v = stress_level if cfg.mode == "voltage" else 0.0
                i = 0.0 if cfg.mode == "voltage" else stress_level

            sm = StressMonitorPoint(cycle=self.current_cycle,
                                    stress_level=stress_level,
                                    timestamp=now,
                                    elapsed_s=elapsed, voltage=v, current=i)
            self.stress_monitor.append(sm)
            cycle_stress.append(sm)
            if self.on_stress_point:
                self.on_stress_point(sm)
            sample_count += 1

            # Spectrum acquisition
            if (self.spec and self.spec.connected and self.config.spec.enabled
                    and (now - last_spec_time) >= spec_interval_s):
                spectrum, spec_ts = self.spec.measure(timeout_ms=5000)
                if spectrum and self.spec.wavelength:
                    sp_pt = self._make_spectrum_point(
                        "stress", spec_index, spec_ts or now,
                        elapsed, v, i, spectrum
                    )
                    self.spectra_data.append(sp_pt)
                    spec_index += 1
                    if self.on_spectrum:
                        self.on_spectrum(sp_pt)
                last_spec_time = now

            if sample_count % max(1, int(10 / max(cfg.sample_interval_s, 0.1))) == 0:
                self._log(f"  Stress t={elapsed:.1f}s V={v:.3f}V I={i:.4e}A")

            next_t = t0 + sample_count * cfg.sample_interval_s
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(min(sleep, 0.2))

        self._save_stress_cycle(cycle_stress)
        self._save_stress_spectra_cycle()
        self._log(f"Stress done: {len(cycle_stress)} samples, {spec_index} spectra")

    # ------------------------------------------------------------------
    # Helpers – data processing
    # ------------------------------------------------------------------

    def _make_spectrum_point(self, phase: str, index: int, ts: float,
                             rel_t: float, v: float, i: float,
                             spectrum: List[float]) -> SpectrumPoint:
        peak_idx = spectrum.index(max(spectrum))
        return SpectrumPoint(
            cycle=self.current_cycle,
            stress_level=self.current_stress_level,
            phase=phase,
            index=index,
            timestamp=ts,
            relative_time=rel_t,
            bias_voltage=v,
            bias_current=i,
            wavelength=list(self.spec.wavelength),
            intensity=list(spectrum),
            peak_wavelength=self.spec.wavelength[peak_idx],
            peak_intensity=spectrum[peak_idx],
            integrated_intensity=sum(spectrum)
        )

    def _make_summary(self, data: List[IVPoint]) -> CycleSummary:
        currents = [p.current for p in data]
        peak_i = max(currents) if currents else 0.0

        # Peak intensity from the last measurement spectrum for this cycle
        meas_spectra = [s for s in self.spectra_data
                        if s.cycle == self.current_cycle and s.phase == "measurement"]
        peak_spec = max((s.peak_intensity for s in meas_spectra), default=0.0)

        return CycleSummary(
            cycle=self.current_cycle,
            stress_level=self.current_stress_level,
            timestamp=datetime.now().isoformat(),
            peak_current=peak_i,
            peak_spec_intensity=peak_spec
        )

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _create_session_folder(self):
        base = Path(self.config.output_folder)
        if not base.is_absolute():
            base = Path(__file__).parent / base
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "step_stress" if self._using_step_stress() else "cycle"
        self.session_folder = base / f"{self.config.device_name}_{suffix}_{ts}"
        self.session_folder.mkdir(parents=True, exist_ok=True)
        self._log(f"Session folder: {self.session_folder}")

    def _save_measurement_cycle(self, data: List[IVPoint]):
        if not data or not self.config.autosave or not self.session_folder:
            return
        if self._using_step_stress():
            fp = self.session_folder / (
                f"measurement_step_{self.current_cycle:03d}_"
                f"stress_{self.current_stress_level:.4f}.csv"
            )
        else:
            fp = self.session_folder / f"iv_cycle_{self.current_cycle:03d}.csv"
        with open(fp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            mode = self.config.sweep.mode.upper()
            if self._using_step_stress():
                w.writerow([f"# {mode} Sweep - Step {self.current_cycle}"])
                w.writerow([
                    "Step", "Stress_Level", "Point", "Timestamp", "Setpoint",
                    "Voltage_V", "Current_A"
                ])
            else:
                w.writerow([f"# {mode} Sweep - Cycle {self.current_cycle}"])
                w.writerow(["Point", "Timestamp", "Setpoint", "Voltage_V", "Current_A"])
            for p in data:
                if self._using_step_stress():
                    w.writerow([
                        p.cycle,
                        f"{p.stress_level:.6e}",
                        p.point_index,
                        datetime.fromtimestamp(p.timestamp).isoformat(),
                        f"{p.setpoint:.9g}", f"{p.voltage:.9g}", f"{p.current:.12g}"
                    ])
                else:
                    w.writerow([p.point_index,
                                datetime.fromtimestamp(p.timestamp).isoformat(),
                                f"{p.setpoint:.9g}", f"{p.voltage:.9g}", f"{p.current:.12g}"])

    def _save_stress_cycle(self, data: List[StressMonitorPoint]):
        if not data or not self.config.autosave or not self.session_folder:
            return
        if self._using_step_stress():
            fp = self.session_folder / (
                f"stress_step_{self.current_cycle:03d}_"
                f"level_{self.current_stress_level:.4f}.csv"
            )
        else:
            fp = self.session_folder / f"stress_cycle_{self.current_cycle:03d}.csv"
        with open(fp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if self._using_step_stress():
                w.writerow([f"# Stress Monitor - Step {self.current_cycle}"])
                w.writerow(["Step", "Stress_Level", "Elapsed_s", "Timestamp", "Voltage_V", "Current_A"])
            else:
                w.writerow([f"# Stress Monitor - Cycle {self.current_cycle}"])
                w.writerow(["Elapsed_s", "Timestamp", "Voltage_V", "Current_A"])
            for p in data:
                if self._using_step_stress():
                    w.writerow([
                        p.cycle,
                        f"{p.stress_level:.6e}",
                        f"{p.elapsed_s:.4f}",
                        datetime.fromtimestamp(p.timestamp).isoformat(),
                        f"{p.voltage:.9g}",
                        f"{p.current:.12g}"
                    ])
                else:
                    w.writerow([f"{p.elapsed_s:.4f}",
                                datetime.fromtimestamp(p.timestamp).isoformat(),
                                f"{p.voltage:.9g}", f"{p.current:.12g}"])

    def _save_meas_spectra_cycle(self):
        """Save measurement-phase spectra for the current cycle (one file per cycle)."""
        if not self.config.autosave or not self.session_folder:
            return
        pts = [s for s in self.spectra_data
               if s.cycle == self.current_cycle and s.phase == "measurement"]
        if not pts:
            return
        if self._using_step_stress():
            fp = self.session_folder / (
                f"measurement_spectra_step_{self.current_cycle:03d}_"
                f"stress_{self.current_stress_level:.4f}.csv"
            )
        else:
            fp = self.session_folder / f"meas_spectra_cycle_{self.current_cycle:03d}.csv"
        with open(fp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if self._using_step_stress():
                w.writerow([f"# Measurement Spectra - Step {self.current_cycle}"])
            else:
                w.writerow([f"# Measurement Spectra - Cycle {self.current_cycle}"])
            # Column header encodes the bias point at which the spectrum was taken
            w.writerow(["Wavelength_nm"] + [
                f"I={s.bias_current:.4e}A_V={s.bias_voltage:.4f}V" for s in pts])
            for wl_idx, wl in enumerate(pts[0].wavelength):
                row = [f"{wl:.4f}"] + [f"{s.intensity[wl_idx]:.2f}" for s in pts]
                w.writerow(row)

    def _save_stress_spectra_cycle(self):
        """Save stress-phase spectra + current trace together for the current cycle."""
        if not self.config.autosave or not self.session_folder:
            return
        pts = [s for s in self.spectra_data
               if s.cycle == self.current_cycle and s.phase == "stress"]
        if not pts:
            return
        # Spectra file: wavelength + one column per time-point
        if self._using_step_stress():
            fp_sp = self.session_folder / (
                f"stress_spectra_step_{self.current_cycle:03d}_"
                f"level_{self.current_stress_level:.4f}.csv"
            )
        else:
            fp_sp = self.session_folder / f"stress_spectra_cycle_{self.current_cycle:03d}.csv"
        with open(fp_sp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if self._using_step_stress():
                w.writerow([f"# Stress Spectra - Step {self.current_cycle}"])
            else:
                w.writerow([f"# Stress Spectra - Cycle {self.current_cycle}"])
            w.writerow(["Wavelength_nm"] + [
                f"t={s.relative_time:.2f}s_I={s.bias_current:.4e}A" for s in pts])
            for wl_idx, wl in enumerate(pts[0].wavelength):
                row = [f"{wl:.4f}"] + [f"{s.intensity[wl_idx]:.2f}" for s in pts]
                w.writerow(row)
        # Current trace file: one row per spectrum sample (elapsed time + bias)
        if self._using_step_stress():
            fp_i = self.session_folder / (
                f"stress_current_step_{self.current_cycle:03d}_"
                f"level_{self.current_stress_level:.4f}.csv"
            )
        else:
            fp_i = self.session_folder / f"stress_current_cycle_{self.current_cycle:03d}.csv"
        with open(fp_i, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if self._using_step_stress():
                w.writerow([f"# Stress Current (at spectrum times) - Step {self.current_cycle}"])
            else:
                w.writerow([f"# Stress Current (at spectrum times) - Cycle {self.current_cycle}"])
            w.writerow(["Elapsed_s", "Timestamp", "Voltage_V", "Current_A",
                        "Peak_Wavelength_nm", "Peak_Intensity"])
            for s in pts:
                w.writerow([f"{s.relative_time:.4f}",
                             datetime.fromtimestamp(s.timestamp).isoformat(),
                             f"{s.bias_voltage:.9g}", f"{s.bias_current:.12g}",
                             f"{s.peak_wavelength:.4f}", f"{s.peak_intensity:.2f}"])

    def _save_summary(self):
        if not self.session_folder:
            return
        # Step/cycle summary
        fp = self.session_folder / ("step_summary.csv" if self._using_step_stress() else "cycle_summary.csv")
        with open(fp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if self._using_step_stress():
                w.writerow(["Step", "Stress_Level", "Timestamp", "Peak_Current_A", "Peak_Spectrum_Intensity"])
            else:
                w.writerow(["Cycle", "Timestamp", "Peak_Current_A", "Peak_Spectrum_Intensity"])
            for s in self.cycle_summaries:
                if self._using_step_stress():
                    w.writerow([
                        s.cycle,
                        f"{s.stress_level:.6e}",
                        s.timestamp,
                        f"{s.peak_current:.9g}",
                        f"{s.peak_spec_intensity:.1f}"
                    ])
                else:
                    w.writerow([s.cycle, s.timestamp,
                                f"{s.peak_current:.9g}", f"{s.peak_spec_intensity:.1f}"])

        # All spectra (stacked per phase)
        for phase in ("measurement", "stress"):
            pts = [s for s in self.spectra_data if s.phase == phase]
            if not pts:
                continue
            fp2 = self.session_folder / f"spectra_{phase}_all.csv"
            with open(fp2, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                prefix = "step" if self._using_step_stress() else "cycle"
                header = ["Wavelength_nm"] + [
                    f"{prefix}{s.cycle}_t{s.relative_time:.1f}s" for s in pts]
                w.writerow(header)
                wls = pts[0].wavelength
                for idx, wl in enumerate(wls):
                    row = [f"{wl:.4f}"] + [f"{s.intensity[idx]:.2f}" for s in pts]
                    w.writerow(row)

        self._log(f"All data saved to: {self.session_folder}")


# =============================================================================
# Worker thread
# =============================================================================

class CycleWorker(QThread):
    iv_point = pyqtSignal(object)
    spectrum_acquired = pyqtSignal(object)
    stress_point = pyqtSignal(object)
    phase_change = pyqtSignal(object)
    cycle_complete = pyqtSignal(int)
    progress = pyqtSignal(int, int)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, engine: StressCycleEngine):
        super().__init__()
        self.engine = engine
        engine.on_iv_point = lambda p: self.iv_point.emit(p)
        engine.on_spectrum = lambda p: self.spectrum_acquired.emit(p)
        engine.on_stress_point = lambda p: self.stress_point.emit(p)
        engine.on_phase_change = lambda p: self.phase_change.emit(p)
        engine.on_cycle_complete = lambda c: self.cycle_complete.emit(c)
        engine.on_progress = lambda c, t: self.progress.emit(c, t)
        engine.on_log = lambda m: self.log_message.emit(m)

    def run(self):
        self.engine.run()
        self.finished_signal.emit()


# =============================================================================
# Main GUI
# =============================================================================

class StressCycleSpectroscopyGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.b1500 = B1500Controller()
        self.spectrometer = SpectrometerController()
        self.worker: Optional[CycleWorker] = None

        # In-memory plot buffers
        self._iv_by_cycle: dict = {}        # cycle → {v:[], i:[]}
        self._meas_spectra: List[SpectrumPoint] = []
        self._stress_spectra: List[SpectrumPoint] = []
        self._stress_times: List[float] = []
        self._stress_currents: List[float] = []
        self._summary_cycles: List[int] = []
        self._summary_peak_i: List[float] = []
        self._summary_peak_spec: List[float] = []

        self._plot_dirty = False
        self.setWindowTitle("B1500 Step-Stress + Spectrometer")
        self.setMinimumSize(1600, 950)
        self._setup_ui()

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(500)
        self._plot_timer.timeout.connect(self._flush_plots)
        self._plot_timer.start()

        self._refresh_resources()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(8)

        # ── left panel (controls) ─────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(320)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setSpacing(8)

        # 1. Connection
        grp = QGroupBox("1. Instrument Connection")
        g = QGridLayout(grp)

        g.addWidget(QLabel("GPIB Resource:"), 0, 0)
        self.combo_resource = QComboBox()
        self.combo_resource.setMinimumWidth(200)
        g.addWidget(self.combo_resource, 0, 1, 1, 2)

        self.btn_connect_b1500 = QPushButton("Connect B1500")
        self.btn_connect_b1500.clicked.connect(self._connect_b1500)
        g.addWidget(self.btn_connect_b1500, 1, 0)

        self.btn_disconnect_b1500 = QPushButton("Disconnect")
        self.btn_disconnect_b1500.clicked.connect(self._disconnect_b1500)
        self.btn_disconnect_b1500.setEnabled(False)
        g.addWidget(self.btn_disconnect_b1500, 1, 1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_resources)
        g.addWidget(self.btn_refresh, 1, 2)

        self.lbl_b1500 = QLabel("Not connected")
        self.lbl_b1500.setStyleSheet("color:red; font-weight:bold;")
        g.addWidget(self.lbl_b1500, 2, 0, 1, 3)

        self.btn_connect_spec = QPushButton("Connect Spectrometer")
        self.btn_connect_spec.clicked.connect(self._connect_spec)
        g.addWidget(self.btn_connect_spec, 3, 0, 1, 2)

        self.btn_disconnect_spec = QPushButton("Disconnect")
        self.btn_disconnect_spec.clicked.connect(self._disconnect_spec)
        self.btn_disconnect_spec.setEnabled(False)
        g.addWidget(self.btn_disconnect_spec, 3, 2)

        self.lbl_spec = QLabel("Not connected")
        self.lbl_spec.setStyleSheet("color:red; font-weight:bold;")
        g.addWidget(self.lbl_spec, 4, 0, 1, 3)

        lv.addWidget(grp)

        # 2. Device
        grp = QGroupBox("2. Device Settings")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Device Name:"), 0, 0)
        self.edit_device = QLineEdit("Device_001")
        g.addWidget(self.edit_device, 0, 1, 1, 2)
        g.addWidget(QLabel("SMU:"), 1, 0)
        self.spin_smu = QSpinBox()
        self.spin_smu.setRange(1, 10)
        self.spin_smu.setValue(1)
        g.addWidget(self.spin_smu, 1, 1)
        lv.addWidget(grp)

        # 3. IV sweep
        grp = QGroupBox("3. IV Characterization (Before/After Each Stress)")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Mode:"), 0, 0)
        self.combo_iv_mode = QComboBox()
        self.combo_iv_mode.addItems(["IV (Source V, Meas I)", "VI (Source I, Meas V)"])
        g.addWidget(self.combo_iv_mode, 0, 1, 1, 3)

        g.addWidget(QLabel("Start:"), 1, 0)
        self.spin_iv_start = QDoubleSpinBox()
        self.spin_iv_start.setRange(-200, 200); self.spin_iv_start.setDecimals(4)
        self.spin_iv_start.setValue(0); self.spin_iv_start.setSuffix(" V")
        g.addWidget(self.spin_iv_start, 1, 1)

        g.addWidget(QLabel("Stop:"), 1, 2)
        self.spin_iv_stop = QDoubleSpinBox()
        self.spin_iv_stop.setRange(-200, 200); self.spin_iv_stop.setDecimals(4)
        self.spin_iv_stop.setValue(2.0); self.spin_iv_stop.setSuffix(" V")
        g.addWidget(self.spin_iv_stop, 1, 3)

        g.addWidget(QLabel("Points:"), 2, 0)
        self.spin_iv_steps = QSpinBox()
        self.spin_iv_steps.setRange(2, 501); self.spin_iv_steps.setValue(21)
        g.addWidget(self.spin_iv_steps, 2, 1)

        g.addWidget(QLabel("Dwell (s):"), 2, 2)
        self.spin_iv_dwell = QDoubleSpinBox()
        self.spin_iv_dwell.setRange(0, 10); self.spin_iv_dwell.setDecimals(3)
        self.spin_iv_dwell.setValue(0.1)
        g.addWidget(self.spin_iv_dwell, 2, 3)

        g.addWidget(QLabel("Compliance:"), 3, 0)
        self.spin_iv_compliance = QDoubleSpinBox()
        self.spin_iv_compliance.setRange(1e-12, 1.0); self.spin_iv_compliance.setDecimals(6)
        self.spin_iv_compliance.setValue(0.1); self.spin_iv_compliance.setSuffix(" A")
        g.addWidget(self.spin_iv_compliance, 3, 1, 1, 3)
        lv.addWidget(grp)

        # 4. Step stress
        grp = QGroupBox("4. Stress Configuration")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Stress Mode:"), 0, 0)
        self.combo_stress_mode = QComboBox()
        self.combo_stress_mode.addItems(["Voltage Stress", "Current Stress"])
        self.combo_stress_mode.currentIndexChanged.connect(self._on_stress_mode_changed)
        g.addWidget(self.combo_stress_mode, 0, 1, 1, 3)

        self.lbl_stress_start = QLabel("Start Stress (V):")
        g.addWidget(self.lbl_stress_start, 1, 0)
        self.spin_stress_start = QDoubleSpinBox()
        self.spin_stress_start.setRange(-200, 200); self.spin_stress_start.setDecimals(6)
        self.spin_stress_start.setValue(2.0)
        g.addWidget(self.spin_stress_start, 1, 1)

        self.lbl_stress_stop = QLabel("Stop Stress (V):")
        g.addWidget(self.lbl_stress_stop, 1, 2)
        self.spin_stress_stop = QDoubleSpinBox()
        self.spin_stress_stop.setRange(-200, 200); self.spin_stress_stop.setDecimals(6)
        self.spin_stress_stop.setValue(5.0)
        g.addWidget(self.spin_stress_stop, 1, 3)

        self.lbl_stress_step = QLabel("Stress Step (V):")
        g.addWidget(self.lbl_stress_step, 2, 0)
        self.spin_stress_step = QDoubleSpinBox()
        self.spin_stress_step.setRange(-100, 100); self.spin_stress_step.setDecimals(6)
        self.spin_stress_step.setValue(0.5)
        g.addWidget(self.spin_stress_step, 2, 1)

        g.addWidget(QLabel("Duration per Step (s):"), 2, 2)
        self.spin_stress_duration = QDoubleSpinBox()
        self.spin_stress_duration.setRange(1e-6, 1_000_000)
        self.spin_stress_duration.setDecimals(6)
        self.spin_stress_duration.setSingleStep(0.01)
        self.spin_stress_duration.setValue(60.0)
        g.addWidget(self.spin_stress_duration, 2, 3)

        g.addWidget(QLabel("Monitor Interval (s):"), 3, 0)
        self.spin_stress_interval = QDoubleSpinBox()
        self.spin_stress_interval.setRange(0.1, 60); self.spin_stress_interval.setDecimals(2)
        self.spin_stress_interval.setValue(1.0)
        g.addWidget(self.spin_stress_interval, 3, 1)

        self.lbl_stress_comp = QLabel("Compliance (A):")
        g.addWidget(self.lbl_stress_comp, 3, 2)
        self.spin_stress_comp = QDoubleSpinBox()
        self.spin_stress_comp.setRange(1e-12, 1.0); self.spin_stress_comp.setDecimals(6)
        self.spin_stress_comp.setValue(0.1)
        g.addWidget(self.spin_stress_comp, 3, 3)

        self.spin_stress_start.valueChanged.connect(lambda *_: self._update_step_count_hint())
        self.spin_stress_stop.valueChanged.connect(lambda *_: self._update_step_count_hint())
        self.spin_stress_step.valueChanged.connect(lambda *_: self._update_step_count_hint())
        lv.addWidget(grp)

        # 5. Step settings
        grp = QGroupBox("5. Step Settings")
        g = QGridLayout(grp)
        self.check_initial = QCheckBox("Initial Measurement (Baseline)")
        self.check_initial.setChecked(True)
        g.addWidget(self.check_initial, 0, 0, 1, 4)

        self.lbl_step_count = QLabel("Steps: -")
        self.lbl_step_count.setStyleSheet("color:#005A9C; font-size:10pt; font-weight:bold;")
        self.lbl_step_count.setWordWrap(True)
        g.addWidget(self.lbl_step_count, 1, 0, 1, 4)

        self._update_step_count_hint()
        lv.addWidget(grp)

        # 6. Spectrometer settings  ← KEY: separate integration times
        grp = QGroupBox("6. Spectrometer Settings")
        g = QGridLayout(grp)
        self.check_spec_enable = QCheckBox("Enable Spectrometer")
        self.check_spec_enable.setChecked(True)
        g.addWidget(self.check_spec_enable, 0, 0, 1, 4)

        g.addWidget(QLabel("Meas. Integration (ms):"), 1, 0)
        self.spin_meas_integ = QDoubleSpinBox()
        self.spin_meas_integ.setRange(0.001, 60000); self.spin_meas_integ.setDecimals(3)
        self.spin_meas_integ.setValue(100.0)
        self.spin_meas_integ.setToolTip("Integration time used during the IV characterization phase")
        g.addWidget(self.spin_meas_integ, 1, 1)

        g.addWidget(QLabel("Stress Integration (ms):"), 1, 2)
        self.spin_stress_integ = QDoubleSpinBox()
        self.spin_stress_integ.setRange(0.001, 60000); self.spin_stress_integ.setDecimals(3)
        self.spin_stress_integ.setValue(100.0)
        self.spin_stress_integ.setToolTip("Integration time used during the stress phase")
        g.addWidget(self.spin_stress_integ, 1, 3)

        g.addWidget(QLabel("Averages:"), 2, 0)
        self.spin_spec_avg = QSpinBox()
        self.spin_spec_avg.setRange(1, 100); self.spin_spec_avg.setValue(1)
        g.addWidget(self.spin_spec_avg, 2, 1)

        g.addWidget(QLabel("Stress Spec. Interval (ms):"), 2, 2)
        self.spin_stress_spec_interval = QDoubleSpinBox()
        self.spin_stress_spec_interval.setRange(100, 60000)
        self.spin_stress_spec_interval.setValue(1000)
        self.spin_stress_spec_interval.setToolTip("Time between spectrum acquisitions during stress")
        g.addWidget(self.spin_stress_spec_interval, 2, 3)
        lv.addWidget(grp)

        # 7. Save
        grp = QGroupBox("7. Save Data")
        g = QGridLayout(grp)
        g.addWidget(QLabel("Save Folder:"), 0, 0)
        self.edit_folder = QLineEdit(str(Path(__file__).resolve().parent / "results"))
        self.edit_folder.setReadOnly(True)
        g.addWidget(self.edit_folder, 0, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_folder)
        g.addWidget(btn_browse, 0, 2)
        self.check_autosave = QCheckBox("Auto-save after each step")
        self.check_autosave.setChecked(True)
        g.addWidget(self.check_autosave, 1, 0, 1, 3)
        lv.addWidget(grp)

        # 8. Control
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start Test")
        self.btn_start.setMinimumHeight(42)
        self.btn_start.setStyleSheet("background:#4CAF50;color:white;font-weight:bold;font-size:13px;")
        self.btn_start.clicked.connect(self._start_test)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("◼ Stop")
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background:#f44336;color:white;font-weight:bold;font-size:13px;")
        self.btn_stop.clicked.connect(self._stop_test)
        btn_row.addWidget(self.btn_stop)
        lv.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        lv.addWidget(self.progress_bar)

        self.lbl_phase = QLabel("Phase: IDLE")
        self.lbl_phase.setStyleSheet("font-weight:bold;")
        lv.addWidget(self.lbl_phase)

        log_grp = QGroupBox("Log")
        log_l = QVBoxLayout(log_grp)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(130)
        log_l.addWidget(self.txt_log)
        lv.addWidget(log_grp)
        lv.addStretch()

        scroll.setWidget(left)

        # ── right panel (plots) ───────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)

        tabs = QTabWidget()

        # Tab 1: Live plots
        plot_tab = QWidget()
        pl = QVBoxLayout(plot_tab)
        self.figure = Figure(figsize=(14, 9), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], figure=self.figure)
        self.ax_iv      = self.figure.add_subplot(gs[0, 0])
        self.ax_meas    = self.figure.add_subplot(gs[0, 1])
        self.ax_cbar_m  = self.figure.add_subplot(gs[0, 2])
        self.ax_stress  = self.figure.add_subplot(gs[1, 0])
        self.ax_str_sp  = self.figure.add_subplot(gs[1, 1])
        self.ax_cbar_s  = self.figure.add_subplot(gs[1, 2])
        self._cbar_m = None
        self._cbar_s = None
        self._setup_plot_axes()
        self.figure.tight_layout(pad=1.5)
        pl.addWidget(self.canvas)
        tabs.addTab(plot_tab, "Live Plots")

        # Tab 2: Degradation
        deg_tab = QWidget()
        dl = QVBoxLayout(deg_tab)
        self.fig_deg = Figure(figsize=(12, 5), dpi=100)
        self.canvas_deg = FigureCanvas(self.fig_deg)
        self.ax_deg_i = self.fig_deg.add_subplot(1, 2, 1)
        self.ax_deg_s = self.fig_deg.add_subplot(1, 2, 2)
        self._setup_deg_axes()
        self.fig_deg.tight_layout()
        dl.addWidget(self.canvas_deg)
        tabs.addTab(deg_tab, "Degradation")

        # Tab 3: Summary table
        tbl_tab = QWidget()
        tbl_l = QVBoxLayout(tbl_tab)
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(5)
        self.summary_table.setHorizontalHeaderLabels(
            ["Step", "Stress Level", "Timestamp", "Peak Current (A)", "Peak Spec. Intensity"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl_l.addWidget(self.summary_table)
        tabs.addTab(tbl_tab, "Step Summary")

        rv.addWidget(tabs)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([500, 900])
        layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _setup_plot_axes(self):
        self.ax_iv.set_title("IV Curves by Step"); self.ax_iv.set_xlabel("Voltage (V)")
        self.ax_iv.set_ylabel("Current (A)"); self.ax_iv.grid(True, alpha=0.3)

        self.ax_meas.set_title("Measurement Spectra by Step")
        self.ax_meas.set_xlabel("Wavelength (nm)"); self.ax_meas.set_ylabel("Intensity + Offset")
        self.ax_meas.grid(True, alpha=0.3)
        self.ax_cbar_m.set_visible(False)

        self.ax_stress.set_title("Current During Stress"); self.ax_stress.set_xlabel("Time (s)")
        self.ax_stress.set_ylabel("Current (A)"); self.ax_stress.grid(True, alpha=0.3)

        self.ax_str_sp.set_title("Stress Spectra (latest step)")
        self.ax_str_sp.set_xlabel("Wavelength (nm)"); self.ax_str_sp.set_ylabel("Intensity + Offset")
        self.ax_str_sp.grid(True, alpha=0.3)
        self.ax_cbar_s.set_visible(False)

    def _setup_deg_axes(self):
        self.ax_deg_i.set_title("Peak Current vs Step"); self.ax_deg_i.set_xlabel("Step")
        self.ax_deg_i.set_ylabel("Peak Current (A)"); self.ax_deg_i.grid(True, alpha=0.3)
        self.ax_deg_s.set_title("Peak Spec. Intensity vs Step"); self.ax_deg_s.set_xlabel("Step")
        self.ax_deg_s.set_ylabel("Peak Intensity"); self.ax_deg_s.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Instrument connection
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{ts}] {msg}")
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum())

    def _refresh_resources(self):
        self.combo_resource.clear()
        if PYVISA_AVAILABLE:
            resources = self.b1500.list_gpib()
            self.combo_resource.addItems(resources)
            self._log(f"Found {len(resources)} GPIB resource(s)")

    def _connect_b1500(self):
        res = self.combo_resource.currentText()
        if not res:
            QMessageBox.warning(self, "Connect", "Select a GPIB resource first")
            return
        ok, msg = self.b1500.connect(res)
        if ok:
            self.lbl_b1500.setText(f"Connected: {self.b1500.idn[:45]}…")
            self.lbl_b1500.setStyleSheet("color:green;font-weight:bold;")
            self.btn_connect_b1500.setEnabled(False)
            self.btn_disconnect_b1500.setEnabled(True)
        else:
            QMessageBox.critical(self, "Connection Failed", msg)
        self._log(msg)

    def _disconnect_b1500(self):
        self.b1500.disconnect()
        self.lbl_b1500.setText("Not connected")
        self.lbl_b1500.setStyleSheet("color:red;font-weight:bold;")
        self.btn_connect_b1500.setEnabled(True)
        self.btn_disconnect_b1500.setEnabled(False)
        self._log("B1500 disconnected")

    def _connect_spec(self):
        self._log("Initialising spectrometer…")
        if not self.spectrometer.initialize():
            QMessageBox.warning(self, "Spectrometer", "Failed to initialise SDK")
            return
        if not self.spectrometer.connect_device(0):
            QMessageBox.warning(self, "Spectrometer", "No device found")
            return
        wl = self.spectrometer.wavelength
        self.lbl_spec.setText(f"Connected: {self.spectrometer.num_pixels}px  "
                              f"{wl[0]:.0f}–{wl[-1]:.0f}nm")
        self.lbl_spec.setStyleSheet("color:green;font-weight:bold;")
        self.btn_connect_spec.setEnabled(False)
        self.btn_disconnect_spec.setEnabled(True)
        self._log("Spectrometer connected")

    def _disconnect_spec(self):
        self.spectrometer.cleanup()
        self.lbl_spec.setText("Not connected")
        self.lbl_spec.setStyleSheet("color:red;font-weight:bold;")
        self.btn_connect_spec.setEnabled(True)
        self.btn_disconnect_spec.setEnabled(False)
        self._log("Spectrometer disconnected")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _on_stress_mode_changed(self):
        is_v = self.combo_stress_mode.currentIndex() == 0
        self.lbl_stress_start.setText("Start Stress (V):" if is_v else "Start Stress (A):")
        self.lbl_stress_stop.setText("Stop Stress (V):" if is_v else "Stop Stress (A):")
        self.lbl_stress_step.setText("Stress Step (V):" if is_v else "Stress Step (A):")
        self.lbl_stress_comp.setText("Compliance (A):" if is_v else "Compliance (V):")
        if is_v:
            self.spin_stress_start.setRange(-200, 200)
            self.spin_stress_stop.setRange(-200, 200)
            self.spin_stress_step.setRange(-100, 100)
            self.spin_stress_comp.setRange(1e-12, 1.0)
        else:
            self.spin_stress_start.setRange(-1, 1)
            self.spin_stress_stop.setRange(-1, 1)
            self.spin_stress_step.setRange(-1, 1)
            self.spin_stress_comp.setRange(0, 200)

    def _calculate_stress_levels(self) -> List[float]:
        start = float(self.spin_stress_start.value())
        stop = float(self.spin_stress_stop.value())
        step = float(self.spin_stress_step.value())

        if abs(step) < 1e-12:
            return []
        if step > 0 and start > stop:
            return []
        if step < 0 and start < stop:
            return []

        levels: List[float] = []
        current = start
        if step > 0:
            while current <= stop + 1e-9:
                levels.append(current)
                current += step
        else:
            while current >= stop - 1e-9:
                levels.append(current)
                current += step
        return levels

    def _update_step_count_hint(self):
        if not hasattr(self, "lbl_step_count"):
            return

        levels = self._calculate_stress_levels()
        if not levels:
            self.lbl_step_count.setText(
                "Steps: invalid range or step size. "
                "Use positive step for start<=stop, negative for start>=stop."
            )
            self.lbl_step_count.setStyleSheet("color:#C0392B; font-size:10pt; font-weight:bold;")
            return

        self.lbl_step_count.setStyleSheet("color:#005A9C; font-size:10pt; font-weight:bold;")
        self.lbl_step_count.setText(
            f"Steps: {len(levels)}  (first={levels[0]:.6g}, last={levels[-1]:.6g})"
        )

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder",
                                                  self.edit_folder.text())
        if folder:
            self.edit_folder.setText(folder)

    def _build_config(self) -> CycleConfig:
        iv_mode = "iv" if self.combo_iv_mode.currentIndex() == 0 else "vi"
        st_mode = "voltage" if self.combo_stress_mode.currentIndex() == 0 else "current"
        levels = self._calculate_stress_levels()
        step_count = max(1, len(levels))
        return CycleConfig(
            sweep=SweepConfig(
                smu=self.spin_smu.value(),
                mode=iv_mode,
                start=self.spin_iv_start.value(),
                stop=self.spin_iv_stop.value(),
                steps=self.spin_iv_steps.value(),
                dwell_s=self.spin_iv_dwell.value(),
                compliance=self.spin_iv_compliance.value()
            ),
            stress=StressConfig(
                mode=st_mode,
                value=self.spin_stress_start.value(),
                start_value=self.spin_stress_start.value(),
                stop_value=self.spin_stress_stop.value(),
                step_value=self.spin_stress_step.value(),
                duration_s=self.spin_stress_duration.value(),
                sample_interval_s=self.spin_stress_interval.value(),
                compliance=self.spin_stress_comp.value()
            ),
            spec=SpecConfig(
                enabled=self.check_spec_enable.isChecked() and self.spectrometer.connected,
                meas_integration_ms=self.spin_meas_integ.value(),
                stress_integration_ms=self.spin_stress_integ.value(),
                num_averages=self.spin_spec_avg.value(),
                stress_interval_ms=self.spin_stress_spec_interval.value()
            ),
            num_cycles=step_count,
            use_step_stress=True,
            initial_measurement=self.check_initial.isChecked(),
            output_folder=self.edit_folder.text(),
            device_name=self.edit_device.text(),
            autosave=self.check_autosave.isChecked()
        )

    # ------------------------------------------------------------------
    # Test control
    # ------------------------------------------------------------------

    def _start_test(self):
        if not self.b1500.connected:
            QMessageBox.warning(self, "Start", "Connect B1500 first")
            return

        levels = self._calculate_stress_levels()
        if not levels:
            QMessageBox.warning(
                self,
                "Invalid Step Settings",
                "Invalid start/stop/step configuration for step stress."
            )
            return

        # Clear buffers
        self._iv_by_cycle.clear()
        self._meas_spectra.clear()
        self._stress_spectra.clear()
        self._stress_times.clear()
        self._stress_currents.clear()
        self._summary_cycles.clear()
        self._summary_peak_i.clear()
        self._summary_peak_spec.clear()
        self.summary_table.setRowCount(0)

        for ax in [self.ax_iv, self.ax_meas, self.ax_stress, self.ax_str_sp]:
            ax.clear()
        self.ax_cbar_m.clear(); self.ax_cbar_m.set_visible(False)
        self.ax_cbar_s.clear(); self.ax_cbar_s.set_visible(False)
        self._cbar_m = self._cbar_s = None
        self._setup_plot_axes()
        self.canvas.draw()

        config = self._build_config()
        spec = self.spectrometer if config.spec.enabled else None
        engine = StressCycleEngine(self.b1500, spec, config)

        self.worker = CycleWorker(engine)
        self.worker.iv_point.connect(self._on_iv_point)
        self.worker.spectrum_acquired.connect(self._on_spectrum)
        self.worker.stress_point.connect(self._on_stress_point)
        self.worker.phase_change.connect(self._on_phase_change)
        self.worker.cycle_complete.connect(self._on_cycle_complete)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.finished_signal.connect(self._on_test_done)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.worker.start()

    def _stop_test(self):
        if self.worker:
            self.worker.engine.stop()
            self._log("Stop requested…")

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_iv_point(self, pt: IVPoint):
        c = pt.cycle
        if c not in self._iv_by_cycle:
            self._iv_by_cycle[c] = {"v": [], "i": []}
        self._iv_by_cycle[c]["v"].append(pt.voltage)
        self._iv_by_cycle[c]["i"].append(pt.current)
        self._plot_dirty = True

    def _on_spectrum(self, sp: SpectrumPoint):
        if sp.phase == "measurement":
            self._meas_spectra.append(sp)
        else:
            self._stress_spectra.append(sp)
        self._plot_dirty = True

    def _on_stress_point(self, sm: StressMonitorPoint):
        self._stress_times.append(sm.elapsed_s)
        self._stress_currents.append(sm.current)
        self._plot_dirty = True

    def _on_phase_change(self, phase: TestPhase):
        self.lbl_phase.setText(f"Phase: {phase.value.upper()}")
        if phase == TestPhase.STRESS:
            # Reset stress plot buffers for new cycle
            self._stress_times.clear()
            self._stress_currents.clear()

    def _on_cycle_complete(self, cycle: int):
        # Update summary table
        if self.worker and self.worker.engine.cycle_summaries:
            s = self.worker.engine.cycle_summaries[-1]
            self._summary_cycles.append(s.cycle)
            self._summary_peak_i.append(s.peak_current)
            self._summary_peak_spec.append(s.peak_spec_intensity)
            row = self.summary_table.rowCount()
            self.summary_table.insertRow(row)
            self.summary_table.setItem(row, 0, QTableWidgetItem(str(s.cycle)))
            self.summary_table.setItem(row, 1, QTableWidgetItem(f"{s.stress_level:.6e}"))
            self.summary_table.setItem(row, 2, QTableWidgetItem(s.timestamp[:19]))
            self.summary_table.setItem(row, 3,
                QTableWidgetItem(f"{s.peak_current:.4e}"))
            self.summary_table.setItem(row, 4,
                QTableWidgetItem(f"{s.peak_spec_intensity:.1f}"))

    def _on_progress(self, current: int, total: int):
        pct = int(100 * current / total) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.status_bar.showMessage(f"Step {current}/{total}")

    def _on_test_done(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self._flush_plots()
        self._update_degradation_plot()
        self._log("Test complete")
        self.status_bar.showMessage("Test complete")

    # ------------------------------------------------------------------
    # Plot rendering
    # ------------------------------------------------------------------

    def _flush_plots(self):
        if not self._plot_dirty:
            return
        self._plot_dirty = False
        self._update_iv_plot()
        self._update_meas_spectra_plot()
        self._update_stress_current_plot()
        self._update_stress_spectra_plot()
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _update_iv_plot(self):
        self.ax_iv.clear()
        self.ax_iv.set_title("IV Curves by Step")
        self.ax_iv.set_xlabel("Voltage (V)")
        self.ax_iv.set_ylabel("Current (A)")
        self.ax_iv.grid(True, alpha=0.3)
        cycles = sorted(self._iv_by_cycle.keys())
        if not cycles:
            return
        cmap = plt.cm.plasma
        norm = Normalize(vmin=cycles[0], vmax=max(cycles[-1], cycles[0] + 1))
        for c in cycles:
            d = self._iv_by_cycle[c]
            color = cmap(norm(c))
            self.ax_iv.plot(d["v"], d["i"], color=color, linewidth=1, alpha=0.8)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Small inline colorbar hint via annotation
        self.ax_iv.annotate(f"Steps: {cycles[0]}-{cycles[-1]}",
                            xy=(0.02, 0.98), xycoords='axes fraction',
                            fontsize=7, va='top', style='italic')

    def _draw_waterfall(self, ax, ax_cbar, spectra: List[SpectrumPoint],
                        cbar_label: str, cbar_ref: Optional[object],
                        color_by: str = "cycle"):
        ax.clear()
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity + Offset")
        ax.grid(True, alpha=0.3)
        if not spectra:
            ax_cbar.clear(); ax_cbar.set_visible(False)
            return None
        if color_by == "cycle":
            vals = [s.cycle for s in spectra]
        else:
            vals = [s.relative_time for s in spectra]
        norm = Normalize(vmin=min(vals), vmax=max(vals) if max(vals) > min(vals) else min(vals) + 1)
        cmap = plt.cm.viridis
        max_int = max(max(s.intensity) for s in spectra)
        offset_step = max_int * 0.1 if max_int > 0 else 1.0
        for idx, sp in enumerate(spectra):
            color = cmap(norm(vals[idx]))
            y = [v + idx * offset_step for v in sp.intensity]
            ax.plot(sp.wavelength, y, color=color, linewidth=0.5, alpha=0.8)
        ax.annotate(f"n={len(spectra)}", xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=7, va='top', style='italic')
        ax_cbar.cla()
        ax_cbar.set_visible(True)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        return self.figure.colorbar(sm, cax=ax_cbar, label=cbar_label)

    def _update_meas_spectra_plot(self):
        self.ax_meas.set_title("Measurement Spectra by Step")
        self._cbar_m = self._draw_waterfall(
            self.ax_meas, self.ax_cbar_m, self._meas_spectra,
            "Step", self._cbar_m, color_by="cycle")

    def _update_stress_spectra_plot(self):
        self.ax_str_sp.set_title("Stress Spectra (current step)")
        # Show only spectra from the current stress step
        last_cycle = max((s.cycle for s in self._stress_spectra), default=0) if self._stress_spectra else 0
        subset = [s for s in self._stress_spectra if s.cycle == last_cycle]
        self._cbar_s = self._draw_waterfall(
            self.ax_str_sp, self.ax_cbar_s, subset,
            "Time (s)", self._cbar_s, color_by="time")

    def _update_stress_current_plot(self):
        self.ax_stress.clear()
        self.ax_stress.set_title("Current During Stress")
        self.ax_stress.set_xlabel("Time (s)")
        self.ax_stress.set_ylabel("Current (A)")
        self.ax_stress.grid(True, alpha=0.3)
        if self._stress_times:
            self.ax_stress.plot(self._stress_times, self._stress_currents,
                                'b-', linewidth=1)

    def _update_degradation_plot(self):
        self.ax_deg_i.clear(); self.ax_deg_s.clear()
        self._setup_deg_axes()
        if self._summary_cycles:
            self.ax_deg_i.plot(self._summary_cycles, self._summary_peak_i,
                               'bo-', markersize=5, linewidth=1)
            self.ax_deg_s.plot(self._summary_cycles, self._summary_peak_spec,
                               'rs-', markersize=5, linewidth=1)
        self.fig_deg.tight_layout()
        try:
            self.canvas_deg.draw_idle()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.engine.stop()
            self.worker.wait(3000)
        self.b1500.disconnect()
        self.spectrometer.cleanup()
        event.accept()


# =============================================================================
# Entry point
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    win = StressCycleSpectroscopyGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
