#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B1500 + Avantes Spectrometer – Two-Phase Stress/Recovery Cycling Test

Flow:
    Part 1: Measurement → Stress → Measurement  (N stress cycles)
    Part 2: Measurement → Recovery → Measurement  (N recovery cycles)

The two parts run back-to-back without reconnecting instruments.
Data layout:
    <save_folder>/<device>_spectrometer_stress_recovery_scycles_<timestamp>/
        stress/
            iv_cycle_000.csv, stress_cycle_001.csv, measurement_spectra_cycle_000.csv, ...
        recovery/
            iv_cycle_000.csv, stress_cycle_001.csv, measurement_spectra_cycle_000.csv, ...

Notes:
- Spectrometer integration time for Measurement and Stress/Recovery are configured separately.
- Stress and Recovery have independent sweep, bias, duration, spectrum interval and cycle settings.
- This file reuses the verified B1500, Avantes DLL and one-phase spectroscopy engine from
  b1500_stress_cycle_spectroscopy.py, and adds a two-phase GUI/worker + fixed output folders.

Author: Veronica GaoZhan
Date: June 2026
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
from dataclasses import replace

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QTextEdit, QFileDialog, QMessageBox, QProgressBar,
    QSplitter, QScrollArea, QTabWidget, QCheckBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# Reuse low-level instruments and spectroscopy cycling logic from the existing file.
from b1500_Step_stress_spectroscopy import (
    B1500Controller,
    SpectrometerController,
    SweepConfig,
    StressConfig,
    SpecConfig,
    CycleConfig,
    StressCycleEngine,
    TestPhase,
    TIME_DESIGN_LINEAR,
    TIME_DESIGN_LOG,
    LOG_POINTS_PER_DECADE_OPTIONS,
    LOG_CYCLE_MANTISSAS,
)


# =============================================================================
# Time design constants and helpers
# =============================================================================

# Note: TIME_DESIGN_LINEAR, TIME_DESIGN_LOG, LOG_POINTS_PER_DECADE_OPTIONS, and
# LOG_CYCLE_MANTISSAS are imported from b1500_Step_stress_spectroscopy.py


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _safe_name(text: str) -> str:
    """Make a filesystem-friendly device/session name."""
    bad = '<>:"/\\|?*'
    out = ''.join('_' if c in bad else c for c in text.strip())
    return out or 'Device_001'


def _spin(value=0.0, minimum=-1e9, maximum=1e9, decimals=6, step=0.1, suffix=''):
    w = QDoubleSpinBox()
    w.setRange(minimum, maximum)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    w.setMinimumWidth(130)
    return w


# -----------------------------------------------------------------------------
# Engine subclass: force output to <base>/stress and <base>/recovery
# -----------------------------------------------------------------------------

class FixedFolderSpectroscopyEngine(StressCycleEngine):
    """One-phase spectroscopy engine with a preselected session folder."""

    def __init__(self, *args, fixed_folder: Path, phase_label: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_folder = Path(fixed_folder)
        self.phase_label = phase_label

    def _create_session_folder(self):
        self.session_folder = self.fixed_folder
        self.session_folder.mkdir(parents=True, exist_ok=True)
        self._log(f"{self.phase_label.capitalize()} data folder: {self.session_folder}")

    def _set_phase(self, phase):
        # Keep original enum for engine logic, but the worker emits phase label separately.
        super()._set_phase(phase)


class TwoPhaseSpectroscopyWorker(QThread):
    iv_point = pyqtSignal(object, str)             # point, phase_label
    spectrum_acquired = pyqtSignal(object, str)    # spectrum, phase_label
    stress_point = pyqtSignal(object, str)         # monitor point, phase_label
    phase_change = pyqtSignal(str, object)         # phase_label, TestPhase
    cycle_complete = pyqtSignal(str, int)
    progress = pyqtSignal(str, int, int)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, b1500, spectrometer, stress_cfg, recovery_cfg, base_folder: Path):
        super().__init__()
        self.b1500 = b1500
        self.spectrometer = spectrometer
        self.stress_cfg = stress_cfg
        self.recovery_cfg = recovery_cfg
        self.base_folder = Path(base_folder)
        self.engines = []
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True
        for e in self.engines:
            e.stop()

    def _bind(self, engine, label):
        engine.on_iv_point = lambda p, lab=label: self.iv_point.emit(p, lab)
        engine.on_spectrum = lambda p, lab=label: self.spectrum_acquired.emit(p, lab)
        engine.on_stress_point = lambda p, lab=label: self.stress_point.emit(p, lab)
        engine.on_phase_change = lambda ph, lab=label: self.phase_change.emit(lab, ph)
        engine.on_cycle_complete = lambda c, lab=label: self.cycle_complete.emit(lab, c)
        engine.on_progress = lambda c, n, lab=label: self.progress.emit(lab, c, n)
        engine.on_log = self.log_message.emit

    def run(self):
        try:
            self.base_folder.mkdir(parents=True, exist_ok=True)
            phases = [
                ('stress', self.stress_cfg, self.base_folder / 'stress'),
                ('recovery', self.recovery_cfg, self.base_folder / 'recovery'),
            ]
            for label, cfg, folder in phases:
                if self._stop_requested:
                    break
                self.log_message.emit(f"\n========== START {label.upper()} PART ==========")
                engine = FixedFolderSpectroscopyEngine(
                    self.b1500, self.spectrometer, cfg,
                    fixed_folder=folder, phase_label=label
                )
                self._bind(engine, label)
                self.engines.append(engine)
                engine.run()
                if self._stop_requested:
                    break
                self.log_message.emit(f"========== END {label.upper()} PART ==========\n")
            if self.b1500 and self.b1500.connected:
                try:
                    self.b1500.output_off(self.stress_cfg.sweep.smu)
                    self.b1500.output_off(self.recovery_cfg.sweep.smu)
                except Exception:
                    pass
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.log_message.emit(f"Two-phase worker error: {exc}")
        finally:
            self.finished_signal.emit()


# -----------------------------------------------------------------------------
# Scrollable per-phase settings panel
# -----------------------------------------------------------------------------

class PhaseConfigWidget(QWidget):
    """Independent settings for Stress or Recovery. Put inside a scroll area."""

    def __init__(self, title: str, is_recovery: bool = False):
        super().__init__()
        self.title = title
        self.is_recovery = is_recovery
        default_cycles = 5 if self.is_recovery else 10
        self._active_time_design = TIME_DESIGN_LINEAR
        self._cycle_counts_by_design = {
            TIME_DESIGN_LINEAR: int(default_cycles),
            TIME_DESIGN_LOG: int(default_cycles),
        }
        self._build_ui(default_cycles)

    def _build_ui(self, default_num_cycles: int):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # Cycle settings
        cycle_box = QGroupBox(f"{self.title}: Cycles")
        g = QGridLayout(cycle_box)
        self.num_cycles = QSpinBox()
        self.num_cycles.setRange(1, 10000)
        self.num_cycles.setValue(default_num_cycles)
        self.num_cycles.valueChanged.connect(self._on_cycle_no_changed)

        self.initial_meas = QCheckBox("Initial measurement before cycle 1")
        self.initial_meas.setChecked(True)
        self.initial_meas.toggled.connect(lambda *_: self._update_cycle_mode_hint())

        self.lbl_cycle_mode_hint = QLabel()
        self.lbl_cycle_mode_hint.setStyleSheet("color:#555; font-size:9pt;")
        self.lbl_cycle_mode_hint.setWordWrap(True)

        g.addWidget(QLabel("Number of cycles"), 0, 0)
        g.addWidget(self.num_cycles, 0, 1)
        g.addWidget(self.initial_meas, 1, 0, 1, 2)
        g.addWidget(self.lbl_cycle_mode_hint, 2, 0, 1, 2)
        root.addWidget(cycle_box)

        # IV/VI measurement sweep settings
        sweep_box = QGroupBox(f"{self.title}: Measurement sweep (IV/VI)")
        g = QGridLayout(sweep_box)

        self.smu = QSpinBox()
        self.smu.setRange(1, 10)
        self.smu.setValue(1)

        self.mode = QComboBox()
        self.mode.addItems(['iv', 'vi'])

        self.start = _spin(0.0, decimals=6, step=0.1)
        self.stop = _spin(2.0 if not self.is_recovery else 1.0, decimals=6, step=0.1)

        self.steps = QSpinBox()
        self.steps.setRange(2, 10001)
        self.steps.setValue(21)

        self.dwell = _spin(0.1, minimum=0, decimals=3, step=0.01, suffix=' s')
        self.sweep_comp = _spin(0.1, minimum=0, decimals=9, step=0.01)
        self.steps.valueChanged.connect(lambda *_: self._update_cycle_mode_hint())
        self.dwell.valueChanged.connect(lambda *_: self._update_cycle_mode_hint())

        rows = [
            ('SMU', self.smu),
            ('Mode', self.mode),
            ('Start', self.start),
            ('Stop', self.stop),
            ('Steps', self.steps),
            ('Dwell', self.dwell),
            ('Compliance', self.sweep_comp),
        ]

        for r, (lab, wid) in enumerate(rows):
            g.addWidget(QLabel(lab), r, 0)
            g.addWidget(wid, r, 1)

        root.addWidget(sweep_box)

        # Bias settings. For recovery this is recovery bias; engine internally calls it stress.
        bias_box = QGroupBox(
            f"{self.title}: {'Recovery' if self.is_recovery else 'Stress'} bias settings"
        )
        bias_box.setMinimumHeight(380)  # increased for time design fields

        g = QGridLayout(bias_box)

        self.bias_mode = QComboBox()
        self.bias_mode.addItems(['voltage', 'current'])

        self.bias_value = _spin(0.0 if self.is_recovery else 2.0, decimals=9, step=0.1)
        
        # Linear duration field
        self.bias_duration = _spin(60.0, minimum=0, decimals=3, step=1.0, suffix=' s')
        self.bias_duration.valueChanged.connect(lambda *_: self._update_cycle_mode_hint())
        
        # Log t0 duration field (for log mode)
        self.bias_duration_log_t0 = _spin(60.0, minimum=0, decimals=3, step=1.0, suffix=' s')
        self.bias_duration_log_t0.valueChanged.connect(lambda *_: self._update_cycle_mode_hint())
        
        self.bias_interval = _spin(1.0, minimum=0.001, decimals=3, step=0.1, suffix=' s')
        self.bias_comp = _spin(0.1, minimum=0, decimals=9, step=0.01)
        
        # Time design controls
        self.time_design = QComboBox()
        self.time_design.addItem("Linear (equal cycle dt)", TIME_DESIGN_LINEAR)
        self.time_design.addItem("Log", TIME_DESIGN_LOG)
        self.time_design.currentIndexChanged.connect(self._on_time_design_changed)
        
        self.log_ppd = QComboBox()
        for option in LOG_POINTS_PER_DECADE_OPTIONS:
            self.log_ppd.addItem(str(option), option)
        self.log_ppd.setCurrentIndex(LOG_POINTS_PER_DECADE_OPTIONS.index(5))
        self.log_ppd.setToolTip(
            "Log density D: how many cycles are placed in each decade. "
            "This controls spacing only, not total cycle count N."
        )
        self.log_ppd.currentIndexChanged.connect(lambda *_: self._update_cycle_mode_hint())

        g.addWidget(QLabel("Bias mode"), 0, 0)
        g.addWidget(self.bias_mode, 0, 1)
        
        g.addWidget(QLabel("Bias value (V or A)"), 1, 0)
        g.addWidget(self.bias_value, 1, 1)
        
        g.addWidget(QLabel("Linear cycle duration (s)"), 2, 0)
        g.addWidget(self.bias_duration, 2, 1)
        
        self.lbl_log_t0 = QLabel("Log start duration t0 (s):")
        g.addWidget(self.lbl_log_t0, 2, 2)
        g.addWidget(self.bias_duration_log_t0, 2, 3)
        
        g.addWidget(QLabel("B1500 monitor interval"), 3, 0)
        g.addWidget(self.bias_interval, 3, 1)
        
        g.addWidget(QLabel("Time Design:"), 3, 2)
        g.addWidget(self.time_design, 3, 3)
        
        g.addWidget(QLabel("Bias compliance"), 4, 0)
        g.addWidget(self.bias_comp, 4, 1)
        
        self.lbl_log_ppd = QLabel("Log cycles/decade (density D):")
        g.addWidget(self.lbl_log_ppd, 4, 2)
        g.addWidget(self.log_ppd, 4, 3)
        
        # Initially hide log-specific controls
        self._on_time_design_changed()
        self._update_cycle_mode_hint()

        root.addWidget(bias_box)

        # Spectrometer settings
        spec_box = QGroupBox(f"{self.title}: Spectrometer settings")
        g = QGridLayout(spec_box)

        self.spec_enabled = QCheckBox("Enable spectrometer")
        self.spec_enabled.setChecked(True)

        self.meas_int = _spin(100.0, minimum=0.001, decimals=3, step=10.0, suffix=' ms')
        self.bias_int = _spin(100.0, minimum=0.001, decimals=3, step=10.0, suffix=' ms')

        self.avg = QSpinBox()
        self.avg.setRange(1, 10000)
        self.avg.setValue(1)

        self.spec_interval = _spin(1000.0, minimum=1, decimals=3, step=100.0, suffix=' ms')

        g.addWidget(self.spec_enabled, 0, 0, 1, 2)

        g.addWidget(QLabel("Integration during MEASUREMENT"), 1, 0)
        g.addWidget(self.meas_int, 1, 1)

        g.addWidget(
            QLabel(f"Integration during {'RECOVERY' if self.is_recovery else 'STRESS'}"),
            2,
            0
        )
        g.addWidget(self.bias_int, 2, 1)

        g.addWidget(QLabel("Averages"), 3, 0)
        g.addWidget(self.avg, 3, 1)

        g.addWidget(
            QLabel(f"Spectrum interval during {'recovery' if self.is_recovery else 'stress'}"),
            4,
            0
        )
        g.addWidget(self.spec_interval, 4, 1)

        root.addWidget(spec_box)

        root.addStretch(1)
        self.setMinimumWidth(430)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def build_config(self, output_folder: str, device_name: str) -> CycleConfig:
        sweep = SweepConfig(
            smu=self.smu.value(),
            mode=self.mode.currentText(),
            start=self.start.value(),
            stop=self.stop.value(),
            steps=self.steps.value(),
            dwell_s=self.dwell.value(),
            compliance=self.sweep_comp.value()
        )

        stress = StressConfig(
            mode=self.bias_mode.currentText(),
            value=self.bias_value.value(),
            duration_s=(
                self.bias_duration.value()
                if self.active_time_design() == TIME_DESIGN_LINEAR
                else self.bias_duration_log_t0.value()
            ),
            time_design=str(self.time_design.currentData() or TIME_DESIGN_LINEAR),
            log_points_per_decade=int(self.log_ppd.currentData() or 5),
            sample_interval_s=self.bias_interval.value(),
            compliance=self.bias_comp.value()
        )

        spec = SpecConfig(
            enabled=self.spec_enabled.isChecked(),
            meas_integration_ms=self.meas_int.value(),
            stress_integration_ms=self.bias_int.value(),
            num_averages=self.avg.value(),
            stress_interval_ms=self.spec_interval.value()
        )

        return CycleConfig(
            sweep=sweep,
            stress=stress,
            spec=spec,
            num_cycles=self.num_cycles.value(),
            initial_measurement=self.initial_meas.isChecked(),
            output_folder=output_folder,
            device_name=device_name,
            autosave=True
        )

    def _on_time_design_changed(self, *_):
        new_design = str(self.time_design.currentData() or TIME_DESIGN_LINEAR)

        prev_design = self._active_time_design
        self._cycle_counts_by_design[prev_design] = int(self.num_cycles.value())
        self._active_time_design = new_design
        new_count = int(self._cycle_counts_by_design.get(new_design, self.num_cycles.value()))
        if self.num_cycles.value() != new_count:
            self.num_cycles.blockSignals(True)
            self.num_cycles.setValue(new_count)
            self.num_cycles.blockSignals(False)

        is_log = new_design == TIME_DESIGN_LOG
        self.lbl_log_t0.setVisible(is_log)
        self.bias_duration_log_t0.setVisible(is_log)
        self.lbl_log_ppd.setVisible(is_log)
        self.log_ppd.setVisible(is_log)
        self._update_cycle_mode_hint()

    def _on_cycle_no_changed(self, value: int):
        self._cycle_counts_by_design[self._active_time_design] = int(value)
        self._update_cycle_mode_hint()

    @staticmethod
    def _build_preview_timing(total_cycles: int, base_duration_s: float,
                              time_design: str, log_points_per_decade: int):
        base = max(1e-9, float(base_duration_s))
        design = (time_design or TIME_DESIGN_LINEAR).strip().lower()
        if design == TIME_DESIGN_LOG:
            ppd = int(log_points_per_decade)
            if ppd not in LOG_POINTS_PER_DECADE_OPTIONS:
                ppd = min(LOG_POINTS_PER_DECADE_OPTIONS, key=lambda v: abs(v - ppd))
            mantissas = LOG_CYCLE_MANTISSAS[ppd]
            return [base * mantissas[idx % ppd] * (10 ** (idx // ppd)) for idx in range(total_cycles)]
        return [base for _ in range(total_cycles)]

    def _update_cycle_mode_hint(self):
        active_label = "LOG" if self._active_time_design == TIME_DESIGN_LOG else "LINEAR"
        linear_n = int(self._cycle_counts_by_design.get(TIME_DESIGN_LINEAR, self.num_cycles.value()))
        log_n = int(self._cycle_counts_by_design.get(TIME_DESIGN_LOG, self.num_cycles.value()))
        log_density = int(self.log_ppd.currentData() or LOG_POINTS_PER_DECADE_OPTIONS[0])
        active_n = log_n if self._active_time_design == TIME_DESIGN_LOG else linear_n
        linear_dt = float(self.bias_duration.value())
        log_t0 = float(self.bias_duration_log_t0.value())
        linear_bias_total = linear_dt * linear_n
        log_bias_total = sum(self._build_preview_timing(
            log_n, log_t0, TIME_DESIGN_LOG, log_density
        ))
        steps = int(self.steps.value())
        dwell = float(self.dwell.value())
        init_extra = 1 if self.initial_meas.isChecked() else 0
        linear_meas_total = (linear_n + init_extra) * steps * dwell
        log_meas_total = (log_n + init_extra) * steps * dwell
        self.lbl_cycle_mode_hint.setText(
            f"Meaning: D={log_density} cycles/decade is log spacing density only; "
            f"N={active_n} is total Stress->Measurement loop count.\n"
            f"Active mode: {active_label} | Stored N -> Linear={linear_n}, Log={log_n}\n"
            f"Estimated total duration (bias only): Linear={linear_bias_total:g}s, Log={log_bias_total:g}s\n"
            f"Estimated total duration (+measurement approx): "
            f"Linear={linear_bias_total + linear_meas_total:g}s, "
            f"Log={log_bias_total + log_meas_total:g}s"
        )

    def cycle_count_for_design(self, design: str) -> int:
        return int(self._cycle_counts_by_design.get(design, self.num_cycles.value()))

    def set_cycle_count_for_design(self, design: str, value: int):
        count = int(max(1, value))
        self._cycle_counts_by_design[design] = count
        if self._active_time_design == design:
            self.num_cycles.blockSignals(True)
            self.num_cycles.setValue(count)
            self.num_cycles.blockSignals(False)
        self._update_cycle_mode_hint()

    def active_time_design(self) -> str:
        return self._active_time_design


# -----------------------------------------------------------------------------
# Main GUI
# -----------------------------------------------------------------------------

class SpectrometerStressRecoveryGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('B1500 + Spectrometer Stress/Recovery Cycles')
        self.resize(1500, 900)

        self.b1500 = B1500Controller()
        self.spec = SpectrometerController()
        self.worker = None
        self._syncing_phase_cycle_controls = False

        self._build_ui()

    def _scroll(self, widget):
        area = QScrollArea()
        area.setWidget(widget)
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return area

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        main.addWidget(splitter)

        # LEFT WINDOW: adjustable width + both scroll bars
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(6, 6, 6, 6)

        conn = QGroupBox('Connections')
        g = QGridLayout(conn)

        self.b1500_addr = QLineEdit('GPIB0::17::INSTR')

        self.btn_b1500 = QPushButton('Connect B1500')
        self.btn_b1500.clicked.connect(self._connect_b1500)

        self.btn_spec = QPushButton('Connect Spectrometer')
        self.btn_spec.clicked.connect(self._connect_spec)

        g.addWidget(QLabel('B1500 VISA address'), 0, 0)
        g.addWidget(self.b1500_addr, 0, 1)
        g.addWidget(self.btn_b1500, 1, 0)
        g.addWidget(self.btn_spec, 1, 1)

        left_layout.addWidget(conn)

        out = QGroupBox('Output')
        g = QGridLayout(out)

        self.device_name = QLineEdit('Device_001')
        self.save_folder = QLineEdit(str(Path.cwd() / 'results'))

        browse = QPushButton('Browse...')
        browse.clicked.connect(self._browse)

        g.addWidget(QLabel('Device name'), 0, 0)
        g.addWidget(self.device_name, 0, 1, 1, 2)
        g.addWidget(QLabel('Save folder'), 1, 0)
        g.addWidget(self.save_folder, 1, 1)
        g.addWidget(browse, 1, 2)

        left_layout.addWidget(out)

        self.phase_tabs = QTabWidget()

        self.stress_widget = PhaseConfigWidget('Part 1', is_recovery=False)
        self.recovery_widget = PhaseConfigWidget('Part 2', is_recovery=True)
        self._wire_phase_cycle_sync()

        self.phase_tabs.addTab(
            self._scroll(self.stress_widget),
            'Measurement-Stress-Measurement'
        )
        self.phase_tabs.addTab(
            self._scroll(self.recovery_widget),
            'Measurement-Recovery-Measurement'
        )

        left_layout.addWidget(self.phase_tabs, 1)

        controls = QHBoxLayout()

        self.start_btn = QPushButton('START: Stress then Recovery')
        self.stop_btn = QPushButton('STOP')
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)

        left_layout.addLayout(controls)

        left_scroll = self._scroll(left_container)
        left_scroll.setMinimumWidth(480)

        splitter.addWidget(left_scroll)

        # RIGHT WINDOW: status/logs; splitter lets user resize left width.
        right = QWidget()
        rv = QVBoxLayout(right)

        status = QGroupBox('Status')
        g = QGridLayout(status)

        self.phase_label = QLabel('Idle')

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        g.addWidget(QLabel('Current phase'), 0, 0)
        g.addWidget(self.phase_label, 0, 1)
        g.addWidget(QLabel('Progress'), 1, 0)
        g.addWidget(self.progress, 1, 1)

        rv.addWidget(status)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.NoWrap)

        rv.addWidget(self.log, 1)

        splitter.addWidget(right)
        splitter.setSizes([620, 880])

    def _wire_phase_cycle_sync(self):
        self.stress_widget.time_design.currentIndexChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.stress_widget, self.recovery_widget)
        )
        self.recovery_widget.time_design.currentIndexChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.recovery_widget, self.stress_widget)
        )
        self.stress_widget.log_ppd.currentIndexChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.stress_widget, self.recovery_widget)
        )
        self.recovery_widget.log_ppd.currentIndexChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.recovery_widget, self.stress_widget)
        )
        self.stress_widget.num_cycles.valueChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.stress_widget, self.recovery_widget)
        )
        self.recovery_widget.num_cycles.valueChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.recovery_widget, self.stress_widget)
        )
        self.stress_widget.bias_duration.valueChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.stress_widget, self.recovery_widget)
        )
        self.recovery_widget.bias_duration.valueChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.recovery_widget, self.stress_widget)
        )
        self.stress_widget.bias_duration_log_t0.valueChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.stress_widget, self.recovery_widget)
        )
        self.recovery_widget.bias_duration_log_t0.valueChanged.connect(
            lambda *_: self._sync_phase_cycle_controls(self.recovery_widget, self.stress_widget)
        )

    def _sync_phase_cycle_controls(self, source: PhaseConfigWidget,
                                   target: PhaseConfigWidget):
        if self._syncing_phase_cycle_controls:
            return

        self._syncing_phase_cycle_controls = True
        try:
            target.time_design.setCurrentIndex(source.time_design.currentIndex())
            target.log_ppd.setCurrentIndex(source.log_ppd.currentIndex())
            target.bias_duration.setValue(source.bias_duration.value())
            target.bias_duration_log_t0.setValue(source.bias_duration_log_t0.value())
            target.set_cycle_count_for_design(
                TIME_DESIGN_LINEAR,
                source.cycle_count_for_design(TIME_DESIGN_LINEAR)
            )
            target.set_cycle_count_for_design(
                TIME_DESIGN_LOG,
                source.cycle_count_for_design(TIME_DESIGN_LOG)
            )
        finally:
            self._syncing_phase_cycle_controls = False

    def _log(self, text):
        self.log.append(str(text))

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self,
            'Select save folder',
            self.save_folder.text()
        )
        if d:
            self.save_folder.setText(d)

    def _connect_b1500(self):
        try:
            if self.b1500.connected:
                self.b1500.disconnect()
                self.btn_b1500.setText('Connect B1500')
                self._log('B1500 disconnected')
                return

            self.b1500.connect(self.b1500_addr.text().strip())
            self.btn_b1500.setText('Disconnect B1500')
            self._log(f'B1500 connected: {self.b1500.idn}')

        except Exception as e:
            QMessageBox.critical(self, 'B1500 connection error', str(e))

    def _connect_spec(self):
        try:
            if self.spec.connected:
                self.spec.disconnect()
                self.btn_spec.setText('Connect Spectrometer')
                self._log('Spectrometer disconnected')
                return

            self.spec.initialize()
            self.spec.connect_device()

            self.btn_spec.setText('Disconnect Spectrometer')
            self._log('Spectrometer connected')

        except Exception as e:
            QMessageBox.critical(self, 'Spectrometer connection error', str(e))

    def _start(self):
        if self.worker and self.worker.isRunning():
            return

        device = _safe_name(self.device_name.text())
        base = Path(self.save_folder.text()).expanduser()

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        session = base / f'{device}_spectrometer_stress_recovery_scycles_{ts}'

        stress_cfg = self.stress_widget.build_config(
            str(session / 'stress'),
            device
        )
        recovery_cfg = self.recovery_widget.build_config(
            str(session / 'recovery'),
            device
        )

        self.worker = TwoPhaseSpectroscopyWorker(
            self.b1500,
            self.spec,
            stress_cfg,
            recovery_cfg,
            session
        )

        self.worker.log_message.connect(self._log)
        self.worker.phase_change.connect(self._on_phase)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_signal.connect(self._finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)

        self._log(f'Session root: {session}')

        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self._log('Stop requested...')

    def _on_phase(self, label, phase):
        name = phase.value if hasattr(phase, 'value') else str(phase)

        if label == 'recovery' and name == 'stress':
            name = 'recovery'

        self.phase_label.setText(f'{label}: {name}')

    def _on_progress(self, label, cycle, total):
        if total > 0:
            self.progress.setValue(int(100 * cycle / total))

        self._log(f'{label}: cycle {cycle}/{total} complete')

    def _finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.phase_label.setText('Finished')
        self._log('All requested parts finished.')

    def closeEvent(self, event):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(2000)

            if self.b1500.connected:
                self.b1500.disconnect()

            if self.spec.connected:
                self.spec.disconnect()

        finally:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    win = SpectrometerStressRecoveryGUI()
    win.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()