#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stress_recovery_spectra
=======================
Series 6 optical spectrum stress-recovery workflow package.

(c) Veronica GaoZhan, 2026
"""

__version__ = "1.0.0"
__author__ = "Veronica GaoZhan"

__all__ = [
    "PhaseConfigWidget",
    "TwoPhaseSpectroscopyWorker",
    "SpectrometerStressRecoveryGUI",
    "main",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from b1500_spectrometer_stress_rcovery_cycle import __dict__ as runtime_symbols

    if name not in runtime_symbols:
        raise AttributeError(f"runtime module has no attribute {name!r}")

    value = runtime_symbols[name]
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
