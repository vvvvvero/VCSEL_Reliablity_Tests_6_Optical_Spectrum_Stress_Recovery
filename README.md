# VCSEL Reliability Tests - Series 6: Optical Spectrum Stress Recovery

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Synchronized Keysight B1500 + Avantes spectrometer stress-recovery cycling with live monitoring and automated data export.**

Â© Veronica GaoZhan - 2026

---

## Series Context

This repository is part of the Veronica GaoZhan VCSEL Reliability Test Series.

- Series ID: VGZ-VRLS
- Track: Single-device reliability progression
- Position: 6
- Protocol name: stress_recovery_spectroscopy
- Author: Veronica GaoZhan

---

## Overview

Series 6 implements a two-phase stress-recovery spectroscopy workflow.
The run executes stress cycles first, then recovery cycles, with independent
configuration per phase.

Main flow:

```text
Part 1: Measurement -> Stress -> Measurement   (N stress cycles)
Part 2: Measurement -> Recovery -> Measurement (N recovery cycles)
```

---

## Features

| Feature | Detail |
|---------|--------|
| **Two-Phase Cycling** | Stress phase followed by recovery phase in one session |
| **Independent Phase Settings** | Separate IV sweep, bias, duration, interval, and compliance for each phase |
| **Linear/Log Timing Design** | Linear equal-dt mode or logarithmic timing with cycles-per-decade density |
| **Synchronized Timing Controls** | Shared time-design alignment between stress and recovery tabs |
| **IV + Spectra Acquisition** | Electrical sweep data plus optical spectra for measurement and bias intervals |
| **Live GUI** | PyQt5 interface with status logging and progress updates |
| **CSV Export** | Per-cycle electrical and spectra files under stress/recovery subfolders |

---

## Installation

### From Source

```bash
git clone https://github.com/vvvvvero/VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery.git
cd VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery
pip install -e .
```

### Dependencies

- numpy
- pyvisa
- pyvisa-py
- PyQt5
- matplotlib

Or install directly:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Package Entry

```bash
python -m stress_recovery_spectra
```

### Console Script Entry

```bash
stress_recovery_spectra
```

### Legacy Script Entry

```bash
python b1500_spectrometer_stress_rcovery_cycle.py
```

### List Available GPIB Resources

```bash
python -m stress_recovery_spectra --list-resources
```

---

## Output Layout

Each run writes one timestamped session folder:

```text
<save_folder>/<device>_spectrometer_stress_recovery_scycles_<timestamp>/
    stress/
        iv_cycle_000.csv
        stress_cycle_001.csv
        measurement_spectra_cycle_000.csv
        ...
    recovery/
        iv_cycle_000.csv
        stress_cycle_001.csv
        measurement_spectra_cycle_000.csv
        ...
```

---

## Repository Structure

```text
.
â”śâ”€â”€ stress_recovery_spectra/
â”‚   â”śâ”€â”€ __init__.py
â”‚   â””â”€â”€ __main__.py
â”śâ”€â”€ b1500_spectrometer_stress_rcovery_cycle.py
â”śâ”€â”€ b1500_Step_stress_spectroscopy.py
â”śâ”€â”€ pyproject.toml
â”śâ”€â”€ setup.py
â”śâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Related Series

- Series 4 step-stress spectroscopy: https://github.com/vvvvvero/VCSEL_Reliablity_Tests_4_Optical_Spectrum_Step_Stress

