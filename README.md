# VCSEL Reliability Tests - Series 6: Optical Spectrum Stress Recovery Tests

A comprehensive Python package for automated stress/recovery cycling tests on VCSELs (Vertical-Cavity Surface-Emitting Lasers) using Keysight B1500 Semiconductor Parameter Analyzer and Avantes Spectrometer.

## Features

- **Two-phase cycling**: Stress phase followed by Recovery phase
- **Independent phase configuration**: Each phase has separate sweep, bias, spectrum, and cycle parameters
- **Flexible timing modes**: 
  - Linear mode: Equal time intervals between cycles
  - Log mode: Logarithmically distributed cycles with configurable density (cycles/decade)
- **Real-time monitoring**: Live IV curves, power measurements, and spectral data
- **PyQt5 GUI**: Professional graphical interface for parameter configuration and test execution
- **Automatic data organization**: Hierarchical folder structure for stress and recovery phase results
- **Spectrometer integration**: Support for Avantes spectrometer with configurable integration times

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows OS (for B1500 and spectrometer drivers)
- Keysight B1500 GPIB connection
- Avantes spectrometer (optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vvvvvero/VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery.git
cd VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install the package in development mode:
```bash
pip install -e .
```

## Usage

### GUI Application

```bash
python b1500_spectrometer_stress_rcovery_cycle.py
```

### Configuration

The GUI provides comprehensive configuration options:

1. **Connections**
   - B1500 GPIB address (default: GPIB0::17::INSTR)
   - Spectrometer connection

2. **Output Settings**
   - Device name and save folder for results
   
3. **Stress Phase (Part 1)**
   - IV sweep parameters (start, stop, steps, dwell, compliance)
   - Stress bias settings (mode, value, duration)
   - Time design (Linear or Log with cycles/decade)
   - Spectrometer integration times
   - Number of stress cycles

4. **Recovery Phase (Part 2)**
   - Similar configuration to stress phase
   - Lower default bias values
   - Separate cycle count and timing

### Data Output

Results are organized in the following structure:
```
<device>_spectrometer_stress_recovery_scycles_<timestamp>/
├── stress/
│   ├── iv_cycle_000.csv
│   ├── iv_cycle_001.csv
│   ├── stress_cycle_001.csv
│   ├── measurement_spectra_cycle_000.csv
│   └── ...
└── recovery/
    ├── iv_cycle_000.csv
    ├── iv_cycle_001.csv
    ├── stress_cycle_001.csv
    ├── measurement_spectra_cycle_000.csv
    └── ...
```

## Architecture

### Core Components

- **B1500Controller**: Low-level interface to Keysight B1500 via PyVISA
- **SpectrometerController**: Interface to Avantes spectrometer SDK
- **StressCycleEngine**: Core measurement engine handling measurement-stress-measurement sequences
- **TwoPhaseSpectroscopyWorker**: Qt worker thread managing two-phase test execution
- **PhaseConfigWidget**: Reusable UI panel for per-phase configuration

### Key Features

#### Time Design System
- **Linear mode**: Equal duration between cycles (Δt = constant)
- **Log mode**: Logarithmically distributed cycles with configurable density
  - Maintains separate cycle counts for each mode
  - Automatic time estimation for user preview

#### Cross-Phase Synchronization
- Time design parameters automatically sync between stress and recovery tabs
- Cycle count, duration, and density controls synchronized
- Prevents parameter misalignment during multi-phase testing

## API Reference

### Launching GUI

```python
from b1500_spectrometer_stress_rcovery_cycle import main
main()
```

### Using Configuration Classes

```python
from b1500_Step_stress_spectroscopy import SweepConfig, StressConfig, SpecConfig, CycleConfig

sweep = SweepConfig(
    smu=1,
    mode="iv",
    start=0.0,
    stop=2.0,
    steps=21,
    dwell_s=0.1,
    compliance=0.1
)

stress = StressConfig(
    mode="voltage",
    value=2.0,
    duration_s=60.0,
    time_design="linear",  # or "log"
    log_points_per_decade=5,
    sample_interval_s=1.0,
    compliance=0.1
)

spec = SpecConfig(
    enabled=True,
    meas_integration_ms=100.0,
    stress_integration_ms=100.0,
    num_averages=1,
    stress_interval_ms=1000.0
)

config = CycleConfig(
    sweep=sweep,
    stress=stress,
    spec=spec,
    num_cycles=10,
    initial_measurement=True,
    output_folder="results",
    device_name="Device_001"
)
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Author

**Veronica GaoZhan**  


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Keysight Technologies for B1500 instrumentation
- Avantes for spectrometer SDK and support
- PyQt5 community for the excellent GUI framework

## References

- Keysight B1500 Semiconductor Parameter Analyzer Programming Guide
- Avantes Spectrometer SDK Documentation
- PyQt5 Official Documentation

## Related Projects

- [VCSEL_Reliablity_Tests_4_Optical_Spectrum_Step_Stress](https://github.com/vvvvvero/VCSEL_Reliablity_Tests_4_Optical_Spectrum_Step_Stress)
- [VCSEL IV Analysis Tools](https://github.com/vvvvvero)

---

*Last Updated: October 2026*
