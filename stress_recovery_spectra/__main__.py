#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI entry point for Series 6 stress-recovery spectroscopy package.

Usage:
  python -m stress_recovery_spectra
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stress_recovery_spectra",
        description="Series 6 Optical Spectrum Stress-Recovery workflow (GUI)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )
    parser.add_argument(
        "--list-resources",
        action="store_true",
        help="List VISA GPIB resources and exit",
    )
    return parser


def _list_gpib_resources() -> int:
    try:
        import pyvisa
    except Exception as exc:
        print(f"PyVISA unavailable: {exc}")
        return 1

    rm = None
    try:
        try:
            rm = pyvisa.ResourceManager()
        except Exception:
            rm = pyvisa.ResourceManager("@py")

        resources = sorted(r for r in rm.list_resources() if "GPIB" in r.upper())
        if not resources:
            print("No GPIB resources found")
        else:
            print("Available GPIB resources:")
            for res in resources:
                print(f"  {res}")
        return 0
    except Exception as exc:
        print(f"Failed to list VISA resources: {exc}")
        return 1
    finally:
        if rm is not None:
            try:
                rm.close()
            except Exception:
                pass


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_resources:
        return _list_gpib_resources()

    from b1500_spectrometer_stress_rcovery_cycle import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
