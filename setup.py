#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup configuration for VCSEL Optical Spectrum Stress/Recovery Testing Package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vcsel-spectrum-stress-recovery",
    version="1.0.0",
    author="Veronica GaoZhan",
    author_email="veronica.gaozhan@centrum.cz",
    description="B1500 + Avantes Spectrometer two-phase stress/recovery cycling test with PyQt5 GUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vvvvvero/VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vcsel-stress-recovery=vcsel_spectrum_stress_recovery.main:main",
        ],
    },
    keywords="VCSEL testing stress recovery optical spectrum",
    project_urls={
        "Bug Reports": "https://github.com/vvvvvero/VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery/issues",
        "Source": "https://github.com/vvvvvero/VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery",
    },
)
