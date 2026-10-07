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
    name="stress_recovery_spectra",
    version="1.0.0",
    author="Veronica GaoZhan",
    author_email="",
    description="Series 6 VCSEL optical spectrum stress-recovery cycling workflow with B1500 + Avantes spectrometer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vvvvvero/VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery",
    packages=find_packages(include=["stress_recovery_spectra", "stress_recovery_spectra.*"]),
    py_modules=["b1500_spectrometer_stress_rcovery_cycle", "b1500_Step_stress_spectroscopy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "stress_recovery_spectra=stress_recovery_spectra.__main__:main",
        ],
    },
    keywords="vcsel b1500 avantes spectroscopy stress recovery reliability",
    project_urls={
        "Bug Reports": "https://github.com/vvvvvero/VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery/issues",
        "Source": "https://github.com/vvvvvero/VCSEL_Reliablity_Tests_6_Optical_Spectrum_Stress_Recovery",
    },
)
