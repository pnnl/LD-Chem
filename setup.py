#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup configuration for MultiPart package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multipart",
    version="0.1.0",
    author="Laura Fierce",
    author_email="laura.fierce@pnnl.gov",
    description="Multiscale Particle-based Microphysics Model for aerosol-cloud interactions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PNNL-MulitPart/multipart",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "xarray",
        "netCDF4",
        "numba",
        "assimulo",
        "pyyaml",
        "mat73",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [],
    },
)
