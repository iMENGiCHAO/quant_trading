#!/usr/bin/env python3
"""Hyperion Quant - Setup"""
from setuptools import setup, find_packages

setup(
    name="hyperion-quant",
    version="1.0.0",
    description="Production-Grade A-Share Quantitative Trading Framework",
    author="Hyperion Quant Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "akshare>=1.14.0",
        "optuna>=3.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "hyperion=hyperion.cli:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
