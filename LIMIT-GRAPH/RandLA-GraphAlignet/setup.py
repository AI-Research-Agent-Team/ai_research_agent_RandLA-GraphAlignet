#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for RandLA-GraphAlignNet
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="randla-graphalignnet",
    version="1.0.0",
    author="RandLA-GraphAlignNet Team",
    author_email="nurcholisadam@gmail.com",
    description="Multilingual Spatial Reasoning for 3D Point Clouds with Semantic Graph Alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AI-Research-Agent-Team/ai_research_agent_RandLA-GraphAlignet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: CC BY-NC-SA 4.0 License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["faiss-gpu>=1.7.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0", "black>=22.0.0", "flake8>=4.0.0"],
        "viz": ["dash>=2.6.0", "plotly>=5.10.0", "neo4j>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "randla-graphalign=main:main",
            "randla-demo=demo_complete_integration:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)