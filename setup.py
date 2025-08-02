#!/usr/bin/env python3
"""
Setup script for Mental Health Detector
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="mental-health-detector",
    version="1.0.0",
    author="Mental Health Detector Team",
    author_email="contact@mentalhealthdetector.com",
    description="AI-powered mental health analysis tool with ethical safeguards",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/midlaj-muhammed/Mental-Health-Detector-AI-Application-Built",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mental-health-detector=src.cli.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.yaml", "*.yml"],
    },
    keywords="mental health, AI, NLP, sentiment analysis, emotion detection, healthcare",
    project_urls={
        "Bug Reports": "https://github.com/midlaj-muhammed/Mental-Health-Detector-AI-Application-Built/issues",
        "Source": "https://github.com/midlaj-muhammed/Mental-Health-Detector-AI-Application-Built",
        "Documentation": "https://github.com/midlaj-muhammed/Mental-Health-Detector-AI-Application-Built#readme",
    },
)
