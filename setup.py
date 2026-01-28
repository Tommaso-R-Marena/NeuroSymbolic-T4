"""Setup script for NeuroSymbolic-T4."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="neurosymbolic-t4",
    version="0.1.0",
    author="Tommaso R. Marena",
    author_email="marena@cua.edu",
    description="State-of-the-art neurosymbolic AI system optimized for Google T4 GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tommaso-R-Marena/NeuroSymbolic-T4",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "tensorboard>=2.13.0",
        ],
    },
)