# lake-vision/setup.py
from setuptools import setup, find_packages

setup(
    name="lakevision",
    version="0.1.0",
    description="Supraglacial lake drainage classification using deep learning",
    author="Josh Rines",
    author_email="jrines@stanford.edu",
    url="https://github.com/YaoGroup/lake-vision",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "xarray>=2023.1.0",
        "h5py>=3.8.0",
        "pandas>=2.0.0",
        "rasterio>=1.3.0",
        "pillow>=9.5.0",
        "scikit-learn>=1.3.0",
        # Upper bounds exist because Sherlock's compute nodes are glibc 2.17 and
        # can only install manylinux2014 / manylinux_2_17 wheels. netCDF4 >=1.7.3
        # ships no cp312 x86_64 linux wheel and blosc2 >=4.8 ships manylinux_2_28
        # only; past those bounds pip source-builds, pulling numpy>=2.1 (which
        # torch 2.2.1 cannot run on) and needing GCC >=10.3 (stack has 10.1.0).
        # blosc2's lower bound is the CParams API that build_cache.py uses, and
        # it is why python_requires is >=3.10.
        "netCDF4>=1.6.0,<1.7.3",
        "blosc2>=3.3,<4.8",
    ],
    extras_require={
        # Composite synthesis only (engine/preprocessing/synthesize_region.py,
        # lakevision/data/synthesis.py). Both import these lazily inside
        # functions, so the core package works without them.
        "synthesis": [
            "geopandas>=0.14.0",
            "affine>=2.4.0",
            "shapely>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-timeout>=2.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ]
    },
    # 3.10 is the floor for blosc2 >=3.3 (the CParams API). Sherlock runs 3.12.1;
    # 3.9 reached end-of-life in October 2025.
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)