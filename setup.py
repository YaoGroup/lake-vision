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
        "netCDF4>=1.6.0",
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
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)