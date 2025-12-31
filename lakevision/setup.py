# setup.py
from setuptools import setup, find_packages

setup(
    name="lakevision",
    version="0.1.0",
    description="Supraglacial lake drainage classification using deep learning",
    author="Joshua H. Rines",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    python_requires=">=3.8",
)