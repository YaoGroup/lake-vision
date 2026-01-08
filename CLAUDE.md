# CLAUDE.md

This file provides guidance for Claude Code when working with this repository.

## Project Overview

**Lake-Vision** is a deep learning framework for classifying supraglacial lake drainage events on glaciers using multi-temporal Sentinel-2 satellite imagery and water area time series data. The project focuses on Greenland but is applicable globally.

**Classification Task**: 4-class drainage type classification
- **ND** (No Drainage): Lake persists
- **ED** (Englacial Drainage): Water drains through moulins into glacier interior
- **LD** (Lateral Drainage): Water flows horizontally to adjacent lakes
- **CD** (Crevasse Drainage): Water escapes via crevasses

## Build & Test Commands

```bash
# Install in development mode
pip install -e .

# Run all tests
pytest lakevision/tests/

# Run specific test module
pytest lakevision/tests/test_classifier.py -v

# Run with coverage
pytest lakevision/tests/ --cov=lakevision

# Preprocess data
python scripts/preprocess_tstacks.py \
    --tstack_dir /path/to/tstacks \
    --area_file /path/to/all_lakes_2019.nc \
    --output_dir /path/to/processed
```

## Project Structure

```
lakevision/
├── models/           # Neural network components
│   ├── classifier.py # LakeDrainageClassifier (main model)
│   ├── clstm.py      # Convolutional LSTM
│   ├── attention.py  # CBAM attention mechanisms
│   └── blocks.py     # Reusable components (FrontCNN, ScalarLSTM, ClassHeadMLP)
├── data/             # Data handling
│   ├── preprocessing.py  # Loading, filtering, combining NetCDF data
│   ├── datasets.py       # PyTorch Dataset class
│   └── utils.py
└── tests/            # pytest test suite
scripts/              # Data preprocessing CLI tools
demos/                # Jupyter notebook examples
datasets/             # Sample data (not in git)
```

## Key Architecture

The main model (`LakeDrainageClassifier`) is a multi-stream temporal architecture:

- **Image stream**: FrontCNN → ConvLSTM → GlobalPooling → features
- **Area stream**: ScalarLSTM → features
- **Cloud stream** (optional): ScalarLSTM → features
- Features are concatenated and passed to ClassHeadMLP for classification

Input tensor shapes: `[B, T, C, H, W]` for imagery, `[B, T, 1]` for scalars.

## Code Conventions

- **Classes**: PascalCase (e.g., `LakeDrainageClassifier`, `FrontCNN`)
- **Functions**: snake_case (e.g., `load_area_sequences`, `combine_lake_data`)
- **Docstrings**: Comprehensive with Args, Input/Output shapes, Examples
- **Type hints**: Used throughout for function signatures
- **Tensor shapes**: Documented as `[B, T, C, H, W]` notation

## Data Format

Data uses NetCDF (xarray) format:
- Imagery: `[time, channel, y, x]` with channels `['red', 'green', 'blue', 'mask']`
- Water area: `[time]` float values
- Imagery normalized by dividing by 10000.0 (Sentinel-2 reflectance scaling)

## Important Notes

- CLSTM requires odd kernel sizes
- At least one of `use_imgseq` or `use_areaseq` must be True
- `use_cloudyseq` requires either imgseq or areaseq to also be enabled
- NaN values in time series are filled via forward/backward fill or interpolation
