# CLAUDE.md

This file provides guidance for Claude Code when working with this repository.

## Project Overview

**Lake-Vision** is a deep learning framework for classifying supraglacial lake drainage events on glaciers using multi-temporal Sentinel-2 satellite imagery and water area time series data. The project focuses on Greenland but is applicable globally.

**Classification Task**: 4-class drainage type classification
- **ND** (No Drainage): Lake persists (label=0)
- **ED** (Englacial Drainage): Water drains through moulins into glacier interior (label=1)
- **LD** (Lateral Drainage): Water flows horizontally to adjacent lakes (label=2)
- **CD** (Crevasse Drainage): Water escapes via crevasses (label=3)

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
python engine/preprocessing/preprocess_tstacks.py \
    --tstack_dir /path/to/tstacks \
    --area_file /path/to/all_lakes_2019.nc \
    --output_dir /path/to/processed
```

## Project Structure

```
lake-vision/
├── lakevision/           # Main Python package
│   ├── models/           # Neural network components
│   │   ├── classifier.py # LakeDrainageClassifier (main model)
│   │   ├── clstm.py      # Convolutional LSTM
│   │   ├── attention.py  # CBAM attention mechanisms
│   │   └── blocks.py     # Reusable components (FrontCNN, ScalarLSTM, ClassHeadMLP)
│   ├── data/             # Data handling
│   │   ├── preprocessing.py  # Loading, filtering, combining NetCDF data
│   │   ├── datasets.py       # PyTorch Dataset class (LakeDataset)
│   │   └── utils.py
│   └── tests/            # pytest test suite
├── engine/               # Runnable scripts (entry points)
│   ├── training/
│   │   ├── run_training.py   # Main training script with wandb
│   │   ├── run_training.sh   # SLURM script for Sherlock
│   │   └── sweep.yaml        # Wandb sweep configuration
│   ├── preprocessing/
│   │   ├── preprocess_tstacks.py  # Combine imagery + water area into NC files
│   │   └── preprocess_tstacks.sh  # SLURM script for preprocessing
│   ├── inference/        # (placeholder for inference scripts)
│   └── labeling/         # (placeholder for labeling scripts)
├── demos/                # Jupyter notebook examples
└── datasets/             # Sample data (not in git)
```

## Key Architecture

The main model (`LakeDrainageClassifier`) is a multi-stream temporal architecture:

- **Image stream**: FrontCNN → optional CBAM Attention → ConvLSTM → GlobalPooling → features
- **Area stream**: ScalarLSTM → features
- **Cloud stream** (optional): ScalarLSTM → features
- Features are concatenated and passed to ClassHeadMLP for 4-class classification

Input tensor shapes: `[B, T, C, H, W]` for imagery, `[B, T, 1]` for scalars.

### Spectral Band Support

The model supports additional spectral bands beyond RGB:
- `use_nir`: Near-infrared band
- `use_swir16`: SWIR Band 11 (1.6μm)
- `use_swir22`: SWIR Band 12 (2.2μm)

Channel order in NC files: `['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'mask']`

### Attention Types

- `'none'`: No attention
- `'spatial'`: Spatial CBAM attention
- `'full'`: Full CBAM (channel + spatial)
- `'arch'`: Architectural attention with separate mask pathway

## Data Format

### NetCDF Lake Files

Each lake has a combined NC file with:
- `imagery`: `[time, channel, y, x]` - Multi-spectral satellite imagery
- `water_area`: `[time]` - Water area time series (from Dunmire+25)
- `cloudy_seq_*`: `[time]` - Tile usefulness flags (from cloudy-tile inference)
  - `cloudy_seq_rgb`: RGB model predictions
  - `cloudy_seq_rgbn`: RGB+NIR model predictions
  - `cloudy_seq_bns16`: B+NIR+SWIR16 model predictions

### Labels CSV

Labels file format:
- `new_id`: Lake identifier (matches NC filename without .nc extension)
- `label_rines`: Integer label (0=ND, 1=ED, 2=LD, 3=CD)

### LakeDataset

The `LakeDataset` class returns per sample:
```python
img_seq, area_seq, cloudy_seq, label, lake_id = dataset[idx]
# img_seq:    [seq_len, C, H, W] - image sequence
# area_seq:   [seq_len, 1] - water area sequence
# cloudy_seq: [seq_len, 1] - cloudy/useful flags
# label:      int tensor - class label
# lake_id:    str - lake identifier
```

## Training

### Local Training

```bash
python engine/training/run_training.py \
    --labels_csv /path/to/labels.csv \
    --nc_dir /path/to/nc/files \
    --id_col "new_id" \
    --label_col "label_rines" \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --save_path model.pth
```

### Sherlock HPC Training

```bash
cd /oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/engine/training
sbatch run_training.sh
```

After training completes, sync wandb runs:
```bash
cd /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision
wandb sync wandb/offline-run-*
```

## Key File Paths on Sherlock

| Resource | Path |
|----------|------|
| Repository | `/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision` |
| Lake NC files | `/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_processed` |
| Band statistics | `/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats.json` |
| Labels CSV | `/oak/stanford/groups/cyaolai/JoshRines/data/labels/lake_labels.csv` |
| Sherlock workdir | `/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision` |
| Saved models | `/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/models` |
| Logs | `/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs` |

## Code Conventions

- **Classes**: PascalCase (e.g., `LakeDrainageClassifier`, `FrontCNN`)
- **Functions**: snake_case (e.g., `load_area_sequences`, `combine_lake_data`)
- **Docstrings**: Comprehensive with Args, Input/Output shapes, Examples
- **Type hints**: Used throughout for function signatures
- **Tensor shapes**: Documented as `[B, T, C, H, W]` notation

## Important Notes

- CLSTM requires odd kernel sizes
- At least one of `use_imgseq` or `use_areaseq` must be True
- `use_cloudyseq` requires either imgseq or areaseq to also be enabled
- NaN values in time series are filled via forward/backward fill or interpolation
- Per-band normalization using `band_stats.json` is recommended for multi-spectral data
- GPU constraint for Sherlock: use `-C GPU_SKU:A100_SXM4` (H100s have CUDA compatibility issues)

## Related Projects

- [cloudy-tile](../cloudy-tile) - Preprocessing model that generates `cloudy_seq` variables

## Changelog

### January 2026
- Added multi-spectral support (`use_nir`, `use_swir16`, `use_swir22` flags)
- Added `cloudy_seq_var` parameter to `LakeDataset` for loading cloud/usefulness sequences
- Updated `LakeDataset` to return 5-tuple: `(img_seq, area_seq, cloudy_seq, label, lake_id)`
- Created `engine/training/run_training.py` - Main training script with wandb integration
- Created `engine/training/run_training.sh` - SLURM script for Sherlock training
- Added `band_stats` support for per-band normalization in `LakeDataset`
- Consolidated all scripts under `engine/` directory (removed `scripts/`)
