# lake-vision
Framework for tracking supraglacial lake evolution.  Developed for Greenland.  Applicable globally.

Note: This repo is under active development

## Introduction
Introduction...

## Installation
Installation guide...

## Training

<img src="assets/training_visualization.gif" alt="Training Visualization" width="480px" />

The next planned experiment is a cross-validation grid over architecture and
input options — see [docs/CV_GRID.md](docs/CV_GRID.md) for the axes, the
protocol, and the workload shape the training pipeline should be optimized for.

## Data Pipeline

The data preprocessing pipeline combines multi-source lake data into standardized NetCDF files for training and inference.

### Data Sources

- **Imagery timestacks**: Sentinel-2 satellite imagery sequences (local samples in `datasets/imgseqs/`)
  - Format: `.nc` files with reflectance bands; 153 observations per lake (May–September)
  - Spatial resolution: 512×512 pixels at 10m/pixel
  - Canonical source for new work: the ESSD SDR deposit (DOI 10.25740/sf350xp4038, stacks_v2
    format: 6-band `reflectance` + `band_name`, `lake_boundary`, `water_mask_ndwi`, `p_water`).
    `LakeDataset` reads this format directly; `p_water` (NaN on unusable days) is
    ffill/bfill-filled at load and used as the area sequence.

- **Area sequences** (legacy composite pipeline): Lake water area time series from
  [Dunmire et al. 2025](https://zenodo.org/records/14587026)
  - Format: Single `.nc` file with daily water area measurements (local samples in `datasets/areaseqs/`)
  - Variables: `S2_water` (Sentinel-2 derived water area)

### Preprocessing Functions

The [lakevision/data/preprocessing.py](lakevision/data/preprocessing.py) module provides utilities for:

1. **Loading and filtering data**:
   - `load_area_sequences()`: Load and filter area data by lake ID and date range
   - `load_imagery_timestack()`: Load imagery with optional band selection
   - `filter_lakes_by_substring()`: Filter lakes by ID pattern (e.g., 'CW2019')

2. **Extracting channels**:
   - `extract_rgb_channels()`: Extract RGB bands from imagery
   - `extract_mask_channel()`: Extract lake mask/SCL band

3. **Time series processing**:
   - `fill_nan_timeseries()`: Fill missing values (forward/backward fill or interpolation)
   - `align_water_area_to_imagery()`: Align daily area data to imagery timestamps
   - `get_lake_water_area()`: Extract and process water area for a single lake

4. **Combining data**:
   - `combine_lake_data()`: Merge imagery and area data into single standardized `.nc` file

### Combined Dataset Format

Each processed lake is saved as a single `.nc` file with the following structure:

```python
xr.Dataset {
    dimensions:
        time: 153      # number of observations
        channel: 4     # RGB + mask (7-channel variants add NIR/SWIR/cloudmask)
        y: 512         # image height
        x: 512         # image width

    data_vars:
        imagery (time, channel, y, x): float32
            # image sequences; raw DN scale (divide by 10000 at load)

        water_area (time,): float32
            # 1D water area time series — guaranteed NaN-free at write time;
            # LakeDataset raises if a NaN slips through

        cloudy_seq_rgb (time,): float32          # optional, some builds only
            # Tile usefulness predictions from RGB model (1=useful, 0=cloudy)

    coords:
        time: datetime64[ns]
        channel: ['red', 'green', 'blue', 'mask']
        lake_id: str
}
```

### Usage Example

```python
from lakevision.data.preprocessing import load_area_sequences, combine_lake_data

# Load area data (dates optional — alignment to imagery timestamps
# selects the melt season, so no year needs to be hardcoded)
area_ds = load_area_sequences('datasets/areaseqs/all_lakes_2019.nc')

# Combine imagery and area data for a single lake
ds = combine_lake_data(
    imagery_path='datasets/imgseqs/tstack_CW2019_1579.nc',
    area_ds=area_ds,
    lake_id='CW2019_1579',
    output_path='datasets/processed/CW2019_1579.nc'
)

print(ds['imagery'].shape)      # (153, 7, 512, 512)
print(ds['water_area'].shape)   # (153,)
print(ds['cloudy_seq_rgb'].shape)  # (153,) - tile usefulness from cloudy-tile model
```

## Model Architecture

The `LakeDrainageClassifier` is a multi-stream temporal model that processes satellite imagery sequences and/or scalar time series to classify lake drainage events.

```
Input Streams (configurable):
├── Image Sequence [B, T, C, H, W] ──► FrontCNN ──► CLSTM ──► GlobalPooling ──► features
├── Area Sequence [B, T, 1] ─────────────────────► ScalarLSTM ─────────────► features
└── Cloudy Sequence [B, T, 1] ────────────────────► ScalarLSTM ─────────────► features
                                                                                  │
                                                          concatenate ◄──────────┘
                                                              │
                                                              ▼
                                                        ClassHeadMLP
                                                              │
                                                              ▼
                                                      logits [B, num_classes]

Where:
  B = batch size
  T = sequence length (number of timesteps)
  C = channels (3-7 depending on spectral bands: RGB + optional NIR/SWIR16/SWIR22 + mask)
  H, W = spatial dimensions (512×512)
```

### Configurable Hyperparameters

#### Input Streams
| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `use_imgseq` | True/False | True | Use satellite imagery sequence |
| `use_areaseq` | True/False | True | Use water area time series |
| `use_cloudyseq` | True/False | False | Use cloudy tile usefulness sequence |
| `use_nir` | True/False | False | Include NIR band in imagery |
| `use_swir16` | True/False | False | Include SWIR16 band in imagery |
| `use_swir22` | True/False | False | Include SWIR22 band in imagery |

#### Normalization

The active path divides reflectance by 10,000 and clips to [0,1] (mask passed
through untouched); the water-area sequence is min-max normalized to [0,1] per
sample. Per-band mean/std standardization is a **planned CV-grid axis, not a
used feature**: the `band_stats` plumbing exists in `LakeDataset` (JSON of
`{"red": {"mean": ..., "std": ...}, ...}`), but no stats file has ever been
computed — see `docs/CV_GRID.md` axis 1 before using it.

#### Learned Temporal Weights
| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `learn_area_weights` | True/False | False | Learn per-timestep weights for area sequence |
| `learn_cloudy_weights` | True/False | False | Learn per-timestep weights for cloudy sequence |

`seq_len` is pinned to 153 (the full melt season) in `run_training.py`; it is
deliberately not a CLI knob or CV-grid axis right now.

#### FrontCNN (Image Feature Extraction)
| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `frontcnn_base_channels` | 8, 16, 32 | 8 | Base channel count (doubles each layer) |
| `frontcnn_num_layers` | 2, 3, 4 | 4 | Number of conv layers (each halves H/W) |
| `frontcnn_out_hw` | None, (1,1), (16,16), ... | None | None keeps the conv stack's natural output (32×32 for 512² input, 4 layers). Pooling *down* is allowed; upsampling requests raise. ESSD repro: see `docs/PROVENANCE_ESSD.md` |
| `frontcnn_pool` | 'max', 'avg', 'none' | 'max' | Pooling used when out_hw is below the conv output |

**Vector mode**: When `frontcnn_out_hw=(1,1)`, the model uses a standard `nn.LSTM` (same architecture as ScalarLSTM) instead of ConvLSTM for temporal processing. The spatial dimensions are squeezed after FrontCNN, and the resulting `[B, T, C]` tensor is processed through a regular LSTM. This is ~3x more parameter efficient but removes spatial reasoning across time.

#### CLSTM (Spatiotemporal Processing)

Only used when `frontcnn_out_hw` > (1,1). In vector mode, the image stream uses a standard `nn.LSTM` instead. Single cell, single layer (there is no layer-stacking parameter).

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `clstm_hidden` | 16, 32, 64 | 32 | Hidden state channels |
| `clstm_kernel` | 3, 5, 7 (odd) | 3 | Convolution kernel size for the gates |

#### Attention Mechanisms
| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `attention_type` | 'none', 'spatial', 'full', 'arch' | 'none' | Type of attention |
| `pool_type` | 'avg', 'max', 'both' | 'avg' | Global pooling method |

**Attention types explained:**
- **none**: No attention, direct CLSTM output
- **spatial**: CBAM spatial attention only (learns *where* to focus)
- **full**: CBAM channel + spatial attention (learns *what* features and *where*)
- **arch**: Architecture-specific attention modifications

**Global pooling types explained:**

After the CLSTM processes the sequence, the final hidden state has shape `[B, C_hidden, H, W]`. GlobalPooling reduces the spatial dimensions to produce a fixed-size feature vector:
- **avg**: Average pooling over spatial dimensions → `[B, C_hidden]`
- **max**: Max pooling over spatial dimensions → `[B, C_hidden]`
- **both**: Concatenates avg and max pooled features → `[B, 2*C_hidden]`

#### ScalarLSTM (Time Series Processing)

Processes 1D scalar sequences (water area, cloud fraction) through standard `nn.LSTM` layers. In **vector mode** (when `frontcnn_out_hw=(1,1)`), the image stream also uses this same LSTM architecture instead of ConvLSTM.

The model supports multiple input stream configurations:

| Configuration | Description |
|---------------|-------------|
| `use_imgseq=True, use_areaseq=True` | Full model: imagery + area sequences (default) |
| `use_imgseq=True, use_areaseq=False` | Imagery only: image stream without the area baseline |
| `use_imgseq=False, use_areaseq=True` | Area only: lightweight baseline using just water area time series |
| `use_imgseq=True, use_areaseq=True, use_cloudyseq=True` | All streams: adds cloud fraction |

**Area-only mode** is useful as a baseline or when imagery is unavailable. The water area time series alone can capture drainage signatures (rapid area decrease = drainage event).


#### ClassHeadMLP (Classification)
| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `classhead_hidden` | None, 32, 64, [64,32] | 64 | Hidden layer dimensions |
| `classhead_dropout` | 0.0 - 0.5 | 0.0 (0.3 in training CLI) | Dropout probability |
| `classhead_activation` | 'relu', 'leakyrelu', 'gelu' | 'relu' | Activation function |
| `num_classes` | 4, 5, ... | 4 (model) / 5 (training CLI, ESSD) | Number of output classes |

### Training Hyperparameters

Argparse defaults in `engine/training/run_training.py` are the ESSD baseline.

| Parameter | Baseline | Description |
|-----------|----------|-------------|
| `seq_len` | 153 (pinned) | Full melt season; not a CLI knob |
| `batch_size` | 8 | bs≥32 OOMs a 40 GB A100 on main's pipeline |
| `lr` | 1e-4 fixed | No scheduler (flag exists for ablations) |
| `weight_decay` | 1e-5 | L2 regularization |
| `epochs` | 400 | Baselines are compute-limited, not converged |


### Model Inference
Model inference information

## Tests

The project includes comprehensive test coverage for all core components. Tests are written using pytest and located in the [lakevision/tests](lakevision/tests) directory.

### Running Tests

To run all tests:
```bash
pytest lakevision/tests/
```

To run tests for a specific module:
```bash
pytest lakevision/tests/test_attention.py
pytest lakevision/tests/test_clstm.py
pytest lakevision/tests/test_blocks.py
pytest lakevision/tests/test_classifier.py
```

To run tests with verbose output:
```bash
pytest lakevision/tests/ -v
```

To run a specific test function:
```bash
pytest lakevision/tests/test_attention.py::TestSpatialCBAM::test_basic_forward
```

### Test Coverage

The test suite covers:
- **Attention mechanisms** ([test_attention.py](lakevision/tests/test_attention.py)): SpatialCBAM and FullCBAM
- **Convolutional LSTM** ([test_clstm.py](lakevision/tests/test_clstm.py)): CellCLSTM and CLSTM
- **Building blocks** ([test_blocks.py](lakevision/tests/test_blocks.py)): ScalarLSTM, ClassHeadMLP, GlobalPooling
- **Full classifier** ([test_classifier.py](lakevision/tests/test_classifier.py)): LakeDrainageClassifier (integrated model)
- **Dataset loading** ([test_datasets.py](lakevision/tests/test_datasets.py)): LakeDataset class for loading processed NetCDF files
- **End-to-end training** ([test_training.py](lakevision/tests/test_training.py)): Full training pipeline (data loading → forward pass → loss → backward pass)
