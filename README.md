# lake-vision
Framework for tracking supraglacial lake evolution.  Developed for Greenland.  Applicable globally.

Note: This repo is under active development

## Introduction
Introduction...

## Installation
Installation guide...

## Training

<img src="assets/training_visualization.gif" alt="Training Visualization" width="480px" />

## Data Pipeline

The data preprocessing pipeline combines multi-source lake data into standardized NetCDF files for training and inference.

### Data Sources

- **Imagery timestacks**: Sentinel-2 satellite imagery sequences (`lakevision/data/samples/imgseqs/`)
  - Format: `.nc` files with reflectance bands (red, green, blue, mask/SCL)
  - Temporal resolution: ~153 observations per lake (May-September 2019)
  - Spatial resolution: 512×512 pixels at 10m/pixel

- **Area sequences**: Lake water area time series (`lakevision/data/samples/areaseqs/`)
  - Format: Single `.nc` file with daily water area measurements
  - Variables: `S2_water` (Sentinel-2 derived water area)
  - Temporal coverage: Full year 2019

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
        channel: 4     # RGB + mask
        y: 512         # image height
        x: 512         # image width

    data_vars:
        imagery (time, channel, y, x): float32
            # 4D image sequences [red, green, blue, mask]

        water_area (time,): float32
            # 1D water area time series (NaNs filled)

    coords:
        time: datetime64[ns]
        channel: ['red', 'green', 'blue', 'mask']
        lake_id: str
}
```

### Usage Example

```python
from lakevision.data.preprocessing import load_area_sequences, combine_lake_data

# Load area data for all CW2019 lakes (May-September 2019)
area_ds = load_area_sequences(
    'lakevision/data/samples/areaseqs/all_lakes_2019.nc',
    start_date='2019-05-01',
    end_date='2019-09-30'
)

# Combine imagery and area data for a single lake
ds = combine_lake_data(
    imagery_path='lakevision/data/samples/imgseqs/tstack_CW2019_1579.nc',
    area_ds=area_ds,
    lake_id='CW2019_1579',
    output_path='lakevision/data/samples/processed/CW2019_1579.nc'
)

print(ds['imagery'].shape)    # (153, 4, 512, 512)
print(ds['water_area'].shape) # (153,)
```

### Model Training
Model training information

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
