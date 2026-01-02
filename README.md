# lake-vision
Framework for tracking supraglacial lake evolution.  Developed for Greenland.  Applicable globally.

Note: This repo is under active development

## Introduction
Introduction...

## Installation
Installation guide...

## Training

<img src="assets/training_visualization.gif" alt="Training Visualization" width="480px" />

## Usage
Usage guide for the different functionalities...

### Data Pipeline
Pipeline to acquire and prepare ml-ready imagery data

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
