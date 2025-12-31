"""
Lakevision package for supraglacial lake drainage classification.
"""

__version__ = "0.1.0"

# Import models
from .models.blocks import (
    FrontCNN,
    ScalarLSTM,
    ClassHeadMLP,
    GlobalPooling,
)

from .models.clstm import (
    ConvLSTMCell,
    ConvLSTM,
)

from .models.attention import (
    SpatialCBAM,
    FullCBAM,
)

from .models.classifier import (
    LakeDrainageClassifier,
)

__all__ = [
    # Blocks
    'FrontCNN',
    'ScalarLSTM',
    'ClassHeadMLP',
    'GlobalPooling',
    # ConvLSTM
    'CellCLSTM',
    'CLSTM',
    # Attention
    'SpatialCBAM',
    'FullCBAM',
    # Classifier
    'LakeDrainageClassifier',
]