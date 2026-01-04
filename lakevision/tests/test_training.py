"""
End-to-end training pipeline test.

Tests the full workflow: data loading -> model forward -> loss -> backward.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from lakevision.data.datasets import LakeDataset
from lakevision.models.classifier import LakeDrainageClassifier

# paths to real processed data and labels
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent / "datasets" / "processed" / "CW2019_1579.nc"
LABELS_PATH = Path(__file__).parent.parent.parent / "labels" / "labels_2019_volumes_CW_demo.csv"


@pytest.mark.skipif(
    not PROCESSED_DATA_PATH.exists() or not LABELS_PATH.exists(),
    reason="Processed data or labels not available"
)
class TestTrainingPipeline:
    """End-to-end training pipeline tests using real data."""

    def test_forward_pass(self):
        """Test a single forward pass through the model."""
        dataset = LakeDataset(
            PROCESSED_DATA_PATH,
            seq_len=21,
            labels_file=LABELS_PATH,
            id_col='new_id',
            label_col='label_rines'
        )
        loader = DataLoader(dataset, batch_size=1)

        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            use_cloudyseq=False,
            attention_type='none',
        )

        img_seq, area_seq, label, lake_id = next(iter(loader))
        logits = model(img_seq, area_seq, None)

        assert logits.shape == (1, 4)
        assert not torch.isnan(logits).any()

    def test_forward_backward_pass(self):
        """Test forward and backward pass (gradient computation)."""
        dataset = LakeDataset(
            PROCESSED_DATA_PATH,
            seq_len=21,
            labels_file=LABELS_PATH,
            id_col='new_id',
            label_col='label_rines'
        )
        loader = DataLoader(dataset, batch_size=1)

        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            use_cloudyseq=False,
            attention_type='none',
        )
        criterion = nn.CrossEntropyLoss()

        img_seq, area_seq, label, lake_id = next(iter(loader))
        logits = model(img_seq, area_seq, None)
        loss = criterion(logits, label)
        loss.backward()

        # check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_training_step(self):
        """Test a complete training step with optimizer."""
        dataset = LakeDataset(
            PROCESSED_DATA_PATH,
            seq_len=21,
            labels_file=LABELS_PATH,
            id_col='new_id',
            label_col='label_rines'
        )
        loader = DataLoader(dataset, batch_size=1)

        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            use_cloudyseq=False,
            attention_type='none',
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        img_seq, area_seq, label, lake_id = next(iter(loader))

        optimizer.zero_grad()
        logits = model(img_seq, area_seq, None)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        assert not np.isnan(loss.item()), "Loss is NaN"
        assert loss.item() > 0, "Loss should be positive"

    def test_attention_types(self):
        """Test training with different attention mechanisms."""
        dataset = LakeDataset(
            PROCESSED_DATA_PATH,
            seq_len=21,
            labels_file=LABELS_PATH,
            id_col='new_id',
            label_col='label_rines'
        )
        loader = DataLoader(dataset, batch_size=1)
        img_seq, area_seq, label, lake_id = next(iter(loader))

        for attention_type in ['none', 'spatial', 'full', 'arch']:
            model = LakeDrainageClassifier(
                use_imgseq=True,
                use_areaseq=True,
                use_cloudyseq=False,
                attention_type=attention_type,
            )
            criterion = nn.CrossEntropyLoss()

            logits = model(img_seq, area_seq, None)
            loss = criterion(logits, label)
            loss.backward()

            assert not torch.isnan(logits).any(), f"NaN in logits with {attention_type}"
            assert not np.isnan(loss.item()), f"NaN loss with {attention_type}"

    def test_imgseq_only(self):
        """Test training with image sequence only."""
        dataset = LakeDataset(
            PROCESSED_DATA_PATH,
            seq_len=21,
            labels_file=LABELS_PATH,
            id_col='new_id',
            label_col='label_rines'
        )
        loader = DataLoader(dataset, batch_size=1)

        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=False,
            use_cloudyseq=False,
        )
        criterion = nn.CrossEntropyLoss()

        img_seq, area_seq, label, lake_id = next(iter(loader))
        logits = model(img_seq, None, None)
        loss = criterion(logits, label)
        loss.backward()

        assert logits.shape == (1, 4)

    def test_areaseq_only(self):
        """Test training with area sequence only."""
        dataset = LakeDataset(
            PROCESSED_DATA_PATH,
            seq_len=21,
            labels_file=LABELS_PATH,
            id_col='new_id',
            label_col='label_rines'
        )
        loader = DataLoader(dataset, batch_size=1)

        model = LakeDrainageClassifier(
            use_imgseq=False,
            use_areaseq=True,
            use_cloudyseq=False,
        )
        criterion = nn.CrossEntropyLoss()

        img_seq, area_seq, label, lake_id = next(iter(loader))
        logits = model(None, area_seq, None)
        loss = criterion(logits, label)
        loss.backward()

        assert logits.shape == (1, 4)