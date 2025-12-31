"""
Tests for lakevision.models.blocks module.

Tests individual components from blocks.py module:
FrontCNN, ScalarLSTM, ClassHeadMLP, and GlobalPooling
"""
import pytest
import torch
import torch.nn as nn

from lakevision.models.blocks import (
    FrontCNN,
    ScalarLSTM,
    ClassHeadMLP,
    GlobalPooling,
)

class TestFrontCNN:
    """
    Tests for FrontCNN block.
    """
    def test_basic_forward(self):
        """Test basic forward pass of FrontCNN."""
        model = FrontCNN(in_channels=3, base_channels=8, num_layers=3)
        x = torch.randn(2, 153, 3, 512, 512)  # [B=2, T=153, C=4, H=512, W=512]
        out = model(x)

        # with 3 layers and default out_hw=(32,32) we expect output shape to be [B=2, T=153, C=32, H=32, W=32]
        assert out.shape == (2, 153, 32, 32, 32), f"Expected output shape (2, 153, 32, 32, 32), but got {out.shape}"

        # check for nans or infs
        assert not torch.isnan(out).any(), "Output contains NaNs"
        assert not torch.isinf(out).any(), "Output contains Infs"

    def test_output_channels(self):
        """Test output_channels attribute."""
        model = FrontCNN(in_channels=4, base_channels=8, num_layers=2)
        expected = 8 * (2 ** (2 - 1))  # base_channels * 2^(num_layers-1
        assert model.output_channels == expected, f"Expected output_channels {expected}, but got {model.output_channels}"

    def test_conditional_pooling_needed(self):
        """Test conditional pooling logic."""
        # with 3 layers, natural output is 512/(2^3) = 64x64
        # but let's say we want 32x32, so adaptive pooling should kick in
        model = FrontCNN(
            in_channels=3,
            base_channels=8,
            num_layers=3,
            out_hw=(32,32),
            pool='avg',
        )
        x = torch.randn(2, 153, 3, 512, 512)
        out = model(x)
        assert out.shape[-2:] == (32, 32), f"Expected output spatial size (32,32), but got {out.shape[-2,:]}"

    def test_conditional_pooling_not_needed(self):
        """Test when conditional pooling is not needed."""
        # with 3 layers, natural output is 512/(2^3) = 64x64,
        # so therefore setting out_hw to 64x64 should skip pooling
        model = FrontCNN(
            in_channels=3,
            base_channels=8,
            num_layers=3,
            out_hw=(64,64),
            pool='max',
        )
        x = torch.randn(2, 153, 3, 512, 512)
        out = model(x)
        assert out.shape[-2,:] == (64, 64), f"Expected output spatial size (64,64), but got {out.shape[-2:]}"

    def test_pool_to_1x1(self):
        """Test pooling to 1x1 output (e.g., if we wanted to just use the LSTM later not the CLSTM)."""
        model = FrontCNN(
            in_channels=3,
            base_channels=8,
            num_layers=3,
            out_hw=(1,1),
            pool='avg',
        )
        x = torch.randn(2, 153, 3, 512, 512)
        out = model(x)
        assert out.shape[-2,:] == (1, 1), f"Expected output spatial size (1,1), but got {out.shape[-2,:]}"

    def test_pool_none_matching_dims(self):
        """Test pool='none' when dimensions naturally match."""
        # 3 layers gives 64x64 naturally from 512x512 input, so this should work
        model = FrontCNN(
            in_channels=3,
            base_channels=8,
            num_layers=3,
            out_hw=(64,64),
            pool='none',
        )
        x = torch.randn(2, 153, 3, 512, 512)
        out = model(x)
        assert out.shape[-2,:] == (64, 64), f"Expected output spatial size (64,64), but got {out.shape[-2,:]}"

    def test_pool_none_mismatched_dims_raise_error(self):
        """Test that pool='none' with mismatched dims raises error."""
        model = FrontCNN(
            in_channels=3,
            base_channels=8,
            num_layers=3,
            out_hw=(32,32),  # mismatched from natural 64x64
            pool='none',
        )
        x = torch.randn(2, 153, 3, 512, 512)
        with pytest.raises(ValueError, match="Output dimensions .* do not match target"):
            model(x)

    def test_different_pool_types(self):
        """Test different pooling types."""
        for pool_type in ['max', 'avg']:
            model = FrontCNN(
                in_channels=3,
                base_channels=8,
                num_layers=3,
                out_hw=(32,32),
                pool=pool_type,
            )
            x = torch.randn(2, 153, 3, 512, 512)
            out = model(x)
            assert out.shape[-2,:] == (32, 32), f"Expected output spatial size (32,32) with pool={pool_type}, but got {out.shape[-2,:]}"

    def test_invalid_pool_type_raises_error(self):
        """Test that invalid pool type raises error."""
        with pytest.raises(ValueError, match="pool must be one of"):
            FrontCNN(in_channels=3, base_channels=8, num_layers=2, pool='invalid')

    def test_different_input_sizes(self):
        """Test with different input spatial sizes (though we will always use 512x512)."""
        model = FrontCNN(in_channels=3, base_channels=8, num_layers=2, out_hw=(64,64))

        sizes = [(256,256), (512,512), (1024,1024)]
        for H, W in sizes:
            x = torch.randn(1, 5, 3, H, W)
            out = model(x)
            assert out.shape[0] == 1 and out.shape[1] == 5, \
                f"Batch/time dims changed for input {H}x{W}"
            assert out.shape[-2:] == (64,64), \
                f"Output dims incorrect for input {H}x{W}"