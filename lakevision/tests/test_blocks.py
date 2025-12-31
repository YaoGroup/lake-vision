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
        x = torch.randn(2, 13, 3, 512, 512)  # [B=2, T=13, C=4, H=512, W=512]
        out = model(x)

        # with 3 layers and default out_hw=(32,32) we expect output shape to be [B=2, T=13, C=32, H=32, W=32]
        assert out.shape == (2, 13, 32, 32, 32), f"Expected output shape (2, 13, 32, 32, 32), but got {out.shape}"

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
        x = torch.randn(2, 13, 3, 512, 512)
        out = model(x)
        assert out.shape[-2:] == (32, 32), f"Expected output spatial size (32,32), but got {out.shape[-2:]}"

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
        x = torch.randn(2, 13, 3, 512, 512)
        out = model(x)
        assert out.shape[-2:] == (64, 64), f"Expected output spatial size (64,64), but got {out.shape[-2:]}"

    def test_pool_to_1x1(self):
        """Test pooling to 1x1 output (e.g., if we wanted to just use the LSTM later not the CLSTM)."""
        model = FrontCNN(
            in_channels=3,
            base_channels=8,
            num_layers=3,
            out_hw=(1,1),
            pool='avg',
        )
        x = torch.randn(2, 13, 3, 512, 512)
        out = model(x)
        assert out.shape[-2:] == (1, 1), f"Expected output spatial size (1,1), but got {out.shape[-2:]}"

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
        x = torch.randn(2, 13, 3, 512, 512)
        out = model(x)
        assert out.shape[-2:] == (64, 64), f"Expected output spatial size (64,64), but got {out.shape[-2:]}"

    # def test_pool_none_mismatched_dims_raise_error(self):
    #     """Test that pool='none' with mismatched dims raises error."""
    #     model = FrontCNN(
    #         in_channels=3,
    #         base_channels=8,
    #         num_layers=3,
    #         out_hw=(32,32),  # mismatched from natural 64x64
    #         pool='none',
    #     )
    #     x = torch.randn(2, 13, 3, 512, 512)
    #     with pytest.raises(ValueError, match="Output dimensions .* do not match target"):
    #         model(x)

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
            x = torch.randn(2, 13, 3, 512, 512)
            out = model(x)
            assert out.shape[-2:] == (32, 32), f"Expected output spatial size (32,32) with pool={pool_type}, but got {out.shape[-2:]}"

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
            

class TestScalarLSTM:
    """
    Tests for ScalarLSTM block.
    """
    def test_basic_forward(self):
        """Test for basic forward pass of ScalarLSTM."""
        model = ScalarLSTM(hidden_dim=16, num_layers=1)
        x = torch.randn(2, 13, 1) # [B=2, T=153, 1]
        out = model(x)

        # expected output shape: [B=2, T=13, hidden_dim=16]
        assert out.shape == (2, 13, 16), f"Expected output shape (2, 13, 16), but got {out.shape}"

        # check for nans or infs
        assert not torch.isnan(out).any(), "Output contains NaNs"
        assert not torch.isinf(out).any(), "Output contains Infs"

    def test_multiple_layers(self):
        """Test ScalarLSTM with multiple stacked layers."""
        model = ScalarLSTM(hidden_dim=16, num_layers=3)
        x = torch.randn(4, 13, 1) # [B=4, T=13, 1]
        out = model(x)

        assert out.shape == (4, 13, 16), f"Expected output shape (4, 13, 16), but got {out.shape}"

    def test_with_dropout(self):
        """Test ScalarLSTM with dropout enabled."""
        model = ScalarLSTM(hidden_dim=16, num_layers=2, dropout=0.5)
        x = torch.randn(2, 13, 1)

        # in trainig mode, dropout should be active
        model.train()
        out = model(x)
        assert out.shape == (2, 13, 16), f"Expected output shape (2, 13, 16) but got {out.shape}"

        # in eval mode
        model.eval()
        out_eval = model(x)
        assert out_eval.shape == (2, 13, 16), f"Expected output shape (2, 13, 16) but got {out_eval.shape}"

    def test_invalid_input_shape(self):
        """Test that invalid input shape raises error."""
        model = ScalarLSTM(hidden_dim=16)
        x = torch.randn(2, 13, 3) # purposely wrong here, should be [B, T, 1]

        with pytest.raises(ValueError, match="Expected input with last dimension = 1"):
            model(x)

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        model = ScalarLSTM(hidden_dim=16)

        for batch_size in [1, 8 ,16]:
            x = torch.randn(batch_size, 13, 1)
            out = model(x)
            assert out.shape == (batch_size, 13, 16), f"Failed for batch_size={batch_size}"

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        model = ScalarLSTM(hidden_dim=16)

        for seq_len in [2, 13, 50]:
            x = torch.randn(2, seq_len, 1)
            out = model(x)
            assert out.shape == (2, seq_len, 16), f"Failed for seq_len={seq_len}"

class TestClassHeadMLP:
    """
    Tests for ClassHeadMLP block.
    """
    def test_basic_forward(self):
        """Test basic forward pass with single hidden layer."""
        model = ClassHeadMLP(input_dim=80, hidden_dims=64, num_classes=4)
        x = torch.randn(16, 80) # [B=16, input_dim=80]
        out = model(x)

        # expected output shape: [B=16, num_classes=4]
        assert out.shape == (16, 4), f"Expected output shape (16, 4) but got {out.shape}"

        # Check for nans or infs
        assert not torch.isnan(out).any(), "Output contains NaNs"
        assert not torch.isinf(out).any(), "Output contains Infs"

    def test_multiple_hidden_layers(self):
        """Test with multiple hidden layers."""
        model = ClassHeadMLP(input_dim=80, hidden_dims=[64, 32], num_classes=4)
        x = torch.randn(8, 80)
        out = model(x)

        assert out.shape == (8, 4), f"Expected output shape (8, 4), but got {out.shape}"

    def test_no_hidden_layers(self):
        """Test direct linear projection (no hidden layers)."""
        model = ClassHeadMLP(input_dim=80, hidden_dims=None, num_classes=4)
        x = torch.randn(4, 80)
        out = model(x)

        assert out.shape == (4, 4), f"Expected output shape (4, 4), but got {out.shape}"

    def test_with_dropout(self):
        """Test with dropout enabled."""
        model = ClassHeadMLP(input_dim=80, hidden_dims=64, num_classes=4, dropout=0.5)
        x = torch.randn(16, 80)

        # in training mode
        model.train()
        out = model(x)
        assert out.shape == (16, 4), f"Expected output shape (16, 4) but got {out.shape}"

        # in eval mode
        model.eval()
        out_eval = model(x)
        assert out_eval.shape == (16, 4), f"Expected output shape (16, 4) but got {out_eval.shape}"

    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['relu', 'leakyrelu', 'gelu']

        for act in activations:
            model = ClassHeadMLP(input_dim=80, hidden_dims=64, num_classes=4, activation=act)
            x = torch.randn(8, 80)
            out = model(x)
            assert out.shape == (8, 4), f"Failed for activation={act}"

    def test_invalid_activation_raises_error(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="activation must be one of"):
            ClassHeadMLP(input_dim=80, hidden_dims=64, num_classes=4, activation='invalid')

    def test_invalid_hidden_dims_type_raises_error(self):
        """Test that invalid hidden_dims type raises error."""
        with pytest.raises(ValueError, match="hidden_dims must be an int"):
            ClassHeadMLP(input_dim=80, hidden_dims="invalid", num_classes=4)

    def test_different_num_classes(self):
        """Test with different number of output classes."""
        for num_classes in [2, 4, 10]:
            model = ClassHeadMLP(input_dim=80, hidden_dims=64, num_classes=num_classes)
            x = torch.randn(8, 80)
            out = model(x)
            assert out.shape == (8, num_classes), \
                f"Failed for num_classes={num_classes}"
            
    def test_softmax_and_argmax(self):
        """Test that output can be used with softmax and argmax."""
        model = ClassHeadMLP(input_dim=80, hidden_dims=64, num_classes=4)
        x = torch.randn(16, 80)
        logits = model(x)

        # Apply softmax
        probs = torch.softmax(logits, dim=1)
        assert probs.shape == (16, 4), "Softmax output shape mismatch"
        assert torch.allclose(probs.sum(dim=1), torch.ones(16)), \
            "Softmax probabilities don't sum to 1"

        # Apply argmax
        preds = torch.argmax(logits, dim=1)
        assert preds.shape == (16,), "Argmax output shape mismatch"
        assert (preds >= 0).all() and (preds < 4).all(), \
            "Predictions out of class range"
        
class TestGlobalPooling:
    """
    Tests for GlobalPooling block.
    """
    def test_avg_pooling_4d(self):
        """Test average pooling with 4D input [B, C, H, W]."""
        pool = GlobalPooling(pool_type='avg')
        x = torch.randn(2, 32, 64, 64)  # [B=2, C=32, H=64, W=64]
        out = pool(x)

        # Expected output shape: [B=2, C=32]
        assert out.shape == (2, 32), f"Expected output shape (2, 32), but got {out.shape}"

        # Check for nans or infs
        assert not torch.isnan(out).any(), "Output contains NaNs"
        assert not torch.isinf(out).any(), "Output contains Infs"

    def test_max_pooling_4d(self):
        """Test max pooling with 4D input [B, C, H, W]."""
        pool = GlobalPooling(pool_type='max')
        x = torch.randn(2, 32, 64, 64)
        out = pool(x)

        assert out.shape == (2, 32), f"Expected output shape (2, 32), but got {out.shape}"

    def test_both_pooling_4d(self):
        """Test both (avg+max) pooling with 4D input."""
        pool = GlobalPooling(pool_type='both')
        x = torch.randn(2, 32, 64, 64)
        out = pool(x)

        # Channels should be doubled: [B=2, C=64]
        assert out.shape == (2, 64), f"Expected output shape (2, 64), but got {out.shape}"

    def test_avg_pooling_5d(self):
        """Test average pooling with 5D input [B, T, C, H, W]."""
        pool = GlobalPooling(pool_type='avg')
        x = torch.randn(2, 10, 32, 64, 64)  # [B=2, T=10, C=32, H=64, W=64]
        out = pool(x)

        # Expected output shape: [B=2, T=10, C=32]
        assert out.shape == (2, 10, 32), f"Expected output shape (2, 10, 32), but got {out.shape}"

    def test_max_pooling_5d(self):
        """Test max pooling with 5D input [B, T, C, H, W]."""
        pool = GlobalPooling(pool_type='max')
        x = torch.randn(2, 10, 32, 64, 64)
        out = pool(x)

        assert out.shape == (2, 10, 32), f"Expected output shape (2, 10, 32), but got {out.shape}"

    def test_both_pooling_5d(self):
        """Test both pooling with 5D input [B, T, C, H, W]."""
        pool = GlobalPooling(pool_type='both')
        x = torch.randn(2, 10, 32, 64, 64)
        out = pool(x)

        # Channels should be doubled: [B=2, T=10, C=64]
        assert out.shape == (2, 10, 64), f"Expected output shape (2, 10, 64), but got {out.shape}"

    def test_invalid_pool_type_raises_error(self):
        """Test that invalid pool_type raises error."""
        with pytest.raises(ValueError, match="pool_type must be one of"):
            GlobalPooling(pool_type='invalid')

    def test_different_spatial_sizes(self):
        """Test with different spatial dimensions."""
        pool = GlobalPooling(pool_type='avg')

        spatial_sizes = [(32, 32), (64, 64), (128, 128), (16, 8)]
        for H, W in spatial_sizes:
            x = torch.randn(2, 32, H, W)
            out = pool(x)
            assert out.shape == (2, 32), \
                f"Failed for spatial size ({H}, {W})"

    def test_avg_pooling_correctness(self):
        """Test that average pooling computes correct mean."""
        pool = GlobalPooling(pool_type='avg')

        # Create a simple tensor with known values
        x = torch.ones(1, 2, 4, 4)  # [B=1, C=2, H=4, W=4]
        x[0, 0, :, :] = 2.0  # First channel all 2s
        x[0, 1, :, :] = 3.0  # Second channel all 3s

        out = pool(x)

        # Output should be [1, 2] with values [2.0, 3.0]
        assert torch.allclose(out[0, 0], torch.tensor(2.0)), "Avg pooling incorrect for channel 0"
        assert torch.allclose(out[0, 1], torch.tensor(3.0)), "Avg pooling incorrect for channel 1"

    def test_max_pooling_correctness(self):
        """Test that max pooling computes correct maximum."""
        pool = GlobalPooling(pool_type='max')

        # Create a tensor with one max value per channel
        x = torch.zeros(1, 2, 4, 4)  # [B=1, C=2, H=4, W=4]
        x[0, 0, 2, 2] = 5.0  # Max in first channel
        x[0, 1, 1, 3] = 7.0  # Max in second channel

        out = pool(x)

        # Output should be [1, 2] with values [5.0, 7.0]
        assert torch.allclose(out[0, 0], torch.tensor(5.0)), "Max pooling incorrect for channel 0"
        assert torch.allclose(out[0, 1], torch.tensor(7.0)), "Max pooling incorrect for channel 1"

    def test_both_pooling_concatenates_correctly(self):
        """Test that 'both' mode correctly concatenates avg and max."""
        pool = GlobalPooling(pool_type='both')

        # Create simple tensor
        x = torch.ones(1, 2, 4, 4) * 2.0
        x[0, 0, 0, 0] = 10.0  # Add a max value

        out = pool(x)

        # Output should have 4 channels (2 original * 2 for avg+max)
        assert out.shape == (1, 4), f"Expected shape (1, 4), got {out.shape}"