"""
Tests for lakevision.models.attention module.

Tests spatial attention mechanisms:
SpatialCBAM (spatial-only CBAM) and FullCBAM (channel + spatial CBAM).
"""
import pytest
import torch
import torch.nn as nn

from lakevision.models.attention import(
    SpatialCBAM,
    FullCBAM,
)

class TestSpatialCBAM:
    """
    Tests for SpatialCBAM.
    """
    def test_basic_forward(self):
        """Test basic forward pass with default kernel size."""
        attn = SpatialCBAM(in_channels=32, kernel_size=7)
        x = torch.randn(2, 13, 32, 64, 64) # [B=2, T=13, C=32, H=64, W=64]
        out = attn(x)

        # expected output shape: same as input [B=2, T=13, C=32, H=64, W=64]
        assert out.shape == (2, 13, 32, 64, 64), f"Expected output shape (2, 13, 32, 64, 64) but got {out.shape}"

        # check for nans or infs
        assert not torch.isnan(out).any(), "Output contains NaNs"
        assert not torch.isinf(out).any(), "Output contains Infs"

    def test_spatial_dimensions_preserved(self):
        """Test that spatial dimensions are preserved."""
        attn = SpatialCBAM(in_channels=64, kernel_size=3)

        spatial_sizes = [(32,32), (64,64), (128,128)]
        for H, W in spatial_sizes:
            x = torch.randn(2, 13, 64, H, W)
            out = attn(x)
            assert out.shape == (2, 13, 64, H, W), f"Spatial dimensions not preserved for {H}x{W}"

    def test_different_kernel_sizes(self):
        """Test with different odd kernel sizes."""
        kernel_sizes = [1, 3, 5, 7]

        for ks in kernel_sizes:
            attn = SpatialCBAM(in_channels=32, kernel_size=ks)
            x = torch.randn(2, 13, 32, 64, 64)
            out = attn(x)
            assert out.shape == (2, 13, 32, 64, 64), f"Failed for kernel_size={ks}"

    def test_even_kernel_size_raises_error(self):
        """Test that a choice of even kernel size raises an error."""
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            SpatialCBAM(in_channels=32, kernel_size=4)

    def test_different_channel_configs(self):
        """Test with different channel configurations."""
        channel_configs = [8, 16, 32, 64, 128]

        for C in channel_configs:
            attn = SpatialCBAM(in_channels=C, kernel_size=7)
            x = torch.randn(2, 5, C, 64, 64)
            out = attn(x)
            assert out.shape == (2, 5, C, 64, 64), \
                f"Failed for in_channels={C}"
            
    def test_attention_modifies_features(self):
        """Test that attention actually modifies the input features."""
        attn = SpatialCBAM(in_channels=32, kernel_size=7)
        x = torch.randn(2, 5, 32, 64, 64)
        out = attn(x)

        # Output should be different from input (attention applied)
        assert not torch.allclose(out, x), \
            "Attention didn't modify input features"
        
    def test_single_timestep(self):
        """Test with single timestep (T=1)."""
        attn = SpatialCBAM(in_channels=32, kernel_size=7)
        x = torch.randn(2, 1, 32, 64, 64)  # T=1
        out = attn(x)

        assert out.shape == (2, 1, 32, 64, 64), \
            f"Failed for single timestep, got shape {out.shape}"
        
    def test_gradient_flow(self):
        """Test that the gradients flow through the attention module."""
        attn = SpatialCBAM(in_channels=32, kernel_size=7)
        x = torch.randn(2, 13, 32, 64, 64, requires_grad=True)

        out = attn(x)
        loss = out.sum()
        loss.backward()

        # check that the gradient exists
        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaNs"

    def test_train_eval_consistency(self):
        """Test that attention produces consistent results in train vs eval mode."""
        attn = SpatialCBAM(in_channels=32, kernel_size=7)
        x = torch.randn(2, 5, 32, 64, 64)

        attn.train()
        out_train = attn(x)

        attn.eval()
        out_eval = attn(x)

        # No dropout/batchnorm, so outputs should be identical
        assert torch.allclose(out_train, out_eval, atol=1e-6), \
            "Train and eval outputs differ (should be identical)"
        
    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        attn = SpatialCBAM(in_channels=32, kernel_size=7)

        # Process batch of 2
        x_batch = torch.randn(2, 5, 32, 64, 64)
        out_batch = attn(x_batch)

        # Process each sample individually
        out_0 = attn(x_batch[0:1])
        out_1 = attn(x_batch[1:2])

        # Results should match
        assert torch.allclose(out_batch[0], out_0[0], atol=1e-6), \
            "Batch processing differs from individual processing"
        assert torch.allclose(out_batch[1], out_1[0], atol=1e-6), \
            "Batch processing differs from individual processing"
        
    def test_attention_map_range(self):
        """Test that attention map values are in [0, 1] range due to sigmoid."""
        attn = SpatialCBAM(in_channels=32, kernel_size=7)
        x = torch.randn(2, 5, 32, 64, 64)

        # Since attention uses sigmoid, output should be scaled version of input
        # but not exceed input magnitude significantly
        out = attn(x)

        # Attention map is in [0,1], so output magnitude should not exceed input
        assert out.abs().max() <= x.abs().max() * 1.1, \
            "Output magnitude unexpectedly large"
        

class TestFullCBAM:
    """
    Tests for FullCBAM (channel + spatial attention).
    """
    def test_basic_forward(self):
        """Test basic forward pass with default parameters."""
        attn = FullCBAM(in_channels=32)
        x = torch.randn(2, 13, 32, 64, 64) # [B=2, T=13, C=32, H=64, W=64]
        out = attn(x)

        # expected output shape: same as input [B=2, T=13, C=32, H=64, W=64]
        assert out.shape == (2, 13, 32, 64, 64), f"Expected output shape (2, 13, 32, 64, 64) but got {out.shape}"

        # check for nans or infs
        assert not torch.isnan(out).any(), "Output contains NaNs"
        assert not torch.isinf(out).any(), "Output contains Infs"

    def test_spatial_dimensions_preserved(self):
        """Test that spatial dimensions are preserved."""
        attn = FullCBAM(in_channels=64, reduction_ratio=16, kernel_size=7)

        spatial_sizes = [(32, 32), (64, 64), (128, 128)]
        for H, W in spatial_sizes:
            x = torch.randn(2, 5, 64, H, W)
            out = attn(x)
            assert out.shape == (2, 5, 64, H, W), \
                f"Spatial dimensions not preserved for {H}x{W}"

    def test_different_reduction_ratios(self):
        """Test with different channel reduction ratios."""
        reduction_ratios = [4, 8, 16, 32]

        for ratio in reduction_ratios:
            attn = FullCBAM(in_channels=64, reduction_ratio=ratio, kernel_size=7)
            x = torch.randn(2, 5, 64, 64, 64)
            out = attn(x)
            assert out.shape == (2, 5, 64, 64, 64), \
                f"Failed for reduction_ratio={ratio}"

    def test_different_kernel_sizes(self):
        """Test with different spatial kernel sizes."""
        kernel_sizes = [1, 3, 5, 7]

        for ks in kernel_sizes:
            attn = FullCBAM(in_channels=32, reduction_ratio=16, kernel_size=ks)
            x = torch.randn(2, 5, 32, 64, 64)
            out = attn(x)
            assert out.shape == (2, 5, 32, 64, 64), \
                f"Failed for kernel_size={ks}"

    def test_different_channel_configs(self):
        """Test with different channel configurations."""
        channel_configs = [8, 16, 32, 64, 128]

        for C in channel_configs:
            attn = FullCBAM(in_channels=C, reduction_ratio=8, kernel_size=7)
            x = torch.randn(2, 5, C, 64, 64)
            out = attn(x)
            assert out.shape == (2, 5, C, 64, 64), \
                f"Failed for in_channels={C}"

    def test_small_channels_with_large_reduction(self):
        """Test that small channels with large reduction ratio still works (bottleneck >= 1)."""
        # With 8 channels and reduction_ratio=16, bottleneck should be max(8//16, 1) = 1
        attn = FullCBAM(in_channels=8, reduction_ratio=16, kernel_size=7)
        x = torch.randn(2, 5, 8, 64, 64)
        out = attn(x)

        assert out.shape == (2, 5, 8, 64, 64), \
            "Failed for small channels with large reduction ratio"

    def test_attention_modifies_features(self):
        """Test that attention actually modifies the input features."""
        attn = FullCBAM(in_channels=32, reduction_ratio=16, kernel_size=7)
        x = torch.randn(2, 5, 32, 64, 64)
        out = attn(x)

        # Output should be different from input (attention applied)
        assert not torch.allclose(out, x), \
            "Attention didn't modify input features"

    def test_single_timestep(self):
        """Test with single timestep (T=1)."""
        attn = FullCBAM(in_channels=32, reduction_ratio=16, kernel_size=7)
        x = torch.randn(2, 1, 32, 64, 64)  # T=1
        out = attn(x)

        assert out.shape == (2, 1, 32, 64, 64), \
            f"Failed for single timestep, got shape {out.shape}"

    def test_gradient_flow(self):
        """Test that gradients flow through the attention module."""
        attn = FullCBAM(in_channels=32, reduction_ratio=16, kernel_size=7)
        x = torch.randn(2, 5, 32, 64, 64, requires_grad=True)

        out = attn(x)
        loss = out.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaNs"

    def test_train_eval_consistency(self):
        """Test that attention produces consistent results in train vs eval mode."""
        attn = FullCBAM(in_channels=32, reduction_ratio=16, kernel_size=7)
        x = torch.randn(2, 5, 32, 64, 64)

        attn.train()
        out_train = attn(x)

        attn.eval()
        out_eval = attn(x)

        # No dropout/batchnorm, so outputs should be identical
        assert torch.allclose(out_train, out_eval, atol=1e-6), \
            "Train and eval outputs differ (should be identical)"

    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        attn = FullCBAM(in_channels=32, reduction_ratio=16, kernel_size=7)

        # Process batch of 2
        x_batch = torch.randn(2, 5, 32, 64, 64)
        out_batch = attn(x_batch)

        # Process each sample individually
        out_0 = attn(x_batch[0:1])
        out_1 = attn(x_batch[1:2])

        # Results should match
        assert torch.allclose(out_batch[0], out_0[0], atol=1e-6), \
            "Batch processing differs from individual processing"
        assert torch.allclose(out_batch[1], out_1[0], atol=1e-6), \
            "Batch processing differs from individual processing"

