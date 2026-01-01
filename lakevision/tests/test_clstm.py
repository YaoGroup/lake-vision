"""
Tests for lakevision.models.clstm module.

Tests convolutional LSTM components:
CellCLSTM (convolutional LSTM cell) and CLSTM (full model).
"""
import pytest
import torch
import torch.nn as nn

from lakevision.models.clstm import (
    CellCLSTM,
    CLSTM,
)

class TestCellCLSTM:
    """
    Tests for CellCLSTM (convolutional LSTM cell).
    """
    def test_basic_forward(self):
        cell = CellCLSTM(input_channels=32, hidden_channels=64, kernel_size=3)

        # create input tensors
        x = torch.randn(2, 32, 64, 64) # [B=2, C_in=32, H=64, W=64]
        h_prev = torch.zeros(2, 64, 64, 64) # [B=2, C_hidden=54, H=64, W=64]
        c_prev = torch.zeros(2, 64, 64, 64) # [B=2, C_hidden=64, H=64, W=64]

        h_new, c_new = cell(x, h_prev, c_prev)

        # Expected output shapes
        assert h_new.shape == (2, 64, 64, 64), f"Expected h shape (2, 64, 64, 64), but got {h_new.shape}"
        assert c_new.shape == (2, 64, 64, 64), f"Expected c shape (2, 64, 64, 64), but got {c_new.shape}"

        # Check for nans or infs
        assert not torch.isnan(h_new).any(), "h_new contains NaNs"
        assert not torch.isinf(h_new).any(), "h_new contains Infs"
        assert not torch.isnan(c_new).any(), "c_new contains NaNs"
        assert not torch.isinf(c_new).any(), "c_new contains Infs"

    def test_hidden_state_changes(self):
        """Test that hidden and cell states change from previous timestep."""
        cell = CellCLSTM(input_channels=32, hidden_channels=64, kernel_size=3)

        x = torch.randn(2, 32, 64, 64)
        h_prev = torch.zeros(2, 64, 64, 64)
        c_prev = torch.zeros(2, 64, 64, 64)

        h_new, c_new = cell(x, h_prev, c_prev)

        # States should change (not stay all zeros)
        assert not torch.allclose(h_new, h_prev), "Hidden state didn't change"
        assert not torch.allclose(c_new, c_prev), "Cell state didn't change"

    def test_spatial_dimensions_preserved(self):
        """Test that spatial dimensions are preserved through the cell."""
        cell = CellCLSTM(input_channels=16, hidden_channels=32, kernel_size=3)

        # Test different spatial sizes
        spatial_sizes = [(32, 32), (64, 64), (128, 128)]

        for H, W in spatial_sizes:
            x = torch.randn(2, 16, H, W)
            h_prev = torch.zeros(2, 32, H, W)
            c_prev = torch.zeros(2, 32, H, W)

            h_new, c_new = cell(x, h_prev, c_prev)

            assert h_new.shape[-2:] == (H, W), f"Spatial dims changed for {H}x{W}"
            assert c_new.shape[-2:] == (H, W), f"Spatial dims changed for {H}x{W}"

    def test_different_kernel_sizes(self):
        """Test with different odd kernel sizes."""
        kernel_sizes = [1, 3, 5, 7]

        for ks in kernel_sizes:
            cell = CellCLSTM(input_channels=16, hidden_channels=32, kernel_size=ks)

            x = torch.randn(2, 16, 32, 32)
            h_prev = torch.zeros(2, 32, 32, 32)
            c_prev = torch.zeros(2, 32, 32, 32)

            h_new, c_new = cell(x, h_prev, c_prev)

            assert h_new.shape == (2, 32, 32, 32), f"Failed for kernel_size={ks}"
            assert c_new.shape == (2, 32, 32, 32), f"Failed for kernel_size={ks}"

    def test_even_kernel_size_raises_error(self):
        """Test that even kernel size raises error."""
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            CellCLSTM(input_channels=16, hidden_channels=32, kernel_size=4)

    def test_different_channel_configs(self):
        """Test with different input/hidden channel configurations."""
        configs = [
            (8, 16),
            (16, 32),
            (32, 64),
            (64, 128),
        ]

        for in_ch, hid_ch in configs:
            cell = CellCLSTM(input_channels=in_ch, hidden_channels=hid_ch)

            x = torch.randn(2, in_ch, 32, 32)
            h_prev = torch.zeros(2, hid_ch, 32, 32)
            c_prev = torch.zeros(2, hid_ch, 32, 32)

            h_new, c_new = cell(x, h_prev, c_prev)

            assert h_new.shape == (2, hid_ch, 32, 32), \
                f"Failed for in_ch={in_ch}, hid_ch={hid_ch}"
            
    def test_gate_activations_in_range(self):
        """Test that gates produce values in expected ranges."""
        cell = CellCLSTM(input_channels=16, hidden_channels=32, kernel_size=3)

        x = torch.randn(1, 16, 32, 32)
        h_prev = torch.zeros(1, 32, 32, 32)
        c_prev = torch.zeros(1, 32, 32, 32)

        h_new, c_new = cell(x, h_prev, c_prev)

        # Hidden state goes through tanh(o * tanh(c)), so should be in [-1, 1]
        assert h_new.min() >= -1.0, "Hidden state values below -1"
        assert h_new.max() <= 1.0, "Hidden state values above 1"

    def test_sequential_timesteps(self):
        """Test processing multiple sequential timesteps."""
        cell = CellCLSTM(input_channels=16, hidden_channels=32, kernel_size=3)

        # Process sequence manually through cell
        B, T, C, H, W = 2, 5, 16, 32, 32
        x_seq = torch.randn(B, T, C, H, W)

        h = torch.zeros(B, 32, H, W)
        c = torch.zeros(B, 32, H, W)

        for t in range(T):
            h, c = cell(x_seq[:, t], h, c)

        # Final hidden and cell states should be valid
        assert h.shape == (B, 32, H, W), "Final hidden state shape incorrect"
        assert c.shape == (B, 32, H, W), "Final cell state shape incorrect"
        assert not torch.isnan(h).any(), "Final hidden state contains NaNs"
        assert not torch.isnan(c).any(), "Final cell state contains NaNs"


class TestCLSTM:
    """
    Tests for CLSTM (full convolutional CLSTM module).
    """
    def test_basic_forward_return_sequence(self):
        """Test basic forward pass with return_sequence=True."""
        model = CLSTM(input_channels=32, hidden_channels=64, return_sequence=True)
        x = torch.randn(2, 13, 32, 64, 64) # [B=2, T=10, C=32, H=64, W=64]
        out = model(x)

        # expected output shape: [B=2, T=13, C_hidden=64, H=64, W=64]
        assert out.shape == (2, 13, 64, 64, 64), f"Expected output shape (2, 13, 64, 64, 64), but got {out.shape}"

        # check for nans or infs
        assert not torch.isnan(out).any(), "Output contains NaNs"
        assert not torch.isinf(out).any(), "Output contains Infs"

    def test_basic_forward_return_final(self):
        """Test basic forward pass with return_sequence=False."""
        model = CLSTM(input_channels=32, hidden_channels=64, return_sequence=False)
        x = torch.randn(2, 10, 32, 64, 64)  # [B=2, T=10, C=32, H=64, W=64]
        out = model(x)

        # Expected output shape: [B=2, C_hidden=64, H=64, W=64]
        assert out.shape == (2, 64, 64, 64), \
            f"Expected output shape (2, 64, 64, 64), but got {out.shape}"

        # Check for nans or infs
        assert not torch.isnan(out).any(), "Output contains NaNs"
        assert not torch.isinf(out).any(), "Output contains Infs"

    def test_return_sequence_consistency(self):
        """Test that final timestep of sequence matches return_sequence=False output."""
        x = torch.randn(2, 10, 32, 64, 64)

        # Model returning full sequence
        model_seq = CLSTM(input_channels=32, hidden_channels=64, return_sequence=True)
        out_seq = model_seq(x)

        # Model returning only final timestep
        model_final = CLSTM(input_channels=32, hidden_channels=64, return_sequence=False)
        # Copy weights to ensure same computation
        model_final.load_state_dict(model_seq.state_dict())
        out_final = model_final(x)

        # Final timestep of sequence should match final-only output
        assert torch.allclose(out_seq[:, -1], out_final, atol=1e-6), \
            "Final timestep doesn't match return_sequence=False output"
        
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        model = CLSTM(input_channels=32, hidden_channels=64, return_sequence=True)

        for T in [1, 5, 10, 50]:
            x = torch.randn(2, T, 32, 64, 64)
            out = model(x)
            assert out.shape == (2, T, 64, 64, 64), \
                f"Failed for sequence length T={T}"
            
    def test_different_spatial_sizes(self):
        """Test with different spatial dimensions."""
        model = CLSTM(input_channels=32, hidden_channels=64, return_sequence=True)

        spatial_sizes = [(32, 32), (64, 64), (128, 128)]
        for H, W in spatial_sizes:
            x = torch.randn(2, 5, 32, H, W)
            out = model(x)
            assert out.shape == (2, 5, 64, H, W), \
                f"Failed for spatial size {H}x{W}"
            
    def test_different_hidden_channels(self):
        """Test with different hidden channel sizes."""
        hidden_channels_list = [16, 32, 64, 128]

        for hid_ch in hidden_channels_list:
            model = CLSTM(input_channels=32, hidden_channels=hid_ch, return_sequence=True)
            x = torch.randn(2, 5, 32, 64, 64)
            out = model(x)
            assert out.shape == (2, 5, hid_ch, 64, 64), \
                f"Failed for hidden_channels={hid_ch}"
            
    def test_different_kernel_sizes(self):
        """Test with different kernel sizes."""
        kernel_sizes = [1, 3, 5]

        for ks in kernel_sizes:
            model = CLSTM(input_channels=32, hidden_channels=64, kernel_size=ks, return_sequence=True)
            x = torch.randn(2, 5, 32, 64, 64)
            out = model(x)
            assert out.shape == (2, 5, 64, 64, 64), \
                f"Failed for kernel_size={ks}"
            
    def test_init_hidden(self):
        """Test initialization of hidden states."""
        model = CLSTM(input_channels=32, hidden_channels=64)

        B, H, W = 2, 64, 64
        device = torch.device('cpu')

        h, c = model.init_hidden(B, H, W, device)

        # Check shapes
        assert h.shape == (B, 64, H, W), "Hidden state shape incorrect"
        assert c.shape == (B, 64, H, W), "Cell state shape incorrect"

        # Check they're initialized to zeros
        assert torch.allclose(h, torch.zeros_like(h)), "Hidden state not initialized to zeros"
        assert torch.allclose(c, torch.zeros_like(c)), "Cell state not initialized to zeros"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = CLSTM(input_channels=32, hidden_channels=64, return_sequence=False)
        x = torch.randn(2, 5, 32, 64, 64, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaNs"

    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        model = CLSTM(input_channels=32, hidden_channels=64, return_sequence=False)

        # Process batch of 2
        x_batch = torch.randn(2, 5, 32, 64, 64)
        out_batch = model(x_batch)

        # Process each sample individually
        out_0 = model(x_batch[0:1])
        out_1 = model(x_batch[1:2])

        # Results should match
        assert torch.allclose(out_batch[0], out_0[0], atol=1e-6), \
            "Batch processing differs from individual processing"
        assert torch.allclose(out_batch[1], out_1[0], atol=1e-6), \
            "Batch processing differs from individual processing"
        
    def test_single_timestep(self):
        """Test processing with single timestep (T=1)."""
        model = CLSTM(input_channels=32, hidden_channels=64, return_sequence=True)
        x = torch.randn(2, 1, 32, 64, 64)  # T=1
        out = model(x)

        assert out.shape == (2, 1, 64, 64, 64), \
            f"Failed for single timestep, got shape {out.shape}"
        
    def test_train_eval_consistency(self):
        """Test that model produces consistent results in train vs eval mode (no dropout/batchnorm)."""
        model = CLSTM(input_channels=32, hidden_channels=64, return_sequence=False)
        x = torch.randn(2, 5, 32, 64, 64)

        model.train()
        out_train = model(x)

        model.eval()
        out_eval = model(x)

        # Since CLSTM has no dropout or batchnorm, outputs should be identical
        assert torch.allclose(out_train, out_eval, atol=1e-6), \
            "Train and eval outputs differ (should be identical for CLSTM)"