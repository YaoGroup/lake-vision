"""
Tests for lakevision.models.classifier module.

Tests the full LakeDrainageClassifier model that integrates all components:
FrontCNN, CLSTM, attention mechanisms, ScalarLSTM, and ClassHeadMLP.
"""
import pytest
import torch
import torch.nn as nn

from lakevision.models.classifier import LakeDrainageClassifier


class TestLakeDrainageClassifier:
    """
    Tests for LakeDrainageClassifier (full integrated model).
    """

    def test_basic_forward_all_features(self):
        """Test basic forward pass with all features enabled."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            use_cloudyseq=True,
            attention_type='none',
        )

        # Create input tensors
        x = torch.randn(2, 13, 4, 512, 512)  # [B=2, T=13, C=4, H=512, W=512]
        area_seq = torch.randn(2, 13, 1)     # [B=2, T=13, 1]
        cloudy_seq = torch.randn(2, 13, 1)   # [B=2, T=13, 1]

        logits = model(x, area_seq, cloudy_seq)

        # Expected output: [B=2, num_classes=4]
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

        # Check for nans or infs
        assert not torch.isnan(logits).any(), "Output contains NaNs"
        assert not torch.isinf(logits).any(), "Output contains Infs"

    def test_forward_imgseq_only(self):
        """Test with only image sequences enabled."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=False,
            use_cloudyseq=False,
        )

        x = torch.randn(2, 13, 4, 512, 512)
        area_seq = None
        cloudy_seq = None

        logits = model(x, area_seq, cloudy_seq)
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

    def test_forward_areaseq_only(self):
        """Test with only area sequences enabled."""
        model = LakeDrainageClassifier(
            use_imgseq=False,
            use_areaseq=True,
            use_cloudyseq=False,
        )

        x = None
        area_seq = torch.randn(2, 13, 1)
        cloudy_seq = None

        logits = model(x, area_seq, cloudy_seq)
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

    def test_forward_imgseq_and_areaseq(self):
        """Test with image and area sequences (typical configuration)."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            use_cloudyseq=False,
        )

        x = torch.randn(2, 13, 4, 512, 512)
        area_seq = torch.randn(2, 13, 1)
        cloudy_seq = None

        logits = model(x, area_seq, cloudy_seq)
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

    def test_forward_imgseq_and_cloudyseq(self):
        """Test with image and cloudy sequences (valid combination)."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=False,
            use_cloudyseq=True,
        )

        x = torch.randn(2, 13, 4, 512, 512)
        area_seq = None
        cloudy_seq = torch.randn(2, 13, 1)

        logits = model(x, area_seq, cloudy_seq)
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

    def test_forward_areaseq_and_cloudyseq(self):
        """Test with area and cloudy sequences (valid combination)."""
        model = LakeDrainageClassifier(
            use_imgseq=False,
            use_areaseq=True,
            use_cloudyseq=True,
        )

        x = None
        area_seq = torch.randn(2, 13, 1)
        cloudy_seq = torch.randn(2, 13, 1)

        logits = model(x, area_seq, cloudy_seq)
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

    def test_attention_none(self):
        """Test with no attention mechanism."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            attention_type='none',
        )

        x = torch.randn(2, 13, 4, 512, 512)
        area_seq = torch.randn(2, 13, 1)
        cloudy_seq = None

        logits = model(x, area_seq, cloudy_seq)
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

        # Check that attention is Identity
        assert isinstance(model.attention, nn.Identity), "Expected nn.Identity for attention_type='none'"

    def test_attention_spatial(self):
        """Test with spatial attention (SpatialCBAM)."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            attention_type='spatial',
        )

        x = torch.randn(2, 13, 4, 512, 512)
        area_seq = torch.randn(2, 13, 1)
        cloudy_seq = None

        logits = model(x, area_seq, cloudy_seq)
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

        # Check that spatial attention is used
        from lakevision.models.attention import SpatialCBAM
        assert isinstance(model.attention, SpatialCBAM), "Expected SpatialCBAM for attention_type='spatial'"

    def test_attention_full(self):
        """Test with FullCBAM attention (channel + spatial)."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            attention_type='full',
        )

        x = torch.randn(2, 13, 4, 512, 512)
        area_seq = torch.randn(2, 13, 1)
        cloudy_seq = None

        logits = model(x, area_seq, cloudy_seq)
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

        # Check that full attention is used
        from lakevision.models.attention import FullCBAM
        assert isinstance(model.attention, FullCBAM), "Expected FullCBAM for attention_type='full'"

    def test_attention_arch(self):
        """Test with architectural attention (dual pathway with mask)."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            attention_type='arch',
        )

        x = torch.randn(2, 13, 4, 512, 512)
        area_seq = torch.randn(2, 13, 1)
        cloudy_seq = None

        logits = model(x, area_seq, cloudy_seq)
        assert logits.shape == (2, 4), f"Expected output shape (2, 4), but got {logits.shape}"

        # Check that separate mask pathway exists
        assert hasattr(model, 'frontcnn_mask'), "Expected frontcnn_mask for attention_type='arch'"

    def test_invalid_attention_type_raises_error(self):
        """Test that invalid attention type raises error."""
        with pytest.raises(ValueError, match="Invalid attention_type"):
            LakeDrainageClassifier(
                use_imgseq=True,
                use_areaseq=True,
                attention_type='invalid',
            )

    def test_cloudyseq_only_raises_error(self):
        """Test that enabling only cloudyseq raises error."""
        with pytest.raises(ValueError, match="use_cloudyseq cannot be enabled alone"):
            LakeDrainageClassifier(
                use_imgseq=False,
                use_areaseq=False,
                use_cloudyseq=True,
            )

    def test_no_features_raises_error(self):
        """Test that disabling all features raises error."""
        with pytest.raises(ValueError, match="At least one of"):
            LakeDrainageClassifier(
                use_imgseq=False,
                use_areaseq=False,
                use_cloudyseq=False,
            )

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
        )

        x = torch.randn(2, 13, 4, 512, 512, requires_grad=True)
        area_seq = torch.randn(2, 13, 1, requires_grad=True)
        cloudy_seq = None

        logits = model(x, area_seq, cloudy_seq)
        loss = logits.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None, "No gradient for image input"
        assert area_seq.grad is not None, "No gradient for area_seq input"
        assert not torch.isnan(x.grad).any(), "Image gradient contains NaNs"
        assert not torch.isnan(area_seq.grad).any(), "Area gradient contains NaNs"

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
        )

        batch_sizes = [1, 2, 4, 8]
        for B in batch_sizes:
            x = torch.randn(B, 13, 4, 512, 512)
            area_seq = torch.randn(B, 13, 1)
            cloudy_seq = None

            logits = model(x, area_seq, cloudy_seq)
            assert logits.shape == (B, 4), f"Expected output shape ({B}, 4), but got {logits.shape}"

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
        )

        sequence_lengths = [1, 5, 13, 50]
        for T in sequence_lengths:
            x = torch.randn(2, T, 4, 512, 512)
            area_seq = torch.randn(2, T, 1)
            cloudy_seq = None

            logits = model(x, area_seq, cloudy_seq)
            assert logits.shape == (2, 4), f"Expected output shape (2, 4) for T={T}, but got {logits.shape}"

    def test_different_pool_types(self):
        """Test with different pooling types before classification head."""
        pool_types = ['avg', 'max', 'both']

        for pool_type in pool_types:
            model = LakeDrainageClassifier(
                use_imgseq=True,
                use_areaseq=True,
                pool_type=pool_type,
            )

            x = torch.randn(2, 13, 4, 512, 512)
            area_seq = torch.randn(2, 13, 1)
            cloudy_seq = None

            logits = model(x, area_seq, cloudy_seq)
            assert logits.shape == (2, 4), f"Expected output shape (2, 4) for pool_type='{pool_type}', but got {logits.shape}"

    def test_different_num_classes(self):
        """Test with different numbers of output classes."""
        num_classes_list = [2, 3, 4, 5]

        for num_classes in num_classes_list:
            model = LakeDrainageClassifier(
                use_imgseq=True,
                use_areaseq=True,
                num_classes=num_classes,
            )

            x = torch.randn(2, 13, 4, 512, 512)
            area_seq = torch.randn(2, 13, 1)
            cloudy_seq = None

            logits = model(x, area_seq, cloudy_seq)
            assert logits.shape == (2, num_classes), f"Expected output shape (2, {num_classes}), but got {logits.shape}"

    def test_get_feature_dims(self):
        """Test get_feature_dims helper method returns correct dimensions."""
        # Test with imgseq + areaseq
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            use_cloudyseq=False,
            clstm_hidden=32,
            slstm_hidden=16,
            pool_type='avg',
        )

        dims = model.get_feature_dims()

        assert 'img_features' in dims, "Expected 'img_features' in feature dims"
        assert 'area_features' in dims, "Expected 'area_features' in feature dims"
        assert 'total' in dims, "Expected 'total' in feature dims"
        assert dims['img_features'] == 32, f"Expected img_features=32, but got {dims['img_features']}"
        assert dims['area_features'] == 16, f"Expected area_features=16, but got {dims['area_features']}"
        assert dims['total'] == 48, f"Expected total=48, but got {dims['total']}"

    def test_repr(self):
        """Test __repr__ method returns informative string representation."""
        model = LakeDrainageClassifier(
            use_imgseq=True,
            use_areaseq=True,
            use_cloudyseq=False,
            attention_type='spatial',
        )

        repr_str = repr(model)

        # Check that the representation contains key information
        assert 'LakeDrainageClassifier' in repr_str, "Expected class name in repr"
        assert 'imgseq=True' in repr_str, "Expected imgseq flag in repr"
        assert 'areaseq=True' in repr_str, "Expected areaseq flag in repr"
        assert 'cloudyseq=False' in repr_str, "Expected cloudyseq flag in repr"
        assert 'Attention: spatial' in repr_str, "Expected attention type in repr"
        assert 'Feature dims:' in repr_str, "Expected feature dims in repr"
        assert 'Total parameters:' in repr_str, "Expected parameter count in repr"




