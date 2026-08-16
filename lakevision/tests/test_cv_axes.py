"""
Tests for the CV grid's model axes: chunked FrontCNN, normalization, temporal
readout, forget-gate bias, and spatial pooling.

The load-bearing claim is that frontcnn_chunk_size is a MEMORY knob, not a
model change — chunking must not alter the computation (beyond float
non-associativity) for norm in {none, group}. Everything else here locks in
that the ESSD defaults still reproduce the pre-existing behavior, so the
published tags stay bit-for-bit valid while the new axes are opt-in.
"""
import pytest
import torch

from lakevision.models.blocks import FrontCNN
from lakevision.models.classifier import LakeDrainageClassifier
from lakevision.models.clstm import CLSTM

# Tolerance for "same computation, different kernel/reduction order". Chunking
# changes cuDNN's algorithm choice, which perturbs results at ~1e-6 in float32.
FLOAT_NOISE = 1e-4


def tiny_model(**overrides):
    kwargs = dict(
        use_imgseq=True, use_areaseq=False, use_cloudyseq=False,
        seq_len=6, input_H=64, input_W=64,
        frontcnn_base_channels=4, clstm_hidden=8, classhead_hidden=8,
        num_classes=5,
    )
    kwargs.update(overrides)
    return LakeDrainageClassifier(**kwargs).eval()


class TestChunkedFrontCNN:

    @pytest.mark.parametrize("norm", ["none", "group"])
    @pytest.mark.parametrize("chunk", [18, 6, 4, 1])
    def test_chunking_is_numerically_equivalent(self, norm, chunk):
        """Chunking slices the B*T axis; every op is per-image, so the result
        must match the unchunked pass to float noise."""
        torch.manual_seed(0)
        x = torch.randn(3, 6, 4, 64, 64)          # B*T = 18
        m = FrontCNN(in_channels=4, base_channels=4, num_layers=3, norm=norm).eval()
        with torch.no_grad():
            ref = m(x)
            m.chunk_size = chunk
            got = m(x)
        assert got.shape == ref.shape
        assert (got - ref).abs().max().item() < FLOAT_NOISE

    def test_chunking_preserves_output_shape_when_not_divisible(self):
        """B*T=18 with chunk=5 leaves a ragged final slice of 3."""
        x = torch.randn(3, 6, 4, 64, 64)
        m = FrontCNN(in_channels=4, base_channels=4, num_layers=3, chunk_size=5).eval()
        with torch.no_grad():
            assert m(x).shape == (3, 6, 16, 8, 8)

    def test_chunk_larger_than_batch_is_a_noop(self):
        x = torch.randn(2, 4, 4, 64, 64)
        m = FrontCNN(in_channels=4, base_channels=4, num_layers=3).eval()
        with torch.no_grad():
            ref = m(x)
            m.chunk_size = 10_000
            assert torch.equal(m(x), ref)

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            FrontCNN(in_channels=4, chunk_size=0)

    def test_gradients_flow_through_every_chunk(self):
        """A bug that dropped a chunk would leave part of the batch gradient-free."""
        x = torch.randn(3, 6, 4, 64, 64, requires_grad=True)
        m = FrontCNN(in_channels=4, base_channels=4, num_layers=3, chunk_size=4)
        m(x).sum().backward()
        per_sample = x.grad.abs().flatten(1).sum(1)
        assert (per_sample > 0).all(), f"some samples got no gradient: {per_sample}"


class TestNorm:

    @pytest.mark.parametrize("norm,expected", [("none", 0), ("group", 3), ("batch", 3)])
    def test_norm_layers_are_inserted(self, norm, expected):
        m = FrontCNN(in_channels=4, base_channels=4, num_layers=3, norm=norm)
        n = sum(isinstance(l, (torch.nn.GroupNorm, torch.nn.BatchNorm2d))
                for l in m.conv_block)
        assert n == expected

    def test_default_is_no_norm(self):
        """ESSD reproducibility: the published runs had no normalization."""
        m = FrontCNN(in_channels=4)
        assert not any(isinstance(l, (torch.nn.GroupNorm, torch.nn.BatchNorm2d))
                       for l in m.conv_block)

    def test_groupnorm_groups_clamped_to_channels(self):
        """base_channels=4 with norm_groups=8 must not construct GroupNorm(8, 4)."""
        m = FrontCNN(in_channels=4, base_channels=4, num_layers=1,
                     norm='group', norm_groups=8)
        gn = [l for l in m.conv_block if isinstance(l, torch.nn.GroupNorm)][0]
        assert gn.num_groups == 4

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError, match="norm must be"):
            FrontCNN(in_channels=4, norm='layer')


class TestTemporalReadout:

    @pytest.mark.parametrize("readout", ["last", "mean", "max"])
    def test_forward_shape(self, readout):
        m = tiny_model(temporal_readout=readout)
        with torch.no_grad():
            assert m(torch.randn(2, 6, 3, 64, 64), None, None).shape == (2, 5)

    def test_last_skips_stacking_the_hidden_sequence(self):
        """'last' only needs the final state, so return_sequence must be off —
        otherwise 152 of 153 hidden states are stacked purely to be discarded."""
        assert tiny_model(temporal_readout='last').clstm.return_sequence is False
        assert tiny_model(temporal_readout='mean').clstm.return_sequence is True

    def test_readouts_differ(self):
        """If mean/max collapsed to the same thing as last, the axis is inert."""
        torch.manual_seed(0)
        x = torch.randn(2, 6, 3, 64, 64)
        outs = {}
        for r in ["last", "mean", "max"]:
            torch.manual_seed(0)                  # identical weights
            with torch.no_grad():
                outs[r] = tiny_model(temporal_readout=r)(x, None, None)
        assert not torch.allclose(outs['last'], outs['mean'], atol=1e-6)
        assert not torch.allclose(outs['mean'], outs['max'], atol=1e-6)

    def test_invalid_readout_raises(self):
        with pytest.raises(ValueError, match="temporal_readout"):
            tiny_model(temporal_readout='attention')


class TestForgetBias:

    def test_default_leaves_conv_init_untouched(self):
        """ESSD reproducibility. Conv2d's default bias init is uniform, not zeros,
        so the check is that forget_bias=0.0 changes *nothing* — not that the
        bias equals zero."""
        torch.manual_seed(0)
        default = CLSTM(input_channels=4, hidden_channels=8)
        torch.manual_seed(0)
        explicit_zero = CLSTM(input_channels=4, hidden_channels=8, forget_bias=0.0)
        assert torch.equal(default.cell.conv.bias, explicit_zero.cell.conv.bias)

    def test_forget_slice_only(self):
        """Bias must land on the f gate (second quarter), leaving i/o/g alone."""
        c = CLSTM(input_channels=4, hidden_channels=8, forget_bias=1.0)
        b = c.cell.conv.bias
        assert torch.allclose(b[8:16], torch.ones(8))
        assert not torch.allclose(b[0:8], torch.ones(8))
        assert not torch.allclose(b[16:24], torch.ones(8))
        assert not torch.allclose(b[24:32], torch.ones(8))


class TestPoolType:

    def test_both_doubles_readout_for_1pct_params(self):
        avg = tiny_model(pool_type='avg')
        both = tiny_model(pool_type='both')
        n_avg = sum(p.numel() for p in avg.parameters())
        n_both = sum(p.numel() for p in both.parameters())
        assert n_both > n_avg
        assert n_both / n_avg < 1.10, "pool_type='both' should be nearly free"
        with torch.no_grad():
            assert both(torch.randn(2, 6, 3, 64, 64), None, None).shape == (2, 5)
