"""
Regression tests for the mask-channel wiring and channel-count enforcement.

These promote the two pre-CV audit probes (2026-08-14, §8.1/§8.2) into tests:

  A1 — the trailing mask channel was consumed ONLY by attention_type='arch'
       and silently discarded under 'none'/'spatial'/'full'. That is the ESSD
       baseline behavior and must stay reproducible (default flags), but
       mask_as_channel=True must make the mask a live input everywhere.
  B1 — a model given MORE channels than its band flags accounted for silently
       sliced off the extras, so a "6-band" run could quietly train on 3 bands.
       With expect_mask_channel set, the count is enforced exactly.
"""
import pytest
import torch

from lakevision.data.cached_dataset import derive_band_flags
from lakevision.models.classifier import LakeDrainageClassifier


def tiny_model(**overrides):
    """Small-but-spatial model: 64px input, 4 conv layers -> CLSTM sees 4x4."""
    kwargs = dict(
        use_imgseq=True,
        use_areaseq=False,
        use_cloudyseq=False,
        seq_len=4,
        input_H=64,
        input_W=64,
        frontcnn_base_channels=4,
        clstm_hidden=8,
        classhead_hidden=8,
        num_classes=5,
    )
    kwargs.update(overrides)
    model = LakeDrainageClassifier(**kwargs)
    model.eval()
    return model


def rgb_plus_mask(seed=0):
    """[B=2, T=4, C=4, 64, 64] input whose trailing channel is a 0/1 mask."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(2, 4, 4, 64, 64, generator=g)
    x[:, :, 3] = (x[:, :, 3] > 0.5).float()
    return x


def logit_delta_when_mask_inverted(model, x):
    x_flip = x.clone()
    x_flip[:, :, 3] = 1.0 - x_flip[:, :, 3]
    with torch.no_grad():
        return (model(x, None, None) - model(x_flip, None, None)).abs().max().item()


class TestMaskAsChannel:

    @pytest.mark.parametrize("attention", ["none", "spatial", "full"])
    def test_default_discards_mask(self, attention):
        """ESSD baseline semantics: without mask_as_channel, inverting the mask
        changes nothing outside 'arch'. Locks the behavior the published runs had."""
        model = tiny_model(attention_type=attention)
        assert logit_delta_when_mask_inverted(model, rgb_plus_mask()) == 0.0

    @pytest.mark.parametrize("attention", ["none", "spatial", "full"])
    def test_mask_as_channel_is_live(self, attention):
        """With mask_as_channel=True the mask must reach the logits."""
        torch.manual_seed(0)
        model = tiny_model(attention_type=attention, mask_as_channel=True)
        assert logit_delta_when_mask_inverted(model, rgb_plus_mask()) > 0.0

    def test_arch_rejects_mask_as_channel(self):
        with pytest.raises(ValueError, match="arch"):
            tiny_model(attention_type='arch', mask_as_channel=True)

    def test_mask_as_channel_requires_mask(self):
        with pytest.raises(ValueError, match="mask channel"):
            tiny_model(mask_as_channel=True, expect_mask_channel=False)

    def test_arch_requires_mask(self):
        with pytest.raises(ValueError, match="mask channel"):
            tiny_model(attention_type='arch', expect_mask_channel=False)

    def test_mask_as_channel_grows_frontcnn_input(self):
        assert tiny_model(mask_as_channel=True).frontcnn.conv_block[0].in_channels == 4
        assert tiny_model().frontcnn.conv_block[0].in_channels == 3


class TestChannelCountEnforcement:

    def test_extra_channels_raise_when_no_mask_expected(self):
        """B1: RGB model + 4-channel input used to silently drop the 4th."""
        model = tiny_model(expect_mask_channel=False)
        with pytest.raises(RuntimeError, match="4 channels"):
            model(rgb_plus_mask(), None, None)

    def test_missing_mask_raises_when_mask_expected(self):
        model = tiny_model(expect_mask_channel=True)
        with pytest.raises(RuntimeError, match="3 channels"):
            model(rgb_plus_mask()[:, :, :3], None, None)

    def test_exact_count_passes(self):
        model = tiny_model(expect_mask_channel=True)
        logits = model(rgb_plus_mask(), None, None)
        assert logits.shape == (2, 5)
        model = tiny_model(expect_mask_channel=False)
        assert model(rgb_plus_mask()[:, :, :3], None, None).shape == (2, 5)

    def test_unset_stays_lenient(self):
        """Legacy default: no declaration, no check — extra channel is sliced
        off exactly as the ESSD runs did."""
        model = tiny_model()
        assert model(rgb_plus_mask(), None, None).shape == (2, 5)


class TestDeriveBandFlags:

    def test_rgb_only(self):
        assert derive_band_flags(["B04", "B03", "B02"]) == {
            "use_nir": False, "use_swir16": False, "use_swir22": False}

    def test_all_extras(self):
        flags = derive_band_flags(["B04", "B03", "B02", "B08", "B11", "B12"])
        assert flags == {"use_nir": True, "use_swir16": True, "use_swir22": True}

    def test_nir_only(self):
        assert derive_band_flags(["B04", "B03", "B02", "B08"])["use_nir"] is True

    def test_unknown_band_raises(self):
        with pytest.raises(ValueError, match="Unknown band"):
            derive_band_flags(["B04", "B03", "B02", "B99"])

    def test_missing_rgb_raises(self):
        with pytest.raises(ValueError, match="RGB"):
            derive_band_flags(["B04", "B03"])
