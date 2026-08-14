"""Tests for the blosc2 training cache.

This path had no coverage until job 38671174 burned a queue wait discovering
that `pip install blosc2` resolves to a version whose wheel does not exist on
Sherlock. A source build then needs numpy>=2.1 and GCC>=10.3, neither of which
the module stack provides.

The version wall itself is not testable off-Sherlock, but the thing it broke is:
`test_blosc2_api_surface` fails immediately on any blosc2 that lacks the API
build_cache.py calls, which is every 2.x release. That plus the round-trip tests
below turn a 40-minute-into-the-job failure into a 5-second local one.
"""
import numpy as np
import pytest
import torch

blosc2 = pytest.importorskip("blosc2")

from lakevision.data.cached_dataset import (  # noqa: E402
    DN_SCALE,
    NODATA_U16,
    QUANTIFICATION,
    CachedLakeDataset,
    normalize_batch,
    worker_init,
)

T, H, W = 4, 8, 8
BANDS = ["B04", "B03", "B02"]


def _cparams():
    return blosc2.CParams(
        codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.BITSHUFFLE], clevel=5
    )


def _write_cache(root, lake_ids, mask_name=None, static_mask=False,
                 boa=0.0, nodata_at=None):
    """Build a miniature cache with the same layout build_cache.py writes."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)

    for lid in lake_ids:
        for bi, band in enumerate(BANDS):
            a = (rng.integers(0, 5000, size=(T, H, W)) * DN_SCALE).astype(np.uint16)
            if nodata_at is not None and bi == 0:
                a[nodata_at] = NODATA_U16
            d = root / band
            d.mkdir(parents=True, exist_ok=True)
            blosc2.asarray(a, urlpath=str(d / f"{lid}.b2nd"), mode="w",
                           chunks=(2, H, W), cparams=_cparams())

        if mask_name:
            shape = (H, W) if static_mask else (T, H, W)
            m = np.zeros(shape, dtype=np.uint8)
            m[..., : W // 2] = 1          # left half is "water"
            d = root / mask_name
            d.mkdir(parents=True, exist_ok=True)
            blosc2.asarray(m, urlpath=str(d / f"{lid}.b2nd"), mode="w",
                           chunks=None if static_mask else (2, H, W),
                           cparams=_cparams())

        d = root / "scalars"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(
            d / f"{lid}.npz",
            p_water=np.linspace(0, 1, T).astype(np.float32),
            eo_cloud_cover=np.full(T, 50.0, dtype=np.float32),
            boa_add_offset=np.full(T, boa, dtype=np.float32),
            time=np.arange(T, dtype=np.int64),
        )
    return root


# --------------------------------------------------------------------------
# the check that would have caught job 38671174's failure locally
# --------------------------------------------------------------------------

def test_blosc2_api_surface():
    """build_cache.py needs blosc2 >= 3.3. Every 2.x lacks CParams."""
    assert hasattr(blosc2, "CParams"), (
        f"blosc2 {blosc2.__version__} has no CParams; build_cache.py requires "
        f">=3.3. Note the ceiling too: >=4.8 has no manylinux_2_17 wheel and so "
        f"cannot install on Sherlock."
    )
    assert hasattr(blosc2, "set_nthreads")
    assert hasattr(blosc2.Codec, "LZ4")
    assert hasattr(blosc2.Filter, "BITSHUFFLE")


def test_worker_init_pins_single_thread():
    """Without this, N DataLoader workers each spawn an N-thread pool.

    blosc2 exposes the current count as the module attribute `nthreads`; there
    is no get_nthreads() to pair with set_nthreads().
    """
    blosc2.set_nthreads(4)
    worker_init(0)
    assert blosc2.nthreads == 1


def test_uint16_roundtrip_is_bit_exact(tmp_path):
    a = np.array([0, 1, 2, 30000, NODATA_U16], dtype=np.uint16).reshape(1, 1, 5)
    p = tmp_path / "rt.b2nd"
    blosc2.asarray(a, urlpath=str(p), mode="w", cparams=_cparams())
    np.testing.assert_array_equal(blosc2.open(str(p))[:], a)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def test_returns_six_tuple_of_uint16(tmp_path):
    """The queue must stay uint16; float32 here is what OOMed the node."""
    _write_cache(tmp_path, ["L1"])
    ds = CachedLakeDataset(tmp_path, lake_ids=["L1"], bands=BANDS,
                           labels_dict={"L1": 2}, seq_len=T)
    out = ds[0]
    assert len(out) == 6, "callers unpack six; see CachedLakeDataset docstring"

    img, area, cloudy, label, lake_id, boa = out
    assert img.dtype == torch.uint16
    assert img.shape == (T, len(BANDS), H, W)
    assert area.shape == (T, 1) and cloudy.shape == (T, 1)
    assert boa.shape == (T,)
    assert int(label) == 2 and lake_id == "L1"
    assert ds.n_refl == len(BANDS)


def test_static_mask_is_broadcast_over_time(tmp_path):
    """lake_boundary is stored once as [H,W]; it must cost no extra I/O."""
    _write_cache(tmp_path, ["L1"], mask_name="lake_boundary", static_mask=True)
    ds = CachedLakeDataset(tmp_path, lake_ids=["L1"], bands=BANDS,
                           mask="lake_boundary", seq_len=T)
    img = ds[0][0]
    assert img.shape == (T, len(BANDS) + 1, H, W)
    assert ds.n_refl == len(BANDS) and ds.n_channels == len(BANDS) + 1
    # torch.equal has no uint16 CPU kernel; compare in a supported dtype.
    m = img[:, -1].to(torch.int32)
    assert torch.equal(m[0], m[-1]), "static mask must be identical across T"


def test_dynamic_mask_keeps_time_axis(tmp_path):
    _write_cache(tmp_path, ["L1"], mask_name="water_mask_ndwi")
    ds = CachedLakeDataset(tmp_path, lake_ids=["L1"], bands=BANDS,
                           mask="water_mask_ndwi", seq_len=T)
    assert ds[0][0].shape == (T, len(BANDS) + 1, H, W)


def test_seq_len_truncates_and_edge_pads(tmp_path):
    _write_cache(tmp_path, ["L1"])
    short = CachedLakeDataset(tmp_path, lake_ids=["L1"], bands=BANDS, seq_len=2)
    assert short[0][0].shape[0] == 2

    long = CachedLakeDataset(tmp_path, lake_ids=["L1"], bands=BANDS, seq_len=T + 3)
    img = long[0][0].to(torch.int32)   # torch.equal lacks a uint16 CPU kernel
    assert img.shape[0] == T + 3
    assert torch.equal(img[T - 1], img[-1]), "pad must repeat the edge frame"


def test_scalar_left_physical_by_default(tmp_path):
    """p_water is already a fraction; min-maxing it erases cross-lake amplitude."""
    _write_cache(tmp_path, ["L1"])
    raw = CachedLakeDataset(tmp_path, lake_ids=["L1"], bands=BANDS,
                            seq_len=T, scalar_var="p_water")[0][1]
    torch.testing.assert_close(raw.squeeze(-1),
                               torch.linspace(0, 1, T))

    scaled = CachedLakeDataset(tmp_path, lake_ids=["L1"], bands=BANDS, seq_len=T,
                               scalar_var="p_water", normalize_scalar=True)[0][1]
    assert scaled.min() == 0.0 and scaled.max() == pytest.approx(1.0, abs=1e-6)


def test_missing_lakes_are_dropped_not_fatal(tmp_path):
    _write_cache(tmp_path, ["L1"])
    ds = CachedLakeDataset(tmp_path, lake_ids=["L1", "ghost"], bands=BANDS, seq_len=T)
    assert ds.lake_ids == ["L1"]


# --------------------------------------------------------------------------
# normalize_batch
# --------------------------------------------------------------------------

def test_normalize_inverts_dn_scale():
    dn = 4200.0
    img = torch.full((1, 1, 3, 2, 2), int(dn * DN_SCALE), dtype=torch.uint16)
    out = normalize_batch(img)
    torch.testing.assert_close(out, torch.full_like(out, dn / QUANTIFICATION))


def test_normalize_applies_boa_offset():
    img = torch.full((1, 2, 3, 2, 2), int(2000 * DN_SCALE), dtype=torch.uint16)
    boa = torch.tensor([[0.0, -1000.0]])
    out = normalize_batch(img, boa)
    assert out[0, 0].mean().item() == pytest.approx(0.2)
    assert out[0, 1].mean().item() == pytest.approx(0.1)


def test_nodata_becomes_zero():
    img = torch.full((1, 1, 1, 2, 2), NODATA_U16, dtype=torch.uint16)
    assert torch.all(normalize_batch(img) == 0.0)


def test_boa_offset_is_not_applied_to_mask_channel():
    """Regression: the mask is an indicator, not a measurement.

    Without n_refl the offset shifts the mask off 0/1, and a *static* boundary
    starts varying across T whenever the processing baseline changes mid-series
    -- a temporal signal the ConvLSTM could learn from that is pure artifact.
    Latent on CW 2018/2019 (baseline 02.12, offset 0 throughout) but silent.
    """
    mask_dn = int(QUANTIFICATION * DN_SCALE)
    img = torch.zeros((1, 2, 4, 2, 2), dtype=torch.uint16)
    img[:, :, :3] = int(2000 * DN_SCALE)
    img[:, :, 3] = mask_dn                       # trailing channel = all water
    boa = torch.tensor([[0.0, -1000.0]])         # baseline changes at t=1

    out = normalize_batch(img, boa, n_refl=3)

    mask_out = out[:, :, 3]
    torch.testing.assert_close(mask_out, torch.ones_like(mask_out))
    assert torch.equal(mask_out[0, 0], mask_out[0, 1]), \
        "static mask must not vary with the per-timestep BOA offset"

    # reflectance channels still get the offset
    assert out[0, 0, :3].mean().item() == pytest.approx(0.2)
    assert out[0, 1, :3].mean().item() == pytest.approx(0.1)


def test_mask_channel_lands_on_zero_one_end_to_end(tmp_path):
    """The DN-domain scaling in the dataset must survive normalize_batch."""
    _write_cache(tmp_path, ["L1"], mask_name="water_mask_ndwi", boa=-1000.0)
    ds = CachedLakeDataset(tmp_path, lake_ids=["L1"], bands=BANDS,
                           mask="water_mask_ndwi", seq_len=T)
    img, _, _, _, _, boa = ds[0]
    out = normalize_batch(img.unsqueeze(0), boa.unsqueeze(0), n_refl=ds.n_refl)

    m = out[0, :, -1]
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}
    assert torch.all(m[..., : W // 2] == 1.0)
    assert torch.all(m[..., W // 2:] == 0.0)


def test_normalize_defaults_to_all_reflectance():
    """n_refl=None must stay correct for the no-mask case."""
    img = torch.full((1, 1, 3, 2, 2), int(2000 * DN_SCALE), dtype=torch.uint16)
    boa = torch.tensor([[-1000.0]])
    torch.testing.assert_close(
        normalize_batch(img, boa), normalize_batch(img, boa, n_refl=3)
    )
