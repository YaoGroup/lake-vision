"""
Tests for the ESSD SDR deposit (stacks_v2) schema path in LakeDataset.

The deposit (DOI 10.25740/sf350xp4038) stores band names in a 'band_name'
char array beside an integer 'band' index, and a 'p_water' series that is
NaN on unusable (cloudy) days. These fixtures mirror that schema — verified
against the deposit files in 2026-08 — so CI covers the real data format,
including all six stored bands, without shipping real data.
"""
import netCDF4
import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

from lakevision.data.datasets import LakeDataset, _ffill_bfill_1d

# Deposit band order: B04,B03,B02,B08,B11,B12 -> red,green,blue,nir,swir16,swir22
SDR_BAND_NAMES = ["B04", "B03", "B02", "B08", "B11", "B12"]
T, H, W = 12, 16, 16
SEQ_LEN = 11  # odd, < T so the peak-centered window is exercised


def write_sdr_stack(fp, p_water, seed=0):
    """Write a miniature stacks_v2-style file (integer band + band_name chars)."""
    rng = np.random.default_rng(seed)
    reflectance = rng.uniform(0, 11000, (T, 6, H, W)).astype(np.float32)
    with netCDF4.Dataset(fp, "w") as nc:
        nc.createDimension("time", T)
        nc.createDimension("band", 6)
        nc.createDimension("y", H)
        nc.createDimension("x", W)
        nc.createDimension("string3", 3)
        v = nc.createVariable("reflectance", "f4", ("time", "band", "y", "x"))
        v[:] = reflectance
        v = nc.createVariable("band", "i4", ("band",))
        v[:] = np.arange(6)
        v = nc.createVariable("band_name", "S1", ("band", "string3"))
        v.set_auto_chartostring(False)
        v[:] = np.array(SDR_BAND_NAMES, dtype="S3").view("S1").reshape(6, 3)
        if p_water is not None:
            v = nc.createVariable("p_water", "f4", ("time",))
            v[:] = p_water
        v = nc.createVariable("lake_boundary", "u1", ("y", "x"))
        v[:] = np.ones((H, W), dtype=np.uint8)
    return reflectance


def partial_nan_p_water():
    """Deposit-like p_water: mostly NaN, a few observed fractions."""
    pw = np.full(T, np.nan, dtype=np.float32)
    pw[1] = 0.2
    pw[4] = 0.9  # peak -> window center
    pw[7] = 0.4
    return pw


@pytest.fixture
def sdr_file(tmp_path):
    fp = tmp_path / "CW2019_0001.nc"
    reflectance = write_sdr_stack(fp, partial_nan_p_water())
    return fp, reflectance


class TestFfillBfill:
    def test_interior_leading_trailing(self):
        a = np.array([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=np.float32)
        out = _ffill_bfill_1d(a)
        assert np.allclose(out, [1.0, 1.0, 1.0, 3.0, 3.0])

    def test_all_nan_stays_all_nan(self):
        out = _ffill_bfill_1d(np.full(4, np.nan, dtype=np.float32))
        assert np.isnan(out).all()


class TestSDRSchema:
    def test_band_name_autodetect_rgb(self, sdr_file):
        fp, _ = sdr_file
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False)
        img_seq, area_seq, cloudy_seq, label, lake_id = ds[0]
        assert img_seq.shape == (SEQ_LEN, 3, H, W)
        assert lake_id == "CW2019_0001"  # filename stem (deposit has no lake_id attr)
        assert label.item() == -1

    def test_all_six_deposit_bands(self, sdr_file):
        fp, reflectance = sdr_file
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False,
                         use_nir=True, use_swir16=True, use_swir22=True)
        img_seq, _, _, _, _ = ds[0]
        assert img_seq.shape == (SEQ_LEN, 6, H, W)
        # Channel order is [red,green,blue,nir,swir16,swir22]; red is deposit
        # band 0 (B04). Window: peak at t=4, half=5 -> frames 0..10.
        expected_red = np.clip(reflectance[:SEQ_LEN, 0] / 10000.0, 0.0, 1.0)
        assert torch.allclose(img_seq[:, 0], torch.tensor(expected_red), atol=1e-6)
        expected_swir22 = np.clip(reflectance[:SEQ_LEN, 5] / 10000.0, 0.0, 1.0)
        assert torch.allclose(img_seq[:, 5], torch.tensor(expected_swir22), atol=1e-6)

    def test_p_water_is_filled_and_minmaxed(self, sdr_file):
        fp, _ = sdr_file
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False)
        _, area_seq, _, _, _ = ds[0]
        assert torch.isfinite(area_seq).all()
        assert area_seq.min().item() == pytest.approx(0.0, abs=1e-6)
        assert area_seq.max().item() == pytest.approx(1.0, abs=1e-4)

    def test_all_nan_p_water_raises(self, tmp_path):
        fp = tmp_path / "CW2019_0002.nc"
        write_sdr_stack(fp, np.full(T, np.nan, dtype=np.float32))
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False)
        with pytest.raises(ValueError, match="all-NaN"):
            ds[0]

    def test_requesting_mask_raises_helpfully(self, sdr_file):
        fp, _ = sdr_file
        ds = LakeDataset(fp, seq_len=SEQ_LEN)  # use_mask defaults to True
        with pytest.raises(ValueError, match="mask"):
            ds[0]

    def test_dataloader_end_to_end(self, tmp_path):
        for i in range(2):
            write_sdr_stack(tmp_path / f"CW2019_{i:04d}.nc",
                            partial_nan_p_water(), seed=i)
        ds = LakeDataset(tmp_path, seq_len=SEQ_LEN, use_mask=False)
        img, area, cloudy, labels, ids = next(iter(DataLoader(ds, batch_size=2)))
        assert img.shape == (2, SEQ_LEN, 3, H, W)
        assert area.shape == (2, SEQ_LEN, 1)
        assert torch.isfinite(img).all() and torch.isfinite(area).all()


class TestCompositeNaNGuard:
    def _write_composite(self, fp, water_area):
        imagery = np.random.rand(T, 4, H, W).astype(np.float32)
        ds = xr.Dataset(
            {
                "imagery": (["time", "channel", "y", "x"], imagery),
                "water_area": (["time"], water_area),
            },
            coords={"time": np.arange(T),
                    "channel": ["red", "green", "blue", "mask"]},
            attrs={"lake_id": "TEST_NAN"},
        )
        ds.to_netcdf(fp)
        ds.close()

    def test_nan_water_area_raises(self, tmp_path):
        fp = tmp_path / "bad.nc"
        wa = np.random.rand(T)
        wa[3] = np.nan
        self._write_composite(fp, wa)
        ds = LakeDataset(fp, seq_len=SEQ_LEN)
        with pytest.raises(ValueError, match="water_area"):
            ds[0]

    def test_clean_water_area_ok(self, tmp_path):
        fp = tmp_path / "good.nc"
        self._write_composite(fp, np.random.rand(T))
        ds = LakeDataset(fp, seq_len=SEQ_LEN)
        img_seq, area_seq, _, _, _ = ds[0]
        assert img_seq.shape == (SEQ_LEN, 4, H, W)
        assert torch.isfinite(area_seq).all()
