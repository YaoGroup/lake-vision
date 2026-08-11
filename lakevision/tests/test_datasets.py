"""
Tests for LakeDataset class.
"""
import pytest
import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from pathlib import Path

from lakevision.data.datasets import LakeDataset

# Path to real sample data (if available)
SAMPLE_DATA_PATH = Path(__file__).parent.parent.parent / "datasets" / "processed" / "CW2019_1579.nc"


@pytest.fixture
def sample_nc_file(tmp_path):
    """Create a minimal synthetic NetCDF file for testing."""
    T, C, H, W = 30, 4, 64, 64
    
    # Create synthetic data
    imagery = np.random.rand(T, C, H, W).astype(np.float32)
    water_area = np.zeros(T)
    water_area[15] = 1.0  # Peak at timestep 15
    water_area[10:20] = np.linspace(0, 1, 10)
    
    ds = xr.Dataset(
        {
            "imagery": (["time", "channel", "y", "x"], imagery),
            "water_area": (["time"], water_area),
        },
        coords={
            "time": np.arange(T),
            "channel": ["red", "green", "blue", "mask"],
        },
        attrs={"lake_id": "TEST_001"},
    )
    
    fp = tmp_path / "test_lake.nc"
    ds.to_netcdf(fp)
    ds.close()
    return fp


@pytest.fixture
def sample_nc_dir(tmp_path):
    """Create a directory with multiple synthetic NetCDF files."""
    for i in range(3):
        T, C, H, W = 30, 4, 64, 64
        imagery = np.random.rand(T, C, H, W).astype(np.float32)
        water_area = np.random.rand(T)
        
        ds = xr.Dataset(
            {
                "imagery": (["time", "channel", "y", "x"], imagery),
                "water_area": (["time"], water_area),
            },
            coords={"channel": ["red", "green", "blue", "mask"]},
            attrs={"lake_id": f"TEST_{i:03d}"},
        )
        ds.to_netcdf(tmp_path / f"lake_{i}.nc")
        ds.close()
    return tmp_path


class TestLakeDatasetSynthetic:
    """Tests using synthetic data."""

    def test_load_single_file(self, sample_nc_file):
        """Test loading a single NetCDF file."""
        dataset = LakeDataset(sample_nc_file, seq_len=21)
        assert len(dataset) == 1

    def test_load_directory(self, sample_nc_dir):
        """Test loading from a directory of NetCDF files."""
        dataset = LakeDataset(sample_nc_dir, seq_len=21)
        assert len(dataset) == 3

    def test_load_list_of_paths(self, sample_nc_dir):
        """Test loading from a list of paths."""
        paths = list(sample_nc_dir.glob("*.nc"))[:2]
        dataset = LakeDataset(paths, seq_len=21)
        assert len(dataset) == 2

    def test_getitem_returns_correct_types(self, sample_nc_file):
        """Test that __getitem__ returns correct types."""
        dataset = LakeDataset(sample_nc_file, seq_len=21)
        img_seq, area_seq, cloudy_seq, label, lake_id = dataset[0]
        
        assert isinstance(img_seq, torch.Tensor)
        assert isinstance(area_seq, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(lake_id, str)

    def test_getitem_returns_correct_shapes(self, sample_nc_file):
        """Test that __getitem__ returns correct shapes."""
        seq_len = 21
        dataset = LakeDataset(sample_nc_file, seq_len=seq_len)
        img_seq, area_seq, cloudy_seq, label, lake_id = dataset[0]
        
        assert img_seq.shape[0] == seq_len  # [seq_len, C, H, W]
        assert img_seq.shape[1] == 4  # RGBM channels
        assert area_seq.shape == (seq_len, 1)
        assert label.shape == ()

    def test_getitem_returns_correct_dtypes(self, sample_nc_file):
        """Test that __getitem__ returns correct dtypes."""
        dataset = LakeDataset(sample_nc_file, seq_len=21)
        img_seq, area_seq, cloudy_seq, label, lake_id = dataset[0]
        
        assert img_seq.dtype == torch.float32
        assert area_seq.dtype == torch.float32
        assert label.dtype == torch.long

    def test_default_label(self, sample_nc_file):
        """Test that default label is applied."""
        dataset = LakeDataset(sample_nc_file, seq_len=21, label=2)
        _, _, _, label, _ = dataset[0]
        assert label.item() == 2

    def test_no_label_returns_minus_one(self, sample_nc_file):
        """Test that missing label returns -1."""
        dataset = LakeDataset(sample_nc_file, seq_len=21)
        _, _, _, label, _ = dataset[0]
        assert label.item() == -1

    def test_labels_file(self, sample_nc_file, tmp_path):
        """Test loading labels from CSV file."""
        # Create labels file
        labels_fp = tmp_path / "labels.csv"
        labels_fp.write_text("lake_id,label\nTEST_001,3\n")
        
        dataset = LakeDataset(sample_nc_file, seq_len=21, labels_file=labels_fp)
        _, _, _, label, _ = dataset[0]
        assert label.item() == 3

    def test_sequence_centered_on_max_area(self, sample_nc_file):
        """Test that sequence is centered on max water area."""
        dataset = LakeDataset(sample_nc_file, seq_len=21)
        _, area_seq, _, _, _ = dataset[0]
        
        # Max should be near the center of the sequence
        max_idx = area_seq.squeeze().argmax().item()
        center = len(area_seq) // 2
        assert abs(max_idx - center) <= 1

    def test_handles_nans_in_imagery(self, tmp_path):
        """Test that NaNs in imagery are handled."""
        T, C, H, W = 30, 4, 64, 64
        imagery = np.random.rand(T, C, H, W).astype(np.float32)
        imagery[5, :, :10, :10] = np.nan  # Add some NaNs
        water_area = np.random.rand(T)
        
        ds = xr.Dataset(
            {
                "imagery": (["time", "channel", "y", "x"], imagery),
                "water_area": (["time"], water_area),
            },
            coords={"channel": ["red", "green", "blue", "mask"]},
            attrs={"lake_id": "TEST_NAN"},
        )
        fp = tmp_path / "nan_lake.nc"
        ds.to_netcdf(fp)
        ds.close()
        
        dataset = LakeDataset(fp, seq_len=21)
        img_seq, _, _, _, _ = dataset[0]
        
        assert not torch.isnan(img_seq).any()

    def test_empty_directory_raises(self, tmp_path):
        """Test that empty directory raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No .nc files found"):
            LakeDataset(empty_dir)

    def test_nonexistent_path_raises(self):
        """Test that nonexistent path raises ValueError."""
        with pytest.raises(ValueError, match="Path does not exist"):
            LakeDataset("/nonexistent/path.nc")


@pytest.mark.skipif(not SAMPLE_DATA_PATH.exists(), reason="Sample data not available")
class TestLakeDatasetRealData:
    """Tests using real sample data."""

    def test_load_real_sample(self):
        """Test loading real sample data."""
        dataset = LakeDataset(SAMPLE_DATA_PATH, seq_len=21)
        assert len(dataset) == 1

    def test_real_sample_shapes(self):
        """Test shapes from real sample data."""
        dataset = LakeDataset(SAMPLE_DATA_PATH, seq_len=21)
        img_seq, area_seq, cloudy_seq, label, lake_id = dataset[0]
        
        assert img_seq.shape == (21, 4, 512, 512)
        assert area_seq.shape == (21, 1)
        assert lake_id == "CW2019_1579"

    def test_real_sample_with_dataloader(self):
        """Test that real sample works with PyTorch DataLoader."""        
        dataset = LakeDataset(SAMPLE_DATA_PATH, seq_len=21)
        loader = DataLoader(dataset, batch_size=1)
        
        batch = next(iter(loader))
        img_seq, area_seq, cloudy_seq, label, lake_id = batch
        
        assert img_seq.shape == (1, 21, 4, 512, 512)
        assert area_seq.shape == (1, 21, 1)
