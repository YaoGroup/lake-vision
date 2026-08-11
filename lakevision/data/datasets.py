"""
PyTorch Dataset classes for lake drainage classification.
"""
import json
import torch
from torch.utils.data import Dataset
import netCDF4
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict


def load_band_stats(stats_path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Load band statistics from JSON file.

    Args:
        stats_path: Path to JSON file with band statistics

    Returns:
        dict: Band statistics with format {"band_name": {"mean": X, "std": Y}, ...}
    """
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats

class LakeDataset(Dataset):
    """
    Dataset for loading combined lake NetCDF files.

    Supports two NC formats:

    1. **Preprocessed** (from combine_lake_data):
       - ``imagery``: [time, channel, y, x] with channels
         [red, green, blue, nir, swir16, swir22, mask]
       - ``water_area``: [time]
       - Optional ``cloudy_seq_*`` variables

    2. **Raw sat-tile-stack** (from sattile_stack):
       - ``reflectance``: [time, band, y, x] with bands like
         [B04, B03, B02, B08, B11, SCL, cloudmask]
       - No water_area or cloudy_seq variables

    The format is auto-detected per file. When water_area is absent the full
    time dimension is used (no windowing around peak area).

    Args:
        data_paths: path to a single .nc file, a directory of .nc files, or a list of .nc file paths
        seq_len: length of temporal sequence to extract (centered on max water area date)
        label: optional integer label for all samples (for single-class datasets).
        labels_file: optional path to a labels CSV file.
        label_col: column name for labels in CSV (default: 'label')
        id_col: column name for lake IDs in CSV (default: 'lake_id')
        labels_dict: optional pre-computed dict mapping lake_id -> label (int).
            If provided, overrides labels_file. Useful when labels need remapping
            before being passed to the dataset (e.g., label_mode='ed_split').
        normalize_imagery: whether to normalize imagery channels (default: True)
        imagery_scale: scale factor for legacy normalization if band_stats=None (default: 10000.0)
        normalize_area: whether to min-max normalize water area per sample (default: True)
        transform: Optional transform to apply to imagery tensors (applied after normalization).
        use_nir: whether to include NIR band (default: False)
        use_swir16: whether to include SWIR16 band (default: False)
        use_swir22: whether to include SWIR22 band (default: False)
        use_mask: whether to include the mask band as the last channel (default: True).
            Set to False when mask is not available (e.g. raw sat-tile-stack files).
        band_stats: Path to JSON file with band statistics, or dict with stats.
            If provided, uses per-band mean/std normalization instead of simple scaling.
        cloudy_seq_var: Name of the cloudy_seq variable in NC files (default: 'cloudy_seq_rgb').
            Set to None to disable cloudy_seq loading.
        preload_to_ram: Whether to preload all NC files into RAM during initialization.
            This eliminates I/O during training but requires ~1GB per lake file.
            Recommended for training sets when sufficient memory is available (e.g., 800GB for ~700 lakes).

    Returns per sample:
        img_seq: Tensor of shape [seq_len, C, H, W]
        area_seq: Tensor of shape [seq_len, 1] (water area, or ones if unavailable)
        cloudy_seq: Tensor of shape [seq_len, 1] (cloud/usefulness, or ones if unavailable)
        label: Integer label tensor (or -1 if no label provided)
        lake_id: String identifier for the lake
    """

    # Channel order in preprocessed NC files
    CHANNEL_ORDER = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22', 'mask']

    # Mapping from sat-tile-stack band names to canonical channel names
    BAND_TO_CHANNEL = {
        'B04': 'red', 'B03': 'green', 'B02': 'blue',
        'B08': 'nir', 'B11': 'swir16', 'B12': 'swir22',
    }

    def __init__(
        self,
        data_paths: Union[str, Path, List[str], List[Path]],
        seq_len: int = 153,
        label: Optional[int] = None,
        labels_file: Optional[Union[str, Path]] = None,
        label_col: str = 'label',
        id_col: str = 'lake_id',
        normalize_imagery: bool = True,
        imagery_scale: float = 10000.0,
        normalize_area: bool = True,
        transform=None,
        # Spectral band flags
        use_nir: bool = False,
        use_swir16: bool = False,
        use_swir22: bool = False,
        use_mask: bool = True,
        # Band statistics for normalization
        band_stats: Optional[Union[str, Path, Dict]] = None,
        # Cloudy sequence variable name
        cloudy_seq_var: Optional[str] = 'cloudy_seq_rgb',
        # Pre-computed labels dict (overrides labels_file)
        labels_dict: Optional[Dict[str, int]] = None,
        # RAM preloading
        preload_to_ram: bool = False,
    ):
        self.seq_len = seq_len
        self.default_label = label
        self.normalize_imagery = normalize_imagery
        self.imagery_scale = imagery_scale
        self.normalize_area = normalize_area
        self.transform = transform
        self.use_nir = use_nir
        self.use_swir16 = use_swir16
        self.use_swir22 = use_swir22
        self.use_mask = use_mask
        self.cloudy_seq_var = cloudy_seq_var

        # Load band statistics if provided
        if band_stats is None:
            self.band_stats = None
        elif isinstance(band_stats, (str, Path)):
            self.band_stats = load_band_stats(band_stats)
            print(f"Loaded band statistics from {band_stats}")
        else:
            self.band_stats = band_stats

        # Build list of channels to load (order matters: RGB, optional bands, optional mask)
        self.channels_to_load = ['red', 'green', 'blue']
        if use_nir:
            self.channels_to_load.append('nir')
        if use_swir16:
            self.channels_to_load.append('swir16')
        if use_swir22:
            self.channels_to_load.append('swir22')
        if use_mask:
            self.channels_to_load.append('mask')

        self.n_channels = len(self.channels_to_load)
        print(f"Loading {self.n_channels} channels: {self.channels_to_load}")

        # collect all .nc file paths
        self.file_paths = self._collect_paths(data_paths)
        if len(self.file_paths)==0:
            raise ValueError(f"No .nc files found in {data_paths}")

        # load labels: labels_dict takes priority over labels_file
        self.labels = {}
        if labels_dict is not None:
            self.labels = labels_dict
        elif labels_file is not None:
            self.labels = self._load_labels(labels_file, id_col, label_col)

        # Preload data to RAM if requested
        self.preload_to_ram = preload_to_ram
        self._cache = None
        if preload_to_ram:
            self._preload_all_data()

    def _collect_paths(self, data_paths) -> List[Path]:
        """Collect all .nc file paths from input."""
        if isinstance(data_paths, (str, Path)):
            path = Path(data_paths)
            if path.is_file():
                return [path]
            elif path.is_dir():
                return sorted(path.glob("*.nc"))
            else:
                raise ValueError(f"Path does not exist: {path}")
        else:
            # list of paths
            return [Path(p) for p in data_paths]

    def _load_labels(self, labels_file: Union[str, Path], id_col: str, label_col: str) -> dict:
        """Load labels from CSV file.

        Args:
            labels_file: path to CSV file
            id_col: column name for lake IDs (e.g., 'lake_id', 'new_id')
            label_col: column name for labels (e.g., 'label', 'final_label')

        Returns:
            dict mapping lake_id -> label (int)
        """
        df = pd.read_csv(labels_file)
        # Drop rows with missing values in required columns
        df = df.dropna(subset=[id_col, label_col])
        return dict(zip(df[id_col], df[label_col].astype(int)))

    def _preload_all_data(self):
        """Preload all NC files into RAM for faster training.

        Stores raw data (imagery, water_area, cloudy_seq, lake_id) for each file.
        Processing (normalization, windowing) is still done in __getitem__.
        Uses _load_from_disk() so both NC formats are supported.
        """
        import time
        print(f"Preloading {len(self.file_paths)} NC files to RAM...")
        start_time = time.time()

        self._cache = []
        for i, fp in enumerate(self.file_paths):
            imagery, water_area, cloudy_seq_data, lake_id = self._load_from_disk(fp)

            self._cache.append({
                'imagery': imagery,
                'water_area': water_area,
                'cloudy_seq': cloudy_seq_data,
                'lake_id': lake_id,
            })

            if (i + 1) % 100 == 0 or (i + 1) == len(self.file_paths):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(self.file_paths) - i - 1) / rate if rate > 0 else 0
                print(f"  Loaded {i + 1}/{len(self.file_paths)} files "
                      f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

        total_time = time.time() - start_time
        print(f"Preloading complete: {len(self.file_paths)} files in {total_time:.1f}s")

    def __len__(self):
        return len(self.file_paths)

    def _load_from_disk(self, fp):
        """Load a single NC file, auto-detecting format.

        Returns (imagery, water_area, cloudy_seq_data, lake_id) where
        imagery is [T, C_selected, H, W] and water_area/cloudy_seq may be None.

        Uses netCDF4 directly (not xarray) because xarray's open_dataset +
        .isel().values path adds ~2x peak memory and per-file overhead. The
        imagery variable is HDF5-chunked along the channel axis with size 1
        (chunks [51, 1, 171, 171]), so a hyperslab read on selected channels
        only decompresses the chunks we need.
        """
        with netCDF4.Dataset(str(fp)) as nc:
            nc.set_auto_mask(False)  # NaN-fill is already encoded; skip MaskedArray wrap

            def _decode(v):
                # netCDF4 returns either numpy.str_ (NC4 string) or bytes (char array)
                return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)

            # --- Auto-detect format ---
            if 'imagery' in nc.variables:
                # Preprocessed format: imagery [T, channel, H, W]
                var_name = 'imagery'
                coord_name = 'channel'
                nc_channels_all = [_decode(c) for c in nc.variables[coord_name][:]]
            elif 'reflectance' in nc.variables:
                # Raw sat-tile-stack format: reflectance [T, band, H, W]
                var_name = 'reflectance'
                coord_name = 'band'
                raw_bands = [_decode(b) for b in nc.variables[coord_name][:]]
                # Map band names (B04, B03, ...) to canonical names (red, green, ...)
                nc_channels_all = [self.BAND_TO_CHANNEL.get(b, b) for b in raw_bands]
            else:
                raise ValueError(
                    f"NC file {fp} has neither 'imagery' nor 'reflectance' variable"
                )

            # Pick the band indices we want
            channel_indices = []
            for ch in self.channels_to_load:
                if ch in nc_channels_all:
                    channel_indices.append(nc_channels_all.index(ch))
                else:
                    raise ValueError(
                        f"Channel '{ch}' not found in NC file {fp}. "
                        f"Available: {nc_channels_all}"
                    )

            # Hyperslab read: only decompresses chunks for the requested channels.
            imagery = np.asarray(
                nc.variables[var_name][:, channel_indices, :, :],
                dtype=np.float32,
            )

            water_area = None
            if 'water_area' in nc.variables:
                water_area = np.asarray(nc.variables['water_area'][:], dtype=np.float32)

            cloudy_seq_data = None
            if self.cloudy_seq_var and self.cloudy_seq_var in nc.variables:
                cloudy_seq_data = np.asarray(nc.variables[self.cloudy_seq_var][:])

            lake_id = nc.getncattr('lake_id') if 'lake_id' in nc.ncattrs() else fp.stem

        return imagery, water_area, cloudy_seq_data, lake_id

    def __getitem__(self, idx):
        # Load data from cache or disk
        if self._cache is not None:
            cached = self._cache[idx]
            imagery = cached['imagery']
            water_area = cached['water_area']
            cloudy_seq_data = cached['cloudy_seq']
            lake_id = cached['lake_id']
        else:
            fp = self.file_paths[idx]
            imagery, water_area, cloudy_seq_data, lake_id = self._load_from_disk(fp)

        n_timesteps = imagery.shape[0]

        # Determine sequence window
        if water_area is not None:
            # Center on peak water area
            water_area_filled = np.nan_to_num(water_area, nan=0.0)
            center = int(np.argmax(water_area_filled))
            half = self.seq_len // 2
            start = max(0, center - half)
            end = min(n_timesteps, center + half + 1)
            if end - start < self.seq_len:
                if start == 0:
                    end = min(self.seq_len, n_timesteps)
                else:
                    start = max(0, n_timesteps - self.seq_len)
        else:
            # No water_area: use full sequence (or first seq_len timesteps)
            start = 0
            end = min(self.seq_len, n_timesteps)

        # Extract sequences
        img_seq = imagery[start:end]  # [seq_len, C, H, W]

        if water_area is not None:
            area_seq = water_area[start:end]
        else:
            area_seq = np.ones(end - start, dtype=np.float32)

        if cloudy_seq_data is not None:
            cloudy_seq = cloudy_seq_data[start:end]
        else:
            cloudy_seq = np.ones(end - start, dtype=np.float32)

        # Handle NaNs in imagery
        img_seq = np.nan_to_num(img_seq, nan=0.0)

        # Get label
        if lake_id in self.labels:
            label = self.labels[lake_id]
        elif self.default_label is not None:
            label = self.default_label
        else:
            label = -1

        # Normalize imagery channels (all except mask if present)
        if self.normalize_imagery:
            n_imagery_channels = len(self.channels_to_load)
            if self.use_mask:
                n_imagery_channels -= 1  # exclude mask (last channel)

            if self.band_stats is not None:
                for i, ch in enumerate(self.channels_to_load[:n_imagery_channels]):
                    if ch in self.band_stats:
                        mean = self.band_stats[ch]['mean']
                        std = self.band_stats[ch]['std']
                        img_seq[:, i, :, :] = (img_seq[:, i, :, :] - mean) / std
                    else:
                        img_seq[:, i, :, :] = img_seq[:, i, :, :] / self.imagery_scale
            else:
                img_seq[:, :n_imagery_channels, :, :] = np.clip(
                    img_seq[:, :n_imagery_channels, :, :] / self.imagery_scale, 0.0, 1.0
                )

        # Convert to tensors
        img_seq = torch.tensor(img_seq, dtype=torch.float32)
        area_seq = torch.tensor(area_seq, dtype=torch.float32).unsqueeze(-1)
        cloudy_seq = torch.tensor(cloudy_seq, dtype=torch.float32).unsqueeze(-1)
        label = torch.tensor(label, dtype=torch.long)

        # Min-max normalize water area per sample
        if self.normalize_area:
            area_min = area_seq.min()
            area_max = area_seq.max()
            area_seq = (area_seq - area_min) / (area_max - area_min + 1e-8)

        # Apply transform if provided
        if self.transform:
            img_seq = self.transform(img_seq)

        return img_seq, area_seq, cloudy_seq, label, lake_id