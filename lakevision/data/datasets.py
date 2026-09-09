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


def _ffill_bfill_1d(a: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaNs in a 1-D float array.

    Mirrors the ffill/bfill convention the composite writers use at write time
    (see lakevision/data/synthesis.py). An all-NaN input comes back all-NaN —
    callers must guard for that.
    """
    a = np.asarray(a, dtype=np.float32).copy()
    n = a.size
    nan = np.isnan(a)
    idx = np.where(~nan, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    nan = np.isnan(a)  # leading NaNs survive the forward fill
    if nan.any():
        idx = np.where(~nan, np.arange(n), n - 1)
        idx = np.minimum.accumulate(idx[::-1])[::-1]
        a = a[idx]
    return a


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

    2. **Raw sat-tile-stack / ESSD SDR deposit** (from sattile_stack):
       - ``reflectance``: [time, band, y, x]. Band names come from a string
         ``band`` coordinate or, in the SDR deposit, from the ``band_name``
         char array beside an integer ``band`` index
         (B04/B03/B02/B08/B11/B12 -> red/green/blue/nir/swir16/swir22)
       - ``p_water``: [time] water fraction, used as the area sequence.
         It is NaN on unusable (cloudy) days and is ffill/bfill-filled at
         load — the same convention composites apply at write time.

    The format is auto-detected per file. When neither water_area nor p_water
    is present the full time dimension is used (no windowing around peak area).

    NaN guards (fail loud, not silent): a composite ``water_area`` containing
    any NaN raises (composites are written NaN-filled, so a NaN means a bad
    file), and an all-NaN ``p_water`` raises (no usable observation all
    season). Without these, one NaN would turn the whole min-max-normalized
    area sequence — and the loss — into NaN.

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
            Set to None to disable cloudy_seq loading. The special name
            ``'observed'`` derives the series from the file instead of reading a
            variable: 1.0 where ``p_water`` was measured that day, 0.0 where it
            is NaN and gets forward-filled (composites, which carry no NaN,
            yield all ones).
        validity_channel: append an aux channel that is 1 where the red band was
            observed (finite) and 0 where it was NaN, per pixel per day. Computed
            BEFORE any fill. (default: False)
        fill: how NaN pixels are filled: 'zero' (default, ESSD) or 'mean' (the
            training-split band mean from ``band_stats``, so the pixel standardises
            to 0 instead of about -7 sigma). 'mean' requires band_stats.
        mask_source: deposit masks to append as aux channels: None (default),
            'static' (``lake_boundary`` [y, x], broadcast over time), 'dynamic'
            (``water_mask_ndwi`` [time, y, x], fill value 255 -> 0) or 'both'.
            These are separate variables in the SDR deposit, not a band, so they
            are independent of ``use_mask`` (the composites' trailing band).

    Channel layout of img_seq: [spectral bands in channels_to_load order,
    aux channels (validity, static mask, dynamic mask; only those enabled),
    legacy trailing 'mask' band if use_mask]. ``n_channels`` is the total and
    ``n_aux_channels`` the aux count; the model asserts against them.
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
        # Aux channels and fill policy (all off = ESSD-identical)
        validity_channel: bool = False,
        fill: str = 'zero',
        mask_source: Optional[str] = None,
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

        if fill not in ('zero', 'mean'):
            raise ValueError(f"fill must be 'zero' or 'mean', got '{fill}'")
        if mask_source not in (None, 'static', 'dynamic', 'both'):
            raise ValueError(f"mask_source must be None, 'static', 'dynamic' or 'both', got '{mask_source}'")
        self.validity_channel = validity_channel
        self.fill = fill
        self.mask_source = mask_source
        self.load_static_mask = mask_source in ('static', 'both')
        self.load_dynamic_mask = mask_source in ('dynamic', 'both')
        self.aux_channel_names = (
            (['validity'] if validity_channel else [])
            + (['mask_static'] if self.load_static_mask else [])
            + (['mask_dynamic'] if self.load_dynamic_mask else []))
        self.n_aux_channels = len(self.aux_channel_names)

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

        if self.fill == 'mean' and self.band_stats is None:
            raise ValueError("fill='mean' needs band_stats (the training-split band means)")

        self.n_spectral_channels = len(self.channels_to_load) - (1 if use_mask else 0)
        self.n_channels = len(self.channels_to_load) + self.n_aux_channels
        print(f"Loading {self.n_channels} channels: {self.channels_to_load[:self.n_spectral_channels]}"
              f" + aux {self.aux_channel_names}"
              f"{' + [mask]' if use_mask else ''}")

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
            imagery, water_area, cloudy_seq_data, lake_id, aux = self._load_from_disk(fp)

            self._cache.append({
                'imagery': imagery,
                'water_area': water_area,
                'cloudy_seq': cloudy_seq_data,
                'lake_id': lake_id,
                'aux': aux,
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

        Returns (imagery, water_area, cloudy_seq_data, lake_id, aux) where
        imagery is [T, C_selected, H, W] (NaNs intact), water_area/cloudy_seq
        may be None, and aux is [T, n_aux, H, W] float32 or None.

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
                # Raw sat-tile-stack / SDR format: reflectance [T, band, H, W].
                # The SDR deposit stores an integer 'band' index with names in
                # a 'band_name' char array; older stacks use a string 'band'
                # coordinate directly.
                var_name = 'reflectance'
                if 'band_name' in nc.variables:
                    raw_bands = [
                        ''.join(_decode(c) for c in np.atleast_1d(row))
                        .strip('\x00').strip()
                        for row in nc.variables['band_name'][:]
                    ]
                else:
                    raw_bands = [_decode(b) for b in nc.variables['band'][:]]
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

            # --- Aux channels, computed on the raw (NaN-carrying) reflectance ---
            aux_parts = []
            if self.validity_channel:
                # Red is always channel 0 of channels_to_load.
                aux_parts.append(np.isfinite(imagery[:, 0]).astype(np.float32))
            if self.load_static_mask:
                if 'lake_boundary' not in nc.variables:
                    raise ValueError(f"{fp}: mask_source asks for 'lake_boundary' but the file has none")
                lb = np.asarray(nc.variables['lake_boundary'][:]) == 1
                aux_parts.append(np.broadcast_to(lb, imagery.shape[0:1] + lb.shape).astype(np.float32))
            if self.load_dynamic_mask:
                if 'water_mask_ndwi' not in nc.variables:
                    raise ValueError(f"{fp}: mask_source asks for 'water_mask_ndwi' but the file has none")
                # uint8: 0 no_water, 1 water, 255 fill (unobserved) -> 0; the
                # validity channel is where "unobserved" is carried.
                wm = np.asarray(nc.variables['water_mask_ndwi'][:]) == 1
                aux_parts.append(wm.astype(np.float32))
            aux = np.stack(aux_parts, axis=1) if aux_parts else None   # [T, n_aux, H, W]

            water_area = None
            observed = None
            if 'water_area' in nc.variables:
                # Composites are written NaN-filled; a NaN here means a bad
                # file (e.g. an all-NaN Dunmire series slipped through the
                # writer). One NaN would poison the min-max-normalized
                # area_seq, so refuse it.
                water_area = np.asarray(nc.variables['water_area'][:], dtype=np.float32)
                n_nan = int(np.isnan(water_area).sum())
                if n_nan:
                    raise ValueError(
                        f"{fp}: water_area contains {n_nan}/{water_area.size} "
                        f"NaN(s). Composites are written NaN-filled — rebuild "
                        f"this file (lakevision/data/synthesis.py)."
                    )
            elif 'p_water' in nc.variables:
                # SDR deposit: p_water is NaN on unusable days by design.
                # Fill at load; only an all-NaN season is an error.
                water_area = np.asarray(nc.variables['p_water'][:], dtype=np.float32)
                if np.isnan(water_area).all():
                    raise ValueError(
                        f"{fp}: p_water is all-NaN (no usable observation all "
                        f"season) — no such lake should exist in the deposit."
                    )
                observed = np.isfinite(water_area).astype(np.float32)  # before the fill
                water_area = _ffill_bfill_1d(water_area)

            cloudy_seq_data = None
            if self.cloudy_seq_var == 'observed':
                if water_area is None:
                    raise ValueError(f"{fp}: cloudy_seq_var='observed' needs p_water or water_area")
                # Composites carry no NaN, so every day counts as observed.
                cloudy_seq_data = observed if observed is not None else np.ones(water_area.shape, np.float32)
            elif self.cloudy_seq_var and self.cloudy_seq_var in nc.variables:
                cloudy_seq_data = np.asarray(nc.variables[self.cloudy_seq_var][:])

            lake_id = nc.getncattr('lake_id') if 'lake_id' in nc.ncattrs() else fp.stem

        return imagery, water_area, cloudy_seq_data, lake_id, aux

    def __getitem__(self, idx):
        # Load data from cache or disk
        if self._cache is not None:
            cached = self._cache[idx]
            imagery = cached['imagery']
            water_area = cached['water_area']
            cloudy_seq_data = cached['cloudy_seq']
            lake_id = cached['lake_id']
            aux = cached['aux']
        else:
            fp = self.file_paths[idx]
            imagery, water_area, cloudy_seq_data, lake_id, aux = self._load_from_disk(fp)

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

        # Extract sequences. With preload_to_ram the slice is a view into the
        # cached array and the fill/normalisation below are in place, so copy.
        img_seq = imagery[start:end]  # [seq_len, C, H, W]
        if self._cache is not None:
            img_seq = img_seq.copy()
        aux_seq = None if aux is None else aux[start:end]

        if water_area is not None:
            area_seq = water_area[start:end]
        else:
            area_seq = np.ones(end - start, dtype=np.float32)

        if cloudy_seq_data is not None:
            cloudy_seq = cloudy_seq_data[start:end]
        else:
            cloudy_seq = np.ones(end - start, dtype=np.float32)

        # Handle NaNs in imagery, in place. 'mean' fills each spectral band at its
        # training-split mean so the pixel standardises to 0; anything else
        # (the legacy mask band, or fill='zero') gets 0.
        n_spec = self.n_spectral_channels
        if self.fill == 'mean':
            for i, ch in enumerate(self.channels_to_load[:n_spec]):
                mean = self.band_stats[ch]['mean'] if ch in self.band_stats else 0.0
                np.nan_to_num(img_seq[:, i], nan=mean, copy=False)
            if img_seq.shape[1] > n_spec:
                np.nan_to_num(img_seq[:, n_spec:], nan=0.0, copy=False)
        else:
            np.nan_to_num(img_seq, nan=0.0, copy=False)

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

        # Assemble [spectral..., aux..., legacy mask]
        if aux_seq is not None:
            img_seq = np.concatenate(
                [img_seq[:, :n_spec], aux_seq, img_seq[:, n_spec:]], axis=1)

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