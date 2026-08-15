"""
Dataset over the blosc2 cache built by engine/preprocessing/build_cache.py.

Key difference from LakeDataset: this hands the collate function **uint16**, not
float32. Normalization ((DN + boa_add_offset)/10000) and NaN handling are done on
the GPU by :func:`normalize_batch`. That halves every byte in the DataLoader
queue, which is what makes batch sizes of 32-64 fit in host RAM at all --
workers x prefetch x batch x sample is the term that OOMed the node before.

Worker threading
----------------
blosc2 defaults to using every core. Inside N DataLoader workers that means N
thread pools fighting each other. Always pass :func:`worker_init` as the
DataLoader's ``worker_init_fn``.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import blosc2
except ImportError:  # keep the module importable without blosc2 installed
    blosc2 = None

NODATA_U16 = 65535
QUANTIFICATION = 10000.0
# Must match build_cache.DN_SCALE. Reflectance is stored as round(DN * DN_SCALE)
# because the v2 stacks contain half-integer DN values (two observations averaged
# into one cadence bin) — ~0.6% of RGB and ~40% of B11/B12. Scaling by 2 makes
# the uint16 cast exactly lossless instead of silently rounding.
DN_SCALE = 2.0


def worker_init(worker_id):
    """Pin blosc2 to one thread per worker. Use as DataLoader(worker_init_fn=...)."""
    if blosc2 is not None:
        blosc2.set_nthreads(1)


# Band flags the model understands, keyed by Sentinel-2 band name.
_EXTRA_BAND_FLAGS = {"B08": "use_nir", "B11": "use_swir16", "B12": "use_swir22"}
_RGB_BANDS = {"B04", "B03", "B02"}


def derive_band_flags(bands):
    """Derive the model's spectral-band flags from a cache band list.

    The classifier sizes its input from use_nir/use_swir16/use_swir22, NOT from
    how many channels the dataset hands it — and it used to silently slice off
    any extras (pre-CV audit B1: a "6-band" run could quietly train on 3 bands).
    Deriving the flags from the band list makes that mismatch impossible.

    Args:
        bands: e.g. ["B04", "B03", "B02", "B08"].

    Returns:
        dict like {"use_nir": True, "use_swir16": False, "use_swir22": False}.

    Raises:
        ValueError: on a band name the model has no flag for, or missing RGB.
    """
    bands = list(bands)
    unknown = [b for b in bands if b not in _RGB_BANDS and b not in _EXTRA_BAND_FLAGS]
    if unknown:
        raise ValueError(
            f"Unknown band(s) {unknown}: the model only knows "
            f"{sorted(_RGB_BANDS)} + {sorted(_EXTRA_BAND_FLAGS)}.")
    missing_rgb = _RGB_BANDS - set(bands)
    if missing_rgb:
        raise ValueError(
            f"RGB bands {sorted(missing_rgb)} missing from {bands}: the model "
            f"assumes 3 RGB channels as its base input.")
    return {flag: band in bands for band, flag in _EXTRA_BAND_FLAGS.items()}


def normalize_batch(img_u16, boa_offset=None, nodata=NODATA_U16, n_refl=None):
    """raw DN -> float32 surface reflectance, on whatever device it is on.

    Applies (stored / DN_SCALE + boa_add_offset) / 10000.

    Args:
        img_u16: [B, T, C, H, W] as produced by CachedLakeDataset — **int16**,
            carrying uint16 bits (see that class for why). A genuine uint16
            tensor is also accepted, on torch builds that have the dtype.
        boa_offset: optional [B, T] additive offset per timestep. ESA processing
            baseline >= 04.00 uses -1000, which is exactly why the offset is not
            baked into the cache: applying it before the uint16 cast would wrap
            dark pixels around to ~65k.
        nodata: sentinel value written by the cache builder.
        n_refl: how many leading channels carry radiometry. Any trailing channels
            are treated as 0/1 indicator masks: still rescaled to [0, 1], but the
            BOA offset is NOT added to them. Defaults to all channels, which is
            correct whenever no mask channel was appended. Pass
            ``CachedLakeDataset.n_refl``.

    Returns:
        [B, T, C, H, W] float32, no-data filled with 0.
    """
    C = img_u16.shape[-3]
    if n_refl is None:
        n_refl = C

    x = img_u16.to(torch.float32)
    if x is img_u16:                    # already float32; don't mutate the caller's
        x = x.clone()

    if img_u16.dtype == torch.int16:
        # Undo the uint16 -> int16 bit view: anything with the high bit set came
        # back negative. remainder maps -1 -> 65535 in a single in-place pass
        # with no temporaries, which is the point -- at bs=32 this tensor is
        # ~15 GB, so one stray temp would cost more VRAM than the whole model.
        x.remainder_(65536.0)

    # Computed after the wrap is undone: the sentinel is -1 in the int16 view.
    missing = x == float(nodata)

    x.div_(DN_SCALE)
    if boa_offset is not None:
        off = boa_offset.to(x.dtype).view(*boa_offset.shape, 1, 1, 1)
        if n_refl < C:
            # A mask channel is an indicator, not a measurement. Adding the BOA
            # offset to it would land it at (1e4 + offset)/1e4 instead of 1, and
            # -- worse -- would make a *static* lake boundary vary over time
            # whenever the processing baseline changes mid-series, inventing a
            # temporal signal the ConvLSTM could latch onto. Harmless on CW
            # 2018/2019, where every timestep is baseline 02.12 with offset 0,
            # but it activates silently on any reprocessed or later-year stack.
            off = off.repeat(*([1] * boa_offset.ndim), C, 1, 1)
            off[..., n_refl:, :, :] = 0.0
        x.add_(off)
    x.div_(QUANTIFICATION)
    return x.masked_fill_(missing, 0.0)


class CachedLakeDataset(Dataset):
    """Reads per-channel blosc2 arrays and returns 2-byte-per-pixel sequences.

    The image tensor is **int16 carrying uint16 bits**, not float32 and not
    uint16: torch 2.2 (Sherlock's module) has no uint16 dtype, so the bytes are
    reinterpreted rather than widened. :func:`normalize_batch` undoes it.

    Returns LakeDataset's 5-tuple plus a sixth element, the per-timestep BOA
    offset needed to finish the conversion on the GPU:
        (img_seq[T,C,H,W] uint16, area_seq[T,1], cloudy_seq[T,1],
         label, lake_id, boa_offset[T])
    So it is NOT a drop-in swap for LakeDataset — callers must unpack six.

    Args:
        cache_root: directory written by build_cache.py.
        lake_ids: which lakes to include. Defaults to the manifest's list.
        bands: reflectance channels, in the order the model should see them.
        mask: None | 'lake_boundary' | 'water_mask_ndwi'. Appended as a trailing
            channel. 'lake_boundary' is static [H,W] and is broadcast over T at
            read time -- it is stored once, so this costs no I/O.
        labels_dict: lake_id -> int.
        seq_len: truncate/pad the time axis to this length.
        scalar_var: which per-timestep scalar feeds area_seq. 'p_water' is the
            Dunmire 2025 S2_water fractional extent in [0,1].
        normalize_scalar: min-max the scalar per sample. Default False, unlike
            LakeDataset -- p_water is ALREADY a physically meaningful fraction,
            and per-sample min-max would erase cross-lake amplitude (a lake that
            only ever half-fills gets stretched to look like one that fills
            completely, which is plausibly the ND signal).
    """

    def __init__(
        self,
        cache_root: Union[str, Path],
        lake_ids: Optional[Sequence[str]] = None,
        bands: Sequence[str] = ("B04", "B03", "B02"),
        mask: Optional[str] = None,
        labels_dict: Optional[Dict[str, int]] = None,
        seq_len: int = 153,
        scalar_var: str = "p_water",
        normalize_scalar: bool = False,
    ):
        if blosc2 is None:
            raise ImportError("blosc2 is required. pip install --user blosc2")

        self.root = Path(cache_root)
        self.bands = list(bands)
        self.mask = mask
        self.labels = labels_dict or {}
        self.seq_len = seq_len
        self.scalar_var = scalar_var
        self.normalize_scalar = normalize_scalar

        if lake_ids is None:
            manifest = self.root / "manifest.json"
            if not manifest.exists():
                raise FileNotFoundError(
                    f"No lake_ids given and no manifest at {manifest}")
            lake_ids = json.loads(manifest.read_text())["lake_ids"]
        self.lake_ids: List[str] = list(lake_ids)

        missing = [
            lid for lid in self.lake_ids
            if not (self.root / self.bands[0] / f"{lid}.b2nd").exists()
        ]
        if missing:
            print(f"  CachedLakeDataset: dropping {len(missing)} lakes with no "
                  f"cache entry (e.g. {missing[:3]})")
            self.lake_ids = [l for l in self.lake_ids if l not in set(missing)]
        if not self.lake_ids:
            raise ValueError(f"No cached lakes found under {self.root}")

        # n_refl is the count of radiometric channels; the mask, if present, is
        # the trailing channel. normalize_batch needs this to avoid applying the
        # BOA offset to an indicator channel.
        self.n_refl = len(self.bands)
        self.n_channels = self.n_refl + (1 if mask else 0)

    def __len__(self):
        return len(self.lake_ids)

    def _read(self, channel, lake_id):
        return blosc2.open(str(self.root / channel / f"{lake_id}.b2nd"))[:]

    def _fit_time(self, a):
        """Truncate or edge-pad the leading time axis to seq_len."""
        T = a.shape[0]
        if T == self.seq_len:
            return a
        if T > self.seq_len:
            return a[: self.seq_len]
        pad = [(0, self.seq_len - T)] + [(0, 0)] * (a.ndim - 1)
        return np.pad(a, pad, mode="edge")

    def __getitem__(self, idx):
        lake_id = self.lake_ids[idx]

        planes = [self._read(b, lake_id) for b in self.bands]   # each [T,H,W] u16
        img = np.stack(planes, axis=1)                          # [T,C,H,W] u16
        img = self._fit_time(img)
        del planes

        if self.mask:
            m = self._read(self.mask, lake_id)
            if m.ndim == 2:                                     # static boundary
                m = np.broadcast_to(m, (img.shape[0],) + m.shape)
            else:
                m = self._fit_time(m)
            # Mask channels are 0/1/255; scale to the same DN domain so that a
            # single /10000 on the GPU leaves them at 0 or 1.
            m = (m.astype(np.uint16) == 1) * np.uint16(QUANTIFICATION * DN_SCALE)
            img = np.concatenate([img, m[:, None]], axis=1)
            del m

        sc = np.load(self.root / "scalars" / f"{lake_id}.npz")
        if self.scalar_var in sc:
            area = np.nan_to_num(sc[self.scalar_var].astype(np.float32), nan=0.0)
        else:
            area = np.zeros(img.shape[0], dtype=np.float32)
        area = self._fit_time(area)

        boa = np.nan_to_num(sc["boa_add_offset"].astype(np.float32), nan=0.0) \
            if "boa_add_offset" in sc else np.zeros(img.shape[0], dtype=np.float32)
        boa = self._fit_time(boa)

        cloudy = np.nan_to_num(sc["eo_cloud_cover"].astype(np.float32), nan=0.0) / 100.0 \
            if "eo_cloud_cover" in sc else np.zeros(img.shape[0], dtype=np.float32)
        cloudy = self._fit_time(cloudy)

        # Sherlock's py-pytorch/2.2.1 has no uint16 tensor dtype — uint8 is the
        # only unsigned type it supports (uint16/32/64 arrived in torch 2.3).
        # Reinterpreting the same bytes as int16 is a pure bit-level view: no
        # copy, no value change on the wire, and the queue stays at 2 bytes per
        # element, which is the entire reason the cache exists. normalize_batch
        # undoes the wrap on the GPU with a single in-place remainder.
        img_t = torch.from_numpy(np.ascontiguousarray(img).view(np.int16))
        area_t = torch.from_numpy(area).unsqueeze(-1)
        cloudy_t = torch.from_numpy(cloudy).unsqueeze(-1)
        boa_t = torch.from_numpy(boa)

        if self.normalize_scalar:
            lo, hi = area_t.min(), area_t.max()
            area_t = (area_t - lo) / (hi - lo + 1e-8)

        label = torch.tensor(self.labels.get(lake_id, -1), dtype=torch.long)
        return img_t, area_t, cloudy_t, label, lake_id, boa_t
