"""
ESSD dataset synthesis: compose the final per-lake NetCDF from sat-tile-stack
raw output, Dunmire water-area time series, Dunmire lake polygons, and
(optionally) GUI-derived labels.

The cloudy-tile classifier's useful/cloudy-flag time series is appended in
a separate pass (see engine/preprocessing/synthesize_region.py phase 2).
This module handles everything except that GPU-bound step.

Output schema (CF-1.8 compliant)
--------------------------------
Dimensions:
    time     = 153
    channel  = 7
    y        = 512
    x        = 512

Data variables:
    imagery        (time, channel, y, x)  float32
                    [red, green, blue, nir, swir16, cloudmask_scl, mask]
                    Reflectance bands are sat-tile-stack raw / 10000 (CF units='1').
                    cloudmask_scl is binary 0/1 (SCL-derived per-pixel cloud flag).
                    mask is binary 0/1 (Dunmire polygon rasterized, broadcast over T).
    water_area     (time,)                float32
                    Dunmire S2_water time series, NaN-filled via ffill/bfill.
    cloudy_seq_rgb (time,)                int32    (appended later by cloudy-tile)

Coordinates:
    time          datetime64   (May 1 – Sep 30, daily)
    channel       str          ['red','green','blue','nir','swir16','cloudmask_scl','mask']
    band          str          ['B04','B03','B02','B08','B11','','']  (aux coord on channel)
    x, y          float64      (UTM meters, inherited from raw sat-tile-stack)
    crs           int32        (grid_mapping variable, inherited)
    lake_id       str scalar

Global attributes:
    Conventions           = 'CF-1.8'
    source_raw_nc         = path of sat-tile-stack input
    source_dunmire_area   = path of all_lakes_{YEAR}.nc
    source_dunmire_poly   = path of labels_{YEAR}_volumes.geojson
    lake_id               = 'CW2018_1077'
    year                  = 2018
    processing_history    = ISO timestamp + software versions
    region                = 'CW'
    (if labels supplied:)  label, p_ND, p_HF, p_MD, p_LD, p_CD, notes, flagged

Usage
-----
    synth = LakeDatasetSynthesizer(
        raw_nc=Path('/oak/.../stacks/CW_2018/CW2018_1077.nc'),
        dunmire_area_ds=xr.open_dataset('all_lakes_2018.nc'),
        dunmire_polygons_gdf=gpd.read_file('labels_2018_volumes.geojson'),
        lake_id='CW2018_1077',
        year=2018,
        label_row=labels_df.loc['CW2018_1077'],  # optional
    )
    output_path = synth.synthesize(Path('/oak/.../composites/CW_2018/'))
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .preprocessing import get_lake_water_area


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# sat-tile-stack raw band names → canonical channel names
BAND_TO_CHANNEL = {
    'B04': 'red',
    'B03': 'green',
    'B02': 'blue',
    'B08': 'nir',
    'B11': 'swir16',
    # SCL and cloudmask stay as-is; we remap cloudmask to a renamed channel below
}

# Order of channels in the output imagery tensor
CHANNEL_ORDER = ['red', 'green', 'blue', 'nir', 'swir16', 'cloudmask_scl', 'mask']

# Reverse lookup: channel name → raw sat-tile-stack band name (where applicable)
CHANNEL_TO_BAND = {v: k for k, v in BAND_TO_CHANNEL.items()}
CHANNEL_TO_BAND['cloudmask_scl'] = 'cloudmask'  # sat-tile-stack names it 'cloudmask'
CHANNEL_TO_BAND['mask'] = ''                    # no raw band — rasterized from polygon

REFLECTANCE_SCALE = 10000.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _rasterize_polygon_to_grid(
    geom,
    transform,
    out_shape,
) -> np.ndarray:
    """Rasterize a shapely geometry onto a raster grid defined by an affine
    transform, returning a uint8 {0, 1} mask of shape ``out_shape``.

    Uses ``rasterio.features.rasterize``.
    """
    from rasterio.features import rasterize  # local import to keep top-level light

    arr = rasterize(
        shapes=[(geom, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8',
        all_touched=False,
    )
    return arr


def _build_affine_from_raw(raw_ds: xr.Dataset):
    """Reconstruct an affine.Affine from the raw sat-tile-stack dataset.

    sat-tile-stack writes the transform as a 9-element list in the
    ``transform`` global attribute (see ``sat_tile_stack.io``).
    """
    from affine import Affine

    t = raw_ds.attrs.get('transform')
    if t is None:
        raise ValueError("raw .nc is missing global 'transform' attribute; "
                         "cannot reconstruct affine")
    # Expected: [a, b, c, d, e, f, 0, 0, 1]
    t = list(t)
    return Affine(t[0], t[1], t[2], t[3], t[4], t[5])


def _labels_to_attrs(label_row: Optional[pd.Series]) -> dict:
    """Convert a row of the GUI labels CSV to a flat dict of NetCDF attrs.

    Returns an empty dict if ``label_row`` is None.
    """
    if label_row is None:
        return {}
    out = {}
    # Expected columns: label, p_ND, p_HF, p_MD, p_LD, p_CD, notes, flagged
    for col in ('label', 'p_ND', 'p_HF', 'p_MD', 'p_LD', 'p_CD'):
        if col in label_row and pd.notna(label_row[col]):
            val = label_row[col]
            # NetCDF attrs should be python-native
            if col == 'label':
                out[col] = str(val)
            else:
                out[col] = float(val)
    if 'notes' in label_row and pd.notna(label_row['notes']) and str(label_row['notes']).strip():
        out['notes'] = str(label_row['notes'])
    if 'flagged' in label_row and pd.notna(label_row['flagged']):
        # NetCDF attrs don't have a bool type — store as int8 0/1
        out['flagged'] = int(bool(label_row['flagged']))
    return out


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class LakeDatasetSynthesizer:
    """Compose a single lake's composite NetCDF for the ESSD benchmark dataset.

    This object is single-use: construct once per lake, call ``synthesize()``
    once. The ``synthesize()`` method returns the output path.

    Parameters
    ----------
    raw_nc : Path
        Path to the raw sat-tile-stack NetCDF for this lake. Must contain
        ``reflectance[time, band, y, x]`` with ``band`` coord values
        ``['B04','B03','B02','B08','B11','SCL','cloudmask']`` (order-agnostic).
    dunmire_area_ds : xr.Dataset
        Already-opened Dunmire area dataset (``all_lakes_{YEAR}.nc``). The
        caller should open this once in the batch driver and pass the same
        reference to every synthesizer to avoid repeated file opens.
    dunmire_polygons_gdf : geopandas.GeoDataFrame
        Already-loaded Dunmire polygons GeoDataFrame (from
        ``labels_{YEAR}_volumes.geojson``). Must have a ``new_id`` column
        matching ``lake_id`` values and a valid geometry column.
    lake_id : str
        Lake identifier, e.g. ``'CW2018_1077'``. Must match the raw NC
        filename stem, the Dunmire dataset ``ids`` dimension, and the
        GeoDataFrame ``new_id`` column.
    year : int
        Year (2018 or 2019). Used for time slicing + attrs.
    label_row : pd.Series, optional
        A single row from the labels CSV (indexed by lake_id). If provided,
        label information is stored as dataset-level attrs on the output.

    Notes
    -----
    - CF-1.8 compliance is maintained: the raw sat-tile-stack CRS + grid
      mapping is carried through unchanged, time is encoded CF-standard,
      all new variables get ``long_name`` and ``units``/``flag_values``.
    - The ``cloudy_seq_rgb`` variable is NOT added here — it's a separate
      pass that runs the cloudy-tile classifier on the output of this step.
    """

    # --- channel composition pipeline ---

    def __init__(
        self,
        raw_nc: Path,
        dunmire_area_ds: xr.Dataset,
        dunmire_polygons_gdf,
        lake_id: str,
        year: int,
        label_row: Optional[pd.Series] = None,
    ):
        self.raw_nc = Path(raw_nc)
        self.dunmire_area_ds = dunmire_area_ds
        self.dunmire_polygons_gdf = dunmire_polygons_gdf
        self.lake_id = lake_id
        self.year = int(year)
        self.label_row = label_row

        # State populated during synthesize()
        self._raw_ds: Optional[xr.Dataset] = None
        self._imagery: Optional[xr.DataArray] = None
        self._water_area: Optional[np.ndarray] = None

    # ---- stage methods (internal, called in sequence by synthesize) ----

    def _load_raw(self) -> None:
        """Open the raw sat-tile-stack NetCDF and pull out reflectance."""
        ds = xr.open_dataset(self.raw_nc)
        if 'reflectance' not in ds:
            ds.close()
            raise ValueError(f"{self.raw_nc} does not contain 'reflectance'")
        self._raw_ds = ds

    def _close_raw(self) -> None:
        if self._raw_ds is not None:
            self._raw_ds.close()
            self._raw_ds = None

    def _build_reflectance_channels(self) -> dict:
        """Extract and rename the 5 reflectance bands + SCL-derived cloudmask.

        Returns a dict mapping channel_name → np.ndarray [T, H, W].
        Reflectance channels are scaled /10000 and cast to float32.
        ``cloudmask_scl`` stays binary 0/1 (cast to float32 for consistent
        channel dtype in the final [T, C, H, W] tensor).
        """
        raw_bands = list(self._raw_ds['reflectance'].coords['band'].values)
        refl = self._raw_ds['reflectance'].values  # [T, band, H, W]

        out = {}
        for band_name, channel_name in BAND_TO_CHANNEL.items():
            if band_name not in raw_bands:
                raise ValueError(
                    f"Expected band '{band_name}' not in raw .nc. Found: {raw_bands}"
                )
            idx = raw_bands.index(band_name)
            out[channel_name] = (refl[:, idx, :, :] / REFLECTANCE_SCALE).astype(np.float32)

        # cloudmask (renamed cloudmask_scl in the composite schema)
        if 'cloudmask' in raw_bands:
            idx = raw_bands.index('cloudmask')
            out['cloudmask_scl'] = refl[:, idx, :, :].astype(np.float32)
        else:
            # Shouldn't happen for builds using --cloudmask scl, but be defensive
            T, _, H, W = refl.shape
            out['cloudmask_scl'] = np.zeros((T, H, W), dtype=np.float32)

        return out

    def _build_static_dunmire_mask(self) -> np.ndarray:
        """Rasterize the lake's Dunmire polygon onto the raw tstack grid.

        Returns a 2D [H, W] uint8 {0, 1} array. We broadcast to [T, H, W]
        when assembling the final channel tensor.
        """
        # Look up this lake's polygon
        gdf = self.dunmire_polygons_gdf
        if 'new_id' not in gdf.columns:
            raise ValueError("dunmire_polygons_gdf must have 'new_id' column")
        row = gdf[gdf['new_id'] == self.lake_id]
        if len(row) == 0:
            raise ValueError(f"{self.lake_id} not found in Dunmire polygons")
        geom_wgs84 = row.geometry.iloc[0]

        # Reproject the polygon into the raw .nc's CRS
        raw_crs = self._raw_ds.attrs.get('crs')
        if raw_crs is None:
            raise ValueError("raw .nc is missing 'crs' global attribute")
        # raw_crs is a string like 'epsg:32622'
        geom_projected = (
            gdf.loc[row.index]
               .to_crs(raw_crs)
               .geometry.iloc[0]
        )

        # Rasterize
        transform = _build_affine_from_raw(self._raw_ds)
        H = int(self._raw_ds.sizes['y'])
        W = int(self._raw_ds.sizes['x'])
        mask = _rasterize_polygon_to_grid(geom_projected, transform, (H, W))
        return mask  # uint8 [H, W]

    def _build_imagery_tensor(self) -> xr.DataArray:
        """Stack all 7 channels into a single DataArray with proper coords."""
        channels = self._build_reflectance_channels()

        # Rasterize static mask, broadcast to [T, H, W]
        static_mask = self._build_static_dunmire_mask()  # [H, W] uint8
        T = channels['red'].shape[0]
        mask_tensor = np.broadcast_to(static_mask[None, :, :], (T, *static_mask.shape))
        channels['mask'] = mask_tensor.astype(np.float32)

        # Stack in canonical order: [T, C, H, W]
        stacked = np.stack([channels[c] for c in CHANNEL_ORDER], axis=1)

        # Build coords: channel (canonical names) + band (aux coord, sat-tile-stack band names)
        time_coord = self._raw_ds['time']
        y_coord = self._raw_ds['y']
        x_coord = self._raw_ds['x']
        band_aux = [CHANNEL_TO_BAND[c] for c in CHANNEL_ORDER]  # '' for mask

        da = xr.DataArray(
            stacked,
            dims=('time', 'channel', 'y', 'x'),
            coords={
                'time': time_coord,
                'channel': ('channel', CHANNEL_ORDER),
                'band': ('channel', band_aux),
                'y': y_coord,
                'x': x_coord,
            },
            name='imagery',
            attrs={
                'long_name': 'multispectral imagery time series',
                'units': '1',
                'description': (
                    'Surface reflectance bands (red/green/blue/nir/swir16) scaled '
                    '/10000 from Sentinel-2 L2A raw DN; cloudmask_scl is the '
                    'Sen2Cor SCL-derived per-pixel cloud flag (0=clear,1=cloudy); '
                    'mask is the Dunmire+ 2025 static lake polygon rasterized '
                    'and broadcast across the time dimension (0=off-lake,1=on-lake).'
                ),
                'coordinates': 'time channel band y x',
                'grid_mapping': 'crs',
                '_FillValue': np.float32(np.nan),
            },
        )
        return da

    def _build_water_area(self) -> np.ndarray:
        """Extract Dunmire S2_water for this lake, slice to May-Sep, fill NaNs.

        Returns a float32 [153] array.
        """
        start = f'{self.year}-05-01'
        end = f'{self.year}-09-30'
        # Slice time first (avoid pulling full-year per lake)
        sliced = self.dunmire_area_ds.sel(time=slice(start, end))
        # Extract this lake's series
        area, _ = get_lake_water_area(sliced, self.lake_id, variable='S2_water',
                                     fill_nans=True, fill_method='ffill_bfill')
        return area.astype(np.float32)

    def _assemble_dataset(self) -> xr.Dataset:
        """Compose imagery + water_area + coords + attrs into a final Dataset."""
        imagery = self._imagery

        water_area_da = xr.DataArray(
            self._water_area,
            dims=('time',),
            coords={'time': imagery['time']},
            name='water_area',
            attrs={
                'long_name': 'lake water surface area',
                'units': 'km^2',
                'description': (
                    'Sentinel-2 derived water surface area time series from '
                    'Dunmire+ 2025 (variable S2_water). NaNs filled via '
                    'ffill/bfill.'
                ),
                'source': 'Dunmire et al. 2025',
                '_FillValue': np.float32(np.nan),
            },
        )

        # Carry the CRS variable through from raw
        data_vars = {'imagery': imagery, 'water_area': water_area_da}
        if 'crs' in self._raw_ds.variables:
            data_vars['crs'] = self._raw_ds['crs']

        # Build output dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'lake_id': self.lake_id,
            },
        )

        # Carry through scalar time coordinates that sat-tile-stack exposes
        # (eo_cloud_cover and pct_nans per timestep, if present)
        for aux in ('eo_cloud_cover', 'pct_nans'):
            if aux in self._raw_ds.coords:
                ds = ds.assign_coords({aux: self._raw_ds[aux]})

        # --- Global attrs (CF-1.8 + provenance) ---
        now = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        attrs = {
            'Conventions': 'CF-1.8',
            'title': 'Lake drainage benchmark dataset — per-lake composite',
            'lake_id': self.lake_id,
            'year': np.int64(self.year),
            'region': self.lake_id.split('_')[0].rstrip('0123456789')[:2]
                      if self.lake_id else '',
            'source_raw_nc': str(self.raw_nc),
            'processing_history': (
                f'{now}: synthesized by '
                f'lakevision.data.synthesis.LakeDatasetSynthesizer '
                f'from sat-tile-stack raw .nc + Dunmire water area + '
                f'Dunmire static lake polygon.'
            ),
        }
        # Inherit a few informative attrs from raw
        for carry in ('crs', 'transform', 'resolution',
                      'stac_band_metadata', 'instruments', 'constellation',
                      's2:product_type'):
            if carry in self._raw_ds.attrs:
                attrs[f'raw_{carry}' if carry in ('crs', 'transform', 'resolution')
                      else carry] = self._raw_ds.attrs[carry]
        # Label attrs (if any)
        attrs.update(_labels_to_attrs(self.label_row))
        ds.attrs = attrs
        return ds

    # ---- public entry point ----

    def synthesize(self, output_dir: Path) -> Path:
        """Build and write the composite NetCDF for this lake.

        Returns the output path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outfile = output_dir / f'{self.lake_id}.nc'

        try:
            self._load_raw()
            self._imagery = self._build_imagery_tensor()
            self._water_area = self._build_water_area()
            ds = self._assemble_dataset()

            # zlib-compress imagery + water_area (matches sat-tile-stack convention)
            encoding = {
                'imagery': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
                'water_area': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
            }
            if 'time' in ds:
                encoding['time'] = {
                    'units': f'days since {self.year}-05-01',
                    'calendar': 'proleptic_gregorian',
                }

            ds.to_netcdf(outfile, engine='netcdf4', format='NETCDF4',
                        mode='w', encoding=encoding)
        finally:
            self._close_raw()

        return outfile
