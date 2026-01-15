"""
Data preprocessing utilities for lake drainage classification.

Functions for loading, filtering, and preprocessing lake data from various sources.
In particular, combining information from Dunmire+ 2025 and the tstacks from Rines+.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Union, List
from pathlib import Path

def load_area_sequences(
        filepath: str,
        lake_ids: Optional[Union[str, List[str]]] = None,
        start_date: str = '2019-05-01',
        end_date: str = '2019-09-30',
) -> xr.Dataset:
    """
    Load area sequence data and filter by lake ids and date range.

    Args:
        filepath: Path to the area sequence .nc file (e.g., all_lakes_2019.nc)
        lake_ids: Single lake ID or list of lake IDs to filter. If None, return all lakes.
        start_date: Start date for temporal filtering (default: '2019-05-01')
        end_date: End date for temporal filtering (default: '2019-09-30')

    Returns:
        xr.Dataset: filtered area sequence dataset

    Example:
        >>> # load single lake:
        >>> ds = load_area_sequences('all_lakes_2019.nc', lake_ids='CW2019_1579')
        >>> # load multiple lakes:
        >>> ds = load_area_sequences('all_lakes_2019.nc', lake_ids=['CW2019_1579, 'CW2019_1580'])
        >>> # load all CW2019 lakes (using substring filter)
        >>> ds = load_area_sequences('all_lakes_2019.nc')
        >>> # then manually filter: ds = filter_lakes_by_substring(ds, 'CW2019')
    """
    # load dataset
    ds = xr.open_dataset(filepath)

    # filter by lake IDs if provided:
    if lake_ids is not None:
        if isinstance(lake_ids, str):
            lake_ids = [lake_ids]

        # handle bytes vs string encoding
        ids_array = ds['ids'].values
        if isinstance(ids_array[0], bytes):
            lake_ids_encoded = [lid.encode('utf-8') for lid in lake_ids]
            ds = ds.sel(ids=lake_ids_encoded)
        else:
            ds = ds.sel(ids=lake_ids)
        
    # filter by time range 
    ds = ds.sel(time=slice(start_date, end_date))

    return ds

def filter_lakes_by_substring(ds: xr.Dataset, substring: str) -> xr.Dataset:
    """
    Filter lakes by substring match in lake ID.

    Args:
        ds: xarray Dataset with 'ids' dimension
        substring: Substring to search for in lake IDs (e.g., 'CW2019')

    Returns:
        xr.Dataset: Filtered dataset containing only lakes with matching IDs

    Example:
        >>> ds = xr.open_dataset('all_lakes_2019.nc')
        >>> ds_cw2019 = filter_lakes_by_substring(ds, 'CW2019')
    """
    ids = ds['ids'].values
    
    # Handle bytes vs string encoding
    if isinstance(ids[0], bytes):
        mask = np.array([substring in id.decode('utf-8') for id in ids])
    else:
        mask = np.array([substring in id for id in ids])

    return ds.isel(ids=mask)

def fill_nan_timeseries(
        data: Union[xr.DataArray, pd.Series, np.ndarray],
        method: str = 'ffil_bfill',
        time_coord: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fill NaN values in time series data.

    Args:
        data: time series data (xarray DataArray, panda Series or numpy array)
        method: Filling method. Options:
            - 'ffill_bfill': Forward fill then backward fill (default)
            - 'ffill': Forward fill only (use previous value)
            - 'bfill': Backward fill only (use next value)
            - 'interpolate': Linear interpolation
        time_coord: Optional time coordinates for proper indexing (used if data is numpy array)

    Returns:
        np.ndarray: Filled time series with no NaNs

    Example:
        >>> # with xarray
        >>> filled = fill_nan_timeseries(ds['S2_water'])

        >>> # with numpy array
        >>> filled = fill_nan_timeseries(area_data, time_coord=time_array)
    """
    # Convert to pandas Series for consistent handling
    if isinstance(data, xr.DataArray):
        series = pd.Series(data.values, index=data.coords[data.dims[0]].values)
    elif isinstance(data, pd.Series):
        series = data
    elif isinstance(data, np.ndarray):
        if time_coord is not None:
            series = pd.Series(data, index=time_coord)
        else:
            series = pd.Series(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Apply filling method
    if method == 'ffill_bfill':
        filled = series.ffill().bfill()
    elif method == 'ffill':
        filled = series.ffill()
    elif method == 'bfill':
        filled = series.bfill()
    elif method == 'interpolate':
        filled = series.interpolate(method='linear')
    else:
        raise ValueError(f"Unknown fill method: {method}. "
                        f"Choose from: 'ffill_bfill', 'ffill', 'bfill', 'interpolate'")

    return filled.values

def get_lake_water_area(
    ds: xr.Dataset,
    lake_id: str,
    variable: str = 'S2_water',
    fill_nans: bool = True,
    fill_method: str = 'ffill_bfill',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract water area time series for a specific lake.

    Args:
        ds: xarray Dataset containing lake area data
        lake_id: Lake ID to extract
        variable: Variable name for water area (default: 'S2_water')
        fill_nans: Whether to fill NaN values (default: True)
        fill_method: Method for filling NaNs (default: 'ffill_bfill')

    Returns:
        tuple: (time_coords, water_area_values)
            - time_coords: Array of datetime64 timestamps
            - water_area_values: Array of water area values (with NaNs filled if requested)

    Example:
        >>> ds = load_area_sequences('all_lakes_2019.nc')
        >>> time, area = get_lake_water_area(ds, 'CW2019_1579')
        >>> print(f"Shape: {area.shape}, NaNs: {np.isnan(area).sum()}")
    """
    # Select single lake
    ids_array = ds['ids'].values
    if isinstance(ids_array[0], bytes):
        ds_lake = ds.sel(ids=lake_id.encode('utf-8'))
    else:
        ds_lake = ds.sel(ids=lake_id)

    # Extract time and water area
    time_coords = ds_lake['time'].values
    water_area = ds_lake[variable].values

    # Fill NaNs if requested
    if fill_nans:
        water_area = fill_nan_timeseries(water_area, method=fill_method, time_coord=time_coords)

    return time_coords, water_area

def load_imagery_timestack(
    filepath: str,
    bands: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Load imagery timestack from .nc file.

    Args:
        filepath: Path to imagery timestack .nc file
        bands: List of band names to extract. If None, loads all bands.
            Common options: ['red', 'green', 'blue', 'nir', etc.]

    Returns:
        xr.Dataset: Imagery timestack dataset

    Example:
        >>> ds_img = load_imagery_timestack('tstack_CW2019_1579.nc')
        >>> ds_rgb = load_imagery_timestack('tstack_CW2019_1579.nc', bands=['red', 'green', 'blue'])
    """
    ds = xr.open_dataset(filepath)

    # Filter bands if requested
    if bands is not None:
        band_names = ds.coords['common_name'].values
        band_indices = [np.where(band_names == band)[0][0] for band in bands]
        ds = ds.isel(band=band_indices)

    return ds

def extract_rgb_channels(
    ds_img: xr.Dataset,
    reflectance_var: str = 'reflectance',
) -> np.ndarray:
    """
    Extract RGB channels from imagery dataset.

    Args:
        ds_img: Imagery dataset with reflectance data
        reflectance_var: Name of reflectance variable (default: 'reflectance')

    Returns:
        np.ndarray: RGB array with shape [time, 3, y, x]

    Example:
        >>> ds_img = load_imagery_timestack('tstack_CW2019_1579.nc')
        >>> rgb = extract_rgb_channels(ds_img)
        >>> print(rgb.shape)  # (153, 3, 512, 512)
    """
    band_names = ds_img.coords['common_name'].values

    # Find RGB band indices
    red_idx = np.where(band_names == 'red')[0][0]
    green_idx = np.where(band_names == 'green')[0][0]
    blue_idx = np.where(band_names == 'blue')[0][0]

    # Extract RGB bands
    rgb = ds_img[reflectance_var][:, [red_idx, green_idx, blue_idx], :, :]

    return rgb.values

def extract_mask_channel(
    ds_img: xr.Dataset,
    mask_band_name: str = 'mask',
    reflectance_var: str = 'reflectance',
) -> np.ndarray:
    """
    Extract mask channel from imagery dataset.

    Args:
        ds_img: Imagery dataset with reflectance data
        mask_band_name: Name of the mask band in common_name coordinate (default: 'mask')
        reflectance_var: Name of reflectance variable (default: 'reflectance')

    Returns:
        np.ndarray: Mask array with shape [time, 1, y, x]

    Example:
        >>> ds_img = load_imagery_timestack('tstack_CW2019_1579.nc')
        >>> mask = extract_mask_channel(ds_img, mask_band_name='mask')
        >>> print(mask.shape)  # (153, 1, 512, 512)
    """
    band_names = ds_img.coords['common_name'].values

    # Find mask band index
    mask_idx = np.where(band_names == mask_band_name)[0]

    if len(mask_idx) == 0:
        available_bands = ', '.join(band_names)
        raise ValueError(f"Mask band '{mask_band_name}' not found. "
                        f"Available bands: {available_bands}")

    mask_idx = mask_idx[0]

    # Extract mask band (keeping as [time, 1, y, x] for consistency)
    mask = ds_img[reflectance_var][:, [mask_idx], :, :]

    return mask.values


def extract_spectral_channels(
    ds_img: xr.Dataset,
    band_names_to_extract: List[str],
    reflectance_var: str = 'reflectance',
) -> np.ndarray:
    """
    Extract specified spectral channels from imagery dataset.

    Args:
        ds_img: Imagery dataset with reflectance data
        band_names_to_extract: List of band names to extract (e.g., ['nir', 'swir1', 'swir2'])
        reflectance_var: Name of reflectance variable (default: 'reflectance')

    Returns:
        np.ndarray: Spectral array with shape [time, len(band_names_to_extract), y, x]

    Example:
        >>> ds_img = load_imagery_timestack('tstack_CW2019_1579.nc')
        >>> spectral = extract_spectral_channels(ds_img, ['nir', 'swir1', 'swir2'])
        >>> print(spectral.shape)  # (153, 3, 512, 512)
    """
    available_bands = ds_img.coords['common_name'].values
    band_indices = []

    for band_name in band_names_to_extract:
        idx = np.where(available_bands == band_name)[0]
        if len(idx) == 0:
            available_str = ', '.join(available_bands)
            raise ValueError(f"Band '{band_name}' not found. "
                           f"Available bands: {available_str}")
        band_indices.append(idx[0])

    # Extract bands
    spectral = ds_img[reflectance_var][:, band_indices, :, :]

    return spectral.values

def combine_lake_data(
    imagery_path: str,
    area_ds: xr.Dataset,
    lake_id: str,
    output_path: Optional[str] = None,
    mask_band_name: str = 'mask',
    fill_nans: bool = True,
    include_spectral_bands: bool = True,
    spectral_bands: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Combine imagery and area data into a single dataset for one lake.

    Creates a standardized dataset with:
    - imagery: [time, channel, y, x] where channel = [red, green, blue, nir, swir1, swir2, mask]
      (or [red, green, blue, mask] if include_spectral_bands=False)
    - water_area: [time] scalar sequence (NaNs filled)

    Args:
        imagery_path: Path to imagery timestack .nc file (e.g., 'tstack_CW2019_1579.nc')
        area_ds: xarray Dataset containing area sequences (already loaded)
        lake_id: Lake ID to extract (e.g., 'CW2019_1579')
        output_path: Optional path to save combined .nc file. If None, doesn't save.
        mask_band_name: Name of mask band (default: 'mask')
        fill_nans: Whether to fill NaNs in water area (default: True)
        include_spectral_bands: Whether to include NIR and SWIR bands (default: True)
        spectral_bands: List of spectral bands to include. Default: ['nir', 'swir1', 'swir2']

    Returns:
        xr.Dataset: Combined dataset with imagery and water_area

    Example:
        >>> # Load area data once
        >>> area_ds = load_area_sequences('all_lakes_2019.nc')
        >>>
        >>> # Combine for single lake (with NIR + SWIR)
        >>> ds = combine_lake_data(
        ...     imagery_path='tstack_CW2019_1579.nc',
        ...     area_ds=area_ds,
        ...     lake_id='CW2019_1579',
        ...     output_path='processed/CW2019_1579.nc'
        ... )
        >>>
        >>> # Combine for single lake (RGB only, legacy mode)
        >>> ds = combine_lake_data(
        ...     imagery_path='tstack_CW2019_1579.nc',
        ...     area_ds=area_ds,
        ...     lake_id='CW2019_1579',
        ...     include_spectral_bands=False
        ... )
    """
    if spectral_bands is None:
        spectral_bands = ['nir', 'swir1', 'swir2']

    # load imagery
    ds_img = load_imagery_timestack(imagery_path)

    # extract RGB channels
    rgb = extract_rgb_channels(ds_img)  # [time, 3, y, x]

    # extract spectral bands if requested
    if include_spectral_bands:
        try:
            spectral = extract_spectral_channels(ds_img, spectral_bands)  # [time, N, y, x]
            channel_names = ['red', 'green', 'blue'] + spectral_bands + ['mask']
        except ValueError as e:
            print(f"Warning: Could not extract spectral bands: {e}")
            print("Falling back to RGB-only mode.")
            spectral = None
            channel_names = ['red', 'green', 'blue', 'mask']
    else:
        spectral = None
        channel_names = ['red', 'green', 'blue', 'mask']

    # extract mask channel
    mask = extract_mask_channel(ds_img, mask_band_name=mask_band_name)  # [time, 1, y, x]

    # combine into a single imagery array
    if spectral is not None:
        imagery = np.concatenate([rgb, spectral, mask], axis=1)  # [time, 7, y, x]
    else:
        imagery = np.concatenate([rgb, mask], axis=1)  # [time, 4, y, x]

    # get water area for this lake
    time_coords, water_area = get_lake_water_area(
        area_ds,
        lake_id,
        fill_nans=fill_nans,
    )

    # get imagery time coordinates
    img_time_coords = ds_img['time'].values

    # check time alignment
    if len(img_time_coords) != len(time_coords):
        print(f"Warning: Imagery has {len(img_time_coords)} timesteps, "
              f"area data has {len(time_coords)} timesteps")
        print(f"Aligning water area to imagery timestamps...")

        # Align water area to imagery times
        water_area = align_water_area_to_imagery(
            water_area=water_area,
            water_time=time_coords,
            imagery_time=img_time_coords,
            method='interpolate'
        )

    # use imagery time coordinate as primary
    time_to_use = img_time_coords

    # create combined dataset
    ds_combined = xr.Dataset(
        data_vars={
            'imagery': (['time', 'channel', 'y', 'x'], imagery),
            'water_area': (['time'], water_area)
        },
        coords={
            'time': time_to_use,
            'channel': channel_names,
            'lake_id': lake_id,
        },
        attrs={
            'description': 'Combined lake drainage dataset',
            'lake_id': lake_id,
            'year': 2019,
            'source_imagery': imagery_path,
            'channels': ', '.join(channel_names),
        },
    )

    # save if output path is provided
    if output_path is not None:
        # create directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        ds_combined.to_netcdf(output_path)
        print(f"saved combined dataset to {output_path}")

    return ds_combined

def align_water_area_to_imagery(
    water_area: np.ndarray,
    water_time: np.ndarray,
    imagery_time: np.ndarray,
    method: str = 'interpolate',
) -> np.ndarray:
    """
    Align water area time series to match imagery timestamps.

    Handles cases where water area and imagery have different timestamps
    by interpolating or selecting nearest values.

    Args:
        water_area: Water area values [T_water]
        water_time: Time coordinates for water area [T_water]
        imagery_time: Time coordinates for imagery [T_img]
        method: Alignment method. Options:
            - 'interpolate': Linear interpolation (default)
            - 'nearest': Use nearest water area value for each imagery time
            - 'ffill': Forward fill from water area times

    Returns:
        np.ndarray: Water area aligned to imagery times [T_img]

    Example:
        >>> water_time = np.array([...])  # 366 days
        >>> imagery_time = np.array([...])  # 153 images
        >>> aligned = align_water_area_to_imagery(water_area, water_time, imagery_time)
        >>> print(aligned.shape)  # (153,) - matches imagery
    """
    # Convert to pandas Series for time-based operations
    water_series = pd.Series(water_area, index=pd.DatetimeIndex(water_time))
    imagery_times_dt = pd.DatetimeIndex(imagery_time)

    if method == 'interpolate':
        # Linear interpolation to imagery timestamps
        aligned = water_series.reindex(
            water_series.index.union(imagery_times_dt)
        ).interpolate(method='time').loc[imagery_times_dt]

    elif method == 'nearest':
        # Use nearest water area value
        aligned = water_series.reindex(imagery_times_dt, method='nearest')

    elif method == 'ffill':
        # Forward fill
        aligned = water_series.reindex(imagery_times_dt, method='ffill')

    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Choose from: 'interpolate', 'nearest', 'ffill'")

    return aligned.values