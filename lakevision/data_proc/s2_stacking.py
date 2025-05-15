## ======= ##
## IMPORTS ##
## ======= ##

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import zarr
import rioxarray
import matplotlib.pyplot as plt
import os
import dask
import dask.array
import math

from tqdm.auto import tqdm
import time, datetime as dt

from collections import Counter

import pystac_client
from pystac.extensions.projection import ProjectionExtension as proj

import planetary_computer
import rasterio
import rasterio.features
from rasterio.features import rasterize

import stackstac
import pyproj

import dask.diagnostics

from shapely.geometry import box
from shapely.ops import transform

from scipy.ndimage import binary_propagation
from scipy.ndimage import label

import sat_tile_stack
from sat_tile_stack import sattile_stack, sat_mask_array, write_netcdf_from_da, timestack_to_movie


## =============================================================================== ##
## LOOP OVER LAKES IN THE .csv FILE AND SAVE EACH TIMESTACK AS A SEPARATE .nc FILE ##
## =============================================================================== ##

from tqdm.auto import tqdm
import time, datetime as dt

# READ IN THE LAKE INFORMATION .geojson FILE
fp_geojson = "/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision/sandbox/dunmire_cw2019_shuttle.geojson"
gdf = gpd.read_file(fp_geojson)
gdf_mask = gpd.read_file(fp_geojson)

# DECIDE WHETHER TO NORMALIZE THE IMAGERY UPON COMPLIING OR NOT
normalize = True

# SPECIFY TIME RANGE
time_range = '2019-05-01/2019-09-30'

## CONNECT TO MICROSOFT PLANETARY COMPUTER
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,)
print(f"connected to Microsoft Planetary Computer")

# SPECIFY IMAGERY BANDS
band_names = ["B04",  # red (665 nm)
              "B03",  # green (560 nm)
              "B02",  # blue (490 nm)
              "B08"]  # NIR (842 nm)
              # "B11"]  # SWIR1 (1610 nm)

total_start = time.perf_counter()

# LOOP OVER LAKE LOCATIONS
for idx_lake in tqdm(range(0,len(gdf)), desc="Processing lakes"):
    
    iter_start = time.perf_counter()

    print(f"\n working on lake {gdf.iloc[idx_lake].lakenum}")

    # LAKE CENTROID
    centroid = (gdf.iloc[idx_lake].centroid_x, gdf.iloc[idx_lake].centroid_y)

    # CALL FUNCTION TO GENERATE TIMESTACK
    timestack = sattile_stack(catalog, centroid, band_names, pix_res=10, tile_size=512, time_range=time_range, normalize=False, mask=gdf_mask.iloc[[idx_lake]], pull_to_mem=True)

    # WRITE TIMESTACK AS .nc FILE TO DISK
    outfile = f"/oak/stanford/groups/cyaolai/JoshRines/data/2019cw_tstacks/tstack_{gdf.iloc[idx_lake].lakenum}.nc"
    write_netcdf_from_da(timestack, outfile)
    
    tqdm.write(f"lake {gdf.iloc[idx_lake].lakenum} → {dt.timedelta(seconds=time.perf_counter()-iter_start)}")
    
tqdm.write(f"all lakes → {dt.timedelta(seconds=time.perf_counter()-total_start)}")


print(f"successfully ran {__file__} at {dt.datetime.now()}")