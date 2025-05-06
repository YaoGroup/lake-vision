## ======= ##
## IMPORTS ##
## ======= ##

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import os
import dask
import dask.array

import datetime

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

## ======================================= ##
## CONNECT TO MICROSOFT PLANETARY COMPUTER ##
## ======================================= ##
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
print(f"connected to Microsoft Planetary Computer")

## ========= ##
## FUNCTIONS ##
## ========= ##

## FUNCTION TO DEFINE BOUNDING BOX AROUND A GIVEN CENTROID
def bounds_latlon_around(center_lon, center_lat, side_m=10000):
    """
    center_lon, center_lat : centroid in decimal degrees (EPSG:4326)
    side_m                 : length of box side in meters (default 10 km)
    returns                 : (minx, miny, maxx, maxy) in lon/lat
    """
    # 1) set up transformers
    to_ps = pyproj.Transformer.from_crs(4326, 3413, always_xy=True).transform
    to_ll = pyproj.Transformer.from_crs(3413, 4326, always_xy=True).transform

    # 2) project centroid into EPSG:3413 (units = m)
    x0, y0 = to_ps(center_lon, center_lat)

    # 3) build a square of side `side_m` centered on (x0,y0)
    half = side_m / 2.0
    sq_m = box(x0 - half, y0 - half, x0 + half, y0 + half)

    # 4) reproject that square back to lon/lat and grab its bounds
    sq_ll = transform(to_ll, sq_m)
    return sq_ll.bounds  # (minx, miny, maxx, maxy)



