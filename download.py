#!/usr/bin/env python3
"""Fast, resumable downloader for MAIAC AOD, ERA5-Land & MERRA-2 over India.

Downloads analysis-ready daily products for mainland India and stores them in
Zarr v2 directory stores. Run the one-day pilot first, then explicitly choose
the date range for any larger download.

Usage:
    python download.py --years 2023                 # Explicit full-year run
    python download.py --years 2019-2023            # Explicit 5-year run
    python download.py --years 2023 --datasets aod  # AOD only
    python download.py --pilot                      # One day (2023-01-15)

Requirements:
    conda activate adp
    Earth Engine authenticated for project 'additional-project-511307'
"""
from __future__ import annotations

import argparse
import calendar
import concurrent.futures as cf
import datetime as dt
import fcntl
import json
import math
import os
import random
import shutil
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import ee
import numpy as np
import rasterio
import zarr
from numcodecs import Blosc
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape, mapping
from shapely.ops import transform as geom_transform, unary_union
from tqdm import tqdm


# ── Configuration ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'processed'
TEMP_ROOT = ROOT / 'data' / 'raw' / 'transfer'
TEMP = None
EE_PROJECT = 'additional-project-511307'
EE_ENDPOINT = 'https://earthengine-highvolume.googleapis.com'

# Earth Engine collection IDs
MAIAC = 'MODIS/061/MCD19A2_GRANULES'
ERA5 = 'ECMWF/ERA5_LAND/DAILY_AGGR'
MERRA = 'NASA/GSFC/MERRA/aer/2'
BOUNDARY_ASSET = 'FAO/GAUL/2025/level1'

# MAIAC sinusoidal grid parameters
RES = 926.6254330558334                     # pixel size in metres
XORIGIN = -3335851.559                      # affine origin x
YORIGIN = -8895604.157333                   # affine origin y
SINU = CRS.from_proj4('+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +units=m +no_defs')
GEE_SINU = 'SR-ORG:6974'

# ERA5-Land bands to download
ERA5_BANDS = [
    'temperature_2m', 'dewpoint_temperature_2m',
    'u_component_of_wind_10m', 'v_component_of_wind_10m',
    'surface_pressure', 'total_precipitation_sum',
]

# Native-grid affine transforms (pixel-edge convention, x/y scale first)
ERA5_NATIVE_TF = [0.1, 0.0, -180.05, 0.0, -0.1, 90.05]
MERRA_NATIVE_TF = [0.625, 0.0, -180.3125, 0.0, 0.5, -90.25]

# Expected output transforms for validation (Earth Engine clips to the bbox)
ERA5_OUT_TF = [66.95, 0.1, 0.0, 39.05, 0.0, -0.1]

# India bounding box for ERA5/MERRA-2 (lon_min, lat_min, lon_max, lat_max)
INDIA_BBOX = (67.0, 5.0, 99.0, 39.0)


def _positive_grid_crop(native_tf, bbox):
    """Outward-snap a bbox to a positive-x/positive-y native cell-edge grid."""
    xscale, _, xorigin, _, yscale, yorigin = native_tf
    xmin, ymin, xmax, ymax = bbox
    x0 = xorigin + math.floor((xmin - xorigin) / xscale) * xscale
    x1 = xorigin + math.ceil((xmax - xorigin) / xscale) * xscale
    y0 = yorigin + math.floor((ymin - yorigin) / yscale) * yscale
    y1 = yorigin + math.ceil((ymax - yorigin) / yscale) * yscale
    cols = int(round((x1 - x0) / xscale))
    rows = int(round((y1 - y0) / yscale))
    return [x0, xscale, 0.0, y0, 0.0, yscale], rows, cols


MERRA_OUT_TF, MERRA_ROWS, MERRA_COLS = _positive_grid_crop(
    MERRA_NATIVE_TF, INDIA_BBOX)

# Zarr storage settings
CHUNK = 1024
COMPRESSOR = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
AOD_RULE = ('MCD19A2.061 Optical_Depth_055 raw 0..3000; '
            'AOD_QA cloud=1, land=0, adjacency=0, quality=0; '
            'mean across granules, round to integer; scale=0.001')
MAIAC_START = dt.date(2000, 2, 24)


# ── Boundary & Grid ────────────────────────────────────────────────────────────

def build_boundary():
    """Build the India mainland boundary and buffered analysis region.

    Returns server-side (Earth Engine) buffered geometry, client-side
    (Shapely) sinusoidal geometries for rasterisation, and the raw
    FeatureCollection info dict for archival.
    """
    fc = (ee.FeatureCollection(BOUNDARY_ASSET)
          .filter(ee.Filter.eq('GAUL0_NAME', 'India'))
          .filter(ee.Filter.neq('GAUL1_NAME', 'Andaman & Nicobar'))
          .filter(ee.Filter.neq('GAUL1_NAME', 'Lakshadweep')))

    server_geom = fc.geometry(maxError=500).simplify(maxError=500)
    buffered = server_geom.buffer(50000, maxError=500)

    info = fc.getInfo()
    mainland = unary_union([shape(f['geometry']) for f in info['features']])

    to_sinu = Transformer.from_crs('EPSG:4326', SINU, always_xy=True).transform
    to_ea = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True).transform
    from_ea = Transformer.from_crs('EPSG:6933', 'EPSG:4326', always_xy=True).transform

    mainland_sinu = geom_transform(to_sinu, mainland)
    buffer_ll = geom_transform(from_ea, geom_transform(to_ea, mainland).buffer(50000))
    buffer_sinu = geom_transform(to_sinu, buffer_ll)

    return buffered, mainland_sinu, buffer_sinu, info


def compute_grid(buffer_sinu):
    """Snap the bounding rectangle to the MAIAC sinusoidal affine grid.

    Returns (width, height, transform_6tuple).
    """
    xmin, ymin, xmax, ymax = buffer_sinu.bounds
    col0 = math.floor((xmin - XORIGIN) / RES)
    row0 = math.floor((ymax - YORIGIN) / -RES)
    x0 = XORIGIN + col0 * RES
    y0 = YORIGIN - row0 * RES
    width = math.ceil((xmax - x0) / RES)
    height = math.ceil((y0 - ymin) / RES)
    return width, height, (RES, 0.0, x0, 0.0, -RES, y0)


def ee_transform_to_gdal(tf6):
    """Convert Earth Engine affine order to GDAL geotransform order."""
    return (tf6[2], tf6[0], tf6[1], tf6[5], tf6[3], tf6[4])


def _write_mask(group, name, values):
    """Store zero/one mask values with an unused xarray fill code."""
    values = np.asarray(values, dtype='uint8')
    if name in group and group[name].shape == values.shape:
        mask = group[name]
        if mask.fill_value == 255:
            mask[:] = values
            mask.attrs['_ARRAY_DIMENSIONS'] = ['y', 'x']
            return
        del group[name]
    elif name in group:
        del group[name]
    mask = group.create_dataset(name, data=values, chunks=(CHUNK, CHUNK),
                                dtype='uint8', fill_value=255,
                                compressor=COMPRESSOR)
    mask.attrs['_ARRAY_DIMENSIONS'] = ['y', 'x']


def _ensure_zero_is_data(group, name, dimensions):
    """Use an unused fill code and materialize zero-valued completion flags."""
    if name not in group:
        return
    old = group[name]
    old_dims = list(dimensions)
    if old.fill_value == 255:
        old.attrs['_ARRAY_DIMENSIONS'] = old_dims
        values = np.asarray(old[:])
        missing = values == 255
        if missing.any():
            values[missing] = 0
            old[:] = values
        return
    values = (np.asarray(old[:]) if old.fill_value is not None
              else np.zeros(old.shape, dtype=old.dtype))
    shape, chunks, dtype = old.shape, old.chunks, old.dtype
    del group[name]
    new = group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype,
                               fill_value=255, compressor=COMPRESSOR)
    new.attrs['_ARRAY_DIMENSIONS'] = old_dims
    new[:] = 0
    for index in np.argwhere(values != 0):
        key = tuple(int(i) for i in index)
        new[key] = values[key]


def _ensure_count_fill(group):
    """Use uint16 sentinel 65535; copy only committed AOD tiles in bounded chunks."""
    if 'accepted_count' not in group:
        return
    old = group['accepted_count']
    old.attrs['_ARRAY_DIMENSIONS'] = ['time', 'y', 'x']
    if old.dtype == np.dtype('uint16') and old.fill_value == 65535:
        return
    if 'complete' not in group:
        raise ValueError('Cannot migrate accepted_count without AOD completion flags')
    temp_name = 'accepted_count_fill_none_tmp'
    if temp_name in group:
        del group[temp_name]
    shape, chunks, dtype = old.shape, old.chunks, old.dtype
    trustworthy_counts = old.fill_value == 0
    if not trustworthy_counts:
        # A null Zarr fill value has undefined implicit chunk contents in v2.
        # Force refetch of all counts rather than trusting possibly repeated data.
        group['complete'][:] = 0
    new = group.create_dataset(temp_name, shape=shape, chunks=chunks, dtype='uint16',
                               fill_value=65535, compressor=COMPRESSOR)
    new.attrs['_ARRAY_DIMENSIONS'] = ['time', 'y', 'x']
    complete = group['complete']
    if trustworthy_counts:
        for day, row, col in np.argwhere(np.asarray(complete[:]) > 0):
            y0, x0 = int(row) * CHUNK, int(col) * CHUNK
            height = min(CHUNK, shape[1] - y0)
            width = min(CHUNK, shape[2] - x0)
            new[int(day), y0:y0 + height, x0:x0 + width] = \
                old[int(day), y0:y0 + height, x0:x0 + width]
    del group['accepted_count']
    group.move(temp_name, 'accepted_count')
    group['accepted_count'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'y', 'x']


def raster_mask(geometry, height, width, tf6):
    """Rasterise a Shapely geometry into a uint8 mask."""
    aff = Affine.from_gdal(tf6[2], tf6[0], tf6[1], tf6[5], tf6[3], tf6[4])
    return rasterize([(mapping(geometry), 1)], out_shape=(height, width),
                     transform=aff, fill=0, dtype='uint8', all_touched=False)


# ── Zarr Initialisation ────────────────────────────────────────────────────────

def init_static(mainland_sinu, buffer_sinu, boundary_info, width, height, tf6):
    """Create static.zarr with target and context masks."""
    DATA.mkdir(parents=True, exist_ok=True)

    target = raster_mask(mainland_sinu, height, width, tf6)
    context = raster_mask(buffer_sinu, height, width, tf6)

    static = zarr.open_group(str(DATA / 'static.zarr'), mode='a')
    for name, arr in [('target_mask', target), ('context_mask', context)]:
        _write_mask(static, name, arr)

    static.attrs.update({
        'crs': GEE_SINU, 'resolution_m': RES,
        'ee_crs_transform': list(tf6),
        'transform_gdal': list(ee_transform_to_gdal(tf6)),
    })

    geo_path = DATA / 'mainland_india.geojson'
    if not geo_path.exists():
        geo_path.write_text(json.dumps(boundary_info, separators=(',', ':')))

    return context


def init_year(year, width, height, tf6):
    """Create Zarr groups for one year of AOD, ERA5, and MERRA-2.

    Returns a dict of (dataset, year) -> zarr.Group for this year.
    """
    days = 366 if calendar.isleap(year) else 365
    stores = {}

    # ── AOD ──
    ag = zarr.open_group(str(DATA / 'aod.zarr'), mode='a').require_group(str(year))
    if 'aod' not in ag:
        ag.create_dataset('aod', shape=(days, height, width),
                          chunks=(1, CHUNK, CHUNK), dtype='int16',
                          fill_value=-32768, compressor=COMPRESSOR)
    if ag['aod'].shape != (days, height, width):
        raise ValueError(f'AOD shape mismatch in {year}: {ag["aod"].shape}')
    ag['aod'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'y', 'x']
    if 'accepted_count' not in ag:
        ag.create_dataset('accepted_count', shape=(days, height, width),
                          chunks=(1, CHUNK, CHUNK), dtype='uint16',
                          fill_value=65535, compressor=COMPRESSOR)
    if ag['accepted_count'].shape != (days, height, width):
        raise ValueError(f'AOD count shape mismatch in {year}: {ag["accepted_count"].shape}')
    ag['accepted_count'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'y', 'x']
    if 'complete' not in ag:
        ncy = math.ceil(height / CHUNK)
        ncx = math.ceil(width / CHUNK)
        ag.create_dataset('complete', shape=(days, ncy, ncx), chunks=(1, 1, 1),
                          dtype='uint8', fill_value=255, compressor=COMPRESSOR)
    if ag['complete'].shape != (days, math.ceil(height / CHUNK), math.ceil(width / CHUNK)):
        raise ValueError(f'AOD completion shape mismatch in {year}: {ag["complete"].shape}')
    _ensure_zero_is_data(ag, 'complete', ['time', 'chunk_y', 'chunk_x'])
    _ensure_count_fill(ag)
    _write_sinu_coords(ag, year, days, width, height, tf6)
    old_rule = ag.attrs.get('aod_rule')
    if old_rule is not None and old_rule != AOD_RULE:
        raise ValueError('AOD rule changed; refusing to resume this Zarr group')
    ag.attrs.update({'crs': GEE_SINU, 'resolution_m': RES,
                     'ee_crs_transform': list(tf6),
                     'transform_gdal': list(ee_transform_to_gdal(tf6)),
                     'aod_scale': 0.001, 'aod_fill': -32768,
                     'aod_rule': AOD_RULE,
                     'collection_start': MAIAC_START.isoformat()})
    stores[('aod', year)] = ag

    # ── ERA5 ──
    eg = zarr.open_group(str(DATA / 'era5.zarr'), mode='a').require_group(str(year))
    if 'era5' not in eg:
        eg.create_dataset('era5', shape=(days, 6, 341, 321),
                          chunks=(1, 6, 128, 128), dtype='float32',
                          fill_value=-9999, compressor=COMPRESSOR)
        eg['era5'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'band', 'y', 'x']
    if eg['era5'].shape != (days, 6, 341, 321):
        raise ValueError(f'ERA5 shape mismatch in {year}: {eg["era5"].shape}')
    eg['era5'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'band', 'y', 'x']
    if 'complete' not in eg:
        eg.create_dataset('complete', shape=(days,), chunks=(1,),
                          dtype='uint8', fill_value=255, compressor=COMPRESSOR)
    _ensure_zero_is_data(eg, 'complete', ['time'])
    _write_ll_coords(eg, year, days, ERA5_OUT_TF, 321, 341, ERA5_BANDS)
    eg.attrs.update({'crs': 'EPSG:4326', 'bands': ERA5_BANDS,
                     'transform_gdal': ERA5_OUT_TF,
                     'native_transform_gdal': ERA5_OUT_TF})
    stores[('era5', year)] = eg

    # ── MERRA-2 ──
    mg = zarr.open_group(str(DATA / 'merra2.zarr'), mode='a').require_group(str(year))
    expected_merra_shape = (days, MERRA_ROWS, MERRA_COLS)
    if 'totexttau' in mg and mg['totexttau'].shape != expected_merra_shape:
        old_shape = mg['totexttau'].shape
        grid_tag = f'{old_shape[-2]}x{old_shape[-1]}'
        legacy_name = f'totexttau_legacy_{grid_tag}'
        if legacy_name in mg:
            raise ValueError(f'Cannot preserve old MERRA array; {legacy_name} already exists')
        mg.move('totexttau', legacy_name)
        legacy_y, legacy_x = f'legacy_y_{grid_tag}', f'legacy_x_{grid_tag}'
        mg[legacy_name].attrs['_ARRAY_DIMENSIONS'] = ['time', legacy_y, legacy_x]
        mg[legacy_name].attrs['grid_status'] = 'superseded: incorrect native-grid offset'
        mg.attrs[f'legacy_transform_gdal_{grid_tag}'] = mg.attrs.get('transform_gdal', mg.attrs.get('native_transform_gdal', []))
        for coord, legacy_coord, dim in [('x', legacy_x, legacy_x),
                                         ('y', legacy_y, legacy_y)]:
            if coord in mg:
                if legacy_coord in mg:
                    raise ValueError(f'Cannot preserve old MERRA coordinate; {legacy_coord} already exists')
                mg.move(coord, legacy_coord)
                mg[legacy_coord].attrs['_ARRAY_DIMENSIONS'] = [dim]
        if 'complete' in mg:
            old_complete = f'complete_legacy_{grid_tag}'
            if old_complete in mg:
                raise ValueError(f'Cannot preserve old completion flags; {old_complete} already exists')
            mg.move('complete', old_complete)
            _ensure_zero_is_data(mg, old_complete, ['time'])
        print(f'Preserved superseded MERRA-2 array {year} {old_shape}; rebuilding on verified cell centres.', flush=True)
    if 'totexttau' not in mg:
        mg.create_dataset('totexttau', shape=expected_merra_shape,
                          chunks=(1, MERRA_ROWS, MERRA_COLS), dtype='float32',
                          fill_value=-9999, compressor=COMPRESSOR)
    if mg['totexttau'].shape != expected_merra_shape:
        raise ValueError(f'MERRA-2 shape mismatch in {year}: {mg["totexttau"].shape}')
    mg['totexttau'].attrs['_ARRAY_DIMENSIONS'] = ['time', 'y', 'x']
    if 'complete' not in mg:
        mg.create_dataset('complete', shape=(days,), chunks=(1,),
                          dtype='uint8', fill_value=255, compressor=COMPRESSOR)
    _ensure_zero_is_data(mg, 'complete', ['time'])
    for legacy_flag in [k for k in mg.array_keys() if k.startswith('complete_legacy')]:
        _ensure_zero_is_data(mg, legacy_flag, ['time'])
    _write_ll_coords(mg, year, days, MERRA_OUT_TF, MERRA_COLS, MERRA_ROWS)
    mg.attrs.update({'crs': 'EPSG:4326',
                     'transform_gdal': MERRA_OUT_TF,
                     'native_transform_gdal': MERRA_OUT_TF,
                     'source_grid_cell_centres': 'lon -180..179.375 by 0.625; lat -90..90 by 0.5'})
    stores[('merra2', year)] = mg

    return stores


def _epoch_days(year, n):
    """Time coordinate: days since 1970-01-01 for each day-of-year."""
    base = dt.date(year, 1, 1)
    epoch = dt.date(1970, 1, 1)
    return np.array([(base + dt.timedelta(i) - epoch).days
                     for i in range(n)], dtype='int32')


def _write_sinu_coords(g, year, days, width, height, tf6):
    """Write time/x/y for sinusoidal AOD grid."""
    if 'time' in g:
        return
    g.create_dataset('time', data=_epoch_days(year, days),
                     chunks=(days,), compressor=COMPRESSOR)
    g['time'].attrs.update({'units': 'days since 1970-01-01',
                            'calendar': 'proleptic_gregorian',
                            '_ARRAY_DIMENSIONS': ['time']})
    g.create_dataset('x', data=tf6[2] + (np.arange(width) + 0.5) * RES,
                     chunks=(min(width, 2048),), compressor=COMPRESSOR)
    g.create_dataset('y', data=tf6[5] - (np.arange(height) + 0.5) * RES,
                     chunks=(min(height, 2048),), compressor=COMPRESSOR)
    g['x'].attrs['_ARRAY_DIMENSIONS'] = ['x']
    g['y'].attrs['_ARRAY_DIMENSIONS'] = ['y']


def _write_ll_coords(g, year, days, out_tf, cols, rows, bands=None):
    """Write time/x/y (and optionally band) for lat-lon grids."""
    if 'time' not in g:
        g.create_dataset('time', data=_epoch_days(year, days),
                         chunks=(days,), compressor=COMPRESSOR)
        g['time'].attrs.update({'units': 'days since 1970-01-01',
                                'calendar': 'proleptic_gregorian',
                                '_ARRAY_DIMENSIONS': ['time']})
    if bands and 'band' not in g:
        g.create_dataset('band', data=np.array(bands, dtype='S32'),
                         chunks=(len(bands),), compressor=COMPRESSOR)
        g['band'].attrs['_ARRAY_DIMENSIONS'] = ['band']
    if 'x' not in g:
        g.create_dataset('x', data=out_tf[0] + (np.arange(cols) + 0.5) * out_tf[1],
                         chunks=(cols,), compressor=COMPRESSOR)
        g['x'].attrs['_ARRAY_DIMENSIONS'] = ['x']
    if 'y' not in g:
        g.create_dataset('y', data=out_tf[3] + (np.arange(rows) + 0.5) * out_tf[5],
                         chunks=(rows,), compressor=COMPRESSOR)
        g['y'].attrs['_ARRAY_DIMENSIONS'] = ['y']


# ── Download Helpers ────────────────────────────────────────────────────────────

def fetch_tif(image, region, path, *, crs=None, transform=None):
    """Download an Earth Engine image as GeoTIFF; return (array, gdal_transform)."""
    params = {'region': region, 'format': 'GEO_TIFF'}
    if crs:
        params['crs'] = crs
    if transform:
        params['crs_transform'] = list(transform)

    url = image.getDownloadURL(params)
    tmp = Path(str(path) + '.part')
    try:
        with urllib.request.urlopen(url, timeout=180) as resp, tmp.open('wb') as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        os.replace(tmp, path)
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', errors='replace')[:2000]
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'EE HTTP {e.code}: {detail}') from e
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    with rasterio.open(path) as ds:
        return ds.read(), list(ds.transform.to_gdal())


def doy(day):
    """Day-of-year index (0-based)."""
    return day.timetuple().tm_yday - 1


# ── AOD Tile Download ──────────────────────────────────────────────────────────

def aod_tiles(context):
    """Return list of (row, col, x_px, y_px, width_px, height_px) for non-empty tiles."""
    h, w = context.shape
    tiles = []
    for r, y in enumerate(range(0, h, CHUNK)):
        for c, x in enumerate(range(0, w, CHUNK)):
            hh, ww = min(CHUNK, h - y), min(CHUNK, w - x)
            if context[y:y + hh, x:x + ww].any():
                tiles.append((r, c, x, y, ww, hh))
    return tiles


def align_aod_array(arr, target_h, target_w, row_offset, col_offset):
    """Place a returned 2-band raster on the requested tile grid."""
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise ValueError(f'Expected two-band tile array, got {arr.shape}')
    aligned = np.empty((2, target_h, target_w), dtype=arr.dtype)
    aligned[0].fill(-32768)
    aligned[1].fill(0)
    sy, sx = max(0, row_offset), max(0, col_offset)
    dy, dx = max(0, -row_offset), max(0, -col_offset)
    ch = min(target_h - dy, arr.shape[1] - sy)
    cw = min(target_w - dx, arr.shape[2] - sx)
    if ch > 0 and cw > 0:
        aligned[:, dy:dy + ch, dx:dx + cw] = arr[:, sy:sy + ch, sx:sx + cw]
    return aligned


def download_aod_tile(day, grid, tile, context, stores, fetch_budget=None):
    """Download and write one AOD spatial chunk. Returns True if newly downloaded."""
    width, height, tf6 = grid
    r, c, x, y, ww, hh = tile
    group = stores[('aod', day.year)]
    idx = doy(day)

    if int(group['complete'][idx, r, c]):
        return False

    d0 = day.isoformat()
    d1 = (day + dt.timedelta(days=1)).isoformat()

    x0 = tf6[2] + x * RES
    y0 = tf6[5] - y * RES
    tile_tf = [RES, 0.0, x0, 0.0, -RES, y0]

    # Slight inset avoids Earth Engine adding a spurious edge pixel
    rect = ee.Geometry.Rectangle(
        [x0, y0 - hh * RES + 0.01, x0 + ww * RES - 0.01, y0],
        proj=GEE_SINU, geodesic=False)

    def qa_filter(img):
        q = img.select('AOD_QA')
        good = (q.bitwiseAnd(7).eq(1)                        # cloud clear
                .And(q.rightShift(3).bitwiseAnd(3).eq(0))     # land
                .And(q.rightShift(5).bitwiseAnd(7).eq(0))     # adjacency clear
                .And(q.rightShift(8).bitwiseAnd(15).eq(0)))   # best QA
        val = img.select('Optical_Depth_055')
        return val.updateMask(good.And(val.gte(0)).And(val.lte(3000)))

    col = (ee.ImageCollection(MAIAC)
           .filterDate(d0, d1).filterBounds(rect).map(qa_filter))

    # Empty sentinel ensures .mean()/.count() never operate on an empty collection
    empty = (ee.Image.constant(0).rename('Optical_Depth_055').toInt16()
             .updateMask(ee.Image.constant(0))
             .setDefaultProjection(ee.Projection(GEE_SINU,
                                                 [RES, 0, XORIGIN, 0, -RES, YORIGIN])))
    col = col.merge(ee.ImageCollection.fromImages([empty]))

    aod_img = col.mean().round().toInt16().unmask(-32768, False)
    cnt_img = col.count().min(255).toUint8().unmask(0, False)
    packed = ee.Image.cat([aod_img, cnt_img.toInt16()]).rename(['aod', 'count'])

    tif = TEMP / f'{day:%Y%m%d}_r{r:02d}c{c:02d}.tif'
    try:
        if fetch_budget is not None:
            fetch_budget.reserve()
        arr, gdal_tf = fetch_tif(packed, rect, tif, crs=GEE_SINU, transform=tile_tf)

        if arr.shape[0] != 2 or abs(arr.shape[1] - hh) > 1 or abs(arr.shape[2] - ww) > 1:
            raise ValueError(f'Tile {r},{c}: shape {arr.shape}, expected ~{(2, hh, ww)}')

        # Align only integral-pixel offsets; a fractional shift is a grid error.
        actual = np.asarray(gdal_tf, dtype='float64')
        col_offset = (x0 - actual[0]) / RES
        row_offset = (actual[3] - y0) / RES
        if not np.allclose([row_offset, col_offset],
                           np.rint([row_offset, col_offset]), atol=1e-4):
            raise ValueError(f'Tile {r},{c}: non-integral returned grid offset '
                             f'row={row_offset}, col={col_offset}')
        aligned = align_aod_array(arr, hh, ww,
                                  int(round(row_offset)), int(round(col_offset)))

        ad = np.rint(aligned[0]).astype('int16')
        cnt = np.clip(np.rint(aligned[1]), 0, 255).astype('uint8')

        # Zero out pixels outside the context mask
        ctx = context[y:y + hh, x:x + ww]
        ad[ctx == 0] = -32768
        cnt[ctx == 0] = 0

        group['aod'][idx, y:y + hh, x:x + ww] = ad
        group['accepted_count'][idx, y:y + hh, x:x + ww] = cnt
        group['complete'][idx, r, c] = 1
        return True
    finally:
        tif.unlink(missing_ok=True)


# ── ERA5-Land & MERRA-2 ────────────────────────────────────────────────────────

def download_era5(day, stores, fetch_budget=None):
    """Download ERA5-Land for one day. Returns True if newly downloaded."""
    group = stores[('era5', day.year)]
    idx = doy(day)
    if int(group['complete'][idx]):
        return False

    d0, d1 = day.isoformat(), (day + dt.timedelta(days=1)).isoformat()
    region = ee.Geometry.BBox(*INDIA_BBOX)
    collection = ee.ImageCollection(ERA5).filterDate(d0, d1)
    if collection.size().getInfo() == 0:
        raise RuntimeError(f'No ERA5-Land image for {day}')
    image = (collection.first().select(ERA5_BANDS)
             .toFloat().unmask(-9999, False))

    tif = TEMP / f'{day:%Y%m%d}_era5.tif'
    try:
        if fetch_budget is not None:
            fetch_budget.reserve()
        arr, gdal_tf = fetch_tif(image, region, tif,
                                 crs='EPSG:4326', transform=ERA5_NATIVE_TF)
        if arr.shape != (6, 341, 321):
            raise ValueError(f'ERA5 shape {arr.shape}, expected (6, 341, 321)')
        if not np.allclose(gdal_tf, ERA5_OUT_TF, atol=1e-7):
            raise ValueError(f'ERA5 transform mismatch: {gdal_tf}')

        group['era5'][idx] = arr.astype('float32')
        group['complete'][idx] = 1
        return True
    finally:
        tif.unlink(missing_ok=True)


def download_merra2(day, stores, fetch_budget=None):
    """Download MERRA-2 TOTEXTTAU for one day. Returns True if newly downloaded."""
    group = stores[('merra2', day.year)]
    idx = doy(day)
    if int(group['complete'][idx]):
        return False

    d0, d1 = day.isoformat(), (day + dt.timedelta(days=1)).isoformat()
    region = ee.Geometry.BBox(*INDIA_BBOX)

    collection = ee.ImageCollection(MERRA).filterDate(d0, d1)
    if collection.size().getInfo() == 0:
        raise RuntimeError(f'No MERRA-2 image for {day}')
    image = (collection.select('TOTEXTTAU').mean()
             .toFloat().unmask(-9999, False))

    tif = TEMP / f'{day:%Y%m%d}_merra2.tif'
    try:
        if fetch_budget is not None:
            fetch_budget.reserve()
        arr, gdal_tf = fetch_tif(image, region, tif,
                                 crs='EPSG:4326', transform=MERRA_NATIVE_TF)
        if arr.shape != (1, MERRA_ROWS, MERRA_COLS):
            raise ValueError(f'MERRA-2 shape {arr.shape}, expected '
                             f'(1, {MERRA_ROWS}, {MERRA_COLS})')
        if not np.allclose(gdal_tf, MERRA_OUT_TF, atol=1e-7):
            raise ValueError(f'MERRA-2 transform mismatch: {gdal_tf}')

        group['totexttau'][idx] = arr[0].astype('float32')
        group['complete'][idx] = 1
        return True
    finally:
        tif.unlink(missing_ok=True)


# ── Day Processing ──────────────────────────────────────────────────────────────

class FetchBudget:
    """Thread-safe cap on TIFF fetch jobs attempted for one day."""
    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.lock = threading.Lock()

    def reserve(self):
        with self.lock:
            if self.used >= self.limit:
                raise FetchLimitExceeded(
                    f'Daily TIFF fetch cap reached ({self.limit})')
            self.used += 1


class FetchLimitExceeded(RuntimeError):
    """The per-day TIFF transfer budget was exhausted."""


def process_day(day, boundary, grid, datasets, context, stores, tile_workers,
                on_aod_tile=None, max_fetches_per_day=32):
    """Download all selected datasets for one day, with retries."""
    fetch_budget = FetchBudget(max_fetches_per_day)
    last_err = None
    for attempt in range(5):
        try:
            _do_day(day, grid, datasets, context, stores,
                    tile_workers, on_aod_tile, fetch_budget)
            return fetch_budget.used
        except Exception as e:
            if isinstance(e, (FetchLimitExceeded, ValueError)):
                e.tiff_fetch_attempts = fetch_budget.used
                raise
            last_err = e
            if attempt < 4:
                time.sleep(min(60, 2 ** attempt) + random.random())
    last_err.tiff_fetch_attempts = fetch_budget.used
    raise last_err


def _do_day(day, grid, datasets, context, stores, tile_workers, on_aod_tile,
            fetch_budget=None):
    """Inner day download (no retries)."""
    if 'aod' in datasets:
        tiles = aod_tiles(context)
        with cf.ThreadPoolExecutor(max_workers=tile_workers) as pool:
            futs = [pool.submit(download_aod_tile, day, grid, t, context, stores,
                                fetch_budget)
                    for t in tiles]
            for f in cf.as_completed(futs):
                is_new = f.result()
                if on_aod_tile:
                    on_aod_tile(is_new)

    if 'era5' in datasets:
        download_era5(day, stores, fetch_budget)

    if 'merra2' in datasets:
        download_merra2(day, stores, fetch_budget)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_years(args):
    """Parse '2023', '2019-2023', or '2019 2020 2021' into sorted list."""
    years = set()
    for a in args:
        if '-' in a and not a.startswith('-'):
            lo, hi = a.split('-', 1)
            years.update(range(int(lo), int(hi) + 1))
        else:
            years.add(int(a))
    return sorted(years)


def build_dates(years, start, end):
    """Generate date list from years with optional start/end bounds."""
    lo = dt.date.fromisoformat(start) if start else dt.date(min(years), 1, 1)
    hi = dt.date.fromisoformat(end) if end else dt.date(max(years), 12, 31)
    dates = []
    for year in years:
        cur = max(lo, dt.date(year, 1, 1))
        last = min(hi, dt.date(year, 12, 31))
        while cur <= last:
            dates.append(cur)
            cur += dt.timedelta(days=1)
    return dates


def main():
    ap = argparse.ArgumentParser(
        description='Download MAIAC AOD, ERA5-Land & MERRA-2 for mainland India',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap())
    ap.add_argument('--years', nargs='+',
                    help='Years: 2023, 2019-2023, or 2000-2025')
    ap.add_argument('--start-date', help='Override start date (YYYY-MM-DD)')
    ap.add_argument('--end-date', help='Override end date (YYYY-MM-DD)')
    ap.add_argument('--datasets', nargs='+', default=['aod', 'era5', 'merra2'],
                    choices=['aod', 'era5', 'merra2'],
                    help='Which datasets to download (default: all)')
    ap.add_argument('--workers', type=int, default=4,
                    help='Parallel day-level workers (default: 4)')
    ap.add_argument('--tile-workers', type=int, default=9,
                    help='AOD tile workers per day (default: 9; 36 total with 4 day workers)')
    ap.add_argument('--max-fetches-per-day', type=int, default=32,
                    help='Maximum TIFF fetch attempts per day, including retries')
    ap.add_argument('--pilot', action='store_true',
                    help='Quick test: download 2023-01-15 only')
    ap.add_argument('--project', default=EE_PROJECT,
                    help='Earth Engine cloud project')
    args = ap.parse_args()

    # Handle pilot mode
    if args.pilot:
        if args.years:
            ap.error('--pilot cannot be combined with --years')
        years = [2023]
        args.start_date = args.end_date = '2023-01-15'
    else:
        if not args.years:
            ap.error('choose --pilot for the validated one-day run, or pass --years explicitly')
        years = parse_years(args.years)

    # Guard against exceeding Earth Engine's default 40 concurrent request quota
    if args.workers * args.tile_workers > 36:
        raise SystemExit('ERROR: workers × tile-workers must be ≤ 36 '
                         '(stay under EE 40-request quota)')
    if args.max_fetches_per_day < 1:
        raise SystemExit('ERROR: --max-fetches-per-day must be positive')

    # ── Initialise ──
    ee.Initialize(project=args.project, opt_url=EE_ENDPOINT)
    DATA.mkdir(parents=True, exist_ok=True)
    lock_file = (DATA / '.download.lock').open('a')
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise SystemExit(f'Another downloader is using {DATA}') from exc
    global TEMP
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP = Path(tempfile.mkdtemp(prefix=f'run-{os.getpid()}-', dir=TEMP_ROOT))

    print('Building India boundary and analysis grid...', flush=True)
    buffered, mainland_sinu, buffer_sinu, info = build_boundary()
    width, height, tf6 = compute_grid(buffer_sinu)
    context = init_static(mainland_sinu, buffer_sinu, info, width, height, tf6)

    print(f'Grid: {width}×{height} pixels | {len(years)} year(s): '
          f'{years[0]}–{years[-1]}', flush=True)

    stores = {}
    for year in years:
        stores.update(init_year(year, width, height, tf6))

    # Refresh consolidated metadata before work so an interrupted prior run
    # cannot leave xarray readers pointed at stale group/array metadata.
    for name in ['aod.zarr', 'era5.zarr', 'merra2.zarr', 'static.zarr']:
        p = DATA / name
        if p.exists():
            zarr.consolidate_metadata(str(p))

    dates = build_dates(years, args.start_date, args.end_date)
    if not dates:
        raise SystemExit('No dates selected')

    ds_set = set(args.datasets)
    tiles = aod_tiles(context) if 'aod' in ds_set else []
    total_aod = len(dates) * len(tiles)

    # Count already-completed AOD tiles for accurate progress
    done_aod = 0
    if tiles:
        for day in dates:
            g = stores[('aod', day.year)]
            idx = doy(day)
            done_aod += sum(int(g['complete'][idx, r, c]) for r, c, *_ in tiles)

    print(f'Downloading {len(dates)} days | datasets={args.datasets} | '
          f'workers={args.workers}×{args.tile_workers}', flush=True)
    if tiles:
        print(f'AOD tiles: {done_aod}/{total_aod} already done', flush=True)

    # ── Progress bars ──
    aod_bar = (tqdm(total=total_aod, initial=done_aod, desc='AOD tiles',
                    unit='tile', position=0, dynamic_ncols=True,
                    mininterval=0.5)
               if tiles else None)
    day_bar = tqdm(total=len(dates), desc='Days', unit='day',
                   position=1 if tiles else 0, dynamic_ncols=True,
                   mininterval=0.5)

    lock = threading.Lock()

    def on_tile(is_new):
        if aod_bar and is_new:
            with lock:
                aod_bar.update(1)

    # ── Download loop ──
    failed = []
    fetches_by_day = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = iter(dates)
        fut_map = {}

        # Prime the pool
        for _ in range(min(args.workers, len(dates))):
            d = next(pending)
            fut_map[pool.submit(process_day, d, buffered,
                                (width, height, tf6), ds_set, context,
                                stores, args.tile_workers, on_tile,
                                args.max_fetches_per_day)] = d

        while fut_map:
            done_futs, _ = cf.wait(fut_map, return_when=cf.FIRST_COMPLETED)
            for f in done_futs:
                d = fut_map.pop(f)
                try:
                    fetches_by_day[d.isoformat()] = f.result()
                    day_bar.update(1)
                    day_bar.set_postfix(date=d.isoformat())
                except Exception as e:
                    fetches_by_day[d.isoformat()] = getattr(
                        e, 'tiff_fetch_attempts', None)
                    failed.append((d.isoformat(), str(e)))
                    day_bar.update(1)
                    day_bar.set_postfix(date=d.isoformat(), status='FAIL')
                    tqdm.write(f'FAILED {d}: {e}')

                # Submit next day
                try:
                    nd = next(pending)
                    fut_map[pool.submit(
                        process_day, nd, buffered, (width, height, tf6),
                        ds_set, context, stores, args.tile_workers,
                        on_tile, args.max_fetches_per_day)] = nd
                except StopIteration:
                    pass

    day_bar.close()
    if aod_bar:
        aod_bar.close()

    # ── Finalise ──
    for name in ['aod.zarr', 'era5.zarr', 'merra2.zarr', 'static.zarr']:
        p = DATA / name
        if p.exists():
            zarr.consolidate_metadata(str(p))

    manifest = {
        'years': years,
        'datasets': args.datasets,
        'project': args.project,
        'aod_rule': AOD_RULE if 'aod' in ds_set else None,
        'grid': {
            'crs': GEE_SINU, 'width': width, 'height': height,
            'resolution_m': RES,
            'ee_crs_transform': list(tf6),
            'transform_gdal': list(ee_transform_to_gdal(tf6)),
        },
        'total_days': len(dates),
        'max_fetches_per_day': args.max_fetches_per_day,
        'tiff_fetch_attempts': fetches_by_day,
        'failed_days': len(failed),
        'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if failed:
        manifest['failed'] = failed
    manifest_tmp = DATA / 'manifest.json.part'
    manifest_tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(manifest_tmp, DATA / 'manifest.json')

    shutil.rmtree(TEMP, ignore_errors=True)

    if failed:
        print(f'\n{len(failed)} days failed (resumable — re-run same command).',
              flush=True)
    else:
        print(f'\nDone! {len(dates)} days downloaded to {DATA}', flush=True)


def textwrap():
    return """Examples:
  python download.py --pilot                   # One day (2023-01-15)
  python download.py --years 2023              # Explicit full-year run
  python download.py --years 2023 --datasets aod era5"""


if __name__ == '__main__':
    main()
