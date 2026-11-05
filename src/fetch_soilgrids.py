"""Step 4 - Soil properties per district from ISRIC SoilGrids 2.0 (250 m).

Reads each global SoilGrids layer straight from ISRIC's server (only the
Pakistan area is transferred), reprojects it to lat/lon at ~1 km with
average resampling, saves it, then averages it inside each district.
The 0-5, 5-15 and 15-30 cm layers are combined into one 0-30 cm value,
weighted by layer thickness.

Outputs: data/raw/soilgrids/{property}_{depth}.tif
         data/interim/soil_features.csv
             district_id, soil_ph, soil_oc, soil_clay, soil_sand, soil_nitrogen

Usage:  python -m src.fetch_soilgrids

Only real SoilGrids pixels inside each district are used (all_touched=True, so
every pixel a district touches counts). If any district has no real pixels, the
script stops with an error naming it and writes nothing.
"""

import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterstats import zonal_stats

VRT_URL = "https://files.isric.org/soilgrids/latest/data/{prop}/{prop}_{depth}_mean.vrt"

# SoilGrids stores scaled integers. Divide by the factor to get real units.
PROPERTIES = {
    # property: (output column, divide by, final unit)
    "phh2o": ("soil_ph", 10, "pH"),
    "soc": ("soil_oc", 10, "g/kg"),
    "clay": ("soil_clay", 10, "%"),
    "sand": ("soil_sand", 10, "%"),
    "nitrogen": ("soil_nitrogen", 100, "g/kg"),
}
# depth: thickness in cm (weights for the 0-30 cm average)
DEPTHS = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}

# Pakistan box, lon/lat, and output grid of 0.01 deg (~1 km)
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = 60.5, 23.5, 77.9, 37.2
RES = 0.01
SOIL_NODATA = -32768

POLYGONS_PATH = "data/interim/districts.gpkg"
RAW_DIR = "data/raw/soilgrids"
OUT_PATH = "data/interim/soil_features.csv"

GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  # don't list the server folder
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "5",
    "VSI_CACHE": "TRUE",
}


def target_grid():
    width = round((LON_MAX - LON_MIN) / RES)
    height = round((LAT_MAX - LAT_MIN) / RES)
    return from_origin(LON_MIN, LAT_MAX, RES, RES), width, height


def download_layer(prop, depth, raw_dir=RAW_DIR):
    out = os.path.join(raw_dir, f"{prop}_{depth}.tif")
    if os.path.exists(out):
        return out  # resume support
    transform, width, height = target_grid()
    url = "/vsicurl/" + VRT_URL.format(prop=prop, depth=depth)
    with (
        rasterio.Env(**GDAL_OPTIONS),
        rasterio.open(url) as src,
        WarpedVRT(
            src,
            crs="EPSG:4326",
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling.average,
            src_nodata=SOIL_NODATA,
            nodata=SOIL_NODATA,
        ) as vrt,
    ):
        data = vrt.read(1)
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "count": 1,
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": SOIL_NODATA,
        "compress": "deflate",
    }
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
    return out


def district_means(raster_path, districts):
    """Mean of the real SoilGrids pixels inside each district. No estimates or fills.

    all_touched=True counts every pixel the district touches (not only pixels whose
    centre is inside), so small or mostly-urban districts still get their own pixels.
    If a district still has no real pixels, the script stops with an error.
    """
    stats = zonal_stats(
        districts, raster_path, stats=["mean"], nodata=SOIL_NODATA, all_touched=True
    )
    vals = np.array(
        [s["mean"] if s["mean"] is not None else np.nan for s in stats], dtype=float
    )
    empty = districts.district_id[np.isnan(vals)].tolist()
    if empty:
        raise RuntimeError(
            f"No real SoilGrids pixels inside {empty} in {os.path.basename(raster_path)}. "
            "Nothing was written. Do not fill these with estimates; report it instead."
        )
    return vals


def depth_weighted(values_by_depth):
    """Thickness-weighted 0-30 cm mean. values_by_depth: {depth: array}."""
    total = sum(DEPTHS.values())
    return sum(values_by_depth[d] * (w / total) for d, w in DEPTHS.items())


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    districts = gpd.read_file(POLYGONS_PATH)
    out = districts[["district_id"]].copy()
    for prop, (col, scale, unit) in PROPERTIES.items():
        by_depth = {}
        for depth in DEPTHS:
            print(f"{prop} {depth} ...")
            path = download_layer(prop, depth)
            by_depth[depth] = district_means(path, districts)
        out[col] = (depth_weighted(by_depth) / scale).round(3)
        print(f"  {col} ({unit}): min {out[col].min()}, max {out[col].max()}")
    assert not out.isna().any().any(), (
        "Missing soil values (should have stopped earlier)"
    )
    out.to_csv(OUT_PATH, index=False)
    print(
        f"Wrote {OUT_PATH}: {len(out)} districts, all from real SoilGrids pixels, none missing"
    )


if __name__ == "__main__":
    main()
