"""Step 3 - Monthly rainfall per district from CHIRPS v2.0 (~5 km grid), 1981-2025.

Why not the single global NetCDF?  It is several GB and the UCSB server is
slow, so the download times out.  Instead we ask for ONLY Pakistan's box:

  Source "iri" (default): IRI Data Library cuts CHIRPS to Pakistan on their
      server and sends a small NetCDF.  We ask one decade at a time
      (~40-50 MB each), so a failure only repeats one piece.
  Source "chc" (fallback): one GeoTIFF per month from UCSB (gzipped, global);
      each is cropped to Pakistan right after download and only the crop is kept.

Downloads are cached in data/raw/chirps/, so re-running resumes where it stopped.

Output: data/interim/chirps_monthly.csv   columns: district_id, year, month, rain_mm
        (rain_mm = district mean rainfall for that month, in mm)

Usage:  python -m src.fetch_chirps              # IRI (recommended)
        python -m src.fetch_chirps --source chc # fallback (needs rasterio)

Needs only: geopandas, xarray, scipy, h5netcdf, h5py, requests
(no rasterio / rasterstats / netCDF4, which often fail to install on new Python versions).
"""

import argparse
import gzip
import os
import shutil
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely

START_YEAR, END_YEAR = 1981, 2025
# Pakistan box (lon/lat) with a small margin
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 60.5, 78.0, 23.5, 37.5

IRI_URL = (
    "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/"
    "X/{x0}/{x1}/RANGEEDGES/Y/{y0}/{y1}/RANGEEDGES/T/(Jan%20{a})/(Dec%20{b})/RANGEEDGES/data.nc"
)
CHC_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.{y}.{m:02d}.tif.gz"

POLYGONS_PATH = "data/interim/districts.gpkg"
RAW_DIR = "data/raw/chirps"
OUT_PATH = "data/interim/chirps_monthly.csv"
NODATA = -9999.0


def download(url, path, timeout=600, tries=4):
    """Stream url to path (via a .part file, so a broken download is never mistaken for a finished one)."""
    if os.path.exists(path):
        return path
    tmp = path + ".part"
    for attempt in range(1, tries + 1):
        try:
            with requests.get(url, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    f.writelines(r.iter_content(chunk_size=1 << 20))
            os.replace(tmp, path)
            return path
        except (requests.RequestException, OSError) as e:
            print(f"    attempt {attempt}/{tries} failed: {e}")
            if attempt < tries:
                time.sleep(20 * attempt)
    raise RuntimeError(f"Could not download {url}")


# ---------- grids: every source is turned into (year, month, 2-D array, grid) ----------
# grid = (west edge, cell width, north edge, cell height); rows run north -> south


def grid_from_centres(lons, lats):
    """Grid for a regular lon/lat raster given cell-centre coordinates (lats must be descending)."""
    dx = float(lons[1] - lons[0])
    dy = float(lats[0] - lats[1])
    return (float(lons[0]) - dx / 2, dx, float(lats[0]) + dy / 2, dy)


def open_nc(xr, path):
    """Open a NetCDF file with whichever reader is installed.

    netCDF4 often fails to install on new Python versions (no ready-made wheel),
    so scipy (NetCDF3 files) and h5netcdf (NetCDF4 files) are tried as well.
    """
    errors = []
    for engine in ("netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(path, decode_times=False, engine=engine)
        except Exception as e:  # noqa: BLE001  (engine not installed or wrong file format)
            errors.append(f"{engine}: {e}")
    raise RuntimeError(
        f"Could not open {path}. Run: pip install scipy h5netcdf h5py\n"
        + "\n".join(errors)
    )


def iri_months(raw_dir):
    """Yield (year, month, array, grid) from IRI decade files."""
    import xarray as xr

    for a in range(START_YEAR, END_YEAR + 1, 10):
        b = min(a + 9, END_YEAR)
        path = os.path.join(raw_dir, f"chirps_pak_{a}_{b}.nc")
        url = IRI_URL.format(x0=LON_MIN, x1=LON_MAX, y0=LAT_MIN, y1=LAT_MAX, a=a, b=b)
        print(
            f"  {a}-{b}: downloading"
            if not os.path.exists(path)
            else f"  {a}-{b}: cached"
        )
        download(url, path)
        # IRI time is "months since 1960-01-01" (360-day calendar), so decode it ourselves
        ds = open_nc(xr, path)
        var = "precipitation" if "precipitation" in ds else next(iter(ds.data_vars))
        da = ds[var]
        lon_name = "X" if "X" in da.dims else "lon"
        lat_name = "Y" if "Y" in da.dims else "lat"
        time_name = "T" if "T" in da.dims else "time"
        da = da.sortby(lat_name, ascending=False)  # north to south, as rasters expect
        grid = grid_from_centres(da[lon_name].values, da[lat_name].values)
        for t in da[time_name].values:
            k = int(np.floor(float(t)))
            year, month = 1960 + k // 12, k % 12 + 1
            arr = da.sel({time_name: t}).values.astype("float64")
            arr[~np.isfinite(arr)] = NODATA
            arr[arr < 0] = NODATA
            yield year, month, arr, grid
        ds.close()


def chc_months(raw_dir):
    """Yield (year, month, array, grid) from UCSB monthly GeoTIFFs, cropped to Pakistan."""
    import rasterio
    from rasterio.windows import from_bounds

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            crop = os.path.join(raw_dir, f"chirps_pak_{year}_{month:02d}.npz")
            if not os.path.exists(crop):
                gz = os.path.join(raw_dir, f"tmp_{year}_{month:02d}.tif.gz")
                try:
                    download(CHC_URL.format(y=year, m=month), gz)
                except RuntimeError as e:
                    print(f"  {year}-{month:02d}: skipped ({e})")
                    continue
                tif = gz[:-3]
                with gzip.open(gz, "rb") as fi, open(tif, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                with rasterio.open(tif) as src:
                    win = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
                    arr = src.read(1, window=win).astype("float64")
                    t = src.window_transform(win)
                    grid = np.array([t.c, t.a, t.f, -t.e])
                    nod = src.nodata if src.nodata is not None else NODATA
                arr[(arr == nod) | (arr < 0)] = NODATA
                np.savez_compressed(crop, arr=arr, grid=grid)
                os.remove(gz)
                os.remove(tif)
                print(f"  {year}-{month:02d}: done")
            z = np.load(crop)
            yield year, month, z["arr"], tuple(float(v) for v in z["grid"])


# ---------- district means ----------


def district_cells(districts, shape, grid):
    """For each district, the flat indices of the grid cells whose centre lies inside it.

    A district smaller than one cell (none of its centres inside) gets the single
    cell under its representative point, so no district is left empty.
    """
    x0, dx, y0, dy = grid
    nrow, ncol = shape
    xs = x0 + (np.arange(ncol) + 0.5) * dx
    ys = y0 - (np.arange(nrow) + 0.5) * dy
    cells = []
    for geom in districts.geometry:
        minx, miny, maxx, maxy = geom.bounds
        c0, c1 = max(int((minx - x0) / dx) - 1, 0), min(int((maxx - x0) / dx) + 2, ncol)
        r0, r1 = max(int((y0 - maxy) / dy) - 1, 0), min(int((y0 - miny) / dy) + 2, nrow)
        X, Y = np.meshgrid(xs[c0:c1], ys[r0:r1])
        inside = shapely.contains_xy(geom, X, Y)
        rr, cc = np.nonzero(inside)
        idx = (rr + r0) * ncol + (cc + c0)
        if idx.size == 0:
            pt = geom.representative_point()
            r = min(max(int((y0 - pt.y) / dy), 0), nrow - 1)
            c = min(max(int((pt.x - x0) / dx), 0), ncol - 1)
            idx = np.array([r * ncol + c])
        cells.append(idx)
    return cells


def district_rows(months_iter, districts):
    rows, cache = [], {}
    ids = list(districts.district_id)
    for year, month, arr, grid in months_iter:
        key = (arr.shape, tuple(round(v, 6) for v in grid))
        if key not in cache:  # same grid every month: work out the cells once
            cache[key] = district_cells(districts, arr.shape, grid)
        flat = np.where(arr.ravel() == NODATA, np.nan, arr.ravel())
        for did, idx in zip(ids, cache[key]):
            vals = flat[idx]
            mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
            rows.append(
                {
                    "district_id": did,
                    "year": year,
                    "month": month,
                    "rain_mm": None if mean is None else round(mean, 2),
                }
            )
    return pd.DataFrame(rows)


def check(df, n_districts):
    """Fail loudly instead of silently writing a broken file."""
    assert len(df) > 0, "No rainfall rows were produced"
    per_month = df.groupby(["year", "month"]).size()
    assert (per_month == n_districts).all(), "Some months are missing districts"
    missing = df.rain_mm.isna().sum()
    yrs = sorted(df.year.unique())
    annual = df.groupby(["district_id", "year"]).rain_mm.sum().groupby("year").mean()
    print(
        f"Years {yrs[0]}-{yrs[-1]} ({len(yrs)} years), {len(per_month)} months, {missing} missing values"
    )
    print(
        "Average annual rain over all districts (mm), last 10 years:",
        annual.tail(10).round(0).to_dict(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["iri", "chc"], default="iri")
    args = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    districts = gpd.read_file(POLYGONS_PATH).to_crs(epsg=4326)
    print(
        f"CHIRPS from {args.source.upper()}, {START_YEAR}-{END_YEAR}, {len(districts)} districts"
    )
    it = iri_months(RAW_DIR) if args.source == "iri" else chc_months(RAW_DIR)
    df = district_rows(it, districts)
    check(df, len(districts))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
