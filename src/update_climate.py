"""DataOps - add the newest finished crop seasons to data/interim/climate_features.csv.

The full climate table (1981-2025) was built once in M1 by fetch_nasa_power, fetch_chirps
and build_climate_features. This step only adds what is new, so it is quick enough to run
every month in GitHub Actions:

  1. From climate_features.csv, find the next season of each crop that is not in it yet
     (wheat 2026 = Nov 2025 - Apr 2026; cotton / rice / sugarcane 2026 = May - Oct 2026).
  2. Download only the months since then, up to the end of last month:
       NASA POWER  daily temperature, humidity, sunlight  (one request per district point)
       CHIRPS      monthly rainfall                        (one small file for Pakistan)
  3. Build the season features with the same code as build_climate_features. The rain
     anomaly uses the same baseline (each district's mean season rain up to 2017), taken
     from the rows already in the table.
  4. Append a season only when it is finished and complete: every month has data from both
     sources, for every district. Otherwise nothing is added and the report says what is
     still missing (CHIRPS is published a few weeks after the month ends).

Rows already in the table are never changed.

Usage:  python -m src.update_climate              (writes data/interim/climate_features.csv)
        python -m src.update_climate --report r.md (also appends a summary to r.md)
"""

import argparse
import calendar
import os
import sys
import tempfile
import time
from datetime import date

import geopandas as gpd
import pandas as pd
import requests

from src import build_climate_features as bcf
from src import fetch_chirps as chirps_src
from src import fetch_nasa_power as power_src

CLIMATE_PATH = "data/interim/climate_features.csv"
POINTS_PATH = "data/interim/district_points.csv"
POLYGONS_PATH = "data/interim/districts.gpkg"
MIN_DAYS_SHARE = 0.9  # a month counts only if NASA POWER has data for at least 90% of its days
CHIRPS_TRIES = 4  # if the newest month is not published yet, try up to 3 months earlier

IRI_RANGE_URL = (
    "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/"
    "X/{x0}/{x1}/RANGEEDGES/Y/{y0}/{y1}/RANGEEDGES/T/({m0}%20{a})/({m1}%20{b})/RANGEEDGES/data.nc"
)


def month_end(year, month):
    return date(year, month, calendar.monthrange(year, month)[1])


def last_finished_month(today):
    first = today.replace(day=1)
    prev = first.replace(year=first.year - 1, month=12) if first.month == 1 else first.replace(month=first.month - 1)
    return prev.year, prev.month


def next_seasons(climate):
    """{crop: next harvest-year label} after the newest year in the table."""
    return {crop: int(climate.loc[climate["crop"] == crop, "year"].max()) + 1 for crop in bcf.CROP_SEASON}


def season_months(crop, year):
    """[(year, month), ...] of a crop's season with harvest-year label `year`."""
    season = bcf.CROP_SEASON[crop]
    return [(year - 1 if (season == "rabi" and m >= 11) else year, m) for m in bcf.SEASON_MONTHS[season]]


def window(climate, today):
    """First and last day to download: from the first month of the earliest missing season
    to the end of last month. None if no missing season has started yet."""
    starts = [season_months(c, y)[0] for c, y in next_seasons(climate).items()]
    y0, m0 = min(starts)
    y1, m1 = last_finished_month(today)
    if (y0, m0) > (y1, m1):
        return None
    return date(y0, m0, 1), month_end(y1, m1)


# ---------- downloads ----------

def fetch_power(points, start, end, session=None, pause=power_src.PAUSE_SECONDS):
    """Daily NASA POWER rows for every district point between start and end."""
    session = session or requests.Session()
    frames, failed = [], []
    for i, row in enumerate(points.itertuples(), 1):
        print(f"  POWER [{i}/{len(points)}] {row.district}")
        try:
            df = power_src.fetch_point(row.lat, row.lon, session, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        except RuntimeError as e:  # that district's seasons then count as incomplete: nothing is added for them
            print(f"    failed: {e}")
            failed.append(row.district_id)
            continue
        df.insert(0, "district_id", row.district_id)
        frames.append(df)
        time.sleep(pause)
    if failed:
        print(f"  NASA POWER failed for {len(failed)} districts: {failed}")
    columns = ["district_id", "date"] + power_src.PARAMETERS
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def complete_power_months(power):
    """Keep only months where NASA POWER has data for nearly every day (recent days can be missing)."""
    p = power.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["ym"] = p["date"].dt.to_period("M")
    share = p.groupby(["district_id", "ym"])["T2M_MAX"].apply(lambda s: s.notna().mean())
    good = share[share >= MIN_DAYS_SHARE].index
    p = p.set_index(["district_id", "ym"]).loc[lambda d: d.index.isin(good)].reset_index()
    return p.drop(columns="ym")


def fetch_chirps(start, end, districts, folder):
    """Monthly CHIRPS rain per district from start to end; if the last months are not
    published yet, the request is retried ending one month earlier (up to CHIRPS_TRIES times)."""
    y1, m1 = end.year, end.month
    for _ in range(CHIRPS_TRIES):
        url = IRI_RANGE_URL.format(x0=chirps_src.LON_MIN, x1=chirps_src.LON_MAX, y0=chirps_src.LAT_MIN,
                                   y1=chirps_src.LAT_MAX, m0=calendar.month_abbr[start.month], a=start.year,
                                   m1=calendar.month_abbr[m1], b=y1)
        path = os.path.join(folder, f"chirps_{start:%Y%m}_{y1}{m1:02d}.nc")
        try:
            chirps_src.download(url, path, timeout=300, tries=2)
            return chirps_src.district_rows(_months_from_nc(path), districts)
        except RuntimeError as e:
            print(f"  CHIRPS up to {calendar.month_abbr[m1]} {y1} not available ({e}); trying one month earlier")
            y1, m1 = (y1 - 1, 12) if m1 == 1 else (y1, m1 - 1)
            if (y1, m1) < (start.year, start.month):
                break
    return pd.DataFrame(columns=["district_id", "year", "month", "rain_mm"])


def _months_from_nc(path):
    """(year, month, array, grid) for each month in an IRI CHIRPS NetCDF file."""
    import numpy as np
    import xarray as xr

    ds = chirps_src.open_nc(xr, path)
    var = "precipitation" if "precipitation" in ds else next(iter(ds.data_vars))
    da = ds[var]
    lon_name = "X" if "X" in da.dims else "lon"
    lat_name = "Y" if "Y" in da.dims else "lat"
    time_name = "T" if "T" in da.dims else "time"
    da = da.sortby(lat_name, ascending=False)
    grid = chirps_src.grid_from_centres(da[lon_name].values, da[lat_name].values)
    for t in da[time_name].values:
        k = int(np.floor(float(t)))
        arr = da.sel({time_name: t}).values.astype("float64")
        arr[~np.isfinite(arr)] = chirps_src.NODATA
        arr[arr < 0] = chirps_src.NODATA
        yield 1960 + k // 12, k % 12 + 1, arr, grid
    ds.close()


# ---------- new season rows ----------

def new_rows(power, chirps, climate):
    """Season features for finished seasons not yet in the table, and a list of what was not added.

    A crop-year is added only if every district in the table has a complete season.
    """
    wanted = next_seasons(climate)
    districts = set(climate["district_id"])
    frames, notes = [], []
    for season in bcf.SEASON_MONTHS:
        p = bcf.power_season_features(power, season)
        r = bcf.chirps_season_rain(chirps, season) if len(chirps) else \
            pd.DataFrame(columns=["district_id", "season_year", "season_rain_mm"])
        f = p.merge(r, on=["district_id", "season_year"], how="inner")
        f = f.drop(columns="power_rain_mm").rename(columns={"season_year": "year"})
        for crop, crop_season in bcf.CROP_SEASON.items():
            if crop_season != season:
                continue
            year = wanted[crop]
            got = f[f["year"] == year].assign(crop=crop)
            missing = districts - set(got["district_id"])
            if missing:
                have_power = set(p.loc[p["season_year"] == year, "district_id"])
                why = ("season not finished, or NASA POWER incomplete" if len(have_power) < len(districts)
                       else "CHIRPS rainfall not published yet for every month")
                notes.append(f"{crop} {year}: not added ({why}; {len(missing)} of {len(districts)} districts missing)")
                continue
            frames.append(got[got["district_id"].isin(districts)])
            notes.append(f"{crop} {year}: added for all {len(districts)} districts")
    if not frames:
        return pd.DataFrame(columns=climate.columns), notes
    new = pd.concat(frames, ignore_index=True)
    base = (climate[climate["year"] <= bcf.BASELINE_LAST_YEAR]
            .groupby(["district_id", "crop"])["season_rain_mm"].mean().rename("rain_baseline"))
    new = new.join(base, on=["district_id", "crop"])
    new["rain_anomaly"] = new["season_rain_mm"] - new["rain_baseline"]
    return new[list(climate.columns)].round(3), notes


def append(climate, new):
    """Existing rows unchanged; only crop-years not already present are added."""
    have = set(zip(climate["crop"], climate["year"]))
    new = new[[(c, y) not in have for c, y in zip(new["crop"], new["year"])]]
    out = pd.concat([climate, new], ignore_index=True)
    return out.sort_values(["district_id", "crop", "year"]).reset_index(drop=True)


def main(argv=None, today=None):
    ap = argparse.ArgumentParser(description="Add the newest finished crop seasons to climate_features.csv")
    ap.add_argument("--report", help="markdown file to append a summary to (e.g. the pull request text)")
    args = ap.parse_args(argv)
    climate = pd.read_csv(CLIMATE_PATH)
    span = window(climate, today or date.today())
    lines = ["## Climate data (NASA POWER + CHIRPS)", ""]
    if span is None:
        lines.append("No new season has started since the last update: nothing to download.")
    else:
        start, end = span
        print(f"Downloading {start} to {end}")
        points = pd.read_csv(POINTS_PATH)
        power = complete_power_months(fetch_power(points, start, end))
        districts = gpd.read_file(POLYGONS_PATH).to_crs(epsg=4326)
        with tempfile.TemporaryDirectory() as tmp:
            rain = fetch_chirps(start, end, districts, tmp)
        new, notes = new_rows(power, rain, climate)
        updated = append(climate, new)
        added = len(updated) - len(climate)
        if added:
            updated.to_csv(CLIMATE_PATH, index=False)
        lines.append(f"Downloaded {start} to {end}. Rows added: {added}.")
        lines += [f"- {n}" for n in notes]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.report:
        with open(args.report, "a") as f:
            f.write("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
