"""Step 2 - Daily weather per district from the NASA POWER API.

One request per district point (no API key needed). Each district is saved
to its own CSV so the script can be stopped and re-run: finished districts
are skipped. At the end all districts are combined into one file.

Output: data/interim/power_daily.csv
        columns: district_id, date, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, ALLSKY_SFC_SW_DWN

Usage:  python -m src.fetch_nasa_power
"""

import glob
import os
import time

import pandas as pd
import requests

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"]
START = "19810101"  # Rabi 2000 wheat season starts Nov 1999
END = "20251231"
FILL_VALUE = -999  # POWER's code for "no data"

POINTS_PATH = "data/interim/district_points.csv"
RAW_DIR = "data/raw/nasa_power"
OUT_PATH = "data/interim/power_daily.csv"

PAUSE_SECONDS = 1  # be polite to the free API
MAX_RETRIES = 4


def build_params(lat, lon, start=START, end=END):
    return {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",  # agriculture community: units suited to crop work
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
    }


def parse_power_json(payload):
    """Turn POWER's {param: {YYYYMMDD: value}} JSON into a daily DataFrame."""
    data = payload["properties"]["parameter"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.replace(FILL_VALUE, float("nan"))
    return df.reset_index()


def fetch_point(lat, lon, session=None):
    """Request one point, retrying with growing waits on errors or rate limits."""
    session = session or requests.Session()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(API_URL, params=build_params(lat, lon), timeout=180)
            if r.status_code == 429:
                raise requests.HTTPError("rate limited (429)")
            r.raise_for_status()
            return parse_power_json(r.json())
        except (requests.RequestException, KeyError, ValueError) as e:
            wait = 10 * attempt
            print(f"  attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts for ({lat}, {lon})")


def combine(raw_dir=RAW_DIR, out_path=OUT_PATH):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    df = pd.concat(
        [pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True
    )
    df.to_csv(out_path, index=False)
    print(f"Combined {len(files)} districts, {len(df)} rows -> {out_path}")
    return df


def main():
    points = pd.read_csv(POINTS_PATH)
    os.makedirs(RAW_DIR, exist_ok=True)
    session = requests.Session()
    failed = []

    for i, row in points.iterrows():
        out = os.path.join(RAW_DIR, f"{row.district_id}.csv")
        if os.path.exists(out):
            continue  # already fetched: lets you resume after an interruption
        print(f"[{i + 1}/{len(points)}] {row.district} ({row.lat}, {row.lon})")
        try:
            df = fetch_point(row.lat, row.lon, session)
        except RuntimeError as e:
            print(f"  SKIPPED: {e}")
            failed.append(row.district_id)
            continue
        df.insert(0, "district_id", row.district_id)
        df.to_csv(out, index=False)
        time.sleep(PAUSE_SECONDS)

    if failed:
        print(f"{len(failed)} districts failed, re-run the script to retry: {failed}")
    combine()


if __name__ == "__main__":
    main()
