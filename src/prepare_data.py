"""Step 8 - Data preparation (separate from training, as required for M1).

Reads the merged master table (district x crop x year) made by
src/build_master.py. If it does not exist yet, a synthetic placeholder with
the SAME columns is generated so training, the API and tests can still run.

Cleaning (each step prints how many rows it removed and why):
  1. rows with any missing feature or target (e.g. the first 3 years of a series
     have no 3 previous years, or a year is missing in the records)
  2. rows whose yield is outside a plausible range for the crop
Then validation checks the result and stops on any remaining problem.

Output: data/processed.csv
"""

import os

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/master.csv"
OUT_PATH = "data/processed.csv"

CROPS = ["wheat", "cotton", "rice", "sugarcane"]
PROVINCES = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad"]

FEATURES = [
    "crop",
    "province",
    "year",
    "season_tmax_mean",  # mean daily max temp over the crop's season (C)
    "season_rain_mm",  # total rainfall over the crop's season (mm, CHIRPS)
    "heat_days_40c",  # days above 40C in the season
    "rain_anomaly",  # season rain minus district mean of training years (mm)
    "soil_ph",
    "soil_oc",  # organic carbon (g/kg)
    "soil_clay",  # %
    "soil_sand",  # %
    "soil_nitrogen",  # g/kg
    "prev_yield",  # same district & crop, exactly 1 year before (t/ha)
    "prev2_yield",  # exactly 2 years before (t/ha)
    "prev3_yield",  # exactly 3 years before (t/ha)
    "trend_ratio",  # prev_yield / prev2_yield: is yield going up or down lately?
]
TARGET = "yield_t_ha"
DERIVED = ["trend_ratio"]  # computed here from other columns (see add_derived)
ID_COLS = ["district_id", "district"]
KEY = ["district_id", "crop", "year"]
REQUIRED_COLS = ID_COLS + FEATURES + [TARGET]

# Plausible yield ranges (t/ha). Rows outside are dropped (and reported), not fixed.
YIELD_BOUNDS = {
    "wheat": (0.3, 6.0),
    "cotton": (0.1, 4.0),
    "rice": (0.3, 6.0),
    "sugarcane": (10.0, 120.0),
}

# Typical mean yields (t/ha), used only by the synthetic generator.
_BASE_YIELD = {"wheat": 3.0, "cotton": 0.8, "rice": 2.5, "sugarcane": 60.0}


def make_synthetic(n_districts=40, start_year=2000, end_year=2023, seed=42):
    """Synthetic placeholder with the real columns. Includes a 2010 and 2022
    flood shock so the drift experiment in M2 has something to detect."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_districts):
        province = PROVINCES[d % 4]
        soil = {
            "soil_ph": rng.uniform(7.5, 8.6),
            "soil_oc": rng.uniform(2.0, 9.0),
            "soil_clay": rng.uniform(10, 40),
            "soil_sand": rng.uniform(20, 60),
            "soil_nitrogen": rng.uniform(0.3, 1.2),
        }
        for crop in CROPS:
            prev = prev2 = prev3 = _BASE_YIELD[crop]
            for year in range(start_year, end_year + 1):
                shock = year in (2010, 2022)
                tmax = (
                    rng.normal(34, 1.5) + (year - start_year) * 0.02
                )  # slow warming, in C
                rain = rng.normal(250, 60) + (400 if shock else 0)
                heat = max(0, int(rng.normal(12, 4) + (year - start_year) * 0.2))
                effect = (
                    1
                    - 0.01 * heat
                    - (0.35 if shock else 0)
                    + 0.02 * soil["soil_oc"] / 9
                    + rng.normal(0, 0.05)
                )
                yld = max(_BASE_YIELD[crop] * effect, YIELD_BOUNDS[crop][0] + 0.01)
                rows.append(
                    {
                        "district_id": f"SYN_{d:02d}",
                        "district": f"district_{d:02d}",
                        "province": province,
                        "crop": crop,
                        "year": year,
                        "season_tmax_mean": round(tmax, 2),
                        "season_rain_mm": round(max(rain, 0), 1),
                        "heat_days_40c": heat,
                        "rain_anomaly": round(rain - 250, 1),
                        **{k: round(v, 3) for k, v in soil.items()},
                        "prev_yield": round(prev, 3),
                        "prev2_yield": round(prev2, 3),
                        "prev3_yield": round(prev3, 3),
                        TARGET: round(yld, 3),
                    }
                )
                prev3, prev2, prev = prev2, prev, yld
    return add_derived(pd.DataFrame(rows))


def add_derived(df):
    """Columns computed from others. Used by prepare_data AND the API, so both match."""
    df = df.copy()
    df["trend_ratio"] = df["prev_yield"] / df["prev2_yield"]
    return df


def drop_incomplete(df):
    """Drop rows with any missing required value, reporting which column caused it."""
    missing = df[REQUIRED_COLS].isna()
    bad = missing.any(axis=1)
    if bad.any():
        counts = missing[bad].sum()
        print(
            f"Dropped {int(bad.sum())} rows with missing values: {counts[counts > 0].to_dict()}"
        )
    return df[~bad].copy()


def drop_missing_prev_yield(df):
    """Kept for compatibility: drop rows with no prev_yield (first year of a series)."""
    before = len(df)
    df = df.dropna(subset=["prev_yield"])
    if before - len(df):
        print(
            f"Dropped {before - len(df)} rows with no prev_yield (first year of a series)"
        )
    return df


def drop_implausible(df):
    """Drop rows whose yield is outside YIELD_BOUNDS for their crop, and report them."""
    keep = pd.Series(True, index=df.index)
    for crop, (lo, hi) in YIELD_BOUNDS.items():
        m = (df["crop"] == crop) & ((df[TARGET] < lo) | (df[TARGET] > hi))
        if m.any():
            print(
                f"Dropped {int(m.sum())} {crop} rows with yield outside [{lo}, {hi}] t/ha"
            )
        keep &= ~m
    return df[keep].copy()


def validate(df):
    """Validation rules. Raises AssertionError with a clear message."""
    missing = set(REQUIRED_COLS) - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert len(df) > 0, "Empty dataset"
    assert df[REQUIRED_COLS].isnull().sum().sum() == 0, "Missing values found"
    assert not df.duplicated(KEY).any(), "Duplicate district x year x crop rows"
    assert set(df["crop"]) <= set(CROPS), f"Unknown crop in {set(df['crop'])}"
    assert set(df["province"]) <= set(PROVINCES), (
        f"Unknown province in {set(df['province']) - set(PROVINCES)}"
    )
    for crop, (lo, hi) in YIELD_BOUNDS.items():
        y = df.loc[df["crop"] == crop, TARGET]
        bad = y[(y < lo) | (y > hi)]
        assert bad.empty, f"{len(bad)} {crop} yields outside [{lo}, {hi}] t/ha"
    return True


def prepare_data(raw_path=RAW_PATH, out_path=OUT_PATH):
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        source = raw_path
    else:
        df = make_synthetic()
        source = "SYNTHETIC placeholder (run build_yield_labels + build_master for real data)"
    print(f"Read {len(df)} rows from {source}")
    df = drop_implausible(drop_incomplete(add_derived(df)))[REQUIRED_COLS]
    validate(df)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Data ready: {len(df)} rows -> {out_path}")
    print(df.groupby("crop").size().to_dict())
    return df


if __name__ == "__main__":
    prepare_data()
