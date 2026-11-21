"""Step 8 - Data preparation (separate from training, as required for M1).

Reads the merged master table (district x crop x year) made by
src/build_master.py. If that file does not exist, the script stops with an
error: it never generates or substitutes data.

Cleaning (each step prints how many rows it removed and why):
  1. rows with any missing feature or target (e.g. the first 3 years of a series
     have no 3 previous years, or a year is missing in the records)
  2. rows whose yield is outside a plausible range for the crop
Then validation checks the result and stops on any remaining problem.

Output: data/processed.csv
"""

import os

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
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found. Build it from the real data first: "
            "python -m src.build_yield_labels && python -m src.build_master "
            "(or run everything with python -m src.run_pipeline)."
        )
    df = pd.read_csv(raw_path)
    print(f"Read {len(df)} rows from {raw_path}")
    df = drop_implausible(drop_incomplete(add_derived(df)))[REQUIRED_COLS]
    validate(df)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Data ready: {len(df)} rows -> {out_path}")
    print(df.groupby("crop").size().to_dict())
    return df


if __name__ == "__main__":
    prepare_data()
