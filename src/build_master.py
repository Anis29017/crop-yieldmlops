"""Step 7 - Merge yields, season weather and soil into one table.

Inputs:  data/raw/yield_labels.csv            (step 6)
         data/interim/climate_features.csv    (step 5)
         data/interim/soil_features.csv       (step 4)
Output:  data/raw/master.csv   one row per district x crop x year

prev_yield, prev2_yield, prev3_yield = the same district and crop's yield in
EXACTLY 1, 2 and 3 years before. If one of those years is missing (a gap in the
records), the value is left empty instead of borrowing another year;
prepare_data.py then drops the row.

Usage:  python -m src.build_master
"""

import pandas as pd

LABELS_PATH = "data/raw/yield_labels.csv"
CLIMATE_PATH = "data/interim/climate_features.csv"
SOIL_PATH = "data/interim/soil_features.csv"
OUT_PATH = "data/raw/master.csv"

KEY = ["district_id", "crop", "year"]


LAGS = {1: "prev_yield", 2: "prev2_yield", 3: "prev3_yield"}


def add_prev_yield(labels):
    """Yields from exactly year-1, year-2 and year-3 (exact-year joins, not 'the previous row')."""
    out = labels
    for lag, col in LAGS.items():
        prev = labels[KEY + ["yield_t_ha"]].copy()
        prev["year"] = prev["year"] + lag
        out = out.merge(prev.rename(columns={"yield_t_ha": col}), on=KEY, how="left")
    return out


def build(labels, climate, soil):
    for name, df, key in (("climate", climate, KEY), ("soil", soil, ["district_id"])):
        if df.duplicated(key).any():
            raise ValueError(f"Duplicate keys in {name} features")
    df = add_prev_yield(labels)
    df = df.merge(climate, on=KEY, how="left").merge(soil, on="district_id", how="left")
    report = {
        "yield rows": len(df),
        "no season weather": int(df["season_tmax_mean"].isna().sum()),
        "no season rain": int(df["season_rain_mm"].isna().sum()),
        "no soil": int(df["soil_ph"].isna().sum()),
        "no previous-year yield": int(df["prev_yield"].isna().sum()),
        "missing any of the 3 previous years": int(
            df[list(LAGS.values())].isna().any(axis=1).sum()
        ),
    }
    first = ["district_id", "province", "district", "crop", "year"]
    rest = [c for c in df.columns if c not in first + ["yield_t_ha"]]
    return df[first + rest + ["yield_t_ha"]].sort_values(KEY).reset_index(
        drop=True
    ), report


def main():
    labels = pd.read_csv(LABELS_PATH)
    climate = pd.read_csv(CLIMATE_PATH)
    soil = pd.read_csv(SOIL_PATH)
    df, report = build(labels, climate, soil)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_PATH}")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
