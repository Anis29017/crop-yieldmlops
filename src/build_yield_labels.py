"""Step 6 - Yield labels: district crop yields with the district_id used by the weather/soil files.

Input:   data/raw/Pakistan_Master_Yield_1981-2025.xlsx  (sheet "All Data": every usable
         district x crop x year yield, compiled from PBS, provincial Crop Reporting /
         Development Statistics books and MNFSR; see that workbook's Notes sheet)
         data/sources/new_yields/*.csv  (rows read from new books by the DataOps pipeline,
         added through reviewed pull requests; same columns as "All Data")
         data/interim/district_points.csv  (district_id for each province + district)
Output:  data/raw/yield_labels.csv
         columns: district_id, province, district, crop, year, area_ha, production_t, yield_t_ha

When the same district, crop and year is in more than one place, one row is kept:
  1. the workbook (M1, checked by hand) always wins;
  2. then the source M1 used for that province: Punjab CRS for Punjab, MNFSR elsewhere;
  3. then the newer book (the one whose newest year is later).

The combined table is validated (src/validate_yields.py): any failed check stops this
step with a report, before the data is used.

Districts are matched on (province, district). The district names in
district_points.csv were made from the yield workbook, so every row must match;
the script stops if any does not.

Usage:  python -m src.build_yield_labels
"""

import glob
import os

import pandas as pd

from src.validate_yields import validate_or_stop

YIELD_XLSX = "data/raw/Pakistan_Master_Yield_1981-2025.xlsx"
YIELD_SHEET = "All Data"
POINTS_PATH = "data/interim/district_points.csv"
NEW_DIR = "data/sources/new_yields"
OUT_PATH = "data/raw/yield_labels.csv"
KEY = ["province", "district", "crop", "year"]
CROPS = {"wheat", "rice", "cotton", "sugarcane"}


def load_yields(path=YIELD_XLSX, sheet=YIELD_SHEET, df=None):
    if df is None:
        df = pd.read_excel(path, sheet_name=sheet)
    df = df[
        [
            "province",
            "district",
            "crop",
            "year",
            "area_ha",
            "production_t",
            "yield_t_ha",
        ]
    ].copy()
    df["year"] = df["year"].astype(int)
    df = df.dropna(subset=["yield_t_ha"])
    return df


def source_rank(source_id, province):
    """Lower wins. 0 is the workbook; then the source M1 used for that province."""
    preferred = "punjab_crs" if province == "Punjab" else "mnfsr"
    return 1 if source_id == preferred else 2


def combine_with_new(workbook, new_dir=NEW_DIR):
    """Workbook rows plus the rows of new books, one row per district x crop x year (see above)."""
    files = sorted(glob.glob(os.path.join(new_dir, "*.csv")))
    frames = [workbook.assign(_rank=0, _recency=0)]
    for f in files:
        source_id = os.path.basename(f).split("__")[0]
        df = pd.read_csv(f)
        df["_rank"] = [source_rank(source_id, p) for p in df["province"]]
        df["_recency"] = -int(df["year"].max())  # newer book first
        frames.append(df)
    allrows = pd.concat(frames, ignore_index=True)
    allrows["year"] = allrows["year"].astype(int)
    kept = (allrows.sort_values(["_rank", "_recency"], kind="stable")
            .drop_duplicates(KEY, keep="first")
            .drop(columns=["_rank", "_recency"]))
    added = len(kept) - len(workbook.drop_duplicates(KEY))
    print(f"New book files: {len(files)}; rows added to the workbook's: {added}")
    return kept.reset_index(drop=True)


def attach_ids(yields, points):
    out = yields.merge(
        points[["district_id", "province", "district"]],
        on=["province", "district"],
        how="left",
    )
    missing = out[out["district_id"].isna()][["province", "district"]].drop_duplicates()
    if not missing.empty:
        raise ValueError(
            f"Yield districts with no district_id in district_points.csv:\n{missing.to_string(index=False)}"
        )
    dup = out.duplicated(["district_id", "crop", "year"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate district x crop x year rows")
    unknown = set(out["crop"]) - CROPS
    if unknown:
        raise ValueError(f"Unknown crops: {unknown}")
    cols = [
        "district_id",
        "province",
        "district",
        "crop",
        "year",
        "area_ha",
        "production_t",
        "yield_t_ha",
    ]
    return out[cols].sort_values(["district_id", "crop", "year"]).reset_index(drop=True)


def main():
    raw = combine_with_new(pd.read_excel(YIELD_XLSX, sheet_name=YIELD_SHEET))
    points = pd.read_csv(POINTS_PATH)
    validate_or_stop(raw, points)
    labels = attach_ids(load_yields(df=raw), points)
    labels.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(labels)} rows -> {OUT_PATH}")
    print(
        labels.groupby("crop")
        .agg(
            rows=("year", "size"),
            districts=("district_id", "nunique"),
            first=("year", "min"),
            last=("year", "max"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
