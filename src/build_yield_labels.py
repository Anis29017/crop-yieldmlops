"""Step 6 - Yield labels: district crop yields with the district_id used by the weather/soil files.

Input:   data/raw/Pakistan_Master_Yield_1981-2025.xlsx  (sheet "All Data": every usable
         district x crop x year yield, compiled from PBS, provincial Crop Reporting /
         Development Statistics books and MNFSR; see that workbook's Notes sheet)
         data/interim/district_points.csv  (district_id for each province + district)
Output:  data/raw/yield_labels.csv
         columns: district_id, province, district, crop, year, area_ha, production_t, yield_t_ha

Districts are matched on (province, district). The district names in
district_points.csv were made from the yield workbook, so every row must match;
the script stops if any does not.

Usage:  python -m src.build_yield_labels
"""

import pandas as pd

YIELD_XLSX = "data/raw/Pakistan_Master_Yield_1981-2025.xlsx"
YIELD_SHEET = "All Data"
POINTS_PATH = "data/interim/district_points.csv"
OUT_PATH = "data/raw/yield_labels.csv"
CROPS = {"wheat", "rice", "cotton", "sugarcane"}


def load_yields(path=YIELD_XLSX, sheet=YIELD_SHEET):
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
    labels = attach_ids(load_yields(), pd.read_csv(POINTS_PATH))
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
