"""DataOps - validation of the incoming yield data (data/raw/Pakistan_Master_Yield_1981-2025.xlsx).

Runs before the yield workbook is used (build_yield_labels calls it), so a bad
new year of data - a typo in a book, a missing column, a district spelled a new
way - stops the pipeline with a clear report instead of reaching the model.

ERRORS stop the pipeline:
  E1  required columns are present
  E2  no missing values in the key columns
  E3  year is a whole number between FIRST_YEAR and next year
  E4  crop and province are known values
  E5  area > 0 and production >= 0
  E6  yield agrees with production / area (within YIELD_TOLERANCE)
  E7  one row per province x district x crop x year
  E8  every district matches data/interim/district_points.csv (so it gets weather and soil)
  E9  every row names its source book (traceability)

WARNINGS are reported but do not stop it (they can be real, e.g. a flood year):
  W1  yield outside the realistic range for the crop (prepare_data drops these rows)
  W2  yield more than JUMP_FACTOR x higher or lower than the same district's previous year

Usage:  python -m src.validate_yields     (exit code 1 if there are errors)
"""

import sys
from datetime import date

import pandas as pd

from src.prepare_data import CROPS, PROVINCES, YIELD_BOUNDS

YIELD_XLSX = "data/raw/Pakistan_Master_Yield_1981-2025.xlsx"
YIELD_SHEET = "All Data"
POINTS_PATH = "data/interim/district_points.csv"

REQUIRED = ["province", "district", "crop", "year", "area_ha", "production_t", "yield_t_ha", "source"]
KEY = ["province", "district", "crop", "year"]
FIRST_YEAR = 1981
YIELD_TOLERANCE = 0.01  # 1%: the books round production and area
JUMP_FACTOR = 3
SHOW = 10  # example rows printed per problem


def _examples(rows, cols):
    return rows[cols].head(SHOW).to_dict("records")


def validate_yield_table(df, points):
    """Check the yield table. Returns {"errors": [...], "warnings": [...], "rows": n}.

    Each problem is {"check", "message", "count", "examples"}.
    """
    errors, warnings = [], []

    def add(target, check, message, rows, cols):
        if len(rows):
            target.append({"check": check, "message": message, "count": int(len(rows)), "examples": _examples(rows, cols)})

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        errors.append({"check": "E1", "message": f"missing columns: {missing_cols}", "count": len(missing_cols), "examples": []})
        return {"errors": errors, "warnings": warnings, "rows": len(df)}

    add(errors, "E2", "missing values in required columns", df[df[REQUIRED].isna().any(axis=1)], KEY)
    df = df.dropna(subset=REQUIRED)

    year = pd.to_numeric(df["year"], errors="coerce")
    bad_year = year.isna() | (year % 1 != 0) | (year < FIRST_YEAR) | (year > date.today().year + 1)
    add(errors, "E3", f"year must be a whole number from {FIRST_YEAR} to next year", df[bad_year], KEY)

    add(errors, "E4", f"crop must be one of {CROPS}", df[~df["crop"].isin(CROPS)], KEY)
    add(errors, "E4", f"province must be one of {PROVINCES}", df[~df["province"].isin(PROVINCES)], KEY)

    add(errors, "E5", "area must be > 0 and production >= 0",
        df[(df["area_ha"] <= 0) | (df["production_t"] < 0)], KEY + ["area_ha", "production_t"])

    ok = df[df["area_ha"] > 0]
    gap = (ok["production_t"] / ok["area_ha"] - ok["yield_t_ha"]).abs() / ok["yield_t_ha"].abs().clip(lower=1e-9)
    add(errors, "E6", f"yield differs from production / area by more than {YIELD_TOLERANCE:.0%}",
        ok[gap > YIELD_TOLERANCE], KEY + ["area_ha", "production_t", "yield_t_ha"])

    add(errors, "E7", "duplicate province x district x crop x year rows", df[df.duplicated(KEY, keep=False)], KEY)

    known = points[["province", "district"]].drop_duplicates()
    matched = df.merge(known, on=["province", "district"], how="left", indicator=True)
    unmatched = matched[matched["_merge"] == "left_only"].drop_duplicates(["province", "district"])
    add(errors, "E8", f"district not in {POINTS_PATH} (no weather/soil for it)", unmatched, ["province", "district"])

    add(errors, "E9", "no source book given", df[df["source"].astype(str).str.strip() == ""], KEY)

    for crop, (lo, hi) in YIELD_BOUNDS.items():
        rows = df[(df["crop"] == crop) & ((df["yield_t_ha"] < lo) | (df["yield_t_ha"] > hi))]
        add(warnings, "W1", f"{crop} yield outside [{lo}, {hi}] t/ha (prepare_data drops these)", rows, KEY + ["yield_t_ha"])

    prev = df[KEY + ["yield_t_ha"]].copy()
    prev["year"] = prev["year"] + 1
    joined = df.merge(prev, on=KEY, suffixes=("", "_prev_year"))
    ratio = joined["yield_t_ha"] / joined["yield_t_ha_prev_year"]
    jumps = joined[(ratio > JUMP_FACTOR) | (ratio < 1 / JUMP_FACTOR)]
    add(warnings, "W2", f"yield changed more than {JUMP_FACTOR}x from the previous year (check the book)",
        jumps, KEY + ["yield_t_ha_prev_year", "yield_t_ha"])

    return {"errors": errors, "warnings": warnings, "rows": len(df)}


def print_report(report):
    print(f"Yield data validation: {report['rows']} rows, "
          f"{len(report['errors'])} error checks failed, {len(report['warnings'])} warning checks raised")
    for level in ("errors", "warnings"):
        for p in report[level]:
            print(f"  {level[:-1].upper()} {p['check']}: {p['message']} - {p['count']} rows")
            for ex in p["examples"][:3]:
                print(f"      e.g. {ex}")


def validate_or_stop(df, points):
    """Validate and print the report; raise ValueError if any error check failed."""
    report = validate_yield_table(df, points)
    print_report(report)
    if report["errors"]:
        raise ValueError(f"Yield data failed validation: {[p['check'] for p in report['errors']]} (see report above)")
    return report


def main():
    df = pd.read_excel(YIELD_XLSX, sheet_name=YIELD_SHEET)
    report = validate_yield_table(df, pd.read_csv(POINTS_PATH))
    print_report(report)
    sys.exit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
