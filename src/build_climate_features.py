"""Step 5 - Turn daily weather + monthly rainfall into season features.

One row per district x crop x year (year = harvest-year label):
  wheat (Rabi):                      Nov of year-1 .. Apr of year
  cotton, rice, sugarcane (Kharif):  May .. Oct of year

Inputs:  data/interim/power_daily.csv, data/interim/chirps_monthly.csv
Output:  data/interim/climate_features.csv
  district_id, crop, year, season_tmax_mean, heat_days_40c, season_rain_mm,
  rain_anomaly, season_rh_mean, season_solar_mean

Usage:  python -m src.build_climate_features
"""

import pandas as pd

POWER_PATH = "data/interim/power_daily.csv"
CHIRPS_PATH = "data/interim/chirps_monthly.csv"
OUT_PATH = "data/interim/climate_features.csv"

# season name -> months, in order; months before the harvest year are listed first
SEASON_MONTHS = {
    "rabi": [11, 12, 1, 2, 3, 4],
    "kharif": [5, 6, 7, 8, 9, 10],
}
CROP_SEASON = {
    "wheat": "rabi",
    "cotton": "kharif",
    "rice": "kharif",
    "sugarcane": "kharif",
}
HEAT_THRESHOLD_C = 40
BASELINE_LAST_YEAR = 2017  # anomaly baseline uses training years only (no leakage)


def season_year(month, year, season):
    """Harvest-year label for a calendar month, or None if outside the season.

    Rabi: Nov-Dec belong to the NEXT year's harvest; Jan-Apr to this year's.
    """
    if month not in SEASON_MONTHS[season]:
        return None
    if season == "rabi" and month >= 11:
        return year + 1
    return year


def label_season(df, season, month_col="month", year_col="year"):
    """Add a 'season_year' column and drop rows outside the season."""
    out = df.copy()
    out["season_year"] = [
        season_year(m, y, season) for m, y in zip(out[month_col], out[year_col])
    ]
    return out.dropna(subset=["season_year"]).astype({"season_year": int})


def power_season_features(power, season):
    p = power.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["year"], p["month"] = p["date"].dt.year, p["date"].dt.month
    p = label_season(p, season)
    g = p.groupby(["district_id", "season_year"])
    feats = pd.DataFrame(
        {
            "season_tmax_mean": g["T2M_MAX"].mean(),
            "heat_days_40c": g["T2M_MAX"].apply(
                lambda s: int((s > HEAT_THRESHOLD_C).sum())
            ),
            "power_rain_mm": g["PRECTOTCORR"].sum(min_count=1),
            "season_rh_mean": g["RH2M"].mean(),
            "season_solar_mean": g["ALLSKY_SFC_SW_DWN"].mean(),
            "n_months": g["month"].nunique(),
        }
    ).reset_index()
    # keep only complete seasons (e.g. drops a half-observed season at the end)
    return feats[feats["n_months"] == len(SEASON_MONTHS[season])].drop(
        columns="n_months"
    )


def chirps_season_rain(chirps, season):
    c = label_season(chirps, season)
    g = c.groupby(["district_id", "season_year"])
    out = pd.DataFrame(
        {
            "season_rain_mm": g["rain_mm"].sum(min_count=1),
            "n_months": g["month"].nunique(),
        }
    ).reset_index()
    out = out[out["n_months"] == len(SEASON_MONTHS[season])]
    return out.drop(columns="n_months")


def add_rain_anomaly(df, baseline_last_year=BASELINE_LAST_YEAR):
    base = (
        df[df["year"] <= baseline_last_year]
        .groupby(["district_id", "crop"])["season_rain_mm"]
        .mean()
        .rename("rain_baseline")
    )
    df = df.join(base, on=["district_id", "crop"])
    df["rain_anomaly"] = df["season_rain_mm"] - df["rain_baseline"]
    return df.drop(columns="rain_baseline")


def build(power, chirps):
    frames = []
    for season in SEASON_MONTHS:
        f = power_season_features(power, season).merge(
            chirps_season_rain(chirps, season),
            on=["district_id", "season_year"],
            how="left",
        )
        # Rain comes from CHIRPS only. NASA POWER's PRECTOTCORR is NOT used to fill gaps:
        # its values jump from 2020 (all-district mean ~1,250 mm vs ~400 mm before),
        # which would look like climate drift exactly in the test years.
        f = f.drop(columns="power_rain_mm").rename(columns={"season_year": "year"})
        for crop, crop_season in CROP_SEASON.items():
            if crop_season == season:
                frames.append(f.assign(crop=crop))
    df = add_rain_anomaly(pd.concat(frames, ignore_index=True))
    cols = [
        "district_id",
        "crop",
        "year",
        "season_tmax_mean",
        "heat_days_40c",
        "season_rain_mm",
        "rain_anomaly",
        "season_rh_mean",
        "season_solar_mean",
    ]
    return df[cols].sort_values(["district_id", "crop", "year"]).round(3)


def main():
    power = pd.read_csv(POWER_PATH)
    chirps = pd.read_csv(CHIRPS_PATH)
    df = build(power, chirps)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_PATH}; years {df.year.min()}-{df.year.max()}")
    print(df.isna().sum()[lambda s: s > 0].to_string() or "No missing values")
    print(
        "(season_solar_mean is empty before 1984: NASA POWER has no sunlight data for 1981-1983)"
    )


if __name__ == "__main__":
    main()
