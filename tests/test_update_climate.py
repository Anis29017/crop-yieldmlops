"""Offline tests for adding new seasons to the climate table (src/update_climate.py).

The existing table is the real data/interim/climate_features.csv (two districts of it);
the downloaded weather is replaced by small daily / monthly tables.
"""

from datetime import date

import pandas as pd

from src import update_climate as uc

CLIMATE = pd.read_csv(uc.CLIMATE_PATH)
TWO = CLIMATE[CLIMATE["district_id"].isin(["PB_Faisalabad", "SD_Badin"])].reset_index(drop=True)


def daily(district_ids, start, end, tmax=30.0):
    days = pd.date_range(start, end, freq="D")
    return pd.concat([pd.DataFrame({"district_id": d, "date": days, "T2M_MAX": tmax, "T2M_MIN": 15.0,
                                    "PRECTOTCORR": 0.0, "RH2M": 40.0, "ALLSKY_SFC_SW_DWN": 18.0})
                      for d in district_ids], ignore_index=True)


def monthly(district_ids, months, rain=5.0):
    return pd.DataFrame([{"district_id": d, "year": y, "month": m, "rain_mm": rain}
                         for d in district_ids for y, m in months])


def test_next_seasons_follow_the_table():
    assert uc.next_seasons(CLIMATE) == {"wheat": 2026, "cotton": 2026, "rice": 2026, "sugarcane": 2026}
    assert uc.season_months("wheat", 2026)[0] == (2025, 11)
    assert uc.season_months("cotton", 2026) == [(2026, m) for m in range(5, 11)]


def test_window_runs_from_the_first_missing_month_to_the_end_of_last_month():
    assert uc.window(CLIMATE, date(2026, 11, 24)) == (date(2025, 11, 1), date(2026, 10, 31))
    assert uc.window(CLIMATE, date(2025, 11, 15)) is None  # wheat 2026 has not started a full month yet


def test_finished_wheat_season_is_added_kharif_waits_for_rain():
    ids = ["PB_Faisalabad", "SD_Badin"]
    power = daily(ids, "2025-11-01", "2026-10-31")
    rain = monthly(ids, [(2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 9)])  # Sep, Oct not published
    new, notes = uc.new_rows(power, rain, TWO)
    assert set(zip(new["crop"], new["year"])) == {("wheat", 2026)}
    assert len(new) == 2 and new["season_rain_mm"].eq(30.0).all()
    assert any("cotton 2026: not added (CHIRPS" in n for n in notes)


def test_rain_anomaly_uses_the_same_baseline_as_the_table():
    ids = ["PB_Faisalabad"]
    one = TWO[TWO["district_id"] == "PB_Faisalabad"]
    power = daily(ids, "2025-11-01", "2026-04-30")
    rain = monthly(ids, [(2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 5)])
    new, _ = uc.new_rows(power, rain, one)
    base = one[(one["crop"] == "wheat") & (one["year"] <= 2017)]["season_rain_mm"].mean()
    assert abs(new.iloc[0]["rain_anomaly"] - round(30.0 - base, 3)) < 1e-6
    # the table's own anomalies were made with the same baseline
    row = one[(one["crop"] == "wheat") & (one["year"] == 2025)].iloc[0]
    assert abs(row["rain_anomaly"] - (row["season_rain_mm"] - base)) < 0.01


def test_month_with_missing_days_is_not_used():
    power = daily(["PB_Faisalabad"], "2026-03-01", "2026-04-30")
    power.loc[power["date"] >= "2026-04-20", "T2M_MAX"] = float("nan")  # last 11 days of April missing
    kept = uc.complete_power_months(power)
    assert set(pd.to_datetime(kept["date"]).dt.month) == {3}


def test_a_crop_year_needs_every_district():
    power = daily(["PB_Faisalabad"], "2025-11-01", "2026-04-30")  # Badin missing
    rain = monthly(["PB_Faisalabad", "SD_Badin"], [(2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 5)])
    new, notes = uc.new_rows(power, rain, TWO)
    assert new.empty and any("wheat 2026: not added" in n for n in notes)


def test_existing_rows_are_never_changed():
    new = TWO[TWO["year"] == 2025].assign(season_tmax_mean=99.0)  # same crop-years as the table
    out = uc.append(TWO, new)
    assert len(out) == len(TWO) and not out["season_tmax_mean"].eq(99.0).any()
