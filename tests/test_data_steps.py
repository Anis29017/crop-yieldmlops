"""Tests for the yield-label, merge and cleaning steps (offline, small made-up tables)."""

import pandas as pd
import pytest

from src.build_master import add_prev_yield, build
from src.build_yield_labels import attach_ids
from src.prepare_data import (
    PROVINCES,
    REQUIRED_COLS,
    drop_implausible,
    drop_incomplete,
    make_synthetic,
    validate,
)

POINTS = pd.DataFrame(
    {
        "district_id": ["PB_A", "SD_B"],
        "province": ["Punjab", "Sindh"],
        "district": ["A", "B"],
        "lat": [31, 25],
        "lon": [73, 68],
    }
)


def _labels():
    rows = []
    for d, p, n in (("PB_A", "Punjab", "A"), ("SD_B", "Sindh", "B")):
        for y in (2000, 2001, 2002, 2003, 2005):  # 2004 missing on purpose
            rows.append(
                {
                    "district_id": d,
                    "province": p,
                    "district": n,
                    "crop": "wheat",
                    "year": y,
                    "area_ha": 1000.0,
                    "production_t": 3000.0,
                    "yield_t_ha": 3.0 + (y - 2000) / 10,
                }
            )
    return pd.DataFrame(rows)


# ---------- yield labels ----------


def test_attach_ids_matches_on_province_and_district():
    y = _labels().drop(columns="district_id")
    out = attach_ids(y, POINTS)
    assert out.district_id.notna().all()
    assert set(out.district_id) == {"PB_A", "SD_B"}


def test_attach_ids_stops_on_unknown_district():
    y = _labels().drop(columns="district_id")
    y.loc[0, "district"] = "Nowhere"
    with pytest.raises(ValueError, match="Nowhere"):
        attach_ids(y, POINTS)


# ---------- merge ----------


def test_previous_yields_use_exact_years():
    out = add_prev_yield(_labels()).set_index(["district_id", "year"])
    assert pd.isna(
        out.loc[("PB_A", 2000), "prev_yield"]
    )  # first year: nothing before it
    assert out.loc[("PB_A", 2001), "prev_yield"] == pytest.approx(3.0)
    r = out.loc[("PB_A", 2003)]
    assert (r.prev_yield, r.prev2_yield, r.prev3_yield) == pytest.approx(
        (3.2, 3.1, 3.0)
    )
    assert pd.isna(
        out.loc[("PB_A", 2005), "prev_yield"]
    )  # 2004 missing: NOT filled from 2003
    assert out.loc[("PB_A", 2005), "prev2_yield"] == pytest.approx(3.3)


def test_build_master_joins_weather_and_soil():
    labels = _labels()
    climate = labels[["district_id", "crop", "year"]].assign(
        season_tmax_mean=30.0,
        heat_days_40c=5,
        season_rain_mm=200.0,
        rain_anomaly=0.0,
        season_rh_mean=40.0,
        season_solar_mean=20.0,
    )
    soil = pd.DataFrame(
        {
            "district_id": ["PB_A", "SD_B"],
            "soil_ph": [8.0, 8.1],
            "soil_oc": [6, 5],
            "soil_clay": [25, 24],
            "soil_sand": [35, 45],
            "soil_nitrogen": [0.8, 0.7],
        }
    )
    df, report = build(labels, climate, soil)
    assert report["no season weather"] == 0 and report["no soil"] == 0
    assert report["no previous-year yield"] == 4  # 2000 and 2005 for each district
    assert (
        report["missing any of the 3 previous years"] == 8
    )  # 2000, 2001, 2002, 2005 per district
    assert len(df) == len(labels)


# ---------- cleaning ----------


def test_drop_incomplete_and_implausible():
    df = make_synthetic(n_districts=2)
    df.loc[df.index[df.crop == "cotton"][0], "prev_yield"] = None  # 1 incomplete row
    df.loc[df.index[df.crop == "wheat"][:3], "yield_t_ha"] = (
        50.0  # 3 impossible wheat yields
    )
    out = drop_implausible(drop_incomplete(df))
    assert len(out) == len(df) - 4
    validate(out[REQUIRED_COLS])


def test_validate_rejects_unknown_province():
    df = make_synthetic(n_districts=2)
    df.loc[0, "province"] = "KP"  # must be the full name
    with pytest.raises(AssertionError, match="province"):
        validate(df)


def test_province_names_match_the_yield_data():
    assert "Khyber Pakhtunkhwa" in PROVINCES and "KP" not in PROVINCES
