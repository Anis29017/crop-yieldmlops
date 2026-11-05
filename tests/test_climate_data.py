"""Offline tests for the weather/soil retrieval and feature code (no internet)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from src.build_climate_features import build, season_year
from src.fetch_chirps import NODATA, district_rows, grid_from_centres
from src.fetch_districts import query_points
from src.fetch_nasa_power import parse_power_json

# ---------- NASA POWER ----------


def test_parse_power_json_turns_fill_value_into_nan():
    payload = {
        "properties": {
            "parameter": {
                "T2M_MAX": {"20200101": 20.5, "20200102": -999},
                "PRECTOTCORR": {"20200101": 0.0, "20200102": 3.2},
            }
        }
    }
    df = parse_power_json(payload)
    assert list(df["date"].dt.day) == [1, 2]
    assert df.loc[1, "T2M_MAX"] != df.loc[1, "T2M_MAX"]  # NaN
    assert df.loc[1, "PRECTOTCORR"] == 3.2


# ---------- season logic ----------


@pytest.mark.parametrize(
    "month,year,season,expected",
    [
        (11, 2016, "rabi", 2017),  # Nov 2016 belongs to the 2017 wheat harvest
        (4, 2017, "rabi", 2017),
        (5, 2017, "rabi", None),
        (5, 2017, "kharif", 2017),
        (11, 2017, "kharif", None),
    ],
)
def test_season_year(month, year, season, expected):
    assert season_year(month, year, season) == expected


def _synthetic_inputs():
    dates = pd.date_range("2016-11-01", "2018-12-31", freq="D")
    power = pd.DataFrame(
        {
            "district_id": "D1",
            "date": dates,
            "T2M_MAX": np.where(dates.month == 6, 42.0, 30.0),  # June heatwave
            "T2M_MIN": 15.0,
            "PRECTOTCORR": 1.0,
            "RH2M": 50.0,
            "ALLSKY_SFC_SW_DWN": 20.0,
        }
    )
    chirps = pd.DataFrame(
        [
            {"district_id": "D1", "year": d.year, "month": d.month, "rain_mm": 10.0}
            for d in pd.date_range("2016-11-01", "2018-12-01", freq="MS")
        ]
    )
    return power, chirps


def test_build_season_features():
    power, chirps = _synthetic_inputs()
    df = build(power, chirps)
    wheat_2017 = df[(df.crop == "wheat") & (df.year == 2017)].iloc[0]
    cotton_2017 = df[(df.crop == "cotton") & (df.year == 2017)].iloc[0]
    assert wheat_2017.season_rain_mm == 60.0  # 6 months x 10 mm
    assert wheat_2017.heat_days_40c == 0
    assert cotton_2017.heat_days_40c == 30  # all of June above 40C
    # Rabi 2019 has only Nov-Dec 2018 in the data -> incomplete, must be dropped
    assert not ((df.crop == "wheat") & (df.year == 2019)).any()


def test_rain_anomaly_uses_training_years_only():
    power, chirps = _synthetic_inputs()
    chirps.loc[chirps.year == 2018, "rain_mm"] = 100.0  # wet 2018 after the baseline
    df = build(power, chirps)
    cotton = df[df.crop == "cotton"].set_index("year")
    assert cotton.loc[2017, "rain_anomaly"] == 0.0  # 2017 is the whole baseline
    assert cotton.loc[2018, "rain_anomaly"] == 540.0  # 600 - 60


def test_every_crop_present():
    power, chirps = _synthetic_inputs()
    assert set(build(power, chirps).crop) == {"wheat", "cotton", "rice", "sugarcane"}


# ---------- CHIRPS grid -> district means ----------


@pytest.fixture
def grid_and_districts():
    """10x10 grid over lon 60-70, lat 30-40 (1 degree cells); left half = 10, right half = 20."""
    arr = np.full((10, 10), 10.0)
    arr[:, 5:] = 20.0
    arr[0, 0] = NODATA  # one empty pixel
    grid = grid_from_centres(np.arange(60.5, 70, 1.0), np.arange(39.5, 30, -1.0))
    districts = gpd.GeoDataFrame(
        {"district_id": ["LEFT", "RIGHT", "TINY"]},
        geometry=[
            box(60, 30, 65, 40),
            box(65, 30, 70, 40),
            box(66.1, 31.1, 66.3, 31.3),
        ],
        crs="EPSG:4326",
    )
    return arr, grid, districts


def test_grid_from_centres():
    grid = grid_from_centres(np.array([60.5, 61.5]), np.array([39.5, 38.5]))
    assert grid == (60.0, 1.0, 40.0, 1.0)


def test_district_means_ignore_nodata(grid_and_districts):
    arr, grid, districts = grid_and_districts
    df = district_rows(iter([(2020, 7, arr, grid)]), districts).set_index("district_id")
    assert df.loc["LEFT", "rain_mm"] == pytest.approx(
        10.0
    )  # the empty pixel is ignored
    assert df.loc["RIGHT", "rain_mm"] == pytest.approx(20.0)
    assert set(df.year) == {2020} and set(df.month) == {7}


def test_district_smaller_than_a_cell_still_gets_a_value(grid_and_districts):
    arr, grid, districts = grid_and_districts
    df = district_rows(iter([(2020, 7, arr, grid)]), districts).set_index("district_id")
    assert df.loc["TINY", "rain_mm"] == pytest.approx(20.0)  # the cell it sits in


# ---------- districts + soil ----------


def test_query_points_lie_inside_polygons():
    # a U-shaped district whose centroid falls outside it
    u_shape = Polygon([(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)])
    gdf = gpd.GeoDataFrame(
        {"district_id": ["U"], "province": ["P"], "district": ["U"]},
        geometry=[u_shape],
        crs="EPSG:4326",
    )
    pts = query_points(gdf)
    point = gpd.points_from_xy(pts.lon, pts.lat)[0]
    assert u_shape.contains(point)


def test_depth_weighted_soil_average():
    pytest.importorskip(
        "rasterio"
    )  # soil step needs rasterio; skip where it is not installed
    from src.fetch_soilgrids import DEPTHS, depth_weighted

    vals = {
        "0-5cm": np.array([60.0]),
        "5-15cm": np.array([90.0]),
        "15-30cm": np.array([120.0]),
    }
    # (60*5 + 90*10 + 120*15) / 30 = 100
    assert depth_weighted(vals)[0] == pytest.approx(100.0)
    assert sum(DEPTHS.values()) == 30
