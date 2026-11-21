"""Unit tests: data prep, model loading/training, and API endpoints."""

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app import app, load_model
from src.model import RecentYieldRatioRegressor
from src.prepare_data import (
    FEATURES,
    REQUIRED_COLS,
    TARGET,
    add_derived,
    drop_missing_prev_yield,
    prepare_data,
    validate,
)
from src.train import metrics, split

VALID_INPUT = {
    "district": "Faisalabad",
    "province": "Punjab",
    "crop": "wheat",
    "year": 2019,
    "season_tmax_mean": 33.5,
    "season_rain_mm": 240.0,
    "heat_days_40c": 14,
    "rain_anomaly": -10.0,
    "soil_ph": 8.1,
    "soil_oc": 5.2,
    "soil_clay": 22.0,
    "soil_sand": 41.0,
    "soil_nitrogen": 0.7,
    "prev_yield": 3.1,
    "prev2_yield": 3.0,
    "prev3_yield": 2.9,
}

client = TestClient(app)

REAL_DATA = "data/processed.csv"  # real prepared rows, committed to the repo


def _real_rows():
    """Real rows for two districts (all 4 crops, 1984-2025). Tests never use made-up yields."""
    df = pd.read_csv(REAL_DATA)
    return df[df["district_id"].isin(["PB_Faisalabad", "SD_Hyderabad"])].reset_index(drop=True)


# ---------- data preparation ----------


def test_prepare_data_stops_when_master_is_missing(tmp_path):
    """No real input -> an error, never a made-up dataset."""
    with pytest.raises(FileNotFoundError):
        prepare_data(raw_path=str(tmp_path / "missing.csv"), out_path=str(tmp_path / "out.csv"))
    assert not (tmp_path / "out.csv").exists()


def test_prepare_data_on_real_rows(tmp_path):
    raw = tmp_path / "master.csv"
    _real_rows().drop(columns="trend_ratio").to_csv(raw, index=False)
    out = tmp_path / "processed.csv"
    prepare_data(raw_path=str(raw), out_path=str(out))
    df = pd.read_csv(out)
    assert len(df) > 0
    assert df.isnull().sum().sum() == 0
    assert list(df.columns) == REQUIRED_COLS


def test_validate_rejects_missing_values():
    df = _real_rows()
    df.loc[0, "soil_ph"] = None
    with pytest.raises(AssertionError, match="Missing values"):
        validate(df)


def test_validate_rejects_duplicate_rows():
    df = _real_rows()
    df = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(AssertionError, match="Duplicate"):
        validate(df)


def test_validate_rejects_implausible_yield():
    df = _real_rows()
    df.loc[df["crop"] == "wheat", TARGET] = 50.0  # wheat cannot be 50 t/ha
    with pytest.raises(AssertionError, match="wheat"):
        validate(df)


def test_time_split_has_no_future_leakage():
    df = _real_rows()
    train, val = split(df)
    assert train["year"].max() <= 2017
    assert val["year"].min() >= 2018 and val["year"].max() <= 2020
    assert train["year"].max() < val["year"].min()


def test_first_year_rows_without_prev_yield_are_dropped():
    df = _real_rows()
    df.loc[[0, 1], "prev_yield"] = None
    out = drop_missing_prev_yield(df)
    assert len(out) == len(df) - 2
    assert out["prev_yield"].isnull().sum() == 0
    validate(out)


def test_wrapper_returns_recent_average_when_change_is_zero():
    """With a model that always predicts 'no change', the forecast is the 3-year average."""
    from sklearn.dummy import DummyRegressor

    df = _real_rows()
    m = RecentYieldRatioRegressor(
        DummyRegressor(strategy="constant", constant=0.0)
    ).fit(df[FEATURES], df[TARGET])
    expected = df[["prev_yield", "prev2_yield", "prev3_yield"]].mean(axis=1)
    assert m.predict(df[FEATURES]) == pytest.approx(expected.to_numpy())


def test_metrics_include_per_crop_rmse():
    df = _real_rows()
    m = metrics(df[TARGET], df["prev_yield"], df["crop"])
    assert {"val_rmse", "val_rmse_wheat", "val_rmse_sugarcane"} <= set(m)


# ---------- model ----------


@pytest.mark.skipif(not os.path.exists("models/model.pkl"), reason="run train.py first")
def test_model_loads_and_predicts_positive_yield():
    model = load_model()
    row = add_derived(pd.DataFrame([VALID_INPUT]))[FEATURES]
    pred = model.predict(row)
    assert len(pred) == 1
    assert pred[0] > 0


# ---------- API ----------


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.skipif(not os.path.exists("models/model.pkl"), reason="run train.py first")
def test_predict_valid_input():
    r = client.post("/predict", json=VALID_INPUT)
    assert r.status_code == 200
    body = r.json()
    assert body["crop"] == "wheat"
    assert body["predicted_yield_t_ha"] > 0


def test_predict_invalid_type():
    bad = {**VALID_INPUT, "season_rain_mm": "not_a_number"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_unknown_crop():
    bad = {**VALID_INPUT, "crop": "banana"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_short_province_name():
    bad = {**VALID_INPUT, "province": "KP"}  # the data uses "Khyber Pakhtunkhwa"
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_missing_field():
    bad = {k: v for k, v in VALID_INPUT.items() if k != "prev3_yield"}
    assert client.post("/predict", json=bad).status_code == 422


# ---------- monitoring ----------


def test_stats_counts_requests_and_reports_latency():
    before = client.get("/stats").json()
    client.get("/health")
    client.get("/health")
    client.post("/predict", json={**VALID_INPUT, "crop": "banana"})  # rejected (422) but still counted
    after = client.get("/stats").json()
    # the 3 calls above plus the first /stats call
    assert after["total_requests"] - before["total_requests"] == 4
    assert after["average_latency_ms"] > 0
    health = after["endpoints"]["/health"]
    assert health["count"] >= 2
    assert 0 < health["average_latency_ms"] <= health["max_latency_ms"]
    assert health["p95_latency_ms"] <= health["max_latency_ms"]
    assert after["status_codes"]["4xx"] >= 1
    assert after["uptime_seconds"] >= 0


def test_every_response_carries_its_latency():
    r = client.get("/health")
    assert float(r.headers["X-Latency-ms"]) > 0


def test_unknown_paths_are_grouped_as_other():
    client.get("/no-such-page")
    endpoints = client.get("/stats").json()["endpoints"]
    assert "other" in endpoints and "/no-such-page" not in endpoints


@pytest.mark.skipif(not os.path.exists("models/model.pkl"), reason="run train.py first")
def test_stats_tracks_model_latency():
    before = client.get("/stats").json()["model"]["predictions"]
    assert client.post("/predict", json=VALID_INPUT).status_code == 200
    model = client.get("/stats").json()["model"]
    assert model["predictions"] == before + 1
    assert model["average_model_latency_ms"] > 0


def test_predict_latency_reports_single_row_and_batch_speed():
    from src.train import predict_latency

    df = _real_rows()
    out = predict_latency(lambda X: X["prev_yield"].to_numpy(), df)
    assert out["latency_single_row_ms"] > 0
    assert out["latency_ms_per_1k_rows"] > 0
