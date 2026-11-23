"""Tests for Phase 6D (data drift + Discord alert) and Phase 6B (retraining gate)."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.drift import detect_feature_drift, run_drift_check, send_discord_alert
from src.retrain import compute_expanding_splits, retrain, should_promote


def _make_sample_df(n_per_year=40):
    rng = np.random.default_rng(42)
    rows = []
    for year in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
        for i in range(n_per_year):
            crop = ["wheat", "cotton", "rice", "sugarcane"][i % 4]
            province = ["Punjab", "Sindh", "KP", "Balochistan"][i % 4]
            # Simulate severe 2022 flood rainfall anomaly shift
            rain = float(rng.normal(620, 60)) if year == 2022 else float(rng.normal(210, 40))
            anomaly = float(rng.normal(380, 50)) if year == 2022 else float(rng.normal(0, 35))
            prev1 = 3.2 + (i % 5) * 0.1
            prev2 = 3.1 + (i % 5) * 0.1
            prev3 = 3.0 + (i % 5) * 0.1
            rows.append(
                {
                    "district_id": f"D_{i}",
                    "district": f"Dist_{i}",
                    "province": province,
                    "crop": crop,
                    "year": year,
                    "season_tmax_mean": float(rng.normal(31.0, 2.0)),
                    "heat_days_40c": int(rng.integers(5, 25)),
                    "season_rain_mm": rain,
                    "rain_anomaly": anomaly,
                    "soil_ph": 7.8,
                    "soil_oc": 6.5,
                    "soil_clay": 22.0,
                    "soil_sand": 35.0,
                    "soil_nitrogen": 0.8,
                    "prev_yield": prev1,
                    "prev2_yield": prev2,
                    "prev3_yield": prev3,
                    "trend_ratio": prev1 / prev2,
                    "yield_t_ha": prev1 * 1.02,
                }
            )
    return pd.DataFrame(rows)


def test_detect_drift_flags_2022_floods():
    df = _make_sample_df()
    ref = df[df["year"] <= 2017]
    flood_2022 = df[df["year"] == 2022]

    report = detect_feature_drift(ref, flood_2022)
    assert report["drift_detected"] is True
    assert "season_rain_mm" in report["drifted_features"]
    assert "rain_anomaly" in report["drifted_features"]


def test_detect_drift_no_drift_on_normal_year():
    df = _make_sample_df()
    ref = df[df["year"] <= 2017]
    normal_2019 = df[df["year"] == 2019]

    report = detect_feature_drift(ref, normal_2019)
    assert report["drift_detected"] is False


def test_discord_alert_posts_when_webhook_configured():
    report = {
        "drift_detected": True,
        "drifted_feature_count": 2,
        "drifted_features": ["season_rain_mm", "rain_anomaly"],
        "current_rows": 100,
        "reference_rows": 500,
        "features": {
            "season_rain_mm": {"reference_mean": 200.0, "current_mean": 550.0, "ks_statistic": 0.62, "p_value": 0.0},
            "rain_anomaly": {"reference_mean": 0.0, "current_mean": 310.0, "ks_statistic": 0.65, "p_value": 0.0},
        },
    }
    mock_resp = MagicMock(status_code=204)
    with patch("src.drift.requests.post", return_value=mock_resp) as mock_post:
        sent = send_discord_alert(report, label="year=2022", webhook_url="https://discord.com/api/webhooks/test/token")
        assert sent is True
        mock_post.assert_called_once()


def test_run_drift_check_writes_report(tmp_path):
    df = _make_sample_df()
    csv_path = tmp_path / "processed.csv"
    report_path = tmp_path / "drift_report.json"
    df.to_csv(csv_path, index=False)

    report = run_drift_check(data_path=str(csv_path), target_year=2022, report_path=str(report_path))
    assert report_path.exists()
    assert report["drift_detected"] is True


def test_should_promote_gate():
    assert should_promote(0.145, 0.150) is True
    assert should_promote(0.155, 0.150) is False
    assert should_promote(0.150, 0.150) is False
    assert should_promote(0.160, None) is True


def test_expanding_splits_preserve_baseline_and_advance():
    df_2023 = _make_sample_df(n_per_year=8)
    _, _, _, bounds_2023 = compute_expanding_splits(df_2023)
    assert bounds_2023 == (2017, 2018, 2020, 2021, 2023)

    extra = df_2023[df_2023["year"] == 2023].copy()
    extra["year"] = 2024
    df_2024 = pd.concat([df_2023, extra], ignore_index=True)
    _, _, _, bounds_2024 = compute_expanding_splits(df_2024)
    assert bounds_2024 == (2018, 2019, 2021, 2022, 2024)


def test_retrain_rejects_when_not_better(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _make_sample_df(n_per_year=16)
    csv_path = tmp_path / "processed.csv"
    df.to_csv(csv_path, index=False)

    best_run_path = tmp_path / "models" / "best_run.json"
    best_run_path.parent.mkdir(parents=True, exist_ok=True)
    best_run_path.write_text(json.dumps({"val_score": 0.0}))

    tiny_candidate = [("random_forest", RandomForestRegressor, {"n_estimators": 10, "max_depth": 3})]
    res = retrain(
        data_path=str(csv_path),
        best_run_path=str(best_run_path),
        model_path=str(tmp_path / "models" / "model.pkl"),
        force=True,
        candidates=tiny_candidate,
    )
    assert res["promoted"] is False
    assert res["status"] == "rejected_not_better"