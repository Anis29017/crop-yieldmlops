"""Phase 6D - Climate-driven data drift detection and Discord alerting.

Compares incoming data (e.g. a target year such as the 2022 floods, or a new CSV)
against the training baseline (year <= 2017) using two-sample Kolmogorov-Smirnov
(KS) tests on key climate and yield-lag features.

If drift is detected, writes a JSON summary report and sends an alert to a
Discord webhook when DISCORD_WEBHOOK_URL is set.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp

from src.prepare_data import FEATURES

DATA_PATH = "data/processed.csv"
REPORT_PATH = "models/drift_report.json"
TRAIN_END = 2017

DRIFT_FEATURES = [
    "season_tmax_mean",
    "heat_days_40c",
    "season_rain_mm",
    "rain_anomaly",
    "prev_yield",
]


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    p_threshold: float = 0.01,
    ks_threshold: float = 0.15,
    min_drifted_features: int = 2,
) -> dict:
    """Compare current_df against reference_df feature-by-feature using KS tests."""
    cols = [c for c in (features or DRIFT_FEATURES) if c in reference_df.columns and c in current_df.columns]
    if reference_df.empty or current_df.empty:
        raise ValueError("Reference and current DataFrames must both be non-empty.")

    feature_results = {}
    drifted_features = []

    for col in cols:
        ref_vals = reference_df[col].dropna().to_numpy(dtype=float)
        cur_vals = current_df[col].dropna().to_numpy(dtype=float)

        ks_stat, p_value = ks_2samp(ref_vals, cur_vals)
        ref_mean = float(np.mean(ref_vals))
        cur_mean = float(np.mean(cur_vals))
        ref_std = float(np.std(ref_vals))
        z_shift = float((cur_mean - ref_mean) / ref_std) if ref_std > 0 else 0.0

        drifted = bool((p_value < p_threshold) and (ks_stat >= ks_threshold))
        if drifted:
            drifted_features.append(col)

        feature_results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 6),
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "z_shift": round(z_shift, 4),
            "drifted": drifted,
        }

    drift_detected = len(drifted_features) >= min_drifted_features
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "p_threshold": p_threshold,
        "ks_threshold": ks_threshold,
        "drift_detected": drift_detected,
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features,
        "features": feature_results,
    }


def send_discord_alert(report: dict, label: str = "incoming batch", webhook_url: str | None = None) -> bool:
    """Post a drift alert to Discord if a webhook URL is configured."""
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    drifted = ", ".join(report["drifted_features"]) or "none"
    summary_lines = [
        f"**Crop Yield API — Data Drift Alert ({label})**",
        f"• Drift detected: `{report['drift_detected']}` ({report['drifted_feature_count']} features shifted)",
        f"• Drifted features: `{drifted}`",
        f"• Rows compared: `{report['current_rows']}` vs `{report['reference_rows']}` baseline",
    ]
    for feat in report["drifted_features"]:
        stats = report["features"][feat]
        summary_lines.append(
            f"  - `{feat}`: mean `{stats['reference_mean']}` → `{stats['current_mean']}` "
            f"(KS={stats['ks_statistic']}, p={stats['p_value']})"
        )
    message = "\n".join(summary_lines)

    if not url:
        print("[ALERT - DRY RUN (DISCORD_WEBHOOK_URL not set)]\n" + message)
        return False

    resp = requests.post(url, json={"content": message}, timeout=10)
    resp.raise_for_status()
    print(f"[ALERT SENT] Discord webhook notified (status {resp.status_code}).")
    return True


def run_drift_check(
    data_path: str = DATA_PATH,
    target_year: int = 2022,
    new_data_path: str | None = None,
    report_path: str = REPORT_PATH,
) -> dict:
    """Run drift detection comparing target_year (or new_data_path) to train <= 2017."""
    df = pd.read_csv(data_path)
    reference_df = df[df["year"] <= TRAIN_END]

    if new_data_path:
        current_df = pd.read_csv(new_data_path)
        label = os.path.basename(new_data_path)
    else:
        current_df = df[df["year"] == target_year]
        label = f"year={target_year}"

    available_features = [f for f in DRIFT_FEATURES if f in FEATURES]
    report = detect_feature_drift(reference_df, current_df, features=available_features)
    report["target_label"] = label

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Drift check ({label}): drift_detected={report['drift_detected']}, drifted={report['drifted_features']}")
    if report["drift_detected"]:
        send_discord_alert(report, label=label)

    return report


def main():
    parser = argparse.ArgumentParser(description="Check incoming crop-yield features for climate drift.")
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to master processed CSV")
    parser.add_argument("--year", type=int, default=2022, help="Year slice to check against training baseline")
    parser.add_argument("--new-data", default=None, help="Optional separate CSV of new incoming rows")
    parser.add_argument("--report-path", default=REPORT_PATH, help="Where to save JSON drift report")
    args = parser.parse_args()

    run_drift_check(
        data_path=args.data_path,
        target_year=args.year,
        new_data_path=args.new_data,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()