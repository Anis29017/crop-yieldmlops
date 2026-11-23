"""Phase 6B - Automated annual / drift-triggered retraining with promotion gate.

Workflow:
1. Optionally check drift first (when --check-drift is passed). If no drift is
   detected and --force is not set, skip retraining cleanly.
2. Train candidate models on the updated training window and evaluate on the
   validation window using the same `val_score` metric as src/train.py.
3. Compare the best candidate against the currently active model in
   models/best_run.json (and MLflow Model Registry).
4. Promote (register new MLflow version + update models/model.pkl and
   models/best_run.json) ONLY if `new_val_score < current_val_score`.
"""

import argparse
import json
import os
import pickle
import shutil
from datetime import datetime, timezone

import mlflow
import pandas as pd

from src.drift import run_drift_check
from src.prepare_data import FEATURES, TARGET
from src.train import (
    BEST_RUN_PATH,
    CANDIDATES,
    DATA_PATH,
    EXPERIMENT,
    MODEL_PATH,
    REGISTERED_NAME,
    metrics,
    train_one,
)


def compute_expanding_splits(df: pd.DataFrame, val_years: int = 3, test_years: int = 3):
    """Build train / val / test splits dynamically from the latest available year.

    Preserves the M1 baseline split (train <= 2017, val 2018-2020, test 2021-2023)
    when max_year <= 2023, and advances forward as new annual data is added.
    """
    max_year = int(df["year"].max())
    if max_year <= 2023:
        train_end, val_start, val_end = 2017, 2018, 2020
        test_start, test_end = 2021, 2023
    else:
        test_end = max_year
        test_start = max_year - test_years + 1
        val_end = test_start - 1
        val_start = val_end - val_years + 1
        train_end = val_start - 1

    train_df = df[df["year"] <= train_end]
    val_df = df[(df["year"] >= val_start) & (df["year"] <= val_end)]
    test_df = df[(df["year"] >= test_start) & (df["year"] <= test_end)]
    return train_df, val_df, test_df, (train_end, val_start, val_end, test_start, test_end)


def load_current_best_score(best_run_path: str = BEST_RUN_PATH) -> float | None:
    """Read current production model's validation score if available."""
    if not os.path.exists(best_run_path):
        return None
    with open(best_run_path) as f:
        data = json.load(f)
    score = data.get("val_score")
    return float(score) if score is not None else None


def should_promote(new_val_score: float, current_val_score: float | None, min_improvement: float = 0.0) -> bool:
    """Return True if there is no existing model or new_val_score beats current_val_score."""
    if current_val_score is None:
        return True
    return new_val_score < (current_val_score - min_improvement)


def retrain(
    data_path: str = DATA_PATH,
    best_run_path: str = BEST_RUN_PATH,
    model_path: str = MODEL_PATH,
    check_drift: bool = False,
    drift_year: int = 2022,
    force: bool = True,
    candidates=None,
) -> dict:
    """Run candidate retraining and promote only if validation score improves."""
    if check_drift and not force:
        drift_report = run_drift_check(data_path=data_path, target_year=drift_year)
        if not drift_report["drift_detected"]:
            result = {
                "status": "skipped_no_drift",
                "promoted": False,
                "drift_year": drift_year,
            }
            print("No drift detected and --force not set; skipping retraining.")
            return result

    df = pd.read_csv(data_path)
    train_df, val_df, test_df, bounds = compute_expanding_splits(df)
    train_end, val_start, val_end, test_start, test_end = bounds
    print(
        f"Retraining splits: train <= {train_end} ({len(train_df)} rows), "
        f"val {val_start}-{val_end} ({len(val_df)} rows), "
        f"test {test_start}-{test_end} ({len(test_df)} rows)"
    )

    mlflow.set_experiment(EXPERIMENT)
    model_candidates = candidates if candidates is not None else CANDIDATES

    best = None
    for name, cls, params in model_candidates:
        model, m, run_id, uri = train_one(name, cls, params, train_df, val_df)
        if best is None or m["val_score"] < best["metrics"]["val_score"]:
            best = {
                "model": model,
                "metrics": m,
                "run_id": run_id,
                "uri": uri,
                "model_type": name,
                "params": params,
            }

    current_score = load_current_best_score(best_run_path)
    new_score = best["metrics"]["val_score"]
    promoted = should_promote(new_score, current_score)

    test_m = metrics(
        test_df[TARGET],
        best["model"].predict(test_df[FEATURES]),
        test_df["crop"],
        prefix="test",
    )

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics(test_m)
        mlflow.set_tag("retrain_candidate", "true")
        mlflow.set_tag("promoted", str(promoted).lower())

    summary = {
        "status": "promoted" if promoted else "rejected_not_better",
        "promoted": promoted,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": best["run_id"],
        "model_type": best["model_type"],
        "params": best["params"],
        "previous_val_score": current_score,
        "new_val_score": new_score,
        **best["metrics"],
        **test_m,
    }

    if promoted:
        version = mlflow.register_model(best["uri"], REGISTERED_NAME)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        versioned = os.path.join(
            os.path.dirname(model_path),
            f"model_v{version.version}_{best['model_type']}.pkl",
        )
        with open(versioned, "wb") as f:
            pickle.dump(best["model"], f)
        shutil.copyfile(versioned, model_path)

        summary["registered_version"] = int(version.version)
        summary["model_file"] = versioned
        with open(best_run_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(
            f"PROMOTED '{REGISTERED_NAME}' v{version.version}: "
            f"val_score {new_score:.4f} (previous={current_score})"
        )
    else:
        print(
            f"NOT PROMOTED: candidate val_score {new_score:.4f} did not beat "
            f"current val_score {current_score:.4f}"
        )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Annual / drift-triggered model retraining pipeline.")
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to processed dataset CSV")
    parser.add_argument("--check-drift", action="store_true", help="Only retrain if drift check triggers")
    parser.add_argument("--drift-year", type=int, default=2022, help="Year slice to check when --check-drift is set")
    parser.add_argument("--force", action="store_true", help="Force retraining even if no drift is detected")
    args = parser.parse_args()

    force = args.force or not args.check_drift
    retrain(
        data_path=args.data_path,
        check_drift=args.check_drift,
        drift_year=args.drift_year,
        force=force,
    )


if __name__ == "__main__":
    main()