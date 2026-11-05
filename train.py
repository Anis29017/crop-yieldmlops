"""Step 9 - Training with MLflow tracking.

- Time-based split: train <= 2017, validate 2018-2020, test 2021-2023.
  Models are compared and the best one is CHOSEN on validation only; the chosen
  model is then scored ONCE on the held-out test years.
- Every model is wrapped in RecentYieldRatioRegressor (src/model.py): it learns
  the change from the average of the previous 3 years, which suits slowly
  rising yields and keeps sugarcane (~60 t/ha) from dominating.
- Logs two simple baselines (last year's yield; 3-year average) and 9 model
  runs (RandomForest / XGBoost / LightGBM, from cautious to flexible settings).
- Best model = lowest val_score = average over the 4 crops of
  (RMSE for that crop / mean yield of that crop).
- Saves the best model as models/model_v<N>_<type>.pkl (N = MLflow registry
  version) plus models/model.pkl (the copy the API loads), and registers it.
"""

import json
import os
import pickle
import shutil
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.model import RecentYieldRatioRegressor, recent_average
from src.prepare_data import CROPS, FEATURES, PROVINCES, TARGET

warnings.filterwarnings("ignore", message=".*pickle or cloudpickle format.*")

DATA_PATH = "data/processed.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
BEST_RUN_PATH = os.path.join(MODEL_DIR, "best_run.json")
EXPERIMENT = "crop-yield-m1"
REGISTERED_NAME = "crop-yield-model"

TRAIN_END = 2017
VAL_START, VAL_END = 2018, 2020
TEST_START, TEST_END = 2021, 2023

CATEGORICAL = ["crop", "province"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


def split(df):
    """Train / validation split (the test years are taken separately by test_split)."""
    train = df[df["year"] <= TRAIN_END]
    val = df[(df["year"] >= VAL_START) & (df["year"] <= VAL_END)]
    return train, val


def test_split(df):
    return df[(df["year"] >= TEST_START) & (df["year"] <= TEST_END)]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred, crops=None, prefix="val"):
    """Pooled metrics, plus per-crop RMSE and the selection score when `crops` is given.

    {prefix}_score = mean over crops of RMSE_crop / mean(yield_crop), i.e. the
    typical error as a fraction of that crop's normal yield (0.15 = 15%). Lower is better.
    """
    out = {
        f"{prefix}_rmse": rmse(y_true, y_pred),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }
    if crops is not None:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        crops = np.asarray(crops)
        rel = []
        for crop in CROPS:
            mask = crops == crop
            if mask.any():
                r = rmse(y_true[mask], y_pred[mask])
                out[f"{prefix}_rmse_{crop}"] = r
                rel.append(r / float(np.mean(y_true[mask])))
        out[f"{prefix}_score"] = float(np.mean(rel))
    return out


def build_pipeline(model):
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(categories=[CROPS, PROVINCES], handle_unknown="ignore"),
                CATEGORICAL,
            ),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return RecentYieldRatioRegressor(Pipeline([("pre", pre), ("model", model)]))


# Each entry = one MLflow run: 9 model runs (+ 2 baselines = 11 runs).
# Settings go from cautious (few, shallow trees; big leaves) to flexible.
CANDIDATES = [
    (
        "random_forest",
        RandomForestRegressor,
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 50},
    ),
    (
        "random_forest",
        RandomForestRegressor,
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20},
    ),
    (
        "random_forest",
        RandomForestRegressor,
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5},
    ),
    (
        "xgboost",
        XGBRegressor,
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 50,
            "reg_lambda": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ),
    (
        "xgboost",
        XGBRegressor,
        {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 20,
        },
    ),
    (
        "xgboost",
        XGBRegressor,
        {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 1,
        },
    ),
    (
        "lightgbm",
        LGBMRegressor,
        {
            "n_estimators": 300,
            "num_leaves": 7,
            "learning_rate": 0.03,
            "min_child_samples": 100,
            "reg_lambda": 5,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
        },
    ),
    (
        "lightgbm",
        LGBMRegressor,
        {
            "n_estimators": 300,
            "num_leaves": 15,
            "learning_rate": 0.03,
            "min_child_samples": 50,
        },
    ),
    (
        "lightgbm",
        LGBMRegressor,
        {
            "n_estimators": 600,
            "num_leaves": 63,
            "learning_rate": 0.03,
            "min_child_samples": 20,
        },
    ),
]

BASELINES = {
    "baseline_last_year": lambda d: d["prev_yield"].to_numpy(dtype=float),
    "baseline_3yr_average": recent_average,
}


def log_baselines(val, test):
    """Naive forecasts the model must beat. Returns {name: val_score}."""
    scores = {}
    for name, forecast in BASELINES.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            m = metrics(val[TARGET], forecast(val), val["crop"])
            m.update(metrics(test[TARGET], forecast(test), test["crop"], prefix="test"))
            mlflow.log_metrics(m)
            print(
                f"{name:22s} val_score={m['val_score']:.4f}  test_score={m['test_score']:.4f}"
            )
            scores[name] = {"val_score": m["val_score"], "test_score": m["test_score"]}
    return scores


def train_one(name, cls, params, train, val):
    extra = {"random_state": 42}
    if name == "lightgbm":
        extra["verbose"] = -1
    if name == "random_forest":
        extra["n_jobs"] = -1
    model = build_pipeline(cls(**params, **extra))
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_param("target", "log(yield / mean of previous 3 years)")
        mlflow.log_params(params)
        model.fit(train[FEATURES], train[TARGET])
        m = metrics(val[TARGET], model.predict(val[FEATURES]), val["crop"])
        mlflow.log_metrics(m)
        # cloudpickle: we trained this model ourselves, so it is a trusted source
        info = mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle"
        )
        print(f"{name:14s} {params}  val_score={m['val_score']:.4f}")
        return model, m, run.info.run_id, info.model_uri


def train(data_path=DATA_PATH, candidates=CANDIDATES):
    df = pd.read_csv(data_path)
    train_df, val_df = split(df)
    test_df = test_split(df)
    print(
        f"Rows: train {len(train_df)} (<= {TRAIN_END}), validation {len(val_df)} ({VAL_START}-{VAL_END}), "
        f"test {len(test_df)} ({TEST_START}-{TEST_END}, held out)"
    )
    mlflow.set_experiment(EXPERIMENT)

    baselines = log_baselines(val_df, test_df)
    best = None
    for name, cls, params in candidates:
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

    # Held-out test: scored once, only for the chosen model, logged on its run.
    test_m = metrics(
        test_df[TARGET],
        best["model"].predict(test_df[FEATURES]),
        test_df["crop"],
        prefix="test",
    )
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics(test_m)
        mlflow.set_tag("best_model", "true")

    version = mlflow.register_model(best["uri"], REGISTERED_NAME)
    print(f"Registered '{REGISTERED_NAME}' version {version.version}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    versioned = os.path.join(
        MODEL_DIR, f"model_v{version.version}_{best['model_type']}.pkl"
    )
    with open(versioned, "wb") as f:
        pickle.dump(best["model"], f)
    shutil.copyfile(versioned, MODEL_PATH)

    best_baseline = min(baselines, key=lambda k: baselines[k]["val_score"])
    summary = {
        "run_id": best["run_id"],
        "registered_version": int(version.version),
        "model_file": versioned,
        "model_type": best["model_type"],
        "params": best["params"],
        **best["metrics"],
        **test_m,
        "baselines": baselines,
        "best_baseline": best_baseline,
        "beats_baseline": best["metrics"]["val_score"]
        < baselines[best_baseline]["val_score"],
        "beats_baseline_on_test": test_m["test_score"]
        < baselines[best_baseline]["test_score"],
    }
    with open(BEST_RUN_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Best run:", json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    train()
