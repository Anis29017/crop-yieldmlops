"""FastAPI prediction service: /health, /predict, /stats (+ auto /docs).

Monitoring: a middleware times every request. /stats shows the request count,
average latency overall and per endpoint (with p95 and max), and how long the
model itself takes per prediction. Every request is also logged with its
latency (visible in `docker logs`). Counters live in memory and reset when the
app restarts.
"""

import logging
import os
import pickle
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.prepare_data import FEATURES, add_derived

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

app = FastAPI(
    title="Pakistan District Crop Yield API",
    description="Predicts district-level crop yield (t/ha).",
    version="0.2.0",
)

logger = logging.getLogger("crop_yield_api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# ---------- monitoring ----------

RECENT = 1000  # latencies kept per endpoint for the p95 figure
_lock = threading.Lock()
_started = time.time()
_stats = {"count": 0, "total_ms": 0.0, "status": {}, "paths": {}, "model_count": 0, "model_ms": 0.0}


def _path_label(path):
    """Known endpoints by name; anything else (typos, scanners) is grouped as 'other'."""
    known = {route.path for route in app.routes}
    return path if path in known else "other"


def _record(path, status, ms):
    label = _path_label(path)
    with _lock:
        _stats["count"] += 1
        _stats["total_ms"] += ms
        group = f"{status // 100}xx"
        _stats["status"][group] = _stats["status"].get(group, 0) + 1
        p = _stats["paths"].setdefault(label, {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "recent": deque(maxlen=RECENT)})
        p["count"] += 1
        p["total_ms"] += ms
        p["max_ms"] = max(p["max_ms"], ms)
        p["recent"].append(ms)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Count and time every request; log one line per request."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        ms = (time.perf_counter() - start) * 1000
        _record(request.url.path, 500, ms)
        logger.exception("%s %s 500 %.1fms", request.method, request.url.path, ms)
        raise
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Latency-ms"] = f"{ms:.1f}"
    _record(request.url.path, response.status_code, ms)
    logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, ms)
    return response

_model = None


def load_model(path=MODEL_PATH):
    """Load the trained pipeline once and cache it."""
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run python -m src.train first.")
        with open(path, "rb") as f:
            _model = pickle.load(f)
    return _model


class YieldInput(BaseModel):
    district: str = Field(..., min_length=1, examples=["Faisalabad"])
    province: Literal[
        "Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad"
    ]
    crop: Literal["wheat", "cotton", "rice", "sugarcane"]
    year: int = Field(..., ge=1980, le=2100)
    season_tmax_mean: float = Field(..., ge=-10, le=60)
    season_rain_mm: float = Field(..., ge=0, le=5000)
    heat_days_40c: int = Field(..., ge=0, le=200)
    rain_anomaly: float
    soil_ph: float = Field(..., ge=3, le=11)
    soil_oc: float = Field(..., ge=0)
    soil_clay: float = Field(..., ge=0, le=100)
    soil_sand: float = Field(..., ge=0, le=100)
    soil_nitrogen: float = Field(..., ge=0)
    prev_yield: float = Field(..., gt=0, description="Yield 1 year before (t/ha)")
    prev2_yield: float = Field(..., gt=0, description="Yield 2 years before (t/ha)")
    prev3_yield: float = Field(..., gt=0, description="Yield 3 years before (t/ha)")


class YieldOutput(BaseModel):
    district: str
    crop: str
    year: int
    predicted_yield_t_ha: float


@app.get("/", include_in_schema=False)
def root():
    """The bare address opens the interactive docs instead of 'Not Found'."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=YieldOutput)
def predict(data: YieldInput):
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    start = time.perf_counter()
    row = add_derived(pd.DataFrame([data.model_dump()]))[FEATURES]
    pred = float(model.predict(row)[0])
    model_ms = (time.perf_counter() - start) * 1000
    with _lock:
        _stats["model_count"] += 1
        _stats["model_ms"] += model_ms
    return YieldOutput(
        district=data.district,
        crop=data.crop,
        year=data.year,
        predicted_yield_t_ha=round(pred, 3),
    )


@app.get("/stats")
def stats():
    """Request count and latency since the app started (resets on restart)."""
    with _lock:
        count = _stats["count"]
        endpoints = {
            path: {
                "count": p["count"],
                "average_latency_ms": round(p["total_ms"] / p["count"], 2),
                "p95_latency_ms": round(float(np.percentile(list(p["recent"]), 95)), 2),
                "max_latency_ms": round(p["max_ms"], 2),
            }
            for path, p in sorted(_stats["paths"].items())
        }
        model_count = _stats["model_count"]
        return {
            "total_requests": count,
            "average_latency_ms": round(_stats["total_ms"] / count, 2) if count else 0.0,
            "status_codes": dict(sorted(_stats["status"].items())),
            "endpoints": endpoints,
            "model": {
                "predictions": model_count,
                "average_model_latency_ms": round(_stats["model_ms"] / model_count, 2) if model_count else 0.0,
            },
            "uptime_seconds": round(time.time() - _started, 1),
            "started_at": datetime.fromtimestamp(_started, tz=timezone.utc).isoformat(timespec="seconds"),
        }
