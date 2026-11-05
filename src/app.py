"""FastAPI prediction service: /health, /predict (+ auto /docs)."""

import os
import pickle
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prepare_data import FEATURES, add_derived

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

app = FastAPI(
    title="Pakistan District Crop Yield API",
    description="Predicts district-level crop yield (t/ha).",
    version="0.1.0",
)

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=YieldOutput)
def predict(data: YieldInput):
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    row = add_derived(pd.DataFrame([data.model_dump()]))[FEATURES]
    pred = float(model.predict(row)[0])
    return YieldOutput(
        district=data.district,
        crop=data.crop,
        year=data.year,
        predicted_yield_t_ha=round(pred, 3),
    )
