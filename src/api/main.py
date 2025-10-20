"""FastAPI application exposing forecasts, alerts, and allocation heuristics."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel

from src.config import PROCESSED_DATA_DIR
from src.models.predictor import ForecastPredictor
from src.services.allocation import compute_allocation
from src.services.alerts import compute_alerts

app = FastAPI(title="Flu Planning POC", version="0.1.0")


class ForecastItem(BaseModel):
    dep_code: str
    year: int
    week: int
    week_start: str
    prediction: float


class AlertItem(BaseModel):
    dep_code: str
    week_start: str
    signal: float
    level: str
    p80: float
    p90: float


class AllocationItem(BaseModel):
    dep_code: str
    week_start: str
    forecast_need: float
    coverage: float
    allocation: float
    equity_bonus: float


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_DIR / "training_dataset.csv", parse_dates=["week_start"])


@lru_cache(maxsize=1)
def _load_predictor() -> ForecastPredictor:
    return ForecastPredictor()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _forecast_impl(horizon: int) -> List[ForecastItem]:
    predictor = _load_predictor()
    preds = predictor.forecast(horizon=horizon)
    records = preds.to_dict(orient="records")
    return [
        ForecastItem(
            dep_code=rec["dep_code"],
            year=int(rec["year"]),
            week=int(rec["week"]),
            week_start=pd.Timestamp(rec["week_start"]).date().isoformat(),
            prediction=float(rec["prediction"]),
        )
        for rec in records
    ]


@app.get("/forecast", response_model=List[ForecastItem])
def forecast(horizon: int = Query(4, ge=1, le=6)) -> List[ForecastItem]:
    return _forecast_impl(horizon)


def _alerts_impl() -> List[AlertItem]:
    dataset = _load_dataset()
    alerts_data = compute_alerts(dataset)
    return [
        AlertItem(
            dep_code=item["dep_code"],
            week_start=pd.Timestamp(item["week_start"]).date().isoformat(),
            signal=item["signal"],
            level=item["level"],
            p80=item["p80"],
            p90=item["p90"],
        )
        for item in alerts_data
    ]


@app.get("/alerts", response_model=List[AlertItem])
def alerts() -> List[AlertItem]:
    return _alerts_impl()


def _allocation_impl(total_stock: Optional[float]) -> List[AllocationItem]:
    dataset = _load_dataset()
    coverage = (
        dataset.sort_values("week_start")
        .groupby("dep_code")
        .tail(1)[["dep_code", "coverage_rate"]]
    )
    predictor = _load_predictor()
    forecasts = predictor.forecast(horizon=4)
    allocation_data = compute_allocation(
        forecasts=forecasts,
        coverage=coverage,
        total_stock=total_stock,
    )
    return [
        AllocationItem(
            dep_code=item["dep_code"],
            week_start=pd.Timestamp(item["week_start"]).date().isoformat(),
            forecast_need=item["forecast_need"],
            coverage=item["coverage"],
            allocation=item["allocation"],
            equity_bonus=item["equity_bonus"],
        )
        for item in allocation_data
    ]


@app.get("/allocation", response_model=List[AllocationItem])
def allocation(total_stock: Optional[float] = Query(default=None, ge=0.0)) -> List[AllocationItem]:
    return _allocation_impl(total_stock)
