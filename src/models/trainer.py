"""Train baseline Gradient Boosting model for flu demand forecasting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR


@dataclass
class TrainingConfig:
    processed_dataset: Path = PROCESSED_DATA_DIR / "training_dataset.csv"
    model_path: Path = MODELS_DIR / "gbm_forecast.joblib"
    metrics_path: Path = REPORTS_DIR / "metrics.json"
    forecast_horizon_weeks: int = 4


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "y",
        "dep_code",
        "year",
        "week",
        "week_start",
        "year_week",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in excluded]


def _train_model(
    train_df: pd.DataFrame, feature_cols: List[str]
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        random_state=42,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
    )
    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["y"].astype(float)
    model.fit(X_train, y_train)
    return model


def _evaluate_model(
    model: GradientBoostingRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> dict:
    if df.empty:
        return {"mae": None, "rmse": None, "samples": 0}
    X = df[feature_cols].fillna(0.0)
    y_true = df["y"].astype(float)
    y_pred = model.predict(X)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse, "samples": len(df)}


def train(config: TrainingConfig = TrainingConfig()) -> dict:
    df = pd.read_csv(config.processed_dataset, parse_dates=["week_start"])
    df = df.sort_values(["dep_code", "week_start"])
    feature_cols = _select_feature_columns(df)

    cutoff = df["week_start"].max() - pd.Timedelta(
        weeks=config.forecast_horizon_weeks
    )
    train_df = df[df["week_start"] <= cutoff]
    test_df = df[df["week_start"] > cutoff]
    if train_df.empty:
        raise ValueError("Training dataset is empty after applying cutoff.")

    model = _train_model(train_df, feature_cols)
    metrics = {
        "train": _evaluate_model(model, train_df, feature_cols),
        "test": _evaluate_model(model, test_df, feature_cols),
        "feature_importances": dict(
            zip(feature_cols, map(float, model.feature_importances_))
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model_path)
    with config.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    metrics = train()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
