"""Prediction utilities for serving forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.models.trainer import _select_feature_columns


@dataclass
class PredictorConfig:
    model_path: Path = MODELS_DIR / "gbm_forecast.joblib"
    processed_dataset: Path = PROCESSED_DATA_DIR / "training_dataset.csv"
    horizon: int = 4


class ForecastPredictor:
    def __init__(self, config: PredictorConfig = PredictorConfig()):
        self.config = config
        self.model = joblib.load(config.model_path)
        self.dataset = pd.read_csv(config.processed_dataset, parse_dates=["week_start"])
        self.dataset = self.dataset.sort_values(["dep_code", "week_start"])
        self.dataset["dep_code"] = self.dataset["dep_code"].astype(str).str.zfill(2)
        self.feature_cols = _select_feature_columns(self.dataset)

    def forecast(self, horizon: int | None = None) -> pd.DataFrame:
        horizon = horizon or self.config.horizon
        outputs = []
        for dep_code, dep_df in self.dataset.groupby("dep_code"):
            history = dep_df.copy()
            y_history = history["y"].tolist()
            urg_history = history["urgences_total"].tolist()
            sos_history = history["sos_grippe"].tolist()
            coverage = history["coverage_rate"].ffill().iloc[-1]
            current_week_start = pd.Timestamp(history["week_start"].iloc[-1])

            def _lag(arr: List[float], offset: int) -> float:
                arr = [float(x) for x in arr if not pd.isna(x)]
                if not arr:
                    return 0.0
                if len(arr) < offset:
                    return float(arr[0])
                return float(arr[-offset])

            dep_predictions = []
            for step in range(1, horizon + 1):
                next_week_start = current_week_start + timedelta(weeks=step)
                year, week, _ = next_week_start.isocalendar()
                features = {
                    "urgences_total": float(urg_history[-1]),
                    "sos_grippe": float(sos_history[-1]),
                    "coverage_rate": float(coverage),
                    "y_lag1": _lag(y_history, 1),
                    "y_lag2": _lag(y_history, 2),
                    "y_lag3": _lag(y_history, 3),
                    "y_lag4": _lag(y_history, 4),
                    "urgences_total_lag1": _lag(urg_history, 1),
                    "urgences_total_lag2": _lag(urg_history, 2),
                    "urgences_total_lag3": _lag(urg_history, 3),
                    "urgences_total_lag4": _lag(urg_history, 4),
                    "sos_grippe_lag1": _lag(sos_history, 1),
                    "sos_grippe_lag2": _lag(sos_history, 2),
                    "sos_grippe_lag3": _lag(sos_history, 3),
                    "sos_grippe_lag4": _lag(sos_history, 4),
                }
                y_tail = y_history[-8:] if len(y_history) >= 8 else y_history
                for window in (2, 4, 8):
                    subset = y_tail[-window:] if len(y_tail) >= window else y_tail
                    features[f"y_roll{window}"] = float(np.mean(subset or [0.0]))

                week_norm = (week - 1) / 52.0
                features["week_norm"] = week_norm
                features["sin_week"] = float(np.sin(2 * np.pi * week_norm))
                features["cos_week"] = float(np.cos(2 * np.pi * week_norm))
                features["is_winter"] = int(week >= 44 or week <= 8)
                features["month"] = next_week_start.month

                feature_vector = (
                    pd.DataFrame([features])[self.feature_cols].fillna(0.0)
                )
                prediction = float(self.model.predict(feature_vector)[0])
                dep_predictions.append(
                    {
                        "dep_code": str(dep_code),
                        "year": int(year),
                        "week": int(week),
                        "week_start": next_week_start,
                        "prediction": prediction,
                    }
                )
                y_history.append(prediction)
                urg_history.append(urg_history[-1])
                sos_history.append(sos_history[-1])
            outputs.append(pd.DataFrame(dep_predictions))
        return pd.concat(outputs, ignore_index=True)
