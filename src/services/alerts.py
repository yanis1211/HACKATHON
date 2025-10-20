"""Risk alert computation based on recent activity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class AlertConfig:
    metric: str = "y"
    smoothing_window: int = 3
    high_quantile: float = 0.9
    medium_quantile: float = 0.8


def compute_alerts(
    df: pd.DataFrame, config: AlertConfig = AlertConfig()
) -> List[Dict]:
    if df.empty:
        return []
    df = df.copy()
    df["signal"] = (
        df.groupby("dep_code")[config.metric]
        .rolling(window=config.smoothing_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    alerts: List[Dict] = []
    thresholds = (
        df.groupby("dep_code")[config.metric]
        .quantile([config.medium_quantile, config.high_quantile])
        .unstack(level=-1)
        .rename(
            columns={
                config.medium_quantile: "p80",
                config.high_quantile: "p90",
            }
        )
    )

    latest = df.sort_values("week_start").groupby("dep_code").tail(1)
    latest = latest.merge(
        thresholds, left_on="dep_code", right_index=True, how="left"
    )

    for _, row in latest.iterrows():
        level = "normal"
        if row["signal"] >= row.get("p90", float("inf")):
            level = "high"
        elif row["signal"] >= row.get("p80", float("inf")):
            level = "medium"
        alerts.append(
            {
                "dep_code": str(row["dep_code"]).zfill(2),
                "week_start": row["week_start"],
                "signal": float(row["signal"]),
                "level": level,
                "p80": float(row.get("p80", 0) or 0),
                "p90": float(row.get("p90", 0) or 0),
            }
        )
    return alerts
