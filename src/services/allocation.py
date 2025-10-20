"""Heuristic allocation engine to anticipate vaccine distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AllocationConfig:
    target_coverage: float = 60.0  # %
    equity_bonus_weight: float = 0.2
    min_share: float = 0.02


def compute_allocation(
    forecasts: pd.DataFrame,
    coverage: pd.DataFrame,
    total_stock: Optional[float] = None,
    config: AllocationConfig = AllocationConfig(),
) -> List[Dict]:
    if forecasts.empty:
        return []
    latest_forecast = forecasts.sort_values("week_start").groupby("dep_code").tail(1)
    coverage = coverage.rename(columns={"coverage_rate": "coverage"})
    coverage["dep_code"] = coverage["dep_code"].astype(str).str.zfill(2)
    merged = latest_forecast.merge(
        coverage, on="dep_code", how="left", validate="1:1"
    )

    merged["coverage"] = merged["coverage"].fillna(config.target_coverage)
    merged["need"] = merged["prediction"].clip(lower=0)
    merged["equity_bonus"] = (
        (config.target_coverage - merged["coverage"]).clip(lower=0)
        * merged["need"]
        * config.equity_bonus_weight
    )
    merged["weighted_need"] = merged["need"] + merged["equity_bonus"]
    if merged["weighted_need"].sum() == 0:
        merged["weighted_need"] = 1

    total_stock = total_stock or float(merged["need"].sum())
    shares = merged["weighted_need"] / merged["weighted_need"].sum()
    shares = shares.clip(lower=config.min_share)
    shares = shares / shares.sum()
    merged["allocation"] = shares * total_stock

    results: List[Dict] = []
    for _, row in merged.iterrows():
        results.append(
            {
                "dep_code": str(row["dep_code"]).zfill(2),
                "week_start": row["week_start"],
                "forecast_need": float(row["need"]),
                "coverage": float(row["coverage"]),
                "allocation": float(row["allocation"]),
                "equity_bonus": float(row["equity_bonus"]),
            }
        )
    return results
