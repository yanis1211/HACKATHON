from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .data_loader import DataBundle

WINTER_MONTHS = {1, 2, 11, 12}


def _prepare_monthly_dataframe(bundle: DataBundle) -> pd.DataFrame:
    monthly = bundle.monthly_metrics.copy()
    if monthly.empty:
        return monthly
    monthly = monthly.sort_values(["departement", "mois"]).reset_index(drop=True)
    monthly["ias_norm"] = monthly.groupby("departement")["incidence_mois"].transform(
        lambda s: ((s - s.mean()) / (s.std(ddof=0) + 1e-6)).clip(-2, 2)
    )
    monthly["month_dt"] = pd.to_datetime(monthly["mois"] + "-01")
    return monthly


def _project_future_month(bundle: DataBundle, target_month: str, monthly: pd.DataFrame) -> pd.DataFrame:
    last_per_dept = monthly.sort_values("month_dt").groupby("departement").tail(1)
    projected = last_per_dept.copy()
    projected["mois"] = target_month
    projected["month_dt"] = pd.to_datetime(f"{target_month}-01")
    projected = projected.drop(columns=["coverage_ma3", "coverage_trend", "flux_ma3", "flux_trend"], errors="ignore")
    projected["weeks_count"] = 0
    projected["confidence"] = "faible"
    projected["ias_norm"] = projected["ias_norm"].fillna(0)
    projected["couverture_mois"] = np.clip(
        projected["couverture_mois"].fillna(projected["couverture_mois"].mean()),
        20,
        95,
    )
    projected["flux_mois"] = projected["flux_mois"].fillna(0)
    return projected


@dataclass
class PredictionResult:
    month: str
    per_department: pd.DataFrame
    timeline: pd.DataFrame
    national: pd.DataFrame
    coverage_target_pct: float
    season_uplift_pct: float
    ias_coef: float


def predict_needs(
    bundle: DataBundle,
    month: str,
    coverage_target_pct: float,
    season_uplift_pct: float,
    ias_coef: float,
) -> PredictionResult:
    monthly = _prepare_monthly_dataframe(bundle)
    if monthly.empty:
        return PredictionResult(month, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), coverage_target_pct, season_uplift_pct, ias_coef)

    if month not in monthly["mois"].unique():
        future_row = _project_future_month(bundle, month, monthly)
        monthly = pd.concat([monthly, future_row], ignore_index=True)

    monthly = monthly.sort_values(["departement", "month_dt"]).reset_index(drop=True)
    monthly["coverage_ma3"] = monthly.groupby("departement")["couverture_mois"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    monthly["coverage_trend"] = monthly.groupby("departement")["coverage_ma3"].diff().fillna(0)
    monthly["flux_ma3"] = monthly.groupby("departement")["flux_mois"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    monthly["flux_trend"] = monthly.groupby("departement")["flux_ma3"].diff().fillna(0)

    coverage_target = coverage_target_pct / 100.0
    season_factor = season_uplift_pct / 100.0

    monthly["season_multiplier"] = monthly["month_dt"].dt.month.apply(
        lambda m: season_factor if m in WINTER_MONTHS else 0.0
    )
    future_flag = monthly["weeks_count"].fillna(0) == 0
    monthly["is_future"] = future_flag.astype(bool)
    coverage_projection = np.clip(
        monthly["coverage_ma3"] + monthly["coverage_trend"] + (ias_coef * monthly["ias_norm"] * 10.0),
        0,
        95,
    )
    monthly["coverage_used"] = monthly["couverture_mois"].where(~future_flag, coverage_projection)
    monthly.loc[future_flag, "couverture_mois"] = monthly.loc[future_flag, "coverage_used"]

    flux_projection = np.clip(monthly["flux_ma3"] + monthly["flux_trend"], 0, None)
    monthly["flux_proj"] = monthly["flux_mois"].where(~future_flag, flux_projection)
    monthly.loc[future_flag, "flux_mois"] = monthly.loc[future_flag, "flux_proj"]

    target_doses = monthly["population_proxy"] * coverage_target
    available_doses = monthly["population_proxy"] * monthly["coverage_used"] / 100.0
    monthly["besoin_prevu"] = np.clip((target_doses - available_doses) * (1 + monthly["season_multiplier"]), 0, None)

    monthly["risk_level"] = pd.cut(
        monthly["coverage_used"],
        bins=[-1, 60, 75, 200],
        labels=["rouge", "orange", "vert"],
    ).astype(str)

    timeline = monthly.sort_values("month_dt").copy()
    timeline["mois_label"] = timeline["month_dt"].dt.strftime("%Y-%m")

    per_department = timeline[timeline["mois_label"] == month].copy()
    national = (
        timeline.groupby("mois_label")
        .agg(
            couverture_moyenne=("couverture_mois", "mean"),
            besoin_total=("besoin_prevu", "sum"),
            flux_total=("flux_proj", "sum"),
        )
        .reset_index()
    )

    return PredictionResult(
        month=month,
        per_department=per_department,
        timeline=timeline,
        national=national,
        coverage_target_pct=coverage_target_pct,
        season_uplift_pct=season_uplift_pct,
        ias_coef=ias_coef,
    )


def allocation_plan(prediction: PredictionResult, stock_national: int, coverage_target_pct: float) -> pd.DataFrame:
    df = prediction.per_department.copy()
    if df.empty:
        return df
    df["ratio"] = df["besoin_prevu"] / df["besoin_prevu"].sum()
    df["allocation_proposee"] = np.floor(df["ratio"] * stock_national)
    df.loc[df["couverture_mois"] >= coverage_target_pct, "allocation_proposee"] *= 0.1
    df["allocation_proposee"] = df["allocation_proposee"].round(0).astype(int)
    df["stock_restant"] = stock_national - df["allocation_proposee"].cumsum()
    return df[[
        "departement",
        "nom",
        "besoin_prevu",
        "allocation_proposee",
        "stock_restant",
        "risk_level",
        "confidence",
    ]]
