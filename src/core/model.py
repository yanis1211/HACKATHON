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
    projected["couverture_mois"] = np.clip(
        projected["couverture_mois"] + projected["couverture_trend"],
        20,
        95,
    )
    projected["incidence_mois"] = projected["incidence_mois"].fillna(projected["incidence_mois"].mean())
    projected["flux_mois"] = np.clip(projected["flux_mois"] + projected["flux_trend"], 0, None)
    projected["weeks_count"] = 0
    projected["ias_norm"] = projected["ias_norm"].fillna(0)
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

    coverage_target = coverage_target_pct / 100.0
    season_factor = season_uplift_pct / 100.0

    monthly["season_multiplier"] = monthly["month_dt"].apply(
        lambda dt: season_factor if dt.month in WINTER_MONTHS else 0.0
    )
    monthly["coverage_proj"] = np.clip(
        monthly["couverture_mois"] + (ias_coef * monthly["ias_norm"] * 10.0),
        0,
        95,
    )
    target_doses = monthly["population_proxy"] * coverage_target
    available_doses = monthly["population_proxy"] * monthly["coverage_proj"] / 100.0
    monthly["besoin_prevu"] = np.clip((target_doses - available_doses) * (1 + monthly["season_multiplier"]), 0, None)
    monthly["flux_proj"] = np.clip(monthly["flux_mois"] + monthly["flux_trend"].fillna(0), 0, None)

    monthly["risk_level"] = pd.cut(
        monthly["couverture_mois"],
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
