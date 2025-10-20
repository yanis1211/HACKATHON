"""Feature engineering and dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


@dataclass
class DatasetPaths:
    urgences: Path = RAW_DATA_DIR / "urgences.csv"
    coverage: Path = RAW_DATA_DIR / "coverage.csv"
    output: Path = PROCESSED_DATA_DIR / "training_dataset.csv"


def _canonicalize_urgences(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "code_departement": "dep_code",
        "code": "dep_code",
        "annee": "year",
        "semaine": "week",
        "passages_urgences_grippe": "y",
        "passages_urgences_total": "urgences_total",
        "passages_urgences": "urgences_total",
        "passages_grippe": "y",
        "actes_sos_medecins_grippe": "sos_grippe",
        "sos_medecins_grippe": "sos_grippe",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "dep_code" not in df.columns:
        raise ValueError("Urgences dataset must include a departement code column.")
    if "year" not in df.columns or "week" not in df.columns:
        raise ValueError("Urgences dataset must include year and week columns.")
    if "y" not in df.columns:
        raise ValueError("Urgences dataset must include a target metric (grippe).")
    relevant_cols = [
        "dep_code",
        "year",
        "week",
        "y",
        "urgences_total",
        "sos_grippe",
    ]
    for col in relevant_cols:
        if col not in df.columns:
            df[col] = np.nan
    df["dep_code"] = df["dep_code"].astype(str).str.zfill(2)
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df = df.sort_values(["dep_code", "year", "week"])
    df["week_start"] = pd.to_datetime(
        df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "-1",
        format="%G%V-%u",
        errors="coerce",
    )
    return df


def _canonicalize_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "code": "dep_code",
        "code_departement": "dep_code",
        "annee": "year",
        "couverture_vaccinale": "coverage_rate",
        "taux_couverture": "coverage_rate",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if {"dep_code", "year", "coverage_rate"} - set(df.columns):
        raise ValueError("Coverage dataset missing required columns.")
    df["dep_code"] = df["dep_code"].astype(str).str.zfill(2)
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["dep_code", "year"])
    return df[["dep_code", "year", "coverage_rate"]]


def _add_lag_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = (
                df.groupby("dep_code")[col].shift(lag).astype(float)
            )
    return df


def _add_rolling_features(
    df: pd.DataFrame, col: str, windows: List[int]
) -> pd.DataFrame:
    for window in windows:
        df[f"{col}_roll{window}"] = (
            df.groupby("dep_code")[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df


def _add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    df["week_norm"] = (df["week"] - 1) / 52.0
    df["sin_week"] = np.sin(2 * np.pi * df["week_norm"])
    df["cos_week"] = np.cos(2 * np.pi * df["week_norm"])
    df["is_winter"] = df["week"].apply(lambda w: 1 if (w >= 44 or w <= 8) else 0)
    return df


def build_training_dataset(paths: DatasetPaths = DatasetPaths()) -> Path:
    urgences = pd.read_csv(paths.urgences)
    coverage = pd.read_csv(paths.coverage)

    urgences = _canonicalize_urgences(urgences)
    coverage = _canonicalize_coverage(coverage)

    features = urgences.merge(
        coverage, on=["dep_code", "year"], how="left", validate="m:1"
    )
    features["coverage_rate"] = features.groupby("dep_code")["coverage_rate"].ffill()

    features = _add_lag_features(
        features, cols=["y", "urgences_total", "sos_grippe"], lags=[1, 2, 3, 4]
    )
    features = _add_rolling_features(features, col="y", windows=[2, 4, 8])
    features = _add_seasonality_features(features)
    features["month"] = features["week_start"].dt.month
    features["year_week"] = (
        features["year"].astype(str)
        + "-"
        + features["week"].astype(str).str.zfill(2)
    )

    features = features.dropna(subset=["y_lag1", "y_lag2"])

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(paths.output, index=False)
    return paths.output


def main() -> None:
    output = build_training_dataset()
    print(f"Wrote training dataset to {output}")


if __name__ == "__main__":
    main()
