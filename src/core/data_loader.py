from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .mock_data import MockConfig, build_config, ensure_weeks, generate_dataset


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "manual"

CSV_DATASETS = ["vaccination_trends", "ias", "urgences", "distribution", "coverage", "meteo"]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, low_memory=False)


def normalize_departement(df: pd.DataFrame, column: str = "departement") -> pd.DataFrame:
    if column not in df.columns:
        return df
    df[column] = df[column].astype(str).str.strip().str.upper()
    mask = df[column].str.contains(r"[A-Z]$")
    df.loc[~mask, column] = df.loc[~mask, column].str.zfill(2)
    return df


def normalize_week(df: pd.DataFrame, column: str = "semaine") -> pd.DataFrame:
    if column not in df.columns:
        return df
    df[column] = df[column].astype(str).str.upper().str.replace("W", "", regex=False)
    return df


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_departement(df, "departement")
    df = normalize_week(df, "semaine")
    return df


def _load_geojson(path: Path) -> dict:
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_departements_from_geojson(geojson: dict) -> List[Tuple[str, str]]:
    departements: list[tuple[str, str]] = []
    features = geojson.get("features", [])
    for feature in features:
        props = feature.get("properties") or {}
        code = str(props.get("code") or props.get("code_insee") or "").strip()
        name = str(props.get("nom") or props.get("name") or code)
        if code:
            departements.append((code.upper(), name))
    return departements


def _extract_departements_from_frames(frames: Iterable[pd.DataFrame]) -> List[Tuple[str, str]]:
    mapping: Dict[str, str] = {}
    for frame in frames:
        if frame is None or frame.empty or "departement" not in frame.columns:
            continue
        names = frame.get("nom")
        if names is not None:
            for code, nom in zip(frame["departement"], names.fillna(""), strict=False):
                code = str(code).strip().upper()
                if not code:
                    continue
                if code not in mapping or not mapping[code]:
                    mapping[code] = str(nom) if str(nom) else code
        else:
            for code in frame["departement"]:
                code = str(code).strip().upper()
                mapping.setdefault(code, code)
    return sorted(mapping.items())


@dataclass(frozen=True)
class DataBundle:
    vaccination_trends: pd.DataFrame
    ias: pd.DataFrame
    urgences: pd.DataFrame
    distribution: pd.DataFrame
    coverage: pd.DataFrame
    meteo: pd.DataFrame
    geojson: dict
    departements: List[Tuple[str, str]]
    weeks: List[str]
    months: List[str]
    monthly_metrics: pd.DataFrame

    @property
    def has_meteo(self) -> bool:
        return not self.meteo.empty


def _load_csv_dataset(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        return None
    df = _read_csv(path)
    df = _normalize_dataframe(df)
    return df


def _rename_standard_columns(name: str, df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        low = column.lower()
        if low in {"departement code", "code departement", "code_departement", "dep_code", "code"}:
            rename_map[column] = "departement"
        elif low in {"departement", "département", "num_dep", "numero_departement", "dept", "dep", "code_insee"}:
            if "departement" not in df.columns:
                rename_map[column] = "departement"
        elif low in {"departement", "département"} and "nom" not in df.columns:
            rename_map[column] = "nom"
        elif name in {"vaccination_trends", "coverage"} and low.startswith("couverture"):
            rename_map[column] = "couverture_vaccinale_percent"
        elif name in {"vaccination_trends", "coverage"} and low in {"semaine", "week", "periode"}:
            rename_map[column] = "semaine"
        elif name == "ias":
            if "ias" in low and "ias_signal" not in df.columns:
                rename_map[column] = "ias_signal"
            elif low in {"semaine", "week", "periode"}:
                rename_map[column] = "semaine"
        elif name == "urgences":
            if "urgences" in low and "urgences_grippe" not in df.columns:
                rename_map[column] = "urgences_grippe"
            elif "sos" in low and "sos_medecins" not in df.columns:
                rename_map[column] = "sos_medecins"
            elif low in {"semaine", "week", "periode"}:
                rename_map[column] = "semaine"
        elif name == "distribution":
            if "dose" in low and "doses_distribuees" not in df.columns:
                rename_map[column] = "doses_distribuees"
            elif "acte" in low and "pharm" in low and "actes_pharmacie" not in df.columns:
                rename_map[column] = "actes_pharmacie"
            elif low in {"semaine", "week", "periode"}:
                rename_map[column] = "semaine"
        elif name == "meteo":
            if low in {"semaine", "week", "periode"}:
                rename_map[column] = "semaine"
            elif low in {"departement", "département", "num_dep", "numero_departement", "dept", "dep", "code_insee"}:
                rename_map[column] = "departement"
            elif "temp" in low and "moy" in low and "temp_moy" not in df.columns:
                rename_map[column] = "temp_moy"
            elif "temp" in low and "min" in low and "temp_min" not in df.columns:
                rename_map[column] = "temp_min"
            elif "humid" in low and "humidite" not in df.columns:
                rename_map[column] = "humidite"
            elif ("pluie" in low or "precip" in low) and "precipitations" not in df.columns:
                rename_map[column] = "precipitations"
            elif "anom" in low and "temp" in low and "anomalie_temp" not in df.columns:
                rename_map[column] = "anomalie_temp"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _generate_missing_dataset(name: str, cfg: MockConfig) -> pd.DataFrame:
    df = generate_dataset(name, cfg)
    return _normalize_dataframe(df)


def load_data_bundle() -> DataBundle:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    geojson_path = DATA_DIR / "departements.geojson"
    geojson = _load_geojson(geojson_path)

    initial_frames: Dict[str, pd.DataFrame | None] = {}
    for name in CSV_DATASETS:
        df = _load_csv_dataset(name)
        if df is not None:
            df = _rename_standard_columns(name, df)
        initial_frames[name] = df

    departements = _extract_departements_from_geojson(geojson)
    if not departements:
        departements = _extract_departements_from_frames(df for df in initial_frames.values() if df is not None)
    cfg = build_config(departements=departements or None)

    existing_weeks: list[str] = []
    for key in ["vaccination_trends", "ias", "urgences", "distribution"]:
        df = initial_frames.get(key)
        if df is not None and "semaine" in df.columns:
            existing_weeks.extend(df["semaine"].dropna().astype(str).tolist())
    weeks = ensure_weeks(existing_weeks) or cfg.weeks

    if not departements:
        departements = cfg.departements
    else:
        departements = sorted({(code, name) for code, name in departements})

    cfg = build_config(departements=departements, weeks=weeks)

    final_frames: Dict[str, pd.DataFrame] = {}
    for name in CSV_DATASETS:
        df = initial_frames.get(name)
        if df is None or df.empty:
            df = _generate_missing_dataset(name, cfg)
        else:
            df = df.copy()
        final_frames[name] = df

    vaccination = final_frames["vaccination_trends"]
    ias = final_frames["ias"]
    urgences = final_frames["urgences"]
    distribution = final_frames["distribution"]
    coverage = final_frames.get("coverage", pd.DataFrame())
    meteo = final_frames.get("meteo", pd.DataFrame())

    if coverage.empty and not vaccination.empty:
        coverage = (
            vaccination.sort_values("semaine")
            .groupby("departement")
            .tail(1)[["departement", "nom", "couverture_vaccinale_percent"]]
            .reset_index(drop=True)
        )

    monthly_metrics = _compute_monthly_metrics(
        vaccination,
        ias,
        urgences,
        distribution,
    )
    months = sorted(monthly_metrics["mois"].unique().tolist())

    bundle = DataBundle(
        vaccination_trends=vaccination,
        ias=ias,
        urgences=urgences,
        distribution=distribution,
        coverage=coverage,
        meteo=meteo if not meteo.empty else pd.DataFrame(),
        geojson=geojson,
        departements=list(departements),
        weeks=list(weeks),
        months=months,
        monthly_metrics=monthly_metrics,
    )
    return bundle


def generate_future_months(last_month: str, horizon: int = 4) -> List[str]:
    if horizon <= 0:
        return []

    year, month = map(int, last_month.split("-"))
    current = date(year, month, 1)
    results: List[str] = []
    for _ in range(horizon):
        next_month = current + timedelta(days=32)
        next_month = next_month.replace(day=1)
        results.append(f"{next_month.year}-{next_month.month:02d}")
        current = next_month
    return results


def _week_to_month(week_str: str) -> str:
    year, week = map(int, week_str.split("-"))
    start = date.fromisocalendar(year, week, 1)
    end = start + timedelta(days=6)
    return f"{end.year}-{end.month:02d}"


def _winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    if series.empty:
        return series
    low_val = series.quantile(lower)
    high_val = series.quantile(upper)
    return series.clip(lower=low_val, upper=high_val)


def _compute_monthly_metrics(
    vaccination: pd.DataFrame,
    ias: pd.DataFrame,
    urgences: pd.DataFrame,
    distribution: pd.DataFrame,
) -> pd.DataFrame:
    if vaccination.empty:
        return pd.DataFrame()

    merged = vaccination.merge(
        ias,
        on=["departement", "semaine"],
        how="left",
        suffixes=("", "_ias"),
    ).merge(
        urgences,
        on=["departement", "semaine"],
        how="left",
    ).merge(
        distribution,
        on=["departement", "semaine"],
        how="left",
        suffixes=("", "_dist"),
    )

    merged["activity_total"] = merged["urgences_grippe"].fillna(0) + merged["sos_medecins"].fillna(0)
    merged["population_proxy"] = (
        merged["doses_distribuees"].fillna(0)
        + merged["activity_total"].fillna(0) * 25
        + 5000
    )
    merged["population_proxy"] = merged["population_proxy"].replace({0: 1_000})

    merged["mois"] = merged["semaine"].apply(_week_to_month)

    merged["ias_signal"] = merged.groupby("departement")["ias_signal"].transform(_winsorize)

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        weights = group["population_proxy"].fillna(group["population_proxy"].mean()).replace({0: 1})
        couverture = np.average(
            group["couverture_vaccinale_percent"].fillna(group["couverture_vaccinale_percent"].mean()),
            weights=weights,
        )
        incidence = group["ias_signal"].mean()
        flux = group["activity_total"].sum()
        population = weights.sum()
        weeks_count = group["semaine"].nunique()
        urgency = group["urgences_grippe"].sum()
        sos = group["sos_medecins"].sum()
        dist = group["doses_distribuees"].sum()
        actes = group["actes_pharmacie"].sum()
        return pd.Series(
            {
                "nom": group["nom"].iloc[0] if "nom" in group.columns else "",
                "couverture_mois": couverture,
                "incidence_mois": incidence,
                "flux_mois": flux,
                "population_proxy": population,
                "urgences_mois": urgency,
                "sos_mois": sos,
                "doses_mois": dist,
                "actes_mois": actes,
                "weeks_count": weeks_count,
            }
        )

    monthly = merged.groupby(["departement", "mois"], as_index=False).apply(_aggregate)

    monthly["couverture_mois"] = monthly.groupby("departement")["couverture_mois"].transform(
        lambda s: s.fillna(s.mean())
    )
    monthly["incidence_mois"] = monthly.groupby("departement")["incidence_mois"].transform(
        lambda s: s.fillna(s.mean())
    )

    overall_cov = monthly["couverture_mois"].mean()
    overall_inc = monthly["incidence_mois"].mean()
    monthly["couverture_mois"] = monthly["couverture_mois"].fillna(overall_cov)
    monthly["incidence_mois"] = monthly["incidence_mois"].fillna(overall_inc)

    monthly["flux_mois"] = monthly["flux_mois"].fillna(0)
    monthly["population_proxy"] = monthly["population_proxy"].replace({0: 1_000})

    monthly["mois"] = monthly["mois"].astype(str)
    monthly["departement"] = monthly["departement"].astype(str)

    monthly = monthly.sort_values(["departement", "mois"]).reset_index(drop=True)
    monthly["flux_trend"] = monthly.groupby("departement")["flux_mois"].diff().fillna(0)
    monthly["couverture_trend"] = monthly.groupby("departement")["couverture_mois"].diff().fillna(0)

    counts = monthly.groupby("departement")["mois"].transform("count")
    monthly["confidence"] = np.where(counts >= 9, "élevée", np.where(counts >= 5, "moyenne", "faible"))

    return monthly

    weeks: List[str] = []
    current = last_week
    for _ in range(horizon):
        current = _next_week(current)
        weeks.append(current)
    return weeks
