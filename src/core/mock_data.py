from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MockConfig:
    weeks: Sequence[str]
    departements: Sequence[tuple[str, str]]
    seed: int = 42


EXCLUDED_PREFIXES = ("97", "98")


def _filter_departements(pairs: Sequence[tuple[str, str]] | None) -> list[tuple[str, str]]:
    if not pairs:
        return []
    filtered = []
    for code, name in pairs:
        code_str = str(code).strip().upper()
        if code_str.startswith(EXCLUDED_PREFIXES):
            continue
        filtered.append((code_str, name))
    return filtered


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def build_config(
    departements: Sequence[tuple[str, str]] | None = None,
    weeks: Sequence[str] | None = None,
) -> MockConfig:
    default_weeks = ["2024-45", "2024-46", "2024-47", "2024-48", "2024-49"]
    default_departements = _filter_departements([
        ("01", "Ain"),
        ("02", "Aisne"),
        ("03", "Allier"),
        ("13", "Bouches-du-Rhône"),
        ("33", "Gironde"),
        ("44", "Loire-Atlantique"),
        ("59", "Nord"),
        ("69", "Rhône"),
        ("75", "Paris"),
        ("83", "Var"),
        ("2A", "Corse-du-Sud"),
        ("2B", "Haute-Corse"),
    ])
    filtered = _filter_departements(departements)
    return MockConfig(
        weeks=weeks or default_weeks,
        departements=filtered or default_departements,
        seed=42,
    )


def mock_vaccination_trends(cfg: MockConfig) -> pd.DataFrame:
    rng = _rng(cfg.seed)
    records: list[dict[str, object]] = []
    for code, name in cfg.departements:
        coverage = rng.uniform(30, 65)
        sentiment = rng.uniform(-1.0, 1.2)
        for week in cfg.weeks:
            drift = rng.uniform(-2.0, 2.5) + sentiment * 0.4
            coverage = float(np.clip(coverage + drift, 25.0, 85.0))
            records.append(
                {
                    "departement": code,
                    "nom": name,
                    "semaine": week,
                    "couverture_vaccinale_percent": round(coverage, 2),
                }
            )
    return pd.DataFrame(records)


def mock_ias(cfg: MockConfig) -> pd.DataFrame:
    rng = _rng(cfg.seed + 1)
    rows: list[dict[str, object]] = []
    for code, name in cfg.departements:
        baseline = rng.uniform(20, 60)
        for week in cfg.weeks:
            swing = rng.gauss(0, 7)
            value = float(np.clip(baseline + swing, 5, 95))
            rows.append(
                {
                    "departement": code,
                    "nom": name,
                    "semaine": week,
                    "ias_signal": round(value, 2),
                }
            )
    return pd.DataFrame(rows)


def mock_urgences(cfg: MockConfig) -> pd.DataFrame:
    rng = _rng(cfg.seed + 2)
    data: list[dict[str, object]] = []
    for code, name in cfg.departements:
        pressure = rng.uniform(0.6, 1.6)
        for week in cfg.weeks:
            urg = int(np.clip(rng.gauss(180 * pressure, 35), 5, None))
            sos = int(np.clip(rng.gauss(220 * pressure, 40), 5, None))
            data.append(
                {
                    "departement": code,
                    "nom": name,
                    "semaine": week,
                    "urgences_grippe": urg,
                    "sos_medecins": sos,
                }
            )
    return pd.DataFrame(data)


def mock_distribution(cfg: MockConfig) -> pd.DataFrame:
    rng = _rng(cfg.seed + 3)
    rows: list[dict[str, object]] = []
    for code, name in cfg.departements:
        base = rng.uniform(3500, 8000)
        for week in cfg.weeks:
            doses = int(np.clip(rng.gauss(base, 650), 800, None))
            actes = int(np.clip(rng.gauss(base * 0.4, 180), 50, None))
            rows.append(
                {
                    "departement": code,
                    "nom": name,
                    "semaine": week,
                    "doses_distribuees": doses,
                    "actes_pharmacie": actes,
                }
            )
    return pd.DataFrame(rows)


def mock_meteo(cfg: MockConfig) -> pd.DataFrame:
    rng = _rng(cfg.seed + 4)
    data: list[dict[str, object]] = []
    for code, name in cfg.departements:
        base_temp = rng.uniform(0, 12)
        for week in cfg.weeks:
            anomaly = rng.uniform(-4, 4)
            temp_moy = base_temp + anomaly
            temp_min = temp_moy - rng.uniform(3, 8)
            humidite = np.clip(rng.gauss(70, 8), 40, 100)
            precip = np.clip(rng.gauss(25, 10), 0, 80)
            data.append(
                {
                    "departement": code,
                    "nom": name,
                    "semaine": week,
                    "temp_moy": round(temp_moy, 2),
                    "temp_min": round(temp_min, 2),
                    "humidite": round(humidite, 1),
                    "precipitations": round(precip, 1),
                    "anomalie_temp": round(anomaly, 2),
                }
            )
    return pd.DataFrame(data)


def mock_coverage(cfg: MockConfig) -> pd.DataFrame:
    df = mock_vaccination_trends(cfg)
    latest = df.sort_values("semaine").groupby("departement").tail(1)
    return latest[["departement", "nom", "couverture_vaccinale_percent"]].reset_index(drop=True)


def generate_dataset(name: str, cfg: MockConfig) -> pd.DataFrame:
    name = name.lower()
    if name == "vaccination_trends":
        return mock_vaccination_trends(cfg)
    if name == "ias":
        return mock_ias(cfg)
    if name == "urgences":
        return mock_urgences(cfg)
    if name == "distribution":
        return mock_distribution(cfg)
    if name == "meteo":
        return mock_meteo(cfg)
    if name == "coverage":
        return mock_coverage(cfg)
    raise ValueError(f"Unsupported mock dataset: {name}")


def ensure_weeks(values: Iterable[str]) -> list[str]:
    uniques = sorted({str(v) for v in values if isinstance(v, str)})
    return uniques or build_config().weeks
