"""Fetch a minimal subset of datasets to feed the POC pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import io
import ssl
from typing import Dict, Optional

import pandas as pd
import requests

from src.config import RAW_DATA_DIR

ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


URGENCES_DATASET = (
    "grippe-passages-aux-urgences-et-actes-sos-medecins-departement"
)
COVERAGE_DATASET = (
    "couvertures-vaccinales-des-adolescent-et-adultes-departement"
)


def download_odisse_csv(
    dataset: str, limit: int = 5000, extra_params: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    url = (
        "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/"
        f"{dataset}/exports/csv"
    )
    params = {"limit": str(limit)}
    if extra_params:
        params.update(extra_params)
    LOGGER.info("Downloading %s with params=%s", dataset, params)
    response = requests.get(url, params=params, timeout=60, verify=False)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        LOGGER.warning(
            "HTTP error while fetching %s: %s", dataset, exc.response.text[:200]
        )
        raise
    buffer = io.BytesIO(response.content)
    df = pd.read_csv(buffer, sep=";")
    LOGGER.info("Downloaded %s rows for %s", len(df), dataset)
    return df


def generate_mock_urgences(limit: int) -> pd.DataFrame:
    LOGGER.info("Generating mock urgences dataset (offline fallback)")
    weeks = pd.date_range("2021-01-04", periods=max(10, limit // 3), freq="W-MON")
    departements = ["75", "69", "13"]
    rows = []
    for dep in departements:
        noise = pd.Series(range(len(weeks))) * 0.5
        seasonal = (pd.Series(range(len(weeks))) % 52) / 52
        base = 110 + 35 * seasonal + noise
        for week_idx, week in enumerate(weeks):
            rows.append(
                {
                    "annee": week.year,
                    "semaine": int(week.strftime("%V")),
                    "code_departement": dep,
                    "libelle_departement": f"Département {dep}",
                    "passages_urgences_total": float(
                        max(0, base.iloc[week_idx] + 25 * (dep == "75"))
                    ),
                    "passages_urgences_grippe": float(
                        max(0, base.iloc[week_idx] * 0.12 + 7 * (dep == "69"))
                    ),
                    "actes_sos_medecins_grippe": float(
                        max(0, base.iloc[week_idx] * 0.06 + 5 * (dep == "13"))
                    ),
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["code_departement", "annee", "semaine"])


def generate_mock_coverage(limit: int) -> pd.DataFrame:
    LOGGER.info("Generating mock coverage dataset (offline fallback)")
    departements = ["75", "69", "13"]
    years = [2021, 2022, 2023]
    rows = []
    for dep in departements:
        for year in years:
            rows.append(
                {
                    "annee": year,
                    "code": dep,
                    "libelle": f"Département {dep}",
                    "couverture_vaccinale": float(
                        max(
                            0,
                            43
                            + 8 * (year - years[0])
                            + (6 if dep == "75" else -4 if dep == "13" else 0),
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def fetch_and_store(
    dataset: str,
    filename: str,
    limit: int = 5000,
    extra_params: Optional[Dict[str, str]] = None,
    offline_factory=None,
) -> Path:
    try:
        df = download_odisse_csv(dataset, limit=limit, extra_params=extra_params)
    except requests.HTTPError:
        if offline_factory is None:
            raise
        LOGGER.warning(
            "Falling back to locally generated dataset for %s", dataset
        )
        df = offline_factory(limit)
    path = RAW_DATA_DIR / filename
    if filename.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    LOGGER.info("Saved %s rows to %s", len(df), path)
    return path


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetch_and_store(
        URGENCES_DATASET,
        "urgences.csv",
        limit=10000,
        extra_params={"order_by": "-semaine"},
        offline_factory=generate_mock_urgences,
    )
    fetch_and_store(
        COVERAGE_DATASET,
        "coverage.csv",
        limit=10000,
        extra_params={
            "order_by": "-annee",
            "refine": "tranche_d_age_larger:65 ans et plus",
        },
        offline_factory=generate_mock_coverage,
    )
    LOGGER.info("Sample datasets downloaded successfully.")


if __name__ == "__main__":
    main()
