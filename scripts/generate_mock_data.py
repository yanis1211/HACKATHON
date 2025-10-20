"""Utility script to generate synthetic datasets for the influenza vaccination POC.

The script reads the administrative boundaries GeoJSON to extract the list of
French departments and then creates several CSV files with simulated metrics.
Running this script is optional: the generated CSVs are committed so the Streamlit
application can run without regenerating them, but keeping the script makes the
process reproducible if the synthetic data ever need to be refreshed.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "manual"

GEOJSON_PATH = DATA_DIR / "departements.geojson"
OUTPUT_COVERAGE = DATA_DIR / "coverage.csv"
OUTPUT_VACC_TRENDS = DATA_DIR / "vaccination_trends.csv"
OUTPUT_DISTRIBUTION = DATA_DIR / "distribution.csv"
OUTPUT_IAS = DATA_DIR / "ias.csv"
OUTPUT_URGENCES = DATA_DIR / "urgences.csv"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def main() -> None:
    with GEOJSON_PATH.open("r", encoding="utf-8") as fh:
        geojson = json.load(fh)

    features = geojson["features"]
    rng = random.Random(26)

    weeks = ["2024-44", "2024-45", "2024-46", "2024-47", "2024-48"]

    coverage_rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    ias_rows: list[dict[str, object]] = []
    urgences_rows: list[dict[str, object]] = []

    for feature in features:
        props = feature["properties"]
        dep_code = str(props["code"])
        dep_name = props["nom"]

        # Create a synthetic baseline for each department.
        population_factor = rng.uniform(0.6, 1.4)
        baseline_cov = rng.uniform(28.0, 65.0)
        sentiment = rng.uniform(-1.0, 1.2)  # drives demand (negative -> more need)

        coverage_history: list[float] = []
        for week in weeks:
            # Coverage evolves gradually with some noise and tendency to saturate.
            baseline_cov += rng.uniform(-1.5, 2.5) + sentiment * 0.3
            baseline_cov = clamp(baseline_cov, 22.0, 80.0)
            coverage_history.append(round(baseline_cov, 2))

            demand_pressure = clamp(1.4 - baseline_cov / 100.0 + sentiment * 0.05, 0.6, 1.8)
            ias_signal = clamp(
                rng.gauss(mu=45.0 + (1 - baseline_cov / 100.0) * 35.0 + sentiment * 8.0, sigma=6.5),
                5.0,
                95.0,
            )
            doses_distrib = int(
                clamp(
                    rng.gauss(mu=5200 * population_factor * demand_pressure, sigma=650),
                    500.0,
                    math.inf,
                )
            )
            actes_pharmacie = int(
                clamp(
                    rng.gauss(mu=2100 * population_factor * (baseline_cov / 100.0 + 0.35), sigma=250),
                    50.0,
                    math.inf,
                )
            )
            urgences = int(
                clamp(
                    rng.gauss(mu=180 * population_factor * demand_pressure, sigma=35),
                    5.0,
                    math.inf,
                )
            )
            sos = int(
                clamp(
                    rng.gauss(mu=220 * population_factor * demand_pressure, sigma=40),
                    5.0,
                    math.inf,
                )
            )

            trend_rows.append(
                {
                    "departement": dep_code,
                    "nom": dep_name,
                    "semaine": week,
                    "couverture_vaccinale_percent": coverage_history[-1],
                }
            )
            distribution_rows.append(
                {
                    "departement": dep_code,
                    "nom": dep_name,
                    "semaine": week,
                    "doses_distribuees": doses_distrib,
                    "actes_pharmacie": actes_pharmacie,
                }
            )
            ias_rows.append(
                {
                    "departement": dep_code,
                    "nom": dep_name,
                    "semaine": week,
                    "ias_signal": round(ias_signal, 2),
                }
            )
            urgences_rows.append(
                {
                    "departement": dep_code,
                    "nom": dep_name,
                    "semaine": week,
                    "urgences_grippe": urgences,
                    "sos_medecins": sos,
                }
            )

        coverage_rows.append(
            {
                "departement": dep_code,
                "nom": dep_name,
                "couverture_vaccinale_percent": coverage_history[-1],
            }
        )

    OUTPUT_COVERAGE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_COVERAGE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["departement", "nom", "couverture_vaccinale_percent"],
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    with OUTPUT_VACC_TRENDS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["departement", "nom", "semaine", "couverture_vaccinale_percent"],
        )
        writer.writeheader()
        writer.writerows(trend_rows)

    with OUTPUT_DISTRIBUTION.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["departement", "nom", "semaine", "doses_distribuees", "actes_pharmacie"],
        )
        writer.writeheader()
        writer.writerows(distribution_rows)

    with OUTPUT_IAS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["departement", "nom", "semaine", "ias_signal"],
        )
        writer.writeheader()
        writer.writerows(ias_rows)

    with OUTPUT_URGENCES.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["departement", "nom", "semaine", "urgences_grippe", "sos_medecins"],
        )
        writer.writeheader()
        writer.writerows(urgences_rows)

    print("Synthetic datasets generated in:", DATA_DIR)


if __name__ == "__main__":
    main()
