from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _sanitize_value(value: float | int | None, fallback: str = "—") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    return str(value)


def _build_recommendation(row: pd.Series, coverage_target: float, under_threshold: int) -> str:
    coverage = row.get("couverture_vaccinale_percent", np.nan)
    risk_level = (row.get("risk_level") or "").lower()
    besoins = row.get("besoin_prevu", np.nan)
    activity = row.get("activity_total", np.nan)
    doses = row.get("doses_distribuees", np.nan)

    low_coverage = not np.isnan(coverage) and coverage < under_threshold
    below_target = not np.isnan(coverage) and coverage < coverage_target * 100
    high_need = not np.isnan(besoins) and besoins > np.nanmean([besoins, doses]) if not np.isnan(doses) else False
    high_activity = not np.isnan(activity) and activity > 250

    if risk_level == "élevé" and low_coverage:
        return "Renforcer la campagne et accélérer l’approvisionnement."
    if risk_level == "élevé":
        return "Surveiller la pression sanitaire, prévoir un renfort ciblé."
    if low_coverage and high_need:
        return "Augmenter les doses livrées et coordonner la distribution."
    if below_target and high_activity:
        return "Organiser des actions locales, activité sanitaire supérieure à la moyenne."
    if below_target:
        return "Poursuivre la promotion vaccinale pour atteindre la cible."
    return "Maintenir le suivi et ajuster selon l’évolution."


def annotate_with_suggestions(
    df: pd.DataFrame,
    *,
    coverage_target: float,
    under_threshold: int,
) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = df.copy()
    enriched["suggestion"] = enriched.apply(
        lambda row: _build_recommendation(row, coverage_target=coverage_target, under_threshold=under_threshold),
        axis=1,
    )
    return enriched


def summarise_insights(rows: Iterable[pd.Series]) -> list[str]:
    summaries: list[str] = []
    for row in rows:
        nom = row.get("nom", "Inconnu")
        suggestion = row.get("suggestion", "")
        summaries.append(f"{nom}: {suggestion}")
    return summaries
