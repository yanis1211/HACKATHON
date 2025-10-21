from __future__ import annotations

import pandas as pd
import streamlit as st

from core.model import allocation_plan, PredictionResult
from .components import download_button


def render_distribution_view(prediction: PredictionResult, default_stock: int) -> None:
    st.markdown("### Allocation des doses")
    df = prediction.per_department.copy()
    if df.empty:
        st.warning("Données indisponibles pour cette période.")
        return

    st.markdown(
        "- **Comment lire :** colonne `besoin_prevu` = doses à couvrir, `allocation_proposee` = suggestion automatique.\n"
        "- **Hypothèse :** répartition proportionnelle aux besoins, plafonnée si couverture ≥ cible.\n"
        "- **Action :** valider / ajuster l’allocation avant la prochaine livraison."
    )

    col1, col2 = st.columns(2)
    with col1:
        stock_input = st.number_input("Stock national disponible", min_value=0, value=int(default_stock), step=1000)
    with col2:
        cible_input = st.slider("Couverture visée (%)", min_value=50, max_value=90, value=int(prediction.coverage_target_pct), step=1)
        st.caption("Plafonnement automatique des départements déjà ≥ cible.")

    if st.button("Calculer l'allocation"):
        plan = allocation_plan(prediction, stock_input, cible_input)
        if plan.empty:
            st.success("Aucun besoin de redistribution détecté.")
        else:
            if "is_future" in df.columns:
                plan = plan.merge(
                    df[["departement", "is_future"]],
                    on="departement",
                    how="left",
                )
                plan["type"] = plan["is_future"].map({True: "Prévision", False: "Historique"})
                plan.drop(columns=["is_future"], inplace=True)
            st.dataframe(plan, use_container_width=True, hide_index=True)
            download_button("⬇️ Exporter l'allocation", plan, f"allocation_{prediction.month}.csv")
        st.caption("Action : redistribuer les doses avant saturation des zones rouges.")
    else:
        st.caption("Indiquez un stock national puis calculez l’allocation proposée.")
