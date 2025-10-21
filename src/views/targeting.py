from __future__ import annotations

import pandas as pd
import streamlit as st

from .components import choropleth_map, download_button


def render(
    week_df: pd.DataFrame,
    geojson: dict,
    coverage_threshold: float,
) -> None:
    st.subheader("Vue Ciblage territorial – sous-vaccination")
    if week_df.empty:
        st.warning("Pas de données disponibles pour cette semaine.")
        return

    df = week_df.copy()
    df["sous_vaccination"] = df["couverture_vaccinale_percent"].fillna(0) < coverage_threshold
    df["priorite_score"] = (
        (df["sous_vaccination"].astype(int) * 2)
        + (df["risk_level"].map({"élevé": 2, "modéré": 1, "faible": 0}).fillna(0))
        + (df["besoin_prevu"] / (df["besoin_prevu"].max() + 1e-6))
    )

    st.plotly_chart(
        choropleth_map(df, geojson, "couverture_vaccinale_percent", "Couverture vaccinale (%)", palette="YlGnBu"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    targets = df[df["sous_vaccination"]].copy()
    targets = targets.sort_values(["priorite_score", "risk_score"], ascending=[False, False])
    display_cols = [
        "departement",
        "nom",
        "couverture_vaccinale_percent",
        "besoin_prevu",
        "activity_total",
        "risk_level",
        "suggestion",
    ]

    st.markdown(f"#### Départements < {coverage_threshold:.0f} % de couverture")
    if targets.empty:
        st.success("Aucun département sous le seuil paramétré.")
    else:
        st.dataframe(targets[display_cols], use_container_width=True, hide_index=True)
        download_button(
            "⬇️ Exporter la liste cible",
            targets[display_cols],
            f"ciblage_{df['semaine'].iloc[0]}.csv",
        )
