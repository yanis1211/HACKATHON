from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .components import choropleth_map, download_button


def render(week_df: pd.DataFrame, geojson: dict) -> None:
    if "temp_moy" not in week_df.columns:
        st.info("Aucune donnée météo disponible pour cette semaine.")
        return

    st.subheader("Vue Environnement – météo & dynamique virale")

    st.plotly_chart(
        choropleth_map(week_df, geojson, "temp_moy", "Température moyenne (°C)", palette="RdBu_r"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    scatter = px.scatter(
        week_df,
        x="temp_moy",
        y="activity_total",
        color="couverture_vaccinale_percent",
        color_continuous_scale="YlGnBu",
        labels={"temp_moy": "Température moyenne (°C)", "activity_total": "Urgences + SOS"},
        title="Température vs activité grippe",
    )
    st.plotly_chart(scatter, use_container_width=True, config={"displayModeBar": False})

    export_cols = [
        "departement",
        "nom",
        "semaine",
        "temp_moy",
        "temp_min",
        "humidite",
        "precipitations",
        "anomalie_temp",
        "activity_total",
        "besoin_prevu" if "besoin_prevu" in week_df.columns else "activity_total",
    ]
    export_cols = [col for col in export_cols if col in week_df.columns]
    download_button("⬇️ Exporter la météo", week_df[export_cols], f"meteo_{week_df['semaine'].iloc[0]}.csv")
