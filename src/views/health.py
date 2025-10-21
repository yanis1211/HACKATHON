from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .components import choropleth_map, download_button


def render(
    week_df: pd.DataFrame,
    geojson: dict,
    activity_threshold: float,
) -> None:
    st.subheader("Vue Urgences & IAS – surveillance sanitaire")
    if week_df.empty:
        st.warning("Pas de données disponibles pour cette semaine.")
        return

    st.plotly_chart(
        choropleth_map(week_df, geojson, "activity_total", "Urgences + SOS (hebdo)", palette="OrRd"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    col1, col2 = st.columns(2)
    with col1:
        top_activity = week_df.sort_values("activity_total", ascending=False).head(10)[
            ["departement", "nom", "activity_total", "urgences_grippe", "sos_medecins"]
        ]
        st.markdown("**Hotspots activité grippe**")
        st.dataframe(top_activity, use_container_width=True, hide_index=True)
    with col2:
        top_ias = week_df.sort_values("ias_signal", ascending=False).head(10)[["departement", "nom", "ias_signal"]]
        st.markdown("**Signal IAS – Top 10**")
        st.dataframe(top_ias, use_container_width=True, hide_index=True)

    scatter_cols = ["ias_signal", "activity_total"]
    if all(col in week_df.columns for col in scatter_cols):
        scatter = px.scatter(
            week_df,
            x="ias_signal",
            y="activity_total",
            color="risk_level",
            hover_name="nom",
            labels={"ias_signal": "IAS", "activity_total": "Urgences + SOS"},
            title="Corrélation IAS / Activité grippe",
        )
        st.plotly_chart(scatter, use_container_width=True, config={"displayModeBar": False})

    alerts = week_df[week_df["activity_total"] >= activity_threshold][
        ["departement", "nom", "activity_total", "ias_signal", "risk_level"]
    ].sort_values("activity_total", ascending=False)

    st.markdown(f"#### Liste d'alerte (activité ≥ {int(activity_threshold)})")
    if alerts.empty:
        st.success("Pas de département au-dessus du seuil configuré.")
    else:
        st.dataframe(alerts, use_container_width=True, hide_index=True)
        download_button(
            "⬇️ Exporter la liste d'alerte",
            alerts,
            f"alertes_{week_df['semaine'].iloc[0]}.csv",
        )
