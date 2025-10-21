from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.model import PredictionResult
from .components import choropleth_map


def render_map(prediction: PredictionResult, geojson: dict) -> None:
    st.markdown("### Couverture vs Besoins")
    df = prediction.per_department.copy()
    if df.empty:
        st.warning("Données indisponibles pour cette période.")
        return

    layer_choice = st.radio(
        "Couche affichée",
        ["Couverture vaccinale (%)", "Besoin prédit (doses)"],
        horizontal=True,
    )

    metric = "couverture_mois" if layer_choice.startswith("Couverture") else "besoin_prevu"
    title = "Couverture vaccinale (%)" if metric == "couverture_mois" else "Besoin prédit (doses)"

    fig = choropleth_map(df, geojson, metric, title, palette="YlGnBu" if metric == "couverture_mois" else "YlOrRd")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "- **Comment lire :** vert ≥ 75 %, orange 60–75 %, rouge < 60 %.\n"
        "- **Hypothèse :** besoin basé sur couverture cible et signaux IAS.\n"
        "- **Action :** cibler les départements rouges avant la prochaine campagne."
    )


def render_forecast_view(prediction: PredictionResult) -> None:
    st.markdown("### Prévisions vaccins & flux")
    timeline = prediction.timeline.copy()
    if timeline.empty:
        st.warning("Pas de données historiques pour afficher la tendance.")
        return

    dept_options = sorted(timeline["departement"].unique())
    selected_dept = st.selectbox("Département", dept_options)
    dept_history = timeline[timeline["departement"] == selected_dept].copy()
    dept_history = dept_history.sort_values("month_dt")
    dept_history = dept_history.tail(12)

    if dept_history.empty:
        st.info("Historique insuffisant pour ce département.")
        return

    last_row = dept_history.iloc[-1]
    history_len = len(dept_history.dropna(subset=["couverture_mois"]))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Besoin prédit", f"{int(last_row['besoin_prevu']):,}".replace(",", " "))
    with col2:
        st.metric("Couverture actuelle", f"{last_row['couverture_mois']:.1f} %")
    with col3:
        st.metric("Confiance", last_row.get("confidence", "-"))
        if history_len < 6:
            st.caption("Historique court : prévision à interpréter avec prudence")

    dept_history["Besoin prédit"] = dept_history["besoin_prevu"]
    dept_history["Couverture (%)"] = dept_history["couverture_mois"]
    if "flux_proj" not in dept_history.columns:
        dept_history["flux_proj"] = dept_history.get("flux_mois", 0)
    months = dept_history["mois_label"]

    combo_fig = go.Figure()
    combo_fig.add_trace(
        go.Scatter(
            x=dept_history["mois_label"],
            y=dept_history["Besoin prédit"],
            name="Besoin vaccinal (doses)",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=3),
        )
    )
    combo_fig.add_trace(
        go.Scatter(
            x=dept_history["mois_label"],
            y=dept_history["flux_proj"],
            name="Flux patients (urgences + SOS)",
            mode="lines+markers",
            line=dict(color="#d62728", width=2, dash="dash"),
            yaxis="y2",
        )
    )
    combo_fig.update_layout(
        height=360,
        xaxis=dict(title="Mois"),
        yaxis=dict(title="Besoins (doses)"),
        yaxis2=dict(
            title="Flux patients",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=60, b=40),
        title="Besoin vaccinal vs flux patients – 12 mois",
    )
    st.plotly_chart(combo_fig, use_container_width=True, config={"displayModeBar": False})

    national = prediction.national.copy()
    national = national.sort_values("mois_label").tail(12)
    national_fig = px.line(
        national,
        x="mois_label",
        y="besoin_total",
        markers=True,
        title="Besoin national (12 mois)",
        labels={"mois_label": "Mois", "besoin_total": "Doses"},
    )
    national_fig.update_layout(height=280, hovermode="x unified")
    st.plotly_chart(national_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "- **Comment lire :** la courbe bleue représente le besoin estimé en doses, la courbe rouge les flux patients.\n"
        "- **Hypothèses :** moyenne mobile 3 mois, IAS k = {:.2f}, uplift hiver {} %.\n"
        "- **Action :** ajuster campagnes et ressources avant les pics hivernaux.".format(
            prediction.ias_coef,
            prediction.season_uplift_pct,
        )
    )
