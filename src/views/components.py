from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_PALETTES = {
    "needs": "YlOrRd",
    "coverage": "YlGnBu",
    "activity": "OrRd",
    "temperature": "RdBu_r",
}


def format_hover(df: pd.DataFrame, metric: str) -> str:
    primary_lookup = {
        "besoin_prevu": "Besoins estimés : %{z:,.0f} doses",
        "activity_total": "Urgences + SOS : %{z:,.0f}",
        "flux_proj": "Flux projeté : %{z:,.0f}",
        "couverture_vaccinale_percent": "Couverture : %{z:.1f} %",
        "couverture_mois": "Couverture : %{z:.1f} %",
        "temp_moy": "Température moyenne : %{z:.1f} °C",
    }
    primary_line = primary_lookup.get(metric, f"{metric} : %{{z}}")
    hover_lines = [
        "<b>%{customdata[0]}</b> — %{location}",
        primary_line,
        "Couverture : %{customdata[1]:.1f} %",
        "Besoins : %{customdata[2]:,.0f} doses",
        "Flux : %{customdata[3]:,.0f}",
        "Actes pharmacie : %{customdata[4]:,.0f}",
        "Risque : %{customdata[5]}",
    ]
    if "suggestion" in df.columns:
        hover_lines.append("Suggestion : %{customdata[6]}")
    hover_lines[-1] += "<extra></extra>"
    return "<br>".join(hover_lines)


def build_custom_data(df: pd.DataFrame) -> pd.DataFrame:
    coverage_col = "couverture_vaccinale_percent" if "couverture_vaccinale_percent" in df.columns else "couverture_mois"
    activity_col = "activity_total" if "activity_total" in df.columns else "flux_proj"
    actes_col = "actes_pharmacie" if "actes_pharmacie" in df.columns else "actes_mois"
    return df.assign(
        couverture_hover=df.get(coverage_col, pd.Series(dtype=float)).fillna(0.0),
        besoin_hover=df.get("besoin_prevu", pd.Series(dtype=float)).fillna(0.0),
        activity_hover=df.get(activity_col, pd.Series(dtype=float)).fillna(0.0),
        actes_hover=df.get(actes_col, pd.Series(dtype=float)).fillna(0.0),
        risk_hover=df.get("risk_level", pd.Series(dtype=object)).fillna("n/a"),
        suggestion_hover=df.get("suggestion", pd.Series(dtype=object)).fillna(""),
    )


def choropleth_map(
    df: pd.DataFrame,
    geojson: dict,
    metric: str,
    title: str,
    palette: Optional[str] = None,
) -> px.choropleth_mapbox:
    palette = palette or DEFAULT_PALETTES.get(metric, "Viridis")
    enriched = build_custom_data(df)
    hovertemplate = format_hover(df, metric)
    custom_cols = [
        "nom",
        "couverture_hover",
        "besoin_hover",
        "activity_hover",
        "actes_hover",
        "risk_hover",
    ]
    if "suggestion" in df.columns:
        custom_cols.append("suggestion_hover")

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        locations="departement",
        featureidkey="properties.code",
        color=metric,
        color_continuous_scale=palette,
        hover_name="nom",
        mapbox_style="carto-positron",
        center={"lat": 46.6, "lon": 1.9},
        zoom=4.7,
        opacity=0.8,
        labels={metric: title},
    )
    fig.update_traces(customdata=enriched[custom_cols].to_numpy(), hovertemplate=hovertemplate)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=title))
    return fig


def download_button(label: str, df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )
