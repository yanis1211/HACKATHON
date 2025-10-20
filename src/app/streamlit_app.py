"""Streamlit application for the flu planning POC."""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_DIR, PROCESSED_DATA_DIR
from src.features.builder import build_training_dataset, DatasetPaths
from src.models.predictor import ForecastPredictor
from src.models.trainer import train
from src.services.allocation import compute_allocation
from src.services.alerts import compute_alerts
from geopandas import GeoDataFrame, read_file


st.set_page_config(page_title="Flu Planning POC", layout="wide")
st.title("📈 Flu Vaccination Planning – POC")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_DIR / "training_dataset.csv", parse_dates=["week_start"])


@st.cache_resource
def load_predictor() -> ForecastPredictor:
    return ForecastPredictor()


def load_geo_data(filename: str) -> Optional[GeoDataFrame]:
    path = DATA_DIR / "manual" / filename
    if not path.exists():
        return None
    gdf = read_file(path)
    candidate_cols = [
        "code",
        "code_insee",
        "code_insee_region",
        "code_insee_reg",
        "code_reg",
        "code_dep",
        "num_dep",
        "id",
    ]
    for col in candidate_cols:
        if col in gdf.columns:
            gdf["code"] = gdf[col]
            break
    if "code" not in gdf.columns:
        return None
    gdf["code"] = gdf["code"].astype(str).str.zfill(2)
    return gdf


def refresh_pipeline() -> None:
    with st.spinner("Refreshing data, features, and model…"):
        from src.etl.fetch_sample import main as fetch_sample_main

        fetch_sample_main()
        build_training_dataset(DatasetPaths())
        train()
        load_dataset.clear()
        load_predictor.clear()


with st.sidebar:
    st.header("Actions")
    if st.button("🔄 Rafraîchir les données"):
        refresh_pipeline()
        st.success("Pipeline recalculé !")
        st.experimental_rerun()

    horizon = st.slider("Horizon de prévision (semaines)", 1, 6, 4)
    total_stock = st.number_input("Stock global (doses)", min_value=0, value=5000, step=500)


dataset = load_dataset()
predictor = load_predictor()

st.subheader("Prévisions de passages grippe (1–6 semaines)")
forecasts = predictor.forecast(horizon=horizon)
st.dataframe(forecasts)

st.subheader("Alertes actuelles (badge semaine à risque)")
alerts = compute_alerts(dataset)
alerts_df = pd.DataFrame(alerts)
if alerts_df.empty:
    st.info("Pas d'alertes disponibles – lancer un rafraîchissement.")
else:
    st.dataframe(alerts_df)

st.subheader("Allocation anti-rupture (heuristique)")
coverage_latest = (
    dataset.sort_values("week_start")
    .groupby("dep_code")
    .tail(1)[["dep_code", "coverage_rate"]]
)
allocation = compute_allocation(
    forecasts=forecasts,
    coverage=coverage_latest,
    total_stock=float(total_stock),
)
st.dataframe(pd.DataFrame(allocation))

st.subheader("Visualisation")
selected_dep = st.selectbox("Département", sorted(dataset["dep_code"].unique()))
hist = dataset[dataset["dep_code"] == selected_dep].copy()
hist = hist.sort_values("week_start").tail(52)
hist["type"] = "Historique"
forecast_dep = forecasts[forecasts["dep_code"] == selected_dep].copy()
forecast_dep["type"] = "Prévision"
forecast_dep = forecast_dep.rename(columns={"prediction": "y"})
merged = pd.concat(
    [
        hist[["week_start", "y", "type"]],
        forecast_dep[["week_start", "y", "type"]],
    ],
    ignore_index=True,
)
merged = merged.sort_values("week_start")
st.line_chart(
    merged,
    x="week_start",
    y="y",
    color="type",
)

st.caption(
    "Pipeline : ETL APIs → features (lags/saisonnalité) → GradientBoostingRegressor → API FastAPI → Dashboard Streamlit."
)

st.subheader("Carte France – Prévisions hebdomadaires")
choropleth_df = forecasts.groupby("dep_code", as_index=False)["prediction"].mean()
choropleth_df["dep_code"] = choropleth_df["dep_code"].astype(str).str.zfill(2)
nb_codes = choropleth_df["dep_code"].nunique()

if nb_codes > 20:
    geo_gdf = load_geo_data("departements.geojson")
    map_label = "Prévision moyenne (passages grippe) – départements"
else:
    geo_gdf = load_geo_data("regions.geojson")
    map_label = "Prévision moyenne (passages grippe) – régions"

if geo_gdf is None:
    st.info(
        "Ajoute un GeoJSON utilisable (e.g. `departements.geojson` ou `regions.geojson`) "
        "dans `data/manual/` pour afficher la carte choroplèthe."
    )
else:
    merged = geo_gdf.merge(
        choropleth_df, left_on="code", right_on="dep_code", how="left"
    )
    geojson = json.loads(merged.to_json())
    fig = px.choropleth(
        merged,
        geojson=geojson,
        locations="code",
        featureidkey="properties.code",
        color="prediction",
        color_continuous_scale="Reds",
        labels={"prediction": "Prévision"},
        title=map_label,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
