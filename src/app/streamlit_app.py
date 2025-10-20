from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "manual"
GEOJSON_PATH = DATA_DIR / "departements.geojson"


@dataclass(frozen=True)
class PredictionPayload:
    features: pd.DataFrame
    predictions: pd.Series
    model: LinearRegression


@st.cache_data(show_spinner=False)
def load_geojson() -> Dict:
    with GEOJSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(filename: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / filename, dtype={"departement": str})
    df["departement"] = df["departement"].str.zfill(2)
    return df


@st.cache_data(show_spinner=False)
def load_datasets() -> Dict[str, pd.DataFrame]:
    return {
        "coverage": _read_csv("coverage.csv"),
        "distribution": _read_csv("distribution.csv"),
        "ias": _read_csv("ias.csv"),
        "urgences": _read_csv("urgences.csv"),
        "vaccination_trends": _read_csv("vaccination_trends.csv"),
    }


def _split_week(week_str: str) -> Tuple[int, int]:
    year, week = week_str.split("-")
    return int(year), int(week)


@st.cache_data(show_spinner=False)
def build_training_frame() -> pd.DataFrame:
    data = load_datasets()
    vacc = data["vaccination_trends"].copy()
    ias = data["ias"].copy()
    urgences = data["urgences"].copy()

    vacc[["year", "week_num"]] = vacc["semaine"].apply(lambda x: pd.Series(_split_week(x)))
    vacc.sort_values(["departement", "year", "week_num"], inplace=True)
    vacc["prev_coverage"] = vacc.groupby("departement")["couverture_vaccinale_percent"].shift(1)
    vacc["trend"] = vacc["couverture_vaccinale_percent"] - vacc["prev_coverage"]

    merged = vacc.merge(
        ias[["departement", "semaine", "ias_signal"]],
        on=["departement", "semaine"],
        how="left",
    ).merge(
        urgences[["departement", "semaine", "urgences_grippe", "sos_medecins"]],
        on=["departement", "semaine"],
        how="left",
    )

    merged["activity_total"] = merged["urgences_grippe"].fillna(0) + merged["sos_medecins"].fillna(0)
    merged["trend"] = merged["trend"].fillna(0.0)

    merged["besoin_reel"] = (
        2200
        + (100 - merged["couverture_vaccinale_percent"]) * 38
        + merged["ias_signal"].fillna(0) * 24
        + merged["activity_total"] * 6.5
        - merged["trend"] * 115
    ).clip(lower=500)

    merged.dropna(subset=["ias_signal", "prev_coverage"], inplace=True)
    return merged


@st.cache_resource(show_spinner=False)
def train_predictor() -> PredictionPayload:
    training_df = build_training_frame()
    feature_cols = [
        "couverture_vaccinale_percent",
        "trend",
        "ias_signal",
        "activity_total",
    ]
    model = LinearRegression()
    model.fit(training_df[feature_cols], training_df["besoin_reel"])
    training_df["prediction"] = model.predict(training_df[feature_cols])

    return PredictionPayload(
        features=training_df,
        predictions=training_df["prediction"],
        model=model,
    )


def compute_latest_metrics() -> pd.DataFrame:
    data = load_datasets()
    distribution = data["distribution"].copy()
    ias = data["ias"].copy()
    urgences = data["urgences"].copy()
    trends = data["vaccination_trends"].copy()

    distribution[["year", "week_num"]] = distribution["semaine"].apply(lambda x: pd.Series(_split_week(x)))
    latest_year, latest_week = distribution.sort_values(["year", "week_num"]).iloc[-1][["year", "week_num"]]
    latest_week_str = f"{latest_year}-{latest_week:02d}"

    latest_dist = distribution.loc[
        (distribution["year"] == latest_year) & (distribution["week_num"] == latest_week)
    ]
    latest_ias = ias[ias["semaine"] == latest_week_str]
    latest_urg = urgences[urgences["semaine"] == latest_week_str]

    latest_trend = trends[trends["semaine"] == latest_week_str].copy()
    prev_trend = trends[trends["semaine"] < latest_week_str].sort_values(["departement", "semaine"])
    prev_values = prev_trend.groupby("departement").tail(1)[["departement", "couverture_vaccinale_percent"]]
    prev_values = prev_values.rename(columns={"couverture_vaccinale_percent": "coverage_prev"})

    latest_trend = latest_trend.merge(prev_values, on="departement", how="left")
    latest_trend["trend"] = latest_trend["couverture_vaccinale_percent"] - latest_trend["coverage_prev"]

    predictor = train_predictor()
    feature_cols = [
        "couverture_vaccinale_percent",
        "trend",
        "ias_signal",
        "activity_total",
    ]

    latest_features = latest_trend.merge(
        latest_ias[["departement", "ias_signal"]],
        on="departement",
        how="left",
    ).merge(
        latest_urg[["departement", "urgences_grippe", "sos_medecins"]],
        on="departement",
        how="left",
    )
    latest_features["activity_total"] = (
        latest_features["urgences_grippe"].fillna(0) + latest_features["sos_medecins"].fillna(0)
    )
    latest_features["trend"] = latest_features["trend"].fillna(0.0)

    predicted_needs = predictor.model.predict(latest_features[feature_cols].fillna(0))
    latest_features["besoin_prevu"] = np.round(predicted_needs, 0)

    latest_combined = latest_features.merge(
        latest_dist[["departement", "doses_distribuees", "actes_pharmacie"]],
        on="departement",
        how="left",
    )

    latest_combined["semaine"] = latest_week_str
    return latest_combined


def build_map(df: pd.DataFrame, geojson: Dict, view: str) -> px.choropleth_mapbox:
    color_column = {
        "Vaccination": "couverture_vaccinale_percent",
        "Distribution": "besoin_prevu",
        "Urgences": "activity_total",
    }[view]

    color_title = {
        "Vaccination": "Taux de couverture (%)",
        "Distribution": "Besoin vaccinal prévu",
        "Urgences": "Activité grippe (urgences + SOS)",
    }[view]

    hover_data = {
        "departement": True,
        "nom": True,
        "couverture_vaccinale_percent": ":.1f",
        "besoin_prevu": ":,.0f",
        "doses_distribuees": ":,",
        "actes_pharmacie": ":,",
        "urgences_grippe": ":,",
        "sos_medecins": ":,",
        "activity_total": ":,",
    }

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        locations="departement",
        featureidkey="properties.code",
        color=color_column,
        color_continuous_scale="YlGnBu" if view == "Vaccination" else "OrRd",
        hover_name="nom",
        hover_data=hover_data,
        mapbox_style="carto-positron",
        center={"lat": 46.6, "lon": 1.9},
        zoom=4.7,
        opacity=0.75,
        labels={color_column: color_title},
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def kpi_cards(df: pd.DataFrame) -> None:
    mean_coverage = df["couverture_vaccinale_percent"].mean()
    under_vaccinated = df[df["couverture_vaccinale_percent"] < 45]
    total_predicted = df["besoin_prevu"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Taux moyen de couverture", f"{mean_coverage:.1f} %")
    col2.metric("Départements sous-vaccinés", f"{len(under_vaccinated)}")
    col3.metric("Total doses prévues", f"{int(total_predicted):,}".replace(",", " "))

    if len(under_vaccinated) > 0:
        st.caption(
            "Sous-vaccinés: "
            + ", ".join(sorted(set(under_vaccinated["nom"].tolist()), key=lambda x: x))
        )


def render_graphs(df: pd.DataFrame) -> None:
    hosp_fig = px.bar(
        df.sort_values("activity_total", ascending=False).head(15),
        x="nom",
        y="activity_total",
        labels={"nom": "Département", "activity_total": "Urgences + SOS (hebdo)"},
        title="Activité grippe (Top 15 départements)",
        color="activity_total",
        color_continuous_scale="Reds",
    )
    hosp_fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False, height=420)

    coverage_fig = px.line(
        build_training_frame()
        .groupby(["semaine"])
        .agg(couverture_moyenne=("couverture_vaccinale_percent", "mean"))
        .reset_index(),
        x="semaine",
        y="couverture_moyenne",
        markers=True,
        labels={"semaine": "Semaine", "couverture_moyenne": "Couverture moyenne (%)"},
        title="Tendance hebdomadaire de la couverture vaccinale",
    )
    coverage_fig.update_layout(height=320)

    st.plotly_chart(hosp_fig, use_container_width=True)
    st.plotly_chart(coverage_fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Surveillance vaccinale grippe - POC",
        layout="wide",
        page_icon="💉",
    )

    st.title("POC - Vaccination grippe en France")
    st.caption(
        "Prototype interactif : couverture vaccinale, besoins prévisionnels et activité sanitaire."
    )

    data = compute_latest_metrics()
    data["activity_total"] = data["activity_total"].fillna(
        data["urgences_grippe"].fillna(0) + data["sos_medecins"].fillna(0)
    )
    geojson = load_geojson()

    with st.sidebar:
        st.header("Paramètres")
        view = st.radio(
            "Vue à afficher",
            ["Vaccination", "Distribution", "Urgences"],
            index=0,
        )
        st.write(
            "Infobulle: % du vaccin, besoins prévus, activité SOS/urgences et distribution pharmaceutique."
        )

    kpi_cards(data)
    st.plotly_chart(build_map(data, geojson, view), use_container_width=True)

    st.subheader("Analyses complémentaires")
    render_graphs(data)

    st.markdown(
        """
        ### Observations
        - Les départements à faible couverture vaccinale ressortent en priorité pour le renforcement des campagnes.
        - Le modèle prédictif anticipe les besoins en croisant tendances vaccinales et signal IAS.
        - Les pics d'activité SOS Médecins/urgences guident la priorisation logistique.
        """
    )


if __name__ == "__main__":
    main()
