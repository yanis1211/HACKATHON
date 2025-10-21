from __future__ import annotations

from datetime import date, timedelta
from typing import Tuple

import pandas as pd
import streamlit as st

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from core.data_loader import DataBundle, generate_future_months, load_data_bundle
from core.model import predict_needs
from views import logistics, prediction


st.set_page_config(
    page_title="VaxiScope – Assistant décisionnel grippe",
    page_icon="💉",
    layout="wide",
)


def render_assistant_panel(df: pd.DataFrame, coverage_target_pct: float, under_threshold_pct: float) -> None:
    st.subheader("Assistant décisionnel – recommandations clés")
    if df.empty:
        st.info("Les recommandations s’afficheront lorsque des données seront disponibles.")
        return

    work = df.copy()
    work["coverage_gap"] = coverage_target_pct - work["couverture_mois"].fillna(0)
    work["risk_flag"] = work["risk_level"].map({"rouge": 2, "orange": 1, "vert": 0}).fillna(0)
    work["priority"] = work["coverage_gap"] + work["risk_flag"] * 10

    cols = st.columns(3)

    top_priority = work.sort_values(["priority", "besoin_prevu"], ascending=[False, False]).head(3)
    with cols[0]:
        st.markdown("**Priorité sanitaire**")
        if top_priority.empty:
            st.write("Situation maîtrisée.")
        else:
            for _, row in top_priority.iterrows():
                st.markdown(
                    f"- **{row['nom']}** — besoin {int(row['besoin_prevu']):,} doses, couverture {row['couverture_mois']:.1f}%"
                )

    deficits = work.copy()
    if "doses_mois" in deficits.columns:
        deficits["logistic_gap"] = deficits["doses_mois"].fillna(0) - deficits["besoin_prevu"].fillna(0)
        top_deficits = deficits.sort_values("logistic_gap").head(3)
    else:
        top_deficits = pd.DataFrame()
    with cols[1]:
        st.markdown("**Rééquilibrer la logistique**")
        if top_deficits.empty:
            st.write("Distribution conforme aux besoins.")
        else:
            for _, row in top_deficits.iterrows():
                st.markdown(f"- **{row['nom']}** — déficit estimé {abs(int(row['logistic_gap'])):,} doses")

    under = work[work["couverture_mois"] < under_threshold_pct].sort_values("coverage_gap", ascending=False).head(3)
    with cols[2]:
        st.markdown("**Renforcer la campagne**")
        if under.empty:
            st.write("Couverture alignée sur le seuil cible.")
        else:
            for _, row in under.iterrows():
                gap = max(row["coverage_gap"], 0)
                st.markdown(
                    f"- **{row['nom']}** — cible {coverage_target_pct:.0f} %, manque {gap:.1f} pts"
                )


def render_global_kpis(df: pd.DataFrame, coverage_target_pct: float) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Doses estimées", f"{int(df['besoin_prevu'].sum()):,}".replace(",", " "))
    col2.metric("Couverture moyenne", f"{df['couverture_mois'].mean():.1f} %")
    high_risk = (df["risk_level"] == "rouge").sum()
    col3.metric("Départements à risque", str(int(high_risk)))
    col4.metric("Flux patients", f"{int(df['flux_proj'].sum()):,}".replace(",", " "))


def month_to_dates(month_str: str) -> Tuple[str, str]:
    year, month = map(int, month_str.split("-"))
    start = date(year, month, 1)
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    end = next_month - timedelta(days=1)
    return start.strftime("%d %b %Y"), end.strftime("%d %b %Y")


def render_month_calendar(historical: list[str], future: list[str], selected_month: str) -> None:
    calendar = []
    for month in historical:
        start, end = month_to_dates(month)
        calendar.append(
            {
                "Mois": month,
                "Du": start,
                "Au": end,
                "Type": "Historique",
                "Sélectionné": "✅" if month == selected_month else "",
            }
        )
    for month in future:
        start, end = month_to_dates(month)
        calendar.append(
            {
                "Mois": month,
                "Du": start,
                "Au": end,
                "Type": "Prévision",
                "Sélectionné": "✅" if month == selected_month else "",
            }
        )
    calendar_df = pd.DataFrame(calendar)
    st.markdown("#### Calendrier des mois (historique & prévisions)")
    st.dataframe(calendar_df, hide_index=True, use_container_width=True)


def main() -> None:
    bundle = load_data_bundle()
    if not bundle.months:
        st.error("Aucune donnée disponible. Ajoutez des CSV dans data/manual/ pour démarrer.")
        st.stop()

    future_months = generate_future_months(bundle.months[-1], horizon=3)
    historical_months = bundle.months
    month_options = historical_months + future_months
    labels = [month if month in historical_months else f"{month} (prévision)" for month in month_options]

    with st.sidebar:
        st.header("Paramètres")
        selected_label = st.selectbox("Mois", options=labels, index=len(labels) - 1)
        month = month_options[labels.index(selected_label)]
        coverage_target_pct = st.slider("Cible de couverture (%)", min_value=40, max_value=85, value=60, step=1)
        under_threshold_pct = st.slider("Seuil sous-vaccination (%)", min_value=30, max_value=70, value=45, step=1)
        season_uplift_pct = st.slider("Uplift hiver (%)", min_value=0, max_value=30, value=20, step=5)
        ias_coef = st.slider("Coefficient IAS", min_value=0.0, max_value=0.3, value=0.15, step=0.05)
        st.caption(f"Historique disponible : {historical_months[0]} → {historical_months[-1]}")

    prediction_result = predict_needs(bundle, month, coverage_target_pct, season_uplift_pct, ias_coef)
    per_dept = prediction_result.per_department.copy()

    render_global_kpis(per_dept, coverage_target_pct)
    render_assistant_panel(per_dept, coverage_target_pct, under_threshold_pct)
    render_month_calendar(historical_months, future_months, month)

    tabs = st.tabs(["🗺️ Carte", "📈 Prévisions", "🚚 Distribution", "ℹ️ Notes"])

    with tabs[0]:
        prediction.render_map(prediction_result, bundle.geojson)

    with tabs[1]:
        prediction.render_forecast_view(prediction_result)

    with tabs[2]:
        default_stock = int(per_dept["besoin_prevu"].sum()) if not per_dept.empty else 0
        logistics.render_distribution_view(prediction_result, default_stock)

    with tabs[3]:
        st.markdown("**Hypothèses principales**")
        st.markdown(
            "- Moyenne mobile 3 mois + correction IAS (k = {:.2f}).\n"
            "- Uplift saisonnier {}% appliqué à novembre, décembre, janvier et février.\n"
            "- Allocation proportionnelle aux besoins (plafonnement si couverture ≥ cible).".format(
                prediction_result.ias_coef,
                prediction_result.season_uplift_pct,
            )
        )


if __name__ == "__main__":
    main()
