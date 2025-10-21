from __future__ import annotations

from datetime import date
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
    work["label"] = work["nom"] + work["weeks_count"].fillna(0).apply(lambda x: " (prévision)" if x == 0 else "")

    cols = st.columns(3)

    top_priority = work.sort_values(["priority", "besoin_prevu"], ascending=[False, False]).head(3)
    with cols[0]:
        st.markdown("**Priorité sanitaire**")
        if top_priority.empty:
            st.write("Situation maîtrisée.")
        else:
            for _, row in top_priority.iterrows():
                st.markdown(
                    f"- **{row['label']}** — besoin {int(row['besoin_prevu']):,} doses, couverture {row['couverture_mois']:.1f}%"
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
                st.markdown(f"- **{row['label']}** — déficit estimé {abs(int(row['logistic_gap'])):,} doses")

    under = work[work["couverture_mois"] < under_threshold_pct].sort_values("coverage_gap", ascending=False).head(3)
    with cols[2]:
        st.markdown("**Renforcer la campagne**")
        if under.empty:
            st.write("Couverture alignée sur le seuil cible.")
        else:
            for _, row in under.iterrows():
                gap = max(row["coverage_gap"], 0)
                st.markdown(
                    f"- **{row['label']}** — cible {coverage_target_pct:.0f} %, manque {gap:.1f} pts"
                )


def render_global_kpis(df: pd.DataFrame, coverage_target_pct: float) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Doses estimées", f"{int(df['besoin_prevu'].sum()):,}".replace(",", " "))
    col2.metric("Couverture moyenne", f"{df['couverture_mois'].mean():.1f} %")
    high_risk = (df["risk_level"] == "rouge").sum()
    col3.metric("Départements à risque", str(int(high_risk)))
    col4.metric("Flux patients", f"{int(df['flux_proj'].sum()):,}".replace(",", " "))


def month_to_dates(month_str: str) -> Tuple[str, str]:
    dt = pd.to_datetime(f"{month_str}-01")
    start = dt.date()
    end = (dt + pd.offsets.MonthEnd(0)).date()
    return start.strftime("%d %b %Y"), end.strftime("%d %b %Y")


def render_parameter_legend(coverage_target_pct: float, season_uplift_pct: float, ias_coef: float) -> None:
    st.markdown(
        f"**Paramètres actifs** · Cible couverture : {coverage_target_pct:.0f} % · Uplift hiver : {season_uplift_pct:.0f} % · Coefficient IAS : {ias_coef:.2f}"
    )


def main() -> None:
    bundle = load_data_bundle()
    if not bundle.months:
        st.error("Aucune donnée disponible. Ajoutez des CSV dans data/manual/ pour démarrer.")
        st.stop()

    future_months = generate_future_months(bundle.months[-1], horizon=3)
    historical_months = bundle.months
    month_options = historical_months + future_months

    min_date = pd.to_datetime(historical_months[0] + "-01").date()
    max_date = pd.to_datetime(month_options[-1] + "-01").date()

    with st.sidebar:
        st.header("Paramètres")
        st.caption("📆 Choisissez un mois via le calendrier")
        default_date = pd.to_datetime(month_options[-1] + "-01").date()
        selected_date = st.date_input(
            "Mois",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
        )
        month = f"{selected_date.year}-{selected_date.month:02d}"
        coverage_target_pct = st.slider("Cible de couverture (%)", min_value=40, max_value=85, value=60, step=1)
        st.caption("Taux souhaité pour la population cible.")
        under_threshold_pct = st.slider("Seuil sous-vaccination (%)", min_value=30, max_value=70, value=45, step=1)
        st.caption("Départements < seuil affichés en priorité.")
        season_uplift_pct = st.slider("Uplift hiver (%)", min_value=0, max_value=30, value=20, step=5)
        st.caption("Amplification des besoins sur novembre-février.")
        ias_coef = st.slider("Coefficient IAS", min_value=0.0, max_value=0.3, value=0.15, step=0.05)
        st.caption("Sensibilité aux signaux IAS (0 = neutre).")
        if month in future_months:
            st.caption("Sélection : mois prévisionnel (données simulées).")
        else:
            st.caption("Sélection : mois historique (données observées).")
        st.caption(f"Historique disponible : {historical_months[0]} → {historical_months[-1]}")

    prediction_result = predict_needs(bundle, month, coverage_target_pct, season_uplift_pct, ias_coef)
    per_dept = prediction_result.per_department.copy()

    render_global_kpis(per_dept, coverage_target_pct)
    render_parameter_legend(coverage_target_pct, season_uplift_pct, ias_coef)
    render_assistant_panel(per_dept, coverage_target_pct, under_threshold_pct)

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
