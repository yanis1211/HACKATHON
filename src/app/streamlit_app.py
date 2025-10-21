from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

try:
    from suggestions import annotate_with_suggestions
except ImportError:  # pragma: no cover
    from .suggestions import annotate_with_suggestions  # type: ignore


APP_DIR = Path(__file__).resolve().parent
CANDIDATES = [
    APP_DIR / "data" / "manual",
    APP_DIR.parent / "data" / "manual",
    APP_DIR.parents[1] / "data" / "manual",
    Path.cwd() / "data" / "manual",
]

LOCAL_DATA = next((p.resolve() for p in CANDIDATES if p.exists()), (APP_DIR.parents[1] / "data" / "manual").resolve())
FALLBACK_DATA = Path("/mnt/data/poc_aligned")
REQUIRED_FILES = {
    "geojson": "departements.geojson",
    "vaccination_trends": "vaccination_trends.csv",
    "ias": "ias.csv",
    "urgences": "urgences.csv",
    "distribution": "distribution.csv",
    "coverage": "coverage.csv",
}

PALETTES = {
    "Surveillance": "YlGnBu",
    "Besoins prévus": "YlOrRd",
    "Urgences": "YlOrRd",
    "Ciblage": "Rocket",
}


st.set_page_config(
    page_title="Carte prédictive vaccination grippe",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_help_sidebar() -> None:
    with st.sidebar.expander("ℹ️ Aide rapide", expanded=False):
        st.markdown(
            """
**Contrôles**
• Semaine analysée  
• Cible de couverture (%)  
• Seuil sous-vaccination (%)  
• Palette de couleurs

**Légendes**
- Surveillance : plus foncé = meilleure couverture.
- Besoins prévus : rouge = priorité logistique.
- Urgences : rouge = forte pression sanitaire.
- Ciblage : rouge = zone à suivre.
            """
        )


@dataclass(frozen=True)
class PredBundle:
    model: LinearRegression
    features: List[str]
    r2: float


def _mtime(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    return path.stat().st_mtime


def _split_week(week: str) -> Tuple[int, int]:
    w = str(week).upper().replace("W", "")
    year, week_num = w.split("-")
    return int(year), int(week_num)


def _zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std(ddof=0) + 1e-6)


def fmt_int(value: float | int) -> str:
    try:
        return f"{int(round(float(value))):,}".replace(",", " ")
    except Exception:
        return "-"


def _read_csv_generic(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, low_memory=False)


def normalize_dept(df: pd.DataFrame, column: str = "departement") -> pd.DataFrame:
    if column not in df.columns:
        return df
    df[column] = df[column].astype(str).str.strip().str.upper()
    mask = df[column].str.contains(r"[A-Z]$")
    df.loc[~mask, column] = df.loc[~mask, column].str.zfill(2)
    return df


def normalize_week(df: pd.DataFrame, column: str = "semaine") -> pd.DataFrame:
    if column not in df.columns:
        return df
    df[column] = df[column].astype(str).str.upper().str.replace("W", "", regex=False)
    return df


def find_file(name: str) -> Path | None:
    candidate = LOCAL_DATA / name
    if candidate.exists():
        return candidate
    fallback = FALLBACK_DATA / name
    if fallback.exists():
        return fallback
    return None


def load_geojson_file() -> dict | None:
    geo_path = find_file(REQUIRED_FILES["geojson"])
    if geo_path:
        with geo_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def data_signature() -> Tuple[float, ...]:
    return tuple(_mtime(find_file(fname)) for fname in REQUIRED_FILES.values())


def load_all_csvs(sig: Tuple[float, ...]) -> Dict[str, pd.DataFrame]:
    del sig
    datasets: Dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_FILES.items():
        if key == "geojson":
            continue
        source = find_file(filename)
        if source is None:
            datasets[key] = pd.DataFrame()
            continue
        frame = _read_csv_generic(source)
        rename_map: Dict[str, str] = {}
        for column in frame.columns:
            low = column.lower()
            if low in {"departement code", "département code", "code departement", "code_departement", "dep_code", "code"}:
                rename_map[column] = "departement"
            elif low in {"departement", "département", "depnom", "nom departement"} and "nom" not in frame.columns:
                rename_map[column] = "nom"
            elif key in {"vaccination_trends", "coverage"} and low.startswith("couverture"):
                rename_map[column] = "couverture_vaccinale_percent"
            elif key in {"vaccination_trends", "coverage"} and low in {"semaine", "week", "periode"}:
                rename_map[column] = "semaine"
            elif key == "ias":
                if "ias" in low and "ias_signal" not in frame.columns:
                    rename_map[column] = "ias_signal"
                elif low in {"semaine", "week", "periode"}:
                    rename_map[column] = "semaine"
            elif key == "urgences":
                if "urgences" in low and "urgences_grippe" not in frame.columns:
                    rename_map[column] = "urgences_grippe"
                elif "sos" in low and "sos_medecins" not in frame.columns:
                    rename_map[column] = "sos_medecins"
                elif low in {"semaine", "week", "periode"}:
                    rename_map[column] = "semaine"
            elif key == "distribution":
                if "dose" in low and "doses_distribuees" not in frame.columns:
                    rename_map[column] = "doses_distribuees"
                elif "acte" in low and "pharm" in low and "actes_pharmacie" not in frame.columns:
                    rename_map[column] = "actes_pharmacie"
                elif low in {"semaine", "week", "periode"}:
                    rename_map[column] = "semaine"
        if rename_map:
            frame = frame.rename(columns=rename_map)
        frame = normalize_dept(frame, "departement")
        frame = normalize_week(frame, "semaine")
        datasets[key] = frame
    return datasets


@st.cache_data(show_spinner=False)
def cached_load_all_csvs(sig: Tuple[float, ...]) -> Dict[str, pd.DataFrame]:
    return load_all_csvs(sig)


def build_training_frame(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    vacc = data["vaccination_trends"].copy()
    ias = data["ias"].copy()
    urg = data["urgences"].copy()
    if vacc.empty or not {"departement", "semaine", "couverture_vaccinale_percent"}.issubset(vacc.columns):
        return pd.DataFrame()
    vacc[["year", "week_num"]] = vacc["semaine"].apply(lambda s: pd.Series(_split_week(s)))
    vacc = vacc.sort_values(["departement", "year", "week_num"])
    vacc["prev_cov"] = vacc.groupby("departement")["couverture_vaccinale_percent"].shift(1)
    vacc["trend_cov"] = (vacc["couverture_vaccinale_percent"] - vacc["prev_cov"]).fillna(0.0)
    merged = (
        vacc.merge(ias[["departement", "semaine", "ias_signal"]], on=["departement", "semaine"], how="left")
        .merge(
            urg[["departement", "semaine", "urgences_grippe", "sos_medecins"]],
            on=["departement", "semaine"],
            how="left",
        )
    )
    merged["activity_total"] = merged["urgences_grippe"].fillna(0) + merged["sos_medecins"].fillna(0)
    merged["target_proxy"] = (
        2200
        + (100 - merged["couverture_vaccinale_percent"]) * 38
        + merged["ias_signal"].fillna(0) * 24
        + merged["activity_total"] * 6.5
        - merged["trend_cov"] * 115
    ).clip(lower=500)
    return merged.dropna(subset=["couverture_vaccinale_percent"])


def train_predictor(train_df: pd.DataFrame) -> PredBundle | None:
    if train_df.empty:
        return None
    feature_cols = ["couverture_vaccinale_percent", "trend_cov", "ias_signal", "activity_total"]
    X = train_df[feature_cols].fillna(0.0)
    y = train_df["target_proxy"]
    z_ias = _zscore(train_df["ias_signal"].fillna(train_df["ias_signal"].median() if train_df["ias_signal"].notna().any() else 0))
    z_urg = _zscore(train_df["activity_total"].fillna(0))
    coverage_gap = np.clip(65 - train_df["couverture_vaccinale_percent"], a_min=0, a_max=None)
    weights = (1 + 0.15 * z_ias.clip(-1, 3) + 0.1 * z_urg.clip(-1, 3) + 0.2 * (coverage_gap / 100.0)).clip(lower=0.2)
    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)
    r2 = r2_score(y, model.predict(X))
    return PredBundle(model=model, features=feature_cols, r2=r2)


def latest_week_from_any(data: Dict[str, pd.DataFrame]) -> str | None:
    weeks: List[str] = []
    for key in ("distribution", "vaccination_trends", "ias", "urgences"):
        if "semaine" in data[key].columns:
            weeks.extend([w for w in data[key]["semaine"].dropna().tolist() if isinstance(w, str)])
    if not weeks:
        return None
    weeks = list(set(weeks))
    return sorted(weeks, key=lambda w: _split_week(w))[-1]


def build_latest_frame(data: Dict[str, pd.DataFrame], selected_week: str | None) -> pd.DataFrame:
    week = selected_week or latest_week_from_any(data)
    if week is None:
        return pd.DataFrame()

    def select(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if df.empty or "semaine" not in df.columns:
            return pd.DataFrame(columns=cols)
        return df[df["semaine"] == week][cols]

    base = select(data["vaccination_trends"], ["departement", "nom", "semaine", "couverture_vaccinale_percent"])
    if base.empty:
        base = select(data["coverage"], ["departement", "nom", "couverture_vaccinale_percent"])
        base["semaine"] = week
    if not data["vaccination_trends"].empty:
        tmp = data["vaccination_trends"].copy()
        tmp[["year", "week_num"]] = tmp["semaine"].apply(lambda s: pd.Series(_split_week(s)))
        tmp = tmp.sort_values(["departement", "year", "week_num"])
        prev = (
            tmp[tmp["semaine"] < week]
            .groupby("departement")
            .tail(1)[["departement", "couverture_vaccinale_percent"]]
            .rename(columns={"couverture_vaccinale_percent": "prev_cov"})
        )
        base = base.merge(prev, on="departement", how="left")
        base["trend_cov"] = (base["couverture_vaccinale_percent"] - base["prev_cov"]).fillna(0.0)
    else:
        base["trend_cov"] = 0.0

    pieces = [
        select(data["ias"], ["departement", "ias_signal"]),
        select(data["urgences"], ["departement", "urgences_grippe", "sos_medecins"]),
        select(data["distribution"], ["departement", "doses_distribuees", "actes_pharmacie"]),
    ]
    for frame in pieces:
        base = base.merge(frame, on="departement", how="left")
    base["activity_total"] = base["urgences_grippe"].fillna(0) + base["sos_medecins"].fillna(0)
    return base


def predict_needs(latest_df: pd.DataFrame, bundle: PredBundle | None, coverage_target: float) -> pd.DataFrame:
    result = latest_df.copy()
    if result.empty:
        return result
    if bundle is None:
        result["besoin_prevu"] = np.nan
        result["risk_score"] = np.nan
        result["risk_level"] = "indéterminé"
        return result
    X = result[bundle.features].fillna(0.0)
    predictions = np.clip(bundle.model.predict(X), a_min=0, a_max=None)
    gap = np.clip(coverage_target * 100 - result["couverture_vaccinale_percent"], a_min=0, a_max=None)
    z_ias = _zscore(result["ias_signal"].fillna(result["ias_signal"].median() if result["ias_signal"].notna().any() else 0))
    z_urg = _zscore(result["activity_total"].fillna(0))
    adjust = (1 + 0.25 * (gap / 100.0) + 0.12 * z_ias.clip(-1, 2) + 0.08 * z_urg.clip(-1, 2)).clip(lower=0.6)
    result["besoin_prevu"] = (predictions * adjust).astype(float)
    risk = 0.45 * z_ias + 0.45 * z_urg + 0.1 * (-_zscore(result["trend_cov"].fillna(0)))
    result["risk_score"] = risk
    result["risk_level"] = pd.cut(
        risk,
        bins=[-1e9, -0.25, 0.75, 1e9],
        labels=["faible", "modéré", "élevé"],
    ).astype(str)
    return result


def propose_redistribution(df: pd.DataFrame, flex_pct: float = 0.05) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["balance"] = work["doses_distribuees"].fillna(0) - work["besoin_prevu"].fillna(0)
    donors = work[work["balance"] > flex_pct * (work["besoin_prevu"].fillna(0) + 1)].sort_values("balance", ascending=False)
    takers = work[work["balance"] < -flex_pct * (work["besoin_prevu"].fillna(0) + 1)].sort_values("balance")
    donors = donors.reset_index(drop=True)
    takers = takers.reset_index(drop=True)
    moves = []
    i = j = 0
    while i < len(donors) and j < len(takers):
        give = donors.loc[i, "balance"]
        need = -takers.loc[j, "balance"]
        quantity = min(give, need)
        if quantity <= 0:
            break
        moves.append(
            {
                "departement_source": donors.loc[i, "departement"],
                "nom_source": donors.loc[i, "nom"],
                "departement_cible": takers.loc[j, "departement"],
                "nom_cible": takers.loc[j, "nom"],
                "quantite_suggeree": int(quantity),
            }
        )
        donors.at[i, "balance"] -= quantity
        takers.at[j, "balance"] += quantity
        if donors.loc[i, "balance"] <= flex_pct * (donors.loc[i, "besoin_prevu"] + 1):
            i += 1
        if takers.loc[j, "balance"] >= -flex_pct * (takers.loc[j, "besoin_prevu"] + 1):
            j += 1
    return pd.DataFrame(moves)


def sidebar_controls(data: Dict[str, pd.DataFrame]) -> dict:
    st.sidebar.header("⚙️ Paramètres")
    weeks = []
    for key in ("distribution", "vaccination_trends", "ias", "urgences"):
        if "semaine" in data[key].columns:
            weeks.extend(data[key]["semaine"].dropna().unique().tolist())
    weeks = sorted(set(weeks), key=lambda w: _split_week(w) if "-" in str(w) else (0, 0))
    selected_week = st.sidebar.selectbox("Semaine", options=weeks or ["N/A"], index=len(weeks) - 1 if weeks else 0)
    st.sidebar.markdown("---")
    palette = st.sidebar.selectbox("Palette cartographique", ["YlGnBu", "YlOrRd", "Viridis", "Cividis", "Plasma"], index=0)
    st.sidebar.markdown("---")
    coverage_target = st.sidebar.slider("Cible de couverture (%)", 30, 80, 60, 1)
    under_threshold = st.sidebar.slider("Seuil sous-vaccination (%)", 30, 70, 45, 1)
    render_help_sidebar()
    st.sidebar.caption(f"Données : {LOCAL_DATA}")
    return {
        "week": selected_week,
        "palette": palette,
        "coverage_target": coverage_target / 100.0,
        "under_threshold": under_threshold,
    }


def kpi_header(df: pd.DataFrame, under_threshold: int, selected_week: str, r2_value: float | None) -> None:
    coverage_avg = df["couverture_vaccinale_percent"].mean()
    total_needs = df["besoin_prevu"].sum()
    activity_mean = df["activity_total"].mean()
    high_risk = (df["risk_level"] == "élevé").sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Couverture moyenne", f"{coverage_avg:.1f} %", help="Moyenne pondérée des taux de couverture sur la semaine sélectionnée.")
    c2.metric("Doses à prévoir", fmt_int(total_needs), help="Somme des besoins estimés pour la cible sélectionnée.")
    c3.metric("Urgences + SOS (moy.)", f"{activity_mean:.0f}", help="Activité sanitaire hebdomadaire moyenne (urgences + SOS Médecins).")
    c4.metric("Départements à risque", f"{int(high_risk)}", help="Nombre de départements classés en risque élevé.")
    under = df[df["couverture_vaccinale_percent"] < under_threshold]
    caption = f"Semaine analysée : {selected_week}"
    if r2_value is not None:
        caption += f" • R² modèle : {r2_value:.2f}"
    st.caption(caption + f" • Source données : {LOCAL_DATA}")
    if not under.empty:
        st.caption(
            "Sous-vaccination (<{} %) : {}".format(
                under_threshold,
                ", ".join(sorted(under["nom"].dropna().astype(str).unique())),
            )
        )


def render_decision_support(df: pd.DataFrame, coverage_target: float, under_threshold: int, selected_week: str) -> None:
    st.subheader("Assistant décisionnel VaxiScope")
    if df.empty:
        st.info("Les recommandations apparaîtront dès que les données hebdomadaires seront disponibles.")
        return

    work = df.copy()
    risk_weights = {"élevé": 2, "modéré": 1, "faible": 0}
    work["risk_weight"] = work["risk_level"].map(risk_weights).fillna(0)
    work["logistic_gap"] = work["besoin_prevu"].fillna(0) - work["doses_distribuees"].fillna(0)
    work["coverage_gap"] = coverage_target * 100 - work["couverture_vaccinale_percent"].fillna(0)

    top_priority = work.sort_values(["risk_weight", "risk_score"], ascending=[False, False]).head(3)
    top_logistics = work[work["logistic_gap"] > 250].sort_values("logistic_gap", ascending=False).head(3)
    top_campaign = work[work["coverage_gap"] > 0].sort_values("coverage_gap", ascending=False).head(3)

    cols = st.columns(3)

    def render_list(target_df: pd.DataFrame, container, empty_message: str, formatter) -> None:
        with container:
            if target_df.empty:
                st.write(empty_message)
            else:
                lines = [formatter(row) for _, row in target_df.iterrows()]
                st.markdown("\n".join(f"- {line}" for line in lines))

    render_list(
        top_priority,
        cols[0],
        "Situation maîtrisée.",
        lambda row: (
            f"**{row['nom']}** — {row['suggestion']} (risque {row['risk_level']}, couverture {row['couverture_vaccinale_percent']:.1f}%)"
        ),
    )
    cols[0].caption("Priorité : risque sanitaire")

    render_list(
        top_logistics,
        cols[1],
        "Distribution conforme aux besoins.",
        lambda row: (
            f"**{row['nom']}** — +{fmt_int(row['logistic_gap'])} doses à couvrir (distribution actuelle {fmt_int(row['doses_distribuees'])})"
        ),
    )
    cols[1].caption("Priorité : logistique & stocks")

    render_list(
        top_campaign,
        cols[2],
        "Couverture alignée sur la cible.",
        lambda row: (
            f"**{row['nom']}** — campagne à intensifier (gap {max(row['coverage_gap'],0):.1f} pts)"
        ),
    )
    cols[2].caption("Priorité : campagne vaccinale")

    st.caption(f"Semaine {selected_week} – recommandations générées automatiquement par le modèle.")


def build_map(df: pd.DataFrame, geojson: dict, metric: str, palette: str, title: str) -> px.choropleth_mapbox:
    scale = palette if palette else PALETTES.get(metric, "Viridis")
    display_df = df.copy()
    columns = [
        "nom",
        "couverture_vaccinale_percent",
        "besoin_prevu",
        "activity_total",
        "doses_distribuees",
        "actes_pharmacie",
        "risk_level",
        "suggestion",
    ]
    for col in columns:
        if col not in display_df.columns:
            display_df[col] = np.nan
    display_df["couverture_vaccinale_percent"] = display_df["couverture_vaccinale_percent"].fillna(0)
    display_df["besoin_prevu"] = display_df["besoin_prevu"].fillna(0)
    display_df["activity_total"] = display_df["activity_total"].fillna(0)
    display_df["doses_distribuees"] = display_df["doses_distribuees"].fillna(0)
    display_df["actes_pharmacie"] = display_df["actes_pharmacie"].fillna(0)
    display_df["risk_level"] = display_df["risk_level"].fillna("non défini")
    display_df["suggestion"] = display_df["suggestion"].fillna("Maintenir le suivi.")

    primary_line = {
        "couverture_vaccinale_percent": "Couverture : %{z:.1f} %",
        "besoin_prevu": "Besoins estimés : %{z:,.0f} doses",
        "activity_total": "Urgences + SOS : %{z:,.0f}",
        "risk_score": "Score de risque : %{z:.2f}",
    }.get(metric, f"{title} : %{{z}}")

    hovertemplate = (
        "<b>%{customdata[0]}</b> • %{location}<br>"
        f"{primary_line}<br>"
        "Couverture : %{customdata[1]:.1f} %<br>"
        "Besoins (doses) : %{customdata[2]:,.0f}<br>"
        "Doses distribuées : %{customdata[4]:,.0f}<br>"
        "Urgences + SOS : %{customdata[3]:,.0f}<br>"
        "Actes pharmacie : %{customdata[5]:,.0f}<br>"
        "Risque : %{customdata[6]}<br>"
        "Suggestion : %{customdata[7]}<extra></extra>"
    )

    fig = px.choropleth_mapbox(
        display_df,
        geojson=geojson,
        locations="departement",
        featureidkey="properties.code",
        color=metric,
        color_continuous_scale=scale,
        hover_name="nom",
        hover_data=None,
        custom_data=display_df[columns],
        mapbox_style="carto-positron",
        center={"lat": 46.6, "lon": 1.9},
        zoom=4.7,
        opacity=0.82,
        labels={metric: title},
    )
    fig.update_traces(hovertemplate=hovertemplate)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=title))
    return fig


def download_button(label: str, frame: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def render_prediction_tab(latest_df: pd.DataFrame, geojson: dict, palette: str, train_df: pd.DataFrame, bundle: PredBundle | None) -> None:
    st.subheader("Vision prédictive")
    if not latest_df.empty and geojson.get("features"):
        st.plotly_chart(
            build_map(latest_df, geojson, "besoin_prevu", palette, "Besoins vaccinaux (doses)"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    cols = st.columns(2)
    with cols[0]:
        if not train_df.empty:
            trend = (
                train_df.groupby("semaine")
                .agg(couverture_moyenne=("couverture_vaccinale_percent", "mean"))
                .reset_index()
            )
            fig = px.line(
                trend,
                x="semaine",
                y="couverture_moyenne",
                markers=True,
                title="Couverture vaccinale – moyenne hebdomadaire",
                labels={"semaine": "Semaine", "couverture_moyenne": "Couverture (%)"},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Historique de couverture indisponible pour tracer la tendance.")
    with cols[1]:
        corr_cols = ["couverture_vaccinale_percent", "ias_signal", "activity_total", "target_proxy"]
        corr = train_df[corr_cols].dropna() if not train_df.empty else pd.DataFrame()
        if len(corr) >= 5:
            st.markdown("**Corrélations (train)**")
            st.dataframe(corr.corr().round(2), use_container_width=True)
        else:
            st.info("Données insuffisantes pour une matrice de corrélation.")
    if not latest_df.empty:
        download_button("⬇️ Exporter les besoins estimés", latest_df, f"besoins_{latest_df['semaine'].iloc[0]}.csv")


def render_logistics_tab(latest_df: pd.DataFrame) -> None:
    st.subheader("Logistique vaccinale")
    if latest_df.empty:
        st.info("Données logistiques indisponibles.")
        return
    col1, col2 = st.columns(2)
    with col1:
        top_dist = latest_df.sort_values("doses_distribuees", ascending=False)[
            ["nom", "departement", "doses_distribuees", "actes_pharmacie"]
        ].head(15)
        st.markdown("**Volume distribué (Top 15)**")
        st.dataframe(top_dist, use_container_width=True, hide_index=True)
    with col2:
        plan = propose_redistribution(latest_df)
        st.markdown("**Redistribution suggérée (±5 %)**")
        if plan.empty:
            st.success("Ajustements non nécessaires selon le seuil choisi.")
        else:
            st.dataframe(plan, use_container_width=True, hide_index=True)
            download_button("⬇️ Exporter la redistribution", plan, f"redistribution_{latest_df['semaine'].iloc[0]}.csv")
    download_button("⬇️ Exporter la logistique", latest_df[
        ["departement", "nom", "semaine", "doses_distribuees", "actes_pharmacie", "besoin_prevu", "risk_level", "suggestion"]
    ], f"logistique_{latest_df['semaine'].iloc[0]}.csv")


def render_health_tab(latest_df: pd.DataFrame, geojson: dict, palette: str) -> None:
    st.subheader("Urgences & IAS")
    if not latest_df.empty and geojson.get("features"):
        st.plotly_chart(
            build_map(latest_df, geojson, "activity_total", palette, "Urgences + SOS (hebdo)"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    col1, col2 = st.columns(2)
    with col1:
        top_activity = latest_df.sort_values("activity_total", ascending=False)[
            ["nom", "departement", "activity_total", "urgences_grippe", "sos_medecins"]
        ].head(15)
        st.markdown("**Pression sanitaire (Top 15)**")
        st.dataframe(top_activity, use_container_width=True, hide_index=True)
    with col2:
        if "ias_signal" in latest_df.columns and latest_df["ias_signal"].notna().any():
            fig = px.histogram(
                latest_df,
                x="ias_signal",
                nbins=15,
                title="Distribution du signal IAS",
                labels={"ias_signal": "IAS"},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Signal IAS indisponible pour la semaine analysée.")
    download_button("⬇️ Exporter les indicateurs sanitaires", latest_df[
        ["departement", "nom", "semaine", "ias_signal", "urgences_grippe", "sos_medecins", "activity_total", "suggestion"]
    ], f"sante_{latest_df['semaine'].iloc[0]}.csv")


def render_targeting_tab(latest_df: pd.DataFrame, geojson: dict, palette: str, under_threshold: int) -> None:
    st.subheader("Ciblage territorial")
    if latest_df.empty:
        st.info("Données de ciblage indisponibles.")
        return
    targeting = latest_df.copy()
    targeting["sous_vaccination"] = targeting["couverture_vaccinale_percent"] < under_threshold
    targeting["priorite"] = np.select(
        [
            (targeting["sous_vaccination"]) & (targeting["risk_level"] == "élevé"),
            (targeting["sous_vaccination"]),
            (targeting["risk_level"] == "élevé"),
        ],
        ["critique", "surveiller", "surveiller"],
        default="stable",
    )
    if geojson.get("features"):
        st.plotly_chart(
            build_map(targeting, geojson, "risk_score", palette, "Score de risque"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    focus = targeting[
        targeting["priorite"].isin(["critique", "surveiller"])
    ][
        [
            "nom",
            "departement",
            "couverture_vaccinale_percent",
            "besoin_prevu",
            "risk_level",
            "risk_score",
            "sous_vaccination",
            "priorite",
            "suggestion",
        ]
    ].sort_values(["priorite", "risk_score"], ascending=[True, False])
    if focus.empty:
        st.success("Aucun département sous le seuil de couverture.")
    else:
        st.markdown("**Zones à surveiller**")
        st.dataframe(focus, use_container_width=True, hide_index=True)
        download_button("⬇️ Exporter le ciblage", focus, f"ciblage_{latest_df['semaine'].iloc[0]}.csv")


def main() -> None:
    st.title("Carte prédictive vaccination grippe")

    geojson = load_geojson_file()
    if geojson is None:
        st.warning("Le fichier departements.geojson est introuvable. Ajoute-le pour activer la carte.")
        uploaded = st.file_uploader("Importer un GeoJSON de départements", type=["geojson", "json"])
        if uploaded is not None:
            geojson = json.load(io.StringIO(uploaded.getvalue().decode("utf-8")))
        else:
            geojson = {"type": "FeatureCollection", "features": []}

    sig = data_signature()
    data = cached_load_all_csvs(sig)

    missing = [key for key in ("vaccination_trends", "ias", "urgences", "distribution") if data[key].empty]
    if missing:
        st.error("Fichiers manquants ou vides : " + ", ".join(missing))
        with st.expander("Importer des fichiers CSV"):
            for key in missing:
                file = st.file_uploader(f"Déposer {REQUIRED_FILES[key]}", type=["csv"], key=f"up_{key}")
                if file is not None:
                    df_upload = pd.read_csv(file)
                    df_upload = normalize_dept(df_upload, "departement")
                    df_upload = normalize_week(df_upload, "semaine")
                    data[key] = df_upload
        if any(data[k].empty for k in ("vaccination_trends", "ias", "urgences", "distribution")):
            st.stop()

    controls = sidebar_controls(data)
    train_df = build_training_frame(data)
    bundle = train_predictor(train_df)
    latest_df = build_latest_frame(data, controls["week"])
    latest_df = predict_needs(latest_df, bundle, controls["coverage_target"])
    latest_df = annotate_with_suggestions(
        latest_df,
        coverage_target=controls["coverage_target"],
        under_threshold=controls["under_threshold"],
    )

    kpi_header(latest_df, controls["under_threshold"], controls["week"], bundle.r2 if bundle else None)
    render_decision_support(latest_df, controls["coverage_target"], controls["under_threshold"], controls["week"])

    tabs = st.tabs(["🧠 Prédiction", "🚚 Logistique", "🏥 Urgences & IAS", "🎯 Ciblage territorial"])
    with tabs[0]:
        render_prediction_tab(latest_df, geojson, controls["palette"], train_df, bundle)
    with tabs[1]:
        render_logistics_tab(latest_df)
    with tabs[2]:
        render_health_tab(latest_df, geojson, controls["palette"])
    with tabs[3]:
        render_targeting_tab(latest_df, geojson, controls["palette"], controls["under_threshold"])


if __name__ == "__main__":
    main()
