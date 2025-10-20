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

# ============
# CONFIG / IO
# ============
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent          # .../src/app
# On cherche un dossier data/manual en priorité aux niveaux supérieurs
CANDIDATES = [
    APP_DIR / "data" / "manual",                   # .../src/app/data/manual
    APP_DIR.parent / "data" / "manual",            # .../src/data/manual
    APP_DIR.parents[1] / "data" / "manual",        # .../data/manual   <-- ton cas
    Path.cwd() / "data" / "manual",                # ./data/manual (si tu lances depuis la racine)
]

LOCAL_DATA = None
for p in CANDIDATES:
    if p.exists():
        LOCAL_DATA = p.resolve()
        break
if LOCAL_DATA is None:
    # par défaut, on pointe sur la racine/data/manual (même si absent, pour que l'UI propose l'upload)
    LOCAL_DATA = (APP_DIR.parents[1] / "data" / "manual").resolve()

FALLBACK_DATA = Path("/mnt/data/poc_aligned")  # optionnel (les CSV générés automatiquement)
REQUIRED_FILES = {
    "geojson": "departements.geojson",
    "vaccination_trends": "vaccination_trends.csv",
    "ias": "ias.csv",
    "urgences": "urgences.csv",
    "distribution": "distribution.csv",
    "coverage": "coverage.csv",  # optionnel
}

st.set_page_config(
    page_title="POC – Carte Prédictive Vaccination Grippe",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.caption(f"📂 Données: {LOCAL_DATA}")


# =============
# UTILITAIRES
# =============
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def _split_week(week_str: str) -> Tuple[int, int]:
    """Accepte 'YYYY-ww' ou 'YYYY-Www'."""
    w = str(week_str).strip().upper().replace("W", "")
    year, week = w.split("-")
    return int(year), int(week)


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-6)


def fmt_int(n: float | int) -> str:
    try:
        return f"{int(round(float(n))):,}".replace(",", " ")
    except Exception:
        return "-"


def _read_csv_generic(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, sep=";")
        except Exception:
            return pd.read_csv(p, low_memory=False)


def normalize_dept(df: pd.DataFrame, col="departement") -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[col] = df[col].astype(str).str.strip().str.upper()
    mask_corsica = df[col].str.contains(r"[A-Z]$")
    df.loc[~mask_corsica, col] = df.loc[~mask_corsica, col].str.zfill(2)
    return df


def normalize_week(df: pd.DataFrame, col="semaine") -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[col] = df[col].astype(str).str.upper().str.replace("W", "", regex=False)
    return df


def find_file(name: str) -> Path | None:
    """Recherche en priorité dans data/manual/, sinon dans /mnt/data/poc_aligned/."""
    p1 = LOCAL_DATA / name
    if p1.exists():
        return p1
    p2 = FALLBACK_DATA / name
    if p2.exists():
        return p2
    return None


# ==================
# CHARGEMENT DONNÉES
# ==================
def load_geojson_file() -> dict | None:
    geo_path = find_file(REQUIRED_FILES["geojson"])
    if geo_path:
        with geo_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


@st.cache_data(show_spinner=False)
def load_all_csvs(sig: Tuple[float, ...]) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for key, fname in REQUIRED_FILES.items():
        if key == "geojson":
            continue
        p = find_file(fname)
        if p is None:
            dfs[key] = pd.DataFrame()
            continue
        df = _read_csv_generic(p)
        # mapping minimal FR -> schéma POC
        if key in ("vaccination_trends", "coverage"):
            # attend: departement, nom, semaine?, couverture_vaccinale_percent
            ren = {}
            for c in df.columns:
                lc = c.lower()
                if lc in {"departement code", "département code", "code departement", "code_departement", "dep_code", "code"}:
                    ren[c] = "departement"
                elif lc in {"departement", "département", "depnom", "nom departement"} and "nom" not in df.columns:
                    ren[c] = "nom"
                elif lc.startswith("couverture") and "couverture_vaccinale_percent" not in df.columns:
                    ren[c] = "couverture_vaccinale_percent"
                elif lc in {"semaine", "week", "periode"}:
                    ren[c] = "semaine"
            if ren:
                df = df.rename(columns=ren)

        if key == "ias":
            # departement, nom, semaine, ias_signal
            ren = {}
            for c in df.columns:
                lc = c.lower()
                if lc in {"departement code", "département code", "dep_code", "code"}:
                    ren[c] = "departement"
                elif lc in {"departement", "département", "depnom"} and "nom" not in df.columns:
                    ren[c] = "nom"
                elif "ias" in lc and "ias_signal" not in df.columns:
                    ren[c] = "ias_signal"
                elif lc in {"semaine", "week", "periode"}:
                    ren[c] = "semaine"
            if ren:
                df = df.rename(columns=ren)

        if key == "urgences":
            # data ODISSE peuvent être au format taux; colonnes FR : "Département Code", "Département", "Semaine", ...
            # Schéma POC: departement, nom, semaine, urgences_grippe, sos_medecins
            ren = {}
            for c in df.columns:
                lc = c.lower()
                if lc in {"département code", "departement code", "dep_code", "code departement", "code"}:
                    ren[c] = "departement"
                elif lc in {"département", "departement", "depnom"} and "nom" not in df.columns:
                    ren[c] = "nom"
                elif lc in {"semaine", "week", "periode"}:
                    ren[c] = "semaine"
                elif "urgences" in lc and "urgences_grippe" not in df.columns:
                    ren[c] = "urgences_grippe"
                elif ("sos" in lc or "actes médicaux" in lc) and "sos_medecins" not in df.columns:
                    ren[c] = "sos_medecins"
            if ren:
                df = df.rename(columns=ren)

        if key == "distribution":
            # departement, nom, semaine, doses_distribuees, actes_pharmacie
            ren = {}
            for c in df.columns:
                lc = c.lower()
                if lc in {"département code", "departement code", "dep_code", "code"}:
                    ren[c] = "departement"
                elif lc in {"département", "departement", "depnom"} and "nom" not in df.columns:
                    ren[c] = "nom"
                elif lc in {"semaine", "week", "periode"}:
                    ren[c] = "semaine"
                elif "dose" in lc and "doses_distribuees" not in df.columns:
                    ren[c] = "doses_distribuees"
                elif "acte" in lc and "pharm" in lc and "actes_pharmacie" not in df.columns:
                    ren[c] = "actes_pharmacie"
            if ren:
                df = df.rename(columns=ren)

        # normalisations communes
        df = normalize_dept(df, "departement")
        df = normalize_week(df, "semaine")
        dfs[key] = df

    return dfs


def data_signature() -> Tuple[float, ...]:
    mtimes = []
    for key, fname in REQUIRED_FILES.items():
        if key == "geojson":
            p = find_file(fname)
            mtimes.append(_mtime(p) if p else 0.0)
        else:
            p = find_file(fname)
            mtimes.append(_mtime(p) if p else 0.0)
    return tuple(mtimes)


# ======================
# PRÉPARATION / MODÈLES
# ======================
@dataclass(frozen=True)
class PredBundle:
    model: LinearRegression
    features_used: List[str]


def build_training_frame(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    vacc = d["vaccination_trends"].copy()
    ias = d["ias"].copy()
    urg = d["urgences"].copy()

    need = {"departement", "semaine", "couverture_vaccinale_percent"}
    if vacc.empty or not need.issubset(vacc.columns):
        return pd.DataFrame()

    vacc[["year", "week_num"]] = vacc["semaine"].apply(lambda s: pd.Series(_split_week(s)))
    vacc.sort_values(["departement", "year", "week_num"], inplace=True)
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

    # Cible proxy (POC) si pas d'historique de distribution hebdo au même grain
    merged["target_proxy"] = (
        2200
        + (100 - merged["couverture_vaccinale_percent"]) * 38
        + merged["ias_signal"].fillna(0) * 24
        + merged["activity_total"] * 6.5
        - merged["trend_cov"] * 115
    ).clip(lower=500)

    merged = merged.dropna(subset=["couverture_vaccinale_percent"]).copy()
    return merged


def train_predictor(train_df: pd.DataFrame) -> PredBundle | None:
    if train_df.empty:
        return None
    feats = ["couverture_vaccinale_percent", "trend_cov", "ias_signal", "activity_total"]
    m = LinearRegression()
    X = train_df[feats].fillna(0.0)
    y = train_df["target_proxy"]
    m.fit(X, y)
    return PredBundle(model=m, features_used=feats)


def latest_week_from_any(d: Dict[str, pd.DataFrame]) -> str | None:
    pool = []
    for key in ("distribution", "vaccination_trends", "ias", "urgences"):
        if "semaine" in d[key].columns:
            pool.extend(d[key]["semaine"].dropna().tolist())
    if not pool:
        return None
    pool = list(set(pool))

    def k(w: str) -> Tuple[int, int]:
        try:
            return _split_week(w)
        except Exception:
            return (0, 0)

    return sorted(pool, key=k)[-1]


def build_latest_frame(d: Dict[str, pd.DataFrame], force_week: str | None) -> pd.DataFrame:
    wk = force_week or latest_week_from_any(d) or "N/A"

    def sel(df, cols):
        if df.empty:
            return pd.DataFrame(columns=cols)
        return df[df["semaine"] == wk][cols]

    last_trend = sel(
        d["vaccination_trends"],
        ["departement", "nom", "semaine", "couverture_vaccinale_percent"],
    )
    # calc trend_cov à partir de la semaine précédente
    if not d["vaccination_trends"].empty:
        tmp = d["vaccination_trends"].copy()
        tmp[["y", "w"]] = tmp["semaine"].apply(lambda s: pd.Series(_split_week(s)))
        tmp = tmp.sort_values(["departement", "y", "w"])
        prev = tmp[tmp["semaine"] < wk].groupby("departement").tail(1)[
            ["departement", "couverture_vaccinale_percent"]
        ].rename(columns={"couverture_vaccinale_percent": "prev_cov"})
        last_trend = last_trend.merge(prev, on="departement", how="left")
        last_trend["trend_cov"] = (last_trend["couverture_vaccinale_percent"] - last_trend["prev_cov"]).fillna(0.0)
    else:
        last_trend["trend_cov"] = 0.0

    last_ias = sel(d["ias"], ["departement", "ias_signal"]).copy()
    last_urg = sel(d["urgences"], ["departement", "urgences_grippe", "sos_medecins"]).copy()
    last_dist = sel(d["distribution"], ["departement", "doses_distribuees", "actes_pharmacie"]).copy()

    base = last_trend.copy()
    for part in (last_ias, last_urg, last_dist):
        base = base.merge(part, on="departement", how="left")

    base["activity_total"] = base["urgences_grippe"].fillna(0) + base["sos_medecins"].fillna(0)
    return base


def predict_needs(df: pd.DataFrame, bundle: PredBundle | None, coverage_target: float) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if bundle is None:
        out["besoin_prevu"] = np.nan
        out["risk_score"] = np.nan
        out["risk_level"] = "unknown"
        return out

    X = out[bundle.features_used].fillna(0.0)
    raw_pred = bundle.model.predict(X).clip(min=0)

    # Ajustement vers cible couverture (uplift si sous la cible)
    gap = np.clip(coverage_target * 100 - out["couverture_vaccinale_percent"], a_min=0, a_max=None)
    z_ias = _zscore(out["ias_signal"].fillna(out["ias_signal"].median() if out["ias_signal"].notna().any() else 0))
    ias_boost = np.clip(1 + 0.10 * z_ias, 0.8, 1.25)
    out["besoin_prevu"] = (raw_pred * (1 + 0.30 * (gap / 100.0)) * ias_boost).astype(float)

    # Risk scoring
    z_urg = _zscore(out["activity_total"].fillna(0))
    z_trd = _zscore(out["trend_cov"].fillna(0))
    risk = 0.5 * z_ias + 0.4 * z_urg + 0.1 * (-z_trd)
    out["risk_score"] = risk
    out["risk_level"] = pd.cut(risk, bins=[-1e9, -0.25, 0.75, 1e9], labels=["low", "med", "high"]).astype(str)
    return out


def propose_redistribution(df: pd.DataFrame, flex_pct: float = 0.05) -> pd.DataFrame:
    tmp = df.copy()
    tmp["balance"] = tmp["doses_distribuees"].fillna(0) - tmp["besoin_prevu"].fillna(0)
    donors = tmp[tmp["balance"] > flex_pct * (tmp["besoin_prevu"].fillna(0) + 1)].copy()
    takers = tmp[tmp["balance"] < -flex_pct * (tmp["besoin_prevu"].fillna(0) + 1)].copy()

    donors = donors.sort_values("balance", ascending=False).reset_index(drop=True)
    takers = takers.sort_values("balance", ascending=True).reset_index(drop=True)

    moves = []
    i = j = 0
    while i < len(donors) and j < len(takers):
        give = donors.loc[i, "balance"]
        need = -takers.loc[j, "balance"]
        qty = min(give, need)
        if qty <= 0:
            break
        moves.append(
            {
                "from_dept": donors.loc[i, "departement"],
                "from_nom": donors.loc[i, "nom"],
                "to_dept": takers.loc[j, "departement"],
                "to_nom": takers.loc[j, "nom"],
                "qty_suggested": int(qty),
            }
        )
        donors.at[i, "balance"] -= qty
        takers.at[j, "balance"] += qty
        if donors.loc[i, "balance"] <= flex_pct * (donors.loc[i, "besoin_prevu"] + 1):
            i += 1
        if takers.loc[j, "balance"] >= -flex_pct * (takers.loc[j, "besoin_prevu"] + 1):
            j += 1

    return pd.DataFrame(moves)


# ============
# UI HELPERS
# ============
def sidebar_controls(d: Dict[str, pd.DataFrame]) -> dict:
    st.sidebar.header("⚙️ Paramètres")

    pool = []
    for key in ("distribution", "vaccination_trends", "ias", "urgences"):
        if "semaine" in d[key].columns:
            pool.extend(d[key]["semaine"].dropna().unique().tolist())
    pool = sorted(set(pool), key=lambda w: _split_week(w) if "-" in str(w) else (0, 0))
    sel_week = st.sidebar.selectbox("Semaine", options=pool or ["N/A"], index=len(pool) - 1 if pool else 0)

    st.sidebar.markdown("---")
    view = st.sidebar.radio("Vue carte", ["Surveillance", "Besoins prévus", "Urgences"], index=0)

    st.sidebar.markdown("---")
    cov_target_pct = st.sidebar.slider("Cible de couverture (%)", 30, 80, 60, 1)
    under_threshold = st.sidebar.slider("Seuil sous-vaccination (%)", 30, 70, 45, 1)
    palette = st.sidebar.selectbox("Palette", ["YlGnBu", "YlOrRd", "Viridis"], index=0)

    st.sidebar.caption("Données : ODISSE / IAS® / IQVIA (versions POC, CSV locaux).")
    return dict(
        week=sel_week,
        view=view,
        coverage_target=cov_target_pct / 100.0,
        under_threshold=under_threshold,
        palette=palette,
    )


def kpi_header(df: pd.DataFrame, under_threshold: int) -> None:
    c1, c2, c3, c4 = st.columns(4)
    cov_avg = df["couverture_vaccinale_percent"].mean()
    needs_total = df["besoin_prevu"].sum()
    urg_avg = df["activity_total"].mean()
    high_risk = (df["risk_level"] == "high").sum()

    c1.metric("Couverture moyenne", f"{cov_avg:.1f} %")
    c2.metric("Doses à prévoir (total)", fmt_int(needs_total))
    c3.metric("Urgences+SOS (moy.)", f"{urg_avg:.0f}")
    c4.metric("Départements à risque (high)", f"{int(high_risk)}")

    under = df[df["couverture_vaccinale_percent"] < under_threshold]
    if not under.empty:
        st.caption("Sous-vaccinés (<{}%) : {}".format(
            under_threshold,
            ", ".join(sorted(under["nom"].dropna().astype(str).unique()))
        ))


def build_map(df: pd.DataFrame, geojson: dict, view: str, palette: str) -> px.choropleth_mapbox:
    metric = {
        "Surveillance": "couverture_vaccinale_percent",
        "Besoins prévus": "besoin_prevu",
        "Urgences": "activity_total",
    }[view]
    title = {
        "Surveillance": "Taux de couverture (%)",
        "Besoins prévus": "Besoins vaccinaux (doses)",
        "Urgences": "Urgences + SOS (hebdo)",
    }[view]
    scale = {"Surveillance": "YlGnBu", "Besoins prévus": "YlOrRd", "Urgences": "YlOrRd"}[view]
    if palette != "YlGnBu" and view == "Surveillance":
        scale = palette
    if palette != "YlOrRd" and view != "Surveillance":
        scale = palette

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
        "risk_level": True,
    }

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        locations="departement",
        featureidkey="properties.code",
        color=metric,
        color_continuous_scale=scale,
        hover_name="nom",
        hover_data=hover_data,
        mapbox_style="carto-positron",
        center={"lat": 46.6, "lon": 1.9},
        zoom=4.7,
        opacity=0.78,
        labels={metric: title},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=title))
    return fig


def render_tabs(latest_df: pd.DataFrame, geojson: dict, controls: dict, all_data: Dict[str, pd.DataFrame]) -> None:
    tab1, tab2, tab3 = st.tabs(["🗺️ Surveillance", "📈 Tendances & Prédictions", "🚚 Logistique"])
    with tab1:
        st.plotly_chart(
            build_map(latest_df, geojson, "Surveillance" if controls["view"] == "Surveillance" else controls["view"], controls["palette"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Classements rapides
        c1, c2 = st.columns(2)
        with c1:
            top_need = latest_df.sort_values("besoin_prevu", ascending=False)[["nom", "departement", "besoin_prevu"]].head(15)
            st.markdown("**Top besoins (doses)**")
            st.dataframe(top_need, use_container_width=True, hide_index=True)
        with c2:
            top_urg = latest_df.sort_values("activity_total", ascending=False)[["nom", "departement", "activity_total"]].head(15)
            st.markdown("**Top activité grippe (Urgences + SOS)**")
            st.dataframe(top_urg, use_container_width=True, hide_index=True)

    with tab2:
        # Tendance couverture (moyenne nationale)
        hist = all_data["vaccination_trends"].copy()
        if not hist.empty:
            trend = (
                hist.groupby("semaine")
                .agg(couverture_moyenne=("couverture_vaccinale_percent", "mean"))
                .reset_index()
            )
            fig = px.line(
                trend,
                x="semaine",
                y="couverture_moyenne",
                markers=True,
                title="Couverture vaccinale – moyenne hebdomadaire (France)",
                labels={"semaine": "Semaine", "couverture_moyenne": "Couverture (%)"},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Pas d'historique de couverture (vaccination_trends.csv).")

        # Corrélation simple IAS / Urgences / Besoin
        corr_df = latest_df[["ias_signal", "activity_total", "besoin_prevu"]].dropna()
        if len(corr_df) >= 5:
            st.markdown("**Corrélation IAS / Activité / Besoin**")
            st.dataframe(corr_df.corr().round(2), use_container_width=True)
        else:
            st.info("Trop peu de données pour une corrélation exploitable sur la semaine sélectionnée.")

    with tab3:
        st.markdown("**Propositions de redistribution (excédents → déficits)**")
        plan = propose_redistribution(latest_df, flex_pct=0.05)
        if plan.empty:
            st.success("Aucune redistribution nécessaire (seuil ±5%).")
        else:
            st.dataframe(plan, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Exporter les propositions (CSV)",
                data=plan.to_csv(index=False).encode("utf-8"),
                file_name=f"redistribution_{latest_df['semaine'].iloc[0]}.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.markdown("**Table complète – Export**")
        display_cols = [
            "departement","nom","semaine",
            "couverture_vaccinale_percent","ias_signal",
            "urgences_grippe","sos_medecins","activity_total",
            "doses_distribuees","actes_pharmacie","trend_cov",
            "besoin_prevu","risk_level","risk_score",
        ]
        table = latest_df[[c for c in display_cols if c in latest_df.columns]].copy()
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Exporter la table (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"poc_table_{latest_df['semaine'].iloc[0]}.csv",
            mime="text/csv",
        )


# =========
# MAIN APP
# =========
def main() -> None:
    st.title("POC – Carte Prédictive Vaccination Grippe")
    st.caption("Exploration interactive : couverture vaccinale, besoins prévus, activité SOS/urgences et logistique.")

    # Chargement GeoJSON (ou uploader)
    geojson = load_geojson_file()
    if geojson is None:
        st.warning("`departements.geojson` introuvable. Dépose le fichier ci-dessous pour activer la carte :")
        up = st.file_uploader("Uploader departements.geojson", type=["geojson", "json"])
        if up is not None:
            geojson = json.load(io.StringIO(up.getvalue().decode("utf-8")))
        else:
            st.info("La carte sera masquée tant que le GeoJSON n'est pas fourni.")

    # Chargement CSV (avec fallback + normalisation)
    sig = data_signature()
    data = load_all_csvs(sig)

    # Si un fichier est manquant, proposer des uploaders ciblés
    missing = [k for k in ("vaccination_trends","ias","urgences","distribution") if data[k].empty]
    if missing:
        st.error("Fichiers CSV manquants ou vides : " + ", ".join(missing))
        with st.expander("Uploader des CSV (format POC)"):
            for key in missing:
                up = st.file_uploader(f"Déposer {REQUIRED_FILES[key]}", type=["csv"], key=f"up_{key}")
                if up is not None:
                    df = _read_csv_generic(Path(up.name))
                    df = pd.read_csv(up)  # lecture directe
                    # normalisations minimales
                    df = normalize_dept(df, "departement")
                    df = normalize_week(df, "semaine")
                    data[key] = df
        if any(data[k].empty for k in ("vaccination_trends","ias","urgences","distribution")):
            st.stop()

    # Contrôles UI
    controls = sidebar_controls(data)

    # Entraînement modèle
    train_df = build_training_frame(data)
    bundle = train_predictor(train_df)

    # Semaine sélectionnée
    latest_df = build_latest_frame(data, force_week=controls["week"])
    latest_df = predict_needs(latest_df, bundle, coverage_target=controls["coverage_target"])

    # KPI
    kpi_header(latest_df, under_threshold=controls["under_threshold"])

    # Vues / Onglets
    if geojson is not None:
        render_tabs(latest_df, geojson, controls, data)
    else:
        st.info("Carte indisponible (GeoJSON manquant). Les tableaux restent exportables.")
        # Affiche tout de même les tableaux / exports
        render_tabs(latest_df, {"type":"FeatureCollection","features":[]}, controls, data)


if __name__ == "__main__":
    main()