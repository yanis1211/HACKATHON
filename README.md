# EpiTrack – carnet d'utilisation (POC grippe)

Ce document explique **pas à pas** comment lancer et comprendre EpiTrack. Imagine que tu accompagnes un collègue qui découvre l'outil pour la première fois : on suit les étapes ensemble, avec des mots simples et aucune boîte noire.

---

## 1. À quoi sert EpiTrack ?

EpiTrack est un tableau de bord Streamlit qui aide les acteurs de la santé publique à :

1. Repérer les **départements sous-vaccinés** (couverture grippe faible).
2. Estimer les **besoins en doses** mois par mois (prévision 12 mois).
3. Visualiser la **pression sur les urgences/SOS Médecins**.
4. Répartir un **stock national** de vaccins selon les besoins.
5. Comprendre les hypothèses (moyenne mobile, correction IAS, effort hiver) via des sliders visibles.

> **Important** : seules les données métropolitaines (01–95, 2A, 2B) sont conservées. Les codes 97/98 (DOM‑TOM) sont filtrés d’office.

Cliquer [ici](#10-résumé-express-3-minutes-chrono) pour le getting started express

---

## 2. Préparer l'environnement

1. Avoir **Python 3.10+** installé.
2. Cloner ou télécharger le repo, puis depuis la racine :
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # sous Windows : .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Vérifier le dossier `data/manual/`. Les fichiers suivants peuvent être collés manuellement (CSV/Excel) :

   | Fichier | Colonnes minimales | Utilité |
   | --- | --- | --- |
   | `vaccination_trends.csv` | `departement`, `semaine` (YYYY-ww), `couverture_vaccinale_percent`, `nom` (optionnel) | Taux hebdomadaire grippe |
   | `ias.csv` | `departement`, `semaine`, `ias_signal` | Signal IAS (incidence) |
   | `urgences.csv` | `departement`, `semaine`, `urgences_grippe`, `sos_medecins` | Passages hebdo |
   | `distribution.csv` | `departement`, `semaine`, `doses_distribuees`, `actes_pharmacie` | Livraisons hebdo |
   | `coverage.csv` *(facultatif)* | `departement`, `couverture_vaccinale_percent`, `nom` | Valeur fallback |
   | `meteo.csv` *(facultatif)* | Exemple : `temp_moy`, `humidite`… | Pas utilisé pour ce POC mais géré |
   | `departements.geojson` | `properties.code`, `properties.nom` | Carte des départements métropolitains |

   Si un fichier manque, EpiTrack génère des données **mock** pour garder la démo fonctionnelle.

---

## 3. Lancer l'application

```bash
streamlit run src/app/streamlit_app.py
```

Une page s’ouvre dans ton navigateur (`http://localhost:8501`). Sur la gauche, tu vois la **barre latérale** avec :

- un **sélecteur de mois** (historique ou prévision M+1/M+2/M+3),
- des **curseurs** : `Cible couverture`, `Seuil sous-vaccination`, `Uplift hiver (%)`, `Coefficient IAS (k)`.

Les paramètres actifs sont rappelés en haut de l’écran principal pour savoir d’un coup d’œil ce qui est appliqué.

---

## 4. Ce que fait EpiTrack sur les données

1. **Normalisation** : noms de colonnes harmonisés (`Code Departement` → `departement`, etc.), zfill des codes (`1` → `01`).
2. **Agrégation mensuelle** : chaque semaine est rattachée au mois de sa fin de semaine.
   - `couverture_mois` = moyenne pondérée par un proxy population (`doses + flux*25 + 5000`).
   - `incidence_mois` = IAS média (winsorized P5/P95 pour lisser les extrêmes).
   - `flux_mois` = urgences + SOS à l’échelle du mois.
   - `confidence` = faible (<5 mois), moyenne (5–8), élevée (≥9).
3. **Prévision** : moyenne mobile 3 mois + tendance + corrections.
   - `coverage_proj = MA3 + trend + k * zscore(IAS)` *(tronqué entre -2 et 2)*.
   - `besoin_prevu = max(0, pop_proxy * cible – pop_proxy * coverage_proj /100) * (1 + uplift hiver)`.
   - L’uplift hiver s’applique aux mois {novembre, décembre, janvier, février}.
4. **Flux patients** : `flux_proj = max(0, flux_ma3 + trend_flux)`.
5. **Allocation** : `ratio = besoin_dpt / somme(besoins)` puis `allocation = floor(ratio * stock)` avec un plafonnement si `couverture_mois ≥ cible`.

Les mois futurs apparaissent avec une étiquette `(prévision)` dans l’interface.

---

## 5. Naviguer dans l’application

### 5.1 KPI globaux

En haut :

- Doses estimées (total des besoins du mois sélectionné).
- Couverture moyenne pondérée.
- Nombre de départements en risque rouge (<60 %).
- Flux patients (urgences + SOS).

### 5.2 Onglet « 🗺️ Carte »

- Basculer entre `Couverture (%)` et `Besoin (doses)`.
- Infobulle : `nom`, `couverture`, `besoin`, `flux`, `risque`.
- Notice en bas : **Comment lire / Hypothèse / Action** (ex. “cibler les zones rouges avant la campagne”).

### 5.3 Onglet « 📈 Prévisions vaccins »

- Sélectionner un département.
- Indicateurs : besoin estimé, couverture actuelle, badge de confiance.
- Graphique combiné (axe gauche = besoin, axe droit = flux). La légende rappelle la signification.
- Courbe nationale (besoin total) sur 12 mois.
- Note “Historique court : prudence” si < 6 mois d’historique.

### 5.4 Onglet « 🚚 Distribution »

- Entrer le stock national (valeur numérique).
- Définir la couverture cible.
- Cliquer “Calculer l’allocation” : le tableau propose un plan (colonne `Historique/Prévision`).
- Bouton “Télécharger CSV”.
- Note “Action : redistribuer avant saturation des zones rouges”.

### 5.5 Onglet « ℹ️ Notes »

- Rappelle les formules utilisées.
- Donne une interprétation simple des sliders.

---

## 6. Exemple rapide

1. Mois choisi : `2025-10`.
2. Cible couverture : `65 %`. Uplift hiver : `20 %`. k IAS : `0.15`.
3. Carte → identifier les départements rouges (<60 %). Légende suggère campagne ciblée.
4. Prévisions → département “Nord (59)” : besoin ~3 200 doses, flux en hausse → alerte sur renfort logistique.
5. Distribution → stock national `80 000`. Allocation exportable (`allocation_2025-10.csv`).

---

## 7. Commandes utiles (pour vérifier les données)

- Aperçu des données agrégées :
  ```bash
  PYTHONPATH=src python3 - <<'PY'
  from core.data_loader import load_data_bundle
  bundle = load_data_bundle()
  print(bundle.monthly_metrics.head())
  PY
  ```

- Calcul d’une prévision mensuelle :
  ```bash
  PYTHONPATH=src python3 - <<'PY'
  from core.data_loader import load_data_bundle
  from core.model import predict_needs

  bundle = load_data_bundle()
  month = bundle.months[-1]
  result = predict_needs(bundle, month, 60, 20, 0.15)
  print(result.per_department[['departement','besoin_prevu','confidence']].head())
  PY
  ```

---

## 8. Bonnes pratiques et limites

- Ajouter idéalement **12 mois** d’historique pour de meilleures tendances.
- Vérifier les entêtes lors de la création des CSV (même si le loader renomme, garder les champs essentiels).
- Les ressources DOM‑TOM sont exclues ; si tu souhaites les intégrer, adapter le code.
- La météo est chargée mais non utilisée (prévu pour une future version).
- Les formules sont simples (pas de machine learning avancé) pour rester expliquables.

---

## 9. Pistes d’évolution

- Intégrer un modèle ARIMA/Prophet pour raffiner la prévision.
- Ajouter des données socio-démographiques (densité, âge) dans `monthly_metrics`.
- Exposer les résultats via une API (FastAPI) pour d’autres dashboards.
- Connecter une base (Postgres, BigQuery) pour éviter la saisie manuelle.

---

## 10. Résumé express (3 minutes chrono)

1. `source .venv/bin/activate` puis `pip install -r requirements.txt`.
2. Vérifier / coller les fichiers CSV dans `data/manual/`.
3. `streamlit run src/app/streamlit_app.py`.
4. Choisir un mois dans la barre latérale et régler les sliders.
5. Explorer Carte → Prévisions → Distribution → Notes.
6. Exporter le plan d’allocation si besoin.
