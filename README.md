# EpiTrack – POC Tableau de bord grippe

EpiTrack est un prototype Streamlit destiné aux équipes santé publique (ARS, ministère, hôpitaux, pharmaciens). Il consolide les données de vaccination grippe, signaux IAS, passages urgences/SOS et distribution pour :

- visualiser la couverture vaccinale et les besoins prévisionnels par département,
- anticiper les flux patients sur 12 mois,
- proposer une allocation de stocks proportionnelle aux besoins,
- présenter clairement les hypothèses (corrigé IAS, saisonnalité) sans boîte noire.

## 1. Fonctionnalités principales

| Vue | Objectif | Contenu |
| --- | --- | --- |
| 🗺️ Carte | Surveiller zones sous-vaccinées | Choroplèthe métropole, toggle `Couverture` / `Besoin`, infobulle actionnable |
| 📈 Prévisions vaccins | Suivre besoin vs flux | Graphique combiné Besoin (axe gauche) + Flux patients (axe droit) sur 12 mois, badge de confiance, résumé hypothèses |
| 🚚 Distribution | Allouer le stock national | Paramètres `Stock` & `Cible`, heuristique d’allocation (proportionnelle + cap), export CSV |
| ℹ️ Notes | Transparence du modèle | Rappel des formules (moyenne mobile 3 mois, correction IAS, uplift hiver) |

Les DOM‑TOM (codes INSEE 97/98) sont exclus pour rester sur un périmètre métropolitain cohérent.

## 2. Prérequis

- Python 3.10 ou plus
- `pip` ou `pipenv`

## 3. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate  # sous Windows : .\.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Lancer l’application

```bash
streamlit run src/app/streamlit_app.py
```

L’interface est accessible sur `http://localhost:8501`. La barre latérale propose :

1. Sélecteur de mois (`historique` ou `prévision` M+1/M+2/M+3)
2. Curseurs `Cible couverture`, `Seuil sous-vaccination`, `Uplift hiver (%)`, `Coefficient IAS`
3. Rappels sur les données historiques disponibles.

Les paramètres actifs (cible, uplift, IAS) sont rappelés en haut de l’écran, juste sous les KPI.

## 5. Jeux de données attendus (`data/manual/`)

Chaque fichier peut être collé manuellement depuis Excel / CSV. Les noms de colonnes sont normalisés automatiquement (lowercase, accents, etc.) ; si un fichier manque, un jeu de données mock est généré pour garder la démo fonctionnelle.

| Fichier | Colonnes minimales | Remarques |
| --- | --- | --- |
| `vaccination_trends.csv` | `departement`, `semaine` (YYYY-ww), `couverture_vaccinale_percent`, `nom` (optionnel) | Taux hebdomadaire de couverture |
| `ias.csv` | `departement`, `semaine`, `ias_signal` | Incidence / signal IAS |
| `urgences.csv` | `departement`, `semaine`, `urgences_grippe`, `sos_medecins` | Nombre d’actes hebdomadaires |
| `distribution.csv` | `departement`, `semaine`, `doses_distribuees`, `actes_pharmacie` | Livraisons hebdomadaires |
| `coverage.csv` (optionnel) | `departement`, `couverture_vaccinale_percent`, `nom` | Valeur fallback si l’historique manque |
| `meteo.csv` (optionnel) | `departement`, `semaine`, `temp_moy`, `temp_min`, `humidite`, `precipitations`, `anomalie_temp` | Non exploité dans le modèle actuel mais structure déjà prévue |
| `departements.geojson` | `properties.code`, `properties.nom` | Base géographique carte métropolitaine |

**Important :** les codes INSEE sont gérés en chaîne (`01`, `2A`, `2B`), et les départements `97*` / `98*` sont filtrés.

Le moteur charge automatiquement des mocks si un fichier est manquant. Pour créer manuellement des CSV, vous pouvez vous inspirer de `scripts/generate_mock_data.py` ou déposer vos propres extractions dans `data/manual/` en respectant les schémas ci-dessus.

## Pharmacies (données officielles)

Si vous voulez activer la recherche des pharmacies dans l'application, placez le fichier officiel fourni par Santé publique France dans `data/manual/`.

- Nom attendu : `santefr-lieux-vaccination-grippe-pharmacie.csv`
- Source officielle : https://www.data.gouv.fr/datasets/lieux-de-vaccination-contre-la-grippe-pharmacies-sante-fr/

Le loader priorise ce fichier manuel (séparateur `;`) et utilise les colonnes `Finess`, `Titre`, `Adresse_voie 1`, `Adresse_codepostal`, `Adresse_ville`, `Adresse_latitude`, `Adresse_longitude` pour construire la liste des lieux. Si le fichier est présent, la fonctionnalité "Pharmacies" dans le dashboard lira directement ces lieux sans appeler l'API Overpass.

Remarques :
- Le champ `Modalites_accueil` contient du HTML/texte libre (horaires, modalités RDV) — on peut implémenter un nettoyage si nécessaire.
- Si vous préférez récupérer les lieux depuis OpenStreetMap, utilisez `src/etl/fetch_pharmacies.py` (interroge Overpass). Attention aux quotas des services publics.
## 6. Flux de transformation

1. **Normalisation** : chaque CSV est aligné (`departement`, `semaine`).
2. **Agrégation mensuelle** :
   - `couverture_mois` = moyenne pondérée par un proxy population (`doses + flux*25 + offset`).
   - `incidence_mois` = moyenne winsorized (P5/P95) pour éviter les outliers.
   - `flux_mois` = somme urgences + SOS sur les semaines du mois (rattachées au mois de fin de semaine).
   - `confidence` = élevée ≥ 9 mois d’historique, moyenne 5–8, faible sinon.
3. **Prévision** : moyenne mobile 3 mois + tendance + corrections (IAS, uplift hiver).

## 7. Modèle

### 7.1 Besoin vaccinal (par département × mois)

```
couverture_proj = coverage_ma3 + trend_cov + k * zscore(IAS)
besoin_prevu   = max(0, population_proxy * cible – population_proxy * couverture_proj / 100) * (1 + uplift_hiver)
```

- `coverage_ma3` : moyenne mobile 3 mois
- `trend_cov` : différence MA3 vs mois-1 (corrige la pente)
- `k` : coefficient IAS (slider 0 – 0,30)
- `uplift_hiver` : appliqué aux mois 11-12-01-02 (slider 0 – 30 %)

### 7.2 Flux patients (urgences + SOS)

```
flux_proj = max(0, flux_ma3 + trend_flux)
```

- Fallback “confiance faible” si < 6 mois d’historique.

### 7.3 Allocation des stocks

```
ratio_dpt = besoin_prevu_dpt / somme(besoin_prevu)
allocation = floor(ratio_dpt * stock_national)
si couverture_mois ≥ cible => allocation *= 0.1
```

Le tableau affiche `allocation_proposée`, `stock_restant` et un badge “Historique / Prévision”.

## 8. Utilisation pas à pas

1. **Vérifier / alimenter `data/manual/`**. Un simple `ls data/manual` permet de voir les fichiers en place. Les colonnes peuvent être à entête libre (le loader les renomme), mais il faut garder l’intitulé des variables (voir section 5).
2. **Lancer Streamlit** et sélectionner un mois dans la barre latérale.
3. **Ajuster les sliders** :
   - `Cible couverture (%)`: objectif de campagne.
   - `Seuil sous-vaccination (%)`: pour l’assistant décisionnel.
   - `Uplift hiver (%)`: intensité saisonnière.
   - `Coefficient IAS`: sensibilité aux signaux IAS.
4. **Explorer les onglets** :
   - **Carte** : repérer les zones rouge/orange ; survoler pour voir besoin, flux, tendance.
   - **Prévisions vaccins** : choisir un département, analyser besoin vs flux, noter la confiance.
   - **Distribution** : renseigner le stock national et calculer l’allocation ; exporter le CSV.
   - **Notes** : rappeler les hypothèses et expliquer le changement des sliders.

## 9. Exemples de commandes utiles

- Vérifier la présence des colonnes après normalisation :

  ```bash
  PYTHONPATH=src python3 - <<'PY'
  from core.data_loader import load_data_bundle
  bundle = load_data_bundle()
  print(bundle.vaccination_trends.head())
  PY
  ```

- Générer les prévisions pour le dernier mois :

  ```bash
  PYTHONPATH=src python3 - <<'PY'
  from core.data_loader import load_data_bundle, generate_future_months
  from core.model import predict_needs

  bundle = load_data_bundle()
  month = bundle.months[-1]
  result = predict_needs(bundle, month, 60, 20, 0.15)
  print(result.per_department[['departement','besoin_prevu','confidence']].head())
  PY
  ```

## 10. Notes & limites

- POC volontairement simple : pas de ML “boîte noire”, tout est basé sur moyennes mobiles / corrections paramétrables.
- Les prévisions futures sont signalées `(prévision)` dans l’UI.
- Les départements manquants ou incohérents se voient attribuer des valeurs mock pour garder une démo fluide.
- Pour ajouter des jeux de données optionnels (météo, socio-démo), l’architecture de `core/data_loader.py` permet de fusionner de nouvelles sources.

---
