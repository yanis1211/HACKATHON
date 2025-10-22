# POC – Surveillance vaccinale grippe

VaxiScope est un assistant décisionnel Streamlit pour piloter la vaccination antigrippale : il consolide les données hebdomadaires (couverture, IAS, urgences, distribution, météo) et fournit une prévision de besoins, des alertes logistiques et sanitaires, ainsi qu’un ciblage territorial.

## Vues métier
- **🗺️ Carte** : choroplèthe unique (toggle Couverture / Besoin), légende rouge‑orange‑vert, infobulle synthétique.
- **📈 Prévisions vaccins** : sélection d’un département → courbe besoins sur 12 mois, flux patients projetés, encadré « Besoin / Confiance ».
- **🚚 Distribution** : saisie du stock national + cible de couverture → bouton « Calculer l’allocation » (répartition proportionnelle), export CSV et note d’action.
- **ℹ️ Notes** : rappel des hypothèses (coeff IAS, uplift hiver, méthode d’allocation).

## Modèles simplifiés (POC)
- **Besoin vaccinal** : moyenne mobile (3 mois) de la couverture, ajustée par le signal IAS (coeff. doux 0–0,3) et un uplift hiver paramétrable (0–30 % sur nov.–févr.). Formule affichée dans l’UI.
- **Urgences/SOS** : tendance glissante sur les 12 derniers mois (fallback si historique < 6 mois, badge de confiance dans l’UI).
- **Allocation** : heuristique proportionnelle (besoin_dpt / somme(besoins)) avec plafonnement automatique pour les départements déjà ≥ cible.

Toutes les hypothèses sont affichées dans l’UI ; aucune boîte noire. Les départements 971‑973 restent regroupés pour lisibilité Antilles‑Guyane.

## Installation
Python 3.10+ et `pip`. Une fois le dépôt cloné :

```bash
pip install -r requirements.txt
```

## Lancer le POC

```bash
streamlit run src/app/streamlit_app.py
```

L’application s’ouvre sur `http://localhost:8501`. La barre latérale permet de sélectionner le mois, la cible de couverture, l’uplift hiver et le coefficient IAS.

## Jeux de données attendus

Les fichiers sont lus dans `data/manual/`. S’ils sont absents, le loader génère automatiquement des données simulées (mock) pour conserver une démonstration fonctionnelle.

- `departements.geojson` — `properties.code` (INSEE) + `properties.nom`.
- `vaccination_trends.csv` — `departement`, `semaine` (YYYY-ww), `nom` (optionnel), `couverture_vaccinale_percent`.
- `ias.csv` — `departement`, `semaine`, `ias_signal`.
- `urgences.csv` — `departement`, `semaine`, `urgences_grippe`, `sos_medecins`.
- `distribution.csv` — `departement`, `semaine`, `doses_distribuees`, `actes_pharmacie`.
- `coverage.csv` (fallback) — dernières valeurs de couverture (facultatif si `vaccination_trends` complet).
- `meteo.csv` (optionnel) — `departement`, `semaine`, `temp_moy`, `temp_min`, `humidite`, `precipitations`, `anomalie_temp`.

Grain pivot : `departement` (chaîne INSEE) × `semaine` (ISO). Les vues agrègent ensuite au mois (`YYYY-MM`) avec :

- `activity_total = urgences_grippe + sos_medecins`
- `trend_cov` (moyenne mobile 3 semaines)
- `besoin_prevu` (formule explicite : cible – doses estimées)
- `risk_level` (`rouge`, `orange`, `vert` selon couverture)

## Générer des données de démonstration

Le moteur charge automatiquement des mocks si un fichier est manquant. Pour créer manuellement des CSV, vous pouvez vous inspirer de `scripts/generate_mock_data.py` ou déposer vos propres extractions dans `data/manual/` en respectant les schémas ci-dessus.

## Pharmacies (données officielles)

Si vous voulez activer la recherche des pharmacies dans l'application, placez le fichier officiel fourni par Santé publique France dans `data/manual/`.

- Nom attendu : `santefr-lieux-vaccination-grippe-pharmacie.csv`
- Source officielle : https://www.data.gouv.fr/datasets/lieux-de-vaccination-contre-la-grippe-pharmacies-sante-fr/

Le loader priorise ce fichier manuel (séparateur `;`) et utilise les colonnes `Finess`, `Titre`, `Adresse_voie 1`, `Adresse_codepostal`, `Adresse_ville`, `Adresse_latitude`, `Adresse_longitude` pour construire la liste des lieux. Si le fichier est présent, la fonctionnalité "Pharmacies" dans le dashboard lira directement ces lieux sans appeler l'API Overpass.

Remarques :
- Le champ `Modalites_accueil` contient du HTML/texte libre (horaires, modalités RDV) — on peut implémenter un nettoyage si nécessaire.
- Si vous préférez récupérer les lieux depuis OpenStreetMap, utilisez `src/etl/fetch_pharmacies.py` (interroge Overpass). Attention aux quotas des services publics.
