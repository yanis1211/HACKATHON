# POC – Surveillance vaccinale grippe

Prototype Streamlit qui illustre un outil d'aide à la décision pour suivre la couverture vaccinale contre la grippe à l'échelle des départements français.

## Contenu
- **Carte interactive** des départements (GeoJSON intégré) avec trois vues : couverture, besoins prévisionnels et activité urgences/SOS Médecins.
- **Modèle prédictif** (régression linéaire) apprenant à partir de tendances vaccinales simulées et d'un signal IAS pour estimer les besoins en vaccins.
- **Tableau de bord** avec KPIs synthétiques et graphiques (barplot & courbe de tendance).
- **Jeu de données synthétique** stocké dans `data/manual/`, généré pour chaque département et plusieurs semaines.

## Prérequis
Python 3.10+ et `pip`. Une fois le dépôt cloné :

```bash
pip install -r requirements.txt
```

## Lancer le POC

```bash
streamlit run src/app/streamlit_app.py
```

Une interface web s'ouvre (ou `http://localhost:8501`). Utiliser la barre latérale pour basculer entre les vues.

## Régénérer les données simulées (optionnel)

Les CSV fournis suffisent pour le POC. Pour régénérer un nouveau scénario :

```bash
python3 scripts/generate_mock_data.py
```

Les fichiers sont écrasés dans `data/manual/`.
