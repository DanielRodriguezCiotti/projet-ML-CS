# Prediction des Retards de TGV

Ce projet a pour objectif de prédire le temps de retard des TGV à l'arrivée, ainsi que d'identifier les principales causes de ces retards, en se concentrant sur les trains réservés par la SNCF.

## Obtenir les Données

Les données nécessaires pour ce projet peuvent être obtenues en suivant ces étapes :

1. Rendez-vous sur [cette page](https://www.data.gouv.fr/fr/datasets/regularite-mensuelle-tgv-par-liaisons/).
2. Allez tout en bas de la page.
3. Téléchargez le fichier au format CSV.
4. Utilisez la bibliothèque Python Pandas pour manipuler ces données.

Pour la visualisation préliminaire des données, vous pouvez utiliser [cette ressource](https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst/analyze/?sort=date&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJzcGxpbmUiLCJmdW5jIjoiQVZHIiwieUF4aXMiOiJwcmN0X2NhdXNlX2V4dGVybmUiLCJzY2llbnRpZmljRGlzcGxheSI6dHJ1ZSwiY29sb3IiOiJyYW5nZS1TZXQxIn1dLCJ4QXhpcyI6ImRhdGUiLCJtYXhwb2ludHMiOiIiLCJ0aW1lc2NhbGUiOiJtb250aCIsInNvcnQiOiIiLCJzZXJpZXNCcmVha2Rvd24iOiJzZXJ2aWNlIiwiY29uZmlnIjp7ImRhdGFzZXQiOiJyZWd1bGFyaXRlLW1lbnN1ZWxsZS10Z3YtYXFzdCIsIm9wdGlvbnMiOnsic29ydCI6ImRhdGUifX19XSwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZSwidGltZXNjYWxlIjoiIn0%3D) pour créer divers types de graphiques.

## Objectifs du Projet

Les principaux objectifs de ce projet sont :

- Appliquer différentes approches de machine learning sur le dataset pour prédire les retards des trains à l'arrivée au cours des 6 derniers mois de 2023, ainsi que d'identifier les principales causes de ces retards en se basant sur les pourcentages relevés dans le dataset. Les données de test seront celles des mois de 2023, tandis que les données d'entraînement seront celles des années précédentes.
  
- Comparer les performances des différents modèles à l'aide des métriques vues en cours.

- Tirer des conclusions sur l'approche la plus appropriée pour prédire les retards des trains et les causes associées.

## Rapport

Un rapport détaillé sur le travail effectué et les résultats obtenus sera rédigé. Il comprendra environ 5 pages et couvrira les différentes étapes du projet, les méthodologies appliquées, les résultats des modèles et les conclusions finales.
