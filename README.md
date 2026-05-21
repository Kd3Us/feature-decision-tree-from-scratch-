# Well-Being Project

Projet d'introduction au machine learning appliqué à la classification du bien-être.

## Description

Ce projet met en œuvre un pipeline de classification basé sur l'algorithme **K-Nearest Neighbors (KNN)** sur un jeu de données de bien-être (`bienetre.csv`). Il inclut une recherche automatique de la valeur optimale de `k` via une grille de recherche artisanale avec validation croisée.

## Pipeline

1. **Chargement des données** (`load_data`) — lecture du CSV, affichage de la distribution des classes (équilibre du dataset), séparation features/cible.
2. **Normalisation** (`nomalize`) — standardisation des features avec `StandardScaler`.
3. **Recherche de k optimal** (`find_optimal_kvalue`) — grid search artisanal sur `k ∈ [1, 99]` (valeurs impaires), sélection du meilleur `k` selon le F1 moyen.

## Utilisation

```bash
python main.py
```

Le script charge `bienetre.csv`, normalise les données, puis cherche la valeur de `k` optimale et affiche les résultats.

## Dépendances

- Python 3.x
- pandas
- scikit-learn
