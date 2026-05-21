# Support de cours — L'algorithme KNN implémenté en Python

**Niveau :** Débutant  
**Prérequis :** Bases de Python (variables, boucles, fonctions)  
**Objectifs :** Comprendre le machine learning et la programmation orientée objet à travers l'implémentation de l'algorithme KNN

---

## 1. Introduction — Qu'est-ce que le Machine Learning ?

Le **machine learning** (apprentissage automatique) est une branche de l'informatique où l'on apprend à une machine à prendre des décisions à partir d'**exemples**, sans lui écrire les règles à la main.

### Les 3 phases clés

```
  Données          Entraînement        Prédiction
┌──────────┐      ┌────────────┐      ┌───────────┐
│ bienetre │ ───> │    fit()   │ ───> │ predict() │
│   .csv   │      │  (mémoriser│      │ (deviner) │
└──────────┘      │  les ex.) │      └───────────┘
                  └────────────┘
```

| Phase | Ce qui se passe | Dans notre code |
|---|---|---|
| **Charger** | Lire les données | `load_normalized_data()` |
| **Entraîner** | Mémoriser les exemples | `knn.fit(X, Y)` |
| **Prédire** | Répondre à une nouvelle question | `knn.predict(x)` |

> **Analogie :** Un médecin apprend à diagnostiquer une maladie en examinant des milliers de patients. Le machine learning, c'est pareil, mais avec un programme informatique.

---

## 2. L'algorithme KNN — Intuition

**KNN** signifie **K-Nearest Neighbors** (K plus proches voisins). L'idée est simple :

> _"Pour classer un nouveau point, regarde ses K voisins les plus proches et fais un vote."_

### Exemple visuel

Imaginons des données en 2D (ici, `stress` et `imc`) :

```
  imc
   |
 + |    +       ?        Question : à quelle classe
 + |      +    /         appartient le point "?" ?
   |       +  /
   |        +
 - |  - -
 - | -
   +---------------------> stress

  + = classe 1 (bien-être élevé)
  - = classe 0 (bien-être faible)
  ? = nouveau patient à classer
```

Avec **K = 3**, on regarde les 3 voisins les plus proches de `?` : s'il y a 2 voisins `+` et 1 voisin `-`, le vote donne **classe 1**.

### Le rôle de K

| K petit (ex: K=1) | K grand (ex: K=20) |
|---|---|
| Très sensible aux anomalies | Plus stable, moins précis |
| Risque de sur-apprentissage | Risque de sous-apprentissage |

Dans notre code, K se règle au moment de créer le modèle :

```python
knn_object = KNN(n_neighbors=7)  # K = 7
```

---

## 3. La distance euclidienne — Mesurer la ressemblance

Pour trouver les voisins les plus proches, il faut **mesurer la distance** entre deux points. On utilise la **distance euclidienne**, qui est simplement la distance en ligne droite entre deux points.

### La formule

Pour deux points `x1 = (a₁, a₂, ..., aₙ)` et `x2 = (b₁, b₂, ..., bₙ)` :

```
d(x1, x2) = √[ (a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)² ]
```

### Traduction en Python

```python
@staticmethod
def euclidian_distance(x1, x2):
    x1, x2 = list(x1), list(x2)[0]   # (1) convertir en listes

    return sum([(a-b)**2 for a, b in zip(x1, x2, strict=True)])**(1/2)
    #           ^^^^^^^^^^^^^^^^^^^^^^^^                            ^^^^
    #           (a-b)² pour chaque paire de features           racine carrée
```

**Décortiquons ligne par ligne :**

- `list(x1), list(x2)[0]` — on convertit les données en listes Python simples
- `zip(x1, x2, strict=True)` — associe `a₁` avec `b₁`, `a₂` avec `b₂`, etc. Le `strict=True` garantit que les deux points ont exactement le même nombre de dimensions (sinon erreur)
- `(a-b)**2` — calcul de chaque terme au carré
- `sum([...])**(1/2)` — somme puis racine carrée

---

## 4. La POO — Pourquoi encapsuler dans une classe ?

### Sans classe : le problème

```python
# Code sans classe — difficile à maintenir
X_train = None
Y_train = None
n_neighbors = 7

def fit(X, Y):
    global X_train, Y_train   # variables globales dangereuses
    X_train = X
    Y_train = Y

def predict(x):
    if X_train is None:        # on doit vérifier manuellement
        raise Exception("Non entraîné !")
    # ...
```

Ce code fonctionne pour **un seul modèle**. Si on veut tester plusieurs valeurs de K simultanément, tout est mélangé.

### Avec une classe : la solution

Une **classe** regroupe des **données** (attributs) et des **comportements** (méthodes) dans un seul objet cohérent :

```
┌─────────────────────────────────┐
│           Objet KNN             │
├─────────────────────────────────┤
│  Attributs (état)               │
│  ├── n_neighbors = 7            │
│  ├── X_normalized = [[...]]     │
│  ├── Y = [0, 1, 1, 0, ...]      │
│  └── label_weight = {0:1, 1:1.2}│
├─────────────────────────────────┤
│  Méthodes (comportements)       │
│  ├── fit(X, Y)                  │
│  ├── predict(x)                 │
│  ├── euclidian_distance(x1, x2) │
│  └── sorted_dict_by_values(d)   │
└─────────────────────────────────┘
```

On peut maintenant créer **plusieurs modèles indépendants** :

```python
knn_3 = KNN(n_neighbors=3)   # modèle avec K=3
knn_7 = KNN(n_neighbors=7)   # modèle avec K=7

knn_3.fit(X, Y)
knn_7.fit(X, Y)

# Les deux coexistent sans interférer
print(knn_3.predict([x]))
print(knn_7.predict([x]))
```

### Le cycle de vie d'un objet KNN

```
KNN(n_neighbors=7)    →    fit(X, Y)    →    predict(x)
      ↓                        ↓                  ↓
  Création              Entraînement         Prédiction
  (stocke K)         (mémorise X et Y)   (retourne la classe)
```

---

## 5. Décomposition complète de la classe KNN

### 5.1 `__init__` — Le constructeur

```python
class KNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
```

`__init__` est appelé automatiquement à la création de l'objet. `self` désigne l'objet lui-même. Ici, on stocke simplement le paramètre K.

```python
knn = KNN(n_neighbors=7)
# Maintenant : knn.n_neighbors == 7
```

---

### 5.2 `fit` — L'entraînement

```python
def fit(self, X_normalized, Y):
    self.X_normalized = X_normalized     # (1) mémoriser les données
    self.Y = Y
    self.set_of_labels = set(Y)          # (2) ex: {0, 1}

    label_occurences = {label : 0 for label in self.set_of_labels}
    for y_value in Y:
        label_occurences[y_value] += 1   # (3) compter chaque classe

    max_occurences = max(label_occurences.values())

    self.label_weight = {label : max_occurences / label_occurences[label] \
                            for label in self.set_of_labels}  # (4) poids
```

**Points importants :**

**(1) KNN ne "calcule" rien lors de l'entraînement** — il mémorise simplement les exemples. C'est pourquoi on appelle KNN un *lazy learner* (apprenant paresseux).

**(2) `set(Y)`** crée un ensemble sans doublons. Si Y = `[0,1,1,0,1]`, alors `set(Y)` = `{0, 1}`.

**(3) Compter les occurrences** : on construit un dictionnaire `{0: 150, 1: 50}` si la classe 0 apparaît 3 fois plus que la classe 1.

**(4) Les poids de classe** compensent le déséquilibre. Si la classe 1 est 3 fois moins représentée, son vote comptera 3 fois plus lors de la prédiction.

```
Exemple :
  label_occurences = {0: 150, 1: 50}
  max_occurences   = 150
  label_weight     = {0: 150/150=1.0,  1: 150/50=3.0}
                                                  ^^^
                               la classe 1 vote 3 fois plus fort
```

---

### 5.3 `predict` — La prédiction

```python
def predict(self, x_to_forcast):
    if not hasattr(self, "X_normalized") or not hasattr(self, "Y"):
        raise Exception("The model is not fit")      # (1) garde-fou

    distances = {}
    for index_x, current_x in tqdm(enumerate(self.X_normalized),  # (2) boucle
                                    total=len(self.X_normalized)):
        current_distance = KNN.euclidian_distance(current_x, x_to_forcast)
        distances[index_x] = float(f"{current_distance:.2f}")      # (3) arrondi

    sorted_distances = KNN.sorted_dict_by_values(distances)        # (4) tri

    indexes_nearest_neighbors = list(sorted_distances.keys())[: self.n_neighbors]  # (5) K voisins

    label_counter = {label : 0 for label in self.set_of_labels}
    for index_point in indexes_nearest_neighbors:
        label = self.Y[index_point]
        label_counter[label] += self.label_weight[label]           # (6) vote pondéré

    sorted_label_counter = KNN.sorted_dict_by_values(label_counter)
    predicted_label = list(sorted_label_counter.keys())[-1]        # (7) gagnant

    return predicted_label
```

**Étape par étape :**

**(1)** `hasattr()` vérifie si l'attribut existe sur l'objet. Si on appelle `predict()` sans avoir fait `fit()` avant, on lève une erreur explicite plutôt que de planter de façon mystérieuse.

**(2)** `enumerate()` donne à la fois l'index et la valeur à chaque itération :
```python
for index_x, current_x in enumerate(self.X_normalized):
    # index_x = 0, 1, 2, ...
    # current_x = les features du point correspondant
```
`tqdm` affiche une barre de progression dans le terminal — utile car la boucle peut être longue.

**(3)** `f"{current_distance:.2f}"` arrondit à 2 décimales. Cela évite que deux distances très proches (ex: 1.000001 vs 1.000002) soient considérées comme différentes.

**(4) & (5)** On trie les distances du plus proche au plus loin, puis on prend les K premiers index.

**(6)** Le **vote pondéré** : chaque voisin vote pour sa classe, mais avec un poids qui dépend de la rareté de cette classe.

**(7)** Le dictionnaire trié par valeur croissante : le gagnant est le **dernier** (valeur maximale).

---

### 5.4 Les méthodes statiques (`@staticmethod`)

```python
@staticmethod
def euclidian_distance(x1, x2):
    ...

@staticmethod
def sorted_dict_by_values(dict_object):
    sorted_dict = dict(sorted(dict_object.items(), key=lambda item: item[1]))
    return sorted_dict
```

Une méthode `@staticmethod` **n'utilise pas `self`** — elle ne dépend d'aucun attribut de l'objet. C'est une fonction utilitaire logiquement liée à la classe, mais qui pourrait exister seule.

**Quand utiliser `@staticmethod` ?** Quand la méthode :
- ne lit ni ne modifie aucun attribut de l'objet
- est liée conceptuellement à la classe (ici, c'est un outil de KNN)

```python
# On peut l'appeler sans créer d'instance :
d = KNN.euclidian_distance(x1, x2)
```

---

## 6. Le pipeline complet

```
bienetre.csv
     │
     ▼
load_data()          ← pandas lit le CSV, sépare X (features) et Y (cible)
     │
     ▼
normalize()          ← StandardScaler centre et réduit chaque feature
     │
     ▼
KNN.fit(X, Y)        ← mémorise les exemples, calcule les poids
     │
     ▼
KNN.predict(x)       ← retourne la classe prédite
```

### Pourquoi normaliser ?

Sans normalisation, certaines features dominent le calcul de distance :

```
Patient A : age=30,  revenu=2000, stress=5
Patient B : age=31,  revenu=3000, stress=6

Distance brute ≈ √[(30-31)² + (2000-3000)² + (5-6)²]
              ≈ √[1 + 1000000 + 1]
              ≈ 1000   ← dominé par le revenu !
```

Après normalisation (StandardScaler recentre chaque feature sur sa moyenne, divisé par son écart-type), toutes les features contribuent équitablement.

### Le code complet d'utilisation

```python
# Extrait du bloc if __name__ == "__main__" dans knn.py
from data_loader import load_normalized_data

X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")

knn_object = KNN(n_neighbors=7)
knn_object.fit(X_normalized, Y)

print(knn_object.predict([X_normalized[7]]))  # prédiction pour le patient n°7
print(Y[7])                                   # vraie valeur (pour comparer)
```

> **Note :** le bloc `if __name__ == "__main__":` garantit que ce code ne s'exécute que lorsque le fichier est lancé directement (pas quand il est importé par un autre module).

---

## 7. Récapitulatif — Concepts Python utilisés

| Concept Python | Où dans le code | Pourquoi |
|---|---|---|
| `class` / `self` | `class KNN`, toutes les méthodes | Encapsuler état et comportements |
| `__init__` | `def __init__(self, n_neighbors)` | Initialiser l'objet à la création |
| `@staticmethod` | `euclidian_distance`, `sorted_dict_by_values` | Méthode utilitaire sans état |
| `hasattr()` | début de `predict()` | Vérifier qu'un attribut existe |
| `enumerate()` | boucle dans `predict()` | Obtenir index et valeur simultanément |
| `set()` | `set(Y)` dans `fit()` | Dédupliquer les labels |
| `dict` | `distances`, `label_counter` | Associer index/label à une valeur |
| `zip(strict=True)` | `euclidian_distance` | Itérer sur deux listes en parallèle |
| `tqdm` | boucle dans `predict()` | Afficher une barre de progression |
| `lambda` | `sorted_dict_by_values` | Fonction anonyme pour le tri |
| `raise Exception` | `predict()`, `load_data()` | Signaler une erreur explicitement |

---

## 8. Exercices

### Niveau 1 — Observation (sans modifier le code)

Lancez le fichier `knn.py` tel quel :
```bash
python knn.py
```
Puis modifiez `n_neighbors` de 7 à 1, puis à 15. Observez comment la prédiction change.

**Question :** Pour quelle valeur de K la prédiction est-elle identique à la vraie valeur `Y[7]` ?

---

### Niveau 2 — Modifier l'algorithme

Implémentez la **distance de Manhattan** (somme des différences absolues) en remplacement de la distance euclidienne :

```
d_manhattan(x1, x2) = |a₁-b₁| + |a₂-b₂| + ... + |aₙ-bₙ|
```

**Indice :** dans `euclidian_distance`, remplacez `(a-b)**2` et `**(1/2)` par les opérations correspondantes.

**Question :** La prédiction change-t-elle pour `X_normalized[7]` ?

---

### Niveau 3 — Enrichir la classe

Ajoutez une méthode `score(self, X_test, Y_test)` à la classe `KNN` qui :
1. Prédit la classe pour chaque point de `X_test`
2. Compare avec les vraies valeurs `Y_test`
3. Retourne le taux de bonnes prédictions (entre 0.0 et 1.0)

```python
def score(self, X_test, Y_test):
    # À compléter
    ...
```

**Exemple d'utilisation attendu :**
```python
accuracy = knn_object.score(X_normalized[:50], Y[:50])
print(f"Précision : {accuracy:.1%}")  # ex: "Précision : 82.0%"
```

---

*Document généré à partir du commit `c51bf50 — feature : knn vanilla implementation`*
