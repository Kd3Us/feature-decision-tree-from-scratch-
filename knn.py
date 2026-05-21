from data_loader import load_normalized_data
from tqdm import tqdm

# La classe KNN encapsule toute la logique de l'algorithme.
# Une classe regroupe des données (attributs) et des comportements (méthodes)
# dans un seul objet cohérent. Ici, un objet KNN sait s'entraîner et prédire.
class KNN:

	# __init__ est le "constructeur" : il s'exécute automatiquement dès qu'on
	# crée un objet KNN (ex: KNN(n_neighbors=7)).
	# self désigne l'objet lui-même — c'est par lui qu'on accède aux attributs.
	def __init__(self, n_neighbors):
		self.n_neighbors = n_neighbors  # K : le nombre de voisins à consulter


	# Phase 1 — Entraînement.
	# KNN est un "lazy learner" : il ne calcule rien ici, il mémorise simplement
	# les exemples. Tout le travail se fait au moment de predict().
	def fit(self, X_normalized, Y):
		self.X_normalized = X_normalized  # features normalisées (tableau 2D)
		self.Y = Y                        # étiquettes correspondantes (classes)
		self.set_of_labels = set(Y)       # set() déduplique : [0,1,1,0] → {0,1}

		# Compter combien d'exemples appartiennent à chaque classe.
		# dict comprehension : crée {0: 0, 1: 0} puis on incrémente.
		label_occurences = {label : 0 for label in self.set_of_labels}
		for y_value in Y:
			label_occurences[y_value] += 1

		max_occurences = max(label_occurences.values())

		# Calcul des poids pour compenser le déséquilibre entre classes.
		# Si la classe 1 est 3x moins fréquente, son vote comptera 3x plus fort.
		# Formule : poids = max_occurrences / occurrences_de_cette_classe
		self.label_weight = {label : max_occurences / label_occurences[label] \
									for label in self.set_of_labels}


	# Phase 2 — Prédiction.
	# Pour un nouveau point, on cherche ses K voisins les plus proches dans les
	# données d'entraînement, puis on fait un vote pondéré.
	def predict(self, x_to_forcast):
		# Garde-fou : si fit() n'a pas été appelé, les attributs n'existent pas.
		# hasattr() vérifie l'existence d'un attribut sans lever d'erreur.
		if not hasattr(self, "X_normalized") or not hasattr(self, "Y"):
			raise Exception("The model is not fit")

		# Étape 1 — Calculer la distance entre x_to_forcast et CHAQUE point connu.
		# enumerate() fournit à la fois l'index et la valeur à chaque itération.
		# tqdm affiche une barre de progression (utile : cette boucle peut être longue).
		distances = {}
		for index_x, current_x in tqdm(enumerate(self.X_normalized), total=len(self.X_normalized)):
			current_distance = KNN.euclidian_distance(current_x, x_to_forcast)
			# Arrondi à 2 décimales : évite que 1.000001 et 1.000002 soient traités
			# comme des distances différentes alors qu'elles sont pratiquement égales.
			distances[index_x] = float(f"{current_distance:.2f}")

		# Étape 2 — Trier les points du plus proche au plus loin.
		sorted_distances = KNN.sorted_dict_by_values(distances)

		# Étape 3 — Garder uniquement les K premiers index (les K plus proches voisins).
		# [:self.n_neighbors] est un slice : prend les n premiers éléments de la liste.
		indexes_nearest_neighbors = list(sorted_distances.keys())[: self.n_neighbors]

		# Étape 4 — Vote pondéré : chaque voisin vote pour sa classe.
		# Le poids compense le déséquilibre : une classe rare vote plus fort.
		label_counter = {label : 0 for label in self.set_of_labels}
		for index_point in indexes_nearest_neighbors:
			label = self.Y[index_point]
			label_counter[label] += self.label_weight[label]

		# Étape 5 — La classe avec le score total le plus élevé est la prédiction.
		# Le dictionnaire est trié par ordre croissant, donc le gagnant est le dernier [-1].
		sorted_label_counter = KNN.sorted_dict_by_values(label_counter)
		predicted_label = list(sorted_label_counter.keys())[-1]

		return predicted_label


	# @staticmethod : cette méthode n'utilise pas self, elle ne dépend d'aucun
	# attribut de l'objet. C'est un outil utilitaire lié conceptuellement à KNN.
	# On peut l'appeler directement : KNN.euclidian_distance(x1, x2)
	@staticmethod
	def euclidian_distance(x1, x2):
		x1, x2 = list(x1), list(x2)[0]  # conversion en listes Python standard

		# Formule : √[ Σ(aᵢ - bᵢ)² ]
		# zip(strict=True) associe les features deux à deux et lève une erreur
		# si les deux points n'ont pas le même nombre de dimensions.
		return sum([(a-b)**2 for a, b in zip(x1, x2, strict=True)])**(1/2)


	# Trie un dictionnaire par ses valeurs (ordre croissant).
	# lambda est une fonction anonyme : lambda item: item[1] extrait la valeur
	# de chaque paire (clé, valeur) pour l'utiliser comme critère de tri.
	@staticmethod
	def sorted_dict_by_values(dict_object):
		sorted_dict = dict(sorted(dict_object.items(), key=lambda item: item[1]))
		return sorted_dict





if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	knn_object = KNN(n_neighbors=7)
	knn_object.fit(X_normalized, Y)
	print(knn_object.predict([X_normalized[7]]))
	print(Y[7])


