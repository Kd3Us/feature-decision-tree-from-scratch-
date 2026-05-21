from data_loader import load_normalized_data
from model_evaluator import ModelEvaluator


class DecisionTree:

	def __init__(self, max_depth=10, min_samples_split=2):
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.tree = None

	def fit(self, X, Y):
		X_list = self._to_list_of_rows(X)
		Y_list = self._to_list(Y)
		self.tree = self._build_tree(X_list, Y_list, depth=0)

	def predict(self, X):
		if self.tree is None:
			raise Exception("The model is not fit")

		X_list = self._to_list_of_rows(X)

		predictions = []
		for point in X_list:
			predicted_label = self._predict_one(point, self.tree)
			predictions.append(predicted_label)
		return predictions

	def _to_list_of_rows(self, X):
		rows = []
		for row in X:
			rows.append(list(row))
		return rows

	def _to_list(self, Y):
		if hasattr(Y, "tolist"):
			return Y.tolist()
		return list(Y)

	def _gini(self, Y):
		total = len(Y)
		if total == 0:
			return 0

		counts = {}
		for label in Y:
			if label not in counts:
				counts[label] = 0
			counts[label] += 1

		sum_of_squares = 0
		for label in counts:
			proportion = counts[label] / total
			sum_of_squares += proportion ** 2

		return 1 - sum_of_squares

	def _weighted_gini(self, Y_left, Y_right):
		total = len(Y_left) + len(Y_right)

		gini_left = self._gini(Y_left)
		gini_right = self._gini(Y_right)

		weight_left = len(Y_left) / total
		weight_right = len(Y_right) / total

		return weight_left * gini_left + weight_right * gini_right

	def _split_dataset(self, X, Y, feature_index, threshold):
		X_left = []
		Y_left = []
		X_right = []
		Y_right = []

		for i in range(len(X)):
			if X[i][feature_index] <= threshold:
				X_left.append(X[i])
				Y_left.append(Y[i])
			else:
				X_right.append(X[i])
				Y_right.append(Y[i])

		return X_left, Y_left, X_right, Y_right

	def _best_split(self, X, Y):
		number_of_features = len(X[0])

		best_feature_index = None
		best_threshold = None
		best_impurity = float("inf")

		for feature_index in range(number_of_features):
			feature_values = []
			for row in X:
				feature_values.append(row[feature_index])

			candidate_thresholds = sorted(set(feature_values))

			for threshold in candidate_thresholds:
				X_left, Y_left, X_right, Y_right = self._split_dataset(X, Y, feature_index, threshold)

				if len(Y_left) == 0 or len(Y_right) == 0:
					continue

				impurity = self._weighted_gini(Y_left, Y_right)

				if impurity < best_impurity:
					best_impurity = impurity
					best_feature_index = feature_index
					best_threshold = threshold

		return best_feature_index, best_threshold

	def _majority_label(self, Y):
		counts = {}
		for label in Y:
			if label not in counts:
				counts[label] = 0
			counts[label] += 1
		return max(counts, key=counts.get)

	def _build_tree(self, X, Y, depth):
		if len(set(Y)) == 1:
			return {"label": Y[0]}

		if depth >= self.max_depth:
			return {"label": self._majority_label(Y)}

		if len(Y) < self.min_samples_split:
			return {"label": self._majority_label(Y)}

		best_feature_index, best_threshold = self._best_split(X, Y)

		if best_feature_index is None:
			return {"label": self._majority_label(Y)}

		X_left, Y_left, X_right, Y_right = self._split_dataset(X, Y, best_feature_index, best_threshold)

		if len(Y_left) == 0 or len(Y_right) == 0:
			return {"label": self._majority_label(Y)}

		left_subtree = self._build_tree(X_left, Y_left, depth + 1)
		right_subtree = self._build_tree(X_right, Y_right, depth + 1)

		return {
			"feature_index": best_feature_index,
			"threshold": best_threshold,
			"left": left_subtree,
			"right": right_subtree,
		}

	def _predict_one(self, point, node):
		if "label" in node:
			return node["label"]

		if point[node["feature_index"]] <= node["threshold"]:
			return self._predict_one(point, node["left"])
		else:
			return self._predict_one(point, node["right"])


if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")

	evaluator = ModelEvaluator(
		model_class=DecisionTree,
		param_grid={
			"max_depth": [3, 5, 10],
			"min_samples_split": [2, 5, 10],
		},
		n_splits=5,
	)
	best_params, best_score, results = evaluator.grid_search(X_normalized[:500], Y.iloc[:500])