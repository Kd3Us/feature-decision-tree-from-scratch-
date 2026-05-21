import math
import random
import itertools

from data_loader import load_normalized_data


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

	def evaluate(self, X_test, Y_test):
		predictions = self.predict(X_test)
		Y_test_list = self._to_list(Y_test)
		score = self._f1_weighted(Y_test_list, predictions)
		print(f"F1 weighted : {score}")
		return score

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

	def _precision(self, Y_true, Y_pred, target_class):
		true_positive = 0
		false_positive = 0
		for i in range(len(Y_true)):
			if Y_pred[i] == target_class and Y_true[i] == target_class:
				true_positive += 1
			elif Y_pred[i] == target_class and Y_true[i] != target_class:
				false_positive += 1
		if true_positive + false_positive == 0:
			return 0
		return true_positive / (true_positive + false_positive)

	def _recall(self, Y_true, Y_pred, target_class):
		true_positive = 0
		false_negative = 0
		for i in range(len(Y_true)):
			if Y_true[i] == target_class and Y_pred[i] == target_class:
				true_positive += 1
			elif Y_true[i] == target_class and Y_pred[i] != target_class:
				false_negative += 1
		if true_positive + false_negative == 0:
			return 0
		return true_positive / (true_positive + false_negative)

	def _f1_weighted(self, Y_true, Y_pred):
		classes = set(Y_true)
		total = len(Y_true)

		weighted_f1 = 0
		for c in classes:
			precision = self._precision(Y_true, Y_pred, c)
			recall = self._recall(Y_true, Y_pred, c)
			if precision + recall == 0:
				f1_class = 0
			else:
				f1_class = 2 * precision * recall / (precision + recall)

			support = 0
			for label in Y_true:
				if label == c:
					support += 1

			weighted_f1 += (support / total) * f1_class
		return weighted_f1

	def _stratified_k_fold_split(self, X, Y, n_splits=10, random_state=None):
		if hasattr(Y, "tolist"):
			Y_list = Y.tolist()
		else:
			Y_list = list(Y)

		indices_by_class = {}
		for i in range(len(Y_list)):
			label = Y_list[i]
			if label not in indices_by_class:
				indices_by_class[label] = []
			indices_by_class[label].append(i)

		shuffler = random.Random(random_state)
		for label in indices_by_class:
			shuffler.shuffle(indices_by_class[label])

		validation_indices_per_fold = []
		for fold_index in range(n_splits):
			validation_indices_per_fold.append([])

		for label in indices_by_class:
			class_indices = indices_by_class[label]
			for position in range(len(class_indices)):
				fold_index = position % n_splits
				index = class_indices[position]
				validation_indices_per_fold[fold_index].append(index)

		folds = []
		for fold_index in range(n_splits):
			validation_indices = validation_indices_per_fold[fold_index]
			validation_set = set(validation_indices)

			train_indices = []
			for i in range(len(X)):
				if i not in validation_set:
					train_indices.append(i)

			X_train_fold = []
			Y_train_fold = []
			for i in train_indices:
				X_train_fold.append(X[i])
				Y_train_fold.append(Y_list[i])

			X_validation_fold = []
			Y_validation_fold = []
			for i in validation_indices:
				X_validation_fold.append(X[i])
				Y_validation_fold.append(Y_list[i])

			folds.append((X_train_fold, Y_train_fold, X_validation_fold, Y_validation_fold))
		return folds

	def _cross_validation(self, X, Y, n_splits=10, random_state=None):
		folds = self._stratified_k_fold_split(X, Y, n_splits=n_splits, random_state=random_state)

		scores = []
		for fold_index in range(len(folds)):
			X_train_fold, Y_train_fold, X_validation_fold, Y_validation_fold = folds[fold_index]

			fold_tree_object = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
			fold_tree_object.fit(X_train_fold, Y_train_fold)
			predictions = fold_tree_object.predict(X_validation_fold)
			score = fold_tree_object._f1_weighted(Y_validation_fold, predictions)
			scores.append(score)
			print(f"Fold {fold_index + 1}/{n_splits} : f1_weighted = {score}")

		mean_score = sum(scores) / len(scores)

		variance = 0
		for score in scores:
			variance += (score - mean_score) ** 2
		variance = variance / len(scores)
		std_score = math.sqrt(variance)

		print(f"Mean f1_weighted : {mean_score} +/- {std_score}")
		return mean_score, std_score

	def grid_search(self, X, Y, param_grid, n_splits=10, random_state=None):
		param_names = list(param_grid.keys())
		value_lists = [param_grid[name] for name in param_names]
		all_combinations = list(itertools.product(*value_lists))

		results = {}
		for combination in all_combinations:
			params = {}
			for i in range(len(param_names)):
				params[param_names[i]] = combination[i]
			print(f"Testing params = {params}")

			candidate_tree_object = DecisionTree(**params)
			mean_score, std_score = candidate_tree_object._cross_validation(X, Y, n_splits=n_splits, random_state=random_state)
			results[combination] = mean_score
			print("-"*20)

		best_combination = max(results, key=results.get)
		best_score = results[best_combination]

		best_params = {}
		for i in range(len(param_names)):
			best_params[param_names[i]] = best_combination[i]
		print(f"Best params : {best_params} with f1_weighted = {best_score}")

		self.max_depth = best_params["max_depth"]
		self.min_samples_split = best_params["min_samples_split"]
		self.fit(X, Y)
		return best_params, best_score, results


if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	decision_tree_object = DecisionTree()
	best_params, best_score, results = decision_tree_object.grid_search(
		X_normalized[:500],
		Y.iloc[:500],
		param_grid={
			"max_depth": [3, 5, 10],
			"min_samples_split": [2, 5, 10],
		},
		n_splits=5
	)