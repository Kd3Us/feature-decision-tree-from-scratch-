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
		pass

	def predict(self, X):
		pass

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