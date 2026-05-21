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