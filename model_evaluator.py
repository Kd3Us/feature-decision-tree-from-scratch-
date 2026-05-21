import math
import random
import itertools


class ModelEvaluator:

	def __init__(self, model_class, param_grid=None, n_splits=10, random_state=None):
		self.model_class = model_class
		self.param_grid = param_grid
		self.n_splits = n_splits
		self.random_state = random_state

	def _to_list(self, Y):
		if hasattr(Y, "tolist"):
			return Y.tolist()
		return list(Y)

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