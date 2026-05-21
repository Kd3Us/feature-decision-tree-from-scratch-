import math
import random
import itertools


class ModelEvaluator:

	def __init__(self, model_class, param_grid=None, n_splits=10, random_state=None):
		self.model_class = model_class
		self.param_grid = param_grid
		self.n_splits = n_splits
		self.random_state = random_state

	def evaluate(self, model, X_test, Y_test):
		predictions = model.predict(X_test)
		Y_test_list = self._to_list(Y_test)
		score = self._f1_weighted(Y_test_list, predictions)
		print(f"F1 weighted : {score}")
		return score

	def grid_search(self, X, Y):
		param_names = list(self.param_grid.keys())
		value_lists = [self.param_grid[name] for name in param_names]
		all_combinations = list(itertools.product(*value_lists))

		results = {}
		for combination in all_combinations:
			params = {}
			for i in range(len(param_names)):
				params[param_names[i]] = combination[i]
			print(f"Testing params = {params}")

			mean_score, std_score = self._cross_validation(X, Y, params)
			results[combination] = mean_score
			print("-"*20)

		best_combination = max(results, key=results.get)
		best_score = results[best_combination]

		best_params = {}
		for i in range(len(param_names)):
			best_params[param_names[i]] = best_combination[i]
		print(f"Best params : {best_params} with f1_weighted = {best_score}")

		return best_params, best_score, results

	def _cross_validation(self, X, Y, params):
		folds = self._stratified_k_fold_split(X, Y)

		scores = []
		for fold_index in range(len(folds)):
			X_train_fold, Y_train_fold, X_validation_fold, Y_validation_fold = folds[fold_index]

			fold_model_object = self.model_class(**params)
			fold_model_object.fit(X_train_fold, Y_train_fold)
			predictions = fold_model_object.predict(X_validation_fold)
			score = self._f1_weighted(Y_validation_fold, predictions)
			scores.append(score)
			print(f"Fold {fold_index + 1}/{self.n_splits} : f1_weighted = {score}")

		mean_score = sum(scores) / len(scores)

		variance = 0
		for score in scores:
			variance += (score - mean_score) ** 2
		variance = variance / len(scores)
		std_score = math.sqrt(variance)

		print(f"Mean f1_weighted : {mean_score} +/- {std_score}")
		return mean_score, std_score

	def _stratified_k_fold_split(self, X, Y):
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

		shuffler = random.Random(self.random_state)
		for label in indices_by_class:
			shuffler.shuffle(indices_by_class[label])

		validation_indices_per_fold = []
		for fold_index in range(self.n_splits):
			validation_indices_per_fold.append([])

		for label in indices_by_class:
			class_indices = indices_by_class[label]
			for position in range(len(class_indices)):
				fold_index = position % self.n_splits
				index = class_indices[position]
				validation_indices_per_fold[fold_index].append(index)

		folds = []
		for fold_index in range(self.n_splits):
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