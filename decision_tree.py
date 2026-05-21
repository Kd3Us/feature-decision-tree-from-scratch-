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