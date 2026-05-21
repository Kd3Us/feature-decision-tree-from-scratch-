from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



def find_optimal_params(
		model_name,
		model,
		X_normalized,
		Y,
		param_grid,
		n_splits=10,
		scoring="f1_weighted",
	):
	print(f"Finding optimal params for {model_name}")
	cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


	grid_search = GridSearchCV(model(), param_grid, cv=cross_validation_object, scoring=scoring)
	grid_search.fit(X_normalized, Y)


	print("Best Params")
	print(grid_search.best_params_)
	print(f"{scoring} : {grid_search.best_score_}")

	print("-"*100)
	return grid_search.best_score_, grid_search.best_params_




def find_optimal_model(
		X_normalized,
		Y,
		n_splits=10,
		scoring="f1_weighted",
	):

	dict_parms_per_model = {
		"KNeighborsClassifier": (
			KNeighborsClassifier,
			{
				'n_neighbors': list(range(1, 100, 2)),
			},
		),
		"DecisionTreeClassifier": (
			DecisionTreeClassifier,
			{
				'criterion': ['gini', 'entropy'],
				'max_depth': [None, 3, 5, 7, 10, 15, 20],
				'min_samples_split': [2, 5, 10, 20],
				'min_samples_leaf': [1, 2, 5, 10],
				'random_state': [42],
			},
		),
		"RandomForestClassifier": (
			RandomForestClassifier,
			{
				'n_estimators': [100, 200],
				'max_depth': [None, 10, 20],
				'min_samples_split': [2, 5],
				'min_samples_leaf': [1, 2],
				'random_state': [42],
			},
		),
	}

	best_score = -float("inf")
	best_model_name = None
	best_params = None
	for model_name, model_to_optimize in dict_parms_per_model.items():
		score, params = find_optimal_params(
			model_name=model_name,
			model= model_to_optimize[0],
			X_normalized=X_normalized,
			Y=Y,
			param_grid=model_to_optimize[1],
			n_splits=10,
			scoring="f1_weighted",
		)
		if score > best_score:
			best_score = score
			best_model_name = model_name
			best_params = params

	print(f"Le model le plus optimisé est {best_model_name}")
	print(f"Best Params : {best_params}")
	print(f"{scoring} : {best_score}")

	return best_model_name, best_params, best_score