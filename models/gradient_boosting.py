import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def train_gradient_boosting(games_df, x_train, x_test, y_train, y_test):
    model = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred


def train_gradient_boosting_with_finetune(games_df, x_train, x_test, y_train, y_test):
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 300],  # Example values for n_estimators
        'learning_rate': [0.1, 0.05, 0.01],  # Example values for learning_rate
        'max_depth': [3, 4, 5, 6, 7],  # Example values for max_depth
    }
    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(best_params)
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred