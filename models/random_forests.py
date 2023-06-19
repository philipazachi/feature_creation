import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def train_random_forest(games_df, x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=500, max_depth=10)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the DataFrame to a CSV file
    # predictions_actuals_df.to_csv('result/predictions_actuals.csv', index=False)
    return accuracy, y_test, y_pred


def train_random_forest_with_finetune(games_df, x_train, x_test, y_train, y_test):
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],  # Example values for n_estimators
        'max_depth': [None, 3, 5, 8, 10],  # Example values for max_depth
        'min_samples_split': [2, 5, 7, 10],  # Example values for min_samples_split
    }
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(best_params)
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred
