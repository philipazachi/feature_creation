import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def train_xgboost(games_df, x_train, x_test, y_train, y_test):
    # find_best_hyperparameters(x_train, y_train)

    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data = xgb.DMatrix(x_test, label=y_test)

    # Step 4: Set hyperparameters
    params = {'max_depth': 6, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'error'}
    num_rounds = 200

    # Step 5: Train the model
    model = xgb.train(params, train_data, num_rounds)

    # Step 6: Evaluate the model
    y_pred = model.predict(test_data)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
    accuracy = accuracy_score(y_test, y_pred_binary)

    # Save the DataFrame to a CSV file
    # predictions_actuals_df.to_csv('result/predictions_actuals.csv', index=False)
    return accuracy, y_test, y_pred


def find_best_hyperparameters(X, y):
    # Define the parameter grid to search over
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [1.5, 1.2, 1.1, 1, 0.9, 0.5, 0.1, 0.01, 0.001],
        'n_estimators': [100, 200, 300],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Create an XGBoost classifier
    xgb_model = xgb.XGBClassifier()

    # Perform grid search using cross-validation
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5)
    grid_search.fit(X, y)

    # Print the best hyperparameters and their corresponding score
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)
