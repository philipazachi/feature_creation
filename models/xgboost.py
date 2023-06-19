import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def train_xgboost(games_df, x_train, x_test, y_train, y_test):
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


def train_xgboost_with_finetune(games_df, x_train, x_test, y_train, y_test):
    param_grid = {
        'max_depth': [3, 5, 7, 9],  # Example values for max_depth
        'learning_rate': [0.1, 0.05, 0.01],  # Example values for learning_rate
        'n_estimators': [100, 200, 300],  # Example values for n_estimators
        'objective': ['binary:logistic', 'binary:logitraw']  # Example values for objective
    }
    model = xgb.XGBClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(best_params)
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred
