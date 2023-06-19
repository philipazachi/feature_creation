from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def adaboost(games_df, x_train, x_test, y_train, y_test, n_estimators=100, learning_rate=1.1):
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred


def adaboost_with_finetune(games_df, x_train, x_test, y_train, y_test):
    param_grid = {
        'n_estimators': [50, 100, 150, 200],  # Example values for n_estimators
        'learning_rate': [1.2, 1, 0.5, 0.1, 0.05, 0.01],  # Example values for learning_rate
    }
    model = AdaBoostClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(best_params)
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred