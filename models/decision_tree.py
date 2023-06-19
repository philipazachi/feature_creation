import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def train_decision_tree(games_df, x_train, x_test, y_train, y_test):
    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'min_samples_split': [2, 3, 4, 5, 7, 10],
    }
    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(best_params)
    # return accuracy, y_test, y_pred

    # predictions_actuals_df = pd.DataFrame({
    #     'Match_Num': games_df.loc[y_test.index, 'match_num'],
    #     'Player_1_Name': games_df.loc[y_test.index, 'player_1_name'],
    #     'Player_2_Name': games_df.loc[y_test.index, 'player_2_name'],
    #     'Predicted': y_pred,
    #     'Actual': y_test
    # })

    # x_test['model_predictions'] = y_pred
    # x_test['player_1_won'] = y_test
    # filtered_df = x_test[x_test['model_predictions'] != x_test['player_1_won']]
    # filtered_df.to_csv('result/predictions_actuals.csv', index=False)
    # visualize_decision_tree(model, x_train)
    # show_feature_importance(best_model, x_train.columns.tolist())
    return accuracy, y_test, y_pred


def predict_winner_decision_tree(game_features, model):
    # new_match_features = pd.DataFrame({'rank_diff': [10], 'height_diff': [5], ...})
    new_match_features = pd.DataFrame(game_features)
    predicted_winner = model.predict(new_match_features)
    return predicted_winner


def visualize_decision_tree(model, x_train):
    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_tree(model, feature_names=x_train.columns, filled=True, ax=ax)
    plt.show()


def show_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    top_feature_names = [feature_names[i] for i in indices[:5]]
    top_importances = importances[indices[:5]]
    plt.figure(figsize=(10, 6))
    plt.title("Top {} Feature Importances".format(5))
    plt.bar(range(len(top_importances)), top_importances)
    plt.xticks(range(len(top_importances)), top_feature_names, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()
