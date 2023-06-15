import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def train_decision_tree(games_df, x_train, x_test, y_train, y_test):
    model = DecisionTreeClassifier(max_depth=6)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    predictions_actuals_df = pd.DataFrame({
        'Match_Num': games_df.loc[y_test.index, 'match_num'],
        'Player_1_Name': games_df.loc[y_test.index, 'player_1_name'],
        'Player_2_Name': games_df.loc[y_test.index, 'player_2_name'],
        'Predicted': y_pred,
        'Actual': y_test
    })

    # Save the DataFrame to a CSV file
    predictions_actuals_df.to_csv('result/predictions_actuals.csv', index=False)
    visualize_decision_tree(model, x_train)
    show_feature_importance(model, x_train)
    return model, accuracy


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


def show_feature_importance(model, x_train):
    # Access the feature importance
    importance = model.feature_importances_

    # Create a pandas Series with feature names as the index and feature importance as the values
    feature_importance = pd.Series(importance, index=x_train.columns)

    # Sort the feature importance values in descending order
    feature_importance = feature_importance.sort_values(ascending=False)

    # Plot the feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance.plot(kind='bar', ax=ax)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()
