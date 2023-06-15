import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_random_forest(games_df, x_train, x_test, y_train, y_test):

    model = RandomForestClassifier(n_estimators=500, max_depth=10)
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
    return model, accuracy
