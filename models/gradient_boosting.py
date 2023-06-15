import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_gradient_boosting(games_df, x_train, x_test, y_train, y_test):
    model = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
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

    predictions_actuals_df.to_csv('result/predictions_actuals.csv', index=False)
    return model, accuracy