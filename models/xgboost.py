import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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

    predictions_actuals_df = pd.DataFrame({
        'Match_Num': games_df.loc[y_test.index, 'match_num'],
        'Player_1_Name': games_df.loc[y_test.index, 'player_1_name'],
        'Player_2_Name': games_df.loc[y_test.index, 'player_2_name'],
        'Predicted': y_pred_binary,
        'Actual': y_test
    })

    # Save the DataFrame to a CSV file
    predictions_actuals_df.to_csv('result/predictions_actuals.csv', index=False)
    return model, accuracy
