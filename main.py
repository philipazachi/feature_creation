import numpy as np
import pandas as pd

from df_preperation import convert_column_names, add_match_date_column, fill_nulls, prepare_df
from features.features_calculation import feature_diff_percent_by_court, feature_diff_wins_by_court, \
    feature_diff_wins_percent, feature_diff_wins, feature_diff_wins_between_players, feature_diff_days_since_last_match, \
    feature_players_retired, feature_intensity
from models_picker import train_all_models


def create_features(games_df):
    games_df['diff_ages'] = games_df["player_1_age"] - games_df["player_2_age"]
    games_df['diff_ht'] = games_df["player_1_ht"] - games_df["player_2_ht"]
    games_df['same_hands'] = np.where(games_df['player_1_hand'] == games_df['player_2_hand'], 1, 0)
    games_df['diff_rank'] = games_df['player_1_rank'] - games_df['player_2_rank']
    games_df['diff_rank_relative'] = (games_df['player_1_rank'] - games_df['player_2_rank']) / \
                                     (games_df['player_1_rank'] + games_df['player_2_rank'])
    games_df['diff_rank_points'] = games_df['player_1_rank_points'] - games_df['player_2_rank_points']
    games_df['diff_rank_points_relative'] = (games_df['player_1_rank_points'] - games_df['player_2_rank_points']) / \
                                            (games_df['player_1_rank_points'] + games_df['player_2_rank_points'])
    games_df['diff_60_days_intensity'] = feature_intensity(games_df, 60)
    print("diff_60_days_intensity")
    games_df['diff_retired_in_last_30_days'] = feature_players_retired(games_df, 30)
    print("diff_retired_in_last_30_days")
    games_df['diff_wins_between_players'] = feature_diff_wins_between_players(games_df, 1000)
    print("diff_wins_between_players")
    games_df['diff_number_of_days_since_last_played'] = feature_diff_days_since_last_match(games_df)
    print("diff_number_of_days_since_last_played")
    games_df['diff_percent_by_court'] = feature_diff_percent_by_court(games_df, 1000)
    print("diff_percent_by_court")
    games_df['diff_wins_by_court'] = feature_diff_wins_by_court(games_df, 1000)
    print("diff_wins_by_court")
    games_df['diff_wins_percent'] = feature_diff_wins_percent(games_df, 1000)
    print("diff_wins_percent")
    games_df['diff_wins'] = feature_diff_wins(games_df, 1000)
    print("diff_wins")
    return games_df


if __name__ == '__main__':
    df = pd.read_csv('data/atp_matches_1968.csv')
    df = prepare_df(df)
    print("prepared df")
    df.to_csv('result/games_with_match_date.csv', index=False)
    # df = pd.read_csv('result/games_with_match_date.csv')
    # df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date')
    df = create_features(df)
    df.to_csv('result/games_features_final.csv', index=False)
    # df = pd.read_csv('result/games_features_final.csv')
    # df['match_date'] = pd.to_datetime(df['match_date'])
    df = df[df['match_date'] >= pd.to_datetime('2010-01-01')]
    features_names = ['diff_ages', 'diff_ht', 'same_hands', 'diff_rank', 'diff_rank_relative',
                      'diff_rank_points', 'diff_percent_by_court', 'diff_wins_by_court', 'diff_wins_percent',
                      'diff_wins', 'diff_wins_between_players', 'diff_60_days_intensity',
                      'diff_retired_in_last_30_days', 'diff_number_of_days_since_last_played']
    results = train_all_models("tourney_level", df, features_names)

    # estimator = DecisionTreeClassifier(max_depth=6)
    # model, accuracy = adaboost(df, x_train, x_test, y_train, y_test, estimator, 100, 1.1)
    # print("adaboost accuracy: " + str(accuracy))
    # estimator = SVC(probability=True, kernel='linear')
    # model, accuracy = adaboost(df, x_train, x_test, y_train, y_test, estimator, 100, 1.1)
    # print("adaboost accuracy: " + str(accuracy))
    # estimator = SVC(kernel='rbf', probability=True)
    # model, accuracy = adaboost(df, x_train, x_test, y_train, y_test, estimator, 100, 1.1)
    # print("adaboost accuracy: " + str(accuracy))
    # estimator = keras.Sequential([
    #     keras.layers.Dense(64, activation='relu', input_shape=(x_test.shape[1],)),
    #     keras.layers.Dense(64, activation='relu'),
    #     keras.layers.Dense(1, activation='sigmoid')
    # ])
    # estimator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model, accuracy = adaboost(df, x_train, x_test, y_train, y_test, estimator, 100, 1.1)
    # print("adaboost accuracy: " + str(accuracy))
    # estimator = GaussianNB()
    # model, accuracy = adaboost(df, x_train, x_test, y_train, y_test, estimator, 100, 1.1)
    # print("adaboost accuracy: " + str(accuracy))
