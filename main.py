import numpy as np
import pandas as pd

from df_preperation import add_match_date_column, convert_column_names
from features.features_calculation import feature_diff_percent_by_court, feature_diff_wins_by_court, \
    feature_diff_wins_percent, feature_diff_wins, feature_diff_wins_between_players, feature_diff_days_since_last_match
from models.decision_tree import train_decision_tree
from models.gradient_boosting import train_gradient_boosting
from models.random_forests import train_random_forest
from models.xgboost import train_xgboost


def create_features(games_df):
    games_df['diff_ages'] = games_df["player_1_age"] - games_df["player_2_age"]
    games_df['diff_ht'] = games_df["player_1_ht"] - games_df["player_2_ht"]
    games_df['same_hands'] = np.where(games_df['player_1_hand'] == games_df['player_2_hand'], 1, 0)
    games_df['diff_rank'] = games_df['player_1_rank'] - games_df['player_2_rank']
    games_df['diff_rank_relative'] = (games_df['player_1_rank'] - games_df['player_2_rank']) / games_df['player_1_rank']
    games_df['diff_rank_points'] = games_df['player_1_rank_points'] - games_df['player_2_rank_points']
    games_df['diff_percent_by_court'] = feature_diff_percent_by_court(games_df, 730)
    print("diff_percent_by_court")
    games_df['diff_wins_by_court'] = feature_diff_wins_by_court(games_df, 730)
    print("diff_wins_by_court")
    games_df['diff_wins_percent'] = feature_diff_wins_percent(games_df, 730)
    print("diff_wins_percent")
    games_df['diff_wins'] = feature_diff_wins(games_df, 730)
    print("diff_wins")
    games_df['diff_wins_between_players'] = feature_diff_wins_between_players(games_df, 730)
    print("diff_wins_between_players")
    games_df['diff_60_days_intensity'] = games_df['player_1_60_days_intensity'] - games_df['player_2_60_days_intensity']
    games_df['diff_retired_in_last_30_days'] = games_df['has_player_1_retired_in_last_30_days'] - games_df[
        'has_player_2_retired_in_last_30_days']
    games_df['diff_number_of_days_since_last_played'] = feature_diff_days_since_last_match(games_df)
    return games_df


def get_train_test(games_df, features):
    games_df = games_df.sort_values('match_date')
    split_index = int(0.8 * len(games_df))
    train_df = games_df[:split_index]
    test_df = games_df[split_index:]
    return train_df[features], test_df[features], train_df['predict'], test_df['predict']


if __name__ == '__main__':
    # df = pd.read_csv('data/edited_data_with_attributes.csv')
    # df = convert_column_names(df)
    # df = add_match_date_column(df)
    # df.to_csv('result/games_with_match_date.csv', index=False)
    df = pd.read_csv('result/games_features_final.csv')
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date')
    # df = create_features(df)
    # df.to_csv('result/games_features_final.csv', index=False)
    df = df[df['match_date'] >= pd.to_datetime('2010-01-01')]
    features_names = ['diff_ages', 'diff_ht', 'same_hands', 'diff_rank', 'diff_rank_relative',
                      'diff_rank_points', 'diff_percent_by_court', 'diff_wins_by_court', 'diff_wins_percent',
                      'diff_wins', 'diff_wins_between_players', 'diff_60_days_intensity',
                      'diff_retired_in_last_30_days', 'diff_number_of_days_since_last_played']
    x_train, x_test, y_train, y_test = get_train_test(df, features_names)
    model, accuracy = train_decision_tree(df,  x_train, x_test, y_train, y_test)
    print("decision tree accuracy: " + str(accuracy))
    model, accuracy = train_random_forest(df, x_train, x_test, y_train, y_test)
    print("random forests accuracy: " + str(accuracy))
    model, accuracy = train_gradient_boosting(df, x_train, x_test, y_train, y_test)
    print("gradient boosting accuracy: " + str(accuracy))
    model, accuracy = train_xgboost(df, x_train, x_test, y_train, y_test)
    print("xgboost accuracy: " + str(accuracy))
