import numpy as np
import pandas as pd

from df_preperation import add_match_date_column, convert_column_names
from features.features_calculation import feature_diff_percent_by_court, feature_diff_wins_by_court, \
    feature_diff_wins_percent, feature_diff_wins
from models.decision_tree import train_decision_tree


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
    games_df['diff_60_days_intensity'] = games_df['player_1_60_days_intensity'] - games_df['player_2_60_days_intensity']
    games_df['diff_retired_in_last_30_days'] = games_df['has_player_1_retired_in_last_30_days'] - games_df[
        'has_player_2_retired_in_last_30_days']
    games_df['diff_number_of_days_since_last_played'] = games_df['player_1_number_of_days_since_last_played'] - \
                                                        games_df[
                                                            'player_2_number_of_days_since_last_played']
    return games_df


if __name__ == '__main__':
    # df = pd.read_csv('data/edited_data_with_attributes.csv')
    # df = convert_column_names(df)
    # df = add_match_date_column(df)
    # df.to_csv('result/games_with_match_date.csv', index=False)
    # df = create_features(df)
    df = pd.read_csv('result/games_features_final.csv')
    df['match_date'] = pd.to_datetime(df['match_date'])
    df.to_csv('result/games_features_final.csv', index=False)
    features_names = ['diff_ages', 'diff_ht', 'same_hands', 'diff_rank', 'diff_rank_relative',
                      'diff_rank_points', 'diff_percent_by_court', 'diff_wins_by_court', 'diff_wins_percent',
                      'diff_wins', 'diff_60_days_intensity', 'diff_retired_in_last_30_days',
                      'diff_number_of_days_since_last_played']
    model, accuracy = train_decision_tree(df, features_names)
    print("accuracy: " + str(accuracy))
