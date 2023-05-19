from features.days_since_last_match import calculate_days_since_last_match
from features.wins_between_players import count_wins_between_players
from features.wins_in_court import calculate_win_percent_by_court, calculate_wins_by_court
from features.wins_momentum import calculate_win_percent, calculate_wins_general


def feature_diff_days_since_last_match(games_df):
    games_df["player_1_last_match"] = games_df.apply(calculate_days_since_last_match, axis=1,
                                                  args=('player_1_id', games_df)).fillna(1000)
    games_df["player_2_last_match"] = games_df.apply(calculate_days_since_last_match, axis=1,
                                                  args=('player_2_id', games_df)).fillna(1000)
    return games_df['player_1_last_match'] - games_df['player_2_last_match']


def feature_diff_wins_between_players(games_df, time_range):
    return games_df.apply(count_wins_between_players, axis=1, args=(games_df, time_range))


def feature_diff_wins_percent(games_df, time_range):
    games_df['player_1_percent'] = games_df.apply(calculate_win_percent, axis=1,
                                                  args=('player_1_id', games_df, time_range))
    games_df['player_2_percent'] = games_df.apply(calculate_win_percent, axis=1,
                                                  args=('player_2_id', games_df, time_range))
    return games_df['player_1_percent'] - games_df['player_2_percent']


def feature_diff_wins(games_df, time_range):
    games_df['player_1_wins'] = games_df.apply(calculate_wins_general, axis=1,
                                               args=('player_1_id', games_df, time_range))
    games_df['player_2_wins'] = games_df.apply(calculate_wins_general, axis=1,
                                               args=('player_2_id', games_df, time_range))
    return games_df['player_1_wins'] - games_df['player_2_wins']


def feature_diff_wins_by_court(games_df, time_range):
    groups = games_df.groupby('surface')  # group the DataFrame by type of court
    games_df['player_1_wins_by_court'] = games_df.apply(calculate_wins_by_court, axis=1,
                                                        args=('player_1_id', groups, time_range))
    games_df['player_2_wins_by_court'] = games_df.apply(calculate_wins_by_court, axis=1,
                                                        args=('player_2_id', groups, time_range))
    return games_df['player_1_wins_by_court'] - games_df['player_2_wins_by_court']


def feature_diff_percent_by_court(games_df, time_range):
    groups = games_df.groupby('surface')  # group the DataFrame by type of court
    games_df['player_1_percent_by_court'] = games_df.apply(calculate_win_percent_by_court, axis=1,
                                                           args=('player_1_id', groups, time_range))
    games_df['player_2_percent_by_court'] = games_df.apply(calculate_win_percent_by_court, axis=1,
                                                           args=('player_2_id', groups, time_range))
    return games_df['player_1_percent_by_court'] - games_df['player_2_percent_by_court']
