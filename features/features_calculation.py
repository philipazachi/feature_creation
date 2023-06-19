from features.days_since_last_match import calculate_days_since_last_match
from features.player_intensity import calculate_player_intensity
from features.player_retired import calculate_player_retired
from features.tourney_stats import calculate_minutes_in_tourney, calculate_grade_in_tourney
from features.wins_between_players import count_wins_between_players, games_between_players
from features.wins_in_court import calculate_win_percent_by_court, calculate_wins_by_court
from features.wins_momentum import calculate_win_percent, calculate_wins_general


def feature_players_retired(games_df, time_range):
    games_df["has_player_1_retired_in_last_30_days"] = games_df.apply(calculate_player_retired, axis=1,
                                                                      args=(
                                                                          'player_1_id', games_df, time_range)).fillna(
        0)
    print("feature_players_retired: finished has_player_1_retired_in_last_30_days")
    games_df["has_player_2_retired_in_last_30_days"] = games_df.apply(calculate_player_retired, axis=1,
                                                                      args=(
                                                                          'player_2_id', games_df, time_range)).fillna(
        0)
    print("feature_players_retired: finished has_player_2_retired_in_last_30_days")
    return games_df['has_player_1_retired_in_last_30_days'] - games_df['has_player_2_retired_in_last_30_days']


def feature_intensity(games_df, time_range):
    games_df["player_1_intensity"] = games_df.apply(calculate_player_intensity, axis=1,
                                                    args=('player_1_id', games_df, time_range)).fillna(0)
    print("feature_intensity: finished player_1_intensity")
    games_df["player_2_intensity"] = games_df.apply(calculate_player_intensity, axis=1,
                                                    args=('player_2_id', games_df, time_range)).fillna(0)
    print("feature_intensity: finished player_2_intensity")
    return games_df['player_1_intensity'] - games_df['player_2_intensity']


def feature_diff_days_since_last_match(games_df):
    games_df["player_1_last_match"] = games_df.apply(calculate_days_since_last_match, axis=1,
                                                     args=('player_1_id', games_df)).fillna(1000)
    print("feature_diff_days_since_last_match: finished player_1_last_match")
    games_df["player_2_last_match"] = games_df.apply(calculate_days_since_last_match, axis=1,
                                                     args=('player_2_id', games_df)).fillna(1000)
    print("feature_diff_days_since_last_match: finished player_2_last_match")
    return games_df['player_1_last_match'] - games_df['player_2_last_match']


def feature_diff_wins_between_players(games_df, time_range):
    return games_df.apply(count_wins_between_players, axis=1, args=(games_df, time_range))


def feature_games_between_players(games_df, time_range):
    return games_df.apply(games_between_players, axis=1, args=(games_df, time_range))


def feature_diff_wins_percent(games_df, time_range):
    games_df['player_1_percent'] = games_df.apply(calculate_win_percent, axis=1,
                                                  args=('player_1_id', games_df, time_range))
    print("feature_diff_wins_percent: finished player_1_percent")
    games_df['player_2_percent'] = games_df.apply(calculate_win_percent, axis=1,
                                                  args=('player_2_id', games_df, time_range))
    print("feature_diff_wins_percent: finished player_2_percent")
    return games_df['player_1_percent'] - games_df['player_2_percent']


def feature_diff_wins(games_df, time_range):
    games_df['player_1_wins'] = games_df.apply(calculate_wins_general, axis=1,
                                               args=('player_1_id', games_df, time_range))
    print("feature_diff_wins: finished player_1_wins")
    games_df['player_2_wins'] = games_df.apply(calculate_wins_general, axis=1,
                                               args=('player_2_id', games_df, time_range))
    print("feature_diff_wins: finished player_2_wins")
    return games_df['player_1_wins'] - games_df['player_2_wins']


def feature_diff_wins_by_court(games_df, time_range):
    groups = games_df.groupby('surface')  # group the DataFrame by type of court
    games_df['player_1_wins_by_court'] = games_df.apply(calculate_wins_by_court, axis=1,
                                                        args=('player_1_id', groups, time_range))
    print("feature_diff_wins_by_court: finished player_1_wins_by_court")
    games_df['player_2_wins_by_court'] = games_df.apply(calculate_wins_by_court, axis=1,
                                                        args=('player_2_id', groups, time_range))
    print("feature_diff_wins_by_court: finished player_2_wins_by_court")
    return games_df['player_1_wins_by_court'] - games_df['player_2_wins_by_court']


def feature_diff_percent_by_court(games_df, time_range):
    groups = games_df.groupby('surface')  # group the DataFrame by type of court
    games_df['player_1_percent_by_court'] = games_df.apply(calculate_win_percent_by_court, axis=1,
                                                           args=('player_1_id', groups, time_range))
    print("feature_diff_percent_by_court: finished player_1_percent_by_court")
    games_df['player_2_percent_by_court'] = games_df.apply(calculate_win_percent_by_court, axis=1,
                                                           args=('player_2_id', groups, time_range))
    print("feature_diff_percent_by_court: finished player_2_percent_by_court")
    return games_df['player_1_percent_by_court'] - games_df['player_2_percent_by_court']


def feature_minutes_in_tourney(games_df):
    games_df["player_1_minutes_in_tourney"] = games_df.apply(calculate_minutes_in_tourney, axis=1,
                                                             args=('player_1_id', games_df))
    games_df["player_2_minutes_in_tourney"] = games_df.apply(calculate_minutes_in_tourney, axis=1,
                                                             args=('player_2_id', games_df))
    return games_df['player_1_minutes_in_tourney'] - games_df['player_2_minutes_in_tourney']


def feature_grade_in_tourney(games_df):
    games_df["player_1_grade_in_tourney"] = games_df.apply(calculate_grade_in_tourney, axis=1,
                                                             args=('player_1_id', games_df))
    games_df["player_2_grade_in_tourney"] = games_df.apply(calculate_grade_in_tourney, axis=1,
                                                             args=('player_2_id', games_df))
    return games_df['player_1_grade_in_tourney'] - games_df['player_2_grade_in_tourney']
