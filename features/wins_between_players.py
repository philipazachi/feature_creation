from datetime import datetime, timedelta


def count_wins_between_players(row, df, time_range):
    player_1_id = row['player_1_id']
    player_2_id = row['player_2_id']
    match_date = row['match_date']

    # Calculate the start date for considering previous matches (2 years before the current match)
    start_date = match_date - timedelta(days=time_range)

    # Filter the DataFrame to include previous matches where player_1_id and player_2_id played against each other
    previous_matches = df[((df['player_1_id'] == player_1_id) & (df['player_2_id'] == player_2_id)) | (
            (df['player_1_id'] == player_2_id) & (df['player_2_id'] == player_1_id))]

    # Filter previous matches to include only those within the last 2 years
    previous_matches_last_2_years = previous_matches[
        (previous_matches['match_date'] >= start_date) & (previous_matches['match_date'] < match_date)]

    # Count the number of times player_1_id won against player_2_id in previous matches within the last 2 years
    win_count = len(previous_matches_last_2_years[(previous_matches_last_2_years['player_1_id'] == player_1_id) & (
            previous_matches_last_2_years['player_1_won'] == 1)]) + len(previous_matches_last_2_years[(previous_matches_last_2_years['player_2_id'] == player_1_id) & (
                            previous_matches_last_2_years['player_1_won'] == 0)])

    return (win_count - len(previous_matches_last_2_years) / 2) * 2


def games_between_players(row, df, time_range):
    player_1_id = row['player_1_id']
    player_2_id = row['player_2_id']
    match_date = row['match_date']

    # Calculate the start date for considering previous matches (2 years before the current match)
    start_date = match_date - timedelta(days=time_range)

    # Filter the DataFrame to include previous matches where player_1_id and player_2_id played against each other
    previous_matches = df[((df['player_1_id'] == player_1_id) & (df['player_2_id'] == player_2_id)) | (
            (df['player_1_id'] == player_2_id) & (df['player_2_id'] == player_1_id))]

    # Filter previous matches to include only those within the last 2 years
    previous_matches_last_2_years = previous_matches[
        (previous_matches['match_date'] >= start_date) & (previous_matches['match_date'] < match_date)]

    return len(previous_matches_last_2_years)
