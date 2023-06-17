from datetime import timedelta


def calculate_player_retired(row, player_id, df, time_range):
    player = row[player_id]
    match_date = row['match_date']
    start_date = match_date - timedelta(days=time_range)
    previous_matches = df[(df['match_date'] >= start_date) & (df['match_date'] < match_date)]

    lost_matches = previous_matches[
        (previous_matches['player_1_id'] == player) & (previous_matches['player_1_won'] == 0) |
        (previous_matches['player_2_id'] == player) & (previous_matches['player_1_won'] == 1)]

    return 1 if len(lost_matches[lost_matches['score'].str.contains('RET', na=False)]) > 0 else 0
