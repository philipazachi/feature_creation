from datetime import timedelta


def calculate_player_intensity(row, player_id, df, time_range):
    player = row[player_id]
    match_date = row['match_date']
    start_date = match_date - timedelta(days=time_range)
    previous_matches = df[(df['match_date'] >= start_date) & (df['match_date'] < match_date)]
    matches_played = previous_matches[
        (previous_matches['player_1_id'] == player) |(previous_matches['player_2_id'] == player)]
    return len(matches_played)/time_range

