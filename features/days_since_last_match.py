from datetime import datetime


def calculate_days_since_last_match(row, player_id, games_df):
    player_1_id = row[player_id]
    match_date = row['match_date']
    # Filter the DataFrame to include previous matches where player_1_id participated
    previous_matches = games_df[(games_df['match_date'] < match_date) & ((games_df['player_1_id'] == player_1_id) |
                                                                         (games_df['player_2_id'] == player_1_id))]
    if len(previous_matches) == 0:
        return 1000
    days_since_last_match = (match_date - previous_matches['match_date'].max()).days
    return days_since_last_match
