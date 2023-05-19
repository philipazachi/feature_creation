from datetime import datetime


def calculate_days_since_last_match(row, player_id, games_df):
    player_1_id = row[player_id]
    match_date = row['match_date']

    # Filter the DataFrame to include previous matches where player_1_id participated
    previous_matches = games_df[(games_df['match_date'] < match_date) & ((games_df['player_1_id'] == player_1_id) | (games_df['player_2_id'] == player_1_id))]

    if len(previous_matches) == 0:
        # If no previous matches found, assign a default value of 1000 days
        return 1000

    # Sort the previous matches DataFrame by match_date in descending order
    previous_matches = previous_matches.sort_values('match_date', ascending=False)

    # Find the index of the first match where player_1_id participated
    first_match_index = previous_matches.index[previous_matches['match_date'] < match_date][0]

    # Get the match_date of the last match where player_1_id participated
    last_match_date = previous_matches.loc[first_match_index, 'match_date']

    # Calculate the number of days since the last match
    days_since_last_match = (match_date - last_match_date).days

    return days_since_last_match
