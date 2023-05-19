from datetime import timedelta


def calculate_wins_general(row, player_id, groups, num_of_days):
    wins, matches_played = calculate_wins(row, player_id, groups, num_of_days)
    return wins


def calculate_win_percent(row, player_id, df, num_of_days):
    # calculate the win percentage for the player for this court
    wins, matches_played = calculate_wins(row, player_id, df, num_of_days)
    if matches_played > 0:
        win_percent = (wins / matches_played) * 100
    else:
        win_percent = 0
    return win_percent


def calculate_wins(row, player_id, df, num_of_days):
    date = row['match_date']
    player_1_id = row[player_id]

    # filter the DataFrame to only include matches from the last 2 years for this player and this court
    two_years_ago = date - timedelta(days=num_of_days)
    last_two_years = df[(df['match_date'] >= two_years_ago) & (df['match_date'] < date) & (
            (df['player_1_id'] == player_1_id) | (df['player_2_id'] == player_1_id))]

    # calculate the number of matches the player has won in the last 2 years for this court
    wins = last_two_years[player_id].value_counts().get(player_1_id, 0)

    # calculate the total number of matches the player has played in the last 2 years for this court
    matches_played = last_two_years['player_1_id'].value_counts().get(player_1_id, 0) + last_two_years[
        'player_2_id'].value_counts().get(player_1_id, 0)

    return wins, matches_played
