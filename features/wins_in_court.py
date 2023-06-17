# define a function to calculate the win percentage for each row
from datetime import timedelta


def calculate_wins_by_court(row, player_id, groups, num_of_days):
    wins, matches_played = calculate_wins(row, player_id, groups, num_of_days)
    return wins


def calculate_win_percent_by_court(row, player_id, groups, num_of_days):
    # calculate the win percentage for the player for this court
    wins, matches_played = calculate_wins(row, player_id, groups, num_of_days)
    if matches_played > 0:
        win_percent = (wins / matches_played) * 100
    else:
        win_percent = 0
    return win_percent


def calculate_wins(row, player_id, groups, num_of_days):
    surface = row['surface']
    date = row['match_date']
    player = row[player_id]

    # filter the DataFrame to only include matches from the last 2 years for this player and this court
    two_years_ago = date - timedelta(days=num_of_days)
    court_matches = groups.get_group(surface)
    matches_played = court_matches[(court_matches['match_date'] >= two_years_ago) &
                                   (court_matches['match_date'] < date) & (
                                           (court_matches['player_1_id'] == player) | ( court_matches['player_2_id'] == player))]

    # calculate the number of matches the player has won in the last 2 years for this court
    wins = len(matches_played[(matches_played['player_1_id'] == player) & (matches_played['player_1_won'] == 1) |
                   (matches_played['player_2_id'] == player) & (matches_played['player_1_won'] == 0)])

    return wins, len(matches_played)
