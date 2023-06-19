import math
from datetime import timedelta


def calculate_minutes_in_tourney(row, player_id, games_df):
    date = row['match_date']
    player = row[player_id]
    tourney_id = row['tourney_id']
    two_months_ago = date - timedelta(days=60)
    matches_played_in_tourney = games_df[
        (games_df['match_date'] >= two_months_ago) & (games_df['match_date'] < date) & (
                (games_df['player_1_id'] == player) | (games_df['player_2_id'] == player)) & games_df[
            'tourney_id'] == tourney_id]
    return matches_played_in_tourney['minutes'].sum()


def calculate_grade_in_tourney(row, player_id, games_df):
    date = row['match_date']
    player = row[player_id]
    tourney_id = row['tourney_id']
    two_months_ago = date - timedelta(days=60)
    matches_played_in_tourney = games_df[
        (games_df['match_date'] >= two_months_ago) & (games_df['match_date'] < date) & (
                (games_df['player_1_id'] == player) | (games_df['player_2_id'] == player)) & games_df[
            'tourney_id'] == tourney_id]
    games_df["game_point_for_tourney_grade"] = math.sqrt(matches_played_in_tourney['minutes']) * math.sqrt(
        matches_played_in_tourney['player_2_rank'] if games_df['player_1_id'] == player else matches_played_in_tourney['player_1_rank'])
    return matches_played_in_tourney['game_point_for_tourney_grade'].sum()
