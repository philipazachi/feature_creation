import pandas as pd


def prepare_df(init_df):
    init_df = fill_nulls(init_df)
    init_df = convert_column_names(init_df)
    init_df = add_match_date_column(init_df)
    return init_df


def fill_nulls(init_df):
    init_df['loser_rank'].fillna(1000, inplace=True)
    init_df['winner_rank'].fillna(1000, inplace=True)
    init_df['loser_rank_points'].fillna(0, inplace=True)
    init_df['winner_rank_points'].fillna(0, inplace=True)
    init_df['surface'].fillna("Hard", inplace=True)
    mean_value = init_df['loser_ht'].mean()
    init_df['loser_ht'].fillna(mean_value, inplace=True)
    mean_value = init_df['winner_ht'].mean()
    init_df['winner_ht'].fillna(mean_value, inplace=True)
    mean_value = init_df['loser_age'].mean()
    init_df['loser_age'].fillna(mean_value, inplace=True)
    mean_value = init_df['winner_age'].mean()
    init_df['winner_age'].fillna(mean_value, inplace=True)
    return init_df


def convert_column_names(init_df):
    df1 = init_df.sample(frac=0.5, random_state=1)  # First half
    df2 = init_df.drop(df1.index)  # Second half
    df1.rename(columns=lambda col: col.replace('winner', 'player_1') if 'winner' in col else col, inplace=True)
    df1.rename(columns=lambda col: col.replace('loser', 'player_2') if 'loser' in col else col, inplace=True)
    df1["player_1_won"] = 1
    df2.rename(columns=lambda col: col.replace('winner', 'player_2') if 'winner' in col else col, inplace=True)
    df2.rename(columns=lambda col: col.replace('loser', 'player_1') if 'loser' in col else col, inplace=True)
    df2["player_1_won"] = 0
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df['tourney_date'] = pd.to_datetime(merged_df['tourney_date'], format='%Y%m%d')
    merged_df = merged_df.sort_values('tourney_date')
    return merged_df


def add_match_date_column(df_init):
    def calculate_match_date(row):
        count = df_init[((df_init['tourney_date'] == row['tourney_date']) &
                         ((df_init['player_1_id'] == row['player_1_id']) | (
                                     df_init['player_2_id'] == row['player_1_id'])) &
                         (df_init['match_num'] < row['match_num']))].shape[0]
        match_date = row['tourney_date'] + pd.DateOffset(days=2 * count)
        return match_date
    df_init['match_date'] = df_init.apply(calculate_match_date, axis=1)
    return df_init
