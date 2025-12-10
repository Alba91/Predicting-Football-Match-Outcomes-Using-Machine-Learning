import requests
import os
import pandas as pd
import numpy as np

OUTPUT_PATH= '../data/'
RAW_PATH= os.path.join(OUTPUT_PATH, 'raw/')
SEASONS= ["2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]

def download_files():

    def season_to_code(season):

        start_year, end_year = season.split("-")
        return start_year[-2:] + end_year[-2:]

    for season in SEASONS:
        code = season_to_code(season)
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        filename = f"{season}.csv"

        response = requests.get(url)
        response.raise_for_status()  # pour lever une erreur si le téléchargement a échoué
        
        filepath = os.path.join(OUTPUT_PATH, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"Fichier {filename} téléchargé avec succès.")

def load_data():

    files= [file for file in os.listdir(OUTPUT_PATH) if file.endswith('.csv')]
    data= pd.DataFrame()

    for file in files:
        file_path= os.path.join(OUTPUT_PATH, file)
        season= file.split('.')[0]
        df= pd.read_csv(file_path)
        df['Season']= season
        data= pd.concat([data, df], ignore_index=True)

    return data

def clean_data(data):

    with open(os.path.join(RAW_PATH, 'notes.txt'), 'r', encoding='utf-8') as f:
        content = f.read()
        sections= content.split('\n\n')
    
    def get_columns_from_section(section):
        col_names= []
        lines= sections[section].split('\n')
        for elt in lines:
            if '=' in elt:
                parts= elt.split('and')
                for part in parts:
                    col_name= part.split('=')[0].strip()
                    col_names.append(col_name)
        return col_names

    cols_third_section= get_columns_from_section(3)
    cols_fourth_section= get_columns_from_section(4)
    cols_to_keep= ['Season'] + cols_third_section + cols_fourth_section

    existing_cols = [c for c in cols_to_keep if c in data.columns]
    final_data= data[existing_cols]

    return final_data

def create_features(final_data):

    final_data['Date'] = pd.to_datetime(final_data['Date'], dayfirst=True, errors='coerce')
    final_data_sorted = final_data.sort_values(['Season', 'Date']).reset_index(drop=True)
    final_data_sorted['match_id'] = final_data_sorted.index

    home_long = final_data_sorted[['match_id', 'Season', 'Date', 'HomeTeam']].rename(columns={'HomeTeam':'Team'})
    away_long = final_data_sorted[['match_id', 'Season', 'Date', 'AwayTeam']].rename(columns={'AwayTeam':'Team'})
    long = pd.concat([home_long.assign(is_home=1), away_long.assign(is_home=0)], ignore_index=True)
    long = long.sort_values(['Team', 'Season', 'Date'])

    long['prev_date'] = long.groupby(['Team', 'Season'])['Date'].shift(1)
    long['days_rest'] = (long['Date'] - long['prev_date']).dt.days.fillna(0).astype(int)

    home_rest = long[long['is_home']==1][['match_id','days_rest']].rename(columns={'days_rest':'home_days_rest'})
    away_rest = long[long['is_home']==0][['match_id','days_rest']].rename(columns={'days_rest':'away_days_rest'})

    final = final_data_sorted.merge(home_rest, on='match_id', how='left').merge(away_rest, on='match_id', how='left')
    final['home_days_rest'] = final['home_days_rest'].fillna(0).astype(int)
    final['away_days_rest'] = final['away_days_rest'].fillna(0).astype(int)

    return final

def create_enriched_features(data, recent_matches=5):

    features = data.copy()
    
    # Features de base
    base_cols = [
        'home_avg_goals','home_avg_conceded','away_avg_goals','away_avg_conceded',
        'home_recent_form','away_recent_form',
        'home_h2h_wins','away_h2h_wins',
        'home_home_record','away_away_record',
        'goal_diff_avg','conceded_diff_avg','form_diff'
    ]
    for col in base_cols:
        features[col] = np.nan

    def points_for_team(hist_row, team_name):
        if hist_row['HomeTeam'] == team_name:
            if hist_row['FTR'] == 'H': return 3
            if hist_row['FTR'] == 'D': return 1
            return 0
        else:
            if hist_row['FTR'] == 'A': return 3
            if hist_row['FTR'] == 'D': return 1
            return 0

    for idx, row in features.iterrows():
        if idx == 0: continue
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        history_window = features.iloc[:idx]

        # ===== Moyennes glissantes =====
        def compute_team_stats(team_name):
            matches = history_window[(history_window['HomeTeam']==team_name)|(history_window['AwayTeam']==team_name)]
            goals_scored = matches.loc[matches['HomeTeam']==team_name,'FTHG'].sum() + matches.loc[matches['AwayTeam']==team_name,'FTAG'].sum()
            goals_conceded = matches.loc[matches['HomeTeam']==team_name,'FTAG'].sum() + matches.loc[matches['AwayTeam']==team_name,'FTHG'].sum()
            return (goals_scored/len(matches) if len(matches)>0 else 0,
                    goals_conceded/len(matches) if len(matches)>0 else 0,
                    matches)
        
        home_avg_goals, home_avg_conceded, home_matches = compute_team_stats(home_team)
        away_avg_goals, away_avg_conceded, away_matches = compute_team_stats(away_team)

        features.at[idx,'home_avg_goals'] = home_avg_goals
        features.at[idx,'home_avg_conceded'] = home_avg_conceded
        features.at[idx,'away_avg_goals'] = away_avg_goals
        features.at[idx,'away_avg_conceded'] = away_avg_conceded

        # ===== Forme récente pondérée =====
        def recent_points(matches, team_name):
            last = matches.tail(recent_matches)
            weights = np.arange(len(last),0,-1)
            points = last.apply(lambda r: points_for_team(r, team_name), axis=1)
            return np.sum(points * weights)
        
        features.at[idx,'home_recent_form'] = recent_points(home_matches, home_team)
        features.at[idx,'away_recent_form'] = recent_points(away_matches, away_team)

        # ===== H2H complet =====
        h2h = history_window[
            ((history_window['HomeTeam']==home_team) & (history_window['AwayTeam']==away_team)) |
            ((history_window['HomeTeam']==away_team) & (history_window['AwayTeam']==home_team))
        ]
        home_h2h_wins = len(h2h[((h2h['HomeTeam']==home_team)&(h2h['FTR']=='H')) | ((h2h['AwayTeam']==home_team)&(h2h['FTR']=='A'))])
        away_h2h_wins = len(h2h[((h2h['HomeTeam']==away_team)&(h2h['FTR']=='H')) | ((h2h['AwayTeam']==away_team)&(h2h['FTR']=='A'))])
        features.at[idx,'home_h2h_wins'] = home_h2h_wins
        features.at[idx,'away_h2h_wins'] = away_h2h_wins

        # ===== Record domicile/extérieur =====
        home_at_home = history_window[history_window['HomeTeam']==home_team]
        away_at_away = history_window[history_window['AwayTeam']==away_team]
        features.at[idx,'home_home_record'] = len(home_at_home[home_at_home['FTR']=='H'])/len(home_at_home) if len(home_at_home)>0 else 0
        features.at[idx,'away_away_record'] = len(away_at_away[away_at_away['FTR']=='A'])/len(away_at_away) if len(away_at_away)>0 else 0

        # ===== Différences synthétiques =====
        features.at[idx,'goal_diff_avg'] = home_avg_goals - away_avg_goals
        features.at[idx,'conceded_diff_avg'] = home_avg_conceded - away_avg_conceded
        features.at[idx,'form_diff'] = features.at[idx,'home_recent_form'] - features.at[idx,'away_recent_form']

    return features

def del_cols(df):

    cols_to_delete= ['HomeTeam', 'AwayTeam', 'Time', 'match_id', 'Div', 'Date', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    df= df.drop(columns= cols_to_delete)

    return df

def generate_data():

    download_files()
    raw_data= load_data()
    clean_data_df= clean_data(raw_data)
    return clean_data_df

data= generate_data()
